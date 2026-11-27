; ModuleID = '/shared/amdgpu/home/zhiming_ding_qle/yanguahe/code/wk_perf_test/aiter/aiter/jit/build/module_moe_mxfp4_gemm/blob/instances/mxfp4_moe_g1_a4w4_NE385_H7168_E512_BM128.cu'
source_filename = "/shared/amdgpu/home/zhiming_ding_qle/yanguahe/code/wk_perf_test/aiter/aiter/jit/build/module_moe_mxfp4_gemm/blob/instances/mxfp4_moe_g1_a4w4_NE385_H7168_E512_BM128.cu"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

%union.LDSPool = type { [32768 x float] }

$_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16 = comdat any

$_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds = comdat any

@_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds = linkonce_odr hidden addrspace(3) global %union.LDSPool undef, comdat, align 16
@__hip_cuid_809f07884f64694b = addrspace(1) global i8 0
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @__hip_cuid_809f07884f64694b to ptr)], section "llvm.metadata"

; Function Attrs: convergent mustprogress norecurse nounwind
define protected amdgpu_kernel void @_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16(ptr addrspace(1) noalias noundef %0, ptr addrspace(1) noalias noundef %1, ptr addrspace(1) noalias noundef %2, ptr addrspace(1) noalias noundef %3, ptr addrspace(1) noalias nocapture noundef readonly %4, ptr addrspace(1) noalias nocapture noundef readonly %5, ptr addrspace(1) noalias noundef %6, i32 noundef %7, ptr addrspace(1) noalias noundef %8, ptr addrspace(1) noalias noundef %9, ptr addrspace(1) noalias nocapture noundef readnone %10) local_unnamed_addr #0 comdat {
  %12 = ptrtoint ptr addrspace(1) %0 to i64
  %13 = inttoptr i64 %12 to ptr
  %14 = ptrtoint ptr addrspace(1) %2 to i64
  %15 = inttoptr i64 %14 to ptr
  %16 = ptrtoint ptr addrspace(1) %3 to i64
  %17 = inttoptr i64 %16 to ptr
  %18 = ptrtoint ptr addrspace(1) %9 to i64
  %19 = inttoptr i64 %18 to ptr
  %20 = tail call noundef i32 @llvm.amdgcn.workgroup.id.x()
  %21 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %22 = icmp samesign ult i32 %21, 256
  tail call void @llvm.assume(i1 %22)
  %23 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %21)
  %24 = mul nsw i32 %7, 3584
  %25 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p0(ptr readnone %13, i16 0, i32 %24, i32 131072)
  %26 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p0(ptr readnone %15, i16 0, i32 1412956160, i32 131072)
  %27 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p0(ptr readnone %17, i16 0, i32 88309760, i32 131072)
  %28 = load i32, ptr addrspace(1) %5, align 4, !tbaa !7
  %29 = sdiv i32 %28, 128
  %30 = shl nsw i32 %29, 2
  %31 = icmp slt i32 %20, %30
  br i1 %31, label %32, label %5205

32:                                               ; preds = %11
  %33 = ptrtoint ptr addrspace(1) %1 to i64
  %34 = inttoptr i64 %33 to ptr
  %35 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p0(ptr readnone %34, i16 0, i32 146800640, i32 131072)
  %36 = and i32 %21, 63
  %37 = lshr i32 %23, 6
  %38 = ptrtoint ptr addrspace(1) %8 to i64
  %39 = inttoptr i64 %38 to ptr
  %40 = ptrtoint ptr addrspace(1) %6 to i64
  %41 = inttoptr i64 %40 to ptr
  %42 = sdiv i32 %20, 4
  %43 = mul i32 %42, 4
  %44 = sub i32 %20, %43
  %45 = sext i32 %42 to i64
  %46 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %45
  %47 = load i32, ptr addrspace(1) %46, align 4, !tbaa !7
  %48 = shl nsw i32 %42, 7
  %49 = icmp samesign ult i32 %47, 385
  tail call void @llvm.assume(i1 %49)
  %50 = lshr i32 %36, 3
  %51 = shl nuw nsw i32 %37, 5
  %52 = or disjoint i32 %50, %48
  %53 = add i32 %51, %52
  %54 = sext i32 %53 to i64
  %55 = getelementptr inbounds i32, ptr %41, i64 %54
  %56 = load i32, ptr %55, align 4, !tbaa !7
  %57 = or disjoint i32 %53, 8
  %58 = sext i32 %57 to i64
  %59 = getelementptr inbounds i32, ptr %41, i64 %58
  %60 = load i32, ptr %59, align 4, !tbaa !7
  %61 = or disjoint i32 %53, 16
  %62 = sext i32 %61 to i64
  %63 = getelementptr inbounds i32, ptr %41, i64 %62
  %64 = load i32, ptr %63, align 4, !tbaa !7
  %65 = or disjoint i32 %53, 24
  %66 = sext i32 %65 to i64
  %67 = getelementptr inbounds i32, ptr %41, i64 %66
  %68 = load i32, ptr %67, align 4, !tbaa !7
  %69 = shl nuw nsw i32 %47, 10
  %70 = shl nsw i32 %44, 8
  %71 = add nsw i32 %69, %70
  %72 = and i32 %23, -64
  %73 = add i32 %71, %72
  %74 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %73)
  %75 = mul nsw i32 %74, 3584
  %76 = or disjoint i32 %72, 16
  %77 = add i32 %76, %71
  %78 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %77)
  %79 = mul nsw i32 %78, 3584
  %80 = or disjoint i32 %72, 32
  %81 = add i32 %80, %71
  %82 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %81)
  %83 = mul nsw i32 %82, 3584
  %84 = or disjoint i32 %72, 48
  %85 = add i32 %84, %71
  %86 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %85)
  %87 = mul nsw i32 %86, 3584
  %88 = shl nsw i32 %44, 3
  %89 = shl nuw nsw i32 %37, 1
  %90 = add nsw i32 %89, %88
  %91 = mul nuw nsw i32 %47, 57344
  %92 = mul i32 %90, 1792
  %93 = add i32 %91, %92
  %94 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %93)
  %95 = shl nsw i32 %94, 2
  %96 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %95)
  %97 = add nsw i32 %96, 4096
  %98 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %93)
  %99 = shl i32 %98, 2
  %100 = add i32 %99, 7168
  %101 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %100)
  %102 = add nsw i32 %101, 4096
  %103 = or disjoint i32 %72, %36
  %104 = shl nsw i32 %103, 4
  %105 = shl nsw i32 %103, 2
  %106 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 range(i32 -536870912, 16777215) %42)
  %107 = mul nsw i32 %106, 28672
  %108 = shl nsw i32 %37, 10
  %109 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %108
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %109, i32 noundef 16, i32 noundef %104, i32 noundef %107, i32 noundef 0, i32 noundef 0) #11
  %110 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %107)
  %111 = add i32 %110, 4096
  %112 = shl nsw i32 %37, 8
  %113 = add nuw nsw i32 %112, 4096
  %114 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %113
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %114, i32 noundef 4, i32 noundef %105, i32 noundef %111, i32 noundef 0, i32 noundef 0) #11
  %115 = add i32 %110, 5120
  %116 = add nuw nsw i32 %112, 5120
  %117 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %116
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %117, i32 noundef 4, i32 noundef %105, i32 noundef %115, i32 noundef 0, i32 noundef 0) #11
  %118 = add i32 %110, 6144
  %119 = add nuw nsw i32 %112, 6144
  %120 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %119
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %120, i32 noundef 4, i32 noundef %105, i32 noundef %118, i32 noundef 0, i32 noundef 0) #11
  %121 = add i32 %107, 7168
  %122 = add nuw nsw i32 %108, 7168
  %123 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %122
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %123, i32 noundef 16, i32 noundef %104, i32 noundef %121, i32 noundef 0, i32 noundef 0) #11
  %124 = add i32 %110, 11264
  %125 = add nuw nsw i32 %112, 11264
  %126 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %125
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %126, i32 noundef 4, i32 noundef %105, i32 noundef %124, i32 noundef 0, i32 noundef 0) #11
  %127 = add i32 %110, 12288
  %128 = add nuw nsw i32 %112, 12288
  %129 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %128
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %129, i32 noundef 4, i32 noundef %105, i32 noundef %127, i32 noundef 0, i32 noundef 0) #11
  %130 = add i32 %110, 13312
  %131 = add nuw nsw i32 %112, 13312
  %132 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %131
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %132, i32 noundef 4, i32 noundef %105, i32 noundef %130, i32 noundef 0, i32 noundef 0) #11
  %133 = add i32 %107, 14336
  %134 = add nuw nsw i32 %108, 14336
  %135 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %134
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %135, i32 noundef 16, i32 noundef %104, i32 noundef %133, i32 noundef 0, i32 noundef 0) #11
  %136 = add i32 %110, 18432
  %137 = add nuw nsw i32 %112, 18432
  %138 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %137
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %138, i32 noundef 4, i32 noundef %105, i32 noundef %136, i32 noundef 0, i32 noundef 0) #11
  %139 = add i32 %110, 19456
  %140 = add nuw nsw i32 %112, 19456
  %141 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %140
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %141, i32 noundef 4, i32 noundef %105, i32 noundef %139, i32 noundef 0, i32 noundef 0) #11
  %142 = add i32 %110, 20480
  %143 = add nuw nsw i32 %112, 20480
  %144 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %143
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %144, i32 noundef 4, i32 noundef %105, i32 noundef %142, i32 noundef 0, i32 noundef 0) #11
  %145 = add i32 %107, 21504
  %146 = add nuw nsw i32 %108, 21504
  %147 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %146
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %147, i32 noundef 16, i32 noundef %104, i32 noundef %145, i32 noundef 0, i32 noundef 0) #11
  %148 = add i32 %110, 25600
  %149 = add nuw nsw i32 %112, 25600
  %150 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %149
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %150, i32 noundef 4, i32 noundef %105, i32 noundef %148, i32 noundef 0, i32 noundef 0) #11
  %151 = add i32 %110, 26624
  %152 = add nuw nsw i32 %112, 26624
  %153 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %152
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %153, i32 noundef 4, i32 noundef %105, i32 noundef %151, i32 noundef 0, i32 noundef 0) #11
  %154 = add i32 %110, 27648
  %155 = add nuw nsw i32 %112, 27648
  %156 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %155
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %156, i32 noundef 4, i32 noundef %105, i32 noundef %154, i32 noundef 0, i32 noundef 0) #11
  %157 = and i32 %21, 48
  %158 = shl nuw nsw i32 %21, 4
  %159 = and i32 %158, 112
  %160 = xor i32 %159, %157
  %161 = mul nsw i32 %56, 3584
  %162 = or disjoint i32 %161, %160
  %163 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 0, i32 %51
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef %163, i32 noundef 16, i32 noundef %162, i32 noundef 0, i32 noundef 0, i32 noundef 0) #11
  %164 = or disjoint i32 %51, 8
  %165 = xor i32 %160, 64
  %166 = mul nsw i32 %60, 3584
  %167 = or disjoint i32 %166, %165
  %168 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 0, i32 %164
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %168, i32 noundef 16, i32 noundef %167, i32 noundef 0, i32 noundef 0, i32 noundef 0) #11
  %169 = or disjoint i32 %51, 16
  %170 = mul nsw i32 %64, 3584
  %171 = or disjoint i32 %170, %160
  %172 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 0, i32 %169
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %172, i32 noundef 16, i32 noundef %171, i32 noundef 0, i32 noundef 0, i32 noundef 0) #11
  %173 = or disjoint i32 %51, 24
  %174 = mul nsw i32 %68, 3584
  %175 = or disjoint i32 %174, %165
  %176 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 0, i32 %173
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %176, i32 noundef 16, i32 noundef %175, i32 noundef 0, i32 noundef 0, i32 noundef 0) #11
  %177 = lshr i32 %36, 4
  %178 = shl nuw nsw i32 %177, 8
  %179 = and i32 %21, 15
  %180 = shl nuw nsw i32 %179, 4
  %181 = or disjoint i32 %178, %180
  %182 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %181, i32 %75, i32 0)
  %183 = or disjoint i32 %181, 1024
  %184 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %183, i32 %75, i32 0)
  %185 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %181, i32 %79, i32 0)
  %186 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %183, i32 %79, i32 0)
  %187 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %181, i32 %83, i32 0)
  %188 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %183, i32 %83, i32 0)
  %189 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %181, i32 %87, i32 0)
  %190 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %183, i32 %87, i32 0)
  %191 = shl nuw nsw i32 %177, 6
  %192 = shl nuw nsw i32 %179, 2
  %193 = or disjoint i32 %191, %192
  %194 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %193, i32 %95, i32 0)
  %195 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %193, i32 %100, i32 0)
  %196 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 1, i32 %51
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %196, i32 noundef 16, i32 noundef %162, i32 noundef 128, i32 noundef 0, i32 noundef 0) #11
  %197 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 1, i32 %164
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %197, i32 noundef 16, i32 noundef %167, i32 noundef 128, i32 noundef 0, i32 noundef 0) #11
  %198 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 1, i32 %169
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %198, i32 noundef 16, i32 noundef %171, i32 noundef 128, i32 noundef 0, i32 noundef 0) #11
  %199 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 1, i32 %173
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %199, i32 noundef 16, i32 noundef %175, i32 noundef 128, i32 noundef 0, i32 noundef 0) #11
  %200 = or disjoint i32 %181, 2048
  %201 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %200, i32 %75, i32 0)
  %202 = or disjoint i32 %181, 3072
  %203 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %202, i32 %75, i32 0)
  %204 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %200, i32 %79, i32 0)
  %205 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %202, i32 %79, i32 0)
  %206 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %200, i32 %83, i32 0)
  %207 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %202, i32 %83, i32 0)
  %208 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %200, i32 %87, i32 0)
  %209 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %202, i32 %87, i32 0)
  %210 = or disjoint i32 %193, 256
  %211 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %210, i32 %95, i32 0)
  %212 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %210, i32 %100, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %213 = shl nuw nsw i32 %21, 3
  %214 = and i32 %213, 112
  %215 = xor i32 %214, %157
  %216 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 0, i32 %179, i32 %215
  %217 = load <4 x i32>, ptr addrspace(3) %216, align 16, !tbaa !11
  %218 = or disjoint i32 %179, 16
  %219 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 0, i32 %218, i32 %215
  %220 = load <4 x i32>, ptr addrspace(3) %219, align 16, !tbaa !11
  %221 = or disjoint i32 %179, 32
  %222 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 0, i32 %221, i32 %215
  %223 = load <4 x i32>, ptr addrspace(3) %222, align 16, !tbaa !11
  %224 = or disjoint i32 %179, 48
  %225 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 0, i32 %224, i32 %215
  %226 = load <4 x i32>, ptr addrspace(3) %225, align 16, !tbaa !11
  %227 = or disjoint i32 %179, 64
  %228 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 0, i32 %227, i32 %215
  %229 = load <4 x i32>, ptr addrspace(3) %228, align 16, !tbaa !11
  %230 = or disjoint i32 %179, 80
  %231 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 0, i32 %230, i32 %215
  %232 = load <4 x i32>, ptr addrspace(3) %231, align 16, !tbaa !11
  %233 = or disjoint i32 %179, 96
  %234 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 0, i32 %233, i32 %215
  %235 = load <4 x i32>, ptr addrspace(3) %234, align 16, !tbaa !11
  %236 = or disjoint i32 %179, 112
  %237 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 0, i32 %236, i32 %215
  %238 = load <4 x i32>, ptr addrspace(3) %237, align 16, !tbaa !11
  %239 = or disjoint i32 %157, 64
  %240 = xor i32 %239, %214
  %241 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 0, i32 %179, i32 %240
  %242 = load <4 x i32>, ptr addrspace(3) %241, align 16, !tbaa !11
  %243 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 0, i32 %218, i32 %240
  %244 = load <4 x i32>, ptr addrspace(3) %243, align 16, !tbaa !11
  %245 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 0, i32 %221, i32 %240
  %246 = load <4 x i32>, ptr addrspace(3) %245, align 16, !tbaa !11
  %247 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 0, i32 %224, i32 %240
  %248 = load <4 x i32>, ptr addrspace(3) %247, align 16, !tbaa !11
  %249 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 0, i32 %227, i32 %240
  %250 = load <4 x i32>, ptr addrspace(3) %249, align 16, !tbaa !11
  %251 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 0, i32 %230, i32 %240
  %252 = load <4 x i32>, ptr addrspace(3) %251, align 16, !tbaa !11
  %253 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 0, i32 %233, i32 %240
  %254 = load <4 x i32>, ptr addrspace(3) %253, align 16, !tbaa !11
  %255 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 0, i32 %236, i32 %240
  %256 = load <4 x i32>, ptr addrspace(3) %255, align 16, !tbaa !11
  %257 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %193
  %258 = load i32, ptr addrspace(3) %257, align 4, !tbaa !7
  %259 = or disjoint i32 %193, 7168
  %260 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %259
  %261 = load i32, ptr addrspace(3) %260, align 4, !tbaa !7
  %262 = or disjoint i32 %193, 14336
  %263 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %262
  %264 = load i32, ptr addrspace(3) %263, align 4, !tbaa !7
  %265 = or disjoint i32 %193, 21504
  %266 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %265
  %267 = load i32, ptr addrspace(3) %266, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef %163, i32 noundef 16, i32 noundef %162, i32 noundef 256, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %168, i32 noundef 16, i32 noundef %167, i32 noundef 256, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %172, i32 noundef 16, i32 noundef %171, i32 noundef 256, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %176, i32 noundef 16, i32 noundef %175, i32 noundef 256, i32 noundef 0, i32 noundef 0) #11
  %268 = tail call contract <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %217, <4 x i32> %182, i32 %258, i32 %194) #11, !srcloc !12
  %269 = tail call contract <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %220, <4 x i32> %182, i32 %258, i32 %194) #11, !srcloc !13
  %270 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %242, <4 x i32> noundef %184, <4 x float> noundef %268, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %258, i32 noundef 2, i32 noundef %194) #11
  %271 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %244, <4 x i32> noundef %184, <4 x float> noundef %269, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %258, i32 noundef 2, i32 noundef %194) #11
  %272 = tail call contract <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %223, <4 x i32> %182, i32 %261, i32 %194) #11, !srcloc !12
  %273 = tail call contract <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %226, <4 x i32> %182, i32 %261, i32 %194) #11, !srcloc !13
  %274 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %246, <4 x i32> noundef %184, <4 x float> noundef %272, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %261, i32 noundef 2, i32 noundef %194) #11
  %275 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %248, <4 x i32> noundef %184, <4 x float> noundef %273, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %261, i32 noundef 2, i32 noundef %194) #11
  %276 = tail call contract <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %229, <4 x i32> %182, i32 %264, i32 %194) #11, !srcloc !12
  %277 = tail call contract <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %232, <4 x i32> %182, i32 %264, i32 %194) #11, !srcloc !13
  %278 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %250, <4 x i32> noundef %184, <4 x float> noundef %276, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %264, i32 noundef 2, i32 noundef %194) #11
  %279 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %252, <4 x i32> noundef %184, <4 x float> noundef %277, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %264, i32 noundef 2, i32 noundef %194) #11
  %280 = tail call contract <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %235, <4 x i32> %182, i32 %267, i32 %194) #11, !srcloc !12
  %281 = tail call contract <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %238, <4 x i32> %182, i32 %267, i32 %194) #11, !srcloc !13
  %282 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %254, <4 x i32> noundef %184, <4 x float> noundef %280, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %267, i32 noundef 2, i32 noundef %194) #11
  %283 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %256, <4 x i32> noundef %184, <4 x float> noundef %281, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %267, i32 noundef 2, i32 noundef %194) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %284 = or disjoint i32 %181, 4096
  %285 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %284, i32 %75, i32 0)
  %286 = or disjoint i32 %181, 5120
  %287 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %286, i32 %75, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %288 = tail call contract <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %217, <4 x i32> %185, i32 %258, i32 %194) #11, !srcloc !14
  %289 = tail call contract <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %220, <4 x i32> %185, i32 %258, i32 %194) #11, !srcloc !15
  %290 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %242, <4 x i32> noundef %186, <4 x float> noundef %288, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %258, i32 noundef 3, i32 noundef %194) #11
  %291 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %244, <4 x i32> noundef %186, <4 x float> noundef %289, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %258, i32 noundef 3, i32 noundef %194) #11
  %292 = tail call contract <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %223, <4 x i32> %185, i32 %261, i32 %194) #11, !srcloc !14
  %293 = tail call contract <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %226, <4 x i32> %185, i32 %261, i32 %194) #11, !srcloc !15
  %294 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %246, <4 x i32> noundef %186, <4 x float> noundef %292, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %261, i32 noundef 3, i32 noundef %194) #11
  %295 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %248, <4 x i32> noundef %186, <4 x float> noundef %293, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %261, i32 noundef 3, i32 noundef %194) #11
  %296 = tail call contract <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %229, <4 x i32> %185, i32 %264, i32 %194) #11, !srcloc !14
  %297 = tail call contract <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %232, <4 x i32> %185, i32 %264, i32 %194) #11, !srcloc !15
  %298 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %250, <4 x i32> noundef %186, <4 x float> noundef %296, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %264, i32 noundef 3, i32 noundef %194) #11
  %299 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %252, <4 x i32> noundef %186, <4 x float> noundef %297, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %264, i32 noundef 3, i32 noundef %194) #11
  %300 = tail call contract <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %235, <4 x i32> %185, i32 %267, i32 %194) #11, !srcloc !14
  %301 = tail call contract <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %238, <4 x i32> %185, i32 %267, i32 %194) #11, !srcloc !15
  %302 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %254, <4 x i32> noundef %186, <4 x float> noundef %300, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %267, i32 noundef 3, i32 noundef %194) #11
  %303 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %256, <4 x i32> noundef %186, <4 x float> noundef %301, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %267, i32 noundef 3, i32 noundef %194) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %304 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %284, i32 %79, i32 0)
  %305 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %286, i32 %79, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %306 = tail call contract <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %217, <4 x i32> %187, i32 %258, i32 %195) #11, !srcloc !12
  %307 = tail call contract <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %220, <4 x i32> %187, i32 %258, i32 %195) #11, !srcloc !13
  %308 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %242, <4 x i32> noundef %188, <4 x float> noundef %306, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %258, i32 noundef 2, i32 noundef %195) #11
  %309 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %244, <4 x i32> noundef %188, <4 x float> noundef %307, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %258, i32 noundef 2, i32 noundef %195) #11
  %310 = tail call contract <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %223, <4 x i32> %187, i32 %261, i32 %195) #11, !srcloc !12
  %311 = tail call contract <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %226, <4 x i32> %187, i32 %261, i32 %195) #11, !srcloc !13
  %312 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %246, <4 x i32> noundef %188, <4 x float> noundef %310, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %261, i32 noundef 2, i32 noundef %195) #11
  %313 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %248, <4 x i32> noundef %188, <4 x float> noundef %311, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %261, i32 noundef 2, i32 noundef %195) #11
  %314 = tail call contract <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %229, <4 x i32> %187, i32 %264, i32 %195) #11, !srcloc !12
  %315 = tail call contract <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %232, <4 x i32> %187, i32 %264, i32 %195) #11, !srcloc !13
  %316 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %250, <4 x i32> noundef %188, <4 x float> noundef %314, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %264, i32 noundef 2, i32 noundef %195) #11
  %317 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %252, <4 x i32> noundef %188, <4 x float> noundef %315, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %264, i32 noundef 2, i32 noundef %195) #11
  %318 = tail call contract <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %235, <4 x i32> %187, i32 %267, i32 %195) #11, !srcloc !12
  %319 = tail call contract <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %238, <4 x i32> %187, i32 %267, i32 %195) #11, !srcloc !13
  %320 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %254, <4 x i32> noundef %188, <4 x float> noundef %318, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %267, i32 noundef 2, i32 noundef %195) #11
  %321 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %256, <4 x i32> noundef %188, <4 x float> noundef %319, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %267, i32 noundef 2, i32 noundef %195) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %322 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %284, i32 %83, i32 0)
  %323 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %286, i32 %83, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %324 = tail call contract <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %217, <4 x i32> %189, i32 %258, i32 %195) #11, !srcloc !14
  %325 = tail call contract <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %220, <4 x i32> %189, i32 %258, i32 %195) #11, !srcloc !15
  %326 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %242, <4 x i32> noundef %190, <4 x float> noundef %324, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %258, i32 noundef 3, i32 noundef %195) #11
  %327 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %244, <4 x i32> noundef %190, <4 x float> noundef %325, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %258, i32 noundef 3, i32 noundef %195) #11
  %328 = tail call contract <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %223, <4 x i32> %189, i32 %261, i32 %195) #11, !srcloc !14
  %329 = tail call contract <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %226, <4 x i32> %189, i32 %261, i32 %195) #11, !srcloc !15
  %330 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %246, <4 x i32> noundef %190, <4 x float> noundef %328, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %261, i32 noundef 3, i32 noundef %195) #11
  %331 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %248, <4 x i32> noundef %190, <4 x float> noundef %329, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %261, i32 noundef 3, i32 noundef %195) #11
  %332 = tail call contract <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %229, <4 x i32> %189, i32 %264, i32 %195) #11, !srcloc !14
  %333 = tail call contract <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %232, <4 x i32> %189, i32 %264, i32 %195) #11, !srcloc !15
  %334 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %250, <4 x i32> noundef %190, <4 x float> noundef %332, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %264, i32 noundef 3, i32 noundef %195) #11
  %335 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %252, <4 x i32> noundef %190, <4 x float> noundef %333, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %264, i32 noundef 3, i32 noundef %195) #11
  %336 = tail call contract <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %235, <4 x i32> %189, i32 %267, i32 %195) #11, !srcloc !14
  %337 = tail call contract <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %238, <4 x i32> %189, i32 %267, i32 %195) #11, !srcloc !15
  %338 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %254, <4 x i32> noundef %190, <4 x float> noundef %336, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %267, i32 noundef 3, i32 noundef %195) #11
  %339 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %256, <4 x i32> noundef %190, <4 x float> noundef %337, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %267, i32 noundef 3, i32 noundef %195) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %340 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %284, i32 %87, i32 0)
  %341 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %286, i32 %87, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %342 = or disjoint i32 %193, 512
  %343 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %342, i32 %95, i32 0)
  %344 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %342, i32 %100, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %345 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 1, i32 %179, i32 %215
  %346 = load <4 x i32>, ptr addrspace(3) %345, align 16, !tbaa !11
  %347 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 1, i32 %218, i32 %215
  %348 = load <4 x i32>, ptr addrspace(3) %347, align 16, !tbaa !11
  %349 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 1, i32 %221, i32 %215
  %350 = load <4 x i32>, ptr addrspace(3) %349, align 16, !tbaa !11
  %351 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 1, i32 %224, i32 %215
  %352 = load <4 x i32>, ptr addrspace(3) %351, align 16, !tbaa !11
  %353 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 1, i32 %227, i32 %215
  %354 = load <4 x i32>, ptr addrspace(3) %353, align 16, !tbaa !11
  %355 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 1, i32 %230, i32 %215
  %356 = load <4 x i32>, ptr addrspace(3) %355, align 16, !tbaa !11
  %357 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 1, i32 %233, i32 %215
  %358 = load <4 x i32>, ptr addrspace(3) %357, align 16, !tbaa !11
  %359 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 1, i32 %236, i32 %215
  %360 = load <4 x i32>, ptr addrspace(3) %359, align 16, !tbaa !11
  %361 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 1, i32 %179, i32 %240
  %362 = load <4 x i32>, ptr addrspace(3) %361, align 16, !tbaa !11
  %363 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 1, i32 %218, i32 %240
  %364 = load <4 x i32>, ptr addrspace(3) %363, align 16, !tbaa !11
  %365 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 1, i32 %221, i32 %240
  %366 = load <4 x i32>, ptr addrspace(3) %365, align 16, !tbaa !11
  %367 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 1, i32 %224, i32 %240
  %368 = load <4 x i32>, ptr addrspace(3) %367, align 16, !tbaa !11
  %369 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 1, i32 %227, i32 %240
  %370 = load <4 x i32>, ptr addrspace(3) %369, align 16, !tbaa !11
  %371 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 1, i32 %230, i32 %240
  %372 = load <4 x i32>, ptr addrspace(3) %371, align 16, !tbaa !11
  %373 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 1, i32 %233, i32 %240
  %374 = load <4 x i32>, ptr addrspace(3) %373, align 16, !tbaa !11
  %375 = getelementptr inbounds nuw [2 x [128 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 1, i32 %236, i32 %240
  %376 = load <4 x i32>, ptr addrspace(3) %375, align 16, !tbaa !11
  %377 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %210
  %378 = load i32, ptr addrspace(3) %377, align 4, !tbaa !7
  %379 = or disjoint i32 %193, 7424
  %380 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %379
  %381 = load i32, ptr addrspace(3) %380, align 4, !tbaa !7
  %382 = or disjoint i32 %193, 14592
  %383 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %382
  %384 = load i32, ptr addrspace(3) %383, align 4, !tbaa !7
  %385 = or disjoint i32 %193, 21760
  %386 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %385
  %387 = load i32, ptr addrspace(3) %386, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %196, i32 noundef 16, i32 noundef %162, i32 noundef 384, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %197, i32 noundef 16, i32 noundef %167, i32 noundef 384, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %198, i32 noundef 16, i32 noundef %171, i32 noundef 384, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %199, i32 noundef 16, i32 noundef %175, i32 noundef 384, i32 noundef 0, i32 noundef 0) #11
  %388 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %346, <4 x i32> noundef %201, <4 x float> noundef %270, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %378, i32 noundef 0, i32 noundef %211) #11
  %389 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %348, <4 x i32> noundef %201, <4 x float> noundef %271, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %378, i32 noundef 0, i32 noundef %211) #11
  %390 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %362, <4 x i32> noundef %203, <4 x float> noundef %388, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %378, i32 noundef 2, i32 noundef %211) #11
  %391 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %364, <4 x i32> noundef %203, <4 x float> noundef %389, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %378, i32 noundef 2, i32 noundef %211) #11
  %392 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %350, <4 x i32> noundef %201, <4 x float> noundef %274, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %381, i32 noundef 0, i32 noundef %211) #11
  %393 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %352, <4 x i32> noundef %201, <4 x float> noundef %275, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %381, i32 noundef 0, i32 noundef %211) #11
  %394 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %366, <4 x i32> noundef %203, <4 x float> noundef %392, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %381, i32 noundef 2, i32 noundef %211) #11
  %395 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %368, <4 x i32> noundef %203, <4 x float> noundef %393, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %381, i32 noundef 2, i32 noundef %211) #11
  %396 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %354, <4 x i32> noundef %201, <4 x float> noundef %278, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %384, i32 noundef 0, i32 noundef %211) #11
  %397 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %356, <4 x i32> noundef %201, <4 x float> noundef %279, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %384, i32 noundef 0, i32 noundef %211) #11
  %398 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %370, <4 x i32> noundef %203, <4 x float> noundef %396, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %384, i32 noundef 2, i32 noundef %211) #11
  %399 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %372, <4 x i32> noundef %203, <4 x float> noundef %397, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %384, i32 noundef 2, i32 noundef %211) #11
  %400 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %358, <4 x i32> noundef %201, <4 x float> noundef %282, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %387, i32 noundef 0, i32 noundef %211) #11
  %401 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %360, <4 x i32> noundef %201, <4 x float> noundef %283, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %387, i32 noundef 0, i32 noundef %211) #11
  %402 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %374, <4 x i32> noundef %203, <4 x float> noundef %400, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %387, i32 noundef 2, i32 noundef %211) #11
  %403 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %376, <4 x i32> noundef %203, <4 x float> noundef %401, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %387, i32 noundef 2, i32 noundef %211) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %404 = or disjoint i32 %181, 6144
  %405 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %404, i32 %75, i32 0)
  %406 = or disjoint i32 %181, 7168
  %407 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %406, i32 %75, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %408 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %346, <4 x i32> noundef %204, <4 x float> noundef %290, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %378, i32 noundef 1, i32 noundef %211) #11
  %409 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %348, <4 x i32> noundef %204, <4 x float> noundef %291, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %378, i32 noundef 1, i32 noundef %211) #11
  %410 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %362, <4 x i32> noundef %205, <4 x float> noundef %408, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %378, i32 noundef 3, i32 noundef %211) #11
  %411 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %364, <4 x i32> noundef %205, <4 x float> noundef %409, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %378, i32 noundef 3, i32 noundef %211) #11
  %412 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %350, <4 x i32> noundef %204, <4 x float> noundef %294, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %381, i32 noundef 1, i32 noundef %211) #11
  %413 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %352, <4 x i32> noundef %204, <4 x float> noundef %295, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %381, i32 noundef 1, i32 noundef %211) #11
  %414 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %366, <4 x i32> noundef %205, <4 x float> noundef %412, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %381, i32 noundef 3, i32 noundef %211) #11
  %415 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %368, <4 x i32> noundef %205, <4 x float> noundef %413, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %381, i32 noundef 3, i32 noundef %211) #11
  %416 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %354, <4 x i32> noundef %204, <4 x float> noundef %298, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %384, i32 noundef 1, i32 noundef %211) #11
  %417 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %356, <4 x i32> noundef %204, <4 x float> noundef %299, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %384, i32 noundef 1, i32 noundef %211) #11
  %418 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %370, <4 x i32> noundef %205, <4 x float> noundef %416, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %384, i32 noundef 3, i32 noundef %211) #11
  %419 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %372, <4 x i32> noundef %205, <4 x float> noundef %417, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %384, i32 noundef 3, i32 noundef %211) #11
  %420 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %358, <4 x i32> noundef %204, <4 x float> noundef %302, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %387, i32 noundef 1, i32 noundef %211) #11
  %421 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %360, <4 x i32> noundef %204, <4 x float> noundef %303, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %387, i32 noundef 1, i32 noundef %211) #11
  %422 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %374, <4 x i32> noundef %205, <4 x float> noundef %420, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %387, i32 noundef 3, i32 noundef %211) #11
  %423 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %376, <4 x i32> noundef %205, <4 x float> noundef %421, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %387, i32 noundef 3, i32 noundef %211) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %424 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %404, i32 %79, i32 0)
  %425 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %406, i32 %79, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %426 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %346, <4 x i32> noundef %206, <4 x float> noundef %308, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %378, i32 noundef 0, i32 noundef %212) #11
  %427 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %348, <4 x i32> noundef %206, <4 x float> noundef %309, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %378, i32 noundef 0, i32 noundef %212) #11
  %428 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %362, <4 x i32> noundef %207, <4 x float> noundef %426, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %378, i32 noundef 2, i32 noundef %212) #11
  %429 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %364, <4 x i32> noundef %207, <4 x float> noundef %427, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %378, i32 noundef 2, i32 noundef %212) #11
  %430 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %350, <4 x i32> noundef %206, <4 x float> noundef %312, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %381, i32 noundef 0, i32 noundef %212) #11
  %431 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %352, <4 x i32> noundef %206, <4 x float> noundef %313, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %381, i32 noundef 0, i32 noundef %212) #11
  %432 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %366, <4 x i32> noundef %207, <4 x float> noundef %430, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %381, i32 noundef 2, i32 noundef %212) #11
  %433 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %368, <4 x i32> noundef %207, <4 x float> noundef %431, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %381, i32 noundef 2, i32 noundef %212) #11
  %434 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %354, <4 x i32> noundef %206, <4 x float> noundef %316, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %384, i32 noundef 0, i32 noundef %212) #11
  %435 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %356, <4 x i32> noundef %206, <4 x float> noundef %317, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %384, i32 noundef 0, i32 noundef %212) #11
  %436 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %370, <4 x i32> noundef %207, <4 x float> noundef %434, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %384, i32 noundef 2, i32 noundef %212) #11
  %437 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %372, <4 x i32> noundef %207, <4 x float> noundef %435, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %384, i32 noundef 2, i32 noundef %212) #11
  %438 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %358, <4 x i32> noundef %206, <4 x float> noundef %320, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %387, i32 noundef 0, i32 noundef %212) #11
  %439 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %360, <4 x i32> noundef %206, <4 x float> noundef %321, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %387, i32 noundef 0, i32 noundef %212) #11
  %440 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %374, <4 x i32> noundef %207, <4 x float> noundef %438, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %387, i32 noundef 2, i32 noundef %212) #11
  %441 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %376, <4 x i32> noundef %207, <4 x float> noundef %439, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %387, i32 noundef 2, i32 noundef %212) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %442 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %404, i32 %83, i32 0)
  %443 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %406, i32 %83, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %444 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %346, <4 x i32> noundef %208, <4 x float> noundef %326, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %378, i32 noundef 1, i32 noundef %212) #11
  %445 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %348, <4 x i32> noundef %208, <4 x float> noundef %327, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %378, i32 noundef 1, i32 noundef %212) #11
  %446 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %362, <4 x i32> noundef %209, <4 x float> noundef %444, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %378, i32 noundef 3, i32 noundef %212) #11
  %447 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %364, <4 x i32> noundef %209, <4 x float> noundef %445, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %378, i32 noundef 3, i32 noundef %212) #11
  %448 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %350, <4 x i32> noundef %208, <4 x float> noundef %330, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %381, i32 noundef 1, i32 noundef %212) #11
  %449 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %352, <4 x i32> noundef %208, <4 x float> noundef %331, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %381, i32 noundef 1, i32 noundef %212) #11
  %450 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %366, <4 x i32> noundef %209, <4 x float> noundef %448, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %381, i32 noundef 3, i32 noundef %212) #11
  %451 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %368, <4 x i32> noundef %209, <4 x float> noundef %449, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %381, i32 noundef 3, i32 noundef %212) #11
  %452 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %354, <4 x i32> noundef %208, <4 x float> noundef %334, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %384, i32 noundef 1, i32 noundef %212) #11
  %453 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %356, <4 x i32> noundef %208, <4 x float> noundef %335, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %384, i32 noundef 1, i32 noundef %212) #11
  %454 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %370, <4 x i32> noundef %209, <4 x float> noundef %452, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %384, i32 noundef 3, i32 noundef %212) #11
  %455 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %372, <4 x i32> noundef %209, <4 x float> noundef %453, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %384, i32 noundef 3, i32 noundef %212) #11
  %456 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %358, <4 x i32> noundef %208, <4 x float> noundef %338, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %387, i32 noundef 1, i32 noundef %212) #11
  %457 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %360, <4 x i32> noundef %208, <4 x float> noundef %339, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %387, i32 noundef 1, i32 noundef %212) #11
  %458 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %374, <4 x i32> noundef %209, <4 x float> noundef %456, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %387, i32 noundef 3, i32 noundef %212) #11
  %459 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %376, <4 x i32> noundef %209, <4 x float> noundef %457, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %387, i32 noundef 3, i32 noundef %212) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %460 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %404, i32 %87, i32 0)
  %461 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %406, i32 %87, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %462 = or disjoint i32 %193, 768
  %463 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %462, i32 %95, i32 0)
  %464 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %462, i32 %100, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %465 = load <4 x i32>, ptr addrspace(3) %216, align 16, !tbaa !11
  %466 = load <4 x i32>, ptr addrspace(3) %219, align 16, !tbaa !11
  %467 = load <4 x i32>, ptr addrspace(3) %222, align 16, !tbaa !11
  %468 = load <4 x i32>, ptr addrspace(3) %225, align 16, !tbaa !11
  %469 = load <4 x i32>, ptr addrspace(3) %228, align 16, !tbaa !11
  %470 = load <4 x i32>, ptr addrspace(3) %231, align 16, !tbaa !11
  %471 = load <4 x i32>, ptr addrspace(3) %234, align 16, !tbaa !11
  %472 = load <4 x i32>, ptr addrspace(3) %237, align 16, !tbaa !11
  %473 = load <4 x i32>, ptr addrspace(3) %241, align 16, !tbaa !11
  %474 = load <4 x i32>, ptr addrspace(3) %243, align 16, !tbaa !11
  %475 = load <4 x i32>, ptr addrspace(3) %245, align 16, !tbaa !11
  %476 = load <4 x i32>, ptr addrspace(3) %247, align 16, !tbaa !11
  %477 = load <4 x i32>, ptr addrspace(3) %249, align 16, !tbaa !11
  %478 = load <4 x i32>, ptr addrspace(3) %251, align 16, !tbaa !11
  %479 = load <4 x i32>, ptr addrspace(3) %253, align 16, !tbaa !11
  %480 = load <4 x i32>, ptr addrspace(3) %255, align 16, !tbaa !11
  %481 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %342
  %482 = load i32, ptr addrspace(3) %481, align 4, !tbaa !7
  %483 = or disjoint i32 %193, 7680
  %484 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %483
  %485 = load i32, ptr addrspace(3) %484, align 4, !tbaa !7
  %486 = or disjoint i32 %193, 14848
  %487 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %486
  %488 = load i32, ptr addrspace(3) %487, align 4, !tbaa !7
  %489 = or disjoint i32 %193, 22016
  %490 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %489
  %491 = load i32, ptr addrspace(3) %490, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef %163, i32 noundef 16, i32 noundef %162, i32 noundef 512, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %168, i32 noundef 16, i32 noundef %167, i32 noundef 512, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %172, i32 noundef 16, i32 noundef %171, i32 noundef 512, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %176, i32 noundef 16, i32 noundef %175, i32 noundef 512, i32 noundef 0, i32 noundef 0) #11
  %492 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %465, <4 x i32> noundef %285, <4 x float> noundef %390, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %482, i32 noundef 0, i32 noundef %343) #11
  %493 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %466, <4 x i32> noundef %285, <4 x float> noundef %391, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %482, i32 noundef 0, i32 noundef %343) #11
  %494 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %473, <4 x i32> noundef %287, <4 x float> noundef %492, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %482, i32 noundef 2, i32 noundef %343) #11
  %495 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %474, <4 x i32> noundef %287, <4 x float> noundef %493, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %482, i32 noundef 2, i32 noundef %343) #11
  %496 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %467, <4 x i32> noundef %285, <4 x float> noundef %394, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %485, i32 noundef 0, i32 noundef %343) #11
  %497 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %468, <4 x i32> noundef %285, <4 x float> noundef %395, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %485, i32 noundef 0, i32 noundef %343) #11
  %498 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %475, <4 x i32> noundef %287, <4 x float> noundef %496, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %485, i32 noundef 2, i32 noundef %343) #11
  %499 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %476, <4 x i32> noundef %287, <4 x float> noundef %497, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %485, i32 noundef 2, i32 noundef %343) #11
  %500 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %469, <4 x i32> noundef %285, <4 x float> noundef %398, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %488, i32 noundef 0, i32 noundef %343) #11
  %501 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %470, <4 x i32> noundef %285, <4 x float> noundef %399, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %488, i32 noundef 0, i32 noundef %343) #11
  %502 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %477, <4 x i32> noundef %287, <4 x float> noundef %500, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %488, i32 noundef 2, i32 noundef %343) #11
  %503 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %478, <4 x i32> noundef %287, <4 x float> noundef %501, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %488, i32 noundef 2, i32 noundef %343) #11
  %504 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %471, <4 x i32> noundef %285, <4 x float> noundef %402, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %491, i32 noundef 0, i32 noundef %343) #11
  %505 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %472, <4 x i32> noundef %285, <4 x float> noundef %403, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %491, i32 noundef 0, i32 noundef %343) #11
  %506 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %479, <4 x i32> noundef %287, <4 x float> noundef %504, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %491, i32 noundef 2, i32 noundef %343) #11
  %507 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %480, <4 x i32> noundef %287, <4 x float> noundef %505, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %491, i32 noundef 2, i32 noundef %343) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %508 = or disjoint i32 %181, 8192
  %509 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %508, i32 %75, i32 0)
  %510 = or disjoint i32 %181, 9216
  %511 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %510, i32 %75, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %512 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %465, <4 x i32> noundef %304, <4 x float> noundef %410, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %482, i32 noundef 1, i32 noundef %343) #11
  %513 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %466, <4 x i32> noundef %304, <4 x float> noundef %411, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %482, i32 noundef 1, i32 noundef %343) #11
  %514 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %473, <4 x i32> noundef %305, <4 x float> noundef %512, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %482, i32 noundef 3, i32 noundef %343) #11
  %515 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %474, <4 x i32> noundef %305, <4 x float> noundef %513, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %482, i32 noundef 3, i32 noundef %343) #11
  %516 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %467, <4 x i32> noundef %304, <4 x float> noundef %414, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %485, i32 noundef 1, i32 noundef %343) #11
  %517 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %468, <4 x i32> noundef %304, <4 x float> noundef %415, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %485, i32 noundef 1, i32 noundef %343) #11
  %518 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %475, <4 x i32> noundef %305, <4 x float> noundef %516, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %485, i32 noundef 3, i32 noundef %343) #11
  %519 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %476, <4 x i32> noundef %305, <4 x float> noundef %517, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %485, i32 noundef 3, i32 noundef %343) #11
  %520 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %469, <4 x i32> noundef %304, <4 x float> noundef %418, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %488, i32 noundef 1, i32 noundef %343) #11
  %521 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %470, <4 x i32> noundef %304, <4 x float> noundef %419, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %488, i32 noundef 1, i32 noundef %343) #11
  %522 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %477, <4 x i32> noundef %305, <4 x float> noundef %520, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %488, i32 noundef 3, i32 noundef %343) #11
  %523 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %478, <4 x i32> noundef %305, <4 x float> noundef %521, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %488, i32 noundef 3, i32 noundef %343) #11
  %524 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %471, <4 x i32> noundef %304, <4 x float> noundef %422, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %491, i32 noundef 1, i32 noundef %343) #11
  %525 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %472, <4 x i32> noundef %304, <4 x float> noundef %423, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %491, i32 noundef 1, i32 noundef %343) #11
  %526 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %479, <4 x i32> noundef %305, <4 x float> noundef %524, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %491, i32 noundef 3, i32 noundef %343) #11
  %527 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %480, <4 x i32> noundef %305, <4 x float> noundef %525, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %491, i32 noundef 3, i32 noundef %343) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %528 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %508, i32 %79, i32 0)
  %529 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %510, i32 %79, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %530 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %465, <4 x i32> noundef %322, <4 x float> noundef %428, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %482, i32 noundef 0, i32 noundef %344) #11
  %531 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %466, <4 x i32> noundef %322, <4 x float> noundef %429, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %482, i32 noundef 0, i32 noundef %344) #11
  %532 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %473, <4 x i32> noundef %323, <4 x float> noundef %530, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %482, i32 noundef 2, i32 noundef %344) #11
  %533 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %474, <4 x i32> noundef %323, <4 x float> noundef %531, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %482, i32 noundef 2, i32 noundef %344) #11
  %534 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %467, <4 x i32> noundef %322, <4 x float> noundef %432, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %485, i32 noundef 0, i32 noundef %344) #11
  %535 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %468, <4 x i32> noundef %322, <4 x float> noundef %433, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %485, i32 noundef 0, i32 noundef %344) #11
  %536 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %475, <4 x i32> noundef %323, <4 x float> noundef %534, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %485, i32 noundef 2, i32 noundef %344) #11
  %537 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %476, <4 x i32> noundef %323, <4 x float> noundef %535, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %485, i32 noundef 2, i32 noundef %344) #11
  %538 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %469, <4 x i32> noundef %322, <4 x float> noundef %436, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %488, i32 noundef 0, i32 noundef %344) #11
  %539 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %470, <4 x i32> noundef %322, <4 x float> noundef %437, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %488, i32 noundef 0, i32 noundef %344) #11
  %540 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %477, <4 x i32> noundef %323, <4 x float> noundef %538, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %488, i32 noundef 2, i32 noundef %344) #11
  %541 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %478, <4 x i32> noundef %323, <4 x float> noundef %539, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %488, i32 noundef 2, i32 noundef %344) #11
  %542 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %471, <4 x i32> noundef %322, <4 x float> noundef %440, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %491, i32 noundef 0, i32 noundef %344) #11
  %543 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %472, <4 x i32> noundef %322, <4 x float> noundef %441, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %491, i32 noundef 0, i32 noundef %344) #11
  %544 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %479, <4 x i32> noundef %323, <4 x float> noundef %542, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %491, i32 noundef 2, i32 noundef %344) #11
  %545 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %480, <4 x i32> noundef %323, <4 x float> noundef %543, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %491, i32 noundef 2, i32 noundef %344) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %546 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %508, i32 %83, i32 0)
  %547 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %510, i32 %83, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %548 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %465, <4 x i32> noundef %340, <4 x float> noundef %446, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %482, i32 noundef 1, i32 noundef %344) #11
  %549 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %466, <4 x i32> noundef %340, <4 x float> noundef %447, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %482, i32 noundef 1, i32 noundef %344) #11
  %550 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %473, <4 x i32> noundef %341, <4 x float> noundef %548, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %482, i32 noundef 3, i32 noundef %344) #11
  %551 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %474, <4 x i32> noundef %341, <4 x float> noundef %549, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %482, i32 noundef 3, i32 noundef %344) #11
  %552 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %467, <4 x i32> noundef %340, <4 x float> noundef %450, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %485, i32 noundef 1, i32 noundef %344) #11
  %553 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %468, <4 x i32> noundef %340, <4 x float> noundef %451, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %485, i32 noundef 1, i32 noundef %344) #11
  %554 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %475, <4 x i32> noundef %341, <4 x float> noundef %552, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %485, i32 noundef 3, i32 noundef %344) #11
  %555 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %476, <4 x i32> noundef %341, <4 x float> noundef %553, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %485, i32 noundef 3, i32 noundef %344) #11
  %556 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %469, <4 x i32> noundef %340, <4 x float> noundef %454, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %488, i32 noundef 1, i32 noundef %344) #11
  %557 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %470, <4 x i32> noundef %340, <4 x float> noundef %455, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %488, i32 noundef 1, i32 noundef %344) #11
  %558 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %477, <4 x i32> noundef %341, <4 x float> noundef %556, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %488, i32 noundef 3, i32 noundef %344) #11
  %559 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %478, <4 x i32> noundef %341, <4 x float> noundef %557, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %488, i32 noundef 3, i32 noundef %344) #11
  %560 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %471, <4 x i32> noundef %340, <4 x float> noundef %458, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %491, i32 noundef 1, i32 noundef %344) #11
  %561 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %472, <4 x i32> noundef %340, <4 x float> noundef %459, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %491, i32 noundef 1, i32 noundef %344) #11
  %562 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %479, <4 x i32> noundef %341, <4 x float> noundef %560, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %491, i32 noundef 3, i32 noundef %344) #11
  %563 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %480, <4 x i32> noundef %341, <4 x float> noundef %561, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %491, i32 noundef 3, i32 noundef %344) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %564 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %508, i32 %87, i32 0)
  %565 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %510, i32 %87, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %566 = or disjoint i32 %193, 1024
  %567 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %566, i32 %95, i32 0)
  %568 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %566, i32 %100, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %569 = load <4 x i32>, ptr addrspace(3) %345, align 16, !tbaa !11
  %570 = load <4 x i32>, ptr addrspace(3) %347, align 16, !tbaa !11
  %571 = load <4 x i32>, ptr addrspace(3) %349, align 16, !tbaa !11
  %572 = load <4 x i32>, ptr addrspace(3) %351, align 16, !tbaa !11
  %573 = load <4 x i32>, ptr addrspace(3) %353, align 16, !tbaa !11
  %574 = load <4 x i32>, ptr addrspace(3) %355, align 16, !tbaa !11
  %575 = load <4 x i32>, ptr addrspace(3) %357, align 16, !tbaa !11
  %576 = load <4 x i32>, ptr addrspace(3) %359, align 16, !tbaa !11
  %577 = load <4 x i32>, ptr addrspace(3) %361, align 16, !tbaa !11
  %578 = load <4 x i32>, ptr addrspace(3) %363, align 16, !tbaa !11
  %579 = load <4 x i32>, ptr addrspace(3) %365, align 16, !tbaa !11
  %580 = load <4 x i32>, ptr addrspace(3) %367, align 16, !tbaa !11
  %581 = load <4 x i32>, ptr addrspace(3) %369, align 16, !tbaa !11
  %582 = load <4 x i32>, ptr addrspace(3) %371, align 16, !tbaa !11
  %583 = load <4 x i32>, ptr addrspace(3) %373, align 16, !tbaa !11
  %584 = load <4 x i32>, ptr addrspace(3) %375, align 16, !tbaa !11
  %585 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %462
  %586 = load i32, ptr addrspace(3) %585, align 4, !tbaa !7
  %587 = or disjoint i32 %193, 7936
  %588 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %587
  %589 = load i32, ptr addrspace(3) %588, align 4, !tbaa !7
  %590 = or disjoint i32 %193, 15104
  %591 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %590
  %592 = load i32, ptr addrspace(3) %591, align 4, !tbaa !7
  %593 = or disjoint i32 %193, 22272
  %594 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %593
  %595 = load i32, ptr addrspace(3) %594, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %196, i32 noundef 16, i32 noundef %162, i32 noundef 640, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %197, i32 noundef 16, i32 noundef %167, i32 noundef 640, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %198, i32 noundef 16, i32 noundef %171, i32 noundef 640, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %199, i32 noundef 16, i32 noundef %175, i32 noundef 640, i32 noundef 0, i32 noundef 0) #11
  %596 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %569, <4 x i32> noundef %405, <4 x float> noundef %494, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %586, i32 noundef 0, i32 noundef %463) #11
  %597 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %570, <4 x i32> noundef %405, <4 x float> noundef %495, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %586, i32 noundef 0, i32 noundef %463) #11
  %598 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %577, <4 x i32> noundef %407, <4 x float> noundef %596, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %586, i32 noundef 2, i32 noundef %463) #11
  %599 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %578, <4 x i32> noundef %407, <4 x float> noundef %597, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %586, i32 noundef 2, i32 noundef %463) #11
  %600 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %571, <4 x i32> noundef %405, <4 x float> noundef %498, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %589, i32 noundef 0, i32 noundef %463) #11
  %601 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %572, <4 x i32> noundef %405, <4 x float> noundef %499, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %589, i32 noundef 0, i32 noundef %463) #11
  %602 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %579, <4 x i32> noundef %407, <4 x float> noundef %600, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %589, i32 noundef 2, i32 noundef %463) #11
  %603 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %580, <4 x i32> noundef %407, <4 x float> noundef %601, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %589, i32 noundef 2, i32 noundef %463) #11
  %604 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %573, <4 x i32> noundef %405, <4 x float> noundef %502, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %592, i32 noundef 0, i32 noundef %463) #11
  %605 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %574, <4 x i32> noundef %405, <4 x float> noundef %503, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %592, i32 noundef 0, i32 noundef %463) #11
  %606 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %581, <4 x i32> noundef %407, <4 x float> noundef %604, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %592, i32 noundef 2, i32 noundef %463) #11
  %607 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %582, <4 x i32> noundef %407, <4 x float> noundef %605, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %592, i32 noundef 2, i32 noundef %463) #11
  %608 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %575, <4 x i32> noundef %405, <4 x float> noundef %506, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %595, i32 noundef 0, i32 noundef %463) #11
  %609 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %576, <4 x i32> noundef %405, <4 x float> noundef %507, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %595, i32 noundef 0, i32 noundef %463) #11
  %610 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %583, <4 x i32> noundef %407, <4 x float> noundef %608, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %595, i32 noundef 2, i32 noundef %463) #11
  %611 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %584, <4 x i32> noundef %407, <4 x float> noundef %609, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %595, i32 noundef 2, i32 noundef %463) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %612 = or disjoint i32 %181, 10240
  %613 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %612, i32 %75, i32 0)
  %614 = or disjoint i32 %181, 11264
  %615 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %614, i32 %75, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %616 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %569, <4 x i32> noundef %424, <4 x float> noundef %514, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %586, i32 noundef 1, i32 noundef %463) #11
  %617 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %570, <4 x i32> noundef %424, <4 x float> noundef %515, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %586, i32 noundef 1, i32 noundef %463) #11
  %618 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %577, <4 x i32> noundef %425, <4 x float> noundef %616, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %586, i32 noundef 3, i32 noundef %463) #11
  %619 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %578, <4 x i32> noundef %425, <4 x float> noundef %617, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %586, i32 noundef 3, i32 noundef %463) #11
  %620 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %571, <4 x i32> noundef %424, <4 x float> noundef %518, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %589, i32 noundef 1, i32 noundef %463) #11
  %621 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %572, <4 x i32> noundef %424, <4 x float> noundef %519, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %589, i32 noundef 1, i32 noundef %463) #11
  %622 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %579, <4 x i32> noundef %425, <4 x float> noundef %620, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %589, i32 noundef 3, i32 noundef %463) #11
  %623 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %580, <4 x i32> noundef %425, <4 x float> noundef %621, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %589, i32 noundef 3, i32 noundef %463) #11
  %624 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %573, <4 x i32> noundef %424, <4 x float> noundef %522, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %592, i32 noundef 1, i32 noundef %463) #11
  %625 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %574, <4 x i32> noundef %424, <4 x float> noundef %523, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %592, i32 noundef 1, i32 noundef %463) #11
  %626 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %581, <4 x i32> noundef %425, <4 x float> noundef %624, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %592, i32 noundef 3, i32 noundef %463) #11
  %627 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %582, <4 x i32> noundef %425, <4 x float> noundef %625, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %592, i32 noundef 3, i32 noundef %463) #11
  %628 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %575, <4 x i32> noundef %424, <4 x float> noundef %526, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %595, i32 noundef 1, i32 noundef %463) #11
  %629 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %576, <4 x i32> noundef %424, <4 x float> noundef %527, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %595, i32 noundef 1, i32 noundef %463) #11
  %630 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %583, <4 x i32> noundef %425, <4 x float> noundef %628, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %595, i32 noundef 3, i32 noundef %463) #11
  %631 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %584, <4 x i32> noundef %425, <4 x float> noundef %629, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %595, i32 noundef 3, i32 noundef %463) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %632 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %612, i32 %79, i32 0)
  %633 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %614, i32 %79, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %634 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %569, <4 x i32> noundef %442, <4 x float> noundef %532, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %586, i32 noundef 0, i32 noundef %464) #11
  %635 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %570, <4 x i32> noundef %442, <4 x float> noundef %533, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %586, i32 noundef 0, i32 noundef %464) #11
  %636 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %577, <4 x i32> noundef %443, <4 x float> noundef %634, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %586, i32 noundef 2, i32 noundef %464) #11
  %637 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %578, <4 x i32> noundef %443, <4 x float> noundef %635, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %586, i32 noundef 2, i32 noundef %464) #11
  %638 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %571, <4 x i32> noundef %442, <4 x float> noundef %536, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %589, i32 noundef 0, i32 noundef %464) #11
  %639 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %572, <4 x i32> noundef %442, <4 x float> noundef %537, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %589, i32 noundef 0, i32 noundef %464) #11
  %640 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %579, <4 x i32> noundef %443, <4 x float> noundef %638, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %589, i32 noundef 2, i32 noundef %464) #11
  %641 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %580, <4 x i32> noundef %443, <4 x float> noundef %639, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %589, i32 noundef 2, i32 noundef %464) #11
  %642 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %573, <4 x i32> noundef %442, <4 x float> noundef %540, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %592, i32 noundef 0, i32 noundef %464) #11
  %643 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %574, <4 x i32> noundef %442, <4 x float> noundef %541, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %592, i32 noundef 0, i32 noundef %464) #11
  %644 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %581, <4 x i32> noundef %443, <4 x float> noundef %642, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %592, i32 noundef 2, i32 noundef %464) #11
  %645 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %582, <4 x i32> noundef %443, <4 x float> noundef %643, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %592, i32 noundef 2, i32 noundef %464) #11
  %646 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %575, <4 x i32> noundef %442, <4 x float> noundef %544, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %595, i32 noundef 0, i32 noundef %464) #11
  %647 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %576, <4 x i32> noundef %442, <4 x float> noundef %545, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %595, i32 noundef 0, i32 noundef %464) #11
  %648 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %583, <4 x i32> noundef %443, <4 x float> noundef %646, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %595, i32 noundef 2, i32 noundef %464) #11
  %649 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %584, <4 x i32> noundef %443, <4 x float> noundef %647, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %595, i32 noundef 2, i32 noundef %464) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %650 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %612, i32 %83, i32 0)
  %651 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %614, i32 %83, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %652 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %569, <4 x i32> noundef %460, <4 x float> noundef %550, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %586, i32 noundef 1, i32 noundef %464) #11
  %653 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %570, <4 x i32> noundef %460, <4 x float> noundef %551, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %586, i32 noundef 1, i32 noundef %464) #11
  %654 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %577, <4 x i32> noundef %461, <4 x float> noundef %652, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %586, i32 noundef 3, i32 noundef %464) #11
  %655 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %578, <4 x i32> noundef %461, <4 x float> noundef %653, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %586, i32 noundef 3, i32 noundef %464) #11
  %656 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %571, <4 x i32> noundef %460, <4 x float> noundef %554, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %589, i32 noundef 1, i32 noundef %464) #11
  %657 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %572, <4 x i32> noundef %460, <4 x float> noundef %555, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %589, i32 noundef 1, i32 noundef %464) #11
  %658 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %579, <4 x i32> noundef %461, <4 x float> noundef %656, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %589, i32 noundef 3, i32 noundef %464) #11
  %659 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %580, <4 x i32> noundef %461, <4 x float> noundef %657, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %589, i32 noundef 3, i32 noundef %464) #11
  %660 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %573, <4 x i32> noundef %460, <4 x float> noundef %558, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %592, i32 noundef 1, i32 noundef %464) #11
  %661 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %574, <4 x i32> noundef %460, <4 x float> noundef %559, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %592, i32 noundef 1, i32 noundef %464) #11
  %662 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %581, <4 x i32> noundef %461, <4 x float> noundef %660, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %592, i32 noundef 3, i32 noundef %464) #11
  %663 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %582, <4 x i32> noundef %461, <4 x float> noundef %661, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %592, i32 noundef 3, i32 noundef %464) #11
  %664 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %575, <4 x i32> noundef %460, <4 x float> noundef %562, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %595, i32 noundef 1, i32 noundef %464) #11
  %665 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %576, <4 x i32> noundef %460, <4 x float> noundef %563, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %595, i32 noundef 1, i32 noundef %464) #11
  %666 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %583, <4 x i32> noundef %461, <4 x float> noundef %664, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %595, i32 noundef 3, i32 noundef %464) #11
  %667 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %584, <4 x i32> noundef %461, <4 x float> noundef %665, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %595, i32 noundef 3, i32 noundef %464) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %668 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %612, i32 %87, i32 0)
  %669 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %614, i32 %87, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %670 = or disjoint i32 %193, 1280
  %671 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %670, i32 %95, i32 0)
  %672 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %670, i32 %100, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %673 = load <4 x i32>, ptr addrspace(3) %216, align 16, !tbaa !11
  %674 = load <4 x i32>, ptr addrspace(3) %219, align 16, !tbaa !11
  %675 = load <4 x i32>, ptr addrspace(3) %222, align 16, !tbaa !11
  %676 = load <4 x i32>, ptr addrspace(3) %225, align 16, !tbaa !11
  %677 = load <4 x i32>, ptr addrspace(3) %228, align 16, !tbaa !11
  %678 = load <4 x i32>, ptr addrspace(3) %231, align 16, !tbaa !11
  %679 = load <4 x i32>, ptr addrspace(3) %234, align 16, !tbaa !11
  %680 = load <4 x i32>, ptr addrspace(3) %237, align 16, !tbaa !11
  %681 = load <4 x i32>, ptr addrspace(3) %241, align 16, !tbaa !11
  %682 = load <4 x i32>, ptr addrspace(3) %243, align 16, !tbaa !11
  %683 = load <4 x i32>, ptr addrspace(3) %245, align 16, !tbaa !11
  %684 = load <4 x i32>, ptr addrspace(3) %247, align 16, !tbaa !11
  %685 = load <4 x i32>, ptr addrspace(3) %249, align 16, !tbaa !11
  %686 = load <4 x i32>, ptr addrspace(3) %251, align 16, !tbaa !11
  %687 = load <4 x i32>, ptr addrspace(3) %253, align 16, !tbaa !11
  %688 = load <4 x i32>, ptr addrspace(3) %255, align 16, !tbaa !11
  %689 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %566
  %690 = load i32, ptr addrspace(3) %689, align 4, !tbaa !7
  %691 = or disjoint i32 %193, 8192
  %692 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %691
  %693 = load i32, ptr addrspace(3) %692, align 4, !tbaa !7
  %694 = or disjoint i32 %193, 15360
  %695 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %694
  %696 = load i32, ptr addrspace(3) %695, align 4, !tbaa !7
  %697 = or disjoint i32 %193, 22528
  %698 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %697
  %699 = load i32, ptr addrspace(3) %698, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef %163, i32 noundef 16, i32 noundef %162, i32 noundef 768, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %168, i32 noundef 16, i32 noundef %167, i32 noundef 768, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %172, i32 noundef 16, i32 noundef %171, i32 noundef 768, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %176, i32 noundef 16, i32 noundef %175, i32 noundef 768, i32 noundef 0, i32 noundef 0) #11
  %700 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %673, <4 x i32> noundef %509, <4 x float> noundef %598, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %690, i32 noundef 0, i32 noundef %567) #11
  %701 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %674, <4 x i32> noundef %509, <4 x float> noundef %599, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %690, i32 noundef 0, i32 noundef %567) #11
  %702 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %681, <4 x i32> noundef %511, <4 x float> noundef %700, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %690, i32 noundef 2, i32 noundef %567) #11
  %703 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %682, <4 x i32> noundef %511, <4 x float> noundef %701, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %690, i32 noundef 2, i32 noundef %567) #11
  %704 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %675, <4 x i32> noundef %509, <4 x float> noundef %602, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %693, i32 noundef 0, i32 noundef %567) #11
  %705 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %676, <4 x i32> noundef %509, <4 x float> noundef %603, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %693, i32 noundef 0, i32 noundef %567) #11
  %706 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %683, <4 x i32> noundef %511, <4 x float> noundef %704, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %693, i32 noundef 2, i32 noundef %567) #11
  %707 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %684, <4 x i32> noundef %511, <4 x float> noundef %705, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %693, i32 noundef 2, i32 noundef %567) #11
  %708 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %677, <4 x i32> noundef %509, <4 x float> noundef %606, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %696, i32 noundef 0, i32 noundef %567) #11
  %709 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %678, <4 x i32> noundef %509, <4 x float> noundef %607, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %696, i32 noundef 0, i32 noundef %567) #11
  %710 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %685, <4 x i32> noundef %511, <4 x float> noundef %708, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %696, i32 noundef 2, i32 noundef %567) #11
  %711 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %686, <4 x i32> noundef %511, <4 x float> noundef %709, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %696, i32 noundef 2, i32 noundef %567) #11
  %712 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %679, <4 x i32> noundef %509, <4 x float> noundef %610, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %699, i32 noundef 0, i32 noundef %567) #11
  %713 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %680, <4 x i32> noundef %509, <4 x float> noundef %611, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %699, i32 noundef 0, i32 noundef %567) #11
  %714 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %687, <4 x i32> noundef %511, <4 x float> noundef %712, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %699, i32 noundef 2, i32 noundef %567) #11
  %715 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %688, <4 x i32> noundef %511, <4 x float> noundef %713, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %699, i32 noundef 2, i32 noundef %567) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %716 = or disjoint i32 %181, 12288
  %717 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %716, i32 %75, i32 0)
  %718 = or disjoint i32 %181, 13312
  %719 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %718, i32 %75, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %720 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %673, <4 x i32> noundef %528, <4 x float> noundef %618, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %690, i32 noundef 1, i32 noundef %567) #11
  %721 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %674, <4 x i32> noundef %528, <4 x float> noundef %619, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %690, i32 noundef 1, i32 noundef %567) #11
  %722 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %681, <4 x i32> noundef %529, <4 x float> noundef %720, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %690, i32 noundef 3, i32 noundef %567) #11
  %723 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %682, <4 x i32> noundef %529, <4 x float> noundef %721, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %690, i32 noundef 3, i32 noundef %567) #11
  %724 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %675, <4 x i32> noundef %528, <4 x float> noundef %622, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %693, i32 noundef 1, i32 noundef %567) #11
  %725 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %676, <4 x i32> noundef %528, <4 x float> noundef %623, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %693, i32 noundef 1, i32 noundef %567) #11
  %726 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %683, <4 x i32> noundef %529, <4 x float> noundef %724, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %693, i32 noundef 3, i32 noundef %567) #11
  %727 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %684, <4 x i32> noundef %529, <4 x float> noundef %725, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %693, i32 noundef 3, i32 noundef %567) #11
  %728 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %677, <4 x i32> noundef %528, <4 x float> noundef %626, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %696, i32 noundef 1, i32 noundef %567) #11
  %729 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %678, <4 x i32> noundef %528, <4 x float> noundef %627, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %696, i32 noundef 1, i32 noundef %567) #11
  %730 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %685, <4 x i32> noundef %529, <4 x float> noundef %728, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %696, i32 noundef 3, i32 noundef %567) #11
  %731 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %686, <4 x i32> noundef %529, <4 x float> noundef %729, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %696, i32 noundef 3, i32 noundef %567) #11
  %732 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %679, <4 x i32> noundef %528, <4 x float> noundef %630, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %699, i32 noundef 1, i32 noundef %567) #11
  %733 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %680, <4 x i32> noundef %528, <4 x float> noundef %631, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %699, i32 noundef 1, i32 noundef %567) #11
  %734 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %687, <4 x i32> noundef %529, <4 x float> noundef %732, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %699, i32 noundef 3, i32 noundef %567) #11
  %735 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %688, <4 x i32> noundef %529, <4 x float> noundef %733, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %699, i32 noundef 3, i32 noundef %567) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %736 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %716, i32 %79, i32 0)
  %737 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %718, i32 %79, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %738 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %673, <4 x i32> noundef %546, <4 x float> noundef %636, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %690, i32 noundef 0, i32 noundef %568) #11
  %739 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %674, <4 x i32> noundef %546, <4 x float> noundef %637, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %690, i32 noundef 0, i32 noundef %568) #11
  %740 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %681, <4 x i32> noundef %547, <4 x float> noundef %738, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %690, i32 noundef 2, i32 noundef %568) #11
  %741 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %682, <4 x i32> noundef %547, <4 x float> noundef %739, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %690, i32 noundef 2, i32 noundef %568) #11
  %742 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %675, <4 x i32> noundef %546, <4 x float> noundef %640, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %693, i32 noundef 0, i32 noundef %568) #11
  %743 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %676, <4 x i32> noundef %546, <4 x float> noundef %641, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %693, i32 noundef 0, i32 noundef %568) #11
  %744 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %683, <4 x i32> noundef %547, <4 x float> noundef %742, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %693, i32 noundef 2, i32 noundef %568) #11
  %745 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %684, <4 x i32> noundef %547, <4 x float> noundef %743, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %693, i32 noundef 2, i32 noundef %568) #11
  %746 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %677, <4 x i32> noundef %546, <4 x float> noundef %644, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %696, i32 noundef 0, i32 noundef %568) #11
  %747 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %678, <4 x i32> noundef %546, <4 x float> noundef %645, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %696, i32 noundef 0, i32 noundef %568) #11
  %748 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %685, <4 x i32> noundef %547, <4 x float> noundef %746, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %696, i32 noundef 2, i32 noundef %568) #11
  %749 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %686, <4 x i32> noundef %547, <4 x float> noundef %747, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %696, i32 noundef 2, i32 noundef %568) #11
  %750 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %679, <4 x i32> noundef %546, <4 x float> noundef %648, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %699, i32 noundef 0, i32 noundef %568) #11
  %751 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %680, <4 x i32> noundef %546, <4 x float> noundef %649, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %699, i32 noundef 0, i32 noundef %568) #11
  %752 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %687, <4 x i32> noundef %547, <4 x float> noundef %750, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %699, i32 noundef 2, i32 noundef %568) #11
  %753 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %688, <4 x i32> noundef %547, <4 x float> noundef %751, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %699, i32 noundef 2, i32 noundef %568) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %754 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %716, i32 %83, i32 0)
  %755 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %718, i32 %83, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %756 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %673, <4 x i32> noundef %564, <4 x float> noundef %654, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %690, i32 noundef 1, i32 noundef %568) #11
  %757 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %674, <4 x i32> noundef %564, <4 x float> noundef %655, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %690, i32 noundef 1, i32 noundef %568) #11
  %758 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %681, <4 x i32> noundef %565, <4 x float> noundef %756, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %690, i32 noundef 3, i32 noundef %568) #11
  %759 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %682, <4 x i32> noundef %565, <4 x float> noundef %757, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %690, i32 noundef 3, i32 noundef %568) #11
  %760 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %675, <4 x i32> noundef %564, <4 x float> noundef %658, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %693, i32 noundef 1, i32 noundef %568) #11
  %761 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %676, <4 x i32> noundef %564, <4 x float> noundef %659, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %693, i32 noundef 1, i32 noundef %568) #11
  %762 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %683, <4 x i32> noundef %565, <4 x float> noundef %760, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %693, i32 noundef 3, i32 noundef %568) #11
  %763 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %684, <4 x i32> noundef %565, <4 x float> noundef %761, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %693, i32 noundef 3, i32 noundef %568) #11
  %764 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %677, <4 x i32> noundef %564, <4 x float> noundef %662, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %696, i32 noundef 1, i32 noundef %568) #11
  %765 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %678, <4 x i32> noundef %564, <4 x float> noundef %663, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %696, i32 noundef 1, i32 noundef %568) #11
  %766 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %685, <4 x i32> noundef %565, <4 x float> noundef %764, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %696, i32 noundef 3, i32 noundef %568) #11
  %767 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %686, <4 x i32> noundef %565, <4 x float> noundef %765, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %696, i32 noundef 3, i32 noundef %568) #11
  %768 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %679, <4 x i32> noundef %564, <4 x float> noundef %666, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %699, i32 noundef 1, i32 noundef %568) #11
  %769 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %680, <4 x i32> noundef %564, <4 x float> noundef %667, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %699, i32 noundef 1, i32 noundef %568) #11
  %770 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %687, <4 x i32> noundef %565, <4 x float> noundef %768, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %699, i32 noundef 3, i32 noundef %568) #11
  %771 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %688, <4 x i32> noundef %565, <4 x float> noundef %769, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %699, i32 noundef 3, i32 noundef %568) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %772 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %716, i32 %87, i32 0)
  %773 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %718, i32 %87, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %774 = or disjoint i32 %193, 1536
  %775 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %774, i32 %95, i32 0)
  %776 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %774, i32 %100, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %777 = load <4 x i32>, ptr addrspace(3) %345, align 16, !tbaa !11
  %778 = load <4 x i32>, ptr addrspace(3) %347, align 16, !tbaa !11
  %779 = load <4 x i32>, ptr addrspace(3) %349, align 16, !tbaa !11
  %780 = load <4 x i32>, ptr addrspace(3) %351, align 16, !tbaa !11
  %781 = load <4 x i32>, ptr addrspace(3) %353, align 16, !tbaa !11
  %782 = load <4 x i32>, ptr addrspace(3) %355, align 16, !tbaa !11
  %783 = load <4 x i32>, ptr addrspace(3) %357, align 16, !tbaa !11
  %784 = load <4 x i32>, ptr addrspace(3) %359, align 16, !tbaa !11
  %785 = load <4 x i32>, ptr addrspace(3) %361, align 16, !tbaa !11
  %786 = load <4 x i32>, ptr addrspace(3) %363, align 16, !tbaa !11
  %787 = load <4 x i32>, ptr addrspace(3) %365, align 16, !tbaa !11
  %788 = load <4 x i32>, ptr addrspace(3) %367, align 16, !tbaa !11
  %789 = load <4 x i32>, ptr addrspace(3) %369, align 16, !tbaa !11
  %790 = load <4 x i32>, ptr addrspace(3) %371, align 16, !tbaa !11
  %791 = load <4 x i32>, ptr addrspace(3) %373, align 16, !tbaa !11
  %792 = load <4 x i32>, ptr addrspace(3) %375, align 16, !tbaa !11
  %793 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %670
  %794 = load i32, ptr addrspace(3) %793, align 4, !tbaa !7
  %795 = or disjoint i32 %193, 8448
  %796 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %795
  %797 = load i32, ptr addrspace(3) %796, align 4, !tbaa !7
  %798 = or disjoint i32 %193, 15616
  %799 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %798
  %800 = load i32, ptr addrspace(3) %799, align 4, !tbaa !7
  %801 = or disjoint i32 %193, 22784
  %802 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %801
  %803 = load i32, ptr addrspace(3) %802, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %196, i32 noundef 16, i32 noundef %162, i32 noundef 896, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %197, i32 noundef 16, i32 noundef %167, i32 noundef 896, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %198, i32 noundef 16, i32 noundef %171, i32 noundef 896, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %199, i32 noundef 16, i32 noundef %175, i32 noundef 896, i32 noundef 0, i32 noundef 0) #11
  %804 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %777, <4 x i32> noundef %613, <4 x float> noundef %702, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %794, i32 noundef 0, i32 noundef %671) #11
  %805 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %778, <4 x i32> noundef %613, <4 x float> noundef %703, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %794, i32 noundef 0, i32 noundef %671) #11
  %806 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %785, <4 x i32> noundef %615, <4 x float> noundef %804, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %794, i32 noundef 2, i32 noundef %671) #11
  %807 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %786, <4 x i32> noundef %615, <4 x float> noundef %805, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %794, i32 noundef 2, i32 noundef %671) #11
  %808 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %779, <4 x i32> noundef %613, <4 x float> noundef %706, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %797, i32 noundef 0, i32 noundef %671) #11
  %809 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %780, <4 x i32> noundef %613, <4 x float> noundef %707, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %797, i32 noundef 0, i32 noundef %671) #11
  %810 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %787, <4 x i32> noundef %615, <4 x float> noundef %808, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %797, i32 noundef 2, i32 noundef %671) #11
  %811 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %788, <4 x i32> noundef %615, <4 x float> noundef %809, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %797, i32 noundef 2, i32 noundef %671) #11
  %812 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %781, <4 x i32> noundef %613, <4 x float> noundef %710, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %800, i32 noundef 0, i32 noundef %671) #11
  %813 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %782, <4 x i32> noundef %613, <4 x float> noundef %711, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %800, i32 noundef 0, i32 noundef %671) #11
  %814 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %789, <4 x i32> noundef %615, <4 x float> noundef %812, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %800, i32 noundef 2, i32 noundef %671) #11
  %815 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %790, <4 x i32> noundef %615, <4 x float> noundef %813, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %800, i32 noundef 2, i32 noundef %671) #11
  %816 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %783, <4 x i32> noundef %613, <4 x float> noundef %714, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %803, i32 noundef 0, i32 noundef %671) #11
  %817 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %784, <4 x i32> noundef %613, <4 x float> noundef %715, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %803, i32 noundef 0, i32 noundef %671) #11
  %818 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %791, <4 x i32> noundef %615, <4 x float> noundef %816, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %803, i32 noundef 2, i32 noundef %671) #11
  %819 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %792, <4 x i32> noundef %615, <4 x float> noundef %817, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %803, i32 noundef 2, i32 noundef %671) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %820 = or disjoint i32 %181, 14336
  %821 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %820, i32 %75, i32 0)
  %822 = or disjoint i32 %181, 15360
  %823 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %822, i32 %75, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %824 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %777, <4 x i32> noundef %632, <4 x float> noundef %722, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %794, i32 noundef 1, i32 noundef %671) #11
  %825 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %778, <4 x i32> noundef %632, <4 x float> noundef %723, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %794, i32 noundef 1, i32 noundef %671) #11
  %826 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %785, <4 x i32> noundef %633, <4 x float> noundef %824, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %794, i32 noundef 3, i32 noundef %671) #11
  %827 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %786, <4 x i32> noundef %633, <4 x float> noundef %825, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %794, i32 noundef 3, i32 noundef %671) #11
  %828 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %779, <4 x i32> noundef %632, <4 x float> noundef %726, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %797, i32 noundef 1, i32 noundef %671) #11
  %829 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %780, <4 x i32> noundef %632, <4 x float> noundef %727, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %797, i32 noundef 1, i32 noundef %671) #11
  %830 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %787, <4 x i32> noundef %633, <4 x float> noundef %828, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %797, i32 noundef 3, i32 noundef %671) #11
  %831 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %788, <4 x i32> noundef %633, <4 x float> noundef %829, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %797, i32 noundef 3, i32 noundef %671) #11
  %832 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %781, <4 x i32> noundef %632, <4 x float> noundef %730, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %800, i32 noundef 1, i32 noundef %671) #11
  %833 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %782, <4 x i32> noundef %632, <4 x float> noundef %731, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %800, i32 noundef 1, i32 noundef %671) #11
  %834 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %789, <4 x i32> noundef %633, <4 x float> noundef %832, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %800, i32 noundef 3, i32 noundef %671) #11
  %835 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %790, <4 x i32> noundef %633, <4 x float> noundef %833, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %800, i32 noundef 3, i32 noundef %671) #11
  %836 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %783, <4 x i32> noundef %632, <4 x float> noundef %734, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %803, i32 noundef 1, i32 noundef %671) #11
  %837 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %784, <4 x i32> noundef %632, <4 x float> noundef %735, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %803, i32 noundef 1, i32 noundef %671) #11
  %838 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %791, <4 x i32> noundef %633, <4 x float> noundef %836, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %803, i32 noundef 3, i32 noundef %671) #11
  %839 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %792, <4 x i32> noundef %633, <4 x float> noundef %837, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %803, i32 noundef 3, i32 noundef %671) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %840 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %820, i32 %79, i32 0)
  %841 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %822, i32 %79, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %842 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %777, <4 x i32> noundef %650, <4 x float> noundef %740, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %794, i32 noundef 0, i32 noundef %672) #11
  %843 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %778, <4 x i32> noundef %650, <4 x float> noundef %741, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %794, i32 noundef 0, i32 noundef %672) #11
  %844 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %785, <4 x i32> noundef %651, <4 x float> noundef %842, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %794, i32 noundef 2, i32 noundef %672) #11
  %845 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %786, <4 x i32> noundef %651, <4 x float> noundef %843, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %794, i32 noundef 2, i32 noundef %672) #11
  %846 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %779, <4 x i32> noundef %650, <4 x float> noundef %744, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %797, i32 noundef 0, i32 noundef %672) #11
  %847 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %780, <4 x i32> noundef %650, <4 x float> noundef %745, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %797, i32 noundef 0, i32 noundef %672) #11
  %848 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %787, <4 x i32> noundef %651, <4 x float> noundef %846, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %797, i32 noundef 2, i32 noundef %672) #11
  %849 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %788, <4 x i32> noundef %651, <4 x float> noundef %847, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %797, i32 noundef 2, i32 noundef %672) #11
  %850 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %781, <4 x i32> noundef %650, <4 x float> noundef %748, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %800, i32 noundef 0, i32 noundef %672) #11
  %851 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %782, <4 x i32> noundef %650, <4 x float> noundef %749, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %800, i32 noundef 0, i32 noundef %672) #11
  %852 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %789, <4 x i32> noundef %651, <4 x float> noundef %850, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %800, i32 noundef 2, i32 noundef %672) #11
  %853 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %790, <4 x i32> noundef %651, <4 x float> noundef %851, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %800, i32 noundef 2, i32 noundef %672) #11
  %854 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %783, <4 x i32> noundef %650, <4 x float> noundef %752, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %803, i32 noundef 0, i32 noundef %672) #11
  %855 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %784, <4 x i32> noundef %650, <4 x float> noundef %753, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %803, i32 noundef 0, i32 noundef %672) #11
  %856 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %791, <4 x i32> noundef %651, <4 x float> noundef %854, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %803, i32 noundef 2, i32 noundef %672) #11
  %857 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %792, <4 x i32> noundef %651, <4 x float> noundef %855, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %803, i32 noundef 2, i32 noundef %672) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %858 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %820, i32 %83, i32 0)
  %859 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %822, i32 %83, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %860 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %777, <4 x i32> noundef %668, <4 x float> noundef %758, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %794, i32 noundef 1, i32 noundef %672) #11
  %861 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %778, <4 x i32> noundef %668, <4 x float> noundef %759, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %794, i32 noundef 1, i32 noundef %672) #11
  %862 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %785, <4 x i32> noundef %669, <4 x float> noundef %860, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %794, i32 noundef 3, i32 noundef %672) #11
  %863 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %786, <4 x i32> noundef %669, <4 x float> noundef %861, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %794, i32 noundef 3, i32 noundef %672) #11
  %864 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %779, <4 x i32> noundef %668, <4 x float> noundef %762, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %797, i32 noundef 1, i32 noundef %672) #11
  %865 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %780, <4 x i32> noundef %668, <4 x float> noundef %763, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %797, i32 noundef 1, i32 noundef %672) #11
  %866 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %787, <4 x i32> noundef %669, <4 x float> noundef %864, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %797, i32 noundef 3, i32 noundef %672) #11
  %867 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %788, <4 x i32> noundef %669, <4 x float> noundef %865, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %797, i32 noundef 3, i32 noundef %672) #11
  %868 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %781, <4 x i32> noundef %668, <4 x float> noundef %766, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %800, i32 noundef 1, i32 noundef %672) #11
  %869 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %782, <4 x i32> noundef %668, <4 x float> noundef %767, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %800, i32 noundef 1, i32 noundef %672) #11
  %870 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %789, <4 x i32> noundef %669, <4 x float> noundef %868, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %800, i32 noundef 3, i32 noundef %672) #11
  %871 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %790, <4 x i32> noundef %669, <4 x float> noundef %869, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %800, i32 noundef 3, i32 noundef %672) #11
  %872 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %783, <4 x i32> noundef %668, <4 x float> noundef %770, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %803, i32 noundef 1, i32 noundef %672) #11
  %873 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %784, <4 x i32> noundef %668, <4 x float> noundef %771, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %803, i32 noundef 1, i32 noundef %672) #11
  %874 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %791, <4 x i32> noundef %669, <4 x float> noundef %872, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %803, i32 noundef 3, i32 noundef %672) #11
  %875 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %792, <4 x i32> noundef %669, <4 x float> noundef %873, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %803, i32 noundef 3, i32 noundef %672) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %876 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %820, i32 %87, i32 0)
  %877 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %822, i32 %87, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %878 = or disjoint i32 %193, 1792
  %879 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %878, i32 %95, i32 0)
  %880 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %878, i32 %100, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %881 = load <4 x i32>, ptr addrspace(3) %216, align 16, !tbaa !11
  %882 = load <4 x i32>, ptr addrspace(3) %219, align 16, !tbaa !11
  %883 = load <4 x i32>, ptr addrspace(3) %222, align 16, !tbaa !11
  %884 = load <4 x i32>, ptr addrspace(3) %225, align 16, !tbaa !11
  %885 = load <4 x i32>, ptr addrspace(3) %228, align 16, !tbaa !11
  %886 = load <4 x i32>, ptr addrspace(3) %231, align 16, !tbaa !11
  %887 = load <4 x i32>, ptr addrspace(3) %234, align 16, !tbaa !11
  %888 = load <4 x i32>, ptr addrspace(3) %237, align 16, !tbaa !11
  %889 = load <4 x i32>, ptr addrspace(3) %241, align 16, !tbaa !11
  %890 = load <4 x i32>, ptr addrspace(3) %243, align 16, !tbaa !11
  %891 = load <4 x i32>, ptr addrspace(3) %245, align 16, !tbaa !11
  %892 = load <4 x i32>, ptr addrspace(3) %247, align 16, !tbaa !11
  %893 = load <4 x i32>, ptr addrspace(3) %249, align 16, !tbaa !11
  %894 = load <4 x i32>, ptr addrspace(3) %251, align 16, !tbaa !11
  %895 = load <4 x i32>, ptr addrspace(3) %253, align 16, !tbaa !11
  %896 = load <4 x i32>, ptr addrspace(3) %255, align 16, !tbaa !11
  %897 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %774
  %898 = load i32, ptr addrspace(3) %897, align 4, !tbaa !7
  %899 = or disjoint i32 %193, 8704
  %900 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %899
  %901 = load i32, ptr addrspace(3) %900, align 4, !tbaa !7
  %902 = or disjoint i32 %193, 15872
  %903 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %902
  %904 = load i32, ptr addrspace(3) %903, align 4, !tbaa !7
  %905 = or disjoint i32 %193, 23040
  %906 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %905
  %907 = load i32, ptr addrspace(3) %906, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef %163, i32 noundef 16, i32 noundef %162, i32 noundef 1024, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %168, i32 noundef 16, i32 noundef %167, i32 noundef 1024, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %172, i32 noundef 16, i32 noundef %171, i32 noundef 1024, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %176, i32 noundef 16, i32 noundef %175, i32 noundef 1024, i32 noundef 0, i32 noundef 0) #11
  %908 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %881, <4 x i32> noundef %717, <4 x float> noundef %806, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %898, i32 noundef 0, i32 noundef %775) #11
  %909 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %882, <4 x i32> noundef %717, <4 x float> noundef %807, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %898, i32 noundef 0, i32 noundef %775) #11
  %910 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %889, <4 x i32> noundef %719, <4 x float> noundef %908, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %898, i32 noundef 2, i32 noundef %775) #11
  %911 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %890, <4 x i32> noundef %719, <4 x float> noundef %909, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %898, i32 noundef 2, i32 noundef %775) #11
  %912 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %883, <4 x i32> noundef %717, <4 x float> noundef %810, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %901, i32 noundef 0, i32 noundef %775) #11
  %913 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %884, <4 x i32> noundef %717, <4 x float> noundef %811, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %901, i32 noundef 0, i32 noundef %775) #11
  %914 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %891, <4 x i32> noundef %719, <4 x float> noundef %912, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %901, i32 noundef 2, i32 noundef %775) #11
  %915 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %892, <4 x i32> noundef %719, <4 x float> noundef %913, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %901, i32 noundef 2, i32 noundef %775) #11
  %916 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %885, <4 x i32> noundef %717, <4 x float> noundef %814, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %904, i32 noundef 0, i32 noundef %775) #11
  %917 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %886, <4 x i32> noundef %717, <4 x float> noundef %815, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %904, i32 noundef 0, i32 noundef %775) #11
  %918 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %893, <4 x i32> noundef %719, <4 x float> noundef %916, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %904, i32 noundef 2, i32 noundef %775) #11
  %919 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %894, <4 x i32> noundef %719, <4 x float> noundef %917, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %904, i32 noundef 2, i32 noundef %775) #11
  %920 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %887, <4 x i32> noundef %717, <4 x float> noundef %818, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %907, i32 noundef 0, i32 noundef %775) #11
  %921 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %888, <4 x i32> noundef %717, <4 x float> noundef %819, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %907, i32 noundef 0, i32 noundef %775) #11
  %922 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %895, <4 x i32> noundef %719, <4 x float> noundef %920, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %907, i32 noundef 2, i32 noundef %775) #11
  %923 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %896, <4 x i32> noundef %719, <4 x float> noundef %921, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %907, i32 noundef 2, i32 noundef %775) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %924 = or disjoint i32 %181, 16384
  %925 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %924, i32 %75, i32 0)
  %926 = or disjoint i32 %181, 17408
  %927 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %926, i32 %75, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %928 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %881, <4 x i32> noundef %736, <4 x float> noundef %826, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %898, i32 noundef 1, i32 noundef %775) #11
  %929 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %882, <4 x i32> noundef %736, <4 x float> noundef %827, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %898, i32 noundef 1, i32 noundef %775) #11
  %930 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %889, <4 x i32> noundef %737, <4 x float> noundef %928, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %898, i32 noundef 3, i32 noundef %775) #11
  %931 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %890, <4 x i32> noundef %737, <4 x float> noundef %929, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %898, i32 noundef 3, i32 noundef %775) #11
  %932 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %883, <4 x i32> noundef %736, <4 x float> noundef %830, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %901, i32 noundef 1, i32 noundef %775) #11
  %933 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %884, <4 x i32> noundef %736, <4 x float> noundef %831, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %901, i32 noundef 1, i32 noundef %775) #11
  %934 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %891, <4 x i32> noundef %737, <4 x float> noundef %932, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %901, i32 noundef 3, i32 noundef %775) #11
  %935 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %892, <4 x i32> noundef %737, <4 x float> noundef %933, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %901, i32 noundef 3, i32 noundef %775) #11
  %936 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %885, <4 x i32> noundef %736, <4 x float> noundef %834, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %904, i32 noundef 1, i32 noundef %775) #11
  %937 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %886, <4 x i32> noundef %736, <4 x float> noundef %835, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %904, i32 noundef 1, i32 noundef %775) #11
  %938 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %893, <4 x i32> noundef %737, <4 x float> noundef %936, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %904, i32 noundef 3, i32 noundef %775) #11
  %939 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %894, <4 x i32> noundef %737, <4 x float> noundef %937, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %904, i32 noundef 3, i32 noundef %775) #11
  %940 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %887, <4 x i32> noundef %736, <4 x float> noundef %838, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %907, i32 noundef 1, i32 noundef %775) #11
  %941 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %888, <4 x i32> noundef %736, <4 x float> noundef %839, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %907, i32 noundef 1, i32 noundef %775) #11
  %942 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %895, <4 x i32> noundef %737, <4 x float> noundef %940, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %907, i32 noundef 3, i32 noundef %775) #11
  %943 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %896, <4 x i32> noundef %737, <4 x float> noundef %941, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %907, i32 noundef 3, i32 noundef %775) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %944 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %924, i32 %79, i32 0)
  %945 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %926, i32 %79, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %946 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %881, <4 x i32> noundef %754, <4 x float> noundef %844, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %898, i32 noundef 0, i32 noundef %776) #11
  %947 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %882, <4 x i32> noundef %754, <4 x float> noundef %845, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %898, i32 noundef 0, i32 noundef %776) #11
  %948 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %889, <4 x i32> noundef %755, <4 x float> noundef %946, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %898, i32 noundef 2, i32 noundef %776) #11
  %949 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %890, <4 x i32> noundef %755, <4 x float> noundef %947, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %898, i32 noundef 2, i32 noundef %776) #11
  %950 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %883, <4 x i32> noundef %754, <4 x float> noundef %848, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %901, i32 noundef 0, i32 noundef %776) #11
  %951 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %884, <4 x i32> noundef %754, <4 x float> noundef %849, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %901, i32 noundef 0, i32 noundef %776) #11
  %952 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %891, <4 x i32> noundef %755, <4 x float> noundef %950, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %901, i32 noundef 2, i32 noundef %776) #11
  %953 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %892, <4 x i32> noundef %755, <4 x float> noundef %951, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %901, i32 noundef 2, i32 noundef %776) #11
  %954 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %885, <4 x i32> noundef %754, <4 x float> noundef %852, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %904, i32 noundef 0, i32 noundef %776) #11
  %955 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %886, <4 x i32> noundef %754, <4 x float> noundef %853, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %904, i32 noundef 0, i32 noundef %776) #11
  %956 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %893, <4 x i32> noundef %755, <4 x float> noundef %954, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %904, i32 noundef 2, i32 noundef %776) #11
  %957 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %894, <4 x i32> noundef %755, <4 x float> noundef %955, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %904, i32 noundef 2, i32 noundef %776) #11
  %958 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %887, <4 x i32> noundef %754, <4 x float> noundef %856, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %907, i32 noundef 0, i32 noundef %776) #11
  %959 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %888, <4 x i32> noundef %754, <4 x float> noundef %857, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %907, i32 noundef 0, i32 noundef %776) #11
  %960 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %895, <4 x i32> noundef %755, <4 x float> noundef %958, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %907, i32 noundef 2, i32 noundef %776) #11
  %961 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %896, <4 x i32> noundef %755, <4 x float> noundef %959, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %907, i32 noundef 2, i32 noundef %776) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %962 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %924, i32 %83, i32 0)
  %963 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %926, i32 %83, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %964 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %881, <4 x i32> noundef %772, <4 x float> noundef %862, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %898, i32 noundef 1, i32 noundef %776) #11
  %965 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %882, <4 x i32> noundef %772, <4 x float> noundef %863, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %898, i32 noundef 1, i32 noundef %776) #11
  %966 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %889, <4 x i32> noundef %773, <4 x float> noundef %964, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %898, i32 noundef 3, i32 noundef %776) #11
  %967 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %890, <4 x i32> noundef %773, <4 x float> noundef %965, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %898, i32 noundef 3, i32 noundef %776) #11
  %968 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %883, <4 x i32> noundef %772, <4 x float> noundef %866, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %901, i32 noundef 1, i32 noundef %776) #11
  %969 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %884, <4 x i32> noundef %772, <4 x float> noundef %867, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %901, i32 noundef 1, i32 noundef %776) #11
  %970 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %891, <4 x i32> noundef %773, <4 x float> noundef %968, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %901, i32 noundef 3, i32 noundef %776) #11
  %971 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %892, <4 x i32> noundef %773, <4 x float> noundef %969, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %901, i32 noundef 3, i32 noundef %776) #11
  %972 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %885, <4 x i32> noundef %772, <4 x float> noundef %870, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %904, i32 noundef 1, i32 noundef %776) #11
  %973 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %886, <4 x i32> noundef %772, <4 x float> noundef %871, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %904, i32 noundef 1, i32 noundef %776) #11
  %974 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %893, <4 x i32> noundef %773, <4 x float> noundef %972, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %904, i32 noundef 3, i32 noundef %776) #11
  %975 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %894, <4 x i32> noundef %773, <4 x float> noundef %973, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %904, i32 noundef 3, i32 noundef %776) #11
  %976 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %887, <4 x i32> noundef %772, <4 x float> noundef %874, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %907, i32 noundef 1, i32 noundef %776) #11
  %977 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %888, <4 x i32> noundef %772, <4 x float> noundef %875, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %907, i32 noundef 1, i32 noundef %776) #11
  %978 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %895, <4 x i32> noundef %773, <4 x float> noundef %976, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %907, i32 noundef 3, i32 noundef %776) #11
  %979 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %896, <4 x i32> noundef %773, <4 x float> noundef %977, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %907, i32 noundef 3, i32 noundef %776) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %980 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %924, i32 %87, i32 0)
  %981 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %926, i32 %87, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %982 = or disjoint i32 %193, 2048
  %983 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %982, i32 %95, i32 0)
  %984 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %982, i32 %100, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %985 = load <4 x i32>, ptr addrspace(3) %345, align 16, !tbaa !11
  %986 = load <4 x i32>, ptr addrspace(3) %347, align 16, !tbaa !11
  %987 = load <4 x i32>, ptr addrspace(3) %349, align 16, !tbaa !11
  %988 = load <4 x i32>, ptr addrspace(3) %351, align 16, !tbaa !11
  %989 = load <4 x i32>, ptr addrspace(3) %353, align 16, !tbaa !11
  %990 = load <4 x i32>, ptr addrspace(3) %355, align 16, !tbaa !11
  %991 = load <4 x i32>, ptr addrspace(3) %357, align 16, !tbaa !11
  %992 = load <4 x i32>, ptr addrspace(3) %359, align 16, !tbaa !11
  %993 = load <4 x i32>, ptr addrspace(3) %361, align 16, !tbaa !11
  %994 = load <4 x i32>, ptr addrspace(3) %363, align 16, !tbaa !11
  %995 = load <4 x i32>, ptr addrspace(3) %365, align 16, !tbaa !11
  %996 = load <4 x i32>, ptr addrspace(3) %367, align 16, !tbaa !11
  %997 = load <4 x i32>, ptr addrspace(3) %369, align 16, !tbaa !11
  %998 = load <4 x i32>, ptr addrspace(3) %371, align 16, !tbaa !11
  %999 = load <4 x i32>, ptr addrspace(3) %373, align 16, !tbaa !11
  %1000 = load <4 x i32>, ptr addrspace(3) %375, align 16, !tbaa !11
  %1001 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %878
  %1002 = load i32, ptr addrspace(3) %1001, align 4, !tbaa !7
  %1003 = or disjoint i32 %193, 8960
  %1004 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1003
  %1005 = load i32, ptr addrspace(3) %1004, align 4, !tbaa !7
  %1006 = or disjoint i32 %193, 16128
  %1007 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1006
  %1008 = load i32, ptr addrspace(3) %1007, align 4, !tbaa !7
  %1009 = or disjoint i32 %193, 23296
  %1010 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1009
  %1011 = load i32, ptr addrspace(3) %1010, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %196, i32 noundef 16, i32 noundef %162, i32 noundef 1152, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %197, i32 noundef 16, i32 noundef %167, i32 noundef 1152, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %198, i32 noundef 16, i32 noundef %171, i32 noundef 1152, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %199, i32 noundef 16, i32 noundef %175, i32 noundef 1152, i32 noundef 0, i32 noundef 0) #11
  %1012 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %985, <4 x i32> noundef %821, <4 x float> noundef %910, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1002, i32 noundef 0, i32 noundef %879) #11
  %1013 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %986, <4 x i32> noundef %821, <4 x float> noundef %911, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1002, i32 noundef 0, i32 noundef %879) #11
  %1014 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %993, <4 x i32> noundef %823, <4 x float> noundef %1012, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1002, i32 noundef 2, i32 noundef %879) #11
  %1015 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %994, <4 x i32> noundef %823, <4 x float> noundef %1013, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1002, i32 noundef 2, i32 noundef %879) #11
  %1016 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %987, <4 x i32> noundef %821, <4 x float> noundef %914, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1005, i32 noundef 0, i32 noundef %879) #11
  %1017 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %988, <4 x i32> noundef %821, <4 x float> noundef %915, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1005, i32 noundef 0, i32 noundef %879) #11
  %1018 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %995, <4 x i32> noundef %823, <4 x float> noundef %1016, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1005, i32 noundef 2, i32 noundef %879) #11
  %1019 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %996, <4 x i32> noundef %823, <4 x float> noundef %1017, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1005, i32 noundef 2, i32 noundef %879) #11
  %1020 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %989, <4 x i32> noundef %821, <4 x float> noundef %918, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1008, i32 noundef 0, i32 noundef %879) #11
  %1021 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %990, <4 x i32> noundef %821, <4 x float> noundef %919, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1008, i32 noundef 0, i32 noundef %879) #11
  %1022 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %997, <4 x i32> noundef %823, <4 x float> noundef %1020, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1008, i32 noundef 2, i32 noundef %879) #11
  %1023 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %998, <4 x i32> noundef %823, <4 x float> noundef %1021, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1008, i32 noundef 2, i32 noundef %879) #11
  %1024 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %991, <4 x i32> noundef %821, <4 x float> noundef %922, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1011, i32 noundef 0, i32 noundef %879) #11
  %1025 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %992, <4 x i32> noundef %821, <4 x float> noundef %923, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1011, i32 noundef 0, i32 noundef %879) #11
  %1026 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %999, <4 x i32> noundef %823, <4 x float> noundef %1024, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1011, i32 noundef 2, i32 noundef %879) #11
  %1027 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1000, <4 x i32> noundef %823, <4 x float> noundef %1025, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1011, i32 noundef 2, i32 noundef %879) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1028 = or disjoint i32 %181, 18432
  %1029 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1028, i32 %75, i32 0)
  %1030 = or disjoint i32 %181, 19456
  %1031 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1030, i32 %75, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1032 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %985, <4 x i32> noundef %840, <4 x float> noundef %930, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1002, i32 noundef 1, i32 noundef %879) #11
  %1033 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %986, <4 x i32> noundef %840, <4 x float> noundef %931, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1002, i32 noundef 1, i32 noundef %879) #11
  %1034 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %993, <4 x i32> noundef %841, <4 x float> noundef %1032, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1002, i32 noundef 3, i32 noundef %879) #11
  %1035 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %994, <4 x i32> noundef %841, <4 x float> noundef %1033, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1002, i32 noundef 3, i32 noundef %879) #11
  %1036 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %987, <4 x i32> noundef %840, <4 x float> noundef %934, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1005, i32 noundef 1, i32 noundef %879) #11
  %1037 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %988, <4 x i32> noundef %840, <4 x float> noundef %935, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1005, i32 noundef 1, i32 noundef %879) #11
  %1038 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %995, <4 x i32> noundef %841, <4 x float> noundef %1036, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1005, i32 noundef 3, i32 noundef %879) #11
  %1039 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %996, <4 x i32> noundef %841, <4 x float> noundef %1037, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1005, i32 noundef 3, i32 noundef %879) #11
  %1040 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %989, <4 x i32> noundef %840, <4 x float> noundef %938, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1008, i32 noundef 1, i32 noundef %879) #11
  %1041 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %990, <4 x i32> noundef %840, <4 x float> noundef %939, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1008, i32 noundef 1, i32 noundef %879) #11
  %1042 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %997, <4 x i32> noundef %841, <4 x float> noundef %1040, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1008, i32 noundef 3, i32 noundef %879) #11
  %1043 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %998, <4 x i32> noundef %841, <4 x float> noundef %1041, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1008, i32 noundef 3, i32 noundef %879) #11
  %1044 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %991, <4 x i32> noundef %840, <4 x float> noundef %942, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1011, i32 noundef 1, i32 noundef %879) #11
  %1045 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %992, <4 x i32> noundef %840, <4 x float> noundef %943, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1011, i32 noundef 1, i32 noundef %879) #11
  %1046 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %999, <4 x i32> noundef %841, <4 x float> noundef %1044, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1011, i32 noundef 3, i32 noundef %879) #11
  %1047 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1000, <4 x i32> noundef %841, <4 x float> noundef %1045, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1011, i32 noundef 3, i32 noundef %879) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1048 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1028, i32 %79, i32 0)
  %1049 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1030, i32 %79, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1050 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %985, <4 x i32> noundef %858, <4 x float> noundef %948, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1002, i32 noundef 0, i32 noundef %880) #11
  %1051 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %986, <4 x i32> noundef %858, <4 x float> noundef %949, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1002, i32 noundef 0, i32 noundef %880) #11
  %1052 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %993, <4 x i32> noundef %859, <4 x float> noundef %1050, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1002, i32 noundef 2, i32 noundef %880) #11
  %1053 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %994, <4 x i32> noundef %859, <4 x float> noundef %1051, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1002, i32 noundef 2, i32 noundef %880) #11
  %1054 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %987, <4 x i32> noundef %858, <4 x float> noundef %952, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1005, i32 noundef 0, i32 noundef %880) #11
  %1055 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %988, <4 x i32> noundef %858, <4 x float> noundef %953, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1005, i32 noundef 0, i32 noundef %880) #11
  %1056 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %995, <4 x i32> noundef %859, <4 x float> noundef %1054, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1005, i32 noundef 2, i32 noundef %880) #11
  %1057 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %996, <4 x i32> noundef %859, <4 x float> noundef %1055, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1005, i32 noundef 2, i32 noundef %880) #11
  %1058 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %989, <4 x i32> noundef %858, <4 x float> noundef %956, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1008, i32 noundef 0, i32 noundef %880) #11
  %1059 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %990, <4 x i32> noundef %858, <4 x float> noundef %957, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1008, i32 noundef 0, i32 noundef %880) #11
  %1060 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %997, <4 x i32> noundef %859, <4 x float> noundef %1058, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1008, i32 noundef 2, i32 noundef %880) #11
  %1061 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %998, <4 x i32> noundef %859, <4 x float> noundef %1059, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1008, i32 noundef 2, i32 noundef %880) #11
  %1062 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %991, <4 x i32> noundef %858, <4 x float> noundef %960, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1011, i32 noundef 0, i32 noundef %880) #11
  %1063 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %992, <4 x i32> noundef %858, <4 x float> noundef %961, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1011, i32 noundef 0, i32 noundef %880) #11
  %1064 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %999, <4 x i32> noundef %859, <4 x float> noundef %1062, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1011, i32 noundef 2, i32 noundef %880) #11
  %1065 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1000, <4 x i32> noundef %859, <4 x float> noundef %1063, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1011, i32 noundef 2, i32 noundef %880) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1066 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1028, i32 %83, i32 0)
  %1067 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1030, i32 %83, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1068 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %985, <4 x i32> noundef %876, <4 x float> noundef %966, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1002, i32 noundef 1, i32 noundef %880) #11
  %1069 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %986, <4 x i32> noundef %876, <4 x float> noundef %967, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1002, i32 noundef 1, i32 noundef %880) #11
  %1070 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %993, <4 x i32> noundef %877, <4 x float> noundef %1068, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1002, i32 noundef 3, i32 noundef %880) #11
  %1071 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %994, <4 x i32> noundef %877, <4 x float> noundef %1069, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1002, i32 noundef 3, i32 noundef %880) #11
  %1072 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %987, <4 x i32> noundef %876, <4 x float> noundef %970, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1005, i32 noundef 1, i32 noundef %880) #11
  %1073 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %988, <4 x i32> noundef %876, <4 x float> noundef %971, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1005, i32 noundef 1, i32 noundef %880) #11
  %1074 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %995, <4 x i32> noundef %877, <4 x float> noundef %1072, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1005, i32 noundef 3, i32 noundef %880) #11
  %1075 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %996, <4 x i32> noundef %877, <4 x float> noundef %1073, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1005, i32 noundef 3, i32 noundef %880) #11
  %1076 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %989, <4 x i32> noundef %876, <4 x float> noundef %974, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1008, i32 noundef 1, i32 noundef %880) #11
  %1077 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %990, <4 x i32> noundef %876, <4 x float> noundef %975, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1008, i32 noundef 1, i32 noundef %880) #11
  %1078 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %997, <4 x i32> noundef %877, <4 x float> noundef %1076, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1008, i32 noundef 3, i32 noundef %880) #11
  %1079 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %998, <4 x i32> noundef %877, <4 x float> noundef %1077, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1008, i32 noundef 3, i32 noundef %880) #11
  %1080 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %991, <4 x i32> noundef %876, <4 x float> noundef %978, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1011, i32 noundef 1, i32 noundef %880) #11
  %1081 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %992, <4 x i32> noundef %876, <4 x float> noundef %979, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1011, i32 noundef 1, i32 noundef %880) #11
  %1082 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %999, <4 x i32> noundef %877, <4 x float> noundef %1080, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1011, i32 noundef 3, i32 noundef %880) #11
  %1083 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1000, <4 x i32> noundef %877, <4 x float> noundef %1081, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1011, i32 noundef 3, i32 noundef %880) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1084 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1028, i32 %87, i32 0)
  %1085 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1030, i32 %87, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1086 = or disjoint i32 %193, 2304
  %1087 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %1086, i32 %95, i32 0)
  %1088 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %1086, i32 %100, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1089 = load <4 x i32>, ptr addrspace(3) %216, align 16, !tbaa !11
  %1090 = load <4 x i32>, ptr addrspace(3) %219, align 16, !tbaa !11
  %1091 = load <4 x i32>, ptr addrspace(3) %222, align 16, !tbaa !11
  %1092 = load <4 x i32>, ptr addrspace(3) %225, align 16, !tbaa !11
  %1093 = load <4 x i32>, ptr addrspace(3) %228, align 16, !tbaa !11
  %1094 = load <4 x i32>, ptr addrspace(3) %231, align 16, !tbaa !11
  %1095 = load <4 x i32>, ptr addrspace(3) %234, align 16, !tbaa !11
  %1096 = load <4 x i32>, ptr addrspace(3) %237, align 16, !tbaa !11
  %1097 = load <4 x i32>, ptr addrspace(3) %241, align 16, !tbaa !11
  %1098 = load <4 x i32>, ptr addrspace(3) %243, align 16, !tbaa !11
  %1099 = load <4 x i32>, ptr addrspace(3) %245, align 16, !tbaa !11
  %1100 = load <4 x i32>, ptr addrspace(3) %247, align 16, !tbaa !11
  %1101 = load <4 x i32>, ptr addrspace(3) %249, align 16, !tbaa !11
  %1102 = load <4 x i32>, ptr addrspace(3) %251, align 16, !tbaa !11
  %1103 = load <4 x i32>, ptr addrspace(3) %253, align 16, !tbaa !11
  %1104 = load <4 x i32>, ptr addrspace(3) %255, align 16, !tbaa !11
  %1105 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %982
  %1106 = load i32, ptr addrspace(3) %1105, align 4, !tbaa !7
  %1107 = or disjoint i32 %193, 9216
  %1108 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1107
  %1109 = load i32, ptr addrspace(3) %1108, align 4, !tbaa !7
  %1110 = or disjoint i32 %193, 16384
  %1111 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1110
  %1112 = load i32, ptr addrspace(3) %1111, align 4, !tbaa !7
  %1113 = or disjoint i32 %193, 23552
  %1114 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1113
  %1115 = load i32, ptr addrspace(3) %1114, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef %163, i32 noundef 16, i32 noundef %162, i32 noundef 1280, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %168, i32 noundef 16, i32 noundef %167, i32 noundef 1280, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %172, i32 noundef 16, i32 noundef %171, i32 noundef 1280, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %176, i32 noundef 16, i32 noundef %175, i32 noundef 1280, i32 noundef 0, i32 noundef 0) #11
  %1116 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1089, <4 x i32> noundef %925, <4 x float> noundef %1014, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1106, i32 noundef 0, i32 noundef %983) #11
  %1117 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1090, <4 x i32> noundef %925, <4 x float> noundef %1015, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1106, i32 noundef 0, i32 noundef %983) #11
  %1118 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1097, <4 x i32> noundef %927, <4 x float> noundef %1116, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1106, i32 noundef 2, i32 noundef %983) #11
  %1119 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1098, <4 x i32> noundef %927, <4 x float> noundef %1117, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1106, i32 noundef 2, i32 noundef %983) #11
  %1120 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1091, <4 x i32> noundef %925, <4 x float> noundef %1018, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1109, i32 noundef 0, i32 noundef %983) #11
  %1121 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1092, <4 x i32> noundef %925, <4 x float> noundef %1019, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1109, i32 noundef 0, i32 noundef %983) #11
  %1122 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1099, <4 x i32> noundef %927, <4 x float> noundef %1120, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1109, i32 noundef 2, i32 noundef %983) #11
  %1123 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1100, <4 x i32> noundef %927, <4 x float> noundef %1121, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1109, i32 noundef 2, i32 noundef %983) #11
  %1124 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1093, <4 x i32> noundef %925, <4 x float> noundef %1022, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1112, i32 noundef 0, i32 noundef %983) #11
  %1125 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1094, <4 x i32> noundef %925, <4 x float> noundef %1023, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1112, i32 noundef 0, i32 noundef %983) #11
  %1126 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1101, <4 x i32> noundef %927, <4 x float> noundef %1124, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1112, i32 noundef 2, i32 noundef %983) #11
  %1127 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1102, <4 x i32> noundef %927, <4 x float> noundef %1125, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1112, i32 noundef 2, i32 noundef %983) #11
  %1128 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1095, <4 x i32> noundef %925, <4 x float> noundef %1026, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1115, i32 noundef 0, i32 noundef %983) #11
  %1129 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1096, <4 x i32> noundef %925, <4 x float> noundef %1027, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1115, i32 noundef 0, i32 noundef %983) #11
  %1130 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1103, <4 x i32> noundef %927, <4 x float> noundef %1128, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1115, i32 noundef 2, i32 noundef %983) #11
  %1131 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1104, <4 x i32> noundef %927, <4 x float> noundef %1129, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1115, i32 noundef 2, i32 noundef %983) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1132 = or disjoint i32 %181, 20480
  %1133 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1132, i32 %75, i32 0)
  %1134 = or disjoint i32 %181, 21504
  %1135 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1134, i32 %75, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1136 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1089, <4 x i32> noundef %944, <4 x float> noundef %1034, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1106, i32 noundef 1, i32 noundef %983) #11
  %1137 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1090, <4 x i32> noundef %944, <4 x float> noundef %1035, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1106, i32 noundef 1, i32 noundef %983) #11
  %1138 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1097, <4 x i32> noundef %945, <4 x float> noundef %1136, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1106, i32 noundef 3, i32 noundef %983) #11
  %1139 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1098, <4 x i32> noundef %945, <4 x float> noundef %1137, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1106, i32 noundef 3, i32 noundef %983) #11
  %1140 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1091, <4 x i32> noundef %944, <4 x float> noundef %1038, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1109, i32 noundef 1, i32 noundef %983) #11
  %1141 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1092, <4 x i32> noundef %944, <4 x float> noundef %1039, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1109, i32 noundef 1, i32 noundef %983) #11
  %1142 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1099, <4 x i32> noundef %945, <4 x float> noundef %1140, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1109, i32 noundef 3, i32 noundef %983) #11
  %1143 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1100, <4 x i32> noundef %945, <4 x float> noundef %1141, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1109, i32 noundef 3, i32 noundef %983) #11
  %1144 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1093, <4 x i32> noundef %944, <4 x float> noundef %1042, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1112, i32 noundef 1, i32 noundef %983) #11
  %1145 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1094, <4 x i32> noundef %944, <4 x float> noundef %1043, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1112, i32 noundef 1, i32 noundef %983) #11
  %1146 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1101, <4 x i32> noundef %945, <4 x float> noundef %1144, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1112, i32 noundef 3, i32 noundef %983) #11
  %1147 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1102, <4 x i32> noundef %945, <4 x float> noundef %1145, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1112, i32 noundef 3, i32 noundef %983) #11
  %1148 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1095, <4 x i32> noundef %944, <4 x float> noundef %1046, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1115, i32 noundef 1, i32 noundef %983) #11
  %1149 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1096, <4 x i32> noundef %944, <4 x float> noundef %1047, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1115, i32 noundef 1, i32 noundef %983) #11
  %1150 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1103, <4 x i32> noundef %945, <4 x float> noundef %1148, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1115, i32 noundef 3, i32 noundef %983) #11
  %1151 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1104, <4 x i32> noundef %945, <4 x float> noundef %1149, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1115, i32 noundef 3, i32 noundef %983) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1152 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1132, i32 %79, i32 0)
  %1153 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1134, i32 %79, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1154 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1089, <4 x i32> noundef %962, <4 x float> noundef %1052, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1106, i32 noundef 0, i32 noundef %984) #11
  %1155 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1090, <4 x i32> noundef %962, <4 x float> noundef %1053, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1106, i32 noundef 0, i32 noundef %984) #11
  %1156 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1097, <4 x i32> noundef %963, <4 x float> noundef %1154, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1106, i32 noundef 2, i32 noundef %984) #11
  %1157 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1098, <4 x i32> noundef %963, <4 x float> noundef %1155, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1106, i32 noundef 2, i32 noundef %984) #11
  %1158 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1091, <4 x i32> noundef %962, <4 x float> noundef %1056, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1109, i32 noundef 0, i32 noundef %984) #11
  %1159 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1092, <4 x i32> noundef %962, <4 x float> noundef %1057, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1109, i32 noundef 0, i32 noundef %984) #11
  %1160 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1099, <4 x i32> noundef %963, <4 x float> noundef %1158, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1109, i32 noundef 2, i32 noundef %984) #11
  %1161 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1100, <4 x i32> noundef %963, <4 x float> noundef %1159, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1109, i32 noundef 2, i32 noundef %984) #11
  %1162 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1093, <4 x i32> noundef %962, <4 x float> noundef %1060, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1112, i32 noundef 0, i32 noundef %984) #11
  %1163 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1094, <4 x i32> noundef %962, <4 x float> noundef %1061, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1112, i32 noundef 0, i32 noundef %984) #11
  %1164 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1101, <4 x i32> noundef %963, <4 x float> noundef %1162, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1112, i32 noundef 2, i32 noundef %984) #11
  %1165 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1102, <4 x i32> noundef %963, <4 x float> noundef %1163, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1112, i32 noundef 2, i32 noundef %984) #11
  %1166 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1095, <4 x i32> noundef %962, <4 x float> noundef %1064, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1115, i32 noundef 0, i32 noundef %984) #11
  %1167 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1096, <4 x i32> noundef %962, <4 x float> noundef %1065, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1115, i32 noundef 0, i32 noundef %984) #11
  %1168 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1103, <4 x i32> noundef %963, <4 x float> noundef %1166, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1115, i32 noundef 2, i32 noundef %984) #11
  %1169 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1104, <4 x i32> noundef %963, <4 x float> noundef %1167, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1115, i32 noundef 2, i32 noundef %984) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1170 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1132, i32 %83, i32 0)
  %1171 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1134, i32 %83, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1172 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1089, <4 x i32> noundef %980, <4 x float> noundef %1070, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1106, i32 noundef 1, i32 noundef %984) #11
  %1173 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1090, <4 x i32> noundef %980, <4 x float> noundef %1071, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1106, i32 noundef 1, i32 noundef %984) #11
  %1174 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1097, <4 x i32> noundef %981, <4 x float> noundef %1172, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1106, i32 noundef 3, i32 noundef %984) #11
  %1175 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1098, <4 x i32> noundef %981, <4 x float> noundef %1173, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1106, i32 noundef 3, i32 noundef %984) #11
  %1176 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1091, <4 x i32> noundef %980, <4 x float> noundef %1074, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1109, i32 noundef 1, i32 noundef %984) #11
  %1177 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1092, <4 x i32> noundef %980, <4 x float> noundef %1075, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1109, i32 noundef 1, i32 noundef %984) #11
  %1178 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1099, <4 x i32> noundef %981, <4 x float> noundef %1176, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1109, i32 noundef 3, i32 noundef %984) #11
  %1179 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1100, <4 x i32> noundef %981, <4 x float> noundef %1177, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1109, i32 noundef 3, i32 noundef %984) #11
  %1180 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1093, <4 x i32> noundef %980, <4 x float> noundef %1078, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1112, i32 noundef 1, i32 noundef %984) #11
  %1181 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1094, <4 x i32> noundef %980, <4 x float> noundef %1079, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1112, i32 noundef 1, i32 noundef %984) #11
  %1182 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1101, <4 x i32> noundef %981, <4 x float> noundef %1180, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1112, i32 noundef 3, i32 noundef %984) #11
  %1183 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1102, <4 x i32> noundef %981, <4 x float> noundef %1181, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1112, i32 noundef 3, i32 noundef %984) #11
  %1184 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1095, <4 x i32> noundef %980, <4 x float> noundef %1082, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1115, i32 noundef 1, i32 noundef %984) #11
  %1185 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1096, <4 x i32> noundef %980, <4 x float> noundef %1083, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1115, i32 noundef 1, i32 noundef %984) #11
  %1186 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1103, <4 x i32> noundef %981, <4 x float> noundef %1184, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1115, i32 noundef 3, i32 noundef %984) #11
  %1187 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1104, <4 x i32> noundef %981, <4 x float> noundef %1185, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1115, i32 noundef 3, i32 noundef %984) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1188 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1132, i32 %87, i32 0)
  %1189 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1134, i32 %87, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1190 = or disjoint i32 %193, 2560
  %1191 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %1190, i32 %95, i32 0)
  %1192 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %1190, i32 %100, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1193 = load <4 x i32>, ptr addrspace(3) %345, align 16, !tbaa !11
  %1194 = load <4 x i32>, ptr addrspace(3) %347, align 16, !tbaa !11
  %1195 = load <4 x i32>, ptr addrspace(3) %349, align 16, !tbaa !11
  %1196 = load <4 x i32>, ptr addrspace(3) %351, align 16, !tbaa !11
  %1197 = load <4 x i32>, ptr addrspace(3) %353, align 16, !tbaa !11
  %1198 = load <4 x i32>, ptr addrspace(3) %355, align 16, !tbaa !11
  %1199 = load <4 x i32>, ptr addrspace(3) %357, align 16, !tbaa !11
  %1200 = load <4 x i32>, ptr addrspace(3) %359, align 16, !tbaa !11
  %1201 = load <4 x i32>, ptr addrspace(3) %361, align 16, !tbaa !11
  %1202 = load <4 x i32>, ptr addrspace(3) %363, align 16, !tbaa !11
  %1203 = load <4 x i32>, ptr addrspace(3) %365, align 16, !tbaa !11
  %1204 = load <4 x i32>, ptr addrspace(3) %367, align 16, !tbaa !11
  %1205 = load <4 x i32>, ptr addrspace(3) %369, align 16, !tbaa !11
  %1206 = load <4 x i32>, ptr addrspace(3) %371, align 16, !tbaa !11
  %1207 = load <4 x i32>, ptr addrspace(3) %373, align 16, !tbaa !11
  %1208 = load <4 x i32>, ptr addrspace(3) %375, align 16, !tbaa !11
  %1209 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1086
  %1210 = load i32, ptr addrspace(3) %1209, align 4, !tbaa !7
  %1211 = or disjoint i32 %193, 9472
  %1212 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1211
  %1213 = load i32, ptr addrspace(3) %1212, align 4, !tbaa !7
  %1214 = or disjoint i32 %193, 16640
  %1215 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1214
  %1216 = load i32, ptr addrspace(3) %1215, align 4, !tbaa !7
  %1217 = or disjoint i32 %193, 23808
  %1218 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1217
  %1219 = load i32, ptr addrspace(3) %1218, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %196, i32 noundef 16, i32 noundef %162, i32 noundef 1408, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %197, i32 noundef 16, i32 noundef %167, i32 noundef 1408, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %198, i32 noundef 16, i32 noundef %171, i32 noundef 1408, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %199, i32 noundef 16, i32 noundef %175, i32 noundef 1408, i32 noundef 0, i32 noundef 0) #11
  %1220 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1193, <4 x i32> noundef %1029, <4 x float> noundef %1118, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1210, i32 noundef 0, i32 noundef %1087) #11
  %1221 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1194, <4 x i32> noundef %1029, <4 x float> noundef %1119, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1210, i32 noundef 0, i32 noundef %1087) #11
  %1222 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1201, <4 x i32> noundef %1031, <4 x float> noundef %1220, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1210, i32 noundef 2, i32 noundef %1087) #11
  %1223 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1202, <4 x i32> noundef %1031, <4 x float> noundef %1221, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1210, i32 noundef 2, i32 noundef %1087) #11
  %1224 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1195, <4 x i32> noundef %1029, <4 x float> noundef %1122, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1213, i32 noundef 0, i32 noundef %1087) #11
  %1225 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1196, <4 x i32> noundef %1029, <4 x float> noundef %1123, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1213, i32 noundef 0, i32 noundef %1087) #11
  %1226 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1203, <4 x i32> noundef %1031, <4 x float> noundef %1224, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1213, i32 noundef 2, i32 noundef %1087) #11
  %1227 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1204, <4 x i32> noundef %1031, <4 x float> noundef %1225, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1213, i32 noundef 2, i32 noundef %1087) #11
  %1228 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1197, <4 x i32> noundef %1029, <4 x float> noundef %1126, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1216, i32 noundef 0, i32 noundef %1087) #11
  %1229 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1198, <4 x i32> noundef %1029, <4 x float> noundef %1127, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1216, i32 noundef 0, i32 noundef %1087) #11
  %1230 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1205, <4 x i32> noundef %1031, <4 x float> noundef %1228, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1216, i32 noundef 2, i32 noundef %1087) #11
  %1231 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1206, <4 x i32> noundef %1031, <4 x float> noundef %1229, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1216, i32 noundef 2, i32 noundef %1087) #11
  %1232 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1199, <4 x i32> noundef %1029, <4 x float> noundef %1130, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1219, i32 noundef 0, i32 noundef %1087) #11
  %1233 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1200, <4 x i32> noundef %1029, <4 x float> noundef %1131, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1219, i32 noundef 0, i32 noundef %1087) #11
  %1234 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1207, <4 x i32> noundef %1031, <4 x float> noundef %1232, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1219, i32 noundef 2, i32 noundef %1087) #11
  %1235 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1208, <4 x i32> noundef %1031, <4 x float> noundef %1233, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1219, i32 noundef 2, i32 noundef %1087) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1236 = or disjoint i32 %181, 22528
  %1237 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1236, i32 %75, i32 0)
  %1238 = or disjoint i32 %181, 23552
  %1239 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1238, i32 %75, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1240 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1193, <4 x i32> noundef %1048, <4 x float> noundef %1138, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1210, i32 noundef 1, i32 noundef %1087) #11
  %1241 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1194, <4 x i32> noundef %1048, <4 x float> noundef %1139, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1210, i32 noundef 1, i32 noundef %1087) #11
  %1242 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1201, <4 x i32> noundef %1049, <4 x float> noundef %1240, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1210, i32 noundef 3, i32 noundef %1087) #11
  %1243 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1202, <4 x i32> noundef %1049, <4 x float> noundef %1241, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1210, i32 noundef 3, i32 noundef %1087) #11
  %1244 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1195, <4 x i32> noundef %1048, <4 x float> noundef %1142, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1213, i32 noundef 1, i32 noundef %1087) #11
  %1245 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1196, <4 x i32> noundef %1048, <4 x float> noundef %1143, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1213, i32 noundef 1, i32 noundef %1087) #11
  %1246 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1203, <4 x i32> noundef %1049, <4 x float> noundef %1244, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1213, i32 noundef 3, i32 noundef %1087) #11
  %1247 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1204, <4 x i32> noundef %1049, <4 x float> noundef %1245, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1213, i32 noundef 3, i32 noundef %1087) #11
  %1248 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1197, <4 x i32> noundef %1048, <4 x float> noundef %1146, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1216, i32 noundef 1, i32 noundef %1087) #11
  %1249 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1198, <4 x i32> noundef %1048, <4 x float> noundef %1147, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1216, i32 noundef 1, i32 noundef %1087) #11
  %1250 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1205, <4 x i32> noundef %1049, <4 x float> noundef %1248, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1216, i32 noundef 3, i32 noundef %1087) #11
  %1251 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1206, <4 x i32> noundef %1049, <4 x float> noundef %1249, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1216, i32 noundef 3, i32 noundef %1087) #11
  %1252 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1199, <4 x i32> noundef %1048, <4 x float> noundef %1150, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1219, i32 noundef 1, i32 noundef %1087) #11
  %1253 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1200, <4 x i32> noundef %1048, <4 x float> noundef %1151, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1219, i32 noundef 1, i32 noundef %1087) #11
  %1254 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1207, <4 x i32> noundef %1049, <4 x float> noundef %1252, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1219, i32 noundef 3, i32 noundef %1087) #11
  %1255 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1208, <4 x i32> noundef %1049, <4 x float> noundef %1253, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1219, i32 noundef 3, i32 noundef %1087) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1256 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1236, i32 %79, i32 0)
  %1257 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1238, i32 %79, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1258 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1193, <4 x i32> noundef %1066, <4 x float> noundef %1156, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1210, i32 noundef 0, i32 noundef %1088) #11
  %1259 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1194, <4 x i32> noundef %1066, <4 x float> noundef %1157, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1210, i32 noundef 0, i32 noundef %1088) #11
  %1260 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1201, <4 x i32> noundef %1067, <4 x float> noundef %1258, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1210, i32 noundef 2, i32 noundef %1088) #11
  %1261 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1202, <4 x i32> noundef %1067, <4 x float> noundef %1259, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1210, i32 noundef 2, i32 noundef %1088) #11
  %1262 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1195, <4 x i32> noundef %1066, <4 x float> noundef %1160, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1213, i32 noundef 0, i32 noundef %1088) #11
  %1263 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1196, <4 x i32> noundef %1066, <4 x float> noundef %1161, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1213, i32 noundef 0, i32 noundef %1088) #11
  %1264 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1203, <4 x i32> noundef %1067, <4 x float> noundef %1262, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1213, i32 noundef 2, i32 noundef %1088) #11
  %1265 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1204, <4 x i32> noundef %1067, <4 x float> noundef %1263, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1213, i32 noundef 2, i32 noundef %1088) #11
  %1266 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1197, <4 x i32> noundef %1066, <4 x float> noundef %1164, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1216, i32 noundef 0, i32 noundef %1088) #11
  %1267 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1198, <4 x i32> noundef %1066, <4 x float> noundef %1165, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1216, i32 noundef 0, i32 noundef %1088) #11
  %1268 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1205, <4 x i32> noundef %1067, <4 x float> noundef %1266, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1216, i32 noundef 2, i32 noundef %1088) #11
  %1269 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1206, <4 x i32> noundef %1067, <4 x float> noundef %1267, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1216, i32 noundef 2, i32 noundef %1088) #11
  %1270 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1199, <4 x i32> noundef %1066, <4 x float> noundef %1168, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1219, i32 noundef 0, i32 noundef %1088) #11
  %1271 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1200, <4 x i32> noundef %1066, <4 x float> noundef %1169, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1219, i32 noundef 0, i32 noundef %1088) #11
  %1272 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1207, <4 x i32> noundef %1067, <4 x float> noundef %1270, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1219, i32 noundef 2, i32 noundef %1088) #11
  %1273 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1208, <4 x i32> noundef %1067, <4 x float> noundef %1271, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1219, i32 noundef 2, i32 noundef %1088) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1274 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1236, i32 %83, i32 0)
  %1275 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1238, i32 %83, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1276 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1193, <4 x i32> noundef %1084, <4 x float> noundef %1174, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1210, i32 noundef 1, i32 noundef %1088) #11
  %1277 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1194, <4 x i32> noundef %1084, <4 x float> noundef %1175, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1210, i32 noundef 1, i32 noundef %1088) #11
  %1278 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1201, <4 x i32> noundef %1085, <4 x float> noundef %1276, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1210, i32 noundef 3, i32 noundef %1088) #11
  %1279 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1202, <4 x i32> noundef %1085, <4 x float> noundef %1277, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1210, i32 noundef 3, i32 noundef %1088) #11
  %1280 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1195, <4 x i32> noundef %1084, <4 x float> noundef %1178, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1213, i32 noundef 1, i32 noundef %1088) #11
  %1281 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1196, <4 x i32> noundef %1084, <4 x float> noundef %1179, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1213, i32 noundef 1, i32 noundef %1088) #11
  %1282 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1203, <4 x i32> noundef %1085, <4 x float> noundef %1280, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1213, i32 noundef 3, i32 noundef %1088) #11
  %1283 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1204, <4 x i32> noundef %1085, <4 x float> noundef %1281, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1213, i32 noundef 3, i32 noundef %1088) #11
  %1284 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1197, <4 x i32> noundef %1084, <4 x float> noundef %1182, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1216, i32 noundef 1, i32 noundef %1088) #11
  %1285 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1198, <4 x i32> noundef %1084, <4 x float> noundef %1183, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1216, i32 noundef 1, i32 noundef %1088) #11
  %1286 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1205, <4 x i32> noundef %1085, <4 x float> noundef %1284, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1216, i32 noundef 3, i32 noundef %1088) #11
  %1287 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1206, <4 x i32> noundef %1085, <4 x float> noundef %1285, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1216, i32 noundef 3, i32 noundef %1088) #11
  %1288 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1199, <4 x i32> noundef %1084, <4 x float> noundef %1186, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1219, i32 noundef 1, i32 noundef %1088) #11
  %1289 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1200, <4 x i32> noundef %1084, <4 x float> noundef %1187, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1219, i32 noundef 1, i32 noundef %1088) #11
  %1290 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1207, <4 x i32> noundef %1085, <4 x float> noundef %1288, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1219, i32 noundef 3, i32 noundef %1088) #11
  %1291 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1208, <4 x i32> noundef %1085, <4 x float> noundef %1289, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1219, i32 noundef 3, i32 noundef %1088) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1292 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1236, i32 %87, i32 0)
  %1293 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1238, i32 %87, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1294 = or disjoint i32 %193, 2816
  %1295 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %1294, i32 %95, i32 0)
  %1296 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %1294, i32 %100, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1297 = load <4 x i32>, ptr addrspace(3) %216, align 16, !tbaa !11
  %1298 = load <4 x i32>, ptr addrspace(3) %219, align 16, !tbaa !11
  %1299 = load <4 x i32>, ptr addrspace(3) %222, align 16, !tbaa !11
  %1300 = load <4 x i32>, ptr addrspace(3) %225, align 16, !tbaa !11
  %1301 = load <4 x i32>, ptr addrspace(3) %228, align 16, !tbaa !11
  %1302 = load <4 x i32>, ptr addrspace(3) %231, align 16, !tbaa !11
  %1303 = load <4 x i32>, ptr addrspace(3) %234, align 16, !tbaa !11
  %1304 = load <4 x i32>, ptr addrspace(3) %237, align 16, !tbaa !11
  %1305 = load <4 x i32>, ptr addrspace(3) %241, align 16, !tbaa !11
  %1306 = load <4 x i32>, ptr addrspace(3) %243, align 16, !tbaa !11
  %1307 = load <4 x i32>, ptr addrspace(3) %245, align 16, !tbaa !11
  %1308 = load <4 x i32>, ptr addrspace(3) %247, align 16, !tbaa !11
  %1309 = load <4 x i32>, ptr addrspace(3) %249, align 16, !tbaa !11
  %1310 = load <4 x i32>, ptr addrspace(3) %251, align 16, !tbaa !11
  %1311 = load <4 x i32>, ptr addrspace(3) %253, align 16, !tbaa !11
  %1312 = load <4 x i32>, ptr addrspace(3) %255, align 16, !tbaa !11
  %1313 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1190
  %1314 = load i32, ptr addrspace(3) %1313, align 4, !tbaa !7
  %1315 = or disjoint i32 %193, 9728
  %1316 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1315
  %1317 = load i32, ptr addrspace(3) %1316, align 4, !tbaa !7
  %1318 = or disjoint i32 %193, 16896
  %1319 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1318
  %1320 = load i32, ptr addrspace(3) %1319, align 4, !tbaa !7
  %1321 = or disjoint i32 %193, 24064
  %1322 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1321
  %1323 = load i32, ptr addrspace(3) %1322, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef %163, i32 noundef 16, i32 noundef %162, i32 noundef 1536, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %168, i32 noundef 16, i32 noundef %167, i32 noundef 1536, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %172, i32 noundef 16, i32 noundef %171, i32 noundef 1536, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %176, i32 noundef 16, i32 noundef %175, i32 noundef 1536, i32 noundef 0, i32 noundef 0) #11
  %1324 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1297, <4 x i32> noundef %1133, <4 x float> noundef %1222, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1314, i32 noundef 0, i32 noundef %1191) #11
  %1325 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1298, <4 x i32> noundef %1133, <4 x float> noundef %1223, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1314, i32 noundef 0, i32 noundef %1191) #11
  %1326 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1305, <4 x i32> noundef %1135, <4 x float> noundef %1324, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1314, i32 noundef 2, i32 noundef %1191) #11
  %1327 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1306, <4 x i32> noundef %1135, <4 x float> noundef %1325, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1314, i32 noundef 2, i32 noundef %1191) #11
  %1328 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1299, <4 x i32> noundef %1133, <4 x float> noundef %1226, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1317, i32 noundef 0, i32 noundef %1191) #11
  %1329 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1300, <4 x i32> noundef %1133, <4 x float> noundef %1227, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1317, i32 noundef 0, i32 noundef %1191) #11
  %1330 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1307, <4 x i32> noundef %1135, <4 x float> noundef %1328, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1317, i32 noundef 2, i32 noundef %1191) #11
  %1331 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1308, <4 x i32> noundef %1135, <4 x float> noundef %1329, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1317, i32 noundef 2, i32 noundef %1191) #11
  %1332 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1301, <4 x i32> noundef %1133, <4 x float> noundef %1230, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1320, i32 noundef 0, i32 noundef %1191) #11
  %1333 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1302, <4 x i32> noundef %1133, <4 x float> noundef %1231, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1320, i32 noundef 0, i32 noundef %1191) #11
  %1334 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1309, <4 x i32> noundef %1135, <4 x float> noundef %1332, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1320, i32 noundef 2, i32 noundef %1191) #11
  %1335 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1310, <4 x i32> noundef %1135, <4 x float> noundef %1333, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1320, i32 noundef 2, i32 noundef %1191) #11
  %1336 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1303, <4 x i32> noundef %1133, <4 x float> noundef %1234, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1323, i32 noundef 0, i32 noundef %1191) #11
  %1337 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1304, <4 x i32> noundef %1133, <4 x float> noundef %1235, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1323, i32 noundef 0, i32 noundef %1191) #11
  %1338 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1311, <4 x i32> noundef %1135, <4 x float> noundef %1336, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1323, i32 noundef 2, i32 noundef %1191) #11
  %1339 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1312, <4 x i32> noundef %1135, <4 x float> noundef %1337, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1323, i32 noundef 2, i32 noundef %1191) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1340 = or disjoint i32 %181, 24576
  %1341 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1340, i32 %75, i32 0)
  %1342 = or disjoint i32 %181, 25600
  %1343 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1342, i32 %75, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1344 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1297, <4 x i32> noundef %1152, <4 x float> noundef %1242, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1314, i32 noundef 1, i32 noundef %1191) #11
  %1345 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1298, <4 x i32> noundef %1152, <4 x float> noundef %1243, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1314, i32 noundef 1, i32 noundef %1191) #11
  %1346 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1305, <4 x i32> noundef %1153, <4 x float> noundef %1344, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1314, i32 noundef 3, i32 noundef %1191) #11
  %1347 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1306, <4 x i32> noundef %1153, <4 x float> noundef %1345, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1314, i32 noundef 3, i32 noundef %1191) #11
  %1348 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1299, <4 x i32> noundef %1152, <4 x float> noundef %1246, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1317, i32 noundef 1, i32 noundef %1191) #11
  %1349 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1300, <4 x i32> noundef %1152, <4 x float> noundef %1247, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1317, i32 noundef 1, i32 noundef %1191) #11
  %1350 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1307, <4 x i32> noundef %1153, <4 x float> noundef %1348, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1317, i32 noundef 3, i32 noundef %1191) #11
  %1351 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1308, <4 x i32> noundef %1153, <4 x float> noundef %1349, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1317, i32 noundef 3, i32 noundef %1191) #11
  %1352 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1301, <4 x i32> noundef %1152, <4 x float> noundef %1250, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1320, i32 noundef 1, i32 noundef %1191) #11
  %1353 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1302, <4 x i32> noundef %1152, <4 x float> noundef %1251, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1320, i32 noundef 1, i32 noundef %1191) #11
  %1354 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1309, <4 x i32> noundef %1153, <4 x float> noundef %1352, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1320, i32 noundef 3, i32 noundef %1191) #11
  %1355 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1310, <4 x i32> noundef %1153, <4 x float> noundef %1353, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1320, i32 noundef 3, i32 noundef %1191) #11
  %1356 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1303, <4 x i32> noundef %1152, <4 x float> noundef %1254, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1323, i32 noundef 1, i32 noundef %1191) #11
  %1357 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1304, <4 x i32> noundef %1152, <4 x float> noundef %1255, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1323, i32 noundef 1, i32 noundef %1191) #11
  %1358 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1311, <4 x i32> noundef %1153, <4 x float> noundef %1356, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1323, i32 noundef 3, i32 noundef %1191) #11
  %1359 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1312, <4 x i32> noundef %1153, <4 x float> noundef %1357, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1323, i32 noundef 3, i32 noundef %1191) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1360 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1340, i32 %79, i32 0)
  %1361 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1342, i32 %79, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1362 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1297, <4 x i32> noundef %1170, <4 x float> noundef %1260, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1314, i32 noundef 0, i32 noundef %1192) #11
  %1363 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1298, <4 x i32> noundef %1170, <4 x float> noundef %1261, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1314, i32 noundef 0, i32 noundef %1192) #11
  %1364 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1305, <4 x i32> noundef %1171, <4 x float> noundef %1362, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1314, i32 noundef 2, i32 noundef %1192) #11
  %1365 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1306, <4 x i32> noundef %1171, <4 x float> noundef %1363, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1314, i32 noundef 2, i32 noundef %1192) #11
  %1366 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1299, <4 x i32> noundef %1170, <4 x float> noundef %1264, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1317, i32 noundef 0, i32 noundef %1192) #11
  %1367 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1300, <4 x i32> noundef %1170, <4 x float> noundef %1265, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1317, i32 noundef 0, i32 noundef %1192) #11
  %1368 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1307, <4 x i32> noundef %1171, <4 x float> noundef %1366, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1317, i32 noundef 2, i32 noundef %1192) #11
  %1369 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1308, <4 x i32> noundef %1171, <4 x float> noundef %1367, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1317, i32 noundef 2, i32 noundef %1192) #11
  %1370 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1301, <4 x i32> noundef %1170, <4 x float> noundef %1268, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1320, i32 noundef 0, i32 noundef %1192) #11
  %1371 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1302, <4 x i32> noundef %1170, <4 x float> noundef %1269, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1320, i32 noundef 0, i32 noundef %1192) #11
  %1372 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1309, <4 x i32> noundef %1171, <4 x float> noundef %1370, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1320, i32 noundef 2, i32 noundef %1192) #11
  %1373 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1310, <4 x i32> noundef %1171, <4 x float> noundef %1371, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1320, i32 noundef 2, i32 noundef %1192) #11
  %1374 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1303, <4 x i32> noundef %1170, <4 x float> noundef %1272, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1323, i32 noundef 0, i32 noundef %1192) #11
  %1375 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1304, <4 x i32> noundef %1170, <4 x float> noundef %1273, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1323, i32 noundef 0, i32 noundef %1192) #11
  %1376 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1311, <4 x i32> noundef %1171, <4 x float> noundef %1374, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1323, i32 noundef 2, i32 noundef %1192) #11
  %1377 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1312, <4 x i32> noundef %1171, <4 x float> noundef %1375, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1323, i32 noundef 2, i32 noundef %1192) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1378 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1340, i32 %83, i32 0)
  %1379 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1342, i32 %83, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1380 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1297, <4 x i32> noundef %1188, <4 x float> noundef %1278, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1314, i32 noundef 1, i32 noundef %1192) #11
  %1381 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1298, <4 x i32> noundef %1188, <4 x float> noundef %1279, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1314, i32 noundef 1, i32 noundef %1192) #11
  %1382 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1305, <4 x i32> noundef %1189, <4 x float> noundef %1380, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1314, i32 noundef 3, i32 noundef %1192) #11
  %1383 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1306, <4 x i32> noundef %1189, <4 x float> noundef %1381, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1314, i32 noundef 3, i32 noundef %1192) #11
  %1384 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1299, <4 x i32> noundef %1188, <4 x float> noundef %1282, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1317, i32 noundef 1, i32 noundef %1192) #11
  %1385 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1300, <4 x i32> noundef %1188, <4 x float> noundef %1283, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1317, i32 noundef 1, i32 noundef %1192) #11
  %1386 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1307, <4 x i32> noundef %1189, <4 x float> noundef %1384, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1317, i32 noundef 3, i32 noundef %1192) #11
  %1387 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1308, <4 x i32> noundef %1189, <4 x float> noundef %1385, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1317, i32 noundef 3, i32 noundef %1192) #11
  %1388 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1301, <4 x i32> noundef %1188, <4 x float> noundef %1286, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1320, i32 noundef 1, i32 noundef %1192) #11
  %1389 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1302, <4 x i32> noundef %1188, <4 x float> noundef %1287, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1320, i32 noundef 1, i32 noundef %1192) #11
  %1390 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1309, <4 x i32> noundef %1189, <4 x float> noundef %1388, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1320, i32 noundef 3, i32 noundef %1192) #11
  %1391 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1310, <4 x i32> noundef %1189, <4 x float> noundef %1389, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1320, i32 noundef 3, i32 noundef %1192) #11
  %1392 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1303, <4 x i32> noundef %1188, <4 x float> noundef %1290, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1323, i32 noundef 1, i32 noundef %1192) #11
  %1393 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1304, <4 x i32> noundef %1188, <4 x float> noundef %1291, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1323, i32 noundef 1, i32 noundef %1192) #11
  %1394 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1311, <4 x i32> noundef %1189, <4 x float> noundef %1392, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1323, i32 noundef 3, i32 noundef %1192) #11
  %1395 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1312, <4 x i32> noundef %1189, <4 x float> noundef %1393, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1323, i32 noundef 3, i32 noundef %1192) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1396 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1340, i32 %87, i32 0)
  %1397 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1342, i32 %87, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1398 = or disjoint i32 %193, 3072
  %1399 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %1398, i32 %95, i32 0)
  %1400 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %1398, i32 %100, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1401 = load <4 x i32>, ptr addrspace(3) %345, align 16, !tbaa !11
  %1402 = load <4 x i32>, ptr addrspace(3) %347, align 16, !tbaa !11
  %1403 = load <4 x i32>, ptr addrspace(3) %349, align 16, !tbaa !11
  %1404 = load <4 x i32>, ptr addrspace(3) %351, align 16, !tbaa !11
  %1405 = load <4 x i32>, ptr addrspace(3) %353, align 16, !tbaa !11
  %1406 = load <4 x i32>, ptr addrspace(3) %355, align 16, !tbaa !11
  %1407 = load <4 x i32>, ptr addrspace(3) %357, align 16, !tbaa !11
  %1408 = load <4 x i32>, ptr addrspace(3) %359, align 16, !tbaa !11
  %1409 = load <4 x i32>, ptr addrspace(3) %361, align 16, !tbaa !11
  %1410 = load <4 x i32>, ptr addrspace(3) %363, align 16, !tbaa !11
  %1411 = load <4 x i32>, ptr addrspace(3) %365, align 16, !tbaa !11
  %1412 = load <4 x i32>, ptr addrspace(3) %367, align 16, !tbaa !11
  %1413 = load <4 x i32>, ptr addrspace(3) %369, align 16, !tbaa !11
  %1414 = load <4 x i32>, ptr addrspace(3) %371, align 16, !tbaa !11
  %1415 = load <4 x i32>, ptr addrspace(3) %373, align 16, !tbaa !11
  %1416 = load <4 x i32>, ptr addrspace(3) %375, align 16, !tbaa !11
  %1417 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1294
  %1418 = load i32, ptr addrspace(3) %1417, align 4, !tbaa !7
  %1419 = or disjoint i32 %193, 9984
  %1420 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1419
  %1421 = load i32, ptr addrspace(3) %1420, align 4, !tbaa !7
  %1422 = or disjoint i32 %193, 17152
  %1423 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1422
  %1424 = load i32, ptr addrspace(3) %1423, align 4, !tbaa !7
  %1425 = or disjoint i32 %193, 24320
  %1426 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1425
  %1427 = load i32, ptr addrspace(3) %1426, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %196, i32 noundef 16, i32 noundef %162, i32 noundef 1664, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %197, i32 noundef 16, i32 noundef %167, i32 noundef 1664, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %198, i32 noundef 16, i32 noundef %171, i32 noundef 1664, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %199, i32 noundef 16, i32 noundef %175, i32 noundef 1664, i32 noundef 0, i32 noundef 0) #11
  %1428 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1401, <4 x i32> noundef %1237, <4 x float> noundef %1326, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1418, i32 noundef 0, i32 noundef %1295) #11
  %1429 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1402, <4 x i32> noundef %1237, <4 x float> noundef %1327, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1418, i32 noundef 0, i32 noundef %1295) #11
  %1430 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1409, <4 x i32> noundef %1239, <4 x float> noundef %1428, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1418, i32 noundef 2, i32 noundef %1295) #11
  %1431 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1410, <4 x i32> noundef %1239, <4 x float> noundef %1429, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1418, i32 noundef 2, i32 noundef %1295) #11
  %1432 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1403, <4 x i32> noundef %1237, <4 x float> noundef %1330, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1421, i32 noundef 0, i32 noundef %1295) #11
  %1433 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1404, <4 x i32> noundef %1237, <4 x float> noundef %1331, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1421, i32 noundef 0, i32 noundef %1295) #11
  %1434 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1411, <4 x i32> noundef %1239, <4 x float> noundef %1432, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1421, i32 noundef 2, i32 noundef %1295) #11
  %1435 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1412, <4 x i32> noundef %1239, <4 x float> noundef %1433, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1421, i32 noundef 2, i32 noundef %1295) #11
  %1436 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1405, <4 x i32> noundef %1237, <4 x float> noundef %1334, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1424, i32 noundef 0, i32 noundef %1295) #11
  %1437 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1406, <4 x i32> noundef %1237, <4 x float> noundef %1335, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1424, i32 noundef 0, i32 noundef %1295) #11
  %1438 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1413, <4 x i32> noundef %1239, <4 x float> noundef %1436, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1424, i32 noundef 2, i32 noundef %1295) #11
  %1439 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1414, <4 x i32> noundef %1239, <4 x float> noundef %1437, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1424, i32 noundef 2, i32 noundef %1295) #11
  %1440 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1407, <4 x i32> noundef %1237, <4 x float> noundef %1338, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1427, i32 noundef 0, i32 noundef %1295) #11
  %1441 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1408, <4 x i32> noundef %1237, <4 x float> noundef %1339, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1427, i32 noundef 0, i32 noundef %1295) #11
  %1442 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1415, <4 x i32> noundef %1239, <4 x float> noundef %1440, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1427, i32 noundef 2, i32 noundef %1295) #11
  %1443 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1416, <4 x i32> noundef %1239, <4 x float> noundef %1441, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1427, i32 noundef 2, i32 noundef %1295) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1444 = or disjoint i32 %181, 26624
  %1445 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1444, i32 %75, i32 0)
  %1446 = or disjoint i32 %181, 27648
  %1447 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1446, i32 %75, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1448 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1401, <4 x i32> noundef %1256, <4 x float> noundef %1346, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1418, i32 noundef 1, i32 noundef %1295) #11
  %1449 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1402, <4 x i32> noundef %1256, <4 x float> noundef %1347, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1418, i32 noundef 1, i32 noundef %1295) #11
  %1450 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1409, <4 x i32> noundef %1257, <4 x float> noundef %1448, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1418, i32 noundef 3, i32 noundef %1295) #11
  %1451 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1410, <4 x i32> noundef %1257, <4 x float> noundef %1449, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1418, i32 noundef 3, i32 noundef %1295) #11
  %1452 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1403, <4 x i32> noundef %1256, <4 x float> noundef %1350, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1421, i32 noundef 1, i32 noundef %1295) #11
  %1453 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1404, <4 x i32> noundef %1256, <4 x float> noundef %1351, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1421, i32 noundef 1, i32 noundef %1295) #11
  %1454 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1411, <4 x i32> noundef %1257, <4 x float> noundef %1452, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1421, i32 noundef 3, i32 noundef %1295) #11
  %1455 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1412, <4 x i32> noundef %1257, <4 x float> noundef %1453, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1421, i32 noundef 3, i32 noundef %1295) #11
  %1456 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1405, <4 x i32> noundef %1256, <4 x float> noundef %1354, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1424, i32 noundef 1, i32 noundef %1295) #11
  %1457 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1406, <4 x i32> noundef %1256, <4 x float> noundef %1355, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1424, i32 noundef 1, i32 noundef %1295) #11
  %1458 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1413, <4 x i32> noundef %1257, <4 x float> noundef %1456, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1424, i32 noundef 3, i32 noundef %1295) #11
  %1459 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1414, <4 x i32> noundef %1257, <4 x float> noundef %1457, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1424, i32 noundef 3, i32 noundef %1295) #11
  %1460 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1407, <4 x i32> noundef %1256, <4 x float> noundef %1358, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1427, i32 noundef 1, i32 noundef %1295) #11
  %1461 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1408, <4 x i32> noundef %1256, <4 x float> noundef %1359, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1427, i32 noundef 1, i32 noundef %1295) #11
  %1462 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1415, <4 x i32> noundef %1257, <4 x float> noundef %1460, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1427, i32 noundef 3, i32 noundef %1295) #11
  %1463 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1416, <4 x i32> noundef %1257, <4 x float> noundef %1461, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1427, i32 noundef 3, i32 noundef %1295) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1464 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1444, i32 %79, i32 0)
  %1465 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1446, i32 %79, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1466 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1401, <4 x i32> noundef %1274, <4 x float> noundef %1364, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1418, i32 noundef 0, i32 noundef %1296) #11
  %1467 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1402, <4 x i32> noundef %1274, <4 x float> noundef %1365, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1418, i32 noundef 0, i32 noundef %1296) #11
  %1468 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1409, <4 x i32> noundef %1275, <4 x float> noundef %1466, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1418, i32 noundef 2, i32 noundef %1296) #11
  %1469 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1410, <4 x i32> noundef %1275, <4 x float> noundef %1467, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1418, i32 noundef 2, i32 noundef %1296) #11
  %1470 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1403, <4 x i32> noundef %1274, <4 x float> noundef %1368, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1421, i32 noundef 0, i32 noundef %1296) #11
  %1471 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1404, <4 x i32> noundef %1274, <4 x float> noundef %1369, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1421, i32 noundef 0, i32 noundef %1296) #11
  %1472 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1411, <4 x i32> noundef %1275, <4 x float> noundef %1470, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1421, i32 noundef 2, i32 noundef %1296) #11
  %1473 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1412, <4 x i32> noundef %1275, <4 x float> noundef %1471, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1421, i32 noundef 2, i32 noundef %1296) #11
  %1474 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1405, <4 x i32> noundef %1274, <4 x float> noundef %1372, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1424, i32 noundef 0, i32 noundef %1296) #11
  %1475 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1406, <4 x i32> noundef %1274, <4 x float> noundef %1373, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1424, i32 noundef 0, i32 noundef %1296) #11
  %1476 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1413, <4 x i32> noundef %1275, <4 x float> noundef %1474, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1424, i32 noundef 2, i32 noundef %1296) #11
  %1477 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1414, <4 x i32> noundef %1275, <4 x float> noundef %1475, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1424, i32 noundef 2, i32 noundef %1296) #11
  %1478 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1407, <4 x i32> noundef %1274, <4 x float> noundef %1376, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1427, i32 noundef 0, i32 noundef %1296) #11
  %1479 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1408, <4 x i32> noundef %1274, <4 x float> noundef %1377, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1427, i32 noundef 0, i32 noundef %1296) #11
  %1480 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1415, <4 x i32> noundef %1275, <4 x float> noundef %1478, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1427, i32 noundef 2, i32 noundef %1296) #11
  %1481 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1416, <4 x i32> noundef %1275, <4 x float> noundef %1479, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1427, i32 noundef 2, i32 noundef %1296) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1482 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1444, i32 %83, i32 0)
  %1483 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1446, i32 %83, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1484 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1401, <4 x i32> noundef %1292, <4 x float> noundef %1382, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1418, i32 noundef 1, i32 noundef %1296) #11
  %1485 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1402, <4 x i32> noundef %1292, <4 x float> noundef %1383, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1418, i32 noundef 1, i32 noundef %1296) #11
  %1486 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1409, <4 x i32> noundef %1293, <4 x float> noundef %1484, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1418, i32 noundef 3, i32 noundef %1296) #11
  %1487 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1410, <4 x i32> noundef %1293, <4 x float> noundef %1485, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1418, i32 noundef 3, i32 noundef %1296) #11
  %1488 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1403, <4 x i32> noundef %1292, <4 x float> noundef %1386, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1421, i32 noundef 1, i32 noundef %1296) #11
  %1489 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1404, <4 x i32> noundef %1292, <4 x float> noundef %1387, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1421, i32 noundef 1, i32 noundef %1296) #11
  %1490 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1411, <4 x i32> noundef %1293, <4 x float> noundef %1488, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1421, i32 noundef 3, i32 noundef %1296) #11
  %1491 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1412, <4 x i32> noundef %1293, <4 x float> noundef %1489, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1421, i32 noundef 3, i32 noundef %1296) #11
  %1492 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1405, <4 x i32> noundef %1292, <4 x float> noundef %1390, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1424, i32 noundef 1, i32 noundef %1296) #11
  %1493 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1406, <4 x i32> noundef %1292, <4 x float> noundef %1391, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1424, i32 noundef 1, i32 noundef %1296) #11
  %1494 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1413, <4 x i32> noundef %1293, <4 x float> noundef %1492, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1424, i32 noundef 3, i32 noundef %1296) #11
  %1495 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1414, <4 x i32> noundef %1293, <4 x float> noundef %1493, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1424, i32 noundef 3, i32 noundef %1296) #11
  %1496 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1407, <4 x i32> noundef %1292, <4 x float> noundef %1394, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1427, i32 noundef 1, i32 noundef %1296) #11
  %1497 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1408, <4 x i32> noundef %1292, <4 x float> noundef %1395, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1427, i32 noundef 1, i32 noundef %1296) #11
  %1498 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1415, <4 x i32> noundef %1293, <4 x float> noundef %1496, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1427, i32 noundef 3, i32 noundef %1296) #11
  %1499 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1416, <4 x i32> noundef %1293, <4 x float> noundef %1497, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1427, i32 noundef 3, i32 noundef %1296) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1500 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1444, i32 %87, i32 0)
  %1501 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1446, i32 %87, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1502 = or disjoint i32 %193, 3328
  %1503 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %1502, i32 %95, i32 0)
  %1504 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %1502, i32 %100, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1505 = load <4 x i32>, ptr addrspace(3) %216, align 16, !tbaa !11
  %1506 = load <4 x i32>, ptr addrspace(3) %219, align 16, !tbaa !11
  %1507 = load <4 x i32>, ptr addrspace(3) %222, align 16, !tbaa !11
  %1508 = load <4 x i32>, ptr addrspace(3) %225, align 16, !tbaa !11
  %1509 = load <4 x i32>, ptr addrspace(3) %228, align 16, !tbaa !11
  %1510 = load <4 x i32>, ptr addrspace(3) %231, align 16, !tbaa !11
  %1511 = load <4 x i32>, ptr addrspace(3) %234, align 16, !tbaa !11
  %1512 = load <4 x i32>, ptr addrspace(3) %237, align 16, !tbaa !11
  %1513 = load <4 x i32>, ptr addrspace(3) %241, align 16, !tbaa !11
  %1514 = load <4 x i32>, ptr addrspace(3) %243, align 16, !tbaa !11
  %1515 = load <4 x i32>, ptr addrspace(3) %245, align 16, !tbaa !11
  %1516 = load <4 x i32>, ptr addrspace(3) %247, align 16, !tbaa !11
  %1517 = load <4 x i32>, ptr addrspace(3) %249, align 16, !tbaa !11
  %1518 = load <4 x i32>, ptr addrspace(3) %251, align 16, !tbaa !11
  %1519 = load <4 x i32>, ptr addrspace(3) %253, align 16, !tbaa !11
  %1520 = load <4 x i32>, ptr addrspace(3) %255, align 16, !tbaa !11
  %1521 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1398
  %1522 = load i32, ptr addrspace(3) %1521, align 4, !tbaa !7
  %1523 = or disjoint i32 %193, 10240
  %1524 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1523
  %1525 = load i32, ptr addrspace(3) %1524, align 4, !tbaa !7
  %1526 = or disjoint i32 %193, 17408
  %1527 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1526
  %1528 = load i32, ptr addrspace(3) %1527, align 4, !tbaa !7
  %1529 = or disjoint i32 %193, 24576
  %1530 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1529
  %1531 = load i32, ptr addrspace(3) %1530, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef %163, i32 noundef 16, i32 noundef %162, i32 noundef 1792, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %168, i32 noundef 16, i32 noundef %167, i32 noundef 1792, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %172, i32 noundef 16, i32 noundef %171, i32 noundef 1792, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %176, i32 noundef 16, i32 noundef %175, i32 noundef 1792, i32 noundef 0, i32 noundef 0) #11
  %1532 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1505, <4 x i32> noundef %1341, <4 x float> noundef %1430, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1522, i32 noundef 0, i32 noundef %1399) #11
  %1533 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1506, <4 x i32> noundef %1341, <4 x float> noundef %1431, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1522, i32 noundef 0, i32 noundef %1399) #11
  %1534 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1513, <4 x i32> noundef %1343, <4 x float> noundef %1532, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1522, i32 noundef 2, i32 noundef %1399) #11
  %1535 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1514, <4 x i32> noundef %1343, <4 x float> noundef %1533, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1522, i32 noundef 2, i32 noundef %1399) #11
  %1536 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1507, <4 x i32> noundef %1341, <4 x float> noundef %1434, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1525, i32 noundef 0, i32 noundef %1399) #11
  %1537 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1508, <4 x i32> noundef %1341, <4 x float> noundef %1435, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1525, i32 noundef 0, i32 noundef %1399) #11
  %1538 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1515, <4 x i32> noundef %1343, <4 x float> noundef %1536, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1525, i32 noundef 2, i32 noundef %1399) #11
  %1539 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1516, <4 x i32> noundef %1343, <4 x float> noundef %1537, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1525, i32 noundef 2, i32 noundef %1399) #11
  %1540 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1509, <4 x i32> noundef %1341, <4 x float> noundef %1438, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1528, i32 noundef 0, i32 noundef %1399) #11
  %1541 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1510, <4 x i32> noundef %1341, <4 x float> noundef %1439, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1528, i32 noundef 0, i32 noundef %1399) #11
  %1542 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1517, <4 x i32> noundef %1343, <4 x float> noundef %1540, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1528, i32 noundef 2, i32 noundef %1399) #11
  %1543 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1518, <4 x i32> noundef %1343, <4 x float> noundef %1541, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1528, i32 noundef 2, i32 noundef %1399) #11
  %1544 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1511, <4 x i32> noundef %1341, <4 x float> noundef %1442, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1531, i32 noundef 0, i32 noundef %1399) #11
  %1545 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1512, <4 x i32> noundef %1341, <4 x float> noundef %1443, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1531, i32 noundef 0, i32 noundef %1399) #11
  %1546 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1519, <4 x i32> noundef %1343, <4 x float> noundef %1544, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1531, i32 noundef 2, i32 noundef %1399) #11
  %1547 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1520, <4 x i32> noundef %1343, <4 x float> noundef %1545, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1531, i32 noundef 2, i32 noundef %1399) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1548 = or disjoint i32 %181, 28672
  %1549 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1548, i32 %75, i32 0)
  %1550 = or disjoint i32 %181, 29696
  %1551 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1550, i32 %75, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1552 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1505, <4 x i32> noundef %1360, <4 x float> noundef %1450, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1522, i32 noundef 1, i32 noundef %1399) #11
  %1553 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1506, <4 x i32> noundef %1360, <4 x float> noundef %1451, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1522, i32 noundef 1, i32 noundef %1399) #11
  %1554 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1513, <4 x i32> noundef %1361, <4 x float> noundef %1552, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1522, i32 noundef 3, i32 noundef %1399) #11
  %1555 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1514, <4 x i32> noundef %1361, <4 x float> noundef %1553, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1522, i32 noundef 3, i32 noundef %1399) #11
  %1556 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1507, <4 x i32> noundef %1360, <4 x float> noundef %1454, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1525, i32 noundef 1, i32 noundef %1399) #11
  %1557 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1508, <4 x i32> noundef %1360, <4 x float> noundef %1455, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1525, i32 noundef 1, i32 noundef %1399) #11
  %1558 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1515, <4 x i32> noundef %1361, <4 x float> noundef %1556, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1525, i32 noundef 3, i32 noundef %1399) #11
  %1559 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1516, <4 x i32> noundef %1361, <4 x float> noundef %1557, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1525, i32 noundef 3, i32 noundef %1399) #11
  %1560 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1509, <4 x i32> noundef %1360, <4 x float> noundef %1458, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1528, i32 noundef 1, i32 noundef %1399) #11
  %1561 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1510, <4 x i32> noundef %1360, <4 x float> noundef %1459, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1528, i32 noundef 1, i32 noundef %1399) #11
  %1562 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1517, <4 x i32> noundef %1361, <4 x float> noundef %1560, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1528, i32 noundef 3, i32 noundef %1399) #11
  %1563 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1518, <4 x i32> noundef %1361, <4 x float> noundef %1561, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1528, i32 noundef 3, i32 noundef %1399) #11
  %1564 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1511, <4 x i32> noundef %1360, <4 x float> noundef %1462, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1531, i32 noundef 1, i32 noundef %1399) #11
  %1565 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1512, <4 x i32> noundef %1360, <4 x float> noundef %1463, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1531, i32 noundef 1, i32 noundef %1399) #11
  %1566 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1519, <4 x i32> noundef %1361, <4 x float> noundef %1564, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1531, i32 noundef 3, i32 noundef %1399) #11
  %1567 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1520, <4 x i32> noundef %1361, <4 x float> noundef %1565, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1531, i32 noundef 3, i32 noundef %1399) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1568 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1548, i32 %79, i32 0)
  %1569 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1550, i32 %79, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1570 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1505, <4 x i32> noundef %1378, <4 x float> noundef %1468, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1522, i32 noundef 0, i32 noundef %1400) #11
  %1571 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1506, <4 x i32> noundef %1378, <4 x float> noundef %1469, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1522, i32 noundef 0, i32 noundef %1400) #11
  %1572 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1513, <4 x i32> noundef %1379, <4 x float> noundef %1570, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1522, i32 noundef 2, i32 noundef %1400) #11
  %1573 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1514, <4 x i32> noundef %1379, <4 x float> noundef %1571, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1522, i32 noundef 2, i32 noundef %1400) #11
  %1574 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1507, <4 x i32> noundef %1378, <4 x float> noundef %1472, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1525, i32 noundef 0, i32 noundef %1400) #11
  %1575 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1508, <4 x i32> noundef %1378, <4 x float> noundef %1473, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1525, i32 noundef 0, i32 noundef %1400) #11
  %1576 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1515, <4 x i32> noundef %1379, <4 x float> noundef %1574, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1525, i32 noundef 2, i32 noundef %1400) #11
  %1577 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1516, <4 x i32> noundef %1379, <4 x float> noundef %1575, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1525, i32 noundef 2, i32 noundef %1400) #11
  %1578 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1509, <4 x i32> noundef %1378, <4 x float> noundef %1476, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1528, i32 noundef 0, i32 noundef %1400) #11
  %1579 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1510, <4 x i32> noundef %1378, <4 x float> noundef %1477, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1528, i32 noundef 0, i32 noundef %1400) #11
  %1580 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1517, <4 x i32> noundef %1379, <4 x float> noundef %1578, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1528, i32 noundef 2, i32 noundef %1400) #11
  %1581 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1518, <4 x i32> noundef %1379, <4 x float> noundef %1579, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1528, i32 noundef 2, i32 noundef %1400) #11
  %1582 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1511, <4 x i32> noundef %1378, <4 x float> noundef %1480, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1531, i32 noundef 0, i32 noundef %1400) #11
  %1583 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1512, <4 x i32> noundef %1378, <4 x float> noundef %1481, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1531, i32 noundef 0, i32 noundef %1400) #11
  %1584 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1519, <4 x i32> noundef %1379, <4 x float> noundef %1582, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1531, i32 noundef 2, i32 noundef %1400) #11
  %1585 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1520, <4 x i32> noundef %1379, <4 x float> noundef %1583, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1531, i32 noundef 2, i32 noundef %1400) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1586 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1548, i32 %83, i32 0)
  %1587 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1550, i32 %83, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1588 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1505, <4 x i32> noundef %1396, <4 x float> noundef %1486, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1522, i32 noundef 1, i32 noundef %1400) #11
  %1589 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1506, <4 x i32> noundef %1396, <4 x float> noundef %1487, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1522, i32 noundef 1, i32 noundef %1400) #11
  %1590 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1513, <4 x i32> noundef %1397, <4 x float> noundef %1588, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1522, i32 noundef 3, i32 noundef %1400) #11
  %1591 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1514, <4 x i32> noundef %1397, <4 x float> noundef %1589, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1522, i32 noundef 3, i32 noundef %1400) #11
  %1592 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1507, <4 x i32> noundef %1396, <4 x float> noundef %1490, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1525, i32 noundef 1, i32 noundef %1400) #11
  %1593 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1508, <4 x i32> noundef %1396, <4 x float> noundef %1491, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1525, i32 noundef 1, i32 noundef %1400) #11
  %1594 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1515, <4 x i32> noundef %1397, <4 x float> noundef %1592, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1525, i32 noundef 3, i32 noundef %1400) #11
  %1595 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1516, <4 x i32> noundef %1397, <4 x float> noundef %1593, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1525, i32 noundef 3, i32 noundef %1400) #11
  %1596 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1509, <4 x i32> noundef %1396, <4 x float> noundef %1494, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1528, i32 noundef 1, i32 noundef %1400) #11
  %1597 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1510, <4 x i32> noundef %1396, <4 x float> noundef %1495, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1528, i32 noundef 1, i32 noundef %1400) #11
  %1598 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1517, <4 x i32> noundef %1397, <4 x float> noundef %1596, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1528, i32 noundef 3, i32 noundef %1400) #11
  %1599 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1518, <4 x i32> noundef %1397, <4 x float> noundef %1597, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1528, i32 noundef 3, i32 noundef %1400) #11
  %1600 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1511, <4 x i32> noundef %1396, <4 x float> noundef %1498, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1531, i32 noundef 1, i32 noundef %1400) #11
  %1601 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1512, <4 x i32> noundef %1396, <4 x float> noundef %1499, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1531, i32 noundef 1, i32 noundef %1400) #11
  %1602 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1519, <4 x i32> noundef %1397, <4 x float> noundef %1600, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1531, i32 noundef 3, i32 noundef %1400) #11
  %1603 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1520, <4 x i32> noundef %1397, <4 x float> noundef %1601, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1531, i32 noundef 3, i32 noundef %1400) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1604 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1548, i32 %87, i32 0)
  %1605 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1550, i32 %87, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1606 = or disjoint i32 %193, 3584
  %1607 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %1606, i32 %95, i32 0)
  %1608 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %1606, i32 %100, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1609 = load <4 x i32>, ptr addrspace(3) %345, align 16, !tbaa !11
  %1610 = load <4 x i32>, ptr addrspace(3) %347, align 16, !tbaa !11
  %1611 = load <4 x i32>, ptr addrspace(3) %349, align 16, !tbaa !11
  %1612 = load <4 x i32>, ptr addrspace(3) %351, align 16, !tbaa !11
  %1613 = load <4 x i32>, ptr addrspace(3) %353, align 16, !tbaa !11
  %1614 = load <4 x i32>, ptr addrspace(3) %355, align 16, !tbaa !11
  %1615 = load <4 x i32>, ptr addrspace(3) %357, align 16, !tbaa !11
  %1616 = load <4 x i32>, ptr addrspace(3) %359, align 16, !tbaa !11
  %1617 = load <4 x i32>, ptr addrspace(3) %361, align 16, !tbaa !11
  %1618 = load <4 x i32>, ptr addrspace(3) %363, align 16, !tbaa !11
  %1619 = load <4 x i32>, ptr addrspace(3) %365, align 16, !tbaa !11
  %1620 = load <4 x i32>, ptr addrspace(3) %367, align 16, !tbaa !11
  %1621 = load <4 x i32>, ptr addrspace(3) %369, align 16, !tbaa !11
  %1622 = load <4 x i32>, ptr addrspace(3) %371, align 16, !tbaa !11
  %1623 = load <4 x i32>, ptr addrspace(3) %373, align 16, !tbaa !11
  %1624 = load <4 x i32>, ptr addrspace(3) %375, align 16, !tbaa !11
  %1625 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1502
  %1626 = load i32, ptr addrspace(3) %1625, align 4, !tbaa !7
  %1627 = or disjoint i32 %193, 10496
  %1628 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1627
  %1629 = load i32, ptr addrspace(3) %1628, align 4, !tbaa !7
  %1630 = or disjoint i32 %193, 17664
  %1631 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1630
  %1632 = load i32, ptr addrspace(3) %1631, align 4, !tbaa !7
  %1633 = or disjoint i32 %193, 24832
  %1634 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1633
  %1635 = load i32, ptr addrspace(3) %1634, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %196, i32 noundef 16, i32 noundef %162, i32 noundef 1920, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %197, i32 noundef 16, i32 noundef %167, i32 noundef 1920, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %198, i32 noundef 16, i32 noundef %171, i32 noundef 1920, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %199, i32 noundef 16, i32 noundef %175, i32 noundef 1920, i32 noundef 0, i32 noundef 0) #11
  %1636 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1609, <4 x i32> noundef %1445, <4 x float> noundef %1534, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1626, i32 noundef 0, i32 noundef %1503) #11
  %1637 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1610, <4 x i32> noundef %1445, <4 x float> noundef %1535, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1626, i32 noundef 0, i32 noundef %1503) #11
  %1638 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1617, <4 x i32> noundef %1447, <4 x float> noundef %1636, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1626, i32 noundef 2, i32 noundef %1503) #11
  %1639 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1618, <4 x i32> noundef %1447, <4 x float> noundef %1637, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1626, i32 noundef 2, i32 noundef %1503) #11
  %1640 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1611, <4 x i32> noundef %1445, <4 x float> noundef %1538, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1629, i32 noundef 0, i32 noundef %1503) #11
  %1641 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1612, <4 x i32> noundef %1445, <4 x float> noundef %1539, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1629, i32 noundef 0, i32 noundef %1503) #11
  %1642 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1619, <4 x i32> noundef %1447, <4 x float> noundef %1640, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1629, i32 noundef 2, i32 noundef %1503) #11
  %1643 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1620, <4 x i32> noundef %1447, <4 x float> noundef %1641, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1629, i32 noundef 2, i32 noundef %1503) #11
  %1644 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1613, <4 x i32> noundef %1445, <4 x float> noundef %1542, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1632, i32 noundef 0, i32 noundef %1503) #11
  %1645 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1614, <4 x i32> noundef %1445, <4 x float> noundef %1543, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1632, i32 noundef 0, i32 noundef %1503) #11
  %1646 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1621, <4 x i32> noundef %1447, <4 x float> noundef %1644, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1632, i32 noundef 2, i32 noundef %1503) #11
  %1647 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1622, <4 x i32> noundef %1447, <4 x float> noundef %1645, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1632, i32 noundef 2, i32 noundef %1503) #11
  %1648 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1615, <4 x i32> noundef %1445, <4 x float> noundef %1546, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1635, i32 noundef 0, i32 noundef %1503) #11
  %1649 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1616, <4 x i32> noundef %1445, <4 x float> noundef %1547, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1635, i32 noundef 0, i32 noundef %1503) #11
  %1650 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1623, <4 x i32> noundef %1447, <4 x float> noundef %1648, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1635, i32 noundef 2, i32 noundef %1503) #11
  %1651 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1624, <4 x i32> noundef %1447, <4 x float> noundef %1649, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1635, i32 noundef 2, i32 noundef %1503) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1652 = or disjoint i32 %181, 30720
  %1653 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1652, i32 %75, i32 0)
  %1654 = or disjoint i32 %181, 31744
  %1655 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1654, i32 %75, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1656 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1609, <4 x i32> noundef %1464, <4 x float> noundef %1554, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1626, i32 noundef 1, i32 noundef %1503) #11
  %1657 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1610, <4 x i32> noundef %1464, <4 x float> noundef %1555, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1626, i32 noundef 1, i32 noundef %1503) #11
  %1658 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1617, <4 x i32> noundef %1465, <4 x float> noundef %1656, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1626, i32 noundef 3, i32 noundef %1503) #11
  %1659 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1618, <4 x i32> noundef %1465, <4 x float> noundef %1657, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1626, i32 noundef 3, i32 noundef %1503) #11
  %1660 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1611, <4 x i32> noundef %1464, <4 x float> noundef %1558, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1629, i32 noundef 1, i32 noundef %1503) #11
  %1661 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1612, <4 x i32> noundef %1464, <4 x float> noundef %1559, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1629, i32 noundef 1, i32 noundef %1503) #11
  %1662 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1619, <4 x i32> noundef %1465, <4 x float> noundef %1660, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1629, i32 noundef 3, i32 noundef %1503) #11
  %1663 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1620, <4 x i32> noundef %1465, <4 x float> noundef %1661, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1629, i32 noundef 3, i32 noundef %1503) #11
  %1664 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1613, <4 x i32> noundef %1464, <4 x float> noundef %1562, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1632, i32 noundef 1, i32 noundef %1503) #11
  %1665 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1614, <4 x i32> noundef %1464, <4 x float> noundef %1563, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1632, i32 noundef 1, i32 noundef %1503) #11
  %1666 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1621, <4 x i32> noundef %1465, <4 x float> noundef %1664, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1632, i32 noundef 3, i32 noundef %1503) #11
  %1667 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1622, <4 x i32> noundef %1465, <4 x float> noundef %1665, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1632, i32 noundef 3, i32 noundef %1503) #11
  %1668 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1615, <4 x i32> noundef %1464, <4 x float> noundef %1566, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1635, i32 noundef 1, i32 noundef %1503) #11
  %1669 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1616, <4 x i32> noundef %1464, <4 x float> noundef %1567, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1635, i32 noundef 1, i32 noundef %1503) #11
  %1670 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1623, <4 x i32> noundef %1465, <4 x float> noundef %1668, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1635, i32 noundef 3, i32 noundef %1503) #11
  %1671 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1624, <4 x i32> noundef %1465, <4 x float> noundef %1669, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1635, i32 noundef 3, i32 noundef %1503) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1672 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1652, i32 %79, i32 0)
  %1673 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1654, i32 %79, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1674 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1609, <4 x i32> noundef %1482, <4 x float> noundef %1572, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1626, i32 noundef 0, i32 noundef %1504) #11
  %1675 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1610, <4 x i32> noundef %1482, <4 x float> noundef %1573, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1626, i32 noundef 0, i32 noundef %1504) #11
  %1676 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1617, <4 x i32> noundef %1483, <4 x float> noundef %1674, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1626, i32 noundef 2, i32 noundef %1504) #11
  %1677 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1618, <4 x i32> noundef %1483, <4 x float> noundef %1675, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1626, i32 noundef 2, i32 noundef %1504) #11
  %1678 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1611, <4 x i32> noundef %1482, <4 x float> noundef %1576, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1629, i32 noundef 0, i32 noundef %1504) #11
  %1679 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1612, <4 x i32> noundef %1482, <4 x float> noundef %1577, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1629, i32 noundef 0, i32 noundef %1504) #11
  %1680 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1619, <4 x i32> noundef %1483, <4 x float> noundef %1678, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1629, i32 noundef 2, i32 noundef %1504) #11
  %1681 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1620, <4 x i32> noundef %1483, <4 x float> noundef %1679, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1629, i32 noundef 2, i32 noundef %1504) #11
  %1682 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1613, <4 x i32> noundef %1482, <4 x float> noundef %1580, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1632, i32 noundef 0, i32 noundef %1504) #11
  %1683 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1614, <4 x i32> noundef %1482, <4 x float> noundef %1581, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1632, i32 noundef 0, i32 noundef %1504) #11
  %1684 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1621, <4 x i32> noundef %1483, <4 x float> noundef %1682, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1632, i32 noundef 2, i32 noundef %1504) #11
  %1685 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1622, <4 x i32> noundef %1483, <4 x float> noundef %1683, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1632, i32 noundef 2, i32 noundef %1504) #11
  %1686 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1615, <4 x i32> noundef %1482, <4 x float> noundef %1584, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1635, i32 noundef 0, i32 noundef %1504) #11
  %1687 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1616, <4 x i32> noundef %1482, <4 x float> noundef %1585, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1635, i32 noundef 0, i32 noundef %1504) #11
  %1688 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1623, <4 x i32> noundef %1483, <4 x float> noundef %1686, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1635, i32 noundef 2, i32 noundef %1504) #11
  %1689 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1624, <4 x i32> noundef %1483, <4 x float> noundef %1687, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1635, i32 noundef 2, i32 noundef %1504) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1690 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1652, i32 %83, i32 0)
  %1691 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1654, i32 %83, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1692 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1609, <4 x i32> noundef %1500, <4 x float> noundef %1590, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1626, i32 noundef 1, i32 noundef %1504) #11
  %1693 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1610, <4 x i32> noundef %1500, <4 x float> noundef %1591, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1626, i32 noundef 1, i32 noundef %1504) #11
  %1694 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1617, <4 x i32> noundef %1501, <4 x float> noundef %1692, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1626, i32 noundef 3, i32 noundef %1504) #11
  %1695 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1618, <4 x i32> noundef %1501, <4 x float> noundef %1693, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1626, i32 noundef 3, i32 noundef %1504) #11
  %1696 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1611, <4 x i32> noundef %1500, <4 x float> noundef %1594, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1629, i32 noundef 1, i32 noundef %1504) #11
  %1697 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1612, <4 x i32> noundef %1500, <4 x float> noundef %1595, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1629, i32 noundef 1, i32 noundef %1504) #11
  %1698 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1619, <4 x i32> noundef %1501, <4 x float> noundef %1696, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1629, i32 noundef 3, i32 noundef %1504) #11
  %1699 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1620, <4 x i32> noundef %1501, <4 x float> noundef %1697, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1629, i32 noundef 3, i32 noundef %1504) #11
  %1700 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1613, <4 x i32> noundef %1500, <4 x float> noundef %1598, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1632, i32 noundef 1, i32 noundef %1504) #11
  %1701 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1614, <4 x i32> noundef %1500, <4 x float> noundef %1599, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1632, i32 noundef 1, i32 noundef %1504) #11
  %1702 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1621, <4 x i32> noundef %1501, <4 x float> noundef %1700, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1632, i32 noundef 3, i32 noundef %1504) #11
  %1703 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1622, <4 x i32> noundef %1501, <4 x float> noundef %1701, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1632, i32 noundef 3, i32 noundef %1504) #11
  %1704 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1615, <4 x i32> noundef %1500, <4 x float> noundef %1602, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1635, i32 noundef 1, i32 noundef %1504) #11
  %1705 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1616, <4 x i32> noundef %1500, <4 x float> noundef %1603, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1635, i32 noundef 1, i32 noundef %1504) #11
  %1706 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1623, <4 x i32> noundef %1501, <4 x float> noundef %1704, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1635, i32 noundef 3, i32 noundef %1504) #11
  %1707 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1624, <4 x i32> noundef %1501, <4 x float> noundef %1705, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1635, i32 noundef 3, i32 noundef %1504) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1708 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1652, i32 %87, i32 0)
  %1709 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1654, i32 %87, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1710 = or disjoint i32 %193, 3840
  %1711 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %1710, i32 %95, i32 0)
  %1712 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %1710, i32 %100, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1713 = load <4 x i32>, ptr addrspace(3) %216, align 16, !tbaa !11
  %1714 = load <4 x i32>, ptr addrspace(3) %219, align 16, !tbaa !11
  %1715 = load <4 x i32>, ptr addrspace(3) %222, align 16, !tbaa !11
  %1716 = load <4 x i32>, ptr addrspace(3) %225, align 16, !tbaa !11
  %1717 = load <4 x i32>, ptr addrspace(3) %228, align 16, !tbaa !11
  %1718 = load <4 x i32>, ptr addrspace(3) %231, align 16, !tbaa !11
  %1719 = load <4 x i32>, ptr addrspace(3) %234, align 16, !tbaa !11
  %1720 = load <4 x i32>, ptr addrspace(3) %237, align 16, !tbaa !11
  %1721 = load <4 x i32>, ptr addrspace(3) %241, align 16, !tbaa !11
  %1722 = load <4 x i32>, ptr addrspace(3) %243, align 16, !tbaa !11
  %1723 = load <4 x i32>, ptr addrspace(3) %245, align 16, !tbaa !11
  %1724 = load <4 x i32>, ptr addrspace(3) %247, align 16, !tbaa !11
  %1725 = load <4 x i32>, ptr addrspace(3) %249, align 16, !tbaa !11
  %1726 = load <4 x i32>, ptr addrspace(3) %251, align 16, !tbaa !11
  %1727 = load <4 x i32>, ptr addrspace(3) %253, align 16, !tbaa !11
  %1728 = load <4 x i32>, ptr addrspace(3) %255, align 16, !tbaa !11
  %1729 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1606
  %1730 = load i32, ptr addrspace(3) %1729, align 4, !tbaa !7
  %1731 = or disjoint i32 %193, 10752
  %1732 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1731
  %1733 = load i32, ptr addrspace(3) %1732, align 4, !tbaa !7
  %1734 = or disjoint i32 %193, 17920
  %1735 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1734
  %1736 = load i32, ptr addrspace(3) %1735, align 4, !tbaa !7
  %1737 = or disjoint i32 %193, 25088
  %1738 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1737
  %1739 = load i32, ptr addrspace(3) %1738, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef %163, i32 noundef 16, i32 noundef %162, i32 noundef 2048, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %168, i32 noundef 16, i32 noundef %167, i32 noundef 2048, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %172, i32 noundef 16, i32 noundef %171, i32 noundef 2048, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %176, i32 noundef 16, i32 noundef %175, i32 noundef 2048, i32 noundef 0, i32 noundef 0) #11
  %1740 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1713, <4 x i32> noundef %1549, <4 x float> noundef %1638, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1730, i32 noundef 0, i32 noundef %1607) #11
  %1741 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1714, <4 x i32> noundef %1549, <4 x float> noundef %1639, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1730, i32 noundef 0, i32 noundef %1607) #11
  %1742 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1721, <4 x i32> noundef %1551, <4 x float> noundef %1740, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1730, i32 noundef 2, i32 noundef %1607) #11
  %1743 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1722, <4 x i32> noundef %1551, <4 x float> noundef %1741, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1730, i32 noundef 2, i32 noundef %1607) #11
  %1744 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1715, <4 x i32> noundef %1549, <4 x float> noundef %1642, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1733, i32 noundef 0, i32 noundef %1607) #11
  %1745 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1716, <4 x i32> noundef %1549, <4 x float> noundef %1643, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1733, i32 noundef 0, i32 noundef %1607) #11
  %1746 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1723, <4 x i32> noundef %1551, <4 x float> noundef %1744, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1733, i32 noundef 2, i32 noundef %1607) #11
  %1747 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1724, <4 x i32> noundef %1551, <4 x float> noundef %1745, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1733, i32 noundef 2, i32 noundef %1607) #11
  %1748 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1717, <4 x i32> noundef %1549, <4 x float> noundef %1646, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1736, i32 noundef 0, i32 noundef %1607) #11
  %1749 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1718, <4 x i32> noundef %1549, <4 x float> noundef %1647, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1736, i32 noundef 0, i32 noundef %1607) #11
  %1750 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1725, <4 x i32> noundef %1551, <4 x float> noundef %1748, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1736, i32 noundef 2, i32 noundef %1607) #11
  %1751 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1726, <4 x i32> noundef %1551, <4 x float> noundef %1749, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1736, i32 noundef 2, i32 noundef %1607) #11
  %1752 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1719, <4 x i32> noundef %1549, <4 x float> noundef %1650, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1739, i32 noundef 0, i32 noundef %1607) #11
  %1753 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1720, <4 x i32> noundef %1549, <4 x float> noundef %1651, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1739, i32 noundef 0, i32 noundef %1607) #11
  %1754 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1727, <4 x i32> noundef %1551, <4 x float> noundef %1752, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1739, i32 noundef 2, i32 noundef %1607) #11
  %1755 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1728, <4 x i32> noundef %1551, <4 x float> noundef %1753, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1739, i32 noundef 2, i32 noundef %1607) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1756 = or disjoint i32 %181, 32768
  %1757 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1756, i32 %75, i32 0)
  %1758 = or disjoint i32 %181, 33792
  %1759 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1758, i32 %75, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1760 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1713, <4 x i32> noundef %1568, <4 x float> noundef %1658, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1730, i32 noundef 1, i32 noundef %1607) #11
  %1761 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1714, <4 x i32> noundef %1568, <4 x float> noundef %1659, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1730, i32 noundef 1, i32 noundef %1607) #11
  %1762 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1721, <4 x i32> noundef %1569, <4 x float> noundef %1760, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1730, i32 noundef 3, i32 noundef %1607) #11
  %1763 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1722, <4 x i32> noundef %1569, <4 x float> noundef %1761, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1730, i32 noundef 3, i32 noundef %1607) #11
  %1764 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1715, <4 x i32> noundef %1568, <4 x float> noundef %1662, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1733, i32 noundef 1, i32 noundef %1607) #11
  %1765 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1716, <4 x i32> noundef %1568, <4 x float> noundef %1663, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1733, i32 noundef 1, i32 noundef %1607) #11
  %1766 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1723, <4 x i32> noundef %1569, <4 x float> noundef %1764, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1733, i32 noundef 3, i32 noundef %1607) #11
  %1767 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1724, <4 x i32> noundef %1569, <4 x float> noundef %1765, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1733, i32 noundef 3, i32 noundef %1607) #11
  %1768 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1717, <4 x i32> noundef %1568, <4 x float> noundef %1666, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1736, i32 noundef 1, i32 noundef %1607) #11
  %1769 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1718, <4 x i32> noundef %1568, <4 x float> noundef %1667, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1736, i32 noundef 1, i32 noundef %1607) #11
  %1770 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1725, <4 x i32> noundef %1569, <4 x float> noundef %1768, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1736, i32 noundef 3, i32 noundef %1607) #11
  %1771 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1726, <4 x i32> noundef %1569, <4 x float> noundef %1769, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1736, i32 noundef 3, i32 noundef %1607) #11
  %1772 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1719, <4 x i32> noundef %1568, <4 x float> noundef %1670, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1739, i32 noundef 1, i32 noundef %1607) #11
  %1773 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1720, <4 x i32> noundef %1568, <4 x float> noundef %1671, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1739, i32 noundef 1, i32 noundef %1607) #11
  %1774 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1727, <4 x i32> noundef %1569, <4 x float> noundef %1772, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1739, i32 noundef 3, i32 noundef %1607) #11
  %1775 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1728, <4 x i32> noundef %1569, <4 x float> noundef %1773, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1739, i32 noundef 3, i32 noundef %1607) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1776 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1756, i32 %79, i32 0)
  %1777 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1758, i32 %79, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1778 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1713, <4 x i32> noundef %1586, <4 x float> noundef %1676, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1730, i32 noundef 0, i32 noundef %1608) #11
  %1779 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1714, <4 x i32> noundef %1586, <4 x float> noundef %1677, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1730, i32 noundef 0, i32 noundef %1608) #11
  %1780 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1721, <4 x i32> noundef %1587, <4 x float> noundef %1778, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1730, i32 noundef 2, i32 noundef %1608) #11
  %1781 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1722, <4 x i32> noundef %1587, <4 x float> noundef %1779, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1730, i32 noundef 2, i32 noundef %1608) #11
  %1782 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1715, <4 x i32> noundef %1586, <4 x float> noundef %1680, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1733, i32 noundef 0, i32 noundef %1608) #11
  %1783 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1716, <4 x i32> noundef %1586, <4 x float> noundef %1681, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1733, i32 noundef 0, i32 noundef %1608) #11
  %1784 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1723, <4 x i32> noundef %1587, <4 x float> noundef %1782, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1733, i32 noundef 2, i32 noundef %1608) #11
  %1785 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1724, <4 x i32> noundef %1587, <4 x float> noundef %1783, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1733, i32 noundef 2, i32 noundef %1608) #11
  %1786 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1717, <4 x i32> noundef %1586, <4 x float> noundef %1684, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1736, i32 noundef 0, i32 noundef %1608) #11
  %1787 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1718, <4 x i32> noundef %1586, <4 x float> noundef %1685, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1736, i32 noundef 0, i32 noundef %1608) #11
  %1788 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1725, <4 x i32> noundef %1587, <4 x float> noundef %1786, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1736, i32 noundef 2, i32 noundef %1608) #11
  %1789 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1726, <4 x i32> noundef %1587, <4 x float> noundef %1787, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1736, i32 noundef 2, i32 noundef %1608) #11
  %1790 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1719, <4 x i32> noundef %1586, <4 x float> noundef %1688, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1739, i32 noundef 0, i32 noundef %1608) #11
  %1791 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1720, <4 x i32> noundef %1586, <4 x float> noundef %1689, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1739, i32 noundef 0, i32 noundef %1608) #11
  %1792 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1727, <4 x i32> noundef %1587, <4 x float> noundef %1790, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1739, i32 noundef 2, i32 noundef %1608) #11
  %1793 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1728, <4 x i32> noundef %1587, <4 x float> noundef %1791, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1739, i32 noundef 2, i32 noundef %1608) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1794 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1756, i32 %83, i32 0)
  %1795 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1758, i32 %83, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1796 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1713, <4 x i32> noundef %1604, <4 x float> noundef %1694, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1730, i32 noundef 1, i32 noundef %1608) #11
  %1797 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1714, <4 x i32> noundef %1604, <4 x float> noundef %1695, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1730, i32 noundef 1, i32 noundef %1608) #11
  %1798 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1721, <4 x i32> noundef %1605, <4 x float> noundef %1796, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1730, i32 noundef 3, i32 noundef %1608) #11
  %1799 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1722, <4 x i32> noundef %1605, <4 x float> noundef %1797, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1730, i32 noundef 3, i32 noundef %1608) #11
  %1800 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1715, <4 x i32> noundef %1604, <4 x float> noundef %1698, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1733, i32 noundef 1, i32 noundef %1608) #11
  %1801 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1716, <4 x i32> noundef %1604, <4 x float> noundef %1699, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1733, i32 noundef 1, i32 noundef %1608) #11
  %1802 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1723, <4 x i32> noundef %1605, <4 x float> noundef %1800, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1733, i32 noundef 3, i32 noundef %1608) #11
  %1803 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1724, <4 x i32> noundef %1605, <4 x float> noundef %1801, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1733, i32 noundef 3, i32 noundef %1608) #11
  %1804 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1717, <4 x i32> noundef %1604, <4 x float> noundef %1702, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1736, i32 noundef 1, i32 noundef %1608) #11
  %1805 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1718, <4 x i32> noundef %1604, <4 x float> noundef %1703, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1736, i32 noundef 1, i32 noundef %1608) #11
  %1806 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1725, <4 x i32> noundef %1605, <4 x float> noundef %1804, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1736, i32 noundef 3, i32 noundef %1608) #11
  %1807 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1726, <4 x i32> noundef %1605, <4 x float> noundef %1805, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1736, i32 noundef 3, i32 noundef %1608) #11
  %1808 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1719, <4 x i32> noundef %1604, <4 x float> noundef %1706, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1739, i32 noundef 1, i32 noundef %1608) #11
  %1809 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1720, <4 x i32> noundef %1604, <4 x float> noundef %1707, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1739, i32 noundef 1, i32 noundef %1608) #11
  %1810 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1727, <4 x i32> noundef %1605, <4 x float> noundef %1808, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1739, i32 noundef 3, i32 noundef %1608) #11
  %1811 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1728, <4 x i32> noundef %1605, <4 x float> noundef %1809, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1739, i32 noundef 3, i32 noundef %1608) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1812 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1756, i32 %87, i32 0)
  %1813 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1758, i32 %87, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1814 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %193, i32 %97, i32 0)
  %1815 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %193, i32 %102, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1816 = load <4 x i32>, ptr addrspace(3) %345, align 16, !tbaa !11
  %1817 = load <4 x i32>, ptr addrspace(3) %347, align 16, !tbaa !11
  %1818 = load <4 x i32>, ptr addrspace(3) %349, align 16, !tbaa !11
  %1819 = load <4 x i32>, ptr addrspace(3) %351, align 16, !tbaa !11
  %1820 = load <4 x i32>, ptr addrspace(3) %353, align 16, !tbaa !11
  %1821 = load <4 x i32>, ptr addrspace(3) %355, align 16, !tbaa !11
  %1822 = load <4 x i32>, ptr addrspace(3) %357, align 16, !tbaa !11
  %1823 = load <4 x i32>, ptr addrspace(3) %359, align 16, !tbaa !11
  %1824 = load <4 x i32>, ptr addrspace(3) %361, align 16, !tbaa !11
  %1825 = load <4 x i32>, ptr addrspace(3) %363, align 16, !tbaa !11
  %1826 = load <4 x i32>, ptr addrspace(3) %365, align 16, !tbaa !11
  %1827 = load <4 x i32>, ptr addrspace(3) %367, align 16, !tbaa !11
  %1828 = load <4 x i32>, ptr addrspace(3) %369, align 16, !tbaa !11
  %1829 = load <4 x i32>, ptr addrspace(3) %371, align 16, !tbaa !11
  %1830 = load <4 x i32>, ptr addrspace(3) %373, align 16, !tbaa !11
  %1831 = load <4 x i32>, ptr addrspace(3) %375, align 16, !tbaa !11
  %1832 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1710
  %1833 = load i32, ptr addrspace(3) %1832, align 4, !tbaa !7
  %1834 = or disjoint i32 %193, 11008
  %1835 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1834
  %1836 = load i32, ptr addrspace(3) %1835, align 4, !tbaa !7
  %1837 = or disjoint i32 %193, 18176
  %1838 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1837
  %1839 = load i32, ptr addrspace(3) %1838, align 4, !tbaa !7
  %1840 = or disjoint i32 %193, 25344
  %1841 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1840
  %1842 = load i32, ptr addrspace(3) %1841, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %196, i32 noundef 16, i32 noundef %162, i32 noundef 2176, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %197, i32 noundef 16, i32 noundef %167, i32 noundef 2176, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %198, i32 noundef 16, i32 noundef %171, i32 noundef 2176, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %199, i32 noundef 16, i32 noundef %175, i32 noundef 2176, i32 noundef 0, i32 noundef 0) #11
  %1843 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1816, <4 x i32> noundef %1653, <4 x float> noundef %1742, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1833, i32 noundef 0, i32 noundef %1711) #11
  %1844 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1817, <4 x i32> noundef %1653, <4 x float> noundef %1743, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1833, i32 noundef 0, i32 noundef %1711) #11
  %1845 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1824, <4 x i32> noundef %1655, <4 x float> noundef %1843, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1833, i32 noundef 2, i32 noundef %1711) #11
  %1846 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1825, <4 x i32> noundef %1655, <4 x float> noundef %1844, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1833, i32 noundef 2, i32 noundef %1711) #11
  %1847 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1818, <4 x i32> noundef %1653, <4 x float> noundef %1746, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1836, i32 noundef 0, i32 noundef %1711) #11
  %1848 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1819, <4 x i32> noundef %1653, <4 x float> noundef %1747, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1836, i32 noundef 0, i32 noundef %1711) #11
  %1849 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1826, <4 x i32> noundef %1655, <4 x float> noundef %1847, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1836, i32 noundef 2, i32 noundef %1711) #11
  %1850 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1827, <4 x i32> noundef %1655, <4 x float> noundef %1848, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1836, i32 noundef 2, i32 noundef %1711) #11
  %1851 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1820, <4 x i32> noundef %1653, <4 x float> noundef %1750, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1839, i32 noundef 0, i32 noundef %1711) #11
  %1852 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1821, <4 x i32> noundef %1653, <4 x float> noundef %1751, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1839, i32 noundef 0, i32 noundef %1711) #11
  %1853 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1828, <4 x i32> noundef %1655, <4 x float> noundef %1851, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1839, i32 noundef 2, i32 noundef %1711) #11
  %1854 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1829, <4 x i32> noundef %1655, <4 x float> noundef %1852, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1839, i32 noundef 2, i32 noundef %1711) #11
  %1855 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1822, <4 x i32> noundef %1653, <4 x float> noundef %1754, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1842, i32 noundef 0, i32 noundef %1711) #11
  %1856 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1823, <4 x i32> noundef %1653, <4 x float> noundef %1755, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1842, i32 noundef 0, i32 noundef %1711) #11
  %1857 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1830, <4 x i32> noundef %1655, <4 x float> noundef %1855, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1842, i32 noundef 2, i32 noundef %1711) #11
  %1858 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1831, <4 x i32> noundef %1655, <4 x float> noundef %1856, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1842, i32 noundef 2, i32 noundef %1711) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1859 = or disjoint i32 %181, 34816
  %1860 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1859, i32 %75, i32 0)
  %1861 = or disjoint i32 %181, 35840
  %1862 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1861, i32 %75, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1863 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1816, <4 x i32> noundef %1672, <4 x float> noundef %1762, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1833, i32 noundef 1, i32 noundef %1711) #11
  %1864 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1817, <4 x i32> noundef %1672, <4 x float> noundef %1763, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1833, i32 noundef 1, i32 noundef %1711) #11
  %1865 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1824, <4 x i32> noundef %1673, <4 x float> noundef %1863, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1833, i32 noundef 3, i32 noundef %1711) #11
  %1866 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1825, <4 x i32> noundef %1673, <4 x float> noundef %1864, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1833, i32 noundef 3, i32 noundef %1711) #11
  %1867 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1818, <4 x i32> noundef %1672, <4 x float> noundef %1766, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1836, i32 noundef 1, i32 noundef %1711) #11
  %1868 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1819, <4 x i32> noundef %1672, <4 x float> noundef %1767, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1836, i32 noundef 1, i32 noundef %1711) #11
  %1869 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1826, <4 x i32> noundef %1673, <4 x float> noundef %1867, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1836, i32 noundef 3, i32 noundef %1711) #11
  %1870 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1827, <4 x i32> noundef %1673, <4 x float> noundef %1868, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1836, i32 noundef 3, i32 noundef %1711) #11
  %1871 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1820, <4 x i32> noundef %1672, <4 x float> noundef %1770, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1839, i32 noundef 1, i32 noundef %1711) #11
  %1872 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1821, <4 x i32> noundef %1672, <4 x float> noundef %1771, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1839, i32 noundef 1, i32 noundef %1711) #11
  %1873 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1828, <4 x i32> noundef %1673, <4 x float> noundef %1871, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1839, i32 noundef 3, i32 noundef %1711) #11
  %1874 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1829, <4 x i32> noundef %1673, <4 x float> noundef %1872, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1839, i32 noundef 3, i32 noundef %1711) #11
  %1875 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1822, <4 x i32> noundef %1672, <4 x float> noundef %1774, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1842, i32 noundef 1, i32 noundef %1711) #11
  %1876 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1823, <4 x i32> noundef %1672, <4 x float> noundef %1775, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1842, i32 noundef 1, i32 noundef %1711) #11
  %1877 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1830, <4 x i32> noundef %1673, <4 x float> noundef %1875, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1842, i32 noundef 3, i32 noundef %1711) #11
  %1878 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1831, <4 x i32> noundef %1673, <4 x float> noundef %1876, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1842, i32 noundef 3, i32 noundef %1711) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1879 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1859, i32 %79, i32 0)
  %1880 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1861, i32 %79, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1881 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1816, <4 x i32> noundef %1690, <4 x float> noundef %1780, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1833, i32 noundef 0, i32 noundef %1712) #11
  %1882 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1817, <4 x i32> noundef %1690, <4 x float> noundef %1781, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1833, i32 noundef 0, i32 noundef %1712) #11
  %1883 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1824, <4 x i32> noundef %1691, <4 x float> noundef %1881, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1833, i32 noundef 2, i32 noundef %1712) #11
  %1884 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1825, <4 x i32> noundef %1691, <4 x float> noundef %1882, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1833, i32 noundef 2, i32 noundef %1712) #11
  %1885 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1818, <4 x i32> noundef %1690, <4 x float> noundef %1784, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1836, i32 noundef 0, i32 noundef %1712) #11
  %1886 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1819, <4 x i32> noundef %1690, <4 x float> noundef %1785, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1836, i32 noundef 0, i32 noundef %1712) #11
  %1887 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1826, <4 x i32> noundef %1691, <4 x float> noundef %1885, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1836, i32 noundef 2, i32 noundef %1712) #11
  %1888 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1827, <4 x i32> noundef %1691, <4 x float> noundef %1886, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1836, i32 noundef 2, i32 noundef %1712) #11
  %1889 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1820, <4 x i32> noundef %1690, <4 x float> noundef %1788, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1839, i32 noundef 0, i32 noundef %1712) #11
  %1890 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1821, <4 x i32> noundef %1690, <4 x float> noundef %1789, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1839, i32 noundef 0, i32 noundef %1712) #11
  %1891 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1828, <4 x i32> noundef %1691, <4 x float> noundef %1889, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1839, i32 noundef 2, i32 noundef %1712) #11
  %1892 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1829, <4 x i32> noundef %1691, <4 x float> noundef %1890, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1839, i32 noundef 2, i32 noundef %1712) #11
  %1893 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1822, <4 x i32> noundef %1690, <4 x float> noundef %1792, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1842, i32 noundef 0, i32 noundef %1712) #11
  %1894 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1823, <4 x i32> noundef %1690, <4 x float> noundef %1793, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1842, i32 noundef 0, i32 noundef %1712) #11
  %1895 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1830, <4 x i32> noundef %1691, <4 x float> noundef %1893, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1842, i32 noundef 2, i32 noundef %1712) #11
  %1896 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1831, <4 x i32> noundef %1691, <4 x float> noundef %1894, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1842, i32 noundef 2, i32 noundef %1712) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1897 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1859, i32 %83, i32 0)
  %1898 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1861, i32 %83, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1899 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1816, <4 x i32> noundef %1708, <4 x float> noundef %1798, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1833, i32 noundef 1, i32 noundef %1712) #11
  %1900 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1817, <4 x i32> noundef %1708, <4 x float> noundef %1799, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1833, i32 noundef 1, i32 noundef %1712) #11
  %1901 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1824, <4 x i32> noundef %1709, <4 x float> noundef %1899, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1833, i32 noundef 3, i32 noundef %1712) #11
  %1902 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1825, <4 x i32> noundef %1709, <4 x float> noundef %1900, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1833, i32 noundef 3, i32 noundef %1712) #11
  %1903 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1818, <4 x i32> noundef %1708, <4 x float> noundef %1802, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1836, i32 noundef 1, i32 noundef %1712) #11
  %1904 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1819, <4 x i32> noundef %1708, <4 x float> noundef %1803, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1836, i32 noundef 1, i32 noundef %1712) #11
  %1905 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1826, <4 x i32> noundef %1709, <4 x float> noundef %1903, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1836, i32 noundef 3, i32 noundef %1712) #11
  %1906 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1827, <4 x i32> noundef %1709, <4 x float> noundef %1904, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1836, i32 noundef 3, i32 noundef %1712) #11
  %1907 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1820, <4 x i32> noundef %1708, <4 x float> noundef %1806, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1839, i32 noundef 1, i32 noundef %1712) #11
  %1908 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1821, <4 x i32> noundef %1708, <4 x float> noundef %1807, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1839, i32 noundef 1, i32 noundef %1712) #11
  %1909 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1828, <4 x i32> noundef %1709, <4 x float> noundef %1907, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1839, i32 noundef 3, i32 noundef %1712) #11
  %1910 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1829, <4 x i32> noundef %1709, <4 x float> noundef %1908, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1839, i32 noundef 3, i32 noundef %1712) #11
  %1911 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1822, <4 x i32> noundef %1708, <4 x float> noundef %1810, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1842, i32 noundef 1, i32 noundef %1712) #11
  %1912 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1823, <4 x i32> noundef %1708, <4 x float> noundef %1811, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1842, i32 noundef 1, i32 noundef %1712) #11
  %1913 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1830, <4 x i32> noundef %1709, <4 x float> noundef %1911, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1842, i32 noundef 3, i32 noundef %1712) #11
  %1914 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1831, <4 x i32> noundef %1709, <4 x float> noundef %1912, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1842, i32 noundef 3, i32 noundef %1712) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1915 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1859, i32 %87, i32 0)
  %1916 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1861, i32 %87, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1917 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %210, i32 %97, i32 0)
  %1918 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %210, i32 %102, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1919 = load <4 x i32>, ptr addrspace(3) %216, align 16, !tbaa !11
  %1920 = load <4 x i32>, ptr addrspace(3) %219, align 16, !tbaa !11
  %1921 = load <4 x i32>, ptr addrspace(3) %222, align 16, !tbaa !11
  %1922 = load <4 x i32>, ptr addrspace(3) %225, align 16, !tbaa !11
  %1923 = load <4 x i32>, ptr addrspace(3) %228, align 16, !tbaa !11
  %1924 = load <4 x i32>, ptr addrspace(3) %231, align 16, !tbaa !11
  %1925 = load <4 x i32>, ptr addrspace(3) %234, align 16, !tbaa !11
  %1926 = load <4 x i32>, ptr addrspace(3) %237, align 16, !tbaa !11
  %1927 = load <4 x i32>, ptr addrspace(3) %241, align 16, !tbaa !11
  %1928 = load <4 x i32>, ptr addrspace(3) %243, align 16, !tbaa !11
  %1929 = load <4 x i32>, ptr addrspace(3) %245, align 16, !tbaa !11
  %1930 = load <4 x i32>, ptr addrspace(3) %247, align 16, !tbaa !11
  %1931 = load <4 x i32>, ptr addrspace(3) %249, align 16, !tbaa !11
  %1932 = load <4 x i32>, ptr addrspace(3) %251, align 16, !tbaa !11
  %1933 = load <4 x i32>, ptr addrspace(3) %253, align 16, !tbaa !11
  %1934 = load <4 x i32>, ptr addrspace(3) %255, align 16, !tbaa !11
  %1935 = or disjoint i32 %193, 4096
  %1936 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1935
  %1937 = load i32, ptr addrspace(3) %1936, align 4, !tbaa !7
  %1938 = or disjoint i32 %193, 11264
  %1939 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1938
  %1940 = load i32, ptr addrspace(3) %1939, align 4, !tbaa !7
  %1941 = or disjoint i32 %193, 18432
  %1942 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1941
  %1943 = load i32, ptr addrspace(3) %1942, align 4, !tbaa !7
  %1944 = or disjoint i32 %193, 25600
  %1945 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %1944
  %1946 = load i32, ptr addrspace(3) %1945, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef %163, i32 noundef 16, i32 noundef %162, i32 noundef 2304, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %168, i32 noundef 16, i32 noundef %167, i32 noundef 2304, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %172, i32 noundef 16, i32 noundef %171, i32 noundef 2304, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %176, i32 noundef 16, i32 noundef %175, i32 noundef 2304, i32 noundef 0, i32 noundef 0) #11
  %1947 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1919, <4 x i32> noundef %1757, <4 x float> noundef %1845, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1937, i32 noundef 0, i32 noundef %1814) #11
  %1948 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1920, <4 x i32> noundef %1757, <4 x float> noundef %1846, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1937, i32 noundef 0, i32 noundef %1814) #11
  %1949 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1927, <4 x i32> noundef %1759, <4 x float> noundef %1947, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1937, i32 noundef 2, i32 noundef %1814) #11
  %1950 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1928, <4 x i32> noundef %1759, <4 x float> noundef %1948, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1937, i32 noundef 2, i32 noundef %1814) #11
  %1951 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1921, <4 x i32> noundef %1757, <4 x float> noundef %1849, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1940, i32 noundef 0, i32 noundef %1814) #11
  %1952 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1922, <4 x i32> noundef %1757, <4 x float> noundef %1850, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1940, i32 noundef 0, i32 noundef %1814) #11
  %1953 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1929, <4 x i32> noundef %1759, <4 x float> noundef %1951, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1940, i32 noundef 2, i32 noundef %1814) #11
  %1954 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1930, <4 x i32> noundef %1759, <4 x float> noundef %1952, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1940, i32 noundef 2, i32 noundef %1814) #11
  %1955 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1923, <4 x i32> noundef %1757, <4 x float> noundef %1853, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1943, i32 noundef 0, i32 noundef %1814) #11
  %1956 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1924, <4 x i32> noundef %1757, <4 x float> noundef %1854, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1943, i32 noundef 0, i32 noundef %1814) #11
  %1957 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1931, <4 x i32> noundef %1759, <4 x float> noundef %1955, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1943, i32 noundef 2, i32 noundef %1814) #11
  %1958 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1932, <4 x i32> noundef %1759, <4 x float> noundef %1956, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1943, i32 noundef 2, i32 noundef %1814) #11
  %1959 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1925, <4 x i32> noundef %1757, <4 x float> noundef %1857, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1946, i32 noundef 0, i32 noundef %1814) #11
  %1960 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1926, <4 x i32> noundef %1757, <4 x float> noundef %1858, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1946, i32 noundef 0, i32 noundef %1814) #11
  %1961 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1933, <4 x i32> noundef %1759, <4 x float> noundef %1959, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1946, i32 noundef 2, i32 noundef %1814) #11
  %1962 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1934, <4 x i32> noundef %1759, <4 x float> noundef %1960, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1946, i32 noundef 2, i32 noundef %1814) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1963 = or disjoint i32 %181, 36864
  %1964 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1963, i32 %75, i32 0)
  %1965 = or disjoint i32 %181, 37888
  %1966 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1965, i32 %75, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1967 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1919, <4 x i32> noundef %1776, <4 x float> noundef %1865, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1937, i32 noundef 1, i32 noundef %1814) #11
  %1968 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1920, <4 x i32> noundef %1776, <4 x float> noundef %1866, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1937, i32 noundef 1, i32 noundef %1814) #11
  %1969 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1927, <4 x i32> noundef %1777, <4 x float> noundef %1967, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1937, i32 noundef 3, i32 noundef %1814) #11
  %1970 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1928, <4 x i32> noundef %1777, <4 x float> noundef %1968, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1937, i32 noundef 3, i32 noundef %1814) #11
  %1971 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1921, <4 x i32> noundef %1776, <4 x float> noundef %1869, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1940, i32 noundef 1, i32 noundef %1814) #11
  %1972 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1922, <4 x i32> noundef %1776, <4 x float> noundef %1870, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1940, i32 noundef 1, i32 noundef %1814) #11
  %1973 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1929, <4 x i32> noundef %1777, <4 x float> noundef %1971, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1940, i32 noundef 3, i32 noundef %1814) #11
  %1974 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1930, <4 x i32> noundef %1777, <4 x float> noundef %1972, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1940, i32 noundef 3, i32 noundef %1814) #11
  %1975 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1923, <4 x i32> noundef %1776, <4 x float> noundef %1873, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1943, i32 noundef 1, i32 noundef %1814) #11
  %1976 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1924, <4 x i32> noundef %1776, <4 x float> noundef %1874, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1943, i32 noundef 1, i32 noundef %1814) #11
  %1977 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1931, <4 x i32> noundef %1777, <4 x float> noundef %1975, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1943, i32 noundef 3, i32 noundef %1814) #11
  %1978 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1932, <4 x i32> noundef %1777, <4 x float> noundef %1976, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1943, i32 noundef 3, i32 noundef %1814) #11
  %1979 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1925, <4 x i32> noundef %1776, <4 x float> noundef %1877, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1946, i32 noundef 1, i32 noundef %1814) #11
  %1980 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1926, <4 x i32> noundef %1776, <4 x float> noundef %1878, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1946, i32 noundef 1, i32 noundef %1814) #11
  %1981 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1933, <4 x i32> noundef %1777, <4 x float> noundef %1979, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1946, i32 noundef 3, i32 noundef %1814) #11
  %1982 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1934, <4 x i32> noundef %1777, <4 x float> noundef %1980, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1946, i32 noundef 3, i32 noundef %1814) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1983 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1963, i32 %79, i32 0)
  %1984 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1965, i32 %79, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1985 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1919, <4 x i32> noundef %1794, <4 x float> noundef %1883, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1937, i32 noundef 0, i32 noundef %1815) #11
  %1986 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1920, <4 x i32> noundef %1794, <4 x float> noundef %1884, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1937, i32 noundef 0, i32 noundef %1815) #11
  %1987 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1927, <4 x i32> noundef %1795, <4 x float> noundef %1985, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1937, i32 noundef 2, i32 noundef %1815) #11
  %1988 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1928, <4 x i32> noundef %1795, <4 x float> noundef %1986, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1937, i32 noundef 2, i32 noundef %1815) #11
  %1989 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1921, <4 x i32> noundef %1794, <4 x float> noundef %1887, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1940, i32 noundef 0, i32 noundef %1815) #11
  %1990 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1922, <4 x i32> noundef %1794, <4 x float> noundef %1888, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1940, i32 noundef 0, i32 noundef %1815) #11
  %1991 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1929, <4 x i32> noundef %1795, <4 x float> noundef %1989, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1940, i32 noundef 2, i32 noundef %1815) #11
  %1992 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1930, <4 x i32> noundef %1795, <4 x float> noundef %1990, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1940, i32 noundef 2, i32 noundef %1815) #11
  %1993 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1923, <4 x i32> noundef %1794, <4 x float> noundef %1891, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1943, i32 noundef 0, i32 noundef %1815) #11
  %1994 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1924, <4 x i32> noundef %1794, <4 x float> noundef %1892, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1943, i32 noundef 0, i32 noundef %1815) #11
  %1995 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1931, <4 x i32> noundef %1795, <4 x float> noundef %1993, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1943, i32 noundef 2, i32 noundef %1815) #11
  %1996 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1932, <4 x i32> noundef %1795, <4 x float> noundef %1994, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1943, i32 noundef 2, i32 noundef %1815) #11
  %1997 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1925, <4 x i32> noundef %1794, <4 x float> noundef %1895, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1946, i32 noundef 0, i32 noundef %1815) #11
  %1998 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1926, <4 x i32> noundef %1794, <4 x float> noundef %1896, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1946, i32 noundef 0, i32 noundef %1815) #11
  %1999 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1933, <4 x i32> noundef %1795, <4 x float> noundef %1997, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1946, i32 noundef 2, i32 noundef %1815) #11
  %2000 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1934, <4 x i32> noundef %1795, <4 x float> noundef %1998, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1946, i32 noundef 2, i32 noundef %1815) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2001 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1963, i32 %83, i32 0)
  %2002 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1965, i32 %83, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2003 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1919, <4 x i32> noundef %1812, <4 x float> noundef %1901, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1937, i32 noundef 1, i32 noundef %1815) #11
  %2004 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1920, <4 x i32> noundef %1812, <4 x float> noundef %1902, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1937, i32 noundef 1, i32 noundef %1815) #11
  %2005 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1927, <4 x i32> noundef %1813, <4 x float> noundef %2003, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1937, i32 noundef 3, i32 noundef %1815) #11
  %2006 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1928, <4 x i32> noundef %1813, <4 x float> noundef %2004, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1937, i32 noundef 3, i32 noundef %1815) #11
  %2007 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1921, <4 x i32> noundef %1812, <4 x float> noundef %1905, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1940, i32 noundef 1, i32 noundef %1815) #11
  %2008 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1922, <4 x i32> noundef %1812, <4 x float> noundef %1906, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1940, i32 noundef 1, i32 noundef %1815) #11
  %2009 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1929, <4 x i32> noundef %1813, <4 x float> noundef %2007, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1940, i32 noundef 3, i32 noundef %1815) #11
  %2010 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1930, <4 x i32> noundef %1813, <4 x float> noundef %2008, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1940, i32 noundef 3, i32 noundef %1815) #11
  %2011 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1923, <4 x i32> noundef %1812, <4 x float> noundef %1909, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1943, i32 noundef 1, i32 noundef %1815) #11
  %2012 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1924, <4 x i32> noundef %1812, <4 x float> noundef %1910, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1943, i32 noundef 1, i32 noundef %1815) #11
  %2013 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1931, <4 x i32> noundef %1813, <4 x float> noundef %2011, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1943, i32 noundef 3, i32 noundef %1815) #11
  %2014 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1932, <4 x i32> noundef %1813, <4 x float> noundef %2012, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1943, i32 noundef 3, i32 noundef %1815) #11
  %2015 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1925, <4 x i32> noundef %1812, <4 x float> noundef %1913, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1946, i32 noundef 1, i32 noundef %1815) #11
  %2016 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1926, <4 x i32> noundef %1812, <4 x float> noundef %1914, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1946, i32 noundef 1, i32 noundef %1815) #11
  %2017 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1933, <4 x i32> noundef %1813, <4 x float> noundef %2015, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1946, i32 noundef 3, i32 noundef %1815) #11
  %2018 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1934, <4 x i32> noundef %1813, <4 x float> noundef %2016, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1946, i32 noundef 3, i32 noundef %1815) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2019 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1963, i32 %87, i32 0)
  %2020 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %1965, i32 %87, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2021 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %342, i32 %97, i32 0)
  %2022 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %342, i32 %102, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %2023 = load <4 x i32>, ptr addrspace(3) %345, align 16, !tbaa !11
  %2024 = load <4 x i32>, ptr addrspace(3) %347, align 16, !tbaa !11
  %2025 = load <4 x i32>, ptr addrspace(3) %349, align 16, !tbaa !11
  %2026 = load <4 x i32>, ptr addrspace(3) %351, align 16, !tbaa !11
  %2027 = load <4 x i32>, ptr addrspace(3) %353, align 16, !tbaa !11
  %2028 = load <4 x i32>, ptr addrspace(3) %355, align 16, !tbaa !11
  %2029 = load <4 x i32>, ptr addrspace(3) %357, align 16, !tbaa !11
  %2030 = load <4 x i32>, ptr addrspace(3) %359, align 16, !tbaa !11
  %2031 = load <4 x i32>, ptr addrspace(3) %361, align 16, !tbaa !11
  %2032 = load <4 x i32>, ptr addrspace(3) %363, align 16, !tbaa !11
  %2033 = load <4 x i32>, ptr addrspace(3) %365, align 16, !tbaa !11
  %2034 = load <4 x i32>, ptr addrspace(3) %367, align 16, !tbaa !11
  %2035 = load <4 x i32>, ptr addrspace(3) %369, align 16, !tbaa !11
  %2036 = load <4 x i32>, ptr addrspace(3) %371, align 16, !tbaa !11
  %2037 = load <4 x i32>, ptr addrspace(3) %373, align 16, !tbaa !11
  %2038 = load <4 x i32>, ptr addrspace(3) %375, align 16, !tbaa !11
  %2039 = or disjoint i32 %193, 4352
  %2040 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2039
  %2041 = load i32, ptr addrspace(3) %2040, align 4, !tbaa !7
  %2042 = or disjoint i32 %193, 11520
  %2043 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2042
  %2044 = load i32, ptr addrspace(3) %2043, align 4, !tbaa !7
  %2045 = or disjoint i32 %193, 18688
  %2046 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2045
  %2047 = load i32, ptr addrspace(3) %2046, align 4, !tbaa !7
  %2048 = or disjoint i32 %193, 25856
  %2049 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2048
  %2050 = load i32, ptr addrspace(3) %2049, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %196, i32 noundef 16, i32 noundef %162, i32 noundef 2432, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %197, i32 noundef 16, i32 noundef %167, i32 noundef 2432, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %198, i32 noundef 16, i32 noundef %171, i32 noundef 2432, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %199, i32 noundef 16, i32 noundef %175, i32 noundef 2432, i32 noundef 0, i32 noundef 0) #11
  %2051 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2023, <4 x i32> noundef %1860, <4 x float> noundef %1949, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2041, i32 noundef 0, i32 noundef %1917) #11
  %2052 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2024, <4 x i32> noundef %1860, <4 x float> noundef %1950, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2041, i32 noundef 0, i32 noundef %1917) #11
  %2053 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2031, <4 x i32> noundef %1862, <4 x float> noundef %2051, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2041, i32 noundef 2, i32 noundef %1917) #11
  %2054 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2032, <4 x i32> noundef %1862, <4 x float> noundef %2052, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2041, i32 noundef 2, i32 noundef %1917) #11
  %2055 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2025, <4 x i32> noundef %1860, <4 x float> noundef %1953, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2044, i32 noundef 0, i32 noundef %1917) #11
  %2056 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2026, <4 x i32> noundef %1860, <4 x float> noundef %1954, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2044, i32 noundef 0, i32 noundef %1917) #11
  %2057 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2033, <4 x i32> noundef %1862, <4 x float> noundef %2055, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2044, i32 noundef 2, i32 noundef %1917) #11
  %2058 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2034, <4 x i32> noundef %1862, <4 x float> noundef %2056, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2044, i32 noundef 2, i32 noundef %1917) #11
  %2059 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2027, <4 x i32> noundef %1860, <4 x float> noundef %1957, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2047, i32 noundef 0, i32 noundef %1917) #11
  %2060 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2028, <4 x i32> noundef %1860, <4 x float> noundef %1958, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2047, i32 noundef 0, i32 noundef %1917) #11
  %2061 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2035, <4 x i32> noundef %1862, <4 x float> noundef %2059, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2047, i32 noundef 2, i32 noundef %1917) #11
  %2062 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2036, <4 x i32> noundef %1862, <4 x float> noundef %2060, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2047, i32 noundef 2, i32 noundef %1917) #11
  %2063 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2029, <4 x i32> noundef %1860, <4 x float> noundef %1961, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2050, i32 noundef 0, i32 noundef %1917) #11
  %2064 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2030, <4 x i32> noundef %1860, <4 x float> noundef %1962, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2050, i32 noundef 0, i32 noundef %1917) #11
  %2065 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2037, <4 x i32> noundef %1862, <4 x float> noundef %2063, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2050, i32 noundef 2, i32 noundef %1917) #11
  %2066 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2038, <4 x i32> noundef %1862, <4 x float> noundef %2064, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2050, i32 noundef 2, i32 noundef %1917) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2067 = or disjoint i32 %181, 38912
  %2068 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2067, i32 %75, i32 0)
  %2069 = or disjoint i32 %181, 39936
  %2070 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2069, i32 %75, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2071 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2023, <4 x i32> noundef %1879, <4 x float> noundef %1969, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2041, i32 noundef 1, i32 noundef %1917) #11
  %2072 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2024, <4 x i32> noundef %1879, <4 x float> noundef %1970, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2041, i32 noundef 1, i32 noundef %1917) #11
  %2073 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2031, <4 x i32> noundef %1880, <4 x float> noundef %2071, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2041, i32 noundef 3, i32 noundef %1917) #11
  %2074 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2032, <4 x i32> noundef %1880, <4 x float> noundef %2072, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2041, i32 noundef 3, i32 noundef %1917) #11
  %2075 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2025, <4 x i32> noundef %1879, <4 x float> noundef %1973, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2044, i32 noundef 1, i32 noundef %1917) #11
  %2076 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2026, <4 x i32> noundef %1879, <4 x float> noundef %1974, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2044, i32 noundef 1, i32 noundef %1917) #11
  %2077 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2033, <4 x i32> noundef %1880, <4 x float> noundef %2075, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2044, i32 noundef 3, i32 noundef %1917) #11
  %2078 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2034, <4 x i32> noundef %1880, <4 x float> noundef %2076, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2044, i32 noundef 3, i32 noundef %1917) #11
  %2079 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2027, <4 x i32> noundef %1879, <4 x float> noundef %1977, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2047, i32 noundef 1, i32 noundef %1917) #11
  %2080 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2028, <4 x i32> noundef %1879, <4 x float> noundef %1978, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2047, i32 noundef 1, i32 noundef %1917) #11
  %2081 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2035, <4 x i32> noundef %1880, <4 x float> noundef %2079, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2047, i32 noundef 3, i32 noundef %1917) #11
  %2082 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2036, <4 x i32> noundef %1880, <4 x float> noundef %2080, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2047, i32 noundef 3, i32 noundef %1917) #11
  %2083 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2029, <4 x i32> noundef %1879, <4 x float> noundef %1981, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2050, i32 noundef 1, i32 noundef %1917) #11
  %2084 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2030, <4 x i32> noundef %1879, <4 x float> noundef %1982, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2050, i32 noundef 1, i32 noundef %1917) #11
  %2085 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2037, <4 x i32> noundef %1880, <4 x float> noundef %2083, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2050, i32 noundef 3, i32 noundef %1917) #11
  %2086 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2038, <4 x i32> noundef %1880, <4 x float> noundef %2084, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2050, i32 noundef 3, i32 noundef %1917) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2087 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2067, i32 %79, i32 0)
  %2088 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2069, i32 %79, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2089 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2023, <4 x i32> noundef %1897, <4 x float> noundef %1987, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2041, i32 noundef 0, i32 noundef %1918) #11
  %2090 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2024, <4 x i32> noundef %1897, <4 x float> noundef %1988, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2041, i32 noundef 0, i32 noundef %1918) #11
  %2091 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2031, <4 x i32> noundef %1898, <4 x float> noundef %2089, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2041, i32 noundef 2, i32 noundef %1918) #11
  %2092 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2032, <4 x i32> noundef %1898, <4 x float> noundef %2090, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2041, i32 noundef 2, i32 noundef %1918) #11
  %2093 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2025, <4 x i32> noundef %1897, <4 x float> noundef %1991, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2044, i32 noundef 0, i32 noundef %1918) #11
  %2094 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2026, <4 x i32> noundef %1897, <4 x float> noundef %1992, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2044, i32 noundef 0, i32 noundef %1918) #11
  %2095 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2033, <4 x i32> noundef %1898, <4 x float> noundef %2093, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2044, i32 noundef 2, i32 noundef %1918) #11
  %2096 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2034, <4 x i32> noundef %1898, <4 x float> noundef %2094, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2044, i32 noundef 2, i32 noundef %1918) #11
  %2097 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2027, <4 x i32> noundef %1897, <4 x float> noundef %1995, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2047, i32 noundef 0, i32 noundef %1918) #11
  %2098 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2028, <4 x i32> noundef %1897, <4 x float> noundef %1996, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2047, i32 noundef 0, i32 noundef %1918) #11
  %2099 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2035, <4 x i32> noundef %1898, <4 x float> noundef %2097, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2047, i32 noundef 2, i32 noundef %1918) #11
  %2100 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2036, <4 x i32> noundef %1898, <4 x float> noundef %2098, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2047, i32 noundef 2, i32 noundef %1918) #11
  %2101 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2029, <4 x i32> noundef %1897, <4 x float> noundef %1999, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2050, i32 noundef 0, i32 noundef %1918) #11
  %2102 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2030, <4 x i32> noundef %1897, <4 x float> noundef %2000, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2050, i32 noundef 0, i32 noundef %1918) #11
  %2103 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2037, <4 x i32> noundef %1898, <4 x float> noundef %2101, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2050, i32 noundef 2, i32 noundef %1918) #11
  %2104 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2038, <4 x i32> noundef %1898, <4 x float> noundef %2102, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2050, i32 noundef 2, i32 noundef %1918) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2105 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2067, i32 %83, i32 0)
  %2106 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2069, i32 %83, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2107 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2023, <4 x i32> noundef %1915, <4 x float> noundef %2005, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2041, i32 noundef 1, i32 noundef %1918) #11
  %2108 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2024, <4 x i32> noundef %1915, <4 x float> noundef %2006, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2041, i32 noundef 1, i32 noundef %1918) #11
  %2109 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2031, <4 x i32> noundef %1916, <4 x float> noundef %2107, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2041, i32 noundef 3, i32 noundef %1918) #11
  %2110 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2032, <4 x i32> noundef %1916, <4 x float> noundef %2108, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2041, i32 noundef 3, i32 noundef %1918) #11
  %2111 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2025, <4 x i32> noundef %1915, <4 x float> noundef %2009, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2044, i32 noundef 1, i32 noundef %1918) #11
  %2112 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2026, <4 x i32> noundef %1915, <4 x float> noundef %2010, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2044, i32 noundef 1, i32 noundef %1918) #11
  %2113 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2033, <4 x i32> noundef %1916, <4 x float> noundef %2111, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2044, i32 noundef 3, i32 noundef %1918) #11
  %2114 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2034, <4 x i32> noundef %1916, <4 x float> noundef %2112, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2044, i32 noundef 3, i32 noundef %1918) #11
  %2115 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2027, <4 x i32> noundef %1915, <4 x float> noundef %2013, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2047, i32 noundef 1, i32 noundef %1918) #11
  %2116 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2028, <4 x i32> noundef %1915, <4 x float> noundef %2014, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2047, i32 noundef 1, i32 noundef %1918) #11
  %2117 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2035, <4 x i32> noundef %1916, <4 x float> noundef %2115, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2047, i32 noundef 3, i32 noundef %1918) #11
  %2118 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2036, <4 x i32> noundef %1916, <4 x float> noundef %2116, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2047, i32 noundef 3, i32 noundef %1918) #11
  %2119 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2029, <4 x i32> noundef %1915, <4 x float> noundef %2017, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2050, i32 noundef 1, i32 noundef %1918) #11
  %2120 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2030, <4 x i32> noundef %1915, <4 x float> noundef %2018, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2050, i32 noundef 1, i32 noundef %1918) #11
  %2121 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2037, <4 x i32> noundef %1916, <4 x float> noundef %2119, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2050, i32 noundef 3, i32 noundef %1918) #11
  %2122 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2038, <4 x i32> noundef %1916, <4 x float> noundef %2120, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2050, i32 noundef 3, i32 noundef %1918) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2123 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2067, i32 %87, i32 0)
  %2124 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2069, i32 %87, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2125 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %462, i32 %97, i32 0)
  %2126 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %462, i32 %102, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %2127 = load <4 x i32>, ptr addrspace(3) %216, align 16, !tbaa !11
  %2128 = load <4 x i32>, ptr addrspace(3) %219, align 16, !tbaa !11
  %2129 = load <4 x i32>, ptr addrspace(3) %222, align 16, !tbaa !11
  %2130 = load <4 x i32>, ptr addrspace(3) %225, align 16, !tbaa !11
  %2131 = load <4 x i32>, ptr addrspace(3) %228, align 16, !tbaa !11
  %2132 = load <4 x i32>, ptr addrspace(3) %231, align 16, !tbaa !11
  %2133 = load <4 x i32>, ptr addrspace(3) %234, align 16, !tbaa !11
  %2134 = load <4 x i32>, ptr addrspace(3) %237, align 16, !tbaa !11
  %2135 = load <4 x i32>, ptr addrspace(3) %241, align 16, !tbaa !11
  %2136 = load <4 x i32>, ptr addrspace(3) %243, align 16, !tbaa !11
  %2137 = load <4 x i32>, ptr addrspace(3) %245, align 16, !tbaa !11
  %2138 = load <4 x i32>, ptr addrspace(3) %247, align 16, !tbaa !11
  %2139 = load <4 x i32>, ptr addrspace(3) %249, align 16, !tbaa !11
  %2140 = load <4 x i32>, ptr addrspace(3) %251, align 16, !tbaa !11
  %2141 = load <4 x i32>, ptr addrspace(3) %253, align 16, !tbaa !11
  %2142 = load <4 x i32>, ptr addrspace(3) %255, align 16, !tbaa !11
  %2143 = or disjoint i32 %193, 4608
  %2144 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2143
  %2145 = load i32, ptr addrspace(3) %2144, align 4, !tbaa !7
  %2146 = or disjoint i32 %193, 11776
  %2147 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2146
  %2148 = load i32, ptr addrspace(3) %2147, align 4, !tbaa !7
  %2149 = or disjoint i32 %193, 18944
  %2150 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2149
  %2151 = load i32, ptr addrspace(3) %2150, align 4, !tbaa !7
  %2152 = or disjoint i32 %193, 26112
  %2153 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2152
  %2154 = load i32, ptr addrspace(3) %2153, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef %163, i32 noundef 16, i32 noundef %162, i32 noundef 2560, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %168, i32 noundef 16, i32 noundef %167, i32 noundef 2560, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %172, i32 noundef 16, i32 noundef %171, i32 noundef 2560, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %176, i32 noundef 16, i32 noundef %175, i32 noundef 2560, i32 noundef 0, i32 noundef 0) #11
  %2155 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2127, <4 x i32> noundef %1964, <4 x float> noundef %2053, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2145, i32 noundef 0, i32 noundef %2021) #11
  %2156 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2128, <4 x i32> noundef %1964, <4 x float> noundef %2054, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2145, i32 noundef 0, i32 noundef %2021) #11
  %2157 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2135, <4 x i32> noundef %1966, <4 x float> noundef %2155, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2145, i32 noundef 2, i32 noundef %2021) #11
  %2158 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2136, <4 x i32> noundef %1966, <4 x float> noundef %2156, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2145, i32 noundef 2, i32 noundef %2021) #11
  %2159 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2129, <4 x i32> noundef %1964, <4 x float> noundef %2057, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2148, i32 noundef 0, i32 noundef %2021) #11
  %2160 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2130, <4 x i32> noundef %1964, <4 x float> noundef %2058, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2148, i32 noundef 0, i32 noundef %2021) #11
  %2161 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2137, <4 x i32> noundef %1966, <4 x float> noundef %2159, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2148, i32 noundef 2, i32 noundef %2021) #11
  %2162 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2138, <4 x i32> noundef %1966, <4 x float> noundef %2160, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2148, i32 noundef 2, i32 noundef %2021) #11
  %2163 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2131, <4 x i32> noundef %1964, <4 x float> noundef %2061, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2151, i32 noundef 0, i32 noundef %2021) #11
  %2164 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2132, <4 x i32> noundef %1964, <4 x float> noundef %2062, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2151, i32 noundef 0, i32 noundef %2021) #11
  %2165 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2139, <4 x i32> noundef %1966, <4 x float> noundef %2163, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2151, i32 noundef 2, i32 noundef %2021) #11
  %2166 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2140, <4 x i32> noundef %1966, <4 x float> noundef %2164, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2151, i32 noundef 2, i32 noundef %2021) #11
  %2167 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2133, <4 x i32> noundef %1964, <4 x float> noundef %2065, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2154, i32 noundef 0, i32 noundef %2021) #11
  %2168 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2134, <4 x i32> noundef %1964, <4 x float> noundef %2066, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2154, i32 noundef 0, i32 noundef %2021) #11
  %2169 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2141, <4 x i32> noundef %1966, <4 x float> noundef %2167, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2154, i32 noundef 2, i32 noundef %2021) #11
  %2170 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2142, <4 x i32> noundef %1966, <4 x float> noundef %2168, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2154, i32 noundef 2, i32 noundef %2021) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2171 = or disjoint i32 %181, 40960
  %2172 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2171, i32 %75, i32 0)
  %2173 = or disjoint i32 %181, 41984
  %2174 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2173, i32 %75, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2175 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2127, <4 x i32> noundef %1983, <4 x float> noundef %2073, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2145, i32 noundef 1, i32 noundef %2021) #11
  %2176 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2128, <4 x i32> noundef %1983, <4 x float> noundef %2074, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2145, i32 noundef 1, i32 noundef %2021) #11
  %2177 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2135, <4 x i32> noundef %1984, <4 x float> noundef %2175, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2145, i32 noundef 3, i32 noundef %2021) #11
  %2178 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2136, <4 x i32> noundef %1984, <4 x float> noundef %2176, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2145, i32 noundef 3, i32 noundef %2021) #11
  %2179 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2129, <4 x i32> noundef %1983, <4 x float> noundef %2077, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2148, i32 noundef 1, i32 noundef %2021) #11
  %2180 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2130, <4 x i32> noundef %1983, <4 x float> noundef %2078, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2148, i32 noundef 1, i32 noundef %2021) #11
  %2181 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2137, <4 x i32> noundef %1984, <4 x float> noundef %2179, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2148, i32 noundef 3, i32 noundef %2021) #11
  %2182 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2138, <4 x i32> noundef %1984, <4 x float> noundef %2180, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2148, i32 noundef 3, i32 noundef %2021) #11
  %2183 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2131, <4 x i32> noundef %1983, <4 x float> noundef %2081, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2151, i32 noundef 1, i32 noundef %2021) #11
  %2184 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2132, <4 x i32> noundef %1983, <4 x float> noundef %2082, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2151, i32 noundef 1, i32 noundef %2021) #11
  %2185 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2139, <4 x i32> noundef %1984, <4 x float> noundef %2183, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2151, i32 noundef 3, i32 noundef %2021) #11
  %2186 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2140, <4 x i32> noundef %1984, <4 x float> noundef %2184, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2151, i32 noundef 3, i32 noundef %2021) #11
  %2187 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2133, <4 x i32> noundef %1983, <4 x float> noundef %2085, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2154, i32 noundef 1, i32 noundef %2021) #11
  %2188 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2134, <4 x i32> noundef %1983, <4 x float> noundef %2086, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2154, i32 noundef 1, i32 noundef %2021) #11
  %2189 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2141, <4 x i32> noundef %1984, <4 x float> noundef %2187, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2154, i32 noundef 3, i32 noundef %2021) #11
  %2190 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2142, <4 x i32> noundef %1984, <4 x float> noundef %2188, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2154, i32 noundef 3, i32 noundef %2021) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2191 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2171, i32 %79, i32 0)
  %2192 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2173, i32 %79, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2193 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2127, <4 x i32> noundef %2001, <4 x float> noundef %2091, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2145, i32 noundef 0, i32 noundef %2022) #11
  %2194 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2128, <4 x i32> noundef %2001, <4 x float> noundef %2092, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2145, i32 noundef 0, i32 noundef %2022) #11
  %2195 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2135, <4 x i32> noundef %2002, <4 x float> noundef %2193, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2145, i32 noundef 2, i32 noundef %2022) #11
  %2196 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2136, <4 x i32> noundef %2002, <4 x float> noundef %2194, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2145, i32 noundef 2, i32 noundef %2022) #11
  %2197 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2129, <4 x i32> noundef %2001, <4 x float> noundef %2095, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2148, i32 noundef 0, i32 noundef %2022) #11
  %2198 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2130, <4 x i32> noundef %2001, <4 x float> noundef %2096, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2148, i32 noundef 0, i32 noundef %2022) #11
  %2199 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2137, <4 x i32> noundef %2002, <4 x float> noundef %2197, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2148, i32 noundef 2, i32 noundef %2022) #11
  %2200 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2138, <4 x i32> noundef %2002, <4 x float> noundef %2198, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2148, i32 noundef 2, i32 noundef %2022) #11
  %2201 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2131, <4 x i32> noundef %2001, <4 x float> noundef %2099, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2151, i32 noundef 0, i32 noundef %2022) #11
  %2202 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2132, <4 x i32> noundef %2001, <4 x float> noundef %2100, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2151, i32 noundef 0, i32 noundef %2022) #11
  %2203 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2139, <4 x i32> noundef %2002, <4 x float> noundef %2201, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2151, i32 noundef 2, i32 noundef %2022) #11
  %2204 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2140, <4 x i32> noundef %2002, <4 x float> noundef %2202, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2151, i32 noundef 2, i32 noundef %2022) #11
  %2205 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2133, <4 x i32> noundef %2001, <4 x float> noundef %2103, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2154, i32 noundef 0, i32 noundef %2022) #11
  %2206 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2134, <4 x i32> noundef %2001, <4 x float> noundef %2104, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2154, i32 noundef 0, i32 noundef %2022) #11
  %2207 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2141, <4 x i32> noundef %2002, <4 x float> noundef %2205, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2154, i32 noundef 2, i32 noundef %2022) #11
  %2208 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2142, <4 x i32> noundef %2002, <4 x float> noundef %2206, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2154, i32 noundef 2, i32 noundef %2022) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2209 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2171, i32 %83, i32 0)
  %2210 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2173, i32 %83, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2211 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2127, <4 x i32> noundef %2019, <4 x float> noundef %2109, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2145, i32 noundef 1, i32 noundef %2022) #11
  %2212 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2128, <4 x i32> noundef %2019, <4 x float> noundef %2110, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2145, i32 noundef 1, i32 noundef %2022) #11
  %2213 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2135, <4 x i32> noundef %2020, <4 x float> noundef %2211, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2145, i32 noundef 3, i32 noundef %2022) #11
  %2214 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2136, <4 x i32> noundef %2020, <4 x float> noundef %2212, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2145, i32 noundef 3, i32 noundef %2022) #11
  %2215 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2129, <4 x i32> noundef %2019, <4 x float> noundef %2113, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2148, i32 noundef 1, i32 noundef %2022) #11
  %2216 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2130, <4 x i32> noundef %2019, <4 x float> noundef %2114, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2148, i32 noundef 1, i32 noundef %2022) #11
  %2217 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2137, <4 x i32> noundef %2020, <4 x float> noundef %2215, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2148, i32 noundef 3, i32 noundef %2022) #11
  %2218 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2138, <4 x i32> noundef %2020, <4 x float> noundef %2216, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2148, i32 noundef 3, i32 noundef %2022) #11
  %2219 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2131, <4 x i32> noundef %2019, <4 x float> noundef %2117, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2151, i32 noundef 1, i32 noundef %2022) #11
  %2220 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2132, <4 x i32> noundef %2019, <4 x float> noundef %2118, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2151, i32 noundef 1, i32 noundef %2022) #11
  %2221 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2139, <4 x i32> noundef %2020, <4 x float> noundef %2219, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2151, i32 noundef 3, i32 noundef %2022) #11
  %2222 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2140, <4 x i32> noundef %2020, <4 x float> noundef %2220, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2151, i32 noundef 3, i32 noundef %2022) #11
  %2223 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2133, <4 x i32> noundef %2019, <4 x float> noundef %2121, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2154, i32 noundef 1, i32 noundef %2022) #11
  %2224 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2134, <4 x i32> noundef %2019, <4 x float> noundef %2122, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2154, i32 noundef 1, i32 noundef %2022) #11
  %2225 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2141, <4 x i32> noundef %2020, <4 x float> noundef %2223, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2154, i32 noundef 3, i32 noundef %2022) #11
  %2226 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2142, <4 x i32> noundef %2020, <4 x float> noundef %2224, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2154, i32 noundef 3, i32 noundef %2022) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2227 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2171, i32 %87, i32 0)
  %2228 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2173, i32 %87, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2229 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %566, i32 %97, i32 0)
  %2230 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %566, i32 %102, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %2231 = load <4 x i32>, ptr addrspace(3) %345, align 16, !tbaa !11
  %2232 = load <4 x i32>, ptr addrspace(3) %347, align 16, !tbaa !11
  %2233 = load <4 x i32>, ptr addrspace(3) %349, align 16, !tbaa !11
  %2234 = load <4 x i32>, ptr addrspace(3) %351, align 16, !tbaa !11
  %2235 = load <4 x i32>, ptr addrspace(3) %353, align 16, !tbaa !11
  %2236 = load <4 x i32>, ptr addrspace(3) %355, align 16, !tbaa !11
  %2237 = load <4 x i32>, ptr addrspace(3) %357, align 16, !tbaa !11
  %2238 = load <4 x i32>, ptr addrspace(3) %359, align 16, !tbaa !11
  %2239 = load <4 x i32>, ptr addrspace(3) %361, align 16, !tbaa !11
  %2240 = load <4 x i32>, ptr addrspace(3) %363, align 16, !tbaa !11
  %2241 = load <4 x i32>, ptr addrspace(3) %365, align 16, !tbaa !11
  %2242 = load <4 x i32>, ptr addrspace(3) %367, align 16, !tbaa !11
  %2243 = load <4 x i32>, ptr addrspace(3) %369, align 16, !tbaa !11
  %2244 = load <4 x i32>, ptr addrspace(3) %371, align 16, !tbaa !11
  %2245 = load <4 x i32>, ptr addrspace(3) %373, align 16, !tbaa !11
  %2246 = load <4 x i32>, ptr addrspace(3) %375, align 16, !tbaa !11
  %2247 = or disjoint i32 %193, 4864
  %2248 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2247
  %2249 = load i32, ptr addrspace(3) %2248, align 4, !tbaa !7
  %2250 = or disjoint i32 %193, 12032
  %2251 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2250
  %2252 = load i32, ptr addrspace(3) %2251, align 4, !tbaa !7
  %2253 = or disjoint i32 %193, 19200
  %2254 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2253
  %2255 = load i32, ptr addrspace(3) %2254, align 4, !tbaa !7
  %2256 = or disjoint i32 %193, 26368
  %2257 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2256
  %2258 = load i32, ptr addrspace(3) %2257, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %196, i32 noundef 16, i32 noundef %162, i32 noundef 2688, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %197, i32 noundef 16, i32 noundef %167, i32 noundef 2688, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %198, i32 noundef 16, i32 noundef %171, i32 noundef 2688, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %199, i32 noundef 16, i32 noundef %175, i32 noundef 2688, i32 noundef 0, i32 noundef 0) #11
  %2259 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2231, <4 x i32> noundef %2068, <4 x float> noundef %2157, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2249, i32 noundef 0, i32 noundef %2125) #11
  %2260 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2232, <4 x i32> noundef %2068, <4 x float> noundef %2158, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2249, i32 noundef 0, i32 noundef %2125) #11
  %2261 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2239, <4 x i32> noundef %2070, <4 x float> noundef %2259, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2249, i32 noundef 2, i32 noundef %2125) #11
  %2262 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2240, <4 x i32> noundef %2070, <4 x float> noundef %2260, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2249, i32 noundef 2, i32 noundef %2125) #11
  %2263 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2233, <4 x i32> noundef %2068, <4 x float> noundef %2161, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2252, i32 noundef 0, i32 noundef %2125) #11
  %2264 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2234, <4 x i32> noundef %2068, <4 x float> noundef %2162, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2252, i32 noundef 0, i32 noundef %2125) #11
  %2265 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2241, <4 x i32> noundef %2070, <4 x float> noundef %2263, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2252, i32 noundef 2, i32 noundef %2125) #11
  %2266 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2242, <4 x i32> noundef %2070, <4 x float> noundef %2264, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2252, i32 noundef 2, i32 noundef %2125) #11
  %2267 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2235, <4 x i32> noundef %2068, <4 x float> noundef %2165, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2255, i32 noundef 0, i32 noundef %2125) #11
  %2268 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2236, <4 x i32> noundef %2068, <4 x float> noundef %2166, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2255, i32 noundef 0, i32 noundef %2125) #11
  %2269 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2243, <4 x i32> noundef %2070, <4 x float> noundef %2267, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2255, i32 noundef 2, i32 noundef %2125) #11
  %2270 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2244, <4 x i32> noundef %2070, <4 x float> noundef %2268, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2255, i32 noundef 2, i32 noundef %2125) #11
  %2271 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2237, <4 x i32> noundef %2068, <4 x float> noundef %2169, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2258, i32 noundef 0, i32 noundef %2125) #11
  %2272 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2238, <4 x i32> noundef %2068, <4 x float> noundef %2170, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2258, i32 noundef 0, i32 noundef %2125) #11
  %2273 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2245, <4 x i32> noundef %2070, <4 x float> noundef %2271, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2258, i32 noundef 2, i32 noundef %2125) #11
  %2274 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2246, <4 x i32> noundef %2070, <4 x float> noundef %2272, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2258, i32 noundef 2, i32 noundef %2125) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2275 = or disjoint i32 %181, 43008
  %2276 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2275, i32 %75, i32 0)
  %2277 = or disjoint i32 %181, 44032
  %2278 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2277, i32 %75, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2279 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2231, <4 x i32> noundef %2087, <4 x float> noundef %2177, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2249, i32 noundef 1, i32 noundef %2125) #11
  %2280 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2232, <4 x i32> noundef %2087, <4 x float> noundef %2178, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2249, i32 noundef 1, i32 noundef %2125) #11
  %2281 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2239, <4 x i32> noundef %2088, <4 x float> noundef %2279, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2249, i32 noundef 3, i32 noundef %2125) #11
  %2282 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2240, <4 x i32> noundef %2088, <4 x float> noundef %2280, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2249, i32 noundef 3, i32 noundef %2125) #11
  %2283 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2233, <4 x i32> noundef %2087, <4 x float> noundef %2181, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2252, i32 noundef 1, i32 noundef %2125) #11
  %2284 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2234, <4 x i32> noundef %2087, <4 x float> noundef %2182, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2252, i32 noundef 1, i32 noundef %2125) #11
  %2285 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2241, <4 x i32> noundef %2088, <4 x float> noundef %2283, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2252, i32 noundef 3, i32 noundef %2125) #11
  %2286 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2242, <4 x i32> noundef %2088, <4 x float> noundef %2284, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2252, i32 noundef 3, i32 noundef %2125) #11
  %2287 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2235, <4 x i32> noundef %2087, <4 x float> noundef %2185, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2255, i32 noundef 1, i32 noundef %2125) #11
  %2288 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2236, <4 x i32> noundef %2087, <4 x float> noundef %2186, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2255, i32 noundef 1, i32 noundef %2125) #11
  %2289 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2243, <4 x i32> noundef %2088, <4 x float> noundef %2287, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2255, i32 noundef 3, i32 noundef %2125) #11
  %2290 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2244, <4 x i32> noundef %2088, <4 x float> noundef %2288, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2255, i32 noundef 3, i32 noundef %2125) #11
  %2291 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2237, <4 x i32> noundef %2087, <4 x float> noundef %2189, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2258, i32 noundef 1, i32 noundef %2125) #11
  %2292 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2238, <4 x i32> noundef %2087, <4 x float> noundef %2190, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2258, i32 noundef 1, i32 noundef %2125) #11
  %2293 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2245, <4 x i32> noundef %2088, <4 x float> noundef %2291, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2258, i32 noundef 3, i32 noundef %2125) #11
  %2294 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2246, <4 x i32> noundef %2088, <4 x float> noundef %2292, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2258, i32 noundef 3, i32 noundef %2125) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2295 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2275, i32 %79, i32 0)
  %2296 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2277, i32 %79, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2297 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2231, <4 x i32> noundef %2105, <4 x float> noundef %2195, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2249, i32 noundef 0, i32 noundef %2126) #11
  %2298 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2232, <4 x i32> noundef %2105, <4 x float> noundef %2196, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2249, i32 noundef 0, i32 noundef %2126) #11
  %2299 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2239, <4 x i32> noundef %2106, <4 x float> noundef %2297, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2249, i32 noundef 2, i32 noundef %2126) #11
  %2300 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2240, <4 x i32> noundef %2106, <4 x float> noundef %2298, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2249, i32 noundef 2, i32 noundef %2126) #11
  %2301 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2233, <4 x i32> noundef %2105, <4 x float> noundef %2199, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2252, i32 noundef 0, i32 noundef %2126) #11
  %2302 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2234, <4 x i32> noundef %2105, <4 x float> noundef %2200, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2252, i32 noundef 0, i32 noundef %2126) #11
  %2303 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2241, <4 x i32> noundef %2106, <4 x float> noundef %2301, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2252, i32 noundef 2, i32 noundef %2126) #11
  %2304 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2242, <4 x i32> noundef %2106, <4 x float> noundef %2302, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2252, i32 noundef 2, i32 noundef %2126) #11
  %2305 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2235, <4 x i32> noundef %2105, <4 x float> noundef %2203, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2255, i32 noundef 0, i32 noundef %2126) #11
  %2306 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2236, <4 x i32> noundef %2105, <4 x float> noundef %2204, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2255, i32 noundef 0, i32 noundef %2126) #11
  %2307 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2243, <4 x i32> noundef %2106, <4 x float> noundef %2305, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2255, i32 noundef 2, i32 noundef %2126) #11
  %2308 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2244, <4 x i32> noundef %2106, <4 x float> noundef %2306, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2255, i32 noundef 2, i32 noundef %2126) #11
  %2309 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2237, <4 x i32> noundef %2105, <4 x float> noundef %2207, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2258, i32 noundef 0, i32 noundef %2126) #11
  %2310 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2238, <4 x i32> noundef %2105, <4 x float> noundef %2208, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2258, i32 noundef 0, i32 noundef %2126) #11
  %2311 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2245, <4 x i32> noundef %2106, <4 x float> noundef %2309, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2258, i32 noundef 2, i32 noundef %2126) #11
  %2312 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2246, <4 x i32> noundef %2106, <4 x float> noundef %2310, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2258, i32 noundef 2, i32 noundef %2126) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2313 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2275, i32 %83, i32 0)
  %2314 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2277, i32 %83, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2315 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2231, <4 x i32> noundef %2123, <4 x float> noundef %2213, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2249, i32 noundef 1, i32 noundef %2126) #11
  %2316 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2232, <4 x i32> noundef %2123, <4 x float> noundef %2214, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2249, i32 noundef 1, i32 noundef %2126) #11
  %2317 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2239, <4 x i32> noundef %2124, <4 x float> noundef %2315, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2249, i32 noundef 3, i32 noundef %2126) #11
  %2318 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2240, <4 x i32> noundef %2124, <4 x float> noundef %2316, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2249, i32 noundef 3, i32 noundef %2126) #11
  %2319 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2233, <4 x i32> noundef %2123, <4 x float> noundef %2217, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2252, i32 noundef 1, i32 noundef %2126) #11
  %2320 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2234, <4 x i32> noundef %2123, <4 x float> noundef %2218, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2252, i32 noundef 1, i32 noundef %2126) #11
  %2321 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2241, <4 x i32> noundef %2124, <4 x float> noundef %2319, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2252, i32 noundef 3, i32 noundef %2126) #11
  %2322 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2242, <4 x i32> noundef %2124, <4 x float> noundef %2320, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2252, i32 noundef 3, i32 noundef %2126) #11
  %2323 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2235, <4 x i32> noundef %2123, <4 x float> noundef %2221, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2255, i32 noundef 1, i32 noundef %2126) #11
  %2324 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2236, <4 x i32> noundef %2123, <4 x float> noundef %2222, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2255, i32 noundef 1, i32 noundef %2126) #11
  %2325 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2243, <4 x i32> noundef %2124, <4 x float> noundef %2323, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2255, i32 noundef 3, i32 noundef %2126) #11
  %2326 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2244, <4 x i32> noundef %2124, <4 x float> noundef %2324, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2255, i32 noundef 3, i32 noundef %2126) #11
  %2327 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2237, <4 x i32> noundef %2123, <4 x float> noundef %2225, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2258, i32 noundef 1, i32 noundef %2126) #11
  %2328 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2238, <4 x i32> noundef %2123, <4 x float> noundef %2226, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2258, i32 noundef 1, i32 noundef %2126) #11
  %2329 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2245, <4 x i32> noundef %2124, <4 x float> noundef %2327, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2258, i32 noundef 3, i32 noundef %2126) #11
  %2330 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2246, <4 x i32> noundef %2124, <4 x float> noundef %2328, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2258, i32 noundef 3, i32 noundef %2126) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2331 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2275, i32 %87, i32 0)
  %2332 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2277, i32 %87, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2333 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %670, i32 %97, i32 0)
  %2334 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %670, i32 %102, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %2335 = load <4 x i32>, ptr addrspace(3) %216, align 16, !tbaa !11
  %2336 = load <4 x i32>, ptr addrspace(3) %219, align 16, !tbaa !11
  %2337 = load <4 x i32>, ptr addrspace(3) %222, align 16, !tbaa !11
  %2338 = load <4 x i32>, ptr addrspace(3) %225, align 16, !tbaa !11
  %2339 = load <4 x i32>, ptr addrspace(3) %228, align 16, !tbaa !11
  %2340 = load <4 x i32>, ptr addrspace(3) %231, align 16, !tbaa !11
  %2341 = load <4 x i32>, ptr addrspace(3) %234, align 16, !tbaa !11
  %2342 = load <4 x i32>, ptr addrspace(3) %237, align 16, !tbaa !11
  %2343 = load <4 x i32>, ptr addrspace(3) %241, align 16, !tbaa !11
  %2344 = load <4 x i32>, ptr addrspace(3) %243, align 16, !tbaa !11
  %2345 = load <4 x i32>, ptr addrspace(3) %245, align 16, !tbaa !11
  %2346 = load <4 x i32>, ptr addrspace(3) %247, align 16, !tbaa !11
  %2347 = load <4 x i32>, ptr addrspace(3) %249, align 16, !tbaa !11
  %2348 = load <4 x i32>, ptr addrspace(3) %251, align 16, !tbaa !11
  %2349 = load <4 x i32>, ptr addrspace(3) %253, align 16, !tbaa !11
  %2350 = load <4 x i32>, ptr addrspace(3) %255, align 16, !tbaa !11
  %2351 = or disjoint i32 %193, 5120
  %2352 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2351
  %2353 = load i32, ptr addrspace(3) %2352, align 4, !tbaa !7
  %2354 = or disjoint i32 %193, 12288
  %2355 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2354
  %2356 = load i32, ptr addrspace(3) %2355, align 4, !tbaa !7
  %2357 = or disjoint i32 %193, 19456
  %2358 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2357
  %2359 = load i32, ptr addrspace(3) %2358, align 4, !tbaa !7
  %2360 = or disjoint i32 %193, 26624
  %2361 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2360
  %2362 = load i32, ptr addrspace(3) %2361, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef %163, i32 noundef 16, i32 noundef %162, i32 noundef 2816, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %168, i32 noundef 16, i32 noundef %167, i32 noundef 2816, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %172, i32 noundef 16, i32 noundef %171, i32 noundef 2816, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %176, i32 noundef 16, i32 noundef %175, i32 noundef 2816, i32 noundef 0, i32 noundef 0) #11
  %2363 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2335, <4 x i32> noundef %2172, <4 x float> noundef %2261, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2353, i32 noundef 0, i32 noundef %2229) #11
  %2364 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2336, <4 x i32> noundef %2172, <4 x float> noundef %2262, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2353, i32 noundef 0, i32 noundef %2229) #11
  %2365 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2343, <4 x i32> noundef %2174, <4 x float> noundef %2363, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2353, i32 noundef 2, i32 noundef %2229) #11
  %2366 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2344, <4 x i32> noundef %2174, <4 x float> noundef %2364, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2353, i32 noundef 2, i32 noundef %2229) #11
  %2367 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2337, <4 x i32> noundef %2172, <4 x float> noundef %2265, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2356, i32 noundef 0, i32 noundef %2229) #11
  %2368 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2338, <4 x i32> noundef %2172, <4 x float> noundef %2266, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2356, i32 noundef 0, i32 noundef %2229) #11
  %2369 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2345, <4 x i32> noundef %2174, <4 x float> noundef %2367, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2356, i32 noundef 2, i32 noundef %2229) #11
  %2370 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2346, <4 x i32> noundef %2174, <4 x float> noundef %2368, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2356, i32 noundef 2, i32 noundef %2229) #11
  %2371 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2339, <4 x i32> noundef %2172, <4 x float> noundef %2269, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2359, i32 noundef 0, i32 noundef %2229) #11
  %2372 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2340, <4 x i32> noundef %2172, <4 x float> noundef %2270, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2359, i32 noundef 0, i32 noundef %2229) #11
  %2373 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2347, <4 x i32> noundef %2174, <4 x float> noundef %2371, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2359, i32 noundef 2, i32 noundef %2229) #11
  %2374 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2348, <4 x i32> noundef %2174, <4 x float> noundef %2372, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2359, i32 noundef 2, i32 noundef %2229) #11
  %2375 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2341, <4 x i32> noundef %2172, <4 x float> noundef %2273, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2362, i32 noundef 0, i32 noundef %2229) #11
  %2376 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2342, <4 x i32> noundef %2172, <4 x float> noundef %2274, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2362, i32 noundef 0, i32 noundef %2229) #11
  %2377 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2349, <4 x i32> noundef %2174, <4 x float> noundef %2375, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2362, i32 noundef 2, i32 noundef %2229) #11
  %2378 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2350, <4 x i32> noundef %2174, <4 x float> noundef %2376, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2362, i32 noundef 2, i32 noundef %2229) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2379 = or disjoint i32 %181, 45056
  %2380 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2379, i32 %75, i32 0)
  %2381 = or disjoint i32 %181, 46080
  %2382 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2381, i32 %75, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2383 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2335, <4 x i32> noundef %2191, <4 x float> noundef %2281, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2353, i32 noundef 1, i32 noundef %2229) #11
  %2384 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2336, <4 x i32> noundef %2191, <4 x float> noundef %2282, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2353, i32 noundef 1, i32 noundef %2229) #11
  %2385 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2343, <4 x i32> noundef %2192, <4 x float> noundef %2383, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2353, i32 noundef 3, i32 noundef %2229) #11
  %2386 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2344, <4 x i32> noundef %2192, <4 x float> noundef %2384, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2353, i32 noundef 3, i32 noundef %2229) #11
  %2387 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2337, <4 x i32> noundef %2191, <4 x float> noundef %2285, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2356, i32 noundef 1, i32 noundef %2229) #11
  %2388 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2338, <4 x i32> noundef %2191, <4 x float> noundef %2286, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2356, i32 noundef 1, i32 noundef %2229) #11
  %2389 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2345, <4 x i32> noundef %2192, <4 x float> noundef %2387, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2356, i32 noundef 3, i32 noundef %2229) #11
  %2390 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2346, <4 x i32> noundef %2192, <4 x float> noundef %2388, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2356, i32 noundef 3, i32 noundef %2229) #11
  %2391 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2339, <4 x i32> noundef %2191, <4 x float> noundef %2289, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2359, i32 noundef 1, i32 noundef %2229) #11
  %2392 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2340, <4 x i32> noundef %2191, <4 x float> noundef %2290, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2359, i32 noundef 1, i32 noundef %2229) #11
  %2393 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2347, <4 x i32> noundef %2192, <4 x float> noundef %2391, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2359, i32 noundef 3, i32 noundef %2229) #11
  %2394 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2348, <4 x i32> noundef %2192, <4 x float> noundef %2392, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2359, i32 noundef 3, i32 noundef %2229) #11
  %2395 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2341, <4 x i32> noundef %2191, <4 x float> noundef %2293, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2362, i32 noundef 1, i32 noundef %2229) #11
  %2396 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2342, <4 x i32> noundef %2191, <4 x float> noundef %2294, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2362, i32 noundef 1, i32 noundef %2229) #11
  %2397 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2349, <4 x i32> noundef %2192, <4 x float> noundef %2395, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2362, i32 noundef 3, i32 noundef %2229) #11
  %2398 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2350, <4 x i32> noundef %2192, <4 x float> noundef %2396, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2362, i32 noundef 3, i32 noundef %2229) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2399 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2379, i32 %79, i32 0)
  %2400 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2381, i32 %79, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2401 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2335, <4 x i32> noundef %2209, <4 x float> noundef %2299, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2353, i32 noundef 0, i32 noundef %2230) #11
  %2402 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2336, <4 x i32> noundef %2209, <4 x float> noundef %2300, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2353, i32 noundef 0, i32 noundef %2230) #11
  %2403 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2343, <4 x i32> noundef %2210, <4 x float> noundef %2401, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2353, i32 noundef 2, i32 noundef %2230) #11
  %2404 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2344, <4 x i32> noundef %2210, <4 x float> noundef %2402, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2353, i32 noundef 2, i32 noundef %2230) #11
  %2405 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2337, <4 x i32> noundef %2209, <4 x float> noundef %2303, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2356, i32 noundef 0, i32 noundef %2230) #11
  %2406 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2338, <4 x i32> noundef %2209, <4 x float> noundef %2304, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2356, i32 noundef 0, i32 noundef %2230) #11
  %2407 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2345, <4 x i32> noundef %2210, <4 x float> noundef %2405, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2356, i32 noundef 2, i32 noundef %2230) #11
  %2408 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2346, <4 x i32> noundef %2210, <4 x float> noundef %2406, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2356, i32 noundef 2, i32 noundef %2230) #11
  %2409 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2339, <4 x i32> noundef %2209, <4 x float> noundef %2307, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2359, i32 noundef 0, i32 noundef %2230) #11
  %2410 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2340, <4 x i32> noundef %2209, <4 x float> noundef %2308, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2359, i32 noundef 0, i32 noundef %2230) #11
  %2411 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2347, <4 x i32> noundef %2210, <4 x float> noundef %2409, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2359, i32 noundef 2, i32 noundef %2230) #11
  %2412 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2348, <4 x i32> noundef %2210, <4 x float> noundef %2410, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2359, i32 noundef 2, i32 noundef %2230) #11
  %2413 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2341, <4 x i32> noundef %2209, <4 x float> noundef %2311, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2362, i32 noundef 0, i32 noundef %2230) #11
  %2414 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2342, <4 x i32> noundef %2209, <4 x float> noundef %2312, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2362, i32 noundef 0, i32 noundef %2230) #11
  %2415 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2349, <4 x i32> noundef %2210, <4 x float> noundef %2413, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2362, i32 noundef 2, i32 noundef %2230) #11
  %2416 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2350, <4 x i32> noundef %2210, <4 x float> noundef %2414, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2362, i32 noundef 2, i32 noundef %2230) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2417 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2379, i32 %83, i32 0)
  %2418 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2381, i32 %83, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2419 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2335, <4 x i32> noundef %2227, <4 x float> noundef %2317, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2353, i32 noundef 1, i32 noundef %2230) #11
  %2420 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2336, <4 x i32> noundef %2227, <4 x float> noundef %2318, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2353, i32 noundef 1, i32 noundef %2230) #11
  %2421 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2343, <4 x i32> noundef %2228, <4 x float> noundef %2419, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2353, i32 noundef 3, i32 noundef %2230) #11
  %2422 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2344, <4 x i32> noundef %2228, <4 x float> noundef %2420, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2353, i32 noundef 3, i32 noundef %2230) #11
  %2423 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2337, <4 x i32> noundef %2227, <4 x float> noundef %2321, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2356, i32 noundef 1, i32 noundef %2230) #11
  %2424 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2338, <4 x i32> noundef %2227, <4 x float> noundef %2322, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2356, i32 noundef 1, i32 noundef %2230) #11
  %2425 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2345, <4 x i32> noundef %2228, <4 x float> noundef %2423, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2356, i32 noundef 3, i32 noundef %2230) #11
  %2426 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2346, <4 x i32> noundef %2228, <4 x float> noundef %2424, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2356, i32 noundef 3, i32 noundef %2230) #11
  %2427 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2339, <4 x i32> noundef %2227, <4 x float> noundef %2325, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2359, i32 noundef 1, i32 noundef %2230) #11
  %2428 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2340, <4 x i32> noundef %2227, <4 x float> noundef %2326, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2359, i32 noundef 1, i32 noundef %2230) #11
  %2429 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2347, <4 x i32> noundef %2228, <4 x float> noundef %2427, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2359, i32 noundef 3, i32 noundef %2230) #11
  %2430 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2348, <4 x i32> noundef %2228, <4 x float> noundef %2428, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2359, i32 noundef 3, i32 noundef %2230) #11
  %2431 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2341, <4 x i32> noundef %2227, <4 x float> noundef %2329, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2362, i32 noundef 1, i32 noundef %2230) #11
  %2432 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2342, <4 x i32> noundef %2227, <4 x float> noundef %2330, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2362, i32 noundef 1, i32 noundef %2230) #11
  %2433 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2349, <4 x i32> noundef %2228, <4 x float> noundef %2431, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2362, i32 noundef 3, i32 noundef %2230) #11
  %2434 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2350, <4 x i32> noundef %2228, <4 x float> noundef %2432, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2362, i32 noundef 3, i32 noundef %2230) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2435 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2379, i32 %87, i32 0)
  %2436 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2381, i32 %87, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2437 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %774, i32 %97, i32 0)
  %2438 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %774, i32 %102, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %2439 = load <4 x i32>, ptr addrspace(3) %345, align 16, !tbaa !11
  %2440 = load <4 x i32>, ptr addrspace(3) %347, align 16, !tbaa !11
  %2441 = load <4 x i32>, ptr addrspace(3) %349, align 16, !tbaa !11
  %2442 = load <4 x i32>, ptr addrspace(3) %351, align 16, !tbaa !11
  %2443 = load <4 x i32>, ptr addrspace(3) %353, align 16, !tbaa !11
  %2444 = load <4 x i32>, ptr addrspace(3) %355, align 16, !tbaa !11
  %2445 = load <4 x i32>, ptr addrspace(3) %357, align 16, !tbaa !11
  %2446 = load <4 x i32>, ptr addrspace(3) %359, align 16, !tbaa !11
  %2447 = load <4 x i32>, ptr addrspace(3) %361, align 16, !tbaa !11
  %2448 = load <4 x i32>, ptr addrspace(3) %363, align 16, !tbaa !11
  %2449 = load <4 x i32>, ptr addrspace(3) %365, align 16, !tbaa !11
  %2450 = load <4 x i32>, ptr addrspace(3) %367, align 16, !tbaa !11
  %2451 = load <4 x i32>, ptr addrspace(3) %369, align 16, !tbaa !11
  %2452 = load <4 x i32>, ptr addrspace(3) %371, align 16, !tbaa !11
  %2453 = load <4 x i32>, ptr addrspace(3) %373, align 16, !tbaa !11
  %2454 = load <4 x i32>, ptr addrspace(3) %375, align 16, !tbaa !11
  %2455 = or disjoint i32 %193, 5376
  %2456 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2455
  %2457 = load i32, ptr addrspace(3) %2456, align 4, !tbaa !7
  %2458 = or disjoint i32 %193, 12544
  %2459 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2458
  %2460 = load i32, ptr addrspace(3) %2459, align 4, !tbaa !7
  %2461 = or disjoint i32 %193, 19712
  %2462 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2461
  %2463 = load i32, ptr addrspace(3) %2462, align 4, !tbaa !7
  %2464 = or disjoint i32 %193, 26880
  %2465 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2464
  %2466 = load i32, ptr addrspace(3) %2465, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %196, i32 noundef 16, i32 noundef %162, i32 noundef 2944, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %197, i32 noundef 16, i32 noundef %167, i32 noundef 2944, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %198, i32 noundef 16, i32 noundef %171, i32 noundef 2944, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %199, i32 noundef 16, i32 noundef %175, i32 noundef 2944, i32 noundef 0, i32 noundef 0) #11
  %2467 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2439, <4 x i32> noundef %2276, <4 x float> noundef %2365, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2457, i32 noundef 0, i32 noundef %2333) #11
  %2468 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2440, <4 x i32> noundef %2276, <4 x float> noundef %2366, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2457, i32 noundef 0, i32 noundef %2333) #11
  %2469 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2447, <4 x i32> noundef %2278, <4 x float> noundef %2467, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2457, i32 noundef 2, i32 noundef %2333) #11
  %2470 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2448, <4 x i32> noundef %2278, <4 x float> noundef %2468, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2457, i32 noundef 2, i32 noundef %2333) #11
  %2471 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2441, <4 x i32> noundef %2276, <4 x float> noundef %2369, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2460, i32 noundef 0, i32 noundef %2333) #11
  %2472 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2442, <4 x i32> noundef %2276, <4 x float> noundef %2370, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2460, i32 noundef 0, i32 noundef %2333) #11
  %2473 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2449, <4 x i32> noundef %2278, <4 x float> noundef %2471, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2460, i32 noundef 2, i32 noundef %2333) #11
  %2474 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2450, <4 x i32> noundef %2278, <4 x float> noundef %2472, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2460, i32 noundef 2, i32 noundef %2333) #11
  %2475 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2443, <4 x i32> noundef %2276, <4 x float> noundef %2373, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2463, i32 noundef 0, i32 noundef %2333) #11
  %2476 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2444, <4 x i32> noundef %2276, <4 x float> noundef %2374, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2463, i32 noundef 0, i32 noundef %2333) #11
  %2477 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2451, <4 x i32> noundef %2278, <4 x float> noundef %2475, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2463, i32 noundef 2, i32 noundef %2333) #11
  %2478 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2452, <4 x i32> noundef %2278, <4 x float> noundef %2476, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2463, i32 noundef 2, i32 noundef %2333) #11
  %2479 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2445, <4 x i32> noundef %2276, <4 x float> noundef %2377, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2466, i32 noundef 0, i32 noundef %2333) #11
  %2480 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2446, <4 x i32> noundef %2276, <4 x float> noundef %2378, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2466, i32 noundef 0, i32 noundef %2333) #11
  %2481 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2453, <4 x i32> noundef %2278, <4 x float> noundef %2479, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2466, i32 noundef 2, i32 noundef %2333) #11
  %2482 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2454, <4 x i32> noundef %2278, <4 x float> noundef %2480, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2466, i32 noundef 2, i32 noundef %2333) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2483 = or disjoint i32 %181, 47104
  %2484 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2483, i32 %75, i32 0)
  %2485 = or disjoint i32 %181, 48128
  %2486 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2485, i32 %75, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2487 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2439, <4 x i32> noundef %2295, <4 x float> noundef %2385, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2457, i32 noundef 1, i32 noundef %2333) #11
  %2488 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2440, <4 x i32> noundef %2295, <4 x float> noundef %2386, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2457, i32 noundef 1, i32 noundef %2333) #11
  %2489 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2447, <4 x i32> noundef %2296, <4 x float> noundef %2487, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2457, i32 noundef 3, i32 noundef %2333) #11
  %2490 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2448, <4 x i32> noundef %2296, <4 x float> noundef %2488, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2457, i32 noundef 3, i32 noundef %2333) #11
  %2491 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2441, <4 x i32> noundef %2295, <4 x float> noundef %2389, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2460, i32 noundef 1, i32 noundef %2333) #11
  %2492 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2442, <4 x i32> noundef %2295, <4 x float> noundef %2390, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2460, i32 noundef 1, i32 noundef %2333) #11
  %2493 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2449, <4 x i32> noundef %2296, <4 x float> noundef %2491, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2460, i32 noundef 3, i32 noundef %2333) #11
  %2494 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2450, <4 x i32> noundef %2296, <4 x float> noundef %2492, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2460, i32 noundef 3, i32 noundef %2333) #11
  %2495 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2443, <4 x i32> noundef %2295, <4 x float> noundef %2393, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2463, i32 noundef 1, i32 noundef %2333) #11
  %2496 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2444, <4 x i32> noundef %2295, <4 x float> noundef %2394, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2463, i32 noundef 1, i32 noundef %2333) #11
  %2497 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2451, <4 x i32> noundef %2296, <4 x float> noundef %2495, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2463, i32 noundef 3, i32 noundef %2333) #11
  %2498 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2452, <4 x i32> noundef %2296, <4 x float> noundef %2496, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2463, i32 noundef 3, i32 noundef %2333) #11
  %2499 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2445, <4 x i32> noundef %2295, <4 x float> noundef %2397, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2466, i32 noundef 1, i32 noundef %2333) #11
  %2500 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2446, <4 x i32> noundef %2295, <4 x float> noundef %2398, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2466, i32 noundef 1, i32 noundef %2333) #11
  %2501 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2453, <4 x i32> noundef %2296, <4 x float> noundef %2499, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2466, i32 noundef 3, i32 noundef %2333) #11
  %2502 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2454, <4 x i32> noundef %2296, <4 x float> noundef %2500, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2466, i32 noundef 3, i32 noundef %2333) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2503 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2483, i32 %79, i32 0)
  %2504 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2485, i32 %79, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2505 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2439, <4 x i32> noundef %2313, <4 x float> noundef %2403, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2457, i32 noundef 0, i32 noundef %2334) #11
  %2506 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2440, <4 x i32> noundef %2313, <4 x float> noundef %2404, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2457, i32 noundef 0, i32 noundef %2334) #11
  %2507 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2447, <4 x i32> noundef %2314, <4 x float> noundef %2505, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2457, i32 noundef 2, i32 noundef %2334) #11
  %2508 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2448, <4 x i32> noundef %2314, <4 x float> noundef %2506, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2457, i32 noundef 2, i32 noundef %2334) #11
  %2509 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2441, <4 x i32> noundef %2313, <4 x float> noundef %2407, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2460, i32 noundef 0, i32 noundef %2334) #11
  %2510 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2442, <4 x i32> noundef %2313, <4 x float> noundef %2408, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2460, i32 noundef 0, i32 noundef %2334) #11
  %2511 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2449, <4 x i32> noundef %2314, <4 x float> noundef %2509, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2460, i32 noundef 2, i32 noundef %2334) #11
  %2512 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2450, <4 x i32> noundef %2314, <4 x float> noundef %2510, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2460, i32 noundef 2, i32 noundef %2334) #11
  %2513 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2443, <4 x i32> noundef %2313, <4 x float> noundef %2411, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2463, i32 noundef 0, i32 noundef %2334) #11
  %2514 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2444, <4 x i32> noundef %2313, <4 x float> noundef %2412, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2463, i32 noundef 0, i32 noundef %2334) #11
  %2515 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2451, <4 x i32> noundef %2314, <4 x float> noundef %2513, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2463, i32 noundef 2, i32 noundef %2334) #11
  %2516 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2452, <4 x i32> noundef %2314, <4 x float> noundef %2514, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2463, i32 noundef 2, i32 noundef %2334) #11
  %2517 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2445, <4 x i32> noundef %2313, <4 x float> noundef %2415, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2466, i32 noundef 0, i32 noundef %2334) #11
  %2518 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2446, <4 x i32> noundef %2313, <4 x float> noundef %2416, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2466, i32 noundef 0, i32 noundef %2334) #11
  %2519 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2453, <4 x i32> noundef %2314, <4 x float> noundef %2517, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2466, i32 noundef 2, i32 noundef %2334) #11
  %2520 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2454, <4 x i32> noundef %2314, <4 x float> noundef %2518, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2466, i32 noundef 2, i32 noundef %2334) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2521 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2483, i32 %83, i32 0)
  %2522 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2485, i32 %83, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2523 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2439, <4 x i32> noundef %2331, <4 x float> noundef %2421, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2457, i32 noundef 1, i32 noundef %2334) #11
  %2524 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2440, <4 x i32> noundef %2331, <4 x float> noundef %2422, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2457, i32 noundef 1, i32 noundef %2334) #11
  %2525 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2447, <4 x i32> noundef %2332, <4 x float> noundef %2523, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2457, i32 noundef 3, i32 noundef %2334) #11
  %2526 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2448, <4 x i32> noundef %2332, <4 x float> noundef %2524, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2457, i32 noundef 3, i32 noundef %2334) #11
  %2527 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2441, <4 x i32> noundef %2331, <4 x float> noundef %2425, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2460, i32 noundef 1, i32 noundef %2334) #11
  %2528 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2442, <4 x i32> noundef %2331, <4 x float> noundef %2426, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2460, i32 noundef 1, i32 noundef %2334) #11
  %2529 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2449, <4 x i32> noundef %2332, <4 x float> noundef %2527, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2460, i32 noundef 3, i32 noundef %2334) #11
  %2530 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2450, <4 x i32> noundef %2332, <4 x float> noundef %2528, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2460, i32 noundef 3, i32 noundef %2334) #11
  %2531 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2443, <4 x i32> noundef %2331, <4 x float> noundef %2429, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2463, i32 noundef 1, i32 noundef %2334) #11
  %2532 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2444, <4 x i32> noundef %2331, <4 x float> noundef %2430, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2463, i32 noundef 1, i32 noundef %2334) #11
  %2533 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2451, <4 x i32> noundef %2332, <4 x float> noundef %2531, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2463, i32 noundef 3, i32 noundef %2334) #11
  %2534 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2452, <4 x i32> noundef %2332, <4 x float> noundef %2532, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2463, i32 noundef 3, i32 noundef %2334) #11
  %2535 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2445, <4 x i32> noundef %2331, <4 x float> noundef %2433, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2466, i32 noundef 1, i32 noundef %2334) #11
  %2536 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2446, <4 x i32> noundef %2331, <4 x float> noundef %2434, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2466, i32 noundef 1, i32 noundef %2334) #11
  %2537 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2453, <4 x i32> noundef %2332, <4 x float> noundef %2535, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2466, i32 noundef 3, i32 noundef %2334) #11
  %2538 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2454, <4 x i32> noundef %2332, <4 x float> noundef %2536, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2466, i32 noundef 3, i32 noundef %2334) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2539 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2483, i32 %87, i32 0)
  %2540 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2485, i32 %87, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2541 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %878, i32 %97, i32 0)
  %2542 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %878, i32 %102, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %2543 = load <4 x i32>, ptr addrspace(3) %216, align 16, !tbaa !11
  %2544 = load <4 x i32>, ptr addrspace(3) %219, align 16, !tbaa !11
  %2545 = load <4 x i32>, ptr addrspace(3) %222, align 16, !tbaa !11
  %2546 = load <4 x i32>, ptr addrspace(3) %225, align 16, !tbaa !11
  %2547 = load <4 x i32>, ptr addrspace(3) %228, align 16, !tbaa !11
  %2548 = load <4 x i32>, ptr addrspace(3) %231, align 16, !tbaa !11
  %2549 = load <4 x i32>, ptr addrspace(3) %234, align 16, !tbaa !11
  %2550 = load <4 x i32>, ptr addrspace(3) %237, align 16, !tbaa !11
  %2551 = load <4 x i32>, ptr addrspace(3) %241, align 16, !tbaa !11
  %2552 = load <4 x i32>, ptr addrspace(3) %243, align 16, !tbaa !11
  %2553 = load <4 x i32>, ptr addrspace(3) %245, align 16, !tbaa !11
  %2554 = load <4 x i32>, ptr addrspace(3) %247, align 16, !tbaa !11
  %2555 = load <4 x i32>, ptr addrspace(3) %249, align 16, !tbaa !11
  %2556 = load <4 x i32>, ptr addrspace(3) %251, align 16, !tbaa !11
  %2557 = load <4 x i32>, ptr addrspace(3) %253, align 16, !tbaa !11
  %2558 = load <4 x i32>, ptr addrspace(3) %255, align 16, !tbaa !11
  %2559 = or disjoint i32 %193, 5632
  %2560 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2559
  %2561 = load i32, ptr addrspace(3) %2560, align 4, !tbaa !7
  %2562 = or disjoint i32 %193, 12800
  %2563 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2562
  %2564 = load i32, ptr addrspace(3) %2563, align 4, !tbaa !7
  %2565 = or disjoint i32 %193, 19968
  %2566 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2565
  %2567 = load i32, ptr addrspace(3) %2566, align 4, !tbaa !7
  %2568 = or disjoint i32 %193, 27136
  %2569 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2568
  %2570 = load i32, ptr addrspace(3) %2569, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef %163, i32 noundef 16, i32 noundef %162, i32 noundef 3072, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %168, i32 noundef 16, i32 noundef %167, i32 noundef 3072, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %172, i32 noundef 16, i32 noundef %171, i32 noundef 3072, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %176, i32 noundef 16, i32 noundef %175, i32 noundef 3072, i32 noundef 0, i32 noundef 0) #11
  %2571 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2543, <4 x i32> noundef %2380, <4 x float> noundef %2469, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2561, i32 noundef 0, i32 noundef %2437) #11
  %2572 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2544, <4 x i32> noundef %2380, <4 x float> noundef %2470, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2561, i32 noundef 0, i32 noundef %2437) #11
  %2573 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2551, <4 x i32> noundef %2382, <4 x float> noundef %2571, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2561, i32 noundef 2, i32 noundef %2437) #11
  %2574 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2552, <4 x i32> noundef %2382, <4 x float> noundef %2572, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2561, i32 noundef 2, i32 noundef %2437) #11
  %2575 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2545, <4 x i32> noundef %2380, <4 x float> noundef %2473, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2564, i32 noundef 0, i32 noundef %2437) #11
  %2576 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2546, <4 x i32> noundef %2380, <4 x float> noundef %2474, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2564, i32 noundef 0, i32 noundef %2437) #11
  %2577 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2553, <4 x i32> noundef %2382, <4 x float> noundef %2575, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2564, i32 noundef 2, i32 noundef %2437) #11
  %2578 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2554, <4 x i32> noundef %2382, <4 x float> noundef %2576, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2564, i32 noundef 2, i32 noundef %2437) #11
  %2579 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2547, <4 x i32> noundef %2380, <4 x float> noundef %2477, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2567, i32 noundef 0, i32 noundef %2437) #11
  %2580 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2548, <4 x i32> noundef %2380, <4 x float> noundef %2478, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2567, i32 noundef 0, i32 noundef %2437) #11
  %2581 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2555, <4 x i32> noundef %2382, <4 x float> noundef %2579, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2567, i32 noundef 2, i32 noundef %2437) #11
  %2582 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2556, <4 x i32> noundef %2382, <4 x float> noundef %2580, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2567, i32 noundef 2, i32 noundef %2437) #11
  %2583 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2549, <4 x i32> noundef %2380, <4 x float> noundef %2481, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2570, i32 noundef 0, i32 noundef %2437) #11
  %2584 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2550, <4 x i32> noundef %2380, <4 x float> noundef %2482, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2570, i32 noundef 0, i32 noundef %2437) #11
  %2585 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2557, <4 x i32> noundef %2382, <4 x float> noundef %2583, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2570, i32 noundef 2, i32 noundef %2437) #11
  %2586 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2558, <4 x i32> noundef %2382, <4 x float> noundef %2584, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2570, i32 noundef 2, i32 noundef %2437) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2587 = or disjoint i32 %181, 49152
  %2588 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2587, i32 %75, i32 0)
  %2589 = or disjoint i32 %181, 50176
  %2590 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2589, i32 %75, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2591 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2543, <4 x i32> noundef %2399, <4 x float> noundef %2489, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2561, i32 noundef 1, i32 noundef %2437) #11
  %2592 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2544, <4 x i32> noundef %2399, <4 x float> noundef %2490, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2561, i32 noundef 1, i32 noundef %2437) #11
  %2593 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2551, <4 x i32> noundef %2400, <4 x float> noundef %2591, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2561, i32 noundef 3, i32 noundef %2437) #11
  %2594 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2552, <4 x i32> noundef %2400, <4 x float> noundef %2592, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2561, i32 noundef 3, i32 noundef %2437) #11
  %2595 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2545, <4 x i32> noundef %2399, <4 x float> noundef %2493, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2564, i32 noundef 1, i32 noundef %2437) #11
  %2596 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2546, <4 x i32> noundef %2399, <4 x float> noundef %2494, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2564, i32 noundef 1, i32 noundef %2437) #11
  %2597 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2553, <4 x i32> noundef %2400, <4 x float> noundef %2595, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2564, i32 noundef 3, i32 noundef %2437) #11
  %2598 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2554, <4 x i32> noundef %2400, <4 x float> noundef %2596, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2564, i32 noundef 3, i32 noundef %2437) #11
  %2599 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2547, <4 x i32> noundef %2399, <4 x float> noundef %2497, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2567, i32 noundef 1, i32 noundef %2437) #11
  %2600 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2548, <4 x i32> noundef %2399, <4 x float> noundef %2498, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2567, i32 noundef 1, i32 noundef %2437) #11
  %2601 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2555, <4 x i32> noundef %2400, <4 x float> noundef %2599, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2567, i32 noundef 3, i32 noundef %2437) #11
  %2602 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2556, <4 x i32> noundef %2400, <4 x float> noundef %2600, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2567, i32 noundef 3, i32 noundef %2437) #11
  %2603 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2549, <4 x i32> noundef %2399, <4 x float> noundef %2501, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2570, i32 noundef 1, i32 noundef %2437) #11
  %2604 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2550, <4 x i32> noundef %2399, <4 x float> noundef %2502, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2570, i32 noundef 1, i32 noundef %2437) #11
  %2605 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2557, <4 x i32> noundef %2400, <4 x float> noundef %2603, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2570, i32 noundef 3, i32 noundef %2437) #11
  %2606 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2558, <4 x i32> noundef %2400, <4 x float> noundef %2604, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2570, i32 noundef 3, i32 noundef %2437) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2607 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2587, i32 %79, i32 0)
  %2608 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2589, i32 %79, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2609 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2543, <4 x i32> noundef %2417, <4 x float> noundef %2507, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2561, i32 noundef 0, i32 noundef %2438) #11
  %2610 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2544, <4 x i32> noundef %2417, <4 x float> noundef %2508, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2561, i32 noundef 0, i32 noundef %2438) #11
  %2611 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2551, <4 x i32> noundef %2418, <4 x float> noundef %2609, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2561, i32 noundef 2, i32 noundef %2438) #11
  %2612 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2552, <4 x i32> noundef %2418, <4 x float> noundef %2610, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2561, i32 noundef 2, i32 noundef %2438) #11
  %2613 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2545, <4 x i32> noundef %2417, <4 x float> noundef %2511, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2564, i32 noundef 0, i32 noundef %2438) #11
  %2614 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2546, <4 x i32> noundef %2417, <4 x float> noundef %2512, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2564, i32 noundef 0, i32 noundef %2438) #11
  %2615 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2553, <4 x i32> noundef %2418, <4 x float> noundef %2613, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2564, i32 noundef 2, i32 noundef %2438) #11
  %2616 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2554, <4 x i32> noundef %2418, <4 x float> noundef %2614, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2564, i32 noundef 2, i32 noundef %2438) #11
  %2617 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2547, <4 x i32> noundef %2417, <4 x float> noundef %2515, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2567, i32 noundef 0, i32 noundef %2438) #11
  %2618 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2548, <4 x i32> noundef %2417, <4 x float> noundef %2516, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2567, i32 noundef 0, i32 noundef %2438) #11
  %2619 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2555, <4 x i32> noundef %2418, <4 x float> noundef %2617, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2567, i32 noundef 2, i32 noundef %2438) #11
  %2620 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2556, <4 x i32> noundef %2418, <4 x float> noundef %2618, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2567, i32 noundef 2, i32 noundef %2438) #11
  %2621 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2549, <4 x i32> noundef %2417, <4 x float> noundef %2519, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2570, i32 noundef 0, i32 noundef %2438) #11
  %2622 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2550, <4 x i32> noundef %2417, <4 x float> noundef %2520, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2570, i32 noundef 0, i32 noundef %2438) #11
  %2623 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2557, <4 x i32> noundef %2418, <4 x float> noundef %2621, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2570, i32 noundef 2, i32 noundef %2438) #11
  %2624 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2558, <4 x i32> noundef %2418, <4 x float> noundef %2622, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2570, i32 noundef 2, i32 noundef %2438) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2625 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2587, i32 %83, i32 0)
  %2626 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2589, i32 %83, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2627 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2543, <4 x i32> noundef %2435, <4 x float> noundef %2525, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2561, i32 noundef 1, i32 noundef %2438) #11
  %2628 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2544, <4 x i32> noundef %2435, <4 x float> noundef %2526, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2561, i32 noundef 1, i32 noundef %2438) #11
  %2629 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2551, <4 x i32> noundef %2436, <4 x float> noundef %2627, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2561, i32 noundef 3, i32 noundef %2438) #11
  %2630 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2552, <4 x i32> noundef %2436, <4 x float> noundef %2628, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2561, i32 noundef 3, i32 noundef %2438) #11
  %2631 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2545, <4 x i32> noundef %2435, <4 x float> noundef %2529, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2564, i32 noundef 1, i32 noundef %2438) #11
  %2632 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2546, <4 x i32> noundef %2435, <4 x float> noundef %2530, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2564, i32 noundef 1, i32 noundef %2438) #11
  %2633 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2553, <4 x i32> noundef %2436, <4 x float> noundef %2631, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2564, i32 noundef 3, i32 noundef %2438) #11
  %2634 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2554, <4 x i32> noundef %2436, <4 x float> noundef %2632, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2564, i32 noundef 3, i32 noundef %2438) #11
  %2635 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2547, <4 x i32> noundef %2435, <4 x float> noundef %2533, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2567, i32 noundef 1, i32 noundef %2438) #11
  %2636 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2548, <4 x i32> noundef %2435, <4 x float> noundef %2534, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2567, i32 noundef 1, i32 noundef %2438) #11
  %2637 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2555, <4 x i32> noundef %2436, <4 x float> noundef %2635, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2567, i32 noundef 3, i32 noundef %2438) #11
  %2638 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2556, <4 x i32> noundef %2436, <4 x float> noundef %2636, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2567, i32 noundef 3, i32 noundef %2438) #11
  %2639 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2549, <4 x i32> noundef %2435, <4 x float> noundef %2537, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2570, i32 noundef 1, i32 noundef %2438) #11
  %2640 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2550, <4 x i32> noundef %2435, <4 x float> noundef %2538, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2570, i32 noundef 1, i32 noundef %2438) #11
  %2641 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2557, <4 x i32> noundef %2436, <4 x float> noundef %2639, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2570, i32 noundef 3, i32 noundef %2438) #11
  %2642 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2558, <4 x i32> noundef %2436, <4 x float> noundef %2640, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2570, i32 noundef 3, i32 noundef %2438) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2643 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2587, i32 %87, i32 0)
  %2644 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2589, i32 %87, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2645 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %982, i32 %97, i32 0)
  %2646 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %982, i32 %102, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %2647 = load <4 x i32>, ptr addrspace(3) %345, align 16, !tbaa !11
  %2648 = load <4 x i32>, ptr addrspace(3) %347, align 16, !tbaa !11
  %2649 = load <4 x i32>, ptr addrspace(3) %349, align 16, !tbaa !11
  %2650 = load <4 x i32>, ptr addrspace(3) %351, align 16, !tbaa !11
  %2651 = load <4 x i32>, ptr addrspace(3) %353, align 16, !tbaa !11
  %2652 = load <4 x i32>, ptr addrspace(3) %355, align 16, !tbaa !11
  %2653 = load <4 x i32>, ptr addrspace(3) %357, align 16, !tbaa !11
  %2654 = load <4 x i32>, ptr addrspace(3) %359, align 16, !tbaa !11
  %2655 = load <4 x i32>, ptr addrspace(3) %361, align 16, !tbaa !11
  %2656 = load <4 x i32>, ptr addrspace(3) %363, align 16, !tbaa !11
  %2657 = load <4 x i32>, ptr addrspace(3) %365, align 16, !tbaa !11
  %2658 = load <4 x i32>, ptr addrspace(3) %367, align 16, !tbaa !11
  %2659 = load <4 x i32>, ptr addrspace(3) %369, align 16, !tbaa !11
  %2660 = load <4 x i32>, ptr addrspace(3) %371, align 16, !tbaa !11
  %2661 = load <4 x i32>, ptr addrspace(3) %373, align 16, !tbaa !11
  %2662 = load <4 x i32>, ptr addrspace(3) %375, align 16, !tbaa !11
  %2663 = or disjoint i32 %193, 5888
  %2664 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2663
  %2665 = load i32, ptr addrspace(3) %2664, align 4, !tbaa !7
  %2666 = or disjoint i32 %193, 13056
  %2667 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2666
  %2668 = load i32, ptr addrspace(3) %2667, align 4, !tbaa !7
  %2669 = or disjoint i32 %193, 20224
  %2670 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2669
  %2671 = load i32, ptr addrspace(3) %2670, align 4, !tbaa !7
  %2672 = or disjoint i32 %193, 27392
  %2673 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2672
  %2674 = load i32, ptr addrspace(3) %2673, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %196, i32 noundef 16, i32 noundef %162, i32 noundef 3200, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %197, i32 noundef 16, i32 noundef %167, i32 noundef 3200, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %198, i32 noundef 16, i32 noundef %171, i32 noundef 3200, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %199, i32 noundef 16, i32 noundef %175, i32 noundef 3200, i32 noundef 0, i32 noundef 0) #11
  %2675 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2647, <4 x i32> noundef %2484, <4 x float> noundef %2573, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2665, i32 noundef 0, i32 noundef %2541) #11
  %2676 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2648, <4 x i32> noundef %2484, <4 x float> noundef %2574, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2665, i32 noundef 0, i32 noundef %2541) #11
  %2677 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2655, <4 x i32> noundef %2486, <4 x float> noundef %2675, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2665, i32 noundef 2, i32 noundef %2541) #11
  %2678 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2656, <4 x i32> noundef %2486, <4 x float> noundef %2676, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2665, i32 noundef 2, i32 noundef %2541) #11
  %2679 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2649, <4 x i32> noundef %2484, <4 x float> noundef %2577, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2668, i32 noundef 0, i32 noundef %2541) #11
  %2680 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2650, <4 x i32> noundef %2484, <4 x float> noundef %2578, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2668, i32 noundef 0, i32 noundef %2541) #11
  %2681 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2657, <4 x i32> noundef %2486, <4 x float> noundef %2679, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2668, i32 noundef 2, i32 noundef %2541) #11
  %2682 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2658, <4 x i32> noundef %2486, <4 x float> noundef %2680, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2668, i32 noundef 2, i32 noundef %2541) #11
  %2683 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2651, <4 x i32> noundef %2484, <4 x float> noundef %2581, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2671, i32 noundef 0, i32 noundef %2541) #11
  %2684 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2652, <4 x i32> noundef %2484, <4 x float> noundef %2582, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2671, i32 noundef 0, i32 noundef %2541) #11
  %2685 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2659, <4 x i32> noundef %2486, <4 x float> noundef %2683, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2671, i32 noundef 2, i32 noundef %2541) #11
  %2686 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2660, <4 x i32> noundef %2486, <4 x float> noundef %2684, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2671, i32 noundef 2, i32 noundef %2541) #11
  %2687 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2653, <4 x i32> noundef %2484, <4 x float> noundef %2585, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2674, i32 noundef 0, i32 noundef %2541) #11
  %2688 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2654, <4 x i32> noundef %2484, <4 x float> noundef %2586, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2674, i32 noundef 0, i32 noundef %2541) #11
  %2689 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2661, <4 x i32> noundef %2486, <4 x float> noundef %2687, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2674, i32 noundef 2, i32 noundef %2541) #11
  %2690 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2662, <4 x i32> noundef %2486, <4 x float> noundef %2688, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2674, i32 noundef 2, i32 noundef %2541) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2691 = or disjoint i32 %181, 51200
  %2692 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2691, i32 %75, i32 0)
  %2693 = or disjoint i32 %181, 52224
  %2694 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2693, i32 %75, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2695 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2647, <4 x i32> noundef %2503, <4 x float> noundef %2593, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2665, i32 noundef 1, i32 noundef %2541) #11
  %2696 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2648, <4 x i32> noundef %2503, <4 x float> noundef %2594, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2665, i32 noundef 1, i32 noundef %2541) #11
  %2697 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2655, <4 x i32> noundef %2504, <4 x float> noundef %2695, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2665, i32 noundef 3, i32 noundef %2541) #11
  %2698 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2656, <4 x i32> noundef %2504, <4 x float> noundef %2696, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2665, i32 noundef 3, i32 noundef %2541) #11
  %2699 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2649, <4 x i32> noundef %2503, <4 x float> noundef %2597, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2668, i32 noundef 1, i32 noundef %2541) #11
  %2700 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2650, <4 x i32> noundef %2503, <4 x float> noundef %2598, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2668, i32 noundef 1, i32 noundef %2541) #11
  %2701 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2657, <4 x i32> noundef %2504, <4 x float> noundef %2699, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2668, i32 noundef 3, i32 noundef %2541) #11
  %2702 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2658, <4 x i32> noundef %2504, <4 x float> noundef %2700, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2668, i32 noundef 3, i32 noundef %2541) #11
  %2703 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2651, <4 x i32> noundef %2503, <4 x float> noundef %2601, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2671, i32 noundef 1, i32 noundef %2541) #11
  %2704 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2652, <4 x i32> noundef %2503, <4 x float> noundef %2602, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2671, i32 noundef 1, i32 noundef %2541) #11
  %2705 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2659, <4 x i32> noundef %2504, <4 x float> noundef %2703, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2671, i32 noundef 3, i32 noundef %2541) #11
  %2706 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2660, <4 x i32> noundef %2504, <4 x float> noundef %2704, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2671, i32 noundef 3, i32 noundef %2541) #11
  %2707 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2653, <4 x i32> noundef %2503, <4 x float> noundef %2605, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2674, i32 noundef 1, i32 noundef %2541) #11
  %2708 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2654, <4 x i32> noundef %2503, <4 x float> noundef %2606, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2674, i32 noundef 1, i32 noundef %2541) #11
  %2709 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2661, <4 x i32> noundef %2504, <4 x float> noundef %2707, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2674, i32 noundef 3, i32 noundef %2541) #11
  %2710 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2662, <4 x i32> noundef %2504, <4 x float> noundef %2708, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2674, i32 noundef 3, i32 noundef %2541) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2711 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2691, i32 %79, i32 0)
  %2712 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2693, i32 %79, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2713 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2647, <4 x i32> noundef %2521, <4 x float> noundef %2611, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2665, i32 noundef 0, i32 noundef %2542) #11
  %2714 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2648, <4 x i32> noundef %2521, <4 x float> noundef %2612, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2665, i32 noundef 0, i32 noundef %2542) #11
  %2715 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2655, <4 x i32> noundef %2522, <4 x float> noundef %2713, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2665, i32 noundef 2, i32 noundef %2542) #11
  %2716 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2656, <4 x i32> noundef %2522, <4 x float> noundef %2714, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2665, i32 noundef 2, i32 noundef %2542) #11
  %2717 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2649, <4 x i32> noundef %2521, <4 x float> noundef %2615, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2668, i32 noundef 0, i32 noundef %2542) #11
  %2718 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2650, <4 x i32> noundef %2521, <4 x float> noundef %2616, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2668, i32 noundef 0, i32 noundef %2542) #11
  %2719 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2657, <4 x i32> noundef %2522, <4 x float> noundef %2717, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2668, i32 noundef 2, i32 noundef %2542) #11
  %2720 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2658, <4 x i32> noundef %2522, <4 x float> noundef %2718, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2668, i32 noundef 2, i32 noundef %2542) #11
  %2721 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2651, <4 x i32> noundef %2521, <4 x float> noundef %2619, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2671, i32 noundef 0, i32 noundef %2542) #11
  %2722 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2652, <4 x i32> noundef %2521, <4 x float> noundef %2620, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2671, i32 noundef 0, i32 noundef %2542) #11
  %2723 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2659, <4 x i32> noundef %2522, <4 x float> noundef %2721, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2671, i32 noundef 2, i32 noundef %2542) #11
  %2724 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2660, <4 x i32> noundef %2522, <4 x float> noundef %2722, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2671, i32 noundef 2, i32 noundef %2542) #11
  %2725 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2653, <4 x i32> noundef %2521, <4 x float> noundef %2623, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2674, i32 noundef 0, i32 noundef %2542) #11
  %2726 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2654, <4 x i32> noundef %2521, <4 x float> noundef %2624, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2674, i32 noundef 0, i32 noundef %2542) #11
  %2727 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2661, <4 x i32> noundef %2522, <4 x float> noundef %2725, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2674, i32 noundef 2, i32 noundef %2542) #11
  %2728 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2662, <4 x i32> noundef %2522, <4 x float> noundef %2726, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2674, i32 noundef 2, i32 noundef %2542) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2729 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2691, i32 %83, i32 0)
  %2730 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2693, i32 %83, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2731 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2647, <4 x i32> noundef %2539, <4 x float> noundef %2629, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2665, i32 noundef 1, i32 noundef %2542) #11
  %2732 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2648, <4 x i32> noundef %2539, <4 x float> noundef %2630, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2665, i32 noundef 1, i32 noundef %2542) #11
  %2733 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2655, <4 x i32> noundef %2540, <4 x float> noundef %2731, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2665, i32 noundef 3, i32 noundef %2542) #11
  %2734 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2656, <4 x i32> noundef %2540, <4 x float> noundef %2732, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2665, i32 noundef 3, i32 noundef %2542) #11
  %2735 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2649, <4 x i32> noundef %2539, <4 x float> noundef %2633, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2668, i32 noundef 1, i32 noundef %2542) #11
  %2736 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2650, <4 x i32> noundef %2539, <4 x float> noundef %2634, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2668, i32 noundef 1, i32 noundef %2542) #11
  %2737 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2657, <4 x i32> noundef %2540, <4 x float> noundef %2735, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2668, i32 noundef 3, i32 noundef %2542) #11
  %2738 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2658, <4 x i32> noundef %2540, <4 x float> noundef %2736, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2668, i32 noundef 3, i32 noundef %2542) #11
  %2739 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2651, <4 x i32> noundef %2539, <4 x float> noundef %2637, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2671, i32 noundef 1, i32 noundef %2542) #11
  %2740 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2652, <4 x i32> noundef %2539, <4 x float> noundef %2638, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2671, i32 noundef 1, i32 noundef %2542) #11
  %2741 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2659, <4 x i32> noundef %2540, <4 x float> noundef %2739, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2671, i32 noundef 3, i32 noundef %2542) #11
  %2742 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2660, <4 x i32> noundef %2540, <4 x float> noundef %2740, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2671, i32 noundef 3, i32 noundef %2542) #11
  %2743 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2653, <4 x i32> noundef %2539, <4 x float> noundef %2641, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2674, i32 noundef 1, i32 noundef %2542) #11
  %2744 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2654, <4 x i32> noundef %2539, <4 x float> noundef %2642, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2674, i32 noundef 1, i32 noundef %2542) #11
  %2745 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2661, <4 x i32> noundef %2540, <4 x float> noundef %2743, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2674, i32 noundef 3, i32 noundef %2542) #11
  %2746 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2662, <4 x i32> noundef %2540, <4 x float> noundef %2744, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2674, i32 noundef 3, i32 noundef %2542) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2747 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2691, i32 %87, i32 0)
  %2748 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2693, i32 %87, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2749 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %1086, i32 %97, i32 0)
  %2750 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %1086, i32 %102, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %2751 = load <4 x i32>, ptr addrspace(3) %216, align 16, !tbaa !11
  %2752 = load <4 x i32>, ptr addrspace(3) %219, align 16, !tbaa !11
  %2753 = load <4 x i32>, ptr addrspace(3) %222, align 16, !tbaa !11
  %2754 = load <4 x i32>, ptr addrspace(3) %225, align 16, !tbaa !11
  %2755 = load <4 x i32>, ptr addrspace(3) %228, align 16, !tbaa !11
  %2756 = load <4 x i32>, ptr addrspace(3) %231, align 16, !tbaa !11
  %2757 = load <4 x i32>, ptr addrspace(3) %234, align 16, !tbaa !11
  %2758 = load <4 x i32>, ptr addrspace(3) %237, align 16, !tbaa !11
  %2759 = load <4 x i32>, ptr addrspace(3) %241, align 16, !tbaa !11
  %2760 = load <4 x i32>, ptr addrspace(3) %243, align 16, !tbaa !11
  %2761 = load <4 x i32>, ptr addrspace(3) %245, align 16, !tbaa !11
  %2762 = load <4 x i32>, ptr addrspace(3) %247, align 16, !tbaa !11
  %2763 = load <4 x i32>, ptr addrspace(3) %249, align 16, !tbaa !11
  %2764 = load <4 x i32>, ptr addrspace(3) %251, align 16, !tbaa !11
  %2765 = load <4 x i32>, ptr addrspace(3) %253, align 16, !tbaa !11
  %2766 = load <4 x i32>, ptr addrspace(3) %255, align 16, !tbaa !11
  %2767 = or disjoint i32 %193, 6144
  %2768 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2767
  %2769 = load i32, ptr addrspace(3) %2768, align 4, !tbaa !7
  %2770 = or disjoint i32 %193, 13312
  %2771 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2770
  %2772 = load i32, ptr addrspace(3) %2771, align 4, !tbaa !7
  %2773 = or disjoint i32 %193, 20480
  %2774 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2773
  %2775 = load i32, ptr addrspace(3) %2774, align 4, !tbaa !7
  %2776 = or disjoint i32 %193, 27648
  %2777 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2776
  %2778 = load i32, ptr addrspace(3) %2777, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef %163, i32 noundef 16, i32 noundef %162, i32 noundef 3328, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %168, i32 noundef 16, i32 noundef %167, i32 noundef 3328, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %172, i32 noundef 16, i32 noundef %171, i32 noundef 3328, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %176, i32 noundef 16, i32 noundef %175, i32 noundef 3328, i32 noundef 0, i32 noundef 0) #11
  %2779 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2751, <4 x i32> noundef %2588, <4 x float> noundef %2677, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2769, i32 noundef 0, i32 noundef %2645) #11
  %2780 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2752, <4 x i32> noundef %2588, <4 x float> noundef %2678, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2769, i32 noundef 0, i32 noundef %2645) #11
  %2781 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2759, <4 x i32> noundef %2590, <4 x float> noundef %2779, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2769, i32 noundef 2, i32 noundef %2645) #11
  %2782 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2760, <4 x i32> noundef %2590, <4 x float> noundef %2780, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2769, i32 noundef 2, i32 noundef %2645) #11
  %2783 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2753, <4 x i32> noundef %2588, <4 x float> noundef %2681, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2772, i32 noundef 0, i32 noundef %2645) #11
  %2784 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2754, <4 x i32> noundef %2588, <4 x float> noundef %2682, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2772, i32 noundef 0, i32 noundef %2645) #11
  %2785 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2761, <4 x i32> noundef %2590, <4 x float> noundef %2783, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2772, i32 noundef 2, i32 noundef %2645) #11
  %2786 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2762, <4 x i32> noundef %2590, <4 x float> noundef %2784, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2772, i32 noundef 2, i32 noundef %2645) #11
  %2787 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2755, <4 x i32> noundef %2588, <4 x float> noundef %2685, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2775, i32 noundef 0, i32 noundef %2645) #11
  %2788 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2756, <4 x i32> noundef %2588, <4 x float> noundef %2686, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2775, i32 noundef 0, i32 noundef %2645) #11
  %2789 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2763, <4 x i32> noundef %2590, <4 x float> noundef %2787, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2775, i32 noundef 2, i32 noundef %2645) #11
  %2790 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2764, <4 x i32> noundef %2590, <4 x float> noundef %2788, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2775, i32 noundef 2, i32 noundef %2645) #11
  %2791 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2757, <4 x i32> noundef %2588, <4 x float> noundef %2689, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2778, i32 noundef 0, i32 noundef %2645) #11
  %2792 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2758, <4 x i32> noundef %2588, <4 x float> noundef %2690, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2778, i32 noundef 0, i32 noundef %2645) #11
  %2793 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2765, <4 x i32> noundef %2590, <4 x float> noundef %2791, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2778, i32 noundef 2, i32 noundef %2645) #11
  %2794 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2766, <4 x i32> noundef %2590, <4 x float> noundef %2792, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2778, i32 noundef 2, i32 noundef %2645) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2795 = or disjoint i32 %181, 53248
  %2796 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2795, i32 %75, i32 0)
  %2797 = or disjoint i32 %181, 54272
  %2798 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2797, i32 %75, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2799 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2751, <4 x i32> noundef %2607, <4 x float> noundef %2697, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2769, i32 noundef 1, i32 noundef %2645) #11
  %2800 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2752, <4 x i32> noundef %2607, <4 x float> noundef %2698, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2769, i32 noundef 1, i32 noundef %2645) #11
  %2801 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2759, <4 x i32> noundef %2608, <4 x float> noundef %2799, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2769, i32 noundef 3, i32 noundef %2645) #11
  %2802 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2760, <4 x i32> noundef %2608, <4 x float> noundef %2800, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2769, i32 noundef 3, i32 noundef %2645) #11
  %2803 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2753, <4 x i32> noundef %2607, <4 x float> noundef %2701, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2772, i32 noundef 1, i32 noundef %2645) #11
  %2804 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2754, <4 x i32> noundef %2607, <4 x float> noundef %2702, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2772, i32 noundef 1, i32 noundef %2645) #11
  %2805 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2761, <4 x i32> noundef %2608, <4 x float> noundef %2803, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2772, i32 noundef 3, i32 noundef %2645) #11
  %2806 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2762, <4 x i32> noundef %2608, <4 x float> noundef %2804, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2772, i32 noundef 3, i32 noundef %2645) #11
  %2807 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2755, <4 x i32> noundef %2607, <4 x float> noundef %2705, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2775, i32 noundef 1, i32 noundef %2645) #11
  %2808 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2756, <4 x i32> noundef %2607, <4 x float> noundef %2706, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2775, i32 noundef 1, i32 noundef %2645) #11
  %2809 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2763, <4 x i32> noundef %2608, <4 x float> noundef %2807, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2775, i32 noundef 3, i32 noundef %2645) #11
  %2810 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2764, <4 x i32> noundef %2608, <4 x float> noundef %2808, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2775, i32 noundef 3, i32 noundef %2645) #11
  %2811 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2757, <4 x i32> noundef %2607, <4 x float> noundef %2709, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2778, i32 noundef 1, i32 noundef %2645) #11
  %2812 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2758, <4 x i32> noundef %2607, <4 x float> noundef %2710, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2778, i32 noundef 1, i32 noundef %2645) #11
  %2813 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2765, <4 x i32> noundef %2608, <4 x float> noundef %2811, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2778, i32 noundef 3, i32 noundef %2645) #11
  %2814 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2766, <4 x i32> noundef %2608, <4 x float> noundef %2812, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2778, i32 noundef 3, i32 noundef %2645) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2815 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2795, i32 %79, i32 0)
  %2816 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2797, i32 %79, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2817 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2751, <4 x i32> noundef %2625, <4 x float> noundef %2715, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2769, i32 noundef 0, i32 noundef %2646) #11
  %2818 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2752, <4 x i32> noundef %2625, <4 x float> noundef %2716, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2769, i32 noundef 0, i32 noundef %2646) #11
  %2819 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2759, <4 x i32> noundef %2626, <4 x float> noundef %2817, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2769, i32 noundef 2, i32 noundef %2646) #11
  %2820 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2760, <4 x i32> noundef %2626, <4 x float> noundef %2818, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2769, i32 noundef 2, i32 noundef %2646) #11
  %2821 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2753, <4 x i32> noundef %2625, <4 x float> noundef %2719, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2772, i32 noundef 0, i32 noundef %2646) #11
  %2822 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2754, <4 x i32> noundef %2625, <4 x float> noundef %2720, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2772, i32 noundef 0, i32 noundef %2646) #11
  %2823 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2761, <4 x i32> noundef %2626, <4 x float> noundef %2821, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2772, i32 noundef 2, i32 noundef %2646) #11
  %2824 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2762, <4 x i32> noundef %2626, <4 x float> noundef %2822, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2772, i32 noundef 2, i32 noundef %2646) #11
  %2825 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2755, <4 x i32> noundef %2625, <4 x float> noundef %2723, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2775, i32 noundef 0, i32 noundef %2646) #11
  %2826 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2756, <4 x i32> noundef %2625, <4 x float> noundef %2724, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2775, i32 noundef 0, i32 noundef %2646) #11
  %2827 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2763, <4 x i32> noundef %2626, <4 x float> noundef %2825, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2775, i32 noundef 2, i32 noundef %2646) #11
  %2828 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2764, <4 x i32> noundef %2626, <4 x float> noundef %2826, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2775, i32 noundef 2, i32 noundef %2646) #11
  %2829 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2757, <4 x i32> noundef %2625, <4 x float> noundef %2727, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2778, i32 noundef 0, i32 noundef %2646) #11
  %2830 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2758, <4 x i32> noundef %2625, <4 x float> noundef %2728, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2778, i32 noundef 0, i32 noundef %2646) #11
  %2831 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2765, <4 x i32> noundef %2626, <4 x float> noundef %2829, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2778, i32 noundef 2, i32 noundef %2646) #11
  %2832 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2766, <4 x i32> noundef %2626, <4 x float> noundef %2830, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2778, i32 noundef 2, i32 noundef %2646) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2833 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2795, i32 %83, i32 0)
  %2834 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2797, i32 %83, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2835 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2751, <4 x i32> noundef %2643, <4 x float> noundef %2733, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2769, i32 noundef 1, i32 noundef %2646) #11
  %2836 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2752, <4 x i32> noundef %2643, <4 x float> noundef %2734, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2769, i32 noundef 1, i32 noundef %2646) #11
  %2837 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2759, <4 x i32> noundef %2644, <4 x float> noundef %2835, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2769, i32 noundef 3, i32 noundef %2646) #11
  %2838 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2760, <4 x i32> noundef %2644, <4 x float> noundef %2836, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2769, i32 noundef 3, i32 noundef %2646) #11
  %2839 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2753, <4 x i32> noundef %2643, <4 x float> noundef %2737, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2772, i32 noundef 1, i32 noundef %2646) #11
  %2840 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2754, <4 x i32> noundef %2643, <4 x float> noundef %2738, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2772, i32 noundef 1, i32 noundef %2646) #11
  %2841 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2761, <4 x i32> noundef %2644, <4 x float> noundef %2839, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2772, i32 noundef 3, i32 noundef %2646) #11
  %2842 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2762, <4 x i32> noundef %2644, <4 x float> noundef %2840, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2772, i32 noundef 3, i32 noundef %2646) #11
  %2843 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2755, <4 x i32> noundef %2643, <4 x float> noundef %2741, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2775, i32 noundef 1, i32 noundef %2646) #11
  %2844 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2756, <4 x i32> noundef %2643, <4 x float> noundef %2742, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2775, i32 noundef 1, i32 noundef %2646) #11
  %2845 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2763, <4 x i32> noundef %2644, <4 x float> noundef %2843, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2775, i32 noundef 3, i32 noundef %2646) #11
  %2846 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2764, <4 x i32> noundef %2644, <4 x float> noundef %2844, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2775, i32 noundef 3, i32 noundef %2646) #11
  %2847 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2757, <4 x i32> noundef %2643, <4 x float> noundef %2745, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2778, i32 noundef 1, i32 noundef %2646) #11
  %2848 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2758, <4 x i32> noundef %2643, <4 x float> noundef %2746, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2778, i32 noundef 1, i32 noundef %2646) #11
  %2849 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2765, <4 x i32> noundef %2644, <4 x float> noundef %2847, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2778, i32 noundef 3, i32 noundef %2646) #11
  %2850 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2766, <4 x i32> noundef %2644, <4 x float> noundef %2848, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2778, i32 noundef 3, i32 noundef %2646) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2851 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2795, i32 %87, i32 0)
  %2852 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2797, i32 %87, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2853 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %1190, i32 %97, i32 0)
  %2854 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %1190, i32 %102, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %2855 = load <4 x i32>, ptr addrspace(3) %345, align 16, !tbaa !11
  %2856 = load <4 x i32>, ptr addrspace(3) %347, align 16, !tbaa !11
  %2857 = load <4 x i32>, ptr addrspace(3) %349, align 16, !tbaa !11
  %2858 = load <4 x i32>, ptr addrspace(3) %351, align 16, !tbaa !11
  %2859 = load <4 x i32>, ptr addrspace(3) %353, align 16, !tbaa !11
  %2860 = load <4 x i32>, ptr addrspace(3) %355, align 16, !tbaa !11
  %2861 = load <4 x i32>, ptr addrspace(3) %357, align 16, !tbaa !11
  %2862 = load <4 x i32>, ptr addrspace(3) %359, align 16, !tbaa !11
  %2863 = load <4 x i32>, ptr addrspace(3) %361, align 16, !tbaa !11
  %2864 = load <4 x i32>, ptr addrspace(3) %363, align 16, !tbaa !11
  %2865 = load <4 x i32>, ptr addrspace(3) %365, align 16, !tbaa !11
  %2866 = load <4 x i32>, ptr addrspace(3) %367, align 16, !tbaa !11
  %2867 = load <4 x i32>, ptr addrspace(3) %369, align 16, !tbaa !11
  %2868 = load <4 x i32>, ptr addrspace(3) %371, align 16, !tbaa !11
  %2869 = load <4 x i32>, ptr addrspace(3) %373, align 16, !tbaa !11
  %2870 = load <4 x i32>, ptr addrspace(3) %375, align 16, !tbaa !11
  %2871 = or disjoint i32 %193, 6400
  %2872 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2871
  %2873 = load i32, ptr addrspace(3) %2872, align 4, !tbaa !7
  %2874 = or disjoint i32 %193, 13568
  %2875 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2874
  %2876 = load i32, ptr addrspace(3) %2875, align 4, !tbaa !7
  %2877 = or disjoint i32 %193, 20736
  %2878 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2877
  %2879 = load i32, ptr addrspace(3) %2878, align 4, !tbaa !7
  %2880 = or disjoint i32 %193, 27904
  %2881 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2880
  %2882 = load i32, ptr addrspace(3) %2881, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %196, i32 noundef 16, i32 noundef %162, i32 noundef 3456, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %197, i32 noundef 16, i32 noundef %167, i32 noundef 3456, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %198, i32 noundef 16, i32 noundef %171, i32 noundef 3456, i32 noundef 0, i32 noundef 0) #11
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %25, ptr addrspace(3) noundef nonnull %199, i32 noundef 16, i32 noundef %175, i32 noundef 3456, i32 noundef 0, i32 noundef 0) #11
  %2883 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2855, <4 x i32> noundef %2692, <4 x float> noundef %2781, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2873, i32 noundef 0, i32 noundef %2749) #11
  %2884 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2856, <4 x i32> noundef %2692, <4 x float> noundef %2782, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2873, i32 noundef 0, i32 noundef %2749) #11
  %2885 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2863, <4 x i32> noundef %2694, <4 x float> noundef %2883, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2873, i32 noundef 2, i32 noundef %2749) #11
  %2886 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2864, <4 x i32> noundef %2694, <4 x float> noundef %2884, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2873, i32 noundef 2, i32 noundef %2749) #11
  %2887 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2857, <4 x i32> noundef %2692, <4 x float> noundef %2785, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2876, i32 noundef 0, i32 noundef %2749) #11
  %2888 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2858, <4 x i32> noundef %2692, <4 x float> noundef %2786, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2876, i32 noundef 0, i32 noundef %2749) #11
  %2889 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2865, <4 x i32> noundef %2694, <4 x float> noundef %2887, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2876, i32 noundef 2, i32 noundef %2749) #11
  %2890 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2866, <4 x i32> noundef %2694, <4 x float> noundef %2888, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2876, i32 noundef 2, i32 noundef %2749) #11
  %2891 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2859, <4 x i32> noundef %2692, <4 x float> noundef %2789, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2879, i32 noundef 0, i32 noundef %2749) #11
  %2892 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2860, <4 x i32> noundef %2692, <4 x float> noundef %2790, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2879, i32 noundef 0, i32 noundef %2749) #11
  %2893 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2867, <4 x i32> noundef %2694, <4 x float> noundef %2891, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2879, i32 noundef 2, i32 noundef %2749) #11
  %2894 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2868, <4 x i32> noundef %2694, <4 x float> noundef %2892, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2879, i32 noundef 2, i32 noundef %2749) #11
  %2895 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2861, <4 x i32> noundef %2692, <4 x float> noundef %2793, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2882, i32 noundef 0, i32 noundef %2749) #11
  %2896 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2862, <4 x i32> noundef %2692, <4 x float> noundef %2794, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2882, i32 noundef 0, i32 noundef %2749) #11
  %2897 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2869, <4 x i32> noundef %2694, <4 x float> noundef %2895, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2882, i32 noundef 2, i32 noundef %2749) #11
  %2898 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2870, <4 x i32> noundef %2694, <4 x float> noundef %2896, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2882, i32 noundef 2, i32 noundef %2749) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2899 = or disjoint i32 %181, 55296
  %2900 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2899, i32 %75, i32 0)
  %2901 = or disjoint i32 %181, 56320
  %2902 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2901, i32 %75, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2903 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2855, <4 x i32> noundef %2711, <4 x float> noundef %2801, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2873, i32 noundef 1, i32 noundef %2749) #11
  %2904 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2856, <4 x i32> noundef %2711, <4 x float> noundef %2802, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2873, i32 noundef 1, i32 noundef %2749) #11
  %2905 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2863, <4 x i32> noundef %2712, <4 x float> noundef %2903, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2873, i32 noundef 3, i32 noundef %2749) #11
  %2906 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2864, <4 x i32> noundef %2712, <4 x float> noundef %2904, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2873, i32 noundef 3, i32 noundef %2749) #11
  %2907 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2857, <4 x i32> noundef %2711, <4 x float> noundef %2805, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2876, i32 noundef 1, i32 noundef %2749) #11
  %2908 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2858, <4 x i32> noundef %2711, <4 x float> noundef %2806, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2876, i32 noundef 1, i32 noundef %2749) #11
  %2909 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2865, <4 x i32> noundef %2712, <4 x float> noundef %2907, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2876, i32 noundef 3, i32 noundef %2749) #11
  %2910 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2866, <4 x i32> noundef %2712, <4 x float> noundef %2908, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2876, i32 noundef 3, i32 noundef %2749) #11
  %2911 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2859, <4 x i32> noundef %2711, <4 x float> noundef %2809, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2879, i32 noundef 1, i32 noundef %2749) #11
  %2912 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2860, <4 x i32> noundef %2711, <4 x float> noundef %2810, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2879, i32 noundef 1, i32 noundef %2749) #11
  %2913 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2867, <4 x i32> noundef %2712, <4 x float> noundef %2911, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2879, i32 noundef 3, i32 noundef %2749) #11
  %2914 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2868, <4 x i32> noundef %2712, <4 x float> noundef %2912, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2879, i32 noundef 3, i32 noundef %2749) #11
  %2915 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2861, <4 x i32> noundef %2711, <4 x float> noundef %2813, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2882, i32 noundef 1, i32 noundef %2749) #11
  %2916 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2862, <4 x i32> noundef %2711, <4 x float> noundef %2814, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2882, i32 noundef 1, i32 noundef %2749) #11
  %2917 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2869, <4 x i32> noundef %2712, <4 x float> noundef %2915, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2882, i32 noundef 3, i32 noundef %2749) #11
  %2918 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2870, <4 x i32> noundef %2712, <4 x float> noundef %2916, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2882, i32 noundef 3, i32 noundef %2749) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2919 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2899, i32 %79, i32 0)
  %2920 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2901, i32 %79, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2921 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2855, <4 x i32> noundef %2729, <4 x float> noundef %2819, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2873, i32 noundef 0, i32 noundef %2750) #11
  %2922 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2856, <4 x i32> noundef %2729, <4 x float> noundef %2820, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2873, i32 noundef 0, i32 noundef %2750) #11
  %2923 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2863, <4 x i32> noundef %2730, <4 x float> noundef %2921, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2873, i32 noundef 2, i32 noundef %2750) #11
  %2924 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2864, <4 x i32> noundef %2730, <4 x float> noundef %2922, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2873, i32 noundef 2, i32 noundef %2750) #11
  %2925 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2857, <4 x i32> noundef %2729, <4 x float> noundef %2823, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2876, i32 noundef 0, i32 noundef %2750) #11
  %2926 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2858, <4 x i32> noundef %2729, <4 x float> noundef %2824, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2876, i32 noundef 0, i32 noundef %2750) #11
  %2927 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2865, <4 x i32> noundef %2730, <4 x float> noundef %2925, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2876, i32 noundef 2, i32 noundef %2750) #11
  %2928 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2866, <4 x i32> noundef %2730, <4 x float> noundef %2926, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2876, i32 noundef 2, i32 noundef %2750) #11
  %2929 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2859, <4 x i32> noundef %2729, <4 x float> noundef %2827, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2879, i32 noundef 0, i32 noundef %2750) #11
  %2930 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2860, <4 x i32> noundef %2729, <4 x float> noundef %2828, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2879, i32 noundef 0, i32 noundef %2750) #11
  %2931 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2867, <4 x i32> noundef %2730, <4 x float> noundef %2929, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2879, i32 noundef 2, i32 noundef %2750) #11
  %2932 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2868, <4 x i32> noundef %2730, <4 x float> noundef %2930, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2879, i32 noundef 2, i32 noundef %2750) #11
  %2933 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2861, <4 x i32> noundef %2729, <4 x float> noundef %2831, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2882, i32 noundef 0, i32 noundef %2750) #11
  %2934 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2862, <4 x i32> noundef %2729, <4 x float> noundef %2832, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2882, i32 noundef 0, i32 noundef %2750) #11
  %2935 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2869, <4 x i32> noundef %2730, <4 x float> noundef %2933, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2882, i32 noundef 2, i32 noundef %2750) #11
  %2936 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2870, <4 x i32> noundef %2730, <4 x float> noundef %2934, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2882, i32 noundef 2, i32 noundef %2750) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2937 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2899, i32 %83, i32 0)
  %2938 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2901, i32 %83, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2939 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2855, <4 x i32> noundef %2747, <4 x float> noundef %2837, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2873, i32 noundef 1, i32 noundef %2750) #11
  %2940 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2856, <4 x i32> noundef %2747, <4 x float> noundef %2838, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2873, i32 noundef 1, i32 noundef %2750) #11
  %2941 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2863, <4 x i32> noundef %2748, <4 x float> noundef %2939, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2873, i32 noundef 3, i32 noundef %2750) #11
  %2942 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2864, <4 x i32> noundef %2748, <4 x float> noundef %2940, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2873, i32 noundef 3, i32 noundef %2750) #11
  %2943 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2857, <4 x i32> noundef %2747, <4 x float> noundef %2841, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2876, i32 noundef 1, i32 noundef %2750) #11
  %2944 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2858, <4 x i32> noundef %2747, <4 x float> noundef %2842, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2876, i32 noundef 1, i32 noundef %2750) #11
  %2945 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2865, <4 x i32> noundef %2748, <4 x float> noundef %2943, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2876, i32 noundef 3, i32 noundef %2750) #11
  %2946 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2866, <4 x i32> noundef %2748, <4 x float> noundef %2944, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2876, i32 noundef 3, i32 noundef %2750) #11
  %2947 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2859, <4 x i32> noundef %2747, <4 x float> noundef %2845, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2879, i32 noundef 1, i32 noundef %2750) #11
  %2948 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2860, <4 x i32> noundef %2747, <4 x float> noundef %2846, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2879, i32 noundef 1, i32 noundef %2750) #11
  %2949 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2867, <4 x i32> noundef %2748, <4 x float> noundef %2947, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2879, i32 noundef 3, i32 noundef %2750) #11
  %2950 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2868, <4 x i32> noundef %2748, <4 x float> noundef %2948, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2879, i32 noundef 3, i32 noundef %2750) #11
  %2951 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2861, <4 x i32> noundef %2747, <4 x float> noundef %2849, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2882, i32 noundef 1, i32 noundef %2750) #11
  %2952 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2862, <4 x i32> noundef %2747, <4 x float> noundef %2850, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2882, i32 noundef 1, i32 noundef %2750) #11
  %2953 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2869, <4 x i32> noundef %2748, <4 x float> noundef %2951, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2882, i32 noundef 3, i32 noundef %2750) #11
  %2954 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2870, <4 x i32> noundef %2748, <4 x float> noundef %2952, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2882, i32 noundef 3, i32 noundef %2750) #11
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2955 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2899, i32 %87, i32 0)
  %2956 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %26, i32 %2901, i32 %87, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2957 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %1294, i32 %97, i32 0)
  %2958 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %27, i32 %1294, i32 %102, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %2959 = load <4 x i32>, ptr addrspace(3) %216, align 16, !tbaa !11
  %2960 = load <4 x i32>, ptr addrspace(3) %219, align 16, !tbaa !11
  %2961 = load <4 x i32>, ptr addrspace(3) %222, align 16, !tbaa !11
  %2962 = load <4 x i32>, ptr addrspace(3) %225, align 16, !tbaa !11
  %2963 = load <4 x i32>, ptr addrspace(3) %228, align 16, !tbaa !11
  %2964 = load <4 x i32>, ptr addrspace(3) %231, align 16, !tbaa !11
  %2965 = load <4 x i32>, ptr addrspace(3) %234, align 16, !tbaa !11
  %2966 = load <4 x i32>, ptr addrspace(3) %237, align 16, !tbaa !11
  %2967 = load <4 x i32>, ptr addrspace(3) %241, align 16, !tbaa !11
  %2968 = load <4 x i32>, ptr addrspace(3) %243, align 16, !tbaa !11
  %2969 = load <4 x i32>, ptr addrspace(3) %245, align 16, !tbaa !11
  %2970 = load <4 x i32>, ptr addrspace(3) %247, align 16, !tbaa !11
  %2971 = load <4 x i32>, ptr addrspace(3) %249, align 16, !tbaa !11
  %2972 = load <4 x i32>, ptr addrspace(3) %251, align 16, !tbaa !11
  %2973 = load <4 x i32>, ptr addrspace(3) %253, align 16, !tbaa !11
  %2974 = load <4 x i32>, ptr addrspace(3) %255, align 16, !tbaa !11
  %2975 = or disjoint i32 %193, 6656
  %2976 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2975
  %2977 = load i32, ptr addrspace(3) %2976, align 4, !tbaa !7
  %2978 = or disjoint i32 %193, 13824
  %2979 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2978
  %2980 = load i32, ptr addrspace(3) %2979, align 4, !tbaa !7
  %2981 = or disjoint i32 %193, 20992
  %2982 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2981
  %2983 = load i32, ptr addrspace(3) %2982, align 4, !tbaa !7
  %2984 = or disjoint i32 %193, 28160
  %2985 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %2984
  %2986 = load i32, ptr addrspace(3) %2985, align 4, !tbaa !7
  %2987 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2959, <4 x i32> noundef %2796, <4 x float> noundef %2885, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2977, i32 noundef 0, i32 noundef %2853) #11
  %2988 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2960, <4 x i32> noundef %2796, <4 x float> noundef %2886, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2977, i32 noundef 0, i32 noundef %2853) #11
  %2989 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2967, <4 x i32> noundef %2798, <4 x float> noundef %2987, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2977, i32 noundef 2, i32 noundef %2853) #11
  %2990 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2968, <4 x i32> noundef %2798, <4 x float> noundef %2988, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2977, i32 noundef 2, i32 noundef %2853) #11
  %2991 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2961, <4 x i32> noundef %2796, <4 x float> noundef %2889, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2980, i32 noundef 0, i32 noundef %2853) #11
  %2992 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2962, <4 x i32> noundef %2796, <4 x float> noundef %2890, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2980, i32 noundef 0, i32 noundef %2853) #11
  %2993 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2969, <4 x i32> noundef %2798, <4 x float> noundef %2991, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2980, i32 noundef 2, i32 noundef %2853) #11
  %2994 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2970, <4 x i32> noundef %2798, <4 x float> noundef %2992, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2980, i32 noundef 2, i32 noundef %2853) #11
  %2995 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2963, <4 x i32> noundef %2796, <4 x float> noundef %2893, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2983, i32 noundef 0, i32 noundef %2853) #11
  %2996 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2964, <4 x i32> noundef %2796, <4 x float> noundef %2894, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2983, i32 noundef 0, i32 noundef %2853) #11
  %2997 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2971, <4 x i32> noundef %2798, <4 x float> noundef %2995, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2983, i32 noundef 2, i32 noundef %2853) #11
  %2998 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2972, <4 x i32> noundef %2798, <4 x float> noundef %2996, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2983, i32 noundef 2, i32 noundef %2853) #11
  %2999 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2965, <4 x i32> noundef %2796, <4 x float> noundef %2897, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2986, i32 noundef 0, i32 noundef %2853) #11
  %3000 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2966, <4 x i32> noundef %2796, <4 x float> noundef %2898, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2986, i32 noundef 0, i32 noundef %2853) #11
  %3001 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2973, <4 x i32> noundef %2798, <4 x float> noundef %2999, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2986, i32 noundef 2, i32 noundef %2853) #11
  %3002 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2974, <4 x i32> noundef %2798, <4 x float> noundef %3000, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2986, i32 noundef 2, i32 noundef %2853) #11
  %3003 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2959, <4 x i32> noundef %2815, <4 x float> noundef %2905, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2977, i32 noundef 1, i32 noundef %2853) #11
  %3004 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2960, <4 x i32> noundef %2815, <4 x float> noundef %2906, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2977, i32 noundef 1, i32 noundef %2853) #11
  %3005 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2967, <4 x i32> noundef %2816, <4 x float> noundef %3003, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2977, i32 noundef 3, i32 noundef %2853) #11
  %3006 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2968, <4 x i32> noundef %2816, <4 x float> noundef %3004, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2977, i32 noundef 3, i32 noundef %2853) #11
  %3007 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2961, <4 x i32> noundef %2815, <4 x float> noundef %2909, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2980, i32 noundef 1, i32 noundef %2853) #11
  %3008 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2962, <4 x i32> noundef %2815, <4 x float> noundef %2910, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2980, i32 noundef 1, i32 noundef %2853) #11
  %3009 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2969, <4 x i32> noundef %2816, <4 x float> noundef %3007, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2980, i32 noundef 3, i32 noundef %2853) #11
  %3010 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2970, <4 x i32> noundef %2816, <4 x float> noundef %3008, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2980, i32 noundef 3, i32 noundef %2853) #11
  %3011 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2963, <4 x i32> noundef %2815, <4 x float> noundef %2913, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2983, i32 noundef 1, i32 noundef %2853) #11
  %3012 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2964, <4 x i32> noundef %2815, <4 x float> noundef %2914, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2983, i32 noundef 1, i32 noundef %2853) #11
  %3013 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2971, <4 x i32> noundef %2816, <4 x float> noundef %3011, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2983, i32 noundef 3, i32 noundef %2853) #11
  %3014 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2972, <4 x i32> noundef %2816, <4 x float> noundef %3012, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2983, i32 noundef 3, i32 noundef %2853) #11
  %3015 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2965, <4 x i32> noundef %2815, <4 x float> noundef %2917, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2986, i32 noundef 1, i32 noundef %2853) #11
  %3016 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2966, <4 x i32> noundef %2815, <4 x float> noundef %2918, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2986, i32 noundef 1, i32 noundef %2853) #11
  %3017 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2973, <4 x i32> noundef %2816, <4 x float> noundef %3015, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2986, i32 noundef 3, i32 noundef %2853) #11
  %3018 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2974, <4 x i32> noundef %2816, <4 x float> noundef %3016, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2986, i32 noundef 3, i32 noundef %2853) #11
  %3019 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2959, <4 x i32> noundef %2833, <4 x float> noundef %2923, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2977, i32 noundef 0, i32 noundef %2854) #11
  %3020 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2960, <4 x i32> noundef %2833, <4 x float> noundef %2924, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2977, i32 noundef 0, i32 noundef %2854) #11
  %3021 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2967, <4 x i32> noundef %2834, <4 x float> noundef %3019, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2977, i32 noundef 2, i32 noundef %2854) #11
  %3022 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2968, <4 x i32> noundef %2834, <4 x float> noundef %3020, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2977, i32 noundef 2, i32 noundef %2854) #11
  %3023 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2961, <4 x i32> noundef %2833, <4 x float> noundef %2927, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2980, i32 noundef 0, i32 noundef %2854) #11
  %3024 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2962, <4 x i32> noundef %2833, <4 x float> noundef %2928, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2980, i32 noundef 0, i32 noundef %2854) #11
  %3025 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2969, <4 x i32> noundef %2834, <4 x float> noundef %3023, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2980, i32 noundef 2, i32 noundef %2854) #11
  %3026 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2970, <4 x i32> noundef %2834, <4 x float> noundef %3024, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2980, i32 noundef 2, i32 noundef %2854) #11
  %3027 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2963, <4 x i32> noundef %2833, <4 x float> noundef %2931, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2983, i32 noundef 0, i32 noundef %2854) #11
  %3028 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2964, <4 x i32> noundef %2833, <4 x float> noundef %2932, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2983, i32 noundef 0, i32 noundef %2854) #11
  %3029 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2971, <4 x i32> noundef %2834, <4 x float> noundef %3027, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2983, i32 noundef 2, i32 noundef %2854) #11
  %3030 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2972, <4 x i32> noundef %2834, <4 x float> noundef %3028, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2983, i32 noundef 2, i32 noundef %2854) #11
  %3031 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2965, <4 x i32> noundef %2833, <4 x float> noundef %2935, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2986, i32 noundef 0, i32 noundef %2854) #11
  %3032 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2966, <4 x i32> noundef %2833, <4 x float> noundef %2936, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2986, i32 noundef 0, i32 noundef %2854) #11
  %3033 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2973, <4 x i32> noundef %2834, <4 x float> noundef %3031, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2986, i32 noundef 2, i32 noundef %2854) #11
  %3034 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2974, <4 x i32> noundef %2834, <4 x float> noundef %3032, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2986, i32 noundef 2, i32 noundef %2854) #11
  %3035 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2959, <4 x i32> noundef %2851, <4 x float> noundef %2941, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2977, i32 noundef 1, i32 noundef %2854) #11
  %3036 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2960, <4 x i32> noundef %2851, <4 x float> noundef %2942, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2977, i32 noundef 1, i32 noundef %2854) #11
  %3037 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2967, <4 x i32> noundef %2852, <4 x float> noundef %3035, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2977, i32 noundef 3, i32 noundef %2854) #11
  %3038 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2968, <4 x i32> noundef %2852, <4 x float> noundef %3036, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2977, i32 noundef 3, i32 noundef %2854) #11
  %3039 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2961, <4 x i32> noundef %2851, <4 x float> noundef %2945, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2980, i32 noundef 1, i32 noundef %2854) #11
  %3040 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2962, <4 x i32> noundef %2851, <4 x float> noundef %2946, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2980, i32 noundef 1, i32 noundef %2854) #11
  %3041 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2969, <4 x i32> noundef %2852, <4 x float> noundef %3039, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2980, i32 noundef 3, i32 noundef %2854) #11
  %3042 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2970, <4 x i32> noundef %2852, <4 x float> noundef %3040, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2980, i32 noundef 3, i32 noundef %2854) #11
  %3043 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2963, <4 x i32> noundef %2851, <4 x float> noundef %2949, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2983, i32 noundef 1, i32 noundef %2854) #11
  %3044 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2964, <4 x i32> noundef %2851, <4 x float> noundef %2950, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2983, i32 noundef 1, i32 noundef %2854) #11
  %3045 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2971, <4 x i32> noundef %2852, <4 x float> noundef %3043, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2983, i32 noundef 3, i32 noundef %2854) #11
  %3046 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2972, <4 x i32> noundef %2852, <4 x float> noundef %3044, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2983, i32 noundef 3, i32 noundef %2854) #11
  %3047 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2965, <4 x i32> noundef %2851, <4 x float> noundef %2953, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2986, i32 noundef 1, i32 noundef %2854) #11
  %3048 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2966, <4 x i32> noundef %2851, <4 x float> noundef %2954, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %2986, i32 noundef 1, i32 noundef %2854) #11
  %3049 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2973, <4 x i32> noundef %2852, <4 x float> noundef %3047, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2986, i32 noundef 3, i32 noundef %2854) #11
  %3050 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2974, <4 x i32> noundef %2852, <4 x float> noundef %3048, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %2986, i32 noundef 3, i32 noundef %2854) #11
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %3051 = load <4 x i32>, ptr addrspace(3) %345, align 16, !tbaa !11
  %3052 = load <4 x i32>, ptr addrspace(3) %347, align 16, !tbaa !11
  %3053 = load <4 x i32>, ptr addrspace(3) %349, align 16, !tbaa !11
  %3054 = load <4 x i32>, ptr addrspace(3) %351, align 16, !tbaa !11
  %3055 = load <4 x i32>, ptr addrspace(3) %353, align 16, !tbaa !11
  %3056 = load <4 x i32>, ptr addrspace(3) %355, align 16, !tbaa !11
  %3057 = load <4 x i32>, ptr addrspace(3) %357, align 16, !tbaa !11
  %3058 = load <4 x i32>, ptr addrspace(3) %359, align 16, !tbaa !11
  %3059 = load <4 x i32>, ptr addrspace(3) %361, align 16, !tbaa !11
  %3060 = load <4 x i32>, ptr addrspace(3) %363, align 16, !tbaa !11
  %3061 = load <4 x i32>, ptr addrspace(3) %365, align 16, !tbaa !11
  %3062 = load <4 x i32>, ptr addrspace(3) %367, align 16, !tbaa !11
  %3063 = load <4 x i32>, ptr addrspace(3) %369, align 16, !tbaa !11
  %3064 = load <4 x i32>, ptr addrspace(3) %371, align 16, !tbaa !11
  %3065 = load <4 x i32>, ptr addrspace(3) %373, align 16, !tbaa !11
  %3066 = load <4 x i32>, ptr addrspace(3) %375, align 16, !tbaa !11
  %3067 = or disjoint i32 %193, 6912
  %3068 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %3067
  %3069 = load i32, ptr addrspace(3) %3068, align 4, !tbaa !7
  %3070 = or disjoint i32 %193, 14080
  %3071 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %3070
  %3072 = load i32, ptr addrspace(3) %3071, align 4, !tbaa !7
  %3073 = or disjoint i32 %193, 21248
  %3074 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %3073
  %3075 = load i32, ptr addrspace(3) %3074, align 4, !tbaa !7
  %3076 = or disjoint i32 %193, 28416
  %3077 = getelementptr inbounds nuw [28672 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 32768), i32 0, i32 %3076
  %3078 = load i32, ptr addrspace(3) %3077, align 4, !tbaa !7
  %3079 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3051, <4 x i32> noundef %2900, <4 x float> noundef %2989, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %3069, i32 noundef 0, i32 noundef %2957) #11
  %3080 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3052, <4 x i32> noundef %2900, <4 x float> noundef %2990, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %3069, i32 noundef 0, i32 noundef %2957) #11
  %3081 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3059, <4 x i32> noundef %2902, <4 x float> noundef %3079, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %3069, i32 noundef 2, i32 noundef %2957) #11
  %3082 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3060, <4 x i32> noundef %2902, <4 x float> noundef %3080, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %3069, i32 noundef 2, i32 noundef %2957) #11
  %3083 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3053, <4 x i32> noundef %2900, <4 x float> noundef %2993, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %3072, i32 noundef 0, i32 noundef %2957) #11
  %3084 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3054, <4 x i32> noundef %2900, <4 x float> noundef %2994, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %3072, i32 noundef 0, i32 noundef %2957) #11
  %3085 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3061, <4 x i32> noundef %2902, <4 x float> noundef %3083, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %3072, i32 noundef 2, i32 noundef %2957) #11
  %3086 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3062, <4 x i32> noundef %2902, <4 x float> noundef %3084, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %3072, i32 noundef 2, i32 noundef %2957) #11
  %3087 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3055, <4 x i32> noundef %2900, <4 x float> noundef %2997, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %3075, i32 noundef 0, i32 noundef %2957) #11
  %3088 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3056, <4 x i32> noundef %2900, <4 x float> noundef %2998, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %3075, i32 noundef 0, i32 noundef %2957) #11
  %3089 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3063, <4 x i32> noundef %2902, <4 x float> noundef %3087, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %3075, i32 noundef 2, i32 noundef %2957) #11
  %3090 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3064, <4 x i32> noundef %2902, <4 x float> noundef %3088, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %3075, i32 noundef 2, i32 noundef %2957) #11
  %3091 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3057, <4 x i32> noundef %2900, <4 x float> noundef %3001, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %3078, i32 noundef 0, i32 noundef %2957) #11
  %3092 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3058, <4 x i32> noundef %2900, <4 x float> noundef %3002, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %3078, i32 noundef 0, i32 noundef %2957) #11
  %3093 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3065, <4 x i32> noundef %2902, <4 x float> noundef %3091, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %3078, i32 noundef 2, i32 noundef %2957) #11
  %3094 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3066, <4 x i32> noundef %2902, <4 x float> noundef %3092, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %3078, i32 noundef 2, i32 noundef %2957) #11
  %3095 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3051, <4 x i32> noundef %2919, <4 x float> noundef %3005, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %3069, i32 noundef 1, i32 noundef %2957) #11
  %3096 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3052, <4 x i32> noundef %2919, <4 x float> noundef %3006, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %3069, i32 noundef 1, i32 noundef %2957) #11
  %3097 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3059, <4 x i32> noundef %2920, <4 x float> noundef %3095, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %3069, i32 noundef 3, i32 noundef %2957) #11
  %3098 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3060, <4 x i32> noundef %2920, <4 x float> noundef %3096, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %3069, i32 noundef 3, i32 noundef %2957) #11
  %3099 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3053, <4 x i32> noundef %2919, <4 x float> noundef %3009, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %3072, i32 noundef 1, i32 noundef %2957) #11
  %3100 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3054, <4 x i32> noundef %2919, <4 x float> noundef %3010, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %3072, i32 noundef 1, i32 noundef %2957) #11
  %3101 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3061, <4 x i32> noundef %2920, <4 x float> noundef %3099, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %3072, i32 noundef 3, i32 noundef %2957) #11
  %3102 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3062, <4 x i32> noundef %2920, <4 x float> noundef %3100, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %3072, i32 noundef 3, i32 noundef %2957) #11
  %3103 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3055, <4 x i32> noundef %2919, <4 x float> noundef %3013, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %3075, i32 noundef 1, i32 noundef %2957) #11
  %3104 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3056, <4 x i32> noundef %2919, <4 x float> noundef %3014, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %3075, i32 noundef 1, i32 noundef %2957) #11
  %3105 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3063, <4 x i32> noundef %2920, <4 x float> noundef %3103, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %3075, i32 noundef 3, i32 noundef %2957) #11
  %3106 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3064, <4 x i32> noundef %2920, <4 x float> noundef %3104, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %3075, i32 noundef 3, i32 noundef %2957) #11
  %3107 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3057, <4 x i32> noundef %2919, <4 x float> noundef %3017, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %3078, i32 noundef 1, i32 noundef %2957) #11
  %3108 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3058, <4 x i32> noundef %2919, <4 x float> noundef %3018, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %3078, i32 noundef 1, i32 noundef %2957) #11
  %3109 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3065, <4 x i32> noundef %2920, <4 x float> noundef %3107, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %3078, i32 noundef 3, i32 noundef %2957) #11
  %3110 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3066, <4 x i32> noundef %2920, <4 x float> noundef %3108, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %3078, i32 noundef 3, i32 noundef %2957) #11
  %3111 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3051, <4 x i32> noundef %2937, <4 x float> noundef %3021, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %3069, i32 noundef 0, i32 noundef %2958) #11
  %3112 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3052, <4 x i32> noundef %2937, <4 x float> noundef %3022, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %3069, i32 noundef 0, i32 noundef %2958) #11
  %3113 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3059, <4 x i32> noundef %2938, <4 x float> noundef %3111, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %3069, i32 noundef 2, i32 noundef %2958) #11
  %3114 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3060, <4 x i32> noundef %2938, <4 x float> noundef %3112, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %3069, i32 noundef 2, i32 noundef %2958) #11
  %3115 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3053, <4 x i32> noundef %2937, <4 x float> noundef %3025, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %3072, i32 noundef 0, i32 noundef %2958) #11
  %3116 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3054, <4 x i32> noundef %2937, <4 x float> noundef %3026, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %3072, i32 noundef 0, i32 noundef %2958) #11
  %3117 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3061, <4 x i32> noundef %2938, <4 x float> noundef %3115, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %3072, i32 noundef 2, i32 noundef %2958) #11
  %3118 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3062, <4 x i32> noundef %2938, <4 x float> noundef %3116, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %3072, i32 noundef 2, i32 noundef %2958) #11
  %3119 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3055, <4 x i32> noundef %2937, <4 x float> noundef %3029, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %3075, i32 noundef 0, i32 noundef %2958) #11
  %3120 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3056, <4 x i32> noundef %2937, <4 x float> noundef %3030, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %3075, i32 noundef 0, i32 noundef %2958) #11
  %3121 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3063, <4 x i32> noundef %2938, <4 x float> noundef %3119, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %3075, i32 noundef 2, i32 noundef %2958) #11
  %3122 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3064, <4 x i32> noundef %2938, <4 x float> noundef %3120, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %3075, i32 noundef 2, i32 noundef %2958) #11
  %3123 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3057, <4 x i32> noundef %2937, <4 x float> noundef %3033, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %3078, i32 noundef 0, i32 noundef %2958) #11
  %3124 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3058, <4 x i32> noundef %2937, <4 x float> noundef %3034, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %3078, i32 noundef 0, i32 noundef %2958) #11
  %3125 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3065, <4 x i32> noundef %2938, <4 x float> noundef %3123, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %3078, i32 noundef 2, i32 noundef %2958) #11
  %3126 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3066, <4 x i32> noundef %2938, <4 x float> noundef %3124, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %3078, i32 noundef 2, i32 noundef %2958) #11
  %3127 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3051, <4 x i32> noundef %2955, <4 x float> noundef %3037, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %3069, i32 noundef 1, i32 noundef %2958) #11
  %3128 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3052, <4 x i32> noundef %2955, <4 x float> noundef %3038, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %3069, i32 noundef 1, i32 noundef %2958) #11
  %3129 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3059, <4 x i32> noundef %2956, <4 x float> noundef %3127, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %3069, i32 noundef 3, i32 noundef %2958) #11
  %3130 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3060, <4 x i32> noundef %2956, <4 x float> noundef %3128, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %3069, i32 noundef 3, i32 noundef %2958) #11
  %3131 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3053, <4 x i32> noundef %2955, <4 x float> noundef %3041, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %3072, i32 noundef 1, i32 noundef %2958) #11
  %3132 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3054, <4 x i32> noundef %2955, <4 x float> noundef %3042, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %3072, i32 noundef 1, i32 noundef %2958) #11
  %3133 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3061, <4 x i32> noundef %2956, <4 x float> noundef %3131, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %3072, i32 noundef 3, i32 noundef %2958) #11
  %3134 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3062, <4 x i32> noundef %2956, <4 x float> noundef %3132, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %3072, i32 noundef 3, i32 noundef %2958) #11
  %3135 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3055, <4 x i32> noundef %2955, <4 x float> noundef %3045, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %3075, i32 noundef 1, i32 noundef %2958) #11
  %3136 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3056, <4 x i32> noundef %2955, <4 x float> noundef %3046, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %3075, i32 noundef 1, i32 noundef %2958) #11
  %3137 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3063, <4 x i32> noundef %2956, <4 x float> noundef %3135, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %3075, i32 noundef 3, i32 noundef %2958) #11
  %3138 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3064, <4 x i32> noundef %2956, <4 x float> noundef %3136, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %3075, i32 noundef 3, i32 noundef %2958) #11
  %3139 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3057, <4 x i32> noundef %2955, <4 x float> noundef %3049, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %3078, i32 noundef 1, i32 noundef %2958) #11
  %3140 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3058, <4 x i32> noundef %2955, <4 x float> noundef %3050, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %3078, i32 noundef 1, i32 noundef %2958) #11
  %3141 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3065, <4 x i32> noundef %2956, <4 x float> noundef %3139, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %3078, i32 noundef 3, i32 noundef %2958) #11
  %3142 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3066, <4 x i32> noundef %2956, <4 x float> noundef %3140, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %3078, i32 noundef 3, i32 noundef %2958) #11
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  tail call void @llvm.experimental.noalias.scope.decl(metadata !16)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !19)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !21)
  %3143 = shl nuw nsw i32 %177, 10
  %3144 = or disjoint i32 %51, %179
  %3145 = add nuw i32 %3144, 128
  %3146 = extractelement <4 x float> %3081, i64 0
  %3147 = add nuw nsw i32 %3144, %3143
  %3148 = zext nneg i32 %3147 to i64
  %3149 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3148
  %3150 = addrspacecast ptr %3149 to ptr addrspace(3)
  store float %3146, ptr addrspace(3) %3150, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3151 = extractelement <4 x float> %3081, i64 1
  %3152 = or disjoint i32 %3143, 256
  %3153 = add nuw nsw i32 %3144, %3152
  %3154 = zext nneg i32 %3153 to i64
  %3155 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3154
  %3156 = addrspacecast ptr %3155 to ptr addrspace(3)
  store float %3151, ptr addrspace(3) %3156, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3157 = extractelement <4 x float> %3081, i64 2
  %3158 = or disjoint i32 %3143, 512
  %3159 = add nuw nsw i32 %3144, %3158
  %3160 = zext nneg i32 %3159 to i64
  %3161 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3160
  %3162 = addrspacecast ptr %3161 to ptr addrspace(3)
  store float %3157, ptr addrspace(3) %3162, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3163 = extractelement <4 x float> %3081, i64 3
  %3164 = or disjoint i32 %3143, 768
  %3165 = add nuw nsw i32 %3144, %3164
  %3166 = zext nneg i32 %3165 to i64
  %3167 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3166
  %3168 = addrspacecast ptr %3167 to ptr addrspace(3)
  store float %3163, ptr addrspace(3) %3168, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3169 = extractelement <4 x float> %3097, i64 0
  %3170 = add nuw nsw i32 %3145, %3143
  %3171 = sext i32 %3170 to i64
  %3172 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3171
  %3173 = addrspacecast ptr %3172 to ptr addrspace(3)
  store float %3169, ptr addrspace(3) %3173, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3174 = extractelement <4 x float> %3097, i64 1
  %3175 = add nuw nsw i32 %3145, %3152
  %3176 = sext i32 %3175 to i64
  %3177 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3176
  %3178 = addrspacecast ptr %3177 to ptr addrspace(3)
  store float %3174, ptr addrspace(3) %3178, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3179 = extractelement <4 x float> %3097, i64 2
  %3180 = add nuw nsw i32 %3145, %3158
  %3181 = sext i32 %3180 to i64
  %3182 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3181
  %3183 = addrspacecast ptr %3182 to ptr addrspace(3)
  store float %3179, ptr addrspace(3) %3183, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3184 = extractelement <4 x float> %3097, i64 3
  %3185 = add nuw nsw i32 %3145, %3164
  %3186 = sext i32 %3185 to i64
  %3187 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3186
  %3188 = addrspacecast ptr %3187 to ptr addrspace(3)
  store float %3184, ptr addrspace(3) %3188, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3189 = or disjoint i32 %3144, 16
  %3190 = extractelement <4 x float> %3113, i64 0
  %3191 = add nuw nsw i32 %3189, %3143
  %3192 = zext nneg i32 %3191 to i64
  %3193 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3192
  %3194 = addrspacecast ptr %3193 to ptr addrspace(3)
  store float %3190, ptr addrspace(3) %3194, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3195 = extractelement <4 x float> %3113, i64 1
  %3196 = add nuw nsw i32 %3189, %3152
  %3197 = zext nneg i32 %3196 to i64
  %3198 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3197
  %3199 = addrspacecast ptr %3198 to ptr addrspace(3)
  store float %3195, ptr addrspace(3) %3199, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3200 = extractelement <4 x float> %3113, i64 2
  %3201 = add nuw nsw i32 %3189, %3158
  %3202 = zext nneg i32 %3201 to i64
  %3203 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3202
  %3204 = addrspacecast ptr %3203 to ptr addrspace(3)
  store float %3200, ptr addrspace(3) %3204, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3205 = extractelement <4 x float> %3113, i64 3
  %3206 = add nuw nsw i32 %3189, %3164
  %3207 = zext nneg i32 %3206 to i64
  %3208 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3207
  %3209 = addrspacecast ptr %3208 to ptr addrspace(3)
  store float %3205, ptr addrspace(3) %3209, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3210 = add nuw i32 %3144, 144
  %3211 = extractelement <4 x float> %3129, i64 0
  %3212 = add nuw nsw i32 %3210, %3143
  %3213 = sext i32 %3212 to i64
  %3214 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3213
  %3215 = addrspacecast ptr %3214 to ptr addrspace(3)
  store float %3211, ptr addrspace(3) %3215, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3216 = extractelement <4 x float> %3129, i64 1
  %3217 = add nuw nsw i32 %3210, %3152
  %3218 = sext i32 %3217 to i64
  %3219 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3218
  %3220 = addrspacecast ptr %3219 to ptr addrspace(3)
  store float %3216, ptr addrspace(3) %3220, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3221 = extractelement <4 x float> %3129, i64 2
  %3222 = add nuw nsw i32 %3210, %3158
  %3223 = sext i32 %3222 to i64
  %3224 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3223
  %3225 = addrspacecast ptr %3224 to ptr addrspace(3)
  store float %3221, ptr addrspace(3) %3225, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3226 = extractelement <4 x float> %3129, i64 3
  %3227 = add nuw nsw i32 %3210, %3164
  %3228 = sext i32 %3227 to i64
  %3229 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3228
  %3230 = addrspacecast ptr %3229 to ptr addrspace(3)
  store float %3226, ptr addrspace(3) %3230, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3231 = extractelement <4 x float> %3082, i64 0
  %3232 = or disjoint i32 %3143, 4096
  %3233 = add nuw nsw i32 %3144, %3232
  %3234 = zext nneg i32 %3233 to i64
  %3235 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3234
  %3236 = addrspacecast ptr %3235 to ptr addrspace(3)
  store float %3231, ptr addrspace(3) %3236, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3237 = extractelement <4 x float> %3082, i64 1
  %3238 = or disjoint i32 %3143, 4352
  %3239 = add nuw nsw i32 %3144, %3238
  %3240 = zext nneg i32 %3239 to i64
  %3241 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3240
  %3242 = addrspacecast ptr %3241 to ptr addrspace(3)
  store float %3237, ptr addrspace(3) %3242, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3243 = extractelement <4 x float> %3082, i64 2
  %3244 = or disjoint i32 %3143, 4608
  %3245 = add nuw nsw i32 %3144, %3244
  %3246 = zext nneg i32 %3245 to i64
  %3247 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3246
  %3248 = addrspacecast ptr %3247 to ptr addrspace(3)
  store float %3243, ptr addrspace(3) %3248, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3249 = extractelement <4 x float> %3082, i64 3
  %3250 = or disjoint i32 %3143, 4864
  %3251 = add nuw nsw i32 %3144, %3250
  %3252 = zext nneg i32 %3251 to i64
  %3253 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3252
  %3254 = addrspacecast ptr %3253 to ptr addrspace(3)
  store float %3249, ptr addrspace(3) %3254, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3255 = extractelement <4 x float> %3098, i64 0
  %3256 = add nuw nsw i32 %3145, %3232
  %3257 = sext i32 %3256 to i64
  %3258 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3257
  %3259 = addrspacecast ptr %3258 to ptr addrspace(3)
  store float %3255, ptr addrspace(3) %3259, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3260 = extractelement <4 x float> %3098, i64 1
  %3261 = add nuw nsw i32 %3145, %3238
  %3262 = sext i32 %3261 to i64
  %3263 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3262
  %3264 = addrspacecast ptr %3263 to ptr addrspace(3)
  store float %3260, ptr addrspace(3) %3264, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3265 = extractelement <4 x float> %3098, i64 2
  %3266 = add nuw nsw i32 %3145, %3244
  %3267 = sext i32 %3266 to i64
  %3268 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3267
  %3269 = addrspacecast ptr %3268 to ptr addrspace(3)
  store float %3265, ptr addrspace(3) %3269, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3270 = extractelement <4 x float> %3098, i64 3
  %3271 = add nuw nsw i32 %3145, %3250
  %3272 = sext i32 %3271 to i64
  %3273 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3272
  %3274 = addrspacecast ptr %3273 to ptr addrspace(3)
  store float %3270, ptr addrspace(3) %3274, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3275 = extractelement <4 x float> %3114, i64 0
  %3276 = add nuw nsw i32 %3189, %3232
  %3277 = zext nneg i32 %3276 to i64
  %3278 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3277
  %3279 = addrspacecast ptr %3278 to ptr addrspace(3)
  store float %3275, ptr addrspace(3) %3279, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3280 = extractelement <4 x float> %3114, i64 1
  %3281 = add nuw nsw i32 %3189, %3238
  %3282 = zext nneg i32 %3281 to i64
  %3283 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3282
  %3284 = addrspacecast ptr %3283 to ptr addrspace(3)
  store float %3280, ptr addrspace(3) %3284, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3285 = extractelement <4 x float> %3114, i64 2
  %3286 = add nuw nsw i32 %3189, %3244
  %3287 = zext nneg i32 %3286 to i64
  %3288 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3287
  %3289 = addrspacecast ptr %3288 to ptr addrspace(3)
  store float %3285, ptr addrspace(3) %3289, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3290 = extractelement <4 x float> %3114, i64 3
  %3291 = add nuw nsw i32 %3189, %3250
  %3292 = zext nneg i32 %3291 to i64
  %3293 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3292
  %3294 = addrspacecast ptr %3293 to ptr addrspace(3)
  store float %3290, ptr addrspace(3) %3294, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3295 = extractelement <4 x float> %3130, i64 0
  %3296 = add nuw nsw i32 %3210, %3232
  %3297 = sext i32 %3296 to i64
  %3298 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3297
  %3299 = addrspacecast ptr %3298 to ptr addrspace(3)
  store float %3295, ptr addrspace(3) %3299, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3300 = extractelement <4 x float> %3130, i64 1
  %3301 = add nuw nsw i32 %3210, %3238
  %3302 = sext i32 %3301 to i64
  %3303 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3302
  %3304 = addrspacecast ptr %3303 to ptr addrspace(3)
  store float %3300, ptr addrspace(3) %3304, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3305 = extractelement <4 x float> %3130, i64 2
  %3306 = add nuw nsw i32 %3210, %3244
  %3307 = sext i32 %3306 to i64
  %3308 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3307
  %3309 = addrspacecast ptr %3308 to ptr addrspace(3)
  store float %3305, ptr addrspace(3) %3309, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3310 = extractelement <4 x float> %3130, i64 3
  %3311 = add nuw nsw i32 %3210, %3250
  %3312 = sext i32 %3311 to i64
  %3313 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3312
  %3314 = addrspacecast ptr %3313 to ptr addrspace(3)
  store float %3310, ptr addrspace(3) %3314, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3315 = extractelement <4 x float> %3085, i64 0
  %3316 = or disjoint i32 %3143, 8192
  %3317 = add nuw nsw i32 %3144, %3316
  %3318 = zext nneg i32 %3317 to i64
  %3319 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3318
  %3320 = addrspacecast ptr %3319 to ptr addrspace(3)
  store float %3315, ptr addrspace(3) %3320, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3321 = extractelement <4 x float> %3085, i64 1
  %3322 = or disjoint i32 %3143, 8448
  %3323 = add nuw nsw i32 %3144, %3322
  %3324 = zext nneg i32 %3323 to i64
  %3325 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3324
  %3326 = addrspacecast ptr %3325 to ptr addrspace(3)
  store float %3321, ptr addrspace(3) %3326, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3327 = extractelement <4 x float> %3085, i64 2
  %3328 = or disjoint i32 %3143, 8704
  %3329 = add nuw nsw i32 %3144, %3328
  %3330 = zext nneg i32 %3329 to i64
  %3331 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3330
  %3332 = addrspacecast ptr %3331 to ptr addrspace(3)
  store float %3327, ptr addrspace(3) %3332, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3333 = extractelement <4 x float> %3085, i64 3
  %3334 = or disjoint i32 %3143, 8960
  %3335 = add nuw nsw i32 %3144, %3334
  %3336 = zext nneg i32 %3335 to i64
  %3337 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3336
  %3338 = addrspacecast ptr %3337 to ptr addrspace(3)
  store float %3333, ptr addrspace(3) %3338, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3339 = extractelement <4 x float> %3101, i64 0
  %3340 = add nuw nsw i32 %3145, %3316
  %3341 = sext i32 %3340 to i64
  %3342 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3341
  %3343 = addrspacecast ptr %3342 to ptr addrspace(3)
  store float %3339, ptr addrspace(3) %3343, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3344 = extractelement <4 x float> %3101, i64 1
  %3345 = add nuw nsw i32 %3145, %3322
  %3346 = sext i32 %3345 to i64
  %3347 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3346
  %3348 = addrspacecast ptr %3347 to ptr addrspace(3)
  store float %3344, ptr addrspace(3) %3348, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3349 = extractelement <4 x float> %3101, i64 2
  %3350 = add nuw nsw i32 %3145, %3328
  %3351 = sext i32 %3350 to i64
  %3352 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3351
  %3353 = addrspacecast ptr %3352 to ptr addrspace(3)
  store float %3349, ptr addrspace(3) %3353, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3354 = extractelement <4 x float> %3101, i64 3
  %3355 = add nuw nsw i32 %3145, %3334
  %3356 = sext i32 %3355 to i64
  %3357 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3356
  %3358 = addrspacecast ptr %3357 to ptr addrspace(3)
  store float %3354, ptr addrspace(3) %3358, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3359 = extractelement <4 x float> %3117, i64 0
  %3360 = add nuw nsw i32 %3189, %3316
  %3361 = zext nneg i32 %3360 to i64
  %3362 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3361
  %3363 = addrspacecast ptr %3362 to ptr addrspace(3)
  store float %3359, ptr addrspace(3) %3363, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3364 = extractelement <4 x float> %3117, i64 1
  %3365 = add nuw nsw i32 %3189, %3322
  %3366 = zext nneg i32 %3365 to i64
  %3367 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3366
  %3368 = addrspacecast ptr %3367 to ptr addrspace(3)
  store float %3364, ptr addrspace(3) %3368, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3369 = extractelement <4 x float> %3117, i64 2
  %3370 = add nuw nsw i32 %3189, %3328
  %3371 = zext nneg i32 %3370 to i64
  %3372 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3371
  %3373 = addrspacecast ptr %3372 to ptr addrspace(3)
  store float %3369, ptr addrspace(3) %3373, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3374 = extractelement <4 x float> %3117, i64 3
  %3375 = add nuw nsw i32 %3189, %3334
  %3376 = zext nneg i32 %3375 to i64
  %3377 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3376
  %3378 = addrspacecast ptr %3377 to ptr addrspace(3)
  store float %3374, ptr addrspace(3) %3378, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3379 = extractelement <4 x float> %3133, i64 0
  %3380 = add nuw nsw i32 %3210, %3316
  %3381 = sext i32 %3380 to i64
  %3382 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3381
  %3383 = addrspacecast ptr %3382 to ptr addrspace(3)
  store float %3379, ptr addrspace(3) %3383, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3384 = extractelement <4 x float> %3133, i64 1
  %3385 = add nuw nsw i32 %3210, %3322
  %3386 = sext i32 %3385 to i64
  %3387 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3386
  %3388 = addrspacecast ptr %3387 to ptr addrspace(3)
  store float %3384, ptr addrspace(3) %3388, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3389 = extractelement <4 x float> %3133, i64 2
  %3390 = add nuw nsw i32 %3210, %3328
  %3391 = sext i32 %3390 to i64
  %3392 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3391
  %3393 = addrspacecast ptr %3392 to ptr addrspace(3)
  store float %3389, ptr addrspace(3) %3393, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3394 = extractelement <4 x float> %3133, i64 3
  %3395 = add nuw nsw i32 %3210, %3334
  %3396 = sext i32 %3395 to i64
  %3397 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3396
  %3398 = addrspacecast ptr %3397 to ptr addrspace(3)
  store float %3394, ptr addrspace(3) %3398, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3399 = extractelement <4 x float> %3086, i64 0
  %3400 = or disjoint i32 %3143, 12288
  %3401 = add nuw nsw i32 %3144, %3400
  %3402 = zext nneg i32 %3401 to i64
  %3403 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3402
  %3404 = addrspacecast ptr %3403 to ptr addrspace(3)
  store float %3399, ptr addrspace(3) %3404, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3405 = extractelement <4 x float> %3086, i64 1
  %3406 = or disjoint i32 %3143, 12544
  %3407 = add nuw nsw i32 %3144, %3406
  %3408 = zext nneg i32 %3407 to i64
  %3409 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3408
  %3410 = addrspacecast ptr %3409 to ptr addrspace(3)
  store float %3405, ptr addrspace(3) %3410, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3411 = extractelement <4 x float> %3086, i64 2
  %3412 = or disjoint i32 %3143, 12800
  %3413 = add nuw nsw i32 %3144, %3412
  %3414 = zext nneg i32 %3413 to i64
  %3415 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3414
  %3416 = addrspacecast ptr %3415 to ptr addrspace(3)
  store float %3411, ptr addrspace(3) %3416, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3417 = extractelement <4 x float> %3086, i64 3
  %3418 = or disjoint i32 %3143, 13056
  %3419 = add nuw nsw i32 %3144, %3418
  %3420 = zext nneg i32 %3419 to i64
  %3421 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3420
  %3422 = addrspacecast ptr %3421 to ptr addrspace(3)
  store float %3417, ptr addrspace(3) %3422, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3423 = extractelement <4 x float> %3102, i64 0
  %3424 = add nuw nsw i32 %3145, %3400
  %3425 = sext i32 %3424 to i64
  %3426 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3425
  %3427 = addrspacecast ptr %3426 to ptr addrspace(3)
  store float %3423, ptr addrspace(3) %3427, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3428 = extractelement <4 x float> %3102, i64 1
  %3429 = add nuw nsw i32 %3145, %3406
  %3430 = sext i32 %3429 to i64
  %3431 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3430
  %3432 = addrspacecast ptr %3431 to ptr addrspace(3)
  store float %3428, ptr addrspace(3) %3432, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3433 = extractelement <4 x float> %3102, i64 2
  %3434 = add nuw nsw i32 %3145, %3412
  %3435 = sext i32 %3434 to i64
  %3436 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3435
  %3437 = addrspacecast ptr %3436 to ptr addrspace(3)
  store float %3433, ptr addrspace(3) %3437, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3438 = extractelement <4 x float> %3102, i64 3
  %3439 = add nuw nsw i32 %3145, %3418
  %3440 = sext i32 %3439 to i64
  %3441 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3440
  %3442 = addrspacecast ptr %3441 to ptr addrspace(3)
  store float %3438, ptr addrspace(3) %3442, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3443 = extractelement <4 x float> %3118, i64 0
  %3444 = add nuw nsw i32 %3189, %3400
  %3445 = zext nneg i32 %3444 to i64
  %3446 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3445
  %3447 = addrspacecast ptr %3446 to ptr addrspace(3)
  store float %3443, ptr addrspace(3) %3447, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3448 = extractelement <4 x float> %3118, i64 1
  %3449 = add nuw nsw i32 %3189, %3406
  %3450 = zext nneg i32 %3449 to i64
  %3451 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3450
  %3452 = addrspacecast ptr %3451 to ptr addrspace(3)
  store float %3448, ptr addrspace(3) %3452, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3453 = extractelement <4 x float> %3118, i64 2
  %3454 = add nuw nsw i32 %3189, %3412
  %3455 = zext nneg i32 %3454 to i64
  %3456 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3455
  %3457 = addrspacecast ptr %3456 to ptr addrspace(3)
  store float %3453, ptr addrspace(3) %3457, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3458 = extractelement <4 x float> %3118, i64 3
  %3459 = add nuw nsw i32 %3189, %3418
  %3460 = zext nneg i32 %3459 to i64
  %3461 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3460
  %3462 = addrspacecast ptr %3461 to ptr addrspace(3)
  store float %3458, ptr addrspace(3) %3462, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3463 = extractelement <4 x float> %3134, i64 0
  %3464 = add nuw nsw i32 %3210, %3400
  %3465 = sext i32 %3464 to i64
  %3466 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3465
  %3467 = addrspacecast ptr %3466 to ptr addrspace(3)
  store float %3463, ptr addrspace(3) %3467, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3468 = extractelement <4 x float> %3134, i64 1
  %3469 = add nuw nsw i32 %3210, %3406
  %3470 = sext i32 %3469 to i64
  %3471 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3470
  %3472 = addrspacecast ptr %3471 to ptr addrspace(3)
  store float %3468, ptr addrspace(3) %3472, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3473 = extractelement <4 x float> %3134, i64 2
  %3474 = add nuw nsw i32 %3210, %3412
  %3475 = sext i32 %3474 to i64
  %3476 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3475
  %3477 = addrspacecast ptr %3476 to ptr addrspace(3)
  store float %3473, ptr addrspace(3) %3477, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3478 = extractelement <4 x float> %3134, i64 3
  %3479 = add nuw nsw i32 %3210, %3418
  %3480 = sext i32 %3479 to i64
  %3481 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3480
  %3482 = addrspacecast ptr %3481 to ptr addrspace(3)
  store float %3478, ptr addrspace(3) %3482, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3483 = extractelement <4 x float> %3089, i64 0
  %3484 = or disjoint i32 %3143, 16384
  %3485 = add nuw nsw i32 %3144, %3484
  %3486 = zext nneg i32 %3485 to i64
  %3487 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3486
  %3488 = addrspacecast ptr %3487 to ptr addrspace(3)
  store float %3483, ptr addrspace(3) %3488, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3489 = extractelement <4 x float> %3089, i64 1
  %3490 = or disjoint i32 %3143, 16640
  %3491 = add nuw nsw i32 %3144, %3490
  %3492 = zext nneg i32 %3491 to i64
  %3493 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3492
  %3494 = addrspacecast ptr %3493 to ptr addrspace(3)
  store float %3489, ptr addrspace(3) %3494, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3495 = extractelement <4 x float> %3089, i64 2
  %3496 = or disjoint i32 %3143, 16896
  %3497 = add nuw nsw i32 %3144, %3496
  %3498 = zext nneg i32 %3497 to i64
  %3499 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3498
  %3500 = addrspacecast ptr %3499 to ptr addrspace(3)
  store float %3495, ptr addrspace(3) %3500, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3501 = extractelement <4 x float> %3089, i64 3
  %3502 = or disjoint i32 %3143, 17152
  %3503 = add nuw nsw i32 %3144, %3502
  %3504 = zext nneg i32 %3503 to i64
  %3505 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3504
  %3506 = addrspacecast ptr %3505 to ptr addrspace(3)
  store float %3501, ptr addrspace(3) %3506, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3507 = extractelement <4 x float> %3105, i64 0
  %3508 = add nuw nsw i32 %3145, %3484
  %3509 = sext i32 %3508 to i64
  %3510 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3509
  %3511 = addrspacecast ptr %3510 to ptr addrspace(3)
  store float %3507, ptr addrspace(3) %3511, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3512 = extractelement <4 x float> %3105, i64 1
  %3513 = add nuw nsw i32 %3145, %3490
  %3514 = sext i32 %3513 to i64
  %3515 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3514
  %3516 = addrspacecast ptr %3515 to ptr addrspace(3)
  store float %3512, ptr addrspace(3) %3516, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3517 = extractelement <4 x float> %3105, i64 2
  %3518 = add nuw nsw i32 %3145, %3496
  %3519 = sext i32 %3518 to i64
  %3520 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3519
  %3521 = addrspacecast ptr %3520 to ptr addrspace(3)
  store float %3517, ptr addrspace(3) %3521, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3522 = extractelement <4 x float> %3105, i64 3
  %3523 = add nuw nsw i32 %3145, %3502
  %3524 = sext i32 %3523 to i64
  %3525 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3524
  %3526 = addrspacecast ptr %3525 to ptr addrspace(3)
  store float %3522, ptr addrspace(3) %3526, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3527 = extractelement <4 x float> %3121, i64 0
  %3528 = add nuw nsw i32 %3189, %3484
  %3529 = zext nneg i32 %3528 to i64
  %3530 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3529
  %3531 = addrspacecast ptr %3530 to ptr addrspace(3)
  store float %3527, ptr addrspace(3) %3531, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3532 = extractelement <4 x float> %3121, i64 1
  %3533 = add nuw nsw i32 %3189, %3490
  %3534 = zext nneg i32 %3533 to i64
  %3535 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3534
  %3536 = addrspacecast ptr %3535 to ptr addrspace(3)
  store float %3532, ptr addrspace(3) %3536, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3537 = extractelement <4 x float> %3121, i64 2
  %3538 = add nuw nsw i32 %3189, %3496
  %3539 = zext nneg i32 %3538 to i64
  %3540 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3539
  %3541 = addrspacecast ptr %3540 to ptr addrspace(3)
  store float %3537, ptr addrspace(3) %3541, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3542 = extractelement <4 x float> %3121, i64 3
  %3543 = add nuw nsw i32 %3189, %3502
  %3544 = zext nneg i32 %3543 to i64
  %3545 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3544
  %3546 = addrspacecast ptr %3545 to ptr addrspace(3)
  store float %3542, ptr addrspace(3) %3546, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3547 = extractelement <4 x float> %3137, i64 0
  %3548 = add nuw nsw i32 %3210, %3484
  %3549 = sext i32 %3548 to i64
  %3550 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3549
  %3551 = addrspacecast ptr %3550 to ptr addrspace(3)
  store float %3547, ptr addrspace(3) %3551, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3552 = extractelement <4 x float> %3137, i64 1
  %3553 = add nuw nsw i32 %3210, %3490
  %3554 = sext i32 %3553 to i64
  %3555 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3554
  %3556 = addrspacecast ptr %3555 to ptr addrspace(3)
  store float %3552, ptr addrspace(3) %3556, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3557 = extractelement <4 x float> %3137, i64 2
  %3558 = add nuw nsw i32 %3210, %3496
  %3559 = sext i32 %3558 to i64
  %3560 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3559
  %3561 = addrspacecast ptr %3560 to ptr addrspace(3)
  store float %3557, ptr addrspace(3) %3561, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3562 = extractelement <4 x float> %3137, i64 3
  %3563 = add nuw nsw i32 %3210, %3502
  %3564 = sext i32 %3563 to i64
  %3565 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3564
  %3566 = addrspacecast ptr %3565 to ptr addrspace(3)
  store float %3562, ptr addrspace(3) %3566, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3567 = extractelement <4 x float> %3090, i64 0
  %3568 = or disjoint i32 %3143, 20480
  %3569 = add nuw nsw i32 %3144, %3568
  %3570 = zext nneg i32 %3569 to i64
  %3571 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3570
  %3572 = addrspacecast ptr %3571 to ptr addrspace(3)
  store float %3567, ptr addrspace(3) %3572, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3573 = extractelement <4 x float> %3090, i64 1
  %3574 = or disjoint i32 %3143, 20736
  %3575 = add nuw nsw i32 %3144, %3574
  %3576 = zext nneg i32 %3575 to i64
  %3577 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3576
  %3578 = addrspacecast ptr %3577 to ptr addrspace(3)
  store float %3573, ptr addrspace(3) %3578, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3579 = extractelement <4 x float> %3090, i64 2
  %3580 = or disjoint i32 %3143, 20992
  %3581 = add nuw nsw i32 %3144, %3580
  %3582 = zext nneg i32 %3581 to i64
  %3583 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3582
  %3584 = addrspacecast ptr %3583 to ptr addrspace(3)
  store float %3579, ptr addrspace(3) %3584, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3585 = extractelement <4 x float> %3090, i64 3
  %3586 = or disjoint i32 %3143, 21248
  %3587 = add nuw nsw i32 %3144, %3586
  %3588 = zext nneg i32 %3587 to i64
  %3589 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3588
  %3590 = addrspacecast ptr %3589 to ptr addrspace(3)
  store float %3585, ptr addrspace(3) %3590, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3591 = extractelement <4 x float> %3106, i64 0
  %3592 = add nuw nsw i32 %3145, %3568
  %3593 = sext i32 %3592 to i64
  %3594 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3593
  %3595 = addrspacecast ptr %3594 to ptr addrspace(3)
  store float %3591, ptr addrspace(3) %3595, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3596 = extractelement <4 x float> %3106, i64 1
  %3597 = add nuw nsw i32 %3145, %3574
  %3598 = sext i32 %3597 to i64
  %3599 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3598
  %3600 = addrspacecast ptr %3599 to ptr addrspace(3)
  store float %3596, ptr addrspace(3) %3600, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3601 = extractelement <4 x float> %3106, i64 2
  %3602 = add nuw nsw i32 %3145, %3580
  %3603 = sext i32 %3602 to i64
  %3604 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3603
  %3605 = addrspacecast ptr %3604 to ptr addrspace(3)
  store float %3601, ptr addrspace(3) %3605, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3606 = extractelement <4 x float> %3106, i64 3
  %3607 = add nuw nsw i32 %3145, %3586
  %3608 = sext i32 %3607 to i64
  %3609 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3608
  %3610 = addrspacecast ptr %3609 to ptr addrspace(3)
  store float %3606, ptr addrspace(3) %3610, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3611 = extractelement <4 x float> %3122, i64 0
  %3612 = add nuw nsw i32 %3189, %3568
  %3613 = zext nneg i32 %3612 to i64
  %3614 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3613
  %3615 = addrspacecast ptr %3614 to ptr addrspace(3)
  store float %3611, ptr addrspace(3) %3615, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3616 = extractelement <4 x float> %3122, i64 1
  %3617 = add nuw nsw i32 %3189, %3574
  %3618 = zext nneg i32 %3617 to i64
  %3619 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3618
  %3620 = addrspacecast ptr %3619 to ptr addrspace(3)
  store float %3616, ptr addrspace(3) %3620, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3621 = extractelement <4 x float> %3122, i64 2
  %3622 = add nuw nsw i32 %3189, %3580
  %3623 = zext nneg i32 %3622 to i64
  %3624 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3623
  %3625 = addrspacecast ptr %3624 to ptr addrspace(3)
  store float %3621, ptr addrspace(3) %3625, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3626 = extractelement <4 x float> %3122, i64 3
  %3627 = add nuw nsw i32 %3189, %3586
  %3628 = zext nneg i32 %3627 to i64
  %3629 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3628
  %3630 = addrspacecast ptr %3629 to ptr addrspace(3)
  store float %3626, ptr addrspace(3) %3630, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3631 = extractelement <4 x float> %3138, i64 0
  %3632 = add nuw nsw i32 %3210, %3568
  %3633 = sext i32 %3632 to i64
  %3634 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3633
  %3635 = addrspacecast ptr %3634 to ptr addrspace(3)
  store float %3631, ptr addrspace(3) %3635, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3636 = extractelement <4 x float> %3138, i64 1
  %3637 = add nuw nsw i32 %3210, %3574
  %3638 = sext i32 %3637 to i64
  %3639 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3638
  %3640 = addrspacecast ptr %3639 to ptr addrspace(3)
  store float %3636, ptr addrspace(3) %3640, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3641 = extractelement <4 x float> %3138, i64 2
  %3642 = add nuw nsw i32 %3210, %3580
  %3643 = sext i32 %3642 to i64
  %3644 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3643
  %3645 = addrspacecast ptr %3644 to ptr addrspace(3)
  store float %3641, ptr addrspace(3) %3645, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3646 = extractelement <4 x float> %3138, i64 3
  %3647 = add nuw nsw i32 %3210, %3586
  %3648 = sext i32 %3647 to i64
  %3649 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3648
  %3650 = addrspacecast ptr %3649 to ptr addrspace(3)
  store float %3646, ptr addrspace(3) %3650, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3651 = extractelement <4 x float> %3093, i64 0
  %3652 = or disjoint i32 %3143, 24576
  %3653 = add nuw nsw i32 %3144, %3652
  %3654 = zext nneg i32 %3653 to i64
  %3655 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3654
  %3656 = addrspacecast ptr %3655 to ptr addrspace(3)
  store float %3651, ptr addrspace(3) %3656, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3657 = extractelement <4 x float> %3093, i64 1
  %3658 = or disjoint i32 %3143, 24832
  %3659 = add nuw nsw i32 %3144, %3658
  %3660 = zext nneg i32 %3659 to i64
  %3661 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3660
  %3662 = addrspacecast ptr %3661 to ptr addrspace(3)
  store float %3657, ptr addrspace(3) %3662, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3663 = extractelement <4 x float> %3093, i64 2
  %3664 = or disjoint i32 %3143, 25088
  %3665 = add nuw nsw i32 %3144, %3664
  %3666 = zext nneg i32 %3665 to i64
  %3667 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3666
  %3668 = addrspacecast ptr %3667 to ptr addrspace(3)
  store float %3663, ptr addrspace(3) %3668, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3669 = extractelement <4 x float> %3093, i64 3
  %3670 = or disjoint i32 %3143, 25344
  %3671 = add nuw nsw i32 %3144, %3670
  %3672 = zext nneg i32 %3671 to i64
  %3673 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3672
  %3674 = addrspacecast ptr %3673 to ptr addrspace(3)
  store float %3669, ptr addrspace(3) %3674, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3675 = extractelement <4 x float> %3109, i64 0
  %3676 = add nuw nsw i32 %3145, %3652
  %3677 = sext i32 %3676 to i64
  %3678 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3677
  %3679 = addrspacecast ptr %3678 to ptr addrspace(3)
  store float %3675, ptr addrspace(3) %3679, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3680 = extractelement <4 x float> %3109, i64 1
  %3681 = add nuw nsw i32 %3145, %3658
  %3682 = sext i32 %3681 to i64
  %3683 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3682
  %3684 = addrspacecast ptr %3683 to ptr addrspace(3)
  store float %3680, ptr addrspace(3) %3684, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3685 = extractelement <4 x float> %3109, i64 2
  %3686 = add nuw nsw i32 %3145, %3664
  %3687 = sext i32 %3686 to i64
  %3688 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3687
  %3689 = addrspacecast ptr %3688 to ptr addrspace(3)
  store float %3685, ptr addrspace(3) %3689, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3690 = extractelement <4 x float> %3109, i64 3
  %3691 = add nuw nsw i32 %3145, %3670
  %3692 = sext i32 %3691 to i64
  %3693 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3692
  %3694 = addrspacecast ptr %3693 to ptr addrspace(3)
  store float %3690, ptr addrspace(3) %3694, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3695 = extractelement <4 x float> %3125, i64 0
  %3696 = add nuw nsw i32 %3189, %3652
  %3697 = zext nneg i32 %3696 to i64
  %3698 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3697
  %3699 = addrspacecast ptr %3698 to ptr addrspace(3)
  store float %3695, ptr addrspace(3) %3699, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3700 = extractelement <4 x float> %3125, i64 1
  %3701 = add nuw nsw i32 %3189, %3658
  %3702 = zext nneg i32 %3701 to i64
  %3703 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3702
  %3704 = addrspacecast ptr %3703 to ptr addrspace(3)
  store float %3700, ptr addrspace(3) %3704, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3705 = extractelement <4 x float> %3125, i64 2
  %3706 = add nuw nsw i32 %3189, %3664
  %3707 = zext nneg i32 %3706 to i64
  %3708 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3707
  %3709 = addrspacecast ptr %3708 to ptr addrspace(3)
  store float %3705, ptr addrspace(3) %3709, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3710 = extractelement <4 x float> %3125, i64 3
  %3711 = add nuw nsw i32 %3189, %3670
  %3712 = zext nneg i32 %3711 to i64
  %3713 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3712
  %3714 = addrspacecast ptr %3713 to ptr addrspace(3)
  store float %3710, ptr addrspace(3) %3714, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3715 = extractelement <4 x float> %3141, i64 0
  %3716 = add nuw nsw i32 %3210, %3652
  %3717 = sext i32 %3716 to i64
  %3718 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3717
  %3719 = addrspacecast ptr %3718 to ptr addrspace(3)
  store float %3715, ptr addrspace(3) %3719, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3720 = extractelement <4 x float> %3141, i64 1
  %3721 = add nuw nsw i32 %3210, %3658
  %3722 = sext i32 %3721 to i64
  %3723 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3722
  %3724 = addrspacecast ptr %3723 to ptr addrspace(3)
  store float %3720, ptr addrspace(3) %3724, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3725 = extractelement <4 x float> %3141, i64 2
  %3726 = add nuw nsw i32 %3210, %3664
  %3727 = sext i32 %3726 to i64
  %3728 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3727
  %3729 = addrspacecast ptr %3728 to ptr addrspace(3)
  store float %3725, ptr addrspace(3) %3729, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3730 = extractelement <4 x float> %3141, i64 3
  %3731 = add nuw nsw i32 %3210, %3670
  %3732 = sext i32 %3731 to i64
  %3733 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3732
  %3734 = addrspacecast ptr %3733 to ptr addrspace(3)
  store float %3730, ptr addrspace(3) %3734, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3735 = extractelement <4 x float> %3094, i64 0
  %3736 = or disjoint i32 %3143, 28672
  %3737 = add nuw nsw i32 %3144, %3736
  %3738 = zext nneg i32 %3737 to i64
  %3739 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3738
  %3740 = addrspacecast ptr %3739 to ptr addrspace(3)
  store float %3735, ptr addrspace(3) %3740, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3741 = extractelement <4 x float> %3094, i64 1
  %3742 = or disjoint i32 %3143, 28928
  %3743 = add nuw nsw i32 %3144, %3742
  %3744 = zext nneg i32 %3743 to i64
  %3745 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3744
  %3746 = addrspacecast ptr %3745 to ptr addrspace(3)
  store float %3741, ptr addrspace(3) %3746, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3747 = extractelement <4 x float> %3094, i64 2
  %3748 = or disjoint i32 %3143, 29184
  %3749 = add nuw nsw i32 %3144, %3748
  %3750 = zext nneg i32 %3749 to i64
  %3751 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3750
  %3752 = addrspacecast ptr %3751 to ptr addrspace(3)
  store float %3747, ptr addrspace(3) %3752, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3753 = extractelement <4 x float> %3094, i64 3
  %3754 = or disjoint i32 %3143, 29440
  %3755 = add nuw nsw i32 %3144, %3754
  %3756 = zext nneg i32 %3755 to i64
  %3757 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3756
  %3758 = addrspacecast ptr %3757 to ptr addrspace(3)
  store float %3753, ptr addrspace(3) %3758, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3759 = extractelement <4 x float> %3110, i64 0
  %3760 = add nuw nsw i32 %3145, %3736
  %3761 = sext i32 %3760 to i64
  %3762 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3761
  %3763 = addrspacecast ptr %3762 to ptr addrspace(3)
  store float %3759, ptr addrspace(3) %3763, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3764 = extractelement <4 x float> %3110, i64 1
  %3765 = add nuw nsw i32 %3145, %3742
  %3766 = sext i32 %3765 to i64
  %3767 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3766
  %3768 = addrspacecast ptr %3767 to ptr addrspace(3)
  store float %3764, ptr addrspace(3) %3768, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3769 = extractelement <4 x float> %3110, i64 2
  %3770 = add nuw nsw i32 %3145, %3748
  %3771 = sext i32 %3770 to i64
  %3772 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3771
  %3773 = addrspacecast ptr %3772 to ptr addrspace(3)
  store float %3769, ptr addrspace(3) %3773, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3774 = extractelement <4 x float> %3110, i64 3
  %3775 = add nuw nsw i32 %3145, %3754
  %3776 = sext i32 %3775 to i64
  %3777 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3776
  %3778 = addrspacecast ptr %3777 to ptr addrspace(3)
  store float %3774, ptr addrspace(3) %3778, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3779 = extractelement <4 x float> %3126, i64 0
  %3780 = add nuw nsw i32 %3189, %3736
  %3781 = zext nneg i32 %3780 to i64
  %3782 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3781
  %3783 = addrspacecast ptr %3782 to ptr addrspace(3)
  store float %3779, ptr addrspace(3) %3783, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3784 = extractelement <4 x float> %3126, i64 1
  %3785 = add nuw nsw i32 %3189, %3742
  %3786 = zext nneg i32 %3785 to i64
  %3787 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3786
  %3788 = addrspacecast ptr %3787 to ptr addrspace(3)
  store float %3784, ptr addrspace(3) %3788, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3789 = extractelement <4 x float> %3126, i64 2
  %3790 = add nuw nsw i32 %3189, %3748
  %3791 = zext nneg i32 %3790 to i64
  %3792 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3791
  %3793 = addrspacecast ptr %3792 to ptr addrspace(3)
  store float %3789, ptr addrspace(3) %3793, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3794 = extractelement <4 x float> %3126, i64 3
  %3795 = add nuw nsw i32 %3189, %3754
  %3796 = zext nneg i32 %3795 to i64
  %3797 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3796
  %3798 = addrspacecast ptr %3797 to ptr addrspace(3)
  store float %3794, ptr addrspace(3) %3798, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3799 = extractelement <4 x float> %3142, i64 0
  %3800 = add nuw nsw i32 %3210, %3736
  %3801 = sext i32 %3800 to i64
  %3802 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3801
  %3803 = addrspacecast ptr %3802 to ptr addrspace(3)
  store float %3799, ptr addrspace(3) %3803, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3804 = extractelement <4 x float> %3142, i64 1
  %3805 = add nuw nsw i32 %3210, %3742
  %3806 = sext i32 %3805 to i64
  %3807 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3806
  %3808 = addrspacecast ptr %3807 to ptr addrspace(3)
  store float %3804, ptr addrspace(3) %3808, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3809 = extractelement <4 x float> %3142, i64 2
  %3810 = add nuw nsw i32 %3210, %3748
  %3811 = sext i32 %3810 to i64
  %3812 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3811
  %3813 = addrspacecast ptr %3812 to ptr addrspace(3)
  store float %3809, ptr addrspace(3) %3813, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3814 = extractelement <4 x float> %3142, i64 3
  %3815 = add nuw nsw i32 %3210, %3754
  %3816 = sext i32 %3815 to i64
  %3817 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3816
  %3818 = addrspacecast ptr %3817 to ptr addrspace(3)
  store float %3814, ptr addrspace(3) %3818, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier(), !noalias !25
  fence syncscope("workgroup") acquire
  %3819 = lshr i32 %21, 4
  %3820 = lshr i32 %21, 2
  %3821 = and i32 %3820, 3
  %3822 = and i32 %21, 3
  %3823 = shl nuw nsw i32 %3822, 3
  %3824 = shl nuw nsw i32 %3821, 5
  %3825 = or disjoint i32 %3824, %3823
  %3826 = or disjoint i32 %3825, 128
  %3827 = shl nsw i32 %44, 6
  %3828 = shl nuw nsw i32 %3821, 4
  %3829 = shl nuw nsw i32 %3822, 2
  %3830 = or disjoint i32 %48, %3819
  %3831 = shl nuw nsw i32 %3819, 8
  %3832 = or disjoint i32 %3825, %3831
  %3833 = or disjoint i32 %3826, %3831
  %3834 = zext nneg i32 %3832 to i64
  %3835 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3834
  %3836 = addrspacecast ptr %3835 to ptr addrspace(3)
  %3837 = load float, ptr addrspace(3) %3836, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3838 = zext nneg i32 %3833 to i64
  %3839 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3838
  %3840 = addrspacecast ptr %3839 to ptr addrspace(3)
  %3841 = load float, ptr addrspace(3) %3840, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3842 = or disjoint i32 %3832, 1
  %3843 = zext nneg i32 %3842 to i64
  %3844 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3843
  %3845 = addrspacecast ptr %3844 to ptr addrspace(3)
  %3846 = load float, ptr addrspace(3) %3845, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3847 = or disjoint i32 %3833, 1
  %3848 = zext nneg i32 %3847 to i64
  %3849 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3848
  %3850 = addrspacecast ptr %3849 to ptr addrspace(3)
  %3851 = load float, ptr addrspace(3) %3850, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3852 = or disjoint i32 %3832, 2
  %3853 = zext nneg i32 %3852 to i64
  %3854 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3853
  %3855 = addrspacecast ptr %3854 to ptr addrspace(3)
  %3856 = load float, ptr addrspace(3) %3855, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3857 = or disjoint i32 %3833, 2
  %3858 = zext nneg i32 %3857 to i64
  %3859 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3858
  %3860 = addrspacecast ptr %3859 to ptr addrspace(3)
  %3861 = load float, ptr addrspace(3) %3860, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3862 = or disjoint i32 %3832, 3
  %3863 = zext nneg i32 %3862 to i64
  %3864 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3863
  %3865 = addrspacecast ptr %3864 to ptr addrspace(3)
  %3866 = load float, ptr addrspace(3) %3865, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3867 = or disjoint i32 %3833, 3
  %3868 = zext nneg i32 %3867 to i64
  %3869 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3868
  %3870 = addrspacecast ptr %3869 to ptr addrspace(3)
  %3871 = load float, ptr addrspace(3) %3870, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3872 = or disjoint i32 %3832, 4
  %3873 = zext nneg i32 %3872 to i64
  %3874 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3873
  %3875 = addrspacecast ptr %3874 to ptr addrspace(3)
  %3876 = load float, ptr addrspace(3) %3875, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3877 = or disjoint i32 %3833, 4
  %3878 = zext nneg i32 %3877 to i64
  %3879 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3878
  %3880 = addrspacecast ptr %3879 to ptr addrspace(3)
  %3881 = load float, ptr addrspace(3) %3880, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3882 = or disjoint i32 %3832, 5
  %3883 = zext nneg i32 %3882 to i64
  %3884 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3883
  %3885 = addrspacecast ptr %3884 to ptr addrspace(3)
  %3886 = load float, ptr addrspace(3) %3885, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3887 = or disjoint i32 %3833, 5
  %3888 = zext nneg i32 %3887 to i64
  %3889 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3888
  %3890 = addrspacecast ptr %3889 to ptr addrspace(3)
  %3891 = load float, ptr addrspace(3) %3890, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3892 = or disjoint i32 %3832, 6
  %3893 = zext nneg i32 %3892 to i64
  %3894 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3893
  %3895 = addrspacecast ptr %3894 to ptr addrspace(3)
  %3896 = load float, ptr addrspace(3) %3895, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3897 = or disjoint i32 %3833, 6
  %3898 = zext nneg i32 %3897 to i64
  %3899 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3898
  %3900 = addrspacecast ptr %3899 to ptr addrspace(3)
  %3901 = load float, ptr addrspace(3) %3900, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3902 = or disjoint i32 %3832, 7
  %3903 = zext nneg i32 %3902 to i64
  %3904 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3903
  %3905 = addrspacecast ptr %3904 to ptr addrspace(3)
  %3906 = load float, ptr addrspace(3) %3905, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3907 = or disjoint i32 %3833, 7
  %3908 = zext nneg i32 %3907 to i64
  %3909 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3908
  %3910 = addrspacecast ptr %3909 to ptr addrspace(3)
  %3911 = load float, ptr addrspace(3) %3910, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %3912 = fmul contract float %3837, 0xBFF7154760000000
  %3913 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %3912)
  %3914 = fadd contract float %3913, 1.000000e+00
  %3915 = tail call contract float @llvm.amdgcn.rcp.f32(float %3914)
  %3916 = fmul contract float %3837, %3915
  %3917 = fmul contract float %3841, %3916
  %3918 = fmul contract float %3846, 0xBFF7154760000000
  %3919 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %3918)
  %3920 = fadd contract float %3919, 1.000000e+00
  %3921 = tail call contract float @llvm.amdgcn.rcp.f32(float %3920)
  %3922 = fmul contract float %3846, %3921
  %3923 = fmul contract float %3851, %3922
  %3924 = fmul contract float %3856, 0xBFF7154760000000
  %3925 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %3924)
  %3926 = fadd contract float %3925, 1.000000e+00
  %3927 = tail call contract float @llvm.amdgcn.rcp.f32(float %3926)
  %3928 = fmul contract float %3856, %3927
  %3929 = fmul contract float %3861, %3928
  %3930 = fmul contract float %3866, 0xBFF7154760000000
  %3931 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %3930)
  %3932 = fadd contract float %3931, 1.000000e+00
  %3933 = tail call contract float @llvm.amdgcn.rcp.f32(float %3932)
  %3934 = fmul contract float %3866, %3933
  %3935 = fmul contract float %3871, %3934
  %3936 = fmul contract float %3876, 0xBFF7154760000000
  %3937 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %3936)
  %3938 = fadd contract float %3937, 1.000000e+00
  %3939 = tail call contract float @llvm.amdgcn.rcp.f32(float %3938)
  %3940 = fmul contract float %3876, %3939
  %3941 = fmul contract float %3881, %3940
  %3942 = fmul contract float %3886, 0xBFF7154760000000
  %3943 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %3942)
  %3944 = fadd contract float %3943, 1.000000e+00
  %3945 = tail call contract float @llvm.amdgcn.rcp.f32(float %3944)
  %3946 = fmul contract float %3886, %3945
  %3947 = fmul contract float %3891, %3946
  %3948 = fmul contract float %3896, 0xBFF7154760000000
  %3949 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %3948)
  %3950 = fadd contract float %3949, 1.000000e+00
  %3951 = tail call contract float @llvm.amdgcn.rcp.f32(float %3950)
  %3952 = fmul contract float %3896, %3951
  %3953 = fmul contract float %3901, %3952
  %3954 = fmul contract float %3906, 0xBFF7154760000000
  %3955 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %3954)
  %3956 = fadd contract float %3955, 1.000000e+00
  %3957 = tail call contract float @llvm.amdgcn.rcp.f32(float %3956)
  %3958 = fmul contract float %3906, %3957
  %3959 = fmul contract float %3911, %3958
  %3960 = tail call contract noundef float @llvm.fabs.f32(float %3917)
  %3961 = tail call contract noundef float @llvm.fabs.f32(float %3923)
  %3962 = tail call contract noundef float @llvm.maxnum.f32(float %3960, float %3961)
  %3963 = tail call contract noundef float @llvm.fabs.f32(float %3929)
  %3964 = tail call contract noundef float @llvm.maxnum.f32(float %3962, float %3963)
  %3965 = tail call contract noundef float @llvm.fabs.f32(float %3935)
  %3966 = tail call contract noundef float @llvm.maxnum.f32(float %3964, float %3965)
  %3967 = tail call contract noundef float @llvm.fabs.f32(float %3941)
  %3968 = tail call contract noundef float @llvm.maxnum.f32(float %3966, float %3967)
  %3969 = tail call contract noundef float @llvm.fabs.f32(float %3947)
  %3970 = tail call contract noundef float @llvm.maxnum.f32(float %3968, float %3969)
  %3971 = tail call contract noundef float @llvm.fabs.f32(float %3953)
  %3972 = tail call contract noundef float @llvm.maxnum.f32(float %3970, float %3971)
  %3973 = tail call contract noundef float @llvm.fabs.f32(float %3959)
  %3974 = tail call contract noundef float @llvm.maxnum.f32(float %3972, float %3973)
  %3975 = bitcast float %3974 to i32
  %3976 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %3975, i32 177, i32 15, i32 15, i1 true)
  %3977 = bitcast i32 %3976 to float
  %3978 = tail call contract noundef float @llvm.maxnum.f32(float %3974, float %3977)
  %3979 = bitcast float %3978 to i32
  %3980 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %3979, i32 78, i32 15, i32 15, i1 true)
  %3981 = bitcast i32 %3980 to float
  %3982 = tail call contract noundef float @llvm.maxnum.f32(float %3978, float %3981)
  %3983 = bitcast float %3982 to i32
  %3984 = add i32 %3983, 2097152
  %3985 = bitcast i32 %3984 to float
  %3986 = fmul contract float %3985, 2.500000e-01
  %3987 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 0, float %3917, float %3923, float %3986, i32 0)
  %3988 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %3987, float %3929, float %3935, float %3986, i32 1)
  %3989 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %3988, float %3941, float %3947, float %3986, i32 2)
  %3990 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %3989, float %3953, float %3959, float %3986, i32 3)
  %3991 = shl nsw i32 %3830, 8
  %3992 = or disjoint i32 %3829, %3827
  %3993 = or disjoint i32 %3992, %3828
  %3994 = add i32 %3993, %3991
  %3995 = sext i32 %3994 to i64
  %3996 = getelementptr inbounds i8, ptr %39, i64 %3995
  store i32 %3990, ptr %3996, align 4, !tbaa !7, !alias.scope !16, !noalias !26, !nontemporal !27
  %3997 = or disjoint i32 %3831, 4096
  %3998 = or disjoint i32 %3825, %3997
  %3999 = or disjoint i32 %3826, %3997
  %4000 = zext nneg i32 %3998 to i64
  %4001 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4000
  %4002 = addrspacecast ptr %4001 to ptr addrspace(3)
  %4003 = load float, ptr addrspace(3) %4002, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4004 = zext nneg i32 %3999 to i64
  %4005 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4004
  %4006 = addrspacecast ptr %4005 to ptr addrspace(3)
  %4007 = load float, ptr addrspace(3) %4006, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4008 = or disjoint i32 %3998, 1
  %4009 = zext nneg i32 %4008 to i64
  %4010 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4009
  %4011 = addrspacecast ptr %4010 to ptr addrspace(3)
  %4012 = load float, ptr addrspace(3) %4011, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4013 = or disjoint i32 %3999, 1
  %4014 = zext nneg i32 %4013 to i64
  %4015 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4014
  %4016 = addrspacecast ptr %4015 to ptr addrspace(3)
  %4017 = load float, ptr addrspace(3) %4016, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4018 = or disjoint i32 %3998, 2
  %4019 = zext nneg i32 %4018 to i64
  %4020 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4019
  %4021 = addrspacecast ptr %4020 to ptr addrspace(3)
  %4022 = load float, ptr addrspace(3) %4021, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4023 = or disjoint i32 %3999, 2
  %4024 = zext nneg i32 %4023 to i64
  %4025 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4024
  %4026 = addrspacecast ptr %4025 to ptr addrspace(3)
  %4027 = load float, ptr addrspace(3) %4026, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4028 = or disjoint i32 %3998, 3
  %4029 = zext nneg i32 %4028 to i64
  %4030 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4029
  %4031 = addrspacecast ptr %4030 to ptr addrspace(3)
  %4032 = load float, ptr addrspace(3) %4031, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4033 = or disjoint i32 %3999, 3
  %4034 = zext nneg i32 %4033 to i64
  %4035 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4034
  %4036 = addrspacecast ptr %4035 to ptr addrspace(3)
  %4037 = load float, ptr addrspace(3) %4036, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4038 = or disjoint i32 %3998, 4
  %4039 = zext nneg i32 %4038 to i64
  %4040 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4039
  %4041 = addrspacecast ptr %4040 to ptr addrspace(3)
  %4042 = load float, ptr addrspace(3) %4041, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4043 = or disjoint i32 %3999, 4
  %4044 = zext nneg i32 %4043 to i64
  %4045 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4044
  %4046 = addrspacecast ptr %4045 to ptr addrspace(3)
  %4047 = load float, ptr addrspace(3) %4046, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4048 = or disjoint i32 %3998, 5
  %4049 = zext nneg i32 %4048 to i64
  %4050 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4049
  %4051 = addrspacecast ptr %4050 to ptr addrspace(3)
  %4052 = load float, ptr addrspace(3) %4051, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4053 = or disjoint i32 %3999, 5
  %4054 = zext nneg i32 %4053 to i64
  %4055 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4054
  %4056 = addrspacecast ptr %4055 to ptr addrspace(3)
  %4057 = load float, ptr addrspace(3) %4056, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4058 = or disjoint i32 %3998, 6
  %4059 = zext nneg i32 %4058 to i64
  %4060 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4059
  %4061 = addrspacecast ptr %4060 to ptr addrspace(3)
  %4062 = load float, ptr addrspace(3) %4061, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4063 = or disjoint i32 %3999, 6
  %4064 = zext nneg i32 %4063 to i64
  %4065 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4064
  %4066 = addrspacecast ptr %4065 to ptr addrspace(3)
  %4067 = load float, ptr addrspace(3) %4066, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4068 = or disjoint i32 %3998, 7
  %4069 = zext nneg i32 %4068 to i64
  %4070 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4069
  %4071 = addrspacecast ptr %4070 to ptr addrspace(3)
  %4072 = load float, ptr addrspace(3) %4071, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4073 = or disjoint i32 %3999, 7
  %4074 = zext nneg i32 %4073 to i64
  %4075 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4074
  %4076 = addrspacecast ptr %4075 to ptr addrspace(3)
  %4077 = load float, ptr addrspace(3) %4076, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4078 = fmul contract float %4003, 0xBFF7154760000000
  %4079 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4078)
  %4080 = fadd contract float %4079, 1.000000e+00
  %4081 = tail call contract float @llvm.amdgcn.rcp.f32(float %4080)
  %4082 = fmul contract float %4003, %4081
  %4083 = fmul contract float %4007, %4082
  %4084 = fmul contract float %4012, 0xBFF7154760000000
  %4085 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4084)
  %4086 = fadd contract float %4085, 1.000000e+00
  %4087 = tail call contract float @llvm.amdgcn.rcp.f32(float %4086)
  %4088 = fmul contract float %4012, %4087
  %4089 = fmul contract float %4017, %4088
  %4090 = fmul contract float %4022, 0xBFF7154760000000
  %4091 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4090)
  %4092 = fadd contract float %4091, 1.000000e+00
  %4093 = tail call contract float @llvm.amdgcn.rcp.f32(float %4092)
  %4094 = fmul contract float %4022, %4093
  %4095 = fmul contract float %4027, %4094
  %4096 = fmul contract float %4032, 0xBFF7154760000000
  %4097 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4096)
  %4098 = fadd contract float %4097, 1.000000e+00
  %4099 = tail call contract float @llvm.amdgcn.rcp.f32(float %4098)
  %4100 = fmul contract float %4032, %4099
  %4101 = fmul contract float %4037, %4100
  %4102 = fmul contract float %4042, 0xBFF7154760000000
  %4103 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4102)
  %4104 = fadd contract float %4103, 1.000000e+00
  %4105 = tail call contract float @llvm.amdgcn.rcp.f32(float %4104)
  %4106 = fmul contract float %4042, %4105
  %4107 = fmul contract float %4047, %4106
  %4108 = fmul contract float %4052, 0xBFF7154760000000
  %4109 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4108)
  %4110 = fadd contract float %4109, 1.000000e+00
  %4111 = tail call contract float @llvm.amdgcn.rcp.f32(float %4110)
  %4112 = fmul contract float %4052, %4111
  %4113 = fmul contract float %4057, %4112
  %4114 = fmul contract float %4062, 0xBFF7154760000000
  %4115 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4114)
  %4116 = fadd contract float %4115, 1.000000e+00
  %4117 = tail call contract float @llvm.amdgcn.rcp.f32(float %4116)
  %4118 = fmul contract float %4062, %4117
  %4119 = fmul contract float %4067, %4118
  %4120 = fmul contract float %4072, 0xBFF7154760000000
  %4121 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4120)
  %4122 = fadd contract float %4121, 1.000000e+00
  %4123 = tail call contract float @llvm.amdgcn.rcp.f32(float %4122)
  %4124 = fmul contract float %4072, %4123
  %4125 = fmul contract float %4077, %4124
  %4126 = tail call contract noundef float @llvm.fabs.f32(float %4083)
  %4127 = tail call contract noundef float @llvm.fabs.f32(float %4089)
  %4128 = tail call contract noundef float @llvm.maxnum.f32(float %4126, float %4127)
  %4129 = tail call contract noundef float @llvm.fabs.f32(float %4095)
  %4130 = tail call contract noundef float @llvm.maxnum.f32(float %4128, float %4129)
  %4131 = tail call contract noundef float @llvm.fabs.f32(float %4101)
  %4132 = tail call contract noundef float @llvm.maxnum.f32(float %4130, float %4131)
  %4133 = tail call contract noundef float @llvm.fabs.f32(float %4107)
  %4134 = tail call contract noundef float @llvm.maxnum.f32(float %4132, float %4133)
  %4135 = tail call contract noundef float @llvm.fabs.f32(float %4113)
  %4136 = tail call contract noundef float @llvm.maxnum.f32(float %4134, float %4135)
  %4137 = tail call contract noundef float @llvm.fabs.f32(float %4119)
  %4138 = tail call contract noundef float @llvm.maxnum.f32(float %4136, float %4137)
  %4139 = tail call contract noundef float @llvm.fabs.f32(float %4125)
  %4140 = tail call contract noundef float @llvm.maxnum.f32(float %4138, float %4139)
  %4141 = bitcast float %4140 to i32
  %4142 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %4141, i32 177, i32 15, i32 15, i1 true)
  %4143 = bitcast i32 %4142 to float
  %4144 = tail call contract noundef float @llvm.maxnum.f32(float %4140, float %4143)
  %4145 = bitcast float %4144 to i32
  %4146 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %4145, i32 78, i32 15, i32 15, i1 true)
  %4147 = bitcast i32 %4146 to float
  %4148 = tail call contract noundef float @llvm.maxnum.f32(float %4144, float %4147)
  %4149 = bitcast float %4148 to i32
  %4150 = add i32 %4149, 2097152
  %4151 = bitcast i32 %4150 to float
  %4152 = fmul contract float %4151, 2.500000e-01
  %4153 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 0, float %4083, float %4089, float %4152, i32 0)
  %4154 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %4153, float %4095, float %4101, float %4152, i32 1)
  %4155 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %4154, float %4107, float %4113, float %4152, i32 2)
  %4156 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %4155, float %4119, float %4125, float %4152, i32 3)
  %4157 = add i32 %3994, 4096
  %4158 = sext i32 %4157 to i64
  %4159 = getelementptr inbounds i8, ptr %39, i64 %4158
  store i32 %4156, ptr %4159, align 4, !tbaa !7, !alias.scope !16, !noalias !26, !nontemporal !27
  %4160 = or disjoint i32 %3831, 8192
  %4161 = or disjoint i32 %3825, %4160
  %4162 = or disjoint i32 %3826, %4160
  %4163 = zext nneg i32 %4161 to i64
  %4164 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4163
  %4165 = addrspacecast ptr %4164 to ptr addrspace(3)
  %4166 = load float, ptr addrspace(3) %4165, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4167 = zext nneg i32 %4162 to i64
  %4168 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4167
  %4169 = addrspacecast ptr %4168 to ptr addrspace(3)
  %4170 = load float, ptr addrspace(3) %4169, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4171 = or disjoint i32 %4161, 1
  %4172 = zext nneg i32 %4171 to i64
  %4173 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4172
  %4174 = addrspacecast ptr %4173 to ptr addrspace(3)
  %4175 = load float, ptr addrspace(3) %4174, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4176 = or disjoint i32 %4162, 1
  %4177 = zext nneg i32 %4176 to i64
  %4178 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4177
  %4179 = addrspacecast ptr %4178 to ptr addrspace(3)
  %4180 = load float, ptr addrspace(3) %4179, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4181 = or disjoint i32 %4161, 2
  %4182 = zext nneg i32 %4181 to i64
  %4183 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4182
  %4184 = addrspacecast ptr %4183 to ptr addrspace(3)
  %4185 = load float, ptr addrspace(3) %4184, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4186 = or disjoint i32 %4162, 2
  %4187 = zext nneg i32 %4186 to i64
  %4188 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4187
  %4189 = addrspacecast ptr %4188 to ptr addrspace(3)
  %4190 = load float, ptr addrspace(3) %4189, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4191 = or disjoint i32 %4161, 3
  %4192 = zext nneg i32 %4191 to i64
  %4193 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4192
  %4194 = addrspacecast ptr %4193 to ptr addrspace(3)
  %4195 = load float, ptr addrspace(3) %4194, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4196 = or disjoint i32 %4162, 3
  %4197 = zext nneg i32 %4196 to i64
  %4198 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4197
  %4199 = addrspacecast ptr %4198 to ptr addrspace(3)
  %4200 = load float, ptr addrspace(3) %4199, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4201 = or disjoint i32 %4161, 4
  %4202 = zext nneg i32 %4201 to i64
  %4203 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4202
  %4204 = addrspacecast ptr %4203 to ptr addrspace(3)
  %4205 = load float, ptr addrspace(3) %4204, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4206 = or disjoint i32 %4162, 4
  %4207 = zext nneg i32 %4206 to i64
  %4208 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4207
  %4209 = addrspacecast ptr %4208 to ptr addrspace(3)
  %4210 = load float, ptr addrspace(3) %4209, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4211 = or disjoint i32 %4161, 5
  %4212 = zext nneg i32 %4211 to i64
  %4213 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4212
  %4214 = addrspacecast ptr %4213 to ptr addrspace(3)
  %4215 = load float, ptr addrspace(3) %4214, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4216 = or disjoint i32 %4162, 5
  %4217 = zext nneg i32 %4216 to i64
  %4218 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4217
  %4219 = addrspacecast ptr %4218 to ptr addrspace(3)
  %4220 = load float, ptr addrspace(3) %4219, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4221 = or disjoint i32 %4161, 6
  %4222 = zext nneg i32 %4221 to i64
  %4223 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4222
  %4224 = addrspacecast ptr %4223 to ptr addrspace(3)
  %4225 = load float, ptr addrspace(3) %4224, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4226 = or disjoint i32 %4162, 6
  %4227 = zext nneg i32 %4226 to i64
  %4228 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4227
  %4229 = addrspacecast ptr %4228 to ptr addrspace(3)
  %4230 = load float, ptr addrspace(3) %4229, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4231 = or disjoint i32 %4161, 7
  %4232 = zext nneg i32 %4231 to i64
  %4233 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4232
  %4234 = addrspacecast ptr %4233 to ptr addrspace(3)
  %4235 = load float, ptr addrspace(3) %4234, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4236 = or disjoint i32 %4162, 7
  %4237 = zext nneg i32 %4236 to i64
  %4238 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4237
  %4239 = addrspacecast ptr %4238 to ptr addrspace(3)
  %4240 = load float, ptr addrspace(3) %4239, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4241 = fmul contract float %4166, 0xBFF7154760000000
  %4242 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4241)
  %4243 = fadd contract float %4242, 1.000000e+00
  %4244 = tail call contract float @llvm.amdgcn.rcp.f32(float %4243)
  %4245 = fmul contract float %4166, %4244
  %4246 = fmul contract float %4170, %4245
  %4247 = fmul contract float %4175, 0xBFF7154760000000
  %4248 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4247)
  %4249 = fadd contract float %4248, 1.000000e+00
  %4250 = tail call contract float @llvm.amdgcn.rcp.f32(float %4249)
  %4251 = fmul contract float %4175, %4250
  %4252 = fmul contract float %4180, %4251
  %4253 = fmul contract float %4185, 0xBFF7154760000000
  %4254 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4253)
  %4255 = fadd contract float %4254, 1.000000e+00
  %4256 = tail call contract float @llvm.amdgcn.rcp.f32(float %4255)
  %4257 = fmul contract float %4185, %4256
  %4258 = fmul contract float %4190, %4257
  %4259 = fmul contract float %4195, 0xBFF7154760000000
  %4260 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4259)
  %4261 = fadd contract float %4260, 1.000000e+00
  %4262 = tail call contract float @llvm.amdgcn.rcp.f32(float %4261)
  %4263 = fmul contract float %4195, %4262
  %4264 = fmul contract float %4200, %4263
  %4265 = fmul contract float %4205, 0xBFF7154760000000
  %4266 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4265)
  %4267 = fadd contract float %4266, 1.000000e+00
  %4268 = tail call contract float @llvm.amdgcn.rcp.f32(float %4267)
  %4269 = fmul contract float %4205, %4268
  %4270 = fmul contract float %4210, %4269
  %4271 = fmul contract float %4215, 0xBFF7154760000000
  %4272 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4271)
  %4273 = fadd contract float %4272, 1.000000e+00
  %4274 = tail call contract float @llvm.amdgcn.rcp.f32(float %4273)
  %4275 = fmul contract float %4215, %4274
  %4276 = fmul contract float %4220, %4275
  %4277 = fmul contract float %4225, 0xBFF7154760000000
  %4278 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4277)
  %4279 = fadd contract float %4278, 1.000000e+00
  %4280 = tail call contract float @llvm.amdgcn.rcp.f32(float %4279)
  %4281 = fmul contract float %4225, %4280
  %4282 = fmul contract float %4230, %4281
  %4283 = fmul contract float %4235, 0xBFF7154760000000
  %4284 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4283)
  %4285 = fadd contract float %4284, 1.000000e+00
  %4286 = tail call contract float @llvm.amdgcn.rcp.f32(float %4285)
  %4287 = fmul contract float %4235, %4286
  %4288 = fmul contract float %4240, %4287
  %4289 = tail call contract noundef float @llvm.fabs.f32(float %4246)
  %4290 = tail call contract noundef float @llvm.fabs.f32(float %4252)
  %4291 = tail call contract noundef float @llvm.maxnum.f32(float %4289, float %4290)
  %4292 = tail call contract noundef float @llvm.fabs.f32(float %4258)
  %4293 = tail call contract noundef float @llvm.maxnum.f32(float %4291, float %4292)
  %4294 = tail call contract noundef float @llvm.fabs.f32(float %4264)
  %4295 = tail call contract noundef float @llvm.maxnum.f32(float %4293, float %4294)
  %4296 = tail call contract noundef float @llvm.fabs.f32(float %4270)
  %4297 = tail call contract noundef float @llvm.maxnum.f32(float %4295, float %4296)
  %4298 = tail call contract noundef float @llvm.fabs.f32(float %4276)
  %4299 = tail call contract noundef float @llvm.maxnum.f32(float %4297, float %4298)
  %4300 = tail call contract noundef float @llvm.fabs.f32(float %4282)
  %4301 = tail call contract noundef float @llvm.maxnum.f32(float %4299, float %4300)
  %4302 = tail call contract noundef float @llvm.fabs.f32(float %4288)
  %4303 = tail call contract noundef float @llvm.maxnum.f32(float %4301, float %4302)
  %4304 = bitcast float %4303 to i32
  %4305 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %4304, i32 177, i32 15, i32 15, i1 true)
  %4306 = bitcast i32 %4305 to float
  %4307 = tail call contract noundef float @llvm.maxnum.f32(float %4303, float %4306)
  %4308 = bitcast float %4307 to i32
  %4309 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %4308, i32 78, i32 15, i32 15, i1 true)
  %4310 = bitcast i32 %4309 to float
  %4311 = tail call contract noundef float @llvm.maxnum.f32(float %4307, float %4310)
  %4312 = bitcast float %4311 to i32
  %4313 = add i32 %4312, 2097152
  %4314 = bitcast i32 %4313 to float
  %4315 = fmul contract float %4314, 2.500000e-01
  %4316 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 0, float %4246, float %4252, float %4315, i32 0)
  %4317 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %4316, float %4258, float %4264, float %4315, i32 1)
  %4318 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %4317, float %4270, float %4276, float %4315, i32 2)
  %4319 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %4318, float %4282, float %4288, float %4315, i32 3)
  %4320 = add i32 %3994, 8192
  %4321 = sext i32 %4320 to i64
  %4322 = getelementptr inbounds i8, ptr %39, i64 %4321
  store i32 %4319, ptr %4322, align 4, !tbaa !7, !alias.scope !16, !noalias !26, !nontemporal !27
  %4323 = or disjoint i32 %3831, 12288
  %4324 = or disjoint i32 %3825, %4323
  %4325 = or disjoint i32 %3826, %4323
  %4326 = zext nneg i32 %4324 to i64
  %4327 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4326
  %4328 = addrspacecast ptr %4327 to ptr addrspace(3)
  %4329 = load float, ptr addrspace(3) %4328, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4330 = zext nneg i32 %4325 to i64
  %4331 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4330
  %4332 = addrspacecast ptr %4331 to ptr addrspace(3)
  %4333 = load float, ptr addrspace(3) %4332, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4334 = or disjoint i32 %4324, 1
  %4335 = zext nneg i32 %4334 to i64
  %4336 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4335
  %4337 = addrspacecast ptr %4336 to ptr addrspace(3)
  %4338 = load float, ptr addrspace(3) %4337, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4339 = or disjoint i32 %4325, 1
  %4340 = zext nneg i32 %4339 to i64
  %4341 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4340
  %4342 = addrspacecast ptr %4341 to ptr addrspace(3)
  %4343 = load float, ptr addrspace(3) %4342, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4344 = or disjoint i32 %4324, 2
  %4345 = zext nneg i32 %4344 to i64
  %4346 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4345
  %4347 = addrspacecast ptr %4346 to ptr addrspace(3)
  %4348 = load float, ptr addrspace(3) %4347, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4349 = or disjoint i32 %4325, 2
  %4350 = zext nneg i32 %4349 to i64
  %4351 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4350
  %4352 = addrspacecast ptr %4351 to ptr addrspace(3)
  %4353 = load float, ptr addrspace(3) %4352, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4354 = or disjoint i32 %4324, 3
  %4355 = zext nneg i32 %4354 to i64
  %4356 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4355
  %4357 = addrspacecast ptr %4356 to ptr addrspace(3)
  %4358 = load float, ptr addrspace(3) %4357, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4359 = or disjoint i32 %4325, 3
  %4360 = zext nneg i32 %4359 to i64
  %4361 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4360
  %4362 = addrspacecast ptr %4361 to ptr addrspace(3)
  %4363 = load float, ptr addrspace(3) %4362, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4364 = or disjoint i32 %4324, 4
  %4365 = zext nneg i32 %4364 to i64
  %4366 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4365
  %4367 = addrspacecast ptr %4366 to ptr addrspace(3)
  %4368 = load float, ptr addrspace(3) %4367, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4369 = or disjoint i32 %4325, 4
  %4370 = zext nneg i32 %4369 to i64
  %4371 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4370
  %4372 = addrspacecast ptr %4371 to ptr addrspace(3)
  %4373 = load float, ptr addrspace(3) %4372, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4374 = or disjoint i32 %4324, 5
  %4375 = zext nneg i32 %4374 to i64
  %4376 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4375
  %4377 = addrspacecast ptr %4376 to ptr addrspace(3)
  %4378 = load float, ptr addrspace(3) %4377, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4379 = or disjoint i32 %4325, 5
  %4380 = zext nneg i32 %4379 to i64
  %4381 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4380
  %4382 = addrspacecast ptr %4381 to ptr addrspace(3)
  %4383 = load float, ptr addrspace(3) %4382, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4384 = or disjoint i32 %4324, 6
  %4385 = zext nneg i32 %4384 to i64
  %4386 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4385
  %4387 = addrspacecast ptr %4386 to ptr addrspace(3)
  %4388 = load float, ptr addrspace(3) %4387, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4389 = or disjoint i32 %4325, 6
  %4390 = zext nneg i32 %4389 to i64
  %4391 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4390
  %4392 = addrspacecast ptr %4391 to ptr addrspace(3)
  %4393 = load float, ptr addrspace(3) %4392, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4394 = or disjoint i32 %4324, 7
  %4395 = zext nneg i32 %4394 to i64
  %4396 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4395
  %4397 = addrspacecast ptr %4396 to ptr addrspace(3)
  %4398 = load float, ptr addrspace(3) %4397, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4399 = or disjoint i32 %4325, 7
  %4400 = zext nneg i32 %4399 to i64
  %4401 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4400
  %4402 = addrspacecast ptr %4401 to ptr addrspace(3)
  %4403 = load float, ptr addrspace(3) %4402, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4404 = fmul contract float %4329, 0xBFF7154760000000
  %4405 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4404)
  %4406 = fadd contract float %4405, 1.000000e+00
  %4407 = tail call contract float @llvm.amdgcn.rcp.f32(float %4406)
  %4408 = fmul contract float %4329, %4407
  %4409 = fmul contract float %4333, %4408
  %4410 = fmul contract float %4338, 0xBFF7154760000000
  %4411 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4410)
  %4412 = fadd contract float %4411, 1.000000e+00
  %4413 = tail call contract float @llvm.amdgcn.rcp.f32(float %4412)
  %4414 = fmul contract float %4338, %4413
  %4415 = fmul contract float %4343, %4414
  %4416 = fmul contract float %4348, 0xBFF7154760000000
  %4417 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4416)
  %4418 = fadd contract float %4417, 1.000000e+00
  %4419 = tail call contract float @llvm.amdgcn.rcp.f32(float %4418)
  %4420 = fmul contract float %4348, %4419
  %4421 = fmul contract float %4353, %4420
  %4422 = fmul contract float %4358, 0xBFF7154760000000
  %4423 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4422)
  %4424 = fadd contract float %4423, 1.000000e+00
  %4425 = tail call contract float @llvm.amdgcn.rcp.f32(float %4424)
  %4426 = fmul contract float %4358, %4425
  %4427 = fmul contract float %4363, %4426
  %4428 = fmul contract float %4368, 0xBFF7154760000000
  %4429 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4428)
  %4430 = fadd contract float %4429, 1.000000e+00
  %4431 = tail call contract float @llvm.amdgcn.rcp.f32(float %4430)
  %4432 = fmul contract float %4368, %4431
  %4433 = fmul contract float %4373, %4432
  %4434 = fmul contract float %4378, 0xBFF7154760000000
  %4435 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4434)
  %4436 = fadd contract float %4435, 1.000000e+00
  %4437 = tail call contract float @llvm.amdgcn.rcp.f32(float %4436)
  %4438 = fmul contract float %4378, %4437
  %4439 = fmul contract float %4383, %4438
  %4440 = fmul contract float %4388, 0xBFF7154760000000
  %4441 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4440)
  %4442 = fadd contract float %4441, 1.000000e+00
  %4443 = tail call contract float @llvm.amdgcn.rcp.f32(float %4442)
  %4444 = fmul contract float %4388, %4443
  %4445 = fmul contract float %4393, %4444
  %4446 = fmul contract float %4398, 0xBFF7154760000000
  %4447 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4446)
  %4448 = fadd contract float %4447, 1.000000e+00
  %4449 = tail call contract float @llvm.amdgcn.rcp.f32(float %4448)
  %4450 = fmul contract float %4398, %4449
  %4451 = fmul contract float %4403, %4450
  %4452 = tail call contract noundef float @llvm.fabs.f32(float %4409)
  %4453 = tail call contract noundef float @llvm.fabs.f32(float %4415)
  %4454 = tail call contract noundef float @llvm.maxnum.f32(float %4452, float %4453)
  %4455 = tail call contract noundef float @llvm.fabs.f32(float %4421)
  %4456 = tail call contract noundef float @llvm.maxnum.f32(float %4454, float %4455)
  %4457 = tail call contract noundef float @llvm.fabs.f32(float %4427)
  %4458 = tail call contract noundef float @llvm.maxnum.f32(float %4456, float %4457)
  %4459 = tail call contract noundef float @llvm.fabs.f32(float %4433)
  %4460 = tail call contract noundef float @llvm.maxnum.f32(float %4458, float %4459)
  %4461 = tail call contract noundef float @llvm.fabs.f32(float %4439)
  %4462 = tail call contract noundef float @llvm.maxnum.f32(float %4460, float %4461)
  %4463 = tail call contract noundef float @llvm.fabs.f32(float %4445)
  %4464 = tail call contract noundef float @llvm.maxnum.f32(float %4462, float %4463)
  %4465 = tail call contract noundef float @llvm.fabs.f32(float %4451)
  %4466 = tail call contract noundef float @llvm.maxnum.f32(float %4464, float %4465)
  %4467 = bitcast float %4466 to i32
  %4468 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %4467, i32 177, i32 15, i32 15, i1 true)
  %4469 = bitcast i32 %4468 to float
  %4470 = tail call contract noundef float @llvm.maxnum.f32(float %4466, float %4469)
  %4471 = bitcast float %4470 to i32
  %4472 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %4471, i32 78, i32 15, i32 15, i1 true)
  %4473 = bitcast i32 %4472 to float
  %4474 = tail call contract noundef float @llvm.maxnum.f32(float %4470, float %4473)
  %4475 = bitcast float %4474 to i32
  %4476 = add i32 %4475, 2097152
  %4477 = bitcast i32 %4476 to float
  %4478 = fmul contract float %4477, 2.500000e-01
  %4479 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 0, float %4409, float %4415, float %4478, i32 0)
  %4480 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %4479, float %4421, float %4427, float %4478, i32 1)
  %4481 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %4480, float %4433, float %4439, float %4478, i32 2)
  %4482 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %4481, float %4445, float %4451, float %4478, i32 3)
  %4483 = add i32 %3994, 12288
  %4484 = sext i32 %4483 to i64
  %4485 = getelementptr inbounds i8, ptr %39, i64 %4484
  store i32 %4482, ptr %4485, align 4, !tbaa !7, !alias.scope !16, !noalias !26, !nontemporal !27
  %4486 = or disjoint i32 %3831, 16384
  %4487 = or disjoint i32 %3825, %4486
  %4488 = or disjoint i32 %3826, %4486
  %4489 = zext nneg i32 %4487 to i64
  %4490 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4489
  %4491 = addrspacecast ptr %4490 to ptr addrspace(3)
  %4492 = load float, ptr addrspace(3) %4491, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4493 = zext nneg i32 %4488 to i64
  %4494 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4493
  %4495 = addrspacecast ptr %4494 to ptr addrspace(3)
  %4496 = load float, ptr addrspace(3) %4495, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4497 = or disjoint i32 %4487, 1
  %4498 = zext nneg i32 %4497 to i64
  %4499 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4498
  %4500 = addrspacecast ptr %4499 to ptr addrspace(3)
  %4501 = load float, ptr addrspace(3) %4500, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4502 = or disjoint i32 %4488, 1
  %4503 = zext nneg i32 %4502 to i64
  %4504 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4503
  %4505 = addrspacecast ptr %4504 to ptr addrspace(3)
  %4506 = load float, ptr addrspace(3) %4505, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4507 = or disjoint i32 %4487, 2
  %4508 = zext nneg i32 %4507 to i64
  %4509 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4508
  %4510 = addrspacecast ptr %4509 to ptr addrspace(3)
  %4511 = load float, ptr addrspace(3) %4510, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4512 = or disjoint i32 %4488, 2
  %4513 = zext nneg i32 %4512 to i64
  %4514 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4513
  %4515 = addrspacecast ptr %4514 to ptr addrspace(3)
  %4516 = load float, ptr addrspace(3) %4515, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4517 = or disjoint i32 %4487, 3
  %4518 = zext nneg i32 %4517 to i64
  %4519 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4518
  %4520 = addrspacecast ptr %4519 to ptr addrspace(3)
  %4521 = load float, ptr addrspace(3) %4520, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4522 = or disjoint i32 %4488, 3
  %4523 = zext nneg i32 %4522 to i64
  %4524 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4523
  %4525 = addrspacecast ptr %4524 to ptr addrspace(3)
  %4526 = load float, ptr addrspace(3) %4525, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4527 = or disjoint i32 %4487, 4
  %4528 = zext nneg i32 %4527 to i64
  %4529 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4528
  %4530 = addrspacecast ptr %4529 to ptr addrspace(3)
  %4531 = load float, ptr addrspace(3) %4530, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4532 = or disjoint i32 %4488, 4
  %4533 = zext nneg i32 %4532 to i64
  %4534 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4533
  %4535 = addrspacecast ptr %4534 to ptr addrspace(3)
  %4536 = load float, ptr addrspace(3) %4535, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4537 = or disjoint i32 %4487, 5
  %4538 = zext nneg i32 %4537 to i64
  %4539 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4538
  %4540 = addrspacecast ptr %4539 to ptr addrspace(3)
  %4541 = load float, ptr addrspace(3) %4540, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4542 = or disjoint i32 %4488, 5
  %4543 = zext nneg i32 %4542 to i64
  %4544 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4543
  %4545 = addrspacecast ptr %4544 to ptr addrspace(3)
  %4546 = load float, ptr addrspace(3) %4545, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4547 = or disjoint i32 %4487, 6
  %4548 = zext nneg i32 %4547 to i64
  %4549 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4548
  %4550 = addrspacecast ptr %4549 to ptr addrspace(3)
  %4551 = load float, ptr addrspace(3) %4550, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4552 = or disjoint i32 %4488, 6
  %4553 = zext nneg i32 %4552 to i64
  %4554 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4553
  %4555 = addrspacecast ptr %4554 to ptr addrspace(3)
  %4556 = load float, ptr addrspace(3) %4555, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4557 = or disjoint i32 %4487, 7
  %4558 = zext nneg i32 %4557 to i64
  %4559 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4558
  %4560 = addrspacecast ptr %4559 to ptr addrspace(3)
  %4561 = load float, ptr addrspace(3) %4560, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4562 = or disjoint i32 %4488, 7
  %4563 = zext nneg i32 %4562 to i64
  %4564 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4563
  %4565 = addrspacecast ptr %4564 to ptr addrspace(3)
  %4566 = load float, ptr addrspace(3) %4565, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4567 = fmul contract float %4492, 0xBFF7154760000000
  %4568 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4567)
  %4569 = fadd contract float %4568, 1.000000e+00
  %4570 = tail call contract float @llvm.amdgcn.rcp.f32(float %4569)
  %4571 = fmul contract float %4492, %4570
  %4572 = fmul contract float %4496, %4571
  %4573 = fmul contract float %4501, 0xBFF7154760000000
  %4574 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4573)
  %4575 = fadd contract float %4574, 1.000000e+00
  %4576 = tail call contract float @llvm.amdgcn.rcp.f32(float %4575)
  %4577 = fmul contract float %4501, %4576
  %4578 = fmul contract float %4506, %4577
  %4579 = fmul contract float %4511, 0xBFF7154760000000
  %4580 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4579)
  %4581 = fadd contract float %4580, 1.000000e+00
  %4582 = tail call contract float @llvm.amdgcn.rcp.f32(float %4581)
  %4583 = fmul contract float %4511, %4582
  %4584 = fmul contract float %4516, %4583
  %4585 = fmul contract float %4521, 0xBFF7154760000000
  %4586 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4585)
  %4587 = fadd contract float %4586, 1.000000e+00
  %4588 = tail call contract float @llvm.amdgcn.rcp.f32(float %4587)
  %4589 = fmul contract float %4521, %4588
  %4590 = fmul contract float %4526, %4589
  %4591 = fmul contract float %4531, 0xBFF7154760000000
  %4592 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4591)
  %4593 = fadd contract float %4592, 1.000000e+00
  %4594 = tail call contract float @llvm.amdgcn.rcp.f32(float %4593)
  %4595 = fmul contract float %4531, %4594
  %4596 = fmul contract float %4536, %4595
  %4597 = fmul contract float %4541, 0xBFF7154760000000
  %4598 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4597)
  %4599 = fadd contract float %4598, 1.000000e+00
  %4600 = tail call contract float @llvm.amdgcn.rcp.f32(float %4599)
  %4601 = fmul contract float %4541, %4600
  %4602 = fmul contract float %4546, %4601
  %4603 = fmul contract float %4551, 0xBFF7154760000000
  %4604 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4603)
  %4605 = fadd contract float %4604, 1.000000e+00
  %4606 = tail call contract float @llvm.amdgcn.rcp.f32(float %4605)
  %4607 = fmul contract float %4551, %4606
  %4608 = fmul contract float %4556, %4607
  %4609 = fmul contract float %4561, 0xBFF7154760000000
  %4610 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4609)
  %4611 = fadd contract float %4610, 1.000000e+00
  %4612 = tail call contract float @llvm.amdgcn.rcp.f32(float %4611)
  %4613 = fmul contract float %4561, %4612
  %4614 = fmul contract float %4566, %4613
  %4615 = tail call contract noundef float @llvm.fabs.f32(float %4572)
  %4616 = tail call contract noundef float @llvm.fabs.f32(float %4578)
  %4617 = tail call contract noundef float @llvm.maxnum.f32(float %4615, float %4616)
  %4618 = tail call contract noundef float @llvm.fabs.f32(float %4584)
  %4619 = tail call contract noundef float @llvm.maxnum.f32(float %4617, float %4618)
  %4620 = tail call contract noundef float @llvm.fabs.f32(float %4590)
  %4621 = tail call contract noundef float @llvm.maxnum.f32(float %4619, float %4620)
  %4622 = tail call contract noundef float @llvm.fabs.f32(float %4596)
  %4623 = tail call contract noundef float @llvm.maxnum.f32(float %4621, float %4622)
  %4624 = tail call contract noundef float @llvm.fabs.f32(float %4602)
  %4625 = tail call contract noundef float @llvm.maxnum.f32(float %4623, float %4624)
  %4626 = tail call contract noundef float @llvm.fabs.f32(float %4608)
  %4627 = tail call contract noundef float @llvm.maxnum.f32(float %4625, float %4626)
  %4628 = tail call contract noundef float @llvm.fabs.f32(float %4614)
  %4629 = tail call contract noundef float @llvm.maxnum.f32(float %4627, float %4628)
  %4630 = bitcast float %4629 to i32
  %4631 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %4630, i32 177, i32 15, i32 15, i1 true)
  %4632 = bitcast i32 %4631 to float
  %4633 = tail call contract noundef float @llvm.maxnum.f32(float %4629, float %4632)
  %4634 = bitcast float %4633 to i32
  %4635 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %4634, i32 78, i32 15, i32 15, i1 true)
  %4636 = bitcast i32 %4635 to float
  %4637 = tail call contract noundef float @llvm.maxnum.f32(float %4633, float %4636)
  %4638 = bitcast float %4637 to i32
  %4639 = add i32 %4638, 2097152
  %4640 = bitcast i32 %4639 to float
  %4641 = fmul contract float %4640, 2.500000e-01
  %4642 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 0, float %4572, float %4578, float %4641, i32 0)
  %4643 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %4642, float %4584, float %4590, float %4641, i32 1)
  %4644 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %4643, float %4596, float %4602, float %4641, i32 2)
  %4645 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %4644, float %4608, float %4614, float %4641, i32 3)
  %4646 = add i32 %3994, 16384
  %4647 = sext i32 %4646 to i64
  %4648 = getelementptr inbounds i8, ptr %39, i64 %4647
  store i32 %4645, ptr %4648, align 4, !tbaa !7, !alias.scope !16, !noalias !26, !nontemporal !27
  %4649 = or disjoint i32 %3831, 20480
  %4650 = or disjoint i32 %3825, %4649
  %4651 = or disjoint i32 %3826, %4649
  %4652 = zext nneg i32 %4650 to i64
  %4653 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4652
  %4654 = addrspacecast ptr %4653 to ptr addrspace(3)
  %4655 = load float, ptr addrspace(3) %4654, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4656 = zext nneg i32 %4651 to i64
  %4657 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4656
  %4658 = addrspacecast ptr %4657 to ptr addrspace(3)
  %4659 = load float, ptr addrspace(3) %4658, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4660 = or disjoint i32 %4650, 1
  %4661 = zext nneg i32 %4660 to i64
  %4662 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4661
  %4663 = addrspacecast ptr %4662 to ptr addrspace(3)
  %4664 = load float, ptr addrspace(3) %4663, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4665 = or disjoint i32 %4651, 1
  %4666 = zext nneg i32 %4665 to i64
  %4667 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4666
  %4668 = addrspacecast ptr %4667 to ptr addrspace(3)
  %4669 = load float, ptr addrspace(3) %4668, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4670 = or disjoint i32 %4650, 2
  %4671 = zext nneg i32 %4670 to i64
  %4672 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4671
  %4673 = addrspacecast ptr %4672 to ptr addrspace(3)
  %4674 = load float, ptr addrspace(3) %4673, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4675 = or disjoint i32 %4651, 2
  %4676 = zext nneg i32 %4675 to i64
  %4677 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4676
  %4678 = addrspacecast ptr %4677 to ptr addrspace(3)
  %4679 = load float, ptr addrspace(3) %4678, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4680 = or disjoint i32 %4650, 3
  %4681 = zext nneg i32 %4680 to i64
  %4682 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4681
  %4683 = addrspacecast ptr %4682 to ptr addrspace(3)
  %4684 = load float, ptr addrspace(3) %4683, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4685 = or disjoint i32 %4651, 3
  %4686 = zext nneg i32 %4685 to i64
  %4687 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4686
  %4688 = addrspacecast ptr %4687 to ptr addrspace(3)
  %4689 = load float, ptr addrspace(3) %4688, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4690 = or disjoint i32 %4650, 4
  %4691 = zext nneg i32 %4690 to i64
  %4692 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4691
  %4693 = addrspacecast ptr %4692 to ptr addrspace(3)
  %4694 = load float, ptr addrspace(3) %4693, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4695 = or disjoint i32 %4651, 4
  %4696 = zext nneg i32 %4695 to i64
  %4697 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4696
  %4698 = addrspacecast ptr %4697 to ptr addrspace(3)
  %4699 = load float, ptr addrspace(3) %4698, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4700 = or disjoint i32 %4650, 5
  %4701 = zext nneg i32 %4700 to i64
  %4702 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4701
  %4703 = addrspacecast ptr %4702 to ptr addrspace(3)
  %4704 = load float, ptr addrspace(3) %4703, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4705 = or disjoint i32 %4651, 5
  %4706 = zext nneg i32 %4705 to i64
  %4707 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4706
  %4708 = addrspacecast ptr %4707 to ptr addrspace(3)
  %4709 = load float, ptr addrspace(3) %4708, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4710 = or disjoint i32 %4650, 6
  %4711 = zext nneg i32 %4710 to i64
  %4712 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4711
  %4713 = addrspacecast ptr %4712 to ptr addrspace(3)
  %4714 = load float, ptr addrspace(3) %4713, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4715 = or disjoint i32 %4651, 6
  %4716 = zext nneg i32 %4715 to i64
  %4717 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4716
  %4718 = addrspacecast ptr %4717 to ptr addrspace(3)
  %4719 = load float, ptr addrspace(3) %4718, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4720 = or disjoint i32 %4650, 7
  %4721 = zext nneg i32 %4720 to i64
  %4722 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4721
  %4723 = addrspacecast ptr %4722 to ptr addrspace(3)
  %4724 = load float, ptr addrspace(3) %4723, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4725 = or disjoint i32 %4651, 7
  %4726 = zext nneg i32 %4725 to i64
  %4727 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4726
  %4728 = addrspacecast ptr %4727 to ptr addrspace(3)
  %4729 = load float, ptr addrspace(3) %4728, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4730 = fmul contract float %4655, 0xBFF7154760000000
  %4731 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4730)
  %4732 = fadd contract float %4731, 1.000000e+00
  %4733 = tail call contract float @llvm.amdgcn.rcp.f32(float %4732)
  %4734 = fmul contract float %4655, %4733
  %4735 = fmul contract float %4659, %4734
  %4736 = fmul contract float %4664, 0xBFF7154760000000
  %4737 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4736)
  %4738 = fadd contract float %4737, 1.000000e+00
  %4739 = tail call contract float @llvm.amdgcn.rcp.f32(float %4738)
  %4740 = fmul contract float %4664, %4739
  %4741 = fmul contract float %4669, %4740
  %4742 = fmul contract float %4674, 0xBFF7154760000000
  %4743 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4742)
  %4744 = fadd contract float %4743, 1.000000e+00
  %4745 = tail call contract float @llvm.amdgcn.rcp.f32(float %4744)
  %4746 = fmul contract float %4674, %4745
  %4747 = fmul contract float %4679, %4746
  %4748 = fmul contract float %4684, 0xBFF7154760000000
  %4749 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4748)
  %4750 = fadd contract float %4749, 1.000000e+00
  %4751 = tail call contract float @llvm.amdgcn.rcp.f32(float %4750)
  %4752 = fmul contract float %4684, %4751
  %4753 = fmul contract float %4689, %4752
  %4754 = fmul contract float %4694, 0xBFF7154760000000
  %4755 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4754)
  %4756 = fadd contract float %4755, 1.000000e+00
  %4757 = tail call contract float @llvm.amdgcn.rcp.f32(float %4756)
  %4758 = fmul contract float %4694, %4757
  %4759 = fmul contract float %4699, %4758
  %4760 = fmul contract float %4704, 0xBFF7154760000000
  %4761 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4760)
  %4762 = fadd contract float %4761, 1.000000e+00
  %4763 = tail call contract float @llvm.amdgcn.rcp.f32(float %4762)
  %4764 = fmul contract float %4704, %4763
  %4765 = fmul contract float %4709, %4764
  %4766 = fmul contract float %4714, 0xBFF7154760000000
  %4767 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4766)
  %4768 = fadd contract float %4767, 1.000000e+00
  %4769 = tail call contract float @llvm.amdgcn.rcp.f32(float %4768)
  %4770 = fmul contract float %4714, %4769
  %4771 = fmul contract float %4719, %4770
  %4772 = fmul contract float %4724, 0xBFF7154760000000
  %4773 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4772)
  %4774 = fadd contract float %4773, 1.000000e+00
  %4775 = tail call contract float @llvm.amdgcn.rcp.f32(float %4774)
  %4776 = fmul contract float %4724, %4775
  %4777 = fmul contract float %4729, %4776
  %4778 = tail call contract noundef float @llvm.fabs.f32(float %4735)
  %4779 = tail call contract noundef float @llvm.fabs.f32(float %4741)
  %4780 = tail call contract noundef float @llvm.maxnum.f32(float %4778, float %4779)
  %4781 = tail call contract noundef float @llvm.fabs.f32(float %4747)
  %4782 = tail call contract noundef float @llvm.maxnum.f32(float %4780, float %4781)
  %4783 = tail call contract noundef float @llvm.fabs.f32(float %4753)
  %4784 = tail call contract noundef float @llvm.maxnum.f32(float %4782, float %4783)
  %4785 = tail call contract noundef float @llvm.fabs.f32(float %4759)
  %4786 = tail call contract noundef float @llvm.maxnum.f32(float %4784, float %4785)
  %4787 = tail call contract noundef float @llvm.fabs.f32(float %4765)
  %4788 = tail call contract noundef float @llvm.maxnum.f32(float %4786, float %4787)
  %4789 = tail call contract noundef float @llvm.fabs.f32(float %4771)
  %4790 = tail call contract noundef float @llvm.maxnum.f32(float %4788, float %4789)
  %4791 = tail call contract noundef float @llvm.fabs.f32(float %4777)
  %4792 = tail call contract noundef float @llvm.maxnum.f32(float %4790, float %4791)
  %4793 = bitcast float %4792 to i32
  %4794 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %4793, i32 177, i32 15, i32 15, i1 true)
  %4795 = bitcast i32 %4794 to float
  %4796 = tail call contract noundef float @llvm.maxnum.f32(float %4792, float %4795)
  %4797 = bitcast float %4796 to i32
  %4798 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %4797, i32 78, i32 15, i32 15, i1 true)
  %4799 = bitcast i32 %4798 to float
  %4800 = tail call contract noundef float @llvm.maxnum.f32(float %4796, float %4799)
  %4801 = bitcast float %4800 to i32
  %4802 = add i32 %4801, 2097152
  %4803 = bitcast i32 %4802 to float
  %4804 = fmul contract float %4803, 2.500000e-01
  %4805 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 0, float %4735, float %4741, float %4804, i32 0)
  %4806 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %4805, float %4747, float %4753, float %4804, i32 1)
  %4807 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %4806, float %4759, float %4765, float %4804, i32 2)
  %4808 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %4807, float %4771, float %4777, float %4804, i32 3)
  %4809 = add i32 %3994, 20480
  %4810 = sext i32 %4809 to i64
  %4811 = getelementptr inbounds i8, ptr %39, i64 %4810
  store i32 %4808, ptr %4811, align 4, !tbaa !7, !alias.scope !16, !noalias !26, !nontemporal !27
  %4812 = or disjoint i32 %3831, 24576
  %4813 = or disjoint i32 %3825, %4812
  %4814 = or disjoint i32 %3826, %4812
  %4815 = zext nneg i32 %4813 to i64
  %4816 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4815
  %4817 = addrspacecast ptr %4816 to ptr addrspace(3)
  %4818 = load float, ptr addrspace(3) %4817, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4819 = zext nneg i32 %4814 to i64
  %4820 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4819
  %4821 = addrspacecast ptr %4820 to ptr addrspace(3)
  %4822 = load float, ptr addrspace(3) %4821, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4823 = or disjoint i32 %4813, 1
  %4824 = zext nneg i32 %4823 to i64
  %4825 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4824
  %4826 = addrspacecast ptr %4825 to ptr addrspace(3)
  %4827 = load float, ptr addrspace(3) %4826, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4828 = or disjoint i32 %4814, 1
  %4829 = zext nneg i32 %4828 to i64
  %4830 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4829
  %4831 = addrspacecast ptr %4830 to ptr addrspace(3)
  %4832 = load float, ptr addrspace(3) %4831, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4833 = or disjoint i32 %4813, 2
  %4834 = zext nneg i32 %4833 to i64
  %4835 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4834
  %4836 = addrspacecast ptr %4835 to ptr addrspace(3)
  %4837 = load float, ptr addrspace(3) %4836, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4838 = or disjoint i32 %4814, 2
  %4839 = zext nneg i32 %4838 to i64
  %4840 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4839
  %4841 = addrspacecast ptr %4840 to ptr addrspace(3)
  %4842 = load float, ptr addrspace(3) %4841, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4843 = or disjoint i32 %4813, 3
  %4844 = zext nneg i32 %4843 to i64
  %4845 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4844
  %4846 = addrspacecast ptr %4845 to ptr addrspace(3)
  %4847 = load float, ptr addrspace(3) %4846, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4848 = or disjoint i32 %4814, 3
  %4849 = zext nneg i32 %4848 to i64
  %4850 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4849
  %4851 = addrspacecast ptr %4850 to ptr addrspace(3)
  %4852 = load float, ptr addrspace(3) %4851, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4853 = or disjoint i32 %4813, 4
  %4854 = zext nneg i32 %4853 to i64
  %4855 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4854
  %4856 = addrspacecast ptr %4855 to ptr addrspace(3)
  %4857 = load float, ptr addrspace(3) %4856, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4858 = or disjoint i32 %4814, 4
  %4859 = zext nneg i32 %4858 to i64
  %4860 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4859
  %4861 = addrspacecast ptr %4860 to ptr addrspace(3)
  %4862 = load float, ptr addrspace(3) %4861, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4863 = or disjoint i32 %4813, 5
  %4864 = zext nneg i32 %4863 to i64
  %4865 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4864
  %4866 = addrspacecast ptr %4865 to ptr addrspace(3)
  %4867 = load float, ptr addrspace(3) %4866, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4868 = or disjoint i32 %4814, 5
  %4869 = zext nneg i32 %4868 to i64
  %4870 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4869
  %4871 = addrspacecast ptr %4870 to ptr addrspace(3)
  %4872 = load float, ptr addrspace(3) %4871, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4873 = or disjoint i32 %4813, 6
  %4874 = zext nneg i32 %4873 to i64
  %4875 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4874
  %4876 = addrspacecast ptr %4875 to ptr addrspace(3)
  %4877 = load float, ptr addrspace(3) %4876, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4878 = or disjoint i32 %4814, 6
  %4879 = zext nneg i32 %4878 to i64
  %4880 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4879
  %4881 = addrspacecast ptr %4880 to ptr addrspace(3)
  %4882 = load float, ptr addrspace(3) %4881, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4883 = or disjoint i32 %4813, 7
  %4884 = zext nneg i32 %4883 to i64
  %4885 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4884
  %4886 = addrspacecast ptr %4885 to ptr addrspace(3)
  %4887 = load float, ptr addrspace(3) %4886, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4888 = or disjoint i32 %4814, 7
  %4889 = zext nneg i32 %4888 to i64
  %4890 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4889
  %4891 = addrspacecast ptr %4890 to ptr addrspace(3)
  %4892 = load float, ptr addrspace(3) %4891, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4893 = fmul contract float %4818, 0xBFF7154760000000
  %4894 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4893)
  %4895 = fadd contract float %4894, 1.000000e+00
  %4896 = tail call contract float @llvm.amdgcn.rcp.f32(float %4895)
  %4897 = fmul contract float %4818, %4896
  %4898 = fmul contract float %4822, %4897
  %4899 = fmul contract float %4827, 0xBFF7154760000000
  %4900 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4899)
  %4901 = fadd contract float %4900, 1.000000e+00
  %4902 = tail call contract float @llvm.amdgcn.rcp.f32(float %4901)
  %4903 = fmul contract float %4827, %4902
  %4904 = fmul contract float %4832, %4903
  %4905 = fmul contract float %4837, 0xBFF7154760000000
  %4906 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4905)
  %4907 = fadd contract float %4906, 1.000000e+00
  %4908 = tail call contract float @llvm.amdgcn.rcp.f32(float %4907)
  %4909 = fmul contract float %4837, %4908
  %4910 = fmul contract float %4842, %4909
  %4911 = fmul contract float %4847, 0xBFF7154760000000
  %4912 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4911)
  %4913 = fadd contract float %4912, 1.000000e+00
  %4914 = tail call contract float @llvm.amdgcn.rcp.f32(float %4913)
  %4915 = fmul contract float %4847, %4914
  %4916 = fmul contract float %4852, %4915
  %4917 = fmul contract float %4857, 0xBFF7154760000000
  %4918 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4917)
  %4919 = fadd contract float %4918, 1.000000e+00
  %4920 = tail call contract float @llvm.amdgcn.rcp.f32(float %4919)
  %4921 = fmul contract float %4857, %4920
  %4922 = fmul contract float %4862, %4921
  %4923 = fmul contract float %4867, 0xBFF7154760000000
  %4924 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4923)
  %4925 = fadd contract float %4924, 1.000000e+00
  %4926 = tail call contract float @llvm.amdgcn.rcp.f32(float %4925)
  %4927 = fmul contract float %4867, %4926
  %4928 = fmul contract float %4872, %4927
  %4929 = fmul contract float %4877, 0xBFF7154760000000
  %4930 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4929)
  %4931 = fadd contract float %4930, 1.000000e+00
  %4932 = tail call contract float @llvm.amdgcn.rcp.f32(float %4931)
  %4933 = fmul contract float %4877, %4932
  %4934 = fmul contract float %4882, %4933
  %4935 = fmul contract float %4887, 0xBFF7154760000000
  %4936 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %4935)
  %4937 = fadd contract float %4936, 1.000000e+00
  %4938 = tail call contract float @llvm.amdgcn.rcp.f32(float %4937)
  %4939 = fmul contract float %4887, %4938
  %4940 = fmul contract float %4892, %4939
  %4941 = tail call contract noundef float @llvm.fabs.f32(float %4898)
  %4942 = tail call contract noundef float @llvm.fabs.f32(float %4904)
  %4943 = tail call contract noundef float @llvm.maxnum.f32(float %4941, float %4942)
  %4944 = tail call contract noundef float @llvm.fabs.f32(float %4910)
  %4945 = tail call contract noundef float @llvm.maxnum.f32(float %4943, float %4944)
  %4946 = tail call contract noundef float @llvm.fabs.f32(float %4916)
  %4947 = tail call contract noundef float @llvm.maxnum.f32(float %4945, float %4946)
  %4948 = tail call contract noundef float @llvm.fabs.f32(float %4922)
  %4949 = tail call contract noundef float @llvm.maxnum.f32(float %4947, float %4948)
  %4950 = tail call contract noundef float @llvm.fabs.f32(float %4928)
  %4951 = tail call contract noundef float @llvm.maxnum.f32(float %4949, float %4950)
  %4952 = tail call contract noundef float @llvm.fabs.f32(float %4934)
  %4953 = tail call contract noundef float @llvm.maxnum.f32(float %4951, float %4952)
  %4954 = tail call contract noundef float @llvm.fabs.f32(float %4940)
  %4955 = tail call contract noundef float @llvm.maxnum.f32(float %4953, float %4954)
  %4956 = bitcast float %4955 to i32
  %4957 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %4956, i32 177, i32 15, i32 15, i1 true)
  %4958 = bitcast i32 %4957 to float
  %4959 = tail call contract noundef float @llvm.maxnum.f32(float %4955, float %4958)
  %4960 = bitcast float %4959 to i32
  %4961 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %4960, i32 78, i32 15, i32 15, i1 true)
  %4962 = bitcast i32 %4961 to float
  %4963 = tail call contract noundef float @llvm.maxnum.f32(float %4959, float %4962)
  %4964 = bitcast float %4963 to i32
  %4965 = add i32 %4964, 2097152
  %4966 = bitcast i32 %4965 to float
  %4967 = fmul contract float %4966, 2.500000e-01
  %4968 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 0, float %4898, float %4904, float %4967, i32 0)
  %4969 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %4968, float %4910, float %4916, float %4967, i32 1)
  %4970 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %4969, float %4922, float %4928, float %4967, i32 2)
  %4971 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %4970, float %4934, float %4940, float %4967, i32 3)
  %4972 = add i32 %3994, 24576
  %4973 = sext i32 %4972 to i64
  %4974 = getelementptr inbounds i8, ptr %39, i64 %4973
  store i32 %4971, ptr %4974, align 4, !tbaa !7, !alias.scope !16, !noalias !26, !nontemporal !27
  %4975 = or disjoint i32 %3831, 28672
  %4976 = or disjoint i32 %3825, %4975
  %4977 = or disjoint i32 %3826, %4975
  %4978 = zext nneg i32 %4976 to i64
  %4979 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4978
  %4980 = addrspacecast ptr %4979 to ptr addrspace(3)
  %4981 = load float, ptr addrspace(3) %4980, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4982 = zext nneg i32 %4977 to i64
  %4983 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4982
  %4984 = addrspacecast ptr %4983 to ptr addrspace(3)
  %4985 = load float, ptr addrspace(3) %4984, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4986 = or disjoint i32 %4976, 1
  %4987 = zext nneg i32 %4986 to i64
  %4988 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4987
  %4989 = addrspacecast ptr %4988 to ptr addrspace(3)
  %4990 = load float, ptr addrspace(3) %4989, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4991 = or disjoint i32 %4977, 1
  %4992 = zext nneg i32 %4991 to i64
  %4993 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4992
  %4994 = addrspacecast ptr %4993 to ptr addrspace(3)
  %4995 = load float, ptr addrspace(3) %4994, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %4996 = or disjoint i32 %4976, 2
  %4997 = zext nneg i32 %4996 to i64
  %4998 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %4997
  %4999 = addrspacecast ptr %4998 to ptr addrspace(3)
  %5000 = load float, ptr addrspace(3) %4999, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %5001 = or disjoint i32 %4977, 2
  %5002 = zext nneg i32 %5001 to i64
  %5003 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %5002
  %5004 = addrspacecast ptr %5003 to ptr addrspace(3)
  %5005 = load float, ptr addrspace(3) %5004, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %5006 = or disjoint i32 %4976, 3
  %5007 = zext nneg i32 %5006 to i64
  %5008 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %5007
  %5009 = addrspacecast ptr %5008 to ptr addrspace(3)
  %5010 = load float, ptr addrspace(3) %5009, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %5011 = or disjoint i32 %4977, 3
  %5012 = zext nneg i32 %5011 to i64
  %5013 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %5012
  %5014 = addrspacecast ptr %5013 to ptr addrspace(3)
  %5015 = load float, ptr addrspace(3) %5014, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %5016 = or disjoint i32 %4976, 4
  %5017 = zext nneg i32 %5016 to i64
  %5018 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %5017
  %5019 = addrspacecast ptr %5018 to ptr addrspace(3)
  %5020 = load float, ptr addrspace(3) %5019, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %5021 = or disjoint i32 %4977, 4
  %5022 = zext nneg i32 %5021 to i64
  %5023 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %5022
  %5024 = addrspacecast ptr %5023 to ptr addrspace(3)
  %5025 = load float, ptr addrspace(3) %5024, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %5026 = or disjoint i32 %4976, 5
  %5027 = zext nneg i32 %5026 to i64
  %5028 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %5027
  %5029 = addrspacecast ptr %5028 to ptr addrspace(3)
  %5030 = load float, ptr addrspace(3) %5029, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %5031 = or disjoint i32 %4977, 5
  %5032 = zext nneg i32 %5031 to i64
  %5033 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %5032
  %5034 = addrspacecast ptr %5033 to ptr addrspace(3)
  %5035 = load float, ptr addrspace(3) %5034, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %5036 = or disjoint i32 %4976, 6
  %5037 = zext nneg i32 %5036 to i64
  %5038 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %5037
  %5039 = addrspacecast ptr %5038 to ptr addrspace(3)
  %5040 = load float, ptr addrspace(3) %5039, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %5041 = or disjoint i32 %4977, 6
  %5042 = zext nneg i32 %5041 to i64
  %5043 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %5042
  %5044 = addrspacecast ptr %5043 to ptr addrspace(3)
  %5045 = load float, ptr addrspace(3) %5044, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %5046 = or disjoint i32 %4976, 7
  %5047 = zext nneg i32 %5046 to i64
  %5048 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %5047
  %5049 = addrspacecast ptr %5048 to ptr addrspace(3)
  %5050 = load float, ptr addrspace(3) %5049, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %5051 = or disjoint i32 %4977, 7
  %5052 = zext nneg i32 %5051 to i64
  %5053 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %5052
  %5054 = addrspacecast ptr %5053 to ptr addrspace(3)
  %5055 = load float, ptr addrspace(3) %5054, align 4, !tbaa !23, !alias.scope !21, !noalias !25
  %5056 = fmul contract float %4981, 0xBFF7154760000000
  %5057 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %5056)
  %5058 = fadd contract float %5057, 1.000000e+00
  %5059 = tail call contract float @llvm.amdgcn.rcp.f32(float %5058)
  %5060 = fmul contract float %4981, %5059
  %5061 = fmul contract float %4985, %5060
  %5062 = fmul contract float %4990, 0xBFF7154760000000
  %5063 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %5062)
  %5064 = fadd contract float %5063, 1.000000e+00
  %5065 = tail call contract float @llvm.amdgcn.rcp.f32(float %5064)
  %5066 = fmul contract float %4990, %5065
  %5067 = fmul contract float %4995, %5066
  %5068 = fmul contract float %5000, 0xBFF7154760000000
  %5069 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %5068)
  %5070 = fadd contract float %5069, 1.000000e+00
  %5071 = tail call contract float @llvm.amdgcn.rcp.f32(float %5070)
  %5072 = fmul contract float %5000, %5071
  %5073 = fmul contract float %5005, %5072
  %5074 = fmul contract float %5010, 0xBFF7154760000000
  %5075 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %5074)
  %5076 = fadd contract float %5075, 1.000000e+00
  %5077 = tail call contract float @llvm.amdgcn.rcp.f32(float %5076)
  %5078 = fmul contract float %5010, %5077
  %5079 = fmul contract float %5015, %5078
  %5080 = fmul contract float %5020, 0xBFF7154760000000
  %5081 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %5080)
  %5082 = fadd contract float %5081, 1.000000e+00
  %5083 = tail call contract float @llvm.amdgcn.rcp.f32(float %5082)
  %5084 = fmul contract float %5020, %5083
  %5085 = fmul contract float %5025, %5084
  %5086 = fmul contract float %5030, 0xBFF7154760000000
  %5087 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %5086)
  %5088 = fadd contract float %5087, 1.000000e+00
  %5089 = tail call contract float @llvm.amdgcn.rcp.f32(float %5088)
  %5090 = fmul contract float %5030, %5089
  %5091 = fmul contract float %5035, %5090
  %5092 = fmul contract float %5040, 0xBFF7154760000000
  %5093 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %5092)
  %5094 = fadd contract float %5093, 1.000000e+00
  %5095 = tail call contract float @llvm.amdgcn.rcp.f32(float %5094)
  %5096 = fmul contract float %5040, %5095
  %5097 = fmul contract float %5045, %5096
  %5098 = fmul contract float %5050, 0xBFF7154760000000
  %5099 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %5098)
  %5100 = fadd contract float %5099, 1.000000e+00
  %5101 = tail call contract float @llvm.amdgcn.rcp.f32(float %5100)
  %5102 = fmul contract float %5050, %5101
  %5103 = fmul contract float %5055, %5102
  %5104 = tail call contract noundef float @llvm.fabs.f32(float %5061)
  %5105 = tail call contract noundef float @llvm.fabs.f32(float %5067)
  %5106 = tail call contract noundef float @llvm.maxnum.f32(float %5104, float %5105)
  %5107 = tail call contract noundef float @llvm.fabs.f32(float %5073)
  %5108 = tail call contract noundef float @llvm.maxnum.f32(float %5106, float %5107)
  %5109 = tail call contract noundef float @llvm.fabs.f32(float %5079)
  %5110 = tail call contract noundef float @llvm.maxnum.f32(float %5108, float %5109)
  %5111 = tail call contract noundef float @llvm.fabs.f32(float %5085)
  %5112 = tail call contract noundef float @llvm.maxnum.f32(float %5110, float %5111)
  %5113 = tail call contract noundef float @llvm.fabs.f32(float %5091)
  %5114 = tail call contract noundef float @llvm.maxnum.f32(float %5112, float %5113)
  %5115 = tail call contract noundef float @llvm.fabs.f32(float %5097)
  %5116 = tail call contract noundef float @llvm.maxnum.f32(float %5114, float %5115)
  %5117 = tail call contract noundef float @llvm.fabs.f32(float %5103)
  %5118 = tail call contract noundef float @llvm.maxnum.f32(float %5116, float %5117)
  %5119 = bitcast float %5118 to i32
  %5120 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %5119, i32 177, i32 15, i32 15, i1 true)
  %5121 = bitcast i32 %5120 to float
  %5122 = tail call contract noundef float @llvm.maxnum.f32(float %5118, float %5121)
  %5123 = bitcast float %5122 to i32
  %5124 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %5123, i32 78, i32 15, i32 15, i1 true)
  %5125 = bitcast i32 %5124 to float
  %5126 = tail call contract noundef float @llvm.maxnum.f32(float %5122, float %5125)
  %5127 = bitcast float %5126 to i32
  %5128 = add i32 %5127, 2097152
  %5129 = bitcast i32 %5128 to float
  %5130 = fmul contract float %5129, 2.500000e-01
  %5131 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 0, float %5061, float %5067, float %5130, i32 0)
  %5132 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %5131, float %5073, float %5079, float %5130, i32 1)
  %5133 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %5132, float %5085, float %5091, float %5130, i32 2)
  %5134 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %5133, float %5097, float %5103, float %5130, i32 3)
  %5135 = add i32 %3994, 28672
  %5136 = sext i32 %5135 to i64
  %5137 = getelementptr inbounds i8, ptr %39, i64 %5136
  store i32 %5134, ptr %5137, align 4, !tbaa !7, !alias.scope !16, !noalias !26, !nontemporal !27
  %5138 = icmp eq i32 %3822, 0
  br i1 %5138, label %5139, label %5205

5139:                                             ; preds = %32
  %5140 = bitcast float %5130 to i32
  %5141 = lshr i32 %5140, 23
  %5142 = tail call noundef range(i32 0, 255) i32 @llvm.umin.i32(i32 range(i32 0, 512) %5141, i32 254)
  %5143 = bitcast float %4967 to i32
  %5144 = lshr i32 %5143, 23
  %5145 = tail call noundef range(i32 0, 255) i32 @llvm.umin.i32(i32 range(i32 0, 512) %5144, i32 254)
  %5146 = bitcast float %4804 to i32
  %5147 = lshr i32 %5146, 23
  %5148 = tail call noundef range(i32 0, 255) i32 @llvm.umin.i32(i32 range(i32 0, 512) %5147, i32 254)
  %5149 = bitcast float %4641 to i32
  %5150 = lshr i32 %5149, 23
  %5151 = tail call noundef range(i32 0, 255) i32 @llvm.umin.i32(i32 range(i32 0, 512) %5150, i32 254)
  %5152 = bitcast float %4478 to i32
  %5153 = lshr i32 %5152, 23
  %5154 = tail call noundef range(i32 0, 255) i32 @llvm.umin.i32(i32 range(i32 0, 512) %5153, i32 254)
  %5155 = bitcast float %4315 to i32
  %5156 = lshr i32 %5155, 23
  %5157 = tail call noundef range(i32 0, 255) i32 @llvm.umin.i32(i32 range(i32 0, 512) %5156, i32 254)
  %5158 = bitcast float %4152 to i32
  %5159 = lshr i32 %5158, 23
  %5160 = tail call noundef range(i32 0, 255) i32 @llvm.umin.i32(i32 range(i32 0, 512) %5159, i32 254)
  %5161 = bitcast float %3986 to i32
  %5162 = lshr i32 %5161, 23
  %5163 = tail call noundef range(i32 0, 255) i32 @llvm.umin.i32(i32 range(i32 0, 512) %5162, i32 254)
  %5164 = shl i32 %42, 9
  %5165 = shl nsw i32 %44, 5
  %5166 = and i32 %5165, 1073741760
  %5167 = add i32 %5166, %5164
  %5168 = or disjoint i32 %5167, %3819
  %5169 = shl nsw i32 %44, 1
  %5170 = and i32 %5169, 2
  %5171 = trunc nuw nsw i32 %5163 to i16
  %5172 = trunc nuw nsw i32 %5160 to i16
  %5173 = shl nuw i16 %5172, 8
  %5174 = or disjoint i16 %5173, %5171
  %5175 = shl nuw nsw i32 %3821, 6
  %5176 = shl i32 %5168, 2
  %5177 = or disjoint i32 %5176, %5175
  %5178 = or disjoint i32 %5177, %5170
  %5179 = sext i32 %5178 to i64
  %5180 = getelementptr inbounds i8, ptr %19, i64 %5179
  store i16 %5174, ptr %5180, align 2, !tbaa !28, !alias.scope !19, !noalias !30
  %5181 = trunc nuw nsw i32 %5157 to i16
  %5182 = trunc nuw nsw i32 %5154 to i16
  %5183 = shl nuw i16 %5182, 8
  %5184 = or disjoint i16 %5183, %5181
  %5185 = add i32 %5177, 512
  %5186 = or disjoint i32 %5185, %5170
  %5187 = sext i32 %5186 to i64
  %5188 = getelementptr inbounds i8, ptr %19, i64 %5187
  store i16 %5184, ptr %5188, align 2, !tbaa !28, !alias.scope !19, !noalias !30
  %5189 = trunc nuw nsw i32 %5151 to i16
  %5190 = trunc nuw nsw i32 %5148 to i16
  %5191 = shl nuw i16 %5190, 8
  %5192 = or disjoint i16 %5191, %5189
  %5193 = add i32 %5177, 1024
  %5194 = or disjoint i32 %5193, %5170
  %5195 = sext i32 %5194 to i64
  %5196 = getelementptr inbounds i8, ptr %19, i64 %5195
  store i16 %5192, ptr %5196, align 2, !tbaa !28, !alias.scope !19, !noalias !30
  %5197 = trunc nuw nsw i32 %5145 to i16
  %5198 = trunc nuw nsw i32 %5142 to i16
  %5199 = shl nuw i16 %5198, 8
  %5200 = or disjoint i16 %5199, %5197
  %5201 = add i32 %5177, 1536
  %5202 = or disjoint i32 %5201, %5170
  %5203 = sext i32 %5202 to i64
  %5204 = getelementptr inbounds i8, ptr %19, i64 %5203
  store i16 %5200, ptr %5204, align 2, !tbaa !28, !alias.scope !19, !noalias !30
  br label %5205

5205:                                             ; preds = %5139, %32, %11
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

; Function Attrs: convergent mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32>, <4 x i32>, <4 x float>, i32 immarg, i32 immarg, i32 immarg, i32, i32 immarg, i32) #7

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #6

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.update.dpp.i32(i32, i32, i32 immarg, i32 immarg, i32 immarg, i1 immarg) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32, float, float, float, i32 immarg) #8

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.rcp.f32(float) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workitem.id.x() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umin.i32(i32, i32) #9

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #10

attributes #0 = { convergent mustprogress norecurse nounwind "amdgpu-flat-work-group-size"="1,256" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-waves-per-eu"="1" "denormal-fp-math-f32"="preserve-sign,preserve-sign" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx950" "target-features"="+16-bit-insts,+ashr-pk-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-buffer-pk-add-bf16-inst,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-global-pk-add-bf16-inst,+bf8-cvt-scale-insts,+bitop3-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot12-insts,+dot13-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+f16bf16-to-fp6bf6-cvt-scale-insts,+f32-to-f16bf16-cvt-sr-insts,+fp4-cvt-scale-insts,+fp6bf6-cvt-scale-insts,+fp8-conversion-insts,+fp8-cvt-scale-insts,+fp8-insts,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+gfx940-insts,+gfx950-insts,+mai-insts,+permlane16-swap,+permlane32-swap,+prng-inst,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" "uniform-work-group-size"="false" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #2 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #3 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #6 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #7 = { convergent mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #8 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #9 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #10 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #11 = { convergent nounwind }

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
!12 = !{i64 2158260115}
!13 = !{i64 2158262531}
!14 = !{i64 2158260719}
!15 = !{i64 2158263135}
!16 = !{!17}
!17 = distinct !{!17, !18, !"_ZN5aiter9mxfp4_moe11gemm_common27apply_cshuffle_quant_epilogILi1024ELi128EEEvRAdvT0_Li16E_A4_KDv4_fPhS8_iiiiiiiPf: argument 0"}
!18 = distinct !{!18, !"_ZN5aiter9mxfp4_moe11gemm_common27apply_cshuffle_quant_epilogILi1024ELi128EEEvRAdvT0_Li16E_A4_KDv4_fPhS8_iiiiiiiPf"}
!19 = !{!20}
!20 = distinct !{!20, !18, !"_ZN5aiter9mxfp4_moe11gemm_common27apply_cshuffle_quant_epilogILi1024ELi128EEEvRAdvT0_Li16E_A4_KDv4_fPhS8_iiiiiiiPf: argument 1"}
!21 = !{!22}
!22 = distinct !{!22, !18, !"_ZN5aiter9mxfp4_moe11gemm_common27apply_cshuffle_quant_epilogILi1024ELi128EEEvRAdvT0_Li16E_A4_KDv4_fPhS8_iiiiiiiPf: argument 2"}
!23 = !{!24, !24, i64 0}
!24 = !{!"float", !9, i64 0}
!25 = !{!17, !20}
!26 = !{!20, !22}
!27 = !{i32 1}
!28 = !{!29, !29, i64 0}
!29 = !{!"short", !9, i64 0}
!30 = !{!17, !22}
