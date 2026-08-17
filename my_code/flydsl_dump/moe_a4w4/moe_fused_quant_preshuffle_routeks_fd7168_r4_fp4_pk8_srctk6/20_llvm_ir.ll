; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

define amdgpu_kernel void @moe_fused_quant_preshuffle_routeks_fd7168_r4_fp4_pk8_srctk6(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, i32 %5, i32 %6, ptr addrspace(1) %7) #0 !reqd_work_group_size !1 {
  %9 = call range(i32 0, 256) i32 @llvm.amdgcn.workitem.id.x()
  %10 = sext i32 %9 to i64
  %11 = trunc i64 %10 to i32
  %12 = call i32 @llvm.amdgcn.workgroup.id.x()
  %13 = sext i32 %12 to i64
  %14 = trunc i64 %13 to i32
  %15 = udiv i32 %11, 32
  %16 = mul i32 %15, 32
  %17 = sub i32 %11, %16
  %18 = mul i32 %14, 8
  %19 = add i32 %18, %15
  %20 = ptrtoint ptr addrspace(1) %7 to i64
  %21 = icmp ne i64 %20, 0
  br i1 %21, label %22, label %26

22:                                               ; preds = %8
  %23 = inttoptr i64 %20 to ptr
  %24 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %23, i16 0, i64 4294967295, i32 159744)
  %25 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %24, i32 0, i32 0, i32 0)
  br label %27

26:                                               ; preds = %8
  br label %27

27:                                               ; preds = %22, %26
  %28 = phi i32 [ %6, %26 ], [ %25, %22 ]
  br label %29

29:                                               ; preds = %27
  %30 = icmp ult i32 %19, %28
  br i1 %30, label %31, label %156

31:                                               ; preds = %29
  %32 = ptrtoint ptr addrspace(1) %3 to i64
  %33 = inttoptr i64 %32 to ptr
  %34 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %33, i16 0, i64 4294967295, i32 159744)
  %35 = mul i32 %19, 4
  %36 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %34, i32 %35, i32 0, i32 0)
  %37 = udiv i32 %36, 64
  %38 = mul i32 %37, 64
  %39 = sub i32 %36, %38
  %40 = udiv i32 %39, 16
  %41 = mul i32 %40, 16
  %42 = sub i32 %39, %41
  %43 = mul i32 %37, 3584
  %44 = add i32 %43, %41
  %45 = add i32 %44, %42
  %46 = udiv i32 %19, 6
  %47 = ptrtoint ptr addrspace(1) %2 to i64
  %48 = inttoptr i64 %47 to ptr
  %49 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %48, i16 0, i64 4294967295, i32 159744)
  %50 = ptrtoint ptr addrspace(1) %1 to i64
  %51 = ptrtoint ptr addrspace(1) %0 to i64
  %52 = udiv i32 %17, 4
  %53 = mul i32 %52, 4
  %54 = sub i32 %17, %53
  %55 = icmp eq i32 %54, 0
  %56 = call i32 @llvm.amdgcn.workgroup.id.y()
  %57 = sext i32 %56 to i64
  %58 = trunc i64 %57 to i32
  %59 = zext i32 %36 to i64
  %60 = mul i64 %59, 3584
  %61 = add i64 %50, %60
  %62 = inttoptr i64 %61 to ptr
  %63 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %62, i16 0, i64 3584, i32 159744)
  %64 = zext i32 %46 to i64
  %65 = mul i64 %64, 14336
  %66 = add i64 %51, %65
  %67 = inttoptr i64 %66 to ptr
  %68 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %67, i16 0, i64 14336, i32 159744)
  %69 = mul i32 %58, 8
  %70 = add i32 %69, %52
  %71 = icmp ult i32 %70, 224
  br i1 %71, label %72, label %155

72:                                               ; preds = %31
  %73 = mul i32 %70, 32
  %74 = mul i32 %54, 8
  %75 = add i32 %73, %74
  %76 = lshr i32 %75, 1
  %77 = mul i32 %76, 4
  %78 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %68, i32 %77, i32 0, i32 0)
  %79 = bitcast <4 x i32> %78 to <8 x bfloat>
  %80 = fpext <8 x bfloat> %79 to <8 x float>
  %81 = extractelement <8 x float> %80, i64 0
  %82 = call float @llvm.fabs.f32(float %81)
  %83 = call float @llvm.maximum.f32(float %82, float 0.000000e+00)
  %84 = extractelement <8 x float> %80, i64 1
  %85 = call float @llvm.fabs.f32(float %84)
  %86 = call float @llvm.maximum.f32(float %83, float %85)
  %87 = extractelement <8 x float> %80, i64 2
  %88 = call float @llvm.fabs.f32(float %87)
  %89 = call float @llvm.maximum.f32(float %86, float %88)
  %90 = extractelement <8 x float> %80, i64 3
  %91 = call float @llvm.fabs.f32(float %90)
  %92 = call float @llvm.maximum.f32(float %89, float %91)
  %93 = extractelement <8 x float> %80, i64 4
  %94 = call float @llvm.fabs.f32(float %93)
  %95 = call float @llvm.maximum.f32(float %92, float %94)
  %96 = extractelement <8 x float> %80, i64 5
  %97 = call float @llvm.fabs.f32(float %96)
  %98 = call float @llvm.maximum.f32(float %95, float %97)
  %99 = extractelement <8 x float> %80, i64 6
  %100 = call float @llvm.fabs.f32(float %99)
  %101 = call float @llvm.maximum.f32(float %98, float %100)
  %102 = extractelement <8 x float> %80, i64 7
  %103 = call float @llvm.fabs.f32(float %102)
  %104 = call float @llvm.maximum.f32(float %101, float %103)
  %105 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %106 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %105)
  %107 = add i32 %106, 32
  %108 = and i32 %107, -32
  %109 = xor i32 %106, 1
  %110 = icmp slt i32 %109, %108
  %111 = select i1 %110, i32 %109, i32 %106
  %112 = shl i32 %111, 2
  %113 = bitcast float %104 to i32
  %114 = call i32 @llvm.amdgcn.ds.bpermute(i32 %112, i32 %113)
  %115 = bitcast i32 %114 to float
  %116 = call float @llvm.maximum.f32(float %104, float %115)
  %117 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %118 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %117)
  %119 = add i32 %118, 32
  %120 = and i32 %119, -32
  %121 = xor i32 %118, 2
  %122 = icmp slt i32 %121, %120
  %123 = select i1 %122, i32 %121, i32 %118
  %124 = shl i32 %123, 2
  %125 = bitcast float %116 to i32
  %126 = call i32 @llvm.amdgcn.ds.bpermute(i32 %124, i32 %125)
  %127 = bitcast i32 %126 to float
  %128 = call float @llvm.maximum.f32(float %116, float %127)
  %129 = fmul float %128, 0x3FC5555560000000
  %130 = bitcast float %129 to i32
  %131 = and i32 %130, 8388607
  %132 = lshr i32 %130, 23
  %133 = and i32 %132, 255
  %134 = icmp ne i32 %131, 0
  %135 = add i32 %133, 1
  %136 = select i1 %134, i32 %135, i32 %133
  %137 = call i32 @llvm.smax.i32(i32 %136, i32 0)
  %138 = call i32 @llvm.smin.i32(i32 %137, i32 255)
  %139 = shl i32 %138, 23
  %140 = bitcast i32 %139 to float
  %141 = call i32 asm "v_cvt_scalef32_pk8_fp4_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %79, float %140)
  %142 = udiv i32 %70, 4
  %143 = mul i32 %142, 4
  %144 = sub i32 %70, %143
  %145 = trunc i32 %138 to i8
  %146 = mul i32 %70, 16
  %147 = mul i32 %54, 4
  %148 = add i32 %146, %147
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %141, ptr addrspace(8) %63, i32 %148, i32 0, i32 0)
  br i1 %55, label %149, label %154

149:                                              ; preds = %72
  %150 = mul i32 %142, 64
  %151 = add i32 %45, %150
  %152 = mul i32 %151, 4
  %153 = add i32 %152, %144
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %145, ptr addrspace(8) %49, i32 %153, i32 0, i32 0)
  br label %154

154:                                              ; preds = %149, %72
  br label %155

155:                                              ; preds = %154, %31
  br label %156

156:                                              ; preds = %155, %29
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr readnone, i16, i64, i32) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #3

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximum.f32(float, float) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #4

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #4

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #5

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32, ptr addrspace(8) writeonly captures(none), i32, i32, i32 immarg) #6

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8, ptr addrspace(8) writeonly captures(none), i32, i32, i32 immarg) #6

attributes #0 = { "amdgpu-flat-work-group-size"="256,256" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #4 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #5 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #6 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 256, i32 1, i32 1}
