; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

@__shared_alloc_2 = external dso_local addrspace(3) global [4 x i8], align 16
@__shared_alloc_1 = external dso_local addrspace(3) global [2048 x i8], align 16
@__shared_alloc_0 = external dso_local addrspace(3) global [2048 x i8], align 16

define amdgpu_kernel void @moe_contiguous_psum_remap(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, i32 %5, i32 %6, i32 %7, i32 %8, ptr addrspace(1) %9) #0 !reqd_work_group_size !1 {
  %11 = call range(i32 0, 512) i32 @llvm.amdgcn.workitem.id.x()
  %12 = sext i32 %11 to i64
  %13 = trunc i64 %12 to i32
  %14 = sub i32 %8, 1
  %15 = ptrtoint ptr addrspace(1) %0 to i64
  %16 = inttoptr i64 %15 to ptr
  %17 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %16, i16 0, i64 4294967295, i32 159744)
  %18 = ptrtoint ptr addrspace(1) %1 to i64
  %19 = inttoptr i64 %18 to ptr
  %20 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %19, i16 0, i64 4294967295, i32 159744)
  %21 = ptrtoint ptr addrspace(1) %2 to i64
  %22 = inttoptr i64 %21 to ptr
  %23 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %22, i16 0, i64 4294967295, i32 159744)
  %24 = ptrtoint ptr addrspace(1) %3 to i64
  %25 = inttoptr i64 %24 to ptr
  %26 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %25, i16 0, i64 4294967295, i32 159744)
  %27 = ptrtoint ptr addrspace(1) %4 to i64
  %28 = inttoptr i64 %27 to ptr
  %29 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %28, i16 0, i64 4294967295, i32 159744)
  %30 = icmp eq i32 %13, 0
  br i1 %30, label %31, label %32

31:                                               ; preds = %10
  store i32 0, ptr addrspace(3) @__shared_alloc_2, align 4
  br label %32

32:                                               ; preds = %31, %10
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  br label %33

33:                                               ; preds = %179, %32
  %34 = phi i32 [ %180, %179 ], [ 0, %32 ]
  %35 = icmp slt i32 %34, %6
  br i1 %35, label %36, label %181

36:                                               ; preds = %33
  %37 = add i32 %34, %13
  %38 = icmp ult i32 %37, %6
  br i1 %38, label %39, label %42

39:                                               ; preds = %36
  %40 = mul i32 %37, 4
  %41 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %17, i32 %40, i32 0, i32 0)
  br label %43

42:                                               ; preds = %36
  br label %43

43:                                               ; preds = %39, %42
  %44 = phi i32 [ 0, %42 ], [ %41, %39 ]
  br label %45

45:                                               ; preds = %43
  %46 = add i32 %44, %14
  %47 = udiv i32 %46, %8
  %48 = mul i32 %47, %8
  %49 = zext i32 %13 to i64
  %50 = getelementptr i32, ptr addrspace(3) @__shared_alloc_0, i64 %49
  store i32 %48, ptr addrspace(3) %50, align 4
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %51 = load i32, ptr addrspace(3) %50, align 4
  %52 = icmp uge i32 %13, 1
  br i1 %52, label %53, label %58

53:                                               ; preds = %45
  %54 = sub i32 %13, 1
  %55 = zext i32 %54 to i64
  %56 = getelementptr i32, ptr addrspace(3) @__shared_alloc_0, i64 %55
  %57 = load i32, ptr addrspace(3) %56, align 4
  br label %59

58:                                               ; preds = %45
  br label %59

59:                                               ; preds = %53, %58
  %60 = phi i32 [ 0, %58 ], [ %57, %53 ]
  br label %61

61:                                               ; preds = %59
  %62 = add i32 %51, %60
  %63 = getelementptr i32, ptr addrspace(3) @__shared_alloc_1, i64 %49
  store i32 %62, ptr addrspace(3) %63, align 4
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %64 = load i32, ptr addrspace(3) %63, align 4
  %65 = icmp uge i32 %13, 2
  br i1 %65, label %66, label %71

66:                                               ; preds = %61
  %67 = sub i32 %13, 2
  %68 = zext i32 %67 to i64
  %69 = getelementptr i32, ptr addrspace(3) @__shared_alloc_1, i64 %68
  %70 = load i32, ptr addrspace(3) %69, align 4
  br label %72

71:                                               ; preds = %61
  br label %72

72:                                               ; preds = %66, %71
  %73 = phi i32 [ 0, %71 ], [ %70, %66 ]
  br label %74

74:                                               ; preds = %72
  %75 = add i32 %64, %73
  store i32 %75, ptr addrspace(3) %50, align 4
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %76 = load i32, ptr addrspace(3) %50, align 4
  %77 = icmp uge i32 %13, 4
  br i1 %77, label %78, label %83

78:                                               ; preds = %74
  %79 = sub i32 %13, 4
  %80 = zext i32 %79 to i64
  %81 = getelementptr i32, ptr addrspace(3) @__shared_alloc_0, i64 %80
  %82 = load i32, ptr addrspace(3) %81, align 4
  br label %84

83:                                               ; preds = %74
  br label %84

84:                                               ; preds = %78, %83
  %85 = phi i32 [ 0, %83 ], [ %82, %78 ]
  br label %86

86:                                               ; preds = %84
  %87 = add i32 %76, %85
  store i32 %87, ptr addrspace(3) %63, align 4
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %88 = load i32, ptr addrspace(3) %63, align 4
  %89 = icmp uge i32 %13, 8
  br i1 %89, label %90, label %95

90:                                               ; preds = %86
  %91 = sub i32 %13, 8
  %92 = zext i32 %91 to i64
  %93 = getelementptr i32, ptr addrspace(3) @__shared_alloc_1, i64 %92
  %94 = load i32, ptr addrspace(3) %93, align 4
  br label %96

95:                                               ; preds = %86
  br label %96

96:                                               ; preds = %90, %95
  %97 = phi i32 [ 0, %95 ], [ %94, %90 ]
  br label %98

98:                                               ; preds = %96
  %99 = add i32 %88, %97
  store i32 %99, ptr addrspace(3) %50, align 4
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %100 = load i32, ptr addrspace(3) %50, align 4
  %101 = icmp uge i32 %13, 16
  br i1 %101, label %102, label %107

102:                                              ; preds = %98
  %103 = sub i32 %13, 16
  %104 = zext i32 %103 to i64
  %105 = getelementptr i32, ptr addrspace(3) @__shared_alloc_0, i64 %104
  %106 = load i32, ptr addrspace(3) %105, align 4
  br label %108

107:                                              ; preds = %98
  br label %108

108:                                              ; preds = %102, %107
  %109 = phi i32 [ 0, %107 ], [ %106, %102 ]
  br label %110

110:                                              ; preds = %108
  %111 = add i32 %100, %109
  store i32 %111, ptr addrspace(3) %63, align 4
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %112 = load i32, ptr addrspace(3) %63, align 4
  %113 = icmp uge i32 %13, 32
  br i1 %113, label %114, label %119

114:                                              ; preds = %110
  %115 = sub i32 %13, 32
  %116 = zext i32 %115 to i64
  %117 = getelementptr i32, ptr addrspace(3) @__shared_alloc_1, i64 %116
  %118 = load i32, ptr addrspace(3) %117, align 4
  br label %120

119:                                              ; preds = %110
  br label %120

120:                                              ; preds = %114, %119
  %121 = phi i32 [ 0, %119 ], [ %118, %114 ]
  br label %122

122:                                              ; preds = %120
  %123 = add i32 %112, %121
  store i32 %123, ptr addrspace(3) %50, align 4
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %124 = load i32, ptr addrspace(3) %50, align 4
  %125 = icmp uge i32 %13, 64
  br i1 %125, label %126, label %131

126:                                              ; preds = %122
  %127 = sub i32 %13, 64
  %128 = zext i32 %127 to i64
  %129 = getelementptr i32, ptr addrspace(3) @__shared_alloc_0, i64 %128
  %130 = load i32, ptr addrspace(3) %129, align 4
  br label %132

131:                                              ; preds = %122
  br label %132

132:                                              ; preds = %126, %131
  %133 = phi i32 [ 0, %131 ], [ %130, %126 ]
  br label %134

134:                                              ; preds = %132
  %135 = add i32 %124, %133
  store i32 %135, ptr addrspace(3) %63, align 4
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %136 = load i32, ptr addrspace(3) %63, align 4
  %137 = icmp uge i32 %13, 128
  br i1 %137, label %138, label %143

138:                                              ; preds = %134
  %139 = sub i32 %13, 128
  %140 = zext i32 %139 to i64
  %141 = getelementptr i32, ptr addrspace(3) @__shared_alloc_1, i64 %140
  %142 = load i32, ptr addrspace(3) %141, align 4
  br label %144

143:                                              ; preds = %134
  br label %144

144:                                              ; preds = %138, %143
  %145 = phi i32 [ 0, %143 ], [ %142, %138 ]
  br label %146

146:                                              ; preds = %144
  %147 = add i32 %136, %145
  store i32 %147, ptr addrspace(3) %50, align 4
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %148 = load i32, ptr addrspace(3) %50, align 4
  %149 = icmp uge i32 %13, 256
  br i1 %149, label %150, label %155

150:                                              ; preds = %146
  %151 = sub i32 %13, 256
  %152 = zext i32 %151 to i64
  %153 = getelementptr i32, ptr addrspace(3) @__shared_alloc_0, i64 %152
  %154 = load i32, ptr addrspace(3) %153, align 4
  br label %156

155:                                              ; preds = %146
  br label %156

156:                                              ; preds = %150, %155
  %157 = phi i32 [ 0, %155 ], [ %154, %150 ]
  br label %158

158:                                              ; preds = %156
  %159 = add i32 %148, %157
  store i32 %159, ptr addrspace(3) %63, align 4
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %160 = load i32, ptr addrspace(3) @__shared_alloc_2, align 4
  br i1 %38, label %161, label %175

161:                                              ; preds = %158
  %162 = icmp ne i32 %13, 0
  br i1 %162, label %163, label %168

163:                                              ; preds = %161
  %164 = sub i32 %13, 1
  %165 = zext i32 %164 to i64
  %166 = getelementptr i32, ptr addrspace(3) @__shared_alloc_1, i64 %165
  %167 = load i32, ptr addrspace(3) %166, align 4
  br label %169

168:                                              ; preds = %161
  br label %169

169:                                              ; preds = %163, %168
  %170 = phi i32 [ 0, %168 ], [ %167, %163 ]
  br label %171

171:                                              ; preds = %169
  %172 = add i32 %170, %160
  %173 = mul i32 %37, 4
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %172, ptr addrspace(8) %23, i32 %173, i32 0, i32 0)
  %174 = add i32 %172, %44
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %174, ptr addrspace(8) %26, i32 %173, i32 0, i32 0)
  br label %175

175:                                              ; preds = %171, %158
  %176 = load i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_alloc_1, i32 2044), align 4
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  br i1 %30, label %177, label %179

177:                                              ; preds = %175
  %178 = add i32 %160, %176
  store i32 %178, ptr addrspace(3) @__shared_alloc_2, align 4
  br label %179

179:                                              ; preds = %177, %175
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %180 = add i32 %34, 512
  br label %33

181:                                              ; preds = %33
  br i1 %30, label %182, label %186

182:                                              ; preds = %181
  %183 = load i32, ptr addrspace(3) @__shared_alloc_2, align 4
  %184 = icmp sgt i32 %183, %8
  %185 = select i1 %184, i32 %183, i32 %8
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %185, ptr addrspace(8) %29, i32 0, i32 0, i32 0)
  br label %186

186:                                              ; preds = %182, %181
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %187 = ptrtoint ptr addrspace(1) %9 to i64
  %188 = icmp ne i64 %187, 0
  br i1 %188, label %189, label %193

189:                                              ; preds = %186
  %190 = inttoptr i64 %187 to ptr
  %191 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %190, i16 0, i64 4294967295, i32 159744)
  %192 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %191, i32 0, i32 0, i32 0)
  br label %194

193:                                              ; preds = %186
  br label %194

194:                                              ; preds = %189, %193
  %195 = phi i32 [ %5, %193 ], [ %192, %189 ]
  br label %196

196:                                              ; preds = %194
  br label %197

197:                                              ; preds = %200, %196
  %198 = phi i32 [ %209, %200 ], [ %13, %196 ]
  %199 = icmp slt i32 %198, %195
  br i1 %199, label %200, label %210

200:                                              ; preds = %197
  %201 = mul i32 %198, 4
  %202 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %20, i32 %201, i32 0, i32 0)
  %203 = udiv i32 %202, %7
  %204 = mul i32 %203, %7
  %205 = sub i32 %202, %204
  %206 = mul i32 %203, 4
  %207 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %23, i32 %206, i32 0, i32 0)
  %208 = add i32 %207, %205
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %208, ptr addrspace(8) %20, i32 %201, i32 0, i32 0)
  %209 = add i32 %198, 512
  br label %197

210:                                              ; preds = %197
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr readnone, i16, i64, i32) #2

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier.signal(i32 immarg) #3

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier.wait(i16 immarg) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32, ptr addrspace(8) writeonly captures(none), i32, i32, i32 immarg) #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #5

attributes #0 = { "amdgpu-flat-work-group-size"="512,512" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nocallback nofree nounwind willreturn }
attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
attributes #5 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 512, i32 1, i32 1}
