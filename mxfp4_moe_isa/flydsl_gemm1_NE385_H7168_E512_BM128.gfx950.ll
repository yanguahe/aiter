; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

@smem_g1 = external addrspace(3) global [131072 x i8], align 1024

define amdgpu_kernel void @gemm1_kernel_0(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, ptr addrspace(1) %5, ptr addrspace(1) %6, ptr addrspace(1) %7, ptr addrspace(1) %8, ptr addrspace(1) %9, i32 %10, i32 %11) #0 {
  %13 = call i32 @llvm.amdgcn.workitem.id.x()
  %14 = sext i32 %13 to i64
  %15 = call i32 @llvm.amdgcn.workgroup.id.x()
  %16 = sext i32 %15 to i64
  %17 = udiv i64 %16, 4
  %18 = urem i64 %16, 4
  %19 = lshr i64 %14, 6
  %20 = and i64 %19, 3
  %21 = and i64 %14, 63
  %22 = lshr i64 %21, 4
  %23 = and i64 %22, 3
  %24 = and i64 %21, 15
  %25 = sext i32 %10 to i64
  %26 = sext i32 %11 to i64
  %27 = mul i64 %25, 3584
  %28 = trunc i64 %27 to i32
  %29 = ptrtoint ptr addrspace(1) %0 to i64
  %30 = inttoptr i64 %29 to ptr
  %31 = sext i32 %28 to i64
  %32 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %30, i16 0, i64 %31, i32 159744)
  %33 = ptrtoint ptr addrspace(1) %2 to i64
  %34 = inttoptr i64 %33 to ptr
  %35 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %34, i16 0, i64 1412956160, i32 159744)
  %36 = udiv i64 %26, 128
  %37 = mul i64 %36, 4
  %38 = trunc i64 %37 to i32
  %39 = ptrtoint ptr addrspace(1) %4 to i64
  %40 = inttoptr i64 %39 to ptr
  %41 = sext i32 %38 to i64
  %42 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %40, i16 0, i64 %41, i32 159744)
  %43 = trunc i64 %17 to i32
  %44 = mul i32 %43, 4
  %45 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %42, i32 %44, i32 0, i32 0)
  %46 = mul i64 %17, 128
  %47 = mul i64 %26, 4
  %48 = trunc i64 %47 to i32
  %49 = ptrtoint ptr addrspace(1) %6 to i64
  %50 = inttoptr i64 %49 to ptr
  %51 = sext i32 %48 to i64
  %52 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %50, i16 0, i64 %51, i32 159744)
  %53 = udiv i64 %26, 32
  %54 = add i64 %53, 2
  %55 = mul i64 %54, 7168
  %56 = trunc i64 %55 to i32
  %57 = ptrtoint ptr addrspace(1) %1 to i64
  %58 = inttoptr i64 %57 to ptr
  %59 = sext i32 %56 to i64
  %60 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %58, i16 0, i64 %59, i32 159744)
  %61 = ptrtoint ptr addrspace(1) %3 to i64
  %62 = inttoptr i64 %61 to ptr
  %63 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %62, i16 0, i64 88309760, i32 159744)
  %64 = sext i32 %45 to i64
  %65 = mul i64 %14, 4
  %66 = trunc i64 %65 to i32
  %67 = sdiv i32 %66, 32
  %68 = srem i32 %67, 128
  %69 = srem i32 %66, 32
  %70 = sext i32 %68 to i64
  %71 = sext i32 %69 to i64
  %72 = add i64 %46, %70
  %73 = trunc i64 %72 to i32
  %74 = mul i32 %73, 4
  %75 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %52, i32 %74, i32 0, i32 0)
  %76 = sext i32 %75 to i64
  %77 = mul i64 %76, 896
  %78 = add i64 %65, 1024
  %79 = trunc i64 %78 to i32
  %80 = sdiv i32 %79, 32
  %81 = srem i32 %80, 128
  %82 = srem i32 %79, 32
  %83 = sext i32 %81 to i64
  %84 = sext i32 %82 to i64
  %85 = add i64 %46, %83
  %86 = trunc i64 %85 to i32
  %87 = mul i32 %86, 4
  %88 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %52, i32 %87, i32 0, i32 0)
  %89 = sext i32 %88 to i64
  %90 = mul i64 %89, 896
  %91 = add i64 %65, 2048
  %92 = trunc i64 %91 to i32
  %93 = sdiv i32 %92, 32
  %94 = srem i32 %93, 128
  %95 = srem i32 %92, 32
  %96 = sext i32 %94 to i64
  %97 = sext i32 %95 to i64
  %98 = add i64 %46, %96
  %99 = trunc i64 %98 to i32
  %100 = mul i32 %99, 4
  %101 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %52, i32 %100, i32 0, i32 0)
  %102 = sext i32 %101 to i64
  %103 = mul i64 %102, 896
  %104 = add i64 %65, 3072
  %105 = trunc i64 %104 to i32
  %106 = sdiv i32 %105, 32
  %107 = srem i32 %106, 128
  %108 = srem i32 %105, 32
  %109 = sext i32 %107 to i64
  %110 = sext i32 %108 to i64
  %111 = add i64 %46, %109
  %112 = trunc i64 %111 to i32
  %113 = mul i32 %112, 4
  %114 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %52, i32 %113, i32 0, i32 0)
  %115 = sext i32 %114 to i64
  %116 = mul i64 %115, 896
  %117 = add i64 %65, 4096
  %118 = trunc i64 %117 to i32
  %119 = sdiv i32 %118, 32
  %120 = srem i32 %119, 128
  %121 = srem i32 %118, 32
  %122 = sext i32 %120 to i64
  %123 = sext i32 %121 to i64
  %124 = add i64 %46, %122
  %125 = trunc i64 %124 to i32
  %126 = mul i32 %125, 4
  %127 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %52, i32 %126, i32 0, i32 0)
  %128 = sext i32 %127 to i64
  %129 = mul i64 %128, 896
  %130 = add i64 %65, 5120
  %131 = trunc i64 %130 to i32
  %132 = sdiv i32 %131, 32
  %133 = srem i32 %132, 128
  %134 = srem i32 %131, 32
  %135 = sext i32 %133 to i64
  %136 = sext i32 %134 to i64
  %137 = add i64 %46, %135
  %138 = trunc i64 %137 to i32
  %139 = mul i32 %138, 4
  %140 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %52, i32 %139, i32 0, i32 0)
  %141 = sext i32 %140 to i64
  %142 = mul i64 %141, 896
  %143 = add i64 %65, 6144
  %144 = trunc i64 %143 to i32
  %145 = sdiv i32 %144, 32
  %146 = srem i32 %145, 128
  %147 = srem i32 %144, 32
  %148 = sext i32 %146 to i64
  %149 = sext i32 %147 to i64
  %150 = add i64 %46, %148
  %151 = trunc i64 %150 to i32
  %152 = mul i32 %151, 4
  %153 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %52, i32 %152, i32 0, i32 0)
  %154 = sext i32 %153 to i64
  %155 = mul i64 %154, 896
  %156 = add i64 %65, 7168
  %157 = trunc i64 %156 to i32
  %158 = sdiv i32 %157, 32
  %159 = srem i32 %158, 128
  %160 = srem i32 %157, 32
  %161 = sext i32 %159 to i64
  %162 = sext i32 %160 to i64
  %163 = add i64 %46, %161
  %164 = trunc i64 %163 to i32
  %165 = mul i32 %164, 4
  %166 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %52, i32 %165, i32 0, i32 0)
  %167 = sext i32 %166 to i64
  %168 = mul i64 %167, 896
  %169 = mul i64 %64, 1024
  %170 = mul i64 %18, 256
  %171 = add i64 %169, %170
  %172 = mul i64 %20, 64
  %173 = add i64 %171, %172
  %174 = mul i64 %173, 3584
  %175 = trunc i64 %174 to i32
  %176 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %175)
  %177 = add i64 %173, 16
  %178 = mul i64 %177, 3584
  %179 = trunc i64 %178 to i32
  %180 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %179)
  %181 = add i64 %173, 32
  %182 = mul i64 %181, 3584
  %183 = trunc i64 %182 to i32
  %184 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %183)
  %185 = add i64 %173, 48
  %186 = mul i64 %185, 3584
  %187 = trunc i64 %186 to i32
  %188 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %187)
  %189 = mul i64 %23, 64
  %190 = mul i64 %24, 4
  %191 = add i64 %189, %190
  %192 = trunc i64 %191 to i32
  %193 = mul i64 %23, 16
  %194 = add i64 %193, %24
  %195 = udiv i64 %46, 32
  %196 = mul i64 %18, 8
  %197 = mul i64 %20, 2
  %198 = add i64 %196, %197
  %199 = mul i64 %64, 57344
  %200 = mul i64 %198, 1792
  %201 = add i64 %199, %200
  %202 = mul i64 %201, 4
  %203 = trunc i64 %202 to i32
  %204 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %203)
  %205 = add i64 %198, 1
  %206 = mul i64 %205, 1792
  %207 = add i64 %199, %206
  %208 = mul i64 %207, 4
  %209 = trunc i64 %208 to i32
  %210 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %209)
  %211 = trunc i64 %194 to i32
  %212 = ptrtoint ptr addrspace(1) %5 to i64
  %213 = inttoptr i64 %212 to ptr
  %214 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %213, i16 0, i64 4, i32 159744)
  %215 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %214, i32 0, i32 0, i32 0)
  %216 = trunc i64 %46 to i32
  %217 = icmp ult i32 %216, %215
  br i1 %217, label %218, label %8264

218:                                              ; preds = %12
  %219 = add i64 %172, %21
  %220 = mul i64 %219, 16
  %221 = trunc i64 %220 to i32
  %222 = mul i64 %219, 4
  %223 = trunc i64 %222 to i32
  %224 = mul i64 %195, 7168
  %225 = trunc i64 %224 to i32
  %226 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %225)
  %227 = mul i64 %20, 1024
  %228 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304) to i64), %227
  %229 = call i64 @llvm.amdgcn.readfirstlane.i64(i64 %228)
  %230 = inttoptr i64 %229 to ptr addrspace(3)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %60, ptr addrspace(3) %230, i32 16, i32 %221, i32 %226, i32 0, i32 0)
  %231 = add i32 %226, 4096
  %232 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %231)
  %233 = mul i64 %20, 256
  %234 = add i64 add (i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304) to i64), i64 4096), %233
  %235 = call i64 @llvm.amdgcn.readfirstlane.i64(i64 %234)
  %236 = inttoptr i64 %235 to ptr addrspace(3)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %60, ptr addrspace(3) %236, i32 4, i32 %223, i32 %232, i32 0, i32 0)
  %237 = add i32 %226, 5120
  %238 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %237)
  %239 = add i64 add (i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304) to i64), i64 5120), %233
  %240 = call i64 @llvm.amdgcn.readfirstlane.i64(i64 %239)
  %241 = inttoptr i64 %240 to ptr addrspace(3)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %60, ptr addrspace(3) %241, i32 4, i32 %223, i32 %238, i32 0, i32 0)
  %242 = add i32 %226, 6144
  %243 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %242)
  %244 = add i64 add (i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304) to i64), i64 6144), %233
  %245 = call i64 @llvm.amdgcn.readfirstlane.i64(i64 %244)
  %246 = inttoptr i64 %245 to ptr addrspace(3)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %60, ptr addrspace(3) %246, i32 4, i32 %223, i32 %243, i32 0, i32 0)
  %247 = add i64 %195, 1
  %248 = mul i64 %247, 7168
  %249 = trunc i64 %248 to i32
  %250 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %249)
  %251 = add i64 add (i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304) to i64), i64 7168), %227
  %252 = call i64 @llvm.amdgcn.readfirstlane.i64(i64 %251)
  %253 = inttoptr i64 %252 to ptr addrspace(3)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %60, ptr addrspace(3) %253, i32 16, i32 %221, i32 %250, i32 0, i32 0)
  %254 = add i32 %250, 4096
  %255 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %254)
  %256 = add i64 add (i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304) to i64), i64 11264), %233
  %257 = call i64 @llvm.amdgcn.readfirstlane.i64(i64 %256)
  %258 = inttoptr i64 %257 to ptr addrspace(3)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %60, ptr addrspace(3) %258, i32 4, i32 %223, i32 %255, i32 0, i32 0)
  %259 = add i32 %250, 5120
  %260 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %259)
  %261 = add i64 add (i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304) to i64), i64 12288), %233
  %262 = call i64 @llvm.amdgcn.readfirstlane.i64(i64 %261)
  %263 = inttoptr i64 %262 to ptr addrspace(3)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %60, ptr addrspace(3) %263, i32 4, i32 %223, i32 %260, i32 0, i32 0)
  %264 = add i32 %250, 6144
  %265 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %264)
  %266 = add i64 add (i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304) to i64), i64 13312), %233
  %267 = call i64 @llvm.amdgcn.readfirstlane.i64(i64 %266)
  %268 = inttoptr i64 %267 to ptr addrspace(3)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %60, ptr addrspace(3) %268, i32 4, i32 %223, i32 %265, i32 0, i32 0)
  %269 = add i64 %195, 2
  %270 = mul i64 %269, 7168
  %271 = trunc i64 %270 to i32
  %272 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %271)
  %273 = add i64 add (i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304) to i64), i64 14336), %227
  %274 = call i64 @llvm.amdgcn.readfirstlane.i64(i64 %273)
  %275 = inttoptr i64 %274 to ptr addrspace(3)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %60, ptr addrspace(3) %275, i32 16, i32 %221, i32 %272, i32 0, i32 0)
  %276 = add i32 %272, 4096
  %277 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %276)
  %278 = add i64 add (i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304) to i64), i64 18432), %233
  %279 = call i64 @llvm.amdgcn.readfirstlane.i64(i64 %278)
  %280 = inttoptr i64 %279 to ptr addrspace(3)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %60, ptr addrspace(3) %280, i32 4, i32 %223, i32 %277, i32 0, i32 0)
  %281 = add i32 %272, 5120
  %282 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %281)
  %283 = add i64 add (i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304) to i64), i64 19456), %233
  %284 = call i64 @llvm.amdgcn.readfirstlane.i64(i64 %283)
  %285 = inttoptr i64 %284 to ptr addrspace(3)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %60, ptr addrspace(3) %285, i32 4, i32 %223, i32 %282, i32 0, i32 0)
  %286 = add i32 %272, 6144
  %287 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %286)
  %288 = add i64 add (i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304) to i64), i64 20480), %233
  %289 = call i64 @llvm.amdgcn.readfirstlane.i64(i64 %288)
  %290 = inttoptr i64 %289 to ptr addrspace(3)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %60, ptr addrspace(3) %290, i32 4, i32 %223, i32 %287, i32 0, i32 0)
  %291 = add i64 %195, 3
  %292 = mul i64 %291, 7168
  %293 = trunc i64 %292 to i32
  %294 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %293)
  %295 = add i64 add (i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304) to i64), i64 21504), %227
  %296 = call i64 @llvm.amdgcn.readfirstlane.i64(i64 %295)
  %297 = inttoptr i64 %296 to ptr addrspace(3)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %60, ptr addrspace(3) %297, i32 16, i32 %221, i32 %294, i32 0, i32 0)
  %298 = add i32 %294, 4096
  %299 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %298)
  %300 = add i64 add (i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304) to i64), i64 25600), %233
  %301 = call i64 @llvm.amdgcn.readfirstlane.i64(i64 %300)
  %302 = inttoptr i64 %301 to ptr addrspace(3)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %60, ptr addrspace(3) %302, i32 4, i32 %223, i32 %299, i32 0, i32 0)
  %303 = add i32 %294, 5120
  %304 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %303)
  %305 = add i64 add (i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304) to i64), i64 26624), %233
  %306 = call i64 @llvm.amdgcn.readfirstlane.i64(i64 %305)
  %307 = inttoptr i64 %306 to ptr addrspace(3)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %60, ptr addrspace(3) %307, i32 4, i32 %223, i32 %304, i32 0, i32 0)
  %308 = add i32 %294, 6144
  %309 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %308)
  %310 = add i64 add (i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304) to i64), i64 27648), %233
  %311 = call i64 @llvm.amdgcn.readfirstlane.i64(i64 %310)
  %312 = inttoptr i64 %311 to ptr addrspace(3)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %60, ptr addrspace(3) %312, i32 4, i32 %223, i32 %309, i32 0, i32 0)
  %313 = add i64 %77, %71
  %314 = trunc i64 %313 to i32
  %315 = mul i32 %314, 4
  %316 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %315, i32 0, i32 0)
  %317 = bitcast <4 x i32> %316 to <16 x i8>
  %318 = add i64 %90, %84
  %319 = trunc i64 %318 to i32
  %320 = mul i32 %319, 4
  %321 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %320, i32 0, i32 0)
  %322 = bitcast <4 x i32> %321 to <16 x i8>
  %323 = add i64 %103, %97
  %324 = trunc i64 %323 to i32
  %325 = mul i32 %324, 4
  %326 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %325, i32 0, i32 0)
  %327 = bitcast <4 x i32> %326 to <16 x i8>
  %328 = add i64 %116, %110
  %329 = trunc i64 %328 to i32
  %330 = mul i32 %329, 4
  %331 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %330, i32 0, i32 0)
  %332 = bitcast <4 x i32> %331 to <16 x i8>
  %333 = add i64 %129, %123
  %334 = trunc i64 %333 to i32
  %335 = mul i32 %334, 4
  %336 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %335, i32 0, i32 0)
  %337 = bitcast <4 x i32> %336 to <16 x i8>
  %338 = add i64 %142, %136
  %339 = trunc i64 %338 to i32
  %340 = mul i32 %339, 4
  %341 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %340, i32 0, i32 0)
  %342 = bitcast <4 x i32> %341 to <16 x i8>
  %343 = add i64 %155, %149
  %344 = trunc i64 %343 to i32
  %345 = mul i32 %344, 4
  %346 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %345, i32 0, i32 0)
  %347 = bitcast <4 x i32> %346 to <16 x i8>
  %348 = add i64 %168, %162
  %349 = trunc i64 %348 to i32
  %350 = mul i32 %349, 4
  %351 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %350, i32 0, i32 0)
  %352 = bitcast <4 x i32> %351 to <16 x i8>
  %353 = mul i64 %71, 4
  %354 = and i64 %70, 15
  %355 = mul i64 %354, 16
  %356 = xor i64 %353, %355
  %357 = trunc i64 %356 to i32
  %358 = mul i32 %68, 256
  %359 = add i32 %358, %357
  %360 = sext i32 %359 to i64
  %361 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %360
  store <16 x i8> %317, ptr addrspace(3) %361, align 1
  %362 = mul i64 %84, 4
  %363 = and i64 %83, 15
  %364 = mul i64 %363, 16
  %365 = xor i64 %362, %364
  %366 = trunc i64 %365 to i32
  %367 = mul i32 %81, 256
  %368 = add i32 %367, %366
  %369 = sext i32 %368 to i64
  %370 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %369
  store <16 x i8> %322, ptr addrspace(3) %370, align 1
  %371 = mul i64 %97, 4
  %372 = and i64 %96, 15
  %373 = mul i64 %372, 16
  %374 = xor i64 %371, %373
  %375 = trunc i64 %374 to i32
  %376 = mul i32 %94, 256
  %377 = add i32 %376, %375
  %378 = sext i32 %377 to i64
  %379 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %378
  store <16 x i8> %327, ptr addrspace(3) %379, align 1
  %380 = mul i64 %110, 4
  %381 = and i64 %109, 15
  %382 = mul i64 %381, 16
  %383 = xor i64 %380, %382
  %384 = trunc i64 %383 to i32
  %385 = mul i32 %107, 256
  %386 = add i32 %385, %384
  %387 = sext i32 %386 to i64
  %388 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %387
  store <16 x i8> %332, ptr addrspace(3) %388, align 1
  %389 = mul i64 %123, 4
  %390 = and i64 %122, 15
  %391 = mul i64 %390, 16
  %392 = xor i64 %389, %391
  %393 = trunc i64 %392 to i32
  %394 = mul i32 %120, 256
  %395 = add i32 %394, %393
  %396 = sext i32 %395 to i64
  %397 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %396
  store <16 x i8> %337, ptr addrspace(3) %397, align 1
  %398 = mul i64 %136, 4
  %399 = and i64 %135, 15
  %400 = mul i64 %399, 16
  %401 = xor i64 %398, %400
  %402 = trunc i64 %401 to i32
  %403 = mul i32 %133, 256
  %404 = add i32 %403, %402
  %405 = sext i32 %404 to i64
  %406 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %405
  store <16 x i8> %342, ptr addrspace(3) %406, align 1
  %407 = mul i64 %149, 4
  %408 = and i64 %148, 15
  %409 = mul i64 %408, 16
  %410 = xor i64 %407, %409
  %411 = trunc i64 %410 to i32
  %412 = mul i32 %146, 256
  %413 = add i32 %412, %411
  %414 = sext i32 %413 to i64
  %415 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %414
  store <16 x i8> %347, ptr addrspace(3) %415, align 1
  %416 = mul i64 %162, 4
  %417 = and i64 %161, 15
  %418 = mul i64 %417, 16
  %419 = xor i64 %416, %418
  %420 = trunc i64 %419 to i32
  %421 = mul i32 %159, 256
  %422 = add i32 %421, %420
  %423 = sext i32 %422 to i64
  %424 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %423
  store <16 x i8> %352, ptr addrspace(3) %424, align 1
  %425 = mul i32 %192, 4
  %426 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %425, i32 %176, i32 2)
  %427 = add i32 %192, 256
  %428 = mul i32 %427, 4
  %429 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %428, i32 %176, i32 2)
  %430 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %425, i32 %180, i32 2)
  %431 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %428, i32 %180, i32 2)
  %432 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %425, i32 %184, i32 2)
  %433 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %428, i32 %184, i32 2)
  %434 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %425, i32 %188, i32 2)
  %435 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %428, i32 %188, i32 2)
  %436 = mul i32 %211, 4
  %437 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %436, i32 %204, i32 0)
  %438 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %436, i32 %210, i32 0)
  %439 = add i64 %77, 32
  %440 = add i64 %439, %71
  %441 = trunc i64 %440 to i32
  %442 = mul i32 %441, 4
  %443 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %442, i32 0, i32 0)
  %444 = bitcast <4 x i32> %443 to <16 x i8>
  %445 = add i64 %90, 32
  %446 = add i64 %445, %84
  %447 = trunc i64 %446 to i32
  %448 = mul i32 %447, 4
  %449 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %448, i32 0, i32 0)
  %450 = bitcast <4 x i32> %449 to <16 x i8>
  %451 = add i64 %103, 32
  %452 = add i64 %451, %97
  %453 = trunc i64 %452 to i32
  %454 = mul i32 %453, 4
  %455 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %454, i32 0, i32 0)
  %456 = bitcast <4 x i32> %455 to <16 x i8>
  %457 = add i64 %116, 32
  %458 = add i64 %457, %110
  %459 = trunc i64 %458 to i32
  %460 = mul i32 %459, 4
  %461 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %460, i32 0, i32 0)
  %462 = bitcast <4 x i32> %461 to <16 x i8>
  %463 = add i64 %129, 32
  %464 = add i64 %463, %123
  %465 = trunc i64 %464 to i32
  %466 = mul i32 %465, 4
  %467 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %466, i32 0, i32 0)
  %468 = bitcast <4 x i32> %467 to <16 x i8>
  %469 = add i64 %142, 32
  %470 = add i64 %469, %136
  %471 = trunc i64 %470 to i32
  %472 = mul i32 %471, 4
  %473 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %472, i32 0, i32 0)
  %474 = bitcast <4 x i32> %473 to <16 x i8>
  %475 = add i64 %155, 32
  %476 = add i64 %475, %149
  %477 = trunc i64 %476 to i32
  %478 = mul i32 %477, 4
  %479 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %478, i32 0, i32 0)
  %480 = bitcast <4 x i32> %479 to <16 x i8>
  %481 = add i64 %168, 32
  %482 = add i64 %481, %162
  %483 = trunc i64 %482 to i32
  %484 = mul i32 %483, 4
  %485 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %484, i32 0, i32 0)
  %486 = bitcast <4 x i32> %485 to <16 x i8>
  %487 = add i64 %360, 32768
  %488 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %487
  store <16 x i8> %444, ptr addrspace(3) %488, align 1
  %489 = add i64 %369, 32768
  %490 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %489
  store <16 x i8> %450, ptr addrspace(3) %490, align 1
  %491 = add i64 %378, 32768
  %492 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %491
  store <16 x i8> %456, ptr addrspace(3) %492, align 1
  %493 = add i64 %387, 32768
  %494 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %493
  store <16 x i8> %462, ptr addrspace(3) %494, align 1
  %495 = add i64 %396, 32768
  %496 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %495
  store <16 x i8> %468, ptr addrspace(3) %496, align 1
  %497 = add i64 %405, 32768
  %498 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %497
  store <16 x i8> %474, ptr addrspace(3) %498, align 1
  %499 = add i64 %414, 32768
  %500 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %499
  store <16 x i8> %480, ptr addrspace(3) %500, align 1
  %501 = add i64 %423, 32768
  %502 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %501
  store <16 x i8> %486, ptr addrspace(3) %502, align 1
  %503 = add i32 %192, 512
  %504 = mul i32 %503, 4
  %505 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %504, i32 %176, i32 2)
  %506 = add i32 %192, 768
  %507 = mul i32 %506, 4
  %508 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %507, i32 %176, i32 2)
  %509 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %504, i32 %180, i32 2)
  %510 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %507, i32 %180, i32 2)
  %511 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %504, i32 %184, i32 2)
  %512 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %507, i32 %184, i32 2)
  %513 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %504, i32 %188, i32 2)
  %514 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %507, i32 %188, i32 2)
  %515 = add i32 %211, 64
  %516 = mul i32 %515, 4
  %517 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %516, i32 %204, i32 0)
  %518 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %516, i32 %210, i32 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %519 = mul i64 %24, 16
  %520 = xor i64 %193, %519
  %521 = mul i64 %24, 256
  %522 = add i64 %521, %520
  %523 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %522
  %524 = load <16 x i8>, ptr addrspace(3) %523, align 1
  %525 = add i64 %193, 64
  %526 = xor i64 %525, %519
  %527 = add i64 %521, %526
  %528 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %527
  %529 = load <16 x i8>, ptr addrspace(3) %528, align 1
  %530 = add i64 %24, 16
  %531 = and i64 %530, 15
  %532 = mul i64 %531, 16
  %533 = xor i64 %193, %532
  %534 = mul i64 %530, 256
  %535 = add i64 %534, %533
  %536 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %535
  %537 = load <16 x i8>, ptr addrspace(3) %536, align 1
  %538 = xor i64 %525, %532
  %539 = add i64 %534, %538
  %540 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %539
  %541 = load <16 x i8>, ptr addrspace(3) %540, align 1
  %542 = add i64 %24, 32
  %543 = and i64 %542, 15
  %544 = mul i64 %543, 16
  %545 = xor i64 %193, %544
  %546 = mul i64 %542, 256
  %547 = add i64 %546, %545
  %548 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %547
  %549 = load <16 x i8>, ptr addrspace(3) %548, align 1
  %550 = xor i64 %525, %544
  %551 = add i64 %546, %550
  %552 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %551
  %553 = load <16 x i8>, ptr addrspace(3) %552, align 1
  %554 = add i64 %24, 48
  %555 = and i64 %554, 15
  %556 = mul i64 %555, 16
  %557 = xor i64 %193, %556
  %558 = mul i64 %554, 256
  %559 = add i64 %558, %557
  %560 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %559
  %561 = load <16 x i8>, ptr addrspace(3) %560, align 1
  %562 = xor i64 %525, %556
  %563 = add i64 %558, %562
  %564 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %563
  %565 = load <16 x i8>, ptr addrspace(3) %564, align 1
  %566 = add i64 %24, 64
  %567 = and i64 %566, 15
  %568 = mul i64 %567, 16
  %569 = xor i64 %193, %568
  %570 = mul i64 %566, 256
  %571 = add i64 %570, %569
  %572 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %571
  %573 = load <16 x i8>, ptr addrspace(3) %572, align 1
  %574 = xor i64 %525, %568
  %575 = add i64 %570, %574
  %576 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %575
  %577 = load <16 x i8>, ptr addrspace(3) %576, align 1
  %578 = add i64 %24, 80
  %579 = and i64 %578, 15
  %580 = mul i64 %579, 16
  %581 = xor i64 %193, %580
  %582 = mul i64 %578, 256
  %583 = add i64 %582, %581
  %584 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %583
  %585 = load <16 x i8>, ptr addrspace(3) %584, align 1
  %586 = xor i64 %525, %580
  %587 = add i64 %582, %586
  %588 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %587
  %589 = load <16 x i8>, ptr addrspace(3) %588, align 1
  %590 = add i64 %24, 96
  %591 = and i64 %590, 15
  %592 = mul i64 %591, 16
  %593 = xor i64 %193, %592
  %594 = mul i64 %590, 256
  %595 = add i64 %594, %593
  %596 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %595
  %597 = load <16 x i8>, ptr addrspace(3) %596, align 1
  %598 = xor i64 %525, %592
  %599 = add i64 %594, %598
  %600 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %599
  %601 = load <16 x i8>, ptr addrspace(3) %600, align 1
  %602 = add i64 %24, 112
  %603 = and i64 %602, 15
  %604 = mul i64 %603, 16
  %605 = xor i64 %193, %604
  %606 = mul i64 %602, 256
  %607 = add i64 %606, %605
  %608 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %607
  %609 = load <16 x i8>, ptr addrspace(3) %608, align 1
  %610 = xor i64 %525, %604
  %611 = add i64 %606, %610
  %612 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %611
  %613 = load <16 x i8>, ptr addrspace(3) %612, align 1
  %614 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %194
  %615 = load <1 x i32>, ptr addrspace(3) %614, align 4
  %616 = extractelement <1 x i32> %615, i64 0
  %617 = add i64 %193, 1792
  %618 = add i64 %617, %24
  %619 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %618
  %620 = load <1 x i32>, ptr addrspace(3) %619, align 4
  %621 = extractelement <1 x i32> %620, i64 0
  %622 = add i64 %193, 3584
  %623 = add i64 %622, %24
  %624 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %623
  %625 = load <1 x i32>, ptr addrspace(3) %624, align 4
  %626 = extractelement <1 x i32> %625, i64 0
  %627 = add i64 %193, 5376
  %628 = add i64 %627, %24
  %629 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %628
  %630 = load <1 x i32>, ptr addrspace(3) %629, align 4
  %631 = extractelement <1 x i32> %630, i64 0
  %632 = add i64 %77, 64
  %633 = add i64 %632, %71
  %634 = trunc i64 %633 to i32
  %635 = mul i32 %634, 4
  %636 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %635, i32 0, i32 0)
  %637 = bitcast <4 x i32> %636 to <16 x i8>
  %638 = add i64 %90, 64
  %639 = add i64 %638, %84
  %640 = trunc i64 %639 to i32
  %641 = mul i32 %640, 4
  %642 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %641, i32 0, i32 0)
  %643 = bitcast <4 x i32> %642 to <16 x i8>
  %644 = add i64 %103, 64
  %645 = add i64 %644, %97
  %646 = trunc i64 %645 to i32
  %647 = mul i32 %646, 4
  %648 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %647, i32 0, i32 0)
  %649 = bitcast <4 x i32> %648 to <16 x i8>
  %650 = add i64 %116, 64
  %651 = add i64 %650, %110
  %652 = trunc i64 %651 to i32
  %653 = mul i32 %652, 4
  %654 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %653, i32 0, i32 0)
  %655 = bitcast <4 x i32> %654 to <16 x i8>
  %656 = add i64 %129, 64
  %657 = add i64 %656, %123
  %658 = trunc i64 %657 to i32
  %659 = mul i32 %658, 4
  %660 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %659, i32 0, i32 0)
  %661 = bitcast <4 x i32> %660 to <16 x i8>
  %662 = add i64 %142, 64
  %663 = add i64 %662, %136
  %664 = trunc i64 %663 to i32
  %665 = mul i32 %664, 4
  %666 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %665, i32 0, i32 0)
  %667 = bitcast <4 x i32> %666 to <16 x i8>
  %668 = add i64 %155, 64
  %669 = add i64 %668, %149
  %670 = trunc i64 %669 to i32
  %671 = mul i32 %670, 4
  %672 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %671, i32 0, i32 0)
  %673 = bitcast <4 x i32> %672 to <16 x i8>
  %674 = add i64 %168, 64
  %675 = add i64 %674, %162
  %676 = trunc i64 %675 to i32
  %677 = mul i32 %676, 4
  %678 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %677, i32 0, i32 0)
  %679 = bitcast <4 x i32> %678 to <16 x i8>
  %680 = add i64 %360, 65536
  %681 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %680
  store <16 x i8> %637, ptr addrspace(3) %681, align 1
  %682 = add i64 %369, 65536
  %683 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %682
  store <16 x i8> %643, ptr addrspace(3) %683, align 1
  %684 = add i64 %378, 65536
  %685 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %684
  store <16 x i8> %649, ptr addrspace(3) %685, align 1
  %686 = add i64 %387, 65536
  %687 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %686
  store <16 x i8> %655, ptr addrspace(3) %687, align 1
  %688 = add i64 %396, 65536
  %689 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %688
  store <16 x i8> %661, ptr addrspace(3) %689, align 1
  %690 = add i64 %405, 65536
  %691 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %690
  store <16 x i8> %667, ptr addrspace(3) %691, align 1
  %692 = add i64 %414, 65536
  %693 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %692
  store <16 x i8> %673, ptr addrspace(3) %693, align 1
  %694 = add i64 %423, 65536
  %695 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %694
  store <16 x i8> %679, ptr addrspace(3) %695, align 1
  %696 = bitcast <16 x i8> %524 to <4 x i32>
  %697 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %696, <4 x i32> %426, i32 %616, i32 %437)
  %698 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> zeroinitializer, <4 x i32> %696, <4 x i32> %426, i32 %616, i32 %437)
  %699 = bitcast <16 x i8> %529 to <4 x i32>
  %700 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %699, <4 x i32> %429, i32 %616, i32 %437)
  %701 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %698, <4 x i32> %699, <4 x i32> %429, i32 %616, i32 %437)
  %702 = bitcast <16 x i8> %537 to <4 x i32>
  %703 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %702, <4 x i32> %426, i32 %616, i32 %437)
  %704 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> zeroinitializer, <4 x i32> %702, <4 x i32> %426, i32 %616, i32 %437)
  %705 = bitcast <16 x i8> %541 to <4 x i32>
  %706 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %705, <4 x i32> %429, i32 %616, i32 %437)
  %707 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %704, <4 x i32> %705, <4 x i32> %429, i32 %616, i32 %437)
  %708 = bitcast <16 x i8> %549 to <4 x i32>
  %709 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %708, <4 x i32> %426, i32 %621, i32 %437)
  %710 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> zeroinitializer, <4 x i32> %708, <4 x i32> %426, i32 %621, i32 %437)
  %711 = bitcast <16 x i8> %553 to <4 x i32>
  %712 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %711, <4 x i32> %429, i32 %621, i32 %437)
  %713 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %710, <4 x i32> %711, <4 x i32> %429, i32 %621, i32 %437)
  %714 = bitcast <16 x i8> %561 to <4 x i32>
  %715 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %714, <4 x i32> %426, i32 %621, i32 %437)
  %716 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> zeroinitializer, <4 x i32> %714, <4 x i32> %426, i32 %621, i32 %437)
  %717 = bitcast <16 x i8> %565 to <4 x i32>
  %718 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %717, <4 x i32> %429, i32 %621, i32 %437)
  %719 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %716, <4 x i32> %717, <4 x i32> %429, i32 %621, i32 %437)
  %720 = bitcast <16 x i8> %573 to <4 x i32>
  %721 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %720, <4 x i32> %426, i32 %626, i32 %437)
  %722 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> zeroinitializer, <4 x i32> %720, <4 x i32> %426, i32 %626, i32 %437)
  %723 = bitcast <16 x i8> %577 to <4 x i32>
  %724 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %723, <4 x i32> %429, i32 %626, i32 %437)
  %725 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %722, <4 x i32> %723, <4 x i32> %429, i32 %626, i32 %437)
  %726 = bitcast <16 x i8> %585 to <4 x i32>
  %727 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %726, <4 x i32> %426, i32 %626, i32 %437)
  %728 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> zeroinitializer, <4 x i32> %726, <4 x i32> %426, i32 %626, i32 %437)
  %729 = bitcast <16 x i8> %589 to <4 x i32>
  %730 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %729, <4 x i32> %429, i32 %626, i32 %437)
  %731 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %728, <4 x i32> %729, <4 x i32> %429, i32 %626, i32 %437)
  %732 = bitcast <16 x i8> %597 to <4 x i32>
  %733 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %732, <4 x i32> %426, i32 %631, i32 %437)
  %734 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> zeroinitializer, <4 x i32> %732, <4 x i32> %426, i32 %631, i32 %437)
  %735 = bitcast <16 x i8> %601 to <4 x i32>
  %736 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %735, <4 x i32> %429, i32 %631, i32 %437)
  %737 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %734, <4 x i32> %735, <4 x i32> %429, i32 %631, i32 %437)
  %738 = bitcast <16 x i8> %609 to <4 x i32>
  %739 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %738, <4 x i32> %426, i32 %631, i32 %437)
  %740 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> zeroinitializer, <4 x i32> %738, <4 x i32> %426, i32 %631, i32 %437)
  %741 = bitcast <16 x i8> %613 to <4 x i32>
  %742 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %741, <4 x i32> %429, i32 %631, i32 %437)
  %743 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %740, <4 x i32> %741, <4 x i32> %429, i32 %631, i32 %437)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %744 = add i32 %192, 1024
  %745 = mul i32 %744, 4
  %746 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %745, i32 %176, i32 2)
  %747 = add i32 %192, 1280
  %748 = mul i32 %747, 4
  %749 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %748, i32 %176, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %750 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %696, <4 x i32> %430, i32 %616, i32 %437)
  %751 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> zeroinitializer, <4 x i32> %696, <4 x i32> %430, i32 %616, i32 %437)
  %752 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %699, <4 x i32> %431, i32 %616, i32 %437)
  %753 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %751, <4 x i32> %699, <4 x i32> %431, i32 %616, i32 %437)
  %754 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %702, <4 x i32> %430, i32 %616, i32 %437)
  %755 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> zeroinitializer, <4 x i32> %702, <4 x i32> %430, i32 %616, i32 %437)
  %756 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %705, <4 x i32> %431, i32 %616, i32 %437)
  %757 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %755, <4 x i32> %705, <4 x i32> %431, i32 %616, i32 %437)
  %758 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %708, <4 x i32> %430, i32 %621, i32 %437)
  %759 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> zeroinitializer, <4 x i32> %708, <4 x i32> %430, i32 %621, i32 %437)
  %760 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %711, <4 x i32> %431, i32 %621, i32 %437)
  %761 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %759, <4 x i32> %711, <4 x i32> %431, i32 %621, i32 %437)
  %762 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %714, <4 x i32> %430, i32 %621, i32 %437)
  %763 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> zeroinitializer, <4 x i32> %714, <4 x i32> %430, i32 %621, i32 %437)
  %764 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %717, <4 x i32> %431, i32 %621, i32 %437)
  %765 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %763, <4 x i32> %717, <4 x i32> %431, i32 %621, i32 %437)
  %766 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %720, <4 x i32> %430, i32 %626, i32 %437)
  %767 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> zeroinitializer, <4 x i32> %720, <4 x i32> %430, i32 %626, i32 %437)
  %768 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %723, <4 x i32> %431, i32 %626, i32 %437)
  %769 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %767, <4 x i32> %723, <4 x i32> %431, i32 %626, i32 %437)
  %770 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %726, <4 x i32> %430, i32 %626, i32 %437)
  %771 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> zeroinitializer, <4 x i32> %726, <4 x i32> %430, i32 %626, i32 %437)
  %772 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %729, <4 x i32> %431, i32 %626, i32 %437)
  %773 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %771, <4 x i32> %729, <4 x i32> %431, i32 %626, i32 %437)
  %774 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %732, <4 x i32> %430, i32 %631, i32 %437)
  %775 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> zeroinitializer, <4 x i32> %732, <4 x i32> %430, i32 %631, i32 %437)
  %776 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %735, <4 x i32> %431, i32 %631, i32 %437)
  %777 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %775, <4 x i32> %735, <4 x i32> %431, i32 %631, i32 %437)
  %778 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %738, <4 x i32> %430, i32 %631, i32 %437)
  %779 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> zeroinitializer, <4 x i32> %738, <4 x i32> %430, i32 %631, i32 %437)
  %780 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %741, <4 x i32> %431, i32 %631, i32 %437)
  %781 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %779, <4 x i32> %741, <4 x i32> %431, i32 %631, i32 %437)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %782 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %745, i32 %180, i32 2)
  %783 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %748, i32 %180, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %784 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %696, <4 x i32> %432, i32 %616, i32 %438)
  %785 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> zeroinitializer, <4 x i32> %696, <4 x i32> %432, i32 %616, i32 %438)
  %786 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %699, <4 x i32> %433, i32 %616, i32 %438)
  %787 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %785, <4 x i32> %699, <4 x i32> %433, i32 %616, i32 %438)
  %788 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %702, <4 x i32> %432, i32 %616, i32 %438)
  %789 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> zeroinitializer, <4 x i32> %702, <4 x i32> %432, i32 %616, i32 %438)
  %790 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %705, <4 x i32> %433, i32 %616, i32 %438)
  %791 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %789, <4 x i32> %705, <4 x i32> %433, i32 %616, i32 %438)
  %792 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %708, <4 x i32> %432, i32 %621, i32 %438)
  %793 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> zeroinitializer, <4 x i32> %708, <4 x i32> %432, i32 %621, i32 %438)
  %794 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %711, <4 x i32> %433, i32 %621, i32 %438)
  %795 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %793, <4 x i32> %711, <4 x i32> %433, i32 %621, i32 %438)
  %796 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %714, <4 x i32> %432, i32 %621, i32 %438)
  %797 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> zeroinitializer, <4 x i32> %714, <4 x i32> %432, i32 %621, i32 %438)
  %798 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %717, <4 x i32> %433, i32 %621, i32 %438)
  %799 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %797, <4 x i32> %717, <4 x i32> %433, i32 %621, i32 %438)
  %800 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %720, <4 x i32> %432, i32 %626, i32 %438)
  %801 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> zeroinitializer, <4 x i32> %720, <4 x i32> %432, i32 %626, i32 %438)
  %802 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %723, <4 x i32> %433, i32 %626, i32 %438)
  %803 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %801, <4 x i32> %723, <4 x i32> %433, i32 %626, i32 %438)
  %804 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %726, <4 x i32> %432, i32 %626, i32 %438)
  %805 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> zeroinitializer, <4 x i32> %726, <4 x i32> %432, i32 %626, i32 %438)
  %806 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %729, <4 x i32> %433, i32 %626, i32 %438)
  %807 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %805, <4 x i32> %729, <4 x i32> %433, i32 %626, i32 %438)
  %808 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %732, <4 x i32> %432, i32 %631, i32 %438)
  %809 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> zeroinitializer, <4 x i32> %732, <4 x i32> %432, i32 %631, i32 %438)
  %810 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %735, <4 x i32> %433, i32 %631, i32 %438)
  %811 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %809, <4 x i32> %735, <4 x i32> %433, i32 %631, i32 %438)
  %812 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %738, <4 x i32> %432, i32 %631, i32 %438)
  %813 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> zeroinitializer, <4 x i32> %738, <4 x i32> %432, i32 %631, i32 %438)
  %814 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %741, <4 x i32> %433, i32 %631, i32 %438)
  %815 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %813, <4 x i32> %741, <4 x i32> %433, i32 %631, i32 %438)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %816 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %745, i32 %184, i32 2)
  %817 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %748, i32 %184, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %818 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %696, <4 x i32> %434, i32 %616, i32 %438)
  %819 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> zeroinitializer, <4 x i32> %696, <4 x i32> %434, i32 %616, i32 %438)
  %820 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %699, <4 x i32> %435, i32 %616, i32 %438)
  %821 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %819, <4 x i32> %699, <4 x i32> %435, i32 %616, i32 %438)
  %822 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %702, <4 x i32> %434, i32 %616, i32 %438)
  %823 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> zeroinitializer, <4 x i32> %702, <4 x i32> %434, i32 %616, i32 %438)
  %824 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %705, <4 x i32> %435, i32 %616, i32 %438)
  %825 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %823, <4 x i32> %705, <4 x i32> %435, i32 %616, i32 %438)
  %826 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %708, <4 x i32> %434, i32 %621, i32 %438)
  %827 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> zeroinitializer, <4 x i32> %708, <4 x i32> %434, i32 %621, i32 %438)
  %828 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %711, <4 x i32> %435, i32 %621, i32 %438)
  %829 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %827, <4 x i32> %711, <4 x i32> %435, i32 %621, i32 %438)
  %830 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %714, <4 x i32> %434, i32 %621, i32 %438)
  %831 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> zeroinitializer, <4 x i32> %714, <4 x i32> %434, i32 %621, i32 %438)
  %832 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %717, <4 x i32> %435, i32 %621, i32 %438)
  %833 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %831, <4 x i32> %717, <4 x i32> %435, i32 %621, i32 %438)
  %834 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %720, <4 x i32> %434, i32 %626, i32 %438)
  %835 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> zeroinitializer, <4 x i32> %720, <4 x i32> %434, i32 %626, i32 %438)
  %836 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %723, <4 x i32> %435, i32 %626, i32 %438)
  %837 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %835, <4 x i32> %723, <4 x i32> %435, i32 %626, i32 %438)
  %838 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %726, <4 x i32> %434, i32 %626, i32 %438)
  %839 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> zeroinitializer, <4 x i32> %726, <4 x i32> %434, i32 %626, i32 %438)
  %840 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %729, <4 x i32> %435, i32 %626, i32 %438)
  %841 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %839, <4 x i32> %729, <4 x i32> %435, i32 %626, i32 %438)
  %842 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %732, <4 x i32> %434, i32 %631, i32 %438)
  %843 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> zeroinitializer, <4 x i32> %732, <4 x i32> %434, i32 %631, i32 %438)
  %844 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %735, <4 x i32> %435, i32 %631, i32 %438)
  %845 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %843, <4 x i32> %735, <4 x i32> %435, i32 %631, i32 %438)
  %846 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %738, <4 x i32> %434, i32 %631, i32 %438)
  %847 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> zeroinitializer, <4 x i32> %738, <4 x i32> %434, i32 %631, i32 %438)
  %848 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,v,v,v,v"(<4 x i32> %741, <4 x i32> %435, i32 %631, i32 %438)
  %849 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %847, <4 x i32> %741, <4 x i32> %435, i32 %631, i32 %438)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %850 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %745, i32 %188, i32 2)
  %851 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %748, i32 %188, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %852 = add i32 %211, 128
  %853 = mul i32 %852, 4
  %854 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %853, i32 %204, i32 0)
  %855 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %853, i32 %210, i32 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %856 = add i64 %522, 32768
  %857 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %856
  %858 = load <16 x i8>, ptr addrspace(3) %857, align 1
  %859 = add i64 %527, 32768
  %860 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %859
  %861 = load <16 x i8>, ptr addrspace(3) %860, align 1
  %862 = add i64 %535, 32768
  %863 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %862
  %864 = load <16 x i8>, ptr addrspace(3) %863, align 1
  %865 = add i64 %539, 32768
  %866 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %865
  %867 = load <16 x i8>, ptr addrspace(3) %866, align 1
  %868 = add i64 %547, 32768
  %869 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %868
  %870 = load <16 x i8>, ptr addrspace(3) %869, align 1
  %871 = add i64 %551, 32768
  %872 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %871
  %873 = load <16 x i8>, ptr addrspace(3) %872, align 1
  %874 = add i64 %559, 32768
  %875 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %874
  %876 = load <16 x i8>, ptr addrspace(3) %875, align 1
  %877 = add i64 %563, 32768
  %878 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %877
  %879 = load <16 x i8>, ptr addrspace(3) %878, align 1
  %880 = add i64 %571, 32768
  %881 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %880
  %882 = load <16 x i8>, ptr addrspace(3) %881, align 1
  %883 = add i64 %575, 32768
  %884 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %883
  %885 = load <16 x i8>, ptr addrspace(3) %884, align 1
  %886 = add i64 %583, 32768
  %887 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %886
  %888 = load <16 x i8>, ptr addrspace(3) %887, align 1
  %889 = add i64 %587, 32768
  %890 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %889
  %891 = load <16 x i8>, ptr addrspace(3) %890, align 1
  %892 = add i64 %595, 32768
  %893 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %892
  %894 = load <16 x i8>, ptr addrspace(3) %893, align 1
  %895 = add i64 %599, 32768
  %896 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %895
  %897 = load <16 x i8>, ptr addrspace(3) %896, align 1
  %898 = add i64 %607, 32768
  %899 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %898
  %900 = load <16 x i8>, ptr addrspace(3) %899, align 1
  %901 = add i64 %611, 32768
  %902 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %901
  %903 = load <16 x i8>, ptr addrspace(3) %902, align 1
  %904 = add i64 %525, %24
  %905 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %904
  %906 = load <1 x i32>, ptr addrspace(3) %905, align 4
  %907 = extractelement <1 x i32> %906, i64 0
  %908 = add i64 %193, 1856
  %909 = add i64 %908, %24
  %910 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %909
  %911 = load <1 x i32>, ptr addrspace(3) %910, align 4
  %912 = extractelement <1 x i32> %911, i64 0
  %913 = add i64 %193, 3648
  %914 = add i64 %913, %24
  %915 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %914
  %916 = load <1 x i32>, ptr addrspace(3) %915, align 4
  %917 = extractelement <1 x i32> %916, i64 0
  %918 = add i64 %193, 5440
  %919 = add i64 %918, %24
  %920 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %919
  %921 = load <1 x i32>, ptr addrspace(3) %920, align 4
  %922 = extractelement <1 x i32> %921, i64 0
  %923 = add i64 %77, 96
  %924 = add i64 %923, %71
  %925 = trunc i64 %924 to i32
  %926 = mul i32 %925, 4
  %927 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %926, i32 0, i32 0)
  %928 = bitcast <4 x i32> %927 to <16 x i8>
  %929 = add i64 %90, 96
  %930 = add i64 %929, %84
  %931 = trunc i64 %930 to i32
  %932 = mul i32 %931, 4
  %933 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %932, i32 0, i32 0)
  %934 = bitcast <4 x i32> %933 to <16 x i8>
  %935 = add i64 %103, 96
  %936 = add i64 %935, %97
  %937 = trunc i64 %936 to i32
  %938 = mul i32 %937, 4
  %939 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %938, i32 0, i32 0)
  %940 = bitcast <4 x i32> %939 to <16 x i8>
  %941 = add i64 %116, 96
  %942 = add i64 %941, %110
  %943 = trunc i64 %942 to i32
  %944 = mul i32 %943, 4
  %945 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %944, i32 0, i32 0)
  %946 = bitcast <4 x i32> %945 to <16 x i8>
  %947 = add i64 %129, 96
  %948 = add i64 %947, %123
  %949 = trunc i64 %948 to i32
  %950 = mul i32 %949, 4
  %951 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %950, i32 0, i32 0)
  %952 = bitcast <4 x i32> %951 to <16 x i8>
  %953 = add i64 %142, 96
  %954 = add i64 %953, %136
  %955 = trunc i64 %954 to i32
  %956 = mul i32 %955, 4
  %957 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %956, i32 0, i32 0)
  %958 = bitcast <4 x i32> %957 to <16 x i8>
  %959 = add i64 %155, 96
  %960 = add i64 %959, %149
  %961 = trunc i64 %960 to i32
  %962 = mul i32 %961, 4
  %963 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %962, i32 0, i32 0)
  %964 = bitcast <4 x i32> %963 to <16 x i8>
  %965 = add i64 %168, 96
  %966 = add i64 %965, %162
  %967 = trunc i64 %966 to i32
  %968 = mul i32 %967, 4
  %969 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %968, i32 0, i32 0)
  %970 = bitcast <4 x i32> %969 to <16 x i8>
  %971 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %360
  store <16 x i8> %928, ptr addrspace(3) %971, align 1
  %972 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %369
  store <16 x i8> %934, ptr addrspace(3) %972, align 1
  %973 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %378
  store <16 x i8> %940, ptr addrspace(3) %973, align 1
  %974 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %387
  store <16 x i8> %946, ptr addrspace(3) %974, align 1
  %975 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %396
  store <16 x i8> %952, ptr addrspace(3) %975, align 1
  %976 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %405
  store <16 x i8> %958, ptr addrspace(3) %976, align 1
  %977 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %414
  store <16 x i8> %964, ptr addrspace(3) %977, align 1
  %978 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %423
  store <16 x i8> %970, ptr addrspace(3) %978, align 1
  %979 = bitcast <16 x i8> %858 to <4 x i32>
  %980 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %701, <4 x i32> %979, <4 x i32> %505, i32 %907, i32 %517)
  %981 = bitcast <16 x i8> %861 to <4 x i32>
  %982 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %980, <4 x i32> %981, <4 x i32> %508, i32 %907, i32 %517)
  %983 = bitcast <16 x i8> %864 to <4 x i32>
  %984 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %707, <4 x i32> %983, <4 x i32> %505, i32 %907, i32 %517)
  %985 = bitcast <16 x i8> %867 to <4 x i32>
  %986 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %984, <4 x i32> %985, <4 x i32> %508, i32 %907, i32 %517)
  %987 = bitcast <16 x i8> %870 to <4 x i32>
  %988 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %713, <4 x i32> %987, <4 x i32> %505, i32 %912, i32 %517)
  %989 = bitcast <16 x i8> %873 to <4 x i32>
  %990 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %988, <4 x i32> %989, <4 x i32> %508, i32 %912, i32 %517)
  %991 = bitcast <16 x i8> %876 to <4 x i32>
  %992 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %719, <4 x i32> %991, <4 x i32> %505, i32 %912, i32 %517)
  %993 = bitcast <16 x i8> %879 to <4 x i32>
  %994 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %992, <4 x i32> %993, <4 x i32> %508, i32 %912, i32 %517)
  %995 = bitcast <16 x i8> %882 to <4 x i32>
  %996 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %725, <4 x i32> %995, <4 x i32> %505, i32 %917, i32 %517)
  %997 = bitcast <16 x i8> %885 to <4 x i32>
  %998 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %996, <4 x i32> %997, <4 x i32> %508, i32 %917, i32 %517)
  %999 = bitcast <16 x i8> %888 to <4 x i32>
  %1000 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %731, <4 x i32> %999, <4 x i32> %505, i32 %917, i32 %517)
  %1001 = bitcast <16 x i8> %891 to <4 x i32>
  %1002 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1000, <4 x i32> %1001, <4 x i32> %508, i32 %917, i32 %517)
  %1003 = bitcast <16 x i8> %894 to <4 x i32>
  %1004 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %737, <4 x i32> %1003, <4 x i32> %505, i32 %922, i32 %517)
  %1005 = bitcast <16 x i8> %897 to <4 x i32>
  %1006 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1004, <4 x i32> %1005, <4 x i32> %508, i32 %922, i32 %517)
  %1007 = bitcast <16 x i8> %900 to <4 x i32>
  %1008 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %743, <4 x i32> %1007, <4 x i32> %505, i32 %922, i32 %517)
  %1009 = bitcast <16 x i8> %903 to <4 x i32>
  %1010 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1008, <4 x i32> %1009, <4 x i32> %508, i32 %922, i32 %517)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1011 = add i32 %192, 1536
  %1012 = mul i32 %1011, 4
  %1013 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1012, i32 %176, i32 2)
  %1014 = add i32 %192, 1792
  %1015 = mul i32 %1014, 4
  %1016 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1015, i32 %176, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1017 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %753, <4 x i32> %979, <4 x i32> %509, i32 %907, i32 %517)
  %1018 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1017, <4 x i32> %981, <4 x i32> %510, i32 %907, i32 %517)
  %1019 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %757, <4 x i32> %983, <4 x i32> %509, i32 %907, i32 %517)
  %1020 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1019, <4 x i32> %985, <4 x i32> %510, i32 %907, i32 %517)
  %1021 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %761, <4 x i32> %987, <4 x i32> %509, i32 %912, i32 %517)
  %1022 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1021, <4 x i32> %989, <4 x i32> %510, i32 %912, i32 %517)
  %1023 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %765, <4 x i32> %991, <4 x i32> %509, i32 %912, i32 %517)
  %1024 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1023, <4 x i32> %993, <4 x i32> %510, i32 %912, i32 %517)
  %1025 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %769, <4 x i32> %995, <4 x i32> %509, i32 %917, i32 %517)
  %1026 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1025, <4 x i32> %997, <4 x i32> %510, i32 %917, i32 %517)
  %1027 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %773, <4 x i32> %999, <4 x i32> %509, i32 %917, i32 %517)
  %1028 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1027, <4 x i32> %1001, <4 x i32> %510, i32 %917, i32 %517)
  %1029 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %777, <4 x i32> %1003, <4 x i32> %509, i32 %922, i32 %517)
  %1030 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1029, <4 x i32> %1005, <4 x i32> %510, i32 %922, i32 %517)
  %1031 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %781, <4 x i32> %1007, <4 x i32> %509, i32 %922, i32 %517)
  %1032 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1031, <4 x i32> %1009, <4 x i32> %510, i32 %922, i32 %517)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1033 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1012, i32 %180, i32 2)
  %1034 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1015, i32 %180, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1035 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %787, <4 x i32> %979, <4 x i32> %511, i32 %907, i32 %518)
  %1036 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1035, <4 x i32> %981, <4 x i32> %512, i32 %907, i32 %518)
  %1037 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %791, <4 x i32> %983, <4 x i32> %511, i32 %907, i32 %518)
  %1038 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1037, <4 x i32> %985, <4 x i32> %512, i32 %907, i32 %518)
  %1039 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %795, <4 x i32> %987, <4 x i32> %511, i32 %912, i32 %518)
  %1040 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1039, <4 x i32> %989, <4 x i32> %512, i32 %912, i32 %518)
  %1041 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %799, <4 x i32> %991, <4 x i32> %511, i32 %912, i32 %518)
  %1042 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1041, <4 x i32> %993, <4 x i32> %512, i32 %912, i32 %518)
  %1043 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %803, <4 x i32> %995, <4 x i32> %511, i32 %917, i32 %518)
  %1044 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1043, <4 x i32> %997, <4 x i32> %512, i32 %917, i32 %518)
  %1045 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %807, <4 x i32> %999, <4 x i32> %511, i32 %917, i32 %518)
  %1046 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1045, <4 x i32> %1001, <4 x i32> %512, i32 %917, i32 %518)
  %1047 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %811, <4 x i32> %1003, <4 x i32> %511, i32 %922, i32 %518)
  %1048 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1047, <4 x i32> %1005, <4 x i32> %512, i32 %922, i32 %518)
  %1049 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %815, <4 x i32> %1007, <4 x i32> %511, i32 %922, i32 %518)
  %1050 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1049, <4 x i32> %1009, <4 x i32> %512, i32 %922, i32 %518)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1051 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1012, i32 %184, i32 2)
  %1052 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1015, i32 %184, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1053 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %821, <4 x i32> %979, <4 x i32> %513, i32 %907, i32 %518)
  %1054 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1053, <4 x i32> %981, <4 x i32> %514, i32 %907, i32 %518)
  %1055 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %825, <4 x i32> %983, <4 x i32> %513, i32 %907, i32 %518)
  %1056 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1055, <4 x i32> %985, <4 x i32> %514, i32 %907, i32 %518)
  %1057 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %829, <4 x i32> %987, <4 x i32> %513, i32 %912, i32 %518)
  %1058 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1057, <4 x i32> %989, <4 x i32> %514, i32 %912, i32 %518)
  %1059 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %833, <4 x i32> %991, <4 x i32> %513, i32 %912, i32 %518)
  %1060 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1059, <4 x i32> %993, <4 x i32> %514, i32 %912, i32 %518)
  %1061 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %837, <4 x i32> %995, <4 x i32> %513, i32 %917, i32 %518)
  %1062 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1061, <4 x i32> %997, <4 x i32> %514, i32 %917, i32 %518)
  %1063 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %841, <4 x i32> %999, <4 x i32> %513, i32 %917, i32 %518)
  %1064 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1063, <4 x i32> %1001, <4 x i32> %514, i32 %917, i32 %518)
  %1065 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %845, <4 x i32> %1003, <4 x i32> %513, i32 %922, i32 %518)
  %1066 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1065, <4 x i32> %1005, <4 x i32> %514, i32 %922, i32 %518)
  %1067 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %849, <4 x i32> %1007, <4 x i32> %513, i32 %922, i32 %518)
  %1068 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1067, <4 x i32> %1009, <4 x i32> %514, i32 %922, i32 %518)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1069 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1012, i32 %188, i32 2)
  %1070 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1015, i32 %188, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1071 = add i32 %211, 192
  %1072 = mul i32 %1071, 4
  %1073 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %1072, i32 %204, i32 0)
  %1074 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %1072, i32 %210, i32 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1075 = add i64 %522, 65536
  %1076 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1075
  %1077 = load <16 x i8>, ptr addrspace(3) %1076, align 1
  %1078 = add i64 %527, 65536
  %1079 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1078
  %1080 = load <16 x i8>, ptr addrspace(3) %1079, align 1
  %1081 = add i64 %535, 65536
  %1082 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1081
  %1083 = load <16 x i8>, ptr addrspace(3) %1082, align 1
  %1084 = add i64 %539, 65536
  %1085 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1084
  %1086 = load <16 x i8>, ptr addrspace(3) %1085, align 1
  %1087 = add i64 %547, 65536
  %1088 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1087
  %1089 = load <16 x i8>, ptr addrspace(3) %1088, align 1
  %1090 = add i64 %551, 65536
  %1091 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1090
  %1092 = load <16 x i8>, ptr addrspace(3) %1091, align 1
  %1093 = add i64 %559, 65536
  %1094 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1093
  %1095 = load <16 x i8>, ptr addrspace(3) %1094, align 1
  %1096 = add i64 %563, 65536
  %1097 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1096
  %1098 = load <16 x i8>, ptr addrspace(3) %1097, align 1
  %1099 = add i64 %571, 65536
  %1100 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1099
  %1101 = load <16 x i8>, ptr addrspace(3) %1100, align 1
  %1102 = add i64 %575, 65536
  %1103 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1102
  %1104 = load <16 x i8>, ptr addrspace(3) %1103, align 1
  %1105 = add i64 %583, 65536
  %1106 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1105
  %1107 = load <16 x i8>, ptr addrspace(3) %1106, align 1
  %1108 = add i64 %587, 65536
  %1109 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1108
  %1110 = load <16 x i8>, ptr addrspace(3) %1109, align 1
  %1111 = add i64 %595, 65536
  %1112 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1111
  %1113 = load <16 x i8>, ptr addrspace(3) %1112, align 1
  %1114 = add i64 %599, 65536
  %1115 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1114
  %1116 = load <16 x i8>, ptr addrspace(3) %1115, align 1
  %1117 = add i64 %607, 65536
  %1118 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1117
  %1119 = load <16 x i8>, ptr addrspace(3) %1118, align 1
  %1120 = add i64 %611, 65536
  %1121 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1120
  %1122 = load <16 x i8>, ptr addrspace(3) %1121, align 1
  %1123 = add i64 %193, 128
  %1124 = add i64 %1123, %24
  %1125 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %1124
  %1126 = load <1 x i32>, ptr addrspace(3) %1125, align 4
  %1127 = extractelement <1 x i32> %1126, i64 0
  %1128 = add i64 %193, 1920
  %1129 = add i64 %1128, %24
  %1130 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %1129
  %1131 = load <1 x i32>, ptr addrspace(3) %1130, align 4
  %1132 = extractelement <1 x i32> %1131, i64 0
  %1133 = add i64 %193, 3712
  %1134 = add i64 %1133, %24
  %1135 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %1134
  %1136 = load <1 x i32>, ptr addrspace(3) %1135, align 4
  %1137 = extractelement <1 x i32> %1136, i64 0
  %1138 = add i64 %193, 5504
  %1139 = add i64 %1138, %24
  %1140 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %1139
  %1141 = load <1 x i32>, ptr addrspace(3) %1140, align 4
  %1142 = extractelement <1 x i32> %1141, i64 0
  %1143 = add i64 %77, 128
  %1144 = add i64 %1143, %71
  %1145 = trunc i64 %1144 to i32
  %1146 = mul i32 %1145, 4
  %1147 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1146, i32 0, i32 0)
  %1148 = bitcast <4 x i32> %1147 to <16 x i8>
  %1149 = add i64 %90, 128
  %1150 = add i64 %1149, %84
  %1151 = trunc i64 %1150 to i32
  %1152 = mul i32 %1151, 4
  %1153 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1152, i32 0, i32 0)
  %1154 = bitcast <4 x i32> %1153 to <16 x i8>
  %1155 = add i64 %103, 128
  %1156 = add i64 %1155, %97
  %1157 = trunc i64 %1156 to i32
  %1158 = mul i32 %1157, 4
  %1159 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1158, i32 0, i32 0)
  %1160 = bitcast <4 x i32> %1159 to <16 x i8>
  %1161 = add i64 %116, 128
  %1162 = add i64 %1161, %110
  %1163 = trunc i64 %1162 to i32
  %1164 = mul i32 %1163, 4
  %1165 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1164, i32 0, i32 0)
  %1166 = bitcast <4 x i32> %1165 to <16 x i8>
  %1167 = add i64 %129, 128
  %1168 = add i64 %1167, %123
  %1169 = trunc i64 %1168 to i32
  %1170 = mul i32 %1169, 4
  %1171 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1170, i32 0, i32 0)
  %1172 = bitcast <4 x i32> %1171 to <16 x i8>
  %1173 = add i64 %142, 128
  %1174 = add i64 %1173, %136
  %1175 = trunc i64 %1174 to i32
  %1176 = mul i32 %1175, 4
  %1177 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1176, i32 0, i32 0)
  %1178 = bitcast <4 x i32> %1177 to <16 x i8>
  %1179 = add i64 %155, 128
  %1180 = add i64 %1179, %149
  %1181 = trunc i64 %1180 to i32
  %1182 = mul i32 %1181, 4
  %1183 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1182, i32 0, i32 0)
  %1184 = bitcast <4 x i32> %1183 to <16 x i8>
  %1185 = add i64 %168, 128
  %1186 = add i64 %1185, %162
  %1187 = trunc i64 %1186 to i32
  %1188 = mul i32 %1187, 4
  %1189 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1188, i32 0, i32 0)
  %1190 = bitcast <4 x i32> %1189 to <16 x i8>
  %1191 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %487
  store <16 x i8> %1148, ptr addrspace(3) %1191, align 1
  %1192 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %489
  store <16 x i8> %1154, ptr addrspace(3) %1192, align 1
  %1193 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %491
  store <16 x i8> %1160, ptr addrspace(3) %1193, align 1
  %1194 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %493
  store <16 x i8> %1166, ptr addrspace(3) %1194, align 1
  %1195 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %495
  store <16 x i8> %1172, ptr addrspace(3) %1195, align 1
  %1196 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %497
  store <16 x i8> %1178, ptr addrspace(3) %1196, align 1
  %1197 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %499
  store <16 x i8> %1184, ptr addrspace(3) %1197, align 1
  %1198 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %501
  store <16 x i8> %1190, ptr addrspace(3) %1198, align 1
  %1199 = bitcast <16 x i8> %1077 to <4 x i32>
  %1200 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %982, <4 x i32> %1199, <4 x i32> %746, i32 %1127, i32 %854)
  %1201 = bitcast <16 x i8> %1080 to <4 x i32>
  %1202 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1200, <4 x i32> %1201, <4 x i32> %749, i32 %1127, i32 %854)
  %1203 = bitcast <16 x i8> %1083 to <4 x i32>
  %1204 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %986, <4 x i32> %1203, <4 x i32> %746, i32 %1127, i32 %854)
  %1205 = bitcast <16 x i8> %1086 to <4 x i32>
  %1206 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1204, <4 x i32> %1205, <4 x i32> %749, i32 %1127, i32 %854)
  %1207 = bitcast <16 x i8> %1089 to <4 x i32>
  %1208 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %990, <4 x i32> %1207, <4 x i32> %746, i32 %1132, i32 %854)
  %1209 = bitcast <16 x i8> %1092 to <4 x i32>
  %1210 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1208, <4 x i32> %1209, <4 x i32> %749, i32 %1132, i32 %854)
  %1211 = bitcast <16 x i8> %1095 to <4 x i32>
  %1212 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %994, <4 x i32> %1211, <4 x i32> %746, i32 %1132, i32 %854)
  %1213 = bitcast <16 x i8> %1098 to <4 x i32>
  %1214 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1212, <4 x i32> %1213, <4 x i32> %749, i32 %1132, i32 %854)
  %1215 = bitcast <16 x i8> %1101 to <4 x i32>
  %1216 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %998, <4 x i32> %1215, <4 x i32> %746, i32 %1137, i32 %854)
  %1217 = bitcast <16 x i8> %1104 to <4 x i32>
  %1218 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1216, <4 x i32> %1217, <4 x i32> %749, i32 %1137, i32 %854)
  %1219 = bitcast <16 x i8> %1107 to <4 x i32>
  %1220 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1002, <4 x i32> %1219, <4 x i32> %746, i32 %1137, i32 %854)
  %1221 = bitcast <16 x i8> %1110 to <4 x i32>
  %1222 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1220, <4 x i32> %1221, <4 x i32> %749, i32 %1137, i32 %854)
  %1223 = bitcast <16 x i8> %1113 to <4 x i32>
  %1224 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1006, <4 x i32> %1223, <4 x i32> %746, i32 %1142, i32 %854)
  %1225 = bitcast <16 x i8> %1116 to <4 x i32>
  %1226 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1224, <4 x i32> %1225, <4 x i32> %749, i32 %1142, i32 %854)
  %1227 = bitcast <16 x i8> %1119 to <4 x i32>
  %1228 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1010, <4 x i32> %1227, <4 x i32> %746, i32 %1142, i32 %854)
  %1229 = bitcast <16 x i8> %1122 to <4 x i32>
  %1230 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1228, <4 x i32> %1229, <4 x i32> %749, i32 %1142, i32 %854)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1231 = add i32 %192, 2048
  %1232 = mul i32 %1231, 4
  %1233 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1232, i32 %176, i32 2)
  %1234 = add i32 %192, 2304
  %1235 = mul i32 %1234, 4
  %1236 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1235, i32 %176, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1237 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1018, <4 x i32> %1199, <4 x i32> %782, i32 %1127, i32 %854)
  %1238 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1237, <4 x i32> %1201, <4 x i32> %783, i32 %1127, i32 %854)
  %1239 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1020, <4 x i32> %1203, <4 x i32> %782, i32 %1127, i32 %854)
  %1240 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1239, <4 x i32> %1205, <4 x i32> %783, i32 %1127, i32 %854)
  %1241 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1022, <4 x i32> %1207, <4 x i32> %782, i32 %1132, i32 %854)
  %1242 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1241, <4 x i32> %1209, <4 x i32> %783, i32 %1132, i32 %854)
  %1243 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1024, <4 x i32> %1211, <4 x i32> %782, i32 %1132, i32 %854)
  %1244 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1243, <4 x i32> %1213, <4 x i32> %783, i32 %1132, i32 %854)
  %1245 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1026, <4 x i32> %1215, <4 x i32> %782, i32 %1137, i32 %854)
  %1246 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1245, <4 x i32> %1217, <4 x i32> %783, i32 %1137, i32 %854)
  %1247 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1028, <4 x i32> %1219, <4 x i32> %782, i32 %1137, i32 %854)
  %1248 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1247, <4 x i32> %1221, <4 x i32> %783, i32 %1137, i32 %854)
  %1249 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1030, <4 x i32> %1223, <4 x i32> %782, i32 %1142, i32 %854)
  %1250 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1249, <4 x i32> %1225, <4 x i32> %783, i32 %1142, i32 %854)
  %1251 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1032, <4 x i32> %1227, <4 x i32> %782, i32 %1142, i32 %854)
  %1252 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1251, <4 x i32> %1229, <4 x i32> %783, i32 %1142, i32 %854)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1253 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1232, i32 %180, i32 2)
  %1254 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1235, i32 %180, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1255 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1036, <4 x i32> %1199, <4 x i32> %816, i32 %1127, i32 %855)
  %1256 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1255, <4 x i32> %1201, <4 x i32> %817, i32 %1127, i32 %855)
  %1257 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1038, <4 x i32> %1203, <4 x i32> %816, i32 %1127, i32 %855)
  %1258 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1257, <4 x i32> %1205, <4 x i32> %817, i32 %1127, i32 %855)
  %1259 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1040, <4 x i32> %1207, <4 x i32> %816, i32 %1132, i32 %855)
  %1260 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1259, <4 x i32> %1209, <4 x i32> %817, i32 %1132, i32 %855)
  %1261 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1042, <4 x i32> %1211, <4 x i32> %816, i32 %1132, i32 %855)
  %1262 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1261, <4 x i32> %1213, <4 x i32> %817, i32 %1132, i32 %855)
  %1263 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1044, <4 x i32> %1215, <4 x i32> %816, i32 %1137, i32 %855)
  %1264 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1263, <4 x i32> %1217, <4 x i32> %817, i32 %1137, i32 %855)
  %1265 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1046, <4 x i32> %1219, <4 x i32> %816, i32 %1137, i32 %855)
  %1266 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1265, <4 x i32> %1221, <4 x i32> %817, i32 %1137, i32 %855)
  %1267 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1048, <4 x i32> %1223, <4 x i32> %816, i32 %1142, i32 %855)
  %1268 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1267, <4 x i32> %1225, <4 x i32> %817, i32 %1142, i32 %855)
  %1269 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1050, <4 x i32> %1227, <4 x i32> %816, i32 %1142, i32 %855)
  %1270 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1269, <4 x i32> %1229, <4 x i32> %817, i32 %1142, i32 %855)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1271 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1232, i32 %184, i32 2)
  %1272 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1235, i32 %184, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1273 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1054, <4 x i32> %1199, <4 x i32> %850, i32 %1127, i32 %855)
  %1274 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1273, <4 x i32> %1201, <4 x i32> %851, i32 %1127, i32 %855)
  %1275 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1056, <4 x i32> %1203, <4 x i32> %850, i32 %1127, i32 %855)
  %1276 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1275, <4 x i32> %1205, <4 x i32> %851, i32 %1127, i32 %855)
  %1277 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1058, <4 x i32> %1207, <4 x i32> %850, i32 %1132, i32 %855)
  %1278 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1277, <4 x i32> %1209, <4 x i32> %851, i32 %1132, i32 %855)
  %1279 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1060, <4 x i32> %1211, <4 x i32> %850, i32 %1132, i32 %855)
  %1280 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1279, <4 x i32> %1213, <4 x i32> %851, i32 %1132, i32 %855)
  %1281 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1062, <4 x i32> %1215, <4 x i32> %850, i32 %1137, i32 %855)
  %1282 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1281, <4 x i32> %1217, <4 x i32> %851, i32 %1137, i32 %855)
  %1283 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1064, <4 x i32> %1219, <4 x i32> %850, i32 %1137, i32 %855)
  %1284 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1283, <4 x i32> %1221, <4 x i32> %851, i32 %1137, i32 %855)
  %1285 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1066, <4 x i32> %1223, <4 x i32> %850, i32 %1142, i32 %855)
  %1286 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1285, <4 x i32> %1225, <4 x i32> %851, i32 %1142, i32 %855)
  %1287 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1068, <4 x i32> %1227, <4 x i32> %850, i32 %1142, i32 %855)
  %1288 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1287, <4 x i32> %1229, <4 x i32> %851, i32 %1142, i32 %855)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1289 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1232, i32 %188, i32 2)
  %1290 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1235, i32 %188, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1291 = add i32 %211, 256
  %1292 = mul i32 %1291, 4
  %1293 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %1292, i32 %204, i32 0)
  %1294 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %1292, i32 %210, i32 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1295 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %522
  %1296 = load <16 x i8>, ptr addrspace(3) %1295, align 1
  %1297 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %527
  %1298 = load <16 x i8>, ptr addrspace(3) %1297, align 1
  %1299 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %535
  %1300 = load <16 x i8>, ptr addrspace(3) %1299, align 1
  %1301 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %539
  %1302 = load <16 x i8>, ptr addrspace(3) %1301, align 1
  %1303 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %547
  %1304 = load <16 x i8>, ptr addrspace(3) %1303, align 1
  %1305 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %551
  %1306 = load <16 x i8>, ptr addrspace(3) %1305, align 1
  %1307 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %559
  %1308 = load <16 x i8>, ptr addrspace(3) %1307, align 1
  %1309 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %563
  %1310 = load <16 x i8>, ptr addrspace(3) %1309, align 1
  %1311 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %571
  %1312 = load <16 x i8>, ptr addrspace(3) %1311, align 1
  %1313 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %575
  %1314 = load <16 x i8>, ptr addrspace(3) %1313, align 1
  %1315 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %583
  %1316 = load <16 x i8>, ptr addrspace(3) %1315, align 1
  %1317 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %587
  %1318 = load <16 x i8>, ptr addrspace(3) %1317, align 1
  %1319 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %595
  %1320 = load <16 x i8>, ptr addrspace(3) %1319, align 1
  %1321 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %599
  %1322 = load <16 x i8>, ptr addrspace(3) %1321, align 1
  %1323 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %607
  %1324 = load <16 x i8>, ptr addrspace(3) %1323, align 1
  %1325 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %611
  %1326 = load <16 x i8>, ptr addrspace(3) %1325, align 1
  %1327 = add i64 %193, 192
  %1328 = add i64 %1327, %24
  %1329 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %1328
  %1330 = load <1 x i32>, ptr addrspace(3) %1329, align 4
  %1331 = extractelement <1 x i32> %1330, i64 0
  %1332 = add i64 %193, 1984
  %1333 = add i64 %1332, %24
  %1334 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %1333
  %1335 = load <1 x i32>, ptr addrspace(3) %1334, align 4
  %1336 = extractelement <1 x i32> %1335, i64 0
  %1337 = add i64 %193, 3776
  %1338 = add i64 %1337, %24
  %1339 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %1338
  %1340 = load <1 x i32>, ptr addrspace(3) %1339, align 4
  %1341 = extractelement <1 x i32> %1340, i64 0
  %1342 = add i64 %193, 5568
  %1343 = add i64 %1342, %24
  %1344 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %1343
  %1345 = load <1 x i32>, ptr addrspace(3) %1344, align 4
  %1346 = extractelement <1 x i32> %1345, i64 0
  %1347 = add i64 %77, 160
  %1348 = add i64 %1347, %71
  %1349 = trunc i64 %1348 to i32
  %1350 = mul i32 %1349, 4
  %1351 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1350, i32 0, i32 0)
  %1352 = bitcast <4 x i32> %1351 to <16 x i8>
  %1353 = add i64 %90, 160
  %1354 = add i64 %1353, %84
  %1355 = trunc i64 %1354 to i32
  %1356 = mul i32 %1355, 4
  %1357 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1356, i32 0, i32 0)
  %1358 = bitcast <4 x i32> %1357 to <16 x i8>
  %1359 = add i64 %103, 160
  %1360 = add i64 %1359, %97
  %1361 = trunc i64 %1360 to i32
  %1362 = mul i32 %1361, 4
  %1363 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1362, i32 0, i32 0)
  %1364 = bitcast <4 x i32> %1363 to <16 x i8>
  %1365 = add i64 %116, 160
  %1366 = add i64 %1365, %110
  %1367 = trunc i64 %1366 to i32
  %1368 = mul i32 %1367, 4
  %1369 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1368, i32 0, i32 0)
  %1370 = bitcast <4 x i32> %1369 to <16 x i8>
  %1371 = add i64 %129, 160
  %1372 = add i64 %1371, %123
  %1373 = trunc i64 %1372 to i32
  %1374 = mul i32 %1373, 4
  %1375 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1374, i32 0, i32 0)
  %1376 = bitcast <4 x i32> %1375 to <16 x i8>
  %1377 = add i64 %142, 160
  %1378 = add i64 %1377, %136
  %1379 = trunc i64 %1378 to i32
  %1380 = mul i32 %1379, 4
  %1381 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1380, i32 0, i32 0)
  %1382 = bitcast <4 x i32> %1381 to <16 x i8>
  %1383 = add i64 %155, 160
  %1384 = add i64 %1383, %149
  %1385 = trunc i64 %1384 to i32
  %1386 = mul i32 %1385, 4
  %1387 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1386, i32 0, i32 0)
  %1388 = bitcast <4 x i32> %1387 to <16 x i8>
  %1389 = add i64 %168, 160
  %1390 = add i64 %1389, %162
  %1391 = trunc i64 %1390 to i32
  %1392 = mul i32 %1391, 4
  %1393 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1392, i32 0, i32 0)
  %1394 = bitcast <4 x i32> %1393 to <16 x i8>
  %1395 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %680
  store <16 x i8> %1352, ptr addrspace(3) %1395, align 1
  %1396 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %682
  store <16 x i8> %1358, ptr addrspace(3) %1396, align 1
  %1397 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %684
  store <16 x i8> %1364, ptr addrspace(3) %1397, align 1
  %1398 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %686
  store <16 x i8> %1370, ptr addrspace(3) %1398, align 1
  %1399 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %688
  store <16 x i8> %1376, ptr addrspace(3) %1399, align 1
  %1400 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %690
  store <16 x i8> %1382, ptr addrspace(3) %1400, align 1
  %1401 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %692
  store <16 x i8> %1388, ptr addrspace(3) %1401, align 1
  %1402 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %694
  store <16 x i8> %1394, ptr addrspace(3) %1402, align 1
  %1403 = bitcast <16 x i8> %1296 to <4 x i32>
  %1404 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1202, <4 x i32> %1403, <4 x i32> %1013, i32 %1331, i32 %1073)
  %1405 = bitcast <16 x i8> %1298 to <4 x i32>
  %1406 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1404, <4 x i32> %1405, <4 x i32> %1016, i32 %1331, i32 %1073)
  %1407 = bitcast <16 x i8> %1300 to <4 x i32>
  %1408 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1206, <4 x i32> %1407, <4 x i32> %1013, i32 %1331, i32 %1073)
  %1409 = bitcast <16 x i8> %1302 to <4 x i32>
  %1410 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1408, <4 x i32> %1409, <4 x i32> %1016, i32 %1331, i32 %1073)
  %1411 = bitcast <16 x i8> %1304 to <4 x i32>
  %1412 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1210, <4 x i32> %1411, <4 x i32> %1013, i32 %1336, i32 %1073)
  %1413 = bitcast <16 x i8> %1306 to <4 x i32>
  %1414 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1412, <4 x i32> %1413, <4 x i32> %1016, i32 %1336, i32 %1073)
  %1415 = bitcast <16 x i8> %1308 to <4 x i32>
  %1416 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1214, <4 x i32> %1415, <4 x i32> %1013, i32 %1336, i32 %1073)
  %1417 = bitcast <16 x i8> %1310 to <4 x i32>
  %1418 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1416, <4 x i32> %1417, <4 x i32> %1016, i32 %1336, i32 %1073)
  %1419 = bitcast <16 x i8> %1312 to <4 x i32>
  %1420 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1218, <4 x i32> %1419, <4 x i32> %1013, i32 %1341, i32 %1073)
  %1421 = bitcast <16 x i8> %1314 to <4 x i32>
  %1422 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1420, <4 x i32> %1421, <4 x i32> %1016, i32 %1341, i32 %1073)
  %1423 = bitcast <16 x i8> %1316 to <4 x i32>
  %1424 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1222, <4 x i32> %1423, <4 x i32> %1013, i32 %1341, i32 %1073)
  %1425 = bitcast <16 x i8> %1318 to <4 x i32>
  %1426 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1424, <4 x i32> %1425, <4 x i32> %1016, i32 %1341, i32 %1073)
  %1427 = bitcast <16 x i8> %1320 to <4 x i32>
  %1428 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1226, <4 x i32> %1427, <4 x i32> %1013, i32 %1346, i32 %1073)
  %1429 = bitcast <16 x i8> %1322 to <4 x i32>
  %1430 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1428, <4 x i32> %1429, <4 x i32> %1016, i32 %1346, i32 %1073)
  %1431 = bitcast <16 x i8> %1324 to <4 x i32>
  %1432 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1230, <4 x i32> %1431, <4 x i32> %1013, i32 %1346, i32 %1073)
  %1433 = bitcast <16 x i8> %1326 to <4 x i32>
  %1434 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1432, <4 x i32> %1433, <4 x i32> %1016, i32 %1346, i32 %1073)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1435 = add i32 %192, 2560
  %1436 = mul i32 %1435, 4
  %1437 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1436, i32 %176, i32 2)
  %1438 = add i32 %192, 2816
  %1439 = mul i32 %1438, 4
  %1440 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1439, i32 %176, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1441 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1238, <4 x i32> %1403, <4 x i32> %1033, i32 %1331, i32 %1073)
  %1442 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1441, <4 x i32> %1405, <4 x i32> %1034, i32 %1331, i32 %1073)
  %1443 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1240, <4 x i32> %1407, <4 x i32> %1033, i32 %1331, i32 %1073)
  %1444 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1443, <4 x i32> %1409, <4 x i32> %1034, i32 %1331, i32 %1073)
  %1445 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1242, <4 x i32> %1411, <4 x i32> %1033, i32 %1336, i32 %1073)
  %1446 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1445, <4 x i32> %1413, <4 x i32> %1034, i32 %1336, i32 %1073)
  %1447 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1244, <4 x i32> %1415, <4 x i32> %1033, i32 %1336, i32 %1073)
  %1448 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1447, <4 x i32> %1417, <4 x i32> %1034, i32 %1336, i32 %1073)
  %1449 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1246, <4 x i32> %1419, <4 x i32> %1033, i32 %1341, i32 %1073)
  %1450 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1449, <4 x i32> %1421, <4 x i32> %1034, i32 %1341, i32 %1073)
  %1451 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1248, <4 x i32> %1423, <4 x i32> %1033, i32 %1341, i32 %1073)
  %1452 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1451, <4 x i32> %1425, <4 x i32> %1034, i32 %1341, i32 %1073)
  %1453 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1250, <4 x i32> %1427, <4 x i32> %1033, i32 %1346, i32 %1073)
  %1454 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1453, <4 x i32> %1429, <4 x i32> %1034, i32 %1346, i32 %1073)
  %1455 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1252, <4 x i32> %1431, <4 x i32> %1033, i32 %1346, i32 %1073)
  %1456 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1455, <4 x i32> %1433, <4 x i32> %1034, i32 %1346, i32 %1073)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1457 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1436, i32 %180, i32 2)
  %1458 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1439, i32 %180, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1459 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1256, <4 x i32> %1403, <4 x i32> %1051, i32 %1331, i32 %1074)
  %1460 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1459, <4 x i32> %1405, <4 x i32> %1052, i32 %1331, i32 %1074)
  %1461 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1258, <4 x i32> %1407, <4 x i32> %1051, i32 %1331, i32 %1074)
  %1462 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1461, <4 x i32> %1409, <4 x i32> %1052, i32 %1331, i32 %1074)
  %1463 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1260, <4 x i32> %1411, <4 x i32> %1051, i32 %1336, i32 %1074)
  %1464 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1463, <4 x i32> %1413, <4 x i32> %1052, i32 %1336, i32 %1074)
  %1465 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1262, <4 x i32> %1415, <4 x i32> %1051, i32 %1336, i32 %1074)
  %1466 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1465, <4 x i32> %1417, <4 x i32> %1052, i32 %1336, i32 %1074)
  %1467 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1264, <4 x i32> %1419, <4 x i32> %1051, i32 %1341, i32 %1074)
  %1468 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1467, <4 x i32> %1421, <4 x i32> %1052, i32 %1341, i32 %1074)
  %1469 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1266, <4 x i32> %1423, <4 x i32> %1051, i32 %1341, i32 %1074)
  %1470 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1469, <4 x i32> %1425, <4 x i32> %1052, i32 %1341, i32 %1074)
  %1471 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1268, <4 x i32> %1427, <4 x i32> %1051, i32 %1346, i32 %1074)
  %1472 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1471, <4 x i32> %1429, <4 x i32> %1052, i32 %1346, i32 %1074)
  %1473 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1270, <4 x i32> %1431, <4 x i32> %1051, i32 %1346, i32 %1074)
  %1474 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1473, <4 x i32> %1433, <4 x i32> %1052, i32 %1346, i32 %1074)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1475 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1436, i32 %184, i32 2)
  %1476 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1439, i32 %184, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1477 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1274, <4 x i32> %1403, <4 x i32> %1069, i32 %1331, i32 %1074)
  %1478 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1477, <4 x i32> %1405, <4 x i32> %1070, i32 %1331, i32 %1074)
  %1479 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1276, <4 x i32> %1407, <4 x i32> %1069, i32 %1331, i32 %1074)
  %1480 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1479, <4 x i32> %1409, <4 x i32> %1070, i32 %1331, i32 %1074)
  %1481 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1278, <4 x i32> %1411, <4 x i32> %1069, i32 %1336, i32 %1074)
  %1482 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1481, <4 x i32> %1413, <4 x i32> %1070, i32 %1336, i32 %1074)
  %1483 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1280, <4 x i32> %1415, <4 x i32> %1069, i32 %1336, i32 %1074)
  %1484 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1483, <4 x i32> %1417, <4 x i32> %1070, i32 %1336, i32 %1074)
  %1485 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1282, <4 x i32> %1419, <4 x i32> %1069, i32 %1341, i32 %1074)
  %1486 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1485, <4 x i32> %1421, <4 x i32> %1070, i32 %1341, i32 %1074)
  %1487 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1284, <4 x i32> %1423, <4 x i32> %1069, i32 %1341, i32 %1074)
  %1488 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1487, <4 x i32> %1425, <4 x i32> %1070, i32 %1341, i32 %1074)
  %1489 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1286, <4 x i32> %1427, <4 x i32> %1069, i32 %1346, i32 %1074)
  %1490 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1489, <4 x i32> %1429, <4 x i32> %1070, i32 %1346, i32 %1074)
  %1491 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1288, <4 x i32> %1431, <4 x i32> %1069, i32 %1346, i32 %1074)
  %1492 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1491, <4 x i32> %1433, <4 x i32> %1070, i32 %1346, i32 %1074)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1493 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1436, i32 %188, i32 2)
  %1494 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1439, i32 %188, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1495 = add i32 %211, 320
  %1496 = mul i32 %1495, 4
  %1497 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %1496, i32 %204, i32 0)
  %1498 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %1496, i32 %210, i32 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1499 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %856
  %1500 = load <16 x i8>, ptr addrspace(3) %1499, align 1
  %1501 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %859
  %1502 = load <16 x i8>, ptr addrspace(3) %1501, align 1
  %1503 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %862
  %1504 = load <16 x i8>, ptr addrspace(3) %1503, align 1
  %1505 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %865
  %1506 = load <16 x i8>, ptr addrspace(3) %1505, align 1
  %1507 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %868
  %1508 = load <16 x i8>, ptr addrspace(3) %1507, align 1
  %1509 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %871
  %1510 = load <16 x i8>, ptr addrspace(3) %1509, align 1
  %1511 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %874
  %1512 = load <16 x i8>, ptr addrspace(3) %1511, align 1
  %1513 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %877
  %1514 = load <16 x i8>, ptr addrspace(3) %1513, align 1
  %1515 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %880
  %1516 = load <16 x i8>, ptr addrspace(3) %1515, align 1
  %1517 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %883
  %1518 = load <16 x i8>, ptr addrspace(3) %1517, align 1
  %1519 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %886
  %1520 = load <16 x i8>, ptr addrspace(3) %1519, align 1
  %1521 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %889
  %1522 = load <16 x i8>, ptr addrspace(3) %1521, align 1
  %1523 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %892
  %1524 = load <16 x i8>, ptr addrspace(3) %1523, align 1
  %1525 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %895
  %1526 = load <16 x i8>, ptr addrspace(3) %1525, align 1
  %1527 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %898
  %1528 = load <16 x i8>, ptr addrspace(3) %1527, align 1
  %1529 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %901
  %1530 = load <16 x i8>, ptr addrspace(3) %1529, align 1
  %1531 = add i64 %193, 256
  %1532 = add i64 %1531, %24
  %1533 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %1532
  %1534 = load <1 x i32>, ptr addrspace(3) %1533, align 4
  %1535 = extractelement <1 x i32> %1534, i64 0
  %1536 = add i64 %193, 2048
  %1537 = add i64 %1536, %24
  %1538 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %1537
  %1539 = load <1 x i32>, ptr addrspace(3) %1538, align 4
  %1540 = extractelement <1 x i32> %1539, i64 0
  %1541 = add i64 %193, 3840
  %1542 = add i64 %1541, %24
  %1543 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %1542
  %1544 = load <1 x i32>, ptr addrspace(3) %1543, align 4
  %1545 = extractelement <1 x i32> %1544, i64 0
  %1546 = add i64 %193, 5632
  %1547 = add i64 %1546, %24
  %1548 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %1547
  %1549 = load <1 x i32>, ptr addrspace(3) %1548, align 4
  %1550 = extractelement <1 x i32> %1549, i64 0
  %1551 = add i64 %77, 192
  %1552 = add i64 %1551, %71
  %1553 = trunc i64 %1552 to i32
  %1554 = mul i32 %1553, 4
  %1555 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1554, i32 0, i32 0)
  %1556 = bitcast <4 x i32> %1555 to <16 x i8>
  %1557 = add i64 %90, 192
  %1558 = add i64 %1557, %84
  %1559 = trunc i64 %1558 to i32
  %1560 = mul i32 %1559, 4
  %1561 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1560, i32 0, i32 0)
  %1562 = bitcast <4 x i32> %1561 to <16 x i8>
  %1563 = add i64 %103, 192
  %1564 = add i64 %1563, %97
  %1565 = trunc i64 %1564 to i32
  %1566 = mul i32 %1565, 4
  %1567 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1566, i32 0, i32 0)
  %1568 = bitcast <4 x i32> %1567 to <16 x i8>
  %1569 = add i64 %116, 192
  %1570 = add i64 %1569, %110
  %1571 = trunc i64 %1570 to i32
  %1572 = mul i32 %1571, 4
  %1573 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1572, i32 0, i32 0)
  %1574 = bitcast <4 x i32> %1573 to <16 x i8>
  %1575 = add i64 %129, 192
  %1576 = add i64 %1575, %123
  %1577 = trunc i64 %1576 to i32
  %1578 = mul i32 %1577, 4
  %1579 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1578, i32 0, i32 0)
  %1580 = bitcast <4 x i32> %1579 to <16 x i8>
  %1581 = add i64 %142, 192
  %1582 = add i64 %1581, %136
  %1583 = trunc i64 %1582 to i32
  %1584 = mul i32 %1583, 4
  %1585 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1584, i32 0, i32 0)
  %1586 = bitcast <4 x i32> %1585 to <16 x i8>
  %1587 = add i64 %155, 192
  %1588 = add i64 %1587, %149
  %1589 = trunc i64 %1588 to i32
  %1590 = mul i32 %1589, 4
  %1591 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1590, i32 0, i32 0)
  %1592 = bitcast <4 x i32> %1591 to <16 x i8>
  %1593 = add i64 %168, 192
  %1594 = add i64 %1593, %162
  %1595 = trunc i64 %1594 to i32
  %1596 = mul i32 %1595, 4
  %1597 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1596, i32 0, i32 0)
  %1598 = bitcast <4 x i32> %1597 to <16 x i8>
  %1599 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %360
  store <16 x i8> %1556, ptr addrspace(3) %1599, align 1
  %1600 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %369
  store <16 x i8> %1562, ptr addrspace(3) %1600, align 1
  %1601 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %378
  store <16 x i8> %1568, ptr addrspace(3) %1601, align 1
  %1602 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %387
  store <16 x i8> %1574, ptr addrspace(3) %1602, align 1
  %1603 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %396
  store <16 x i8> %1580, ptr addrspace(3) %1603, align 1
  %1604 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %405
  store <16 x i8> %1586, ptr addrspace(3) %1604, align 1
  %1605 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %414
  store <16 x i8> %1592, ptr addrspace(3) %1605, align 1
  %1606 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %423
  store <16 x i8> %1598, ptr addrspace(3) %1606, align 1
  %1607 = bitcast <16 x i8> %1500 to <4 x i32>
  %1608 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1406, <4 x i32> %1607, <4 x i32> %1233, i32 %1535, i32 %1293)
  %1609 = bitcast <16 x i8> %1502 to <4 x i32>
  %1610 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1608, <4 x i32> %1609, <4 x i32> %1236, i32 %1535, i32 %1293)
  %1611 = bitcast <16 x i8> %1504 to <4 x i32>
  %1612 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1410, <4 x i32> %1611, <4 x i32> %1233, i32 %1535, i32 %1293)
  %1613 = bitcast <16 x i8> %1506 to <4 x i32>
  %1614 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1612, <4 x i32> %1613, <4 x i32> %1236, i32 %1535, i32 %1293)
  %1615 = bitcast <16 x i8> %1508 to <4 x i32>
  %1616 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1414, <4 x i32> %1615, <4 x i32> %1233, i32 %1540, i32 %1293)
  %1617 = bitcast <16 x i8> %1510 to <4 x i32>
  %1618 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1616, <4 x i32> %1617, <4 x i32> %1236, i32 %1540, i32 %1293)
  %1619 = bitcast <16 x i8> %1512 to <4 x i32>
  %1620 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1418, <4 x i32> %1619, <4 x i32> %1233, i32 %1540, i32 %1293)
  %1621 = bitcast <16 x i8> %1514 to <4 x i32>
  %1622 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1620, <4 x i32> %1621, <4 x i32> %1236, i32 %1540, i32 %1293)
  %1623 = bitcast <16 x i8> %1516 to <4 x i32>
  %1624 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1422, <4 x i32> %1623, <4 x i32> %1233, i32 %1545, i32 %1293)
  %1625 = bitcast <16 x i8> %1518 to <4 x i32>
  %1626 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1624, <4 x i32> %1625, <4 x i32> %1236, i32 %1545, i32 %1293)
  %1627 = bitcast <16 x i8> %1520 to <4 x i32>
  %1628 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1426, <4 x i32> %1627, <4 x i32> %1233, i32 %1545, i32 %1293)
  %1629 = bitcast <16 x i8> %1522 to <4 x i32>
  %1630 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1628, <4 x i32> %1629, <4 x i32> %1236, i32 %1545, i32 %1293)
  %1631 = bitcast <16 x i8> %1524 to <4 x i32>
  %1632 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1430, <4 x i32> %1631, <4 x i32> %1233, i32 %1550, i32 %1293)
  %1633 = bitcast <16 x i8> %1526 to <4 x i32>
  %1634 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1632, <4 x i32> %1633, <4 x i32> %1236, i32 %1550, i32 %1293)
  %1635 = bitcast <16 x i8> %1528 to <4 x i32>
  %1636 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1434, <4 x i32> %1635, <4 x i32> %1233, i32 %1550, i32 %1293)
  %1637 = bitcast <16 x i8> %1530 to <4 x i32>
  %1638 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1636, <4 x i32> %1637, <4 x i32> %1236, i32 %1550, i32 %1293)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1639 = add i32 %192, 3072
  %1640 = mul i32 %1639, 4
  %1641 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1640, i32 %176, i32 2)
  %1642 = add i32 %192, 3328
  %1643 = mul i32 %1642, 4
  %1644 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1643, i32 %176, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1645 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1442, <4 x i32> %1607, <4 x i32> %1253, i32 %1535, i32 %1293)
  %1646 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1645, <4 x i32> %1609, <4 x i32> %1254, i32 %1535, i32 %1293)
  %1647 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1444, <4 x i32> %1611, <4 x i32> %1253, i32 %1535, i32 %1293)
  %1648 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1647, <4 x i32> %1613, <4 x i32> %1254, i32 %1535, i32 %1293)
  %1649 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1446, <4 x i32> %1615, <4 x i32> %1253, i32 %1540, i32 %1293)
  %1650 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1649, <4 x i32> %1617, <4 x i32> %1254, i32 %1540, i32 %1293)
  %1651 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1448, <4 x i32> %1619, <4 x i32> %1253, i32 %1540, i32 %1293)
  %1652 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1651, <4 x i32> %1621, <4 x i32> %1254, i32 %1540, i32 %1293)
  %1653 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1450, <4 x i32> %1623, <4 x i32> %1253, i32 %1545, i32 %1293)
  %1654 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1653, <4 x i32> %1625, <4 x i32> %1254, i32 %1545, i32 %1293)
  %1655 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1452, <4 x i32> %1627, <4 x i32> %1253, i32 %1545, i32 %1293)
  %1656 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1655, <4 x i32> %1629, <4 x i32> %1254, i32 %1545, i32 %1293)
  %1657 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1454, <4 x i32> %1631, <4 x i32> %1253, i32 %1550, i32 %1293)
  %1658 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1657, <4 x i32> %1633, <4 x i32> %1254, i32 %1550, i32 %1293)
  %1659 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1456, <4 x i32> %1635, <4 x i32> %1253, i32 %1550, i32 %1293)
  %1660 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1659, <4 x i32> %1637, <4 x i32> %1254, i32 %1550, i32 %1293)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1661 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1640, i32 %180, i32 2)
  %1662 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1643, i32 %180, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1663 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1460, <4 x i32> %1607, <4 x i32> %1271, i32 %1535, i32 %1294)
  %1664 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1663, <4 x i32> %1609, <4 x i32> %1272, i32 %1535, i32 %1294)
  %1665 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1462, <4 x i32> %1611, <4 x i32> %1271, i32 %1535, i32 %1294)
  %1666 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1665, <4 x i32> %1613, <4 x i32> %1272, i32 %1535, i32 %1294)
  %1667 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1464, <4 x i32> %1615, <4 x i32> %1271, i32 %1540, i32 %1294)
  %1668 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1667, <4 x i32> %1617, <4 x i32> %1272, i32 %1540, i32 %1294)
  %1669 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1466, <4 x i32> %1619, <4 x i32> %1271, i32 %1540, i32 %1294)
  %1670 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1669, <4 x i32> %1621, <4 x i32> %1272, i32 %1540, i32 %1294)
  %1671 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1468, <4 x i32> %1623, <4 x i32> %1271, i32 %1545, i32 %1294)
  %1672 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1671, <4 x i32> %1625, <4 x i32> %1272, i32 %1545, i32 %1294)
  %1673 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1470, <4 x i32> %1627, <4 x i32> %1271, i32 %1545, i32 %1294)
  %1674 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1673, <4 x i32> %1629, <4 x i32> %1272, i32 %1545, i32 %1294)
  %1675 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1472, <4 x i32> %1631, <4 x i32> %1271, i32 %1550, i32 %1294)
  %1676 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1675, <4 x i32> %1633, <4 x i32> %1272, i32 %1550, i32 %1294)
  %1677 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1474, <4 x i32> %1635, <4 x i32> %1271, i32 %1550, i32 %1294)
  %1678 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1677, <4 x i32> %1637, <4 x i32> %1272, i32 %1550, i32 %1294)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1679 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1640, i32 %184, i32 2)
  %1680 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1643, i32 %184, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1681 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1478, <4 x i32> %1607, <4 x i32> %1289, i32 %1535, i32 %1294)
  %1682 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1681, <4 x i32> %1609, <4 x i32> %1290, i32 %1535, i32 %1294)
  %1683 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1480, <4 x i32> %1611, <4 x i32> %1289, i32 %1535, i32 %1294)
  %1684 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1683, <4 x i32> %1613, <4 x i32> %1290, i32 %1535, i32 %1294)
  %1685 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1482, <4 x i32> %1615, <4 x i32> %1289, i32 %1540, i32 %1294)
  %1686 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1685, <4 x i32> %1617, <4 x i32> %1290, i32 %1540, i32 %1294)
  %1687 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1484, <4 x i32> %1619, <4 x i32> %1289, i32 %1540, i32 %1294)
  %1688 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1687, <4 x i32> %1621, <4 x i32> %1290, i32 %1540, i32 %1294)
  %1689 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1486, <4 x i32> %1623, <4 x i32> %1289, i32 %1545, i32 %1294)
  %1690 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1689, <4 x i32> %1625, <4 x i32> %1290, i32 %1545, i32 %1294)
  %1691 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1488, <4 x i32> %1627, <4 x i32> %1289, i32 %1545, i32 %1294)
  %1692 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1691, <4 x i32> %1629, <4 x i32> %1290, i32 %1545, i32 %1294)
  %1693 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1490, <4 x i32> %1631, <4 x i32> %1289, i32 %1550, i32 %1294)
  %1694 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1693, <4 x i32> %1633, <4 x i32> %1290, i32 %1550, i32 %1294)
  %1695 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1492, <4 x i32> %1635, <4 x i32> %1289, i32 %1550, i32 %1294)
  %1696 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1695, <4 x i32> %1637, <4 x i32> %1290, i32 %1550, i32 %1294)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1697 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1640, i32 %188, i32 2)
  %1698 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1643, i32 %188, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1699 = add i32 %211, 384
  %1700 = mul i32 %1699, 4
  %1701 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %1700, i32 %204, i32 0)
  %1702 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %1700, i32 %210, i32 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1703 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1075
  %1704 = load <16 x i8>, ptr addrspace(3) %1703, align 1
  %1705 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1078
  %1706 = load <16 x i8>, ptr addrspace(3) %1705, align 1
  %1707 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1081
  %1708 = load <16 x i8>, ptr addrspace(3) %1707, align 1
  %1709 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1084
  %1710 = load <16 x i8>, ptr addrspace(3) %1709, align 1
  %1711 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1087
  %1712 = load <16 x i8>, ptr addrspace(3) %1711, align 1
  %1713 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1090
  %1714 = load <16 x i8>, ptr addrspace(3) %1713, align 1
  %1715 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1093
  %1716 = load <16 x i8>, ptr addrspace(3) %1715, align 1
  %1717 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1096
  %1718 = load <16 x i8>, ptr addrspace(3) %1717, align 1
  %1719 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1099
  %1720 = load <16 x i8>, ptr addrspace(3) %1719, align 1
  %1721 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1102
  %1722 = load <16 x i8>, ptr addrspace(3) %1721, align 1
  %1723 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1105
  %1724 = load <16 x i8>, ptr addrspace(3) %1723, align 1
  %1725 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1108
  %1726 = load <16 x i8>, ptr addrspace(3) %1725, align 1
  %1727 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1111
  %1728 = load <16 x i8>, ptr addrspace(3) %1727, align 1
  %1729 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1114
  %1730 = load <16 x i8>, ptr addrspace(3) %1729, align 1
  %1731 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1117
  %1732 = load <16 x i8>, ptr addrspace(3) %1731, align 1
  %1733 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1120
  %1734 = load <16 x i8>, ptr addrspace(3) %1733, align 1
  %1735 = add i64 %193, 320
  %1736 = add i64 %1735, %24
  %1737 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %1736
  %1738 = load <1 x i32>, ptr addrspace(3) %1737, align 4
  %1739 = extractelement <1 x i32> %1738, i64 0
  %1740 = add i64 %193, 2112
  %1741 = add i64 %1740, %24
  %1742 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %1741
  %1743 = load <1 x i32>, ptr addrspace(3) %1742, align 4
  %1744 = extractelement <1 x i32> %1743, i64 0
  %1745 = add i64 %193, 3904
  %1746 = add i64 %1745, %24
  %1747 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %1746
  %1748 = load <1 x i32>, ptr addrspace(3) %1747, align 4
  %1749 = extractelement <1 x i32> %1748, i64 0
  %1750 = add i64 %193, 5696
  %1751 = add i64 %1750, %24
  %1752 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %1751
  %1753 = load <1 x i32>, ptr addrspace(3) %1752, align 4
  %1754 = extractelement <1 x i32> %1753, i64 0
  %1755 = add i64 %77, 224
  %1756 = add i64 %1755, %71
  %1757 = trunc i64 %1756 to i32
  %1758 = mul i32 %1757, 4
  %1759 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1758, i32 0, i32 0)
  %1760 = bitcast <4 x i32> %1759 to <16 x i8>
  %1761 = add i64 %90, 224
  %1762 = add i64 %1761, %84
  %1763 = trunc i64 %1762 to i32
  %1764 = mul i32 %1763, 4
  %1765 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1764, i32 0, i32 0)
  %1766 = bitcast <4 x i32> %1765 to <16 x i8>
  %1767 = add i64 %103, 224
  %1768 = add i64 %1767, %97
  %1769 = trunc i64 %1768 to i32
  %1770 = mul i32 %1769, 4
  %1771 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1770, i32 0, i32 0)
  %1772 = bitcast <4 x i32> %1771 to <16 x i8>
  %1773 = add i64 %116, 224
  %1774 = add i64 %1773, %110
  %1775 = trunc i64 %1774 to i32
  %1776 = mul i32 %1775, 4
  %1777 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1776, i32 0, i32 0)
  %1778 = bitcast <4 x i32> %1777 to <16 x i8>
  %1779 = add i64 %129, 224
  %1780 = add i64 %1779, %123
  %1781 = trunc i64 %1780 to i32
  %1782 = mul i32 %1781, 4
  %1783 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1782, i32 0, i32 0)
  %1784 = bitcast <4 x i32> %1783 to <16 x i8>
  %1785 = add i64 %142, 224
  %1786 = add i64 %1785, %136
  %1787 = trunc i64 %1786 to i32
  %1788 = mul i32 %1787, 4
  %1789 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1788, i32 0, i32 0)
  %1790 = bitcast <4 x i32> %1789 to <16 x i8>
  %1791 = add i64 %155, 224
  %1792 = add i64 %1791, %149
  %1793 = trunc i64 %1792 to i32
  %1794 = mul i32 %1793, 4
  %1795 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1794, i32 0, i32 0)
  %1796 = bitcast <4 x i32> %1795 to <16 x i8>
  %1797 = add i64 %168, 224
  %1798 = add i64 %1797, %162
  %1799 = trunc i64 %1798 to i32
  %1800 = mul i32 %1799, 4
  %1801 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1800, i32 0, i32 0)
  %1802 = bitcast <4 x i32> %1801 to <16 x i8>
  %1803 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %487
  store <16 x i8> %1760, ptr addrspace(3) %1803, align 1
  %1804 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %489
  store <16 x i8> %1766, ptr addrspace(3) %1804, align 1
  %1805 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %491
  store <16 x i8> %1772, ptr addrspace(3) %1805, align 1
  %1806 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %493
  store <16 x i8> %1778, ptr addrspace(3) %1806, align 1
  %1807 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %495
  store <16 x i8> %1784, ptr addrspace(3) %1807, align 1
  %1808 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %497
  store <16 x i8> %1790, ptr addrspace(3) %1808, align 1
  %1809 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %499
  store <16 x i8> %1796, ptr addrspace(3) %1809, align 1
  %1810 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %501
  store <16 x i8> %1802, ptr addrspace(3) %1810, align 1
  %1811 = bitcast <16 x i8> %1704 to <4 x i32>
  %1812 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1610, <4 x i32> %1811, <4 x i32> %1437, i32 %1739, i32 %1497)
  %1813 = bitcast <16 x i8> %1706 to <4 x i32>
  %1814 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1812, <4 x i32> %1813, <4 x i32> %1440, i32 %1739, i32 %1497)
  %1815 = bitcast <16 x i8> %1708 to <4 x i32>
  %1816 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1614, <4 x i32> %1815, <4 x i32> %1437, i32 %1739, i32 %1497)
  %1817 = bitcast <16 x i8> %1710 to <4 x i32>
  %1818 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1816, <4 x i32> %1817, <4 x i32> %1440, i32 %1739, i32 %1497)
  %1819 = bitcast <16 x i8> %1712 to <4 x i32>
  %1820 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1618, <4 x i32> %1819, <4 x i32> %1437, i32 %1744, i32 %1497)
  %1821 = bitcast <16 x i8> %1714 to <4 x i32>
  %1822 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1820, <4 x i32> %1821, <4 x i32> %1440, i32 %1744, i32 %1497)
  %1823 = bitcast <16 x i8> %1716 to <4 x i32>
  %1824 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1622, <4 x i32> %1823, <4 x i32> %1437, i32 %1744, i32 %1497)
  %1825 = bitcast <16 x i8> %1718 to <4 x i32>
  %1826 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1824, <4 x i32> %1825, <4 x i32> %1440, i32 %1744, i32 %1497)
  %1827 = bitcast <16 x i8> %1720 to <4 x i32>
  %1828 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1626, <4 x i32> %1827, <4 x i32> %1437, i32 %1749, i32 %1497)
  %1829 = bitcast <16 x i8> %1722 to <4 x i32>
  %1830 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1828, <4 x i32> %1829, <4 x i32> %1440, i32 %1749, i32 %1497)
  %1831 = bitcast <16 x i8> %1724 to <4 x i32>
  %1832 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1630, <4 x i32> %1831, <4 x i32> %1437, i32 %1749, i32 %1497)
  %1833 = bitcast <16 x i8> %1726 to <4 x i32>
  %1834 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1832, <4 x i32> %1833, <4 x i32> %1440, i32 %1749, i32 %1497)
  %1835 = bitcast <16 x i8> %1728 to <4 x i32>
  %1836 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1634, <4 x i32> %1835, <4 x i32> %1437, i32 %1754, i32 %1497)
  %1837 = bitcast <16 x i8> %1730 to <4 x i32>
  %1838 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1836, <4 x i32> %1837, <4 x i32> %1440, i32 %1754, i32 %1497)
  %1839 = bitcast <16 x i8> %1732 to <4 x i32>
  %1840 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1638, <4 x i32> %1839, <4 x i32> %1437, i32 %1754, i32 %1497)
  %1841 = bitcast <16 x i8> %1734 to <4 x i32>
  %1842 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1840, <4 x i32> %1841, <4 x i32> %1440, i32 %1754, i32 %1497)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1843 = add i32 %192, 3584
  %1844 = mul i32 %1843, 4
  %1845 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1844, i32 %176, i32 2)
  %1846 = add i32 %192, 3840
  %1847 = mul i32 %1846, 4
  %1848 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1847, i32 %176, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1849 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1646, <4 x i32> %1811, <4 x i32> %1457, i32 %1739, i32 %1497)
  %1850 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1849, <4 x i32> %1813, <4 x i32> %1458, i32 %1739, i32 %1497)
  %1851 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1648, <4 x i32> %1815, <4 x i32> %1457, i32 %1739, i32 %1497)
  %1852 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1851, <4 x i32> %1817, <4 x i32> %1458, i32 %1739, i32 %1497)
  %1853 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1650, <4 x i32> %1819, <4 x i32> %1457, i32 %1744, i32 %1497)
  %1854 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1853, <4 x i32> %1821, <4 x i32> %1458, i32 %1744, i32 %1497)
  %1855 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1652, <4 x i32> %1823, <4 x i32> %1457, i32 %1744, i32 %1497)
  %1856 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1855, <4 x i32> %1825, <4 x i32> %1458, i32 %1744, i32 %1497)
  %1857 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1654, <4 x i32> %1827, <4 x i32> %1457, i32 %1749, i32 %1497)
  %1858 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1857, <4 x i32> %1829, <4 x i32> %1458, i32 %1749, i32 %1497)
  %1859 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1656, <4 x i32> %1831, <4 x i32> %1457, i32 %1749, i32 %1497)
  %1860 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1859, <4 x i32> %1833, <4 x i32> %1458, i32 %1749, i32 %1497)
  %1861 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1658, <4 x i32> %1835, <4 x i32> %1457, i32 %1754, i32 %1497)
  %1862 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1861, <4 x i32> %1837, <4 x i32> %1458, i32 %1754, i32 %1497)
  %1863 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1660, <4 x i32> %1839, <4 x i32> %1457, i32 %1754, i32 %1497)
  %1864 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1863, <4 x i32> %1841, <4 x i32> %1458, i32 %1754, i32 %1497)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1865 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1844, i32 %180, i32 2)
  %1866 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1847, i32 %180, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1867 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1664, <4 x i32> %1811, <4 x i32> %1475, i32 %1739, i32 %1498)
  %1868 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1867, <4 x i32> %1813, <4 x i32> %1476, i32 %1739, i32 %1498)
  %1869 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1666, <4 x i32> %1815, <4 x i32> %1475, i32 %1739, i32 %1498)
  %1870 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1869, <4 x i32> %1817, <4 x i32> %1476, i32 %1739, i32 %1498)
  %1871 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1668, <4 x i32> %1819, <4 x i32> %1475, i32 %1744, i32 %1498)
  %1872 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1871, <4 x i32> %1821, <4 x i32> %1476, i32 %1744, i32 %1498)
  %1873 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1670, <4 x i32> %1823, <4 x i32> %1475, i32 %1744, i32 %1498)
  %1874 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1873, <4 x i32> %1825, <4 x i32> %1476, i32 %1744, i32 %1498)
  %1875 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1672, <4 x i32> %1827, <4 x i32> %1475, i32 %1749, i32 %1498)
  %1876 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1875, <4 x i32> %1829, <4 x i32> %1476, i32 %1749, i32 %1498)
  %1877 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1674, <4 x i32> %1831, <4 x i32> %1475, i32 %1749, i32 %1498)
  %1878 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1877, <4 x i32> %1833, <4 x i32> %1476, i32 %1749, i32 %1498)
  %1879 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1676, <4 x i32> %1835, <4 x i32> %1475, i32 %1754, i32 %1498)
  %1880 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1879, <4 x i32> %1837, <4 x i32> %1476, i32 %1754, i32 %1498)
  %1881 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1678, <4 x i32> %1839, <4 x i32> %1475, i32 %1754, i32 %1498)
  %1882 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1881, <4 x i32> %1841, <4 x i32> %1476, i32 %1754, i32 %1498)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1883 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1844, i32 %184, i32 2)
  %1884 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1847, i32 %184, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1885 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1682, <4 x i32> %1811, <4 x i32> %1493, i32 %1739, i32 %1498)
  %1886 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1885, <4 x i32> %1813, <4 x i32> %1494, i32 %1739, i32 %1498)
  %1887 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1684, <4 x i32> %1815, <4 x i32> %1493, i32 %1739, i32 %1498)
  %1888 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1887, <4 x i32> %1817, <4 x i32> %1494, i32 %1739, i32 %1498)
  %1889 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1686, <4 x i32> %1819, <4 x i32> %1493, i32 %1744, i32 %1498)
  %1890 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1889, <4 x i32> %1821, <4 x i32> %1494, i32 %1744, i32 %1498)
  %1891 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1688, <4 x i32> %1823, <4 x i32> %1493, i32 %1744, i32 %1498)
  %1892 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1891, <4 x i32> %1825, <4 x i32> %1494, i32 %1744, i32 %1498)
  %1893 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1690, <4 x i32> %1827, <4 x i32> %1493, i32 %1749, i32 %1498)
  %1894 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1893, <4 x i32> %1829, <4 x i32> %1494, i32 %1749, i32 %1498)
  %1895 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1692, <4 x i32> %1831, <4 x i32> %1493, i32 %1749, i32 %1498)
  %1896 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1895, <4 x i32> %1833, <4 x i32> %1494, i32 %1749, i32 %1498)
  %1897 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1694, <4 x i32> %1835, <4 x i32> %1493, i32 %1754, i32 %1498)
  %1898 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1897, <4 x i32> %1837, <4 x i32> %1494, i32 %1754, i32 %1498)
  %1899 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1696, <4 x i32> %1839, <4 x i32> %1493, i32 %1754, i32 %1498)
  %1900 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1899, <4 x i32> %1841, <4 x i32> %1494, i32 %1754, i32 %1498)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1901 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1844, i32 %188, i32 2)
  %1902 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %1847, i32 %188, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1903 = add i32 %211, 448
  %1904 = mul i32 %1903, 4
  %1905 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %1904, i32 %204, i32 0)
  %1906 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %1904, i32 %210, i32 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1907 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %522
  %1908 = load <16 x i8>, ptr addrspace(3) %1907, align 1
  %1909 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %527
  %1910 = load <16 x i8>, ptr addrspace(3) %1909, align 1
  %1911 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %535
  %1912 = load <16 x i8>, ptr addrspace(3) %1911, align 1
  %1913 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %539
  %1914 = load <16 x i8>, ptr addrspace(3) %1913, align 1
  %1915 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %547
  %1916 = load <16 x i8>, ptr addrspace(3) %1915, align 1
  %1917 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %551
  %1918 = load <16 x i8>, ptr addrspace(3) %1917, align 1
  %1919 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %559
  %1920 = load <16 x i8>, ptr addrspace(3) %1919, align 1
  %1921 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %563
  %1922 = load <16 x i8>, ptr addrspace(3) %1921, align 1
  %1923 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %571
  %1924 = load <16 x i8>, ptr addrspace(3) %1923, align 1
  %1925 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %575
  %1926 = load <16 x i8>, ptr addrspace(3) %1925, align 1
  %1927 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %583
  %1928 = load <16 x i8>, ptr addrspace(3) %1927, align 1
  %1929 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %587
  %1930 = load <16 x i8>, ptr addrspace(3) %1929, align 1
  %1931 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %595
  %1932 = load <16 x i8>, ptr addrspace(3) %1931, align 1
  %1933 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %599
  %1934 = load <16 x i8>, ptr addrspace(3) %1933, align 1
  %1935 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %607
  %1936 = load <16 x i8>, ptr addrspace(3) %1935, align 1
  %1937 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %611
  %1938 = load <16 x i8>, ptr addrspace(3) %1937, align 1
  %1939 = add i64 %193, 384
  %1940 = add i64 %1939, %24
  %1941 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %1940
  %1942 = load <1 x i32>, ptr addrspace(3) %1941, align 4
  %1943 = extractelement <1 x i32> %1942, i64 0
  %1944 = add i64 %193, 2176
  %1945 = add i64 %1944, %24
  %1946 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %1945
  %1947 = load <1 x i32>, ptr addrspace(3) %1946, align 4
  %1948 = extractelement <1 x i32> %1947, i64 0
  %1949 = add i64 %193, 3968
  %1950 = add i64 %1949, %24
  %1951 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %1950
  %1952 = load <1 x i32>, ptr addrspace(3) %1951, align 4
  %1953 = extractelement <1 x i32> %1952, i64 0
  %1954 = add i64 %193, 5760
  %1955 = add i64 %1954, %24
  %1956 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %1955
  %1957 = load <1 x i32>, ptr addrspace(3) %1956, align 4
  %1958 = extractelement <1 x i32> %1957, i64 0
  %1959 = add i64 %77, 256
  %1960 = add i64 %1959, %71
  %1961 = trunc i64 %1960 to i32
  %1962 = mul i32 %1961, 4
  %1963 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1962, i32 0, i32 0)
  %1964 = bitcast <4 x i32> %1963 to <16 x i8>
  %1965 = add i64 %90, 256
  %1966 = add i64 %1965, %84
  %1967 = trunc i64 %1966 to i32
  %1968 = mul i32 %1967, 4
  %1969 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1968, i32 0, i32 0)
  %1970 = bitcast <4 x i32> %1969 to <16 x i8>
  %1971 = add i64 %103, 256
  %1972 = add i64 %1971, %97
  %1973 = trunc i64 %1972 to i32
  %1974 = mul i32 %1973, 4
  %1975 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1974, i32 0, i32 0)
  %1976 = bitcast <4 x i32> %1975 to <16 x i8>
  %1977 = add i64 %116, 256
  %1978 = add i64 %1977, %110
  %1979 = trunc i64 %1978 to i32
  %1980 = mul i32 %1979, 4
  %1981 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1980, i32 0, i32 0)
  %1982 = bitcast <4 x i32> %1981 to <16 x i8>
  %1983 = add i64 %129, 256
  %1984 = add i64 %1983, %123
  %1985 = trunc i64 %1984 to i32
  %1986 = mul i32 %1985, 4
  %1987 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1986, i32 0, i32 0)
  %1988 = bitcast <4 x i32> %1987 to <16 x i8>
  %1989 = add i64 %142, 256
  %1990 = add i64 %1989, %136
  %1991 = trunc i64 %1990 to i32
  %1992 = mul i32 %1991, 4
  %1993 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1992, i32 0, i32 0)
  %1994 = bitcast <4 x i32> %1993 to <16 x i8>
  %1995 = add i64 %155, 256
  %1996 = add i64 %1995, %149
  %1997 = trunc i64 %1996 to i32
  %1998 = mul i32 %1997, 4
  %1999 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %1998, i32 0, i32 0)
  %2000 = bitcast <4 x i32> %1999 to <16 x i8>
  %2001 = add i64 %168, 256
  %2002 = add i64 %2001, %162
  %2003 = trunc i64 %2002 to i32
  %2004 = mul i32 %2003, 4
  %2005 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2004, i32 0, i32 0)
  %2006 = bitcast <4 x i32> %2005 to <16 x i8>
  %2007 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %680
  store <16 x i8> %1964, ptr addrspace(3) %2007, align 1
  %2008 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %682
  store <16 x i8> %1970, ptr addrspace(3) %2008, align 1
  %2009 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %684
  store <16 x i8> %1976, ptr addrspace(3) %2009, align 1
  %2010 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %686
  store <16 x i8> %1982, ptr addrspace(3) %2010, align 1
  %2011 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %688
  store <16 x i8> %1988, ptr addrspace(3) %2011, align 1
  %2012 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %690
  store <16 x i8> %1994, ptr addrspace(3) %2012, align 1
  %2013 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %692
  store <16 x i8> %2000, ptr addrspace(3) %2013, align 1
  %2014 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %694
  store <16 x i8> %2006, ptr addrspace(3) %2014, align 1
  %2015 = bitcast <16 x i8> %1908 to <4 x i32>
  %2016 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1814, <4 x i32> %2015, <4 x i32> %1641, i32 %1943, i32 %1701)
  %2017 = bitcast <16 x i8> %1910 to <4 x i32>
  %2018 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2016, <4 x i32> %2017, <4 x i32> %1644, i32 %1943, i32 %1701)
  %2019 = bitcast <16 x i8> %1912 to <4 x i32>
  %2020 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1818, <4 x i32> %2019, <4 x i32> %1641, i32 %1943, i32 %1701)
  %2021 = bitcast <16 x i8> %1914 to <4 x i32>
  %2022 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2020, <4 x i32> %2021, <4 x i32> %1644, i32 %1943, i32 %1701)
  %2023 = bitcast <16 x i8> %1916 to <4 x i32>
  %2024 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1822, <4 x i32> %2023, <4 x i32> %1641, i32 %1948, i32 %1701)
  %2025 = bitcast <16 x i8> %1918 to <4 x i32>
  %2026 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2024, <4 x i32> %2025, <4 x i32> %1644, i32 %1948, i32 %1701)
  %2027 = bitcast <16 x i8> %1920 to <4 x i32>
  %2028 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1826, <4 x i32> %2027, <4 x i32> %1641, i32 %1948, i32 %1701)
  %2029 = bitcast <16 x i8> %1922 to <4 x i32>
  %2030 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2028, <4 x i32> %2029, <4 x i32> %1644, i32 %1948, i32 %1701)
  %2031 = bitcast <16 x i8> %1924 to <4 x i32>
  %2032 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1830, <4 x i32> %2031, <4 x i32> %1641, i32 %1953, i32 %1701)
  %2033 = bitcast <16 x i8> %1926 to <4 x i32>
  %2034 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2032, <4 x i32> %2033, <4 x i32> %1644, i32 %1953, i32 %1701)
  %2035 = bitcast <16 x i8> %1928 to <4 x i32>
  %2036 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1834, <4 x i32> %2035, <4 x i32> %1641, i32 %1953, i32 %1701)
  %2037 = bitcast <16 x i8> %1930 to <4 x i32>
  %2038 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2036, <4 x i32> %2037, <4 x i32> %1644, i32 %1953, i32 %1701)
  %2039 = bitcast <16 x i8> %1932 to <4 x i32>
  %2040 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1838, <4 x i32> %2039, <4 x i32> %1641, i32 %1958, i32 %1701)
  %2041 = bitcast <16 x i8> %1934 to <4 x i32>
  %2042 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2040, <4 x i32> %2041, <4 x i32> %1644, i32 %1958, i32 %1701)
  %2043 = bitcast <16 x i8> %1936 to <4 x i32>
  %2044 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1842, <4 x i32> %2043, <4 x i32> %1641, i32 %1958, i32 %1701)
  %2045 = bitcast <16 x i8> %1938 to <4 x i32>
  %2046 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2044, <4 x i32> %2045, <4 x i32> %1644, i32 %1958, i32 %1701)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2047 = add i32 %192, 4096
  %2048 = mul i32 %2047, 4
  %2049 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2048, i32 %176, i32 2)
  %2050 = add i32 %192, 4352
  %2051 = mul i32 %2050, 4
  %2052 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2051, i32 %176, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2053 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1850, <4 x i32> %2015, <4 x i32> %1661, i32 %1943, i32 %1701)
  %2054 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2053, <4 x i32> %2017, <4 x i32> %1662, i32 %1943, i32 %1701)
  %2055 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1852, <4 x i32> %2019, <4 x i32> %1661, i32 %1943, i32 %1701)
  %2056 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2055, <4 x i32> %2021, <4 x i32> %1662, i32 %1943, i32 %1701)
  %2057 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1854, <4 x i32> %2023, <4 x i32> %1661, i32 %1948, i32 %1701)
  %2058 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2057, <4 x i32> %2025, <4 x i32> %1662, i32 %1948, i32 %1701)
  %2059 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1856, <4 x i32> %2027, <4 x i32> %1661, i32 %1948, i32 %1701)
  %2060 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2059, <4 x i32> %2029, <4 x i32> %1662, i32 %1948, i32 %1701)
  %2061 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1858, <4 x i32> %2031, <4 x i32> %1661, i32 %1953, i32 %1701)
  %2062 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2061, <4 x i32> %2033, <4 x i32> %1662, i32 %1953, i32 %1701)
  %2063 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1860, <4 x i32> %2035, <4 x i32> %1661, i32 %1953, i32 %1701)
  %2064 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2063, <4 x i32> %2037, <4 x i32> %1662, i32 %1953, i32 %1701)
  %2065 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1862, <4 x i32> %2039, <4 x i32> %1661, i32 %1958, i32 %1701)
  %2066 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2065, <4 x i32> %2041, <4 x i32> %1662, i32 %1958, i32 %1701)
  %2067 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1864, <4 x i32> %2043, <4 x i32> %1661, i32 %1958, i32 %1701)
  %2068 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2067, <4 x i32> %2045, <4 x i32> %1662, i32 %1958, i32 %1701)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2069 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2048, i32 %180, i32 2)
  %2070 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2051, i32 %180, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2071 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1868, <4 x i32> %2015, <4 x i32> %1679, i32 %1943, i32 %1702)
  %2072 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2071, <4 x i32> %2017, <4 x i32> %1680, i32 %1943, i32 %1702)
  %2073 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1870, <4 x i32> %2019, <4 x i32> %1679, i32 %1943, i32 %1702)
  %2074 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2073, <4 x i32> %2021, <4 x i32> %1680, i32 %1943, i32 %1702)
  %2075 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1872, <4 x i32> %2023, <4 x i32> %1679, i32 %1948, i32 %1702)
  %2076 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2075, <4 x i32> %2025, <4 x i32> %1680, i32 %1948, i32 %1702)
  %2077 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1874, <4 x i32> %2027, <4 x i32> %1679, i32 %1948, i32 %1702)
  %2078 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2077, <4 x i32> %2029, <4 x i32> %1680, i32 %1948, i32 %1702)
  %2079 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1876, <4 x i32> %2031, <4 x i32> %1679, i32 %1953, i32 %1702)
  %2080 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2079, <4 x i32> %2033, <4 x i32> %1680, i32 %1953, i32 %1702)
  %2081 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1878, <4 x i32> %2035, <4 x i32> %1679, i32 %1953, i32 %1702)
  %2082 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2081, <4 x i32> %2037, <4 x i32> %1680, i32 %1953, i32 %1702)
  %2083 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1880, <4 x i32> %2039, <4 x i32> %1679, i32 %1958, i32 %1702)
  %2084 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2083, <4 x i32> %2041, <4 x i32> %1680, i32 %1958, i32 %1702)
  %2085 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1882, <4 x i32> %2043, <4 x i32> %1679, i32 %1958, i32 %1702)
  %2086 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2085, <4 x i32> %2045, <4 x i32> %1680, i32 %1958, i32 %1702)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2087 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2048, i32 %184, i32 2)
  %2088 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2051, i32 %184, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2089 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1886, <4 x i32> %2015, <4 x i32> %1697, i32 %1943, i32 %1702)
  %2090 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2089, <4 x i32> %2017, <4 x i32> %1698, i32 %1943, i32 %1702)
  %2091 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1888, <4 x i32> %2019, <4 x i32> %1697, i32 %1943, i32 %1702)
  %2092 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2091, <4 x i32> %2021, <4 x i32> %1698, i32 %1943, i32 %1702)
  %2093 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1890, <4 x i32> %2023, <4 x i32> %1697, i32 %1948, i32 %1702)
  %2094 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2093, <4 x i32> %2025, <4 x i32> %1698, i32 %1948, i32 %1702)
  %2095 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1892, <4 x i32> %2027, <4 x i32> %1697, i32 %1948, i32 %1702)
  %2096 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2095, <4 x i32> %2029, <4 x i32> %1698, i32 %1948, i32 %1702)
  %2097 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1894, <4 x i32> %2031, <4 x i32> %1697, i32 %1953, i32 %1702)
  %2098 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2097, <4 x i32> %2033, <4 x i32> %1698, i32 %1953, i32 %1702)
  %2099 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1896, <4 x i32> %2035, <4 x i32> %1697, i32 %1953, i32 %1702)
  %2100 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2099, <4 x i32> %2037, <4 x i32> %1698, i32 %1953, i32 %1702)
  %2101 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1898, <4 x i32> %2039, <4 x i32> %1697, i32 %1958, i32 %1702)
  %2102 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2101, <4 x i32> %2041, <4 x i32> %1698, i32 %1958, i32 %1702)
  %2103 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %1900, <4 x i32> %2043, <4 x i32> %1697, i32 %1958, i32 %1702)
  %2104 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2103, <4 x i32> %2045, <4 x i32> %1698, i32 %1958, i32 %1702)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2105 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2048, i32 %188, i32 2)
  %2106 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2051, i32 %188, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2107 = add i32 %211, 512
  %2108 = mul i32 %2107, 4
  %2109 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %2108, i32 %204, i32 0)
  %2110 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %2108, i32 %210, i32 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %2111 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %856
  %2112 = load <16 x i8>, ptr addrspace(3) %2111, align 1
  %2113 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %859
  %2114 = load <16 x i8>, ptr addrspace(3) %2113, align 1
  %2115 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %862
  %2116 = load <16 x i8>, ptr addrspace(3) %2115, align 1
  %2117 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %865
  %2118 = load <16 x i8>, ptr addrspace(3) %2117, align 1
  %2119 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %868
  %2120 = load <16 x i8>, ptr addrspace(3) %2119, align 1
  %2121 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %871
  %2122 = load <16 x i8>, ptr addrspace(3) %2121, align 1
  %2123 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %874
  %2124 = load <16 x i8>, ptr addrspace(3) %2123, align 1
  %2125 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %877
  %2126 = load <16 x i8>, ptr addrspace(3) %2125, align 1
  %2127 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %880
  %2128 = load <16 x i8>, ptr addrspace(3) %2127, align 1
  %2129 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %883
  %2130 = load <16 x i8>, ptr addrspace(3) %2129, align 1
  %2131 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %886
  %2132 = load <16 x i8>, ptr addrspace(3) %2131, align 1
  %2133 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %889
  %2134 = load <16 x i8>, ptr addrspace(3) %2133, align 1
  %2135 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %892
  %2136 = load <16 x i8>, ptr addrspace(3) %2135, align 1
  %2137 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %895
  %2138 = load <16 x i8>, ptr addrspace(3) %2137, align 1
  %2139 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %898
  %2140 = load <16 x i8>, ptr addrspace(3) %2139, align 1
  %2141 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %901
  %2142 = load <16 x i8>, ptr addrspace(3) %2141, align 1
  %2143 = add i64 %193, 448
  %2144 = add i64 %2143, %24
  %2145 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %2144
  %2146 = load <1 x i32>, ptr addrspace(3) %2145, align 4
  %2147 = extractelement <1 x i32> %2146, i64 0
  %2148 = add i64 %193, 2240
  %2149 = add i64 %2148, %24
  %2150 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %2149
  %2151 = load <1 x i32>, ptr addrspace(3) %2150, align 4
  %2152 = extractelement <1 x i32> %2151, i64 0
  %2153 = add i64 %193, 4032
  %2154 = add i64 %2153, %24
  %2155 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %2154
  %2156 = load <1 x i32>, ptr addrspace(3) %2155, align 4
  %2157 = extractelement <1 x i32> %2156, i64 0
  %2158 = add i64 %193, 5824
  %2159 = add i64 %2158, %24
  %2160 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %2159
  %2161 = load <1 x i32>, ptr addrspace(3) %2160, align 4
  %2162 = extractelement <1 x i32> %2161, i64 0
  %2163 = add i64 %77, 288
  %2164 = add i64 %2163, %71
  %2165 = trunc i64 %2164 to i32
  %2166 = mul i32 %2165, 4
  %2167 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2166, i32 0, i32 0)
  %2168 = bitcast <4 x i32> %2167 to <16 x i8>
  %2169 = add i64 %90, 288
  %2170 = add i64 %2169, %84
  %2171 = trunc i64 %2170 to i32
  %2172 = mul i32 %2171, 4
  %2173 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2172, i32 0, i32 0)
  %2174 = bitcast <4 x i32> %2173 to <16 x i8>
  %2175 = add i64 %103, 288
  %2176 = add i64 %2175, %97
  %2177 = trunc i64 %2176 to i32
  %2178 = mul i32 %2177, 4
  %2179 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2178, i32 0, i32 0)
  %2180 = bitcast <4 x i32> %2179 to <16 x i8>
  %2181 = add i64 %116, 288
  %2182 = add i64 %2181, %110
  %2183 = trunc i64 %2182 to i32
  %2184 = mul i32 %2183, 4
  %2185 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2184, i32 0, i32 0)
  %2186 = bitcast <4 x i32> %2185 to <16 x i8>
  %2187 = add i64 %129, 288
  %2188 = add i64 %2187, %123
  %2189 = trunc i64 %2188 to i32
  %2190 = mul i32 %2189, 4
  %2191 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2190, i32 0, i32 0)
  %2192 = bitcast <4 x i32> %2191 to <16 x i8>
  %2193 = add i64 %142, 288
  %2194 = add i64 %2193, %136
  %2195 = trunc i64 %2194 to i32
  %2196 = mul i32 %2195, 4
  %2197 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2196, i32 0, i32 0)
  %2198 = bitcast <4 x i32> %2197 to <16 x i8>
  %2199 = add i64 %155, 288
  %2200 = add i64 %2199, %149
  %2201 = trunc i64 %2200 to i32
  %2202 = mul i32 %2201, 4
  %2203 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2202, i32 0, i32 0)
  %2204 = bitcast <4 x i32> %2203 to <16 x i8>
  %2205 = add i64 %168, 288
  %2206 = add i64 %2205, %162
  %2207 = trunc i64 %2206 to i32
  %2208 = mul i32 %2207, 4
  %2209 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2208, i32 0, i32 0)
  %2210 = bitcast <4 x i32> %2209 to <16 x i8>
  %2211 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %360
  store <16 x i8> %2168, ptr addrspace(3) %2211, align 1
  %2212 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %369
  store <16 x i8> %2174, ptr addrspace(3) %2212, align 1
  %2213 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %378
  store <16 x i8> %2180, ptr addrspace(3) %2213, align 1
  %2214 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %387
  store <16 x i8> %2186, ptr addrspace(3) %2214, align 1
  %2215 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %396
  store <16 x i8> %2192, ptr addrspace(3) %2215, align 1
  %2216 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %405
  store <16 x i8> %2198, ptr addrspace(3) %2216, align 1
  %2217 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %414
  store <16 x i8> %2204, ptr addrspace(3) %2217, align 1
  %2218 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %423
  store <16 x i8> %2210, ptr addrspace(3) %2218, align 1
  %2219 = bitcast <16 x i8> %2112 to <4 x i32>
  %2220 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2018, <4 x i32> %2219, <4 x i32> %1845, i32 %2147, i32 %1905)
  %2221 = bitcast <16 x i8> %2114 to <4 x i32>
  %2222 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2220, <4 x i32> %2221, <4 x i32> %1848, i32 %2147, i32 %1905)
  %2223 = bitcast <16 x i8> %2116 to <4 x i32>
  %2224 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2022, <4 x i32> %2223, <4 x i32> %1845, i32 %2147, i32 %1905)
  %2225 = bitcast <16 x i8> %2118 to <4 x i32>
  %2226 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2224, <4 x i32> %2225, <4 x i32> %1848, i32 %2147, i32 %1905)
  %2227 = bitcast <16 x i8> %2120 to <4 x i32>
  %2228 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2026, <4 x i32> %2227, <4 x i32> %1845, i32 %2152, i32 %1905)
  %2229 = bitcast <16 x i8> %2122 to <4 x i32>
  %2230 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2228, <4 x i32> %2229, <4 x i32> %1848, i32 %2152, i32 %1905)
  %2231 = bitcast <16 x i8> %2124 to <4 x i32>
  %2232 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2030, <4 x i32> %2231, <4 x i32> %1845, i32 %2152, i32 %1905)
  %2233 = bitcast <16 x i8> %2126 to <4 x i32>
  %2234 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2232, <4 x i32> %2233, <4 x i32> %1848, i32 %2152, i32 %1905)
  %2235 = bitcast <16 x i8> %2128 to <4 x i32>
  %2236 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2034, <4 x i32> %2235, <4 x i32> %1845, i32 %2157, i32 %1905)
  %2237 = bitcast <16 x i8> %2130 to <4 x i32>
  %2238 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2236, <4 x i32> %2237, <4 x i32> %1848, i32 %2157, i32 %1905)
  %2239 = bitcast <16 x i8> %2132 to <4 x i32>
  %2240 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2038, <4 x i32> %2239, <4 x i32> %1845, i32 %2157, i32 %1905)
  %2241 = bitcast <16 x i8> %2134 to <4 x i32>
  %2242 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2240, <4 x i32> %2241, <4 x i32> %1848, i32 %2157, i32 %1905)
  %2243 = bitcast <16 x i8> %2136 to <4 x i32>
  %2244 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2042, <4 x i32> %2243, <4 x i32> %1845, i32 %2162, i32 %1905)
  %2245 = bitcast <16 x i8> %2138 to <4 x i32>
  %2246 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2244, <4 x i32> %2245, <4 x i32> %1848, i32 %2162, i32 %1905)
  %2247 = bitcast <16 x i8> %2140 to <4 x i32>
  %2248 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2046, <4 x i32> %2247, <4 x i32> %1845, i32 %2162, i32 %1905)
  %2249 = bitcast <16 x i8> %2142 to <4 x i32>
  %2250 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2248, <4 x i32> %2249, <4 x i32> %1848, i32 %2162, i32 %1905)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2251 = add i32 %192, 4608
  %2252 = mul i32 %2251, 4
  %2253 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2252, i32 %176, i32 2)
  %2254 = add i32 %192, 4864
  %2255 = mul i32 %2254, 4
  %2256 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2255, i32 %176, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2257 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2054, <4 x i32> %2219, <4 x i32> %1865, i32 %2147, i32 %1905)
  %2258 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2257, <4 x i32> %2221, <4 x i32> %1866, i32 %2147, i32 %1905)
  %2259 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2056, <4 x i32> %2223, <4 x i32> %1865, i32 %2147, i32 %1905)
  %2260 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2259, <4 x i32> %2225, <4 x i32> %1866, i32 %2147, i32 %1905)
  %2261 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2058, <4 x i32> %2227, <4 x i32> %1865, i32 %2152, i32 %1905)
  %2262 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2261, <4 x i32> %2229, <4 x i32> %1866, i32 %2152, i32 %1905)
  %2263 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2060, <4 x i32> %2231, <4 x i32> %1865, i32 %2152, i32 %1905)
  %2264 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2263, <4 x i32> %2233, <4 x i32> %1866, i32 %2152, i32 %1905)
  %2265 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2062, <4 x i32> %2235, <4 x i32> %1865, i32 %2157, i32 %1905)
  %2266 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2265, <4 x i32> %2237, <4 x i32> %1866, i32 %2157, i32 %1905)
  %2267 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2064, <4 x i32> %2239, <4 x i32> %1865, i32 %2157, i32 %1905)
  %2268 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2267, <4 x i32> %2241, <4 x i32> %1866, i32 %2157, i32 %1905)
  %2269 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2066, <4 x i32> %2243, <4 x i32> %1865, i32 %2162, i32 %1905)
  %2270 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2269, <4 x i32> %2245, <4 x i32> %1866, i32 %2162, i32 %1905)
  %2271 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2068, <4 x i32> %2247, <4 x i32> %1865, i32 %2162, i32 %1905)
  %2272 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2271, <4 x i32> %2249, <4 x i32> %1866, i32 %2162, i32 %1905)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2273 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2252, i32 %180, i32 2)
  %2274 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2255, i32 %180, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2275 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2072, <4 x i32> %2219, <4 x i32> %1883, i32 %2147, i32 %1906)
  %2276 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2275, <4 x i32> %2221, <4 x i32> %1884, i32 %2147, i32 %1906)
  %2277 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2074, <4 x i32> %2223, <4 x i32> %1883, i32 %2147, i32 %1906)
  %2278 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2277, <4 x i32> %2225, <4 x i32> %1884, i32 %2147, i32 %1906)
  %2279 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2076, <4 x i32> %2227, <4 x i32> %1883, i32 %2152, i32 %1906)
  %2280 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2279, <4 x i32> %2229, <4 x i32> %1884, i32 %2152, i32 %1906)
  %2281 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2078, <4 x i32> %2231, <4 x i32> %1883, i32 %2152, i32 %1906)
  %2282 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2281, <4 x i32> %2233, <4 x i32> %1884, i32 %2152, i32 %1906)
  %2283 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2080, <4 x i32> %2235, <4 x i32> %1883, i32 %2157, i32 %1906)
  %2284 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2283, <4 x i32> %2237, <4 x i32> %1884, i32 %2157, i32 %1906)
  %2285 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2082, <4 x i32> %2239, <4 x i32> %1883, i32 %2157, i32 %1906)
  %2286 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2285, <4 x i32> %2241, <4 x i32> %1884, i32 %2157, i32 %1906)
  %2287 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2084, <4 x i32> %2243, <4 x i32> %1883, i32 %2162, i32 %1906)
  %2288 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2287, <4 x i32> %2245, <4 x i32> %1884, i32 %2162, i32 %1906)
  %2289 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2086, <4 x i32> %2247, <4 x i32> %1883, i32 %2162, i32 %1906)
  %2290 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2289, <4 x i32> %2249, <4 x i32> %1884, i32 %2162, i32 %1906)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2291 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2252, i32 %184, i32 2)
  %2292 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2255, i32 %184, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2293 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2090, <4 x i32> %2219, <4 x i32> %1901, i32 %2147, i32 %1906)
  %2294 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2293, <4 x i32> %2221, <4 x i32> %1902, i32 %2147, i32 %1906)
  %2295 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2092, <4 x i32> %2223, <4 x i32> %1901, i32 %2147, i32 %1906)
  %2296 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2295, <4 x i32> %2225, <4 x i32> %1902, i32 %2147, i32 %1906)
  %2297 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2094, <4 x i32> %2227, <4 x i32> %1901, i32 %2152, i32 %1906)
  %2298 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2297, <4 x i32> %2229, <4 x i32> %1902, i32 %2152, i32 %1906)
  %2299 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2096, <4 x i32> %2231, <4 x i32> %1901, i32 %2152, i32 %1906)
  %2300 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2299, <4 x i32> %2233, <4 x i32> %1902, i32 %2152, i32 %1906)
  %2301 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2098, <4 x i32> %2235, <4 x i32> %1901, i32 %2157, i32 %1906)
  %2302 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2301, <4 x i32> %2237, <4 x i32> %1902, i32 %2157, i32 %1906)
  %2303 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2100, <4 x i32> %2239, <4 x i32> %1901, i32 %2157, i32 %1906)
  %2304 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2303, <4 x i32> %2241, <4 x i32> %1902, i32 %2157, i32 %1906)
  %2305 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2102, <4 x i32> %2243, <4 x i32> %1901, i32 %2162, i32 %1906)
  %2306 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2305, <4 x i32> %2245, <4 x i32> %1902, i32 %2162, i32 %1906)
  %2307 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2104, <4 x i32> %2247, <4 x i32> %1901, i32 %2162, i32 %1906)
  %2308 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2307, <4 x i32> %2249, <4 x i32> %1902, i32 %2162, i32 %1906)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2309 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2252, i32 %188, i32 2)
  %2310 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2255, i32 %188, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2311 = add i32 %211, 576
  %2312 = mul i32 %2311, 4
  %2313 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %2312, i32 %204, i32 0)
  %2314 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %2312, i32 %210, i32 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %2315 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1075
  %2316 = load <16 x i8>, ptr addrspace(3) %2315, align 1
  %2317 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1078
  %2318 = load <16 x i8>, ptr addrspace(3) %2317, align 1
  %2319 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1081
  %2320 = load <16 x i8>, ptr addrspace(3) %2319, align 1
  %2321 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1084
  %2322 = load <16 x i8>, ptr addrspace(3) %2321, align 1
  %2323 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1087
  %2324 = load <16 x i8>, ptr addrspace(3) %2323, align 1
  %2325 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1090
  %2326 = load <16 x i8>, ptr addrspace(3) %2325, align 1
  %2327 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1093
  %2328 = load <16 x i8>, ptr addrspace(3) %2327, align 1
  %2329 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1096
  %2330 = load <16 x i8>, ptr addrspace(3) %2329, align 1
  %2331 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1099
  %2332 = load <16 x i8>, ptr addrspace(3) %2331, align 1
  %2333 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1102
  %2334 = load <16 x i8>, ptr addrspace(3) %2333, align 1
  %2335 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1105
  %2336 = load <16 x i8>, ptr addrspace(3) %2335, align 1
  %2337 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1108
  %2338 = load <16 x i8>, ptr addrspace(3) %2337, align 1
  %2339 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1111
  %2340 = load <16 x i8>, ptr addrspace(3) %2339, align 1
  %2341 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1114
  %2342 = load <16 x i8>, ptr addrspace(3) %2341, align 1
  %2343 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1117
  %2344 = load <16 x i8>, ptr addrspace(3) %2343, align 1
  %2345 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1120
  %2346 = load <16 x i8>, ptr addrspace(3) %2345, align 1
  %2347 = add i64 %193, 512
  %2348 = add i64 %2347, %24
  %2349 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %2348
  %2350 = load <1 x i32>, ptr addrspace(3) %2349, align 4
  %2351 = extractelement <1 x i32> %2350, i64 0
  %2352 = add i64 %193, 2304
  %2353 = add i64 %2352, %24
  %2354 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %2353
  %2355 = load <1 x i32>, ptr addrspace(3) %2354, align 4
  %2356 = extractelement <1 x i32> %2355, i64 0
  %2357 = add i64 %193, 4096
  %2358 = add i64 %2357, %24
  %2359 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %2358
  %2360 = load <1 x i32>, ptr addrspace(3) %2359, align 4
  %2361 = extractelement <1 x i32> %2360, i64 0
  %2362 = add i64 %193, 5888
  %2363 = add i64 %2362, %24
  %2364 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %2363
  %2365 = load <1 x i32>, ptr addrspace(3) %2364, align 4
  %2366 = extractelement <1 x i32> %2365, i64 0
  %2367 = add i64 %77, 320
  %2368 = add i64 %2367, %71
  %2369 = trunc i64 %2368 to i32
  %2370 = mul i32 %2369, 4
  %2371 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2370, i32 0, i32 0)
  %2372 = bitcast <4 x i32> %2371 to <16 x i8>
  %2373 = add i64 %90, 320
  %2374 = add i64 %2373, %84
  %2375 = trunc i64 %2374 to i32
  %2376 = mul i32 %2375, 4
  %2377 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2376, i32 0, i32 0)
  %2378 = bitcast <4 x i32> %2377 to <16 x i8>
  %2379 = add i64 %103, 320
  %2380 = add i64 %2379, %97
  %2381 = trunc i64 %2380 to i32
  %2382 = mul i32 %2381, 4
  %2383 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2382, i32 0, i32 0)
  %2384 = bitcast <4 x i32> %2383 to <16 x i8>
  %2385 = add i64 %116, 320
  %2386 = add i64 %2385, %110
  %2387 = trunc i64 %2386 to i32
  %2388 = mul i32 %2387, 4
  %2389 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2388, i32 0, i32 0)
  %2390 = bitcast <4 x i32> %2389 to <16 x i8>
  %2391 = add i64 %129, 320
  %2392 = add i64 %2391, %123
  %2393 = trunc i64 %2392 to i32
  %2394 = mul i32 %2393, 4
  %2395 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2394, i32 0, i32 0)
  %2396 = bitcast <4 x i32> %2395 to <16 x i8>
  %2397 = add i64 %142, 320
  %2398 = add i64 %2397, %136
  %2399 = trunc i64 %2398 to i32
  %2400 = mul i32 %2399, 4
  %2401 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2400, i32 0, i32 0)
  %2402 = bitcast <4 x i32> %2401 to <16 x i8>
  %2403 = add i64 %155, 320
  %2404 = add i64 %2403, %149
  %2405 = trunc i64 %2404 to i32
  %2406 = mul i32 %2405, 4
  %2407 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2406, i32 0, i32 0)
  %2408 = bitcast <4 x i32> %2407 to <16 x i8>
  %2409 = add i64 %168, 320
  %2410 = add i64 %2409, %162
  %2411 = trunc i64 %2410 to i32
  %2412 = mul i32 %2411, 4
  %2413 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2412, i32 0, i32 0)
  %2414 = bitcast <4 x i32> %2413 to <16 x i8>
  %2415 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %487
  store <16 x i8> %2372, ptr addrspace(3) %2415, align 1
  %2416 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %489
  store <16 x i8> %2378, ptr addrspace(3) %2416, align 1
  %2417 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %491
  store <16 x i8> %2384, ptr addrspace(3) %2417, align 1
  %2418 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %493
  store <16 x i8> %2390, ptr addrspace(3) %2418, align 1
  %2419 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %495
  store <16 x i8> %2396, ptr addrspace(3) %2419, align 1
  %2420 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %497
  store <16 x i8> %2402, ptr addrspace(3) %2420, align 1
  %2421 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %499
  store <16 x i8> %2408, ptr addrspace(3) %2421, align 1
  %2422 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %501
  store <16 x i8> %2414, ptr addrspace(3) %2422, align 1
  %2423 = bitcast <16 x i8> %2316 to <4 x i32>
  %2424 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2222, <4 x i32> %2423, <4 x i32> %2049, i32 %2351, i32 %2109)
  %2425 = bitcast <16 x i8> %2318 to <4 x i32>
  %2426 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2424, <4 x i32> %2425, <4 x i32> %2052, i32 %2351, i32 %2109)
  %2427 = bitcast <16 x i8> %2320 to <4 x i32>
  %2428 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2226, <4 x i32> %2427, <4 x i32> %2049, i32 %2351, i32 %2109)
  %2429 = bitcast <16 x i8> %2322 to <4 x i32>
  %2430 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2428, <4 x i32> %2429, <4 x i32> %2052, i32 %2351, i32 %2109)
  %2431 = bitcast <16 x i8> %2324 to <4 x i32>
  %2432 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2230, <4 x i32> %2431, <4 x i32> %2049, i32 %2356, i32 %2109)
  %2433 = bitcast <16 x i8> %2326 to <4 x i32>
  %2434 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2432, <4 x i32> %2433, <4 x i32> %2052, i32 %2356, i32 %2109)
  %2435 = bitcast <16 x i8> %2328 to <4 x i32>
  %2436 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2234, <4 x i32> %2435, <4 x i32> %2049, i32 %2356, i32 %2109)
  %2437 = bitcast <16 x i8> %2330 to <4 x i32>
  %2438 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2436, <4 x i32> %2437, <4 x i32> %2052, i32 %2356, i32 %2109)
  %2439 = bitcast <16 x i8> %2332 to <4 x i32>
  %2440 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2238, <4 x i32> %2439, <4 x i32> %2049, i32 %2361, i32 %2109)
  %2441 = bitcast <16 x i8> %2334 to <4 x i32>
  %2442 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2440, <4 x i32> %2441, <4 x i32> %2052, i32 %2361, i32 %2109)
  %2443 = bitcast <16 x i8> %2336 to <4 x i32>
  %2444 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2242, <4 x i32> %2443, <4 x i32> %2049, i32 %2361, i32 %2109)
  %2445 = bitcast <16 x i8> %2338 to <4 x i32>
  %2446 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2444, <4 x i32> %2445, <4 x i32> %2052, i32 %2361, i32 %2109)
  %2447 = bitcast <16 x i8> %2340 to <4 x i32>
  %2448 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2246, <4 x i32> %2447, <4 x i32> %2049, i32 %2366, i32 %2109)
  %2449 = bitcast <16 x i8> %2342 to <4 x i32>
  %2450 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2448, <4 x i32> %2449, <4 x i32> %2052, i32 %2366, i32 %2109)
  %2451 = bitcast <16 x i8> %2344 to <4 x i32>
  %2452 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2250, <4 x i32> %2451, <4 x i32> %2049, i32 %2366, i32 %2109)
  %2453 = bitcast <16 x i8> %2346 to <4 x i32>
  %2454 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2452, <4 x i32> %2453, <4 x i32> %2052, i32 %2366, i32 %2109)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2455 = add i32 %192, 5120
  %2456 = mul i32 %2455, 4
  %2457 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2456, i32 %176, i32 2)
  %2458 = add i32 %192, 5376
  %2459 = mul i32 %2458, 4
  %2460 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2459, i32 %176, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2461 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2258, <4 x i32> %2423, <4 x i32> %2069, i32 %2351, i32 %2109)
  %2462 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2461, <4 x i32> %2425, <4 x i32> %2070, i32 %2351, i32 %2109)
  %2463 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2260, <4 x i32> %2427, <4 x i32> %2069, i32 %2351, i32 %2109)
  %2464 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2463, <4 x i32> %2429, <4 x i32> %2070, i32 %2351, i32 %2109)
  %2465 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2262, <4 x i32> %2431, <4 x i32> %2069, i32 %2356, i32 %2109)
  %2466 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2465, <4 x i32> %2433, <4 x i32> %2070, i32 %2356, i32 %2109)
  %2467 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2264, <4 x i32> %2435, <4 x i32> %2069, i32 %2356, i32 %2109)
  %2468 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2467, <4 x i32> %2437, <4 x i32> %2070, i32 %2356, i32 %2109)
  %2469 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2266, <4 x i32> %2439, <4 x i32> %2069, i32 %2361, i32 %2109)
  %2470 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2469, <4 x i32> %2441, <4 x i32> %2070, i32 %2361, i32 %2109)
  %2471 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2268, <4 x i32> %2443, <4 x i32> %2069, i32 %2361, i32 %2109)
  %2472 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2471, <4 x i32> %2445, <4 x i32> %2070, i32 %2361, i32 %2109)
  %2473 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2270, <4 x i32> %2447, <4 x i32> %2069, i32 %2366, i32 %2109)
  %2474 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2473, <4 x i32> %2449, <4 x i32> %2070, i32 %2366, i32 %2109)
  %2475 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2272, <4 x i32> %2451, <4 x i32> %2069, i32 %2366, i32 %2109)
  %2476 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2475, <4 x i32> %2453, <4 x i32> %2070, i32 %2366, i32 %2109)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2477 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2456, i32 %180, i32 2)
  %2478 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2459, i32 %180, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2479 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2276, <4 x i32> %2423, <4 x i32> %2087, i32 %2351, i32 %2110)
  %2480 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2479, <4 x i32> %2425, <4 x i32> %2088, i32 %2351, i32 %2110)
  %2481 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2278, <4 x i32> %2427, <4 x i32> %2087, i32 %2351, i32 %2110)
  %2482 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2481, <4 x i32> %2429, <4 x i32> %2088, i32 %2351, i32 %2110)
  %2483 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2280, <4 x i32> %2431, <4 x i32> %2087, i32 %2356, i32 %2110)
  %2484 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2483, <4 x i32> %2433, <4 x i32> %2088, i32 %2356, i32 %2110)
  %2485 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2282, <4 x i32> %2435, <4 x i32> %2087, i32 %2356, i32 %2110)
  %2486 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2485, <4 x i32> %2437, <4 x i32> %2088, i32 %2356, i32 %2110)
  %2487 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2284, <4 x i32> %2439, <4 x i32> %2087, i32 %2361, i32 %2110)
  %2488 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2487, <4 x i32> %2441, <4 x i32> %2088, i32 %2361, i32 %2110)
  %2489 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2286, <4 x i32> %2443, <4 x i32> %2087, i32 %2361, i32 %2110)
  %2490 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2489, <4 x i32> %2445, <4 x i32> %2088, i32 %2361, i32 %2110)
  %2491 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2288, <4 x i32> %2447, <4 x i32> %2087, i32 %2366, i32 %2110)
  %2492 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2491, <4 x i32> %2449, <4 x i32> %2088, i32 %2366, i32 %2110)
  %2493 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2290, <4 x i32> %2451, <4 x i32> %2087, i32 %2366, i32 %2110)
  %2494 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2493, <4 x i32> %2453, <4 x i32> %2088, i32 %2366, i32 %2110)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2495 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2456, i32 %184, i32 2)
  %2496 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2459, i32 %184, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2497 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2294, <4 x i32> %2423, <4 x i32> %2105, i32 %2351, i32 %2110)
  %2498 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2497, <4 x i32> %2425, <4 x i32> %2106, i32 %2351, i32 %2110)
  %2499 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2296, <4 x i32> %2427, <4 x i32> %2105, i32 %2351, i32 %2110)
  %2500 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2499, <4 x i32> %2429, <4 x i32> %2106, i32 %2351, i32 %2110)
  %2501 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2298, <4 x i32> %2431, <4 x i32> %2105, i32 %2356, i32 %2110)
  %2502 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2501, <4 x i32> %2433, <4 x i32> %2106, i32 %2356, i32 %2110)
  %2503 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2300, <4 x i32> %2435, <4 x i32> %2105, i32 %2356, i32 %2110)
  %2504 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2503, <4 x i32> %2437, <4 x i32> %2106, i32 %2356, i32 %2110)
  %2505 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2302, <4 x i32> %2439, <4 x i32> %2105, i32 %2361, i32 %2110)
  %2506 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2505, <4 x i32> %2441, <4 x i32> %2106, i32 %2361, i32 %2110)
  %2507 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2304, <4 x i32> %2443, <4 x i32> %2105, i32 %2361, i32 %2110)
  %2508 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2507, <4 x i32> %2445, <4 x i32> %2106, i32 %2361, i32 %2110)
  %2509 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2306, <4 x i32> %2447, <4 x i32> %2105, i32 %2366, i32 %2110)
  %2510 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2509, <4 x i32> %2449, <4 x i32> %2106, i32 %2366, i32 %2110)
  %2511 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2308, <4 x i32> %2451, <4 x i32> %2105, i32 %2366, i32 %2110)
  %2512 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2511, <4 x i32> %2453, <4 x i32> %2106, i32 %2366, i32 %2110)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2513 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2456, i32 %188, i32 2)
  %2514 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2459, i32 %188, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2515 = add i32 %211, 640
  %2516 = mul i32 %2515, 4
  %2517 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %2516, i32 %204, i32 0)
  %2518 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %2516, i32 %210, i32 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %2519 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %522
  %2520 = load <16 x i8>, ptr addrspace(3) %2519, align 1
  %2521 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %527
  %2522 = load <16 x i8>, ptr addrspace(3) %2521, align 1
  %2523 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %535
  %2524 = load <16 x i8>, ptr addrspace(3) %2523, align 1
  %2525 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %539
  %2526 = load <16 x i8>, ptr addrspace(3) %2525, align 1
  %2527 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %547
  %2528 = load <16 x i8>, ptr addrspace(3) %2527, align 1
  %2529 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %551
  %2530 = load <16 x i8>, ptr addrspace(3) %2529, align 1
  %2531 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %559
  %2532 = load <16 x i8>, ptr addrspace(3) %2531, align 1
  %2533 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %563
  %2534 = load <16 x i8>, ptr addrspace(3) %2533, align 1
  %2535 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %571
  %2536 = load <16 x i8>, ptr addrspace(3) %2535, align 1
  %2537 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %575
  %2538 = load <16 x i8>, ptr addrspace(3) %2537, align 1
  %2539 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %583
  %2540 = load <16 x i8>, ptr addrspace(3) %2539, align 1
  %2541 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %587
  %2542 = load <16 x i8>, ptr addrspace(3) %2541, align 1
  %2543 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %595
  %2544 = load <16 x i8>, ptr addrspace(3) %2543, align 1
  %2545 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %599
  %2546 = load <16 x i8>, ptr addrspace(3) %2545, align 1
  %2547 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %607
  %2548 = load <16 x i8>, ptr addrspace(3) %2547, align 1
  %2549 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %611
  %2550 = load <16 x i8>, ptr addrspace(3) %2549, align 1
  %2551 = add i64 %193, 576
  %2552 = add i64 %2551, %24
  %2553 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %2552
  %2554 = load <1 x i32>, ptr addrspace(3) %2553, align 4
  %2555 = extractelement <1 x i32> %2554, i64 0
  %2556 = add i64 %193, 2368
  %2557 = add i64 %2556, %24
  %2558 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %2557
  %2559 = load <1 x i32>, ptr addrspace(3) %2558, align 4
  %2560 = extractelement <1 x i32> %2559, i64 0
  %2561 = add i64 %193, 4160
  %2562 = add i64 %2561, %24
  %2563 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %2562
  %2564 = load <1 x i32>, ptr addrspace(3) %2563, align 4
  %2565 = extractelement <1 x i32> %2564, i64 0
  %2566 = add i64 %193, 5952
  %2567 = add i64 %2566, %24
  %2568 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %2567
  %2569 = load <1 x i32>, ptr addrspace(3) %2568, align 4
  %2570 = extractelement <1 x i32> %2569, i64 0
  %2571 = add i64 %77, 352
  %2572 = add i64 %2571, %71
  %2573 = trunc i64 %2572 to i32
  %2574 = mul i32 %2573, 4
  %2575 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2574, i32 0, i32 0)
  %2576 = bitcast <4 x i32> %2575 to <16 x i8>
  %2577 = add i64 %90, 352
  %2578 = add i64 %2577, %84
  %2579 = trunc i64 %2578 to i32
  %2580 = mul i32 %2579, 4
  %2581 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2580, i32 0, i32 0)
  %2582 = bitcast <4 x i32> %2581 to <16 x i8>
  %2583 = add i64 %103, 352
  %2584 = add i64 %2583, %97
  %2585 = trunc i64 %2584 to i32
  %2586 = mul i32 %2585, 4
  %2587 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2586, i32 0, i32 0)
  %2588 = bitcast <4 x i32> %2587 to <16 x i8>
  %2589 = add i64 %116, 352
  %2590 = add i64 %2589, %110
  %2591 = trunc i64 %2590 to i32
  %2592 = mul i32 %2591, 4
  %2593 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2592, i32 0, i32 0)
  %2594 = bitcast <4 x i32> %2593 to <16 x i8>
  %2595 = add i64 %129, 352
  %2596 = add i64 %2595, %123
  %2597 = trunc i64 %2596 to i32
  %2598 = mul i32 %2597, 4
  %2599 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2598, i32 0, i32 0)
  %2600 = bitcast <4 x i32> %2599 to <16 x i8>
  %2601 = add i64 %142, 352
  %2602 = add i64 %2601, %136
  %2603 = trunc i64 %2602 to i32
  %2604 = mul i32 %2603, 4
  %2605 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2604, i32 0, i32 0)
  %2606 = bitcast <4 x i32> %2605 to <16 x i8>
  %2607 = add i64 %155, 352
  %2608 = add i64 %2607, %149
  %2609 = trunc i64 %2608 to i32
  %2610 = mul i32 %2609, 4
  %2611 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2610, i32 0, i32 0)
  %2612 = bitcast <4 x i32> %2611 to <16 x i8>
  %2613 = add i64 %168, 352
  %2614 = add i64 %2613, %162
  %2615 = trunc i64 %2614 to i32
  %2616 = mul i32 %2615, 4
  %2617 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2616, i32 0, i32 0)
  %2618 = bitcast <4 x i32> %2617 to <16 x i8>
  %2619 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %680
  store <16 x i8> %2576, ptr addrspace(3) %2619, align 1
  %2620 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %682
  store <16 x i8> %2582, ptr addrspace(3) %2620, align 1
  %2621 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %684
  store <16 x i8> %2588, ptr addrspace(3) %2621, align 1
  %2622 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %686
  store <16 x i8> %2594, ptr addrspace(3) %2622, align 1
  %2623 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %688
  store <16 x i8> %2600, ptr addrspace(3) %2623, align 1
  %2624 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %690
  store <16 x i8> %2606, ptr addrspace(3) %2624, align 1
  %2625 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %692
  store <16 x i8> %2612, ptr addrspace(3) %2625, align 1
  %2626 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %694
  store <16 x i8> %2618, ptr addrspace(3) %2626, align 1
  %2627 = bitcast <16 x i8> %2520 to <4 x i32>
  %2628 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2426, <4 x i32> %2627, <4 x i32> %2253, i32 %2555, i32 %2313)
  %2629 = bitcast <16 x i8> %2522 to <4 x i32>
  %2630 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2628, <4 x i32> %2629, <4 x i32> %2256, i32 %2555, i32 %2313)
  %2631 = bitcast <16 x i8> %2524 to <4 x i32>
  %2632 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2430, <4 x i32> %2631, <4 x i32> %2253, i32 %2555, i32 %2313)
  %2633 = bitcast <16 x i8> %2526 to <4 x i32>
  %2634 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2632, <4 x i32> %2633, <4 x i32> %2256, i32 %2555, i32 %2313)
  %2635 = bitcast <16 x i8> %2528 to <4 x i32>
  %2636 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2434, <4 x i32> %2635, <4 x i32> %2253, i32 %2560, i32 %2313)
  %2637 = bitcast <16 x i8> %2530 to <4 x i32>
  %2638 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2636, <4 x i32> %2637, <4 x i32> %2256, i32 %2560, i32 %2313)
  %2639 = bitcast <16 x i8> %2532 to <4 x i32>
  %2640 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2438, <4 x i32> %2639, <4 x i32> %2253, i32 %2560, i32 %2313)
  %2641 = bitcast <16 x i8> %2534 to <4 x i32>
  %2642 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2640, <4 x i32> %2641, <4 x i32> %2256, i32 %2560, i32 %2313)
  %2643 = bitcast <16 x i8> %2536 to <4 x i32>
  %2644 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2442, <4 x i32> %2643, <4 x i32> %2253, i32 %2565, i32 %2313)
  %2645 = bitcast <16 x i8> %2538 to <4 x i32>
  %2646 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2644, <4 x i32> %2645, <4 x i32> %2256, i32 %2565, i32 %2313)
  %2647 = bitcast <16 x i8> %2540 to <4 x i32>
  %2648 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2446, <4 x i32> %2647, <4 x i32> %2253, i32 %2565, i32 %2313)
  %2649 = bitcast <16 x i8> %2542 to <4 x i32>
  %2650 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2648, <4 x i32> %2649, <4 x i32> %2256, i32 %2565, i32 %2313)
  %2651 = bitcast <16 x i8> %2544 to <4 x i32>
  %2652 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2450, <4 x i32> %2651, <4 x i32> %2253, i32 %2570, i32 %2313)
  %2653 = bitcast <16 x i8> %2546 to <4 x i32>
  %2654 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2652, <4 x i32> %2653, <4 x i32> %2256, i32 %2570, i32 %2313)
  %2655 = bitcast <16 x i8> %2548 to <4 x i32>
  %2656 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2454, <4 x i32> %2655, <4 x i32> %2253, i32 %2570, i32 %2313)
  %2657 = bitcast <16 x i8> %2550 to <4 x i32>
  %2658 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2656, <4 x i32> %2657, <4 x i32> %2256, i32 %2570, i32 %2313)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2659 = add i32 %192, 5632
  %2660 = mul i32 %2659, 4
  %2661 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2660, i32 %176, i32 2)
  %2662 = add i32 %192, 5888
  %2663 = mul i32 %2662, 4
  %2664 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2663, i32 %176, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2665 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2462, <4 x i32> %2627, <4 x i32> %2273, i32 %2555, i32 %2313)
  %2666 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2665, <4 x i32> %2629, <4 x i32> %2274, i32 %2555, i32 %2313)
  %2667 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2464, <4 x i32> %2631, <4 x i32> %2273, i32 %2555, i32 %2313)
  %2668 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2667, <4 x i32> %2633, <4 x i32> %2274, i32 %2555, i32 %2313)
  %2669 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2466, <4 x i32> %2635, <4 x i32> %2273, i32 %2560, i32 %2313)
  %2670 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2669, <4 x i32> %2637, <4 x i32> %2274, i32 %2560, i32 %2313)
  %2671 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2468, <4 x i32> %2639, <4 x i32> %2273, i32 %2560, i32 %2313)
  %2672 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2671, <4 x i32> %2641, <4 x i32> %2274, i32 %2560, i32 %2313)
  %2673 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2470, <4 x i32> %2643, <4 x i32> %2273, i32 %2565, i32 %2313)
  %2674 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2673, <4 x i32> %2645, <4 x i32> %2274, i32 %2565, i32 %2313)
  %2675 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2472, <4 x i32> %2647, <4 x i32> %2273, i32 %2565, i32 %2313)
  %2676 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2675, <4 x i32> %2649, <4 x i32> %2274, i32 %2565, i32 %2313)
  %2677 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2474, <4 x i32> %2651, <4 x i32> %2273, i32 %2570, i32 %2313)
  %2678 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2677, <4 x i32> %2653, <4 x i32> %2274, i32 %2570, i32 %2313)
  %2679 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2476, <4 x i32> %2655, <4 x i32> %2273, i32 %2570, i32 %2313)
  %2680 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2679, <4 x i32> %2657, <4 x i32> %2274, i32 %2570, i32 %2313)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2681 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2660, i32 %180, i32 2)
  %2682 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2663, i32 %180, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2683 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2480, <4 x i32> %2627, <4 x i32> %2291, i32 %2555, i32 %2314)
  %2684 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2683, <4 x i32> %2629, <4 x i32> %2292, i32 %2555, i32 %2314)
  %2685 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2482, <4 x i32> %2631, <4 x i32> %2291, i32 %2555, i32 %2314)
  %2686 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2685, <4 x i32> %2633, <4 x i32> %2292, i32 %2555, i32 %2314)
  %2687 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2484, <4 x i32> %2635, <4 x i32> %2291, i32 %2560, i32 %2314)
  %2688 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2687, <4 x i32> %2637, <4 x i32> %2292, i32 %2560, i32 %2314)
  %2689 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2486, <4 x i32> %2639, <4 x i32> %2291, i32 %2560, i32 %2314)
  %2690 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2689, <4 x i32> %2641, <4 x i32> %2292, i32 %2560, i32 %2314)
  %2691 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2488, <4 x i32> %2643, <4 x i32> %2291, i32 %2565, i32 %2314)
  %2692 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2691, <4 x i32> %2645, <4 x i32> %2292, i32 %2565, i32 %2314)
  %2693 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2490, <4 x i32> %2647, <4 x i32> %2291, i32 %2565, i32 %2314)
  %2694 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2693, <4 x i32> %2649, <4 x i32> %2292, i32 %2565, i32 %2314)
  %2695 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2492, <4 x i32> %2651, <4 x i32> %2291, i32 %2570, i32 %2314)
  %2696 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2695, <4 x i32> %2653, <4 x i32> %2292, i32 %2570, i32 %2314)
  %2697 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2494, <4 x i32> %2655, <4 x i32> %2291, i32 %2570, i32 %2314)
  %2698 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2697, <4 x i32> %2657, <4 x i32> %2292, i32 %2570, i32 %2314)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2699 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2660, i32 %184, i32 2)
  %2700 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2663, i32 %184, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2701 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2498, <4 x i32> %2627, <4 x i32> %2309, i32 %2555, i32 %2314)
  %2702 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2701, <4 x i32> %2629, <4 x i32> %2310, i32 %2555, i32 %2314)
  %2703 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2500, <4 x i32> %2631, <4 x i32> %2309, i32 %2555, i32 %2314)
  %2704 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2703, <4 x i32> %2633, <4 x i32> %2310, i32 %2555, i32 %2314)
  %2705 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2502, <4 x i32> %2635, <4 x i32> %2309, i32 %2560, i32 %2314)
  %2706 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2705, <4 x i32> %2637, <4 x i32> %2310, i32 %2560, i32 %2314)
  %2707 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2504, <4 x i32> %2639, <4 x i32> %2309, i32 %2560, i32 %2314)
  %2708 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2707, <4 x i32> %2641, <4 x i32> %2310, i32 %2560, i32 %2314)
  %2709 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2506, <4 x i32> %2643, <4 x i32> %2309, i32 %2565, i32 %2314)
  %2710 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2709, <4 x i32> %2645, <4 x i32> %2310, i32 %2565, i32 %2314)
  %2711 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2508, <4 x i32> %2647, <4 x i32> %2309, i32 %2565, i32 %2314)
  %2712 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2711, <4 x i32> %2649, <4 x i32> %2310, i32 %2565, i32 %2314)
  %2713 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2510, <4 x i32> %2651, <4 x i32> %2309, i32 %2570, i32 %2314)
  %2714 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2713, <4 x i32> %2653, <4 x i32> %2310, i32 %2570, i32 %2314)
  %2715 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2512, <4 x i32> %2655, <4 x i32> %2309, i32 %2570, i32 %2314)
  %2716 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2715, <4 x i32> %2657, <4 x i32> %2310, i32 %2570, i32 %2314)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2717 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2660, i32 %188, i32 2)
  %2718 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2663, i32 %188, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2719 = add i32 %211, 704
  %2720 = mul i32 %2719, 4
  %2721 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %2720, i32 %204, i32 0)
  %2722 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %2720, i32 %210, i32 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %2723 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %856
  %2724 = load <16 x i8>, ptr addrspace(3) %2723, align 1
  %2725 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %859
  %2726 = load <16 x i8>, ptr addrspace(3) %2725, align 1
  %2727 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %862
  %2728 = load <16 x i8>, ptr addrspace(3) %2727, align 1
  %2729 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %865
  %2730 = load <16 x i8>, ptr addrspace(3) %2729, align 1
  %2731 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %868
  %2732 = load <16 x i8>, ptr addrspace(3) %2731, align 1
  %2733 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %871
  %2734 = load <16 x i8>, ptr addrspace(3) %2733, align 1
  %2735 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %874
  %2736 = load <16 x i8>, ptr addrspace(3) %2735, align 1
  %2737 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %877
  %2738 = load <16 x i8>, ptr addrspace(3) %2737, align 1
  %2739 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %880
  %2740 = load <16 x i8>, ptr addrspace(3) %2739, align 1
  %2741 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %883
  %2742 = load <16 x i8>, ptr addrspace(3) %2741, align 1
  %2743 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %886
  %2744 = load <16 x i8>, ptr addrspace(3) %2743, align 1
  %2745 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %889
  %2746 = load <16 x i8>, ptr addrspace(3) %2745, align 1
  %2747 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %892
  %2748 = load <16 x i8>, ptr addrspace(3) %2747, align 1
  %2749 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %895
  %2750 = load <16 x i8>, ptr addrspace(3) %2749, align 1
  %2751 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %898
  %2752 = load <16 x i8>, ptr addrspace(3) %2751, align 1
  %2753 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %901
  %2754 = load <16 x i8>, ptr addrspace(3) %2753, align 1
  %2755 = add i64 %193, 640
  %2756 = add i64 %2755, %24
  %2757 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %2756
  %2758 = load <1 x i32>, ptr addrspace(3) %2757, align 4
  %2759 = extractelement <1 x i32> %2758, i64 0
  %2760 = add i64 %193, 2432
  %2761 = add i64 %2760, %24
  %2762 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %2761
  %2763 = load <1 x i32>, ptr addrspace(3) %2762, align 4
  %2764 = extractelement <1 x i32> %2763, i64 0
  %2765 = add i64 %193, 4224
  %2766 = add i64 %2765, %24
  %2767 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %2766
  %2768 = load <1 x i32>, ptr addrspace(3) %2767, align 4
  %2769 = extractelement <1 x i32> %2768, i64 0
  %2770 = add i64 %193, 6016
  %2771 = add i64 %2770, %24
  %2772 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %2771
  %2773 = load <1 x i32>, ptr addrspace(3) %2772, align 4
  %2774 = extractelement <1 x i32> %2773, i64 0
  %2775 = add i64 %77, 384
  %2776 = add i64 %2775, %71
  %2777 = trunc i64 %2776 to i32
  %2778 = mul i32 %2777, 4
  %2779 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2778, i32 0, i32 0)
  %2780 = bitcast <4 x i32> %2779 to <16 x i8>
  %2781 = add i64 %90, 384
  %2782 = add i64 %2781, %84
  %2783 = trunc i64 %2782 to i32
  %2784 = mul i32 %2783, 4
  %2785 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2784, i32 0, i32 0)
  %2786 = bitcast <4 x i32> %2785 to <16 x i8>
  %2787 = add i64 %103, 384
  %2788 = add i64 %2787, %97
  %2789 = trunc i64 %2788 to i32
  %2790 = mul i32 %2789, 4
  %2791 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2790, i32 0, i32 0)
  %2792 = bitcast <4 x i32> %2791 to <16 x i8>
  %2793 = add i64 %116, 384
  %2794 = add i64 %2793, %110
  %2795 = trunc i64 %2794 to i32
  %2796 = mul i32 %2795, 4
  %2797 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2796, i32 0, i32 0)
  %2798 = bitcast <4 x i32> %2797 to <16 x i8>
  %2799 = add i64 %129, 384
  %2800 = add i64 %2799, %123
  %2801 = trunc i64 %2800 to i32
  %2802 = mul i32 %2801, 4
  %2803 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2802, i32 0, i32 0)
  %2804 = bitcast <4 x i32> %2803 to <16 x i8>
  %2805 = add i64 %142, 384
  %2806 = add i64 %2805, %136
  %2807 = trunc i64 %2806 to i32
  %2808 = mul i32 %2807, 4
  %2809 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2808, i32 0, i32 0)
  %2810 = bitcast <4 x i32> %2809 to <16 x i8>
  %2811 = add i64 %155, 384
  %2812 = add i64 %2811, %149
  %2813 = trunc i64 %2812 to i32
  %2814 = mul i32 %2813, 4
  %2815 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2814, i32 0, i32 0)
  %2816 = bitcast <4 x i32> %2815 to <16 x i8>
  %2817 = add i64 %168, 384
  %2818 = add i64 %2817, %162
  %2819 = trunc i64 %2818 to i32
  %2820 = mul i32 %2819, 4
  %2821 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2820, i32 0, i32 0)
  %2822 = bitcast <4 x i32> %2821 to <16 x i8>
  %2823 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %360
  store <16 x i8> %2780, ptr addrspace(3) %2823, align 1
  %2824 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %369
  store <16 x i8> %2786, ptr addrspace(3) %2824, align 1
  %2825 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %378
  store <16 x i8> %2792, ptr addrspace(3) %2825, align 1
  %2826 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %387
  store <16 x i8> %2798, ptr addrspace(3) %2826, align 1
  %2827 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %396
  store <16 x i8> %2804, ptr addrspace(3) %2827, align 1
  %2828 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %405
  store <16 x i8> %2810, ptr addrspace(3) %2828, align 1
  %2829 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %414
  store <16 x i8> %2816, ptr addrspace(3) %2829, align 1
  %2830 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %423
  store <16 x i8> %2822, ptr addrspace(3) %2830, align 1
  %2831 = bitcast <16 x i8> %2724 to <4 x i32>
  %2832 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2630, <4 x i32> %2831, <4 x i32> %2457, i32 %2759, i32 %2517)
  %2833 = bitcast <16 x i8> %2726 to <4 x i32>
  %2834 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2832, <4 x i32> %2833, <4 x i32> %2460, i32 %2759, i32 %2517)
  %2835 = bitcast <16 x i8> %2728 to <4 x i32>
  %2836 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2634, <4 x i32> %2835, <4 x i32> %2457, i32 %2759, i32 %2517)
  %2837 = bitcast <16 x i8> %2730 to <4 x i32>
  %2838 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2836, <4 x i32> %2837, <4 x i32> %2460, i32 %2759, i32 %2517)
  %2839 = bitcast <16 x i8> %2732 to <4 x i32>
  %2840 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2638, <4 x i32> %2839, <4 x i32> %2457, i32 %2764, i32 %2517)
  %2841 = bitcast <16 x i8> %2734 to <4 x i32>
  %2842 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2840, <4 x i32> %2841, <4 x i32> %2460, i32 %2764, i32 %2517)
  %2843 = bitcast <16 x i8> %2736 to <4 x i32>
  %2844 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2642, <4 x i32> %2843, <4 x i32> %2457, i32 %2764, i32 %2517)
  %2845 = bitcast <16 x i8> %2738 to <4 x i32>
  %2846 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2844, <4 x i32> %2845, <4 x i32> %2460, i32 %2764, i32 %2517)
  %2847 = bitcast <16 x i8> %2740 to <4 x i32>
  %2848 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2646, <4 x i32> %2847, <4 x i32> %2457, i32 %2769, i32 %2517)
  %2849 = bitcast <16 x i8> %2742 to <4 x i32>
  %2850 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2848, <4 x i32> %2849, <4 x i32> %2460, i32 %2769, i32 %2517)
  %2851 = bitcast <16 x i8> %2744 to <4 x i32>
  %2852 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2650, <4 x i32> %2851, <4 x i32> %2457, i32 %2769, i32 %2517)
  %2853 = bitcast <16 x i8> %2746 to <4 x i32>
  %2854 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2852, <4 x i32> %2853, <4 x i32> %2460, i32 %2769, i32 %2517)
  %2855 = bitcast <16 x i8> %2748 to <4 x i32>
  %2856 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2654, <4 x i32> %2855, <4 x i32> %2457, i32 %2774, i32 %2517)
  %2857 = bitcast <16 x i8> %2750 to <4 x i32>
  %2858 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2856, <4 x i32> %2857, <4 x i32> %2460, i32 %2774, i32 %2517)
  %2859 = bitcast <16 x i8> %2752 to <4 x i32>
  %2860 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2658, <4 x i32> %2859, <4 x i32> %2457, i32 %2774, i32 %2517)
  %2861 = bitcast <16 x i8> %2754 to <4 x i32>
  %2862 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2860, <4 x i32> %2861, <4 x i32> %2460, i32 %2774, i32 %2517)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2863 = add i32 %192, 6144
  %2864 = mul i32 %2863, 4
  %2865 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2864, i32 %176, i32 2)
  %2866 = add i32 %192, 6400
  %2867 = mul i32 %2866, 4
  %2868 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2867, i32 %176, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2869 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2666, <4 x i32> %2831, <4 x i32> %2477, i32 %2759, i32 %2517)
  %2870 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2869, <4 x i32> %2833, <4 x i32> %2478, i32 %2759, i32 %2517)
  %2871 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2668, <4 x i32> %2835, <4 x i32> %2477, i32 %2759, i32 %2517)
  %2872 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2871, <4 x i32> %2837, <4 x i32> %2478, i32 %2759, i32 %2517)
  %2873 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2670, <4 x i32> %2839, <4 x i32> %2477, i32 %2764, i32 %2517)
  %2874 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2873, <4 x i32> %2841, <4 x i32> %2478, i32 %2764, i32 %2517)
  %2875 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2672, <4 x i32> %2843, <4 x i32> %2477, i32 %2764, i32 %2517)
  %2876 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2875, <4 x i32> %2845, <4 x i32> %2478, i32 %2764, i32 %2517)
  %2877 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2674, <4 x i32> %2847, <4 x i32> %2477, i32 %2769, i32 %2517)
  %2878 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2877, <4 x i32> %2849, <4 x i32> %2478, i32 %2769, i32 %2517)
  %2879 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2676, <4 x i32> %2851, <4 x i32> %2477, i32 %2769, i32 %2517)
  %2880 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2879, <4 x i32> %2853, <4 x i32> %2478, i32 %2769, i32 %2517)
  %2881 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2678, <4 x i32> %2855, <4 x i32> %2477, i32 %2774, i32 %2517)
  %2882 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2881, <4 x i32> %2857, <4 x i32> %2478, i32 %2774, i32 %2517)
  %2883 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2680, <4 x i32> %2859, <4 x i32> %2477, i32 %2774, i32 %2517)
  %2884 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2883, <4 x i32> %2861, <4 x i32> %2478, i32 %2774, i32 %2517)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2885 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2864, i32 %180, i32 2)
  %2886 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2867, i32 %180, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2887 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2684, <4 x i32> %2831, <4 x i32> %2495, i32 %2759, i32 %2518)
  %2888 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2887, <4 x i32> %2833, <4 x i32> %2496, i32 %2759, i32 %2518)
  %2889 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2686, <4 x i32> %2835, <4 x i32> %2495, i32 %2759, i32 %2518)
  %2890 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2889, <4 x i32> %2837, <4 x i32> %2496, i32 %2759, i32 %2518)
  %2891 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2688, <4 x i32> %2839, <4 x i32> %2495, i32 %2764, i32 %2518)
  %2892 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2891, <4 x i32> %2841, <4 x i32> %2496, i32 %2764, i32 %2518)
  %2893 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2690, <4 x i32> %2843, <4 x i32> %2495, i32 %2764, i32 %2518)
  %2894 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2893, <4 x i32> %2845, <4 x i32> %2496, i32 %2764, i32 %2518)
  %2895 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2692, <4 x i32> %2847, <4 x i32> %2495, i32 %2769, i32 %2518)
  %2896 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2895, <4 x i32> %2849, <4 x i32> %2496, i32 %2769, i32 %2518)
  %2897 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2694, <4 x i32> %2851, <4 x i32> %2495, i32 %2769, i32 %2518)
  %2898 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2897, <4 x i32> %2853, <4 x i32> %2496, i32 %2769, i32 %2518)
  %2899 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2696, <4 x i32> %2855, <4 x i32> %2495, i32 %2774, i32 %2518)
  %2900 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2899, <4 x i32> %2857, <4 x i32> %2496, i32 %2774, i32 %2518)
  %2901 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2698, <4 x i32> %2859, <4 x i32> %2495, i32 %2774, i32 %2518)
  %2902 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2901, <4 x i32> %2861, <4 x i32> %2496, i32 %2774, i32 %2518)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2903 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2864, i32 %184, i32 2)
  %2904 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2867, i32 %184, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2905 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2702, <4 x i32> %2831, <4 x i32> %2513, i32 %2759, i32 %2518)
  %2906 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2905, <4 x i32> %2833, <4 x i32> %2514, i32 %2759, i32 %2518)
  %2907 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2704, <4 x i32> %2835, <4 x i32> %2513, i32 %2759, i32 %2518)
  %2908 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2907, <4 x i32> %2837, <4 x i32> %2514, i32 %2759, i32 %2518)
  %2909 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2706, <4 x i32> %2839, <4 x i32> %2513, i32 %2764, i32 %2518)
  %2910 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2909, <4 x i32> %2841, <4 x i32> %2514, i32 %2764, i32 %2518)
  %2911 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2708, <4 x i32> %2843, <4 x i32> %2513, i32 %2764, i32 %2518)
  %2912 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2911, <4 x i32> %2845, <4 x i32> %2514, i32 %2764, i32 %2518)
  %2913 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2710, <4 x i32> %2847, <4 x i32> %2513, i32 %2769, i32 %2518)
  %2914 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2913, <4 x i32> %2849, <4 x i32> %2514, i32 %2769, i32 %2518)
  %2915 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2712, <4 x i32> %2851, <4 x i32> %2513, i32 %2769, i32 %2518)
  %2916 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2915, <4 x i32> %2853, <4 x i32> %2514, i32 %2769, i32 %2518)
  %2917 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2714, <4 x i32> %2855, <4 x i32> %2513, i32 %2774, i32 %2518)
  %2918 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2917, <4 x i32> %2857, <4 x i32> %2514, i32 %2774, i32 %2518)
  %2919 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2716, <4 x i32> %2859, <4 x i32> %2513, i32 %2774, i32 %2518)
  %2920 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2919, <4 x i32> %2861, <4 x i32> %2514, i32 %2774, i32 %2518)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2921 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2864, i32 %188, i32 2)
  %2922 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %2867, i32 %188, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2923 = add i32 %211, 768
  %2924 = mul i32 %2923, 4
  %2925 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %2924, i32 %204, i32 0)
  %2926 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %2924, i32 %210, i32 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %2927 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1075
  %2928 = load <16 x i8>, ptr addrspace(3) %2927, align 1
  %2929 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1078
  %2930 = load <16 x i8>, ptr addrspace(3) %2929, align 1
  %2931 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1081
  %2932 = load <16 x i8>, ptr addrspace(3) %2931, align 1
  %2933 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1084
  %2934 = load <16 x i8>, ptr addrspace(3) %2933, align 1
  %2935 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1087
  %2936 = load <16 x i8>, ptr addrspace(3) %2935, align 1
  %2937 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1090
  %2938 = load <16 x i8>, ptr addrspace(3) %2937, align 1
  %2939 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1093
  %2940 = load <16 x i8>, ptr addrspace(3) %2939, align 1
  %2941 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1096
  %2942 = load <16 x i8>, ptr addrspace(3) %2941, align 1
  %2943 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1099
  %2944 = load <16 x i8>, ptr addrspace(3) %2943, align 1
  %2945 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1102
  %2946 = load <16 x i8>, ptr addrspace(3) %2945, align 1
  %2947 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1105
  %2948 = load <16 x i8>, ptr addrspace(3) %2947, align 1
  %2949 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1108
  %2950 = load <16 x i8>, ptr addrspace(3) %2949, align 1
  %2951 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1111
  %2952 = load <16 x i8>, ptr addrspace(3) %2951, align 1
  %2953 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1114
  %2954 = load <16 x i8>, ptr addrspace(3) %2953, align 1
  %2955 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1117
  %2956 = load <16 x i8>, ptr addrspace(3) %2955, align 1
  %2957 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1120
  %2958 = load <16 x i8>, ptr addrspace(3) %2957, align 1
  %2959 = add i64 %193, 704
  %2960 = add i64 %2959, %24
  %2961 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %2960
  %2962 = load <1 x i32>, ptr addrspace(3) %2961, align 4
  %2963 = extractelement <1 x i32> %2962, i64 0
  %2964 = add i64 %193, 2496
  %2965 = add i64 %2964, %24
  %2966 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %2965
  %2967 = load <1 x i32>, ptr addrspace(3) %2966, align 4
  %2968 = extractelement <1 x i32> %2967, i64 0
  %2969 = add i64 %193, 4288
  %2970 = add i64 %2969, %24
  %2971 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %2970
  %2972 = load <1 x i32>, ptr addrspace(3) %2971, align 4
  %2973 = extractelement <1 x i32> %2972, i64 0
  %2974 = add i64 %193, 6080
  %2975 = add i64 %2974, %24
  %2976 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %2975
  %2977 = load <1 x i32>, ptr addrspace(3) %2976, align 4
  %2978 = extractelement <1 x i32> %2977, i64 0
  %2979 = add i64 %77, 416
  %2980 = add i64 %2979, %71
  %2981 = trunc i64 %2980 to i32
  %2982 = mul i32 %2981, 4
  %2983 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2982, i32 0, i32 0)
  %2984 = bitcast <4 x i32> %2983 to <16 x i8>
  %2985 = add i64 %90, 416
  %2986 = add i64 %2985, %84
  %2987 = trunc i64 %2986 to i32
  %2988 = mul i32 %2987, 4
  %2989 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2988, i32 0, i32 0)
  %2990 = bitcast <4 x i32> %2989 to <16 x i8>
  %2991 = add i64 %103, 416
  %2992 = add i64 %2991, %97
  %2993 = trunc i64 %2992 to i32
  %2994 = mul i32 %2993, 4
  %2995 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %2994, i32 0, i32 0)
  %2996 = bitcast <4 x i32> %2995 to <16 x i8>
  %2997 = add i64 %116, 416
  %2998 = add i64 %2997, %110
  %2999 = trunc i64 %2998 to i32
  %3000 = mul i32 %2999, 4
  %3001 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3000, i32 0, i32 0)
  %3002 = bitcast <4 x i32> %3001 to <16 x i8>
  %3003 = add i64 %129, 416
  %3004 = add i64 %3003, %123
  %3005 = trunc i64 %3004 to i32
  %3006 = mul i32 %3005, 4
  %3007 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3006, i32 0, i32 0)
  %3008 = bitcast <4 x i32> %3007 to <16 x i8>
  %3009 = add i64 %142, 416
  %3010 = add i64 %3009, %136
  %3011 = trunc i64 %3010 to i32
  %3012 = mul i32 %3011, 4
  %3013 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3012, i32 0, i32 0)
  %3014 = bitcast <4 x i32> %3013 to <16 x i8>
  %3015 = add i64 %155, 416
  %3016 = add i64 %3015, %149
  %3017 = trunc i64 %3016 to i32
  %3018 = mul i32 %3017, 4
  %3019 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3018, i32 0, i32 0)
  %3020 = bitcast <4 x i32> %3019 to <16 x i8>
  %3021 = add i64 %168, 416
  %3022 = add i64 %3021, %162
  %3023 = trunc i64 %3022 to i32
  %3024 = mul i32 %3023, 4
  %3025 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3024, i32 0, i32 0)
  %3026 = bitcast <4 x i32> %3025 to <16 x i8>
  %3027 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %487
  store <16 x i8> %2984, ptr addrspace(3) %3027, align 1
  %3028 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %489
  store <16 x i8> %2990, ptr addrspace(3) %3028, align 1
  %3029 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %491
  store <16 x i8> %2996, ptr addrspace(3) %3029, align 1
  %3030 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %493
  store <16 x i8> %3002, ptr addrspace(3) %3030, align 1
  %3031 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %495
  store <16 x i8> %3008, ptr addrspace(3) %3031, align 1
  %3032 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %497
  store <16 x i8> %3014, ptr addrspace(3) %3032, align 1
  %3033 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %499
  store <16 x i8> %3020, ptr addrspace(3) %3033, align 1
  %3034 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %501
  store <16 x i8> %3026, ptr addrspace(3) %3034, align 1
  %3035 = bitcast <16 x i8> %2928 to <4 x i32>
  %3036 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2834, <4 x i32> %3035, <4 x i32> %2661, i32 %2963, i32 %2721)
  %3037 = bitcast <16 x i8> %2930 to <4 x i32>
  %3038 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3036, <4 x i32> %3037, <4 x i32> %2664, i32 %2963, i32 %2721)
  %3039 = bitcast <16 x i8> %2932 to <4 x i32>
  %3040 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2838, <4 x i32> %3039, <4 x i32> %2661, i32 %2963, i32 %2721)
  %3041 = bitcast <16 x i8> %2934 to <4 x i32>
  %3042 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3040, <4 x i32> %3041, <4 x i32> %2664, i32 %2963, i32 %2721)
  %3043 = bitcast <16 x i8> %2936 to <4 x i32>
  %3044 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2842, <4 x i32> %3043, <4 x i32> %2661, i32 %2968, i32 %2721)
  %3045 = bitcast <16 x i8> %2938 to <4 x i32>
  %3046 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3044, <4 x i32> %3045, <4 x i32> %2664, i32 %2968, i32 %2721)
  %3047 = bitcast <16 x i8> %2940 to <4 x i32>
  %3048 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2846, <4 x i32> %3047, <4 x i32> %2661, i32 %2968, i32 %2721)
  %3049 = bitcast <16 x i8> %2942 to <4 x i32>
  %3050 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3048, <4 x i32> %3049, <4 x i32> %2664, i32 %2968, i32 %2721)
  %3051 = bitcast <16 x i8> %2944 to <4 x i32>
  %3052 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2850, <4 x i32> %3051, <4 x i32> %2661, i32 %2973, i32 %2721)
  %3053 = bitcast <16 x i8> %2946 to <4 x i32>
  %3054 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3052, <4 x i32> %3053, <4 x i32> %2664, i32 %2973, i32 %2721)
  %3055 = bitcast <16 x i8> %2948 to <4 x i32>
  %3056 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2854, <4 x i32> %3055, <4 x i32> %2661, i32 %2973, i32 %2721)
  %3057 = bitcast <16 x i8> %2950 to <4 x i32>
  %3058 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3056, <4 x i32> %3057, <4 x i32> %2664, i32 %2973, i32 %2721)
  %3059 = bitcast <16 x i8> %2952 to <4 x i32>
  %3060 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2858, <4 x i32> %3059, <4 x i32> %2661, i32 %2978, i32 %2721)
  %3061 = bitcast <16 x i8> %2954 to <4 x i32>
  %3062 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3060, <4 x i32> %3061, <4 x i32> %2664, i32 %2978, i32 %2721)
  %3063 = bitcast <16 x i8> %2956 to <4 x i32>
  %3064 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2862, <4 x i32> %3063, <4 x i32> %2661, i32 %2978, i32 %2721)
  %3065 = bitcast <16 x i8> %2958 to <4 x i32>
  %3066 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3064, <4 x i32> %3065, <4 x i32> %2664, i32 %2978, i32 %2721)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3067 = add i32 %192, 6656
  %3068 = mul i32 %3067, 4
  %3069 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3068, i32 %176, i32 2)
  %3070 = add i32 %192, 6912
  %3071 = mul i32 %3070, 4
  %3072 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3071, i32 %176, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3073 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2870, <4 x i32> %3035, <4 x i32> %2681, i32 %2963, i32 %2721)
  %3074 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3073, <4 x i32> %3037, <4 x i32> %2682, i32 %2963, i32 %2721)
  %3075 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2872, <4 x i32> %3039, <4 x i32> %2681, i32 %2963, i32 %2721)
  %3076 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3075, <4 x i32> %3041, <4 x i32> %2682, i32 %2963, i32 %2721)
  %3077 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2874, <4 x i32> %3043, <4 x i32> %2681, i32 %2968, i32 %2721)
  %3078 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3077, <4 x i32> %3045, <4 x i32> %2682, i32 %2968, i32 %2721)
  %3079 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2876, <4 x i32> %3047, <4 x i32> %2681, i32 %2968, i32 %2721)
  %3080 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3079, <4 x i32> %3049, <4 x i32> %2682, i32 %2968, i32 %2721)
  %3081 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2878, <4 x i32> %3051, <4 x i32> %2681, i32 %2973, i32 %2721)
  %3082 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3081, <4 x i32> %3053, <4 x i32> %2682, i32 %2973, i32 %2721)
  %3083 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2880, <4 x i32> %3055, <4 x i32> %2681, i32 %2973, i32 %2721)
  %3084 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3083, <4 x i32> %3057, <4 x i32> %2682, i32 %2973, i32 %2721)
  %3085 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2882, <4 x i32> %3059, <4 x i32> %2681, i32 %2978, i32 %2721)
  %3086 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3085, <4 x i32> %3061, <4 x i32> %2682, i32 %2978, i32 %2721)
  %3087 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2884, <4 x i32> %3063, <4 x i32> %2681, i32 %2978, i32 %2721)
  %3088 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3087, <4 x i32> %3065, <4 x i32> %2682, i32 %2978, i32 %2721)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3089 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3068, i32 %180, i32 2)
  %3090 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3071, i32 %180, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3091 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2888, <4 x i32> %3035, <4 x i32> %2699, i32 %2963, i32 %2722)
  %3092 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3091, <4 x i32> %3037, <4 x i32> %2700, i32 %2963, i32 %2722)
  %3093 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2890, <4 x i32> %3039, <4 x i32> %2699, i32 %2963, i32 %2722)
  %3094 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3093, <4 x i32> %3041, <4 x i32> %2700, i32 %2963, i32 %2722)
  %3095 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2892, <4 x i32> %3043, <4 x i32> %2699, i32 %2968, i32 %2722)
  %3096 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3095, <4 x i32> %3045, <4 x i32> %2700, i32 %2968, i32 %2722)
  %3097 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2894, <4 x i32> %3047, <4 x i32> %2699, i32 %2968, i32 %2722)
  %3098 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3097, <4 x i32> %3049, <4 x i32> %2700, i32 %2968, i32 %2722)
  %3099 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2896, <4 x i32> %3051, <4 x i32> %2699, i32 %2973, i32 %2722)
  %3100 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3099, <4 x i32> %3053, <4 x i32> %2700, i32 %2973, i32 %2722)
  %3101 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2898, <4 x i32> %3055, <4 x i32> %2699, i32 %2973, i32 %2722)
  %3102 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3101, <4 x i32> %3057, <4 x i32> %2700, i32 %2973, i32 %2722)
  %3103 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2900, <4 x i32> %3059, <4 x i32> %2699, i32 %2978, i32 %2722)
  %3104 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3103, <4 x i32> %3061, <4 x i32> %2700, i32 %2978, i32 %2722)
  %3105 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2902, <4 x i32> %3063, <4 x i32> %2699, i32 %2978, i32 %2722)
  %3106 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3105, <4 x i32> %3065, <4 x i32> %2700, i32 %2978, i32 %2722)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3107 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3068, i32 %184, i32 2)
  %3108 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3071, i32 %184, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3109 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2906, <4 x i32> %3035, <4 x i32> %2717, i32 %2963, i32 %2722)
  %3110 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3109, <4 x i32> %3037, <4 x i32> %2718, i32 %2963, i32 %2722)
  %3111 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2908, <4 x i32> %3039, <4 x i32> %2717, i32 %2963, i32 %2722)
  %3112 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3111, <4 x i32> %3041, <4 x i32> %2718, i32 %2963, i32 %2722)
  %3113 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2910, <4 x i32> %3043, <4 x i32> %2717, i32 %2968, i32 %2722)
  %3114 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3113, <4 x i32> %3045, <4 x i32> %2718, i32 %2968, i32 %2722)
  %3115 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2912, <4 x i32> %3047, <4 x i32> %2717, i32 %2968, i32 %2722)
  %3116 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3115, <4 x i32> %3049, <4 x i32> %2718, i32 %2968, i32 %2722)
  %3117 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2914, <4 x i32> %3051, <4 x i32> %2717, i32 %2973, i32 %2722)
  %3118 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3117, <4 x i32> %3053, <4 x i32> %2718, i32 %2973, i32 %2722)
  %3119 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2916, <4 x i32> %3055, <4 x i32> %2717, i32 %2973, i32 %2722)
  %3120 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3119, <4 x i32> %3057, <4 x i32> %2718, i32 %2973, i32 %2722)
  %3121 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2918, <4 x i32> %3059, <4 x i32> %2717, i32 %2978, i32 %2722)
  %3122 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3121, <4 x i32> %3061, <4 x i32> %2718, i32 %2978, i32 %2722)
  %3123 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %2920, <4 x i32> %3063, <4 x i32> %2717, i32 %2978, i32 %2722)
  %3124 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3123, <4 x i32> %3065, <4 x i32> %2718, i32 %2978, i32 %2722)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3125 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3068, i32 %188, i32 2)
  %3126 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3071, i32 %188, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3127 = add i32 %211, 832
  %3128 = mul i32 %3127, 4
  %3129 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %3128, i32 %204, i32 0)
  %3130 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %3128, i32 %210, i32 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %3131 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %522
  %3132 = load <16 x i8>, ptr addrspace(3) %3131, align 1
  %3133 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %527
  %3134 = load <16 x i8>, ptr addrspace(3) %3133, align 1
  %3135 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %535
  %3136 = load <16 x i8>, ptr addrspace(3) %3135, align 1
  %3137 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %539
  %3138 = load <16 x i8>, ptr addrspace(3) %3137, align 1
  %3139 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %547
  %3140 = load <16 x i8>, ptr addrspace(3) %3139, align 1
  %3141 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %551
  %3142 = load <16 x i8>, ptr addrspace(3) %3141, align 1
  %3143 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %559
  %3144 = load <16 x i8>, ptr addrspace(3) %3143, align 1
  %3145 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %563
  %3146 = load <16 x i8>, ptr addrspace(3) %3145, align 1
  %3147 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %571
  %3148 = load <16 x i8>, ptr addrspace(3) %3147, align 1
  %3149 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %575
  %3150 = load <16 x i8>, ptr addrspace(3) %3149, align 1
  %3151 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %583
  %3152 = load <16 x i8>, ptr addrspace(3) %3151, align 1
  %3153 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %587
  %3154 = load <16 x i8>, ptr addrspace(3) %3153, align 1
  %3155 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %595
  %3156 = load <16 x i8>, ptr addrspace(3) %3155, align 1
  %3157 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %599
  %3158 = load <16 x i8>, ptr addrspace(3) %3157, align 1
  %3159 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %607
  %3160 = load <16 x i8>, ptr addrspace(3) %3159, align 1
  %3161 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %611
  %3162 = load <16 x i8>, ptr addrspace(3) %3161, align 1
  %3163 = add i64 %193, 768
  %3164 = add i64 %3163, %24
  %3165 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %3164
  %3166 = load <1 x i32>, ptr addrspace(3) %3165, align 4
  %3167 = extractelement <1 x i32> %3166, i64 0
  %3168 = add i64 %193, 2560
  %3169 = add i64 %3168, %24
  %3170 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %3169
  %3171 = load <1 x i32>, ptr addrspace(3) %3170, align 4
  %3172 = extractelement <1 x i32> %3171, i64 0
  %3173 = add i64 %193, 4352
  %3174 = add i64 %3173, %24
  %3175 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %3174
  %3176 = load <1 x i32>, ptr addrspace(3) %3175, align 4
  %3177 = extractelement <1 x i32> %3176, i64 0
  %3178 = add i64 %193, 6144
  %3179 = add i64 %3178, %24
  %3180 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %3179
  %3181 = load <1 x i32>, ptr addrspace(3) %3180, align 4
  %3182 = extractelement <1 x i32> %3181, i64 0
  %3183 = add i64 %77, 448
  %3184 = add i64 %3183, %71
  %3185 = trunc i64 %3184 to i32
  %3186 = mul i32 %3185, 4
  %3187 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3186, i32 0, i32 0)
  %3188 = bitcast <4 x i32> %3187 to <16 x i8>
  %3189 = add i64 %90, 448
  %3190 = add i64 %3189, %84
  %3191 = trunc i64 %3190 to i32
  %3192 = mul i32 %3191, 4
  %3193 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3192, i32 0, i32 0)
  %3194 = bitcast <4 x i32> %3193 to <16 x i8>
  %3195 = add i64 %103, 448
  %3196 = add i64 %3195, %97
  %3197 = trunc i64 %3196 to i32
  %3198 = mul i32 %3197, 4
  %3199 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3198, i32 0, i32 0)
  %3200 = bitcast <4 x i32> %3199 to <16 x i8>
  %3201 = add i64 %116, 448
  %3202 = add i64 %3201, %110
  %3203 = trunc i64 %3202 to i32
  %3204 = mul i32 %3203, 4
  %3205 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3204, i32 0, i32 0)
  %3206 = bitcast <4 x i32> %3205 to <16 x i8>
  %3207 = add i64 %129, 448
  %3208 = add i64 %3207, %123
  %3209 = trunc i64 %3208 to i32
  %3210 = mul i32 %3209, 4
  %3211 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3210, i32 0, i32 0)
  %3212 = bitcast <4 x i32> %3211 to <16 x i8>
  %3213 = add i64 %142, 448
  %3214 = add i64 %3213, %136
  %3215 = trunc i64 %3214 to i32
  %3216 = mul i32 %3215, 4
  %3217 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3216, i32 0, i32 0)
  %3218 = bitcast <4 x i32> %3217 to <16 x i8>
  %3219 = add i64 %155, 448
  %3220 = add i64 %3219, %149
  %3221 = trunc i64 %3220 to i32
  %3222 = mul i32 %3221, 4
  %3223 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3222, i32 0, i32 0)
  %3224 = bitcast <4 x i32> %3223 to <16 x i8>
  %3225 = add i64 %168, 448
  %3226 = add i64 %3225, %162
  %3227 = trunc i64 %3226 to i32
  %3228 = mul i32 %3227, 4
  %3229 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3228, i32 0, i32 0)
  %3230 = bitcast <4 x i32> %3229 to <16 x i8>
  %3231 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %680
  store <16 x i8> %3188, ptr addrspace(3) %3231, align 1
  %3232 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %682
  store <16 x i8> %3194, ptr addrspace(3) %3232, align 1
  %3233 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %684
  store <16 x i8> %3200, ptr addrspace(3) %3233, align 1
  %3234 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %686
  store <16 x i8> %3206, ptr addrspace(3) %3234, align 1
  %3235 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %688
  store <16 x i8> %3212, ptr addrspace(3) %3235, align 1
  %3236 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %690
  store <16 x i8> %3218, ptr addrspace(3) %3236, align 1
  %3237 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %692
  store <16 x i8> %3224, ptr addrspace(3) %3237, align 1
  %3238 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %694
  store <16 x i8> %3230, ptr addrspace(3) %3238, align 1
  %3239 = bitcast <16 x i8> %3132 to <4 x i32>
  %3240 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3038, <4 x i32> %3239, <4 x i32> %2865, i32 %3167, i32 %2925)
  %3241 = bitcast <16 x i8> %3134 to <4 x i32>
  %3242 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3240, <4 x i32> %3241, <4 x i32> %2868, i32 %3167, i32 %2925)
  %3243 = bitcast <16 x i8> %3136 to <4 x i32>
  %3244 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3042, <4 x i32> %3243, <4 x i32> %2865, i32 %3167, i32 %2925)
  %3245 = bitcast <16 x i8> %3138 to <4 x i32>
  %3246 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3244, <4 x i32> %3245, <4 x i32> %2868, i32 %3167, i32 %2925)
  %3247 = bitcast <16 x i8> %3140 to <4 x i32>
  %3248 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3046, <4 x i32> %3247, <4 x i32> %2865, i32 %3172, i32 %2925)
  %3249 = bitcast <16 x i8> %3142 to <4 x i32>
  %3250 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3248, <4 x i32> %3249, <4 x i32> %2868, i32 %3172, i32 %2925)
  %3251 = bitcast <16 x i8> %3144 to <4 x i32>
  %3252 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3050, <4 x i32> %3251, <4 x i32> %2865, i32 %3172, i32 %2925)
  %3253 = bitcast <16 x i8> %3146 to <4 x i32>
  %3254 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3252, <4 x i32> %3253, <4 x i32> %2868, i32 %3172, i32 %2925)
  %3255 = bitcast <16 x i8> %3148 to <4 x i32>
  %3256 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3054, <4 x i32> %3255, <4 x i32> %2865, i32 %3177, i32 %2925)
  %3257 = bitcast <16 x i8> %3150 to <4 x i32>
  %3258 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3256, <4 x i32> %3257, <4 x i32> %2868, i32 %3177, i32 %2925)
  %3259 = bitcast <16 x i8> %3152 to <4 x i32>
  %3260 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3058, <4 x i32> %3259, <4 x i32> %2865, i32 %3177, i32 %2925)
  %3261 = bitcast <16 x i8> %3154 to <4 x i32>
  %3262 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3260, <4 x i32> %3261, <4 x i32> %2868, i32 %3177, i32 %2925)
  %3263 = bitcast <16 x i8> %3156 to <4 x i32>
  %3264 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3062, <4 x i32> %3263, <4 x i32> %2865, i32 %3182, i32 %2925)
  %3265 = bitcast <16 x i8> %3158 to <4 x i32>
  %3266 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3264, <4 x i32> %3265, <4 x i32> %2868, i32 %3182, i32 %2925)
  %3267 = bitcast <16 x i8> %3160 to <4 x i32>
  %3268 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3066, <4 x i32> %3267, <4 x i32> %2865, i32 %3182, i32 %2925)
  %3269 = bitcast <16 x i8> %3162 to <4 x i32>
  %3270 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3268, <4 x i32> %3269, <4 x i32> %2868, i32 %3182, i32 %2925)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3271 = add i32 %192, 7168
  %3272 = mul i32 %3271, 4
  %3273 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3272, i32 %176, i32 2)
  %3274 = add i32 %192, 7424
  %3275 = mul i32 %3274, 4
  %3276 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3275, i32 %176, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3277 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3074, <4 x i32> %3239, <4 x i32> %2885, i32 %3167, i32 %2925)
  %3278 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3277, <4 x i32> %3241, <4 x i32> %2886, i32 %3167, i32 %2925)
  %3279 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3076, <4 x i32> %3243, <4 x i32> %2885, i32 %3167, i32 %2925)
  %3280 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3279, <4 x i32> %3245, <4 x i32> %2886, i32 %3167, i32 %2925)
  %3281 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3078, <4 x i32> %3247, <4 x i32> %2885, i32 %3172, i32 %2925)
  %3282 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3281, <4 x i32> %3249, <4 x i32> %2886, i32 %3172, i32 %2925)
  %3283 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3080, <4 x i32> %3251, <4 x i32> %2885, i32 %3172, i32 %2925)
  %3284 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3283, <4 x i32> %3253, <4 x i32> %2886, i32 %3172, i32 %2925)
  %3285 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3082, <4 x i32> %3255, <4 x i32> %2885, i32 %3177, i32 %2925)
  %3286 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3285, <4 x i32> %3257, <4 x i32> %2886, i32 %3177, i32 %2925)
  %3287 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3084, <4 x i32> %3259, <4 x i32> %2885, i32 %3177, i32 %2925)
  %3288 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3287, <4 x i32> %3261, <4 x i32> %2886, i32 %3177, i32 %2925)
  %3289 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3086, <4 x i32> %3263, <4 x i32> %2885, i32 %3182, i32 %2925)
  %3290 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3289, <4 x i32> %3265, <4 x i32> %2886, i32 %3182, i32 %2925)
  %3291 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3088, <4 x i32> %3267, <4 x i32> %2885, i32 %3182, i32 %2925)
  %3292 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3291, <4 x i32> %3269, <4 x i32> %2886, i32 %3182, i32 %2925)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3293 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3272, i32 %180, i32 2)
  %3294 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3275, i32 %180, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3295 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3092, <4 x i32> %3239, <4 x i32> %2903, i32 %3167, i32 %2926)
  %3296 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3295, <4 x i32> %3241, <4 x i32> %2904, i32 %3167, i32 %2926)
  %3297 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3094, <4 x i32> %3243, <4 x i32> %2903, i32 %3167, i32 %2926)
  %3298 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3297, <4 x i32> %3245, <4 x i32> %2904, i32 %3167, i32 %2926)
  %3299 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3096, <4 x i32> %3247, <4 x i32> %2903, i32 %3172, i32 %2926)
  %3300 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3299, <4 x i32> %3249, <4 x i32> %2904, i32 %3172, i32 %2926)
  %3301 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3098, <4 x i32> %3251, <4 x i32> %2903, i32 %3172, i32 %2926)
  %3302 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3301, <4 x i32> %3253, <4 x i32> %2904, i32 %3172, i32 %2926)
  %3303 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3100, <4 x i32> %3255, <4 x i32> %2903, i32 %3177, i32 %2926)
  %3304 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3303, <4 x i32> %3257, <4 x i32> %2904, i32 %3177, i32 %2926)
  %3305 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3102, <4 x i32> %3259, <4 x i32> %2903, i32 %3177, i32 %2926)
  %3306 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3305, <4 x i32> %3261, <4 x i32> %2904, i32 %3177, i32 %2926)
  %3307 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3104, <4 x i32> %3263, <4 x i32> %2903, i32 %3182, i32 %2926)
  %3308 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3307, <4 x i32> %3265, <4 x i32> %2904, i32 %3182, i32 %2926)
  %3309 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3106, <4 x i32> %3267, <4 x i32> %2903, i32 %3182, i32 %2926)
  %3310 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3309, <4 x i32> %3269, <4 x i32> %2904, i32 %3182, i32 %2926)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3311 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3272, i32 %184, i32 2)
  %3312 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3275, i32 %184, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3313 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3110, <4 x i32> %3239, <4 x i32> %2921, i32 %3167, i32 %2926)
  %3314 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3313, <4 x i32> %3241, <4 x i32> %2922, i32 %3167, i32 %2926)
  %3315 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3112, <4 x i32> %3243, <4 x i32> %2921, i32 %3167, i32 %2926)
  %3316 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3315, <4 x i32> %3245, <4 x i32> %2922, i32 %3167, i32 %2926)
  %3317 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3114, <4 x i32> %3247, <4 x i32> %2921, i32 %3172, i32 %2926)
  %3318 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3317, <4 x i32> %3249, <4 x i32> %2922, i32 %3172, i32 %2926)
  %3319 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3116, <4 x i32> %3251, <4 x i32> %2921, i32 %3172, i32 %2926)
  %3320 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3319, <4 x i32> %3253, <4 x i32> %2922, i32 %3172, i32 %2926)
  %3321 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3118, <4 x i32> %3255, <4 x i32> %2921, i32 %3177, i32 %2926)
  %3322 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3321, <4 x i32> %3257, <4 x i32> %2922, i32 %3177, i32 %2926)
  %3323 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3120, <4 x i32> %3259, <4 x i32> %2921, i32 %3177, i32 %2926)
  %3324 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3323, <4 x i32> %3261, <4 x i32> %2922, i32 %3177, i32 %2926)
  %3325 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3122, <4 x i32> %3263, <4 x i32> %2921, i32 %3182, i32 %2926)
  %3326 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3325, <4 x i32> %3265, <4 x i32> %2922, i32 %3182, i32 %2926)
  %3327 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3124, <4 x i32> %3267, <4 x i32> %2921, i32 %3182, i32 %2926)
  %3328 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3327, <4 x i32> %3269, <4 x i32> %2922, i32 %3182, i32 %2926)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3329 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3272, i32 %188, i32 2)
  %3330 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3275, i32 %188, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3331 = add i32 %211, 896
  %3332 = mul i32 %3331, 4
  %3333 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %3332, i32 %204, i32 0)
  %3334 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %3332, i32 %210, i32 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %3335 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %856
  %3336 = load <16 x i8>, ptr addrspace(3) %3335, align 1
  %3337 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %859
  %3338 = load <16 x i8>, ptr addrspace(3) %3337, align 1
  %3339 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %862
  %3340 = load <16 x i8>, ptr addrspace(3) %3339, align 1
  %3341 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %865
  %3342 = load <16 x i8>, ptr addrspace(3) %3341, align 1
  %3343 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %868
  %3344 = load <16 x i8>, ptr addrspace(3) %3343, align 1
  %3345 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %871
  %3346 = load <16 x i8>, ptr addrspace(3) %3345, align 1
  %3347 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %874
  %3348 = load <16 x i8>, ptr addrspace(3) %3347, align 1
  %3349 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %877
  %3350 = load <16 x i8>, ptr addrspace(3) %3349, align 1
  %3351 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %880
  %3352 = load <16 x i8>, ptr addrspace(3) %3351, align 1
  %3353 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %883
  %3354 = load <16 x i8>, ptr addrspace(3) %3353, align 1
  %3355 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %886
  %3356 = load <16 x i8>, ptr addrspace(3) %3355, align 1
  %3357 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %889
  %3358 = load <16 x i8>, ptr addrspace(3) %3357, align 1
  %3359 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %892
  %3360 = load <16 x i8>, ptr addrspace(3) %3359, align 1
  %3361 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %895
  %3362 = load <16 x i8>, ptr addrspace(3) %3361, align 1
  %3363 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %898
  %3364 = load <16 x i8>, ptr addrspace(3) %3363, align 1
  %3365 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %901
  %3366 = load <16 x i8>, ptr addrspace(3) %3365, align 1
  %3367 = add i64 %193, 832
  %3368 = add i64 %3367, %24
  %3369 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %3368
  %3370 = load <1 x i32>, ptr addrspace(3) %3369, align 4
  %3371 = extractelement <1 x i32> %3370, i64 0
  %3372 = add i64 %193, 2624
  %3373 = add i64 %3372, %24
  %3374 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %3373
  %3375 = load <1 x i32>, ptr addrspace(3) %3374, align 4
  %3376 = extractelement <1 x i32> %3375, i64 0
  %3377 = add i64 %193, 4416
  %3378 = add i64 %3377, %24
  %3379 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %3378
  %3380 = load <1 x i32>, ptr addrspace(3) %3379, align 4
  %3381 = extractelement <1 x i32> %3380, i64 0
  %3382 = add i64 %193, 6208
  %3383 = add i64 %3382, %24
  %3384 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %3383
  %3385 = load <1 x i32>, ptr addrspace(3) %3384, align 4
  %3386 = extractelement <1 x i32> %3385, i64 0
  %3387 = add i64 %77, 480
  %3388 = add i64 %3387, %71
  %3389 = trunc i64 %3388 to i32
  %3390 = mul i32 %3389, 4
  %3391 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3390, i32 0, i32 0)
  %3392 = bitcast <4 x i32> %3391 to <16 x i8>
  %3393 = add i64 %90, 480
  %3394 = add i64 %3393, %84
  %3395 = trunc i64 %3394 to i32
  %3396 = mul i32 %3395, 4
  %3397 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3396, i32 0, i32 0)
  %3398 = bitcast <4 x i32> %3397 to <16 x i8>
  %3399 = add i64 %103, 480
  %3400 = add i64 %3399, %97
  %3401 = trunc i64 %3400 to i32
  %3402 = mul i32 %3401, 4
  %3403 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3402, i32 0, i32 0)
  %3404 = bitcast <4 x i32> %3403 to <16 x i8>
  %3405 = add i64 %116, 480
  %3406 = add i64 %3405, %110
  %3407 = trunc i64 %3406 to i32
  %3408 = mul i32 %3407, 4
  %3409 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3408, i32 0, i32 0)
  %3410 = bitcast <4 x i32> %3409 to <16 x i8>
  %3411 = add i64 %129, 480
  %3412 = add i64 %3411, %123
  %3413 = trunc i64 %3412 to i32
  %3414 = mul i32 %3413, 4
  %3415 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3414, i32 0, i32 0)
  %3416 = bitcast <4 x i32> %3415 to <16 x i8>
  %3417 = add i64 %142, 480
  %3418 = add i64 %3417, %136
  %3419 = trunc i64 %3418 to i32
  %3420 = mul i32 %3419, 4
  %3421 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3420, i32 0, i32 0)
  %3422 = bitcast <4 x i32> %3421 to <16 x i8>
  %3423 = add i64 %155, 480
  %3424 = add i64 %3423, %149
  %3425 = trunc i64 %3424 to i32
  %3426 = mul i32 %3425, 4
  %3427 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3426, i32 0, i32 0)
  %3428 = bitcast <4 x i32> %3427 to <16 x i8>
  %3429 = add i64 %168, 480
  %3430 = add i64 %3429, %162
  %3431 = trunc i64 %3430 to i32
  %3432 = mul i32 %3431, 4
  %3433 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3432, i32 0, i32 0)
  %3434 = bitcast <4 x i32> %3433 to <16 x i8>
  %3435 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %360
  store <16 x i8> %3392, ptr addrspace(3) %3435, align 1
  %3436 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %369
  store <16 x i8> %3398, ptr addrspace(3) %3436, align 1
  %3437 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %378
  store <16 x i8> %3404, ptr addrspace(3) %3437, align 1
  %3438 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %387
  store <16 x i8> %3410, ptr addrspace(3) %3438, align 1
  %3439 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %396
  store <16 x i8> %3416, ptr addrspace(3) %3439, align 1
  %3440 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %405
  store <16 x i8> %3422, ptr addrspace(3) %3440, align 1
  %3441 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %414
  store <16 x i8> %3428, ptr addrspace(3) %3441, align 1
  %3442 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %423
  store <16 x i8> %3434, ptr addrspace(3) %3442, align 1
  %3443 = bitcast <16 x i8> %3336 to <4 x i32>
  %3444 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3242, <4 x i32> %3443, <4 x i32> %3069, i32 %3371, i32 %3129)
  %3445 = bitcast <16 x i8> %3338 to <4 x i32>
  %3446 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3444, <4 x i32> %3445, <4 x i32> %3072, i32 %3371, i32 %3129)
  %3447 = bitcast <16 x i8> %3340 to <4 x i32>
  %3448 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3246, <4 x i32> %3447, <4 x i32> %3069, i32 %3371, i32 %3129)
  %3449 = bitcast <16 x i8> %3342 to <4 x i32>
  %3450 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3448, <4 x i32> %3449, <4 x i32> %3072, i32 %3371, i32 %3129)
  %3451 = bitcast <16 x i8> %3344 to <4 x i32>
  %3452 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3250, <4 x i32> %3451, <4 x i32> %3069, i32 %3376, i32 %3129)
  %3453 = bitcast <16 x i8> %3346 to <4 x i32>
  %3454 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3452, <4 x i32> %3453, <4 x i32> %3072, i32 %3376, i32 %3129)
  %3455 = bitcast <16 x i8> %3348 to <4 x i32>
  %3456 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3254, <4 x i32> %3455, <4 x i32> %3069, i32 %3376, i32 %3129)
  %3457 = bitcast <16 x i8> %3350 to <4 x i32>
  %3458 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3456, <4 x i32> %3457, <4 x i32> %3072, i32 %3376, i32 %3129)
  %3459 = bitcast <16 x i8> %3352 to <4 x i32>
  %3460 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3258, <4 x i32> %3459, <4 x i32> %3069, i32 %3381, i32 %3129)
  %3461 = bitcast <16 x i8> %3354 to <4 x i32>
  %3462 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3460, <4 x i32> %3461, <4 x i32> %3072, i32 %3381, i32 %3129)
  %3463 = bitcast <16 x i8> %3356 to <4 x i32>
  %3464 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3262, <4 x i32> %3463, <4 x i32> %3069, i32 %3381, i32 %3129)
  %3465 = bitcast <16 x i8> %3358 to <4 x i32>
  %3466 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3464, <4 x i32> %3465, <4 x i32> %3072, i32 %3381, i32 %3129)
  %3467 = bitcast <16 x i8> %3360 to <4 x i32>
  %3468 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3266, <4 x i32> %3467, <4 x i32> %3069, i32 %3386, i32 %3129)
  %3469 = bitcast <16 x i8> %3362 to <4 x i32>
  %3470 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3468, <4 x i32> %3469, <4 x i32> %3072, i32 %3386, i32 %3129)
  %3471 = bitcast <16 x i8> %3364 to <4 x i32>
  %3472 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3270, <4 x i32> %3471, <4 x i32> %3069, i32 %3386, i32 %3129)
  %3473 = bitcast <16 x i8> %3366 to <4 x i32>
  %3474 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3472, <4 x i32> %3473, <4 x i32> %3072, i32 %3386, i32 %3129)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3475 = add i32 %192, 7680
  %3476 = mul i32 %3475, 4
  %3477 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3476, i32 %176, i32 2)
  %3478 = add i32 %192, 7936
  %3479 = mul i32 %3478, 4
  %3480 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3479, i32 %176, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3481 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3278, <4 x i32> %3443, <4 x i32> %3089, i32 %3371, i32 %3129)
  %3482 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3481, <4 x i32> %3445, <4 x i32> %3090, i32 %3371, i32 %3129)
  %3483 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3280, <4 x i32> %3447, <4 x i32> %3089, i32 %3371, i32 %3129)
  %3484 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3483, <4 x i32> %3449, <4 x i32> %3090, i32 %3371, i32 %3129)
  %3485 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3282, <4 x i32> %3451, <4 x i32> %3089, i32 %3376, i32 %3129)
  %3486 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3485, <4 x i32> %3453, <4 x i32> %3090, i32 %3376, i32 %3129)
  %3487 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3284, <4 x i32> %3455, <4 x i32> %3089, i32 %3376, i32 %3129)
  %3488 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3487, <4 x i32> %3457, <4 x i32> %3090, i32 %3376, i32 %3129)
  %3489 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3286, <4 x i32> %3459, <4 x i32> %3089, i32 %3381, i32 %3129)
  %3490 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3489, <4 x i32> %3461, <4 x i32> %3090, i32 %3381, i32 %3129)
  %3491 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3288, <4 x i32> %3463, <4 x i32> %3089, i32 %3381, i32 %3129)
  %3492 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3491, <4 x i32> %3465, <4 x i32> %3090, i32 %3381, i32 %3129)
  %3493 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3290, <4 x i32> %3467, <4 x i32> %3089, i32 %3386, i32 %3129)
  %3494 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3493, <4 x i32> %3469, <4 x i32> %3090, i32 %3386, i32 %3129)
  %3495 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3292, <4 x i32> %3471, <4 x i32> %3089, i32 %3386, i32 %3129)
  %3496 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3495, <4 x i32> %3473, <4 x i32> %3090, i32 %3386, i32 %3129)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3497 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3476, i32 %180, i32 2)
  %3498 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3479, i32 %180, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3499 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3296, <4 x i32> %3443, <4 x i32> %3107, i32 %3371, i32 %3130)
  %3500 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3499, <4 x i32> %3445, <4 x i32> %3108, i32 %3371, i32 %3130)
  %3501 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3298, <4 x i32> %3447, <4 x i32> %3107, i32 %3371, i32 %3130)
  %3502 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3501, <4 x i32> %3449, <4 x i32> %3108, i32 %3371, i32 %3130)
  %3503 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3300, <4 x i32> %3451, <4 x i32> %3107, i32 %3376, i32 %3130)
  %3504 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3503, <4 x i32> %3453, <4 x i32> %3108, i32 %3376, i32 %3130)
  %3505 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3302, <4 x i32> %3455, <4 x i32> %3107, i32 %3376, i32 %3130)
  %3506 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3505, <4 x i32> %3457, <4 x i32> %3108, i32 %3376, i32 %3130)
  %3507 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3304, <4 x i32> %3459, <4 x i32> %3107, i32 %3381, i32 %3130)
  %3508 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3507, <4 x i32> %3461, <4 x i32> %3108, i32 %3381, i32 %3130)
  %3509 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3306, <4 x i32> %3463, <4 x i32> %3107, i32 %3381, i32 %3130)
  %3510 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3509, <4 x i32> %3465, <4 x i32> %3108, i32 %3381, i32 %3130)
  %3511 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3308, <4 x i32> %3467, <4 x i32> %3107, i32 %3386, i32 %3130)
  %3512 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3511, <4 x i32> %3469, <4 x i32> %3108, i32 %3386, i32 %3130)
  %3513 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3310, <4 x i32> %3471, <4 x i32> %3107, i32 %3386, i32 %3130)
  %3514 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3513, <4 x i32> %3473, <4 x i32> %3108, i32 %3386, i32 %3130)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3515 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3476, i32 %184, i32 2)
  %3516 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3479, i32 %184, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3517 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3314, <4 x i32> %3443, <4 x i32> %3125, i32 %3371, i32 %3130)
  %3518 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3517, <4 x i32> %3445, <4 x i32> %3126, i32 %3371, i32 %3130)
  %3519 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3316, <4 x i32> %3447, <4 x i32> %3125, i32 %3371, i32 %3130)
  %3520 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3519, <4 x i32> %3449, <4 x i32> %3126, i32 %3371, i32 %3130)
  %3521 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3318, <4 x i32> %3451, <4 x i32> %3125, i32 %3376, i32 %3130)
  %3522 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3521, <4 x i32> %3453, <4 x i32> %3126, i32 %3376, i32 %3130)
  %3523 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3320, <4 x i32> %3455, <4 x i32> %3125, i32 %3376, i32 %3130)
  %3524 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3523, <4 x i32> %3457, <4 x i32> %3126, i32 %3376, i32 %3130)
  %3525 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3322, <4 x i32> %3459, <4 x i32> %3125, i32 %3381, i32 %3130)
  %3526 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3525, <4 x i32> %3461, <4 x i32> %3126, i32 %3381, i32 %3130)
  %3527 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3324, <4 x i32> %3463, <4 x i32> %3125, i32 %3381, i32 %3130)
  %3528 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3527, <4 x i32> %3465, <4 x i32> %3126, i32 %3381, i32 %3130)
  %3529 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3326, <4 x i32> %3467, <4 x i32> %3125, i32 %3386, i32 %3130)
  %3530 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3529, <4 x i32> %3469, <4 x i32> %3126, i32 %3386, i32 %3130)
  %3531 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3328, <4 x i32> %3471, <4 x i32> %3125, i32 %3386, i32 %3130)
  %3532 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3531, <4 x i32> %3473, <4 x i32> %3126, i32 %3386, i32 %3130)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3533 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3476, i32 %188, i32 2)
  %3534 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3479, i32 %188, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3535 = add i32 %211, 960
  %3536 = mul i32 %3535, 4
  %3537 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %3536, i32 %204, i32 0)
  %3538 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %3536, i32 %210, i32 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %3539 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1075
  %3540 = load <16 x i8>, ptr addrspace(3) %3539, align 1
  %3541 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1078
  %3542 = load <16 x i8>, ptr addrspace(3) %3541, align 1
  %3543 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1081
  %3544 = load <16 x i8>, ptr addrspace(3) %3543, align 1
  %3545 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1084
  %3546 = load <16 x i8>, ptr addrspace(3) %3545, align 1
  %3547 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1087
  %3548 = load <16 x i8>, ptr addrspace(3) %3547, align 1
  %3549 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1090
  %3550 = load <16 x i8>, ptr addrspace(3) %3549, align 1
  %3551 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1093
  %3552 = load <16 x i8>, ptr addrspace(3) %3551, align 1
  %3553 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1096
  %3554 = load <16 x i8>, ptr addrspace(3) %3553, align 1
  %3555 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1099
  %3556 = load <16 x i8>, ptr addrspace(3) %3555, align 1
  %3557 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1102
  %3558 = load <16 x i8>, ptr addrspace(3) %3557, align 1
  %3559 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1105
  %3560 = load <16 x i8>, ptr addrspace(3) %3559, align 1
  %3561 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1108
  %3562 = load <16 x i8>, ptr addrspace(3) %3561, align 1
  %3563 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1111
  %3564 = load <16 x i8>, ptr addrspace(3) %3563, align 1
  %3565 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1114
  %3566 = load <16 x i8>, ptr addrspace(3) %3565, align 1
  %3567 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1117
  %3568 = load <16 x i8>, ptr addrspace(3) %3567, align 1
  %3569 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1120
  %3570 = load <16 x i8>, ptr addrspace(3) %3569, align 1
  %3571 = add i64 %193, 896
  %3572 = add i64 %3571, %24
  %3573 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %3572
  %3574 = load <1 x i32>, ptr addrspace(3) %3573, align 4
  %3575 = extractelement <1 x i32> %3574, i64 0
  %3576 = add i64 %193, 2688
  %3577 = add i64 %3576, %24
  %3578 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %3577
  %3579 = load <1 x i32>, ptr addrspace(3) %3578, align 4
  %3580 = extractelement <1 x i32> %3579, i64 0
  %3581 = add i64 %193, 4480
  %3582 = add i64 %3581, %24
  %3583 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %3582
  %3584 = load <1 x i32>, ptr addrspace(3) %3583, align 4
  %3585 = extractelement <1 x i32> %3584, i64 0
  %3586 = add i64 %193, 6272
  %3587 = add i64 %3586, %24
  %3588 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %3587
  %3589 = load <1 x i32>, ptr addrspace(3) %3588, align 4
  %3590 = extractelement <1 x i32> %3589, i64 0
  %3591 = add i64 %77, 512
  %3592 = add i64 %3591, %71
  %3593 = trunc i64 %3592 to i32
  %3594 = mul i32 %3593, 4
  %3595 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3594, i32 0, i32 0)
  %3596 = bitcast <4 x i32> %3595 to <16 x i8>
  %3597 = add i64 %90, 512
  %3598 = add i64 %3597, %84
  %3599 = trunc i64 %3598 to i32
  %3600 = mul i32 %3599, 4
  %3601 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3600, i32 0, i32 0)
  %3602 = bitcast <4 x i32> %3601 to <16 x i8>
  %3603 = add i64 %103, 512
  %3604 = add i64 %3603, %97
  %3605 = trunc i64 %3604 to i32
  %3606 = mul i32 %3605, 4
  %3607 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3606, i32 0, i32 0)
  %3608 = bitcast <4 x i32> %3607 to <16 x i8>
  %3609 = add i64 %116, 512
  %3610 = add i64 %3609, %110
  %3611 = trunc i64 %3610 to i32
  %3612 = mul i32 %3611, 4
  %3613 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3612, i32 0, i32 0)
  %3614 = bitcast <4 x i32> %3613 to <16 x i8>
  %3615 = add i64 %129, 512
  %3616 = add i64 %3615, %123
  %3617 = trunc i64 %3616 to i32
  %3618 = mul i32 %3617, 4
  %3619 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3618, i32 0, i32 0)
  %3620 = bitcast <4 x i32> %3619 to <16 x i8>
  %3621 = add i64 %142, 512
  %3622 = add i64 %3621, %136
  %3623 = trunc i64 %3622 to i32
  %3624 = mul i32 %3623, 4
  %3625 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3624, i32 0, i32 0)
  %3626 = bitcast <4 x i32> %3625 to <16 x i8>
  %3627 = add i64 %155, 512
  %3628 = add i64 %3627, %149
  %3629 = trunc i64 %3628 to i32
  %3630 = mul i32 %3629, 4
  %3631 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3630, i32 0, i32 0)
  %3632 = bitcast <4 x i32> %3631 to <16 x i8>
  %3633 = add i64 %168, 512
  %3634 = add i64 %3633, %162
  %3635 = trunc i64 %3634 to i32
  %3636 = mul i32 %3635, 4
  %3637 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3636, i32 0, i32 0)
  %3638 = bitcast <4 x i32> %3637 to <16 x i8>
  %3639 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %487
  store <16 x i8> %3596, ptr addrspace(3) %3639, align 1
  %3640 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %489
  store <16 x i8> %3602, ptr addrspace(3) %3640, align 1
  %3641 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %491
  store <16 x i8> %3608, ptr addrspace(3) %3641, align 1
  %3642 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %493
  store <16 x i8> %3614, ptr addrspace(3) %3642, align 1
  %3643 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %495
  store <16 x i8> %3620, ptr addrspace(3) %3643, align 1
  %3644 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %497
  store <16 x i8> %3626, ptr addrspace(3) %3644, align 1
  %3645 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %499
  store <16 x i8> %3632, ptr addrspace(3) %3645, align 1
  %3646 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %501
  store <16 x i8> %3638, ptr addrspace(3) %3646, align 1
  %3647 = bitcast <16 x i8> %3540 to <4 x i32>
  %3648 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3446, <4 x i32> %3647, <4 x i32> %3273, i32 %3575, i32 %3333)
  %3649 = bitcast <16 x i8> %3542 to <4 x i32>
  %3650 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3648, <4 x i32> %3649, <4 x i32> %3276, i32 %3575, i32 %3333)
  %3651 = bitcast <16 x i8> %3544 to <4 x i32>
  %3652 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3450, <4 x i32> %3651, <4 x i32> %3273, i32 %3575, i32 %3333)
  %3653 = bitcast <16 x i8> %3546 to <4 x i32>
  %3654 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3652, <4 x i32> %3653, <4 x i32> %3276, i32 %3575, i32 %3333)
  %3655 = bitcast <16 x i8> %3548 to <4 x i32>
  %3656 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3454, <4 x i32> %3655, <4 x i32> %3273, i32 %3580, i32 %3333)
  %3657 = bitcast <16 x i8> %3550 to <4 x i32>
  %3658 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3656, <4 x i32> %3657, <4 x i32> %3276, i32 %3580, i32 %3333)
  %3659 = bitcast <16 x i8> %3552 to <4 x i32>
  %3660 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3458, <4 x i32> %3659, <4 x i32> %3273, i32 %3580, i32 %3333)
  %3661 = bitcast <16 x i8> %3554 to <4 x i32>
  %3662 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3660, <4 x i32> %3661, <4 x i32> %3276, i32 %3580, i32 %3333)
  %3663 = bitcast <16 x i8> %3556 to <4 x i32>
  %3664 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3462, <4 x i32> %3663, <4 x i32> %3273, i32 %3585, i32 %3333)
  %3665 = bitcast <16 x i8> %3558 to <4 x i32>
  %3666 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3664, <4 x i32> %3665, <4 x i32> %3276, i32 %3585, i32 %3333)
  %3667 = bitcast <16 x i8> %3560 to <4 x i32>
  %3668 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3466, <4 x i32> %3667, <4 x i32> %3273, i32 %3585, i32 %3333)
  %3669 = bitcast <16 x i8> %3562 to <4 x i32>
  %3670 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3668, <4 x i32> %3669, <4 x i32> %3276, i32 %3585, i32 %3333)
  %3671 = bitcast <16 x i8> %3564 to <4 x i32>
  %3672 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3470, <4 x i32> %3671, <4 x i32> %3273, i32 %3590, i32 %3333)
  %3673 = bitcast <16 x i8> %3566 to <4 x i32>
  %3674 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3672, <4 x i32> %3673, <4 x i32> %3276, i32 %3590, i32 %3333)
  %3675 = bitcast <16 x i8> %3568 to <4 x i32>
  %3676 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3474, <4 x i32> %3675, <4 x i32> %3273, i32 %3590, i32 %3333)
  %3677 = bitcast <16 x i8> %3570 to <4 x i32>
  %3678 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3676, <4 x i32> %3677, <4 x i32> %3276, i32 %3590, i32 %3333)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3679 = add i32 %192, 8192
  %3680 = mul i32 %3679, 4
  %3681 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3680, i32 %176, i32 2)
  %3682 = add i32 %192, 8448
  %3683 = mul i32 %3682, 4
  %3684 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3683, i32 %176, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3685 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3482, <4 x i32> %3647, <4 x i32> %3293, i32 %3575, i32 %3333)
  %3686 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3685, <4 x i32> %3649, <4 x i32> %3294, i32 %3575, i32 %3333)
  %3687 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3484, <4 x i32> %3651, <4 x i32> %3293, i32 %3575, i32 %3333)
  %3688 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3687, <4 x i32> %3653, <4 x i32> %3294, i32 %3575, i32 %3333)
  %3689 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3486, <4 x i32> %3655, <4 x i32> %3293, i32 %3580, i32 %3333)
  %3690 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3689, <4 x i32> %3657, <4 x i32> %3294, i32 %3580, i32 %3333)
  %3691 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3488, <4 x i32> %3659, <4 x i32> %3293, i32 %3580, i32 %3333)
  %3692 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3691, <4 x i32> %3661, <4 x i32> %3294, i32 %3580, i32 %3333)
  %3693 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3490, <4 x i32> %3663, <4 x i32> %3293, i32 %3585, i32 %3333)
  %3694 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3693, <4 x i32> %3665, <4 x i32> %3294, i32 %3585, i32 %3333)
  %3695 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3492, <4 x i32> %3667, <4 x i32> %3293, i32 %3585, i32 %3333)
  %3696 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3695, <4 x i32> %3669, <4 x i32> %3294, i32 %3585, i32 %3333)
  %3697 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3494, <4 x i32> %3671, <4 x i32> %3293, i32 %3590, i32 %3333)
  %3698 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3697, <4 x i32> %3673, <4 x i32> %3294, i32 %3590, i32 %3333)
  %3699 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3496, <4 x i32> %3675, <4 x i32> %3293, i32 %3590, i32 %3333)
  %3700 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3699, <4 x i32> %3677, <4 x i32> %3294, i32 %3590, i32 %3333)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3701 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3680, i32 %180, i32 2)
  %3702 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3683, i32 %180, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3703 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3500, <4 x i32> %3647, <4 x i32> %3311, i32 %3575, i32 %3334)
  %3704 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3703, <4 x i32> %3649, <4 x i32> %3312, i32 %3575, i32 %3334)
  %3705 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3502, <4 x i32> %3651, <4 x i32> %3311, i32 %3575, i32 %3334)
  %3706 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3705, <4 x i32> %3653, <4 x i32> %3312, i32 %3575, i32 %3334)
  %3707 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3504, <4 x i32> %3655, <4 x i32> %3311, i32 %3580, i32 %3334)
  %3708 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3707, <4 x i32> %3657, <4 x i32> %3312, i32 %3580, i32 %3334)
  %3709 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3506, <4 x i32> %3659, <4 x i32> %3311, i32 %3580, i32 %3334)
  %3710 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3709, <4 x i32> %3661, <4 x i32> %3312, i32 %3580, i32 %3334)
  %3711 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3508, <4 x i32> %3663, <4 x i32> %3311, i32 %3585, i32 %3334)
  %3712 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3711, <4 x i32> %3665, <4 x i32> %3312, i32 %3585, i32 %3334)
  %3713 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3510, <4 x i32> %3667, <4 x i32> %3311, i32 %3585, i32 %3334)
  %3714 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3713, <4 x i32> %3669, <4 x i32> %3312, i32 %3585, i32 %3334)
  %3715 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3512, <4 x i32> %3671, <4 x i32> %3311, i32 %3590, i32 %3334)
  %3716 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3715, <4 x i32> %3673, <4 x i32> %3312, i32 %3590, i32 %3334)
  %3717 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3514, <4 x i32> %3675, <4 x i32> %3311, i32 %3590, i32 %3334)
  %3718 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3717, <4 x i32> %3677, <4 x i32> %3312, i32 %3590, i32 %3334)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3719 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3680, i32 %184, i32 2)
  %3720 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3683, i32 %184, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3721 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3518, <4 x i32> %3647, <4 x i32> %3329, i32 %3575, i32 %3334)
  %3722 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3721, <4 x i32> %3649, <4 x i32> %3330, i32 %3575, i32 %3334)
  %3723 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3520, <4 x i32> %3651, <4 x i32> %3329, i32 %3575, i32 %3334)
  %3724 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3723, <4 x i32> %3653, <4 x i32> %3330, i32 %3575, i32 %3334)
  %3725 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3522, <4 x i32> %3655, <4 x i32> %3329, i32 %3580, i32 %3334)
  %3726 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3725, <4 x i32> %3657, <4 x i32> %3330, i32 %3580, i32 %3334)
  %3727 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3524, <4 x i32> %3659, <4 x i32> %3329, i32 %3580, i32 %3334)
  %3728 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3727, <4 x i32> %3661, <4 x i32> %3330, i32 %3580, i32 %3334)
  %3729 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3526, <4 x i32> %3663, <4 x i32> %3329, i32 %3585, i32 %3334)
  %3730 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3729, <4 x i32> %3665, <4 x i32> %3330, i32 %3585, i32 %3334)
  %3731 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3528, <4 x i32> %3667, <4 x i32> %3329, i32 %3585, i32 %3334)
  %3732 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3731, <4 x i32> %3669, <4 x i32> %3330, i32 %3585, i32 %3334)
  %3733 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3530, <4 x i32> %3671, <4 x i32> %3329, i32 %3590, i32 %3334)
  %3734 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3733, <4 x i32> %3673, <4 x i32> %3330, i32 %3590, i32 %3334)
  %3735 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3532, <4 x i32> %3675, <4 x i32> %3329, i32 %3590, i32 %3334)
  %3736 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3735, <4 x i32> %3677, <4 x i32> %3330, i32 %3590, i32 %3334)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3737 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3680, i32 %188, i32 2)
  %3738 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3683, i32 %188, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3739 = add i32 %211, 1024
  %3740 = mul i32 %3739, 4
  %3741 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %3740, i32 %204, i32 0)
  %3742 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %3740, i32 %210, i32 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %3743 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %522
  %3744 = load <16 x i8>, ptr addrspace(3) %3743, align 1
  %3745 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %527
  %3746 = load <16 x i8>, ptr addrspace(3) %3745, align 1
  %3747 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %535
  %3748 = load <16 x i8>, ptr addrspace(3) %3747, align 1
  %3749 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %539
  %3750 = load <16 x i8>, ptr addrspace(3) %3749, align 1
  %3751 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %547
  %3752 = load <16 x i8>, ptr addrspace(3) %3751, align 1
  %3753 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %551
  %3754 = load <16 x i8>, ptr addrspace(3) %3753, align 1
  %3755 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %559
  %3756 = load <16 x i8>, ptr addrspace(3) %3755, align 1
  %3757 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %563
  %3758 = load <16 x i8>, ptr addrspace(3) %3757, align 1
  %3759 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %571
  %3760 = load <16 x i8>, ptr addrspace(3) %3759, align 1
  %3761 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %575
  %3762 = load <16 x i8>, ptr addrspace(3) %3761, align 1
  %3763 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %583
  %3764 = load <16 x i8>, ptr addrspace(3) %3763, align 1
  %3765 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %587
  %3766 = load <16 x i8>, ptr addrspace(3) %3765, align 1
  %3767 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %595
  %3768 = load <16 x i8>, ptr addrspace(3) %3767, align 1
  %3769 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %599
  %3770 = load <16 x i8>, ptr addrspace(3) %3769, align 1
  %3771 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %607
  %3772 = load <16 x i8>, ptr addrspace(3) %3771, align 1
  %3773 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %611
  %3774 = load <16 x i8>, ptr addrspace(3) %3773, align 1
  %3775 = add i64 %193, 960
  %3776 = add i64 %3775, %24
  %3777 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %3776
  %3778 = load <1 x i32>, ptr addrspace(3) %3777, align 4
  %3779 = extractelement <1 x i32> %3778, i64 0
  %3780 = add i64 %193, 2752
  %3781 = add i64 %3780, %24
  %3782 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %3781
  %3783 = load <1 x i32>, ptr addrspace(3) %3782, align 4
  %3784 = extractelement <1 x i32> %3783, i64 0
  %3785 = add i64 %193, 4544
  %3786 = add i64 %3785, %24
  %3787 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %3786
  %3788 = load <1 x i32>, ptr addrspace(3) %3787, align 4
  %3789 = extractelement <1 x i32> %3788, i64 0
  %3790 = add i64 %193, 6336
  %3791 = add i64 %3790, %24
  %3792 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %3791
  %3793 = load <1 x i32>, ptr addrspace(3) %3792, align 4
  %3794 = extractelement <1 x i32> %3793, i64 0
  %3795 = add i64 %77, 544
  %3796 = add i64 %3795, %71
  %3797 = trunc i64 %3796 to i32
  %3798 = mul i32 %3797, 4
  %3799 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3798, i32 0, i32 0)
  %3800 = bitcast <4 x i32> %3799 to <16 x i8>
  %3801 = add i64 %90, 544
  %3802 = add i64 %3801, %84
  %3803 = trunc i64 %3802 to i32
  %3804 = mul i32 %3803, 4
  %3805 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3804, i32 0, i32 0)
  %3806 = bitcast <4 x i32> %3805 to <16 x i8>
  %3807 = add i64 %103, 544
  %3808 = add i64 %3807, %97
  %3809 = trunc i64 %3808 to i32
  %3810 = mul i32 %3809, 4
  %3811 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3810, i32 0, i32 0)
  %3812 = bitcast <4 x i32> %3811 to <16 x i8>
  %3813 = add i64 %116, 544
  %3814 = add i64 %3813, %110
  %3815 = trunc i64 %3814 to i32
  %3816 = mul i32 %3815, 4
  %3817 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3816, i32 0, i32 0)
  %3818 = bitcast <4 x i32> %3817 to <16 x i8>
  %3819 = add i64 %129, 544
  %3820 = add i64 %3819, %123
  %3821 = trunc i64 %3820 to i32
  %3822 = mul i32 %3821, 4
  %3823 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3822, i32 0, i32 0)
  %3824 = bitcast <4 x i32> %3823 to <16 x i8>
  %3825 = add i64 %142, 544
  %3826 = add i64 %3825, %136
  %3827 = trunc i64 %3826 to i32
  %3828 = mul i32 %3827, 4
  %3829 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3828, i32 0, i32 0)
  %3830 = bitcast <4 x i32> %3829 to <16 x i8>
  %3831 = add i64 %155, 544
  %3832 = add i64 %3831, %149
  %3833 = trunc i64 %3832 to i32
  %3834 = mul i32 %3833, 4
  %3835 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3834, i32 0, i32 0)
  %3836 = bitcast <4 x i32> %3835 to <16 x i8>
  %3837 = add i64 %168, 544
  %3838 = add i64 %3837, %162
  %3839 = trunc i64 %3838 to i32
  %3840 = mul i32 %3839, 4
  %3841 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %3840, i32 0, i32 0)
  %3842 = bitcast <4 x i32> %3841 to <16 x i8>
  %3843 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %680
  store <16 x i8> %3800, ptr addrspace(3) %3843, align 1
  %3844 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %682
  store <16 x i8> %3806, ptr addrspace(3) %3844, align 1
  %3845 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %684
  store <16 x i8> %3812, ptr addrspace(3) %3845, align 1
  %3846 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %686
  store <16 x i8> %3818, ptr addrspace(3) %3846, align 1
  %3847 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %688
  store <16 x i8> %3824, ptr addrspace(3) %3847, align 1
  %3848 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %690
  store <16 x i8> %3830, ptr addrspace(3) %3848, align 1
  %3849 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %692
  store <16 x i8> %3836, ptr addrspace(3) %3849, align 1
  %3850 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %694
  store <16 x i8> %3842, ptr addrspace(3) %3850, align 1
  %3851 = bitcast <16 x i8> %3744 to <4 x i32>
  %3852 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3650, <4 x i32> %3851, <4 x i32> %3477, i32 %3779, i32 %3537)
  %3853 = bitcast <16 x i8> %3746 to <4 x i32>
  %3854 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3852, <4 x i32> %3853, <4 x i32> %3480, i32 %3779, i32 %3537)
  %3855 = bitcast <16 x i8> %3748 to <4 x i32>
  %3856 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3654, <4 x i32> %3855, <4 x i32> %3477, i32 %3779, i32 %3537)
  %3857 = bitcast <16 x i8> %3750 to <4 x i32>
  %3858 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3856, <4 x i32> %3857, <4 x i32> %3480, i32 %3779, i32 %3537)
  %3859 = bitcast <16 x i8> %3752 to <4 x i32>
  %3860 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3658, <4 x i32> %3859, <4 x i32> %3477, i32 %3784, i32 %3537)
  %3861 = bitcast <16 x i8> %3754 to <4 x i32>
  %3862 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3860, <4 x i32> %3861, <4 x i32> %3480, i32 %3784, i32 %3537)
  %3863 = bitcast <16 x i8> %3756 to <4 x i32>
  %3864 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3662, <4 x i32> %3863, <4 x i32> %3477, i32 %3784, i32 %3537)
  %3865 = bitcast <16 x i8> %3758 to <4 x i32>
  %3866 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3864, <4 x i32> %3865, <4 x i32> %3480, i32 %3784, i32 %3537)
  %3867 = bitcast <16 x i8> %3760 to <4 x i32>
  %3868 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3666, <4 x i32> %3867, <4 x i32> %3477, i32 %3789, i32 %3537)
  %3869 = bitcast <16 x i8> %3762 to <4 x i32>
  %3870 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3868, <4 x i32> %3869, <4 x i32> %3480, i32 %3789, i32 %3537)
  %3871 = bitcast <16 x i8> %3764 to <4 x i32>
  %3872 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3670, <4 x i32> %3871, <4 x i32> %3477, i32 %3789, i32 %3537)
  %3873 = bitcast <16 x i8> %3766 to <4 x i32>
  %3874 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3872, <4 x i32> %3873, <4 x i32> %3480, i32 %3789, i32 %3537)
  %3875 = bitcast <16 x i8> %3768 to <4 x i32>
  %3876 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3674, <4 x i32> %3875, <4 x i32> %3477, i32 %3794, i32 %3537)
  %3877 = bitcast <16 x i8> %3770 to <4 x i32>
  %3878 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3876, <4 x i32> %3877, <4 x i32> %3480, i32 %3794, i32 %3537)
  %3879 = bitcast <16 x i8> %3772 to <4 x i32>
  %3880 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3678, <4 x i32> %3879, <4 x i32> %3477, i32 %3794, i32 %3537)
  %3881 = bitcast <16 x i8> %3774 to <4 x i32>
  %3882 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3880, <4 x i32> %3881, <4 x i32> %3480, i32 %3794, i32 %3537)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3883 = add i32 %192, 8704
  %3884 = mul i32 %3883, 4
  %3885 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3884, i32 %176, i32 2)
  %3886 = add i32 %192, 8960
  %3887 = mul i32 %3886, 4
  %3888 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3887, i32 %176, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3889 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3686, <4 x i32> %3851, <4 x i32> %3497, i32 %3779, i32 %3537)
  %3890 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3889, <4 x i32> %3853, <4 x i32> %3498, i32 %3779, i32 %3537)
  %3891 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3688, <4 x i32> %3855, <4 x i32> %3497, i32 %3779, i32 %3537)
  %3892 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3891, <4 x i32> %3857, <4 x i32> %3498, i32 %3779, i32 %3537)
  %3893 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3690, <4 x i32> %3859, <4 x i32> %3497, i32 %3784, i32 %3537)
  %3894 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3893, <4 x i32> %3861, <4 x i32> %3498, i32 %3784, i32 %3537)
  %3895 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3692, <4 x i32> %3863, <4 x i32> %3497, i32 %3784, i32 %3537)
  %3896 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3895, <4 x i32> %3865, <4 x i32> %3498, i32 %3784, i32 %3537)
  %3897 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3694, <4 x i32> %3867, <4 x i32> %3497, i32 %3789, i32 %3537)
  %3898 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3897, <4 x i32> %3869, <4 x i32> %3498, i32 %3789, i32 %3537)
  %3899 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3696, <4 x i32> %3871, <4 x i32> %3497, i32 %3789, i32 %3537)
  %3900 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3899, <4 x i32> %3873, <4 x i32> %3498, i32 %3789, i32 %3537)
  %3901 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3698, <4 x i32> %3875, <4 x i32> %3497, i32 %3794, i32 %3537)
  %3902 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3901, <4 x i32> %3877, <4 x i32> %3498, i32 %3794, i32 %3537)
  %3903 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3700, <4 x i32> %3879, <4 x i32> %3497, i32 %3794, i32 %3537)
  %3904 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3903, <4 x i32> %3881, <4 x i32> %3498, i32 %3794, i32 %3537)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3905 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3884, i32 %180, i32 2)
  %3906 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3887, i32 %180, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3907 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3704, <4 x i32> %3851, <4 x i32> %3515, i32 %3779, i32 %3538)
  %3908 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3907, <4 x i32> %3853, <4 x i32> %3516, i32 %3779, i32 %3538)
  %3909 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3706, <4 x i32> %3855, <4 x i32> %3515, i32 %3779, i32 %3538)
  %3910 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3909, <4 x i32> %3857, <4 x i32> %3516, i32 %3779, i32 %3538)
  %3911 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3708, <4 x i32> %3859, <4 x i32> %3515, i32 %3784, i32 %3538)
  %3912 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3911, <4 x i32> %3861, <4 x i32> %3516, i32 %3784, i32 %3538)
  %3913 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3710, <4 x i32> %3863, <4 x i32> %3515, i32 %3784, i32 %3538)
  %3914 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3913, <4 x i32> %3865, <4 x i32> %3516, i32 %3784, i32 %3538)
  %3915 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3712, <4 x i32> %3867, <4 x i32> %3515, i32 %3789, i32 %3538)
  %3916 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3915, <4 x i32> %3869, <4 x i32> %3516, i32 %3789, i32 %3538)
  %3917 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3714, <4 x i32> %3871, <4 x i32> %3515, i32 %3789, i32 %3538)
  %3918 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3917, <4 x i32> %3873, <4 x i32> %3516, i32 %3789, i32 %3538)
  %3919 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3716, <4 x i32> %3875, <4 x i32> %3515, i32 %3794, i32 %3538)
  %3920 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3919, <4 x i32> %3877, <4 x i32> %3516, i32 %3794, i32 %3538)
  %3921 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3718, <4 x i32> %3879, <4 x i32> %3515, i32 %3794, i32 %3538)
  %3922 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3921, <4 x i32> %3881, <4 x i32> %3516, i32 %3794, i32 %3538)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3923 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3884, i32 %184, i32 2)
  %3924 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3887, i32 %184, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3925 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3722, <4 x i32> %3851, <4 x i32> %3533, i32 %3779, i32 %3538)
  %3926 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3925, <4 x i32> %3853, <4 x i32> %3534, i32 %3779, i32 %3538)
  %3927 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3724, <4 x i32> %3855, <4 x i32> %3533, i32 %3779, i32 %3538)
  %3928 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3927, <4 x i32> %3857, <4 x i32> %3534, i32 %3779, i32 %3538)
  %3929 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3726, <4 x i32> %3859, <4 x i32> %3533, i32 %3784, i32 %3538)
  %3930 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3929, <4 x i32> %3861, <4 x i32> %3534, i32 %3784, i32 %3538)
  %3931 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3728, <4 x i32> %3863, <4 x i32> %3533, i32 %3784, i32 %3538)
  %3932 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3931, <4 x i32> %3865, <4 x i32> %3534, i32 %3784, i32 %3538)
  %3933 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3730, <4 x i32> %3867, <4 x i32> %3533, i32 %3789, i32 %3538)
  %3934 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3933, <4 x i32> %3869, <4 x i32> %3534, i32 %3789, i32 %3538)
  %3935 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3732, <4 x i32> %3871, <4 x i32> %3533, i32 %3789, i32 %3538)
  %3936 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3935, <4 x i32> %3873, <4 x i32> %3534, i32 %3789, i32 %3538)
  %3937 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3734, <4 x i32> %3875, <4 x i32> %3533, i32 %3794, i32 %3538)
  %3938 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3937, <4 x i32> %3877, <4 x i32> %3534, i32 %3794, i32 %3538)
  %3939 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3736, <4 x i32> %3879, <4 x i32> %3533, i32 %3794, i32 %3538)
  %3940 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3939, <4 x i32> %3881, <4 x i32> %3534, i32 %3794, i32 %3538)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3941 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3884, i32 %188, i32 2)
  %3942 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %3887, i32 %188, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %3943 = add i32 %211, 1088
  %3944 = mul i32 %3943, 4
  %3945 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %3944, i32 %204, i32 0)
  %3946 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %3944, i32 %210, i32 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %3947 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %856
  %3948 = load <16 x i8>, ptr addrspace(3) %3947, align 1
  %3949 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %859
  %3950 = load <16 x i8>, ptr addrspace(3) %3949, align 1
  %3951 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %862
  %3952 = load <16 x i8>, ptr addrspace(3) %3951, align 1
  %3953 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %865
  %3954 = load <16 x i8>, ptr addrspace(3) %3953, align 1
  %3955 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %868
  %3956 = load <16 x i8>, ptr addrspace(3) %3955, align 1
  %3957 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %871
  %3958 = load <16 x i8>, ptr addrspace(3) %3957, align 1
  %3959 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %874
  %3960 = load <16 x i8>, ptr addrspace(3) %3959, align 1
  %3961 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %877
  %3962 = load <16 x i8>, ptr addrspace(3) %3961, align 1
  %3963 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %880
  %3964 = load <16 x i8>, ptr addrspace(3) %3963, align 1
  %3965 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %883
  %3966 = load <16 x i8>, ptr addrspace(3) %3965, align 1
  %3967 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %886
  %3968 = load <16 x i8>, ptr addrspace(3) %3967, align 1
  %3969 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %889
  %3970 = load <16 x i8>, ptr addrspace(3) %3969, align 1
  %3971 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %892
  %3972 = load <16 x i8>, ptr addrspace(3) %3971, align 1
  %3973 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %895
  %3974 = load <16 x i8>, ptr addrspace(3) %3973, align 1
  %3975 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %898
  %3976 = load <16 x i8>, ptr addrspace(3) %3975, align 1
  %3977 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %901
  %3978 = load <16 x i8>, ptr addrspace(3) %3977, align 1
  %3979 = add i64 %193, 1024
  %3980 = add i64 %3979, %24
  %3981 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %3980
  %3982 = load <1 x i32>, ptr addrspace(3) %3981, align 4
  %3983 = extractelement <1 x i32> %3982, i64 0
  %3984 = add i64 %193, 2816
  %3985 = add i64 %3984, %24
  %3986 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %3985
  %3987 = load <1 x i32>, ptr addrspace(3) %3986, align 4
  %3988 = extractelement <1 x i32> %3987, i64 0
  %3989 = add i64 %193, 4608
  %3990 = add i64 %3989, %24
  %3991 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %3990
  %3992 = load <1 x i32>, ptr addrspace(3) %3991, align 4
  %3993 = extractelement <1 x i32> %3992, i64 0
  %3994 = add i64 %193, 6400
  %3995 = add i64 %3994, %24
  %3996 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %3995
  %3997 = load <1 x i32>, ptr addrspace(3) %3996, align 4
  %3998 = extractelement <1 x i32> %3997, i64 0
  %3999 = add i64 %77, 576
  %4000 = add i64 %3999, %71
  %4001 = trunc i64 %4000 to i32
  %4002 = mul i32 %4001, 4
  %4003 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4002, i32 0, i32 0)
  %4004 = bitcast <4 x i32> %4003 to <16 x i8>
  %4005 = add i64 %90, 576
  %4006 = add i64 %4005, %84
  %4007 = trunc i64 %4006 to i32
  %4008 = mul i32 %4007, 4
  %4009 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4008, i32 0, i32 0)
  %4010 = bitcast <4 x i32> %4009 to <16 x i8>
  %4011 = add i64 %103, 576
  %4012 = add i64 %4011, %97
  %4013 = trunc i64 %4012 to i32
  %4014 = mul i32 %4013, 4
  %4015 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4014, i32 0, i32 0)
  %4016 = bitcast <4 x i32> %4015 to <16 x i8>
  %4017 = add i64 %116, 576
  %4018 = add i64 %4017, %110
  %4019 = trunc i64 %4018 to i32
  %4020 = mul i32 %4019, 4
  %4021 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4020, i32 0, i32 0)
  %4022 = bitcast <4 x i32> %4021 to <16 x i8>
  %4023 = add i64 %129, 576
  %4024 = add i64 %4023, %123
  %4025 = trunc i64 %4024 to i32
  %4026 = mul i32 %4025, 4
  %4027 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4026, i32 0, i32 0)
  %4028 = bitcast <4 x i32> %4027 to <16 x i8>
  %4029 = add i64 %142, 576
  %4030 = add i64 %4029, %136
  %4031 = trunc i64 %4030 to i32
  %4032 = mul i32 %4031, 4
  %4033 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4032, i32 0, i32 0)
  %4034 = bitcast <4 x i32> %4033 to <16 x i8>
  %4035 = add i64 %155, 576
  %4036 = add i64 %4035, %149
  %4037 = trunc i64 %4036 to i32
  %4038 = mul i32 %4037, 4
  %4039 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4038, i32 0, i32 0)
  %4040 = bitcast <4 x i32> %4039 to <16 x i8>
  %4041 = add i64 %168, 576
  %4042 = add i64 %4041, %162
  %4043 = trunc i64 %4042 to i32
  %4044 = mul i32 %4043, 4
  %4045 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4044, i32 0, i32 0)
  %4046 = bitcast <4 x i32> %4045 to <16 x i8>
  %4047 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %360
  store <16 x i8> %4004, ptr addrspace(3) %4047, align 1
  %4048 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %369
  store <16 x i8> %4010, ptr addrspace(3) %4048, align 1
  %4049 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %378
  store <16 x i8> %4016, ptr addrspace(3) %4049, align 1
  %4050 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %387
  store <16 x i8> %4022, ptr addrspace(3) %4050, align 1
  %4051 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %396
  store <16 x i8> %4028, ptr addrspace(3) %4051, align 1
  %4052 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %405
  store <16 x i8> %4034, ptr addrspace(3) %4052, align 1
  %4053 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %414
  store <16 x i8> %4040, ptr addrspace(3) %4053, align 1
  %4054 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %423
  store <16 x i8> %4046, ptr addrspace(3) %4054, align 1
  %4055 = bitcast <16 x i8> %3948 to <4 x i32>
  %4056 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3854, <4 x i32> %4055, <4 x i32> %3681, i32 %3983, i32 %3741)
  %4057 = bitcast <16 x i8> %3950 to <4 x i32>
  %4058 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4056, <4 x i32> %4057, <4 x i32> %3684, i32 %3983, i32 %3741)
  %4059 = bitcast <16 x i8> %3952 to <4 x i32>
  %4060 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3858, <4 x i32> %4059, <4 x i32> %3681, i32 %3983, i32 %3741)
  %4061 = bitcast <16 x i8> %3954 to <4 x i32>
  %4062 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4060, <4 x i32> %4061, <4 x i32> %3684, i32 %3983, i32 %3741)
  %4063 = bitcast <16 x i8> %3956 to <4 x i32>
  %4064 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3862, <4 x i32> %4063, <4 x i32> %3681, i32 %3988, i32 %3741)
  %4065 = bitcast <16 x i8> %3958 to <4 x i32>
  %4066 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4064, <4 x i32> %4065, <4 x i32> %3684, i32 %3988, i32 %3741)
  %4067 = bitcast <16 x i8> %3960 to <4 x i32>
  %4068 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3866, <4 x i32> %4067, <4 x i32> %3681, i32 %3988, i32 %3741)
  %4069 = bitcast <16 x i8> %3962 to <4 x i32>
  %4070 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4068, <4 x i32> %4069, <4 x i32> %3684, i32 %3988, i32 %3741)
  %4071 = bitcast <16 x i8> %3964 to <4 x i32>
  %4072 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3870, <4 x i32> %4071, <4 x i32> %3681, i32 %3993, i32 %3741)
  %4073 = bitcast <16 x i8> %3966 to <4 x i32>
  %4074 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4072, <4 x i32> %4073, <4 x i32> %3684, i32 %3993, i32 %3741)
  %4075 = bitcast <16 x i8> %3968 to <4 x i32>
  %4076 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3874, <4 x i32> %4075, <4 x i32> %3681, i32 %3993, i32 %3741)
  %4077 = bitcast <16 x i8> %3970 to <4 x i32>
  %4078 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4076, <4 x i32> %4077, <4 x i32> %3684, i32 %3993, i32 %3741)
  %4079 = bitcast <16 x i8> %3972 to <4 x i32>
  %4080 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3878, <4 x i32> %4079, <4 x i32> %3681, i32 %3998, i32 %3741)
  %4081 = bitcast <16 x i8> %3974 to <4 x i32>
  %4082 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4080, <4 x i32> %4081, <4 x i32> %3684, i32 %3998, i32 %3741)
  %4083 = bitcast <16 x i8> %3976 to <4 x i32>
  %4084 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3882, <4 x i32> %4083, <4 x i32> %3681, i32 %3998, i32 %3741)
  %4085 = bitcast <16 x i8> %3978 to <4 x i32>
  %4086 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4084, <4 x i32> %4085, <4 x i32> %3684, i32 %3998, i32 %3741)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4087 = add i32 %192, 9216
  %4088 = mul i32 %4087, 4
  %4089 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4088, i32 %176, i32 2)
  %4090 = add i32 %192, 9472
  %4091 = mul i32 %4090, 4
  %4092 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4091, i32 %176, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4093 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3890, <4 x i32> %4055, <4 x i32> %3701, i32 %3983, i32 %3741)
  %4094 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4093, <4 x i32> %4057, <4 x i32> %3702, i32 %3983, i32 %3741)
  %4095 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3892, <4 x i32> %4059, <4 x i32> %3701, i32 %3983, i32 %3741)
  %4096 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4095, <4 x i32> %4061, <4 x i32> %3702, i32 %3983, i32 %3741)
  %4097 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3894, <4 x i32> %4063, <4 x i32> %3701, i32 %3988, i32 %3741)
  %4098 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4097, <4 x i32> %4065, <4 x i32> %3702, i32 %3988, i32 %3741)
  %4099 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3896, <4 x i32> %4067, <4 x i32> %3701, i32 %3988, i32 %3741)
  %4100 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4099, <4 x i32> %4069, <4 x i32> %3702, i32 %3988, i32 %3741)
  %4101 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3898, <4 x i32> %4071, <4 x i32> %3701, i32 %3993, i32 %3741)
  %4102 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4101, <4 x i32> %4073, <4 x i32> %3702, i32 %3993, i32 %3741)
  %4103 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3900, <4 x i32> %4075, <4 x i32> %3701, i32 %3993, i32 %3741)
  %4104 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4103, <4 x i32> %4077, <4 x i32> %3702, i32 %3993, i32 %3741)
  %4105 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3902, <4 x i32> %4079, <4 x i32> %3701, i32 %3998, i32 %3741)
  %4106 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4105, <4 x i32> %4081, <4 x i32> %3702, i32 %3998, i32 %3741)
  %4107 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3904, <4 x i32> %4083, <4 x i32> %3701, i32 %3998, i32 %3741)
  %4108 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4107, <4 x i32> %4085, <4 x i32> %3702, i32 %3998, i32 %3741)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4109 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4088, i32 %180, i32 2)
  %4110 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4091, i32 %180, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4111 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3908, <4 x i32> %4055, <4 x i32> %3719, i32 %3983, i32 %3742)
  %4112 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4111, <4 x i32> %4057, <4 x i32> %3720, i32 %3983, i32 %3742)
  %4113 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3910, <4 x i32> %4059, <4 x i32> %3719, i32 %3983, i32 %3742)
  %4114 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4113, <4 x i32> %4061, <4 x i32> %3720, i32 %3983, i32 %3742)
  %4115 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3912, <4 x i32> %4063, <4 x i32> %3719, i32 %3988, i32 %3742)
  %4116 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4115, <4 x i32> %4065, <4 x i32> %3720, i32 %3988, i32 %3742)
  %4117 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3914, <4 x i32> %4067, <4 x i32> %3719, i32 %3988, i32 %3742)
  %4118 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4117, <4 x i32> %4069, <4 x i32> %3720, i32 %3988, i32 %3742)
  %4119 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3916, <4 x i32> %4071, <4 x i32> %3719, i32 %3993, i32 %3742)
  %4120 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4119, <4 x i32> %4073, <4 x i32> %3720, i32 %3993, i32 %3742)
  %4121 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3918, <4 x i32> %4075, <4 x i32> %3719, i32 %3993, i32 %3742)
  %4122 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4121, <4 x i32> %4077, <4 x i32> %3720, i32 %3993, i32 %3742)
  %4123 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3920, <4 x i32> %4079, <4 x i32> %3719, i32 %3998, i32 %3742)
  %4124 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4123, <4 x i32> %4081, <4 x i32> %3720, i32 %3998, i32 %3742)
  %4125 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3922, <4 x i32> %4083, <4 x i32> %3719, i32 %3998, i32 %3742)
  %4126 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4125, <4 x i32> %4085, <4 x i32> %3720, i32 %3998, i32 %3742)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4127 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4088, i32 %184, i32 2)
  %4128 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4091, i32 %184, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4129 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3926, <4 x i32> %4055, <4 x i32> %3737, i32 %3983, i32 %3742)
  %4130 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4129, <4 x i32> %4057, <4 x i32> %3738, i32 %3983, i32 %3742)
  %4131 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3928, <4 x i32> %4059, <4 x i32> %3737, i32 %3983, i32 %3742)
  %4132 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4131, <4 x i32> %4061, <4 x i32> %3738, i32 %3983, i32 %3742)
  %4133 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3930, <4 x i32> %4063, <4 x i32> %3737, i32 %3988, i32 %3742)
  %4134 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4133, <4 x i32> %4065, <4 x i32> %3738, i32 %3988, i32 %3742)
  %4135 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3932, <4 x i32> %4067, <4 x i32> %3737, i32 %3988, i32 %3742)
  %4136 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4135, <4 x i32> %4069, <4 x i32> %3738, i32 %3988, i32 %3742)
  %4137 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3934, <4 x i32> %4071, <4 x i32> %3737, i32 %3993, i32 %3742)
  %4138 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4137, <4 x i32> %4073, <4 x i32> %3738, i32 %3993, i32 %3742)
  %4139 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3936, <4 x i32> %4075, <4 x i32> %3737, i32 %3993, i32 %3742)
  %4140 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4139, <4 x i32> %4077, <4 x i32> %3738, i32 %3993, i32 %3742)
  %4141 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3938, <4 x i32> %4079, <4 x i32> %3737, i32 %3998, i32 %3742)
  %4142 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4141, <4 x i32> %4081, <4 x i32> %3738, i32 %3998, i32 %3742)
  %4143 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %3940, <4 x i32> %4083, <4 x i32> %3737, i32 %3998, i32 %3742)
  %4144 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4143, <4 x i32> %4085, <4 x i32> %3738, i32 %3998, i32 %3742)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4145 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4088, i32 %188, i32 2)
  %4146 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4091, i32 %188, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4147 = add i32 %211, 1152
  %4148 = mul i32 %4147, 4
  %4149 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %4148, i32 %204, i32 0)
  %4150 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %4148, i32 %210, i32 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %4151 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1075
  %4152 = load <16 x i8>, ptr addrspace(3) %4151, align 1
  %4153 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1078
  %4154 = load <16 x i8>, ptr addrspace(3) %4153, align 1
  %4155 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1081
  %4156 = load <16 x i8>, ptr addrspace(3) %4155, align 1
  %4157 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1084
  %4158 = load <16 x i8>, ptr addrspace(3) %4157, align 1
  %4159 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1087
  %4160 = load <16 x i8>, ptr addrspace(3) %4159, align 1
  %4161 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1090
  %4162 = load <16 x i8>, ptr addrspace(3) %4161, align 1
  %4163 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1093
  %4164 = load <16 x i8>, ptr addrspace(3) %4163, align 1
  %4165 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1096
  %4166 = load <16 x i8>, ptr addrspace(3) %4165, align 1
  %4167 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1099
  %4168 = load <16 x i8>, ptr addrspace(3) %4167, align 1
  %4169 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1102
  %4170 = load <16 x i8>, ptr addrspace(3) %4169, align 1
  %4171 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1105
  %4172 = load <16 x i8>, ptr addrspace(3) %4171, align 1
  %4173 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1108
  %4174 = load <16 x i8>, ptr addrspace(3) %4173, align 1
  %4175 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1111
  %4176 = load <16 x i8>, ptr addrspace(3) %4175, align 1
  %4177 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1114
  %4178 = load <16 x i8>, ptr addrspace(3) %4177, align 1
  %4179 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1117
  %4180 = load <16 x i8>, ptr addrspace(3) %4179, align 1
  %4181 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1120
  %4182 = load <16 x i8>, ptr addrspace(3) %4181, align 1
  %4183 = add i64 %193, 1088
  %4184 = add i64 %4183, %24
  %4185 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %4184
  %4186 = load <1 x i32>, ptr addrspace(3) %4185, align 4
  %4187 = extractelement <1 x i32> %4186, i64 0
  %4188 = add i64 %193, 2880
  %4189 = add i64 %4188, %24
  %4190 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %4189
  %4191 = load <1 x i32>, ptr addrspace(3) %4190, align 4
  %4192 = extractelement <1 x i32> %4191, i64 0
  %4193 = add i64 %193, 4672
  %4194 = add i64 %4193, %24
  %4195 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %4194
  %4196 = load <1 x i32>, ptr addrspace(3) %4195, align 4
  %4197 = extractelement <1 x i32> %4196, i64 0
  %4198 = add i64 %193, 6464
  %4199 = add i64 %4198, %24
  %4200 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %4199
  %4201 = load <1 x i32>, ptr addrspace(3) %4200, align 4
  %4202 = extractelement <1 x i32> %4201, i64 0
  %4203 = add i64 %77, 608
  %4204 = add i64 %4203, %71
  %4205 = trunc i64 %4204 to i32
  %4206 = mul i32 %4205, 4
  %4207 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4206, i32 0, i32 0)
  %4208 = bitcast <4 x i32> %4207 to <16 x i8>
  %4209 = add i64 %90, 608
  %4210 = add i64 %4209, %84
  %4211 = trunc i64 %4210 to i32
  %4212 = mul i32 %4211, 4
  %4213 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4212, i32 0, i32 0)
  %4214 = bitcast <4 x i32> %4213 to <16 x i8>
  %4215 = add i64 %103, 608
  %4216 = add i64 %4215, %97
  %4217 = trunc i64 %4216 to i32
  %4218 = mul i32 %4217, 4
  %4219 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4218, i32 0, i32 0)
  %4220 = bitcast <4 x i32> %4219 to <16 x i8>
  %4221 = add i64 %116, 608
  %4222 = add i64 %4221, %110
  %4223 = trunc i64 %4222 to i32
  %4224 = mul i32 %4223, 4
  %4225 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4224, i32 0, i32 0)
  %4226 = bitcast <4 x i32> %4225 to <16 x i8>
  %4227 = add i64 %129, 608
  %4228 = add i64 %4227, %123
  %4229 = trunc i64 %4228 to i32
  %4230 = mul i32 %4229, 4
  %4231 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4230, i32 0, i32 0)
  %4232 = bitcast <4 x i32> %4231 to <16 x i8>
  %4233 = add i64 %142, 608
  %4234 = add i64 %4233, %136
  %4235 = trunc i64 %4234 to i32
  %4236 = mul i32 %4235, 4
  %4237 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4236, i32 0, i32 0)
  %4238 = bitcast <4 x i32> %4237 to <16 x i8>
  %4239 = add i64 %155, 608
  %4240 = add i64 %4239, %149
  %4241 = trunc i64 %4240 to i32
  %4242 = mul i32 %4241, 4
  %4243 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4242, i32 0, i32 0)
  %4244 = bitcast <4 x i32> %4243 to <16 x i8>
  %4245 = add i64 %168, 608
  %4246 = add i64 %4245, %162
  %4247 = trunc i64 %4246 to i32
  %4248 = mul i32 %4247, 4
  %4249 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4248, i32 0, i32 0)
  %4250 = bitcast <4 x i32> %4249 to <16 x i8>
  %4251 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %487
  store <16 x i8> %4208, ptr addrspace(3) %4251, align 1
  %4252 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %489
  store <16 x i8> %4214, ptr addrspace(3) %4252, align 1
  %4253 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %491
  store <16 x i8> %4220, ptr addrspace(3) %4253, align 1
  %4254 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %493
  store <16 x i8> %4226, ptr addrspace(3) %4254, align 1
  %4255 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %495
  store <16 x i8> %4232, ptr addrspace(3) %4255, align 1
  %4256 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %497
  store <16 x i8> %4238, ptr addrspace(3) %4256, align 1
  %4257 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %499
  store <16 x i8> %4244, ptr addrspace(3) %4257, align 1
  %4258 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %501
  store <16 x i8> %4250, ptr addrspace(3) %4258, align 1
  %4259 = bitcast <16 x i8> %4152 to <4 x i32>
  %4260 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4058, <4 x i32> %4259, <4 x i32> %3885, i32 %4187, i32 %3945)
  %4261 = bitcast <16 x i8> %4154 to <4 x i32>
  %4262 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4260, <4 x i32> %4261, <4 x i32> %3888, i32 %4187, i32 %3945)
  %4263 = bitcast <16 x i8> %4156 to <4 x i32>
  %4264 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4062, <4 x i32> %4263, <4 x i32> %3885, i32 %4187, i32 %3945)
  %4265 = bitcast <16 x i8> %4158 to <4 x i32>
  %4266 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4264, <4 x i32> %4265, <4 x i32> %3888, i32 %4187, i32 %3945)
  %4267 = bitcast <16 x i8> %4160 to <4 x i32>
  %4268 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4066, <4 x i32> %4267, <4 x i32> %3885, i32 %4192, i32 %3945)
  %4269 = bitcast <16 x i8> %4162 to <4 x i32>
  %4270 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4268, <4 x i32> %4269, <4 x i32> %3888, i32 %4192, i32 %3945)
  %4271 = bitcast <16 x i8> %4164 to <4 x i32>
  %4272 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4070, <4 x i32> %4271, <4 x i32> %3885, i32 %4192, i32 %3945)
  %4273 = bitcast <16 x i8> %4166 to <4 x i32>
  %4274 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4272, <4 x i32> %4273, <4 x i32> %3888, i32 %4192, i32 %3945)
  %4275 = bitcast <16 x i8> %4168 to <4 x i32>
  %4276 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4074, <4 x i32> %4275, <4 x i32> %3885, i32 %4197, i32 %3945)
  %4277 = bitcast <16 x i8> %4170 to <4 x i32>
  %4278 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4276, <4 x i32> %4277, <4 x i32> %3888, i32 %4197, i32 %3945)
  %4279 = bitcast <16 x i8> %4172 to <4 x i32>
  %4280 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4078, <4 x i32> %4279, <4 x i32> %3885, i32 %4197, i32 %3945)
  %4281 = bitcast <16 x i8> %4174 to <4 x i32>
  %4282 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4280, <4 x i32> %4281, <4 x i32> %3888, i32 %4197, i32 %3945)
  %4283 = bitcast <16 x i8> %4176 to <4 x i32>
  %4284 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4082, <4 x i32> %4283, <4 x i32> %3885, i32 %4202, i32 %3945)
  %4285 = bitcast <16 x i8> %4178 to <4 x i32>
  %4286 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4284, <4 x i32> %4285, <4 x i32> %3888, i32 %4202, i32 %3945)
  %4287 = bitcast <16 x i8> %4180 to <4 x i32>
  %4288 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4086, <4 x i32> %4287, <4 x i32> %3885, i32 %4202, i32 %3945)
  %4289 = bitcast <16 x i8> %4182 to <4 x i32>
  %4290 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4288, <4 x i32> %4289, <4 x i32> %3888, i32 %4202, i32 %3945)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4291 = add i32 %192, 9728
  %4292 = mul i32 %4291, 4
  %4293 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4292, i32 %176, i32 2)
  %4294 = add i32 %192, 9984
  %4295 = mul i32 %4294, 4
  %4296 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4295, i32 %176, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4297 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4094, <4 x i32> %4259, <4 x i32> %3905, i32 %4187, i32 %3945)
  %4298 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4297, <4 x i32> %4261, <4 x i32> %3906, i32 %4187, i32 %3945)
  %4299 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4096, <4 x i32> %4263, <4 x i32> %3905, i32 %4187, i32 %3945)
  %4300 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4299, <4 x i32> %4265, <4 x i32> %3906, i32 %4187, i32 %3945)
  %4301 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4098, <4 x i32> %4267, <4 x i32> %3905, i32 %4192, i32 %3945)
  %4302 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4301, <4 x i32> %4269, <4 x i32> %3906, i32 %4192, i32 %3945)
  %4303 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4100, <4 x i32> %4271, <4 x i32> %3905, i32 %4192, i32 %3945)
  %4304 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4303, <4 x i32> %4273, <4 x i32> %3906, i32 %4192, i32 %3945)
  %4305 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4102, <4 x i32> %4275, <4 x i32> %3905, i32 %4197, i32 %3945)
  %4306 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4305, <4 x i32> %4277, <4 x i32> %3906, i32 %4197, i32 %3945)
  %4307 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4104, <4 x i32> %4279, <4 x i32> %3905, i32 %4197, i32 %3945)
  %4308 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4307, <4 x i32> %4281, <4 x i32> %3906, i32 %4197, i32 %3945)
  %4309 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4106, <4 x i32> %4283, <4 x i32> %3905, i32 %4202, i32 %3945)
  %4310 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4309, <4 x i32> %4285, <4 x i32> %3906, i32 %4202, i32 %3945)
  %4311 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4108, <4 x i32> %4287, <4 x i32> %3905, i32 %4202, i32 %3945)
  %4312 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4311, <4 x i32> %4289, <4 x i32> %3906, i32 %4202, i32 %3945)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4313 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4292, i32 %180, i32 2)
  %4314 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4295, i32 %180, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4315 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4112, <4 x i32> %4259, <4 x i32> %3923, i32 %4187, i32 %3946)
  %4316 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4315, <4 x i32> %4261, <4 x i32> %3924, i32 %4187, i32 %3946)
  %4317 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4114, <4 x i32> %4263, <4 x i32> %3923, i32 %4187, i32 %3946)
  %4318 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4317, <4 x i32> %4265, <4 x i32> %3924, i32 %4187, i32 %3946)
  %4319 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4116, <4 x i32> %4267, <4 x i32> %3923, i32 %4192, i32 %3946)
  %4320 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4319, <4 x i32> %4269, <4 x i32> %3924, i32 %4192, i32 %3946)
  %4321 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4118, <4 x i32> %4271, <4 x i32> %3923, i32 %4192, i32 %3946)
  %4322 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4321, <4 x i32> %4273, <4 x i32> %3924, i32 %4192, i32 %3946)
  %4323 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4120, <4 x i32> %4275, <4 x i32> %3923, i32 %4197, i32 %3946)
  %4324 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4323, <4 x i32> %4277, <4 x i32> %3924, i32 %4197, i32 %3946)
  %4325 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4122, <4 x i32> %4279, <4 x i32> %3923, i32 %4197, i32 %3946)
  %4326 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4325, <4 x i32> %4281, <4 x i32> %3924, i32 %4197, i32 %3946)
  %4327 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4124, <4 x i32> %4283, <4 x i32> %3923, i32 %4202, i32 %3946)
  %4328 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4327, <4 x i32> %4285, <4 x i32> %3924, i32 %4202, i32 %3946)
  %4329 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4126, <4 x i32> %4287, <4 x i32> %3923, i32 %4202, i32 %3946)
  %4330 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4329, <4 x i32> %4289, <4 x i32> %3924, i32 %4202, i32 %3946)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4331 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4292, i32 %184, i32 2)
  %4332 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4295, i32 %184, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4333 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4130, <4 x i32> %4259, <4 x i32> %3941, i32 %4187, i32 %3946)
  %4334 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4333, <4 x i32> %4261, <4 x i32> %3942, i32 %4187, i32 %3946)
  %4335 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4132, <4 x i32> %4263, <4 x i32> %3941, i32 %4187, i32 %3946)
  %4336 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4335, <4 x i32> %4265, <4 x i32> %3942, i32 %4187, i32 %3946)
  %4337 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4134, <4 x i32> %4267, <4 x i32> %3941, i32 %4192, i32 %3946)
  %4338 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4337, <4 x i32> %4269, <4 x i32> %3942, i32 %4192, i32 %3946)
  %4339 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4136, <4 x i32> %4271, <4 x i32> %3941, i32 %4192, i32 %3946)
  %4340 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4339, <4 x i32> %4273, <4 x i32> %3942, i32 %4192, i32 %3946)
  %4341 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4138, <4 x i32> %4275, <4 x i32> %3941, i32 %4197, i32 %3946)
  %4342 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4341, <4 x i32> %4277, <4 x i32> %3942, i32 %4197, i32 %3946)
  %4343 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4140, <4 x i32> %4279, <4 x i32> %3941, i32 %4197, i32 %3946)
  %4344 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4343, <4 x i32> %4281, <4 x i32> %3942, i32 %4197, i32 %3946)
  %4345 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4142, <4 x i32> %4283, <4 x i32> %3941, i32 %4202, i32 %3946)
  %4346 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4345, <4 x i32> %4285, <4 x i32> %3942, i32 %4202, i32 %3946)
  %4347 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4144, <4 x i32> %4287, <4 x i32> %3941, i32 %4202, i32 %3946)
  %4348 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4347, <4 x i32> %4289, <4 x i32> %3942, i32 %4202, i32 %3946)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4349 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4292, i32 %188, i32 2)
  %4350 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4295, i32 %188, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4351 = add i32 %211, 1216
  %4352 = mul i32 %4351, 4
  %4353 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %4352, i32 %204, i32 0)
  %4354 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %4352, i32 %210, i32 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %4355 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %522
  %4356 = load <16 x i8>, ptr addrspace(3) %4355, align 1
  %4357 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %527
  %4358 = load <16 x i8>, ptr addrspace(3) %4357, align 1
  %4359 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %535
  %4360 = load <16 x i8>, ptr addrspace(3) %4359, align 1
  %4361 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %539
  %4362 = load <16 x i8>, ptr addrspace(3) %4361, align 1
  %4363 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %547
  %4364 = load <16 x i8>, ptr addrspace(3) %4363, align 1
  %4365 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %551
  %4366 = load <16 x i8>, ptr addrspace(3) %4365, align 1
  %4367 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %559
  %4368 = load <16 x i8>, ptr addrspace(3) %4367, align 1
  %4369 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %563
  %4370 = load <16 x i8>, ptr addrspace(3) %4369, align 1
  %4371 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %571
  %4372 = load <16 x i8>, ptr addrspace(3) %4371, align 1
  %4373 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %575
  %4374 = load <16 x i8>, ptr addrspace(3) %4373, align 1
  %4375 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %583
  %4376 = load <16 x i8>, ptr addrspace(3) %4375, align 1
  %4377 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %587
  %4378 = load <16 x i8>, ptr addrspace(3) %4377, align 1
  %4379 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %595
  %4380 = load <16 x i8>, ptr addrspace(3) %4379, align 1
  %4381 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %599
  %4382 = load <16 x i8>, ptr addrspace(3) %4381, align 1
  %4383 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %607
  %4384 = load <16 x i8>, ptr addrspace(3) %4383, align 1
  %4385 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %611
  %4386 = load <16 x i8>, ptr addrspace(3) %4385, align 1
  %4387 = add i64 %193, 1152
  %4388 = add i64 %4387, %24
  %4389 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %4388
  %4390 = load <1 x i32>, ptr addrspace(3) %4389, align 4
  %4391 = extractelement <1 x i32> %4390, i64 0
  %4392 = add i64 %193, 2944
  %4393 = add i64 %4392, %24
  %4394 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %4393
  %4395 = load <1 x i32>, ptr addrspace(3) %4394, align 4
  %4396 = extractelement <1 x i32> %4395, i64 0
  %4397 = add i64 %193, 4736
  %4398 = add i64 %4397, %24
  %4399 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %4398
  %4400 = load <1 x i32>, ptr addrspace(3) %4399, align 4
  %4401 = extractelement <1 x i32> %4400, i64 0
  %4402 = add i64 %193, 6528
  %4403 = add i64 %4402, %24
  %4404 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %4403
  %4405 = load <1 x i32>, ptr addrspace(3) %4404, align 4
  %4406 = extractelement <1 x i32> %4405, i64 0
  %4407 = add i64 %77, 640
  %4408 = add i64 %4407, %71
  %4409 = trunc i64 %4408 to i32
  %4410 = mul i32 %4409, 4
  %4411 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4410, i32 0, i32 0)
  %4412 = bitcast <4 x i32> %4411 to <16 x i8>
  %4413 = add i64 %90, 640
  %4414 = add i64 %4413, %84
  %4415 = trunc i64 %4414 to i32
  %4416 = mul i32 %4415, 4
  %4417 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4416, i32 0, i32 0)
  %4418 = bitcast <4 x i32> %4417 to <16 x i8>
  %4419 = add i64 %103, 640
  %4420 = add i64 %4419, %97
  %4421 = trunc i64 %4420 to i32
  %4422 = mul i32 %4421, 4
  %4423 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4422, i32 0, i32 0)
  %4424 = bitcast <4 x i32> %4423 to <16 x i8>
  %4425 = add i64 %116, 640
  %4426 = add i64 %4425, %110
  %4427 = trunc i64 %4426 to i32
  %4428 = mul i32 %4427, 4
  %4429 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4428, i32 0, i32 0)
  %4430 = bitcast <4 x i32> %4429 to <16 x i8>
  %4431 = add i64 %129, 640
  %4432 = add i64 %4431, %123
  %4433 = trunc i64 %4432 to i32
  %4434 = mul i32 %4433, 4
  %4435 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4434, i32 0, i32 0)
  %4436 = bitcast <4 x i32> %4435 to <16 x i8>
  %4437 = add i64 %142, 640
  %4438 = add i64 %4437, %136
  %4439 = trunc i64 %4438 to i32
  %4440 = mul i32 %4439, 4
  %4441 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4440, i32 0, i32 0)
  %4442 = bitcast <4 x i32> %4441 to <16 x i8>
  %4443 = add i64 %155, 640
  %4444 = add i64 %4443, %149
  %4445 = trunc i64 %4444 to i32
  %4446 = mul i32 %4445, 4
  %4447 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4446, i32 0, i32 0)
  %4448 = bitcast <4 x i32> %4447 to <16 x i8>
  %4449 = add i64 %168, 640
  %4450 = add i64 %4449, %162
  %4451 = trunc i64 %4450 to i32
  %4452 = mul i32 %4451, 4
  %4453 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4452, i32 0, i32 0)
  %4454 = bitcast <4 x i32> %4453 to <16 x i8>
  %4455 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %680
  store <16 x i8> %4412, ptr addrspace(3) %4455, align 1
  %4456 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %682
  store <16 x i8> %4418, ptr addrspace(3) %4456, align 1
  %4457 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %684
  store <16 x i8> %4424, ptr addrspace(3) %4457, align 1
  %4458 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %686
  store <16 x i8> %4430, ptr addrspace(3) %4458, align 1
  %4459 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %688
  store <16 x i8> %4436, ptr addrspace(3) %4459, align 1
  %4460 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %690
  store <16 x i8> %4442, ptr addrspace(3) %4460, align 1
  %4461 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %692
  store <16 x i8> %4448, ptr addrspace(3) %4461, align 1
  %4462 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %694
  store <16 x i8> %4454, ptr addrspace(3) %4462, align 1
  %4463 = bitcast <16 x i8> %4356 to <4 x i32>
  %4464 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4262, <4 x i32> %4463, <4 x i32> %4089, i32 %4391, i32 %4149)
  %4465 = bitcast <16 x i8> %4358 to <4 x i32>
  %4466 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4464, <4 x i32> %4465, <4 x i32> %4092, i32 %4391, i32 %4149)
  %4467 = bitcast <16 x i8> %4360 to <4 x i32>
  %4468 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4266, <4 x i32> %4467, <4 x i32> %4089, i32 %4391, i32 %4149)
  %4469 = bitcast <16 x i8> %4362 to <4 x i32>
  %4470 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4468, <4 x i32> %4469, <4 x i32> %4092, i32 %4391, i32 %4149)
  %4471 = bitcast <16 x i8> %4364 to <4 x i32>
  %4472 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4270, <4 x i32> %4471, <4 x i32> %4089, i32 %4396, i32 %4149)
  %4473 = bitcast <16 x i8> %4366 to <4 x i32>
  %4474 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4472, <4 x i32> %4473, <4 x i32> %4092, i32 %4396, i32 %4149)
  %4475 = bitcast <16 x i8> %4368 to <4 x i32>
  %4476 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4274, <4 x i32> %4475, <4 x i32> %4089, i32 %4396, i32 %4149)
  %4477 = bitcast <16 x i8> %4370 to <4 x i32>
  %4478 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4476, <4 x i32> %4477, <4 x i32> %4092, i32 %4396, i32 %4149)
  %4479 = bitcast <16 x i8> %4372 to <4 x i32>
  %4480 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4278, <4 x i32> %4479, <4 x i32> %4089, i32 %4401, i32 %4149)
  %4481 = bitcast <16 x i8> %4374 to <4 x i32>
  %4482 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4480, <4 x i32> %4481, <4 x i32> %4092, i32 %4401, i32 %4149)
  %4483 = bitcast <16 x i8> %4376 to <4 x i32>
  %4484 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4282, <4 x i32> %4483, <4 x i32> %4089, i32 %4401, i32 %4149)
  %4485 = bitcast <16 x i8> %4378 to <4 x i32>
  %4486 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4484, <4 x i32> %4485, <4 x i32> %4092, i32 %4401, i32 %4149)
  %4487 = bitcast <16 x i8> %4380 to <4 x i32>
  %4488 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4286, <4 x i32> %4487, <4 x i32> %4089, i32 %4406, i32 %4149)
  %4489 = bitcast <16 x i8> %4382 to <4 x i32>
  %4490 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4488, <4 x i32> %4489, <4 x i32> %4092, i32 %4406, i32 %4149)
  %4491 = bitcast <16 x i8> %4384 to <4 x i32>
  %4492 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4290, <4 x i32> %4491, <4 x i32> %4089, i32 %4406, i32 %4149)
  %4493 = bitcast <16 x i8> %4386 to <4 x i32>
  %4494 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4492, <4 x i32> %4493, <4 x i32> %4092, i32 %4406, i32 %4149)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4495 = add i32 %192, 10240
  %4496 = mul i32 %4495, 4
  %4497 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4496, i32 %176, i32 2)
  %4498 = add i32 %192, 10496
  %4499 = mul i32 %4498, 4
  %4500 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4499, i32 %176, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4501 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4298, <4 x i32> %4463, <4 x i32> %4109, i32 %4391, i32 %4149)
  %4502 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4501, <4 x i32> %4465, <4 x i32> %4110, i32 %4391, i32 %4149)
  %4503 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4300, <4 x i32> %4467, <4 x i32> %4109, i32 %4391, i32 %4149)
  %4504 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4503, <4 x i32> %4469, <4 x i32> %4110, i32 %4391, i32 %4149)
  %4505 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4302, <4 x i32> %4471, <4 x i32> %4109, i32 %4396, i32 %4149)
  %4506 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4505, <4 x i32> %4473, <4 x i32> %4110, i32 %4396, i32 %4149)
  %4507 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4304, <4 x i32> %4475, <4 x i32> %4109, i32 %4396, i32 %4149)
  %4508 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4507, <4 x i32> %4477, <4 x i32> %4110, i32 %4396, i32 %4149)
  %4509 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4306, <4 x i32> %4479, <4 x i32> %4109, i32 %4401, i32 %4149)
  %4510 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4509, <4 x i32> %4481, <4 x i32> %4110, i32 %4401, i32 %4149)
  %4511 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4308, <4 x i32> %4483, <4 x i32> %4109, i32 %4401, i32 %4149)
  %4512 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4511, <4 x i32> %4485, <4 x i32> %4110, i32 %4401, i32 %4149)
  %4513 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4310, <4 x i32> %4487, <4 x i32> %4109, i32 %4406, i32 %4149)
  %4514 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4513, <4 x i32> %4489, <4 x i32> %4110, i32 %4406, i32 %4149)
  %4515 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4312, <4 x i32> %4491, <4 x i32> %4109, i32 %4406, i32 %4149)
  %4516 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4515, <4 x i32> %4493, <4 x i32> %4110, i32 %4406, i32 %4149)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4517 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4496, i32 %180, i32 2)
  %4518 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4499, i32 %180, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4519 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4316, <4 x i32> %4463, <4 x i32> %4127, i32 %4391, i32 %4150)
  %4520 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4519, <4 x i32> %4465, <4 x i32> %4128, i32 %4391, i32 %4150)
  %4521 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4318, <4 x i32> %4467, <4 x i32> %4127, i32 %4391, i32 %4150)
  %4522 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4521, <4 x i32> %4469, <4 x i32> %4128, i32 %4391, i32 %4150)
  %4523 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4320, <4 x i32> %4471, <4 x i32> %4127, i32 %4396, i32 %4150)
  %4524 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4523, <4 x i32> %4473, <4 x i32> %4128, i32 %4396, i32 %4150)
  %4525 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4322, <4 x i32> %4475, <4 x i32> %4127, i32 %4396, i32 %4150)
  %4526 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4525, <4 x i32> %4477, <4 x i32> %4128, i32 %4396, i32 %4150)
  %4527 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4324, <4 x i32> %4479, <4 x i32> %4127, i32 %4401, i32 %4150)
  %4528 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4527, <4 x i32> %4481, <4 x i32> %4128, i32 %4401, i32 %4150)
  %4529 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4326, <4 x i32> %4483, <4 x i32> %4127, i32 %4401, i32 %4150)
  %4530 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4529, <4 x i32> %4485, <4 x i32> %4128, i32 %4401, i32 %4150)
  %4531 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4328, <4 x i32> %4487, <4 x i32> %4127, i32 %4406, i32 %4150)
  %4532 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4531, <4 x i32> %4489, <4 x i32> %4128, i32 %4406, i32 %4150)
  %4533 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4330, <4 x i32> %4491, <4 x i32> %4127, i32 %4406, i32 %4150)
  %4534 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4533, <4 x i32> %4493, <4 x i32> %4128, i32 %4406, i32 %4150)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4535 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4496, i32 %184, i32 2)
  %4536 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4499, i32 %184, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4537 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4334, <4 x i32> %4463, <4 x i32> %4145, i32 %4391, i32 %4150)
  %4538 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4537, <4 x i32> %4465, <4 x i32> %4146, i32 %4391, i32 %4150)
  %4539 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4336, <4 x i32> %4467, <4 x i32> %4145, i32 %4391, i32 %4150)
  %4540 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4539, <4 x i32> %4469, <4 x i32> %4146, i32 %4391, i32 %4150)
  %4541 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4338, <4 x i32> %4471, <4 x i32> %4145, i32 %4396, i32 %4150)
  %4542 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4541, <4 x i32> %4473, <4 x i32> %4146, i32 %4396, i32 %4150)
  %4543 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4340, <4 x i32> %4475, <4 x i32> %4145, i32 %4396, i32 %4150)
  %4544 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4543, <4 x i32> %4477, <4 x i32> %4146, i32 %4396, i32 %4150)
  %4545 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4342, <4 x i32> %4479, <4 x i32> %4145, i32 %4401, i32 %4150)
  %4546 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4545, <4 x i32> %4481, <4 x i32> %4146, i32 %4401, i32 %4150)
  %4547 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4344, <4 x i32> %4483, <4 x i32> %4145, i32 %4401, i32 %4150)
  %4548 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4547, <4 x i32> %4485, <4 x i32> %4146, i32 %4401, i32 %4150)
  %4549 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4346, <4 x i32> %4487, <4 x i32> %4145, i32 %4406, i32 %4150)
  %4550 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4549, <4 x i32> %4489, <4 x i32> %4146, i32 %4406, i32 %4150)
  %4551 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4348, <4 x i32> %4491, <4 x i32> %4145, i32 %4406, i32 %4150)
  %4552 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4551, <4 x i32> %4493, <4 x i32> %4146, i32 %4406, i32 %4150)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4553 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4496, i32 %188, i32 2)
  %4554 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4499, i32 %188, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4555 = add i32 %211, 1280
  %4556 = mul i32 %4555, 4
  %4557 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %4556, i32 %204, i32 0)
  %4558 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %4556, i32 %210, i32 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %4559 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %856
  %4560 = load <16 x i8>, ptr addrspace(3) %4559, align 1
  %4561 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %859
  %4562 = load <16 x i8>, ptr addrspace(3) %4561, align 1
  %4563 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %862
  %4564 = load <16 x i8>, ptr addrspace(3) %4563, align 1
  %4565 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %865
  %4566 = load <16 x i8>, ptr addrspace(3) %4565, align 1
  %4567 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %868
  %4568 = load <16 x i8>, ptr addrspace(3) %4567, align 1
  %4569 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %871
  %4570 = load <16 x i8>, ptr addrspace(3) %4569, align 1
  %4571 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %874
  %4572 = load <16 x i8>, ptr addrspace(3) %4571, align 1
  %4573 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %877
  %4574 = load <16 x i8>, ptr addrspace(3) %4573, align 1
  %4575 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %880
  %4576 = load <16 x i8>, ptr addrspace(3) %4575, align 1
  %4577 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %883
  %4578 = load <16 x i8>, ptr addrspace(3) %4577, align 1
  %4579 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %886
  %4580 = load <16 x i8>, ptr addrspace(3) %4579, align 1
  %4581 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %889
  %4582 = load <16 x i8>, ptr addrspace(3) %4581, align 1
  %4583 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %892
  %4584 = load <16 x i8>, ptr addrspace(3) %4583, align 1
  %4585 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %895
  %4586 = load <16 x i8>, ptr addrspace(3) %4585, align 1
  %4587 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %898
  %4588 = load <16 x i8>, ptr addrspace(3) %4587, align 1
  %4589 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %901
  %4590 = load <16 x i8>, ptr addrspace(3) %4589, align 1
  %4591 = add i64 %193, 1216
  %4592 = add i64 %4591, %24
  %4593 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %4592
  %4594 = load <1 x i32>, ptr addrspace(3) %4593, align 4
  %4595 = extractelement <1 x i32> %4594, i64 0
  %4596 = add i64 %193, 3008
  %4597 = add i64 %4596, %24
  %4598 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %4597
  %4599 = load <1 x i32>, ptr addrspace(3) %4598, align 4
  %4600 = extractelement <1 x i32> %4599, i64 0
  %4601 = add i64 %193, 4800
  %4602 = add i64 %4601, %24
  %4603 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %4602
  %4604 = load <1 x i32>, ptr addrspace(3) %4603, align 4
  %4605 = extractelement <1 x i32> %4604, i64 0
  %4606 = add i64 %193, 6592
  %4607 = add i64 %4606, %24
  %4608 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %4607
  %4609 = load <1 x i32>, ptr addrspace(3) %4608, align 4
  %4610 = extractelement <1 x i32> %4609, i64 0
  %4611 = add i64 %77, 672
  %4612 = add i64 %4611, %71
  %4613 = trunc i64 %4612 to i32
  %4614 = mul i32 %4613, 4
  %4615 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4614, i32 0, i32 0)
  %4616 = bitcast <4 x i32> %4615 to <16 x i8>
  %4617 = add i64 %90, 672
  %4618 = add i64 %4617, %84
  %4619 = trunc i64 %4618 to i32
  %4620 = mul i32 %4619, 4
  %4621 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4620, i32 0, i32 0)
  %4622 = bitcast <4 x i32> %4621 to <16 x i8>
  %4623 = add i64 %103, 672
  %4624 = add i64 %4623, %97
  %4625 = trunc i64 %4624 to i32
  %4626 = mul i32 %4625, 4
  %4627 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4626, i32 0, i32 0)
  %4628 = bitcast <4 x i32> %4627 to <16 x i8>
  %4629 = add i64 %116, 672
  %4630 = add i64 %4629, %110
  %4631 = trunc i64 %4630 to i32
  %4632 = mul i32 %4631, 4
  %4633 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4632, i32 0, i32 0)
  %4634 = bitcast <4 x i32> %4633 to <16 x i8>
  %4635 = add i64 %129, 672
  %4636 = add i64 %4635, %123
  %4637 = trunc i64 %4636 to i32
  %4638 = mul i32 %4637, 4
  %4639 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4638, i32 0, i32 0)
  %4640 = bitcast <4 x i32> %4639 to <16 x i8>
  %4641 = add i64 %142, 672
  %4642 = add i64 %4641, %136
  %4643 = trunc i64 %4642 to i32
  %4644 = mul i32 %4643, 4
  %4645 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4644, i32 0, i32 0)
  %4646 = bitcast <4 x i32> %4645 to <16 x i8>
  %4647 = add i64 %155, 672
  %4648 = add i64 %4647, %149
  %4649 = trunc i64 %4648 to i32
  %4650 = mul i32 %4649, 4
  %4651 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4650, i32 0, i32 0)
  %4652 = bitcast <4 x i32> %4651 to <16 x i8>
  %4653 = add i64 %168, 672
  %4654 = add i64 %4653, %162
  %4655 = trunc i64 %4654 to i32
  %4656 = mul i32 %4655, 4
  %4657 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4656, i32 0, i32 0)
  %4658 = bitcast <4 x i32> %4657 to <16 x i8>
  %4659 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %360
  store <16 x i8> %4616, ptr addrspace(3) %4659, align 1
  %4660 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %369
  store <16 x i8> %4622, ptr addrspace(3) %4660, align 1
  %4661 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %378
  store <16 x i8> %4628, ptr addrspace(3) %4661, align 1
  %4662 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %387
  store <16 x i8> %4634, ptr addrspace(3) %4662, align 1
  %4663 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %396
  store <16 x i8> %4640, ptr addrspace(3) %4663, align 1
  %4664 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %405
  store <16 x i8> %4646, ptr addrspace(3) %4664, align 1
  %4665 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %414
  store <16 x i8> %4652, ptr addrspace(3) %4665, align 1
  %4666 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %423
  store <16 x i8> %4658, ptr addrspace(3) %4666, align 1
  %4667 = bitcast <16 x i8> %4560 to <4 x i32>
  %4668 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4466, <4 x i32> %4667, <4 x i32> %4293, i32 %4595, i32 %4353)
  %4669 = bitcast <16 x i8> %4562 to <4 x i32>
  %4670 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4668, <4 x i32> %4669, <4 x i32> %4296, i32 %4595, i32 %4353)
  %4671 = bitcast <16 x i8> %4564 to <4 x i32>
  %4672 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4470, <4 x i32> %4671, <4 x i32> %4293, i32 %4595, i32 %4353)
  %4673 = bitcast <16 x i8> %4566 to <4 x i32>
  %4674 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4672, <4 x i32> %4673, <4 x i32> %4296, i32 %4595, i32 %4353)
  %4675 = bitcast <16 x i8> %4568 to <4 x i32>
  %4676 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4474, <4 x i32> %4675, <4 x i32> %4293, i32 %4600, i32 %4353)
  %4677 = bitcast <16 x i8> %4570 to <4 x i32>
  %4678 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4676, <4 x i32> %4677, <4 x i32> %4296, i32 %4600, i32 %4353)
  %4679 = bitcast <16 x i8> %4572 to <4 x i32>
  %4680 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4478, <4 x i32> %4679, <4 x i32> %4293, i32 %4600, i32 %4353)
  %4681 = bitcast <16 x i8> %4574 to <4 x i32>
  %4682 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4680, <4 x i32> %4681, <4 x i32> %4296, i32 %4600, i32 %4353)
  %4683 = bitcast <16 x i8> %4576 to <4 x i32>
  %4684 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4482, <4 x i32> %4683, <4 x i32> %4293, i32 %4605, i32 %4353)
  %4685 = bitcast <16 x i8> %4578 to <4 x i32>
  %4686 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4684, <4 x i32> %4685, <4 x i32> %4296, i32 %4605, i32 %4353)
  %4687 = bitcast <16 x i8> %4580 to <4 x i32>
  %4688 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4486, <4 x i32> %4687, <4 x i32> %4293, i32 %4605, i32 %4353)
  %4689 = bitcast <16 x i8> %4582 to <4 x i32>
  %4690 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4688, <4 x i32> %4689, <4 x i32> %4296, i32 %4605, i32 %4353)
  %4691 = bitcast <16 x i8> %4584 to <4 x i32>
  %4692 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4490, <4 x i32> %4691, <4 x i32> %4293, i32 %4610, i32 %4353)
  %4693 = bitcast <16 x i8> %4586 to <4 x i32>
  %4694 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4692, <4 x i32> %4693, <4 x i32> %4296, i32 %4610, i32 %4353)
  %4695 = bitcast <16 x i8> %4588 to <4 x i32>
  %4696 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4494, <4 x i32> %4695, <4 x i32> %4293, i32 %4610, i32 %4353)
  %4697 = bitcast <16 x i8> %4590 to <4 x i32>
  %4698 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4696, <4 x i32> %4697, <4 x i32> %4296, i32 %4610, i32 %4353)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4699 = add i32 %192, 10752
  %4700 = mul i32 %4699, 4
  %4701 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4700, i32 %176, i32 2)
  %4702 = add i32 %192, 11008
  %4703 = mul i32 %4702, 4
  %4704 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4703, i32 %176, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4705 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4502, <4 x i32> %4667, <4 x i32> %4313, i32 %4595, i32 %4353)
  %4706 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4705, <4 x i32> %4669, <4 x i32> %4314, i32 %4595, i32 %4353)
  %4707 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4504, <4 x i32> %4671, <4 x i32> %4313, i32 %4595, i32 %4353)
  %4708 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4707, <4 x i32> %4673, <4 x i32> %4314, i32 %4595, i32 %4353)
  %4709 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4506, <4 x i32> %4675, <4 x i32> %4313, i32 %4600, i32 %4353)
  %4710 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4709, <4 x i32> %4677, <4 x i32> %4314, i32 %4600, i32 %4353)
  %4711 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4508, <4 x i32> %4679, <4 x i32> %4313, i32 %4600, i32 %4353)
  %4712 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4711, <4 x i32> %4681, <4 x i32> %4314, i32 %4600, i32 %4353)
  %4713 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4510, <4 x i32> %4683, <4 x i32> %4313, i32 %4605, i32 %4353)
  %4714 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4713, <4 x i32> %4685, <4 x i32> %4314, i32 %4605, i32 %4353)
  %4715 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4512, <4 x i32> %4687, <4 x i32> %4313, i32 %4605, i32 %4353)
  %4716 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4715, <4 x i32> %4689, <4 x i32> %4314, i32 %4605, i32 %4353)
  %4717 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4514, <4 x i32> %4691, <4 x i32> %4313, i32 %4610, i32 %4353)
  %4718 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4717, <4 x i32> %4693, <4 x i32> %4314, i32 %4610, i32 %4353)
  %4719 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4516, <4 x i32> %4695, <4 x i32> %4313, i32 %4610, i32 %4353)
  %4720 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4719, <4 x i32> %4697, <4 x i32> %4314, i32 %4610, i32 %4353)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4721 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4700, i32 %180, i32 2)
  %4722 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4703, i32 %180, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4723 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4520, <4 x i32> %4667, <4 x i32> %4331, i32 %4595, i32 %4354)
  %4724 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4723, <4 x i32> %4669, <4 x i32> %4332, i32 %4595, i32 %4354)
  %4725 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4522, <4 x i32> %4671, <4 x i32> %4331, i32 %4595, i32 %4354)
  %4726 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4725, <4 x i32> %4673, <4 x i32> %4332, i32 %4595, i32 %4354)
  %4727 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4524, <4 x i32> %4675, <4 x i32> %4331, i32 %4600, i32 %4354)
  %4728 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4727, <4 x i32> %4677, <4 x i32> %4332, i32 %4600, i32 %4354)
  %4729 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4526, <4 x i32> %4679, <4 x i32> %4331, i32 %4600, i32 %4354)
  %4730 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4729, <4 x i32> %4681, <4 x i32> %4332, i32 %4600, i32 %4354)
  %4731 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4528, <4 x i32> %4683, <4 x i32> %4331, i32 %4605, i32 %4354)
  %4732 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4731, <4 x i32> %4685, <4 x i32> %4332, i32 %4605, i32 %4354)
  %4733 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4530, <4 x i32> %4687, <4 x i32> %4331, i32 %4605, i32 %4354)
  %4734 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4733, <4 x i32> %4689, <4 x i32> %4332, i32 %4605, i32 %4354)
  %4735 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4532, <4 x i32> %4691, <4 x i32> %4331, i32 %4610, i32 %4354)
  %4736 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4735, <4 x i32> %4693, <4 x i32> %4332, i32 %4610, i32 %4354)
  %4737 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4534, <4 x i32> %4695, <4 x i32> %4331, i32 %4610, i32 %4354)
  %4738 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4737, <4 x i32> %4697, <4 x i32> %4332, i32 %4610, i32 %4354)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4739 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4700, i32 %184, i32 2)
  %4740 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4703, i32 %184, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4741 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4538, <4 x i32> %4667, <4 x i32> %4349, i32 %4595, i32 %4354)
  %4742 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4741, <4 x i32> %4669, <4 x i32> %4350, i32 %4595, i32 %4354)
  %4743 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4540, <4 x i32> %4671, <4 x i32> %4349, i32 %4595, i32 %4354)
  %4744 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4743, <4 x i32> %4673, <4 x i32> %4350, i32 %4595, i32 %4354)
  %4745 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4542, <4 x i32> %4675, <4 x i32> %4349, i32 %4600, i32 %4354)
  %4746 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4745, <4 x i32> %4677, <4 x i32> %4350, i32 %4600, i32 %4354)
  %4747 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4544, <4 x i32> %4679, <4 x i32> %4349, i32 %4600, i32 %4354)
  %4748 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4747, <4 x i32> %4681, <4 x i32> %4350, i32 %4600, i32 %4354)
  %4749 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4546, <4 x i32> %4683, <4 x i32> %4349, i32 %4605, i32 %4354)
  %4750 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4749, <4 x i32> %4685, <4 x i32> %4350, i32 %4605, i32 %4354)
  %4751 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4548, <4 x i32> %4687, <4 x i32> %4349, i32 %4605, i32 %4354)
  %4752 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4751, <4 x i32> %4689, <4 x i32> %4350, i32 %4605, i32 %4354)
  %4753 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4550, <4 x i32> %4691, <4 x i32> %4349, i32 %4610, i32 %4354)
  %4754 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4753, <4 x i32> %4693, <4 x i32> %4350, i32 %4610, i32 %4354)
  %4755 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4552, <4 x i32> %4695, <4 x i32> %4349, i32 %4610, i32 %4354)
  %4756 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4755, <4 x i32> %4697, <4 x i32> %4350, i32 %4610, i32 %4354)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4757 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4700, i32 %188, i32 2)
  %4758 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4703, i32 %188, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4759 = add i32 %211, 1344
  %4760 = mul i32 %4759, 4
  %4761 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %4760, i32 %204, i32 0)
  %4762 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %4760, i32 %210, i32 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %4763 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1075
  %4764 = load <16 x i8>, ptr addrspace(3) %4763, align 1
  %4765 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1078
  %4766 = load <16 x i8>, ptr addrspace(3) %4765, align 1
  %4767 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1081
  %4768 = load <16 x i8>, ptr addrspace(3) %4767, align 1
  %4769 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1084
  %4770 = load <16 x i8>, ptr addrspace(3) %4769, align 1
  %4771 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1087
  %4772 = load <16 x i8>, ptr addrspace(3) %4771, align 1
  %4773 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1090
  %4774 = load <16 x i8>, ptr addrspace(3) %4773, align 1
  %4775 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1093
  %4776 = load <16 x i8>, ptr addrspace(3) %4775, align 1
  %4777 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1096
  %4778 = load <16 x i8>, ptr addrspace(3) %4777, align 1
  %4779 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1099
  %4780 = load <16 x i8>, ptr addrspace(3) %4779, align 1
  %4781 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1102
  %4782 = load <16 x i8>, ptr addrspace(3) %4781, align 1
  %4783 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1105
  %4784 = load <16 x i8>, ptr addrspace(3) %4783, align 1
  %4785 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1108
  %4786 = load <16 x i8>, ptr addrspace(3) %4785, align 1
  %4787 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1111
  %4788 = load <16 x i8>, ptr addrspace(3) %4787, align 1
  %4789 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1114
  %4790 = load <16 x i8>, ptr addrspace(3) %4789, align 1
  %4791 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1117
  %4792 = load <16 x i8>, ptr addrspace(3) %4791, align 1
  %4793 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1120
  %4794 = load <16 x i8>, ptr addrspace(3) %4793, align 1
  %4795 = add i64 %193, 1280
  %4796 = add i64 %4795, %24
  %4797 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %4796
  %4798 = load <1 x i32>, ptr addrspace(3) %4797, align 4
  %4799 = extractelement <1 x i32> %4798, i64 0
  %4800 = add i64 %193, 3072
  %4801 = add i64 %4800, %24
  %4802 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %4801
  %4803 = load <1 x i32>, ptr addrspace(3) %4802, align 4
  %4804 = extractelement <1 x i32> %4803, i64 0
  %4805 = add i64 %193, 4864
  %4806 = add i64 %4805, %24
  %4807 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %4806
  %4808 = load <1 x i32>, ptr addrspace(3) %4807, align 4
  %4809 = extractelement <1 x i32> %4808, i64 0
  %4810 = add i64 %193, 6656
  %4811 = add i64 %4810, %24
  %4812 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %4811
  %4813 = load <1 x i32>, ptr addrspace(3) %4812, align 4
  %4814 = extractelement <1 x i32> %4813, i64 0
  %4815 = add i64 %77, 704
  %4816 = add i64 %4815, %71
  %4817 = trunc i64 %4816 to i32
  %4818 = mul i32 %4817, 4
  %4819 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4818, i32 0, i32 0)
  %4820 = bitcast <4 x i32> %4819 to <16 x i8>
  %4821 = add i64 %90, 704
  %4822 = add i64 %4821, %84
  %4823 = trunc i64 %4822 to i32
  %4824 = mul i32 %4823, 4
  %4825 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4824, i32 0, i32 0)
  %4826 = bitcast <4 x i32> %4825 to <16 x i8>
  %4827 = add i64 %103, 704
  %4828 = add i64 %4827, %97
  %4829 = trunc i64 %4828 to i32
  %4830 = mul i32 %4829, 4
  %4831 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4830, i32 0, i32 0)
  %4832 = bitcast <4 x i32> %4831 to <16 x i8>
  %4833 = add i64 %116, 704
  %4834 = add i64 %4833, %110
  %4835 = trunc i64 %4834 to i32
  %4836 = mul i32 %4835, 4
  %4837 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4836, i32 0, i32 0)
  %4838 = bitcast <4 x i32> %4837 to <16 x i8>
  %4839 = add i64 %129, 704
  %4840 = add i64 %4839, %123
  %4841 = trunc i64 %4840 to i32
  %4842 = mul i32 %4841, 4
  %4843 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4842, i32 0, i32 0)
  %4844 = bitcast <4 x i32> %4843 to <16 x i8>
  %4845 = add i64 %142, 704
  %4846 = add i64 %4845, %136
  %4847 = trunc i64 %4846 to i32
  %4848 = mul i32 %4847, 4
  %4849 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4848, i32 0, i32 0)
  %4850 = bitcast <4 x i32> %4849 to <16 x i8>
  %4851 = add i64 %155, 704
  %4852 = add i64 %4851, %149
  %4853 = trunc i64 %4852 to i32
  %4854 = mul i32 %4853, 4
  %4855 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4854, i32 0, i32 0)
  %4856 = bitcast <4 x i32> %4855 to <16 x i8>
  %4857 = add i64 %168, 704
  %4858 = add i64 %4857, %162
  %4859 = trunc i64 %4858 to i32
  %4860 = mul i32 %4859, 4
  %4861 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %4860, i32 0, i32 0)
  %4862 = bitcast <4 x i32> %4861 to <16 x i8>
  %4863 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %487
  store <16 x i8> %4820, ptr addrspace(3) %4863, align 1
  %4864 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %489
  store <16 x i8> %4826, ptr addrspace(3) %4864, align 1
  %4865 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %491
  store <16 x i8> %4832, ptr addrspace(3) %4865, align 1
  %4866 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %493
  store <16 x i8> %4838, ptr addrspace(3) %4866, align 1
  %4867 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %495
  store <16 x i8> %4844, ptr addrspace(3) %4867, align 1
  %4868 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %497
  store <16 x i8> %4850, ptr addrspace(3) %4868, align 1
  %4869 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %499
  store <16 x i8> %4856, ptr addrspace(3) %4869, align 1
  %4870 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %501
  store <16 x i8> %4862, ptr addrspace(3) %4870, align 1
  %4871 = bitcast <16 x i8> %4764 to <4 x i32>
  %4872 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4670, <4 x i32> %4871, <4 x i32> %4497, i32 %4799, i32 %4557)
  %4873 = bitcast <16 x i8> %4766 to <4 x i32>
  %4874 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4872, <4 x i32> %4873, <4 x i32> %4500, i32 %4799, i32 %4557)
  %4875 = bitcast <16 x i8> %4768 to <4 x i32>
  %4876 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4674, <4 x i32> %4875, <4 x i32> %4497, i32 %4799, i32 %4557)
  %4877 = bitcast <16 x i8> %4770 to <4 x i32>
  %4878 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4876, <4 x i32> %4877, <4 x i32> %4500, i32 %4799, i32 %4557)
  %4879 = bitcast <16 x i8> %4772 to <4 x i32>
  %4880 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4678, <4 x i32> %4879, <4 x i32> %4497, i32 %4804, i32 %4557)
  %4881 = bitcast <16 x i8> %4774 to <4 x i32>
  %4882 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4880, <4 x i32> %4881, <4 x i32> %4500, i32 %4804, i32 %4557)
  %4883 = bitcast <16 x i8> %4776 to <4 x i32>
  %4884 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4682, <4 x i32> %4883, <4 x i32> %4497, i32 %4804, i32 %4557)
  %4885 = bitcast <16 x i8> %4778 to <4 x i32>
  %4886 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4884, <4 x i32> %4885, <4 x i32> %4500, i32 %4804, i32 %4557)
  %4887 = bitcast <16 x i8> %4780 to <4 x i32>
  %4888 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4686, <4 x i32> %4887, <4 x i32> %4497, i32 %4809, i32 %4557)
  %4889 = bitcast <16 x i8> %4782 to <4 x i32>
  %4890 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4888, <4 x i32> %4889, <4 x i32> %4500, i32 %4809, i32 %4557)
  %4891 = bitcast <16 x i8> %4784 to <4 x i32>
  %4892 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4690, <4 x i32> %4891, <4 x i32> %4497, i32 %4809, i32 %4557)
  %4893 = bitcast <16 x i8> %4786 to <4 x i32>
  %4894 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4892, <4 x i32> %4893, <4 x i32> %4500, i32 %4809, i32 %4557)
  %4895 = bitcast <16 x i8> %4788 to <4 x i32>
  %4896 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4694, <4 x i32> %4895, <4 x i32> %4497, i32 %4814, i32 %4557)
  %4897 = bitcast <16 x i8> %4790 to <4 x i32>
  %4898 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4896, <4 x i32> %4897, <4 x i32> %4500, i32 %4814, i32 %4557)
  %4899 = bitcast <16 x i8> %4792 to <4 x i32>
  %4900 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4698, <4 x i32> %4899, <4 x i32> %4497, i32 %4814, i32 %4557)
  %4901 = bitcast <16 x i8> %4794 to <4 x i32>
  %4902 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4900, <4 x i32> %4901, <4 x i32> %4500, i32 %4814, i32 %4557)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4903 = add i32 %192, 11264
  %4904 = mul i32 %4903, 4
  %4905 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4904, i32 %176, i32 2)
  %4906 = add i32 %192, 11520
  %4907 = mul i32 %4906, 4
  %4908 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4907, i32 %176, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4909 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4706, <4 x i32> %4871, <4 x i32> %4517, i32 %4799, i32 %4557)
  %4910 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4909, <4 x i32> %4873, <4 x i32> %4518, i32 %4799, i32 %4557)
  %4911 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4708, <4 x i32> %4875, <4 x i32> %4517, i32 %4799, i32 %4557)
  %4912 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4911, <4 x i32> %4877, <4 x i32> %4518, i32 %4799, i32 %4557)
  %4913 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4710, <4 x i32> %4879, <4 x i32> %4517, i32 %4804, i32 %4557)
  %4914 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4913, <4 x i32> %4881, <4 x i32> %4518, i32 %4804, i32 %4557)
  %4915 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4712, <4 x i32> %4883, <4 x i32> %4517, i32 %4804, i32 %4557)
  %4916 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4915, <4 x i32> %4885, <4 x i32> %4518, i32 %4804, i32 %4557)
  %4917 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4714, <4 x i32> %4887, <4 x i32> %4517, i32 %4809, i32 %4557)
  %4918 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4917, <4 x i32> %4889, <4 x i32> %4518, i32 %4809, i32 %4557)
  %4919 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4716, <4 x i32> %4891, <4 x i32> %4517, i32 %4809, i32 %4557)
  %4920 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4919, <4 x i32> %4893, <4 x i32> %4518, i32 %4809, i32 %4557)
  %4921 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4718, <4 x i32> %4895, <4 x i32> %4517, i32 %4814, i32 %4557)
  %4922 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4921, <4 x i32> %4897, <4 x i32> %4518, i32 %4814, i32 %4557)
  %4923 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4720, <4 x i32> %4899, <4 x i32> %4517, i32 %4814, i32 %4557)
  %4924 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4923, <4 x i32> %4901, <4 x i32> %4518, i32 %4814, i32 %4557)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4925 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4904, i32 %180, i32 2)
  %4926 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4907, i32 %180, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4927 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4724, <4 x i32> %4871, <4 x i32> %4535, i32 %4799, i32 %4558)
  %4928 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4927, <4 x i32> %4873, <4 x i32> %4536, i32 %4799, i32 %4558)
  %4929 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4726, <4 x i32> %4875, <4 x i32> %4535, i32 %4799, i32 %4558)
  %4930 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4929, <4 x i32> %4877, <4 x i32> %4536, i32 %4799, i32 %4558)
  %4931 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4728, <4 x i32> %4879, <4 x i32> %4535, i32 %4804, i32 %4558)
  %4932 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4931, <4 x i32> %4881, <4 x i32> %4536, i32 %4804, i32 %4558)
  %4933 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4730, <4 x i32> %4883, <4 x i32> %4535, i32 %4804, i32 %4558)
  %4934 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4933, <4 x i32> %4885, <4 x i32> %4536, i32 %4804, i32 %4558)
  %4935 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4732, <4 x i32> %4887, <4 x i32> %4535, i32 %4809, i32 %4558)
  %4936 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4935, <4 x i32> %4889, <4 x i32> %4536, i32 %4809, i32 %4558)
  %4937 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4734, <4 x i32> %4891, <4 x i32> %4535, i32 %4809, i32 %4558)
  %4938 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4937, <4 x i32> %4893, <4 x i32> %4536, i32 %4809, i32 %4558)
  %4939 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4736, <4 x i32> %4895, <4 x i32> %4535, i32 %4814, i32 %4558)
  %4940 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4939, <4 x i32> %4897, <4 x i32> %4536, i32 %4814, i32 %4558)
  %4941 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4738, <4 x i32> %4899, <4 x i32> %4535, i32 %4814, i32 %4558)
  %4942 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4941, <4 x i32> %4901, <4 x i32> %4536, i32 %4814, i32 %4558)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4943 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4904, i32 %184, i32 2)
  %4944 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4907, i32 %184, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4945 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4742, <4 x i32> %4871, <4 x i32> %4553, i32 %4799, i32 %4558)
  %4946 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4945, <4 x i32> %4873, <4 x i32> %4554, i32 %4799, i32 %4558)
  %4947 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4744, <4 x i32> %4875, <4 x i32> %4553, i32 %4799, i32 %4558)
  %4948 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4947, <4 x i32> %4877, <4 x i32> %4554, i32 %4799, i32 %4558)
  %4949 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4746, <4 x i32> %4879, <4 x i32> %4553, i32 %4804, i32 %4558)
  %4950 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4949, <4 x i32> %4881, <4 x i32> %4554, i32 %4804, i32 %4558)
  %4951 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4748, <4 x i32> %4883, <4 x i32> %4553, i32 %4804, i32 %4558)
  %4952 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4951, <4 x i32> %4885, <4 x i32> %4554, i32 %4804, i32 %4558)
  %4953 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4750, <4 x i32> %4887, <4 x i32> %4553, i32 %4809, i32 %4558)
  %4954 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4953, <4 x i32> %4889, <4 x i32> %4554, i32 %4809, i32 %4558)
  %4955 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4752, <4 x i32> %4891, <4 x i32> %4553, i32 %4809, i32 %4558)
  %4956 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4955, <4 x i32> %4893, <4 x i32> %4554, i32 %4809, i32 %4558)
  %4957 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4754, <4 x i32> %4895, <4 x i32> %4553, i32 %4814, i32 %4558)
  %4958 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4957, <4 x i32> %4897, <4 x i32> %4554, i32 %4814, i32 %4558)
  %4959 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4756, <4 x i32> %4899, <4 x i32> %4553, i32 %4814, i32 %4558)
  %4960 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4959, <4 x i32> %4901, <4 x i32> %4554, i32 %4814, i32 %4558)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4961 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4904, i32 %188, i32 2)
  %4962 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %4907, i32 %188, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %4963 = add i32 %211, 1408
  %4964 = mul i32 %4963, 4
  %4965 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %4964, i32 %204, i32 0)
  %4966 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %4964, i32 %210, i32 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %4967 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %522
  %4968 = load <16 x i8>, ptr addrspace(3) %4967, align 1
  %4969 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %527
  %4970 = load <16 x i8>, ptr addrspace(3) %4969, align 1
  %4971 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %535
  %4972 = load <16 x i8>, ptr addrspace(3) %4971, align 1
  %4973 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %539
  %4974 = load <16 x i8>, ptr addrspace(3) %4973, align 1
  %4975 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %547
  %4976 = load <16 x i8>, ptr addrspace(3) %4975, align 1
  %4977 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %551
  %4978 = load <16 x i8>, ptr addrspace(3) %4977, align 1
  %4979 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %559
  %4980 = load <16 x i8>, ptr addrspace(3) %4979, align 1
  %4981 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %563
  %4982 = load <16 x i8>, ptr addrspace(3) %4981, align 1
  %4983 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %571
  %4984 = load <16 x i8>, ptr addrspace(3) %4983, align 1
  %4985 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %575
  %4986 = load <16 x i8>, ptr addrspace(3) %4985, align 1
  %4987 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %583
  %4988 = load <16 x i8>, ptr addrspace(3) %4987, align 1
  %4989 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %587
  %4990 = load <16 x i8>, ptr addrspace(3) %4989, align 1
  %4991 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %595
  %4992 = load <16 x i8>, ptr addrspace(3) %4991, align 1
  %4993 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %599
  %4994 = load <16 x i8>, ptr addrspace(3) %4993, align 1
  %4995 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %607
  %4996 = load <16 x i8>, ptr addrspace(3) %4995, align 1
  %4997 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %611
  %4998 = load <16 x i8>, ptr addrspace(3) %4997, align 1
  %4999 = add i64 %193, 1344
  %5000 = add i64 %4999, %24
  %5001 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %5000
  %5002 = load <1 x i32>, ptr addrspace(3) %5001, align 4
  %5003 = extractelement <1 x i32> %5002, i64 0
  %5004 = add i64 %193, 3136
  %5005 = add i64 %5004, %24
  %5006 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %5005
  %5007 = load <1 x i32>, ptr addrspace(3) %5006, align 4
  %5008 = extractelement <1 x i32> %5007, i64 0
  %5009 = add i64 %193, 4928
  %5010 = add i64 %5009, %24
  %5011 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %5010
  %5012 = load <1 x i32>, ptr addrspace(3) %5011, align 4
  %5013 = extractelement <1 x i32> %5012, i64 0
  %5014 = add i64 %193, 6720
  %5015 = add i64 %5014, %24
  %5016 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %5015
  %5017 = load <1 x i32>, ptr addrspace(3) %5016, align 4
  %5018 = extractelement <1 x i32> %5017, i64 0
  %5019 = add i64 %77, 736
  %5020 = add i64 %5019, %71
  %5021 = trunc i64 %5020 to i32
  %5022 = mul i32 %5021, 4
  %5023 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5022, i32 0, i32 0)
  %5024 = bitcast <4 x i32> %5023 to <16 x i8>
  %5025 = add i64 %90, 736
  %5026 = add i64 %5025, %84
  %5027 = trunc i64 %5026 to i32
  %5028 = mul i32 %5027, 4
  %5029 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5028, i32 0, i32 0)
  %5030 = bitcast <4 x i32> %5029 to <16 x i8>
  %5031 = add i64 %103, 736
  %5032 = add i64 %5031, %97
  %5033 = trunc i64 %5032 to i32
  %5034 = mul i32 %5033, 4
  %5035 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5034, i32 0, i32 0)
  %5036 = bitcast <4 x i32> %5035 to <16 x i8>
  %5037 = add i64 %116, 736
  %5038 = add i64 %5037, %110
  %5039 = trunc i64 %5038 to i32
  %5040 = mul i32 %5039, 4
  %5041 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5040, i32 0, i32 0)
  %5042 = bitcast <4 x i32> %5041 to <16 x i8>
  %5043 = add i64 %129, 736
  %5044 = add i64 %5043, %123
  %5045 = trunc i64 %5044 to i32
  %5046 = mul i32 %5045, 4
  %5047 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5046, i32 0, i32 0)
  %5048 = bitcast <4 x i32> %5047 to <16 x i8>
  %5049 = add i64 %142, 736
  %5050 = add i64 %5049, %136
  %5051 = trunc i64 %5050 to i32
  %5052 = mul i32 %5051, 4
  %5053 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5052, i32 0, i32 0)
  %5054 = bitcast <4 x i32> %5053 to <16 x i8>
  %5055 = add i64 %155, 736
  %5056 = add i64 %5055, %149
  %5057 = trunc i64 %5056 to i32
  %5058 = mul i32 %5057, 4
  %5059 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5058, i32 0, i32 0)
  %5060 = bitcast <4 x i32> %5059 to <16 x i8>
  %5061 = add i64 %168, 736
  %5062 = add i64 %5061, %162
  %5063 = trunc i64 %5062 to i32
  %5064 = mul i32 %5063, 4
  %5065 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5064, i32 0, i32 0)
  %5066 = bitcast <4 x i32> %5065 to <16 x i8>
  %5067 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %680
  store <16 x i8> %5024, ptr addrspace(3) %5067, align 1
  %5068 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %682
  store <16 x i8> %5030, ptr addrspace(3) %5068, align 1
  %5069 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %684
  store <16 x i8> %5036, ptr addrspace(3) %5069, align 1
  %5070 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %686
  store <16 x i8> %5042, ptr addrspace(3) %5070, align 1
  %5071 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %688
  store <16 x i8> %5048, ptr addrspace(3) %5071, align 1
  %5072 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %690
  store <16 x i8> %5054, ptr addrspace(3) %5072, align 1
  %5073 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %692
  store <16 x i8> %5060, ptr addrspace(3) %5073, align 1
  %5074 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %694
  store <16 x i8> %5066, ptr addrspace(3) %5074, align 1
  %5075 = bitcast <16 x i8> %4968 to <4 x i32>
  %5076 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4874, <4 x i32> %5075, <4 x i32> %4701, i32 %5003, i32 %4761)
  %5077 = bitcast <16 x i8> %4970 to <4 x i32>
  %5078 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5076, <4 x i32> %5077, <4 x i32> %4704, i32 %5003, i32 %4761)
  %5079 = bitcast <16 x i8> %4972 to <4 x i32>
  %5080 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4878, <4 x i32> %5079, <4 x i32> %4701, i32 %5003, i32 %4761)
  %5081 = bitcast <16 x i8> %4974 to <4 x i32>
  %5082 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5080, <4 x i32> %5081, <4 x i32> %4704, i32 %5003, i32 %4761)
  %5083 = bitcast <16 x i8> %4976 to <4 x i32>
  %5084 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4882, <4 x i32> %5083, <4 x i32> %4701, i32 %5008, i32 %4761)
  %5085 = bitcast <16 x i8> %4978 to <4 x i32>
  %5086 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5084, <4 x i32> %5085, <4 x i32> %4704, i32 %5008, i32 %4761)
  %5087 = bitcast <16 x i8> %4980 to <4 x i32>
  %5088 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4886, <4 x i32> %5087, <4 x i32> %4701, i32 %5008, i32 %4761)
  %5089 = bitcast <16 x i8> %4982 to <4 x i32>
  %5090 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5088, <4 x i32> %5089, <4 x i32> %4704, i32 %5008, i32 %4761)
  %5091 = bitcast <16 x i8> %4984 to <4 x i32>
  %5092 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4890, <4 x i32> %5091, <4 x i32> %4701, i32 %5013, i32 %4761)
  %5093 = bitcast <16 x i8> %4986 to <4 x i32>
  %5094 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5092, <4 x i32> %5093, <4 x i32> %4704, i32 %5013, i32 %4761)
  %5095 = bitcast <16 x i8> %4988 to <4 x i32>
  %5096 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4894, <4 x i32> %5095, <4 x i32> %4701, i32 %5013, i32 %4761)
  %5097 = bitcast <16 x i8> %4990 to <4 x i32>
  %5098 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5096, <4 x i32> %5097, <4 x i32> %4704, i32 %5013, i32 %4761)
  %5099 = bitcast <16 x i8> %4992 to <4 x i32>
  %5100 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4898, <4 x i32> %5099, <4 x i32> %4701, i32 %5018, i32 %4761)
  %5101 = bitcast <16 x i8> %4994 to <4 x i32>
  %5102 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5100, <4 x i32> %5101, <4 x i32> %4704, i32 %5018, i32 %4761)
  %5103 = bitcast <16 x i8> %4996 to <4 x i32>
  %5104 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4902, <4 x i32> %5103, <4 x i32> %4701, i32 %5018, i32 %4761)
  %5105 = bitcast <16 x i8> %4998 to <4 x i32>
  %5106 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5104, <4 x i32> %5105, <4 x i32> %4704, i32 %5018, i32 %4761)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5107 = add i32 %192, 11776
  %5108 = mul i32 %5107, 4
  %5109 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5108, i32 %176, i32 2)
  %5110 = add i32 %192, 12032
  %5111 = mul i32 %5110, 4
  %5112 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5111, i32 %176, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5113 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4910, <4 x i32> %5075, <4 x i32> %4721, i32 %5003, i32 %4761)
  %5114 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5113, <4 x i32> %5077, <4 x i32> %4722, i32 %5003, i32 %4761)
  %5115 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4912, <4 x i32> %5079, <4 x i32> %4721, i32 %5003, i32 %4761)
  %5116 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5115, <4 x i32> %5081, <4 x i32> %4722, i32 %5003, i32 %4761)
  %5117 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4914, <4 x i32> %5083, <4 x i32> %4721, i32 %5008, i32 %4761)
  %5118 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5117, <4 x i32> %5085, <4 x i32> %4722, i32 %5008, i32 %4761)
  %5119 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4916, <4 x i32> %5087, <4 x i32> %4721, i32 %5008, i32 %4761)
  %5120 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5119, <4 x i32> %5089, <4 x i32> %4722, i32 %5008, i32 %4761)
  %5121 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4918, <4 x i32> %5091, <4 x i32> %4721, i32 %5013, i32 %4761)
  %5122 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5121, <4 x i32> %5093, <4 x i32> %4722, i32 %5013, i32 %4761)
  %5123 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4920, <4 x i32> %5095, <4 x i32> %4721, i32 %5013, i32 %4761)
  %5124 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5123, <4 x i32> %5097, <4 x i32> %4722, i32 %5013, i32 %4761)
  %5125 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4922, <4 x i32> %5099, <4 x i32> %4721, i32 %5018, i32 %4761)
  %5126 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5125, <4 x i32> %5101, <4 x i32> %4722, i32 %5018, i32 %4761)
  %5127 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4924, <4 x i32> %5103, <4 x i32> %4721, i32 %5018, i32 %4761)
  %5128 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5127, <4 x i32> %5105, <4 x i32> %4722, i32 %5018, i32 %4761)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5129 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5108, i32 %180, i32 2)
  %5130 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5111, i32 %180, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5131 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4928, <4 x i32> %5075, <4 x i32> %4739, i32 %5003, i32 %4762)
  %5132 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5131, <4 x i32> %5077, <4 x i32> %4740, i32 %5003, i32 %4762)
  %5133 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4930, <4 x i32> %5079, <4 x i32> %4739, i32 %5003, i32 %4762)
  %5134 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5133, <4 x i32> %5081, <4 x i32> %4740, i32 %5003, i32 %4762)
  %5135 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4932, <4 x i32> %5083, <4 x i32> %4739, i32 %5008, i32 %4762)
  %5136 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5135, <4 x i32> %5085, <4 x i32> %4740, i32 %5008, i32 %4762)
  %5137 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4934, <4 x i32> %5087, <4 x i32> %4739, i32 %5008, i32 %4762)
  %5138 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5137, <4 x i32> %5089, <4 x i32> %4740, i32 %5008, i32 %4762)
  %5139 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4936, <4 x i32> %5091, <4 x i32> %4739, i32 %5013, i32 %4762)
  %5140 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5139, <4 x i32> %5093, <4 x i32> %4740, i32 %5013, i32 %4762)
  %5141 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4938, <4 x i32> %5095, <4 x i32> %4739, i32 %5013, i32 %4762)
  %5142 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5141, <4 x i32> %5097, <4 x i32> %4740, i32 %5013, i32 %4762)
  %5143 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4940, <4 x i32> %5099, <4 x i32> %4739, i32 %5018, i32 %4762)
  %5144 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5143, <4 x i32> %5101, <4 x i32> %4740, i32 %5018, i32 %4762)
  %5145 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4942, <4 x i32> %5103, <4 x i32> %4739, i32 %5018, i32 %4762)
  %5146 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5145, <4 x i32> %5105, <4 x i32> %4740, i32 %5018, i32 %4762)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5147 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5108, i32 %184, i32 2)
  %5148 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5111, i32 %184, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5149 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4946, <4 x i32> %5075, <4 x i32> %4757, i32 %5003, i32 %4762)
  %5150 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5149, <4 x i32> %5077, <4 x i32> %4758, i32 %5003, i32 %4762)
  %5151 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4948, <4 x i32> %5079, <4 x i32> %4757, i32 %5003, i32 %4762)
  %5152 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5151, <4 x i32> %5081, <4 x i32> %4758, i32 %5003, i32 %4762)
  %5153 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4950, <4 x i32> %5083, <4 x i32> %4757, i32 %5008, i32 %4762)
  %5154 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5153, <4 x i32> %5085, <4 x i32> %4758, i32 %5008, i32 %4762)
  %5155 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4952, <4 x i32> %5087, <4 x i32> %4757, i32 %5008, i32 %4762)
  %5156 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5155, <4 x i32> %5089, <4 x i32> %4758, i32 %5008, i32 %4762)
  %5157 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4954, <4 x i32> %5091, <4 x i32> %4757, i32 %5013, i32 %4762)
  %5158 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5157, <4 x i32> %5093, <4 x i32> %4758, i32 %5013, i32 %4762)
  %5159 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4956, <4 x i32> %5095, <4 x i32> %4757, i32 %5013, i32 %4762)
  %5160 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5159, <4 x i32> %5097, <4 x i32> %4758, i32 %5013, i32 %4762)
  %5161 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4958, <4 x i32> %5099, <4 x i32> %4757, i32 %5018, i32 %4762)
  %5162 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5161, <4 x i32> %5101, <4 x i32> %4758, i32 %5018, i32 %4762)
  %5163 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %4960, <4 x i32> %5103, <4 x i32> %4757, i32 %5018, i32 %4762)
  %5164 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5163, <4 x i32> %5105, <4 x i32> %4758, i32 %5018, i32 %4762)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5165 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5108, i32 %188, i32 2)
  %5166 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5111, i32 %188, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5167 = add i32 %211, 1472
  %5168 = mul i32 %5167, 4
  %5169 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %5168, i32 %204, i32 0)
  %5170 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %5168, i32 %210, i32 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %5171 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %856
  %5172 = load <16 x i8>, ptr addrspace(3) %5171, align 1
  %5173 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %859
  %5174 = load <16 x i8>, ptr addrspace(3) %5173, align 1
  %5175 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %862
  %5176 = load <16 x i8>, ptr addrspace(3) %5175, align 1
  %5177 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %865
  %5178 = load <16 x i8>, ptr addrspace(3) %5177, align 1
  %5179 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %868
  %5180 = load <16 x i8>, ptr addrspace(3) %5179, align 1
  %5181 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %871
  %5182 = load <16 x i8>, ptr addrspace(3) %5181, align 1
  %5183 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %874
  %5184 = load <16 x i8>, ptr addrspace(3) %5183, align 1
  %5185 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %877
  %5186 = load <16 x i8>, ptr addrspace(3) %5185, align 1
  %5187 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %880
  %5188 = load <16 x i8>, ptr addrspace(3) %5187, align 1
  %5189 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %883
  %5190 = load <16 x i8>, ptr addrspace(3) %5189, align 1
  %5191 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %886
  %5192 = load <16 x i8>, ptr addrspace(3) %5191, align 1
  %5193 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %889
  %5194 = load <16 x i8>, ptr addrspace(3) %5193, align 1
  %5195 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %892
  %5196 = load <16 x i8>, ptr addrspace(3) %5195, align 1
  %5197 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %895
  %5198 = load <16 x i8>, ptr addrspace(3) %5197, align 1
  %5199 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %898
  %5200 = load <16 x i8>, ptr addrspace(3) %5199, align 1
  %5201 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %901
  %5202 = load <16 x i8>, ptr addrspace(3) %5201, align 1
  %5203 = add i64 %193, 1408
  %5204 = add i64 %5203, %24
  %5205 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %5204
  %5206 = load <1 x i32>, ptr addrspace(3) %5205, align 4
  %5207 = extractelement <1 x i32> %5206, i64 0
  %5208 = add i64 %193, 3200
  %5209 = add i64 %5208, %24
  %5210 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %5209
  %5211 = load <1 x i32>, ptr addrspace(3) %5210, align 4
  %5212 = extractelement <1 x i32> %5211, i64 0
  %5213 = add i64 %193, 4992
  %5214 = add i64 %5213, %24
  %5215 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %5214
  %5216 = load <1 x i32>, ptr addrspace(3) %5215, align 4
  %5217 = extractelement <1 x i32> %5216, i64 0
  %5218 = add i64 %193, 6784
  %5219 = add i64 %5218, %24
  %5220 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %5219
  %5221 = load <1 x i32>, ptr addrspace(3) %5220, align 4
  %5222 = extractelement <1 x i32> %5221, i64 0
  %5223 = add i64 %77, 768
  %5224 = add i64 %5223, %71
  %5225 = trunc i64 %5224 to i32
  %5226 = mul i32 %5225, 4
  %5227 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5226, i32 0, i32 0)
  %5228 = bitcast <4 x i32> %5227 to <16 x i8>
  %5229 = add i64 %90, 768
  %5230 = add i64 %5229, %84
  %5231 = trunc i64 %5230 to i32
  %5232 = mul i32 %5231, 4
  %5233 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5232, i32 0, i32 0)
  %5234 = bitcast <4 x i32> %5233 to <16 x i8>
  %5235 = add i64 %103, 768
  %5236 = add i64 %5235, %97
  %5237 = trunc i64 %5236 to i32
  %5238 = mul i32 %5237, 4
  %5239 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5238, i32 0, i32 0)
  %5240 = bitcast <4 x i32> %5239 to <16 x i8>
  %5241 = add i64 %116, 768
  %5242 = add i64 %5241, %110
  %5243 = trunc i64 %5242 to i32
  %5244 = mul i32 %5243, 4
  %5245 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5244, i32 0, i32 0)
  %5246 = bitcast <4 x i32> %5245 to <16 x i8>
  %5247 = add i64 %129, 768
  %5248 = add i64 %5247, %123
  %5249 = trunc i64 %5248 to i32
  %5250 = mul i32 %5249, 4
  %5251 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5250, i32 0, i32 0)
  %5252 = bitcast <4 x i32> %5251 to <16 x i8>
  %5253 = add i64 %142, 768
  %5254 = add i64 %5253, %136
  %5255 = trunc i64 %5254 to i32
  %5256 = mul i32 %5255, 4
  %5257 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5256, i32 0, i32 0)
  %5258 = bitcast <4 x i32> %5257 to <16 x i8>
  %5259 = add i64 %155, 768
  %5260 = add i64 %5259, %149
  %5261 = trunc i64 %5260 to i32
  %5262 = mul i32 %5261, 4
  %5263 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5262, i32 0, i32 0)
  %5264 = bitcast <4 x i32> %5263 to <16 x i8>
  %5265 = add i64 %168, 768
  %5266 = add i64 %5265, %162
  %5267 = trunc i64 %5266 to i32
  %5268 = mul i32 %5267, 4
  %5269 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5268, i32 0, i32 0)
  %5270 = bitcast <4 x i32> %5269 to <16 x i8>
  %5271 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %360
  store <16 x i8> %5228, ptr addrspace(3) %5271, align 1
  %5272 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %369
  store <16 x i8> %5234, ptr addrspace(3) %5272, align 1
  %5273 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %378
  store <16 x i8> %5240, ptr addrspace(3) %5273, align 1
  %5274 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %387
  store <16 x i8> %5246, ptr addrspace(3) %5274, align 1
  %5275 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %396
  store <16 x i8> %5252, ptr addrspace(3) %5275, align 1
  %5276 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %405
  store <16 x i8> %5258, ptr addrspace(3) %5276, align 1
  %5277 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %414
  store <16 x i8> %5264, ptr addrspace(3) %5277, align 1
  %5278 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %423
  store <16 x i8> %5270, ptr addrspace(3) %5278, align 1
  %5279 = bitcast <16 x i8> %5172 to <4 x i32>
  %5280 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5078, <4 x i32> %5279, <4 x i32> %4905, i32 %5207, i32 %4965)
  %5281 = bitcast <16 x i8> %5174 to <4 x i32>
  %5282 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5280, <4 x i32> %5281, <4 x i32> %4908, i32 %5207, i32 %4965)
  %5283 = bitcast <16 x i8> %5176 to <4 x i32>
  %5284 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5082, <4 x i32> %5283, <4 x i32> %4905, i32 %5207, i32 %4965)
  %5285 = bitcast <16 x i8> %5178 to <4 x i32>
  %5286 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5284, <4 x i32> %5285, <4 x i32> %4908, i32 %5207, i32 %4965)
  %5287 = bitcast <16 x i8> %5180 to <4 x i32>
  %5288 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5086, <4 x i32> %5287, <4 x i32> %4905, i32 %5212, i32 %4965)
  %5289 = bitcast <16 x i8> %5182 to <4 x i32>
  %5290 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5288, <4 x i32> %5289, <4 x i32> %4908, i32 %5212, i32 %4965)
  %5291 = bitcast <16 x i8> %5184 to <4 x i32>
  %5292 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5090, <4 x i32> %5291, <4 x i32> %4905, i32 %5212, i32 %4965)
  %5293 = bitcast <16 x i8> %5186 to <4 x i32>
  %5294 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5292, <4 x i32> %5293, <4 x i32> %4908, i32 %5212, i32 %4965)
  %5295 = bitcast <16 x i8> %5188 to <4 x i32>
  %5296 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5094, <4 x i32> %5295, <4 x i32> %4905, i32 %5217, i32 %4965)
  %5297 = bitcast <16 x i8> %5190 to <4 x i32>
  %5298 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5296, <4 x i32> %5297, <4 x i32> %4908, i32 %5217, i32 %4965)
  %5299 = bitcast <16 x i8> %5192 to <4 x i32>
  %5300 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5098, <4 x i32> %5299, <4 x i32> %4905, i32 %5217, i32 %4965)
  %5301 = bitcast <16 x i8> %5194 to <4 x i32>
  %5302 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5300, <4 x i32> %5301, <4 x i32> %4908, i32 %5217, i32 %4965)
  %5303 = bitcast <16 x i8> %5196 to <4 x i32>
  %5304 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5102, <4 x i32> %5303, <4 x i32> %4905, i32 %5222, i32 %4965)
  %5305 = bitcast <16 x i8> %5198 to <4 x i32>
  %5306 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5304, <4 x i32> %5305, <4 x i32> %4908, i32 %5222, i32 %4965)
  %5307 = bitcast <16 x i8> %5200 to <4 x i32>
  %5308 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5106, <4 x i32> %5307, <4 x i32> %4905, i32 %5222, i32 %4965)
  %5309 = bitcast <16 x i8> %5202 to <4 x i32>
  %5310 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5308, <4 x i32> %5309, <4 x i32> %4908, i32 %5222, i32 %4965)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5311 = add i32 %192, 12288
  %5312 = mul i32 %5311, 4
  %5313 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5312, i32 %176, i32 2)
  %5314 = add i32 %192, 12544
  %5315 = mul i32 %5314, 4
  %5316 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5315, i32 %176, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5317 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5114, <4 x i32> %5279, <4 x i32> %4925, i32 %5207, i32 %4965)
  %5318 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5317, <4 x i32> %5281, <4 x i32> %4926, i32 %5207, i32 %4965)
  %5319 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5116, <4 x i32> %5283, <4 x i32> %4925, i32 %5207, i32 %4965)
  %5320 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5319, <4 x i32> %5285, <4 x i32> %4926, i32 %5207, i32 %4965)
  %5321 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5118, <4 x i32> %5287, <4 x i32> %4925, i32 %5212, i32 %4965)
  %5322 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5321, <4 x i32> %5289, <4 x i32> %4926, i32 %5212, i32 %4965)
  %5323 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5120, <4 x i32> %5291, <4 x i32> %4925, i32 %5212, i32 %4965)
  %5324 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5323, <4 x i32> %5293, <4 x i32> %4926, i32 %5212, i32 %4965)
  %5325 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5122, <4 x i32> %5295, <4 x i32> %4925, i32 %5217, i32 %4965)
  %5326 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5325, <4 x i32> %5297, <4 x i32> %4926, i32 %5217, i32 %4965)
  %5327 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5124, <4 x i32> %5299, <4 x i32> %4925, i32 %5217, i32 %4965)
  %5328 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5327, <4 x i32> %5301, <4 x i32> %4926, i32 %5217, i32 %4965)
  %5329 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5126, <4 x i32> %5303, <4 x i32> %4925, i32 %5222, i32 %4965)
  %5330 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5329, <4 x i32> %5305, <4 x i32> %4926, i32 %5222, i32 %4965)
  %5331 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5128, <4 x i32> %5307, <4 x i32> %4925, i32 %5222, i32 %4965)
  %5332 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5331, <4 x i32> %5309, <4 x i32> %4926, i32 %5222, i32 %4965)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5333 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5312, i32 %180, i32 2)
  %5334 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5315, i32 %180, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5335 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5132, <4 x i32> %5279, <4 x i32> %4943, i32 %5207, i32 %4966)
  %5336 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5335, <4 x i32> %5281, <4 x i32> %4944, i32 %5207, i32 %4966)
  %5337 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5134, <4 x i32> %5283, <4 x i32> %4943, i32 %5207, i32 %4966)
  %5338 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5337, <4 x i32> %5285, <4 x i32> %4944, i32 %5207, i32 %4966)
  %5339 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5136, <4 x i32> %5287, <4 x i32> %4943, i32 %5212, i32 %4966)
  %5340 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5339, <4 x i32> %5289, <4 x i32> %4944, i32 %5212, i32 %4966)
  %5341 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5138, <4 x i32> %5291, <4 x i32> %4943, i32 %5212, i32 %4966)
  %5342 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5341, <4 x i32> %5293, <4 x i32> %4944, i32 %5212, i32 %4966)
  %5343 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5140, <4 x i32> %5295, <4 x i32> %4943, i32 %5217, i32 %4966)
  %5344 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5343, <4 x i32> %5297, <4 x i32> %4944, i32 %5217, i32 %4966)
  %5345 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5142, <4 x i32> %5299, <4 x i32> %4943, i32 %5217, i32 %4966)
  %5346 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5345, <4 x i32> %5301, <4 x i32> %4944, i32 %5217, i32 %4966)
  %5347 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5144, <4 x i32> %5303, <4 x i32> %4943, i32 %5222, i32 %4966)
  %5348 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5347, <4 x i32> %5305, <4 x i32> %4944, i32 %5222, i32 %4966)
  %5349 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5146, <4 x i32> %5307, <4 x i32> %4943, i32 %5222, i32 %4966)
  %5350 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5349, <4 x i32> %5309, <4 x i32> %4944, i32 %5222, i32 %4966)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5351 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5312, i32 %184, i32 2)
  %5352 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5315, i32 %184, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5353 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5150, <4 x i32> %5279, <4 x i32> %4961, i32 %5207, i32 %4966)
  %5354 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5353, <4 x i32> %5281, <4 x i32> %4962, i32 %5207, i32 %4966)
  %5355 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5152, <4 x i32> %5283, <4 x i32> %4961, i32 %5207, i32 %4966)
  %5356 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5355, <4 x i32> %5285, <4 x i32> %4962, i32 %5207, i32 %4966)
  %5357 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5154, <4 x i32> %5287, <4 x i32> %4961, i32 %5212, i32 %4966)
  %5358 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5357, <4 x i32> %5289, <4 x i32> %4962, i32 %5212, i32 %4966)
  %5359 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5156, <4 x i32> %5291, <4 x i32> %4961, i32 %5212, i32 %4966)
  %5360 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5359, <4 x i32> %5293, <4 x i32> %4962, i32 %5212, i32 %4966)
  %5361 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5158, <4 x i32> %5295, <4 x i32> %4961, i32 %5217, i32 %4966)
  %5362 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5361, <4 x i32> %5297, <4 x i32> %4962, i32 %5217, i32 %4966)
  %5363 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5160, <4 x i32> %5299, <4 x i32> %4961, i32 %5217, i32 %4966)
  %5364 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5363, <4 x i32> %5301, <4 x i32> %4962, i32 %5217, i32 %4966)
  %5365 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5162, <4 x i32> %5303, <4 x i32> %4961, i32 %5222, i32 %4966)
  %5366 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5365, <4 x i32> %5305, <4 x i32> %4962, i32 %5222, i32 %4966)
  %5367 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5164, <4 x i32> %5307, <4 x i32> %4961, i32 %5222, i32 %4966)
  %5368 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5367, <4 x i32> %5309, <4 x i32> %4962, i32 %5222, i32 %4966)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5369 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5312, i32 %188, i32 2)
  %5370 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5315, i32 %188, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5371 = add i32 %211, 1536
  %5372 = mul i32 %5371, 4
  %5373 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %5372, i32 %204, i32 0)
  %5374 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %5372, i32 %210, i32 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %5375 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1075
  %5376 = load <16 x i8>, ptr addrspace(3) %5375, align 1
  %5377 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1078
  %5378 = load <16 x i8>, ptr addrspace(3) %5377, align 1
  %5379 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1081
  %5380 = load <16 x i8>, ptr addrspace(3) %5379, align 1
  %5381 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1084
  %5382 = load <16 x i8>, ptr addrspace(3) %5381, align 1
  %5383 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1087
  %5384 = load <16 x i8>, ptr addrspace(3) %5383, align 1
  %5385 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1090
  %5386 = load <16 x i8>, ptr addrspace(3) %5385, align 1
  %5387 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1093
  %5388 = load <16 x i8>, ptr addrspace(3) %5387, align 1
  %5389 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1096
  %5390 = load <16 x i8>, ptr addrspace(3) %5389, align 1
  %5391 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1099
  %5392 = load <16 x i8>, ptr addrspace(3) %5391, align 1
  %5393 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1102
  %5394 = load <16 x i8>, ptr addrspace(3) %5393, align 1
  %5395 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1105
  %5396 = load <16 x i8>, ptr addrspace(3) %5395, align 1
  %5397 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1108
  %5398 = load <16 x i8>, ptr addrspace(3) %5397, align 1
  %5399 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1111
  %5400 = load <16 x i8>, ptr addrspace(3) %5399, align 1
  %5401 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1114
  %5402 = load <16 x i8>, ptr addrspace(3) %5401, align 1
  %5403 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1117
  %5404 = load <16 x i8>, ptr addrspace(3) %5403, align 1
  %5405 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1120
  %5406 = load <16 x i8>, ptr addrspace(3) %5405, align 1
  %5407 = add i64 %193, 1472
  %5408 = add i64 %5407, %24
  %5409 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %5408
  %5410 = load <1 x i32>, ptr addrspace(3) %5409, align 4
  %5411 = extractelement <1 x i32> %5410, i64 0
  %5412 = add i64 %193, 3264
  %5413 = add i64 %5412, %24
  %5414 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %5413
  %5415 = load <1 x i32>, ptr addrspace(3) %5414, align 4
  %5416 = extractelement <1 x i32> %5415, i64 0
  %5417 = add i64 %193, 5056
  %5418 = add i64 %5417, %24
  %5419 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %5418
  %5420 = load <1 x i32>, ptr addrspace(3) %5419, align 4
  %5421 = extractelement <1 x i32> %5420, i64 0
  %5422 = add i64 %193, 6848
  %5423 = add i64 %5422, %24
  %5424 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %5423
  %5425 = load <1 x i32>, ptr addrspace(3) %5424, align 4
  %5426 = extractelement <1 x i32> %5425, i64 0
  %5427 = add i64 %77, 800
  %5428 = add i64 %5427, %71
  %5429 = trunc i64 %5428 to i32
  %5430 = mul i32 %5429, 4
  %5431 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5430, i32 0, i32 0)
  %5432 = bitcast <4 x i32> %5431 to <16 x i8>
  %5433 = add i64 %90, 800
  %5434 = add i64 %5433, %84
  %5435 = trunc i64 %5434 to i32
  %5436 = mul i32 %5435, 4
  %5437 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5436, i32 0, i32 0)
  %5438 = bitcast <4 x i32> %5437 to <16 x i8>
  %5439 = add i64 %103, 800
  %5440 = add i64 %5439, %97
  %5441 = trunc i64 %5440 to i32
  %5442 = mul i32 %5441, 4
  %5443 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5442, i32 0, i32 0)
  %5444 = bitcast <4 x i32> %5443 to <16 x i8>
  %5445 = add i64 %116, 800
  %5446 = add i64 %5445, %110
  %5447 = trunc i64 %5446 to i32
  %5448 = mul i32 %5447, 4
  %5449 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5448, i32 0, i32 0)
  %5450 = bitcast <4 x i32> %5449 to <16 x i8>
  %5451 = add i64 %129, 800
  %5452 = add i64 %5451, %123
  %5453 = trunc i64 %5452 to i32
  %5454 = mul i32 %5453, 4
  %5455 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5454, i32 0, i32 0)
  %5456 = bitcast <4 x i32> %5455 to <16 x i8>
  %5457 = add i64 %142, 800
  %5458 = add i64 %5457, %136
  %5459 = trunc i64 %5458 to i32
  %5460 = mul i32 %5459, 4
  %5461 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5460, i32 0, i32 0)
  %5462 = bitcast <4 x i32> %5461 to <16 x i8>
  %5463 = add i64 %155, 800
  %5464 = add i64 %5463, %149
  %5465 = trunc i64 %5464 to i32
  %5466 = mul i32 %5465, 4
  %5467 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5466, i32 0, i32 0)
  %5468 = bitcast <4 x i32> %5467 to <16 x i8>
  %5469 = add i64 %168, 800
  %5470 = add i64 %5469, %162
  %5471 = trunc i64 %5470 to i32
  %5472 = mul i32 %5471, 4
  %5473 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5472, i32 0, i32 0)
  %5474 = bitcast <4 x i32> %5473 to <16 x i8>
  %5475 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %487
  store <16 x i8> %5432, ptr addrspace(3) %5475, align 1
  %5476 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %489
  store <16 x i8> %5438, ptr addrspace(3) %5476, align 1
  %5477 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %491
  store <16 x i8> %5444, ptr addrspace(3) %5477, align 1
  %5478 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %493
  store <16 x i8> %5450, ptr addrspace(3) %5478, align 1
  %5479 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %495
  store <16 x i8> %5456, ptr addrspace(3) %5479, align 1
  %5480 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %497
  store <16 x i8> %5462, ptr addrspace(3) %5480, align 1
  %5481 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %499
  store <16 x i8> %5468, ptr addrspace(3) %5481, align 1
  %5482 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %501
  store <16 x i8> %5474, ptr addrspace(3) %5482, align 1
  %5483 = bitcast <16 x i8> %5376 to <4 x i32>
  %5484 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5282, <4 x i32> %5483, <4 x i32> %5109, i32 %5411, i32 %5169)
  %5485 = bitcast <16 x i8> %5378 to <4 x i32>
  %5486 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5484, <4 x i32> %5485, <4 x i32> %5112, i32 %5411, i32 %5169)
  %5487 = bitcast <16 x i8> %5380 to <4 x i32>
  %5488 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5286, <4 x i32> %5487, <4 x i32> %5109, i32 %5411, i32 %5169)
  %5489 = bitcast <16 x i8> %5382 to <4 x i32>
  %5490 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5488, <4 x i32> %5489, <4 x i32> %5112, i32 %5411, i32 %5169)
  %5491 = bitcast <16 x i8> %5384 to <4 x i32>
  %5492 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5290, <4 x i32> %5491, <4 x i32> %5109, i32 %5416, i32 %5169)
  %5493 = bitcast <16 x i8> %5386 to <4 x i32>
  %5494 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5492, <4 x i32> %5493, <4 x i32> %5112, i32 %5416, i32 %5169)
  %5495 = bitcast <16 x i8> %5388 to <4 x i32>
  %5496 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5294, <4 x i32> %5495, <4 x i32> %5109, i32 %5416, i32 %5169)
  %5497 = bitcast <16 x i8> %5390 to <4 x i32>
  %5498 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5496, <4 x i32> %5497, <4 x i32> %5112, i32 %5416, i32 %5169)
  %5499 = bitcast <16 x i8> %5392 to <4 x i32>
  %5500 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5298, <4 x i32> %5499, <4 x i32> %5109, i32 %5421, i32 %5169)
  %5501 = bitcast <16 x i8> %5394 to <4 x i32>
  %5502 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5500, <4 x i32> %5501, <4 x i32> %5112, i32 %5421, i32 %5169)
  %5503 = bitcast <16 x i8> %5396 to <4 x i32>
  %5504 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5302, <4 x i32> %5503, <4 x i32> %5109, i32 %5421, i32 %5169)
  %5505 = bitcast <16 x i8> %5398 to <4 x i32>
  %5506 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5504, <4 x i32> %5505, <4 x i32> %5112, i32 %5421, i32 %5169)
  %5507 = bitcast <16 x i8> %5400 to <4 x i32>
  %5508 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5306, <4 x i32> %5507, <4 x i32> %5109, i32 %5426, i32 %5169)
  %5509 = bitcast <16 x i8> %5402 to <4 x i32>
  %5510 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5508, <4 x i32> %5509, <4 x i32> %5112, i32 %5426, i32 %5169)
  %5511 = bitcast <16 x i8> %5404 to <4 x i32>
  %5512 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5310, <4 x i32> %5511, <4 x i32> %5109, i32 %5426, i32 %5169)
  %5513 = bitcast <16 x i8> %5406 to <4 x i32>
  %5514 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5512, <4 x i32> %5513, <4 x i32> %5112, i32 %5426, i32 %5169)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5515 = add i32 %192, 12800
  %5516 = mul i32 %5515, 4
  %5517 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5516, i32 %176, i32 2)
  %5518 = add i32 %192, 13056
  %5519 = mul i32 %5518, 4
  %5520 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5519, i32 %176, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5521 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5318, <4 x i32> %5483, <4 x i32> %5129, i32 %5411, i32 %5169)
  %5522 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5521, <4 x i32> %5485, <4 x i32> %5130, i32 %5411, i32 %5169)
  %5523 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5320, <4 x i32> %5487, <4 x i32> %5129, i32 %5411, i32 %5169)
  %5524 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5523, <4 x i32> %5489, <4 x i32> %5130, i32 %5411, i32 %5169)
  %5525 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5322, <4 x i32> %5491, <4 x i32> %5129, i32 %5416, i32 %5169)
  %5526 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5525, <4 x i32> %5493, <4 x i32> %5130, i32 %5416, i32 %5169)
  %5527 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5324, <4 x i32> %5495, <4 x i32> %5129, i32 %5416, i32 %5169)
  %5528 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5527, <4 x i32> %5497, <4 x i32> %5130, i32 %5416, i32 %5169)
  %5529 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5326, <4 x i32> %5499, <4 x i32> %5129, i32 %5421, i32 %5169)
  %5530 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5529, <4 x i32> %5501, <4 x i32> %5130, i32 %5421, i32 %5169)
  %5531 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5328, <4 x i32> %5503, <4 x i32> %5129, i32 %5421, i32 %5169)
  %5532 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5531, <4 x i32> %5505, <4 x i32> %5130, i32 %5421, i32 %5169)
  %5533 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5330, <4 x i32> %5507, <4 x i32> %5129, i32 %5426, i32 %5169)
  %5534 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5533, <4 x i32> %5509, <4 x i32> %5130, i32 %5426, i32 %5169)
  %5535 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5332, <4 x i32> %5511, <4 x i32> %5129, i32 %5426, i32 %5169)
  %5536 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5535, <4 x i32> %5513, <4 x i32> %5130, i32 %5426, i32 %5169)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5537 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5516, i32 %180, i32 2)
  %5538 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5519, i32 %180, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5539 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5336, <4 x i32> %5483, <4 x i32> %5147, i32 %5411, i32 %5170)
  %5540 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5539, <4 x i32> %5485, <4 x i32> %5148, i32 %5411, i32 %5170)
  %5541 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5338, <4 x i32> %5487, <4 x i32> %5147, i32 %5411, i32 %5170)
  %5542 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5541, <4 x i32> %5489, <4 x i32> %5148, i32 %5411, i32 %5170)
  %5543 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5340, <4 x i32> %5491, <4 x i32> %5147, i32 %5416, i32 %5170)
  %5544 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5543, <4 x i32> %5493, <4 x i32> %5148, i32 %5416, i32 %5170)
  %5545 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5342, <4 x i32> %5495, <4 x i32> %5147, i32 %5416, i32 %5170)
  %5546 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5545, <4 x i32> %5497, <4 x i32> %5148, i32 %5416, i32 %5170)
  %5547 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5344, <4 x i32> %5499, <4 x i32> %5147, i32 %5421, i32 %5170)
  %5548 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5547, <4 x i32> %5501, <4 x i32> %5148, i32 %5421, i32 %5170)
  %5549 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5346, <4 x i32> %5503, <4 x i32> %5147, i32 %5421, i32 %5170)
  %5550 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5549, <4 x i32> %5505, <4 x i32> %5148, i32 %5421, i32 %5170)
  %5551 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5348, <4 x i32> %5507, <4 x i32> %5147, i32 %5426, i32 %5170)
  %5552 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5551, <4 x i32> %5509, <4 x i32> %5148, i32 %5426, i32 %5170)
  %5553 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5350, <4 x i32> %5511, <4 x i32> %5147, i32 %5426, i32 %5170)
  %5554 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5553, <4 x i32> %5513, <4 x i32> %5148, i32 %5426, i32 %5170)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5555 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5516, i32 %184, i32 2)
  %5556 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5519, i32 %184, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5557 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5354, <4 x i32> %5483, <4 x i32> %5165, i32 %5411, i32 %5170)
  %5558 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5557, <4 x i32> %5485, <4 x i32> %5166, i32 %5411, i32 %5170)
  %5559 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5356, <4 x i32> %5487, <4 x i32> %5165, i32 %5411, i32 %5170)
  %5560 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5559, <4 x i32> %5489, <4 x i32> %5166, i32 %5411, i32 %5170)
  %5561 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5358, <4 x i32> %5491, <4 x i32> %5165, i32 %5416, i32 %5170)
  %5562 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5561, <4 x i32> %5493, <4 x i32> %5166, i32 %5416, i32 %5170)
  %5563 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5360, <4 x i32> %5495, <4 x i32> %5165, i32 %5416, i32 %5170)
  %5564 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5563, <4 x i32> %5497, <4 x i32> %5166, i32 %5416, i32 %5170)
  %5565 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5362, <4 x i32> %5499, <4 x i32> %5165, i32 %5421, i32 %5170)
  %5566 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5565, <4 x i32> %5501, <4 x i32> %5166, i32 %5421, i32 %5170)
  %5567 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5364, <4 x i32> %5503, <4 x i32> %5165, i32 %5421, i32 %5170)
  %5568 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5567, <4 x i32> %5505, <4 x i32> %5166, i32 %5421, i32 %5170)
  %5569 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5366, <4 x i32> %5507, <4 x i32> %5165, i32 %5426, i32 %5170)
  %5570 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5569, <4 x i32> %5509, <4 x i32> %5166, i32 %5426, i32 %5170)
  %5571 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5368, <4 x i32> %5511, <4 x i32> %5165, i32 %5426, i32 %5170)
  %5572 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5571, <4 x i32> %5513, <4 x i32> %5166, i32 %5426, i32 %5170)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5573 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5516, i32 %188, i32 2)
  %5574 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5519, i32 %188, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5575 = add i32 %211, 1600
  %5576 = mul i32 %5575, 4
  %5577 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %5576, i32 %204, i32 0)
  %5578 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %5576, i32 %210, i32 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %5579 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %522
  %5580 = load <16 x i8>, ptr addrspace(3) %5579, align 1
  %5581 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %527
  %5582 = load <16 x i8>, ptr addrspace(3) %5581, align 1
  %5583 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %535
  %5584 = load <16 x i8>, ptr addrspace(3) %5583, align 1
  %5585 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %539
  %5586 = load <16 x i8>, ptr addrspace(3) %5585, align 1
  %5587 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %547
  %5588 = load <16 x i8>, ptr addrspace(3) %5587, align 1
  %5589 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %551
  %5590 = load <16 x i8>, ptr addrspace(3) %5589, align 1
  %5591 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %559
  %5592 = load <16 x i8>, ptr addrspace(3) %5591, align 1
  %5593 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %563
  %5594 = load <16 x i8>, ptr addrspace(3) %5593, align 1
  %5595 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %571
  %5596 = load <16 x i8>, ptr addrspace(3) %5595, align 1
  %5597 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %575
  %5598 = load <16 x i8>, ptr addrspace(3) %5597, align 1
  %5599 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %583
  %5600 = load <16 x i8>, ptr addrspace(3) %5599, align 1
  %5601 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %587
  %5602 = load <16 x i8>, ptr addrspace(3) %5601, align 1
  %5603 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %595
  %5604 = load <16 x i8>, ptr addrspace(3) %5603, align 1
  %5605 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %599
  %5606 = load <16 x i8>, ptr addrspace(3) %5605, align 1
  %5607 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %607
  %5608 = load <16 x i8>, ptr addrspace(3) %5607, align 1
  %5609 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %611
  %5610 = load <16 x i8>, ptr addrspace(3) %5609, align 1
  %5611 = add i64 %193, 1536
  %5612 = add i64 %5611, %24
  %5613 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %5612
  %5614 = load <1 x i32>, ptr addrspace(3) %5613, align 4
  %5615 = extractelement <1 x i32> %5614, i64 0
  %5616 = add i64 %193, 3328
  %5617 = add i64 %5616, %24
  %5618 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %5617
  %5619 = load <1 x i32>, ptr addrspace(3) %5618, align 4
  %5620 = extractelement <1 x i32> %5619, i64 0
  %5621 = add i64 %193, 5120
  %5622 = add i64 %5621, %24
  %5623 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %5622
  %5624 = load <1 x i32>, ptr addrspace(3) %5623, align 4
  %5625 = extractelement <1 x i32> %5624, i64 0
  %5626 = add i64 %193, 6912
  %5627 = add i64 %5626, %24
  %5628 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %5627
  %5629 = load <1 x i32>, ptr addrspace(3) %5628, align 4
  %5630 = extractelement <1 x i32> %5629, i64 0
  %5631 = add i64 %77, 832
  %5632 = add i64 %5631, %71
  %5633 = trunc i64 %5632 to i32
  %5634 = mul i32 %5633, 4
  %5635 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5634, i32 0, i32 0)
  %5636 = bitcast <4 x i32> %5635 to <16 x i8>
  %5637 = add i64 %90, 832
  %5638 = add i64 %5637, %84
  %5639 = trunc i64 %5638 to i32
  %5640 = mul i32 %5639, 4
  %5641 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5640, i32 0, i32 0)
  %5642 = bitcast <4 x i32> %5641 to <16 x i8>
  %5643 = add i64 %103, 832
  %5644 = add i64 %5643, %97
  %5645 = trunc i64 %5644 to i32
  %5646 = mul i32 %5645, 4
  %5647 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5646, i32 0, i32 0)
  %5648 = bitcast <4 x i32> %5647 to <16 x i8>
  %5649 = add i64 %116, 832
  %5650 = add i64 %5649, %110
  %5651 = trunc i64 %5650 to i32
  %5652 = mul i32 %5651, 4
  %5653 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5652, i32 0, i32 0)
  %5654 = bitcast <4 x i32> %5653 to <16 x i8>
  %5655 = add i64 %129, 832
  %5656 = add i64 %5655, %123
  %5657 = trunc i64 %5656 to i32
  %5658 = mul i32 %5657, 4
  %5659 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5658, i32 0, i32 0)
  %5660 = bitcast <4 x i32> %5659 to <16 x i8>
  %5661 = add i64 %142, 832
  %5662 = add i64 %5661, %136
  %5663 = trunc i64 %5662 to i32
  %5664 = mul i32 %5663, 4
  %5665 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5664, i32 0, i32 0)
  %5666 = bitcast <4 x i32> %5665 to <16 x i8>
  %5667 = add i64 %155, 832
  %5668 = add i64 %5667, %149
  %5669 = trunc i64 %5668 to i32
  %5670 = mul i32 %5669, 4
  %5671 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5670, i32 0, i32 0)
  %5672 = bitcast <4 x i32> %5671 to <16 x i8>
  %5673 = add i64 %168, 832
  %5674 = add i64 %5673, %162
  %5675 = trunc i64 %5674 to i32
  %5676 = mul i32 %5675, 4
  %5677 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5676, i32 0, i32 0)
  %5678 = bitcast <4 x i32> %5677 to <16 x i8>
  %5679 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %680
  store <16 x i8> %5636, ptr addrspace(3) %5679, align 1
  %5680 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %682
  store <16 x i8> %5642, ptr addrspace(3) %5680, align 1
  %5681 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %684
  store <16 x i8> %5648, ptr addrspace(3) %5681, align 1
  %5682 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %686
  store <16 x i8> %5654, ptr addrspace(3) %5682, align 1
  %5683 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %688
  store <16 x i8> %5660, ptr addrspace(3) %5683, align 1
  %5684 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %690
  store <16 x i8> %5666, ptr addrspace(3) %5684, align 1
  %5685 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %692
  store <16 x i8> %5672, ptr addrspace(3) %5685, align 1
  %5686 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %694
  store <16 x i8> %5678, ptr addrspace(3) %5686, align 1
  %5687 = bitcast <16 x i8> %5580 to <4 x i32>
  %5688 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5486, <4 x i32> %5687, <4 x i32> %5313, i32 %5615, i32 %5373)
  %5689 = bitcast <16 x i8> %5582 to <4 x i32>
  %5690 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5688, <4 x i32> %5689, <4 x i32> %5316, i32 %5615, i32 %5373)
  %5691 = bitcast <16 x i8> %5584 to <4 x i32>
  %5692 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5490, <4 x i32> %5691, <4 x i32> %5313, i32 %5615, i32 %5373)
  %5693 = bitcast <16 x i8> %5586 to <4 x i32>
  %5694 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5692, <4 x i32> %5693, <4 x i32> %5316, i32 %5615, i32 %5373)
  %5695 = bitcast <16 x i8> %5588 to <4 x i32>
  %5696 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5494, <4 x i32> %5695, <4 x i32> %5313, i32 %5620, i32 %5373)
  %5697 = bitcast <16 x i8> %5590 to <4 x i32>
  %5698 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5696, <4 x i32> %5697, <4 x i32> %5316, i32 %5620, i32 %5373)
  %5699 = bitcast <16 x i8> %5592 to <4 x i32>
  %5700 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5498, <4 x i32> %5699, <4 x i32> %5313, i32 %5620, i32 %5373)
  %5701 = bitcast <16 x i8> %5594 to <4 x i32>
  %5702 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5700, <4 x i32> %5701, <4 x i32> %5316, i32 %5620, i32 %5373)
  %5703 = bitcast <16 x i8> %5596 to <4 x i32>
  %5704 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5502, <4 x i32> %5703, <4 x i32> %5313, i32 %5625, i32 %5373)
  %5705 = bitcast <16 x i8> %5598 to <4 x i32>
  %5706 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5704, <4 x i32> %5705, <4 x i32> %5316, i32 %5625, i32 %5373)
  %5707 = bitcast <16 x i8> %5600 to <4 x i32>
  %5708 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5506, <4 x i32> %5707, <4 x i32> %5313, i32 %5625, i32 %5373)
  %5709 = bitcast <16 x i8> %5602 to <4 x i32>
  %5710 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5708, <4 x i32> %5709, <4 x i32> %5316, i32 %5625, i32 %5373)
  %5711 = bitcast <16 x i8> %5604 to <4 x i32>
  %5712 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5510, <4 x i32> %5711, <4 x i32> %5313, i32 %5630, i32 %5373)
  %5713 = bitcast <16 x i8> %5606 to <4 x i32>
  %5714 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5712, <4 x i32> %5713, <4 x i32> %5316, i32 %5630, i32 %5373)
  %5715 = bitcast <16 x i8> %5608 to <4 x i32>
  %5716 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5514, <4 x i32> %5715, <4 x i32> %5313, i32 %5630, i32 %5373)
  %5717 = bitcast <16 x i8> %5610 to <4 x i32>
  %5718 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5716, <4 x i32> %5717, <4 x i32> %5316, i32 %5630, i32 %5373)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5719 = add i32 %192, 13312
  %5720 = mul i32 %5719, 4
  %5721 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5720, i32 %176, i32 2)
  %5722 = add i32 %192, 13568
  %5723 = mul i32 %5722, 4
  %5724 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5723, i32 %176, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5725 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5522, <4 x i32> %5687, <4 x i32> %5333, i32 %5615, i32 %5373)
  %5726 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5725, <4 x i32> %5689, <4 x i32> %5334, i32 %5615, i32 %5373)
  %5727 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5524, <4 x i32> %5691, <4 x i32> %5333, i32 %5615, i32 %5373)
  %5728 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5727, <4 x i32> %5693, <4 x i32> %5334, i32 %5615, i32 %5373)
  %5729 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5526, <4 x i32> %5695, <4 x i32> %5333, i32 %5620, i32 %5373)
  %5730 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5729, <4 x i32> %5697, <4 x i32> %5334, i32 %5620, i32 %5373)
  %5731 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5528, <4 x i32> %5699, <4 x i32> %5333, i32 %5620, i32 %5373)
  %5732 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5731, <4 x i32> %5701, <4 x i32> %5334, i32 %5620, i32 %5373)
  %5733 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5530, <4 x i32> %5703, <4 x i32> %5333, i32 %5625, i32 %5373)
  %5734 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5733, <4 x i32> %5705, <4 x i32> %5334, i32 %5625, i32 %5373)
  %5735 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5532, <4 x i32> %5707, <4 x i32> %5333, i32 %5625, i32 %5373)
  %5736 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5735, <4 x i32> %5709, <4 x i32> %5334, i32 %5625, i32 %5373)
  %5737 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5534, <4 x i32> %5711, <4 x i32> %5333, i32 %5630, i32 %5373)
  %5738 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5737, <4 x i32> %5713, <4 x i32> %5334, i32 %5630, i32 %5373)
  %5739 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5536, <4 x i32> %5715, <4 x i32> %5333, i32 %5630, i32 %5373)
  %5740 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5739, <4 x i32> %5717, <4 x i32> %5334, i32 %5630, i32 %5373)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5741 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5720, i32 %180, i32 2)
  %5742 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5723, i32 %180, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5743 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5540, <4 x i32> %5687, <4 x i32> %5351, i32 %5615, i32 %5374)
  %5744 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5743, <4 x i32> %5689, <4 x i32> %5352, i32 %5615, i32 %5374)
  %5745 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5542, <4 x i32> %5691, <4 x i32> %5351, i32 %5615, i32 %5374)
  %5746 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5745, <4 x i32> %5693, <4 x i32> %5352, i32 %5615, i32 %5374)
  %5747 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5544, <4 x i32> %5695, <4 x i32> %5351, i32 %5620, i32 %5374)
  %5748 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5747, <4 x i32> %5697, <4 x i32> %5352, i32 %5620, i32 %5374)
  %5749 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5546, <4 x i32> %5699, <4 x i32> %5351, i32 %5620, i32 %5374)
  %5750 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5749, <4 x i32> %5701, <4 x i32> %5352, i32 %5620, i32 %5374)
  %5751 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5548, <4 x i32> %5703, <4 x i32> %5351, i32 %5625, i32 %5374)
  %5752 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5751, <4 x i32> %5705, <4 x i32> %5352, i32 %5625, i32 %5374)
  %5753 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5550, <4 x i32> %5707, <4 x i32> %5351, i32 %5625, i32 %5374)
  %5754 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5753, <4 x i32> %5709, <4 x i32> %5352, i32 %5625, i32 %5374)
  %5755 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5552, <4 x i32> %5711, <4 x i32> %5351, i32 %5630, i32 %5374)
  %5756 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5755, <4 x i32> %5713, <4 x i32> %5352, i32 %5630, i32 %5374)
  %5757 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5554, <4 x i32> %5715, <4 x i32> %5351, i32 %5630, i32 %5374)
  %5758 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5757, <4 x i32> %5717, <4 x i32> %5352, i32 %5630, i32 %5374)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5759 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5720, i32 %184, i32 2)
  %5760 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5723, i32 %184, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5761 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5558, <4 x i32> %5687, <4 x i32> %5369, i32 %5615, i32 %5374)
  %5762 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5761, <4 x i32> %5689, <4 x i32> %5370, i32 %5615, i32 %5374)
  %5763 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5560, <4 x i32> %5691, <4 x i32> %5369, i32 %5615, i32 %5374)
  %5764 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5763, <4 x i32> %5693, <4 x i32> %5370, i32 %5615, i32 %5374)
  %5765 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5562, <4 x i32> %5695, <4 x i32> %5369, i32 %5620, i32 %5374)
  %5766 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5765, <4 x i32> %5697, <4 x i32> %5370, i32 %5620, i32 %5374)
  %5767 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5564, <4 x i32> %5699, <4 x i32> %5369, i32 %5620, i32 %5374)
  %5768 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5767, <4 x i32> %5701, <4 x i32> %5370, i32 %5620, i32 %5374)
  %5769 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5566, <4 x i32> %5703, <4 x i32> %5369, i32 %5625, i32 %5374)
  %5770 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5769, <4 x i32> %5705, <4 x i32> %5370, i32 %5625, i32 %5374)
  %5771 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5568, <4 x i32> %5707, <4 x i32> %5369, i32 %5625, i32 %5374)
  %5772 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5771, <4 x i32> %5709, <4 x i32> %5370, i32 %5625, i32 %5374)
  %5773 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5570, <4 x i32> %5711, <4 x i32> %5369, i32 %5630, i32 %5374)
  %5774 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5773, <4 x i32> %5713, <4 x i32> %5370, i32 %5630, i32 %5374)
  %5775 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5572, <4 x i32> %5715, <4 x i32> %5369, i32 %5630, i32 %5374)
  %5776 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5775, <4 x i32> %5717, <4 x i32> %5370, i32 %5630, i32 %5374)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5777 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5720, i32 %188, i32 2)
  %5778 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5723, i32 %188, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5779 = add i32 %211, 1664
  %5780 = mul i32 %5779, 4
  %5781 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %5780, i32 %204, i32 0)
  %5782 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %5780, i32 %210, i32 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %5783 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %856
  %5784 = load <16 x i8>, ptr addrspace(3) %5783, align 1
  %5785 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %859
  %5786 = load <16 x i8>, ptr addrspace(3) %5785, align 1
  %5787 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %862
  %5788 = load <16 x i8>, ptr addrspace(3) %5787, align 1
  %5789 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %865
  %5790 = load <16 x i8>, ptr addrspace(3) %5789, align 1
  %5791 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %868
  %5792 = load <16 x i8>, ptr addrspace(3) %5791, align 1
  %5793 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %871
  %5794 = load <16 x i8>, ptr addrspace(3) %5793, align 1
  %5795 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %874
  %5796 = load <16 x i8>, ptr addrspace(3) %5795, align 1
  %5797 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %877
  %5798 = load <16 x i8>, ptr addrspace(3) %5797, align 1
  %5799 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %880
  %5800 = load <16 x i8>, ptr addrspace(3) %5799, align 1
  %5801 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %883
  %5802 = load <16 x i8>, ptr addrspace(3) %5801, align 1
  %5803 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %886
  %5804 = load <16 x i8>, ptr addrspace(3) %5803, align 1
  %5805 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %889
  %5806 = load <16 x i8>, ptr addrspace(3) %5805, align 1
  %5807 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %892
  %5808 = load <16 x i8>, ptr addrspace(3) %5807, align 1
  %5809 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %895
  %5810 = load <16 x i8>, ptr addrspace(3) %5809, align 1
  %5811 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %898
  %5812 = load <16 x i8>, ptr addrspace(3) %5811, align 1
  %5813 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %901
  %5814 = load <16 x i8>, ptr addrspace(3) %5813, align 1
  %5815 = add i64 %193, 1600
  %5816 = add i64 %5815, %24
  %5817 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %5816
  %5818 = load <1 x i32>, ptr addrspace(3) %5817, align 4
  %5819 = extractelement <1 x i32> %5818, i64 0
  %5820 = add i64 %193, 3392
  %5821 = add i64 %5820, %24
  %5822 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %5821
  %5823 = load <1 x i32>, ptr addrspace(3) %5822, align 4
  %5824 = extractelement <1 x i32> %5823, i64 0
  %5825 = add i64 %193, 5184
  %5826 = add i64 %5825, %24
  %5827 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %5826
  %5828 = load <1 x i32>, ptr addrspace(3) %5827, align 4
  %5829 = extractelement <1 x i32> %5828, i64 0
  %5830 = add i64 %193, 6976
  %5831 = add i64 %5830, %24
  %5832 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %5831
  %5833 = load <1 x i32>, ptr addrspace(3) %5832, align 4
  %5834 = extractelement <1 x i32> %5833, i64 0
  %5835 = add i64 %77, 864
  %5836 = add i64 %5835, %71
  %5837 = trunc i64 %5836 to i32
  %5838 = mul i32 %5837, 4
  %5839 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5838, i32 0, i32 0)
  %5840 = bitcast <4 x i32> %5839 to <16 x i8>
  %5841 = add i64 %90, 864
  %5842 = add i64 %5841, %84
  %5843 = trunc i64 %5842 to i32
  %5844 = mul i32 %5843, 4
  %5845 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5844, i32 0, i32 0)
  %5846 = bitcast <4 x i32> %5845 to <16 x i8>
  %5847 = add i64 %103, 864
  %5848 = add i64 %5847, %97
  %5849 = trunc i64 %5848 to i32
  %5850 = mul i32 %5849, 4
  %5851 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5850, i32 0, i32 0)
  %5852 = bitcast <4 x i32> %5851 to <16 x i8>
  %5853 = add i64 %116, 864
  %5854 = add i64 %5853, %110
  %5855 = trunc i64 %5854 to i32
  %5856 = mul i32 %5855, 4
  %5857 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5856, i32 0, i32 0)
  %5858 = bitcast <4 x i32> %5857 to <16 x i8>
  %5859 = add i64 %129, 864
  %5860 = add i64 %5859, %123
  %5861 = trunc i64 %5860 to i32
  %5862 = mul i32 %5861, 4
  %5863 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5862, i32 0, i32 0)
  %5864 = bitcast <4 x i32> %5863 to <16 x i8>
  %5865 = add i64 %142, 864
  %5866 = add i64 %5865, %136
  %5867 = trunc i64 %5866 to i32
  %5868 = mul i32 %5867, 4
  %5869 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5868, i32 0, i32 0)
  %5870 = bitcast <4 x i32> %5869 to <16 x i8>
  %5871 = add i64 %155, 864
  %5872 = add i64 %5871, %149
  %5873 = trunc i64 %5872 to i32
  %5874 = mul i32 %5873, 4
  %5875 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5874, i32 0, i32 0)
  %5876 = bitcast <4 x i32> %5875 to <16 x i8>
  %5877 = add i64 %168, 864
  %5878 = add i64 %5877, %162
  %5879 = trunc i64 %5878 to i32
  %5880 = mul i32 %5879, 4
  %5881 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %32, i32 %5880, i32 0, i32 0)
  %5882 = bitcast <4 x i32> %5881 to <16 x i8>
  %5883 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %360
  store <16 x i8> %5840, ptr addrspace(3) %5883, align 1
  %5884 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %369
  store <16 x i8> %5846, ptr addrspace(3) %5884, align 1
  %5885 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %378
  store <16 x i8> %5852, ptr addrspace(3) %5885, align 1
  %5886 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %387
  store <16 x i8> %5858, ptr addrspace(3) %5886, align 1
  %5887 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %396
  store <16 x i8> %5864, ptr addrspace(3) %5887, align 1
  %5888 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %405
  store <16 x i8> %5870, ptr addrspace(3) %5888, align 1
  %5889 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %414
  store <16 x i8> %5876, ptr addrspace(3) %5889, align 1
  %5890 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %423
  store <16 x i8> %5882, ptr addrspace(3) %5890, align 1
  %5891 = bitcast <16 x i8> %5784 to <4 x i32>
  %5892 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5690, <4 x i32> %5891, <4 x i32> %5517, i32 %5819, i32 %5577)
  %5893 = bitcast <16 x i8> %5786 to <4 x i32>
  %5894 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5892, <4 x i32> %5893, <4 x i32> %5520, i32 %5819, i32 %5577)
  %5895 = bitcast <16 x i8> %5788 to <4 x i32>
  %5896 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5694, <4 x i32> %5895, <4 x i32> %5517, i32 %5819, i32 %5577)
  %5897 = bitcast <16 x i8> %5790 to <4 x i32>
  %5898 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5896, <4 x i32> %5897, <4 x i32> %5520, i32 %5819, i32 %5577)
  %5899 = bitcast <16 x i8> %5792 to <4 x i32>
  %5900 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5698, <4 x i32> %5899, <4 x i32> %5517, i32 %5824, i32 %5577)
  %5901 = bitcast <16 x i8> %5794 to <4 x i32>
  %5902 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5900, <4 x i32> %5901, <4 x i32> %5520, i32 %5824, i32 %5577)
  %5903 = bitcast <16 x i8> %5796 to <4 x i32>
  %5904 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5702, <4 x i32> %5903, <4 x i32> %5517, i32 %5824, i32 %5577)
  %5905 = bitcast <16 x i8> %5798 to <4 x i32>
  %5906 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5904, <4 x i32> %5905, <4 x i32> %5520, i32 %5824, i32 %5577)
  %5907 = bitcast <16 x i8> %5800 to <4 x i32>
  %5908 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5706, <4 x i32> %5907, <4 x i32> %5517, i32 %5829, i32 %5577)
  %5909 = bitcast <16 x i8> %5802 to <4 x i32>
  %5910 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5908, <4 x i32> %5909, <4 x i32> %5520, i32 %5829, i32 %5577)
  %5911 = bitcast <16 x i8> %5804 to <4 x i32>
  %5912 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5710, <4 x i32> %5911, <4 x i32> %5517, i32 %5829, i32 %5577)
  %5913 = bitcast <16 x i8> %5806 to <4 x i32>
  %5914 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5912, <4 x i32> %5913, <4 x i32> %5520, i32 %5829, i32 %5577)
  %5915 = bitcast <16 x i8> %5808 to <4 x i32>
  %5916 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5714, <4 x i32> %5915, <4 x i32> %5517, i32 %5834, i32 %5577)
  %5917 = bitcast <16 x i8> %5810 to <4 x i32>
  %5918 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5916, <4 x i32> %5917, <4 x i32> %5520, i32 %5834, i32 %5577)
  %5919 = bitcast <16 x i8> %5812 to <4 x i32>
  %5920 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5718, <4 x i32> %5919, <4 x i32> %5517, i32 %5834, i32 %5577)
  %5921 = bitcast <16 x i8> %5814 to <4 x i32>
  %5922 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5920, <4 x i32> %5921, <4 x i32> %5520, i32 %5834, i32 %5577)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5923 = add i32 %192, 13824
  %5924 = mul i32 %5923, 4
  %5925 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5924, i32 %176, i32 2)
  %5926 = add i32 %192, 14080
  %5927 = mul i32 %5926, 4
  %5928 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5927, i32 %176, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5929 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5726, <4 x i32> %5891, <4 x i32> %5537, i32 %5819, i32 %5577)
  %5930 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5929, <4 x i32> %5893, <4 x i32> %5538, i32 %5819, i32 %5577)
  %5931 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5728, <4 x i32> %5895, <4 x i32> %5537, i32 %5819, i32 %5577)
  %5932 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5931, <4 x i32> %5897, <4 x i32> %5538, i32 %5819, i32 %5577)
  %5933 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5730, <4 x i32> %5899, <4 x i32> %5537, i32 %5824, i32 %5577)
  %5934 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5933, <4 x i32> %5901, <4 x i32> %5538, i32 %5824, i32 %5577)
  %5935 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5732, <4 x i32> %5903, <4 x i32> %5537, i32 %5824, i32 %5577)
  %5936 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5935, <4 x i32> %5905, <4 x i32> %5538, i32 %5824, i32 %5577)
  %5937 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5734, <4 x i32> %5907, <4 x i32> %5537, i32 %5829, i32 %5577)
  %5938 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5937, <4 x i32> %5909, <4 x i32> %5538, i32 %5829, i32 %5577)
  %5939 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5736, <4 x i32> %5911, <4 x i32> %5537, i32 %5829, i32 %5577)
  %5940 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5939, <4 x i32> %5913, <4 x i32> %5538, i32 %5829, i32 %5577)
  %5941 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5738, <4 x i32> %5915, <4 x i32> %5537, i32 %5834, i32 %5577)
  %5942 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5941, <4 x i32> %5917, <4 x i32> %5538, i32 %5834, i32 %5577)
  %5943 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5740, <4 x i32> %5919, <4 x i32> %5537, i32 %5834, i32 %5577)
  %5944 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5943, <4 x i32> %5921, <4 x i32> %5538, i32 %5834, i32 %5577)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5945 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5924, i32 %180, i32 2)
  %5946 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5927, i32 %180, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5947 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5744, <4 x i32> %5891, <4 x i32> %5555, i32 %5819, i32 %5578)
  %5948 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5947, <4 x i32> %5893, <4 x i32> %5556, i32 %5819, i32 %5578)
  %5949 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5746, <4 x i32> %5895, <4 x i32> %5555, i32 %5819, i32 %5578)
  %5950 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5949, <4 x i32> %5897, <4 x i32> %5556, i32 %5819, i32 %5578)
  %5951 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5748, <4 x i32> %5899, <4 x i32> %5555, i32 %5824, i32 %5578)
  %5952 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5951, <4 x i32> %5901, <4 x i32> %5556, i32 %5824, i32 %5578)
  %5953 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5750, <4 x i32> %5903, <4 x i32> %5555, i32 %5824, i32 %5578)
  %5954 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5953, <4 x i32> %5905, <4 x i32> %5556, i32 %5824, i32 %5578)
  %5955 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5752, <4 x i32> %5907, <4 x i32> %5555, i32 %5829, i32 %5578)
  %5956 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5955, <4 x i32> %5909, <4 x i32> %5556, i32 %5829, i32 %5578)
  %5957 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5754, <4 x i32> %5911, <4 x i32> %5555, i32 %5829, i32 %5578)
  %5958 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5957, <4 x i32> %5913, <4 x i32> %5556, i32 %5829, i32 %5578)
  %5959 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5756, <4 x i32> %5915, <4 x i32> %5555, i32 %5834, i32 %5578)
  %5960 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5959, <4 x i32> %5917, <4 x i32> %5556, i32 %5834, i32 %5578)
  %5961 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5758, <4 x i32> %5919, <4 x i32> %5555, i32 %5834, i32 %5578)
  %5962 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5961, <4 x i32> %5921, <4 x i32> %5556, i32 %5834, i32 %5578)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5963 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5924, i32 %184, i32 2)
  %5964 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5927, i32 %184, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5965 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5762, <4 x i32> %5891, <4 x i32> %5573, i32 %5819, i32 %5578)
  %5966 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5965, <4 x i32> %5893, <4 x i32> %5574, i32 %5819, i32 %5578)
  %5967 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5764, <4 x i32> %5895, <4 x i32> %5573, i32 %5819, i32 %5578)
  %5968 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5967, <4 x i32> %5897, <4 x i32> %5574, i32 %5819, i32 %5578)
  %5969 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5766, <4 x i32> %5899, <4 x i32> %5573, i32 %5824, i32 %5578)
  %5970 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5969, <4 x i32> %5901, <4 x i32> %5574, i32 %5824, i32 %5578)
  %5971 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5768, <4 x i32> %5903, <4 x i32> %5573, i32 %5824, i32 %5578)
  %5972 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5971, <4 x i32> %5905, <4 x i32> %5574, i32 %5824, i32 %5578)
  %5973 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5770, <4 x i32> %5907, <4 x i32> %5573, i32 %5829, i32 %5578)
  %5974 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5973, <4 x i32> %5909, <4 x i32> %5574, i32 %5829, i32 %5578)
  %5975 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5772, <4 x i32> %5911, <4 x i32> %5573, i32 %5829, i32 %5578)
  %5976 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5975, <4 x i32> %5913, <4 x i32> %5574, i32 %5829, i32 %5578)
  %5977 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5774, <4 x i32> %5915, <4 x i32> %5573, i32 %5834, i32 %5578)
  %5978 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5977, <4 x i32> %5917, <4 x i32> %5574, i32 %5834, i32 %5578)
  %5979 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5776, <4 x i32> %5919, <4 x i32> %5573, i32 %5834, i32 %5578)
  %5980 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5979, <4 x i32> %5921, <4 x i32> %5574, i32 %5834, i32 %5578)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5981 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5924, i32 %188, i32 2)
  %5982 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %35, i32 %5927, i32 %188, i32 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %5983 = add i32 %211, 1728
  %5984 = mul i32 %5983, 4
  %5985 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %5984, i32 %204, i32 0)
  %5986 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %63, i32 %5984, i32 %210, i32 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %5987 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1075
  %5988 = load <16 x i8>, ptr addrspace(3) %5987, align 1
  %5989 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1078
  %5990 = load <16 x i8>, ptr addrspace(3) %5989, align 1
  %5991 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1081
  %5992 = load <16 x i8>, ptr addrspace(3) %5991, align 1
  %5993 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1084
  %5994 = load <16 x i8>, ptr addrspace(3) %5993, align 1
  %5995 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1087
  %5996 = load <16 x i8>, ptr addrspace(3) %5995, align 1
  %5997 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1090
  %5998 = load <16 x i8>, ptr addrspace(3) %5997, align 1
  %5999 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1093
  %6000 = load <16 x i8>, ptr addrspace(3) %5999, align 1
  %6001 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1096
  %6002 = load <16 x i8>, ptr addrspace(3) %6001, align 1
  %6003 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1099
  %6004 = load <16 x i8>, ptr addrspace(3) %6003, align 1
  %6005 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1102
  %6006 = load <16 x i8>, ptr addrspace(3) %6005, align 1
  %6007 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1105
  %6008 = load <16 x i8>, ptr addrspace(3) %6007, align 1
  %6009 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1108
  %6010 = load <16 x i8>, ptr addrspace(3) %6009, align 1
  %6011 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1111
  %6012 = load <16 x i8>, ptr addrspace(3) %6011, align 1
  %6013 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1114
  %6014 = load <16 x i8>, ptr addrspace(3) %6013, align 1
  %6015 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1117
  %6016 = load <16 x i8>, ptr addrspace(3) %6015, align 1
  %6017 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %1120
  %6018 = load <16 x i8>, ptr addrspace(3) %6017, align 1
  %6019 = add i64 %193, 1664
  %6020 = add i64 %6019, %24
  %6021 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %6020
  %6022 = load <1 x i32>, ptr addrspace(3) %6021, align 4
  %6023 = extractelement <1 x i32> %6022, i64 0
  %6024 = add i64 %193, 3456
  %6025 = add i64 %6024, %24
  %6026 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %6025
  %6027 = load <1 x i32>, ptr addrspace(3) %6026, align 4
  %6028 = extractelement <1 x i32> %6027, i64 0
  %6029 = add i64 %193, 5248
  %6030 = add i64 %6029, %24
  %6031 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %6030
  %6032 = load <1 x i32>, ptr addrspace(3) %6031, align 4
  %6033 = extractelement <1 x i32> %6032, i64 0
  %6034 = add i64 %193, 7040
  %6035 = add i64 %6034, %24
  %6036 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %6035
  %6037 = load <1 x i32>, ptr addrspace(3) %6036, align 4
  %6038 = extractelement <1 x i32> %6037, i64 0
  %6039 = bitcast <16 x i8> %5988 to <4 x i32>
  %6040 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5894, <4 x i32> %6039, <4 x i32> %5721, i32 %6023, i32 %5781)
  %6041 = bitcast <16 x i8> %5990 to <4 x i32>
  %6042 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6040, <4 x i32> %6041, <4 x i32> %5724, i32 %6023, i32 %5781)
  %6043 = bitcast <16 x i8> %5992 to <4 x i32>
  %6044 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5898, <4 x i32> %6043, <4 x i32> %5721, i32 %6023, i32 %5781)
  %6045 = bitcast <16 x i8> %5994 to <4 x i32>
  %6046 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6044, <4 x i32> %6045, <4 x i32> %5724, i32 %6023, i32 %5781)
  %6047 = bitcast <16 x i8> %5996 to <4 x i32>
  %6048 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5902, <4 x i32> %6047, <4 x i32> %5721, i32 %6028, i32 %5781)
  %6049 = bitcast <16 x i8> %5998 to <4 x i32>
  %6050 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6048, <4 x i32> %6049, <4 x i32> %5724, i32 %6028, i32 %5781)
  %6051 = bitcast <16 x i8> %6000 to <4 x i32>
  %6052 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5906, <4 x i32> %6051, <4 x i32> %5721, i32 %6028, i32 %5781)
  %6053 = bitcast <16 x i8> %6002 to <4 x i32>
  %6054 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6052, <4 x i32> %6053, <4 x i32> %5724, i32 %6028, i32 %5781)
  %6055 = bitcast <16 x i8> %6004 to <4 x i32>
  %6056 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5910, <4 x i32> %6055, <4 x i32> %5721, i32 %6033, i32 %5781)
  %6057 = bitcast <16 x i8> %6006 to <4 x i32>
  %6058 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6056, <4 x i32> %6057, <4 x i32> %5724, i32 %6033, i32 %5781)
  %6059 = bitcast <16 x i8> %6008 to <4 x i32>
  %6060 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5914, <4 x i32> %6059, <4 x i32> %5721, i32 %6033, i32 %5781)
  %6061 = bitcast <16 x i8> %6010 to <4 x i32>
  %6062 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6060, <4 x i32> %6061, <4 x i32> %5724, i32 %6033, i32 %5781)
  %6063 = bitcast <16 x i8> %6012 to <4 x i32>
  %6064 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5918, <4 x i32> %6063, <4 x i32> %5721, i32 %6038, i32 %5781)
  %6065 = bitcast <16 x i8> %6014 to <4 x i32>
  %6066 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6064, <4 x i32> %6065, <4 x i32> %5724, i32 %6038, i32 %5781)
  %6067 = bitcast <16 x i8> %6016 to <4 x i32>
  %6068 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5922, <4 x i32> %6067, <4 x i32> %5721, i32 %6038, i32 %5781)
  %6069 = bitcast <16 x i8> %6018 to <4 x i32>
  %6070 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6068, <4 x i32> %6069, <4 x i32> %5724, i32 %6038, i32 %5781)
  %6071 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5930, <4 x i32> %6039, <4 x i32> %5741, i32 %6023, i32 %5781)
  %6072 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6071, <4 x i32> %6041, <4 x i32> %5742, i32 %6023, i32 %5781)
  %6073 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5932, <4 x i32> %6043, <4 x i32> %5741, i32 %6023, i32 %5781)
  %6074 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6073, <4 x i32> %6045, <4 x i32> %5742, i32 %6023, i32 %5781)
  %6075 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5934, <4 x i32> %6047, <4 x i32> %5741, i32 %6028, i32 %5781)
  %6076 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6075, <4 x i32> %6049, <4 x i32> %5742, i32 %6028, i32 %5781)
  %6077 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5936, <4 x i32> %6051, <4 x i32> %5741, i32 %6028, i32 %5781)
  %6078 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6077, <4 x i32> %6053, <4 x i32> %5742, i32 %6028, i32 %5781)
  %6079 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5938, <4 x i32> %6055, <4 x i32> %5741, i32 %6033, i32 %5781)
  %6080 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6079, <4 x i32> %6057, <4 x i32> %5742, i32 %6033, i32 %5781)
  %6081 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5940, <4 x i32> %6059, <4 x i32> %5741, i32 %6033, i32 %5781)
  %6082 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6081, <4 x i32> %6061, <4 x i32> %5742, i32 %6033, i32 %5781)
  %6083 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5942, <4 x i32> %6063, <4 x i32> %5741, i32 %6038, i32 %5781)
  %6084 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6083, <4 x i32> %6065, <4 x i32> %5742, i32 %6038, i32 %5781)
  %6085 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5944, <4 x i32> %6067, <4 x i32> %5741, i32 %6038, i32 %5781)
  %6086 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6085, <4 x i32> %6069, <4 x i32> %5742, i32 %6038, i32 %5781)
  %6087 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5948, <4 x i32> %6039, <4 x i32> %5759, i32 %6023, i32 %5782)
  %6088 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6087, <4 x i32> %6041, <4 x i32> %5760, i32 %6023, i32 %5782)
  %6089 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5950, <4 x i32> %6043, <4 x i32> %5759, i32 %6023, i32 %5782)
  %6090 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6089, <4 x i32> %6045, <4 x i32> %5760, i32 %6023, i32 %5782)
  %6091 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5952, <4 x i32> %6047, <4 x i32> %5759, i32 %6028, i32 %5782)
  %6092 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6091, <4 x i32> %6049, <4 x i32> %5760, i32 %6028, i32 %5782)
  %6093 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5954, <4 x i32> %6051, <4 x i32> %5759, i32 %6028, i32 %5782)
  %6094 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6093, <4 x i32> %6053, <4 x i32> %5760, i32 %6028, i32 %5782)
  %6095 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5956, <4 x i32> %6055, <4 x i32> %5759, i32 %6033, i32 %5782)
  %6096 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6095, <4 x i32> %6057, <4 x i32> %5760, i32 %6033, i32 %5782)
  %6097 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5958, <4 x i32> %6059, <4 x i32> %5759, i32 %6033, i32 %5782)
  %6098 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6097, <4 x i32> %6061, <4 x i32> %5760, i32 %6033, i32 %5782)
  %6099 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5960, <4 x i32> %6063, <4 x i32> %5759, i32 %6038, i32 %5782)
  %6100 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6099, <4 x i32> %6065, <4 x i32> %5760, i32 %6038, i32 %5782)
  %6101 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5962, <4 x i32> %6067, <4 x i32> %5759, i32 %6038, i32 %5782)
  %6102 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6101, <4 x i32> %6069, <4 x i32> %5760, i32 %6038, i32 %5782)
  %6103 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5966, <4 x i32> %6039, <4 x i32> %5777, i32 %6023, i32 %5782)
  %6104 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6103, <4 x i32> %6041, <4 x i32> %5778, i32 %6023, i32 %5782)
  %6105 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5968, <4 x i32> %6043, <4 x i32> %5777, i32 %6023, i32 %5782)
  %6106 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6105, <4 x i32> %6045, <4 x i32> %5778, i32 %6023, i32 %5782)
  %6107 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5970, <4 x i32> %6047, <4 x i32> %5777, i32 %6028, i32 %5782)
  %6108 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6107, <4 x i32> %6049, <4 x i32> %5778, i32 %6028, i32 %5782)
  %6109 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5972, <4 x i32> %6051, <4 x i32> %5777, i32 %6028, i32 %5782)
  %6110 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6109, <4 x i32> %6053, <4 x i32> %5778, i32 %6028, i32 %5782)
  %6111 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5974, <4 x i32> %6055, <4 x i32> %5777, i32 %6033, i32 %5782)
  %6112 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6111, <4 x i32> %6057, <4 x i32> %5778, i32 %6033, i32 %5782)
  %6113 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5976, <4 x i32> %6059, <4 x i32> %5777, i32 %6033, i32 %5782)
  %6114 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6113, <4 x i32> %6061, <4 x i32> %5778, i32 %6033, i32 %5782)
  %6115 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5978, <4 x i32> %6063, <4 x i32> %5777, i32 %6038, i32 %5782)
  %6116 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6115, <4 x i32> %6065, <4 x i32> %5778, i32 %6038, i32 %5782)
  %6117 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %5980, <4 x i32> %6067, <4 x i32> %5777, i32 %6038, i32 %5782)
  %6118 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6117, <4 x i32> %6069, <4 x i32> %5778, i32 %6038, i32 %5782)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %6119 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %522
  %6120 = load <16 x i8>, ptr addrspace(3) %6119, align 1
  %6121 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %527
  %6122 = load <16 x i8>, ptr addrspace(3) %6121, align 1
  %6123 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %535
  %6124 = load <16 x i8>, ptr addrspace(3) %6123, align 1
  %6125 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %539
  %6126 = load <16 x i8>, ptr addrspace(3) %6125, align 1
  %6127 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %547
  %6128 = load <16 x i8>, ptr addrspace(3) %6127, align 1
  %6129 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %551
  %6130 = load <16 x i8>, ptr addrspace(3) %6129, align 1
  %6131 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %559
  %6132 = load <16 x i8>, ptr addrspace(3) %6131, align 1
  %6133 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %563
  %6134 = load <16 x i8>, ptr addrspace(3) %6133, align 1
  %6135 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %571
  %6136 = load <16 x i8>, ptr addrspace(3) %6135, align 1
  %6137 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %575
  %6138 = load <16 x i8>, ptr addrspace(3) %6137, align 1
  %6139 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %583
  %6140 = load <16 x i8>, ptr addrspace(3) %6139, align 1
  %6141 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %587
  %6142 = load <16 x i8>, ptr addrspace(3) %6141, align 1
  %6143 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %595
  %6144 = load <16 x i8>, ptr addrspace(3) %6143, align 1
  %6145 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %599
  %6146 = load <16 x i8>, ptr addrspace(3) %6145, align 1
  %6147 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %607
  %6148 = load <16 x i8>, ptr addrspace(3) %6147, align 1
  %6149 = getelementptr i8, ptr addrspace(3) @smem_g1, i64 %611
  %6150 = load <16 x i8>, ptr addrspace(3) %6149, align 1
  %6151 = add i64 %193, 1728
  %6152 = add i64 %6151, %24
  %6153 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %6152
  %6154 = load <1 x i32>, ptr addrspace(3) %6153, align 4
  %6155 = extractelement <1 x i32> %6154, i64 0
  %6156 = add i64 %193, 3520
  %6157 = add i64 %6156, %24
  %6158 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %6157
  %6159 = load <1 x i32>, ptr addrspace(3) %6158, align 4
  %6160 = extractelement <1 x i32> %6159, i64 0
  %6161 = add i64 %193, 5312
  %6162 = add i64 %6161, %24
  %6163 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %6162
  %6164 = load <1 x i32>, ptr addrspace(3) %6163, align 4
  %6165 = extractelement <1 x i32> %6164, i64 0
  %6166 = add i64 %193, 7104
  %6167 = add i64 %6166, %24
  %6168 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem_g1, i32 98304), i64 %6167
  %6169 = load <1 x i32>, ptr addrspace(3) %6168, align 4
  %6170 = extractelement <1 x i32> %6169, i64 0
  %6171 = bitcast <16 x i8> %6120 to <4 x i32>
  %6172 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6042, <4 x i32> %6171, <4 x i32> %5925, i32 %6155, i32 %5985)
  %6173 = bitcast <16 x i8> %6122 to <4 x i32>
  %6174 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6172, <4 x i32> %6173, <4 x i32> %5928, i32 %6155, i32 %5985)
  %6175 = bitcast <16 x i8> %6124 to <4 x i32>
  %6176 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6046, <4 x i32> %6175, <4 x i32> %5925, i32 %6155, i32 %5985)
  %6177 = bitcast <16 x i8> %6126 to <4 x i32>
  %6178 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6176, <4 x i32> %6177, <4 x i32> %5928, i32 %6155, i32 %5985)
  %6179 = bitcast <16 x i8> %6128 to <4 x i32>
  %6180 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6050, <4 x i32> %6179, <4 x i32> %5925, i32 %6160, i32 %5985)
  %6181 = bitcast <16 x i8> %6130 to <4 x i32>
  %6182 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6180, <4 x i32> %6181, <4 x i32> %5928, i32 %6160, i32 %5985)
  %6183 = bitcast <16 x i8> %6132 to <4 x i32>
  %6184 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6054, <4 x i32> %6183, <4 x i32> %5925, i32 %6160, i32 %5985)
  %6185 = bitcast <16 x i8> %6134 to <4 x i32>
  %6186 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6184, <4 x i32> %6185, <4 x i32> %5928, i32 %6160, i32 %5985)
  %6187 = bitcast <16 x i8> %6136 to <4 x i32>
  %6188 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6058, <4 x i32> %6187, <4 x i32> %5925, i32 %6165, i32 %5985)
  %6189 = bitcast <16 x i8> %6138 to <4 x i32>
  %6190 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6188, <4 x i32> %6189, <4 x i32> %5928, i32 %6165, i32 %5985)
  %6191 = bitcast <16 x i8> %6140 to <4 x i32>
  %6192 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6062, <4 x i32> %6191, <4 x i32> %5925, i32 %6165, i32 %5985)
  %6193 = bitcast <16 x i8> %6142 to <4 x i32>
  %6194 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6192, <4 x i32> %6193, <4 x i32> %5928, i32 %6165, i32 %5985)
  %6195 = bitcast <16 x i8> %6144 to <4 x i32>
  %6196 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6066, <4 x i32> %6195, <4 x i32> %5925, i32 %6170, i32 %5985)
  %6197 = bitcast <16 x i8> %6146 to <4 x i32>
  %6198 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6196, <4 x i32> %6197, <4 x i32> %5928, i32 %6170, i32 %5985)
  %6199 = bitcast <16 x i8> %6148 to <4 x i32>
  %6200 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6070, <4 x i32> %6199, <4 x i32> %5925, i32 %6170, i32 %5985)
  %6201 = bitcast <16 x i8> %6150 to <4 x i32>
  %6202 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6200, <4 x i32> %6201, <4 x i32> %5928, i32 %6170, i32 %5985)
  %6203 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6072, <4 x i32> %6171, <4 x i32> %5945, i32 %6155, i32 %5985)
  %6204 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6203, <4 x i32> %6173, <4 x i32> %5946, i32 %6155, i32 %5985)
  %6205 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6074, <4 x i32> %6175, <4 x i32> %5945, i32 %6155, i32 %5985)
  %6206 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6205, <4 x i32> %6177, <4 x i32> %5946, i32 %6155, i32 %5985)
  %6207 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6076, <4 x i32> %6179, <4 x i32> %5945, i32 %6160, i32 %5985)
  %6208 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6207, <4 x i32> %6181, <4 x i32> %5946, i32 %6160, i32 %5985)
  %6209 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6078, <4 x i32> %6183, <4 x i32> %5945, i32 %6160, i32 %5985)
  %6210 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6209, <4 x i32> %6185, <4 x i32> %5946, i32 %6160, i32 %5985)
  %6211 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6080, <4 x i32> %6187, <4 x i32> %5945, i32 %6165, i32 %5985)
  %6212 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6211, <4 x i32> %6189, <4 x i32> %5946, i32 %6165, i32 %5985)
  %6213 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6082, <4 x i32> %6191, <4 x i32> %5945, i32 %6165, i32 %5985)
  %6214 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6213, <4 x i32> %6193, <4 x i32> %5946, i32 %6165, i32 %5985)
  %6215 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6084, <4 x i32> %6195, <4 x i32> %5945, i32 %6170, i32 %5985)
  %6216 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6215, <4 x i32> %6197, <4 x i32> %5946, i32 %6170, i32 %5985)
  %6217 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6086, <4 x i32> %6199, <4 x i32> %5945, i32 %6170, i32 %5985)
  %6218 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6217, <4 x i32> %6201, <4 x i32> %5946, i32 %6170, i32 %5985)
  %6219 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6088, <4 x i32> %6171, <4 x i32> %5963, i32 %6155, i32 %5986)
  %6220 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6219, <4 x i32> %6173, <4 x i32> %5964, i32 %6155, i32 %5986)
  %6221 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6090, <4 x i32> %6175, <4 x i32> %5963, i32 %6155, i32 %5986)
  %6222 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6221, <4 x i32> %6177, <4 x i32> %5964, i32 %6155, i32 %5986)
  %6223 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6092, <4 x i32> %6179, <4 x i32> %5963, i32 %6160, i32 %5986)
  %6224 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6223, <4 x i32> %6181, <4 x i32> %5964, i32 %6160, i32 %5986)
  %6225 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6094, <4 x i32> %6183, <4 x i32> %5963, i32 %6160, i32 %5986)
  %6226 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6225, <4 x i32> %6185, <4 x i32> %5964, i32 %6160, i32 %5986)
  %6227 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6096, <4 x i32> %6187, <4 x i32> %5963, i32 %6165, i32 %5986)
  %6228 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6227, <4 x i32> %6189, <4 x i32> %5964, i32 %6165, i32 %5986)
  %6229 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6098, <4 x i32> %6191, <4 x i32> %5963, i32 %6165, i32 %5986)
  %6230 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6229, <4 x i32> %6193, <4 x i32> %5964, i32 %6165, i32 %5986)
  %6231 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6100, <4 x i32> %6195, <4 x i32> %5963, i32 %6170, i32 %5986)
  %6232 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6231, <4 x i32> %6197, <4 x i32> %5964, i32 %6170, i32 %5986)
  %6233 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6102, <4 x i32> %6199, <4 x i32> %5963, i32 %6170, i32 %5986)
  %6234 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6233, <4 x i32> %6201, <4 x i32> %5964, i32 %6170, i32 %5986)
  %6235 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6104, <4 x i32> %6171, <4 x i32> %5981, i32 %6155, i32 %5986)
  %6236 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6235, <4 x i32> %6173, <4 x i32> %5982, i32 %6155, i32 %5986)
  %6237 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6106, <4 x i32> %6175, <4 x i32> %5981, i32 %6155, i32 %5986)
  %6238 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6237, <4 x i32> %6177, <4 x i32> %5982, i32 %6155, i32 %5986)
  %6239 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6108, <4 x i32> %6179, <4 x i32> %5981, i32 %6160, i32 %5986)
  %6240 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6239, <4 x i32> %6181, <4 x i32> %5982, i32 %6160, i32 %5986)
  %6241 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6110, <4 x i32> %6183, <4 x i32> %5981, i32 %6160, i32 %5986)
  %6242 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6241, <4 x i32> %6185, <4 x i32> %5982, i32 %6160, i32 %5986)
  %6243 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6112, <4 x i32> %6187, <4 x i32> %5981, i32 %6165, i32 %5986)
  %6244 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6243, <4 x i32> %6189, <4 x i32> %5982, i32 %6165, i32 %5986)
  %6245 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6114, <4 x i32> %6191, <4 x i32> %5981, i32 %6165, i32 %5986)
  %6246 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6245, <4 x i32> %6193, <4 x i32> %5982, i32 %6165, i32 %5986)
  %6247 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6116, <4 x i32> %6195, <4 x i32> %5981, i32 %6170, i32 %5986)
  %6248 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6247, <4 x i32> %6197, <4 x i32> %5982, i32 %6170, i32 %5986)
  %6249 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6118, <4 x i32> %6199, <4 x i32> %5981, i32 %6170, i32 %5986)
  %6250 = call <4 x float> asm sideeffect "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4", "=a,0,v,v,v,v"(<4 x float> %6249, <4 x i32> %6201, <4 x i32> %5982, i32 %6170, i32 %5986)
  %6251 = mul i64 %54, 512
  %6252 = trunc i64 %6251 to i32
  %6253 = ptrtoint ptr addrspace(1) %8 to i64
  %6254 = inttoptr i64 %6253 to ptr
  %6255 = sext i32 %6252 to i64
  %6256 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %6254, i16 0, i64 %6255, i32 159744)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %6257 = mul i64 %23, 4
  %6258 = mul i64 %20, 32
  %6259 = add i64 %6258, %24
  %6260 = mul i64 %23, 1024
  %6261 = add i64 %6260, %6259
  %6262 = extractelement <4 x float> %6174, i64 0
  %6263 = insertelement <1 x float> poison, float %6262, i32 0
  %6264 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6261
  store <1 x float> %6263, ptr addrspace(3) %6264, align 4
  %6265 = add i64 %6257, 1
  %6266 = mul i64 %6265, 256
  %6267 = add i64 %6266, %6259
  %6268 = extractelement <4 x float> %6174, i64 1
  %6269 = insertelement <1 x float> poison, float %6268, i32 0
  %6270 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6267
  store <1 x float> %6269, ptr addrspace(3) %6270, align 4
  %6271 = add i64 %6257, 2
  %6272 = mul i64 %6271, 256
  %6273 = add i64 %6272, %6259
  %6274 = extractelement <4 x float> %6174, i64 2
  %6275 = insertelement <1 x float> poison, float %6274, i32 0
  %6276 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6273
  store <1 x float> %6275, ptr addrspace(3) %6276, align 4
  %6277 = add i64 %6257, 3
  %6278 = mul i64 %6277, 256
  %6279 = add i64 %6278, %6259
  %6280 = extractelement <4 x float> %6174, i64 3
  %6281 = insertelement <1 x float> poison, float %6280, i32 0
  %6282 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6279
  store <1 x float> %6281, ptr addrspace(3) %6282, align 4
  %6283 = add i64 %6259, 128
  %6284 = add i64 %6260, %6283
  %6285 = extractelement <4 x float> %6204, i64 0
  %6286 = insertelement <1 x float> poison, float %6285, i32 0
  %6287 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6284
  store <1 x float> %6286, ptr addrspace(3) %6287, align 4
  %6288 = add i64 %6266, %6283
  %6289 = extractelement <4 x float> %6204, i64 1
  %6290 = insertelement <1 x float> poison, float %6289, i32 0
  %6291 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6288
  store <1 x float> %6290, ptr addrspace(3) %6291, align 4
  %6292 = add i64 %6272, %6283
  %6293 = extractelement <4 x float> %6204, i64 2
  %6294 = insertelement <1 x float> poison, float %6293, i32 0
  %6295 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6292
  store <1 x float> %6294, ptr addrspace(3) %6295, align 4
  %6296 = add i64 %6278, %6283
  %6297 = extractelement <4 x float> %6204, i64 3
  %6298 = insertelement <1 x float> poison, float %6297, i32 0
  %6299 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6296
  store <1 x float> %6298, ptr addrspace(3) %6299, align 4
  %6300 = add i64 %6258, 16
  %6301 = add i64 %6300, %24
  %6302 = add i64 %6260, %6301
  %6303 = extractelement <4 x float> %6220, i64 0
  %6304 = insertelement <1 x float> poison, float %6303, i32 0
  %6305 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6302
  store <1 x float> %6304, ptr addrspace(3) %6305, align 4
  %6306 = add i64 %6266, %6301
  %6307 = extractelement <4 x float> %6220, i64 1
  %6308 = insertelement <1 x float> poison, float %6307, i32 0
  %6309 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6306
  store <1 x float> %6308, ptr addrspace(3) %6309, align 4
  %6310 = add i64 %6272, %6301
  %6311 = extractelement <4 x float> %6220, i64 2
  %6312 = insertelement <1 x float> poison, float %6311, i32 0
  %6313 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6310
  store <1 x float> %6312, ptr addrspace(3) %6313, align 4
  %6314 = add i64 %6278, %6301
  %6315 = extractelement <4 x float> %6220, i64 3
  %6316 = insertelement <1 x float> poison, float %6315, i32 0
  %6317 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6314
  store <1 x float> %6316, ptr addrspace(3) %6317, align 4
  %6318 = add i64 %6301, 128
  %6319 = add i64 %6260, %6318
  %6320 = extractelement <4 x float> %6236, i64 0
  %6321 = insertelement <1 x float> poison, float %6320, i32 0
  %6322 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6319
  store <1 x float> %6321, ptr addrspace(3) %6322, align 4
  %6323 = add i64 %6266, %6318
  %6324 = extractelement <4 x float> %6236, i64 1
  %6325 = insertelement <1 x float> poison, float %6324, i32 0
  %6326 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6323
  store <1 x float> %6325, ptr addrspace(3) %6326, align 4
  %6327 = add i64 %6272, %6318
  %6328 = extractelement <4 x float> %6236, i64 2
  %6329 = insertelement <1 x float> poison, float %6328, i32 0
  %6330 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6327
  store <1 x float> %6329, ptr addrspace(3) %6330, align 4
  %6331 = add i64 %6278, %6318
  %6332 = extractelement <4 x float> %6236, i64 3
  %6333 = insertelement <1 x float> poison, float %6332, i32 0
  %6334 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6331
  store <1 x float> %6333, ptr addrspace(3) %6334, align 4
  %6335 = add i64 %6257, 16
  %6336 = mul i64 %6335, 256
  %6337 = add i64 %6336, %6259
  %6338 = extractelement <4 x float> %6178, i64 0
  %6339 = insertelement <1 x float> poison, float %6338, i32 0
  %6340 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6337
  store <1 x float> %6339, ptr addrspace(3) %6340, align 4
  %6341 = add i64 %6257, 17
  %6342 = mul i64 %6341, 256
  %6343 = add i64 %6342, %6259
  %6344 = extractelement <4 x float> %6178, i64 1
  %6345 = insertelement <1 x float> poison, float %6344, i32 0
  %6346 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6343
  store <1 x float> %6345, ptr addrspace(3) %6346, align 4
  %6347 = add i64 %6257, 18
  %6348 = mul i64 %6347, 256
  %6349 = add i64 %6348, %6259
  %6350 = extractelement <4 x float> %6178, i64 2
  %6351 = insertelement <1 x float> poison, float %6350, i32 0
  %6352 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6349
  store <1 x float> %6351, ptr addrspace(3) %6352, align 4
  %6353 = add i64 %6257, 19
  %6354 = mul i64 %6353, 256
  %6355 = add i64 %6354, %6259
  %6356 = extractelement <4 x float> %6178, i64 3
  %6357 = insertelement <1 x float> poison, float %6356, i32 0
  %6358 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6355
  store <1 x float> %6357, ptr addrspace(3) %6358, align 4
  %6359 = add i64 %6336, %6283
  %6360 = extractelement <4 x float> %6206, i64 0
  %6361 = insertelement <1 x float> poison, float %6360, i32 0
  %6362 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6359
  store <1 x float> %6361, ptr addrspace(3) %6362, align 4
  %6363 = add i64 %6342, %6283
  %6364 = extractelement <4 x float> %6206, i64 1
  %6365 = insertelement <1 x float> poison, float %6364, i32 0
  %6366 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6363
  store <1 x float> %6365, ptr addrspace(3) %6366, align 4
  %6367 = add i64 %6348, %6283
  %6368 = extractelement <4 x float> %6206, i64 2
  %6369 = insertelement <1 x float> poison, float %6368, i32 0
  %6370 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6367
  store <1 x float> %6369, ptr addrspace(3) %6370, align 4
  %6371 = add i64 %6354, %6283
  %6372 = extractelement <4 x float> %6206, i64 3
  %6373 = insertelement <1 x float> poison, float %6372, i32 0
  %6374 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6371
  store <1 x float> %6373, ptr addrspace(3) %6374, align 4
  %6375 = add i64 %6336, %6301
  %6376 = extractelement <4 x float> %6222, i64 0
  %6377 = insertelement <1 x float> poison, float %6376, i32 0
  %6378 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6375
  store <1 x float> %6377, ptr addrspace(3) %6378, align 4
  %6379 = add i64 %6342, %6301
  %6380 = extractelement <4 x float> %6222, i64 1
  %6381 = insertelement <1 x float> poison, float %6380, i32 0
  %6382 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6379
  store <1 x float> %6381, ptr addrspace(3) %6382, align 4
  %6383 = add i64 %6348, %6301
  %6384 = extractelement <4 x float> %6222, i64 2
  %6385 = insertelement <1 x float> poison, float %6384, i32 0
  %6386 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6383
  store <1 x float> %6385, ptr addrspace(3) %6386, align 4
  %6387 = add i64 %6354, %6301
  %6388 = extractelement <4 x float> %6222, i64 3
  %6389 = insertelement <1 x float> poison, float %6388, i32 0
  %6390 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6387
  store <1 x float> %6389, ptr addrspace(3) %6390, align 4
  %6391 = add i64 %6336, %6318
  %6392 = extractelement <4 x float> %6238, i64 0
  %6393 = insertelement <1 x float> poison, float %6392, i32 0
  %6394 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6391
  store <1 x float> %6393, ptr addrspace(3) %6394, align 4
  %6395 = add i64 %6342, %6318
  %6396 = extractelement <4 x float> %6238, i64 1
  %6397 = insertelement <1 x float> poison, float %6396, i32 0
  %6398 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6395
  store <1 x float> %6397, ptr addrspace(3) %6398, align 4
  %6399 = add i64 %6348, %6318
  %6400 = extractelement <4 x float> %6238, i64 2
  %6401 = insertelement <1 x float> poison, float %6400, i32 0
  %6402 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6399
  store <1 x float> %6401, ptr addrspace(3) %6402, align 4
  %6403 = add i64 %6354, %6318
  %6404 = extractelement <4 x float> %6238, i64 3
  %6405 = insertelement <1 x float> poison, float %6404, i32 0
  %6406 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6403
  store <1 x float> %6405, ptr addrspace(3) %6406, align 4
  %6407 = add i64 %6257, 32
  %6408 = mul i64 %6407, 256
  %6409 = add i64 %6408, %6259
  %6410 = extractelement <4 x float> %6182, i64 0
  %6411 = insertelement <1 x float> poison, float %6410, i32 0
  %6412 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6409
  store <1 x float> %6411, ptr addrspace(3) %6412, align 4
  %6413 = add i64 %6257, 33
  %6414 = mul i64 %6413, 256
  %6415 = add i64 %6414, %6259
  %6416 = extractelement <4 x float> %6182, i64 1
  %6417 = insertelement <1 x float> poison, float %6416, i32 0
  %6418 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6415
  store <1 x float> %6417, ptr addrspace(3) %6418, align 4
  %6419 = add i64 %6257, 34
  %6420 = mul i64 %6419, 256
  %6421 = add i64 %6420, %6259
  %6422 = extractelement <4 x float> %6182, i64 2
  %6423 = insertelement <1 x float> poison, float %6422, i32 0
  %6424 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6421
  store <1 x float> %6423, ptr addrspace(3) %6424, align 4
  %6425 = add i64 %6257, 35
  %6426 = mul i64 %6425, 256
  %6427 = add i64 %6426, %6259
  %6428 = extractelement <4 x float> %6182, i64 3
  %6429 = insertelement <1 x float> poison, float %6428, i32 0
  %6430 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6427
  store <1 x float> %6429, ptr addrspace(3) %6430, align 4
  %6431 = add i64 %6408, %6283
  %6432 = extractelement <4 x float> %6208, i64 0
  %6433 = insertelement <1 x float> poison, float %6432, i32 0
  %6434 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6431
  store <1 x float> %6433, ptr addrspace(3) %6434, align 4
  %6435 = add i64 %6414, %6283
  %6436 = extractelement <4 x float> %6208, i64 1
  %6437 = insertelement <1 x float> poison, float %6436, i32 0
  %6438 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6435
  store <1 x float> %6437, ptr addrspace(3) %6438, align 4
  %6439 = add i64 %6420, %6283
  %6440 = extractelement <4 x float> %6208, i64 2
  %6441 = insertelement <1 x float> poison, float %6440, i32 0
  %6442 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6439
  store <1 x float> %6441, ptr addrspace(3) %6442, align 4
  %6443 = add i64 %6426, %6283
  %6444 = extractelement <4 x float> %6208, i64 3
  %6445 = insertelement <1 x float> poison, float %6444, i32 0
  %6446 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6443
  store <1 x float> %6445, ptr addrspace(3) %6446, align 4
  %6447 = add i64 %6408, %6301
  %6448 = extractelement <4 x float> %6224, i64 0
  %6449 = insertelement <1 x float> poison, float %6448, i32 0
  %6450 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6447
  store <1 x float> %6449, ptr addrspace(3) %6450, align 4
  %6451 = add i64 %6414, %6301
  %6452 = extractelement <4 x float> %6224, i64 1
  %6453 = insertelement <1 x float> poison, float %6452, i32 0
  %6454 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6451
  store <1 x float> %6453, ptr addrspace(3) %6454, align 4
  %6455 = add i64 %6420, %6301
  %6456 = extractelement <4 x float> %6224, i64 2
  %6457 = insertelement <1 x float> poison, float %6456, i32 0
  %6458 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6455
  store <1 x float> %6457, ptr addrspace(3) %6458, align 4
  %6459 = add i64 %6426, %6301
  %6460 = extractelement <4 x float> %6224, i64 3
  %6461 = insertelement <1 x float> poison, float %6460, i32 0
  %6462 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6459
  store <1 x float> %6461, ptr addrspace(3) %6462, align 4
  %6463 = add i64 %6408, %6318
  %6464 = extractelement <4 x float> %6240, i64 0
  %6465 = insertelement <1 x float> poison, float %6464, i32 0
  %6466 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6463
  store <1 x float> %6465, ptr addrspace(3) %6466, align 4
  %6467 = add i64 %6414, %6318
  %6468 = extractelement <4 x float> %6240, i64 1
  %6469 = insertelement <1 x float> poison, float %6468, i32 0
  %6470 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6467
  store <1 x float> %6469, ptr addrspace(3) %6470, align 4
  %6471 = add i64 %6420, %6318
  %6472 = extractelement <4 x float> %6240, i64 2
  %6473 = insertelement <1 x float> poison, float %6472, i32 0
  %6474 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6471
  store <1 x float> %6473, ptr addrspace(3) %6474, align 4
  %6475 = add i64 %6426, %6318
  %6476 = extractelement <4 x float> %6240, i64 3
  %6477 = insertelement <1 x float> poison, float %6476, i32 0
  %6478 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6475
  store <1 x float> %6477, ptr addrspace(3) %6478, align 4
  %6479 = add i64 %6257, 48
  %6480 = mul i64 %6479, 256
  %6481 = add i64 %6480, %6259
  %6482 = extractelement <4 x float> %6186, i64 0
  %6483 = insertelement <1 x float> poison, float %6482, i32 0
  %6484 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6481
  store <1 x float> %6483, ptr addrspace(3) %6484, align 4
  %6485 = add i64 %6257, 49
  %6486 = mul i64 %6485, 256
  %6487 = add i64 %6486, %6259
  %6488 = extractelement <4 x float> %6186, i64 1
  %6489 = insertelement <1 x float> poison, float %6488, i32 0
  %6490 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6487
  store <1 x float> %6489, ptr addrspace(3) %6490, align 4
  %6491 = add i64 %6257, 50
  %6492 = mul i64 %6491, 256
  %6493 = add i64 %6492, %6259
  %6494 = extractelement <4 x float> %6186, i64 2
  %6495 = insertelement <1 x float> poison, float %6494, i32 0
  %6496 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6493
  store <1 x float> %6495, ptr addrspace(3) %6496, align 4
  %6497 = add i64 %6257, 51
  %6498 = mul i64 %6497, 256
  %6499 = add i64 %6498, %6259
  %6500 = extractelement <4 x float> %6186, i64 3
  %6501 = insertelement <1 x float> poison, float %6500, i32 0
  %6502 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6499
  store <1 x float> %6501, ptr addrspace(3) %6502, align 4
  %6503 = add i64 %6480, %6283
  %6504 = extractelement <4 x float> %6210, i64 0
  %6505 = insertelement <1 x float> poison, float %6504, i32 0
  %6506 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6503
  store <1 x float> %6505, ptr addrspace(3) %6506, align 4
  %6507 = add i64 %6486, %6283
  %6508 = extractelement <4 x float> %6210, i64 1
  %6509 = insertelement <1 x float> poison, float %6508, i32 0
  %6510 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6507
  store <1 x float> %6509, ptr addrspace(3) %6510, align 4
  %6511 = add i64 %6492, %6283
  %6512 = extractelement <4 x float> %6210, i64 2
  %6513 = insertelement <1 x float> poison, float %6512, i32 0
  %6514 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6511
  store <1 x float> %6513, ptr addrspace(3) %6514, align 4
  %6515 = add i64 %6498, %6283
  %6516 = extractelement <4 x float> %6210, i64 3
  %6517 = insertelement <1 x float> poison, float %6516, i32 0
  %6518 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6515
  store <1 x float> %6517, ptr addrspace(3) %6518, align 4
  %6519 = add i64 %6480, %6301
  %6520 = extractelement <4 x float> %6226, i64 0
  %6521 = insertelement <1 x float> poison, float %6520, i32 0
  %6522 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6519
  store <1 x float> %6521, ptr addrspace(3) %6522, align 4
  %6523 = add i64 %6486, %6301
  %6524 = extractelement <4 x float> %6226, i64 1
  %6525 = insertelement <1 x float> poison, float %6524, i32 0
  %6526 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6523
  store <1 x float> %6525, ptr addrspace(3) %6526, align 4
  %6527 = add i64 %6492, %6301
  %6528 = extractelement <4 x float> %6226, i64 2
  %6529 = insertelement <1 x float> poison, float %6528, i32 0
  %6530 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6527
  store <1 x float> %6529, ptr addrspace(3) %6530, align 4
  %6531 = add i64 %6498, %6301
  %6532 = extractelement <4 x float> %6226, i64 3
  %6533 = insertelement <1 x float> poison, float %6532, i32 0
  %6534 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6531
  store <1 x float> %6533, ptr addrspace(3) %6534, align 4
  %6535 = add i64 %6480, %6318
  %6536 = extractelement <4 x float> %6242, i64 0
  %6537 = insertelement <1 x float> poison, float %6536, i32 0
  %6538 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6535
  store <1 x float> %6537, ptr addrspace(3) %6538, align 4
  %6539 = add i64 %6486, %6318
  %6540 = extractelement <4 x float> %6242, i64 1
  %6541 = insertelement <1 x float> poison, float %6540, i32 0
  %6542 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6539
  store <1 x float> %6541, ptr addrspace(3) %6542, align 4
  %6543 = add i64 %6492, %6318
  %6544 = extractelement <4 x float> %6242, i64 2
  %6545 = insertelement <1 x float> poison, float %6544, i32 0
  %6546 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6543
  store <1 x float> %6545, ptr addrspace(3) %6546, align 4
  %6547 = add i64 %6498, %6318
  %6548 = extractelement <4 x float> %6242, i64 3
  %6549 = insertelement <1 x float> poison, float %6548, i32 0
  %6550 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6547
  store <1 x float> %6549, ptr addrspace(3) %6550, align 4
  %6551 = add i64 %6257, 64
  %6552 = mul i64 %6551, 256
  %6553 = add i64 %6552, %6259
  %6554 = extractelement <4 x float> %6190, i64 0
  %6555 = insertelement <1 x float> poison, float %6554, i32 0
  %6556 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6553
  store <1 x float> %6555, ptr addrspace(3) %6556, align 4
  %6557 = add i64 %6257, 65
  %6558 = mul i64 %6557, 256
  %6559 = add i64 %6558, %6259
  %6560 = extractelement <4 x float> %6190, i64 1
  %6561 = insertelement <1 x float> poison, float %6560, i32 0
  %6562 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6559
  store <1 x float> %6561, ptr addrspace(3) %6562, align 4
  %6563 = add i64 %6257, 66
  %6564 = mul i64 %6563, 256
  %6565 = add i64 %6564, %6259
  %6566 = extractelement <4 x float> %6190, i64 2
  %6567 = insertelement <1 x float> poison, float %6566, i32 0
  %6568 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6565
  store <1 x float> %6567, ptr addrspace(3) %6568, align 4
  %6569 = add i64 %6257, 67
  %6570 = mul i64 %6569, 256
  %6571 = add i64 %6570, %6259
  %6572 = extractelement <4 x float> %6190, i64 3
  %6573 = insertelement <1 x float> poison, float %6572, i32 0
  %6574 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6571
  store <1 x float> %6573, ptr addrspace(3) %6574, align 4
  %6575 = add i64 %6552, %6283
  %6576 = extractelement <4 x float> %6212, i64 0
  %6577 = insertelement <1 x float> poison, float %6576, i32 0
  %6578 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6575
  store <1 x float> %6577, ptr addrspace(3) %6578, align 4
  %6579 = add i64 %6558, %6283
  %6580 = extractelement <4 x float> %6212, i64 1
  %6581 = insertelement <1 x float> poison, float %6580, i32 0
  %6582 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6579
  store <1 x float> %6581, ptr addrspace(3) %6582, align 4
  %6583 = add i64 %6564, %6283
  %6584 = extractelement <4 x float> %6212, i64 2
  %6585 = insertelement <1 x float> poison, float %6584, i32 0
  %6586 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6583
  store <1 x float> %6585, ptr addrspace(3) %6586, align 4
  %6587 = add i64 %6570, %6283
  %6588 = extractelement <4 x float> %6212, i64 3
  %6589 = insertelement <1 x float> poison, float %6588, i32 0
  %6590 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6587
  store <1 x float> %6589, ptr addrspace(3) %6590, align 4
  %6591 = add i64 %6552, %6301
  %6592 = extractelement <4 x float> %6228, i64 0
  %6593 = insertelement <1 x float> poison, float %6592, i32 0
  %6594 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6591
  store <1 x float> %6593, ptr addrspace(3) %6594, align 4
  %6595 = add i64 %6558, %6301
  %6596 = extractelement <4 x float> %6228, i64 1
  %6597 = insertelement <1 x float> poison, float %6596, i32 0
  %6598 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6595
  store <1 x float> %6597, ptr addrspace(3) %6598, align 4
  %6599 = add i64 %6564, %6301
  %6600 = extractelement <4 x float> %6228, i64 2
  %6601 = insertelement <1 x float> poison, float %6600, i32 0
  %6602 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6599
  store <1 x float> %6601, ptr addrspace(3) %6602, align 4
  %6603 = add i64 %6570, %6301
  %6604 = extractelement <4 x float> %6228, i64 3
  %6605 = insertelement <1 x float> poison, float %6604, i32 0
  %6606 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6603
  store <1 x float> %6605, ptr addrspace(3) %6606, align 4
  %6607 = add i64 %6552, %6318
  %6608 = extractelement <4 x float> %6244, i64 0
  %6609 = insertelement <1 x float> poison, float %6608, i32 0
  %6610 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6607
  store <1 x float> %6609, ptr addrspace(3) %6610, align 4
  %6611 = add i64 %6558, %6318
  %6612 = extractelement <4 x float> %6244, i64 1
  %6613 = insertelement <1 x float> poison, float %6612, i32 0
  %6614 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6611
  store <1 x float> %6613, ptr addrspace(3) %6614, align 4
  %6615 = add i64 %6564, %6318
  %6616 = extractelement <4 x float> %6244, i64 2
  %6617 = insertelement <1 x float> poison, float %6616, i32 0
  %6618 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6615
  store <1 x float> %6617, ptr addrspace(3) %6618, align 4
  %6619 = add i64 %6570, %6318
  %6620 = extractelement <4 x float> %6244, i64 3
  %6621 = insertelement <1 x float> poison, float %6620, i32 0
  %6622 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6619
  store <1 x float> %6621, ptr addrspace(3) %6622, align 4
  %6623 = add i64 %6257, 80
  %6624 = mul i64 %6623, 256
  %6625 = add i64 %6624, %6259
  %6626 = extractelement <4 x float> %6194, i64 0
  %6627 = insertelement <1 x float> poison, float %6626, i32 0
  %6628 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6625
  store <1 x float> %6627, ptr addrspace(3) %6628, align 4
  %6629 = add i64 %6257, 81
  %6630 = mul i64 %6629, 256
  %6631 = add i64 %6630, %6259
  %6632 = extractelement <4 x float> %6194, i64 1
  %6633 = insertelement <1 x float> poison, float %6632, i32 0
  %6634 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6631
  store <1 x float> %6633, ptr addrspace(3) %6634, align 4
  %6635 = add i64 %6257, 82
  %6636 = mul i64 %6635, 256
  %6637 = add i64 %6636, %6259
  %6638 = extractelement <4 x float> %6194, i64 2
  %6639 = insertelement <1 x float> poison, float %6638, i32 0
  %6640 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6637
  store <1 x float> %6639, ptr addrspace(3) %6640, align 4
  %6641 = add i64 %6257, 83
  %6642 = mul i64 %6641, 256
  %6643 = add i64 %6642, %6259
  %6644 = extractelement <4 x float> %6194, i64 3
  %6645 = insertelement <1 x float> poison, float %6644, i32 0
  %6646 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6643
  store <1 x float> %6645, ptr addrspace(3) %6646, align 4
  %6647 = add i64 %6624, %6283
  %6648 = extractelement <4 x float> %6214, i64 0
  %6649 = insertelement <1 x float> poison, float %6648, i32 0
  %6650 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6647
  store <1 x float> %6649, ptr addrspace(3) %6650, align 4
  %6651 = add i64 %6630, %6283
  %6652 = extractelement <4 x float> %6214, i64 1
  %6653 = insertelement <1 x float> poison, float %6652, i32 0
  %6654 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6651
  store <1 x float> %6653, ptr addrspace(3) %6654, align 4
  %6655 = add i64 %6636, %6283
  %6656 = extractelement <4 x float> %6214, i64 2
  %6657 = insertelement <1 x float> poison, float %6656, i32 0
  %6658 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6655
  store <1 x float> %6657, ptr addrspace(3) %6658, align 4
  %6659 = add i64 %6642, %6283
  %6660 = extractelement <4 x float> %6214, i64 3
  %6661 = insertelement <1 x float> poison, float %6660, i32 0
  %6662 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6659
  store <1 x float> %6661, ptr addrspace(3) %6662, align 4
  %6663 = add i64 %6624, %6301
  %6664 = extractelement <4 x float> %6230, i64 0
  %6665 = insertelement <1 x float> poison, float %6664, i32 0
  %6666 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6663
  store <1 x float> %6665, ptr addrspace(3) %6666, align 4
  %6667 = add i64 %6630, %6301
  %6668 = extractelement <4 x float> %6230, i64 1
  %6669 = insertelement <1 x float> poison, float %6668, i32 0
  %6670 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6667
  store <1 x float> %6669, ptr addrspace(3) %6670, align 4
  %6671 = add i64 %6636, %6301
  %6672 = extractelement <4 x float> %6230, i64 2
  %6673 = insertelement <1 x float> poison, float %6672, i32 0
  %6674 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6671
  store <1 x float> %6673, ptr addrspace(3) %6674, align 4
  %6675 = add i64 %6642, %6301
  %6676 = extractelement <4 x float> %6230, i64 3
  %6677 = insertelement <1 x float> poison, float %6676, i32 0
  %6678 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6675
  store <1 x float> %6677, ptr addrspace(3) %6678, align 4
  %6679 = add i64 %6624, %6318
  %6680 = extractelement <4 x float> %6246, i64 0
  %6681 = insertelement <1 x float> poison, float %6680, i32 0
  %6682 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6679
  store <1 x float> %6681, ptr addrspace(3) %6682, align 4
  %6683 = add i64 %6630, %6318
  %6684 = extractelement <4 x float> %6246, i64 1
  %6685 = insertelement <1 x float> poison, float %6684, i32 0
  %6686 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6683
  store <1 x float> %6685, ptr addrspace(3) %6686, align 4
  %6687 = add i64 %6636, %6318
  %6688 = extractelement <4 x float> %6246, i64 2
  %6689 = insertelement <1 x float> poison, float %6688, i32 0
  %6690 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6687
  store <1 x float> %6689, ptr addrspace(3) %6690, align 4
  %6691 = add i64 %6642, %6318
  %6692 = extractelement <4 x float> %6246, i64 3
  %6693 = insertelement <1 x float> poison, float %6692, i32 0
  %6694 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6691
  store <1 x float> %6693, ptr addrspace(3) %6694, align 4
  %6695 = add i64 %6257, 96
  %6696 = mul i64 %6695, 256
  %6697 = add i64 %6696, %6259
  %6698 = extractelement <4 x float> %6198, i64 0
  %6699 = insertelement <1 x float> poison, float %6698, i32 0
  %6700 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6697
  store <1 x float> %6699, ptr addrspace(3) %6700, align 4
  %6701 = add i64 %6257, 97
  %6702 = mul i64 %6701, 256
  %6703 = add i64 %6702, %6259
  %6704 = extractelement <4 x float> %6198, i64 1
  %6705 = insertelement <1 x float> poison, float %6704, i32 0
  %6706 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6703
  store <1 x float> %6705, ptr addrspace(3) %6706, align 4
  %6707 = add i64 %6257, 98
  %6708 = mul i64 %6707, 256
  %6709 = add i64 %6708, %6259
  %6710 = extractelement <4 x float> %6198, i64 2
  %6711 = insertelement <1 x float> poison, float %6710, i32 0
  %6712 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6709
  store <1 x float> %6711, ptr addrspace(3) %6712, align 4
  %6713 = add i64 %6257, 99
  %6714 = mul i64 %6713, 256
  %6715 = add i64 %6714, %6259
  %6716 = extractelement <4 x float> %6198, i64 3
  %6717 = insertelement <1 x float> poison, float %6716, i32 0
  %6718 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6715
  store <1 x float> %6717, ptr addrspace(3) %6718, align 4
  %6719 = add i64 %6696, %6283
  %6720 = extractelement <4 x float> %6216, i64 0
  %6721 = insertelement <1 x float> poison, float %6720, i32 0
  %6722 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6719
  store <1 x float> %6721, ptr addrspace(3) %6722, align 4
  %6723 = add i64 %6702, %6283
  %6724 = extractelement <4 x float> %6216, i64 1
  %6725 = insertelement <1 x float> poison, float %6724, i32 0
  %6726 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6723
  store <1 x float> %6725, ptr addrspace(3) %6726, align 4
  %6727 = add i64 %6708, %6283
  %6728 = extractelement <4 x float> %6216, i64 2
  %6729 = insertelement <1 x float> poison, float %6728, i32 0
  %6730 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6727
  store <1 x float> %6729, ptr addrspace(3) %6730, align 4
  %6731 = add i64 %6714, %6283
  %6732 = extractelement <4 x float> %6216, i64 3
  %6733 = insertelement <1 x float> poison, float %6732, i32 0
  %6734 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6731
  store <1 x float> %6733, ptr addrspace(3) %6734, align 4
  %6735 = add i64 %6696, %6301
  %6736 = extractelement <4 x float> %6232, i64 0
  %6737 = insertelement <1 x float> poison, float %6736, i32 0
  %6738 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6735
  store <1 x float> %6737, ptr addrspace(3) %6738, align 4
  %6739 = add i64 %6702, %6301
  %6740 = extractelement <4 x float> %6232, i64 1
  %6741 = insertelement <1 x float> poison, float %6740, i32 0
  %6742 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6739
  store <1 x float> %6741, ptr addrspace(3) %6742, align 4
  %6743 = add i64 %6708, %6301
  %6744 = extractelement <4 x float> %6232, i64 2
  %6745 = insertelement <1 x float> poison, float %6744, i32 0
  %6746 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6743
  store <1 x float> %6745, ptr addrspace(3) %6746, align 4
  %6747 = add i64 %6714, %6301
  %6748 = extractelement <4 x float> %6232, i64 3
  %6749 = insertelement <1 x float> poison, float %6748, i32 0
  %6750 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6747
  store <1 x float> %6749, ptr addrspace(3) %6750, align 4
  %6751 = add i64 %6696, %6318
  %6752 = extractelement <4 x float> %6248, i64 0
  %6753 = insertelement <1 x float> poison, float %6752, i32 0
  %6754 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6751
  store <1 x float> %6753, ptr addrspace(3) %6754, align 4
  %6755 = add i64 %6702, %6318
  %6756 = extractelement <4 x float> %6248, i64 1
  %6757 = insertelement <1 x float> poison, float %6756, i32 0
  %6758 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6755
  store <1 x float> %6757, ptr addrspace(3) %6758, align 4
  %6759 = add i64 %6708, %6318
  %6760 = extractelement <4 x float> %6248, i64 2
  %6761 = insertelement <1 x float> poison, float %6760, i32 0
  %6762 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6759
  store <1 x float> %6761, ptr addrspace(3) %6762, align 4
  %6763 = add i64 %6714, %6318
  %6764 = extractelement <4 x float> %6248, i64 3
  %6765 = insertelement <1 x float> poison, float %6764, i32 0
  %6766 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6763
  store <1 x float> %6765, ptr addrspace(3) %6766, align 4
  %6767 = add i64 %6257, 112
  %6768 = mul i64 %6767, 256
  %6769 = add i64 %6768, %6259
  %6770 = extractelement <4 x float> %6202, i64 0
  %6771 = insertelement <1 x float> poison, float %6770, i32 0
  %6772 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6769
  store <1 x float> %6771, ptr addrspace(3) %6772, align 4
  %6773 = add i64 %6257, 113
  %6774 = mul i64 %6773, 256
  %6775 = add i64 %6774, %6259
  %6776 = extractelement <4 x float> %6202, i64 1
  %6777 = insertelement <1 x float> poison, float %6776, i32 0
  %6778 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6775
  store <1 x float> %6777, ptr addrspace(3) %6778, align 4
  %6779 = add i64 %6257, 114
  %6780 = mul i64 %6779, 256
  %6781 = add i64 %6780, %6259
  %6782 = extractelement <4 x float> %6202, i64 2
  %6783 = insertelement <1 x float> poison, float %6782, i32 0
  %6784 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6781
  store <1 x float> %6783, ptr addrspace(3) %6784, align 4
  %6785 = add i64 %6257, 115
  %6786 = mul i64 %6785, 256
  %6787 = add i64 %6786, %6259
  %6788 = extractelement <4 x float> %6202, i64 3
  %6789 = insertelement <1 x float> poison, float %6788, i32 0
  %6790 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6787
  store <1 x float> %6789, ptr addrspace(3) %6790, align 4
  %6791 = add i64 %6768, %6283
  %6792 = extractelement <4 x float> %6218, i64 0
  %6793 = insertelement <1 x float> poison, float %6792, i32 0
  %6794 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6791
  store <1 x float> %6793, ptr addrspace(3) %6794, align 4
  %6795 = add i64 %6774, %6283
  %6796 = extractelement <4 x float> %6218, i64 1
  %6797 = insertelement <1 x float> poison, float %6796, i32 0
  %6798 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6795
  store <1 x float> %6797, ptr addrspace(3) %6798, align 4
  %6799 = add i64 %6780, %6283
  %6800 = extractelement <4 x float> %6218, i64 2
  %6801 = insertelement <1 x float> poison, float %6800, i32 0
  %6802 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6799
  store <1 x float> %6801, ptr addrspace(3) %6802, align 4
  %6803 = add i64 %6786, %6283
  %6804 = extractelement <4 x float> %6218, i64 3
  %6805 = insertelement <1 x float> poison, float %6804, i32 0
  %6806 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6803
  store <1 x float> %6805, ptr addrspace(3) %6806, align 4
  %6807 = add i64 %6768, %6301
  %6808 = extractelement <4 x float> %6234, i64 0
  %6809 = insertelement <1 x float> poison, float %6808, i32 0
  %6810 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6807
  store <1 x float> %6809, ptr addrspace(3) %6810, align 4
  %6811 = add i64 %6774, %6301
  %6812 = extractelement <4 x float> %6234, i64 1
  %6813 = insertelement <1 x float> poison, float %6812, i32 0
  %6814 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6811
  store <1 x float> %6813, ptr addrspace(3) %6814, align 4
  %6815 = add i64 %6780, %6301
  %6816 = extractelement <4 x float> %6234, i64 2
  %6817 = insertelement <1 x float> poison, float %6816, i32 0
  %6818 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6815
  store <1 x float> %6817, ptr addrspace(3) %6818, align 4
  %6819 = add i64 %6786, %6301
  %6820 = extractelement <4 x float> %6234, i64 3
  %6821 = insertelement <1 x float> poison, float %6820, i32 0
  %6822 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6819
  store <1 x float> %6821, ptr addrspace(3) %6822, align 4
  %6823 = add i64 %6768, %6318
  %6824 = extractelement <4 x float> %6250, i64 0
  %6825 = insertelement <1 x float> poison, float %6824, i32 0
  %6826 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6823
  store <1 x float> %6825, ptr addrspace(3) %6826, align 4
  %6827 = add i64 %6774, %6318
  %6828 = extractelement <4 x float> %6250, i64 1
  %6829 = insertelement <1 x float> poison, float %6828, i32 0
  %6830 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6827
  store <1 x float> %6829, ptr addrspace(3) %6830, align 4
  %6831 = add i64 %6780, %6318
  %6832 = extractelement <4 x float> %6250, i64 2
  %6833 = insertelement <1 x float> poison, float %6832, i32 0
  %6834 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6831
  store <1 x float> %6833, ptr addrspace(3) %6834, align 4
  %6835 = add i64 %6786, %6318
  %6836 = extractelement <4 x float> %6250, i64 3
  %6837 = insertelement <1 x float> poison, float %6836, i32 0
  %6838 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6835
  store <1 x float> %6837, ptr addrspace(3) %6838, align 4
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %6839 = udiv i64 %14, 16
  %6840 = urem i64 %14, 16
  %6841 = udiv i64 %6840, 4
  %6842 = urem i64 %6840, 4
  %6843 = mul i64 %6841, 32
  %6844 = mul i64 %6842, 8
  %6845 = add i64 %6843, %6844
  %6846 = mul i64 %6839, 256
  %6847 = add i64 %6846, %6845
  %6848 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6847
  %6849 = load <1 x float>, ptr addrspace(3) %6848, align 4
  %6850 = extractelement <1 x float> %6849, i64 0
  %6851 = add i64 %6847, 128
  %6852 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6851
  %6853 = load <1 x float>, ptr addrspace(3) %6852, align 4
  %6854 = extractelement <1 x float> %6853, i64 0
  %6855 = fmul float %6850, 0xBFF7154760000000
  %6856 = call float @llvm.amdgcn.exp2.f32(float %6855)
  %6857 = fadd float %6856, 1.000000e+00
  %6858 = call float @llvm.amdgcn.rcp.f32(float %6857)
  %6859 = fmul float %6850, %6858
  %6860 = fmul float %6859, %6854
  %6861 = add i64 %6845, 1
  %6862 = add i64 %6846, %6861
  %6863 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6862
  %6864 = load <1 x float>, ptr addrspace(3) %6863, align 4
  %6865 = extractelement <1 x float> %6864, i64 0
  %6866 = add i64 %6862, 128
  %6867 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6866
  %6868 = load <1 x float>, ptr addrspace(3) %6867, align 4
  %6869 = extractelement <1 x float> %6868, i64 0
  %6870 = fmul float %6865, 0xBFF7154760000000
  %6871 = call float @llvm.amdgcn.exp2.f32(float %6870)
  %6872 = fadd float %6871, 1.000000e+00
  %6873 = call float @llvm.amdgcn.rcp.f32(float %6872)
  %6874 = fmul float %6865, %6873
  %6875 = fmul float %6874, %6869
  %6876 = add i64 %6845, 2
  %6877 = add i64 %6846, %6876
  %6878 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6877
  %6879 = load <1 x float>, ptr addrspace(3) %6878, align 4
  %6880 = extractelement <1 x float> %6879, i64 0
  %6881 = add i64 %6877, 128
  %6882 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6881
  %6883 = load <1 x float>, ptr addrspace(3) %6882, align 4
  %6884 = extractelement <1 x float> %6883, i64 0
  %6885 = fmul float %6880, 0xBFF7154760000000
  %6886 = call float @llvm.amdgcn.exp2.f32(float %6885)
  %6887 = fadd float %6886, 1.000000e+00
  %6888 = call float @llvm.amdgcn.rcp.f32(float %6887)
  %6889 = fmul float %6880, %6888
  %6890 = fmul float %6889, %6884
  %6891 = add i64 %6845, 3
  %6892 = add i64 %6846, %6891
  %6893 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6892
  %6894 = load <1 x float>, ptr addrspace(3) %6893, align 4
  %6895 = extractelement <1 x float> %6894, i64 0
  %6896 = add i64 %6892, 128
  %6897 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6896
  %6898 = load <1 x float>, ptr addrspace(3) %6897, align 4
  %6899 = extractelement <1 x float> %6898, i64 0
  %6900 = fmul float %6895, 0xBFF7154760000000
  %6901 = call float @llvm.amdgcn.exp2.f32(float %6900)
  %6902 = fadd float %6901, 1.000000e+00
  %6903 = call float @llvm.amdgcn.rcp.f32(float %6902)
  %6904 = fmul float %6895, %6903
  %6905 = fmul float %6904, %6899
  %6906 = add i64 %6845, 4
  %6907 = add i64 %6846, %6906
  %6908 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6907
  %6909 = load <1 x float>, ptr addrspace(3) %6908, align 4
  %6910 = extractelement <1 x float> %6909, i64 0
  %6911 = add i64 %6907, 128
  %6912 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6911
  %6913 = load <1 x float>, ptr addrspace(3) %6912, align 4
  %6914 = extractelement <1 x float> %6913, i64 0
  %6915 = fmul float %6910, 0xBFF7154760000000
  %6916 = call float @llvm.amdgcn.exp2.f32(float %6915)
  %6917 = fadd float %6916, 1.000000e+00
  %6918 = call float @llvm.amdgcn.rcp.f32(float %6917)
  %6919 = fmul float %6910, %6918
  %6920 = fmul float %6919, %6914
  %6921 = add i64 %6845, 5
  %6922 = add i64 %6846, %6921
  %6923 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6922
  %6924 = load <1 x float>, ptr addrspace(3) %6923, align 4
  %6925 = extractelement <1 x float> %6924, i64 0
  %6926 = add i64 %6922, 128
  %6927 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6926
  %6928 = load <1 x float>, ptr addrspace(3) %6927, align 4
  %6929 = extractelement <1 x float> %6928, i64 0
  %6930 = fmul float %6925, 0xBFF7154760000000
  %6931 = call float @llvm.amdgcn.exp2.f32(float %6930)
  %6932 = fadd float %6931, 1.000000e+00
  %6933 = call float @llvm.amdgcn.rcp.f32(float %6932)
  %6934 = fmul float %6925, %6933
  %6935 = fmul float %6934, %6929
  %6936 = add i64 %6845, 6
  %6937 = add i64 %6846, %6936
  %6938 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6937
  %6939 = load <1 x float>, ptr addrspace(3) %6938, align 4
  %6940 = extractelement <1 x float> %6939, i64 0
  %6941 = add i64 %6937, 128
  %6942 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6941
  %6943 = load <1 x float>, ptr addrspace(3) %6942, align 4
  %6944 = extractelement <1 x float> %6943, i64 0
  %6945 = fmul float %6940, 0xBFF7154760000000
  %6946 = call float @llvm.amdgcn.exp2.f32(float %6945)
  %6947 = fadd float %6946, 1.000000e+00
  %6948 = call float @llvm.amdgcn.rcp.f32(float %6947)
  %6949 = fmul float %6940, %6948
  %6950 = fmul float %6949, %6944
  %6951 = add i64 %6845, 7
  %6952 = add i64 %6846, %6951
  %6953 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6952
  %6954 = load <1 x float>, ptr addrspace(3) %6953, align 4
  %6955 = extractelement <1 x float> %6954, i64 0
  %6956 = add i64 %6952, 128
  %6957 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %6956
  %6958 = load <1 x float>, ptr addrspace(3) %6957, align 4
  %6959 = extractelement <1 x float> %6958, i64 0
  %6960 = fmul float %6955, 0xBFF7154760000000
  %6961 = call float @llvm.amdgcn.exp2.f32(float %6960)
  %6962 = fadd float %6961, 1.000000e+00
  %6963 = call float @llvm.amdgcn.rcp.f32(float %6962)
  %6964 = fmul float %6955, %6963
  %6965 = fmul float %6964, %6959
  %6966 = bitcast float %6860 to i32
  %6967 = and i32 %6966, 2147483647
  %6968 = bitcast i32 %6967 to float
  %6969 = bitcast float %6875 to i32
  %6970 = and i32 %6969, 2147483647
  %6971 = bitcast i32 %6970 to float
  %6972 = call float @llvm.maximum.f32(float %6968, float %6971)
  %6973 = bitcast float %6890 to i32
  %6974 = and i32 %6973, 2147483647
  %6975 = bitcast i32 %6974 to float
  %6976 = call float @llvm.maximum.f32(float %6972, float %6975)
  %6977 = bitcast float %6905 to i32
  %6978 = and i32 %6977, 2147483647
  %6979 = bitcast i32 %6978 to float
  %6980 = call float @llvm.maximum.f32(float %6976, float %6979)
  %6981 = bitcast float %6920 to i32
  %6982 = and i32 %6981, 2147483647
  %6983 = bitcast i32 %6982 to float
  %6984 = call float @llvm.maximum.f32(float %6980, float %6983)
  %6985 = bitcast float %6935 to i32
  %6986 = and i32 %6985, 2147483647
  %6987 = bitcast i32 %6986 to float
  %6988 = call float @llvm.maximum.f32(float %6984, float %6987)
  %6989 = bitcast float %6950 to i32
  %6990 = and i32 %6989, 2147483647
  %6991 = bitcast i32 %6990 to float
  %6992 = call float @llvm.maximum.f32(float %6988, float %6991)
  %6993 = bitcast float %6965 to i32
  %6994 = and i32 %6993, 2147483647
  %6995 = bitcast i32 %6994 to float
  %6996 = call float @llvm.maximum.f32(float %6992, float %6995)
  %6997 = bitcast float %6996 to i32
  %6998 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %6997, i32 %6997, i32 177, i32 15, i32 15, i1 true)
  %6999 = bitcast i32 %6998 to float
  %7000 = call float @llvm.maximum.f32(float %6996, float %6999)
  %7001 = bitcast float %7000 to i32
  %7002 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %7001, i32 %7001, i32 78, i32 15, i32 15, i1 true)
  %7003 = bitcast i32 %7002 to float
  %7004 = call float @llvm.maximum.f32(float %7000, float %7003)
  %7005 = bitcast float %7004 to i32
  %7006 = add i32 %7005, 2097152
  %7007 = bitcast i32 %7006 to float
  %7008 = fmul float %7007, 2.500000e-01
  %7009 = bitcast float %7008 to i32
  %7010 = lshr i32 %7009, 23
  %7011 = call i32 @llvm.smin.i32(i32 %7010, i32 254)
  %7012 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 0, float %6860, float %6875, float %7008, i32 0)
  %7013 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %7012, float %6890, float %6905, float %7008, i32 1)
  %7014 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %7013, float %6920, float %6935, float %7008, i32 2)
  %7015 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %7014, float %6950, float %6965, float %7008, i32 3)
  %7016 = mul i64 %18, 64
  %7017 = mul i64 %6841, 16
  %7018 = add i64 %7016, %7017
  %7019 = mul i64 %6842, 4
  %7020 = add i64 %7018, %7019
  %7021 = add i64 %46, %6839
  %7022 = mul i64 %7021, 256
  %7023 = add i64 %7022, %7020
  %7024 = ptrtoint ptr addrspace(1) %7 to i64
  %7025 = add i64 %7024, %7023
  %7026 = inttoptr i64 %7025 to ptr addrspace(1)
  store i32 %7015, ptr addrspace(1) %7026, align 4
  %7027 = add i64 %6839, 16
  %7028 = mul i64 %7027, 256
  %7029 = add i64 %7028, %6845
  %7030 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7029
  %7031 = load <1 x float>, ptr addrspace(3) %7030, align 4
  %7032 = extractelement <1 x float> %7031, i64 0
  %7033 = add i64 %7029, 128
  %7034 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7033
  %7035 = load <1 x float>, ptr addrspace(3) %7034, align 4
  %7036 = extractelement <1 x float> %7035, i64 0
  %7037 = fmul float %7032, 0xBFF7154760000000
  %7038 = call float @llvm.amdgcn.exp2.f32(float %7037)
  %7039 = fadd float %7038, 1.000000e+00
  %7040 = call float @llvm.amdgcn.rcp.f32(float %7039)
  %7041 = fmul float %7032, %7040
  %7042 = fmul float %7041, %7036
  %7043 = add i64 %7028, %6861
  %7044 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7043
  %7045 = load <1 x float>, ptr addrspace(3) %7044, align 4
  %7046 = extractelement <1 x float> %7045, i64 0
  %7047 = add i64 %7043, 128
  %7048 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7047
  %7049 = load <1 x float>, ptr addrspace(3) %7048, align 4
  %7050 = extractelement <1 x float> %7049, i64 0
  %7051 = fmul float %7046, 0xBFF7154760000000
  %7052 = call float @llvm.amdgcn.exp2.f32(float %7051)
  %7053 = fadd float %7052, 1.000000e+00
  %7054 = call float @llvm.amdgcn.rcp.f32(float %7053)
  %7055 = fmul float %7046, %7054
  %7056 = fmul float %7055, %7050
  %7057 = add i64 %7028, %6876
  %7058 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7057
  %7059 = load <1 x float>, ptr addrspace(3) %7058, align 4
  %7060 = extractelement <1 x float> %7059, i64 0
  %7061 = add i64 %7057, 128
  %7062 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7061
  %7063 = load <1 x float>, ptr addrspace(3) %7062, align 4
  %7064 = extractelement <1 x float> %7063, i64 0
  %7065 = fmul float %7060, 0xBFF7154760000000
  %7066 = call float @llvm.amdgcn.exp2.f32(float %7065)
  %7067 = fadd float %7066, 1.000000e+00
  %7068 = call float @llvm.amdgcn.rcp.f32(float %7067)
  %7069 = fmul float %7060, %7068
  %7070 = fmul float %7069, %7064
  %7071 = add i64 %7028, %6891
  %7072 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7071
  %7073 = load <1 x float>, ptr addrspace(3) %7072, align 4
  %7074 = extractelement <1 x float> %7073, i64 0
  %7075 = add i64 %7071, 128
  %7076 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7075
  %7077 = load <1 x float>, ptr addrspace(3) %7076, align 4
  %7078 = extractelement <1 x float> %7077, i64 0
  %7079 = fmul float %7074, 0xBFF7154760000000
  %7080 = call float @llvm.amdgcn.exp2.f32(float %7079)
  %7081 = fadd float %7080, 1.000000e+00
  %7082 = call float @llvm.amdgcn.rcp.f32(float %7081)
  %7083 = fmul float %7074, %7082
  %7084 = fmul float %7083, %7078
  %7085 = add i64 %7028, %6906
  %7086 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7085
  %7087 = load <1 x float>, ptr addrspace(3) %7086, align 4
  %7088 = extractelement <1 x float> %7087, i64 0
  %7089 = add i64 %7085, 128
  %7090 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7089
  %7091 = load <1 x float>, ptr addrspace(3) %7090, align 4
  %7092 = extractelement <1 x float> %7091, i64 0
  %7093 = fmul float %7088, 0xBFF7154760000000
  %7094 = call float @llvm.amdgcn.exp2.f32(float %7093)
  %7095 = fadd float %7094, 1.000000e+00
  %7096 = call float @llvm.amdgcn.rcp.f32(float %7095)
  %7097 = fmul float %7088, %7096
  %7098 = fmul float %7097, %7092
  %7099 = add i64 %7028, %6921
  %7100 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7099
  %7101 = load <1 x float>, ptr addrspace(3) %7100, align 4
  %7102 = extractelement <1 x float> %7101, i64 0
  %7103 = add i64 %7099, 128
  %7104 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7103
  %7105 = load <1 x float>, ptr addrspace(3) %7104, align 4
  %7106 = extractelement <1 x float> %7105, i64 0
  %7107 = fmul float %7102, 0xBFF7154760000000
  %7108 = call float @llvm.amdgcn.exp2.f32(float %7107)
  %7109 = fadd float %7108, 1.000000e+00
  %7110 = call float @llvm.amdgcn.rcp.f32(float %7109)
  %7111 = fmul float %7102, %7110
  %7112 = fmul float %7111, %7106
  %7113 = add i64 %7028, %6936
  %7114 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7113
  %7115 = load <1 x float>, ptr addrspace(3) %7114, align 4
  %7116 = extractelement <1 x float> %7115, i64 0
  %7117 = add i64 %7113, 128
  %7118 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7117
  %7119 = load <1 x float>, ptr addrspace(3) %7118, align 4
  %7120 = extractelement <1 x float> %7119, i64 0
  %7121 = fmul float %7116, 0xBFF7154760000000
  %7122 = call float @llvm.amdgcn.exp2.f32(float %7121)
  %7123 = fadd float %7122, 1.000000e+00
  %7124 = call float @llvm.amdgcn.rcp.f32(float %7123)
  %7125 = fmul float %7116, %7124
  %7126 = fmul float %7125, %7120
  %7127 = add i64 %7028, %6951
  %7128 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7127
  %7129 = load <1 x float>, ptr addrspace(3) %7128, align 4
  %7130 = extractelement <1 x float> %7129, i64 0
  %7131 = add i64 %7127, 128
  %7132 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7131
  %7133 = load <1 x float>, ptr addrspace(3) %7132, align 4
  %7134 = extractelement <1 x float> %7133, i64 0
  %7135 = fmul float %7130, 0xBFF7154760000000
  %7136 = call float @llvm.amdgcn.exp2.f32(float %7135)
  %7137 = fadd float %7136, 1.000000e+00
  %7138 = call float @llvm.amdgcn.rcp.f32(float %7137)
  %7139 = fmul float %7130, %7138
  %7140 = fmul float %7139, %7134
  %7141 = bitcast float %7042 to i32
  %7142 = and i32 %7141, 2147483647
  %7143 = bitcast i32 %7142 to float
  %7144 = bitcast float %7056 to i32
  %7145 = and i32 %7144, 2147483647
  %7146 = bitcast i32 %7145 to float
  %7147 = call float @llvm.maximum.f32(float %7143, float %7146)
  %7148 = bitcast float %7070 to i32
  %7149 = and i32 %7148, 2147483647
  %7150 = bitcast i32 %7149 to float
  %7151 = call float @llvm.maximum.f32(float %7147, float %7150)
  %7152 = bitcast float %7084 to i32
  %7153 = and i32 %7152, 2147483647
  %7154 = bitcast i32 %7153 to float
  %7155 = call float @llvm.maximum.f32(float %7151, float %7154)
  %7156 = bitcast float %7098 to i32
  %7157 = and i32 %7156, 2147483647
  %7158 = bitcast i32 %7157 to float
  %7159 = call float @llvm.maximum.f32(float %7155, float %7158)
  %7160 = bitcast float %7112 to i32
  %7161 = and i32 %7160, 2147483647
  %7162 = bitcast i32 %7161 to float
  %7163 = call float @llvm.maximum.f32(float %7159, float %7162)
  %7164 = bitcast float %7126 to i32
  %7165 = and i32 %7164, 2147483647
  %7166 = bitcast i32 %7165 to float
  %7167 = call float @llvm.maximum.f32(float %7163, float %7166)
  %7168 = bitcast float %7140 to i32
  %7169 = and i32 %7168, 2147483647
  %7170 = bitcast i32 %7169 to float
  %7171 = call float @llvm.maximum.f32(float %7167, float %7170)
  %7172 = bitcast float %7171 to i32
  %7173 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %7172, i32 %7172, i32 177, i32 15, i32 15, i1 true)
  %7174 = bitcast i32 %7173 to float
  %7175 = call float @llvm.maximum.f32(float %7171, float %7174)
  %7176 = bitcast float %7175 to i32
  %7177 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %7176, i32 %7176, i32 78, i32 15, i32 15, i1 true)
  %7178 = bitcast i32 %7177 to float
  %7179 = call float @llvm.maximum.f32(float %7175, float %7178)
  %7180 = bitcast float %7179 to i32
  %7181 = add i32 %7180, 2097152
  %7182 = bitcast i32 %7181 to float
  %7183 = fmul float %7182, 2.500000e-01
  %7184 = bitcast float %7183 to i32
  %7185 = lshr i32 %7184, 23
  %7186 = call i32 @llvm.smin.i32(i32 %7185, i32 254)
  %7187 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 0, float %7042, float %7056, float %7183, i32 0)
  %7188 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %7187, float %7070, float %7084, float %7183, i32 1)
  %7189 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %7188, float %7098, float %7112, float %7183, i32 2)
  %7190 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %7189, float %7126, float %7140, float %7183, i32 3)
  %7191 = add i64 %46, %7027
  %7192 = mul i64 %7191, 256
  %7193 = add i64 %7192, %7020
  %7194 = add i64 %7024, %7193
  %7195 = inttoptr i64 %7194 to ptr addrspace(1)
  store i32 %7190, ptr addrspace(1) %7195, align 4
  %7196 = add i64 %6839, 32
  %7197 = mul i64 %7196, 256
  %7198 = add i64 %7197, %6845
  %7199 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7198
  %7200 = load <1 x float>, ptr addrspace(3) %7199, align 4
  %7201 = extractelement <1 x float> %7200, i64 0
  %7202 = add i64 %7198, 128
  %7203 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7202
  %7204 = load <1 x float>, ptr addrspace(3) %7203, align 4
  %7205 = extractelement <1 x float> %7204, i64 0
  %7206 = fmul float %7201, 0xBFF7154760000000
  %7207 = call float @llvm.amdgcn.exp2.f32(float %7206)
  %7208 = fadd float %7207, 1.000000e+00
  %7209 = call float @llvm.amdgcn.rcp.f32(float %7208)
  %7210 = fmul float %7201, %7209
  %7211 = fmul float %7210, %7205
  %7212 = add i64 %7197, %6861
  %7213 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7212
  %7214 = load <1 x float>, ptr addrspace(3) %7213, align 4
  %7215 = extractelement <1 x float> %7214, i64 0
  %7216 = add i64 %7212, 128
  %7217 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7216
  %7218 = load <1 x float>, ptr addrspace(3) %7217, align 4
  %7219 = extractelement <1 x float> %7218, i64 0
  %7220 = fmul float %7215, 0xBFF7154760000000
  %7221 = call float @llvm.amdgcn.exp2.f32(float %7220)
  %7222 = fadd float %7221, 1.000000e+00
  %7223 = call float @llvm.amdgcn.rcp.f32(float %7222)
  %7224 = fmul float %7215, %7223
  %7225 = fmul float %7224, %7219
  %7226 = add i64 %7197, %6876
  %7227 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7226
  %7228 = load <1 x float>, ptr addrspace(3) %7227, align 4
  %7229 = extractelement <1 x float> %7228, i64 0
  %7230 = add i64 %7226, 128
  %7231 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7230
  %7232 = load <1 x float>, ptr addrspace(3) %7231, align 4
  %7233 = extractelement <1 x float> %7232, i64 0
  %7234 = fmul float %7229, 0xBFF7154760000000
  %7235 = call float @llvm.amdgcn.exp2.f32(float %7234)
  %7236 = fadd float %7235, 1.000000e+00
  %7237 = call float @llvm.amdgcn.rcp.f32(float %7236)
  %7238 = fmul float %7229, %7237
  %7239 = fmul float %7238, %7233
  %7240 = add i64 %7197, %6891
  %7241 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7240
  %7242 = load <1 x float>, ptr addrspace(3) %7241, align 4
  %7243 = extractelement <1 x float> %7242, i64 0
  %7244 = add i64 %7240, 128
  %7245 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7244
  %7246 = load <1 x float>, ptr addrspace(3) %7245, align 4
  %7247 = extractelement <1 x float> %7246, i64 0
  %7248 = fmul float %7243, 0xBFF7154760000000
  %7249 = call float @llvm.amdgcn.exp2.f32(float %7248)
  %7250 = fadd float %7249, 1.000000e+00
  %7251 = call float @llvm.amdgcn.rcp.f32(float %7250)
  %7252 = fmul float %7243, %7251
  %7253 = fmul float %7252, %7247
  %7254 = add i64 %7197, %6906
  %7255 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7254
  %7256 = load <1 x float>, ptr addrspace(3) %7255, align 4
  %7257 = extractelement <1 x float> %7256, i64 0
  %7258 = add i64 %7254, 128
  %7259 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7258
  %7260 = load <1 x float>, ptr addrspace(3) %7259, align 4
  %7261 = extractelement <1 x float> %7260, i64 0
  %7262 = fmul float %7257, 0xBFF7154760000000
  %7263 = call float @llvm.amdgcn.exp2.f32(float %7262)
  %7264 = fadd float %7263, 1.000000e+00
  %7265 = call float @llvm.amdgcn.rcp.f32(float %7264)
  %7266 = fmul float %7257, %7265
  %7267 = fmul float %7266, %7261
  %7268 = add i64 %7197, %6921
  %7269 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7268
  %7270 = load <1 x float>, ptr addrspace(3) %7269, align 4
  %7271 = extractelement <1 x float> %7270, i64 0
  %7272 = add i64 %7268, 128
  %7273 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7272
  %7274 = load <1 x float>, ptr addrspace(3) %7273, align 4
  %7275 = extractelement <1 x float> %7274, i64 0
  %7276 = fmul float %7271, 0xBFF7154760000000
  %7277 = call float @llvm.amdgcn.exp2.f32(float %7276)
  %7278 = fadd float %7277, 1.000000e+00
  %7279 = call float @llvm.amdgcn.rcp.f32(float %7278)
  %7280 = fmul float %7271, %7279
  %7281 = fmul float %7280, %7275
  %7282 = add i64 %7197, %6936
  %7283 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7282
  %7284 = load <1 x float>, ptr addrspace(3) %7283, align 4
  %7285 = extractelement <1 x float> %7284, i64 0
  %7286 = add i64 %7282, 128
  %7287 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7286
  %7288 = load <1 x float>, ptr addrspace(3) %7287, align 4
  %7289 = extractelement <1 x float> %7288, i64 0
  %7290 = fmul float %7285, 0xBFF7154760000000
  %7291 = call float @llvm.amdgcn.exp2.f32(float %7290)
  %7292 = fadd float %7291, 1.000000e+00
  %7293 = call float @llvm.amdgcn.rcp.f32(float %7292)
  %7294 = fmul float %7285, %7293
  %7295 = fmul float %7294, %7289
  %7296 = add i64 %7197, %6951
  %7297 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7296
  %7298 = load <1 x float>, ptr addrspace(3) %7297, align 4
  %7299 = extractelement <1 x float> %7298, i64 0
  %7300 = add i64 %7296, 128
  %7301 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7300
  %7302 = load <1 x float>, ptr addrspace(3) %7301, align 4
  %7303 = extractelement <1 x float> %7302, i64 0
  %7304 = fmul float %7299, 0xBFF7154760000000
  %7305 = call float @llvm.amdgcn.exp2.f32(float %7304)
  %7306 = fadd float %7305, 1.000000e+00
  %7307 = call float @llvm.amdgcn.rcp.f32(float %7306)
  %7308 = fmul float %7299, %7307
  %7309 = fmul float %7308, %7303
  %7310 = bitcast float %7211 to i32
  %7311 = and i32 %7310, 2147483647
  %7312 = bitcast i32 %7311 to float
  %7313 = bitcast float %7225 to i32
  %7314 = and i32 %7313, 2147483647
  %7315 = bitcast i32 %7314 to float
  %7316 = call float @llvm.maximum.f32(float %7312, float %7315)
  %7317 = bitcast float %7239 to i32
  %7318 = and i32 %7317, 2147483647
  %7319 = bitcast i32 %7318 to float
  %7320 = call float @llvm.maximum.f32(float %7316, float %7319)
  %7321 = bitcast float %7253 to i32
  %7322 = and i32 %7321, 2147483647
  %7323 = bitcast i32 %7322 to float
  %7324 = call float @llvm.maximum.f32(float %7320, float %7323)
  %7325 = bitcast float %7267 to i32
  %7326 = and i32 %7325, 2147483647
  %7327 = bitcast i32 %7326 to float
  %7328 = call float @llvm.maximum.f32(float %7324, float %7327)
  %7329 = bitcast float %7281 to i32
  %7330 = and i32 %7329, 2147483647
  %7331 = bitcast i32 %7330 to float
  %7332 = call float @llvm.maximum.f32(float %7328, float %7331)
  %7333 = bitcast float %7295 to i32
  %7334 = and i32 %7333, 2147483647
  %7335 = bitcast i32 %7334 to float
  %7336 = call float @llvm.maximum.f32(float %7332, float %7335)
  %7337 = bitcast float %7309 to i32
  %7338 = and i32 %7337, 2147483647
  %7339 = bitcast i32 %7338 to float
  %7340 = call float @llvm.maximum.f32(float %7336, float %7339)
  %7341 = bitcast float %7340 to i32
  %7342 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %7341, i32 %7341, i32 177, i32 15, i32 15, i1 true)
  %7343 = bitcast i32 %7342 to float
  %7344 = call float @llvm.maximum.f32(float %7340, float %7343)
  %7345 = bitcast float %7344 to i32
  %7346 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %7345, i32 %7345, i32 78, i32 15, i32 15, i1 true)
  %7347 = bitcast i32 %7346 to float
  %7348 = call float @llvm.maximum.f32(float %7344, float %7347)
  %7349 = bitcast float %7348 to i32
  %7350 = add i32 %7349, 2097152
  %7351 = bitcast i32 %7350 to float
  %7352 = fmul float %7351, 2.500000e-01
  %7353 = bitcast float %7352 to i32
  %7354 = lshr i32 %7353, 23
  %7355 = call i32 @llvm.smin.i32(i32 %7354, i32 254)
  %7356 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 0, float %7211, float %7225, float %7352, i32 0)
  %7357 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %7356, float %7239, float %7253, float %7352, i32 1)
  %7358 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %7357, float %7267, float %7281, float %7352, i32 2)
  %7359 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %7358, float %7295, float %7309, float %7352, i32 3)
  %7360 = add i64 %46, %7196
  %7361 = mul i64 %7360, 256
  %7362 = add i64 %7361, %7020
  %7363 = add i64 %7024, %7362
  %7364 = inttoptr i64 %7363 to ptr addrspace(1)
  store i32 %7359, ptr addrspace(1) %7364, align 4
  %7365 = add i64 %6839, 48
  %7366 = mul i64 %7365, 256
  %7367 = add i64 %7366, %6845
  %7368 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7367
  %7369 = load <1 x float>, ptr addrspace(3) %7368, align 4
  %7370 = extractelement <1 x float> %7369, i64 0
  %7371 = add i64 %7367, 128
  %7372 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7371
  %7373 = load <1 x float>, ptr addrspace(3) %7372, align 4
  %7374 = extractelement <1 x float> %7373, i64 0
  %7375 = fmul float %7370, 0xBFF7154760000000
  %7376 = call float @llvm.amdgcn.exp2.f32(float %7375)
  %7377 = fadd float %7376, 1.000000e+00
  %7378 = call float @llvm.amdgcn.rcp.f32(float %7377)
  %7379 = fmul float %7370, %7378
  %7380 = fmul float %7379, %7374
  %7381 = add i64 %7366, %6861
  %7382 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7381
  %7383 = load <1 x float>, ptr addrspace(3) %7382, align 4
  %7384 = extractelement <1 x float> %7383, i64 0
  %7385 = add i64 %7381, 128
  %7386 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7385
  %7387 = load <1 x float>, ptr addrspace(3) %7386, align 4
  %7388 = extractelement <1 x float> %7387, i64 0
  %7389 = fmul float %7384, 0xBFF7154760000000
  %7390 = call float @llvm.amdgcn.exp2.f32(float %7389)
  %7391 = fadd float %7390, 1.000000e+00
  %7392 = call float @llvm.amdgcn.rcp.f32(float %7391)
  %7393 = fmul float %7384, %7392
  %7394 = fmul float %7393, %7388
  %7395 = add i64 %7366, %6876
  %7396 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7395
  %7397 = load <1 x float>, ptr addrspace(3) %7396, align 4
  %7398 = extractelement <1 x float> %7397, i64 0
  %7399 = add i64 %7395, 128
  %7400 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7399
  %7401 = load <1 x float>, ptr addrspace(3) %7400, align 4
  %7402 = extractelement <1 x float> %7401, i64 0
  %7403 = fmul float %7398, 0xBFF7154760000000
  %7404 = call float @llvm.amdgcn.exp2.f32(float %7403)
  %7405 = fadd float %7404, 1.000000e+00
  %7406 = call float @llvm.amdgcn.rcp.f32(float %7405)
  %7407 = fmul float %7398, %7406
  %7408 = fmul float %7407, %7402
  %7409 = add i64 %7366, %6891
  %7410 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7409
  %7411 = load <1 x float>, ptr addrspace(3) %7410, align 4
  %7412 = extractelement <1 x float> %7411, i64 0
  %7413 = add i64 %7409, 128
  %7414 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7413
  %7415 = load <1 x float>, ptr addrspace(3) %7414, align 4
  %7416 = extractelement <1 x float> %7415, i64 0
  %7417 = fmul float %7412, 0xBFF7154760000000
  %7418 = call float @llvm.amdgcn.exp2.f32(float %7417)
  %7419 = fadd float %7418, 1.000000e+00
  %7420 = call float @llvm.amdgcn.rcp.f32(float %7419)
  %7421 = fmul float %7412, %7420
  %7422 = fmul float %7421, %7416
  %7423 = add i64 %7366, %6906
  %7424 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7423
  %7425 = load <1 x float>, ptr addrspace(3) %7424, align 4
  %7426 = extractelement <1 x float> %7425, i64 0
  %7427 = add i64 %7423, 128
  %7428 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7427
  %7429 = load <1 x float>, ptr addrspace(3) %7428, align 4
  %7430 = extractelement <1 x float> %7429, i64 0
  %7431 = fmul float %7426, 0xBFF7154760000000
  %7432 = call float @llvm.amdgcn.exp2.f32(float %7431)
  %7433 = fadd float %7432, 1.000000e+00
  %7434 = call float @llvm.amdgcn.rcp.f32(float %7433)
  %7435 = fmul float %7426, %7434
  %7436 = fmul float %7435, %7430
  %7437 = add i64 %7366, %6921
  %7438 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7437
  %7439 = load <1 x float>, ptr addrspace(3) %7438, align 4
  %7440 = extractelement <1 x float> %7439, i64 0
  %7441 = add i64 %7437, 128
  %7442 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7441
  %7443 = load <1 x float>, ptr addrspace(3) %7442, align 4
  %7444 = extractelement <1 x float> %7443, i64 0
  %7445 = fmul float %7440, 0xBFF7154760000000
  %7446 = call float @llvm.amdgcn.exp2.f32(float %7445)
  %7447 = fadd float %7446, 1.000000e+00
  %7448 = call float @llvm.amdgcn.rcp.f32(float %7447)
  %7449 = fmul float %7440, %7448
  %7450 = fmul float %7449, %7444
  %7451 = add i64 %7366, %6936
  %7452 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7451
  %7453 = load <1 x float>, ptr addrspace(3) %7452, align 4
  %7454 = extractelement <1 x float> %7453, i64 0
  %7455 = add i64 %7451, 128
  %7456 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7455
  %7457 = load <1 x float>, ptr addrspace(3) %7456, align 4
  %7458 = extractelement <1 x float> %7457, i64 0
  %7459 = fmul float %7454, 0xBFF7154760000000
  %7460 = call float @llvm.amdgcn.exp2.f32(float %7459)
  %7461 = fadd float %7460, 1.000000e+00
  %7462 = call float @llvm.amdgcn.rcp.f32(float %7461)
  %7463 = fmul float %7454, %7462
  %7464 = fmul float %7463, %7458
  %7465 = add i64 %7366, %6951
  %7466 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7465
  %7467 = load <1 x float>, ptr addrspace(3) %7466, align 4
  %7468 = extractelement <1 x float> %7467, i64 0
  %7469 = add i64 %7465, 128
  %7470 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7469
  %7471 = load <1 x float>, ptr addrspace(3) %7470, align 4
  %7472 = extractelement <1 x float> %7471, i64 0
  %7473 = fmul float %7468, 0xBFF7154760000000
  %7474 = call float @llvm.amdgcn.exp2.f32(float %7473)
  %7475 = fadd float %7474, 1.000000e+00
  %7476 = call float @llvm.amdgcn.rcp.f32(float %7475)
  %7477 = fmul float %7468, %7476
  %7478 = fmul float %7477, %7472
  %7479 = bitcast float %7380 to i32
  %7480 = and i32 %7479, 2147483647
  %7481 = bitcast i32 %7480 to float
  %7482 = bitcast float %7394 to i32
  %7483 = and i32 %7482, 2147483647
  %7484 = bitcast i32 %7483 to float
  %7485 = call float @llvm.maximum.f32(float %7481, float %7484)
  %7486 = bitcast float %7408 to i32
  %7487 = and i32 %7486, 2147483647
  %7488 = bitcast i32 %7487 to float
  %7489 = call float @llvm.maximum.f32(float %7485, float %7488)
  %7490 = bitcast float %7422 to i32
  %7491 = and i32 %7490, 2147483647
  %7492 = bitcast i32 %7491 to float
  %7493 = call float @llvm.maximum.f32(float %7489, float %7492)
  %7494 = bitcast float %7436 to i32
  %7495 = and i32 %7494, 2147483647
  %7496 = bitcast i32 %7495 to float
  %7497 = call float @llvm.maximum.f32(float %7493, float %7496)
  %7498 = bitcast float %7450 to i32
  %7499 = and i32 %7498, 2147483647
  %7500 = bitcast i32 %7499 to float
  %7501 = call float @llvm.maximum.f32(float %7497, float %7500)
  %7502 = bitcast float %7464 to i32
  %7503 = and i32 %7502, 2147483647
  %7504 = bitcast i32 %7503 to float
  %7505 = call float @llvm.maximum.f32(float %7501, float %7504)
  %7506 = bitcast float %7478 to i32
  %7507 = and i32 %7506, 2147483647
  %7508 = bitcast i32 %7507 to float
  %7509 = call float @llvm.maximum.f32(float %7505, float %7508)
  %7510 = bitcast float %7509 to i32
  %7511 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %7510, i32 %7510, i32 177, i32 15, i32 15, i1 true)
  %7512 = bitcast i32 %7511 to float
  %7513 = call float @llvm.maximum.f32(float %7509, float %7512)
  %7514 = bitcast float %7513 to i32
  %7515 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %7514, i32 %7514, i32 78, i32 15, i32 15, i1 true)
  %7516 = bitcast i32 %7515 to float
  %7517 = call float @llvm.maximum.f32(float %7513, float %7516)
  %7518 = bitcast float %7517 to i32
  %7519 = add i32 %7518, 2097152
  %7520 = bitcast i32 %7519 to float
  %7521 = fmul float %7520, 2.500000e-01
  %7522 = bitcast float %7521 to i32
  %7523 = lshr i32 %7522, 23
  %7524 = call i32 @llvm.smin.i32(i32 %7523, i32 254)
  %7525 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 0, float %7380, float %7394, float %7521, i32 0)
  %7526 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %7525, float %7408, float %7422, float %7521, i32 1)
  %7527 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %7526, float %7436, float %7450, float %7521, i32 2)
  %7528 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %7527, float %7464, float %7478, float %7521, i32 3)
  %7529 = add i64 %46, %7365
  %7530 = mul i64 %7529, 256
  %7531 = add i64 %7530, %7020
  %7532 = add i64 %7024, %7531
  %7533 = inttoptr i64 %7532 to ptr addrspace(1)
  store i32 %7528, ptr addrspace(1) %7533, align 4
  %7534 = add i64 %6839, 64
  %7535 = mul i64 %7534, 256
  %7536 = add i64 %7535, %6845
  %7537 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7536
  %7538 = load <1 x float>, ptr addrspace(3) %7537, align 4
  %7539 = extractelement <1 x float> %7538, i64 0
  %7540 = add i64 %7536, 128
  %7541 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7540
  %7542 = load <1 x float>, ptr addrspace(3) %7541, align 4
  %7543 = extractelement <1 x float> %7542, i64 0
  %7544 = fmul float %7539, 0xBFF7154760000000
  %7545 = call float @llvm.amdgcn.exp2.f32(float %7544)
  %7546 = fadd float %7545, 1.000000e+00
  %7547 = call float @llvm.amdgcn.rcp.f32(float %7546)
  %7548 = fmul float %7539, %7547
  %7549 = fmul float %7548, %7543
  %7550 = add i64 %7535, %6861
  %7551 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7550
  %7552 = load <1 x float>, ptr addrspace(3) %7551, align 4
  %7553 = extractelement <1 x float> %7552, i64 0
  %7554 = add i64 %7550, 128
  %7555 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7554
  %7556 = load <1 x float>, ptr addrspace(3) %7555, align 4
  %7557 = extractelement <1 x float> %7556, i64 0
  %7558 = fmul float %7553, 0xBFF7154760000000
  %7559 = call float @llvm.amdgcn.exp2.f32(float %7558)
  %7560 = fadd float %7559, 1.000000e+00
  %7561 = call float @llvm.amdgcn.rcp.f32(float %7560)
  %7562 = fmul float %7553, %7561
  %7563 = fmul float %7562, %7557
  %7564 = add i64 %7535, %6876
  %7565 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7564
  %7566 = load <1 x float>, ptr addrspace(3) %7565, align 4
  %7567 = extractelement <1 x float> %7566, i64 0
  %7568 = add i64 %7564, 128
  %7569 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7568
  %7570 = load <1 x float>, ptr addrspace(3) %7569, align 4
  %7571 = extractelement <1 x float> %7570, i64 0
  %7572 = fmul float %7567, 0xBFF7154760000000
  %7573 = call float @llvm.amdgcn.exp2.f32(float %7572)
  %7574 = fadd float %7573, 1.000000e+00
  %7575 = call float @llvm.amdgcn.rcp.f32(float %7574)
  %7576 = fmul float %7567, %7575
  %7577 = fmul float %7576, %7571
  %7578 = add i64 %7535, %6891
  %7579 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7578
  %7580 = load <1 x float>, ptr addrspace(3) %7579, align 4
  %7581 = extractelement <1 x float> %7580, i64 0
  %7582 = add i64 %7578, 128
  %7583 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7582
  %7584 = load <1 x float>, ptr addrspace(3) %7583, align 4
  %7585 = extractelement <1 x float> %7584, i64 0
  %7586 = fmul float %7581, 0xBFF7154760000000
  %7587 = call float @llvm.amdgcn.exp2.f32(float %7586)
  %7588 = fadd float %7587, 1.000000e+00
  %7589 = call float @llvm.amdgcn.rcp.f32(float %7588)
  %7590 = fmul float %7581, %7589
  %7591 = fmul float %7590, %7585
  %7592 = add i64 %7535, %6906
  %7593 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7592
  %7594 = load <1 x float>, ptr addrspace(3) %7593, align 4
  %7595 = extractelement <1 x float> %7594, i64 0
  %7596 = add i64 %7592, 128
  %7597 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7596
  %7598 = load <1 x float>, ptr addrspace(3) %7597, align 4
  %7599 = extractelement <1 x float> %7598, i64 0
  %7600 = fmul float %7595, 0xBFF7154760000000
  %7601 = call float @llvm.amdgcn.exp2.f32(float %7600)
  %7602 = fadd float %7601, 1.000000e+00
  %7603 = call float @llvm.amdgcn.rcp.f32(float %7602)
  %7604 = fmul float %7595, %7603
  %7605 = fmul float %7604, %7599
  %7606 = add i64 %7535, %6921
  %7607 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7606
  %7608 = load <1 x float>, ptr addrspace(3) %7607, align 4
  %7609 = extractelement <1 x float> %7608, i64 0
  %7610 = add i64 %7606, 128
  %7611 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7610
  %7612 = load <1 x float>, ptr addrspace(3) %7611, align 4
  %7613 = extractelement <1 x float> %7612, i64 0
  %7614 = fmul float %7609, 0xBFF7154760000000
  %7615 = call float @llvm.amdgcn.exp2.f32(float %7614)
  %7616 = fadd float %7615, 1.000000e+00
  %7617 = call float @llvm.amdgcn.rcp.f32(float %7616)
  %7618 = fmul float %7609, %7617
  %7619 = fmul float %7618, %7613
  %7620 = add i64 %7535, %6936
  %7621 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7620
  %7622 = load <1 x float>, ptr addrspace(3) %7621, align 4
  %7623 = extractelement <1 x float> %7622, i64 0
  %7624 = add i64 %7620, 128
  %7625 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7624
  %7626 = load <1 x float>, ptr addrspace(3) %7625, align 4
  %7627 = extractelement <1 x float> %7626, i64 0
  %7628 = fmul float %7623, 0xBFF7154760000000
  %7629 = call float @llvm.amdgcn.exp2.f32(float %7628)
  %7630 = fadd float %7629, 1.000000e+00
  %7631 = call float @llvm.amdgcn.rcp.f32(float %7630)
  %7632 = fmul float %7623, %7631
  %7633 = fmul float %7632, %7627
  %7634 = add i64 %7535, %6951
  %7635 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7634
  %7636 = load <1 x float>, ptr addrspace(3) %7635, align 4
  %7637 = extractelement <1 x float> %7636, i64 0
  %7638 = add i64 %7634, 128
  %7639 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7638
  %7640 = load <1 x float>, ptr addrspace(3) %7639, align 4
  %7641 = extractelement <1 x float> %7640, i64 0
  %7642 = fmul float %7637, 0xBFF7154760000000
  %7643 = call float @llvm.amdgcn.exp2.f32(float %7642)
  %7644 = fadd float %7643, 1.000000e+00
  %7645 = call float @llvm.amdgcn.rcp.f32(float %7644)
  %7646 = fmul float %7637, %7645
  %7647 = fmul float %7646, %7641
  %7648 = bitcast float %7549 to i32
  %7649 = and i32 %7648, 2147483647
  %7650 = bitcast i32 %7649 to float
  %7651 = bitcast float %7563 to i32
  %7652 = and i32 %7651, 2147483647
  %7653 = bitcast i32 %7652 to float
  %7654 = call float @llvm.maximum.f32(float %7650, float %7653)
  %7655 = bitcast float %7577 to i32
  %7656 = and i32 %7655, 2147483647
  %7657 = bitcast i32 %7656 to float
  %7658 = call float @llvm.maximum.f32(float %7654, float %7657)
  %7659 = bitcast float %7591 to i32
  %7660 = and i32 %7659, 2147483647
  %7661 = bitcast i32 %7660 to float
  %7662 = call float @llvm.maximum.f32(float %7658, float %7661)
  %7663 = bitcast float %7605 to i32
  %7664 = and i32 %7663, 2147483647
  %7665 = bitcast i32 %7664 to float
  %7666 = call float @llvm.maximum.f32(float %7662, float %7665)
  %7667 = bitcast float %7619 to i32
  %7668 = and i32 %7667, 2147483647
  %7669 = bitcast i32 %7668 to float
  %7670 = call float @llvm.maximum.f32(float %7666, float %7669)
  %7671 = bitcast float %7633 to i32
  %7672 = and i32 %7671, 2147483647
  %7673 = bitcast i32 %7672 to float
  %7674 = call float @llvm.maximum.f32(float %7670, float %7673)
  %7675 = bitcast float %7647 to i32
  %7676 = and i32 %7675, 2147483647
  %7677 = bitcast i32 %7676 to float
  %7678 = call float @llvm.maximum.f32(float %7674, float %7677)
  %7679 = bitcast float %7678 to i32
  %7680 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %7679, i32 %7679, i32 177, i32 15, i32 15, i1 true)
  %7681 = bitcast i32 %7680 to float
  %7682 = call float @llvm.maximum.f32(float %7678, float %7681)
  %7683 = bitcast float %7682 to i32
  %7684 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %7683, i32 %7683, i32 78, i32 15, i32 15, i1 true)
  %7685 = bitcast i32 %7684 to float
  %7686 = call float @llvm.maximum.f32(float %7682, float %7685)
  %7687 = bitcast float %7686 to i32
  %7688 = add i32 %7687, 2097152
  %7689 = bitcast i32 %7688 to float
  %7690 = fmul float %7689, 2.500000e-01
  %7691 = bitcast float %7690 to i32
  %7692 = lshr i32 %7691, 23
  %7693 = call i32 @llvm.smin.i32(i32 %7692, i32 254)
  %7694 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 0, float %7549, float %7563, float %7690, i32 0)
  %7695 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %7694, float %7577, float %7591, float %7690, i32 1)
  %7696 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %7695, float %7605, float %7619, float %7690, i32 2)
  %7697 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %7696, float %7633, float %7647, float %7690, i32 3)
  %7698 = add i64 %46, %7534
  %7699 = mul i64 %7698, 256
  %7700 = add i64 %7699, %7020
  %7701 = add i64 %7024, %7700
  %7702 = inttoptr i64 %7701 to ptr addrspace(1)
  store i32 %7697, ptr addrspace(1) %7702, align 4
  %7703 = add i64 %6839, 80
  %7704 = mul i64 %7703, 256
  %7705 = add i64 %7704, %6845
  %7706 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7705
  %7707 = load <1 x float>, ptr addrspace(3) %7706, align 4
  %7708 = extractelement <1 x float> %7707, i64 0
  %7709 = add i64 %7705, 128
  %7710 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7709
  %7711 = load <1 x float>, ptr addrspace(3) %7710, align 4
  %7712 = extractelement <1 x float> %7711, i64 0
  %7713 = fmul float %7708, 0xBFF7154760000000
  %7714 = call float @llvm.amdgcn.exp2.f32(float %7713)
  %7715 = fadd float %7714, 1.000000e+00
  %7716 = call float @llvm.amdgcn.rcp.f32(float %7715)
  %7717 = fmul float %7708, %7716
  %7718 = fmul float %7717, %7712
  %7719 = add i64 %7704, %6861
  %7720 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7719
  %7721 = load <1 x float>, ptr addrspace(3) %7720, align 4
  %7722 = extractelement <1 x float> %7721, i64 0
  %7723 = add i64 %7719, 128
  %7724 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7723
  %7725 = load <1 x float>, ptr addrspace(3) %7724, align 4
  %7726 = extractelement <1 x float> %7725, i64 0
  %7727 = fmul float %7722, 0xBFF7154760000000
  %7728 = call float @llvm.amdgcn.exp2.f32(float %7727)
  %7729 = fadd float %7728, 1.000000e+00
  %7730 = call float @llvm.amdgcn.rcp.f32(float %7729)
  %7731 = fmul float %7722, %7730
  %7732 = fmul float %7731, %7726
  %7733 = add i64 %7704, %6876
  %7734 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7733
  %7735 = load <1 x float>, ptr addrspace(3) %7734, align 4
  %7736 = extractelement <1 x float> %7735, i64 0
  %7737 = add i64 %7733, 128
  %7738 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7737
  %7739 = load <1 x float>, ptr addrspace(3) %7738, align 4
  %7740 = extractelement <1 x float> %7739, i64 0
  %7741 = fmul float %7736, 0xBFF7154760000000
  %7742 = call float @llvm.amdgcn.exp2.f32(float %7741)
  %7743 = fadd float %7742, 1.000000e+00
  %7744 = call float @llvm.amdgcn.rcp.f32(float %7743)
  %7745 = fmul float %7736, %7744
  %7746 = fmul float %7745, %7740
  %7747 = add i64 %7704, %6891
  %7748 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7747
  %7749 = load <1 x float>, ptr addrspace(3) %7748, align 4
  %7750 = extractelement <1 x float> %7749, i64 0
  %7751 = add i64 %7747, 128
  %7752 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7751
  %7753 = load <1 x float>, ptr addrspace(3) %7752, align 4
  %7754 = extractelement <1 x float> %7753, i64 0
  %7755 = fmul float %7750, 0xBFF7154760000000
  %7756 = call float @llvm.amdgcn.exp2.f32(float %7755)
  %7757 = fadd float %7756, 1.000000e+00
  %7758 = call float @llvm.amdgcn.rcp.f32(float %7757)
  %7759 = fmul float %7750, %7758
  %7760 = fmul float %7759, %7754
  %7761 = add i64 %7704, %6906
  %7762 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7761
  %7763 = load <1 x float>, ptr addrspace(3) %7762, align 4
  %7764 = extractelement <1 x float> %7763, i64 0
  %7765 = add i64 %7761, 128
  %7766 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7765
  %7767 = load <1 x float>, ptr addrspace(3) %7766, align 4
  %7768 = extractelement <1 x float> %7767, i64 0
  %7769 = fmul float %7764, 0xBFF7154760000000
  %7770 = call float @llvm.amdgcn.exp2.f32(float %7769)
  %7771 = fadd float %7770, 1.000000e+00
  %7772 = call float @llvm.amdgcn.rcp.f32(float %7771)
  %7773 = fmul float %7764, %7772
  %7774 = fmul float %7773, %7768
  %7775 = add i64 %7704, %6921
  %7776 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7775
  %7777 = load <1 x float>, ptr addrspace(3) %7776, align 4
  %7778 = extractelement <1 x float> %7777, i64 0
  %7779 = add i64 %7775, 128
  %7780 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7779
  %7781 = load <1 x float>, ptr addrspace(3) %7780, align 4
  %7782 = extractelement <1 x float> %7781, i64 0
  %7783 = fmul float %7778, 0xBFF7154760000000
  %7784 = call float @llvm.amdgcn.exp2.f32(float %7783)
  %7785 = fadd float %7784, 1.000000e+00
  %7786 = call float @llvm.amdgcn.rcp.f32(float %7785)
  %7787 = fmul float %7778, %7786
  %7788 = fmul float %7787, %7782
  %7789 = add i64 %7704, %6936
  %7790 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7789
  %7791 = load <1 x float>, ptr addrspace(3) %7790, align 4
  %7792 = extractelement <1 x float> %7791, i64 0
  %7793 = add i64 %7789, 128
  %7794 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7793
  %7795 = load <1 x float>, ptr addrspace(3) %7794, align 4
  %7796 = extractelement <1 x float> %7795, i64 0
  %7797 = fmul float %7792, 0xBFF7154760000000
  %7798 = call float @llvm.amdgcn.exp2.f32(float %7797)
  %7799 = fadd float %7798, 1.000000e+00
  %7800 = call float @llvm.amdgcn.rcp.f32(float %7799)
  %7801 = fmul float %7792, %7800
  %7802 = fmul float %7801, %7796
  %7803 = add i64 %7704, %6951
  %7804 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7803
  %7805 = load <1 x float>, ptr addrspace(3) %7804, align 4
  %7806 = extractelement <1 x float> %7805, i64 0
  %7807 = add i64 %7803, 128
  %7808 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7807
  %7809 = load <1 x float>, ptr addrspace(3) %7808, align 4
  %7810 = extractelement <1 x float> %7809, i64 0
  %7811 = fmul float %7806, 0xBFF7154760000000
  %7812 = call float @llvm.amdgcn.exp2.f32(float %7811)
  %7813 = fadd float %7812, 1.000000e+00
  %7814 = call float @llvm.amdgcn.rcp.f32(float %7813)
  %7815 = fmul float %7806, %7814
  %7816 = fmul float %7815, %7810
  %7817 = bitcast float %7718 to i32
  %7818 = and i32 %7817, 2147483647
  %7819 = bitcast i32 %7818 to float
  %7820 = bitcast float %7732 to i32
  %7821 = and i32 %7820, 2147483647
  %7822 = bitcast i32 %7821 to float
  %7823 = call float @llvm.maximum.f32(float %7819, float %7822)
  %7824 = bitcast float %7746 to i32
  %7825 = and i32 %7824, 2147483647
  %7826 = bitcast i32 %7825 to float
  %7827 = call float @llvm.maximum.f32(float %7823, float %7826)
  %7828 = bitcast float %7760 to i32
  %7829 = and i32 %7828, 2147483647
  %7830 = bitcast i32 %7829 to float
  %7831 = call float @llvm.maximum.f32(float %7827, float %7830)
  %7832 = bitcast float %7774 to i32
  %7833 = and i32 %7832, 2147483647
  %7834 = bitcast i32 %7833 to float
  %7835 = call float @llvm.maximum.f32(float %7831, float %7834)
  %7836 = bitcast float %7788 to i32
  %7837 = and i32 %7836, 2147483647
  %7838 = bitcast i32 %7837 to float
  %7839 = call float @llvm.maximum.f32(float %7835, float %7838)
  %7840 = bitcast float %7802 to i32
  %7841 = and i32 %7840, 2147483647
  %7842 = bitcast i32 %7841 to float
  %7843 = call float @llvm.maximum.f32(float %7839, float %7842)
  %7844 = bitcast float %7816 to i32
  %7845 = and i32 %7844, 2147483647
  %7846 = bitcast i32 %7845 to float
  %7847 = call float @llvm.maximum.f32(float %7843, float %7846)
  %7848 = bitcast float %7847 to i32
  %7849 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %7848, i32 %7848, i32 177, i32 15, i32 15, i1 true)
  %7850 = bitcast i32 %7849 to float
  %7851 = call float @llvm.maximum.f32(float %7847, float %7850)
  %7852 = bitcast float %7851 to i32
  %7853 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %7852, i32 %7852, i32 78, i32 15, i32 15, i1 true)
  %7854 = bitcast i32 %7853 to float
  %7855 = call float @llvm.maximum.f32(float %7851, float %7854)
  %7856 = bitcast float %7855 to i32
  %7857 = add i32 %7856, 2097152
  %7858 = bitcast i32 %7857 to float
  %7859 = fmul float %7858, 2.500000e-01
  %7860 = bitcast float %7859 to i32
  %7861 = lshr i32 %7860, 23
  %7862 = call i32 @llvm.smin.i32(i32 %7861, i32 254)
  %7863 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 0, float %7718, float %7732, float %7859, i32 0)
  %7864 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %7863, float %7746, float %7760, float %7859, i32 1)
  %7865 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %7864, float %7774, float %7788, float %7859, i32 2)
  %7866 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %7865, float %7802, float %7816, float %7859, i32 3)
  %7867 = add i64 %46, %7703
  %7868 = mul i64 %7867, 256
  %7869 = add i64 %7868, %7020
  %7870 = add i64 %7024, %7869
  %7871 = inttoptr i64 %7870 to ptr addrspace(1)
  store i32 %7866, ptr addrspace(1) %7871, align 4
  %7872 = add i64 %6839, 96
  %7873 = mul i64 %7872, 256
  %7874 = add i64 %7873, %6845
  %7875 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7874
  %7876 = load <1 x float>, ptr addrspace(3) %7875, align 4
  %7877 = extractelement <1 x float> %7876, i64 0
  %7878 = add i64 %7874, 128
  %7879 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7878
  %7880 = load <1 x float>, ptr addrspace(3) %7879, align 4
  %7881 = extractelement <1 x float> %7880, i64 0
  %7882 = fmul float %7877, 0xBFF7154760000000
  %7883 = call float @llvm.amdgcn.exp2.f32(float %7882)
  %7884 = fadd float %7883, 1.000000e+00
  %7885 = call float @llvm.amdgcn.rcp.f32(float %7884)
  %7886 = fmul float %7877, %7885
  %7887 = fmul float %7886, %7881
  %7888 = add i64 %7873, %6861
  %7889 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7888
  %7890 = load <1 x float>, ptr addrspace(3) %7889, align 4
  %7891 = extractelement <1 x float> %7890, i64 0
  %7892 = add i64 %7888, 128
  %7893 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7892
  %7894 = load <1 x float>, ptr addrspace(3) %7893, align 4
  %7895 = extractelement <1 x float> %7894, i64 0
  %7896 = fmul float %7891, 0xBFF7154760000000
  %7897 = call float @llvm.amdgcn.exp2.f32(float %7896)
  %7898 = fadd float %7897, 1.000000e+00
  %7899 = call float @llvm.amdgcn.rcp.f32(float %7898)
  %7900 = fmul float %7891, %7899
  %7901 = fmul float %7900, %7895
  %7902 = add i64 %7873, %6876
  %7903 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7902
  %7904 = load <1 x float>, ptr addrspace(3) %7903, align 4
  %7905 = extractelement <1 x float> %7904, i64 0
  %7906 = add i64 %7902, 128
  %7907 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7906
  %7908 = load <1 x float>, ptr addrspace(3) %7907, align 4
  %7909 = extractelement <1 x float> %7908, i64 0
  %7910 = fmul float %7905, 0xBFF7154760000000
  %7911 = call float @llvm.amdgcn.exp2.f32(float %7910)
  %7912 = fadd float %7911, 1.000000e+00
  %7913 = call float @llvm.amdgcn.rcp.f32(float %7912)
  %7914 = fmul float %7905, %7913
  %7915 = fmul float %7914, %7909
  %7916 = add i64 %7873, %6891
  %7917 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7916
  %7918 = load <1 x float>, ptr addrspace(3) %7917, align 4
  %7919 = extractelement <1 x float> %7918, i64 0
  %7920 = add i64 %7916, 128
  %7921 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7920
  %7922 = load <1 x float>, ptr addrspace(3) %7921, align 4
  %7923 = extractelement <1 x float> %7922, i64 0
  %7924 = fmul float %7919, 0xBFF7154760000000
  %7925 = call float @llvm.amdgcn.exp2.f32(float %7924)
  %7926 = fadd float %7925, 1.000000e+00
  %7927 = call float @llvm.amdgcn.rcp.f32(float %7926)
  %7928 = fmul float %7919, %7927
  %7929 = fmul float %7928, %7923
  %7930 = add i64 %7873, %6906
  %7931 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7930
  %7932 = load <1 x float>, ptr addrspace(3) %7931, align 4
  %7933 = extractelement <1 x float> %7932, i64 0
  %7934 = add i64 %7930, 128
  %7935 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7934
  %7936 = load <1 x float>, ptr addrspace(3) %7935, align 4
  %7937 = extractelement <1 x float> %7936, i64 0
  %7938 = fmul float %7933, 0xBFF7154760000000
  %7939 = call float @llvm.amdgcn.exp2.f32(float %7938)
  %7940 = fadd float %7939, 1.000000e+00
  %7941 = call float @llvm.amdgcn.rcp.f32(float %7940)
  %7942 = fmul float %7933, %7941
  %7943 = fmul float %7942, %7937
  %7944 = add i64 %7873, %6921
  %7945 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7944
  %7946 = load <1 x float>, ptr addrspace(3) %7945, align 4
  %7947 = extractelement <1 x float> %7946, i64 0
  %7948 = add i64 %7944, 128
  %7949 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7948
  %7950 = load <1 x float>, ptr addrspace(3) %7949, align 4
  %7951 = extractelement <1 x float> %7950, i64 0
  %7952 = fmul float %7947, 0xBFF7154760000000
  %7953 = call float @llvm.amdgcn.exp2.f32(float %7952)
  %7954 = fadd float %7953, 1.000000e+00
  %7955 = call float @llvm.amdgcn.rcp.f32(float %7954)
  %7956 = fmul float %7947, %7955
  %7957 = fmul float %7956, %7951
  %7958 = add i64 %7873, %6936
  %7959 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7958
  %7960 = load <1 x float>, ptr addrspace(3) %7959, align 4
  %7961 = extractelement <1 x float> %7960, i64 0
  %7962 = add i64 %7958, 128
  %7963 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7962
  %7964 = load <1 x float>, ptr addrspace(3) %7963, align 4
  %7965 = extractelement <1 x float> %7964, i64 0
  %7966 = fmul float %7961, 0xBFF7154760000000
  %7967 = call float @llvm.amdgcn.exp2.f32(float %7966)
  %7968 = fadd float %7967, 1.000000e+00
  %7969 = call float @llvm.amdgcn.rcp.f32(float %7968)
  %7970 = fmul float %7961, %7969
  %7971 = fmul float %7970, %7965
  %7972 = add i64 %7873, %6951
  %7973 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7972
  %7974 = load <1 x float>, ptr addrspace(3) %7973, align 4
  %7975 = extractelement <1 x float> %7974, i64 0
  %7976 = add i64 %7972, 128
  %7977 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %7976
  %7978 = load <1 x float>, ptr addrspace(3) %7977, align 4
  %7979 = extractelement <1 x float> %7978, i64 0
  %7980 = fmul float %7975, 0xBFF7154760000000
  %7981 = call float @llvm.amdgcn.exp2.f32(float %7980)
  %7982 = fadd float %7981, 1.000000e+00
  %7983 = call float @llvm.amdgcn.rcp.f32(float %7982)
  %7984 = fmul float %7975, %7983
  %7985 = fmul float %7984, %7979
  %7986 = bitcast float %7887 to i32
  %7987 = and i32 %7986, 2147483647
  %7988 = bitcast i32 %7987 to float
  %7989 = bitcast float %7901 to i32
  %7990 = and i32 %7989, 2147483647
  %7991 = bitcast i32 %7990 to float
  %7992 = call float @llvm.maximum.f32(float %7988, float %7991)
  %7993 = bitcast float %7915 to i32
  %7994 = and i32 %7993, 2147483647
  %7995 = bitcast i32 %7994 to float
  %7996 = call float @llvm.maximum.f32(float %7992, float %7995)
  %7997 = bitcast float %7929 to i32
  %7998 = and i32 %7997, 2147483647
  %7999 = bitcast i32 %7998 to float
  %8000 = call float @llvm.maximum.f32(float %7996, float %7999)
  %8001 = bitcast float %7943 to i32
  %8002 = and i32 %8001, 2147483647
  %8003 = bitcast i32 %8002 to float
  %8004 = call float @llvm.maximum.f32(float %8000, float %8003)
  %8005 = bitcast float %7957 to i32
  %8006 = and i32 %8005, 2147483647
  %8007 = bitcast i32 %8006 to float
  %8008 = call float @llvm.maximum.f32(float %8004, float %8007)
  %8009 = bitcast float %7971 to i32
  %8010 = and i32 %8009, 2147483647
  %8011 = bitcast i32 %8010 to float
  %8012 = call float @llvm.maximum.f32(float %8008, float %8011)
  %8013 = bitcast float %7985 to i32
  %8014 = and i32 %8013, 2147483647
  %8015 = bitcast i32 %8014 to float
  %8016 = call float @llvm.maximum.f32(float %8012, float %8015)
  %8017 = bitcast float %8016 to i32
  %8018 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %8017, i32 %8017, i32 177, i32 15, i32 15, i1 true)
  %8019 = bitcast i32 %8018 to float
  %8020 = call float @llvm.maximum.f32(float %8016, float %8019)
  %8021 = bitcast float %8020 to i32
  %8022 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %8021, i32 %8021, i32 78, i32 15, i32 15, i1 true)
  %8023 = bitcast i32 %8022 to float
  %8024 = call float @llvm.maximum.f32(float %8020, float %8023)
  %8025 = bitcast float %8024 to i32
  %8026 = add i32 %8025, 2097152
  %8027 = bitcast i32 %8026 to float
  %8028 = fmul float %8027, 2.500000e-01
  %8029 = bitcast float %8028 to i32
  %8030 = lshr i32 %8029, 23
  %8031 = call i32 @llvm.smin.i32(i32 %8030, i32 254)
  %8032 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 0, float %7887, float %7901, float %8028, i32 0)
  %8033 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %8032, float %7915, float %7929, float %8028, i32 1)
  %8034 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %8033, float %7943, float %7957, float %8028, i32 2)
  %8035 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %8034, float %7971, float %7985, float %8028, i32 3)
  %8036 = add i64 %46, %7872
  %8037 = mul i64 %8036, 256
  %8038 = add i64 %8037, %7020
  %8039 = add i64 %7024, %8038
  %8040 = inttoptr i64 %8039 to ptr addrspace(1)
  store i32 %8035, ptr addrspace(1) %8040, align 4
  %8041 = add i64 %6839, 112
  %8042 = mul i64 %8041, 256
  %8043 = add i64 %8042, %6845
  %8044 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %8043
  %8045 = load <1 x float>, ptr addrspace(3) %8044, align 4
  %8046 = extractelement <1 x float> %8045, i64 0
  %8047 = add i64 %8043, 128
  %8048 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %8047
  %8049 = load <1 x float>, ptr addrspace(3) %8048, align 4
  %8050 = extractelement <1 x float> %8049, i64 0
  %8051 = fmul float %8046, 0xBFF7154760000000
  %8052 = call float @llvm.amdgcn.exp2.f32(float %8051)
  %8053 = fadd float %8052, 1.000000e+00
  %8054 = call float @llvm.amdgcn.rcp.f32(float %8053)
  %8055 = fmul float %8046, %8054
  %8056 = fmul float %8055, %8050
  %8057 = add i64 %8042, %6861
  %8058 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %8057
  %8059 = load <1 x float>, ptr addrspace(3) %8058, align 4
  %8060 = extractelement <1 x float> %8059, i64 0
  %8061 = add i64 %8057, 128
  %8062 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %8061
  %8063 = load <1 x float>, ptr addrspace(3) %8062, align 4
  %8064 = extractelement <1 x float> %8063, i64 0
  %8065 = fmul float %8060, 0xBFF7154760000000
  %8066 = call float @llvm.amdgcn.exp2.f32(float %8065)
  %8067 = fadd float %8066, 1.000000e+00
  %8068 = call float @llvm.amdgcn.rcp.f32(float %8067)
  %8069 = fmul float %8060, %8068
  %8070 = fmul float %8069, %8064
  %8071 = add i64 %8042, %6876
  %8072 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %8071
  %8073 = load <1 x float>, ptr addrspace(3) %8072, align 4
  %8074 = extractelement <1 x float> %8073, i64 0
  %8075 = add i64 %8071, 128
  %8076 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %8075
  %8077 = load <1 x float>, ptr addrspace(3) %8076, align 4
  %8078 = extractelement <1 x float> %8077, i64 0
  %8079 = fmul float %8074, 0xBFF7154760000000
  %8080 = call float @llvm.amdgcn.exp2.f32(float %8079)
  %8081 = fadd float %8080, 1.000000e+00
  %8082 = call float @llvm.amdgcn.rcp.f32(float %8081)
  %8083 = fmul float %8074, %8082
  %8084 = fmul float %8083, %8078
  %8085 = add i64 %8042, %6891
  %8086 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %8085
  %8087 = load <1 x float>, ptr addrspace(3) %8086, align 4
  %8088 = extractelement <1 x float> %8087, i64 0
  %8089 = add i64 %8085, 128
  %8090 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %8089
  %8091 = load <1 x float>, ptr addrspace(3) %8090, align 4
  %8092 = extractelement <1 x float> %8091, i64 0
  %8093 = fmul float %8088, 0xBFF7154760000000
  %8094 = call float @llvm.amdgcn.exp2.f32(float %8093)
  %8095 = fadd float %8094, 1.000000e+00
  %8096 = call float @llvm.amdgcn.rcp.f32(float %8095)
  %8097 = fmul float %8088, %8096
  %8098 = fmul float %8097, %8092
  %8099 = add i64 %8042, %6906
  %8100 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %8099
  %8101 = load <1 x float>, ptr addrspace(3) %8100, align 4
  %8102 = extractelement <1 x float> %8101, i64 0
  %8103 = add i64 %8099, 128
  %8104 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %8103
  %8105 = load <1 x float>, ptr addrspace(3) %8104, align 4
  %8106 = extractelement <1 x float> %8105, i64 0
  %8107 = fmul float %8102, 0xBFF7154760000000
  %8108 = call float @llvm.amdgcn.exp2.f32(float %8107)
  %8109 = fadd float %8108, 1.000000e+00
  %8110 = call float @llvm.amdgcn.rcp.f32(float %8109)
  %8111 = fmul float %8102, %8110
  %8112 = fmul float %8111, %8106
  %8113 = add i64 %8042, %6921
  %8114 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %8113
  %8115 = load <1 x float>, ptr addrspace(3) %8114, align 4
  %8116 = extractelement <1 x float> %8115, i64 0
  %8117 = add i64 %8113, 128
  %8118 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %8117
  %8119 = load <1 x float>, ptr addrspace(3) %8118, align 4
  %8120 = extractelement <1 x float> %8119, i64 0
  %8121 = fmul float %8116, 0xBFF7154760000000
  %8122 = call float @llvm.amdgcn.exp2.f32(float %8121)
  %8123 = fadd float %8122, 1.000000e+00
  %8124 = call float @llvm.amdgcn.rcp.f32(float %8123)
  %8125 = fmul float %8116, %8124
  %8126 = fmul float %8125, %8120
  %8127 = add i64 %8042, %6936
  %8128 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %8127
  %8129 = load <1 x float>, ptr addrspace(3) %8128, align 4
  %8130 = extractelement <1 x float> %8129, i64 0
  %8131 = add i64 %8127, 128
  %8132 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %8131
  %8133 = load <1 x float>, ptr addrspace(3) %8132, align 4
  %8134 = extractelement <1 x float> %8133, i64 0
  %8135 = fmul float %8130, 0xBFF7154760000000
  %8136 = call float @llvm.amdgcn.exp2.f32(float %8135)
  %8137 = fadd float %8136, 1.000000e+00
  %8138 = call float @llvm.amdgcn.rcp.f32(float %8137)
  %8139 = fmul float %8130, %8138
  %8140 = fmul float %8139, %8134
  %8141 = add i64 %8042, %6951
  %8142 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %8141
  %8143 = load <1 x float>, ptr addrspace(3) %8142, align 4
  %8144 = extractelement <1 x float> %8143, i64 0
  %8145 = add i64 %8141, 128
  %8146 = getelementptr float, ptr addrspace(3) @smem_g1, i64 %8145
  %8147 = load <1 x float>, ptr addrspace(3) %8146, align 4
  %8148 = extractelement <1 x float> %8147, i64 0
  %8149 = fmul float %8144, 0xBFF7154760000000
  %8150 = call float @llvm.amdgcn.exp2.f32(float %8149)
  %8151 = fadd float %8150, 1.000000e+00
  %8152 = call float @llvm.amdgcn.rcp.f32(float %8151)
  %8153 = fmul float %8144, %8152
  %8154 = fmul float %8153, %8148
  %8155 = bitcast float %8056 to i32
  %8156 = and i32 %8155, 2147483647
  %8157 = bitcast i32 %8156 to float
  %8158 = bitcast float %8070 to i32
  %8159 = and i32 %8158, 2147483647
  %8160 = bitcast i32 %8159 to float
  %8161 = call float @llvm.maximum.f32(float %8157, float %8160)
  %8162 = bitcast float %8084 to i32
  %8163 = and i32 %8162, 2147483647
  %8164 = bitcast i32 %8163 to float
  %8165 = call float @llvm.maximum.f32(float %8161, float %8164)
  %8166 = bitcast float %8098 to i32
  %8167 = and i32 %8166, 2147483647
  %8168 = bitcast i32 %8167 to float
  %8169 = call float @llvm.maximum.f32(float %8165, float %8168)
  %8170 = bitcast float %8112 to i32
  %8171 = and i32 %8170, 2147483647
  %8172 = bitcast i32 %8171 to float
  %8173 = call float @llvm.maximum.f32(float %8169, float %8172)
  %8174 = bitcast float %8126 to i32
  %8175 = and i32 %8174, 2147483647
  %8176 = bitcast i32 %8175 to float
  %8177 = call float @llvm.maximum.f32(float %8173, float %8176)
  %8178 = bitcast float %8140 to i32
  %8179 = and i32 %8178, 2147483647
  %8180 = bitcast i32 %8179 to float
  %8181 = call float @llvm.maximum.f32(float %8177, float %8180)
  %8182 = bitcast float %8154 to i32
  %8183 = and i32 %8182, 2147483647
  %8184 = bitcast i32 %8183 to float
  %8185 = call float @llvm.maximum.f32(float %8181, float %8184)
  %8186 = bitcast float %8185 to i32
  %8187 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %8186, i32 %8186, i32 177, i32 15, i32 15, i1 true)
  %8188 = bitcast i32 %8187 to float
  %8189 = call float @llvm.maximum.f32(float %8185, float %8188)
  %8190 = bitcast float %8189 to i32
  %8191 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %8190, i32 %8190, i32 78, i32 15, i32 15, i1 true)
  %8192 = bitcast i32 %8191 to float
  %8193 = call float @llvm.maximum.f32(float %8189, float %8192)
  %8194 = bitcast float %8193 to i32
  %8195 = add i32 %8194, 2097152
  %8196 = bitcast i32 %8195 to float
  %8197 = fmul float %8196, 2.500000e-01
  %8198 = bitcast float %8197 to i32
  %8199 = lshr i32 %8198, 23
  %8200 = call i32 @llvm.smin.i32(i32 %8199, i32 254)
  %8201 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 0, float %8056, float %8070, float %8197, i32 0)
  %8202 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %8201, float %8084, float %8098, float %8197, i32 1)
  %8203 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %8202, float %8112, float %8126, float %8197, i32 2)
  %8204 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %8203, float %8140, float %8154, float %8197, i32 3)
  %8205 = add i64 %46, %8041
  %8206 = mul i64 %8205, 256
  %8207 = add i64 %8206, %7020
  %8208 = add i64 %7024, %8207
  %8209 = inttoptr i64 %8208 to ptr addrspace(1)
  store i32 %8204, ptr addrspace(1) %8209, align 4
  %8210 = trunc i64 %6842 to i32
  %8211 = icmp eq i32 %8210, 0
  br i1 %8211, label %8212, label %8263

8212:                                             ; preds = %218
  %8213 = udiv i64 %18, 2
  %8214 = urem i64 %18, 2
  %8215 = trunc i64 %8214 to i32
  %8216 = mul i32 %8215, 2
  %8217 = sext i32 %8216 to i64
  %8218 = mul i64 %17, 4
  %8219 = mul i64 %17, 512
  %8220 = mul i64 %8213, 64
  %8221 = add i64 %8219, %8220
  %8222 = add i64 %8221, %7017
  %8223 = add i64 %8222, %6839
  %8224 = mul i64 %8223, 4
  %8225 = add i64 %8224, %8217
  %8226 = shl i32 %7186, 8
  %8227 = or i32 %7011, %8226
  %8228 = trunc i32 %8227 to i16
  %8229 = trunc i64 %8225 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %8228, ptr addrspace(8) %6256, i32 %8229, i32 0, i32 0)
  %8230 = add i64 %8218, 1
  %8231 = mul i64 %8230, 128
  %8232 = add i64 %8231, %8220
  %8233 = add i64 %8232, %7017
  %8234 = add i64 %8233, %6839
  %8235 = mul i64 %8234, 4
  %8236 = add i64 %8235, %8217
  %8237 = shl i32 %7524, 8
  %8238 = or i32 %7355, %8237
  %8239 = trunc i32 %8238 to i16
  %8240 = trunc i64 %8236 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %8239, ptr addrspace(8) %6256, i32 %8240, i32 0, i32 0)
  %8241 = add i64 %8218, 2
  %8242 = mul i64 %8241, 128
  %8243 = add i64 %8242, %8220
  %8244 = add i64 %8243, %7017
  %8245 = add i64 %8244, %6839
  %8246 = mul i64 %8245, 4
  %8247 = add i64 %8246, %8217
  %8248 = shl i32 %7862, 8
  %8249 = or i32 %7693, %8248
  %8250 = trunc i32 %8249 to i16
  %8251 = trunc i64 %8247 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %8250, ptr addrspace(8) %6256, i32 %8251, i32 0, i32 0)
  %8252 = add i64 %8218, 3
  %8253 = mul i64 %8252, 128
  %8254 = add i64 %8253, %8220
  %8255 = add i64 %8254, %7017
  %8256 = add i64 %8255, %6839
  %8257 = mul i64 %8256, 4
  %8258 = add i64 %8257, %8217
  %8259 = shl i32 %8200, 8
  %8260 = or i32 %8031, %8259
  %8261 = trunc i32 %8260 to i16
  %8262 = trunc i64 %8258 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %8261, ptr addrspace(8) %6256, i32 %8262, i32 0, i32 0)
  br label %8263

8263:                                             ; preds = %8212, %218
  br label %8264

8264:                                             ; preds = %8263, %12
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

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #4

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i64 @llvm.amdgcn.readfirstlane.i64(i64) #4

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly captures(none), ptr addrspace(3) writeonly captures(none), i32 immarg, i32, i32, i32 immarg, i32 immarg) #5

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #3

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #6

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.rcp.f32(float) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximum.f32(float, float) #2

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.update.dpp.i32(i32, i32, i32 immarg, i32 immarg, i32 immarg, i1 immarg) #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32, float, float, float, i32 immarg range(i32 0, 4)) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16, ptr addrspace(8) writeonly captures(none), i32, i32, i32 immarg) #8

attributes #0 = { "amdgpu-flat-work-group-size"="1,256" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #4 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #5 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #6 = { convergent nocallback nofree nounwind willreturn }
attributes #7 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #8 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
