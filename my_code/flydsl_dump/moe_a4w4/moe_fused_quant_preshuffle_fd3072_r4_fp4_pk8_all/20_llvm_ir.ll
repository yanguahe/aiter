; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

define amdgpu_kernel void @moe_fused_quant_preshuffle_fd3072_r4_fp4_pk8_all(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, i32 %4, i32 %5) #0 !reqd_work_group_size !1 {
  %7 = call range(i32 0, 256) i32 @llvm.amdgcn.workitem.id.x()
  %8 = sext i32 %7 to i64
  %9 = trunc i64 %8 to i32
  %10 = call i32 @llvm.amdgcn.workgroup.id.x()
  %11 = sext i32 %10 to i64
  %12 = trunc i64 %11 to i32
  %13 = udiv i32 %9, 32
  %14 = mul i32 %13, 32
  %15 = sub i32 %9, %14
  %16 = mul i32 %12, 8
  %17 = add i32 %16, %13
  %18 = icmp ult i32 %17, %4
  br i1 %18, label %19, label %1084

19:                                               ; preds = %6
  %20 = udiv i32 %17, %5
  %21 = mul i32 %20, %5
  %22 = sub i32 %17, %21
  %23 = udiv i32 %22, 64
  %24 = mul i32 %23, 64
  %25 = sub i32 %22, %24
  %26 = udiv i32 %25, 16
  %27 = mul i32 %26, 16
  %28 = sub i32 %25, %27
  %29 = mul i32 %5, 24
  %30 = mul i32 %20, %29
  %31 = mul i32 %23, 1536
  %32 = add i32 %30, %31
  %33 = add i32 %32, %27
  %34 = add i32 %33, %28
  %35 = ptrtoint ptr addrspace(1) %1 to i64
  %36 = ptrtoint ptr addrspace(1) %0 to i64
  %37 = ptrtoint ptr addrspace(1) %2 to i64
  %38 = inttoptr i64 %37 to ptr
  %39 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %38, i16 0, i64 4294967295, i32 159744)
  %40 = udiv i32 %15, 4
  %41 = mul i32 %40, 4
  %42 = sub i32 %15, %41
  %43 = icmp eq i32 %42, 0
  %44 = zext i32 %17 to i64
  %45 = mul i64 %44, 1536
  %46 = add i64 %35, %45
  %47 = inttoptr i64 %46 to ptr
  %48 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %47, i16 0, i64 1536, i32 159744)
  %49 = mul i64 %44, 6144
  %50 = add i64 %36, %49
  %51 = inttoptr i64 %50 to ptr
  %52 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %51, i16 0, i64 6144, i32 159744)
  %53 = icmp ult i32 %40, 96
  br i1 %53, label %54, label %137

54:                                               ; preds = %19
  %55 = mul i32 %40, 32
  %56 = mul i32 %42, 8
  %57 = add i32 %55, %56
  %58 = lshr i32 %57, 1
  %59 = mul i32 %58, 4
  %60 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %52, i32 %59, i32 0, i32 0)
  %61 = bitcast <4 x i32> %60 to <8 x bfloat>
  %62 = fpext <8 x bfloat> %61 to <8 x float>
  %63 = extractelement <8 x float> %62, i64 0
  %64 = call float @llvm.fabs.f32(float %63)
  %65 = call float @llvm.maximum.f32(float %64, float 0.000000e+00)
  %66 = extractelement <8 x float> %62, i64 1
  %67 = call float @llvm.fabs.f32(float %66)
  %68 = call float @llvm.maximum.f32(float %65, float %67)
  %69 = extractelement <8 x float> %62, i64 2
  %70 = call float @llvm.fabs.f32(float %69)
  %71 = call float @llvm.maximum.f32(float %68, float %70)
  %72 = extractelement <8 x float> %62, i64 3
  %73 = call float @llvm.fabs.f32(float %72)
  %74 = call float @llvm.maximum.f32(float %71, float %73)
  %75 = extractelement <8 x float> %62, i64 4
  %76 = call float @llvm.fabs.f32(float %75)
  %77 = call float @llvm.maximum.f32(float %74, float %76)
  %78 = extractelement <8 x float> %62, i64 5
  %79 = call float @llvm.fabs.f32(float %78)
  %80 = call float @llvm.maximum.f32(float %77, float %79)
  %81 = extractelement <8 x float> %62, i64 6
  %82 = call float @llvm.fabs.f32(float %81)
  %83 = call float @llvm.maximum.f32(float %80, float %82)
  %84 = extractelement <8 x float> %62, i64 7
  %85 = call float @llvm.fabs.f32(float %84)
  %86 = call float @llvm.maximum.f32(float %83, float %85)
  %87 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %88 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %87)
  %89 = add i32 %88, 32
  %90 = and i32 %89, -32
  %91 = xor i32 %88, 1
  %92 = icmp slt i32 %91, %90
  %93 = select i1 %92, i32 %91, i32 %88
  %94 = shl i32 %93, 2
  %95 = bitcast float %86 to i32
  %96 = call i32 @llvm.amdgcn.ds.bpermute(i32 %94, i32 %95)
  %97 = bitcast i32 %96 to float
  %98 = call float @llvm.maximum.f32(float %86, float %97)
  %99 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %100 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %99)
  %101 = add i32 %100, 32
  %102 = and i32 %101, -32
  %103 = xor i32 %100, 2
  %104 = icmp slt i32 %103, %102
  %105 = select i1 %104, i32 %103, i32 %100
  %106 = shl i32 %105, 2
  %107 = bitcast float %98 to i32
  %108 = call i32 @llvm.amdgcn.ds.bpermute(i32 %106, i32 %107)
  %109 = bitcast i32 %108 to float
  %110 = call float @llvm.maximum.f32(float %98, float %109)
  %111 = fmul float %110, 0x3FC5555560000000
  %112 = bitcast float %111 to i32
  %113 = and i32 %112, 8388607
  %114 = lshr i32 %112, 23
  %115 = and i32 %114, 255
  %116 = icmp ne i32 %113, 0
  %117 = add i32 %115, 1
  %118 = select i1 %116, i32 %117, i32 %115
  %119 = call i32 @llvm.smax.i32(i32 %118, i32 0)
  %120 = call i32 @llvm.smin.i32(i32 %119, i32 255)
  %121 = shl i32 %120, 23
  %122 = bitcast i32 %121 to float
  %123 = call i32 asm "v_cvt_scalef32_pk8_fp4_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %61, float %122)
  %124 = udiv i32 %40, 4
  %125 = mul i32 %124, 4
  %126 = sub i32 %40, %125
  %127 = trunc i32 %120 to i8
  %128 = mul i32 %40, 16
  %129 = mul i32 %42, 4
  %130 = add i32 %128, %129
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %123, ptr addrspace(8) %48, i32 %130, i32 0, i32 0)
  br i1 %43, label %131, label %136

131:                                              ; preds = %54
  %132 = mul i32 %124, 64
  %133 = add i32 %34, %132
  %134 = mul i32 %133, 4
  %135 = add i32 %134, %126
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %127, ptr addrspace(8) %39, i32 %135, i32 0, i32 0)
  br label %136

136:                                              ; preds = %131, %54
  br label %137

137:                                              ; preds = %136, %19
  %138 = add i32 %40, 8
  %139 = icmp ult i32 %138, 96
  br i1 %139, label %140, label %223

140:                                              ; preds = %137
  %141 = mul i32 %138, 32
  %142 = mul i32 %42, 8
  %143 = add i32 %141, %142
  %144 = lshr i32 %143, 1
  %145 = mul i32 %144, 4
  %146 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %52, i32 %145, i32 0, i32 0)
  %147 = bitcast <4 x i32> %146 to <8 x bfloat>
  %148 = fpext <8 x bfloat> %147 to <8 x float>
  %149 = extractelement <8 x float> %148, i64 0
  %150 = call float @llvm.fabs.f32(float %149)
  %151 = call float @llvm.maximum.f32(float %150, float 0.000000e+00)
  %152 = extractelement <8 x float> %148, i64 1
  %153 = call float @llvm.fabs.f32(float %152)
  %154 = call float @llvm.maximum.f32(float %151, float %153)
  %155 = extractelement <8 x float> %148, i64 2
  %156 = call float @llvm.fabs.f32(float %155)
  %157 = call float @llvm.maximum.f32(float %154, float %156)
  %158 = extractelement <8 x float> %148, i64 3
  %159 = call float @llvm.fabs.f32(float %158)
  %160 = call float @llvm.maximum.f32(float %157, float %159)
  %161 = extractelement <8 x float> %148, i64 4
  %162 = call float @llvm.fabs.f32(float %161)
  %163 = call float @llvm.maximum.f32(float %160, float %162)
  %164 = extractelement <8 x float> %148, i64 5
  %165 = call float @llvm.fabs.f32(float %164)
  %166 = call float @llvm.maximum.f32(float %163, float %165)
  %167 = extractelement <8 x float> %148, i64 6
  %168 = call float @llvm.fabs.f32(float %167)
  %169 = call float @llvm.maximum.f32(float %166, float %168)
  %170 = extractelement <8 x float> %148, i64 7
  %171 = call float @llvm.fabs.f32(float %170)
  %172 = call float @llvm.maximum.f32(float %169, float %171)
  %173 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %174 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %173)
  %175 = add i32 %174, 32
  %176 = and i32 %175, -32
  %177 = xor i32 %174, 1
  %178 = icmp slt i32 %177, %176
  %179 = select i1 %178, i32 %177, i32 %174
  %180 = shl i32 %179, 2
  %181 = bitcast float %172 to i32
  %182 = call i32 @llvm.amdgcn.ds.bpermute(i32 %180, i32 %181)
  %183 = bitcast i32 %182 to float
  %184 = call float @llvm.maximum.f32(float %172, float %183)
  %185 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %186 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %185)
  %187 = add i32 %186, 32
  %188 = and i32 %187, -32
  %189 = xor i32 %186, 2
  %190 = icmp slt i32 %189, %188
  %191 = select i1 %190, i32 %189, i32 %186
  %192 = shl i32 %191, 2
  %193 = bitcast float %184 to i32
  %194 = call i32 @llvm.amdgcn.ds.bpermute(i32 %192, i32 %193)
  %195 = bitcast i32 %194 to float
  %196 = call float @llvm.maximum.f32(float %184, float %195)
  %197 = fmul float %196, 0x3FC5555560000000
  %198 = bitcast float %197 to i32
  %199 = and i32 %198, 8388607
  %200 = lshr i32 %198, 23
  %201 = and i32 %200, 255
  %202 = icmp ne i32 %199, 0
  %203 = add i32 %201, 1
  %204 = select i1 %202, i32 %203, i32 %201
  %205 = call i32 @llvm.smax.i32(i32 %204, i32 0)
  %206 = call i32 @llvm.smin.i32(i32 %205, i32 255)
  %207 = shl i32 %206, 23
  %208 = bitcast i32 %207 to float
  %209 = call i32 asm "v_cvt_scalef32_pk8_fp4_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %147, float %208)
  %210 = udiv i32 %138, 4
  %211 = mul i32 %210, 4
  %212 = sub i32 %138, %211
  %213 = trunc i32 %206 to i8
  %214 = mul i32 %138, 16
  %215 = mul i32 %42, 4
  %216 = add i32 %214, %215
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %209, ptr addrspace(8) %48, i32 %216, i32 0, i32 0)
  br i1 %43, label %217, label %222

217:                                              ; preds = %140
  %218 = mul i32 %210, 64
  %219 = add i32 %34, %218
  %220 = mul i32 %219, 4
  %221 = add i32 %220, %212
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %213, ptr addrspace(8) %39, i32 %221, i32 0, i32 0)
  br label %222

222:                                              ; preds = %217, %140
  br label %223

223:                                              ; preds = %222, %137
  %224 = add i32 %40, 16
  %225 = icmp ult i32 %224, 96
  br i1 %225, label %226, label %309

226:                                              ; preds = %223
  %227 = mul i32 %224, 32
  %228 = mul i32 %42, 8
  %229 = add i32 %227, %228
  %230 = lshr i32 %229, 1
  %231 = mul i32 %230, 4
  %232 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %52, i32 %231, i32 0, i32 0)
  %233 = bitcast <4 x i32> %232 to <8 x bfloat>
  %234 = fpext <8 x bfloat> %233 to <8 x float>
  %235 = extractelement <8 x float> %234, i64 0
  %236 = call float @llvm.fabs.f32(float %235)
  %237 = call float @llvm.maximum.f32(float %236, float 0.000000e+00)
  %238 = extractelement <8 x float> %234, i64 1
  %239 = call float @llvm.fabs.f32(float %238)
  %240 = call float @llvm.maximum.f32(float %237, float %239)
  %241 = extractelement <8 x float> %234, i64 2
  %242 = call float @llvm.fabs.f32(float %241)
  %243 = call float @llvm.maximum.f32(float %240, float %242)
  %244 = extractelement <8 x float> %234, i64 3
  %245 = call float @llvm.fabs.f32(float %244)
  %246 = call float @llvm.maximum.f32(float %243, float %245)
  %247 = extractelement <8 x float> %234, i64 4
  %248 = call float @llvm.fabs.f32(float %247)
  %249 = call float @llvm.maximum.f32(float %246, float %248)
  %250 = extractelement <8 x float> %234, i64 5
  %251 = call float @llvm.fabs.f32(float %250)
  %252 = call float @llvm.maximum.f32(float %249, float %251)
  %253 = extractelement <8 x float> %234, i64 6
  %254 = call float @llvm.fabs.f32(float %253)
  %255 = call float @llvm.maximum.f32(float %252, float %254)
  %256 = extractelement <8 x float> %234, i64 7
  %257 = call float @llvm.fabs.f32(float %256)
  %258 = call float @llvm.maximum.f32(float %255, float %257)
  %259 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %260 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %259)
  %261 = add i32 %260, 32
  %262 = and i32 %261, -32
  %263 = xor i32 %260, 1
  %264 = icmp slt i32 %263, %262
  %265 = select i1 %264, i32 %263, i32 %260
  %266 = shl i32 %265, 2
  %267 = bitcast float %258 to i32
  %268 = call i32 @llvm.amdgcn.ds.bpermute(i32 %266, i32 %267)
  %269 = bitcast i32 %268 to float
  %270 = call float @llvm.maximum.f32(float %258, float %269)
  %271 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %272 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %271)
  %273 = add i32 %272, 32
  %274 = and i32 %273, -32
  %275 = xor i32 %272, 2
  %276 = icmp slt i32 %275, %274
  %277 = select i1 %276, i32 %275, i32 %272
  %278 = shl i32 %277, 2
  %279 = bitcast float %270 to i32
  %280 = call i32 @llvm.amdgcn.ds.bpermute(i32 %278, i32 %279)
  %281 = bitcast i32 %280 to float
  %282 = call float @llvm.maximum.f32(float %270, float %281)
  %283 = fmul float %282, 0x3FC5555560000000
  %284 = bitcast float %283 to i32
  %285 = and i32 %284, 8388607
  %286 = lshr i32 %284, 23
  %287 = and i32 %286, 255
  %288 = icmp ne i32 %285, 0
  %289 = add i32 %287, 1
  %290 = select i1 %288, i32 %289, i32 %287
  %291 = call i32 @llvm.smax.i32(i32 %290, i32 0)
  %292 = call i32 @llvm.smin.i32(i32 %291, i32 255)
  %293 = shl i32 %292, 23
  %294 = bitcast i32 %293 to float
  %295 = call i32 asm "v_cvt_scalef32_pk8_fp4_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %233, float %294)
  %296 = udiv i32 %224, 4
  %297 = mul i32 %296, 4
  %298 = sub i32 %224, %297
  %299 = trunc i32 %292 to i8
  %300 = mul i32 %224, 16
  %301 = mul i32 %42, 4
  %302 = add i32 %300, %301
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %295, ptr addrspace(8) %48, i32 %302, i32 0, i32 0)
  br i1 %43, label %303, label %308

303:                                              ; preds = %226
  %304 = mul i32 %296, 64
  %305 = add i32 %34, %304
  %306 = mul i32 %305, 4
  %307 = add i32 %306, %298
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %299, ptr addrspace(8) %39, i32 %307, i32 0, i32 0)
  br label %308

308:                                              ; preds = %303, %226
  br label %309

309:                                              ; preds = %308, %223
  %310 = add i32 %40, 24
  %311 = icmp ult i32 %310, 96
  br i1 %311, label %312, label %395

312:                                              ; preds = %309
  %313 = mul i32 %310, 32
  %314 = mul i32 %42, 8
  %315 = add i32 %313, %314
  %316 = lshr i32 %315, 1
  %317 = mul i32 %316, 4
  %318 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %52, i32 %317, i32 0, i32 0)
  %319 = bitcast <4 x i32> %318 to <8 x bfloat>
  %320 = fpext <8 x bfloat> %319 to <8 x float>
  %321 = extractelement <8 x float> %320, i64 0
  %322 = call float @llvm.fabs.f32(float %321)
  %323 = call float @llvm.maximum.f32(float %322, float 0.000000e+00)
  %324 = extractelement <8 x float> %320, i64 1
  %325 = call float @llvm.fabs.f32(float %324)
  %326 = call float @llvm.maximum.f32(float %323, float %325)
  %327 = extractelement <8 x float> %320, i64 2
  %328 = call float @llvm.fabs.f32(float %327)
  %329 = call float @llvm.maximum.f32(float %326, float %328)
  %330 = extractelement <8 x float> %320, i64 3
  %331 = call float @llvm.fabs.f32(float %330)
  %332 = call float @llvm.maximum.f32(float %329, float %331)
  %333 = extractelement <8 x float> %320, i64 4
  %334 = call float @llvm.fabs.f32(float %333)
  %335 = call float @llvm.maximum.f32(float %332, float %334)
  %336 = extractelement <8 x float> %320, i64 5
  %337 = call float @llvm.fabs.f32(float %336)
  %338 = call float @llvm.maximum.f32(float %335, float %337)
  %339 = extractelement <8 x float> %320, i64 6
  %340 = call float @llvm.fabs.f32(float %339)
  %341 = call float @llvm.maximum.f32(float %338, float %340)
  %342 = extractelement <8 x float> %320, i64 7
  %343 = call float @llvm.fabs.f32(float %342)
  %344 = call float @llvm.maximum.f32(float %341, float %343)
  %345 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %346 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %345)
  %347 = add i32 %346, 32
  %348 = and i32 %347, -32
  %349 = xor i32 %346, 1
  %350 = icmp slt i32 %349, %348
  %351 = select i1 %350, i32 %349, i32 %346
  %352 = shl i32 %351, 2
  %353 = bitcast float %344 to i32
  %354 = call i32 @llvm.amdgcn.ds.bpermute(i32 %352, i32 %353)
  %355 = bitcast i32 %354 to float
  %356 = call float @llvm.maximum.f32(float %344, float %355)
  %357 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %358 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %357)
  %359 = add i32 %358, 32
  %360 = and i32 %359, -32
  %361 = xor i32 %358, 2
  %362 = icmp slt i32 %361, %360
  %363 = select i1 %362, i32 %361, i32 %358
  %364 = shl i32 %363, 2
  %365 = bitcast float %356 to i32
  %366 = call i32 @llvm.amdgcn.ds.bpermute(i32 %364, i32 %365)
  %367 = bitcast i32 %366 to float
  %368 = call float @llvm.maximum.f32(float %356, float %367)
  %369 = fmul float %368, 0x3FC5555560000000
  %370 = bitcast float %369 to i32
  %371 = and i32 %370, 8388607
  %372 = lshr i32 %370, 23
  %373 = and i32 %372, 255
  %374 = icmp ne i32 %371, 0
  %375 = add i32 %373, 1
  %376 = select i1 %374, i32 %375, i32 %373
  %377 = call i32 @llvm.smax.i32(i32 %376, i32 0)
  %378 = call i32 @llvm.smin.i32(i32 %377, i32 255)
  %379 = shl i32 %378, 23
  %380 = bitcast i32 %379 to float
  %381 = call i32 asm "v_cvt_scalef32_pk8_fp4_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %319, float %380)
  %382 = udiv i32 %310, 4
  %383 = mul i32 %382, 4
  %384 = sub i32 %310, %383
  %385 = trunc i32 %378 to i8
  %386 = mul i32 %310, 16
  %387 = mul i32 %42, 4
  %388 = add i32 %386, %387
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %381, ptr addrspace(8) %48, i32 %388, i32 0, i32 0)
  br i1 %43, label %389, label %394

389:                                              ; preds = %312
  %390 = mul i32 %382, 64
  %391 = add i32 %34, %390
  %392 = mul i32 %391, 4
  %393 = add i32 %392, %384
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %385, ptr addrspace(8) %39, i32 %393, i32 0, i32 0)
  br label %394

394:                                              ; preds = %389, %312
  br label %395

395:                                              ; preds = %394, %309
  %396 = add i32 %40, 32
  %397 = icmp ult i32 %396, 96
  br i1 %397, label %398, label %481

398:                                              ; preds = %395
  %399 = mul i32 %396, 32
  %400 = mul i32 %42, 8
  %401 = add i32 %399, %400
  %402 = lshr i32 %401, 1
  %403 = mul i32 %402, 4
  %404 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %52, i32 %403, i32 0, i32 0)
  %405 = bitcast <4 x i32> %404 to <8 x bfloat>
  %406 = fpext <8 x bfloat> %405 to <8 x float>
  %407 = extractelement <8 x float> %406, i64 0
  %408 = call float @llvm.fabs.f32(float %407)
  %409 = call float @llvm.maximum.f32(float %408, float 0.000000e+00)
  %410 = extractelement <8 x float> %406, i64 1
  %411 = call float @llvm.fabs.f32(float %410)
  %412 = call float @llvm.maximum.f32(float %409, float %411)
  %413 = extractelement <8 x float> %406, i64 2
  %414 = call float @llvm.fabs.f32(float %413)
  %415 = call float @llvm.maximum.f32(float %412, float %414)
  %416 = extractelement <8 x float> %406, i64 3
  %417 = call float @llvm.fabs.f32(float %416)
  %418 = call float @llvm.maximum.f32(float %415, float %417)
  %419 = extractelement <8 x float> %406, i64 4
  %420 = call float @llvm.fabs.f32(float %419)
  %421 = call float @llvm.maximum.f32(float %418, float %420)
  %422 = extractelement <8 x float> %406, i64 5
  %423 = call float @llvm.fabs.f32(float %422)
  %424 = call float @llvm.maximum.f32(float %421, float %423)
  %425 = extractelement <8 x float> %406, i64 6
  %426 = call float @llvm.fabs.f32(float %425)
  %427 = call float @llvm.maximum.f32(float %424, float %426)
  %428 = extractelement <8 x float> %406, i64 7
  %429 = call float @llvm.fabs.f32(float %428)
  %430 = call float @llvm.maximum.f32(float %427, float %429)
  %431 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %432 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %431)
  %433 = add i32 %432, 32
  %434 = and i32 %433, -32
  %435 = xor i32 %432, 1
  %436 = icmp slt i32 %435, %434
  %437 = select i1 %436, i32 %435, i32 %432
  %438 = shl i32 %437, 2
  %439 = bitcast float %430 to i32
  %440 = call i32 @llvm.amdgcn.ds.bpermute(i32 %438, i32 %439)
  %441 = bitcast i32 %440 to float
  %442 = call float @llvm.maximum.f32(float %430, float %441)
  %443 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %444 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %443)
  %445 = add i32 %444, 32
  %446 = and i32 %445, -32
  %447 = xor i32 %444, 2
  %448 = icmp slt i32 %447, %446
  %449 = select i1 %448, i32 %447, i32 %444
  %450 = shl i32 %449, 2
  %451 = bitcast float %442 to i32
  %452 = call i32 @llvm.amdgcn.ds.bpermute(i32 %450, i32 %451)
  %453 = bitcast i32 %452 to float
  %454 = call float @llvm.maximum.f32(float %442, float %453)
  %455 = fmul float %454, 0x3FC5555560000000
  %456 = bitcast float %455 to i32
  %457 = and i32 %456, 8388607
  %458 = lshr i32 %456, 23
  %459 = and i32 %458, 255
  %460 = icmp ne i32 %457, 0
  %461 = add i32 %459, 1
  %462 = select i1 %460, i32 %461, i32 %459
  %463 = call i32 @llvm.smax.i32(i32 %462, i32 0)
  %464 = call i32 @llvm.smin.i32(i32 %463, i32 255)
  %465 = shl i32 %464, 23
  %466 = bitcast i32 %465 to float
  %467 = call i32 asm "v_cvt_scalef32_pk8_fp4_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %405, float %466)
  %468 = udiv i32 %396, 4
  %469 = mul i32 %468, 4
  %470 = sub i32 %396, %469
  %471 = trunc i32 %464 to i8
  %472 = mul i32 %396, 16
  %473 = mul i32 %42, 4
  %474 = add i32 %472, %473
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %467, ptr addrspace(8) %48, i32 %474, i32 0, i32 0)
  br i1 %43, label %475, label %480

475:                                              ; preds = %398
  %476 = mul i32 %468, 64
  %477 = add i32 %34, %476
  %478 = mul i32 %477, 4
  %479 = add i32 %478, %470
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %471, ptr addrspace(8) %39, i32 %479, i32 0, i32 0)
  br label %480

480:                                              ; preds = %475, %398
  br label %481

481:                                              ; preds = %480, %395
  %482 = add i32 %40, 40
  %483 = icmp ult i32 %482, 96
  br i1 %483, label %484, label %567

484:                                              ; preds = %481
  %485 = mul i32 %482, 32
  %486 = mul i32 %42, 8
  %487 = add i32 %485, %486
  %488 = lshr i32 %487, 1
  %489 = mul i32 %488, 4
  %490 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %52, i32 %489, i32 0, i32 0)
  %491 = bitcast <4 x i32> %490 to <8 x bfloat>
  %492 = fpext <8 x bfloat> %491 to <8 x float>
  %493 = extractelement <8 x float> %492, i64 0
  %494 = call float @llvm.fabs.f32(float %493)
  %495 = call float @llvm.maximum.f32(float %494, float 0.000000e+00)
  %496 = extractelement <8 x float> %492, i64 1
  %497 = call float @llvm.fabs.f32(float %496)
  %498 = call float @llvm.maximum.f32(float %495, float %497)
  %499 = extractelement <8 x float> %492, i64 2
  %500 = call float @llvm.fabs.f32(float %499)
  %501 = call float @llvm.maximum.f32(float %498, float %500)
  %502 = extractelement <8 x float> %492, i64 3
  %503 = call float @llvm.fabs.f32(float %502)
  %504 = call float @llvm.maximum.f32(float %501, float %503)
  %505 = extractelement <8 x float> %492, i64 4
  %506 = call float @llvm.fabs.f32(float %505)
  %507 = call float @llvm.maximum.f32(float %504, float %506)
  %508 = extractelement <8 x float> %492, i64 5
  %509 = call float @llvm.fabs.f32(float %508)
  %510 = call float @llvm.maximum.f32(float %507, float %509)
  %511 = extractelement <8 x float> %492, i64 6
  %512 = call float @llvm.fabs.f32(float %511)
  %513 = call float @llvm.maximum.f32(float %510, float %512)
  %514 = extractelement <8 x float> %492, i64 7
  %515 = call float @llvm.fabs.f32(float %514)
  %516 = call float @llvm.maximum.f32(float %513, float %515)
  %517 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %518 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %517)
  %519 = add i32 %518, 32
  %520 = and i32 %519, -32
  %521 = xor i32 %518, 1
  %522 = icmp slt i32 %521, %520
  %523 = select i1 %522, i32 %521, i32 %518
  %524 = shl i32 %523, 2
  %525 = bitcast float %516 to i32
  %526 = call i32 @llvm.amdgcn.ds.bpermute(i32 %524, i32 %525)
  %527 = bitcast i32 %526 to float
  %528 = call float @llvm.maximum.f32(float %516, float %527)
  %529 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %530 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %529)
  %531 = add i32 %530, 32
  %532 = and i32 %531, -32
  %533 = xor i32 %530, 2
  %534 = icmp slt i32 %533, %532
  %535 = select i1 %534, i32 %533, i32 %530
  %536 = shl i32 %535, 2
  %537 = bitcast float %528 to i32
  %538 = call i32 @llvm.amdgcn.ds.bpermute(i32 %536, i32 %537)
  %539 = bitcast i32 %538 to float
  %540 = call float @llvm.maximum.f32(float %528, float %539)
  %541 = fmul float %540, 0x3FC5555560000000
  %542 = bitcast float %541 to i32
  %543 = and i32 %542, 8388607
  %544 = lshr i32 %542, 23
  %545 = and i32 %544, 255
  %546 = icmp ne i32 %543, 0
  %547 = add i32 %545, 1
  %548 = select i1 %546, i32 %547, i32 %545
  %549 = call i32 @llvm.smax.i32(i32 %548, i32 0)
  %550 = call i32 @llvm.smin.i32(i32 %549, i32 255)
  %551 = shl i32 %550, 23
  %552 = bitcast i32 %551 to float
  %553 = call i32 asm "v_cvt_scalef32_pk8_fp4_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %491, float %552)
  %554 = udiv i32 %482, 4
  %555 = mul i32 %554, 4
  %556 = sub i32 %482, %555
  %557 = trunc i32 %550 to i8
  %558 = mul i32 %482, 16
  %559 = mul i32 %42, 4
  %560 = add i32 %558, %559
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %553, ptr addrspace(8) %48, i32 %560, i32 0, i32 0)
  br i1 %43, label %561, label %566

561:                                              ; preds = %484
  %562 = mul i32 %554, 64
  %563 = add i32 %34, %562
  %564 = mul i32 %563, 4
  %565 = add i32 %564, %556
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %557, ptr addrspace(8) %39, i32 %565, i32 0, i32 0)
  br label %566

566:                                              ; preds = %561, %484
  br label %567

567:                                              ; preds = %566, %481
  %568 = add i32 %40, 48
  %569 = icmp ult i32 %568, 96
  br i1 %569, label %570, label %653

570:                                              ; preds = %567
  %571 = mul i32 %568, 32
  %572 = mul i32 %42, 8
  %573 = add i32 %571, %572
  %574 = lshr i32 %573, 1
  %575 = mul i32 %574, 4
  %576 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %52, i32 %575, i32 0, i32 0)
  %577 = bitcast <4 x i32> %576 to <8 x bfloat>
  %578 = fpext <8 x bfloat> %577 to <8 x float>
  %579 = extractelement <8 x float> %578, i64 0
  %580 = call float @llvm.fabs.f32(float %579)
  %581 = call float @llvm.maximum.f32(float %580, float 0.000000e+00)
  %582 = extractelement <8 x float> %578, i64 1
  %583 = call float @llvm.fabs.f32(float %582)
  %584 = call float @llvm.maximum.f32(float %581, float %583)
  %585 = extractelement <8 x float> %578, i64 2
  %586 = call float @llvm.fabs.f32(float %585)
  %587 = call float @llvm.maximum.f32(float %584, float %586)
  %588 = extractelement <8 x float> %578, i64 3
  %589 = call float @llvm.fabs.f32(float %588)
  %590 = call float @llvm.maximum.f32(float %587, float %589)
  %591 = extractelement <8 x float> %578, i64 4
  %592 = call float @llvm.fabs.f32(float %591)
  %593 = call float @llvm.maximum.f32(float %590, float %592)
  %594 = extractelement <8 x float> %578, i64 5
  %595 = call float @llvm.fabs.f32(float %594)
  %596 = call float @llvm.maximum.f32(float %593, float %595)
  %597 = extractelement <8 x float> %578, i64 6
  %598 = call float @llvm.fabs.f32(float %597)
  %599 = call float @llvm.maximum.f32(float %596, float %598)
  %600 = extractelement <8 x float> %578, i64 7
  %601 = call float @llvm.fabs.f32(float %600)
  %602 = call float @llvm.maximum.f32(float %599, float %601)
  %603 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %604 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %603)
  %605 = add i32 %604, 32
  %606 = and i32 %605, -32
  %607 = xor i32 %604, 1
  %608 = icmp slt i32 %607, %606
  %609 = select i1 %608, i32 %607, i32 %604
  %610 = shl i32 %609, 2
  %611 = bitcast float %602 to i32
  %612 = call i32 @llvm.amdgcn.ds.bpermute(i32 %610, i32 %611)
  %613 = bitcast i32 %612 to float
  %614 = call float @llvm.maximum.f32(float %602, float %613)
  %615 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %616 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %615)
  %617 = add i32 %616, 32
  %618 = and i32 %617, -32
  %619 = xor i32 %616, 2
  %620 = icmp slt i32 %619, %618
  %621 = select i1 %620, i32 %619, i32 %616
  %622 = shl i32 %621, 2
  %623 = bitcast float %614 to i32
  %624 = call i32 @llvm.amdgcn.ds.bpermute(i32 %622, i32 %623)
  %625 = bitcast i32 %624 to float
  %626 = call float @llvm.maximum.f32(float %614, float %625)
  %627 = fmul float %626, 0x3FC5555560000000
  %628 = bitcast float %627 to i32
  %629 = and i32 %628, 8388607
  %630 = lshr i32 %628, 23
  %631 = and i32 %630, 255
  %632 = icmp ne i32 %629, 0
  %633 = add i32 %631, 1
  %634 = select i1 %632, i32 %633, i32 %631
  %635 = call i32 @llvm.smax.i32(i32 %634, i32 0)
  %636 = call i32 @llvm.smin.i32(i32 %635, i32 255)
  %637 = shl i32 %636, 23
  %638 = bitcast i32 %637 to float
  %639 = call i32 asm "v_cvt_scalef32_pk8_fp4_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %577, float %638)
  %640 = udiv i32 %568, 4
  %641 = mul i32 %640, 4
  %642 = sub i32 %568, %641
  %643 = trunc i32 %636 to i8
  %644 = mul i32 %568, 16
  %645 = mul i32 %42, 4
  %646 = add i32 %644, %645
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %639, ptr addrspace(8) %48, i32 %646, i32 0, i32 0)
  br i1 %43, label %647, label %652

647:                                              ; preds = %570
  %648 = mul i32 %640, 64
  %649 = add i32 %34, %648
  %650 = mul i32 %649, 4
  %651 = add i32 %650, %642
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %643, ptr addrspace(8) %39, i32 %651, i32 0, i32 0)
  br label %652

652:                                              ; preds = %647, %570
  br label %653

653:                                              ; preds = %652, %567
  %654 = add i32 %40, 56
  %655 = icmp ult i32 %654, 96
  br i1 %655, label %656, label %739

656:                                              ; preds = %653
  %657 = mul i32 %654, 32
  %658 = mul i32 %42, 8
  %659 = add i32 %657, %658
  %660 = lshr i32 %659, 1
  %661 = mul i32 %660, 4
  %662 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %52, i32 %661, i32 0, i32 0)
  %663 = bitcast <4 x i32> %662 to <8 x bfloat>
  %664 = fpext <8 x bfloat> %663 to <8 x float>
  %665 = extractelement <8 x float> %664, i64 0
  %666 = call float @llvm.fabs.f32(float %665)
  %667 = call float @llvm.maximum.f32(float %666, float 0.000000e+00)
  %668 = extractelement <8 x float> %664, i64 1
  %669 = call float @llvm.fabs.f32(float %668)
  %670 = call float @llvm.maximum.f32(float %667, float %669)
  %671 = extractelement <8 x float> %664, i64 2
  %672 = call float @llvm.fabs.f32(float %671)
  %673 = call float @llvm.maximum.f32(float %670, float %672)
  %674 = extractelement <8 x float> %664, i64 3
  %675 = call float @llvm.fabs.f32(float %674)
  %676 = call float @llvm.maximum.f32(float %673, float %675)
  %677 = extractelement <8 x float> %664, i64 4
  %678 = call float @llvm.fabs.f32(float %677)
  %679 = call float @llvm.maximum.f32(float %676, float %678)
  %680 = extractelement <8 x float> %664, i64 5
  %681 = call float @llvm.fabs.f32(float %680)
  %682 = call float @llvm.maximum.f32(float %679, float %681)
  %683 = extractelement <8 x float> %664, i64 6
  %684 = call float @llvm.fabs.f32(float %683)
  %685 = call float @llvm.maximum.f32(float %682, float %684)
  %686 = extractelement <8 x float> %664, i64 7
  %687 = call float @llvm.fabs.f32(float %686)
  %688 = call float @llvm.maximum.f32(float %685, float %687)
  %689 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %690 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %689)
  %691 = add i32 %690, 32
  %692 = and i32 %691, -32
  %693 = xor i32 %690, 1
  %694 = icmp slt i32 %693, %692
  %695 = select i1 %694, i32 %693, i32 %690
  %696 = shl i32 %695, 2
  %697 = bitcast float %688 to i32
  %698 = call i32 @llvm.amdgcn.ds.bpermute(i32 %696, i32 %697)
  %699 = bitcast i32 %698 to float
  %700 = call float @llvm.maximum.f32(float %688, float %699)
  %701 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %702 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %701)
  %703 = add i32 %702, 32
  %704 = and i32 %703, -32
  %705 = xor i32 %702, 2
  %706 = icmp slt i32 %705, %704
  %707 = select i1 %706, i32 %705, i32 %702
  %708 = shl i32 %707, 2
  %709 = bitcast float %700 to i32
  %710 = call i32 @llvm.amdgcn.ds.bpermute(i32 %708, i32 %709)
  %711 = bitcast i32 %710 to float
  %712 = call float @llvm.maximum.f32(float %700, float %711)
  %713 = fmul float %712, 0x3FC5555560000000
  %714 = bitcast float %713 to i32
  %715 = and i32 %714, 8388607
  %716 = lshr i32 %714, 23
  %717 = and i32 %716, 255
  %718 = icmp ne i32 %715, 0
  %719 = add i32 %717, 1
  %720 = select i1 %718, i32 %719, i32 %717
  %721 = call i32 @llvm.smax.i32(i32 %720, i32 0)
  %722 = call i32 @llvm.smin.i32(i32 %721, i32 255)
  %723 = shl i32 %722, 23
  %724 = bitcast i32 %723 to float
  %725 = call i32 asm "v_cvt_scalef32_pk8_fp4_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %663, float %724)
  %726 = udiv i32 %654, 4
  %727 = mul i32 %726, 4
  %728 = sub i32 %654, %727
  %729 = trunc i32 %722 to i8
  %730 = mul i32 %654, 16
  %731 = mul i32 %42, 4
  %732 = add i32 %730, %731
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %725, ptr addrspace(8) %48, i32 %732, i32 0, i32 0)
  br i1 %43, label %733, label %738

733:                                              ; preds = %656
  %734 = mul i32 %726, 64
  %735 = add i32 %34, %734
  %736 = mul i32 %735, 4
  %737 = add i32 %736, %728
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %729, ptr addrspace(8) %39, i32 %737, i32 0, i32 0)
  br label %738

738:                                              ; preds = %733, %656
  br label %739

739:                                              ; preds = %738, %653
  %740 = add i32 %40, 64
  %741 = icmp ult i32 %740, 96
  br i1 %741, label %742, label %825

742:                                              ; preds = %739
  %743 = mul i32 %740, 32
  %744 = mul i32 %42, 8
  %745 = add i32 %743, %744
  %746 = lshr i32 %745, 1
  %747 = mul i32 %746, 4
  %748 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %52, i32 %747, i32 0, i32 0)
  %749 = bitcast <4 x i32> %748 to <8 x bfloat>
  %750 = fpext <8 x bfloat> %749 to <8 x float>
  %751 = extractelement <8 x float> %750, i64 0
  %752 = call float @llvm.fabs.f32(float %751)
  %753 = call float @llvm.maximum.f32(float %752, float 0.000000e+00)
  %754 = extractelement <8 x float> %750, i64 1
  %755 = call float @llvm.fabs.f32(float %754)
  %756 = call float @llvm.maximum.f32(float %753, float %755)
  %757 = extractelement <8 x float> %750, i64 2
  %758 = call float @llvm.fabs.f32(float %757)
  %759 = call float @llvm.maximum.f32(float %756, float %758)
  %760 = extractelement <8 x float> %750, i64 3
  %761 = call float @llvm.fabs.f32(float %760)
  %762 = call float @llvm.maximum.f32(float %759, float %761)
  %763 = extractelement <8 x float> %750, i64 4
  %764 = call float @llvm.fabs.f32(float %763)
  %765 = call float @llvm.maximum.f32(float %762, float %764)
  %766 = extractelement <8 x float> %750, i64 5
  %767 = call float @llvm.fabs.f32(float %766)
  %768 = call float @llvm.maximum.f32(float %765, float %767)
  %769 = extractelement <8 x float> %750, i64 6
  %770 = call float @llvm.fabs.f32(float %769)
  %771 = call float @llvm.maximum.f32(float %768, float %770)
  %772 = extractelement <8 x float> %750, i64 7
  %773 = call float @llvm.fabs.f32(float %772)
  %774 = call float @llvm.maximum.f32(float %771, float %773)
  %775 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %776 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %775)
  %777 = add i32 %776, 32
  %778 = and i32 %777, -32
  %779 = xor i32 %776, 1
  %780 = icmp slt i32 %779, %778
  %781 = select i1 %780, i32 %779, i32 %776
  %782 = shl i32 %781, 2
  %783 = bitcast float %774 to i32
  %784 = call i32 @llvm.amdgcn.ds.bpermute(i32 %782, i32 %783)
  %785 = bitcast i32 %784 to float
  %786 = call float @llvm.maximum.f32(float %774, float %785)
  %787 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %788 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %787)
  %789 = add i32 %788, 32
  %790 = and i32 %789, -32
  %791 = xor i32 %788, 2
  %792 = icmp slt i32 %791, %790
  %793 = select i1 %792, i32 %791, i32 %788
  %794 = shl i32 %793, 2
  %795 = bitcast float %786 to i32
  %796 = call i32 @llvm.amdgcn.ds.bpermute(i32 %794, i32 %795)
  %797 = bitcast i32 %796 to float
  %798 = call float @llvm.maximum.f32(float %786, float %797)
  %799 = fmul float %798, 0x3FC5555560000000
  %800 = bitcast float %799 to i32
  %801 = and i32 %800, 8388607
  %802 = lshr i32 %800, 23
  %803 = and i32 %802, 255
  %804 = icmp ne i32 %801, 0
  %805 = add i32 %803, 1
  %806 = select i1 %804, i32 %805, i32 %803
  %807 = call i32 @llvm.smax.i32(i32 %806, i32 0)
  %808 = call i32 @llvm.smin.i32(i32 %807, i32 255)
  %809 = shl i32 %808, 23
  %810 = bitcast i32 %809 to float
  %811 = call i32 asm "v_cvt_scalef32_pk8_fp4_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %749, float %810)
  %812 = udiv i32 %740, 4
  %813 = mul i32 %812, 4
  %814 = sub i32 %740, %813
  %815 = trunc i32 %808 to i8
  %816 = mul i32 %740, 16
  %817 = mul i32 %42, 4
  %818 = add i32 %816, %817
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %811, ptr addrspace(8) %48, i32 %818, i32 0, i32 0)
  br i1 %43, label %819, label %824

819:                                              ; preds = %742
  %820 = mul i32 %812, 64
  %821 = add i32 %34, %820
  %822 = mul i32 %821, 4
  %823 = add i32 %822, %814
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %815, ptr addrspace(8) %39, i32 %823, i32 0, i32 0)
  br label %824

824:                                              ; preds = %819, %742
  br label %825

825:                                              ; preds = %824, %739
  %826 = add i32 %40, 72
  %827 = icmp ult i32 %826, 96
  br i1 %827, label %828, label %911

828:                                              ; preds = %825
  %829 = mul i32 %826, 32
  %830 = mul i32 %42, 8
  %831 = add i32 %829, %830
  %832 = lshr i32 %831, 1
  %833 = mul i32 %832, 4
  %834 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %52, i32 %833, i32 0, i32 0)
  %835 = bitcast <4 x i32> %834 to <8 x bfloat>
  %836 = fpext <8 x bfloat> %835 to <8 x float>
  %837 = extractelement <8 x float> %836, i64 0
  %838 = call float @llvm.fabs.f32(float %837)
  %839 = call float @llvm.maximum.f32(float %838, float 0.000000e+00)
  %840 = extractelement <8 x float> %836, i64 1
  %841 = call float @llvm.fabs.f32(float %840)
  %842 = call float @llvm.maximum.f32(float %839, float %841)
  %843 = extractelement <8 x float> %836, i64 2
  %844 = call float @llvm.fabs.f32(float %843)
  %845 = call float @llvm.maximum.f32(float %842, float %844)
  %846 = extractelement <8 x float> %836, i64 3
  %847 = call float @llvm.fabs.f32(float %846)
  %848 = call float @llvm.maximum.f32(float %845, float %847)
  %849 = extractelement <8 x float> %836, i64 4
  %850 = call float @llvm.fabs.f32(float %849)
  %851 = call float @llvm.maximum.f32(float %848, float %850)
  %852 = extractelement <8 x float> %836, i64 5
  %853 = call float @llvm.fabs.f32(float %852)
  %854 = call float @llvm.maximum.f32(float %851, float %853)
  %855 = extractelement <8 x float> %836, i64 6
  %856 = call float @llvm.fabs.f32(float %855)
  %857 = call float @llvm.maximum.f32(float %854, float %856)
  %858 = extractelement <8 x float> %836, i64 7
  %859 = call float @llvm.fabs.f32(float %858)
  %860 = call float @llvm.maximum.f32(float %857, float %859)
  %861 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %862 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %861)
  %863 = add i32 %862, 32
  %864 = and i32 %863, -32
  %865 = xor i32 %862, 1
  %866 = icmp slt i32 %865, %864
  %867 = select i1 %866, i32 %865, i32 %862
  %868 = shl i32 %867, 2
  %869 = bitcast float %860 to i32
  %870 = call i32 @llvm.amdgcn.ds.bpermute(i32 %868, i32 %869)
  %871 = bitcast i32 %870 to float
  %872 = call float @llvm.maximum.f32(float %860, float %871)
  %873 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %874 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %873)
  %875 = add i32 %874, 32
  %876 = and i32 %875, -32
  %877 = xor i32 %874, 2
  %878 = icmp slt i32 %877, %876
  %879 = select i1 %878, i32 %877, i32 %874
  %880 = shl i32 %879, 2
  %881 = bitcast float %872 to i32
  %882 = call i32 @llvm.amdgcn.ds.bpermute(i32 %880, i32 %881)
  %883 = bitcast i32 %882 to float
  %884 = call float @llvm.maximum.f32(float %872, float %883)
  %885 = fmul float %884, 0x3FC5555560000000
  %886 = bitcast float %885 to i32
  %887 = and i32 %886, 8388607
  %888 = lshr i32 %886, 23
  %889 = and i32 %888, 255
  %890 = icmp ne i32 %887, 0
  %891 = add i32 %889, 1
  %892 = select i1 %890, i32 %891, i32 %889
  %893 = call i32 @llvm.smax.i32(i32 %892, i32 0)
  %894 = call i32 @llvm.smin.i32(i32 %893, i32 255)
  %895 = shl i32 %894, 23
  %896 = bitcast i32 %895 to float
  %897 = call i32 asm "v_cvt_scalef32_pk8_fp4_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %835, float %896)
  %898 = udiv i32 %826, 4
  %899 = mul i32 %898, 4
  %900 = sub i32 %826, %899
  %901 = trunc i32 %894 to i8
  %902 = mul i32 %826, 16
  %903 = mul i32 %42, 4
  %904 = add i32 %902, %903
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %897, ptr addrspace(8) %48, i32 %904, i32 0, i32 0)
  br i1 %43, label %905, label %910

905:                                              ; preds = %828
  %906 = mul i32 %898, 64
  %907 = add i32 %34, %906
  %908 = mul i32 %907, 4
  %909 = add i32 %908, %900
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %901, ptr addrspace(8) %39, i32 %909, i32 0, i32 0)
  br label %910

910:                                              ; preds = %905, %828
  br label %911

911:                                              ; preds = %910, %825
  %912 = add i32 %40, 80
  %913 = icmp ult i32 %912, 96
  br i1 %913, label %914, label %997

914:                                              ; preds = %911
  %915 = mul i32 %912, 32
  %916 = mul i32 %42, 8
  %917 = add i32 %915, %916
  %918 = lshr i32 %917, 1
  %919 = mul i32 %918, 4
  %920 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %52, i32 %919, i32 0, i32 0)
  %921 = bitcast <4 x i32> %920 to <8 x bfloat>
  %922 = fpext <8 x bfloat> %921 to <8 x float>
  %923 = extractelement <8 x float> %922, i64 0
  %924 = call float @llvm.fabs.f32(float %923)
  %925 = call float @llvm.maximum.f32(float %924, float 0.000000e+00)
  %926 = extractelement <8 x float> %922, i64 1
  %927 = call float @llvm.fabs.f32(float %926)
  %928 = call float @llvm.maximum.f32(float %925, float %927)
  %929 = extractelement <8 x float> %922, i64 2
  %930 = call float @llvm.fabs.f32(float %929)
  %931 = call float @llvm.maximum.f32(float %928, float %930)
  %932 = extractelement <8 x float> %922, i64 3
  %933 = call float @llvm.fabs.f32(float %932)
  %934 = call float @llvm.maximum.f32(float %931, float %933)
  %935 = extractelement <8 x float> %922, i64 4
  %936 = call float @llvm.fabs.f32(float %935)
  %937 = call float @llvm.maximum.f32(float %934, float %936)
  %938 = extractelement <8 x float> %922, i64 5
  %939 = call float @llvm.fabs.f32(float %938)
  %940 = call float @llvm.maximum.f32(float %937, float %939)
  %941 = extractelement <8 x float> %922, i64 6
  %942 = call float @llvm.fabs.f32(float %941)
  %943 = call float @llvm.maximum.f32(float %940, float %942)
  %944 = extractelement <8 x float> %922, i64 7
  %945 = call float @llvm.fabs.f32(float %944)
  %946 = call float @llvm.maximum.f32(float %943, float %945)
  %947 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %948 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %947)
  %949 = add i32 %948, 32
  %950 = and i32 %949, -32
  %951 = xor i32 %948, 1
  %952 = icmp slt i32 %951, %950
  %953 = select i1 %952, i32 %951, i32 %948
  %954 = shl i32 %953, 2
  %955 = bitcast float %946 to i32
  %956 = call i32 @llvm.amdgcn.ds.bpermute(i32 %954, i32 %955)
  %957 = bitcast i32 %956 to float
  %958 = call float @llvm.maximum.f32(float %946, float %957)
  %959 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %960 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %959)
  %961 = add i32 %960, 32
  %962 = and i32 %961, -32
  %963 = xor i32 %960, 2
  %964 = icmp slt i32 %963, %962
  %965 = select i1 %964, i32 %963, i32 %960
  %966 = shl i32 %965, 2
  %967 = bitcast float %958 to i32
  %968 = call i32 @llvm.amdgcn.ds.bpermute(i32 %966, i32 %967)
  %969 = bitcast i32 %968 to float
  %970 = call float @llvm.maximum.f32(float %958, float %969)
  %971 = fmul float %970, 0x3FC5555560000000
  %972 = bitcast float %971 to i32
  %973 = and i32 %972, 8388607
  %974 = lshr i32 %972, 23
  %975 = and i32 %974, 255
  %976 = icmp ne i32 %973, 0
  %977 = add i32 %975, 1
  %978 = select i1 %976, i32 %977, i32 %975
  %979 = call i32 @llvm.smax.i32(i32 %978, i32 0)
  %980 = call i32 @llvm.smin.i32(i32 %979, i32 255)
  %981 = shl i32 %980, 23
  %982 = bitcast i32 %981 to float
  %983 = call i32 asm "v_cvt_scalef32_pk8_fp4_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %921, float %982)
  %984 = udiv i32 %912, 4
  %985 = mul i32 %984, 4
  %986 = sub i32 %912, %985
  %987 = trunc i32 %980 to i8
  %988 = mul i32 %912, 16
  %989 = mul i32 %42, 4
  %990 = add i32 %988, %989
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %983, ptr addrspace(8) %48, i32 %990, i32 0, i32 0)
  br i1 %43, label %991, label %996

991:                                              ; preds = %914
  %992 = mul i32 %984, 64
  %993 = add i32 %34, %992
  %994 = mul i32 %993, 4
  %995 = add i32 %994, %986
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %987, ptr addrspace(8) %39, i32 %995, i32 0, i32 0)
  br label %996

996:                                              ; preds = %991, %914
  br label %997

997:                                              ; preds = %996, %911
  %998 = add i32 %40, 88
  %999 = icmp ult i32 %998, 96
  br i1 %999, label %1000, label %1083

1000:                                             ; preds = %997
  %1001 = mul i32 %998, 32
  %1002 = mul i32 %42, 8
  %1003 = add i32 %1001, %1002
  %1004 = lshr i32 %1003, 1
  %1005 = mul i32 %1004, 4
  %1006 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %52, i32 %1005, i32 0, i32 0)
  %1007 = bitcast <4 x i32> %1006 to <8 x bfloat>
  %1008 = fpext <8 x bfloat> %1007 to <8 x float>
  %1009 = extractelement <8 x float> %1008, i64 0
  %1010 = call float @llvm.fabs.f32(float %1009)
  %1011 = call float @llvm.maximum.f32(float %1010, float 0.000000e+00)
  %1012 = extractelement <8 x float> %1008, i64 1
  %1013 = call float @llvm.fabs.f32(float %1012)
  %1014 = call float @llvm.maximum.f32(float %1011, float %1013)
  %1015 = extractelement <8 x float> %1008, i64 2
  %1016 = call float @llvm.fabs.f32(float %1015)
  %1017 = call float @llvm.maximum.f32(float %1014, float %1016)
  %1018 = extractelement <8 x float> %1008, i64 3
  %1019 = call float @llvm.fabs.f32(float %1018)
  %1020 = call float @llvm.maximum.f32(float %1017, float %1019)
  %1021 = extractelement <8 x float> %1008, i64 4
  %1022 = call float @llvm.fabs.f32(float %1021)
  %1023 = call float @llvm.maximum.f32(float %1020, float %1022)
  %1024 = extractelement <8 x float> %1008, i64 5
  %1025 = call float @llvm.fabs.f32(float %1024)
  %1026 = call float @llvm.maximum.f32(float %1023, float %1025)
  %1027 = extractelement <8 x float> %1008, i64 6
  %1028 = call float @llvm.fabs.f32(float %1027)
  %1029 = call float @llvm.maximum.f32(float %1026, float %1028)
  %1030 = extractelement <8 x float> %1008, i64 7
  %1031 = call float @llvm.fabs.f32(float %1030)
  %1032 = call float @llvm.maximum.f32(float %1029, float %1031)
  %1033 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %1034 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1033)
  %1035 = add i32 %1034, 32
  %1036 = and i32 %1035, -32
  %1037 = xor i32 %1034, 1
  %1038 = icmp slt i32 %1037, %1036
  %1039 = select i1 %1038, i32 %1037, i32 %1034
  %1040 = shl i32 %1039, 2
  %1041 = bitcast float %1032 to i32
  %1042 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1040, i32 %1041)
  %1043 = bitcast i32 %1042 to float
  %1044 = call float @llvm.maximum.f32(float %1032, float %1043)
  %1045 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %1046 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1045)
  %1047 = add i32 %1046, 32
  %1048 = and i32 %1047, -32
  %1049 = xor i32 %1046, 2
  %1050 = icmp slt i32 %1049, %1048
  %1051 = select i1 %1050, i32 %1049, i32 %1046
  %1052 = shl i32 %1051, 2
  %1053 = bitcast float %1044 to i32
  %1054 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1052, i32 %1053)
  %1055 = bitcast i32 %1054 to float
  %1056 = call float @llvm.maximum.f32(float %1044, float %1055)
  %1057 = fmul float %1056, 0x3FC5555560000000
  %1058 = bitcast float %1057 to i32
  %1059 = and i32 %1058, 8388607
  %1060 = lshr i32 %1058, 23
  %1061 = and i32 %1060, 255
  %1062 = icmp ne i32 %1059, 0
  %1063 = add i32 %1061, 1
  %1064 = select i1 %1062, i32 %1063, i32 %1061
  %1065 = call i32 @llvm.smax.i32(i32 %1064, i32 0)
  %1066 = call i32 @llvm.smin.i32(i32 %1065, i32 255)
  %1067 = shl i32 %1066, 23
  %1068 = bitcast i32 %1067 to float
  %1069 = call i32 asm "v_cvt_scalef32_pk8_fp4_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %1007, float %1068)
  %1070 = udiv i32 %998, 4
  %1071 = mul i32 %1070, 4
  %1072 = sub i32 %998, %1071
  %1073 = trunc i32 %1066 to i8
  %1074 = mul i32 %998, 16
  %1075 = mul i32 %42, 4
  %1076 = add i32 %1074, %1075
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %1069, ptr addrspace(8) %48, i32 %1076, i32 0, i32 0)
  br i1 %43, label %1077, label %1082

1077:                                             ; preds = %1000
  %1078 = mul i32 %1070, 64
  %1079 = add i32 %34, %1078
  %1080 = mul i32 %1079, 4
  %1081 = add i32 %1080, %1072
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %1073, ptr addrspace(8) %39, i32 %1081, i32 0, i32 0)
  br label %1082

1082:                                             ; preds = %1077, %1000
  br label %1083

1083:                                             ; preds = %1082, %997
  br label %1084

1084:                                             ; preds = %1083, %6
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr readnone, i16, i64, i32) #2

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
