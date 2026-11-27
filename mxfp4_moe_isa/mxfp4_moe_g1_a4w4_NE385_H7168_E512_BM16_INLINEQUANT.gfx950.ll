; ModuleID = '/shared/amdgpu/home/zhiming_ding_qle/yanguahe/code/wk_perf_test/aiter/aiter/jit/build/module_moe_mxfp4_gemm/blob/instances/mxfp4_moe_g1_a4w4_NE385_H7168_E512_BM16_INLINEQUANT.cu'
source_filename = "/shared/amdgpu/home/zhiming_ding_qle/yanguahe/code/wk_perf_test/aiter/aiter/jit/build/module_moe_mxfp4_gemm/blob/instances/mxfp4_moe_g1_a4w4_NE385_H7168_E512_BM16_INLINEQUANT.cu"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

%union.LDSPool = type { [4096 x float] }

$_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16 = comdat any

$_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds = comdat any

@_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds = linkonce_odr hidden local_unnamed_addr addrspace(3) global %union.LDSPool undef, comdat, align 16
@__hip_cuid_510a480b16533ff = addrspace(1) global i8 0
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @__hip_cuid_510a480b16533ff to ptr)], section "llvm.metadata"

; Function Attrs: convergent mustprogress norecurse nounwind
define protected amdgpu_kernel void @_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16(ptr addrspace(1) noalias nocapture noundef readnone %0, ptr addrspace(1) noalias nocapture noundef readnone %1, ptr addrspace(1) noalias noundef %2, ptr addrspace(1) noalias noundef %3, ptr addrspace(1) noalias nocapture noundef readonly %4, ptr addrspace(1) noalias nocapture noundef readonly %5, ptr addrspace(1) noalias noundef %6, i32 noundef %7, ptr addrspace(1) noalias noundef %8, ptr addrspace(1) noalias noundef %9, ptr addrspace(1) noalias noundef %10) local_unnamed_addr #0 comdat {
  %12 = ptrtoint ptr addrspace(1) %2 to i64
  %13 = inttoptr i64 %12 to ptr
  %14 = ptrtoint ptr addrspace(1) %3 to i64
  %15 = inttoptr i64 %14 to ptr
  %16 = ptrtoint ptr addrspace(1) %9 to i64
  %17 = inttoptr i64 %16 to ptr
  %18 = ptrtoint ptr addrspace(1) %10 to i64
  %19 = inttoptr i64 %18 to ptr
  %20 = tail call noundef i32 @llvm.amdgcn.workgroup.id.x()
  %21 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %22 = icmp samesign ult i32 %21, 256
  tail call void @llvm.assume(i1 %22)
  %23 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %21)
  %24 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p0(ptr readnone %13, i16 0, i32 1412956160, i32 131072)
  %25 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p0(ptr readnone %15, i16 0, i32 88309760, i32 131072)
  %26 = mul i32 %7, 14336
  %27 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p0(ptr readnone %19, i16 0, i32 %26, i32 131072)
  %28 = load i32, ptr addrspace(1) %5, align 4, !tbaa !7
  %29 = sdiv i32 %28, 16
  %30 = shl nsw i32 %29, 2
  %31 = icmp slt i32 %20, %30
  br i1 %31, label %32, label %3569

32:                                               ; preds = %11
  %33 = and i32 %21, 63
  %34 = lshr i32 %23, 6
  %35 = ptrtoint ptr addrspace(1) %8 to i64
  %36 = inttoptr i64 %35 to ptr
  %37 = ptrtoint ptr addrspace(1) %6 to i64
  %38 = inttoptr i64 %37 to ptr
  %39 = sdiv i32 %20, 4
  %40 = mul i32 %39, 4
  %41 = sub i32 %20, %40
  %42 = sext i32 %39 to i64
  %43 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %42
  %44 = load i32, ptr addrspace(1) %43, align 4, !tbaa !7
  %45 = shl nsw i32 %39, 4
  %46 = icmp samesign ult i32 %44, 385
  tail call void @llvm.assume(i1 %46)
  %47 = shl nuw nsw i32 %34, 2
  %48 = lshr i32 %33, 4
  %49 = add i32 %47, %45
  %50 = or disjoint i32 %49, %48
  %51 = sext i32 %50 to i64
  %52 = getelementptr inbounds i32, ptr %38, i64 %51
  %53 = load i32, ptr %52, align 4, !tbaa !7
  %54 = shl nuw nsw i32 %44, 10
  %55 = shl nsw i32 %41, 8
  %56 = add nsw i32 %54, %55
  %57 = and i32 %23, -64
  %58 = add i32 %56, %57
  %59 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %58)
  %60 = mul nsw i32 %59, 3584
  %61 = or disjoint i32 %57, 16
  %62 = add i32 %61, %56
  %63 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %62)
  %64 = mul nsw i32 %63, 3584
  %65 = or disjoint i32 %57, 32
  %66 = add i32 %65, %56
  %67 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %66)
  %68 = mul nsw i32 %67, 3584
  %69 = or disjoint i32 %57, 48
  %70 = add i32 %69, %56
  %71 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %70)
  %72 = mul nsw i32 %71, 3584
  %73 = shl nsw i32 %41, 3
  %74 = shl nuw nsw i32 %34, 1
  %75 = add nsw i32 %74, %73
  %76 = mul nuw nsw i32 %44, 57344
  %77 = mul i32 %75, 1792
  %78 = add i32 %76, %77
  %79 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %78)
  %80 = shl nsw i32 %79, 2
  %81 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %80)
  %82 = add nsw i32 %81, 4096
  %83 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %78)
  %84 = shl i32 %83, 2
  %85 = add i32 %84, 7168
  %86 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %85)
  %87 = add nsw i32 %86, 4096
  %88 = mul nsw i32 %53, 14336
  %89 = shl nuw nsw i32 %21, 4
  %90 = and i32 %89, 240
  %91 = or disjoint i32 %88, %90
  %92 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %27, i32 %91, i32 0, i32 0)
  %93 = extractelement <4 x i32> %92, i64 0
  %94 = and i32 %93, 2147450879
  %95 = extractelement <4 x i32> %92, i64 1
  %96 = and i32 %95, 2147450879
  %97 = extractelement <4 x i32> %92, i64 2
  %98 = and i32 %97, 2147450879
  %99 = extractelement <4 x i32> %92, i64 3
  %100 = and i32 %99, 2147450879
  %101 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %94, i32 %96) #11, !srcloc !11
  %102 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %98, i32 %100) #11, !srcloc !11
  %103 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %101, i32 %102) #11, !srcloc !11
  %104 = trunc i32 %103 to i16
  %105 = lshr i32 %103, 16
  %106 = trunc nuw i32 %105 to i16
  %107 = tail call noundef i16 @llvm.umax.i16(i16 %104, i16 %106)
  %108 = zext i16 %107 to i32
  %109 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %108, i32 177, i32 15, i32 15, i1 true)
  %110 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %108, i32 %109)
  %111 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %110, i32 78, i32 15, i32 15, i1 true)
  %112 = tail call noundef i32 @llvm.umax.i32(i32 %110, i32 %111)
  %113 = shl i32 %112, 16
  %114 = add i32 %113, 2097152
  %115 = lshr i32 %114, 23
  %116 = and i32 %115, 255
  %117 = tail call i32 @llvm.umax.i32(i32 %116, i32 2)
  %118 = add nuw nsw i32 %117, 254
  %119 = and i32 %118, 255
  %120 = shl nuw nsw i32 %119, 23
  %121 = bitcast i32 %120 to float
  %122 = bitcast <4 x i32> %92 to <8 x bfloat>
  %123 = shufflevector <8 x bfloat> %122, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %124 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %123, float %121, i32 0)
  %125 = bitcast <4 x i32> %92 to <8 x bfloat>
  %126 = shufflevector <8 x bfloat> %125, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %127 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %124, <2 x bfloat> %126, float %121, i32 1)
  %128 = bitcast <4 x i32> %92 to <8 x bfloat>
  %129 = shufflevector <8 x bfloat> %128, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %130 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %127, <2 x bfloat> %129, float %121, i32 2)
  %131 = bitcast <4 x i32> %92 to <8 x bfloat>
  %132 = shufflevector <8 x bfloat> %131, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %133 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %130, <2 x bfloat> %132, float %121, i32 3)
  %134 = or disjoint i32 %47, %48
  %135 = shl nuw nsw i32 %134, 3
  %136 = and i32 %135, 112
  %137 = shl nuw nsw i32 %33, 2
  %138 = and i32 %137, 12
  %139 = and i32 %137, 48
  %140 = xor i32 %136, %139
  %141 = or disjoint i32 %140, %138
  %142 = getelementptr inbounds nuw [3 x [16 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 0, i32 %134, i32 %141
  store i32 %133, ptr addrspace(3) %142, align 4, !tbaa !7
  %143 = shl nuw nsw i32 %48, 8
  %144 = and i32 %21, 15
  %145 = shl nuw nsw i32 %144, 4
  %146 = or disjoint i32 %143, %145
  %147 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %146, i32 %60, i32 2)
  %148 = or disjoint i32 %146, 1024
  %149 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %148, i32 %60, i32 2)
  %150 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %146, i32 %64, i32 2)
  %151 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %148, i32 %64, i32 2)
  %152 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %27, i32 %91, i32 256, i32 0)
  %153 = extractelement <4 x i32> %152, i64 0
  %154 = and i32 %153, 2147450879
  %155 = extractelement <4 x i32> %152, i64 1
  %156 = and i32 %155, 2147450879
  %157 = extractelement <4 x i32> %152, i64 2
  %158 = and i32 %157, 2147450879
  %159 = extractelement <4 x i32> %152, i64 3
  %160 = and i32 %159, 2147450879
  %161 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %154, i32 %156) #11, !srcloc !11
  %162 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %158, i32 %160) #11, !srcloc !11
  %163 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %161, i32 %162) #11, !srcloc !11
  %164 = trunc i32 %163 to i16
  %165 = lshr i32 %163, 16
  %166 = trunc nuw i32 %165 to i16
  %167 = tail call noundef i16 @llvm.umax.i16(i16 %164, i16 %166)
  %168 = zext i16 %167 to i32
  %169 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %168, i32 177, i32 15, i32 15, i1 true)
  %170 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %168, i32 %169)
  %171 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %170, i32 78, i32 15, i32 15, i1 true)
  %172 = tail call noundef i32 @llvm.umax.i32(i32 %170, i32 %171)
  %173 = shl i32 %172, 16
  %174 = add i32 %173, 2097152
  %175 = lshr i32 %174, 23
  %176 = and i32 %175, 255
  %177 = tail call i32 @llvm.umax.i32(i32 %176, i32 2)
  %178 = add nuw nsw i32 %177, 254
  %179 = and i32 %178, 255
  %180 = shl nuw nsw i32 %179, 23
  %181 = bitcast i32 %180 to float
  %182 = bitcast <4 x i32> %152 to <8 x bfloat>
  %183 = shufflevector <8 x bfloat> %182, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %184 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %183, float %181, i32 0)
  %185 = bitcast <4 x i32> %152 to <8 x bfloat>
  %186 = shufflevector <8 x bfloat> %185, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %187 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %184, <2 x bfloat> %186, float %181, i32 1)
  %188 = bitcast <4 x i32> %152 to <8 x bfloat>
  %189 = shufflevector <8 x bfloat> %188, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %190 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %187, <2 x bfloat> %189, float %181, i32 2)
  %191 = bitcast <4 x i32> %152 to <8 x bfloat>
  %192 = shufflevector <8 x bfloat> %191, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %193 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %190, <2 x bfloat> %192, float %181, i32 3)
  %194 = or disjoint i32 %139, 64
  %195 = xor i32 %136, %194
  %196 = or disjoint i32 %195, %138
  %197 = getelementptr inbounds nuw [3 x [16 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 0, i32 %134, i32 %196
  store i32 %193, ptr addrspace(3) %197, align 4, !tbaa !7
  %198 = shl nuw nsw i32 %179, 16
  %199 = or disjoint i32 %198, %119
  %200 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %146, i32 %68, i32 2)
  %201 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %148, i32 %68, i32 2)
  %202 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %146, i32 %72, i32 2)
  %203 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %148, i32 %72, i32 2)
  %204 = add nuw nsw i32 %134, %139
  %205 = shl nuw nsw i32 %204, 2
  %206 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %205
  store i32 %199, ptr addrspace(3) %206, align 4, !tbaa !7
  %207 = shl nuw nsw i32 %48, 6
  %208 = shl nuw nsw i32 %144, 2
  %209 = or disjoint i32 %207, %208
  %210 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %209, i32 %80, i32 0)
  %211 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %209, i32 %85, i32 0)
  %212 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %27, i32 %91, i32 512, i32 0)
  %213 = extractelement <4 x i32> %212, i64 0
  %214 = and i32 %213, 2147450879
  %215 = extractelement <4 x i32> %212, i64 1
  %216 = and i32 %215, 2147450879
  %217 = extractelement <4 x i32> %212, i64 2
  %218 = and i32 %217, 2147450879
  %219 = extractelement <4 x i32> %212, i64 3
  %220 = and i32 %219, 2147450879
  %221 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %214, i32 %216) #11, !srcloc !11
  %222 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %218, i32 %220) #11, !srcloc !11
  %223 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %221, i32 %222) #11, !srcloc !11
  %224 = trunc i32 %223 to i16
  %225 = lshr i32 %223, 16
  %226 = trunc nuw i32 %225 to i16
  %227 = tail call noundef i16 @llvm.umax.i16(i16 %224, i16 %226)
  %228 = zext i16 %227 to i32
  %229 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %228, i32 177, i32 15, i32 15, i1 true)
  %230 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %228, i32 %229)
  %231 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %230, i32 78, i32 15, i32 15, i1 true)
  %232 = tail call noundef i32 @llvm.umax.i32(i32 %230, i32 %231)
  %233 = shl i32 %232, 16
  %234 = add i32 %233, 2097152
  %235 = lshr i32 %234, 23
  %236 = and i32 %235, 255
  %237 = tail call i32 @llvm.umax.i32(i32 %236, i32 2)
  %238 = add nuw nsw i32 %237, 254
  %239 = and i32 %238, 255
  %240 = shl nuw nsw i32 %239, 23
  %241 = bitcast i32 %240 to float
  %242 = bitcast <4 x i32> %212 to <8 x bfloat>
  %243 = shufflevector <8 x bfloat> %242, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %244 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %243, float %241, i32 0)
  %245 = bitcast <4 x i32> %212 to <8 x bfloat>
  %246 = shufflevector <8 x bfloat> %245, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %247 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %244, <2 x bfloat> %246, float %241, i32 1)
  %248 = bitcast <4 x i32> %212 to <8 x bfloat>
  %249 = shufflevector <8 x bfloat> %248, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %250 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %247, <2 x bfloat> %249, float %241, i32 2)
  %251 = bitcast <4 x i32> %212 to <8 x bfloat>
  %252 = shufflevector <8 x bfloat> %251, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %253 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %250, <2 x bfloat> %252, float %241, i32 3)
  %254 = getelementptr inbounds nuw [3 x [16 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 1, i32 %134, i32 %141
  store i32 %253, ptr addrspace(3) %254, align 4, !tbaa !7
  %255 = or disjoint i32 %146, 2048
  %256 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %255, i32 %60, i32 2)
  %257 = or disjoint i32 %146, 3072
  %258 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %257, i32 %60, i32 2)
  %259 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %255, i32 %64, i32 2)
  %260 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %257, i32 %64, i32 2)
  %261 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %27, i32 %91, i32 768, i32 0)
  %262 = extractelement <4 x i32> %261, i64 0
  %263 = and i32 %262, 2147450879
  %264 = extractelement <4 x i32> %261, i64 1
  %265 = and i32 %264, 2147450879
  %266 = extractelement <4 x i32> %261, i64 2
  %267 = and i32 %266, 2147450879
  %268 = extractelement <4 x i32> %261, i64 3
  %269 = and i32 %268, 2147450879
  %270 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %263, i32 %265) #11, !srcloc !11
  %271 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %267, i32 %269) #11, !srcloc !11
  %272 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %270, i32 %271) #11, !srcloc !11
  %273 = trunc i32 %272 to i16
  %274 = lshr i32 %272, 16
  %275 = trunc nuw i32 %274 to i16
  %276 = tail call noundef i16 @llvm.umax.i16(i16 %273, i16 %275)
  %277 = zext i16 %276 to i32
  %278 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %277, i32 177, i32 15, i32 15, i1 true)
  %279 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %277, i32 %278)
  %280 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %279, i32 78, i32 15, i32 15, i1 true)
  %281 = tail call noundef i32 @llvm.umax.i32(i32 %279, i32 %280)
  %282 = shl i32 %281, 16
  %283 = add i32 %282, 2097152
  %284 = lshr i32 %283, 23
  %285 = and i32 %284, 255
  %286 = tail call i32 @llvm.umax.i32(i32 %285, i32 2)
  %287 = add nuw nsw i32 %286, 254
  %288 = and i32 %287, 255
  %289 = shl nuw nsw i32 %288, 23
  %290 = bitcast i32 %289 to float
  %291 = bitcast <4 x i32> %261 to <8 x bfloat>
  %292 = shufflevector <8 x bfloat> %291, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %293 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %292, float %290, i32 0)
  %294 = bitcast <4 x i32> %261 to <8 x bfloat>
  %295 = shufflevector <8 x bfloat> %294, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %296 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %293, <2 x bfloat> %295, float %290, i32 1)
  %297 = bitcast <4 x i32> %261 to <8 x bfloat>
  %298 = shufflevector <8 x bfloat> %297, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %299 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %296, <2 x bfloat> %298, float %290, i32 2)
  %300 = bitcast <4 x i32> %261 to <8 x bfloat>
  %301 = shufflevector <8 x bfloat> %300, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %302 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %299, <2 x bfloat> %301, float %290, i32 3)
  %303 = getelementptr inbounds nuw [3 x [16 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 1, i32 %134, i32 %196
  store i32 %302, ptr addrspace(3) %303, align 4, !tbaa !7
  %304 = shl nuw nsw i32 %288, 16
  %305 = or disjoint i32 %304, %239
  %306 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %255, i32 %68, i32 2)
  %307 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %257, i32 %68, i32 2)
  %308 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %255, i32 %72, i32 2)
  %309 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %257, i32 %72, i32 2)
  %310 = add nuw nsw i32 %205, 256
  %311 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %310
  store i32 %305, ptr addrspace(3) %311, align 4, !tbaa !7
  %312 = or disjoint i32 %209, 256
  %313 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %312, i32 %80, i32 0)
  %314 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %312, i32 %85, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %315 = and i32 %21, 48
  %316 = shl nuw nsw i32 %21, 3
  %317 = and i32 %316, 112
  %318 = xor i32 %317, %315
  %319 = getelementptr inbounds nuw [3 x [16 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 0, i32 %144, i32 %318
  %320 = load <4 x i32>, ptr addrspace(3) %319, align 16, !tbaa !12
  %321 = or disjoint i32 %315, 64
  %322 = xor i32 %321, %317
  %323 = getelementptr inbounds nuw [3 x [16 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 0, i32 %144, i32 %322
  %324 = load <4 x i32>, ptr addrspace(3) %323, align 16, !tbaa !12
  %325 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %209
  %326 = load i32, ptr addrspace(3) %325, align 4, !tbaa !7
  %327 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 1024, i32 0)
  %328 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 1280, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %329 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %320, <4 x i32> noundef %147, <4 x float> noundef zeroinitializer, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %326, i32 noundef 0, i32 noundef %210) #12
  %330 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %324, <4 x i32> noundef %149, <4 x float> noundef %329, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %326, i32 noundef 2, i32 noundef %210) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %331 = or disjoint i32 %146, 4096
  %332 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %331, i32 %60, i32 2)
  %333 = or disjoint i32 %146, 5120
  %334 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %333, i32 %60, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %335 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %320, <4 x i32> noundef %150, <4 x float> noundef zeroinitializer, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %326, i32 noundef 1, i32 noundef %210) #12
  %336 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %324, <4 x i32> noundef %151, <4 x float> noundef %335, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %326, i32 noundef 3, i32 noundef %210) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %337 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %331, i32 %64, i32 2)
  %338 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %333, i32 %64, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %339 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %320, <4 x i32> noundef %200, <4 x float> noundef zeroinitializer, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %326, i32 noundef 0, i32 noundef %211) #12
  %340 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %324, <4 x i32> noundef %201, <4 x float> noundef %339, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %326, i32 noundef 2, i32 noundef %211) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %341 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %331, i32 %68, i32 2)
  %342 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %333, i32 %68, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %343 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %320, <4 x i32> noundef %202, <4 x float> noundef zeroinitializer, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %326, i32 noundef 1, i32 noundef %211) #12
  %344 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %324, <4 x i32> noundef %203, <4 x float> noundef %343, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %326, i32 noundef 3, i32 noundef %211) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %345 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %331, i32 %72, i32 2)
  %346 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %333, i32 %72, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %347 = or disjoint i32 %209, 512
  %348 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %347, i32 %80, i32 0)
  %349 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %347, i32 %85, i32 0)
  %350 = extractelement <4 x i32> %327, i64 0
  %351 = and i32 %350, 2147450879
  %352 = extractelement <4 x i32> %327, i64 1
  %353 = and i32 %352, 2147450879
  %354 = extractelement <4 x i32> %327, i64 2
  %355 = and i32 %354, 2147450879
  %356 = extractelement <4 x i32> %327, i64 3
  %357 = and i32 %356, 2147450879
  %358 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %351, i32 %353) #11, !srcloc !11
  %359 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %355, i32 %357) #11, !srcloc !11
  %360 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %358, i32 %359) #11, !srcloc !11
  %361 = trunc i32 %360 to i16
  %362 = lshr i32 %360, 16
  %363 = trunc nuw i32 %362 to i16
  %364 = tail call noundef i16 @llvm.umax.i16(i16 %361, i16 %363)
  %365 = zext i16 %364 to i32
  %366 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %365, i32 177, i32 15, i32 15, i1 true)
  %367 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %365, i32 %366)
  %368 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %367, i32 78, i32 15, i32 15, i1 true)
  %369 = tail call noundef i32 @llvm.umax.i32(i32 %367, i32 %368)
  %370 = shl i32 %369, 16
  %371 = add i32 %370, 2097152
  %372 = lshr i32 %371, 23
  %373 = and i32 %372, 255
  %374 = tail call i32 @llvm.umax.i32(i32 %373, i32 2)
  %375 = add nuw nsw i32 %374, 254
  %376 = and i32 %375, 255
  %377 = shl nuw nsw i32 %376, 23
  %378 = bitcast i32 %377 to float
  %379 = bitcast <4 x i32> %327 to <8 x bfloat>
  %380 = shufflevector <8 x bfloat> %379, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %381 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %380, float %378, i32 0)
  %382 = bitcast <4 x i32> %327 to <8 x bfloat>
  %383 = shufflevector <8 x bfloat> %382, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %384 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %381, <2 x bfloat> %383, float %378, i32 1)
  %385 = bitcast <4 x i32> %327 to <8 x bfloat>
  %386 = shufflevector <8 x bfloat> %385, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %387 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %384, <2 x bfloat> %386, float %378, i32 2)
  %388 = bitcast <4 x i32> %327 to <8 x bfloat>
  %389 = shufflevector <8 x bfloat> %388, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %390 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %387, <2 x bfloat> %389, float %378, i32 3)
  %391 = getelementptr inbounds nuw [3 x [16 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 2, i32 %134, i32 %141
  store i32 %390, ptr addrspace(3) %391, align 4, !tbaa !7
  %392 = extractelement <4 x i32> %328, i64 0
  %393 = and i32 %392, 2147450879
  %394 = extractelement <4 x i32> %328, i64 1
  %395 = and i32 %394, 2147450879
  %396 = extractelement <4 x i32> %328, i64 2
  %397 = and i32 %396, 2147450879
  %398 = extractelement <4 x i32> %328, i64 3
  %399 = and i32 %398, 2147450879
  %400 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %393, i32 %395) #11, !srcloc !11
  %401 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %397, i32 %399) #11, !srcloc !11
  %402 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %400, i32 %401) #11, !srcloc !11
  %403 = trunc i32 %402 to i16
  %404 = lshr i32 %402, 16
  %405 = trunc nuw i32 %404 to i16
  %406 = tail call noundef i16 @llvm.umax.i16(i16 %403, i16 %405)
  %407 = zext i16 %406 to i32
  %408 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %407, i32 177, i32 15, i32 15, i1 true)
  %409 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %407, i32 %408)
  %410 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %409, i32 78, i32 15, i32 15, i1 true)
  %411 = tail call noundef i32 @llvm.umax.i32(i32 %409, i32 %410)
  %412 = shl i32 %411, 16
  %413 = add i32 %412, 2097152
  %414 = lshr i32 %413, 23
  %415 = and i32 %414, 255
  %416 = tail call i32 @llvm.umax.i32(i32 %415, i32 2)
  %417 = add nuw nsw i32 %416, 254
  %418 = and i32 %417, 255
  %419 = shl nuw nsw i32 %418, 23
  %420 = bitcast i32 %419 to float
  %421 = bitcast <4 x i32> %328 to <8 x bfloat>
  %422 = shufflevector <8 x bfloat> %421, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %423 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %422, float %420, i32 0)
  %424 = bitcast <4 x i32> %328 to <8 x bfloat>
  %425 = shufflevector <8 x bfloat> %424, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %426 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %423, <2 x bfloat> %425, float %420, i32 1)
  %427 = bitcast <4 x i32> %328 to <8 x bfloat>
  %428 = shufflevector <8 x bfloat> %427, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %429 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %426, <2 x bfloat> %428, float %420, i32 2)
  %430 = bitcast <4 x i32> %328 to <8 x bfloat>
  %431 = shufflevector <8 x bfloat> %430, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %432 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %429, <2 x bfloat> %431, float %420, i32 3)
  %433 = getelementptr inbounds nuw [3 x [16 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 2, i32 %134, i32 %196
  store i32 %432, ptr addrspace(3) %433, align 4, !tbaa !7
  %434 = shl nuw nsw i32 %418, 16
  %435 = or disjoint i32 %434, %376
  %436 = add nuw nsw i32 %205, 512
  %437 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %436
  store i32 %435, ptr addrspace(3) %437, align 4, !tbaa !7
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %438 = getelementptr inbounds nuw [3 x [16 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 1, i32 %144, i32 %318
  %439 = load <4 x i32>, ptr addrspace(3) %438, align 16, !tbaa !12
  %440 = getelementptr inbounds nuw [3 x [16 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 1, i32 %144, i32 %322
  %441 = load <4 x i32>, ptr addrspace(3) %440, align 16, !tbaa !12
  %442 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %312
  %443 = load i32, ptr addrspace(3) %442, align 4, !tbaa !7
  %444 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 1536, i32 0)
  %445 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 1792, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %446 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %439, <4 x i32> noundef %256, <4 x float> noundef %330, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %443, i32 noundef 0, i32 noundef %313) #12
  %447 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %441, <4 x i32> noundef %258, <4 x float> noundef %446, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %443, i32 noundef 2, i32 noundef %313) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %448 = or disjoint i32 %146, 6144
  %449 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %448, i32 %60, i32 2)
  %450 = or disjoint i32 %146, 7168
  %451 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %450, i32 %60, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %452 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %439, <4 x i32> noundef %259, <4 x float> noundef %336, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %443, i32 noundef 1, i32 noundef %313) #12
  %453 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %441, <4 x i32> noundef %260, <4 x float> noundef %452, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %443, i32 noundef 3, i32 noundef %313) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %454 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %448, i32 %64, i32 2)
  %455 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %450, i32 %64, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %456 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %439, <4 x i32> noundef %306, <4 x float> noundef %340, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %443, i32 noundef 0, i32 noundef %314) #12
  %457 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %441, <4 x i32> noundef %307, <4 x float> noundef %456, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %443, i32 noundef 2, i32 noundef %314) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %458 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %448, i32 %68, i32 2)
  %459 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %450, i32 %68, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %460 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %439, <4 x i32> noundef %308, <4 x float> noundef %344, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %443, i32 noundef 1, i32 noundef %314) #12
  %461 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %441, <4 x i32> noundef %309, <4 x float> noundef %460, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %443, i32 noundef 3, i32 noundef %314) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %462 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %448, i32 %72, i32 2)
  %463 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %450, i32 %72, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %464 = or disjoint i32 %209, 768
  %465 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %464, i32 %80, i32 0)
  %466 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %464, i32 %85, i32 0)
  %467 = extractelement <4 x i32> %444, i64 0
  %468 = and i32 %467, 2147450879
  %469 = extractelement <4 x i32> %444, i64 1
  %470 = and i32 %469, 2147450879
  %471 = extractelement <4 x i32> %444, i64 2
  %472 = and i32 %471, 2147450879
  %473 = extractelement <4 x i32> %444, i64 3
  %474 = and i32 %473, 2147450879
  %475 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %468, i32 %470) #11, !srcloc !11
  %476 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %472, i32 %474) #11, !srcloc !11
  %477 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %475, i32 %476) #11, !srcloc !11
  %478 = trunc i32 %477 to i16
  %479 = lshr i32 %477, 16
  %480 = trunc nuw i32 %479 to i16
  %481 = tail call noundef i16 @llvm.umax.i16(i16 %478, i16 %480)
  %482 = zext i16 %481 to i32
  %483 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %482, i32 177, i32 15, i32 15, i1 true)
  %484 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %482, i32 %483)
  %485 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %484, i32 78, i32 15, i32 15, i1 true)
  %486 = tail call noundef i32 @llvm.umax.i32(i32 %484, i32 %485)
  %487 = shl i32 %486, 16
  %488 = add i32 %487, 2097152
  %489 = lshr i32 %488, 23
  %490 = and i32 %489, 255
  %491 = tail call i32 @llvm.umax.i32(i32 %490, i32 2)
  %492 = add nuw nsw i32 %491, 254
  %493 = and i32 %492, 255
  %494 = shl nuw nsw i32 %493, 23
  %495 = bitcast i32 %494 to float
  %496 = bitcast <4 x i32> %444 to <8 x bfloat>
  %497 = shufflevector <8 x bfloat> %496, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %498 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %497, float %495, i32 0)
  %499 = bitcast <4 x i32> %444 to <8 x bfloat>
  %500 = shufflevector <8 x bfloat> %499, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %501 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %498, <2 x bfloat> %500, float %495, i32 1)
  %502 = bitcast <4 x i32> %444 to <8 x bfloat>
  %503 = shufflevector <8 x bfloat> %502, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %504 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %501, <2 x bfloat> %503, float %495, i32 2)
  %505 = bitcast <4 x i32> %444 to <8 x bfloat>
  %506 = shufflevector <8 x bfloat> %505, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %507 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %504, <2 x bfloat> %506, float %495, i32 3)
  store i32 %507, ptr addrspace(3) %142, align 4, !tbaa !7
  %508 = extractelement <4 x i32> %445, i64 0
  %509 = and i32 %508, 2147450879
  %510 = extractelement <4 x i32> %445, i64 1
  %511 = and i32 %510, 2147450879
  %512 = extractelement <4 x i32> %445, i64 2
  %513 = and i32 %512, 2147450879
  %514 = extractelement <4 x i32> %445, i64 3
  %515 = and i32 %514, 2147450879
  %516 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %509, i32 %511) #11, !srcloc !11
  %517 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %513, i32 %515) #11, !srcloc !11
  %518 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %516, i32 %517) #11, !srcloc !11
  %519 = trunc i32 %518 to i16
  %520 = lshr i32 %518, 16
  %521 = trunc nuw i32 %520 to i16
  %522 = tail call noundef i16 @llvm.umax.i16(i16 %519, i16 %521)
  %523 = zext i16 %522 to i32
  %524 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %523, i32 177, i32 15, i32 15, i1 true)
  %525 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %523, i32 %524)
  %526 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %525, i32 78, i32 15, i32 15, i1 true)
  %527 = tail call noundef i32 @llvm.umax.i32(i32 %525, i32 %526)
  %528 = shl i32 %527, 16
  %529 = add i32 %528, 2097152
  %530 = lshr i32 %529, 23
  %531 = and i32 %530, 255
  %532 = tail call i32 @llvm.umax.i32(i32 %531, i32 2)
  %533 = add nuw nsw i32 %532, 254
  %534 = and i32 %533, 255
  %535 = shl nuw nsw i32 %534, 23
  %536 = bitcast i32 %535 to float
  %537 = bitcast <4 x i32> %445 to <8 x bfloat>
  %538 = shufflevector <8 x bfloat> %537, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %539 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %538, float %536, i32 0)
  %540 = bitcast <4 x i32> %445 to <8 x bfloat>
  %541 = shufflevector <8 x bfloat> %540, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %542 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %539, <2 x bfloat> %541, float %536, i32 1)
  %543 = bitcast <4 x i32> %445 to <8 x bfloat>
  %544 = shufflevector <8 x bfloat> %543, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %545 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %542, <2 x bfloat> %544, float %536, i32 2)
  %546 = bitcast <4 x i32> %445 to <8 x bfloat>
  %547 = shufflevector <8 x bfloat> %546, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %548 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %545, <2 x bfloat> %547, float %536, i32 3)
  store i32 %548, ptr addrspace(3) %197, align 4, !tbaa !7
  %549 = shl nuw nsw i32 %534, 16
  %550 = or disjoint i32 %549, %493
  %551 = add nuw nsw i32 %205, 768
  %552 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %551
  store i32 %550, ptr addrspace(3) %552, align 4, !tbaa !7
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %553 = getelementptr inbounds nuw [3 x [16 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 2, i32 %144, i32 %318
  %554 = load <4 x i32>, ptr addrspace(3) %553, align 16, !tbaa !12
  %555 = getelementptr inbounds nuw [3 x [16 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 2, i32 %144, i32 %322
  %556 = load <4 x i32>, ptr addrspace(3) %555, align 16, !tbaa !12
  %557 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %347
  %558 = load i32, ptr addrspace(3) %557, align 4, !tbaa !7
  %559 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 2048, i32 0)
  %560 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 2304, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %561 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %554, <4 x i32> noundef %332, <4 x float> noundef %447, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %558, i32 noundef 0, i32 noundef %348) #12
  %562 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %556, <4 x i32> noundef %334, <4 x float> noundef %561, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %558, i32 noundef 2, i32 noundef %348) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %563 = or disjoint i32 %146, 8192
  %564 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %563, i32 %60, i32 2)
  %565 = or disjoint i32 %146, 9216
  %566 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %565, i32 %60, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %567 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %554, <4 x i32> noundef %337, <4 x float> noundef %453, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %558, i32 noundef 1, i32 noundef %348) #12
  %568 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %556, <4 x i32> noundef %338, <4 x float> noundef %567, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %558, i32 noundef 3, i32 noundef %348) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %569 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %563, i32 %64, i32 2)
  %570 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %565, i32 %64, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %571 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %554, <4 x i32> noundef %341, <4 x float> noundef %457, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %558, i32 noundef 0, i32 noundef %349) #12
  %572 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %556, <4 x i32> noundef %342, <4 x float> noundef %571, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %558, i32 noundef 2, i32 noundef %349) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %573 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %563, i32 %68, i32 2)
  %574 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %565, i32 %68, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %575 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %554, <4 x i32> noundef %345, <4 x float> noundef %461, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %558, i32 noundef 1, i32 noundef %349) #12
  %576 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %556, <4 x i32> noundef %346, <4 x float> noundef %575, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %558, i32 noundef 3, i32 noundef %349) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %577 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %563, i32 %72, i32 2)
  %578 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %565, i32 %72, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %579 = or disjoint i32 %209, 1024
  %580 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %579, i32 %80, i32 0)
  %581 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %579, i32 %85, i32 0)
  %582 = extractelement <4 x i32> %559, i64 0
  %583 = and i32 %582, 2147450879
  %584 = extractelement <4 x i32> %559, i64 1
  %585 = and i32 %584, 2147450879
  %586 = extractelement <4 x i32> %559, i64 2
  %587 = and i32 %586, 2147450879
  %588 = extractelement <4 x i32> %559, i64 3
  %589 = and i32 %588, 2147450879
  %590 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %583, i32 %585) #11, !srcloc !11
  %591 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %587, i32 %589) #11, !srcloc !11
  %592 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %590, i32 %591) #11, !srcloc !11
  %593 = trunc i32 %592 to i16
  %594 = lshr i32 %592, 16
  %595 = trunc nuw i32 %594 to i16
  %596 = tail call noundef i16 @llvm.umax.i16(i16 %593, i16 %595)
  %597 = zext i16 %596 to i32
  %598 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %597, i32 177, i32 15, i32 15, i1 true)
  %599 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %597, i32 %598)
  %600 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %599, i32 78, i32 15, i32 15, i1 true)
  %601 = tail call noundef i32 @llvm.umax.i32(i32 %599, i32 %600)
  %602 = shl i32 %601, 16
  %603 = add i32 %602, 2097152
  %604 = lshr i32 %603, 23
  %605 = and i32 %604, 255
  %606 = tail call i32 @llvm.umax.i32(i32 %605, i32 2)
  %607 = add nuw nsw i32 %606, 254
  %608 = and i32 %607, 255
  %609 = shl nuw nsw i32 %608, 23
  %610 = bitcast i32 %609 to float
  %611 = bitcast <4 x i32> %559 to <8 x bfloat>
  %612 = shufflevector <8 x bfloat> %611, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %613 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %612, float %610, i32 0)
  %614 = bitcast <4 x i32> %559 to <8 x bfloat>
  %615 = shufflevector <8 x bfloat> %614, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %616 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %613, <2 x bfloat> %615, float %610, i32 1)
  %617 = bitcast <4 x i32> %559 to <8 x bfloat>
  %618 = shufflevector <8 x bfloat> %617, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %619 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %616, <2 x bfloat> %618, float %610, i32 2)
  %620 = bitcast <4 x i32> %559 to <8 x bfloat>
  %621 = shufflevector <8 x bfloat> %620, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %622 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %619, <2 x bfloat> %621, float %610, i32 3)
  store i32 %622, ptr addrspace(3) %254, align 4, !tbaa !7
  %623 = extractelement <4 x i32> %560, i64 0
  %624 = and i32 %623, 2147450879
  %625 = extractelement <4 x i32> %560, i64 1
  %626 = and i32 %625, 2147450879
  %627 = extractelement <4 x i32> %560, i64 2
  %628 = and i32 %627, 2147450879
  %629 = extractelement <4 x i32> %560, i64 3
  %630 = and i32 %629, 2147450879
  %631 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %624, i32 %626) #11, !srcloc !11
  %632 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %628, i32 %630) #11, !srcloc !11
  %633 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %631, i32 %632) #11, !srcloc !11
  %634 = trunc i32 %633 to i16
  %635 = lshr i32 %633, 16
  %636 = trunc nuw i32 %635 to i16
  %637 = tail call noundef i16 @llvm.umax.i16(i16 %634, i16 %636)
  %638 = zext i16 %637 to i32
  %639 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %638, i32 177, i32 15, i32 15, i1 true)
  %640 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %638, i32 %639)
  %641 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %640, i32 78, i32 15, i32 15, i1 true)
  %642 = tail call noundef i32 @llvm.umax.i32(i32 %640, i32 %641)
  %643 = shl i32 %642, 16
  %644 = add i32 %643, 2097152
  %645 = lshr i32 %644, 23
  %646 = and i32 %645, 255
  %647 = tail call i32 @llvm.umax.i32(i32 %646, i32 2)
  %648 = add nuw nsw i32 %647, 254
  %649 = and i32 %648, 255
  %650 = shl nuw nsw i32 %649, 23
  %651 = bitcast i32 %650 to float
  %652 = bitcast <4 x i32> %560 to <8 x bfloat>
  %653 = shufflevector <8 x bfloat> %652, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %654 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %653, float %651, i32 0)
  %655 = bitcast <4 x i32> %560 to <8 x bfloat>
  %656 = shufflevector <8 x bfloat> %655, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %657 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %654, <2 x bfloat> %656, float %651, i32 1)
  %658 = bitcast <4 x i32> %560 to <8 x bfloat>
  %659 = shufflevector <8 x bfloat> %658, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %660 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %657, <2 x bfloat> %659, float %651, i32 2)
  %661 = bitcast <4 x i32> %560 to <8 x bfloat>
  %662 = shufflevector <8 x bfloat> %661, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %663 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %660, <2 x bfloat> %662, float %651, i32 3)
  store i32 %663, ptr addrspace(3) %303, align 4, !tbaa !7
  %664 = shl nuw nsw i32 %649, 16
  %665 = or disjoint i32 %664, %608
  %666 = add nuw nsw i32 %205, 1024
  %667 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %666
  store i32 %665, ptr addrspace(3) %667, align 4, !tbaa !7
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %668 = load <4 x i32>, ptr addrspace(3) %319, align 16, !tbaa !12
  %669 = load <4 x i32>, ptr addrspace(3) %323, align 16, !tbaa !12
  %670 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %464
  %671 = load i32, ptr addrspace(3) %670, align 4, !tbaa !7
  %672 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 2560, i32 0)
  %673 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 2816, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %674 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %668, <4 x i32> noundef %449, <4 x float> noundef %562, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %671, i32 noundef 0, i32 noundef %465) #12
  %675 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %669, <4 x i32> noundef %451, <4 x float> noundef %674, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %671, i32 noundef 2, i32 noundef %465) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %676 = or disjoint i32 %146, 10240
  %677 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %676, i32 %60, i32 2)
  %678 = or disjoint i32 %146, 11264
  %679 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %678, i32 %60, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %680 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %668, <4 x i32> noundef %454, <4 x float> noundef %568, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %671, i32 noundef 1, i32 noundef %465) #12
  %681 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %669, <4 x i32> noundef %455, <4 x float> noundef %680, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %671, i32 noundef 3, i32 noundef %465) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %682 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %676, i32 %64, i32 2)
  %683 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %678, i32 %64, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %684 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %668, <4 x i32> noundef %458, <4 x float> noundef %572, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %671, i32 noundef 0, i32 noundef %466) #12
  %685 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %669, <4 x i32> noundef %459, <4 x float> noundef %684, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %671, i32 noundef 2, i32 noundef %466) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %686 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %676, i32 %68, i32 2)
  %687 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %678, i32 %68, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %688 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %668, <4 x i32> noundef %462, <4 x float> noundef %576, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %671, i32 noundef 1, i32 noundef %466) #12
  %689 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %669, <4 x i32> noundef %463, <4 x float> noundef %688, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %671, i32 noundef 3, i32 noundef %466) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %690 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %676, i32 %72, i32 2)
  %691 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %678, i32 %72, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %692 = or disjoint i32 %209, 1280
  %693 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %692, i32 %80, i32 0)
  %694 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %692, i32 %85, i32 0)
  %695 = extractelement <4 x i32> %672, i64 0
  %696 = and i32 %695, 2147450879
  %697 = extractelement <4 x i32> %672, i64 1
  %698 = and i32 %697, 2147450879
  %699 = extractelement <4 x i32> %672, i64 2
  %700 = and i32 %699, 2147450879
  %701 = extractelement <4 x i32> %672, i64 3
  %702 = and i32 %701, 2147450879
  %703 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %696, i32 %698) #11, !srcloc !11
  %704 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %700, i32 %702) #11, !srcloc !11
  %705 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %703, i32 %704) #11, !srcloc !11
  %706 = trunc i32 %705 to i16
  %707 = lshr i32 %705, 16
  %708 = trunc nuw i32 %707 to i16
  %709 = tail call noundef i16 @llvm.umax.i16(i16 %706, i16 %708)
  %710 = zext i16 %709 to i32
  %711 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %710, i32 177, i32 15, i32 15, i1 true)
  %712 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %710, i32 %711)
  %713 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %712, i32 78, i32 15, i32 15, i1 true)
  %714 = tail call noundef i32 @llvm.umax.i32(i32 %712, i32 %713)
  %715 = shl i32 %714, 16
  %716 = add i32 %715, 2097152
  %717 = lshr i32 %716, 23
  %718 = and i32 %717, 255
  %719 = tail call i32 @llvm.umax.i32(i32 %718, i32 2)
  %720 = add nuw nsw i32 %719, 254
  %721 = and i32 %720, 255
  %722 = shl nuw nsw i32 %721, 23
  %723 = bitcast i32 %722 to float
  %724 = bitcast <4 x i32> %672 to <8 x bfloat>
  %725 = shufflevector <8 x bfloat> %724, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %726 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %725, float %723, i32 0)
  %727 = bitcast <4 x i32> %672 to <8 x bfloat>
  %728 = shufflevector <8 x bfloat> %727, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %729 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %726, <2 x bfloat> %728, float %723, i32 1)
  %730 = bitcast <4 x i32> %672 to <8 x bfloat>
  %731 = shufflevector <8 x bfloat> %730, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %732 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %729, <2 x bfloat> %731, float %723, i32 2)
  %733 = bitcast <4 x i32> %672 to <8 x bfloat>
  %734 = shufflevector <8 x bfloat> %733, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %735 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %732, <2 x bfloat> %734, float %723, i32 3)
  store i32 %735, ptr addrspace(3) %391, align 4, !tbaa !7
  %736 = extractelement <4 x i32> %673, i64 0
  %737 = and i32 %736, 2147450879
  %738 = extractelement <4 x i32> %673, i64 1
  %739 = and i32 %738, 2147450879
  %740 = extractelement <4 x i32> %673, i64 2
  %741 = and i32 %740, 2147450879
  %742 = extractelement <4 x i32> %673, i64 3
  %743 = and i32 %742, 2147450879
  %744 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %737, i32 %739) #11, !srcloc !11
  %745 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %741, i32 %743) #11, !srcloc !11
  %746 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %744, i32 %745) #11, !srcloc !11
  %747 = trunc i32 %746 to i16
  %748 = lshr i32 %746, 16
  %749 = trunc nuw i32 %748 to i16
  %750 = tail call noundef i16 @llvm.umax.i16(i16 %747, i16 %749)
  %751 = zext i16 %750 to i32
  %752 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %751, i32 177, i32 15, i32 15, i1 true)
  %753 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %751, i32 %752)
  %754 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %753, i32 78, i32 15, i32 15, i1 true)
  %755 = tail call noundef i32 @llvm.umax.i32(i32 %753, i32 %754)
  %756 = shl i32 %755, 16
  %757 = add i32 %756, 2097152
  %758 = lshr i32 %757, 23
  %759 = and i32 %758, 255
  %760 = tail call i32 @llvm.umax.i32(i32 %759, i32 2)
  %761 = add nuw nsw i32 %760, 254
  %762 = and i32 %761, 255
  %763 = shl nuw nsw i32 %762, 23
  %764 = bitcast i32 %763 to float
  %765 = bitcast <4 x i32> %673 to <8 x bfloat>
  %766 = shufflevector <8 x bfloat> %765, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %767 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %766, float %764, i32 0)
  %768 = bitcast <4 x i32> %673 to <8 x bfloat>
  %769 = shufflevector <8 x bfloat> %768, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %770 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %767, <2 x bfloat> %769, float %764, i32 1)
  %771 = bitcast <4 x i32> %673 to <8 x bfloat>
  %772 = shufflevector <8 x bfloat> %771, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %773 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %770, <2 x bfloat> %772, float %764, i32 2)
  %774 = bitcast <4 x i32> %673 to <8 x bfloat>
  %775 = shufflevector <8 x bfloat> %774, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %776 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %773, <2 x bfloat> %775, float %764, i32 3)
  store i32 %776, ptr addrspace(3) %433, align 4, !tbaa !7
  %777 = shl nuw nsw i32 %762, 16
  %778 = or disjoint i32 %777, %721
  %779 = add nuw nsw i32 %205, 1280
  %780 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %779
  store i32 %778, ptr addrspace(3) %780, align 4, !tbaa !7
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %781 = load <4 x i32>, ptr addrspace(3) %438, align 16, !tbaa !12
  %782 = load <4 x i32>, ptr addrspace(3) %440, align 16, !tbaa !12
  %783 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %579
  %784 = load i32, ptr addrspace(3) %783, align 4, !tbaa !7
  %785 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 3072, i32 0)
  %786 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 3328, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %787 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %781, <4 x i32> noundef %564, <4 x float> noundef %675, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %784, i32 noundef 0, i32 noundef %580) #12
  %788 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %782, <4 x i32> noundef %566, <4 x float> noundef %787, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %784, i32 noundef 2, i32 noundef %580) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %789 = or disjoint i32 %146, 12288
  %790 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %789, i32 %60, i32 2)
  %791 = or disjoint i32 %146, 13312
  %792 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %791, i32 %60, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %793 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %781, <4 x i32> noundef %569, <4 x float> noundef %681, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %784, i32 noundef 1, i32 noundef %580) #12
  %794 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %782, <4 x i32> noundef %570, <4 x float> noundef %793, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %784, i32 noundef 3, i32 noundef %580) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %795 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %789, i32 %64, i32 2)
  %796 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %791, i32 %64, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %797 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %781, <4 x i32> noundef %573, <4 x float> noundef %685, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %784, i32 noundef 0, i32 noundef %581) #12
  %798 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %782, <4 x i32> noundef %574, <4 x float> noundef %797, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %784, i32 noundef 2, i32 noundef %581) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %799 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %789, i32 %68, i32 2)
  %800 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %791, i32 %68, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %801 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %781, <4 x i32> noundef %577, <4 x float> noundef %689, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %784, i32 noundef 1, i32 noundef %581) #12
  %802 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %782, <4 x i32> noundef %578, <4 x float> noundef %801, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %784, i32 noundef 3, i32 noundef %581) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %803 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %789, i32 %72, i32 2)
  %804 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %791, i32 %72, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %805 = or disjoint i32 %209, 1536
  %806 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %805, i32 %80, i32 0)
  %807 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %805, i32 %85, i32 0)
  %808 = extractelement <4 x i32> %785, i64 0
  %809 = and i32 %808, 2147450879
  %810 = extractelement <4 x i32> %785, i64 1
  %811 = and i32 %810, 2147450879
  %812 = extractelement <4 x i32> %785, i64 2
  %813 = and i32 %812, 2147450879
  %814 = extractelement <4 x i32> %785, i64 3
  %815 = and i32 %814, 2147450879
  %816 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %809, i32 %811) #11, !srcloc !11
  %817 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %813, i32 %815) #11, !srcloc !11
  %818 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %816, i32 %817) #11, !srcloc !11
  %819 = trunc i32 %818 to i16
  %820 = lshr i32 %818, 16
  %821 = trunc nuw i32 %820 to i16
  %822 = tail call noundef i16 @llvm.umax.i16(i16 %819, i16 %821)
  %823 = zext i16 %822 to i32
  %824 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %823, i32 177, i32 15, i32 15, i1 true)
  %825 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %823, i32 %824)
  %826 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %825, i32 78, i32 15, i32 15, i1 true)
  %827 = tail call noundef i32 @llvm.umax.i32(i32 %825, i32 %826)
  %828 = shl i32 %827, 16
  %829 = add i32 %828, 2097152
  %830 = lshr i32 %829, 23
  %831 = and i32 %830, 255
  %832 = tail call i32 @llvm.umax.i32(i32 %831, i32 2)
  %833 = add nuw nsw i32 %832, 254
  %834 = and i32 %833, 255
  %835 = shl nuw nsw i32 %834, 23
  %836 = bitcast i32 %835 to float
  %837 = bitcast <4 x i32> %785 to <8 x bfloat>
  %838 = shufflevector <8 x bfloat> %837, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %839 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %838, float %836, i32 0)
  %840 = bitcast <4 x i32> %785 to <8 x bfloat>
  %841 = shufflevector <8 x bfloat> %840, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %842 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %839, <2 x bfloat> %841, float %836, i32 1)
  %843 = bitcast <4 x i32> %785 to <8 x bfloat>
  %844 = shufflevector <8 x bfloat> %843, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %845 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %842, <2 x bfloat> %844, float %836, i32 2)
  %846 = bitcast <4 x i32> %785 to <8 x bfloat>
  %847 = shufflevector <8 x bfloat> %846, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %848 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %845, <2 x bfloat> %847, float %836, i32 3)
  store i32 %848, ptr addrspace(3) %142, align 4, !tbaa !7
  %849 = extractelement <4 x i32> %786, i64 0
  %850 = and i32 %849, 2147450879
  %851 = extractelement <4 x i32> %786, i64 1
  %852 = and i32 %851, 2147450879
  %853 = extractelement <4 x i32> %786, i64 2
  %854 = and i32 %853, 2147450879
  %855 = extractelement <4 x i32> %786, i64 3
  %856 = and i32 %855, 2147450879
  %857 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %850, i32 %852) #11, !srcloc !11
  %858 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %854, i32 %856) #11, !srcloc !11
  %859 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %857, i32 %858) #11, !srcloc !11
  %860 = trunc i32 %859 to i16
  %861 = lshr i32 %859, 16
  %862 = trunc nuw i32 %861 to i16
  %863 = tail call noundef i16 @llvm.umax.i16(i16 %860, i16 %862)
  %864 = zext i16 %863 to i32
  %865 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %864, i32 177, i32 15, i32 15, i1 true)
  %866 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %864, i32 %865)
  %867 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %866, i32 78, i32 15, i32 15, i1 true)
  %868 = tail call noundef i32 @llvm.umax.i32(i32 %866, i32 %867)
  %869 = shl i32 %868, 16
  %870 = add i32 %869, 2097152
  %871 = lshr i32 %870, 23
  %872 = and i32 %871, 255
  %873 = tail call i32 @llvm.umax.i32(i32 %872, i32 2)
  %874 = add nuw nsw i32 %873, 254
  %875 = and i32 %874, 255
  %876 = shl nuw nsw i32 %875, 23
  %877 = bitcast i32 %876 to float
  %878 = bitcast <4 x i32> %786 to <8 x bfloat>
  %879 = shufflevector <8 x bfloat> %878, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %880 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %879, float %877, i32 0)
  %881 = bitcast <4 x i32> %786 to <8 x bfloat>
  %882 = shufflevector <8 x bfloat> %881, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %883 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %880, <2 x bfloat> %882, float %877, i32 1)
  %884 = bitcast <4 x i32> %786 to <8 x bfloat>
  %885 = shufflevector <8 x bfloat> %884, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %886 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %883, <2 x bfloat> %885, float %877, i32 2)
  %887 = bitcast <4 x i32> %786 to <8 x bfloat>
  %888 = shufflevector <8 x bfloat> %887, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %889 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %886, <2 x bfloat> %888, float %877, i32 3)
  store i32 %889, ptr addrspace(3) %197, align 4, !tbaa !7
  %890 = shl nuw nsw i32 %875, 16
  %891 = or disjoint i32 %890, %834
  %892 = add nuw nsw i32 %205, 1536
  %893 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %892
  store i32 %891, ptr addrspace(3) %893, align 4, !tbaa !7
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %894 = load <4 x i32>, ptr addrspace(3) %553, align 16, !tbaa !12
  %895 = load <4 x i32>, ptr addrspace(3) %555, align 16, !tbaa !12
  %896 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %692
  %897 = load i32, ptr addrspace(3) %896, align 4, !tbaa !7
  %898 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 3584, i32 0)
  %899 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 3840, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %900 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %894, <4 x i32> noundef %677, <4 x float> noundef %788, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %897, i32 noundef 0, i32 noundef %693) #12
  %901 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %895, <4 x i32> noundef %679, <4 x float> noundef %900, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %897, i32 noundef 2, i32 noundef %693) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %902 = or disjoint i32 %146, 14336
  %903 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %902, i32 %60, i32 2)
  %904 = or disjoint i32 %146, 15360
  %905 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %904, i32 %60, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %906 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %894, <4 x i32> noundef %682, <4 x float> noundef %794, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %897, i32 noundef 1, i32 noundef %693) #12
  %907 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %895, <4 x i32> noundef %683, <4 x float> noundef %906, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %897, i32 noundef 3, i32 noundef %693) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %908 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %902, i32 %64, i32 2)
  %909 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %904, i32 %64, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %910 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %894, <4 x i32> noundef %686, <4 x float> noundef %798, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %897, i32 noundef 0, i32 noundef %694) #12
  %911 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %895, <4 x i32> noundef %687, <4 x float> noundef %910, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %897, i32 noundef 2, i32 noundef %694) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %912 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %902, i32 %68, i32 2)
  %913 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %904, i32 %68, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %914 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %894, <4 x i32> noundef %690, <4 x float> noundef %802, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %897, i32 noundef 1, i32 noundef %694) #12
  %915 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %895, <4 x i32> noundef %691, <4 x float> noundef %914, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %897, i32 noundef 3, i32 noundef %694) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %916 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %902, i32 %72, i32 2)
  %917 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %904, i32 %72, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %918 = or disjoint i32 %209, 1792
  %919 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %918, i32 %80, i32 0)
  %920 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %918, i32 %85, i32 0)
  %921 = extractelement <4 x i32> %898, i64 0
  %922 = and i32 %921, 2147450879
  %923 = extractelement <4 x i32> %898, i64 1
  %924 = and i32 %923, 2147450879
  %925 = extractelement <4 x i32> %898, i64 2
  %926 = and i32 %925, 2147450879
  %927 = extractelement <4 x i32> %898, i64 3
  %928 = and i32 %927, 2147450879
  %929 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %922, i32 %924) #11, !srcloc !11
  %930 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %926, i32 %928) #11, !srcloc !11
  %931 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %929, i32 %930) #11, !srcloc !11
  %932 = trunc i32 %931 to i16
  %933 = lshr i32 %931, 16
  %934 = trunc nuw i32 %933 to i16
  %935 = tail call noundef i16 @llvm.umax.i16(i16 %932, i16 %934)
  %936 = zext i16 %935 to i32
  %937 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %936, i32 177, i32 15, i32 15, i1 true)
  %938 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %936, i32 %937)
  %939 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %938, i32 78, i32 15, i32 15, i1 true)
  %940 = tail call noundef i32 @llvm.umax.i32(i32 %938, i32 %939)
  %941 = shl i32 %940, 16
  %942 = add i32 %941, 2097152
  %943 = lshr i32 %942, 23
  %944 = and i32 %943, 255
  %945 = tail call i32 @llvm.umax.i32(i32 %944, i32 2)
  %946 = add nuw nsw i32 %945, 254
  %947 = and i32 %946, 255
  %948 = shl nuw nsw i32 %947, 23
  %949 = bitcast i32 %948 to float
  %950 = bitcast <4 x i32> %898 to <8 x bfloat>
  %951 = shufflevector <8 x bfloat> %950, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %952 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %951, float %949, i32 0)
  %953 = bitcast <4 x i32> %898 to <8 x bfloat>
  %954 = shufflevector <8 x bfloat> %953, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %955 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %952, <2 x bfloat> %954, float %949, i32 1)
  %956 = bitcast <4 x i32> %898 to <8 x bfloat>
  %957 = shufflevector <8 x bfloat> %956, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %958 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %955, <2 x bfloat> %957, float %949, i32 2)
  %959 = bitcast <4 x i32> %898 to <8 x bfloat>
  %960 = shufflevector <8 x bfloat> %959, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %961 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %958, <2 x bfloat> %960, float %949, i32 3)
  store i32 %961, ptr addrspace(3) %254, align 4, !tbaa !7
  %962 = extractelement <4 x i32> %899, i64 0
  %963 = and i32 %962, 2147450879
  %964 = extractelement <4 x i32> %899, i64 1
  %965 = and i32 %964, 2147450879
  %966 = extractelement <4 x i32> %899, i64 2
  %967 = and i32 %966, 2147450879
  %968 = extractelement <4 x i32> %899, i64 3
  %969 = and i32 %968, 2147450879
  %970 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %963, i32 %965) #11, !srcloc !11
  %971 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %967, i32 %969) #11, !srcloc !11
  %972 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %970, i32 %971) #11, !srcloc !11
  %973 = trunc i32 %972 to i16
  %974 = lshr i32 %972, 16
  %975 = trunc nuw i32 %974 to i16
  %976 = tail call noundef i16 @llvm.umax.i16(i16 %973, i16 %975)
  %977 = zext i16 %976 to i32
  %978 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %977, i32 177, i32 15, i32 15, i1 true)
  %979 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %977, i32 %978)
  %980 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %979, i32 78, i32 15, i32 15, i1 true)
  %981 = tail call noundef i32 @llvm.umax.i32(i32 %979, i32 %980)
  %982 = shl i32 %981, 16
  %983 = add i32 %982, 2097152
  %984 = lshr i32 %983, 23
  %985 = and i32 %984, 255
  %986 = tail call i32 @llvm.umax.i32(i32 %985, i32 2)
  %987 = add nuw nsw i32 %986, 254
  %988 = and i32 %987, 255
  %989 = shl nuw nsw i32 %988, 23
  %990 = bitcast i32 %989 to float
  %991 = bitcast <4 x i32> %899 to <8 x bfloat>
  %992 = shufflevector <8 x bfloat> %991, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %993 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %992, float %990, i32 0)
  %994 = bitcast <4 x i32> %899 to <8 x bfloat>
  %995 = shufflevector <8 x bfloat> %994, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %996 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %993, <2 x bfloat> %995, float %990, i32 1)
  %997 = bitcast <4 x i32> %899 to <8 x bfloat>
  %998 = shufflevector <8 x bfloat> %997, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %999 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %996, <2 x bfloat> %998, float %990, i32 2)
  %1000 = bitcast <4 x i32> %899 to <8 x bfloat>
  %1001 = shufflevector <8 x bfloat> %1000, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1002 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %999, <2 x bfloat> %1001, float %990, i32 3)
  store i32 %1002, ptr addrspace(3) %303, align 4, !tbaa !7
  %1003 = shl nuw nsw i32 %988, 16
  %1004 = or disjoint i32 %1003, %947
  %1005 = add nuw nsw i32 %205, 1792
  %1006 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %1005
  store i32 %1004, ptr addrspace(3) %1006, align 4, !tbaa !7
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1007 = load <4 x i32>, ptr addrspace(3) %319, align 16, !tbaa !12
  %1008 = load <4 x i32>, ptr addrspace(3) %323, align 16, !tbaa !12
  %1009 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %805
  %1010 = load i32, ptr addrspace(3) %1009, align 4, !tbaa !7
  %1011 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 4096, i32 0)
  %1012 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 4352, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1013 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1007, <4 x i32> noundef %790, <4 x float> noundef %901, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1010, i32 noundef 0, i32 noundef %806) #12
  %1014 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1008, <4 x i32> noundef %792, <4 x float> noundef %1013, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1010, i32 noundef 2, i32 noundef %806) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1015 = or disjoint i32 %146, 16384
  %1016 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1015, i32 %60, i32 2)
  %1017 = or disjoint i32 %146, 17408
  %1018 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1017, i32 %60, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1019 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1007, <4 x i32> noundef %795, <4 x float> noundef %907, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1010, i32 noundef 1, i32 noundef %806) #12
  %1020 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1008, <4 x i32> noundef %796, <4 x float> noundef %1019, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1010, i32 noundef 3, i32 noundef %806) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1021 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1015, i32 %64, i32 2)
  %1022 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1017, i32 %64, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1023 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1007, <4 x i32> noundef %799, <4 x float> noundef %911, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1010, i32 noundef 0, i32 noundef %807) #12
  %1024 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1008, <4 x i32> noundef %800, <4 x float> noundef %1023, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1010, i32 noundef 2, i32 noundef %807) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1025 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1015, i32 %68, i32 2)
  %1026 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1017, i32 %68, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1027 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1007, <4 x i32> noundef %803, <4 x float> noundef %915, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1010, i32 noundef 1, i32 noundef %807) #12
  %1028 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1008, <4 x i32> noundef %804, <4 x float> noundef %1027, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1010, i32 noundef 3, i32 noundef %807) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1029 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1015, i32 %72, i32 2)
  %1030 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1017, i32 %72, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1031 = or disjoint i32 %209, 2048
  %1032 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %1031, i32 %80, i32 0)
  %1033 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %1031, i32 %85, i32 0)
  %1034 = extractelement <4 x i32> %1011, i64 0
  %1035 = and i32 %1034, 2147450879
  %1036 = extractelement <4 x i32> %1011, i64 1
  %1037 = and i32 %1036, 2147450879
  %1038 = extractelement <4 x i32> %1011, i64 2
  %1039 = and i32 %1038, 2147450879
  %1040 = extractelement <4 x i32> %1011, i64 3
  %1041 = and i32 %1040, 2147450879
  %1042 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1035, i32 %1037) #11, !srcloc !11
  %1043 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1039, i32 %1041) #11, !srcloc !11
  %1044 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1042, i32 %1043) #11, !srcloc !11
  %1045 = trunc i32 %1044 to i16
  %1046 = lshr i32 %1044, 16
  %1047 = trunc nuw i32 %1046 to i16
  %1048 = tail call noundef i16 @llvm.umax.i16(i16 %1045, i16 %1047)
  %1049 = zext i16 %1048 to i32
  %1050 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %1049, i32 177, i32 15, i32 15, i1 true)
  %1051 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %1049, i32 %1050)
  %1052 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1051, i32 78, i32 15, i32 15, i1 true)
  %1053 = tail call noundef i32 @llvm.umax.i32(i32 %1051, i32 %1052)
  %1054 = shl i32 %1053, 16
  %1055 = add i32 %1054, 2097152
  %1056 = lshr i32 %1055, 23
  %1057 = and i32 %1056, 255
  %1058 = tail call i32 @llvm.umax.i32(i32 %1057, i32 2)
  %1059 = add nuw nsw i32 %1058, 254
  %1060 = and i32 %1059, 255
  %1061 = shl nuw nsw i32 %1060, 23
  %1062 = bitcast i32 %1061 to float
  %1063 = bitcast <4 x i32> %1011 to <8 x bfloat>
  %1064 = shufflevector <8 x bfloat> %1063, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1065 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %1064, float %1062, i32 0)
  %1066 = bitcast <4 x i32> %1011 to <8 x bfloat>
  %1067 = shufflevector <8 x bfloat> %1066, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1068 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1065, <2 x bfloat> %1067, float %1062, i32 1)
  %1069 = bitcast <4 x i32> %1011 to <8 x bfloat>
  %1070 = shufflevector <8 x bfloat> %1069, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1071 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1068, <2 x bfloat> %1070, float %1062, i32 2)
  %1072 = bitcast <4 x i32> %1011 to <8 x bfloat>
  %1073 = shufflevector <8 x bfloat> %1072, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1074 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1071, <2 x bfloat> %1073, float %1062, i32 3)
  store i32 %1074, ptr addrspace(3) %391, align 4, !tbaa !7
  %1075 = extractelement <4 x i32> %1012, i64 0
  %1076 = and i32 %1075, 2147450879
  %1077 = extractelement <4 x i32> %1012, i64 1
  %1078 = and i32 %1077, 2147450879
  %1079 = extractelement <4 x i32> %1012, i64 2
  %1080 = and i32 %1079, 2147450879
  %1081 = extractelement <4 x i32> %1012, i64 3
  %1082 = and i32 %1081, 2147450879
  %1083 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1076, i32 %1078) #11, !srcloc !11
  %1084 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1080, i32 %1082) #11, !srcloc !11
  %1085 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1083, i32 %1084) #11, !srcloc !11
  %1086 = trunc i32 %1085 to i16
  %1087 = lshr i32 %1085, 16
  %1088 = trunc nuw i32 %1087 to i16
  %1089 = tail call noundef i16 @llvm.umax.i16(i16 %1086, i16 %1088)
  %1090 = zext i16 %1089 to i32
  %1091 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %1090, i32 177, i32 15, i32 15, i1 true)
  %1092 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %1090, i32 %1091)
  %1093 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1092, i32 78, i32 15, i32 15, i1 true)
  %1094 = tail call noundef i32 @llvm.umax.i32(i32 %1092, i32 %1093)
  %1095 = shl i32 %1094, 16
  %1096 = add i32 %1095, 2097152
  %1097 = lshr i32 %1096, 23
  %1098 = and i32 %1097, 255
  %1099 = tail call i32 @llvm.umax.i32(i32 %1098, i32 2)
  %1100 = add nuw nsw i32 %1099, 254
  %1101 = and i32 %1100, 255
  %1102 = shl nuw nsw i32 %1101, 23
  %1103 = bitcast i32 %1102 to float
  %1104 = bitcast <4 x i32> %1012 to <8 x bfloat>
  %1105 = shufflevector <8 x bfloat> %1104, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1106 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %1105, float %1103, i32 0)
  %1107 = bitcast <4 x i32> %1012 to <8 x bfloat>
  %1108 = shufflevector <8 x bfloat> %1107, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1109 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1106, <2 x bfloat> %1108, float %1103, i32 1)
  %1110 = bitcast <4 x i32> %1012 to <8 x bfloat>
  %1111 = shufflevector <8 x bfloat> %1110, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1112 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1109, <2 x bfloat> %1111, float %1103, i32 2)
  %1113 = bitcast <4 x i32> %1012 to <8 x bfloat>
  %1114 = shufflevector <8 x bfloat> %1113, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1115 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1112, <2 x bfloat> %1114, float %1103, i32 3)
  store i32 %1115, ptr addrspace(3) %433, align 4, !tbaa !7
  %1116 = shl nuw nsw i32 %1101, 16
  %1117 = or disjoint i32 %1116, %1060
  %1118 = add nuw nsw i32 %205, 2048
  %1119 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %1118
  store i32 %1117, ptr addrspace(3) %1119, align 4, !tbaa !7
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1120 = load <4 x i32>, ptr addrspace(3) %438, align 16, !tbaa !12
  %1121 = load <4 x i32>, ptr addrspace(3) %440, align 16, !tbaa !12
  %1122 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %918
  %1123 = load i32, ptr addrspace(3) %1122, align 4, !tbaa !7
  %1124 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 4608, i32 0)
  %1125 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 4864, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1126 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1120, <4 x i32> noundef %903, <4 x float> noundef %1014, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1123, i32 noundef 0, i32 noundef %919) #12
  %1127 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1121, <4 x i32> noundef %905, <4 x float> noundef %1126, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1123, i32 noundef 2, i32 noundef %919) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1128 = or disjoint i32 %146, 18432
  %1129 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1128, i32 %60, i32 2)
  %1130 = or disjoint i32 %146, 19456
  %1131 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1130, i32 %60, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1132 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1120, <4 x i32> noundef %908, <4 x float> noundef %1020, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1123, i32 noundef 1, i32 noundef %919) #12
  %1133 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1121, <4 x i32> noundef %909, <4 x float> noundef %1132, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1123, i32 noundef 3, i32 noundef %919) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1134 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1128, i32 %64, i32 2)
  %1135 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1130, i32 %64, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1136 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1120, <4 x i32> noundef %912, <4 x float> noundef %1024, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1123, i32 noundef 0, i32 noundef %920) #12
  %1137 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1121, <4 x i32> noundef %913, <4 x float> noundef %1136, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1123, i32 noundef 2, i32 noundef %920) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1138 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1128, i32 %68, i32 2)
  %1139 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1130, i32 %68, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1140 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1120, <4 x i32> noundef %916, <4 x float> noundef %1028, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1123, i32 noundef 1, i32 noundef %920) #12
  %1141 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1121, <4 x i32> noundef %917, <4 x float> noundef %1140, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1123, i32 noundef 3, i32 noundef %920) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1142 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1128, i32 %72, i32 2)
  %1143 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1130, i32 %72, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1144 = or disjoint i32 %209, 2304
  %1145 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %1144, i32 %80, i32 0)
  %1146 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %1144, i32 %85, i32 0)
  %1147 = extractelement <4 x i32> %1124, i64 0
  %1148 = and i32 %1147, 2147450879
  %1149 = extractelement <4 x i32> %1124, i64 1
  %1150 = and i32 %1149, 2147450879
  %1151 = extractelement <4 x i32> %1124, i64 2
  %1152 = and i32 %1151, 2147450879
  %1153 = extractelement <4 x i32> %1124, i64 3
  %1154 = and i32 %1153, 2147450879
  %1155 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1148, i32 %1150) #11, !srcloc !11
  %1156 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1152, i32 %1154) #11, !srcloc !11
  %1157 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1155, i32 %1156) #11, !srcloc !11
  %1158 = trunc i32 %1157 to i16
  %1159 = lshr i32 %1157, 16
  %1160 = trunc nuw i32 %1159 to i16
  %1161 = tail call noundef i16 @llvm.umax.i16(i16 %1158, i16 %1160)
  %1162 = zext i16 %1161 to i32
  %1163 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %1162, i32 177, i32 15, i32 15, i1 true)
  %1164 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %1162, i32 %1163)
  %1165 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1164, i32 78, i32 15, i32 15, i1 true)
  %1166 = tail call noundef i32 @llvm.umax.i32(i32 %1164, i32 %1165)
  %1167 = shl i32 %1166, 16
  %1168 = add i32 %1167, 2097152
  %1169 = lshr i32 %1168, 23
  %1170 = and i32 %1169, 255
  %1171 = tail call i32 @llvm.umax.i32(i32 %1170, i32 2)
  %1172 = add nuw nsw i32 %1171, 254
  %1173 = and i32 %1172, 255
  %1174 = shl nuw nsw i32 %1173, 23
  %1175 = bitcast i32 %1174 to float
  %1176 = bitcast <4 x i32> %1124 to <8 x bfloat>
  %1177 = shufflevector <8 x bfloat> %1176, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1178 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %1177, float %1175, i32 0)
  %1179 = bitcast <4 x i32> %1124 to <8 x bfloat>
  %1180 = shufflevector <8 x bfloat> %1179, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1181 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1178, <2 x bfloat> %1180, float %1175, i32 1)
  %1182 = bitcast <4 x i32> %1124 to <8 x bfloat>
  %1183 = shufflevector <8 x bfloat> %1182, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1184 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1181, <2 x bfloat> %1183, float %1175, i32 2)
  %1185 = bitcast <4 x i32> %1124 to <8 x bfloat>
  %1186 = shufflevector <8 x bfloat> %1185, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1187 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1184, <2 x bfloat> %1186, float %1175, i32 3)
  store i32 %1187, ptr addrspace(3) %142, align 4, !tbaa !7
  %1188 = extractelement <4 x i32> %1125, i64 0
  %1189 = and i32 %1188, 2147450879
  %1190 = extractelement <4 x i32> %1125, i64 1
  %1191 = and i32 %1190, 2147450879
  %1192 = extractelement <4 x i32> %1125, i64 2
  %1193 = and i32 %1192, 2147450879
  %1194 = extractelement <4 x i32> %1125, i64 3
  %1195 = and i32 %1194, 2147450879
  %1196 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1189, i32 %1191) #11, !srcloc !11
  %1197 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1193, i32 %1195) #11, !srcloc !11
  %1198 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1196, i32 %1197) #11, !srcloc !11
  %1199 = trunc i32 %1198 to i16
  %1200 = lshr i32 %1198, 16
  %1201 = trunc nuw i32 %1200 to i16
  %1202 = tail call noundef i16 @llvm.umax.i16(i16 %1199, i16 %1201)
  %1203 = zext i16 %1202 to i32
  %1204 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %1203, i32 177, i32 15, i32 15, i1 true)
  %1205 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %1203, i32 %1204)
  %1206 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1205, i32 78, i32 15, i32 15, i1 true)
  %1207 = tail call noundef i32 @llvm.umax.i32(i32 %1205, i32 %1206)
  %1208 = shl i32 %1207, 16
  %1209 = add i32 %1208, 2097152
  %1210 = lshr i32 %1209, 23
  %1211 = and i32 %1210, 255
  %1212 = tail call i32 @llvm.umax.i32(i32 %1211, i32 2)
  %1213 = add nuw nsw i32 %1212, 254
  %1214 = and i32 %1213, 255
  %1215 = shl nuw nsw i32 %1214, 23
  %1216 = bitcast i32 %1215 to float
  %1217 = bitcast <4 x i32> %1125 to <8 x bfloat>
  %1218 = shufflevector <8 x bfloat> %1217, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1219 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %1218, float %1216, i32 0)
  %1220 = bitcast <4 x i32> %1125 to <8 x bfloat>
  %1221 = shufflevector <8 x bfloat> %1220, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1222 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1219, <2 x bfloat> %1221, float %1216, i32 1)
  %1223 = bitcast <4 x i32> %1125 to <8 x bfloat>
  %1224 = shufflevector <8 x bfloat> %1223, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1225 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1222, <2 x bfloat> %1224, float %1216, i32 2)
  %1226 = bitcast <4 x i32> %1125 to <8 x bfloat>
  %1227 = shufflevector <8 x bfloat> %1226, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1228 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1225, <2 x bfloat> %1227, float %1216, i32 3)
  store i32 %1228, ptr addrspace(3) %197, align 4, !tbaa !7
  %1229 = shl nuw nsw i32 %1214, 16
  %1230 = or disjoint i32 %1229, %1173
  %1231 = add nuw nsw i32 %205, 2304
  %1232 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %1231
  store i32 %1230, ptr addrspace(3) %1232, align 4, !tbaa !7
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1233 = load <4 x i32>, ptr addrspace(3) %553, align 16, !tbaa !12
  %1234 = load <4 x i32>, ptr addrspace(3) %555, align 16, !tbaa !12
  %1235 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %1031
  %1236 = load i32, ptr addrspace(3) %1235, align 4, !tbaa !7
  %1237 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 5120, i32 0)
  %1238 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 5376, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1239 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1233, <4 x i32> noundef %1016, <4 x float> noundef %1127, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1236, i32 noundef 0, i32 noundef %1032) #12
  %1240 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1234, <4 x i32> noundef %1018, <4 x float> noundef %1239, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1236, i32 noundef 2, i32 noundef %1032) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1241 = or disjoint i32 %146, 20480
  %1242 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1241, i32 %60, i32 2)
  %1243 = or disjoint i32 %146, 21504
  %1244 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1243, i32 %60, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1245 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1233, <4 x i32> noundef %1021, <4 x float> noundef %1133, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1236, i32 noundef 1, i32 noundef %1032) #12
  %1246 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1234, <4 x i32> noundef %1022, <4 x float> noundef %1245, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1236, i32 noundef 3, i32 noundef %1032) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1247 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1241, i32 %64, i32 2)
  %1248 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1243, i32 %64, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1249 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1233, <4 x i32> noundef %1025, <4 x float> noundef %1137, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1236, i32 noundef 0, i32 noundef %1033) #12
  %1250 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1234, <4 x i32> noundef %1026, <4 x float> noundef %1249, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1236, i32 noundef 2, i32 noundef %1033) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1251 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1241, i32 %68, i32 2)
  %1252 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1243, i32 %68, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1253 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1233, <4 x i32> noundef %1029, <4 x float> noundef %1141, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1236, i32 noundef 1, i32 noundef %1033) #12
  %1254 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1234, <4 x i32> noundef %1030, <4 x float> noundef %1253, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1236, i32 noundef 3, i32 noundef %1033) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1255 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1241, i32 %72, i32 2)
  %1256 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1243, i32 %72, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1257 = or disjoint i32 %209, 2560
  %1258 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %1257, i32 %80, i32 0)
  %1259 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %1257, i32 %85, i32 0)
  %1260 = extractelement <4 x i32> %1237, i64 0
  %1261 = and i32 %1260, 2147450879
  %1262 = extractelement <4 x i32> %1237, i64 1
  %1263 = and i32 %1262, 2147450879
  %1264 = extractelement <4 x i32> %1237, i64 2
  %1265 = and i32 %1264, 2147450879
  %1266 = extractelement <4 x i32> %1237, i64 3
  %1267 = and i32 %1266, 2147450879
  %1268 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1261, i32 %1263) #11, !srcloc !11
  %1269 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1265, i32 %1267) #11, !srcloc !11
  %1270 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1268, i32 %1269) #11, !srcloc !11
  %1271 = trunc i32 %1270 to i16
  %1272 = lshr i32 %1270, 16
  %1273 = trunc nuw i32 %1272 to i16
  %1274 = tail call noundef i16 @llvm.umax.i16(i16 %1271, i16 %1273)
  %1275 = zext i16 %1274 to i32
  %1276 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %1275, i32 177, i32 15, i32 15, i1 true)
  %1277 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %1275, i32 %1276)
  %1278 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1277, i32 78, i32 15, i32 15, i1 true)
  %1279 = tail call noundef i32 @llvm.umax.i32(i32 %1277, i32 %1278)
  %1280 = shl i32 %1279, 16
  %1281 = add i32 %1280, 2097152
  %1282 = lshr i32 %1281, 23
  %1283 = and i32 %1282, 255
  %1284 = tail call i32 @llvm.umax.i32(i32 %1283, i32 2)
  %1285 = add nuw nsw i32 %1284, 254
  %1286 = and i32 %1285, 255
  %1287 = shl nuw nsw i32 %1286, 23
  %1288 = bitcast i32 %1287 to float
  %1289 = bitcast <4 x i32> %1237 to <8 x bfloat>
  %1290 = shufflevector <8 x bfloat> %1289, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1291 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %1290, float %1288, i32 0)
  %1292 = bitcast <4 x i32> %1237 to <8 x bfloat>
  %1293 = shufflevector <8 x bfloat> %1292, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1294 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1291, <2 x bfloat> %1293, float %1288, i32 1)
  %1295 = bitcast <4 x i32> %1237 to <8 x bfloat>
  %1296 = shufflevector <8 x bfloat> %1295, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1297 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1294, <2 x bfloat> %1296, float %1288, i32 2)
  %1298 = bitcast <4 x i32> %1237 to <8 x bfloat>
  %1299 = shufflevector <8 x bfloat> %1298, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1300 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1297, <2 x bfloat> %1299, float %1288, i32 3)
  store i32 %1300, ptr addrspace(3) %254, align 4, !tbaa !7
  %1301 = extractelement <4 x i32> %1238, i64 0
  %1302 = and i32 %1301, 2147450879
  %1303 = extractelement <4 x i32> %1238, i64 1
  %1304 = and i32 %1303, 2147450879
  %1305 = extractelement <4 x i32> %1238, i64 2
  %1306 = and i32 %1305, 2147450879
  %1307 = extractelement <4 x i32> %1238, i64 3
  %1308 = and i32 %1307, 2147450879
  %1309 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1302, i32 %1304) #11, !srcloc !11
  %1310 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1306, i32 %1308) #11, !srcloc !11
  %1311 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1309, i32 %1310) #11, !srcloc !11
  %1312 = trunc i32 %1311 to i16
  %1313 = lshr i32 %1311, 16
  %1314 = trunc nuw i32 %1313 to i16
  %1315 = tail call noundef i16 @llvm.umax.i16(i16 %1312, i16 %1314)
  %1316 = zext i16 %1315 to i32
  %1317 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %1316, i32 177, i32 15, i32 15, i1 true)
  %1318 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %1316, i32 %1317)
  %1319 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1318, i32 78, i32 15, i32 15, i1 true)
  %1320 = tail call noundef i32 @llvm.umax.i32(i32 %1318, i32 %1319)
  %1321 = shl i32 %1320, 16
  %1322 = add i32 %1321, 2097152
  %1323 = lshr i32 %1322, 23
  %1324 = and i32 %1323, 255
  %1325 = tail call i32 @llvm.umax.i32(i32 %1324, i32 2)
  %1326 = add nuw nsw i32 %1325, 254
  %1327 = and i32 %1326, 255
  %1328 = shl nuw nsw i32 %1327, 23
  %1329 = bitcast i32 %1328 to float
  %1330 = bitcast <4 x i32> %1238 to <8 x bfloat>
  %1331 = shufflevector <8 x bfloat> %1330, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1332 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %1331, float %1329, i32 0)
  %1333 = bitcast <4 x i32> %1238 to <8 x bfloat>
  %1334 = shufflevector <8 x bfloat> %1333, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1335 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1332, <2 x bfloat> %1334, float %1329, i32 1)
  %1336 = bitcast <4 x i32> %1238 to <8 x bfloat>
  %1337 = shufflevector <8 x bfloat> %1336, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1338 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1335, <2 x bfloat> %1337, float %1329, i32 2)
  %1339 = bitcast <4 x i32> %1238 to <8 x bfloat>
  %1340 = shufflevector <8 x bfloat> %1339, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1341 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1338, <2 x bfloat> %1340, float %1329, i32 3)
  store i32 %1341, ptr addrspace(3) %303, align 4, !tbaa !7
  %1342 = shl nuw nsw i32 %1327, 16
  %1343 = or disjoint i32 %1342, %1286
  %1344 = add nuw nsw i32 %205, 2560
  %1345 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %1344
  store i32 %1343, ptr addrspace(3) %1345, align 4, !tbaa !7
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1346 = load <4 x i32>, ptr addrspace(3) %319, align 16, !tbaa !12
  %1347 = load <4 x i32>, ptr addrspace(3) %323, align 16, !tbaa !12
  %1348 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %1144
  %1349 = load i32, ptr addrspace(3) %1348, align 4, !tbaa !7
  %1350 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 5632, i32 0)
  %1351 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 5888, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1352 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1346, <4 x i32> noundef %1129, <4 x float> noundef %1240, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1349, i32 noundef 0, i32 noundef %1145) #12
  %1353 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1347, <4 x i32> noundef %1131, <4 x float> noundef %1352, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1349, i32 noundef 2, i32 noundef %1145) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1354 = or disjoint i32 %146, 22528
  %1355 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1354, i32 %60, i32 2)
  %1356 = or disjoint i32 %146, 23552
  %1357 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1356, i32 %60, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1358 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1346, <4 x i32> noundef %1134, <4 x float> noundef %1246, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1349, i32 noundef 1, i32 noundef %1145) #12
  %1359 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1347, <4 x i32> noundef %1135, <4 x float> noundef %1358, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1349, i32 noundef 3, i32 noundef %1145) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1360 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1354, i32 %64, i32 2)
  %1361 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1356, i32 %64, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1362 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1346, <4 x i32> noundef %1138, <4 x float> noundef %1250, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1349, i32 noundef 0, i32 noundef %1146) #12
  %1363 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1347, <4 x i32> noundef %1139, <4 x float> noundef %1362, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1349, i32 noundef 2, i32 noundef %1146) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1364 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1354, i32 %68, i32 2)
  %1365 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1356, i32 %68, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1366 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1346, <4 x i32> noundef %1142, <4 x float> noundef %1254, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1349, i32 noundef 1, i32 noundef %1146) #12
  %1367 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1347, <4 x i32> noundef %1143, <4 x float> noundef %1366, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1349, i32 noundef 3, i32 noundef %1146) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1368 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1354, i32 %72, i32 2)
  %1369 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1356, i32 %72, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1370 = or disjoint i32 %209, 2816
  %1371 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %1370, i32 %80, i32 0)
  %1372 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %1370, i32 %85, i32 0)
  %1373 = extractelement <4 x i32> %1350, i64 0
  %1374 = and i32 %1373, 2147450879
  %1375 = extractelement <4 x i32> %1350, i64 1
  %1376 = and i32 %1375, 2147450879
  %1377 = extractelement <4 x i32> %1350, i64 2
  %1378 = and i32 %1377, 2147450879
  %1379 = extractelement <4 x i32> %1350, i64 3
  %1380 = and i32 %1379, 2147450879
  %1381 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1374, i32 %1376) #11, !srcloc !11
  %1382 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1378, i32 %1380) #11, !srcloc !11
  %1383 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1381, i32 %1382) #11, !srcloc !11
  %1384 = trunc i32 %1383 to i16
  %1385 = lshr i32 %1383, 16
  %1386 = trunc nuw i32 %1385 to i16
  %1387 = tail call noundef i16 @llvm.umax.i16(i16 %1384, i16 %1386)
  %1388 = zext i16 %1387 to i32
  %1389 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %1388, i32 177, i32 15, i32 15, i1 true)
  %1390 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %1388, i32 %1389)
  %1391 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1390, i32 78, i32 15, i32 15, i1 true)
  %1392 = tail call noundef i32 @llvm.umax.i32(i32 %1390, i32 %1391)
  %1393 = shl i32 %1392, 16
  %1394 = add i32 %1393, 2097152
  %1395 = lshr i32 %1394, 23
  %1396 = and i32 %1395, 255
  %1397 = tail call i32 @llvm.umax.i32(i32 %1396, i32 2)
  %1398 = add nuw nsw i32 %1397, 254
  %1399 = and i32 %1398, 255
  %1400 = shl nuw nsw i32 %1399, 23
  %1401 = bitcast i32 %1400 to float
  %1402 = bitcast <4 x i32> %1350 to <8 x bfloat>
  %1403 = shufflevector <8 x bfloat> %1402, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1404 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %1403, float %1401, i32 0)
  %1405 = bitcast <4 x i32> %1350 to <8 x bfloat>
  %1406 = shufflevector <8 x bfloat> %1405, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1407 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1404, <2 x bfloat> %1406, float %1401, i32 1)
  %1408 = bitcast <4 x i32> %1350 to <8 x bfloat>
  %1409 = shufflevector <8 x bfloat> %1408, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1410 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1407, <2 x bfloat> %1409, float %1401, i32 2)
  %1411 = bitcast <4 x i32> %1350 to <8 x bfloat>
  %1412 = shufflevector <8 x bfloat> %1411, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1413 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1410, <2 x bfloat> %1412, float %1401, i32 3)
  store i32 %1413, ptr addrspace(3) %391, align 4, !tbaa !7
  %1414 = extractelement <4 x i32> %1351, i64 0
  %1415 = and i32 %1414, 2147450879
  %1416 = extractelement <4 x i32> %1351, i64 1
  %1417 = and i32 %1416, 2147450879
  %1418 = extractelement <4 x i32> %1351, i64 2
  %1419 = and i32 %1418, 2147450879
  %1420 = extractelement <4 x i32> %1351, i64 3
  %1421 = and i32 %1420, 2147450879
  %1422 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1415, i32 %1417) #11, !srcloc !11
  %1423 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1419, i32 %1421) #11, !srcloc !11
  %1424 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1422, i32 %1423) #11, !srcloc !11
  %1425 = trunc i32 %1424 to i16
  %1426 = lshr i32 %1424, 16
  %1427 = trunc nuw i32 %1426 to i16
  %1428 = tail call noundef i16 @llvm.umax.i16(i16 %1425, i16 %1427)
  %1429 = zext i16 %1428 to i32
  %1430 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %1429, i32 177, i32 15, i32 15, i1 true)
  %1431 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %1429, i32 %1430)
  %1432 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1431, i32 78, i32 15, i32 15, i1 true)
  %1433 = tail call noundef i32 @llvm.umax.i32(i32 %1431, i32 %1432)
  %1434 = shl i32 %1433, 16
  %1435 = add i32 %1434, 2097152
  %1436 = lshr i32 %1435, 23
  %1437 = and i32 %1436, 255
  %1438 = tail call i32 @llvm.umax.i32(i32 %1437, i32 2)
  %1439 = add nuw nsw i32 %1438, 254
  %1440 = and i32 %1439, 255
  %1441 = shl nuw nsw i32 %1440, 23
  %1442 = bitcast i32 %1441 to float
  %1443 = bitcast <4 x i32> %1351 to <8 x bfloat>
  %1444 = shufflevector <8 x bfloat> %1443, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1445 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %1444, float %1442, i32 0)
  %1446 = bitcast <4 x i32> %1351 to <8 x bfloat>
  %1447 = shufflevector <8 x bfloat> %1446, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1448 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1445, <2 x bfloat> %1447, float %1442, i32 1)
  %1449 = bitcast <4 x i32> %1351 to <8 x bfloat>
  %1450 = shufflevector <8 x bfloat> %1449, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1451 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1448, <2 x bfloat> %1450, float %1442, i32 2)
  %1452 = bitcast <4 x i32> %1351 to <8 x bfloat>
  %1453 = shufflevector <8 x bfloat> %1452, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1454 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1451, <2 x bfloat> %1453, float %1442, i32 3)
  store i32 %1454, ptr addrspace(3) %433, align 4, !tbaa !7
  %1455 = shl nuw nsw i32 %1440, 16
  %1456 = or disjoint i32 %1455, %1399
  %1457 = add nuw nsw i32 %205, 2816
  %1458 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %1457
  store i32 %1456, ptr addrspace(3) %1458, align 4, !tbaa !7
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1459 = load <4 x i32>, ptr addrspace(3) %438, align 16, !tbaa !12
  %1460 = load <4 x i32>, ptr addrspace(3) %440, align 16, !tbaa !12
  %1461 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %1257
  %1462 = load i32, ptr addrspace(3) %1461, align 4, !tbaa !7
  %1463 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 6144, i32 0)
  %1464 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 6400, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1465 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1459, <4 x i32> noundef %1242, <4 x float> noundef %1353, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1462, i32 noundef 0, i32 noundef %1258) #12
  %1466 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1460, <4 x i32> noundef %1244, <4 x float> noundef %1465, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1462, i32 noundef 2, i32 noundef %1258) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1467 = or disjoint i32 %146, 24576
  %1468 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1467, i32 %60, i32 2)
  %1469 = or disjoint i32 %146, 25600
  %1470 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1469, i32 %60, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1471 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1459, <4 x i32> noundef %1247, <4 x float> noundef %1359, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1462, i32 noundef 1, i32 noundef %1258) #12
  %1472 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1460, <4 x i32> noundef %1248, <4 x float> noundef %1471, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1462, i32 noundef 3, i32 noundef %1258) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1473 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1467, i32 %64, i32 2)
  %1474 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1469, i32 %64, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1475 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1459, <4 x i32> noundef %1251, <4 x float> noundef %1363, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1462, i32 noundef 0, i32 noundef %1259) #12
  %1476 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1460, <4 x i32> noundef %1252, <4 x float> noundef %1475, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1462, i32 noundef 2, i32 noundef %1259) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1477 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1467, i32 %68, i32 2)
  %1478 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1469, i32 %68, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1479 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1459, <4 x i32> noundef %1255, <4 x float> noundef %1367, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1462, i32 noundef 1, i32 noundef %1259) #12
  %1480 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1460, <4 x i32> noundef %1256, <4 x float> noundef %1479, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1462, i32 noundef 3, i32 noundef %1259) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1481 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1467, i32 %72, i32 2)
  %1482 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1469, i32 %72, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1483 = or disjoint i32 %209, 3072
  %1484 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %1483, i32 %80, i32 0)
  %1485 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %1483, i32 %85, i32 0)
  %1486 = extractelement <4 x i32> %1463, i64 0
  %1487 = and i32 %1486, 2147450879
  %1488 = extractelement <4 x i32> %1463, i64 1
  %1489 = and i32 %1488, 2147450879
  %1490 = extractelement <4 x i32> %1463, i64 2
  %1491 = and i32 %1490, 2147450879
  %1492 = extractelement <4 x i32> %1463, i64 3
  %1493 = and i32 %1492, 2147450879
  %1494 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1487, i32 %1489) #11, !srcloc !11
  %1495 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1491, i32 %1493) #11, !srcloc !11
  %1496 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1494, i32 %1495) #11, !srcloc !11
  %1497 = trunc i32 %1496 to i16
  %1498 = lshr i32 %1496, 16
  %1499 = trunc nuw i32 %1498 to i16
  %1500 = tail call noundef i16 @llvm.umax.i16(i16 %1497, i16 %1499)
  %1501 = zext i16 %1500 to i32
  %1502 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %1501, i32 177, i32 15, i32 15, i1 true)
  %1503 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %1501, i32 %1502)
  %1504 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1503, i32 78, i32 15, i32 15, i1 true)
  %1505 = tail call noundef i32 @llvm.umax.i32(i32 %1503, i32 %1504)
  %1506 = shl i32 %1505, 16
  %1507 = add i32 %1506, 2097152
  %1508 = lshr i32 %1507, 23
  %1509 = and i32 %1508, 255
  %1510 = tail call i32 @llvm.umax.i32(i32 %1509, i32 2)
  %1511 = add nuw nsw i32 %1510, 254
  %1512 = and i32 %1511, 255
  %1513 = shl nuw nsw i32 %1512, 23
  %1514 = bitcast i32 %1513 to float
  %1515 = bitcast <4 x i32> %1463 to <8 x bfloat>
  %1516 = shufflevector <8 x bfloat> %1515, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1517 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %1516, float %1514, i32 0)
  %1518 = bitcast <4 x i32> %1463 to <8 x bfloat>
  %1519 = shufflevector <8 x bfloat> %1518, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1520 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1517, <2 x bfloat> %1519, float %1514, i32 1)
  %1521 = bitcast <4 x i32> %1463 to <8 x bfloat>
  %1522 = shufflevector <8 x bfloat> %1521, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1523 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1520, <2 x bfloat> %1522, float %1514, i32 2)
  %1524 = bitcast <4 x i32> %1463 to <8 x bfloat>
  %1525 = shufflevector <8 x bfloat> %1524, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1526 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1523, <2 x bfloat> %1525, float %1514, i32 3)
  store i32 %1526, ptr addrspace(3) %142, align 4, !tbaa !7
  %1527 = extractelement <4 x i32> %1464, i64 0
  %1528 = and i32 %1527, 2147450879
  %1529 = extractelement <4 x i32> %1464, i64 1
  %1530 = and i32 %1529, 2147450879
  %1531 = extractelement <4 x i32> %1464, i64 2
  %1532 = and i32 %1531, 2147450879
  %1533 = extractelement <4 x i32> %1464, i64 3
  %1534 = and i32 %1533, 2147450879
  %1535 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1528, i32 %1530) #11, !srcloc !11
  %1536 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1532, i32 %1534) #11, !srcloc !11
  %1537 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1535, i32 %1536) #11, !srcloc !11
  %1538 = trunc i32 %1537 to i16
  %1539 = lshr i32 %1537, 16
  %1540 = trunc nuw i32 %1539 to i16
  %1541 = tail call noundef i16 @llvm.umax.i16(i16 %1538, i16 %1540)
  %1542 = zext i16 %1541 to i32
  %1543 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %1542, i32 177, i32 15, i32 15, i1 true)
  %1544 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %1542, i32 %1543)
  %1545 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1544, i32 78, i32 15, i32 15, i1 true)
  %1546 = tail call noundef i32 @llvm.umax.i32(i32 %1544, i32 %1545)
  %1547 = shl i32 %1546, 16
  %1548 = add i32 %1547, 2097152
  %1549 = lshr i32 %1548, 23
  %1550 = and i32 %1549, 255
  %1551 = tail call i32 @llvm.umax.i32(i32 %1550, i32 2)
  %1552 = add nuw nsw i32 %1551, 254
  %1553 = and i32 %1552, 255
  %1554 = shl nuw nsw i32 %1553, 23
  %1555 = bitcast i32 %1554 to float
  %1556 = bitcast <4 x i32> %1464 to <8 x bfloat>
  %1557 = shufflevector <8 x bfloat> %1556, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1558 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %1557, float %1555, i32 0)
  %1559 = bitcast <4 x i32> %1464 to <8 x bfloat>
  %1560 = shufflevector <8 x bfloat> %1559, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1561 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1558, <2 x bfloat> %1560, float %1555, i32 1)
  %1562 = bitcast <4 x i32> %1464 to <8 x bfloat>
  %1563 = shufflevector <8 x bfloat> %1562, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1564 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1561, <2 x bfloat> %1563, float %1555, i32 2)
  %1565 = bitcast <4 x i32> %1464 to <8 x bfloat>
  %1566 = shufflevector <8 x bfloat> %1565, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1567 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1564, <2 x bfloat> %1566, float %1555, i32 3)
  store i32 %1567, ptr addrspace(3) %197, align 4, !tbaa !7
  %1568 = shl nuw nsw i32 %1553, 16
  %1569 = or disjoint i32 %1568, %1512
  %1570 = add nuw nsw i32 %205, 3072
  %1571 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %1570
  store i32 %1569, ptr addrspace(3) %1571, align 4, !tbaa !7
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1572 = load <4 x i32>, ptr addrspace(3) %553, align 16, !tbaa !12
  %1573 = load <4 x i32>, ptr addrspace(3) %555, align 16, !tbaa !12
  %1574 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %1370
  %1575 = load i32, ptr addrspace(3) %1574, align 4, !tbaa !7
  %1576 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 6656, i32 0)
  %1577 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 6912, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1578 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1572, <4 x i32> noundef %1355, <4 x float> noundef %1466, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1575, i32 noundef 0, i32 noundef %1371) #12
  %1579 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1573, <4 x i32> noundef %1357, <4 x float> noundef %1578, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1575, i32 noundef 2, i32 noundef %1371) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1580 = or disjoint i32 %146, 26624
  %1581 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1580, i32 %60, i32 2)
  %1582 = or disjoint i32 %146, 27648
  %1583 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1582, i32 %60, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1584 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1572, <4 x i32> noundef %1360, <4 x float> noundef %1472, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1575, i32 noundef 1, i32 noundef %1371) #12
  %1585 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1573, <4 x i32> noundef %1361, <4 x float> noundef %1584, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1575, i32 noundef 3, i32 noundef %1371) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1586 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1580, i32 %64, i32 2)
  %1587 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1582, i32 %64, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1588 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1572, <4 x i32> noundef %1364, <4 x float> noundef %1476, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1575, i32 noundef 0, i32 noundef %1372) #12
  %1589 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1573, <4 x i32> noundef %1365, <4 x float> noundef %1588, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1575, i32 noundef 2, i32 noundef %1372) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1590 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1580, i32 %68, i32 2)
  %1591 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1582, i32 %68, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1592 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1572, <4 x i32> noundef %1368, <4 x float> noundef %1480, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1575, i32 noundef 1, i32 noundef %1372) #12
  %1593 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1573, <4 x i32> noundef %1369, <4 x float> noundef %1592, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1575, i32 noundef 3, i32 noundef %1372) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1594 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1580, i32 %72, i32 2)
  %1595 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1582, i32 %72, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1596 = or disjoint i32 %209, 3328
  %1597 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %1596, i32 %80, i32 0)
  %1598 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %1596, i32 %85, i32 0)
  %1599 = extractelement <4 x i32> %1576, i64 0
  %1600 = and i32 %1599, 2147450879
  %1601 = extractelement <4 x i32> %1576, i64 1
  %1602 = and i32 %1601, 2147450879
  %1603 = extractelement <4 x i32> %1576, i64 2
  %1604 = and i32 %1603, 2147450879
  %1605 = extractelement <4 x i32> %1576, i64 3
  %1606 = and i32 %1605, 2147450879
  %1607 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1600, i32 %1602) #11, !srcloc !11
  %1608 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1604, i32 %1606) #11, !srcloc !11
  %1609 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1607, i32 %1608) #11, !srcloc !11
  %1610 = trunc i32 %1609 to i16
  %1611 = lshr i32 %1609, 16
  %1612 = trunc nuw i32 %1611 to i16
  %1613 = tail call noundef i16 @llvm.umax.i16(i16 %1610, i16 %1612)
  %1614 = zext i16 %1613 to i32
  %1615 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %1614, i32 177, i32 15, i32 15, i1 true)
  %1616 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %1614, i32 %1615)
  %1617 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1616, i32 78, i32 15, i32 15, i1 true)
  %1618 = tail call noundef i32 @llvm.umax.i32(i32 %1616, i32 %1617)
  %1619 = shl i32 %1618, 16
  %1620 = add i32 %1619, 2097152
  %1621 = lshr i32 %1620, 23
  %1622 = and i32 %1621, 255
  %1623 = tail call i32 @llvm.umax.i32(i32 %1622, i32 2)
  %1624 = add nuw nsw i32 %1623, 254
  %1625 = and i32 %1624, 255
  %1626 = shl nuw nsw i32 %1625, 23
  %1627 = bitcast i32 %1626 to float
  %1628 = bitcast <4 x i32> %1576 to <8 x bfloat>
  %1629 = shufflevector <8 x bfloat> %1628, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1630 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %1629, float %1627, i32 0)
  %1631 = bitcast <4 x i32> %1576 to <8 x bfloat>
  %1632 = shufflevector <8 x bfloat> %1631, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1633 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1630, <2 x bfloat> %1632, float %1627, i32 1)
  %1634 = bitcast <4 x i32> %1576 to <8 x bfloat>
  %1635 = shufflevector <8 x bfloat> %1634, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1636 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1633, <2 x bfloat> %1635, float %1627, i32 2)
  %1637 = bitcast <4 x i32> %1576 to <8 x bfloat>
  %1638 = shufflevector <8 x bfloat> %1637, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1639 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1636, <2 x bfloat> %1638, float %1627, i32 3)
  store i32 %1639, ptr addrspace(3) %254, align 4, !tbaa !7
  %1640 = extractelement <4 x i32> %1577, i64 0
  %1641 = and i32 %1640, 2147450879
  %1642 = extractelement <4 x i32> %1577, i64 1
  %1643 = and i32 %1642, 2147450879
  %1644 = extractelement <4 x i32> %1577, i64 2
  %1645 = and i32 %1644, 2147450879
  %1646 = extractelement <4 x i32> %1577, i64 3
  %1647 = and i32 %1646, 2147450879
  %1648 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1641, i32 %1643) #11, !srcloc !11
  %1649 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1645, i32 %1647) #11, !srcloc !11
  %1650 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1648, i32 %1649) #11, !srcloc !11
  %1651 = trunc i32 %1650 to i16
  %1652 = lshr i32 %1650, 16
  %1653 = trunc nuw i32 %1652 to i16
  %1654 = tail call noundef i16 @llvm.umax.i16(i16 %1651, i16 %1653)
  %1655 = zext i16 %1654 to i32
  %1656 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %1655, i32 177, i32 15, i32 15, i1 true)
  %1657 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %1655, i32 %1656)
  %1658 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1657, i32 78, i32 15, i32 15, i1 true)
  %1659 = tail call noundef i32 @llvm.umax.i32(i32 %1657, i32 %1658)
  %1660 = shl i32 %1659, 16
  %1661 = add i32 %1660, 2097152
  %1662 = lshr i32 %1661, 23
  %1663 = and i32 %1662, 255
  %1664 = tail call i32 @llvm.umax.i32(i32 %1663, i32 2)
  %1665 = add nuw nsw i32 %1664, 254
  %1666 = and i32 %1665, 255
  %1667 = shl nuw nsw i32 %1666, 23
  %1668 = bitcast i32 %1667 to float
  %1669 = bitcast <4 x i32> %1577 to <8 x bfloat>
  %1670 = shufflevector <8 x bfloat> %1669, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1671 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %1670, float %1668, i32 0)
  %1672 = bitcast <4 x i32> %1577 to <8 x bfloat>
  %1673 = shufflevector <8 x bfloat> %1672, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1674 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1671, <2 x bfloat> %1673, float %1668, i32 1)
  %1675 = bitcast <4 x i32> %1577 to <8 x bfloat>
  %1676 = shufflevector <8 x bfloat> %1675, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1677 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1674, <2 x bfloat> %1676, float %1668, i32 2)
  %1678 = bitcast <4 x i32> %1577 to <8 x bfloat>
  %1679 = shufflevector <8 x bfloat> %1678, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1680 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1677, <2 x bfloat> %1679, float %1668, i32 3)
  store i32 %1680, ptr addrspace(3) %303, align 4, !tbaa !7
  %1681 = shl nuw nsw i32 %1666, 16
  %1682 = or disjoint i32 %1681, %1625
  %1683 = add nuw nsw i32 %205, 3328
  %1684 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %1683
  store i32 %1682, ptr addrspace(3) %1684, align 4, !tbaa !7
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1685 = load <4 x i32>, ptr addrspace(3) %319, align 16, !tbaa !12
  %1686 = load <4 x i32>, ptr addrspace(3) %323, align 16, !tbaa !12
  %1687 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %1483
  %1688 = load i32, ptr addrspace(3) %1687, align 4, !tbaa !7
  %1689 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 7168, i32 0)
  %1690 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 7424, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1691 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1685, <4 x i32> noundef %1468, <4 x float> noundef %1579, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1688, i32 noundef 0, i32 noundef %1484) #12
  %1692 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1686, <4 x i32> noundef %1470, <4 x float> noundef %1691, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1688, i32 noundef 2, i32 noundef %1484) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1693 = or disjoint i32 %146, 28672
  %1694 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1693, i32 %60, i32 2)
  %1695 = or disjoint i32 %146, 29696
  %1696 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1695, i32 %60, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1697 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1685, <4 x i32> noundef %1473, <4 x float> noundef %1585, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1688, i32 noundef 1, i32 noundef %1484) #12
  %1698 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1686, <4 x i32> noundef %1474, <4 x float> noundef %1697, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1688, i32 noundef 3, i32 noundef %1484) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1699 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1693, i32 %64, i32 2)
  %1700 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1695, i32 %64, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1701 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1685, <4 x i32> noundef %1477, <4 x float> noundef %1589, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1688, i32 noundef 0, i32 noundef %1485) #12
  %1702 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1686, <4 x i32> noundef %1478, <4 x float> noundef %1701, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1688, i32 noundef 2, i32 noundef %1485) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1703 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1693, i32 %68, i32 2)
  %1704 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1695, i32 %68, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1705 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1685, <4 x i32> noundef %1481, <4 x float> noundef %1593, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1688, i32 noundef 1, i32 noundef %1485) #12
  %1706 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1686, <4 x i32> noundef %1482, <4 x float> noundef %1705, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1688, i32 noundef 3, i32 noundef %1485) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1707 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1693, i32 %72, i32 2)
  %1708 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1695, i32 %72, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1709 = or disjoint i32 %209, 3584
  %1710 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %1709, i32 %80, i32 0)
  %1711 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %1709, i32 %85, i32 0)
  %1712 = extractelement <4 x i32> %1689, i64 0
  %1713 = and i32 %1712, 2147450879
  %1714 = extractelement <4 x i32> %1689, i64 1
  %1715 = and i32 %1714, 2147450879
  %1716 = extractelement <4 x i32> %1689, i64 2
  %1717 = and i32 %1716, 2147450879
  %1718 = extractelement <4 x i32> %1689, i64 3
  %1719 = and i32 %1718, 2147450879
  %1720 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1713, i32 %1715) #11, !srcloc !11
  %1721 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1717, i32 %1719) #11, !srcloc !11
  %1722 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1720, i32 %1721) #11, !srcloc !11
  %1723 = trunc i32 %1722 to i16
  %1724 = lshr i32 %1722, 16
  %1725 = trunc nuw i32 %1724 to i16
  %1726 = tail call noundef i16 @llvm.umax.i16(i16 %1723, i16 %1725)
  %1727 = zext i16 %1726 to i32
  %1728 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %1727, i32 177, i32 15, i32 15, i1 true)
  %1729 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %1727, i32 %1728)
  %1730 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1729, i32 78, i32 15, i32 15, i1 true)
  %1731 = tail call noundef i32 @llvm.umax.i32(i32 %1729, i32 %1730)
  %1732 = shl i32 %1731, 16
  %1733 = add i32 %1732, 2097152
  %1734 = lshr i32 %1733, 23
  %1735 = and i32 %1734, 255
  %1736 = tail call i32 @llvm.umax.i32(i32 %1735, i32 2)
  %1737 = add nuw nsw i32 %1736, 254
  %1738 = and i32 %1737, 255
  %1739 = shl nuw nsw i32 %1738, 23
  %1740 = bitcast i32 %1739 to float
  %1741 = bitcast <4 x i32> %1689 to <8 x bfloat>
  %1742 = shufflevector <8 x bfloat> %1741, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1743 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %1742, float %1740, i32 0)
  %1744 = bitcast <4 x i32> %1689 to <8 x bfloat>
  %1745 = shufflevector <8 x bfloat> %1744, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1746 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1743, <2 x bfloat> %1745, float %1740, i32 1)
  %1747 = bitcast <4 x i32> %1689 to <8 x bfloat>
  %1748 = shufflevector <8 x bfloat> %1747, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1749 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1746, <2 x bfloat> %1748, float %1740, i32 2)
  %1750 = bitcast <4 x i32> %1689 to <8 x bfloat>
  %1751 = shufflevector <8 x bfloat> %1750, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1752 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1749, <2 x bfloat> %1751, float %1740, i32 3)
  store i32 %1752, ptr addrspace(3) %391, align 4, !tbaa !7
  %1753 = extractelement <4 x i32> %1690, i64 0
  %1754 = and i32 %1753, 2147450879
  %1755 = extractelement <4 x i32> %1690, i64 1
  %1756 = and i32 %1755, 2147450879
  %1757 = extractelement <4 x i32> %1690, i64 2
  %1758 = and i32 %1757, 2147450879
  %1759 = extractelement <4 x i32> %1690, i64 3
  %1760 = and i32 %1759, 2147450879
  %1761 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1754, i32 %1756) #11, !srcloc !11
  %1762 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1758, i32 %1760) #11, !srcloc !11
  %1763 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1761, i32 %1762) #11, !srcloc !11
  %1764 = trunc i32 %1763 to i16
  %1765 = lshr i32 %1763, 16
  %1766 = trunc nuw i32 %1765 to i16
  %1767 = tail call noundef i16 @llvm.umax.i16(i16 %1764, i16 %1766)
  %1768 = zext i16 %1767 to i32
  %1769 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %1768, i32 177, i32 15, i32 15, i1 true)
  %1770 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %1768, i32 %1769)
  %1771 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1770, i32 78, i32 15, i32 15, i1 true)
  %1772 = tail call noundef i32 @llvm.umax.i32(i32 %1770, i32 %1771)
  %1773 = shl i32 %1772, 16
  %1774 = add i32 %1773, 2097152
  %1775 = lshr i32 %1774, 23
  %1776 = and i32 %1775, 255
  %1777 = tail call i32 @llvm.umax.i32(i32 %1776, i32 2)
  %1778 = add nuw nsw i32 %1777, 254
  %1779 = and i32 %1778, 255
  %1780 = shl nuw nsw i32 %1779, 23
  %1781 = bitcast i32 %1780 to float
  %1782 = bitcast <4 x i32> %1690 to <8 x bfloat>
  %1783 = shufflevector <8 x bfloat> %1782, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1784 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %1783, float %1781, i32 0)
  %1785 = bitcast <4 x i32> %1690 to <8 x bfloat>
  %1786 = shufflevector <8 x bfloat> %1785, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1787 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1784, <2 x bfloat> %1786, float %1781, i32 1)
  %1788 = bitcast <4 x i32> %1690 to <8 x bfloat>
  %1789 = shufflevector <8 x bfloat> %1788, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1790 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1787, <2 x bfloat> %1789, float %1781, i32 2)
  %1791 = bitcast <4 x i32> %1690 to <8 x bfloat>
  %1792 = shufflevector <8 x bfloat> %1791, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1793 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1790, <2 x bfloat> %1792, float %1781, i32 3)
  store i32 %1793, ptr addrspace(3) %433, align 4, !tbaa !7
  %1794 = shl nuw nsw i32 %1779, 16
  %1795 = or disjoint i32 %1794, %1738
  %1796 = add nuw nsw i32 %205, 3584
  %1797 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %1796
  store i32 %1795, ptr addrspace(3) %1797, align 4, !tbaa !7
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1798 = load <4 x i32>, ptr addrspace(3) %438, align 16, !tbaa !12
  %1799 = load <4 x i32>, ptr addrspace(3) %440, align 16, !tbaa !12
  %1800 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %1596
  %1801 = load i32, ptr addrspace(3) %1800, align 4, !tbaa !7
  %1802 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 7680, i32 0)
  %1803 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 7936, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1804 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1798, <4 x i32> noundef %1581, <4 x float> noundef %1692, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1801, i32 noundef 0, i32 noundef %1597) #12
  %1805 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1799, <4 x i32> noundef %1583, <4 x float> noundef %1804, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1801, i32 noundef 2, i32 noundef %1597) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1806 = or disjoint i32 %146, 30720
  %1807 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1806, i32 %60, i32 2)
  %1808 = or disjoint i32 %146, 31744
  %1809 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1808, i32 %60, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1810 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1798, <4 x i32> noundef %1586, <4 x float> noundef %1698, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1801, i32 noundef 1, i32 noundef %1597) #12
  %1811 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1799, <4 x i32> noundef %1587, <4 x float> noundef %1810, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1801, i32 noundef 3, i32 noundef %1597) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1812 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1806, i32 %64, i32 2)
  %1813 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1808, i32 %64, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1814 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1798, <4 x i32> noundef %1590, <4 x float> noundef %1702, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1801, i32 noundef 0, i32 noundef %1598) #12
  %1815 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1799, <4 x i32> noundef %1591, <4 x float> noundef %1814, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1801, i32 noundef 2, i32 noundef %1598) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1816 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1806, i32 %68, i32 2)
  %1817 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1808, i32 %68, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1818 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1798, <4 x i32> noundef %1594, <4 x float> noundef %1706, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1801, i32 noundef 1, i32 noundef %1598) #12
  %1819 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1799, <4 x i32> noundef %1595, <4 x float> noundef %1818, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1801, i32 noundef 3, i32 noundef %1598) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1820 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1806, i32 %72, i32 2)
  %1821 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1808, i32 %72, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1822 = or disjoint i32 %209, 3840
  %1823 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %1822, i32 %80, i32 0)
  %1824 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %1822, i32 %85, i32 0)
  %1825 = extractelement <4 x i32> %1802, i64 0
  %1826 = and i32 %1825, 2147450879
  %1827 = extractelement <4 x i32> %1802, i64 1
  %1828 = and i32 %1827, 2147450879
  %1829 = extractelement <4 x i32> %1802, i64 2
  %1830 = and i32 %1829, 2147450879
  %1831 = extractelement <4 x i32> %1802, i64 3
  %1832 = and i32 %1831, 2147450879
  %1833 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1826, i32 %1828) #11, !srcloc !11
  %1834 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1830, i32 %1832) #11, !srcloc !11
  %1835 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1833, i32 %1834) #11, !srcloc !11
  %1836 = trunc i32 %1835 to i16
  %1837 = lshr i32 %1835, 16
  %1838 = trunc nuw i32 %1837 to i16
  %1839 = tail call noundef i16 @llvm.umax.i16(i16 %1836, i16 %1838)
  %1840 = zext i16 %1839 to i32
  %1841 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %1840, i32 177, i32 15, i32 15, i1 true)
  %1842 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %1840, i32 %1841)
  %1843 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1842, i32 78, i32 15, i32 15, i1 true)
  %1844 = tail call noundef i32 @llvm.umax.i32(i32 %1842, i32 %1843)
  %1845 = shl i32 %1844, 16
  %1846 = add i32 %1845, 2097152
  %1847 = lshr i32 %1846, 23
  %1848 = and i32 %1847, 255
  %1849 = tail call i32 @llvm.umax.i32(i32 %1848, i32 2)
  %1850 = add nuw nsw i32 %1849, 254
  %1851 = and i32 %1850, 255
  %1852 = shl nuw nsw i32 %1851, 23
  %1853 = bitcast i32 %1852 to float
  %1854 = bitcast <4 x i32> %1802 to <8 x bfloat>
  %1855 = shufflevector <8 x bfloat> %1854, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1856 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %1855, float %1853, i32 0)
  %1857 = bitcast <4 x i32> %1802 to <8 x bfloat>
  %1858 = shufflevector <8 x bfloat> %1857, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1859 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1856, <2 x bfloat> %1858, float %1853, i32 1)
  %1860 = bitcast <4 x i32> %1802 to <8 x bfloat>
  %1861 = shufflevector <8 x bfloat> %1860, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1862 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1859, <2 x bfloat> %1861, float %1853, i32 2)
  %1863 = bitcast <4 x i32> %1802 to <8 x bfloat>
  %1864 = shufflevector <8 x bfloat> %1863, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1865 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1862, <2 x bfloat> %1864, float %1853, i32 3)
  store i32 %1865, ptr addrspace(3) %142, align 4, !tbaa !7
  %1866 = extractelement <4 x i32> %1803, i64 0
  %1867 = and i32 %1866, 2147450879
  %1868 = extractelement <4 x i32> %1803, i64 1
  %1869 = and i32 %1868, 2147450879
  %1870 = extractelement <4 x i32> %1803, i64 2
  %1871 = and i32 %1870, 2147450879
  %1872 = extractelement <4 x i32> %1803, i64 3
  %1873 = and i32 %1872, 2147450879
  %1874 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1867, i32 %1869) #11, !srcloc !11
  %1875 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1871, i32 %1873) #11, !srcloc !11
  %1876 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1874, i32 %1875) #11, !srcloc !11
  %1877 = trunc i32 %1876 to i16
  %1878 = lshr i32 %1876, 16
  %1879 = trunc nuw i32 %1878 to i16
  %1880 = tail call noundef i16 @llvm.umax.i16(i16 %1877, i16 %1879)
  %1881 = zext i16 %1880 to i32
  %1882 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %1881, i32 177, i32 15, i32 15, i1 true)
  %1883 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %1881, i32 %1882)
  %1884 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1883, i32 78, i32 15, i32 15, i1 true)
  %1885 = tail call noundef i32 @llvm.umax.i32(i32 %1883, i32 %1884)
  %1886 = shl i32 %1885, 16
  %1887 = add i32 %1886, 2097152
  %1888 = lshr i32 %1887, 23
  %1889 = and i32 %1888, 255
  %1890 = tail call i32 @llvm.umax.i32(i32 %1889, i32 2)
  %1891 = add nuw nsw i32 %1890, 254
  %1892 = and i32 %1891, 255
  %1893 = shl nuw nsw i32 %1892, 23
  %1894 = bitcast i32 %1893 to float
  %1895 = bitcast <4 x i32> %1803 to <8 x bfloat>
  %1896 = shufflevector <8 x bfloat> %1895, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1897 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %1896, float %1894, i32 0)
  %1898 = bitcast <4 x i32> %1803 to <8 x bfloat>
  %1899 = shufflevector <8 x bfloat> %1898, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1900 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1897, <2 x bfloat> %1899, float %1894, i32 1)
  %1901 = bitcast <4 x i32> %1803 to <8 x bfloat>
  %1902 = shufflevector <8 x bfloat> %1901, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1903 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1900, <2 x bfloat> %1902, float %1894, i32 2)
  %1904 = bitcast <4 x i32> %1803 to <8 x bfloat>
  %1905 = shufflevector <8 x bfloat> %1904, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1906 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1903, <2 x bfloat> %1905, float %1894, i32 3)
  store i32 %1906, ptr addrspace(3) %197, align 4, !tbaa !7
  %1907 = shl nuw nsw i32 %1892, 16
  %1908 = or disjoint i32 %1907, %1851
  %1909 = add nuw nsw i32 %205, 3840
  %1910 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %1909
  store i32 %1908, ptr addrspace(3) %1910, align 4, !tbaa !7
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1911 = load <4 x i32>, ptr addrspace(3) %553, align 16, !tbaa !12
  %1912 = load <4 x i32>, ptr addrspace(3) %555, align 16, !tbaa !12
  %1913 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %1709
  %1914 = load i32, ptr addrspace(3) %1913, align 4, !tbaa !7
  %1915 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 8192, i32 0)
  %1916 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 8448, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1917 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1911, <4 x i32> noundef %1694, <4 x float> noundef %1805, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1914, i32 noundef 0, i32 noundef %1710) #12
  %1918 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1912, <4 x i32> noundef %1696, <4 x float> noundef %1917, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1914, i32 noundef 2, i32 noundef %1710) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1919 = or disjoint i32 %146, 32768
  %1920 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1919, i32 %60, i32 2)
  %1921 = or disjoint i32 %146, 33792
  %1922 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1921, i32 %60, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1923 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1911, <4 x i32> noundef %1699, <4 x float> noundef %1811, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1914, i32 noundef 1, i32 noundef %1710) #12
  %1924 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1912, <4 x i32> noundef %1700, <4 x float> noundef %1923, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1914, i32 noundef 3, i32 noundef %1710) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1925 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1919, i32 %64, i32 2)
  %1926 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1921, i32 %64, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1927 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1911, <4 x i32> noundef %1703, <4 x float> noundef %1815, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1914, i32 noundef 0, i32 noundef %1711) #12
  %1928 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1912, <4 x i32> noundef %1704, <4 x float> noundef %1927, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1914, i32 noundef 2, i32 noundef %1711) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1929 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1919, i32 %68, i32 2)
  %1930 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1921, i32 %68, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1931 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1911, <4 x i32> noundef %1707, <4 x float> noundef %1819, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1914, i32 noundef 1, i32 noundef %1711) #12
  %1932 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1912, <4 x i32> noundef %1708, <4 x float> noundef %1931, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1914, i32 noundef 3, i32 noundef %1711) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1933 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1919, i32 %72, i32 2)
  %1934 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %1921, i32 %72, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1935 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %209, i32 %82, i32 0)
  %1936 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %209, i32 %87, i32 0)
  %1937 = extractelement <4 x i32> %1915, i64 0
  %1938 = and i32 %1937, 2147450879
  %1939 = extractelement <4 x i32> %1915, i64 1
  %1940 = and i32 %1939, 2147450879
  %1941 = extractelement <4 x i32> %1915, i64 2
  %1942 = and i32 %1941, 2147450879
  %1943 = extractelement <4 x i32> %1915, i64 3
  %1944 = and i32 %1943, 2147450879
  %1945 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1938, i32 %1940) #11, !srcloc !11
  %1946 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1942, i32 %1944) #11, !srcloc !11
  %1947 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1945, i32 %1946) #11, !srcloc !11
  %1948 = trunc i32 %1947 to i16
  %1949 = lshr i32 %1947, 16
  %1950 = trunc nuw i32 %1949 to i16
  %1951 = tail call noundef i16 @llvm.umax.i16(i16 %1948, i16 %1950)
  %1952 = zext i16 %1951 to i32
  %1953 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %1952, i32 177, i32 15, i32 15, i1 true)
  %1954 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %1952, i32 %1953)
  %1955 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1954, i32 78, i32 15, i32 15, i1 true)
  %1956 = tail call noundef i32 @llvm.umax.i32(i32 %1954, i32 %1955)
  %1957 = shl i32 %1956, 16
  %1958 = add i32 %1957, 2097152
  %1959 = lshr i32 %1958, 23
  %1960 = and i32 %1959, 255
  %1961 = tail call i32 @llvm.umax.i32(i32 %1960, i32 2)
  %1962 = add nuw nsw i32 %1961, 254
  %1963 = and i32 %1962, 255
  %1964 = shl nuw nsw i32 %1963, 23
  %1965 = bitcast i32 %1964 to float
  %1966 = bitcast <4 x i32> %1915 to <8 x bfloat>
  %1967 = shufflevector <8 x bfloat> %1966, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1968 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %1967, float %1965, i32 0)
  %1969 = bitcast <4 x i32> %1915 to <8 x bfloat>
  %1970 = shufflevector <8 x bfloat> %1969, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1971 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1968, <2 x bfloat> %1970, float %1965, i32 1)
  %1972 = bitcast <4 x i32> %1915 to <8 x bfloat>
  %1973 = shufflevector <8 x bfloat> %1972, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1974 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1971, <2 x bfloat> %1973, float %1965, i32 2)
  %1975 = bitcast <4 x i32> %1915 to <8 x bfloat>
  %1976 = shufflevector <8 x bfloat> %1975, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1977 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %1974, <2 x bfloat> %1976, float %1965, i32 3)
  store i32 %1977, ptr addrspace(3) %254, align 4, !tbaa !7
  %1978 = extractelement <4 x i32> %1916, i64 0
  %1979 = and i32 %1978, 2147450879
  %1980 = extractelement <4 x i32> %1916, i64 1
  %1981 = and i32 %1980, 2147450879
  %1982 = extractelement <4 x i32> %1916, i64 2
  %1983 = and i32 %1982, 2147450879
  %1984 = extractelement <4 x i32> %1916, i64 3
  %1985 = and i32 %1984, 2147450879
  %1986 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1979, i32 %1981) #11, !srcloc !11
  %1987 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1983, i32 %1985) #11, !srcloc !11
  %1988 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %1986, i32 %1987) #11, !srcloc !11
  %1989 = trunc i32 %1988 to i16
  %1990 = lshr i32 %1988, 16
  %1991 = trunc nuw i32 %1990 to i16
  %1992 = tail call noundef i16 @llvm.umax.i16(i16 %1989, i16 %1991)
  %1993 = zext i16 %1992 to i32
  %1994 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %1993, i32 177, i32 15, i32 15, i1 true)
  %1995 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %1993, i32 %1994)
  %1996 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1995, i32 78, i32 15, i32 15, i1 true)
  %1997 = tail call noundef i32 @llvm.umax.i32(i32 %1995, i32 %1996)
  %1998 = shl i32 %1997, 16
  %1999 = add i32 %1998, 2097152
  %2000 = lshr i32 %1999, 23
  %2001 = and i32 %2000, 255
  %2002 = tail call i32 @llvm.umax.i32(i32 %2001, i32 2)
  %2003 = add nuw nsw i32 %2002, 254
  %2004 = and i32 %2003, 255
  %2005 = shl nuw nsw i32 %2004, 23
  %2006 = bitcast i32 %2005 to float
  %2007 = bitcast <4 x i32> %1916 to <8 x bfloat>
  %2008 = shufflevector <8 x bfloat> %2007, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %2009 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %2008, float %2006, i32 0)
  %2010 = bitcast <4 x i32> %1916 to <8 x bfloat>
  %2011 = shufflevector <8 x bfloat> %2010, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %2012 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2009, <2 x bfloat> %2011, float %2006, i32 1)
  %2013 = bitcast <4 x i32> %1916 to <8 x bfloat>
  %2014 = shufflevector <8 x bfloat> %2013, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %2015 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2012, <2 x bfloat> %2014, float %2006, i32 2)
  %2016 = bitcast <4 x i32> %1916 to <8 x bfloat>
  %2017 = shufflevector <8 x bfloat> %2016, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %2018 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2015, <2 x bfloat> %2017, float %2006, i32 3)
  store i32 %2018, ptr addrspace(3) %303, align 4, !tbaa !7
  %2019 = shl nuw nsw i32 %2004, 16
  %2020 = or disjoint i32 %2019, %1963
  %2021 = add nuw nsw i32 %205, 4096
  %2022 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %2021
  store i32 %2020, ptr addrspace(3) %2022, align 4, !tbaa !7
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %2023 = load <4 x i32>, ptr addrspace(3) %319, align 16, !tbaa !12
  %2024 = load <4 x i32>, ptr addrspace(3) %323, align 16, !tbaa !12
  %2025 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %1822
  %2026 = load i32, ptr addrspace(3) %2025, align 4, !tbaa !7
  %2027 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 8704, i32 0)
  %2028 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 8960, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2029 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2023, <4 x i32> noundef %1807, <4 x float> noundef %1918, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2026, i32 noundef 0, i32 noundef %1823) #12
  %2030 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2024, <4 x i32> noundef %1809, <4 x float> noundef %2029, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2026, i32 noundef 2, i32 noundef %1823) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2031 = or disjoint i32 %146, 34816
  %2032 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2031, i32 %60, i32 2)
  %2033 = or disjoint i32 %146, 35840
  %2034 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2033, i32 %60, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2035 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2023, <4 x i32> noundef %1812, <4 x float> noundef %1924, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2026, i32 noundef 1, i32 noundef %1823) #12
  %2036 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2024, <4 x i32> noundef %1813, <4 x float> noundef %2035, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2026, i32 noundef 3, i32 noundef %1823) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2037 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2031, i32 %64, i32 2)
  %2038 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2033, i32 %64, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2039 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2023, <4 x i32> noundef %1816, <4 x float> noundef %1928, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2026, i32 noundef 0, i32 noundef %1824) #12
  %2040 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2024, <4 x i32> noundef %1817, <4 x float> noundef %2039, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2026, i32 noundef 2, i32 noundef %1824) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2041 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2031, i32 %68, i32 2)
  %2042 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2033, i32 %68, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2043 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2023, <4 x i32> noundef %1820, <4 x float> noundef %1932, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2026, i32 noundef 1, i32 noundef %1824) #12
  %2044 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2024, <4 x i32> noundef %1821, <4 x float> noundef %2043, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2026, i32 noundef 3, i32 noundef %1824) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2045 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2031, i32 %72, i32 2)
  %2046 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2033, i32 %72, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2047 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %312, i32 %82, i32 0)
  %2048 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %312, i32 %87, i32 0)
  %2049 = extractelement <4 x i32> %2027, i64 0
  %2050 = and i32 %2049, 2147450879
  %2051 = extractelement <4 x i32> %2027, i64 1
  %2052 = and i32 %2051, 2147450879
  %2053 = extractelement <4 x i32> %2027, i64 2
  %2054 = and i32 %2053, 2147450879
  %2055 = extractelement <4 x i32> %2027, i64 3
  %2056 = and i32 %2055, 2147450879
  %2057 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2050, i32 %2052) #11, !srcloc !11
  %2058 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2054, i32 %2056) #11, !srcloc !11
  %2059 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2057, i32 %2058) #11, !srcloc !11
  %2060 = trunc i32 %2059 to i16
  %2061 = lshr i32 %2059, 16
  %2062 = trunc nuw i32 %2061 to i16
  %2063 = tail call noundef i16 @llvm.umax.i16(i16 %2060, i16 %2062)
  %2064 = zext i16 %2063 to i32
  %2065 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %2064, i32 177, i32 15, i32 15, i1 true)
  %2066 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %2064, i32 %2065)
  %2067 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %2066, i32 78, i32 15, i32 15, i1 true)
  %2068 = tail call noundef i32 @llvm.umax.i32(i32 %2066, i32 %2067)
  %2069 = shl i32 %2068, 16
  %2070 = add i32 %2069, 2097152
  %2071 = lshr i32 %2070, 23
  %2072 = and i32 %2071, 255
  %2073 = tail call i32 @llvm.umax.i32(i32 %2072, i32 2)
  %2074 = add nuw nsw i32 %2073, 254
  %2075 = and i32 %2074, 255
  %2076 = shl nuw nsw i32 %2075, 23
  %2077 = bitcast i32 %2076 to float
  %2078 = bitcast <4 x i32> %2027 to <8 x bfloat>
  %2079 = shufflevector <8 x bfloat> %2078, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %2080 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %2079, float %2077, i32 0)
  %2081 = bitcast <4 x i32> %2027 to <8 x bfloat>
  %2082 = shufflevector <8 x bfloat> %2081, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %2083 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2080, <2 x bfloat> %2082, float %2077, i32 1)
  %2084 = bitcast <4 x i32> %2027 to <8 x bfloat>
  %2085 = shufflevector <8 x bfloat> %2084, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %2086 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2083, <2 x bfloat> %2085, float %2077, i32 2)
  %2087 = bitcast <4 x i32> %2027 to <8 x bfloat>
  %2088 = shufflevector <8 x bfloat> %2087, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %2089 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2086, <2 x bfloat> %2088, float %2077, i32 3)
  store i32 %2089, ptr addrspace(3) %391, align 4, !tbaa !7
  %2090 = extractelement <4 x i32> %2028, i64 0
  %2091 = and i32 %2090, 2147450879
  %2092 = extractelement <4 x i32> %2028, i64 1
  %2093 = and i32 %2092, 2147450879
  %2094 = extractelement <4 x i32> %2028, i64 2
  %2095 = and i32 %2094, 2147450879
  %2096 = extractelement <4 x i32> %2028, i64 3
  %2097 = and i32 %2096, 2147450879
  %2098 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2091, i32 %2093) #11, !srcloc !11
  %2099 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2095, i32 %2097) #11, !srcloc !11
  %2100 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2098, i32 %2099) #11, !srcloc !11
  %2101 = trunc i32 %2100 to i16
  %2102 = lshr i32 %2100, 16
  %2103 = trunc nuw i32 %2102 to i16
  %2104 = tail call noundef i16 @llvm.umax.i16(i16 %2101, i16 %2103)
  %2105 = zext i16 %2104 to i32
  %2106 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %2105, i32 177, i32 15, i32 15, i1 true)
  %2107 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %2105, i32 %2106)
  %2108 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %2107, i32 78, i32 15, i32 15, i1 true)
  %2109 = tail call noundef i32 @llvm.umax.i32(i32 %2107, i32 %2108)
  %2110 = shl i32 %2109, 16
  %2111 = add i32 %2110, 2097152
  %2112 = lshr i32 %2111, 23
  %2113 = and i32 %2112, 255
  %2114 = tail call i32 @llvm.umax.i32(i32 %2113, i32 2)
  %2115 = add nuw nsw i32 %2114, 254
  %2116 = and i32 %2115, 255
  %2117 = shl nuw nsw i32 %2116, 23
  %2118 = bitcast i32 %2117 to float
  %2119 = bitcast <4 x i32> %2028 to <8 x bfloat>
  %2120 = shufflevector <8 x bfloat> %2119, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %2121 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %2120, float %2118, i32 0)
  %2122 = bitcast <4 x i32> %2028 to <8 x bfloat>
  %2123 = shufflevector <8 x bfloat> %2122, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %2124 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2121, <2 x bfloat> %2123, float %2118, i32 1)
  %2125 = bitcast <4 x i32> %2028 to <8 x bfloat>
  %2126 = shufflevector <8 x bfloat> %2125, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %2127 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2124, <2 x bfloat> %2126, float %2118, i32 2)
  %2128 = bitcast <4 x i32> %2028 to <8 x bfloat>
  %2129 = shufflevector <8 x bfloat> %2128, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %2130 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2127, <2 x bfloat> %2129, float %2118, i32 3)
  store i32 %2130, ptr addrspace(3) %433, align 4, !tbaa !7
  %2131 = shl nuw nsw i32 %2116, 16
  %2132 = or disjoint i32 %2131, %2075
  %2133 = add nuw nsw i32 %205, 4352
  %2134 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %2133
  store i32 %2132, ptr addrspace(3) %2134, align 4, !tbaa !7
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %2135 = load <4 x i32>, ptr addrspace(3) %438, align 16, !tbaa !12
  %2136 = load <4 x i32>, ptr addrspace(3) %440, align 16, !tbaa !12
  %2137 = or disjoint i32 %209, 4096
  %2138 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %2137
  %2139 = load i32, ptr addrspace(3) %2138, align 4, !tbaa !7
  %2140 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 9216, i32 0)
  %2141 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 9472, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2142 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2135, <4 x i32> noundef %1920, <4 x float> noundef %2030, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2139, i32 noundef 0, i32 noundef %1935) #12
  %2143 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2136, <4 x i32> noundef %1922, <4 x float> noundef %2142, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2139, i32 noundef 2, i32 noundef %1935) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2144 = or disjoint i32 %146, 36864
  %2145 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2144, i32 %60, i32 2)
  %2146 = or disjoint i32 %146, 37888
  %2147 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2146, i32 %60, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2148 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2135, <4 x i32> noundef %1925, <4 x float> noundef %2036, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2139, i32 noundef 1, i32 noundef %1935) #12
  %2149 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2136, <4 x i32> noundef %1926, <4 x float> noundef %2148, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2139, i32 noundef 3, i32 noundef %1935) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2150 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2144, i32 %64, i32 2)
  %2151 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2146, i32 %64, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2152 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2135, <4 x i32> noundef %1929, <4 x float> noundef %2040, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2139, i32 noundef 0, i32 noundef %1936) #12
  %2153 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2136, <4 x i32> noundef %1930, <4 x float> noundef %2152, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2139, i32 noundef 2, i32 noundef %1936) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2154 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2144, i32 %68, i32 2)
  %2155 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2146, i32 %68, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2156 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2135, <4 x i32> noundef %1933, <4 x float> noundef %2044, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2139, i32 noundef 1, i32 noundef %1936) #12
  %2157 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2136, <4 x i32> noundef %1934, <4 x float> noundef %2156, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2139, i32 noundef 3, i32 noundef %1936) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2158 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2144, i32 %72, i32 2)
  %2159 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2146, i32 %72, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2160 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %347, i32 %82, i32 0)
  %2161 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %347, i32 %87, i32 0)
  %2162 = extractelement <4 x i32> %2140, i64 0
  %2163 = and i32 %2162, 2147450879
  %2164 = extractelement <4 x i32> %2140, i64 1
  %2165 = and i32 %2164, 2147450879
  %2166 = extractelement <4 x i32> %2140, i64 2
  %2167 = and i32 %2166, 2147450879
  %2168 = extractelement <4 x i32> %2140, i64 3
  %2169 = and i32 %2168, 2147450879
  %2170 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2163, i32 %2165) #11, !srcloc !11
  %2171 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2167, i32 %2169) #11, !srcloc !11
  %2172 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2170, i32 %2171) #11, !srcloc !11
  %2173 = trunc i32 %2172 to i16
  %2174 = lshr i32 %2172, 16
  %2175 = trunc nuw i32 %2174 to i16
  %2176 = tail call noundef i16 @llvm.umax.i16(i16 %2173, i16 %2175)
  %2177 = zext i16 %2176 to i32
  %2178 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %2177, i32 177, i32 15, i32 15, i1 true)
  %2179 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %2177, i32 %2178)
  %2180 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %2179, i32 78, i32 15, i32 15, i1 true)
  %2181 = tail call noundef i32 @llvm.umax.i32(i32 %2179, i32 %2180)
  %2182 = shl i32 %2181, 16
  %2183 = add i32 %2182, 2097152
  %2184 = lshr i32 %2183, 23
  %2185 = and i32 %2184, 255
  %2186 = tail call i32 @llvm.umax.i32(i32 %2185, i32 2)
  %2187 = add nuw nsw i32 %2186, 254
  %2188 = and i32 %2187, 255
  %2189 = shl nuw nsw i32 %2188, 23
  %2190 = bitcast i32 %2189 to float
  %2191 = bitcast <4 x i32> %2140 to <8 x bfloat>
  %2192 = shufflevector <8 x bfloat> %2191, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %2193 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %2192, float %2190, i32 0)
  %2194 = bitcast <4 x i32> %2140 to <8 x bfloat>
  %2195 = shufflevector <8 x bfloat> %2194, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %2196 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2193, <2 x bfloat> %2195, float %2190, i32 1)
  %2197 = bitcast <4 x i32> %2140 to <8 x bfloat>
  %2198 = shufflevector <8 x bfloat> %2197, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %2199 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2196, <2 x bfloat> %2198, float %2190, i32 2)
  %2200 = bitcast <4 x i32> %2140 to <8 x bfloat>
  %2201 = shufflevector <8 x bfloat> %2200, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %2202 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2199, <2 x bfloat> %2201, float %2190, i32 3)
  store i32 %2202, ptr addrspace(3) %142, align 4, !tbaa !7
  %2203 = extractelement <4 x i32> %2141, i64 0
  %2204 = and i32 %2203, 2147450879
  %2205 = extractelement <4 x i32> %2141, i64 1
  %2206 = and i32 %2205, 2147450879
  %2207 = extractelement <4 x i32> %2141, i64 2
  %2208 = and i32 %2207, 2147450879
  %2209 = extractelement <4 x i32> %2141, i64 3
  %2210 = and i32 %2209, 2147450879
  %2211 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2204, i32 %2206) #11, !srcloc !11
  %2212 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2208, i32 %2210) #11, !srcloc !11
  %2213 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2211, i32 %2212) #11, !srcloc !11
  %2214 = trunc i32 %2213 to i16
  %2215 = lshr i32 %2213, 16
  %2216 = trunc nuw i32 %2215 to i16
  %2217 = tail call noundef i16 @llvm.umax.i16(i16 %2214, i16 %2216)
  %2218 = zext i16 %2217 to i32
  %2219 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %2218, i32 177, i32 15, i32 15, i1 true)
  %2220 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %2218, i32 %2219)
  %2221 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %2220, i32 78, i32 15, i32 15, i1 true)
  %2222 = tail call noundef i32 @llvm.umax.i32(i32 %2220, i32 %2221)
  %2223 = shl i32 %2222, 16
  %2224 = add i32 %2223, 2097152
  %2225 = lshr i32 %2224, 23
  %2226 = and i32 %2225, 255
  %2227 = tail call i32 @llvm.umax.i32(i32 %2226, i32 2)
  %2228 = add nuw nsw i32 %2227, 254
  %2229 = and i32 %2228, 255
  %2230 = shl nuw nsw i32 %2229, 23
  %2231 = bitcast i32 %2230 to float
  %2232 = bitcast <4 x i32> %2141 to <8 x bfloat>
  %2233 = shufflevector <8 x bfloat> %2232, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %2234 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %2233, float %2231, i32 0)
  %2235 = bitcast <4 x i32> %2141 to <8 x bfloat>
  %2236 = shufflevector <8 x bfloat> %2235, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %2237 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2234, <2 x bfloat> %2236, float %2231, i32 1)
  %2238 = bitcast <4 x i32> %2141 to <8 x bfloat>
  %2239 = shufflevector <8 x bfloat> %2238, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %2240 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2237, <2 x bfloat> %2239, float %2231, i32 2)
  %2241 = bitcast <4 x i32> %2141 to <8 x bfloat>
  %2242 = shufflevector <8 x bfloat> %2241, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %2243 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2240, <2 x bfloat> %2242, float %2231, i32 3)
  store i32 %2243, ptr addrspace(3) %197, align 4, !tbaa !7
  %2244 = shl nuw nsw i32 %2229, 16
  %2245 = or disjoint i32 %2244, %2188
  %2246 = add nuw nsw i32 %205, 4608
  %2247 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %2246
  store i32 %2245, ptr addrspace(3) %2247, align 4, !tbaa !7
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %2248 = load <4 x i32>, ptr addrspace(3) %553, align 16, !tbaa !12
  %2249 = load <4 x i32>, ptr addrspace(3) %555, align 16, !tbaa !12
  %2250 = or disjoint i32 %209, 4352
  %2251 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %2250
  %2252 = load i32, ptr addrspace(3) %2251, align 4, !tbaa !7
  %2253 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 9728, i32 0)
  %2254 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 9984, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2255 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2248, <4 x i32> noundef %2032, <4 x float> noundef %2143, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2252, i32 noundef 0, i32 noundef %2047) #12
  %2256 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2249, <4 x i32> noundef %2034, <4 x float> noundef %2255, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2252, i32 noundef 2, i32 noundef %2047) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2257 = or disjoint i32 %146, 38912
  %2258 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2257, i32 %60, i32 2)
  %2259 = or disjoint i32 %146, 39936
  %2260 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2259, i32 %60, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2261 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2248, <4 x i32> noundef %2037, <4 x float> noundef %2149, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2252, i32 noundef 1, i32 noundef %2047) #12
  %2262 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2249, <4 x i32> noundef %2038, <4 x float> noundef %2261, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2252, i32 noundef 3, i32 noundef %2047) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2263 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2257, i32 %64, i32 2)
  %2264 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2259, i32 %64, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2265 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2248, <4 x i32> noundef %2041, <4 x float> noundef %2153, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2252, i32 noundef 0, i32 noundef %2048) #12
  %2266 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2249, <4 x i32> noundef %2042, <4 x float> noundef %2265, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2252, i32 noundef 2, i32 noundef %2048) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2267 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2257, i32 %68, i32 2)
  %2268 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2259, i32 %68, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2269 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2248, <4 x i32> noundef %2045, <4 x float> noundef %2157, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2252, i32 noundef 1, i32 noundef %2048) #12
  %2270 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2249, <4 x i32> noundef %2046, <4 x float> noundef %2269, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2252, i32 noundef 3, i32 noundef %2048) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2271 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2257, i32 %72, i32 2)
  %2272 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2259, i32 %72, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2273 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %464, i32 %82, i32 0)
  %2274 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %464, i32 %87, i32 0)
  %2275 = extractelement <4 x i32> %2253, i64 0
  %2276 = and i32 %2275, 2147450879
  %2277 = extractelement <4 x i32> %2253, i64 1
  %2278 = and i32 %2277, 2147450879
  %2279 = extractelement <4 x i32> %2253, i64 2
  %2280 = and i32 %2279, 2147450879
  %2281 = extractelement <4 x i32> %2253, i64 3
  %2282 = and i32 %2281, 2147450879
  %2283 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2276, i32 %2278) #11, !srcloc !11
  %2284 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2280, i32 %2282) #11, !srcloc !11
  %2285 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2283, i32 %2284) #11, !srcloc !11
  %2286 = trunc i32 %2285 to i16
  %2287 = lshr i32 %2285, 16
  %2288 = trunc nuw i32 %2287 to i16
  %2289 = tail call noundef i16 @llvm.umax.i16(i16 %2286, i16 %2288)
  %2290 = zext i16 %2289 to i32
  %2291 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %2290, i32 177, i32 15, i32 15, i1 true)
  %2292 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %2290, i32 %2291)
  %2293 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %2292, i32 78, i32 15, i32 15, i1 true)
  %2294 = tail call noundef i32 @llvm.umax.i32(i32 %2292, i32 %2293)
  %2295 = shl i32 %2294, 16
  %2296 = add i32 %2295, 2097152
  %2297 = lshr i32 %2296, 23
  %2298 = and i32 %2297, 255
  %2299 = tail call i32 @llvm.umax.i32(i32 %2298, i32 2)
  %2300 = add nuw nsw i32 %2299, 254
  %2301 = and i32 %2300, 255
  %2302 = shl nuw nsw i32 %2301, 23
  %2303 = bitcast i32 %2302 to float
  %2304 = bitcast <4 x i32> %2253 to <8 x bfloat>
  %2305 = shufflevector <8 x bfloat> %2304, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %2306 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %2305, float %2303, i32 0)
  %2307 = bitcast <4 x i32> %2253 to <8 x bfloat>
  %2308 = shufflevector <8 x bfloat> %2307, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %2309 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2306, <2 x bfloat> %2308, float %2303, i32 1)
  %2310 = bitcast <4 x i32> %2253 to <8 x bfloat>
  %2311 = shufflevector <8 x bfloat> %2310, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %2312 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2309, <2 x bfloat> %2311, float %2303, i32 2)
  %2313 = bitcast <4 x i32> %2253 to <8 x bfloat>
  %2314 = shufflevector <8 x bfloat> %2313, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %2315 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2312, <2 x bfloat> %2314, float %2303, i32 3)
  store i32 %2315, ptr addrspace(3) %254, align 4, !tbaa !7
  %2316 = extractelement <4 x i32> %2254, i64 0
  %2317 = and i32 %2316, 2147450879
  %2318 = extractelement <4 x i32> %2254, i64 1
  %2319 = and i32 %2318, 2147450879
  %2320 = extractelement <4 x i32> %2254, i64 2
  %2321 = and i32 %2320, 2147450879
  %2322 = extractelement <4 x i32> %2254, i64 3
  %2323 = and i32 %2322, 2147450879
  %2324 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2317, i32 %2319) #11, !srcloc !11
  %2325 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2321, i32 %2323) #11, !srcloc !11
  %2326 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2324, i32 %2325) #11, !srcloc !11
  %2327 = trunc i32 %2326 to i16
  %2328 = lshr i32 %2326, 16
  %2329 = trunc nuw i32 %2328 to i16
  %2330 = tail call noundef i16 @llvm.umax.i16(i16 %2327, i16 %2329)
  %2331 = zext i16 %2330 to i32
  %2332 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %2331, i32 177, i32 15, i32 15, i1 true)
  %2333 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %2331, i32 %2332)
  %2334 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %2333, i32 78, i32 15, i32 15, i1 true)
  %2335 = tail call noundef i32 @llvm.umax.i32(i32 %2333, i32 %2334)
  %2336 = shl i32 %2335, 16
  %2337 = add i32 %2336, 2097152
  %2338 = lshr i32 %2337, 23
  %2339 = and i32 %2338, 255
  %2340 = tail call i32 @llvm.umax.i32(i32 %2339, i32 2)
  %2341 = add nuw nsw i32 %2340, 254
  %2342 = and i32 %2341, 255
  %2343 = shl nuw nsw i32 %2342, 23
  %2344 = bitcast i32 %2343 to float
  %2345 = bitcast <4 x i32> %2254 to <8 x bfloat>
  %2346 = shufflevector <8 x bfloat> %2345, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %2347 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %2346, float %2344, i32 0)
  %2348 = bitcast <4 x i32> %2254 to <8 x bfloat>
  %2349 = shufflevector <8 x bfloat> %2348, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %2350 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2347, <2 x bfloat> %2349, float %2344, i32 1)
  %2351 = bitcast <4 x i32> %2254 to <8 x bfloat>
  %2352 = shufflevector <8 x bfloat> %2351, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %2353 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2350, <2 x bfloat> %2352, float %2344, i32 2)
  %2354 = bitcast <4 x i32> %2254 to <8 x bfloat>
  %2355 = shufflevector <8 x bfloat> %2354, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %2356 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2353, <2 x bfloat> %2355, float %2344, i32 3)
  store i32 %2356, ptr addrspace(3) %303, align 4, !tbaa !7
  %2357 = shl nuw nsw i32 %2342, 16
  %2358 = or disjoint i32 %2357, %2301
  %2359 = add nuw nsw i32 %205, 4864
  %2360 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %2359
  store i32 %2358, ptr addrspace(3) %2360, align 4, !tbaa !7
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %2361 = load <4 x i32>, ptr addrspace(3) %319, align 16, !tbaa !12
  %2362 = load <4 x i32>, ptr addrspace(3) %323, align 16, !tbaa !12
  %2363 = or disjoint i32 %209, 4608
  %2364 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %2363
  %2365 = load i32, ptr addrspace(3) %2364, align 4, !tbaa !7
  %2366 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 10240, i32 0)
  %2367 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 10496, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2368 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2361, <4 x i32> noundef %2145, <4 x float> noundef %2256, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2365, i32 noundef 0, i32 noundef %2160) #12
  %2369 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2362, <4 x i32> noundef %2147, <4 x float> noundef %2368, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2365, i32 noundef 2, i32 noundef %2160) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2370 = or disjoint i32 %146, 40960
  %2371 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2370, i32 %60, i32 2)
  %2372 = or disjoint i32 %146, 41984
  %2373 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2372, i32 %60, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2374 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2361, <4 x i32> noundef %2150, <4 x float> noundef %2262, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2365, i32 noundef 1, i32 noundef %2160) #12
  %2375 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2362, <4 x i32> noundef %2151, <4 x float> noundef %2374, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2365, i32 noundef 3, i32 noundef %2160) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2376 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2370, i32 %64, i32 2)
  %2377 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2372, i32 %64, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2378 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2361, <4 x i32> noundef %2154, <4 x float> noundef %2266, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2365, i32 noundef 0, i32 noundef %2161) #12
  %2379 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2362, <4 x i32> noundef %2155, <4 x float> noundef %2378, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2365, i32 noundef 2, i32 noundef %2161) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2380 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2370, i32 %68, i32 2)
  %2381 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2372, i32 %68, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2382 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2361, <4 x i32> noundef %2158, <4 x float> noundef %2270, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2365, i32 noundef 1, i32 noundef %2161) #12
  %2383 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2362, <4 x i32> noundef %2159, <4 x float> noundef %2382, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2365, i32 noundef 3, i32 noundef %2161) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2384 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2370, i32 %72, i32 2)
  %2385 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2372, i32 %72, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2386 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %579, i32 %82, i32 0)
  %2387 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %579, i32 %87, i32 0)
  %2388 = extractelement <4 x i32> %2366, i64 0
  %2389 = and i32 %2388, 2147450879
  %2390 = extractelement <4 x i32> %2366, i64 1
  %2391 = and i32 %2390, 2147450879
  %2392 = extractelement <4 x i32> %2366, i64 2
  %2393 = and i32 %2392, 2147450879
  %2394 = extractelement <4 x i32> %2366, i64 3
  %2395 = and i32 %2394, 2147450879
  %2396 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2389, i32 %2391) #11, !srcloc !11
  %2397 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2393, i32 %2395) #11, !srcloc !11
  %2398 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2396, i32 %2397) #11, !srcloc !11
  %2399 = trunc i32 %2398 to i16
  %2400 = lshr i32 %2398, 16
  %2401 = trunc nuw i32 %2400 to i16
  %2402 = tail call noundef i16 @llvm.umax.i16(i16 %2399, i16 %2401)
  %2403 = zext i16 %2402 to i32
  %2404 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %2403, i32 177, i32 15, i32 15, i1 true)
  %2405 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %2403, i32 %2404)
  %2406 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %2405, i32 78, i32 15, i32 15, i1 true)
  %2407 = tail call noundef i32 @llvm.umax.i32(i32 %2405, i32 %2406)
  %2408 = shl i32 %2407, 16
  %2409 = add i32 %2408, 2097152
  %2410 = lshr i32 %2409, 23
  %2411 = and i32 %2410, 255
  %2412 = tail call i32 @llvm.umax.i32(i32 %2411, i32 2)
  %2413 = add nuw nsw i32 %2412, 254
  %2414 = and i32 %2413, 255
  %2415 = shl nuw nsw i32 %2414, 23
  %2416 = bitcast i32 %2415 to float
  %2417 = bitcast <4 x i32> %2366 to <8 x bfloat>
  %2418 = shufflevector <8 x bfloat> %2417, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %2419 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %2418, float %2416, i32 0)
  %2420 = bitcast <4 x i32> %2366 to <8 x bfloat>
  %2421 = shufflevector <8 x bfloat> %2420, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %2422 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2419, <2 x bfloat> %2421, float %2416, i32 1)
  %2423 = bitcast <4 x i32> %2366 to <8 x bfloat>
  %2424 = shufflevector <8 x bfloat> %2423, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %2425 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2422, <2 x bfloat> %2424, float %2416, i32 2)
  %2426 = bitcast <4 x i32> %2366 to <8 x bfloat>
  %2427 = shufflevector <8 x bfloat> %2426, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %2428 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2425, <2 x bfloat> %2427, float %2416, i32 3)
  store i32 %2428, ptr addrspace(3) %391, align 4, !tbaa !7
  %2429 = extractelement <4 x i32> %2367, i64 0
  %2430 = and i32 %2429, 2147450879
  %2431 = extractelement <4 x i32> %2367, i64 1
  %2432 = and i32 %2431, 2147450879
  %2433 = extractelement <4 x i32> %2367, i64 2
  %2434 = and i32 %2433, 2147450879
  %2435 = extractelement <4 x i32> %2367, i64 3
  %2436 = and i32 %2435, 2147450879
  %2437 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2430, i32 %2432) #11, !srcloc !11
  %2438 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2434, i32 %2436) #11, !srcloc !11
  %2439 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2437, i32 %2438) #11, !srcloc !11
  %2440 = trunc i32 %2439 to i16
  %2441 = lshr i32 %2439, 16
  %2442 = trunc nuw i32 %2441 to i16
  %2443 = tail call noundef i16 @llvm.umax.i16(i16 %2440, i16 %2442)
  %2444 = zext i16 %2443 to i32
  %2445 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %2444, i32 177, i32 15, i32 15, i1 true)
  %2446 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %2444, i32 %2445)
  %2447 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %2446, i32 78, i32 15, i32 15, i1 true)
  %2448 = tail call noundef i32 @llvm.umax.i32(i32 %2446, i32 %2447)
  %2449 = shl i32 %2448, 16
  %2450 = add i32 %2449, 2097152
  %2451 = lshr i32 %2450, 23
  %2452 = and i32 %2451, 255
  %2453 = tail call i32 @llvm.umax.i32(i32 %2452, i32 2)
  %2454 = add nuw nsw i32 %2453, 254
  %2455 = and i32 %2454, 255
  %2456 = shl nuw nsw i32 %2455, 23
  %2457 = bitcast i32 %2456 to float
  %2458 = bitcast <4 x i32> %2367 to <8 x bfloat>
  %2459 = shufflevector <8 x bfloat> %2458, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %2460 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %2459, float %2457, i32 0)
  %2461 = bitcast <4 x i32> %2367 to <8 x bfloat>
  %2462 = shufflevector <8 x bfloat> %2461, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %2463 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2460, <2 x bfloat> %2462, float %2457, i32 1)
  %2464 = bitcast <4 x i32> %2367 to <8 x bfloat>
  %2465 = shufflevector <8 x bfloat> %2464, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %2466 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2463, <2 x bfloat> %2465, float %2457, i32 2)
  %2467 = bitcast <4 x i32> %2367 to <8 x bfloat>
  %2468 = shufflevector <8 x bfloat> %2467, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %2469 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2466, <2 x bfloat> %2468, float %2457, i32 3)
  store i32 %2469, ptr addrspace(3) %433, align 4, !tbaa !7
  %2470 = shl nuw nsw i32 %2455, 16
  %2471 = or disjoint i32 %2470, %2414
  %2472 = add nuw nsw i32 %205, 5120
  %2473 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %2472
  store i32 %2471, ptr addrspace(3) %2473, align 4, !tbaa !7
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %2474 = load <4 x i32>, ptr addrspace(3) %438, align 16, !tbaa !12
  %2475 = load <4 x i32>, ptr addrspace(3) %440, align 16, !tbaa !12
  %2476 = or disjoint i32 %209, 4864
  %2477 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %2476
  %2478 = load i32, ptr addrspace(3) %2477, align 4, !tbaa !7
  %2479 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 10752, i32 0)
  %2480 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 11008, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2481 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2474, <4 x i32> noundef %2258, <4 x float> noundef %2369, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2478, i32 noundef 0, i32 noundef %2273) #12
  %2482 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2475, <4 x i32> noundef %2260, <4 x float> noundef %2481, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2478, i32 noundef 2, i32 noundef %2273) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2483 = or disjoint i32 %146, 43008
  %2484 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2483, i32 %60, i32 2)
  %2485 = or disjoint i32 %146, 44032
  %2486 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2485, i32 %60, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2487 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2474, <4 x i32> noundef %2263, <4 x float> noundef %2375, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2478, i32 noundef 1, i32 noundef %2273) #12
  %2488 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2475, <4 x i32> noundef %2264, <4 x float> noundef %2487, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2478, i32 noundef 3, i32 noundef %2273) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2489 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2483, i32 %64, i32 2)
  %2490 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2485, i32 %64, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2491 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2474, <4 x i32> noundef %2267, <4 x float> noundef %2379, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2478, i32 noundef 0, i32 noundef %2274) #12
  %2492 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2475, <4 x i32> noundef %2268, <4 x float> noundef %2491, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2478, i32 noundef 2, i32 noundef %2274) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2493 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2483, i32 %68, i32 2)
  %2494 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2485, i32 %68, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2495 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2474, <4 x i32> noundef %2271, <4 x float> noundef %2383, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2478, i32 noundef 1, i32 noundef %2274) #12
  %2496 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2475, <4 x i32> noundef %2272, <4 x float> noundef %2495, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2478, i32 noundef 3, i32 noundef %2274) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2497 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2483, i32 %72, i32 2)
  %2498 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2485, i32 %72, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2499 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %692, i32 %82, i32 0)
  %2500 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %692, i32 %87, i32 0)
  %2501 = extractelement <4 x i32> %2479, i64 0
  %2502 = and i32 %2501, 2147450879
  %2503 = extractelement <4 x i32> %2479, i64 1
  %2504 = and i32 %2503, 2147450879
  %2505 = extractelement <4 x i32> %2479, i64 2
  %2506 = and i32 %2505, 2147450879
  %2507 = extractelement <4 x i32> %2479, i64 3
  %2508 = and i32 %2507, 2147450879
  %2509 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2502, i32 %2504) #11, !srcloc !11
  %2510 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2506, i32 %2508) #11, !srcloc !11
  %2511 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2509, i32 %2510) #11, !srcloc !11
  %2512 = trunc i32 %2511 to i16
  %2513 = lshr i32 %2511, 16
  %2514 = trunc nuw i32 %2513 to i16
  %2515 = tail call noundef i16 @llvm.umax.i16(i16 %2512, i16 %2514)
  %2516 = zext i16 %2515 to i32
  %2517 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %2516, i32 177, i32 15, i32 15, i1 true)
  %2518 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %2516, i32 %2517)
  %2519 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %2518, i32 78, i32 15, i32 15, i1 true)
  %2520 = tail call noundef i32 @llvm.umax.i32(i32 %2518, i32 %2519)
  %2521 = shl i32 %2520, 16
  %2522 = add i32 %2521, 2097152
  %2523 = lshr i32 %2522, 23
  %2524 = and i32 %2523, 255
  %2525 = tail call i32 @llvm.umax.i32(i32 %2524, i32 2)
  %2526 = add nuw nsw i32 %2525, 254
  %2527 = and i32 %2526, 255
  %2528 = shl nuw nsw i32 %2527, 23
  %2529 = bitcast i32 %2528 to float
  %2530 = bitcast <4 x i32> %2479 to <8 x bfloat>
  %2531 = shufflevector <8 x bfloat> %2530, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %2532 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %2531, float %2529, i32 0)
  %2533 = bitcast <4 x i32> %2479 to <8 x bfloat>
  %2534 = shufflevector <8 x bfloat> %2533, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %2535 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2532, <2 x bfloat> %2534, float %2529, i32 1)
  %2536 = bitcast <4 x i32> %2479 to <8 x bfloat>
  %2537 = shufflevector <8 x bfloat> %2536, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %2538 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2535, <2 x bfloat> %2537, float %2529, i32 2)
  %2539 = bitcast <4 x i32> %2479 to <8 x bfloat>
  %2540 = shufflevector <8 x bfloat> %2539, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %2541 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2538, <2 x bfloat> %2540, float %2529, i32 3)
  store i32 %2541, ptr addrspace(3) %142, align 4, !tbaa !7
  %2542 = extractelement <4 x i32> %2480, i64 0
  %2543 = and i32 %2542, 2147450879
  %2544 = extractelement <4 x i32> %2480, i64 1
  %2545 = and i32 %2544, 2147450879
  %2546 = extractelement <4 x i32> %2480, i64 2
  %2547 = and i32 %2546, 2147450879
  %2548 = extractelement <4 x i32> %2480, i64 3
  %2549 = and i32 %2548, 2147450879
  %2550 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2543, i32 %2545) #11, !srcloc !11
  %2551 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2547, i32 %2549) #11, !srcloc !11
  %2552 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2550, i32 %2551) #11, !srcloc !11
  %2553 = trunc i32 %2552 to i16
  %2554 = lshr i32 %2552, 16
  %2555 = trunc nuw i32 %2554 to i16
  %2556 = tail call noundef i16 @llvm.umax.i16(i16 %2553, i16 %2555)
  %2557 = zext i16 %2556 to i32
  %2558 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %2557, i32 177, i32 15, i32 15, i1 true)
  %2559 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %2557, i32 %2558)
  %2560 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %2559, i32 78, i32 15, i32 15, i1 true)
  %2561 = tail call noundef i32 @llvm.umax.i32(i32 %2559, i32 %2560)
  %2562 = shl i32 %2561, 16
  %2563 = add i32 %2562, 2097152
  %2564 = lshr i32 %2563, 23
  %2565 = and i32 %2564, 255
  %2566 = tail call i32 @llvm.umax.i32(i32 %2565, i32 2)
  %2567 = add nuw nsw i32 %2566, 254
  %2568 = and i32 %2567, 255
  %2569 = shl nuw nsw i32 %2568, 23
  %2570 = bitcast i32 %2569 to float
  %2571 = bitcast <4 x i32> %2480 to <8 x bfloat>
  %2572 = shufflevector <8 x bfloat> %2571, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %2573 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %2572, float %2570, i32 0)
  %2574 = bitcast <4 x i32> %2480 to <8 x bfloat>
  %2575 = shufflevector <8 x bfloat> %2574, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %2576 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2573, <2 x bfloat> %2575, float %2570, i32 1)
  %2577 = bitcast <4 x i32> %2480 to <8 x bfloat>
  %2578 = shufflevector <8 x bfloat> %2577, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %2579 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2576, <2 x bfloat> %2578, float %2570, i32 2)
  %2580 = bitcast <4 x i32> %2480 to <8 x bfloat>
  %2581 = shufflevector <8 x bfloat> %2580, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %2582 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2579, <2 x bfloat> %2581, float %2570, i32 3)
  store i32 %2582, ptr addrspace(3) %197, align 4, !tbaa !7
  %2583 = shl nuw nsw i32 %2568, 16
  %2584 = or disjoint i32 %2583, %2527
  %2585 = add nuw nsw i32 %205, 5376
  %2586 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %2585
  store i32 %2584, ptr addrspace(3) %2586, align 4, !tbaa !7
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %2587 = load <4 x i32>, ptr addrspace(3) %553, align 16, !tbaa !12
  %2588 = load <4 x i32>, ptr addrspace(3) %555, align 16, !tbaa !12
  %2589 = or disjoint i32 %209, 5120
  %2590 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %2589
  %2591 = load i32, ptr addrspace(3) %2590, align 4, !tbaa !7
  %2592 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 11264, i32 0)
  %2593 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 11520, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2594 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2587, <4 x i32> noundef %2371, <4 x float> noundef %2482, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2591, i32 noundef 0, i32 noundef %2386) #12
  %2595 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2588, <4 x i32> noundef %2373, <4 x float> noundef %2594, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2591, i32 noundef 2, i32 noundef %2386) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2596 = or disjoint i32 %146, 45056
  %2597 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2596, i32 %60, i32 2)
  %2598 = or disjoint i32 %146, 46080
  %2599 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2598, i32 %60, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2600 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2587, <4 x i32> noundef %2376, <4 x float> noundef %2488, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2591, i32 noundef 1, i32 noundef %2386) #12
  %2601 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2588, <4 x i32> noundef %2377, <4 x float> noundef %2600, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2591, i32 noundef 3, i32 noundef %2386) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2602 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2596, i32 %64, i32 2)
  %2603 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2598, i32 %64, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2604 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2587, <4 x i32> noundef %2380, <4 x float> noundef %2492, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2591, i32 noundef 0, i32 noundef %2387) #12
  %2605 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2588, <4 x i32> noundef %2381, <4 x float> noundef %2604, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2591, i32 noundef 2, i32 noundef %2387) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2606 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2596, i32 %68, i32 2)
  %2607 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2598, i32 %68, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2608 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2587, <4 x i32> noundef %2384, <4 x float> noundef %2496, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2591, i32 noundef 1, i32 noundef %2387) #12
  %2609 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2588, <4 x i32> noundef %2385, <4 x float> noundef %2608, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2591, i32 noundef 3, i32 noundef %2387) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2610 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2596, i32 %72, i32 2)
  %2611 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2598, i32 %72, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2612 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %805, i32 %82, i32 0)
  %2613 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %805, i32 %87, i32 0)
  %2614 = extractelement <4 x i32> %2592, i64 0
  %2615 = and i32 %2614, 2147450879
  %2616 = extractelement <4 x i32> %2592, i64 1
  %2617 = and i32 %2616, 2147450879
  %2618 = extractelement <4 x i32> %2592, i64 2
  %2619 = and i32 %2618, 2147450879
  %2620 = extractelement <4 x i32> %2592, i64 3
  %2621 = and i32 %2620, 2147450879
  %2622 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2615, i32 %2617) #11, !srcloc !11
  %2623 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2619, i32 %2621) #11, !srcloc !11
  %2624 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2622, i32 %2623) #11, !srcloc !11
  %2625 = trunc i32 %2624 to i16
  %2626 = lshr i32 %2624, 16
  %2627 = trunc nuw i32 %2626 to i16
  %2628 = tail call noundef i16 @llvm.umax.i16(i16 %2625, i16 %2627)
  %2629 = zext i16 %2628 to i32
  %2630 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %2629, i32 177, i32 15, i32 15, i1 true)
  %2631 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %2629, i32 %2630)
  %2632 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %2631, i32 78, i32 15, i32 15, i1 true)
  %2633 = tail call noundef i32 @llvm.umax.i32(i32 %2631, i32 %2632)
  %2634 = shl i32 %2633, 16
  %2635 = add i32 %2634, 2097152
  %2636 = lshr i32 %2635, 23
  %2637 = and i32 %2636, 255
  %2638 = tail call i32 @llvm.umax.i32(i32 %2637, i32 2)
  %2639 = add nuw nsw i32 %2638, 254
  %2640 = and i32 %2639, 255
  %2641 = shl nuw nsw i32 %2640, 23
  %2642 = bitcast i32 %2641 to float
  %2643 = bitcast <4 x i32> %2592 to <8 x bfloat>
  %2644 = shufflevector <8 x bfloat> %2643, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %2645 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %2644, float %2642, i32 0)
  %2646 = bitcast <4 x i32> %2592 to <8 x bfloat>
  %2647 = shufflevector <8 x bfloat> %2646, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %2648 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2645, <2 x bfloat> %2647, float %2642, i32 1)
  %2649 = bitcast <4 x i32> %2592 to <8 x bfloat>
  %2650 = shufflevector <8 x bfloat> %2649, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %2651 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2648, <2 x bfloat> %2650, float %2642, i32 2)
  %2652 = bitcast <4 x i32> %2592 to <8 x bfloat>
  %2653 = shufflevector <8 x bfloat> %2652, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %2654 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2651, <2 x bfloat> %2653, float %2642, i32 3)
  store i32 %2654, ptr addrspace(3) %254, align 4, !tbaa !7
  %2655 = extractelement <4 x i32> %2593, i64 0
  %2656 = and i32 %2655, 2147450879
  %2657 = extractelement <4 x i32> %2593, i64 1
  %2658 = and i32 %2657, 2147450879
  %2659 = extractelement <4 x i32> %2593, i64 2
  %2660 = and i32 %2659, 2147450879
  %2661 = extractelement <4 x i32> %2593, i64 3
  %2662 = and i32 %2661, 2147450879
  %2663 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2656, i32 %2658) #11, !srcloc !11
  %2664 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2660, i32 %2662) #11, !srcloc !11
  %2665 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2663, i32 %2664) #11, !srcloc !11
  %2666 = trunc i32 %2665 to i16
  %2667 = lshr i32 %2665, 16
  %2668 = trunc nuw i32 %2667 to i16
  %2669 = tail call noundef i16 @llvm.umax.i16(i16 %2666, i16 %2668)
  %2670 = zext i16 %2669 to i32
  %2671 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %2670, i32 177, i32 15, i32 15, i1 true)
  %2672 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %2670, i32 %2671)
  %2673 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %2672, i32 78, i32 15, i32 15, i1 true)
  %2674 = tail call noundef i32 @llvm.umax.i32(i32 %2672, i32 %2673)
  %2675 = shl i32 %2674, 16
  %2676 = add i32 %2675, 2097152
  %2677 = lshr i32 %2676, 23
  %2678 = and i32 %2677, 255
  %2679 = tail call i32 @llvm.umax.i32(i32 %2678, i32 2)
  %2680 = add nuw nsw i32 %2679, 254
  %2681 = and i32 %2680, 255
  %2682 = shl nuw nsw i32 %2681, 23
  %2683 = bitcast i32 %2682 to float
  %2684 = bitcast <4 x i32> %2593 to <8 x bfloat>
  %2685 = shufflevector <8 x bfloat> %2684, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %2686 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %2685, float %2683, i32 0)
  %2687 = bitcast <4 x i32> %2593 to <8 x bfloat>
  %2688 = shufflevector <8 x bfloat> %2687, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %2689 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2686, <2 x bfloat> %2688, float %2683, i32 1)
  %2690 = bitcast <4 x i32> %2593 to <8 x bfloat>
  %2691 = shufflevector <8 x bfloat> %2690, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %2692 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2689, <2 x bfloat> %2691, float %2683, i32 2)
  %2693 = bitcast <4 x i32> %2593 to <8 x bfloat>
  %2694 = shufflevector <8 x bfloat> %2693, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %2695 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2692, <2 x bfloat> %2694, float %2683, i32 3)
  store i32 %2695, ptr addrspace(3) %303, align 4, !tbaa !7
  %2696 = shl nuw nsw i32 %2681, 16
  %2697 = or disjoint i32 %2696, %2640
  %2698 = add nuw nsw i32 %205, 5632
  %2699 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %2698
  store i32 %2697, ptr addrspace(3) %2699, align 4, !tbaa !7
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %2700 = load <4 x i32>, ptr addrspace(3) %319, align 16, !tbaa !12
  %2701 = load <4 x i32>, ptr addrspace(3) %323, align 16, !tbaa !12
  %2702 = or disjoint i32 %209, 5376
  %2703 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %2702
  %2704 = load i32, ptr addrspace(3) %2703, align 4, !tbaa !7
  %2705 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 11776, i32 0)
  %2706 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 12032, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2707 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2700, <4 x i32> noundef %2484, <4 x float> noundef %2595, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2704, i32 noundef 0, i32 noundef %2499) #12
  %2708 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2701, <4 x i32> noundef %2486, <4 x float> noundef %2707, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2704, i32 noundef 2, i32 noundef %2499) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2709 = or disjoint i32 %146, 47104
  %2710 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2709, i32 %60, i32 2)
  %2711 = or disjoint i32 %146, 48128
  %2712 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2711, i32 %60, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2713 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2700, <4 x i32> noundef %2489, <4 x float> noundef %2601, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2704, i32 noundef 1, i32 noundef %2499) #12
  %2714 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2701, <4 x i32> noundef %2490, <4 x float> noundef %2713, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2704, i32 noundef 3, i32 noundef %2499) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2715 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2709, i32 %64, i32 2)
  %2716 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2711, i32 %64, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2717 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2700, <4 x i32> noundef %2493, <4 x float> noundef %2605, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2704, i32 noundef 0, i32 noundef %2500) #12
  %2718 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2701, <4 x i32> noundef %2494, <4 x float> noundef %2717, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2704, i32 noundef 2, i32 noundef %2500) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2719 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2709, i32 %68, i32 2)
  %2720 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2711, i32 %68, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2721 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2700, <4 x i32> noundef %2497, <4 x float> noundef %2609, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2704, i32 noundef 1, i32 noundef %2500) #12
  %2722 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2701, <4 x i32> noundef %2498, <4 x float> noundef %2721, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2704, i32 noundef 3, i32 noundef %2500) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2723 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2709, i32 %72, i32 2)
  %2724 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2711, i32 %72, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2725 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %918, i32 %82, i32 0)
  %2726 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %918, i32 %87, i32 0)
  %2727 = extractelement <4 x i32> %2705, i64 0
  %2728 = and i32 %2727, 2147450879
  %2729 = extractelement <4 x i32> %2705, i64 1
  %2730 = and i32 %2729, 2147450879
  %2731 = extractelement <4 x i32> %2705, i64 2
  %2732 = and i32 %2731, 2147450879
  %2733 = extractelement <4 x i32> %2705, i64 3
  %2734 = and i32 %2733, 2147450879
  %2735 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2728, i32 %2730) #11, !srcloc !11
  %2736 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2732, i32 %2734) #11, !srcloc !11
  %2737 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2735, i32 %2736) #11, !srcloc !11
  %2738 = trunc i32 %2737 to i16
  %2739 = lshr i32 %2737, 16
  %2740 = trunc nuw i32 %2739 to i16
  %2741 = tail call noundef i16 @llvm.umax.i16(i16 %2738, i16 %2740)
  %2742 = zext i16 %2741 to i32
  %2743 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %2742, i32 177, i32 15, i32 15, i1 true)
  %2744 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %2742, i32 %2743)
  %2745 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %2744, i32 78, i32 15, i32 15, i1 true)
  %2746 = tail call noundef i32 @llvm.umax.i32(i32 %2744, i32 %2745)
  %2747 = shl i32 %2746, 16
  %2748 = add i32 %2747, 2097152
  %2749 = lshr i32 %2748, 23
  %2750 = and i32 %2749, 255
  %2751 = tail call i32 @llvm.umax.i32(i32 %2750, i32 2)
  %2752 = add nuw nsw i32 %2751, 254
  %2753 = and i32 %2752, 255
  %2754 = shl nuw nsw i32 %2753, 23
  %2755 = bitcast i32 %2754 to float
  %2756 = bitcast <4 x i32> %2705 to <8 x bfloat>
  %2757 = shufflevector <8 x bfloat> %2756, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %2758 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %2757, float %2755, i32 0)
  %2759 = bitcast <4 x i32> %2705 to <8 x bfloat>
  %2760 = shufflevector <8 x bfloat> %2759, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %2761 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2758, <2 x bfloat> %2760, float %2755, i32 1)
  %2762 = bitcast <4 x i32> %2705 to <8 x bfloat>
  %2763 = shufflevector <8 x bfloat> %2762, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %2764 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2761, <2 x bfloat> %2763, float %2755, i32 2)
  %2765 = bitcast <4 x i32> %2705 to <8 x bfloat>
  %2766 = shufflevector <8 x bfloat> %2765, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %2767 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2764, <2 x bfloat> %2766, float %2755, i32 3)
  store i32 %2767, ptr addrspace(3) %391, align 4, !tbaa !7
  %2768 = extractelement <4 x i32> %2706, i64 0
  %2769 = and i32 %2768, 2147450879
  %2770 = extractelement <4 x i32> %2706, i64 1
  %2771 = and i32 %2770, 2147450879
  %2772 = extractelement <4 x i32> %2706, i64 2
  %2773 = and i32 %2772, 2147450879
  %2774 = extractelement <4 x i32> %2706, i64 3
  %2775 = and i32 %2774, 2147450879
  %2776 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2769, i32 %2771) #11, !srcloc !11
  %2777 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2773, i32 %2775) #11, !srcloc !11
  %2778 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2776, i32 %2777) #11, !srcloc !11
  %2779 = trunc i32 %2778 to i16
  %2780 = lshr i32 %2778, 16
  %2781 = trunc nuw i32 %2780 to i16
  %2782 = tail call noundef i16 @llvm.umax.i16(i16 %2779, i16 %2781)
  %2783 = zext i16 %2782 to i32
  %2784 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %2783, i32 177, i32 15, i32 15, i1 true)
  %2785 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %2783, i32 %2784)
  %2786 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %2785, i32 78, i32 15, i32 15, i1 true)
  %2787 = tail call noundef i32 @llvm.umax.i32(i32 %2785, i32 %2786)
  %2788 = shl i32 %2787, 16
  %2789 = add i32 %2788, 2097152
  %2790 = lshr i32 %2789, 23
  %2791 = and i32 %2790, 255
  %2792 = tail call i32 @llvm.umax.i32(i32 %2791, i32 2)
  %2793 = add nuw nsw i32 %2792, 254
  %2794 = and i32 %2793, 255
  %2795 = shl nuw nsw i32 %2794, 23
  %2796 = bitcast i32 %2795 to float
  %2797 = bitcast <4 x i32> %2706 to <8 x bfloat>
  %2798 = shufflevector <8 x bfloat> %2797, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %2799 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %2798, float %2796, i32 0)
  %2800 = bitcast <4 x i32> %2706 to <8 x bfloat>
  %2801 = shufflevector <8 x bfloat> %2800, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %2802 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2799, <2 x bfloat> %2801, float %2796, i32 1)
  %2803 = bitcast <4 x i32> %2706 to <8 x bfloat>
  %2804 = shufflevector <8 x bfloat> %2803, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %2805 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2802, <2 x bfloat> %2804, float %2796, i32 2)
  %2806 = bitcast <4 x i32> %2706 to <8 x bfloat>
  %2807 = shufflevector <8 x bfloat> %2806, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %2808 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2805, <2 x bfloat> %2807, float %2796, i32 3)
  store i32 %2808, ptr addrspace(3) %433, align 4, !tbaa !7
  %2809 = shl nuw nsw i32 %2794, 16
  %2810 = or disjoint i32 %2809, %2753
  %2811 = add nuw nsw i32 %205, 5888
  %2812 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %2811
  store i32 %2810, ptr addrspace(3) %2812, align 4, !tbaa !7
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %2813 = load <4 x i32>, ptr addrspace(3) %438, align 16, !tbaa !12
  %2814 = load <4 x i32>, ptr addrspace(3) %440, align 16, !tbaa !12
  %2815 = or disjoint i32 %209, 5632
  %2816 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %2815
  %2817 = load i32, ptr addrspace(3) %2816, align 4, !tbaa !7
  %2818 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 12288, i32 0)
  %2819 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 12544, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2820 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2813, <4 x i32> noundef %2597, <4 x float> noundef %2708, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2817, i32 noundef 0, i32 noundef %2612) #12
  %2821 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2814, <4 x i32> noundef %2599, <4 x float> noundef %2820, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2817, i32 noundef 2, i32 noundef %2612) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2822 = or disjoint i32 %146, 49152
  %2823 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2822, i32 %60, i32 2)
  %2824 = or disjoint i32 %146, 50176
  %2825 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2824, i32 %60, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2826 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2813, <4 x i32> noundef %2602, <4 x float> noundef %2714, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2817, i32 noundef 1, i32 noundef %2612) #12
  %2827 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2814, <4 x i32> noundef %2603, <4 x float> noundef %2826, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2817, i32 noundef 3, i32 noundef %2612) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2828 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2822, i32 %64, i32 2)
  %2829 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2824, i32 %64, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2830 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2813, <4 x i32> noundef %2606, <4 x float> noundef %2718, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2817, i32 noundef 0, i32 noundef %2613) #12
  %2831 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2814, <4 x i32> noundef %2607, <4 x float> noundef %2830, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2817, i32 noundef 2, i32 noundef %2613) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2832 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2822, i32 %68, i32 2)
  %2833 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2824, i32 %68, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2834 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2813, <4 x i32> noundef %2610, <4 x float> noundef %2722, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2817, i32 noundef 1, i32 noundef %2613) #12
  %2835 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2814, <4 x i32> noundef %2611, <4 x float> noundef %2834, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2817, i32 noundef 3, i32 noundef %2613) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2836 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2822, i32 %72, i32 2)
  %2837 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2824, i32 %72, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2838 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %1031, i32 %82, i32 0)
  %2839 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %1031, i32 %87, i32 0)
  %2840 = extractelement <4 x i32> %2818, i64 0
  %2841 = and i32 %2840, 2147450879
  %2842 = extractelement <4 x i32> %2818, i64 1
  %2843 = and i32 %2842, 2147450879
  %2844 = extractelement <4 x i32> %2818, i64 2
  %2845 = and i32 %2844, 2147450879
  %2846 = extractelement <4 x i32> %2818, i64 3
  %2847 = and i32 %2846, 2147450879
  %2848 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2841, i32 %2843) #11, !srcloc !11
  %2849 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2845, i32 %2847) #11, !srcloc !11
  %2850 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2848, i32 %2849) #11, !srcloc !11
  %2851 = trunc i32 %2850 to i16
  %2852 = lshr i32 %2850, 16
  %2853 = trunc nuw i32 %2852 to i16
  %2854 = tail call noundef i16 @llvm.umax.i16(i16 %2851, i16 %2853)
  %2855 = zext i16 %2854 to i32
  %2856 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %2855, i32 177, i32 15, i32 15, i1 true)
  %2857 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %2855, i32 %2856)
  %2858 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %2857, i32 78, i32 15, i32 15, i1 true)
  %2859 = tail call noundef i32 @llvm.umax.i32(i32 %2857, i32 %2858)
  %2860 = shl i32 %2859, 16
  %2861 = add i32 %2860, 2097152
  %2862 = lshr i32 %2861, 23
  %2863 = and i32 %2862, 255
  %2864 = tail call i32 @llvm.umax.i32(i32 %2863, i32 2)
  %2865 = add nuw nsw i32 %2864, 254
  %2866 = and i32 %2865, 255
  %2867 = shl nuw nsw i32 %2866, 23
  %2868 = bitcast i32 %2867 to float
  %2869 = bitcast <4 x i32> %2818 to <8 x bfloat>
  %2870 = shufflevector <8 x bfloat> %2869, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %2871 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %2870, float %2868, i32 0)
  %2872 = bitcast <4 x i32> %2818 to <8 x bfloat>
  %2873 = shufflevector <8 x bfloat> %2872, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %2874 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2871, <2 x bfloat> %2873, float %2868, i32 1)
  %2875 = bitcast <4 x i32> %2818 to <8 x bfloat>
  %2876 = shufflevector <8 x bfloat> %2875, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %2877 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2874, <2 x bfloat> %2876, float %2868, i32 2)
  %2878 = bitcast <4 x i32> %2818 to <8 x bfloat>
  %2879 = shufflevector <8 x bfloat> %2878, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %2880 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2877, <2 x bfloat> %2879, float %2868, i32 3)
  store i32 %2880, ptr addrspace(3) %142, align 4, !tbaa !7
  %2881 = extractelement <4 x i32> %2819, i64 0
  %2882 = and i32 %2881, 2147450879
  %2883 = extractelement <4 x i32> %2819, i64 1
  %2884 = and i32 %2883, 2147450879
  %2885 = extractelement <4 x i32> %2819, i64 2
  %2886 = and i32 %2885, 2147450879
  %2887 = extractelement <4 x i32> %2819, i64 3
  %2888 = and i32 %2887, 2147450879
  %2889 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2882, i32 %2884) #11, !srcloc !11
  %2890 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2886, i32 %2888) #11, !srcloc !11
  %2891 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2889, i32 %2890) #11, !srcloc !11
  %2892 = trunc i32 %2891 to i16
  %2893 = lshr i32 %2891, 16
  %2894 = trunc nuw i32 %2893 to i16
  %2895 = tail call noundef i16 @llvm.umax.i16(i16 %2892, i16 %2894)
  %2896 = zext i16 %2895 to i32
  %2897 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %2896, i32 177, i32 15, i32 15, i1 true)
  %2898 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %2896, i32 %2897)
  %2899 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %2898, i32 78, i32 15, i32 15, i1 true)
  %2900 = tail call noundef i32 @llvm.umax.i32(i32 %2898, i32 %2899)
  %2901 = shl i32 %2900, 16
  %2902 = add i32 %2901, 2097152
  %2903 = lshr i32 %2902, 23
  %2904 = and i32 %2903, 255
  %2905 = tail call i32 @llvm.umax.i32(i32 %2904, i32 2)
  %2906 = add nuw nsw i32 %2905, 254
  %2907 = and i32 %2906, 255
  %2908 = shl nuw nsw i32 %2907, 23
  %2909 = bitcast i32 %2908 to float
  %2910 = bitcast <4 x i32> %2819 to <8 x bfloat>
  %2911 = shufflevector <8 x bfloat> %2910, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %2912 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %2911, float %2909, i32 0)
  %2913 = bitcast <4 x i32> %2819 to <8 x bfloat>
  %2914 = shufflevector <8 x bfloat> %2913, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %2915 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2912, <2 x bfloat> %2914, float %2909, i32 1)
  %2916 = bitcast <4 x i32> %2819 to <8 x bfloat>
  %2917 = shufflevector <8 x bfloat> %2916, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %2918 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2915, <2 x bfloat> %2917, float %2909, i32 2)
  %2919 = bitcast <4 x i32> %2819 to <8 x bfloat>
  %2920 = shufflevector <8 x bfloat> %2919, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %2921 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2918, <2 x bfloat> %2920, float %2909, i32 3)
  store i32 %2921, ptr addrspace(3) %197, align 4, !tbaa !7
  %2922 = shl nuw nsw i32 %2907, 16
  %2923 = or disjoint i32 %2922, %2866
  %2924 = add nuw nsw i32 %205, 6144
  %2925 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %2924
  store i32 %2923, ptr addrspace(3) %2925, align 4, !tbaa !7
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %2926 = load <4 x i32>, ptr addrspace(3) %553, align 16, !tbaa !12
  %2927 = load <4 x i32>, ptr addrspace(3) %555, align 16, !tbaa !12
  %2928 = or disjoint i32 %209, 5888
  %2929 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %2928
  %2930 = load i32, ptr addrspace(3) %2929, align 4, !tbaa !7
  %2931 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 12800, i32 0)
  %2932 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 13056, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2933 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2926, <4 x i32> noundef %2710, <4 x float> noundef %2821, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2930, i32 noundef 0, i32 noundef %2725) #12
  %2934 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2927, <4 x i32> noundef %2712, <4 x float> noundef %2933, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2930, i32 noundef 2, i32 noundef %2725) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2935 = or disjoint i32 %146, 51200
  %2936 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2935, i32 %60, i32 2)
  %2937 = or disjoint i32 %146, 52224
  %2938 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2937, i32 %60, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2939 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2926, <4 x i32> noundef %2715, <4 x float> noundef %2827, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2930, i32 noundef 1, i32 noundef %2725) #12
  %2940 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2927, <4 x i32> noundef %2716, <4 x float> noundef %2939, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2930, i32 noundef 3, i32 noundef %2725) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2941 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2935, i32 %64, i32 2)
  %2942 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2937, i32 %64, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2943 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2926, <4 x i32> noundef %2719, <4 x float> noundef %2831, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2930, i32 noundef 0, i32 noundef %2726) #12
  %2944 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2927, <4 x i32> noundef %2720, <4 x float> noundef %2943, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2930, i32 noundef 2, i32 noundef %2726) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2945 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2935, i32 %68, i32 2)
  %2946 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2937, i32 %68, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %2947 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2926, <4 x i32> noundef %2723, <4 x float> noundef %2835, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %2930, i32 noundef 1, i32 noundef %2726) #12
  %2948 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %2927, <4 x i32> noundef %2724, <4 x float> noundef %2947, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %2930, i32 noundef 3, i32 noundef %2726) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2949 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2935, i32 %72, i32 2)
  %2950 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %2937, i32 %72, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %2951 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %1144, i32 %82, i32 0)
  %2952 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %1144, i32 %87, i32 0)
  %2953 = extractelement <4 x i32> %2931, i64 0
  %2954 = and i32 %2953, 2147450879
  %2955 = extractelement <4 x i32> %2931, i64 1
  %2956 = and i32 %2955, 2147450879
  %2957 = extractelement <4 x i32> %2931, i64 2
  %2958 = and i32 %2957, 2147450879
  %2959 = extractelement <4 x i32> %2931, i64 3
  %2960 = and i32 %2959, 2147450879
  %2961 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2954, i32 %2956) #11, !srcloc !11
  %2962 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2958, i32 %2960) #11, !srcloc !11
  %2963 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2961, i32 %2962) #11, !srcloc !11
  %2964 = trunc i32 %2963 to i16
  %2965 = lshr i32 %2963, 16
  %2966 = trunc nuw i32 %2965 to i16
  %2967 = tail call noundef i16 @llvm.umax.i16(i16 %2964, i16 %2966)
  %2968 = zext i16 %2967 to i32
  %2969 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %2968, i32 177, i32 15, i32 15, i1 true)
  %2970 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %2968, i32 %2969)
  %2971 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %2970, i32 78, i32 15, i32 15, i1 true)
  %2972 = tail call noundef i32 @llvm.umax.i32(i32 %2970, i32 %2971)
  %2973 = shl i32 %2972, 16
  %2974 = add i32 %2973, 2097152
  %2975 = lshr i32 %2974, 23
  %2976 = and i32 %2975, 255
  %2977 = tail call i32 @llvm.umax.i32(i32 %2976, i32 2)
  %2978 = add nuw nsw i32 %2977, 254
  %2979 = and i32 %2978, 255
  %2980 = shl nuw nsw i32 %2979, 23
  %2981 = bitcast i32 %2980 to float
  %2982 = bitcast <4 x i32> %2931 to <8 x bfloat>
  %2983 = shufflevector <8 x bfloat> %2982, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %2984 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %2983, float %2981, i32 0)
  %2985 = bitcast <4 x i32> %2931 to <8 x bfloat>
  %2986 = shufflevector <8 x bfloat> %2985, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %2987 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2984, <2 x bfloat> %2986, float %2981, i32 1)
  %2988 = bitcast <4 x i32> %2931 to <8 x bfloat>
  %2989 = shufflevector <8 x bfloat> %2988, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %2990 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2987, <2 x bfloat> %2989, float %2981, i32 2)
  %2991 = bitcast <4 x i32> %2931 to <8 x bfloat>
  %2992 = shufflevector <8 x bfloat> %2991, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %2993 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %2990, <2 x bfloat> %2992, float %2981, i32 3)
  store i32 %2993, ptr addrspace(3) %254, align 4, !tbaa !7
  %2994 = extractelement <4 x i32> %2932, i64 0
  %2995 = and i32 %2994, 2147450879
  %2996 = extractelement <4 x i32> %2932, i64 1
  %2997 = and i32 %2996, 2147450879
  %2998 = extractelement <4 x i32> %2932, i64 2
  %2999 = and i32 %2998, 2147450879
  %3000 = extractelement <4 x i32> %2932, i64 3
  %3001 = and i32 %3000, 2147450879
  %3002 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2995, i32 %2997) #11, !srcloc !11
  %3003 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %2999, i32 %3001) #11, !srcloc !11
  %3004 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %3002, i32 %3003) #11, !srcloc !11
  %3005 = trunc i32 %3004 to i16
  %3006 = lshr i32 %3004, 16
  %3007 = trunc nuw i32 %3006 to i16
  %3008 = tail call noundef i16 @llvm.umax.i16(i16 %3005, i16 %3007)
  %3009 = zext i16 %3008 to i32
  %3010 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %3009, i32 177, i32 15, i32 15, i1 true)
  %3011 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %3009, i32 %3010)
  %3012 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %3011, i32 78, i32 15, i32 15, i1 true)
  %3013 = tail call noundef i32 @llvm.umax.i32(i32 %3011, i32 %3012)
  %3014 = shl i32 %3013, 16
  %3015 = add i32 %3014, 2097152
  %3016 = lshr i32 %3015, 23
  %3017 = and i32 %3016, 255
  %3018 = tail call i32 @llvm.umax.i32(i32 %3017, i32 2)
  %3019 = add nuw nsw i32 %3018, 254
  %3020 = and i32 %3019, 255
  %3021 = shl nuw nsw i32 %3020, 23
  %3022 = bitcast i32 %3021 to float
  %3023 = bitcast <4 x i32> %2932 to <8 x bfloat>
  %3024 = shufflevector <8 x bfloat> %3023, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %3025 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %3024, float %3022, i32 0)
  %3026 = bitcast <4 x i32> %2932 to <8 x bfloat>
  %3027 = shufflevector <8 x bfloat> %3026, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %3028 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %3025, <2 x bfloat> %3027, float %3022, i32 1)
  %3029 = bitcast <4 x i32> %2932 to <8 x bfloat>
  %3030 = shufflevector <8 x bfloat> %3029, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %3031 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %3028, <2 x bfloat> %3030, float %3022, i32 2)
  %3032 = bitcast <4 x i32> %2932 to <8 x bfloat>
  %3033 = shufflevector <8 x bfloat> %3032, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %3034 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %3031, <2 x bfloat> %3033, float %3022, i32 3)
  store i32 %3034, ptr addrspace(3) %303, align 4, !tbaa !7
  %3035 = shl nuw nsw i32 %3020, 16
  %3036 = or disjoint i32 %3035, %2979
  %3037 = add nuw nsw i32 %205, 6400
  %3038 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %3037
  store i32 %3036, ptr addrspace(3) %3038, align 4, !tbaa !7
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %3039 = load <4 x i32>, ptr addrspace(3) %319, align 16, !tbaa !12
  %3040 = load <4 x i32>, ptr addrspace(3) %323, align 16, !tbaa !12
  %3041 = or disjoint i32 %209, 6144
  %3042 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %3041
  %3043 = load i32, ptr addrspace(3) %3042, align 4, !tbaa !7
  %3044 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 13312, i32 0)
  %3045 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 13568, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %3046 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3039, <4 x i32> noundef %2823, <4 x float> noundef %2934, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %3043, i32 noundef 0, i32 noundef %2838) #12
  %3047 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3040, <4 x i32> noundef %2825, <4 x float> noundef %3046, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %3043, i32 noundef 2, i32 noundef %2838) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %3048 = or disjoint i32 %146, 53248
  %3049 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %3048, i32 %60, i32 2)
  %3050 = or disjoint i32 %146, 54272
  %3051 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %3050, i32 %60, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %3052 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3039, <4 x i32> noundef %2828, <4 x float> noundef %2940, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %3043, i32 noundef 1, i32 noundef %2838) #12
  %3053 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3040, <4 x i32> noundef %2829, <4 x float> noundef %3052, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %3043, i32 noundef 3, i32 noundef %2838) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %3054 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %3048, i32 %64, i32 2)
  %3055 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %3050, i32 %64, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %3056 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3039, <4 x i32> noundef %2832, <4 x float> noundef %2944, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %3043, i32 noundef 0, i32 noundef %2839) #12
  %3057 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3040, <4 x i32> noundef %2833, <4 x float> noundef %3056, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %3043, i32 noundef 2, i32 noundef %2839) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %3058 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %3048, i32 %68, i32 2)
  %3059 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %3050, i32 %68, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %3060 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3039, <4 x i32> noundef %2836, <4 x float> noundef %2948, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %3043, i32 noundef 1, i32 noundef %2839) #12
  %3061 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3040, <4 x i32> noundef %2837, <4 x float> noundef %3060, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %3043, i32 noundef 3, i32 noundef %2839) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %3062 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %3048, i32 %72, i32 2)
  %3063 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %3050, i32 %72, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %3064 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %1257, i32 %82, i32 0)
  %3065 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %1257, i32 %87, i32 0)
  %3066 = extractelement <4 x i32> %3044, i64 0
  %3067 = and i32 %3066, 2147450879
  %3068 = extractelement <4 x i32> %3044, i64 1
  %3069 = and i32 %3068, 2147450879
  %3070 = extractelement <4 x i32> %3044, i64 2
  %3071 = and i32 %3070, 2147450879
  %3072 = extractelement <4 x i32> %3044, i64 3
  %3073 = and i32 %3072, 2147450879
  %3074 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %3067, i32 %3069) #11, !srcloc !11
  %3075 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %3071, i32 %3073) #11, !srcloc !11
  %3076 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %3074, i32 %3075) #11, !srcloc !11
  %3077 = trunc i32 %3076 to i16
  %3078 = lshr i32 %3076, 16
  %3079 = trunc nuw i32 %3078 to i16
  %3080 = tail call noundef i16 @llvm.umax.i16(i16 %3077, i16 %3079)
  %3081 = zext i16 %3080 to i32
  %3082 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %3081, i32 177, i32 15, i32 15, i1 true)
  %3083 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %3081, i32 %3082)
  %3084 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %3083, i32 78, i32 15, i32 15, i1 true)
  %3085 = tail call noundef i32 @llvm.umax.i32(i32 %3083, i32 %3084)
  %3086 = shl i32 %3085, 16
  %3087 = add i32 %3086, 2097152
  %3088 = lshr i32 %3087, 23
  %3089 = and i32 %3088, 255
  %3090 = tail call i32 @llvm.umax.i32(i32 %3089, i32 2)
  %3091 = add nuw nsw i32 %3090, 254
  %3092 = and i32 %3091, 255
  %3093 = shl nuw nsw i32 %3092, 23
  %3094 = bitcast i32 %3093 to float
  %3095 = bitcast <4 x i32> %3044 to <8 x bfloat>
  %3096 = shufflevector <8 x bfloat> %3095, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %3097 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %3096, float %3094, i32 0)
  %3098 = bitcast <4 x i32> %3044 to <8 x bfloat>
  %3099 = shufflevector <8 x bfloat> %3098, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %3100 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %3097, <2 x bfloat> %3099, float %3094, i32 1)
  %3101 = bitcast <4 x i32> %3044 to <8 x bfloat>
  %3102 = shufflevector <8 x bfloat> %3101, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %3103 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %3100, <2 x bfloat> %3102, float %3094, i32 2)
  %3104 = bitcast <4 x i32> %3044 to <8 x bfloat>
  %3105 = shufflevector <8 x bfloat> %3104, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %3106 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %3103, <2 x bfloat> %3105, float %3094, i32 3)
  store i32 %3106, ptr addrspace(3) %391, align 4, !tbaa !7
  %3107 = extractelement <4 x i32> %3045, i64 0
  %3108 = and i32 %3107, 2147450879
  %3109 = extractelement <4 x i32> %3045, i64 1
  %3110 = and i32 %3109, 2147450879
  %3111 = extractelement <4 x i32> %3045, i64 2
  %3112 = and i32 %3111, 2147450879
  %3113 = extractelement <4 x i32> %3045, i64 3
  %3114 = and i32 %3113, 2147450879
  %3115 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %3108, i32 %3110) #11, !srcloc !11
  %3116 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %3112, i32 %3114) #11, !srcloc !11
  %3117 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %3115, i32 %3116) #11, !srcloc !11
  %3118 = trunc i32 %3117 to i16
  %3119 = lshr i32 %3117, 16
  %3120 = trunc nuw i32 %3119 to i16
  %3121 = tail call noundef i16 @llvm.umax.i16(i16 %3118, i16 %3120)
  %3122 = zext i16 %3121 to i32
  %3123 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %3122, i32 177, i32 15, i32 15, i1 true)
  %3124 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %3122, i32 %3123)
  %3125 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %3124, i32 78, i32 15, i32 15, i1 true)
  %3126 = tail call noundef i32 @llvm.umax.i32(i32 %3124, i32 %3125)
  %3127 = shl i32 %3126, 16
  %3128 = add i32 %3127, 2097152
  %3129 = lshr i32 %3128, 23
  %3130 = and i32 %3129, 255
  %3131 = tail call i32 @llvm.umax.i32(i32 %3130, i32 2)
  %3132 = add nuw nsw i32 %3131, 254
  %3133 = and i32 %3132, 255
  %3134 = shl nuw nsw i32 %3133, 23
  %3135 = bitcast i32 %3134 to float
  %3136 = bitcast <4 x i32> %3045 to <8 x bfloat>
  %3137 = shufflevector <8 x bfloat> %3136, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %3138 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %3137, float %3135, i32 0)
  %3139 = bitcast <4 x i32> %3045 to <8 x bfloat>
  %3140 = shufflevector <8 x bfloat> %3139, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %3141 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %3138, <2 x bfloat> %3140, float %3135, i32 1)
  %3142 = bitcast <4 x i32> %3045 to <8 x bfloat>
  %3143 = shufflevector <8 x bfloat> %3142, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %3144 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %3141, <2 x bfloat> %3143, float %3135, i32 2)
  %3145 = bitcast <4 x i32> %3045 to <8 x bfloat>
  %3146 = shufflevector <8 x bfloat> %3145, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %3147 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %3144, <2 x bfloat> %3146, float %3135, i32 3)
  store i32 %3147, ptr addrspace(3) %433, align 4, !tbaa !7
  %3148 = shl nuw nsw i32 %3133, 16
  %3149 = or disjoint i32 %3148, %3092
  %3150 = add nuw nsw i32 %205, 6656
  %3151 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %3150
  store i32 %3149, ptr addrspace(3) %3151, align 4, !tbaa !7
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %3152 = load <4 x i32>, ptr addrspace(3) %438, align 16, !tbaa !12
  %3153 = load <4 x i32>, ptr addrspace(3) %440, align 16, !tbaa !12
  %3154 = or disjoint i32 %209, 6400
  %3155 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %3154
  %3156 = load i32, ptr addrspace(3) %3155, align 4, !tbaa !7
  %3157 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 13824, i32 0)
  %3158 = tail call noundef <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %27, i32 %91, i32 14080, i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %3159 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3152, <4 x i32> noundef %2936, <4 x float> noundef %3047, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %3156, i32 noundef 0, i32 noundef %2951) #12
  %3160 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3153, <4 x i32> noundef %2938, <4 x float> noundef %3159, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %3156, i32 noundef 2, i32 noundef %2951) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %3161 = or disjoint i32 %146, 55296
  %3162 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %3161, i32 %60, i32 2)
  %3163 = or disjoint i32 %146, 56320
  %3164 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %3163, i32 %60, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %3165 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3152, <4 x i32> noundef %2941, <4 x float> noundef %3053, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %3156, i32 noundef 1, i32 noundef %2951) #12
  %3166 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3153, <4 x i32> noundef %2942, <4 x float> noundef %3165, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %3156, i32 noundef 3, i32 noundef %2951) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %3167 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %3161, i32 %64, i32 2)
  %3168 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %3163, i32 %64, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %3169 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3152, <4 x i32> noundef %2945, <4 x float> noundef %3057, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %3156, i32 noundef 0, i32 noundef %2952) #12
  %3170 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3153, <4 x i32> noundef %2946, <4 x float> noundef %3169, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %3156, i32 noundef 2, i32 noundef %2952) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %3171 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %3161, i32 %68, i32 2)
  %3172 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %3163, i32 %68, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %3173 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3152, <4 x i32> noundef %2949, <4 x float> noundef %3061, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %3156, i32 noundef 1, i32 noundef %2952) #12
  %3174 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3153, <4 x i32> noundef %2950, <4 x float> noundef %3173, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %3156, i32 noundef 3, i32 noundef %2952) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %3175 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %3161, i32 %72, i32 2)
  %3176 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %24, i32 %3163, i32 %72, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %3177 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %1370, i32 %82, i32 0)
  %3178 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %25, i32 %1370, i32 %87, i32 0)
  %3179 = extractelement <4 x i32> %3157, i64 0
  %3180 = and i32 %3179, 2147450879
  %3181 = extractelement <4 x i32> %3157, i64 1
  %3182 = and i32 %3181, 2147450879
  %3183 = extractelement <4 x i32> %3157, i64 2
  %3184 = and i32 %3183, 2147450879
  %3185 = extractelement <4 x i32> %3157, i64 3
  %3186 = and i32 %3185, 2147450879
  %3187 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %3180, i32 %3182) #11, !srcloc !11
  %3188 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %3184, i32 %3186) #11, !srcloc !11
  %3189 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %3187, i32 %3188) #11, !srcloc !11
  %3190 = trunc i32 %3189 to i16
  %3191 = lshr i32 %3189, 16
  %3192 = trunc nuw i32 %3191 to i16
  %3193 = tail call noundef i16 @llvm.umax.i16(i16 %3190, i16 %3192)
  %3194 = zext i16 %3193 to i32
  %3195 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %3194, i32 177, i32 15, i32 15, i1 true)
  %3196 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %3194, i32 %3195)
  %3197 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %3196, i32 78, i32 15, i32 15, i1 true)
  %3198 = tail call noundef i32 @llvm.umax.i32(i32 %3196, i32 %3197)
  %3199 = shl i32 %3198, 16
  %3200 = add i32 %3199, 2097152
  %3201 = lshr i32 %3200, 23
  %3202 = and i32 %3201, 255
  %3203 = tail call i32 @llvm.umax.i32(i32 %3202, i32 2)
  %3204 = add nuw nsw i32 %3203, 254
  %3205 = and i32 %3204, 255
  %3206 = shl nuw nsw i32 %3205, 23
  %3207 = bitcast i32 %3206 to float
  %3208 = bitcast <4 x i32> %3157 to <8 x bfloat>
  %3209 = shufflevector <8 x bfloat> %3208, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %3210 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %3209, float %3207, i32 0)
  %3211 = bitcast <4 x i32> %3157 to <8 x bfloat>
  %3212 = shufflevector <8 x bfloat> %3211, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %3213 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %3210, <2 x bfloat> %3212, float %3207, i32 1)
  %3214 = bitcast <4 x i32> %3157 to <8 x bfloat>
  %3215 = shufflevector <8 x bfloat> %3214, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %3216 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %3213, <2 x bfloat> %3215, float %3207, i32 2)
  %3217 = bitcast <4 x i32> %3157 to <8 x bfloat>
  %3218 = shufflevector <8 x bfloat> %3217, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %3219 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %3216, <2 x bfloat> %3218, float %3207, i32 3)
  store i32 %3219, ptr addrspace(3) %142, align 4, !tbaa !7
  %3220 = extractelement <4 x i32> %3158, i64 0
  %3221 = and i32 %3220, 2147450879
  %3222 = extractelement <4 x i32> %3158, i64 1
  %3223 = and i32 %3222, 2147450879
  %3224 = extractelement <4 x i32> %3158, i64 2
  %3225 = and i32 %3224, 2147450879
  %3226 = extractelement <4 x i32> %3158, i64 3
  %3227 = and i32 %3226, 2147450879
  %3228 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %3221, i32 %3223) #11, !srcloc !11
  %3229 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %3225, i32 %3227) #11, !srcloc !11
  %3230 = tail call noundef i32 asm "v_pk_max_u16 $0, $1, $2", "=v,v,v"(i32 %3228, i32 %3229) #11, !srcloc !11
  %3231 = trunc i32 %3230 to i16
  %3232 = lshr i32 %3230, 16
  %3233 = trunc nuw i32 %3232 to i16
  %3234 = tail call noundef i16 @llvm.umax.i16(i16 %3231, i16 %3233)
  %3235 = zext i16 %3234 to i32
  %3236 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 range(i32 0, 65536) %3235, i32 177, i32 15, i32 15, i1 true)
  %3237 = tail call noundef i32 @llvm.umax.i32(i32 range(i32 0, 65536) %3235, i32 %3236)
  %3238 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %3237, i32 78, i32 15, i32 15, i1 true)
  %3239 = tail call noundef i32 @llvm.umax.i32(i32 %3237, i32 %3238)
  %3240 = shl i32 %3239, 16
  %3241 = add i32 %3240, 2097152
  %3242 = lshr i32 %3241, 23
  %3243 = and i32 %3242, 255
  %3244 = tail call i32 @llvm.umax.i32(i32 %3243, i32 2)
  %3245 = add nuw nsw i32 %3244, 254
  %3246 = and i32 %3245, 255
  %3247 = shl nuw nsw i32 %3246, 23
  %3248 = bitcast i32 %3247 to float
  %3249 = bitcast <4 x i32> %3158 to <8 x bfloat>
  %3250 = shufflevector <8 x bfloat> %3249, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %3251 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 0, <2 x bfloat> %3250, float %3248, i32 0)
  %3252 = bitcast <4 x i32> %3158 to <8 x bfloat>
  %3253 = shufflevector <8 x bfloat> %3252, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %3254 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %3251, <2 x bfloat> %3253, float %3248, i32 1)
  %3255 = bitcast <4 x i32> %3158 to <8 x bfloat>
  %3256 = shufflevector <8 x bfloat> %3255, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %3257 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %3254, <2 x bfloat> %3256, float %3248, i32 2)
  %3258 = bitcast <4 x i32> %3158 to <8 x bfloat>
  %3259 = shufflevector <8 x bfloat> %3258, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %3260 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %3257, <2 x bfloat> %3259, float %3248, i32 3)
  store i32 %3260, ptr addrspace(3) %197, align 4, !tbaa !7
  %3261 = shl nuw nsw i32 %3246, 16
  %3262 = or disjoint i32 %3261, %3205
  %3263 = add nuw nsw i32 %205, 6912
  %3264 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %3263
  store i32 %3262, ptr addrspace(3) %3264, align 4, !tbaa !7
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %3265 = load <4 x i32>, ptr addrspace(3) %553, align 16, !tbaa !12
  %3266 = load <4 x i32>, ptr addrspace(3) %555, align 16, !tbaa !12
  %3267 = or disjoint i32 %209, 6656
  %3268 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %3267
  %3269 = load i32, ptr addrspace(3) %3268, align 4, !tbaa !7
  %3270 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3265, <4 x i32> noundef %3049, <4 x float> noundef %3160, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %3269, i32 noundef 0, i32 noundef %3064) #12
  %3271 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3266, <4 x i32> noundef %3051, <4 x float> noundef %3270, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %3269, i32 noundef 2, i32 noundef %3064) #12
  %3272 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3265, <4 x i32> noundef %3054, <4 x float> noundef %3166, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %3269, i32 noundef 1, i32 noundef %3064) #12
  %3273 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3266, <4 x i32> noundef %3055, <4 x float> noundef %3272, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %3269, i32 noundef 3, i32 noundef %3064) #12
  %3274 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3265, <4 x i32> noundef %3058, <4 x float> noundef %3170, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %3269, i32 noundef 0, i32 noundef %3065) #12
  %3275 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3266, <4 x i32> noundef %3059, <4 x float> noundef %3274, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %3269, i32 noundef 2, i32 noundef %3065) #12
  %3276 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3265, <4 x i32> noundef %3062, <4 x float> noundef %3174, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %3269, i32 noundef 1, i32 noundef %3065) #12
  %3277 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3266, <4 x i32> noundef %3063, <4 x float> noundef %3276, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %3269, i32 noundef 3, i32 noundef %3065) #12
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %3278 = load <4 x i32>, ptr addrspace(3) %319, align 16, !tbaa !12
  %3279 = load <4 x i32>, ptr addrspace(3) %323, align 16, !tbaa !12
  %3280 = or disjoint i32 %209, 6912
  %3281 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 6144), i32 0, i32 %3280
  %3282 = load i32, ptr addrspace(3) %3281, align 4, !tbaa !7
  %3283 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3278, <4 x i32> noundef %3162, <4 x float> noundef %3271, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %3282, i32 noundef 0, i32 noundef %3177) #12
  %3284 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3279, <4 x i32> noundef %3164, <4 x float> noundef %3283, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %3282, i32 noundef 2, i32 noundef %3177) #12
  %3285 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3278, <4 x i32> noundef %3167, <4 x float> noundef %3273, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %3282, i32 noundef 1, i32 noundef %3177) #12
  %3286 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3279, <4 x i32> noundef %3168, <4 x float> noundef %3285, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %3282, i32 noundef 3, i32 noundef %3177) #12
  %3287 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3278, <4 x i32> noundef %3171, <4 x float> noundef %3275, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %3282, i32 noundef 0, i32 noundef %3178) #12
  %3288 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3279, <4 x i32> noundef %3172, <4 x float> noundef %3287, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %3282, i32 noundef 2, i32 noundef %3178) #12
  %3289 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3278, <4 x i32> noundef %3175, <4 x float> noundef %3277, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %3282, i32 noundef 1, i32 noundef %3178) #12
  %3290 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %3279, <4 x i32> noundef %3176, <4 x float> noundef %3289, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %3282, i32 noundef 3, i32 noundef %3178) #12
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  tail call void @llvm.experimental.noalias.scope.decl(metadata !13)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !16)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !18)
  %3291 = shl nuw nsw i32 %34, 5
  %3292 = shl nuw nsw i32 %48, 10
  %3293 = or disjoint i32 %3292, %144
  %3294 = add nuw i32 %3293, %3291
  %3295 = extractelement <4 x float> %3284, i64 0
  %3296 = sext i32 %3294 to i64
  %3297 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3296
  %3298 = addrspacecast ptr %3297 to ptr addrspace(3)
  store float %3295, ptr addrspace(3) %3298, align 4, !tbaa !20, !alias.scope !18, !noalias !22
  %3299 = extractelement <4 x float> %3284, i64 1
  %3300 = add nuw i32 %3294, 256
  %3301 = sext i32 %3300 to i64
  %3302 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3301
  %3303 = addrspacecast ptr %3302 to ptr addrspace(3)
  store float %3299, ptr addrspace(3) %3303, align 4, !tbaa !20, !alias.scope !18, !noalias !22
  %3304 = extractelement <4 x float> %3284, i64 2
  %3305 = add nuw i32 %3294, 512
  %3306 = sext i32 %3305 to i64
  %3307 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3306
  %3308 = addrspacecast ptr %3307 to ptr addrspace(3)
  store float %3304, ptr addrspace(3) %3308, align 4, !tbaa !20, !alias.scope !18, !noalias !22
  %3309 = extractelement <4 x float> %3284, i64 3
  %3310 = add nuw i32 %3294, 768
  %3311 = sext i32 %3310 to i64
  %3312 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3311
  %3313 = addrspacecast ptr %3312 to ptr addrspace(3)
  store float %3309, ptr addrspace(3) %3313, align 4, !tbaa !20, !alias.scope !18, !noalias !22
  %3314 = add nuw i32 %3294, 128
  %3315 = extractelement <4 x float> %3286, i64 0
  %3316 = sext i32 %3314 to i64
  %3317 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3316
  %3318 = addrspacecast ptr %3317 to ptr addrspace(3)
  store float %3315, ptr addrspace(3) %3318, align 4, !tbaa !20, !alias.scope !18, !noalias !22
  %3319 = extractelement <4 x float> %3286, i64 1
  %3320 = add nuw i32 %3294, 384
  %3321 = sext i32 %3320 to i64
  %3322 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3321
  %3323 = addrspacecast ptr %3322 to ptr addrspace(3)
  store float %3319, ptr addrspace(3) %3323, align 4, !tbaa !20, !alias.scope !18, !noalias !22
  %3324 = extractelement <4 x float> %3286, i64 2
  %3325 = add nuw i32 %3294, 640
  %3326 = sext i32 %3325 to i64
  %3327 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3326
  %3328 = addrspacecast ptr %3327 to ptr addrspace(3)
  store float %3324, ptr addrspace(3) %3328, align 4, !tbaa !20, !alias.scope !18, !noalias !22
  %3329 = extractelement <4 x float> %3286, i64 3
  %3330 = add nuw i32 %3294, 896
  %3331 = sext i32 %3330 to i64
  %3332 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3331
  %3333 = addrspacecast ptr %3332 to ptr addrspace(3)
  store float %3329, ptr addrspace(3) %3333, align 4, !tbaa !20, !alias.scope !18, !noalias !22
  %3334 = or disjoint i32 %3294, 16
  %3335 = extractelement <4 x float> %3288, i64 0
  %3336 = sext i32 %3334 to i64
  %3337 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3336
  %3338 = addrspacecast ptr %3337 to ptr addrspace(3)
  store float %3335, ptr addrspace(3) %3338, align 4, !tbaa !20, !alias.scope !18, !noalias !22
  %3339 = extractelement <4 x float> %3288, i64 1
  %3340 = add nuw i32 %3294, 272
  %3341 = sext i32 %3340 to i64
  %3342 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3341
  %3343 = addrspacecast ptr %3342 to ptr addrspace(3)
  store float %3339, ptr addrspace(3) %3343, align 4, !tbaa !20, !alias.scope !18, !noalias !22
  %3344 = extractelement <4 x float> %3288, i64 2
  %3345 = add nuw i32 %3294, 528
  %3346 = sext i32 %3345 to i64
  %3347 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3346
  %3348 = addrspacecast ptr %3347 to ptr addrspace(3)
  store float %3344, ptr addrspace(3) %3348, align 4, !tbaa !20, !alias.scope !18, !noalias !22
  %3349 = extractelement <4 x float> %3288, i64 3
  %3350 = add nuw i32 %3294, 784
  %3351 = sext i32 %3350 to i64
  %3352 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3351
  %3353 = addrspacecast ptr %3352 to ptr addrspace(3)
  store float %3349, ptr addrspace(3) %3353, align 4, !tbaa !20, !alias.scope !18, !noalias !22
  %3354 = add nuw i32 %3294, 144
  %3355 = extractelement <4 x float> %3290, i64 0
  %3356 = sext i32 %3354 to i64
  %3357 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3356
  %3358 = addrspacecast ptr %3357 to ptr addrspace(3)
  store float %3355, ptr addrspace(3) %3358, align 4, !tbaa !20, !alias.scope !18, !noalias !22
  %3359 = extractelement <4 x float> %3290, i64 1
  %3360 = add nuw i32 %3294, 400
  %3361 = sext i32 %3360 to i64
  %3362 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3361
  %3363 = addrspacecast ptr %3362 to ptr addrspace(3)
  store float %3359, ptr addrspace(3) %3363, align 4, !tbaa !20, !alias.scope !18, !noalias !22
  %3364 = extractelement <4 x float> %3290, i64 2
  %3365 = add nuw i32 %3294, 656
  %3366 = sext i32 %3365 to i64
  %3367 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3366
  %3368 = addrspacecast ptr %3367 to ptr addrspace(3)
  store float %3364, ptr addrspace(3) %3368, align 4, !tbaa !20, !alias.scope !18, !noalias !22
  %3369 = extractelement <4 x float> %3290, i64 3
  %3370 = add nuw i32 %3294, 912
  %3371 = sext i32 %3370 to i64
  %3372 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3371
  %3373 = addrspacecast ptr %3372 to ptr addrspace(3)
  store float %3369, ptr addrspace(3) %3373, align 4, !tbaa !20, !alias.scope !18, !noalias !22
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier(), !noalias !23
  fence syncscope("workgroup") acquire
  %3374 = lshr i32 %21, 4
  %3375 = lshr i32 %21, 2
  %3376 = and i32 %3375, 3
  %3377 = and i32 %21, 3
  %3378 = shl nuw nsw i32 %3377, 3
  %3379 = shl nuw nsw i32 %3376, 5
  %3380 = shl nuw nsw i32 %3374, 8
  %3381 = or disjoint i32 %3378, %3380
  %3382 = or disjoint i32 %3381, %3379
  %3383 = or disjoint i32 %3382, 128
  %3384 = zext nneg i32 %3382 to i64
  %3385 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3384
  %3386 = addrspacecast ptr %3385 to ptr addrspace(3)
  %3387 = load float, ptr addrspace(3) %3386, align 4, !tbaa !20, !alias.scope !18, !noalias !22
  %3388 = zext nneg i32 %3383 to i64
  %3389 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3388
  %3390 = addrspacecast ptr %3389 to ptr addrspace(3)
  %3391 = load float, ptr addrspace(3) %3390, align 4, !tbaa !20, !alias.scope !18, !noalias !22
  %3392 = or disjoint i32 %3382, 1
  %3393 = zext nneg i32 %3392 to i64
  %3394 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3393
  %3395 = addrspacecast ptr %3394 to ptr addrspace(3)
  %3396 = load float, ptr addrspace(3) %3395, align 4, !tbaa !20, !alias.scope !18, !noalias !22
  %3397 = or disjoint i32 %3382, 129
  %3398 = zext nneg i32 %3397 to i64
  %3399 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3398
  %3400 = addrspacecast ptr %3399 to ptr addrspace(3)
  %3401 = load float, ptr addrspace(3) %3400, align 4, !tbaa !20, !alias.scope !18, !noalias !22
  %3402 = or disjoint i32 %3382, 2
  %3403 = zext nneg i32 %3402 to i64
  %3404 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3403
  %3405 = addrspacecast ptr %3404 to ptr addrspace(3)
  %3406 = load float, ptr addrspace(3) %3405, align 4, !tbaa !20, !alias.scope !18, !noalias !22
  %3407 = or disjoint i32 %3382, 130
  %3408 = zext nneg i32 %3407 to i64
  %3409 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3408
  %3410 = addrspacecast ptr %3409 to ptr addrspace(3)
  %3411 = load float, ptr addrspace(3) %3410, align 4, !tbaa !20, !alias.scope !18, !noalias !22
  %3412 = or disjoint i32 %3382, 3
  %3413 = zext nneg i32 %3412 to i64
  %3414 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3413
  %3415 = addrspacecast ptr %3414 to ptr addrspace(3)
  %3416 = load float, ptr addrspace(3) %3415, align 4, !tbaa !20, !alias.scope !18, !noalias !22
  %3417 = or disjoint i32 %3382, 131
  %3418 = zext nneg i32 %3417 to i64
  %3419 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3418
  %3420 = addrspacecast ptr %3419 to ptr addrspace(3)
  %3421 = load float, ptr addrspace(3) %3420, align 4, !tbaa !20, !alias.scope !18, !noalias !22
  %3422 = or disjoint i32 %3382, 4
  %3423 = zext nneg i32 %3422 to i64
  %3424 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3423
  %3425 = addrspacecast ptr %3424 to ptr addrspace(3)
  %3426 = load float, ptr addrspace(3) %3425, align 4, !tbaa !20, !alias.scope !18, !noalias !22
  %3427 = or disjoint i32 %3382, 132
  %3428 = zext nneg i32 %3427 to i64
  %3429 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3428
  %3430 = addrspacecast ptr %3429 to ptr addrspace(3)
  %3431 = load float, ptr addrspace(3) %3430, align 4, !tbaa !20, !alias.scope !18, !noalias !22
  %3432 = or disjoint i32 %3382, 5
  %3433 = zext nneg i32 %3432 to i64
  %3434 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3433
  %3435 = addrspacecast ptr %3434 to ptr addrspace(3)
  %3436 = load float, ptr addrspace(3) %3435, align 4, !tbaa !20, !alias.scope !18, !noalias !22
  %3437 = or disjoint i32 %3382, 133
  %3438 = zext nneg i32 %3437 to i64
  %3439 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3438
  %3440 = addrspacecast ptr %3439 to ptr addrspace(3)
  %3441 = load float, ptr addrspace(3) %3440, align 4, !tbaa !20, !alias.scope !18, !noalias !22
  %3442 = or disjoint i32 %3382, 6
  %3443 = zext nneg i32 %3442 to i64
  %3444 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3443
  %3445 = addrspacecast ptr %3444 to ptr addrspace(3)
  %3446 = load float, ptr addrspace(3) %3445, align 4, !tbaa !20, !alias.scope !18, !noalias !22
  %3447 = or disjoint i32 %3382, 134
  %3448 = zext nneg i32 %3447 to i64
  %3449 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3448
  %3450 = addrspacecast ptr %3449 to ptr addrspace(3)
  %3451 = load float, ptr addrspace(3) %3450, align 4, !tbaa !20, !alias.scope !18, !noalias !22
  %3452 = or disjoint i32 %3382, 7
  %3453 = zext nneg i32 %3452 to i64
  %3454 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3453
  %3455 = addrspacecast ptr %3454 to ptr addrspace(3)
  %3456 = load float, ptr addrspace(3) %3455, align 4, !tbaa !20, !alias.scope !18, !noalias !22
  %3457 = or disjoint i32 %3382, 135
  %3458 = zext nneg i32 %3457 to i64
  %3459 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %3458
  %3460 = addrspacecast ptr %3459 to ptr addrspace(3)
  %3461 = load float, ptr addrspace(3) %3460, align 4, !tbaa !20, !alias.scope !18, !noalias !22
  %3462 = fmul contract float %3387, 0xBFF7154760000000
  %3463 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %3462)
  %3464 = fadd contract float %3463, 1.000000e+00
  %3465 = tail call contract float @llvm.amdgcn.rcp.f32(float %3464)
  %3466 = fmul contract float %3387, %3465
  %3467 = fmul contract float %3391, %3466
  %3468 = fmul contract float %3396, 0xBFF7154760000000
  %3469 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %3468)
  %3470 = fadd contract float %3469, 1.000000e+00
  %3471 = tail call contract float @llvm.amdgcn.rcp.f32(float %3470)
  %3472 = fmul contract float %3396, %3471
  %3473 = fmul contract float %3401, %3472
  %3474 = fmul contract float %3406, 0xBFF7154760000000
  %3475 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %3474)
  %3476 = fadd contract float %3475, 1.000000e+00
  %3477 = tail call contract float @llvm.amdgcn.rcp.f32(float %3476)
  %3478 = fmul contract float %3406, %3477
  %3479 = fmul contract float %3411, %3478
  %3480 = fmul contract float %3416, 0xBFF7154760000000
  %3481 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %3480)
  %3482 = fadd contract float %3481, 1.000000e+00
  %3483 = tail call contract float @llvm.amdgcn.rcp.f32(float %3482)
  %3484 = fmul contract float %3416, %3483
  %3485 = fmul contract float %3421, %3484
  %3486 = fmul contract float %3426, 0xBFF7154760000000
  %3487 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %3486)
  %3488 = fadd contract float %3487, 1.000000e+00
  %3489 = tail call contract float @llvm.amdgcn.rcp.f32(float %3488)
  %3490 = fmul contract float %3426, %3489
  %3491 = fmul contract float %3431, %3490
  %3492 = fmul contract float %3436, 0xBFF7154760000000
  %3493 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %3492)
  %3494 = fadd contract float %3493, 1.000000e+00
  %3495 = tail call contract float @llvm.amdgcn.rcp.f32(float %3494)
  %3496 = fmul contract float %3436, %3495
  %3497 = fmul contract float %3441, %3496
  %3498 = fmul contract float %3446, 0xBFF7154760000000
  %3499 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %3498)
  %3500 = fadd contract float %3499, 1.000000e+00
  %3501 = tail call contract float @llvm.amdgcn.rcp.f32(float %3500)
  %3502 = fmul contract float %3446, %3501
  %3503 = fmul contract float %3451, %3502
  %3504 = fmul contract float %3456, 0xBFF7154760000000
  %3505 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %3504)
  %3506 = fadd contract float %3505, 1.000000e+00
  %3507 = tail call contract float @llvm.amdgcn.rcp.f32(float %3506)
  %3508 = fmul contract float %3456, %3507
  %3509 = fmul contract float %3461, %3508
  %3510 = tail call contract noundef float @llvm.fabs.f32(float %3467)
  %3511 = tail call contract noundef float @llvm.fabs.f32(float %3473)
  %3512 = tail call contract noundef float @llvm.maxnum.f32(float %3510, float %3511)
  %3513 = tail call contract noundef float @llvm.fabs.f32(float %3479)
  %3514 = tail call contract noundef float @llvm.maxnum.f32(float %3512, float %3513)
  %3515 = tail call contract noundef float @llvm.fabs.f32(float %3485)
  %3516 = tail call contract noundef float @llvm.maxnum.f32(float %3514, float %3515)
  %3517 = tail call contract noundef float @llvm.fabs.f32(float %3491)
  %3518 = tail call contract noundef float @llvm.maxnum.f32(float %3516, float %3517)
  %3519 = tail call contract noundef float @llvm.fabs.f32(float %3497)
  %3520 = tail call contract noundef float @llvm.maxnum.f32(float %3518, float %3519)
  %3521 = tail call contract noundef float @llvm.fabs.f32(float %3503)
  %3522 = tail call contract noundef float @llvm.maxnum.f32(float %3520, float %3521)
  %3523 = tail call contract noundef float @llvm.fabs.f32(float %3509)
  %3524 = tail call contract noundef float @llvm.maxnum.f32(float %3522, float %3523)
  %3525 = bitcast float %3524 to i32
  %3526 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %3525, i32 177, i32 15, i32 15, i1 true)
  %3527 = bitcast i32 %3526 to float
  %3528 = tail call contract noundef float @llvm.maxnum.f32(float %3524, float %3527)
  %3529 = bitcast float %3528 to i32
  %3530 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %3529, i32 78, i32 15, i32 15, i1 true)
  %3531 = bitcast i32 %3530 to float
  %3532 = tail call contract noundef float @llvm.maxnum.f32(float %3528, float %3531)
  %3533 = bitcast float %3532 to i32
  %3534 = add i32 %3533, 2097152
  %3535 = bitcast i32 %3534 to float
  %3536 = fmul contract float %3535, 2.500000e-01
  %3537 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 0, float %3467, float %3473, float %3536, i32 0)
  %3538 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %3537, float %3479, float %3485, float %3536, i32 1)
  %3539 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %3538, float %3491, float %3497, float %3536, i32 2)
  %3540 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %3539, float %3503, float %3509, float %3536, i32 3)
  %3541 = shl nsw i32 %41, 6
  %3542 = shl nuw nsw i32 %3376, 4
  %3543 = shl nuw nsw i32 %3377, 2
  %3544 = or disjoint i32 %45, %3374
  %3545 = shl nsw i32 %3544, 8
  %3546 = add i32 %3545, %3541
  %3547 = or disjoint i32 %3546, %3543
  %3548 = or disjoint i32 %3547, %3542
  %3549 = sext i32 %3548 to i64
  %3550 = getelementptr inbounds i8, ptr %36, i64 %3549
  store i32 %3540, ptr %3550, align 4, !tbaa !7, !alias.scope !13, !noalias !24, !nontemporal !25
  %3551 = icmp eq i32 %3377, 0
  br i1 %3551, label %3552, label %3569

3552:                                             ; preds = %32
  %3553 = bitcast float %3536 to i32
  %3554 = lshr i32 %3553, 23
  %3555 = tail call noundef range(i32 0, 255) i32 @llvm.umin.i32(i32 range(i32 0, 512) %3554, i32 254)
  %3556 = trunc nuw i32 %3555 to i8
  %3557 = shl nsw i32 %39, 7
  %3558 = shl nsw i32 %41, 5
  %3559 = and i32 %3558, -64
  %3560 = add nsw i32 %3559, %3557
  %3561 = or disjoint i32 %3560, %3374
  %3562 = or disjoint i32 %3561, %3542
  %3563 = shl nsw i32 %3562, 2
  %3564 = shl nsw i32 %41, 1
  %3565 = and i32 %3564, 2
  %3566 = or disjoint i32 %3563, %3565
  %3567 = sext i32 %3566 to i64
  %3568 = getelementptr inbounds i8, ptr %17, i64 %3567
  store i8 %3556, ptr %3568, align 1, !tbaa !12, !alias.scope !16, !noalias !26
  br label %3569

3569:                                             ; preds = %3552, %32, %11
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #1

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p0(ptr readnone, i16, i32, i32) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) nocapture readonly, i32, i32, i32 immarg) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32, <2 x bfloat>, float, i32 immarg) #5

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.update.dpp.i32(i32, i32, i32 immarg, i32 immarg, i32 immarg, i1 immarg) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) nocapture readonly, i32, i32, i32 immarg) #4

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #6

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.amdgcn.s.setprio(i16 immarg) #7

; Function Attrs: convergent mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32>, <4 x i32>, <4 x float>, i32 immarg, i32 immarg, i32 immarg, i32, i32 immarg, i32) #8

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #6

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32, float, float, float, i32 immarg) #5

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
declare i16 @llvm.umax.i16(i16, i16) #9

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umax.i32(i32, i32) #9

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umin.i32(i32, i32) #9

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #10

attributes #0 = { convergent mustprogress norecurse nounwind "amdgpu-flat-work-group-size"="1,256" "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-waves-per-eu"="3" "denormal-fp-math-f32"="preserve-sign,preserve-sign" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx950" "target-features"="+16-bit-insts,+ashr-pk-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-buffer-pk-add-bf16-inst,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-global-pk-add-bf16-inst,+bf8-cvt-scale-insts,+bitop3-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot12-insts,+dot13-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+f16bf16-to-fp6bf6-cvt-scale-insts,+f32-to-f16bf16-cvt-sr-insts,+fp4-cvt-scale-insts,+fp6bf6-cvt-scale-insts,+fp8-conversion-insts,+fp8-cvt-scale-insts,+fp8-insts,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+gfx940-insts,+gfx950-insts,+mai-insts,+permlane16-swap,+permlane32-swap,+prng-inst,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" "uniform-work-group-size"="false" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #2 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #3 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #5 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #6 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #7 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #8 = { convergent mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #9 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #10 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #11 = { convergent nounwind memory(none) }
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
!11 = !{i64 10445668}
!12 = !{!9, !9, i64 0}
!13 = !{!14}
!14 = distinct !{!14, !15, !"_ZN5aiter9mxfp4_moe11gemm_common27apply_cshuffle_quant_epilogILi1024ELi16EEEvRAdvT0_Li16E_A4_KDv4_fPhS8_iiiiiiiPf: argument 0"}
!15 = distinct !{!15, !"_ZN5aiter9mxfp4_moe11gemm_common27apply_cshuffle_quant_epilogILi1024ELi16EEEvRAdvT0_Li16E_A4_KDv4_fPhS8_iiiiiiiPf"}
!16 = !{!17}
!17 = distinct !{!17, !15, !"_ZN5aiter9mxfp4_moe11gemm_common27apply_cshuffle_quant_epilogILi1024ELi16EEEvRAdvT0_Li16E_A4_KDv4_fPhS8_iiiiiiiPf: argument 1"}
!18 = !{!19}
!19 = distinct !{!19, !15, !"_ZN5aiter9mxfp4_moe11gemm_common27apply_cshuffle_quant_epilogILi1024ELi16EEEvRAdvT0_Li16E_A4_KDv4_fPhS8_iiiiiiiPf: argument 2"}
!20 = !{!21, !21, i64 0}
!21 = !{!"float", !9, i64 0}
!22 = !{!14, !17}
!23 = !{!14, !17, !19}
!24 = !{!17, !19}
!25 = !{i32 1}
!26 = !{!14, !19}
