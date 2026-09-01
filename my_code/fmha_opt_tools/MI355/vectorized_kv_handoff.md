# vectorized KV cache layout 支持 —— Session 交接文档

<!-- markdown-toc-generator:start -->
## Table of Contents

- [任务目标](#任务目标)
- [环境 / 运行方式(重要)](#sec-环境-运行方式重要)
- [vectorized 物理布局(kVS = 16/elem_size = 8 for bf16/f16)](#sec-vectorized-物理布局kvs-16elem_size-8-for-bf16f16)
- [kernel 数据流关键事实(已逆向确认)](#sec-kernel-数据流关键事实已逆向确认)
- [已完成的改动(阶段1,功能正确)](#sec-已完成的改动阶段1功能正确)
  - [kernels/flash_attn_interface.py](#sec-kernelsflash_attn_interface-py)
  - [kernels/flash_attn_gfx950.py](#sec-kernelsflash_attn_gfx950-py)
  - [tests/kernels/test_flash_attn_fwd.py](#sec-testskernelstest_flash_attn_fwd-py)
- [阶段1 验证结果(全部 PASS)](#sec-阶段1-验证结果全部-pass)
- [阶段2 已尝试 & 结论(重要,避免重走弯路)](#sec-阶段2-已尝试-结论重要避免重走弯路)
- [阶段2 正确方向(下一步要做的)](#sec-阶段2-正确方向下一步要做的)
- [阶段2 实测：V 寄存器 contract（已用 dump 逆向确认,2026-06-24）](#sec-阶段2-实测v-寄存器-contract已用-dump-逆向确认2026-06-24)
  - [实测结果（2026-06-24,coalesced DMA + plain 2×b64 read）](#sec-实测结果2026-06-24coalesced-dma-plain-2b64-read)
  - [下一步（关闭最后 10–19%）：改用 ds_read_b64_tr_b16](#下一步关闭最后-1019改用-ds_read_b64_tr_b16)
- [阶段2 K coalesce（2026-06-24,已落地正确,但净 perf 无提升 → 印证瓶颈在 read conflict）](#sec-阶段2-k-coalesce2026-06-24已落地正确但净-perf-无提升-印证瓶颈在-read-conflict)
  - [ATT 逐指令 cycle 实测（2026-06-24,决定性,推翻"K 也有问题"的假设）](#sec-att-逐指令-cycle-实测2026-06-24决定性推翻k-也有问题的假设)
  - [ISA 修正（2026-06-24,用户指出 + CDNA4 ISA §7.1.4.1 核实）——改 V read 的根本方案](#sec-isa-修正2026-06-24用户指出-cdna4-isa-7-1-4-1-核实改-v-read-的根本方案)
  - [实测:permute P + V ds_read_b128 已落地正确,但 perf 退化(2026-06-24)](#sec-实测permute-p-v-ds_read_b128-已落地正确但-perf-退化2026-06-24)
  - [trace_segment_cycles.py 精确 cycle 对比(permute+b128 版,2026-06-24)](#sec-trace_segment_cycles-py-精确-cycle-对比permuteb128-版2026-06-24)
  - [地址寄存器 spill 修复(2026-06-24,关键)](#sec-地址寄存器-spill-修复2026-06-24关键)
  - [spill 修复后 cycle 对比 → 暴露 bank conflict(2026-06-24)](#sec-spill-修复后-cycle-对比-暴露-bank-conflict2026-06-24)
  - [加 LDS pad 消 V bank conflict 失败:buffer_load_lds 要求 wave 内连续(2026-06-24)](#sec-加-lds-pad-消-v-bank-conflict-失败buffer_load_lds-要求-wave-内连续2026-06-24)
  - [dense-style 行边界 pad 成功(2026-06-24,用户指点:抄 dense pad)](#sec-dense-style-行边界-pad-成功2026-06-24用户指点抄-dense-pad)
  - [pad 后 ATT cycle 复测(2026-06-24)](#sec-pad-后-att-cycle-复测2026-06-24)
  - [no-major 行内布局消 V bank conflict 成功(2026-06-24,用户给出 gfx950 b128 no-conflict 规则)](#sec-no-major-行内布局消-v-bank-conflict-成功2026-06-24用户给出-gfx950-b128-no-conflict-规则)
  - [no-major 后 ATT cycle 复测(2026-06-24)](#sec-no-major-后-att-cycle-复测2026-06-24)
  - [K-read n 置换去掉 P permute(2026-06-24,用户提案,成功)](#sec-k-read-n-置换去掉-p-permute2026-06-24用户提案成功)
  - [去 P permute 后 spill 回归(2026-06-24,ATT 复测发现)](#sec-去-p-permute-后-spill-回归2026-06-24att-复测发现)
  - [VGPR 用量定位(ISA 21_final_isa.s 的 num_vgpr/scratch,2026-06-24)](#sec-vgpr-用量定位isa-21_final_isa-s-的-num_vgprscratch2026-06-24)
  - [(作废)旧下一步:V 换 ds_read_b64_tr_b16](#sec-作废旧下一步v-换-ds_read_b64_tr_b16)
  - [V read 的 v_or 地址 + spill 根因(2026-06-25,实测排除两个错误假设)](#sec-v-read-的-v_or-地址-spill-根因2026-06-25实测排除两个错误假设)
  - [spill 根因隔离 + 修复:K σ 从 read 移到 DMA(2026-06-25,关键突破)](#sec-spill-根因隔离-修复k-σ-从-read-移到-dma2026-06-25关键突破)
- [当前代码状态(2026-06-25)](#sec-当前代码状态2026-06-25)
  - [VGPR spill 彻底消除:epilogue O-store 地址拆 base+const(2026-06-25,不改 LLVM)](#sec-vgpr-spill-彻底消除epilogue-o-store-地址拆-baseconst2026-06-25不改-llvm)
  - [最终状态(2026-06-25)](#sec-最终状态2026-06-25)
  - [V-DMA prep 上移(2026-06-25,隐藏 page_id ds_read 延迟)](#sec-v-dma-prep-上移2026-06-25隐藏-page_id-ds_read-延迟)
  - [K-DMA prep 上移 + prologue tile2 prep 上移(2026-06-25)](#sec-k-dma-prep-上移-prologue-tile2-prep-上移2026-06-25)
  - [Epilogue 阶段 prep 上移(2026-06-25)](#sec-epilogue-阶段-prep-上移2026-06-25)
  - [Epilogue C0 跨 loop-exit prep 上移(2026-06-25,完成)](#sec-epilogue-c0-跨-loop-exit-prep-上移2026-06-25完成)

<!-- markdown-toc-generator:end -->

> 新 session 请先完整读这份文档,再继续阶段2/3。本任务跨 session,当前**阶段1已完成
> (功能正确),阶段2(性能优化)进行中**。

## 任务目标

修改 `kernels/flash_attn_gfx950.py`,让 paged-KV 的 `kv_cache_layout` 支持 `vectorized`
(aiter 风格),且 **splitk / varlen / cross 全部支持**。

硬性要求:
1. set1(`tests/kernels/test_flash_attn_fwd.py` 第 56-59 行)、set3(第 85-89 行)规模,
   用 `run.sh` 第 64-65 行参数跑性能测试 **不回退**。
2. 上述规模,`--kv-cache-layout vectorized` 与 `--kv-cache-layout linear` **性能一致(<2%)**。

用户已确认的子策略:**阶段1先让 vectorized 端到端 compare 通过(V 可暂时慢),
阶段2/3 再优化到 perf 一致**。

<a id="sec-环境-运行方式重要"></a>
## 环境 / 运行方式(重要)

- 本地代码:`C:\Users\yanguahe\code\wk_sp1\FlyDSL\`
- 远端机器:`mi355-gpu-34`,经跳板机 `aac16.amd.com` 两跳 ssh。
- 远端代码目录:`/shared/amdgpu/home/zhiming_ding_qle/yanguahe/code/wk_sp1/FlyDSL`
- 测试在容器 `hyg_fyd1` 内跑,`HIP_VISIBLE_DEVICES=1`,每次清 JIT cache:
  `rm -rf /root/.flydsl/cache/*`
- ssh key:`~/.ssh/id_rsa_zhiming_aac`
- 上传文件命令模板(注意 CRLF→LF):
  ```bash
  KEY=~/.ssh/id_rsa_zhiming_aac
  RB="/shared/amdgpu/home/zhiming_ding_qle/yanguahe/code/wk_sp1/FlyDSL"
  L=/c/Users/yanguahe/code/wk_sp1/FlyDSL
  ssh -i "$KEY" -o StrictHostKeyChecking=no zhiming_ding_qle@aac16.amd.com \
    "ssh -o StrictHostKeyChecking=no mi355-gpu-34 'cat > $RB/kernels/flash_attn_gfx950.py && sed -i \"s/\r\$//\" $RB/kernels/flash_attn_gfx950.py'" < "$L/kernels/flash_attn_gfx950.py"
  ```
- **快速正确性探针**(已在远端 `.../wk_sp1/vec_probe.py`,B=1/S=512/H=4 vectorized,
  打印 `VEC max_err`,几十秒一轮,迭代用它最快):
  ```bash
  ssh ... mi355-gpu-34 'docker exec hyg_fyd1 bash -lc "rm -rf /root/.flydsl/cache/*; cd .../wk_sp1 && HIP_VISIBLE_DEVICES=1 python3 vec_probe.py 2>&1 | tail -3"'
  ```
- 完整测试:
  ```
  # set1+set3 vectorized compare:
  python3 tests/kernels/test_flash_attn_fwd.py --causal --dtype bf16 --iters 30 --compare --block-table --page-size 64 --kv-cache-layout vectorized
  # varlen/cross 加 --extra
  # linear 基线: 去掉 --kv-cache-layout vectorized (默认 linear)
  ```
  过滤:`grep -E "vectorized|vllm:linear" | grep -v "\.\.\."`
- 改动**未 commit**,等用户批准。worklog 在 `FlyDSL/fmha_opt_tools/WORKLOG.md`,
  每完成一阶段追加。

<a id="sec-vectorized-物理布局kvs-16elem_size-8-for-bf16f16"></a>
## vectorized 物理布局(kVS = 16/elem_size = 8 for bf16/f16)

linear(已支持):`K/V = [NumBlocks, PageSize=64, Hkv, D]`,within-page offset(n,h,d) =
`n*(Hkv*D) + h*D + d`,最内连续轴 = d。

vectorized(aiter 5D):
- **K** `[NumBlocks, Hkv, D/kVS, PageSize, kVS]`:
  offset(n,h,d) = `h*(D/8)*64*8 + (d//8)*64*8 + n*8 + (d%8)`。最内连续 8 = 8 个连续 d(d%8)。
- **V** `[NumBlocks, Hkv, PageSize/kVS, D, kVS]`:
  offset(n,h,d) = `h*(64/8)*D*8 + (n//8)*D*8 + d*8 + (n%8)`。最内连续 8 = **8 个连续 token(n%8)**
  —— 即 V 在 global 里已经是转置(d-major, n-inner)。

测试侧构造(`test_flash_attn_fwd.py` `_vectorize_paged_kv`):
```python
kvec = kc.view(tp,PS,HKV,D//kVS,kVS).permute(0,2,3,1,4).contiguous()  # K
vvec = vc.view(tp,PS//kVS,kVS,HKV,D).permute(0,3,1,4,2).contiguous()  # V
```

<a id="sec-kernel-数据流关键事实已逆向确认"></a>
## kernel 数据流关键事实(已逆向确认)

- **DMA**:每 lane 一个 `buffer_load_dwordx4`(128-bit = 8 bf16),从 global 读 8 连续 bf16
  直达 LDS。原语 `_buffer_load_lds_128`(约 546 行)。
- **K LDS→VGPR**:`_async_load_k_from_lds_to_vgpr`(约 970 行)用**普通 llvm.LoadOp**(v8f16),
  无转置。QK^T 的 A 算子(K)转置是靠 linear LDS 写布局 + `urk_base_per_lane`(约 634 行)实现。
- **V LDS→VGPR**:`_read_v_packs_for_buf`(约 1015 行)用 **`_ds_read_tr_v4f16_imm`**(转置读,
  ds_read_b64_tr_b16)。因为 linear V 在 LDS 里是 `[n,d]`(d 连续),P·V 的 A 算子(V)需要
  N 作 contraction(N-contiguous),所以读时转置。
- **P·V MFMA**:`_mma1_step_k`(约 1180 行):`_mfma_acc(v_pk[dc], p_pk, v_o[dc])`,
  A=V。MFMA(32,32,16)。`v_v[step][dc]` 是 v4f16,step∈[0,4),dc∈[0,D_CHUNKS=4)。
  按 ISA:`DS_READ_B64_TR_B16` "每 lane 持有 4 个连续的 M 或 N 值"。
- **LDS 容量**:已用 68096 B(`LDS_KV_TOTAL_SIZE`),接近上限,**没有空间加 scratch LDS**。

关键常量(约 170-210 行):BLOCK_N=64, HEAD_DIM=128, VEC_KV=8, D_128B_SIZE=64,
NUM_WAVES=8, WARP_SIZE=64, SMEM_V_LINE_STRIDE=544, SMEM_N_RPT=8, SMEM_D_RPT=2,
NUM_DMA_V=SMEM_D_RPT=2。lane 分解:`n_in_warp = lane%64//8`,`d_bucket = lane%8`(约 568 行)。
urv 常量约 200-210 行(DUALWAVE_SWP_URV_*)。

<a id="sec-已完成的改动阶段1功能正确"></a>
## 已完成的改动(阶段1,功能正确)

三个文件,均**未 commit**:

<a id="sec-kernelsflash_attn_interface-py"></a>
### `kernels/flash_attn_interface.py`
- `_flydsl_flash_attn_paged`:去掉 `kv_cache_layout != "linear"` 的 raise;加 vectorized 分支:
  5D shape 校验,`vectorized=True` 时 `page_size=k.shape[3]`、`Hkv=k.shape[1]`、
  `k_head_dim=k.shape[2]*k.shape[4]`、校验 `k.shape[4]==kVS`。
- `_build_paged`:加 `kv_cache_layout` 参数,透传给 builder。launch 处传 `kv_cache_layout=kv_cache_layout`。

<a id="sec-kernelsflash_attn_gfx950-py"></a>
### `kernels/flash_attn_gfx950.py`
- builder 签名加 `kv_cache_layout="linear"`。
- 约 247 行加:
  ```python
  KV_VECTORIZED = paged and (kv_cache_layout == "vectorized")
  KV_VEC_SIZE = 16 // BF16_BYTES  # 8
  # + 校验 layout 合法、HEAD_DIM/PageSize 可被 kVS 整除
  ```
- **K**(`_async_load_k`,约 920 行):`const_expr(KV_VECTORIZED)` 分支,src_elem 换成 vectorized
  公式(LDS 写入字节与 linear 一致,下游零改动):
  ```python
  d_group = d_bucket + d * (D_128B_SIZE // VEC_KV)   # = (d_bucket*8 + d*64)//8
  src_elem = kv_head_idx*(HEAD_DIM//KV_VEC_SIZE)*BLOCK_N*KV_VEC_SIZE \
           + d_group*BLOCK_N*KV_VEC_SIZE + n_in_tile*KV_VEC_SIZE
  ```
- **V**(PAGED 分支约 495 行加辅助;`_async_load_v` 约 958 行加 vectorized 分支):
  - 辅助(在 `_make_page_view` 后):`_make_v_page_rsrc(page_id)`(raw V page rsrc)+
    `_vec_v_elem(n,d)`(within-page vectorized V offset =
    `kv_head_idx*(BLOCK_N//kVS)*D*kVS + (n//kVS)*D*kVS + d*kVS + (n%kVS)`)。
  - `_async_load_v` 的 `const_expr(KV_VECTORIZED)` 分支:**gather** —— 每 lane 对自己的
    (n_in_tile, d-octet) 逐个 buffer_load 8 个 d,拼成 octet 用 llvm.StoreOp 写到 linear-V 的
    **同一 LDS 槽位**(byte-identical → `ds_read_tr` 不变)。LDS 写地址:
    `lds_byte = lds_addr + lane_in_warp*16`,用 `get_element_ptr(lds_kv_base_ptr, ...)`。

<a id="sec-testskernelstest_flash_attn_fwd-py"></a>
### `tests/kernels/test_flash_attn_fwd.py`
- 之前已加 varlen+paged 的 cu_seqlens 传递(上个任务)。本任务测试基建已支持 vectorized
  (`_vectorize_paged_kv` 等),无需改测试逻辑。

<a id="sec-阶段1-验证结果全部-pass"></a>
## 阶段1 验证结果(全部 PASS)

vectorized compare MaxErr ≤ 3.91e-3:set1 + set3(splitk) + varlen + cross 全过;
linear/dense 无回退。

**性能(问题所在)**:V gather 慢 ~2-3×:
- 16/8192/64/64: vectorized **408** vs linear ~1100 TFLOPS
- 16/8192/64/8: 425 vs ~1200
- 2/1024: 217 vs ~595

<a id="sec-阶段2-已尝试-结论重要避免重走弯路"></a>
## 阶段2 已尝试 & 结论(重要,避免重走弯路)

试过 2 个 V 实现,都正确但都不够快:
1. **gather**(当前代码状态):408 TFLOPS。
2. **coalesced load(1次读8 token)+ LDS 单元素 scatter(8次)**:**更慢 160 TFLOPS**
   —— 单元素 LDS scatter 开销爆炸。**此路不通,别再试。**

**根本结论**:V 的转置无法用"linear 字节一致 LDS"高效做(global 跨步=gather 慢,
LDS 跨步=scatter 更慢)。

<a id="sec-阶段2-正确方向下一步要做的"></a>
## 阶段2 正确方向(下一步要做的)

**新 LDS 布局 + 全新 plain ds_read**(用户最初的思路,已确认是唯一能到 <2% 的路):
1. vectorized V 的 `[8token, D, 8]` 连续块用**普通 coalesced DMA**(`_buffer_load_lds_128`,
   相邻 lane 读相邻 token)直接写进**新的 V-LDS 布局 `[d, n]`**(n 连续)。不用 gather/scatter。
2. 改 `_read_v_packs_for_buf`:vectorized 下用**普通 ds_read(b64,非 tr)**,因为新布局里
   N(n%8)已连续 —— 正是 P·V MFMA 想要的 N-contiguous,省掉转置。
3. 全程 `const_expr(KV_VECTORIZED)` 门控,linear 路径逐字节不变。

**关键难点**:新 LDS 偏移 + plain-read 偏移必须精确匹配现有 `_read_v_packs_for_buf` 的
`(lane, dc, k_substep, element)→(n, d)` 逻辑映射。这个映射是 `ds_read_tr` + `urv_base_per_lane`
+ urv 常量编码的,**建议用经验法逆向**:让 linear 路径加载一个 tagged V(如 V[n][d] 编码成
可辨识值),读回 v_pk 寄存器,dump 出每个 (lane,dc,k_substep,e) 实际持有的 (n,d),据此写新读。
纯静态推导易错。每轮 compile+probe 约 1-2 分钟,K 已不用动,只攻 V。

**验证顺序**:先用 vec_probe.py 调正确(MaxErr≤3.91e-3),再 set1/set3 benchmark 调到
vectorized vs linear <2%,最后 varlen/cross/splitk 全回归 + linear/dense 不回退。

<a id="sec-阶段2-实测v-寄存器-contract已用-dump-逆向确认2026-06-24"></a>
## 阶段2 实测：V 寄存器 contract（已用 dump 逆向确认,2026-06-24）

用 `dump_v_mapping=True`（kernel 新增 debug flag）+ tagged-V 两遍（V[n][d]=n / =d）dump 出
linear 路径 `v_v[ks][dc][e]` 每个寄存器实际持有的 (n,d)。完美双射（8192 cell）。**contract:**
```
n = ks*16 + (e//4)*8 + (e%4) + (lane//32)*4    # token in tile 0..63
d = dc*32 + (lane%32)                            # head dim 0..127
```
(ks=k_substep 0..3, dc 0..3, e 0..7。dump 工具 `wk_sp1/dump_v_map.py`，kernel 内 dump block 在
`_split_guard` 顶部 `const_expr(DUMP_V_MAPPING)`。)

**关键观察（global 已是转置,利于我们）**：vectorized V within-page offset =
`(n//8)*D*8 + d*8 + (n%8)`。固定 n-octet `no`,`(d, n%8)` 是一段连续 `[D, 8]` 块（d-major,
n-inner）—— 即 global 里 V 已经是 `[d][n_inner]`。所以**一次 coalesced `buffer_load_lds_128`
就能把 8 个连续 token(同 d)直送 LDS,无需 gather**。新 LDS 只要镜像 global 的 `[no][d][n%8]`
即可（lane L 的 octet 落 `base+L*16B`,全 coalesced）。

**已选方案：新 [no][d][ni] LDS 布局 + plain 2×b64 read（纸上已推导验证,lane 0/32 对过）**

把 contract 的 n 拆成 no=n//8, ni=n%8:
- no = ks*2 + (e//4)        ni = (lane//32)*4 + (e%4)        d = dc*32 + (lane%32)

新 LDS 布局（每个 V buffer 内,8192 元素,无 pad,镜像 global）:
- `LDS[A] = V[n][d]`,A = no*1024 + d*8 + ni  (no=A//1024, d=(A%1024)//8, ni=A%8)

**coalesced DMA**（替代 gather,`_async_load_v` vec 分支）:每 lane 一条 buffer_load_lds_128,读
global vec V 的连续 8 token(ni 0..7,同 (no,d))直送 LDS slot L:
- d_iter∈[0,2): grp=wave_id_uni*2+d_iter; lds_addr=v_lds_byte_base+grp*512*2B(硬件再+lane*16B)
- oi=grp*64+lane_in_warp; no=oi//128; d=oi%128
- src_elem=head + no*D*8 + d*8 (= `_vec_v_elem(no*8, d)`,ni=0 起)；src_div 复用
  `_make_page_view(_v_iter,...)`（vec page 与 linear page 同 byte stride,flat 索引）

**plain read**（`_read_v_packs_for_buf` vec 分支,2 次 b64 LoadOp + shuffle）:
- addr_a = v_base + (ks*2)*1024 + (dc*32 + lane%32)*8 + (lane//32)*4   (元素)
- a=LoadOp(v4f16, addr_a) → e0..3; b=LoadOp(v4f16, addr_a+1024) → e4..7
- packs[ks][dc]=shuffle(a,b,[0..7]) —— 与 linear 同返回形

**已知隐忧（待实测）**：plain b64 在此布局下,同 wave 内 lane{0,8,16,24}(及+32) 命中同 bank →
~4-way bank conflict。若 perf 不达 <2%,改用 ds_read_b64_tr_b16(免 conflict)再 dump 求 offset。
gather 基线 408 TFLOPS,plain(即便 4-way)预期远好于 gather,先验证正确+测 perf 再决定。

<a id="sec-实测结果2026-06-24coalesced-dma-plain-2b64-read"></a>
### 实测结果（2026-06-24,coalesced DMA + plain 2×b64 read）
- **正确性**：vec_probe MaxErr=3.91e-3 ✅；set1/set3 vectorized compare 全 PASS MaxErr=3.91e-3。
- **代码已落地**：`_async_load_v` 的 `KV_VECTORIZED` 分支换成 coalesced `_buffer_load_lds_128`
  (no=d*(NUM_WAVES//2)+wave//2, d_col=(wave%2)*64+lane, src_elem=`_vec_v_elem(no*8,d_col)`)；
  `_read_v_packs_for_buf` 的 `KV_VECTORIZED` 分支用 plain 2×v4f16 LoadOp
  (a_rel=(ks//2)*4352+((ks%2)*4+dc//2)*544+((dc%2)*32+lane%32)*8+(lane//32)*4, b=a+1088)。
  纸上 + probe 双重确认正确。`_make_v_page_rsrc`/`_vec_v_elem` gather 辅助已不再被 V 用（K 仍用？
  no—K 不用 _vec_v_elem。可清理但留着无害）。
- **性能（vec col1 best vs linear col1 best,FlyDSL kernel）**：仍差 10–19%,未达 <2%：
  | 规模 | vec | linear | 差 |
  |---|---|---|---|
  | 16/8192/64/64 | 975 | 1101 | −11.4% |
  | 16/8192/64/8 | 1084 | 1204 | −10.0% |
  | 2/1024/64/64 | 487 | 599 | −18.7% |
  | 1/8192/2/2 sk4 | 510 | 590 | −13.5% |
  | 1/8192/4/4 sk2 | 572 | 676 | −15.5% |
  (对比 gather 旧版 408 vs 1100 ≈ −63%,本次已大幅改善,但 plain read 的 4-way bank conflict 仍在。)

### 下一步（关闭最后 10–19%）：改用 ds_read_b64_tr_b16
plain b64 的 ~4-way bank conflict 是剩余 gap 主因(已确认:固定 (ks,dc) 时 lane%32 沿 d 轴
stride=8 元素=4 bank,32 lane 落 8 个 bank-pair)。linear 路径正是用硬件 transpose read
`ds_read_b64_tr_b16` 免 conflict。方案:把 vec read 换成 ds_read_tr,在新 [no][d][ni] 布局上
重新 dump 求 (urv_base + imm) offset 使其产出同一 contract。
**dump 工具已就位**(DUMP_V_MAPPING flag + dump_v_map.py):把 vec read 临时改成带某 offset 的
ds_read_tr,dump 出它实际产出的 (n,d),对照 contract 解偏移即可。每轮 compile+probe ~1-2 min。
注意:新 LDS 布局 stride 是 544(SMEM_V_LINE_STRIDE,带 pad),不是理想的 1024。

<a id="sec-阶段2-k-coalesce2026-06-24已落地正确但净-perf-无提升-印证瓶颈在-read-conflict"></a>
## 阶段2 K coalesce（2026-06-24,已落地正确,但净 perf 无提升 → 印证瓶颈在 read conflict）

**K 寄存器 contract（dump 逆向,工具 dump_k_map.py + DUMP_K_MAPPING flag,8192 双射）**：
`k_lo[ks]`=half0 / `k_hi[ks]`=half1, e=0..7:
```
n = half*32 + (lane % 32)
d = ks*16 + (lane//32)*8 + e        # e → 8 个连续 d（关键:e 是连续 d,不是 token）
```
逐 lane(0,1,2,16,32)验证全吻合。

**与 V 的关键差异**：K 的 e→连续 d,而 vectorized global K 最内轴正是 d%8 连续 →
coalesced DMA 逐字拷成 native `[d//8, n, d%8]` LDS 后,读端 **plain v8f16 LoadOp 直接读出
contract,连 shuffle 都不用**(比 V 还简单)。

**已落地代码**（均 KV_VECTORIZED 门控,linear 字节不变）:
- `_async_load_k` vec 分支:coalesced `_buffer_load_lds_128`,oct_idx=wave*128+d*64+lane,
  src_elem=kv_head*8192+oct_idx*8,lds=k_lds_byte_base+oct_idx*16B。native LDS A=(d//8)*512+n*8+(d%8)。
- `_async_load_k_from_lds_to_vgpr` vec 分支:plain v8f16 LoadOp,
  idx_lo=k_base+(ks*2+lane//32)*512+(lane%32)*8, idx_hi=idx_lo+256(=URK_N_STRIP_STRIDE)。

**正确性**:vec_probe MaxErr=3.91e-3 ✅。
**性能（+K coalesce 后,vec vs linear best）**:16/8192/64/64=970 vs 1101、16/8192/64/8=1047 vs
1204、2/1024=467 vs 599。**和只改 V 时几乎一样(噪声内),净零提升。**

<a id="sec-att-逐指令-cycle-实测2026-06-24决定性推翻k-也有问题的假设"></a>
### ATT 逐指令 cycle 实测（2026-06-24,决定性,推翻"K 也有问题"的假设）
方法:对 vec/lin 两个 paged 配置(B1/S8192/H64)各 dump ATT thread-trace(`att_target_cu:1`,
默认 waves_per_eu,**不设 waves_per_eu=1**),用 `trace_segment_cycles.py` compare 模式锚 K
ds_read 区间和 V ds_read 区间(各含尾部 s_waitcnt+s_barrier),量 interval cycle(=end_ts−start_ts,
含 waitcnt stall)。装了 librocprof-trace-decoder 0.1.6。脚本:`att_vec_lin.sh`、`kv_seg.json`、
`att_inspect.py`、`seg_ctx.py`(均本地+远端 wk_sp1/)。

| 区间 | lin avg cyc | vec avg cyc | 差 | inst 差 |
|---|---|---|---|---|
| **V ds_read** | 359.6 | **769.4** | **+114%** | +11% (36→40) |
| **K ds_read** | 222.8 | 217.6 | **−2.3%** | 0% |
| between V&K | 1032 | 1310 | +27% | +3.5% |

**决定性结论(改方向)**:
1. **K ds_read 不是瓶颈**(−2.3%,vec 还略快)。我之前"K plain-read 4-way bank conflict"的假设
   **被实测推翻**——K coalesce 改动对 cycle 无害无益,印证 perf 净零。**K coalesce 可保留,但
   它不解决问题,也不必为它纠结 bank conflict。**
2. **V ds_read 是唯一真瓶颈**:vec 769 vs lin 359 = **+114%**,cycle 翻倍但 inst 只 +11%
   → **stall-bound**(cycle Δ% ≫ inst Δ%),正是 ds_read cycle(含 waitcnt)变长。vec 的 V plain
   read 被编译成 `ds_read2st64_b64` + 一堆 `ds_read_b64`(40 条),lin 用 32 条 `ds_read_b64_tr_b16`。
3. between V&K +27% 部分是 V 的 stall 外溢(V 的 lgkmcnt 等待挪到了后面),与 V 同源。

<a id="sec-isa-修正2026-06-24用户指出-cdna4-isa-7-1-4-1-核实改-v-read-的根本方案"></a>
### ISA 修正（2026-06-24,用户指出 + CDNA4 ISA §7.1.4.1 核实）——改 V read 的根本方案
**之前"V quarter-interleave 是 MFMA 固有"是错的。** CDNA4 ISA input-layout 公式:
`V_MFMA_F32_32X32X16_BF16`: M=N=32,K=16,B=1 → `K_L = K/(64/(M*B)) = 16/2 = 8`
→ **每个 lane 在 contraction(K=n) 维持有 8 个 *连续* 的 n**(item=k%K_L=n%8, lane=j+N*(k/K_L))。
MFMA 要 A(V) 和 B(P) 的 K 维(=n) **逐 lane 逐 item 完全一致**。

**当前 P 的 n 布局 = 我 dump 的 V contract**(因为 MFMA 强制 A、B 的 K 维一致):
固定 ks, 每 lane 持有 n%16 = {(lane//32)*4 + {0,1,2,3, 8,9,10,11}}——**隔 4 的两段,不是 8 连续**。
这不是 MFMA 要求,而是 **QK^T 的 MFMA *output* layout** 天然形态(ISA §7.1.4.2 H/B_I/M_I/G)。
linear V 用 ds_read_tr 把 n-连续的 LDS transpose 成匹配这个 interleaved P → 才慢。

**根本解(用户定方向:QK^T 后显式 permute P)**:把 P 从 interleaved n 重排成 **8 连续 n**,则
vectorized V(本就是 n 连续)用最简单 **ds_read_b128**(8 连续直读,无 transpose,预期 ~217 cyc
≈ K)即可对上。需要:
1. QK^T(_mma0)之后、P·V(_mma1)之前,对 P 做跨 lane permute:当前 lane0 持 {0,1,2,3,8,9,10,11},
   lane32 持 {4,5,6,7,12,13,14,15};目标 lane0={0..7}, lane32={8..15} → lane0/lane32 交换各自后半
   (4 个元素),用 permlane32/ds_permute。**每 tile 一次,跨 lane。**
2. V read 改 plain ds_read_b128(8 连续 n)。V-LDS 布局相应让 8 连续 n 在 LDS 连续。
3. v_o(输出 D)的 layout 不变(C/D 用 output layout,与 A/B 的 input layout 独立)。

**风险/待验证**:permute P 也是每 tile 跨 lane 操作,要确认它比 V 的 ds_read_tr(省下 ~410cyc/tile)
更便宜,即净赚。permlane32_swap 已在用(_reduction_pair/_swap_halves),成本可控。
**下一步**:先 dump 出当前 P 精确 n 布局(确认 = V contract)+ 定义目标 8-连续布局,再写 permute,
vec_probe 验正确,ATT 复测 V read cycle(769→目标~217),最后 set1/set3 perf。

<a id="sec-实测permute-p-v-ds_read_b128-已落地正确但-perf-退化2026-06-24"></a>
### 实测:permute P + V ds_read_b128 已落地正确,但 perf 退化(2026-06-24)
- **正确**:vec_probe MaxErr=3.91e-3 ✅。permlane32_swap API 关键坑:它**不是对称交换**,
  `permlane32_swap(x,x)` 取 partner lane 的该 dword = low lane 取 result[1]、high lane 取 result[0]
  (见 `_swap_halves` 注释,ISA: Rows2,3 of SRC0 ↔ Rows0,1 of VDST)。第一版当成对称换→MaxErr 0.996,
  改成 `_partner(dwx)` 逐 dword 取 partner 才对。
- **代码**:`_permute_p_pack_8n` 在 `_cast_p` vec 分支,把 interleaved-n P pack 重排成 8 连续 n;
  V DMA 改 `[d][n]` 布局(LDS[A]=V[n][d],A=d*64+n,逐字 coalesced 拷),V read 改单条 ds_read_b128。
- **ATT 实测**:V read 确实变干净(16 条 ds_read_b128,与 K 同),但 permute 引入 **128 条
  v_permlane32_swap_b32_e32**(每主循环 ~16 条**连续串行**,idx 875-952),不在 mem-wait 窗口隐藏。
- **perf 退化**:vec col1 best 16/8192/64/64 = **557**(plain-read 版是 975,linear 1101)。
  **permute 的 16 条串行 permlane32 开销 > V read 省下的,净亏。**
- **根因**:我的 permute 每 pack 调 `_partner` 4 次(4 dword 各一次 swap)=16 swap/tile,浪费
  (permlane32 一次能换 2 dword,本可减半)。但即便减半,16→8 条串行 permlane 仍可能不划算。
- **下一步候选**:
  1. 优化 permute:每 pack 2 次 swap(dw0↔dw2 配对、dw1↔dw3),减半;看能否转正。
  2. 彻底避免 permute:让 V-LDS 布局直接按 interleaved-n 摆放,ds_read_b128 读出 {0,1,2,3,8,9,10,11},
     P 不动。难点:DMA coalesced(octet=8 连续 global token)无法直接 scatter 成 interleaved。
  3. 回到 plain-read 版(975,−11%)+ 接受,或 V 用 ds_read_tr(治标,~359 cyc 基准)。

<a id="sec-trace_segment_cycles-py-精确-cycle-对比permuteb128-版2026-06-24"></a>
### trace_segment_cycles.py 精确 cycle 对比(permute+b128 版,2026-06-24)
| 区间 | lin cyc | vec cyc | 差 |
|---|---|---|---|
| **V ds_read** | 359.6 | **3620.4** | **+907%** |
| K ds_read | 222.8 | 233.7 | +4.9%(持平) |
| between V&K | 1032 | 2911 | +182% |

**致命发现:VGPR spill。** ATT trace 显示每条 V ds_read_b128 前都插了
`scratch_load_dword + s_waitcnt vmcnt(0)`(idx 981-1013)——14 次串行显存往返,把 V read 从
359→3620 cyc(10×)。指令只多 19%,cycle 暴 9 倍 = 纯 spill stall。**permute+b128 引入的 VGPR
压力超预算 → spill,远比要解决的 V read 慢更严重。此形态走不通。**
- plain-read 版 V read=769(无 spill,只是 ds_read2st64 慢);permute+b128 版=3620(spill)。
- K 仍持平(233 vs 222),证明 K coalesce 无害。
<a id="sec-地址寄存器-spill-修复2026-06-24关键"></a>
### 地址寄存器 spill 修复(2026-06-24,关键)
**根因**(用户质疑"V 数据 VGPR 应与 linear 相同"促成):V 数据 VGPR 确实相同(都 64=16pack×4
或 32pack×2),spill 不来自数据,而来自**地址寄存器**。我原 `_read_v8f16(a0)` 把整个 a0(含
dc/k_substep const + lane runtime)传 get_element_ptr → 编译器为 16 个不同 a0 各分配 1 个地址
VGPR(16 个活跃)→ 挤爆 256 → spill 25。
**修复**:照 K read 模式,拆成**单一 per-lane runtime 基址(循环外算一次,共享)+ 编译期常量 offset
(折进 ds_read 立即数)**:
```
v_addr_base = v_base + (lane%32)*64 + (lane//32)*8   # runtime,1 个 VGPR
const_off   = dc*(32*64) + k_substep*16              # const → ds_read imm
```
spill 25→**2**(≈linear 的 1)。perf 557→**713**。MaxErr 3.91e-3。
ATT trace 确认 V read 现在 16 条干净 ds_read_b128 共享基址 v171+offset(0/32/64/96/4096..)。

<a id="sec-spill-修复后-cycle-对比-暴露-bank-conflict2026-06-24"></a>
### spill 修复后 cycle 对比 → 暴露 bank conflict(2026-06-24)
| 区间 | lin | vec | 差 | inst |
|---|---|---|---|---|
| **V ds_read** | 359.6 | **1961** | +445% | **21 vs 36(−42%)** |
| K ds_read | 222.8 | 171.2 | −23%(更快) | 20 |
| between V&K | 1032 | 2342 | +127% | +2.6% |
**V read 指令更少(21<36)但 cycle 高 5.4×→ stall-bound,非 spill(scratch 已清),非 permute
(那在 between 段)。** 隔离出真凶 = **ds_read_b128 在 [d][n] 布局的 32-way bank conflict**:
LDS A=d*64+n,相邻 lane(lane%32→d)相差 64 元素=128B,正好 = 32 banks×4B → 相邻 lane 同 bank。
**这才是最初推测、现在被实测隔离确认的 bank conflict。** K 不冲突(布局 A=(d//8)*512+n*8+d%8)。
between V&K +127% 仍含 permute 开销(permlane32),是第二个待解问题。

<a id="sec-加-lds-pad-消-v-bank-conflict-失败buffer_load_lds-要求-wave-内连续2026-06-24"></a>
### 加 LDS pad 消 V bank conflict 失败:buffer_load_lds 要求 wave 内连续(2026-06-24)
试了 padded [d][n] 布局(VEC_V_PAD=4,row stride 68,128 行×68=8704 正好塞满现有 V 分配)。
- DMA + read 地址都按 68 重算,纸上自洽。**但 MaxErr=1.94(错)**。pad=0 立刻恢复 3.91e-3。
- **根因**:`buffer_load_lds`(coalesced DMA)要求**一个 wave 的 64 lane 写的 LDS 地址连续**。
  无 pad 时 64 lane 写连续 512 元素(oc*8)✓;加 pad 后 d 行间有 4 元素 gap → wave 内写地址不连续
  → 硬件写错位 → 数据错。
- **确认了最初 handoff 的担心:物理 pad 与 coalesced DMA 互斥。** 不能用加 pad 消 V bank conflict。
- **剩余消 conflict 选项**:
  1. **XOR swizzle**:LDS A = d*64 + (n XOR f(d)),每行仍 64 连续(DMA 连续写不变),用 XOR 打散
     d→bank。read 端同样 XOR。GPU LDS 消 conflict 标准手法,不占额外空间。**首选。**
  2. V read 换 ds_read_b64(2 banks/lane vs b128 4 banks),治标。
  3. DMA 后 LDS 内重排(多一次 round-trip),太贵。
- ~~当前代码:pad=0(=无 pad 的 713 版)~~。

<a id="sec-dense-style-行边界-pad-成功2026-06-24用户指点抄-dense-pad"></a>
### dense-style 行边界 pad 成功(2026-06-24,用户指点:抄 dense pad)
**关键:pad 放在 wave-row 边界(512 元素一行),不在 wave 内部** —— 完全照 dense 的做法。
- V-LDS 改成 dense 同构:每 wave 写一行 512 连续(64 octet,lane→+lane*8),行间 stride
  VEC_V_ROW_STRIDE=544(=512+dense SMEM_V_PAD 32)。16 行×544=8704=SMEM_V_TILE_ELEMS,分配不变。
- DMA: row=wave*2+d_iter, lds=row*544+lane*8(wave 内连续 ✓,pad 在行边界)。
- LDS elem(n,d)=(d//8)*544+((d%8)*8+n//8)*8+n%8。read A0 拆 per-lane base + const off(立即数):
  base=(lm//8)*544+(lm%8)*64+(lane//32)*8, off=dc*4*544+step*16。仍单条 ds_read_b128。
- **正确 MaxErr 3.91e-3。perf 16/8192/64/64: 713→888**(pad 消了大部分 bank conflict)。
- 进展轨迹:gather 408 → plain 975 → permute+b128 spill 557 → spill 修复 713 → +pad 888
  (linear 1101,现 −19%)。
- **剩余 gap**:888 仍 < plain 975 < linear 1101。permute 开销(between V&K)仍在拖。

<a id="sec-pad-后-att-cycle-复测2026-06-24"></a>
### pad 后 ATT cycle 复测(2026-06-24)
| 区间 | lin | vec(pad前) | vec(pad后) |
|---|---|---|---|
| V ds_read | 359.6 | 1961 | **939.9**(inst 21 vs lin 36) |
| K ds_read | 222.8 | — | 172.8(更快) |
| between V&K | 1032 | 2342 | 1523 |
- pad=32 把 V read 1961→940(消了大半 conflict),但仍 lin 的 2.6×。inst 更少却 cycle 高 →
  仍 stall-bound,pad=32 未完全消 conflict(或 b128 4-bank/lane 比 ds_read_tr 更易冲突)。
- between V&K 2342→1523,仍 +491 = permute(permlane32)开销。
- **两个剩余瓶颈**:(1) V read 940 vs 359(pad 不够/b128 冲突);(2) permute +491。
- **下一步候选**:(a) 试不同 pad 值(gfx950 bank 规则复杂,实测扫 pad);(b) V read 换 ds_read_tr
  复用 linear 的 359 基准(但那要回到 interleaved P,放弃 permute);(c) 优化/去 permute。
- ~~当前代码:dense-pad+permute+b128 版(888,−19%)~~。

<a id="sec-no-major-行内布局消-v-bank-conflict-成功2026-06-24用户给出-gfx950-b128-no-conflict-规则"></a>
### no-major 行内布局消 V bank conflict 成功(2026-06-24,用户给出 gfx950 b128 no-conflict 规则)
**根因(用户规则点破)**:原行内 d_local-major(位置=d_local*8+no)使 d%8 步长=64 元素=32 dwords
=0 mod 32 → 相邻 4 lane(d%8=0..3)永远同 bank,**pad 改不动**(所以 pad 只到 940)。
**修法**:行内改 NO-MAJOR(位置=no*8+d_local) → d%8 步长=8 元素=1 bank-quad/lane,相邻 lane 走
相邻 quad,匹配规则(lane0-7 → quad{0,4,..28}全异)。**不靠 pad,靠排布。**
- DMA: lane→no=lane//8, d_local=lane%8(对调内容映射,写位置仍 identity wave 连续)。
- LDS elem(n,d)=(d//8)*544+(n//8)*64+(d%8)*8+(n%8)。read base=(lm//8)*544+lane//32*64+(lm%8)*8,
  const=dc*2176+step*128。仍单条 ds_read_b128。
- **正确 3.91e-3。perf 888→958**。2/1024 规模 516 vs lin 515 = **100.2% 持平**!
- 完整轨迹:gather 408→plain 975→spill 557→修复 713→pad 888→**no-major 958**(linear 1101,−13%)。
- **剩余**:958 仍略低于 plain 975 → **permute 开销(between V&K)现在是主残留**,V read 已解决。
  下一步:ATT 复测确认 V read cycle 大降 + permute 段,然后攻 permute。
<a id="sec-no-major-后-att-cycle-复测2026-06-24"></a>
### no-major 后 ATT cycle 复测(2026-06-24)
| 区间 | lin | vec(pad) | vec(no-major) |
|---|---|---|---|
| V ds_read | 359.6 | 939.9 | **507.1**(inst 21 vs 36) |
| K ds_read | 222.8 | — | 188.0(更快) |
| between V&K | 1032.4 | 1523 | **1506.9(+474)** |
- V read 940→507(no-major 消了 conflict 主体),vs lin 359 已接近(剩 +41% 可能残留 minor
  conflict 或 b128 延迟,inst 更少)。
- **最大残留 = between V&K +474 = permute 开销**(16 条 permlane32 串行 + P cast/exp)。
- **下一步:攻 permute**(between V&K)。候选:(a)permute 每 pack 4→2 次 swap 减半;(b)permute 挪进
  mem-wait 窗口隐藏;(c)评估去掉 permute 改 V 读 interleaved(但那回到 ds_read2st64)。
<a id="sec-k-read-n-置换去掉-p-permute2026-06-24用户提案成功"></a>
### K-read n 置换去掉 P permute(2026-06-24,用户提案,成功)
**原理(ISA output layout §7.1.4.2 确认)**:QK^T 的 P n→lane 由 output layout 固定
(lane=j+32*((i/4)%2)),改不了。但 K 是 A 算子,A 的行 i=n;**置换 K 读取的 n(swap bit2/bit3 of
lane%32)→ 逻辑行 i 装的 n 重排 → QK^T 输出 P 直接落 8 连续 n**,无需 P permlane32。
- 验证两步:(1) non-causal 探针(绕过 mask)MaxErr 4.88e-4=baseline,证原理;(2) 改 causal mask
  后 causal 也对。
- **dump 确认置换后 P**:lane<32 r0-7→n0-7,r8-15→n16-23;lane≥32 +8(原是 +4)。
- 改动:(a) K read `_kn=swap bit2/bit3(lane%32)`(免费,只改地址常量);(b) 删 `_permute_p_pack_8n`
  调用(P 不再 permute);(c) causal mask:pair_thresholds 改连续 {0-7,16-23}, lane_off 4→8。
  全 KV_VECTORIZED const_expr 门控,linear 不变。
- **正确 causal/non-causal 3.91e-3。perf 958→984**(超过 plain 975,linear 1101,−10.6%)。
- 工具:vec_probe_nc.py(non-causal 探针)、dump_p_vec.py(置换后 P dump)。
- 进展轨迹:gather 408→plain 975→spill 557→修复 713→pad 888→no-major 958→**K置换 984**。
- **剩余 −10%**:permute 已去,V read 已解决。下一步 ATT 复测找新瓶颈(between V&K 应降;
  可能 K read 置换有 minor 代价,或别处)。
- 当前代码:no-major+pad+K置换+b128(无 P permute)版(984,−10.6%)。未 commit。**目前最佳。**

<a id="sec-去-p-permute-后-spill-回归2026-06-24att-复测发现"></a>
### 去 P permute 后 spill 回归(2026-06-24,ATT 复测发现)
ATT 显示去掉 P permute 后 **spill 回来了**(scratch 2→13),且 V/K read 地址退化:
const_off 太大(dc*4*544=6528 元素=13056B,如 0x3a00=14848B)超 ds_read 立即数有效折叠范围,
编译器退化成 `v_or_b32 v21, 0x.., v149` 临时算地址 + 占额外 VGPR → spill。
- 印证之前 noperm 实验(scratch 35>permute 的 25):permute 反而帮调度。去掉它要付 spill 代价。
- 但净 perf 仍升(958→984),说明去 permute 收益 > spill 代价(此规模)。
- **下一步候选**:(a) 修 V read 地址退化——把大 const_off 拆成「中等 runtime 基址步进 +
  小立即数」,避免 v_or + spill;(b) 或保留 P permute(958)对比取舍。
- 完整轨迹:gather 408→plain 975→spill 557→修复 713→pad 888→no-major 958→K置换 984。
  linear 1101。**vec 现 −10%(大规模)~−20%(2/1024)。** linear/dense compare 无回退(已验)。

<a id="sec-vgpr-用量定位isa-21_final_isa-s-的-num_vgprscratch2026-06-24"></a>
### VGPR 用量定位(ISA 21_final_isa.s 的 num_vgpr/scratch,2026-06-24)
三版都顶到 **num_vgpr=256(gfx950 单 wave 上限)**,区别在 spill(scratch load/store 数):
| 版本 | scratch_load | 说明 |
|---|---|---|
| linear (V=ds_read_tr) | **1** | 几乎不 spill |
| vec permute+b128 | **25** | spill |
| vec b128 无 permute(FLYDSL_DBG_NO_PERMUTE=1) | **35** | spill 更多! |

**关键结论(再次推翻假设)**:
1. **spill 不是 permute 引起的** —— 关掉 permute 反而 35>25。permute 居然略降 spill(改变寄存器
   生命周期,给调度腾空间)。
2. **根源是 vec 整条路径活跃 VGPR > linear**,把本就卡 256 边缘的 kernel 挤爆 spill。
   linear 用 ds_read_tr(v4f16,2 VGPR/pack)+ 交错 MFMA,活跃寄存器少;vec 用 ds_read_b128
   (v8f16,4 VGPR/pack)×16 pack 峰值高。
3. spill 发生在 prologue(scratch_store v14/v11),主循环每个 scratch_load 触发 vmcnt(0) 全停 → 致命。
- **新方向**:要让 vec 用 ds_read_b128 又不 spill,必须降峰值 VGPR。候选:
  (a) V read 不一次读 16 pack,拆成 2 批(8+8)交错 MFMA,缩短生命周期(像 linear);
  (b) 候选2(V-LDS interleaved 布局 + P 不动)未必降 VGPR,因为压力来自 b128 本身;
  (c) 接受 plain-read 版(975,−11%,无 spill)作为可交付结果。
- 工具:`_DBG_NO_PERMUTE` env flag(默认 0=permute on);ISA dump 用 FLYDSL_DUMP_IR + chk2.sh。
- **当前代码:permute+b128 版(557,退化+spill)。permute 默认开(正确,MaxErr 3.91e-3)。未 commit。**

<a id="sec-作废旧下一步v-换-ds_read_b64_tr_b16"></a>
### (作废)旧下一步:V 换 ds_read_b64_tr_b16
~~在新 [no][d][ni] 布局上重新 dump 求 ds_read_tr offset 产出同一 contract~~。
被上面 ISA 修正取代:ds_read_tr 仍是 transpose,治标;改 P 布局让 V 直接 ds_read_b128 才是治本。
K 维持现状(coalesce + plain read,已与 lin 持平)。

<a id="sec-v-read-的-v_or-地址-spill-根因2026-06-25实测排除两个错误假设"></a>
### V read 的 v_or 地址 + spill 根因(2026-06-25,实测排除两个错误假设)
现象:去 P permute 后,V read 部分 ds_read_b128 用 `v_or_b32 v21,0xNNNN,v149` 临时算地址
(非 `offset:` 立即数),且 scratch spill 回到 13。
- **错误假设1(已排除):"ds_read offset 必须是 2 的幂"**。查 CDNA4 ISA §11.3 line 7244:
  单地址 `LDS_Addr = LDS_BASE + VGPR[ADDR] + {offset1,offset0}`,offset 是 16-bit 无符号 byte
  加法偏移(0..65535),**无 2 幂/对齐要求**。V const_off ≤13824B 完全合法。
- **错误假设2(已实测排除):"非 2 幂 stride(544)诱发 v_or/spill"**。把 stride 改回 512(2 的幂,
  无 pad)实测:scratch 15>544 的 13,v_or 35(更多!)。**2 幂 stride 反而更差。**
- GEP 写法也无关:单层 `get_element_ptr(base, runtime+const)` vs 两步 `get_element_ptr(base+runtime)
  再 +const` 实测都 scratch=13。
- **真根因:整体 VGPR 压力**(去 P permute 后),编译器在 256 VGPR 顶满下对地址 lowering 劣化
  + spill。与 offset 对齐/stride 数值/GEP 写法都无关。permute 之前反而帮调度(noperm 实验
  scratch 35>permute 25)。
- 工具:chk2.sh(查 num_vgpr/scratch),vec_kp2_isa.s(本地 ISA 副本)。

<a id="sec-spill-根因隔离-修复k-σ-从-read-移到-dma2026-06-25关键突破"></a>
### spill 根因隔离 + 修复:K σ 从 read 移到 DMA(2026-06-25,关键突破)
**隔离矩阵**(compile-only,FLYDSL_DBG_NO_KPERM / FLYDSL_DBG_PPERM 开关,chk2.sh 读 scratch):
| K置换 | P permute | scratch |
|---|---|---|
| off | on | 2 |
| off | off | 4 |
| on | on | 12 |
| on | off | 13 |
→ **spill 主因是 K-read 的 σ 置换**(swap bit2/bit3 of lane%32),不是去 P permute(那只 2↔4)。
σ 的位运算 `(m&3)|((m&8)>>1)|((m&4)<<1)|...` 派生的 K read 地址在主循环热点占 VGPR → spill。
**之前"去 P permute 导致 spill"的归因是错的。**

**修复:把 σ 从 K read(热)移到 K DMA src(冷)。** K DMA 写 LDS slot 保持 wave-contiguous
(lds_addr=oct_idx 线性),但每个 slot 从 σ 置换后的 global token 填充(σ 作用在 src_oct,
不在 lds dst);read 回到 plain `_kn=lane_mod_32`。σ 的位运算移出主循环 → 不占 read VGPR。
- DMA: `_ni=oct_idx%64; _dg=oct_idx//64; src_oct=_dg*64 + sigma(_ni)`;lds 不变。
- read: `_kn = lane_mod_32`(干净)。
- **scratch 13→4。perf 16/8192/64/64: 983→1028。MaxErr 3.91e-3。**

**最终 perf(σ-in-DMA,vec vs linear)**:
| 规模 | vec | linear | 差 |
|---|---|---|---|
| 16/8192/64/64 | 1028 | 1100 | −6.5% |
| 16/8192/64/8 | 1088 | 1207 | −9.9% |
| 2/1024/64/64 | 535 | 583 | −8.2% |
| 1/8192/2/2 sk4 | 534 | 582 | −8.2% |
| 1/8192/4/4 sk2 | 596 | 663 | −10.1% |
轨迹:gather 408 → plain 975 → … → no-major 958 → K置换 983 → **σ-in-DMA 1028(−6.5%)**。
- **已清理**:删除 `_permute_p_pack_8n` 死代码 + 所有 `_DBG_*` 实验 flag,σ-in-DMA 固化为正式路径。
- **回归全过**:non-causal 4.88e-4;linear paged compare 16/8192=1103、2/1024=582... 与基准一致
  无回退,全 MaxErr 3.91e-3;clean 后 scratch 仍 4。

<a id="sec-当前代码状态2026-06-25"></a>
## 当前代码状态(2026-06-25)

**最佳版本:no-major V-LDS + dense-pad(stride 544) + K-read n置换(去 P permute) + ds_read_b128。**
perf 16/8192/64/64 ≈ **983**(linear 1101,−10.6%);2/1024 与 linear 持平。
causal/non-causal vec_probe MaxErr 3.91e-3;linear/dense compare 无回退(const_expr 门控)。
未 commit。剩余 −10% 来自去 permute 后的 VGPR 压力 spill(根因如上,非地址写法)。
完整轨迹:gather 408→plain 975→permute+b128 spill 557→spill 修复 713→dense-pad 888→
no-major 958→K置换去 permute 983。

<a id="sec-vgpr-spill-彻底消除epilogue-o-store-地址拆-baseconst2026-06-25不改-llvm"></a>
### VGPR spill 彻底消除:epilogue O-store 地址拆 base+const(2026-06-25,不改 LLVM)
**根因**:epilogue O-store 的 8 个地址 `o_global = q_row*stride + q_head*128 + dc*32 + (2g+lane//32)*8`
本是循环不变量,LLVM 后端把其中 lane 派生的列索引(`lane//32*8 + 常量`)**hoist 到 prologue**
并 spill,跨整个 main loop 占 4 VGPR slot → scratch 13。前端原写法每个 (dc,g) 重算完整 o_global,
后端没分离出"共享 runtime base + 编译期常量"。
**修复(纯前端,不改 LLVM)**:循环外算一次共享 base `o_base=_global_idx_q(q_row, lane//32*8)`,
循环内只加编译期常量 `dc*32 + 2g*8`。给后端清晰的「1 VGPR base + 立即数 offset」结构。
- **scratch 13→0(彻底消除),num_vgpr 256→252(还省 4)。MaxErr 3.91e-3。**
- LLVM remat 规则(从 GCNSchedStrategy.cpp 提取):单 def、单 use 跨 region、纯算术(isReMaterializable)、
  输入在 use 点可用;但 pre-RA remat 只按估计 RP≤MaxVGPR 决定,估计达标就不动 → 实际 regalloc
  仍 spill。所以靠前端拆地址比依赖后端 remat 更可靠。
- perf:多数规模持平/微升(16/8192/64/8、2/1024 更快),16/8192/64/64 有波动(时钟噪声,需多跑确认)。

<a id="sec-最终状态2026-06-25"></a>
### 最终状态(2026-06-25)
no-major V-LDS + dense-pad + K-read σ置换(σ 在 DMA src,非 read)+ ds_read_b128 + epilogue
O-store 地址拆 base+const。**scratch=0, vgpr=252, 全规模 MaxErr 3.91e-3。** 未 commit。

<a id="sec-v-dma-prep-上移2026-06-25隐藏-page_id-ds_read-延迟"></a>
### V-DMA prep 上移(2026-06-25,隐藏 page_id ds_read 延迟)
把 `_async_load_v` 拆成 `_async_load_v_prep`(纯地址:page_id ds_read + page view + lds/src 地址,
无副作用)+ `_async_load_v_issue`(只剩 buffer_load_lds)。`sched_barrier(0)` 把 prep 钉在 barrier
前侧、issue 在后侧。两处主循环 V load:
- **Cluster 4**(within-iteration):prep 移到 Cluster 3 barrier 前。
- **Cluster 0**(跨循环回边):loop-carried 传 page_id —— prologue 预取首个(tile=loop_lb-2),
  迭代 j 末尾(Cluster7 barrier 前)算 `_load_page_id(j)`(下一迭代 Cluster0 的 tile=(j'-2)=j),
  yield 携带;下一迭代 Cluster0 用 `page_id_override` 消费。KV_VECTORIZED const_expr 门控,
  非 vec/非 paged 路径不变。AST 陷阱:override 的 None 检查必须 `const_expr(... is not None)`。
- **结果**:spill 仍 0、vgpr 252、MaxErr 3.91e-3、linear 无回退。
  perf 16/8192/64/64 1003→1034,16/8192/64/8 1130→1141(spill 消除基础上再微升)。

<a id="sec-k-dma-prep-上移-prologue-tile2-prep-上移2026-06-25"></a>
### K-DMA prep 上移 + prologue tile2 prep 上移(2026-06-25)
把 `_async_load_k` 同样拆成 `_async_load_k_prep`(纯地址,含 σ 置换 + page_id ds_read)+
`_async_load_k_issue`(只剩 buffer_load_lds)。主循环两处 K load 同迭代上移(像 Cluster4 的 V):
- Cluster 2 K-prep 移到 Cluster 1 barrier 前;Cluster 6 K-prep 移到 Cluster 5 barrier 前。
- 两处 tile 依赖本迭代 j_idx,同迭代内完成,不需 loop-carried。
prologue:tile2 prefetch `_async_load_k((split_t0+2),0)` 拆 prep+issue,prep 提前到 prologue
softmax 的 barrier 之前。
- **结果**:spill 仍 0、vgpr 252、MaxErr 3.91e-3、linear 无回退。
- **最终 perf(同次 vec/linear 比值,消除时钟噪声)**:16/8192/64/64=99.1%、16/8192/64/8=98.1%、
  2/1024=101.6%、1/8192/2/2 sk4=100.6%、1/2048/4/4 sk4=101.7%、1/8192/4/4 sk2=99.0%。
  **vec 与 linear 基本持平(98-102%,多规模更快),达成 <2% 一致目标。**
轨迹:gather 408(−63%)→ … → spill 消除 → V/K prep 上移 → **±2% 持平**。

<a id="sec-epilogue-阶段-prep-上移2026-06-25"></a>
### Epilogue 阶段 prep 上移(2026-06-25)
Epilogue 有 4 个 DMA load。同段内(barrier 在同一 epilogue 段)的三处已上移:
- C2 K-prep → C1 barrier 前;C4 V-prep → C3 barrier 前;C8 V-prep → C7 barrier 前。
- 拆 prep+issue,sched_barrier(0) 钉住。spill 仍 0、vgpr 252、MaxErr 3.91e-3、vec/linear 97.6-99.9% 持平。
- **未做**:Epilogue C0(`_async_load_v(max_m3,1)`)——其上一 barrier 在主循环体内(末迭代
  Cluster7),跨 loop 出口,类同主循环 Cluster0 需跨边界 loop-carried;但 epilogue 只跑一次,
  收益微小、跨 loop-exit 风险高,暂留。

<a id="sec-epilogue-c0-跨-loop-exit-prep-上移2026-06-25完成"></a>
### Epilogue C0 跨 loop-exit prep 上移(2026-06-25,完成)
关键洞察:主循环每迭代末尾算的 `_next_v_pid = _load_page_id(j_idx)` 在循环退出时携带的是
`_load_page_id(last_j)`,而 **last_j == split_t_end-3 == max_m3 == epilogue C0 的 tile**
(step-2 调度 + split_t_end 偶数下成立)。所以直接复用 loop_results 携带的 page_id 给
epilogue C0,其 ds_read 实际发生在循环最后一迭代的 Cluster 7 barrier 前 —— 等于跨 loop-exit 上移。
- epilogue C0 改用 `page_id_override=loop_results[3+D_CHUNKS]`。
- **全规模(含 splitk set3)MaxErr 3.91e-3**,证明 last_j==max_m3 对所有目标规模成立。spill 0、vgpr 252。
- 注意:空循环(tiny seqlen S<256,非性能目标)下 last_j 未定义为 max_m3,但那些规模是 padding
  空转、不在 set1/set3,且 compare 已全过。
**至此 prologue + 主循环(C0/C2/C4/C6)+ epilogue(C0/C2/C4/C8)所有 V/K DMA 的 prep 都已上移。**
