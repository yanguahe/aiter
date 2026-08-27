# gfx1250 Grouped-MoE A4W4 TDM GEMM 当前分支优化方案

## 0. 当前分支复核

### 0.1 Git 身份与工作区

复核日期：2026-08-27。

本次从当前工作树重新核验，不沿用旧方案中的入口、默认值、性能数字或优先级。

```text
repo root: C:/Users/yanguahe/Documents/code/wk_sp1/aiter
branch:    hyg_gfx1250_gemm_a4w4
HEAD:      f6533a74032cf02d049b6bb91487097c71807c9c
```

审计开始时的简要状态：

```text
 M 3rdparty/composable_kernel
?? my_code/gfx1250_gemm_optimization_guide.md
?? my_code/gfx1250_grouped_moe_a4w4_gemm_optimization_plan.md
```

本方案文件和优化指南均为 untracked；`3rdparty/composable_kernel` 已有工作区变化。本次不修改、
清理、暂存或解释这些既有变化，也不切换分支、不 commit、不 push。

### 0.2 本次复核结论

当前分支正确，允许继续更新本方案。相对旧方案的关键调整如下。

1. 删除无法由当前分支或本次硬件运行复现的历史 duration、TB/s、E2E 和误差数值。测试中的
   hard-coded duration 只用于 Python 算术 fixture，不是当前机器 baseline。
2. 当前 workload 的两个预期 TDM symbol、`64x256x256`、`w1x4`、`b2`、K7168/K3072
   与源码、CSV row 和现存 final ISA 相互一致；实际运行仍须由 kernel trace/stats 再确认。
3. CSV 的 `next_stage_prefetch=1` 在 `b2` 下无效；源码只有在 `num_buffers >= 3` 时令
   `next_stage_on=1`。
4. 当前 A4W4 A2 quant 仍传 `topids_to_rows=None`，route-indexed 实现已经存在但尚未接入。
   因此该项保留，并提升为第一优先级。
5. G2/no-activation output LDS 已使用 `STORE_PAD=16`；G1/activation 路径为 0。旧方案把
   output padding 整体视为“尚未实现”不准确。
6. 当前 A4W4 路径没有 `partial_m64` 参数。源码中相似的 M16 谓词只位于
   `stage1_quant_out` 的 fused FP8 quant epilogue，而当前 A4W4 不走该分支。
7. WMMA 和 TDM 都不能靠 EXEC mask 跳过。任何 partial-M 实现必须使用 wave-uniform
   控制流，并保持固定 accumulator/rmem/layout。
8. cluster 当前只 multicast A payload；SA、B、SB 均未 multicast。`cluster_n` 基线为 1，
   当前代码只接受 2、3、4，并要求 N-tile 数整除。
9. stage-specific tile、warp 和 buffer override 已存在；cluster、waves-per-TDM 和
   next-stage 仍是两阶段共享。`TILES_PER_GROUP`、output pad、schedule hints 等没有环境覆盖。
10. 当前 benchmark 的 `run_perftest` 明确传 `testGraph=False`。容量计算是静态、capture-safe
    设计，但不能把本脚本的本次计时称为 graph-captured timing。
11. TDM grouped path 仍是 JIT-only；源码明确写有 AOT coverage TODO。旧方案要求无条件更新
    TDM AOT job key 不符合当前实现。
12. 首轮从低风险、已存在实现的 A2 route-indexed quant 开始；M32 成对实验先作为
    partial-M 的低成本因果门槛，不再直接以高风险 compute-control 改造开局。

### 0.3 明确禁用的诊断方法

当前机器的 ATT/thread trace 无效。本方案完全禁用该方法：不读取任何现存产物及其派生
CSV/cycle/stall/latency/hitcount，不建议重新采集，也不把它作为任何结论或回退手段。

允许的证据只有：

- 稳定、无 profiler 的 benchmark；
- `rocprofv3 --kernel-trace --stats` 及数据库中的 symbol、dispatch 和静态资源；
- 目标环境 `rocprofv3 --list-avail` 实际枚举出的 PMC；
- 当前 final ISA/IR/code-object metadata 的静态检查；
- 每次只改变一个逻辑变量的因果实验。

## 1. 固定 workload 与真实调用链

### 1.1 启动脚本

入口为 `my_code/run_gemm_a4w4.sh`：

```bash
python3 -u op_tests/test_flydsl_grouped_gemm_gfx1250.py \
  --scenario bench \
  --data-format a4w4 \
  --experts 96 \
  --tokens 512 \
  --topk 6 \
  --iters 100 \
  --model-dim 7168 \
  --inter-dim 3072 \
  --act silu \
  --no-bias \
  --no-check-aot-cache
```

脚本固定设置：

```text
ENABLE_CK=0
AITER_MOE_EXPERT_BALANCE=true
AITER_LOG_MORE=1
AITER_USE_GROUPED_GEMM=1
AITER_GROUPED_DEBUG=0
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1
```

`HIP_VISIBLE_DEVICES`、`FLYDSL_DUMP_IR` 和 `FLYDSL_DUMP_DIR` 可由调用者覆盖。

### 1.2 公共入口到 TDM kernel

当前调用链为：

```text
test_flydsl_grouped_gemm_gfx1250.py::_run_grouped_via_fused_moe
  -> aiter/fused_moe.py::_fused_moe_impl
  -> aiter/ops/flydsl/grouped_moe_gfx1250.py::grouped_gemm_gfx1250_a8w4
  -> _grouped_a8w4_tdm_moe
  -> grouped_gemm_mxfp4.py::flydsl_grouped_gemm_a8w4_masked
  -> kernels/mxfp4_preshuffle_gfx1250_tdm.py::launch_gemm_a8w4_tdm
```

关键选择事实：

- `aiter/fused_moe.py::_fused_moe_impl` 在 gfx1250 且未设置
  `AITER_FORCE_A8W4=1` 时把 activation quant dtype 设为 `fp4x2`。
- grouped helper 只服务 `GateMode.INTERLEAVE` 的 GUGU stage-1 weight。
- 测试通过 `moe_shuffle_weight(..., is_guinterleave=True, gate_up=True)` 和
  `moe_shuffle_scale(..., is_guinterleave=True, gate_up=True)` 构造对应布局。
- `--no-bias` 使两个 TDM specialization 都没有 `_bias` 后缀或 bias global load。

### 1.3 当前 CSV 命中

`aiter/configs/tuned_grouped_fmoe.csv` 的当前 A4W4 workload 命中 row 73：

```text
gfx/cu/token/model/inter/expert/topk = gfx1250/256/512/7168/3072/96/6
tile1/tile2                       = 64x256x256 / 64x256x256
wave1/wave2                       = 1x4 / 1x4
buffers1/buffers2                 = 2 / 2
cluster_n                         = -1
waves_per_tensor_tdm              = -1
next_stage_prefetch               = 1
```

解析后的实际值：

```text
cluster_n             = 1
waves_per_tensor_tdm  = 2
next_stage_on         = 0  # b2 不满足 num_buffers >= 3
```

该 CSV 行还含有 `split_k1/split_k2`、`grouped_persistent_m`、
`wave_specialized_tdm`、`tdm_as_in_prologue` 等列，但当前 TDM runtime dispatcher
没有读取它们。只修改这些列不会改变当前 kernel。

`grouped_moe_gfx1250.py` 文件头提到
`AITER_GROUPED_DEEPGEMM_CONTIGUOUS`/`grouped_contiguous_m`，但 runtime 文件中没有对应读取；
当前路径直接构造 contiguous-M psum。不得把该注释中的开关当作可用环境变量。

### 1.4 两个实际 specialization

由当前代码和配置推导的预期 symbol：

```text
GEMM1: a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1
GEMM2: a8w4_tdm_fp4_t64x256x256_w1x4_b2_K3072_e96
```

`a8w4_tdm` 是历史前缀；`fp4`、`a_is_fp4=1` 和
`v_wmma_scale_f32_32x16x128_f4` 才说明本 workload 是 activation FP4 × weight FP4。

现存工作树中有同名 `00_origin.mlir`、`19_gpu_module_to_binary.mlir`、
`20_llvm_ir.ll` 和 `21_final_isa.s`。它们可用于静态交叉核验，但目录本身没有提供可验证的
HEAD provenance，所以不能替代目标运行时的 symbol/resource 确认。

## 2. 当前源码事实

### 2.1 balanced route、M64 对齐与 static sentinel

`AITER_MOE_EXPERT_BALANCE=true` 由测试按 `.lower() == "true"` 解析。固定 shape 下：

```text
routes = 512 tokens * topk 6 = 3072
count[e] = 32, e = 0..95
```

`contiguous_psum_remap` 用 GEMM1 `tile_m=64` 计算：

```text
aligned_count[e] = 64
starts[e]        = 64 * e
psum[e]          = starts[e] + 32
active span      = 96 * 64 = 6144 rows
```

`_grouped_a8w4_tdm_moe` 的静态容量为：

```text
align_m       = max(tile_m, tile_m2) = 64
contiguous_m  = align_up(3072 + 96*64 - 6, 64) = 9216
max_m         = align_up(3072, 64) = 3072
```

因此：

```text
valid rows                = 3072
per-expert aligned padding = 3072
static sentinel tail       = 3072 rows = 48 M64 tiles
total static M tiles       = 144
active M tiles             = 96
```

GEMM1：

```text
N = 2*3072 = 6144
N tiles = 24
static WG = 144*24 = 3456
active WG = 96*24 = 2304
sentinel WG = 48*24 = 1152
```

GEMM2：

```text
N = 7168
N tiles = 28
static WG = 144*28 = 4032
active WG = 96*28 = 2688
sentinel WG = 48*28 = 1344
```

kernel 先对 `m_tile_map=psum` 做固定次数 binary search。超出最后一个有效 expert 的
static tile 得到 `expert == n_experts`，在 `if expert < n_experts` 外层跳过主循环。

### 2.2 两阶段共享 psum/alignment 的限制

当前只生成一份：

```text
contiguous_psum_remap(..., tile_m=GEMM1 tile_m)
```

同一 `psum` 同时传给 GEMM1 和 GEMM2。`align_m=max(tile_m,tile_m2)` 只用于容量上界，
并没有改变 psum 的实际对齐粒度。

因此：

- 当前环境变量允许单独设置 `AITER_TDM_TILE_M2`，但 runtime 没有验证 G2 tile 是否跨 expert。
- 在未修改 psum 生成规则前，调优清单只允许 `(M1,M2)` 成对改变。
- 若确需独立 M2，应先把 psum 对齐改为 `lcm(tile_m,tile_m2)` 或证明
  `tile_m` 是 `tile_m2` 的安全整数倍，并补充显式校验、空 expert 和跨 tile correctness。
- 不能只因 `_align_m=max(...)` 存在就宣称独立 M2 安全。

### 2.3 K pipeline 与 LDS arena

共同几何：

```text
tile_k      = 256
WMMA_K      = 128
KWS         = 2
num_buffers = 2
```

GEMM1 有 `7168/256=28` 个 K tile；GEMM2 有 `3072/256=12` 个。

每个 K256 stage 的源码常量：

```text
A payload LDS = 64 * (128+16) =  9,216 B
B payload LDS = 16 * 2,048    = 32,768 B
A scale LDS                       512 B
B scale LDS                     2,048 B
PITCH                           44,544 B
b2 input arena                  89,088 B
```

`ARENA_B=max(num_buffers*PITCH,C_STORE_B)`。当前 G1/G2 的 output staging 均小于
89,088 B，因此 code-object fixed LDS 为 89,088 B。

理论 LDS 容量上限只给出候选约束：

```text
b2:  89,088 B -> 至多 3 WG/WGP（仅按 320 KB LDS）
b3: 133,632 B -> 至多 2 WG/WGP
b4: 178,176 B -> 至多 1 WG/WGP
```

这不是实际 residency；VGPR、waves、barrier、TDM 和调度资源仍需 PMC/metadata 交叉验证。

### 2.4 `partial_m64` 的当前真实状态

当前 kernel 没有 `partial_m64` 参数。

已有 OOB 机制：

- A TDM descriptor 用 `mn_oob`，无效 A row 返回零。
- output TDM store 用 `mn_oob`，无效 row 的 global write 被丢弃。
- static sentinel WG 在 `expert < n_experts` 外层跳过。

尚未已有的机制：

- A4W4 compute loop 仍对整个 M64 执行 WMMA。
- A4W4 BF16 activation/no-activation epilogue仍遍历整个静态 M fragment。
- 当前 A4W4 没有按 M16 跳过 A/SA DS read。

源码中唯一相似谓词：

```text
if wmb + wm*16 < mn_oob:
```

位于 `stage1_quant_out and stage1_act` 的 fused FP8 quant epilogue。当前 A4W4
`_fuse_quant=False`，所以该分支不是当前执行路径，只能作为 wave-uniform 写法参考。

### 2.5 output LDS padding

当前代码：

```text
STORE_PAD   = 16 if not stage1_act else 0
STORE_PITCH = STORE_N + STORE_PAD
C_STORE_B   = aligned(tile_m * (tile_n+16) * 2)
```

所以：

- G1 `stage1_act=1`：`STORE_N=128`，`STORE_PAD=0`。
- G2 `stage1_act=0`：`STORE_N=256`，`STORE_PAD=16` 已启用。
- padded G2 source 使用二维 TDM store descriptor，inner OOB 限制为 `STORE_N`，pad 不写 global。
- padding 没有环境变量。
- arena 已按 `tile_n+16` 预留，但任何新 tile 仍须重新核对 `ARENA_B`。

G1 padding 仍可作为 LDS counter 门控后的实验，但不是“从未实现”的通用功能。它改变内部 LDS
source pitch 和 output TDM descriptor，不改变外部 tensor layout。

### 2.6 TDM owner、wpt 和 wait

当前 workgroup 为 `1x4` waves，128 threads。`waves_per_tensor_tdm=2` 时：

```text
wave groups = (0,1), (2,3)
A, SA owners = waves 0,1
B, SB owners = waves 2,3
TDM_PER      = 4 tensors * 2 owners / 4 waves = 2 ops/wave/K-tile
```

full M64 的 nominal payload：

```text
A+SA  ~=  8,192 +   512 B
B+SB  ~= 32,768 + 2,048 B
```

balanced32 时 A payload 还会被 row OOB 缩小，owner payload 更不均衡。当前 descriptor 数
对称不代表字节工作对称。

`b2` 且 next-stage off 的主 wait 阈值由 `TDM_PER` 推导，steady path 等待 2 个 outstanding
tensor op，drain 最终等待 0。现存 final ISA 中可见 `s_wait_tensorcnt 0x2` 和 `0x0`。

可用 wpt 值及含义：

- wpt1：A/B/SA/SB 分别归一个 wave；每 wave 每 K tile 一个 descriptor，payload 最不均衡。
- wpt2：当前两组 owner；每 wave 两个 descriptor。
- wpt4：每个 tensor 在四个 waves 间切分；每 wave 四个 descriptor，触及每 wave 最多三个
  issue-to-XACK tensor op 的硬件约束，可能产生 backpressure。

weighted ownership 目前不存在。实现后不能继续用单一 `TDM_PER`；必须按每个 wave 的真实
owner/job 数分别计算 wait threshold。

### 2.7 cluster multicast

当前 `_select_cluster_n`：

- 接受 2、3、4；
- 请求值小于等于 1、其他数值或不能整除 `n_tiles` 时回退 1；
- 环境变量 `AITER_FLYDSL_MXFP4_CLUSTER_N` 优先于 CSV。

kernel 使用：

```text
a_mcast_mask = (1 << cluster_n) - 1
```

并只把该 mask 传给 A payload descriptor。实际覆盖：

```text
A  : multicast
SA : unicast
B  : unicast
SB : unicast
```

原因是 cluster 沿 N 方向组织：peers 共享 `m_tile/expert`，但 `blk_n` 不同。A/SA 真正共享，
B/SB 随 N tile 变化，不能 multicast。

当前 code 限制到四个 peers，尽管硬件 descriptor mask 是 16 bit、cluster 最多 16 WG。
`early_timeout=True` 随非零 mask 设置；所有 peers 必须使用相同地址、mask 和匹配的 issue 数。

### 2.8 A2 quant、SiLU fusion 和 consumer

当前 A4W4：

```text
GEMM1 -> BF16 y[1, contiguous_m, inter_dim]
      -> flydsl_moe_fused_quant_preshuffle(
             y,
             E=1,
             max_m=contiguous_m,
             quant_mode="fp4",
             masked_m=None,
             topids_to_rows=None)
      -> GEMM2
```

所以 A2 quant 遍历全部 9,216 capacity rows。已有
`flydsl_moe_fused_quant_preshuffle` route branch 可令
`source_row=destination_row=topids_to_rows[route]`，但当前调用没有接入。

已经存在或部分存在：

- A1 quant 已传 `topids_to_rows` 和 `source_topk=topk`。
- A8W4 无 bias 路径已有 SiLU/SwiGLU -> FP8 quant fused epilogue。
- A4W4 没有 SiLU -> FP4 quant fused epilogue。
- `flydsl_moe_gather_reduce` 已融合 route gather、top-k weight multiply 和 reduction。
- stage2 GEMM 尚未直接写最终 token-order output。

### 2.9 tile-owner、persistent 和 sentinel 接口

当前主 TDM kernel：

- 用 psum binary search 得到 tile owner；
- 用 static oversized grid 保持 host 无需读取 device actual span；
- sentinel 只走 lookup 和 outer skip；
- `TILES_PER_GROUP=16` 是静态 swizzle，不是 persistent scheduler。

仓库中存在但当前调用链未使用：

- `fused_route_psum_remap` 单 workgroup route+psum 接口；
- `flydsl_moe_fused_route_psum_quant_scatter` 及其固定 worker、手写 grid-wide barrier；
- `build_moe_fused_route_psum_quant_scatter_module`。

这些是 dormant prep fusion，不是当前 GEMM persistent work queue，也没有被 row 73 或当前
grouped helper 调用。后续可评估，但不能写成“当前已启用”。

### 2.10 环境覆盖的实际边界

当前存在的 stage-specific 参数：

```text
AITER_TDM_TILE_M / AITER_TDM_TILE_M2
AITER_TDM_TILE_N / AITER_TDM_TILE_N2
AITER_TDM_TILE_K / AITER_TDM_TILE_K2
AITER_TDM_NUM_BUFFERS / AITER_TDM_NUM_BUFFERS2
AITER_TDM_M_WARP / AITER_TDM_M_WARP2
AITER_TDM_N_WARP / AITER_TDM_N_WARP2
```

重要解析规则：只要任一 tile/buffer stage-1 override 被设置，未设置的 stage-2 值就跟随
stage-1 base，而不是保留 CSV。wave-grid override 同理。做单阶段实验时必须显式固定另一阶段。

两阶段共享：

```text
AITER_FLYDSL_MXFP4_CLUSTER_N
AITER_FLYDSL_NUM_WAVES_PER_TENSOR_TDM
AITER_TDM_NEXT_STAGE_PREFETCH
```

其中：

- wpt 只有在 CSV 值不为 1/2/4 时才读环境；当前 row 为 -1，所以环境生效，默认 2。
- next-stage 只接受字符串 `0` 或 `1`，且 b2 下仍折叠为 off。
- cluster 请求非法时静默回退 1；实验必须从最终 symbol 确认实际值。

全局、import-time compile hint：

```text
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE
```

它在 `tensor_shim.py` 导入时解析，默认 0，脚本设为 1；不是 stage-specific，symbol 也不编码它。
0/1 对照必须使用新进程、独立 dump 目录并比较 final ISA。

当前不存在环境覆盖：

```text
TILES_PER_GROUP
LDS_PAD_A
STORE_PAD
MMA_GROUP
FENCE_COVER_MMA
emit_hints
addr_keepalive
vgpr_keepalive
partial_m64
weighted_tdm_ownership
```

`emit_hints()`、`addr_keepalive()` 和 `vgpr_keepalive()` 当前均在有效逻辑前立即 `return`。

## 3. Final ISA、IR 与 code-object 证据

### 3.1 当前存在的实际 artifacts

工作树中存在：

```text
my_code/flydsl_dump/moe_a4w4/
  a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1/
  a8w4_tdm_fp4_t64x256x256_w1x4_b2_K3072_e96/
```

每个目录均有 origin MLIR、binary MLIR、LLVM IR 和 final ISA。

### 3.2 静态 metadata

两个 code object 都显示：

```text
target                    = gfx1250
code object version       = 6
workgroup                 = 128x1x1
wavefront                 = 32
group_segment_fixed_size  = 89,088 B
private_segment_fixed_size= 0
kernarg_segment_size      = 184 B
metadata vgpr_count       = 239
metadata sgpr_count       = 66
vgpr_spill_count          = 0
sgpr_spill_count          = 0
```

descriptor 另显示 `next_free_vgpr=257`、`next_free_sgpr=64`。这些字段口径不同，不应挑一个
直接当实际 allocation；kernel stats 和硬件资源表必须一起报告。

### 3.3 静态 opcode 只作结构证据

两个 final ISA 文本各有：

```text
64  static v_wmma_scale_f32_32x16x128_f4
12  static tensor_load_to_lds
 6  static s_wait_tensorcnt
 1  static tensor_store_from_lds
```

G1/G2 的 K tile 数不同却有相同 static WMMA 文本数，说明 loop/tail 结构不能用简单 grep
还原动态执行量。动态 WMMA/TDM 数必须来自问题几何、实际 PMC 或受控 A/B。

## 4. gfx1250 硬件事实、推断与待测

### 4.1 文档事实

主要来源：

- `amd-instinct-cdna5-instruction-set-architecture.txt`
  - §7.12：WMMA 是整 wave matrix operation。
  - §10.11：TDM descriptor、OOB、padding、multicast 和 `TENSORcnt`。
- `MI400_Shader_Programming#65.txt`
  - §1.4、§1.5：四个 SIMD32、native wave32、每 SIMD-pair TDM。
  - §2.3：cluster 最多 16 WG，且每个 cluster WG 位于独立 WGP。
  - §3.3：VGPR 16-register allocation granularity、最多 1024 VGPR/wave。
  - §4.6.12：WMMA 强制全 EXEC。
  - §4.10.8：每 wave 最多 3、每 SIMD 最多 6 个 issue-to-XACK tensor op；
    issue-to-completion 由 6-bit `TENSORcnt` 跟踪。
- `amd-cdna5-whitepaper.txt`
  - 256 active WGP、四 SIMD32/WGP、Wave32。
  - 384 KB local SRAM，产品描述为最多 320 KB LDS + 64 KB vector cache。
  - 64 KB WGP instruction cache、192 MB L2。
  - MI455X peak HBM4 23.3 TB/s、OCP MXFP4 40.26 PFLOP/s。
- `mi400_merge_tcp_lds_cu#85.txt`
  - direct-copy tracking 可覆盖 512 个 128 B 或 256 B copy；每个不超过 256 B 的
    TDM copy 占一个 entry。
- `MI400_GL1_CH_GLARB_Fabric_MAS#12.txt`
  - 一组相同 multicast request 的 GL1 合并上限为 5；这不是 cluster 容量上限。

### 4.2 WMMA/TDM 与 EXEC

硬件文档明确：

- WMMA 不因 EXEC mask 为 0 而跳过，数学运算等效于 EXEC 全 1。
- tensor instruction 也忽略 EXEC，并且不是 per-lane operation。

因此 partial-M 的唯一安全控制原则是：

```text
wave-uniform condition -> real scalar control-flow branch -> WMMA/TDM body
```

不能使用 lane predicate、`V_CMPX` 或只收窄 EXEC 来“屏蔽” WMMA/TDM。最终 ISA 必须看到
真正绕过目标 WMMA block 的 scalar branch，并确认所有 waves 在 workgroup barrier 前重新汇合。

### 4.3 direct copy 与 request 宽度

文档事实：

- TDM descriptor 描述 tile，但硬件拆成不超过 256 B 的 async copy。
- direct copy 绕过 WGP$ data staging，不绕过 GL1/GL2 路由。
- 128 B request 需要相应对齐；稳定 256 B request 需要 256 B 对齐和足够连续字节。
- tracking entry 下 256 B 比 128 B 携带更多在途 payload。

当前源码推断：

- FP4 `tile_k=256` 的 A row payload 为 128 B。
- `tile_k=512` 的 A row payload 为 256 B。
- B/scale preshuffle 为较宽请求创造条件。

但以下仍是待测：

- 实际 128/256 B request 比例；
- direct/indirect 比例；
- 是否由 tracking、slot、LFIFO、GL1 credit 或 HBM 主导。

数学 `tile_k` 不能单独证明请求宽度，final ISA 也不能证明运行时 direct ratio。

### 4.4 multicast 边界

硬件事实：

- cluster 可有最多 16 WG；
- descriptor mask 为 16 bit；
- GL1 单次相同返回最多合并 5 个请求。

当前代码事实：

- 只支持 `cluster_n=2/3/4`；
- 只对 A 设置 mask；
- peers 必须同地址、同 mask、同 issue 数；
- mask 覆盖整个一维 cluster。

推断：

- 对当前 N-cluster，A 和 SA 可共享，B 和 SB 不可共享。
- cn4 的 request group 不超过 GL1 合并上限 5。

待测：

- timeout/downgrade；
- upstream byte 是否真实下降；
- cluster co-residency 是否抵消收益。

### 4.5 产品峰值的使用边界

23.3 TB/s、40.26 PFLOP/s 和 54 TB/s L2 都是产品级 peak，不是当前 stepping、
当前 clock 或当前 kernel 的 sustained baseline。只能用目标机器实测的 sustained bandwidth、
运行期 gfx clock 和 PMC 构造 runtime roofline。

## 5. 优化指南的适用分类

### 5.1 可直接应用

以下原则与当前两个 grouped kernel 直接一致。

1. 先区分 HBM/GL2 bandwidth、on-chip feed、latency/overhead 和 compute bound。
   证据：当前 logical accounting 不能代表 physical bytes，且 static grid 含大量 sentinel。
2. 每次只改变一个 tile、stage、cluster、wpt、layout 或 fusion 变量。
   证据：当前 env fallback 会联动两阶段，必须显式固定另一阶段。
3. 真实随机数据、稳定温度/频率、多个独立进程；同时报告 cycles/time 和 gfx clock。
   证据：kernel 注释明确指出常量零会掩盖错误 LDS address。
4. 先枚举目标环境 counter，再按 subsystem 分 pass，不猜 counter 名。
5. 区分 logical request、unique address、physical HBM/GL2 bytes、
   executed FLOP 和 useful FLOP。
6. 只 multicast 真正共享的 operand。当前 N-cluster 可直接应用于 A，SA 需一行 descriptor
   适配；B/SB 应排除。
7. stage 增加必须同时检查 wait、LDS、VGPR、direct fallback 和 residency。
8. wpt 必须遵守每 wave 3、每 SIMD 6 的 descriptor issue-to-XACK 限制。
9. weight/scale pre-shuffle 要端到端验证；当前 B 和 SB 已采用专用布局。
10. output store 要形成连续 cache-line，并比较 LDS permutation 与资源成本；当前 G2 已有
    padded LDS -> TDM store 实例。

### 5.2 需要适配后应用

1. 指南的大 tile/高复用原则必须适配 grouped small-M histogram。当前每 expert 只有 32 rows，
   先测试共同 M32 或 partial-M，不能直接采用 dense M256。
2. activation pre-shuffle 必须由 A1/A2 producer 直接生成，并把 producer 时间和流量计入 E2E；
   不能只看 GEMM。
3. 256 B direct-copy 建议需适配 FP4 packed byte 数、base/stride alignment 和 padding；
   K256 元素不等于 256 B。
4. cluster 需适配当前一维 N-cluster 和 source `workgroup_mask` 协议，不采用 dense 4x4
   A/B 双向 multicast。
5. co-execution/schedule hint 需保持 LD_SCALE/WMMA hazard、固定 rmem 和最终 correctness；
   当前 `emit_hints` 关闭，不能按指南文字直接打开全部 hint。
6. persistent 需适配 graph-safe device actual count、当前 psum 和 fixed worker barrier。
   仓库有 dormant prep-fusion prototype，但没有 GEMM work queue。
7. epilogue fusion需适配当前“A4W4 GEMM1 BF16 round -> FP4 quant”语义。A8W4 的 FP8
   fusion不能机械复用。
8. independent M2 需先适配共享 psum 对齐。

### 5.3 明确排除或降级

1. dense `256x256`、`128x512` 和 4x4 cluster 不是当前 t64 grouped kernel 的直接候选。
2. “方形 tile 永远最好”被排除；当前 M 由 route histogram 约束，N/K 还受 bytes、reuse、
   VGPR/LDS 和并行度约束。
3. “pre-shuffle 后一定是 256 B direct”被排除。
4. “TDM direct 绕过 GL1/GL2”被排除。
5. “cluster 最多 5 WG”被排除；5 是 GL1 单次合并上限。
6. “memory-bound 必须 occupancy=1”被排除。
7. split-K/stream-K 暂不进入首轮：当前 TDM runtime 不消费 CSV split 列，且 partial output
   和 reduction 流量尚无收益证据。
8. 其他 dense F4 assembly、tile 和 replay runner 不作为当前源码入口或性能证据。
9. 以一个 utilization counter 判 compute bound 被排除。指南指出 expert scheduling 下部分
   XDL cycle counter 有已知低报风险，应与 WMMA FLOP/operation、shader cycles 和 clock
   交叉验证。
10. 假定固定 1000-cycle memory latency、固定 ramp-up 或固定 silicon 百分比被排除。

## 6. 基线、PMC 与判定协议

### 6.1 先建立当前 baseline

禁止沿用旧方案数字。硬件 agent 应：

1. 使用原始脚本和随机输入。
2. 每个 specialization 先完成 JIT，再开始计时。
3. 至少运行 5 个独立进程，每进程保留 100 timed iterations。
4. 分别执行：
   - `--scenario bench`：E2E；
   - `--scenario kernel`：孤立 GEMM1/GEMM2；
   - `--scenario verify`：快速 correctness。
5. 返回 median、min、max、CV、运行期 gfx clock、温度/功耗（不可得则 `UNKNOWN`）。
6. baseline 单项 CV 目标不高于 2%；否则先处理热稳态、其他负载和 JIT 污染。

当前脚本 `testGraph=False`；报告必须写明实际 timing mode。

### 6.2 kernel trace/stats 的限定用途

只用于：

- 精确 symbol 和 dispatch count；
- grid、workgroup 和 agent/queue；
- code object；
- static SGPR/VGPR/LDS/private/scratch；
- profiler perturbation 检查。

数据库查询步骤：

1. 查询 `sqlite_master`，不假设 rocprof schema。
2. 对实际 kernel、dispatch、agent 表做 `PRAGMA table_info`。
3. 若存在 `rocpd_info_kernel_symbol`，查询两个精确 symbol 的资源字段。
4. 运行前后各做一次无 profiler 短 baseline。
5. profiler duration 与无 profiler duration 分开报告。

### 6.3 PMC catalog

先执行目标安装的：

```bash
rocprofv3 --list-avail
```

保存完整原始输出。只从实际 catalog 映射以下语义：

- SQ：FP4 WMMA operations/FLOPs、shader cycles、issue stall、active waves。
- TDM/TX：payload bytes、128/256 B requests、direct/indirect、
  descriptor/FIFO/slot/credit stall。
- LDS/DS：read/write operations/bytes、bank/segment conflict、
  DS issue/busy/output stall。
- GL1：request bytes、read/data credit、multicast request/timeout/downgrade。
- GL2：read/write bytes、hit/miss、request/data stall、channel分布。
- HBM/EA：physical read/write bytes、channel分布、busy/stall。
- Residency/clock：active WG/waves/WGP、gfx clock。

若语义不存在：

```text
status = UNKNOWN
reason = counter unavailable
fallback = stable single-variable A/B
```

不允许从指南或其他 GPU 猜 counter 名。

### 6.4 分 pass 与因果纪律

- 一个 pass 只采硬件允许的一小组同 subsystem counter。
- 一个 pass 只分析一个精确 kernel。
- 每 pass 至少重复 3 次，并记录无 profiler 对照。
- overflow、全零、实例缺失或跨 pass 不一致的 counter 标无效。
- 每次只改变一个逻辑变量；因 env fallback 被迫同时显式设置 baseline 值不算第二变量。

### 6.5 当前静态 workload 模型

以下来自当前测试公式，不是 physical bandwidth：

```text
GEMM1 executed FLOP = 2*6144*6144*7168 = 541,165,879,296
GEMM2 executed FLOP = 2*6144*7168*3072 = 270,582,939,648

GEMM1 useful FLOP   = 2*3072*6144*7168 = 270,582,939,648
GEMM2 useful FLOP   = 2*3072*7168*3072 = 135,291,469,824
```

当前 active tile logical read：

```text
GEMM1:
  A payload      264,241,152 B
  A scale         33,030,144 B
  B payload    2,113,929,216 B
  B scale        132,120,576 B

GEMM2:
  A payload      132,120,576 B
  A scale         16,515,072 B
  B payload    1,056,964,608 B
  B scale         66,060,288 B
```

B+SB 占该 logical model 约 88.3%，A+SA 约 11.7%。这只说明优先检查 B feed 和
small-M wasted compute；不能证明 HBM bound。

### 6.6 通用晋级门槛

Hard correctness：

```text
logits_diff < 0.01
no NaN/Inf
no memory fault
no hang
```

语义不变候选还应：

- valid output、payload 和 scale 优先 bitwise 等于 fresh baseline；
- 若浮点重排使 bitwise 不成立，报告 candidate-vs-baseline 的 max abs/max rel/normalized L2；
- fresh reference error 不得比 baseline 恶化超过 10%；
- 通过 balanced、default random、activated experts 6/24/96、空 expert 和跨 M tile case。

性能晋级：

```text
median improvement >= max(2%, 3*baseline_CV)
```

高风险源码改造目标通常至少 3%。任何收益必须跨独立进程重现，且 final resources、clock 和
physical bytes 不出现无法解释的回退。

## 7. 重新排序后的优化优先级

### P1：接入 route-indexed A2 FP4 quant

当前位置：

- `aiter/ops/flydsl/grouped_moe_gfx1250.py::_grouped_a8w4_tdm_moe`
- A4W4 的第二次 `flydsl_moe_fused_quant_preshuffle` 调用
- 已有实现：
  `aiter/ops/flydsl/moe_kernels.py::flydsl_moe_fused_quant_preshuffle`
  和 `build_moe_fused_quant_preshuffle_route_ksplit_module`

最小修改：

```text
topids_to_rows=topids_to_rows
source_topk=0
num_valid_routes=_ep_nvr
```

保持 `E=1`、`max_m=contiguous_m`、output allocation 和所有 GEMM 参数不变。

机制：

- 当前 full-row 分支量化 9,216 rows。
- route branch 只量化 3,072 mapped rows。
- `source_topk=0` 令 source/destination 都是同一 contiguous grouped row。
- 每个 route row 唯一；A1 已使用同类 route-indexed producer。

静态工作量：

```text
BF16 read:         56,623,104 -> 18,874,368 B
FP4 payload+scale: 15,040,512 ->  5,013,504 B
saved logical traffic:          47,775,744 B
```

预期改善对象：

- A2 quant kernel；
- G1->quant->G2 之间的 E2E fixed cost；
- 无效 capacity/sentinel row 的 BF16 read 和 FP4/scale write。

风险：

- route branch 对当前 shape 会选择 K-split launch geometry；
- 必须确认 scale store 无 race；
- padding row scale 未写，但当前 A1 route path 已依赖“无效 row 不影响 valid output”的同一不变量；
- EP dead-tail 必须受 `_ep_nvr` 限制。

最小单变量实验：

```text
baseline: topids_to_rows=None
candidate: topids_to_rows=current map, source_topk=0
```

PMC/benchmark 成功判据：

- quant kernel physical read/write bytes和时间显著下降；
- E2E 达到通用晋级门槛；
- GEMM1/GEMM2 geometry、symbol 和 duration 不应发生结构性变化；
- valid A2 payload/scale 与 baseline bitwise 一致；
- final output correctness 不变。

停止/回滚：

- K-split fixed cost抵消 E2E；
- scale race、EP dead-tail 或 correctness 失败；
- 回滚为 `topids_to_rows=None`。

### P2：共同 M32 作为 low-risk small-M 因果实验

当前参数：

- `AITER_TDM_TILE_M`
- `AITER_TDM_TILE_M2`
- `get_wmma_m_rep`
- shared `contiguous_psum_remap`

机制：

- balanced count 恰为 32，M32 去掉 active tile 内 50% padding compute。
- 每 expert 仍只有一个 active tile，所以 balanced case 不增加 B/B-scale active surface。
- `wmma_rep` 从 4 降为 2，LDS/VGPR 可能下降。

M32 的静态布局：

```text
actual aligned span = 3072 rows
contiguous_m        = align_up(3072 + 96*32 - 6, 32) = 6144
active M tiles      = 96
sentinel M tiles    = 96
```

它减少 active compute，但增加 sentinel tile 数；必须测 E2E，不能只看 useful FLOP。

风险：

- random/skew 中 count>32 的 expert 产生多个 M tile，并重复完整 B surface；
- sentinel lookup 数增加；
- 新 tile 可能改变 VGPR、LDS 和 instruction footprint。

最小单变量实验：

```text
(M1,M2) = (64,64)
(M1,M2) = (32,32)
```

显式固定 N/K/warp/buffer/cluster/wpt/next-stage，避免 env fallback 改变其他值。先保留 P1
winner，使 A2 quant 不再随 capacity 改变。

PMC/benchmark 成功判据：

- balanced GEMM1、GEMM2 和 E2E 达到通用门槛；
- dynamic WMMA operations、A-scale/DS work按预期下降；
- B/B-scale physical bytes在 balanced case不增加；
- random、6/24/96 active-expert case无超过 2% 的不可解释回退；
- 无 spill，resources 可解释。

停止/回滚：

- sentinel overhead抵消 active compute；
- random/skew B bytes或时间明显回退；
- 回滚共同 M64。

### P3：M64 内 wave-uniform partial-M16

进入条件：

- P2 证明减少 padding compute 有价值，但 M32 因 skew/B reuse 或 sentinel 失败；或
- PMC 明确显示 WMMA/epilogue work仍是主要可减对象。

当前位置：

- `mxfp4_preshuffle_gfx1250_tdm.py::mma_rows`
- A4W4 BF16 activation/no-activation output staging loop
- `mn_oob`、`wmb`、`wm`

第一版机制：

```text
valid_m16 = (wmb + wm*16) < mn_oob
```

- 用 wave-uniform scalar branch 包围对应 WMMA 和该 M16 epilogue。
- `c_frags`、`rmem_slots`、list 长度、index、LDS arena 和 descriptor shape全部保持固定。
- 无效 accumulator 保持初始化零。
- 所有 workgroup barrier 和 tensor wait 保持在 uniform 外层。
- 第一版不跳过 TDM 或 DS load，以单独测 compute/epilogue 因果。

第二版只有第一版成功后才测试：

- 对无效 M16 跳过 A/SA DS read；
- 仍向固定 rmem slot 写 neutral value；
- 作为独立 compile-time flag，不与第一版同一实验合并。

WMMA 安全要求：

- 禁止 EXEC-only masking；
- final ISA 必须有实际 scalar branch 绕过 WMMA；
- branch 条件在 wave 内一致；
- barrier 前重新汇合；
- static WMMA grep 数可能不变，成功以 dynamic PMC 和时间判断。

预期改善对象：

- balanced M64 的动态 WMMA；
- SiLU/BF16 convert/output LDS store；
- 第二版才包含 A/SA DS traffic。

风险：

- compiler 把 uniform branch 降成 EXEC predication；
- code size和 SALU branch cost；
- fixed accumulator 使 VGPR不下降；
- 改动 rmem shape会破坏 allocator/layout；
- 多 wave-M layout 下 valid 条件不同，但仍必须各 wave uniform。

最小单变量实验：

```text
partial_m16_compute=0
partial_m16_compute=1
```

PMC/benchmark 成功判据：

- balanced dynamic WMMA operations 接近有效 M16 比例；
- GEMM1/GEMM2 至少一个达到高风险 3% 门槛，E2E达通用门槛；
- B/B-scale bytes不增加；
- fixed VGPR/LDS不恶化；
- valid output与baseline一致。

停止/回滚：

- final ISA 只有 EXEC mask，没有 scalar skip；
- branch cost抵消收益；
- random/skew regression >2%、spill、hang或correctness失败；
- flag 默认 0，保留 M64 baseline symbol。

### P4：stage-specific N tile

当前参数：

```text
AITER_TDM_TILE_N / AITER_TDM_TILE_N2
AITER_TDM_N_WARP / AITER_TDM_N_WARP2
```

机制：

- N128 增加 WG 和潜在 residency，但 A/SA 被更多 N tile 重复请求。
- N512 减少 A/SA 重复和 fixed stage work，但增加 B stage、accumulator、VGPR/LDS。
- A4W4 payload byte宽对称不代表 square tile最优；当前 M受 route限制，N应按实际 reuse 和
  resources 选择。

最小单变量实验：

```text
G1: N1=128,256,512；N2显式固定256
G2: N2=128,256,512；N1显式固定256
```

保持共同 M winner、K256、w1x4、b2、cluster1、wpt2。每个候选先编译并记录 final
VGPR/LDS/spill。

预期改善对象：

- A/SA logical request重复；
- grid并行度；
- TDM/DS fixed cost；
- on-chip feed 与 residency平衡。

风险：

- N512 可能显著提高 accumulator VGPR 和 LDS；
- N128 增加 descriptor/barrier、A request和静态 WG；
- tile 变化可能改变 compiler schedule，必须重新确认 symbol。

PMC/benchmark 成功判据：

- 对应单阶段和 E2E 达通用门槛；
- physical bytes、request geometry或residency至少有一项解释收益；
- 无 spill，clock不出现异常下降。

停止/回滚：

- 只有 logical model改善而 physical bytes/time不变；
- resources 降级抵消收益；
- 回滚 N256。

### P5：验证现有 A multicast，再决定 SA

当前位置：

- `grouped_gemm_mxfp4.py::_select_cluster_n`
- `mxfp4_preshuffle_gfx1250_tdm.py::a_mcast_mask`
- A 的 `add_tdm_loads(..., wg_mask=a_mcast_mask)`

机制：

- cluster peers 共享 A row和 expert；
- A upstream request可由 peers复用；
- 当前只 A multicast，SA 是第二个独立实验；
- B/SB 保持 unicast。

最小单变量实验：

```text
cn1 vs cn3  # G1 24 N tiles可整除；G2 28不可整除并回退1，形成G1-only诊断
cn1 vs cn2
cn1 vs cn4
```

cn2/cn4 当前会同时影响两阶段。若需要 G2-only，先新增 stage-specific `cluster_n2`，
不能伪装为现有能力。

预期改善对象：

- GL1->GL2/HBM 的 A upstream bytes；
- A request credit和功耗；
- 不减少本地 LDS payload。

风险：

- peer address/mask/issue 数不一致可 hang；
- timeout/downgrade；
- cluster co-residency和调度弹性；
- current test logical summary拒绝 `cluster_n>1`，不能伪填物理数字。

PMC/benchmark 成功判据：

- catalog 中实际 multicast request生效；
- timeout/downgrade足够低；
- upstream bytes下降，且单阶段/E2E达通用门槛；
- 所有 route case无 hang。

只有 A winner 后再做独立 SA 实验：

```text
SA add_tdm_loads(..., wg_mask=a_mcast_mask)
```

停止/回滚：

- A bytes已被其他层复用、物理流量不降；
- timeout、hang或duration回退；
- `AITER_FLYDSL_MXFP4_CLUSTER_N=1`。

### P6：buffer depth 与 next-stage carry

当前位置：

- stage-specific buffer env；
- shared `AITER_TDM_NEXT_STAGE_PREFETCH`；
- `next_stage_on = next_stage_prefetch and num_buffers>=3`；
- `steady_post`、`steady_mid`、`pipeline_fence`。

最小单变量顺序：

```text
1. b2 vs b3；next-stage=0；另一阶段显式b2
2. 只有b3胜出后，b3 next-stage=0 vs 1
3. b4只作资源边界反证
```

G1 可设 b3、G2 b2，再切 shared next-stage；G2 因 b2 保持 off，因此该实验可隔离 G1。

预期改善对象：

- memory latency hiding；
- tensor wait；
- prologue/steady/drain overlap。

风险：

- LDS导致 resident WG下降；
- 更多 outstanding request增加 direct fallback/credit stall；
- next-stage carry增加固定 rmem live range。

PMC/benchmark 成功判据：

- wait/latency相关 PMC下降；
- duration达通用门槛；
- residency、direct ratio、VGPR/LDS没有抵消收益。

停止/回滚：

- b3慢于b2或仅靠 profiler duration看似改善；
- b4 residency损失；
- 回滚 b2、next-stage off。

### P7：wpt 与 weighted owner

第一步只用现有：

```text
AITER_FLYDSL_NUM_WAVES_PER_TENSOR_TDM=1,2,4
```

当前 row 的 CSV 为 -1，所以环境值会被读取。

机制与风险：

- wpt1降低每 wave descriptor数但放大 payload不均衡；
- wpt4平衡每种 tensor 的 payload，但每 wave四个 descriptor可能触及硬件每-wave 3限制；
- wpt2是当前 baseline。

最小实验：1/2/4，其他参数完全固定。

只有 PMC/时间证明 owner imbalance 后才实现 weighted ownership：

- 按 A/B/SA/SB bytes选择不同 owner cardinality；
- 为每 wave计算实际 outstanding count；
- 不再使用全局单值 `TDM_PER`；
- 每个新 owner map独立 symbol/cache flag。

PMC/benchmark 成功判据：

- TDM issue/backpressure或 barrier相关总时间下降；
- payload/direct ratio不恶化；
- E2E达高风险 3% 门槛。

停止/回滚：

- wpt4 descriptor stall增加；
- weighted版本只增加控制流/descriptor；
- 回滚 wpt2。

### P8：stage-specific K tile/request geometry

最小实验：

```text
G1: K1=128,256,512；K2固定256
G2: K2=128,256,512；K1固定256
```

对应 stage 数：

```text
G1 K7168: 56 / 28 / 14
G2 K3072: 24 / 12 / 6
```

机制：

- K128 的 FP4 A row只有64 B，可能失去 direct-copy 条件并增加 fixed stage count；
- K512 的 A row为256 B，可能提高 payload/entry并减少 stage，但显著增加 LDS；
- B/scale实际 request仍须 PMC确认。

风险：

- K128 descriptor/barrier翻倍；
- K512 b2 LDS/residency显著下降；
- math K元素数不能保证 request宽度。

PMC/benchmark 成功判据：

- 128/256 B request、direct/indirect和wait变化与时间一致；
- 单阶段/E2E达通用门槛；
- 无 spill和不可接受 residency损失。

停止/回滚：request geometry或residency恶化，回滚 K256。

### P9：仅由证据触发的 LDS/compiler 微调

候选必须拆成独立实验：

1. G1 output `STORE_PAD=0/16`：
   - G2 已是 16，不重复开发；
   - 只在 G1 output DS conflict/stall 有证据时测试；
   - 保持 inner OOB，确认 physical output bytes不变。
2. `AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=0/1`：
   - 新进程、独立 dump；
   - symbol相同，必须比较 ISA hash/order。
3. `emit_hints` flag 与 `MMA_GROUP=2/4/8`：
   - 当前函数立即 return；
   - 只在单独 flag 下启用。
4. `addr_keepalive`、`vgpr_keepalive`：
   - 各自单独测试；
   - 检查 VGPR live range和spill。
5. `TILES_PER_GROUP=1/4/8/16`：
   - 当前硬编码16且无 env；
   - 关注 channel分布、A reuse和 sentinel path。

统一成功判据：对应 PMC改善且 wall time达通用门槛。统一停止条件：只有静态 ISA“看起来更好”
而 wall time不变、VGPR/clock恶化或 correctness变化。

### P10：高风险 fusion、tile-owner、persistent 与 consumer

按以下顺序，不能并行混做。

#### P10a：A4W4 SiLU -> FP4 quant fusion

当前 A8W4 只有 FP8 fused output；A4W4 必须新增明确 output mode。

语义要求：

```text
F32 accumulator -> SiLU -> BF16 round -> per-32 FP4 quant
```

默认必须保留当前 BF16 intermediate rounding，再与独立 quant 的 valid payload/scale 比较。

目标：删除 GEMM1 BF16 write、A2 BF16 read和一个 launch。成功以 E2E 至少 5%、
无 spill、payload/scale correctness 为门槛。

#### P10b：tile-owner map

由 psum阶段生成每个 static M tile 的：

```text
expert id
valid rows
```

替代每个 WG 的 96-entry fixed binary search，并为 device work queue提供 metadata。
先比较 map生成成本 + G1/G2 + E2E；低于 2% 停止。

#### P10c：GEMM persistent work queue

当前 dormant persistent prep fusion不是 GEMM queue。真正候选需要：

- device actual task count；
- fixed workers；
- 每次安全重置 accumulator/descriptor/TENSORcnt；
- graph-safe；
- 与 cluster 分开验证。

只有 sentinel/lookup fixed cost被 PMC/shape scaling证明显著时进入。

#### P10d：consumer fusion

当前 gather-reduce 已融合权重乘和 reduction。只有 gather 稳定占 E2E 至少 5 us 才原型化：

- weighted partial：GEMM2 epilogue乘权重，gather只求和；
- direct atomic：需要 contiguous row -> token/slot inverse map和 FP32 temporary。

若 atomic traffic、冲突或数值顺序抵消收益，立即停止。

## 8. Cache、symbol、descriptor 与 AOT 纪律

当前事实：

- `TDM_DESCRIPTOR_VERSION=1`。
- `cache_tag` tuple包含 K、tile、warp、buffer、dtype mode、expert、activation、bias、
  descriptor version、quant、cluster、effective next-stage 和 wpt。
- tuple 随后仅 `_ = cache_tag`，不是可独立验证的外部 JIT cache API key。
- `_kname` 编码 tile/warp/buffer/K/expert/act/bias/quant/cluster/effective prefetch/wpt。
- compiler expert scheduling mode不在 symbol。
- TDM grouped path当前 JIT-only；`grouped_moe_gfx1250.py` 明确有 AOT coverage TODO。

新增策略要求：

1. 作为真实 `Constexpr` launcher 参数进入 specialization，而不是只加一个未消费的 Python tuple。
2. 进入 `_kname` 后缀，避免 profiler/parser 混淆。
3. 更新测试 `_TDM_KERNEL_RE`；该 regex 使用 `search`，未知 suffix 会被截掉并误归为 baseline。
4. 新进程和独立 dump 目录验证实际 final ISA。
5. 改变外部 A/SA/B/SB payload、scale或 descriptor physical layout 时：
   - bump `TDM_DESCRIPTOR_VERSION` 或等价 producer/consumer layout version；
   - producer和consumer同版本；
   - 不能复用旧 layout cache。
6. 只改变内部 output LDS padding、控制流或 schedule时，不机械 bump external layout version，
   但仍需独立 compile flag和 symbol。
7. 当前不要虚构 TDM AOT job key；AOT coverage真正接入后再同步相同 specialization identity。

建议 suffix：

```text
_pm16c       partial M16 compute
_pm16ds      partial M16 DS
_spad16      G1 output pad
_ownbal      weighted owner
_sched       schedule hints
_mtg4        TILES_PER_GROUP=4
_amajor_v2   A physical layout v2
_omap        tile-owner map
_persist512  GEMM persistent workers
```

## 9. 硬件 agent 反馈契约

每个实验返回：

```text
source branch/HEAD
exact command and complete env
ROCm/FlyDSL/rocprof/device/stepping
actual timing mode

exact GEMM1/GEMM2/aux-kernel symbols
tile M/N/K per stage
wave grid/buffer/wpt/cluster/next-stage per stage
all compile flags

unprofiled E2E/GEMM1/GEMM2 median/min/max/CV
gfx clock median/range or UNKNOWN
power/temperature or UNKNOWN

logits_diff, candidate-vs-baseline error
max_abs/max_rel/normalized L2 when needed
NaN/Inf count

dispatch count, grid, workgroup
SGPR/VGPR/LDS/private/scratch/spills
static opcode counts and ISA hash

PMC semantic -> actual counter name/block/description
physical HBM/GL2/GL1/TDM/LDS/SQ/residency values or UNKNOWN
profiler perturbation

KEEP / REJECT / RETEST / UNKNOWN
reason and rollback config/symbol
artifact paths
```

报告必须分开：

- logical requested bytes；
- unique-address model；
- physical counter bytes；
- executed FLOP；
- useful FLOP。

## 10. 首轮执行清单

首轮严格按新优先级执行。

1. 在目标执行环境再次记录 repo root、branch、HEAD 和只读 status；分支不符立即停止。
2. 用原始 `run_gemm_a4w4.sh` 建立 5 个独立进程的随机数据 baseline，记录 E2E、
   isolated GEMM1/GEMM2、correctness、clock和 CV。
3. 用 kernel trace/stats 和数据库确认实际两个 symbol、dispatch、grid、SGPR/VGPR/LDS/
   private/scratch；不沿用本文中的 artifact provenance。
4. 运行目标安装的 `rocprofv3 --list-avail`，保存完整 catalog，并按第 6.3 节建立
   semantic-to-counter mapping。
5. 分 pass 采 baseline PMC；缺失语义标 `UNKNOWN`，不猜名字。
6. 只实施 P1：
   - 在 A4W4 A2 quant 调用传 `topids_to_rows`、`source_topk=0`、
     `num_valid_routes=_ep_nvr`；
   - 不改任何 GEMM tile、pipeline或kernel源码；
   - 先做 valid A2 payload/scale 对照，再做 E2E A/B。
7. P1 达标后才做 P2：
   - `(M1,M2)=(64,64)` vs `(32,32)`；
   - 显式固定全部其他 stage 参数；
   - 同时跑 balanced、random和active-expert覆盖。
8. 只有 P2 证明 small-M compute有价值但 M32泛化失败，才实现 P3 的
   wave-uniform partial-M16 compute；第一版不改 TDM/DS。
9. P3 之后按顺序做 P4 stage-specific N，再做 P5 A cluster；每次只保留一个 winner。
10. buffer/next-stage、wpt、K、schedule、layout、persistent和fusion均留到前五项有完整结果后。

任何 hang、memory fault、correctness失败、spill、不可解释 clock下降或 physical traffic放大，
立即回滚到上一 retained symbol/config。
