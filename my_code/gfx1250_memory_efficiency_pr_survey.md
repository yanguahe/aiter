# gfx1250 / MI450 / MI455 访存效率相关 PR 调研

调研日期：2026-10-01

本文使用 `gh search prs`、`gh pr view --json ...` 扫描以下两个仓库中已经合入和仍处于 open 状态的 PR：

- [ROCm/aiter](https://github.com/ROCm/aiter)
- [ROCm/FlyDSL](https://github.com/ROCm/FlyDSL)

目标是整理可以用于 gfx1250 MoE GEMM1/GEMM2 以及邻近 kernel 的访存优化经验。筛选范围包括数据布局、实际搬运字节数、事务合并、TDM/async copy、LDS、cache policy、multicast、prefetch、output store，以及访存与计算重叠。通常不收录只有功能使能或配置更新的 PR，除非其中包含有价值的测量结果或失败经验。

PR 状态以 2026-10-01 的 GitHub 状态为准。GitHub 搜索依赖索引，因此本文是经过 PR 正文、commit 标题和 changed files 二次核实的高可信清单，但不声称覆盖所有历史 PR。

## 硬件背景

本地 MI455X white paper 和 CDNA5 ISA 给出了这些优化所依赖的主要硬件条件：

- HBM4 峰值带宽最高约 23.3 TB/s。
- 192 MB L2，GPU 合计 L2 带宽约 54 TB/s。
- 每个 WGP 有 320 KiB LDS 和 64 KiB WGP vector cache。
- 每个 WGP 有一个 TDM，可在 external memory 和 LDS 间执行异步结构化搬运，并支持 multicast。
- gfx1250 使用 wave32、split memory wait counters、scaled WMMA、cluster launch 和 cluster barrier。

因此，访存优化通常需要至少改善一项：搬运字节数、transaction coalescing、cache reuse、并发 outstanding request 数、latency overlap、LDS bank conflict、同步等待或 occupancy。

## ROCm/aiter：已合入 PR

| PR | 范围 | 与访存效率相关的工作及经验 |
|---|---|---|
| [#2968](https://github.com/ROCm/aiter/pull/2968) | MoE metadata | 为 routing、top-k 和 reduce metadata kernel 增加 gfx1250 TDM 路径。经验是 TDM 不只适用于 GEMM 主数据，也可以消除快速 GEMM 周围由标量或普通 vector memory 搬运形成的尾部开销。 |
| [#3293](https://github.com/ROCm/aiter/pull/3293) | MoE GEMM | 优化 A8W4 prefetch pipeline，并使用 `int16` gather index 减少 TDM gather 指令数量，同时改善数据重叠和 descriptor/index 流量。 |
| [#3816](https://github.com/ROCm/aiter/pull/3816) | A8W8 blockscale GEMM | 拆分 bandwidth-bound 和 compute-bound kernel，增加 preshuffle 版本，更新 tensor descriptor 和 pipeline，并让 A/B scale 不再经过 LDS。直接加载 scale 可以减少 LDS 流量和容量压力。 |
| [#3851](https://github.com/ROCm/aiter/pull/3851) | grouped MoE 输入准备 | 融合 route map、prefix-sum、quant 和 scatter，并支持 contiguous-M 输出。主要收益来自删除中间 global-memory round trip 和多个 kernel launch 边界。 |
| [#4089](https://github.com/ROCm/aiter/pull/4089) | QK norm/RoPE/quant | 减少 divergence，通过 register memory 合并 store，一个 workgroup 处理多行，提前加载 cos/sin，并删除重复的 NEOX partner load。体现了 coalescing、load hoist 和 memory-level parallelism 的组合使用。 |
| [#4196](https://github.com/ROCm/aiter/pull/4196) | MoE GEMM setup | 改善 occupancy，预加载 kernel arguments，并重新调参。kernarg/scalar-load 对长 kernel 占比较小，但在 decode 和短 K kernel 中可能成为显著 prologue latency。 |
| [#4281](https://github.com/ROCm/aiter/pull/4281) | QK norm/RoPE/quant | 为 BF16 prefill 增加 TDM deep prefetch。报告从约 `630 us / 6.8 TB/s` 提升到 `319 us / 13.5 TB/s`，说明 shape 足够大时，提前发出多级搬运可以有效隐藏 memory latency。 |
| [#4527](https://github.com/ROCm/aiter/pull/4527) | FlyDSL GEMM common | 统一 gfx1250 GEMM，并重构 LDS load helper。该 PR 没有报告性能提升，但它使后续布局和调度优化能够共享同一套 LDS 搬运实现。 |
| [#4562](https://github.com/ROCm/aiter/pull/4562) | Gluon MoE multicast | 使用 cluster launch 和 TDM multicast，让一次 load 写入多个 CTA 的 LDS。PR 明确记录初始 decode/prefill 没有性能提升，说明 multicast 减少上游重复流量后，cluster geometry、barrier、CTA 工作划分和 owner-wave 偏斜仍可能抵消收益。 |
| [#4826](https://github.com/ROCm/aiter/pull/4826) | A4W4 MoE layout | 切换到新的 preshuffle API，并把 `SCALE_KWIDTH` 从 8 改为 4。说明 payload 和 scale layout 是 producer/consumer 共同遵守的 ABI，不能只在 GEMM 内局部修改。 |
| [#4849](https://github.com/ROCm/aiter/pull/4849) | MXFP8-128 GEMM | 增加 256x256 compute-bound WMMA kernel、TDM multicast 和 split-K。经验是大型 shape 需要与 memory-bound 路径不同的 pipeline，单一 schedule 通常无法覆盖全部工作区间。 |
| [#4984](https://github.com/ROCm/aiter/pull/4984) | MegaMoE dispatch wire | 在 EP dispatch 前完成量化，使每个 source token 只量化一次，并使用 FP8/FP4 wire 替代 BF16。hidden size 7168 时每 token 从 14,336 B 降到 FP8 的 7,424 B 或 FP4 的 3,840 B。这是“先减少字节，再优化搬运指令”的典型案例。 |
| [#5147](https://github.com/ROCm/aiter/pull/5147) | EP 测量 | 把 benchmark 扩展到 production 实际使用的 FP4 dispatch wire，避免根据 BF16 wire 的性能得出错误结论。主要价值是确保优化针对真实数据表示和 bandwidth regime。 |
| [#5273](https://github.com/ROCm/aiter/pull/5273) | dynamic per-group quant | 优化 gfx1250 group-128 quant，并在较大 token 数上获得明显带宽提升。量化输出 packing 和 scale store 会直接影响 GEMM producer 的整体访存成本。 |
| [#5274](https://github.com/ROCm/aiter/pull/5274) | MoE A producer | 从 one-warp-per-route 改为 one-warp-per-source-token，消除最多 `topk` 倍的重复读取和重复量化；使用 zero-length descriptor 代替 divergent branch，并重新调节 GEMM2 以减少 A-scale reread。报告 quant kernel 从 `125.6 us` 降到 `79.5 us`，约 `1.58x`。该 PR 后来被 #5581 revert，因此还必须验证整个 pipeline 合约。 |
| [#5313](https://github.com/ROCm/aiter/pull/5313) | grouped MoE auxiliary kernels | 将原来只在一个 CU 上执行的 route remap 改成 grid-stride；把 route rows/weights 缓存在 LDS；用一个 flat resource 替代循环内 descriptor 构造；增加 token-multidest quant。报告 psum `227.7 -> 5.1 us`、gather-reduce `209.0 -> 143.5 us`，quant 约 13.1 TB/s。该 PR 还测得现有 payload store 已达到线性 dwordx4 stream 的约 99%，继续优化必须减少字节数，而不是只更换 store 指令。 |
| [#5373](https://github.com/ROCm/aiter/pull/5373) | attention compression | 将 runtime K loop 转成固定 trip-count，使 LLVM 能完全 unroll 并批量调度 K loads，同时提前 state-cache load。可迁移经验是删除 `load -> compute -> load` 的串行依赖链。 |
| [#5406](https://github.com/ROCm/aiter/pull/5406) | MXFP8-128 A preshuffle | 增加 A-preshuffled GEMM、persistent N tiles、cluster fallback、B-scale multicast/opsel、kernarg preload，以及 fused/standalone split-K reduction。它把 A layout、scale reuse、cluster reuse 和 reduction traffic 作为同一个设计问题处理。 |
| [#5447](https://github.com/ROCm/aiter/pull/5447) | MegaMoE EP dispatch | 用 TDM bulk movement 替代 per-lane payload stores；wave 内去重 route；用 LDS histogram 按 `(block, peer)` 预留 remote slot；将 metadata 排成 destination-ordered SoA，使 metadata 也能形成连续 TDM run。 |
| [#5507](https://github.com/ROCm/aiter/pull/5507) | MoE ScaleA producer | 先紧凑写 scale，再重建 16-row-interleaved layout，避免每条 cache line 只写 4 B 有效数据。quant 从 `143.1 us` 降到 `75.1 us`。同时记录了重要系统效应：persistent 单 launch 虽减少 launch，但让后续 GEMM 慢约 30 us；isolated kernel winner 不一定是 whole-layer winner。 |
| [#5581](https://github.com/ROCm/aiter/pull/5581) | Revert / 失败经验 | Revert #5274。减少重复流量的方向本身合理，但 producer/consumer layout、persistent scheduling 和 end-to-end 行为可能使局部加速无法落地。 |
| [#5635](https://github.com/ROCm/aiter/pull/5635) | sparse-prefill ASM | 避免 padded CSR tail replay。短或 ragged case 提升约 11%-44%，长且对齐的 case 基本不变。经验是应在 load/store schedule 发出前消除 padding 工作，而不是发出后再 mask。 |
| [#5731](https://github.com/ROCm/aiter/pull/5731) | attention compressor output | 直接写最终 cache layout 的 packed FP4 payload 和 E8M0 scale，删除 BF16 temporary 和额外 per-layer quant kernel。该思路可直接类比 GEMM1 到 GEMM2 的 handoff。 |

## ROCm/aiter：尚未合入 PR

| PR | 范围 | 与访存效率相关的工作及经验 |
|---|---|---|
| [#5465](https://github.com/ROCm/aiter/pull/5465) | MoE ScaleA layout | 尝试删除 row-major A-scale 备用布局，只保留 interleaved layout。原因是 producer 和 consumer 分别推导布局选择，在 EP 路径出现不一致。row-major 写入虽然更 coalesced，但两套 ABI 会显著增加正确性风险。 |
| [#5704](https://github.com/ROCm/aiter/pull/5704) | MegaMoE stage1 fusion | 将 stage1 融入 compact-plan/dispatch pipeline，目标是在 route、quant 和 GEMM consumer 间保持数据驻留，删除独立 transfer 和 launch。 |
| [#5752](https://github.com/ROCm/aiter/pull/5752) | A4W4 MoE GEMM1/GEMM2 | 对选定的 256x256x256 prefill tile 启用 A/ScaleA preshuffle、cluster multicast/prefetch 和 output-store overlap，同时让 decode/untuned shape 保持 row-major fallback。operator benchmark 中 GEMM 得到提升，但 serving 结果是 TTFT 改善、TPOT/throughput 略退化，说明必须计入 producer 成本和 whole-model 影响。 |
| [#5818](https://github.com/ROCm/aiter/pull/5818) | MXFP8 1x32 GEMM | 增加 m32k4/n32k4 bpreshuffle layout、fused quant producer，以及按 shape 选择 ASM/FlyDSL 的 tuned dispatch。关键是不同 backend 共用完全一致的 producer layout。 |
| [#5899](https://github.com/ROCm/aiter/pull/5899) | decode compact-plan overlap | 在当前 dispatch 返回前，使用 side stream 启动下一层 compact plan。目标是覆盖短 decode GEMM 难以隐藏的 payload-copy completion，与短 K GEMM2 的固定 setup/drain 很接近。 |
| [#5978](https://github.com/ROCm/aiter/pull/5978) | MLA segmented cache | 针对低于可用 HBM 带宽的 memory-bound kernel：每 block 处理多个 head/token，先发全部 load 再 store，用 lane exchange 删除第二次 NEOX load，保持 descriptor wave-uniform，以 OOB offset 代替 divergent branch，并用 TDM staging Q。报告 `70.0 -> 27.6 us`，逻辑流量约 12 TB/s。 |
| [#5985](https://github.com/ROCm/aiter/pull/5985) | group quant | 使用完整 128 B row store、interleaved 32 B load、DPP exchange，并在 LDS 中聚合 shuffled scale 后按 word store；按 shape 在普通路径和 TDM tile 路径间选择。一个重要结论是 TDM staging 并非总是更快，generic FP8/FP4 路径移除 TDM 后反而更快。 |

## ROCm/FlyDSL：已合入 PR

| PR | 范围 | 与访存效率相关的工作及经验 |
|---|---|---|
| [#278](https://github.com/ROCm/FlyDSL/pull/278) | gfx1250 基础支持 | 引入 gfx1250 WMMA GEMM、异步 TDM pipeline、MXFP4 scale preshuffle、cluster launch 和 multicast，是后续 AITER 优化所依赖的基础能力。 |
| [#331](https://github.com/ROCm/FlyDSL/pull/331) | unified FP8/FP4 GEMM | 统一 MXFP4/MXFP8/A8W4 kernel，抽取 LDS helper，通过 K 维 descriptor advance 减少重复构造，并启用 expert scheduling 减少 WMMA pipeline 的错误依赖 stall。 |
| [#340](https://github.com/ROCm/FlyDSL/pull/340) | 256x256 FP4 GEMM | 降低 register/LDS 压力，使用更友好的 quadrant 顺序，将 TDM issue 插到 compute 中间，并让 output epilogue 复用已经失效的 input LDS。它同时优化 layout、overlap 和 LDS lifetime。 |
| [#483](https://github.com/ROCm/FlyDSL/pull/483) | two-stage MoE GEMM | 重构 MXScale 并 hoist TDM state；把 sorted token IDs 缓存在 LDS，以一次 `ds_read_b32` 替代每行 global load。说明高复用的小 metadata 也值得进入 LDS。 |
| [#484](https://github.com/ROCm/FlyDSL/pull/484) | TDM descriptor update | 新增只修改 descriptor 中 K-dependent address lane 的 helper，避免每轮重建完整 descriptor；同时提供 carry-safe 64-bit 版本，避免跨 4 GiB 时静默地址回绕。 |
| [#533](https://github.com/ROCm/FlyDSL/pull/533) | FP8/FP4 GEMM pipeline | 在 WMMA schedule 内按 quadrant stream B fragment，而不是先 stage 整个 B tile；把 scale 改为 `buffer_load -> LDS`，释放 TDM slot；重新排列 callback 以隐藏 load/store latency。 |
| [#608](https://github.com/ROCm/FlyDSL/pull/608) | deep GEMM pipeline | 增加 wait placement、panel scheduling、instruction prefetch、cluster fence/signal overlap、避免冲突的 segmented LDS、直接 buffer-to-VGPR scale load，以及 multicast early timeout。这是较完整的 trace-driven memory pipeline 优化案例。 |
| [#649](https://github.com/ROCm/FlyDSL/pull/649) | PTPC/strided GEMM | 允许 runtime M 和 A/C stride，删除每次调用的 padding allocation 和 memcpy；PTPC 跳过 scale TDM/LDS，并把 epilogue scale load 隐藏在最后几条 WMMA 后面。 |
| [#679](https://github.com/ROCm/FlyDSL/pull/679) | scale layout/store policy | 让 B-scale preshuffle 与 tile 无关；A-scale 默认走 VGPR buffer-load ring，较大 M 可走 shuffled-TDM；根据完整 tile、partial tile 或 split-K 自动选择 TDM store、clipped buffer store 或 atomic store。 |
| [#705](https://github.com/ROCm/FlyDSL/pull/705) | decode GEMM | 在 tail 中保留 A-scale VGPR prefetch ring，提前调度 row-major LDS-to-VGPR load，并延后 TDM fence signal，增加 overlap。小 M A8W4 从约 5 us 降到 4.5 us。 |
| [#750](https://github.com/ROCm/FlyDSL/pull/750) | cluster/TDM multicast | 修复 cluster launch，并增加 TDM multicast device test 和 bandwidth benchmark，为判断某个 shape 是否真正受益于 multicast 提供基础设施。 |
| [#830](https://github.com/ROCm/FlyDSL/pull/830) | compiler atoms | 将 scaled WMMA 和 1-5D TDM copy atom 接入标准 layout API，使 kernel 可以表达 scale state 和完整 Global/LDS tile DMA，而不需要手工构造 raw descriptor。 |
| [#963](https://github.com/ROCm/FlyDSL/pull/963) | LDS copy abstraction | 使用 copy atom 替代 raw LLVM LDS helper，并统一 A8W8 kernel。主要是可维护性改动，但为后续 layout 和 schedule 优化提供统一的 LDS 搬运实现。 |
| [#1008](https://github.com/ROCm/FlyDSL/pull/1008) | direct global-to-LDS copy | 为 FP16/BF16 preshuffle GEMM 启用异步 `gmem -> LDS`，每个 K tile 删除 32 条 `ds_write`，MFMA utilization 从 73.3% 提升到 78.8%。PR 明确指出这是 issue-rate/overlap 提升，而不是 bandwidth 提升。 |
| [#1013](https://github.com/ROCm/FlyDSL/pull/1013) | tiled TDM API | 增加 `make_tiled_tdm_atom` 和 `tdm_partition`，通过 layout algebra 描述 TDM ownership 和 tile geometry，减少手工 descriptor slicing。 |
| [#1078](https://github.com/ROCm/FlyDSL/pull/1078) | BF16 GEMM prefetch | 对 wide/tile-M=64 GEMM 更早发出 TDM prefetch，并加入 TDM split 和 XDL arbitration 实验选项。经验是移动 issue point 往往比单纯增加 buffer 更有效。 |
| [#1131](https://github.com/ROCm/FlyDSL/pull/1131) | preshuffle indexing | 用共享 layout algebra 和 basis stride 替代重复 div/mul 地址计算，在若干 gfx950 MX kernel 上报告约 4.8%-5.6% 提升，但随后因 correctness 问题被 #1189 revert。必须覆盖 ragged、async 和 tail layout。 |
| [#1189](https://github.com/ROCm/FlyDSL/pull/1189) | Revert / 失败经验 | 在 MI35X async preshuffle 测试出现 `logits_diff ~0.95` 后 revert #1131。地址代数简化可能静默改变 cooperative DMA ownership，性能测试不能替代数据级 correctness。 |
| [#1200](https://github.com/ROCm/FlyDSL/pull/1200) | 测量正确性 | 将 TDM multicast benchmark 从包含约 40 us host launch 的计时改为 GPU device time。修正后 multicast 结果稳定为约 1.15-1.16x，说明错误的计时边界会完全掩盖访存优化。 |

## ROCm/FlyDSL：尚未合入 PR

| PR | 范围 | 与访存效率相关的工作及经验 |
|---|---|---|
| [#914](https://github.com/ROCm/FlyDSL/pull/914) | LDS address lowering | 阻止 compiler 把常量 LDS offset 合并进 runtime index，使常量仍可编码进 `ds_read ... offset:` immediate。报告案例中可消除 44 个 VGPR spill 和 71 次 scratch operation。 |
| [#971](https://github.com/ROCm/FlyDSL/pull/971) | compute-bound gfx1250 GEMM | 增加手工调度的 256x256 A8W8/A8W4/A4W4 kernel，使用 planar LDS、独立 B LDS segment、每个 ring slot 的 persistent TDM descriptor，以及显式 `s_wait_dscnt`/READY fence。与当前 256x256 MoE kernel 很直接相关。 |
| [#1125](https://github.com/ROCm/FlyDSL/pull/1125) | cross-tile LDS latency hiding | 将每个 K tile 的最后一个 WMMA K-step 延迟到 tile boundary 之后，使下一 tile 的 step-0 LDS read 隐藏在上一 tile 的 WMMA 后面。compute-bound clock regime 报告 7.5%-8.0% 提升，而 memory-bound 状态基本中性。 |

## 可复用的优化规律

### 1. 先减少字节，再优化指令

- 每个 source token 只量化一次，而不是每个 route 一次：AITER #5274/#5313。
- dispatch wire 使用 FP4/FP8，而不是 BF16：AITER #4984。
- 直接产生 consumer 最终 FP4/scale layout：AITER #5731。
- 融合 route、prefix-sum、quant、scatter，删除 intermediates：AITER #3851。
- 不处理 padded/tail 无效数据：AITER #5635。

### 2. 将 payload 和 scale layout 视为同一个 ABI

- A preshuffle：AITER #5752/#5406。
- tile-independent B scale 和 A-scale VGPR ring：FlyDSL #679。
- compact scale write 后重建：AITER #5507。
- 用单一 interleaved layout 避免 producer/consumer 决策不一致：AITER #5465。

### 3. 提高 memory-level parallelism

- 先发出多条 load，再进入 store：AITER #5978。
- 在 WMMA schedule 中 stream B fragment：FlyDSL #533。
- 提前 TDM issue：AITER #4281、FlyDSL #1078。
- 跨 K tile 携带计算以覆盖下一 tile 的 LDS read：FlyDSL #1125。
- side stream overlap compact plan：AITER #5899。

### 4. 并非所有 operand 都应经过 LDS/TDM

- A/B scale 不经过 LDS：AITER #3816。
- scale 使用 buffer-load-to-VGPR ring：FlyDSL #533/#608/#679/#705。
- 普通连续 load/store 比 TDM staging 更快时应按 shape 选择：AITER #5985。

TDM 不是天然快于普通 buffer load；需要考虑 transfer size、descriptor setup、inflight 深度、barrier 和可用于隐藏 latency 的计算量。

### 5. Multicast 必须和 cluster 工作划分匹配

FlyDSL #750 提供 cluster/TDM multicast 机制及正确计时。AITER #4562 则说明只打开 multicast 不一定产生性能收益。必须保证 cluster 内确实共享较大的相同 tile，同时控制 barrier 成本、M/N 分片、owner-wave 发射偏斜和 LDS lifetime。

### 6. 减少 descriptor 和地址生成

- 只修改 TDM descriptor 中变化的地址字段：FlyDSL #484。
- descriptor 保持 wave-uniform：AITER #5978。
- 保留 LDS constant offset，使其进入指令 immediate：FlyDSL #914。
- 循环中复用一个 flat resource：AITER #5313。

这类优化对短 K GEMM2 特别重要，因为 scalar setup 很难被 compute 摊薄。

### 7. 将 output completion 当成独立 pipeline stage

短 K GEMM2 在 input prefetch 已经有效后，最终 LDS write、workgroup synchronization、TDM store 和 `s_wait_tensorcnt` 可能成为主导：

- 复用失效的 input LDS：FlyDSL #340。
- 直接产生量化后的 consumer layout：AITER #5731/#5752。
- 将下一 compact plan 与当前 output drain 重叠：AITER #5899。
- 把 output store 分段并穿插独立 epilogue 工作：AITER #5752。

### 8. 必须测完整 pipeline

三个重要反例：

- AITER #5507：局部更快的 persistent producer 让后续 GEMM 变慢。
- AITER #5274/#5581：显著减少重复流量的实现后来仍被 revert。
- FlyDSL #1131/#1189：地址计算减少且局部性能提高，但 async/ragged correctness 失败。

每个候选都应验证 random correctness、producer/consumer 配对、相邻历史版本和 MoE end-to-end 性能。

## 对当前 MoE GEMM1/GEMM2 的建议优先级

### GEMM1

1. 对访问连续的 ScaleA/ScaleB 尝试 direct VGPR load ring，参考 FlyDSL #533/#608/#679/#705。
2. 更早发出下一批 input TDM，并用独立 WMMA/VALU 覆盖 wait，参考 AITER #4281 和 FlyDSL #1078/#1125。
3. A/ScaleA preshuffle 只在 producer 成本能被摊薄的 shape 上启用，参考 AITER #5752 和 #5465。
4. 重新检查 owner-wave 和 multicast geometry；AITER #4562 表明减少请求数量并不必然减少 kernel latency。

### GEMM2

1. 优先处理最终 LDS-write/barrier/TDM-store drain，而不是继续调整 B input cache hint。
2. 复用 dead input LDS，并更早完成 output setup，参考 FlyDSL #340。
3. 将下一层 compact-plan/dispatch 与当前 output drain 重叠，参考 AITER #5899。
4. 直接写 consumer 所需 packed layout，避免 BF16 intermediate，参考 AITER #5731/#5752。
5. hoist scalar descriptor，并保留 LDS immediate offset，参考 FlyDSL #484/#914。

## 调研复现命令

主要只读命令如下：

```bash
gh search prs gfx1250 --repo ROCm/aiter --match title --merged --limit 1000
gh search prs gfx1250 --repo ROCm/aiter --match title --state open --limit 1000
gh search prs gfx1250 --repo ROCm/FlyDSL --match title --merged --limit 1000
gh search prs gfx1250 --repo ROCm/FlyDSL --match title --state open --limit 1000

gh search prs TDM --repo ROCm/FlyDSL --match title --merged --limit 1000
gh search prs prefetch --repo ROCm/FlyDSL --match title --merged --limit 1000
gh search prs LDS --repo ROCm/FlyDSL --match title --merged --limit 1000

gh pr view <PR> --repo ROCm/aiter \
  --json number,title,state,mergedAt,url,body,files,commits
gh pr view <PR> --repo ROCm/FlyDSL \
  --json number,title,state,mergedAt,url,body,files,commits
```

