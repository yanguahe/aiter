# E64/T1536 GEMM1 四级共享 A128/ScaleA128 thread trace 分析

## 结论与性能口径

四级 ring 的输入 LDS 已按 `4 × (A16 + B32 + ScaleA1 + ScaleB2)` 实现为 **204 KiB**；加两个 output windows 后，实际 ELF 总 LDS 是 **221 KiB（226304 B）**。上下半窗口在不同时刻复用每个 stage 的同一份 A128/ScaleA128。该布局功能可实现，但目前未带来原命令加速：基础 v104 G4 单次 130.013 μs，S0 carry 的 v107 G4 单次 122.235 μs，均慢于正式同源码全机空闲复测中值 94.279 μs。候选没有替换正式路径。

ATT 支持继续检查 LDS 供数、WG 同步与预读调度：`s_wait_dscnt` 和 `s_barrier_wait` 是主要观测阻塞点。现有证据不足以断言 HBM 饱和或 LDS bank conflict，也不能将并行 wave 的占比直接换算为可省 μs。

## Capture 与证据绑定

在 a07-3 的 `dsr1_ep4` 容器，使用用户指定的 `my_code/get_att_b_rocm_sdk.sh` 采集 v104 G4 的 SIMD0–3。preflight、四个 capture 与 SDK analysis 均 exit0。diagnostic workload 仍为 E64/T1536/topk8、random non-balanced、K7168/N4096，但 `iters=2` 仅供 trace，不能当作正式性能数据。正式筛选使用原完整 `iters=20` 命令，profile count19。

```text
remote capture directory:
/data/yanguahe/code/wk_sp1/aiter/my_code/thread_trace_runs/optx_v104_s4_reuse_g4_att_20261018

kernel:
a8w4_tdm_fp4_t256x256x256_w2x4_b4_K7168_e64_act1_q1r8_cn4_cm1_prefetch_apre_persist_optx_g4_s4reuse_sk0_st0_sn1

source SHA256:
0439f568f8a061caf398135c6a5fcc7c05a3082480149248e6d8249fe3be151a

CPU emitted .text SHA256:
1862fd5b9010c0d7bab7e503e5587f7d77f1ac30232f2ff26c06063090b9a446
```

分析入口 `.codex_tmp/gemm1_t1536_20261017/analyze_optx_trace.py`；报告已下载本地并核对 SHA256：

```text
logs/optx_v104_s4_reuse_att_analysis_20261018.json
82326908a24d23e9171f438fa89a4e4492ce376f3aa2b66fabd9d91df126c55c

logs/optx_v104_s4_reuse_att_analysis_20261018.log
e50ddcbc04ad47dec11071237a39ef872faf809d7c8e9d884c6044d2d188511a
```

没有下载 `.out`。上述 `.text` 是 CPU 编译证据，实际性能执行 object 另行逐项比对并写入 manifest；不能仅因 kernel 名相同声称 object 相同。

## 动态 wave 分解

32 个完整 compute waves，没有 incomplete wave；每 wave 3584 条 WMMA。此版没有 padding skip，仍执行原有效 K 的全部正向累加。合计 wave lifetime 为 6870387 cycles。

| phase | elapsed cycles | wave lifetime 占比 |
|---|---:|---:|
| prologue | 284079 | 4.1348% |
| compute | 5907281 | **85.9818%** |
| handoff | 522564 | **7.6060%** |
| final epilogue | 156463 | 2.2774% |

| stall 归属 | stall cycles | wave lifetime 占比 |
|---|---:|---:|
| `s_wait_dscnt` | 1399964 | **20.3768%** |
| `s_barrier_wait` | 1283077 | **18.6755%** |
| WMMA | 562179 | 8.1826% |
| `s_wait_tensorcnt` | 120270 | 1.7506% |
| `s_wait_storecnt_dscnt` | 80369 | 1.1698% |
| DS_load | 60288 | 0.8775% |

最高单条指令类别为 `s_wait_dscnt 0x6`，stall 1377030 cycles，占 20.0430%。这是等待暴露的位置，不能据此把所有等待独立归因于 LDS 带宽；依赖、跨 wave 时序及 prefetch 深度也可能传递到这里。

| WMMA 间隔分类 | 样本数 | median cycles | mean cycles |
|---|---:|---:|---:|
| K128 | 7168 | 141 | 139.6092 |
| K256 | 5376 | 508 | 511.6628 |
| window_group | 1664 | 553 | 612.3984 |
| task | 96 | 5304.5 | 5458.3750 |

handoff 占比比此前最优 trace 的 14.09% 小，并不证明整个 kernel 更快：capture clock、总 lifetime 和 compute 占比均不同。原命令实测仍回退，应优先解决 steady compute 的 LDS/同步开销，而非只比较 handoff 百分比。

## 后续调度实验

v106 G2 在 F/S 之间保留同 group 的 B/ScaleB 寄存器，每 lane cache 136 DWORD，S 每 K128 的源码 LDS 指令从 20 降为 10，但 VGPR 增至 480，单次 129.591 μs。G4 cache 发生 2775 个 VGPR spill，未运行 GPU。该结果说明减少 LDS 读量与寄存器压力需要共同衡量。

v107 G4 将 S0 READY 与最后 F reader RELEASE 移到 F 最后 K128/1 的 WMMA 之前，随后预读 S0 到已空闲 slot0，与 F 的 register-only WMMA 重叠。单次 122.235 μs 比 v104 的 130.013 μs 快 7.778 μs，仍不是经三轮配对验收的稳定收益。v108 把同样 carry 加到 G2 B-cache，单次 126.000 μs，post6 出现他人 E65 workload，保留为受限筛选。

v109 扩展为组内每个 tile 的下一 READY/当前 RELEASE/下一 K128/0 寄存器预读提前到当前 closing WMMA。A/ScaleA 覆盖与 B refill 始终在 DS reader 退休之后，TDM queue 的请求及 wait distance 保持原四槽协议，K 累加顺序不变。此候选正在通过 CPU、GPU 数值及原命令门禁，尚无性能结论。

## 硬件资料依据与推导边界

- `MI400_Shader_Programming#65.txt` §4.10.1，印刷 p.197：同一 wave 的 TDM load/store 有序完成，跨 wave 无序；因此不能用另一个 owner 的 `tensorcnt` 推断本 owner 已 READY。
- 同文 §4.3.7，Table 36 依赖要求：LDS read 到异步覆写需要 DS reader 退休。共享 A 槽的 RELEASE 不能因换小 LDS 布局而删除。
- 同文 §4.3.6.3，p.76–77：WG barrier `-1` 统计本 WG 的 waves；cluster barrier `-3` 统计 cluster 的 WG。现有共享槽继续用已验证的 WG rendezvous 与原 multicast 协议。
- `amd-instinct-cdna5-instruction-set-architecture.txt` §11.1，p.148：LDS 物理容量 320 KB，64 banks × 4 B。两个 221 KiB WG 无法同时容纳；本次 LDS 下降不会自动使 WG occupancy 翻倍。现有资料和 ATT 没有直接给出本 kernel 的 bank conflict 程度。

详细性能命令、正确性覆盖、资源和负载限制见 [optX 优化记录](moe_gemm1_e64_t1536_optx_optimization.md) 的“四级共享 A128/ScaleA128”章节。
