# gfx1250 E64/T512/topk8 GEMM2 t64 thread trace 分析

## 分析范围

本文分析 a07-3 的 `hyg_fyd1` 容器中，下列 MoE 规模使用的 GEMM2 kernel：

```bash
ROUNDS=3 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-const0 \
  --experts 64 \
  --tokens 512 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048
```

实际 kernel：

```text
a8w4_tdm_fp4_t64x256x256_w1x4_b3_K2048_e64
```

ATT 数据目录：

```text
my_code/thread_trace_runs/e64_t512_topk8_gemm2_t64_att_20261001
```

分析同时使用了 `trace_segment_cycles.py` 和
`my_code/analyze_att_capture.py`。本文引用的本地汇总产物位于：

```text
.codex_tmp/t64_trace_analysis/
```

## 性能与 effective bandwidth

const0 三轮结果：

```text
GEMM2 samples = 35.943, 35.756, 36.333 us
GEMM2 median  = 35.943 us
```

测试程序报告的 `15.644 TB/s` 使用 useful logical bytes 口径：

| 数据项 | 字节数 |
|---|---:|
| A payload | 4,194,304 B |
| A scale | 262,144 B |
| B payload | 469,762,048 B |
| B scale | 29,360,128 B |
| BF16 output | 58,720,256 B |
| 合计 | 562,298,880 B |

```text
562,298,880 B / 35.943 us / 1e6 = 15.644 TB/s
```

这不是实测 HBM 流量。该规模没有使用 cluster multicast，28 个 N tile
会重复请求 A/ScaleA。按 TDM payload 请求量计数，总读写量约为
`682,622,976 B`，对应 `18.99 TB/s`。其中重复 A 数据可能命中 cache，
所以这个数也不能直接解释成物理 HBM bandwidth。

执行的计算量为：

```text
2 * (512 * 8) * 7168 * 2048 = 120,259,084,288 FLOP
```

useful-byte arithmetic intensity 为：

```text
120,259,084,288 / 562,298,880 = 213.87 FLOP/B
```

## 资源与 occupancy

| 资源 | 数值 |
|---|---:|
| LDS/workgroup | 133,632 B |
| SGPR/wave | 58 |
| VGPR/wave | 246 |
| ATT 观察到的每 physical SIMD 同时 active slot | 2 |

MI450 每个 SIMD 有 1024 个 VGPR。T512 t64 kernel 的 LDS/VGPR 占用明显低于
T1536 的 `t192/w2x2/b4` earlynext kernel；后者为 243,712 B LDS、661 VGPR/wave，
ATT 只观察到每 physical SIMD 一个 active slot。

因此，t64 的关键优势是每个 physical SIMD 可驻留两个 wave。单个 wave 遇到
TDM 或 barrier wait 时，另一个 wave 可以继续发射指令，隐藏一部分长延迟。

## phase 占比

126 条 active wave 的聚合结果：

| phase | decoded latency 占比 | exposed stall / wave span |
|---|---:|---:|
| prologue | 23.91% | 19.57% |
| compute | 66.91% | 51.81% |
| epilogue | 9.18% | 7.58% |

总 exposed stall 占 active-wave span 的 `78.95%`。因此该 kernel 的单 wave
效率并不高；dispatch 层性能主要依赖第二个 resident wave 隐藏等待。

主要指令类别：

| 类别 | decoded latency 占比 | exposed stall / wave span |
|---|---:|---:|
| `s_wait_*` | 54.84% | 50.99% |
| `s_barrier_wait` | 25.17% | 23.59% |
| `v_wmma*` | 11.18% | 3.11% |
| other SALU | 4.27% | 0.36% |
| LDS instructions | 3.08% | 0.69% |
| other VALU | 1.27% | 0.22% |

## 主要长延迟指令

| phase | instruction | hits | 平均 latency | 最大 latency | decoded latency 占比 |
|---|---|---:|---:|---:|---:|
| compute | `s_barrier_wait 0xffff` | 882 | 399.31 cycles | 8,389 cycles | 19.11% |
| compute | `s_wait_tensorcnt 0x2` | 756 | 317.72 cycles | 9,555 cycles | 13.03% |
| prologue | `s_wait_tensorcnt 0x2` | 126 | 1,516.43 cycles | 4,284 cycles | 10.37% |
| epilogue | `s_wait_tensorcnt 0x0` | 252 | 431.47 cycles | 4,420 cycles | 5.90% |
| prologue | `s_barrier_wait 0xffff` | 126 | 843.11 cycles | 2,531 cycles | 5.76% |
| compute | `s_wait_dscnt 0x8` | 756 | 108.05 cycles | 1,035 cycles | 4.43% |
| compute | `s_wait_dscnt 0x7` | 1,638 | 47.63 cycles | 3,767 cycles | 4.23% |
| prologue | `s_wait_kmcnt 0x0` | 1,134 | 57.24 cycles | 832 cycles | 3.52% |

代表 wave 中可见初始 `s_wait_tensorcnt 0x2` 约 2,091 cycles、多个
数百至上千 cycle 的 `s_barrier_wait`，以及末尾约 741 cycles 的
`s_wait_tensorcnt 0x0`。这些数据说明 t64 不是低等待 kernel，而是通过更高
occupancy 隐藏等待。

## WGP 均衡与 GFXCLK

`analyze_att_capture.py` 汇总：

| 指标 | 数值 |
|---|---:|
| physical-WGP completion imbalance mean | 9.55% |
| physical-WGP completion imbalance median | 8.70% |
| maximum completion imbalance | 13.34% |
| ATT combined mean GFXCLK | 1,956.452 MHz |

作为参照，T1536 earlynext trace 的 combined mean GFXCLK 为
`1,889.708 MHz`。两次采集的频率不同，因此不能把原始微秒差全部归因于
kernel 结构；本文用 cycle 占比和 occupancy 作结构判断。

## 为什么 t64 的 reported bandwidth 很高

1. useful-byte 口径只把每个逻辑 B/ScaleB surface 计算一次。B 和 ScaleB
   占 useful bytes 的约 88.8%，较短的 dispatch 时间会直接形成很高的 TB/s。
2. t64 的资源占用允许每个 physical SIMD 同时驻留两个 wave；它们可以互相
   隐藏长 TDM/barrier wait。
3. 重复的 A/ScaleA 请求并不等于相同数量的 HBM 流量，其中一部分可由 cache
   命中满足。

可移植到 T1536 的核心经验不是照搬 `tile_m=64`，而是降低单 wave 资源占用、
增加 resident wave，并确保额外并发不会破坏 TDM 次序和 LDS 复用协议。

## 硬件依据

- `MI400_Shader_Programming#65.txt` 3.3.2：每个 SIMD 有 1024 个 VGPR。
- `MI400_Shader_Programming#65.txt` 4.10.8：每个 wave 最多 3 个、每个 SIMD
  最多 6 个等待 XACK 的 TDM operation。
- 同一文档的 TDM ordering 说明：同一个 wave 发出的 TDM operation 按顺序完成。
- `amd-cdna5-whitepaper.txt`：MI455X 有 256 个 WGP；峰值 GFXCLK/HBM 参数仅作
  架构背景，本文实际频率来自 ATT。

