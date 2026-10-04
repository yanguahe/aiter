# E64/T1536/topk8 当前最优 GEMM1 thread trace 瓶颈分析（a07-3）

## 测试对象

当前最优 GEMM1 kernel：

```text
a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_cn4_cm1_prefetch_eb4_apre_sh_bth6_rcw_mg4_fc8_xdl0_reuse3_ostore2p_s3_ps4pf2hm_earlynext_o1w_xor_wait1
```

对应配置：

```text
persistent tasks = 4
tile             = 192x256x256
workgroup        = w2x4（8 waves）
buffers          = 4
cluster          = 4x1
B TDM hint       = 6（NT_HT）
WMMA reuse       = reuseB / reuse3
```

测试规模：

```bash
ROUNDS=3 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-const0 \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048
```

重启后的三轮性能为：

```text
GEMM1 samples = 75.131, 74.825, 74.104 us
GEMM1 median  = 74.825 us
```

同机 phase baseline 为 `78.128 us`，当前最优降低 `4.23%`。const0 的 GEMM1、
GEMM2 和最终 MoE output hash 均与 reference 一致。

## Trace 产物和统计口径

trace 目录：

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/thread_trace_runs/e64_t1536_topk8_gemm1_best_invalid_blocks_a07_3_20261003
```

本次只抓取 SIMD selector 3，以减少 trace 次数。decoder 输出了四个 shader engine
上 SIMD3 的两个 slot，共 8 条有效 compute wave trace。另有 22 条不含 `v_wmma*`
的 invalid early-exit trace，它们不计入本节的 kernel hot-path wait 占比。

分析使用：

```text
cursor_rules/fmha_flydsl_new_api_opt/.cursor/rules/trace_segment_cycles.py
my_code/analyze_att_capture.py
my_code/analyze_gemm_wait_cycles.py
```

`trace_segment_cycles.py` 以 kernel 第一条 `global_prefetch_b8` 到 `s_endpgm` 为
完整区间，得到 8 条有效 wave 的 interval span：

```text
count            = 8
average          = 120636.4 cycles
p50              = 120696.0 cycles
p90              = 122250.2 cycles
minimum          = 118678 cycles
maximum          = 122512 cycles
```

wait 百分比使用更完整的 `wave.begin → wave.end` duration 作为分母。8 条有效 wave
的总 span 为：

```text
994047 cycles
```

这是对 active-wave cycle 的加权统计。不能把多个 wave 的 wait cycles 相加后除以一次
dispatch wall time，因为不同 wave 会并发执行。

## `s_wait_tensorcnt` 与 `s_wait_dscnt` 总占比

| wait 类型 | 动态次数 | decoder latency cycles | 占完整 wave span | exposed stall cycles | stall 占完整 wave span | 单 wave 占比 min / median / max |
|---|---:|---:|---:|---:|---:|---:|
| `s_wait_tensorcnt` | 936 | 15,224 | **1.531517%** | 14,288 | **1.437357%** | 0.839339% / 1.486696% / 2.783217% |
| `s_wait_dscnt` | 6,744 | 222,278 | **22.360915%** | 215,534 | **21.682476%** | 19.583260% / 22.345841% / 24.946279% |

两类 wait 合计：

```text
decoder latency = 237502 / 994047 = 23.892432%
exposed stall   = 229822 / 994047 = 23.119832%
```

因此，当前 kernel 中 `s_wait_dscnt` 的代价约为 `s_wait_tensorcnt` 的 `14.6x`。
TDM global-memory latency大部分已经被 persistent task prefetch 和 WMMA 覆盖，LDS
producer/consumer completion 才是更大的等待来源。

## 各 wait immediate 的构成

### `s_wait_tensorcnt`

| instruction | count | latency cycles | 占完整 wave span | exposed stall |
|---|---:|---:|---:|---:|
| `s_wait_tensorcnt 0x2` | 808 | 7,024 | 0.7066% | 6,216 |
| `s_wait_tensorcnt 0x3` | 8 | 5,324 | 0.5356% | 5,316 |
| `s_wait_tensorcnt 0x0` | 64 | 2,539 | 0.2554% | 2,475 |
| `s_wait_tensorcnt 0x1` | 56 | 337 | 0.0339% | 281 |

`0x3` 是每条有效 wave 的初始 pipeline 等待，平均约 `665.5 cycles`；`0x2` 在
steady pipeline 中出现很多次，但绝大多数已经被前置计算覆盖，平均每次约
`8.7 cycles`。最终 `0x0` drain 平均约 `39.7 cycles`。

### `s_wait_dscnt`

| instruction | count | latency cycles | 占完整 wave span | exposed stall |
|---|---:|---:|---:|---:|
| `s_wait_dscnt 0x0` | 1,016 | 118,246 | 11.8954% | 117,230 |
| `s_wait_dscnt 0xa` | 960 | 86,043 | 8.6558% | 85,083 |
| `s_wait_dscnt 0x8` | 952 | 4,591 | 0.4618% | 3,639 |
| `s_wait_dscnt 0x6` | 64 | 4,112 | 0.4137% | 4,048 |
| `s_wait_dscnt 0x1d` | 864 | 3,360 | 0.3380% | 2,496 |
| `s_wait_dscnt 0x5` | 32 | 2,318 | 0.2332% | 2,286 |
| 其余 immediate 合计 | 2,856 | 3,608 | 0.3630% | 1,752 |

`s_wait_dscnt 0x0` 和 `s_wait_dscnt 0xa` 合计占完整 wave span 的
`20.5512%`，占全部 `s_wait_dscnt` latency 的约 `91.91%`。问题集中在必须完全
drain LDS 操作的 stage handoff，以及 WMMA 消费下一组 operands 前的 LDS-read
完成等待。

## 其他主要 cycle 项

| opcode | decoder latency cycles | 占完整 wave span | exposed stall 占比 | 说明 |
|---|---:|---:|---:|---|
| `v_wmma_scale_f32_32x16x128_f4` | 330,660 | 33.2640% | 13.7945% | GEMM 有效计算主体 |
| `s_wait_dscnt` | 222,278 | 22.3609% | 21.6825% | LDS completion 主瓶颈 |
| `s_barrier_wait` | 185,600 | 18.6711% | 18.5746% | workgroup wave 到达偏斜 |
| `ds_load_b128` | 61,814 | 6.2184% | 2.6130% | A/B operand 从 LDS 进入 VGPR |
| `s_wait_kmcnt` | 24,097 | 2.4241% | 2.4169% | prologue 的 scalar load / expert lookup |
| `s_wait_tensorcnt` | 15,224 | 1.5315% | 1.4374% | TDM load/store completion |
| `v_add_nc_u32_e32` | 12,578 | 1.2653% | 0.6086% | 地址更新 |
| `v_tanh_f32_e32` | 12,321 | 1.2395% | 0.6214% | SiLU epilogue |

所有 `s_barrier_wait` 合计占 `18.671149%`。`s_wait_dscnt + s_barrier_wait`
合计占 `41.032064%`；再加上 `s_wait_tensorcnt` 后为 `42.563581%`。

最重的静态 wait 位置为：

| PC | instruction | latency/span | 作用 |
|---|---|---:|---|
| `0x36e0` | `s_barrier_wait 0xffff` | 9.9657% | steady K-loop stage handoff |
| `0x4000` | `s_wait_dscnt 0x0` | 5.4900% | 下一 stage 可读/复用前完整 LDS drain |
| `0x36d8` | `s_wait_dscnt 0x0` | 5.0805% | 下一 stage 可读/复用前完整 LDS drain |
| `0x3e28` | `s_wait_dscnt 0xa` | 4.1319% | WMMA 前等待部分 LDS operands |
| `0x4008` | `s_barrier_wait 0xffff` | 3.7900% | steady K-loop stage handoff |
| `0x3508` | `s_wait_dscnt 0xa` | 2.8735% | WMMA 前等待部分 LDS operands |

前四个主要 `s_wait_dscnt` PC 合计贡献全部 DScnt wait latency 的 `78.60%`；
前两个主要 hotloop barrier PC 合计贡献全部 barrier wait latency 的 `73.67%`。

## 两个 resident slot 的不对称

同一 SIMD 的两个 resident slot 呈现稳定的不对称：

| slot | `s_wait_tensorcnt` | `s_wait_dscnt` | `s_barrier_wait` |
|---|---:|---:|---:|
| slot0，四个 SE 加权 | 1.1737% | 20.9446% | 27.5693% |
| slot1，四个 SE 加权 | 1.8876% | 23.7706% | 9.8147% |

slot1 在 LDS/TDM 数据就绪上更慢，而 slot0 更早到达 barrier，并把差值暴露为较长的
`s_barrier_wait`。因此 barrier 本身不是唯一根因；更准确的描述是：两个 resident
waves 的 LDS operand readiness 和指令推进速度不一致，最后在全 workgroup barrier
处汇合并暴露出偏斜。

## 当前性能瓶颈结论

当前最优 GEMM1 的第一瓶颈是 **K-loop LDS operand pipeline**：

1. 每个 k128 都需要从 LDS 读取 A、B、ScaleA 和 ScaleB 到 VGPR。
2. `s_wait_dscnt 0xa` 在 WMMA 前等待所需 operands，`s_wait_dscnt 0x0` 在 stage
   交接或 buffer 复用前完全 drain LDS 操作。
3. 这些等待本身占 `22.36%`，随后产生的 workgroup barrier 到达偏斜又占
   `18.67%`。

第二个约束是 **两个 resident waves 的推进不均衡**。slot0 的 barrier 等待明显高于
slot1，而 slot1 的 DScnt/TENSORcnt 等待更高。这说明简单删除 barrier 不安全；应当让
慢 slot 更早完成 LDS/TDM 相关工作，或减少 barrier 前必须完成的 LDS 操作。

`s_wait_tensorcnt` 只有 `1.53%`，说明当前 persistent prefetch 已经隐藏了绝大部分
TDM latency。继续单纯增加 TDM prefetch distance、增加 inflight TDM 数或只调整 B
cache hint，预计不会形成大幅收益，并且可能碰到 MI400 每 wave 3 个、每 SIMD 6 个等待
XACK 的 TDM 限制。

`s_wait_kmcnt` 仍占 `2.42%`，主要来自 expert lookup 的 scalar loads。当前 persistent
版本已经让同一 block 的 4 个 N task 共用一次 expert lookup，因此这里剩余的可优化空间
明显小于 LDS/barrier 路径。

## 后续优化优先级

1. 重排每个 k128 的 LDS reads 与 WMMA，使下一批 A/B operands 更早进入 VGPR，并让
   `s_wait_dscnt 0xa` 之前保留更多独立 WMMA 或地址计算。
2. 减少 stage handoff 的全量 `s_wait_dscnt 0x0`；需要按 LDS buffer 的真实读写集合证明
   可以使用较松的 threshold，不能直接删除。
3. 针对 slot1 的 LDS/TDM owner 路径重新平衡 descriptor setup 和 LDS load，减少它成为
   barrier straggler 的概率。
4. 若工具链支持，应在 code-object descriptor 层测试
   `.amdhsa_round_robin_scheduling 1`。MI400 Shader Programming Guide §5.2.2 明确说明
   workgroup equal-priority 模式用于让协作 waves 以相近速度到达 barrier。此前有问题的
   in-kernel `s_setprio_inc_wg` 实验不应继续使用。

相关硬件依据：

- `MI400_Shader_Programming#65.txt` §4.3.7.2.4：LDS 操作由 DScnt 跟踪，read 的
  DScnt 完成表示结果已经可以从 VGPR 使用。
- 同文档 §4.10.1：TDM completion 由 TENSORcnt 跟踪，且同一 wave 的 tensor
  instructions 保持顺序。
- 同文档 §4.10.8：每 wave 最多 3 个、每 SIMD 最多 6 个等待 XACK 的 TDM。
- 同文档 §5.2.2：workgroup equal-priority scheduling 用于减少协作 waves 到达
  synchronization point 的偏斜。

## 复现命令

完整有效 wave interval：

```bash
python3 /data/yanguahe/code/wk_sp1/cursor_rules/fmha_flydsl_new_api_opt/.cursor/rules/trace_segment_cycles.py \
  "$UI_DIR" \
  --wv 0 \
  --interval-start 'global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE' \
  --interval-end 's_endpgm' \
  --rank-by sum \
  -k 40 \
  --top-events \
  --hide-occurrence-details
```

wait/opcode 汇总：

```bash
python3 my_code/analyze_gemm_wait_cycles.py "$UI_DIR" --top 20
```

对应日志：

```text
logs/trace_segment_full_valid_waves.log
logs/wait_cycle_analysis.log
logs/wait_cycle_aggregate.log
logs/wait_immediate_summary.log
logs/per_wave_wait_share.log
logs/top_wait_events.log
logs/top_wait_pc_context.log
```
