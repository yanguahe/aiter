# hyg_dsr1_conc6144 当前最优 GEMM1 的 thread trace 分析

目标更新（2026-10-23）：用户将最终性能目标由80 μs调整为争取70 μs。
当前保留版（2026-10-24）已正式采用DS5基础上的目标pin1寄存器布局、LDS地址tied constraints
和等价i32地址表达式，source=`f3c77d2a…f1fb21ec`，GEMM1 ELF=`c5296627…f8001d2`。
正常正反向候选组中值80.8191、80.6187、80.5554 μs；正式源码原入口三轮复测中值为81.0172 μs，
首轮从空cache运行，完整结果和恢复说明见文末。目标70 μs尚未达到。
匹配trace与本地复算说明见文末「i32地址表达式最佳候选的matched ATT」。
上一DS-first5/pin0的81.6350、81.8183、81.7086 μs及历史PC/capture保持原样，
新source的pin0完整ELF不变；不同ELF的PC与动态统计不混用。

版本说明：最前面的分析保留`3f23fb29...97b68` / `0448a7ab...bc35ef7`的84.9116 μs阶段。
2026-10-23已采用closing24 + MMA3/3，正式中值82.0840 μs，source为`33a882a6...f81c01f09`，
ELF为`ed5fc68a...f396df5`。新版本的独立capture和本地复算写在文末「MMA3/3新capture」一节。
随后已采用MMA3/first2，正式中值81.4412 μs，source=`7f60e033...2b0f7`，
ELF=`86d771a2...abc34a1`；该版未重新采集trace，以下MMA3/3的PC和统计仍只属于其原capture。
其后已采用MMA3/first2 + 末尾closing24覆盖，正式中值81.3419 μs，
source=`4128856c...65115fd`、ELF=`6773d72e...856837d`。该版未重新采集trace；
旧K128 #54拆分用于提出调度假设，不能冒充此ELF的动态PC或占比。
不同ELF的PC、资源、比例分别绑定各自capture；优化数据见参数遍历文档。

2026-10-22 在 a07-3 / `dsr1_ep4` 采集当前正式保留版，四个 SIMD-select captures 均成功。
原 benchmark 正式三轮中值为 **84.9116 μs**，确认组中值为 **85.2573 / 85.3416 μs**，
尚未达到 80 μs。这里的 trace 是诊断数据，不替换上述原入口性能记录。

当前证据支持优先优化 mainloop 的 LDS/WMMA 排布及 producer/consumer 到达时间。
最后 input drain 仅占所采 active waves 累计 timestamp span 的 0.2048%，
不能将它当作主要瓶颈，也没有依据删除现有 fence。

## 版本与复现

仓库为 `/data/yanguahe/code/wk_sp1/aiter_a4w4_prefill_v2_yadai`，
分支为 `hyg_dsr1_conc6144`，HEAD 为 `346438a75506f77b520d3a9a09ba4985f054e377`。
boot ID 为 `cb28eed4-51ca-4cdc-bf5b-ecbc1d54db5f`，FlyDSL distribution 为 `0.3.3.dev882`。
采用组合为 DS_B 计数修正、XDL queue bit、M-group32、首 K128 inline-zero、
A-only standard timeout。开启 A preshuffle，关闭 VGPR pin。

| 项目 | 实际值 |
|---|---|
| 生产 source SHA256 | `3f23fb29a23094fa4335acf66f2515d48d7f331450ce16d16ff4983dd9f97b68` |
| GEMM1 完整 ELF SHA256 | `0448a7abc2e9ea92ea1f4dd81ecd7cd9e60f82af1a7c63fcfd4b35c32bc35ef7` |
| GEMM1 `.text` SHA256 | `d1ee894b1ea13fc9616cdaf8b85c919aa639ae0916aaf332a36981df55ec8dcb` |
| GEMM2 完整 ELF SHA256 | `4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be` |
| VGPR / SGPR | 762 / 58 |
| LDS / private / spills | 327680 B / 0 / 0 |
| executable static instructions | 6158 |

完整 GEMM1 symbol 保留原命名，内部优化不全部体现在名字中：

```text
a8w4_tdm_fp4_apre_t256x256x256_w2x2_b4_K7168_e64_act1_q1r8_cluster1x4_prefetch_bth1_wpt1_mg8_dsfirst16_planarlds_interleavelds_rolleddrain
```

E65 仍采用当前分支原 `t64/w1x4` fallback。本分析不声称 E64/E65 共用上述 `t256` kernel，
也不将 random E65 测试等同于固定 shared expert routing 验证。

```bash
# a07-3 主机；脚本内部进入 dsr1_ep4，并在每次 profile 前等待全机空闲。
bash .codex_tmp/yadai_gemm1_param_20261021/tools/run_best_frame_fixed_att_v1.sh \
  NEW_UNIQUE_BEST_ATT_LABEL
```

该脚本调用用户指定的 `my_code/get_att_b_rocm_sdk.sh`，使用
`--all-simd --iteration 28 --ana-att`，原 workload 为 E64/T1536/topk8、7168/2048、
uniform/auto/seed0、non-balanced 的 `--scenario kernel`，warmup5/iters20 不变。
每个 profiler/decoder child 限时120秒。采集后未发现本任务 GPU/KFD 进程残留。

## 实际阶段与等待

每个 SIMD capture 有16个 active waves，总计64；每个 active wave 执行1792条 WMMA，
即56个 K128 steps × 每步32条 WMMA。四次 capture 是独立执行，不能拼成同一次 GPU 时间线。

阶段从实际 decoded 指令序列确定，逐 wave 精确相加等于第一条指令到 `s_endpgm` 的
timestamp span。wave launch/retirement 外围间隔不在此统计中。

| 阶段 | 64 waves 累计 cycles | 累计占比 | 每 wave 中值 cycles | 范围 cycles |
|---|---:|---:|---:|---|
| prologue → 首条 WMMA | 251693 | 12.7827% | 3543 | 2716–13750 |
| WMMA mainloop | 1404881 | 71.3496% | 20784 | 20603–28815 |
| 最后 input drain / LDS arena 交接 | 4032 | 0.2048% | 63 | 63–63 |
| activation / quant / ScaleA / LDS staging + 输出 barrier | 253992 | 12.8995% | 3968.5 | 3956–3980 |
| 输出 descriptor setup | 3200 | 0.1625% | 50 | 50–50 |
| output TDM / completion | 51212 | 2.6009% | 905 | 168–981 |

按前一条指令至下一条指令的 timestamp gap 归因，WMMA 为35.6319%、
`ds_load_b128` 为15.4625%、`s_barrier_wait` 为11.6144%、
`s_wait_tensorcnt` 为5.7863%、`s_wait_dscnt` 为4.9799%。
这与阶段统计是两种口径，不能相加；gap 包含 arbitration、coexecution 等影响，
不等于对应指令独占硬件的延迟。

| actual PC | ISA | hits | 累计 timestamp gaps | trace stalls | 已执行 WMMA 数范围 |
|---|---|---:|---:|---:|---|
| `0x2580` | `s_barrier_wait 0xffff` | 64 | 84691 | 84435 | 0–0 |
| `0xc3c0` | `s_barrier_wait 0xffff` | 64 | 57620 | 57492 | 1792–1792 |
| `0xc468` | `s_wait_tensorcnt 0x0` | 64 | 44621 | 44557 | 1792–1792 |
| `0x2500` | `s_wait_tensorcnt 0x3` | 64 | 36378 | 36250 | 0–0 |
| `0x61c4` | `s_barrier_wait 0xffff` | 368 | 29551 | 28447 | 96–1504 |
| `0x4300` | `s_wait_tensorcnt 0x2` | 368 | 28641 | 28273 | 96–1504 |

首个 input barrier 和最终输出 barrier 都在等待其他 waves 的进度，不能将全部 gap
归为当前 wave 的 TDM latency。mainloop 的 LDS read / wait 较突出，适合继续测试
小范围 `sched_ds_first`、WMMA grouping、closing compiler hints，必须以原入口实测确认收益。
现有 input ring、硬件 fence、有效累加顺序及 FP32 amax → BF16 RNE → native FP4 规则保持。

## 硬件依据与限制

- `MI400_Shader_Programming#65.txt` §4.3.7.4.2，文档p93：
  `SCHED_MODE[2] DISABLE_XDL_ARB_STALL` 允许同一 wave 连续 issue 多条 WMMA，
  文档预计可 issue 最多5条后由硬件 stall。此特性已在当前正式版开启；
  它支持重新检查 DS/WMMA scheduling 的组合，不能证明某个分组参数必然更快。
- 同文档 §4.10.1，文档p197：TDM 忽略 EXEC，同 wave 按序完成，跨 wave
  及相对其他 memory instructions 无序。不能用 lane mask 或某一 wave 的 wait
  替代 buffer lifetime 所需同步。
- 同文档 §3.3.2.3，文档p34，以及 §5.7.1：`S_SET_VGPR_MSB` 不产生
  `S_SETREG_B32 MODE` 的额外 delay，且属于可以与普通指令 co-issue 的 internal control。
  当前 ISA 有955个 static sites，但 decoded wave instruction events 没有单独记录该 opcode；
  不能据此声称它动态执行0次或耗时为0，也不能把955直接换算成可节省 cycles。

累计 wave gaps 不是 GPU critical path 分解，阶段百分比不能直接预测 kernel 提速。
旧 `moe_gemm1_hyg_dsr1_pin0_thread_trace_analysis.md` 对应 DS-only 88.5421 μs 版本，
其 ELF、PC 和 phase 不能冒充当前版本；不同 capture 的 cycles 也不直接换算为性能改善。

## 本地证据与独立验收

capture label 为 `reboot20261022_best_framefixed_att_v1`，本地目录为
`.codex_tmp/yadai_gemm1_param_20261021/att/reboot20261022_best_framefixed_att_v1/`。

| SIMD-select | `thread_trace/simdN/kernel/rpf_v3/` 下的 UI directory |
|---|---|
| 0 | `ui_output_agent_2865_dispatch_562` |
| 1 | `ui_output_agent_32592_dispatch_562` |
| 2 | `ui_output_agent_61136_dispatch_562` |
| 3 | `ui_output_agent_53722_dispatch_562` |

raw ATT、decoded JSON、实际两阶段 ELF、ISA、原 call inventory 和 source snapshot 均已下载。
省略 `.out` 与 private cache。archive 为240个 regular members、7033974 B，SHA256 为
`1e15d9ae2a6cd8bf90951f492fe8ff2d262319574cf2f13c1d963df8ad7dc364`。

封包v1因缺少尚未上传的本地 archive acceptor 停止；补传该文件并使用v2核对已有
source snapshot 的 bytes 后封包成功。原失败日志保留，没有重新采集或修改原 trace。

`bind_best_frame_fixed_trace_v1.py` 从 capture 的实际 private cache 按原调用 key
提取 artifact，核对 source/global/key/sourceIR/完整 ELF；全部 decoded PC/ISA
逐项与实际 ELF disassembly 匹配。GEMM1 原 call key 为 `62c9ae1cec751fe7`，
GEMM2 为 `da3c5914af92622d`，manager 为 `9d1a558097a04e225f63a98703f67c8a`。

本地独立验证240个 archive members 的大小/SHA，并重新计算64 waves 的所有统计。
首次直接比较 JSON 时因 Windows/Linux 路径分隔符不同停止；仅规范化68个路径字段后，
binding、全部 phase/opcode/wait/wave 数据与远端精确一致。验收记录为
`reboot20261022_best_framefixed_att_v1.local_trace_acceptance.json`，`passed=true`。

## MMA3/3新capture（2026-10-23）

当前保留版组合为DS_B计数修正、XDL queue bit、M-group32、首K128 inline-zero、
A-only standard timeout、closing24、MMA3/3。原入口正向三轮中值为**82.0840 μs**，
反向确认组为81.7698 /82.0861 μs；相对MMA4/4的四项相邻对照改善0.66%～1.15%。
E64 seed0/17、E65 seed0/17完整MoE和8项outside-scope完整ELF核验通过后才采用。

| 项目 | 实际值 |
|---|---|
| source SHA256 | `33a882a69537779c954d83b47b03b6cfc56fc19edf1c7a9b1fa05b3f81c01f09` |
| GEMM1完整ELF SHA256 | `ed5fc68ab38a0d99d65e99846b54aa8799dbb0de0c1ee4f1aaf5f3720f396df5` |
| GEMM1 `.text` SHA256 | `118d893e34a074d2b0b18ea9d69990317a75175e0be9108155fe6c49ccf6a131` |
| GEMM2完整ELF SHA256 | `4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be` |
| VGPR / SGPR / LDS | 797 /58 /327680 B |
| private /spills /executable static instructions | 0 /0 /6260 |

完整kernel名：

```text
a8w4_tdm_fp4_apre_t256x256x256_w2x2_b4_K7168_e64_act1_q1r8_cluster1x4_prefetch_bth1_wpt1_mg3_dsfirst16_planarlds_interleavelds_rolleddrain
```

仍在同一a07-3 / `dsr1_ep4` / boot下，使用用户指定脚本和原workload，
开启A preshuffle、关闭VGPR pin；各profiler/decoder child限时120秒。

```bash
bash .codex_tmp/yadai_gemm1_param_20261021/tools/run_best_mma3_att_v1.sh \
  NEW_UNIQUE_MMA3_ATT_LABEL
```

本次label=`reboot20261023_best_mma3_att_v1`，SIMD0～3采集及decode均rc0，结束检查
没有本任务GPU/KFD进程残留。每个capture16个active waves，共64，每wave1792条WMMA；
四个SIMD仍是独立capture。以下为累计wave timestamp gaps，不是整个GPU的关键路径分解。

| 阶段 | 64 waves累计cycles | 累计占比 | 每wave中值cycles | 范围cycles |
|---|---:|---:|---:|---|
| prologue → 首WMMA | 243414 | 13.9380% | 3699.5 | 2963–5703 |
| WMMA mainloop | 1191267 | 68.2125% | 18147 | 17985–24708 |
| 最后input drain | 4224 | 0.2419% | 66 | 66–66 |
| activation /quant /scale /LDS + barrier | 253443 | 14.5123% | 3960 | 3947–3975 |
| output descriptor setup | 3200 | 0.1832% | 50 | 50–50 |
| output TDM /completion | 50858 | 2.9122% | 900 | 168–974 |

按前一条opcode归因的timestamp gaps中，WMMA占36.2068%、`ds_load_b128`占13.3931%、
`s_barrier_wait`占9.8113%、`s_wait_tensorcnt`占6.5081%、`s_wait_dscnt`占3.0372%。
opcode与阶段口径不能相加；gap包含arbitration/coexecution，不等于指令独占延迟。

| 当前actual PC | ISA | hits | timestamp gaps | trace stalls | 已执行WMMA范围 |
|---|---|---:|---:|---:|---|
| `0x2580` | `s_barrier_wait 0xffff` | 64 | 75126 | 74870 | 0–0 |
| `0xc55c` | `s_barrier_wait 0xffff` | 64 | 54546 | 54418 | 1792–1792 |
| `0xc604` | `s_wait_tensorcnt 0x0` | 64 | 44302 | 44238 | 1792–1792 |
| `0x2500` | `s_wait_tensorcnt 0x3` | 64 | 35911 | 35783 | 0–0 |
| `0x436c` | `s_wait_tensorcnt 0x2` | 368 | 21635 | 21267 | 96–1504 |
| `0x5314` | `s_barrier_wait 0xffff` | 368 | 8538 | 7434 | 96–1504 |

mainloop仍占主要累计span，LDS read与WMMA/tensor waits仍突出；input final drain小，
现有证据不支持以删除必要fence作为主要优化。下一步先测试MMA2/2、MMA3/first2，
分别仅修改现有guard的一到两个默认数字；没有新增persistent、wave布局或pipeline。
这只是依据当前trace选择实验方向，不能从占比推算将节省多少μs。

hardware依据仍为Shader Guide §4.3.7.4.2 /ISA §5.7.2.1：XDL queue bit允许连续issue，
实际最佳group必须实测；§5.3.3允许部分internal control co-issue，不能按
`S_SET_VGPR_MSB`静态条数计算独占开销。DS/TDM数据依赖继续遵守ISA §5.7及原同步。

### MMA3/3本地证据

本地路径为`.codex_tmp/yadai_gemm1_param_20261021/att/reboot20261023_best_mma3_att_v1/`。

| SIMD | UI directory |
|---|---|
| 0 | `ui_output_agent_12169_dispatch_562` |
| 1 | `ui_output_agent_12396_dispatch_562` |
| 2 | `ui_output_agent_29881_dispatch_562` |
| 3 | `ui_output_agent_35562_dispatch_562` |

实际capture private cache按candidate原call inventory核对，不复制其他ELF到cache。
manager=`f0624b07429cb994a9f4ab4f770e4ade`，GEMM1 key=`62c9ae1cec751fe7`，
GEMM2 key=`da3c5914af92622d`；inventory来自本轮candidate源码，sourceIR与globals均匹配。
每个decoded PC和ISA编码逐项对应真实GEMM1 ELF，不复用旧阶段PC。

237 regular members、7058221 B archive已下载，本地逐项大小/SHA及全部wave统计复算通过，
SHA256为`152aa4f27800e12073d4b61ee1451abfe003a0e0899e67a7fc013d087fa3e9ee`。
仍省略`.out`和private cache。验收文件为
`reboot20261023_best_mma3_att_v1.local_archive_acceptance.json`和
`reboot20261023_best_mma3_att_v1.local_analysis_acceptance.json`，均`passed=true`。
仅规范化68个Windows/Linux路径分隔符字段，统计全部与远端完全一致。

### MMA3/3 K128尾部的进一步拆分

`analyze_mma3_ksteps_v1.py`从上述已独立验收的capture重新核对真实ELF和decoded PC/ISA，
按每32条WMMA建立56个连续区间。一个区间从本K128的首WMMA开始，至下一K128首WMMA；
最后一个区间结束于最后WMMA之后的第一条指令。64 waves的所有区间精确相加为
mainloop的1191267 cycles，未混入其后的final input drain。

| K128 index | K256 tile/subtile | 每wave区间中值cycles | 范围cycles |
|---|---|---:|---|
| 46 | 23/0 | 313 | 302–402 |
| 47 | 23/1 | 376 | 317–440 |
| 48 | 24/0 | 421 | 348–6858 |
| 49 | 24/1 | 304 | 301–314 |
| 50 | 25/0 | 358 | 355–464 |
| 51 | 25/1 | 302 | 300–313 |
| 52 | 26/0 | 330 | 315–335 |
| 53 | 26/1 | 298 | 295–300 |
| 54 | 27/0 | 600.5 | 599–602 |
| 55 | 27/1 | 234 | 234–234 |

这些区间包含下一subtile的LDS prefetch及两段WMMA间的调度工作，不能解释成各K128
独占硬件延迟或直接换算μs。大多数steady intervals约313～315 cycles；#54无TDM与barrier，
但含32条LDS b128 reads、地址生成和compiler-generated DS waits，值得检查尾部调度。
#48个别wave的较大值也不能单独证明HBM或外部任务是原因。

在MMA3/first2原入口对照期间只做本地静态准备两个独立候选：让最后K256 tile的
subtile0使用已有closing24 compiler hint；或无下一subtile时将剩余WMMA作为一个compiler group。
分别+1/-1、+2/-1行，不删除硬件wait/barrier，不改变访存、有效累加、功能或shape支持。
实际hint AST展开各112项检查通过，MMA/DS/TDM总数保持；这仍是未测性能的实验方向。

完整可复算记录为`.codex_tmp/yadai_gemm1_param_20261021/logs/mma3_kstep_detail_v1.json`，
SHA256为`2e65423476a86b1283183c7d2ea9abdab48176e820908defcb3a56241bc2b9bc`。

## 当前正式 MMA2/first2 + DS floor capture（2026-10-23）

当前保留组合为DS_B计数修正、XDL queue bit、M-group32、首K128 inline-zero、
A-only standard timeout、closing24（含最后K256/subtile0）、MMA2/first2、future DS floor。
原入口正向完整组三轮中值为**81.3545 μs**，反向为81.2954 /81.3757 μs；尚未达到80 μs。
这是按相邻对照确认的当前版本，不改写历史81.2958 μs的记录。
最近group5和DScenter两项均未同时优于前后相邻control，没有采用。

| 项目 | 当前实际值 |
|---|---|
| production source SHA256 | `36178ad89a4d4c3cda46514289b10574aaf1938f4eed0f0329e3cd6eedb406b0` |
| GEMM1完整ELF SHA256 | `c061a2677dfe515810fb509499fe3d5cdb391db245ef03af7b408a9dfc53fe63` |
| GEMM2完整ELF SHA256 | `4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be` |
| VGPR / SGPR / LDS | 797 /58 /327680 B |
| private /spills /executable static instructions | 0 /0 /6307 |

kernel symbol为：

```text
a8w4_tdm_fp4_apre_t256x256x256_w2x2_b4_K7168_e64_act1_q1r8_cluster1x4_prefetch_bth1_wpt1_mg2_dsfirst16_planarlds_interleavelds_rolleddrain
```

本次仍在a07-3的`dsr1_ep4`，FlyDSL `0.3.3.dev882`，boot为
`cb28eed4-51ca-4cdc-bf5b-ecbc1d54db5f`，开启A preshuffle、关闭VGPR pin。
使用用户脚本`my_code/get_att_b_rocm_sdk.sh`，原workload，
`--all-simd --iteration 28 --ana-att`；每个profiler/decoder child限时120秒。
label=`reboot20261023_best_mma2_floor_att_v1`，host parent3946970已退出。
SIMD0～3 capture/decode均rc0，`att_status=0/analyze_status=0`；decoded code6308行
（6307条executable及标记），无`.long/v_illegal`。结束检查无测试GPU/KFD进程残留。

复现诊断采集：

```bash
bash .codex_tmp/yadai_gemm1_param_20261021/tools/run_best_mma2_floor_att_v1.sh \
  NEW_UNIQUE_MMA2_FLOOR_ATT_LABEL
```

### 当前trace绑定和本地验收

实际capture private cache的manager为`8d1d73d639af5362c80d5027ba2f9a1d`，
GEMM1 key=`62c9ae1cec751fe7`，GEMM2 key=`da3c5914af92622d`。
两阶段实际ELF、sourceIR与当前候选的原call inventory一致；每个decoded PC及ISA编码
重新对应当前GEMM1 ELF，没有沿用MMA3/3的PC或比例。

237 regular members、7006628 B的decoded archive已下载，本地逐大小/SHA验收通过，
archive SHA256为`8933d7f8b48adfe2a098e1c633ddb3d851a50477c5a3b19b70279698dcc44d90`。
未下载`.out`或private cache。全部wave事件和阶段统计独立复算与远端完全一致，
仅规范化68个Windows/Linux路径分隔符字段。

本地验收文件为：

```text
.codex_tmp/yadai_gemm1_param_20261021/logs/reboot20261023_best_mma2_floor_att_v1.local_archive_acceptance.json
.codex_tmp/yadai_gemm1_param_20261021/logs/reboot20261023_best_mma2_floor_att_v1.local_analysis_acceptance.json
```

两项均`passed=true`。analysis acceptance SHA256为
`bec80102c05d470d7c58a8856892227a423448c298c203754a1e28105bc89d6f`。

| SIMD | 当前UI directory | active waves |
|---|---|---:|
| 0 | `ui_output_agent_29642_dispatch_562` | 16 |
| 1 | `ui_output_agent_21048_dispatch_562` | 16 |
| 2 | `ui_output_agent_1650_dispatch_562` | 16 |
| 3 | `ui_output_agent_57121_dispatch_562` | 16 |

### 当前阶段、opcode与wait分布

四个SIMD是独立capture。以下为64个active waves的累计timestamp gaps，
每个wave实际执行1792条WMMA；不构成整个GPU的关键路径分解。

| 阶段 | 累计cycles | 累计占比 | 每wave中值cycles | 范围cycles |
|---|---:|---:|---:|---|
| prologue → 首WMMA | 255778 | 14.6912% | 3655.5 | 2644–11470 |
| WMMA mainloop | 1174401 | 67.4543% | 17987.5 | 17653–26444 |
| 最后input drain | 4224 | 0.2426% | 66 | 66–66 |
| activation /quant /scale /LDS + barrier | 253010 | 14.5322% | 3954 | 3934–3974 |
| output descriptor setup | 3200 | 0.1838% | 50 | 50–50 |
| output TDM /completion | 50418 | 2.8959% | 898.5 | 167–1049 |

按前一条opcode归因，WMMA占34.0975%、`ds_load_b128`占14.9927%、
`s_barrier_wait`占11.8145%、`s_wait_tensorcnt`占5.1515%、`s_wait_dscnt`占3.0369%。
该口径和阶段口径不能相加；gap包含arbitration/coexecution，不等于指令独占延迟。

| 当前actual PC | ISA | hits | timestamp gaps | trace stalls | 已执行WMMA范围 |
|---|---|---:|---:|---:|---|
| `0x2580` | `s_barrier_wait 0xffff` | 64 | 85837 | 85581 | 0–0 |
| `0xc618` | `s_barrier_wait 0xffff` | 64 | 54132 | 54004 | 1792–1792 |
| `0xc6c0` | `s_wait_tensorcnt 0x0` | 64 | 43866 | 43802 | 1792–1792 |
| `0x2500` | `s_wait_tensorcnt 0x3` | 64 | 39129 | 39001 | 0–0 |
| `0x3404` | `s_barrier_wait 0xffff` | 368 | 23390 | 22286 | 96–1504 |
| `0x53e8` | `s_barrier_wait 0xffff` | 368 | 10303 | 9199 | 96–1504 |

mainloop仍是主要累计span，LDS read与WMMA调度值得继续检查。
prologue和epilogue也有明显等待，现有结果不足以证明某次等待由HBM、外部任务或单一依赖造成。
input final drain仅占0.2426%，不以删除必要硬件fence作为当前优化方向。

### 当前K128区间重新分解

`analyze_best_mma2_floor_ksteps_v1.py`重新绑定当前ELF及decoded PC/ISA，
按每32条WMMA建立56个连续区间，所有wave区间精确相加为1174401 cycles。
区间从本K128首WMMA至下一K128首WMMA；最后一个至最后WMMA后的第一条指令。

| K128 index | K256 tile/subtile | 每wave区间中值cycles | 范围cycles |
|---|---|---:|---|
| 0 | 0/0 | 345 | 290–1324 |
| 1 | 0/1 | 342.5 | 326–724 |
| 46 | 23/0 | 316 | 303–473 |
| 47 | 23/1 | 375 | 316–439 |
| 48 | 24/0 | 400 | 340–465 |
| 49 | 24/1 | 300 | 297–325 |
| 50 | 25/0 | 351 | 347–1292 |
| 51 | 25/1 | 299 | 298–319 |
| 52 | 26/0 | 308 | 306–347 |
| 53 | 26/1 | 343.5 | 340–347 |
| 54 | 27/0 | 348.5 | 346–350 |
| 55 | 27/1 | 253 | 253–253 |

多数steady区间中值为310～323 cycles。当前#54为348.5 cycles；旧MMA3/3的600.5 cycles
来自不同ELF及capture，不能当成同版本前后相邻性能对照或据此换算优化收益。
仍有个别wave较大gap；这些区间含prefetch、调度、arbitration和coexecution，
不能单独用于判断硬件延迟或整kernel节省的μs。

完整可复算记录为`.codex_tmp/yadai_gemm1_param_20261021/logs/best_mma2_floor_kstep_detail_v1.json`，
SHA256为`2ab2a8a288c048627d66e9758065798785548120311be3d5df9a01deb7ee1363`。

硬件依据仍为Shader Guide §4.3.7.4.2（文档p93）及CDNA5 ISA §5.7.2.1（p56）：
XDL queue bit允许多条WMMA issue，但可能影响co-execution，不能证明group5更快。
ISA §5.7.1.4记载DScnt按LDS issue/completion计数，保留编译器wait和原同步依赖。
接下来仅冻结当前MMA2上的first1及group1/first2小参数候选，先native及strict，后原入口相邻对照。
此时两项尚未计时；没有新增persistent、wave布局或pipeline。

## 当前正式 C25 + DS-first4 的 thread trace（2026-10-23）

正式保留版为 C25 + DS-first4，原入口第一正向合格候选组中值为 **81.5232 μs**，
反向确认中值为81.5084 /81.7594 μs，尚未达到80 μs。
本次trace对应这一实际版本；此前MMA3/3、MMA2/first2等章节是不同ELF的历史记录。

| 项目 | 当前实际值 |
|---|---|
| production source SHA256 | `d004fdd6a70b1564f07c0a8b5088422578c01f8b0818351881b205a32337bcf6` |
| GEMM1完整ELF SHA256 | `431e11cf71d9ab92596a5a424a5ea4308519d1eaa0f665649e11d02ba176b62d` |
| GEMM2完整ELF SHA256 | `4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be` |
| VGPR /SGPR /LDS | 797 /56 /327680 B |
| private /VGPR spills /SGPR spills | 0 /0 /0 |
| executable static instructions | 6296 |

当前symbol仍包含`dsfirst16`，已有目标guard内实际cap为4，显式
`AITER_FLYDSL_DS_FIRST_N`优先；不是凭symbol数字判断实际hint。

```text
a8w4_tdm_fp4_apre_t256x256x256_w2x2_b4_K7168_e64_act1_q1r8_cluster1x4_prefetch_bth1_wpt1_mg1x2_dsfirst16_planarlds_interleavelds_rolleddrain
```

仍为a07-3、`dsr1_ep4`、FlyDSL `0.3.3.dev882`，boot为
`cb28eed4-51ca-4cdc-bf5b-ecbc1d54db5f`。原用户workload，A preshuffle1/pin0，
warmup5/iters20、uniform/auto/seed0，`--all-simd --iteration 28 --ana-att`。
调用用户脚本`my_code/get_att_b_rocm_sdk.sh`；每个profiler/decoder child120秒。
label=`reboot20261023_best_dsf4_att_v1`，SIMD0～3及analysis全部rc0，结束没有测试GPU/KFD残留。

```bash
bash .codex_tmp/yadai_gemm1_param_20261021/tools/run_best_dsf4_att_v1.sh \
  NEW_UNIQUE_DS_FIRST4_ATT_LABEL
```

### 当前ELF绑定与本地证据

capture实际private cache的manager=`ce6594b23dc43a3b0647083255006387`，
GEMM1 key=`62c9ae1cec751fe7`、GEMM2 key=`da3c5914af92622d`。
原call inventory、sourceIR、globals、source与两stage完整ELF逐项匹配。
四个capture每个decoded PC和ISA编码均核对到真实GEMM1 ELF；没有往cache复制其他artifact。

231 regular members、6976203 B decoded archive已传到本地，未下载`.out`或private cache。
archive SHA256=`c3221b324bc126ac770750e12af7e6eb3e731aa71d9c667d2578ec899fb494c6`。
全部成员大小/SHA、wave事件及phase/opcode/wait统计本地独立复算一致；仅规范化68个路径分隔符字段。
`reboot20261023_best_dsf4_att_v1.local_archive_acceptance.json`和
`reboot20261023_best_dsf4_att_v1.local_analysis_acceptance.json`均`passed=true`。
analysis acceptance SHA256=`e128614ee4b209c763d990c2311c282bd311dfe488ce1783a59824035e42ab49`。

| SIMD | 本地UI directory | active /inactive waves |
|---|---|---:|
| 0 | `thread_trace/simd0/kernel/rpf_v3/ui_output_agent_41927_dispatch_562` | 16 /13 |
| 1 | `thread_trace/simd1/kernel/rpf_v3/ui_output_agent_3506_dispatch_562` | 16 /12 |
| 2 | `thread_trace/simd2/kernel/rpf_v3/ui_output_agent_42192_dispatch_562` | 16 /12 |
| 3 | `thread_trace/simd3/kernel/rpf_v3/ui_output_agent_30304_dispatch_562` | 16 /12 |

本地根目录为`.codex_tmp/yadai_gemm1_param_20261021/att/reboot20261023_best_dsf4_att_v1/`。
每个active wave执行1792条WMMA，64 waves共114688条，符合56个K128步骤×32个accumulators。
四个SIMD是独立capture，不能视为同一次dispatch的合并关键路径。

### 当前阶段与等待分布

以下是64个active waves的累计instruction timestamp gaps。

| 阶段 | 累计cycles | 累计占比 | 每wave中值cycles | 范围cycles |
|---|---:|---:|---:|---|
| prologue → 首WMMA | 244116 | 13.9271% | 3748 | 2991–5053 |
| WMMA mainloop | 1196949 | 68.2875% | 17968.5 | 17729–28912 |
| 最后input drain | 4032 | 0.2300% | 63 | 63–63 |
| activation /quant /scale /LDS + barrier | 253526 | 14.4640% | 3961 | 3949–3984 |
| output descriptor setup | 3200 | 0.1826% | 50 | 50–50 |
| output TDM /completion | 50986 | 2.9088% | 903 | 164–1015 |

按前一条opcode归因的timestamp gaps：WMMA32.9783%，`ds_load_b128`15.6989%，
`s_barrier_wait`10.1366%，`s_wait_tensorcnt`7.3622%，`s_wait_dscnt`3.6285%。
phase与opcode口径不能相加；gap包含arbitration/coexecution，不等于指令独占延迟或可移除开销。

| 当前actual PC | ISA | hits | timestamp gaps | trace stalls | 已执行WMMA范围 |
|---|---|---:|---:|---:|---|
| `0x250c` | `s_barrier_wait 0xffff` | 64 | 74819 | 74563 | 0–0 |
| `0xc5e8` | `s_barrier_wait 0xffff` | 64 | 55027 | 54899 | 1792–1792 |
| `0xc690` | `s_wait_tensorcnt 0x0` | 64 | 44399 | 44335 | 1792–1792 |
| `0x438c` | `s_wait_tensorcnt 0x2` | 368 | 42061 | 41693 | 96–1504 |
| `0x248c` | `s_wait_tensorcnt 0x3` | 64 | 38049 | 37921 | 0–0 |
| `0x5390` | `s_barrier_wait 0xffff` | 368 | 10406 | 9302 | 96–1504 |

mainloop仍占主要累计span，LDS/WMMA调度是下一批小改动的重点；prologue和quant epilogue
也有等待，但现有证据不能单独证明HBM、某个依赖或外部任务为其唯一原因。
final input drain仅0.2300%，不删除必要hardware fences来优化。

### 当前K128区间重新分解

`analyze_best_dsf4_ksteps_v1.py`只替换既有interval分析的label/ELF绑定，实际算法保持。
每32条WMMA建立一个区间，从该K128首WMMA至下一K128首WMMA，最后一个至最后WMMA后的第一条指令。
所有wave的56个interval精确相加为mainloop1196949 cycles。

| K128 index | K256 tile/subtile | 每wave中值cycles | 范围cycles |
|---|---|---:|---|
| 0 | 0/0 | 345 | 299–1608 |
| 1 | 0/1 | 334 | 324–466 |
| 46 | 23/0 | 326 | 314–359 |
| 47 | 23/1 | 369 | 306–429 |
| 48 | 24/0 | 407 | 349–476 |
| 49 | 24/1 | 292 | 289–305 |
| 50 | 25/0 | 358 | 355–396 |
| 51 | 25/1 | 291 | 291–301 |
| 52 | 26/0 | 320 | 315–375 |
| 53 | 26/1 | 297.5 | 297–299 |
| 54 | 27/0 | 343.5 | 341–345 |
| 55 | 27/1 | 253 | 253–253 |

steady的subtile0中值多数326～334 cycles，subtile1多数301～302 cycles。
个别subtile0出现大gap，例如#4最高7867 cycles；这不足以确认其硬件或外部归因。
区间包含prefetch、调度和coexecution，不能当成K128独占硬件延迟，也不能推算整kernel可节省μs。
完整记录`logs/best_dsf4_kstep_detail_v1.json`，SHA256为
`82254fdb4079b597e5d84aa120b779ac7a00f860e8772e5e1ef457449938f2a9`。

### 后续实验与支持资料

DS-first2、udiv和pow2已经在当前DS-first4基准上完成并被淘汰，不重复运行。
下一批仅测试尚未计时的DS-first3和DS-first5：目标guard内cap4各改一个数字，+1/-1行。
实际hint总量、scope外及显式env行为检查保持，数学、输入输出、同步、shape准入没有修改。
先完成native与E64/seed0完整MoE逐字节检查，再做原入口相邻control完整三轮；
性能双侧改善才反向，正反向四项改善且额外strict/scope全部通过才采用。

硬件依据：CDNA5 ISA §5.7.1.4（文档p56）规定DScnt按LDS issue/completion计数，
§5.7.2.1（p56）说明XDL queue bit允许连续issue且可能影响coexecution；
§7.12.1（p98）仍要求满足WMMA数据hazards。保留编译器wait和源码fences。
Shader Guide §5.2/§5.2.2（p212–213）说明wave priority与workgroup内优先级，
只用于理解arbitration；本批没有修改S_SETPRIO或dispatch priority。
技术事实与优化方向推断分开，实际收益须由原入口性能对照确认。

E65仍为原t64/w1x4 fallback；此前E65精度通过不能声称已完成E64/E65共用compiled GEMM1，
固定shared expert1 routing的历史要求也仍未完成。

### 后续保留版的 trace 绑定说明（2026-10-23）

DS-first5 已通过正反向性能对照、追加 strict 和 scope 检查并保留。
当前 source 为 `e53ada2baecb0c8e0185b815052281952213f0c81c5e4413c621baccfb06f93e`，
GEMM1 ELF 为 `62037bcfa621ff1c2d378e0e5face1479e6af47fbcb5beb8bacf8ee39d003897`。
本文件上面的 `reboot20261023_best_dsf4_att_v1` 仍明确绑定 DS4 的
`431e11cf71d9ab92596a5a424a5ea4308519d1eaa0f665649e11d02ba176b62d` ELF。
不将其 PC、timestamp gaps 或阶段比例声称为当前 DS5 的已采集 trace。
该历史 trace 继续作为定位后续实验方向的依据；DS5 当前没有重新采集的 trace。

lastquad 的正式编译及初步完整MoE逐字节检查已通过，本地独立preparation验收为`passed=true`。
候选ELF为`bf47990d73b7a5759cadfe4013b17e7edbe3623332dbedbdde18315909a00dfe`，
资源与DS5相同，但尚未采集该候选trace、完成性能对照或采用。
因此本文件中的DS4 PC和timestamp统计不归属于lastquad；准备流程耗时也不作为kernel性能。
当前最终目标为争取70 μs，后续对照数据继续记入参数遍历文档。

### 当前 DS5 ISA 是否有 s_wait_alu（2026-10-24）

有。重新核对当前正式保留source `e53ada2baecb0c8e0185b815052281952213f0c81c5e4413c621baccfb06f93e`，
及实际ELF `62037bcfa621ff1c2d378e0e5face1479e6af47fbcb5beb8bacf8ee39d003897` 对应的完整反汇编，
`s_wait_alu`共有170处静态指令，与ELF inspection记录一致。

| 指令形式 | 静态数量 |
|---|---:|
| `s_wait_alu depctr_vm_vsrc(0)` | 53 |
| `s_wait_alu depctr_va_vdst(0)` | 37 |
| `s_wait_alu depctr_va_vdst(0) depctr_vm_vsrc(6)` | 4 |
| 其他非零counter阈值形式 | 76 |
| 合计 | 170 |

本地文件为`.codex_tmp/yadai_gemm1_param_20261021/logs/mma2_first1_bitsearch_c25_dsf4_dsfirst5_v1_cpu_v1.isa.log`。
其文件名保留当时基于DS4准备DS5候选的命名；本节通过完整ELF SHA确认它对应当前DS5保留版。
第一处位于PC`0x2538`，指令`s_wait_alu depctr_va_vdst(0)`，machine word为`BF880F9F`。
170是静态编译指令数，loop/分支决定运行时执行次数；数量本身不说明实际等待cycles或kernel耗时占比。

Shader Guide rev65 §4.3.7.4（p91–92）记载：`VA_VDST`用于跟踪VALU完成及目的VGPR写回，
`VM_VSRC`跟踪VMEM源VGPR是否读完；`S_WAIT_ALU`可等待指定counter降到给定阈值。
该资料用于解释指令含义，本节没有由静态指令数推导性能节省，也没有修改这些wait。

### 当前 DS5 pin0/pin1 的新 trace 与 VGPR 控制指令（2026-10-24）

本次重新采集当前 DS5 的 pin0 和原版 pin1，各覆盖四个 SIMD，并分别绑定实际运行的 ELF。
此前 DS4 trace 的 PC 和阶段统计继续只属于 DS4；本节新增证据属于 DS5。
两版每版 64 个 active waves，每个 active wave 动态执行 1792 条 WMMA，计算量一致。
采集和解码全部成功，decoded 文件已下载本地，没有下载 `.out`。

| 项目 | DS5 pin0 | 原版 pin1 |
|---|---:|---:|
| VGPR | 797 | 1024 |
| SGPR | 56 | 66 |
| LDS | 327680 B | 327680 B |
| private / VGPR spill / SGPR spill | 0 / 0 / 0 | 0 / 0 / 0 |
| 静态 executable 指令 | 6308 | 5423 |
| 静态 `s_set_vgpr_msb` | 1198 | 459 |
| 静态 `s_wait_alu` | 170 | 156 |
| 采集 wave 的 mainloop 中值 | 18024 cycles | 20453 cycles |

pin0 ELF：`62037bcfa621ff1c2d378e0e5face1479e6af47fbcb5beb8bacf8ee39d003897`。
pin1 ELF：`ee76c901ebf710caaeeb99952d796b246a182536b3e59ab4a6f41fe20ff64a3b`。
wave cycles 不能直接换算为整个 kernel 可节省的 μs，静态 control 数量也不等于动态等待成本。
pin1 还没有启用部分 pin0 的既有优化，因此差异不能全部归因于 VGPR pin。

当前每个 WG 分配 320 KiB LDS，已达到硬件每 WGP 可用 LDS 上限。
Shader Guide rev65 §1.1 与 MI455X CDNA5 whitepaper 的 WGP/LDS 说明均记载 384 KB WGP SRAM 中
最多 320 KB 用于 LDS，其余 64 KB 用作 vector cache。因此当前配置受 LDS 限制，每 WGP 最多驻留一个 WG。
单独降低 VGPR 使用量不能提高这个 WG 驻留上限；VGPR 编号实验的收益须来自指令调度或减少寄存器复用等待，
不能预先声称来自 occupancy 提升。

当前 ATT decoder 不单独输出 `s_wait_alu` / `s_set_vgpr_msb` 的动态事件，
`code.json` 中这两类指令的 hit/stall 都为 0。这里的 0 表示未报告事件，不能说明指令无成本。
分析工具从相邻已报告 PC 之间的直线指令路径重建被省略的 control span，并排除 branch。
包含 control 的区间占累计 wave timestamp gaps：pin0 为 42.6529%，pin1 为 15.9504%。
区间还包含前一条指令执行、arbitration、coexecution 或 barrier 等待，
不是 control 指令独占延迟，也不是可以直接删除的时间比例。

pin1 在 `0x38d0 / 0x40a0 / 0x4868 / 0x5030` 等区间出现约 17 cycles 的中值间隔。
进一步核对 `0x38c0` 到 `0x3900` 的 machine ISA：

```asm
v_dual_add_nc_u32 v14, s13, v20 :: v_dual_add_nc_u32 v16, s13, v23
s_wait_alu depctr_vm_vsrc(0)
v_dual_add_nc_u32 v24, s15, v21 :: v_dual_bitop2_b32 v25, s50, v22 bitop3:0x54
s_wait_alu depctr_va_vdst(0)
ds_load_2addr_b32 v[10:11], v14 offset0:128 offset1:160
ds_load_2addr_b32 v[12:13], v16 offset0:32 offset1:96
s_wait_alu depctr_vm_vsrc(1)
ds_load_2addr_b32 v[14:15], v14 offset0:192 offset1:224
```

`v14 / v16` 刚由 VALU 计算，然后被 DS 当作地址读取，`VA_VDST` wait 涉及真实地址 RAW hazard。
后续 DS 又把 `v14` 用作目的寄存器，`VM_VSRC` wait 还涉及地址寄存器复用。
`0x3c78` 到 `0x3cbc` 的 `v4 / v6` 地址有类似关系。
现有证据不能将这些 wait 判定为实现 bug，不能直接删除它们。
优化应尝试改变寄存器复用或在依赖间隔中安排独立工作，保留正确的 hazard 保护。
硬件依据为 Shader Guide rev65 §4.3.7.4 / §4.3.7.4.1（p91–93）与 CDNA5 ISA §3.3.2.3 / §15.5。

已编译一个限定目标 GEMM1 的 scale slot 候选：slot0 的 ScaleA/ScaleB 固定为 `v128:131 / v132:135`，
slot1 为 `v136:139 / v140:143`；A/B `v256:511` 与 C `v512:1023` 保持。
资源仍为 VGPR1024/SGPR66/LDS327680 B、zero spills，静态 wait 154、MSB 456。
该候选随后通过完整 MoE 22,020,096 B 零字节差异及原入口验证，GEMM2 ELF 保持。
前/候选/后 GEMM1 中值为 86.8533 / 86.4401 / 85.9678 μs，未同时优于两侧 control，已放弃。
少两条 wait 和三条 MSB 没有转化为稳定性能收益，不能从静态数量推断空泡已经消除。

只增加 `HW_REG_WAVE_SCHED_MODE bit2` 的 ISA 实验已完成：前/候选/后 GEMM1 中值为
86.4148 / 83.9922 / 86.4549 μs，原 precision gate 全通过，GEMM2 ELF 保持。
它确认 scheduling 设置在 pin1 下有约 2.8% 相邻改善，没有证明 control span 全部是无效空泡。
最终落地仍须真实 FlyDSL 源码对照，正式保留版仍为 pin0，81.6350 μs。

本地证据目录：

```text
.codex_tmp/yadai_gemm1_param_20261021/att/reboot20261024_dsf5_pin0_att_v1/
.codex_tmp/yadai_gemm1_param_20261021/att/reboot20261024_dsf5_pin1_att_v1/
```

各目录的 `best_bound_trace_analysis_v1.json` 与 `vgpr_wait_analysis_v1.json` 保存绑定及完整统计。
性能和复现命令见 `my_code/moe_gemm1_hyg_dsr1_parameter_tuning.md` 的本轮 pin1 章节。

当前另有 pin1 启用已有 DS5 小优化的真实 FlyDSL 候选，ELF 为
`71ef64dd038b26ce18d883166cc96df28a179d9ccf5db296c350e095ee1cab2b`。
完整 MoE 零字节差异及 GEMM2 ELF 保持已验证，但还没有该候选的 thread trace 或正式性能对照。
后续 VGPR 编号实验基于这个候选，仍保留编译器 hazard waits；不将本节 pin0/pin1 的 PC 或 cycles
归到尚未采集 trace 的新候选。

本轮 VGPR 编号初筛继续保留所有 hazard waits：rotated 中值 80.9960 μs，compact128 中值 81.2195 μs。
compact128 的 VGPR 从 1024 降至 896，static MSB 从 770 降至 762，wait 仍为 194；
完整 MoE 与原版零字节差异，但尚未完成正式交错对照或采集该候选 trace。
这些修改调整的是 VGPR 的 MSB 编号区间，不能据此声称改善了 VGPR SRAM bank conflict。
Shader Guide rev65 §5.7.1（p230）的 internal ops 说明 `S_WAIT_ALU`、`S_SET_VGPR_MSB`
在某些情况下可以与前一条 normal instruction co-issue，因此静态数量也不能直接换算为节省 cycles。

### rotated 候选四 SIMD trace 与地址寄存器实验（2026-10-24）

新 capture 为 `reboot20261024_dsf5_pin1_rotated_vgpr_att_v1`，仍属于实验候选。
使用用户指定 `my_code/get_att_b_rocm_sdk.sh`，四 SIMD、iteration28、`--ana-att`，
fresh private cache；没有向 cache 注入 ISA。实际原入口 GEMM1 ELF 为
`b5d3fc63fe64e90f170f308021ccf7f9ee8eb77f4b70049e60c5c5137b89b4ed`。
source SHA256 为 `7acfac3b58dc058cd0fa82a7dc8c98d8dbdcdd1e8af97d89ebc023377c36db5d`。
真实 source-dependent keys、实际 ELF 与 decoded PC 全部绑定通过，采集后源码恢复。

decoded 证据已下载到本地，排除 `.out`，127 个 regular members 的 SHA256 全部复核。
archive SHA256 为 `7ab2b5f88e63e21bed86948b8bab221cefc6d1053b028a6d24d4015941cd2d48`。
本地重新核对 ELF/PC、64 个 active waves、每 wave 的 1792 条 WMMA，phase 与 opcode 统计均与远端一致。

| 采集区间中值，cycles/wave | DS5/pin0 | 原版 pin1 | rotated/pin1 |
|---|---:|---:|---:|
| prologue 至首条 WMMA | 3636.5 | 3978.0 | 3767.0 |
| WMMA mainloop | 18024.0 | 20453.0 | 17722.5 |
| final input drain | 63.0 | 54.0 | 63.0 |
| activation/quant/scale LDS 与 barrier | 3953.0 | 4027.0 | 4043.0 |
| output descriptor setup | 50.0 | 49.0 | 49.0 |
| output TDM 与 completion | 903.0 | 910.5 | 903.0 |

rotated 相对 pin0 的 mainloop 区间中值减少 301.5 cycles，约 1.67277%；prologue 和 activation/quant 区间略长。
这些来自独立 SIMD captures，区间中值也不能相加构造某一个 wave 的实际总耗时，
更不能换算为整个 dispatch 可节省的 μs。整个 kernel 的结论仍以原入口空闲交错测试为准。
正向改善较小，反向最后一组 rotated 仅比 pin0 快 0.0189 μs，尚无稳定的新收益，暂不采用。
本轮 DS5/pin0 独立重测中值为 82.2843 μs；前面的 81.6350 μs 是此前时段的正式记录。

rotated 包含被 decoder 省略的 control instructions 的直线区间累计占 wave timestamp gaps 的 25.88108%。
这里仍包含前一条 instruction 执行、arbitration 和 barrier 等待，不是可删除的 control 成本。
`0x3358→0x3368`、`0x61ec→0x61fc`、`0x42e0→0x42f0` 等区间包含
`s_wait_alu depctr_va_vdst(0)`，观测 timestamp gap 中值均为 18 cycles。
下一条为使用 `v12` 地址的 `ds_load_2addr_b32`；wait 的必要性必须结合前面完整的 VALU/DS 数据依赖判断，
不能因为相邻记录之间存在间隔就删除 wait。
硬件依据继续使用 Shader Guide rev65 §4.3.7.4 / §4.3.7.4.1（p91–93）、§5.7.1（p230）与 CDNA5 ISA §15.5。

直接固定 LDS 地址的四个源码写法（packed32、packed64、scalar32、opaque32）
均在 CPU 编译阶段报 `LLVM ERROR: overlapping live intervals in explicit register storage`。
它们没有 GPU launch 或性能数据，暂时搁置该 pin 写法。
随后改为在 input LDS loads 后增加仅限目标 GEMM1 的 empty side-effect asm，
延长只读地址的 live interval，尝试避免 DS 目的寄存器复用这些地址。
这两个写法不固定地址 VGPR，不删除 waits，不改变输入、计算或输出。

| CPU 候选 | 新增地址最后使用 | VGPR / SGPR / LDS B | static executable / wait / MSB |
|---|---|---|---|
| rotated 原版 | 无 | 1024 /62 /327680 | 5836 /194 /770 |
| addr_keepalive | A/B/ScaleA/ScaleB | 1024 /62 /327680 | 5889 /185 /804 |
| scale_addr_keepalive | ScaleA/ScaleB | 1024 /62 /327680 | 5875 /186 /800 |

两者 CPU 编译通过，private 与 spills 均为 0。完整 MoE 输出与 DS5/pin0 零字节差异；
三轮原入口初筛中值分别为 82.5987 /81.9174 μs，均已下载并通过本地复核。
静态 wait 下降同时伴随 MSB 增加，初筛尚无稳定收益，未作为保留版。

随后测试 `addr_precompute`：仅在当前/下一 stage 地址计算后、compute 调度之前消费地址，
尝试将地址计算与后面的 WMMA/DS 重叠。实际 ELF 为
`38ca9f6a027d33365f93809b668976ac377485109e77c6b964d44271aaef181e`，
资源仍为 VGPR1024 /SGPR62 /LDS327680 B、private/spills0；static executable /wait /MSB
为 5866 /181 /780。完整 MoE 输出零字节差异，原 precision gate 通过。
初筛中值 80.8548 μs，正向前/候选/后中值为 81.6921 /81.6224 /82.1347 μs；
最小提升只有 0.0697 μs。后续反向候选/control/候选中值为
82.2147 /81.4494 /81.7677 μs，前后候选分别慢 0.7653 /0.3183 μs；
九轮原 precision、实际 key/ELF 与 ownership 通过，原表下载并本地复核通过。
反向未复现稳定收益，结束本候选验证，不采用，生产继续保留 DS5/pin0。
该候选尚未采集 ATT，不能仅凭 181 条静态 wait 宣布空泡消失。
完整原表、实际 key/ELF 绑定、本地复核和复现命令记录在参数优化文档中。

本地证据目录为 `.codex_tmp/yadai_gemm1_param_20261021/att/reboot20261024_dsf5_pin1_rotated_vgpr_att_v1/`。
`best_bound_trace_analysis_v1.json`、`best_bound_trace_local_review_v1.json` 和 `vgpr_wait_analysis_v1.json`
保存 phase、PC/ELF 复核与 control span 统计；`logs/` 下同名 capture、binding 和 archive receipt 保存采集绑定。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_pin_candidate_att_v1.py \
  dsf5_pin1_rotated_vgpr_v1 reboot20261024_dsf5_pin1_rotated_vgpr_pin0_prepare_v2 \
  reboot20261024_dsf5_pin1_rotated_vgpr_att_v1
```

再次复现需使用未占用的 capture label。

### t128 资源容量实验的边界（2026-10-24）

后续私有 `t128/w2x2/b2`、`b3` 候选复用现有实现，并修正实验内部 retile 与上游 M128
ScaleA chunk 的兼容性；GEMM1 输出的 `quant_wmma_rep=8` 和 GEMM2 ELF 保持不变。
两项完整 MoE seed0 输出均与 DS5/pin0 零字节差异，原 precision gate 通过。
CPU 资源分别为 VGPR455/SGPR58/LDS131072 B、VGPR456/SGPR62/LDS163840 B，private/spills 均为 0。
static executable/wait/MSB 分别为 3480/154/275、3598/117/281。

Shader Guide rev65 §3.3.2.1 记载 MI450 每 SIMD 1024 VGPR、wave32 按 16 分配，
WGP 最多使用 320 KiB LDS。上述资源满足更高 wave/WG 驻留的容量条件，
尚无 t128 ATT 证明实际 occupancy 提高，也不能据此推断 `s_wait_alu` 动态成本下降。
性能和完整复现证据继续记录在参数优化文档中；生产仍为 DS5/pin0。

t128 原入口三轮初筛中值为 b2 174.412 μs、b3 134.814 μs。
随后仅恢复 GEMM1 小调度设置的 b3/closing8、closing4 也只有 133.655 /134.738 μs，
两项资源均为 VGPR472/SGPR58/LDS163840 B、private/spills0，完整 MoE seed0 输出零字节差异。
四项均未接近保留版，已结束本轮 t128 遍历并恢复 DS5/pin0。
静态资源满足两条 wave 的容量条件，未转化为这四项测试的性能收益；
未采集 t128 ATT，不推断实际 occupancy，也不将原 t256 的动态 PC/占比套用到 t128 ELF。

### 固定 LDS 地址 VGPR 候选的 ATT（2026-10-24）

`dsf5_pin1_rotated_addr_constraints_v1` 保留原 w2x2、四级输入 ring 和非 persistent
执行方式，将 current/next LDS 地址通过 tied identity constraints 固定为 `v32:35`、
`v36:39`。C 为 `v256:767`，A/B 两份 rmem 为 `v768:895`、`v896:1023`。
source SHA256 为 `bba7f5a2596261f5c6d9357e2fc7ec85f1c286e7abe0c3566450beb115c04a54`，
真实 GEMM1 ELF 为 `deea50e0ae511e21d6d7a0a825ec53e2bf664ebfed82b29cc7424093ceda0488`。
资源为 VGPR1024 /SGPR62 /LDS327680 B，private 与 spills 均为 0；
static executable /`s_wait_alu` /`s_set_vgpr_msb` 为 5724 /118 /750。

原入口正向候选中值为 81.0531 μs，两侧 DS5/pin0 为 82.2673 /81.9933 μs；
反向候选中值为 81.1796 /81.1247 μs，中间 control 为 81.9807 μs。
共十八轮原 precision、实际 key/ELF 和全机 ownership 检查通过，原表已在本地复核。
各相邻比较改善 0.97718%～1.47592%，仍未达到 70 μs。此处的性能来自原入口空闲测试，
ATT 日志内的 instrumented kernel-bench 数字不作为性能数据。

四个 SIMD capture 的 `att_status` 和 `analyze_status` 均为 0。fresh cache 的原始
source-dependent key、实际 ELF 与 decoded PC/ISA 已绑定；64 个 active waves 均执行
1792 条 WMMA。只下载任务所需的 decoded evidence，不含 `.out`，139 个 regular members
逐文件核对 SHA256 后，本地重算 phase 与 control 区间统计，结果与远端一致。
跨平台复核仅统一 `file`、`ui`、`wave` 字段中的路径分隔符，数值和指令内容未转换。

| 采集区间中值，cycles/wave | DS5/pin0 | rotated/pin1 | 固定地址/pin1 |
|---|---:|---:|---:|
| prologue 至首条 WMMA | 3636.5 | 3767.0 | 3445.5 |
| WMMA mainloop | 18024.0 | 17722.5 | 17331.5 |
| final input drain | 63.0 | 63.0 | 63.0 |
| activation/quant/scale LDS 与 barrier | 3953.0 | 4043.0 | 4060.0 |
| output descriptor setup | 50.0 | 49.0 | 49.0 |
| output TDM 与 completion | 903.0 | 903.0 | 909.5 |

mainloop 相对 DS5/pin0 的区间中值减少 692.5 cycles，约 3.84210%；
相对 rotated 减少 391.0 cycles，约 2.20623%。activation/quant 比 DS5/pin0 长
107 cycles。四个 SIMD 是独立 captures，区间中值不能相加构造某个 wave 的总耗时，
也不能换算为整个 dispatch 可节省的 μs。完整 kernel 的结论以上面的交错实测为准。

固定地址 capture 的累计 wave timestamp gaps 中，mainloop 占 67.34212%，
activation/quant/scale LDS 与 barrier 占 15.66951%，prologue 占 13.57374%。
前条 opcode 归属的 WMMA、`ds_load_b128`、`s_barrier_wait` 区间分别占
35.31953% /13.50578% /10.96388%。这些区间包含 arbitration、coexecution 及省略的
control instructions；不能当作独占执行时间，也不能与 phase 占比相加。

decoder 没有独立报告 `s_wait_alu` 或 `s_set_vgpr_msb` 动态事件，这不表示硬件未执行它们。
由真实 ELF 重构的直线 control spans 累计占 timestamp gaps 的 28.21821%，
仍不等于可删除空泡。最大的 `0x2944→0x294c` 区间包含 prologue
`s_barrier_wait 0xffff` 和 `0x2948` 的 `s_wait_alu depctr_va_vdst(0)`，
其中值为 1247 cycles；不能把 barrier 的等待归入 ALU wait 成本。

mainloop 的 `0x32c8→0x32d8`、`0x61b0→0x61c0`、`0x520c→0x521c` 区间
中值为 17 cycles。以第一项为例，`v_or3_b32 v39, v26, s14, 0x22000` 后有
`s_wait_alu depctr_va_vdst(0)`，随后读取 `v34` 地址的 `ds_load_2addr_b32`。
其更早上下文仍有 current/next 地址的计算和 `depctr_vm_vsrc(0)` 等待；
需要判断完整 RAW/WAR 依赖，不能仅因为最后一条 VALU 写的是 `v39` 就删除 wait。
硬件依据为 Shader Guide rev65 §4.3.7.4/.1（p91–93），以及 CDNA5 ISA
§3.3.2.3（p17–18）：SCHED_MODE2 的相关 hazard 由软件保证，VGPR MSB 设置作用于后续指令。

证据目录为 `.codex_tmp/yadai_gemm1_param_20261021/att/reboot20261024_dsf5_pin1_rotated_addr_constraints_att_v1/`。
其中 `best_bound_trace_analysis_v1.json`、`vgpr_wait_analysis_v1.json` 和
`best_bound_trace_local_review_v1.json` 保存分析与本地复核；`logs/` 同名前缀保存 capture、
binding 和 archive receipt。archive SHA256 为
`2a01a050700eb84e67bf572ee4135adf8bfbba5a5b37f409fcadd62765583acb`。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_pin_candidate_att_v1.py \
  dsf5_pin1_rotated_addr_constraints_v1 \
  reboot20261024_dsf5_pin1_rotated_addr_constraints_pin0_prepare_v1 \
  reboot20261024_dsf5_pin1_rotated_addr_constraints_att_v1
```

复现需换未使用的 capture label。采集和绑定完成后 production source 已恢复 DS5/pin0。
采集时该候选仍需动态 routing replay 与适用范围验证；后续完成情况如下。

### ATT 所对应 code object 的范围修正与严格验证（2026-10-24）

最终候选 `dsf5_pin1_rotated_addr_scoped_v1` 将 pin1 的 `DS_B=4` 扩展限定为完整
`compact_pin_gemm1` guard，修复了此前扩展到 act2 和 SiLU 非量化输出的问题。
其 source SHA256 为 `2273b14396be413c8789c98c9a42c7be3f8678ee9ba847826bc16c56854de17d`。
目标完整 ELF 仍是上述 capture 的 `deea50e0…ceda0488`，CPU 复编及本地逐字节核验通过。
因此本节 ATT 继续描述同一 code object；此处没有把其他候选的 PC 或 trace 复用过来。

范围外 11 项完整 ELF 与 production 相同，seed0/1/2 的目标路径有效 payload/ScaleA、
完整 MoE 输出均为零字节差异；dynamic replay 包括 eager、CUDA Graph、相同地址改写
A/ScaleA/psum、empty/边界/尾部 counts 及输入恢复。K4096 fallback 和 bias 非量化
输出的 guard 外验证也通过，不能将这两项视为目标 kernel 的数值覆盖。
五份原日志及实际 ELF 下载后独立复核，通过。详细命令、指标和范围在
`moe_gemm1_hyg_dsr1_parameter_tuning.md` 的最终范围候选章节。
生产 source 尚未替换，仍为 DS5/pin0；最佳候选的实测中值约 81.1 μs，未达到 70 μs。

地址改为 `v64:71` 的候选 CPU 编译后 static executable /`s_wait_alu` /
`s_set_vgpr_msb` 为 5723 /117 /750，资源使用不变。但原入口 addr32 → addr64 →
addr32 的三轮中值为 81.1754 /81.1498 /80.2666 μs，相对两侧分别改善 0.03154%、
回退 1.10033%，没有稳定收益，已放弃。九轮原始表、实际 key/ELF 和 ownership 证据
下载后本地复核通过。少一条静态 wait 没有证明动态空泡减少，未为这个淘汰候选重抓 ATT。
### VGPR MSB 文档语义及本轮调度实验的限制（2026-10-24）

Shader Guide rev65 §3.3.2.3（文档 p34–35）说明 `S_SETREG_B32 MODE` 有 extra delay，
`S_SET_VGPR_MSB` 没有这项 extra delay。后者仍是一条需要 issue 的指令，文档没有给出
当前 kernel 中每条的独占动态成本；静态 MSB 数量减少不能直接换算为可消除空泡。
CDNA5 ISA §3.3.2.3（p17–18）的地址公式为
`logicalVgprAddr[9:0] = { MSBs[1:0], ISA[7:0] }`，并明确写 bits `9:8`。
Shader Guide 同节公式和 0..1023 表一致，但后文写成 bits `9-10`，存在位号文字冲突。
本轮按 ISA 的公式和实编 ELF 解码核对寄存器，不使用冲突文字推导其他寻址范围。

addr32/pin1 的 MMA3 候选将 static `s_set_vgpr_msb` 从 750 减到 640，
`s_wait_alu` 仍为 118，资源使用量不变；其余首组、closing、DS-first 四项也已完成 CPU
编译和 seed0 完整 MoE preparation。编译器生成的 DS wait 序列有所变化；排除 DS wait
后的 TDM/barrier/store wait 有序序列保持，不能把所有 native fences 说成完全相同。
这些候选正在按原入口初筛，尚无匹配 ATT 结论，不复用 addr32 reference 的 PC 或 phase
统计为新 ELF 的瓶颈证明。实际性能和数值验证记录见参数优化文档。

上述五项原入口初筛已完成并本地复核：MMA3 中值 81.4115 μs，虽然 static MSB
减少 110 条，仍慢于两侧 addr32 的 80.9771 /80.9981 μs。first2 中值 80.6691 μs，
static MSB 为 732，正在做反向确认。这些新 ELF 尚未抓匹配 ATT，不据静态数量
声称消除了动态空泡。全部原始轮次和比例在参数优化文档中保留。

Shader Guide rev65 §3.3.4（文档 p36）、§4.7.1（p160）记载 WGP SRAM 共 384 KiB，
其中 LDS 最多 320 KiB，至少 64 KiB 留给 WGP$。当前 addr32 kernel 的 metadata
请求 LDS327680 B，即 320 KiB。由此可推导，同一 WGP 无法同时容纳两个都请求
320 KiB LDS 的 WG；仅减少 MSB 指令或 VGPR 数不会改变这一 LDS 容量限制。
这不是从 ATT 得出的 dispatch 瓶颈占比，也没有证明减少 LDS 后性能必然提高。

first2 反向三组中值为 81.2911 /80.4543 /80.1907 μs，前后候选对中间 reference
分别回退 1.04009%、改善 0.32764%，未复现双侧稳定收益，因此放弃该调度参数。
没有为这个淘汰候选重抓 ATT；addr32 的匹配 ELF/PC 和动态分析继续作为布局优化依据。

### 降低 VGPR 数量的三项新布局初筛（2026-10-24）

在相同 addr32/scoped 调度下，`c128_ab640`、`c64_ab576`、`c384_ab128` 的 VGPR 请求分别为
896 /832 /896，static `s_set_vgpr_msb` 为 679 /711 /678，`s_wait_alu` 都为 118。
SGPR62、LDS327680 B、private0、spills0 不变，数学和内存操作 count 保留。
原 ELF/ISA 已逐 machine word 复核，native count 使用修正 opcode 拼法后的 v2 记录。

| 布局 | 实际 ELF SHA256 | 原入口三轮 GEMM1 中值 μs |
|---|---|---:|
| `c128_ab640_addr32` | `1a4ae157b7c657a71d4a0e6e7b846f517a34cbdf81d78ab032ffdc6ecd8d0867` | 80.7831 |
| `c64_ab576_addr32` | `66fcecc5e4752bc5185d2bb9805aa234dbb66b73fa22969d7f3c53eee8dbd55d` | 80.7642 |
| `c384_ab128_addr32` | `fba4a60c2be9ade3a56fbfe18ba1de1161accee3db8a18f59df2b8afb9cc7265` | 81.2661 |

两侧 addr32 reference 中值为 80.8949 /81.7530 μs。前两项相对前侧只改善 0.13820% /
0.16157%，尚需反向确认；第三项没有双侧收益，淘汰。
这三项没有匹配 ATT，不能复用 `deea50e0…ceda0488` 的 PC、phase 或 control spans
来宣称新布局消除了动态空泡。减少 VGPR 或 static MSB 也没有改变 320 KiB LDS 对 WG 并驻的容量限制。
全部原始轮次、双侧比例、复现命令与本地复核保留在参数优化文档的相应章节。

`c64` 反向候选前/reference/候选后三轮中值为 81.3525 /81.3566 /81.2090 μs。
两侧改善仅 0.00504% /0.18142%，相对轮间波动缺乏足够稳定证据，未采用；没有为其重新抓 ATT。

### pin1 M64 无效 WMMA 跳过的 native 证据（2026-10-24）

新候选 `dsf5_pin1_empty_m64_scoped_v1` 保留完全相同的输入 ring、输出 helpers 和优化 guard，
只跳过完全无有效行的 M64；部分有效分组仍完整计算。实际 ELF 为
`dd44fff8fab75aedf6fe7a6f217b4f4cf3b9cb1825be70089a282a0287de9565`。
VGPR1024 /SGPR62 /LDS327680 B /private0 /spills0 不变；static executable /`s_wait_alu` /
`s_set_vgpr_msb` 从 5724 /118 /750 变为 5592 /112 /449。

机器字核对及有界 CFG 重算确认 35 个 `s_cbranch_scc1` site，可越过各自 16 条 WMMA 的区域，
区域内不含 DS/TDM/barrier。不是仅用 EXEC=0 masking 作为跳过证明。
DS-to-WMMA、WMMA-to-DS 的物理 VGPR 检查在模型覆盖范围内通过；不能代替完整 hardware hazard 验证。
编译器将部分 fence block 放到 ISA 后部，线性文本中的 fence 顺序变化；真实 CFG 投影后的
TDM/barrier/wait 事件语言与 reference 一致。该比较忽略 branch condition 和 descriptor 值，
其限制以及 DS wait0 static sites 从 18 到 16 的变化均已记录。

seed0 完整 MoE 输出 22,020,096 B 与生产零差异，原 precision 通过，GEMM2 为原 pin1 ELF。
该候选尚无匹配 ATT，不能把 reference 的 PC 或 28.21821% control spans 当作它的动态收益。
原入口独立性能夹测、多 seed/dynamic 验证和最终采用决策在参数优化文档中继续记录。

九轮原入口夹测随后完成：reference 前 /M64 /reference 后 GEMM1 中值为
81.5211 /84.3001 /80.8001 μs。M64 分别回退 3.40893% /4.33168%，明确不采用。
全部原表、实际 ELF、source-dependent call keys、ownership 已下载并逐项复核；
详细三轮数据及 SHA256 见参数优化文档。该对照同时改变 branch CFG 和调度，不能
把 118→112 wait、750→449 MSB 的数量变化单独当作某类指令的成本实验。

### `s_wait_alu` /`s_set_vgpr_msb` 的性能影响：已知事实与未解决部分（2026-10-24）

已分析两类指令，但没有得到各自可独立归因的动态 μs 成本。

`s_wait_alu` 用于保护具体依赖：`VA_VDST` 追踪已发射但尚未写回的 VALU；
`VM_VSRC` 追踪 VMEM/LDS 指令尚未读取完毕的源 VGPR。SCHED_MODE2 关闭这两类
硬件检查，因此需要软件保证 RAW/WAR 安全。WMMA 和其他 VALU 的计数完成可以
乱序，不能仅按经过多少 cycles 或最后一条 VALU 的目的寄存器决定 wait 是否可删。
依据为 Shader Guide rev65 §4.3.7.1/.2（p86–88）、§4.3.7.4（p90–93）。

匹配最佳 reference `deea50e0…ceda0488` 的四 SIMD ATT 中，mainloop
`0x32c8→0x32d8`、`0x61b0→0x61c0`、`0x520c→0x521c` 含 wait 区间中值均为
17 cycles；它们仍包含前后指令执行和调度。`0x2944→0x294c` 中值 1247 cycles
还包含 `s_barrier_wait 0xffff`，不属于可独立归因的 ALU wait 延迟。

`s_set_vgpr_msb` 设置后续向量指令的 VGPR 地址高位，不是 memory fence。
Shader Guide rev65 §3.3.2.3 明确记载：`S_SETREG_B32 MODE` 有 extra delay，
`S_SET_VGPR_MSB` 没有这一额外延迟；§5.7.1 还描述了特定 co-issue 条件。
这不代表它无发射或调度成本，也不代表每次 MSB 切换必然产生固定长度空泡。
删改 MSB 必须保留后续每个 operand 的实际物理 VGPR 映射。

当前 decoder 没有单独报告两类指令的动态事件；重构的 28.21821% control spans
包含 WMMA、DS、barrier 等前后指令等待，不是两类指令的独占成本。
M64 候选减少两类静态指令但慢 3.4%～4.3%，证明数量不能预测收益；
VGPR 布局、分支与调度同时变化，仍不能据此断言某一类指令无影响。

后续实验应从具体物理 VGPR RAW/WAR、wait placement 和 MSB 切换上下文出发。
ISA injection 仅用于快速验证，最终采用必须落实为 FlyDSL 源码并通过原数值标准。

### M64 匹配 ATT：回退来自交错与同步配合的损失（2026-10-24）

候选的四个 SIMD captures 已完成，实际 GEMM1 ELF 为 `dd44fff8…7de9565`，
GEMM2 仍为原 pin1 ELF。source-dependent keys、完整 ELF 与 decoded PC/ISA 绑定通过；
尾部两个 barrier 校验保持。production source 已精确恢复。

64 个 active waves 中，21 个各执行 896 条 WMMA，43 个各执行 1792 条。
按实际 WMMA 数分组，而非继续假定全部执行 1792 条：

| capture 中的 wave 组 | waves | mainloop 区间中值 cycles | final input drain 区间中值 cycles |
|---|---:|---:|---:|
| reference：1792 WMMA | 64 | 17331.5 | 63.0 |
| M64：896 WMMA | 21 | 20004.0 | 379.0 |
| M64：1792 WMMA | 43 | 20589.0 | 71.0 |

进一步只比较完整执行 1792 WMMA 的 waves，将每 32 WMMA 作为一个 K128 计算组：
reference 的 3584 个组中，组内可见 DS load 数中值为 31（1792 组为31，1728组为33，
末组64组为0）；M64 的 2408 个组中，组内 DS load 数全部为0。
这直接显示新增条件 basic blocks 后，原 DS/WMMA 交错被拆开，不能仅用减少的 WMMA
数量预测收益。统计使用原事件顺序，未按耗时选择 wave。

执行 896 WMMA 的 SIMD1/2 组每 wave 在 mainloop 的 `s_barrier_wait` 前条 opcode
归属区间中值为 7355 /7057 cycles；reference 同 SIMD 为 974.5 /729.0 cycles。
仍执行全部 WMMA 的 SIMD0/3 mainloop 区间中值也从 17358.5 /17215.0 增至
20572.5 /20761.0 cycles。减少工作的一部分 waves 仍在等待同 WG/cluster 的其他
工作完成，新增条件判断和失去交错还拖慢了完整计算组。

这些是采样 wave 的区间证据，不是 dispatch 独占成本分解；四 SIMD 是独立 captures，
不能把上述 cycle 比例直接替换为原入口的 3.4%～4.3% 性能回退比例。
这版实现仍明确不采用。后续若再尝试跳过无效计算，必须保住 DS/WMMA 交错和同步配合。

证据目录为 `att/reboot20261024_dsf5_pin1_empty_m64_att_v1/`，archive SHA256 为
`4c4c97d6a4170ff0aed3b6e25be87ec6f01f4a90e0aed4513db96f484785fb0a`。
133 项任务文件逐项核 SHA256，并在本地重算 phase 与 control 报告，全部通过；不含 `.out`。
细分证据为 `logs/reboot20261024_m64_mainloop_opcode_groups_local_v1.json` 和
`logs/reboot20261024_m64_interleave_evidence_local_v1.json`。

### 按前后 start gap 减前条 latency 分析空泡（2026-10-24）

按用户建议改进指标，逐个真实 fallthrough 区间计算：

```text
residual = next_visible_start_timestamp - previous_visible_start_timestamp
           - previous_instruction_decoded_latency
```

SDK `source/lib/att-tool/wave.cpp:65` 将事件写为 `[time, category, stall, duration, pc_index]`；
本次 latency 使用 `event[3]`，即 `inst.duration`。其
`source/docs/how-to/using-thread-trace.rst:218` 定义 gfx10+ 的 Latency 为
`Stall time + Execute time`。

所有 PC 和 machine words 先绑定到实际 ELF；仅重构中间只有 `s_wait_alu` /
`s_set_vgpr_msb`（或单独标注的 auxiliary control）的直线路径，排除越过 branch
或其他隐藏 opcode 的区间。两类同时出现时只计为一组 `both`，不重复相加。
原始负残差保留，不在统计前截为0。

reference `deea50e0…ceda0488` 的64个active waves中，mainloop结果为：

| 中间 control 类型 | 区间次数 | 残差中值 cycles | P95 cycles | 正残差合计 cycles | 负残差次数 |
|---|---:|---:|---:|---:|---:|
| `s_wait_alu` | 7040 | 0.0 | 20 | 43857 | 0 |
| `s_set_vgpr_msb` | 76928 | 0.0 | 4 | 129159 | 36710 |
| 两类同时出现 | 1536 | 3.0 | 3 | 4608 | 0 |

上述正残差占全部64个waves累计timestamp gaps分别为2.64441% /7.78778% /0.27784%，
不是整个 dispatch 可直接节省的性能比例。负残差说明下一指令已在前一条完成前启动，
存在重叠或 co-issue；正残差则是该区间的候选空泡，仍需结合前后 opcode、实际依赖和
issue arbitration 确定能否消除，不能一律独占归因给中间 control。

| reference 区间 | 中间指令 | gap 中值 | 前条 latency 中值 | 逐样本残差中值 | 说明 |
|---|---|---:|---:|---:|---|
| `0x2944→0x294c` | `s_wait_alu depctr_va_vdst(0)` | 1247 | 1244 | 3 | 前条为 prologue barrier；扣除后不再将 barrier 等待误算为 ALU wait |
| `0x32c8→0x32d8` | `s_wait_alu depctr_va_vdst(0)` | 17 | 1 | 16 | 368次，持续正残差，优先检查完整地址RAW依赖 |
| `0x61b0→0x61c0` | `s_wait_alu depctr_va_vdst(0)` | 17 | 1 | 16 | 368次，结果同上 |
| `0x520c→0x521c` | `s_wait_alu depctr_va_vdst(0)` | 17 | 1 | 16 | 368次，结果同上 |
| `0x330c→0x3320` | `s_set_vgpr_msb` | 2 | 9 | -7 | 368次；WMMA后DS已重叠启动，不能计为正空泡 |

MSB-only 区间中也存在 `s_add_nc_u64→s_cbranch_scc1` 的8-cycle正残差，以及
`ds_load_b128→WMMA` 的4～5-cycle正残差。它们的前后指令类型不同，下一条 branch/WMMA
的就绪等待可能进入残差，应分别分析，不能仅根据同一个 `s_set_vgpr_msb` 类型判断原因。

全量结果为 reference/M64 对应 ATT 目录的 `control_gap_minus_latency_v1.json`；
复现工具为 `tools/analyze_control_gap_minus_latency_v1.py`。旧的28.21821% control spans
指标仍保留为历史统计，后续判断空泡以本节减去前条 latency 的残差为优先指标。


### 三处 wait 立即数的私有 ELF 对照（2026-10-24）

只把`0x32d4 /0x5218 /0x61bc`的`s_wait_alu depctr_va_vdst(0)`改为`(1)`，
所有其他ELF字节、TDM机器码和实际计算相同，指令数量不变。原ELF为
`deea50e0…ceda0488`，实验ELF为
`97d1c16c3e8536b6e8d5cacc3aa3e4d647173a8016f5248403c0cd8c56d97346`。

这些热点在wait前最后一条VALU写`v39`，第一条DS load读`v34`。放宽pending计数
用于检验保守等待，尚不是完整RAW/WAR安全证明；第四个同类wait `0x4278`未修改。
原入口的正向GEMM1中值为reference80.8042、实验80.4268、reference82.0335 μs；
反向为实验80.7306、reference80.9203、实验80.5977 μs，反向改善0.23443% /0.39866%。
18轮原seed0门槛、actual keys/ELF和空闲校验通过，未称为通用精度证明或生产源码成绩。

完整逐轮数据、运行命令和archive校验见`my_code/moe_gemm1_hyg_dsr1_parameter_tuning.md`。
用原`my_code/get_att_b_rocm_sdk.sh`成功采集该patched ELF的四SIMD trace；内层workload
指定patched私有cache及RUN_ONLY，覆盖脚本外层fresh-cache，避免重新编译成reference ELF。
每个decoded PC/ISA已绑定实际ELF。最后残差步骤因遗漏上传纯CPU脚本中断，保留失败记录，
补齐工具后复用这份capture恢复统计，完整残差结果见下一节。


### wait1 匹配 trace 的残差结果（2026-10-24）

四SIMD均完成，64个active waves各执行1792 WMMA；尾部两个barrier、实际call keys、
完整ELF、decoded PC/ISA及三个立即数均绑定通过。按同一工具在本地从156项下载证据
重新计算的完整gap报告与远端逐项相同。

| 区间 | 次数 reference /wait1 | start gap中值 cycles reference /wait1 | 减前条latency后的残差中值 cycles reference /wait1 |
|---|---:|---:|---:|
| `0x32c8→0x32d8` | 368 /368 | 17 /16 | 16 /15 |
| `0x520c→0x521c` | 368 /368 | 17 /16 | 16 /15 |
| `0x61b0→0x61c0` | 368 /368 | 17 /16 | 16 /15 |
| `0x426c→0x427c` | 368 /368 | 17 /17 | 16 /16 |

四处的前条decoded latency均为1 cycle。三处改动各把残差中值从16降到15，
未改动的第四处仍为16。mainloop wait-only正残差合计43857→41578 cycles；
MSB-only正残差129159→129118，`both`为4608→4608。
这些是采样wave累计区间，不能直接换算为dispatch可节省的百分比，也不能将全部变化
独占归因给三个wait；四SIMD属于独立captures，prologue和同步等待还会影响总span。

该结果说明放宽一个pending计数只带来很小的局部下降，尚未消除主要wait残差；
反向正常性能改善0.23443% /0.39866%也不足以支持大幅提速结论。此ISA改动保持实验状态，
未采用为生产FlyDSL代码，未把80.4268 μs单组值作为最终保留版性能。

capture目录为`att/reboot20261024_addr32_wait1_isa_att_v1/`，archive SHA256为
`8d273c4918c11bfbd2402b764e8e0bda221bc13730fbc18eb193867ee6f7a40c`，156项逐SHA复核，
不含`.out`。原ELF reference的目录为`att/reboot20261024_dsf5_pin1_rotated_addr_constraints_att_v1/`。
原后处理失败的`.capture_attempt1.json`和binding log完整保留；补传纯CPU脚本后用同一次
capture恢复统计，没有重跑GPU，没有修改原事件或删除失败记录。
本地复核为`logs/reboot20261024_addr32_wait1_isa_att_v1.local_gap_review_v1.json`。

### steady loop 两次展开的性能对照（2026-10-24）

后续用`dsf5_pin1_steady_pair_scoped_v1`试验减少runtime steady loop控制开销，
保留iteration函数体、tile/buffer顺序、fences和原wave布局；源改动为+12/-2行。
正向reference /candidate /reference中值为81.6539 /81.0573 /81.1205 μs，
反向candidate /reference /candidate为80.6860 /81.0506 /80.9516 μs。
两侧收益分别为0.73064% /0.07791%和0.44984% /0.12215%；原精度门槛及
实际ELF绑定均通过，seed0完整MoE与保留版逐字节相同。

该版CPU artifact的SGPR由62降至58，VGPR1024、LDS327680 B不变；
静态wait/MSB数量反而从118 /750增至160 /1110。
尚无这版的matched ATT，不能据此宣称消除了reference中
`s_add_nc_u64→s_cbranch_scc1`区间的8-cycle残差，也不能用静态数量
增加或减少预测动态空泡。当前保留为实验候选，未替换完整验证的reference。
全部十八轮原始数据、命令与archive校验见参数调优文档对应章节。

### 四元素地址约束的对照结果（2026-10-24）

把原compact pin1 guard内四个scalar地址约束合为一个tied vector约束，
物理地址仍为current `v32:35`、next `v36:39`。CPU artifact中的静态
`s_wait_alu` /`s_set_vgpr_msb`为121 /740，reference为118 /750；
WMMA /TDM /DS load仍为640 /24 /720。五处DS wait被编译器分拆，
其他TDM/barrier/store fence顺序相同。这只能说明静态调度发生变化，
不能从少10条MSB指令推断消除了动态空泡。

正向reference /candidate /reference中值为81.4809 /80.5257 /80.9245 μs，
反向candidate /reference /candidate为80.6314 /81.2851 /81.6773 μs。
反向前组改善0.80421%，后组回退0.48250%，没有稳定收益，暂不采用。
seed0完整MoE逐字节一致、全部原precision gate和ELF绑定通过；
未追加该版matched ATT或多seed/guard外验证。

当前完整验证的最佳reference仍为GEMM1 ELF `deea50e0…ceda0488`，
不能把候选的单组低值或静态control数量下降作为新的正式性能结论。
完整21轮原始记录及复現命令见参数调优文档对应章节；原始性能记录不含`.out`。

### i32地址表达式最佳候选的matched ATT（2026-10-24）

在完整验证的addr32 reference上，只把原compact pin1 guard内LDS地址表达式改为等价i32计算，
四个scalar tied constraints、input ring、fences与计算保持。候选源SHA256为
`f3c77d2aaa4698c80eb82b5d377c4f7b529247fc24c5fffce401d9a6f1fb21ec`，
实际GEMM1 ELF为`c5296627d336613a8a8580ec712c3d04766081e8b9022b0012333e13fb8001d2`。
资源为VGPR1024 /SGPR62 /LDS327680 B /private0 /spills0；static executable/wait/MSB为
5701/118/750，reference为5724/118/750。没有删除wait或手工修改ISA立即数。

正常性能正向i32中值80.8191 μs，反向两组80.6187/80.5554 μs，四项相邻对照改善
0.29891%～0.75058%。五项完整MoE strict和11项guard外完整ELF一致性检查通过，
GEMM2 pin1 ELF仍为`a5e9c250…575311`。70 μs尚未达到，生产源码尚未正式替换。

使用原`my_code/get_att_b_rocm_sdk.sh`采集all-SIMD、iteration28的原workload，
四SIMD均完成；64个active waves各1792 WMMA。actual keys、完整ELF与decoded PC/ISA
逐项绑定通过。profile期间的计时不用于正常性能结论。

按用户公式，以SDK `event[3]`为前条latency，保留负值并排除越过隐藏branch的区间：

| mainloop区间 | 次数 | reference正残差cycles | i32正残差cycles |
|---|---:|---:|---:|
| wait-only | 7040 | 43857 | 40781 |
| MSB-only | 76928 | 129159 | 129122 |
| both | 1536 | 4608 | 4608 |

| reference PC | i32 PC | 样本数 | start gap中值cycles | 前条latency中值cycles | residual中值cycles |
|---|---|---:|---:|---:|---:|
| `0x32c8→0x32d8` | `0x32a4→0x32b4` | 368 | 17→16 | 1→1 | 16→15 |
| `0x426c→0x427c` | `0x4224→0x4234` | 368 | 17→16 | 1→1 | 16→15 |
| `0x520c→0x521c` | `0x51a0→0x51b0` | 368 | 17→16 | 1→1 | 16→15 |
| `0x61b0→0x61c0` | `0x6120→0x6130` | 368 | 17→16 | 1→1 | 16→15 |

前条ISA为`v_dual_add_nc_u32 v38, s14, v1 :: v_dual_add_nc_u32 v39, s14, v4`，
后条为相应`ds_load_2addr_b32 ... v34 offset0:128 offset1:160`，wait0保留。
四处各下降1 cycle，说明地址表达式及调度的小改动确有局部改善，主要残差仍未消除。
MSB-only residual中值仍0、P95=4，36713次为负；不能把全部正值当成MSB独占stall。

| i32 phase | 累计cycles | 占比 | per-wave中值cycles |
|---|---:|---:|---:|
| prologue | 221985 | 13.07864% | 3461 |
| mainloop | 1157461 | 68.19389% | 17182 |
| final input drain | 4032 | 0.23755% | 63 |
| activation/quant/scale/LDS+barrier | 260024 | 15.31978% | 4063 |
| output descriptor | 3136 | 0.18476% | 49 |
| output TDM/completion | 50671 | 2.98537% | 905.5 |

reference mainloop per-wave中值17331.5 cycles，但i32 mainloop累计值反而更大，
单wave范围16861～25074。四SIMD与两版本均属于独立captures，存在调度及同步差异，
不能把累计phase或control差值直接换算dispatch可节省比例。正常性能以完整正反向对照为准。
本轮E65仍为分支原t64/w1x4 fallback，未追加i32版固定shared routing和性能验证，
不能宣称满足历史E64/E65共用同一GEMM1要求。

证据目录为`att/reboot20261024_pin_addr32_i32_att_v1/`，decoded archive SHA256为
`2d56ed44392389d6b1f21e6ec69bff21b62312a3e034dd7546d2ffe1e2b5f9c7`。
138项regular members逐SHA验证，不含`.out`；本地从原事件重新计算phases、control和
gap-minus-latency报告均与远端相同。仅规范化JSON字典键类型，没有修改事件或指标。
复核记录为`logs/reboot20261024_pin_addr32_i32_att_v1.local_trace_review_v1.json`，`passed=true`。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_pin_candidate_att_v1.py \
  dsf5_pin1_addr32_i32_scoped_v1 \
  reboot20261024_dsf5_pin1_addr32_i32_scoped_v1_pin0_prepare_v1 \
  reboot20261024_pin_addr32_i32_att_v1
```

### i32版本采用状态与后续地址缓存实验（2026-10-24）

matched ATT完成后补充E65/T1536/topk9的seed0 dynamic和seed17完整MoE验证，
各22,020,096 B与原版零字节差异，原precision gate通过。E65仍为原t64/w1x4/b3 fallback，
不是E64的t256 kernel，也未把random E65称为固定shared expert route验证。
本地及a07-3随后正式采用`f3c77d2a…f1fb21ec`；此前“尚未采用/未补测E65”的状态由本节更新。
源与ELF均与上节capture对应，故上节PC和原事件统计仍适用于实际保留的pin1 GEMM1。

另试四级stage LDS地址预计算。v1不能用runtime `ArithValue`索引Python列表；
v2误用`fx.select`；v3改为`arith.select`后编译和原有界DS/WMMA、TDM/fence路径语言对照通过。
v3资源仍1024/62/327680 B/private0/spills0，但static executable/wait/MSB为5775/129/750。
seed0完整MoE零字节差异；九轮正常对照reference前/candidate/reference后中值为
80.3974/82.5563/80.9201 μs，回退2.68529%/2.02199%，所以放弃。
未为回退版追加matched ATT，不将新增wait的静态数量当作其独占动态成本解释。
完整原表、命令和archive校验见参数调优文档。

采用后的新复测不会切source或注入cache：E64、E65各固定三轮，首轮空cache运行原入口，
随后核对其真实keys和两阶段完整ELF。原warmup5/iters20、数据、route、精度和计时函数保持；
每轮全机idle，test/compiler child限时120秒。完整复测结果如下。

### 正式F3源码原入口复测与原版pin组合澄清（2026-10-24）

正式采用源保持`f3c77d2a…f1fb21ec`，六轮原入口测试已完成。每轮全机idle后启动，
原precision、实际两阶段ELF和测后ownership/settled idle均通过，不切source、不注入artifact。

| 规模 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2三轮 μs | GEMM2中值 μs |
|---|---|---:|---|---:|
| E64/T1536/topk8 | 80.6925 /81.0172 /81.2006 | 81.0172 | 61.0965 /61.4658 /61.0145 | 61.0965 |
| E65/T1536/topk9 | 257.825 /256.345 /254.551 | 256.345 | 159.802 /160.971 /160.728 | 160.728 |

E64仍绑定上节matched ATT的完整ELF`c5296627…f8001d2`；E65绑定原t64/w1x4/b3 fallback，
没有改变该规模dispatch，不将其称为目标t256 kernel的性能收益或固定shared expert route验证。
E64与E65三轮logits_diff分别为`3.61055e-06`和`3.59489e-06`，原precision gate通过。
首个runner在E64首轮计时完成后因host无法读取容器600权限的pickle而发生`PermissionError`，
只修复本实验cache权限。原失败记录与首轮原表、manifest、idle检查保留，恢复后补齐key/ELF audit
并继续其余五轮，没有重跑或替换首轮；首轮host elapsed未保存，明确记为`null`。

118项正常复测归档已下载并逐SHA、原表中值及四个完整ELF复核，不含`.out`，
archive SHA256为`90dccf2e28e66371d978319527583c60d490a0c05da4854f95629057e7e80fc4`。
可复现命令、完整metrics及恢复细节见参数调优文档末尾。E64正式中值距70 μs还有11.0172 μs，
不将累计wave residual或phase占比换算成可直接节省的dispatch耗时。

另补充历史pin实验：已经测过`dsf5_pin1_feature_parity_v1`，只放开六处优化guard，
保留MMA2/first1、inline-zero、A-only timeout等DS5优化，同时沿用原pin编号
（C=`v512:1023`，两份A/B槽=`v256:383`/`v384:511`）。它没有调整LDS地址或加入后来的rotated布局。
其三轮82.9272/81.4724/81.4745 μs、中值81.4745 μs；随后同组合pin0重测
82.5247/82.0713/82.2843 μs、中值82.2843 μs，观测差0.8098 μs（0.98415%）。
两组原precision和ownership通过，但没有双侧交错对照，只支持初筛有收益，不能确认稳定提升。
原版pin1的86.1428 μs属于未接入上述优化的另一组合，不能据此否定原pin叠加优化，
也不能把86.14 μs到当前F3约81 μs的差值全部算作修改VGPR编号的收益。

### 原版pin组合的双向性能对照补充（2026-10-24）

已补完原版pin编号叠加DS5优化与相同优化pin0的正反向十八轮原入口对照。
正向pin0前/原版pin1组合/pin0后中值为81.6348/81.7720/81.9597 μs；
反向原版pin1组合前/pin0/原版pin1组合后为81.7677/81.8858/82.4404 μs。
四项相邻改善为-0.168066%/+0.229015%/+0.144225%/-0.677285%，不支持稳定收益，
所以保持正式F3/pin1最优版，不将历史约0.98%的单向初筛当成已确认收益。
十八轮原precision、source-dependent keys/完整ELF及全机ownership均通过；逐轮原表、
中值和比例在本地重新计算。GEMM2随全局pin开关使用各自的原编译版本，不能归因于GEMM1改动。

继续在F3上试恢复原C/A/B布局、LDS地址`v16:23`及既有四级ring循环展开。
三项CPU编译和seed0完整MoE逐字节对照均通过，GEMM2完整ELF保持。
资源分别为VGPR1024、SGPR62/62/70、LDS327680 B，private/spills0；
static executable/wait/MSB分别为5701/118/750、5710/126/750、10702/255/1830。
这些是静态模板计数，尚不能说明动态空泡或dispatch收益。该轮没有删除wait，也没有新增pipeline。
已有F3 matched ATT的四处15-cycle residual仍保留为诊断证据；新候选是否改善，以正常交错性能为准。

完整原表、命令和候选SHA见参数调优文档末尾。18轮pin对照及三项CPU/准备证据包
SHA256为`05bf8fae910664100ac28069588a3433b868ce39bee8773463ddf72ce58b9542`，
448项regular members逐SHA复核通过，不含`.out`。

### F3上的三项后续候选：正常性能对照不支持采用（2026-10-24）

恢复原C/A/B布局、LDS地址`v16:23`及四级ring展开各完成二侧九轮原入口对照。
下表每个中值来自固定三轮，全部attempt1；原precision、source/实际ELF和全机idle检查通过。

| 候选 | F3前中值 μs | 候选中值 μs | F3后中值 μs | 决策 |
|---|---:|---:|---:|---|
| original_layout_addr32_i32 | 80.1656 | 80.3678 | 80.7115 | 未确认稳定收益，不采用 |
| addr16_i32 | 80.5365 | 80.8589 | 80.7748 | 两侧回退，不采用 |
| ring_unroll_i32 | 80.5512 | 140.494 | 79.9228 | 明显回退，放弃 |

三个候选的完整MoE输出与F3逐字节一致，两阶段真实call keys绑定对应完整ELF，
GEMM2完整ELF保持。计时期间没有并发远端编译、SFTP或额外GPU检查；结束后精确恢复F3。
没有对明显回退的ring版追加ATT，不能将其静态代码膨胀、SGPR增长或wait数量当作已证实的
动态回退原因，也不能根据这些结果推导“所有循环展开必然慢”。正式版已有matched ATT及
四处15-cycle residual仍适用，正常正式中值继续引用81.0172 μs，不挑本轮较快control读数。

27轮原表与prepare driver记录已下载，本地逐SHA和中值/比例复核通过。新包496项，
SHA256=`2715440357c331000de77f2bae64df25a52af1df38b3592498c3dc2675a3c5f3`，
不含`.out`；完整三轮读数、复现命令及候选source SHA见参数调优文档末尾。

### 优化结束与最终源码（2026-10-24）

按用户指令结束优化，仅补完当时正在运行的reuseB反向对照。reuseB正向F3前/候选/F3后
中值为81.3524/80.6082/80.8020 μs；反向候选前/F3/候选后为
80.3846/80.4373/80.5848 μs。反向两侧改善为+0.065517% /-0.183373%，
没有稳定胜过F3，因此不采用，也不追加新的ATT。18轮原precision、全机idle与实际两阶段ELF
绑定均通过，候选完整MoE输出与F3零字节差异；原表和比例在本地重新计算。

iteration pair及量化值pin仅有CPU/静态检查，没有设备性能结论，停止后未执行、未采用。
scale_const100 ISA工具未上传、未执行，编码观察不构成已验证的bug或性能瓶颈。
因此本文已有matched ATT、PC、累计wave phase和四处15-cycle residual仍对应实际保留F3；
没有把残差或累计wave phase占比换算为可直接节省的dispatch时间。

最终英文提交为`6f656cf5b02b2f1acf723949652f91b48e2933d7`，已推送ROCm/aiter的
`hyg_dsr1_conc6144`，a07-3已快进拉取并通过源码SHA检查。
kernel源仍为`f3c77d2a…f1fb21ec`，已有pin1 matched ATT的完整ELF保持
`c5296627…f8001d2`。最终pin=0与pin=1复测使用该commit和原单测入口，不注入ISA、
不切source，详细原始三轮和复现命令记录在参数调优文档中。

### 最终commit拉取后的正常性能结果（2026-10-24）

a07-3的`dsr1_ep4`已在commit `6f656cf5b02b2f1acf723949652f91b48e2933d7`下完成
pin0/pin1各固定三轮交错复测，原单测、数据、routing、precision和计时函数保持。
测试前全机idle，测试后无GPU进程残留；真实call keys及完整两阶段ELF逐轮复核通过。

| 配置 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| pin=0 | 82.3413 /81.7383 /81.4917 | 81.7383 | 64.3523 |
| pin=1 | 81.0487 /80.7177 /80.7452 | 80.7452 | 61.4405 |

六轮均`pass=True`、`logits_diff=3.61055e-06`、`rel_l2=0.00268721`。
pin1的本次GEMM1中值快1.214975%，最终成绩为80.7452 μs，仍未达到70 μs目标。
GEMM2差异来自全局pin配置下各自原编译版，不归因于GEMM1源码优化。
最终pin1 GEMM1 ELF仍为`c5296627…f8001d2`，与已有F3 matched ATT相同；
不重抓trace，也不将既有累计wave phase或bridge residual换算为可直接节省的dispatch时间。
之前81.0172 μs记录保持其当时的测量值，新成绩来自此次用户指定的最终提交复测。

优化工作已经结束。最终六轮原表、中值、复现命令、环境、源码/ELF校验和停止前最后一项
reuseB的双向对照决策均见参数调优文档末尾。最终118项证据包已下载并逐SHA验证，
SHA256=`a4a3bdd37e37df69b63759f3b804fd41d9d44a10e8662d00c4f36eb4839fb41c`，不含`.out`。
