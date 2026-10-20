# GEMM1：`--scenario kernel` 与 `--scenario bench` 的计时差异

## 当前结论（2026-10-19）

两种scenario实际使用**同一个v190 GEMM1完整ELF、`.text`、JIT cache key和launch参数**，没有切换tile、算法、TH6或关闭fused quant。它们测量的执行上下文不同：`kernel`连续调用固定tensor的GEMM1；`bench`每轮执行完整MoE，再提取其中GEMM1的device duration。

正常原命令中，`kernel`为85.9759、87.0495 μs，`bench`为92.2233 μs。重启后新增诊断为 **86.037263 /90.783632 μs**，差4.746368 μs；初次诊断86.760579 /91.074316 μs另批保留。诊断不替换正式三轮成绩。差异已排除换kernel和计时范围混入其他kernel；同地址前序对照也能改变GEMM1耗时，但cache、时钟、前序quant/GEMM2的分别贡献没有counter支持，不能细分。

用户已明确通知a07-3重启，远端操作恢复。**scenario差异诊断已完成**：首次正式GEMM1的有效A/ScaleA按token/topk顺序还原后hash相同，物理行顺序不同；固定输入前序对照通过。原`bench`第一轮646.850 μs仍因明确外部GPU stress并发排除。取回hang前context日志后确认，当时仅在等待GPU空闲，没有进入测试，不能将机器hang归因于该context实验。

## 原命令实测

均为E64/T1536/topk8/K7168/inter2048、A4W4、SiLU、无bias、random non-balanced、warmup5、iters20、testGraph=False。

| 样本 | GEMM1 μs | GEMM2 μs | MoE device sum μs | 使用范围 |
|---|---:|---:|---:|---|
| kernel original r1 | 85.9759 | 59.5533 | 不报告 | 正常原命令样本 |
| kernel original r2 | 87.0495 | 59.0917 | 不报告 | 正常原命令样本 |
| bench original r2 | 92.2233 | 58.3094 | 203.241 | 正常原命令样本 |
| bench original r1 | 646.850 | 460.216 | 1647.48 | 外部并发污染，排除性能，只保留精度 |

正常样本pre均全机idle，立即post无KFD所有者、VRAM176582656 B、busy2–3，6秒及post6恢复idle。性能期间本任务CPU会话闲置，没有并发编译、正确性或idle检查。前后采样不等于连续占用监控，不能绝对排除一项外部任务在两个采样之间完整开始并结束。

异常r1的pre在`2026-10-19T01:56:23+00:00`idle；结束/post在`01:56:44`看到外部PID1532804–1532809均已运行约20秒、GPU busy100、VRAM14793789440 B：

```text
python3 /tmp/ps1q/stress_multi.py /tmp/ps1q/stress_multi.flag 3000 4
```

据运行时长推算stress约在`01:56:24`开始，明确与测试重叠。controller没有杀这些进程，按6秒等待至idle；最终idle不能使此前的污染成绩有效。

## 两种scenario实际计时范围

[`_run_grouped_via_fused_moe`](test_flydsl_grouped_gemm_gfx1250.py)的`kernel_bench`分支（当前1856行起）先执行一次完整`_call()`，通过`grouped.kernel_bench_callable`捕获`functools.partial`，随后依次独立测quant_a1、GEMM1、GEMM2。GEMM1输入、weights、ScaleA、psum和输出地址固定，每轮不重跑routing、GEMM2和combine。

`bench`分支（1879行起）重复完整`_call()`。hidden、weights、topk_ids和topk_weights在闭包中复用，但每轮pipeline重建、写入中间tensor，并运行routing、quant/scatter、GEMM1、GEMM2和gather/combine。GEMM1耗时来自完整profile的`device_time_avg`（2168–2187行）。

两者均使用[`aiter/test_common.py`](../aiter/test_common.py)的`run_perftest`：memory profiling、warmup、`AITER_LOG_MORE=1`额外cuda.Event循环，最后才是实际profiler loop。

- 日志的`avg: ... from cuda.Event`来自额外逐次同步循环，不是最终返回的GEMM1数值。不要把它与profiler的GEMM1行比较。
- `get_trace_perf`丢第一轮device事件，20轮剩19轮；此参数不触发`num_iters>30`的IQR剔除。两种模式都从device事件统计，GEMM1行没有直接加上Python launch gap。
- 两个callback均没有显式tensor位置参数，`device_memory_profiling`的`inputSize=1`；`rotate_args` deepcopy的是空args/kwargs，**没有复制或轮换闭包/partial中的weights**。
- 原benchmark标签“MoE e2e”的返回值是每轮device kernel duration之和，不包含全部CPU gap的墙钟时间。这里保留原计时公式，不将它改成wall time。

## 实际编译对象与时间轴

两个诊断进程经原`fixture.main()`执行，production源文件未修改、原averaging/filtering保留。第一次输入hash在warmup前读取，hook在重复计时前移除；kernel捕获partial在计时前恢复原launch函数。事件/Chrome trace在计时后导出。诊断改变了warmup前的状态，计时单列，不计入目标成绩。

| 项目 | 两种模式核对结果 |
|---|---|
| 完整ELF SHA256 | `24bb623573932faadb22ca2d89d3e84a903adc64c9e116cac4138dd5b686f93c`，相同 |
| `.text` SHA256 | `7b0b0ca23994eba0fd308c388c5925b356051b553ca0e31e7e738a9c4a6805ad`，相同 |
| JIT key /entries | 相同key，各1 entry |
| geometry | t256x256x256/w2x4/b4、cn4/cm1、TH6、prefetch on |
| activation /quant /layout | act1/q1r8、A-preshuffle，相同 |
| runtime E /capacity M /N | 64/28672/4096，相同 |
| parent GEMM1 fallback | 均未使用 |

实际完整symbol：

```text
a8w4_tdm_fp4_t256x256x256_w2x4_b4_K7168_edyn_act1_q1r8_cn4_cm1_prefetch_apre_persist_optx_g1_s4_sk1_st0_sn0_bcache_merge_wavewindow_wavepub190_edyn_earlybanks_splitS0_S1rows0_carryF_carryS
```

`kernel`的GEMM1 profile有20个原始device事件，全部是该GEMM1，没有quant、reference、GEMM2或clone夹带。完整`bench`有200个device事件，每轮10次launch，其中GEMM1/GEMM2各一次；其他阶段单独计时，没有加到GEMM1行。

| 诊断项目 | kernel | bench |
|---|---:|---:|
| GEMM1 19轮平均 μs | 86.760579 | 91.074316 |
| GEMM1 19轮范围 μs | 83.644–92.097 | 89.253–94.020 |
| GEMM1前device空隙中值 μs | 57.727，前一个GEMM1结束后 | 128.454，前一个scatter结束后 |

两种时间轴均有CPU提交产生的device空隙，kernel模式也不是无gap的graph循环。gap不直接计入GEMM1duration，但可能改变运行状态；没有时钟采样，不能将4.314 μs定量归因于频率。

## 输入、padding与精度

seed0 route_ids、route_weights、64项route counts、psum、W1及W1 scales的byte hash均相同，12288条route保留。第一次实际launch与kernel捕获partial的全部tensor地址、shape、dtype、stride及默认参数也相同。两个诊断的MoE均finite、`logits_diff=3.473346e-06`、`rel_l2=0.002635657`，满足原gate<0.01。

A/ScaleA整块storage hash包含未写入的capacity padding。重启后先按`psum[e]-count[e] .. psum[e]`解码，发现有效物理行的A/ScaleA hash也不同；这不能解释为只有padding不同。进一步在首次正式quant调用捕获`topids_to_rows`，确认每条route唯一占据相应expert的有效范围、12288条route完整覆盖，再把解码结果按token/topk顺序重排，A/ScaleA hash均相同。源码`moe_route_maps.py`使用per-expert `AtomicRMWOp(add)`分配slot，expert内行序由执行顺序决定，两个独立进程无需得到相同物理排列。

| 首次正式GEMM1输入项目 | 两种scenario结果 |
|---|---|
| hidden_states SHA256 | 相同，`f27f77f307dcc790f9c1a9190076181a090e527d50a84032ba517c768c89c9de` |
| route ids /weights /counts /psum | 相同，12288条route完整保留 |
| 有效物理行集合 | 相同，SHA256=`ef191fdaf01ed6824554686b1ad46f3ace8a881f74654cd86f091bc6b414dad8` |
| `topids_to_rows` /物理行A与ScaleA hash | 不同，expert内行序不同 |
| token/topk顺序A payload，44040192 B | 相同，SHA256=`e0fc76ae1443de8061ace63d6aa0fda764bd3e452b2be37af01105858e7bb2b7` |
| token/topk顺序ScaleA，2752512 B | 相同，SHA256=`533257615940eb87b1ecf8969f48df3849dbcb88fb021976364c4406db8ced48` |

hook在重复计时前撤掉，CPU receipt不保留GPU tensor以延长中间buffer生命周期。上述是首次正式输入核对；bench随后每次routing的物理行序可以再次变化，未将每轮输入复制到CPU，也未声称物理排列全程一致。额外hash/readback会改变warmup前状态，因此该批性能继续单列为诊断。

kernel打印的e2e精度来自最初完整`_call()`，没有把末次隔离GEMM1/GEMM2输出重新compose为完整MoE。计时后的独立BF16 stage capture会走诊断路径，不能当作计时中fused GEMM1的直接验证。正式v190另有E64/E65 seed0/17/43八hash、动态输入、poison和graph replay验收，不依赖隔离计时的这项限度。

## 性能原因的证据边界

已证实的差异为执行前序、中间tensor复用方式和提交节奏。kernel连续使用同一A/ScaleA/W1/psum/output；bench在两次GEMM1之间重写A/ScaleA并执行GEMM2、routing、quant和combine。逻辑有效输入保持，物理route行序并非固定。这可能改变局部cache复用、memory请求状态和GPU时钟；没有counter与随机交替控制实验，不能细分成本，也不能把route行序变化认定为主要性能原因。

本地MI455X白皮书第10页记载L2为192 MB，每WGP SRAM为384 KB、最多320 KB LDS。实际W1 payload为939524096 B（896 MiB），上层W1 scales tensor为58720256 B（56 MiB，descriptor按有效scale范围寻址）。不能说全部W1驻留L2，也不能用逻辑字节/耗时反推已经测到HBM/L2带宽。

固定地址前序对照已完成，顺序为G1-only、G2→G1、quantA1→G1、G2→quantA1→G1、G1-only。保持原warmup5/iters20/count19/averaging，每项只比较GEMM1自己的device行；每项前后有效A/ScaleA/W1/weight scales/psum hash一致，实际完整ELF仍为v190。

| 同地址前序，执行顺序固定 | GEMM1 device avg μs |
|---|---:|
| G1-only first | 88.384368 |
| G2→G1 | 88.556895 |
| quantA1→G1 | 88.019263 |
| G2→quantA1→G1 | 90.374105 |
| G1-only last | 85.978211 |

首尾G1-only相差2.406158 μs，显示该顺序实验还存在运行状态漂移。它证明相同字节、地址和代码的GEMM1也能测出不同duration；**不能把90.374105−85.978211全部解释成G2/quant的固定惩罚**。这些前序也没有完全复现完整MoE的allocator/route/combine生命周期。

首版重启后context helper误把fixture随后独立BF16 stage capture也收进`kernel_bench_callable`，出现`quant_a2`并覆盖同名GEMM1/GEMM2，在任何context计时前exit1。修正为首次正式GEMM2 callable追加完成后立刻撤掉capture，确认GEMM1 `stage1_quant_out=1`；生产kernel、selector、原benchmark与计时工具未修改。修正版exit0及完整ELF检查通过。该helper问题不是生产GEMM1的数值bug。

新增route-order诊断的20条GEMM1事件，kernel其余19条duration为83.124–92.818 μs，bench为89.493–97.425 μs；前device gap中值分别56.246 /124.307 μs。bench每轮10个device launch，kernel的GEMM1 profile只有GEMM1。gap未计入GEMM1行，不能据gap量化cache或频率。

三项新增诊断均先等全机idle；立即post无KFD、VRAM176582656 B、busy2，六秒后检查看到其他`dev_al`任务已运行约2秒，即根据该样本约在本测试结束4秒后开始。原post与等待过程保留，不能把最终idle覆盖为首个post始终idle。最终post6空闲，无本任务残留；这些证据支持诊断，不升级成新的严格独占正式成绩。

本任务仍以原完整`--scenario bench`的三轮结果判断80 μs目标；不能将较快的隔离数字作为达标成绩。

## 复现与本地证据

用户原命令仅改scenario：

```bash
ENABLE_CK=0 AITER_MOE_EXPERT_BALANCE=false AITER_LOG_MORE=1 \
AITER_USE_GROUPED_GEMM=1 AITER_GROUPED_DEBUG=0 \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
python3 -u my_code/test_flydsl_grouped_gemm_gfx1250.py \
  --scenario kernel --data-format a4w4 --act silu --no-bias --no-check-aot-cache \
  --experts 64 --tokens 1536 --topk 8 --model-dim 7168 --inter-dim 2048 --iters 20
```

完整pipeline为`--scenario bench`。先检查boot、容器、source SHA及全机idle，每项测试120秒上限，用全新label：

```bash
task=.codex_tmp/gemm1_t1536_20261017
bash "$task/run_scenario_compare_host.sh" kernel scenario_kernel_repro01
bash "$task/run_scenario_compare_host.sh" bench scenario_bench_repro01
bash "$task/run_scenario_diagnostic_host.sh" kernel scenario_kernel_input_repro01
bash "$task/run_scenario_diagnostic_host.sh" bench scenario_bench_input_repro01
bash "$task/run_context_diagnostic_host.sh" scenario_context_repro01
```

本地历史冻结22项1092803 B，manifest为`logs/scenario_timing_local_evidence_20261019.sha256`，汇总为[scenario_timing_local_summary_20261019.json](../.codex_tmp/gemm1_t1536_20261017/logs/scenario_timing_local_summary_20261019.json)，保留当时状态。重启后新增原始证据为[固定地址前序receipt](../.codex_tmp/gemm1_t1536_20261017/logs/scenario_fixed_context_resume_20261019_r2.json)、[kernel route-order receipt](../.codex_tmp/gemm1_t1536_20261017/logs/scenario_kernel_route_order_resume_20261019_r1.json)、[bench route-order receipt](../.codex_tmp/gemm1_t1536_20261017/logs/scenario_bench_route_order_resume_20261019_r1.json)，完整log/post6同目录保存。boot、Torch/FlyDSL/生产SHA见[恢复环境receipt](../.codex_tmp/gemm1_t1536_20261017/logs/resume_scenario_environment_20261019_r1.json)。旧context日志已取回并确认没有启动GPU测试，失败首版日志亦保留。

正式目标继续以完整bench判定，v190 E64/E65三轮中值仍91.688/137.596 μs，E64低于80 μs尚未达成。
