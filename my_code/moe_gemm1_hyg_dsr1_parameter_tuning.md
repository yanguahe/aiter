# hyg_dsr1_conc6144 GEMM1参数遍历与小改动优化

最新性能目标（2026-10-23 用户修订）：在原测试口径和空闲 GPU 条件下，争取将目标规模
GEMM1 稳定优化到70 μs以下。原功能、shape支持、精度及小改动约束继续保持。
本文件历史章节中的80 μs为当时目标，不改写历史测量或归档。

当前保留版（2026-10-24）：已正式采用目标pin1路径的C/A/B寄存器布局、current/next LDS地址
tied constraints及等价i32地址表达式；本地与a07-3源SHA256为
`f3c77d2aaa4698c80eb82b5d377c4f7b529247fc24c5fffce401d9a6f1fb21ec`。
GEMM1正反向候选组中值80.8191、80.6187、80.5554 μs，四项相邻addr32 control改善
0.29891%～0.75058%，没有采用最低单组作为统一成绩。正式源码的原入口三轮复测中值为
81.0172 μs，首轮从空cache运行；完整六轮数据与权限恢复说明见末尾。
复现最优路径要开启A preshuffle和`AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=1`。
原pin0路径及其完整GEMM ELF保持；末尾的采用记录、适用范围和新复测为最新状态。

上一pin0保留版（2026-10-23）：DS_B计数修正 + XDL queue bit + M-group32 + 首K128 inline-zero +
A-only standard timeout + closing WMMA hint25（含末尾K256前半段）+ MMA2/first1 + future DS floor + bitsearch + 首DS cap5，
GEMM1原入口正向中值**81.6350 μs**，反向确认组中值81.8183 /81.7086 μs，距当前70 μs目标尚差11.6350 μs。
DS-first5四项相邻对照改善0.4045550% /0.0078392% /0.2366715% /0.3704318%，
追加精度/scope验证通过后采用；最小改善仅0.0064 μs，不能声称显著或稳健提升。
DS-first4的81.5232 μs、反向81.5084 /81.7594 μs及0.1188%～0.9301%改善保留为历史记录。
上一C25正向81.2939 μs、反向81.7466 /81.7276 μs及0.1727%～0.6465%改善保留为历史记录，
不跨时段用绝对耗时比较两个版本的收益。
此前bitsearch阶段81.4810 /81.1352 /81.3419 μs及其0.0801%～0.6458%改善保留为历史记录；
上一MMA2/first1阶段正向80.7494 μs、反向81.4239 /81.0467 μs保留为历史记录，不跨时段直接比较收益。
MMA2/first1相对相邻MMA2/first2 control四项改善0.6096%～1.1926%；
这是通过正反向完整组和追加验证后保留的版本，反向结果也显示绝对耗时仍有波动。
上一MMA2/first2阶段中值81.3545 μs、反向81.2954 /81.3757 μs保留为历史记录。
上一DS floor阶段正式中值81.2958 μs，反向确认组中值81.1924 /81.5634 μs。
DS floor的四个相邻对照改善0.2715%～0.7880%；81.3419 μs为此前末尾closing覆盖阶段成绩。
末尾closing覆盖相对MMA3/first2的四个相邻对照改善0.2424%～0.6898%；81.4412 μs为此前阶段成绩。
MMA3/first2相对MMA3/3的四个相邻对照改善0.2052%～0.8034%；82.0840 μs为此前MMA3/3阶段成绩。
上一pin0保留源码SHA=`e53ada2baecb0c8e0185b815052281952213f0c81c5e4413c621baccfb06f93e`，
已分别保存到本地及a07-3的采用前快照；新source中的pin0完整ELF与该版相同。
E64/E65各seed0/17完整MoE精度与candidate自身8项outside-scope ELF核验通过；E65保留分支原t64/w1x4 fallback。
本文前面的84.6499/85.8308/87.1969/88.5421 μs及更早成绩保留为对应阶段记录；M-group8淘汰。
M-group64/128、A-only TH2/4/6、B-only LDS compact、ScaleA/ScaleB TH2及activation sub8/sub4
均未取得稳定收益，已放弃。84.9116 μs阶段的匹配thread trace已完成；
MMA3/3、MMA2/first2及上一DS-first4匹配trace已完成，逐decoded PC/ISA与实际ELF核对后本地复算通过。
上一DS-first4的累计wave gaps为mainloop68.2875%、prologue13.9271%、activation/quant/LDS14.4640%；
这些统计含arbitration/coexecution，不是GPU关键路径分解。继续小范围调度实验，不复用旧ELF的PC或trace比例。
当前DS5/pin0 的匹配 trace 已在2026-10-24重新采集并完成逐PC/ISA与实际ELF的本地核验；
pin1 i32地址候选已在2026-10-24正式采用；此前性能runner的临时切换记录保持历史原样，
采用后的新复测不切换生产source、不注入cache artifact。
用户报告的原命令冷启动卡住已修复，修复位于FlyDSL source location查询，两个GEMM ELF保持。

## 新基准与约束（2026-10-21）

本次基于本地`hyg_dsr1_conc6144`新提交
`346438a75506f77b520d3a9a09ba4985f054e377`，不是此前`ada68ed342...`基准。
该提交仅改两行：A4W4 prefill MMA默认8/8，B-only TDM默认TH1。
本地tracked工作区开始时无未提交变更；远端`aiter_a4w4_prefill_v2_yadai`仍在
`dev/yadai_dsr1_conc6144` / `ada68ed342...`，但两个compute sources SHA与新提交一致：

- wrapper：`005ccca824a243ead8de62128228ba6a5e6c4e70ec824512b3d501d3f9914d84`
- kernel：`259f9ff2b24dbd034f360fb7a5c491129406791fe651bda90fe9ae8e6fd3b942`

未执行Git清理、切分支、pull、commit或push。测试在a07-3的`dsr1_ep4`容器内，
FlyDSL=`0.3.3.dev882`，boot_id=`1d5146ac-1863-43d0-8211-f0761013cc4a`。
开启`AITER_GROUPED_A4W4_A_PRESHUFFLE=1`、关闭`AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=0`。
不加persistent、w2x4、OPTX pipeline，不改功能、shape支持、量化语义或精度标准；
本次参数实验仅改变GEMM1，GEMM2及其他kernel保持新提交的原配置。

## 复现口径

```bash
cd /data/yanguahe/code/wk_sp1/aiter_a4w4_prefill_v2_yadai
unset AITER_FLYDSL_MMA_GROUP AITER_FLYDSL_MMA_FIRST_GROUP AITER_TDM_B_TH AITER_FLYDSL_DS_FIRST_N
export AITER_GROUPED_A4W4_A_PRESHUFFLE=1
export AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=1
PYTHONPATH=$PWD AITER_USE_GROUPED_GEMM=1 AITER_GROUPED_DEBUG=0 ENABLE_CK=0 \
FLYDSL_DUMP_IR=0 AITER_LOG_MORE=1 AITER_FORCE_GFX1250=1 GPU_ARCHS=gfx1250 CU_NUM=256 \
AITER_MOE_EXPERT_BALANCE=false AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
timeout --signal=TERM --kill-after=5s 120s python3 -u \
op_tests/flydsl_tests/test_flydsl_grouped_gemm.py \
  --scenario kernel --data-format a4w4 --experts 64 --tokens 1536 --topk 8 \
  --model-dim 7168 --inter-dim 2048 --act silu --no-bias --no-check-aot-cache \
  --warmup 5 --iters 20 --seed 0 --data-init uniform --scale-init auto
```

原`run_perftest(testGraph=False)`和precision summary保持。主机每轮必须全机idle后启动；
占用时每6秒等待。计时期间不并发GPU监控、编译或传输；结束即时核查owner/VRAM，
再等6秒核查settled idle。外部owner或无法解释的占用使整组排除。
所有child限制120秒，超时、fault、精度或scope失败即停当前队列，不采用耗时。

## 参数实验范围与工具

`.codex_tmp/yadai_gemm1_param_20261021/candidates/parameters_v1.py`由新提交快照生成，
增加13行实验参数和GEMM1守卫；launcher ABI与其他顶层函数AST不变。SHA为
`0a6331b67a0e26d9ea88dcf18b4caee38fd36a78fb03e6ca8a62af184dcf245d`。
三个`YADAI_*`参数是私有实验输入，没有新增生产env开关；显式公共MMA/TH env清空，
避免全局影响GEMM2。候选只由stage1 launcher选择，非目标geometry仍使用新默认。

离线CPU-only实际wrapper编译确认默认8/8/TH1的完整ELF与新提交相同：
GEMM1=`be083c8c6beeccb4d9164a37f67c9a60ae9fd4ffcbfb668f06318561443ad052`，
GEMM2=`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。
每个原benchmark运行后重新保存实际ELF并检查GEMM2与对应基准一致。

首轮分别扫描`mma_group=1..32`、`mma_first_group=1..32`、`tdm_b_th=0..6`，
其余参数维持8/8/1。当前target每K128有32个accumulators，group和first被clamp到
该次有效MMA长度，因此超过32的值不产生更大有效组。TH7为保留值，不测试。
硬件TH解释依据Shader Guide §4.2.1 Table19 p62；只定义cache策略，不预言性能收益。
每4个初筛点插入默认控制，按距8的距离固定顺序遍历；单轮初筛不当作正式性能结论。
表现较好的参数随后交叉检查、固定三轮组对照；若无稳定收益，才继续独立小feature。

## 新提交基准验收

最初`newbase_perf_v1`的原benchmark精度通过，但计时后工具误选`_last_compiled`中的
GEMM1 ELF检查GEMM2，scope assertion rc1，队列停止。仅修复工具，按K与activation
查找实际compiled object；原失败记录不覆盖，耗时不作为正式结果。

`newbase_perf_v2_try1/try2`各完整三轮精度均通过，第三轮post有未识别已退出KFD entry，
整组排除、中值null，未从其余单轮拼中值。try3通过全机空闲及原精度验收：
GEMM1中值**89.4465 μs**，GEMM2中值**64.5231 μs**。这是本次正式基准，
不与此前不同source/date批次的92.0315或90.3955 μs混算附加收益。

## 单轴初筛进度与占用处理

`axis_sweep_v1`在`mma_group=18`的三个尝试后停止，停止标记与失败记录保留。
三轮原benchmark均精度通过，但post/settled检查有其他容器`zan_att`的测试，
以及无法识别命令的已退出KFD entry。其计时全部排除，不推断每段计时一定并发。

后续`axis_sweep_v2`只继续尚无有效样本的参数，保留前轮已完成结果，不重扫。
每次ownership不合格仍排除，参数保持pending；下次尝试先执行全机idle gate，
每6秒等待。仅ownership等待不再因为三次结束任务；超时、精度、scope或child失败
仍立即停止。规则变更与新plan保存在独立文件中，不覆盖v1的事前上限。

单轴初筛已完成：`mma_group=1..32`、`mma_first_group=1..32`和TH0..6，
共68个非默认点，穿插19个默认控制，共87个有效单轮，另11个ownership不合格轮。
所有有效轮原reference通过，GEMM2 ELF均为新基准。默认控制范围为
89.0945～92.5989 μs，所以局部初筛必须同时查看相邻控制。

| point | GEMM1单轮 μs | 前/后默认控制 μs | 初筛判断 |
|---|---:|---|---|
| mg9/first8/TH1 | 89.9824 | 90.1658 /89.5017 | 未胜过两个控制 |
| mg14/first8/TH1 | 89.9338 | 92.5989 /89.1917 | 前控制较慢；不能认定收益 |
| mg8/first9/TH1 | 89.5794 | 89.7607 /89.5394 | 接近波动 |
| mg8/first10/TH1 | 89.5774 | 89.7607 /89.5394 | 接近波动 |
| mg8/first11/TH1 | 89.5162 | 89.5394 /89.4065 | 未胜过两个控制 |
| mg8/first8/TH4 | 89.7394 | 90.1933 /89.8769 | 单轮约0.15%～0.50%改善，待确认 |
| mg8/first8/TH6 | 89.6870 | 90.1933 /89.8769 | 单轮约0.21%～0.56%改善，待确认 |

`cross_v1.plan.json`已固定35个交叉点：group8/9/14 × first8/9/10/11 × TH1/4/6，
默认组合由穿插控制覆盖。没有声称穷举32×32×7的全部Cartesian space。
这轮完成后再对最有希望的参数做固定完整三轮对照；每个单轮仅作初筛。

## 后续小改动候选的静态准备

交叉初筛已完成35个非默认点、10个默认控制，共45个有效单轮，另12个ownership排除轮。
下列较快的单轮只用于选择正式对照，不直接作为收益：

| group/first/TH | GEMM1单轮 μs | 相邻默认控制 μs |
|---|---:|---|
| 9/8/4 | 89.2231 | 90.1466 /90.2268 |
| 8/8/6 | 89.2717 | 90.1466 /90.2268 |
| 9/9/1 | 89.0316 | 89.9083 /89.5966 |
| 8/11/4 | 89.0905 | 89.4342 /89.4003 |

`confirm_v1`固定完整三轮对照已结束，所有合格轮保持原reference通过和GEMM2 ELF一致：

| group | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| control0 8/8/1 | 89.3645 /88.9658 /90.2520 | 89.3645 | 64.5419 |
| 9/8/4 | 89.9127 /89.1199 /89.7736 | 89.7736 | 64.1943 |
| control1 8/8/1 | 89.4445 /89.6704 /89.4783 | 89.4783 | 64.6328 |
| 8/8/6 | 89.2232 /89.5942 /89.5269 | 89.5269 | 64.6538 |
| control2 8/8/1 | 89.5184 /89.1959 /89.5268 | 89.5184 | 64.8162 |
| 9/9/1 | 89.5014 /90.2583 /89.3813 | 89.5014 | 63.6945 |
| control3 8/8/1 | 89.7734 /90.0604 /89.4383 | 89.7734 | 64.2447 |
| 8/11/4 | 90.0077 /90.2879 /89.5059 | 90.0077 | 64.4724 |
| control4 8/8/1 try4 | 89.8218 /89.3181 /90.2057 | 89.8218 | 64.8900 |

control4的try1～try3在post/settled出现无法核对命令的已退出KFD entry，整组保留但中值null；
try4在新一次全机idle gate之后合格。未从这些排除组选择低读数或拼接有效轮。
9/8/4、8/8/6、8/11/4正式组均慢于各自两侧默认；9/9/1比两侧默认低
0.0170 /0.2720 μs，处于轮间波动范围，不能据此认定稳定改善。

随后固定`confirm_reverse_v1`为9/9/1→8/8/1→9/9/1，每组完整三轮，
继续保留ownership排除和120秒硬限制。该反向确认完成之前不修改生产默认。

反向确认已完成：前候选89.6258 μs、中央默认89.5417 μs、后候选89.4952 μs，
GEMM2中值分别为64.2847 /64.3226 /64.5841 μs。九轮原reference和ownership均合格。
前候选慢0.0841 μs、后候选快0.0465 μs，方向不一致；9/9/1没有稳定收益，不采用。
单轴全有效取值、优先交叉点和正反向确认完成后，保留用户提交的8/8/TH1默认。
该结论仅覆盖已经记录的扫描与对照，未声称全部Cartesian组合均无改善。

随后进入两个私有小改动的CPU-only编译检查；未修改生产tracked源码、原benchmark或计时。
除上述5行DS计数候选，`bmajor_mma_v1.py`增加11行、删除4行，在相同GEMM1守卫内
仅将独立accumulators的发射次序从M-major蛇形改为B-major。全部有效M/N坐标、
每个accumulator的56次K128更新顺序、load helpers、`k_step`、`compute_ktile`和ABI保持；
768项M/N排列静态检查通过。源码SHA为
`b149bc4dc384886011a26709ecc9bddb0c6efaad92e9bb2eb9f71bb916901091`。
CPU排列检查不能替代GPU精度或性能，候选没有预设收益。

FP4的`load_b`调用两次`load_half`，每次两条`lds_load_b128`，即每个B fragment
需要4条B128 LDS read；目前`DS_B=2`使目标`STATE_DS`按32条统计，实际为40条。
这只说明scheduling hint统计与源码load数不一致，不是已证实的精度/race bug，
也不能由此推定性能收益。

私有`ds_hint_count_v1.py`仅增加5行，在原GEMM1/FP4/A preshuffle/pin0/
非persistent/t256/w2x2/b4条件下把`DS_B`设为4。全部load、`mma_rows`、
`k_step`、launcher ABI和其他顶层函数AST不变；实际计算、LDS allocation和
硬件同步保持。源码SHA=`3fb01bdf72d33a022a44d06876e213544125635c33ab3a63c0760316d42922b0`。
目前只在本地准备，没有GPU验证或性能；仅在参数遍历和确认无收益后才进入测试。

原日志、plan、source和ELF保存在上述本轮task目录，生产tracked源码仍为用户新提交。

### 小候选检查工具的环境修正

`small_cpu_v1_baseline`在容器中compile-only成功，完整ELF仍与基准相同；随后主机调用
`llvm-readelf`失败，缺少`librocm_sysdeps_zstd.so.1`，队列立即停止，两个候选尚未编译或运行。
这是检查器运行位置错误，不是kernel失败，也没有该队列的性能数据。
`dsr1_ep4`内的LD_LIBRARY_PATH已经包含对应ROCm SDK库且`llvm-readelf --version`正常。
新的`run_small_cpu_resume_v2.sh`复用成功的baseline ELF，不重复编译它；ISA检查进入容器，
两候选按原顺序继续compile-only，每个子进程仍限制120秒。旧失败日志完整保留。

`ds_hint_count_v1`的branch-local constexpr赋值在120秒内没有完成CPU-only编译，
timeout rc124且队列立即停止，没有GPU dispatch或性能；退出后未发现该编译子进程残留。
没有通过延长时间绕过限制。后备`ds_hint_count_v2`使用本文件已有的constexpr ternary形式，
新增4行、删除1行，全部nested helper AST保持，源码SHA为
`c8a7eaf9e880e751a982b796f5481527e1287887878f8877045df0a611a01b82`。
它在120秒内编译成功；编译超时的原因尚未定位，不能仅凭写法替换断言FlyDSL bug。

| candidate | VGPR | SGPR | LDS B | spills/private | static instructions |
|---|---:|---:|---:|---|---:|
| new baseline | 741 | 58 | 327680 | 0 /0 | 6003 |
| DS hint count v2 | 762 | 58 | 327680 | 0 /0 | 5971 |
| B-major MMA v1 | 749 | 58 | 327680 | 0 /0 | 6035 |

三个ELF中的WMMA、DS、TDM、输出及barrier静态数量一致；本表不是性能预测。
原`audit_small_resources.py`把任何VGPR增加列为停止检查的警告，原警告保留。
人工复核只发现VGPR增加，其余资源、操作及barriers均保持，因此独立
`small_resources_review_v1.json`允许先进行无计时的严格精度验证，未批准生产采用。
用户约束没有要求VGPR数必须不增加，不能仅凭静态VGPR变化舍弃可能有效的优化。

DS v2的E64/seed0检查rc0：27项有效payload/ScaleA、eager/graph canary、same-pointer
A/ScaleA/psum动态replay均零字节差异；完整MoE22020096 B零差异。
原reference为logits_diff=3.6105513e-06、rel_l2=0.0026872100，门限保持。
GEMM1实际ELF与CPU候选一致，GEMM2沿用原launcher；该正确性运行不作性能数据。

B-major的E64/seed0同样rc0、27项零字节差异、完整MoE22020096 B零差异，
原reference指标与DS v2相同。两项实际GEMM2完整ELF及`.text/.rodata/.note`
均与新基准相同。固定`small_bracket_v1`为baseline→DS v2→baseline→B-major→baseline，
每组完整三轮；运行前保存plan，保持原CLI、参数和门限。
首次控制启动前外部`dev_al`容器的`python3 sweep_full.py compute1 compute`占用GPU，
每6秒idle轮询等待；未杀外部进程或争用计时。随后首控制组出现ownership排除，
按固定规则整组保留null中值并重试，不从排除轮挑选低值。

### 小候选第一轮正式对照（2026-10-21）

`small_bracket_v1`已完成，仍使用新提交的8/8/TH1、pin0和原CLI。
各合格组均为连续完整三轮；原reference通过，GEMM2实际完整ELF保持新基准。

| group | GEMM1三轮（μs） | GEMM1中值（μs） | GEMM2中值（μs） |
|---|---|---:|---:|
| baseline before try4 | 89.8766 /89.2757 /89.2358 | 89.2757 | 64.1475 |
| DS hint count v2 try1 | 88.3879 /88.4494 /88.5567 | 88.4494 | 64.0866 |
| baseline middle try2 | 89.6173 /89.8725 /89.7755 | 89.7755 | 64.2679 |
| B-major MMA v1 try1 | 93.3134 /93.8281 /93.4591 | 93.4591 | 63.8164 |
| baseline after try1 | 90.5157 /90.8255 /89.6385 | 90.5157 | 64.7085 |

baseline before try1～try3和baseline middle try1存在ownership不合格轮，
全部保留但整组中值为null，不拼接其余轮。DS v2比两侧默认分别低0.8263 /1.3261 μs，
即0.9256% /1.4771%；仍须反向确认，不能先采用。B-major比两侧默认分别慢
4.10% /3.25%，本方向放弃，不以VGPR变化或静态指令数代替实测判断。

`ds_hint_reverse_v1.plan.json`事前固定为DS v2→baseline→DS v2，每组完整三轮。
只有两侧候选均优于中间默认，且适用范围、完整MoE和原精度验证通过，才考虑最小生产修改。
生产tracked源码目前仍为用户新提交，没有新增生产env开关。

### DS hint反向确认与双规模精度

反向固定队列`ds_hint_reverse_v1`已完成，三组各自取完整合格三轮中值：

| group | GEMM1三轮（μs） | GEMM1中值（μs） | GEMM2中值（μs） |
|---|---|---:|---:|
| DS before try4 | 88.2996 /89.1156 /88.4073 | 88.4073 | 64.4114 |
| baseline try1 | 89.5120 /89.3137 /90.1192 | 89.5120 | 64.2468 |
| DS after try17 | 88.7066 /88.4599 /88.7023 | 88.7023 | 64.6851 |

前后DS分别比中间默认低1.1047 /0.8097 μs，即1.2341% /0.9046%，与正向改善方向一致。
DS before try1～try3、DS after try1～try16均存在ownership不合格轮，全部保留null组中值。
期间外部`dev_al`、`xudong_dsl_450_att`、`lipiwang`测试及无法识别的已退出KFD entry
使队列自动等待或排除；没有将这些组拼接、追认为合格或杀死外部任务。
目前可复现的DS附加收益约0.9%～1.5%，仍约88.4～88.7 μs，未达到80 μs。

随后correctness-only补齐E64/seed17与E65/topk9/seed0、17；结合原E64/seed0共四次。
每次27项有效payload/ScaleA、eager/graph canary和same-pointer A/ScaleA/psum动态对比均0字节差异，
每次完整MoE22020096 B亦0差异，原reference门限保持：

| workload | logits_diff | rel_l2 | 原门限 |
|---|---:|---:|---|
| E64/topk8 seed0 | 3.6105513e-06 | 0.0026872100 | 通过 |
| E64/topk8 seed17 | 3.6202693e-06 | 0.0026908165 | 通过 |
| E65/topk9 seed0 | 3.5948931e-06 | 0.0026813769 | 通过 |
| E65/topk9 seed17 | 3.6086727e-06 | 0.0026865052 | 通过 |

E65在本分支仍无匹配CSV，实际使用原t64/w1x4/b3 fallback，M容量17984；
其验证证明现有fallback保持，不能描述为E65使用E64的同一t256 kernel。
本分支E65测试是65列random gating/topk9，没有追加shared route。
没有修改CSV、routing、shape准入或新增expert specialization。

直接scope工具`compile_direct_scope_v1.py`明确将candidate本身用于stage2 compile，
而非通过stage2原launcher隐藏可能的变更；GEMM2实际完整ELF已与默认相同。
其余FP8/no-apre/pin1/fallback/w1x4/b3/persistent compile-only检查正在继续，
全部child仍有120秒硬限制，未完成之前不落地生产源码。

### 最小生产落地与正常入口验收

八个direct scope现已全部完成：GEMM2、FP8/no-apre、原A、pin1、t64 fallback、w1x4、b3、
已有persistent模式的baseline/candidate完整ELF均完全一致。没有通过stage2原launcher替换
掩盖candidate代码差异，scope运行全部compile-only，无GPU dispatch。

四次严格精度和正反向性能确认通过之后，本地及远端生产kernel接入唯一DS计数表达式，
新增4行、删除1行；源码与验收candidate完整SHA相同：
`c8a7eaf9e880e751a982b796f5481527e1287887878f8877045df0a611a01b82`。
wrapper仍`005ccc...`，没有修改helpers、dispatch、benchmark或GEMM2。
新commit的8/8/TH1保留，pin0、A preshuffle1、非persistent和w2x2保持。

`ds_live_forward_v1`首个控制在原reference通过后，检查器手工写错GEMM2预期SHA而rc1，
队列立即停止，所有该轮耗时不作为验收结果；actual G2 artifact仍为原`4bd626e33a25c929...`。
仅修复检查器，让G1/G2预期SHA直接读取已保存的baseline/candidate artifact，
原v1失败日志保留。`ds_live_forward_v2`以新label重跑完整三轮控制→production→控制；
production模式直接执行原benchmark imports，不替换GEMM1/GEMM2 launcher。
本轮performance和scope结果会继续绑定实际source/ELF，尚未commit/push。

`ds_live_forward_v2`正常入口对照已完成，九轮均为完整合格运行：

| group | GEMM1三轮（μs） | GEMM1中值（μs） | GEMM2中值（μs） |
|---|---|---:|---:|
| frozen control before | 90.0265 /89.7734 /89.6512 | 89.7734 | 64.6367 |
| production DS v2 | 89.8725 /88.9891 /89.5311 | 89.5311 | 64.1012 |
| frozen control after | 89.7315 /89.4678 /89.9103 | 89.7315 | 64.6853 |

正常production不替换launcher，GEMM1 actual ELF等于DS候选，GEMM2等于基准；
GEMM1比两侧默认分别低0.2423 /0.2004 μs，即0.2699% /0.2233%。
改善幅度小于私有候选正反向记录，不能用88.4 μs替代正常入口89.5311 μs。
为验证是否稳定，事前固定`ds_live_reverse_v1`为production→frozen control→production，
每组完整三轮，保持相同ownership规则。正式生产保留决定须结合该反向结果。

正常入口反向`ds_live_reverse_v1`亦完成，九轮均合格：

| group | GEMM1三轮（μs） | GEMM1中值（μs） | GEMM2中值（μs） |
|---|---|---:|---:|
| production before | 88.4346 /89.3645 /88.1774 | 88.4346 | 64.6516 |
| frozen control | 89.4572 /89.6049 /89.7251 | 89.6049 | 64.4492 |
| production after | 88.7405 /89.6447 /88.3078 | 88.7405 | 64.0550 |

两侧正常production分别改善1.3061% /0.9647%，正反向方向一致，保留四行DS计数修改。
应同时报告正常正向改善仅0.22%～0.27%，不同完整批次的幅度有波动；
当前正常入口测到的完整中值为89.5311 /88.4346 /88.7405 μs，不挑最快一个
作为稳定耗时，不把private首轮88.4494 μs替代所有正常入口记录，亦未达到80 μs。
GEMM2在每轮完整ELF一致，计时中值波动不作为GEMM1修改的额外收益。

### 下一项独立小feature：WMMA operand reuse hints

旧E65记录里的hint优化没有在本分支成功测过。本次准备仅增加7行的matrix-A hint版，
以及增加9行的matrix-A/B hint版，仍在GEMM1/FP4/A preshuffle/pin0/
非persistent/t256/w2x2/b4 guard内，不按E64、T1536、seed或激活数值特殊准入。
它们分别基于新提交快照，不与DS计数修改混合；若单独有稳定收益才验证组合。

硬件依据Shader Guide rev65 §4.6（TXT10415–10417，p147）、WMMA encoding（TXT11594–11597），
以及§11.3.7 Table116（TXT24030/24035）：hint对应下一条WMMA的复用operand，
current/previous instruction type须保持；reuseA/B位分别位于load-scale DWORD的bit13/14。
当前native matrix A对应weights/B、16VGPR，matrix B对应activations/A、8VGPR，不能按
数学GEMM A/B名称颠倒hint含义。首条、末条及FRONT/BACK交界不添加未经验证的hint。

CPU工具直接执行实际`mma_rows` AST，768项geometry/row cases、332928次operand调用通过；
所有accumulator、scale与数学更新保持。CPU映射不能替代实际ISA顺序或dependency验证。
`run_operand_cpu_v1.sh`正在按A→AB顺序compile-only，每child120秒，
然后要求实际ISA除hint位外PC、operand、指令、资源一致，复用operand之间无分支或覆盖。
此时尚无该项GPU正确性或性能数据，production仍为DS v2。

A-hint v1在120秒内compile-only成功，完整ELF=`401b00bad40643e338b88e480d651ef470d05b5b7ff0978cb7e0a563361d5150`；
原指令PC/资源保持，但native审计拒绝：PC0x37a0的hint指向weight v[514:529]，
下一条WMMA在PC0x3814使用v[562:577]，中间存在prefetch LDS reads。
这证明源码operand排列一致不足以约束post-RA实际WMMA顺序；未dispatch、未计时，
队列立即停止，AB v1未继续编译。该失败不涉及当前生产DS kernel。

v2增加`wm % 2 == 0`条件，仅在同一M32 pair的偶数row尾→奇数row头设置A hint，
移除跨M32 pair的奇数→偶数边界，不增加barrier、pipeline或wave布局改动。
实际source AST再次通过768项/332928calls；新增行数仍A版7行、AB版9行。
v2继续先compile-only及实际native hint/operand检查，没有预设精度或性能结论。

A v2已通过actual native检查：384个WMMA sites中仅48处bit13改变，
完整ELF=`c017c84595fc4257e1183ed40c38a1010c12199abd9fcc60ea13ac29e7248ef5`。
资源VGPR741/SGPR58/LDS327680 B、zero spills/private，与基准一致；所有有效指令、
operand、PC、硬件fences及输出保持，复用operand之间无分支、VALU写覆盖或冲突LDS destinations。
AB v2在相同native operand检查处失败，未dispatch，双hint方向暂时放弃。
两项compiles均rc0且120秒内，ISA gate失败使队列返回rc1，不掩盖该失败。

A v2随后E64/seed0严格GPU验证rc0：27项payload/ScaleA、eager/graph canary、
same-pointer dynamic replay和完整MoE22020096 B均0差异，原reference保持
logits_diff=3.6105513e-06、rel_l2=0.0026872100；actual loaded ELF匹配上述A v2，
GEMM2仍为新基准。此项运行不作性能数据。

固定`operand_a_v2_forward_v1`为DS v2→A reuse v2→DS v2，各完整三轮；
A reuse单独基于新commit、尚不叠加DS计数，优先与当前已保留DS版本比较。
仅有两侧和反向可复现收益才考虑替代；组合需要重新进行独立native、精度与性能验收。

### 单独A reuse对照与DS组合检查（2026-10-21）

`operand_a_v2_forward_v1`已完成，使用固定DS→A reuse→DS顺序。
单独A reuse仍基于用户新commit，不包含DS修正；每组均取连续完整三轮中值。

| group | GEMM1三轮（μs） | GEMM1中值（μs） | GEMM2中值（μs） |
|---|---|---:|---:|
| DS before try4 | 88.0128 /88.7297 /88.7509 | 88.7297 | 64.7233 |
| isolated A reuse try1 | 89.4930 /89.9800 /90.1866 | 89.9800 | 64.3797 |
| DS after try1 | 88.4409 /89.1431 /89.1284 | 89.1284 | 64.5799 |

前控制try1～try3的ownership不合格，整组中值null。单独A reuse比两侧DS分别慢
1.2503 /0.8516 μs，即1.4091% /0.9555%；不替换生产DS版本，不追加单独A版反向计时。
summary已下载且两端SHA一致：
`e7eadfc4dcde9fed9ff5c0a477186538a1570c8453d8039e5d87ebc6a24f51eb`。

随后私有`ds_operand_a_combo_v1`仅在已保留DS版本上增加7行相同受限A hint，
不修改生产源码。源码SHA为
`f418d5e42ebca391e241f6961890393e3fbb4933f993f82201bbdc67e1e3310d`。
不能从单独A版的hint顺序或性能外推组合；组合已重新compile-only并通过actual ISA审计，
完整ELF=`c7c18faa693cd00e2546659a203866efdb1a7df568044deaaa1ad363928d4d74`。
相对DS实际仅48处bit13改变，所有有效指令、PC、operand及资源一致：
VGPR762/SGPR58/LDS327680 B、zero spills/private。全部复用operand之间无分支或覆盖。
每child仍120秒限制。接下来先做E64/seed0严格精度，再固定DS→组合→DS三轮组对照；
只有收益可复现才考虑生产接入。

组合的E64/seed0严格验证已完成：27项payload/ScaleA、canary、dynamic replay及完整
MoE22020096 B全部0差异，原reference保持，actual ELF匹配上述compile-only组合。
`ds_operand_combo_forward_v1`固定三轮组对照结果如下：

| group | GEMM1三轮（μs） | GEMM1中值（μs） | GEMM2中值（μs） |
|---|---|---:|---:|
| DS before try1 | 88.5947 /88.3713 /88.9091 | 88.5947 | 64.2806 |
| DS+A reuse try1 | 88.6898 /88.9682 /88.4493 | 88.6898 | 64.5736 |
| DS after try2 | 88.6095 /89.9316 /88.9279 | 88.9279 | 64.3922 |

后控制try1 ownership不合格，完整组三轮排除。组合比前控制慢0.0951 μs，
比后控制快0.2381 μs，方向不一致，没有证明附加收益；不接入生产，也不追加反向。
新`run_bound_small_feature_bench_v1.py`在原benchmark之后将实际GEMM1完整ELF与
预先编译的对应candidate绑定，GEMM2沿用基准ELF；九轮合格检查均通过。

下一项独立候选在DS版本上只把closing MMA compiler hint数8改为0/4/12/16/20/24，
每候选新增4行、删除1行，仍使用相同GEMM1守卫，非目标路径的值保持8。
所有其他nested helpers、实际数值、地址、ring、load/store和硬件fences AST保持；
各候选直接运行实际`emit_hints` AST的56项cases，总MMA/DS/TDM hint数不变。
当前仅准备及CPU-only编译，不把closing hint当hardware barrier，不预设性能收益。

六个候选compile-only及native检查已完成，每child rc0且120秒内，`.text`确实改变：

| closing hint MMA | VGPR | SGPR | LDS B | spills/private | static instructions |
|---:|---:|---:|---:|---|---:|
| DS default8 | 762 | 58 | 327680 | 0 /0 | 5971 |
| 0 | 733 | 58 | 327680 | 0 /0 | 6003 |
| 4 | 749 | 58 | 327680 | 0 /0 | 6003 |
| 12 | 765 | 58 | 327680 | 0 /0 | 5971 |
| 16 | 773 | 58 | 327680 | 0 /0 | 5971 |
| 20 | 781 | 58 | 327680 | 0 /0 | 5971 |
| 24 | 789 | 58 | 327680 | 0 /0 | 5971 |

保护的WMMA/DS/TDM/output静态计数、barrier及ordered Tensor wait/barrier/DScnt0
序列均保持。static instruction口径沿用原inspector；与reuse审计parser计数不混用。
资源/计数比较不能证明性能或全部physical hazards。

六项E64/seed0严格device验证亦已完成，每项27条有效payload/ScaleA、canary、
same-pointer dynamic replay和完整MoE22020096 B零差异，原reference通过，actual ELF
绑定各自compile-only candidate。每child rc0且120秒内，correctness-only无性能数据。
固定首筛为DS→0→4→DS→12→16→DS→20→24→DS，保持原CLI、pin0和8/8/TH1；
首筛单轮不视作稳定改善，有潜力者才进入完整三轮组和反向对照。

### ScaleA相邻字节合并与DS修正的组合

原ScaleA独立实验有约0.25%～0.83%的有限收益，但未合入用户新commit。
本次`ds_packed_scale_v1`基于当前DS修正重新组合，新增18行、删除4行；
未混入operand reuse或closing hint，也未修改生产源码。全部其他nested helper、
原FP32 amax、e8m0、BF16 rounding及FP4转换保持，只将同一lane的相邻ScaleA
两字节合并写出；相同目标guard下false路径保持。CPU地址98304项、字节packing
65536项、row validity65792项检查通过，ABI和DS修正AST一致。
source SHA=`2c3bdd33bbcdf22a9b79caebbab76eeea1d6a9a2c790d874e202ab7c787ff765`。

实际compile-only ELF=`d805155595a61e87177f5fc4d39ec788003e1e895384667e7554ac21a452d23c`，
VGPR762/SGPR58/LDS327680 B、zero spills/private，资源等于DS版本。
native的ScaleA stores由16条`buffer_store_b8`合并为8条`buffer_store_b16`，
WMMA/DS/TDM/payload/quant操作计数及所有ordered hardware waits/barriers保持；
static instructions5971→5936。编译数量不作为实测收益。

E64/seed0严格device验证已完成，27项payload/ScaleA、canary/dynamic replay以及
4项完整scale buffer（1835008 B，含padding及non-owner canary）均零字节差异；
完整MoE22020096 B零差异，原reference保持，G1实际ELF匹配compile-only，G2为原基准。
每child rc0、120秒内，该运行不作性能数据。

closing初筛遇到外部`dev_al`长期占用时，每6秒idle检查等待；只在已确认没有GPU
benchmark child、队列处于idle gate时将等待job暂挂，完成上述compile-only与无计时精度。
随后恢复原首筛队列，保持所有旧样本，不在计时期间并发编译、GPU检查或文件传输。
预先固定ScaleA组合的DS→packed ScaleA→DS完整三轮队列；要在closing队列结束之后顺序运行。

### 初始TDM owner判断合并（2026-10-21）

等待外部GPU任务期间，依据旧E65记录中`prime-owner`的独立思路，在当前DS v2上准备
`ds_prime_owner_v1`。只在现有GEMM1/FP4/A preshuffle/pin0/非persistent/t256/w2x2/b4
守卫内，把原初始四级`issue(i, i)`改为先`dispatch_wave_job(prime_inputs)`，再按
原stage0→3执行`issue(i, i, my_jobs)`；新增11行、删除2行，没有新增pipeline、
硬件fence、输入stream、wave布局或kernel ABI。其余nested helpers AST保持；反转此处
补丁恢复完整DS module AST。实际`owns/issue/dispatch_wave_job`及初始block的864项CPU
检查共4592个copy events，每wave的TDM、slot及K阶段顺序逐项相同。
source SHA=`35ec97039773c6da717898e5d14e87cba808ae965b621a216334936bc888d520`。

只在已确认closing队列处于idle gate且没有GPU benchmark child时暂挂性能等待，
按原120秒限制完成compile-only；维护期间另保留6秒全机idle检查，随后恢复原队列。
候选完整ELF=`4d41ec23944b8cd6234cf42640b171200583b15195a5cb8af3618d4770b38adb`。

| native指标 | DS v2 | prime owner |
|---|---:|---:|
| VGPR | 762 | 762 |
| SGPR | 58 | 54 |
| LDS B | 327680 | 327680 |
| spills/private | 0 /0 | 0 /0 |
| static instructions | 5971 | 5797 |
| first DS之前static instructions | 583 | 419 |
| first DS之前条件branches | 18 | 6 |
| first DS之前input TDM sites | 16 | 16 |
| native backedges | 7 | 7 |

v1 native检查器仅识别十六进制branch immediate，objdump实际输出十进制，导致
backedge记录误为0/0。保留v1 receipt，新v2只修解析并独立重算原ISA，正确值7/7；
没有重新编译、改candidate或放宽任何门槛。原WMMA/DS/TDM/output opcode计数、
ordered tensor/WG/DS release/output fences均保持。静态指令和资源下降不能替代实测。

候选E64/seed0 untimed严格验证rc0、120秒内，27项有效payload/ScaleA、eager/graph
canary及same-pointer dynamic replay全部零字节差异，完整MoE22020096 B亦为零差异；
原reference为`logits_diff=3.6105513e-06`、`rel_l2=0.0026872100`。actual GEMM1 ELF
匹配上述compile-only candidate，GEMM2保持原完整ELF。此运行不作为性能数据。

固定`ds_prime_owner_forward_v1.plan.json`为DS→prime owner→DS，各完整三轮；
只有双侧改善后才追加固定反向确认。本候选尚未修改生产文件，性能尚待GPU空闲后测试。
硬件依据Shader Guide rev65 §4.10.1 p197：TDM忽略EXEC，同wave的Tensor instructions
按序完成、跨wave及相对其他memory无序；本修改保持原每wave请求顺序和所有硬件fences。

截至`2026-10-21 05:09 UTC`，外部`dev_al` PID3981390仍持有GPU0/KFD，性能队列
仍停在`closing0_try2`的idle gate，每6秒自动重检。此前`closing0_try1`的92.7357 μs
在ownership不合格后排除，不能作为closing0性能结论。当前唯一生产修改仍为DS计数
4行新增/1行删除，正常入口89.5311/88.4346/88.7405 μs记录保持；尚未达到80 μs。

### 非负swizzle除法与当前队列（2026-10-21）

上述外部PID3981390约在05:17 UTC结束，此后出现其他外部任务。到05:24 UTC，
外部`dev_al` PID362614的`python3 -m aiter.aot.flydsl.gemm`仍持有GPU0/KFD；
即使瞬时busy=0，也不视作全机空闲。closing队列在`closing4_try2`的idle gate等待。
此前closing0的try1～3、closing4的try1均因ownership不合格排除，保留完整原记录。
closing0 try4的93.0648 μs和control0的88.7845 μs是单轮，且中间有较长外部等待；
须结合下一侧control后评价，不作为稳定的相邻对照收益或回退结论。

利用上述等待，确认没有GPU benchmark child后暂挂性能job，完成私有
`ds_unsigned_swizzle_v2`的compile-only、native及无计时精度；维护期间继续6秒idle检查。
此项仅将swizzle中三处已证明非负的坐标除法使用`arith.divui`，新增11行、删除3行；
不加入N4096特例、不改变runtime N/M、cluster映射、expert lookup及原fallback。
所有其他nested helpers和反转补丁后的完整module AST保持。actual AST检查覆盖
413903项映射与91296项非负除法，原full/tail-group结果逐项一致。
source SHA=`0413de4618af042a7742195d16bfb11dbf5139228e8cc63aa763d89a8486f20f`。

本地CPU audit v1曾在每个坐标重复编译两个相同AST，接近90秒时停止了自己的本地
Python audit进程，没有GPU dispatch或artifact。v2只缓存两份immutable compiled AST，
约十几秒完成同一审计，逻辑和验证范围不变；失败和修复记录分开保留。

实际compile-only ELF=`f54a14c60d0cab7020fac6f83e1569e3fd5ed9833b713a1b37538bca2a065f40`，
VGPR762/SGPR58/LDS327680 B、zero spills/private，与DS保持。
static instructions5971→5939，first DS前583→556，backedges7→7；
保护的WMMA/DS/TDM/output及quant opcode计数、ordered hardware fences保持。
该变化减少了编译后的地址计算指令，尚不能说明实际性能提高。

E64/seed0严格device验证rc0且120秒内：27项payload/ScaleA、canary/dynamic replay与
完整MoE22020096 B均零字节差异，原reference保持
`logits_diff=3.6105513e-06`、`rel_l2=0.0026872100`。actual G1 ELF与compile-only绑定，
G2完整ELF仍为原基准。该运行仅用于正确性，不作性能数据。

事前固定性能顺序closing首筛→DS/packed ScaleA/DS→DS/prime owner/DS→
DS/unsigned swizzle/DS，后三项各完整三轮组，不并发运行。新串行runner先核查
上一队列完成、原precision完成标记、对应native/ELF和GEMM2，再执行下一项。
双侧收益成立后才补反向确认与其他seed/shape scope；当前生产仍只有DS修正。

### DS first hint候选（2026-10-21）

关闭pin的已有8/8/TH1参数遍历保持；额外在DS修正上分别准备`ds_first_n=0/4/8/12`，
每候选新增4行、删除1行。原16对照不变，所有device nested helper及反转补丁后的
完整module AST保持；只改变GEMM1守卫内编译hint的首组LDS数量。未增加pipeline、
persistent或wave布局，没有全局env覆盖GEMM2。四项各12个actual赋值/false-scope
检查通过；compile-only均rc0且各自在120秒内。

v1 native gate在first0处拒绝，停止该CPU队列，没有继续GPU dispatch。
随后v2仅对已经编译完的四份ISA分别记录相同门槛，不重新编译、不放宽gate：

| ds_first_n | VGPR | SGPR | LDS B | spills/private | static instructions | native gate |
|---:|---:|---:|---:|---|---:|---|
| DS default16 | 762 | 58 | 327680 | 0 /0 | 5971 | 对照 |
| 0 | 749 | 58 | 327680 | 0 /0 | 6043 | 额外dscnt0导致ordered fences序列不同，拒绝 |
| 4 | 749 | 58 | 327680 | 0 /0 | 6035 | 通过 |
| 8 | 757 | 58 | 327680 | 0 /0 | 6003 | 通过 |
| 12 | 762 | 58 | 327680 | 0 /0 | 5971 | 通过 |

first0门槛不合格不能直接证明kernel bug；仅说明这份native与事前保护的同步序列
不相同，本轮不dispatch它。其余三项保护的numerical/DS/TDM/output计数、硬件barriers
和ordered tensor/barrier/dscnt0均与DS对照保持，`.text`均有实际变化。

first4/8/12随后各一次E64/seed0严格device验证rc0、120秒内，每项27条payload/ScaleA、
canary/dynamic replay和完整MoE22020096 B零字节差异；原reference保持，actual G1
匹配各自compile-only ELF、G2完整ELF保持新基准。此处均correctness-only、没有性能。
固定screen为DS→4→DS→8→DS→12→DS，仅单轮初筛，有潜力才做完整三轮及反向。
此队列必须在前述closing/packedScaleA/prime/swizzle串行实验结束后执行。

unsigned候选的native receipt/ELF/精度日志已下载。本地v1 checker按设计拒绝覆盖
已下载receipt，随后独立read-only工具重算record与原receipt完全相同；不重写原证据。
receipt=`ds_unsigned_swizzle_downloaded_acceptance_v1.json`，无新GPU运行或计时。

### closing hint初筛完成与后续固定对照（2026-10-21 06:01 UTC）

`ds_closing_screen_v1`已完成十个事前固定点，合格单轮如下；所有ownership不合格
尝试仍保留，未拼接或根据耗时重试。`closing16`的try1～7、中央控制的try1～5
均排除，表中只列各点最终合格的单轮，不当作三轮中值。

| point | 合格attempt | GEMM1 μs | GEMM2 μs |
|---|---:|---:|---:|
| DS control0 | 1 | 88.7845 | 64.0465 |
| closing0 | 4 | 93.0648 | 64.8414 |
| closing4 | 6 | 90.6757 | 63.5637 |
| DS control1 | 3 | 88.7824 | 64.1834 |
| closing12 | 1 | 89.7417 | 64.4029 |
| closing16 | 8 | 89.6934 | 64.6728 |
| DS control2 | 6 | 88.5274 | 64.3627 |
| closing20 | 1 | 90.1888 | 64.2047 |
| closing24 | 1 | 87.8947 | 65.0480 |
| DS control3 | 1 | 88.8479 | 64.4618 |

0/4/12/16/20没有首筛优势，不追加这些值的完整性能队列。24比两侧单轮控制低
0.6327/0.9532 μs，允许下一步固定完整DS→24→DS三轮组确认，但仍不能认定稳定
收益。计划为`ds_closing24_forward_v1.plan.json`；仅在两侧完整控制均改善后追加
一次固定24→DS→24反向，任何采用之前还须补其他seed及非目标scope。

第一次v1续跑shell命令在性能idle gate被暂挂后提前执行，plan上传尚未完成，
主机报FileNotFoundError；没有进入container或GPU、没有取得性能。原host.log保留，
完整对照改用`ds_closing24_forward_v2.plan.json`新label，并强制核对前序summary完成。

closing结束后，原`run_pending_small_features_v1.py`已启动，串行执行packed ScaleA、
prime owner、unsigned swizzle的DS→candidate→DS，各完整组三轮，再执行ds_first
首筛。没有并发GPU测试，也没有重新扫描已完成的参数。packed ScaleA前控制try1
ownership不合格、组三轮中值null，try2重新从全机idle gate开始；外部任务保留。

当前生产仍只有DS计数4行新增/1行删除，8/8/TH1和pin0保持；本节没有新的生产
性能成绩。正常入口三组89.5311/88.4346/88.7405 μs记录保持，尚未达到80 μs。

早先`param_raw_checkpoint_20261021_0543`全量封包在约90秒时由本任务主动停止，
以优先恢复GPU空闲后的性能队列；仅有不完整tgz/manifest/log，没有完整receipt，
不能作为已验收归档。后续另用新名称和低压缩的限定证据包，保存source/tools、
原CLI/ownership日志、完整ELF和ISA，体积较大的MLIR保留远端；不下载`.out/.pt`。

### 已下载证据的独立复算（2026-10-21）

限定快照`param_compact_checkpoint_20261021_0612.tgz`已下载，共4878 regular
members、19492136 B，archive SHA256为
`eb7726b5dfaea50ab991df51f5d65abbf170d8e19cd6377728a619817a8fceb5`。
这是已完成source/tools、原CLI/ownership、ELF/ISA的compact checkpoint；重复MLIR
保留远端，当前未完成的packed队列和idle observer日志排除，不称为全量raw归档。

`accept_downloaded_evidence.py`本地rc0，4877项manifest大小/SHA通过；从原CLI
重新计算33个合格完整三轮组中值，所有ownership不合格组仍为null。
`verify_compact_screens_v1.py`本地rc0，额外独立复核单轴98轮、交叉57轮和closing32轮，
确认68/35/6个非默认候选、原uniform/auto E64/T1536/topk8和pin0、结束ownership、
每份实际ELF的SHA及全部GEMM2基准完整ELF。单轮只用于screen，不升级为三轮成绩。
receipts分别为同名`.downloaded_acceptance.json`与`.screens_acceptance.json`。

compact封包v1重复stat/read每个member，120秒到期被timeout终止，没有receipt，
不完整包保留但不下载验收。v2每member读取一次，SHA针对实际写入tar的同一份bytes，
manifest放最后，低压缩约0.72秒完成；没有放宽性能或精度检查。封包/传输期间只暂挂
本任务idle gate，六秒只读observer继续；恢复性能前停止observer，没有GPU计时并发。

### closing24双seed与E65原fallback精度（2026-10-21）

利用外部GPU owner等待，closing24追加E64/seed17、E65/topk9/seed0及17严格验证。
三个child均rc0、120秒内，每次27条有效payload/ScaleA、canary/same-pointer replay
及完整MoE22020096 B零字节差异，原reference门限保持。这些是correctness-only，
全部耗时排除；不需要GPU空闲，也不将外部并发时的读数当性能。

E64 actual ELF精确等于已通过native gate的closing24
`d0e7b45660b437ba6b03e0c1c8da06e1df505a58e52102baef5f5026862fbe2e`。
E65原`tile=64/256/256 waves=1x4` fallback的G1完整ELF仍为
`d173f0b236adfb2d61acb596dc06b2c46b1e02ea9d75aeae8016db902c18fff3`，
G2仍为`3b63a9571dca9410100499f8eb3507d09f3b248494077ca4940d1b09f0abc5ec`；
与之前DS双seed的实际fallback artifact精确相同，未新增E65 specialization或改变dispatch。
本分支E65并非E64同一t256 kernel，此处不改变其既有shape限制和routing。

同一轮维护发现shell排队的ds_first首筛在上一idle gate被暂停时提前启动其第一个
idle gate；确认没有GPU child后暂挂并保留记录，必须在pending三项完成后再恢复。
closing24完整性能runner也增加强制前序summary完成检查，不依赖终端排队作串行保障。

### closing24非目标配置编译与新增证据验收（2026-10-21）

`ds_closing24_direct_scope_v1`的八项直接候选编译均rc0、各自120秒内完成，
没有GPU dispatch。GEMM2、FP8 input、original A、pin1、fallback t64、w1x4、
buffers3和persistent1的完整ELF分别与已验收的原基准artifact完全一致。
这里直接编译候选stage2，未通过原GEMM2 launcher绕开候选来代替scope检查；
不新增persistent路径，也不修改任何shape准入。

新增compact checkpoint `param_compact_checkpoint_20261021_0630.tgz`已下载，
19698090 B、4943 regular members，archive SHA256为
`45ada4ffc0129dcf320374f61fd9e96a9685109f3a82af8668015bf98200b5e7`。
本地独立检查4942项实际member的大小/SHA、33个已完成合格组三轮中值及所有
invalid组的null中值通过；追加三项strict记录、八项direct scope的exit、原日志和
artifact绑定亦逐项通过。活动packed/first性能队列与observer日志未计入此快照，
重复MLIR仍留远端，没有下载`.out/.pt`。

确认维护compile和observer结束后，恢复原pending队列，first首筛继续暂停，
没有在运行中的shell预排后续命令。packed ScaleA的前侧DS控制try1～3因ownership
不合格排除；try4三轮中值为GEMM1 88.4177 μs、GEMM2 64.6537 μs。
候选try1～2同样ownership不合格，整组中值为null；截至06:36 UTC尚无合格候选组。
当前生产仍只有DS计数修正，不据上述排除组认定候选收益。

### packed ScaleA后侧对照fault及限定诊断（2026-10-21）

`ds_packed_scale_forward_v1`的候选try1～5均因ownership不合格排除；try6合格，
GEMM1三轮87.9958 /88.7678 /87.9751 μs，中值87.9958 μs，GEMM2中值63.7408 μs。
前侧DS控制try4为88.9806 /88.3273 /88.4177 μs，中值88.4177 μs。
候选比这一侧快0.4772%，但后侧DS控制try1_r1于06:40 UTC发生
`Memory access fault by GPU node-2 ... on address 0x25a2000. Reason: Page not present or supervisor privilege.`，
child rc134；串行队列立即停止，没有运行prime、unsigned或ds_first的GPU child。
失败轮全部耗时排除，后側组不形成三轮中值；不将本项标记为稳定收益。

故障轮已输出GEMM1/GEMM2的profiler table，尚未完成原reference及最终artifact绑定。
post/settled观察到外部`tingchen_mla_att` PID1522519持有GPU/KFD，但现有证据
不能将fault唯一归因于外部并发、DS hint或reference。检查后未发现本任务GPU child残留。

`fault_stage_probes_v1`按固定原提交→DS各运行一次原benchmark，新增同步/阶段标记，
没有改变任何数值代码、量化或reference门槛；因此两次全部计时排除。
两项均rc0、120秒内，三次原run_perftest、输入和stage2两次quant、两阶段PyTorch
reference及最终精度全部通过同步检查；G1分别绑定原基准/DS ELF，G2保持原基准ELF。
本次未复现fault，也未确定fault原因；不能凭新增同步下通过宣称异步风险已解决。

保留失败v1全部日志与summary，新建`ds_packed_scale_forward_v2.plan.json`重新固定
DS→packed→DS各完整三轮，不拼接v1样本。恢复runner显式要求上述诊断完成，仍在任何
执行/超时/精度/artifact失败时立即停止，只有ownership不合格可自动整组重试。
后续prime/unsigned保持原次序；暂停的first首筛须在该恢复队列完成后才继续。

### owner bit mask静态候选与编译超时（2026-10-21）

等待GPU外部任务期间，检查同一入口owner集合的条件化简。私有`ds_owner_mask_v1`
新增7行，仅在既有GEMM1守卫内，对齐且连续的两/四wave owner集合使用
`(wave & -len(wv)) == wv[0]`；非连续集合、其他配置维持原比较/OR逻辑。
源码SHA为`a7970fb880aff832f6cd3c4cf329f2d07994addd0ef79597a3b3ad9aca69ba83`。
本地直接执行actual `owns` AST的59136项owner/wave/false-scope组合完全一致，
其他helpers及反转后的完整module AST保持；没有改wave分工或TDM issue顺序。

仅在确认没有GPU benchmark child后暂挂恢复队列的idle gate，继续6秒只读observer，
上传上述任务私有文件并执行compile-only。`ds_owner_mask_cpu_v1`在120秒到期返回
rc124，未形成ELF/artifact、没有GPU dispatch。退出检查未发现编译child残留。
该候选未获得编译证据，不计时、不采用、不延长超时；不能由此判断性能或编译超时原因。
维护observer退出后恢复原性能idle队列，first首筛继续暂停，未预排其他命令。

重新核对quant helper发现当前路径以未舍入F32 `all_vals`求amax，再做BF16→native FP4。
旧记录的tied rounded-vector reuse与packed BF16 amax依赖另一种数值路径，不能直接
移植而不改变ScaleA规则。本轮不添加这两个feature，不据转换静态计数认定重复rounding。

### a07-3重启中断packed对照（2026-10-21）

本轮实际检查boot_id为`2b465a09-eea1-4323-8fbd-3e7c225104dd`，
`uptime -s`为`2026-10-21 08:10:02`；此前boot为
`1d5146ac-1863-43d0-8211-f0761013cc4a`。旧runner、idle gate及其测试PID均已消失，
因此没有仍在等待的原性能队列，也不恢复旧PID。

磁盘保留的`ds_packed_scale_forward_v2.summary.json`显示completed=false：
前侧DS try1合格，GEMM1中值88.8309 μs、GEMM2 64.3058 μs；packed try1/try2
不合格，中值为null，后侧对照未完成。本项没有稳定收益结论，不跨boot拼接样本，
不采用旧v1的87.9958 μs作为已验证最优。按照用户优先级先完成f01-1四卡非GEMM
移植任务；后续如恢复packed，须重检新boot环境并建立新的packed-only完整对照，
不得启动旧父程序中暂停的prime/unsigned实验队列。

### EP4任务完成与新boot的packed-only恢复（2026-10-21）

f01-1的四卡非GEMM优化已完成，详见`moe_ep4_non_gemm_optimization_port.md`。
两route共wave copy在三组正反向原命令中将fused中值505.837938降至502.140469 μs/层，
copy中值16.427257降至12.319528 μs。最终normal入口、原FP32 reference与逐字节
baseline/candidate验证均通过；该修改不影响此处A-preshuffle GEMM1路径。

a07-3当前boot仍为`2b465a09-eea1-4323-8fbd-3e7c225104dd`，dsr1_ep4运行中，
FlyDSL为原`0.3.3.dev882`、Torch2.11.0+rocm10.1、HIP7.16.26315、可见一张GPU。
远端HEAD仍`ada68ed342bc31a1e57d676e87f8d3ab98216c51`，branch为
`dev/yadai_dsr1_conc6144`；实际wrapper/kernel分别为`005ccc...`/`c8a7eaf...`，
与本地当前compute source一致，未修改Git状态。

新`ds_packed_scale_reboot_e64_s0_v1`严格验证rc0、120秒内：27项有效payload/ScaleA、
完整scale buffer、canary/dynamic replay通过，完整MoE22020096 B零差异。
logits_diff=3.6105513e-06、rel_l2=0.0026872100，原0.01门限不变。
actual G1 ELF为已验收`d805155...`，G2仍`4bd626e...`；全部耗时为correctness-only。

单独新计划`ds_packed_scale_reboot_forward_v3.plan.json`固定DS→packed→DS各三轮，
不拼接旧boot或旧fault数据，不自动续跑prime/unsigned。该队列当前在第一轮idle gate
等待外部`dev_al`的CPU/AOT进程释放KFD；即使VRAM0、busy0也不能作为全机空闲。
所有旧日志保留，未清理外部任务；每6秒自动重检。

### 完全无效M64分组跳过：私有小候选（2026-10-21）

静态`ds_rowtail_half_v1`基于DS原源码，仅增加guard与两处wave-uniform条件。
每wave原FRONT/BACK各覆盖M64；仅当整个M64分组都在expert有效行之外才跳过
该组WMMA，部分有效分组保留全部M16和lane。有效accumulator原K128更新顺序、
全部输入loads、TDM requests、rmem及hardware fences不变，没有修改shape准入、
布局、persistent、VGPR pin、quantization或GEMM2。

直接执行实际`k_step` AST的2072项边界/调用组合通过：有效更新序列、input issue/load/
fence序列完全一致，关闭guard的结果等同原版，跳过项均为完全padding。
反转补丁恢复整个原module AST。source SHA为
`c1eafffbda988b25bf71eb5c1ecb7fc056b2a2f7dc5a10d7decd397633ddbf4d`。
依据Shader Guide rev65 §4.6.12 p144，WMMA会强制全EXEC计算，不能只依赖
lane masking；必须核验实际native branch跳过。CPU检查不代替ISA或GPU精度。

确认packed队列只有idle child、没有GPU测试后，临时暂停本任务idle gate，另保留
6秒只读idle检查，执行本候选限时compile-only。生产GEMM1仍是DS计数4行新增/
1行删除，当前尚无本候选GPU正确性或性能结论。

### 新boot packed对照完成：回退，放弃（2026-10-21）

`ds_packed_scale_reboot_forward_v3.summary.json`已completed=true。全部有效组均为
同一boot、A preshuffle=1、VGPR pin=0、原warmup5/iters20；每组三次，剔除整组
ownership不合格记录，不取最小值，也不拼接旧boot数据。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| DS before try1 | 94.5405 /95.3249 /94.0745 | 94.5405 | 60.6753 |
| packed try2 | 95.3944 /94.6228 /97.2836 | 95.3944 | 60.5297 |
| DS after try1 | 92.9149 /95.3461 /92.0568 | 92.9149 | 61.4404 |

packed try1因ownership不合格排除，中值为null。try2相对前后DS分别慢0.9032%/
2.6686%；本feature不进入生产，不再做反向对照。虽然将16条B8 ScaleA store
改为8条B16 store通过精度验证、没有spill，但该指令减少没有带来本workload的收益。

114项小证据archive已下载并校验，SHA256为
`501c8213c604fab388c1cb66448b6863e532d2b8e9b69452610b37c496e8b62c`。
本地验收见`.codex_tmp/yadai_gemm1_param_20261021/logs/`
`ds_packed_scale_reboot_forward_v3.local_acceptance.json`。

### M64候选ISA与双seed正确性完成（2026-10-21）

`ds_rowtail_half_cpu_v1`在120秒内compile-only成功，资源为VGPR795、SGPR58、
LDS327680B（320KiB）、private0、VGPR/SGPR spill均0；DS版本VGPR762。
GEMM1 ELF SHA256为
`7d5fc4aafc1c1521daef05abc72e62272f480a53a1366ded77cf74c49f019276`。

`audit_rowtail_native_v1.py`检查实际CFG、`S_SET_VGPR_MSB`物理operand bank和两类
DS/WMMA寄存器hazard。19处forward branch的短body各跳过16条WMMA，不含DS load、
TDM或barrier；protected静态操作数量及barrier计数与DS相同。有限模型没有发现
DS未完成就被WMMA读取、或WMMA仍需读取就被DS覆盖的问题。模型不覆盖全部VALU/VMEM
hazard和descriptor语义，不能作为完整硬件正确性证明。LLVM把一个owner路径放到
`endpgm`后，故不能宣称线性ISA中的所有tensor/barrier次序逐项相同。

两次GPU验证`ds_rowtail_half_strict_e64_s0_v1`/`s1_v1`均rc0、120秒内。
每次27项有效payload/ScaleA检查、4项完整ScaleA buffer canary检查、eager/graph、
同pointer修改A/ScaleA/psum的dynamic replay和输入恢复均零字节差异；
边界包括0/1/63/64/65/127/128/129/191/192/193/255/256/257/511。
完整MoE每次22020096B与DS零字节差异，GEMM2有效payload也完全相同。

| seed | logits_diff | rel_l2 | 原精度门限 |
|---:|---:|---:|---|
| 0 | 3.6105513278025183e-06 | 0.0026872099842876196 | logits_diff < 0.01 |
| 1 | 3.6122000055255672e-06 | 0.0026878188364207745 | logits_diff < 0.01 |

actual G1 ELF等于上述CPU artifact；actual G2 ELF均保持
`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。
本地两seed的`.local_acceptance.json`已形成，所有验证耗时仅供正确性、不计性能。
候选仍在私有文件内，未修改生产的shape准入；本分支E65原有t64/w1x4 fallback
也未改变，不把E65说成走E64的t256 kernel。

### 用户指定正常入口重测（2026-10-21）

用户指定直接运行`op_tests/flydsl_tests/test_flydsl_grouped_gemm.py`。本次正常入口
不使用`runpy`或process-local launcher包装；只在主机临时替换任务相关kernel源码，
退出时校验并恢复DS原bytes。wrapper、benchmark文件及原`run_perftest`不修改。
本地与远端benchmark内容一致，只有CRLF/LF bytes区别。

`run_normal_groups_v1.py`外层启动原命令；A preshuffle=1、VGPR pin=0、
清空MMA/TH覆盖参数、warmup5/iters20、seed0 uniform/auto、原reference门限不变。
`timeout --signal=TERM --kill-after=5s 120s`限制每个child。
`FLYDSL_RUNTIME_CACHE_DIR`只指定任务私有磁盘cache；测试退出且完成post/settled检查后，
另用`audit_normal_cache_v1.py`读取缓存并核对实际GEMM1/GEMM2 ELF，不在计时中添加hook。

首轮`user_normal_entry_bracket_v1`原脚本rc0、精度通过，但post/settled发现外部
`lipiwang` PID1286470占GPU，该轮94.7028/60.2978 μs全部排除。
退出后的cache检查最初漏匹配kernel名的`_apre_`，导致队列停止；检查脚本已修正，
重新核对首轮ELF与原基准一致。原v1失败summary与日志保留，源码已自动恢复，未动外部进程。

新建`user_normal_entry_bracket_v2`，固定baseline→DS→M64→DS→baseline各完整三轮。
此处baseline为参数提交`346438a75506f77b520d3a9a09ba4985f054e377`的kernel源码
`259f9ff2b24dbd034f360fb7a5c491129406791fe651bda90fe9ae8e6fd3b942`；
不将它与早期OPTX任务的`339589f900b3a880ec447f605886fe803cbf9649`基准混用。
每轮全机GPU/KFD空闲后才运行，post ownership与settled全空闲才能接受计时。
该正常入口对照的最终结果将在完成后追加。

### 正常入口最终对照与分支同步（2026-10-21）

`user_normal_entry_bracket_v3`已完成，退出时生产源码恢复为DS版本。完整顺序为
baseline→DS→M64→DS→baseline，各组三轮；全机空闲检查不合格的整组中值为null，
重新完成三轮后才接受。没有选取最小值，也没有跨boot混合数据。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| baseline before | 92.5816 /96.6828 /96.9992 | 96.6828 | 61.0421 |
| DS before | 91.9957 /95.8541 /93.4591 | 93.4591 | 61.2022 |
| M64 skip | 95.5547 /96.2757 /97.1994 | 96.2757 | 61.7882 |
| DS after | 92.2805 /94.1505 /95.2553 | 94.1505 | 60.6141 |
| baseline after | 95.6286 /97.3154 /92.5966 | 95.6286 | 61.0398 |

DS对对应两侧baseline的改善为3.33%/1.55%。M64候选比两侧DS分别慢3.01%/2.26%，
不落地、不继续反向探索。全部有效运行原精度检查通过，logits_diff约3.61055e-06，
门限0.01不变。原benchmark与`run_perftest`不修改；实际GEMM2 ELF始终为
`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。
当前最佳保留feature是DS scheduling计数修正，仍未达到80 μs；它是调度hint修正，
本实验未将其证明为race或数值bug。

证据archive共454项、1916982B，SHA256
`cf2facf40f6da97efa23f4ed4f8166fb5fd89544b61d56ca7333e5e55c8c1a21`。
本地重新校验所有成员、有效组中值及ELF，
`.codex_tmp/yadai_gemm1_param_20261021/logs/user_normal_entry_bracket_v3.local_acceptance.json`
的`passed=true`。

按用户要求，a07-3主机目录
`/data/yanguahe/code/wk_sp1/aiter_a4w4_prefill_v2_yadai`已更新到本地分支
`hyg_dsr1_conc6144`、HEAD`346438a75506f77b520d3a9a09ba4985f054e377`，随后scp
两个未提交修改文件。原远端改动通过stash和原bytes/patch保留，没有clean/reset。
DS文件SHA256为`c8a7eaf9e880e751a982b796f5481527e1287887878f8877045df0a611a01b82`；
copy文件为`ec3919df96c7f6b2a584a35a234bbbe44c061f10ddafd1159c186a175c79f947`。
分支、HEAD、两文件SHA256及wrapper均与本地一致。

同步后直接运行用户原脚本，固定A preshuffle=1、VGPR pin=0，结果
GEMM1 96.2991 μs、GEMM2 60.7258 μs；原精度通过，post全机无GPU/KFD进程。
日志为`a07_after_sync_user_entry_20261021_v1.log`。这是一次正常入口确认，
不与上面三轮中值混算。该单卡BF16 quant入口不会执行EP的prequantized copy，
不能用本次GEMM计时声称测到了非GEMM `_rpg2`收益。

### a07-3重启后的baseline重测（2026-10-21）

用户确认a07-3重启完成后，在新boot
`cb28eed4-51ca-4cdc-bf5b-ecbc1d54db5f`重新执行原正常入口。
分支/HEAD仍为`hyg_dsr1_conc6144` /
`346438a75506f77b520d3a9a09ba4985f054e377`，FlyDSL发行版本经
`importlib.metadata.version("flydsl")`确认是`0.3.3.dev882`（module的
`__version__`显示`0.3.3`）。本轮baseline是该提交的pin0 GEMM1；
不与早期`339589f9`或重启前的耗时混算。

保持A preshuffle=1、VGPR pin=0、random non-balanced、uniform/auto/seed0、
warmup5/iters20，清空MMA/TH/DS_FIRST覆盖参数。原benchmark、计时helper、
计算、精度门限均未改。固定baseline→DS→baseline，每组三次：

| 版本/顺序 | GEMM1三次 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| baseline before | 90.0077 / 89.7777 / 89.2064 | 89.7777 | 64.0491 |
| 当前DS优化版 | 88.5421 / 88.5650 / 87.5848 | 88.5421 | 64.4708 |
| baseline after（重测组） | 89.3471 / 89.4061 / 89.1844 | 89.3471 | 64.0724 |

DS对前后baseline改善1.3763% / 0.9010%，当前仍高于80 μs目标8.5421 μs。
新boot的baseline约89.3–89.8 μs，不能继续把旧boot约96 μs作为本轮基准。
GEMM2源码及实际ELF不变；表内约0.4 μs的组间波动不属于GEMM2代码优化。

九次有效运行均rc0、logits_diff约3.61055e-06，原门限0.01。
后置baseline try1的r1测后空闲检查不合格，整组三次全部排除，中值为null；
try2重新完成完整三次并通过全机GPU/KFD pre/post/settled检查。
没有按快慢剔除样本，没有跨boot拼接。每个测试child限时120秒。

实际GEMM1 ELF仍分别为baseline
`be083c8c6beeccb4d9164a37f67c9a60ae9fd4ffcbfb668f06318561443ad052`、DS
`1f940d496d0e2b343f71ee8d6124653d0adb892c9049e2bd0d93f6a4e5d9d5b2`；
GEMM2保持`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。
退出后恢复并SHA核验生产GEMM1为`c8a7eaf9...11a01b82`，无测试残留。

重测复现仍使用本文开头的原命令。完整三组外层runner：

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_normal_groups_reboot_v1.py \
  NEW_UNIQUE_LABEL baseline ds baseline
```

原始日志、manifest、空闲记录、每轮ELF核验及汇总位于
`.codex_tmp/yadai_gemm1_param_20261021/logs/reboot20261021_baseline_ds_v1*`。
汇总`completed=true`、`source_restored=true`。此次重测未新增GEMM修改。

### 新boot初始owner合并与unsigned swizzle的正常入口结果（2026-10-21）

继续使用相同boot、原benchmark、A preshuffle=1、pin0、8/8/TH1和
warmup5/iters20，依次完成两个DS→candidate→DS对照。每组必须完整三轮
通过pre/post/settled空闲检查，ownership不合格的整组保留null中值并重测。
没有使用私有计时替换原入口，也没有将更快单轮替换中值。

| 实验/组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 初始owner：DS before | 88.2187 / 89.2055 / 88.8069 | 88.8069 | 64.2398 |
| 初始owner：candidate | 89.0474 / 88.7604 / 89.0894 | 89.0474 | 63.9412 |
| 初始owner：DS after | 88.7320 / 88.4853 / 89.3096 | 88.7320 | 63.7551 |
| unsigned swizzle：DS before | 88.9480 / 87.8601 / 88.8297 | 88.8297 | 64.0723 |
| unsigned swizzle：candidate | 88.5240 / 88.4143 / 88.4626 | 88.4626 | 63.9332 |
| unsigned swizzle：DS after | 88.0619 / 88.3172 / 89.2934 | 88.3172 | 64.4307 |

初始owner候选只把原stage0–3的owner判断合并，全部TDM jobs、slot及发射次序保持，
但比两侧DS分别慢0.2405/0.3154 μs，不采用。unsigned候选只显式使用非负除法，
比前DS快0.3671 μs、比后DS慢0.1454 μs，方向不一致，没有稳定收益，不采用。
两个方向均不追加无收益反向对照，生产仍只有DS计数修正。

所有有效运行原reference通过，logits_diff为3.61055e-06，原门限0.01。
actual G1 ELF分别绑定既有DS、prime与unsigned artifact；所有运行的GEMM2完整ELF
保持`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。
两次退出均恢复DS原bytes，未改变wrapper、GEMM2或其他kernel。

两份完整证据archive已下载并逐成员SHA、原表格、空闲记录、实际ELF及中值独立复算：
`reboot20261021_prime_normal_v2.evidence.tgz`为241项，SHA256
`cde646ddb6065b4b2c8407a3e1d3c1de1045333989533af1a57a38884d10c3e2`；
`reboot20261021_unsigned_normal_v1.evidence.tgz`为142项，SHA256
`b5b37568b4aafc5d274ce932068b5cb2f4957272874bb0bba8a259997bfcfcb8`。
对应`.local_acceptance.json`均`passed=true`。

复现外层命令（每次使用新label）如下；内部调用本文原正常入口，child仍限时120秒：

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_normal_small_features_reboot_v2.py \
  NEW_PRIME_LABEL ds prime ds
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_normal_small_features_reboot_v2.py \
  NEW_UNSIGNED_LABEL ds unsigned ds
```

### M32无效WMMA跳过的完整正常入口对照（2026-10-21）

私有`ds_rowtail_m32_v1`只在既有GEMM1/FP4/A preshuffle/pin0/
非persistent/t256/w2x2/b4 scope中，通过wave-uniform的M32 pair行边界判断，
跳过整组完全没有有效行的WMMA。部分有效pair仍保留全部M16/lane和原56次K128更新；
输入TDM/DS、有效数值、同步、activation/quant及GEMM2保持。

源码CPU检查1036项边界，实际ISA有限CFG检查43处各跳8条WMMA，不跨DS/TDM/barrier。
模型不覆盖全部VALU/VMEM/descriptor hazard，不能作为完整硬件正确性证明。
compile-only的ELF为`57bc48d547ed075d711118410820bd4bbcc66a6d94d9e876ea5fdeb562d0f58f`，
VGPR795/SGPR59/LDS327680B、private/spills0。seed0严格检查27项payload/ScaleA/
canary/same-pointer动态replay及完整MoE22020096B均零差异，原reference门限通过。

固定DS→M32→DS，各完整三轮；相同新boot、原benchmark、pin0、A preshuffle=1，
未改原计时helper。全部九轮rc0、原precision通过、whole-machine idle合格。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| DS before | 88.2528 /88.0441 /89.0584 | 88.2528 | 63.7341 |
| M32 candidate | 89.4103 /90.3138 /89.5031 | 89.5031 | 63.9786 |
| DS after | 88.0695 /90.3822 /88.7734 | 88.7734 | 64.3660 |

M32比两侧DS分别慢1.2503/0.7297 μs，即1.4167%/0.8220%；没有收益，
不进入生产、不追加反向实验。退出已恢复DS原bytes。actual GEMM1与各自CPU artifact
绑定，GEMM2完整ELF仍`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。

完整证据archive123项、621874B已下载，SHA256为
`e0727a777e82a260f29b3d42427d5cfd47a881630b69c6b8b52d7f94cb1d654f`。
本地重新核验成员SHA、原表格、空闲、actual ELF与中值，
`reboot20261021_m32_normal_v1.local_acceptance.json`为`passed=true`。

复现外层runner：

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_normal_m32_reboot_v1.py \
  NEW_M32_LABEL ds m32 ds
```

该历史runner锁定原A-preparation SHA；生产已同步正确release/acquire后，不再直接运行它。
后续候选需让控制与候选都使用相同正确A-preparation源码，不能为比较回退同步修复。
当前保留GEMM1仍只有DS计数修正，重启后独立最优验收中值88.5421 μs，尚未80 μs。

### M32 pair内WMMA顺序实验：正确性通过，性能待完成（2026-10-21）

私有`ds_pair_major_mma_v1.py`仅在原GEMM1/FP4/A preshuffle/pin0/
非persistent/t256/w2x2/b4 scope重排独立accumulator：同一个M32 pair内先交替
上下M16，再处理下一N16。所有有效accumulator保留原56次K128更新顺序，不加
reuse hint、不改变input issue/load、fence、epilogue或GEMM2。实际helper AST执行
24种geometry/row组合、53760次调用通过，关闭scope时原序列一致。

candidate source SHA为`2882580ede6aec94c29ff655ef2d9d2c7049878d5ad4219ec55937822a602025`；
compile-only ELF为`457b4722f751e80bff745460b084d2f19c499912a617f89ad825ca23b66151e9`，
VGPR762/SGPR58/LDS327680B、private/spills0，与DS资源相同。

复用M32 skip有限CFG模型时，候选和已经验证的DS控制都报告了
`DS overwrite before WMMA retirement`。两份原失败JSON和rc1均保留。
该模型把`VA_VDST`计数用于所有WMMA输入的完整退休，未实现coexecution时序；
不能把模型告警直接认定为新增race，也不能把无告警当硬件正确性证明。
Shader Guide rev65 §4.3.7.4.1 p92规定SCHED_MODE2的同步责任，
§4.6.12.1 pp148–149的V_NOP表针对VALU写；不能未经核实套用到DS写。
本地LLVM的`GCNHazardRecognizer::checkWMMACoexecutionHazards`也单独处理
WMMA/VALU，未将DS归入这个coexecution表。

差分检查确认原WMMA/DS/TDM数量、DS offsets、operand宽度、scale selector数量、
barrier与资源相同，accumulator不与matrix A/B重叠；Scalar ScaleA和tail Matrix B
的实际register allocation改变，未声称物理operand逐项相同。
`audit_pair_major_differential_v1.acceptance.json`明确记录这些限制。

seed0严格验证`ds_pair_major_strict_e64_s0_v1`在120秒内rc0：27项有效payload/
ScaleA、canary、同pointer动态graph replay检查全部零差异；完整MoE22020096B
零差异，`logits_diff=3.6105513278025183e-06 < 0.01`。actual G1绑定上述ELF，
G2仍为`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。

新正常入口runner `run_normal_pair_major_v1.py`固定当前正确A-preparation SHA
`6fcf74235c7d8cf002cc77527aafbb13e99f187bb84284ababb47c3fb01ec5d9`，
原benchmark/warmup5/iters20不改，按DS→pairmajor→DS各完整三轮对照。
本节暂不声称性能收益；完成结果随后追加，候选尚未进入生产。

### WMMA pair顺序正常入口对照完成：回退，放弃（2026-10-21）

`reboot20261021_pairmajor_normal_v1`全部九轮使用同一正确A-preparation源码，
原单测/warmup5/iters20、A preshuffle1/pin0、uniform/auto/seed0不变。
每轮pre/post/settled全机GPU空闲，原reference通过，actual G1/G2 ELF绑定通过。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| DS before | 88.3479 /88.5947 /88.8687 | 88.5947 | 64.4997 |
| pair-major candidate | 89.3621 /89.1597 /89.3558 | 89.3558 | 64.1053 |
| DS after | 88.4260 /88.8434 /88.8138 | 88.8138 | 64.0105 |

candidate比两侧DS分别慢0.7611/0.5420 μs，约0.8591%/0.6103%。
该顺序没有收益，不进入生产、不追加反向实验。生产DS原bytes已恢复并SHA核验，
GEMM2未改；完整正确性通过不意味着本次调度顺序更快。

139成员、654860B证据archive已下载并独立复算所有原表格、空闲记录、实际ELF、
严格byte检查和三轮中值，SHA256为
`79b8955bb9f908b6e438952e29011b3600ea17df647099c12f83e7d9e4b583af`。
本地`reboot20261021_pairmajor_normal_v1.local_acceptance.json`为`passed=true`。
复现使用新label：

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_normal_pair_major_v1.py \
  NEW_PAIRMAJOR_LABEL ds pairmajor ds
```

接下来使用当前DS production抓取本分支pin0/A preshuffle1的thread trace，
不套用之前persistent/w2x4版本的phase占比。Method B脚本从
`../aiter_dsr1_e65/my_code/get_att_b_rocm_sdk.sh`复制回本分支，
每个profile和analysis child限时120秒、kill-after5秒，失败SIMD后停止，
保留任务私有cache用于actual ELF核验。kernel源码没有因此修改。

### 当前DS/pin0的thread trace与行组批量activation实验（2026-10-22）

当前分支的DS production已完成Method B四个SIMD-select的GEMM1 trace与本地证据验收。
64个active waves各执行1792条WMMA，即56个K128步骤×32个accumulators，每WG只有
一个N-task。prologue、mainloop、activation/quant与output barrier分别占累计wave span
13.3088%、71.9456%、11.9948%；这些占比不等于global critical path或可移除收益。
完整分析见`moe_gemm1_hyg_dsr1_pin0_thread_trace_analysis.md`。

新私有候选`ds_quant_row_batch_v1.py`在同一个有效M16 group内将两个MX block的
16对gate/up合为32对调用原`batched_silu_swiglu`，CHUNK仍为4。然后按原16值切片，
分别执行原F32 amax/E8M0、BF16 RNE、native FP4及输出。不扩大shape准入，不改shared
helper、GEMM2、input、计算分工或hardware fences；只作用于原SiLU/fused quant/FP4/
A preshuffle/pin0/非persistent/t256/w2x2/b4配置。没有引入OPTX pipeline。

实际新旧gather/helper AST逐符号执行64组通过，103组guard检查通过，移除patch后
完整module AST相同；每个scale block值、乘法结合及tanh调用顺序保持。
candidate SHA为`32bb7d452b6721d4ab7140a839f703d6b9d8c922135577dd788db47d792c9dc9`。

容器内compile-only rc0，实际ELF为
`ec12d9658ed13ba92e07d5d7bdb9f989b8dd11bb00e0421092d4603279972d40`。
VGPR762/SGPR58/LDS327680B、private与spills0，与DS保持；5879→5782条static
instructions，WMMA、tanh、BF16/FP4转换及DS/TDM/output数量保持。
ordered protected fences相同，mainloop native文本并非逐项相同，不能声称编译只改变
epilogue。该有限审计也不等于完整SCHED_MODE2 hazard证明。

首次ISA inspector误在主机执行，主机LLVM缺`librocm_sysdeps_zstd.so.1`，rc127。
compile ELF已成功生成，随后在`dsr1_ep4`内检查同一份ELF成功；没有重建工具链、
重编译或采用失败检查结果。runner已改为在容器内inspection，原失败日志保留。

`ds_quant_row_batch_strict_e64_s0_v1`在120秒内rc0，27项有效payload/ScaleA、
canary、同pointer动态graph检查零字节差异，完整MoE22020096B零差异。
logits_diff=`3.6105513278025183e-06`、rel_l2=`0.0026872099842876196`，原0.01门限保持。
actual G1绑定上述ELF，G2仍为
`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。

首轮`reboot20261021_quantrow_normal_v1`使用正确A-preparation和原测试脚本、
uniform/auto/seed0/warmup5/iters20，每轮pre/post/settled全机idle及actual ELF验收通过：

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| DS before | 89.3855 /88.6226 /88.9913 | 88.9913 | 64.2257 |
| row-batch candidate | 88.4494 /88.2006 /89.1557 | 88.4494 | 64.3121 |
| DS after | 88.4917 /88.8309 /88.4643 | 88.4917 | 63.8905 |

相对两侧控制分别低0.5419/0.0423 μs，约0.6089%/0.0478%；后侧差距非常小，
没有据此认定稳定收益。按事前规则继续候选→DS→候选反向完整三轮对照，
ownership不合格整组排除，不拼快样本。候选仍未进入生产，独立已验收的DS最优
88.5421 μs暂时保持。

复现使用新label，主机runner自行等待全机GPU空闲，每个GPU child限时120秒：

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_normal_quant_row_batch_v1.py \
  NEW_QUANTROW_LABEL ds rowbatch ds
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_normal_quant_row_batch_v1.py \
  NEW_QUANTROW_REVERSE_LABEL rowbatch ds rowbatch
```

反向`reboot20261021_quantrow_reverse_v1`已完成（27次运行，含四个被ownership排除的
完整组三轮）；这些排除组中值均为null，原始记录保留，不挑选其中耗时。
合格组为：

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| row-batch before，try2 | 88.5844 /88.2048 /88.8711 | 88.5844 | 64.3564 |
| DS middle，try4 | 89.2505 /88.7909 /88.3629 | 88.7909 | 64.4133 |
| row-batch after，try1 | 88.9679 /89.2590 /88.4285 | 88.9679 | 63.9473 |

前段候选比控制快0.2326%，后段慢0.1993%。未获得两侧一致收益，放弃该调度候选，
不进入生产、不继续据单个快样本追加计时。所有G1/G2实际ELF与strict精度绑定通过，
退出恢复精确DS source SHA。36次正反向运行的原reference全部正常，超时门限保持。

两个完成archive已下载并从原表格、ownership与实际ELF独立复算，本地acceptance
均`passed=true`：

- forward：135成员、647284B，SHA256
  `230a51b0e439665d18b77fcbcd5c3e37356701818ca7a48c3c6e77ec368a8970`。
- reverse：267成员、1119573B，SHA256
  `6c9dc1ab940d543984a8025a34a92cec3a54597a177cf2a4435b77045d7b7cce`。

### FP4 payload的halfword LDS store合并候选（2026-10-22）

新的`ds_quant_payload_b64_v1.py`只在相同SiLU/fused quant/FP4/A preshuffle/pin0/
非persistent/t256/w2x2/b4 scope，将kgrp0/1各两次halfword输出合并。
原nativeFP4 word保持；`shuffle_xor(16)`取得peer word，由kgrp0合并成两个dword，
用一次8B对齐的`lds_store_b64`写出，kgrp1不再重复halfword写。
SiLU、F32 amax、E8M0、BF16 rounding/nativeFP4、ScaleA、output TDM和所有
hardware fences保持，shared helper及GEMM2未改。没有改变有效数据或计算分工。

CPU使用实际`quant_byte_offset` AST验证整tile16384个payload字节、8192个旧
halfword地址，128组packet及64组任意word bit pattern全一致；包括算术右移的符号位
由`&0xFFFF`隔离，原上下两halfword保持。移除patch后完整module AST相同。
candidate source SHA为
`5fdc344f1956ed8eaa7023a0f35a4a0faa29771a9f5b1118eaaab73796ae5cae`。

容器compile rc0、未超过120秒，actual ELF为
`c22edb548e8270335be4bfdd47e638170b54586de45353b88c7a01a8aafe0fa0`。
VGPR762/SGPR58/LDS327680B、private/spills0，与DS相同。
static payload store从64条halfword变成32条b64，permlanex16从16条变成48条，
总static instructions从5879增至6280。input LDS load counts/offsets、protected
math/TDM/ScaleA/fences保持，mainloop文本并非逐项相同。
静态条数增加不能唯一判断性能，先执行严格device等价验证，再原入口固定三轮对照。
`ds_quant_payload_b64_strict_e64_s0_v1`在120秒内rc0，27项有效payload/ScaleA、
canary、同pointer动态graph检查全部零差异，完整MoE22020096B零差异。
logits_diff=`3.6105513278025183e-06`、rel_l2=`0.0026872099842876196`，
原0.01门限保持；actual G1绑定上述ELF，G2仍为原`4bd626e…716be`。

固定DS→payload64→DS的正常入口对照已完成。每组完整三轮，原benchmark、
uniform/auto/seed0/warmup5/iters20、正确A-preparation及A preshuffle1/pin0保持。
后侧控制try1和try2各有ownership不合格轮，整组三轮排除、中值null，原记录保留；
try3完整合格，没有跨组拼接快速样本。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| DS before，try1 | 87.8778 /87.9581 /88.1415 | 87.9581 | 63.6838 |
| payload64，try1 | 94.5489 /93.6045 /94.1758 | 94.1758 | 64.2806 |
| DS after，try3 | 88.2807 /88.5675 /88.6286 | 88.5675 | 64.0928 |

候选比两侧控制分别慢6.2177/5.6083 μs，即7.0689%/6.3322%。没有收益，
放弃，不追加反向计时、不进入生产。全部15次运行原reference通过，queue退出
恢复精确DS源码SHA，actual GEMM2完整ELF不变。本批87.9581 μs是控制中值，
不替代独立验收最优88.5421 μs；也不把候选精度通过误作性能提升。

200成员、885869B证据archive已下载，SHA256为
`165c8cf4347315cb2a8c3ef93264c5b3ae4c65adebe4209d8bd326338c74fec5`。
本地从原表格、pre/post/settled ownership、source绑定、actual ELF与strict输出
独立验收，`reboot20261021_payload64_normal_v1.local_acceptance.json`为`passed=true`。
没有下载`.out`或private cache。

复现外层runner（主机、使用新label，内部执行原容器命令与120秒门限）：

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_normal_quant_payload_b64_v1.py \
  NEW_PAYLOAD64_LABEL ds payload64 ds
```

硬件依据：Shader Guide rev65 §4.7.1（p160）给出64个4B LDS banks，
§5.7.6（p235）给出Wave32 LDS store吞吐受VGPR read bank conflicts等条件影响。
CDNA5 ISA §15.14 `V_PERMLANEX16_B32`（p581）定义跨两个16-lane rows的gather。
这些资料支持设计检查，不能证明更宽store必然加速本kernel；实际编译将
`shuffle_xor(16)`lower为`v_permlanex16_b32`。

### Wave并行expert查找：严格等价通过，正常入口对照中（2026-10-22）

复查当前DS trace的prologue，原ISA在首个input TDM前有9个table load sites，
包括tail检查、串行bisect和最终expert row end。新独立候选
`ds_wave_expert_lookup_v1.py`仅把原GEMM1小expert表的bisect替换为两组Wave32
连续读取，每组比较`psum[expert_lane] <= blk_m`，再ballot/popcount求upper_bound。
原`has_work`早退保持，padding、重复end/empty expert和运行时`psum`均不改变。
不会根据随机seed或tensor值选择特例；更大expert表继续原bisect。

scope为原SiLU/fused quant/FP4/A preshuffle/pin0/非persistent/t256/w2x2/b4、
`n_experts <= 64`且非EP scatter，没有改变wrapper、shape准入或GEMM2。
实际新旧lookup AST执行74808项，涵盖1/2/15/31/32/33/63/64/65/128/256个expert、
重复end、全空表、边界前后和padding；所有读index有界、upper_bound和原bisect一致。
210项scope guard检查通过，移除patch后完整module AST相同。
candidate source SHA为
`1d0b195df9e60bd1cddaf61ec58828c7f829ad2be62c45c13b06069bf73f9ad8`。

容器compile-only在120秒内rc0，完整ELF为
`111b9df5896ca2caee28c95ece00f60246566f3ad0f106c3eacdd5c4ff7d03c4`。
正确解析的static instructions为5879→5838，VGPR762/SGPR58/LDS327680B、
zero spills/private保持。首个input TDM前load sites为9→4，新增两条
`global_load_b32`分别读取前后32个end，native保留`s_wait_loadcnt 1/0`，
两条`s_bcnt1_i32_b32`之后进入原row-end读取。
input DS offsets、WMMA/tanh/BF16/nativeFP4/ScaleA/payload/TDM数量和protected
hardware fences保持；mainloop register allocation并非逐项相同，不声称只改prologue。
这仍是有限native检查，不能替代完整SCHED_MODE2 hazard证明。

`ds_wave_expert_lookup_strict_e64_s0_v1`在120秒内rc0，27项有效payload/ScaleA、
canary、同pointer动态A/ScaleA/psum graph replay与完整MoE22020096B均零差异。
logits_diff=`3.6105513278025183e-06`、rel_l2=`0.0026872099842876196`，
原0.01门限保持；actual G1绑定上述ELF，G2完整ELF仍为原版。
随后启动原正常入口DS→wavelookup→DS完整三轮、两侧控制，尚未认定性能收益，
生产保留版仍为DS。

复现（主机、新label，原容器benchmark及全机空闲门禁由runner执行）：

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_normal_wave_expert_lookup_v1.py \
  NEW_WAVELOOKUP_LABEL ds wavelookup ds
```

硬件依据：CDNA5 ISA §14.3.3记载vector comparison写VCC/SGPR，
§15.3 p244的`S_BCNT1_I32_B32`按32个bits统计1的数量。实际native的
两次global load及completion wait需按实际ISA验收；并行比较减少依赖链，
不能由此推断global critical path或kernel μs的改善。

完整正常入口对照已结束。前侧try1及后侧try1的ownership不合格，整组三轮中值
null，保留原日志；下表均为完整合格组，不拼接单轮。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| DS before，try2 | 88.8014 /88.6877 /88.0952 | 88.6877 | 64.2382 |
| wave lookup，try1 | 88.8054 /88.4663 /88.7382 | 88.7382 | 63.7913 |
| DS after，try2 | 88.8983 /88.3122 /88.1837 | 88.3122 | 64.2594 |

候选比两侧DS分别慢0.0505/0.4260 μs，即0.0569%/0.4824%。没有收益，
放弃，不追加反向性能或进入生产。全部15次运行原reference通过，退出恢复精确
DS源码；GEMM2实际完整ELF保持原版。并行upper_bound改变了表格load依赖结构，
本次测量没有证明它能改善整体GEMM1。

198成员、882350B证据archive已下载并独立复算原表格、ownership、实际ELF和
完整strict输出，SHA256为
`df8030763a3cdb495577ed216271c3cb659f1361b37881941ca6166d691d867c`。
`reboot20261021_wavelookup_normal_v1.local_acceptance.json`为`passed=true`。

此后仅在已保留DS_B=4版本重新检查MMA分组hints。前面新commit的参数扫描
使用原DS_B=2；本次不能复用其candidate ELF或性能作为DS_B=4的新组合结果。

### DS_B=4 上重新扫描 MMA 分组与 B-TDM TH（2026-10-22）

正式保留版仍是新boot独立验收中值88.5421 μs的DS计数修正版。新的固定编译点为
`8/8/1` identity control，以及`6/8/1`、`8/6/1`、`8/9/1`、`9/8/1`、`9/9/1`、
`10/10/1`、`12/12/1`、`16/16/1`、`11/13/1`、`8/8/0`、`8/8/4`、`8/8/6`。
三项依次为`mma_group/mma_first_group/tdm_b_th`。只在原DS GEMM1 guard内
使用固定值，没有用公共env改变GEMM2；launcher参数、dispatch、shape准入、
有效计算、wave布局、原四级K pipeline和hardware fences均未修改。

13个点的CPU compile-only及actual ELF/ISA检查均完成，所有child在120秒内退出。
默认`8/8/1`的完整ELF与DS保留版完全相同。保护的DS/TDM/输出数量与offset、
ordered fences及WMMA/BF16/nativeFP4/tanh数量保持；LDS327680B、SGPR58、
private/spills0。VGPR如下：

| group/first/TH | VGPR | 相对DS的资源检查 |
|---|---:|---|
| 6/8/1、8/6/1、10/10/1 | 757 | 减少5 |
| 12/12/1、16/16/1、11/13/1 | 765 | 增加3 |
| 其余点 | 762 | 相同 |

后三点的`native_acceptance.passed=false`只因`VGPR_no_increase=false`，没有隐藏
该失败项。strict controller按事前规则允许小幅register增加；protected操作、
数学和同步检查仍必须通过。这些有限native检查不等于完整SCHED_MODE2 hazard证明。

12个非默认点的seed0严格device验证全部完成：各27项有效payload/ScaleA、
canary及同pointer动态A/ScaleA/psum graph replay零字节差异，完整MoE22020096B
零差异。原reference为logits_diff=`3.6105513278025183e-06`、
rel_l2=`0.0026872099842876196`，原0.01门限保持；各actual G1完整ELF与对应CPU
artifact绑定，G2保持`4bd626e…716be`。候选尚未进入生产。

完整性能序列是DS与12个候选交替的25组，每组完整三轮，使用本文原正常入口和
正确BF16 A-preparation。ownership不合格整组排除、中值null，按6秒间隔等待全机
空闲后重新完成整组三轮；不按耗时挑轮。所有原始失败/排除记录保留。
`reboot20261021_dsparams_forward_v1`已完整结束：25组合格三轮、另24组ownership
不合格整组排除，共147次原入口运行，原reference均正常，实际GEMM2 ELF保持。
退出恢复精确DS源码SHA。1852成员、8550764B原证据archive已下载，本地复算
原表格、pre/post/settled、source、actual ELF、全部strict输出和中值通过，
`local_acceptance.passed=true`，archive SHA256为
`916fce77da479fc7906f19f565f550f2cedfc3223b6763ca97e6ba672d0d4334`。

| group/first/TH | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs | 两侧DS中值 μs | 相对两侧改善 |
|---|---|---:|---:|---|---|
| 6/8/1 | 87.4163 /87.8526 /88.8647 | 87.8526 | 64.4409 | 88.5378 /88.5696 | +0.7739% /+0.8095% |
| 8/6/1 | 88.6326 /88.8416 /88.5295 | 88.6326 | 64.0171 | 88.5696 /88.3249 | -0.0711% /-0.3484% |
| 8/9/1 | 88.5378 /88.3607 /88.0952 | 88.3607 | 64.0741 | 88.3249 /88.6770 | -0.0405% /+0.3567% |
| 9/8/1 | 88.0045 /88.3565 /90.3428 | 88.3565 | 63.9389 | 88.6770 /88.6245 | +0.3614% /+0.3024% |
| 9/9/1 | 87.5279 /88.0952 /87.7727 | 87.7727 | 64.7485 | 88.6245 /88.3755 | +0.9611% /+0.6821% |
| 10/10/1 | 90.7540 /90.4187 /89.3579 | 90.4187 | 64.0636 | 88.3755 /88.5064 | -2.3120% /-2.1606% |
| 12/12/1 | 90.2374 /90.7054 /91.3527 | 90.7054 | 63.7534 | 88.5064 /88.7783 | -2.4846% /-2.1707% |
| 16/16/1 | 91.6182 /90.7243 /91.7638 | 91.6182 | 63.7978 | 88.7783 /88.1814 | -3.1989% /-3.8974% |
| 11/13/1 | 90.3893 /90.5812 /90.6188 | 90.5812 | 63.8843 | 88.1814 /88.6960 | -2.7214% /-2.1255% |
| 8/8/0 | 89.8851 /89.1345 /90.1699 | 89.8851 | 64.2280 | 88.6960 /88.9976 | -1.3406% /-0.9972% |
| 8/8/4 | 88.3417 /88.6077 /88.8607 | 88.6077 | 64.0949 | 88.9976 /88.4346 | +0.4381% /-0.1957% |
| 8/8/6 | 88.3252 /88.8015 /88.6244 | 88.6244 | 64.0189 | 88.4346 /88.3588 | -0.2146% /-0.3006% |

6/8/1、9/8/1、9/9/1有正向两侧一致改善，分别继续候选→DS→候选完整反向三轮。
其余点无两侧一致收益，不追加无依据的计时。87.7727 μs只是9/9/1的正向候选中值，
尚未作为新生产最优；正式保留纪录仍为88.5421 μs，不以本批更快控制替换该纪录。
证据封装第一次因缺少仅本地的`accept_small_normal_evidence_v1.py`而停止，
补传该验收工具后原证据封装成功；没有重跑benchmark或修改原日志。

`6/8/1`反向完整对照已完成，中间控制try1的ownership不合格整组排除，try2
重新完整合格；原记录保留，不拼轮。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 6/8/1 before | 87.8147 /88.2595 /87.7389 | 87.8147 | 63.3000 |
| DS middle，try2 | 87.7937 /89.0249 /89.0166 | 89.0166 | 63.8924 |
| 6/8/1 after | 88.4788 /87.9307 /88.6792 | 88.4788 | 64.4683 |

两侧候选比中间控制快1.3502%/0.6042%，与正向两侧一致。中间控制及GEMM2有
轮间波动，因此该数据不构成固定绝对耗时保证；没有从三个候选组中挑最快一个
作为正式生产纪录。下一步补seed17和outside-guard direct compile检查，其他
正向候选的反向对照尚未结束，生产仍为原DS保留版。

`reboot20261021_dsparams_reverse_g6f8_v1`的367成员、3259439B archive已下载，
SHA256=`7bf07f5e8c257d9bdf98345870065fcd9049886e5af98a333ea1d3e228c8f7b0`，
本地原表格、ownership、actual ELF、strict与source复算均通过；退出恢复DS源码。

`9/8/1`反向对照也已完成并下载独立验收。前候选try1～try3及后候选try1
ownership不合格，四个完整组排除、中值null；下表只使用重新完成的合格三轮。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 9/8/1 before，try4 | 87.6946 /88.2702 /88.2681 | 88.2681 | 64.1476 |
| DS middle，try1 | 88.2935 /89.0650 /88.5526 | 88.5526 | 63.9664 |
| 9/8/1 after，try2 | 88.3102 /87.8315 /87.9898 | 87.9898 | 64.7634 |

两侧相对中间DS改善0.3213%/0.6356%，正反向方向一致，尚未采用。
466成员、3613756B archive SHA256为
`1f7b98f0bff2bb7f6707d617617e49222140de49d784573514d2ffbd55094ba9`，
本地重新核算原表格、中值、pre/post/settled、实际ELF、strict和source均通过。
`comparison_report.json`在成功local acceptance之后从archive生成，未使用PTY
摘要代替原始计时证据。生产源码退出后恢复精确DS SHA。

`9/9/1`的反向队列已完成。后候选try1有ownership不合格轮，整组三轮排除，
try2重新完整合格，不能因其中89.5394 μs较慢而丢弃该有效轮。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 9/9/1 before，try1 | 87.7154 /88.2576 /88.5547 | 88.2576 | 64.4133 |
| DS middle，try1 | 88.7636 /88.4872 /88.3395 | 88.4872 | 63.9600 |
| 9/9/1 after，try2 | 89.5394 /88.1268 /88.1288 | 88.1288 | 64.2509 |

两侧相对中间DS改善0.2595%/0.4050%，与正向方向一致。367成员、3259769B
archive SHA256=`d0c0dee16562360982446e0f0971cb70873743f28751731e8a6ba0760f6a2a01`，
已下载并从原始计时、ownership、实际ELF和strict独立验收通过。

三个有收益点的正反向确认全部结束。优先验证`6/8/1`的最小默认patch：它的
四个相邻控制改善均为正，最小为0.6042%，VGPR为757；`9/8/1`和`9/9/1`
对应最小改善约0.30%/0.26%，VGPR为762。此选择依据局部对照一致性和较小patch，
不是取全部组中最快的单轮或中值；88.5421 μs的正式生产纪录尚未替换。

准备最小`ds_g6_default_v1`
候选：相同GEMM1 SiLU/fused quant/pin0 scope内把默认`mma_group`改为6，
`mma_first_group`、`ds_first_n`及`tdm_b_th`的原赋值保持，显式
`AITER_FLYDSL_MMA_GROUP`继续覆盖默认。该patch相对DS增加9行、删除1行，
130项实际guard、launcher ABI与移除patch后的完整AST通过；已传到a07-3并开始
CPU-only编译和scope检查，尚未dispatch或进入生产。先验证默认完整ELF是否等于
已确认`6/8/1`，再检查候选本身在原8个outside-guard配置、quant0和显式mg8
下与control完整ELF一致，不把固定实验参数的硬覆盖直接搬进正式源码。

最小patch的CPU-only验收已全部完成，所有child在120秒内退出：默认target完整
ELF=`5336bd13dc6f31649e8875f4aae7f70a7eb3b2b495135f738e7bf9631cb27a50`，
与固定`6/8/1`实测candidate完全相同。GEMM2、FP8输入、原始A、pin1、t64 fallback、
w1x4、b3、persistent1共8个direct compile及quant0、显式mg8两项均与对应control
完整ELF逐字节一致；不是通过dispatch回原GEMM2来隐藏候选本身的变化。
最小source SHA=`3d197eebb11aa9c76e867d9179d6266571aadadcc74fa31d45152d08a5bce044`。
随后补E64 seed17和E65 seed0/17原fallback严格验证；这些正确性运行不记录性能。

三项追加严格验证已完成，每项27个有效payload/ScaleA、canary、动态same-pointer
graph检查和完整MoE22020096B均零差异，child均在120秒内退出。E64 seed17的实际
GEMM1与上述G6 ELF相同，GEMM2保持原版。E65 seed0/17仍是原t64/w1x4 fallback，
GEMM1 ELF=`d173f0b236adfb2d61acb596dc06b2c46b1e02ea9d75aeae8016db902c18fff3`、
GEMM2 ELF=`3b63a9571dca9410100499f8eb3507d09f3b248494077ca4940d1b09f0abc5ec`，
与先前验收control相同；此移植没有把E65改为E64的t256 kernel，也没有新增dispatch
specialization。E65这里是65列random gating/topk9，未额外加入固定shared route。
随后启动最小source的原正常入口DS→g6default→DS，每组完整三轮。

最小source的最终原入口对照全部完成，9次运行均原reference通过、全机空闲
检查合格、actual GEMM2 ELF保持；没有ownership排除组。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| DS before | 88.9173 /88.5083 /88.3985 | 88.5083 | 64.2910 |
| g6default | 88.5316 /87.7241 /88.6854 | 88.5316 | 63.9956 |
| DS after | 88.1098 /88.6497 /89.0122 | 88.6497 | 63.7533 |

候选相对两侧改善为-0.0263%/+0.1332%，没有两侧一致收益。虽然固定`6/8/1`
前面正反向表现较好，且最小source默认ELF完全相同，本次最终独立对照没有
稳定复现该收益，不能由ELF等价或前面较快组替代此次验收。因此不进入生产，
没有执行准备的adoption工具、不继续救援计时；保留DS源SHA及正式88.5421 μs
纪录，转向独立小feature。87.7241 μs只是单轮，不用于更新正式最优。

283成员、1197842B archive已下载，SHA256为
`75c166a4ea6485754b031e6d9f6596e14209383b394f3e3367ab4e744be589b9`。
本地从原表格、pre/post/settled、source/actual ELF、10项scope及seed0/17完整
device证据独立验收；`local_acceptance.passed=true`指证据和正确性通过，
不表示性能优化已采用。`comparison_report.json`明确记录一侧负改善。
复现使用新label：

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_normal_ds_g6_default_v1.py \
  NEW_G6DEFAULT_LABEL ds g6default ds
```

复现需使用新label，以下命令在主机执行，runner在`dsr1_ep4`运行原benchmark：

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_normal_ds_parameters_v1.py \
  NEW_DSPARAMS_LABEL \
  ds g6_f8_th1 ds g8_f6_th1 ds g8_f9_th1 ds g9_f8_th1 \
  ds g9_f9_th1 ds g10_f10_th1 ds g12_f12_th1 ds g16_f16_th1 \
  ds g11_f13_th1 ds g8_f8_th0 ds g8_f8_th4 ds g8_f8_th6 ds
```

同时本地准备独立9行`ds_xdl_queue_v1`候选，仅在原SiLU/fused quant/FP4/
A preshuffle/pin0/非persistent/t256/w2x2/b4 scope写入
`HW_REG_WAVE_SCHED_MODE[2]=1`。未改变原mode2、计算、load/store或fences；120项
实际guard检查及移除patch后的完整module AST一致。它尚未上传、编译或dispatch，
不会与当前性能队列并发。Shader Guide rev65 §4.3.7.4.2（p93）记载该bit可能允许
多个WMMA排队，但也可能限制其他wave的coexecution；不能预先认定有收益。
旧E65和pipeline130的不同kernel已尝试该方向且没有稳定收益，本候选只作为当前
实际DS/pin0 kernel的独立实验，必须重新通过actual ISA及严格device/性能验收。

### XDL queue bit：编译诊断和初步精度（2026-10-22）

`ds_xdl_queue_v1`已上传，但最初CPU compile和附带35秒周期stack的诊断分别达到
120秒上限，exit均为124；没有artifact或GPU dispatch。timeout之后主机进程检查无
本任务child残留。35、70、105秒的stack均落在FlyDSL Python frontend的
`capture_user_location → inspect.getframeinfo → inspect._get_code_position`，不同采样
已经推进到mainloop、drain和epilogue，因此不据此声称LLVM hang或kernel race。

容器Python3.12的`_get_code_position`每次从头迭代`code.co_positions()`。
私有compile wrapper在进程内缓存该表，仍返回相同line/column tuple、每个code object
首次live调用与原函数核对；没有修改安装文件、关闭source location或更改compiler参数。
首次cache wrapper用CodeType作为`lru_cache` key，完整CPU compile用时**83.3326秒**，
exit0；65077次查询、79个code objects。后续v2改用`id(code)`及保留strong references，
避免重复计算较大CodeType的hash。该wrapper仅用于编译准备/正确性，输出耗时不作
benchmark性能证据；正式计时仍使用未修改的用户入口。

actual GEMM1完整ELF为
`299d32d4c420a74c1054a1d6ceddf59e37bf458c7e7fc9abb8d7d98b4767088e`，
`.text`为`1ff65a54faa4872283c103802f674db27c464da80124e986b891d7400526d966`。
VGPR762、SGPR58、LDS327680 B、private/spills0，全部与DS一致；5879→5880条executable
instructions，只新增一个documented bit write。protected memory操作counts/offsets、
主要ordered fences、WMMA/BF16/FP4/tanh counts保持，194条`s_wait_alu`保持。
这些是有限native检查，不能替代完整SCHED_MODE2 hazard证明或GPU严格验证。

E64/seed0严格验证已完成，85.0551秒、exit0：27项有效payload/ScaleA、canary、同pointer
动态A/ScaleA/psum graph及完整MoE22020096 B均零字节差异；actual G1绑定上述CPU ELF，
GEMM2保持`4bd626e…716be`。原reference的logits_diff为
`3.6105513278025183e-06`、rel_l2为`0.0026872099842876196`，原0.01门限保持。
随后补E64 seed17、E65原fallback seed0/17及candidate自身outside-guard direct compile；
尚无该候选正式性能数据，生产仍是DS、正式纪录仍88.5421 μs。

工具为`compile_cached_positions_v1.py`、`compile_cached_positions_v2.py`、
`run_script_cached_positions_v2.py`、`run_xdl_queue_validation_v1.py`，均位于本文task tools。
用户进度更新曾把83.33秒误写为6.39秒，随后已明确更正；采用原log中的83.3326秒。

完整逐条ISA比较进一步通过：去除新增bit write之后，所有decoded操作、寄存器operand、
ALU waits完全相同；新增8 B导致一条relative branch immediate重定位，所有branch targets
对应相同instruction index。首次脚本漏了16-bit negative displacement的sign extension而
在输出前KeyError退出，修正后完成，未采用失败输出。记录为
`ds_xdl_queue_v1.exact_isa_acceptance.json`；这不独立证明原baseline全部硬件hazards安全。

E64/seed0、seed17和E65/seed0、seed17四项严格验证均完成，27项有效字节、动态graph及
完整MoE全部零差异，原reference通过。E65继续原t64/w1x4 fallback，实际G1/G2 ELF与
前面control相同；没有声称该分支E65和E64用同一t256 specialization。8项candidate自身
direct compile（GEMM2、FP8、原A、pin1、t64、w1x4、b3、persistent1）与已验收control
完整ELF逐字节相同，全部child120秒内结束。汇总为
`ds_xdl_queue_validation_v1.acceptance.json`。

开始固定原入口DS→xdlqueue→DS、各完整三轮。为遵守单child120秒上限，先在独立无计时
进程准备每个mode的私有正常入口cache，再退出该wrapper、执行全机idle门禁，重新启动
未修改的用户命令计时；编译准备输出的μs全部丢弃，正式每轮完成后审计实际cache ELF。
原warmup5、iters20、seed0、uniform/auto与precision gate不改，不并发编译/监控/传输。
正式成绩仍88.5421 μs；复现使用新label，在主机执行：

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_normal_xdl_queue_v2.py \
  NEW_XDL_QUEUE_LABEL ds xdlqueue ds
```

### XDL queue 正向原入口对照（2026-10-22）

`run_normal_xdl_queue_v1.py`首次完成DS的独立编译准备后，因局部变量`wrapper`
覆盖原wrapper `Path`，在source检查处报`AttributeError: 'dict' object has no attribute 'read_bytes'`。
没有产生正式计时，finally恢复DS源码。v2只将该准备record变量改名为`position_record`，
保留原入口、timeout、三轮组、ownership排除规则和精度门槛；失败证据仍保留。

v2正向DS→xdlqueue→DS已完成，各采用第一个完整合格三轮组，不按速度挑轮。

| 组 | 合格try | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---:|---|---:|---:|
| DS before | 2 | 88.4113 /88.5886 /88.7255 | 88.5886 | 64.1835 |
| xdlqueue | 3 | 87.1969 /87.2435 /86.1258 | 87.1969 | 64.0380 |
| DS after | 3 | 88.6601 /87.8041 /88.3797 | 88.3797 | 63.8633 |

候选相对两侧改善1.570970%/1.338316%。候选的87.1969 μs距80 μs仍差7.1969 μs，
还需降低约8.2536%。这只是正向候选结果，尚未接入生产，不将86.1258 μs单轮用作新纪录。
GEMM2实际ELF仍为原版；约0.175 μs的后侧耗时差异不是GEMM2源码或编译变化。

完整排除组为DS before try1、xdlqueue try1/2、DS after try1/2。原因包括post
未识别已退出KFD entry及793145344 B VRAM占用，即使随后settled恢复idle，仍按原规则
整组排除，中值为null，不从这些组拼有效三轮。全部测试退出，无本任务GPU child残留。

467个regular members、1752483 B证据包已下载，SHA256为
`e0ad0f64ac32876e214448def4a5ab4a289e873d2493bf097017ef5566da98bb`。
本地`accept_xdl_queue_evidence_v1.py`逐member SHA、原precision表、pre/post/settled、
actual ELF、4项strict及8项candidate自身direct scope独立核验通过。
`local_acceptance.passed=true`不表示已经采用生产优化。反向固定xdlqueue→DS→xdlqueue
各完整三轮已启动；这次确认完成前正式生产纪录仍为88.5421 μs。

```bash
# 在主机运行；runner在dsr1_ep4容器启动未修改的原benchmark。
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_normal_xdl_queue_v2.py \
  NEW_XDL_QUEUE_REVERSE_LABEL xdlqueue ds xdlqueue
```

下一独立小feature已在本地准备：`ds_mgroup8_v1.py`、`ds_mgroup32_v1.py`。
每个相对DS仅增加5行，只在原目标scope改`TILES_PER_GROUP=16`为8/32；150项guard、
87120个实际AST映射点通过，覆盖partial group、capacity tail及N-cluster peers。
每个capacity M/N tile仍恰好一次，cluster peers仍同M；完整module去除patch后与DS相同。
未改wave布局、persistent、pipeline、数值、load/store或同步。尚未上传、编译、GPU执行或测性能，
旧E65 group8的收益不能替代当前kernel的独立验收。

### XDL queue 反向确认及保留（2026-10-22）

固定反向xdlqueue→DS→xdlqueue完成，并回收335个regular members、1281511 B原始证据包，
SHA256=`81f476d276b454505e4bb2be91541a0c2056618b4e88345a17a0166de077ec30`。
本地逐member/原表/ownership/actual ELF及全部strict/scope独立验收通过。

| 组 | 合格try | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---:|---|---:|---:|
| xdlqueue before | 1 | 87.2896 /87.1253 /86.5666 | 87.1253 | 64.0612 |
| DS middle | 2 | 88.0361 /88.1162 /87.8021 | 88.0361 | 64.5988 |
| xdlqueue after | 1 | 87.0873 /87.2433 /86.9102 | 87.0873 | 63.9768 |

候选相对中间DS改善1.034576%/1.077740%；DS middle try1因post ownership不合格整组排除，
未按耗时挑选。这与正向两侧改善一致，共四个相邻对照均为正。三份candidate合格组中值
87.1969/87.1253/87.0873 μs全部保留；正式成绩使用首个正向验收组**87.1969 μs**，
不挑后续最快组或86.1258 μs单轮，也不把先前DS历史中值88.5421 μs用作本轮相邻控制。

通过`compare_xdl_queue_sequences_v1.py`及`adopt_ds_xdl_queue_v1.py`将已验收完整candidate
保留到本地和a07-3源码；没有改Git index、commit或push。相对DS只新增9行，相对本次HEAD
共13行新增、1行删除，包括原DS_B计数修改。A preshuffle1、pin0、非persistent、w2x2和
原pipeline保持；VGPR762、SGPR58、LDS327680 B、private/spills0保持，GEMM2完整ELF保持。
硬件依据限于Shader Guide §4.3.7.4.2 p93：允许多条WMMA排队可能降低该wave的issue阻塞，
同时可能减少其他wave的coexecution。完整ISA除这一bit write及对应relative branch重定位外
逐条一致，四项strict和八项outside-scope通过；本次μs收益不能唯一证明某个WMMA FIFO
或cache已经成为新的硬件瓶颈，也不独立证明原baseline全部SCHED_MODE2 hazards安全。

采用工具首次在任何写入前因Windows本地benchmark使用CRLF而SHA检查拒绝；没有修改源码
或跑GPU。工具修正为只在只读比较时归一化CRLF→LF，归一化SHA与远端完全一致，benchmark
文件本身未改变。生产kernel和本文都保持LF。adoption前后均以原source SHA检查拒绝并发覆盖，
仅恢复/替换本任务管理的文件。

接续在已保留XDL上生成`ds_xdl_mgroup8_v1.py`与`ds_xdl_mgroup32_v1.py`，SHA分别为
`1a49f9d427f5a385d533ee9fb83b4d571822fee7e33f1e2989e2f02f1ceeb17f`和
`de3e10100a1b95a99253f49ad180a086194aad2ab49c02d4026137c959279670`。
相对当前XDL仍只有5行M grouping guard，映射patch逐字节继承已审计DS候选，去除patch后
完整AST精确XDL。两个candidate已上传并校验SHA，`run_mgroup_cpu_v2.py xdl`已启动CPU-only
compile/native检查；尚未GPU dispatch或正式计时。大改动方向继续不添加。

### M-group8/32 编译准备及验证（2026-10-22）

首次`ds_xdl_mgroup8_v1_cpu`在35秒周期faulthandler输出时rc139退出，日志只有不完整
`enum.py` traceback，未生成artifact，没有GPU dispatch。不能据此证明kernel或LLVM fault。
停止该队列后，保留fatal traceback、移除周期sampling进行限定检查：已验收XDL control
23.7987秒完成，完整ELF与原版相同；M-group8则达到120秒timeout、rc124，已终止、无child残留。
因此没有把第一次异常唯一归因到周期sampling。

读取容器Python3.12原`inspect.getframeinfo`确认它在调用`_get_code_position`前还先读取
`frame.f_lineno`，然后用position tuple的line覆盖。新测试进程内`run_exact_frameinfo_v1.py`
保留code positions的line/column和原context0全部返回字段，只跳过已有精确position时的
重复initial `f_lineno`查询；每个code object首次live调用逐字段和positions对照原函数。
context非0、非frame及缺line继续原路径。没有关闭source location、修改安装文件或compiler
options；wrapper仅编译准备/正确性，所有wrapper耗时不作为GPU性能。

新XDL control18.1109秒完成，完整ELF仍逐字节相同。M-group8/32随后分别在110.1547/111.4134秒
CPU-only compile完成，各64604次context0 query、69个code object逐字段核对通过。
两项均在120秒内退出，VGPR762、SGPR58、LDS327680 B、private/spills0、5880条executable
instructions保持，protected操作及offsets、ordered fences、WMMA/BF16/FP4/tanh counts检查通过。
这种有限native检查不能代替完整device精度或硬件hazard证明。

| candidate | GEMM1完整ELF SHA256 | .text SHA256 |
|---|---|---|
| XDL + M-group8 | `78285471320a71be890d1db019adde153a3eaffd509f70ef5f990cb7e6b17ea2` | `80c4a185c31c9940ba6d5758f6dce784459e7114f103e1c9b7488ef801b4c99d` |
| XDL + M-group32 | `d6636a9a1e4d6cf74332be6f30d03fa13327edff96da0533a21c9ec0745e4655` | `5116bc3447de86bdd2cf1d70fc86c14d423dac066843342151ae40108ee8f527` |

开始M-group8的E64/E65 seed0/17 strict和candidate自身8项outside-scope direct compile，
每个child120秒内退出；顺序为group8完整验证、group32完整验证，再固定XDL→group8→XDL→group32→XDL
各完整三轮原入口对照。正常入口需独立untimed cache准备后退出wrapper、全机idle再运行原命令，
保持warmup5/iters20/seed0/uniform/auto；编译准备不参与性能比较。此时仍无M-group正式μs，
生产保持已验收XDL与87.1969 μs。

### M-group 实际 ISA 和首项 device 验证进展（2026-10-22）

下载两项CPU-only实际ELF和完整decoded ISA后，本地逐条对比当前XDL发现两项均只有3个
M/N映射立即数变化：instruction index55/88的` s_lshl_b32` shift由4改为3或5，index95的
`s_min_i32`上限由16改为8或32。5880条executable instructions中其余5877条逐项相同，
包括WMMA、全部memory ops、operand和waits；VGPR762/SGPR58/LDS327680B及zero spills/private保持。
这是本轮两份实际产物的结论，不单独证明原baseline全部hardware hazards安全。
本地复算结果为`logs/xdl_mgroup_exact_mapping_isa_v1.json`。

M-group8的E64/topk8和E65/topk9各seed0/17 strict四项已全部rc0：每项27条BYTE_CHECK为0，
完整MoE输出22020096B逐字节一致。E64 actual G1绑定上述group8 ELF，G2为原版；
E65仍为当前分支原t64/w1x4 fallback，两stage ELF与相应对照相同，没有把E65声称为同一t256 specialization。
candidate自身outside-guard direct compile已经完成GEMM2、FP8、非preshuffle A、pin1、
t64、w1x4、buffers3和persistent八项，完整ELF均与各自control一致。group8队列已完整退出，
`completed=true`；已按顺序启动group32的四项strict及八项direct scope，没有重复启动group8。
此处尚无M-group正式性能数据，没有采用候选或更新87.1969 μs成绩。

已准备`seal_mgroup_evidence_v1.py`/`accept_mgroup_evidence_v1.py`，将完成的原timing表、
pre/post/settled、source、actual ELF、四项strict、八项direct scope和native evidence按label收集，
不传`.out/.pt/.mlir`或private cache。acceptor独立复算median并在全部检查成功后才写完整acceptance，
包含上述三条mapping immediate的精确ISA对比；没有修改原benchmark或source location。

本地另准备`ds_xdl_bounded_lookup_v1.py`，仅在原目标GEMM1 guard中利用`has_work`已证明
`blk_m < tile_map[n_experts-1]`，将serial upper_bound的hi排除已知的最终endpoint。
E64从7轮bisect改为6轮；63,994项actual AST整数检查、143项guard检查通过，覆盖重复end、
空表、边界和更大expert表，outside-guard保持原lookup；2379个active E64静态case确认少1次表读。
source SHA=`0d3f4731cc73b21195da8a5eea1d4eb0f447121a9879291485b5c873d42eb151`。
首个builder在写入candidate之前因解析末尾无body的`for`片段报IndentationError；补足审计片段的
`pass`后重新完成检查，实际candidate没有该`pass`。该候选尚未上传、native compile、GPU验证或计时，
不将它的静态表读减少声称为性能提升；M-group完整对照仍优先。

### M-group 完整验证通过，开始原入口对照（2026-10-22）

M-group32已完成E64/topk8及E65/topk9各seed0/17四项strict，每项27个有效字节检查和
完整MoE输出22020096B均零差异，actual E64 GEMM1绑定对应candidate ELF，GEMM2保持原版。
它自身direct compile的GEMM2、FP8、非preshuffle A、pin1、t64、w1x4、buffers3和
persistent八项完整ELF全部与相应control相同；最终`completed=true`、没有error。
group8和group32验证队列均完整退出。E65继续分支原t64/w1x4 fallback，没有改变dispatch。

新label `reboot20261021_mgroup_normal_v1`在启动前确认不存在，采用事前固定顺序
XDL→M-group8→XDL→M-group32→XDL，各完整三轮。每个mode先独立无计时准备正常入口cache，
退出wrapper后重新等待全机idle，再运行未修改的用户benchmark；只接受正常入口性能。
ownership不合格仍整组排除，不以速度挑轮，每个child120秒TERM/5秒KILL保持。

```bash
# 在a07-3主机运行，runner在dsr1_ep4内启动原benchmark。
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_normal_mgroup_v1.py \
  NEW_MGROUP_LABEL base mgroup8 base mgroup32 base
```

启动时生产仍为已验收DS_B + XDL源码，正式成绩87.1969 μs。两候选只有三条mapping立即数
变化的实际ISA结论不等于已经提速；队列完整结束、下载及独立验收之前不采用候选。

后续bounded lookup候选及CPU/strict/normal适配工具已仅按任务所需文件上传并逐SHA256核对；
保持M-group优先，没有并发启动GPU或native compile。工具只复用已有完整检查流程，
原benchmark、installed FlyDSL、source locations与精度门限均未修改。

### M-group 正常入口编译准备超时及阶段拆分（2026-10-22）

`reboot20261021_mgroup_normal_v1`第一控制try1有一轮post ownership不合格，整组三轮排除。
try2完整合格，三轮GEMM1为86.4629/86.6169/86.8528 μs、中值86.6169 μs，GEMM2中值
64.0146 μs。这是本轮相邻控制数据，不替换正式采用时87.1969 μs的记录。

随后group8在正常入口的独立编译准备进程触发120秒timeout、rc124，队列立即停止，
没有group8正式计时，finally精确恢复XDL生产源码。原失败summary和日志已下载保存，
没有从不完整队列推出M-group收益。主机KFD检查没有本任务GPU进程残留。

检查该准备的私有cache确认GEMM1已经写入有效artifact，actual ELF SHA与已验证group8一致；
GEMM2只有lock、尚无pkl。因一次完整入口同时准备多个kernel而耗尽进程时间，改为
`run_mgroup_split_preparation_v1.py`，在生产module路径下分开CPU-only编译G1/G2，每个
child仍120秒TERM/5秒KILL，不降低正式测试计算量，也不改变benchmark。
group8的G1/G2分开在104.9801/104.0250秒完成，group32的G1/G2分别在105.2769/104.4680秒
完成，完整ELF分别精确对应candidate及原GEMM2。四项准备均已退出，record的
`completed=true/source_restored=true`，本地下载后逐项核对rc、actual ELF SHA与精确位置记录。
此处只记录准备过程，不将其输出或耗时作为性能结果。

检查实际cache发现CPU placeholder调用和正常入口的tensor signature产生不同key；
不能仅由完整ELF相同推断正常入口一定命中。新label `reboot20261021_mgroup_normal_v2`
仍采用原固定完整顺序，先正常入口无计时准备、退出wrapper后重新全机idle，正式计时使用
未修改的benchmark。尚未采用M-group或更改87.1969 μs正式成绩。

### 正常入口 cache ABI 审计修正（2026-10-22）

v2前三个控制组在post检查短暂出现`<已退出>` PID，随后settled均恢复空闲，仍按事前规则
整组三轮排除、中值null。首个合格控制try4为87.8148/86.6448/86.4739 μs，中值86.6448 μs。
group8正常入口准备完成，首轮原benchmark精度通过、原读数88.351 μs，但之后audit因每stage
同时存在CPU placeholder和正常入口两个cache signature而报错，队列停止并恢复精确生产源码。
这个单轮不作为正式候选成绩；旧summary、audit失败和raw日志均保存，不能拼成完整对照。

下载实际entry signature发现，placeholder的ScaleA/ScaleB rank3及stride1位置与正常入口不同；
正常入口ScaleB是rank1，ScaleA/out是rank3末维stride1。两个ABI的actual GPU ELF分别精确
同一候选G1和原G2，但ABI相同不能由ELF相等推断。`audit_normal_cache_signature_v2.py`
以已完整成功控制的entry signature作精确比较，要求每stage恰好一个匹配且原cache key不变，
再做完整ELF核对；额外CPU准备ABI逐项记录排除，不删除、改名或伪造cache artifact。
group8的两个正常entry分别是`13e21c457a09ffb0/c33b05449501ed1b`，此独立审计已通过。

`prepare_normal_stage_v1.py`复用原benchmark创建的真实inputs，在待准备的stage打开原
`COMPILE_ONLY`，完整编译后立即退出，不运行isolated timing或reference；准备G2时先执行
真实G1初始化输入。实际安装版检查确认`COMPILE_ONLY`不进入code-generation cache key。
`run_mgroup_real_stage_preparation_v1.py`只用于group32缺失的两个正常入口ABI，各child
仍120秒、全部准备μs丢弃，finally恢复生产。未修改安装文件、原benchmark或kernel数值路径。
group32逐stage准备已完成：G1/G2分别102.0333/100.7923秒，两个child均rc0，真实入口signature与完整ELF核对通过。
`completed=true/source_restored=true`，原生产SHA已恢复，主机检查无GPU进程。
随后启动新label `reboot20261021_mgroup_normal_v3`：XDL→group8→XDL→group32→XDL，各完整三轮，
使用v2精确ABI审计；编译准备数据不参加性能统计。正式仍87.1969 μs，待完整对照结束后独立验收。

### M-group 原入口正向对照完成（2026-10-22）

`reboot20261021_mgroup_normal_v3`完整退出，生产精确SHA恢复；5个合格完整三轮组如下。
group8、middle control和group32各有一组ownership不合格，整组中值null，未拼接样本。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| XDL before | 86.6929 /86.0013 /87.7406 | 86.6929 | 64.5926 |
| M-group8，try2 | 86.6908 /87.4900 /86.9544 | 86.9544 | 64.0273 |
| XDL middle，try2 | 87.2539 /86.7689 /86.6276 | 86.7689 | 63.9409 |
| M-group32，try2 | 86.6212 /86.7647 /86.5158 | 86.6212 | 64.0316 |
| XDL after | 86.0371 /87.2346 /87.2263 | 87.2263 | 63.9219 |

group8比两侧慢0.301639%/0.213786%，不保留，不追加反向计时。
group32相对两侧改善0.170222%/0.693713%，差距有限，尚不作为稳定收益或正式生产成绩。
按既定规则继续一次新label `reboot20261021_mgroup32_reverse_v1`，group32→XDL→group32，各完整三轮。

567成员、2332864 B完整证据已下载，archive SHA=
`7b34f7772feff38af76e5aac659837fdaf26ae0792baaf88b85e6d198a361b1a`。
本地从原计时表、pre/post/settled、候选完整ELF、四项strict、八项direct scope、真实ABI及原cache key
复核通过。首次v2 acceptor错误要求cache-hit运行也必须有`getframeinfo`调用，故拒绝，未发布acceptance。
原wrapper记录显示这些运行`frameinfo_calls=0/code_objects_frameinfo_checked=0`，实际使用的
`_get_code_position`有477次调用、11个code object检查。v3按实际调用分支检查：调用过frameinfo才要求
对应字段验证；未调用则要求相关计数均0，并保留实际code-position检查、全部精度、源码、ELF及ABI门槛。
原raw与v2工具未修改，未重跑GPU；v3独立验收最终`passed=true`。

后续bounded lookup的真实stage准备和v2精确ABI工具已准备、逐层展开检查并上传核SHA。
仍不与当前反向测试并发编译或GPU运行。另对TDM owner1/2/4的实际partition表达执行48项静态覆盖检查、
798720个source/destination element对应一致且无重叠，原每wave TDM计数匹配`TDM_PER`。
这些静态结果不证明native lowering/device正确性，也不作性能结论；owner2/4 GPU测试排在bounded lookup之后。

### M-group32反向确认、最小落地与新候选基准（2026-10-22）

`reboot20261021_mgroup32_reverse_v1`完整退出，生产临时swap已恢复，主机GPU/KFD无本项残留。
中间control的try1有两项post ownership不合格，整组三轮中值null；try2是首个完整合格组，没有拼轮。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| M-group32 before | 86.8005 /86.6191 /86.8594 | 86.8005 | 63.8208 |
| XDL middle，try2 | 88.0212 /87.2197 /87.6606 | 87.6606 | 64.0212 |
| M-group32 after | 86.4526 /86.5243 /86.5180 | 86.5180 | 63.7892 |

反向两项改善0.9812%/1.3034%；结合正向0.1702%/0.6937%，四个相邻控制全部有收益。
幅度有限，此处只据本次完整正反向对照保留，不由单轮最快数据推导跨机器稳定提升。
432成员、1848655 B archive已下载，SHA=`fb1d4c871f262c886072440ef3e2ede87ef1094235cb351a974da7c685ed450e`。
本地v3 acceptor逐项复算原表格、pre/post/settled、source、实际ELF、strict及原entry ABI通过。
`mgroup_sequences_v1.comparison.json`记录两序列完整结果，M-group8保持淘汰。

`adopt_ds_xdl_mgroup32_v1.py`检查两序列acceptance和精确源码后，在本地及a07-3原生产位置保留同一份
5行GEMM1 swizzle patch，`TILES_PER_GROUP`在原SiLU/fused quant/FP4/A preshuffle/pin0/非persistent/
t256/w2x2/b4/cluster1x4 scope由16改32。功能、原shape准入和launcher参数保持；GEMM2及8项outside-guard
完整ELF相同，E64/E65 seed0/17各27项BYTE_CHECK和完整MoE22020096 B零差异。E65仍为分支原fallback，
本项不表示实现E64/E65共用t256或固定shared1 route。

落地source SHA=`de3e10100a1b95a99253f49ad180a086194aad2ab49c02d4026137c959279670`；
actual G1 ELF=`d6636a9a1e4d6cf74332be6f30d03fa13327edff96da0533a21c9ec0745e4655`，
G2保持`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。
VGPR762/SGPR58/LDS327680 B/private0/spills0。正式成绩使用首个正向验收组86.6212 μs，
不取86.5180 μs替换；距目标6.6212 μs。Git index未改，没有commit/push或清理其他文件。

下一项改用新名称`ds_xdl_mgroup32_bounded_lookup_v1`，明确基于这份已保留source重新生成；旧bounded候选与工具证据保留。
新source SHA=`7a27da676ac3f6c23955028c4642ac586c903f5c44f1b9af1003d38844e424b4`；
重新执行63,994项actual AST lookup和143项guard检查通过。只在has_work已证明最终endpoint大于blk_m时
将E64二分查找7轮改6轮，数值计算、transfer和fences源码保持。已同步所需candidate和native工具，开始CPU-only编译；
尚无该新候选GPU或性能结果。所有child仍限120秒TERM/5秒KILL。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_normal_mgroup_v2.py \
  NEW_MGROUP_FORWARD_LABEL base mgroup8 base mgroup32 base
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_normal_mgroup_v2.py \
  NEW_MGROUP_REVERSE_LABEL mgroup32 base mgroup32
```

上面旧对照runner要求其冻结的XDL baseline源码；生产已是M-group32，不应直接对新生产启动旧runner。
正常性能复现仍用本文开头原用户benchmark命令，A preshuffle1/pin0、warmup5/iters20/seed0/uniform/auto保持。

### bounded lookup 的编译问题与当前验证状态（2026-10-22）

M-group32 落地后的 bounded lookup v1/v2 在 FlyDSL AST transform 阶段触发 120 秒 timeout，
所有 child 均 rc124 并已退出；finally 精确恢复生产 SHA，没有 GPU/KFD 残留，未采用任何性能数据。
单 module 导入仍不能消除长 constexpr `and` 表达式的展开开销。
只读检查安装版 FlyDSL `ast_rewriter.py` 的 `RewriteBoolOps.visit_BoolOp`（line541–575），
发现每新增一项会在多个 AST 位置引用上一项 expression；chained comparisons 又会先展开成 `and`。
本地仅抽取原 class 做表达式验证，7 项 `and` 的引用遍历节点为30,033；相同7项 `all((...))`
为20。该结果只是 Python AST 实验，不能当成完整 compiler profile 或 GPU 性能数据。

v3 将新增 lookup guard 改成同范围的 `all((...))`，重新实际执行63,994项整数 lookup、143项 guard 检查通过。
CPU-only 编译在105.1098秒完成，actual G1 ELF为
`ab1866fd192c50ff8e90c89487cd56e10bf5ab40b407d22019201be4488f343d`。
相对M-group32，WMMA/量化/activation、transfer及关键fences核验相同；
VGPR762/SGPR58/LDS327680 B/private0/spills0，静态ISA条数5880/5880。
减少一次endpoint查找没有转化为静态总指令减少，实际性能仍须配对测量。
v3的首个完整strict进程在多kernel准备期间超时，没有完成精度结果，未据此判断精度通过或性能收益。

v4 还将原M-group32 guard改成相同的紧凑constexpr表达式，实际执行4,264项mapping guard等价检查，
并再次执行63,994项lookup及143项guard。source SHA为
`5b6b47ed4e53249901a692c725099f7f1beb165b6583813dcc95b4d2c17c212e`，相对生产+17/-8行。
全新私有CPU cache下13.5009秒完成，完整ELF与v3逐字节一致，编译选项、source locations和安装文件保持。
strict入口改为直接比较当前生产M-group32，不再额外导入旧baseline module；
仍检查相同输入、完整MoE输出、有效payload/ScaleA字节和same-pointer动态CUDAGraph replay，仍用原reference和精度阈值。
当前开始E64/E65 seed0/17及candidate自身非目标direct compile，尚无v4性能成绩。
正式生产仍是M-group32 `de3e101...670`，正式GEMM1仍86.6212 μs。

### bounded lookup 的 E65 编译拆分与完整精度（2026-10-22）

v4 的 validation-v2/v3 在 E65 seed0 编译准备阶段分别触发 rc124，未完成精度验证。
两个失败记录保持，不延长120秒timeout；child均退出，主机检查没有GPU/KFD残留。
validation-v3虽复用此前验收cache，但当前生产和候选的source hash不同，仍有实际ABI缺失的编译。
因此新增`prepare_original_stage_e65_v1.py`，由未修改的原入口`--scenario verify`构造E65/T1536/topk9输入，
逐stage准备production/candidate，并用正常FlyDSL源码hash与cache key生成artifact，没有把旧artifact改名或复制到cache。
四项独立child分别52.0829、100.1597、11.5247、11.3022秒完成，准备耗时全部不作为性能数据。
finally精确恢复production source SHA。

E65 original fallback actual G1 ELF为`d173f0b236adfb2d61acb596dc06b2c46b1e02ea9d75aeae8016db902c18fff3`，
G2为`3b63a9571dca9410100499f8eb3507d09f3b248494077ca4940d1b09f0abc5ec`。
两stage的candidate与production、此前E65已验收control完整ELF一致。

validation-v4复核已完成的E64 seed0/17原始证据，重新执行E65 seed0/17，两项各27个BYTE_CHECK零差异，
完整MoE22020096 B零差异，原reference及same-pointer动态CUDAGraph replay通过。
E65仍为分支原`t64/w1x4` fallback与random65/topk9，此处不表示固定shared1 routing或E64/E65共用t256。
当前继续candidate自身8项outside-scope direct compile，尚无bounded lookup有效性能结果。

### bounded lookup 完整验证与缓存核验修复（2026-10-22）

validation-v4 已全部完成：E64/E65 seed0/17 的四项完整精度检查通过；candidate 自身八项
outside-scope direct compile（GEMM2、FP8、original A、pin1、t64、w1x4、buffers3、persistent1）
完整 ELF 全部与既有 control 相同。真实 benchmark ABI 的 G1/G2 准备分别在13.4739/12.5063秒完成，
完整 ELF 分别为 bounded candidate G1 与原 GEMM2；finally 恢复生产。准备时间均不作为性能数据。

首个正向队列 `reboot20261022_bounded4_forward_v1` 在第一轮生产计时后的缓存核验退出，未完成三轮组，
`completed=false/results=[]/source_restored=true`。本次未采用任何耗时、未发布性能提升。
v2 核验递归扫描整个准备 cache，混入了 correctness 从 candidate 文件导入生成的另一源码 namespace。
只读检查确定 `81e707b99b296d9f50f1dc7269818e0d/74bcf3959936b647.pkl` 的入口 source location 指向
`candidates/ds_xdl_mgroup32_bounded_lookup_v4.py`，实际生产 artifact 位于
`9ea1c3cdc0833af76698efb2571e85b9/13e21c457a09ffb0.pkl`。不同 namespace 的 ABI 相同，call key 不同，
旧脚本错误地要求两者使用同一 key。

新增 v3 核验先用安装版 `JitFunction._ensure_cache_manager` 实际调用的
`_jit_function_cache_key` 从当前 canonical 生产源码和依赖计算 namespace，再选择原 benchmark 的
G1/G2 精确 call key，核对原 entry ABI、source location 和完整 GPU ELF。没有按 ELF 挑选 namespace、
改名/复制/删除 cache 或修改 benchmark。新的 v5 性能 runner 使用相同用户命令、全机 idle 与120秒限制，
缓存核验在计时进程退出后单独运行；旧失败证据保留。正式成绩仍为86.6212 μs，候选尚无有效性能结果。

### bounded lookup 对照与 TDM owner 的初步 native 检查（2026-10-22）

`reboot20261022_bounded4_forward_v2` 已取得前两组完整合格三轮：

| 组 | GEMM1 三轮 μs | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---|---:|---:|
| M-group32 before | 87.4501 /87.3025 /87.5513 | 87.4501 | 64.3587 |
| bounded lookup，try2 | 86.8069 /87.2644 /86.4212 | 86.8069 | 63.6904 |

候选相对前侧改善0.6432 μs（0.7355%），后侧M-group32仍在补齐完整合格组。
候选try1、后侧control的try1～try3因post/settled ownership不合格整组排除，
不从其他单轮拼中值；当前尚未完成双侧对照，未采用候选。原reference读数为
`logits_diff=3.61055e-06`。生产正式记录保持86.6212 μs。

在外部GPU占用期间暂停的是等待idle的性能parent，没有暂停正在计时的GPU child。
CPU-only编译结束、确认无编译child且生产SHA精确恢复后，已恢复性能parent继续队列。
所有编译/验证child仍限制120秒TERM、5秒后KILL；编译数据不参加性能统计。

`ds_xdl_mgroup32_tdmw2_v2/tdmw4_v2` 使用现有TDM owner partition，
相对M-group32仅增加owner默认选择与等价紧凑mapping guard（+10/-4行），
不增加pipeline或persistent，不改w2x2布局、ring、compute、descriptor构造和fence公式。
154项actual guard检查，以及48项partition、798720个source/destination element覆盖通过。

| TDM owner | VGPR | SGPR | LDS B | private / spills | G1完整ELF SHA256 |
|---:|---:|---:|---:|---|---|
| 1，生产 | 762 | 58 | 327680 | 0 / 0 | `d6636a9a1e4d6cf74332be6f30d03fa13327edff96da0533a21c9ec0745e4655` |
| 2，候选 | 762 | 78 | 327680 | 0 / 0 | `c698db36ad18ea71662894dc0429e7434de83b6f6cbe14ffeab9e936ab351eb1` |
| 4，候选 | 760 | 98 | 327680 | 0 / 0 | `574928aba8f0573c01d3ca51dbde820d38ad563bb634e35003d3782aa5a2c3b8` |

旧native gate因整份ISA的WMMA、DS、fences静态总数不同及SGPR增加而失败，记录保留。
按实际ELF branch target补充检查发现：四/两/一份owner-specific steady body分别供
owner1/2/4使用，每个body每iteration均有64条WMMA、72条LDS load；每wave只选择一组，
原24个steady iteration、3个rolled drain和最后1个tile保持。共同drain和最后tile的
128个静态WMMA site、activation/quant opcode数与非steady DS opcode/offset保持。
静态WMMA总数384/256/192的差异可由共享body解释，不能当作动态少算或性能收益。

`audit_mgroup32_tdm_native_v3.py` 还发现8个LDS literal offset及重复的steady-entry
`s_wait_dscnt 0`数不同，保留失败记录。v4按每owner执行路径核验：
steady Tensor wait由2变4/8，prologue由3变6/12，drain2/1由4/2或8/4替代；
对应barrier与非steadyTensor/store fences序列保持。owner body合并使重复entry DS wait
由4份变2/1份；literal offset差异保留，不能仅凭opcode数证明实际地址或descriptor寄存器值。

v4是允许开展不计时完整device等价验证的初步检查，尚未授权性能采用。
必须先验证全部有效FP4 payload、ScaleA字节、完整MoE输出和same-pointer动态graph；
再以原benchmark真实ABI准备并计时，不改计时口径或精度门槛。
硬件依据为Shader Guide rev65 §4.10.1（p197）：TDM忽略EXEC、同wave Tensor有序、
跨wave与其他memory类型无序。因此保持每wave匹配请求数和既有barrier，不能以EXEC掩码替代owner dispatch。

### bounded lookup 双侧结论：不采用（2026-10-22）

后侧control首个完整合格组为try4，GEMM1三轮86.3430/86.8679/86.2144 μs，
中值86.3430 μs，GEMM2中值63.7450 μs。完整正向序列为：
M-group32 87.4501 → bounded lookup 86.8069 → M-group32 86.3430 μs。
候选相对前侧改善0.7355%，相对后侧回退0.5373%（+0.4639 μs），没有通过双侧采用条件，
不追加反向确认，不采用该候选。生产SHA仍为`de3e101...670`，正式记录仍86.6212 μs。

514成员、2061784 B证据archive已下载，本地独立v3验收复算原计时表、ownership、
canonical源码namespace、原entry ABI/cache key、实际ELF、四项完整strict和八项direct scope通过。
archive SHA=`a9c3353b556150df53f63948c8a2d5f1a10eeb65eca5f4d52b46bacb1637f107`。
`reboot20261022_bounded4_forward_v2.comparison_report.json`记录双侧结果，
`positive_at_every_adjacent_control=false/production_adopted=false`。
计时后确认没有本项GPU/KFD残留。继续TDM owner2/4候选，仍基于保留版原SHA。

### TDM owner2 精度通过、owner4 精度失败（2026-10-22）

`ds_xdl_mgroup32_tdmw2_v2_strict_e64_s0_v1` 在62.3296秒完成，rc0；27项
有效FP4 payload、ScaleA及same-pointer动态graph的BYTE_CHECK全部零差异，完整MoE
22020096 B零差异，原reference和精度阈值保持。当前仅完成E64 seed0初筛，
不表示已经完成seed17、E65及candidate自身outside-scope检查，也不表示性能提升。

`ds_xdl_mgroup32_tdmw4_v2_strict_e64_s0_v1` 在62.6252秒退出，rc1。首项
`baseline_eager_payload` 就失败：valid_rows12288、12582912 B中有6835570 B不同。
不生成通过记录、不运行性能计时、不采用该候选。owner4涉及ScaleA inner partition与
LDS row pitch，当前只是待验证的定位假设，不能将Python partition覆盖等价当成实际TDM存储等价。

owner2原benchmark真实ABI的两个stage已分别在13.2201/12.6002秒完成准备，均小于120秒；
G1完整ELF为`c698db36ad18ea71662894dc0429e7434de83b6f6cbe14ffeab9e936ab351eb1`，
G2保持`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。
`real_stage_preparation_v2.acceptance.json`记录completed/source_restored为true，
全部准备耗时不作为性能数据。生产SHA精确恢复，未发现本项child残留。

新增owner2专用runner，保留原用户计时入口、canonical源码namespace/ABI/cache key/完整ELF
检查、全机idle、120秒TERM/5秒KILL和完整三轮组规则；owner4失败证据保留且不进入队列。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_normal_mgroup32_tdm_owner2_v5.py \
  NEW_TDM_OWNER2_FORWARD_LABEL base owner2 base
```

只有同时优于两侧control才追加反向确认；初筛未完成前正式成绩保持86.6212 μs。

### owner2 原入口首侧 control 与 owner4 修正候选（2026-10-22）

owner2专用runner的首次启动因远端缺少`run_normal_mgroup32_tdm_screen_v2.py`
依赖，在任何源码swap或GPU运行前退出。失败host log保留；仅补传任务所需依赖并核SHA，
使用新label `reboot20261022_tdmowner2_forward_v2`重新启动，没有覆盖失败证据。

前侧M-group32 control完成三轮87.3254/86.2544/87.1443 μs，中值87.1443 μs，
GEMM2中值63.5552 μs，全部原reference、canonical源码namespace/ABI/ELF和ownership合格。
当前仍待owner2与后侧control，尚无完整性能结论。首轮mode的compileprep为60.8727秒，
该准备进程所打印μs不参加性能统计。

按Shader Guide rev65 §4.10.2 pp199–200的实际LDS地址模型重新核查owner4：
原ScaleA inner split让每wave传输2×64 dwords，而消费端布局为2×256 dwords。
TDM按tile宽度紧凑写LDS，单独提供destination view stride不会插入row间隙。
模型得到512个ScaleA写入中192个destination重复；原静态partition脚本使用了完整view row stride，
因此它的覆盖通过不能证明硬件写地址正确。该模型与设备精度失败一致，但完整native descriptor数据流尚未证明。

`ds_xdl_mgroup32_tdmw4_rows_v3`将这项受guard限制的inner partition改为4个单行tile：
每wave传输1×128 dwords，先分outer row再分inner half；不改变compute、输入ring、
每wave四项TDM请求数或fences。基于生产共+17/-4行，其中相对失败owner4仅增加局部row partition。
source SHA=`146230cb1baeb7332abd6772a6d0e699cc1e16288d3a9c5d1c8dad926fa61af6`。
A/B/ScaleA/ScaleB实际source→硬件packed LDS模型均完整覆盖且无destination重叠。
当前仅本地生成、未上传编译或dispatch，不作为已经修复device错误或获得性能提升的结论。

### owner2 原call key核验修正（2026-10-22）

`reboot20261022_tdmowner2_forward_v2`在owner2首轮结束后的cache核验退出，
没有完成候选三轮或后侧control；生产源码恢复，无本项GPU进程残留。
其owner2耗时不作为有效性能样本，前侧control保留为独立历史记录，不拼到新序列。

实际canonical源码namespace为`d943e0b2faded1c6679ef13448477d1f`，owner2原call key为
G1=`62c9ae1cec751fe7`、G2=`da3c5914af92622d`，而M-group32 control为
G1=`13e21c457a09ffb0`、G2=`c33b05449501ed1b`。旧核验错误地假设相同入口ABI必然
具有相同call key。安装版`JitFunction._build_full_cache_key`还包含`_globals_key_prefix`；
逐stage用原benchmark真实inputs调用原launcher、保留原key算法后确认两侧参数signature完全一致，
globals仅新增`dsl_not_`引用，因紧凑constexpr mapping guard的AST展开不同而进入key。
编译后的G1分别为各自正确候选ELF，G2仍相同；没有更改key或复制/改名cache artifact。

`prepare_normal_stage_keys_v2.py`记录原调用由安装版生成的完整key string、参数和globals
snapshot；四项准备child都小于120秒，`mgroup32_tdm_original_keys_v1.json`
completed/source_restored为true。`audit_normal_cache_original_keys_v4.py`先计算源码namespace，
核对当前globals和已由实际原调用记录的key，再读取该唯一artifact；仍核验完整原entry ABI、
source location与ELF。baseline/owner2两项独立read-only核验已通过，没有GPU dispatch。

使用新label `reboot20261022_tdmowner2_forward_v3`重新完整执行base→owner2→base，
各三轮，计时命令和精度阈值不变；旧失败证据保留。
owner4 row修正候选已传入私有candidates目录，尚未编译或dispatch，排在本项之后。

### owner2 新序列首侧与 owner4 descriptor 补充证据（2026-10-22）

`reboot20261022_tdmowner2_forward_v3`的前侧M-group32首个合格组三轮为
86.4399/86.3261/86.5136 μs，中值86.4399 μs，GEMM2中值64.1625 μs。
前侧try1和owner2 try1都因一轮post/settled ownership不合格而整组排除，
中值保持null；当前继续owner2完整重测及后侧control，不将排除组中的单轮用于性能结论。
原benchmark、120秒child限制、A preshuffle1/pin0及精度阈值保持。

对已下载的失败owner4实际ISA进行只读检查：首个ScaleA TDM在PC`0x2024`，
指令为`tensor_load_to_lds s[52:55], s[36:43]`。该路径中的常量定义为：

```text
0x1e40: s_mov_b32 s40, 2
0x2000: s_mov_b32 s39, 0x407fff
0x2008: s_mov_b32 s36, 0x20000
0x2024: tensor_load_to_lds s[52:55], s[36:43]
```

按Shader Guide rev65 §4.10.4 Table80（文档p202–203）解码group1：
`data_size=2`表示4B，`pad_enable=0`，`tile_dim0=64`、`tile_dim1=2`。
检查上述常量至该请求之间没有覆盖，`0x1f10`至请求之间没有branch。
因此这个代表性请求确实是无padding的2×64 dword tile，支持此前硬件packed LDS模型。
本项仍未证明所有descriptor的完整global/LDS地址数据流或所有动态路径；
修正候选仍需独立compile及完整device字节检查。

### owner2 完整对照：无收益，不采用（2026-10-22）

`reboot20261022_tdmowner2_forward_v3`已完整结束。三组均使用第一个完整合格三轮组，
前側try1、owner2 try1/try2因post/settled ownership不合格排除，未按快慢选择或拼接单轮。

| 组 | 合格try | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---:|---|---:|---:|
| M-group32 before | 2 | 86.4399 /86.3261 /86.5136 | 86.4399 | 64.1625 |
| owner2 | 3 | 88.0552 /87.3952 /87.6565 | 87.6565 | 64.6791 |
| M-group32 after | 1 | 86.9166 /86.3768 /86.1871 | 86.3768 | 63.7703 |

owner2比两侧分别慢1.2166/1.2797 μs，即1.4075%/1.4815%。没有通过双侧采用条件，
不追加反向、不采用候选；正式保留版及正式成绩86.6212 μs保持。
GEMM2实际完整ELF未变，表中波动不属于GEMM2代码修改或优化。

389成员、1642822 B完整证据包已下载，本地独立验收逐成员SHA、原precision表、
全机pre/post/settled、canonical源码namespace、由安装版原调用产生的source-dependent keys、
ABI及两stage完整ELF，并重新计算全部三轮中值，验收通过。
archive SHA=`8e0e3333e095da1049fd3607925d1cadbe06616f80a3af6d9b5ea8b63485307c`。
`.local_acceptance.json`记录`positive_at_every_adjacent_control=false/production_adopted=false`。
这里的精度初筛仅E64 seed0；没有声称完成owner2的seed17/E65及candidate自身非目标检查。

退出后核验生产source SHA精确恢复`de3e101...670`，没有本项GPU/KFD或child残留。
随后开始`ds_xdl_mgroup32_tdmw4_rows_v3` CPU-only编译，继续原120秒TERM/5秒KILL限制。
尚无修正owner4的device或性能结果。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_normal_mgroup32_tdm_owner2_v6.py \
  NEW_TDM_OWNER2_LABEL base owner2 base
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_mgroup32_tdm_owner4_rows_cpu_v1.py
```

第二条是冻结label的独立compile-only实验，只可在尚无其输出时运行；不用于性能复现。

### owner4 row 修正通过完整 E64/seed0 精度检查（2026-10-22）

`ds_xdl_mgroup32_tdmw4_rows_v3` 的独立 CPU-only 编译在12.7751秒完成。
GEMM1 完整 ELF SHA=`78f0c6fc204e4112035c84df2278f57aa7ae6e4bcdbc7891dac811cdf66e116b`，
资源为VGPR760、SGPR100、LDS327680 B、private0，VGPR/SGPR spill均为0。
与失败owner4相比，math及LDS load opcode数量、保护fences和LDS容量保持；这些静态检查只用于授权不计时精度验证。

修正后的代表性ScaleA TDM请求位于PC`0x204c`。按Shader Guide rev65 §4.10.4 Table80解码：
`s55=0x807fff`对应`tile_dim0=128`，`s56=s12=1`对应`tile_dim1=1`，
`s52=0x20000`对应4B且`pad_enable=0`，与§4.10.2 pp199–200的packed LDS模型一致。
此处仅核对代表性请求及其常量定义，不声称已证明所有descriptor的完整地址数据流。

`ds_xdl_mgroup32_tdmw4_rows_v3_strict_e64_s0_v1` 在62.1289秒完成，rc0。
27项payload/ScaleA BYTE_CHECK全部零差异；same-pointer动态graph检查和原reference通过。
完整MoE输出shape`(1536,7168)`、22020096 B逐字节零差异，实际GEMM1匹配修正ELF，
GEMM2保持原ELF `4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。
本次仅E64 seed0，不表示已验证candidate的seed17、E65或所有非目标shape。

随后从原benchmark真实inputs与原launcher获得四项stage key，参数signature与保留版一致。
候选canonical namespace为`412f24a5807c4a9260558b176109718d`，
G1/G2原call key分别为`62c9ae1cec751fe7`/`da3c5914af92622d`；没有改key算法或复制、改名cache产物。
`mgroup32_tdm_owner4_rows_original_keys_v1.json` 的completed/source_restored均为true，
SHA=`1a6607ced358aa2feff43ca80f88370016cb1c602cf135de52892ac1daac0bd2`。
四项准备均限制120秒；这些准备及精度运行的耗时不作性能成绩。

接下来使用原计时命令执行base→owner4rows→base，各取第一个完整合格三轮组；
全机空闲、A preshuffle1/pin0、原precision gate及120秒TERM/5秒KILL规则保持。
只有候选同时优于两侧control才追加反向确认及额外scope验证。
尚无四wave候选的有效性能结论，正式保留版仍为86.6212 μs。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_mgroup32_tdm_owner4_rows_key_audit_v1.py
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_normal_mgroup32_tdm_owner4_rows_v1.py \
  NEW_TDM_OWNER4_ROWS_LABEL base owner4rows base
```

第一条是已冻结label的独立read-only核验，不可覆盖已有输出；第二条需使用新label。

### owner4 等待空闲期间完成 inline-zero 的 native 与精度初筛（2026-10-22）

`reboot20261022_tdmowner4rows_forward_v1` 的前侧control try1因两轮post ownership
不合格而整组排除，中值保持null；try2也已出现不合格单轮，尚未完成合格三轮组。
其他用户在a07-3运行连续GEMM实验，runner每6秒检查全机idle，未杀死或驱逐他人任务。
这些不完整/受干扰数据不用于候选收益判断，正式保留版仍为86.6212 μs。

在确认本任务parent唯一child是idle waiter、没有benchmark或compile child、且生产source
精确等于`de3e101...670`后，只暂缓host parent启动下一次计时。idle轮询继续，期间顺序
执行CPU-only编译和不计时精度验证，没有与本任务性能测试重叠。
暂停记录为`owner4rows_wait_cpu_preparation_pause_v1.json`，随后按source恢复检查继续原序列。

新候选`ds_xdl_mgroup32_inlinezero_v2`仅对原SiLU/fused quant/FP4/A preshuffle/pin0/
非persistent/t256/w2x2/b4/cluster1x4、存在steady loop的配置，首个K128 WMMA C/SRC2
使用inline constant zero，后续K128继续正常累加。现有input ring、issue顺序、fences和
activation/quant路径保留，没有新增pipeline或改变launcher参数与shape准入。
相对保留source为+28/-12行；source SHA=`0118e3b85d270114646586804ba7e1ec19551b69efa05b388b07b6db0c0ee167`。
CDNA5 ISA §7.12，文档p96记载WMMA SRC2允许even VGPR或inline constant zero；
这里检查的是C/SRC2，不把ScaleA/ScaleB的zero字段算成初始化优化。

本地actual guard4319项、peeled loop order512项和独立完整K128 order124项检查通过。
原launcher参数及其他top-level function/class AST一致。专用复现工具逐层静态组装通过，
最终性能shell command与`run_normal_pair_major_v1.py`原计时command逐字一致；
mock native只用于组装检查，不作为device或性能证据。

CPU-only编译在14.0928秒完成，rc0，finally精确恢复生产source。
GEMM1 ELF SHA=`bec431f59ec2501b72f620bf90e24fee75e07a329e2f7c9a54cac43bcf4029ac`，
VGPR762/SGPR62/LDS327680 B/private0/spills0；VGPR和LDS与保留版相同，SGPR增加4。
actual ISA有128个首K128 inline-zero C/SRC2 static sites，后续C仍为register。
首tile peel使static WMMA sites由384变640，不能据静态条数解释动态有效计算变化；
原tensor output store、activation/quant opcode数量和LDS容量检查保持。

`ds_xdl_mgroup32_inlinezero_v2_strict_e64_s0_v1`在63.3028秒完成，rc0。
27项BYTE_CHECK全零，完整MoE输出`(1536,7168)`、22020096 B逐字节零差异，
same-pointer动态graph和原reference通过。实际GEMM1匹配上述候选ELF，GEMM2保持
原完整ELF `4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。
strict screen acceptance已通过，但仅覆盖E64 seed0；不表示seed17、E65或outside-scope已验证。
本轮准备和正确性耗时均不作为性能成绩。

接下来先按原benchmark真实inputs和原launcher生成stage keys，核对原入口ABI及实际ELF，
不改key算法、不复制或改名cache。TDM owner4对照恢复并结束后，再执行inline-zero原入口
双侧完整三轮对比；有收益才补反向确认及其他scope。每个child仍限120秒TERM/5秒KILL。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_mgroup32_inlinezero_cpu_v3.py
bash .codex_tmp/yadai_gemm1_param_20261021/tools/run_mgroup32_inlinezero_strict_v1.sh \
  NEW_INLINEZERO_STRICT_LABEL
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/accept_mgroup32_inlinezero_strict_v1.py \
  NEW_INLINEZERO_STRICT_LABEL
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_mgroup32_inlinezero_keys_v1.py
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_mgroup32_inlinezero_key_audit_v1.py
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_normal_mgroup32_inlinezero_v1.py \
  NEW_INLINEZERO_PERF_LABEL base inlinezero base
```

CPU和key准备使用冻结label，仅可在没有已有输出时执行；strict/perf需新label。
此处只记录流程及已完成native/seed0初筛，不将后续尚未执行的步骤记作通过。

### inline-zero 原入口绑定完成，继续 TDM 完整对照（2026-10-22）

inline-zero 的四项原benchmark stage准备均已完成，`mgroup32_inlinezero_original_keys_v1.json`
的completed/source_restored为true。baseline namespace=`9ea1c3cdc0833af76698efb2571e85b9`，
G1/G2 keys=`13e21c457a09ffb0`/`c33b05449501ed1b`；inline-zero namespace=
`455bf576b49d66b8aea8482c340d0ed4`，G1/G2 keys=`62c9ae1cec751fe7`/`da3c5914af92622d`。
四项实际ABI/signature相同，候选G1匹配`bec431f...9ac`，G2保持`4bd626e...716be`。
没有修改key算法、复制或改名cache。两项独立read-only key审计的`passed`均为true。

native和strict原始ELF/ISA/log已下载，本地独立read-only复核
`ds_xdl_mgroup32_inlinezero_v2.downloaded_preflight_acceptance_v1.json`通过。
首个K128使用inline-zero、后续原序累加及27项完整字节检查仍只作为E64/seed0初筛；
尚未计时，也没有声称seed17/E65或非目标配置已经验证。

TDM owner4 parent已在独立准备结束、生产source恢复及三次全机idle后继续，
记录为`owner4rows_wait_cpu_preparation_resume_v1.json`。
恢复后前侧control try3仍有post ownership不合格，整组三轮中值null。
try4也已出现不合格轮，继续按全机idle和完整三轮规则重测；不拼轮、不以这些读数更新正式成绩。
正式生产source及86.6212 μs保持，inline-zero性能对照排在当前TDM序列结束之后。

### owner4 row 完整对照结束：回退7.72%–7.83%，舍弃（2026-10-22）

`reboot20261022_tdmowner4rows_forward_v1`已完成原入口base→owner4rows→base。
各组使用第一个完整合格三轮组；前侧try1–4因post/settled ownership不合格整组排除，
中值均为null，没有跨组拼轮或按读数挑选。最终三组均通过原precision gate及原入口
source/namespace/call keys/ABI/实际ELF核验，A preshuffle1、pin0保持。

| 组 | 首个合格try | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---:|---|---:|---:|
| 前侧保留版 | 5 | 87.0242 /86.9797 /88.3503 | 87.0242 | 64.4808 |
| owner4rows | 1 | 93.7457 /93.5475 /93.9292 | 93.7457 | 64.3564 |
| 后侧保留版 | 1 | 86.7984 /86.9651 /86.9376 | 86.9376 | 64.0971 |

候选分别慢6.7215 /6.8081 μs，即回退7.7237% /7.8310%。不追加反向或额外scope，
不采用候选；其seed0精度通过不能替代性能收益，也不能延伸为seed17/E65验证。
虽然候选VGPR从762降到760，SGPR从58增到100，静态资源与本次耗时不足以单独证明
回退的唯一原因。正式生产版仍是DS_B计数修正+XDL queue bit+M-group32，正式成绩
86.6212 μs保持，距80 μs差6.6212 μs。

队列报告`NORMAL_ENTRY_SOURCE_RESTORED True`和`NORMAL_ENTRY_QUEUE_COMPLETE`。
退出后独立检查生产source SHA=`de3e10100a1b95a99253f49ad180a086194aad2ab49c02d4026137c959279670`，
本项child已退出，04:23:56 UTC的主机GPU检查没有GPU进程残留。
远端branch/HEAD仍为`hyg_dsr1_conc6144` /`346438a75506f77b520d3a9a09ba4985f054e377`。

完整证据已归档并下载，archive SHA=
`a064ff241b77eefce6fde2aa8d67f10f861630fabf0911d99024230c7c1c4c4d`，
1688641 B、366个member。封包包括暂停/恢复记录、真实三轮日志、native/strict证据及
原keys/ABI/ELF。首次封包发现本地builder尚未上传，补传该任务文件并核SHA后完成封包；
没有重新编译或重测，也没有改变原始性能记录。Windows本地独立read-only复算通过：
`reboot20261022_tdmowner4rows_forward_v1.local_acceptance.json`的`passed=true`，
`positive_at_every_adjacent_control=false`、`production_adopted=false`。

```bash
timeout --signal=TERM --kill-after=5s 120s python3 -u \
  .codex_tmp/yadai_gemm1_param_20261021/tools/seal_tdm_owner4_rows_screen_v2.py \
  reboot20261022_tdmowner4rows_forward_v1
python -u .codex_tmp/yadai_gemm1_param_20261021/tools/accept_tdm_owner4_rows_screen_v1.py \
  reboot20261022_tdmowner4rows_forward_v1
```

以上label已经冻结，不能覆盖输出。下一项是已通过native与E64/seed0 strict初筛的
inline-zero原入口双侧三轮对照；性能尚未产生，不先行合入正式生产版。

### inline-zero 正向对照完成：小幅收益，待反向确认（2026-10-22）

`reboot20261022_inlinezero_forward_v1`已完成base→inlinezero→base。
候选try1、try2因post/settled ownership不合格整组排除，中值为null；没有跨组拼轮。
最终各组使用第一个完整合格三轮组，原precision gate、A preshuffle1/pin0及原计时命令保持。

| 组 | 首个合格try | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---:|---|---:|---:|
| 前侧保留版 | 1 | 86.2102 /86.9102 /85.9677 | 86.2102 | 63.8652 |
| inline-zero | 3 | 86.1427 /85.5714 /85.8308 | 85.8308 | 64.3121 |
| 后侧保留版 | 1 | 86.4929 /87.2918 /86.6888 | 86.6888 | 63.9748 |

候选分别快0.3794 /0.8580 μs，即0.4401% /0.9897%。当前仅为正向初筛，
收益较小，尚不能代替反向确认或seed17/E65/outside-scope验证，正式成绩86.6212 μs保持。
两侧GEMM1匹配`d6636a9a...e4655`，候选匹配`bec431f5...029ac`；
所有GEMM2均匹配原ELF `4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。

队列报告source恢复及完成，独立核对生产source SHA仍为`de3e101...670`；
04:46:34 UTC的主机GPU检查无进程，本项child已退出。
完整证据已封包下载，archive SHA=
`8ed5d1734cf776b29c573a5c598bfd9097d561df2674558a4470da5a879f333f`，
1341636 B、291个member。首次封包发现两份builder和旧v1候选仅在本地，
补传并核SHA后完成，没有重新编译或更改测试记录。本地read-only独立复算
`reboot20261022_inlinezero_forward_v1.local_acceptance.json`通过，
`positive_at_every_adjacent_control=true`、`production_adopted=false`。

随后启动新label `reboot20261022_inlinezero_reverse_v1`，顺序inlinezero→base→inlinezero。
只有反向两组候选也均优于中间control，才补额外scope并考虑采用。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_normal_mgroup32_inlinezero_v1.py \
  NEW_INLINEZERO_REVERSE_LABEL inlinezero base inlinezero
timeout --signal=TERM --kill-after=5s 120s python3 -u \
  .codex_tmp/yadai_gemm1_param_20261021/tools/seal_inlinezero_screen_v1.py \
  NEW_INLINEZERO_REVERSE_LABEL
python -u .codex_tmp/yadai_gemm1_param_20261021/tools/accept_inlinezero_screen_v1.py \
  NEW_INLINEZERO_REVERSE_LABEL
```

每个compile、verify、perf及audit child均限120秒TERM/5秒KILL；等待全机idle的host parent继续每6秒轮询。

### inline-zero 反向对照完成：四项相邻比较均有收益，继续兼容性验证（2026-10-22）

`reboot20261022_inlinezero_reverse_v1`已完成inlinezero→base→inlinezero。
中间control try1和后侧候选try1–3均有post/settled ownership不合格，整组三轮排除，中值null。
最终仍使用每组第一个完整合格三轮组，没有拼轮或删除合格组中的偏慢读数。

| 组 | 首个合格try | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---:|---|---:|---:|
| 前侧inline-zero | 1 | 85.5607 /85.5586 /88.1288 | 85.5607 | 64.0613 |
| 中间保留版 | 2 | 87.3149 /86.6382 /86.9797 | 86.9797 | 63.9791 |
| 后侧inline-zero | 4 | 86.3178 /85.6621 /86.2206 | 86.2206 | 63.7679 |

反向两组候选分别快1.4190 /0.7591 μs，即1.6314% /0.8727%。与正向合并后，
四项相邻比较全部为正：0.4401%、0.9897%、1.6314%、0.8727%。三个合格候选组的
中值为85.8308、85.5607、86.2206 μs；如后续验证通过而采用，正式成绩使用首个正向
候选中值85.8308 μs，不选最快单轮或最快组。收益规模约为1%，尚未达到80 μs。
这些数据只对应E64/seed0原入口；不同seed、E65及非目标配置的验证尚未完成。

队列报告`NORMAL_ENTRY_SOURCE_RESTORED True`和`NORMAL_ENTRY_QUEUE_COMPLETE`。
退出后独立核对生产source SHA仍为`de3e10100a1b95a99253f49ad180a086194aad2ab49c02d4026137c959279670`，
branch/HEAD仍为`hyg_dsr1_conc6144` /`346438a75506f77b520d3a9a09ba4985f054e377`，
boot_id保持。05:10:45 UTC主机GPU检查无进程，没有本项benchmark或audit child残留。

反向完整证据已下载，archive SHA=
`0fef1e7b6e7d5241170adbca5998489e90c909553195e5ba8dc4fdfff4b45edd`，
1592289 B、363个member。本地read-only复算
`reboot20261022_inlinezero_reverse_v1.local_acceptance.json`通过；合并记录
`inlinezero_sequences_v1.comparison.json`的`adjacent_comparisons=4`、
`positive_at_every_adjacent_control=true`、`production_adopted=false`。
生产代码和正式成绩86.6212 μs暂时保持。

下一步顺序补E65原输入的两阶段独立编译准备、E64/seed17和E65/seed0/17完整MoE字节验证，
再对候选自身执行8项outside-scope及4项额外guard条件的CPU-only完整ELF比较。
E65仍为原t64/w1x4 fallback，不把此次验证解释成E64/E65共用t256 kernel或固定shared expert1 routing。
新工具仅完成LF/语法及静态组装检查，尚不作为native、device或性能证据。

```bash
python -u .codex_tmp/yadai_gemm1_param_20261021/tools/compare_inlinezero_sequences_v1.py
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_inlinezero_e65_preparation_v1.py
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_inlinezero_validation_v1.py
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_inlinezero_extra_guard_scope_v1.py
```

第一条为Windows本地证据复算；后三条在远端主机运行parent，由parent进入dsr1_ep4执行
各个120秒上限child。固定label的已有输出不可覆盖。全部验证通过后才采用候选。

### inline-zero 完整 MoE 与非目标配置验证（2026-10-22）

E65原输入的production/inlinezero两种source、GEMM1/GEMM2四项独立编译准备全部通过，
child耗时分别为51.7628、53.3151、11.6876、11.3803秒。实际E65 GEMM1/GEMM2完整ELF
在两种source下分别相同，仍为原t64/w1x4 fallback；生产source已恢复。
这些准备耗时不作为kernel性能，不表示已验证固定shared expert1 routing。

| workload | seed | strict child秒数 | BYTE_CHECK | 完整MoE输出不同字节 |
|---|---:|---:|---|---:|
| E64/T1536/topk8 | 0 | 63.3028 | 27/27为0 | 0 |
| E64/T1536/topk8 | 17 | 55.7051 | 27/27为0 | 0 |
| E65/T1536/topk9 | 0 | 61.5770 | 27/27为0 | 0 |
| E65/T1536/topk9 | 17 | 56.2358 | 27/27为0 | 0 |

四项均核对实际两阶段ELF，完整输出shape为`(1536,7168)`、22020096 B，
same-pointer动态graph及原reference通过。各项原始字节检查与日志保留，不改变precision gate。

候选自身的8项outside-scope CPU direct compile也全部通过：GEMM2、FP8输入、
非preshuffled A、pin1、t64 fallback、w1x4、buffers3、persistent1。
每项实际完整ELF与原control逐字节相同，不通过切回原module冒充候选验证。
`ds_xdl_mgroup32_inlinezero_v2.validation_v1.json`的`completed=true`，
strict四项及scope八项均无error。

额外guard检查独立编译control和candidate：K1024无steady body、K256单tile、
关闭next-stage prefetch、关闭fused quant，四项已全部通过完整ELF比较。
八次child均在120秒以内结束，最慢103.3518秒；没有延长timeout或使用失败结果。
至此共12项非适用配置的完整ELF检查通过。

### inline-zero 验收及正式采用（2026-10-22）

完整追加证据归档已下载到Windows本地，archive SHA256为
`c0ea94f2eb1fae83c92f6012c5e6fd1a77b589e75e4582a3ca95d6e14957fa72`，964834 B、225个member。
`accept_inlinezero_followup_v1.py`独立复算四项严格字节检查及十二项完整ELF比较后通过；
正反向性能比较记录与原benchmark/wrapper SHA也通过采用前检查。

`adopt_inlinezero_v1.py inspect/apply`已分别在本地和a07-3执行，原子替换仅此GEMM1源码，
保留前版备份；两端source SHA相同：
`0118e3b85d270114646586804ba7e1ec19551b69efa05b388b07b6db0c0ee167`。
两端adoption记录SHA也一致，未改Git index，未commit/push。

新feature相对前生产版为28行新增/12行删除：原首个steady tile单独展开，
首K128的WMMA C/SRC2使用inline constant zero，后续按原顺序累加。
原input ring、fences、有效WMMA、SiLU及FP32 amax/BF16 RNE/FP4量化规则保持；
没有增加persistent、w2x4或新的pipeline。关闭此guard的配置继续原初始化流程。

实际GEMM1 ELF为`bec431f59ec2501b72f620bf90e24fee75e07a329e2f7c9a54cac43bcf4029ac`，
资源为VGPR762 / SGPR62 / LDS327680 B / private0 / spills0。
GEMM2仍为原完整ELF`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。
正式成绩取首个正向合格候选组中值**85.8308 μs**，另两个确认组中值85.5607、86.2206 μs保留，
不选择更快组或单轮替代正式成绩。距80 μs仍差5.8308 μs，目标尚未达到。

```bash
# 上述固定label已完成；以下采用动作仅适用于仍为前生产SHA且输出尚未生成的树。
python -u .codex_tmp/yadai_gemm1_param_20261021/tools/accept_inlinezero_followup_v1.py
python -u .codex_tmp/yadai_gemm1_param_20261021/tools/adopt_inlinezero_v1.py inspect
python -u .codex_tmp/yadai_gemm1_param_20261021/tools/adopt_inlinezero_v1.py apply
```

继续独立A-only timeout候选：仅preshuffled A payload multicast采用standard timeout，
B/ScaleA/ScaleB及所有地址、workgroup mask、ring、K顺序和hardware fences保持。
硬件依据为Shader Guide rev65 §4.9.8（pp191–192）及§4.10.4 Table80（p203）；
是否降低耗时须由后续实际ISA、严格字节验证及原入口双侧三轮对照确定。
当前只有静态source/AST检查，没有此候选device或性能成绩。

### A-only timeout 编译核验及 seed0 精度（2026-10-22）

组合候选为`ds_xdl_mgroup32_inlinezero_a_timeout_v1.py`，source SHA为
`3f23fb29a23094fa4335acf66f2515d48d7f331450ce16d16ff4983dd9f97b68`，
相对已采用inline-zero只增加9行/删除6行。CPU编译在14.1006秒完成，无GPU dispatch；
实际GEMM1 ELF为`0448a7abc2e9ea92ea1f4dd81ecd7cd9e60f82af1a7c63fcfd4b35c32bc35ef7`，
VGPR762 / SGPR58 / LDS327680 B / private0 / spills0。

原v1检查假定只改变五个literal及branch位移、SGPR保持62；实际LLVM同时重分配
scalar registers、调整部分scalar指令位置，且一个`s_or_b32`改为`s_addk_co_i32`。
该检查报AssertionError并保留原host.log，没有将其标为通过，生产source已恢复。
不能声称其余完整ISA逐条相同。

独立v2核验基于已有实际artifact，不重新编译或覆盖失败记录：去除location后的source IR
仅17行timeout类型变化，六个A payload copy sites的source/destination、extent及mask相同。
实际五个constant definition使六个A load sites由`0x20000f`改为`15`，仅bit21变化；
其余tensor load first descriptor word和TH设置保持。640项static WMMA、786项DS、
336项wait、28项barrier的encoded sequence逐条相同，vector sequence及epilogue尾部相同。
descriptor reaching-write核验只是线性扫描，不作为完整CFG/liveness证明；源IR与device验证一起评估。

`ds_xdl_mgroup32_inlinezero_a_timeout_v1_strict_e64_s0_v2`在19.1470秒完成，rc0。
27项BYTE_CHECK全部为0，完整MoE输出`(1536,7168)`、22020096 B零不同字节，
same-pointer动态graph和原reference通过。实际GEMM1匹配上述组合ELF，GEMM2匹配原ELF。
本轮只覆盖E64/seed0，其他seed/E65及额外scope尚未验证；没有性能数据。

```bash
# 前述CPU编译已完成；v2只读取保留的raw IR/ISA/ELF。
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/inspect_inlinezero_a_timeout_native_v2.py
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_inlinezero_a_timeout_strict_v2.py \
  NEW_A_TIMEOUT_STRICT_LABEL
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/accept_inlinezero_a_timeout_strict_v2.py \
  NEW_A_TIMEOUT_STRICT_LABEL
```

接下来按原benchmark真实inputs生成两阶段原keys，核对实际ABI/ELF后再做
inline-zero→A-timeout→inline-zero完整三轮双侧对照；各child保持120秒上限。
正式生产仍为85.8308 μs的inline-zero版本，未提前采用A-timeout。

### A-only timeout 正向双侧对照完成（2026-10-22）

原benchmark真实inputs的四项两阶段准备和两项original-call key审计已通过。
base/atimeout分别由安装的原launcher生成namespace，stage keys和ABI/signature相同，
实际GEMM1 ELF分别绑定`bec431f5...029ac`和`0448a7ab...bc35ef7`；
没有修改key算法、复制或改名cache artifact。所有GEMM2实际ELF保持原版。

`reboot20261022_inlinezero_atimeout_forward_v1`完成base→atimeout→base：

| 组 | 首个合格try | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---:|---|---:|---:|
| 前侧inline-zero | 1 | 85.2530 /85.9678 /85.3669 | 85.3669 | 63.8885 |
| inline-zero + A-only timeout | 1 | 84.7597 /85.3287 /84.9116 | 84.9116 | 64.6455 |
| 后侧inline-zero | 2 | 86.7311 /85.0253 /86.0668 | 86.0668 | 63.8061 |

候选对两侧分别改善0.4553 /1.1552 μs，即0.5333% /1.3422%。后侧try1的
post/settled ownership检查不合格，完整三轮组排除，中值null；使用后续首个完整合格组，
没有跨组拼轮。这只是正向初筛，正式采用版和正式成绩85.8308 μs保持。

parent报告completed/source_restored，退出后独立核对production SHA恢复为
`0118e3b85d270114646586804ba7e1ec19551b69efa05b388b07b6db0c0ee167`。
06:20 UTC主机检查KFD owners为空、GPU busy0、VRAM176582656 B，未发现本项测试残留。
branch/HEAD和boot_id保持。

完整270成员、1952304 B证据包已下载，archive SHA256为
`dd61db7b0b5f40bc56b09b9b6dc99adf1cc4caef98270bb952cbaba128cc643b`。
本地`accept_inlinezero_a_timeout_screen_v2.py`重新计算原表、ownership、actual ELF、
ABI/key及27项完整字节检查后通过，明确保留v1 exact-ISA预检查失败和v2 scalar重分配证据。
没有把完整ISA变化描述成只有literal变化，也没有将CPU-only核验当性能证据。

随后启动反向`atimeout→base→atimeout`。正反向四项相邻比较均有收益后，
再补seed17、E65原fallback及候选自身非目标配置检查，全部通过才考虑采用。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_normal_inlinezero_a_timeout_v2.py \
  NEW_ATIMEOUT_REVERSE_LABEL atimeout base atimeout
timeout --signal=TERM --kill-after=5s 120s python3 -u \
  .codex_tmp/yadai_gemm1_param_20261021/tools/seal_inlinezero_a_timeout_screen_v2.py \
  NEW_ATIMEOUT_REVERSE_LABEL
python -u .codex_tmp/yadai_gemm1_param_20261021/tools/accept_inlinezero_a_timeout_screen_v2.py \
  NEW_ATIMEOUT_REVERSE_LABEL
```

同时仅在Windows本地完成M-group64/128候选静态准备。每个相对已采用inline-zero仅
替换一行`TILES_PER_GROUP=32`为64/128；实际AST枚举各177232个映射点，覆盖group边界、
capacity tail及N-cluster peers。每个capacity M/N tile恰好一次，cluster peers仍同M。
这些候选尚未上传、编译、GPU执行或测性能，静态检查不表示有收益。

### A-only timeout 反向确认完成，继续完整兼容性验证（2026-10-22）

反向`reboot20261022_inlinezero_atimeout_reverse_v1`三组均为try1合格，所有原precision
gate及pre/post/settled ownership、actual ELF绑定通过：

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前侧A-only timeout | 85.4006 /84.8249 /85.2573 | 85.2573 | 64.3122 |
| 中间inline-zero | 86.7329 /85.8896 /85.3353 | 85.8896 | 64.5672 |
| 后侧A-only timeout | 85.2866 /85.8179 /85.3416 | 85.3416 | 64.1138 |

反向相邻提升0.7362% /0.6380%，与正向合并后四项全部为正：0.5333%、1.3422%、
0.7362%、0.6380%。三个合格候选组中值84.9116、85.2573、85.3416 μs全部保留；
若兼容性通过而采用，正式成绩仍用首个正向候选组中值84.9116 μs，不选择最快单轮。
当前正式生产仍为inline-zero的85.8308 μs，尚未提前合入。

反向234成员、1824671 B证据包已下载，archive SHA256为
`13999d1ad436ccec13bc4b2be2eb9b838eb2474b5314bf965effed5ef600f2f4`。
Windows本地独立复算通过，`compare_inlinezero_a_timeout_sequences_v1.py`合并记录
确认四项相邻收益、原表及key/ABI/ELF绑定。parent退出后production SHA恢复，
06:27 UTC主机KFD owners为空、busy0、VRAM176582656 B，没有本项GPU测试残留。

已开始E65两种source下的原始两阶段独立准备，之后执行E64/seed17、E65/seed0/17
完整MoE和same-pointer动态graph字节检查，再编译候选自身的八项非目标配置。
E65继续原t64/w1x4 fallback；不把本次等价验证解释为E64/E65共用t256或固定shared expert1。
所有child保持120秒TERM、5秒KILL，compile/verify输出耗时不记为性能。

```bash
python -u .codex_tmp/yadai_gemm1_param_20261021/tools/compare_inlinezero_a_timeout_sequences_v1.py
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_inlinezero_a_timeout_e65_preparation_v1.py
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_inlinezero_a_timeout_validation_v1.py
```

### A-only timeout 完整验证、独立验收及采用（2026-10-22）

E65原输入的production/atimeout×GEMM1/GEMM2四项独立准备全部通过，child分别为
11.9153、11.3859、11.5132、11.3536秒；原stage keys与两阶段完整ELF等价，source恢复。
E65仍为原t64/w1x4 fallback，未验证固定shared expert1 routing。

E64 seed0/17、E65 seed0/17四项完整strict验证全部通过：每项27个BYTE_CHECK均为0，
完整MoE输出`(1536,7168)`、22020096 B零不同字节；same-pointer动态graph和原reference
保持。候选自身八项CPU direct compile也全部通过与control的完整ELF逐字节比较：
GEMM2、FP8 input、original A、pin1、t64 fallback、w1x4、buffers3、persistent1。
各compile/verify child均在120秒内结束，没有放宽任何精度或超时标准。

追加验证证据包178成员、1538088 B，archive SHA256为
`6ac00e2a6fb89ec9740b5a9f478c7fef77d9c965a11c69f8fc2927705ec18f18`。
下载后本地`accept_inlinezero_a_timeout_followup_v1.py`逐成员复算通过，原v1失败及
v2实际scalar重分配的限制保留。采用前正反向原始计时、全部字节检查和非目标ELF通过。

`adopt_inlinezero_a_timeout_v1.py inspect/apply`分别在Windows和a07-3执行，
两端production source SHA256一致为
`3f23fb29a23094fa4335acf66f2515d48d7f331450ce16d16ff4983dd9f97b68`。
adoption记录SHA256也一致为
`05a81652b06082bca77de6dc7e0a277ace029920a552f7b7167b5688a05ebd11`。
前inline-zero源码另存备份；未改Git index、commit/push或benchmark/wrapper。

本feature相对inline-zero为9行新增/6行删除，当前GEMM1文件相对HEAD为46行新增/10行删除。
仅A payload multicast改为standard timeout，workgroup mask、地址、input ring、K累加
顺序、数值、activation/quant和hardware fences保持，未加入新的persistent、w2x4或pipeline。
实际GEMM1完整ELF为`0448a7abc2e9ea92ea1f4dd81ecd7cd9e60f82af1a7c63fcfd4b35c32bc35ef7`；
资源VGPR762 / SGPR58 / LDS327680 B / private0 / spills0。
GEMM2完整ELF保持`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。

正式成绩采用首个正向合格候选组中值**84.9116 μs**，另外85.2573、85.3416 μs两个
确认中值保留。距80 μs仍差4.9116 μs，尚未达标；不使用最快单轮更新成绩。
接下来将M-group64/128的一行改动重新建立在已采用A-timeout上，原inline-zero的静态
候选继续保留但不冒充当前组合候选。

### A-timeout 上的 M-group64/128：native 和 seed0 strict 完成（2026-10-22）

组合候选各只将当前guard内`TILES_PER_GROUP=32`改为64/128，一行新增/一行删除。
每个实际AST枚举177232个mapping points通过，原功能、shape guard、load/store、
K顺序和fences保持。两者已同步a07-3，source SHA分别为
`59d5e0a1d42420593229ba0865bd98942af06ae37eee8c8645bd1f95fe82cf8f`、
`0b5b04267431c2e6d43332d40737234a66d01b6d0437984a95ab83b693daf0c6`。

两项CPU-only compile/native检查通过，资源VGPR762 / SGPR58 / LDS327680 B / private0 /
spills0均保持，static instructions各6158。WMMA/BF16/FP4/tanh计数、protected memory
operations和ordered fences保持；这是有限native检查，不替代完整硬件hazard证明。
GEMM1完整ELF分别为`dfeb0ee6943165535b8e38067a9705cbff76676762b85efe6e2f04e3e8f8c716`、
`7b05042032f25564525fb355c450ebc0bd7118025378298864a4c68ab8913d82`。

两项E64/seed0 strict均通过，child分别19.2006、19.3340秒，各27 BYTE_CHECK为0，
完整MoE22020096 B和same-pointer动态graph零差异，原reference通过。
GEMM1实际ELF绑定上述native结果，GEMM2保持原完整ELF。
尚未获得性能，未采用；正在准备原benchmark真实stage keys及read-only cache审计。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_atimeout_mgroup_cpu_v1.py
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_atimeout_mgroup_strict_v1.py 64
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_atimeout_mgroup_strict_v1.py 128
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_atimeout_mgroup_keys_v1.py 64
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_atimeout_mgroup_keys_v1.py 128
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_normal_atimeout_mgroup_v1.py \
  64 NEW_MGROUP64_LABEL base mgroup base
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_normal_atimeout_mgroup_v1.py \
  128 NEW_MGROUP128_LABEL base mgroup base
```

以上fixed compile/strict已完成，不覆盖既有输出。正式计时保持原用户命令、完整三轮双侧
对照、全机idle和120秒child上限；编译准备输出不作性能。正式成绩仍84.9116 μs。

### A-timeout 上的 M-group64：双侧回退，不采用（2026-10-22）

M-group64/128的两source×两stage原输入准备和read-only original-call审计均完成。
每种source独立取得真实namespace，实际两阶段ABI/signature与control相同，未修改JIT key
算法、复制或改名cache artifact。stage keys分别为`62c9ae1cec751fe7`、`da3c5914af92622d`；
group32/group64/group128的manager keys分别为`a691b9dc736c8d1c77366a19cc46493c`、
`2b2a78d45eb3594b97b0b580ab9f99e6`、`5360b5deefdb7d0ef7cca299e77b3c54`。

`reboot20261022_atimeout_mgroup64_forward_v1`已完整结束。候选try1的前两轮post
ownership不合格，整组三轮排除、中值null；按6秒全机idle gate等候后重新完成try2。
raw记录包含短暂外部`lipiwang`任务和已退出KFD entries，不据此断言所有计时阶段都并发。
只有下列完整合格三轮参与比较，原precision gate、actual G1/G2绑定均通过：

| 组 | 合格try | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---:|---|---:|---:|
| 当前group32 before | 1 | 85.4407 /85.1433 /85.4342 | 85.4342 | 63.8273 |
| group64 | 2 | 85.8476 /85.1390 /85.7464 | 85.7464 | 64.0718 |
| 当前group32 after | 1 | 86.0416 /85.1981 /85.4153 | 85.4153 | 64.1139 |

group64分别慢0.3122/0.3311 μs，即0.3654%/0.3876%，没有双侧收益。
不追加反向计时或seed17/E65/scope扩展，不进入生产。已做的E64/seed0完整MoE及dynamic
graph字节等价保持有效，但不把这项初筛解释为全面shape验证。

247成员、1186077 B证据包已下载，archive SHA256为
`2d09f6385677d5b7f12544d000abcef818be2c2ca14d0933adaf448e1e0f78e2`。
本地`accept_atimeout_mgroup_screen_v1.py`重新验证原表格、三轮中值、pre/post/settled、
original keys/ABI/实际ELF、native资源和27项strict字节检查，`passed=true`；
`positive_at_every_adjacent_control=false`、`production_adopted=false`明确保留。
退出后production source恢复`3f23fb29...97b68`，主机GPU/KFD无残留。
当前正式成绩仍84.9116 μs；随后启动group128的双侧三轮对比，不复用group64的性能。

```bash
timeout --signal=TERM --kill-after=5s 120s python3 -u \
  .codex_tmp/yadai_gemm1_param_20261021/tools/seal_atimeout_mgroup_screen_v1.py \
  64 NEW_MGROUP64_LABEL
python -u .codex_tmp/yadai_gemm1_param_20261021/tools/accept_atimeout_mgroup_screen_v1.py \
  64 NEW_MGROUP64_LABEL
```

计时期间只在Windows本地准备下一项A-payload-only TH2/4/6候选，各增加一行
`cache_modifier=... if standard_a_timeout else 0`，其他TDM、mask/address、timeout、
input ring、K顺序与hardware fences保持。真实add_tdm_loads call AST检查通过，
移除该keyword后完整module AST相同；尚未CPU/native/device或性能验证。
Shader Guide rev65 §4.2.1 Table19（p62）记载TH2/4/6的缓存策略，§4.9.8（p191）
记载multicast强制绕过WGP cache；仅据这些资料无法预测收益，后续必须实际测量。

### A-timeout 上的 M-group128：双侧回退，不采用（2026-10-22）

`reboot20261022_atimeout_mgroup128_forward_v1`完成完整9轮，全部原precision gate、
pre/post/settled ownership和实际ELF绑定通过，没有排除组：

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 当前group32 before | 85.4955 /84.6394 /85.2511 | 85.2511 | 63.3084 |
| group128 | 90.5555 /88.9279 /89.6934 | 89.6934 | 64.5314 |
| 当前group32 after | 85.6052 /84.4414 /85.7126 | 85.6052 | 64.6960 |

group128相对两侧慢4.4423/4.0882 μs，即5.2108%/4.7756%，放弃。
不追加反向或扩展兼容性运行，不进入生产。原完整MoE/seed0和same-pointer dynamic graph
初筛正确性有效，但不能声称已补seed17、E65或candidate自身outside-guard编译。
原group32两侧中值只作这项局部对照，不用其最快单轮替换84.9116 μs正式成绩。

211成员、1056731 B证据包已下载，archive SHA256为
`661f960ee5a24d656765c88b09a47580f8963a7eec3afd7944c3eee436b20898`。
本地独立复算原表、中值、ownership、key/ABI/ELF、native和27项strict通过；
`positive_at_every_adjacent_control=false`保留。退出production SHA恢复，主机GPU/KFD无残留。
M-group64/128均未获收益，继续保留group32。

随后启动A-only TH2/4/6的CPU-only编译和有限native检查，尚未GPU验证或测性能。
每项固定compile/inspection child仍限120秒TERM、5秒KILL。计时、precision门限、
GEMM2和其他kernel保持，candidate快照不覆盖当前正式源码。

### A-only TH2/4/6：原始 IR/编码检查及首项精度（2026-10-22）

三个候选各仅在preshuffled A payload的`add_tdm_loads`增加一行
`cache_modifier=2/4/6 if standard_a_timeout else 0`。原guard、A extent/pitch/mask、
A standard timeout、B/ScaleA/ScaleB、四级ring、K累加顺序和hardware fences保持。
CPU-only compile及有限native检查均通过，VGPR762 / SGPR58 / LDS327680 B /
private0 / spills0、6158 static instructions保持。未改变生产源码或正式成绩。

`inspect_atimeout_a_th_native_v1.py 2`的首项检查因函数参数中inline source location
路径不同失败，原失败日志保留；v2补充仅去除inline路径location后，TH2/4/6均通过。
去除location的原始IR只出现`cache=0→2/4/6`，六个实际A copy sites均保留原
extent/pitch和`timeout=false`。24个native `tensor_load_to_lds`中A ordinals
`0,4,8,12,16,17`的TH bits按预期改变，其余TDM TH不变。descriptor SGPR位置可改变；
这是原IR和选定编码检查，不是完整ISA逐字节等价或完整CFG/hazard证明。

TH2 E64/seed0 strict已通过，child19.1997秒：27项BYTE_CHECK全部零差异，
完整MoE输出`(1536,7168)`、22020096 B零不同字节，same-pointer动态graph和原
reference通过。actual GEMM1 ELF为
`4a6cea5be55919c4ec4986b2692012cabfc35d354612ca15e54214a28026f2cc`；GEMM2
保持`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。

TH2两source×两stage原输入keys准备和read-only审计也通过；stage keys保持
`62c9ae1cec751fe7`、`da3c5914af92622d`，候选manager key为
`eac16563772c9bc6cc9fae5718c9e6e5`，control仍`a691b9dc736c8d1c77366a19cc46493c`。
原JIT key算法未改，未复制/改名cache artifact，准备后production source恢复。

已启动`reboot20261022_atimeout_a_th2_forward_v1`的原入口双侧三轮对照；此处尚无
性能结论。TH4/6尚未GPU strict。正式成绩继续84.9116 μs，不把上述compile/verify
耗时或未完整验收的单轮写为新成绩。所有child仍120秒TERM、5秒KILL。

```bash
# 在a07-3主机调用；runner在dsr1_ep4执行验证和未修改的原benchmark。
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_atimeout_a_th_strict_v1.py 2
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_atimeout_a_th_keys_v1.py 2
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_normal_atimeout_a_th_v1.py \
  2 NEW_TH2_LABEL base mgroup base
```

固定strict/keys已经完成，不覆盖既有输出；新perf使用新label。`mgroup`只是复用runner的
实验mode名称，本实验未修改M-group。最终性能证据需按A-only source/IR/native核验，
不能套用M-group的一行32→64断言。

### A-only TH4/6 精度和全部原入口准备完成（2026-10-22）

TH4、TH6的E64/seed0 strict分别在19.6089、19.3619秒完成，rc0。
两项各27个BYTE_CHECK均零差异；完整MoE22020096 B、same-pointer动态graph和
原reference通过。实际GEMM1 ELF分别为
`9dc576a23ad96cf1dea14f171f804345ef2b246f72e8875faba7f526d7cd987d`、
`b68b1792878874ba7ecc40dd6615626085ca7530dd2da86a3f26acb38d16515b`，与原native
检查绑定；GEMM2仍为原完整ELF。seed17/E65/outside-scope尚未扩展，性能初筛若无收益
则不继续追加这些运行。

TH4/6两source×两stage原call keys准备和read-only audit全部完成，stage keys与
control相同；候选manager keys分别为`2b1a21acd9257ccda8034aa355c23e0e`、
`5f435d07e778ec78b1a3f94532d31b5b`。没有修改JIT算法或向cache复制artifact。
所有固定child仍在120秒内完成，编译/精度/审计输出耗时不作为性能。

TH2正式计时因外部GPU工作负载等待全机idle。为补独立TH4/6准备，仅暂停自己处于
`wait_normal_idle_v1.sh`的host parent/等待子进程，检查确认当时没有timing child；
没有暂停或修改kernel测试。准备结束恢复parent和6秒idle gate，pause/resume记录保留。
本轮此处尚无正式timing样本，生产源码SHA仍`3f23fb29...97b68`，正式84.9116 μs保持。

已准备A-only性能证据sealer/local acceptor，新增原IR和ISA入包，并从下载原文件独立
复算六个A copy sites、24个native TDM的TH字段，保留原v1 location失败日志。
原表格、precision gate、完整三轮中值、ownership、keys/ABI/ELF检查仍保留。
该工具的静态组装检查不是device或性能证据，尚未用它声称任何新收益。

### A-only TH2 首次计时被 control fault 中断（2026-10-22）

`reboot20261022_atimeout_a_th2_forward_v1`在08:13:52 UTC的前侧control第三轮发生
`Memory access fault by GPU node-2 ... on address 0x32c0000. Reason: Page not present or supervisor privilege.`，
child rc134。当时仍是正式版，TH2候选没有开始计时。前两轮85.0232/85.6431 μs不能
拼成完整组或作为新的正式成绩；整项尚无性能结论。runner立即停止并恢复production
source SHA `3f23fb29...97b68`，本任务parent和GPU child已退出，没有继续TH4/6计时。

故障日志已输出GEMM1/GEMM2 profiler表，但尚未完成原reference及最终artifact audit。
pre日志最后三次全机idle合格；post/settled出现外部`dev_al`的
`python3 /tmp/fallback_diff/bench.py ...`。这说明测试后存在外部占用，不能据此把fault
唯一归因于并发或任何特定kernel。原失败日志、完整pre/post/settled和summary均保留。

新`run_atimeout_a_th_fault_host_v1.py`按正式版→原提交版各做一次限定诊断：
在原三个`run_perftest`、两次FP4 quant和两阶段PyTorch reference的调用前后增加
同步及阶段标记，不改变数值、输入、原reference门限；所有输出计时排除。
两项分别7.7159/17.9015秒、rc0，全部阶段和原精度通过。
实际GEMM1完整ELF分别为`0448a7ab...bc35ef7`/`be083c8c...43ad052`，GEMM2均为
`4bd626e3...b716be`。生产源码保持，未修改benchmark、wrapper或tracked源码。

本次诊断未复现fault，`fault_cause_established=false`；增加同步后通过不能证明
异步风险已修复。随后使用新label恢复未修改的原入口完整双侧三轮，保留120秒
child限制和6秒全机idle gate，不跨失败序列拼样本，再次执行失败仍立即停止。
正式采用版仍84.9116 μs，TH2/4/6均未采用。

### LDS 对齐空隙：完整重排不采用，B-only 候选通过编译（2026-10-22）

本轮在现有 GEMM1 geometry guard 内尝试消除 `PLANAR_B_OFF` 前的 48 KiB 空隙。
没有复用 input ring 槽，没有修改 pipeline、K 累加顺序、TDM extent/pitch 或同步。
Shader Guide rev65 §3.3.4、§4.7.1（p160）记载 LDS/WGP$ 共享 384 KiB、按 64 KiB
分区，WG 的 LDS 按 2048 B 分配。272 KiB 仍跨五个 64 KiB 分区，不能据容量变化
宣称 occupancy 或 WGP$ 容量提升；§4.10.2（p199）规定的 global stride 也不能
作为 LDS destination padding 使用。

首个 `ds_xdl_inlinezero_atimeout_compact_planar_v1` 将 SA/A/SB/B 改排为 A/B/SA/SB，
源码仅 +6/-1 行，9216 个 allocation AST case 通过。编译 raw LDS 为 278528 B，
但 VGPR762→763、static instructions6158→6239；`ds_load_b32` 增加16、
`ds_load_2addr_b32` 减少8，`s_wait_dscnt` 增加15、`s_wait_alu` 增加18。
原 native gate 的访存 opcode 数量及 VGPR 检查失败，原失败记录保留，未执行 GPU 验证。
ScaleA base 高位变化后部分 paired load 被拆分；本版不继续。

随后改为 `ds_xdl_inlinezero_atimeout_compact_b_v1`，仅移动 B base：

| 项目 | 正式版 | B-only 候选 |
|---|---:|---:|
| A / ScaleA / ScaleB base（B） | 8192 / 0 / 139264 | 8192 / 0 / 139264 |
| B base（B） | 196608 | 147456 |
| raw LDS（B） | 327680 | 278528 |
| VGPR / SGPR | 762 / 58 | 757 / 58 |
| private / VGPR spills / SGPR spills | 0 / 0 / 0 | 0 / 0 / 0 |
| static instructions | 6158 | 6123 |

源码 +3/-1 行，guard 为 `standard_a_timeout and planar_lds_on and not tdm_as_in_prologue`。
9216 个实际 allocation AST case、557056 个逐 byte 相对地址映射通过；A/ScaleA/ScaleB
base、四个独立槽、槽内 offset 和所有 size/stride 保持，guard 外 allocation 相同。
CPU compile child18.0757秒完成、rc0，有限 native gate 全部通过：数学及访存 opcode 数量、
选定 TDM control fields、关键 lifetime fences 顺序、branch 方向顺序保持，raw LDS
与 CPU allocation 一致。尚未完成全部 address/CFG/hazard 证明，以上不是 device 或性能证据。

候选 source SHA256：`73e32c32f73aef55ca16526e3071dae085c14cb08e35b83f4f37a53e2e6e5e6c`；
完整 ELF：`a447f00c2f48d9611710c77bb0569cc13f2d10056d546e0bff55f7a7d5f8f8f5`。
raw source IR、lowered IR、ELF、ISA、resources 和验收记录已下载，完整 ELF SHA 已在本地复算。
与正式版相比，`s_wait_alu`240→219、`s_set_vgpr_msb`955→948，另有地址计算和调度变化；
不能把这些 static counts 直接换算为 kernel 耗时收益。

本候选编译结束后，08:38:02 UTC 已停止本任务的额外只读 observer，并恢复 TH2 v2 的
host parent2064446及 idle waiter2065058，记录 `atimeout_a_th2_wait_resume_v3.json`。
TH2 v2 首轮原 reference 通过，但 post/settled ownership 不合格，`valid=false`；
不跨组拼样本，不把该轮85.1117 μs计为正式成绩。持续按6秒 idle gate 等待并完成整个序列。
正式生产 source 仍 `3f23fb29...97b68`，正式采用版仍84.9116 μs。

```bash
# a07-3 主机运行；固定 compile/inspection child 在 dsr1_ep4，限120秒 TERM、5秒 KILL。
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_atimeout_compact_b_cpu_v1.py
```

固定输出已存在，上述命令仅记录复现流程；重新运行需新 label，不能覆盖原证据。
B-only 候选 GPU strict、原入口 key 准备和双侧性能对照仍待完成，尚未进入生产。

### B-only LDS 候选完整 MoE 初筛和原入口准备通过（2026-10-22）

`ds_xdl_inlinezero_atimeout_compact_b_v1_strict_e64_s0_v1` 在19.0081秒完成、rc0。
27项 byte check 零差异；完整 MoE `(1536,7168)` 的22020096 B零差异，same-pointer
动态 graph 和原 reference 通过。实际 GEMM1完整 ELF仍 `a447f00c...5f8f8f5`，
GEMM2仍 `4bd626e3...b716be`。这仅覆盖 E64/seed0 初筛，尚未追加 seed17/E65/outside-scope。

两 source×两 stage 原 call 编译准备与 read-only key audit 全部通过，实际 stage keys仍为
`62c9ae1cec751fe7` / `da3c5914af92622d`；候选 manager key为
`9086f8acd5fae2b515ff5d1a26f37c31`。原 JIT算法、ABI、benchmark未修改，未复制或重命名cache artifact。
退出后正式生产 source恢复 `3f23fb29...97b68`。

45成员、178502 B准备包已下载，archive SHA256：
`596bf5de3c67960bf7693a1e757a9ffd2bcc3abc018155d5adf401b04808af26`。
本地从原 precision log、完整 ELF、stage preparation和 read-only audit独立复算通过，
记录 `atimeout_compact_b_preparation_v1.local_acceptance.json`，不使用准备运行的耗时作为性能。

source IR在去除source location后只出现53行变化：1行 `allocBytes327680→278528`，
52行B base/constant `196608→147456`；825个 `fly.copy` site一致，其中24个global→shared
TDM site及1个shared→global TDM site一致，六个A payload site的extent/pitch/cache/timeout保持。
`inspect_atimeout_compact_b_ir_v1.py`误将全部source copy site数当作24个native tensor load数，
原失败保留；v2按实际825/25/24/1分别检查后通过，没有修改kernel或放宽数值门限。

08:47:59 UTC已停止额外observer并恢复TH2 v2 parent及idle waiter，
`atimeout_a_th2_wait_resume_v4.json`记录保留。TH2 v2首组三轮整体ownership不合格、中值null；
随后从新完整组开始，不能将被排除组的合格单轮拼入新组。TH2、TH4、TH6、B-only LDS的有效
完整性能对照仍待完成，正式采用版继续84.9116 μs。

```bash
# 固定strict/key输出已经完成，仅记录复现步骤；新实验必须使用独立label。
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_atimeout_compact_b_strict_v1.py
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_atimeout_compact_b_keys_v1.py
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_normal_atimeout_compact_b_v1.py \
  NEW_COMPACT_B_LABEL base mgroup base
```

`mgroup`仅为复用runner的mode名称，本候选没有修改M-group。计时仍执行原`--scenario kernel`
命令，全机idle后运行；每个child限120秒TERM、5秒KILL，外部占用按6秒继续等待。

### A-only TH2：完整双侧对照略回退，不采用（2026-10-22）

`reboot20261022_atimeout_a_th2_forward_v2` 已完成，两侧 control各排除一组三轮后，
取得以下完整有效组；原precision gate及pre/post/settled ownership通过：

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 正式版 before | 85.4071 / 86.0310 / 84.4602 | 85.4071 | 63.6353 |
| A-only TH2 | 85.7252 / 85.6177 / 85.2131 | 85.6177 | 63.7367 |
| 正式版 after | 85.4575 / 85.4724 / 85.5313 | 85.4724 | 64.2065 |

候选相对两侧慢0.2106/0.1453 μs，即0.2466%/0.1700%，不采用，不追加反向或
seed17/E65/outside-scope运行。两组ownership排除记录及其中的全部样本保留，中值null，
没有拼入有效组。新序列未再出现memory fault；这仍不足以确定v1的fault原因。

299成员、2050203 B证据包已下载，archive SHA256：
`800fa25e4c2a684771f78c8e591e310c02a67059016d890bce1844b5c90bdd67`。
本地独立复算source/原IR/选定TDM编码、原表中值、ownership、keys/ABI/ELF和27项strict通过，
`positive_at_every_adjacent_control=false`保留。退出后production source恢复，任务GPU/KFD无残留。
正式采用版继续84.9116 μs，随后顺序进行TH4、TH6、B-only LDS初筛。

### 原命令首次JIT卡住：FlyDSL source location查询修复（2026-10-22）

用户原命令在默认cache下达到120秒上限、rc124；附带30秒周期CPU stack的独立诊断也
达到同一上限。30/60/90秒的stack均落在`capture_user_location` →
`inspect.getframeinfo(frame, context=0)` → `inspect._get_code_position`，采样位置仍从
mainloop推进到epilogue。这次复现的阻塞位于CPU frontend，而非采样到了GPU barrier等待。
此前任务私有wrapper提前准备了cache，因此没有暴露直接冷启动的这项开销。

Python3.12的原查询逐次从头遍历大函数的`code.co_positions()`，`getframeinfo`还重复读取
之后被exact position覆盖的`f_lineno`。在`flydsl/expr/meta.py`新增有界thread-local
`_frame_file_line_col`：保留code强引用，用`id(code)`缓存position tuple，按`f_lasti // 2`
读取；仅缺失line时查询`f_lineno`。完整文件名、line/column、call-site chain和debug规则保持。
没有全局monkeypatch `inspect`，没有修改compiler options、JIT算法或benchmark。

容器仍为`dsr1_ep4`、distribution仍`0.3.3.dev882`，安装文件为
`/usr/local/lib/python3.12/dist-packages/flydsl/expr/meta.py`，修复源码SHA256：
`fecad25479488d1941d6c5951b3b1a3b3ac3c5c258382ba8a7c9d7f3a9203f33`。
本地`../FlyDSL/python/flydsl/expr/meta.py`按其原版本接口移植相同helper和调用，已统一LF；
没有用dev882整文件覆盖本地不同版本。

| 原命令运行 | wall time秒 | exit |
|---|---:|---:|
| 修复后默认cache首次 | 约24 | 0 |
| 独立空cache首次 | 24.0089 | 0 |
| 默认cache warm1 | 8.3430 | 0 |
| 默认cache warm2 | 8.2651 | 0 |

四次原precision为`logits_diff=3.61055e-06`、`rel_l2=0.00268721`、`pass=True`，原0.01门限保持。
这些运行有外部GPU工作负载，仅用于正确性和进程wall time，不能用其GEMM μs更新性能纪录。
394个live frame、0/1/8/32深度chain、192个code object的eviction、31768 B大code的所有
position和两个线程隔离验证通过，默认及`-X no_debug_ranges`均通过。

原benchmark两stage实际call重新推导cache keys并提取完整ELF：GEMM1仍
`0448a7abc...35ef7`，GEMM2仍`4bd626e3...b716be`，逐字节保持。安装源码改变后工具链
fingerprint正常变化；后续性能runner重新准备当前toolchain namespace，不改旧记录，
不复制或重命名旧cache artifact来绕过失效。

58成员、751224 B证据包已下载，本地从原失败stack、raw precision、source和完整ELF
独立复算通过，archive SHA256为
`f02f221257d85cd13b81d9fd0f3b94d82ecf22d6d20f63a0b94ab92809e8f9c3`。
派生验收为`reported_hang_frame_location_fix_v1.local_acceptance.json`；详细修复说明见
[moe_gemm1_flydsl_cold_compile_fix.md](moe_gemm1_flydsl_cold_compile_fix.md)。正式成绩仍84.9116 μs。

### TH4旧序列中断与修复后独立序列（2026-10-22）

旧`reboot20261022_atimeout_a_th4_forward_v1`仅前侧control完整有效，中值85.5862 μs。
候选try1 ownership不合格，整组三轮排除；try2中出现environment source hash断言及
finally source保护断言，完整序列没有完成。事后文件hash回到预期，但无法确定失败瞬间
的变更来源；不能称为已定位或已修复的kernel bug，也不能跨新旧序列拼接样本。

按精确临时候选source hash恢复正式源码，原失败summary保持。101成员失败证据包已下载
并校验，archive SHA256为
`5fbcf0b4cdc8e797257c105380bfcb4bf74db98927f89b409d458f4e4b19b8cc`。

新`reboot20261022_framefixed_th4_forward_v1`使用修复后的安装FlyDSL，重新从两source、
两stage的实际call准备独立cache，GEMM1/GEMM2完整ELF与原冻结候选对应。计时仍运行用户
原命令、warmup5/iters20、A preshuffle1/pin0，每组三次、base→candidate→base，保留
pre/post/settled全机ownership门禁和120秒child上限。TH4只新增一行A payload
`cache_modifier=4 if standard_a_timeout else 0`；尚未采用。

```bash
# a07-3主机启动；固定label已开始，复现时必须改用新的独立label。
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_frame_fixed_small_feature_v1.py \
  th4 NEW_FRAME_FIXED_TH4_LABEL
# 随后依次完成th6和compact_b，不并发性能或额外GPU检查。
```

09:54 UTC全机检查发现外部`dev_al`任务，host parent继续每6秒等待；没有杀外部进程，
没有将等待时间当作测试超时。TH4/TH6/B-only LDS仍待完整性能结论。

### ScaleA / ScaleB 独立 TH2 小候选：native 与完整 E64/seed0 初筛通过（2026-10-22）

两个候选各增加一行 `cache_modifier=2 if standard_a_timeout else 0`，分别仅作用于
原 rotating ScaleA call 和 ScaleB call；复用已验收的 GEMM1 geometry guard。
没有改变 TDM extent/pitch、owner、mask、timeout、四个独立 input 槽、K 累加顺序或
输出量化规则。生产源码仍为 `3f23fb29...97b68`，尚未采用这两个候选。

| 候选 | source SHA256 | GEMM1 完整 ELF SHA256 |
|---|---|---|
| ScaleA TH2 | `9c7a1586f4d540f6df9468cdf80b0d049e1c48ca6872b0fd8d93ef19f1693f0a` | `be4ca7ae94a32b5dba627fb05a2e40b015c5f73af3417d164965badfae6bbb29` |
| ScaleB TH2 | `04f7118645d57431509bdce8edff0ed256768af9036158bd784b1b0237ea4771` | `7a6a155d27d1cac9f5a9a8efa299404e1303181a7dae39a3deb7a4ae047e170f` |

两项 CPU compile 成功，资源均为 VGPR762 / SGPR58 / LDS327680 B / private0 / spills0。
native v1 检查器遗漏函数参数的 inline source location，检查失败记录保留；v2 只补充
location normalization，复用产物检查通过，没有重编译 kernel。normalized source IR
各16行仅 `cache=0→2`，其中六个 global→shared copy site；24个 native tensor load 中，
SA ordinals2/6/10/14/20/21、SB ordinals3/7/11/15/22/23 的 TH 字段按预期变化。
数学/访存 opcode 数、关键 lifetime fences 和资源保持。以上是有限 static gate，
不作为完整 CFG/hazard 证明或性能证据。

两项严格验证顺序在当前 `dsr1_ep4` 内完成：ScaleA child36.7423秒、ScaleB child37.2924秒，
rc0，均在120秒上限内。每项27个 byte check 全部零差异，完整 MoE `(1536,7168)`
的22020096 B零差异，同 pointer 动态 graph 和原 reference 通过。
`logits_diff=3.6105513278025183e-06`、`rel_l2=0.0026872099842876196`，原0.01门限保持；
GEMM2完整ELF仍 `4bd626e3...b716be`。GPU有外部负载，这些运行仅用于正确性，
其耗时不作为 GEMM 性能。初筛仅覆盖 E64/seed0，未重复已采用版的全部验收，
未对候选追加 E65/seed17/outside-scope；有稳定计时收益后再继续完整兼容性验证。

```bash
# a07-3主机顺序执行；固定label已完成，复现需要独立新label，避免覆盖证据。
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_scale_tdm_th_strict_v1.py sa
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_scale_tdm_th_strict_v1.py sb
# 后续原入口双侧三轮，仍是本文原命令、warmup5/iters20、A preshuffle1/pin0。
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/scale_feature_screen_v1.py \
  run scale_sa_th2 NEW_SCALE_SA_LABEL
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/scale_feature_screen_v1.py \
  run scale_sb_th2 NEW_SCALE_SB_LABEL
```

`scale_feature_screen_v1.py`仅适配 plan、native-v2 与记录字段；实际性能 runner 的所有函数、
计时命令和环境 AST 保持，CPU dry transform 验证通过。未复制 cache artifact 或修改 JIT。
Scale strict 完成后已于10:15 UTC恢复 TH4 的 host parent2670391和 idle waiter2679959；
仍每6秒等待全机空闲。正式最优继续84.9116 μs，尚无本轮新增有效性能。

### GEMM1 局部 activation sub-batch 候选准备（2026-10-22）

与此前把两个 MX block 合为32对的 row-batch 方向不同，本轮保持每个 scale block 的16对
gate/up 输入，分为8对或4对调用原 `batched_silu_swiglu`，原 helper 的 CHUNK=4不变。
只替换目标 scope 的一处调用，+10/-7行；guard外仍一次调用完整原列表。
不改 shared helper、F32 amax/E8M0、BF16 RNE、native FP4、output stores 或 input pipeline。

两项各24个实际 helper/call AST 的符号检查通过，逐项 clamp、乘法结合、tanh 输入和
tanh 顺序一致，反转 patch 恢复完整 source。候选仍在 `.codex_tmp`，没有改生产：
sub8 source=`808502952f97c1779241940d13d2d6df987fd6e7531f2a5d5e788ebae5c565f7`；
sub4 source=`c9e8096908cc360237dd57696272226c6ea333f8a5cf82329f0b4169ee648163`。
当前先做有限 CPU compile/ISA 检查，尚无 device 或性能结论；每个 child 保留120秒上限。

sub8首次 host runner 有一个未匹配括号，在启动编译前退出；失败日志
`activation_sub8_cpu_v1.host.log` 保留。修正后先做语法检查，再以独立host label执行；
两项CPU compile、原有限native gate和E64/seed0 strict均完成：

| 候选 | static executable instructions | VGPR/SGPR/LDS B | GEMM1完整ELF SHA256 | strict wall秒 |
|---|---:|---|---|---:|
| 当前采用版 | 6158 | 762/58/327680 | `0448a7abc2e9ea92ea1f4dd81ecd7cd9e60f82af1a7c63fcfd4b35c32bc35ef7` | — |
| activation sub8 | 6131 | 762/58/327680 | `ce33dd28b94641bbb9fc6a30270c925e495ca59bc98ca23b7456b49c4aac87cc` | 36.4836 |
| activation sub4 | 6129 | 762/58/327680 | `fc0e570e1605c6cf94efa9fc3a0d46b705768fcd85141a812d112e2d3f65787f` | 35.8880 |

private和VGPR/SGPR spills均0；math opcode数量、访存opcode/offset计数与关键fences保持，
但整个mainloop native文本不保证相同。每项strict27个byte check零差异，完整MoE22020096 B
零差异、动态graph和原reference通过，GEMM2完整ELF保持。静态指令减少尚不能说明μs收益，
未修改计时脚本或放宽精度。两项仅为E64/seed0初筛，尚未E65/seed17/outside-scope验收。

```bash
# 主机顺序启动，CPU/strict child在dsr1_ep4，全部120秒TERM、再5秒KILL。
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_activation_subbatch_cpu_v1.py 8
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_activation_subbatch_cpu_v1.py 4
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_activation_subbatch_strict_v1.py 8
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_activation_subbatch_strict_v1.py 4
# 固定输出已存在，复现需要新的独立label；下一步性能screen同样不覆盖记录。
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/activation_subbatch_screen_v1.py \
  run activation_sub8 NEW_ACTIVATION_SUB8_LABEL
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/activation_subbatch_screen_v1.py \
  run activation_sub4 NEW_ACTIVATION_SUB4_LABEL
```

10:30 UTC已恢复TH4 host parent及idle waiter，`framefixed_th4_resume_after_activation_strict_v1.json`
记录保留。10:32 UTC远端仍有外部owner，尚无TH4有效计时，生产源码未变。

四项准备证据已封包并下载，134个成员、5847863 B；archive SHA256为
`22c925b8148cf5280163a5c1e2a7d73cf43229740eb463ddb3556fbe6d2b56ae`。
本地独立检查源码、native、原call keys及strict原始日志通过，记录为
`framefixed_scale_activation_preparation_v1.local_acceptance.json`，已回传。
以上仍是正确性与产物绑定证据，没有新增有效性能数据。

10:38 UTC已启动单个host顺序队列PID3013277，等待原TH4 parent2670391结束后，
依次完成TH4、TH6、compact B、ScaleA TH2、ScaleB TH2、activation sub8/sub4。
每项先做 `base→candidate→base`，只有相对前后control都改善时才追加反向
`candidate→base→candidate`；每个计时child仍有120秒上限，空闲等待每6秒检查。
队列逐项恢复生产源码、封包，结果还需要本地独立验收；不会自动采用候选。
10:41 UTC仍在等待外部owner退出，正式最优保持84.9116 μs。

```bash
# 当前队列已经启动，不要重复执行；复现需独立label和对应新输出路径。
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_prepared_small_feature_queue_v1.py \
  > .codex_tmp/yadai_gemm1_param_20261021/logs/reboot20261022_framefixed_prepared_queue_v1.host.log 2>&1 &
```

### 当前版本 thread trace 分析准备（2026-10-22）

性能队列等待期间新增 `analyze_bound_gemm1_trace_v1.py`。该工具先核对指定完整ELF SHA，
再按ELF64 `.text` 逐字节检查objdump编码和完整PC范围，最后逐条比对每个capture的
decoded PC/ISA；阶段边界从实际动态指令序列推导，不使用旧版hardcoded code index。
现有DS-only trace的64个active waves及全部阶段cycles/百分比精确复算一致。
把正式最优ELF与旧trace混用时，5879与6158个executable PC不一致，检查按预期拒绝。
准备验收记录为 `bound_gemm1_trace_analysis_readiness_v1.json`；没有启动新GPU工作。

当前最优 `.text` SHA为 `d1ee894b1ea13fc9616cdaf8b85c919aa639ae0916aaf332a36981df55ec8dcb`，
完整ELF仍 `0448a7ab...bc35ef7`。`s_set_vgpr_msb`有955个static site，但这是静态计数，
不能据此判定它占了多少耗时；Shader Guide rev65 §3.3.2.3 p34明确该指令本身不产生
`S_SETREG_B32 MODE`的额外delay。是否值得减少相关指令，仍须匹配版本的动态trace和性能验证。

`bind_best_frame_fixed_trace_v1.py`用于新capture完成后，从该capture的私有cache按
修复后原call记录取出两stage的实际ELF，并核对source/global/key/ELF与decoded ISA。
只有完成既有小候选的有效screen且没有稳定收益时才抓取正式版新trace；尚未运行。
新trace不会下载`.out`，也不会把profiler下的耗时替代原命令的性能数据。

```bash
# a07-3主机，性能队列结束后，使用独立新label。
bash .codex_tmp/yadai_gemm1_param_20261021/tools/run_best_frame_fixed_att_v1.sh NEW_BEST_ATT_LABEL
# 同一source、pin0/apre1的dsr1_ep4中；仅读取已有产物，120秒上限。
timeout --signal=TERM --kill-after=5s 120s python3 -u \
  .codex_tmp/yadai_gemm1_param_20261021/tools/bind_best_frame_fixed_trace_v1.py NEW_BEST_ATT_LABEL
```

10:56 UTC仍是外部`dev_al` warm_sweep的owner和约1.8 GiB VRAM占用，TH4尚未有效计时。
现有host waiter持续每6秒检查；没有新采用的生产改动，正式成绩84.9116 μs保持。

### 修复后七项小候选顺序对照（2026-10-22）

12:53 UTC的host顺序队列已经完成下列正向组。每组是连续完整三轮，只有
pre/post/settled ownership与原precision合格的整组才计算中值；
TH4前侧try1～3等排除组保持null，没有挑选单轮或跨组拼接。

| 候选 | 前侧当前保留版 μs | 候选 μs | 后侧当前保留版 μs | 相对前/后变化 | 结论 |
|---|---:|---:|---:|---|---|
| A-only TH4 | 85.1687 | 86.7438 | 85.3858 | +1.8494% / +1.5904% | 双侧回退，不采用 |
| A-only TH6 | 85.3943 | 86.0246 | 85.2507 | +0.7381% / +0.9078% | 双侧回退，不采用 |
| B-only LDS compact | 86.1364 | 86.9988 | 85.8453 | +1.0012% / +1.3437% | 双侧回退，不采用 |
| ScaleA TH2 | 85.5902 | 85.8391 | 85.2908 | +0.2908% / +0.6429% | 双侧回退，不采用 |
| ScaleB TH2 | 85.7274 | 85.5713 | 85.1834 | -0.1821% / +0.4554% | 前后方向不一致，不采用 |
| activation sub8 | 85.4705 | 85.4703 | 85.5251 | -0.0002% / -0.0641% | 正向极小差异，反向不一致 |
| activation sub4 | 85.2973 | 85.4152 | 85.6173 | +0.1382% / -0.2361% | 前后方向不一致，不采用 |

activation sub8反向三个中值为85.7444 /85.4322 /84.7893 μs；
前侧候选慢于中间control，后侧候选较快，不能将正向0.0002 μs的差异
描述为稳定收益。候选不采用，也不把反向最快一组更新正式成绩。

12:56 UTC最后的activation sub4对照完成：前侧control中值85.2973 μs、
候选85.4152 μs、后侧85.6173 μs，双侧方向不一致，亦不采用。
七项候选及sub8的反向共八个序列全部结束，queue报告completed/source_restored=true，
无error，生产source恢复为原SHA。八份原始archive及receipt已全部下载，2026-10-22
13:00 UTC本地独立验收完成，对应八份`.local_acceptance.json`均为`passed=true`。
验收逐项复算原始precision表、完整三轮中值、pre/post/settled ownership、source/global/key/ELF，
并核对27项byte checks与完整MoE；本地结论与队列报告一致。验收通过表示证据和正确性通过，
不表示候选获得稳定性能收益；本轮七项候选全部不采用。

正式采用版源码仍`3f23fb29...97b68`，正式成绩84.9116 μs，
另外85.2573 /85.3416 μs确认组保留；这些局部control只用于各自相邻比较。
未修改生产kernel、GEMM2、原benchmark或计时方式，未追加无收益候选的
E65/seed17/outside-scope验证。全部compile/verify/performance child限时120秒，
owner等待每6秒检查；未杀外部任务。

队列结束后，在当前采用版重新抓取Method B thread trace，再从capture私有
cache核对原call key、完整GEMM1/GEMM2 ELF和decoded PC/ISA。
新增`seal_best_bound_trace_v1.py`仅封包原trace及绑定所需输入，省略`.out`
和私有cache；采集完成且binding通过之后再执行封包和本地复算。

### 当前正式保留版匹配trace已排队（2026-10-22）

13:01 UTC复核a07-3为`hyg_dsr1_conc6144` / `346438a75506f77b520d3a9a09ba4985f054e377`，
仅保留原三个tracked修改，GEMM1 source SHA仍`3f23fb29...97b68`。
新capture label为`reboot20261022_best_framefixed_att_v1`，主机parent PID3901534，
通过既有`wait_normal_idle_v1.sh`每6秒等待三个连续全机idle观察；外部任务不受影响。
随后在13:03 UTC完成四个SIMD-select采集与解码，13:04 UTC完成原call/private cache/ELF/PC binding。
新trace已经匹配当前保留版；下方补充本地验收与瓶颈结果。

```bash
# a07-3主机；内部host.log与外部driver.log使用不同路径。
bash .codex_tmp/yadai_gemm1_param_20261021/tools/run_best_frame_fixed_att_v1.sh \
  reboot20261022_best_framefixed_att_v1 \
  > .codex_tmp/yadai_gemm1_param_20261021/logs/reboot20261022_best_framefixed_att_v1.driver.log 2>&1
```

八个性能序列的本地复算使用`accept_frame_fixed_screen_v1.py`、
`scale_feature_screen_v1.py accept`及`activation_subbatch_screen_v1.py accept`，
所有验收child均限制120秒；没有重跑已完成的性能或正式版正确性验收。

### 正式最优版trace本地验收与DS scheduling下一项（2026-10-22）

`reboot20261022_best_framefixed_att_v1`已完成240成员、7033974B原trace证据的
本地SHA校验，未下载`.out`。实际两stage ELF和原call keys均通过绑定。
本地独立复算64个active waves，与远端全部phase/opcode/wait/wave统计精确一致；
仅对Windows/Linux路径分隔符规范化68个路径字段，没有改变统计或放宽ISA检查。

当前mainloop占累计wave timestamp gaps的71.3496%，prologue12.7827%，
activation/quant/输出barrier12.8995%，最后input drain0.2048%。
`ds_load_b128`和`s_wait_dscnt`的opcode gap分别15.4625%/4.9799%；
这些归因包含arbitration/coexecution，不能直接等同关键路径耗时或预计提速。
完整分析及复现见`moe_gemm1_hyg_dsr1_best_thread_trace_analysis.md`。

基于匹配trace准备三个独立私有候选，在当前XDL queue/M-group32/inline-zero/A-timeout上
分别修改`sched_ds_first=8`、`sched_ds_first=24`或closing WMMA hint24。
前两个各+5/-1行，后者+1/-1行；不改ring、fence、计算、wave布局、persistent、量化规则或shape支持。
source反向替换与原bytes相等，实际hint总量56项组合检查通过，生产文件保持。

13:14 UTC首项`best_dsf8_v1` CPU/ISA检查通过，VGPR757、SGPR58、LDS327680B、
spills/private0，static instructions6205。protected DS/TDM/output数量/offsets、
ordered hardware fences、WMMA/BF16/FP4/tanh counts与正式版相同。
该检查不等于完整硬件hazard证明；随后开始完整MoE和动态27项byte checks，尚无性能结论。

```bash
# 主机逐项准备；每个容器compile/verify child限制120秒。
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_best_ds_schedule_v1.py cpu dsf8
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_best_ds_schedule_v1.py native dsf8
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_best_ds_schedule_v1.py strict dsf8
# strict通过后执行原命令三轮双侧对照；内部等待全机idle。
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/best_ds_schedule_screen_v1.py \
  run dsf8 NEW_BEST_DSF8_LABEL base candidate base
```

### 当前最优版上的 DS-first 对照与 closing hint 准备（2026-10-22）

`best_dsf8_v1` 的 E64/seed0 strict 已通过：27项有效 payload/ScaleA/canary/
同 pointer 动态 graph byte checks 全部零差异，完整 MoE 22020096B 零差异，
实际两阶段 ELF 与 CPU/native 检查绑定。原精度门限没有修改。

`best_dsf24_v1` 不进入 GPU 测试。实际 `emit_hints` 的16项当前 geometry 组合
证明，16与24都被第一 slot 的14/10容量截断，完整 hint lists 相同。
`best_dsf24_v1.noop_hint_analysis.json` 保留这一静态结论；不为相同调度重复编译、
精度验证或计时。

`reboot20261022_best_dsf8_forward_v1` 已完整结束，以下每组为连续三轮中值。
ownership 不合格组全部排除，中值为 null，原记录保留；未跨组挑选单轮。

| 组 | GEMM1 三轮 μs | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---|---:|---:|
| 前侧当前保留版 | 85.3541 /84.6415 /84.9514 | 84.9514 | 64.4786 |
| DS-first 8 | 85.9552 /86.1168 /87.0235 | 86.1168 | 64.0819 |
| 后侧当前保留版 | 85.6405 /85.8747 /85.8853 | 85.8747 | 64.0927 |

候选比两侧分别慢1.1654 /0.2421 μs，即回退1.3718% /0.2819%。
不追加反向或 seed17/E65/outside-scope 验证，不采用。summary 为
`completed=true/source_restored=true`，无 error，生产源码恢复 `3f23fb29...97b68`。
240成员、970533B 的证据 archive 已下载，SHA256 为
`9275b8b1e6ee619853d8e006089c0a3dfd277310071570b9e5d8c803a0036039`。
本地从原始表格、所有 ownership、实际 call keys/source/ELF、完整三轮及 strict 输出
独立复算通过，`reboot20261022_best_dsf8_forward_v1.local_acceptance.json` 为 `passed=true`。
验收通过表示证据可信及精度检查通过，不表示性能有收益。

下一项 `best_close24_v1` 仅将目标 guard 内 closing hint 从8改为24，+1/-1行。
CPU/native 中 VGPR 从762增至789，SGPR58、LDS327680B、private/spills0保持；
protected DS/TDM/output 操作、ordered hardware fences 及 WMMA/BF16/FP4/tanh counts
全部保持。原 native gate 仅因 `resource increase: vgpr_count` 报失败，失败记录保留。
独立 `best_close24_v1.resource_review.json` 明确复核该告警，没有删除原失败项或
放宽数学、同步和精度检查。该 resource review 允许继续实验，不代表生产采用。
E64/seed0 strict 在36.1328秒内完成，27项 byte checks 及完整 MoE 零差异，
GEMM2 实际完整 ELF 保持。接下来用原正常入口进行相同双侧三轮对照，尚无性能结论。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/best_close24_reviewed_screen_v1.py \
  run close24 NEW_BEST_CLOSE24_LABEL base candidate base
# 完整对照结束、源码恢复后封包，再下载并在本地独立 accept。
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/best_close24_reviewed_screen_v1.py \
  seal NEW_BEST_CLOSE24_LABEL
```

正式保留成绩仍84.9116 μs，另85.2573 /85.3416 μs确认组保持。
GEMM2、原 benchmark、计时方式及生产 kernel 尚无本轮新增修改。

### closing hint 24 正向双侧改善，反向确认中（2026-10-22）

`reboot20261022_best_close24_forward_v1` 完整结束。前 control 的try1/try2因
ownership不合格整组排除；以下采用首个完整合格组，原失败和排除记录均保留。

| 组 | GEMM1 三轮 μs | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---|---:|---:|
| 前侧当前保留版，try3 | 85.9024 /85.5291 /85.7631 | 85.7631 | 64.0654 |
| closing 24，try1 | 85.4912 /84.4938 /84.6499 | 84.6499 | 64.2446 |
| 后侧当前保留版，try1 | 84.9977 /85.0082 /85.8875 | 85.0082 | 63.8040 |

候选比两侧快1.1132 /0.3583 μs，即1.2980% /0.4215%。尚未采用或更新正式成绩。
退出源码精确恢复，summary为completed/source_restored=true、无error。
245成员、972801B证据已下载，本地重新核对原表、三轮中值、ownership、source/
实际call keys/ELF以及27项strict和完整MoE，`local_acceptance.passed=true`。
archive SHA256为`5874acff66988848a2cd56b1fec3ccf0e76634002df82076c3d8d8c7a386bf78`。
原VGPR告警与单独resource review均保留在archive，未将告警改为native通过。
实际GEMM1完整ELF为`a29452a099597cb97312f42bed936f77e7b7e4e7cc538eba703bfdb1efac3b5e`。

14:01 UTC启动固定反向 `candidate→base→candidate`，label为
`reboot20261022_best_close24_reverse_v1`。各组仍完整三轮、120秒child上限和6秒idle等待；
尚无反向结论。只有四项相邻比较均改善，才补额外seed/E65原fallback和outside-scope验收。

计时期间仅在本地准备三个独立WMMA分组候选：4/4、6/8、9/9，依次为
`mma_group/mma_first_group`，B-TDM TH保持原值。均基于正式source `3f23fb29...97b68`，
保持DS_B=4、XDL queue、M-group32、inline-zero和A-only standard timeout。
旧DS-only扫描不作为本组合的性能证据。候选通过实际assignment的38项outside-scope/
env覆盖检查、56项hint总量检查及移除patch后完整bytes/AST一致检查；新增修改分别
+16/-2、+8/-1、+16/-2行，生产文件未修改。
工具为`build_best_mma_groups_v1.py`、`prepare_best_mma_groups_v1.py`和
`best_mma_groups_screen_v1.py`，run/seal/accept展开语法以及原timing/ownership函数AST检查通过。
仅相关私有候选与工具已上传、SHA/LF核对通过；没有与closing计时并发编译或GPU验证。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/best_close24_reviewed_screen_v1.py \
  run close24 NEW_BEST_CLOSE24_REVERSE_LABEL candidate base candidate
# 以下准备只能在当前性能序列结束后顺序执行。
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_best_mma_groups_v1.py cpu g6f8
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_best_mma_groups_v1.py native g6f8
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_best_mma_groups_v1.py strict g6f8
```

### closing hint 24 正反向均通过，追加验收中（2026-10-22）

`reboot20261022_best_close24_reverse_v1` 已完整结束，summary 为
`completed=true/source_restored=true`，无 error。第一候选组 try1 和中间 control
try1 因 ownership 不合格整组排除，原始记录保留、中值为 null；以下均为首个完整合格三轮组。

| 组 | GEMM1 三轮 μs | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---|---:|---:|
| 前侧 closing24，try2 | 84.2419 /84.1197 /84.4632 | 84.2419 | 64.3172 |
| 中间当前保留版，try2 | 85.5741 /85.4643 /86.0147 | 85.5741 | 64.6247 |
| 后侧 closing24，try1 | 84.0265 /84.6084 /84.5914 | 84.5914 | 64.3525 |

反向两侧分别改善1.5568% /1.1484%；与正向1.2980% /0.4215%一起，四项相邻比较
均为正收益。245成员、974025 B的反向archive已下载，本地从原表、完整三轮、ownership、
原call keys/source/ELF独立复算通过，`local_acceptance.passed=true`。archive SHA256为
`3619bbf495a8a9112445a700944aa7532da87611d0112392cb121a4cb1bb712d`。
没有把反向最快组84.2419 μs或最快单轮84.0265 μs更新为正式成绩。

14:15 UTC开始`best_close24_followup_v1.py run`，顺序验证E64 seed17、E65 seed0/17
的27项byte checks、完整MoE和实际两stage ELF，再直接编译8项outside-scope并复用
已验收control的完整ELF比较。每个compile/verify child限时120秒，生产source保持
`3f23fb29...97b68`，GEMM2与原benchmark没有修改。E65仍应为原`t64/w1x4`fallback；
本任务不声称已做到E64/E65共用`t256`，也不把random E65等同于固定shared expert routing验证。

```bash
# a07-3主机；内部进入dsr1_ep4，结果为正确性证据，不作为性能数据。
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/best_close24_followup_v1.py run
# 全部完成后封包、下载，在本地独立验收。
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/best_close24_followup_v1.py seal
python .codex_tmp/yadai_gemm1_param_20261021/tools/best_close24_followup_v1.py accept
```

### closing24追加验收完成并保留（2026-10-22）

额外strict三项全部通过，E64 seed17耗时36.7161秒，E65 seed0/17分别30.7079 /12.5996秒；
每项27个byte checks及完整MoE22020096 B均零差异。E64实际GEMM1为
`a29452a0...fac3b5e`，GEMM2为`4bd626e3...16be`；E65两stage ELF与原fallback精确相同。
8项outside-scope全部由candidate自身直接编译、完整ELF与已验收control一致，
包括GEMM2、FP8、非preshuffle A、pin1、t64、w1x4、b3、persistent1。
这些compile-only检查不运行persistent优化，也没有新增persistent实现。

89成员、590700 B追加验收archive本地独立复算通过，SHA256为
`f14f31150df19504af13653b7b369ef3f4a280b8d856904abc06e63e2ce21086`。
`best_close24_followup_v1.local_acceptance.json`的`passed=true`。
原VGPR762→789告警及独立review保留，SGPR58、LDS327680B、private/spills0保持。

14:21 UTC在本地与a07-3分别执行同一SHA防覆盖的adopt工具，保留唯一一行修改：

```python
FENCE_COVER_MMA = min(24 if standard_a_timeout else 8, max(0, n_acc - sched_mma_group))
```

生产source SHA为`6503501452e10afcb235cc97d02ba56714b6a7752bc7cc0853be04006bfdb7dd`，
仍保留原GEMM1 symbol与原计时方式。新正式记录取预先安排的第一正向候选组中值
84.6499 μs，反向84.2419 /84.5914 μs列为确认组，不选择最小组更新成绩。
距80 μs差4.6499 μs。没有commit/push、Git清理或修改GEMM2。

此前基于`3f23...`准备的三个MMA候选尚未编译或计时，保留静态记录但取消其GPU队列。
新`close24_mma_groups_v1.plan.json`以刚采用的`650350...`为对照，重新生成4/4、6/8、9/9。
新增delta分别+16/-2、+8/-1、+16/-2，38项outside-scope/env覆盖和56项实际hint总量
检查均通过。run/seal/accept展开语法、原timing/ownership函数AST及120秒child上限检查通过。
后续原正常入口先进行完整双侧三轮，有收益才做反向与额外精度验收。

### closing24上MMA候选的constexpr guard编译修正（2026-10-22）

`close24_mma_g6f8_v1_cpu_v1`首次CPU准备触及120秒，rc124，host elapsed120.1028秒；
未获得ELF、未执行GPU验证、无性能数据。原log/exit/host失败记录保留，测试child已退出，
没有延长限时或覆盖生产source。

新候选的长constexpr `and`和chained comparisons会经过安装版`RewriteBoolOps`重复展开，
与此前bounded lookup遇到的问题相符。既有只读表达式实验中，7项`and`的引用遍历节点30,033，
相同条件`all((...))`为20；这是表达式证据，不能把本次未采样的timeout断言为已定位profile。
v2将新增guard改为同条件的`all((...))`，各38项outside-scope/env和56项hint总量检查通过。
生产kernel不改，原v1失败保留，重新以新candidate/label编译；120秒上限保持。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/build_close24_mma_groups_v2.py
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_close24_mma_groups_v2.py cpu g6f8
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_close24_mma_groups_v2.py native g6f8
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_close24_mma_groups_v2.py strict g6f8
```

### closing24上MMA 6/8通过编译和完整MoE检查（2026-10-22）

紧凑guard的`close24_mma_g6f8_v2`在120秒内完成CPU/native和E64 seed0 strict。
完整GEMM1 ELF为`c4b38a5ca36c0e712dbd174a12f45dbdcb427072b6dc361bc3c7b6b93b3533f2`，
VGPR789、SGPR58、LDS327680B、private/spills0，与closing24资源相同。
6190条static instructions，对照6119；protected operations/offsets、ordered hardware fences
及WMMA/BF16/native FP4/tanh数量相同，`native.passed=true/reasons=[]`。
这些有限检查不能替代完整硬件hazard证明，原compiler waits保持。

strict用时35.8932秒，27项有效payload/ScaleA/canary/same-pointer dynamic graph
byte checks及完整MoE22020096B均零差异，GEMM2完整ELF保持`4bd626e3...16be`。
14:28 UTC启动`reboot20261022_close24_mma_g6f8_forward_v1`原入口对照：
closing24→6/8→closing24，各完整三轮，先通过实际原stage call keys准备，计时中无hook。
每个child仍限时120秒，owner等待每6秒检查；此节不预先宣布性能收益。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/close24_mma_groups_screen_v2.py \
  run g6f8 NEW_CLOSE24_G6F8_FORWARD_LABEL base candidate base
# 完成并确认source恢复后：seal，下载archive/receipt，在本地accept。
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/close24_mma_groups_screen_v2.py \
  seal NEW_CLOSE24_G6F8_FORWARD_LABEL
python .codex_tmp/yadai_gemm1_param_20261021/tools/close24_mma_groups_screen_v2.py \
  accept NEW_CLOSE24_G6F8_FORWARD_LABEL
```

### closing24上MMA 6/8完整双侧回退，放弃（2026-10-22）

`reboot20261022_close24_mma_g6f8_forward_v1`已完整完成，全部九轮原precision及
pre/post/settled ownership合格，实际原call keys/source/ELF逐轮核对通过。没有被排除的组。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前侧closing24 | 84.1884 /84.4542 /84.5320 | 84.4542 | 64.2889 |
| MMA 6/8 | 85.8814 /84.4371 /85.2194 | 85.2194 | 64.3417 |
| 后侧closing24 | 84.5488 /84.2411 /85.1392 | 84.5488 | 63.9476 |

候选分别慢0.7652 /0.6706 μs，即回退0.9061% /0.7932%。不追加反向、seed17/E65
或outside-scope验证，不进入生产；候选中较快的84.4371 μs不替换完整三轮中值。
源码已精确恢复`650350...bfdb7dd`，summary为`completed/source_restored=true`、无error。

173成员、735888B archive已下载，本地独立复算原表格、三轮中值、ownership、完整ELF
及27项strict/完整MoE通过，`local_acceptance.passed=true`且
`positive_at_every_adjacent_control=false`。archive SHA256为
`cfbbb1b9dac259e5ae1fd77d00b16404d4795d32d2c9d40eb1db18a08bc9732f`。
正式保留成绩仍84.6499 μs，没有用本次control更新成绩。

14:35 UTC开始同一冻结plan中的4/4 CPU-only准备；仍逐项compile/native/strict通过后
才执行原入口双侧三轮对照，9/9随后顺序执行。没有改原benchmark、GEMM2或计时方式。

### MMA 4/4资源复核及完整MoE初筛（2026-10-22）

`close24_mma_g4f4_v2`完成CPU-only编译，GEMM1完整ELF为
`73a3735130aabc600e2427ffa950f0a5e270adf03a657209c2ef9a7deffb4c00`。
VGPR789→797，SGPR58、LDS327680B、private/spills0保持；static instructions6119→6218。
原native检查失败，唯一reason为`resource increase: vgpr_count`，记录没有修改。
protected operations/offsets、ordered hardware fences及WMMA/BF16/FP4/tanh数量保持。

单独`review_close24_mma_g4_resources_v1.py`锁定789→797及完整ELF进行资源复核；
`prepare_close24_mma_g4_reviewed_v1.py`和`close24_mma_g4_reviewed_screen_v1.py`
要求原失败JSON的SHA、独立review以及所有原数学、同步、精度检查通过，才继续实验。
新增wrapper展开后的`run/child/observe/environment/swap/stop` AST与原框架相同；
child仍120秒。没有把资源告警改写为native通过，也没有修改精度门限。

E64 seed0 strict已完成，27项byte checks及完整MoE22020096B零差异，实际GEMM2 ELF保持。
14:40 UTC开始原入口closing24→4/4→closing24完整双侧三轮，尚无性能结论。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/review_close24_mma_g4_resources_v1.py
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_close24_mma_g4_reviewed_v1.py strict g4f4
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/close24_mma_g4_reviewed_screen_v1.py \
  run g4f4 NEW_CLOSE24_G4F4_FORWARD_LABEL base candidate base
```

### 等待外部GPU owner期间准备独立LDS读取顺序候选（2026-10-22）

4/4序列前control的r1通过后，r2的idle gate发现外部`dev_al`的GPU/KFD任务。
14:48 UTC核对本任务parent323876唯一child为`wait_normal_idle_v1.sh` PID333737，
没有本任务GPU测试child，随后仅SIGSTOP本任务idle入口，另启同一6秒idle检查。
利用等待时间顺序准备9/9和下一小候选；恢复前须确认所有准备及独立监控已结束，
恢复后重新通过原idle gate再计时。外部任务没有被停止或清理，占用期间不记录性能。

新`close24_lds_bfirst_v1`基于当前正式source`650350...bfdb7dd`，仅在原guard的
interleaved LDS payload读取中将每个i的B放在A之前，+3/-1行。scale读取原顺序、
所有地址、寄存器值、TDM、WMMA、epilogue和hardware fences保持；没有新增pipeline。
实际`load_lds_data` AST执行384组几何/ksl/kt/guard/interleave组合，读取事件集合、
最终寄存器值相同；scope关闭或非interleaved时顺序完全相同。15项原scope排除检查通过。
candidate source SHA=`889261c36f1dce3718bcd5556589275c0b37874fc03ce3af8ee1c125684646f6`。
目前仅静态及wrapper AST检查，尚无该候选native、device或性能结论。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/build_close24_lds_bfirst_v1.py
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_close24_lds_bfirst_v1.py cpu bfirst
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_close24_lds_bfirst_v1.py native bfirst
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_close24_lds_bfirst_v1.py strict bfirst
# 必须等待上述检查及前序性能队列结束，原source恢复后：
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/close24_lds_bfirst_screen_v1.py \
  run bfirst NEW_CLOSE24_BFIRST_FORWARD_LABEL base candidate base
```

### MMA 9/9与B-first完成编译及strict，4/4原入口异常复测（2026-10-22）

14:48～14:52 UTC顺序完成9/9和B-first的CPU/native及E64 seed0 strict；每个child均在
120秒内退出。9/9的实际完整GEMM1 ELF为
`dceab63cb31bb4fcc177bd78a613b8f854454dd6e766e1d302f87920d97c7c91`，
B-first为`b5fe1bcb174dc07d58e9b1cf22d6a273f5da9851be3516c4048c13753114a351`。
两者native均`passed=true/reasons=[]`，VGPR789、SGPR58、LDS327680B、private/spills0
与closing24相同。protected operations/offsets、ordered fences及数学操作数量保持；
各27项byte checks及完整MoE22020096B零差异，实际GEMM2完整ELF保持。
静态instruction counts分别6117、6126，对照6119。上述结果尚不代表性能收益。

14:52 UTC确认准备children和独立idle监控均退出后，恢复本任务idle入口PID333737。
第一4/4对照label`reboot20261022_close24_mma_g4f4_forward_v1`前control r1通过，
r2在14:54 UTC原入口测试中rc134退出：`c10::AcceleratorError` /
`CUDA error: an illegal memory access was encountered`。出错时生产源码是closing24
`650350...bfdb7dd`，尚未开始4/4候选计时。summary为`completed=false`，results为空，
`source_restored=true`；不完整组不计性能，不将其判为MMA4/4回退或正确性失败。
失败log/summary及当时dmesg已下载保留。dmesg记录同时间python3 pid487186的TCP
page faults；现有记录尚未将该host PID精确绑定到容器内测试PID402442，因此不据此
归因于外部任务或某一个kernel。

14:58 UTC以同一control source及原cache运行额外`AMD_SERIALIZE_KERNEL=3`诊断，
完整原入口通过，`logits_diff=3.61055e-6`、`rel_l2=0.00268721`。
该运行标记`performance_data=false`；串行诊断不能证明异步路径不存在race，也不能
替代原命令性能。没有修改benchmark、GEMM2、producer、精度门限或kernel源码。
14:59 UTC重新启动完整原入口对照`reboot20261022_close24_mma_g4f4_forward_v2`，
保留第一次失败，仍通过全机idle再计时，异常时停止整条序列并精确恢复source。

### closing24下DS-first24的独立静态准备（2026-10-22）

新候选`close24_dsfirst24_v1`只覆盖原目标GEMM1 scope中的`ds_first_n`默认值16→24，
显式`AITER_FLYDSL_DS_FIRST_N`覆盖仍优先。源码SHA为
`e1e0cea45b57a979b2c5e00c96693b753e8ac15b158b630d4ba5827883de91e2`，
delta +7/-1行；使用紧凑constexpr guard，其他device helpers AST完全相同，
移除patch后完整源码bytes恢复。19项scope排除/显式env检查及56项实际hint总量检查通过。
closing24使部分第一slot容量升至40，16项hint列表确实变化；旧closing8阶段的DS-first24
被slot容量截断为相同hint，该旧no-op结论不套用到此组合。
run/seal/accept展开语法与原六个timing/ownership函数AST相同，child上限仍120秒。
当前仅在本地冻结候选与工具，尚无编译、device或性能结论，生产文件不改。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/build_close24_dsfirst_v1.py
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_close24_dsfirst_v1.py cpu dsf24
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_close24_dsfirst_v1.py native dsf24
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_close24_dsfirst_v1.py strict dsf24
# 仅在前序对照完成且生产source恢复后，执行完整双侧三轮。
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/close24_dsfirst_screen_v1.py \
  run dsf24 NEW_CLOSE24_DSF24_FORWARD_LABEL base candidate base
```

### MMA 4/4完整正向对照通过，继续反向验收（2026-10-22）

`reboot20261022_close24_mma_g4f4_forward_v2`完整完成，源码恢复到closing24
`650350...bfdb7dd`。前control try1的r3、candidate try1的r1在ownership检查中不合格，
两组三轮整体排除，不从其中选择较快单轮。各自try2以及后control try1全部有效：

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前侧closing24 try2 | 85.6091 /85.2530 /84.2282 | 85.2530 | 63.7303 |
| MMA 4/4 try2 | 82.4889 /83.5707 /82.5101 | 82.5101 | 64.6032 |
| 后侧closing24 try1 | 86.3199 /84.3106 /84.9600 | 84.9600 | 64.0277 |

候选相对两侧分别改善3.2174% /2.8836%。全部有效轮的原precision、实际原call keys、
source/完整ELF及pre/post/settled ownership通过。GEMM2完整ELF均为原`4bd626e3...16be`；
上表计时波动不表示GEMM2代码修改。第一序列的illegal-address记录继续保留，根因尚未确定。

249成员、979049B evidence archive已经下载并独立复算，SHA256为
`05dbdccdda43ffe70a3edd4c4b850310d026e2bc01f424bb2b08e4ba979bc0a7`，
`local_acceptance.passed/positive_at_every_adjacent_control=true`。
目前只是候选正向收益，正式保留成绩仍84.6499 μs。继续反向完整三轮，四项相邻收益均成立后，
再验证E64 seed17、E65 seed0/17各27项byte checks和完整MoE，以及candidate自身8项outside-scope ELF。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/close24_mma_g4_reviewed_screen_v1.py \
  run g4f4 reboot20261022_close24_mma_g4f4_reverse_v1 candidate base candidate
# 正反向独立验收后再执行：
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/close24_mma_g4_followup_v1.py run
```

### MMA 4/4反向、额外精度及scope验收通过，正式采用（2026-10-23）

`reboot20261022_close24_mma_g4f4_reverse_v1`完整完成；最后candidate try1有ownership
不合格，整组排除，使用try2的完整三轮。没有选择其中82.1748 μs的最快单轮。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前侧MMA4/4 try1 | 83.0308 /83.0814 /82.3836 | 83.0308 | 64.0845 |
| 中间closing24 try1 | 83.7225 /84.5868 /84.7513 | 84.5868 | 63.6882 |
| 后侧MMA4/4 try2 | 82.9970 /82.6449 /83.0877 | 82.9970 | 63.6395 |

反向两项改善1.8395% /1.8795%，连同正向四项相邻比较全部有收益。
213成员、856182B reverse archive已下载独立复算通过，SHA256为
`4504d0bfeb6b52584fbe513da486ec6fa3f16d5888f0f5acf64b489ace0bf52d`。

`close24_mma_g4_followup_v1`补充完成E64 seed17、E65 seed0/17各27项byte checks及
完整MoE22020096B对比，全部零差异。GEMM2、FP8、original A、pin1、t64 fallback、w1x4、
buffers3、persistent1八项candidate自身完整ELF与原control相同；测试child均在120秒内退出。
87成员、591041B followup archive在本地独立验收通过，SHA256为
`3d80c014ac16e8ef5b08b55049990b11f76fff2bbac6f9fb99a8be089f3e213b`。
E65保留原`t64/w1x4`fallback；此验收不声称E65与E64共用t256 kernel，也未验证固定shared route。

独立验收后，`adopt_close24_mma_g4_v1.py`在本地和远端按SHA校验、备份、原子替换落地
原+16/-2行候选，两端源码SHA均为
`33e2265e43a85b9ffcfe55ac7c55c7a58b183291a4281ffe0c876a4ba5066656`。
正式中值更新为82.5101 μs，距80 μs差2.5101 μs；反向确认83.0308 /82.9970 μs。
VGPR797、SGPR58、LDS327680B、private/spills0，原VGPR789→797告警保留。
GEMM1完整ELF=`73a3735130aabc600e2427ffa950f0a5e270adf03a657209c2ef9a7deffb4c00`，
GEMM2完整ELF仍`4bd626e3...16be`。没有修改benchmark、精度门限、功能或shape支持，
未commit/push或Git清理。第一序列control illegal-address的失败证据保留，根因仍未确定。

下一步以刚采用的MMA4/4为基准，重新生成DS-first和closing hint候选。旧closing24上
9/9、B-first及DS-first24的冻结实验不得覆盖新保留源码，也不混用旧ELF验证新候选。

```bash
# 本地完成正反向及followup accept后：
python .codex_tmp/yadai_gemm1_param_20261021/tools/adopt_close24_mma_g4_v1.py inspect
python .codex_tmp/yadai_gemm1_param_20261021/tools/adopt_close24_mma_g4_v1.py apply
# 远端收到三个独立acceptance记录后执行同一apply；完成后核对production SHA256。
```

### MMA4/4上的DS-first20准备与超时复测（2026-10-23）

新候选以已采用的source `33e2265e...5066656`和GEMM1 ELF `73a37351...effb4c00`
为对照，只将原目标guard中的DS-first默认16改为20（新增delta +7/-1），显式公共env仍优先。
同批closing28/20候选分别仅改一行，尚未编译、验证或计时。56项实际hint展开检查确认
数学及DS/TDM总量保持、16项hint列表改变；15项scope排除和其他device helpers AST保持。

DS-first20 CPU/native通过：完整ELF `f17e269acc579b657355fe3eb95b6a5abb1d24c855b60d3a0acc356228c54f45`，
VGPR797、SGPR58、LDS327680B、private/spills0；static instructions 6218→6200。
protected memory operations/offsets、ordered hardware fences和数学数量保持。

第一原strict `mma4_hint_dsf20_v1_strict_e64_s0_v1`到120秒后被timeout终止，rc124、
host elapsed120.1326秒。退出前日志停在模块/config导入、主线程接近100% CPU；
根因未确定，不能将其归为GPU deadlock，也不宣称已修复。确认child退出，无该测试残留。
随后同一计算的faulthandler诊断v2在12.9774秒通过，早于15秒采样，因此未取得stack。
再次未经诊断包装的原strict v3在12.7889秒通过：27项byte checks和完整MoE
22020096 B零差异；GEMM2完整ELF仍为 `4bd626e3...16be`。这些均为正确性数据。
失败log与两次复测分别保存，不覆盖、不用验证耗时作性能结论。

已启动 `reboot20261023_mma4_hint_dsf20_forward_v1` 的原入口前control/candidate/后control
完整三轮对照；尚无正式性能结论，每个child限120秒、每轮全机idle后开始。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma4_hint_screen_v1.py \
  run dsf20 NEW_MMA4_DSF20_FORWARD_LABEL base candidate base
```

### 同批closing28/20的native与完整MoE准备完成（2026-10-23）

等待外部KFD owner期间，仅暂停本任务处于idle gate的parent/waiter，未启动性能测试，
另外保持每6秒的只读idle检查。顺序完成closing28/20的CPU、native和原strict；
各child均在120秒内结束。准备完成后停止额外监控、确认源码SHA保持，再恢复原性能队列。
pause/resume receipts单独保存，不把本次准备耗时作为性能。

| candidate | VGPR / SGPR | LDS / private / spills | static instructions | strict elapsed（秒，非性能） |
|---|---|---|---:|---:|
| DS-first20 | 797 / 58 | 327680 B / 0 / 0 | 6200 | 12.7889 |
| closing28 | 797 / 58 | 327680 B / 0 / 0 | 6200 | 36.0508 |
| closing20 | 789 / 58 | 327680 B / 0 / 0 | 6235 | 36.0201 |

closing28完整ELF为 `5ee25d1160f96447fae262ba8992bdd7ee577d4d8c1dc21d18cb7bf2f97fca31`，
closing20为 `c06a6a225cf5ab9c38eb55082e46c824b469db80315f5baf690a5c8a2961d095`。
两项27 BYTE_CHECK、same-pointer dynamic graph、完整MoE22020096 B均零字节差异，
原reference通过；GEMM2完整ELF保持 `4bd626e3...16be`。
本地下载原ISA/ELF/strict log后独立复算source、actual artifacts及原精度检查通过，
记录为 `mma4_hint_preparation_downloaded_acceptance_v1.json`。

closing20的VGPR减少8个不等于已取得性能提升。三项都仍是私有候选，尚无正式计时、
未采用、未覆盖当前生产最优版。随后按DS-first20、closing28、closing20顺序做原入口对照。

### MMA4/4邻近分组的编译准备（2026-10-23）

基于正式source `33e2265e...5066656`冻结MMA3/3和5/5两项，只改现有目标guard中
`sched_mma_group`、`sched_first_group`的两个默认数字（各+2/-2行）。没有新增guard、env开关、
device helper或shape限制；显式公共env仍优先。38项scope/env检查和56项实际hint总量检查通过，
其中48项hint列表改变；DS/TDM/WMMA总量保持。

两项CPU/native顺序完成：VGPR797、SGPR58、LDS327680 B、private/spills0保持，
MMA3/3静态instruction count为6260，5/5为6217，对照6218。
native检查均`passed=true/reasons=[]`，protected操作/offset、ordered hardware fences以及
WMMA/BF16/FP4/tanh counts保持。尚未完成device strict或性能，不将编译结果作为收益。

本任务此前暂停于idle gate的DS-first20 parent/waiter，在确认所有准备children退出、
额外idle monitor已退出且production SHA匹配后，于2026-10-22 16:58:58 UTC恢复。
receipt为`logs/mma4_neighbors_idle_resume_v1.json`；原runner自行每6秒等全机idle再计时，
计时期间不并发编译、传输或GPU监控。邻近分组的strict排在该完整序列结束后。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma4_neighbors_v1.py strict g5f5
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma4_neighbors_v1.py strict g3f3
# strict通过且前序队列退出、production SHA恢复后，再运行原入口三轮对照：
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma4_neighbors_screen_v1.py \
  run g5f5 NEW_MMA4_NEIGHBOR_FORWARD_LABEL base candidate base
```

### MMA4/4上的DS-first20原入口对照完成：不采用（2026-10-23）

`reboot20261023_mma4_hint_dsf20_forward_v1`完整退出，production恢复为
`33e2265e...5066656`。候选try1和后control try1～try5均因post ownership中的
无法识别`<已退出>` KFD entry整组排除，中值null；不拼轮、不从排除组取最快值。
每轮pre均为全机idle，计时期间未并发准备或传输。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| MMA4/4前control try1 | 82.7881 /82.7293 /83.8403 | 82.7881 | 64.0738 |
| DS-first20 try2 | 82.8873 /83.0245 /82.4741 | 82.8873 | 64.2067 |
| MMA4/4后control try6 | 83.0412 /82.8115 /83.0580 | 83.0412 | 64.7802 |

候选相对前侧慢0.0992 μs（0.1198%），相对后侧快0.1539 μs（0.1853%），
方向不一致，没有稳定附加收益；不采用、不追加反向。原reference、actual原call keys、
source/完整ELF及ownership均通过独立复算，GEMM2完整ELF保持原`4bd626e3...16be`。
DS-first20首次strict超时及两次复测证据继续保留，没有将成功复测宣称为根因已修复。

396个member、1485887 B archive已下载到本地并独立验收通过，SHA256为
`19d5dceab39d7aecbe6486eed58dff9394be76ed948bc8a24588187f76344d81`。
`reboot20261023_mma4_hint_dsf20_forward_v1.local_acceptance.json`中
`passed=true/positive_at_every_adjacent_control=false/not_production_adoption=true`。
正式最优成绩仍82.5101 μs。该序列结束后，开始MMA5/5的完整MoE strict准备，
随后继续closing28/20原入口对照；不使用旧closing24阶段冻结的ELF替换当前candidate。

### MMA3/3、5/5完整MoE准备及本地复算完成（2026-10-23）

两项原strict均在120秒内结束，各27项BYTE_CHECK、same-pointer dynamic graph和完整MoE
22020096 B均零字节差异，原reference通过，GEMM2完整ELF保持`4bd626e3...16be`。

| candidate | strict elapsed（秒，非性能） | GEMM1完整ELF | VGPR / SGPR / LDS |
|---|---:|---|---|
| MMA3/3 | 36.4416 | `ed5fc68ab38a0d99d65e99846b54aa8799dbb0de0c1ee4f1aaf5f3720f396df5` | 797 / 58 / 327680 B |
| MMA5/5 | 36.6069 | `edf05db3124f59b5b165c99fd502eb23e675a458505c81d24ca91a7dc13d3373` | 797 / 58 / 327680 B |

48成员、614950 B preparation archive已下载，本地重新解析ISA、逐项复算member SHA，
验证两个默认数字的精确source delta、protected operations/offsets、ordered fences、
数学数量、resources、strict原log及完整ELF，全部通过。archive SHA256为
`60afa1228cd62e63d3f0bdf376ca7ef5ff51c7e881dbc0be0b2b16a53c21e94c`，
独立结果为`logs/mma4_neighbors_preparation_v1.local_acceptance.json`。
目前只验证E64 seed0，不视为性能收益，也未采用。

生产source仍`33e2265e...5066656`。核对同一boot、branch/HEAD、容器状态和FlyDSL
frontend修复SHA后，启动closing28原入口前control/candidate/后control完整三轮，
label为`reboot20261023_mma4_hint_close28_forward_v1`。计时期间不传输文件或并发编译。

### MMA4/4上的closing28双侧回退，不采用（2026-10-23）

完整原入口对照完成，九轮的precision、actual原call keys、source/完整ELF、
pre/post/settled ownership均有效，无整组排除；测试child均在120秒内退出。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前control try1 | 83.8280 /83.1888 /82.9802 | 83.1888 | 64.2658 |
| closing28 try1 | 84.3463 /83.6107 /83.8193 | 83.8193 | 63.7598 |
| 后control try1 | 83.0053 /82.9758 /83.9247 | 83.0053 | 64.0867 |

候选比两侧慢0.6305 /0.8140 μs（0.7579% /0.9807%），不采用、不追加反向。
GEMM2完整ELF保持；其计时波动不属于GEMM2代码优化。
生产source恢复`33e2265e...5066656`，没有本任务child残留。

167成员、731175 B evidence archive已下载、本地独立复算通过，SHA256为
`40d0e780cef7c4d120530d6163e263f9772bfef21b1c6acc2b8a2c4af8ecfbeb`。
`reboot20261023_mma4_hint_close28_forward_v1.local_acceptance.json`为
`passed=true/positive_at_every_adjacent_control=false`。正式成绩仍82.5101 μs；
随后启动closing20完整原入口对照，不把候选的VGPR减少直接当作性能改善。

### MMA4/4上的closing20双侧回退，不采用（2026-10-23）

`reboot20261023_mma4_hint_close20_forward_v1`完成原入口前control/candidate/后control。
前control try1和candidate try1各有一轮post ownership不合格，均整组三轮排除、中值null；
采用各自try2的完整合格组，没有拼轮或选择最快单轮。后control try1全部合格。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前control try2 | 82.6489 /83.2162 /83.5304 | 83.2162 | 63.7786 |
| closing20 try2 | 85.0884 /85.1432 /85.3731 | 85.1432 | 64.1034 |
| 后control try1 | 83.0771 /83.0203 /83.8951 | 83.0771 | 63.9367 |

候选相对两侧慢1.9270 /2.0661 μs（2.3157% /2.4870%），不采用、不追加反向。
VGPR797→789没有转化为收益，SGPR58、LDS327680 B、private/spills0保持；
这项结果仅说明当前调度组合下的测量，不将VGPR数量解释为独立因果证明。

239成员、970496 B archive已下载、本地独立复算通过，SHA256为
`2afca1f4fbf69ce2535780ee5cca874f335a04d46e967dfaa5644b73efcc6d5f`。
`reboot20261023_mma4_hint_close20_forward_v1.local_acceptance.json`为
`passed=true/positive_at_every_adjacent_control=false`。全部原reference、source、
actual原call keys和完整GEMM ELF核验通过，GEMM2 ELF保持。production恢复正式
`33e2265e...5066656`，parent1559202已退出，未发现本任务GPU/KFD进程残留。

随后依次测试已通过E64 seed0完整MoE初筛的MMA5/5和MMA3/3，不使用旧阶段的ELF或PC。

### MMA5/5双侧小幅回退，不采用（2026-10-23）

`reboot20261023_mma4_neighbors_g5f5_forward_v1`完整结束；前control try1～3、
后control try1因ownership不合格整组排除。采用第一个完整合格三轮组，不拼轮。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前control try4 | 82.8687 /83.2713 /82.8835 | 82.8835 | 64.1310 |
| MMA5/5 try1 | 83.0922 /82.7524 /83.2249 | 83.0922 | 64.1161 |
| 后control try2 | 82.9442 /83.1975 /82.6555 | 82.9442 | 63.9454 |

候选相对两侧慢0.2087 /0.1480 μs（0.2518% /0.1784%），不采用、不追加反向。
313成员、1211467 B archive已下载并独立核验，SHA256为
`5fd64969cca14314ae95c24fd9ca0e4d17f1fffe803ba2ed97ea1eed8a5e9db1`；
`reboot20261023_mma4_neighbors_g5f5_forward_v1.local_acceptance.json`为
`passed=true/positive_at_every_adjacent_control=false`。GEMM2 ELF保持原版，
production恢复`33e2265e...5066656`，parent1622988已退出。

本地从已验收preparation archive提取三份真实ELF/ISA，逐条编码核验后比较静态区域。
MMA4/4、5/5、3/3分别6218 /6217 /6260条executable instructions；
5/5与4/4的epilogue/completion均3307条。这些是静态计数，含owner/drain多份body，
不作为动态执行数、延迟或瓶颈比例。`S_SET_VGPR_MSB`可co-issue，不能按条数估计独占开销。
原记录为`logs/mma4_neighbors_static_regions_v1.json`，没有GPU dispatch或新性能数据。

### MMA3/3正向双侧改善，继续反向确认（2026-10-23）

`reboot20261023_mma4_neighbors_g3f3_forward_v1`完整结束，无ownership整组排除；
九轮原precision summary通过，actual原call keys与source/完整ELF核验通过。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| MMA4/4前control try1 | 83.1762 /82.7335 /83.0413 | 83.0413 | 64.2427 |
| MMA3/3 candidate try1 | 81.3841 /82.0840 /82.2697 | 82.0840 | 63.9814 |
| MMA4/4后control try1 | 82.8979 /83.0351 /82.7630 | 82.8979 | 64.0488 |

候选相对前后control分别改善0.9573 /0.8139 μs（1.1528% /0.9818%）。
此时尚未采用；不将81.3841 μs单轮当作正式成绩。production恢复`33e2265e...5066656`，
parent1703656已退出。169成员、733677 B archive已下载到本地并独立复算通过，SHA256为
`17e0765d9c6ad7553826f54b1d660fea596814b6450017153b57ee4305c42f9f`；
`reboot20261023_mma4_neighbors_g3f3_forward_v1.local_acceptance.json`中
`passed=true/positive_at_every_adjacent_control=true/not_production_adoption=true`。

随后启动反向candidate/control/candidate，label为
`reboot20261023_mma4_neighbors_g3f3_reverse_v1`，每组完整三轮、每child120秒。
仅在反向两侧也改善、追加E64 seed17/E65 seed0/17完整MoE和8项candidate自身outside-scope
完整ELF验收通过后，才考虑采用。正式成绩仍82.5101 μs。

### MMA3/3反向确认通过，完整兼容性验证中（2026-10-23）

`reboot20261023_mma4_neighbors_g3f3_reverse_v1`完整结束，无ownership整组排除。
九轮原reference、全机pre/post/settled ownership、actual call keys及完整ELF均通过。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| MMA3/3前candidate try1 | 81.7698 /82.0271 /81.4642 | 81.7698 | 64.0527 |
| MMA4/4 control try1 | 83.2498 /82.6302 /82.4087 | 82.6302 | 64.6980 |
| MMA3/3后candidate try1 | 81.8966 /82.0861 /82.3329 | 82.0861 | 64.2554 |

两侧candidate相对control分别改善0.8604 /0.5441 μs（1.0413% /0.6585%）。
正反向四项相邻对照均改善，但此时尚未采用，也不把反向更快的一组替换正向成绩。
反向169成员、734143 B archive已下载并独立复算通过，SHA256为
`c5882548be8976c31019b9162d45d4b62752a09bd55ea7381ad41ac9bd57af2a`；
`reboot20261023_mma4_neighbors_g3f3_reverse_v1.local_acceptance.json`中
`passed=true/positive_at_every_adjacent_control=true`。parent1745043退出，production恢复
`33e2265e...5066656`。

`mma3_followup_v1.py run`开始追加E64 seed17、E65 seed0/17完整MoE和8项candidate直接
outside-scope compile。每个child120秒，验证耗时不作为性能。为MMA3/3准备的trace脚本
已逐项核对SHA并通过`bash -n`，待全部验收、正式采用后再采集匹配新ELF的trace。

### MMA3/3完整验证通过，最小落地（2026-10-23）

追加E64 seed17、E65 seed0/17各27项BYTE_CHECK、同pointer动态graph与完整MoE
22020096 B均零字节差异，原reference通过。candidate直接编译的GEMM2、FP8输入、
original A、pin1、t64 fallback、w1x4、b3、persistent1共8项，完整ELF与各自control
逐字节相同。E65保持原fallback；没有验证固定shared1 route，也不声称E64/E65共用kernel。

85成员、590418 B followup archive已下载并独立核验通过，SHA256为
`ac9d49fff2bc68a30e3b0564c4d6553d4103ab38b3dab40e336fd5cf1dadcade`；
`mma3_followup_v1.local_acceptance.json`中`passed=true`。验证耗时不作性能数据。

本地只将原target guard内两个默认数字4→3，和已完整测试的candidate源码逐字节相同，
source SHA=`33a882a69537779c954d83b47b03b6cfc56fc19edf1c7a9b1fa05b3f81c01f09`。
远端采用前核对branch/HEAD、旧production SHA、正反向和followup验收，保留旧源码备份，
再原子替换为scp上传的本地同字节文件；无Git清理、commit或push。

正式记录使用事先固定正向组中值82.0840 μs，反向确认81.7698 /82.0861 μs，
不选择最快三轮组或81.3841 μs单轮。相对前后control改善0.66%～1.15%，距离80 μs
还差2.0840 μs。GEMM1完整ELF=`ed5fc68a...f396df5`，VGPR797/SGPR58/LDS327680 B，
private/spills0；GEMM2完整ELF保持原`4bd626e3...16be`。

随后按新source/ELF/kernel名及新candidate原call inventory采集匹配MMA3/3的thread trace，
使用`run_best_mma3_att_v1.sh`；旧MMA4/4及84.9116 μs阶段PC/比例不套用到新capture。

### MMA3/3匹配trace完成及下一组小参数准备（2026-10-23）

新capture=`reboot20261023_best_mma3_att_v1`，四个SIMD-select profiler/decoder均成功。
实际private cache按MMA3/3原call keys、sourceIR和globals核验，两阶段完整ELF匹配，
64 active waves每wave1792条WMMA。237成员、7058221 B archive已下载并逐项验收，
SHA256=`152aa4f27800e12073d4b61ee1451abfe003a0e0899e67a7fc013d087fa3e9ee`。
本地重新核对decoded PC/ISA字节并复算全部wave数据，只规范化68个路径分隔符字段，
结果与远端精确一致；无`.out`或private cache下载，没有本任务GPU/KFD残留。

累计wave timestamp span中mainloop68.2125%、activation/quant/scale/LDS+barrier14.5123%、
final input drain0.2419%；这不是GPU critical-path占比，不能直接换算速度收益。
完整新数据、actual PC和复现已补入`moe_gemm1_hyg_dsr1_best_thread_trace_analysis.md`。

基于当前MMA3/3源码冻结MMA2/2、MMA3/first2两项，分别+2/-2和+1/-1行。
原guard、显式env覆盖、device helpers AST、B-TDM TH保持，38项scope/env检查及
56项实际hints总量检查通过；各有48项hint列表变化。CPU/native检查完成，均
`passed=true/reasons=[]`、VGPR797/SGPR58/LDS327680 B/private0/spills0，
protected operations/offsets、ordered fences及数学数量保持。

| candidate | source SHA256 | 完整GEMM1 ELF | static instructions |
|---|---|---|---:|
| MMA2/2 | `4309cdc378557ba48aa0a0243858f5bb0527cf8bf65a0cad8cd0546422caffff` | `39af7d765014b870ebdd2f84eb8fe7d66c19ea6a65b5c7424c76a2ac3af95c65` | 6368 |
| MMA3/first2 | `7f60e033a44445403f141253ff52dd79bca0078d76b988f4231a83c93802b0f7` | `86d771a2954954747fc24421fe2ca8c3f4ae7b437e05a7052a842adccabc34a1` | 6262 |

两项均已通过完整E64 seed0 strict：27项BYTE_CHECK、same-pointer动态graph和完整MoE
22020096 B均零字节差异，原reference通过，GEMM2完整ELF保持。production保持MMA3/3。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma3_small_groups_v1.py strict g2f2
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma3_small_groups_v1.py strict g3f2
# strict和前序children结束后，完整三轮原入口对照；新label必填。
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma3_small_groups_screen_v1.py \
  run g2f2 NEW_MMA3_SMALL_FORWARD_LABEL base candidate base
```

### MMA2/2未同时改善两侧，不采用（2026-10-23）

`reboot20261023_mma3_small_g2f2_forward_v1`已完成。前control try1、candidate try1
各有一轮ownership不合格，均整组三轮排除、中值null；采用各自第一个完整合格组。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前MMA3/3 control try2 | 82.2930 /81.6962 /81.9303 | 81.9303 | 64.3334 |
| MMA2/2 candidate try2 | 83.0161 /82.2674 /81.9933 | 82.2674 | 64.0531 |
| 后MMA3/3 control try1 | 81.7047 /82.8179 /82.5733 | 82.5733 | 63.8441 |

相对前后control分别为-0.4114% /+0.3705%，不同时胜过两侧，未采用、不追加反向。
248成员、979032 B archive已下载并独立复算通过，SHA256为
`54567e552250194dc1aa58b6a950cf4d09bfe7cc341a4c80be8203ba1e58d26b`。
`reboot20261023_mma3_small_g2f2_forward_v1.local_acceptance.json`为
`passed=true/positive_at_every_adjacent_control=false`。全部原reference、source、actual
原call keys、两阶段完整ELF和ownership核验通过；GEMM2保持，production恢复正式
`33a882a6...f81c01f09`，parent1878599已退出。随后测试MMA3/first2。

### 当前trace的K128尾部拆分与小改动准备（2026-10-23）

`analyze_mma3_ksteps_v1.py`对已验收MMA3/3 capture重新核对ELF、decoded PC/ISA，
按每32条WMMA切分64 waves的56项连续区间，精确相加为mainloop的1191267 cycles。
多数steady区间约313～315 cycles；K128 #54，即最后K256 tile的subtile0，区间中值
600.5 cycles、范围599～602，#55为234 cycles。#54无TDM/barrier，含下一subtile的
LDS reads、地址生成和compiler-generated DS waits。这里是WMMA边界间隔，含跨界prefetch，
不是独占K128延迟，也不是final input drain；不能据此直接预测μs收益。
独立记录为`logs/mma3_kstep_detail_v1.json`，SHA256为
`2e65423476a86b1283183c7d2ea9abdab48176e820908defcb3a56241bc2b9bc`。

基于当前source冻结两个独立候选，分别仅+1/-1和+2/-1行：

- `tailcover`：在现有GEMM1 scope内，让最后K256 tile的subtile0也使用已有closing24 hint。
  仅扩大compiler-hint选择，input ring、实际LDS/TDM、数学顺序和所有hardware同步保持。
- `puretail`：在现有GEMM1 scope、无下一subtile时，将全部剩余WMMA作为一个compiler group。
  不改数学操作，不删除等待，不改变其他区域的first group。

32/36项实际条件表达式检查及其他device helper AST核验通过；没有新增env、shape限制、
persistent、wave布局或pipeline。source SHA分别为`9eed3ad9...e02e3c77`、
`60aad707...891fd4be`。当前只完成本地冻结，尚无编译、GPU验证或性能结论；正式最优仍82.0840 μs。

补充`check_mma3_terminal_hints_v1.py`实际执行`spread/emit_hints`及其tail选择AST，
每项112个完整hint列表检查通过，MMA/DS/TDM总量保持；tailcover有16个列表变化，
puretail有12个。各56项outside-scope完全相同，非目标区域保持；结果为
`logs/mma3_terminal_hint_totals_v1.json`，SHA256=`37e6ac49...c3a44caf`。
静态验证最初将hint计数要求为正数，原`rocdl.sched_dsrd(0)`合法零计数触发断言；
工具改为允许零计数后通过，不修改device代码、计时或精度标准。

### MMA3/first2正向双侧改善，反向确认中（2026-10-23）

`reboot20261023_mma3_small_g3f2_forward_v1`已完成；candidate try1的r3未通过ownership，
整组三轮排除，中值null。前后control try1、candidate try2全部合格：

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前MMA3/3 control | 81.1839 /82.2042 /81.7847 | 81.7847 | 63.8904 |
| MMA3/first2 candidate | 81.0339 /81.6728 /81.4412 | 81.4412 | 63.9453 |
| 后MMA3/3 control | 81.6456 /82.1008 /82.1811 | 82.1008 | 64.0085 |

相对两侧改善0.3435 /0.6596 μs（0.4200% /0.8034%）；候选尚未采用。
212成员、856987 B archive已下载并独立复算通过，SHA256为
`3aa46a3a285a9d3582d98f4313d6d8c636d82fa2bee4b93771bd51a52b42c17d`。
`reboot20261023_mma3_small_g3f2_forward_v1.local_acceptance.json`中
`passed=true/positive_at_every_adjacent_control=true`；原reference、source、actual原call
keys、完整两阶段ELF及ownership核验通过。production恢复MMA3/3，parent1958033退出。

已启动反向`reboot20261023_mma3_small_g3f2_reverse_v1`，顺序candidate/control/candidate，
各完整三轮，每child120秒，性能前全机idle。只有反向也改善并追加E64 seed17、E65 seed0/17
完整MoE和8项candidate直接outside-scope ELF验收通过后才考虑采用。
本地追加验收工具`mma3_first2_followup_v1.py`已展开语法核验，不改变原验证口径；
正式最优仍82.0840 μs，不挑选81.0339 μs单轮。

### MMA3/first2反向双侧通过，追加完整验证（2026-10-23）

`reboot20261023_mma3_small_g3f2_reverse_v1`已完成。前candidate try1 r3、control try1 r1
ownership不合格，均整组三轮排除、中值null；后续第一个完整合格组如下：

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前MMA3/first2 candidate try2 | 81.4641 /81.8079 /82.0759 | 81.8079 | 64.0717 |
| MMA3/3 control try2 | 81.5649 /82.4864 /81.9761 | 81.9761 | 64.2023 |
| 后MMA3/first2 candidate try1 | 81.7464 /80.9979 /81.9952 | 81.7464 | 64.3329 |

两侧分别改善0.1682 /0.2297 μs（0.2052% /0.2802%）。正反向四项相邻对照均为正收益，
但不从正反向组选最低中值，也不将80.9979 μs单轮当作正式成绩。
248成员、976630 B archive下载并独立复算通过，SHA256为
`418235274985de795d738c914e6739d3cfabfc0ab1d1d4ebf0eeae8b454e40d8`；
`reboot20261023_mma3_small_g3f2_reverse_v1.local_acceptance.json`中
`passed=true/positive_at_every_adjacent_control=true`，actual原call keys、两阶段完整ELF、
原reference和ownership均通过。parent2006760退出，production恢复正式MMA3/3源代码。

开始`mma3_first2_followup_v1.py run`：追加E64 seed17、E65 seed0/17各27项BYTE_CHECK、
同pointer动态graph和完整MoE；随后验证candidate自身8项outside-scope完整ELF。
每个child限制120秒，验证耗时不计入性能。E64 seed17已通过，22020096 B完整MoE零字节差异。
全部验证通过前不采用；正式最优仍82.0840 μs，正向候选中值81.4412 μs。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma3_first2_followup_v1.py run
# 完成后封存，下载evidence.tgz/receipt，再在本地独立accept。
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma3_first2_followup_v1.py seal
```

### MMA3/first2完整验收并采用（2026-10-23）

E64 seed17、E65 seed0/17各27项BYTE_CHECK、同pointer动态graph、完整MoE均零字节差异；
分别36.4344 /30.9060 /12.7442秒，属于验证耗时，不是性能数据。
candidate直接编译的GEMM2、FP8输入、original A、pin1、t64 fallback、w1x4、b3、persistent1
共8项完整ELF与各自control逐字节相同。E65保持分支原fallback，没有验证固定shared1 route，
也不声称E64/E65共用t256 kernel。

90成员、592268 B followup archive已下载并独立验收通过，SHA256为
`6462e6e5f6123062a9232dcff27de82811df7bbb02b3f231a4b48ddc0cf1909e`；
`mma3_first2_followup_v1.local_acceptance.json`为`passed=true`。
parent2066985退出，任务未遗留GPU/KFD进程。

`adopt_mma3_first2_v1.py`逐项检查正反向性能验收、严格验证、scope ELF、resources和已有文件SHA，
只将当前GEMM1 guard内`sched_first_group`默认3→2（+1/-1行）。
本地及a07-3均已保留该文件，SHA256为
`7f60e033a44445403f141253ff52dd79bca0078d76b988f4231a83c93802b0f7`；
上传文件与已测试candidate逐字节相同，采用前保存旧源代码备份。无Git清理、commit或push。
正式原入口成绩使用事先固定正向三轮中值81.4412 μs，反向确认81.8079 /81.7464 μs；
距离80 μs还差1.4412 μs。当前GEMM1完整ELF为`86d771a2...abc34a1`、6262条static instructions，
VGPR797 /SGPR58 /LDS327680 B /private0 /spills0；GEMM2仍`4bd626e3...16be`。

新terminal候选已重新以MMA3/first2源码和完整ELF冻结；未复用旧MMA3/3的baseline或绕过SHA。
`mma3_first2_terminal_v1.plan.json`中tailcover/puretail源代码SHA分别为
`4128856c...65115fd` /`518ff540...24c2193`；各112项实际hint总量检查、56项outside-scope
完全相同，原MMA/DS/TDM数量保持。它们尚未编译、验证或产生性能结论。
旧MMA3/3 capture只用于提出候选，不用于声称新ELF的实际PC、延迟或瓶颈占比。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma3_first2_terminal_v1.py cpu tailcover
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma3_first2_terminal_v1.py native tailcover
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma3_first2_terminal_v1.py strict tailcover
```

### Terminal候选通过准备，puretail完整ELF相同（2026-10-23）

`tailcover`的CPU/native检查通过：VGPR797 /SGPR58 /LDS327680 B /private0 /spills0，
protected DS/TDM/payload/ScaleA/barrier数量和offset、ordered fences、WMMA/BF16/FP4/tanh数量保持。
静态executable instructions由6262变为6220，不以该计数推断性能收益。
candidate完整ELF为`6773d72e7973681e76cf739cd40fa22424614061a508f66cd92313371856837d`。
本地以两份实际ELF、ISA和resources重新计算native invariants，通过。
E64 seed0 strict 27项BYTE_CHECK、同pointer动态graph、完整MoE22020096 B均零差异，
验证35.9761秒，GEMM2完整ELF保持。正式源码未改变。

`puretail`也通过native和E64 seed0完整strict（36.9079秒，非性能）；
其完整GEMM1 ELF逐字节等同当前MMA3/first2的`86d771a2...abc34a1`，包括mainloop及epilogue。
本地完整ELF比较亦相同，static instructions仍6262。对当前target没有新的机器码，
不安排性能测试、不采用额外源代码修改，避免将测量噪声当作收益。

`reboot20261023_mma3_first2_tailcover_forward_v1`已开始，顺序MMA3/first2→tailcover→MMA3/first2，
每项完整三轮。原call keys/实际source与两阶段完整ELF准备通过；每child120秒，
GPU全机idle后测试，不并发GPU监控、编译或传输。收益尚待完整对照。

### MMA3/first2末尾closing覆盖的正向对照（2026-10-23）

`reboot20261023_mma3_first2_tailcover_forward_v1`已完整结束。前control try1、try2
分别有一轮ownership不合格，均整组三轮排除、中值null；采用第一个完整合格组try3。
候选和后control均为各自try1，完整结果如下：

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前MMA3/first2 control try3 | 81.9808 /81.9069 /81.7298 | 81.9069 | 63.7175 |
| tailcover candidate try1 | 81.7320 /81.2449 /81.3419 | 81.3419 | 64.3396 |
| 后MMA3/first2 control try1 | 81.0002 /83.3007 /81.5612 | 81.5612 | 64.5125 |

相对两侧改善0.5650 /0.2193 μs（0.6898% /0.2689%）。后control的83.3007 μs保留，
没有按数值排除或挑选其余两轮。257成员、1001451 B archive已下载并在本地独立复算通过，
SHA256=`5e4b52b9c53f45978c49fd9169cbd37bcd943aa30bc023b285ffeff94c64478d`；
`reboot20261023_mma3_first2_tailcover_forward_v1.local_acceptance.json`中
`passed=true/positive_at_every_adjacent_control=true`，SHA256为
`65f2eb2aff1c4b6d72af16ec252f05fdb0b957823dcdc3f72b4497611ea579a2`。
原reference、ownership、source、actual原call keys和两阶段完整ELF通过，production恢复
正式MMA3/first2，parent2216295退出。GEMM2完整ELF保持，不将此处耗时波动解读为其代码改变。

已启动反向`reboot20261023_mma3_first2_tailcover_reverse_v1`（parent2279048），顺序
candidate/control/candidate，各完整三轮；每个child120秒，计时前全机idle，等待间隔6秒。
`mma3_first2_tailcover_followup_v1.py`已在本地完成展开语法及source/timeout检查，尚未运行。
候选暂未采用，正式最优仍81.4412 μs；E65保留原fallback，未验证固定shared1 route。

### 末尾closing覆盖反向通过，追加验证（2026-10-23）

反向`reboot20261023_mma3_first2_tailcover_reverse_v1`已完成。首candidate try1和
control try1各有一轮ownership不合格，整组三轮排除、中值null；完整合格组为：

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前tailcover candidate try2 | 80.9539 /81.4578 /81.8521 | 81.4578 | 63.5511 |
| MMA3/first2 control try2 | 82.1663 /81.1330 /81.7763 | 81.7763 | 64.3671 |
| 后tailcover candidate try1 | 81.6687 /81.5781 /81.0277 | 81.5781 | 64.4913 |

反向分别改善0.3185 /0.1982 μs（0.3895% /0.2424%），正反向四项相邻比较均改善。
257成员、1000361 B archive已在本地独立验收，SHA256为
`6a5fedd45bef8976732246880d457b48793045746e6ef478bf3724223388112b`；
local acceptance为`passed=true/positive_at_every_adjacent_control=true`，SHA256为
`9bd2878242339d04583585c72ee7d9c6c500a445661ebc29487f58b527030c49`。
parent2279048退出，production恢复MMA3/first2。没有选择正反向最小单轮或重拼组。

`mma3_first2_tailcover_followup_v1.py run`（parent2337898）开始追加E64 seed17、
E65 seed0/17完整MoE、同pointer动态graph和8项candidate直接outside-scope ELF核验。
E64 seed17和E65 seed0已各27条BYTE_CHECK零差异，完整MoE22020096 B零差异；
分别35.9093 /30.8827秒，仅为验证耗时。E65仍原t64/w1x4 fallback。
所有验收通过前不采用，正式中值仍81.4412 μs；候选固定正向中值为81.3419 μs。

### 末尾closing覆盖完整验收并采用（2026-10-23）

追加的E64 seed17、E65 seed0/17分别35.9093 /30.8827 /12.6922秒完成；每项27条
BYTE_CHECK、同pointer动态graph和完整MoE22020096 B均零差异。E64 seed0已在准备及
原入口测试中验证。candidate直接编译GEMM2、FP8输入、original A、pin1、t64 fallback、
w1x4、b3、persistent1共8项完整ELF，与各自预验证control逐字节相同。
parent2337898退出，`gpu_users.sh --plain`未发现残留GPU/KFD进程。

88成员、590133 B followup archive已下载并本地独立验收通过，SHA256为
`db7530c3dd5d233517714a8f3ca0baa516f8412a68d507a6045847ace010389f`；
local acceptance SHA256=`0574b49ebf79287da7c53f34bf423b32d975a1596101f138844d710ed99e2316`。
`adopt_mma3_first2_tailcover_v1.py`只替换一行compiler-hint条件，采用前备份原source，
并核对正反向完整组、source/ELF、精度、outside-scope和resource验收。
上传本地source到任务candidate目录后，逐字节cmp已测试candidate，再在主机原子替换production。

本地与a07-3保留源码SHA256均为
`4128856c00e202a91ae0c2ae550108e98a3fd37ac3941806f6890881265115fd`，
两端采用记录SHA256均为
`55b11f33f3653b7c84eebce388d020bef3b5e6aa89b8e8dc995a93e3ebe80c97`。
正式成绩使用事前固定正向三轮中值**81.3419 μs**，反向确认81.4578 /81.5781 μs；
距离80 μs还差1.3419 μs，不选最低单轮或正反向最低组。
GEMM1完整ELF为`6773d72e...856837d`、`.text`为`4d299eb5...e64606c`，
6220条static executable instructions，VGPR797 /SGPR58 /LDS327680 B /private0 /spills0。
GEMM2仍`4bd626e3...16be`。未改Git index、commit、push、功能、shape准入或精度门槛。
E65仍分支原t64/w1x4 fallback，未验证固定shared1 route，未完成E64/E65共用t256要求。

下一轮以新正式source重新冻结MMA3/first1、MMA3/first4两项一行改动；各38项
outside-scope/显式env检查、56项实际hint总量检查通过。source SHA分别为
`1359d3339ec8938882a2a2a343d2b3640a4731332849c6d65901ade0207d46b1`、
`84dadaa93af14793c26c4fae75b9c02ba447c5375a911a2f338637ee47bd62f2`。
没有修改steady group、B-TDM TH、pipeline或device helpers，候选仍未编译或产生性能结论。

### 末尾覆盖正式版上的first-group邻近值准备（2026-10-23）

MMA3/first1、MMA3/first4均已完成CPU compile、native及E64 seed0完整strict。
每项27条BYTE_CHECK、同pointer动态graph和完整MoE22020096 B均零差异，
分别36.2774 /36.2835秒；这些是验证耗时，不作性能。两项保持GEMM2完整ELF。

| 候选 | static executable instructions | VGPR /SGPR /LDS B | GEMM1完整ELF SHA256 |
|---|---:|---|---|
| 正式MMA3/first2 + 末尾覆盖 | 6220 | 797 /58 /327680 | `6773d72e7973681e76cf739cd40fa22424614061a508f66cd92313371856837d` |
| MMA3/first1 | 6227 | 797 /58 /327680 | `9cc57fb661be05ec18fc065d034fd18f8004126710488efeaf1ef5feffe726b9` |
| MMA3/first4 | 6185 | 797 /58 /327680 | `7f2a1e3191972eb17339ec443c6e6923e37167c5aa554058042d29cf8dd5a87b` |

private及VGPR/SGPR spills均0。两项的DS/TDM/payload/ScaleA/barrier计数和offset、
关键fence顺序及WMMA/BF16/native FP4/tanh数量保持；本地下载实际ELF/ISA/resources后
重新复算native invariants亦通过。静态指令多少不能直接说明μs收益。
`mma3_first_neighbors_screen_v1.py`展开后的全部runner函数、env、timing和原CLI参数
AST与原runner一致；只适配plan、baseline SHA和记录名，不修改计时或精度口径。

`reboot20261023_mma3_first_neighbors_g3f1_forward_v1`（parent2396262）开始正向完整对照，
actual原call keys准备通过，尚无有效完整性能结果。完成后才顺序测试first4，
不并发GPU测试、编译、传输或监控；每child120秒，空闲等待6秒。
正式保留源码仍为末尾覆盖MMA3/first2，正式中值81.3419 μs。

### MMA3/first1正向完整对照通过（2026-10-23）

`reboot20261023_mma3_first_neighbors_g3f1_forward_v1`已完成，三组均为try1，
没有ownership排除；各组保留完整三轮，原reference、source、actual call keys和两阶段ELF均通过。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前MMA3/first2 control | 80.9874 /81.6156 /81.2405 | 81.2405 | 63.8565 |
| MMA3/first1 candidate | 81.0694 /81.6389 /81.0845 | 81.0845 | 63.9788 |
| 后MMA3/first2 control | 81.6833 /81.8898 /81.3798 | 81.6833 | 64.0652 |

相对两侧分别改善0.1560 /0.5988 μs（0.1920% /0.7331%）。188成员、762788 B archive
已下载并在本地独立复算通过，SHA256=`f7bebfb13bcb9d6f0e947063736c6c62797831a7d3ec02fcf84f901c04c61845`；
local acceptance SHA256=`e8881e4a8727b2752fd2216efb9443f7d55d77e0b33b554146310ef7827801a9`，`passed=true/positive_at_every_adjacent_control=true`。
production恢复末尾覆盖MMA3/first2 source `4128856c...65115fd`，parent2396262已退出。
候选尚未采用；下一步反向candidate/control/candidate完整对照，通过后再追加E64 seed17、
E65 seed0/17和candidate直接outside-scope验证。正式成绩仍81.3419 μs；first1的81.0845 μs为待反向确认候选。

### MMA3/first1反向未重复收益，不采用（2026-10-23）

`reboot20261023_mma3_first_neighbors_g3f1_reverse_v1`已完整结束，三组均为try1，
没有ownership排除，原reference、source、actual call keys与两阶段完整ELF核验通过。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前MMA3/first1 candidate | 81.5800 /81.5421 /82.1114 | 81.5800 | 64.2636 |
| MMA3/first2 control | 81.9765 /81.1372 /81.5653 | 81.5653 | 64.0738 |
| 后MMA3/first1 candidate | 81.2153 /81.7592 /82.2188 | 81.7592 | 64.4239 |

两侧候选分别比中间control慢0.0147 /0.1939 μs（0.0180% /0.2377%），
正向的0.1920% /0.7331%改善没有在反向重复。不重测挑选更快组，不以正向81.0845 μs
替换正式成绩，不采用first1，不继续该候选追加精度与scope；已准备的followup工具不执行。

188成员、762901 B archive已下载，本地独立复算原表格、ownership、source、keys、ELF
和严格输出通过。archive SHA256为
`1d70da4154401453695751178b593308b57d9836d355e631bc19ff869878cb54`。
local acceptance为`passed=true/positive_at_every_adjacent_control=false`；前者仅说明
证据一致，不能当作优化获得收益。parent2442919已退出，production恢复
正式source `4128856c...65115fd`。下一步顺序测试已经准备的MMA3/first4；正式中值仍81.3419 μs。

### DS/TDM hint取整分配：独立小候选准备（2026-10-23）

基于正式MMA3/first2 + 末尾覆盖源码，只在原`standard_a_timeout` scope内改变
`spread`的compiler-hint分配，分别测试DS或TDM从向上取整改为向下取整。
DS候选+3/-3行，TDM候选+3/-2行；不改group3/first2、ring、实际访存、
WMMA、activation/quant、barrier/wait或任何shape准入和生产env。

当前K256/subtile0的40项future DS hint由14/13/13改为13/13/14；目标配置subtile1
每wave有一项TDM，hint由slot0移为slot10。若每wave有四项TDM，通用分配才是
slots 0/2/5/8移为2/5/8/10。这个小变化用于检查现有ceil分配
对first group的影响；它不是硬件请求减少，也不能从hint数量推算性能。

初始560个真实hint AST组合使用通用TDM_PER=4，追加检查覆盖每wave TDM_PER=1/2/4，
每项共1680个组合，MMA、DS、TDM总量全部保持；其中840个scope关闭
组合的hint事件顺序逐项相同。去掉`spread/emit_hints`局部compiler-hint函数后的
完整module AST相同，原guard、ABI、cache-tag与所有runtime操作保持。
候选source SHA256分别为`778e000c13b222685e0026dc8c9a3731d5ca5a663427c39ef0dc09fac89523a6`
与`fd5f58ced250ab5b06abc8cd7cce4e8a6eb2501deac1abe0705015acec55c247`。

`prepare_mma3_spread_rounding_v1.py`和`mma3_spread_rounding_screen_v1.py`已展开检查：
原bounded child、native gates、完整MoE strict、idle、计时及原CLI AST均保持。
这些只是本地静态准备；first4性能运行结束前不上传、不编译或dispatch候选。
现阶段没有该方向的性能或设备精度结论，正式保留source和81.3419 μs成绩不变。

### MMA3/first4正向无两侧一致收益，不采用（2026-10-23）

`reboot20261023_mma3_first_neighbors_g3f4_forward_v1`完整三组均为try1，
没有ownership排除，原精度、source、actual call keys与两阶段完整ELF通过。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前MMA3/first2 control | 81.2931 /81.9026 /81.5737 | 81.5737 | 64.2004 |
| MMA3/first4 candidate | 81.8752 /81.5063 /81.0972 | 81.5063 | 64.6979 |
| 后MMA3/first2 control | 81.3735 /80.8485 /81.8647 | 81.3735 | 64.4344 |

候选相对前侧改善0.0674 μs（0.0826%），相对后侧回退0.1328 μs（0.1632%）。
不追加反向，不采用；减少35条static instructions没有带来本组稳定收益。
188成员、761691 B archive已下载并在本地复算通过，SHA256为
`ad02e31be795c62464e49b325e0bcfda985eb2fd3ac9b0f337c1f4b84c6265eb`。
local acceptance为`passed=true/positive_at_every_adjacent_control=false`。
parent2483960已退出，production恢复正式source `4128856c...65115fd`；
接下来顺序编译和验证上述DS/TDM取整候选。正式保留成绩仍81.3419 μs。

### DS/TDM取整候选完成准备，TDM不产生新ELF（2026-10-23）

两项均已完成bounded CPU compile、native检查和E64 seed0完整strict。
各27条BYTE_CHECK、同pointer动态graph、完整MoE22020096 B零差异；
分别36.2610 /36.7326秒，仅为验证耗时。GEMM2完整ELF均保持。

| 候选 | static executable instructions | VGPR /SGPR /LDS B | GEMM1完整ELF SHA256 |
|---|---:|---|---|
| 正式control | 6220 | 797 /58 /327680 | `6773d72e7973681e76cf739cd40fa22424614061a508f66cd92313371856837d` |
| DS floor | 6212 | 797 /58 /327680 | `8ee300381b32362f5abdf73fef71ad944af2c78c0089a1d179e85f4522c6164a` |
| TDM floor | 6220 | 797 /58 /327680 | `6773d72e7973681e76cf739cd40fa22424614061a508f66cd92313371856837d` |

private及VGPR/SGPR spills均0。CPU与实际strict运行的完整ELF一致，DS/TDM/payload/
ScaleA/barrier计数和offset、关键fence顺序、WMMA/BF16/native FP4/tanh数量保持。
下载两项ELF、ISA、resources、native/strict及原始日志后，本地
`accept_mma3_spread_rounding_preparation_v1.py`重新复算通过；额外将ISA machine words
与ELF的`.text`逐项核对，避免仅相信远端inspection记录。

TDM floor的完整ELF与control逐字节相同，不运行性能对照，不采用该源码改动。
DS floor的ELF发生变化，下一步只对其运行base/candidate/base，各完整三轮。
本地与远端production仍正式source `4128856c...65115fd`，正式中值81.3419 μs。
E65仍原fallback，尚未完成E64/E65共用GEMM1或固定shared1 route验证。

### 首个WMMA group之后再放future DS：本地静态准备（2026-10-23）

在DS floor完整对照期间，另以正式source准备一项独立hint-only候选
`mma3_ds_after_first_v1`。在已有`standard_a_timeout` scope且用户未显式设置
`AITER_FLYDSL_DS_FIRST_N`时，将首个future DS hint放在first2 WMMA group后；
首K128消费已有rmem前的DS hint保持，steady group3、closing24、DS ceil取整保持。
没有改变actual DS/TDM或WMMA计算，也不改变ring、barrier/wait及shape准入。

新增一项读取已有env的override判断，再替换一行hint选择，+2/-1行。
移除delta后的完整module bytes恢复，除`emit_hints`及该override赋值外的完整module
AST相同。3360个实际hint AST组合的MMA/DS/TDM总量全部保持；其中2520个scope关闭
或显式env组合的完整hint顺序相同。当前subtile0先执行hint first2、再14项future DS；
subtile1的唯一TDM hint保持，first2前的4项future DS hint改到first2后。

候选source SHA256为
`09d3f8b1a04404a21b9267ac23d1a76ae5a0f97fa05c5f012708cd9643a7cc55`。
prepare/screen的cpu/native/strict/run/seal/accept展开语法通过，原runner函数AST保持
（prepare仅规范化cache目录名），原env、timing、CLI及120秒timeout保持。
旧DS-only/8x8组合first0曾因native fence变化被拒绝，不能将旧结果当作新组合的ISA，
也不放宽新候选的native gate。当前仅本地准备，未上传、未编译、无GPU或性能结论。

### DS floor正向双侧通过，继续反向确认（2026-10-23）

`reboot20261023_mma3_spread_rounding_dsfloor_forward_v1`已完成。后control try1的r2
ownership不合格，该组三轮均保留但中值null；取第一个完整合格组try2。
其余两组为try1，原精度、source、actual原call keys及两阶段完整ELF通过。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前正式control | 81.5361 /81.5171 /81.0407 | 81.5171 | 64.2113 |
| DS floor candidate | 80.4186 /81.2958 /82.3817 | 81.2958 | 63.8759 |
| 后正式control try2 | 81.6520 /81.9894 /80.6229 | 81.6520 | 63.9960 |

相对两侧改善0.2213 /0.3562 μs（0.2715% /0.4362%）。candidate的82.3817 μs保留，
没有按耗时排除，也不以80.4186 μs单轮作为正式成绩。228成员、886410 B archive已下载
并在本地独立复算通过，SHA256为
`9629aeff2b501932dee9b65cdd454a5c4cb7d5bcd006d1070dd55b8d28df9c4b`。
local acceptance为`passed=true/positive_at_every_adjacent_control=true`。

parent2574771已退出，production恢复正式source `4128856c...65115fd`。
下一步固定candidate/control/candidate反向三组，各完整三轮；正反向全部通过后再追加
E64 seed17、E65 seed0/17完整MoE及candidate自身8项outside-scope完整ELF检查。
当前候选未采用，正式中值仍81.3419 μs；E65仍原fallback。

### DS floor反向通过，追加完整MoE与scope验证（2026-10-23）

`reboot20261023_mma3_spread_rounding_dsfloor_reverse_v1`已完成。中间control try1的r1
ownership不合格，整组三轮中值null；第一个完整合格组为try2，两个候选组均try1。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前DS floor candidate | 81.1671 /81.9535 /81.1924 | 81.1924 | 63.9349 |
| 正式control try2 | 81.8373 /82.0207 /80.9961 | 81.8373 | 63.7935 |
| 后DS floor candidate | 81.5634 /82.6449 /81.1711 | 81.5634 | 64.2490 |

反向两侧改善0.6449 /0.2739 μs（0.7880% /0.3347%），结合正向0.2715% /0.4362%，
四项相邻比较均改善。82.6449 μs保留，没有按数值重跑或选择最低单轮。
228成员、886356 B archive已下载，本地独立验收通过，SHA256为
`c8fbea5a1f45253b309528cf91bc681395f8eb05415324ae9b08f422a734925a`。
local acceptance为`passed=true/positive_at_every_adjacent_control=true`。
parent2625739已退出，production恢复正式source `4128856c...65115fd`。

`mma3_spread_rounding_dsfloor_followup_v1.py`已展开检查run/seal/accept，原strict helper
和env AST保持。下一步追加E64 seed17、E65 seed0/17完整MoE、同pointer动态graph，
以及candidate自身GEMM2/FP8/original A/pin1/t64/w1x4/b3/persistent1共8项完整ELF。
未完成验收前不采用；正式中值81.3419 μs，候选事前固定正向中值81.2958 μs。
E65仍原t64/w1x4 fallback，未完成共用GEMM1或固定shared1 route要求。

### DS floor完整验证通过并采用（2026-10-23）

追加E64 seed17、E65 seed0/17全部通过。各27条BYTE_CHECK、同pointer动态graph和
完整MoE22020096 B均零差异，验证分别35.9432 /30.9098 /12.6045秒；不作为性能数据。
candidate自身GEMM2、FP8、original A、pin1、t64、w1x4、b3、persistent1八项
CPU-only编译的完整ELF逐字节等于已核验control。全部child在120秒内结束，
parent2675718退出，主机gpu_users检查确认无GPU进程残留。

88成员、590520 B followup证据archive已下载并本地逐项复算，SHA256为
`74e764090e89defdaa0e650990dbb606c3ca02dfabbe72fb1c3f77d579139772`。
local acceptance `passed=true`，SHA256为
`26055d1cc145609bde18d511c8a073a8b907a359fd97014ecb6f3bfccef58a0f`。
`adopt_mma3_spread_rounding_dsfloor_v1.py inspect/apply`在本地和a07-3均通过，
两端source及采用记录SHA分别为`778e000c...523a6`与
`293d87a935105f318ad2e7d8d4d8ccf9415f85abbd8ba28b35270319551f38db`。
源文件原子替换并保留旧版backup，没有改Git index、commit/push或测试入口。

采用成绩固定为事前正向中值81.2958 μs；反向81.1924 /81.5634 μs仅作确认，
不挑最快组替换成绩。同boot初始baseline89.4465 μs，改善9.1124%，未达到80 μs。
GEMM1完整ELF为`8ee300381b32362f5abdf73fef71ad944af2c78c0089a1d179e85f4522c6164a`，
6212条static executable instructions；VGPR797/SGPR58/LDS327680 B，private/spills0，
GEMM2完整ELF仍为`4bd626e...716be`。E65仍原fallback，固定shared1 route未验证。

### DS-after-first基于新保留版重新冻结（2026-10-23）

原`mma3_ds_after_first_v1`仅为旧source上的静态计划，未编译或测试，不直接复用。
新`mma3_ds_after_first_v2`基于已采用DS floor source重新冻结；+2/-1行，
保留future DS floor、MMA3/first2、closing24、全部actual DS/TDM/WMMA及硬件fence。
已有scope且未显式设置DS_FIRST_N时，首批future DS hint移到first2后；
显式env、scope外事件顺序保持。3360组合总量一致、2520组合顺序一致，
完整nonhint module AST保持；新candidate SHA为
`1cf41261f2a061d0811f40e642b229ec0fed72871cd191b17d4d0d512ff2d3bd`。

`check_mma3_ds_after_first_v2.py`展开cpu/native/strict/run/seal/accept并核验继承AST，
原计时函数、env、CLI、120秒child不变。native对照已更新为DS floor的实际ELF，
不使用旧版6220条指令对照；不放宽fence或resource gate。下一步顺序CPU compile、
native检查、E64 seed0 strict，通过后运行固定正向三组三轮对照。

### DS-after-first native拒绝，转向closing相邻值（2026-10-23）

`mma3_ds_after_first_v2`在bounded CPU compile后被原native gate拒绝，唯一理由为
`ordered protected hardware fences differ`。candidate完整ELF为
`77a2fbc7bf51d3361e9185bd3b4b4f8e3afa2177325b81a163fa164a9007167c`，
static executable instructions为6239，control6212；VGPR797/SGPR58/LDS327680 B保持。
DS/TDM/payload/ScaleA/barrier计数与offset、WMMA/BF16/native FP4/tanh数量保持，
但出现5条额外的受保护`s_wait_dscnt 0x0`。本地下载ELF/ISA后复算machine words、
protected计数及fence diff，`review_mma3_ds_after_first_rejection_v2.py`确认拒绝依据一致。
没有device strict或性能结果，不放宽gate，不采用；本地/远端production保持DS floor。

下一批`mma3_floor_closing_neighbors_v1`事前固定顺序23、25、22、26，各只替换
`FENCE_COVER_MMA`的一行目标scope默认值，+1/-1行。没有改ring、actual访存/WMMA、
activation/quant、硬件fence、shape准入及显式env。各5040个真实hint AST组合
总量一致，2520个scope关闭组合顺序一致；移除这一hint assignment后的完整module AST相同。
prepare/screen继承原120秒child、idle、计时、CLI、source/key/ELF与本地archive验收函数，
展开检查通过。采用依然要求正反向四个相邻对照均改善，随后追加strict与scope验证。

### closing23正向没有双侧收益，不采用（2026-10-23）

closing23 CPU/native和E64 seed0 strict通过。27条BYTE_CHECK、同pointer动态graph及
完整MoE22020096 B零差异，strict36.1449秒；GEMM2完整ELF保持。candidate
source为`74f37d75...939f35`，完整ELF为
`b1606acd937e4a0d81e5f677fe0d0ad12deffc935fbc05256b95ed378348fabe`；
6238条static executable instructions，VGPR797/SGPR58/LDS327680 B、private/spills0。
本地复算ISA machine words、protected访存/fence/math及strict ELF全部通过。

`reboot20261023_mma3_floor_closing_c23_forward_v1`完整结束。后control try1
ownership不合格，整组三轮中值null；try2为第一个完整合格组，其他两组try1。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前DS floor control | 81.7969 /80.8145 /82.2441 | 81.7969 | 64.0189 |
| closing23 candidate | 81.5926 /81.0064 /81.6768 | 81.5926 | 64.0631 |
| 后DS floor control try2 | 81.5377 /82.3706 /81.3522 | 81.5377 | 63.7532 |

候选相对前侧改善0.2498%，相对后侧回退0.0673%（0.0549 μs），不追加反向、不采用。
241成员、969958 B archive已下载，本地独立原表格/ownership/source/key/ELF复算通过，
SHA为`9492d8416d86ec4081c6b0d3751299cdf24db7a7626444c83d013c452e1d491f`；
local acceptance SHA为`3c592ce59281e742a37bf54138624f0b2e705626a038ff6d20fc009e14f84b0b`。
`passed=true`仅为证据一致，`positive_at_every_adjacent_control=false`。parent2759091退出，
production恢复正式DS floor source、GPU检查无残留。随后顺序准备closing25。

### closing25完整对照回退，继续邻近值（2026-10-23）

closing25仅将既有scope的`FENCE_COVER_MMA`默认24改为25，+1/-1行。
CPU/native及E64 seed0 strict通过；27条BYTE_CHECK、same-pointer动态graph和完整MoE
22020096 B零差异。strict35.8097秒，GEMM2完整ELF保持。
candidate source=`4d6e7346...b3fe1`，GEMM1完整ELF为
`b6b2788b9a66cc35e72281dbb4498eee80d06b88baa8b0cf226695eb5fefaf6e`；
6193条static executable instructions，VGPR797/SGPR58/LDS327680 B、private/spills0。
本地下载并核对ISA machine words、访存/数学计数、offset和protected fences通过。

`reboot20261023_mma3_floor_closing_c25_forward_v1`完整结束。前control try1的r1
ownership不合格，整组三轮中值null；try2为第一个完整合格组。candidate及后control
各try1，没有按耗时排除样本或跨组拼接。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前DS floor control try2 | 81.2111 /81.6559 /81.1014 | 81.2111 | 64.1625 |
| closing25 candidate | 80.9623 /81.4852 /81.5778 | 81.4852 | 64.1961 |
| 后DS floor control | 81.2827 /81.7656 /81.2617 | 81.2827 | 64.2236 |

候选比前后control慢0.2741 /0.2025 μs，回退0.3375% /0.2491%；不追加反向，不采用。
205成员、849638 B archive已下载并本地从原表格、ownership、source/key/ELF逐项复算通过，
SHA=`3ad1c18315d9926b4718fac6bbdcd5b5590de6a206a8fd1ff4cea6dc53954303`。
local acceptance `passed=true/positive_at_every_adjacent_control=false`，SHA为
`644f05f7b7fa0d9c81ff9da7bc71cb51745ff3b7001530e6e50c9fc229246419`。
parent2868744退出，GPU检查无残留，production恢复正式source `778e000c...523a6`。
正式保留成绩仍81.2958 μs，不用本批更快control替换成绩。

后续closing22已完成CPU/native和E64 seed0 strict：VGPR797/SGPR58/LDS327680 B、
private/spills0，6243条static executable instructions；GEMM1完整ELF为
`f572a40d8f150f8aa91f20db0023bad0a0e057142f50bd86c978aecf8cb686b0`。
36.2043秒内27项BYTE_CHECK及完整MoE零差异，GEMM2完整ELF保持。
接着按事先固定顺序完成closing22、closing26的正常入口对照。

另在本地冻结`mma3_floor_dsfirst12_v1`、`mma3_floor_dsfirst8_v1`两项小候选，
只调整现有scope且未显式DS env时的首批future DS hint上限，各+2/-1行。
DS floor、MMA3/first2、closing24、实际访存/数学和hardware fences不变；没有增加shape限制。
分别3360项实际hint AST总量检查通过，2520项scope外或显式env完整顺序保持；
target subtile0原first13变为first12或first8，剩余1或5项DS移至first2后，
subtile1的原first3及每wave唯一TDM hint保持。
source SHA分别`93b8ecea...3bbb1e`、`a010feb3...77f36a`。
当前仅本地准备，尚未编译、device验证或计时，不与closing性能并发。

### closing22正向没有双侧收益，closing26完成准备（2026-10-23）

`reboot20261023_mma3_floor_closing_c22_forward_v1`完整结束。三组均为try1，
ownership/source/actual call keys和两阶段完整ELF通过，没有排除或拼接样本。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前DS floor control | 80.8716 /81.8802 /80.6975 | 80.8716 | 64.3817 |
| closing22 candidate | 81.2542 /80.9418 /80.6932 | 80.9418 | 64.2894 |
| 后DS floor control | 81.4584 /81.3321 /81.2516 | 81.3321 | 64.3548 |

候选相对前侧回退0.0702 μs（0.0868%），相对后侧改善0.3903 μs（0.4799%）。
不追加反向，不采用。169成员、731973 B archive已下载，本地从原表格、ownership、
source/key/ELF独立复算通过，SHA=`e3bc56a6f8eff9150eff5519c63ac0ffbae55a9bb59c91dae29009c80feb364c`。
local acceptance `passed=true/positive_at_every_adjacent_control=false`，SHA为
`b2fbf0dc4f46be0347a3056554c7008fd94182f8ed52dfa4cb5efa6201eb81eb`，两端一致。
parent2934984退出，GPU检查无残留，production恢复正式source `778e000c...523a6`。

按事前顺序继续closing26，source=`71d5b629...da62e2`。CPU/native和E64 seed0 strict通过，
27条BYTE_CHECK、same-pointer动态graph和完整MoE22020096 B均零差异，
strict36.7418秒；GEMM2完整ELF保持。GEMM1完整ELF为
`9a73b3a54c8c71231163407f4cb03ada24f2e28a66e85e44874985995562d1d3`，
6203条static executable instructions，VGPR797/SGPR58/LDS327680 B、private/spills0。
本地下载ELF/ISA/resources后重新复算machine words、protected访存/math/fences和
strict实际ELF通过。CPU parent2973159、strict parent2982931均退出。
当前尚无closing26性能结论；正式保留中值仍81.2958 μs。

DS-first12/8的候选及tools已上传，尚未编译或计时。修正任务私有
`accept_mma3_floor_dsfirst_preparation_v1.py`误继承closing的source delta模板：
现在逐字节核对DS env检测与first12/8替换，分别3360项总量、2520项scope外或显式env检查。
两feature展开AST、继承ISA signature函数和source delta gate离线检查通过；
不改生产kernel、计时或精度gate。后续仍按12→8顺序测试。

### closing26双侧回退，结束closing邻近值实验（2026-10-23）

`reboot20261023_mma3_floor_closing_c26_forward_v1`完整结束。前control try1的r1
ownership不合格，整组三轮保留、中值null；try2是第一个完整合格组。
candidate和后control均try1，不跨组拼轮、不按耗时重试。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前DS floor control try2 | 80.7855 /81.2978 /81.0236 | 81.0236 | 63.6311 |
| closing26 candidate | 82.8981 /81.5508 /82.1877 | 82.1877 | 64.4492 |
| 后DS floor control | 82.2086 /81.7744 /80.5429 | 81.7744 | 64.2912 |

候选相对两侧回退1.1641 /0.4133 μs（1.4367% /0.5054%），不追加反向、不采用。
203成员、848670 B archive已下载，本地从原表格、ownership、source/key/ELF独立复算通过，
SHA=`fa0b7b0c6af368e293d9b289720aa85af3933c8fdebe8c8a23a1a79f058d53fd`。
local acceptance `passed=true/positive_at_every_adjacent_control=false`，SHA为
`b54b961fab143e7df24bb5e50493ab66ec9105da65ba192c70affa589340a75a`，两端一致。
parent2996418退出，GPU检查无残留，production恢复正式source `778e000c...523a6`。
按事前固定顺序完成的closing23、25、22、26均无双侧收益；保留closing24。
正式保留成绩仍81.2958 μs，不用本批control成绩替换。

### DS-first12完整准备验收通过，开始正向对照（2026-10-23）

`mma3_floor_dsfirst12_v1`只调整既有scope且未显式DS env时的首批future DS hint，
target subtile0由13改12，余下一项移至first2后；subtile1顺序保持。
CPU/native检查通过，GEMM1完整ELF为
`bd649b6bce166dd13e40280ce6bcb0283165ff289fe6cdfaa47e82d4e96a30da`，
6233条static executable instructions；VGPR797/SGPR58/LDS327680 B、private/spills0。
protected访存计数和offset、ordered fences及WMMA/BF16/FP4/tanh计数保持。

E64 seed0 strict在36.9148秒内通过：27条BYTE_CHECK、same-pointer动态graph和
完整MoE22020096 B均零差异；GEMM2完整ELF保持`4bd626e...716be`。
CPU parent3043727和strict parent3049300均退出。本地下载actual ELF/ISA/resources及
原始strict记录，重新复算machine words、source delta和全部native/precision gates通过。
preparation local acceptance SHA为
`424eca634a58452afcd24f789b381a0b7e3c31d10c158d528352a5352cbb2fab`，两端一致。
以上均为准备与正确性结果，不作为性能数据。

已核验远端branch/HEAD、boot和正式source，GPU检查无进程后启动
`reboot20261023_mma3_floor_dsfirst12_forward_v1`，parent3064498。
仍沿用原入口、env、计时与120秒child，按base/candidate/base各完整三轮测试。
计时期间不传输、编译或并发GPU监控；双侧改善才反向，完整追加验证通过才采用。
DS-first8尚未编译、device验证或计时。E65仍原fallback，共用GEMM1和固定shared1 route
要求尚未完成。

### future DS hint容量的小改动准备（2026-10-23）

`audit_mma3_floor_ds_capacity_v1.py`从正式DS floor的实际ELF重新核对6212条executable
instructions的machine words。按branch编码确定四条steady loop的back edge，
各body内只有末尾back branch，均包含64条WMMA和两个连续K128 steps。
每step各32条payload `ds_load_b128`加4条Scale双地址DS load，合计36条DS instructions；
每loop仍有原1条TDM load和1对barrier。audit SHA为
`0b88e3e69e341fe9fa7601ccba44ef9d21b8d49e50248b04f1ff7e49f94ae283`。
这只覆盖四条具体steady body，不把初始block中额外单地址Scale loads也算成36。

本地LLVM HEAD为`7f77ca0dbda4abbf9af06537b2c475f20ccd6007`，与本地FlyDSL
`thirdparty/llvm-hash.txt`相同，但尚未证明与容器installed dev882的LLVM相同。
该源码`AMDGPUIGroupLP.cpp`的`SchedGroup::isFull`按SUnit数量判断容量，
`canAddMI`以`MI.mayLoad() && TII->isDS(MI)`识别DS_READ；
`SILoadStoreOptimizer::mergeRead2Pair`把两个load合并为一个双地址instruction。
这提供compiler hint容量实验依据，不证明installed compiler或必然有性能收益。
硬件依据为CDNA5 ISA §5.7.1.4（p65）及§11.2（p149）：DScnt按指令issue/completion
增减，双地址load返回两个值。compiler hint不替代硬件wait，不据此删除同步。

新`mma3_floor_dscap36_v1`仅在既有scope且没有显式DS env时，将future hint容量改为
`STATE_DS - (sb_pairs + sa_pairs) // 2`；target由40变36，初始hint仍原STATE_DS。
只改hint，不删除actual A/B/Scale loads，不改WMMA、ring、barrier、quant或shape准入。
相对正式source +3/-1行，candidate SHA为
`3b0e5b738eef7b01e67a25a244bd89ed58d824463a4c2f1344bd97314afc278b`。
3360项actual hint检查通过；2520项scope外或显式DS env顺序保持。
移除hint后的完整module AST相同，prepare/screen的原child/env/timing/CLI和ISA signature
函数AST保持；native/precision/performance gates不放宽。
当前仅本地冻结，尚未上传、编译、device验证或计时；仍按DS-first12→8完成后再测试。

### DS-first12正向双侧改善，继续反向确认（2026-10-23）

`reboot20261023_mma3_floor_dsfirst12_forward_v1`完整结束，parent3064498退出，
production恢复正式source `778e000c...523a6`。前control try1/2和candidate try1/2/3
因post ownership/KFD/VRAM不合格整组三轮排除，中值null；原始记录全部保留，
不按耗时选择组、不跨组拼轮。第一组完整合格结果如下。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前DS floor control try3 | 81.5655 /81.0785 /81.8774 | 81.5655 | 63.7514 |
| DS-first12 candidate try4 | 81.2849 /81.0521 /81.8535 | 81.2849 | 64.0915 |
| 后DS floor control try1 | 81.4065 /81.0080 /81.7061 | 81.4065 | 64.2757 |

candidate相对两侧改善0.2806 /0.1216 μs（0.344018% /0.149374%）。380成员、1363794 B
archive已下载并逐成员核验，SHA=`4a1770dc978b4eda7b94fa6e71180a5115a604c0040377b62e5b268c1328e89d`。
本地从原表格、ownership、source/key/ELF独立复算通过，local acceptance
`passed=true/positive_at_every_adjacent_control=true`，SHA为
`fc197b23cbc5f6e4080a254ae5c75daaf0a58c70ad09575d200fd0355d29020a`。

按既定流程继续`reboot20261023_mma3_floor_dsfirst12_reverse_v1`，顺序candidate/base/candidate，
各完整三轮；反向两侧仍改善且追加完整precision/scope验证通过后才采用。
正向收益较小，目前不据此替换正式版。正式保留成绩仍81.2958 μs，距离80 μs为1.2958 μs。
E65仍原fallback；E64/E65共用GEMM1和固定shared1 route要求尚未完成。

### DS-first12反向双侧回退，放弃；继续DS-first8（2026-10-23）

`reboot20261023_mma3_floor_dsfirst12_reverse_v1`完整结束，parent3172475退出，
production恢复正式source `778e000c...523a6`。中control try1、后candidate try1因post
ownership不合格各整组排除，中值null；不按耗时排除有效轮、不跨组拼接。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前DS-first12 candidate try1 | 81.9213 /81.2572 /81.3079 | 81.3079 | 64.0757 |
| 中DS floor control try2 | 81.2194 /81.2804 /81.8962 | 81.2804 | 63.8691 |
| 后DS-first12 candidate try2 | 81.8624 /82.3095 /81.3606 | 81.8624 | 63.9072 |

相对中control，前后candidate回退0.0275 /0.5820 μs（0.033833% /0.716040%）。
正向小幅收益没有在反向重复；不采用、不运行追加followup或adoption。
272成员、1007876 B archive已下载并独立复算，SHA为
`5f59243bb9177cfc275ab11db155c4afabe0a8e25d1784c1f191843e8c86ed8e`。
本地验收`passed=true/positive_at_every_adjacent_control=false`，SHA为
`2524021bde720ec21184ae8b1a0bd0fab1230401eca9122fcf20d543f7f10257`。
结束GPU检查无进程残留。正式保留成绩仍81.2958 μs。

按事前固定12→8顺序开始`mma3_floor_dsfirst8_v1`的CPU-only准备，
source `a010feb3...77f36a`，parent3233558；native、E64 strict和本地证据验收通过后
才测原入口性能。之后测试已冻结的future DS capacity36候选。

### DS-first8完整准备验收，开始原入口对照（2026-10-23）

`mma3_floor_dsfirst8_v1` CPU/native与E64 seed0 strict均通过，CPU parent3233558、
strict parent3237698退出。6230条static executable instructions，GEMM1完整ELF为
`8a55075d331a849b1495274290424cdf322dd07ac489f8641c80e0137a493195`；
VGPR797/SGPR58/LDS327680 B、private/spills0。protected native计数、offset和fence保持。
strict36.0071秒完成27项BYTE_CHECK、same-pointer动态graph和完整MoE22020096 B零差异；
GEMM2完整ELF保持`4bd626e...716be`。

下载actual ELF/ISA/resources与原strict日志后，本地独立machine-word、source delta、
native和精度验收通过。preparation acceptance SHA为
`c5757cc13b69975e579ccb93c97ae0c553111baf147bff19d883e9c42ce73be8`，两端一致。
确认全机idle后开始`reboot20261023_mma3_floor_dsfirst8_forward_v1`，parent3248097，
base/candidate/base各完整三轮。当前尚未取得候选完整合格组，不把排除轮计为性能成绩。

### 当前组合上的M64无效计算跳过：仅本地静态准备（2026-10-23）

历史M64/M32 skip基于较早DS版本，均回退；不把这些记录解释成当前MMA3/first2+
closing24+DS floor的实测结果。新`mma3_floor_empty_m64_v1`基于当前正式source，
复用`standard_a_timeout`原geometry guard，只在`k_step`增加两处wave-uniform M64行组边界判断，
相对正式版+4/-2行。部分有效M64组仍执行所有WMMA；完全padding的组才跳过。
有效行原K128累加与inline-zero初始化保持，input TDM/DS/ScaleA/ScaleB、rmem、
issue/fence顺序、quantization和输出helper保持。不新增pipeline、layout、persistent、
env或shape限制；GEMM2和原scope外不受影响。

24720项实际`k_step` AST边界/加载方式/初始化/fence组合检查通过，关闭scope时完整调用序列相同。
source SHA为`c34160c197607e1eb23a0d7bb950b1420c72c092bf1acca68c9d8d4bdb67b743`。
该候选仅本地冻结，尚未上传、编译、device或计时；完成DS-first8和capacity36后才评估。
编译后必须确认真实branch跳过完全无效组，不能靠EXEC masking声称跳过WMMA；
硬件依据为Shader Guide rev65 §4.6.12（p144）。native gate不放宽。

### DS-first8正向双侧改善，继续反向复测（2026-10-23）

`reboot20261023_mma3_floor_dsfirst8_forward_v1`完成，parent3248097退出，
production恢复正式source `778e000c...523a6`，结束GPU检查无测试进程残留。
candidate try1/2/3各有post ownership不合格，均整组三轮排除、中值null；
try4是首个完整合格组，不按耗时重试、不跨组拼接。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前DS floor control try1 | 82.3558 /82.8639 /81.8352 | 82.3558 | 64.0781 |
| DS-first8 candidate try4 | 80.8715 /80.8504 /81.4220 | 80.8715 | 63.7260 |
| 后DS floor control try1 | 81.9468 /81.8541 /81.5884 | 81.8541 | 64.3776 |

相对两侧改善1.4843 /0.9826 μs（1.802302% /1.200429%）。305成员、1124959 B
archive已下载，SHA=`2ffd4ce602312e5d17cdf5cff0e224fa0e975595476ede379c86a20f316a7f16`。
本地从原始表格、ownership、source/key/ELF独立复算通过，验收为
`passed=true/positive_at_every_adjacent_control=true`，local acceptance SHA为
`a958ffe6c1d0dcaca68a0c359033c322d1e88abe4cfb0a8a3aad1881c29b7ff4`，已回传且两端一致。

继续`reboot20261023_mma3_floor_dsfirst8_reverse_v1`，parent3318310，
顺序candidate/base/candidate，各完整三轮。候选正向中值80.8715 μs仍需反向与
追加precision/scope确认，正式保留成绩仍81.2958 μs。
匹配DS-first8 source/ELF的followup/adoption工具已在本地准备，尚未上传或执行；
不能使用DS-first12的工具采用DS-first8，也不能因单轮接近80 μs就采用。

### DS-first8反向双侧回退，放弃（2026-10-23）

`reboot20261023_mma3_floor_dsfirst8_reverse_v1`完成，parent3318310退出，
production恢复正式source `778e000c...523a6`，GPU检查无测试进程残留。
本序列全部三组try1完整合格，九轮均保留，无按耗时排除或补测。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前DS-first8 candidate | 82.1984 /82.3292 /81.3192 | 82.1984 | 64.1524 |
| 中DS floor control | 81.4307 /82.0866 /81.3928 | 81.4307 | 63.8823 |
| 后DS-first8 candidate | 82.2236 /81.4455 /81.6480 | 81.6480 | 63.8043 |

前后candidate相对中control回退0.7677 /0.2173 μs（0.942765% /0.266853%）。
正向80.8715 μs未稳定复现，DS-first8不采用，不执行followup/adoption。
197成员、767706 B archive已下载并独立复算，SHA为
`afb47fca24ca911925f14638c2f790518108bccaf25ad67291d405b379efabb2`。
local acceptance `passed=true/positive_at_every_adjacent_control=false`，SHA为
`89604f01a6f13c62e11a7ec3f9c312cddd995cdc431c788a81c90ab03a6f766c`。
正式保留成绩仍81.2958 μs。

DS capacity36的candidate、plan与tools已上传并逐文件核SHA一致。
CPU parent3357065已完成，原native gate通过；继续E64 seed0 strict，parent3359761。
目前仅为编译与验证进度，尚无DS capacity36性能数据。

### DS capacity36准备验收通过，开始性能对照（2026-10-23）

`mma3_floor_dscap36_v1`的E64 seed0 strict在36.1831秒内完成，parent3359761退出。
27条BYTE_CHECK、same-pointer动态graph及完整MoE22020096 B均零差异；
GEMM1实际完整ELF为`b534fad27c2fe2b1b6fcbf53c4448955941e4680da9d622ef66286d3afd44c13`，
GEMM2保持`4bd626e...716be`。6244条static executable instructions，
VGPR797/SGPR58/LDS327680 B、private/spills0。
protected native操作计数及offset、ordered fences与WMMA/BF16/FP4/tanh计数保持。

下载actual ELF、ISA、resources和strict原始日志后，本地独立machine-word、source delta、
native与MoE验收通过；preparation local acceptance SHA为
`3d396f052c631318f5d3af7a779cbd1527122ea80e449ea76235bc1d9820b519`。
以上为准备与正确性结果，均不作为性能数据。
继续固定`reboot20261023_mma3_floor_dscap36_forward_v1`，base/candidate/base各完整三轮；
全机idle、原命令和精度门限保持，正向双侧改善才反向确认。

### DS capacity36完整对照回退，放弃；开始M64编译（2026-10-23）

`reboot20261023_mma3_floor_dscap36_forward_v1`完成，parent3370166退出，
production恢复正式source `778e000c...523a6`，结束GPU检查无测试进程残留。
前control try1因r2 post ownership不合格整组排除、中值null，try2是首个合格组；
candidate与后control均try1完整合格。不按耗时排除有效组或跨组拼接。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前DS floor control try2 | 82.7714 /80.9331 /81.6308 | 81.6308 | 64.3081 |
| DS capacity36 candidate try1 | 81.7996 /82.9422 /83.0182 | 82.9422 | 64.1986 |
| 后DS floor control try1 | 80.6715 /82.3078 /80.8464 | 80.8464 | 64.3373 |

candidate相对两侧回退1.3114 /2.0958 μs（1.606501% /2.592323%），不反向、不采用。
237成员、894155 B archive已下载并独立复算，SHA为
`bf5838819cc36c7a1460bafa79d23aad859423bab6380466cc61ae1baad43e5f`。
local acceptance `passed=true/positive_at_every_adjacent_control=false`，SHA为
`930cede2f98cfc02f540acdd67c2ab2d7d82be31e791c64715ecff3dfdba6bab`。
formal保留成绩仍81.2958 μs，不用本组control的80.8464 μs替换正式采用成绩。

M64候选及原CPU/native/strict wrapper和有限CFG检查已上传，逐文件核SHA一致。
开始`mma3_floor_empty_m64_v1` CPU-only编译，parent3419603；
原native gate与有限CFG检查通过后才device验证。尚无该候选性能或硬件正确性结论。

### M64候选在native检查停止，不进入GPU（2026-10-23）

`mma3_floor_empty_m64_v1` CPU-only编译完成，parent3419603退出。原native gate拒绝，
原文理由为`ordered protected hardware fences differ`和`resource increase: sgpr_count`。
没有运行device strict、有限CFG或性能；不放宽原gate，不采用这项候选。

| 项目 | 正式DS floor | M64候选 |
|---|---:|---:|
| executable static instructions | 6212 | 6167 |
| VGPR /SGPR | 797 /58 | 795 /59 |
| LDS /private /spills | 327680 B /0 /0 | 327680 B /0 /0 |
| `s_wait_dscnt 0x0`静态处数 | 18 | 21 |

candidate完整ELF=`12c3ba4b6a2a360b4bf508c76bf84e109f5f79ab2131859d5443e719cc8de1a3`。
实际ELF/ISA、artifact、resources、native拒绝记录已下载，本地
`review_mma3_floor_empty_m64_rejection_v1.py`重新核machine words、资源与fence差异通过。
DS/TDM操作计数和offset、WMMA/BF16/native FP4/tanh静态计数保持，
但这些有限检查不能替代branch/lifetime证明，也不能说明该候选会更快。
production两端仍为`778e000c...523a6`；正式保留成绩仍81.2958 μs。

### DS floor组合上的WMMA group小候选（2026-10-23）

新`mma3_floor_group_neighbors_v1`固定顺序`g4f2`、`g2f2`、`g5f2`。
只替换已有GEMM1 geometry guard中`sched_mma_group`的一个默认数字（各+1/-1行），
保留first2、closing24、DS floor、有效数学、所有input/output和硬件fence。
显式公共MMA env仍优先，没有新增shape准入限制、env开关、pipeline、layout或persistent。

| feature | candidate source SHA256 |
|---|---|
| g4f2 | `bb263b22f049800d0956f65f02c44dea64dc2a98437cc9399abf87217720b657` |
| g2f2 | `36178ad89a4d4c3cda46514289b10574aaf1938f4eed0f0329e3cd6eedb406b0` |
| g5f2 | `3dda7441b5ac7afabc85a727d59f4b74afec23dcddec0c1deb51f2a1e6359f85` |

每项3360个实际hint AST组合总量检查、2520个scope外/显式env完整顺序检查通过；
20项actual group guard和env检查通过。去掉这一默认值的变化后完整module AST相同。
展开cpu/native/strict及run/seal/accept，继承的timing、env、原CLI、ownership、precision、
native gate和120秒child保持。上述仍是静态准备，不是性能收益；原native和device验证
通过后才运行全机idle的正向完整三轮对照，双侧改善才反向确认。
E65仍走原t64/w1x4 fallback，共用GEMM1及固定shared1 route要求仍未完成。

### Group4/first2完整对照双侧回退，不采用（2026-10-23）

`mma3_floor_group_g4f2_v1` CPU/native与E64 seed0 strict通过，完整ELF为
`ca2f713e0e34bc92e5e0758448debe8dda8bf493c778912fec30ca3a9fe9b571`。
VGPR797/SGPR58/LDS327680 B、private/spills0保持，executable instructions6182。
27项BYTE_CHECK、same-pointer dynamic graph及完整MoE22020096 B零差异；strict child36.1061秒。
下载actual ELF/ISA/resources/strict日志后，独立preparation acceptance通过。

`reboot20261023_mma3_floor_group_g4f2_forward_v1`完成，parent3551707退出。
后control try1的r1、r3 ownership不合格，整组中值null；try2是首个完整合格组。
前control及candidate均try1完整合格，不按耗时重试或跨组拼轮。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前DS floor control | 81.7677 /81.1499 /80.8379 | 81.1499 | 64.5397 |
| group4/first2 candidate | 82.1641 /81.1858 /80.6038 | 81.1858 | 63.8651 |
| 后DS floor control try2 | 80.7346 /80.3889 /81.6203 | 80.7346 | 63.6796 |

candidate双侧回退0.0359 /0.4512 μs（0.044239% /0.558868%），不反向、不采用。
234成员、888250 B archive已下载并逐member SHA、原表格、ownership、source/keys/ELF复算通过，
archive SHA=`f1549b33ec85cb06ceb49cb688df251938f510942256d12860ac84b4fde31897`。
production恢复正式`778e000c...523a6`，结束GPU检查无测试残留。
本组control中值80.7346 μs不替换正式采用成绩81.2958 μs。
按事前固定顺序开始group2/first2 CPU-only准备，parent3596928；尚无新性能结论。

### Future DS居中取整：仅本地小候选准备（2026-10-23）

`mma3_floor_dscenter_v1`仅在现有DS floor scope且未显式设置DS env时，把future DS
分配由floor改为居中取整。target subtile0从13/13/14变为13/14/13；subtile1
11个slot的40项hint也只重新分配。首批DS hint、DS/TDM/WMMA总量、实际访存和全部fence保持。
相对正式源码+2/-1行，source=`2e45a50c1ee2c62f912cf4ff9092057ded0ab6ddc09f11e1c72f29192ab73c0e`。
3360项actual hint总量检查、2520项scope外或显式DS env完整事件顺序检查通过，
移除hint helper及新增override flag后完整module AST相同。
这项仅本地冻结，尚未上传、编译、device验证或计时；固定group候选完成后才评估。

### Group2/first2完成准备，进入原入口性能对照（2026-10-23）

`mma3_floor_group_g2f2_v1` CPU/native与E64 seed0 strict均通过，CPU parent3596928、
strict parent3601192已退出。完整GEMM1 ELF为
`c061a2677dfe515810fb509499fe3d5cdb391db245ef03af7b408a9dfc53fe63`，
VGPR797/SGPR58/LDS327680 B、private/spills0保持，executable instructions6307。
protected native计数、offset和ordered fences保持；strict36.5507秒完成27项BYTE_CHECK、
same-pointer dynamic graph及完整MoE22020096 B零差异，GEMM2完整ELF保持。

下载actual ELF/ISA/artifact/resources/native与strict原始日志后，本地独立preparation验收通过，
SHA=`7cf8eab82449741bda31adb79d7321f712ed9c7cf11a735a09e6f2fce253061a`，回传后两端一致。
开始固定正向`reboot20261023_mma3_floor_group_g2f2_forward_v1`，parent3620535；
base/candidate/base各完整三轮。准备检查不能当作性能收益，正式成绩仍81.2958 μs。

DScenter的cpu/native/strict及run/seal/accept wrapper已在本地展开核验：原child env、
timing、CLI、ownership与native/precision gate的函数AST保持，120秒child限制保持。
尚未上传或执行该候选，不构成硬件或性能结论。

### Group2/first2正向双侧改善，继续反向确认（2026-10-23）

`reboot20261023_mma3_floor_group_g2f2_forward_v1`完成，parent3620535退出，
production恢复正式`778e000c...523a6`。九轮全部try1合格，没有排除或按耗时重试。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前DS floor control | 81.8710 /81.5673 /81.7466 | 81.7466 | 63.7767 |
| group2/first2 candidate | 81.2597 /82.1326 /81.3545 | 81.3545 | 63.6628 |
| 后DS floor control | 82.0587 /82.2421 /80.4732 | 82.0587 | 63.9895 |

candidate双侧改善0.3921 /0.7042 μs（0.479653% /0.858166%），尚未采用。
198成员、772292 B archive已下载并独立复算，SHA为
`c70e543f8c96da99d0cbee61bb695fa007f1ca90cf743b8b15f56ad735206c63`。
local acceptance为`passed=true/positive_at_every_adjacent_control=true`，SHA为
`0c2f970cf1b26687a472742cea4d35dcac385d26f9c29b31fda939ba8ea3e8b0`，回传后两端一致。
继续固定`reboot20261023_mma3_floor_group_g2f2_reverse_v1`，parent3663603，
candidate/base/candidate各完整三轮。正式成绩仍81.2958 μs。

### 冗余expert二分探测：仅本地静态候选（2026-10-23）

`mma3_floor_bisect_v1`复用现有GEMM1 geometry guard和`has_work`判断。
`has_work`已证明`blk_m < tile_map[n_experts - 1]`，目标scope中没有expert sentinel这个合法结果，
因此把初始hi从E降为E-1、固定探测轮数从`ceil(log2(E))+1`降为`ceil(log2(E))`，
E64为七轮降为六轮。scope外保持原七轮；没有修改输入输出、数学、同步、layout或dispatcher。
源码仅+2/-2行，SHA为
`6e7128f0e436e1df4d79e90f153724418f15a0bd966b8b8e4e72f608a9db476d`。

实际expert搜索AST在E1～E2048、零计数重复psum、所有边界及random位置共185844项检查中
与`bisect_right`一致，其中92922项scope外保持原行为。原module AST移除两行变化后相同。
构建工具首次误选了同名`has_work`的第二个分支，仅修正静态工具选择位置后通过；
该失败没有写出candidate或运行GPU。此项尚未上传、编译或测试，不能据少一轮probe推断耗时。

正式DS floor版的匹配trace工具亦已在本地准备，绑定source `778e000c...523a6`、
GEMM1 ELF `8ee30038...6164a`和6212条executable instructions；尚未上传或采集。
原MMA3/3 capture的PC与比例仍只属于其原ELF。

### Group2/first2反向双侧改善，追加精度与scope验证（2026-10-23）

`reboot20261023_mma3_floor_group_g2f2_reverse_v1`完成，parent3663603退出，
production恢复正式`778e000c...523a6`。前candidate try1/2/3及后candidate try1
均因ownership不合格整组排除，中值null；不按耗时排除有效组或跨组拼轮。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前group2/first2 candidate try4 | 80.8314 /81.3480 /81.2954 | 81.2954 | 63.9222 |
| DS floor control try1 | 81.3944 /82.3454 /81.8523 | 81.8523 | 64.0801 |
| 后group2/first2 candidate try2 | 81.3757 /81.1120 /81.7762 | 81.3757 | 64.4029 |

反向两侧改善0.5569 /0.4766 μs（0.680372% /0.582268%），正反向四项相邻比较均改善。
342成员、1250515 B archive已下载，本地逐member SHA、原表格、ownership、source/keys/ELF
复算通过。archive SHA=`e005f2733f9a03cb850ab33bd28a4bf5774c7d980d8df557f542a5e5cc70e4c3`；
local acceptance SHA=`328b76015d41f0a03322aa583e72b64747029ee14e87ccbd49954a3f9adc2bde`，
回传后两端一致。尚未采用，也没有低于80 μs的完整候选组。

`mma3_floor_group_g2f2_followup_v1.py`已上传核SHA，开始追加E64 seed17、
E65 seed0/17完整MoE strict及8项outside-scope完整ELF比较，parent3747943。
每个child保持120秒timeout。E65仍为原t64/w1x4 fallback；这项验证不能当作
E64/E65共用GEMM1或固定shared1 route已完成。正式保留成绩仍81.2958 μs。

### Group2/first2完整验收后保留，重冻结剩余group5（2026-10-23）

追加验证parent3747943已退出：E64 seed17、E65 seed0/17各27项BYTE_CHECK、
same-pointer dynamic graph和完整MoE22020096 B均零差异，child分别36.6258 /30.8740 /12.7610秒。
8项candidate直接outside-scope完整ELF均等于既有control，其中GEMM2保持
`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。
91成员、592952 B followup archive已下载复算，SHA为
`59089017bad466ab93ae6eb7bd0dc14748ee3e654787113afaff5045745b8d70`；
local acceptance SHA=`8aca7f12b7818f1f344c32d41d63a560b0df1334a343945dcfebcf713e5fc1b9`，回传后一致。
测试结束gpu_users/KFD为空，无测试进程残留。

`adopt_mma3_floor_group_g2f2_v1.py inspect/apply`在两端通过。新源码
`36178ad89a4d4c3cda46514289b10574aaf1938f4eed0f0329e3cd6eedb406b0`；
新增源码改动+1/-1行，production adoption record两端同SHA
`3e85ab6d31c7adfb3a1081e6baab42274f532dab02905dc509f680090b263cbe`。
完整GEMM1 ELF=`c061a2677dfe515810fb509499fe3d5cdb391db245ef03af7b408a9dfc53fe63`，
VGPR797 /SGPR58 /LDS327680 B、private/spills0；executable static instructions6307。
本轮正向中值81.3545 μs，反向81.2954 /81.3757 μs；不挑最快确认组改写成绩。
相邻对照有小幅重复收益，历史81.2958 μs保留，不能声称绝对耗时进一步下降或达标。
E65仍原fallback，共用GEMM1和fixed shared1 route要求仍未完成；没有commit/push或index变更。

按事前剩余顺序继续group5/first2，但重新绑定当前group2 source/ELF/native control，
使用`mma2_floor_group5_v1.plan.json`和新cache，不把旧group3 baseline套到新production上。
candidate=`mma2_floor_group_g5f2_v1`，source为
`3dda7441b5ac7afabc85a727d59f4b74afec23dcddec0c1deb51f2a1e6359f85`。
3360项hint总量、2520项scope外/显式env、20项guard检查通过，原runtime和完整nonhint AST保持。
cpu/native/strict及run/seal/accept wrapper展开核验通过，原120秒child、CLI、timing、ownership
与精度/native gate保持。seal没有source常量，首次检查仅修工具对seal的常量存在假设，未执行GPU。
八个任务文件上传后逐SHA一致，开始CPU-only编译parent3791306；尚无group5性能结论。

### Group5/first2完整对照未通过双侧改善，放弃（2026-10-23）

重新绑定group2 control的`mma2_floor_group_g5f2_v1`完成CPU/native与E64 seed0 strict。
GEMM1完整ELF为`f1a6f51dede9a519b1982ca5d83398a428a4f57c0a49e65894ce97dbf4fffae6`，
VGPR797 /SGPR58 /LDS327680 B、private/spills0，executable static instructions6149。
protected访存计数和offset、ordered fences及WMMA/BF16/native FP4/tanh数量保持。
strict child36.7351秒完成27项BYTE_CHECK、same-pointer dynamic graph及完整MoE22020096 B零差异；
GEMM2完整ELF保持。实际ELF/ISA/resources及strict日志下载后，preparation acceptance通过，
SHA为`25b23b346092b1d87be8b80daebb90a7d5c365f55226c1e17258c3e1df976fe0`，两端一致。

首次`forward_v1`因screen wrapper未转交CLI参数，在benchmark启动前退出，没有性能数据。
修正`sys.argv = [str(runner), *arguments]`，新增wrapper AST v2检查后使用独立`forward_v2`。
失败日志保留，未修改原测试或计时。parent3809426已退出，production恢复正式`36178ad8...406b0`。
candidate try1的r2、try2的r3 ownership不合格，均整组排除、中值null；try3是首个完整合格组。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前group2 control try1 | 81.6223 /82.2949 /82.1894 | 82.1894 | 64.0990 |
| group5/first2 candidate try3 | 81.4621 /81.7025 /81.2195 | 81.4621 | 63.8716 |
| 后group2 control try1 | 81.4642 /81.3336 /81.2028 | 81.3336 | 64.1012 |

candidate相对前侧改善0.7273 μs（0.884907%），相对后侧回退0.1285 μs（0.157991%）。
不反向、不采用；当前正式成绩仍为group2本轮正向81.3545 μs，未达到80 μs。
274成员、1008765 B archive已下载并独立复算原表格、ownership、source/key/ELF，
SHA为`8e6407001b15f75db36d3f7148b9cd8809341eaf3765b9c62b724334b1cfdcfc`。
local acceptance为`passed=true/positive_at_every_adjacent_control=false`，SHA为
`9600a8ef8ed56179ab7bf21b0ecbcd6736b080818355ab92355e0075a9c97b21`。
结束gpu_users/KFD检查为空，无测试进程残留。

### 基于正式group2重冻结DScenter与trace准备（2026-10-23）

`mma2_floor_dscenter_v1`重新绑定正式source `36178ad8...406b0`及GEMM1 ELF `c061a267...fe63`。
相对正式版+2/-1行，source为`671ece89ebe81abb880b795474b1ed8275dedf11229e0ee27ff6a0e4e12665df`。
3360项hint总量、2520项scope外/显式DS env完整顺序及nonhint module AST核验通过。
group2的subtile0四组future DS为10/10/10/10，居中取整不改变该段；subtile1只重分配2/3。
原DS/TDM/WMMA、硬件wait/barrier和有效累加顺序保持；这项仍需实际native与device验证，
不能据静态hint数量推断性能。

DScenter cpu/native/strict及run/seal/accept wrapper展开检查通过，原120秒child、env、CLI、
timing、ownership和native/precision gate保持。group5放弃后继续评估此小候选。
expert二分查找少一轮此前在M-group32阶段未通过双侧对照，原记录保留；
新静态bisect候选尚未运行，不将已有失败方向作为取得收益的证据。

匹配group2正式版的Method B trace脚本已在本地重冻结，绑定当前source/ELF、6307条executable
instructions、当前candidate原call inventory及`_mg2_dsfirst16` symbol。保持原workload、
`--all-simd --iteration 28 --ana-att`及120秒child，省略`.out`和private cache。
当前group2尚未采集，不复用MMA3/3 capture的PC或比例。

### DScenter完整对照方向不一致，不采用（2026-10-23）

`mma2_floor_dscenter_v1` CPU/native与E64 seed0 strict通过，CPU parent3880484、
strict parent3884450已退出。GEMM1完整ELF为
`0ee639a0939057b826918277fe34cfe6e57212c65969d882e4ae254c8b13ba16`，
VGPR797 /SGPR58 /LDS327680 B、private/spills0，executable static instructions6302。
protected计数/offset及ordered fences保持；strict child36.0235秒完成27项BYTE_CHECK、
same-pointer dynamic graph及完整MoE22020096 B零差异。GEMM2完整ELF保持。
22文件preparation archive逐大小/SHA验收后，本地machine-word/native/precision验收通过，
preparation acceptance SHA为`ed9650473d65db3f03165a2580975215aa866677d46afac03fb10226b3fa1ed4`，
回传后两端一致。

`reboot20261023_mma2_floor_dscenter_forward_v1`完成，parent3893883退出。
candidate try1的r1 ownership不合格，整组中值null；try2为首个完整合格组三轮。
前后control均try1完整合格，没有按耗时重试或拼轮。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前group2 control try1 | 81.0108 /81.9005 /80.5068 | 81.0108 | 64.3564 |
| DScenter candidate try2 | 80.9622 /81.2279 /81.8879 | 81.2279 | 64.0633 |
| 后group2 control try1 | 81.6647 /81.2364 /81.2469 | 81.2469 | 63.9727 |

candidate相对前侧回退0.2171 μs（0.267989%），相对后侧改善0.0190 μs（0.023386%）。
不反向、不采用，production恢复正式`36178ad8...406b0`；正式成绩仍为81.3545 μs。
237成员、892173 B archive下载后独立复算通过，SHA为
`2c957d70d1501caefd238a2213c58178691518f9a573fea4bdd429a19ba99671`。
local acceptance为`passed=true/positive_at_every_adjacent_control=false`，SHA为
`c70df7a0727229cd13272d9f71b7d3a6e09a383d9d047f152ae9a9c8f213c2c7`。
结束gpu_users/KFD为空，无测试进程残留。

group5与DScenter均结束后开始正式group2的匹配Method B trace，
label=`reboot20261023_best_mma2_floor_att_v1`、host parent3946970。
四份工具上传后SHA及`bash -n`通过，source/boot/环境检查和每次capture前全机idle检查保持。
这是诊断采集，profiler下耗时不计为原入口性能；此时尚未完成capture和本地独立验收。

### 正式MMA2/first2 trace本地验收与K128分解完成（2026-10-23）

`reboot20261023_best_mma2_floor_att_v1` SIMD0～3 capture/decode均rc0，parent3946970已退出，
结束检查无测试GPU/KFD残留。source `36178ad8...406b0`、GEMM1 ELF `c061a267...fe63`及
GEMM2 ELF `4bd626e3...16be`绑定实际capture private cache与原call inventory通过。
237成员、7006628 B archive下载并逐字节验收，SHA为
`8933d7f8b48adfe2a098e1c633ddb3d851a50477c5a3b19b70279698dcc44d90`，未下载`.out/private cache`。
全部decoded PC/ISA、64 active waves及阶段统计独立复算与远端一致，规范化68个路径字段。
analysis acceptance为`passed=true`，SHA为
`bec80102c05d470d7c58a8856892227a423448c298c203754a1e28105bc89d6f`。

当前累计wave timestamp gaps中，mainloop67.4543%、prologue14.6912%、activation/quant/LDS14.5322%、
final input drain0.2426%、output descriptor0.1838%、output TDM/completion2.8959%。
`analyze_best_mma2_floor_ksteps_v1.py`重新绑定6307条executable ISA，56个K128区间精确相加为
1174401 cycles。多数steady中值310～323 cycles，最后两项348.5/253 cycles；
K-step record SHA为`2ab2a8a288c048627d66e9758065798785548120311be3d5df9a01deb7ee1363`。
比例/区间包含arbitration、coexecution及prefetch，不是GPU关键路径或硬件独占延迟，不换算μs收益。

完整分析新增于`my_code/moe_gemm1_hyg_dsr1_best_thread_trace_analysis.md`。
正式版本和成绩未变化，仍81.3545 μs，反向81.2954/81.3757 μs；未达80 μs。
后续评估当前MMA2上的first1、group1/first2；各仅一行默认数字变动，先native及完整MoE strict，
再原入口双侧三轮对照。此处尚未计时或采用，两阶段原功能/shape/精度与scope外支持保持。

### 当前MMA2上的first1/group1独立候选及首项准备（2026-10-23）

`build_mma2_floor_neighbors_v1.py`固定顺序g2f1→g1f2，分别仅修改现有guard的一行默认数字，
production保持正式`36178ad8...406b0`。每项3360项hint总量、2520项scope外或显式env顺序、
40项两参数guard/显式MMA env及完整nonhint module AST检查通过。
原geometry、operation、DS/TDM/WMMA总量及scope外分支保持，没有新增shape限制、env或pipeline。

| 候选 | 改动默认参数 | source SHA256 |
|---|---|---|
| `mma2_floor_neighbors_g2f1_v1` | first2→1，group2不变 | `226861d45b681668b6f0da997959d1e512ca18ec60f83cd28729edd829d28f67` |
| `mma2_floor_neighbors_g1f2_v1` | group2→1，first2不变 | `7535f35a7610531225e99e6b446e9534a20ea9ed82fb0841a35e938c3cd637b5` |

全部prepare/run/seal/accept wrapper静态展开后，child environment、timeout120秒、native/strict、
计时命令、GPU idle/ownership及独立验收函数AST保持。必要19项文件上传后大小/SHA全部一致，
包括当前trace的本地验收与文档；upload manifest SHA为
`3d52d4d578171864dd4378eb7746ba9f82446accfbc760b4ac496da359db634a`。

g2f1 CPU parent4004858、strict parent4012485已退出。native的protected opcode/offset、ordered fences
和WMMA/BF16/FP4/tanh counts保持；资源VGPR797 /SGPR58 /LDS327680 B、private/spills0，
executable instructions6349，完整GEMM1 ELF为
`98908070e2d1f73c27842d84fd374b16c3f88d8f7af5eb148507771ad5cfd704`。
strict35.7089秒完成27项BYTE_CHECK、same-pointer dynamic graph及完整MoE22020096 B零差异，
GEMM2完整ELF保持`4bd626e3...16be`。初筛仅E64 seed0，不替代候选采用前的追加兼容性验证。

33成员、396338 B preparation archive已下载、逐大小/SHA验收、native machine words及strict日志独立复算通过；
archive SHA为`b132aa15823e9a8bff8e1c8aab890dca7f1596ff555863d6c3e0ed346b9efdd9`。
local acceptance为`passed=true`，SHA为
`c10184264309faa852b2fc8be71a40829e9377ae4ad21f34e4c41f002597c274`，回传核SHA一致。

g2f1原入口正向label=`reboot20261023_mma2_floor_g2f1_forward_v1`、host parent4022273，
现正按base/candidate/base执行完整三轮组；GEMM1/GEMM2原call keys已分别准备核验。
此处还没有完整有效组中值，不报告候选收益，不采用。正式成绩仍81.3545 μs。

```bash
# a07-3主机；每个CPU/strict/performance child在dsr1_ep4并限时120秒。
# 固定label已经启动，重放必须使用新label及独立输出路径，不能覆盖现有证据。
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_floor_neighbors_v1.py cpu g2f1
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_floor_neighbors_v1.py native g2f1
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_floor_neighbors_v1.py strict g2f1
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_floor_neighbors_screen_v1.py \
  run g2f1 NEW_G2F1_FORWARD_LABEL
```

### MMA2/first1正向双侧改善，继续反向确认（2026-10-23）

`reboot20261023_mma2_floor_g2f1_forward_v1`完整结束，parent4022273退出。
前后control各try1存在ownership不合格轮，整组三轮排除、中值null；各try2为首个完整合格组。
candidate try1完整合格；不按耗时重试、不跨组拼轮，全部原reference与source/key/ELF核验通过。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前group2/first2 control try2 | 81.2027 /81.6138 /81.4409 | 81.4409 | 64.3692 |
| group2/first1 candidate try1 | 80.7494 /81.2008 /80.5892 | 80.7494 | 64.0697 |
| 后group2/first2 control try2 | 81.1754 /82.6026 /81.2447 | 81.2447 | 64.2215 |

candidate比两侧分别快0.6915 /0.4953 μs（0.849082% /0.609640%），进入反向确认。
283成员、1027852 B archive下载后独立复算原表格、ownership、source/原call keys/actual ELF与strict日志通过，
SHA为`b6dba8d19d62460cad984e329a228a546eb5c0a217ad23f2c9340a2113df0815`。
local acceptance为`passed=true/positive_at_every_adjacent_control=true`。
退出后恢复正式source `36178ad8...406b0`；80.7494 μs仍是待反向确认候选，未更新正式保留成绩。
下一步`reboot20261023_mma2_floor_g2f1_reverse_v1`按candidate/base/candidate三轮组执行，
只有四项相邻比较均改善后才追加E64 seed17、E65 seed0/17及scope外完整ELF核验。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_floor_neighbors_screen_v1.py \
  run g2f1 NEW_G2F1_REVERSE_LABEL candidate base candidate
```

### MMA2/first1正反向和追加验证通过，正式保留（2026-10-23）

`reboot20261023_mma2_floor_g2f1_reverse_v1`完整结束，host parent4145189退出。
前candidate try1的第二轮ownership不合格，整组三轮排除、中值null；try2是首个完整合格组。
中间control和后candidate均try1合格。保留全部原始记录，不按耗时重试或拼接轮次。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前group2/first1 candidate try2 | 81.7149 /81.4239 /81.2598 | 81.4239 | 64.2427 |
| group2/first2 control try1 | 82.0249 /82.4656 /81.6583 | 82.0249 | 64.1224 |
| 后group2/first1 candidate try1 | 81.2490 /81.0467 /81.0043 | 81.0467 | 64.0381 |

两侧相对中间control改善0.6010 /0.9782 μs（0.732704% /1.192565%）；加上正向的
0.849082% /0.609640%，四项相邻对照均改善。247成员、908127 B反向archive下载验收通过，
SHA=`b124814363360f485a829d0284876fb80242ff212ed7b10afa1137e2a07074c1`。
local acceptance SHA=`52f40e2a8263ab3f31daf758569de06ccd0b990bf83070fc765883f5527a377f`，回传一致。

随后追加`mma2_floor_g2f1_followup_v1`：E64 seed17、E65 seed0/17分别37.4641 /30.9444 /
12.7020秒完成，每项27个BYTE_CHECK、same-pointer dynamic graph及完整MoE22020096 B均零差异。
8项candidate自身scope外完整ELF与原control逐字节相同；GEMM2保持`4bd626e3...16be`。
E65仍原t64/w1x4 fallback，这些验证不证明E64/E65共用GEMM1或固定shared expert route已完成。
100成员、603000 B followup archive逐大小/SHA、strict日志及实际ELF本地验收通过，
SHA=`de908c35b4881bd099c014101a026d2d2dc8af4022c71de2edb01b26525654ef`。
followup local acceptance SHA=`1d5034b8d86d6cd9909fecc241261a858e4cb762a9fa6f243ac3202c93b01bab`。

全部通过后才将` sched_first_group`默认2→1这一行保留，本地和a07-3 source SHA为
`226861d45b681668b6f0da997959d1e512ca18ec60f83cd28729edd829d28f67`，UTF-8/LF。
GEMM1 ELF=`98908070e2d1f73c27842d84fd374b16c3f88d8f7af5eb148507771ad5cfd704`，
VGPR797 /SGPR58 /LDS327680 B /private0 /spills0，executable instructions6349。
采用记录`logs/mma2_floor_g2f1_production_adoption_v1.json`，
SHA=`432a5b899ceee9a3883c918c8a3abe5cb292848700624969d74a13f846369c67`。
formal成绩固定为正向三轮中值80.7494 μs，不能声称稳定低于80 μs；与80 μs相差0.7494 μs。
末尾GPU/KFD检查无残留，Git index、commit/push、原benchmark没有改动。

```bash
# 完整验证依赖已独立验收的正反向记录；已有label不覆盖、不重复启动。
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_floor_g2f1_followup_v1.py run
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_floor_g2f1_followup_v1.py seal
# 下载evidence.tgz/receipt后，在本地运行accept。
python .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_floor_g2f1_followup_v1.py accept
```

继续评估group1时，将基准更新为已保留MMA2/first1，first保持1，只改变group默认2→1。
原`mma2_floor_neighbors_g1f2_v1`仍未编译/计时，不在新正式版上直接运行绑定旧baseline的runner。

### 已保留first1上的group1准备与group3候选（2026-10-23）

`mma2_first1_group_g1f1_v1`基于正式`226861d4...28f67`，只把group默认2→1，
first保持1。3360项hint总量、2520项scope外/显式env顺序、40项guard检查通过，
其他runtime操作与完整nonhint AST保持。CPU parent55930、strict parent61715已退出。
实际GEMM1 ELF为`56e7b66e8572cd2b130365be8fe9afd0dde61fe82248533c6a935743749fce46`，
VGPR797 /SGPR58 /LDS327680 B /private0 /spills0，executable instructions6587，
相比正式版6349增加238条；静态数量不能直接预测性能。

E64 seed0 strict在36.1674秒完成27项BYTE_CHECK、same-pointer dynamic graph与
完整MoE22020096 B零差异，GEMM2完整ELF保持`4bd626e3...16be`。
39成员、400599 B preparation archive下载后逐大小/SHA及native machine words独立验收通过，
archive SHA=`7b78282e366b041d7174040d4950920fba13cd8dcf90b0ed9b2591948dc149b1`，
local acceptance SHA=`50aeba7f0c921ea2033ddf90840149282f5721c3b84ef8c9216f105a2f2a69fe`。
正向`reboot20261023_mma2_first1_group_g1f1_forward_v1`、parent70810已启动；
完整合格组三轮及后control结束前不采用，不把单轮或被ownership排除的组作为成绩。

在该序列计时期间仅本地冻结`mma2_first1_group3_g3f1_v1`，group默认2→3、first保持1，
同样只有+1/-1行。它保留当前DS floor与末尾closing24组合，区别于此前未保留的旧
MMA3/first1候选；旧候选的性能不能替代本组合的实测。新source SHA为
`11dce232298c53955767e5636d90c0109d3dcc28270efac5eab203e5bf8cf930`。
3360项hint总量、2520项scope外/显式env顺序和40项guard检查通过，原数学、访存、同步、
shape准入和其他device helpers保持。prepare/run/seal/accept展开后原函数、environment、
120秒child、计时与ownership门槛AST保持；当前仅本地准备，尚未编译或上GPU。

```bash
# a07-3主机；当前性能序列结束后才进行下一候选的编译和device验证。
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_group3_v1.py cpu g3f1
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_group3_v1.py native g3f1
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_group3_v1.py strict g3f1
# 原入口固定三轮组；仅当native、strict和preparation验收通过、baseline仍匹配时执行。
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_group3_screen_v1.py \
  run g3f1 NEW_GROUP3_FIRST1_FORWARD_LABEL
```

### group1/first1正向完整对照通过，继续反向（2026-10-23）

`reboot20261023_mma2_first1_group_g1f1_forward_v1`完整结束，parent70810退出，
正式source恢复`226861d4...28f67`。前control try1、后control try1/2/3各含
ownership不合格轮，均整组排除、中值null；采用各自首个完整合格组三轮，候选try1合格。
没有按耗时重试或跨组拼轮，所有原reference、source、原call keys和两阶段ELF验收通过。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前group2/first1 control try2 | 81.7952 /81.0846 /80.5701 | 81.0846 | 64.2721 |
| group1/first1 candidate try1 | 80.7810 /81.0216 /81.8099 | 81.0216 | 64.0886 |
| 后group2/first1 control try4 | 81.3967 /81.4009 /82.4108 | 81.4009 | 64.0084 |

相对两侧改善0.0630 /0.3793 μs（0.077697% /0.465965%）；前侧改善较小，
仍按事前门槛进入反向，当前不采用。358成员、1268963 B archive已下载并逐成员SHA、
原表格、ownership、source、原call keys、ELF与strict日志独立复算通过，
SHA=`9e7a509cfe1f2fce540927ff085cce4620c280e6e627a5b708119cf9a6404a6d`。
local acceptance为`passed=true/positive_at_every_adjacent_control=true`，只说明正向合格。
正式保留中值仍80.7494 μs，反向确认仍81.4239 /81.0467 μs；不从新正向更新成绩。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_group_screen_v1.py \
  run g1f1 NEW_GROUP1_FIRST1_REVERSE_LABEL candidate base candidate
```

### group1/first1反向未复现收益，不保留（2026-10-23）

`reboot20261023_mma2_first1_group_g1f1_reverse_v1`完整结束，host parent194159退出。
前candidate try1、后candidate try1含ownership不合格轮，均整组排除；下表采用各自首个
完整合格组三轮，中间control try1合格。没有按耗时重试或跨组拼轮。

| 反向组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前group1/first1 candidate try2 | 82.4446 /81.1986 /81.4727 | 81.4727 | 64.2361 |
| group2/first1 control try1 | 81.0065 /82.5691 /81.7108 | 81.7108 | 64.0654 |
| 后group1/first1 candidate try2 | 82.4634 /81.8288 /82.8472 | 82.4634 | 64.3016 |

两侧candidate相对control分别改善0.291394%、回退0.921053%；正反向四项相邻比较
未全部改善，故不采用group1，不执行为采用准备的followup或adoption，也不重测救援。
这说明本次实测不能支持稳定收益，不代表静态instruction数量本身证明了回退原因。
286成员、1032265 B archive下载后逐成员大小/SHA、原表格、ownership、source、原call keys、
actual ELF与strict记录本地独立验收通过，archive SHA为
`3fbcba54a4655c9b966e0d786f91c9696573a880d5be2e7693bb06deb5063ff2`。
local acceptance为`passed=true/positive_at_every_adjacent_control=false`，SHA为
`b4bbf84844f43cf44e1ef72088dad75d9704ab021a9ce269cae0bd50ba41ee2c`，回传核验一致。

正式source已恢复`226861d4...28f67`；保留成绩仍为正向80.7494 μs、反向81.4239 /81.0467 μs。
接着评估已冻结的group3/first1候选，保持A preshuffle=1、VGPR pin=0、现有功能和几何配置。

### group3/first1准备验收通过，原入口对照运行中（2026-10-23）

`mma2_first1_group3_g3f1_v1`的CPU parent297011、strict parent304837已结束。
native machine words、原DS/TDM/WMMA计数和offset、protected fences核验通过。
GEMM1 ELF为`9e84350b67a00b0b76d8e5435faa78fc3de1efd3b92c787c63dad2e80a8853d4`，
VGPR797 /SGPR58 /LDS327680 B /private0 /spills0，executable instructions6215，
较正式版6349少134条；减少静态instruction数不等同于已获得性能收益。

E64 seed0 strict在37.0848秒完成27项BYTE_CHECK、same-pointer dynamic graph与
完整MoE22020096 B零差异，GEMM2完整ELF保持`4bd626e3...16be`。
40成员、395573 B preparation archive下载后逐大小/SHA、实际ISA machine words、strict日志
本地独立验收通过，archive SHA为
`e792e251336d5b40b345d05aa2a6246bbc776908a760e2c7f64dd52d16063c0d`。
local acceptance SHA为`4dc8cc511c1349090503fa925ad324caf0d91e6f34993ced7754d1b8c7671488`，
回传核验一致；尚未追加E64其他seed、E65及scope外完整ELF，不提前采用。

原入口正向`reboot20261023_mma2_first1_group_g3f1_forward_v1`，host parent326977，
按control/candidate/control各完整三轮运行。首control第一轮ownership不合格，整组不用于
成绩；随后其他任务占用GPU，idle helper每6秒等待全机空闲，未竞争GPU、未驱逐任务。
全部组结束前不报告候选中值收益，不改正式保留source或已有成绩。

计时期间仅在本地准备`mma2_first1_straight_n_v1`，+1/-1行，将已有scope内的独立
accumulator N发射次序改为正序，scope外保留原蛇形次序。source SHA为
`d928553b874ac5acd99eb32f7086deafc9dcab4c6d844bd48eb457332939ac9a`。
64项实际`mma_rows`组合、900项逐accumulator K累加顺序、32项scope外原顺序检查通过；
3360项hint列表与总量、2520项scope外/显式env列表一致。每个accumulator的有效输入、
scale selectors、初始化、K更新顺序和最终符号值保持，原访存、同步、epilogue、准入不变。
完整nonhint AST不同是独立运算重排的实际变化，记录明确为false；不冒充纯hint修改。
该候选及其工具尚未上传、编译、上GPU或计时；若group3被采用，需要重冻其baseline后再测试。


### group3/first1完整正向无收益，不保留（2026-10-23）

`reboot20261023_mma2_first1_group_g3f1_forward_v1`完整结束，host parent326977退出。前control、candidate、
后control的try1各含ownership不合格轮，均整组三轮排除、中值null；下表采用各自首个完整
合格的try2，没有按耗时重试或跨组拼轮。

| 正向组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前group2/first1 control try2 | 81.8901 /81.3249 /81.9176 | 81.8901 | 64.4027 |
| group3/first1 candidate try2 | 82.1008 /82.2443 /81.2532 | 82.1008 | 64.0675 |
| 后group2/first1 control try2 | 81.5885 /81.2723 /81.8541 | 81.5885 | 64.6852 |

候选相对两侧回退0.257296% /0.627907%，不进入反向、不执行采用followup、不保留group3。
虽静态executable instructions少134条，实测仍无收益，不能以instruction数量代替性能证据。
323成员、1147521 B archive下载后逐成员大小/SHA、原表格、ownership、source、原call keys、
actual ELF与E64 seed0 strict独立验收通过，archive SHA为
`3c75fa87d72a8caf95d3e8be9cf421c12654ceb36865950765ed4dfce26391f8`。
local acceptance为`passed=true/positive_at_every_adjacent_control=false`，SHA为
`f42cb76236403ba60733d8d8103296b6848165de3e49525ef87f09fc41b4139d`。正式source已恢复`226861d4...28f67`，保留成绩
仍为正向80.7494 μs、反向81.4239 /81.0467 μs，尚未稳定低于80 μs。

下一项`mma2_first1_straight_n_v1`基于该正式source，仅将已有scope内独立accumulator
的N发射次序改为正序。按已冻结的11文件manifest上传并核SHA后，依次执行有120秒
限时的CPU、native、strict child；native、strict和preparation本地验收通过后才进行计时。
这时尚无该候选的硬件精度或性能结果，不能声称取得收益。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_straight_n_v1.py cpu straightn
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_straight_n_v1.py native straightn
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_straight_n_v1.py strict straightn
```


### N正序完成准备，M-group24/48仅本地静态检查（2026-10-23）

`mma2_first1_straight_n_v1` CPU parent469994、strict parent477420均已结束。
native通过，实际GEMM1 ELF为
`0e1ab903bd35e6c189fd8d8bf89c488dfe10c0a921a70296c1b150f51552d2df`，
VGPR797 /SGPR58 /LDS327680 B，private/spills0，executable instructions6408。
protected访存计数/offset、ordered fences及WMMA/BF16/native FP4/tanh数量保持；
该有限检查不等同于完整硬件hazard证明。E64 seed0 strict在36.6695秒完成27项BYTE_CHECK、
same-pointer dynamic graph与完整MoE22020096 B零差异，GEMM2仍为原完整ELF。

39成员、401691 B preparation archive已下载，逐成员大小/SHA与ISA machine words独立验收通过，
archive SHA=`2a45c33d2f7e910fc9811ef2f05ead3e71be8366e981a1da6f7bfd6e66044610`。
local acceptance SHA=`1b3bc24b2081d056bd130481951e1f4321e62e86c5340ea67033f8ece9b726aa`，
回传核验一致。原入口正向label=`reboot20261023_mma2_first1_straight_n_forward_v1`、
parent491144已经启动，当前未收齐完整合格组三轮，不能报告候选性能。
前control try1含ownership不合格轮，整组中值null；外部GPU/KFD任务出现时每6秒等待，
不争用计时、不驱逐任务。正式保留source和成绩不因本项尚未完成的实验更新。

等待时只在本地冻结`mma2_first1_mgroup24_v1`和`mma2_first1_mgroup48_v1`，
各仅改已有geometry guard内的`TILES_PER_GROUP=32`一行为24或48。
每项实际mapping AST检查223000个task/186个M/N组合，55750项1x4 peer检查、
78120项scope外精确mapping检查及154项原guard检查通过；所有capacity M/N tile各覆盖一次。
3360项hint总量及2520项scope外/显式env列表保持；原数学、访存、同步、epilogue、
shape支持不改。完整nonhint AST不同如实记录，不能把task重排描述成纯hint修改。
source SHA分别为
`4c7357855606d4fcead7d52b2fb939b5eb749b3ae2ff1944827873aecf722a62`、
`419792312b3eb841dc65e612541229247ed3fff6ff091ee056da955b3a8c3eb3`。
新的prepare/run/seal/accept工具展开后原timing/ownership/child函数AST保持，仍120秒child。
这些是本地候选准备，尚未上传、编译或上GPU，不能根据映射或resource预测收益。
若N正序被采用，需要重新冻结M-group候选baseline，不能混用当前绑定。

```bash
# a07-3主机；label已启动，重放必须用新的唯一label。
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_straight_n_screen_v1.py \
  run straightn NEW_STRAIGHT_N_FORWARD_LABEL
```

### M-group24/48完成准备，恢复N正序对照（2026-10-23）

12项任务文件及manifest上传后逐SHA一致。为利用外部GPU占用等待完成准备，
03:13:58 UTC确认N正序parent491144仅在idle child523498等待、没有timing child后暂停parent；
idle helper继续每6秒检查。暂停记录为`mma2_first1_straight_n_mgroup_cpu_pause_v1.json`。

两项各仅改已有scope内一行`TILES_PER_GROUP`，未增加persistent、wave布局或pipeline。
CPU/native与E64 seed0 strict均通过，27项BYTE_CHECK、same-pointer dynamic graph、
完整MoE22020096 B均与正式版逐字节一致，GEMM2完整ELF保持`4bd626e3...16be`。
VGPR797 /SGPR58 /LDS327680 B，private/spills0，static instructions均6349。
这些验证不构成性能收益，也不替代尚未完成的E65同kernel要求。

| 候选 | CPU parent | strict parent /秒 | GEMM1完整ELF SHA256 |
|---|---:|---|---|
| M-group24 | 605991 | 653544 /36.6557 | `d3c5ba57689dec9e453500fd05ec3eebb8d63ba1e5d6dc89da3ce2d28c8dcdfa` |
| M-group48 | 695409 | 708532 /36.3511 | `d45dd7874d83424d9b4c78aee512ee488ff1ad711ee9fe6c04f4f3dbe8c3e91d` |

各37成员preparation archive下载后逐大小/SHA及实际ISA machine words、strict原日志独立验收通过，
local acceptance均`passed=true/production_adopted=false`，已回传并核SHA一致。

| 候选 | archive bytes /SHA256 | local acceptance SHA256 |
|---|---|---|
| M-group24 | 400226 /`3e5725a682f42865e6f698eb68b8804da41beff9dabc00ecaac41f3f0c253b77` | `8e473591241c41dcaa39d74ec6f9a700d1dc714eca44c4fc79c79094f9723190` |
| M-group48 | 400252 /`e3ed496a669a8454912a5e487af23cd3f06345b464b23a0b96841a098e4b526b` | `fb0585ba5fc09e8978ccb147b5b5db2f06a706cc0a4355d76802137c68e1ebc8` |

03:23:26 UTC确认准备child全部退出、传输完成、生产source仍`226861d4...28f67`，
恢复parent491144；恢复记录为`mma2_first1_straight_n_mgroup_cpu_resume_v1.json`。
继续原N正序对照，不重新启动、不在计时期间并发准备M-group，也不跨组拼轮。
N正序结束后才按m24、m48的固定顺序计时；若N正序被采用，两项需要重新冻结baseline。
正式保留成绩仍为正向80.7494 μs、反向81.4239 /81.0467 μs，尚未稳定低于80 μs。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_mgroup_neighbors_v1.py cpu m24
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_mgroup_neighbors_v1.py native m24
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_mgroup_neighbors_v1.py strict m24
# m48使用相同入口；已有label不可覆盖。
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_mgroup_neighbors_screen_v1.py \
  run m24 NEW_M24_FORWARD_LABEL
```

### first1上的expert二分末轮精简完成准备（2026-10-23）

N正序control try2第三轮ownership不合格，三轮81.4029 /81.5317 /81.0488 μs整组排除，
中值null；try1同样已排除，未跨组拼轮。等待control try3全机空闲时，基于当前first1
重新冻结历史未计时的二分搜索精简，候选`mma2_first1_bisect_v1`，source SHA为
`535a0cb31fcf39d2f6a9eb6a6ab8763fbcc96e3c2c5b449e9b3e3e1c74901277`。

仅改两行：`has_work`已排除E这个sentinel结果，因此已有scope内初始hi由E改为E-1，
固定轮数由`ceil(log2(E))+1`改为`ceil(log2(E))`；E64七轮降为六轮，scope外行为保持。
E1～E2048、零计数重复psum、全部边界和random位置185844项实际搜索AST检查与原搜索及
`bisect_right`一致，92922项scope外保持。3360项hint列表、2520项scope外/override列表一致。
原数学、访存、同步、activation、quant、shape支持保持；完整nonhint AST因搜索变化不同，如实记录false。

16项任务文件及manifest上传后逐SHA通过。03:48:02 UTC确认N正序parent491144已暂停、
唯一child915021为6秒idle helper，没有timing child。首次紧接SIGSTOP的状态断言早于信号
送达而失败；随后重新读`/proc`确认T状态与idle child，保存暂停记录，未重复发起测试。
记录为`mma2_first1_straight_n_bisect_prepare_pause_v1.json`。

CPU parent967131、strict parent971740已退出。native gate原样通过：protected操作/offset、
ordered硬件fences及WMMA/BF16/native FP4/tanh counts保持。GEMM1完整ELF为
`61ab79052c31485f8b6ce96d29d51514628fea151963722fb12a1e6cabe8333f`，
VGPR797 /SGPR58 /LDS327680 B，private/spills0，static instructions6349。
静态instructions总量未降低，不能只凭少一轮source probe宣称减时。

E64 seed0 strict36.9389秒完成27项BYTE_CHECK、same-pointer dynamic graph、完整MoE22020096 B零差异；
GEMM2完整ELF保持`4bd626e3...16be`。43成员、401962 B preparation archive下载后逐大小/SHA、
实际ISA machine words和strict原日志本地验收通过，archive SHA为
`19571e1bc4335f71ec4df0fd1c807cd68dd6a9006f040a1dd73c1b7b1631d22d`。
local acceptance为`passed=true/production_adopted=false`，SHA为
`c5b3ba693f861a483d8bac57ed502f97c909018af81ee7567d075e2e37d1d477`，回传核验一致。

03:52:16 UTC确认准备child退出、传输结束、正式source匹配后恢复N正序parent491144。
恢复记录为`mma2_first1_straight_n_bisect_prepare_resume_v1.json`。
二分候选当前尚未计时或采用；先收完N正序，再按M-group24、48、bisect顺序测。
若前序候选通过四项相邻对照并采用，后续需重新冻结baseline。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_bisect_v1.py cpu bisect
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_bisect_v1.py native bisect
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_bisect_v1.py strict bisect
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bisect_screen_v1.py \
  run bisect NEW_BISECT_FORWARD_LABEL
```


### 完整M-group常数除法候选准备（2026-10-23）

N正序control try3三轮81.6224 /82.0229 /81.1416 μs均ownership不合格，整组中值null；
try1/2/3均排除，没有跨组拼轮，当前尚无完整合格control或candidate性能组。
a07-3外部用户反复启动GPU任务，idle helper继续每6秒检查；正式source和保留成绩不变。

为在等待期间准备下一项，04:12:25 UTC确认N正序parent491144唯一child1137667为idle helper，
并确认SIGSTOP后的T状态，保存`mma2_first1_straight_n_fullgroup_prepare_pause_v1.json`。
未在timing child运行期间编译或传输，也未驱逐外部任务。

`mma2_first1_fullgroup_div_v1`仅在已有GEMM1 geometry guard内区分完整M-group与尾组：
完整组用常数group_m_units除法，尾组保持原group_tiles除法；每个block的M/N task和cluster
mapping保持，原数学、transfer、fences、activation、quant及shape准入保持。
221300项actual AST task mapping、55325项N-cluster peer、78120项scope外和20000项
大整数映射检查通过，其中完整组185600项、尾组35700项，各capacity tile仅一次。
原hint3360项、scope外/显式env2520项一致。

v1首次compile child在15.385978秒rc1退出，报错：
`UnboundLocalError: cannot access local variable 'n_unit' where it is not associated with a value`。
这是新候选runtime分支SSA初始化写法问题，未进行GPU dispatch，不作为生产kernel bug或性能结果。
v1 source、原log、exit及工具均保留，不覆盖失败记录。

v2在分支前初始化`n_unit = in_group * 0`，保持各分支最终结果与v1及原版一致，修改+9/-2行。
新的source SHA=`d11a8879139bda82dc809de9091a65fdb5e6e7b91d9bac0317836da4ab0ee3c8`。
相同全部mapping和hint检查重新通过。14项任务文件上传后逐大小/SHA一致，manifest SHA为
`544c64fd92e08243e918d1bd6698e02198a2e23b17083f89a0f8112d26d6dc17`。
未改production source、benchmark计时、precision gate或compiled cache key逻辑。

CPU parent1197300、native parent1213135完成，GEMM1完整ELF为
`0c2a421515ed963e91431d201b3b9b8a0084388bcb697aa860d1b5bf72cdb076`。
原native gate通过：protected操作/offset、ordered硬件fences及WMMA/BF16/native FP4/tanh counts保持。
VGPR797 /SGPR58 /LDS327680 B，private/spills0，static executable instructions6349→6348。
该变化不能作为μs收益证据。strict parent1216564开始E64 seed0的完整MoE逐字节验证，
此时尚未完成strict或性能计时，不能声称该新候选已精度通过或更快。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_fullgroup_div_v2.py cpu fullgroup
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_fullgroup_div_v2.py native fullgroup
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_fullgroup_div_v2.py strict fullgroup
```


### 完整M-group除法v2精度通过（2026-10-23）

strict parent1216564已退出，E64 seed0在36.677981秒完成27项BYTE_CHECK、
same-pointer dynamic graph与完整MoE22020096 B逐字节零差异，GEMM2完整ELF仍`4bd626e3...16be`。
41成员、402556 B preparation archive已下载并逐成员大小/SHA核验通过，SHA为
`f7307f7d1786a12ee0436ed42bc9928718848df14581c63122456b635e0edd46`。
进一步本地从ISA machine words、native受保护操作/offset/fences与原strict日志独立验收通过；
这些是E64 seed0准备验证，不替代采用前其他seed、scope以及尚未完成的E65同kernel要求。
本项尚未进行性能计时或采用，正式source继续`226861d4...28f67`。


### 完整M-group准备结束；固定bit搜索静态候选（2026-10-23）

04:19:06 UTC完成fullgroup v2准备验收及传输后恢复原N正序parent491144；
`mma2_first1_straight_n_fullgroup_prepare_resume_v1.json`已下载，本地核对其source和恢复记录。
04:30 UTC N正序仍仅有三个ownership不合格control组，等待try4/r1，没有candidate性能中值。
保留版80.7494 /81.4239 /81.0467 μs成绩不变；不从排除轮计算中值。

当前仅本地冻结`mma2_first1_bitsearch_v1`：在原standard_a_timeout geometry guard内，
用固定bit步长upper_bound替换lo/hi双边二分，scope外仍执行原二分。
复用原has_work、psum、task坐标、全部有效计算/transfer/hardware fences和launcher ABI。
E64按6次probe找到expert，原版7次；不把source probe次数直接解释为性能改善。
229944项actual AST边界/重复psum/空expert检查、114972项scope外检查均与原版及
Python bisect_right一致；3360项actual hint和2520项scope外/显式env列表一致。
source SHA=`4428606ca4810c718eb22e32187173073a75820ef94f8ac732a20150ad51fab9`，
+17/-8行，没有新增shape准入限制、pipeline、persistent、wave布局或env开关。

04:33:38 UTC确认N正序唯一child为1137667 idle helper并暂停parent，原测试未并发计时。
七项任务文件上传后逐大小/SHA一致，manifest SHA=`1b2de273217fbab5b43804def412f86f00f58c43727ed44c30515110920889c3`。
compile/native/strict wrapper展开后原child函数AST保持；CPU-only parent1354095已开始，
尚未完成native/device精度或性能，不改生产源码。

核对FlyDSL dev882的ReplaceIfWithDispatch：runtime if生成scf.if区域内的Python辅助函数，
现有源码未提供kernel runtime early return的证明，因此不加入尚未证实的early return。
Shader Guide rev65 §3.2.2.1 p29明确EXEC==0不跳过WMMA/TDM，不用EXEC masking冒充跳过计算。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_bitsearch_v1.py cpu bitsearch
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_bitsearch_v1.py native bitsearch
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_bitsearch_v1.py strict bitsearch
```

### 固定bit搜索编译与完整MoE准备验证通过（2026-10-23）

CPU parent1354095、native parent1359257、strict parent1363000均已完成退出。
原native gate通过，protected counts/offset、ordered fences、WMMA/BF16/native FP4/tanh静态计数保持。
GEMM1完整ELF=`efd1cd0fe7bd8900e0fa0da9a0047038446b5e26708acdc1078e7136e0c0ddf4`，
VGPR797 /SGPR56 /LDS327680 B，private/spills0；相对正式版SGPR58→56、静态instructions6349→6321。
首input TDM前指令239→211，首WMMA前704→676；psum load/wait sites9→8。
线性静态处数不能证明动态关键路径或μs收益，mainloop寄存器分配也有变化，尚未计时。

E64 seed0 strict在36.857527秒完成27项BYTE_CHECK、same-pointer dynamic graph和
完整MoE22020096 B逐字节零差异，GEMM2完整ELF保持`4bd626e3...16be`。
43成员、402034 B preparation archive下载后逐成员大小/SHA核验，archive SHA为
`bf4a71b4967c3f5a8522220ad56fc3a069e61cdd97d9823ea1c124a7ca2b2f39`；
actual ISA machine words、原native和strict日志本地独立验收通过。
这仍仅是E64 seed0准备验证，不替代采用前其他seed/scope和E65同kernel要求。

十二项最终任务文件SHA核验一致，manifest SHA为
`2c71828ccda61dd66b668404d359ca8f8d7b4f3aac0e3447497518a37e80a6aa`。
bitsearch排在N正序、M24、M48、bisect、fullgroup之后；前序如采用新source，后续需重新冻结。
未改production source、benchmark计时、precision gate或compiled cache key。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_screen_v1.py \
  run bitsearch NEW_BITSEARCH_FORWARD_LABEL
```

### 固定bit搜索恢复；完整M-group除法v3准备通过（2026-10-23）

04:38:39 UTC完成bitsearch准备、独立验收与传输后恢复N正序parent491144。
04:48 UTC读取原日志确认control try4/r1耗时81.8796 μs、原precision通过，
但post记录lipiwang进程1424904占用GPU0、VRAM约10.5 GiB，ownership不合格；
随后settled恢复idle不能使这一轮成为有效性能数据。try1/2/3整组中值均null，
当前尚无完整合格control组，没有candidate性能中值，也未采用新生产优化。

04:49:29 UTC确认parent491144唯一child1428522为6秒idle helper，没有timing child，
暂停parent并保存`mma2_first1_straight_n_fullgroup_v3_prepare_pause_v1.json`。
原helper继续检查空闲，未驱逐其他用户任务；后续准备过程与原性能计时不重叠。

v2虽然源代码区分完整组/尾组，ISA仍在分支外计算尾组runtime除法。
v3在尾组中对group_tiles增加保持值的side-effect `s_mov_b32 $0, $1`，
用于阻止该除法提升到分支外；ISA §15.3 p236给出的S_MOV_B32语义为`D0.b32 = S0.b32`。
只在原standard_a_timeout guard内启用，原task映射、全部有效计算、shape准入、
transfer、fences和launcher ABI保持；scope外继续原除法。
修改+16/-2行，source SHA=`2979f4671d6cfa138594b7b50120a4f9602421d8bb466794513c25d82c3b8fdc`。
221300项actual AST映射、55325项N-peer、78120项scope外、20000项大整数映射通过，
其中完整组185600项、尾组35700项；3360项hint和2520项scope外/override保持。

CPU parent1455912、native parent1463952、strict parent1467520均已退出。
GEMM1完整ELF=`b1385bb0cbae50693d50cc858d15f39463b2d27c1846f73d135ce37580f902ee`，
VGPR797 /SGPR58 /LDS327680 B，private/spills0。
原native gate通过，protected操作/offset、ordered fences和WMMA/BF16/native FP4/tanh counts保持。
ISA确实包含绕过尾组reciprocal的分支；静态instructions6349→6399、
首TDM前线性指令239→284。该线性计数包含分支两侧，不是动态执行指令数或性能结论。
新的global psum load及地址寄存器分配也需由实际对照评估，不能只凭常数除法宣称更快。

E64 seed0 strict36.299619秒完成27项BYTE_CHECK、same-pointer dynamic graph与
完整MoE22020096 B逐字节零差异；GEMM2完整ELF保持`4bd626e3...16be`。
44成员、403808 B preparation archive下载后逐成员大小/SHA核验通过，
archive SHA=`a9ea2d0cf5f14c404fc4a49fc00c6a3ba8fcd91ffa1aad8478b429d1dff974bb`。
实际ISA machine words、native和strict原日志本地独立验收通过；未计时、未采用。
仍须完成采用前其他seed/scope验证，E65共用同compiled GEMM1的历史要求仍未完成。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_fullgroup_div_v3.py cpu fullgroup
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_fullgroup_div_v3.py native fullgroup
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_fullgroup_div_v3.py strict fullgroup
# 新label复现；不得覆盖现有记录或与其他性能测试并发。
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_fullgroup_div_screen_v3.py \
  run fullgroup NEW_FULLGROUP_V3_FORWARD_LABEL
```

### 完整M-group v3准备恢复；expert endpoint复用准备通过（2026-10-23）

04:56:07 UTC完成fullgroup v3归档、独立验收和传输后恢复原N正序parent491144。
随后control try4整组仍因外部GPU任务排除；05:07 UTC summary确认try1～4均不合格，
中值全部为null，目前停在base try5/r1，尚无完整合格control或candidate性能中值。
正式保留版80.7494 /81.4239 /81.0467 μs成绩不变，没有新采用的性能优化。

05:01:40 UTC确认唯一child1503826为idle helper、没有timing child后暂停parent491144，
准备独立的`mma2_first1_endpoint_reuse_v1`；该候选不叠加bitsearch或fullgroup。
在原standard_a_timeout scope内保存expert二分搜索最近的左分支endpoint，
查找结束后复用它计算mn_oob，省去最终一次重复table读取。原七轮查找及索引、
有效计算、transfer、fences、launcher ABI、shape准入与scope外代码保持。
三处修改合计+8/-3行，source SHA为
`71085da5785b8cb56017e3d7177532fcb7a7a60b4ff7c7113866fce3d9ddefec`。
231708项actual AST endpoint检查、115854项scope外读取序列检查通过；
scope内115854项逐项证明仅去掉最终一读，其他lookup index与extent保持。
3360项hint、2520项scope外/显式env列表一致，原计时和ownership检查函数AST保持。

CPU parent1523337、native parent1530291、strict parent1534432与seal parent1538682均已退出。
GEMM1完整ELF为
`d3c0df1a49428ecb76ef24b1a65d95f6169cd7654bd2cb220f1a4a257847bc08`。
VGPR797 /SGPR56 /LDS327680 B，private/spills0；正式版SGPR为58。
原native gate通过，protected操作/offset、ordered fences与WMMA/BF16/native FP4/tanh计数保持。
静态instructions6349→6353、首TDM前239→243、首WMMA前704→708、元数据load sites9→8；
这些静态变化不作为动态关键路径或μs收益结论。

E64 seed0 strict在36.436167秒内完成27项BYTE_CHECK、same-pointer dynamic graph与
完整MoE22020096 B逐字节零差异；GEMM2完整ELF保持`4bd626e3...16be`。
44成员、402511 B archive下载并逐成员大小/SHA核验通过，archive SHA为
`33421b6a2ed44ebb8d7520082ba9d38c66ff45b8fd91f15b5f6dae1c6d5af9ad`。
本地从actual ISA machine words、原native与strict日志独立验收通过；尚未计时或采用。
采用前仍须追加其他seed/scope验证，历史E65共用同compiled GEMM1的要求仍未完成。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_endpoint_reuse_v1.py cpu endpoint
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_endpoint_reuse_v1.py native endpoint
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_endpoint_reuse_v1.py strict endpoint
# 新label复现；不得覆盖现有记录或与其他性能测试并发。
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_endpoint_reuse_screen_v1.py \
  run endpoint NEW_ENDPOINT_FORWARD_LABEL
```

### N正序完整对照回退，保留蛇形顺序（2026-10-23）

05:09:04 UTC在重新确认全机idle、准备child全部退出和source匹配后，恢复原parent491144。
原idle child已成为待回收的zombie，恢复前重新完成idle检查；没有重复启动screen。
`mma2_first1_straight_n_endpoint_prepare_resume_v1.json`已下载并核对。
05:14 UTC原N正序screen完整结束，parent491144退出，正式source恢复`226861d4...28f67`，
结束GPU/KFD检查没有本任务残留进程。

前control try1～4均有ownership不合格，整组三轮排除，中值null；try5为首个完整合格组。
candidate和后control各try1完整合格；没有按耗时重试或跨组拼轮。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前MMA2/first1 control try5 | 81.9343 /81.8480 /80.7746 | 81.8480 | 64.2342 |
| N正序 candidate try1 | 82.6027 /82.0799 /81.0847 | 82.0799 | 64.8307 |
| 后MMA2/first1 control try1 | 81.5064 /81.4958 /81.9028 | 81.5064 | 64.3162 |

candidate相对两侧分别回退0.2319 /0.5735 μs（0.283330% /0.703626%），不进入反向、不采用。
358成员、1325710 B archive下载后，逐成员大小/SHA及原表格、ownership、source、
原call keys和实际ELF独立验收通过；archive SHA为
`e404e4fe99ddd71b6bd49199077284aa95150f675f7bfc9b2146a0fa398eced4`。
候选static instructions6349→6408、S_SET_VGPR_MSB sites1210→1261；
不能单凭这两个静态计数判定回退原因，保留原独立accumulator蛇形wn发射顺序。
正式保留成绩继续80.7494 /81.4239 /81.0467 μs，尚未稳定低于80 μs。
下一项按既定顺序评估M-group24，再评估M-group48。

### M-group24正向双侧改善，进入反向确认（2026-10-23）

05:15:30 UTC启动`reboot20261023_mma2_first1_mgroup24_forward_v1`，host parent1600012。
05:23 UTC完整结束，正式source恢复`226861d4...28f67`，完成后GPU/KFD检查无本任务残留。
前control try1的第二轮ownership不合格，整组三轮排除，中值null；try2为首个完整合格组。
candidate与后control各try1完整合格，没有按耗时重试或跨组拼轮。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前M-group32 control try2 | 81.6856 /80.9771 /82.6681 | 81.6856 | 64.6621 |
| M-group24 candidate try1 | 81.0782 /82.5247 /81.2765 | 81.2765 | 64.2384 |
| 后M-group32 control try1 | 81.9132 /82.3834 /82.9253 | 82.3834 | 64.7002 |

candidate比两侧分别快0.4091 /1.1069 μs（0.500823% /1.343596%），进入反向确认。
248成员、910325 B archive下载后，逐成员大小/SHA、原表格、ownership、source、
原call keys与实际ELF独立验收通过；archive SHA为
`1e09f2bde0f8a581f11da93f3b24e832f12d03748c39ea7573b59aca355c411c`。
local acceptance为`passed=true/positive_at_every_adjacent_control=true`，SHA为
`7062cd973ff6ba9853f2c946c3203b1a540959469b5785e910bbd769b83678b8`。
该候选尚未采用，正式成绩不更新。反向按candidate/base/candidate三个完整三轮组执行，
四项相邻对照均改善后才追加其他seed/scope验证。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_mgroup_neighbors_screen_v1.py \
  run m24 NEW_MGROUP24_REVERSE_LABEL candidate base candidate
```

### M-group24反向回退，不采用（2026-10-23）

`reboot20261023_mma2_first1_mgroup24_reverse_v1`已完整结束。
三组的try1分别因ownership检查不合格整组三轮排除，中值null；
每组try2是首个完整合格组，没有按耗时重试或跨组拼轮。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前M-group24 candidate try2 | 82.0966 /81.9889 /81.7993 | 81.9889 | 64.6389 |
| M-group32 control try2 | 81.2426 /81.3839 /82.1621 | 81.3839 | 64.5819 |
| 后M-group24 candidate try2 | 81.5083 /82.2802 /82.0692 | 82.0692 | 64.8835 |

两侧candidate分别比control慢0.6050 /0.6853 μs（0.743390% /0.842058%）。
正向的双侧改善未在反向重复，因此不采用M-group24，不执行其准备好的追加验证工具。
不能将正向81.2765 μs记为正式保留成绩。

320成员、1149649 B archive已下载，逐成员大小/SHA、原表格、ownership、source、
原call keys及实际ELF独立验收通过；archive SHA为
`381048a18c0d5ac1f7b0bf728e60269d29beb50a7ead187e0f0159d6fb854159`。
local acceptance为`passed=true/positive_at_every_adjacent_control=false`。
完成后正式source恢复`226861d4...28f67`，parent1657243已退出，主机GPU检查无残留。

正式保留M-group32、MMA2/first1，成绩继续80.7494 /81.4239 /81.0467 μs，
尚未稳定低于80 μs。下一项评估只改一行分组参数的M-group48，保持同一冻结base，
A preshuffle=1、VGPR pin=0、原warmup/iters和精度标准。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_mgroup_neighbors_screen_v1.py \
  run m48 NEW_MGROUP48_FORWARD_LABEL base candidate base
```

### first1上的closing邻值完成准备，尚未计时（2026-10-23）

M-group48正向队列`reboot20261023_mma2_first1_mgroup48_forward_v1`，parent1799659，
首control已完成两轮，第二轮ownership不合格，整组不计性能。其后持续等待外部任务释放GPU。
06:04:10 UTC确认唯一child1850495为6秒idle helper，没有timing child后，
临时SIGSTOP parent并核对T状态，保存`mma2_first1_mgroup48_closing_prepare_pause_v1.json`。
没有改变计时脚本或正在执行的timing child。

等待期间基于正式MMA2/first1 source`226861d4...28f67`，新冻结closing=23/25/22/26。
各只改`FENCE_COVER_MMA`的一行默认数字，保留原geometry guard、末尾K256/subtile0覆盖、
全部运行时数学、访存与硬件fences，不新增shape限制、pipeline、persistent或wave布局。
这次使用实际first1 hint AST：8400项hint总量检查、4200项scope外原列表检查通过，
并在current target保留MMA2/first1。旧MMA3/first2上的邻值回退不能代替本组合的实测。

16项任务文件及manifest上传逐SHA通过，编译和native checks串行完成。
四项均保持VGPR797 /SGPR58 /LDS327680 B /private0 /spills0，protected操作/offset、
ordered hardware fences、WMMA/BF16/native FP4/tanh counts一致。

| feature | GEMM1完整ELF SHA256 | executable static instructions | E64 seed0 strict耗时 s |
|---|---|---:|---:|
| c23 | `9b432f12e08a63b97fe802b3843432c46efb9379935ac12e844b1520e96b8556` | 6343 | 35.9147 |
| c25 | `f0511f0d197ff7a3e74805ad0ea394ddb8aaa47b1958110eb22d768d5ccdee15` | 6319 | 36.0875 |
| c22 | `34cad869a9b87ff22f1b54d4a16c696e0d9fb19637eb8646e14c4185ca53e425` | 6365 | 36.7399 |
| c26 | `e688551ae3ae1e4ac81ea6cf0dca2cfb874589fbbe0cfa1e4a2a7993b82edcb0` | 6329 | 36.1565 |

每项在120秒child限时内完成27项BYTE_CHECK、same-pointer dynamic graph及完整
MoE22020096 B零差异。GEMM2完整ELF保持`4bd626e3...16be`，这些准备耗时不作为性能数据。
四个37成员archive均下载逐大小/SHA、原strict日志及实际ISA machine words独立验收通过。
preparation local acceptance均为`passed=true/production_adopted=false`：

| feature | archive SHA256 | local acceptance SHA256 |
|---|---|---|
| c23 | `674e537595c5ee9e29ec30d783204167d0a07ebe551f67016ea553c4b4424cf4` | `81482fd75ce4576610a2313583fd51d778266071203e8cf92c635be60379acbe` |
| c25 | `dc4750270efecfb6aed9990f75f9e2a56c730cabb05e64afa093092023a51f18` | `8e430594ebe10fe807a17d6188272b935597f0b9e0c672479f927a73a87e29b8` |
| c22 | `494f3d550a10edc3fe597bad4e9ce28f2fb09fe3b28afdb2394d72541567f17a` | `71d7cbd71e7b5f4ff753ff4ccc08e33a844c4146177e68d4207be28313af7dbb` |
| c26 | `45fd0333875b90bb0ccfdbfebfc9033c4f360fca41519761e3d95eaf6e6da40c` | `b68ed7a432eef59526f97422777a643628406e0a4b89094adb3dbb3100e29959` |

尚未计时或采用；静态instruction数变化不当作性能收益。准备child退出、传输结束、
正式source核对后恢复M-group48队列；既定顺序仍先M48、bisect、fullgroup、bitsearch、endpoint，
再评估本组closing邻值。若前序采用新base，后序必须重新冻结，不能混用旧base。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_closing_neighbors_v1.py cpu c23
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_closing_neighbors_v1.py native c23
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_closing_neighbors_v1.py strict c23
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_closing_neighbors_screen_v1.py \
  run c23 NEW_FIRST1_CLOSING_FORWARD_LABEL base candidate base
```

### 当前候选入口ISA核查与fullgroup v4编译淘汰（2026-10-23）

M-group48仍在原正向队列中等待全机idle。前control try1的第二、三轮ownership不合格，
整组三轮排除，中值null；try2第一、二轮也不合格，不能从这些轮次更新正式成绩。
正式保留MMA2/first1、M-group32，成绩仍80.7494 /81.4239 /81.0467 μs。

等待期间独立检查10个实际ELF：解析ELF64的`.text`，将disassembly每项machine words
与对应section offset逐字节比较。结果写入
`mma2_first1_prepared_candidates_entry_isa_review_v1.json`，SHA为
`d699018a48adc3d43f69113e13f3cc63ffa41f50ef4adb9a9e8af084091df8ea`。
以下是可执行静态sites，不是动态cycles或整个GPU关键路径；不据此声称性能收益。

| candidate | executable static instructions | 首个TDM前static sites | 首个TDM前psum loads / waits |
|---|---:|---:|---|
| 正式MMA2/first1 | 6349 | 239 | 9 × `s_load_b32` /9 × `s_wait_kmcnt` |
| bisect | 6349 | 239 | 8 × `s_load_b32` /8 × `s_wait_kmcnt` |
| fullgroup v3 | 6399 | 284 | 9 × `global_load_b32` /9 × `s_wait_loadcnt` |
| bitsearch | 6321 | 211 | 8 × `s_load_b32` /8 × `s_wait_kmcnt` |
| endpoint | 6353 | 243 | 8 × `s_load_b32` /8 × `s_wait_kmcnt` |
| closing c23 /c25 /c22 /c26 | 6343 /6319 /6365 /6329 | 各239 | 各9 × `s_load_b32` /9 × `s_wait_kmcnt` |

fullgroup v3虽然将尾组runtime除法留在branch内，编译器同时把后续psum读取改成global load。
据此准备独立fullgroup v4：仅在v3的完整组/尾组branch之后增加
`n_unit = rocdl.readfirstlane(T.i32, n_unit)`，尝试恢复编译器对uniform task坐标的识别。
此值只由WG内一致的block ID和shape计算；ISA §15.14 p497–498说明
`V_READFIRSTLANE_B32`读取最低active lane并写入SGPR。
Shader Guide rev65的common-case latency notes同时说明其可能产生额外cycle，
因此即使恢复scalar load也仍需原入口实测，不能仅从指令性质推导收益。

v4 source SHA为`2619cb842004732045432455d98d16c595cdc43bf925fafe8990d329bcb998eb`，
相对正式source新增17行、删除2行；221300项实际AST任务映射、78120项scope外检查、
20000项大整数检查及3360项hint总量/2520项scope外原列表检查通过。
wrapper保留原120秒child、环境、计时和ownership函数AST。12项文件manifest上传逐SHA通过。

06:30:55 UTC确认M48 parent1799659的唯一child2070671是idle waiter后临时SIGSTOP parent，
没有timing child；暂停记录`mma2_first1_mgroup48_fullgroup4_prepare_pause_v1.json`已下载。
串行CPU编译和native checks均完成，资源仍VGPR797 /SGPR58 /LDS327680 B /private0 /spills0，
protected操作/offset、ordered hardware fences与WMMA/BF16/native FP4/tanh counts保持。

然而v4实际ELF `5f5ee944b67c22efbd033f0d03c81d29fa27a68a3016e961298f85bd6ef7c314`
仍有9次global psum loads和10次`v_readfirstlane_b32` sites，未恢复scalar psum load。
6399项executable instructions与v3仅一项差异：index147的等价`S_MUL_I32`源操作数互换。
没有达到此次编译意图，直接淘汰v4；未执行其strict或性能测试，未采用生产代码。
本地编译复核记录SHA为
`34db3c699a07405288d07a29c8390901670cabfc4768b6f206eb599cd88485be`。

06:33:27 UTC确认准备child全部退出、正式source恢复`226861d4...28f67`后SIGCONT恢复M48队列。
原fullgroup v3仍维持已完成的准备验证，不把v4的编译淘汰当作其性能结论。
后续顺序仍为M48、bisect、fullgroup v3、bitsearch、endpoint、closing邻值。
E65继续原fallback；本节不构成历史E64/E65同compiled GEMM1要求的完成。

### fullgroup v5同样未恢复scalar psum load（2026-10-23）

v4的实际`n_unit`仍在SGPR，新增readfirstlane没有改变后续global psum reads。
最后准备独立v5，将readfirstlane标记放到实际构造`blk_m`的`m_unit`之后；
相对正式source新增18行、删除2行，其余沿用v3的完整组/尾组除法及scope。
221300项映射、78120项scope外与20000项大整数检查、3360项hint及wrapper AST通过。
source SHA=`29ce2c3415f4f666fcef3b7568b306ca0fbeaf89ee775c96f10dbb1398375310`。

06:41:21 UTC仅在M48 parent1799659的唯一child2196487为idle waiter时暂停parent，
没有timing child。12项文件manifest上传核SHA，120秒限时CPU-only编译完成。
实际ELF为`46ab5980d2714b6dc6ad6a3a77d788ebdf88b900cce7719a50b586260dcc2218`，
VGPR797 /SGPR58 /LDS327680 B /private0 /spills0，executable instructions6399。
首个TDM前仍有9次`global_load_b32`、9次`s_wait_loadcnt`、10次`v_readfirstlane_b32`，
没有scalar psum loads。因此停止此uniformization探索，不继续v5 native、strict或性能测试。
未采用生产代码，也不将编译通过当性能收益或device正确性通过。

实际ELF及ISA已下载，本地逐machine words核验；复核记录SHA为
`3a8b2012e7e28a59fc0e30ed37589e56f0d17be2e9e7a32b2958a43cbde01693`。
06:43:01 UTC准备child退出、正式source核对后恢复M48原parent；pause/resume记录已下载。
M48前control try3的第二轮ownership不合格，该组仍将整体排除，不计算有效中值。
正式保留成绩和后续既定性能测试顺序不变。

### M-group48正向无双侧收益，保留M-group32（2026-10-23）

`reboot20261023_mma2_first1_mgroup48_forward_v1`已完成，parent1799659退出，
生产source恢复`226861d4...28f67`，未发现本任务测试进程残留。
前control try1/try2/try3均因ownership不合格整组三轮排除，中值为null；
try4是该位置首个完整合格组。没有拼接轮次，也没有根据耗时选择重试。

| 位置 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前M32 control | 80.7136 /81.2532 /81.2976 | 81.2532 | 64.7527 |
| M48 candidate | 81.4893 /80.7682 /81.6580 | 81.4893 | 63.8059 |
| 后M32 control | 81.1666 /82.0461 /81.8519 | 81.8519 | 64.0400 |

M48相对前control回退0.290573%，相对后control改善0.442995%，不满足正向双侧均改善的条件。
因此不安排反向或额外followup，不采用该单行32→48改动。GEMM2 ELF保持不变，
表中GEMM2耗时波动不能归为该改动的收益。

320成员archive已下载，先拒绝absolute/`..`/反斜杠/冒号/非regular路径，
再逐文件大小和SHA检查；已有同名文件只允许bytes一致，未覆盖不同文件。
本地原表格、ownership、source、原入口cache keys和实际ELF验收通过：

- archive SHA256：`af9a594b4f596d53813008f9ed0ffd20c386a65d428fb958d95763f5fd5fa29f`
- local acceptance SHA256：`05e5203bceb0cd64462b5fa4d8528acabf2e15e1f2403301dacec4d2d9234ea7`
- `positive_at_every_adjacent_control=false`，`not_production_adoption=true`。

继续已准备的bisect候选：仅调整expert lookup的两行边界/迭代数，
在现有`standard_a_timeout` guard内将E64由7次probe改为6次，保留scope外原代码。
原生ISA、重复psum边界检查和E64 seed0 strict完整MoE零字节差异已通过，
尚不能据此声称性能收益。使用新label运行原入口正向对照：

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bisect_screen_v1.py \
  run bisect reboot20261023_mma2_first1_bisect_forward_v1 base candidate base
```

正式保留版和80 μs目标状态不变。E65仍为原fallback，本节不构成E64/E65共用compiled GEMM1要求的完成。

### 两行bisect改动无性能收益（2026-10-23）

`reboot20261023_mma2_first1_bisect_forward_v1`完整结束，parent2273957退出，
三个位置均try1即完整合格，没有ownership排除组或按耗时重试。

| 位置 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前原lookup control | 81.3776 /80.9369 /81.5212 | 81.3776 | 64.3437 |
| 6-probe bisect candidate | 81.6223 /81.0192 /82.2463 | 81.6223 | 64.7823 |
| 后原lookup control | 81.9764 /81.1415 /81.5779 | 81.5779 | 64.5903 |

候选相对两侧分别回退0.300697% /0.054427%，不安排反向，不采用该两行修改。
原reference精度及candidate准备阶段完整MoE零字节差异通过；
这说明lookup结果保持，不说明减少一次probe一定改善整体性能。
GEMM2 ELF未改变，生产source恢复`226861d4...28f67`。

218成员、793351 B archive下载后安全路径、逐大小/SHA及原表格/ownership/source/keys/实际ELF本地验收通过，
archive SHA=`91fe51b9aba9ae4fc000f7d959adadc0251b2bc97b731b9986dec5ce7d534af3`。
`positive_at_every_adjacent_control=false`，`not_production_adoption=true`。

下一项沿用已经独立验收的fullgroup v3，不重跑编译淘汰的v4/v5，原入口命令为：

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_fullgroup_div_screen_v3.py \
  run fullgroup reboot20261023_mma2_first1_fullgroup_v3_forward_v1 base candidate base
```

其完整组使用常数除法，尾组保留runtime除法；实际psum reads变为global loads，
因此仍需完整对照，不能用静态常数除法替代性能结论。正式版尚未稳定低于80 μs。

### fullgroup v3正向回退，保留原任务映射（2026-10-23）

`reboot20261023_mma2_first1_fullgroup_v3_forward_v1`完整结束，parent2320104退出；
生产source恢复`226861d4...28f67`，未发现本任务测试进程残留。
前control try1、后control try1/try2因ownership不合格整组三轮排除，中值为null；
分别采用各位置首个完整合格的try2、try1、try3，不拼接轮次，也不按耗时重试。

| 位置 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前原映射control | 82.0226 /82.3516 /81.3900 | 82.0226 | 64.3353 |
| fullgroup v3 candidate | 82.0353 /82.1787 /82.8344 | 82.1787 | 64.6409 |
| 后原映射control | 82.0039 /81.5252 /81.2491 | 81.5252 | 64.3395 |

候选相对前后control分别回退0.190313% /0.801593%，不安排反向，不采用该映射修改。
完整组除法成为常数后，实际ISA仍新增global psum loads；这只是编译结果观察，
不能单独认定为回退的因果证据。已完成的原入口对照足以淘汰候选。
GEMM2 ELF仍为`4bd626e3...716be`；表中波动不属于该候选的GEMM2优化。

327成员archive下载后已检查路径、逐文件大小/SHA、原表格、ownership、source、
原调用cache keys与实际ELF，本地独立验收通过。一个既有0 B preparation seal log
先保留到独立名称，再补入archive内的完整远端日志，未覆盖不同证据。

- archive SHA256：`88ff7dd6cb700da550699798b0fbf7a7a34747c72759343322f69da59d5c0610`
- `positive_at_every_adjacent_control=false`，`not_production_adoption=true`。
- 候选准备阶段E64 seed0完整MoE零字节差异通过；不将精度通过当作性能收益。

继续已独立验收的bitsearch候选，原入口正向命令为：

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_screen_v1.py \
  run bitsearch reboot20261023_mma2_first1_bitsearch_forward_v1 base candidate base
```

正式保留版仍为MMA2/first1；80 μs目标尚未完成。E65仍走原fallback，
本节不构成历史E64/E65共用compiled GEMM1要求的完成。

### bitsearch正向小幅改善，继续反向确认（2026-10-23）

`reboot20261023_mma2_first1_bitsearch_forward_v1`完整结束，parent2427272退出，
生产source恢复`226861d4...28f67`，未发现本任务测试进程残留。
前control和candidate均try1完整合格；后control try1/try2因ownership不合格整组三轮排除，
try3是该位置首个完整合格组。排除组中值为null，不拼轮、不按耗时重试。

| 位置 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前原lookup control | 82.6109 /81.6053 /81.6558 | 81.6558 | 64.1962 |
| bitsearch candidate | 82.0397 /81.4810 /81.3016 | 81.4810 | 65.2251 |
| 后原lookup control | 81.4662 /81.6685 /81.5463 | 81.5463 | 64.6811 |

candidate分别改善0.214069% /0.080077%，正向满足两侧改善，但幅度小，尚未采用。
继续固定candidate/base/candidate反向三组三轮；四项相邻比较均改善后才追加精度及scope验证。
290成员archive已下载，安全路径、逐大小/SHA、原表格、ownership、source、原调用keys及实际ELF
独立验收通过，SHA=`bf4ef9ae1ee714ef0c4a5d2e27331c23beff3fa72bb203c0a08a31b96e2ab445`。
`positive_at_every_adjacent_control=true`、`not_production_adoption=true`。
GEMM2 ELF不变，不把表中波动认作该候选的GEMM2收益或回退。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_screen_v1.py \
  run bitsearch reboot20261023_mma2_first1_bitsearch_reverse_v1 candidate base candidate
```

### first1上的DS-first6/4/2/0本地静态准备（2026-10-23）

四项各新增2行、删除1行，仅在已有`standard_a_timeout`且未显式设置`AITER_FLYDSL_DS_FIRST_N`
时移动第一批future DS hints；保持原显式env、scope外逻辑、访存/WMMA总量、计算公式和硬件fence。
各16800项实际hint总量检查和12600项scope外/override原列表检查通过；完整source可逆，
原child环境、timeout120秒、计时及ownership函数AST保持。

目标geometry由原source推导为每K128共32条WMMA、40个DS hints和每wave `TDM_PER=1`。
prime、steady、drain及最后K256的7种上下文补充检查通过：6/4/2仅改变ksl0，
0同时改变带next-stage的ksl1，最后没有next-stage的ksl1保持。该检查仅说明hint总量及位置，
不构成完整硬件hazard证明或性能收益；后续仍须actual ISA/native及strict device验证。
上下文记录SHA=`433514b9158dda20b6242e932cd05ccf60e3be0973eec35ab15764674c20c735`。

目前只完成本地冻结，尚未编译、device验证或计时；性能队列仍先完成bitsearch反向，
再endpoint、closing23/25/22/26，最后按6/4/2/0顺序评估DS-first。
若前序采用新source，后续候选必须重新绑定新baseline；不能将旧准备结果直接套用。

### bitsearch反向与追加验证通过，正式采用（2026-10-23）

`reboot20261023_mma2_first1_bitsearch_reverse_v1`完整结束，三个位置的有效组如下。
前candidate try1/try2因ownership不合格整组排除，中值为null；try3为首个完整合格组。
其余两位置try1完整合格。没有拼轮或按耗时重试。

| 位置 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前bitsearch candidate | 81.6562 /81.1061 /81.1352 | 81.1352 | 64.8142 |
| 原lookup control | 81.3380 /81.7722 /81.6626 | 81.6626 | 64.4133 |
| 后bitsearch candidate | 80.9204 /81.3419 /81.7152 | 81.3419 | 64.6601 |

反向两项相邻改善为0.645828% /0.392713%；连同正向0.214069% /0.080077%，
四项比较均改善。收益幅度较小，不将本次81.4810 μs与更早80.7494 μs跨时段直接比较。
反向archive SHA256为`a442b583b8f4026ab6c13838b40c98313933ad3e42cb0f630c9053b2d49d8a6d`，
本地原表格、ownership、source、cache keys及actual ELF独立验收通过。

追加验证通过：E64/topk8/seed17、E65/topk9/seed0与17，每项27条BYTE_CHECK均为0，
完整MoE `(1536,7168)`、22020096 B逐字节差异为0。
八项scope外candidate实际ELF与既有control完全相同：GEMM2、FP8 input、original A、pin1、
t64 fallback、w1x4、buffers3、persistent1。这些检查未记录性能数据。
首启动缺失展开helper，在GPU child前退出；补齐helper并核SHA后以独立retry1日志启动，
原失败日志保留。后续child均保持120秒timeout；parent退出，未发现本测试残留GPU进程。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_followup_v1.py run
```

116成员followup archive在本地按安全路径、逐大小与SHA检查后独立验收，
SHA256=`60ea4f1cdde5bddfa5c87a4b74be119f62829531803a1e8241735a03a48c62dc`；
local acceptance SHA256=`08ee22c66c40e653b640eb05d7d06774ad2c88c4b4181033ed56459f61a49365`。
随后本地与a07-3采用相同candidate source，均保持LF：

- production source：`4428606ca4810c718eb22e32187173073a75820ef94f8ac732a20150ad51fab9`。
- GEMM1 ELF：`efd1cd0fe7bd8900e0fa0da9a0047038446b5e26708acdc1078e7136e0c0ddf4`。
- GEMM2 ELF：`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`，保持不变。
- VGPR797、SGPR56、LDS327680 B、private0、VGPR/SGPR spills0；相比前版仅SGPR58→56。
- source新增17行、删除8行，只改既有`standard_a_timeout`内expert lookup；保留计算、传输及fence。

本地测试文件保持原CRLF，其1429处CRLF统一为LF后与Git HEAD及远端测试文件逐字节一致。
本地采用工具明确核对原bytes SHA、换行数量及HEAD blob；未改测试代码、计时或Git index。
未commit/push。正式保留版本次正向中值81.4810 μs，反向确认81.1352 /81.3419 μs，
仍未稳定低于80 μs。E65仍使用原t64/w1x4 fallback；历史共用compiled GEMM1及固定shared expert
route要求未完成，追加精度通过不代表这些要求已完成。

下一项endpoint复用已基于新source重新冻结为`mma2_first1_bitsearch_endpoint_v1`，
新增7行、删除3行。231708项实际AST边界/重复psum/空expert检查及3360项hint检查通过；
只在既有scope减少最后一次重复psum读取，scope外115854项读取列表保持原样。
尚需重新编译、actual ISA/native及strict device验证，再决定是否计时。

### bitsearch上endpoint复用无双侧收益，不采用（2026-10-23）

`mma2_first1_bitsearch_endpoint_v1`从新保留source独立生成，不复用旧endpoint编译结果。
CPU/native与E64 seed0 strict通过；27项BYTE_CHECK和完整MoE22020096 B零差异，strict child36.0283秒。
actual GEMM1 ELF=`b32c5b2788471b51efa9bdbc1b884f9073eaaacf05597f2f24f08947bb0100f7`，
VGPR797 /SGPR56 /LDS327680 B、private/spills0保持，GEMM2 ELF保持。
实际scalar `s_load_b32`及`s_wait_kmcnt`各从8处降到7处；endpoint选择增加其他指令，
executable static instructions6321→6329。减少一次load/wait不能单独认定整体提速。
49成员preparation archive本地逐大小/SHA、source可逆、ISA machine words及完整MoE验收通过，
archive SHA=`9af3aea354966934056e4e92768ea1f94d494eb2e52f76121541cc574b77f99d`。

原入口正向`reboot20261023_mma2_first1_bitsearch_endpoint_forward_v1`完整结束，parent2692967退出。
前control try1/try2因ownership不合格整组三轮排除、中值null；try3为首个完整合格组。
candidate与后control均try1完整合格；不拼轮，不按耗时重试。

| 位置 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前bitsearch control | 81.3504 /81.2114 /80.8381 | 81.2114 | 64.2448 |
| endpoint candidate | 81.4622 /80.7832 /81.6795 | 81.4622 | 64.5462 |
| 后bitsearch control | 82.1409 /82.1494 /80.9582 | 82.1409 | 64.9066 |

candidate相对前control回退0.308824%，相对后control改善0.826263%，未同时改善两侧。
因此不安排反向、followup或采用，生产恢复`4428606c...1fab9`；结束未发现本测试GPU进程残留。
296成员archive本地安全路径、逐大小/SHA、原表格、ownership、source、cache keys及actual ELF复算通过，
archive SHA=`e6a9e47b262f56a9d13b2e3a54f4ac2cce9c5ebfdca805bfa509721322c7b8e9`，
`positive_at_every_adjacent_control=false`、`not_production_adoption=true`。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_endpoint_screen_v1.py \
  run endpoint reboot20261023_mma2_first1_bitsearch_endpoint_forward_v1 base candidate base
```

后续closing23/25/22/26已在本地基于bitsearch重新冻结，各仅+1/-1行，
8400项hint总量及4200项scope外列表检查通过，原bounded child/CLI/计时/ownership函数AST保持。
DS-first6/4/2/0也已在本地重新冻结，各+2/-1行；各16800项hint及12600项scope外/override检查通过，
target prime/steady/drain/last共7种上下文检查通过，记录SHA=
`1d1d22c15fbe4de0e9891bff3673db9b28d1d1e1b71a4cbdca0ed7a92d3270e9`。
这些新候选尚未重新编译、device或计时，不能据此报告收益；按closing、DS-first顺序继续。

### bitsearch上closing23完成编译和精度准备，等待原入口对照（2026-10-23）

`mma2_first1_bitsearch_closing_c23_v1`已从正式bitsearch source重新编译；仅将既有
`standard_a_timeout`内`FENCE_COVER_MMA`默认24改为23，新增1行、删除1行。
CPU/native检查通过：protected operation counts/offsets及ordered hardware fences保持，
实际WMMA、tanh、BF16与FP4 conversion数量保持；static instructions6321→6315。
VGPR797 /SGPR56 /LDS327680 B、private0及VGPR/SGPR spills0保持。

E64/topk8/seed0 strict child36.5713秒完成，27条BYTE_CHECK和完整MoE22020096 B逐字节差异均为0。
GEMM1 ELF=`44d1033a74e02f4675fe43d0a6eba3c374081422018fe343d7288683f1916b5a`；
GEMM2 ELF仍为`4bd626e3...716be`。上述运行不记录性能；native有限检查及seed0正确性
不构成完整SCHED_MODE2 hazard证明，也不代替追加scope/E65验证。
40成员preparation archive已下载、逐大小/SHA核验并本地独立验收，
SHA=`0dff8cbad441e675e01b72d5e6c7fbd7b3e37caf58ae67d0958613af62dee451`。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_bitsearch_closing_neighbors_v1.py cpu c23
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_bitsearch_closing_neighbors_v1.py native c23
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_bitsearch_closing_neighbors_v1.py strict c23
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_closing_neighbors_screen_v1.py \
  run c23 reboot20261023_mma2_first1_bitsearch_closing_c23_forward_v1 base candidate base
```

正向parent2778705已启动，四项原始调用cache keys准备完成；随后因外部GPU/KFD任务等待全机空闲，
自动每6秒检查，不与外部任务竞争。当前尚无性能结论，不采用该candidate。
正式保留版仍为bitsearch；后续c25/c22/c26及DS-first6/4/2/0按既定顺序继续。

### 空闲等待期间完成closing25/22/26准备（2026-10-23）

确认C23 parent的唯一直接child为`wait_normal_idle_v1.sh`后，暂停本任务parent，
核对T状态、child及正式source SHA；空闲检查child继续每6秒运行。
随后按c25→c22→c26顺序进行compile/native及不计时strict，未并发性能测试。
三项准备完成后恢复parent2778705；暂停/恢复记录为
`logs/mma2_first1_bitsearch_closing_idle_compile_pause_v1.json`。

| candidate | static instructions | GEMM1 ELF SHA256 | E64 seed0 strict秒数 |
|---|---:|---|---:|
| c25 | 6291 | `cb95ca7790a251519842dc80fe2fab909b374c1acbee73d607ee508af5fc1d01` | 36.9517 |
| c22 | 6337 | `75559961c4deb27b67eea859eb4fd602efb9ce6592304d62353c564fa01ac6e6` | 36.3037 |
| c26 | 6301 | `1d971a4ebcb65ffb9818e35bee1c758abb7c917d236e91012080b04d82b4e864` | 36.1773 |

三项均保持VGPR797 /SGPR56 /LDS327680 B、private/spills0及受保护的counts/offsets/fences；
每项27条BYTE_CHECK和完整MoE22020096 B零字节差异，GEMM2实际ELF保持。
各37成员preparation archive已下载、安全路径/大小/SHA核验，并由本地工具独立复算实际
machine words、source单行可逆差异及完整MoE证据，均`passed=true`、`performance_data=false`。

- c25 archive：`5a90aa694d2320c01fe10bf05442fbce55cc8f9fef3fc5f3f50ff32c7a4bacdc`。
- c22 archive：`42d8b556110d174278b16dd3a5ccacf0ae72a3acf655903f7869bfa3d86ab80f`。
- c26 archive：`db12757861e55d42518251a60ce0fc5695a397e1216058f8d177c4e780db2b86`。

上述指令数量变化不作为提速结论，三项尚未性能计时或采用。C23继续全机空闲门禁，
当前外部KFD owner包括其他容器的AOT进程；即便busy=0、VRAM很低，也仍须等待owner释放。

### bitsearch上DS-first6/4/2准备通过，DS-first0在native淘汰（2026-10-23）

确认C23仍处于external-owner idle gate后，暂挂parent、保持waiter每6秒运行，
依次完成DS-first6/4/2/0的CPU/native准备；0在原native gate拒绝，队列即时停止。
6/4/2随后串行完成不计时E64 seed0 strict，每项27条BYTE_CHECK和完整MoE22020096 B
逐字节差异均为0，实际GEMM2 ELF保持。所有child仍限120秒；完成准备后恢复C23 parent。

| first DS cap | static instructions | GEMM1 ELF SHA256 | strict秒数 | 准备结论 |
|---:|---:|---|---:|---|
| 6 | 6334 | `08ba74c9c948e67874e01f79fd786d063c43903b7320a3bdb25a09ed437bc93a` | 36.8111 | 通过，尚未计时 |
| 4 | 6328 | `26650bd51a67240ee5042a28d2f7d3155201193c4b359147b7734abccbc0554e` | 36.6903 | 通过，尚未计时 |
| 2 | 6335 | `5431b4d13cf7bae5f0d14935dd89aaac5d192cd6d47592eab92d18667bfd1030` | 36.1092 | 通过，尚未计时 |
| 0 | 6325 | `f6a67cc30edfcd56241dbe4ef334a8c9333ce2fa34c1c8dec5e0bdf58406c3c3` | 未运行 | native拒绝，不进入GPU |

四项资源均VGPR797 /SGPR56 /LDS327680 B、private/spills0。
0的拒绝原文为`ordered protected hardware fences differ`；不放宽检查或声称该变化是安全的。
其ELF/ISA/artifact已下载，本地逐条重核6423条含padding的ISA machine words与ELF一致；
未运行strict或性能，不采用。6/4/2各40成员preparation archive已下载、逐大小/SHA安全核验，
本地独立重算source可逆差异、machine words、fences/resources和完整MoE证据，均通过。

- DS-first6 archive：`69155163e81bc6363f7a88fbf8a9b652a03e27a783dafa4d268492e419705474`。
- DS-first4 archive：`0e2a5096773f21ef27e9e90f8999c8812991d748ab2eff74d7503558502aef0a`。
- DS-first2 archive：`c2a6e71c4f336c91b9d3717921c3fd7292329cb1691c0f4232c422f8957d0337`。

C23恢复后曾通过空闲门禁进入原入口；前control try1/try2在post ownership检查出现
外部实验或无法识别已退出KFD条目，整组三轮中值均为null，不计为性能数据。
当前继续原固定顺序，不按耗时排除有效轮，也不拼接不同try的轮次。

### 当前bitsearch上的非负task除法：新冻结小候选（2026-10-23）

较早DS版测过unsigned swizzle且没有双侧收益，不把旧结果当作当前bitsearch组合的结果。
新`mma2_first1_bitsearch_udiv_v1`只在既有`standard_a_timeout`内，将三处已证明非负的
swizzle除法换为`arith.divui`；新增8行、删除3行。保持runtime M/N、full/tail task mapping、
原geometry guard、expert lookup、mainloop/epilogue、数学和hardware fences，不新增shape限制。
source SHA=`16831acf03db01df619974da77ad385e24f6ccd16dc6f4325f42098944c61d12`。

实际mapping AST的221300项目标点、55325项N-cluster peer、78120项scope外点和20000项
大整数点通过，723900次正数除法输入检查通过；每个capacity M/N tile仍恰好一次，
peer共享相同M，185600项full-group及35700项tail-group结果相同。替换可精确反转。
compile/native/strict wrapper保持原child执行、timeout及检查，只重绑定独立cache路径和base。
等待门禁中暂停parent后仅上传5项所需文件，逐大小/SHA核验通过；开始CPU-only编译。
此时候选尚未取得native、device或性能结论；评估顺序排在closing及DS-first6/4/2后，
若期间采用新source须重新冻结，不能混用旧baseline证据。

### 非负task除法完成native与完整MoE准备验证（2026-10-23）

`mma2_first1_bitsearch_udiv_v1`已完成CPU编译、原native gate及E64/seed0 strict。
VGPR797 /SGPR56 /LDS327680 B、private0和VGPR/SGPR spills0保持；protected counts、offsets、
ordered hardware fences及实际数学指令数量保持。静态可执行指令6321→6294，
首TDM前211→184、首DS前556→529、首WMMA前676→649。
这是编译产物变化，尚未证明运行耗时改善。

strict child36.2723秒完成；27条BYTE_CHECK和完整MoE22020096 B差异均为0。
GEMM1 ELF SHA256=`44fe0b75ad08e9b4f56e1f257429a3c37b36afed574e8193c2e86782325f59a4`；
实际GEMM2 ELF仍为`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。
39成员preparation archive已经安全下载、逐大小/SHA验证并由本地独立工具验收：

- archive SHA256：`1f474284d0df899bbb1357e353d392c8c4e7eef04c2eb4c3d652fb6223de0460`。
- local acceptance SHA256：`620da361d6a69ae250111f785d7ff0f22db5b37ca1efa05edb0fa3da30dee092`。

以上是准备阶段，`performance_data=false`、`production_adopted=false`。
正向/反向screen工具已独立重绑定该候选的实际mapping证明字段，
本地AST检查确认原测试参数、timing、child timeout、ownership整组排除、cache与验收函数保持；
没有改变原测试文件或放宽精度门槛。复现命令如下，按事前顺序排在closing和DS-first后执行。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_bitsearch_udiv_v1.py cpu udiv
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_bitsearch_udiv_v1.py native udiv
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_bitsearch_udiv_v1.py strict udiv
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_udiv_screen_v1.py \
  run udiv reboot20261023_mma2_first1_bitsearch_udiv_forward_v1 base candidate base
```

### C23继续等待外部GPU任务，保持原采样规则（2026-10-23）

parent2778705仍在C23正向序列；前control try1和try2因ownership不合格整组排除，
中值为null。try3的r1/r2也出现外部owner或无法识别已退出KFD条目；
r3当前处于运行前idle gate，必须收完整组三轮后整体判定，不拼接不同try。
当前外部PID3014588仍持有KFD、VRAM约1.4 GB；即使瞬时busy=0也不视为空闲，
waiter每6秒检查，尚无新的有效control或C23 candidate性能结论。

正式source SHA256仍为`4428606ca4810c718eb22e32187173073a75820ef94f8ac732a20150ad51fab9`。
本轮新增内容仅为独立候选、工具和证据，未修改生产kernel。
正式保留版的同轮中值仍为正向81.4810 μs、反向81.1352/81.3419 μs；
稳定低于80 μs的目标尚未达成。E65仍走原t64/w1x4 fallback，
历史要求的E64/E65共用同一个compiled GEMM1及固定shared expert route仍未完成。

### C23完整正向未通过双侧改善，不采用（2026-10-23）

`reboot20261023_mma2_first1_bitsearch_closing_c23_forward_v1`完成；
parent2778705已退出，`completed=true`、`source_restored=true`、无error。
三个合格位置保持原测试入口、原精度门槛、A preshuffle1/pin0、warmup5/iters20。
前control try1/2/3及candidate try1因ownership不合格整组排除，中值均为null；
不从这些组抽取快样本，也不拼接try。

| 位置 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前bitsearch control | 81.4366 /81.2068 /81.7339 | 81.4366 | 64.3394 |
| closing23 candidate | 81.0635 /81.4852 /81.8161 | 81.4852 | 64.4449 |
| 后bitsearch control | 81.6074 /82.3053 /80.9180 | 81.6074 | 64.3247 |

candidate相对前control回退0.059678%，相对后control改善0.149741%；
没有同时优于两侧control，不追加反向、scope或生产采用。
6321→6315条指令及有限native/seed0正确性准备通过没有转换为可确认的性能收益。

360成员、1286116 B性能证据已下载，逐路径/大小/SHA核验，
本地独立工具从原table、ownership、source key和实际ELF重算通过。
archive SHA256=`bfa11da4e36884fc21cdf16a2ee1da8665d2ecd61c8ac9879b9e70774b38e99f`；
`positive_at_every_adjacent_control=false`，不把验收passed解释成性能改善。
退出后检查仅发现其他容器的外部GPU任务，无本序列启动的测试GPU/KFD进程残留。
生产源码精确恢复bitsearch SHA=`4428606ca4810c718eb22e32187173073a75820ef94f8ac732a20150ad51fab9`。

下一项仍按事前固定顺序测c25；其compile/native/seed0 strict准备已完成，不重复执行。
closing followup工具及采用工具已准备为独立路径，仍要求正向/反向四项相邻改善与追加
E64/E65完整MoE和8项scope检查后才能运行采用；本次C23不会进入该阶段。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_closing_neighbors_screen_v1.py \
  seal reboot20261023_mma2_first1_bitsearch_closing_c23_forward_v1
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_closing_neighbors_screen_v1.py \
  accept reboot20261023_mma2_first1_bitsearch_closing_c23_forward_v1
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_closing_neighbors_screen_v1.py \
  run c25 reboot20261023_mma2_first1_bitsearch_closing_c25_forward_v1 base candidate base
```

### C25正向通过双侧筛选，继续反向验证（2026-10-23）

`reboot20261023_mma2_first1_bitsearch_closing_c25_forward_v1`完成，parent3134139已退出；
`completed=true`、`source_restored=true`，无error。后control try1和try2因ownership不合格
整组三轮排除、中值为null；其余三组均完整合格，没有按耗时重试或拼接try。

| 位置 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前bitsearch control | 82.0312 /81.4366 /81.0402 | 81.4366 | 65.1491 |
| closing25 candidate | 81.1126 /82.1965 /81.2939 | 81.2939 | 64.9935 |
| 后bitsearch control | 80.8488 /82.1793 /81.8229 | 81.8229 | 63.9203 |

candidate分别改善0.175228%和0.646518%，通过正向筛选；尚未采用，也尚未达到80 μs。
284成员、1028373 B证据已下载，逐大小/SHA核验，并在本地从原table、ownership、source key
和实际ELF独立重算通过。archive SHA256为
`1fe2594f4c270a20939d939fb69eef41a151ddf0bf8a8d80bd66eea3e25a96fe`，
local acceptance SHA256为`0e39dfe7b0270c6a90c9d9647f5bac4ce792029f9d169893ff3b2d17d52f9cb6`。
原测试文件、计时参数、精度门槛保持，GEMM2 ELF不变；生产source已精确恢复正式bitsearch版。

下一步执行反向candidate/base/candidate，仍要求全部相邻比较改善，并追加完整MoE和scope检查，
通过后才采用。等待外部GPU任务退出期间保持每6秒空闲检查，不竞争GPU。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_closing_neighbors_screen_v1.py \
  seal reboot20261023_mma2_first1_bitsearch_closing_c25_forward_v1
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_closing_neighbors_screen_v1.py \
  accept reboot20261023_mma2_first1_bitsearch_closing_c25_forward_v1
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_closing_neighbors_screen_v1.py \
  run c25 reboot20261023_mma2_first1_bitsearch_closing_c25_reverse_v1 candidate base candidate
```

### 常数除法shift候选完成准备，尚未性能计时（2026-10-23）

`mma2_first1_bitsearch_pow2_v1`在既有`standard_a_timeout` geometry guard内增加6行，
以arithmetic shift/mask重算ceil(M/256)、ceil(N/256)、block/4和N tiles/4；
runtime M/N、full/tail group与psum查找仍保持，不改launcher ABI、有效计算、pipeline或epilogue。
scope外保持原操作。source SHA为
`75a0d13a7ba7c9d2bbeffe9aa9c8dd9fbca1c0aed3619cf8ed0c6e393810ed68`。

实际task mapping AST的221300项目标点、55325项cluster peers、78120项scope外点与20000项
大整数点逐项一致；每个capacity M/N tile恰好一次。额外60063项signed-i32边界及wrapped
addition检查通过，采用signed floor identity，不用unsigned conversion缩小输入域。
CDNA5 ISA §15.1 p212记载`S_ASHR_I32`保留sign bit；此处shift与floor division等价是数学推导，
不是由该文档推断性能。原timing、child120秒、ownership整组排除、cache和验收函数AST保持。

确认C25反向parent3233952只有idle waiter时，临时SIGSTOP parent，并精确保存/恢复其C25
source；串行完成CPU/native/strict准备后SIGCONT。pause/resume记录已下载，期间没有并发计时。
所有测试child均在120秒内退出；E64 seed0 strict36.4966秒完成27条BYTE_CHECK及完整MoE
22020096 B零差异。原native gate通过，protected counts/offsets、ordered hardware fences及数学
指令计数保持；GEMM2完整ELF保持。资源VGPR797 /SGPR56 /LDS327680 B /private0 /spills0。

可执行静态指令6321→6296，GEMM1 ELF为
`a337b91a7dfc52541b0de73dd3ffbb177ea8f2c4e2f51f4c88468bbc3b25dc07`。
41成员、406351 B准备archive已安全下载、逐大小/SHA验证并独立重算通过：

- archive SHA256：`8a1068df6ae53a2caa63bc3252f3c7912e47a8ddf53c2492c4e59faa6ac95fd6`。
- local acceptance SHA256：`d2672078129d0becf55ec3619ffc731d482c4dce182af5d5afbdca10833679e5`。

本候选尚无性能数据、不采用，排在现有closing、DS-first和udiv之后；若采用新base，须重新冻结。
C25反向仍未完成最后一个合格组，正式保留版和成绩不变。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_bitsearch_pow2_v1.py cpu pow2
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_bitsearch_pow2_v1.py native pow2
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_bitsearch_pow2_v1.py strict pow2
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_pow2_screen_v1.py \
  run pow2 NEW_UNIQUE_POW2_FORWARD_LABEL base candidate base
```

### C25反向和追加验证通过，保留一行closing改动（2026-10-23）

`reboot20261023_mma2_first1_bitsearch_closing_c25_reverse_v1`完整结束，parent3233952退出，
`completed=true/source_restored=true`，无error。中control try1/2以及后candidate try1～6
因ownership不合格整组三轮排除、中值null；后candidate try7为首个合格组。
不按耗时重试，不拼接不同try。正向数据仍为前control81.4366、candidate81.2939、
后control81.8229 μs，不以跨时段control重写旧保留成绩。

| 反向位置 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前C25 candidate | 81.7466 /81.2196 /82.2169 | 81.7466 | 64.8857 |
| 中bitsearch control | 81.2996 /82.3477 /81.8880 | 81.8880 | 64.8393 |
| 后C25 candidate | 81.8773 /81.5527 /81.7276 | 81.7276 | 64.4787 |

反向两侧分别改善0.172675%和0.195877%，连同正向0.175228%和0.646518%，
四项相邻比较均改善。差距很小，不解释成已达到80 μs或跨机器稳定收益。
500成员、1747216 B原性能archive安全下载并逐路径/大小/SHA核验，
从原table、ownership、original cache key和实际ELF独立重算通过。
archive SHA=`53a2350b2b691ea5936f97583f5cc96ac0930cb45b8744cf398d9102a837627b`；
local acceptance SHA=`113d5785895b7f18d4621b2579fb1c24671bb954e1f74a6382dcd6c4b9dc8787`。

追加`mma2_first1_bitsearch_closing_c25_followup_v1`完成：E64 seed17、E65 seed0/17
分别36.4784、30.8045、12.8024秒，所有child低于120秒，27条BYTE_CHECK及完整MoE
22020096 B均零差异。8项范围外直接编译的完整ELF逐字节等于各自control，
覆盖GEMM2、FP8 input、original A、pin1、t64 fallback、w1x4、buffers3、persistent1。
111成员、614216 B追加archive本地独立验收通过，archive SHA为
`0e4c5818235d9af6845b543b3891abda9ccf99798e6df0c0ed5c73beb8188c2b`，
followup acceptance SHA为`aa1b0d0ad12640916f08247a07b55846a2eee784a9ef3773ffba4bec1a460367`。

采用工具验证原source可逆单行差异、正反向四项改善、准备及追加验收后，
本地/远端生产源码均只将`FENCE_COVER_MMA`在原`standard_a_timeout` guard内的24改为25。
没有改benchmark、精度门限、有效数学、transfer/fence、shape支持、pipeline或wave布局。
不改Git index、不commit/push。结束检查本机gpu_users/KFD为空，无本任务GPU进程残留。

| 当前保留项 | 值 |
|---|---|
| source SHA256 | `6b372816af6fd3ebbb6f94d0fecf240a0aff477f608f45a7df91574d219672ba` |
| GEMM1 ELF SHA256 | `cb95ca7790a251519842dc80fe2fab909b374c1acbee73d607ee508af5fc1d01` |
| GEMM2 ELF SHA256 | `4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be` |
| VGPR /SGPR /LDS B /private /spills | 797 /56 /327680 /0 /0 |
| executable static instructions | 6291，旧版6321 |
| 原入口正向中值 μs | 81.2939 |
| 反向确认中值 μs | 81.7466 /81.7276 |

稳定低于80 μs的目标仍未达到。E65仍使用原t64/w1x4 fallback；本次完整MoE等价验证
不代表E64/E65共用同一个compiled GEMM1，也未证明固定shared expert1 route。
这两项历史要求仍未完成。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_closing_followup_v1.py run c25
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_closing_followup_v1.py seal c25
# 原追加archive下载并核验后，在本地独立accept。
python .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_closing_followup_v1.py accept c25
python .codex_tmp/yadai_gemm1_param_20261021/tools/adopt_mma2_first1_bitsearch_closing_v1.py inspect c25
python .codex_tmp/yadai_gemm1_param_20261021/tools/adopt_mma2_first1_bitsearch_closing_v1.py apply c25
```

### 后续closing候选重新绑定C25，尚无新性能结论（2026-10-23）

新`mma2_first1_bitsearch_c25_closing`工具和候选以已采用C25 source/ELF为base，
仅保留尚未计时的C22→C26顺序；之后仍为DS-first6→4→2→udiv→pow2。
C22/C26的源码字节与旧候选相同，但不复用旧base上的native/strict/performance结论，
须重新编译、核验和原入口相邻对照。所有计时、child120秒、ownership整组排除及验收函数
AST与原工具保持。17个必要文件已上传，逐个SHA一致，upload manifest SHA为
`4991679884c593e99199b50e035c9b59a839227f425539725c64da1c421bf25f`。
准备checker首次仍引用已移除C23 feature，在任何GPU运行前拒绝；修正该测试feature为C22
后检查通过，没有放宽gate或修改kernel。当前C22进入CPU-only编译，尚未device或计时。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_bitsearch_c25_closing_neighbors_v1.py cpu c22
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_bitsearch_c25_closing_neighbors_v1.py native c22
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_bitsearch_c25_closing_neighbors_v1.py strict c22
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_c25_closing_neighbors_screen_v1.py \
  run c22 NEW_UNIQUE_C25_BASE_C22_FORWARD_LABEL base candidate base
```

### C25上的C22/C26准备验收完成，剩余小候选已冻结（2026-10-23）

两项重新编译及E64 seed0 strict均通过；每项27条BYTE_CHECK和完整MoE
22020096 B零差异，GEMM2完整ELF保持。protected counts/offsets、ordered fences、
WMMA/BF16/FP4/tanh计数保持；资源均为VGPR797 /SGPR56 /LDS327680 B /private0 /spills0。

| 候选 | executable instructions | strict秒 | GEMM1 ELF SHA256 |
|---|---:|---:|---|
| C22 | 6337 | 36.5167 | `75559961c4deb27b67eea859eb4fd602efb9ce6592304d62353c564fa01ac6e6` |
| C26 | 6301 | 35.8860 | `1d971a4ebcb65ffb9818e35bee1c758abb7c917d236e91012080b04d82b4e864` |

两份37成员准备archive已下载并独立重算machine words、native及完整MoE检查，
local acceptance已回传。C22 archive SHA为
`fd05b51264b272616ebf809a531b6c9533ce1df4d49391353ebc6379281d19e6`，
C26为`8d74289f97dc83ec9f22e00234f973863cfd75c195c3534636db1790b9680ee6`。

原入口C22正向`reboot20261023_mma2_first1_bitsearch_c25_closing_c22_forward_v1`
正在运行；首control try1因一次post/settled检查不合格整组三轮排除，中值null。
仅完成准备或某一位置的三轮不构成性能采用结论，当前生产保留版仍为C25。

DS-first6/4/2、udiv、pow2已重新冻结到C25，49个必要文件上传并逐SHA核验。
upload manifest SHA为`11fc553c06036528d38e6ea6d29fef6e93312d6214e3451e15dccef8303de581`。
DS实际hint16800项和scope/override12600项、udiv非负除法723900项、
pow2额外signed-i32 floor/wrapped-add60063项静态检查通过。
这些候选尚无新base上的native/device/performance，不复用旧base结果，
亦不重新引入此前native已拒绝的DS-first0。

### C25上的C22正向双侧改善，反向确认已启动（2026-10-23）

`reboot20261023_mma2_first1_bitsearch_c25_closing_c22_forward_v1`完整退出，
parent3584270已退出，completed/source_restored=true，无error，source精确恢复C25。
首control try1和candidate try1因ownership不合格整组三轮排除、中值null。

| 位置 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前C25 control try2 | 81.9660 /82.4657 /81.3059 | 81.9660 | 63.9767 |
| C22 candidate try2 | 81.4452 /81.3439 /82.0377 | 81.4452 | 64.6242 |
| 后C25 control try1 | 81.8394 /81.6918 /81.8206 | 81.8206 | 64.4281 |

C22相邻改善0.635385% /0.458809%。284成员、1027320 B原证据archive已安全下载，
原table、ownership、source/原call key/实际完整ELF和完整MoE检查独立复算通过。
archive SHA=`956357c887c2f2d6adc9dfe3ff5a20fe46efb6b8ae1d20d11a3269583977d29c`；
local acceptance SHA=`e028887efe05a2a5006b215b29a05a6370da45577e02c94486562741f2a11bf0`，已回传。

反向`reboot20261023_mma2_first1_bitsearch_c25_closing_c22_reverse_v1`已启动，
parent3652260；尚无反向性能结论，候选不采用、不用单侧成绩更新正式保留值。
新的followup/adopt工具绑定C25 source及其已通过的追加验证，6项展开AST检查确认
strict/scope函数、environment、E65 fallback和child120秒上限保持。
历史bitsearch-base的followup/adopt不会用于本轮采用。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_c25_closing_neighbors_screen_v1.py \
  run c22 NEW_UNIQUE_C25_BASE_C22_REVERSE_LABEL candidate base candidate
# 仅正反向四项比较均改善后执行新的追加验证。
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_c25_closing_followup_v1.py run c22
```

### C22反向双侧回退，保持C25并继续C26（2026-10-23）

`reboot20261023_mma2_first1_bitsearch_c25_closing_c22_reverse_v1`完整结束，
parent3652260已退出，completed/source_restored=true，无error；远端source实际SHA
精确恢复C25。九轮全部通过ownership检查，没有排除组。

| 反向位置 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前C22 candidate try1 | 80.9096 /82.2086 /82.7924 | 82.2086 | 64.7233 |
| 中C25 control try1 | 80.9771 /80.5658 /82.5900 | 80.9771 | 64.6265 |
| 后C22 candidate try1 | 81.1183 /82.3094 /81.8478 | 81.8478 | 64.1475 |

从完整原表独立重算，C22反向两侧分别回退1.520800% /1.075242%；
正向的小幅改善未能在反向确认。C22不采用，不追加followup，不修改正式保留成绩。
212成员、789564 B原证据archive已安全解包，逐路径/大小/SHA及原table、ownership、
source、original call key、实际ELF、完整MoE检查独立验收通过。
archive SHA=`710b554c9bdb46e04caaba73a25b4d9aa22c85ec091344046791fe18181257c1`；
local acceptance SHA=`ec948d5e6744e9e5c437db32b022c5c9d21c08170abf4eed2d6fb5f0195b21d3`。

下一项为已在C25 base重新完成native/strict准备验收的C26，使用
`reboot20261023_mma2_first1_bitsearch_c25_closing_c26_forward_v1`执行
base→candidate→base的原入口三轮组对照。尚无C26性能结论；
只有正向双侧改善才启动反向，四项相邻比较全部改善并通过追加检查后才考虑采用。
复现环境、120秒child上限、全机idle及整组ownership排除口径保持。

```bash
python .codex_tmp/yadai_gemm1_param_20261021/tools/verify_downloaded_original_screen_archive_v1.py \
  reboot20261023_mma2_first1_bitsearch_c25_closing_c22_reverse_v1
python .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_c25_closing_neighbors_screen_v1.py \
  accept reboot20261023_mma2_first1_bitsearch_c25_closing_c22_reverse_v1
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_c25_closing_neighbors_screen_v1.py \
  run c26 reboot20261023_mma2_first1_bitsearch_c25_closing_c26_forward_v1 base candidate base
```

### C26正向方向不一致，不采用，继续DS-first6（2026-10-23）

`reboot20261023_mma2_first1_bitsearch_c25_closing_c26_forward_v1`完整结束，
completed/source_restored=true，无error；source实际SHA恢复正式C25。
前control与candidate均首组三轮合格；后control try1因r3 post ownership不合格
整组排除、中值null，try2为首个合格组三轮。没有按耗时重试或拼接不同try。

| 位置 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前C25 control try1 | 82.4803 /81.9595 /82.9949 | 82.4803 | 64.4155 |
| C26 candidate try1 | 82.5288 /81.5486 /81.8692 | 81.8692 | 64.6855 |
| 后C25 control try2 | 81.0446 /81.5064 /81.7128 | 81.5064 | 64.1855 |

C26相对前侧改善0.740904%，相对后侧回退0.445118%，正向未通过双侧gate。
不执行反向或followup，不采用，也不以本序列control重写C25正式成绩。
248成员、906984 B原证据archive已下载并安全解包，逐大小/SHA、原table、ownership、
source、原call key和实际完整ELF独立复算通过。
archive SHA=`7e2de83b900e75020e1815281ab5cb77f4c25497f5a2a30ea0bd8828ac058662`；
local acceptance SHA=`b6616a5ab3bfdae59529b9aaf8bf1de125a6c42d395bb167ac756a3387f92481`。

后续按已冻结顺序DS-first6→4→2→udiv→pow2推进。首项DS-first6只改变现有
GEMM1 guard内future LDS reads的首组compiler hint上限，保留显式DS env优先级、
原DS/TDM/WMMA总量、有效累加、全部硬件fence及scope外支持；相对C25为+2/-1行。
此时仅有静态准备，仍须native、完整MoE strict及相邻原入口性能验收，不能据hint推断收益。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_bitsearch_c25_dsfirst_neighbors_v1.py cpu dsf6
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_bitsearch_c25_dsfirst_neighbors_v1.py native dsf6
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_bitsearch_c25_dsfirst_neighbors_v1.py strict dsf6
```

### DS-first6准备通过，开始C25双侧原入口对照（2026-10-23）

`mma2_first1_bitsearch_c25_dsfirst6_v1` CPU-only及native检查通过，CPU parent3776578已退出。
资源VGPR797 /SGPR56 /LDS327680 B、private/spills0保持；executable instructions6308，
原C25为6291。protected native counts/offsets、ordered fences和WMMA/BF16/FP4/tanh计数保持。
candidate GEMM1 ELF=`c312376b5bc226f69c62110dac9fe5f1c5894cbeb742ffb75a04b709e28463e8`。

E64 seed0 strict在36.1091秒完成，27项BYTE_CHECK和完整MoE22020096 B均零差异，
same-pointer dynamic replay及原reference通过，GEMM2完整ELF保持。
43成员、402495 B preparation archive已安全下载，本地重算source单处hint差异、
machine words、native和完整MoE检查通过。archive SHA为
`8683b18a69495f54a3fb77b6b5fbfe9f93b76e528f0036977ad68b1cf5a54691`；
local acceptance SHA=`f76c846db7041bfdaaa58f8c522b873a5919f2d54ddbf4b6305485aa9c9286ae`。

接下来正向label为`reboot20261023_mma2_first1_bitsearch_c25_dsfirst6_forward_v1`，
base→candidate→base各完整合格三轮，原CLI、idle、ownership、120秒child和cache/ELF绑定保持。
这份准备不构成性能收益，也不替代采用前的E64 seed17、E65及scope追加验证；正式保留版仍C25。

```bash
python3 .codex_tmp/yadai_gemm1_param_20261021/tools/seal_mma2_first1_bitsearch_c25_dsfirst_preparation_v1.py dsf6
python .codex_tmp/yadai_gemm1_param_20261021/tools/verify_downloaded_preparation_archive_v1.py \
  mma2_first1_bitsearch_c25_dsfirst6_v1
python .codex_tmp/yadai_gemm1_param_20261021/tools/accept_mma2_first1_bitsearch_c25_dsfirst_preparation_v1.py dsf6
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_c25_dsfirst_neighbors_screen_v1.py \
  run dsf6 reboot20261023_mma2_first1_bitsearch_c25_dsfirst6_forward_v1 base candidate base
```

### DS-first6正向双侧小幅改善，继续反向确认（2026-10-23）

`reboot20261023_mma2_first1_bitsearch_c25_dsfirst6_forward_v1`完整结束，
completed/source_restored=true，无error，source实际SHA恢复C25。
candidate try1～3因ownership不合格整组三轮排除，中值null；try4为首个合格组。
前后control均try1合格。不按耗时重试、不跨try拼接。

| 位置 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前C25 control try1 | 81.4221 /81.1752 /83.0286 | 81.4221 | 64.3480 |
| DS-first6 candidate try4 | 81.3819 /81.0467 /82.5394 | 81.3819 | 64.1097 |
| 后C25 control try1 | 81.7087 /82.3769 /81.2533 | 81.7087 | 64.8247 |

相邻改善0.049372% /0.399957%；最小差距仅0.0402 μs，不据此宣称稳定收益或达到80 μs。
324成员、1149230 B原证据archive已安全下载并独立复算，SHA为
`d0e8edc91629736028d2ac35b673c970611f797e9b6a3498ba084847afdce944`；
local acceptance SHA=`fcdf8edccab81f89d5bc9ea68d04b2d804e8862d15e0da50f2a5cc3f5e64d0f8`。

反向label=`reboot20261023_mma2_first1_bitsearch_c25_dsfirst6_reverse_v1`，
顺序candidate→base→candidate，各完整合格三轮。只有反向双侧也改善才追加验证。
DS-first6专用followup/adoption工具本地静态检查通过：绑定source/实际ELF，
strict/scope/environment/120秒child及原采用gate保持。checker曾将原组合assert
误按独立字符串查找而拒绝，修正checker的字符串后通过；没有修改或放宽实际验收条件。
这仍为工具准备，不是执行采用。当前C25正式成绩保持，E65原fallback限制如前。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_c25_dsfirst_neighbors_screen_v1.py \
  run dsf6 reboot20261023_mma2_first1_bitsearch_c25_dsfirst6_reverse_v1 candidate base candidate
# 仅正反向四项比较全部改善后执行。
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_c25_dsf6_followup_v1.py run
```

### DS-first6反向未复现双侧收益，不采用；继续DS-first4（2026-10-23）

`reboot20261023_mma2_first1_bitsearch_c25_dsfirst6_reverse_v1`完整结束，
九轮均通过ownership检查，三组各try1合格；completed/source_restored=true、无error，
远端source实际SHA恢复C25，结束检查没有本任务GPU/KFD进程残留。

| 反向位置 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前DS-first6 candidate try1 | 81.2279 /81.7152 /82.1031 | 81.7152 | 64.5294 |
| 中C25 control try1 | 81.3103 /82.4531 /81.2196 | 81.3103 | 64.5294 |
| 后DS-first6 candidate try1 | 82.5965 /80.8421 /80.7704 | 80.8421 | 64.6116 |

相邻改善为−0.497969% /+0.575819%，方向不一致；正向的微小改善未能通过反向确认。
不执行followup/adoption，不以80.7704 μs单轮或后组80.8421 μs更新正式保留成绩。
216成员、791409 B原证据archive已安全下载并独立复算，SHA为
`c4021e33fea2bb24a2387f3e46ea9fee2f98600a2e8e608b91fa70f4363981a1`；
local acceptance SHA为`5b19d13110a9a116529705cfbf23ae936d9615e74bfd704648d308d9fa772767`。
原table、全机idle/ownership、source、原call keys和真实两阶段完整ELF核对通过。

下一项按事前固定顺序准备DS-first4：仅同一GEMM1 guard内首future DS hint上限4，
保留显式env优先级、原计算/transfer/fences及scope外行为。此时尚无DS-first4性能结论。

```bash
python .codex_tmp/yadai_gemm1_param_20261021/tools/verify_downloaded_original_screen_archive_v1.py \
  reboot20261023_mma2_first1_bitsearch_c25_dsfirst6_reverse_v1
python .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_c25_dsfirst_neighbors_screen_v1.py \
  accept reboot20261023_mma2_first1_bitsearch_c25_dsfirst6_reverse_v1
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_bitsearch_c25_dsfirst_neighbors_v1.py cpu dsf4
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_bitsearch_c25_dsfirst_neighbors_v1.py native dsf4
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_bitsearch_c25_dsfirst_neighbors_v1.py strict dsf4
```

### DS-first4完整准备通过，进入原入口对照（2026-10-23）

`mma2_first1_bitsearch_c25_dsfirst4_v1` CPU/native和E64 seed0 strict通过；
strict37.130787秒完成27项BYTE_CHECK、same-pointer dynamic replay及完整MoE22020096 B零差异。
GEMM1完整ELF=`431e11cf71d9ab92596a5a424a5ea4308519d1eaa0f665649e11d02ba176b62d`；
VGPR797 /SGPR56 /LDS327680 B、private/spills0保持，executable static instructions6296，
C25为6291。DS/TDM计数和offset、ordered protected fences以及WMMA/BF16/native FP4/tanh计数保持，
GEMM2完整ELF仍为原版。

40成员、400296 B preparation archive已安全下载和本地独立复算通过，SHA为
`7d0b40359b41040eddc086196e2292240aa1497a49bd8271409bb1fac30242a3`；
local acceptance SHA=`91e0b748a6d5895912c37a8a009f74a087af13439aa8d5748e874ec3a32d605d`。
这仅为准备结果，不构成性能改善；正式保留C25。
下一步label为`reboot20261023_mma2_first1_bitsearch_c25_dsfirst4_forward_v1`，
base/candidate/base各完整合格三轮，原CLI、全机idle、ownership和120秒child上限保持。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_c25_dsfirst_neighbors_screen_v1.py \
  run dsf4 reboot20261023_mma2_first1_bitsearch_c25_dsfirst4_forward_v1 base candidate base
```

### DS-first4正向双侧改善，继续反向确认（2026-10-23）

`reboot20261023_mma2_first1_bitsearch_c25_dsfirst4_forward_v1`完整结束，
completed/source_restored=true、无error；三组各try1全部九轮合格，没有ownership排除组。
原入口precision通过，两阶段实际完整ELF匹配各自原call key，结束source恢复C25。

| 正向位置 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前C25 control try1 | 82.2886 /83.1192 /82.1683 | 82.2886 | 64.3901 |
| DS-first4 candidate try1 | 81.1523 /81.5232 /81.5843 | 81.5232 | 64.5252 |
| 后C25 control try1 | 81.6202 /81.8098 /81.5695 | 81.6202 | 64.7822 |

相邻改善0.930141% /0.118843%，最小差距0.0970 μs。正向通过，仍需反向确认；
不据此采用，不重写C25正式成绩。213成员、789225 B archive已安全下载并本地独立复算，
SHA=`dc3b85e9436f1cd127993df324a31ba90b2e29c8c07f15e3fa2defab0b3f36a6`；
local acceptance SHA=`0d082c2a7b95a58c5e1db3f544e2fbded406eaffa8251a557757fbd20bf01ee0`。

反向label为`reboot20261023_mma2_first1_bitsearch_c25_dsfirst4_reverse_v1`，
candidate/base/candidate各完整合格三轮。只有四项相邻比较全部改善，才追加其他seed和scope验证。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_c25_dsfirst_neighbors_screen_v1.py \
  run dsf4 reboot20261023_mma2_first1_bitsearch_c25_dsfirst4_reverse_v1 candidate base candidate
```

### 反向ownership等待与当前C25 prologue静态复查（2026-10-23）

DS-first4反向前candidate try1/r2 ownership不合格，整组三轮排除，中值null；
try2/r1合格之后，下一轮pre检查发现外部`dewei-triton-1021`容器的GPU/KFD任务，
自动每6秒等待全机空闲，没有终止外部进程，也没有使用占用期间的耗时。

等待期间本地为DS-first4生成独立followup/adoption wrapper并静态验收。
绑定本候选source/真实ELF，原strict、scope、120秒child、四项相邻performance gate保持；
这些工具尚未执行。wrapper AST记录SHA分别为
`987ae93ca7f9dccbdbfa4eef613005177106eab39366ec3629b667e1891c433d`和
`5ffe1a2ecb6ade3cab596ff9f762d3f1fa94dc9bdb4406a9a09f89ba18d4ba6d`。

`review_c25_prologue_v1.py`重新核对C25完整ELF和每条ISA machine words，
首TDM前211条、首WMMA前678条static executable instructions。
首TDM前两处`v_s_rcp_f32`属于当前task mapping的除法展开，8处`s_load_b32`和8处
`s_wait_kmcnt`包含valid-tail检查、expert bitsearch及最终row end读取。
这是静态ISA证据，不是动态关键路径或可节省μs结论；当前C25尚未重新采集thread trace。
后续仍按既定DS-first2→udiv→pow2顺序，以原入口对照验证等价整数表达式是否带来收益。
复算JSON SHA=`067c5379aa1ac4994f328aa1f85c983bde7821c8ca97ec5f272a525bb34daaa2`。

```bash
python .codex_tmp/yadai_gemm1_param_20261021/tools/review_c25_prologue_v1.py
# 新目录复现；已有结果采用写入一次策略，不能覆盖原记录。
python .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_dsf4_followup_wrappers_v1.py
python .codex_tmp/yadai_gemm1_param_20261021/tools/check_mma2_first1_bitsearch_c25_dsf4_followup_v1.py
python .codex_tmp/yadai_gemm1_param_20261021/tools/check_mma2_first1_bitsearch_c25_dsf4_adoption_v1.py
```

### DS-first4反向双侧改善，进入追加验证（2026-10-23）

`reboot20261023_mma2_first1_bitsearch_c25_dsfirst4_reverse_v1`完整结束，
completed/source_restored=true、无error，source实际SHA恢复C25。
前candidate try1/r2因外部ownership不合格，整组三轮排除、中值null；
前candidate try2以及中control、后candidate try1各三轮全部合格，未跨try拼接。
结束检查没有本任务GPU/KFD进程残留。

| 反向位置 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前DS-first4 candidate try2 | 81.4895 /81.5084 /82.2737 | 81.5084 | 64.8139 |
| 中C25 control try1 | 82.7503 /81.1774 /82.2085 | 82.2085 | 64.1814 |
| 后DS-first4 candidate try1 | 81.7129 /81.7594 /82.1683 | 81.7594 | 64.6728 |

反向相邻改善0.851615% /0.546294%，与正向0.930141% /0.118843%均为正。
候选通过正反向四项性能gate，尚未达到80 μs，不使用单轮或跨时段control重写C25成绩。
249成员、909581 B原证据archive已安全下载并本地独立复算，SHA为
`c8b1410dfc9becd4053cd988a5125a220ab0ede803c91e5ec2f349c14edffe4c`；
local acceptance SHA为`118c6fe6bf80d1dd89f07b2f62a92c0c8ecc633626037599d442e29aeaa94002`，
已回传并核SHA。原tables、idle/ownership、source、真实原call keys及两阶段ELF均通过核对。

追加验证工具已同步并核SHA，`mma2_first1_bitsearch_c25_dsf4_followup_v1.py run`
在tmux `%8`启动。E64 seed17、E65 seed0/17完整MoE逐字节验证分别36.343893、
30.485788、12.824452秒完成，每项27个BYTE_CHECK及完整MoE22020096 B零差异。
当前正在完成8项outside-scope编译验证，尚未采用生产源码。
E65仍为原t64/w1x4 fallback；不能据此宣称E64/E65共用compiled GEMM1，
固定shared expert1 route要求也尚未完成。全部追加验证通过后才采用，剩余候选须重新冻结到新base。

```bash
python3 .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_c25_dsfirst_neighbors_screen_v1.py \
  seal reboot20261023_mma2_first1_bitsearch_c25_dsfirst4_reverse_v1
python .codex_tmp/yadai_gemm1_param_20261021/tools/verify_downloaded_original_screen_archive_v1.py \
  reboot20261023_mma2_first1_bitsearch_c25_dsfirst4_reverse_v1
python .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_c25_dsfirst_neighbors_screen_v1.py \
  accept reboot20261023_mma2_first1_bitsearch_c25_dsfirst4_reverse_v1
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_c25_dsf4_followup_v1.py run
```

### DS-first4追加验证通过并保留；后续候选必须重绑（2026-10-23）

追加3项strict及8项outside-scope检查全部通过。8项分别为GEMM2、FP8 input、
非preshuffled A、pin1、t64 fallback、w1x4、buffers3及persistent1，完整ELF与原control一致。
112成员、615270 B followup archive已安全下载、本地复算通过，SHA为
`25bb95255257188f24a02c3b2e22962585e7bc561d09510e36c1892a02ed94d9`；
followup local acceptance SHA为`63dc1988a55e19f9355f4da2cdde7dc96e2c5e449cf9f1c36f9e4ae03fe301f9`，
已回传并核SHA。主机/本地均在原source SHA门禁下执行inspect/apply，保持其他用户修改。

采用记录为`logs/mma2_first1_bitsearch_c25_dsfirst4_production_adoption_v1.json`。
生产source=`d004fdd6a70b1564f07c0a8b5088422578c01f8b0818351881b205a32337bcf6`；
GEMM1 ELF=`431e11cf71d9ab92596a5a424a5ea4308519d1eaa0f665649e11d02ba176b62d`；
GEMM2 ELF=`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。
VGPR797 /SGPR56 /LDS327680 B、private/spills0不变。
正向正式中值81.5232 μs，反向确认81.5084 /81.7594 μs，未达到80 μs。
E65原fallback精度保持；共用compiled GEMM1及固定shared expert route仍未完成。

新增生产改动仅2行增加/1行替换：已有guard内的首future DS cap4，
显式`AITER_FLYDSL_DS_FIRST_N`优先，原计算/transfer/fences/shape支持保持。
后续按DS-first2→udiv→pow2顺序，以DS-first4重新冻结source、编译及验证；
不得复用此前C25的native/device/performance结论。

```bash
python3 .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_c25_dsf4_followup_v1.py seal
python .codex_tmp/yadai_gemm1_param_20261021/tools/verify_downloaded_followup_archive_v1.py \
  mma2_first1_bitsearch_c25_dsf4_followup_v1
python .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_c25_dsf4_followup_v1.py accept
python3 .codex_tmp/yadai_gemm1_param_20261021/tools/adopt_mma2_first1_bitsearch_c25_dsf4_v1.py inspect
python3 .codex_tmp/yadai_gemm1_param_20261021/tools/adopt_mma2_first1_bitsearch_c25_dsf4_v1.py apply
```

### DS-first4 基准上的 DS-first2：完整正向双侧回退（2026-10-23）

以正式 DS-first4 source `d004fdd6...37bcf6`、GEMM1 ELF `431e11cf...6b62d` 为对照，
重新编译并验证 DS-first2，没有复用旧 C25 基准的性能或正确性结论。候选只把已有
guard 内的 first DS cap4 改为2，+1/-1行；显式 `AITER_FLYDSL_DS_FIRST_N` 仍优先，
数学、transfer、硬件 fences 和 shape 支持保持。

候选 source=`d156f1e8f5c955cc45fd4232f4655c733a1d6911620ed6a7e4325a6fd31e2452`，
GEMM1 ELF=`b28c9fd5a091dc4edb9bf8e6227421e5f5afc853c0402320b7acdbcab8f720e5`。
CPU/native gate 和 E64 seed0 strict 通过，27项 BYTE_CHECK 及完整 MoE 22020096 B
逐字节一致；资源保持 VGPR797 /SGPR56 /LDS327680 B、private/spills0，
executable instructions 为6302。strict child 用时36.242648秒，此耗时不是性能数据。

`reboot20261023_mma2_first1_bitsearch_c25_dsf4_dsfirst2_forward_v1` 已完整结束。
candidate try1～try3 的测后 ownership 不合格，均整组三轮排除、中值null；
try4为首个完整合格组。两侧 control 均try1合格，不按耗时选择重试或跨组拼轮。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前 DS-first4 control | 82.0778 /81.5170 /81.6518 | 81.6518 | 64.5842 |
| DS-first2 candidate try4 | 81.8353 /81.8943 /82.3855 | 81.8943 | 64.2026 |
| 后 DS-first4 control | 81.6287 /81.9112 /81.5126 | 81.6287 | 64.2446 |

candidate 相对两侧回退0.2425 /0.2656 μs（0.296993% /0.325376%），
不追加反向、followup或adoption，保留 DS-first4。
319成员、1149909 B archive 已安全下载，本地复算原表格、ownership、source/keys及
完整 ELF 通过，archive SHA=`3a0d701efc5540cb9fe996a3e8c3901b00ea88f0781803c2ecf668db351bf339`。
local acceptance SHA=`4a1f0cff6c622d064a3450c913b7b2104c3dc16948e27d24ccaa9ca587baa1de`，
已回传核SHA；`passed=true` 表示证据一致，`positive_at_every_adjacent_control=false`。
生产 source 精确恢复 DS-first4，测后 GPU 检查无测试残留。

```bash
python3 .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_c25_dsf4_dsfirst_neighbors_screen_v1.py \
  run dsf2 reboot20261023_mma2_first1_bitsearch_c25_dsf4_dsfirst2_forward_v1 base candidate base
python3 .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_c25_dsf4_dsfirst_neighbors_screen_v1.py \
  seal reboot20261023_mma2_first1_bitsearch_c25_dsf4_dsfirst2_forward_v1
python .codex_tmp/yadai_gemm1_param_20261021/tools/verify_downloaded_original_screen_archive_v1.py \
  reboot20261023_mma2_first1_bitsearch_c25_dsf4_dsfirst2_forward_v1
python .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_c25_dsf4_dsfirst_neighbors_screen_v1.py \
  accept reboot20261023_mma2_first1_bitsearch_c25_dsf4_dsfirst2_forward_v1
```

下一项 `udiv` 只将现有非负 task 坐标除法改为 `arith.divui`，+8/-3行，
原 M/N task 映射221300点、scope外78120点和large integer20000点静态复算一致。
tools、candidate 和静态记录12项已上传并逐文件核SHA，正在进行 bounded CPU 编译；
此时尚无该候选的 device 精度或性能结果。

### DS-first4 上的 udiv：编译与完整 MoE 准备通过（2026-10-23）

`mma2_first1_bitsearch_c25_dsf4_udiv_v1` source 为
`6690688437a5ccd1d86635cc6e7f86ba2b5495dbb24fe2127e95085bc034b0c3`，
GEMM1 完整 ELF 为 `c574a44668c7a7cfee8197e3b0c207921d3f7e3047314a118f2cdfefd66df707`。
只改变现有 geometry guard 内三个非负 task 坐标的除法表达式；scope 外沿用原除法。
原 MMA/DS hints、K 累加、输入输出和所有源码 fences 保持。
native 检查的 DS/TDM/ScaleA/payload 数量与 offsets、ordered fences、
WMMA/BF16/FP4/tanh 操作计数相同，资源保持 VGPR797 /SGPR56 /LDS327680 B、private/spills0。

E64 seed0 strict 用时36.734905秒，27项 BYTE_CHECK、动态 same-pointer graph 和
完整 MoE 22020096 B 零差异，GEMM2 ELF 保持 `4bd626e...716be`。
原 reference 指标为 logits_diff=`3.6105513278025183e-06`、rel_l2=`0.0026872099842876196`，
原 gate 未修改。40成员、404153 B preparation archive 已下载、逐文件 SHA 及
ISA machine words 独立复算通过；archive SHA 为
`7ff445f8dd85095bb0b077af09b2571ecf01b7f6e9b6d77dd95d3db67255ad77`。
preparation local acceptance SHA 为
`631fb7d89ed54e340ae83c63fc21fc4cdacd230da3d7e0c1e05909c4d49a321d`，已回传且一致。

`logs/dsf4_udiv_static_prologue_comparison_v1.json` 重新绑定两份实际 ELF 和逐条 machine words：
executable instructions 从6296降到6269；首 TDM 前211→184条，首 WMMA 前670→643条。
两处 reciprocal 和8处 table load 仍存在。以上仅为静态计数，不能当动态延迟或性能收益。
随后启动 `reboot20261023_mma2_first1_bitsearch_c25_dsf4_udiv_forward_v1`，
仍为原 `--scenario kernel`、base/candidate/base，每组完整三轮；此处尚未记录最终性能结论。

```bash
python3 .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_bitsearch_c25_dsf4_udiv_v1.py cpu udiv
python3 .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_bitsearch_c25_dsf4_udiv_v1.py native udiv
python3 .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_bitsearch_c25_dsf4_udiv_v1.py strict udiv
python3 .codex_tmp/yadai_gemm1_param_20261021/tools/seal_mma2_first1_bitsearch_c25_dsf4_udiv_preparation_v1.py udiv
python .codex_tmp/yadai_gemm1_param_20261021/tools/verify_downloaded_preparation_archive_v1.py \
  mma2_first1_bitsearch_c25_dsf4_udiv_v1
python .codex_tmp/yadai_gemm1_param_20261021/tools/accept_mma2_first1_bitsearch_c25_dsf4_udiv_preparation_v1.py udiv
python3 .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_c25_dsf4_udiv_screen_v1.py \
  run udiv reboot20261023_mma2_first1_bitsearch_c25_dsf4_udiv_forward_v1 base candidate base
```

### udiv 正向双侧小幅改善，开始反向确认（2026-10-23）

`reboot20261023_mma2_first1_bitsearch_c25_dsf4_udiv_forward_v1` 完整结束，
生产恢复 DS-first4，结束 GPU 检查无测试残留。后 control try1 的r2 post ownership
不合格，整组三轮排除、中值null；try2为首个完整合格组。其余两组try1合格。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前 DS-first4 control | 82.0040 /81.9363 /81.6814 | 81.9363 | 64.2636 |
| udiv candidate | 81.7933 /81.7531 /81.6940 | 81.7531 | 64.2679 |
| 后 DS-first4 control try2 | 82.0061 /81.2996 /82.0271 | 82.0061 | 64.2699 |

candidate 比两侧快0.1832 /0.2530 μs（0.223588% /0.308514%）。
251成员、910816 B archive 已下载并独立复算原表格、ownership、source/keys/ELF，
SHA=`d5a78582e308588a9efbb96434ecf37080859bf7a720ce0407003d044bdde74a`。
local acceptance SHA=`85e70b8520acdc30ffbb5dc5a910a5a91916c351fddb2a260dd8da3ec3335aee`，
已回传且一致，`positive_at_every_adjacent_control=true`。

继续 `reboot20261023_mma2_first1_bitsearch_c25_dsf4_udiv_reverse_v1`，
candidate/base/candidate，各完整三轮；尚未采用。匹配该 source/ELF 的 followup/adoption
工具仅在本地准备，展开后的原 strict、scope、120秒 child、四项相邻收益及采用门禁均保持，
未以正向收益代替反向或追加精度/scope 验证。

```bash
python3 .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_c25_dsf4_udiv_screen_v1.py \
  run udiv reboot20261023_mma2_first1_bitsearch_c25_dsf4_udiv_reverse_v1 candidate base candidate
```

### udiv 反向完整双侧回退，不采用（2026-10-23）

反向 candidate/base/candidate 已完整结束，各位置首个 ownership 合格的完整三轮均为 try2。
三个 try1 都因 post ownership 不合格整组排除、中值 null；不跨组拼轮、不按耗时重试。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前 udiv candidate try2 | 81.4999 /81.9219 /81.8857 | 81.8857 | 64.2321 |
| DS-first4 control try2 | 81.6160 /82.9107 /81.4430 | 81.6160 | 64.5926 |
| 后 udiv candidate try2 | 82.4741 /81.8710 /81.8142 | 81.8710 | 64.4681 |

udiv 相对中间 control 分别回退0.2697 /0.2550 μs（0.330450% /0.312439%）。
此前正向0.223588% /0.308514%的小幅改善未在反向复现，故放弃该候选，
不上传或执行其 followup/adoption 工具。静态减少27条 executable instructions不代表稳定动态收益。

323成员、1148832 B archive已安全下载，原表格、ownership、source/keys及完整ELF本地独立复算通过。
archive SHA256=`7ff0a4442ba0605ae87edf1366d1b15690c4f569893ba85ff5dfc85a5e1f7955`；
local acceptance SHA256=`dc64d2668525ba8bd9982e63dea71c0cd4b0ac6fe5ed0a4de71bfe6a35758563`，
`positive_at_every_adjacent_control=false`。
summary `completed=true/source_restored=true`；生产SHA恢复
`d004fdd6a70b1564f07c0a8b5088422578c01f8b0818351881b205a32337bcf6`，
结束主机GPU/KFD检查无测试进程残留。正式最优仍为DS-first4，未达80 μs；
E65仍为原fallback，尚未完成E64/E65共用compiled GEMM1及固定shared expert routing要求。

下一项按已冻结顺序测试 `pow2`：只在已有guard内用signed shift/mask重写五个固定除法表达式，
+6/-0行，原 M/N task mapping、signed i32 floor/wrap语义、scope外代码保持。
准备工具和candidate已本地核SHA/LF，在本序列完全结束后上传、重新编译和strict验证；
此处没有该候选的device或性能结论。

```bash
python3 .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_c25_dsf4_udiv_screen_v1.py \
  seal reboot20261023_mma2_first1_bitsearch_c25_dsf4_udiv_reverse_v1
python .codex_tmp/yadai_gemm1_param_20261021/tools/verify_downloaded_original_screen_archive_v1.py \
  reboot20261023_mma2_first1_bitsearch_c25_dsf4_udiv_reverse_v1
python .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_c25_dsf4_udiv_screen_v1.py \
  accept reboot20261023_mma2_first1_bitsearch_c25_dsf4_udiv_reverse_v1
```

### DS-first4 上的 pow2 准备完成，进入原入口对照（2026-10-23）

candidate source=`ca5246fa5065ca8b411c80d2175a3e13a134c9d49ec3d4d41cc9e59b4139a224`，
GEMM1 ELF=`a389f84f74e006d13b94fafa5a42d4295179db8731246cc1876a7f5627d84e05`。
CPU/native通过：executable instructions从6296变为6271，资源仍为
VGPR797 /SGPR56 /LDS327680 B，private/spills0。protected DS/TDM/ScaleA/payload
计数与offsets、ordered fences、WMMA/BF16/FP4/tanh计数保持。
静态计数减少25条尚不代表动态性能收益；没有新增pipeline/persistent/wave布局。

E64 seed0 strict child用时36.587812秒，27项BYTE_CHECK及完整MoE22020096 B零差异，
GEMM2 ELF保持`4bd626e...716be`。strict耗时不作为性能数据。
40成员、403618 B preparation archive已安全下载并本地独立核ISA machine words、
原表格、source/ELF及完整MoE数据；archive SHA为
`f1a1e92f5c39ef40bfe6cf3000293cca699e2ef1a858812851cd84fd9d956b61`，
local acceptance SHA为`4e43d8b55490d2dbe27d98347c7ec9c55e6f5f103d8bdf45d0727a91a42fc946`，已回传核验。

接下来运行`reboot20261023_mma2_first1_bitsearch_c25_dsf4_pow2_forward_v1`：
原`--scenario kernel`，A preshuffle1、VGPR pin0、warmup5/iters20、
base/candidate/base完整三轮组，ownership不合格整组排除，不跨组拼轮。
尚无pow2性能结论；结束后恢复DS-first4源码。

```bash
python3 .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_bitsearch_c25_dsf4_pow2_v1.py cpu pow2
python3 .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_bitsearch_c25_dsf4_pow2_v1.py native pow2
python3 .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_mma2_first1_bitsearch_c25_dsf4_pow2_v1.py strict pow2
python3 .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_c25_dsf4_pow2_screen_v1.py run pow2 reboot20261023_mma2_first1_bitsearch_c25_dsf4_pow2_forward_v1 base candidate base
```

### pow2 正向未通过双侧改善，不采用；抓取当前 DS-first4 trace（2026-10-23）

`reboot20261023_mma2_first1_bitsearch_c25_dsf4_pow2_forward_v1` 已完整结束。
candidate try1～try3 的 post ownership 不合格，整组三轮排除、中值null；try4为首个完整合格组。
两侧 control 均try1完整合格。不根据耗时筛选重试，不跨组拼接单轮。

| 组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 前 DS-first4 control | 81.7906 /82.5561 /82.4211 | 82.4211 | 64.4363 |
| pow2 candidate try4 | 81.3671 /82.7502 /81.6011 | 81.6011 | 64.5145 |
| 后 DS-first4 control | 81.2720 /81.5168 /81.2742 | 81.2742 | 64.3963 |

candidate 相对前侧改善0.8200 μs（0.994891%），相对后侧回退0.3269 μs（0.402219%）。
方向不一致，不追加反向或followup、不采用；减少25条静态指令未取得稳定收益。
323成员、1147787 B archive已安全下载，原表格、ownership、source/keys及实际两阶段ELF独立复算通过，
archive SHA256=`dd7e0bdf76a7cffec38bb4d4cabaf4099b435dc739d7e2fc4581e32efa63cd50`。
local acceptance `passed=true/positive_at_every_adjacent_control=false`，SHA256=`c0bb836c1649672dd6698fdb5e3ba7ab55899fd73b832752b53332672cb6c63b`。
summary `completed=true/source_restored=true`，生产源码恢复正式DS-first4，结束检查无本任务GPU/KFD进程残留。

正向运行期间本地准备了pow2的followup/adoption工具，仅重绑source/ELF、精确+6/-0行delta及反转，
展开后原3项strict、8项scope、所有4项相邻性能改善和120秒child门槛保持。因正向失败，未上传或执行这些工具。

整理本轮文档时发现PowerShell native stdin将新增两节中文转成问号。
已从独立UTF-8文件重写这两节，原前455441 bytes保持、整份文档仍LF；性能原始日志、源码和ELF未受影响。
修复记录为 `logs/dsf4_doc_encoding_repair_v1.json`。今后中文文档通过UTF-8文件写入，避免同类编码转换。

现继续采集正式最优DS-first4的独立SIMD0～3 trace，label=`reboot20261023_best_dsf4_att_v1`。
调用用户指定 `my_code/get_att_b_rocm_sdk.sh`，原workload、A preshuffle1/pin0、warmup5/iters20保持，
`--all-simd --iteration 28 --ana-att`，每个profiler/decoder child120秒。
绑定正式source `d004fdd6...37bcf6` 和GEMM1 ELF `431e11cf...6b62d`；
必须与actual original call inventory、private cache及所有decoded PC/ISA逐项匹配。
这是诊断数据，不能替换原入口性能。当前尚无新DS-first4动态结果，不复用旧MMA2/first2 trace比例。
下载时省略`.out`与private cache。

```bash
python3 .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_c25_dsf4_pow2_screen_v1.py \
  seal reboot20261023_mma2_first1_bitsearch_c25_dsf4_pow2_forward_v1
python .codex_tmp/yadai_gemm1_param_20261021/tools/verify_downloaded_original_screen_archive_v1.py \
  reboot20261023_mma2_first1_bitsearch_c25_dsf4_pow2_forward_v1
python .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_c25_dsf4_pow2_screen_v1.py \
  accept reboot20261023_mma2_first1_bitsearch_c25_dsf4_pow2_forward_v1
bash .codex_tmp/yadai_gemm1_param_20261021/tools/run_best_dsf4_att_v1.sh \
  reboot20261023_best_dsf4_att_v1
```

### 正式 DS-first4 trace 完成；DS-first3/5 完成准备（2026-10-23）

正式保留版仍为 C25 + DS-first4，source SHA256 为
`d004fdd6a70b1564f07c0a8b5088422578c01f8b0818351881b205a32337bcf6`，
GEMM1 ELF SHA256 为 `431e11cf71d9ab92596a5a424a5ea4308519d1eaa0f665649e11d02ba176b62d`。
正式正向中值 81.5232 μs，反向确认 81.5084 /81.7594 μs；尚未达到稳定低于 80 μs。

`reboot20261023_best_dsf4_att_v1` 的独立 SIMD0～3 capture 和 analysis 全部 rc0，
结束检查无本任务 GPU/KFD 进程残留。原 workload、A preshuffle1/pin0、warmup5/iters20
以及用户指定的 `my_code/get_att_b_rocm_sdk.sh` 保持，使用 `--all-simd --iteration 28 --ana-att`。
实际 private cache 的原 call inventory、sourceIR、globals、keys 和两阶段完整 ELF 逐项绑定通过。
全部 decoded PC/ISA machine words、wave events、phase/opcode/wait 统计在本地独立复算一致。
231 个成员、6976203 B archive 已下载，省略 `.out` 和 private cache；SHA256 为
`c3221b324bc126ac770750e12af7e6eb3e731aa71d9c667d2578ec899fb494c6`。
local archive acceptance 和 local analysis acceptance 均已回传、核对 SHA256。

64 个 active waves 各执行 1792 条 WMMA。累计 wave instruction timestamp gaps 中，
mainloop 为 68.2875%，prologue 为 13.9271%，activation/quant/scale/LDS + barrier 为 14.4640%，
最后 input drain 为 0.2300%，output descriptor 为 0.1826%，output TDM/completion 为 2.9088%。
这包含 arbitration/coexecution，四个 SIMD 又是独立 capture，不能当作 GPU 关键路径分解或推算 μs 收益。
多数 steady K128 subtile0 的每 wave 中值为 326～334 cycles，subtile1 为 301～302 cycles。
详细记录见 `my_code/moe_gemm1_hyg_dsr1_best_thread_trace_analysis.md` 和
`logs/best_dsf4_kstep_detail_v1.json`；后者 SHA256 为
`82254fdb4079b597e5d84aa120b779ac7a00f860e8772e5e1ef457449938f2a9`。

接下来按事前固定顺序测试 DS-first3、DS-first5。各候选只把已有目标 guard 中的 cap4 改为3或5，
各 +1/-1 行；显式 `AITER_FLYDSL_DS_FIRST_N` 仍优先。没有新增 persistent、wave 布局、pipeline、
shape 限制或 env switch；所有 runtime operations 和 scope 外行为保持。

| 候选 | source SHA256 | GEMM1 ELF SHA256 | executable instructions | strict child 秒 |
|---|---|---|---:|---:|
| DS-first3 | `3336930c9863abbf5aaf4f2efaed2c65f0bd2ba13465d338fc08cbd0a97e591a` | `65acabbc95ec4ce9427492e1fa729f1d25f9834db5ac0d63a763e84a0522ea72` | 6306 | 36.303822 |
| DS-first5 | `e53ada2baecb0c8e0185b815052281952213f0c81c5e4413c621baccfb06f93e` | `62037bcfa621ff1c2d378e0e5face1479e6af47fbcb5beb8bacf8ee39d003897` | 6308 | 36.608921 |

两项资源均为 VGPR797 /SGPR56 /LDS327680 B、private/spills0；protected native counts/offsets、
ordered fences 和 WMMA/BF16/FP4/tanh counts 保持。各 E64 seed0 strict 的27项 BYTE_CHECK、
same-pointer dynamic graph replay 及完整 MoE 22020096 B 全部零字节差异；GEMM2 ELF 保持原值。
prepare archive 均已本地安全下载、独立验收并回传。原计时、env、CLI、ownership 和120秒 child
检查均保持；这些 strict 耗时只表示验证完成时间，不是性能。

DS-first3 原入口正向 `base/candidate/base` 运行中。前 control 三轮为
81.0444 /81.3756 /81.4767 μs，中值81.3756 μs；candidate 三轮为
82.4935 /81.5068 /82.5989 μs，中值82.4935 μs。候选相对前侧回退1.373753%。
后 control 的完整合格组尚未收齐。已发现 `dev_al` 的外部 GPU/KFD 进程进入 post/settled 窗口，
受影响整组三轮排除、中值null；继续按6秒 idle gate 等待，不杀外部进程、不按耗时重试、不拼轮。
尚未形成 DS-first3 最终双侧结论，DS-first5 尚未性能计时。

E65 保留分支原 t64/w1x4 fallback。已有 E64/E65 完整 MoE 精度核验不表示已完成
E64/E65 共用 compiled GEMM1 和固定 shared expert1 routing 要求；这些要求仍未完成。

```bash
python3 .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_c25_dsf4_dsfirst3_neighbors_screen_v1.py \
  run dsf3 reboot20261023_mma2_first1_bitsearch_c25_dsf4_dsfirst3_forward_v1 base candidate base
python3 .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_c25_dsf4_dsfirst5_neighbors_screen_v1.py \
  run dsf5 NEW_UNIQUE_DS_FIRST5_LABEL base candidate base
```

### DS-first3 完整正向双侧回退，不采用（2026-10-23）

`reboot20261023_mma2_first1_bitsearch_c25_dsf4_dsfirst3_forward_v1` 已完整结束，
`completed=true/source_restored=true`。后 control try1～try6 因 post/settled 检查不合格整组三轮排除，
中值null；try7 为首个完整合格组。前 control 和 candidate 均 try1 完整合格。
没有按耗时重试或跨组拼轮；所有排除记录均保留。

| 组 | GEMM1 三轮 μs | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---|---:|---:|
| 前 DS-first4 control | 81.0444 /81.3756 /81.4767 | 81.3756 | 65.0985 |
| DS-first3 candidate | 82.4935 /81.5068 /82.5989 | 82.4935 | 64.5529 |
| 后 DS-first4 control try7 | 82.3055 /82.8854 /81.2726 | 82.3055 | 64.5358 |

DS-first3 相对两侧分别慢 1.1179 /0.1880 μs，即回退1.373753% /0.228417%。
不反向、不追加 followup、不采用；生产恢复正式 DS-first4 source `d004fdd6...37bcf6`。
GEMM2 ELF 始终保持，表内 GEMM2 时间波动不归因于 GEMM1 调度改动。

428 个成员、1504984 B archive 已下载并安全核验，原表格、ownership、source、keys 和完整两阶段 ELF
在本地独立复算通过。archive SHA256 为
`ffc63c05d11f01e15c8841b47484955b0e4632dc32df831f874556111db11451`，
`local_acceptance.passed=true/positive_at_every_adjacent_control=false`。
结束 GPU/KFD 检查只发现外部 `dev_al` 的两个测试进程，没有本任务测试进程残留；外部进程未清理。
下一项仍按固定顺序测试 DS-first5，先等待全机空闲，保持原入口与全部验证门槛。

```bash
python3 .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_c25_dsf4_dsfirst3_neighbors_screen_v1.py \
  seal reboot20261023_mma2_first1_bitsearch_c25_dsf4_dsfirst3_forward_v1
python .codex_tmp/yadai_gemm1_param_20261021/tools/verify_downloaded_original_screen_archive_v1.py \
  reboot20261023_mma2_first1_bitsearch_c25_dsf4_dsfirst3_forward_v1
python .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_c25_dsf4_dsfirst3_neighbors_screen_v1.py \
  accept reboot20261023_mma2_first1_bitsearch_c25_dsf4_dsfirst3_forward_v1
```

### DS-first5 正向双侧微小改善，继续反向确认（2026-10-23）

`reboot20261023_mma2_first1_bitsearch_c25_dsf4_dsfirst5_forward_v1` 完整结束，
`completed=true/source_restored=true`；生产恢复正式 DS-first4 source `d004fdd6...37bcf6`。
前 control try1～try3 因 post/settled 检查不合格整组三轮排除、中值null，try4 为首个完整合格组；
candidate 和后 control 均 try1 完整合格。不按耗时重试，不跨组拼轮。

| 组 | GEMM1 三轮 μs | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---|---:|---:|
| 前 DS-first4 control try4 | 81.0614 /81.9808 /81.9666 | 81.9666 | 64.1771 |
| DS-first5 candidate | 81.9194 /81.5925 /81.6350 | 81.6350 | 64.5671 |
| 后 DS-first4 control | 82.4002 /81.6011 /81.6414 | 81.6414 | 64.7591 |

相对两侧改善0.3316 /0.0064 μs（0.404555% /0.007839%）。后侧差异非常小，
不能仅凭正向中值认定稳定收益，更不能据此采用。
320 个成员、1150796 B archive 已下载并逐成员安全核验，原表格、ownership、source/keys 和两阶段完整 ELF
在本地独立复算通过；archive SHA256 为
`fe57138ca8720f6bb6b5c110a5b553cdd515e8f3f9de5439107c1f7af016ebef`。
local acceptance 为 `passed=true/positive_at_every_adjacent_control=true`。
结束 GPU/KFD 检查为空，无本任务测试进程残留。

按原流程继续 `candidate/base/candidate` 反向完整三轮。
只有正反向全部四项相邻对照均改善，且追加3项 strict 与8项 scope 验证完成后，才考虑采用。
正式最优仍为 DS-first4 的81.5232 μs；尚未达到稳定低于80 μs。

```bash
python3 .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_c25_dsf4_dsfirst5_neighbors_screen_v1.py \
  run dsf5 reboot20261023_mma2_first1_bitsearch_c25_dsf4_dsfirst5_reverse_v1 candidate base candidate
```

### 最后一次 expert 查询的相邻双值读取：本地准备（2026-10-23）

当前 GEMM1 使用 `psum` 的串行查询定位 expert；现有 grouped 路径没有可直接复用的 tile→expert 表。
`aiter/fused_moe.py` 的部分 FLAT metadata 中，`sorted_expert_ids` 只是 topk scratch alias，不能据名字当作已生成的查找表。
本次不修改 routing producer，也不新增 metadata kernel 或 launcher ABI。

新 `mma2_first1_bitsearch_c25_dsf4_lastpair_v1` 基于正式 DS-first4 source。
仅在既有 `standard_a_timeout` scope 的最后一个 bit 且 expert 数为偶数时，
读取 `psum[lo]` 和 `psum[lo+1]` 两个相邻 int32；前者用于最后一次比较，
按比较结果选择前者或后者作为最终 expert endpoint，省去查询结束后的单独 endpoint load。
进入最后一轮前 `lo` 为偶数；有效行使 `lo <= n_experts-2`，所以双值读取在边界内。
expert 数为奇数或 scope 外保持原查询，不增加任何 shape 准入限制。

CDNA5 ISA §15.6 p295 记载 `S_LOAD_B64` 从 `calcAddr` 和 `calcAddr+4` 读取两个 DWORD；
§5.7.1.1 p55 记载多 DWORD SMEM 操作的 KMcnt 增量为2，SMEM 可以乱序返回，正常等待应使用 `S_WAIT_KMcnt <= 0`。
这里不手工删除或改变 compiler-generated wait。是否实际生成 scalar B64、是否减少依赖链，须以编译后的 ISA 为准。

相对正式版 +10/-3 行，source SHA256 为
`a0b869be1ed99d0c2d1f9f2759cf5b177f5b47214cea48ab17bda2e01d097484`。
实际 metadata block AST 的82206项边界检查通过，其中62276项为 scope 外或奇数 experts 原路径，
19606项有效行执行一次相邻双值读取，下标均为偶数且无越界。
expert id 和 `mn_oob` 与原算法、独立 `bisect_right` reference 一致，包含空 expert、全空表、重复 endpoint、
E64/E65、首尾 expert、padding 和大 row index。有效偶数 expert 路径的 metadata load request 数减少一次。
反转三个精确修改恢复原 module bytes；原 GEMM 数学、输入/输出/量化布局、TDM、barrier、ring 和 wave 布局保持。

prepare wrapper 的 cpu/native/strict 三个 action 展开后，child/environment function AST 与原版本相同；
保留120秒 child、native counts/offsets/fences/resources 和完整 MoE/动态 graph byte gate。
当前只在本地独立文件中准备，尚未上传、编译、device 验证或计时；生产代码没有此改动。
先完成 DS-first5；若正式基准变化，必须重新绑定、编译与验证这个候选。

### DS-first5 反向完整对照通过，继续追加验证（2026-10-23）

反向 `reboot20261023_mma2_first1_bitsearch_c25_dsf4_dsfirst5_reverse_v1` 已完成。
生产 source 已恢复 DS-first4，SHA256 为
`d004fdd6a70b1564f07c0a8b5088422578c01f8b0818351881b205a32337bcf6`。
结束后主机 `gpu_users.sh` 确认没有 GPU 进程；未杀死外部进程。

| 位置 | GEMM1 三轮 μs | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---|---:|---:|
| 前 DS5 candidate | 82.0672 /80.8357 /81.8183 | 81.8183 | 64.5947 |
| DS4 control try2 | 81.9955 /82.0124 /82.0798 | 82.0124 | 65.0374 |
| 后 DS5 candidate try2 | 81.6455 /81.7086 /82.0229 | 81.7086 | 64.4955 |

control try1、后 candidate try1 的 post/settled 检查不合格，整组三轮排除，中值为 null。
没有跨组拼接轮次，没有按耗时决定重试。反向两侧改善分别为
0.2366715% /0.3704318%；与正向的 0.4045550% /0.0078392% 合起来，四项均为正。
最小改善只有 0.0064 μs，仍存在测量波动的限制，不能据此声称稳健提升或达到 80 μs。
GEMM2 ELF 保持原值，表中其耗时变化不归因于 DS5。

284 regular members、1029842 B archive 已下载，本地逐成员大小/SHA、原始表格、
ownership、source、原入口 cache keys 和实际 ELF 独立复算通过。

- archive SHA256：`42941a5148256047ce03286cbdef7213a6c7cd804f9ecba46d3e6998d9757ad2`
- local acceptance SHA256：`a3d6cda7ea8e07f12fef8e2e65ab76148363e35c3f905bc8a513c49847aad1f0`
- acceptance 已回传并核 SHA，`positive_at_every_adjacent_control=true`。

目前仍不采用 DS5。新增的 followup/adoption wrapper 保留原计时、strict、scope 和
resource 门槛，5 项展开 AST 核验通过；4 项文件上传后逐 SHA 匹配。
下一步使用原 120 秒 child 限时，运行 E64/seed17、E65/seed0、E65/seed17 三项完整
MoE 逐字节检查和 8 项非目标配置完整 ELF 对照，全部通过后才考虑落地该 +1/-1 行 cap 改动。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/mma2_first1_bitsearch_c25_dsf4_dsfirst5_followup_v1.py run
```

正式保留成绩仍为 DS4 的 81.5232 μs。相邻双值 psum 查询候选仍只在本地准备，
尚未上传、编译或计时；如果采用 DS5，必须重新绑定其 source 和 ELF 后再验证该候选。
E65 仍使用原 t64/w1x4 fallback，历史 E64/E65 同 compiled GEMM1 和固定 shared expert1
routing 要求仍未完成，本节不把 fallback 精度通过当作这些要求已满足。

### DS-first5 追加验证通过并保留（2026-10-23）

3 项追加 strict 和 8 项非目标配置检查全部通过，本地 archive 复核通过，
已将 DS-first cap=5 的 +1/-1 行改动保留到本地与 a07-3。两端 source SHA256 一致：
`e53ada2baecb0c8e0185b815052281952213f0c81c5e4413c621baccfb06f93e`。
GEMM1 ELF 为 `62037bcfa621ff1c2d378e0e5face1479e6af47fbcb5beb8bacf8ee39d003897`；
GEMM2 ELF 仍为 `4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。

followup archive 为 112 regular members、614829 B，SHA256
`49ba6fe3504e924ea3516742ab0880fa92f2071fdedb1841cf7ed3775f7239e0`；
local acceptance SHA256 为 `c7803cfd4d1629056e08170458dfdb84c3b43b76767ba2dd40e7dbc7e36fdc99`。
E64/seed17、E65/seed0、E65/seed17 各 27 项逐字节检查及完整 MoE 输出均零差异。
非目标 GEMM2、FP8 input、original A、pin1、t64、w1x4、buffers3、persistent1 的
完整 ELF 与原路径一致。VGPR797 /SGPR56 /LDS327680 B /private0 /spills0 保持。
E65 检查仍是原 fallback，不构成共用 compiled GEMM1 或固定 shared expert1 routing 的完成。

当前保留版正式原入口中值为 81.6350 μs，确认组为 81.8183 /81.7086 μs。
同窗口 DS4 controls 为 81.9666 /81.6414 /82.0124 μs；历史 DS4 的 81.5232 μs
属于另一时段，不跨时段直接比较。四项相邻改善为 0.4045550% /0.0078392% /
0.2366715% /0.3704318%，改善很小，仍未达 80 μs。未修改 benchmark、Git index、commit 或 push。

已重新绑定 lastpair 候选到当前 DS5。candidate source SHA256 为
`9d7cae6b6380e7bcdfcd3139bda111de998a3bc4b334ba3cc673478358fdb998`，
82206 项实际 AST endpoint 对照、62276 项 scope 外精确检查、19606 项成对读取边界检查通过。
15 项候选与工具文件上传逐 SHA 匹配。尚无其性能结论。

### DS5 的相邻 endpoint 读取实验（2026-10-23）

正式保留版是 DS-first5，source SHA256 为
`e53ada2baecb0c8e0185b815052281952213f0c81c5e4413c621baccfb06f93e`。
本记录只补充准备和进行中的状态，完整性能结论须以原入口归档独立验收为准。

lastpair 仅合并最后一个 bit 的两个相邻 psum endpoint，+10/-3 行；偶数 experts 且原 guard 生效时使用，
奇数 experts 和 scope 外保持原读取逻辑。不改 GEMM 数学、TDM、输入输出、量化、shape 准入或 wave 布局。
实际 ISA 从 8 条 `s_load_b32`/8 条 KMcnt waits 变为
6 条 `s_load_b32` + 1 条 `s_load_b64`/7 条 KMcnt waits，保留 compiler-generated waits。

- candidate source：`9d7cae6b6380e7bcdfcd3139bda111de998a3bc4b334ba3cc673478358fdb998`
- GEMM1 ELF：`7426a6749a94b271c3bb75652eda5c3662d01324cc061b7151759b271662e0c5`
- VGPR797 /SGPR56 /LDS327680 B /private0 /spills0；GEMM2 ELF 保持。
- 实际 metadata AST 共 82206 项等价检查；62276 项 scope 外读取完全一致；19606 项双值读取边界检查通过。
- E64/seed0 的 27 项逐字节检查及完整 MoE 输出均零差异。
- 40 成员 preparation archive SHA256：`ef058278210b1dfa042a56969ef4104f6c6400d3090e04f9bcd1303c49fc887c`。
- 本地 preparation acceptance SHA256：`99c30db231f9f5365916b7fffea4b4e2945751c7cf728aa49c8d0ca0e2a61da7`，已回传核对。

正向 `reboot20261023_mma2_first1_bitsearch_c25_dsf5_lastpair_forward_v1` 的前 DS5 control
为 81.3607 /81.3798 /81.9196 μs，中值 81.3798 μs，GEMM2 中值 64.3310 μs。
候选 try1 的第三轮 post/settled 检查不合格，整组三轮排除，中值 null；
不拼前两轮、不依据耗时决定重试。原 waiter 自动每 6 秒等待全机空闲，外部进程未杀死。

等待期间曾暂停唯一的本任务 host runner（PID1285824），确认其唯一 child 为 idle waiter，
没有计时 child；在 parent 停止且 timed candidate source 不变的条件下，完成 lastquad 的 CPU preflight。
恢复 helper 在编译结束后重新运行三次连续全机空闲检查，然后 SIGCONT；
恢复记录为 `2026-10-23T15:15:06Z`。暂停、恢复、fresh idle 与 CPU preflight 证据一起保留，
没有在性能计时期间编译、传输或额外检查 GPU。

lastquad 仅将最后两步的四个相邻 endpoint 合并，+14/-3 行，要求原 guard 且 experts 可被4整除；
其他配置保留原算法。82206 项实际 AST 对照、62592 项 scope 外精确检查、19326 项四值边界检查通过。
CPU preflight source 为 `21f8b706c39db0588e285321e3cc6100cecebdd21208ef4c5fc6eae82c9a092c`，
ELF 为 `bf47990d73b7a5759cadfe4013b17e7edbe3623332dbedbdde18315909a00dfe`。
实际 ISA 为 5 条 `s_load_b32` + 1 条 `s_load_b128`/6 条 KMcnt waits；
资源及数学/DS/TDM/必要 fences signature 保持，6313 条 executable instructions。
这只是 CPU preflight，尚未正式 device 验证或性能计时，不视为已经通过候选验收。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_dsf5_lastpair_lookup_v1.py \
  run lastpair reboot20261023_mma2_first1_bitsearch_c25_dsf5_lastpair_forward_v1 base candidate base
# 此 label 已使用；独立复现须用新的 label，避免覆盖已有证据。
```

先完成 lastpair。正向双侧均改善才做反向；四项相邻均改善再补3项 strict 和8项 scope。
若不采用，恢复正式 DS5 后再运行 lastquad 的正式 cpu/native/strict preparation。
若采用，lastquad 必须重新绑定新 baseline source/ELF，不套用旧基准结果。
E65 保持原 t64/w1x4 fallback；本轮不把已有 E65 精度通过说成共用 compiled GEMM1
或固定 shared expert1 routing 已完成。

### endpoint 候选补充非 2 的幂 expert 数检查（2026-10-23）

`check_dsf5_endpoint_nonpow2_v1.py` 对实际 baseline/lastpair/lastquad/lastoct 的 metadata AST
覆盖全部 E1～E257、6种 endpoint 分布、scope 开关及首尾/空表/padding/重复 endpoint 边界。
283802 个输入组合、851406 项候选对原逻辑及独立 `bisect_right` reference 对照全部通过。

| 候选 | scope 外读取完全一致检查 | 合并读取边界检查 | 其中非 2 的幂 expert 数检查 |
|---|---:|---:|---:|
| lastpair | 213090 | 68408 | 66199 |
| lastquad | 248181 | 34469 | 32287 |
| lastoct | 265691 | 17535 | 15389 |

补充 lastoct 只在原 guard 且 expert 数可被8整除时，将最后三次比较及最终 endpoint
所需的8个相邻 int32 合并读取，+20/-3 行。其他 expert 数或 scope 外保留原算法，
source 为 `65a2a07e14a1c287c5ba506a70ff7e60026e8728b57e50d5983884c715998a4a`。
此候选目前仅在本地准备，未上传、编译、device 验证或计时，不提前采用。
CDNA5 ISA §15.6 p296 记载 `S_LOAD_B256`，§5.7.1.1 p55/§8.2 p110 仍要求正确使用 KMcnt；
是否生成该 scalar load 及实际性能取决于后续编译产物和原入口对照。
本节不改变 lastpair→lastquad 的现有验证顺序，不改生产源码、benchmark 或计时。

### 两次 steady iteration 展开：仅本地静态候选（2026-10-23）

`build_dsf5_steady_pair_v1.py` 基于 DS5，仅在既有 `standard_a_timeout` scope 且
`inline_zero_acc` 生效时展开相邻两次 `steady_mid_iteration`，尾部不足两次时精确保留原 tile。
首个 K256 的 inline-zero 调用、`steady_mid_iteration` 函数本体、buffer/fence/TDM
参数、K tile 数及所有 tile 次序保持；其他配置继续原 rolled loop。
新增11行/删除2行，source 为 `7a43304906d0d4d48bf40750076d74eec46c1cd11d2467757e8933cea269d544`。

实际 `steady_mid` AST 在73728项有效非空组合及144项空循环组合，共73872项下通过调用序列检查，
核验10104168条 tile/initialization/cluster sync events。没有修改生产源码、metadata、wave布局或 pipeline。
该候选尚未上传、编译、device 验证或计时，没有性能结论。
loop 展开会改变静态 executable counts，不能套用要求静态 counts 完全相同的现有检查器。
后续需要核对真实 compiled CFG、动态 WMMA/DS/TDM 数及 fence 位置，再做完整 MoE 精度和性能对照；
本节不把源码 call sequence 相同说成硬件 fences 已通过验证，也不绕过原验收门槛。
它排在当前 endpoint 候选之后，不中断正在进行的原入口性能队列。

### lastpair 正向双侧回退，不采用（2026-10-23）

`reboot20261023_mma2_first1_bitsearch_c25_dsf5_lastpair_forward_v1` 完整结束，
`completed=true/source_restored=true`，无 error；本地逐成员核验和原 tables/ownership/source/keys/ELF
独立复算全部通过。生产 source 精确恢复 DS5，SHA256 为
`e53ada2baecb0c8e0185b815052281952213f0c81c5e4413c621baccfb06f93e`。
主机分支仍为 `hyg_dsr1_conc6144`，HEAD346438a75506f77b520d3a9a09ba4985f054e377、boot 未变。
测后 `gpu_users.sh --plain` 为空，没有本任务测试进程残留。

| 位置 | GEMM1 三轮 μs | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---|---:|---:|
| 前 DS5 control try1 | 81.3607 /81.3798 /81.9196 | 81.3798 | 64.3310 |
| lastpair candidate try2 | 81.3333 /82.2379 /82.2823 | 82.2379 | 64.8877 |
| 后 DS5 control try1 | 81.3081 /82.2253 /81.8963 | 81.8963 | 64.6999 |

candidate try1 因第三轮 post/settled ownership 不合格整组排除，中值 null；try2 是首个完整合格组。
没有按耗时重试、拼轮或选取81.3333 μs单轮。候选相对前/后分别慢
0.8581 /0.3416 μs（1.0544386% /0.4171129%），不反向、不追加 followup、不采用。
GEMM2 ELF 全程保持原值，其表中耗时变化不归因于 endpoint 修改。

273 regular members、1123534 B archive 已下载，SHA256 为
`a26eda4643c9536cb2fb9bd2f60fa40d8a101760a7ef76c6d766ef1b7dcd3552`。
原验收为 `passed=true/positive_at_every_adjacent_control=false`，SHA256 为
`b64799a30162500287aae8def1a591da84d282628acb1fd58c409465f0644322`。
附加 interruption 验收通过，确认 CPU preflight 时无计时 child、source 不变，
编译后重新全机空闲才恢复，原计时函数保持。preflight 仍明确不是正式 device/性能验收。

```bash
python3 .codex_tmp/yadai_gemm1_param_20261021/tools/seal_dsf5_lastpair_interrupted_v1.py \
  reboot20261023_mma2_first1_bitsearch_c25_dsf5_lastpair_forward_v1
python .codex_tmp/yadai_gemm1_param_20261021/tools/verify_downloaded_original_screen_archive_v1.py \
  reboot20261023_mma2_first1_bitsearch_c25_dsf5_lastpair_forward_v1
python .codex_tmp/yadai_gemm1_param_20261021/tools/screen_dsf5_lastpair_lookup_v1.py \
  accept reboot20261023_mma2_first1_bitsearch_c25_dsf5_lastpair_forward_v1
python .codex_tmp/yadai_gemm1_param_20261021/tools/accept_dsf5_lastpair_interrupted_v1.py \
  reboot20261023_mma2_first1_bitsearch_c25_dsf5_lastpair_forward_v1
```

正式保留版仍为 DS5 的81.6350 μs和81.8183 /81.7086 μs确认组；尚未达到80 μs。
接下来基于恢复后的同一 DS5，完成 lastquad 的正式 cpu/native/strict preparation，
全部通过并本地独立验收后，再做原入口性能对照。

### 最终性能目标修订为争取70 μs（2026-10-23）

用户将目标规模 GEMM1 最终性能目标由80 μs修改为争取70 μs。
目标规模仍为 E64/T1536/topk8/model_dim7168/inter_dim2048、random non-balanced routing，
沿用本分支原 `--scenario kernel` 命令，A preshuffle1、pin reg0、warmup5/iters20。
当前 DS5 的81.6350 μs原入口中值距离70 μs差11.6350 μs，需要相对当前中值降低约14.2525%。

保持原 kernel 的功能、shape限制和精度，GEMM2及其他kernel保持；不修改计时或精度门限。
不新增 persistent、w2x4 或 OPTX pipeline，继续以独立小feature和原入口对照优化。
先完成在手 lastquad 正式验证，再按原正反向对照及strict/scope门槛筛选；
新目标不改变已有失败候选的淘汰结论，也不改写历史80 μs目标下的结果。

### lastquad 正式准备及本地独立核验通过（2026-10-23）

`mma2_first1_bitsearch_c25_dsf5_lastquad_v1` 已完成正式 CPU/native/E64 seed0 strict 准备，
随后下载49个 regular members、555761 B归档，逐成员核对大小和SHA256，并独立复算通过。
候选仅在原guard且expert数可被4整除时合并末尾4个endpoint读取，+14/-3行；
其他配置保留原算法，不新增shape限制、persistent、wave布局、pipeline或env开关。

候选source为`21f8b706c39db0588e285321e3cc6100cecebdd21208ef4c5fc6eae82c9a092c`，
GEMM1 ELF为`bf47990d73b7a5759cadfe4013b17e7edbe3623332dbedbdde18315909a00dfe`。
VGPR797/SGPR56/LDS327680 B，private和spill均为0，与DS5资源相同。
ISA前缀为5次`s_load_b32`加1次`s_load_b128`、6个wait；static executable counts为6313，DS5为6308。
protected DS/TDM counts及offsets、ordered fences、WMMA/BF16/FP4/tanh counts检查通过，
但这不是完整SCHED_MODE2 hardware hazard proof；编译器wait和原fences保留。

27项BYTE_CHECK及完整MoE输出22020096 B均为零字节差异，动态replay通过，GEMM2 ELF保持。
CPU流程18.247947 s、strict流程36.045325 s均为准备耗时，不是kernel性能数据。
归档SHA256为`fbe31fdf82ef2be13d008b6219d40d6f1eea93f3b3192b018ddfa4bad03488b5`，
本地preparation验收SHA256为`6ae8df92bfb0ba284c6b02adbf68300fc37498934f787f229866cd3156352eae`。

```bash
python .codex_tmp/yadai_gemm1_param_20261021/tools/verify_downloaded_preparation_archive_v1.py \
  mma2_first1_bitsearch_c25_dsf5_lastquad_v1
python .codex_tmp/yadai_gemm1_param_20261021/tools/accept_dsf5_lastquad_preparation_v1.py lastquad
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_dsf5_lastquad_lookup_v1.py \
  run lastquad reboot20261023_mma2_first1_bitsearch_c25_dsf5_lastquad_forward_v1 \
  base candidate base
```

本地及远端生产source仍为DS5。接下来使用新label执行原入口前control/候选/后control三轮完整组，
以空闲检查和实际ELF绑定为准。正向双侧改善才反向，四项相邻改善及追加strict/scope通过才采用。
本阶段尚无lastquad的有效性能结果，当前70 μs目标及81.6350 μs保留版成绩不变。

### lastquad 原入口对照方向不一致，淘汰（2026-10-23）

`reboot20261023_mma2_first1_bitsearch_c25_dsf5_lastquad_forward_v1` 完整结束。
原benchmark、A preshuffle1/pin0、precision gate及warmup5/iters20保持；
所有采用组为首个ownership合格的完整三轮，候选try1因首轮post出现`<已退出>` PID整组排除。

| 位置 | GEMM1 三轮 μs | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---|---:|---:|
| 前 DS5 control | 81.4391 /81.3207 /81.6918 | 81.4391 | 64.4703 |
| lastquad try2 | 81.4452 /81.6582 /81.5819 | 81.5819 | 64.6832 |
| 后 DS5 control | 81.5927 /81.8375 /81.8163 | 81.8163 | 64.5293 |

候选比前control慢0.1428 μs（0.1753457%），比后control快0.2344 μs（0.2864955%）。
未满足正向双侧改善，不反向、不追加followup、不采用。
原reference通过，`logits_diff=3.61055e-06`；GEMM2实际ELF保持，其表中耗时变化不归因于本项修改。

260个regular members、1070157 B归档已下载并逐成员核验，原tables/ownership/source/keys/ELF独立复算通过。
归档SHA256为`b7c379258a98ac8eed4140af5903cf13d823adf78116543c13b3a0c3407550d5`，
本地验收SHA256为`0029614ed7b4dd5c5c744e4896af7caf280063224c28bc37545b8589c853b606`。
`completed=true/source_restored=true`，无error；测后`gpu_users.sh --plain`为空，无本任务GPU进程残留。
生产source仍为DS5的`e53ada2b...6f93e`，正式保留成绩81.6350 μs，当前目标为争取70 μs。

后续lastoct所需16份私有source/plan/工具及已有CPU证明已上传并逐SHA256校验一致，
先做限时cpu/native/strict准备，全部通过并本地核验后才使用原入口计时；此时尚无lastoct性能结果。

### lastoct 准备通过，进入原入口性能对照（2026-10-24）

`mma2_first1_bitsearch_c25_dsf5_lastoct_v1` source为`65a2a07e14a1c287c5ba506a70ff7e60026e8728b57e50d5983884c715998a4a`，
实际GEMM1 ELF为`e43a758e89cd43131aa9e9053ee6b8d8c916d66cc2733c8381f8f0f44948939c`。
ISA生成4次`s_load_b32`及1次`s_load_b256`；静态executable counts为6314，DS5为6308。
VGPR797/SGPR56/LDS327680 B，private/spills0；protected DS/TDM counts及offsets、ordered fences和数学操作counts通过。

27项BYTE_CHECK、动态replay及完整MoE22020096 B均零字节差异，GEMM2 ELF保持。
strict流程35.747466 s是验证流程时间，不是kernel性能数据。
43个regular members、404761 B归档已下载并逐成员核验，本地独立preparation验收通过。
归档SHA256为`fc693495c045ffd3498a27622817f9bcfcf60a0e88023976bf36665c0d6d2fbb`，
preparation验收SHA256为`670b87f65c322b7ec82026f08581a15acad7442f34385f7f37ebdf3982d724dd`。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_dsf5_lastoct_lookup_v1.py \
  run lastoct reboot20261023_mma2_first1_bitsearch_c25_dsf5_lastoct_forward_v1 \
  base candidate base
```

仍以当前DS5为control，先等待全机空闲再执行各位置完整三轮；不在计时期间编译、传输或并发检查GPU。
正向双侧改善才反向，四项相邻改善及追加strict/scope通过才采用；准备通过不表示已取得性能收益。
正式保留版仍为DS5，当前目标争取70 μs。当前DS5 ISA的`s_wait_alu`事实核对见GEMM1分析文档，静态共170处。

### lastoct 正向双侧小幅改善，继续反向确认（2026-10-24）

`reboot20261023_mma2_first1_bitsearch_c25_dsf5_lastoct_forward_v1` 完整结束；
三个位置均首个完整三轮ownership合格，没有排除组，没有按耗时重试。

| 位置 | GEMM1 三轮 μs | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---|---:|---:|
| 前 DS5 control | 82.0691 /82.1536 /82.2507 | 82.1536 | 64.6537 |
| lastoct candidate | 82.2042 /81.8774 /81.7258 | 81.8774 | 64.7781 |
| 后 DS5 control | 82.0503 /82.7777 /81.6371 | 82.0503 | 64.0759 |

候选相对两侧快0.2762 /0.1729 μs（0.3361995% /0.2107244%）。
这是本轮相邻控制下的候选收益，不能与DS5此前81.6350 μs跨时段比较；尚未采用，也尚未达到70 μs。
原precision通过，GEMM2实际ELF保持。`completed=true/source_restored=true`，无error；测后全机空闲。

218个regular members、798350 B归档已下载并逐成员核验，原tables/ownership/source/keys/ELF独立复算通过。
归档SHA256为`1be4a6db4b38c68fa4636e5ba4f3c5a5beb5b92700413967ffc75c0693b2c373`，
本地验收SHA256为`7ecf910f707964d38677d36a94659205c03114eb962f1f89dab308fe8b32c18a`。
生产仍为DS5，下一步使用下列新label反向完整对照；四项相邻改善后才做追加strict/scope并考虑采用。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_dsf5_lastoct_lookup_v1.py \
  run lastoct reboot20261023_mma2_first1_bitsearch_c25_dsf5_lastoct_reverse_v1 \
  candidate base candidate
```

### DS5 pin1 基线与单指令 ISA 实验（2026-10-24）

用户允许开启 `AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=1`，有效方案最终仍须落实到 FlyDSL 源码。
生产文件仍为 DS5，SHA256 `e53ada2baecb0c8e0185b815052281952213f0c81c5e4413c621baccfb06f93e`；
正式保留的 pin0 成绩仍为 81.6350 μs，目标争取低于 70 μs。本节没有采用注入 cache 作为生产实现。

容器 `dsr1_ep4`，FlyDSL `0.3.3.dev882`，远端目录
`/data/yanguahe/code/wk_sp1/aiter_a4w4_prefill_v2_yadai`，boot ID
`cb28eed4-51ca-4cdc-bf5b-ecbc1d54db5f`。保持 E64/T1536/topk8、7168/2048、uniform/auto、
seed0、non-balanced routing、A preshuffle1、warmup5/iters20 和原 precision gate。
每次原入口运行前检查全机空闲，测后检查 KFD；本节各组均为首个 ownership 合格的完整三轮。
每个编译或测试 child 使用 `timeout --signal=TERM --kill-after=5s 120s`。

原版 pin1 基线 label 为 `reboot20261024_dsf5_pin1_baseline_v1`，
GEMM1 为 86.6131 / 86.1428 / 86.1006 μs，中值 86.1428 μs，GEMM2 中值 61.1725 μs。
原 precision 通过，`logits_diff=3.61055e-06`，`rel_l2=0.00268721`。
pin1 实际 GEMM1 ELF 为 `ee76c901ebf710caaeeb99952d796b246a182536b3e59ab4a6f41fe20ff64a3b`，
GEMM2 ELF 为 `a5e9c2504863d38c85e39acdc3e28b35774a99f21957b517e338953558375311`。
pin0/pin1 的历史中值不是同一组交错对照，不能单独归因于 register allocation：
生产中的 MMA2/first1 等既有小优化也排除了 pin1，后续另设优化条件一致的候选。

ISA 工具来自 FlyDSL commit `94485ffdc3e57fa79cc9e0979369a8128becbf17` 的
`tools/inject_asm/inject_asm.py`。dev882 一个 entry 含两个字节相同、target 不同的 Binary ELF；
实验适配保留两个 target，只在原 ELF 完全一致时同时替换。
原 `.s` 有 21 条 TDM 的重组装编码不一致，改写为原 machine words 后，
`original_exact.s` 与完整原 ELF 重组装逐字节一致，再开展实验。
适配只在 `.codex_tmp/`，没有修改安装的编译器。

首个 ISA 实验只增加 `s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1`，
原 WMMA、TDM、wait、barrier 和输出操作均保留。候选 GEMM1 ELF 为
`1b403c68b76ac0d201781ab9091ce7de6a694788f0edc357e023f0d742980f7c`。
CDNA5 ISA §5.7.2.1（文档 p56）将 bit2 定义为 `DISABLE_XDL_ARB_STALL`，允许连续 issue WMMA，
同时可能影响其他 wave 的 coexecution；实际收益以原入口测试为准。

| 位置 | GEMM1 三轮 μs | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---|---:|---:|
| 前 pin1 原 ELF | 86.7332 / 85.7927 / 86.4148 | 86.4148 | 61.6051 |
| bit2 ISA 候选 | 84.4392 / 83.5305 / 83.9922 | 83.9922 | 60.8058 |
| 后 pin1 原 ELF | 86.4191 / 86.4549 / 86.7417 | 86.4549 | 61.3604 |

候选相对两侧改善 2.4226 / 2.4627 μs（2.80345% / 2.84854%）。九轮原 precision gate 全通过，
GEMM2 逐次 audit 都绑定同一 pin1 ELF，表中 GEMM2 耗时变化不归因于本项 GEMM1 修改。
每次 audit 使用真实原入口的 source-dependent call key，没有修改计时函数。
候选仍慢于正式保留版，此处仅确认一个可组合的实验方向，不更新最优成绩。

实际执行命令：

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_xdl_asm_experiment_v1.py
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_pin1_isa_pair_v1.py \
  reboot20261024_dsf5_pin1_xdl_asm_forward_v1
```

结果为 `.codex_tmp/yadai_gemm1_param_20261021/logs/reboot20261024_dsf5_pin1_xdl_asm_forward_v1.summary.json`。
工具拒绝覆盖已有结果文件，再次实验须使用新的 label 与独立计划。
真实源码 bit2 候选仍需后续对照。ScaleA/ScaleB 固定低 VGPR 候选的最终结果见下节。

### ScaleA/ScaleB 低 VGPR 固定槽：精度通过，性能无稳定收益（2026-10-24）

候选 `dsf5_pin1_scale_slots_v1` 仅在目标 pin1 GEMM1 固定两组 scale 寄存器：
slot0 的 ScaleA/ScaleB 为 `v128:131 / v132:135`，slot1 为 `v136:139 / v140:143`。
A/B 的 `v256:511`、C 的 `v512:1023`、LDS、计算、输入输出和同步保持。
source SHA256 为 `96d992af2bdf122c68c8a7a318a2eae90da159a434ed6f26d36897eb9d0e7155`，
真实原入口 GEMM1 ELF 为 `a70d64fa12da54e89848ca121f2b70f26e72d28f84490a4b98a778369fd0b795`。
VGPR1024/SGPR66/LDS327680 B，private/spills 为 0。
静态 executable 指令为 5416，`s_wait_alu` 为 154，`s_set_vgpr_msb` 为 456；
少于 pin1 原版的 5423/156/459，但数量下降没有转化为稳定收益。
统计包含全部 machine-address 行并排除 `s_code_end` padding，不能用旧工具漏掉 `buffer_store_b8` 的 total。

完整 MoE 输出 `[1536,7168]` BF16 共 22,020,096 B，与原版零字节差异；
SHA256 为 `443e8c5959dc9513a655d5c9b59b21c93c738df5807cda3123560fbc8e47b935`。
GEMM2 实际 ELF 保持 pin1 的 `a5e9c2504863d38c85e39acdc3e28b35774a99f21957b517e338953558375311`。

| 位置 | GEMM1 三轮 μs | GEMM1 中值 μs |
|---|---|---:|
| 前 pin1 control | 86.8533 / 87.1126 / 86.6067 | 86.8533 |
| scale slots 候选 | 86.6888 / 85.7146 / 86.4401 | 86.4401 |
| 后 pin1 control | 87.1167 / 85.9678 / 85.6895 | 85.9678 |

候选比前 control 快 0.4132 μs，比后 control 慢 0.4723 μs，未通过双侧改善。
九轮精度、cache/ELF 绑定及 ownership 均通过；`completed=true/source_restored=true`，无残留。
本方向不反向、不采用，正式保留版仍为 DS5/pin0。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_pin_flydsl_v1.py \
  run dsf5_pin1_scale_slots_v1 reboot20261024_dsf5_pin1_scale_slots_prepare_v1 \
  reboot20261024_dsf5_pin1_scale_slots_forward_v1 base candidate base
```

结果：`.codex_tmp/yadai_gemm1_param_20261021/logs/reboot20261024_dsf5_pin1_scale_slots_forward_v1.summary.json`。

### pin1 启用已有 DS5 小优化：准备通过，尚未采用（2026-10-24）

`dsf5_pin1_feature_parity_v1` 改六处 guard，让目标 pin1 GEMM1 使用已有的 MMA2/first1、
swizzle/lookup/fence、inline zero、DS_B4 和 scheduling bit2；原 pin0 行为保持。
没有新增 persistent、pipeline 或 wave layout。
source SHA256 为 `2822a279b0f2bc078fefbda2004f61ef4a304d1391413125cc7cfa3eff6f6bee`，
GEMM1 ELF 为 `71ef64dd038b26ce18d883166cc96df28a179d9ccf5db296c350e095ee1cab2b`。
VGPR1024/SGPR62/LDS327680 B，private/spills 为 0。
真实原入口的 ELF 与 CPU 编译一致，GEMM2 ELF 保持；完整 MoE 22,020,096 B 与原 pin1 零字节差异。
preparation、真实 source-dependent keys 及源码恢复结果已下载并在本地复核。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_pin_flydsl_v2.py \
  run dsf5_pin1_feature_parity_v1 reboot20261024_dsf5_pin1_feature_parity_prepare_v1 \
  reboot20261024_dsf5_pin1_feature_parity_candidate_screen_v1 candidate
```

单组三轮用于初筛，不能代替相邻 control 对照；有竞争力后再完整对照。

初筛三轮 GEMM1 为 82.9272 / 81.4724 / 81.4745 μs，中值 81.4745 μs，
GEMM2 中值 61.4362 μs。随后原 DS5/pin0 重测为 82.5247 / 82.0713 / 82.2843 μs，
GEMM1 中值 82.2843 μs，GEMM2 中值 64.3965 μs。
两组全部通过原 precision gate、真实 key/ELF audit 及 ownership 检查；原表已下载并在本地重算，
没有因耗时选择或重试组。此处仍是连续初筛和重测，尚未完成双侧交错对照，不作为新的正式最优成绩。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/retest_dsf5_pin0_retained_v1.py \
  reboot20261024_dsf5_pin0_retest_v1
```

下一批只调整目标 pin1 GEMM1 的 VGPR 编号。首个 compact 候选将输入 slot0 固定在 `v0:127`，
CPU 编译被 `LLVM ERROR: explicit register conflicts with precolored machine operand: VGPR0` 拒绝，
没有 GPU launch 或性能数据。后续改用 `v128:255`，避开该 precolored operand。
另一个 rotated 候选将 C 固定为 `v256:767`，A/B 两槽为 `v768:895 / v896:1023`；
已通过真实原入口 ELF、完整 MoE 零字节差异及 GEMM2 ELF 保持检查，正在初筛。
其 VGPR1024/SGPR62/LDS327680 B，private/spills 为 0；
static executable 为 5836，`s_wait_alu` 为 194，`s_set_vgpr_msb` 为 770。
寄存器编号与 static counts 只用于说明实验改动，实际收益以交错对照为准。

### VGPR 编号初筛与对照脚本修正（2026-10-24）

rotated 三轮 GEMM1 为 80.8063 / 80.9960 / 81.7655 μs，中值 80.9960 μs，
GEMM2 中值 61.2528 μs，三轮全部通过精度、ELF audit 和 ownership 检查。
这是初筛成绩，尚未作为正式保留版，也尚未达到 70 μs。

compact128 的 C 为 `v256:767`，A/B 两槽为 `v128:255 / v768:895`。
source SHA256 为 `5e280bfa65f4027db759be0e5fb6e0b29c795006dae7f43e5fb419619d240e4b`，
真实 GEMM1 ELF 为 `be878c1341d981a5944dcf0f6dd45b59905b202697a5fc79f8006c9edc6d82f5`。
VGPR896/SGPR62/LDS327680 B，private/spills 为 0；static executable 为 5828，
`s_wait_alu` 为 194，`s_set_vgpr_msb` 为 762。
完整 MoE 与原 DS5/pin0 的 22,020,096 B 输出零字节差异，原 precision gate 通过。

首个 compact128 screen 的 audit 因 pin0/pin1 编译 hint 差异退出，没有完成三轮，
label `reboot20261024_dsf5_pin1_compact128_vgpr_candidate_screen_v1` 的计时不计入成绩。
源码自动恢复，未留下 GPU 进程。脚本误要求含 hint 的 argument signature 完全一致，
实际两个版本仅多出 pin1 原有的 `maxnreg=1024`、`waves_per_eu=1`。
新 audit 仅在 `_hints_` 范围内允许这两个固定项，仍严格检查其他调用参数、
每个环境真实 source-dependent key、原 source globals 和两个实际 ELF；未修改原计时入口或精度标准。
使用新的 label 从头执行三轮，没有复用失败轮的计时。

修正后 compact128 三轮 GEMM1 为 81.0065 / 81.7319 / 81.2195 μs，中值 81.2195 μs，
GEMM2 中值 61.3414 μs。三轮均通过原 precision、实际 key/ELF audit 和 ownership；
原表已下载并在本地重算。暂未优于 rotated 初筛，不采用。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_pin_flydsl_v3.py \
  run dsf5_pin1_compact128_vgpr_v1 reboot20261024_dsf5_pin1_compact128_vgpr_pin0_prepare_v1 \
  reboot20261024_dsf5_pin1_compact128_vgpr_candidate_screen_v2 candidate --base-pin0
```

`--base-pin0` 对照时，pin0 的 GEMM2 绑定原 ELF `4bd626e3...16be`，pin1 的 GEMM2 绑定
原 pin1 ELF `a5e9c250...5311`。两套 ELF 差异来自用户允许的全局 pin 环境开关，
各候选均未改变相同 pin 环境下的 GEMM2 实现或 ELF，不能把跨环境的 GEMM2 耗时变化归因于 GEMM1 修改。

### 另外三种 VGPR 布局初筛（2026-10-24）

以下候选基于 pin1 parity，仅改变 C 和 A/B 的固定 VGPR 编号，保留 compiler hazard waits。
SGPR 均为 62，LDS 均为 327680 B，private 与 spills 均为 0。

| 候选 | C 编号 | 两个 A/B 槽 | VGPR | static executable / wait / MSB | GEMM1 三轮 μs | 中值 μs |
|---|---|---|---:|---|---|---:|
| linear | `v128:639` | `v640:767 /v768:895` | 896 | 5755 /194 /699 | 82.3159 /82.9588 /80.9012 | 82.3159 |
| packed32 | `v32:543` | `v544:671 /v672:799` | 800 | 5861 /194 /812 | 81.3081 /81.7319 /82.3074 | 81.7319 |
| packed64 | `v64:575` | `v576:703 /v704:831` | 832 | 5781 /194 /728 | 81.7951 /81.6033 /81.0066 | 81.6033 |

每个候选均为 attempt=1，没有按耗时选择或重试。原表、真实 source-dependent key、实际 ELF、
manifest、precision 和测试前后 ownership 均通过本地复核，证据保存在
`.codex_tmp/yadai_gemm1_param_20261021/logs/reboot20261024_pin1_vgpr_layout_screen_local_review_v1.json`。
完整 MoE 的 22,020,096 B BF16 输出与 DS5/pin0 零字节差异，原 precision gate 通过。
三者中值均未优于 rotated 初筛的 80.9960 μs，不采用。
static MSB 最少的 linear 没有带来最好的实测性能，数量下降不能直接解释为等待成本下降。

复现时将 `FEATURE` 设为 `dsf5_pin1_linear_vgpr_v1`、`dsf5_pin1_packed32_vgpr_v1`
或 `dsf5_pin1_packed64_vgpr_v1`；`NEW_LABEL` 必须使用未占用的 label，工具拒绝覆盖结果。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_pin_flydsl_v3.py \
  run "$FEATURE" "reboot20261024_${FEATURE}_pin0_prepare_v1" \
  "$NEW_LABEL" candidate --base-pin0
```

生产 source SHA256 保持为 `e53ada2baecb0c8e0185b815052281952213f0c81c5e4413c621baccfb06f93e`。
上述候选没有落入生产源码。

### rotated 与 DS5/pin0 的正向交错对照（2026-10-24）

对照使用 source `e53ada2b...6f93e` 的 DS5/pin0 保留版。rotated/pin1 实际 GEMM1 ELF 为
`b5d3fc63fe64e90f170f308021ccf7f9ee8eb77f4b70049e60c5c5137b89b4ed`。

| 位置 | GEMM1 三轮 μs | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---|---:|---:|
| 前 DS5/pin0 | 82.9736 /81.8415 /82.4214 | 82.4214 | 64.2087 |
| rotated/pin1 | 81.2426 /82.0587 /81.7149 | 81.7149 | 61.0968 |
| 后 DS5/pin0 | 81.9934 /81.4072 /82.1896 | 81.9934 | 64.9257 |

rotated 相对两侧 GEMM1 中值改善 0.7065 /0.2785 μs，即 0.857180% /0.339661%。
三组 attempt 均为 1，九轮 precision、真实 key/ELF audit、测试前后 ownership 全通过。
原表、manifest 与 summary 已下载并本地复核，`local_review_passed=true`。
收益较小，须结合下面的反向对照判断。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_pin_flydsl_v3.py \
  run dsf5_pin1_rotated_vgpr_v1 reboot20261024_dsf5_pin1_rotated_vgpr_pin0_prepare_v2 \
  reboot20261024_dsf5_pin1_rotated_vgpr_pin0_forward_v1 base candidate base --base-pin0
```

完整 MoE 输出与 pin0 零字节差异；各 pin 环境的 GEMM2 分别绑定原 ELF。
跨 pin 环境的 GEMM2 性能变化不归因于本项 GEMM1 修改。
源码自动恢复；证据为 `logs/reboot20261024_dsf5_pin1_rotated_vgpr_pin0_forward_v1.summary.json`
及同名 `.local_review_v1.json`，均位于 `.codex_tmp/yadai_gemm1_param_20261021/`。

### rotated 反向对照：收益不稳定，暂不采用（2026-10-24）

| 位置 | GEMM1 三轮 μs | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---|---:|---:|
| 前 rotated/pin1 | 80.7386 /81.0551 /80.6567 | 80.7386 | 61.5753 |
| DS5/pin0 | 82.7946 /81.4957 /82.0544 | 82.0544 | 64.2658 |
| 后 rotated/pin1 | 82.0355 /81.0761 /83.0265 | 82.0355 | 61.5143 |

前后 rotated 相对 pin0 分别改善 1.3158 /0.0189 μs，即 1.603570% /0.023033%。
最后一组基本持平，因此不能将初筛 80.9960 μs 作为稳定的正式最优成绩。
本轮正反向共十八轮均为 attempt=1，原 precision gate、实际 ELF 和 ownership 检查全通过，
源码恢复。反向原表和 manifest 已下载，`review_pin_result_local_v1.py` 复核通过。
生产仍保留 DS5/pin0；本轮其独立重测中值为 82.2843 μs。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_pin_flydsl_v3.py \
  run dsf5_pin1_rotated_vgpr_v1 reboot20261024_dsf5_pin1_rotated_vgpr_pin0_prepare_v2 \
  reboot20261024_dsf5_pin1_rotated_vgpr_pin0_reverse_v1 candidate base candidate --base-pin0
```

反向证据为 `logs/reboot20261024_dsf5_pin1_rotated_vgpr_pin0_reverse_v1.summary.json`
及同名 `.local_review_v1.json`。再次复现须换新 label。

### 固定 LDS 地址 VGPR 的两个候选：CPU 编译失败（2026-10-24）

基于 rotated，给当前/下一 LDS stage 的 A/B/ScaleA/ScaleB 地址分别配置固定槽，
两个候选使用 `v32:39` 与 `v64:71`，保留原计算和 waits。
两者均被 compiler 拒绝：`LLVM ERROR: overlapping live intervals in explicit register storage`。
CPU child 分别运行 17.4952 /17.5445 秒，returncode 均为 134，没有 GPU launch 或性能数据。

`dsf5_pin1_rotated_addr32_v1` 的 source SHA256 为
`7b59b8d99688882cc10658da5714f89bd7c80a37b76c040c64ea1127939d4f36`；
`dsf5_pin1_rotated_addr64_v1` 为
`d38682051f003fae9623416e35d01c39d3b5c54774ab30fed9b0f224f914f7f7`。
结果在 `logs/reboot20261024_pin1_rotated_lds_address_cpu_batch_v1.summary.json`。
生产源码未修改。这是当前 pin 写法未通过编译校验，不能据此判断 GPU 精度或性能。

### 固定地址的 scalar / opaque 写法：仍未通过 CPU 编译（2026-10-24）

将四 DWORD 地址槽改成独立 scalar 槽，或加入 tied empty asm，均未解决同一个
`LLVM ERROR: overlapping live intervals in explicit register storage`。
两项 returncode 均为 134，没有 GPU launch，不记录性能。

| 候选 | CPU child 秒 | source SHA256 |
|---|---:|---|
| `dsf5_pin1_rotated_addr_scalar32_v1` | 17.5486 | `a48de5858f9e980c1535d19fbdb4a63eebb763b88369fffc0684b2364ff86b05` |
| `dsf5_pin1_rotated_addr_opaque32_v1` | 17.5316 | `d9313fb8f09e559186963de89413f90583b72a6eb29771dc55c3dde50a3764c9` |

证据为 `logs/reboot20261024_pin1_rotated_lds_scalar_cpu_batch_v1.summary.json`
及两个候选的 `.exec.log`，位于 `.codex_tmp/yadai_gemm1_param_20261021/`。
四个直接地址 pin 写法至此均未通过编译校验，暂时搁置，不重复运行。

### LDS 地址 keepalive：两项初筛未证明稳定收益（2026-10-24）

基于 rotated，仅在目标 pin1 GEMM1 的 `load_lds_data()` 末尾增加 empty side-effect asm，
分别消费 A/B/ScaleA/ScaleB 地址或只消费 ScaleA/ScaleB 地址，尝试延长地址 live interval，
避免 compiler 将地址与 DS 目的寄存器复用。没有新增固定地址槽，保留原计算及源码 waits。

| 候选 | GEMM1 三轮 μs | GEMM1 中值 μs | GEMM2 中值 μs | static executable / wait / MSB |
|---|---|---:|---:|---|
| `dsf5_pin1_rotated_addr_keepalive_v1` | 82.5987 /82.2886 /82.8369 | 82.5987 | 61.7904 | 5889 /185 /804 |
| `dsf5_pin1_rotated_scale_addr_keepalive_v1` | 82.4658 /81.9174 /81.3546 | 81.9174 | 61.4343 | 5875 /186 /800 |

两项资源均为 VGPR1024 /SGPR62 /LDS327680 B，private 与 VGPR/SGPR spill 均为 0。
rotated 原候选的 static executable /wait /MSB 为 5836 /194 /770；
wait 数下降的同时，其他静态指令增加，不能仅按 wait 数判断收益。

| 候选 | source SHA256 | 实际 GEMM1 ELF SHA256 |
|---|---|---|
| 全地址 keepalive | `597cf05ba5b18c648eb9ec8758376e2ae902e99401f08f8e1a0b6672a86f6a77` | `38d348359b14bff12dadec70cfe368d1b2c26755fa76e5c14277028e70a52d20` |
| scale 地址 keepalive | `c8829169614178c2eaac15f8a6fcc9493a50518383cb7d964b2ba1236b7d8574` | `d0281cbb1f6e851f703e99cc2085d7624911d31b91f1a74685b6e8d381785a4f` |

完整 MoE `[1536,7168]` BF16 的 22,020,096 B 输出与 DS5/pin0 零字节差异，
SHA256 为 `443e8c5959dc9513a655d5c9b59b21c93c738df5807cda3123560fbc8e47b935`。
原 precision gate 通过，`logits_diff=3.6105513278025183e-06`，
`rel_l2=0.0026872099842876196`。两项各三轮均为 attempt=1，原表、manifest、
真实 source-dependent keys、实际 ELF 和 GPU 空闲/ownership 检查已下载并通过本地复核。
每个 pin 环境的 GEMM2 分别绑定原 ELF，跨 pin 环境的 GEMM2 变化不归因于 GEMM1 修改。

两项都只完成初筛，尚无相邻交错对照，未作为保留版。当前生产仍为 DS5/pin0，
本轮独立重测中值为 82.2843 μs；这些不同时段的初筛数值不能用于确认性能提升。

复现先指定候选，再使用未占用的 `NEW_LABEL`：

```bash
FEATURE=dsf5_pin1_rotated_addr_keepalive_v1
PREP=reboot20261024_dsf5_pin1_rotated_addr_keepalive_pin0_prepare_v1
# 或 FEATURE=dsf5_pin1_rotated_scale_addr_keepalive_v1
#    PREP=reboot20261024_dsf5_pin1_rotated_scale_addr_keepalive_pin0_prepare_v1
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_pin_flydsl_v3.py \
  run "$FEATURE" "$PREP" "$NEW_LABEL" candidate --base-pin0
```

证据为 `logs/reboot20261024_dsf5_pin1_rotated_addr_keepalive_candidate_screen_v1.summary.json`
及 `logs/reboot20261024_dsf5_pin1_rotated_scale_addr_keepalive_candidate_screen_v1.summary.json`，
各自同名 `.local_review_v1.json` 的 `local_review_passed=true`。

### LDS 地址预计算：初筛与正向对照（2026-10-24）

`dsf5_pin1_rotated_addr_precompute_v1` 基于 rotated，新增 11 行，只在目标 pin1 GEMM1 的
`compute_ktile()` 中、`sched_barrier(0)` 前消费当前/下一 stage 的四个 LDS 地址，
尝试阻止 compiler 将地址计算下沉到 DS reads 之前。未删除源码 waits，未改变输入、计算或输出。
source SHA256 为 `95a4a237e0dfd67ef1ee48c5d01e472708d28dcae835699128be1510431c44f0`，
实际 GEMM1 ELF 为 `38ca9f6a027d33365f93809b668976ac377485109e77c6b964d44271aaef181e`。

CPU child 19.3776 秒完成，VGPR1024 /SGPR62 /LDS327680 B，private/spills 均为 0。
static executable /wait /MSB 为 5866 /181 /780，rotated 原候选为 5836 /194 /770。
静态计数已从下载的实际 ISA 本地重算；wait 减少 13 条不等于已证明消除相应动态空泡。
完整 MoE 22,020,096 B 与 DS5/pin0 零字节差异，原 precision gate 通过，
`logits_diff=3.6105513278025183e-06`，`rel_l2=0.0026872099842876196`。

初筛 GEMM1 为 80.8548 /82.5964 /80.8274 μs，中值 80.8548 μs；GEMM2 中值 60.9302 μs。
三轮 attempt=1，原表、实际 key/ELF、manifest、GPU 空闲和 ownership 已通过本地复核。
这组初筛不作为新的正式保留成绩。

| 正向位置 | GEMM1 三轮 μs | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---|---:|---:|
| 前 DS5/pin0 | 81.2302 /81.6921 /81.8457 | 81.6921 | 64.5083 |
| 地址预计算/pin1 | 81.6224 /81.1165 /81.7719 | 81.6224 | 61.5587 |
| 后 DS5/pin0 | 82.3581 /82.1347 /82.1303 | 82.1347 | 64.4534 |

候选相对两侧改善 0.0697 /0.5123 μs，即 0.085320% /0.623732%。
九轮均为 attempt=1，原 precision、实际 ELF 和 ownership 通过，源码恢复，
下载原表与 manifest 后 `review_pin_result_local_v1.py` 本地复核通过。
最小提升很小，仍须反向对照验证；当前暂不采用，生产仍为 DS5/pin0。
每个 pin 环境的 GEMM2 绑定各自原 ELF，不能把这里跨 pin 环境的 GEMM2 差异归因于 GEMM1。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_pin_flydsl_v3.py \
  run dsf5_pin1_rotated_addr_precompute_v1 \
  reboot20261024_dsf5_pin1_rotated_addr_precompute_pin0_prepare_v1 \
  reboot20261024_dsf5_pin1_rotated_addr_precompute_pin0_forward_v1 \
  base candidate base --base-pin0
```

证据为 `logs/reboot20261024_dsf5_pin1_rotated_addr_precompute_candidate_screen_v1.summary.json`
与 `logs/reboot20261024_dsf5_pin1_rotated_addr_precompute_pin0_forward_v1.summary.json`，
各自同名 `.local_review_v1.json` 已生成；重复运行须换未占用的新 label。

### 复用现有 t128/w2x2 配置：ScaleA 兼容与正确性（2026-10-24）

私有候选复用已有 `t128/w2x2/b2`、`b3` mainloop，未新增 persistent、w2x4 或 OPTX pipeline。
仅在原 GEMM1/FP4/SiLU/A preshuffle/pin0/no bias/K7168/t256/w2x2/b4/cluster1x4 scope
内部改变 tile 和 buffer 数；`row_major_ascale`、`tdm_as_in_prologue` 及其他路径继续使用原实现。
输出 `quant_wmma_rep=8` 保持不变，GEMM2 输入的 ScaleA 格式不变。

首版 `dsf5_pin0_t128_b2_v1` 的完整 MoE 正确性失败：`logits_diff=0.21392824938788352`，
`rel_l2=0.6542410254478455`，没有进入性能测试。原因是实验内部改成每 wave M64 后，
旧 ScaleA descriptor/load 把上游仍按 M128 chunk 排列的数据当作 M64 格式解释。
这是 retile 实验引入的兼容错误，不能归因于保留版原有 ScaleA 实现。

v2 保留上游 M128 source chunk：完整读取两个 k128 slice 并平铺到 LDS，
M64 wave base 间隔为 64 DWORD，`load_sa()` 的 k128 slice stride 保持 128 DWORD。
两项 CPU 编译、原入口 key/ELF preparation 和完整 MoE 逐字节对比均通过。

| v2 候选 | VGPR | SGPR | LDS B | private / spills | static executable / wait / MSB |
|---|---:|---:|---:|---|---|
| `dsf5_pin0_t128_b2_scale128_v2` | 455 | 58 | 131072 | 0 / 0 | 3480 /154 /275 |
| `dsf5_pin0_t128_b3_scale128_v2` | 456 | 62 | 163840 | 0 / 0 | 3598 /117 /281 |

| 候选 | source SHA256 | GEMM1 ELF SHA256 |
|---|---|---|
| b2/v2 | `fe5c95521ef02eed4065fb8dce09fd9002097f58816dae2eec0733da8bc9ba18` | `f4b98539c4af6e1d2f126f9327dba95a0af1dee6f2bddfc3b9027b664e26be15` |
| b3/v2 | `b5d0e33c7dfede82f7487087b720c1e18eaf0fc8a15aea5e13dadd83d4f7eef3` | `e928bab6a2b121de35f1667bf01ca9918244e444f448236b21c3bb0bd6cfef78` |

两项 seed0 完整 MoE `[1536,7168]` BF16 的 22,020,096 B 均与 DS5/pin0 零字节差异，
输出 SHA256 为 `443e8c5959dc9513a655d5c9b59b21c93c738df5807cda3123560fbc8e47b935`。
原 precision gate 通过：`logits_diff=3.6105513278025183e-06`，`rel_l2=0.0026872099842876196`。
两项 GEMM2 ELF 均保持 `4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。
preparation 的 summary、source SHA、原 key inventory SHA 和完整输出对比已下载并在本地复核。

硬件依据为 Shader Guide rev65 §3.3.2.1：MI450 每 SIMD 1024 VGPR，wave32 按 16 分配；
WGP LDS 上限为 320 KiB。两项资源具备容纳两条 wave/两组 WG 的容量条件，
但这不是实际 occupancy 或性能提升的证明；当前尚未为 t128 候选采集 ATT。
两项保留在私有 `candidates/`，生产源码仍为 DS5/pin0。

证据位于 `.codex_tmp/yadai_gemm1_param_20261021/logs/`：
`reboot20261024_pin0_t128_scale128_cpu_batch_v2.summary.json`，
以及 `reboot20261024_dsf5_pin0_t128_b{2,3}_scale128_prepare_v2` 的
`.summary.json`、`.preparation.json`、`.original_keys.json`。

### t128/b2、b3 原入口性能初筛：均未采用（2026-10-24）

两项均使用 pin0、A preshuffle=1 和用户原 `--scenario kernel` benchmark，
`warmup=5`、`iters=20`、seed0、uniform/auto、E64/T1536/topk8/7168/2048。
每次运行前完成全机空闲检查；计时阶段不并发编译、传输或 GPU 检查，
每个 child 受 120 秒 timeout 限制。六轮 attempt=1，前后 ownership 检查和原 precision gate 均通过。

| 候选 | GEMM1 三轮 μs | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---|---:|---:|
| `dsf5_pin0_t128_b2_scale128_v2` | 174.909 /174.412 /174.252 | 174.412 | 64.7192 |
| `dsf5_pin0_t128_b3_scale128_v2` | 139.239 /134.814 /132.723 | 134.814 | 64.4767 |

两项明显未接近 DS5/pin0，未进入正式保留版。当前保留版此前正式中值为 81.6350 μs，
本轮独立重测为 82.2843 μs；这些不同时段的成绩不用于构造相邻性能提升百分比。
两项 source-dependent 原 cache key、实际 GEMM1/GEMM2 ELF、原表和 manifest 均已下载并通过本地复核。
GEMM2 ELF 与保留版完全相同，计时波动不归因于 GEMM1 源码修改。
生产源码已经恢复为 `e53ada2baecb0c8e0185b815052281952213f0c81c5e4413c621baccfb06f93e`。

源码检查发现，内部 retile 后既有 t256 优化 guard 不再命中，调度回到通用 MMA10/first10，
未继承 DS5/pin0 的 MMA2/first1 与 DS5 设置；b2 同时使用已有 post-compute issue 路径。
这些是后续实验的可检查因素，尚未用 ATT 证明它们各自占回退的比例。

```bash
# B 取 2 或 3，NEW_LABEL 必须尚未使用。
FEATURE=dsf5_pin0_t128_b${B}_scale128_v2
PREP=reboot20261024_dsf5_pin0_t128_b${B}_scale128_prepare_v2
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_small_tile_flydsl_v1.py \
  run "$FEATURE" "$PREP" "$NEW_LABEL" candidate --base-pin0 --candidate-pin0
```

证据为 `logs/reboot20261024_dsf5_pin0_t128_b{2,3}_scale128_candidate_screen_v1.summary.json`
和各自同名 `.local_review_v1.json`，`local_review_passed=true`。

### t128/b3 恢复小调度设置后仍回退：结束本轮探索（2026-10-24）

在 ScaleA-compatible b3/v2 上增加两行、修改两行，仅针对 `compact_m_tile` 恢复 MMA2/first1
与原 DS5/floor 设置，并分别测试 closing8、closing4。没有改变原 TDM loads、buffer indexing、
memory/barrier waits、输入格式、输出格式或 WMMA 累加顺序，未新增 pipeline 实现。
两项 CPU child 分别为 16.7432 /16.6212 秒，private/spills 均为 0。

| 候选 | VGPR / SGPR / LDS B | static executable / wait / MSB | GEMM1 三轮 μs | 中值 μs | GEMM2 中值 μs |
|---|---|---|---|---:|---:|
| `dsf5_pin0_t128_b3_scale128_mma2_tail8_v3` | 472 /58 /163840 | 3657 /134 /370 | 135.240 /133.655 /133.628 | 133.655 | 64.6262 |
| `dsf5_pin0_t128_b3_scale128_mma2_tail4_v3` | 472 /58 /163840 | 3676 /141 /382 | 135.570 /134.738 /133.094 | 134.738 | 64.9551 |

| 候选 | source SHA256 | 实际 GEMM1 ELF SHA256 |
|---|---|---|
| closing8/v3 | `caeac6c25680447ca5d561f55cc83c3561bfd7998a4142ab47d4c9ba49bde2f3` | `bc5d71d15c5a1893194133eeacb06e372cf528e4489a1660a0f3f2f4c8a0d840` |
| closing4/v3 | `37c057513c7e52eb31efa3c3e6ea5a0d0d2d12e4267fc13c3af0e8f35c7cd803` | `60ecfab19011906da013e7e537f10028502678366c74be1d447610383351b74a` |

两项 seed0 完整 MoE 输出的 22,020,096 B 均与 DS5/pin0 零字节差异，SHA256 仍为
`443e8c5959dc9513a655d5c9b59b21c93c738df5807cda3123560fbc8e47b935`。
原 precision gate 通过，`logits_diff=3.6105513278025183e-06`、`rel_l2=0.0026872099842876196`。
GEMM2 source 和实际 ELF 均与原 pin0 一致。六轮为 attempt=1，GPU 空闲/ownership、
原入口 source-dependent key 和 ELF 绑定通过；下载原表、manifest、pre.log 后本地复核通过。
CPU ELF SHA 与 ISA 静态计数也已从本地文件重新计算。

恢复调度后的两项仍未接近保留版，因此本轮四项 t128 候选均未采用，停止继续遍历这一方向。
这证明已测试的 tile/buffer/调度组合没有收益，未证明硬件无法从其他 t128 实现获益。
当前没有 t128 ATT，也没有证据确定实际 wave occupancy 或各项等待占回退的比例。
生产已经恢复 DS5/pin0，source SHA256 仍为
`e53ada2baecb0c8e0185b815052281952213f0c81c5e4413c621baccfb06f93e`。

```bash
# TAIL 取 8 或 4，NEW_LABEL 必须尚未使用。
FEATURE=dsf5_pin0_t128_b3_scale128_mma2_tail${TAIL}_v3
PREP=reboot20261024_dsf5_pin0_t128_b3_scale128_mma2_tail${TAIL}_prepare_v3
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_small_tile_flydsl_v1.py \
  run "$FEATURE" "$PREP" "$NEW_LABEL" candidate --base-pin0 --candidate-pin0
```

证据为 `logs/reboot20261024_pin0_t128_scale128_sched_cpu_batch_v3.summary.json`，
两项 `reboot20261024_dsf5_pin0_t128_b3_scale128_mma2_tail{8,4}_prepare_v3` 的 preparation，
以及同前缀 `..._candidate_screen_v3.summary.json` 和 `.local_review_v1.json`。

### 固定 LDS 地址改为独立 staged callsite：CPU 编译仍拒绝（2026-10-24）

`dsf5_pin1_rotated_addr_callsite32_v2` 基于此前 packed32 地址候选，
为不同 staged callsite 分配独立的四 DWORD rmem，避免 prologue、steady 和 unrolled drain
复用同一个显式地址槽。只改变目标 pin1 GEMM1 的地址存储分配，未删除 waits。
source SHA256 为 `553cac8c257f2beb27a3f7c0bc0eb76410f5f7902f6d22d474d9214fe88db00e`。

CPU child 在 17.7400 秒退出，returncode=134，报错为
`LLVM ERROR: conflicting FlyDSL register placements after machine optimization`。
没有 GPU launch、有效 ELF、GPU 精度或性能数据。该结果只说明此分配方法仍未通过当前编译器校验，
不能推断 GPU 上的 hazard 或性能。batch 的生产 source 未修改，日志已下载并在本地核验。

证据为 `logs/reboot20261024_pin1_callsite_lds_address_cpu_batch_v2.summary.json`
与 `logs/reboot20261024_dsf5_pin1_rotated_addr_callsite32_v2_cpu_v1.exec.log`。

### 地址预计算反向对照：未取得稳定收益（2026-10-24）

继续完成 `dsf5_pin1_rotated_addr_precompute_v1` 的 candidate/base/candidate 三组，
每组均为完整三轮，使用原入口、A preshuffle=1；候选 pin1，DS5 control pin0。
九轮 attempt=1，原 precision gate、source-dependent key/ELF 绑定和 GPU ownership 均通过。
84.1335 μs 的较慢读数同样保留，没有按耗时筛选或重测。

| 反向位置 | GEMM1 三轮 μs | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---|---:|---:|
| 前地址预计算/pin1 | 82.2147 /81.7276 /84.1335 | 82.2147 | 62.1005 |
| 中 DS5/pin0 | 81.4494 /81.8457 /81.2596 | 81.4494 | 64.7528 |
| 后地址预计算/pin1 | 82.6385 /80.7662 /81.7677 | 81.7677 | 61.5479 |

前后候选分别慢 0.7653 /0.3183 μs，即 0.939602% /0.390795%。
这轮没有复现正向最小 0.0697 μs 的改善，未证明稳定收益，结束本候选的性能验证，不采用。
GEMM2 分别绑定各 pin 环境的原 ELF；不能把跨 pin 的 GEMM2 差异计为 GEMM1 源码收益。
当前保留 DS5/pin0 的此前正式成绩 81.6350 μs，不用本轮 control 81.4494 μs 改写历史记录。

运行结束 `completed=true`、`source_restored=true`；远端 source SHA256 恢复为
`e53ada2baecb0c8e0185b815052281952213f0c81c5e4413c621baccfb06f93e`。
主机 GPU 检查未发现测试进程残留。下载九轮原表、manifest、pre.log 与 summary、original keys 后，
`review_pin_result_local_v1.py` 本地复核通过，`both_sides_improved=false`。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_pin_flydsl_v3.py \
  run dsf5_pin1_rotated_addr_precompute_v1 \
  reboot20261024_dsf5_pin1_rotated_addr_precompute_pin0_prepare_v1 \
  reboot20261024_dsf5_pin1_rotated_addr_precompute_pin0_reverse_v1 \
  candidate base candidate --base-pin0
```

证据为 `logs/reboot20261024_dsf5_pin1_rotated_addr_precompute_pin0_reverse_v1.summary.json`
与同名 `.local_review_v1.json`；再次复现须更换未使用的 run label。

### LLVM tied 地址约束：CPU 编译通过，尚未性能验收（2026-10-24）

`dsf5_pin1_rotated_addr_constraints_v1` 不再给地址 rmem 使用 `fx.set_register`，
改在 `calc_lds_addr()` 中使用 empty tied inline asm constraints，将 current 的
A/B/ScaleA/ScaleB 地址固定至 `v32:35`，next 地址固定至 `v36:39`。
empty asm 的 input/output 为同一值，没有手写 VGPR move 或删改 hazard waits；
指令选择、寄存器分配与 hazard wait 仍由编译器处理。
仅在目标 pin1 GEMM1 scope 生效，原输入、WMMA 累加顺序、TDM/barrier 与输出逻辑保持。

source SHA256 为 `bba7f5a2596261f5c6d9357e2fc7ec85f1c286e7abe0c3566450beb115c04a54`。
CPU 编译在 19.0498 秒完成，ELF SHA256 为
`deea50e0ae511e21d6d7a0a825ec53e2bf664ebfed82b29cc7424093ceda0488`。
资源为 VGPR1024 /SGPR62 /LDS327680 B、private/spills0。
static executable/wait/MSB 为 5724/118/750；rotated 原候选为 5836/194/770。
下载 ELF/ISA/metadata 后本地 SHA、资源和静态计数重新复核通过，全部静态 DS loads
的地址 VGPR 为 `v32:39`。这不证明动态空泡消失或性能改善。
上述为 CPU-only 检查；随后完成的原入口 preparation 和性能初筛记录如下，尚未采用。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/compile_pin_vgpr_batch_v1.py \
  reboot20261024_pin1_constrained_lds_address_cpu_batch_v1 \
  dsf5_pin1_rotated_addr_constraints_v1
```

证据为 `logs/reboot20261024_pin1_constrained_lds_address_cpu_batch_v1.summary.json`
与同名 `.local_review_v1.json`、`reboot20261024_dsf5_pin1_rotated_addr_constraints_v1_cpu_v1.*`。

原入口 preparation 随后完成，实际 GEMM1 ELF 与 CPU 结果一致，GEMM2 为原 pin1 ELF。
seed0 完整 MoE 输出的 22,020,096 B 与 DS5/pin0 零字节差异，输出 SHA256 仍为
`443e8c5959dc9513a655d5c9b59b21c93c738df5807cda3123560fbc8e47b935`；
原 precision gate 通过：`logits_diff=3.6105513278025183e-06`、`rel_l2=0.0026872099842876196`。
summary、original keys、完整输出记录下载后本地复核通过，source 已恢复。

原 `--scenario kernel` 三轮初筛 GEMM1 为 80.6483 /80.6819 /81.2007 μs，
中值 80.6819 μs；GEMM2 中值 61.1368 μs。三轮 attempt=1，原 precision、实际 key/ELF
和测试前后全机 ownership 通过，原表、manifest 与 pre.log 本地复核通过。
该初筛未包含相邻 control，不能据此替换正式 DS5/pin0 成绩或计算其性能收益。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_pin_flydsl_v3.py \
  run dsf5_pin1_rotated_addr_constraints_v1 \
  reboot20261024_dsf5_pin1_rotated_addr_constraints_pin0_prepare_v1 \
  reboot20261024_dsf5_pin1_rotated_addr_constraints_candidate_screen_v1 \
  candidate --base-pin0
```

证据为 `logs/reboot20261024_dsf5_pin1_rotated_addr_constraints_pin0_prepare_v1.preparation.json`
及同前缀 `.local_review_v1.json`，`..._candidate_screen_v1.summary.json` 与 `.local_review_v1.json`。

### 固定地址上的 scale/partition 组合：完成精度与性能初筛（2026-10-24）

固定地址候选上只改变两个独立设置：scale rmem 采用既有的低 VGPR 固定槽方法，
以及 A/B/C 恢复原 partition。前者在 `make_rmem_slot()` 中固定两份 ScaleA/ScaleB 至
`v128:131/v132:135`、`v136:139/v140:143`；后者将 C 恢复 `v512:1023`、
A/B 两槽恢复 `v256:383/v384:511`。current/next LDS 地址仍为 `v32:39`。
这些组合不改变 wave layout、pipeline、TDM/barrier、输入、计算、量化或输出；
均只在原目标 pin1 GEMM1 guard 生效，尚未采用生产源码。

| 私有候选 | source SHA256 | GEMM1 ELF SHA256 | static executable / wait / MSB |
|---|---|---|---|
| `dsf5_pin1_rotated_addr_constraints_scale128_v1` | `8caae10522fa61d07b54c64f1bf2c3a6dd59156780d131f415e21567274f23da` | `4b8b082bdcd44ffa708f4ba48baf5af6509b5cac4288545f4ed7ff7422eaf906` | 5765 /125 /750 |
| `dsf5_pin1_original_partition_addr_constraints_v1` | `14c04da7e2422b5e7492e184f86d6c50c53e72fc47b99a2d2067024854504f3b` | `a73b46f8eedf00a54defb377d6283dc8cfabf368ac989a66f81bf27563ad266e` | 5724 /118 /750 |
| `dsf5_pin1_original_partition_addr_constraints_scale128_v1` | `ca9c215eee728f083aa9defa333ba1446476c2684a53168ff2eaabe375acb960` | `3478ab76dd934dc218d4545bddc19df5854fc3577c6142a0440ce51be69cad84` | 5765 /125 /750 |

三项 CPU 编译分别用时 19.0027 /19.1240 /19.2512 秒，资源均为
VGPR1024 /SGPR62 /LDS327680 B、private/spills0。ELF、完整 ISA machine words、资源及
静态计数下载后在本地重新复核通过；全部 DS load 地址仍位于 `v32:39`。
固定 scale 槽没有继续降低静态 wait 数，但实际耗时仍需原入口验证。
三项随后按固定顺序进入 untimed preparation 和原入口性能初筛，结果如下。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/compile_pin_vgpr_batch_v1.py \
  reboot20261024_pin1_constrained_address_scale_partition_cpu_batch_v1 \
  dsf5_pin1_rotated_addr_constraints_scale128_v1 \
  dsf5_pin1_original_partition_addr_constraints_v1 \
  dsf5_pin1_original_partition_addr_constraints_scale128_v1
```

证据为 `logs/reboot20261024_pin1_constrained_address_scale_partition_cpu_batch_v1.summary.json`
与同名 `.local_review_v1.json` 及各项 `reboot20261024_${FEATURE}_cpu_v1.*`。

三项随后均通过原入口 preparation：真实 GEMM1 ELF 与各自 CPU artifact 一致，
GEMM2 保持原 pin1 ELF。seed0 完整 MoE 的 22,020,096 B 输出均与 DS5/pin0 零字节差异，
SHA256 为 `443e8c5959dc9513a655d5c9b59b21c93c738df5807cda3123560fbc8e47b935`；
原 precision gate 通过，`logits_diff=3.6105513278025183e-06`、`rel_l2=0.0026872099842876196`。
各 preparation summary、original keys、完整 MoE 记录下载后本地复核通过，source 恢复。
三项各三轮的原入口性能初筛按 CPU 列表顺序进行，不根据单轮耗时调整顺序。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_constrained_combinations_v1.py
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_constrained_combinations_v1.py
```

preparation 证据为 `logs/reboot20261024_pin1_constrained_combinations_prepare_batch_v1.summary.json`，
以及三项 `reboot20261024_${FEATURE去掉_v1}_pin0_prepare_v1` 的 `.preparation.json`、
`.original_keys.json` 和 `.local_review_v1.json`。

| 组合 | GEMM1 三轮 μs | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---|---:|---:|
| rotated + 地址约束 + scale128 | 81.1498 /81.0952 /81.2575 | 81.1498 | 60.9408 |
| original partition + 地址约束 | 81.4871 /80.6101 /81.5506 | 81.4871 | 61.4996 |
| original partition + 地址约束 + scale128 | 80.9286 /81.9449 /81.6390 | 81.6390 | 61.4637 |

九轮均为 attempt=1，完整保留原始读数。每轮原 precision gate、实际 source-dependent
key/ELF 与测试前后全机 GPU ownership 均通过，原表、manifest、pre.log、summary 和
original keys 下载后，`review_pin_result_local_v1.py` 三项本地复核均通过。
没有根据单轮耗时删除记录或重测。三组初筛中值未优于单独地址约束的 80.6819 μs，
本轮不采用这些组合；初筛没有相邻 baseline，不将跨时段中值差异计为正式收益。
GEMM2 每轮绑定原 pin1 ELF，其 pin0/pin1 时差不计入 GEMM1 源码优化收益。

batch 完成后 `production_source_restored=true`，远端生产 SHA256 恢复为
`e53ada2baecb0c8e0185b815052281952213f0c81c5e4413c621baccfb06f93e`，
主机 GPU 检查没有发现测试进程残留。保留 DS5/pin0，此前正式中值仍为 81.6350 μs。

性能证据为 `logs/reboot20261024_pin1_constrained_combinations_screen_batch_v1.summary.json`，
以及三项 `reboot20261024_${FEATURE去掉_v1}_candidate_screen_v1.summary.json` 和
同名 `.local_review_v1.json`。复现时应使用未运行过的 label，保留全部三轮。

### 直接约束加载后的 scale SSA 值：CPU 与完整 MoE 检查通过（2026-10-24）

`dsf5_pin1_rotated_addr_scale_constraints_v1` 基于单独地址约束候选，在
`load_lds_data()` 中对 load 返回的 ScaleA/ScaleB SSA 值施加 empty tied LLVM
inline asm constraints。两槽分别使用 `v64:67/v68:71`、`v72:75/v76:79`，
没有给 scale rmem 使用 `fx.set_register`。interleaved 和 legacy load 两分支均在
两次 rmem store 之前返回 constrained 值；这不改变 scale 数据、WMMA 顺序、LDS 容量、
TDM/barrier 或输出。只在原目标 pin1 GEMM1 guard 内生效，未采用生产源码。

source SHA256 为 `fc1e64ca0428b51782bd76613b539db258608f7937b2394060b737f935baf60d`，
ELF SHA256 为 `33716be2f6b9695e6c0309a8c9c089f92e7ed12f4d3440f05514b1af0c9f87b8`。
CPU child 用时 19.0617 秒，资源仍为 VGPR1024 /SGPR62 /LDS327680 B、private/spills0。
static executable /wait /MSB 为 5746 /125 /750；相对单独地址约束版增加 22 条 executable
和 7 条 wait，MSB 数不变。完整 ISA machine words 与 ELF 本地绑定复核通过，
实际 scale DS loads 和 WMMA 的 scale operands 使用指定的低 VGPR。
这只确认约束落在编译结果中，不代表这些新 wait 或原 wait 的动态成本。

原入口 preparation 的真实 GEMM1 ELF 与 CPU artifact 相同，GEMM2 为原 pin1 ELF。
seed0 完整 MoE 的 22,020,096 B 输出与 DS5/pin0 零字节差异，输出 SHA256 为
`443e8c5959dc9513a655d5c9b59b21c93c738df5807cda3123560fbc8e47b935`；
原 precision gate 通过：`logits_diff=3.6105513278025183e-06`、`rel_l2=0.0026872099842876196`。
preparation summary、original keys 与完整输出记录下载后本地复核通过，生产源码已恢复。
三轮性能初筛使用原入口进行，不将 preparation 的耗时作为性能数据。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/compile_pin_vgpr_batch_v1.py \
  reboot20261024_pin1_constrained_scale_values_cpu_batch_v1 \
  dsf5_pin1_rotated_addr_scale_constraints_v1
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_pin_flydsl_v3.py \
  prepare dsf5_pin1_rotated_addr_scale_constraints_v1 \
  reboot20261024_dsf5_pin1_rotated_addr_scale_constraints_pin0_prepare_v1 \
  reboot20261024_dsf5_pin1_rotated_addr_scale_constraints_pin0_prepare_v1 \
  --base-pin0 --artifact=reboot20261024_dsf5_pin1_rotated_addr_scale_constraints_v1_cpu_v1
```

CPU 证据为 `logs/reboot20261024_pin1_constrained_scale_values_cpu_batch_v1.summary.json`
及同名 `.local_review_v1.json`；GPU preparation 证据为
`logs/reboot20261024_dsf5_pin1_rotated_addr_scale_constraints_pin0_prepare_v1.preparation.json`
与同前缀 `.local_review_v1.json`。复现应使用新的 run label。

三轮初筛为 85.0316 /85.8223 /85.3014 μs，中值 85.3014 μs；
GEMM2 中值 61.2506 μs。三轮均 attempt=1，原 precision、source-dependent key/ELF
与测试前后全机 ownership 均通过，未删去较慢读数或根据耗时重测。
性能证据的 35 个 regular members 已下载并逐文件核 SHA256，原始表在本地重算复核通过。
初筛没有相邻 control，不能给出正式回退百分比；该候选没有显示采用价值，本轮不采用。
source 已恢复 DS5/pin0，仍未达到 70 μs 目标。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_pin_flydsl_v3.py \
  run dsf5_pin1_rotated_addr_scale_constraints_v1 \
  reboot20261024_dsf5_pin1_rotated_addr_scale_constraints_pin0_prepare_v1 \
  reboot20261024_dsf5_pin1_rotated_addr_scale_constraints_candidate_screen_v1 \
  candidate --base-pin0
```

性能证据为 `logs/reboot20261024_dsf5_pin1_rotated_addr_scale_constraints_candidate_screen_v1.summary.json`
与同名 `.local_review_v1.json`。证据包 archive SHA256 为
`1a450d612bee04cd309a9a45722ddef31ca52245b0ed96984ebb501382d652ed`，
只包含原表、manifest、ownership、实际 ELF 等任务所需文件，不含 `.out`。

### 将 tied identity constraints 改为 non-volatile：CPU 检查通过（2026-10-24）

对只传递地址/scale 值的 empty tied inline asm 将 `has_side_effects=True` 改为 `False`，
让编译器可以消除或重排这些寄存器值转接。仅改变对应的约束 op 属性，真正的 WMMA
inline asm、调度 ops、TDM、barrier 与输入/输出不变，未手写删减硬件 hazard waits。
两个候选只在原目标 pin1 GEMM1 guard 内使用，不改变 wave layout、pipeline 或 persistent。

| 私有候选 | source SHA256 | ELF SHA256 | static executable /wait /MSB |
|---|---|---|---|
| `dsf5_pin1_rotated_addr_constraints_pure_v1` | `160df0368ef96f1488c6a38a448401f9668e6feed2393f3c42a1ed32142cb733` | `b3354dcf625253049f04bd125b67e970921da1b6f5fb39d9a38ebdfd06ca0595` | 5711 /114 /750 |
| `dsf5_pin1_rotated_addr_scale_constraints_pure_v1` | `a3c0e23a32b6a4274bc7f978fd4457e62c0d934afcc56007010b132763e18b4f` | `958507667d900c5074c0fc42b0ad071b8a02dd44d23b75d79a2f488d61b5b0c1` | 5762 /124 /764 |

CPU child 分别用时 18.9292 /19.0781 秒，资源均为 VGPR1024 /SGPR62 /LDS327680 B、
private/spills0。下载完整 ISA、ELF、资源和 summary 后，CPU 本地复核均通过。
non-volatile 地址版相对原地址约束少 13 条 executable、4 条 wait，MSB 数不变；
叠加 scale 的版本相对直接 scale 约束多 16 条 executable 和 14 条 MSB，wait 仅少 1 条。

实际编译结果的 DS load 地址分布包含 `v2:5` 与 `v32:39`，部分转接已被编译器优化。
不能声称所有 DS 地址仍固定在 `v32:39`，也不能从静态数下降推断动态空泡减少。
地址版随后完成原入口 preparation，实际 GEMM1 ELF 与 CPU artifact 相同，GEMM2 为原 pin1 ELF。
seed0 完整 MoE 的 22,020,096 B 与 DS5/pin0 零字节差异，原 precision gate 通过；
summary、original keys 和完整输出记录下载后本地复核通过。两个候选尚未采用。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/compile_pin_vgpr_batch_v1.py \
  reboot20261024_pin1_pure_vgpr_constraints_cpu_batch_v1 \
  dsf5_pin1_rotated_addr_constraints_pure_v1 \
  dsf5_pin1_rotated_addr_scale_constraints_pure_v1
```

CPU 证据为 `logs/reboot20261024_pin1_pure_vgpr_constraints_cpu_batch_v1.summary.json`
与同名 `.local_review_v1.json`，以及两项 `reboot20261024_${FEATURE}_cpu_v1.*`。

scale 组合也随后完成原入口 preparation，实际 GEMM1 ELF 与 CPU artifact 一致，
GEMM2 为原 pin1 ELF。两个候选的 seed0 完整 MoE 输出均为零字节差异，SHA256 为
`443e8c5959dc9513a655d5c9b59b21c93c738df5807cda3123560fbc8e47b935`，
`logits_diff=3.6105513278025183e-06`、`rel_l2=0.0026872099842876196`。
两项 preparation summary、original keys 和完整输出记录下载后本地复核通过，source 恢复。
这只验证了当前 seed0 实例，尚不代表全部支持 shape 的验证完成。

preparation 证据为两项 `logs/reboot20261024_${FEATURE去掉_v1}_pin0_prepare_v1.preparation.json`
及同名前缀 `.local_review_v1.json`。

两个候选随后按固定顺序完成原 `--scenario kernel` 三轮初筛，原始结果如下。

| non-volatile 组合 | GEMM1 三轮 μs | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---|---:|---:|
| 仅 LDS 地址约束 | 81.9935 /81.2006 /81.0805 | 81.2006 | 61.0968 |
| LDS 地址 + scale SSA 约束 | 87.3887 /87.5660 /87.6124 | 87.5660 | 61.3055 |

六轮均为 attempt=1，原 precision gate、实际 source-dependent key/ELF 和测试前后全机
GPU ownership 通过。原始表、manifest、ownership 与实际 ELF 的证据包各包含 35 个
regular members，下载后逐文件核对 SHA256，并在本地重新计算三轮中值，复核通过。
没有删除较慢读数或根据耗时重测。仅地址版没有显示优于原地址约束初筛的证据；
加入 scale 的版本更慢，本轮均不采用。没有相邻 control，不计算正式收益或回退比例。
GEMM2 始终绑定原 pin1 ELF，不将 pin0/pin1 的 GEMM2 时差归入 GEMM1 源码收益。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_pin_flydsl_v3.py \
  run dsf5_pin1_rotated_addr_constraints_pure_v1 \
  reboot20261024_dsf5_pin1_rotated_addr_constraints_pure_pin0_prepare_v1 \
  reboot20261024_dsf5_pin1_rotated_addr_constraints_pure_candidate_screen_v1 \
  candidate --base-pin0
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_pin_flydsl_v3.py \
  run dsf5_pin1_rotated_addr_scale_constraints_pure_v1 \
  reboot20261024_dsf5_pin1_rotated_addr_scale_constraints_pure_pin0_prepare_v1 \
  reboot20261024_dsf5_pin1_rotated_addr_scale_constraints_pure_candidate_screen_v1 \
  candidate --base-pin0
```

性能证据为两项 `logs/reboot20261024_${FEATURE去掉_v1}_candidate_screen_v1.summary.json`
及同名 `.local_review_v1.json`。仅地址和地址 + scale 的 archive SHA256 分别为
`0e40f4a9d85afcf964c48b865f9e345b44ff51154be12f48382d37a9854de98c`、
`3cb7e9cc43d19a910664797b020a558ec436921b8ac95655421af0f4a7b0f505`。
证据包不包含 `.out`。两项完成后 production source 已恢复，保留 DS5/pin0。

### 固定 LDS 地址 VGPR：正反向对照均有约 1% 改善（2026-10-24）

`dsf5_pin1_rotated_addr_constraints_v1` 完成原入口 `base candidate base` 对照。
base 为 DS5/pin0，candidate 开启 pin1 并使用此前 CPU 编译、完整 MoE 检查过的地址约束源码。
测试开始前全机空闲，各轮只运行当前测试，没有并发远端编译、传输或 GPU 检查。

| 顺序与版本 | GEMM1 三轮 μs | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---|---:|---:|
| 1：DS5/pin0 | 81.8280 /82.2673 /82.3897 | 82.2673 | 64.7379 |
| 2：地址约束/pin1 | 81.0531 /81.1164 /80.9918 | 81.0531 | 61.6787 |
| 3：DS5/pin0 | 82.0988 /81.9745 /81.9933 | 81.9933 | 64.6347 |

候选比前后 control 中值分别减少 1.2142 /0.9402 μs，即 1.47592% /1.14668%。
九轮均 attempt=1，原 precision gate、实际 source-dependent key/ELF 与前后 ownership
检查通过；没有删除慢读数或根据性能重测。summary 的 `completed=true`、
`source_restored=true`，生产源码恢复 DS5/pin0。

证据包的 101 个 regular members 下载后逐文件验证 SHA256，本地从原始 benchmark 表
重新计算各组中值和两侧改善比例，通过 `review_pin_result_local_v1.py` 复核。
原入口和计算规模未改，没有向 cache 注入修改 ELF。GEMM2 在 pin0/pin1 环境分别绑定
相应原 ELF，其时差不属于 GEMM1 源码收益。该结果尚未替换此前 81.6350 μs 的正式记录，
仍未达到 70 μs；下一步按 `candidate base candidate` 反向验证。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_pin_flydsl_v3.py \
  run dsf5_pin1_rotated_addr_constraints_v1 \
  reboot20261024_dsf5_pin1_rotated_addr_constraints_pin0_prepare_v1 \
  reboot20261024_dsf5_pin1_rotated_addr_constraints_pin0_forward_v1 \
  base candidate base --base-pin0
```

证据为 `logs/reboot20261024_dsf5_pin1_rotated_addr_constraints_pin0_forward_v1.summary.json`
与同名 `.local_review_v1.json`。archive SHA256 为
`0809af6e17173e5c975d7bfaca31fd6ded6bb5aa2240c394c8a460418d701276`，不含 `.out`。

随后完成反向 `candidate base candidate`，全部九轮按预定顺序保留。

| 顺序与版本 | GEMM1 三轮 μs | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---|---:|---:|
| 1：地址约束/pin1 | 81.1796 /80.7262 /81.4306 | 81.1796 | 61.5798 |
| 2：DS5/pin0 | 81.8143 /82.0757 /81.9807 | 81.9807 | 64.6326 |
| 3：地址约束/pin1 | 82.2718 /81.1247 /80.0769 | 81.1247 | 60.5675 |

两组候选比中间 control 分别减少 0.8011 /0.8560 μs，即 0.97718% /1.04415%。
没有选用最后一轮的 80.0769 μs 替代三轮中值，亦未排除 82.2718 μs 的慢读数。
九轮均 attempt=1，原 precision、真实 key/ELF 与测试前后 ownership 通过；
`completed=true`、`source_restored=true`。反向证据的 101 个 regular members 下载后
逐文件核 SHA256，本地从原表复核三轮中值和两侧改善比例，通过。

本轮正反向共十八轮，三个候选组中值为 81.0531 /81.1796 /81.1247 μs，
分别在其相邻 control 下显示约 1% 的收益。它们属于三个独立组，不拼接为新的三轮成绩。
保留代码前还需动态 replay、其他 seed 和原 guard 外的适用范围检查，目前生产仍为
DS5/pin0。新候选尚未达到 70 μs，开始用用户指定脚本采集四个 SIMD 的 ATT。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_pin_flydsl_v3.py \
  run dsf5_pin1_rotated_addr_constraints_v1 \
  reboot20261024_dsf5_pin1_rotated_addr_constraints_pin0_prepare_v1 \
  reboot20261024_dsf5_pin1_rotated_addr_constraints_pin0_reverse_v1 \
  candidate base candidate --base-pin0
```

反向证据为 `logs/reboot20261024_dsf5_pin1_rotated_addr_constraints_pin0_reverse_v1.summary.json`
与同名 `.local_review_v1.json`。archive SHA256 为
`7dbae6a48cf0dd8b30589c4f3e1677d9e374b821b69c366b9c78d64203eaae52`，不含 `.out`。

### 地址约束候选的适用范围修正（2026-10-24）

上述候选的四 SIMD ATT 已完成，实际 GEMM1 ELF 仍为 `deea50e0…ceda0488`；
64 个 active waves 均执行 1792 条 WMMA。mainloop 区间中值为 17331.5 cycles/wave，
相比 DS5/pin0 的 18024.0 cycles/wave 减少约 3.84%。完整 kernel 的实测收益仍以
前述正反向约 1% 为准，不能把 ATT 的 phase 比例当作可直接删除的成本。
详细分析和本地重算记录见 `moe_gemm1_hyg_dsr1_best_thread_trace_analysis.md`。

合入前的 CPU-only 范围检查发现，原地址约束候选将 `DS_B` 的 pin1 条件扩大到了
`stage1_act=2`。原条件用 `stage1_act` 非零判断，而扩展没有复用完整的 SiLU guard，
使该路径的 `DS_B` 从 2 变为 4。其完整 ELF 不同，`.text` 从 39680 B 变为 39552 B；
这是已确认的适用范围扩大，尚未证明该路径存在数值错误。候选因此没有合入。

| `act2/pin1` 对照 | 完整 ELF SHA256 | `.text` SHA256 |
|---|---|---|
| DS5 production | `e632209a104f8971709efcd3df68a22ca0ec5173249c2301e3dc561cb24bef28` | `45facdda02a82184948e1bb47d2f7c3ca58269529146860719d03ac1f4a4609f` |
| 原地址约束候选 | `e03c4086ca2bcdb9efc54b69b60f6872d7b94861eaa6a9c17a1b092b6243fa95` | `d0ace0e334d0f7dd712502c6ddfdd1b52fc454ac93752baafc0c67f912c890c5` |

只补 `stage1_act == 1` 的私有修正版 source SHA256 为
`db6ba3b8c3739f8ae571192f41f928247ae22ba9ec5436425c962655e0e94875`。
该版目标 SiLU 的完整 ELF 仍为 `deea50e0…ceda0488`，已完成 CPU 编译比较。
进一步检查发现 pin1 扩展也应限定 `stage1_quant_out`，因此准备最终范围候选
`dsf5_pin1_rotated_addr_scoped_v1`，让 `DS_B` 直接复用 `compact_pin_gemm1`。
其 source SHA256 为 `2273b14396be413c8789c98c9a42c7be3f8678ee9ba847826bc16c56854de17d`。
该 guard 包含 SiLU、量化输出、FP4 preshuffled A、非 persistent、t256/w2x2/b4、
cluster4x1；pin0 原来的非零 activation 条件保留。当时最终候选仍需范围比较和动态验证，
后续完成情况见下面的核验记录。

原范围检查失败记录保存在两项
`logs/reboot20261024_dsf5_pin1_rotated_addr_constraints_guard_scope_v1.scope.json`、
`logs/reboot20261024_dsf5_pin1_rotated_addr_constraints_guard_scope_v2.scope.json`。
前一次还包含 artifact 筛选工具只识别 `_act1_` 的失败；修正筛选后才暴露上述真实 ELF 差异。
旧失败证据保留，未覆盖或删除；生产 source 仍为 `e53ada2b…f06f93e`。

### 最终范围候选：11 项 ELF 对照及完整 MoE 严格验证通过（2026-10-24）

`dsf5_pin1_rotated_addr_scoped_v1` 已完成 CPU 编译与本地 ISA/ELF 复核。
目标 GEMM1 完整 ELF 仍为 `deea50e0ae511e21d6d7a0a825ec53e2bf664ebfed82b29cc7424093ceda0488`，
与前述正反向性能对照、四 SIMD ATT 中实际执行的 code object 完全一致。
VGPR1024 /SGPR62 /LDS327680 B /private0 /spills0；static executable、
`s_wait_alu`、`s_set_vgpr_msb` 分别为 5724 /118 /750。只收紧 guard，没有改变目标 ISA。

以下 11 项与 DS5 production 的完整 ELF 逐字节相同：pin0 目标、GEMM2 pin0、
GEMM2 pin1、原始 A/pin1、FP8 输入/pin0、t64 fallback/pin1、w1x4/pin0、b3/pin1、
act2/pin1、已有 persistent/pin0、SiLU 非量化输出/pin1。
22 份 ELF 及对应 metadata 下载后在本地独立核对。
`act2/pin1` 恢复为 `e632209a…cb24bef28`，SiLU 非量化输出/pin1 为 `977955ed…2c1072`。
这些是已测试路径的范围证据，不作为对所有未测试组合的数值证明。

严格验证使用当前 DS5 production 作 reference，保留原 logits_diff gate 0.01。
所有有效 FP4 payload、ScaleA 检查和完整 MoE 输出均为零字节差异。

| case | 实际 GEMM1 路径 | byte checks | 完整 MoE bytes | logits_diff | rel_l2 |
|---|---|---:|---:|---:|---:|
| E64 seed0 dynamic | 目标 t256/w2x2/pin1 | 27 | 22020096 | 3.6105513e-6 | 0.002687210 |
| E64 seed1 dynamic | 目标 t256/w2x2/pin1 | 27 | 22020096 | 3.6122000e-6 | 0.002687819 |
| E64 seed2 | 目标 t256/w2x2/pin1 | 13 | 22020096 | 3.6004471e-6 | 0.002683450 |
| E64 K4096 seed0 dynamic | 原 t64/w1x4/b3 fallback | 27 | 12582912 | 3.6079736e-6 | 0.002686252 |
| E64 bias seed0 | 原 SiLU 非量化输出，guard 外 | 7 | 22020096 | 3.7544465e-6 | 0.002740237 |

dynamic 包含 eager、CUDA Graph、相同地址修改 A/ScaleA/psum、empty counts、边界和
尾部 counts，以及输入恢复。seed0/1/2 的目标 GEMM1 ELF 为 `deea50e0…ceda0488`，
GEMM2 为原 pin1 ELF `a5e9c250…58375311`。
K4096 和 bias 的实际 kernel symbols、ELF 各不相同，不能把它们计入目标 t256 量化
kernel 的覆盖，也不能要求它们的 GEMM2 ELF 等于目标无 bias 的 code object。
本地重新从五份日志计算 checks/metrics、核对配置和十份 ELF SHA256，复核通过。
这批是正确性验证，耗时不作为性能数据。每个编译/测试 child 均限制为 120 秒。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_pin_guard_scope_v2.py \
  dsf5_pin1_rotated_addr_scoped_v1 \
  reboot20261024_rotated_addr_scoped_guard_scope_v1 \
  reboot20261024_dsf5_pin1_rotated_addr_constraints_guard_scope_v2
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_pin_strict_batch_v1.py \
  dsf5_pin1_rotated_addr_scoped_v1 reboot20261024_rotated_addr_scoped_strict_v1
```

复现需更换未使用的 label。CPU 证据为
`logs/reboot20261024_rotated_addr_scoped_cpu_batch_v1.summary.json` 及同名本地复核；
范围证据为 `logs/reboot20261024_rotated_addr_scoped_guard_scope_v1.scope.json` 和
同名 `.local_review_v1.json`；严格验证为
`logs/reboot20261024_rotated_addr_scoped_strict_v1.strict.json`、五 case 原始日志和
`.local_review_v2.json`。v2 按实际 symbol 区分了 bias 非量化输出的 guard 外路径。
候选 source SHA256 为 `2273b14396be413c8789c98c9a42c7be3f8678ee9ba847826bc16c56854de17d`。
本轮未改变 E65 dispatch；原分支 E65 仍走 t64/w1x4 fallback，不能宣称已满足 E64/E65
共用同一 GEMM1 的历史要求。生产 source 仍保留 DS5/pin0，暂未采用新候选。

### 地址寄存器 v64:71 与两项低编号布局：未取得稳定收益（2026-10-24）

从最终范围候选生成三项独立布局，数学、调度和 guard 不变。

| feature | C VGPR | A/B VGPR | 地址 VGPR | CPU 编译 |
|---|---|---|---|---|
| `dsf5_pin1_low_c_addr800_scoped_v1` | 0:511 | 512:767 | 800:807 | rc134，未 dispatch |
| `dsf5_pin1_low_ab_addr288_scoped_v1` | 512:1023 | 0:255 | 288:295 | rc134，未 dispatch |
| `dsf5_pin1_rotated_addr64_scoped_v1` | 256:767 | 768:1023 | 64:71 | rc0 |

两项失败日志包含 fixed asm output register 分配错误，最终均报
`LLVM ERROR: explicit register conflicts with precolored machine operand: VGPR0`。
这是当前编译器/ABI 下的显式寄存器约束冲突，不能据此判断硬件不支持低编号 VGPR，
也没有取得 device 数值或性能结果。不能笼统将其归为已证明的 live interval overlap。

addr64 的 source SHA256 为
`a25eb849ffd6b5f42727377d27cfaf3ba902a1952927333832b8e4ee69a057e5`，
目标 ELF 为 `1eadf5056ebcb4789f20f13110a4c26ef02caa09700ea7d5b0d0f26c108f55b5`。
VGPR1024 /SGPR62 /LDS327680 B /private0 /spills0；static executable、
`s_wait_alu`、`s_set_vgpr_msb` 为 5723 /117 /750。
原入口 preparation 与 CPU ELF 相符，seed0 完整 MoE 的 22,020,096 B 与 DS5/pin0
零字节差异，输出 SHA256 为 `443e8c59…8e47b935`。准备耗时不作为性能数据。

随后在原入口按 addr32 → addr64 → addr32 固定次序各测三轮，实际全部使用 pin1。
底层 wrapper 的 `--base-pin0` 用于准备 reference 配置；这三组只计时 `candidate`
mode，不能误报为 pin0/pin1 对比。GEMM2 三组均绑定原 pin1 ELF `a5e9c250…58375311`。

| 顺序与布局 | GEMM1 三轮 μs | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---|---:|---:|
| 1：addr32 control | 80.7199 /81.1754 /81.4112 | 81.1754 | 61.2001 |
| 2：addr64 candidate | 81.1498 /81.2154 /80.9772 | 81.1498 | 61.4593 |
| 3：addr32 control | 80.7241 /79.8489 /80.2666 | 80.2666 | 61.2969 |

addr64 相对前侧仅改善 0.0256 μs（0.0315366%），相对后侧回退 0.8832 μs（1.1003331%）。
没有稳定收益，放弃 addr64，继续以 addr32 为优化基础。
保留后侧 control 的 79.8489 μs 单轮和整组三轮中值，未将该单轮作为最优正式成绩。
九轮均 attempt=1，原 precision、实际 source-dependent call key/ELF、测试前后全机
ownership 检查通过；计时期间没有并发远端编译、SFTP 或 GPU 检查。
生产 source 已精确恢复 `e53ada2b…f06f93e`，测试后无 KFD/GPU 测试进程残留。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/compile_pin_vgpr_batch_v1.py \
  reboot20261024_scoped_pin_layouts_cpu_batch_v1 \
  dsf5_pin1_low_c_addr800_scoped_v1 dsf5_pin1_low_ab_addr288_scoped_v1 \
  dsf5_pin1_rotated_addr64_scoped_v1
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/compare_pin_addr64_normal_v1.py \
  reboot20261024_addr64_scoped_normal_v1
```

复现需换未使用的 labels；生成器为 `tools/make_scoped_pin_layouts_v1.py`。
CPU 证据为 `logs/reboot20261024_scoped_pin_layouts_cpu_batch_v1.summary.json` 和本地复核。
性能证据为 `logs/reboot20261024_addr64_scoped_normal_v1.comparison.json`、
三个 group 的原始表及 `.local_review_v1.json`，parent 的
`.comparison.local_review_v1.json` 在本地重新计算两侧比例。
205 个 regular members 的任务证据包下载后逐文件核 SHA256，archive 为
`9d6f249a5025b41b2aa5091ccd9346ce8849824ae0e85ac94f93f020a4f28a4d`，不含 `.out`。

### addr32/pin1 的独立调度参数：编译和 preparation 完成（2026-10-24）

保持最终范围候选的寄存器位置、数学、输入 ring 和 guard，只在原目标 guard 内，
通过 `SUPPORT_EXPLICIT_VGPR_PARTITION` 条件改变一个数字。每项相对范围候选仅改一行，
pin0 保持原值，不添加 env、persistent、w2x4 或新 pipeline。

| feature | 单项修改 | static executable | `s_wait_alu` | `s_set_vgpr_msb` |
|---|---|---:|---:|---:|
| addr32 reference | MMA2/first1/closing25/DS5 | 5724 | 118 | 750 |
| `dsf5_pin1_addr32_mma3_scoped_v1` | MMA2 → MMA3 | 5609 | 118 | 640 |
| `dsf5_pin1_addr32_first2_scoped_v1` | first1 → first2 | 5706 | 118 | 732 |
| `dsf5_pin1_addr32_close20_scoped_v1` | closing25 → closing20 | 5764 | 118 | 790 |
| `dsf5_pin1_addr32_close24_scoped_v1` | closing25 → closing24 | 5744 | 118 | 770 |
| `dsf4_pin1_addr32_scoped_v1` | 首批 DS5 → DS4 | 5714 | 118 | 740 |

五项 CPU 编译均 rc0，每个 child 限时 120 秒。
VGPR1024 /SGPR62 /LDS327680 B，private/spills 均为 0。
五份 ELF/ISA 下载后逐 machine word 绑定、重新计算指令数和资源，复核通过。
各项的静态 WMMA、BF16 转换、FP4 转换、tanh 为 640 /128 /32 /256，
TDM load/store 为 24 /1，DS load 为 720，与 reference 相同。
这些是模板内静态 counts，不作为 runtime WMMA 数或硬件正确性证明。

编译器重排后部分 `s_wait_dscnt` immediate/序列改变；不能宣称完整 native fence 序列相同。
排除 DS wait 后，TDM wait、barrier signal/wait、store wait 的有序序列逐项相同，
`s_wait_dscnt 0x0` 均保持 18 个静态 site。源码中的 wait/barrier helper 调用未改。
该核对用于识别变化范围，候选仍须按数值验证和原入口性能决定是否采用。

五项随后分别生成真实原入口 source-dependent call keys，实际目标 ELF 与 CPU artifact
相符，GEMM2 均为原 pin1 ELF。seed0 完整 MoE 的 22,020,096 B 与 DS5/pin0 reference
零字节差异，原 precision gate 通过。preparation 耗时不作为性能。
开始预定次序 addr32 control → MMA3 → first2 → closing20 → closing24 → DS4 →
addr32 control 的原入口三轮性能初筛；尚未据此采用任何参数候选。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/compile_pin_vgpr_batch_v1.py \
  reboot20261024_scoped_pin_schedule_cpu_batch_v1 \
  dsf5_pin1_addr32_mma3_scoped_v1 dsf5_pin1_addr32_first2_scoped_v1 \
  dsf5_pin1_addr32_close20_scoped_v1 dsf5_pin1_addr32_close24_scoped_v1 \
  dsf4_pin1_addr32_scoped_v1
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/prepare_pin_schedule_batch_v1.py \
  reboot20261024_scoped_pin_schedule_prepare_batch_v1
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/compare_pin_series_v1.py \
  reboot20261024_pin_schedule_normal_screen_v1 \
  dsf5_pin1_addr32_mma3_scoped_v1 dsf5_pin1_addr32_first2_scoped_v1 \
  dsf5_pin1_addr32_close20_scoped_v1 dsf5_pin1_addr32_close24_scoped_v1 \
  dsf4_pin1_addr32_scoped_v1
```

以上为实际归档命令。重新生成/编译需要新的候选、artifact 和 run labels，
已有文件的覆盖保护不能直接绕过。生成器 `tools/make_scoped_pin_schedule_v1.py`、
`logs/reboot20261024_scoped_pin_schedule_v1.plan.json` 记录五项完整 source SHA256。
CPU 证据为 `logs/reboot20261024_scoped_pin_schedule_cpu_batch_v1.summary.json` 及本地复核，
另有 `logs/reboot20261024_scoped_pin_schedule_native_review_v1.json` 和
`logs/reboot20261024_scoped_pin_schedule_fence_review_v1.json`。

### addr32/pin1 调度初筛原始数据复核（2026-10-24）

预定七组共 21 轮全部完成，每组 attempt=1；没有按耗时排除轮次或拼接样本。
计时前全机空闲、原入口 precision、source-dependent call keys 和实际 ELF 均通过。
GEMM2 全程使用原 pin1 ELF `a5e9c250…58375311`。计时期间没有并发远端编译、
SFTP 或额外 GPU 检查，结束后 production source 精确恢复 `e53ada2b…f06f93e`。

| 固定顺序 | GEMM1 三轮 μs | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---|---:|---:|
| addr32 control 前 | 80.9771 /81.5613 /80.7894 | 80.9771 | 61.9657 |
| MMA3 | 81.9071 /81.4115 /81.2869 | 81.4115 | 61.3876 |
| first2 | 81.1058 /80.6691 /80.4205 | 80.6691 | 61.1875 |
| closing20 | 80.9094 /80.8526 /81.0276 | 80.9094 | 61.5480 |
| closing24 | 81.4305 /80.7959 /80.9351 | 80.9351 | 61.1748 |
| DS4 | 81.3461 /81.3481 /80.9919 | 81.3461 | 61.9906 |
| addr32 control 后 | 80.9981 /81.0867 /80.5449 | 80.9981 | 61.5499 |

first2 相对两侧分别改善 0.3803544% /0.4061824%，距 70 μs 仍差 10.6691 μs。
MMA3、DS4 回退；closing20/24 两侧改善均不到 0.12%，暂不采用。
first2 的 source SHA256 为 `60d250bbfbecc76f8eb4bb7a02dbf05f73ae241967affbbaf38cd41b59a2b0dc`，
实际 GEMM1 ELF 为 `2b015b033dd98c8497b96e56905811066299982773c3b2ddd9cf3fefbe3eae8c`。
该项尚未完成反向确认和多 seed 严格验证，不能将初筛结果称为稳定采用成绩。

任务证据包共 494 个 regular members，493 项文件 SHA256 逐项通过本地复核，
不含 `.out`；archive SHA256 为 `9a373ef9eed9c8568ffbf1581d92dd3b7e005e20ae5e1ec2a6c2cbc944a617df`。
五项 preparation 和七组原始性能表均已下载并本地重算通过，parent 为
`logs/reboot20261024_pin_schedule_normal_screen_v1.comparison.json`，本地复核为
同名 `.comparison.local_review_v1.json`。各 group 对应 `_reference_before`、
`_candidate1` 至 `_candidate5`、`_reference_after` 前缀。

继续以固定次序 first2 → addr32 reference → first2，各完整三轮反向确认；
实际全部计时 pin1，不修改原 benchmark。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/compare_pin_series_v1.py \
  reboot20261024_pin_first2_reverse_confirm_v1 \
  dsf5_pin1_addr32_first2_scoped_v1 --reverse
```

复现需换未占用的 label。反向模式只允许一个候选，并分别保留前后候选对同一
中间 reference 的比例；不将两组拼成一组中值。

### first2 反向确认未复现稳定收益，放弃（2026-10-24）

反向九轮均完成且 attempt=1，原 precision、实际两阶段 ELF、全机 ownership 检查通过。
原始表已下载，在本地重算三组中值及比例，复核通过。

| 固定顺序 | GEMM1 三轮 μs | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---|---:|---:|
| first2 前 | 80.3129 /81.6329 /81.2911 | 81.2911 | 61.0231 |
| addr32 reference | 80.1781 /80.6145 /80.4543 | 80.4543 | 61.3266 |
| first2 后 | 80.1907 /80.4141 /79.8743 | 80.1907 | 61.9507 |

前侧 first2 比中间 reference 慢 0.8368 μs（1.0400936%）；后侧快 0.2636 μs（0.3276394%）。
正向初筛的约 0.4% 改善没有在反向两侧同时复现，不采用 first2，也不将后侧中值或
79.8743 μs 单轮作为新的正式成绩。生产仍保留 DS5，后续布局以最终 addr32 范围候选为 reference。

证据为 `logs/reboot20261024_pin_first2_reverse_confirm_v1.comparison.json`、三组原表及
对应本地复核。archive SHA256 为 `40827dbd8705d8ad0bbc06c10a47ba8809141e2b0e8ebd0b7cf3f5261257f264`，
136 项任务文件逐文件核 SHA256，不含 `.out`。

### pin1 非零起始 VGPR 布局：CPU 与数值准备（2026-10-24）

以 `dsf5_pin1_rotated_addr_scoped_v1` 为 reference，保持数学计算、输入 ring、调度参数和
`compact_pin_gemm1` guard，分别调整 C、A/B 和 current/next LDS 地址的 VGPR 编号。
生成器为 `tools/make_scoped_pin_nonzero_layouts_v1.py`，完整 source SHA256 在
`logs/reboot20261024_scoped_pin_nonzero_layouts_v2.plan.json`；不采用先前的 v1 plan。

| feature（均加 `dsf5_pin1_` 前缀、`_scoped_v1` 后缀） | C VGPR | A/B VGPR | 地址 VGPR | 编译 | VGPR 数 | static executable / `s_wait_alu` / `s_set_vgpr_msb` |
|---|---|---|---|---|---:|---|
| `c128_ab640_addr32` | 128:639 | 640:895 | 32:39 | rc0 | 896 | 5648 /118 /679 |
| `c64_ab576_addr32` | 64:575 | 576:831 | 32:39 | rc0 | 832 | 5678 /118 /711 |
| `c384_ab128_addr32` | 384:895 | 128:383 | 32:39 | rc0 | 896 | 5652 /118 /678 |
| `c512_ab64_addr352` | 512:1023 | 64:319 | 352:359 | rc134 | — | — |
| `c240_ab752_addr1008` | 240:751 | 752:1007 | 1008:1015 | rc134 | — | — |

三项成功编译的候选均为 SGPR62 /LDS327680 B /private0 /spills0。两项失败日志出现高编号
fixed asm constraints 分配错误，最终报 `LLVM ERROR: conflicting FlyDSL register placements after machine optimization`。
这是当前编译器约束分配失败，不能据此判断硬件不支持相应 VGPR。

下载原始 ELF/ISA 后，逐 machine word 核对通过。三项均保留 static WMMA640、DS load720、
TDM load24/store1、`v_tanh_f32_e32`256、`v_cvt_pk_bf16_f32`128、
`v_cvt_scalef32_pk8_fp4_bf16`32。排除 DS wait 后的 TDM/barrier/store/load wait 有序序列一致，
DS wait0 均有 18 个 site；没有假定所有 DS wait immediate 相同。
使用修正了 opcode 拼法的 `logs/reboot20261024_scoped_pin_nonzero_native_review_v2.json`，
不以 v1 的零 count 作为依据。

三项原入口 preparation 均通过，seed0 完整 MoE 的 22,020,096 B 与生产 DS5/pin0 零字节差异；
GEMM2 始终为原 pin1 ELF `a5e9c2504863d38c85e39acdc3e28b35774a99f21957b517e338953558375311`。
这些 preparation 耗时不作为性能数据；性能须使用独立的前后夹测和反向确认。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/compile_pin_vgpr_batch_v1.py \
  reboot20261024_scoped_pin_nonzero_cpu_batch_v1 \
  dsf5_pin1_c128_ab640_addr32_scoped_v1 dsf5_pin1_c64_ab576_addr32_scoped_v1 \
  dsf5_pin1_c384_ab128_addr32_scoped_v1 dsf5_pin1_c512_ab64_addr352_scoped_v1 \
  dsf5_pin1_c240_ab752_addr1008_scoped_v1
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/compare_pin_series_v1.py \
  reboot20261024_pin_nonzero_layouts_normal_screen_v1 \
  dsf5_pin1_c128_ab640_addr32_scoped_v1 dsf5_pin1_c64_ab576_addr32_scoped_v1 \
  dsf5_pin1_c384_ab128_addr32_scoped_v1
```

复现需更换未使用的 labels，并使用此前保留的 source-dependent 原入口 preparation。
实际性能序列全部计时 pin1；wrapper 的 `--base-pin0` 只指定 preparation 的参考配置。

十五轮初筛全部完成且 attempt=1。原表、source-dependent call keys/ELF、运行前后全机
ownership 记录下载后逐项复核，中值和两侧比例独立重算通过。
计时期间没有远端编译、SFTP 或额外 GPU 检查；production 精确恢复为 `e53ada2b…f06f93e`。

| 固定顺序 | GEMM1 三轮 μs | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---|---:|---:|
| addr32 reference 前 | 80.9456 /80.8865 /80.8949 | 80.8949 | 61.3687 |
| `c128_ab640_addr32` | 81.5760 /80.7831 /80.6270 | 80.7831 | 61.4636 |
| `c64_ab576_addr32` | 80.6903 /80.7642 /81.0532 | 80.7642 | 61.0673 |
| `c384_ab128_addr32` | 81.8521 /81.2661 /80.5681 | 81.2661 | 61.2909 |
| addr32 reference 后 | 81.7530 /83.0686 /81.0784 | 81.7530 | 61.2507 |

`c128` 相对前/后 reference 改善 0.1382040% /1.1863785%；`c64` 改善
0.1615677% /1.2094969%。前侧差值仅 0.1118 /0.1307 μs，后侧 reference 有明显波动；
没有据此采用新布局，优先反向确认 `c64`。`c384` 对前侧回退 0.4588670%、对后侧改善
0.5955745%，没有双侧收益，淘汰。所有原始轮次保留，包括 reference 的 83.0686 μs。

性能证据为 `logs/reboot20261024_pin_nonzero_layouts_normal_screen_v1.comparison.json`
和同名 `.comparison.local_review_v1.json`；五个 group 各有原表与 `.local_review_v1.json`。
archive SHA256 为 `8ea5191dd404a560e9c40acd83916dcc5bf1170c24e8a103ad4f956dacfe233f`，
226 项任务文件逐文件核验，不含 `.out`。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/compare_pin_series_v1.py \
  reboot20261024_pin_c64_layout_reverse_confirm_v1 \
  dsf5_pin1_c64_ab576_addr32_scoped_v1 --reverse
```

上述反向确认的最终结果须独立记录，不能把初筛中值称为稳定最优成绩。

反向九轮全部完成，attempt=1；原表、实际两阶段 ELF、ownership 证据下载后本地复核通过。

| 固定顺序 | GEMM1 三轮 μs | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---|---:|---:|
| `c64_ab576_addr32` 前 | 81.3525 /80.6419 /82.6235 | 81.3525 | 61.3729 |
| addr32 reference | 81.9345 /81.3566 /81.3333 | 81.3566 | 61.4848 |
| `c64_ab576_addr32` 后 | 80.2962 /81.2090 /82.2294 | 81.2090 | 61.0947 |

两侧候选分别快 0.0041 /0.1476 μs，即 0.0050395% /0.1814235%。方向均为改善，但相对原始
轮间波动仍缺乏足够稳定收益证据，不采用 `c64`；也没有将最小单轮 80.2962 μs 当成正式成绩。
`c128` 尚未反向确认，不将其初筛结果称为失败或稳定最优，也未采用。
继续保留 addr32/scoped reference，下一项为 M64 无效计算跳过。

证据为 `logs/reboot20261024_pin_c64_layout_reverse_confirm_v1.comparison.json` 及同名
`.comparison.local_review_v1.json`。与 M64 preparation 合并的 archive SHA256 为
`d210c7e932bd5a5c513e18f2a769ae49588564eeb4861ebbd8852d8d07b9579c`，
159 项任务文件逐项核 SHA256，不含 `.out`；生产 source 精确恢复，测试后无 KFD/GPU 残留。

### addr32 reference 的 T4608 动态正确性补测（2026-10-24）

默认 T4608 GEMM1 为 `cluster4x4`，在优化 guard 外；添加已有
`AITER_FLYDSL_FORCE_1X4_CLUSTER=1` 后为 guard 内的 `cluster1x4`、unrolled drain 路径。
两项都完成 27 项 byte checks，包含动态 routing replay；完整 MoE 的 66,060,288 B 均零字节差异。
两项 `logits_diff=3.6132431751872913e-6`、`rel_l2=0.0026882083620876074`，通过原精度门槛。
此次没有记录 T4608 性能。

| 路径 | 实际 GEMM1 ELF SHA256 | 实际 GEMM2 ELF SHA256 |
|---|---|---|
| 默认 `cluster4x4`（guard 外） | `92363ef80d0fbec2445ee00b893ab7fb8a668ebb02ad9290411414d9f94e01f5` | `63fcc2b336575614080ad95fcf605559812aa5055527889776a6c5255bd00771` |
| 强制 `cluster1x4`（guard 内） | `6adedb57b9793bf2256e2a5ed4ca9da0b01297a3ff00664bb9a87a70cf1773c6` | 同上 |

强制路径 symbol 包含 `mg1x2_dsfirst16`，其命名顺序是 first1/main2；DS5 内部调度不会将名称
改成 `dsfirst5`。该路径不含 `_rolleddrain`，不能把 T1536 ELF 的 PC 或 ATT 统计套用于此 ELF。
原始命令、输出、两阶段 ELF 和本地复核按下面两个前缀保留在 `logs/`：
`reboot20261024_rotated_addr_scoped_t4608_strict_v1`、
`reboot20261024_rotated_addr_scoped_t4608_cluster1x4_strict_v1`。
复现时在固定环境与相应独立 cache 中运行 `verify_pin_strict_against_production_v1.py`
并添加 `--tokens 4608 --dynamic`；强制路径额外设置上述已有 env。

### pin1/scoped 的完全无效 M64 分组跳过：CPU gate（2026-10-24）

候选 `dsf5_pin1_empty_m64_scoped_v1` 基于 addr32/scoped reference，仅在
`SUPPORT_EXPLICIT_VGPR_PARTITION and compact_pin_gemm1` guard 内，为 `k_step` 的
FRONT/BACK 两组 WMMA 添加有效行判断。每个 wave 的 M128 分为两个 M64 分组：
分组第一行已经超过该 expert 的 `mn_oob` 时，跳过整组 WMMA；只要有一行有效，仍计算完整 M64。
guard 外以及 pin0 的原有调用保留，A/ScaleA/B 的输入 ring、DS/TDM、fence、输出 helpers 未改。
这利用的是允许跳过无效计算的约束，没有减少有效 token 的计算，也没有添加 E64/E65 specialization。

生成器 `tools/make_pin_empty_m64_scoped_v1.py` 针对实际 `k_step` AST 做 17,280 项检查：
12,960 项 guard 外完全一致，1,584 项保留部分有效分组的完整调用；覆盖 initialize、reuse load、
普通 load、issue 和 fence 组合。source SHA256 为
`2af717bee9a0f59e30440d3c468b759918bd4a07591ce6c30e3645e41ab152fa`。

CPU 编译 22.7534 s、rc0。实际 ELF 为
`dd44fff8fab75aedf6fe7a6f217b4f4cf3b9cb1825be70089a282a0287de9565`；
VGPR1024 /SGPR62 /LDS327680 B /private0 /spills0 与 reference 一致。
static executable /`s_wait_alu` /`s_set_vgpr_msb` 为 5592 /112 /449，reference 为 5724 /118 /750。
WMMA640、DS load720、TDM load24/store1 和数学转换操作 count 保留。

原始 ELF/ISA 下载后逐 machine word 复核。CFG 中确认 35 个真实分支 site，各可跳过
16 条 WMMA，跳过区域不含 DS/TDM/barrier；不是仅设置 EXEC=0 后继续发射 WMMA。
物理 DS/WMMA VGPR 经 MSB 解码，并在有界 CFG 中检查 DS completion 到 WMMA、
WMMA input retirement 到 DS overwrite，未发现该模型覆盖范围内的 register hazard。
该模型不证明全部 VALU/VMEM hazard 或 descriptor 语义，仍需 device 数值验证。

编译器把部分 fence block 放到 ISA 后部，线性文本中的非 DS fence 序列因此不同。
基于真实 branch targets 做 epsilon CFG 投影后，TDM 与非 DS fence 的路径事件语言一致，
复核涉及 44 个 state pairs、74 个 transitions。此检查忽略 branch condition，不能代替实际
routing replay；没有因静态文本顺序改变就判为同步变化，也没有假定所有 DS wait immediate 相同。
DS wait0 的 static sites 为 16，reference 为 18。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/compile_pin_vgpr_batch_v1.py \
  reboot20261024_pin_empty_m64_cpu_batch_v1 dsf5_pin1_empty_m64_scoped_v1
```

原始 CPU gate 为 `logs/reboot20261024_pin_empty_m64_scoped_v1.cfg_acceptance.json`；
本地独立重算为同前缀 `.cfg.local_review_v1.json` 和 `.fence_cfg.local_review_v1.json`。
CPU archive SHA256 为 `4fa70178806cbff559fd0166b41d7215f8505885ec379c3af5860c4380929750`，
14 项任务文件逐项核 SHA256，不含 `.out`。此阶段尚无该候选的 GPU 正确性或性能成绩，
production 未采用；后续结果单独记录。

随后原入口 source-dependent call keys preparation 完成，实际 GEMM1 ELF 与 CPU 的
`dd44fff8…7de9565` 一致；GEMM2 使用原 pin1 ELF `a5e9c250…58375311`。
seed0 完整 MoE 的 22,020,096 B 与生产 DS5/pin0 零差异，输出 SHA256 为
`443e8c5959dc9513a655d5c9b59b21c93c738df5807cda3123560fbc8e47b935`。
两者 `logits_diff=3.6105513278025183e-6`、`rel_l2=0.0026872099842876196`，原 precision 通过。
preparation 不记录性能；当前只完成 seed0，不能宣称该新候选已完成多 seed/dynamic 验证。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_pin_flydsl_v3.py \
  prepare dsf5_pin1_empty_m64_scoped_v1 \
  reboot20261024_dsf5_pin1_empty_m64_scoped_v1_pin0_prepare_v1 \
  reboot20261024_dsf5_pin1_empty_m64_scoped_v1_pin0_prepare_v1 \
  --base-pin0 --artifact=reboot20261024_dsf5_pin1_empty_m64_scoped_v1_cpu_v1
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/compare_pin_series_v1.py \
  reboot20261024_pin_empty_m64_normal_screen_v1 dsf5_pin1_empty_m64_scoped_v1
```

复现需更换未占用的 labels；三组性能序列实际全部计时 pin1，不是 pin0/pin1 对照。
preparation 原始证据及本地复核为 `logs/reboot20261024_dsf5_pin1_empty_m64_scoped_v1_pin0_prepare_v1*`，
保存在上述 `d210c7e9…7b9579c` archive 中。性能结果与最终决策另记。

### M64 分组跳过的九轮性能结果：明确回退，不采用（2026-10-24）

按 reference 前三轮、M64 候选三轮、reference 后三轮运行原 `--scenario kernel` 入口。
全部计时 pin1，attempt=1；没有按结果挑选或删除读数。九轮原 precision gate、实际
source-dependent call keys/ELF、测试前后全机 ownership 均通过。

| 组 | GEMM1 三轮 μs | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---|---:|---:|
| addr32/scoped reference 前 | 80.2791 /82.1073 /81.5211 | 81.5211 | 61.9064 |
| `dsf5_pin1_empty_m64_scoped_v1` | 84.2429 /84.5507 /84.3001 | 84.3001 | 61.5036 |
| addr32/scoped reference 后 | 80.8694 /80.7346 /80.8001 | 80.8001 | 61.4952 |

候选相对两侧 reference 分别回退 3.40893% /4.33168%，本实现明确不采用。
减少完全 padding 分组的 WMMA 和 static MSB/wait 并没有提升整个 dispatch 的性能。
本组性能测量结束时尚无匹配 ATT；随后四 SIMD ATT 已确认新增条件 basic blocks
拆开了原 DS/WMMA 交错，具体证据见后面的补充。不能把回退独立归因于某一类指令。

性能证据为 `logs/reboot20261024_pin_empty_m64_normal_screen_v1.comparison.json`，
三组同名前缀 summary、manifest、ownership、完整实际 ELF 和原表均保留。
下载后在本地重算三组中值及双侧比例，四项 `.local_review_v1.json` 全部通过。
archive SHA256 为 `4e187c77cea712e2c72af14a97671b87fbf0044c31f80c4abac59af3276d46cd`，
136 项任务文件逐项核 SHA256，不含 `.out`。production 已恢复为原 DS5/pin0 source
`e53ada2b…06f93e`；充分验证的最佳 reference 仍为 addr32/scoped pin1，约 81.1 μs。

### `s_wait_alu` /`s_set_vgpr_msb` 的性能影响分析进度（2026-10-24）

两类指令已分析，但尚未精确分离独立动态成本。匹配 reference ELF 的 mainloop
三处含 `s_wait_alu` 的直线 timestamp gap 中值为 17 cycles；最大 1247 cycles 区间
同时含 prologue barrier，不能全部计入 ALU wait。decoder 不单独报告两类指令的
动态事件，不能据此认为它们未执行，也不能将 28.21821% control spans 视为可删除成本。

硬件文档的语义、具体 PC 和已验证限制见
`my_code/moe_gemm1_hyg_dsr1_best_thread_trace_analysis.md` 同名章节。
匹配 M64 ELF 的四 SIMD ATT 已完成，按实际 896 /1792 WMMA active waves 分组，
ELF、decoded PC、事件和尾部两个 barrier 的全部原校验通过。用户提出的 start gap
减前条 latency 指标也已全量计算，结果见下方及 thread trace 文档。


### M64 匹配 ATT 与 latency 残差的补充（2026-10-24）

M64的64个active waves中，21个执行896 WMMA、43个执行1792 WMMA，mainloop中值
分别为20004 /20589 cycles；reference的64个完整计算waves为17331.5 cycles。
仅比较完整1792 WMMA waves，每32 WMMA作为一个K128组：reference的3584组内
可见DS load中值为31，而M64的2408组内全部为0，显示原DS/WMMA交错被拆开。
这是匹配trace的调度证据，不是某条control指令的独占成本。

按照用户建议，逐个真实forward fallthrough区间计算：

```text
residual = next_visible_start - previous_visible_start - previous_decoded_latency
```

SDK事件为`[time, category, stall, duration, pc_index]`，latency使用`event[3]`。
保留负残差；中间两类control同时出现计为`both`，不重复相加；排除隐藏branch或其他opcode。
reference mainloop的wait-only /MSB-only /both残差中值为0 /0 /3 cycles，若干wait热点
仍有16-cycle正残差；prologue barrier附近1247-cycle gap扣除前条latency后中值为3 cycles。
全量结果、原始证据和硬件语义见`my_code/moe_gemm1_hyg_dsr1_best_thread_trace_analysis.md`。

### 三处 `VA_VDST(0)→(1)` 的独立 ISA 对照（2026-10-24）

只修改reference `deea50e0…ceda0488` 的`0x32d4 /0x5218 /0x61bc`三个立即数，
`BF880F9F→BF881F9F`；其余ELF字节、TDM机器码、指令数量、有效计算和GEMM2均保持一致。
实验GEMM1 ELF为`97d1c16c3e8536b6e8d5cacc3aa3e4d647173a8016f5248403c0cd8c56d97346`。
它使用私有cache、原source-dependent keys和`FLYDSL_RUNTIME_RUN_ONLY=1`。
此改动只是依赖假设的ISA实验，尚无通用shape安全证明，未落入生产FlyDSL源码。

原单测入口、数据、warmup5、iters20、seed0、A preshuffle1、pin1均保持原样。
每轮前等待全机空闲，之后检查ownership和settled idle；每个child限时120秒。

| 顺序 /组 | GEMM1 三轮 μs | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---|---:|---:|
| 正向 reference 前 | 80.8042 /80.4941 /81.0023 | 80.8042 | 61.2086 |
| 正向 ISA wait1 | 80.3847 /81.4812 /80.4268 | 80.4268 | 61.5480 |
| 正向 reference 后 | 81.0384 /82.2864 /82.0335 | 82.0335 | 60.8671 |
| 反向 ISA wait1 前 | 81.4033 /80.7306 /80.6398 | 80.7306 | 61.5313 |
| 反向 reference | 80.5365 /81.2617 /80.9203 | 80.9203 | 61.3499 |
| 反向 ISA wait1 后 | 80.5977 /81.9155 /80.4818 | 80.5977 | 61.5395 |

正向相对两侧改善0.46705% /1.95859%，reference有明显漂移；反向改善
0.23443% /0.39866%，量级较小。18轮原precision gate均通过，`logits_diff=3.61055e-6`。
每轮actual keys/ELF、原表和ownership均校验通过。本实验只完成seed0原门槛验证，
未称为逐字节、多seed或dynamic通用正确性证明，也未称为已经落地的最优版。

首个正向v1因私有cache遗漏四个非GEMM entries，在GEMM1前cache miss退出，
保留失败记录，不纳入性能结论。补入原固定workload的四个原始entries后完成正向v2，
非GEMM源码与ELF没有修改。

实际运行的命令为：

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_addr32_wait1_host_v2.py \
  reboot20261024_addr32_wait1_isa_forward_v2
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_addr32_wait1_host_v3.py \
  reboot20261024_addr32_wait1_isa_reverse_v1 candidate base candidate
```

重跑需要新label及对应`.plan.json`；这些实验工具校验保留source SHA`e53ada2b…06f93e`，
wrapper结束后精确恢复文件。实验期间用于实际call keys的reference source SHA为
`2273b143…56854de17d`，该源码本身与ISA实验的立即数修改不同。

正向archive SHA256为`a9e40cc9ea4290a3b2c33b40f8abc44ebc2ba3520c31259ecb54e63ee49550d9`，
182项逐项校验；反向及low-AB CPU archive SHA256为
`5b83df57e403931fa3f887e7f4910292c8cf01ceba405adfc3831bfd10d0d924`，136项逐项校验。
两份本地`.local_review_v1.json`重算全部中值和比例，均不含`.out`。

### 小型 VGPR 布局 CPU 筛选（2026-10-24）

均基于同一reference，保留compact guard、输入ring、wave布局和有效计算。
五项均在约17秒CPU编译阶段返回134，未提交GPU，未记录性能，也未采用。

| 候选 | 变化 | 编译诊断 |
|---|---|---|
| `dsf5_pin1_ab32_addr128_scoped_v1` | A/B实际槽32:159和160:287，地址128:135 | 与precolored `VGPR128`冲突 |
| `dsf5_pin1_ab32_pitch96_addr224_scoped_v1` | A/B实际槽32:159和128:255，地址224:231 | 与precolored `VGPR224`冲突 |
| `dsf5_pin1_ab768_pitch96_addr32_scoped_v1` | 保持高位A/B，physical slot pitch128改96 | `overlapping live intervals in explicit register storage` |
| `dsf5_pin1_addr32_rmem_scoped_v1` | 每次calc使用单元素native RMem pin，仍为v32:39 | `overlapping live intervals in explicit register storage` |
| `dsf5_pin1_addr32_rmem_shared_scoped_v1` | 两份四元素address RMem槽复用，仍为v32:39 | `overlapping live intervals in explicit register storage` |

声明范围不重叠不足以证明实际live intervals合法；这些诊断不构成硬件不支持的证明。
两份address RMem槽复用版在17.5426秒返回134，也未进入设备测试。
后三项CPU记录archive为`reboot20261024_pin_small_followup_cpu_evidence_v1.pin_evidence.tgz`，
SHA256为`7ad1421f01fb7e78e05c8e579a680e2dc8e953947d4ca50043a07ec6ac63c21e`；
8项任务记录及manifest均在本地复核，不含`.out`。
wait1的残差后处理已经补齐工具、复用同一次capture恢复，完整结果见下一节。


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

### compact pin1 内两次 steady iteration 展开（2026-10-24）

候选`dsf5_pin1_steady_pair_scoped_v1`基于已验证的reference
`dsf5_pin1_rotated_addr_scoped_v1`；相对reference只增加12行、删除2行。
保留首个inline-zero iteration，将后续steady iterations每两次展开为一个runtime loop，
奇数尾项单独执行。iteration函数体、K-tile和buffer顺序、输入ring、fences及wave布局不变。
只在原compact pin1 guard内启用，未增加persistent、w2x4或OPTX pipeline。
源SHA256为`d41ab961628a57cd022ef39a1acf0443e79ad1ae9368fd6f73055281535eb0e3`。

实际steady AST在7740组参数中核对294624个有序事件，数量和顺序相同；
该检查不能替代compiled CFG及动态fence验证。
CPU编译20.6255秒，完整GEMM1 ELF为
`a99c5398faa42e4d61dbfee6011ee2ea6d921f69b26b0750ce64926affd12866`。
VGPR1024、SGPR58、LDS327680 B，private0、spills0；reference的SGPR为62。
静态executable /`s_wait_alu` /`s_set_vgpr_msb`为7462 /160 /1110，
reference为5724 /118 /750。静态展开增加的数量不能直接当作动态执行成本。

原入口准备后，candidate实际ELF与CPU artifact相同，GEMM2 pin1仍为
`a5e9c2504863d38c85e39acdc3e28b35774a99f21957b517e338953558375311`。
seed0完整MoE的22,020,096 B输出与保留版零字节差异，SHA256为
`443e8c5959dc9513a655d5c9b59b21c93c738df5807cda3123560fbc8e47b935`。

正反向性能序列全部使用pin1、A preshuffle1、原单测及原数据；
每轮前全机空闲，之后检查ownership及settled idle，child限时120秒。
这里的reference源码为`2273b143…56854de17d`，不是pin0；
preparation名称中的`pin0`只表示准备阶段的比较对象，不能据此误读性能序列。

| 顺序 /组 | GEMM1 三轮 μs | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---|---:|---:|
| 正向 reference前 | 81.0846 /82.9569 /81.6539 | 81.6539 | 61.4785 |
| 正向 steady pair | 81.0573 /80.7198 /81.0698 | 81.0573 | 61.8327 |
| 正向 reference后 | 80.6841 /81.1205 /81.8666 | 81.1205 | 61.8556 |
| 反向 steady pair前 | 80.6860 /80.5890 /80.6923 | 80.6860 | 61.3012 |
| 反向 reference | 81.0105 /81.5253 /81.0506 | 81.0506 | 61.1936 |
| 反向 steady pair后 | 80.7323 /82.7627 /80.9516 | 80.9516 | 61.3982 |

正向两侧改善0.73064% /0.07791%，反向改善0.44984% /0.12215%。
十八轮原precision gate、actual keys/ELF、逐轮原表及空闲记录均在本地复核通过。
收益较小且各组存在波动，当前仅保留为实验候选，未替换充分验证的reference；
尚未完成该展开版的多seed、通用shape和matched ATT验证，未将80.6860 μs称为
已经落地的最佳源码版成绩，也未达到70 μs目标。

实际执行命令：

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/compare_pin_series_v1.py \
  reboot20261024_pin_steady_pair_normal_screen_v1 dsf5_pin1_steady_pair_scoped_v1
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/compare_pin_series_v1.py \
  reboot20261024_pin_steady_pair_normal_reverse_v1 dsf5_pin1_steady_pair_scoped_v1 --reverse
```

工具在结束后精确恢复保留source`e53ada2b…06f93e`；复现需要新label。
正向archive SHA256为`faaef892b1167740b4b6ab57e75a30ee033e83d124f19051e572c1df564d3730`，
173项；反向为`b86496f5079193a9e6b575a171dee0fc3a133059bae0e9ebff454f9506beed53`，
137项。均不含`.out`，各项SHA及完整中值/比例已在本地重算。

### LDS 地址的四元素约束与 ScaleA/ScaleB 先算（2026-10-24）

两项独立候选均基于reference `2273b143…56854de17d`，只修改原compact pin1
guard内的`calc_lds_addr`约束块，实际物理地址仍为current `v32:35`、next `v36:39`。
输入pipeline、wave布局、WMMA计算、源码fences及guard条件没有变化。

| 候选 | 源改动 | 源SHA256 | GEMM1 ELF SHA256 |
|---|---|---|---|
| `dsf5_pin1_addr32_vec4_scoped_v1` | 四个独立scalar约束改为一个四元素tied vector约束；+9/-9行 | `9548c00d88601b6f8fb901628671253ecee53a3c69250d80bb8556a11607f8bc` | `79620299782ad0bd10db03bdef34a2533a7097a09c698a6fa7c62c9d4997eec6` |
| `dsf5_pin1_addr32_scale_first_scoped_v1` | 约束顺序由A/B/ScaleA/ScaleB改为ScaleA/ScaleB/A/B，返回值及寄存器映射保持原样；+3/-2行 | `862e74daf5cfafa1606085d77a150110d8560bb8bfaedd8279f713db27ea6e9c` | `f6f9130fc56a66c091ca5ff3f3e9869231483424873391acd50ca7a16d17252e` |

两项CPU编译分别18.7154 /18.7978秒，均为VGPR1024 /SGPR62 /LDS327680 B，
private0、spills0。完整ELF和ISA机器码已在本地逐项绑定并复核：

| artifact | 静态executable | `s_wait_alu` | `s_set_vgpr_msb` | 静态WMMA /TDM /DS load |
|---|---:|---:|---:|---|
| reference | 5724 | 118 | 750 | 640 /24 /720 |
| 四元素约束 | 5714 | 121 | 740 | 640 /24 /720 |
| ScaleA/ScaleB先算 | 5722 | 113 | 750 | 640 /24 /720 |

ScaleA/ScaleB先算版完整静态fence序列与reference相同。
四元素约束版的TDM/barrier/store fence序列也相同；全部静态fence差异集中在
五处编译器生成的DS等待：原单条`s_wait_dscnt 0x6`分别变为两条较大立即数的
DS waits，合计DS waits多5条。该变化不能单独证明动态等待缩短或完整依赖安全。
静态分析原始结果为`logs/reboot20261024_pin_address_small_static_local_v1.json`。

两项均已完成原入口keys准备，actual GEMM1 ELF与对应CPU artifact一致，
GEMM2 pin1仍为`a5e9c250…58375311`。seed0完整MoE输出22,020,096 B逐字节相同，
原precision gate通过；两项完整输出SHA均为`443e8c59…c8e47b935`。
尚未完成多seed、guard外shape及matched ATT验证。原入口正向初筛十二轮已完成：

| 顺序 /组 | GEMM1 三轮 μs | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---|---:|---:|
| reference前 | 80.7176 /81.7972 /81.4809 | 81.4809 | 61.3812 |
| 四元素约束 | 80.5257 /81.3081 /80.3383 | 80.5257 | 61.7103 |
| ScaleA/ScaleB先算 | 81.9512 /80.4416 /81.0234 | 81.0234 | 60.8754 |
| reference后 | 83.5914 /80.5828 /80.9245 | 80.9245 | 61.3732 |

每轮前全机空闲，之后ownership/settled idle校验通过；原precision gate通过，
actual ELF、keys及全部原表在本地复核。四元素约束相对两侧改善1.17230% /0.49281%，
进入反向九轮复测；ScaleA/ScaleB先算相对两侧为+0.56148% /-0.12221%，未采用。
当前四元素约束仅为待进一步验证的候选，未称为已落地的最佳版。

实际执行的正向性能命令：

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/compare_pin_series_v1.py \
  reboot20261024_pin_address_small_normal_screen_v1 \
  dsf5_pin1_addr32_vec4_scoped_v1 dsf5_pin1_addr32_scale_first_scoped_v1
```

正向原始记录archive SHA256为
`28e78c00cf0530c707419836a01ef0680d784028bcd540302fe5f6c23da51a78`，182项，
全部逐SHA校验并在本地重算四组中值与比例，不含`.out`。重跑需要新label；
工具结束后精确恢复保留source`e53ada2b…06f93e`。

CPU archive SHA256为`263975f2f12ce540a573dedd9525cb6d6a19a66bb2997d4e68df769f649529fe`，
26项；准备记录archive为`daf5e8ebf978f27a194107aa54bf7b7ee86e79adf3902631de9dfeb6b25bf571`，
47项。均已下载至本地逐SHA复核并重算原入口/完整MoE记录，不含`.out`。

### 四元素地址约束反向确认：没有稳定收益，暂不采用（2026-10-24）

反向固定次序为candidate /reference /candidate，三组各三轮，均为attempt=1。
actual pin1、A preshuffle1、原precision gate、两阶段ELF及前后ownership均通过。
每个child限时120秒；全部原表和九轮中值/比例已下载，在本地重新核算。

| 固定顺序 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 四元素约束前 | 80.5491 /81.3713 /80.6314 | 80.6314 | 61.2591 |
| addr32 reference | 81.2049 /81.6476 /81.2851 | 81.2851 | 61.3288 |
| 四元素约束后 | 80.9433 /82.0715 /81.6773 | 81.6773 | 61.4278 |

前组相对reference改善0.8042064%，后组回退0.4824993%。结合正向结果，
当前不能证明稳定收益，暂不采用四元素约束，也不把正向80.5257 μs称为新的正式最佳成绩。
没有追加这项的多seed、通用shape或matched ATT，不能宣称已完成这些验证。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/compare_pin_series_v1.py \
  reboot20261024_pin_addr32_vec4_normal_reverse_v1 \
  dsf5_pin1_addr32_vec4_scoped_v1 --reverse
```

反向archive为`reboot20261024_pin_addr32_vec4_reverse_evidence_v1.pin_evidence.tgz`，
SHA256为`d10cc527917fb1d57a60dfcb5cc393f0e9f56a611cb3d8b22c3fa130c7dfc775`；
137项中136份文件逐SHA复核，不含`.out`。wrapper已恢复生产source
`e53ada2b…06f93e`，测试后无GPU/KFD测试进程残留。

接下来准备两个独立候选，仍基于完整验证的`2273b143…56854de17d`，仅修改
`calc_lds_addr`的原compact pin1 guard内代码：

- `dsf5_pin1_addr32_vec2_scoped_v1`：用两个tied vector约束分别固定A/B和ScaleA/ScaleB，
  物理槽仍为current `v32:35`、next `v36:39`；+8/-5行。
- `dsf5_pin1_addr32_i32_scoped_v1`：用等价32-bit LDS地址表达式替换Index表达式的末尾转换，
  保留四个scalar pin和原寄存器映射；+6/-1行。地址的设备精度与compiled ELF尚待核验。

生成器为`tools/build_pin_address_pair_i32_features_v1.py`；仅已完成源码语法及变更范围检查，
尚无CPU编译或GPU性能结果。输入pipeline、wave布局、fences、guard条件和生产文件均未改变。

### 两个 vector 地址约束与32-bit地址表达式：CPU及seed0通过（2026-10-24）

两项均编译成功。完整ELF和全部disassembly machine words已下载，在本地逐项复核。
VGPR1024 /SGPR62 /LDS327680 B均保持原值，private和spills均为0。

| 候选 | CPU编译秒 | static executable | `s_wait_alu` | `s_set_vgpr_msb` | GEMM1 ELF SHA256 |
|---|---:|---:|---:|---:|---|
| reference | — | 5724 | 118 | 750 | `deea50e0…ceda0488` |
| `dsf5_pin1_addr32_vec2_scoped_v1` | 18.9023 | 5719 | 127 | 740 | `bcad51cacb646f07008939ae9f0debab0ef30bd3dab0a7716dd681de448a66ab` |
| `dsf5_pin1_addr32_i32_scoped_v1` | 18.7716 | 5701 | 118 | 750 | `c5296627d336613a8a8580ec712c3d04766081e8b9022b0012333e13fb8001d2` |

两个候选均通过原入口source-dependent call keys准备：actual GEMM1 ELF与CPU artifact相同，
GEMM2为原pin1 ELF `a5e9c250…58375311`。seed0完整MoE与保留版逐字节相同。
此准备阶段不计性能，尚未完成多seed/guard外验证或matched ATT。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/compile_pin_vgpr_batch_v1.py \
  reboot20261024_pin_address_pair_i32_cpu_batch_v1 \
  dsf5_pin1_addr32_vec2_scoped_v1 dsf5_pin1_addr32_i32_scoped_v1
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_pin_flydsl_v3.py \
  prepare dsf5_pin1_addr32_vec2_scoped_v1 \
  reboot20261024_dsf5_pin1_addr32_vec2_scoped_v1_pin0_prepare_v1 \
  reboot20261024_dsf5_pin1_addr32_vec2_scoped_v1_pin0_prepare_v1 \
  --base-pin0 --artifact=reboot20261024_dsf5_pin1_addr32_vec2_scoped_v1_cpu_v1
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_pin_flydsl_v3.py \
  prepare dsf5_pin1_addr32_i32_scoped_v1 \
  reboot20261024_dsf5_pin1_addr32_i32_scoped_v1_pin0_prepare_v1 \
  reboot20261024_dsf5_pin1_addr32_i32_scoped_v1_pin0_prepare_v1 \
  --base-pin0 --artifact=reboot20261024_dsf5_pin1_addr32_i32_scoped_v1_cpu_v1
```

CPU archive为`reboot20261024_pin_address_pair_i32_cpu_evidence_v1.pin_evidence.tgz`，
SHA256为`993cb0bc8245fddc258c1743dab7e68b0916d9c9db5e34be6e86bb434e3a957b`；
24项含23份文件，逐SHA和ISA machine words均在本地复核，不含`.out`。

### 仅跳过无效BACK M64：源码与CPU编译（2026-10-24）

针对前次FRONT/BACK均加分支而打断DS/WMMA交错的回退，新增独立候选
`dsf5_pin1_empty_back_m64_scoped_v1`。基于原reference，只在compact pin1 guard内、
BACK整体无有效行时跳过该组WMMA；FRONT始终保持原调用，部分有效的BACK仍完整计算M64。
相对reference仅+5/-1行；输入和输出helpers、rmem、ring、fences、wave布局未改变。

源SHA256为`27ca8feda5fd88cfa7f9d825270229e0613f44f1583186dac50546f9b5ed53d2`。
实际k_step AST完成17,280项检查，guard外12,960项完全相同，部分有效分组保留。
这不替代设备精度和compiled调度验证。

CPU编译20.3955秒、rc0。GEMM1 ELF为
`088a8bca1275f98a364d7dec4c25e6c93a0574ba068c591497d1afe3ffe9584e`；
VGPR1024 /SGPR62 /LDS327680 B、private0、spills0。
static executable /`s_wait_alu` /`s_set_vgpr_msb`为5475 /113 /449。
仍须检查真实CFG及设备结果，不能从静态MSB数量下降推断性能收益。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/compile_pin_vgpr_batch_v1.py \
  reboot20261024_pin_empty_back_m64_cpu_batch_v1 dsf5_pin1_empty_back_m64_scoped_v1
```

生成器为`tools/make_pin_empty_back_m64_scoped_v1.py`，复用已有实际k_step模型，
明确保留FRONT所有调用；没有加入persistent、w2x4或新的pipeline。

### 地址表达式正向十二轮：i32进入反向确认（2026-10-24）

四组均为attempt=1；每个性能child前全机空闲，之后ownership和settled idle通过。
原precision gate、实际两阶段ELF以及原表均在本地重新核验。

| 固定顺序 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| reference前 | 81.4220 /81.4303 /81.7087 | 81.4303 | 61.4004 |
| 两个vector约束 | 81.1459 /81.4811 /81.0149 | 81.1459 | 61.8285 |
| i32地址表达式 | 80.9307 /80.8191 /80.7261 | 80.8191 | 61.2444 |
| reference后 | 81.0614 /81.6245 /80.7220 | 81.0614 | 61.5734 |

两个vector约束相对两侧为+0.34926% /-0.10424%，暂不采用。
i32地址表达式相对两侧改善0.75058% /0.29891%，进入反向确认；这组结果尚不能称为
新的完整验证最佳成绩。其静态wait/MSB数量仍为118 /750，不能把性能差异归因于这两类
指令数量减少。两项seed0完整MoE输出22,020,096 B逐字节一致，输出SHA256为
`443e8c5959dc9513a655d5c9b59b21c93c738df5807cda3123560fbc8e47b935`。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/compare_pin_series_v1.py \
  reboot20261024_pin_address_pair_i32_normal_screen_v1 \
  dsf5_pin1_addr32_vec2_scoped_v1 dsf5_pin1_addr32_i32_scoped_v1
```

原始archive为`reboot20261024_pin_address_pair_i32_forward_evidence_v1.pin_evidence.tgz`，
SHA256为`b408747fadc7f94942770a8a9cecb9c720f3d7c458f024af5444cc09f13d6169`。
182项中181份文件逐SHA复核，四组原表和parent比例在本地重算，不含`.out`。
原入口准备archive SHA256为`a91307fdbd80e5c0c2f3a7a090a11ff74601565f806b98ad56c35bb354d0deb5`，
准备记录也已在本地复核。工具结束后恢复source `e53ada2b…06f93e`。

### BACK-only跳过分支的CFG提示与wait0修正版（2026-10-24）

无额外wait的BACK-only候选没有通过既有有界DS/WMMA CFG检查。在PC `0x5b50`、
`0x5b58`报告`DS overwrite before WMMA retirement`，涉及`v768:771`、`v772:775`；
之前的`0x5b48`为`s_wait_alu depctr_va_vdst(14)`。模型对conditional paths采用
overapproximation，不能据此直接断定硬件上必然发生race，也不能忽略提示提交GPU。
该候选没有GPU结果或性能数据。

其真实15个M64 skip sites均不跳过DS/TDM/barrier，TDM/非DS fence路径语言与reference
相同（44 state pairs /74 transitions）；这些检查不替代寄存器退休依赖检查。
失败证据archive SHA256为`063b87b797eb9e11f7a8f12028c85d5b06c16178301c722e0a643d72535bf472`，
17项已在本地逐SHA复核，不含`.out`，完整失败记录保留。

修正版`dsf5_pin1_empty_back_m64_wait0_scoped_v1`只在BACK整体无效的跳过分支，
于两侧`sched_barrier(0)`之间执行`s_wait_alu depctr_va_vdst(0)`；FRONT、有效BACK和
guard外执行路径保留。source SHA256为
`eaca3ca1f0ad97df50d48ccfb4c855198fb6781d8db02bdcff55fe84bea4585b`。
实际k_step AST共448项通过，数值事件与BACK-only候选相同，新增wait只在跳过分支。

CPU编译20.9506秒，rc0；VGPR1024 /SGPR62 /LDS327680 B、private0、spills0。
GEMM1 ELF SHA256为`0c2c9e0f23eb0d80ede37a43963e4cb77b25c26aab43030417934d19ac56fc9c`，
static executable /wait /MSB为5608 /133 /459。仍须复核真实CFG和设备精度；静态数量
不能作为动态性能结论。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/compile_pin_vgpr_batch_v1.py \
  reboot20261024_pin_empty_back_m64_wait0_cpu_batch_v1 \
  dsf5_pin1_empty_back_m64_wait0_scoped_v1
```

wait0修正版仍未通过同一个有界CFG检查：`0x5cfc /0x5d04`的A槽
`v768:771 /v772:775`以及`0xb5c0 /0xb5c8`的Scale槽`v13 /v11`被报告未退休覆盖。
在前两处，ISA中wait0位于两条A load之后（`0x5d0c`），之前只有`depctr_va_vdst(14)`；
因此在该位置添加wait0仍未覆盖检查提示的窗口。没有削弱检查或提交GPU。

CPU与失败CFG archive为`reboot20261024_pin_empty_back_m64_wait0_cpu_evidence_v1.pin_evidence.tgz`，
SHA256为`c04d41c27185797b3412b33170356a79ac8a7856b1eeff4005c754e5da6cb9c8`，
16项中15份文件逐SHA复核，CPU artifact机器码也已在本地重新绑定，不含`.out`。
该版没有设备结果或性能数据。

后续独立候选`dsf5_pin1_empty_back_m64_loadwait0_scoped_v1`基于无额外wait的BACK-only版，
在compact pin1 guard内的`load_lds_data`入口添加两侧`sched_barrier(0)`包围的
`s_wait_alu depctr_va_vdst(0)`，覆盖有效和无效BACK路径的LDS loads。原guard、数值代码
及输入ring保持原样；比BACK-only版新增7行。source SHA256为
`4f4d46d0b0c2042bc08e1cde2fe2ea241c5680b5bfd7b627eca62d1cdb225278`。
仅完成本地源码范围和语法检查，CPU编译、真实CFG与设备精度仍待验证，不能视为修复完成。

### i32地址表达式反向九轮通过（2026-10-24）

固定次序为candidate /reference /candidate，各三轮，均为attempt=1。
每轮全机空闲检查、测试后ownership/settled idle、原precision gate及两阶段ELF绑定通过。

| 固定顺序 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| i32候选前 | 80.6187 /80.7960 /80.5745 | 80.6187 | 61.3688 |
| addr32 reference | 80.9876 /81.0973 /82.6974 | 81.0973 | 61.3393 |
| i32候选后 | 80.5554 /80.7093 /80.4310 | 80.5554 | 61.0292 |

两侧分别改善0.5901553% /0.6682097%。加上正向，四项相邻比较均有收益，范围
0.29891%～0.75058%。原始中值和比例已在本地从全部九轮原表重新计算；没有筛选最低单轮。
仍待strict、guard外ELF及matched ATT验证，尚未替换生产source。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/compare_pin_series_v1.py \
  reboot20261024_pin_addr32_i32_normal_reverse_v1 dsf5_pin1_addr32_i32_scoped_v1 --reverse
```

archive为`reboot20261024_pin_addr32_i32_reverse_evidence_v1.pin_evidence.tgz`，
SHA256为`846acdb894ae6273f0331e1cd8ecf5cd137889a33ddd64509864af70f40ad50c`。
137项中136份文件在本地逐SHA复核，不含`.out`；结束后生产source精确恢复。

### LDS load入口wait0的BACK跳过版：CPU依赖检查通过（2026-10-24）

`dsf5_pin1_empty_back_m64_loadwait0_scoped_v1`编译20.6613秒、rc0。
GEMM1 ELF SHA256为`6df68818c3a8106a23eb748d88dda590dffded2b8f7ac86693085bf538a4f0b6`，
static executable /wait /MSB为5495 /133 /449。
VGPR1024 /SGPR62 /LDS327680 B、private0、spills0，资源没有变化。

与前两个BACK-only候选不同，此版通过未削弱的原有界DS/WMMA模型：5495条可达指令，
8523次transfer，15个真实M64 skip sites，DS readiness和WMMA-to-DS overwrite提示均为空。
完整ELF机器码、physical MSB寻址、静态protected operations及barrier数量均绑定并核验。
该模型仍不覆盖所有VALU/VMEM/descriptor依赖，结果不能替代设备正确性或性能验证。
目前尚无该版GPU结果。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/compile_pin_vgpr_batch_v1.py \
  reboot20261024_pin_empty_back_m64_loadwait0_cpu_batch_v1 \
  dsf5_pin1_empty_back_m64_loadwait0_scoped_v1
# 以下检查在dsr1_ep4内执行，child限时120秒。
timeout --signal=TERM --kill-after=5s 120s python3 -u \
  .codex_tmp/yadai_gemm1_param_20261021/tools/audit_pin_empty_back_m64_loadwait0_cfg_v1.py
```

### i32地址候选的strict与guard外验证完成（2026-10-24）

`dsf5_pin1_addr32_i32_scoped_v1`源SHA256为
`f3c77d2aaa4698c80eb82b5d377c4f7b529247fc24c5fffce401d9a6f1fb21ec`；
实际GEMM1 ELF为`c5296627d336613a8a8580ec712c3d04766081e8b9022b0012333e13fb8001d2`。
保持四个scalar tied pin，仅把compact pin1 guard内的LDS地址表达式改为等价i32计算。
正向候选中值80.8191 μs，反向两组80.6187 /80.5554 μs，四项相邻改善
0.29891%～0.75058%；不将最低组单独作为正式成绩。距离70 μs目标仍约10.6～10.8 μs。

五项strict记录已下载并在本地复核：E64 seed0/seed1 dynamic、seed2的完整MoE输出
各22,020,096 B零字节差异；K4096 dynamic输出12,582,912 B零差异；bias输出
22,020,096 B零差异。seed0原精度指标为logits_diff `3.6105513278025183e-6`、
rel_l2 `0.0026872099842876196`。K4096实际走原t64 fallback，bias实际走非量化输出的
guard外路径，不能把这两项称为compact guard内验证。

11项guard外重新编译的完整ELF与已验证baseline逐字节一致：pin0目标、GEMM2 pin0/pin1、
原A、FP8、t64、w1x4、b3、GELU、persistent、非量化输出。
GEMM2 pin1 ELF保持`a5e9c2504863d38c85e39acdc3e28b35774a99f21957b517e338953558375311`。
strict archive SHA256为`00aa5c4948b43bf0a288dc0e98baf841833ee24e8539dd8616382782b57d6db6`，
scope archive为`ee5bd8cd92e23fdcd365178a6bcb3e7b7837ff2b9fc62f230a0ba0b8c5a1108f`，均不含`.out`。

该候选尚未落入生产源码，matched ATT待采集。E65仍沿用分支原t64/w1x4 fallback，
本轮未追加i32版E65固定shared routing和性能验证，不能宣称满足历史E64/E65共用同一GEMM1要求。

### BACK M64/loadwait0的设备验证与性能回退（2026-10-24）

安全候选`dsf5_pin1_empty_back_m64_loadwait0_scoped_v1`的seed0完整MoE输出
22,020,096 B与保留版零字节差异，原精度门槛及实际两阶段ELF绑定通过。
在所有GPU空闲条件下完成固定正向九轮，全部attempt=1；每轮后ownership与settled idle通过。

| 固定顺序 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| reference前 | 80.9561 /81.2766 /81.0403 | 81.0403 | 61.8411 |
| BACK M64/loadwait0 | 83.1634 /83.8003 /83.5093 | 83.5093 | 60.9703 |
| reference后 | 81.2744 /81.1730 /81.8057 | 81.2744 | 61.4827 |

相对两侧分别回退3.04663% /2.74982%，因此弃用，没有追加广泛strict/scope或ATT。
静态MSB数量减少不能作为性能收益证据。
性能archive SHA256为`b5ea6042adde4374b6910c27cb2c2f820469575750d3638d2cd09495cf448494`，
137项中136份文件逐SHA复核，三组原表及parent比例在本地重算，不含`.out`。

### 更细的BACK M16跳过：粒度纠正与CPU检查（2026-10-24）

逻辑`WMMA_M=16`、FP4 `WMMA_N=32`；128×128 warp tile对应8×4 WMMA，
每个逻辑M16 row group包含4条WMMA。此前尚未编译的M32标签与模拟配置不准确，已纠正为
`dsf5_pin1_empty_back_m16_loadwait0_scoped_v2`，旧准备记录保留，不把旧模拟称为真实M32验证。
此版只跳过整块无效的BACK M16，部分有效M16、全部FRONT、原guard和load fences保持。
实际k_step AST的16,576组检查通过；source SHA256为
`63f315e64534a0d904162961b97b28741c05bd0e3b3db35d2c5d9786074c32a0`。

重新核对每槽A/B寄存器容量：A为8×8=64 VGPR，B为4×16=64 VGPR，合计128，
slot pitch为128；两槽实际范围v768:895和v896:1023。不是每槽96 VGPR。
前文低地址与pitch96实验表的实际范围已纠正，编译器失败原记录保留；pitch96会造成实际槽重叠。

CPU编译22.0853秒、rc0；VGPR1024 /SGPR62 /LDS327680 B /private0 /spills0。
实际GEMM1 ELF为`d3160de0f057c0e5d036fd0e5955d5a260ceef9d14e9e6ed90a3c9cf7499bf01`，
static executable /wait /MSB为5877 /133 /569。
未削弱的原有界DS/WMMA模型通过：5877条可达指令、9694次transfer、75个真实M16 skip sites，
DS readiness和WMMA-to-DS overwrite提示为空。仅将skip detector识别单元改为实际4条WMMA，
没有修改hazard transfer或admission条件。此检查不替代其他依赖、设备精度或性能验证。
目前正在进行完整MoE设备对照，尚无该版性能结论。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/compile_pin_vgpr_batch_v1.py \
  reboot20261024_pin_empty_back_m16_loadwait0_cpu_batch_v2 \
  dsf5_pin1_empty_back_m16_loadwait0_scoped_v2
timeout --signal=TERM --kill-after=5s 120s python3 -u \
  .codex_tmp/yadai_gemm1_param_20261021/tools/audit_pin_empty_back_m16_loadwait0_cfg_v2.py
```

### BACK M16/loadwait0完整MoE与九轮性能：回退，弃用（2026-10-24）

seed0完整MoE输出22,020,096 B与DS5/pin0保留版零字节差异，原precision gate通过，
实际GEMM1 ELF绑定到`d3160de0…7499bf01`，GEMM2保持原pin1 ELF。
TDM/非DS fence路径语言对照通过（44 state pairs /74 transitions），没有跳过输入ring操作。
CPU、完整MoE、路径语言及以下全部九轮原始表已在本地复核。

| 固定顺序 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| reference前 | 81.6561 /81.5314 /80.7748 | 81.5314 | 61.6325 |
| BACK M16/loadwait0 | 82.5308 /82.7816 /83.1445 | 82.7816 | 61.9064 |
| reference后 | 80.0239 /81.7129 /80.7853 | 80.7853 | 61.6407 |

各组attempt=1，每轮前全机空闲、原精度门槛、真实两阶段ELF及每轮后ownership/settled idle通过。
相对两侧回退1.53340% /2.47112%，因此弃用，不追加该版broad strict/scope或ATT。
可以确认这个具体安全实现没有取得净收益，不能仅凭WMMA数量减少推断性能。
分支与wait的各自动态成本尚未对这版采集ATT，不能独占归因。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/compare_pin_series_v1.py \
  reboot20261024_pin_empty_back_m16_loadwait0_normal_screen_v2 \
  dsf5_pin1_empty_back_m16_loadwait0_scoped_v2
```

CPU/prepare archive为`reboot20261024_pin_empty_back_m16_loadwait0_preparation_evidence_v2.pin_evidence.tgz`，
SHA256 `70d4d66648bea13e3cb42883015e0fc7740589ca5c3b17337d578d12fdcb6316`，38项。
性能archive为`reboot20261024_pin_empty_back_m16_loadwait0_forward_evidence_v2.pin_evidence.tgz`，
SHA256 `71d80e7d22c9891da35d595701600f0287a87b02479d6e372a9b96ab17a6f3ac`，137项。
全部逐SHA校验并在本地重算三组中值与parent比例，均不含`.out`；生产源码精确恢复。

### ScaleA LDS offset的两行i32候选准备（2026-10-24）

在i32地址最佳候选基础上，`dsf5_pin1_addr32_i32_sa_offset_scoped_v1`只在原compact pin1
guard内将`load_sa`的runtime `kt`先转换为`fx.Int32`，再计算最终由`fx.Int32(off)`消费的
LDS offset。所有相关表达式均为整数加法/乘法，提前按2^32取模与最终取模等价；
不改变global address、ScaleA布局、数据、输入ring、fences或输出。
相对父候选仅+2行，源SHA256为
`0f996d26f2034a6560e321ddf8d79e5c36811e90fbc222970ce605a0c1de0bec`。
仅完成源码范围/语法检查并开始CPU编译，尚无设备正确性或性能结论，不采用生产。

### i32最佳候选的matched ATT及本地复核完成（2026-10-24）

`dsf5_pin1_addr32_i32_scoped_v1`已通过all-SIMD、iteration28的原workload ATT，
实际GEMM1 ELF为`c5296627d336613a8a8580ec712c3d04766081e8b9022b0012333e13fb8001d2`。
四SIMD共64个active waves，每wave执行1792 WMMA；实际两阶段call keys、ELF机器码与
decoded PC/ISA绑定通过，原精度门槛通过。profile期间的计时不作为正常性能数据。

按`next_visible_start - previous_visible_start - event[3]`逐区间复算；保留负残差，
排除隐藏branch，两类control同时出现只记为`both`，不重复累计。

| mainloop control区间 | 次数 | reference正残差合计cycles | i32正残差合计cycles |
|---|---:|---:|---:|
| wait-only | 7040 | 43857 | 40781 |
| MSB-only | 76928 | 129159 | 129122 |
| both | 1536 | 4608 | 4608 |

四处地址计算到DS load的区间各有368个样本，前条decoded latency中值均为1 cycle：

| reference PC | i32 PC | start gap中值cycles | 残差中值cycles |
|---|---|---:|---:|
| `0x32c8→0x32d8` | `0x32a4→0x32b4` | 17→16 | 16→15 |
| `0x426c→0x427c` | `0x4224→0x4234` | 17→16 | 16→15 |
| `0x520c→0x521c` | `0x51a0→0x51b0` | 17→16 | 16→15 |
| `0x61b0→0x61c0` | `0x6120→0x6130` | 17→16 | 16→15 |

i32前条为`v_dual_add_nc_u32 v38, s14, v1 :: v_dual_add_nc_u32 v39, s14, v4`，
后条为对应`ds_load_2addr_b32 ... v34 offset0:128 offset1:160`，中间wait0保留。
static wait/MSB仍为118/750，executable由5724降至5701。MSB-only残差中值仍0、P95=4，
36713个负残差说明这些区间存在重叠，不能将全部正残差归因为MSB的独占stall。
这批累计wave区间不能直接换算dispatch可节省比例；正常性能仍以之前完整正反向组
80.8191、80.6187、80.5554 μs中值及四项相邻改善0.29891%～0.75058%为准。

capture目录为`att/reboot20261024_pin_addr32_i32_att_v1/`。
decoded archive SHA256为`2d56ed44392389d6b1f21e6ec69bff21b62312a3e034dd7546d2ffe1e2b5f9c7`，
138项regular members逐SHA验证，不含`.out`。phase、wait/MSB和gap报告均从本地事件复算并
与远端相同；JSON整数键按字符串规范化后比较，没有修改原始事件或指标。
复核记录为`logs/reboot20261024_pin_addr32_i32_att_v1.local_trace_review_v1.json`，`passed=true`。
此前“matched ATT待采集”的状态已由本节更新；生产源码仍为`e53ada2b…06f93e`，尚未正式采用候选。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_pin_candidate_att_v1.py \
  dsf5_pin1_addr32_i32_scoped_v1 \
  reboot20261024_dsf5_pin1_addr32_i32_scoped_v1_pin0_prepare_v1 \
  reboot20261024_pin_addr32_i32_att_v1
```

### ScaleA提前i32未改变完整ELF，stage-cache v1仅编译失败（2026-10-24）

ScaleA两行候选CPU编译19.0190秒、rc0，完整GEMM1 ELF仍为`c5296627…f8001d2`，
static executable/wait/MSB为5701/118/750，和i32父候选逐字节相同。
编译器已经简化了这组等价表达式，没有提交GPU或记录性能，不增加生产源码。

`dsf5_pin1_addr32_i32_stage_cache_scoped_v1`源SHA256为
`b2b047c8459437f83cf7ca1115e142dd19040bb9ad4640a6a262c6458ca5a7fe`，相对i32父候选+8/-1行。
CPU编译15.5601秒、rc1：runtime `kt % num_buffers`得到`ArithValue`，不能直接索引
Python列表`lds_stage_addrs[slot]`，报`TypeError: list indices must be integers or slices, not ArithValue`。
这属于DSL实现错误，不能当成硬件不支持的证据。没有valid ELF、GPU dispatch或性能数据。
v1失败记录完整保留；另建v2使用显式`fx.select`，编译与设备验证仍待完成。

两项CPU实验的archive为
`reboot20261024_pin_i32_sa_and_stage_cache_cpu_evidence_v1.pin_evidence.tgz`，
SHA256为`0676c408df413fb55924a12495fec1a9f709962966abc9b1b99b75d4c9923dfe`，
18项包括manifest，不含`.out`；具体log与plan保留在`logs/`目录。

### 四级LDS地址缓存v2/v3：修正API后实测回退，放弃（2026-10-24）

v2不再用runtime index直接索引Python列表，改为逐stage选择，但这里误用了`fx.select`。
CPU编译15.3634秒、rc1，报`TypeError: too many positional arguments`，没有提交GPU。
核对容器中实际FlyDSL接口后，v3改用`arith.select(condition, true_value, false_value)`并
封装为`fx.Int32`。v1/v2的失败记录均保留，不能把这两次DSL实现错误称为硬件限制。

v3为`dsf5_pin1_addr32_i32_stage_cache_scoped_v3`，源SHA256为
`9a5245be602a72aa8206ab5edb2d13ebadd74818027cf0f191359107b24b5d7f`，相对i32父候选+20/-1行。
提前构造四组stage地址，约束在`v40:55`；runtime选择后的current/next地址仍约束在
`v32:35`/`v36:39`。原compact guard、input ring、wave布局、fences、数学和输出不变。
CPU编译18.9437秒、rc0，GEMM1完整ELF为
`1c3aba8dcbb5cf29595e17d31a898e0d112715c87adef97a3084b6957c90a6d8`。
资源仍VGPR1024/SGPR62/LDS327680 B/private0/spills0；static executable/wait/MSB为
5775/129/750，相对i32的5701/118/750增加了74条executable和11条wait。

原有界DS/WMMA模型通过：5775条可达指令、6386次transfer、0个M64 skip sites，
DS readiness和WMMA-to-DS overwrite提示均为空，未修改原hazard transfer模型。
TDM/非DS fence路径语言相同，44 state pairs/74 transitions；此检查不覆盖全部VALU/VMEM依赖。
seed0完整MoE输出22,020,096 B与DS5/pin0零字节差异，原precision gate及实际两阶段ELF绑定通过。

在全机空闲条件下完成固定正向九轮；所有组attempt=1，每轮后ownership/settled idle通过。
reference与candidate的计时均为pin1，GEMM2绑定同一原pin1 ELF。

| 固定顺序 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| i32 reference前 | 81.2343 /80.3974 /80.3551 | 80.3974 | 61.2613 |
| 四级地址缓存v3 | 82.5563 /82.7272 /82.3558 | 82.5563 | 61.3625 |
| i32 reference后 | 80.9201 /80.4965 /81.5147 | 80.9201 | 61.8475 |

候选相对两侧回退2.68529%/2.02199%，因此放弃，不追加该版broad strict、guard scope或ATT。
可以确认这个具体实现没有净收益；未采集matched ATT，不能把全部回退独占归因于11条新增wait。
现有i32最佳候选在本轮两组中值80.3974/80.9201 μs，作为追加读数保留，
不挑其中较低组替代此前正式正反向对照结果。源码结束后精确恢复`e53ada2b…06f93e`。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/compare_pin_i32_series_v1.py \
  reboot20261024_pin_addr32_i32_stage_cache_normal_screen_v3 \
  dsf5_pin1_addr32_i32_stage_cache_scoped_v3
```

CPU/prepare archive SHA256为`e767ca8af32ef16ba55ccce1b16494faee1fa02426e678c617e5989c0d7c4ef4`，42项。
正常性能archive SHA256为`58e3aea88f0aad212c91fb3c26e012f590bcd319d9e5e2161a13e8195c6484dd`，137项。
两包均逐SHA验证，不含`.out`；CPU、完整MoE、各组原表和相邻比例已在本地复核。
本地复算native与fence报告均和远端相同。seal工具按名称排除了远端fence报告，
补传同一报告并验证SHA256 `edb53b11293e5f9d4d11f2a4bc6fc0b31290806f6ef3b2cefa0df8c9f2ae1ce5`，
复用已有本地复算结果完成比较，没有重跑GPU或覆盖原始记录。

### 问答：当前最优候选的pin VGPR与原版策略区别（2026-10-24）

当前约80.6 μs的i32最优候选使用`AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=1`，
并要求`AITER_GROUPED_A4W4_A_PRESHUFFLE=1`。本问答时候选尚未正式替换生产源码，
生产仍为DS5/e53；后续是否采用以新的采用记录为准。

| 数据 | 原版pin | i32最优候选 |
|---|---|---|
| C accumulator | `v512:1023` | `v256:767` |
| 两份A/B输入槽 | `v256:383`、`v384:511` | `v768:895`、`v896:1023` |
| current/next LDS地址 | 编译器分配 | tied constraints指定`v32:35`、`v36:39` |

C共512个VGPR；每份A/B槽为A64+B64，共128个VGPR，两份槽不重叠。
ScaleA/ScaleB数据寄存器仍由编译器分配；固定的`sa`/`sb`地址寄存器不等于固定scale数据。

确实调整了原版pin策略，但不能把收益全部归因于换VGPR编号。直接开启原版pin的一组
原入口三轮中值为86.1428 μs；这组与pin0历史读数不是同一交错对照，且当时
MMA2/first1、inline-zero、A-only timeout等已有优化排除了pin1。因此没有据此计算
“pin本身回退百分比”，也不能把86.14→约80.6 μs全部称为register allocation收益。

另外已经测过保留上述优化并沿用原pin编号的`dsf5_pin1_feature_parity_v1`：
初筛中值81.4745 μs，随后同组合pin0重测中值82.2843 μs，观测差0.8098 μs（0.98415%）。
这两组没有双侧交错对照，不能确认稳定收益；完整三轮和改动范围见末尾对应问答。

新候选同时接入上述已验证GEMM1调度优化、调整C/A/B编号、固定current/next LDS地址并
采用等价i32地址表达式。单独旋转C/A/B布局的反向结果并未证明稳定优势；固定LDS地址的
完整正反向对照显示约1%改善，再将地址表达式改为i32后，相邻addr32 control下的四项增量
改善为0.29891%～0.75058%。静态wait/MSB数量仍118/750，不按指令条数推断收益。

### E65原fallback补充strict与i32源码正式采用（2026-10-24）

在仍为e53的生产版上，将i32候选与相同完整MoE输入逐字节对照，补充两项E65/T1536/topk9：

| case | 有效数据/scale等字节检查数 | 完整MoE bytes | differing bytes | logits_diff | rel_l2 |
|---|---:|---:|---:|---:|---:|
| seed0 dynamic | 27 | 22020096 | 0 | 3.594893112763491e-6 | 0.002681376878172159 |
| seed17 | 13 | 22020096 | 0 | 3.6086727178189905e-6 | 0.0026865052059292793 |

原precision gate通过；实际GEMM1为原有
`a8w4_tdm_fp4_apre_t64x256x256_w1x4_b3_K7168_e65_act1_q1r4_bth1_mg10_regpart`，
完整ELF SHA256为`0aff5418b49f9a3c430e6b147a421fc4a71d7073295c2f956664509b2d633cf1`。
GEMM2为同配置K2048的原fallback，ELF为
`b35a54f44a75c1e9b13a3ee2dddb4211845e1aadf938e19c9b35d919c6e1d0a6`。
原wrapper报告`E=65 M=17984 tile=64/256/256 waves=1x4`，没有改变E65的kernel选择。
E65并不和E64共用t256 GEMM1；这两项是random non-balanced的原fallback验证，
不是固定shared expert route验证，也不是性能数据。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/run_pin_i32_e65_followup_v1.py \
  dsf5_pin1_addr32_i32_scoped_v1 reboot20261024_pin_addr32_i32_e65_followup_v1
```

archive SHA256为`bd26f869a7b57e984f5c386fb650984bdae7e7257abe013ee3dd2c823845c704`，22项，
21份任务文件逐SHA验证，不含`.out`。实际config、字节检查、原metrics及两阶段完整ELF在本地复核通过。

随后在本地与a07-3正式采用源SHA256
`f3c77d2aaa4698c80eb82b5d377c4f7b529247fc24c5fffce401d9a6f1fb21ec`，GEMM1 ELF为`c5296627…f8001d2`。
采用前检查了五项strict、上述两项E65 strict、11项guard外完整ELF一致性、正常正反向对照及
matched ATT本地复核。相对采用前e53源为+35/-13行，仅修改目标pin1 GEMM1；
wrapper、原benchmark及两个producer文件逐字节保持，local/remote生产kernel均为UTF-8/LF。
本地原benchmark仍为原有CRLF副本，LF规范化后与远端原benchmark相同，未修改该文件。

两端branch均`hyg_dsr1_conc6144`，HEAD均`346438a75506f77b520d3a9a09ba4985f054e377`。
未执行clean、reset、checkout、commit或push；保留原来三个tracked M文件。
采用前source分别存为`baseline/reboot20261024_pin_addr32_i32_local_adoption_v1.py`和
`baseline/reboot20261024_pin_addr32_i32_remote_adoption_v1.py`，有SHA门槛并使用原子替换。
记录为同名`logs/*.adoption.json`。此前文中的“未正式采用”状态由本节更新。

采用后的新复测使用`run_adopted_pin_i32_normal_v1.py`，E64、E65各固定三轮，首轮从空cache
运行原`--scenario kernel`入口，保持warmup5/iters20、uniform/auto/seed0及non-balanced routing。
真实call keys随后通过原caller重新获取，验证normal入口创建的cache未被key准备修改；
每轮绑定到上述两个规模各自的实际两阶段ELF，不切source、不注入artifact、不根据读数重试。
每轮全机idle后启动，所有test/compiler child限时120秒。完整六轮结果如下。

### 问答：已有pin0优化叠加原版pin VGPR是否测过（2026-10-24）

测过。`dsf5_pin1_feature_parity_v1`只修改六处guard，让目标pin1 GEMM1接入已有
MMA2/first1、inline-zero、A-only standard timeout、M-group32/lookup/closing/DS floor/bitsearch/DS-first5、
DS_B计数修正及XDL queue bit。相对当时e53源的diff没有修改`fx.set_explicit_register`编号：
C仍为`v512:1023`，A/B两份槽仍为`v256:383`、`v384:511`；LDS地址仍由compiler分配。
没有加入后来的rotated布局、固定地址或i32地址表达式。

| 版本 | GEMM1三轮 μs | GEMM1中值 μs |
|---|---|---:|
| 相同DS5优化 + 原版pin1 | 82.9272 /81.4724 /81.4745 | 81.4745 |
| 相同DS5优化 + pin0，随后重测 | 82.5247 /82.0713 /82.2843 | 82.2843 |

两组均为原`--scenario kernel`、E64/T1536/topk8、7168/2048、uniform/auto/seed0、
non-balanced、A preshuffle1、warmup5/iters20。原precision gate、真实keys/ELF及ownership检查通过。
完整MoE输出22,020,096 B与原版pin1零字节差异；`logits_diff=3.61055e-06`、`rel_l2=0.00268721`。

观测中值差为0.8098 μs，按pin0中值计算为0.98415%。这只是连续初筛与随后重测，
没有完成该原版pin编号组合相对pin0的双侧交错对照，因此只支持“初筛有收益”，
不支持“原版pin有稳定约1%提升”。前面的回答漏提了这组实验；86.1428 μs属于没有接入这些
优化的原版pin1，不能用来否定该组合，也不能把它到当前最优版的差值全部算作VGPR编号收益。
全局pin开关还改变GEMM2的原编译版本；各pin环境下GEMM2完整ELF分别保持，不归因于GEMM1改动。

原始结果：`logs/reboot20261024_dsf5_pin1_feature_parity_candidate_screen_v1.summary.json`
和`logs/reboot20261024_dsf5_pin0_retest_v1.summary.json`，均位于本文实验根目录。
两份summary已重新核对，三轮中值重新计算；候选source SHA为`2822a279…6f6bee`，
GEMM1 ELF为`71ef64dd…ab2b`。历史runner固定采用前e53源的SHA门槛，当前F3源不要直接使用旧runner。

### 正式采用源码的六轮原入口复测完成（2026-10-24）

全程使用正式F3源，不切source、不注入cache artifact、不按读数重试。E64、E65各固定三轮；
每个规模首轮cache为空，随后通过原caller获取实际keys并验证key准备未改变原入口创建的cache。
全机idle后测试，测后ownership和settled idle均通过，全部原precision gate与两阶段真实ELF校验通过。

| 规模 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2三轮 μs | GEMM2中值 μs |
|---|---|---:|---|---:|
| E64/T1536/topk8 | 80.6925 /81.0172 /81.2006 | 81.0172 | 61.0965 /61.4658 /61.0145 | 61.0965 |
| E65/T1536/topk9 | 257.825 /256.345 /254.551 | 256.345 | 159.802 /160.971 /160.728 | 160.728 |

E64三轮`logits_diff=3.61055e-06`、`rel_l2=0.00268721`；E65三轮分别为
`3.59489e-06`、`0.00268138`。E65仍走原t64/w1x4/b3 fallback，不能称为t256优化的E65收益。
原入口E64中值距70 μs目标还有11.0172 μs；没有将此前80.5554 μs候选组或本次最低单轮作为统一成绩。
已有matched ATT仍绑定相同F3/GEMM1 ELF，mainloop累计wave span为68.19389%、
prologue13.07864%、activation/quant/scale/LDS+barrier15.31978%；这些不是dispatch关键路径分解。

首个runner完成E64首轮计时和原precision/idle检查后，host读取容器生成的600权限cache pickle时
发生`PermissionError`。失败的`.normal.json`原样保留。只在计时结束后修复本实验cache权限，
恢复runner复用保存的首轮原表、manifest及pre/post/settled记录，补完真实keys/ELF后处理，
随后继续其余五轮。没有重跑或替换首轮读数；首轮host elapsed未保存，记录为`null`，未填造时间。
后续host elapsed为8.49056 /8.44038 /21.45990 /8.49061 /8.50851秒；每个child受120秒timeout限制。

复现正式采用版E64，可在`dsr1_ep4`的远端仓库根目录运行原命令：

```bash
export AITER_GROUPED_A4W4_A_PRESHUFFLE=1
export AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=1
PYTHONPATH=$PWD AITER_USE_GROUPED_GEMM=1 AITER_GROUPED_DEBUG=0 ENABLE_CK=0 \
FLYDSL_DUMP_IR=0 AITER_LOG_MORE=1 AITER_FORCE_GFX1250=1 GPU_ARCHS=gfx1250 CU_NUM=256 \
AITER_MOE_EXPERT_BALANCE=false AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
timeout --signal=TERM --kill-after=5s 120s python3 -u \
op_tests/flydsl_tests/test_flydsl_grouped_gemm.py --scenario kernel --data-format a4w4 \
--experts 64 --tokens 1536 --topk 8 --model-dim 7168 --inter-dim 2048 --act silu \
--no-bias --no-check-aot-cache --warmup 5 --iters 20 --seed 0 --data-init uniform --scale-init auto
```

性能数据需先通过全机idle检查；E65复现只改`--experts 65 --topk 9`，保持分支原dispatch。
完整记录为`logs/reboot20261024_adopted_pin_addr32_i32_normal_v1.normal_resume_v2.json`。
新seal工具以F3源为门槛，保留旧e53工具及原PermissionError记录；归档包括六轮原表、manifest、
idle/ownership、真实keys与完整ELF、采用前snapshot及采用记录，不含`.out`。
archive SHA256为`90dccf2e28e66371d978319527583c60d490a0c05da4854f95629057e7e80fc4`，
118项。包已下载，本地逐member SHA、原表中值、四个完整ELF、权限恢复及保留首轮规则复核通过，
记录为`logs/reboot20261024_adopted_pin_addr32_i32_normal_evidence_v1.local_review_v1.json`。

### 原版pin叠加DS5优化：补完正反向十八轮后未确认稳定收益（2026-10-24）

继续回答原版pin是否有收益：在正式F3源码下，重新准备相同DS5优化的pin0真实call keys，
与历史`dsf5_pin1_feature_parity_v1`做完整双向对照。pin0的完整GEMM1 ELF仍为
`62037bcf…003897`，原版pin1组合为`71ef64dd…ab2b`；候选保留原C=`v512:1023`、
A/B=`v256:383`/`v384:511`，没有固定LDS地址、rotated布局或i32地址表达式。
这组比较用于澄清原版pin组合，与当前正式F3/pin1最优版的直接比较另做。

| 固定顺序 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 正向pin0前 | 81.6348 /81.9577 /81.5611 | 81.6348 | 64.6536 |
| 正向原版pin1组合 | 81.7720 /81.2786 /81.8795 | 81.7720 | 61.4910 |
| 正向pin0后 | 81.9575 /81.9597 /82.1599 | 81.9597 | 64.4197 |
| 反向原版pin1组合前 | 81.9872 /81.7298 /81.7677 | 81.7677 | 61.1243 |
| 反向pin0 | 81.7847 /82.4087 /81.8858 | 81.8858 | 64.5061 |
| 反向原版pin1组合后 | 82.4404 /81.5716 /82.7756 | 82.4404 | 61.2023 |

四次相邻GEMM1比较的改善为-0.168066% /+0.229015% /+0.144225% /-0.677285%。
两次略快、两次更慢，不采用原版pin组合；历史初筛约0.98%的改善没有在双向两侧稳定复现。
全局pin开关使GEMM2分别使用原有pin0和pin1 ELF，其差异不算作本项GEMM1源码优化收益。

十八轮均attempt1；原单测、warmup5/iters20、uniform/auto/seed0、A preshuffle1与non-balanced
保持，原precision gate、逐轮全机idle/ownership、真实call keys及完整ELF校验通过。
两方向原表、manifest和中值/比例已下载后在本地重新计算。host runner结束均精确恢复F3，
计时期间没有并发远端编译、SFTP或额外GPU检查，每个test/compiler child限时120秒。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_adopted_pin_flydsl_v1.py \
  run dsf5_pin1_feature_parity_v1 reboot20261024_pin_parity_adopted_prepare_v1 \
  reboot20261024_pin_parity_adopted_forward_v1 base candidate base --base-pin0
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_adopted_pin_flydsl_v1.py \
  run dsf5_pin1_feature_parity_v1 reboot20261024_pin_parity_adopted_prepare_v1 \
  reboot20261024_pin_parity_adopted_reverse_v1 candidate base candidate --base-pin0
```

以上是已执行的归档命令，重复运行须换未占用的新label。新wrapper以F3为source门槛并
从原caller准备reference keys；原计时入口与precision函数保持，不注入artifact。
证据为两方向`.summary.json`及`.local_review_v1.json`。

### 正式F3上的三个小候选：CPU与完整MoE准备通过（2026-10-24）

三个独立候选均基于F3，只作用于原compact pin1 GEMM1 guard。前两项分别只改两行和一行：
恢复原C/A/B编号但保留固定地址及i32表达式，或将current/next LDS地址槽`v32:39`改为`v16:23`。
第三项在既有steady循环中按四级ring展开，使LDS slot在各phase为constexpr；首tile inline-zero、
K-tile/slot/初始化及cluster sync顺序不变，没有新增pipeline、persistent或wave布局。
actual steady AST的4256组call sequence一致；这项检查不能代替设备数值或完整硬件依赖验证。

| feature简名 | CPU child秒 | VGPR/SGPR/LDS B | static executable /wait/MSB | 完整GEMM1 ELF |
|---|---:|---|---|---|
| original_layout_addr32_i32 | 19.1389 | 1024/62/327680 | 5701/118/750 | `e1c3c8e6…399ac74` |
| addr16_i32 | 19.0684 | 1024/62/327680 | 5710/126/750 | `09abdfab…96c2d6` |
| ring_unroll_i32 | 24.2922 | 1024/70/327680 | 10702/255/1830 | `41c8047b…64bde5` |

private、VGPR spill和SGPR spill均0，三份实际ISA机器码已在本地逐字节与对应ELF绑定，
资源与静态计数重新计算一致。静态展开导致指令模板变大，不据此推算动态性能。
每项实际normal入口的source-dependent keys绑定对应CPU ELF，GEMM2均保持原pin1完整ELF
`a5e9c250…58375311`。seed0完整MoE的22,020,096 B与正式F3均零字节差异，
`logits_diff=3.6105513278025183e-6`、`rel_l2=0.0026872099842876196`。
准备阶段不记录性能；这三项后续直接与F3/pin1最优版做双侧三轮对照。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/compile_adopted_pin_vgpr_batch_v1.py \
  reboot20261024_adopted_pin_followups_cpu_v1 \
  dsf5_pin1_original_layout_addr32_i32_scoped_v1 dsf5_pin1_addr16_i32_scoped_v1 \
  dsf5_pin1_ring_unroll_i32_scoped_v1
```

完整source SHA、改动范围及准备命令见`logs/reboot20261024_adopted_pin_followups_v1.plan.json`、
`logs/reboot20261024_adopted_ring_unroll_v1.plan.json`与三项`*_i32_adopted_prepare_v1.*`。
上述18轮pin对照及三个候选CPU/准备的证据包共448项，447份文件逐SHA通过本地复核，
不含`.out`；archive SHA256为`05bf8fae910664100ac28069588a3433b868ce39bee8773463ddf72ce58b9542`。
本地CPU复核为`logs/reboot20261024_adopted_pin_followups_cpu_v1.local_review_v1.json`。

### 三个F3候选的二侧九轮对照完成：均不采用（2026-10-24）

上述三个候选各完成固定`base candidate base`、每组三轮的原入口对照。共27轮均为attempt1，
没有按读数重跑；原precision gate、真实source-dependent call keys、完整两阶段ELF以及
测前全机idle、测后ownership和settled idle均通过。每次结束恢复正式F3源。

| 候选与顺序 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| original_layout：F3前 | 80.0643 /80.9559 /80.1656 | 80.1656 | 62.2671 |
| original_layout：候选 | 80.4437 /80.2520 /80.3678 | 80.3678 | 61.7463 |
| original_layout：F3后 | 80.5028 /80.7999 /80.7115 | 80.7115 | 61.2697 |
| addr16：F3前 | 81.3270 /79.8662 /80.5365 | 80.5365 | 61.9361 |
| addr16：候选 | 81.0171 /80.3467 /80.8589 | 80.8589 | 61.0252 |
| addr16：F3后 | 80.8843 /80.7748 /80.4500 | 80.7748 | 61.6621 |
| ring_unroll：F3前 | 80.5512 /80.4711 /80.6713 | 80.5512 | 61.3963 |
| ring_unroll：候选 | 160.258 /140.494 /140.399 | 140.494 | 61.7524 |
| ring_unroll：F3后 | 80.7853 /79.9228 /79.8616 | 79.9228 | 60.8121 |

恢复原C/A/B布局的两侧改善为-0.252228% /+0.425838%，未确认稳定收益；addr16为
-0.400315% /-0.104117%，两侧都慢。四级ring展开回退74.4158% /75.7871%，直接放弃，
不追加反向或ATT。编译后的SGPR、指令模板或wait/MSB静态数量不能单独证明回退原因。
各组GEMM2完整ELF相同，其计时差异不算作本项GEMM1优化收益。

本地从27轮原表重新计算中值和比例，并复核三项独立准备记录；完整MoE输出均为
22,020,096 B，与F3零字节差异。三个`.local_review_v1.json`及准备复核均通过。
当前仍保留正式F3，正式原入口E64中值仍引用已归档六轮复测中的81.0172 μs；
不把本次对照中79.9228 μs的较快control组改称新的正式成绩。

已执行的host编排命令如下；每个编译或GPU child在`dsr1_ep4`中受120秒timeout限制。
重复运行必须更换新label，保持原入口、precision和全机idle口径。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_adopted_pin_flydsl_v1.py \
  run dsf5_pin1_original_layout_addr32_i32_scoped_v1 \
  reboot20261024_original_layout_i32_adopted_prepare_v1 \
  reboot20261024_original_layout_i32_adopted_forward_v1 base candidate base
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_adopted_pin_flydsl_v1.py \
  run dsf5_pin1_addr16_i32_scoped_v1 reboot20261024_addr16_i32_adopted_prepare_v1 \
  reboot20261024_addr16_i32_adopted_forward_v1 base candidate base
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_adopted_pin_flydsl_v1.py \
  run dsf5_pin1_ring_unroll_i32_scoped_v1 reboot20261024_ring_unroll_i32_adopted_prepare_v1 \
  reboot20261024_ring_unroll_i32_adopted_forward_v1 base candidate base
```

新证据包为`logs/reboot20261024_adopted_pin_followups_perf_evidence_v1.pin_evidence.tgz`，
SHA256=`2715440357c331000de77f2bae64df25a52af1df38b3592498c3dc2675a3c5f3`，496项，
495份原文件逐SHA通过本地复核，不含`.out`。该包还补齐三项prepare driver summary，
与各`*_prepare_v1.preparation.json`分别绑定，未拼接或修改准备元数据。

### F3上的iteration pair、量化值pin及reuseB：CPU与native检查（2026-10-24）

继续准备三个独立候选，均基于正式F3且保留compact pin1 GEMM1 guard。iteration pair仅展开
既有循环的两次iteration，新增12行、删除2行；actual AST共37,440组、1,276,480个有序事件
与F3一致，保留所有K tile、LDS slot、初始化和cluster sync。没有新增pipeline。
量化值pin新增10行，将激活后的`all_vals`以empty tied LLVM constraints固定到`v128 + value_index`，
不改激活、amax或FP4量化数学。reuseB仅新增两行keyword，去掉该keyword后完整module AST与F3相同。

| feature | CPU child秒 | VGPR/SGPR/LDS B | static executable /wait/MSB | 完整GEMM1 ELF |
|---|---:|---|---|---|
| pair_unroll_i32 | 20.9368 | 1024/58/327680 | 7435/168/1110 | `64bd87e7…6a6e6c2f` |
| quant_values128_i32 | 19.1879 | 1024/62/327680 | 6058/165/981 | `74f7901f…9fd418e1` |
| b_reuse_pair_i32 | 19.3900 | 1024/62/327680 | 5701/118/750 | `dff7fd8c…ae924a9f` |

三个候选的private、VGPR spill和SGPR spill均0。下载后在本地逐机器码与ELF绑定、
重新计算资源及静态计数通过；这些检查不等同于性能验证。
source SHA和改动范围记录在`logs/reboot20261024_adopted_pair_quant_pin_v1.plan.json`与
`logs/reboot20261024_adopted_b_reuse_hint_v1.plan.json`。

reuseB的native检查通过，160处WMMA仅bit14改变；资源、其他机器码、PC及指令顺序保持。
每处检查previous/current/next WMMA类型、current/next native B operand及scale相同，
中间无branch或operand覆盖；首条WMMA无hint。数学activation对应本kernel的native matrix B。
Shader Guide §4.6（TXT10415–10417、11594–11597）说明reuse是给下一条的缓存提示，
current/previous instruction类型不满足条件时结果undefined，故先检查实际ISA再提交GPU。
该检查也已从下载的原ISA在本地重算，与远端完整acceptance记录一致。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/compile_adopted_pin_vgpr_batch_v1.py \
  reboot20261024_adopted_pair_quant_reuse_cpu_v1 \
  dsf5_pin1_pair_unroll_i32_scoped_v1 dsf5_pin1_quant_values128_i32_scoped_v1 \
  dsf5_pin1_b_reuse_pair_i32_scoped_v1
# 以下native audit在dsr1_ep4中执行，不dispatch GPU。
timeout --signal=TERM --kill-after=5s 120s python3 -u \
  .codex_tmp/yadai_gemm1_param_20261021/tools/audit_adopted_operand_reuse_isa_v1.py \
  reboot20261024_dsf5_pin1_addr32_i32_scoped_v1_cpu_v1 \
  reboot20261024_dsf5_pin1_b_reuse_pair_i32_scoped_v1_cpu_v1 --mode b
```

CPU与native证据包为`logs/reboot20261024_adopted_pair_quant_reuse_cpu_evidence_v1.pin_evidence.tgz`，
SHA256=`993c92ddcb08dc006cbc5a37da5be319a5ed010f92c90c2f18ce14cba0264d1a`，40项，
39份文件逐SHA通过本地复核，不含`.out`。reuseB先完成完整MoE准备，再做二侧九轮对照；
其结果及另外两项候选的设备验证见后续记录，CPU/native检查本身不记录性能收益。

### 优化收尾：reuseB双向对照不支持采用（2026-10-24）

按用户停止优化的指令，只补完当时正在运行的reuseB反向对照，之后不再启动新的优化实验。
正式保留F3源`f3c77d2aaa4698c80eb82b5d377c4f7b529247fc24c5fffce401d9a6f1fb21ec`。
reuseB候选实际normal keys及两阶段ELF绑定通过；seed0完整MoE的22,020,096 B与F3零字节差异，
GEMM2完整ELF保持`a5e9c250…58375311`。正反向共18轮均为attempt1，原precision gate、
测前全机idle、测后ownership与settled idle检查通过。性能如下。

| 固定顺序 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs |
|---|---|---:|---:|
| 正向F3前 | 81.3524 /81.0697 /82.0754 | 81.3524 | 61.9677 |
| 正向reuseB | 80.0662 /80.8696 /80.6082 | 80.6082 | 61.0232 |
| 正向F3后 | 80.2684 /80.8020 /81.0109 | 80.8020 | 62.1552 |
| 反向reuseB前 | 80.3846 /80.9919 /80.2139 | 80.3846 | 61.4847 |
| 反向F3 | 80.4373 /80.3741 /81.3629 | 80.4373 | 61.2465 |
| 反向reuseB后 | 80.5848 /80.7999 /80.0220 | 80.5848 | 61.4237 |

正向两侧改善为+0.914786% /+0.239846%，反向为+0.065517% /-0.183373%。
反向结果没有稳定胜过F3，因此不采用reuseB，也不追加ATT或新的性能实验。
两方向全部原表、中值和改善比例已下载后重新计算；没有挑选较快单轮作为正式成绩。

```bash
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_adopted_pin_flydsl_v1.py \
  run dsf5_pin1_b_reuse_pair_i32_scoped_v1 reboot20261024_b_reuse_pair_i32_adopted_prepare_v1 \
  reboot20261024_b_reuse_pair_i32_adopted_forward_v1 base candidate base
python3 -u .codex_tmp/yadai_gemm1_param_20261021/tools/screen_adopted_pin_flydsl_v1.py \
  run dsf5_pin1_b_reuse_pair_i32_scoped_v1 reboot20261024_b_reuse_pair_i32_adopted_prepare_v1 \
  reboot20261024_b_reuse_pair_i32_adopted_reverse_v1 candidate base candidate
```

以上为已执行的host编排命令，每个compiler/test child在`dsr1_ep4`内限时120秒。
重复运行需要新的未占用label。归档为
`logs/reboot20261024_b_reuse_pair_i32_adopted_final_evidence_v1.pin_evidence.tgz`，
SHA256=`0d9ad88c07950aa9b8567954396814b273fda5d3416b1e4be53d43a49f2fe5b1`，
294项，其中293份原文件逐SHA在本地验证通过，不含`.out`。

iteration pair和量化值pin仅完成上一节CPU编译及静态检查，未做设备数值或性能测试，未采用。
`prepare_adopted_scale_const100_isa_v1.py`与`run_adopted_scale_const100_isa_v1.py`未上传、未执行，
也未采用；相关编码观察不能表述为已证明的bug、性能瓶颈或性能收益。

### 最终源码提交与a07-3拉取（2026-10-24）

只执行`git add -- aiter/`，提交三个源码文件，共152行新增、47行删除；文档和实验目录不纳入此提交。
英文commit message为`Fix gfx1250 A4W4 synchronization and tune GEMM1 scheduling`。
commit=`6f656cf5b02b2f1acf723949652f91b48e2933d7`，已推送到
`git@github.com:ROCm/aiter.git`的`hyg_dsr1_conc6144`分支。

a07-3主机上的`/data/yanguahe/code/wk_sp1/aiter_a4w4_prefill_v2_yadai`已通过
`git pull --ff-only rocm hyg_dsr1_conc6144`快进到同一commit，三个源码SHA256与本地一致。
拉取前只将已确认与最终源码一致的三个tracked edits保存到stash
`9493cbc4e7e72e8fc6c5599850c5e01e8627651d`；未跟踪文件保留。
没有执行clean、checkout清理、reset或force push。正式GEMM1 source与matched ATT对应F3保持一致。
最终性能按用户要求在该已拉取commit下分别测pin=0、pin=1，结果在后续收尾记录中给出。

### 已拉取最终commit的pin=0 /pin=1性能复测（2026-10-24）

测试机器为a07-3，容器`dsr1_ep4`，目录
`/data/yanguahe/code/wk_sp1/aiter_a4w4_prefill_v2_yadai`。
HEAD为`6f656cf5b02b2f1acf723949652f91b48e2933d7`，tracked工作区干净。
FlyDSL=`0.3.3.dev882`，Python=`3.12.3`，PyTorch=
`2.11.0+rocm10.1.0a20260811-bkc.20260820`，HIP=`7.16.26315`。
boot ID=`cb28eed4-51ca-4cdc-bf5b-ecbc1d54db5f`。

使用原`--scenario kernel`入口，E64/T1536/topk8/model_dim7168/inter_dim2048，
uniform/auto/seed0、non-balanced routing、A preshuffle1、warmup5/iters20均保持。
各pin配置固定三轮，按0/1/0/1/0/1交错执行；六轮全部有效，没有按读数重跑。
两种配置分别使用隔离的正常cache，先由原caller准备真实source-dependent keys，
计时入口unset `COMPILE_ONLY`及`FLYDSL_RUNTIME_RUN_ONLY`，没有复制ELF进cache、注入ISA、
切换源码、改计时或改precision函数。每个compiler/test/audit child限时120秒。
每轮开始前全机idle，结束后ownership与settled idle通过；计时期间没有并发远端编译、
SFTP或额外GPU检查，全部测试完成后无测试GPU进程残留。

| 配置 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2三轮 μs | GEMM2中值 μs |
|---|---|---:|---|---:|
| pin=0 | 82.3413 /81.7383 /81.4917 | 81.7383 | 63.7136 /64.3523 /64.6011 | 64.3523 |
| pin=1 | 81.0487 /80.7177 /80.7452 | 80.7452 | 61.4405 /60.7530 /61.7589 | 61.4405 |

六轮原precision均为`pass=True`、`logits_diff=3.61055e-06`、`rel_l2=0.00268721`。
原logits_diff gate仍为0.01，没有放宽精度标准。
pin=1在本次GEMM1中值对比中快0.9931 μs，即1.214975%；这一结论只对应此次同commit、
同机器的固定三轮对比。GEMM2随全局pin开关使用各自原编译版，其差异不算作GEMM1源码优化收益。
最终GEMM1成绩引用pin=1的80.7452 μs，距70 μs仍差10.7452 μs，未达到70 μs目标。
按用户指令，优化到此结束；之前81.0172 μs的独立复测记录保留原样。

实际两阶段完整ELF逐轮绑定通过：pin0为`62037bcf…003897` /`4bd626e3…716be`，
pin1为`c5296627…f8001d2` /`a5e9c250…58375311`。pin1 GEMM1与本文采用的F3 matched ATT一致。
六轮原表、中值、命令manifest、精度、测前/测后检查、source-dependent keys与完整ELF
均已下载后在本地复核，没有修改原表。

复现时分别设置pin=0和pin=1，先确认全机GPU空闲，再按原命令各固定测三轮：

```bash
unset COMPILE_ONLY FLYDSL_RUNTIME_RUN_ONLY AITER_FLYDSL_MMA_GROUP AITER_FLYDSL_MMA_FIRST_GROUP AITER_TDM_B_TH AITER_FLYDSL_DS_FIRST_N
export AITER_GROUPED_A4W4_A_PRESHUFFLE=1
export AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=0
PYTHONPATH=$PWD AITER_USE_GROUPED_GEMM=1 AITER_GROUPED_DEBUG=0 ENABLE_CK=0 \
FLYDSL_DUMP_IR=0 AITER_LOG_MORE=1 AITER_FORCE_GFX1250=1 GPU_ARCHS=gfx1250 CU_NUM=256 \
AITER_MOE_EXPERT_BALANCE=false AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
timeout --signal=TERM --kill-after=5s 120s python3 -u \
op_tests/flydsl_tests/test_flydsl_grouped_gemm.py --scenario kernel --data-format a4w4 \
--experts 64 --tokens 1536 --topk 8 --model-dim 7168 --inter-dim 2048 --act silu \
--no-bias --no-check-aot-cache --warmup 5 --iters 20 --seed 0 --data-init uniform --scale-init auto
```

pin=1复现只将`AITER_FLYDSL_EXPLICIT_VGPR_PARTITION`改为1。
收尾证据包为`logs/final_commit_pin_perf_20261024_evidence_v1.pin_evidence.tgz`，
SHA256=`a4a3bdd37e37df69b63759f3b804fd41d9d44a10e8662d00c4f36eb4839fb41c`，
118项，117份原文件逐SHA验证通过，不含`.out`。
本地重算结果为`logs/final_commit_pin_perf_20261024_v1.local_review_v1.json`；
完整六轮记录为`logs/final_commit_pin_perf_20261024_v1.summary.json`。
两份文档和证据均保留在工作目录，不纳入本次仅`aiter/`源码的Git提交。
