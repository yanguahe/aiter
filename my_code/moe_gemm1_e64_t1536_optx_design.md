# OPTX：gfx1250 MoE GEMM1 的 M128/N256 蛇形计算方案

设计日期：2026-10-18，Asia/Shanghai。

本文细化用户提出的 OPTX。一个 persistent N-task 仍计算完整 M256/N256/K7168，内部使用 M128/N256 窗口交替处理上下两个 M half；四级 LDS ring 随 K 滚动覆盖，B/ScaleB 由两半共享。**按用户最新修正，同一 WG 的窗口每次横移跳 1024 列，而不是 256 列。**

状态：设计方案，尚未生成 OPTX kernel、编译、验证或测量。本文的寄存器数量推导是逻辑存活量分析，性能收益是待验证假设。此次只新增本文件，不修改 kernel、launcher、验证脚本或远端环境。

## 1. 目标、参照版本与功能约束

目标 workload 为 E64/T1536/topk8、model_dim=7168、inter_dim=2048、random non-balanced routing、A4W4、融合 SiLU、无 bias。shared expert 单独计算。

参照实现是 `cyclic64_mg8_delivery_review`，正式源码位于 [persistent GEMM1](../aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_prefill_persistent.py)，SHA256：

```text
b36f7ec94524a1cb0bf20c5152f2a82d630fed71c0db2c1b2559d68f9a43e790
```

| 参照项 | 已有结果 |
|---|---|
| kernel | `a8w4_tdm_fp4_t256x256x256_w2x4_b4_K7168_e64_act1_q1r8_cn4_cm1_prefetch_apre_persist` |
| 每个 N-task | M256/N256，完整 K7168 |
| 每个 WG | 8 条 wave，4 个 persistent N-tasks |
| cluster | `cluster_n=4 / cluster_m=1` |
| 正式三轮 GEMM1 中值 | 94.888 μs，原完整 MoE eager profiler |
| ELF VGPR / allocated VGPR | 470 / 480 |
| ELF SGPR | 106 |
| LDS | 278528 B = 272 KiB/WG |
| private / VGPR spills / SGPR spills | 0 / 0 / 0 |

OPTX 保留完整 padded M/N/K、每个 accumulator 的 56 次 K128 contribution 及其顺序、FP32 accumulation、原 activation、BF16 rounding、per32 E8M0/native FP4 conversion。256 行 expert alignment、routing、dropped/empty/tail expert 语义、payload/ScaleA 布局和 GEMM2 接口也保持。不能通过跳过 padding compute、反转 K、改变量化或拆分 partial sums 达到性能目标。

最新 v89 shared 共用 steady body 的空闲补测未确认收益，正式保留版本仍是 review。v89 不是 OPTX 的默认基线，首批 94.357 μs 也不是 OPTX 的性能数据。

## 2. 三个层次的 tile 与蛇形顺序

为避免将计算窗口与 launcher tile 混淆，使用以下定义：

| 层次 | 大小 / 含义 |
|---|---|
| parent task | M256/N256；expert owner 以这个 M256 的起点确定 |
| 内部计算窗口 | M128/N256；`half=0` 为上半，`half=1` 为下半 |
| K256 stage | 连续 256 个 K 元素；一个 TDM input stage |
| K128 contribution | 一个 K256 stage 内两次顺序 accumulation |
| persistent N-task index | `t=0,1,2,3`，每次横移跳 1024 列 |

### 2.1 保留当前 N-task 与 cluster 映射

设 parent M 起点为 `m0`，cluster 内 WG 的 N 位置为 `local_n=0..3`：

```text
n0(t, local_n) = (4*t + local_n) * 256
first_half(t)  = t & 1
second_half(t) = first_half(t) ^ 1
```

所有 cluster peers 使用相同的 `t` 决定上下方向，方向不能由 `local_n` 的奇偶决定。

| local_n | t=0 | t=1 | t=2 | t=3 |
|---:|---|---|---|---|
| 0 | N0–255 | N1024–1279 | N2048–2303 | N3072–3327 |
| 1 | N256–511 | N1280–1535 | N2304–2559 | N3328–3583 |
| 2 | N512–767 | N1536–1791 | N2560–2815 | N3584–3839 |
| 3 | N768–1023 | N1792–2047 | N2816–3071 | N3840–4095 |

四个 WG 合起来覆盖激活前的全部 N4096；每个 WG 只处理表中对应的四个窗口。激活/乘法后输出总宽度仍为 2048。

以 `local_n=0` 为例，窗口的逻辑蛇形顺序为：

```text
                     横轴：N，横移 +1024

上半 M0–127       (t0,h0)               (t1,h0) → (t2,h0)
                     ↓                    ↑         ↓
下半 M128–255     (t0,h1) → (t1,h1)               (t2,h1) → (t3,h1)
                                                                  ↑
                                                               (t3,h0)

顺序：(0,0) → (0,1) → (1,1) → (1,0) → (2,0) → (2,1) → (3,1) → (3,0)
```

该顺序表示每个 N-task 内的 half 优先级和 N-task 间的方向。一个 N-task 在移到下一个 N-task 前必须完成全部 K；不能将所有 N-task 的 partial accumulators 同时常驻，只为了实现一次跨 N 的 K-plane 扫描。

### 2.2 K stage 与用户提出的加载序列

本文的 `q` 表示 K256 stage，因此 `q=0..27`；`p=2*q+ksl` 表示 K128 contribution，其中 `ksl=0,1`。

用户描述的四级加载批次可写为：

| K 批次 | first half 的输入请求 | second half 的输入请求 |
|---:|---|---|
| 0 | A(first), ScaleA(first), B, ScaleB 的 q0–q3 | A(second), ScaleA(second) 的 q0–q3 |
| 1 | 同上，q4–q7 | 同上，q4–q7 |
| … | … | … |
| 6 | 同上，q24–q27 | 同上，q24–q27 |

偶数 `t` 的 first 为上半；奇数 `t` 的 first 为下半。first/second half 的请求集合都需要完整出现，B/ScaleB 每个 q 只请求一份 WG 内逻辑数据。

**四个 q 为一组是描述加载窗口的方式，不把 steady pipeline 改成“每四块全部计算完、清空、再加载四块”的串行循环。** 初始驻留 q0–q3；某个 q 的所有 LDS readers 完成后，可逐槽 refill q+4，而不等待这一批剩余三个 q 全部完成。first/second half 的计算在对应输入可读时插入。

跨 wave 的实际 TDM issue 时间没有天然全序。第一版保留现有 producer wave 分组，用数据 readiness 保证正确性；上述 first-before-second 表示加载与计算优先意图。若以后要求“所有 first-half 的四次请求必定先于任何 second-half 请求”的硬性跨 wave 顺序，需要另行加入 issue rendezvous，并单独评估成本，不应为展示顺序在每个 half 之间新增 WG barrier。

## 3. 推荐的八-wave 计算分工

第一版保留 256-thread/8-wave WG、`w2x4` 和当前 launch/cluster geometry，仅修改内部计算分工：

```text
wave_m = wave_id % 2
wave_n = wave_id // 2

每个 half 内的 wave M 起点 = 128*half + 64*wave_m
每个 wave 的 N 起点        = 64*wave_n
每个 phase 的 wave tile    = M64/N64
```

| wave | wave_n | 上半负责的 M 行 | 下半负责的 M 行 | task 内 N 列 |
|---:|---:|---|---|---|
| 0 | 0 | 0–63 | 128–191 | 0–63 |
| 1 | 0 | 64–127 | 192–255 | 0–63 |
| 2 | 1 | 0–63 | 128–191 | 64–127 |
| 3 | 1 | 64–127 | 192–255 | 64–127 |
| 4 | 2 | 0–63 | 128–191 | 128–191 |
| 5 | 2 | 64–127 | 192–255 | 128–191 |
| 6 | 3 | 0–63 | 128–191 | 192–255 |
| 7 | 3 | 64–127 | 192–255 | 192–255 |

表中 M/N 均为 task 内相对坐标；实际地址加上 `m0`、`n0`。

八条 wave 都先计算 first half，再计算 second half，不拆成两个各自推进 K 的四-wave 组。每条 wave 为两半保留独立 accumulator bank，同一个 K128 的 B/ScaleB operands 可在本 wave 两次 half compute 间复用。

每个 M64/N64 phase 仍使用原 native FP4 WMMA：逻辑 M16/N32 对应一个 vec16 FP32 accumulator。每 half 需要 4 个 M16 × 2 个 N32 = 8 条 WMMA；两半合计 16 条/K128，与 review 相同。完整 wave 仍执行：

```text
4 N-tasks × 56 K128 contributions × 16 WMMA = 3584 WMMA
```

## 4. 四级 LDS ring：保留物理布局，改变消费方式

第一版不缩小 LDS，不改成两套 rings，不增加 full-K A cache。保留四个 64 KiB data segments 和独立 16 KiB scale arena，总分配仍为 272 KiB。

### 4.1 为什么 A 的计算 tile 变小，LDS 暂时没有减少

M128/N256 是一次参与计算的窗口，LDS ring 保存的是四个完整 M256/N256/K256 stage 的输入。当前设计让每个 q 的上下两份 A128 都驻留，再交替读取较小的 A fragments；没有让两份 A 共用同一个可覆盖的 LDS 槽。

| 每个 K256 stage 的输入 | review | 当前 OPTX 设计 |
|---|---:|---:|
| A payload | A256：32 KiB | A128 上半16 KiB + 下半16 KiB = 32 KiB |
| B payload | 32 KiB | 两半共享32 KiB |
| ScaleA | 2 KiB | 上半1 KiB + 下半1 KiB = 2 KiB |
| ScaleB | 2 KiB | 两半共享2 KiB |
| **每个 stage 合计** | **68 KiB** | **68 KiB** |
| **四级 input ring 合计** | **272 KiB** | **272 KiB** |

FP4 payload 按每元素4 bit计算，例如一份 A128/K256 为 `128*256/2 = 16384 B`。这里的 272 KiB 已包含原独立 scale arena；实际 output payload 复用已释放的输入区域，不另加16 KiB。

因此，当前方案减少的是 LDS→VGPR 时每份 A operand slot 的大小与峰值存活量，LDS 中同时保存的 A 总量保持。这是先保留已验证 ring 与 task overlap 的设计选择，**不是硬件要求 M128 也必须分配272 KiB LDS**。

如果希望同时缩小 LDS，可以进一步改为每个 q 只驻留一份 A128/ScaleA128：first half 的两个 K128 所需 LDS reads 全部退休后，在同一 A 槽加载 second half；B/ScaleB 保留到两半都计算结束，随后才允许 q+4 复用该 stage。

这个变体的紧凑 input ring 逻辑容量为：

```text
4 * (A128 16 KiB + B256 32 KiB + ScaleA128 1 KiB + ScaleB256 2 KiB)
= 204 KiB
```

204 KiB 仅是输入数据的容量账目，不是已经实现的 LDS allocation。实际分配还取决于新的 stage pitch、对齐、output staging 是否能安全别名，以及请求/读数的 overlap。若仍保留原四个64 KiB data segments，空出来的 A 区域不会自动降低 allocation，必须重新设计物理布局。

单 A128 驻留还会改变调度：同一 q 需要先完成 first half 的两次 K128 输入捕获，再覆盖 A；原第5节按每次 K128立即交替两半的顺序不能直接照搬。要避免额外的 global→LDS 等待，需评估提前捕获 first-half operands、交错其他 ready stages，或保留/重读 B operands 的代价。不同 half 可以交错，但每个 accumulator 的 K128累加顺序仍必须为0..55。

因此把它作为独立的后续变体评估，不与本文保留双 A128 驻留的第一版混为一谈。即使实际达到204 KiB，`2*204 KiB > 320 KiB`，也仍不足以让同一 WGP 驻留两个同款 WG；是否更快需看等待与调度收益。

### 4.2 原四级 ring 的具体地址

以下地址为相对 `base_ptr` 的 byte offset，省略每个 region 内原有的 preshuffle 行/列布局：

```text
S  = 65536
s  = q & 3
sh = (q + half) & 3

A(q,0)      = s*S                 + 0       # 16 KiB
A(q,1)      = ((q+1)&3)*S         + 49152   # 16 KiB
B(q)        = s*S                 + 16384   # 32 KiB

ScaleA(q,0) = 262144 + s*4096      + 0       # 1 KiB
ScaleA(q,1) = 262144 + ((q+1)&3)*4096 + 1024 # 1 KiB
ScaleB(q)   = 262144 + s*4096      + 2048    # 2 KiB
```

`half=1` 的 A/ScaleA 位于下一个 physical segment 的 tail，但属于逻辑 stage q。不能将 `half` 与 `wave_m` 混用；当前 review 中二者对应的行分工在 OPTX 中已不同。

| 物理 segment 的区域 | 数据 |
|---|---|
| [0,16384) | 本 stage 的上半 A |
| [16384,49152) | 本 stage 的 B |
| [49152,65536) | 前一逻辑 stage 的下半 A |

上半与下半的 A 各搬运一次，B 搬运一次。ScaleA 两半各搬运一次，ScaleB 搬运一次。`on_i32` 的 descriptor/指针运算仍按原规则将 byte offsets 转成 dword offsets，不把表中的 byte 值直接当成 dword 值使用。

### 4.3 producer 保持原分组

| input owner waves | 请求 |
|---|---|
| 0、1 | 分别搬运上、下 M128 的 A |
| 2、3 | 各搬运 B 的一半 N，合起来一份 B256 |
| 4、5 | 分别搬运上、下 M128 的 ScaleA |
| 6、7 | 各搬运 ScaleB 的一部分，合起来一份 ScaleB256 |

producer 的职责与 compute 分工解耦。每个 q 仍为每条 owner wave 一条 input TDM，不让某个 compute wave 为上下 phase 各重新构造并发出一份 B 请求。

原 A multicast mask、实际 source pitch、OOB zero fill、descriptor clamp、`early_timeout`、B temporal hint 和 `cluster_n=4` launch attributes 均保留。两个 M half 使用同一 parent expert 的权重，不能根据下半起点重新 bisect 而误选下一 expert。

### 4.4 ring 覆盖的必要条件

逻辑 stage q 的所有 input 区域只有满足以下条件后，才能被 q+4 覆盖：

1. 上、下两半对 q 的两次 K128 所需 LDS reads 都已完成。
2. 共享 B/ScaleB 的最后 LDS read 已完成；不能只检查 first half。
3. 相关 DS completion 已等待，并由完整 reader/owner rendezvous 将 FREE 条件发布给所有写入者。
4. 所有 global→LDS 输入在消费前均满足原 per-owner TDM completion 和 peer visibility 条件。
5. 该区域若正在充当 output TDM source，还需满足 output completion 条件。

FREE 关注的是 LDS readers 完成；若 closing WMMA 只消费已捕获的 VGPR operands，可以与 refill overlap。但必须先证明最后一次 second-half read 已退休，不能在第二半仍从 LDS 读 A 时释放 stage。

第一版不引入新的 named barrier 协议。沿用当前 READY/REUSE 合并 fence 的语义，每个 K256 的两个 half 在同一组 fence 内完成。原条件下可保持的 fence 数应保持，不能因为两个 phase 就自动翻倍 WG rendezvous。具体 `tensor_wait(n)` 数值需根据每条 owner wave 的实际请求队列审核，不能将 review 的计数常量机械复制到新的发射位置。

## 5. 插入计算：按 K128 复用 B，按需读取 A

四级 ring 保存 K256 输入；计算时仍拆成两次 K128。推荐在每个 K128 contribution 内按 first、second 的顺序更新两份 accumulator，以免为跨多个 q 复用 B 而保留过多 B fragments。

抽象计算顺序如下。这里的函数名表示设计动作，尚不存在对应 OPTX API：

```text
for t = 0..3:
    first = t & 1
    second = first ^ 1
    zero(C_top, C_bottom)
    enter_original_safe_input_output_handoff(t)

    for q = 0..27:
        for ksl = 0,1:
            p = 2*q + ksl
            make_input_readable(q, ksl)

            Breg  = read_B_once(q, ksl)
            SBreg = read_ScaleB_once(q, ksl)
            Areg  = read_A(first, q, ksl)
            SAreg = read_ScaleA(first, q, ksl)
            compute_half(C_first, Areg, Breg, SAreg, SBreg)

            reuse_A_register_storage_after_first_readers_finish()
            Areg  = read_A(second, q, ksl)
            SAreg = read_ScaleA(second, q, ksl)
            compute_half(C_second, Areg, Breg, SAreg, SBreg)

            retire_current_operands_and_prefetch_next_p()
        release_q_and_issue_q_plus_4_if_valid()

    emit_original_numerical_epilogue_and_layout(C_top, C_bottom)
```

这段伪代码只给出语义依赖。实际发射允许在先前 WMMA 尚未完成时预读取下一份 A/B，也允许在 closing WMMA 期间 refill；不应实现成每个函数都 drain 全部 memory counters 的串行版本。

### 5.1 推荐的 operand pipeline

使用两份 M64 A/ScaleA slots、两份 N64 B/ScaleB slots，形成下面的 overlap：

| 当前 compute | 可预读取 | 可复用的旧 register storage |
|---|---|---|
| first(p) | second(p) 的 A/ScaleA；独立 B slot 可读取 B(p+1) | 不覆盖 first(p) 仍被 WMMA 使用的 operands |
| second(p) | first(p+1) 的 A/ScaleA | first(p) 完成使用的 A slot |
| first(p+1) | second(p+1) 的 A/ScaleA | second(p) 完成使用的 A slot |

当前 B/SB 必须保留到 first/second 都消费完成，future B/SB 使用另一份 slot。每个 p 的 shared B/SB 只从 LDS 读取一次；不能调用两次包含 B load 的完整 `load_lds_data` 来实现两个 half。

K128 boundary 需要预读取下一个 contribution；K256 boundary 同时受下一 stage 的 TDM readiness、carry 和 reader release 限制。优先延续 review 的 carry 思路，按新的 DS load 数量重新安排 `sched_dsrd/sched_mfma/sched_vmem`，保留合法的 closing WMMA 覆盖。

half bank 的选择采用编译期已知的两条入口，通过 WG-uniform 的 `t&1` 选择次序。不能用 runtime half index 直接索引 Python 的 rmem 列表，也不应把 28 个 K256 全部展开成独立 body。避免重复 descriptor、循环状态和大段 compute 实例长时间同时存活。

## 6. VGPR、SGPR 与 occupancy 的预估

### 6.1 accumulator 不减少

每条 wave 的一个 M64/N64 half 需要：

```text
64*64 / 32 = 128 个 FP32 accumulator/lane
两半合计 = 256 个 FP32 accumulator/lane
```

两份 partial accumulators 在整个 K loop 内都必须保留。这里与 review 的 accumulator 总数相同；蛇形次序只改变更新顺序，不能将 accumulator 算成 128 个。

不推荐把八条 wave 减成四条 wave轮流计算 M256：那样平均需要 512 个 FP32 accumulator/lane，尚未包括 A/B/scales 和临时量。

### 6.2 有机会减少 operand 的峰值存活量

按照当前 FP4 fragment 的逻辑 rmem 宽度，可做以下账目。数量单位为每 lane 的 32-bit 值，未计 compiler 临时值、地址、epilogue、连续寄存器块的对齐和实际寄存器分配：

| operands | review 的一个 K128 slot | OPTX 的一个 half A slot / 一个 shared B slot |
|---|---:|---:|
| A fragments | 8 个 M16 × 8 = 64 | 4 个 M16 × 8 = 32 |
| ScaleA | 4 | 2 |
| B fragments | 2 个 N32 × 16 = 32 | 32 |
| ScaleB | 2 | 2 |

review 的两个完整 K128 slots 逻辑上合计 `2*(64+4+32+2)=204` 个值。

推荐 OPTX 的两份 A slots 与两份 B slots 合计：

```text
2*(32+2) + 2*(32+2) = 136 个值
```

两者的逻辑 operand 容量差为 68 个值。这提供了降低 VGPR 的方向，**不能直接用 470−68 宣称 OPTX 编译后一定为 402 VGPR**。epilogue、地址生成、SSA promotion 和 scheduler 都可能决定最终峰值。

只有一份 A 与一份 B 的完全串行消费可以进一步降低逻辑存活量，但容易暴露 LDS latency。推荐性能版本保留两份小 A slots 和独立 future B slot，不以最低 register 数作为唯一优化目标。

### 6.3 SGPR 压力控制

硬件普通 SGPR 为 `s0–s105`。review ISA 的最高普通 SGPR 为 `s103`，使用 VCC，LLVM metadata 计入两个 extra SGPR 后为 106。不能将 metadata=106 解读为没有任何普通 SGPR 余量，也不能只检查 metadata 是否大于106来判定 spill。

OPTX 的 `half`、行偏移、ring index 和方向可复用现有标量状态，不需要常驻两套完整 A/ScaleA descriptor。建议：

- 保留当前每条 owner wave 唯一的 input job，descriptor 在 owning-wave 分支内构造。
- M64 的 compute offset 与 M128 的 TDM owner offset 分开，重复地址以常量偏移表达。
- 当前/下一 N-task 地址仅在实际需要的区间存活；global 地址继续采用 carry-safe i64。
- 不为四级 ring 保存四套完整 D#，也不同时常驻上下两半、当前/下一 task 的全部 descriptor。
- 编译验收同时检查 `.sgpr_spill_count`、`.vgpr_spill_count`、private segment，以及实际 `v_writelane_b32/v_readlane_b32` 的用途。SGPR spill 到 VGPR lane 时，private segment 可以仍为0。

SGPR spill 是需要优先排查的实现风险，不是 OPTX 的必然结果。尚无 OPTX ELF，无法给出最终 SGPR 数。

### 6.4 降低 VGPR 不自动增加驻留 WG

本方案保留 272 KiB LDS，`2*272 KiB > 320 KiB`，同一 WGP 仍不能同时驻留两个同款 WG。八条 wave 的 compute 并行度也保持。

因此预期收益来自 operand lifetime、LDS/WMMA overlap、bank/source-cache 行为和 peer 进度的改善，不能将寄存器减少直接解释为 occupancy 翻倍。256 个 accumulator 加 operands 仍通常超过256 VGPR，不能承诺消除全部 VGPR-MSB 指令。

## 7. 保持 FP4 payload 与 ScaleA 的输出布局

新的 compute owner 与旧 output TDM owner不再一一对应，需要显式处理这一步。

令 `r=0..3` 为一个 M64 内的 M16 row group：

```text
row_base_in_task = 128*half + 64*wave_m + 16*r
actual_row      = m0 + row_base_in_task + lane16
```

每个 row 的原 SiLU/gate-up pairing、BF16 rounding、per32 E8M0 amax 与 native FP4 conversion 保持。两份 vec16 accumulator 只在 epilogue 当前 row 需要时拆成原逻辑 N16 views，避免同时生成额外一整套 accumulator 副本。

### 7.1 LDS payload 按旧 output owner 的槽位写入

当前 payload staging 的一个 M16/MX-block 槽大小为 `Q_MX_BLOCK_BYTES=256` B。OPTX writer 使用：

```text
logical_output_owner_wave = 2*wave_n + half
row16_in_output_owner     = 4*wave_m + r
payload_slot              = 8*logical_output_owner_wave + row16_in_output_owner
payload_byte_base         = OUTPUT_LDS_OFF + payload_slot*256
```

槽内 `lane16/kgrp/packed FP4` 的 byte 布局沿用原实现。两个 compute waves 分别写某个旧 output owner 的前、后四个 M16 row groups；所有槽合起来仍恰好覆盖 16 KiB payload。

完成全 WG staging/visibility fence 后，**仍由原八条 output owner waves 各发出一条原布局的 TDM store**，每条写8个 M16 row groups。无需因为计算分成两个 M128 就将 output TDM 请求数从8变成16。这里的 ownership 转换必须做地址双射审核，不能直接沿用旧 `(wave*8+wm)` writer 公式。

输出区仍保留原 `[245760,263168)` 范围：其中前16384 B用于实际 payload，其余为原 allocation 预留。未新增 output LDS arena。

### 7.2 ScaleA 继续直接写 global

输出 ScaleA 沿用当前 canonical M32 地址公式，用 `actual_row` 取代旧固定 M128/wave 的行号。MX block 按 `n0 + 64*wave_n` 计算；原有效行、`kgrp==0` 和 byte owner 条件保持。

ScaleA 继续通过 `fx.ptr_store` 写 global，并保留原 completion requirement。第一版不将 ScaleA 改为 LDS→TDM，也不改变 GEMM2 所需 source pitch 或 quant group。

## 8. persistent task 交接和 tail 的处理

### 8.1 保留原 I0/I1 与旧 output 的安全 overlap

review 已在旧 N-task 的最后 K tile 前发出下一 N-task 的 I0/I1。OPTX 不将这部分优化替换为“每个 N-task 首先等待全部旧 output，再加载四个输入”。

下一 N-task 的上下方向可能反转，但 q0/q1 仍是相同 parent M 的两半输入，first half 只影响消费次序。必须重新审核新的 compute/read 顺序对以下依赖的影响：

- 上一 task 的 final input readers 完成后，下一 task 的 I0/I1 才能覆盖对应区域。
- successor 仅在 I0/I1 可读后 prime 和 compute q0。
- q2/q3 的目的区域与旧 output source 有重叠，必须在旧 output TDM 已完成后 refill。
- q4 覆盖 q0 的区域，只能在 q0 两半最后 LDS reads 完成后进行。
- 每条 owner wave 的 input/output 请求顺序和 `tensorcnt` completion distance 必须逐条核对。

不能由“同一 wave 的 TDM completion 有序”推导跨 wave readers 已完成。跨 wave visibility、reader fence 和 output arena 的保护条件仍需保留。

### 8.2 蛇形不能自动带来跨 N-task 全 K 的 A 复用

在一个 M256/N256 task 内，A_top/A_bottom 各载入一次，B由两半共享；这与 v52 不同。不过 review 在一个 task 内本来也没有重复的 global→LDS A/B 搬运。

task 结束时，四级 ring 主要驻留 q24–q27；下一 N-task 按原精度要求从 q0 开始。横移时即使停在同一个 M half，旧末尾 K 的 A 也不能作为新 q0 的 A。本文不引入 full-K A cache，不反转 K，也不宣称横移后完全免除 A reload。

在推荐分工中，shared B registers 让同一 wave 的两个 half 不重复 LDS→VGPR B load；但 review 的每个 wave 也只加载自己那份 B。因此相对 review，**总 input bytes 和理想的 LDS read 集合大体保持，主要改变它们的发射时间与存活量**。需要以 emitted ISA 和 trace 检查是否出现意外的二次 B read。

### 8.3 non-balanced routing 与尾部

expert 仍由 parent M256 起点查找；不能以 `m0+128` 查找第二个 expert。可用于审核的 half valid extent 为：

```text
valid_half_rows = clamp(parent_valid_rows - 128*half, 0, 128)
```

该值用于 input OOB/output mask 正确性，不用于削减 padded WMMA。原 empty expert、capacity sentinel、极小 expert、只有上半有效、下半部分有效的处理均需验证。尾部 payload 和 ScaleA 的写入条件按原语义保留，不能以 zero input 的结果代替随机数据正确性验证。

## 9. 与 v52 的区别及性能评估

| 项目 | review | v52 | 推荐 OPTX |
|---|---|---|---|
| task 范围 | M256/N256 | M256/N256 | M256/N256 |
| compute waves | 8 | 两组各4条 | 8条都计算两个half |
| 每组独立 K pipeline | 无 | 有 | 无 |
| B/ScaleB 的 WG 内逻辑请求集合 | 一份 | 两份 | 一份 |
| ring | 共享四级 | 两套独立三级 | 共享四级 |
| LDS | 272 KiB | 306 KiB | 目标保持272 KiB |
| partial accumulators/wave | 256个FP32值 | 总量与原分工相近 | 256个FP32值 |
| 性能 | 正式中值94.888 μs | 已有单次114.585 μs，未采用 | 未测量 |

v52 的 114.585 μs 是已有筛选记录，其样本限制见 [原 Q&A](moe_gemm1_e64_t1536_nonbalanced_opt_qa.md#问题-1前面尝试的-m128n256-tile-方案性能如何)。它不能用来推定 OPTX 会回退多少；OPTX 没有采用 v52 的 B duplication、独立三级 ring 和每次 task 的完整 output drain。

### 9.1 已有瓶颈证据

最新正式版四 SIMD ATT 的实际完整 ELF 与 review 一致。采样 elapsed lifetime 中，steady K为67.171%，drain K为9.607%，三次task handoff合计为14.092%。steady K 的主要 stall 归属包括 WG barrier、WMMA、TDM readiness 与 LDS load/DS completion。

这支持评估 OPTX 的供数与 compute 重叠方式，但这些比例不是可直接删除的耗时。详细证据见 [正式版 thread trace 报告](moe_gemm1_e64_t1536_best_thread_trace_analysis.md)。

### 9.2 潜在收益与主要风险

| 方向 | 可能的收益 | 必须检查的代价 |
|---|---|---|
| 小 A slots | 降低 operand lifetime、VGPR allocation或bank压力 | 新读数顺序是否暴露更多 LDS latency |
| 共享 B registers | 两个half消费同一份B operands | B必须跨half保留，不能为省寄存器又重新读取 |
| 统一half推进 | 改变原peer的operand/compute进度差 | producer职责仍不同，不保证消除barrier错位 |
| 半块计算与next A preload | 更多明确的独立操作供scheduler overlap | 若每half新增full WG fence，收益容易被抵消 |
| 蛇形与1024列横移 | 保留原cluster/N覆盖，同时改变half次序 | 不自动节省跨N-task的完整A reload |
| 保留四级272 KiB ring | 保持原prefetch深度和容量 | LDS限制仍在，不会获得第二个同款驻留WG |

另一个具体风险是 segment 访问：review 的两个 M half 的 A 位于不同 cyclic segments；OPTX 一个 half phase 内所有 compute waves 都访问该 half 的 A region。B也在当前 q 的固定 region。新的集中读取可能改变 LDS segment/bank contention，不能因访问范围更小就认定更快。保留 cyclic 布局后仍要检查实际 DS issue 与 WMMA 的交错。

已有 `k0_group4_tail4` 曾将VGPR从470降到444、spills仍为0，但单次筛选为96.069 μs，未采用。这个实验说明降低VGPR本身不足以保证加速，不是对OPTX的否定。

**评估结论：OPTX 有合理的 operand-lifetime 优化潜力，值得先做 CPU 编译与正确性筛选；现有证据不足以承诺快于 review，更不足以承诺低于80 μs。** 相对94.888 μs达到80 μs需要减少14.888 μs，约15.69%。本方案没有减少总计算、原始输入量或LDS容量，要获得这个幅度必须实际改善主要pipeline等待。

## 10. 实现与验收顺序

### 10.1 首个候选的边界

将OPTX限定在review当前M256 fused specialization，保持launcher的parent tile、expert alignment、grid、block与cluster属性。不要把全局`compute_tile_m`改为128，否则会连带改变owner查找、descriptor extent、epilogue与fallback路径。

新增内部M64/wave的half compute帮助逻辑；保留两份编译期accumulator banks，拆开A与B的LDS reads，按第7节进行output staging双射。其他原支持配置保持现有路径。候选标记需要区分OPTX与review，保留源码SHA和实际ELF，不能仅凭原`_persist`symbol判断采用了哪个实现。

### 10.2 CPU 编译与静态审核

在创建GPU executor前完成：

- 确认全部M/N输出位置覆盖一次，每个accumulator的K128次序为0..55；完整wave模型仍为3584次native WMMA。
- 检查四个input slots、两个M half、上下方向、output arena的地址区间与read-before-overwrite关系。
- 核对canonical payload和ScaleA映射，output仍为原八条TDM请求。
- 检查VGPR/allocated VGPR、SGPR、private segment、VGPR/SGPR spills；推荐第一版保持zero-spill门禁。
- 审核实际DS/TDM instructions和descriptor，不把更少source代码实例解释为更少动态计算。
- 保留合法WMMA operand、scale选择与dependency scheduling。register内容复用不等于自动增加WMMA hardware reuse hints；新half次序下既有hint也需要重新核对相邻operand，不能沿用仅由旧`wm`编号推导的条件。
- 完成原十组fallback ELF兼容性审核，并检查target ABI、layout、grid/block/cluster等不变量。

### 10.3 正确性验证

沿用既有完整11 cases：三个T1536 random seeds、skew、sparse、hot8、tiny expert、balanced，以及T1537/T1792/T2048。比较canonical payload、ScaleA、GEMM2 valid rows和MoE output；dynamic input、poison、eager/graph replay也必须通过。

要求与冻结参照的上述比较为bitwise mismatch=0；原benchmark sanity继续使用原门槛。不能修改用户验证脚本、使用const0替代random，或仅验证GEMM1单个数值输出而忽略quant/layout下游。

### 10.4 原性能命令

以下为未来验收使用的原benchmark，不是已运行的OPTX复现。OPTX候选的启用与源码绑定方式待实现后补充。

```bash
PYTHONPATH="$PWD" \
ENABLE_CK=0 \
AITER_MOE_EXPERT_BALANCE=false \
AITER_LOG_MORE=1 \
AITER_USE_GROUPED_GEMM=1 \
AITER_GROUPED_DEBUG=0 \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
timeout --signal=TERM --kill-after=5s 120s \
python3 -u my_code/test_flydsl_grouped_gemm_gfx1250.py \
  --scenario bench \
  --data-format a4w4 \
  --act silu \
  --no-bias \
  --no-check-aot-cache \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048 \
  --iters 20
```

基准口径为完整MoE eager profiler的GEMM1 `device_time_avg`，count应为19。在目标主机全机GPU/KFD、busy与VRAM均idle后运行；有占用每6秒等待，不竞争其他任务。计时期间不运行并发空闲检查，结束后保留首次后置状态并确认无本任务残留。

review与OPTX至少做三轮同批交替对比，第二轮逆序；保留全部原始数据、前置/后置条件和中值。一次较快值不能替代稳定收益。每项编译、验证、benchmark或trace analysis均使用120秒timeout；本文件不启动这些操作。

### 10.5 若进入 ATT，重点比较哪些证据

使用指定 [get_att_b_rocm_sdk.sh](get_att_b_rocm_sdk.sh)，绑定candidate source SHA与实际target ELF，比较同一完整workload：

- 是否仍完整执行3584条WMMA/valid wave。
- first/second half之间是否重复B LDS reads，A slots是否按预期复用。
- K128/K256 boundary是否新增等待，next-A/next-B preload是否实际覆盖在WMMA期间。
- WG barrier次数、steady peer到达差、TDM readiness/DS completion是否改善。
- output publication、旧output与successor I0/I1的overlap是否保留。
- VGPR-MSB/source-cache相关指令变化；不能只看寄存器总数。

不同capture的采样cycles用于诊断，不替代原eager性能结果；不能将stall百分比直接换算为可节省μs。

## 11. 后续结果记录模板

| candidate / source SHA | VGPR / allocated | SGPR | private / spills | 完整精度 / replay | 三轮GEMM1 μs | 同批review μs | 结论 |
|---|---|---|---|---|---|---|---|
| OPTX，待实现 | 未编译 | 未编译 | 未编译 | 未验证 | 未测量 | 未测量 | 仅设计 |

取得收益后，应补充实际候选启用方式、完整复现命令、全部日志、资源/ELF绑定和收益幅度；未取得收益时记录明确瓶颈证据，不改写旧数据。

## 12. 资料依据与已有证据

硬件事实来自本地资料；方案的性能判断与寄存器账目是依据当前源码的推导：

- [MI400 Shader Programming Guide](../../mi400_hw_wiki/raw/papers/mi400_hd_txt/architecture/subsystem/SH/MI400_Shader_Programming%2365.txt)：实际文件位于仓库上级的`mi400_hw_wiki`，相关章节为§3.3.1.1（printed p.30，普通SGPR与VCC）、§3.3.2.1/§3.3.2.3（printed p.32–34，VGPR allocation/MSB）、§4.7.1（printed p.160，LDS容量与segments）、§4.10.1（printed p.197–198，同wave TDM有序、跨wave无序）、§4.10.8（printed p.206，3 ops/wave、6 ops/SIMD的issue→XACK限制，与tensorcnt completion不同）、§5.3.2（printed p.218–219，GPR banking/source-cache）和§5.7.6（printed p.235–236，SIMD-pair共享LDS issue，B128 independent repeat rate=2 cycles、dependent约58 cycles）。
- [正式版thread trace报告](moe_gemm1_e64_t1536_best_thread_trace_analysis.md)：实际ELF绑定、最新phase/stall、occupancy与已有优化的限制。
- [优化记录](moe_gemm1_e64_t1536_nonbalanced_optimization.md)：review正式性能、v51寄存器降低但未提速、v52独立pipeline及v89收尾复测。
- [v52 generator](../.codex_tmp/gemm1_t1536_20261017/build_candidates_v52.py)与[v52 resource](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_split_mhalf_b3_named_drain_inspection.log)：两套独立三级ring、B/ScaleB duplication的实际实现。
- [review ELF comparison](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_delivery_review_comparison.log)与[review ISA](../.codex_tmp/gemm1_t1536_20261017/logs/review_th6_k256_v89_reference.isa.log)：资源值和普通SGPR范围。
- LLVM源码的metadata计数依据：仓库上级`llvm-project/llvm/lib/Target/AMDGPU/AMDGPUAsmPrinter.cpp`、`AMDGPUHSAMetadataStreamer.cpp`及`Utils/AMDGPUBaseInfo.cpp`，`NumSGPR`计入`ExtraSGPRs`，使用VCC时extra为2。该口径解释不等于OPTX已经完成SGPR分配。
