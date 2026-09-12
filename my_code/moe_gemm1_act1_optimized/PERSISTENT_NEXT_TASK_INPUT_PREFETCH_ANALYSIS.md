# 在前一 persistent task 的 SiLU 前预取下一 task input TDM 的可行性分析

## 结论

这个方向可行，但应分成两个不同结论：

1. **可以提前发射下一 task 的第一个 K256 stage；从 LDS 地址看，前两个 stage
   也都可以提前。**它们与当前 task 的 output LDS staging 区间不重叠。
2. **不能在当前布局下提前装入下一 task 的完整四级 input ring。**kernel 已申请
   gfx1250 的 320 KiB LDS 上限，stage 2/3 的 A/B payload 与当前 task 的 output
   staging 发生直接地址重叠。

最稳妥的第一版应只提前一个 stage。该方案不需要新增 VGPR，也不需要提高 SGPR
metadata；它通过复用 hotloop 结束后已经死亡的 SGPR 来构造并发射一个
`tensor_load_to_lds`。如果第一版正确且有收益，再尝试提前两个 stage。

## 当前执行顺序

当前 `persistent_overlap_pad8` 的 task 尾部大致是：

```text
previous task final WMMA / cluster synchronization
  -> s_wait_idle
  -> build current output TDM descriptor and LDS addresses
  -> SiLU + BF16 + LDS stores for banks 0/1
  -> s_wait_dscnt 0
  -> tensor_store_from_lds output half 0
  -> SiLU + BF16 + LDS stores for banks 2/3
  -> s_wait_dscnt 0
  -> tensor_store_from_lds output half 1
  -> persistent cluster boundary
  -> next task scalar/address/descriptor setup
  -> s_wait_tensorcnt 0
  -> next task input TDM
```

thread trace 中下一 task prologue 的：

```asm
s_wait_tensorcnt 0x0        ; ATT PC 0x2dfc
```

约消耗 `998.4 cycles/task`，占 steady task 的 `3.48%`。它等待的主要是前一
task 的两个 output `tensor_store_from_lds`，目的是在下一 task 的 input TDM
重新写入 LDS 前保护旧 output 数据。

前一 task 从 SiLU 开始到 persistent boundary 约有：

```text
output descriptor/address setup       216.8 cycles
SiLU banks 0/1 + first LDS drain     1747.9 cycles
SiLU banks 2/3                       1843.9 cycles
second descriptor finalization        118.0 cycles
output launch + boundary              240.3 cycles
--------------------------------------------------
total                                4166.9 cycles
```

这是足以覆盖一个 input stage TDM latency 的窗口。

## LDS 是否够用

### 当前 input ring

固定规模为 `tile_m=256`、`tile_n=256`、`tile_k=256`、MXFP4 A/B：

```text
A payload      = 32 KiB/stage
B payload      = 32 KiB/stage
ScaleA         =  2 KiB/stage
ScaleB         =  2 KiB/stage
--------------------------------
one K256 stage = 68 KiB
four stages    = 272 KiB
```

汇编中的四级地址为：

| stage | A payload | ScaleA | B payload | ScaleB |
|---|---|---|---|---|
| 0 | `[0x00000,0x08000)` | `[0x10000,0x10800)` | `[0x30000,0x38000)` | `[0x22000,0x22800)` |
| 1 | `[0x08000,0x10000)` | `[0x10800,0x11000)` | `[0x38000,0x40000)` | `[0x22800,0x23000)` |
| 2 | `[0x12000,0x1a000)` | `[0x11000,0x11800)` | `[0x40000,0x48000)` | `[0x23000,0x23800)` |
| 3 | `[0x1a000,0x22000)` | `[0x11800,0x12000)` | `[0x48000,0x50000)` | `[0x23800,0x24000)` |

`[0x24000,0x30000)` 是 48 KiB 的地址空洞，但不足以容纳一个完整的 68 KiB
K256 stage。

### 当前 output staging

K=7168 时共有 28 个 K256 tile。ATT 显示 task 结束时实际执行的是：

```asm
s_mov_b32 s92, 3
```

其他 `s92=0/1/2` 分支没有动态 hit。因此 output LDS base 的选择为：

| wave | output LDS base | 区间 |
|---:|---:|---|
| 0 | `0x12000` | `[0x12000,0x16800)` |
| 1 | `0x40000` | `[0x40000,0x44800)` |
| 2 | `0x1a000` | `[0x1a000,0x1e800)` |
| 3 | `0x48000` | `[0x48000,0x4c800)` |

每个 wave 的双 output half 占：

```text
2 × 64 rows × 144 B = 18 KiB = 0x4800
```

四个 wave 合计占 72 KiB，全部落在 stage 2/3 的 A/B payload 区域。

因此：

| 下一 task 的预取内容 | 与当前 output 冲突 | 结论 |
|---|---|---|
| stage 0 A/B/ScaleA/ScaleB，68 KiB | 否 | 可以提前 |
| stage 1 A/B/ScaleA/ScaleB，68 KiB | 否 | 可以提前 |
| stage 2 A/B payload | 是 | 必须等待 output TDM 读完 LDS |
| stage 3 A/B payload | 是 | 必须等待 output TDM 读完 LDS |
| stage 2/3 ScaleA/ScaleB | 否 | 地址可用，但单独提前收益有限 |
| 完整四级 input ring | 是 | 当前布局下不可行 |

完整 next-task input 加当前 output staging 的容量需求至少为：

```text
272 KiB + 72 KiB = 344 KiB > 320 KiB
```

即使重排掉当前 48 KiB 空洞，完整共存仍超过硬件 LDS 上限 24 KiB。因而不能靠
简单扩大 `.amdhsa_group_segment_fixed_size` 解决。

## SGPR 是否够用

metadata 为：

```text
.amdhsa_next_free_sgpr 104
numbered_sgpr          104
```

gfx1250 提供 `s0`–`s105` 共 106 个普通 SGPR，因此只有 `s104:s105` 两个尚未声明
使用，不能直接追加一套完整 next-task 状态。

但是在 current output descriptor 和四组 output LDS 地址已经物化之后、第一条 SiLU
指令之前，后续 epilogue/boundary 实际还会引用的 numbered SGPR 只有：

```text
s12, s17, s22, s24:s25, s28, s55,
s80:s91, s102:s103
```

其中必须重点保护：

- `s80:s91`：当前 task 的 output TDM descriptor；
- `s12`、`s17`、`s55`：第二个 output descriptor 的范围/偏移更新；
- `s22`：wave ID 和 boundary signal 条件；
- `s28`：persistent task ID；
- `s102:s103`：SiLU clamp 常量。

静态 future-use 扫描显示，扣除需要长期保留的 `s0:s1` kernarg pointer 后，仍有约
80 个 SGPR 可以通过生命周期复用。主要连续空闲区包括：

```text
s2:s11
s29:s54（保留或重映射 s55）
s56:s79
s92
s94:s101
```

一个 input TDM 需要 `s32:s35` 和 `s36:s43` 共 12 个 descriptor SGPR。next-task
的 A/B/ScaleA/ScaleB base pointer 需要 10 个 SGPR，task/expert/swizzle 临时值再需要
约 10–20 个。只要复用 dead SGPR，而不是把整套 prologue 状态机械复制到新编号，
容量足够，且无需提高 SGPR metadata。

推荐在发出早期 TDM 后丢弃 descriptor 和大部分 next-task 临时状态；进入下一 task
时重新计算非关键标量。这样用少量重复 SALU 换取较短的 live range，风险更低。

## VGPR 是否够用

metadata 为：

```text
.amdhsa_next_free_vgpr 1024
```

这已经达到 MI450/gfx1250 单 wave 可访问的 1024 VGPR 上限，不能再分配新 VGPR。
而且在 SiLU 开始前，四组 FP32 accumulator 仍然全部存活，不能保存下一 task 的
另一套 accumulator 或 operand tile。

不过 `tensor_load_to_lds` 使用 SGPR descriptor，payload 直接进入 LDS，不经过 VGPR。
因此只做“next-task descriptor 计算 + input TDM issue”可以保持零新增 VGPR。

当前 epilogue 在第一条 SiLU 指令以后文本上不再使用 `v0:v63`，但由于 kernel 使用
`s_set_vgpr_msb` 访问 1024 VGPR，不能仅根据低 8-bit VGPR 编号判断所有物理 bank
都空闲。第一版不应依赖额外 VGPR；lane ID、下一 task 的 LDS read address 和 rmem
operand 应留到下一 task 正式进入后再构造。

## TDM 顺序和 descriptor 上限

MI400 文档规定：

- 同一 wave 的 Tensor DMA load/store 相互保持 issue 顺序；
- 每 wave 在收到 XACK 前最多有 3 个 TDM descriptor in flight；
- 每 SIMD 最多有 6 个；
- `TENSORcnt` 的 issue-to-completion 计数上限为 63；
- `S_WAIT_TENSORCNT N` 同时执行 `S_WAIT_XCNT N`。

因此最安全的第一版只提前一个 stage。每个 wave 的 TDM 顺序可以设计成：

```text
I0 = next task input stage 0
O0 = current task output half 0
O1 = current task output half 1
```

最多正好 3 个 descriptor，不超过 per-wave 上限。下一 task 开始时执行：

```asm
s_wait_tensorcnt 0x2
```

由于同一 wave 的 TDM load/store 保序，`TENSORcnt<=2` 表示最老的 `I0` 已完成，
而 `O0/O1` 仍可继续在后台写回。随后保留现有 WG/cluster readiness barrier，即可
安全消费 stage 0。

下一 task 可以在 stage 0 计算开始后发射 stage 1。其顺序变为：

```text
O0, O1, I1
```

在第一次向 stage 2 A/B payload 写入前执行：

```asm
s_wait_tensorcnt 0x1
```

按 TDM 保序关系，这保证 `O0/O1` 已经完成，最多只剩不覆盖 output 区域的 `I1`，
此时才允许 stage 2/3 LDS 被新 task 覆盖。

提前两个 stage 也具备 LDS 地址条件，但顺序会成为：

```text
I0, I1, O0, O1
```

第四个 descriptor 的 issue 可能因每-wave 3-slot XACK 限制而停顿，并且 input TDM
会与 output TDM 竞争同一 WGP TDM/LDS 路径。因此它应作为第二阶段实验，而不是
第一版实现。

## cluster barrier 协议

早期 prefetch 不能放在最后一次 hotloop cluster synchronization 之前。否则较快的
WG 可能向 peer LDS 的 stage 0 写入下一 task 数据，而较慢的 WG 仍在读取当前 task
的同一 stage。

安全插入点是：

```text
final hotloop cluster barrier
  -> s_wait_idle
  -> 完成 current output descriptor 和四个 wave 的 output LDS address 物化
  -> 判断 next task 是否有效
  -> 所有 16 个 WG 以完全相同顺序 issue next-task stage-0 multicast TDM
  -> current task SiLU epilogue
```

必须满足：

1. next-task 有效性是 cluster-uniform；最后一轮所有 WG 都跳过 prefetch；
2. 每个 peer 发出的 multicast descriptor 数量、顺序和 mask 完全匹配；
3. 下一 task 消费 stage 0 前保留 WG barrier 和 cluster barrier；
4. stage 2/3 A/B payload 覆盖前确认前一 task output TDM 已完成；
5. 最终 task 仍执行完整 `s_wait_idle`。

## 推荐实现顺序

### 版本 A：只预取下一 task stage 0

这是建议首先实现的版本：

1. 保留 hotloop 尾部 final cluster barrier 和 `s_wait_idle`；
2. 先完成当前 task 的 output descriptor、output global address 和四个 output LDS
   base 的构造；
3. 使用 dead SGPR 计算 `next_task=s28+16`，判断是否有效；
4. 使用 `s32:s43` 构造下一 task stage-0 descriptor，每个 wave 发一条匹配的
   A/B/ScaleA/ScaleB TDM；
5. 不保存 next-task VGPR 状态；
6. 执行当前 task SiLU 和两个 output TDM store；
7. 下一 task 入口以 `s_wait_tensorcnt 0x2` 替代当前的全 drain；
8. stage 0 ready 后开始计算并发射 stage 1；
9. 在 stage 2 payload TDM 前以 `s_wait_tensorcnt 0x1` 保证旧 output 已完成。

### 版本 B：预取 stage 0 和 stage 1

只在版本 A 的正确性和 ATT 均有收益后尝试。该版本可提供更大的 latency hiding，
但需要确认第二个 output store 没有因 TDM descriptor slot/XACK 限制产生新的 issue
stall。

## 预期收益

当前直接相关的稳定热点为：

```text
0x36fc input TDM wait       1792.2 cycles/task
0x2dfc previous-output wait  998.4 cycles/task
-----------------------------------------------
ideal removable total       2790.6 cycles/task
```

相对于 `28,664.3 cycles/task`，完全消除两处等待的理想上限约为 `9.74%`。实际实现
会受到以下因素限制：

- next input TDM 与 SiLU 的 output LDS stores 争用 LDS/WGP 路径；
- next input load 排在 current output store 前，会改变 TDM queue 顺序；
- descriptor 构造和重复 scalar address 计算仍有成本；
- cluster 中 payload owner 与 scale owner 的传输时间不同，barrier 尾部仍可能存在；
- 部分 stall 只会从 `s_wait_tensorcnt` 移动到 TDM issue 或后续 barrier。

因此，第一版比较合理的预期是 `3%–6%` GEMM1 cycle 改善；`9.74%` 只能作为完全
隐藏两处主要 TDM wait 的理论上界。

## 最终判断

| 资源 | 是否足够 | 条件 |
|---|---|---|
| LDS | **部分足够** | 可提前 stage 0/1；不能提前完整四级 input ring |
| SGPR | **足够** | 必须复用 dead SGPR，不能机械新增整套状态 |
| VGPR | **无新增空间，但不阻塞方案** | 只做 descriptor-only TDM prefetch，不保存下一 task operand/accumulator |
| TDM slots | **stage 0 安全** | `I0,O0,O1` 正好 3 个；stage 0+1 需要额外验证 XACK/issue stall |
| barrier | **可保持正确** | prefetch 必须位于 final hotloop cluster barrier 之后，下一 task 继续做 readiness barrier |

所以，**可以在前一 task 的 SiLU VALU 之前发射下一 task 的 stage-0 input TDM**，并且
这是当前 thread trace 指向的高价值优化方向。完整四级预取在现有 320 KiB LDS
layout 下不可行；第一版应严格限制为一个 stage，并通过 random MoE e2e、连续多次
launch 和 ATT 检查 TDM issue stall、`0x36fc`、`0x2dfc` 以及 `0xac44` 的变化。

## 硬件依据

- `MI400_Shader_Programming#65.txt` §1.1.1.1、§3.3.4、§4.7.1：每个 wave/workgroup
  最多 320 KiB LDS，LDS 与 WGP cache 共享 384 KiB SRAM。
- `MI400_Shader_Programming#65.txt` §3.3.1：每个 wave 有 106 个普通 SGPR，编号
  `s0`–`s105`；VCC 位于 `s106:s107`。
- `MI400_Shader_Programming#65.txt` §3.3.2：MI450 每 SIMD 提供 1024 VGPR，wave32
  单 wave 最多访问 1024 个 VGPR。
- `MI400_Shader_Programming#65.txt` §2.4：同一 wave 的 Tensor DMA load/store
  相互保序。
- `MI400_Shader_Programming#65.txt` §4.3.7：wait counter 的依赖语义；等待期间
  wave inactive；`S_WAIT_*CNT N` 同时执行 `S_WAIT_XCNT N`。
- `MI400_Shader_Programming#65.txt` §4.10.8：每 wave 最多 3 个、每 SIMD 最多
  6 个未收到 XACK 的 TDM descriptor；`TENSORcnt` 的完成计数宽度为 6 bit。
