# `fmha_d128_opus_cpp_deep_introduce.md` 内容完善要求（提炼整理）

<!-- markdown-toc-generator:start -->
## Table of Contents

- [0. 文档定位与权威信息源](#0-文档定位与权威信息源)
- [1. 总体原则（适用于全文）](#1-总体原则适用于全文)
- [2. 分项内容要求](#2-分项内容要求)
  - [2.1 Pipeline 总览章节须覆盖完整流水线](#21-pipeline-总览章节须覆盖完整流水线)
  - [2.2 新增“每个操作的主要指令与数量”小节](#22-新增每个操作的主要指令与数量小节)
  - [2.3 伪代码不得省略关键计算步骤](#23-伪代码不得省略关键计算步骤)
  - [2.4 新增“调度原语详解”小节](#24-新增调度原语详解小节)
  - [2.5 Buffer 资源表的 VGPR 份数须用汇编核实](#25-buffer-资源表的-vgpr-份数须用汇编核实)
  - [2.6 Layout / Stride 公式须附底层 C++ 代码并逐式拆解](#26-layout-stride-公式须附底层-c-代码并逐式拆解)
  - [2.7 伪代码变量命名须与 C++ kernel 对齐并校验](#27-伪代码变量命名须与-c-kernel-对齐并校验)
- [3. 验收 Checklist](#3-验收-checklist)

<!-- markdown-toc-generator:end -->

> 本文从历史对话中提取并汇总了用户为完善
> `FlyDSL/fmha_opt_tools/fmha_d128_opus_cpp_deep_introduce.md`
> 所提出的全部内容要求，并对其重新组织、提炼，使逻辑更清晰、语义更准确，
> 作为该深度介绍文档后续撰写、补全与校对的统一依据。

---

## 0. 文档定位与权威信息源

`fmha_d128_opus_cpp_deep_introduce.md` 是对一个 D=128 OPUS GQA Flash-Attention 前向 kernel 的**深度介绍文档**。所有内容必须以下列信息源为准：

| 角色 | 路径 | 说明 |
|---|---|---|
| **介绍对象（C++ kernel）** | `FlyDSL/opus_attn/gqa_d128_kernel_template.hpp` | 文档要讲清楚的目标实现 |
| **事实基准（汇编/ISA）** | `FlyDSL/opus_attn/isa/opus_gqa_d128_bf16_causal_gfx950.s` | 指令数量、寄存器份数等一切“事实”以此为准 |
| **同源前端实现（可参考其注释）** | `FlyDSL/kernels/flash_attn_gfx950.py` | FlyDSL 前端逻辑与 C++ kernel 基本一致，注释可直接借鉴 |

**基准配置**：GQA、causal、bf16，`B=16, S=8192, H=64(=num_kv_heads), D=128`；Q tile `M=256`；目标硬件 gfx950 / MI355X；trait 实例 `opus_gqa_traits<32, 64, 128, 8, true>`。

---

## 1. 总体原则（适用于全文）

这些原则是从用户多次要求中提炼出的共性约束，撰写或修改文档任意章节时都需遵守。

1. **一切结论必须可追溯、有出处**：指令种类与数量、寄存器份数、layout/stride 公式等，必须能在 C++ 源码与对应汇编中找到依据，不得凭记忆或推测下结论。
2. **深扒到底层，不停留在高层函数名**：涉及计算/布局的说明要深入到被调用的底层函数和具体公式，而不是只引用一个高层封装函数的名字就带过。
3. **不省略关键步骤**：伪代码、公式、流程的描述要完整，不能用 `...` 省略掉对理解有意义的关键计算（尤其是 rescale、累加、归一化等）。
4. **命名与源码对齐**：文档伪代码中使用的变量名应尽量与对应 C++ kernel 中的真实变量名保持一致；若无法一致，需显式给出映射关系。
5. **以当前代码为准，重新阅读**：代码仓可能已发生重大变化，凡涉及代码的内容都要重新阅读相关源文件后再下笔，不能依赖历史记忆。

---

## 2. 分项内容要求

下列每一项对应文档中的一个章节/区块，给出“目标章节 + 具体要求 + 依据来源”。

### 2.1 Pipeline 总览章节须覆盖完整流水线
**目标章节**：第 1 章 kernel 概览与软件流水线（含 §1.4 伪代码）。

参照同源前端 `flash_attn_gfx950.py` 的注释，重写该章节，必须讲清楚以下五个方面：

1. **Prologue**：每一个阶段（P1…Pn）都要写出来，不能合并略过。
2. **Cluster 级流程**：main loop 的每个 cluster、epilogue 的每个 cluster 都要逐个说明。
3. **K / V 的多级 buffer**：写清 K、V 在 GM / LDS / VGPR 三级各有几个 buffer；哪些 load 与哪些 MFMA 计算使用对应 buffer；以及哪些 cluster 会用到对应 buffer。
4. **softmax 的拆分**：softmax 被拆成了几个阶段，这些阶段在 prologue、main loop、epilogue 的各 cluster 中具体如何分布。
5. **双 wave-group 协作**：两组 wave 如何“一组 load 的同时另一组 compute”、彼此负载如何交错执行，以及哪一组 wave 的执行节奏走在前面。

### 2.2 新增“每个操作的主要指令与数量”小节
**目标章节**：§1.5（在总览伪代码小节之后新增）。

针对总览伪代码中列出的每一个操作（如 `async K/V -> s_*`、`load Q`、in-flight scale、`mma0`、`if CAUSAL: mask`、`attn_row_max`、`sub_row`、`exp2`、`ds_read K`、`attn_sum`、`cast<bf16>`、`tr_load V`、`step_k`、`mma1`、`l_row *= rescale`、`v_o *= rescale`、`store O`）：

- 列出该操作涉及的**主要 ISA 指令**；
- 给出每个主要指令的**数量**，并写出**基于 tiling 大小的推导过程**（例如按 `KV*D / (BLOCK_SIZE*VEC)`、`E_M*E_N*E_K` 等公式逐步推导）；
- 以 C++ 实现及其对应汇编为依据进行核对。

### 2.3 伪代码不得省略关键计算步骤
**目标章节**：§1.4 伪代码中的 lazy-rescale（C3）分支等。

伪代码中此前被 `...` 省略的部分（如 `l_row *= ...`）必须**补全**，并与 C++ 冷路径 / rescale 分支语义一致，即完整写出：

```text
rescale = exp2(m_row - combined)
v_o    *= rescale
l_row  *= rescale
m_row  <- combined
```

### 2.4 新增“调度原语详解”小节
**目标章节**：§1.6（在 §1.5 之后新增独立章节）。

结合 `flash_attn_gfx950.py` 的注释，为伪代码中出现的调度原语写一份详细介绍，至少覆盖：

- `sched_barrier_exp_pairs<6,3,1>()`
- `sched_barrier_pairs<10,5,1>()`
- `sched_barrier(0)`

需说明它们的语义、模板参数含义、底层 LLVM intrinsic、对 MFMA/VALU/EXP 指令调度的约束作用，以及它们是否产生真实 ISA 指令 / 是否有运行时开销。

### 2.5 Buffer 资源表的 VGPR 份数须用汇编核实
**目标章节**：§1.1 “K/V/Q/O 驻留位置”表格。

通过汇编 `opus_gqa_d128_bf16_causal_gfx950.s` 确认 **S（scores）和 P（probabilities）各占用几份 VGPR**，并将核实结果回填到该表格（例如标注 `v_s` 的具体寄存器区间、`v_p` 的具体寄存器区间，以及它们各自的份数与软件流水线含义）。结论必须来自汇编，而非源码声明的字面推断。

### 2.6 Layout / Stride 公式须附底层 C++ 代码并逐式拆解
**目标章节**：第 2 章 GM/LDS/VGPR 完整布局映射。

该章节中凡涉及 **stride、layout 的公式计算**的内容，都要：

- 附上对应的 C++ kernel 代码片段（注明来源文件与行号，如 `make_layout_q`, `gqa_d128_kernel_template.hpp` 行号）；
- **不要只引用高层封装函数**，而要深入其调用的底层函数，把**每一个公式计算**讲清楚；
- 伪代码中使用的变量名尽量与对应 C++ 代码中的变量名保持一致（以 §2 起始处的 notation / Q 布局推导为范例标准）。

### 2.7 伪代码变量命名须与 C++ kernel 对齐并校验
**目标章节**：第 2 章布局推导（notation 段及各 `make_layout_*` 推导），并适用于全文公式。

对文档伪代码中用到的变量名做一次**一致性校验**：

- 逐个确认这些变量名在 C++ kernel 中是否有对应；
- 若有对应，确认命名是否一致；
- 对不一致或仅文档自造的变量名，需修正为与源码一致，或显式标注“文档变量 ↔ C++ 变量”的映射关系。

---

## 3. 验收 Checklist

完成或修改文档后，逐条对照以下清单确认达标：

- [ ] 概览章节写全了 Prologue 每个阶段、main loop 每个 cluster、epilogue 每个 cluster。（§2.1）
- [ ] 明确给出 K/V 在 GM/LDS/VGPR 的 buffer 数量，并对应到使用它们的 load / MFMA / cluster。（§2.1）
- [ ] softmax 的分阶段及其在 prologue / main loop / epilogue 中的分布已说明。（§2.1）
- [ ] 两组 wave 的交错协作方式与领先关系已说明。（§2.1）
- [ ] §1.5 中每个伪代码操作都列出主要指令、数量及基于 tiling 的推导过程。（§2.2）
- [ ] 伪代码无遗漏的关键计算步骤，lazy-rescale 的 `l_row *= rescale` 等已补全。（§2.3）
- [ ] §1.6 已详解三个调度原语（含模板参数、intrinsic、调度作用、是否产生真实指令）。（§2.4）
- [ ] §1.1 表格中 S、P 的 VGPR 份数已用汇编核实并回填。（§2.5）
- [ ] 第 2 章每个 stride/layout 公式都附了底层 C++ 代码并逐式拆解。（§2.6）
- [ ] 伪代码变量名已与 C++ kernel 校验一致，或给出明确映射。（§2.7）
- [ ] 全部结论可在 `gqa_d128_kernel_template.hpp` 与 `opus_gqa_d128_bf16_causal_gfx950.s` 中追溯。（总体原则 1）
