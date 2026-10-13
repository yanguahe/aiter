# MoE Framework Wiki：DSR1 DP4/EP4

## DSR1：cudagraph、EP 通信与 GEMM1/GEMM2 dispatch（2026-10-13）

本文依据 `a07-3`（`yanguahe@heliosr-1b114-a07-3.mnb.dcgpu`）的 `dsr1_ep4` 容器中实际脚本、模型 config 和 ATOM/AITER 源码，说明下面两条命令对应的服务逻辑。本文适用于调查时的容器版本；升级框架或修改脚本、量化配置、环境变量、CSV 后需要重新核对。

**这套命令配置为 DP4/EP4，每个 DP rank 的 attention 使用 TP1。默认开启 `FULL` cudagraph：四个 rank 都执行 decode 时，各 rank 使用同一个 batch bucket，并分别 replay 自己的整模型图；任一 rank 执行 prefill 时，该轮所有 rank 都普通执行。MoE 使用本地 64 个 routed expert 加 1 个 shared expert，共 E65；源 token 的 dispatch topk 为 9，但 EP CSV 的 topk 查询 key 为 `-1`。当前 CSV 没有这组 DSR1 配置的匹配行，GEMM1/GEMM2 均使用默认 `t64/w1x4/b3`。**

### 调查命令与验证范围

服务端命令：

```bash
bash /app/scripts/dsr1/serve_dsr1fp4_a8w8_dp4ep4.sh \
  /data/models/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4
```

benchmark 命令：

```bash
cd /app/ATOM/scripts/performance
bash /app/scripts/dsr1/bench_dsr1fp4.sh \
  /data/models/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4
```

**本轮只读取脚本、源码、配置和包路径，并用本地标准库执行 CPU CSV 匹配查询。遵照用户“机器 hang，不能运行任何 GPU kernel”的要求，没有运行上述命令、服务、benchmark、GPU 监控、框架 import 或 GPU 测试，也没有重启、清理机器。** 因此，本文说明配置和代码路径，不宣称本轮已成功启动服务、完成 capture 或获得性能结果；旧 TP1 容器日志不作为 EP4 的运行证据。

下文使用四个实际 DP/EP rank 的正常部署路径。`FAKE_EPLB` 的少卡模拟分支会改变实际进程数和 all2all 路径（`engine_core_mgr.py:112–147`）；本轮未探测 GPU，也没有用模拟分支代替用户指定的 DP4/EP4 分析。

### 1. 服务配置、并行结构与 cudagraph 开关

#### 1.1. 服务脚本实际传入的参数

`/app/scripts/dsr1/serve_dsr1fp4_a8w8_dp4ep4.sh:62–63` 启动 `python -m atom.entrypoints.openai_server`，核心参数如下：

```text
-tp 4
--enable-dp-attention
--enable-expert-parallel
--kv_cache_dtype fp8
--block-size 64
--no-enable_prefix_caching
--no-enable-chunked-prefill
--online_quant_config '{"layer_quant_config":{"*.self_attn.q_b_proj":"mxfp4"}}'
--max-num-seqs 2048
--gpu-memory-utilization 0.975
--cudagraph-capture-sizes '[1,2,4,8,16,32,48,64,128,256,512,1024,1536,2048]'
```

脚本没有传入 `--enforce-eager`、`--cudagraph-mode`、`--method`、`--enable-tbo`、`--max-num-batched-tokens` 或 `--all2all-backend`。结合当前 CLI 默认值，实际配置为：

| 设置 | 此命令的值 | 说明与远端源码依据 |
|---|---|---|
| `enforce_eager` | `False` | `arg_utils.py:50` |
| `cudagraph_mode` | `FULL` | `arg_utils.py:252–256` |
| `level` | `3` | `arg_utils.py:248–250`；compile 不等于 graph replay |
| capture sizes | 以上显式传入的 14 个值 | 覆盖 CLI 默认列表，不是默认的 11 个值 |
| 每 rank `max_num_seqs` | `2048` | 脚本显式覆盖默认值 |
| 每 rank `max_num_batched_tokens` | `16384` | `arg_utils.py:385–390` |
| `scheduler_delay_factor` | `0.0` | `arg_utils.py:502–508`；另有默认启用的 DP PrefillDelayer，见第 7 节 |
| speculative/MTP decode | 未启用 | `method=None`，`speculative_config=None`，见 `arg_utils.py:340–351,686–712` |
| TBO | 未启用 | `arg_utils.py:714–717` |
| `moe_all2all_backend` | `auto`，正常四 rank 路径选 `mori` | `arg_utils.py:719–727`、`moe.py:213–229` |

`engine_core.py:144–147` 在非 eager、非 disaggregated decode 的初始化中调用 `capture_cudagraph`。这证明本命令走开启 capture 的配置分支，不代表本轮已经运行并验证 capture 成功。

#### 1.2. `-tp 4` 在 DP attention 下转为四个 TP1 DP rank

不能仅凭命令中的 `-tp 4` 判断模型仍按 TP4 分片。`engine_core_mgr.py:83–109,281–295` 将 DP attention 下的 TP ranks 转成独立 DP engines：原始 DP1 × TP4 转为 **DP4 × TP1，PP1**。

每个 DP rank 有自己的请求队列、scheduler、attention 和 KV cache，处理分配给该 rank 的请求。MoE 开启 EP 后跨四个 DP rank 分配 expert；`moe.py:323–339` 明确在 EP 中每个 rank 完整持有一组 expert，MoE 的 `tp_size=1`。因此本地 expert 的 `moe_intermediate_size` 仍为 **2048**，不因 CLI 的 `-tp 4` 变成 512。

#### 1.3. MTP 权重、fake routing 与关键环境变量

模型 config 有 `num_nextn_predict_layers=1`，路径名称也包含 `MTP`，但脚本没有传入 `--method mtp`。当前普通 decode 每个请求每步输入 **1 个 query token**，`qlen=1`，没有 MTP draft/verification forward 或对应的 draft graph。`--num-speculative-tokens` 的默认值本身不会激活 MTP。

脚本第 11–14 行默认 `FAKE_EPLB=1`，追加：

```bash
--fake-eplb --hf-overrides '{"num_hash_layers": 0}'
```

它让各 MoE layer 使用预生成的人工均衡 router logits。`fake_eplb` 与真正的 `eplb_enable` 是不同设置；本次没有开启真实 EPLB，后者默认 `False`（`config.py:2017`）。fake routing 不控制 cudagraph，也不使 dummy token 自动无效。

与本次 MoE 路径直接相关的环境变量为：

| 设置 | 脚本或当前源码的值 | 本次作用 |
|---|---|---|
| `AITER_USE_GROUPED_GEMM` | 脚本设 `1` | 使用 FlyDSL grouped MoE |
| `AITER_FORCE_A8W4` | 脚本设 `0` | gfx1250 下 MoE activation 使用 FP4，实际为 A4W4 |
| `ATOM_MOE_GU_ITLV` | 脚本设 `1` | 使用 gate/up interleaved weight layout |
| `ATOM_MORI_V2` | 脚本设 `1` | 选择 MoRI v2 接口 |
| `ATOM_MEGA_STAGE2_FUSED` | 当前源码默认 `1` | 绑定 AITER MegaMoE，GEMM2 epilogue 融合 EP scatter |
| `ATOM_MEGA_STAGE1_FUSED` | 脚本设 `0` | 使用 token-major dispatch，不走 compact stage1-fused 路径 |
| `MEGA_DISPATCH` / `MEGA_DISPATCH_WIRE` | 脚本设 `mori` / `fp4` | MoRI FP4 dispatch，携带 quant scale |
| `ATOM_FUSE_SHARED_EXPERT` | 当前源码默认 `1` | 满足量化条件时将本 rank 的 shared replica 融入 grouped MoE |
| `ATOM_MEGA_COMBINE_WIRE` | 当前源码默认 `bf16` | 本次 prefill/decode 都以 BF16 combine |
| `ATOM_DP_LM_HEAD_MODE` | 脚本设 `allgather` | uniform decode 使用 DP vocab 分工的 LM head，见第 8 节 |
| `ATOM_USE_TRITON_MLA` | 脚本设 `1` | attention backend 选择 Triton MLA |
| `ATOM_MLA_PAGE_SIZE` / `ATOM_USE_TRITON_MLA_SHUFFLE_KV` | 脚本设 `1` / `1` | MLA 使用 token-level page metadata；shuffled KV 存储仍要求 scheduler `--block-size 64` |

这里需要按当前源码区分旧变量名：脚本仍设 `ATOM_MORI_V2_FUSED=1`，但当前 transport 读取的是 `ATOM_MEGA_STAGE2_FUSED`，本次因其默认值为 1 而启用 Mega；脚本的 `MEGA_COMBINE_WIRE` 也不能代替 ATOM 实际读取的 `ATOM_MEGA_COMBINE_WIRE`。依据为 `envs.py:161–177`、`mori_v2_prepare_finalize.py:134–138,197–213`。MLA 的 page size 与 scheduler KV block size 也须区分，见 `aiter_mla.py:361–369`。调查到的容器环境没有对应外部覆盖。

脚本的 `ATOM_USE_TRITON_GEMM=1` 不等于启用 Triton MoE；`moe.py:1274–1290` 的默认 Triton MoE 判断排除 EP，本次 expert GEMM 仍走 FlyDSL grouped 路径。

### 2. prefill、统一 decode 与跨 rank 混合阶段

#### 2.1. 先同步各 rank 调度信息，再决定 bucket 和执行方式

`ForwardMode.decide` 在每步通过 CPU DP group 同步各 rank 的 `scheduled_tokens`、`scheduled_bs` 和 `is_prefill`。`ubatching.py:270–278` 得到全组请求数最大值及是否存在 prefill；`forward_context.py:278–311` 据此决定：

```text
B_max = max(各 rank 的 scheduled_bs)
统一 decode = 所有 rank 都没有 prefill
B_bucket = 最小的、满足 b >= B_max 的 capture size
use_cudagraph = 本 rank 非 prefill
                且全组统一 decode
                且非 enforce_eager
                且存在可容纳 B_max 的 capture bucket
```

因此，各 rank 不能各自仅按本地 batch size 选不同 bucket。`FULL` 表示整模型 capture 的范围，不表示 prefill 也会 replay。

| 本轮各 rank 的阶段 | 模型执行 | 模型 token 行数 | GEMM1/GEMM2 选择时机 |
|---|---|---|---|
| 全组 prefill | 每 rank 普通 forward | 各自实际调度的 prompt token 数 | 每次调用按当前 EP 接收 buffer shape 查询配置 |
| 有 rank prefill，其他 rank decode | 全部普通 forward | 各 rank 实际调度的 token 数，不统一为 decode token bucket | 每次调用查询配置；decode rank 本轮也不 replay |
| 全组 decode：capture | 每 rank 对同一组 bucket 分别执行并捕获 | 每 rank `B_bucket × qlen`，本次 `qlen=1` | capture 时执行 dispatch，可以为不同图选不同版本 |
| 全组 decode：replay | 每 rank replay 对应 `(B_bucket,1)` 图 | 每 rank `B_bucket`，包含源端 dummy 行 | 沿用捕获时的 kernel，不再执行 Python GEMM dispatch |

本地 scheduler 选到 prefill 时就返回该 batch，不再混入本地 decode；但四个独立 scheduler 仍可能在同一轮处于不同阶段。默认 PrefillDelayer 尽量协调 prefill，不保证消除所有跨 rank 混合轮次，见第 7 节。

普通执行分支见 `model_runner.py:2817–2819`，replay 见第 2980–2988 行。默认 FULL 不把 compiled pieces 包成 PIECEWISE `CUDAGraphWrapper`（`backends.py:423–448`），所以普通 forward 可能经过 `torch.compile`，仍不能称为 prefill graph replay。

#### 2.2. 源 token 数、接收 token 行数与 token-expert route 数

EP4 中至少要区分三个数量：

- **源端模型行数 `T_r`**：rank r 的 embedding/attention/MoE router 处理多少行 hidden states。统一 decode 的 graph 包含源端 dummy，`T_r=B_bucket`。
- **目标 rank 实际接收的 token 行数 `R_r`**：MoRI 按目标 rank 去重，同一源 token 在同一目标 rank 选中多个 expert 时，token payload 只占一行。`R_r` 不等于该 GPU 的有效 expert route 数。
- **本地 expert 的有效行数 `M_e`**：目标 rank 对其 65 个本地 expert 分别执行多少条 token-expert route；一行接收 token 可以对应多个本地 expert。

`mori_v2_prepare_finalize.py:644–668` 用 `sum(context.running_tokens_across_dp)` 给接收 buffer 提供静态行数上界；因为 dispatch 按目的 rank 去重，这个上界不需要乘 topk。Mega 在第 791–811 行对接收 token、IDs、weights 和 scales 按上界切片，再调用 grouped MoE。

capture 为对称的四 rank bucket，`model_runner.py:3895–3905,3927–3935` 将每 rank 的 token 数设置为 `B_bucket`，因此 **每张 decode 图捕获时的接收 buffer 上界为 `4 × B_bucket`**。实际 `R_r` 通过 GPU 上的 `total_recv` 变化，不会把已捕获的 buffer shape 改成 `R_r`。

需区分 capture 与运行时 Python metadata：`prepare_inputs` 在第 2354–2359、2424–2434 行传入同步到的原始调度 token 数；`set_forward_context` 在 `forward_context.py:953–955` 原样记录该数组。正常 replay 不重新执行 `_recv_bound`、Python 切片或 CSV 查询，使用 capture 固定下来的 `4 × B_bucket` shape。对本次 prefill/跨 rank 混合的普通执行，源模型没有统一 token padding，接收上界使用各 rank 实际调度 token 数之和。

### 3. graph bucket、真实 batch size 与 CSV token key

#### 3.1. capture sizes 的单位是每 rank 的请求位置数

这 14 个 capture sizes 表示每个 rank 预捕获的 batch bucket，不是四 rank 请求总数，也不是 prompt 长度列表。对本次普通 decode：

```text
rank r 的有效输入行数 = B_r × 1
每 rank 图的模型行数 = B_bucket × 1
四 rank 模型行数合计 = 4 × B_bucket
每 rank 图捕获时的 EP 接收 buffer 上界 = 4 × B_bucket
```

prefill 请求可以长短不同，因此其总 token 数应写成 `Σ L_i`。若以后显式开启固定 K 的本地 MTP verification，常见每请求 query 数为 `K+1`（anchor + K draft token），graph key 和 token 预算都需要重新核对，不能将本次 `qlen=1` 的表直接套用。

#### 3.2. EP4 的 CSV key 来自接收 buffer shape

`grouped_moe_gfx1250.py:1452–1486` 从传入的 `topk_ids.shape` 取得 token 行数。当前非 compact 的 Mega 路径传入的是已按静态接收上界切片的 `recv_ids`，不是本地源 token 数，也不是 GPU 上的实际 `total_recv`。随后 `_get_padded_m` 生成 CSV token key：

```text
M < 32768：next power of 2(M)
M >= 32768：按 [32768,131072] tiers，取不大于 M 的最大 tier
```

依据为 `grouped_moe_gfx1250.py:116–125`。例如普通执行的接收上界为 65536 时，当前函数的查询 key 是 32768。这个函数只生成配置 key，不把模型输入或接收 buffer 再补成该 key 的行数。

本次统一 decode 的完整映射为：

| 每 rank batch bucket B | 图 key | 每 rank 模型行数 | capture 时固定 EP 接收上界 | CSV token key |
|---:|---|---:|---:|---:|
| 1 | `(1,1)` | 1 | 4 | 4 |
| 2 | `(2,1)` | 2 | 8 | 8 |
| 4 | `(4,1)` | 4 | 16 | 16 |
| 8 | `(8,1)` | 8 | 32 | 32 |
| 16 | `(16,1)` | 16 | 64 | 64 |
| 32 | `(32,1)` | 32 | 128 | 128 |
| 48 | `(48,1)` | 48 | 192 | 256 |
| 64 | `(64,1)` | 64 | 256 | 256 |
| 128 | `(128,1)` | 128 | 512 | 512 |
| 256 | `(256,1)` | 256 | 1024 | 1024 |
| 512 | `(512,1)` | 512 | 2048 | 2048 |
| 1024 | `(1024,1)` | 1024 | 4096 | 4096 |
| 1536 | `(1536,1)` | 1536 | 6144 | 8192 |
| 2048 | `(2048,1)` | 2048 | 8192 | 8192 |

例如各 rank 有 `[63,50,40,32]` 个有效普通 decode 请求，`B_max=63`，全组选择 `(64,1)`。四个源 rank 各执行 64 行模型，分别有 `[1,14,24,32]` 行源端 padding；每张图捕获时 EP 接收上界为 256，CSV token key 为 256。不能以 rank 0 的 63 行或源端 64 行作为这条 EP CSV 查询的 token key。

如果客户端的 6144 个请求恰好全部处于 decode 且均衡分配为每 rank 1536 个，则使用 `(1536,1)`；capture 接收上界为 6144，CSV token key 为 8192。这是配置推导的例子，实际请求分配和 batch 序列没有实测。

#### 3.3. 63 个有效 token 的支持条件

在上述全组统一 decode、选择 64 bucket 的条件下，框架已经在进入模型/MoE 前准备 graph 所需的 64 行源输入。MoE 不需要先将源输入从 63 行扩成 64 行，但必须正确处理 64 行源 token 的 routing、EP dispatch、本地 partial-M、接收尾部 mask 和输出合并。

这不能推广成“MoE 只支持 1–512 中的 2 的幂就能覆盖服务”。本脚本还 capture 48、1536、2048 等 bucket；prefill/跨 rank 混合普通执行的源行数和接收上界可以是其他值，各 expert 的 `M_e` 也不保证是 2 的幂。

#### 3.4. 源端 padding、BOS 与 EP 接收尾部 padding

对正常稳态 PP1 decode，`tokenIDProcessor` 默认采用 deferred sampled-ID 路径（`model_runner.py:194`）。在模型 replay 之前，`prepare_input_ids` 的第 574–593 行调用 `fill_deferred_decode_ids`；`decode_input_ids.py:59–102` 将实际请求 token 范围以外的尾部 token ID 写成 0。

当前 `fill_to` 查找使用 `reversed(capture_sizes)`，而 capture 后列表按升序排序（`model_runner.py:4037–4038`），因此正常路径可能清零到最大 bucket 2048，超过本轮实际读取的 B。这里能确认的是 graph 读取的 padding 区域已准备合法 ID 0，不能把该实现描述成“恰好只补到 B”。`prev_batch is None` 还有第 508–513 行的直接返回分支，上面的具体填充流程针对正常稳态 decode。

框架也准备源端 dummy 的 attention metadata：

- `attentions/backends.py:610–634` 将尾部 `cu_seqlens_q` 填为实际总 query token 数，使 dummy 序列的 query length 为 0。
- 本次选择 Triton MLA（`selector.py:138–140`）；其 metadata builder 继承 AITER MLA builder（`triton_mla.py:34`）。`aiter_mla.py:2497–2504` 将 `slot_mapping` 初始化为 `-1`，只给有效 token 写真实 slot，尾部 `context_lens=0`；第 2553–2554、2573–2576 行填平尾部 KV offsets 并将 last-page length 设为 0。

模型 config 中 `bos_token_id=0`、`eos_token_id=1`、`pad_token_id=2`。**token ID 0 是合法 BOS，不天然表示无效 token。** 框架使用 0 准备 dummy ID buffer，dummy 的有效性仍由调度、attention metadata 和输出范围决定。

有效 TP1 的 `VocabParallelEmbedding.forward` 使用 `F.embedding(x,self.weight)`（`embed_head.py:171–181`）。同一 rank 的 dummy ID 0 与真实 BOS 刚经过 embedding 时都读取 `W_embed[0]`，向量相同，不保证为 0。后续 position、attention 上下文等可能不同，进入 MoE 的 hidden state 不要求相同，也没有因 dummy 身份自动清零。

源端 dummy 会经过图中的全部 transformer layers，并参加正常 MoE router/topk、FP4 dispatch、GEMM1/GEMM2 和 combine。attention 的 dummy qlen=0 不会自动成为 MoE 的源 token mask；MoE 接收整份源 hidden states，dummy 也使用 fake routing（`moe.py:5117–5131`），可以产生跨 rank routed expert 计算及本来源 rank 的 shared route。这里的“计算规则等价”不表示它与有效行具有相同数值或相同 expert。

**EP 接收 arena 的无效尾部是另一种 padding。** 目标 rank 实际接收到 R 行 token 后，buffer `[R,接收上界)` 没有对应的有效接收 token；Mega 将 GPU `total_recv` 传为 `num_local_tokens`（`mega_moe.py:855`），grouped EP route/mask 据此跳过这些无效行（`grouped_moe_gfx1250.py:693–772`）。源端 dummy 被正常 dispatch 后属于有效接收行，不能与这种 dead tail 混为一谈。

正常 replay 最后按 `scheduled_tokens` 截取 hidden states 和 graph logits（`model_runner.py:2980–2988`），sampling 再处理真实 batch，源端 padding 结果不会作为额外请求的输出。

没有真实请求的 DP rank 也不能随意退出集体执行：`engine_core.py:833–849` 会调用 `dummy_execution`，`model_runner.py:1135–1175` 构造 `Sequence(id=-1)`、token ID 0、普通 decode 的 1 行同步 batch，并设置 `is_dummy_run=True`。此处的 **request ID `-1` 与 token ID 0 是不同概念**，这批同步结果不对应真实用户请求。

### 4. 本地 expert、shared fusion 与实际 GEMM 配置

#### 4.1. 模型逻辑 expert 与 EP dispatch 空间

模型 config 的关键值为：

```text
hidden_size=7168
moe_intermediate_size=2048
n_routed_experts=256
num_experts_per_tok=8
n_shared_experts=1
n_group=8, topk_group=4
num_hidden_layers=61
first_k_dense_replace=3, moe_layer_freq=1
```

因此 target model 有 61 个 transformer layers，前 3 层为 dense，后 58 层为 MoE。每个 token 在每个 MoE layer 选择 8 个 routed expert，并执行 1 个 shared expert。

shared 与 routed expert 使用相同动态 per_1x32 FP4 activation/FP4 weight spec；checkpoint 的 exclude 不排除 target shared expert，脚本的在线量化 override 只改 `*.self_attn.q_b_proj`。当前 `ATOM_FUSE_SHARED_EXPERT` 默认 1，`topK.py:16–86` 的条件允许融合。

真实 EPLB 未启用时，`expert_layout.py:43–80` 选择 **`LOCAL_REPLICA`**：256 个 routed expert 按四 rank 分片，每 rank 再拥有同一 shared expert 的一个本地副本。因此应区分：

| 数量 | 本次含义 |
|---|---|
| 256 routed + 1 shared | 模型逻辑计算，不代表每 token 执行 257 个 expert |
| 260 physical dispatch slots | 256 routed slots + 4 个 shared replica slots |
| 每 rank E65 | 64 个完整 routed expert + 1 个 shared replica |
| 源 token dispatch topk9 | 先选 routed topk8，再追加 1 条来源 rank 的 shared route |
| EP CSV topk `-1` | 配置查询使用的 EP sentinel，不表示真实 route 数为负数 |

`moe.py:3378–3478,3549–3558` 构造本地 E65 权重、260 项 expert mask 和 topk9 dispatch；不能继续使用 TP1 文档的 E257/topk9 或 E256/topk8 作为本次本地 GEMM 的查询 shape。

#### 4.2. shared replica 如何映射、是否单独 GEMM

本次 `shared_expert_dispatch.py:29–60,106–159` 的映射为：

```text
routed checkpoint ID p → dispatch ID p + floor(p / 64)
rank r 的全部 dispatch slots = [65*r, 65*r+65)
rank r 的 shared dispatch ID = 65*r+64
```

| EP rank | routed checkpoint IDs | routed dispatch IDs | shared dispatch ID |
|---:|---|---|---:|
| 0 | 0–63 | 0–63 | 64 |
| 1 | 64–127 | 65–128 | 129 |
| 2 | 128–191 | 130–193 | 194 |
| 3 | 192–255 | 195–258 | 259 |

`moe.py:590–622,3658–3680` 先在 256 个 routed expert 中选 topk8，再 remap routed IDs，追加本来源 rank 的 shared ID。shared route 固定在来源 rank，本 rank 的 shared replica 只为本 rank 源 token 计算 shared 输出；不会因为收到其他 rank 的 routed token，就为这些 token 再加一次本地 shared 输出。

**当前 shared 融入 E65 的 grouped GEMM1/GEMM2，不走独立 shared MLP。** `deepseek_v2.py:1211–1226,1289–1299` 只在 shared fusion 关闭时创建并调用独立 `self.shared_experts`。融合后的每 token 仍是 8 routed + 1 shared 的逻辑运算，既不是每 token 计算所有 65 个本地 expert，也不是计算全部 260 个 dispatch slots。

#### 4.3. MoRI FP4 dispatch、GEMM 与融合 combine

实际包路径为 `/app/ATOM/atom`、`/app/aiter/aiter`、`/app/mori/python/mori`。当前正常四 rank 路径选择 MoRI v2 + AITER MegaMoE，`ATOM_MEGA_STAGE1_FUSED=0`，处理链为：

```text
来源 rank：hidden states → router/topk8 → dispatch-ID remap + shared route
→ MoRI 按目的 rank 去重、quant 为 FP4、发送 token 和 scale
→ 目的 rank：接收 token-major buffer，按本地 E65 路由和 expert M 对齐
→ grouped GEMM1 + SiLU/gating + 中间 activation quant
→ grouped GEMM2：epilogue 应用 route weight，P2P 写来源 rank 的 combine arena
→ combine：通信同步并按来源 token 汇总结果
```

`mori_v2_prepare_finalize.py:303–331` 显式选择 quantizing MoRI dispatch；第 771–797 行调用 Mega 全层执行。`mega_moe.py:791–862` 包含 dispatch、接收切片、`fused_moe` 和 combine；GEMM2 的 `stage2_scatter` 将按 route 加权的结果直接写入 peers 的 combine staging，所以不需要再把普通本地 MoE 完整输出交给单独的传统 combine 搬运。

MoE 源 token capacity 取每 rank scheduler 预算 16384（`fused_moe/config.py:369–396`）。Mega 接收 arena 容量为 `world_size × max_tokens_per_rank`（`mega_moe.py:402–403`），本次为 65536 行；每步/每张图另使用第 2、3 节的较小接收上界。这是容量与 shape 推导，不是本轮实测 VRAM。

`AITER_FORCE_A8W4=0` 在 `aiter/fused_moe.py:800–804` 选择 FP4 activation quant dtype。本次 MoE 实际为 **A4W4**；服务脚本名中的 `a8w8` 不能用来推断 expert GEMM 的数据格式。

#### 4.4. 当前 CSV 无匹配，两个 GEMM 都 fallback 到 t64/w1x4/b3

默认 CSV 为 `/app/aiter/aiter/configs/tuned_grouped_fmoe.csv`，调查版本含 **113 条 data rows**。未找到附加 grouped model CSV、CSV 路径覆盖或本次 TDM tile/warp/buffer 环境覆盖。

本次 grouped MoE 的配置查询包含：

```text
token_num = _get_padded_m(EP 接收 buffer 行数)
model_dim = 7168
inter_dim = 2048
experts = 65
topk = -1
activation = Silu
dtype = bf16
q_dtype_a = FP4，q_dtype_w = FP4
quant_type = per_1x32
ep_fused = True
```

不能只看 CSV 的 token 列；expert 数、I 维度、EP topk sentinel、量化和 `ep_fused` 等也须匹配。CPU 检查从当前源码 AST 提取 `_get_padded_m` 和 `_find_grouped_config` 等函数，以标准库替身提供 dtype、architecture 和 CSV 中 CU key，不 import 框架或查询设备。15 组查询全部无匹配；CSV 连 `model_dim=7168 / inter_dim=2048 / expert=65` 的候选行都没有，因此该结论不依赖本轮 GPU CU 探测。

| 调用范围 | 接收上界/CSV key | 本次 GEMM1 / GEMM2 |
|---|---|---|
| 全部 14 个 decode bucket | 见第 3 节表 | 无匹配 row，两阶段默认 `t64/w1x4/b3` |
| prefill、跨 rank 混合普通执行 | 接收上界为全组本轮源 token 数之和，最多 65536；再应用 `_get_padded_m` | 同一 E65/I2048 查询没有候选 row，两阶段默认 `t64/w1x4/b3` |

这里 `t64/w1x4/b3` 表示每阶段 `tile=(64,256,256)`、`(m_warp,n_warp)=(1,4)`、`num_buffers=3`。默认参数见 `grouped_moe_gfx1250.py:547–590`。无 row 时上层局部日志变量 `num_buffers=2` 不会传入 `_grouped_a8w4_tdm_moe`，调用使用函数默认 b3（第 1470–1472、1518–1547、1604–1629 行），不能把日志局部值当作实际 launcher 参数。

当前容器的 `grouped_gemm_mxfp4.py:74–195` 直接调用 `kernels/mxfp4_preshuffle_gfx1250_tdm.py`，没有旧容器的 optimized/persistent selector。**旧 TP1 文档中“token key 16384 切换 optimized t256”及其他 E64/topk8 开发树的 dispatch 结论，不适用于本次 E65、EP topk=-1 的配置。**

### 5. 在 cudagraph 下按 token 数选择不同 GEMM 的条件

机制上，GEMM1 和 GEMM2 可以分别配置不同 tile、warp、buffer 等参数；CSV 字段包括 `tile_m2/m_warp2/n_warp2/num_buffer_stage2`。二者共享 expert 连续行布局，必须满足 `max(tile_m,tile_m2)` 能被两个 tile M 整除（`grouped_moe_gfx1250.py:648–656`）。

prefill/跨 rank 混合普通执行每次调用 MoE 都经过配置查询，可以按本步 **EP 接收 buffer shape** 选择两个 GEMM 的版本。统一 decode 则在各 bucket 的 capture 期间选择版本；不同 bucket 图可以记录不同 kernel。但当前 CSV 没有匹配行，所有本次 bucket 和普通执行都使用同一套默认 GEMM 配置，不能描述成已经实现了 DSR1 的 token 分档切换。

同一张图 replay 时，输入值、实际 `R_r`、router logits、expert IDs、routing weights、各 expert 的 `M_e` 以及通信数据可以更新；已捕获的 kernel、launch 参数和 tensor shape 沿用 capture 结果。这些 GPU 数据变化不会重新运行 Python CSV/launcher dispatch，也不会自动把 GEMM1/GEMM2 换成另一版本。

若要按同一 bucket 内更细的真实 token 数或 routing 分布切换版本，需要增加预捕获的 bucket/图，并在图外明确选择，或实现相应图内 GPU 分派机制。当前 graph key 只有 `(running_bs,max_seqlen_q)`，不带有效 token 数或 kernel 版本。48 与 64 bucket 甚至共用 CSV token key 256，也不能仅靠 token key 区分这两张图的 GEMM 配置。

修改 CSV、dispatch 代码或编译环境后，已有图不能自动反映变化，需要重新加载缓存配置并重新 capture，通常由重启服务完成。此处只说明后续生效条件，本轮没有修改运行环境或重启服务。

### 6. prefill 长度分布对 attention 与 EP4 MoE 计算量的影响

#### 6.1. attention core 依赖每个请求的 query/KV 长度

同一个 prefill batch 中各请求长度可以不同。对没有历史 KV 的完整 dense prefill，固定 head 数和维度、各请求长度为 `L_i` 时：

```text
总 query token 数 T = Σ L_i
attention core 的主要理论运算量 ∝ Σ L_i²
causal query-key 有效对数 = Σ [L_i × (L_i+1) / 2]
```

| 两次 forward | 各请求长度 | T | Σ L_i² |
|---|---|---:|---:|
| A | `[512,512]` | 1024 | 524288 |
| B | `[256,768]` | 1024 | 655360 |

B 的平方项多 25%，但 causal 有效对数的比例并非恰好 25%，整个 attention 层或实际耗时也不能按该比例推断。线性投影、tile padding 和执行效率另有影响。存在历史 KV 时需使用各请求本轮 query 数与 KV 长度，不能仅看 prompt 总 token 数；本脚本明确关闭 chunked prefill 和 prefix caching。

上述是数学推导。ATOM scheduler 在 profiling 且启用 `ATOM_ENABLE_DETAILED_ANNOTATION` 时另统计每请求 query 平方项和 query×KV 项（`scheduler.py:3517–3558`），这些统计也不是实际 kernel 计时；本脚本的 `TRACE` 默认 0，不代表正常请求总会收集这些详细统计。

#### 6.2. MoE 的理论总运算量与每个 GPU 的工作量不同

设全组源模型行数为 `T_group=Σ_r T_r`，包括统一 decode 的源端 dummy。在同一模型层、每 token 固定 8 routed + 1 shared、expert 尺寸相同且没有丢弃 route 时：

```text
全组有效 token-expert route 总数 = 9 × T_group
```

因此，全组 T 相同可以确定全组理论有效 GEMM 运算量相同，不取决于这些 token 原来属于多少个请求。MoE 不按请求的 KV 长度计算 expert FFN。

但 **仅知道某个来源 rank 的 T，不能确定该 GPU 的 MoE 工作量**。EP4 的该 GPU 处理的是所有来源 rank 分配给它的本地 64 routed expert 的 routes，加上本来源 rank 的 shared routes；实际接收去重、expert 分布、通信流量和 expert tile padding 都可能不同。甚至全组 T 相同，但各来源 rank 的 T 分布不同，也会改变 shared replica 的负载及各 GPU 分工。

grouped GEMM 的实际 M tile 覆盖还受 `Σ_e ceil(M_e/tile_m)` 影响：128 条 routes 分布为 `[64,64]` 只需 2 个 tile，分布为 `[63,65]` 则需 3 个。这里 route 总数不变，但 padding 工作不同；当前实现用 `_align_m` 和 per-expert mask 做对齐，见 `grouped_moe_gfx1250.py:648–660`、`kernels/moe_contiguous_psum.py`。

#### 6.3. FAKE_EPLB=1 下的比较边界

`moe.py:136–189,3628–3645` 按 DP rank 生成交错的 fake routing 表，其中全局 token 索引形如 `t=dp_rank+i*dp_size`，并遵守模型的 group-limited routing；forward 使用本 rank 表的前 `hidden_states.shape[0]` 行（第 5117–5131 行）。

所以，在同一层、各来源 rank 的源模型行数逐项相同、执行模式和配置不变时，改变各 rank 内部请求长度分布不会通过真实 hidden states 改变 fake expert 分布，可将对应 expert/tile/dispatch 工作视为相同。GEMM 仍使用实际 hidden states，输出值不要求相同，耗时也不保证相同。

只要求全组 token 总数相同还不够：来源 rank 分配变化会改变 fake 表前缀组合和 shared 归属。比较 graph bucket 不同时也应计入不同的源端 dummy，不能只比较有效请求数。本节没有运行 GPU 性能对比，例子均为源码分析或理论计算。

### 7. 每次 prefill/decode 的 batch size 如何确定

#### 7.1. 先将客户端请求分配给 DP rank

客户端请求由 engine manager 分配到四个 DP rank，各 rank 的 waiting/running 队列独立。默认负载策略为 **`least_requests`**：优先在途请求最少的 rank，请求数相同再比较在途 prompt-token 负载，完全相同则按旋转游标解平局。源码为 `engine_core_mgr.py:36,900–928,984–1025`。

这不是固定 round-robin，也不能保证任意时刻 6144 个客户端请求恰好每 rank 1536 个。请求数包含等待、prefill、decode 等状态；真实每轮 batch 由各 rank scheduler 再决定。

每 rank 上限为 2048 个活跃 sequence、每轮新 query token 预算 16384，四 rank 理论活跃名额合计 8192。这个数不保证 KV cache 能容纳任意长度的 8192 个请求，实际还受模型长度、输出增长、KV 分配、preemption 和完成/取消等因素影响。

#### 7.2. prefill：完整 prompt 装入每 rank token 预算

当前没有 chunked prefill 和 prefix caching。正常新请求从 waiting 队列依次接纳，须满足 token 预算、活跃 sequence 名额、KV blocks 和静态长度等限制：

- `scheduler.py:1604–1610` 限制本轮 prefill 请求数及 token 总数。
- 第 1652–1654 行的 `if len(self.running) >= self.max_num_seqs:` 表示**已经满额时停止接纳**，将请求放回 waiting 队首并 `break`。不是写错比较号；继续接纳的数量条件是 `< 2048`，等于上限也不能再接纳。已有活跃请求会占用名额。
- 第 1778–1788 行在下一完整 prompt 装不进剩余预算时放回队首并结束接纳，不继续扫描后面的短请求填空。超过完整单步预算的输入还受第 1396–1410 行静态拒绝检查。
- 第 2594–2611 行将接纳的 prefill sequence 放入 running；第 2055 行直接返回 prefill batch，本 rank 本轮不再调度 decode。

设 rank r 本次 prefill 请求数为 `B_prefill,r`、各请求本轮输入长度为 `L_i`：

```text
T_prefill,r = Σ L_i
```

在新请求实际 prompt 长度均为 L、资源足够的简化场景：

```text
B_prefill,r ≤ min(
    本 rank 本轮已到达且可接纳的等待请求数,
    2048 - 接纳前本 rank 已有 running 请求数,
    floor(16384 / L)
)
```

若实际待 forward 的每个 prompt 恰为 1024 token，每 rank 每轮 prefill 最多 16 个，四 rank 理想同步装满最多 64 个请求。实际可以更小，且不是客户端 concurrency6144 的整批 prefill。prefill 的源模型使用实际 token 行数，不按 decode capture sizes 补齐；`running_bs` metadata 可能取 bucket 值，不代表 prompt token 行数也被补齐。

#### 7.3. DP PrefillDelayer 默认启用

`scheduler_delay_factor=0.0` 不能解释成“当前 DP4 没有 prefill 等待/合批机制”。`ATOM_ENABLE_PREFILL_DELAYER` 默认 1，DP engine 在 `engine_core.py:197–219,750–760` 初始化 PrefillDelayer；脚本没有关闭它。

当前默认目标 fill 为 0.9，hold 上限为 200 scheduler ticks，不增长阈值为 10 ticks，KV high watermark 为 0.9；partial 上限为 100 ticks，但本次不开 chunked prefill，通常不涉及该分支。固定 prefill 后 decode 保护间隔默认 0，queue-ms 与 token-usage-low-watermark 默认未启用（`envs.py:705–763`）。

`prefill_delayer.py:330–440` 跨 rank 汇总 pending tokens、decode 活跃数和 KV 等信息。第一次允许 prefill；全组没有 decode 时立即允许。有 decode 时通常等待所有 rank 可 prefill，且平均 pending token fill 达到 0.9；也会因 KV 高水位、连续 200 hold ticks 或已对齐但积累不增长 10 ticks 等条件释放。hold prefill 期间 scheduler 可以继续 decode。

这些 must-fire 条件可能在 rank 尚未对齐时释放，实际 KV 分配和队列变化也会影响结果，所以仍可能出现某 rank prefill、其他 rank decode；该轮统一 decode replay 不成立。200 ticks 不是 200 毫秒，也不是整个请求 TTFT 的硬保证。

#### 7.4. decode：从本 rank 可继续生成的活跃请求中选择

当本 rank 本轮没有选到 prefill 时，`scheduler.py:2057–2099` 才从 running 选择 decode：请求需完成 prefill、未结束并可扩展 KV；partial、阻塞或 preempted 的请求可能不参加本轮。

```text
B_decode,r ≤ min(
    本 rank 本轮可 decode 的活跃请求数,
    2048,
    floor(16384 / q_sched)
)
```

本次 `q_sched=qlen=1`，因此这两个配置中 sequence 上限 2048 先约束 decode；实际可因 KV、到达/完成、取消等因素减少。确定各 rank 的真实 batch 后，CPU DP metadata 同步再选最小容纳全组 `B_max` 的 graph bucket；bucket 不是调度必须凑满的真实请求数。

一个完整生成请求包含一次普通 prefill及多步 decode；其他请求会不断加入或退出合批，因此该请求 prefill 时和以后每步 decode 时的 batch 成员与数量不要求相同。

#### 7.5. benchmark 参数与 batch size 的关系

当前 `bench_dsr1fp4.sh:10–14,36–48` 在只传模型路径时采用：

| 客户端参数 | 本次值 |
|---|---:|
| `CONC` / `--max-concurrency` | 6144 |
| `--random-input-len` | 1024 |
| `--random-output-len` | 4096 |
| `--random-range-ratio` | 1.0 |
| `--num-prompts` | `CONC × 3 = 18432` |
| `--request-rate` | `inf` |
| `--ignore-eos` | 开启 |
| HTTP port | 8000 |

ratio1.0 使 random dataset 的目标采样长度固定为 1024/4096（`benchmark_serving.py:172–243`），但服务端实际 prompt 行数仍按最终 tokenizer/请求内容计算；第 7.2 节的 16 请求例子以实际待 forward 长度恰为 1024 为前提。

6144 表示客户端最多保持这么多请求在途，不固定每轮 prefill 或 decode batch 为 6144；18432 是整个 benchmark 的请求总数；4096 是每请求目标输出长度，不是单步 decode qlen 或 MoE 行数。只有在仅有这批请求、全部可 decode 且均衡分配等条件成立时，才可用每 rank 1536 的 bucket 例子。

`--backend=vllm` 只是客户端协议选择，服务端仍是 ATOM。benchmark 参数不改变服务的 cudagraph、EP、MTP 或 scheduler 配置。

### 8. FULL decode 图的范围与 DP LM head

#### 8.1. 每 rank 每 bucket 捕获一次整模型 forward

`model_runner.py:3999–4015` 在同一 graph context 中执行整段模型、输出写入，以及满足条件时的 logits：

```python
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, self.graph_pool, stream=capture_ctx.stream):
    model_output = self.model(input_ids[:num_tokens], model_positions)
    outputs[:num_tokens] = model_output
    if self.logits_in_graph:
        graph_logits = self.model.compute_logits(outputs[:num_tokens])
self.graphs[(bs, max_q_len)] = graph
```

这是省略外围代码的源码摘录。每个 rank 保存自己的 `(B,1)` graph，四 rank 的图协同执行通信；不是一个跨四 GPU 的单一 graph 对象，也不是只把一个 MoE 或一个 transformer layer 作为标准 FULL graph 的边界。

`deepseek_v2.py:3345–3358,3385–3410` 在 PP1 下执行 embedding、全部 61 layers 和 final norm，其中 58 个 MoE 都在这次模型调用内。因此本次每 rank 图的范围可表示为：

```text
embedding
→ transformer layers 0–2：norm / attention / residual / dense MLP
→ transformer layers 3–60：norm / attention / residual / MoE
   每个 MoE 含 router/topk、shared remap、FP4 EP dispatch、E65 GEMM1/GEMM2、combine
→ final norm
→ outputs buffer 写入
→ DP LM head / logits
```

一张图可以含许多 kernel 和通信节点；整模型 capture 不意味着把全部算子融合成一个 kernel。同一图 replay 仍执行这些节点，但不重跑它们的 Python kernel 选择逻辑。

#### 8.2. 当前 LM head 在图内，并使用 DP vocab 分工

`model_runner.py:638` 的 `self.world_size` 取实际 `config.tensor_parallel_size`，DP attention 转换后为 1；第 3793–3798 行设置：

```python
is_tbo = self.config.enable_tbo and isinstance(self.model, UBatchWrapper)
self.logits_in_graph = self.world_size == 1 and not is_tbo
```

本次未启用 TBO，故 **`logits_in_graph=True`**。不能看到 CLI `-tp 4` 就推断 LM head 在图外。

脚本的 `ATOM_DP_LM_HEAD_MODE=allgather` 在全组统一 decode、有效 TP1、DP4、vocab 可整除时生效（`embed_head.py:355–427,458–462`）。模型 `vocab_size=129280`，每 rank 计算其中 `32320` 个 vocab logits：

```text
每 rank B 行 final hidden states
→ DP all-gather hidden，得到 4B 行
→ 本 rank 使用自己 vocab slice 做 GEMM，得到 [4B,32320]
→ 沿 vocab 维度 DP all-gather，得到 [4B,129280]
→ 保留本来源 rank 的 B 行 logits
→ replay 后再裁剪为真实 scheduled token 行
```

这个版本在每 rank 仍加载完整 LM head weight，运行时取 vocab slice；DP 分工不表示加载时已把权重存储缩为四分之一（`embed_head.py:350–353,421`）。相关 all-gather 和 GEMM 在本次 FULL decode 图内。

prefill或跨 rank 混合轮次不满足 uniform 条件，使用普通 replicated full-vocab head；prefill先取每请求最后一行 hidden state（`embed_head.py:255–266`）。不能让只有本地 decode 的 rank 在混合轮次自行进入固定大小 DP head collective。

#### 8.3. 图外工作与有效输出

请求分配、scheduler、CPU DP metadata 同步、本轮 token ID 和 attention metadata 准备在整模型图之前；sampling、停止条件、请求状态更新在图外。`model_runner.py:2980–2988` replay 后按 `scheduled_tokens` 裁剪 hidden states/logits。

图覆盖一次模型 forward，不覆盖一个请求从 prefill 到生成结束的完整生命周期，也没有一次捕获全部 decode iterations。上述范围针对当前标准 FULL、PP1、未启用 TBO/MTP 的配置；改用其他 graph mode 或并行方式后须重新核对。

### 证据与版本记录

所有结论以本次 `dsr1_ep4` 容器的静态快照为依据。完整材料位于工作区：

- [EP4 调查记录](../../.codex_tmp/dsr1_ep4_investigation/INVESTIGATION.md)：验证范围、关键推导及快照清单。
- [EP4 源码摘录](../../.codex_tmp/dsr1_ep4_investigation/SOURCE_EXCERPTS.md)：本文使用的远端原始路径、行号、源码、模型配置与 SHA256。
- [脚本与模型快照](../../.codex_tmp/dsr1_ep4_investigation/scripts_model.json)：两个脚本原文、模型 config 与过滤后的容器环境。
- [CPU CSV 匹配结果](../../.codex_tmp/dsr1_ep4_investigation/grouped_config_check.json)和[查询脚本](../../.codex_tmp/dsr1_ep4_investigation/check_grouped_config.py)：113 条 data rows、15 组查询及 bucket→接收上界→CSV key 映射；仅标准库，不访问 GPU。
- [远端源码本地镜像](../../.codex_tmp/dsr1_ep4_investigation/sources/)：快照源码，解包时逐文件校验 SHA256；本文引用的行号是远端原文件行号。

本次脚本及模型 config 的 SHA256 为：

```text
serve_dsr1fp4_a8w8_dp4ep4.sh
adb07e0b40f8bfa770a2d3c4ed39b3b493c01c7f9d3781772ae85e29510b3400

bench_dsr1fp4.sh
61859fbd23f89778b1657a9adc12765b6ff76a7c0aebbc38fcbead47d5dc1d6c

/data/models/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4/config.json
65c8f5958031046fa63cd419b86e2f8e6d4d3155d76068eeacfb6313613df2d5
```

保留原 [TP1 文档](moe_framework_wiki.md) 作为独立部署说明。本次只修订 EP4 文档；没有将其结论反向写入 TP1 文档，也没有用旧 TP1 日志或 CSV 查询结果证明 EP4 的运行行为。
