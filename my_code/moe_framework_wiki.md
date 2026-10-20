# MoE Framework Wiki

## DSR1：cudagraph 与 GEMM1/GEMM2 dispatch（2026-10-12）

本节记录在 `a07-3`（`yanguahe@heliosr-1b114-a07-3.mnb.dcgpu`）的 `hyg_fyd_e2e` 容器中，对 DSR1 服务脚本、benchmark 脚本、ATOM/AITER 源码和配置的调查结果。结论适用于调查时的容器版本；后续升级框架、修改脚本或 dispatch 配置时应重新核对。

**结论：这套启动命令默认开启 `FULL` cudagraph，但当前 prefill 普通执行，decode 按 batch bucket 重放整段 model forward 图。prefill 可以按该步 MoE 输入 token 行数选择 GEMM1/GEMM2；不同 decode bucket 也可以捕获不同 kernel，同一张图 replay 时不会重新执行 Python dispatch。当前容器的 DSR1 配置中，所有 decode bucket 都使用默认 `t64/w1x4/b3`，尚未按 decode bucket 配置不同 GEMM 版本。**

### 调查命令与验证范围

服务端命令：

```bash
cd /app/scripts/dsr1
FAKE_EPLB=1 \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
bash /app/scripts/dsr1/serve_dsr1fp4_a8w8_tp1.sh \
  /data/models/DeepSeek-R1-0528-MXFP4
```

benchmark 命令：

```bash
cd /app/ATOM/scripts/performance
bash /app/scripts/dsr1/bench_dsr1fp4.sh \
  /data/models/DeepSeek-R1-0528-MXFP4
```

调查只读检查了这两个脚本、源码、模型 config、CSV、既有日志和进程状态，并用当前源码的匹配函数做了纯 CPU 配置查询。没有执行上述服务或 benchmark 命令，也没有运行新的 GPU 测试。调查时未发现 ATOM 服务或 benchmark 进程。

既有 `/app/scripts/dsr1/server.log` 来自2026-09-29，其 mtime 为 `2026-09-29T01:07:16.041064+00:00`。日志可佐证此前成功 capture，不能作为本轮服务正在运行或已重新实测的证据。

### 1. 服务端是否开启 cudagraph

`/app/scripts/dsr1/serve_dsr1fp4_a8w8_tp1.sh:54–55` 启动 `python -m atom.entrypoints.openai_server`，没有传入 `--enforce-eager` 或覆盖 `--cudagraph-mode`。ATOM 的默认设置为：

| 设置 | 此命令使用的值 | 远端源码依据 |
|---|---|---|
| `enforce_eager` | `False` | `/app/ATOM/atom/model_engine/arg_utils.py:50` |
| `--cudagraph-mode` | `FULL` | `/app/ATOM/atom/model_engine/arg_utils.py:252–256` |
| `--cudagraph-capture-sizes` | `[1,2,4,8,16,32,48,64,128,256,512]` | `/app/ATOM/atom/model_engine/arg_utils.py:243–246` |

这里使用 CLI 默认值。同文件第69行 dataclass 的默认列表没有512，但该服务通过 CLI 启动，第245行的列表生效。

`/app/ATOM/atom/model_engine/engine_core.py:148–155` 在非 `enforce_eager`、非 disaggregated decode 初始化时调用 `capture_cudagraph`。旧 `server.log:8` 记录了 `enforce_eager=False`、`use_cudagraph=True`、`CUDAGraphMode.FULL`；第2779行记录成功捕获上述11个 bucket，capture 耗时4.33秒，pool为0.87GB。这些数值属于9月29日的运行。

`FAKE_EPLB=1` 在服务脚本第14–16行转为 `--fake-eplb --hf-overrides '{"num_hash_layers": 0}'`，不是 cudagraph 开关。`AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1` 从 shell 继承到服务进程，控制 GEMM launcher 的 `amdgpu-expert-scheduling-mode` 编译选项，也不控制 capture/replay 或运行时 token dispatch。

`bench_dsr1fp4.sh` 默认 concurrency64、ISL1024、OSL1024，发送256个 HTTP 请求到8000端口。它不决定服务端 cudagraph 模式；其中 `--backend=vllm` 是 benchmark 客户端协议选择，服务端入口仍然是 ATOM。

### 2. prefill 与 decode 的执行路径

关键判定来自 `/app/ATOM/atom/utils/forward_context.py:248,303–311`：

```python
is_prefill = batch.total_tokens_num_prefill > 0
rung = int(np.searchsorted(capture_sizes, unified_bs))
on_ladder = rung < len(capture_sizes)
running_bs = int(capture_sizes[rung]) if on_ladder else unified_bs
use_cudagraph = not is_prefill and unified and not enforce_eager and on_ladder
```

因此，`FULL` 在这里表示捕获整段 model forward，不表示 prefill 和 decode 都会 replay。

| 阶段 | 当前执行方式 | GEMM1/GEMM2 的选择时机 |
|---|---|---|
| prefill | 普通模型调用，`running_tokens=scheduled_tokens` | 每次调用 MoE 时，可以根据该步输入行数查询配置 |
| decode capture | 对各 bucket 分别执行模型并捕获图 | capture 期间执行 dispatch，各张图可以记录不同 kernel |
| decode replay | 选择已有 bucket 图并 replay | 使用捕获时的 kernel，不再执行 Python dispatch |

prefill 普通调用位于 `/app/ATOM/atom/model_engine/model_runner.py:2789–2791,2850–2855`。默认 FULL 不将 compiled pieces 包成单独的 `CUDAGraphWrapper`，依据是 `/app/ATOM/atom/utils/backends.py:423–448`。因此，此处的 `torch.compile` 执行不能等同于 CUDA/HIP graph replay。若以后启用真正的 prefill graph replay，该阶段也要遵守同样的 capture/replay 约束。

decode 的 `running_tokens=running_bs*max_seqlen_q`，见 `forward_context.py:370–371`。本次命令没有 speculative config，普通 decode 的 `max_seqlen_q=1`，所以 capture 时 MoE 的输入行数就是 bucket batch size。

`model_runner.py:3795,3805,3933–3947` 对各 bucket 调用模型并 capture，保存为 `self.graphs[(bs,max_q_len)]`。实际 decode 在第2942–2944行重放：

```python
graph_key = (running_bs, forward_context.attn_metadata.max_seqlen_q)
self.graphs[graph_key].replay()
hidden_states = self.forward_vars["outputs"][:scheduled_tokens]
```

graph 中记录的 GEMM kernel、launch 配置和输入 shape 在 capture 时已经确定。输入值、topk、expert 路由和各 expert 的有效行数仍可逐步变化，但这些 GPU 数据变化不会触发当前 Python launcher 重新选择 GEMM 版本。

### 3. graph bucket、真实 token 数与 CSV token key

普通 decode 每个序列每步输入一个 token。框架选择最小的、能容纳当前 batch 的 capture bucket：

| 该步实际 decode token 数 | 使用的 graph key | capture 时 MoE 输入行数 | CSV 查询使用的 token key |
|---:|---|---:|---:|
| 33–48 | `(48,1)` | 48 | 64 |
| 49–64 | `(64,1)` | 64 | 64 |
| 65–128 | `(128,1)` | 128 | 128 |

33个和48个有效 decode token 使用同一张48行图，不会在该图 replay 时分别重新选择 kernel。49个和64个有效 token 同理，使用64行图。

graph bucket 与 CSV token key 是两层不同的机制。`/app/aiter/aiter/ops/flydsl/grouped_moe_gfx1250.py:1579–1614` 从 `topk_ids.shape` 读取 token 行数，再用 `_get_padded_m(token_num)` 查询 CSV；低于32768时查询 key 为 next power of 2，见同文件第119–125行。因此48行图的 CSV key 为64，但这次查询不会把模型输入再改为64行。

benchmark 的 `ISL=1024` 表示单请求 prompt 长度，不表示每一步 decode 的 MoE 都有1024行。prefill dispatch 使用该次 forward 处理的总 token 行数，多个请求合批时可以超过1024；concurrency64时，普通 decode 该步通常不超过64个有效输入 token。

#### 3.1. decode 的 padding 在进入 MoE 前由框架完成

对本节 TP1/PP1、非 speculative 的正常 decode 路径，ATOM 在 replay 前已经准备好 graph bucket 所需的输入 buffer 和 attention metadata。以63个有效 decode token 为例，框架选择64行图，进入 MoE 时 `hidden_states.shape=(64,7168)`，MoE 无需自行把总输入从63行扩为64行。

`/app/ATOM/atom/model_engine/model_runner.py:581–598` 的 `CUDAGraph tail padding` 代码计算 bucket 对应的 `fill_to`，将实际 token 数到 bucket 大小之间的 token ID 设为0：

```python
if fill_to > total_tokens_decode:
    self.input_ids.gpu[total_tokens_decode:fill_to].zero_()
```

这里补齐的是模型输入的 token ID buffer。64行模型图在 capture 时已固定输入 shape，replay 时沿用该 shape，因此 padding 行会经过 embedding 和前面的模型层，并作为一行 hidden state 进入 MoE：

```text
框架准备：63个真实 token ID + 1个 dummy token ID（值为0）
→ 重放64行模型图
→ 进入 MoE：hidden_states.shape = (64, 7168)
→ 模型输出后只保留前63行
```

框架同时处理 dummy 序列的 attention metadata：

- `/app/ATOM/atom/model_ops/attentions/backends.py:556–580` 的 `publish_cu_seqlens_q` 将尾部 cumulative query offset 设为实际总 token 数，使 dummy 序列的 query length 为0。
- `/app/ATOM/atom/model_ops/attentions/aiter_mla.py:2211–2216` 将 `slot_mapping` 先设为 `-1`，只为有效 token 写入真实 slot，并将 dummy 序列的 `context_lens` 设为0。

replay 后，`/app/ATOM/atom/model_engine/model_runner.py:2944` 截取有效输出，dummy 行的最终结果被丢弃：

```python
hidden_states = self.forward_vars["outputs"][:scheduled_tokens]
```

**token ID 为0不代表进入 MoE 的 dummy hidden state 为0。** 经过 embedding 和前面的层后，它可能是非零值；MoE 仍会处理这行，并可能为它选择 expert。因此 kernel 不能假定尾部 hidden state 全为0，也不能将所有64行都当作最终有效输出。

框架完成的是总 token 行数的补齐。routing 后各 expert 的有效行数不一定是2的幂，expert 分组、GEMM partial-M 和 tile padding 仍需由 MoE/kernel 正确处理。63个有效 decode token 可以由64行图覆盖，但这不等于只支持2的幂就能覆盖整个服务：默认还会 capture48行图，当前 prefill 也使用该步实际 token 行数。

#### 3.2. dummy 行与 BOS token 的 embedding

在当前 DSR1 的 TP1 路径中，dummy 行的 token ID 被填为0，而模型的 `bos_token_id=0`，`tokenizer.json` 中 ID 0 对应 `<｜begin▁of▁sentence｜>`。**token ID 0 是合法的 BOS token，不天然表示无效；真实 BOS 行与 padding 行的区别由框架的调度信息、attention metadata 和有效输出行数决定。**

两者刚经过 embedding 时得到相同的向量，都读取同一张 embedding 权重表的第0行：

```text
dummy 的 token ID = 0 → W_embed[0]
BOS   的 token ID = 0 → W_embed[0]
```

`/app/ATOM/atom/models/deepseek_v2.py:3132–3133` 的 `get_input_embeddings` 直接调用 `self.embed_tokens(input_ids)`。其 `VocabParallelEmbedding.forward` 在 TP1 分支使用 `/app/ATOM/atom/model_ops/embed_head.py:178–180` 的普通查表：

```python
y = F.embedding(x, self.weight)
return y
```

这个调用只接收 token ID 和权重表，不读取 dummy 标记或 attention metadata，不会因为某行属于 padding 就将其 embedding 清零。因此，dummy 行的 embedding 与真实 BOS 的 embedding 相同，也不保证是零向量。

相同仅指 embedding 刚输出的向量。之后，真实 BOS 行和 dummy 行使用的 position、attention metadata、有效上下文可能不同，经过后续模型层后，进入 MoE 的 hidden state 不一定相同。dummy 行最终被丢弃，是框架依据 metadata 和 `scheduled_tokens` 处理的，不是通过 embedding 向量识别它。

#### 3.3. dummy 行在 MoE 内参与正常 routing 和 GEMM 计算

对当前 TP1、非 EP 的 grouped MoE 路径，dummy 行和有效 token 行使用相同的 MoE 处理规则。MoE 不根据某行是否为 dummy，单独执行跳过或清零分支。以63个有效 token 使用64行图为例，MoE 看到的是64行输入，没有收到“只计算前63行”的 token mask，所有行都参与以下流程：

```text
router / topk
→ expert 分组与输入 quant
→ GEMM1 + activation
→ GEMM2
→ 按 routing weight 合并结果
```

`/app/ATOM/atom/models/deepseek_v2.py:1223–1233` 对传入的整份 `hidden_states` 计算 gate，并将 hidden states 和 router logits 一起传给 `self.experts`。`/app/aiter/aiter/ops/flydsl/grouped_moe_gfx1250.py:610` 从 `topk_ids.shape` 取得总行数，当前非 EP 分支在第782–784行对整份 `topk_ids` 建立 expert 分组。因此，dummy 行也会占用 expert 的计算行数。

这里的“计算逻辑等价”指处理规则相同，不意味着 dummy 行与某个有效 token 的输入、选中 expert、routing weight 或输出数值相同。dummy 行在框架层面的最终有效性，不由这个 MoE 计算路径判断；它的结果在模型输出按 `scheduled_tokens` 截取时被丢弃。

**dummy 行的 MoE 输入一般是非零向量，不能假定整行为0。** token ID 0 只决定最初读取 `W_embed[0]`，进入 MoE 时的数据已经是前面模型层产生的 hidden state。框架没有因为该行属于 dummy，就将其 MoE 输入清零。具体某些元素可能为0，quant 后也可能出现0；本次没有 dump 各层 dummy hidden state，不能据此声称每层、每行都必然非零。

本次命令还设置了 `FAKE_EPLB=1`。`/app/ATOM/atom/model_ops/moe.py:4739–4753` 的 `FusedMoE.forward` 将所有行的 router logits 替换为预先生成的人工均衡 logits：

```python
router_logits = self.balance_router_logits[: hidden_states.shape[0]].to(
    router_logits.dtype
)
return torch.ops.aiter.moe_forward(
    hidden_states, router_logits, self.layer_name
)
```

dummy 行和有效行都使用这个 fake routing 机制，GEMM 仍计算各行实际的 hidden state。这个设置不会将 dummy 行标记为无效或将其输入清零。

### 4. 当前容器的 DSR1 GEMM 配置

容器实际加载的包为 `/app/ATOM/atom` 和 `/app/aiter/aiter`，没有使用历史开发 worktree `/data/yanguahe/code/wk_sp1/aiter_t256w2x4_e64_20261007`。当前容器的 grouped MoE 源码也没有该开发树最新的 E64 token interval table，不能将 E64/topk8 的开发结果直接套到真实 DSR1。

模型 config 为 `hidden_size=7168`、`moe_intermediate_size=2048`、`n_routed_experts=256`、`num_experts_per_tok=8`、`n_shared_experts=1`。当前 shared 与 routed expert 使用相同的动态 per_1x32 FP4 quant spec，没有 shared-expert exclude 或 layer quant override；TP1/DP1 下 shared expert 融合后，grouped MoE 使用 **E257/topk9**。未融合时为 E256/topk8。shared fusion 的依据是 `/app/ATOM/atom/model_ops/topK.py:16–86,227–269` 和 `/app/ATOM/atom/model_ops/moe.py:3193–3241`，旧日志也实际记录了 E257/topk9。

服务脚本设置 `AITER_FORCE_A8W4=0`；`/app/aiter/aiter/fused_moe.py:802–806` 在 gfx1250 上据此选择 FP4 activation quant dtype，因此这里的 grouped MoE 实际走 **A4W4**。脚本名称中的 `a8w8` 不代表 MoE GEMM 使用 A8W8。

默认 CSV 为 `/app/aiter/aiter/configs/tuned_grouped_fmoe.csv`，共120行；调查时没有对应的 `model_configs/*tuned_grouped_fmoe*.csv` 或相关配置路径、tile override。CSV 匹配还要求 M/I/E/topk、activation、dtype、quant type 和硬件 key 一致，不能只看 token 列。

纯 CPU 检查从当前源码 AST 提取 `_next_pow2`、`_get_padded_m` 和 `_find_grouped_config`，执行原始匹配函数，共检查 E257/topk9、E256/topk8 的34组查询。两种 expert/topk key 的结果一致：

| MoE 输入行数 T | CSV token key | 当前 GEMM1 / GEMM2 配置 |
|---|---|---|
| decode buckets：1,2,4,8,16,32,48,64,128,256,512 | 对应 next power of 2 | 全部无匹配 row，均使用默认 `t64/w1x4/b3` |
| prefill `T≤8192` | 对应 next power of 2 | 无匹配 row，均使用默认 `t64/w1x4/b3` |
| prefill `8192<T≤16384` | 16384 | 两阶段均使用 optimized `t256/w2x2/b4` |

`tN` 表示 tile `(N,256,256)`，`wAxB` 表示 `(m_warp,n_warp)`，`bN` 表示 buffer 数。CSV 可以分别通过 `tile_m2/m_warp2/n_warp2/num_buffer_stage2` 配置 GEMM2，所以 GEMM1 和 GEMM2 不要求使用同一版本。

默认 b3 来自 `_grouped_a8w4_tdm_moe` 的函数默认值（`grouped_moe_gfx1250.py:565–570`）。上层无匹配日志打印的局部 `num_buffers=2` 没有在无 CSV row、无 env override 时传入该函数，见第1645–1646、1743–1768行；实际 fallback 使用 b3，不能把日志中的 b2 当作 launch 参数。

DSR1 的16384 row 配置两阶段 tile256、warp2×2、buffers4、`cluster_n=4`、`next_stage_prefetch=1`，符合 optimized launcher 条件。当前 persistent launcher 的选择还要求 E64、t192 等条件（`grouped_gemm_mxfp4.py:338–377`），这两条 DSR1 row 不会命中该 E64 persistent 分支。

### 5. 后续按 token 数切换 kernel 的实现条件

prefill 每次普通调用 MoE 都可以按该步输入行数选择 GEMM1/GEMM2；当前 CSV 已存在 token key16384的配置切换。

decode 可以为不同 capture bucket 配置不同 kernel，在启动时各 bucket 的 capture 期间执行 dispatch，各张图记录对应版本。机制支持这种选择，但当前 DSR1 的所有 decode bucket 都走同一套默认配置。

若同一 bucket 内也要根据真实有效 token 数选择不同版本，需要细化 capture bucket，或在图外选择额外预捕获的图。当前 `graph_key=(running_bs,max_seqlen_q)` 没有额外的版本选择，同一张图 replay 不会自动替换 GEMM kernel。

修改 dispatch 源码、CSV 或编译环境后，要让已有 decode 图反映新选择，需要重新加载配置并 capture，通常通过重启服务完成。CSV 还有进程内 cache，运行中直接编辑文件也不能保证普通调用立即读取新配置。

### 6. prefill 请求长度分布对 attention 与 MoE 计算量的影响（2026-10-13）

**同一次 prefill forward 中，各请求的输入长度可以不同。总 token 数相同、请求长度分布不同，attention core 的计算量可以不同；同一模型层、固定 topk 和 expert 尺寸时，MoE 的理论有效 GEMM 运算量相同，但实际 kernel 工作量还可能受 routing 和 expert tile padding 影响。当前 `FAKE_EPLB=1` 在同一层、相同输入行数和配置下固定 fake routing，可将对应 MoE 的计算工作量视为相同，实际耗时仍可能波动。**

这里的“各请求长度”指同一个 batch 内各请求/序列的长度；“一次推理”指一次 model forward。以下比较使用进入算子的总 token 行数；若模型输入包含 graph padding，MoE 的 `T` 必须计入这些 dummy 行，而不能只使用真实有效 token 数。当前这套命令的 prefill 普通执行，使用该步实际调度的 token 行数。

#### 6.1. attention core 依赖各请求的 query/KV 长度

attention 根据请求边界和 attention metadata 分别处理各序列；一个请求的 query 只关注它自己的 KV。对没有历史 KV 的完整 dense prefill，在 head 数和维度固定时，设各请求长度为 `L_i`：

```text
总输入 token 数：T = Σ L_i
attention core 的主要运算量 ∝ Σ L_i²
causal attention 的有效 query-key 对数 = Σ [L_i × (L_i + 1) / 2]
```

因此，相同 `T` 不能确定相同 attention core 计算量。例如：

| 两次 forward | 各请求长度 | 总 token 数 T | Σ L_i² |
|---|---|---:|---:|
| A | `[512,512]` | 1024 | 524288 |
| B | `[256,768]` | 1024 | 655360 |

B 的 `Σ L_i²` 比 A 多25%，attention core 的理论主要运算量约多25%。这个比例不代表整个 attention 层或实际耗时也多25%：投影等部分的运算量主要随总 token 数线性变化，kernel 的 tile padding 和执行效率也影响实际耗时。

如果是 chunked prefill 或存在 prefix cache，还需要考虑每个请求已有的 KV 长度；仅看本次输入长度分布也不足以确定 attention 工作量。ATOM 的 `/app/ATOM/atom/model_engine/scheduler.py:3001–3015` 分别累计 `Σ N_Q²` 和 `Σ N_Q × N_KV`，其中 `N_Q` 是本次为该请求调度的 query token 数，prefill 的 `N_KV` 为已有 cached token 数加 `N_Q`。这些统计是源码事实；上面的平方项和 causal pair 数是对应 dense attention 的数学推导，不能直接等同于实际 kernel 耗时。

#### 6.2. MoE 的理论有效 GEMM 运算量由总 token 数与 topk 决定

在同一模型层、固定 `topk=k`、expert 尺寸相同且没有丢弃 route 的前提下，每个 token 执行相同数量的 expert FFN。设总输入行数为 `T`，各 expert 收到的有效行数为 `M_e`：

```text
Σ M_e = T × k
```

因此，两次 forward 的 `T` 相同，理论有效 GEMM1/GEMM2 运算量相同，不依赖这些 token 原来属于多少个请求、各请求长度是多少。当前 TP1、非 EP 的 MoE 路径不读取请求的 `cu_seqlens` 或 KV context 长度：`/app/ATOM/atom/models/deepseek_v2.py:1223–1233` 对整份 `hidden_states` 计算 gate 并传给 experts；`/app/aiter/aiter/ops/flydsl/grouped_moe_gfx1250.py:610` 从 `topk_ids.shape` 读取 `token_num, topk`。

本节 DSR1 配置中，routed experts 为 E256/topk8，shared expert 融合后 grouped MoE 为 E257/topk9。只要比较的是相同模型层和配置，shared expert 的计算也不会因为请求长度分布变化而改变理论有效运算量。

#### 6.3. 相同有效运算量不保证相同实际 kernel 工作量或耗时

不同请求长度分布可能使前面的 attention 产生不同 hidden states，从而改变真实 routing，导致各 expert 的 `M_e` 不同。grouped GEMM 按 expert 的 M 维度做 tile 对齐，实际工作量会受到下面这个量的影响：

```text
Σ ceil(M_e / tile_m)
```

以 `tile_m=64`、总共128条 token-expert route 的简化例子说明：

| 两个 expert 的有效行数 | 有效 route 总数 | 覆盖的 M tile 数 |
|---|---:|---:|
| `[64,64]` | 128 | 2 |
| `[63,65]` | 128 | 3 |

有效 route 总数相同，但尾部 tile 的 padding 工作量不同；具体还要遵守 GEMM1/GEMM2 各自的 tile 和共同对齐要求。`grouped_moe_gfx1250.py:837–845` 将 per-expert 的 `_masked_m` 传给 `contiguous_psum_remap`，按 `_align_m` 对齐；`/app/aiter/aiter/ops/flydsl/kernels/moe_contiguous_psum.py:655–658` 对 expert 行数使用 `(m + tile_minus_1) // tile_v * tile_v`。因此，即使理论有效 GEMM 运算量相同，也不能忽略 expert tile padding。routing 分组、访存和并行执行效率也可能影响耗时。

#### 6.4. 当前 FAKE_EPLB=1 下的适用结论

同一服务进程、同一模型层、配置不变且输入行数 `T` 相同时，ATOM 使用同一份预生成 fake logits 的前 `T` 行替换真实 router logits，见 `/app/ATOM/atom/model_ops/moe.py:4739–4753`：

```python
router_logits = self.balance_router_logits[: hidden_states.shape[0]].to(
    router_logits.dtype
)
```

因此，请求长度分布不会通过 hidden states 改变这条路径的 expert 分布，对应的 expert 行数和 tile padding 也相同，可以把 MoE 的计算工作量视为相同。GEMM 仍使用各行实际 hidden state，输出数值不要求相同；逐次测得的耗时也不保证相同。

本节依据源码分析和数学推导，没有运行新的 GPU 性能对比；例子中的25%属于 attention core 的理论主要运算量比较，不是实测性能结果。

### 7. 每次 prefill/decode forward 的 batch size 如何确定（2026-10-13）

**prefill 和 decode 的真实 batch size 都由 scheduler 每轮根据请求队列、请求状态、token 预算和 KV cache 容量动态确定。两者可以不同，也可以恰好相同；没有要求一次 prefill 的请求集合在后续每一步 decode 中保持不变。decode 的真实请求数确定后，框架再选择 cudagraph bucket，bucket 大小不能当作真实请求数。**

完整的一次生成请求通常包含一次 prefill（启用 chunked prefill 时可能包含多次）以及多次 decode forward。服务端会将不同请求合批，所以一个请求的 prefill batch 和后续各步 decode batch 可以具有不同的请求数和成员。

#### 7.1. 当前服务脚本的调度配置与阶段选择

`/app/scripts/dsr1/serve_dsr1fp4_a8w8_tp1.sh:54–55` 没有覆盖以下 CLI 默认值：

| 设置 | 当前命令的值 | 含义与源码依据 |
|---|---:|---|
| `--max-num-seqs` | 512 | sequence 数量上限；`/app/ATOM/atom/model_engine/arg_utils.py:482–487` |
| `--max-num-batched-tokens` | 16384 | 每次 forward 的新 query token 预算；同文件第385–390行 |
| `--scheduler-delay-factor` | 0.0 | 默认不通过该选项刻意延迟 prefill 以积攒请求；同文件第502–508行 |
| `enable_chunked_prefill` | `False` | 脚本显式传入 `--no-enable-chunked-prefill` |
| `enable_prefix_caching` | `False` | 脚本显式传入 `--no-enable_prefix_caching` |
| speculative decode | 未启用 | 当前命令没有 speculative config，普通 decode 每请求每步输入1个 query token |

当前 TP1/DP1 的 `Scheduler._schedule` 先尝试 prefill；若本轮选到 prefill 请求，会在 `scheduler.py:1638` 返回 `prefill_batch`，本轮不会再将 decode 请求混入这个 batch。只有本轮没有选到 prefill 请求时，才继续进入第1640行开始的 decode 调度。因此，不能理解为每个 scheduler tick 都先跑一次 prefill、再跑一次 decode。

#### 7.2. prefill batch size：根据等待请求的输入长度装入 token 预算

当前未开启 chunked prefill 和 prefix caching 的正常新请求路径，从 `waiting` 队列依次取请求，尝试将各请求完整的待计算 prompt 装入本轮 batch。主要条件是：

- 有已到达、可调度的请求；尚未到达的请求不会被预先计入 batch。
- 没有达到 `max_num_seqs`。当前实现用 `if len(self.running) >= self.max_num_seqs:` 判断是否已经满额；满足时将当前请求放回 `waiting` 队首并执行 `break`，停止接纳新请求。继续接纳所需的数量条件是 `len(self.running) < self.max_num_seqs`，同时还须满足 token 预算和 KV cache 等条件；等于上限时已经没有新名额。因此，已有活跃请求也会占用新请求的接纳名额，而不只是限制本轮 prefill 的请求数。
- 各请求待计算的新 token 数之和能装入 `max_num_batched_tokens`。
- KV cache 可以为请求分配所需 blocks；模型最大长度等静态限制也必须满足。

当前不开启 chunked prefill，如果下一请求的完整待计算长度超过本轮剩余 token 预算，会将它放回 `waiting` 队首并结束本轮的接纳循环；这条路径不会继续扫描后面的短请求来填满预算。因此，请求到达时间和队列顺序也会影响 batch size，实际 batch 可以小于预算允许的上界。

设本次 prefill 有 `B_prefill` 个请求，第 i 个请求本次输入 `L_i` 个 query token，则：

```text
T_prefill = Σ L_i
```

这不是总 context 长度，也不是整个 benchmark 的 token 总量。当前普通 prefill 使用该步实际调度的 `T_prefill` 行，不按 decode 的 capture sizes 将请求数补到 bucket。

在无部分 prefill、所有新请求实际待计算长度相同为 `L` 的简化情形，忽略其他资源约束时：

```text
B_prefill ≤ min(
    本轮已到达的待处理请求数,
    max_num_seqs - 接纳前已有的 running 请求数,
    floor(max_num_batched_tokens / L)
)
```

例如每个请求实际需要 forward 1024个 prompt token，当前 token 预算为16384，则单次 prefill 最多装入16个这样的请求；如果本轮只到达3个可处理请求，可以直接执行3个请求的 prefill，不要求等到16个。以后开启 chunked prefill 时，`L_i` 改为本轮调度的 chunk 长度；开启 prefix caching 时，还需要扣除可复用的 prefix，batch size 必须按新的调度规则重新判断。

源码依据：`scheduler.py:1240–1248` 的 waiting 接纳循环，第1283–1285行的 running 数限制，第1347–1358行的 KV 分配检查与新 token 数计算，第1381–1395行的预算处理，第2017–2029行的 `_prefill_chunk_for_budget`，以及第1616–1622行的 prefill batch 计数。

#### 7.3. decode batch size：从活跃、可继续生成的请求中选择

decode 从 `running` 队列选择已经完成 prefill、尚未结束且本轮可继续生成的请求。新请求完成 prefill 后会增加 decode 候选请求；请求达到输出长度限制、结束或被取消后会退出；KV cache 压力引起的 preemption 等情况也会使本轮可调度的请求数减少。因此，即使同一个请求一直在生成，它参与的 decode batch size 仍可能逐步变化。

默认普通 decode 中，每请求本轮调度1个 query token。在同一进程本地进行 MTP verification 时，若 `K=num_speculative_tokens`，默认每请求调度 `K+1` 个 query token（anchor + K 个 draft token）。在无额外阻塞、资源足够的简化情形：

```text
B_decode ≤ min(
    本轮可 decode 的活跃请求数,
    max_num_seqs,
    floor(max_num_batched_tokens / q_sched)
)

普通 decode：q_sched = 1
默认固定 K 的本地 MTP verification：q_sched = K + 1
```

这里 `q_sched` 表示 scheduler 计入预算的每请求 query token 数；默认固定 K 路径中就是本次 forward 的 `qlen`。DSpark 的后续 query-length 缩短属于额外路径，不应据此将 scheduler 的预算也当作已经缩短。

当前命令 `q_sched=1`、`max_num_seqs=512`、token 预算16384，单从这两个配置看，decode 请求数上限为512；实际还受活跃请求数和 KV cache 容量限制。`scheduler.py:1643–1657` 计算每请求 token 数并检查预算，第1658–1682行从 running 中选择、检查能否扩展 KV，第1745–1755行记录真实 decode batch 的请求数和 token 数。

#### 7.4. 真实 decode batch 与 graph bucket 是两个数量

在当前 DP1 的 FULL decode 中，先确定 `B_decode`，再选择最小的、能容纳它的已捕获 batch bucket：

```text
B_bucket = min { b ∈ capture_sizes | b ≥ B_decode }
T_decode_actual = B_decode × qlen       （默认等长路径）
T_decode_graph  = B_bucket × q_capture  （包含 dummy 行）
```

例如63个真实普通 decode 请求选择 `(64,1)` 图：真实请求数为63，图对应64个请求位置，模型/MoE 执行64行；框架最终只保留真实调度行的输出。`capture sizes` 是预捕获档位，不要求 scheduler 把真实请求数凑成这些值。源码见 `/app/ATOM/atom/utils/forward_context.py:248–253,286–311,370–371`。

#### 7.5. 当前 benchmark 的 concurrency 与 batch size

`/app/scripts/dsr1/bench_dsr1fp4.sh` 默认 `CONC=64`、`ISL=1024`、`OSL=1024`，并传入 `--max-concurrency=64`、`--num-prompts=256` 和 `--request-rate=inf`。这些参数表示客户端最多同时保持64个请求在途，总共提交256个请求；64个在途请求可能分别处于等待、prefill 或 decode。

因此，`concurrency=64` 不会固定服务端每次 prefill 或 decode 的真实 batch size 为64；`num-prompts=256` 也不是单次 forward 的 batch size。在仅有这批客户端请求、每请求实际待计算 prompt 恰为1024 token且资源足够的示例中，单次 prefill 最多16个请求，decode 可以在64个请求都完成 prefill且仍在生成时处理64个请求。实际运行还会受到到达时序、请求完成、资源和调度的影响，这个例子不是已测得的 batch 序列。

### 8. decode 的 FULL cudagraph 捕获范围（2026-10-13）

**当前脚本对应的 FULL decode 路径按 `(batch bucket, qlen)` 为整个 model forward 捕获一张图，范围覆盖 embedding、全部 transformer layers 和 final norm；当前 TP1、未开启 TBO 的配置还会将 LM head/logits 计算放入同一张图。每个 MoE 的 routing、GEMM1/GEMM2 和合并过程是这张整模型图内部的计算。**

对当前 TP1/PP1、非 speculative 路径，一张图覆盖的一次 forward 可表示为：

```text
embedding
→ transformer layer 0：norm / attention / residual / dense MLP 或 MoE
→ transformer layer 1：norm / attention / residual / dense MLP 或 MoE
→ ……
→ 最后一个 transformer layer
→ final norm
→ 输出 buffer 写入
→ LM head / logits（当前 TP1 且未开启 TBO）
```

当前 DSR1 的 `num_hidden_layers=61`；PP1 下，模型的 `start_layer` 到 `end_layer` 覆盖全部 transformer layers。因此，MoE 和单个 transformer layer 都包含在捕获范围内，实际图的边界由整个 `self.model(...)` 调用确定。

#### 8.1. capture 与 replay 的直接源码

`/app/ATOM/atom/model_engine/model_runner.py:3933–3949` 在同一个 `torch.cuda.graph` context 中调用整段模型，并保存到 `(bs,max_q_len)` key：

```python
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, self.graph_pool, stream=capture_ctx.stream):
    model_output = self.model(input_ids[:num_tokens], model_positions)
    outputs[:num_tokens] = model_output
    if self.logits_in_graph:
        graph_logits = self.model.compute_logits(outputs[:num_tokens])
self.graphs[(bs, max_q_len)] = graph
```

这是上述原始代码的等价摘录，省略了排版、pool 初始化和日志等外围代码。`self.model(...)` 通过 `/app/ATOM/atom/models/deepseek_v2.py:3288–3290` 调用内部模型，内部 forward 在第3142–3147行计算 embedding，在第3154–3160行循环执行所有本 rank 的 layers，并在第3167行执行 final norm。

`model_runner.py:3732–3734` 设置：

```python
is_tbo = self.config.enable_tbo and isinstance(self.model, UBatchWrapper)
self.logits_in_graph = self.world_size == 1 and not is_tbo
```

其中 `self.world_size=config.tensor_parallel_size`，当前 TP1、未启用 TBO，故 `logits_in_graph=True`。`DeepseekV2ForCausalLM.compute_logits` 在 `deepseek_v2.py:3305–3310` 调用 `self.lm_head(hidden_states)`。如果后续切换 TP/TBO 等配置，logits 是否包含在 graph 内应重新核对，不能把当前结论推广为所有部署都包含 LM head。

运行时 `model_runner.py:2942–2950` 根据 `(running_bs,max_seqlen_q)` 选择已有图并执行一次 `replay()`，再按 `scheduled_tokens` 截取有效输出。当前非 speculative decode 的 `qlen=1`，例如 `(64,1)` 图覆盖64行的整段 model forward；61个 layers 都在这次 replay 的范围内。

#### 8.2. 一张 graph 的含义与图外工作

一张 graph 内可以记录许多 kernel launch 和 GPU 操作。整模型 capture 不意味着将 embedding、attention、MoE 等融合成一个 GPU kernel；MoE 内部的多个 kernel 仍可作为图内节点被执行。同一图 replay 时，输入和 routing 数据可以更新，但不会重新执行 Python 的 MoE kernel dispatch。

请求调度、本轮 token ID 和 attention metadata 的准备在整模型 replay 前完成；sampling、停止条件判断和请求状态更新在图外完成。graph 只覆盖一次模型 forward，不覆盖一个请求从 prefill 到生成结束的完整过程，也不将所有 decode iterations 一次性录成一张图。

本节适用于当前 `cudagraph_mode=FULL` 的标准整模型 capture 分支。改为 PIECEWISE 或其他捕获模式时，图的边界会改变；当前 FULL prefill 仍普通执行，不能由 decode 的整模型 graph 推断 prefill 也 replay。

### 证据与版本记录

本节已记录主要结论和远端源码位置，完整调查材料保存在工作区：

- [调查记录](../../.codex_tmp/dsr1_cudagraph_investigation/INVESTIGATION.md)：完整推导与验证范围。
- [远端源码摘录](../../.codex_tmp/dsr1_cudagraph_investigation/SOURCE_EXCERPTS.md)：保留远端路径、原始行号、模型/CSV 快照和旧 capture 日志。
- [补充采集与 CPU 匹配结果](../../.codex_tmp/dsr1_cudagraph_investigation/supplement.json)：采集元数据、34组查询、环境及进程检查。
- [调度与 capture 范围源码摘录](../../.codex_tmp/dsr1_cudagraph_investigation/scheduler_capture_scope_excerpts.json)：第7、8节核对的 scheduler、CLI 默认值、整模型 capture/replay、模型 forward 和两个脚本的原始行号、源码及 SHA256；仅采集源码，没有启动服务或 GPU 测试。

两轮源码摘录的重叠位置已逐行比对，内容一致。调查时两个脚本的 SHA256 为：

```text
serve_dsr1fp4_a8w8_tp1.sh
f538c9fb2f3f0b8c6d39d0ae7274c3ccac901df0802df5724a323a03edadd9ae

bench_dsr1fp4.sh
db9a5cb595aa65e8c885a5f33117d53d2b92b32d3e7d8884a39dee949fef1edb
```
