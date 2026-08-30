# MI450/MI455X WG → HBM 带宽上限分析

## 1. 目的与适用范围

本文分析 MI450/MI455X（gfx1250/CDNA5）上，从 Workgroup/WGP 到 HBM4 的数据搬运路径，并计算各级理论带宽上限。

主要目标：

1. 把各级带宽写成“每周期字节数 × 对应时钟”的形式。
2. 区分容量、事务粒度、扇出能力和真正的持续吞吐。
3. 分别分析读路径与写路径。
4. 找出目前已知的瓶颈和仍需通过实硅测量确认的环节。

本文主要按 Full MI455X 配置计算：

- 8 XCD
- 2 Shader Engine/XCD
- 16 WGP/SE
- 256 WGP/GPU
- 12 GL1C/SE，共 192 GL1C
- 96 GL2C/FCD，共 2 FCD、192 个有效 GL2C channel
- 12 个 HBM4 stack
- 2048-bit/stack
- 432 GB HBM4

除非特别说明，本文的 `TB/s` 统一按 `1024^4 B/s` 计算，数学上等同于 `TiB/s`。为避免与厂商规格混淆，厂商按 `10^12 B/s` 给出的值明确写作 `decimal TB/s`。

定义十进制 TB/s 到本文 TB/s 的换算常数：

```text
K = 10^12 / 1024^4
  = 1 / 1.024^4
  ≈ 0.9094947018

本文 TB/s 数值 = decimal TB/s 数值 × K

1 B/cycle × 1 GHz
= 10^9 B/s
= 0.001 decimal TB/s
= 0.0009094947018 TB/s（1024^4 B/s）
```

下文显示值按相应小数位四舍五入，推导使用未舍入的 `K`。例如，白皮书的 `23.3 TB/s` 是厂商十进制标称，本文写作 `23.3 decimal TB/s`，等价于约 `21.191227 TB/s（1024^4 B/s）`；不会把 AMD 原始规格改写成二进制单位。

## 2. 时钟符号

本文使用以下符号：

| 符号 | 含义 | 当前机器示例 |
|---|---|---:|
| `fG` | per-XCD current GFXCLK/SYS，XCD Shader/GFX 时钟，GHz | current 2.308–2.322；DPM max 2.4 |
| `fL` | GL2CLK，GL2 cache 专用时钟，GHz | 未确认；公开 AMD-SMI/ROCm-SMI 不报告 |
| `fE` | EA 数据通路实际时钟，GHz | 未确认；公开 AMD-SMI/ROCm-SMI 不报告 |
| `fF` | FCLK/DF selected/current DPM level，GHz | 1.950；不是 workload-average effective clock |
| `fU,API` | AMD-SMI `current_uclk_aid`，per AID，GHz | AID0/AID1 均为 1.900 |
| `fU,int` | HBM controller 内部 post-gearing 数据通路时钟，GHz | 未实测；若按 `/2` 假设则约 0.950 |
| `fM` | MCLK，HBM PHY current/selected configuration，GHz | 1.900 |

公开 API 的 `current_uclk_aid` 与 `MCLK` 在当前机器上都报告 1.900 GHz。因此，不能再用同一个
`UCLK` 名称同时表示公开 API 字段和假设的 `/2` 后内部 controller clock。本文后续涉及 UMC
内部每周期系数时使用 `fU,int`；`fU,API` 只表示 AMD-SMI 原样报告的字段。

下面的 gearing 关系目前只是需要正式 clocking spec 或 silicon gear register 验证的定义/假设，
不是当前实机 telemetry 已确认的事实：

```text
fU,int ?= fM / 2
HBM per-pin data rate ?= 4 × fM = 8 × fU,int
```

代入当前 MCLK configuration 得到的待验证推导为：

```text
fM = 1.9 GHz
fU,int = 0.95 GHz  # derived under the unverified /2 assumption
HBM data rate = 7.6 Gb/s/pin  # derived under the unverified ×4 assumption
```

## 3. 数据路径

### 3.1 普通读

```text
HBM4
  → HBM PHY / UMC
  → DF / SDF
  → EA
  → GL2C / GL2A
  → GLARB
  → GL1C / GL1A
  → TCP / WGP$
  → TXD
  → VGPR / VALU
```

### 3.2 普通写

写路径方向相反：

```text
VGPR / LDS
  → TXD / TCP
  → GL1
  → GL2
  → EA / DF / UMC
  → HBM4
```

### 3.3 TDM/async 路径

对于 direct async copy：

```text
HBM4 → ... → GL1/TCP control → LDS
```

满足 direct-copy 条件时，数据可以直接进入 LDS，不在 vector cache/WGP$ 数据阵列中分配；条件不满足时可能先进入 VC，再复制到 LDS。

### 3.4 命名澄清

- WGP$ 不是 TCP 后面的一个额外 cache level。
- WGP$ 是 TX/TCP 子系统中统一 `LDS + vector-cache` SRAM 的 cache 部分。
- GL1 在该架构中主要承担 buffer、路由和仲裁功能，不是传统意义上的数据 cache。
- GL2 才是共享读写 cache。

## 4. 各级带宽汇总

| 环节 | 适用范围 | 读带宽 | 写带宽 | 时钟域 | 全 GPU 参数化上限 |
|---|---|---:|---:|---|---|
| WGP 本地物理 SRAM 总线 | per WGP | 最高 512 B/cycle | 最高 512 B/cycle | GFXCLK | `119.209290 × fG` TB/s |
| TCP/WGP ↔ GL1 数据接口 | 256 WGP | 256 B/cycle/WGP | 128 B/cycle/WGP | GFXCLK | 读 `59.604645 × fG`，写 `29.802322 × fG` TB/s |
| GL1C | 192 instances | 128 B/cycle/GL1C | 64 B/cycle/GL1C | GFXCLK | 读 `22.351742 × fG`，写 `11.175871 × fG` TB/s |
| GL2C cached path | 192 channels | 128 B/cycle/channel | 128 B/cycle/channel | GL2CLK | 每方向 `22.351742 × fL` TB/s |
| GL2C ↔ EA | 192 channels | 64 B/cycle/channel | 64 B/cycle/channel | EA clock | 每方向 `11.175871 × fE` TB/s |
| DF | 全 GPU | 系数未知 | 系数未知 | FCLK | `CDF,R × fF`、`CDF,W × fF` |
| UMC | 全 GPU | 系数未知 | 系数未知 | 内部 controller clock | `CUMC,R × fU,int`、`CUMC,W × fU,int` |
| HBM4 pins | 12×2048 bit | 3072 B/transfer | 3072 B/transfer | `4×MCLK` data rate | `11.175871 × fM` TB/s |

其中第一项是 WGP 内部物理端口极限，并不代表端到端可以持续达到该值。

## 5. WG/VALU/VGPR ↔ TX/TDM/LDS/WGP$

### 5.1 本地 SRAM 物理路径

文档给出每个 SIMD-pair 一条 2048-bit，即 256 B/cycle 的双向数据路径。每个 WGP 内有两个 SIMD-pair。

因此物理总线最大值为：

```text
per WGP = 2 × 256 = 512 B/cycle
```

全 GPU：

```text
256 WGP × 512 B/cycle
= 131072 B/cycle
= 119.209290 × fG TB/s
```

代入当前 `fG=2.314 GHz`：

```text
119.209290 × 2.314 = 275.850 TB/s
```

这是物理端口总宽度，不是普通 global load/store 的持续带宽。实际还受到以下因素限制：

- 指令类型与数据宽度
- LDS/VC/GL1BUF 端口共享
- SRAM bank conflict
- 64 KB segment conflict
- TCP/GL1 接口
- 请求、tag 和 credit

### 5.2 面向 GL1 的实际接口

文档给出的 per-WGP 接口上限：

```text
read return：256 B/cycle/WGP
write source：128 B/cycle/WGP
```

全 GPU：

```text
read = 256 WGP × 256 B/cycle
     = 65536 B/cycle
     = 59.604645 × fG TB/s

write = 256 WGP × 128 B/cycle
      = 32768 B/cycle
      = 29.802322 × fG TB/s
```

当前 `fG=2.314 GHz`：

```text
read = 59.604645 × 2.314 = 137.925 TB/s
write = 29.802322 × 2.314 = 68.963 TB/s
```

一次请求可以携带 256 B，并不代表读写方向都能持续达到 256 B/cycle。256 B write 在 128 B source bus 上至少需要两个传输周期。

## 6. TDM/async 局部上限

对于 B128 async 操作：

```text
16 threads/cycle × 16 B/thread = 256 B/cycle
```

TX 的 per-WGP 理论表同样给出：

- `global_*_async_*_b128_direct`：256 B/cycle/WGP
- `tensor_load_async_to_lds`：256 B/cycle/WGP

因此全 GPU 局部理想值：

```text
256 WGP × 256 B/cycle
= 65536 B/cycle
= 59.604645 × fG TB/s
```

当前 GFXCLK 下：

```text
59.604645 × 2.314 = 137.925 TB/s
```

但 `256 B/async_load` 首先是 payload 和处理粒度。只有同时满足以下条件，才可能接近 256 B/cycle：

- 连续发射
- 地址满足对齐要求
- 没有 bank/segment conflict
- tag 充足
- 下游 credit 充足
- 足够多的 outstanding requests
- GL1/GL2/EA/DF/HBM 不产生 backpressure

## 7. GL1 上限

每个 SE：

- 12 个 GL1C
- 16 个 WGP
- 每 GL1C read return 目标：128 B/cycle
- 每 GL1C write 目标：64 B/cycle

### 7.1 每 SE

```text
read = 12 × 128
     = 1536 B/cycle/SE
     = 1.396984 × fG TB/s

write = 12 × 64
      = 768 B/cycle/SE
      = 0.698492 × fG TB/s
```

### 7.2 平均每 WGP

```text
read = 1536 / 16 = 96 B/cycle/WGP
write = 768 / 16 = 48 B/cycle/WGP
```

因此“96 B/cycle/WGP”只适用于读返回方向；写方向平均值为 48 B/cycle/WGP。

### 7.3 每 XCD

每个 XCD 有两个 SE：

```text
read = 2 × 1536
     = 3072 B/cycle/XCD
     = 2.793968 × fG TB/s

write = 2 × 768
      = 1536 B/cycle/XCD
      = 1.396984 × fG TB/s
```

### 7.4 全 GPU

全 GPU 有 16 个 SE：

```text
read = 16 × 1536
     = 24576 B/cycle
     = 22.351742 × fG TB/s

write = 16 × 768
      = 12288 B/cycle
      = 11.175871 × fG TB/s
```

代入当前 GFXCLK 2.314 GHz：

```text
GL1 read  = 22.351742 × 2.314 = 51.722 TB/s
GL1 write = 11.175871 × 2.314 = 25.861 TB/s
```

## 8. GL1 读写是否能同时进行

GL1 的读返回和写 source data 使用不同方向的物理接口，所以可以出现以下重叠：

- 向 GL1 发送一个新 write
- 同时接收一个更早 read 的返回数据
- 不同 GL1C/channel 同时处理不同方向流量

但 51.722 TB/s 和 25.861 TB/s 是分别计算的单方向理论最大值，不能相加成 77.583 TB/s 的持续端到端带宽。

原因包括：

1. GL1 请求仲裁、request FIFO 和部分内部资源共享。
2. 一个 GL1 client 每周期最多接收一个 return，不能同时接收 data return 和 ack。
3. GL2 每个 bank 每周期只有一次 tag lookup。
4. GL2 可以在同周期生成一个 natural writeback 和一个 read miss，但进入两个 FIFO 后，`EA_IF` 一次只取一个。
5. EA、DF、UMC 和 HBM 继续共享下游资源。

因此 GL1 支持流水级别的双向重叠，但不保证 read peak 与 write peak 同时兑现。

## 9. GL2 cache 上限

每个 FCD/AID 侧：

- 96 个有效 GL2C channel
- 每 channel 1 MB
- read cached bandwidth：128 B/cycle
- write cached bandwidth：128 B/cycle
- 每 channel 每周期一次 tag lookup

### 9.1 每 FCD

```text
96 × 128 = 12288 B/cycle
```

因此每个方向：

```text
11.175871 × fL TB/s/FCD
```

### 9.2 全 GPU

两个 FCD：

```text
2 × 12288 = 24576 B/cycle
```

每方向：

```text
22.351742 × fL TB/s
```

白皮书给出的 L2 aggregate bandwidth 为约 `54 decimal TB/s`，等价于本文单位下约 `49.112714 TB/s`。若将其解释为上述单向 cached-path 上限，则反推：

```text
fL = (54 × K) / (24.576 × K)
   ≈ 49.112714 / 22.351742
   ≈ 2.197 GHz
```

这是根据白皮书数字做出的推导，不是文档直接公布的 GL2CLK。

## 10. GL2 ↔ EA 上限

每个 GL2C 对应一条 64 B/cycle 的 EA 接口。

### 10.1 每 FCD

```text
96 × 64 = 6144 B/cycle
```

每方向：

```text
5.587935 × fE TB/s
```

### 10.2 全 GPU

```text
192 × 64 = 12288 B/cycle
```

每方向：

```text
11.175871 × fE TB/s
```

该环节是目前已知 HBM miss 路径中系数最窄的内部接口之一。

为了让该接口达到白皮书的 `23.3 decimal TB/s`，即本文单位下约 `21.191227 TB/s`：

```text
11.175871 × fE ≥ 21.191227
fE ≥ 1.896 GHz
```

目前需要进一步确认实硅上 EA 到底运行在哪个时钟域、是否与 GL2CLK 同速，以及是否存在内部 gearing。

## 11. EA → DF → UMC

现有资料可以确认：

- EA 是 GL2 client 与 SDF/DF 之间的 bridge。
- SDP/EA 数据宽度为 512 bit，即 64 B。
- EA 边界已包含在上一节的 `11.175871 × fE` TB/s 中。

但现有资料没有完整、无冲突地给出：

- DF 聚合端口数量
- 每 FCLK 周期的数据 beat 数
- DF → UMC 的实际总线宽度
- UMC 聚合实例与每个内部 controller clock beat 的有效数据宽度
- `fU,API`、MCLK 与内部 controller datapath `fU,int` 的完整 gearing

因此保留为参数：

```text
B_DF_read  = CDF,R × fF
B_DF_write = CDF,W × fF

B_UMC_read  = CUMC,R × fU,int
B_UMC_write = CUMC,W × fU,int
```

这里的 `CDF,*` 和 `CUMC,*` 均定义为本文二进制带宽单位下的 TB/s/GHz 系数；若以后从十进制资料取得系数，也需要乘以 `K`。

不能从白皮书的 `23.3 decimal TB/s`（本文单位约 `21.191227 TB/s`）反推这些系数后，再把反推结果当作文档事实。

## 12. HBM4 原始带宽

### 12.1 总 pin 宽度

```text
12 stack × 2048 bit/stack
= 24576 bit
= 3072 B
```

MI450 的 MCLK 定义为：

```text
HBM per-pin data rate = 4 × MCLK
```

因此：

```text
B_HBM
= 3072 B × 4 × fM
= 12288 × fM × 10^9 B/s
= (12.288 / 1.024^4) × fM TB/s
≈ 11.175871 × fM TB/s
```

在 `fM=1.9 GHz` 时：

```text
B_HBM = 11.175870895 × 1.9
      = 21.234155 TB/s（1024^4 B/s）

同一原始速率 = 23.3472 decimal TB/s
```

这与白皮书四舍五入后的 `23.3 decimal TB/s` 相符；该官方标称等价于本文单位下约 `21.191227 TB/s`。

### 12.2 23.3472 decimal TB/s 是什么

`23.3472 decimal TB/s` 是全部 HBM DQ pins 的原始单向峰值，等价于本文单位下 `21.234155 TB/s`：

```text
pure read theoretical limit  ≈ 21.234155 TB/s
pure write theoretical limit ≈ 21.234155 TB/s
```

HBM DQ pins 是双向复用的，因此读写共享同一总预算：

```text
HBM_read + HBM_write ≤ 21.234155 TB/s
```

不能解释成：

```text
21.234155 TB/s read + 21.234155 TB/s write ≈ 42.468309 TB/s
```

即使全部为纯读或纯写，应用也通常不能长期达到精确的 `21.234155 TB/s`，因为仍有：

- DRAM refresh
- command/address 周期
- bank/row timing
- 地址分布不均
- HBM channel idle bubble
- cache/DF/UMC 仲裁
- 读写方向切换（混合流量）
- PPT/thermal throttling

实际带宽可写为：

```text
B_read_actual  = η_read  × 21.234155 TB/s
B_write_actual = η_write × 21.234155 TB/s

η_read < 1
η_write < 1
```

## 13. Copy/Triad 的带宽统计

### 13.1 Copy

处理 `N` 字节数据：

```text
HBM traffic = N B read + N B write = 2N B
```

因此：

- 如果 benchmark 只把复制的数据算一次，则理想上限约为：

```text
21.234155 / 2 ≈ 10.617077 TB/s
```

- 如果 benchmark 把 read+write 总流量都计入带宽，则显示值理论上可以接近 `21.234155 TB/s`。

### 13.2 Triad

若一次操作是两读一写：

```text
HBM traffic = 3N
```

若只按输出数据量 `N` 统计，则理想输出生成率：

```text
21.234155 / 3 ≈ 7.078052 TB/s
```

如果 benchmark 把两读一写的总流量 `3N` 都统计进去，则其显示带宽理论上仍以 `21.234155 TB/s` 为物理上限。

## 14. Multicast 的有效带宽

文档给出：

- ISA cluster mask 最多覆盖 16 WG。
- GL1 一次最多合并 5 个相同请求。

这两个数字都不是物理带宽倍数。

设实际持续 merge factor 为：

```text
1 ≤ m ≤ 5
```

则：

```text
B_effective ≤ m × B_physical
```

同时还受到本地 WGP/TCP 接口限制：

```text
B_effective,multicast
≤ min(5 × B_physical, 59.604645 × fG)
```

仅按 HBM 名牌值计算的绝对理想上限：

```text
5 × (23.3 decimal TB/s × K)
≈ 5 × 21.191227
≈ 105.956133 TB/s effective
```

这里的 `105.956133 TB/s` 是“向多个 WGP 交付的逻辑有效字节数”，并不代表 HBM pins 真正传输了 `105.956133 TB/s`。

实际 merge factor 还取决于：

- 多个 WG 是否请求相同地址
- requester/mask 是否匹配
- 请求到达时间差
- merge timeout
- 接收 WGP 的本地接口
- cache 命中与复用

## 15. 容量、事务粒度和吞吐的区别

### 15.1 WGP$+LDS 384 KB

这是容量，不是带宽：

```text
LDS ≤ 320 KB
VC ≥ 64 KB
```

容量决定可缓存和可驻留的数据量，但不能直接推出 B/cycle。

### 15.2 TCP line 128 B

128 B 是 cache line/transaction granularity。持续吞吐还需要：

- 每周期 issue 数
- outstanding tags
- round-trip latency
- cache miss rate
- credit
- channel/bank 分布

### 15.3 max 3584 lines

3584 lines 来自较早 TCP cache 方案或 tracking/metadata 描述。

较新的 TX MAS 与最终白皮书描述为：

- 384 KB unified SRAM
- VC 最大约 256 KB
- 约 2048 条 active 128 B cache line

因此 3584 不应直接解释为当前实硅上可驻留的有效 cache lines，更不能用于持续带宽计算。

### 15.4 128/256 B per tag

这是请求和 tracking 粒度，不是每周期吞吐。

要得到实际带宽，还需要：

- tag 数量
- tag 回收延迟
- 每周期可发请求数
- 请求对齐
- source/return bus 宽度
- 下游 credit

### 15.5 Cluster ≤16 WG

这是 ISA 调度与 mask 范围，不表示 16 倍 HBM 带宽。

## 16. 最终端到端上限公式

### 16.1 HBM → WG 读路径

```text
B_read ≤ min(
    59.604645 × fG,    TCP/WGP read return
    22.351742 × fG,    GL1 read
    22.351742 × fL,    GL2 cached read
    11.175871 × fE,    GL2 → EA
    CDF,R × fF,        Data Fabric
    CUMC,R × fU,int,   UMC
    11.175871 × fM     HBM pins
) TB/s
```

### 16.2 WG → HBM 写路径

```text
B_write ≤ min(
    29.802322 × fG,    TCP/WGP write source
    11.175871 × fG,    GL1 write
    22.351742 × fL,    GL2 cached write
    11.175871 × fE,    GL2 → EA
    CDF,W × fF,        Data Fabric
    CUMC,W × fU,int,   UMC
    11.175871 × fM     HBM pins
) TB/s
```

### 16.3 混合读写

不能简单地把 `B_read` 和 `B_write` 的单向峰值相加。

HBM 边界满足：

```text
B_read,HBM + B_write,HBM ≤ 11.175871 × fM
```

在当前 MCLK 下：

```text
B_read,HBM + B_write,HBM ≤ 21.234155 TB/s
```

实际混合带宽还会因为 read/write turnaround 而低于该值。

## 17. 达到 23.3 decimal TB/s 所需的最低频率

白皮书目标 `23.3 decimal TB/s` 等价于本文单位下约 `21.191227 TB/s`。目标和接口系数都乘以同一个 `K`，所以换算不会改变最低 GHz：

| 环节 | 条件 | 最低频率 |
|---|---|---:|
| GL1 read | `22.351742 × fG ≥ 21.191227` | `fG ≥ 0.948 GHz` |
| GL1 write | `11.175871 × fG ≥ 21.191227` | `fG ≥ 1.896 GHz` |
| GL2 cached | `22.351742 × fL ≥ 21.191227` | `fL ≥ 0.948 GHz` |
| GL2 ↔ EA | `11.175871 × fE ≥ 21.191227` | `fE ≥ 1.896 GHz` |
| HBM pins | `11.175871 × fM ≥ 21.191227` | `fM ≥ 1.896 GHz` |

当前机器的公开接口读数：

```text
fG      = 2.308–2.322 GHz per XCD current；2.400 GHz max
fF      = 1.950 GHz selected/current DPM level
fU,API  = 1.900 GHz per AID current_uclk_aid
fM      = 1.900 GHz current/selected MCLK configuration
```

已知部分：

```text
GL1 read  = 51.722 TB/s
GL1 write = 25.861 TB/s
HBM       = 21.234155 TB/s
```

因此从现有系数看，当前 GL1 不是第一个理论瓶颈。

仍需确认：

- GL2CLK `fL`
- EA clock `fE`
- `CDF,R/CDF,W`
- `CUMC,R/CUMC,W`
- 实际 throttle residency

## 18. 资料冲突

### 18.1 GL2 时钟域

较旧 GL2 MAS 的部分描述把 GL2 路径放在 GFX clock domain。

较新的 PMR/GDFLL 和 MI450 DPM 资料明确存在独立 GL2CLK、GL2CLK DFLL/MSFLL，并使用：

```text
GL2CLK_target = max(
    clamp(GFXCLK),
    clamp(FCLK)
)
```

因此本文保留独立 `fL`，不直接令 `fL=fG`。

### 18.2 EA 时钟

旧 EA MAS 描述单时钟、最高约 1.5 GHz。

如果直接代入：

```text
11.175871 × 1.5 ≈ 16.763806 TB/s
```

这低于产品的 `23.3 decimal TB/s` HBM 峰值（本文单位约 `21.191227 TB/s`），存在明显冲突。

可能原因包括：

- EA 后续版本提升频率
- EA 实际跟随更高的 GL2CLK
- 内部存在 gearing
- 旧 MAS 已过时

在获得当前 silicon clock register、STA/clocking spec 或计数器实测前，不能确定。

### 18.3 TCP 3584 lines

早期 TCP cache 方案的 3584 lines 与后期 384 KB unified SRAM、2048 active line 描述冲突。

本文将 3584 视为旧方案或 tracking/metadata 上限，不用于最终 resident cache 容量和带宽计算。

## 19. 实机测量建议

### 19.1 查看 DPM 表

```bash
amd-smi static --clock --json
```

当前示例：

```text
SYS/GFXCLK = 2315 MHz selected current level，2400 MHz max
MEM/MCLK   = 1900 MHz selected/current configuration
DF/FCLK    = 1950 MHz selected current level
SOC        = 1350 MHz selected/current MID SoC clock
```

`static --clock` 给出 DPM frequency levels 和当前选中的 level，不给 workload-average effective
clock。这里的 `SOC` 是 MID SoC clock，不是 GL2CLK、FCLK 或 EA clock。

### 19.2 以一秒间隔采样 current clock

```bash
watch -n 1 'amd-smi metric --clock --json'
```

`watch` 只按一秒间隔重复查询，不计算一秒平均值。当前 AMD-SMI 实现中：

- `clk` 来自 `gpu_metrics` 的 `current_*` 字段；GFXCLK 可按 XCD 报告。
- `min_clk`/`max_clk` 来自 sysfs 的每 clock type 范围，不是采样区间内的最小值/最大值。
- 当前机器的 `average_gfxclk_frequency` 和 `average_uclk_frequency` 都是 `N/A`。
- FCLK 的 `clk` 由 CLI 回退到 DF DPM 当前 level，并非 workload-average effective FCLK。
- `current_uclk_aid` 在 AID0/AID1 均为 1900 MHz，与 MCLK 同值；这不能证明内部
  post-gearing controller clock 也是 1900 MHz，也不能验证 `fU,int=fM/2`。

### 19.3 使用 AGT PMLog

```bash
sudo -n /opt/amd-apps/agt_internal/agt_internal \
  -i=0 \
  -unilog=PM \
  -unilogallgroups \
  -unilogstopcheck \
  -unilognoesckey \
  -unilogperiod=50 \
  -unilogcount=120 \
  -unilogoutput=/path/to/mi450_pm.csv
```

AGT 会创建输出文件；`-unilogcount=120` 在 50 ms period 下采集约 6 秒并由工具自行 flush/退出。
长时间采集则需要用工具支持的 stop-check 或受控 signal 终止，以确保 CSV flush；
采集结束后还必须确认没有遗留 AGT 进程。运行目标 workload 后应按 CSV 的真实 header 区分：

```text
GFXCLK Target Freq / Pre Deep Sleep Freq / Post Deep Sleep Freq per XCD
FCLK Target Freq / pre-deepsleep Freq / post-deepsleep Freq per AID
GL2CLK Target Freq / pre-deepsleep Freq / post-deepsleep Freq per AID
MCLK target/effective fields（按 header 原名）
SocketPowerLimit
PptResidencyAcc
ProchotResidencyAcc
SocketThmResidencyAcc
VrThmResidencyAcc
HbmThmResidencyAcc
```

#### 当前实机 idle 短采集示例

2026-08-29 11:20:53–11:20:59 UTC 在 `heliosr-2b805-b8-2` 上使用 AGT
`4.1.147.0`、SMC FW `00.125.43.00`，以 50 ms 周期采集 120 个 sample；采集前后均无
GPU process，GFX/UMC activity 均为 0%。这是 idle 状态示例，不能推广为 workload 时钟：

- XCD0–7 `Target Freq` 为 2329.88–2329.96 MHz；各 XCD `Pre Deep Sleep Freq` 与
  `Post Deep Sleep Freq` 在本次采集中相同，总范围为 2307.47–2323.89 MHz。
- AID0/AID1 FCLK target 均为 1950 MHz；pre/post 分别为 1950–1952 MHz 和
  1955–1957 MHz。
- AID0/AID1 GL2CLK target 均为 2330 MHz；pre/post 分别为 2301–2302 MHz 和
  2310–2312 MHz。
- AID0/AID1 的 `MCLK_a`/`MCLK_b` target 和 `MCLK_a Eff`/`MCLK_b Eff` 均稳定为
  1900 MHz。
- 同期 AMD-SMI current GFXCLK 为 2308–2323 MHz，DF/FCLK selected level 为
  1950 MHz，`current_uclk_aid` 为 1900 MHz。

AGT 对 GFXCLK/FCLK/GL2CLK 的真实字段名称是 `Target Freq`、`Pre Deep Sleep Freq` 和
`Post Deep Sleep Freq`，不能把 `Target Freq` 当作 effective clock；本次 CSV 只有 MCLK
字段明确使用 `Effective Frequencies ... Eff` 命名，也没有 UCLK 字段。

### 19.4 需要的计数器

为了确认每级 bytes/cycle，需要同时采集：

- GL1 request/return bytes
- GL2 read/write hit/miss bytes
- GL2→EA read/write transactions
- DF CS read/write bandwidth
- UMC/HBM read/write bytes
- HBM data-bus utilization
- outstanding request 数量
- tag/credit stall
- bank conflict、refresh 和 read/write turnaround

对于某一级：

```text
observed bytes/cycle
= measured bytes/s ÷ measured clock frequency
```

然后与本文的理论系数比较，可以判断真正开始掉带宽的位置。

## 20. Memory-bound kernel 优化优先级

1. 确认所有 12 个 HBM stack/channel 地址分布均匀。
2. 让 MCLK 保持 POR 1900 MHz。
3. 确认 FCLK、GL2CLK 在 workload 下处于足够高的档位。
4. 确认没有 PPT、PROCHOT、XCD/AID/HBM thermal throttle。
5. 增加并稳定 outstanding memory requests，覆盖 HBM latency。
6. 使用大块、对齐、连续或经过验证的 channel-hash 友好地址模式。
7. 对纯写优先检查 non-temporal/uncached policy，避免不必要的 write allocate/read-for-ownership。
8. 对 async/TDM 检查 direct-copy 条件、LDS tag、VC 容量和 64 KB segment 冲突。
9. 对 multicast 分别统计物理 HBM bytes 和向多个 WG 交付的 effective bytes。
10. 不要只看 kernel 声明的数据量，应使用实际 HBM read/write traffic 计算效率。

## 21. 主要参考资料

本地资料：

- `C:\Users\yanguahe\Documents\code\wk_sp1\mi400_hw_wiki\raw\papers\mi400_hd_txt\architecture\subsystem\SH\MI400_Shader_Programming#65.txt`
  - memory hierarchy：约 1445–1469
  - TX/TDM local bandwidth：约 15883–15917、16818–16824
- `C:\Users\yanguahe\Documents\code\wk_sp1\mi400_hw_wiki\raw\papers\mi400_hd_txt\block\tx\mi400_tx_MAS#156.txt`
  - WGP/TCP source/return bus
  - TDM direct async
  - request/tag/VC/LDS 限制
- `C:\Users\yanguahe\Documents\code\wk_sp1\mi400_hw_wiki\raw\papers\mi400_hd_txt\subsystem\CMM\GLX\Design\MI400_GL1_CH_GLARB_Fabric_MAS#12.txt`
  - GL1C 数量、request/return bus、仲裁
- `C:\Users\yanguahe\Documents\code\wk_sp1\mi400_hw_wiki\raw\papers\mi400_hd_txt\subsystem\CMM\GL2\Design\MI400_GL2_MAS#8.txt`
  - GL2 128 B/cycle cached path
  - GL2→EA 64 B request
  - tag lookup 与 EA FIFO
- `C:\Users\yanguahe\Documents\code\wk_sp1\mi400_hw_wiki\raw\papers\mi400_hd_txt\subsystem\CMM\EA\Design\MI400_EA_MAS#47.txt`
- `C:\Users\yanguahe\Documents\code\wk_sp1\mi400_hw_wiki\raw\papers\mi400_hd_txt\MI450\amd-instinct-cdna5-instruction-set-architecture.txt`
- `C:\Users\yanguahe\Documents\code\wk_sp1\mi400_hw_wiki\raw\papers\mi400_hd_txt\MI450\amd-cdna5-whitepaper.txt`
  - HBM4：约 225–240
  - 产品汇总：约 624–631

Confluence：

- [MI450 UCLK DPM](https://amd.atlassian.net/wiki/spaces/DCSMU/pages/1142333395/MI450+UCLK+DPM)
- [MI450 GL2CLK DPM](https://amd.atlassian.net/wiki/spaces/DCSMU/pages/1311375743/MI450+GL2CLK+DPM)
- [MI455X/MI450X FCLK GL2CLK GFXCLK DPM](https://amd.atlassian.net/wiki/spaces/DPA/pages/1691702120/MI455X_MI450X+FCLK+GL2CLK+GFXCLK+DPM)
- [MI45X HBM4 Bandwidth](https://amd.atlassian.net/wiki/spaces/DCGPUVAL/pages/1635193408/MI45X+HBM4+Bandwidth)
- [MI45X Memory BW Sanity Check](https://amd.atlassian.net/wiki/spaces/DCP/pages/1449742572/MI45X+Memory+BW+Sanity+Check+Tools+Setup+Guide)

