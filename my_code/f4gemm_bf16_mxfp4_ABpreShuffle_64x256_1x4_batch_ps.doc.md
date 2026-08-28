# gfx1250 MXFP4xMXFP4 64x256_1x4 Persistent GEMM Design

<!-- markdownlint-disable MD013 MD033 MD060 -->

<a id="toc"></a>

## Table of Contents

- [1. Wave Tile and TDM Specialization](#section-1-wave-tile)
  - [1.1 Four Waves in a 1x4 N Strip](#section-1-1-wave-output-strip)
  - [1.2 Four-Slot LDS Ring](#section-1-2-lds-ring)
  - [1.3 Final Kernel Contract](#section-1-3-final-kernel-contract)
- [2. Geometry and Batch-Z Persistent Scheduling](#section-2-cluster-grid)
  - [2.1 Wave Strip versus Workgroup Cluster](#section-2-1-wave-vs-cluster)
  - [2.2 Logical X/Y Space and Shape Contract](#section-2-2-logical-grid)
  - [2.3 Physical Persistent X/Y Mapping](#section-2-3-persistent-launch)
  - [2.4 Batch-Z Planes and Base Pointers](#section-2-4-batch-z-pointers)
  - [2.5 Synchronization, Isolation, and Batch=1](#section-2-5-isolation-compatibility)
  - [2.6 Launch Examples](#section-2-6-launch-examples)
- [3. End-to-End Software Pipeline](#section-3-software-pipeline)
  - [3.1 wave0/2: `B-current -> A-current -> A-next -> B-next`](#section-3-1-wave02-flow)
  - [3.2 wave1/3: `A-current -> B-current -> B-next -> A-next`](#section-3-2-wave13-flow)
- [4. P0 Detailed Pipeline and Wave-Local Tile Coverage](#section-4-p0-details)
  - [4.1 wave0/2 Specialization Schedule](#section-4-1-wave02-details)
    - [4.1.1 wave0/2 Wave-Local Tile Coverage](#section-4-1-1-wave02-host-tile)
  - [4.2 wave1/3 Specialization Schedule](#section-4-2-wave13-details)
    - [4.2.1 wave1/3 Wave-Local Tile Coverage](#section-4-2-1-wave13-host-tile)
- [5. Epilogue Design and Final Store Contract](#section-5-epilogue-design)
  - [5.1 Output Geometry and Fragment Mapping](#section-5-1-output-geometry)
  - [5.2 Candidate Physical VGPR Layout](#section-5-2-vgpr-layout)
  - [5.3 BF16 Conversion and LDS Staging](#section-5-3-bf16-lds-staging)
  - [5.4 Required TDM-Store Contract and Synchronization](#section-5-4-tdm-store-sync)
  - [5.5 Final Drain, P3 Wrap, and Persistent-Task Transition](#section-5-5-final-drain-transition)
  - [5.6 Resource and Validation Checklist](#section-5-6-resource-validation)
- [6. Cluster TDM Multicast](#section-6-cluster-tdm-multicast)
  - [6.1 WG Bit Matrix](#section-6-1-wg-bit-matrix)
  - [6.2 Operand Multicast Masks](#section-6-2-operand-multicast-masks)
  - [6.3 Payload and Cluster Coverage](#section-6-3-payload-cluster-coverage)

<a id="section-1-wave-tile"></a>

## 1. Wave Tile and TDM Specialization

<a id="section-1-1-wave-output-strip"></a>

### 1.1 Four Waves in a 1x4 N Strip

Each wave retains the same `64x64` compute tile. The four waves are arranged
as one host-M wave by four host-N waves, so the WG output is exactly
`64x256`. All four waves have local M origin zero; logical wave `w` has local
N origin `64*w`.

| Wave | WG-relative M | WG-relative N | N-origin formula | TDM specialization |
|---:|---|---|---:|---|
| 0 | `[0,63]` | `[0,63]` | `64*0 = 0` | host A data |
| 1 | `[0,63]` | `[64,127]` | `64*1 = 64` | host B data |
| 2 | `[0,63]` | `[128,191]` | `64*2 = 128` | SA |
| 3 | `[0,63]` | `[192,255]` | `64*3 = 192` | SB |

For logical WG tile indices `(Mtile,Ntile)` and logical wave ID `w`:

```text
Mbase(w) = 64 * Mtile
Nbase(w) = 256 * Ntile + 64 * w
```

Thus wave ID contributes only to N. There is no wave-ID bit that selects a
high-M half in this design.

Each wave still owns eight `32x16` hardware output fragments and executes
16 WMMA instructions per K256 body: two K128 accumulations for each of eight
fragments. The wave0/2 and wave1/3 schedules in Chapters 3 and 4 are retained
only as alternative software schedules associated with TDM specialization.
They do not denote low-M/high-M wave pairs. Every wave covers the same local
M64 and a distinct N64 quarter.

<a id="section-1-2-lds-ring"></a>

### 1.2 Four-Slot LDS Ring

For one WG and one K256 body, the logical TDM payloads are:

| Operand | Payload derivation | Per-slot payload |
|---|---|---:|
| A data | `M64 * K256 * 4 bits` | `0x2000 = 8192 B = 8 KiB` |
| SA | `M64 * (K256 / K32) * 1 B` | `0x0200 = 512 B` |
| B data | `N256 * K256 * 4 bits` | `0x8000 = 32768 B = 32 KiB` |
| SB | `N256 * (K256 / K32) * 1 B` | `0x0800 = 2048 B = 2 KiB` |

The following correctness-first candidate packs the four input arrays
contiguously, keeps a `0x8000` stride between B slots, and reserves a separate
output region. It inserts no explicit layout gap:

| Operand | slot0 | slot1 | slot2 | slot3 | Array end |
|---|---:|---:|---:|---:|---:|
| A data | `0x00000` | `0x02000` | `0x04000` | `0x06000` | `0x08000` |
| SA | `0x08000` | `0x08200` | `0x08400` | `0x08600` | `0x08800` |
| SB | `0x08800` | `0x09000` | `0x09800` | `0x0A000` | `0x0A800` |
| B data | `0x0A800` | `0x12800` | `0x1A800` | `0x22800` | `0x2A800` |

The resulting fixed LDS allocation is:

```text
input ring [0x00000,0x2A800)     = 0x2A800 = 170 KiB
output staging [0x2A800,0x32800) = 0x08000 =  32 KiB
candidate fixed LDS end           = 0x32800 = 202 KiB
```

CDNA5 permits up to 320 KiB of LDS for a WG (CDNA5 ISA Section 2.2, local
text L743-L748), so 202 KiB is within the documented architectural capacity.
It can still reduce residency, and the target object must request the full
`0x32800` group segment.

The semantic TDM-load row counts follow the non-K dimensions. In byte mode,
one packed-data row is `K256/2 = 0x80` bytes and one scale row is
`K256/K32 = 0x08` bytes. This gives the following unassembled descriptor
candidate:

| Operand | Logical rows | Candidate row width | Candidate semantic tile | Payload |
|---|---:|---:|---:|---:|
| A data | 64 | `0x80` B | `tile_dim0=0x80, tile_dim1=64` | `0x2000` |
| SA | 64 | `0x08` B | `tile_dim0=0x08, tile_dim1=64` | `0x0200` |
| B data | 256 | `0x80` B | `tile_dim0=0x80, tile_dim1=256` | `0x8000` |
| SB | 256 | `0x08` B | `tile_dim0=0x08, tile_dim1=256` | `0x0800` |

CDNA5 defines tile dimensions in `data_size` units (Section 10.11.2 and
Section 10.11.4, local text L10215-L10231 and L10487-L10514). The table fixes
the target's logical row counts and payloads, but it does not claim encoded
descriptor words. Whether the target can expose the semantic rows directly
or must regroup the same bytes depends on the AB-preshuffle layout and
strides. Descriptor packing, legal dimensions, B-base/TDM destination
alignment, and bank behavior are target-assembly and hardware validation
boundaries.

For ring slot `s`, every wave uses the same A and SA base because there is no
wave-M offset. Each wave consumes its own local N64 quarter of the WG-wide B
and SB payload:

```text
A_wave_base(s,w)  = A_slot_base[s]
SA_wave_base(s,w) = SA_slot_base[s]
B_wave_base(s,w)  = B_slot_base[s]  + w * 0x2000
SB_wave_base(s,w) = SB_slot_base[s] + w * 0x0200
```

The offsets are respectively one N64 FP4 payload
(`64*256/2 = 0x2000`) and one N64 scale payload
(`64*(256/32) = 0x200`). A/SA have no wave-M high-half offset. No formula in
this target may use `(wave_id & 1)` for A/SA or `(wave_id >> 1)` for B/SB.
The base arithmetic is exact; alignment, DS access legality, and bank-conflict
behavior must be measured on the eventual target ISA.

<a id="section-1-3-final-kernel-contract"></a>

### 1.3 Final Kernel Contract

The following decisions are hard preconditions for this documentation design.
No corresponding assembled target ISA exists yet.

| Property | Final decision |
| --- | --- |
| Kernel symbol basename | `f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps` |
| Wave/WG geometry | Wave `64x64`; wave grid `(host M, host N) = (1,4)`; WG `64x256`. |
| Cluster geometry | `(cluster_y/M, cluster_x/N) = (1,4)`; four WGs per cluster. |
| M precondition | `M % (WG_M * cluster_M) = M % (64*1) = M % 64 == 0`. |
| N precondition | `N % (WG_N * cluster_N) = N % (256*4) = N % 1024 == 0`. |
| K precondition | `K % 256 == 0`; full K256 bodies only. |
| Boundary/tail policy | No M/N boundary tile, K tail, or partial cluster. |
| Epilogue store | One 64-row by 256-BF16 `tensor_store_from_lds` per WG, issued by wave0. |
| Candidate fixed LDS | Input ring `[0x00000,0x2A800)` plus output `[0x2A800,0x32800)`, ending at `0x32800 = 202 KiB`; no explicit layout gap. |

Accepted inputs therefore satisfy:

```text
M % 64   == 0
N % 1024 == 0
K % 256  == 0
```

The logical WG grid `(N/256, M/64)` must divide exactly into clusters of four
WGs along N and one WG along M. Chapter 5 specifies the 64-row output store.
Its width, descriptor encoding, and LDS layout remain implementation
validation work; a historical 256x256 reference ISA provides migration
evidence only for the 64-row-height idiom.

<a id="section-2-cluster-grid"></a>

## 2. Geometry and Batch-Z Persistent Scheduling

This chapter describes the implemented `_batch_ps` ISA and
`gemm_batch_isa_runner.py`, rather than a launch candidate. The batch kernel
keeps the original non-batch kernel's logical X/Y cluster-task traversal and
adds an independent physical grid-Z plane for each matrix. Its only
batch-addressing change is a one-time five-pointer adjustment before the
original descriptor, pipeline, epilogue, and persistent-restart path.

<a id="section-2-1-wave-vs-cluster"></a>

### 2.1 Wave Strip versus Workgroup Cluster

The `1x4` in the kernel name is the output-tile distribution of four waves
*inside one workgroup*. It is not the workgroup-cluster geometry. The two
levels happen to both have a four-wide N dimension, but they are distinct
levels of the hardware hierarchy:

| Level | Implemented distribution or dimensions | Output covered |
| --- | --- | --- |
| One wave | one wave32, one `64x64` output tile | `M64xN64` |
| One WG | four waves arranged `(M,N)=(1,4)`; HIP block `(128,1,1)` | `M64xN256` |
| One workgroup cluster | HIP `clusterDim=(x=N,y=M,z)=(4,1,1)`; four WGs | `M64xN1024` in one Z plane |

Thus one cluster contains four WGs and 16 waves. Within each WG, wave `w`
owns WG-relative N `[64*w,64*w+63]`. Across the cluster, local WG `wg_x`
owns a different `N256` tile. The complete output origin is:

```text
Mbase = 64 * Mtile
Nbase = 256 * Ntile + 64 * wave_id
```

**Documented hardware facts.** CDNA5 defines the hierarchy as
`Grid -> Cluster -> Workgroup -> Wave -> Work-item`; a cluster's WGs are
scheduled on one shader engine and may use cluster barriers and multicast
loads (CDNA5 ISA Sections 2.3-2.3.1, manual pages 10-11, local text
L751-L815). The MI400 Shader Programming Guide independently gives the same
hierarchy and TTMP6/TTMP7/TTMP9 cluster-state layout (Sections 2.3-2.3.1,
manual pages 24-25, local text L1699-L1766). The extended compute AQL packet
also distinguishes a 3-D grid count in clusters from a 3-D cluster size in
WGs (`CLUSTER_COUNT_*` versus `CLUSTER_SIZE_*`; packet reference page 12,
local text L599-L649). The workgroup-cluster scheduling corpus likewise
states that a grid may be 3-D in clusters, a cluster may be 3-D in WGs, and
partial clusters are not supported (Workgroup Clusters Scheduling Section 4,
page 4, local text L277-L304).

**Implementation deduction.** The runner and ISA assign cluster X to N and
cluster Y to M. Therefore the implemented cluster footprint is
`(1*64)x(4*256) = M64xN1024`. Cluster Z is one WG deep and has no output-tile
extent; software uses the cluster-grid Z coordinate as a batch selector.

<a id="section-2-2-logical-grid"></a>

### 2.2 Logical X/Y Space and Shape Contract

Let `W_N,W_M` be the original logical WG counts and `C_N,C_M` the logical
cluster-task counts in X/Y:

```text
W_N = N / 256
W_M = M / 64

C_N = W_N / clusterDim.x = N / (256*4) = N / 1024
C_M = W_M / clusterDim.y = M / (64*1)  = M / 64

T_XY = C_N * C_M                       # cluster tasks in one scheduler plane
```

The non-batch kernel's logical WG grid is `(W_N,W_M,1)` and its logical
cluster grid is `(C_N,C_M,1)`. The batch runner reports
`(W_N,W_M,batch)` and `(C_N,C_M,batch)`, but this appended logical Z count
does not change `T_XY`: each Z plane has its own `T_XY`-task counter. N is
the fastest coordinate in the flattened cluster-task space.

The runner enforces the following exact launch domain:

| Quantity | Implemented constraint | Consequence |
| --- | --- | --- |
| `M` | positive uint32; `M % 64 == 0` | full `M64` WG tiles; `clusterDim.y=1` adds no stronger divisor |
| `N` | positive uint32; `N % 1024 == 0` | full `N256` WG tiles and full groups of four N WGs |
| `K` | positive uint32; `K % 256 == 0` | full K256 bodies; this also satisfies the K32 scale and K128 scale-shuffle gates |
| `batch` | integer `1 <= batch <= 65535` | obeys the runner's HIP grid-Z limit; the largest launched batch ID is 65534 |
| M/N/K tails | none | no boundary WG, partial cluster, or K tail is constructed |

No power-of-two constraint is placed on `C_N`: the ISA uses a shift/mask
path when `C_N` is a power of two and an integer-division path otherwise.
For small non-power-of-two `T_XY`, the physical cluster-grid X count is also
non-power-of-two while the independently encoded recurrence stride remains a
power of two, as described below.

The ABI adds two non-divisibility checks. The five inherited row strides
`N*2`, `K/2`, `K/2`, `K/32`, and `K/32` must fit uint32. Each appended
batch stride is uint64, positive, and is conservatively required by the
runner to satisfy `batch_stride * batch <= 2^64-1`.

There is one inherited validation gap. The ISA stores `T_XY`, persistent
task IDs, and intra-matrix tile-origin byte arithmetic in 32-bit scalar
registers, while the runner does not explicitly prove those derived values
cannot wrap. Since the encoded recurrence stride `S <= 64`, a sufficient
scheduler bound is `T_XY <= 2^32-64`, so the final `task+S` lookahead also
fits. Every
intra-matrix A/B/sA/sB/D tile-origin offset must likewise fit uint32; a
simple conservative condition is that each single-matrix byte extent is at
most `2^32`. These are implementation safety limits, not additional
divisibility rules, and the requested `64x65536x32768` example satisfies
them. The new 64-bit batch offset does not make the inherited intra-matrix
indexing 64-bit.

<a id="section-2-3-persistent-launch"></a>

### 2.3 Physical Persistent X/Y Mapping

The batch-symbol runner chooses the physical seed count `C` and independently
encodes the recurrence stride `S` from the logical cluster-task count:

```text
T_XY = (M/64) * (N/1024), with T_XY >= 1

if T_XY <= 64:
    C = T_XY
    S = next_power_of_two(T_XY)
    cluster_grid_x/y      = (T_XY,1)
    physical cluster grid = (T_XY,1,batch)
    log2_grid_x/y         = (log2(S),0)
else:
    C = 64
    S = 64
    cluster_grid_x/y      = (16,4)
    physical cluster grid = (16,4,batch)
    log2_grid_x/y         = (4,2)

physical HIP WG grid = (cluster_grid_x*4,cluster_grid_y,batch)
clusterDim            = (4,1,1)
block                 = (128,1,1)
encoded recurrence S = 1 << (log2_grid_x+log2_grid_y)
```

**Implementation deduction.** Physical launch size and encoded recurrence
stride are deliberately different for non-power-of-two `T_XY <= 64`.
Exactly `T_XY` one-row seed clusters launch, but the existing log2 ABI encodes
the next power of two `S`. For `T_XY > 64`, the proven `(16,4)` topology,
64 seeds, and stride 64 remain unchanged. Each physical cluster contains four
WGs, so a plane launches `4*C` WGs.

**Launch-interface inspection.** The CDNA5 ISA and MI400 guide document
cluster-grid X as the independent 32-bit `TTMP9` coordinate and cluster-grid
Y/Z in `TTMP7`; they do not define the kernel's `log2_grid_x/y` software ABI
fields. In the runner, HIP `gridDimX/Y/Z` and `clusterDim` are populated from
the physical `LaunchGeometry.grid` and `.cluster`, while the two log2 values
are packed separately into kernargs. Neither `LaunchGeometry` nor the HIP
configuration builder requires the physical cluster counts to equal
`1 << (log2_grid_x+log2_grid_y)`.

For a physical cluster, gfx1250 provides cluster-grid X in `TTMP9`,
cluster-grid Y in `TTMP7[15:0]`, local cluster-WG X/Y in TTMP6, and the
cluster dimensions-minus-one in TTMP6. Decoding the implemented
`clusterDim=(4,1,1)` gives:

```text
cx   = TTMP9
cy   = TTMP7 & 0xffff
wg_x =  TTMP6       & 0xf              # 0..3
wg_y = (TTMP6 >> 4) & 0xf              # always 0

p = cx + (cy << log2_grid_x)
                                            # initial X/Y cluster-task ID
```

Equivalently, for a physical HIP WG coordinate `(gx,gy,gz)`,
`cx=gx/4`, `wg_x=gx%4`, `cy=gy`, and `wg_y=0`. All four WGs in the
cluster have the same `p` and advance in lockstep; `wg_x` selects one of
the four logical N tiles within the cluster task. The implemented traversal
is:

```text
C_N = N / 1024
C_M = M / 64
T_XY = C_N * C_M

for task = p; task < T_XY; task += S:
    cluster_n = task % C_N              # N is fastest
    cluster_m = task / C_N

    Ntile = cluster_n*4 + wg_x
    Mtile = cluster_m*1 + wg_y          # wg_y is zero

    wave_M_origin = Mtile*64
    wave_N_origin = Ntile*256 + wave_id*64
```

For `T_XY <= 64`, physical Y is one, so `cy=0` and the initial equation
reduces to `p=cx` regardless of `log2_grid_x`. The exact X extent launches
`cx=0..T_XY-1`, hence every seed is useful. Since `S >= T_XY`, every
`p+S >= T_XY`; all clusters terminate after one task. This is why a
non-power-of-two physical X extent is safe even though the recurrence remains
power-of-two. It would not provide complete coverage if a second round were
required.

For `T_XY > 64`, physical X/Y is `(16,4)` and encoded log2 X/Y is `(4,2)`,
so `p=cx+16*cy` covers `[0,64)` exactly. Every nonnegative task has one
unique decomposition `task=p+k*64` with `0 <= p < 64`; the 64 persistent
progressions therefore cover every logical task without duplicates or
misses.

The source implements one-task lookahead. At entry, a cluster terminates
immediately if `p >= T_XY`. Otherwise it maps `p`, computes and maps
`p+S` when valid, and records whether the current task is the last one.
After the current epilogue, all four WGs converge at the cluster barrier.
Each wave then rewinds its output pointer by the current task's exact byte
origin:

```text
D_delta(Mtile,Ntile,wave_id)
    = ((Mtile*64)*N + Ntile*256 + wave_id*64) * sizeof(BF16)

D_wave_ptr -= D_delta(current)
```

If the lookahead is out of range, the kernel terminates. Otherwise the
lookahead coordinates become current, A/B/sA/sB descriptors are rebuilt
from their plane-adjusted bases, `D_delta(next)` is added, and another
`+S` lookahead is formed. Thus physical cluster `p` visits exactly
`p,p+S,p+2S,... < T_XY`; no task is taken from another physical cluster.
The ISA does not encode 64 in this mapping: it uses ABI values
`log2_grid_x` and `log2_grid_y` to form both the initial `p` and
`1 << (log2_grid_x+log2_grid_y)` recurrence. The runner's static contract
checks those instruction dependencies.

<a id="section-2-4-batch-z-pointers"></a>

### 2.4 Batch-Z Planes and Base Pointers

**Documented hardware fact.** In cluster mode, gfx1250 exposes cluster-grid
Z as `TTMP7[31:16]`, Y as `TTMP7[15:0]`, X as `TTMP9`, and local
WG-in-cluster X/Y/Z plus cluster dimensions in `TTMP6` (CDNA5 ISA
Section 2.3.1, local text L769-L815). The compute-shader initialization
table further states that TTMP7 holds grid Z/Y and is loaded when grid Y/Z
is enabled (CDNA5 ISA Section 3.5.3.1, manual pages 29-30, local text
L2121-L2166; MI400 Shader Programming Guide Section 3.5.5.1, manual pages
52-53, local text L3693-L3739).

The runner launches:

```text
physical HIP WG grid = (cluster_grid_x*4,cluster_grid_y,batch)
clusterDim            = (4,1,1)
physical cluster grid = (cluster_grid_x,cluster_grid_y,batch)
```

Because `clusterDim.z=1`, every cluster has `wg_z=0`, and its cluster-grid Z
coordinate is also the physical WG Z coordinate. The batch ISA therefore
implements:

```text
batch_id = TTMP7[31:16] = TTMP7 >> 16
```

Calling this value a batch ID is a software design deduction, not a hardware
definition: the hardware documents it only as cluster-grid Z. It becomes a
batch ID because the runner programs exactly one cluster layer per matrix
and a cluster depth of one.

Batch is deliberately not flattened into the original X/Y task counter.
The initial task uses only `TTMP7[15:0]` and `TTMP9`, `T_XY` contains only
`C_N*C_M`, and the `+S` recurrence contains no Z term. The high half of
TTMP7 is consumed only by the following pointer prologue. Consequently
every Z plane traverses the same set of X/Y task IDs and logical
M/N tiles, independently.

The 120-byte batch ABI preserves the original 80-byte prefix and appends
five little-endian uint64 byte strides:

| Pointer after adjustment | Preloaded SGPRs / ABI offset | Contiguous batch-major layout | Byte stride |
| --- | --- | --- | ---: |
| output `D` (runner output `C`) | `s[2:3]`; stride `s[22:23]` at 80 | BF16 `[batch,M,N]` | `M*N*2` |
| `A` | `s[4:5]`; stride `s[24:25]` at 88 | uint8 packed FP4 `[batch,M,K/2]` | `M*(K/2)` |
| `B` | `s[6:7]`; stride `s[26:27]` at 96 | uint8 packed FP4 `[batch,N,K/2]` | `N*(K/2)` |
| `sA` | `s[8:9]`; stride `s[28:29]` at 104 | uint8 E8M0 `[batch,M,K/32]` | `M*(K/32)` |
| `sB` | `s[10:11]`; stride `s[30:31]` at 112 | uint8 E8M0 `[batch,N,K/32]` | `N*(K/32)` |

The runner requires exactly these shapes, dtypes, and contiguous outer-batch
strides. AB pre-shuffling changes the original kernel's within-matrix access
interpretation, not the contiguous matrix byte extent used as the batch
stride.

For each pointer, let `stride = stride_hi*2^32 + stride_lo` and
`b=batch_id`. The entry prologue computes a full two-limb product:

```text
offset_lo = low32(b * stride_lo)
offset_hi = low32(high32(b * stride_lo) + b * stride_hi)
offset    = offset_lo + (offset_hi << 32)
base_b    = base_0 + offset
```

The host-side span check makes this the exact unsigned 64-bit product rather
than an intended wraparound. This adjustment executes once, before the
legacy stream saves the D base or constructs any input descriptor. All
current/next descriptor generation, the K pipeline, output store, D rewind,
and persistent restart then operate unchanged on these five
batch-adjusted bases.

<a id="section-2-5-isolation-compatibility"></a>

### 2.5 Synchronization, Isolation, and Batch=1

**Documented hardware facts.** A workgroup barrier synchronizes waves in one
WG. Cluster barrier `-3` counts WGs in one cluster and releases the waves of
that cluster after its member WGs signal (CDNA5 ISA Section 5.6.6, manual
pages 50-51, local text L3319-L3367; MI400 Shader Programming Guide
Section 4.3.6.6, manual pages 82-83, local text L5364-L5412). Multicast
masks likewise name only WGs *within the same cluster* (CDNA5 ISA Section
10.7, manual pages 134-135, local text L9880-L9919; MI400 Shader Programming
Guide Section 4.9.8, manual pages 190-191, local text L13779-L13814).

**Implementation consequence.** Since `clusterDim.z=1`, one cluster never
contains WGs from two Z planes. The existing WG barriers, cluster barriers,
and A/SA/B/SB multicast masks therefore remain local to one batch plane.
There is no grid-wide barrier, cross-Z multicast bit, shared LDS allocation,
or cross-batch pointer. Different batches may progress concurrently and at
different rates without synchronizing or sharing operand payloads.

For `batch=1`, the only Z coordinate is zero. All five 64-bit products are
exactly zero, so the post-prologue C/D, A, B, sA, and sB base pointers equal
the original non-batch pointers bit-for-bit. The descriptors, tile mapping
for each task, pipeline, barriers, epilogue, rewind, and termination logic
remain the original instruction stream. The batch symbol still executes the
extra zero-offset prologue and consumes the 120-byte ABI, and its physical
X/Y schedule is adaptive: for `T_XY <= 64`, it launches only the useful
initial task IDs and may encode a recurrence stride larger than that physical
seed count. Alternatively, the batch runner accepts the original non-batch
symbol only for `batch=1`; that compatibility path retains the exact original
fixed geometry, 80-byte ABI, and execution schedule.

<a id="section-2-6-launch-examples"></a>

### 2.6 Launch Examples

#### 2.6.1 Exact 64x65536x32768 Launches

For `M=64,N=65536,K=32768`, the runner derives:

```text
W_N = 65536/256 = 256
W_M = 64/64     = 1
C_N = 256/4     = 64
C_M = 1/1       = 1
T_XY             = 64 cluster tasks per Z plane

batch_stride_D  = 64*65536*2       =    8388608 B = 0x00800000
batch_stride_A  = 64*(32768/2)     =    1048576 B = 0x00100000
batch_stride_B  = 65536*(32768/2)  = 1073741824 B = 0x40000000
batch_stride_sA = 64*(32768/32)    =      65536 B = 0x00010000
batch_stride_sB = 65536*(32768/32) =   67108864 B = 0x04000000
```

These are the exact values returned by `make_contiguous_batch_strides`.
`make_batched_launch_geometry` produces:

| Batch | Logical WG grid | Logical cluster grid | Physical HIP WG grid | `clusterDim` | Physical cluster grid | Block / encoded recurrence stride |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | `(256,1,1)` | `(64,1,1)` | `(256,1,1)` | `(4,1,1)` | `(64,1,1)` | `(128,1,1)` / 64 |
| 2 | `(256,1,2)` | `(64,1,2)` | `(256,1,2)` | `(4,1,1)` | `(64,1,2)` | `(128,1,1)` / 64 |

For batch 1, 256 WGs form 64 physical clusters. For batch 2, 512 WGs form
128 physical clusters, split into two identical planes of 256 WGs and 64
clusters. In either plane:

```text
p = cx, where cx=0..63 and cy=0
p spans 0..63 exactly
cluster_n = p
cluster_m = 0
Ntile = 4*p + wg_x
Mtile = 0
next task = p+64 >= T_XY, so every physical cluster terminates after one task
```

For Z=0, the base offsets are zero. In the batch-2 launch, Z=1 follows the
same `p=0..63` X/Y mapping but begins from each base plus the corresponding
stride listed above. The equality of the two X/Y traversals is established
by the ISA's omission of Z from `p` and `T_XY`; their address separation is
established by the one-time Z-derived base adjustment.

#### 2.6.2 Observed Batch-96 64x6144x7168 Launch

For `batch=96,M=64,N=6144,K=7168`, the logical space is:

```text
W_N = 6144/256 = 24
W_M = 64/64    = 1
C_N = 24/4     = 6
C_M = 1/1      = 1
T_XY            = 6 cluster tasks per Z plane

physical seeds C = T_XY = 6
recurrence S      = next_power_of_two(6) = 8
cluster grid      = (6,1,96)
HIP WG grid       = (6*4,1,96) = (24,1,96)
log2_grid_x/y   = (3,0)
encoded recurrence stride = 8
```

Physical Y is one, so the initial `p` values are exactly `cx=0..5`: all six
launched clusters are useful. Each next task is `p+8 >= 8 > 6`, so there is
no second round. Compared with the former maximum topology, each Z plane
drops from 64 clusters/256 WGs to 6 clusters/24 WGs. Across all 96 planes,
that is 576 clusters/2304 WGs instead of 6144 clusters/24576 WGs. This
removes the observed overlaunch while retaining one kernel dispatch and grid
Z equal to batch.

<a id="section-3-software-pipeline"></a>

## 3. End-to-End Software Pipeline

Every wave consumes one local A `M64xK256` tile, one local B `N64xK256`
quarter, and their corresponding scales, so the steady body remains 40 DS
loads and 16 WMMA instructions per wave. The two schedule families below are
specialization-driven software templates. They do not assign different
host-M regions: all waves have local `M[0:63]`, while wave `w` has WG-relative
`N[64*w:64*w+63]`.

Per wave and K256 body, the local operand traffic is exactly 16
`ds_load_b128` operations for A data, 16 for B data, four `ds_load_b32`
operations for SA, and four for SB. Thus `16+16+4+4=40` DS loads feed eight
C/D fragments and 16 WMMA operations.

The wave0/2 versus wave1/3 split is retained from the 128x128 reference ISA's
local instruction ordering because it remains consistent with the same four
TDM specialist roles. It does not prove that the new 1x4 output placement
needs this split. Keeping the split is an unassembled scheduling candidate;
a target implementation may reschedule it after validating dependencies,
register lifetimes, and issue behavior. The 16-WMMA and 40-DS dynamic counts
remain the wave-local invariant.

<a id="section-3-1-wave02-flow"></a>

### 3.1 wave0/2: `B-current -> A-current -> A-next -> B-next`

This template groups the host-A-data specialist (wave0) with the SA
specialist (wave2). It is a scheduling group, not an M-axis pair: wave0 owns
WG-relative N quarter `[0:63]` and wave2 owns `[128:191]`, while both own
local M `[0:63]`.

```text
Prologue (wave0/2 specialization schedule)
issue TDM A/SA slot0/body0              # 2 TDM in this wave group; wave0=A, wave2=SA
issue TDM A/SA slot1/body1              # 2 TDM in this wave group
issue TDM A/SA slot2/body2              # 2 TDM in this wave group
s_wait_tensorcnt 2                      # one wait per wave; ensures the oldest slot0 is ready
s_barrier_signal -1
s_barrier_wait 0xffff                   # rendezvous for A/B/SA/SB slot0 from wave0/1/2/3

SA-current first half # 2 ds_load_b32/wave
    ds_ld32_as0 (0_0)  # current
    ds_ld32_as1 (0_1)  # current

A-current first half # 8 ds_load_b128/wave
    ds_ld128_a0 (0_0)  # current
    ds_ld128_a1 (0_1)  # current
    ds_ld128_a2 (0_2)  # current
    ds_ld128_a3 (0_3)  # current
    ds_ld128_a4 (0_4)  # current
    ds_ld128_a5 (0_5)  # current
    ds_ld128_a6 (0_6)  # current
    ds_ld128_a7 (0_7)  # current

SB-current first half # 2 ds_load_b32/wave
    ds_ld32_bs0 (0_0)  # current
    ds_ld32_bs1 (0_1)  # current

B-current first half # 8 ds_load_b128/wave
    ds_ld128_b0 (0_0)  # current
    ds_ld128_b1 (0_1)  # current
    ds_ld128_b2 (0_2)  # current
    ds_ld128_b3 (0_3)  # current
    ds_ld128_b4 (0_4)  # current
    ds_ld128_b5 (0_5)  # current
    ds_ld128_b6 (0_6)  # current
    ds_ld128_b7 (0_7)  # current
                                        # total 20 DS/wave = 16 b128 + 4 b32
issue TDM A/SA slot3/body3              # 2 TDM in this wave group; wave0=A, wave2=SA

SB-current second half                  # 2 ds_load_b32/wave
    ds_ld32_bs2 (1_0)  # current
    ds_ld32_bs3 (1_1)  # current

B-current second half                   # 8 ds_load_b128/wave
    ds_ld128_b8 (1_0)  # current
    ds_ld128_b9 (1_1)  # current
    ds_ld128_b10 (1_2)  # current
    ds_ld128_b11 (1_3)  # current
    ds_ld128_b12 (1_4)  # current
    ds_ld128_b13 (1_5)  # current
    ds_ld128_b14 (1_6)  # current
    ds_ld128_b15 (1_7)  # current

P0/body0:
SA/A/SB/B-current --> slot0
SA/A/SB/B-current first half 20 DS + SB/B-current second half 10 DS issued
steady total 40 DS/wave, 16 WMMA; 2 TDM in this wave group

s_wait_dscnt 10                         # wait1, SA/A/SB/B current first half ready

SA-current second half # 2 ds_load_b32/wave
    wmma0 (0_0)  # K0
    wmma1 (0_1)  # K0
    ds_ld32_as2 (1_0)  # current
    ds_ld32_as3 (1_1)  # current

A-current second half # 8 ds_load_b128/wave
    ds_ld128_a8 (1_0)  # current
    ds_ld128_a9 (1_1)  # current
    ds_ld128_a10 (1_2)  # current
    ds_ld128_a11 (1_3)  # current
    wmma2 (0_2)  # K1
    wmma3 (0_3)  # K1
    ds_ld128_a12 (1_4)  # current
    ds_ld128_a13 (1_5)  # current
    ds_ld128_a14 (1_6)  # current
    ds_ld128_a15 (1_7)  # current

s_wait_dscnt 10                                     # wait2, SB/B current second half ready

SA/A-next first half + WMMA G1 local M[0:31],N[32:63] + WG barrier  # 2 ds_load_b32 + 8 ds_load_b128 / wave
    s_wait_tensorcnt 2
    s_barrier_signal -1
    wmma4 (1_0)  # K0
    wmma5 (1_1)  # K0
    s_wait_alu depctr_va_vdst(0)
    s_barrier_wait 0xffff
    ds_ld32_as0 (0_0)                   # next
    ds_ld128_a0 (0_0)                   # next
    ds_ld128_a1 (0_1)                   # next
    ds_ld128_a2 (0_2)                   # next
    ds_ld128_a3 (0_3)                   # next
    wmma6 (1_2)  # K1
    wmma7 (1_3)  # K1
    s_wait_alu depctr_va_vdst(0)
    ds_ld32_as1 (0_1)                   # next
    ds_ld128_a4 (0_4)                   # next
    ds_ld128_a5 (0_5)                   # next
    ds_ld128_a6 (0_6)                   # next
    ds_ld128_a7 (0_7)                   # next

s_wait_dscnt 10                         # wait3, SA/A current second half ready
s_barrier_signal -1

SB/B-next first half + WMMA G2 K0 local M[32:63],N[0:31] + issue TDM  # 1 ds_load_b32 + 4 ds_load_b128 / wave + 1 TDM / wave
    wmma8 (2_0)  # K0
    s_barrier_wait 0xffff
    issue TDM A/SA slot0/body4          # 1 TDM/wave, 2 total in this wave group
    wmma9 (2_1)  # K0
    s_wait_alu depctr_va_vdst(0)
    ds_ld32_bs0 (0_0)                   # next
    ds_ld128_b0 (0_0)                   # next
    ds_ld128_b1 (0_1)                   # next
    ds_ld128_b2 (0_2)                   # next
    ds_ld128_b3 (0_3)                   # next

SB/B-next first half + WMMA G2 K1 local M[32:63],N[0:31]  # 1 ds_load_b32 + 4 ds_load_b128 / wave
    wmma10 (2_2)  # K1
    wmma11 (2_3)  # K1
    s_wait_alu depctr_va_vdst(0)
    ds_ld32_bs1 (0_1)                   # next
    ds_ld128_b4 (0_4)                   # next
    ds_ld128_b5 (0_5)                   # next
    ds_ld128_b6 (0_6)                   # next
    ds_ld128_b7 (0_7)                   # next

SB/B-next second half + WMMA G3 K0 local M[32:63],N[32:63] + finish  # 1 ds_load_b32 + 4 ds_load_b128 / wave
    wmma12 (3_0)  # K0
    loop control SALU
    wmma13 (3_1)  # K0
    loop induction update
    loop control compare
    s_wait_alu depctr_va_vdst(0)
    ds_ld32_bs2 (1_0)                   # next
    ds_ld128_b8 (1_0)                   # next
    ds_ld128_b9 (1_1)                   # next
    ds_ld128_b10 (1_2)                  # next
    ds_ld128_b11 (1_3)                  # next

SB/B-next second half + WMMA G3 K1 local M[32:63],N[32:63]  # 1 ds_load_b32 + 4 ds_load_b128 / wave
    wmma14 (3_2)  # K1
    wmma15 (3_3)  # K1
    s_wait_alu depctr_va_vdst(0)
    ds_ld32_bs3 (1_1)                   # next
    ds_ld128_b12 (1_4)                  # next
    ds_ld128_b13 (1_5)                  # next
    ds_ld128_b14 (1_6)                  # next
    ds_ld128_b15 (1_7)                  # next
    loop control branch

P0->P1 boundary
loop branch not taken            # candidate steady path enters P1 directly
phase advance: P0 next becomes P1 current; P1 reuses the corresponding VGPR

P1/body1:
SA/A/SB/B-current --> slot1
SA/A/SB/B-current first half 20 DS + SB/B-current second half 10 DS issued
steady total 40 DS/wave, 16 WMMA; 2 TDM in this wave group

s_wait_dscnt 10                         # wait1, SA/A/SB/B current first half ready

SA-current second half # 2 ds_load_b32/wave
    p1_wmma0 (0_0)  # K0
    p1_wmma1 (0_1)  # K0
    ds_ld32_as2 (1_0)  # current
    ds_ld32_as3 (1_1)  # current

A-current second half # 8 ds_load_b128/wave
    ds_ld128_a8 (1_0)  # current
    ds_ld128_a9 (1_1)  # current
    ds_ld128_a10 (1_2)  # current
    ds_ld128_a11 (1_3)  # current
    p1_wmma2 (0_2)  # K1
    p1_wmma3 (0_3)  # K1
    ds_ld128_a12 (1_4)  # current
    ds_ld128_a13 (1_5)  # current
    ds_ld128_a14 (1_6)  # current
    ds_ld128_a15 (1_7)  # current

s_wait_dscnt 10                         # P1 wait2, SB/B current second half ready

SA/A-next first half + WMMA G1 local M[0:31],N[32:63] + WG barrier  # 2 ds_load_b32 + 8 ds_load_b128 / wave
    s_wait_tensorcnt 2
    s_barrier_signal -1
    p1_wmma4 (1_0)  # K0
    p1_wmma5 (1_1)  # K0
    s_wait_alu depctr_va_vdst(0)
    s_barrier_wait 0xffff
    ds_ld32_as0 (0_0)                   # next
    ds_ld128_a0 (0_0)                   # next
    ds_ld128_a1 (0_1)                   # next
    ds_ld128_a2 (0_2)                   # next
    ds_ld128_a3 (0_3)                   # next
    p1_wmma6 (1_2)  # K1
    p1_wmma7 (1_3)  # K1
    s_wait_alu depctr_va_vdst(0)
    ds_ld32_as1 (0_1)                   # next
    ds_ld128_a4 (0_4)                   # next
    ds_ld128_a5 (0_5)                   # next
    ds_ld128_a6 (0_6)                   # next
    ds_ld128_a7 (0_7)                   # next

s_wait_dscnt 10                         # P1 wait3, SA/A current second half ready
s_barrier_signal -1

P1 current second half ready, A-next first half issued  # next step enters P1 WMMA_G2 + barrier wait/TDM
```

<a id="section-3-2-wave13-flow"></a>

### 3.2 wave1/3: `A-current -> B-current -> B-next -> A-next`

This template groups the host-B-data specialist (wave1) with the SB
specialist (wave3). It is also non-spatial: wave1 owns WG-relative N quarter
`[64:127]` and wave3 owns `[192:255]`, while both own local M `[0:63]`.

```text
Prologue (wave1/3 specialization schedule)
issue TDM B/SB slot0/body0              # 2 TDM in this wave group; wave1=B, wave3=SB
issue TDM B/SB slot1/body1              # 2 TDM in this wave group
issue TDM B/SB slot2/body2              # 2 TDM in this wave group
s_wait_tensorcnt 2                     # one wait per wave; ensures the oldest slot0 is ready
s_barrier_signal -1
s_barrier_wait 0xffff                  # rendezvous for A/B/SA/SB slot0 from wave0/1/2/3

SB-current first half # 2 ds_load_b32/wave
    ds_ld32_bs0 (0_0)  # current
    ds_ld32_bs1 (0_1)  # current

B-current first half # 8 ds_load_b128/wave
    ds_ld128_b0 (0_0)  # current
    ds_ld128_b1 (0_1)  # current
    ds_ld128_b2 (0_2)  # current
    ds_ld128_b3 (0_3)  # current
    ds_ld128_b4 (0_4)  # current
    ds_ld128_b5 (0_5)  # current
    ds_ld128_b6 (0_6)  # current
    ds_ld128_b7 (0_7)  # current

SA-current first half # 2 ds_load_b32/wave
    ds_ld32_as0 (0_0)  # current
    ds_ld32_as1 (0_1)  # current

A-current first half # 8 ds_load_b128/wave
    ds_ld128_a0 (0_0)  # current
    ds_ld128_a1 (0_1)  # current
    ds_ld128_a2 (0_2)  # current
    ds_ld128_a3 (0_3)  # current
    ds_ld128_a4 (0_4)  # current
    ds_ld128_a5 (0_5)  # current
    ds_ld128_a6 (0_6)  # current
    ds_ld128_a7 (0_7)  # current
                                        # total 20 DS/wave = 16 b128 + 4 b32
issue TDM B/SB slot3/body3              # 2 TDM in this wave group

SA-current second half                  # 2 ds_load_b32/wave
    ds_ld32_as2 (1_0)  # current
    ds_ld32_as3 (1_1)  # current

A-current second half                   # 8 ds_load_b128/wave
    ds_ld128_a8 (1_0)  # current
    ds_ld128_a9 (1_1)  # current
    ds_ld128_a10 (1_2)  # current
    ds_ld128_a11 (1_3)  # current
    ds_ld128_a12 (1_4)  # current
    ds_ld128_a13 (1_5)  # current
    ds_ld128_a14 (1_6)  # current
    ds_ld128_a15 (1_7)  # current

P0/body0:
SB/B/SA/A-current --> slot0
SB/B/SA/A-current first half 20 DS + SA/A-current second half 10 DS issued
steady total 40 DS/wave, 16 WMMA; 2 TDM in this wave group

s_wait_dscnt 10                         # wait1, SB/B/SA/A current first half ready

SB-current second half # 2 ds_load_b32/wave
    wmma0 (0_0)  # K0
    wmma1 (0_1)  # K0
    ds_ld32_bs2 (1_0)  # current
    ds_ld32_bs3 (1_1)  # current

B-current second half # 8 ds_load_b128/wave
    ds_ld128_b8 (1_0)  # current
    ds_ld128_b9 (1_1)  # current
    ds_ld128_b10 (1_2)  # current
    ds_ld128_b11 (1_3)  # current
    wmma2 (0_2)  # K1
    wmma3 (0_3)  # K1
    ds_ld128_b12 (1_4)  # current
    ds_ld128_b13 (1_5)  # current
    ds_ld128_b14 (1_6)  # current
    ds_ld128_b15 (1_7)  # current

s_wait_dscnt 10                         # wait2, SA/A current second half ready

SB/B-next first half + WMMA G1 local M[32:63],N[0:31] + WG barrier  # 2 ds_load_b32 + 8 ds_load_b128 / wave
    s_wait_tensorcnt 2
    s_barrier_signal -1
    wmma4 (1_0)  # K0
    wmma5 (1_1)  # K0
    s_wait_alu depctr_va_vdst(0)
    s_barrier_wait 0xffff
    ds_ld32_bs0 (0_0)                   # next
    ds_ld128_b0 (0_0)                   # next
    ds_ld128_b1 (0_1)                   # next
    ds_ld128_b2 (0_2)                   # next
    ds_ld128_b3 (0_3)                   # next
    wmma6 (1_2)  # K1
    wmma7 (1_3)  # K1
    s_wait_alu depctr_va_vdst(0)
    ds_ld32_bs1 (0_1)                   # next
    ds_ld128_b4 (0_4)                   # next
    ds_ld128_b5 (0_5)                   # next
    ds_ld128_b6 (0_6)                   # next
    ds_ld128_b7 (0_7)                   # next

s_wait_dscnt 10                         # wait3, SB/B current second half ready
s_barrier_signal -1

SA/A-next first half + WMMA G2 K0 local M[0:31],N[32:63] + issue TDM  # 1 ds_load_b32 + 4 ds_load_b128 / wave + 1 TDM / wave
    wmma8 (2_0)  # K0
    s_barrier_wait 0xffff
    issue TDM B/SB slot0/body4          # 1 TDM/wave, 2 total in this wave group
    wmma9 (2_1)  # K0
    s_wait_alu depctr_va_vdst(0)
    ds_ld32_as0 (0_0)                   # next
    ds_ld128_a0 (0_0)                   # next
    ds_ld128_a1 (0_1)                   # next
    ds_ld128_a2 (0_2)                   # next
    ds_ld128_a3 (0_3)                   # next

SA/A-next first half + WMMA G2 K1 local M[0:31],N[32:63]  # 1 ds_load_b32 + 4 ds_load_b128 / wave
    wmma10 (2_2)  # K1
    wmma11 (2_3)  # K1
    s_wait_alu depctr_va_vdst(0)
    ds_ld32_as1 (0_1)                   # next
    ds_ld128_a4 (0_4)                   # next
    ds_ld128_a5 (0_5)                   # next
    ds_ld128_a6 (0_6)                   # next
    ds_ld128_a7 (0_7)                   # next

SA/A-next second half + WMMA G3 K0 local M[32:63],N[32:63] + finish  # 1 ds_load_b32 + 4 ds_load_b128 / wave
    wmma12 (3_0)  # K0
    loop control SALU
    wmma13 (3_1)  # K0
    loop induction update
    loop control compare
    s_wait_alu depctr_va_vdst(0)
    ds_ld32_as2 (1_0)                   # next
    ds_ld128_a8 (1_0)                   # next
    ds_ld128_a9 (1_1)                   # next
    ds_ld128_a10 (1_2)                  # next
    ds_ld128_a11 (1_3)                  # next

SA/A-next second half + WMMA G3 K1 local M[32:63],N[32:63]  # 1 ds_load_b32 + 4 ds_load_b128 / wave
    wmma14 (3_2)  # K1
    wmma15 (3_3)  # K1
    s_wait_alu depctr_va_vdst(0)
    ds_ld32_as3 (1_1)                   # next
    ds_ld128_a12 (1_4)                  # next
    ds_ld128_a13 (1_5)                  # next
    ds_ld128_a14 (1_6)                  # next
    ds_ld128_a15 (1_7)                  # next
    loop control branch

P0->P1 boundary
loop branch not taken            # candidate steady path enters P1 directly
phase advance: P0 next becomes P1 current; P1 reuses the corresponding VGPR

P1/body1:
SB/B/SA/A-current --> slot1
SB/B/SA/A-current first half 20 DS + SA/A-current second half 10 DS issued
steady total 40 DS/wave, 16 WMMA; 2 TDM in this wave group

s_wait_dscnt 10                         # wait1, SB/B/SA/A current first half ready

SB-current second half # 2 ds_load_b32/wave
    p1_wmma0 (0_0)  # K0
    p1_wmma1 (0_1)  # K0
    ds_ld32_bs2 (1_0)  # current
    ds_ld32_bs3 (1_1)  # current

B-current second half # 8 ds_load_b128/wave
    ds_ld128_b8 (1_0)  # current
    ds_ld128_b9 (1_1)  # current
    ds_ld128_b10 (1_2)  # current
    ds_ld128_b11 (1_3)  # current
    p1_wmma2 (0_2)  # K1
    p1_wmma3 (0_3)  # K1
    ds_ld128_b12 (1_4)  # current
    ds_ld128_b13 (1_5)  # current
    ds_ld128_b14 (1_6)  # current
    ds_ld128_b15 (1_7)  # current

s_wait_dscnt 10                         # P1 wait2, SA/A current second half ready

SB/B-next first half + WMMA G1 local M[32:63],N[0:31] + WG barrier  # 2 ds_load_b32 + 8 ds_load_b128 / wave
    s_wait_tensorcnt 2
    s_barrier_signal -1
    p1_wmma4 (1_0)  # K0
    p1_wmma5 (1_1)  # K0
    s_wait_alu depctr_va_vdst(0)
    s_barrier_wait 0xffff
    ds_ld32_bs0 (0_0)                   # next
    ds_ld128_b0 (0_0)                   # next
    ds_ld128_b1 (0_1)                   # next
    ds_ld128_b2 (0_2)                   # next
    ds_ld128_b3 (0_3)                   # next
    p1_wmma6 (1_2)  # K1
    p1_wmma7 (1_3)  # K1
    s_wait_alu depctr_va_vdst(0)
    ds_ld32_bs1 (0_1)                   # next
    ds_ld128_b4 (0_4)                   # next
    ds_ld128_b5 (0_5)                   # next
    ds_ld128_b6 (0_6)                   # next
    ds_ld128_b7 (0_7)                   # next

s_wait_dscnt 10                         # P1 wait3, SB/B current second half ready
s_barrier_signal -1

P1 current second half ready, B-next first half issued  # next step enters P1 WMMA_G2 + barrier wait/TDM
```

<a id="section-4-p0-details"></a>

## 4. P0 Detailed Pipeline and Wave-Local Tile Coverage

Chapter 4 describes the unchanged per-wave `64x64xK256` compute body. Table
coordinates are wave-local unless explicitly prefixed by `WG-relative`.
For wave `w`, convert a listed output `(m,n)` to the WG tile with:

```text
WG-relative M = m
WG-relative N = 64*w + n
host M        = wg_m_origin + m
host N        = wg_n_origin + 64*w + n
```

Thus the two schedule families can share local coverage tables without
implying a two-dimensional wave grid.

<a id="section-4-1-wave02-details"></a>

### 4.1 wave0/2 Specialization Schedule

<a id="section-4-1-1-wave02-host-tile"></a>

#### 4.1.1 wave0/2 Wave-Local Tile Coverage

The same local tables apply to both waves. Wave0 adds N origin `0`; wave2
adds N origin `128`. Both add M origin `0`.

**A data: one `ds_load_b128 = M16xK64` per cell**

| Wave-local M / K | K[0:63] | K[64:127] | K[128:191] | K[192:255] |
|---|---|---|---|---|
| M0 [0:15] | `ds_ld128_a0` (`0_0`) | `ds_ld128_a1` (`0_1`) | `ds_ld128_a4` (`0_4`) | `ds_ld128_a5` (`0_5`) |
| M1 [16:31] | `ds_ld128_a2` (`0_2`) | `ds_ld128_a3` (`0_3`) | `ds_ld128_a6` (`0_6`) | `ds_ld128_a7` (`0_7`) |
| M2 [32:47] | `ds_ld128_a8` (`1_0`) | `ds_ld128_a9` (`1_1`) | `ds_ld128_a12` (`1_4`) | `ds_ld128_a13` (`1_5`) |
| M3 [48:63] | `ds_ld128_a10` (`1_2`) | `ds_ld128_a11` (`1_3`) | `ds_ld128_a14` (`1_6`) | `ds_ld128_a15` (`1_7`) |

**A scale: one `ds_load_b32 = M32xK128 scales` per cell**

| Wave-local M / K | K0 [0:127] | K1 [128:255] |
|---|---|---|
| M0 [0:31] | `ds_ld32_as0` (`0_0`) | `ds_ld32_as1` (`0_1`) |
| M1 [32:63] | `ds_ld32_as2` (`1_0`) | `ds_ld32_as3` (`1_1`) |

**B data: one `ds_load_b128 = N16xK64` per cell**

| Wave-local N / K | K[0:63] | K[64:127] | K[128:191] | K[192:255] |
|---|---|---|---|---|
| N0 [0:15] | `ds_ld128_b0` (`0_0`) | `ds_ld128_b1` (`0_1`) | `ds_ld128_b4` (`0_4`) | `ds_ld128_b5` (`0_5`) |
| N1 [16:31] | `ds_ld128_b2` (`0_2`) | `ds_ld128_b3` (`0_3`) | `ds_ld128_b6` (`0_6`) | `ds_ld128_b7` (`0_7`) |
| N2 [32:47] | `ds_ld128_b8` (`1_0`) | `ds_ld128_b9` (`1_1`) | `ds_ld128_b12` (`1_4`) | `ds_ld128_b13` (`1_5`) |
| N3 [48:63] | `ds_ld128_b10` (`1_2`) | `ds_ld128_b11` (`1_3`) | `ds_ld128_b14` (`1_6`) | `ds_ld128_b15` (`1_7`) |

**B scale: one `ds_load_b32 = N32xK128 scales` per cell**

| Wave-local N / K | K0 [0:127] | K1 [128:255] |
|---|---|---|
| N0 [0:31] | `ds_ld32_bs0` (`0_0`) | `ds_ld32_bs1` (`0_1`) |
| N1 [32:63] | `ds_ld32_bs2` (`1_0`) | `ds_ld32_bs3` (`1_1`) |

**Each cell in the K0 table computes `M16xN32xK128`, with a K range of `[0,127]`:**

| K0 [0,127] / wave-local M,N | N0 [0,31] | N1 [32,63] |
|---|---:|---:|
| M0 [0,15] | `wmma0` (`0_0`) | `wmma4` (`1_0`) |
| M1 [16,31] | `wmma1` (`0_1`) | `wmma5` (`1_1`) |
| M2 [32,47] | `wmma8` (`2_0`) | `wmma12` (`3_0`) |
| M3 [48,63] | `wmma9` (`2_1`) | `wmma13` (`3_1`) |

**Each cell in the K1 table computes the second K128 accumulation over `[128,255]` for the same MxN output fragment:**

| K1 [128,255] / wave-local M,N | N0 [0,31] | N1 [32,63] |
|---|---:|---:|
| M0 [0,15] | `wmma2` (`0_2`) | `wmma6` (`1_2`) |
| M1 [16,31] | `wmma3` (`0_3`) | `wmma7` (`1_3`) |
| M2 [32,47] | `wmma10` (`2_2`) | `wmma14` (`3_2`) |
| M3 [48,63] | `wmma11` (`2_3`) | `wmma15` (`3_3`) |

<a id="section-4-2-wave13-details"></a>

### 4.2 wave1/3 Specialization Schedule

<a id="section-4-2-1-wave13-host-tile"></a>

#### 4.2.1 wave1/3 Wave-Local Tile Coverage

The same local tables apply to both waves. Wave1 adds N origin `64`; wave3
adds N origin `192`. Both add M origin `0`.

**A data: one `ds_load_b128 = M16xK64` per cell**

| Wave-local M / K | K[0:63] | K[64:127] | K[128:191] | K[192:255] |
|---|---|---|---|---|
| M0 [0:15] | `ds_ld128_a0` (`0_0`) | `ds_ld128_a1` (`0_1`) | `ds_ld128_a4` (`0_4`) | `ds_ld128_a5` (`0_5`) |
| M1 [16:31] | `ds_ld128_a2` (`0_2`) | `ds_ld128_a3` (`0_3`) | `ds_ld128_a6` (`0_6`) | `ds_ld128_a7` (`0_7`) |
| M2 [32:47] | `ds_ld128_a8` (`1_0`) | `ds_ld128_a9` (`1_1`) | `ds_ld128_a12` (`1_4`) | `ds_ld128_a13` (`1_5`) |
| M3 [48:63] | `ds_ld128_a10` (`1_2`) | `ds_ld128_a11` (`1_3`) | `ds_ld128_a14` (`1_6`) | `ds_ld128_a15` (`1_7`) |

**A scale: one `ds_load_b32 = M32xK128 scales` per cell**

| Wave-local M / K | K0 [0:127] | K1 [128:255] |
|---|---|---|
| M0 [0:31] | `ds_ld32_as0` (`0_0`) | `ds_ld32_as1` (`0_1`) |
| M1 [32:63] | `ds_ld32_as2` (`1_0`) | `ds_ld32_as3` (`1_1`) |

**B data: one `ds_load_b128 = N16xK64` per cell**

| Wave-local N / K | K[0:63] | K[64:127] | K[128:191] | K[192:255] |
|---|---|---|---|---|
| N0 [0:15] | `ds_ld128_b0` (`0_0`) | `ds_ld128_b1` (`0_1`) | `ds_ld128_b4` (`0_4`) | `ds_ld128_b5` (`0_5`) |
| N1 [16:31] | `ds_ld128_b2` (`0_2`) | `ds_ld128_b3` (`0_3`) | `ds_ld128_b6` (`0_6`) | `ds_ld128_b7` (`0_7`) |
| N2 [32:47] | `ds_ld128_b8` (`1_0`) | `ds_ld128_b9` (`1_1`) | `ds_ld128_b12` (`1_4`) | `ds_ld128_b13` (`1_5`) |
| N3 [48:63] | `ds_ld128_b10` (`1_2`) | `ds_ld128_b11` (`1_3`) | `ds_ld128_b14` (`1_6`) | `ds_ld128_b15` (`1_7`) |

**B scale: one `ds_load_b32 = N32xK128 scales` per cell**

| Wave-local N / K | K0 [0:127] | K1 [128:255] |
|---|---|---|
| N0 [0:31] | `ds_ld32_bs0` (`0_0`) | `ds_ld32_bs1` (`0_1`) |
| N1 [32:63] | `ds_ld32_bs2` (`1_0`) | `ds_ld32_bs3` (`1_1`) |

**Each cell in the K0 table computes `M16xN32xK128`, with a K range of `[0,127]`:**

| K0 [0,127] / wave-local M,N | N0 [0,31] | N1 [32,63] |
|---|---:|---:|
| M0 [0,15] | `wmma0` (`0_0`) | `wmma8` (`2_0`) |
| M1 [16,31] | `wmma1` (`0_1`) | `wmma9` (`2_1`) |
| M2 [32,47] | `wmma4` (`1_0`) | `wmma12` (`3_0`) |
| M3 [48,63] | `wmma5` (`1_1`) | `wmma13` (`3_1`) |

**Each cell in the K1 table computes the second K128 accumulation over `[128,255]` for the same MxN output fragment:**

| K1 [128,255] / wave-local M,N | N0 [0,31] | N1 [32,63] |
|---|---:|---:|
| M0 [0,15] | `wmma2` (`0_2`) | `wmma10` (`2_2`) |
| M1 [16,31] | `wmma3` (`0_3`) | `wmma11` (`2_3`) |
| M2 [32,47] | `wmma6` (`1_2`) | `wmma14` (`3_2`) |
| M3 [48,63] | `wmma7` (`1_3`) | `wmma15` (`3_3`) |

<a id="section-5-epilogue-design"></a>

## 5. Epilogue Design and Final Store Contract

This chapter specifies a natural epilogue for the wave-tile `64x64`, WG-tile
`64x256` design. It is design documentation; no assembled gfx1250 ISA exists
for the `f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps` target. The following
evidence labels are used throughout the chapter:

| Label | Meaning |
|---|---|
| Hardware fact | A behavior stated by the CDNA5 ISA or MI400 Shader Programming Guide. |
| 128x128 reference-ISA fact | A local compute or specialization property observed in `f4gemm_bf16_mxfp4_ABpreShuffle_128x128_4x4_ps.s`. It is historical evidence, not a target-geometry fact. |
| 256x256 reference-ISA fact | An output-path instruction or descriptor idiom observed in `my_code/fmha/dump_asm/hsa/gfx1250/f4gemm/f4gemm_bf16_mxfp4_ABpreShuffle_256x256_4x4_ps.s`. It is historical evidence, not a target-address or resource fact. |
| Static derivation | Arithmetic or mapping derived from hardware facts and the geometry contract in Chapters 1-4. |
| Candidate choice | A proposed allocation, descriptor, or schedule that is not final until assembled and run on the target. |
| Validation boundary | A property that cannot be made bit-exact from the available text and must be checked in target assembly, disassembly, metadata, or hardware execution. |

<a id="section-5-1-output-geometry"></a>

### 5.1 Output Geometry and Fragment Mapping

**Hardware facts.** `v_wmma_scale_f32_32x16x128_f4` computes a 32x16
single-precision C/D matrix from FP4 inputs. The CDNA5 ISA states this contract
at L26180-L26200. The C/D storage description at CDNA5 ISA L7367-L7409 and
MI400 Shader Guide L10676-L10718 shows that a 32x16 F32 C/D matrix occupies
16 VGPRs across a wave32. Because host B supplies the hardware Matrix A and
host A supplies the hardware Matrix B, this hardware result is oriented as
host N32 x host M16 and must be transposed into host M x N order during LDS
staging.

**Static derivation.** One 32x16 fragment contains 512 F32 values:

```text
32 * 16 = 512 F32 values
512 / 32 lanes = 16 F32 values per lane
16 values per lane * 4 bytes = 16 VGPRs per fragment
```

The 64x64 wave tile has four wave-local M blocks and two wave-local N blocks,
so it has
eight independent C/D fragments. The K0 and K1 instructions shown below both
accumulate into the same named fragment; they are not separate output
fragments. This table repeats the exact Chapter 4 coordinates and pairings.

| Fragment | Wave-local output coordinates | wave0/2 K0 -> K1 | wave1/3 K0 -> K1 |
|---|---|---|---|
| `F00` | `M[0:15], N[0:31]` | `wmma0 -> wmma2` | `wmma0 -> wmma2` |
| `F01` | `M[0:15], N[32:63]` | `wmma4 -> wmma6` | `wmma8 -> wmma10` |
| `F10` | `M[16:31], N[0:31]` | `wmma1 -> wmma3` | `wmma1 -> wmma3` |
| `F11` | `M[16:31], N[32:63]` | `wmma5 -> wmma7` | `wmma9 -> wmma11` |
| `F20` | `M[32:47], N[0:31]` | `wmma8 -> wmma10` | `wmma4 -> wmma6` |
| `F21` | `M[32:47], N[32:63]` | `wmma12 -> wmma14` | `wmma12 -> wmma14` |
| `F30` | `M[48:63], N[0:31]` | `wmma9 -> wmma11` | `wmma5 -> wmma7` |
| `F31` | `M[48:63], N[32:63]` | `wmma13 -> wmma15` | `wmma13 -> wmma15` |

The `wmmaN` labels describe the Chapter 4 schedule, not fixed physical VGPR
numbers. A code generator must map both members of each K0/K1 pair to the same
physical 16-VGPR accumulator block.

Therefore each wave owns:

```text
8 fragments * 32 * 16 = 4096 BF16 outputs
wave output = 64 * 64 = 4096 BF16 outputs
```

The four wave tiles concatenate only along host N:

| Logical wave `w` | WG-relative M | WG-relative N | Coordinate transform |
|---:|---|---|---|
| 0 | `[0,63]` | `[0,63]` | `(m,n) -> (m,n)` |
| 1 | `[0,63]` | `[64,127]` | `(m,n) -> (m,64+n)` |
| 2 | `[0,63]` | `[128,191]` | `(m,n) -> (m,128+n)` |
| 3 | `[0,63]` | `[192,255]` | `(m,n) -> (m,192+n)` |

The combined WG output is exactly:

```text
4 waves * 4096 elements/wave = 16384 BF16 elements
64 rows * 256 columns         = 16384 BF16 elements
```

<a id="section-5-2-vgpr-layout"></a>

### 5.2 Candidate Physical VGPR Layout

The following is one explicit, internally non-overlapping **candidate choice**.
It is selected for simple addressing and review, not because target metadata
already guarantees it.

| Fragment | Wave-local coordinates | F32 accumulator block | Packed-BF16 staging block |
|---|---|---|---|
| `F00` | `M[0:15], N[0:31]` | `v256:v271` | `v128:v135` |
| `F01` | `M[0:15], N[32:63]` | `v272:v287` | `v136:v143` |
| `F10` | `M[16:31], N[0:31]` | `v288:v303` | `v144:v151` |
| `F11` | `M[16:31], N[32:63]` | `v304:v319` | `v152:v159` |
| `F20` | `M[32:47], N[0:31]` | `v320:v335` | `v160:v167` |
| `F21` | `M[32:47], N[32:63]` | `v336:v351` | `v168:v175` |
| `F30` | `M[48:63], N[0:31]` | `v352:v367` | `v176:v183` |
| `F31` | `M[48:63], N[32:63]` | `v368:v383` | `v184:v191` |

This layout has 128 contiguous accumulator VGPRs in `v256:v383` and 64
contiguous packed-BF16 staging VGPRs in `v128:v191`. Each fragment converts as
follows:

```text
512 F32 values / fragment
= 16 F32 values per lane
= 8 packed BF16 dwords per lane
= 8 packed-BF16 VGPRs per fragment

8 fragments * 8 packed VGPRs = 64 packed-BF16 VGPRs per wave
```

The staging allocation is a lifetime reuse choice. `v128:v191` may be reused
only after all final input-operand and prefetch-address uses of those physical
registers have ended. This epilogue reserves `v0:v127` for lane/LDS addresses,
vector-valued descriptor/index temporaries, and low operand banks; the TDM
descriptor groups themselves remain in SGPRs. `v192:v255` is intentionally
unassigned by this epilogue and remains available to the full-kernel allocator
for operand banks or additional temporaries. The allocation of those lower
regions is outside this epilogue table and still has to be proven
non-overlapping by assembled code.

**VGPR MSB requirement.** CDNA5 ISA L1155-L1197 documents that
`s_set_vgpr_msb` appends two address MSBs independently to the destination and
source operand fields. The accumulator bank `v256:v383` therefore requires
bank value `01` on the WMMA C/D fields. For a VOP3 packed conversion from
physical `v256:v257` to physical `v128`, the conceptual candidate is:

```text
# Candidate encoding; verify the immediate and physical registers in objdump.
s_set_vgpr_msb 0x05
# dst bank=00, src0 bank=01, src1 bank=01, src2 is unused
v_cvt_pk_bf16_f32 v128, v0, v1
# physical operation intended: v128 <- pack_bf16(v256, v257)
```

The documented immediate ordering is `{dst, src2, src1, src0}`. The final
emitter must set the appropriate MSBs around every WMMA, conversion, address,
and DS instruction, and must restore the low bank before low-bank DS sources
or addresses are used. The 256x256 reference ISA demonstrates
`s_set_vgpr_msb` transitions around conversion and DS sequences, for example
at L6522-L6523 and L6587-L6588. It does not establish the register assignment
for this target.

**Dependency requirement.** Before conversion reads the final WMMA results,
the candidate executes `s_wait_alu depctr_va_vdst(0)`. Before DS consumes
freshly converted staging VGPRs, it executes another required VA_VDST wait
unless target scheduling mode and independently verified spacing make that
wait redundant. Before any DS-source VGPR is overwritten, `s_wait_dscnt 0`
must establish that the DS operations have consumed their sources; the MI400
Shader Guide L5918-L5963 describes the relevant RAW, WAW, and WAR boundaries.

The highest explicitly allocated physical VGPR is `v383`, so the candidate
minimum next-free boundary is **at least 384**. This is not a
`next_free_vgpr` metadata claim. Extra operand banks, compiler temporaries,
allocation granularity, or an assembler restriction may raise the emitted
value, and only assembled target metadata may define the final value.

<a id="section-5-3-bf16-lds-staging"></a>

### 5.3 BF16 Conversion and LDS Staging

**256x256 reference-ISA facts.** The reference output path uses
`v_cvt_pk_bf16_f32` at L6458-L6521, `ds_store_b128` at L6588-L6620,
`s_wait_dscnt 0` at L6621, and `tensor_store_from_lds` at L6622, with a
second historical store at L6809. Its lane-address setup and DS offset pattern
provide an implementation idiom only; its tile geometry, LDS bases, and store
count do not describe this target.

CDNA5 ISA L32879-L32889 defines `V_CVT_PK_BF16_F32` as converting two F32
inputs to one packed BF16 dword using round-to-nearest-even. This is the
architecturally appropriate conversion family for a normal BF16 output unless
the target requires a different rounding contract.

The migrated layout principle is to use the low four lane bits as the
wave-local M row, lane bit 4 to select a wave-local N subrange, and consecutive
packed pairs to transpose the hardware N32 x M16 fragment into a host
M16 x N32 row-major tile.

The candidate sequence is:

```text
1. Complete the final K phase and stop issuing a real future prefetch.
   Use a scalar branch, or a validated null TDM descriptor with tile_dim0=0;
   EXEC masking alone is not sufficient because tensor instructions ignore EXEC.

2. Drain speculative and final-pipeline work:
   s_wait_dscnt 0
   s_wait_tensorcnt 0
   s_wait_alu depctr_va_vdst(0)

3. For each F00..F31 accumulator block:
   issue 8 v_cvt_pk_bf16_f32 operations
   map the 16 F32 values/lane to 8 packed BF16 dwords/lane

4. Before DS reads the conversion destinations:
   s_wait_alu depctr_va_vdst(0)

5. Apply the migrated lane/layout transpose and write each wave's 64x64
   N quarter to the row-major 64x256 WG staging tile.

6. Complete all output DS writes:
   s_wait_dscnt 0
```

For an exact candidate address map, define:

```text
lane_row    = lane_id & 15
lane_n_half = lane_id >> 4          # 0 or 1 for wave32

fragment_m = 0, 16, 32, or 48
fragment_n = 0 or 32
store_group = 0 or 1                # first or second four packed VGPRs
wave_n_origin = 64 * logical_wave_id

lds_address =
    0x2A800
  + (fragment_m + lane_row) * 0x200
  + (wave_n_origin + fragment_n
     + 16 * store_group + 8 * lane_n_half) * 2
```

One `ds_store_b128` writes four packed dwords, or eight BF16 values, per lane.
For a fragment, `store_group=0` uses the first four staging VGPRs and
`store_group=1` uses the next four. Lanes 0-15 and 16-31 address the same 16
wave-local M rows but disjoint eight-column wave-local N spans. This produces two
`ds_store_b128` instructions per 32x16 fragment without overlap.

The candidate DS count is therefore rigorous for this specific full-EXEC
`ds_store_b128` mapping:

```text
one ds_store_b128 wave aggregate = 32 lanes * 16 bytes = 512 bytes = 0x200
one wave output                  = 64 * 64 * 2 = 8192 bytes = 0x2000
candidate DS writes per wave     = 0x2000 / 0x200 = 16
candidate DS writes per WG       = 4 * 16 = 64
```

These are candidate instruction counts for the specified DS width and address
map. Another correct code generator may use a different DS width or schedule;
its exact count is target-codegen dependent and must be checked in
disassembly.

The row-major WG staging geometry is:

| Quantity | Candidate value |
|---|---:|
| One wave output | `64x64x2 = 0x2000` bytes |
| One WG output | `64x256x2 = 0x8000` bytes |
| WG output row stride | `256x2 = 0x200` bytes |
| `output_lds_base` | `0x2A800` |
| Output staging range | `[0x2A800,0x32800)` |

The wave row-zero staging origins are:

| Logical wave | WG-relative N origin | Staging origin |
|---:|---:|---|
| 0 | 0 | `output_lds_base + 0x000` |
| 1 | 64 | `output_lds_base + 0x080` |
| 2 | 128 | `output_lds_base + 0x100` |
| 3 | 192 | `output_lds_base + 0x180` |

Each increment of `0x080` is 64 BF16 columns. These are row-major N-quarter
origins, not contiguous `0x2000`-byte wave payloads. Every wave is interleaved
at the full `0x200` WG row stride, so DS addressing must use the row/lane map
above. There is no wave-M offset and no `0x4000` M-quadrant offset.

The mapping has complete static coverage:

```text
fragment_m + lane_row
    = {0,16,32,48} + [0,15]
    = every row in [0,63] exactly once per fragment-N block

fragment_n + 16*store_group + 8*lane_n_half + [0,7]
    = every wave-local column in [0,63] exactly once

wave_n_origin + wave-local column
    = disjoint WG columns [0,63], [64,127], [128,191], [192,255]

covered elements = 64 rows * 256 columns = 16384
covered bytes    = 16384 * 2 = 32768 = 0x8000
```

The row sets, fragment-N sets, store groups, lane halves, and wave-N origins
are pairwise disjoint at each nesting level. Therefore the candidate has
16384 unique element addresses with no holes or collisions. A tagged-lane
runtime test must still confirm that the physical WMMA lane distribution
matches the assumed migration mapping.

An independent temporary Python enumeration expanded every wave, fragment,
store group, lane, and eight-element `ds_store_b128` span. It produced:

```text
elements=16384, collisions=0, holes=0, extras=0
minimum byte address=0x2A800
maximum byte address exclusive=0x32800
```

The output region is deliberately separate from the input ring:

```text
input ring [0x00000,0x2A800)      0x2A800 = 170 KiB
output staging [0x2A800,0x32800)  0x08000 =  32 KiB
fixed LDS end                     0x32800 = 202 KiB
```

This 202-KiB total is the candidate fixed LDS requirement. There is no
explicit layout gap: B starts at the SB end `0x0A800`, and output staging
starts at the B end `0x2A800`. The separate, immediately adjacent output
region makes the first implementation easier to prove and prevents a late
input load or speculative TDM fill from overlapping output staging. Reusing a
drained input-ring slot is a future optimization and is permitted only after
a complete DS/TDM lifetime and barrier proof.

<a id="section-5-4-tdm-store-sync"></a>

### 5.4 Required TDM-Store Contract and Synchronization

**Hardware facts.** CDNA5 ISA L10147-L10156 states that tensor operations
ignore EXEC, increment the issuing wave's TENSORcnt, complete in order within
that wave, and are unordered with tensor operations from other waves.
Descriptor `tile_dim0` and `tile_dim1` are 16-bit fields in data-size units
(CDNA5 ISA L10487-L10503), and `tensor_dim0_stride` is in data-size elements
(L10505-L10514). For a store, `workgroup_mask` is ignored (L10276-L10279);
the candidate descriptor sets it to zero so that the descriptor
unambiguously requests no multicast behavior.

**256x256 reference-ISA facts.** The reference descriptor construction at
L6389-L6434 uses byte-oriented dimensions, `tile_dim1=64`, and a global row
stride supplied through the descriptor. Its stores at L6622 and L6809 are
64-row `tensor_store_from_lds` operations. This is direct evidence for the
required 64-row form; it does not prove this target's `0x200` row width,
packed descriptor bits, LDS address, or SGPR allocation.

The design requires exactly one 64-row by 256-BF16 store for the complete
`64x256` WG tile, issued only by logical wave0:

| Descriptor property | Required semantic value |
| --- | --- |
| Issuer | logical wave0 only, selected by a scalar wave-ID branch |
| Global tile base | `ptr_D + (wg_m_origin * N + wg_n_origin) * 2` |
| LDS tile base | `0x2A800` |
| `count` | exactly one valid descriptor and one store issue |
| `workgroup_mask` | `0` |
| `data_size` | byte mode, 1 byte per descriptor element |
| `tensor_dim0` | `0x200` bytes for the tile's row subrange |
| `tensor_dim1` | `64` rows |
| `tile_dim0` | `0x200` byte-mode elements, equal to 512 row bytes |
| `tile_dim1` | `64` rows |
| `tensor_dim0_stride` | `N * 2` bytes (`0x1000` for `N=2048`) |
| Descriptor padding, gather, iteration, atomic arrival | disabled |

Byte mode makes `tile_dim0=0x200` equal to the required `256*2=512` row
bytes. An element-mode descriptor with two-byte elements and `tile_dim0=256`
is semantically equivalent, but it is not the primary candidate. The packed
descriptor words, semantic-size encoding convention, SGPR assignments,
reserved bits, and accepted LDS/global alignments remain target-assembly and
hardware validation boundaries.

The exact WG synchronization skeleton is:

```text
# All four waves have issued their N-quarter DS writes.
all waves:
    s_wait_dscnt 0
    s_barrier_signal -1
    s_barrier_wait 0xffff       # complete 64x256 staging tile is visible

if logical_wave_id == 0:        # scalar control-flow branch
    tensor_store_from_lds candidate_64_row_descriptor  # exactly once per WG
    s_wait_tensorcnt 0

all waves:
    s_barrier_signal -1
    s_barrier_wait 0xffff       # wave0 has completed the output TDM
```

TENSORcnt is per wave, so the three non-issuing waves cannot wait on wave0's
counter. The second WG barrier is required to carry wave0's observed
completion to those waves before any wave reuses output LDS or begins a new
persistent task. After this WG rendezvous, the cluster-task transition adds a
cluster barrier as specified in Section 5.5.

<a id="section-5-5-final-drain-transition"></a>

### 5.5 Final Drain, P3 Wrap, and Persistent-Task Transition

The P0/P1/P2 phases continue directly to the next phase. P3 is the four-slot
ring boundary and must use one converged decision for all four WGs in the 1x4
cluster. The following pseudocode is the candidate control protocol:

```text
phase = P0

for each persistent logical cluster task:
    while true:
        run_current_K_body(phase)

        if phase == P0:
            phase = P1
            continue
        if phase == P1:
            phase = P2
            continue
        if phase == P2:
            phase = P3
            continue

        # P3 ring-wrap protocol. Exactly one wave per WG signals.
        if logical_wave_id == 0:
            s_barrier_signal -3

        if another_K_body_exists:
            all waves: s_barrier_wait 0xfffd
            phase = P0
            continue

        # Final K path. If P3 already signaled, it must also complete the
        # matching cluster wait; the final path may not abandon that protocol.
        suppress_future_prefetch_with_scalar_branch_or_null_descriptor()
        all waves: s_barrier_wait 0xfffd

        # Drain every class that can touch operands, the input ring, or C/D.
        all waves:
            s_wait_dscnt 0
            s_wait_tensorcnt 0
            s_wait_alu depctr_va_vdst(0)

        convert_pack_and_stage_64x64_N_quarter()
        run_WG_output_store_protocol_from_Section_5_4()

        # No WG may advance while another WG in the physical cluster is still
        # using current-task state or output staging.
        if logical_wave_id == 0:
            s_barrier_signal -3
        all waves: s_barrier_wait 0xfffd

        next_cluster_task = current_cluster_task + persistent_stride
        if next_cluster_task >= logical_cluster_task_count:
            terminate_all_WGs_in_this_physical_cluster()
            return                         # converged exit; no fall-through

        # Reinitialize only after the completed output-store and cluster waits.
        clear_f32_accumulators(v256:v383)
        reset_K_index_and_phase_to_P0()
        rebuild_A_B_SA_SB_and_output_tile_pointers(next_cluster_task)
        rebuild_TDM_descriptors_and_lane_LDS_addresses()
        reset_input_ring_ownership_state()
        current_cluster_task = next_cluster_task
        break
```

For the target shape's N-fastest logical task order, pointer reconstruction
after that barrier uses the following static coordinate derivation:

```text
cluster_n = next_cluster_task % 2
cluster_m = next_cluster_task / 2
Mtile     = cluster_m
Ntile     = 4*cluster_n + wg_x

wg_m_origin = 64  * Mtile
wg_n_origin = 256 * Ntile
wave_m_origin = wg_m_origin
wave_n_origin = wg_n_origin + 64*logical_wave_id
```

The A/SA pointers depend on `wg_m_origin`, B/SB depend on `wg_n_origin`, and
D depends on both origins. All TDM descriptors must be rebuilt from those new
coordinates. These formulas are part of the candidate launcher/swizzle
contract, not assembled-ISA facts.

The future prefetch decision must be effective before a real TDM issue. Tensor
instructions ignore EXEC, so an EXEC-masked instruction is not suppressed.
A scalar branch is sufficient. A descriptor with `tile_dim0=0` is an
alternative only if target validation confirms that it performs no transfer;
it still has to complete its tensor-counter protocol and must have atomic
arrival disabled.

All four WGs derive `another_K_body_exists`, `next_cluster_task`, and task
termination from the same cluster-task state. Exactly one wave in every WG
signals each cluster barrier, and all four waves in every WG wait. Thus each
cluster-barrier generation has four WG signals and 16 wave waits. No WG may
independently skip a signal or wait. For the 576-task logical cluster grid and
64 physical clusters, every physical cluster executes nine tasks with stride
64, but the uniform termination predicate is still required for every
supported shape.

Accumulator clearing and pointer/descriptor initialization may be optimized
or overlapped later. The correctness-first candidate serializes them after
the second WG barrier and the task-boundary cluster barrier. There must be no
outstanding output TDM before output staging or any overlapping input LDS is
reused for the next persistent task.

<a id="section-5-6-resource-validation"></a>

### 5.6 Resource and Validation Checklist

The candidate resource summary is:

| Resource | Candidate requirement | Boundary |
| --- | ---: | --- |
| F32 accumulators per wave | `8 * 16 = 128 VGPRs`, `v256:v383` | Unchanged because each wave remains `64x64` |
| Packed BF16 staging per wave | `8 * 8 = 64 VGPRs`, `v128:v191` | Unchanged; reuse only after input lifetimes end |
| Candidate minimum next-free VGPR boundary | `>= 384` | Not metadata; target assembly may raise it |
| Input LDS ring | `0x2A800 = 170 KiB` | Four contiguous payload arrays; no explicit layout gap |
| Output LDS staging | `0x8000 = 32 KiB` | Separate `[0x2A800,0x32800)` region |
| Candidate fixed LDS total | `0x32800 = 202 KiB` | Must be emitted and accepted as group-segment size |
| Output TDM | one `64x256` BF16 store per WG | `tile_dim1=64`; byte width `0x200` |

The final `next_free_vgpr`, group-segment fixed size, SGPR count, descriptor
SGPR placement, and occupancy must come from the assembled gfx1250 object.
They must not be copied from this candidate table into metadata without that
evidence.

Required validation in the intended ROCm gfx1250 container and on target
hardware:

1. **Assembler:** assemble the exact candidate with the target ROCm toolchain.
   Confirm acceptance of gfx1250 opcodes, `s_set_vgpr_msb`, a `0x32800`
   group segment, all four input descriptors, and the one-store output
   descriptor.
2. **Objdump:** disassemble the object and verify every WMMA C/D range,
   conversion source/destination, DS address/data range, wait, barrier, and
   exactly one `tensor_store_from_lds` issue per WG. Confirm that physical
   `v256:v383` and `v128:v191` do not alias after MSB expansion.
3. **Metadata:** inspect `next_free_vgpr`, SGPR use, kernel descriptor fields,
   `.group_segment_fixed_size`, wave32 mode, and any resource-allocation
   granularity. Reject a build whose metadata does not cover every emitted
   physical register and LDS byte.
4. **Input descriptors and ring:** verify semantic byte-mode tiles
   `0x80*64`, `0x08*64`, `0x80*256`, and `0x08*256` for A, SA, B, and SB,
   respectively; all four slot bases; and every wave-local B/SB quarter
   offset. Confirm any required AB-preshuffle regrouping, descriptor encoding,
   target legality of B bases `0x0A800/0x12800/0x1A800/0x22800`, TDM
   destination alignment, no overlap in `[0,0x2A800)`, and acceptable DS bank
   behavior.
5. **Conversion and layout:** run a tagged-fragment microtest that gives every
   fragment, lane half, row, and column a distinguishable value. Verify
   round-to-nearest-even BF16 packing, the hardware N32 x M16 to host M16 x
   N32 transform, N-quarter bases `0x000/0x080/0x100/0x180`, `0x200` row
   stride, and exactly 16384 unique elements with no overlap or holes in
   `[0x2A800,0x32800)`.
6. **Shape-domain gate:** reject any invocation unless `M % 64 == 0`,
   `N % 1024 == 0`, and `K % 256 == 0`. Verify exact 1x4 cluster division and
   the absence of M/N boundary, K-tail, and partial-cluster paths.
7. **Grid mapping:** for `M=18432,N=2048,K=7168`, verify logical WG grid
   `(8,288)`, logical cluster grid `(2,288)`, 576 cluster tasks, physical
   cluster grid `(16,4)`, WG launch `(64,4)`, stride 64, and nine tasks per
   physical cluster.
8. **Numerical comparison:** compare the complete kernel against the trusted
   BF16 reference for the target `M=18432, N=2048, K=7168` shape, including
   constant, random, signed, zero, overflow, NaN, and rounding-boundary cases
   supported by the test harness.
9. **Pipeline and barriers:** verify 40 DS loads and 16 WMMA instructions per
   wave per K256 body. Stress repeated persistent tasks and verify
   P3 continuation, final null/skip behavior, both WG barriers, both cluster
   barrier protocols, four-wave WG convergence, four-WG cluster convergence,
   one cluster signal per WG, all-wave waits, and uniform termination.
   Confirm that no DS or TDM operation accesses a region after reuse.
10. **Output TDM:** verify that exactly one wave0-issued 64-row
   `tensor_store_from_lds` uses LDS base `0x2A800` and writes the complete
   `64x256` tile (32 KiB) with a global row stride of `N * sizeof(BF16)` (4096
   bytes for `N=2048`) and no multicast. Check wave0 TENSORcnt completion
   before the second WG barrier.
11. **Cluster multicast:** verify `nwg_x=4`, `nwg_y=1`, the one-row WG bit
   mapping, A/SA mask `0xf`, B/SB mask `1<<x`, and 16 total shader TDM load
   issues per cluster per K256 body. Measure request combining separately;
   do not infer memory-transaction counts from masks.
12. **Occupancy and performance:** measure achieved occupancy, VGPR/LDS-limited
   residency, DS bank conflicts, TDM throughput, barrier cost, and end-to-end
   kernel performance. Performance results may choose a later input-ring
   reuse optimization only after the correctness proof remains intact.

<a id="section-6-cluster-tdm-multicast"></a>

## 6. Cluster TDM Multicast

No assembled ISA exists for this target. The 128x128 reference ISA and the
256x256 reference ISA are used only as historical evidence for wave
specialization and descriptor-building idioms. Their instruction addresses,
masks, payload sizes, LDS bases, and 4x4 cluster dimensions do not prove this
design. The target geometry below is a static derivation from the 1x4 contract
and the documented CDNA5 cluster/TDM semantics.

<a id="section-6-1-wg-bit-matrix"></a>

### 6.1 WG Bit Matrix

The hardware facts are:

1. CDNA5 ISA Section 2.3 states that a workgroup cluster is scheduled on one
   shader engine, contains at most 16 WGs, and places each member WG on a
   separate WGP (manual pages 9-10; local text L751-L758).
2. CDNA5 ISA Section 2.3.1 defines `WG_in_Cluster` as the logical WG ID and
   defines the TTMP6 fields (manual page 10; local text L769-L815):
   `nwg_y-1` is in bits 19:16, `nwg_x-1` is in bits 15:12, `wg_y` is in
   bits 7:4, and `wg_x` is in bits 3:0.
3. MI400 Shader Programming Guide Section 2.3.1 independently gives the same
   layout (manual page 24; local text L1720-L1766).
4. The same guide's SGPR-initialization description says that SPI creates
   CS waves in typewriter order, with X as the innermost coordinate
   (`for Z, for Y, for X`; local text L3728-L3734).

For this design, x is host N and y is host M. The semantic cluster dimensions
after decoding the TTMP6 dimension-minus-one fields are `nwg_x=4` and
`nwg_y=1`; consequently `wg_y=0` is the only legal y coordinate. With X
innermost, the flattened ID is:

```text
WGinCluster = x + nwg_x * y
            = x + 4 * y
            = x                 # because y=0

               host N / x
             x=0        x=1        x=2        x=3
host M / y
y=0          bit0/WG0   bit1/WG1   bit2/WG2   bit3/WG3
```

Only bits 0 through 3 represent WGs in this cluster; bits 4 through 15 must
remain zero. This one-row matrix is a static derivation from the documented
TTMP6 fields, X-innermost order, and the required 1x4 launch geometry.

The CDNA5 Tensor DMA group-1 descriptor defines bits 15:0 as
`workgroup_mask`, with one bit per cluster WG (Section 10.11.4, manual page
143; local text L10399-L10408). The multicast protocol says that each selected
WG must make the same memory request through one wave in that WG (Section
10.7, local text L9884-L9918). Consequently, a set mask bit selects a
requester and destination WG; it does not elect one WG as a cluster leader.
In this design, every selected requester issues through its operand-specialized
wave.

<a id="section-6-2-operand-multicast-masks"></a>

### 6.2 Operand Multicast Masks

Historical reference ISA provides migration evidence for assigning wave0 to A
data, wave1 to B data, wave2 to SA, and wave3 to SB. Its cluster masks and
line addresses do not apply here. The masks below are rebuilt from the
target's one M row and four N columns.

| Operand | Specialist | Reuse axis | Required mask |
| --- | --- | --- | --- |
| A data | wave0 | same M64 tile across all four N WGs | `0xf` |
| B data | wave1 | no M reuse because `nwg_y=1` | `1 << x` |
| SA | wave2 | same M64 scale tile across all four N WGs | `0xf` |
| SB | wave3 | no M reuse because `nwg_y=1` | `1 << x` |

The unassembled target mask-construction candidate decodes TTMP6 and then
builds:

```text
nwg_x = ((TTMP6 >> 12) & 0xf) + 1 = 4
nwg_y = ((TTMP6 >> 16) & 0xf) + 1 = 1
wg_y  =  (TTMP6 >> 4) & 0xf       = 0
wg_x  =   TTMP6       & 0xf       = x

row_bits = (1 << nwg_x) - 1       = 0xf
A_SA_mask = row_bits << (nwg_x*wg_y)
          = 0xf << 0              = 0xf
B_SB_mask = 1 << (wg_x + nwg_x*wg_y)
          = 1 << x
```

For A and SA, all WGs have the same cluster M coordinate and request the same
M-side operand tile. The constant `0xf` mask selects WG0 through WG3. All
four A requesters, and separately all four SA requesters, must issue the same
request through their specialized wave. The hardware may then combine
matching requests that arrive within its multicast window.

For B and SB, fixed N/x would normally select a cluster column. That column
contains only the requesting WG because `nwg_y=1`:

```text
WGinCluster = x + 4*0 = x
mask = 1 << WGinCluster = 1 << x
```

The complete examples are:

```text
A/SA, any requester x=0..3 -> mask 0xf -> WG0, WG1, WG2, WG3
B/SB, x=0 -> mask 0x1 -> WG0 only
B/SB, x=1 -> mask 0x2 -> WG1 only
B/SB, x=2 -> mask 0x4 -> WG2 only
B/SB, x=3 -> mask 0x8 -> WG3 only
```

The B/SB masks are nonzero and therefore select the documented cluster-load
path, but each has a requester set of size one and a multicast reuse factor of
one. There is no cross-M reuse to claim. The exact descriptor word,
`early_timeout` policy, and persistent-restart reconstruction are target-ISA
validation boundaries.

<a id="section-6-3-payload-cluster-coverage"></a>

### 6.3 Payload and Cluster Coverage

One WG computes a `64x256` output tile for one K256 body. FP4 consumes four
bits per value, and one E8M0 scale byte covers K32, so each WG-local
destination receives:

| Operand | Logical payload derivation | Bytes |
| --- | --- | ---: |
| A data | `M64 * K256 * 4 bits` | `8 KiB = 0x2000` |
| SA | `M64 * (K256 / K32) * 1 byte` | `512 B = 0x0200` |
| B data | `N256 * K256 * 4 bits` | `32 KiB = 0x8000` |
| SB | `N256 * (K256 / K32) * 1 byte` | `2 KiB = 0x0800` |

The CDNA5 descriptor layout places `tile_dim0` in `s39[31:16]` and
`tile_dim1` in `s40[15:0]`; both are in `data_size` units (Section 10.11.4,
local text L10487-L10514). The target semantic row counts and unassembled
byte-mode descriptor candidate are:

| Operand | Rows | Bytes per row | Candidate semantic tile | Payload |
| --- | ---: | ---: | ---: | ---: |
| A data | 64 | `0x80` | `tile_dim0=0x80, tile_dim1=64` | `0x2000` |
| SA | 64 | `0x08` | `tile_dim0=0x08, tile_dim1=64` | `0x0200` |
| B data | 256 | `0x80` | `tile_dim0=0x80, tile_dim1=256` | `0x8000` |
| SB | 256 | `0x08` | `tile_dim0=0x08, tile_dim1=256` | `0x0800` |

The row counts are fixed by target geometry. Mapping these semantic rows onto
the actual AB-preshuffled global layout, including any equivalent regrouping
required by its strides, is a target-codegen validation boundary. No packed
descriptor word is claimed here.

The corresponding non-overlapping four-slot bases are:

| Operand | slot0 | slot1 | slot2 | slot3 | End |
| --- | ---: | ---: | ---: | ---: | ---: |
| A data | `0x00000` | `0x02000` | `0x04000` | `0x06000` | `0x08000` |
| SA | `0x08000` | `0x08200` | `0x08400` | `0x08600` | `0x08800` |
| SB | `0x08800` | `0x09000` | `0x09800` | `0x0A000` | `0x0A800` |
| B data | `0x0A800` | `0x12800` | `0x1A800` | `0x22800` | `0x2A800` |

Within a slot, all waves use the same A/SA base. Wave `w` uses B offset
`w*0x2000` and SB offset `w*0x0200`, selecting its N64 quarter. The payload
arrays are contiguous, contain `0x2A800 = 170 KiB`, and have no explicit
layout gap. B starts at `0x0A800` with a `0x8000` slot stride, and output
occupies `[0x2A800,0x32800)`. B-base legality, TDM destination alignment, and
DS bank behavior remain target-ISA validation boundaries.

The 1x4 logical cluster contains four WGs and covers
`M64xN(4*256) = M64xN1024`. At shader-issue level, one K256 body has:

| Operand class | Requester identity groups | Shader TDM load issues | Static reuse |
| --- | --- | ---: | --- |
| A | one identical group of four WGs | 4 | potentially combine 4-to-1 upstream |
| SA | one identical group of four WGs | 4 | potentially combine 4-to-1 upstream |
| B | four distinct groups of one WG | 4 | reuse factor 1 |
| SB | four distinct groups of one WG | 4 | reuse factor 1 |
| Total | ten identity groups | 16 | not a transaction count |

Equivalently, every WG has four specialized waves and issues four TDM loads
per K256 body, so `4 WGs * 4 issues/WG = 16` shader TDM issues. Each operand
class has four requester instructions before any multicast combining. There
is no cluster leader that removes requester instructions.
Tensor instructions also ignore EXEC (CDNA5 ISA Section 10.11.1, local text
L10147-L10155), so EXEC masking cannot change this count.

A nonzero TDM `workgroup_mask` makes `TENSOR_LOAD_TO_LDS` use
`CLUSTER_LOAD_ASYNC` rather than `GLOBAL_LOAD_ASYNC` (CDNA5 ISA Section
10.11.3, local text L10269-L10279; MI400 Shader Programming Guide Section
4.10.3, manual page 200, local text L14296-L14305). The upstream semantics
are therefore different for the two reuse classes:

```text
A or SA:
    four selected WGs each issue the same specialized-wave request
    -> GL1 may combine matching requests that arrive in time
    -> the returned data is delivered to each requester's WG-local LDS slot

B or SB:
    one selected WG issues each N-specific request
    -> no cross-M requester exists in this 1x4 cluster
    -> reuse factor is one
```

This is multicast into separate WGP-local LDS destinations, not one
cluster-wide LDS allocation. The MI400 guide states that GL1 can merge at
most five requests into one data return (Section 4.9.8, manual page 190,
local text L13779-L13794), so an A/SA group of four is within that documented
limit. It does not prove that a particular dynamic group combines; arrival
timing and timeout policy still matter. Nothing in this static analysis establishes
the GL1-to-GL2 request count, cache-line transaction count, HBM transaction
count, achieved bandwidth, or performance; those require measurement.

No 64x256 target correctness or performance result is available. Any
bitwise-correct result from the historical 128x128 kernel validates only that
historical kernel and does not validate this design.
