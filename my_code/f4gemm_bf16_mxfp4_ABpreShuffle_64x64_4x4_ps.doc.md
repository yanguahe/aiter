# gfx1250 MXFP4×MXFP4 128×128_4×4 Persistent GEMM Analysis

<a id="toc"></a>

## Table of Contents

- [1. Wave Tile and TDM Specialization](#section-1-wave-tile)
  - [1.1 Four Waves and 2×2 Output Quadrants](#section-1-1-wave-output-quadrants)
  - [1.2 Four-Slot LDS Ring](#section-1-2-lds-ring)
  - [1.3 Final Kernel Contract](#section-1-3-final-kernel-contract)
- [2. 4×4 Cluster, Logical Tasks, and Persistent Grid](#section-2-cluster-grid)
  - [2.1 One Logical Cluster Task](#section-2-1-logical-cluster-task)
  - [2.2 Logical Grid](#section-2-2-logical-grid)
  - [2.3 Persistent Launch Candidate](#section-2-3-persistent-launch)
- [3. End-to-end software pipeline](#section-3-software-pipeline)
  - [3.1 wave0/2: `B-current → A-current → A-next → B-next`](#section-3-1-wave02-flow)
  - [3.2 wave1/3: `A-current → B-current → B-next → A-next`](#section-3-2-wave13-flow)
- [4. P0 Detailed Pipeline and Host Tile Coverage](#section-4-p0-details)
  - [4.1 wave0/2 P0 detailed pipeline](#section-4-1-wave02-details)
    - [4.1.1 wave0/2 P0 Host Tile Coverage](#section-4-1-1-wave02-host-tile)
  - [4.2 wave1/3 P0 detailed pipeline](#section-4-2-wave13-details)
    - [4.2.1 wave1/3 P0 Host Tile Coverage](#section-4-2-1-wave13-host-tile)
- [5. Epilogue Design and Final Store Contract](#section-5-epilogue-design)
  - [5.1 Output Geometry and Fragment Mapping](#section-5-1-output-geometry)
  - [5.2 Candidate Physical VGPR Layout](#section-5-2-vgpr-layout)
  - [5.3 BF16 Conversion and LDS Staging](#section-5-3-bf16-lds-staging)
  - [5.4 Required TDM-Store Contract and Synchronization](#section-5-4-tdm-store-sync)
  - [5.5 Final Drain, P3 Wrap, and Persistent-Task Transition](#section-5-5-final-drain-transition)
  - [5.6 Resource and Validation Checklist](#section-5-6-resource-validation)

<a id="section-1-wave-tile"></a>

## 1. Wave Tile and TDM Specialization

<a id="section-1-1-wave-output-quadrants"></a>

### 1.1 Four Waves and 2×2 Output Quadrants

| wave | Relative M | Relative N | TDM specialization |
|---:|---|---|---|
| 0 | `[0,63]` | `[0,63]` | host A data |
| 1 | `[64,127]` | `[0,63]` | host B data |
| 2 | `[0,63]` | `[64,127]` | SA |
| 3 | `[64,127]` | `[64,127]` | SB |

<a id="section-1-2-lds-ring"></a>

### 1.2 Four-Slot LDS Ring

The following addresses use a tightly packed layout for the 128×128 WG tile:

| operand | slot0 | slot1 | slot2 | slot3 | Per-slot payload |
|---|---:|---:|---:|---:|---:|
| A data | `0x00000` | `0x04000` | `0x08000` | `0x0C000` | 16 KiB |
| SA | `0x10000` | `0x10400` | `0x10800` | `0x10C00` | 1 KiB |
| SB | `0x11000` | `0x11400` | `0x11800` | `0x11C00` | 1 KiB |
| B data | `0x12000` | `0x16000` | `0x1A000` | `0x1E000` | 16 KiB |

The payload can be calculated independently:

```text
FP4 data: 128 rows * K256 / 2 = 16384 B = 0x4000
E8M0 scale: 128 rows * (256/32) = 1024 B = 0x400
```

The tightly packed four-slot ranges for the four categories span `[0x00000,0x22000)`, totaling `0x22000 = 136 KiB`. This is the address layout for the input ring; the final alignment, bank conflicts, padding, and output staging still require validation against the target ISA.

<a id="section-1-3-final-kernel-contract"></a>

### 1.3 Final Kernel Contract

The following decisions are hard implementation preconditions for the target
kernel. They are not optional extensions or fallback choices.

| Property | Final decision |
| --- | --- |
| Kernel symbol basename | `f4gemm_bf16_mxfp4_ABpreShuffle_128x128_4x4_ps` |
| Tile geometry | Wave `64x64`; WG `128x128`; four waves. |
| M/N preconditions | Each is divisible by `128 * 4 = 512`. |
| K precondition | `K % 256 == 0`; full K256 bodies only. |
| Boundary/tail policy | No M/N boundary tiles and no K tail. |
| Cluster requirement | Logical WG grid divides exactly into `4x4` clusters. |
| Epilogue store | One 128-row `tensor_store_from_lds` per WG. |
| Store failure policy | Stop the rewrite; no two-64-row fallback. |

Accepted inputs therefore satisfy `M % 512 == 0`, `N % 512 == 0`, and
`K % 256 == 0`. The logical WG grid `(N / 128, M / 128)` must divide exactly
into `4x4` clusters; partial clusters and M/N boundary tiles are unsupported.
Only complete K256 bodies are supported, with no K tail.

The required store follows Chapter 5 and uses `tile_dim1=128`. If its
descriptor encoding, assembly, or hardware validation fails, stop the kernel
rewrite. Two 64-row stores are not a supported fallback.

<a id="section-2-cluster-grid"></a>

## 2. 4×4 Cluster, Logical Tasks, and Persistent Grid

<a id="section-2-1-logical-cluster-task"></a>

### 2.1 One Logical Cluster Task

The target kernel uses `wave tile=64×64, WG tile=128×128, cluster=4×4`; x
corresponds to N tiles and y corresponds to M tiles. Therefore:

```text
cluster N = 4 * 128 = 512
cluster M = 4 * 128 = 512
```

Section 1.3 makes this exact cluster divisibility a launch precondition. The
kernel does not construct partial 4×4 clusters or M/N boundary tiles.

One logical cluster task contains 16 WGs, with a complete output region of `512×512`. One K256 body covers `512×512×256`; for `K=7168`, it covers `512×512×7168`.

<a id="section-2-2-logical-grid"></a>

### 2.2 Logical Grid

For the target shape `M=18432,N=2048,K=7168`:

```text
N tiles = 2048 / 128 = 16
M tiles = 18432 / 128 = 144

logical WG grid = (16,144,1)
logical WG tasks = 16 * 144 = 2304

N cluster tasks = 16 / 4 = 4
M cluster tasks = 144 / 4 = 36
logical cluster tasks = 4 * 36 = 144
```

<a id="section-2-3-persistent-launch"></a>

### 2.3 Persistent Launch Candidate

If the original F4 launcher's `PERSISTENT_TG=256 WG` and 4×4 cluster are reused:

```text
cluster_size = 4 * 4 = 16 WG
physical clusters = 256 / 16 = 16
physical cluster grid = (4,4,1)
physical WG launch = (16,16,1)
block = (128,1,1)
```

The candidate uses `log2_grid_x=2,log2_grid_y=2`, so:

```text
persistent stride = 1 << (2 + 2) = 16 cluster tasks
```

Physical cluster `p` processes `p,p+16,p+32,...` until the task ID reaches 144. Each `p=0..15` processes 9 logical cluster tasks, covering a total of `9*256=2304` logical WG tasks over nine rounds.

| Name | Value |
|---|---:|
| logical WG grid | `(16,144,1)` |
| logical WG tasks | 2304 |
| logical cluster tasks | 144 |
| physical cluster grid | `(4,4,1)` |
| physical WG launch | `(16,16,1)` |
| launched WG / cluster | 256 / 16 |
| block | `(128,1,1)` |

The persistent launch above remains a design candidate inherited from the original launcher; the target selector, task swizzle, tile/pointer updates, and epilogue must be implemented by the new kernel and validated against the target ISA.

<a id="section-3-software-pipeline"></a>

## 3. End-to-end software pipeline

<a id="section-3-1-wave02-flow"></a>

### 3.1 wave0/2: `B-current → A-current → A-next → B-next`

```text
Prologue (wave0/2 combined)
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

SA/A-next first half + wmma K0/K1 upper right + WGP barrier      # 2 ds_load_b32 + 8 ds_load_b128 / wave
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

SB/B-next first half + wmma K0 lower left + issue TDM       # 1 ds_load_b32 + 4 ds_load_b128 / wave + 1 TDM / wave
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

SB/B-next first half + wmma K1 lower left           # 1 ds_load_b32 + 4 ds_load_b128 / wave
    wmma10 (2_2)  # K1
    wmma11 (2_3)  # K1
    s_wait_alu depctr_va_vdst(0)
    ds_ld32_bs1 (0_1)                   # next
    ds_ld128_b4 (0_4)                   # next
    ds_ld128_b5 (0_5)                   # next
    ds_ld128_b6 (0_6)                   # next
    ds_ld128_b7 (0_7)                   # next

SB/B-next second half + wmma K0 lower right + finish  # 1 ds_load_b32 + 4 ds_load_b128 / wave
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

SB/B-next second half + wmma K1 lower right           # 1 ds_load_b32 + 4 ds_load_b128 / wave
    wmma14 (3_2)  # K1
    wmma15 (3_3)  # K1
    s_wait_alu depctr_va_vdst(0)
    ds_ld32_bs3 (1_1)                   # next
    ds_ld128_b12 (1_4)                  # next
    ds_ld128_b13 (1_5)                  # next
    ds_ld128_b14 (1_6)                  # next
    ds_ld128_b15 (1_7)                  # next
    loop control branch

P0→P1 boundary
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

SA/A-next first half + wmma K0/K1 upper right + WGP barrier      # 2 ds_load_b32 + 8 ds_load_b128 / wave
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

### 3.2 wave1/3: `A-current → B-current → B-next → A-next`

```text
Prologue (wave1/3 combined)
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

SB/B-next first half + wmma K0/K1 lower left + WGP barrier      # 2 ds_load_b32 + 8 ds_load_b128 / wave
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

SA/A-next first half + wmma K0 upper right + issue TDM       # 1 ds_load_b32 + 4 ds_load_b128 / wave + 1 TDM / wave
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

SA/A-next first half + wmma K1 upper right           # 1 ds_load_b32 + 4 ds_load_b128 / wave
    wmma10 (2_2)  # K1
    wmma11 (2_3)  # K1
    s_wait_alu depctr_va_vdst(0)
    ds_ld32_as1 (0_1)                   # next
    ds_ld128_a4 (0_4)                   # next
    ds_ld128_a5 (0_5)                   # next
    ds_ld128_a6 (0_6)                   # next
    ds_ld128_a7 (0_7)                   # next

SA/A-next second half + wmma K0 lower right + finish  # 1 ds_load_b32 + 4 ds_load_b128 / wave
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

SA/A-next second half + wmma K1 lower right           # 1 ds_load_b32 + 4 ds_load_b128 / wave
    wmma14 (3_2)  # K1
    wmma15 (3_3)  # K1
    s_wait_alu depctr_va_vdst(0)
    ds_ld32_as3 (1_1)                   # next
    ds_ld128_a12 (1_4)                  # next
    ds_ld128_a13 (1_5)                  # next
    ds_ld128_a14 (1_6)                  # next
    ds_ld128_a15 (1_7)                  # next
    loop control branch

P0→P1 boundary
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

SB/B-next first half + wmma K0/K1 lower left + WGP barrier      # 2 ds_load_b32 + 8 ds_load_b128 / wave
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

## 4. P0 Detailed Pipeline and Host Tile Coverage

<a id="section-4-1-wave02-details"></a>

### 4.1 wave0/2 P0 detailed pipeline

<a id="section-4-1-1-wave02-host-tile"></a>

#### 4.1.1 wave0/2 P0 Host Tile Coverage

**A data: one `ds_load_b128 = M16×K64` per cell**

| Host M ↓ / K → | K[0:63] | K[64:127] | K[128:191] | K[192:255] |
|---|---|---|---|---|
| M0 [0:15] | `ds_ld128_a0` (`0_0`) | `ds_ld128_a1` (`0_1`) | `ds_ld128_a4` (`0_4`) | `ds_ld128_a5` (`0_5`) |
| M1 [16:31] | `ds_ld128_a2` (`0_2`) | `ds_ld128_a3` (`0_3`) | `ds_ld128_a6` (`0_6`) | `ds_ld128_a7` (`0_7`) |
| M2 [32:47] | `ds_ld128_a8` (`1_0`) | `ds_ld128_a9` (`1_1`) | `ds_ld128_a12` (`1_4`) | `ds_ld128_a13` (`1_5`) |
| M3 [48:63] | `ds_ld128_a10` (`1_2`) | `ds_ld128_a11` (`1_3`) | `ds_ld128_a14` (`1_6`) | `ds_ld128_a15` (`1_7`) |

**A scale: one `ds_load_b32 = M32×K128 scales` per cell**

| Host M ↓ / K → | K0 [0:127] | K1 [128:255] |
|---|---|---|
| M0 [0:31] | `ds_ld32_as0` (`0_0`) | `ds_ld32_as1` (`0_1`) |
| M1 [32:63] | `ds_ld32_as2` (`1_0`) | `ds_ld32_as3` (`1_1`) |

**B data: one `ds_load_b128 = N16×K64` per cell**

| Host N ↓ / K → | K[0:63] | K[64:127] | K[128:191] | K[192:255] |
|---|---|---|---|---|
| N0 [0:15] | `ds_ld128_b0` (`0_0`) | `ds_ld128_b1` (`0_1`) | `ds_ld128_b4` (`0_4`) | `ds_ld128_b5` (`0_5`) |
| N1 [16:31] | `ds_ld128_b2` (`0_2`) | `ds_ld128_b3` (`0_3`) | `ds_ld128_b6` (`0_6`) | `ds_ld128_b7` (`0_7`) |
| N2 [32:47] | `ds_ld128_b8` (`1_0`) | `ds_ld128_b9` (`1_1`) | `ds_ld128_b12` (`1_4`) | `ds_ld128_b13` (`1_5`) |
| N3 [48:63] | `ds_ld128_b10` (`1_2`) | `ds_ld128_b11` (`1_3`) | `ds_ld128_b14` (`1_6`) | `ds_ld128_b15` (`1_7`) |

**B scale: one `ds_load_b32 = N32×K128 scales` per cell**

| Host N ↓ / K → | K0 [0:127] | K1 [128:255] |
|---|---|---|
| N0 [0:31] | `ds_ld32_bs0` (`0_0`) | `ds_ld32_bs1` (`0_1`) |
| N1 [32:63] | `ds_ld32_bs2` (`1_0`) | `ds_ld32_bs3` (`1_1`) |


**Each cell in the K0 table computes `M16×N32×K128`, with a K range of `[0,127]`:**

| K0 [0,127] / Host M ↓, Host N → | N0 [0,31] | N1 [32,63] |
|---|---:|---:|
| M0 [0,15] | `wmma0` (`0_0`) | `wmma4` (`1_0`) |
| M1 [16,31] | `wmma1` (`0_1`) | `wmma5` (`1_1`) |
| M2 [32,47] | `wmma8` (`2_0`) | `wmma12` (`3_0`) |
| M3 [48,63] | `wmma9` (`2_1`) | `wmma13` (`3_1`) |

**Each cell in the K1 table computes the second K128 accumulation over `[128,255]` for the same M×N output fragment:**

| K1 [128,255] / Host M ↓, Host N → | N0 [0,31] | N1 [32,63] |
|---|---:|---:|
| M0 [0,15] | `wmma2` (`0_2`) | `wmma6` (`1_2`) |
| M1 [16,31] | `wmma3` (`0_3`) | `wmma7` (`1_3`) |
| M2 [32,47] | `wmma10` (`2_2`) | `wmma14` (`3_2`) |
| M3 [48,63] | `wmma11` (`2_3`) | `wmma15` (`3_3`) |

<a id="section-4-2-wave13-details"></a>

### 4.2 wave1/3 P0 detailed pipeline

<a id="section-4-2-1-wave13-host-tile"></a>

#### 4.2.1 wave1/3 P0 Host Tile Coverage

**A data: one `ds_load_b128 = M16×K64` per cell**

| Host M ↓ / K → | K[0:63] | K[64:127] | K[128:191] | K[192:255] |
|---|---|---|---|---|
| M0 [0:15] | `ds_ld128_a0` (`0_0`) | `ds_ld128_a1` (`0_1`) | `ds_ld128_a4` (`0_4`) | `ds_ld128_a5` (`0_5`) |
| M1 [16:31] | `ds_ld128_a2` (`0_2`) | `ds_ld128_a3` (`0_3`) | `ds_ld128_a6` (`0_6`) | `ds_ld128_a7` (`0_7`) |
| M2 [32:47] | `ds_ld128_a8` (`1_0`) | `ds_ld128_a9` (`1_1`) | `ds_ld128_a12` (`1_4`) | `ds_ld128_a13` (`1_5`) |
| M3 [48:63] | `ds_ld128_a10` (`1_2`) | `ds_ld128_a11` (`1_3`) | `ds_ld128_a14` (`1_6`) | `ds_ld128_a15` (`1_7`) |

**A scale: one `ds_load_b32 = M32×K128 scales` per cell**

| Host M ↓ / K → | K0 [0:127] | K1 [128:255] |
|---|---|---|
| M0 [0:31] | `ds_ld32_as0` (`0_0`) | `ds_ld32_as1` (`0_1`) |
| M1 [32:63] | `ds_ld32_as2` (`1_0`) | `ds_ld32_as3` (`1_1`) |

**B data: one `ds_load_b128 = N16×K64` per cell**

| Host N ↓ / K → | K[0:63] | K[64:127] | K[128:191] | K[192:255] |
|---|---|---|---|---|
| N0 [0:15] | `ds_ld128_b0` (`0_0`) | `ds_ld128_b1` (`0_1`) | `ds_ld128_b4` (`0_4`) | `ds_ld128_b5` (`0_5`) |
| N1 [16:31] | `ds_ld128_b2` (`0_2`) | `ds_ld128_b3` (`0_3`) | `ds_ld128_b6` (`0_6`) | `ds_ld128_b7` (`0_7`) |
| N2 [32:47] | `ds_ld128_b8` (`1_0`) | `ds_ld128_b9` (`1_1`) | `ds_ld128_b12` (`1_4`) | `ds_ld128_b13` (`1_5`) |
| N3 [48:63] | `ds_ld128_b10` (`1_2`) | `ds_ld128_b11` (`1_3`) | `ds_ld128_b14` (`1_6`) | `ds_ld128_b15` (`1_7`) |

**B scale: one `ds_load_b32 = N32×K128 scales` per cell**

| Host N ↓ / K → | K0 [0:127] | K1 [128:255] |
|---|---|---|
| N0 [0:31] | `ds_ld32_bs0` (`0_0`) | `ds_ld32_bs1` (`0_1`) |
| N1 [32:63] | `ds_ld32_bs2` (`1_0`) | `ds_ld32_bs3` (`1_1`) |


**Each cell in the K0 table computes `M16×N32×K128`, with a K range of `[0,127]`:**

| K0 [0,127] / Host M ↓, Host N → | N0 [0,31] | N1 [32,63] |
|---|---:|---:|
| M0 [0,15] | `wmma0` (`0_0`) | `wmma8` (`2_0`) |
| M1 [16,31] | `wmma1` (`0_1`) | `wmma9` (`2_1`) |
| M2 [32,47] | `wmma4` (`1_0`) | `wmma12` (`3_0`) |
| M3 [48,63] | `wmma5` (`1_1`) | `wmma13` (`3_1`) |

**Each cell in the K1 table computes the second K128 accumulation over `[128,255]` for the same M×N output fragment:**

| K1 [128,255] / Host M ↓, Host N → | N0 [0,31] | N1 [32,63] |
|---|---:|---:|
| M0 [0,15] | `wmma2` (`0_2`) | `wmma10` (`2_2`) |
| M1 [16,31] | `wmma3` (`0_3`) | `wmma11` (`2_3`) |
| M2 [32,47] | `wmma6` (`1_2`) | `wmma14` (`3_2`) |
| M3 [48,63] | `wmma7` (`1_3`) | `wmma15` (`3_3`) |

<a id="section-5-epilogue-design"></a>

## 5. Epilogue Design and Final Store Contract

This chapter specifies a natural epilogue for the wave-tile `64x64`, WG-tile
`128x128` design. It is design documentation, not assembled gfx1250 ISA.
Section 1.3 makes the one-store policy and its failure gate final even where
descriptor encodings remain validation boundaries. The following evidence
labels are used throughout the chapter:

| Label | Meaning |
|---|---|
| Hardware fact | A behavior stated by the CDNA5 ISA or MI400 Shader Programming Guide. |
| Original-ISA fact | An instruction or data-flow property present in the original 256x256 kernel disassembly. Line references use `my_code/fmha/dump_asm/hsa/gfx1250/f4gemm/f4gemm_bf16_mxfp4_ABpreShuffle_256x256_4x4_ps.s`. |
| Static derivation | Arithmetic or mapping derived from hardware facts, the original ISA, and Chapters 1-4 of this document. |
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

The 64x64 wave tile has four host-M blocks and two host-N blocks, so it has
eight independent C/D fragments. The K0 and K1 instructions shown below both
accumulate into the same named fragment; they are not separate output
fragments. This table repeats the exact Chapter 4 coordinates and pairings.

| Fragment | Host-relative output coordinates | wave0/2 K0 -> K1 | wave1/3 K0 -> K1 |
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

The four existing wave quadrants combine without changing the Chapter 1
mapping:

| Logical wave | WG-relative M | WG-relative N |
|---:|---|---|
| 0 | `[0,63]` | `[0,63]` |
| 1 | `[64,127]` | `[0,63]` |
| 2 | `[0,63]` | `[64,127]` |
| 3 | `[64,127]` | `[64,127]` |

The combined WG output is exactly 128x128 BF16 elements.

<a id="section-5-2-vgpr-layout"></a>

### 5.2 Candidate Physical VGPR Layout

The following is one explicit, internally non-overlapping **candidate choice**.
It is selected for simple addressing and review, not because target metadata
already guarantees it.

| Fragment | Host-relative coordinates | F32 accumulator block | Packed-BF16 staging block |
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
or addresses are used. The original ISA demonstrates this requirement with
`s_set_vgpr_msb` transitions around its conversion and DS sequences, for
example L6522-L6523 and L6587-L6588.

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

**Original-ISA facts.** The original kernel drains broadly with `s_wait_idle`
at L6365 / `0xAB84`, then uses the following instruction families:

- `v_cvt_pk_bf16_f32`: first conversion group at L6458-L6521 /
  `0xAD68-0xAF60`, with additional groups after `s_set_vgpr_msb` changes.
- `ds_store_b128`: first output staging groups at L6588-L6620 /
  `0xB170-0xB26C`.
- `s_wait_dscnt 0` followed by `tensor_store_from_lds`: L6621-L6622 /
  `0xB274-0xB278`, with the second store at L6809 / `0xB7FC`.

CDNA5 ISA L32879-L32889 defines `V_CVT_PK_BF16_F32` as converting two F32
inputs to one packed BF16 dword using round-to-nearest-even. This is the
architecturally appropriate conversion family for a normal BF16 output unless
the target requires a different rounding contract.

The original lane address setup at L145-L148 / `0x1BA4-0x1BBC` and the DS
offset pattern at L6588-L6620 provide the layout principle: use the low four
lane bits as the host-M row, use lane bit 4 to select a host-N subrange, and
store consecutive packed pairs so the hardware N32 x M16 fragment becomes a
host M16 x N32 row-major tile.

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

5. Apply the original lane/layout transpose principle and write each wave's
   64x64 quadrant to the row-major 128x128 WG staging tile.

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

lds_address =
    0x22000
  + (wave_m_offset + fragment_m + lane_row) * 0x100
  + (wave_n_offset + fragment_n
     + 16 * store_group + 8 * lane_n_half) * 2
```

One `ds_store_b128` writes four packed dwords, or eight BF16 values, per lane.
For a fragment, `store_group=0` uses the first four staging VGPRs and
`store_group=1` uses the next four. Lanes 0-15 and 16-31 address the same 16
host-M rows but disjoint eight-column host-N spans. This produces two
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
| One WG output | `128x128x2 = 0x8000` bytes |
| WG output row stride | `128x2 = 0x100` bytes |
| `output_lds_base` | `0x22000` |
| Output staging range | `[0x22000,0x2A000)` |

The wave quadrant origins are:

| Logical wave | Quadrant origin |
|---:|---:|
| 0 | `output_lds_base + 0x0000` |
| 1 | `output_lds_base + 0x4000` |
| 2 | `output_lds_base + 0x0080` |
| 3 | `output_lds_base + 0x4080` |

`0x4000` is 64 rows times the `0x100` row stride, and `0x0080` is 64 BF16
columns. These values are row-major quadrant origins, not contiguous
`0x2000`-byte quadrant payloads. Every quadrant is interleaved at the full WG
row stride, so DS addressing must use the row/lane map above.

The output region is deliberately separate from the input ring:

```text
input ring     [0x00000, 0x22000) = 0x22000 = 136 KiB
output staging [0x22000, 0x2A000) = 0x08000 =  32 KiB
fixed LDS end                           0x2A000 = 168 KiB
```

This 168-KiB total, not 136 KiB, is the candidate fixed LDS requirement. A
separate output region makes the first implementation easier to prove and
prevents a late input load or speculative TDM fill from overlapping output
staging. Reusing a drained input-ring slot is a future optimization and is
permitted only after a complete DS/TDM lifetime and barrier proof.

<a id="section-5-4-tdm-store-sync"></a>

### 5.4 Required TDM-Store Contract and Synchronization

**Hardware facts.** CDNA5 ISA L10147-L10156 states that tensor operations
ignore EXEC, increment the issuing wave's TENSORcnt, complete in order within
that wave, and are unordered with tensor operations from other waves.
Descriptor `tile_dim0` and `tile_dim1` are 16-bit fields in data-size units
(CDNA5 ISA L10487-L10503), and `tensor_dim0_stride` is in data-size elements
(L10505-L10514). For a store, `workgroup_mask` is ignored (L10276-L10279);
the target descriptor still sets it to zero so that the descriptor
unambiguously requests no multicast behavior.

**Original-ISA facts.** The original descriptor construction at L6389-L6434
uses byte-oriented dimensions, a `0x100`-byte in-bounds row, a padded
`0x110`-byte LDS row span, `tile_dim1=64`, and global row stride `s12`.
The stores at L6622 and L6809 move two 64-row pieces. This proves the original
64-row form. It does not prove that 64 is the architectural maximum, and the
two-piece form is historical evidence only, not a supported target mode.

The available CDNA5 ISA and MI400 Shader Guide describe `tile_dim1` as "1 to
Max" but do not publish that maximum. Therefore support for a single 128-row
operation is a validation boundary, not a documented hardware fact.
Nevertheless, the final kernel contract requires exactly one
`tensor_store_from_lds` for the 128x128 BF16 tile, issued only by logical
wave0 of each WG:

| Descriptor property | Required semantic value |
| --- | --- |
| Issuer | logical wave0 only, selected by a scalar wave-ID branch |
| Global tile base | `ptr_D + (wg_m_origin * N + wg_n_origin) * 2` |
| LDS tile base | `0x22000` |
| `count` | exactly one valid descriptor and one store issue |
| `workgroup_mask` | `0` |
| `data_size` | byte mode, 1 byte per descriptor element |
| `tensor_dim0` | `0x100` bytes for the tile's row subrange |
| `tensor_dim1` | `128` rows |
| `tile_dim0` | `0x100` byte-mode elements, or 256 row bytes |
| `tile_dim1` | `128` rows |
| `tensor_dim0_stride` | `N * 2` bytes (`0x1000` for `N=2048`) |
| Padding, gather, iteration, atomic arrival | disabled |

Byte mode follows the original ISA and makes `tile_dim0=0x100` exactly the
row byte count. An element-mode descriptor with `data_size=2` and
`tile_dim0=128` is semantically equivalent, but it is not the primary
candidate here. All packed descriptor words, SGPR assignments, reserved bits,
and encoded units remain candidates until target assembly and disassembly
confirm them; this validation boundary does not permit changing the required
one-store policy.

`tile_dim1=128` is a hard validation gate. If the required descriptor cannot
be encoded, the assembler rejects it, or target hardware validation fails,
stop the kernel rewrite. Do not replace it with two 64-row stores; that
historical original-kernel form is not supported by this target contract.

The exact WG synchronization skeleton is:

```text
# All four waves have issued their quadrant DS writes.
all waves:
    s_wait_dscnt 0
    s_barrier_signal -1
    s_barrier_wait 0xffff       # all 128x128 staging writes are visible

if logical_wave_id == 0:        # scalar control-flow branch
    tensor_store_from_lds required_128_row_descriptor  # exactly once per WG
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
ring boundary and must use one converged decision for all 16 WGs in the 4x4
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

        convert_pack_and_stage_64x64_quadrant()
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

The future prefetch decision must be effective before a real TDM issue. Tensor
instructions ignore EXEC, so an EXEC-masked instruction is not suppressed.
A scalar branch is sufficient. A descriptor with `tile_dim0=0` is also an
architectural NOP candidate, but it still has to complete its tensor-counter
protocol and must have atomic arrival disabled.

All 16 WGs derive `another_K_body_exists`, `next_cluster_task`, and task
termination from the same cluster-task state. Exactly one wave in every WG
signals each cluster barrier, and every wave waits. No WG may independently
skip a signal or wait. For the current 144-task logical cluster grid and 16
physical clusters, every physical cluster executes nine tasks, but the
uniform termination predicate is still required for every supported shape.

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
| F32 accumulators | `8 * 16 = 128 VGPRs`, `v256:v383` | Candidate physical allocation |
| Packed BF16 staging | `8 * 8 = 64 VGPRs`, `v128:v191` | Reuses storage only after input lifetimes end |
| Candidate minimum next-free VGPR boundary | `>= 384` | Not metadata; target assembly may raise it |
| Input LDS ring | `0x22000 = 136 KiB` | Existing Chapters 1-4 design |
| Output LDS staging | `0x8000 = 32 KiB` | Candidate separate region |
| Candidate fixed LDS total | `0x2A000 = 168 KiB` | Must be emitted and accepted as group-segment size |
| Output TDM | one 128-row store per WG | Required; failure stops rewrite |

The final `next_free_vgpr`, group-segment fixed size, SGPR count, descriptor
SGPR placement, and occupancy must come from the assembled gfx1250 object.
They must not be copied from this candidate table into metadata without that
evidence.

Required validation in the intended ROCm gfx1250 container and on target
hardware:

1. **Assembler:** assemble the exact candidate with the target ROCm toolchain.
   Confirm acceptance of gfx1250 opcodes, `s_set_vgpr_msb`, a `0x2A000`
   group segment, and the required one-store descriptor. If the 128-row
   descriptor is not accepted, stop the kernel rewrite.
2. **Objdump:** disassemble the object and verify every WMMA C/D range,
   conversion source/destination, DS address/data range, wait, barrier, and
   exactly one `tensor_store_from_lds` issue per WG. Confirm that physical
   `v256:v383` and `v128:v191` do not alias after MSB expansion.
3. **Metadata:** inspect `next_free_vgpr`, SGPR use, kernel descriptor fields,
   `.group_segment_fixed_size`, wave32 mode, and any resource-allocation
   granularity. Reject a build whose metadata does not cover every emitted
   physical register and LDS byte.
4. **Conversion and layout:** run a tagged-fragment microtest that gives every
   fragment, lane half, row, and column a distinguishable value. Verify
   round-to-nearest-even BF16 packing, the hardware N32 x M16 to host M16 x
   N32 transform, all four quadrant bases, `0x100` row stride, and absence of
   overlap or holes in `[0x22000,0x2A000)`.
5. **Shape-domain gate:** reject any invocation unless `M % 512 == 0`,
   `N % 512 == 0`, and `K % 256 == 0`. Verify that the logical WG grid divides
   exactly into 4x4 clusters and that no M/N boundary or K-tail path exists.
6. **Numerical comparison:** compare the complete kernel against the trusted
   BF16 reference for the target `M=18432, N=2048, K=7168` shape, including
   constant, random, signed, zero, overflow, NaN, and rounding-boundary cases
   supported by the test harness.
7. **Race and barrier checks:** stress repeated persistent tasks and verify
   P3 continuation, final null/skip behavior, both WG barriers, both cluster
   barrier protocols, and uniform termination. Confirm that no DS or TDM
   operation accesses a region after another wave or WG has reused it.
8. **TDM validation:** verify that exactly one wave0-issued 128-row
   `tensor_store_from_lds` writes the complete 128x128 tile (32 KiB) with a
   global row stride of `N * sizeof(BF16)` (4096 bytes for `N=2048`) and no
   multicast. Check TENSORcnt completion before the second WG barrier. If
   descriptor encoding, assembly, or hardware validation fails, stop the
   rewrite; do not split the operation into two 64-row stores.
9. **Occupancy and performance:** measure achieved occupancy, VGPR/LDS-limited
   residency, DS bank conflicts, TDM throughput, barrier cost, and end-to-end
   kernel performance. Performance results may choose a later input-ring
   reuse optimization only after the correctness proof remains intact.
