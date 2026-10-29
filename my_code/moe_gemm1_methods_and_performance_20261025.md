# F3 接手后尝试的方法与 GEMM1 耗时（2026-10-25）

本清单覆盖 2026-10-24 从 `hyg_dsr1_conc6144 / 6f656cf5b0 / F3` 接手后的实验，
不把更早 E65/shared1、旧分支或另一 `--scenario bench` 口径混入本次对比。
这里只汇总 GEMM1 μs；编译、审计、ATT wave cycles 和 controller 时间不冒充 kernel 性能。
各项都来自已经保留的原性能表或本地独立 review，没有追加 GPU 测试。

当前采用 v45、pin=0。正式三轮 80.9644 /81.3186 /118.372 μs，中值 **81.3186 μs**；
118.372 μs 异常轮未删除。稳定性补测中值 81.1414 μs 是独立批次，不覆盖正式结果。
v3 历史正向 78.2511 μs 仍保留为候选成绩；其波动原因正在定位，未采用。

## pin=0 的实际测量

统一 workload：E64/T1536/topk8/K7168/inter2048、A4W4/SiLU、random non-balanced，
`uniform/auto/seed0`、A-preshuffle=1、warmup5/iters20、原 profiler timer。
前侧和后侧为同一次固定 `reference → candidate → reference` 中各三轮的中值。
不同批次的数值不作跨批排名；失败组仍在原记录中，不按快慢丢弃。
起始 F3/pin0 正式复测是 82.0483 μs。
表中的 v 编号有不同 family，因此保留 `tail pin0`、`WG pin0` 前缀来区分。

| 方法 | 前侧 reference μs | 候选 GEMM1 μs | 后侧 reference μs | 决策 | 证据 |
|---|---:|---:|---:|---|---|
| tail pin0/v5：loader 两端 compiler boundary＋wait0 | 82.1559 | **85.9257** | 81.4261 | 双侧回退 | [f3_pin0_loader_boundary_comparison_local_review_20261025_v5.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_pin0_loader_boundary_comparison_local_review_20261025_v5.json) |
| tail pin0/v6：保留 boundary，删除新增 wait0 | 82.2021 | **86.0668** | 81.9764 | 双侧回退 | [f3_pin0_loader_boundary_nowait_comparison_local_review_20261025_v6.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_pin0_loader_boundary_nowait_comparison_local_review_20261025_v6.json) |
| WG pin0/v2：所有行覆盖 body 加 loader boundary | 81.6664 | **81.7023** | 81.9027 | 收益不稳定 | [f3_m16_wg_uniform_pin0_boundary_comparison_local_review_20261025_v2.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_m16_wg_uniform_pin0_boundary_comparison_local_review_20261025_v2.json) |
| boundary8/v3：仅八行 body 加 loader boundary | 81.6391 | **78.2511** | 81.8478 | 正向改善；反向波动，未采用 | [f3_m16_wg_uniform_pin0_boundary8_comparison_local_review_20261025_v3.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_m16_wg_uniform_pin0_boundary8_comparison_local_review_20261025_v3.json) |
| boundary8/v5：仅保留 loader 出口 boundary | 78.3900 | **82.8137** | 79.1363 | 双侧回退 | [f3_m16_wg_uniform_pin0_boundary8_exit_comparison_local_review_20261025_v5.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_m16_wg_uniform_pin0_boundary8_exit_comparison_local_review_20261025_v5.json) |
| v8：按行完成 activation | 79.1935 | **79.3305** | 78.0759 | 双侧回退 | [f3_boundary8_row_activation_comparison_local_review_20261025_v8.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_boundary8_row_activation_comparison_local_review_20261025_v8.json) |
| v10：WMMA column-order＋六行 boundary | 83.1911 | **85.9487** | 78.4616 | 双侧回退 | [f3_boundary8_column_order_boundary6_comparison_local_review_20261025_v10.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_boundary8_column_order_boundary6_comparison_local_review_20261025_v10.json) |
| v11：full-row column reuse | 78.5968 | **80.2032** | 79.2966 | 双侧回退 | [f3_boundary8_column_fused_rows_comparison_local_review_20261025_v11.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_boundary8_column_fused_rows_comparison_local_review_20261025_v11.json) |
| v14：mma_group=4 | 79.7521 | **85.6916** | 78.4934 | 双侧回退 | [f3_boundary8_pin0_mg4_comparison_local_review_20261025_v14.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_boundary8_pin0_mg4_comparison_local_review_20261025_v14.json) |
| v16：row-pair WMMA＋六行 boundary | 79.9208 | **87.0323** | 78.4638 | 双侧回退 | [f3_boundary8_row_pair_boundary6_comparison_local_review_20261025_v16.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_boundary8_row_pair_boundary6_comparison_local_review_20261025_v16.json) |
| v17：closing20 | 84.0491 | **81.8942** | 78.9172 | 收益不稳定 | [f3_boundary8_pin0_closing20_comparison_local_review_20261025_v17.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_boundary8_pin0_closing20_comparison_local_review_20261025_v17.json) |
| v18：closing24 | 78.9067 | **81.6708** | 82.8641 | 收益不稳定 | [f3_boundary8_pin0_closing24_comparison_local_review_20261025_v18.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_boundary8_pin0_closing24_comparison_local_review_20261025_v18.json) |
| v20：first DS=2 | 78.9107 | **79.3156** | 78.4871 | 双侧回退 | [f3_boundary8_pin0_dsfirst2_comparison_local_review_20261025_v20.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_boundary8_pin0_dsfirst2_comparison_local_review_20261025_v20.json) |
| v22：first DS=3 | 78.5714 | **79.9904** | 83.3576 | 收益不稳定 | [f3_boundary8_pin0_dsfirst3_comparison_local_review_20261025_v22.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_boundary8_pin0_dsfirst3_comparison_local_review_20261025_v22.json) |
| v24：仅七行 first DS=0 | 78.3690 | **80.9729** | 78.6073 | 双侧回退 | [f3_boundary8_pin0_dsfirst0_seven_comparison_local_review_20261025_v24.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_boundary8_pin0_dsfirst0_seven_comparison_local_review_20261025_v24.json) |
| v25：七/八行 boundary＋closing16 | 78.3288 | **85.0695** | 78.8095 | 双侧回退 | [f3_boundary7_8_pin0_closing16_comparison_local_review_20261025_v25.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_boundary7_8_pin0_closing16_comparison_local_review_20261025_v25.json) |
| v26：仅六行 closing16 | 79.3958 | **78.1519** | 78.0022 | 完整重测收益不稳定；初次异常保留 | [f3_boundary8_pin0_closing16_six_comparison_local_review_20261025_v26_v3.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_boundary8_pin0_closing16_six_comparison_local_review_20261025_v26_v3.json) |
| v28：仅六行两次 iteration 展开 | 89.2147 | **93.9524** | 79.1722 | 双侧回退 | [f3_boundary8_pin0_pair_six_comparison_local_review_20261025_v28.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_boundary8_pin0_pair_six_comparison_local_review_20261025_v28.json) |
| v30：限定 K7168 的 instruction prefetch | 78.3183 | **89.2589** | 79.1279 | 双侧回退 | [f3_boundary8_pin0_instruction_prefetch_k7168_comparison_local_review_20261025_v30.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_boundary8_pin0_instruction_prefetch_k7168_comparison_local_review_20261025_v30.json) |
| v31：first_group=2，steady_group=2 | 84.3422 | **84.1882** | 78.8096 | 收益不稳定 | [f3_boundary8_pin0_first2_comparison_local_review_20261025_v31.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_boundary8_pin0_first2_comparison_local_review_20261025_v31.json) |
| v32：grouped LDS，A-first | 78.5925 | **82.3223** | 90.3090 | 收益不稳定 | [f3_boundary8_pin0_grouped_lds_comparison_local_review_20261025_v32.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_boundary8_pin0_grouped_lds_comparison_local_review_20261025_v32.json) |
| v33：grouped LDS，B-first | 81.8795 | **86.3177** | 81.7151 | 双侧回退 | [f3_boundary8_pin0_grouped_lds_bfirst_vs_v2_comparison_local_review_20261025_v33.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_boundary8_pin0_grouped_lds_bfirst_vs_v2_comparison_local_review_20261025_v33.json) |
| v35：缩小行覆盖 body 集合到 4/6/8 | 81.7298 | **85.6684** | 82.0524 | 双侧回退 | [f3_pin0_coverage468_vs_v2_comparison_local_review_20261025_v35.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_pin0_coverage468_vs_v2_comparison_local_review_20261025_v35.json) |
| v36：单八行 mainloop，WG-uniform 无效行 WMMA guard | 81.6961 | **84.7049** | 81.9090 | 双侧回退 | [f3_pin0_runtime_wmma_skip_vs_v2_comparison_local_review_20261025_v36.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_pin0_runtime_wmma_skip_vs_v2_comparison_local_review_20261025_v36.json) |
| v37：细化最后一行 WMMA guard | 81.5316 | **84.1018** | 82.2379 | 双侧回退 | [f3_pin0_runtime_wmma_skip_vs_v2_comparison_local_review_20261025_v37.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_pin0_runtime_wmma_skip_vs_v2_comparison_local_review_20261025_v37.json) |
| v39：ScaleA 两 byte 合并 i16 store＋loader boundaries | 81.5969 | **87.2771** | 82.2905 | 双侧回退 | [f3_packed_scale16_boundaries_pin0_vs_v2_comparison_local_review_20261025_v39.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_packed_scale16_boundaries_pin0_vs_v2_comparison_local_review_20261025_v39.json) |
| v40：ScaleA i16 store＋terminal tile boundaries | 81.8921 | **80.9223** | 82.6049 | 正向改善；反向回退，未采用 | [f3_packed_scale16_drain_boundaries_pin0_vs_v2_comparison_local_review_20261025_v40_v2.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_packed_scale16_drain_boundaries_pin0_vs_v2_comparison_local_review_20261025_v40_v2.json) |
| v42：FP4 payload stores 合并＋terminal boundaries | 81.4808 | **86.8582** | 81.9432 | 双侧回退 | [f3_fp4_payload64_drain_boundaries_pin0_vs_v2_comparison_local_review_20261025_v42_v1.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_fp4_payload64_drain_boundaries_pin0_vs_v2_comparison_local_review_20261025_v42_v1.json) |
| v43：payload 拼接用 V_PERM_B32 | 81.6584 | **87.1718** | 81.4093 | 双侧回退 | [f3_fp4_payload64_permute_pin0_vs_v2_comparison_local_review_20261025_v43_v1.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_fp4_payload64_permute_pin0_vs_v2_comparison_local_review_20261025_v43_v1.json) |
| v44：v43＋等价 dense LDS i32 地址 | 81.8900 | **84.0656** | 81.6977 | 双侧回退 | [f3_fp4_payload64_permute_addr32_pin0_vs_v2_comparison_local_review_20261025_v44_v1.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_fp4_payload64_permute_addr32_pin0_vs_v2_comparison_local_review_20261025_v44_v1.json) |
| v45：dense LDS i32 地址＋terminal boundaries，原 halfword stores | 82.1239 | **81.6012** | 81.8541 | 正反向四侧均改善；生产采用 | [f3_fp4_payload_addr32_halfword_pin0_vs_v2_comparison_local_review_20261025_v45_v1.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_fp4_payload_addr32_halfword_pin0_vs_v2_comparison_local_review_20261025_v45_v1.json) |
| v49：末尾 M-rep pair guard＋ScaleA wait0 | 81.3819 | **84.6817** | 81.2679 | 双侧回退 | [f3_v49_vs_v45_comparison_independent_local_review_20261025_v1.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_v49_vs_v45_comparison_independent_local_review_20261025_v1.json) |
| v50：v49 的新增 wait0 收窄为 wait8 | 80.6080 | **84.1523** | 81.1437 | 双侧回退 | [f3_v50_vs_v45_comparison_independent_local_review_20261025_v1.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_v50_vs_v45_comparison_independent_local_review_20261025_v1.json) |
| v52：仅对有效 M-row materialize C fragments | 81.3843 | **80.9164** | 81.4664 | 正向改善；反向收益不稳定，未采用 | [f3_v52_vs_v45_comparison_independent_local_review_20261025_v1.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_v52_vs_v45_comparison_independent_local_review_20261025_v1.json) |
| v55：提前 i32 LDS stage 地址，OR 合并 offset | 81.2217 | **84.5993** | 81.0786 | 双侧回退 | [f3_v55_vs_v45_comparison_independent_local_review_20261025_v1.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_v55_vs_v45_comparison_independent_local_review_20261025_v1.json) |
| v56：提前 i32 LDS stage 地址，改用加法 | 81.5654 | **84.3713** | 81.4111 | 双侧回退 | [f3_v56_vs_v45_comparison_independent_local_review_20261025_v1.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_v56_vs_v45_comparison_independent_local_review_20261025_v1.json) |
| v58：literal stage tied SGPR identity | 80.8361 | **82.2234** | 81.3590 | 双侧回退 | [f3_v58_vs_v45_comparison_independent_local_review_20261025_v1.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_v58_vs_v45_comparison_independent_local_review_20261025_v1.json) |

反向固定 `candidate → reference → candidate` 的结果如下。中间列是 reference，
不能把原 generic JSON 的 `candidate_us` 当成反向外侧候选成绩。

| 候选 variant | 前侧候选 μs | 中间 reference μs | 后侧候选 μs | 证据 |
|---|---:|---:|---:|---|
| wg_uniform_boundary8 | 78.7001 | 81.7023 | 89.9738 | [f3_boundary8_pin0_reverse_reference_local_review_20261025_v1.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_boundary8_pin0_reverse_reference_local_review_20261025_v1.json) |
| packed_scale16_drain_boundaries_pin0 | 81.5166 | 81.3122 | 81.7149 | [f3_packed_scale16_drain_boundaries_pin0_reverse_reference_local_review_20261025_v40_v1.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_packed_scale16_drain_boundaries_pin0_reverse_reference_local_review_20261025_v40_v1.json) |
| fp4_payload_addr32_halfword_pin0 | 80.6123 | 81.5717 | 81.1646 | [f3_fp4_payload_addr32_halfword_pin0_reverse_reference_local_review_20261025_v45_v1.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_fp4_payload_addr32_halfword_pin0_reverse_reference_local_review_20261025_v45_v1.json) |
| lazy_quant_accs_pin0 | 81.5042 | 80.9394 | 80.9183 | [f3_v52_reverse_reference_independent_local_review_20261025_v1.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_v52_reverse_reference_independent_local_review_20261025_v1.json) |

v59 正向只完成前侧 reference **81.7719 μs** 和候选
**80.1972 μs**；后侧 reference 第一组 ownership 失败，第二组第三轮
原 v45 baseline child 返回 134 / illegal memory access。整次比较 **未完成、不能采用**，无反向成绩。
异步报错不能直接归因于 v59 或 v3；失败记录见 [f3_v59_incomplete_comparison_independent_local_review_20261025_v1.json](.codex_tmp/hyg_f3_followup_20261024/logs/f3_v59_incomplete_comparison_independent_local_review_20261025_v1.json)。

## pin=1 的历史实验

这些数值只解释历史过程，不作为后续 pin=0 采用依据。各批次 reference 详见原主文档，
不可把表中最低的单个批次中值拿来替换正式成绩。

| 方法 | 历史 GEMM1 μs | 结果 |
|---|---:|---|
| F3 初始正式三轮 | 80.7452 | 接手基准 |
| quant 值 register pin | 82.8976 | 双侧回退 |
| 释放 ScaleA/B 固定地址 register | 81.0087 | 收益不稳定 |
| scaled WMMA reserved field 0x100 | 80.9074 | 收益不稳定 |
| K512/b2 carrywait0 | 93.7732 | 双侧回退 |
| K256/b3 | 86.4085 | 双侧回退 |
| 提前稳态 TDM issue | 90.3234 | 双侧回退 |
| resident ScaleA 两 owner 修正版 | 81.9453 | 双侧回退；首版精度失败 |
| resident ScaleA pinned base | 84.1061 | 双侧回退 |
| resident ScaleA Ktile base 留在 VGPR | 82.3286 | 双侧回退 |
| tail64_outer/v2，正向 | 78.9530 | 正反向均改善；曾采用 |
| tail64_outer/v2，正反向共九轮中值 | 79.2060 | 历史 pin1 成绩 |
| closing8 | 78.9549 | 收益不稳定 |
| ScaleA i16 stores | 78.8835 | 收益不稳定 |
| shared K loop，条件 BACK A load/WMMA | 85.1980 | 双侧回退 |
| outer full-first，正向 | 78.9763 | 反向未稳定，未采用 |
| M16 交错 4/6/8，sawait8/v4，正向 | 79.3874 | 反向未稳定，未采用 |
| M16 4/6/7/8，七行 wait0/v6 | 79.3764 | 双侧回退 |
| WG-uniform 4/6/7/8/v8，正向 | 79.0375 | 后续改为 pin0；未采用此 pin1 版 |

pin1 反向外侧候选中值：full-first 为 78.9932 /79.5624 μs；
M16 sawait8/v4 为 77.7851 /79.4443 μs；WG-uniform/v8 为 78.6179 /79.1512 μs。
它们保留为反向记录，不再重测 pin1。

## 已尝试但没有有效 GEMM1 μs 的方法

“未测”意味着没有合格的设备性能成绩，不以编译秒数、静态指令数或模型结果补一个数字。
native 疑点是原运行前检查的结果，不直接等同于已经观测到硬件或 kernel 功能 bug。

| 方法 | 无有效性能数据的原因 |
|---|---|
| pin1 两次 iteration pair | 48 个原 bounded DS/WMMA 疑点，正式性能未 dispatch |
| pin1 resident ScaleA 首版 | 设备精度失败，后改为两 owner；没有有效性能成绩 |
| pin1 M16/v1、padsa/v2、pinscale/v3 | v1/v3 native 未通过；v2 完整 ELF 与 v1 相同 |
| pin1 七行 wait12/v5、closing24+wait8/v7 | 原 native gate 未通过，各 153 个疑点 |
| tail pin0/v1：只放开 partition guard | 原 native gate 未通过 |
| tail pin0/v2：loader 入口 wait0 | 原 native gate 未通过 |
| tail pin0/v3：保活当前输入和全部 WMMA 输出 | 编译 child 超时 120 秒，未生成可验收对象 |
| tail pin0/v4：每条 WMMA 一个结果元素保活 | private=3508 B，VGPR spill=20886，未 dispatch |
| WG pin0/v1：4/6/7/8 统一覆盖 | 原 native gate 未通过 |
| boundary8/v4：仅入口 boundary | 原 native gate 未通过 |
| boundary8/v6：七行 wait4 | 原 native gate 未通过 |
| v9：WMMA column-order | 原 native gate 未通过，75 个疑点 |
| v12：六行 aggregate wait 参数规划 | 仅 CPU 模型规划，未形成新的设备成绩 |
| v13：row-pair WMMA | 原 native gate 未通过，75 个疑点 |
| v15：mma_group=1 | 原 native gate 未通过，60 个疑点 |
| v19：first DS=0；v21：first DS=1 | 原 native gate 未通过 |
| v23：closing16 | 原 native gate 未通过 |
| v27：全行两次 iteration 展开 | private=12 B，VGPR spill=6，未 dispatch |
| v29：instruction prefetch 草案 | CPU/native 准备后用限定 K7168 的 v30 实测；无独立性能 |
| v34、v47：生产 ATT | 用于 trace 分析，不能当作新 GEMM1 μs 成绩 |
| v38：ScaleA i16 stores，原布局 | 原 native gate 未通过；疑点也存在于旧生产校准对象 |
| v41：FP4 payload stores 合并 | 原 native gate 未通过 |
| v46：只简化 dense LDS 地址 | 原 native gate 未通过；疑点位置附近与旧生产 ISA 相同 |
| v48：单 mainloop 末尾 pair guard | 原 native gate 未通过，五个 ScaleA retirement 疑点 |
| v51：删除 loader 出口 boundary | 原 native gate 未通过，37 个疑点 |
| v53：每 MX block 读取 C；v54：quant row boundary | 与 v52 完整 ELF 逐字节相同；未重复计时，没有新成绩 |
| v57：untied SGPR 空 asm | 静态 identity 合约不成立；未上传、编译或 dispatch |

复现入口、完整失败记录和各批次原始三轮，见
[F3 主优化记录](moe_gemm1_hyg_dsr1_f3_followup_20261024.md)。
v3 当前的固定进程、clock 和 ownership 诊断另见
[v3 波动定位记录](moe_gemm1_v3_variability_diagnosis_20261025.md)，属于诊断，不加入正式最优排名。
E65 的既有验证仍是 routed t64 fallback；此清单不宣称已完成 fixed shared1 或 E64/E65 共用 compiled GEMM1 的历史要求。
