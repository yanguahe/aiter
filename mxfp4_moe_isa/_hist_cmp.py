#!/usr/bin/env python3
"""BM=128 gemm1 ISA instruction histogram: FlyDSL vs HIP.

Counts REAL emitted instructions only: strips ';' comments, then takes the
first whitespace-delimited token of each instruction line as the mnemonic
(skips labels, directives, empty lines).
"""
import re
import sys
from collections import Counter

FLY = "flydsl_gemm1_NE385_H7168_E512_BM128.gfx950.s"
HIP = "mxfp4_moe_g1_a4w4_NE385_H7168_E512_BM128.gfx950.s"


def histogram(path):
    c = Counter()
    total = 0
    for raw in open(path):
        line = raw.split(";", 1)[0]          # drop ';' comment
        s = line.strip()
        if not s:
            continue
        if s.startswith("."):                 # directive
            continue
        if s.endswith(":"):                    # label
            continue
        m = re.match(r"([A-Za-z][A-Za-z0-9_]*)", s)
        if not m:
            continue
        mn = m.group(1)
        if mn in ("s_endpgm",):
            pass
        c[mn] += 1
        total += 1
    return c, total


def category(mn):
    if mn.startswith("v_mfma"):
        return "MFMA"
    if mn.startswith("v_accvgpr"):
        return "AGPR_move"
    if mn.startswith("v_cvt"):
        return "CVT/TRANS"
    if mn.startswith("v_pk_"):
        return "VALU_pk"
    if mn.startswith(("v_cmp", "v_cndmask")):
        return "CMP/CNDMASK"
    if mn.startswith("v_"):
        return "VALU_other"
    if mn.startswith("ds_read"):
        return "LDS_read"
    if mn.startswith("ds_write"):
        return "LDS_write"
    if mn.startswith("ds_"):
        return "LDS_other"
    if mn.startswith(("buffer_load", "global_load", "scratch_load", "flat_load")):
        return "VMEM_load"
    if mn.startswith(("buffer_store", "global_store", "scratch_store", "flat_store")):
        return "VMEM_store"
    if mn.startswith("buffer_") or mn.startswith("global_"):
        return "VMEM_other"
    if mn == "s_waitcnt" or mn.startswith("s_waitcnt"):
        return "WAITCNT"
    if mn.startswith("s_barrier"):
        return "BARRIER"
    if mn.startswith("s_nop"):
        return "NOP"
    if mn.startswith("s_setprio"):
        return "SETPRIO"
    if mn.startswith(("s_branch", "s_cbranch")):
        return "BRANCH"
    if mn.startswith("s_"):
        return "SALU"
    return "OTHER"


fc, ft = histogram(FLY)
hc, ht = histogram(HIP)

# --- category totals ---
cats = {}
for c, lbl in ((fc, "fly"), (hc, "hip")):
    for mn, n in c.items():
        cat = category(mn)
        cats.setdefault(cat, {"fly": 0, "hip": 0})[lbl] += n

order = ["MFMA", "AGPR_move", "CVT/TRANS", "VALU_pk", "CMP/CNDMASK",
         "VALU_other", "LDS_read", "LDS_write", "LDS_other",
         "VMEM_load", "VMEM_store", "VMEM_other",
         "WAITCNT", "BARRIER", "NOP", "SETPRIO", "BRANCH", "SALU", "OTHER"]
seen = set(cats)
order = [c for c in order if c in seen] + sorted(seen - set(order))

print("=" * 60)
print("BM=128 gemm1 ISA category histogram  (FlyDSL vs HIP)")
print("=" * 60)
print(f"{'category':<14}{'FlyDSL':>9}{'HIP':>9}{'F-H':>8}{'F/H':>8}")
print("-" * 60)
sf = sh = 0
for cat in order:
    f = cats[cat]["fly"]
    h = cats[cat]["hip"]
    sf += f
    sh += h
    r = (f / h) if h else float("inf")
    print(f"{cat:<14}{f:>9}{h:>9}{f-h:>+8}{r:>8.2f}")
print("-" * 60)
print(f"{'TOTAL':<14}{sf:>9}{sh:>9}{sf-sh:>+8}{(sf/sh):>8.2f}")

# --- key individual mnemonics ---
keys = ["v_mfma_scale_f32_16x16x128_f8f6f4",
        "v_accvgpr_mov_b32", "v_accvgpr_write_b32", "v_accvgpr_read_b32",
        "buffer_load_dwordx4", "buffer_load_dwordx2", "buffer_load_dword",
        "ds_read_b128", "ds_read_b64", "ds_read_b32",
        "ds_write_b128", "ds_write_b64", "ds_write_b32",
        "v_cvt_scalef32_pk_fp4_f32", "v_pk_mul_f32", "v_pk_add_f32",
        "v_mul_f32_e32", "v_max3_f32", "v_perm_b32",
        "s_waitcnt", "s_barrier", "s_nop", "s_setprio",
        "buffer_load_lds_dword", "buffer_load_lds_dwordx4"]
print("\n" + "=" * 60)
print("key individual mnemonics")
print("=" * 60)
print(f"{'mnemonic':<34}{'FlyDSL':>8}{'HIP':>8}")
print("-" * 60)
for k in keys:
    f = fc.get(k, 0)
    h = hc.get(k, 0)
    if f or h:
        print(f"{k:<34}{f:>8}{h:>8}")

# --- mnemonics only in one side / biggest deltas ---
allm = set(fc) | set(hc)
deltas = sorted(allm, key=lambda m: abs(fc.get(m, 0) - hc.get(m, 0)), reverse=True)
print("\n" + "=" * 60)
print("top-20 by |FlyDSL-HIP| delta")
print("=" * 60)
print(f"{'mnemonic':<34}{'FlyDSL':>8}{'HIP':>8}{'F-H':>8}")
print("-" * 60)
for m in deltas[:20]:
    f = fc.get(m, 0)
    h = hc.get(m, 0)
    print(f"{m:<34}{f:>8}{h:>8}{f-h:>+8}")
print(f"\nFlyDSL total instrs: {ft}   HIP total instrs: {ht}")
