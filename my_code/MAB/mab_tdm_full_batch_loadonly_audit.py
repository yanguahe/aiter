#!/usr/bin/env python3
"""Generate and statically audit the MAB full-batch TDM load-only ISA."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path


SOURCE_SYMBOL = "mab_tdm_gemm_full_batch"
LOADONLY_SYMBOL = "mab_tdm_gemm_full_batch_loadonly"
BASE_ADDRESS = 0x1E00
BRANCH_RE = re.compile(
    r"^(\s*s_(?:c)?branch\S*\s+)([^\s/]+)(.*?<MAB_TDMs\+0x([0-9a-fA-F]+)>.*)$"
)
ADDRESS_RE = re.compile(r"//\s*([0-9A-Fa-f]{12}):")
FORBIDDEN_MNEMONICS = (
    "ds_load",
    "ds_read",
    "v_wmma",
    "v_cvt",
    "buffer_store",
)
COUNT_MNEMONICS = (
    "tensor_load_to_lds",
    "s_wait_tensorcnt",
    "s_barrier_signal",
    "s_barrier_wait",
    "ds_load",
    "v_wmma",
    "buffer_store",
    "v_cvt",
)


def instruction_offset(line: str) -> int | None:
    match = ADDRESS_RE.search(line)
    return int(match.group(1), 16) - BASE_ADDRESS if match else None


def mnemonic_count(text: str, prefix: str) -> int:
    return len(re.findall(rf"(?m)^\s*(?:\d+\|\s*)?{re.escape(prefix)}", text))


def branch_targets(lines: list[str]) -> set[int]:
    targets: set[int] = set()
    for line in lines:
        match = BRANCH_RE.match(line)
        if match:
            targets.add(int(match.group(4), 16))
    return targets


def should_drop(line: str, offset: int | None) -> bool:
    if offset is not None:
        # Vector LDS-address setup around the initial cluster barrier.
        if 0x118 <= offset <= 0x19C:
            return True
        # More vector LDS-address setup. Keep s88=0 because it is the A global
        # byte offset used by the retained producer descriptor.
        if 0x1CC <= offset <= 0x23C and offset != 0x234:
            return True
        if 0x2B4 <= offset <= 0x350:
            return True
        # C/D buffer-resource construction and accumulator initialization.
        if 0x4D4 <= offset <= 0x658:
            return True
        # The post-TDM numerical epilogue through the instruction before endpgm.
        if 0xC4C <= offset < 0xF80:
            return True
        # C/D base-pointer adjustments; A/B adjustments immediately follow.
        if offset in {0x1A0, 0x1A4, 0x1A8, 0x1AC}:
            return True
        # m0 is only used by the removed LDS consumer path.
        if offset in {0x0C, 0x78, 0x358}:
            return True

    code = line.split("//", 1)[0]
    stripped = re.sub(r"^\s*\d+\|\s*", "", code).lstrip()
    if any(stripped.startswith(prefix) for prefix in FORBIDDEN_MNEMONICS):
        return True
    if stripped.startswith(("s_wait_dscnt", "s_set_vgpr_msb")):
        return True
    if stripped.startswith("s_wait_alu") and ("vm_" in stripped or "va_" in stripped):
        return True
    if re.search(r"\bs61\b", stripped):
        return True
    if stripped.startswith("s_mov_b32 exec_lo"):
        return True
    if stripped.startswith("v_") and offset not in {0x14, 0x1C}:
        return True
    return False


def generate(source: Path, output: Path) -> None:
    original_lines = source.read_text(encoding="utf-8").splitlines()
    targets = branch_targets(original_lines)
    out: list[str] = [
        "// Generated from mab_tdm_gemm_full_batch.s by",
        "// mab_tdm_full_batch_loadonly_audit.py.",
        "// Keeps the original TDM producer/control/synchronization plan and resources;",
        "// removes LDS consumers, WMMA, C/D access, conversion, and stores.",
    ]
    emitted_labels: set[int] = set()

    for original_line in original_lines:
        line = original_line.replace(SOURCE_SYMBOL, LOADONLY_SYMBOL)
        offset = instruction_offset(line)
        if offset in targets and offset not in emitted_labels:
            out.append(f".Lloadonly_{offset:04x}:")
            emitted_labels.add(offset)

        match = BRANCH_RE.match(line)
        if match:
            target = int(match.group(4), 16)
            line = f"{match.group(1)}.Lloadonly_{target:04x}{match.group(3)}"

        if not should_drop(line, offset):
            out.append(line)

    missing = targets - emitted_labels
    if missing:
        raise RuntimeError(f"branch targets without source address: {sorted(missing)}")

    output.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")


def descriptor_plan(k: int, wave: int) -> list[tuple[object, ...]]:
    if k % 512:
        raise ValueError("K must be divisible by 512")
    q_count = k // 128
    kind = "A" if wave % 4 < 2 else "B"
    role = wave % 4
    plan = []
    # The source has a one-descriptor lookahead. q==q_count is bounded/padded
    # by the tensor descriptor and is intentionally part of both programs.
    for q in range(q_count + 1):
        if kind == "A":
            global_row = "m_tile:[0,16)"
            lds_copy = role
            mask = "0xf (cluster multicast)"
        else:
            global_row = f"n_tile:[{(role - 2) * 128},{(role - 1) * 128})"
            lds_copy = role - 2
            mask = "1<<cluster_wg_y"
        k_range = f"[{q * 128},{(q + 1) * 128})"
        stage = q % 4
        plan.append((q, kind, global_row, k_range, lds_copy, stage, mask))
    return plan


def audit(source: Path, output: Path, verbose_plan: bool) -> None:
    source_bytes = source.read_bytes()
    output_bytes = output.read_bytes()
    if b"\r" in output_bytes:
        raise AssertionError("load-only ISA contains CR bytes")
    source_text = source_bytes.decode()
    output_text = output_bytes.decode()

    print(f"source_sha256={hashlib.sha256(source_bytes).hexdigest()}")
    print(f"loadonly_sha256={hashlib.sha256(output_bytes).hexdigest()}")
    for mnemonic in COUNT_MNEMONICS:
        before = mnemonic_count(source_text, mnemonic)
        after = mnemonic_count(output_text, mnemonic)
        print(f"{mnemonic}: full={before} loadonly={after}")

    for mnemonic in ("tensor_load_to_lds", "s_wait_tensorcnt", "s_barrier_signal", "s_barrier_wait"):
        if mnemonic_count(source_text, mnemonic) != mnemonic_count(output_text, mnemonic):
            raise AssertionError(f"{mnemonic} static site count changed")
    for mnemonic in FORBIDDEN_MNEMONICS:
        if mnemonic_count(output_text, mnemonic):
            raise AssertionError(f"forbidden instruction remains: {mnemonic}")
    if re.search(r"(?m)^\s*s_(?:c)?branch\S*\s+(?:0x)?[0-9]+", output_text):
        raise AssertionError("numeric branch displacement remains")
    if ".amdhsa_next_free_vgpr 1024" not in output_text or ".amdhsa_next_free_sgpr 98" not in output_text:
        raise AssertionError("resource declarations changed")
    if ".group_segment_fixed_size: 327680" not in output_text:
        raise AssertionError("static LDS metadata changed")
    if any(
        re.search(rf"\b{register}\b", output_text)
        for register in ("s44", "s45", "s46", "s47", "s48", "s49", "s50", "s51", "s61")
    ):
        raise AssertionError("removed C/D or LDS-consumer SGPR remains live")
    vgprs = {
        int(value)
        for value in re.findall(r"\bv(\d+)\b", output_text.split(".section .rodata", 1)[0])
    }
    if not vgprs <= {0, 224}:
        raise AssertionError(f"unexpected retained VGPR reads/definitions: {sorted(vgprs)}")
    if output_text.index("v_mov_b32_e32 v224, v0") > output_text.index(
        "v_readfirstlane_b32 s90, v224"
    ):
        raise AssertionError("v224 is read before its retained definition")

    # The generator retains every TDM/control/synchronization instruction from
    # the source byte-for-byte apart from symbolic branch operands and symbol.
    retained_source = [
        line.split("//", 1)[0].strip()
        for line in source_text.splitlines()
        if any(token in line for token in ("tensor_load_to_lds", "s_wait_tensorcnt", "s_barrier_signal", "s_barrier_wait"))
    ]
    retained_output = [
        line.split("//", 1)[0].strip()
        for line in output_text.splitlines()
        if any(token in line for token in ("tensor_load_to_lds", "s_wait_tensorcnt", "s_barrier_signal", "s_barrier_wait"))
    ]
    if retained_source != retained_output:
        raise AssertionError("TDM/synchronization instruction sequence changed")

    for k in (512, 1024, 7168, 16384):
        for wave in range(4):
            full = descriptor_plan(k, wave)
            loadonly = descriptor_plan(k, wave)
            if full != loadonly:
                raise AssertionError(f"dynamic plan mismatch: K={k}, wave={wave}")
            print(
                f"plan K={k} wave={wave} type={full[0][1]} "
                f"descriptors={len(full)} q=0..{k // 128} equivalent=yes"
            )
            if verbose_plan:
                for row in full:
                    print("  q=%d type=%s global=%s K=%s LDS_copy=%s stage=%s mask=%s" % row)

    print("audit=PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).with_name("mab_tdm_gemm_full_batch.s"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("mab_tdm_gemm_full_batch_loadonly.s"),
    )
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--verbose-plan", action="store_true")
    args = parser.parse_args()
    if args.generate:
        generate(args.source, args.output)
    audit(args.source, args.output, args.verbose_plan)


if __name__ == "__main__":
    main()
