#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

OLD_SYMBOL = "f4gemm_bf16_mxfp4_ABpreShuffle_256x256_4x4_ps"
NEW_SYMBOL = "f4gemm_bf16_mxfp4_ABpreShuffle_128x128_4x4_ps"

CATEGORY_NAMES = (
    "wmma",
    "ds_load_b128",
    "ds_load_b32",
    "ds_store_or_write",
    "tensor_load_to_lds",
    "tensor_store_from_lds",
    "s_wait_dscnt",
    "s_wait_tensorcnt",
    "s_wait_alu",
    "wg_barrier_signal",
    "wg_barrier_wait",
    "cluster_barrier_signal",
    "cluster_barrier_wait",
    "conditional_branch",
    "unconditional_branch",
    "conversion",
)

NEW_RING_ADDRESSES = (
    0x00000, 0x04000, 0x08000, 0x0C000,
    0x10000, 0x10400, 0x10800, 0x10C00,
    0x11000, 0x11400, 0x11800, 0x11C00,
    0x12000, 0x16000, 0x1A000, 0x1E000,
    0x22000,
)
OLD_ONLY_RING_ADDRESSES = (0x30000, 0x38000, 0x40000, 0x48000, 0x50000)
SELECTED_CONSTANTS = tuple(dict.fromkeys(
    NEW_RING_ADDRESSES
    + OLD_ONLY_RING_ADDRESSES
    + (0x40, 0x80, 0x100, 0x400, 0x1000, 0x2000, 0x8000, 0x26000, 0x2A000)
))


def sha256_path(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strip_comments(line):
    line = re.sub(r"/\*.*?\*/", "", line)
    line = re.sub(r";.*$", "", line)
    line = re.sub(r"//.*$", "", line)
    return line.rstrip()


def source_labels(text):
    labels = []
    in_metadata = False
    for raw in text.splitlines():
        clean = strip_comments(raw).strip()
        if clean.startswith(".amdgpu_metadata"):
            in_metadata = True
            continue
        if clean.startswith(".end_amdgpu_metadata"):
            in_metadata = False
            continue
        if in_metadata:
            continue
        match = re.match(r"^([.$A-Za-z_][\w.$]*):\s*$", clean)
        if not match:
            match = re.match(r"^[0-9a-fA-F]+\s+<([^>]+)>:\s*$", clean)
        if match:
            labels.append(match.group(1))
    return labels


def instruction_records(text):
    records = []
    in_metadata = False
    current_label = "<entry>"
    for line_number, raw in enumerate(text.splitlines(), 1):
        clean = strip_comments(raw).strip()
        if clean.startswith(".amdgpu_metadata"):
            in_metadata = True
            continue
        if clean.startswith(".end_amdgpu_metadata"):
            in_metadata = False
            continue
        if in_metadata or not clean:
            continue
        label = re.match(r"^([.$A-Za-z_][\w.$]*):\s*$", clean)
        if not label:
            label = re.match(r"^[0-9a-fA-F]+\s+<([^>]+)>:\s*$", clean)
        if label:
            current_label = label.group(1)
            continue
        if clean.startswith(".") or clean.startswith("#"):
            continue
        # Most AMDGPU llvm-objdump output places address/encoding after //.
        # Also accept conventional address/byte prefixes if a tool emits them.
        clean = re.sub(
            r"^[0-9a-fA-F]+:\s+(?:(?:[0-9a-fA-F]{2}|[0-9a-fA-F]{8})\s+)+",
            "",
            clean,
        )
        match = re.match(r"^([a-z][a-z0-9_.]*)\b(?:\s+(.*))?$", clean)
        if not match:
            continue
        mnemonic = match.group(1).lower()
        if mnemonic in {"file", "format"}:
            continue
        records.append(
            {
                "line": line_number,
                "label_block": current_label,
                "mnemonic": mnemonic,
                "operands": (match.group(2) or "").strip(),
                "text": clean,
            }
        )
    return records


def first_operand(record):
    operands = record["operands"].strip()
    return re.split(r"[\s,]+", operands, maxsplit=1)[0].lower() if operands else ""


def categories(records):
    counts = Counter()
    ds_store_variants = Counter()
    conversion_variants = Counter()
    wmma_variants = Counter()
    for record in records:
        mnemonic = record["mnemonic"]
        operand = first_operand(record)
        if mnemonic.startswith("v_wmma"):
            counts["wmma"] += 1
            wmma_variants[mnemonic] += 1
        if mnemonic == "ds_load_b128":
            counts["ds_load_b128"] += 1
        if mnemonic == "ds_load_b32":
            counts["ds_load_b32"] += 1
        if re.match(r"^ds_(?:store|write)", mnemonic):
            counts["ds_store_or_write"] += 1
            ds_store_variants[mnemonic] += 1
        if mnemonic.startswith("tensor_load_to_lds"):
            counts["tensor_load_to_lds"] += 1
        if mnemonic.startswith("tensor_store_from_lds"):
            counts["tensor_store_from_lds"] += 1
        if mnemonic.startswith("s_wait_dscnt"):
            counts["s_wait_dscnt"] += 1
        if mnemonic.startswith("s_wait_tensorcnt"):
            counts["s_wait_tensorcnt"] += 1
        if mnemonic.startswith("s_wait_alu"):
            counts["s_wait_alu"] += 1
        if mnemonic == "s_barrier_signal" and operand in {"-1", "0xffffffff"}:
            counts["wg_barrier_signal"] += 1
        if mnemonic == "s_barrier_wait" and operand in {"0xffff", "65535"}:
            counts["wg_barrier_wait"] += 1
        if mnemonic == "s_barrier_signal" and operand in {"-3", "0xfffffffd"}:
            counts["cluster_barrier_signal"] += 1
        if mnemonic == "s_barrier_wait" and operand in {"0xfffd", "65533"}:
            counts["cluster_barrier_wait"] += 1
        if mnemonic.startswith("s_cbranch"):
            counts["conditional_branch"] += 1
        if mnemonic == "s_branch":
            counts["unconditional_branch"] += 1
        if (
            re.search(r"(?:^|_)(?:cvt|convert|pack)(?:_|$)", mnemonic)
            or mnemonic.startswith("v_cvt")
        ):
            counts["conversion"] += 1
            conversion_variants[mnemonic] += 1
    return (
        {name: counts[name] for name in CATEGORY_NAMES},
        dict(sorted(ds_store_variants.items())),
        dict(sorted(conversion_variants.items())),
        dict(sorted(wmma_variants.items())),
    )


def parse_integer_literal(token):
    token = token.strip()
    sign = -1 if token.startswith("-") else 1
    unsigned = token[1:] if token[:1] in {"+", "-"} else token
    if re.fullmatch(r"0[xX][0-9a-fA-F]+", unsigned):
        return sign * int(unsigned[2:], 16)
    if re.fullmatch(r"\d+", unsigned):
        return sign * int(unsigned, 10)
    raise ValueError(f"unsupported integer literal: {token!r}")


def remove_disassembly_pc_fields(line):
    line = re.sub(
        r"^\s*[0-9a-fA-F]{8,16}\s+<[^>]+>:\s*",
        "",
        line,
    )
    line = re.sub(
        r"^\s*[0-9a-fA-F]{4,16}:\s+(?:[0-9a-fA-F]{2,16}\s+)+",
        "",
        line,
    )
    return line


def literal_counts(text):
    counter = Counter()
    for raw in text.splitlines():
        clean = remove_disassembly_pc_fields(strip_comments(raw))
        clean = re.sub(r"\b(?:s|v|ttmp)\d+\b", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(
            r"\b[sv]\[\s*\d+\s*:\s*\d+\s*\]",
            " ",
            clean,
            flags=re.IGNORECASE,
        )
        for token in re.findall(
            r"(?<![A-Za-z0-9_])(?:-?0x[0-9a-fA-F]+|-?\d+)(?![A-Za-z0-9_])",
            clean,
        ):
            counter[parse_integer_literal(token)] += 1
    return counter


def run_literal_parser_self_tests():
    cases = {
        "0000000000001900": 1900,
        "172032": 172032,
        "-17": -17,
        "0x2A000": 0x2A000,
        "-0x2a": -42,
    }
    for token, expected in cases.items():
        actual = parse_integer_literal(token)
        assert actual == expected, (token, expected, actual)
    sample = (
        "0000000000001900 <kernel>:\n"
        "  s_mov_b32 s0, 42 // 000000001900: B0804009 00000002\n"
        "  s_add_i32 s1, s1, -7\n"
        "  s_mov_b32 s2, 0x2a\n"
    )
    counts = literal_counts(sample)
    assert counts[1900] == 0, counts
    assert counts[42] == 2, counts
    assert counts[-7] == 1, counts


def kernel_symbols(text):
    symbols = set()
    patterns = (
        r"(?m)^\s*\.amdhsa_kernel\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"(?m)^\s*\.globl\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"(?m)^\s*\.name:\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)",
        r"(?m)^\s*\.symbol:\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)\.kd",
    )
    for pattern in patterns:
        symbols.update(re.findall(pattern, text))
    return sorted(symbols)


def branch_inventory(records, labels, symbols):
    known = {value.lower() for value in labels} | {
        value.lower() for value in symbols
    }
    branches = []
    unresolved = []
    for record in records:
        mnemonic = record["mnemonic"]
        if mnemonic != "s_branch" and not mnemonic.startswith("s_cbranch"):
            continue
        target = first_operand(record)
        symbolic = bool(re.match(r"^[.$A-Za-z_][\w.$]*$", target))
        resolved = (not symbolic) or target in known
        entry = {
            "line": record["line"],
            "mnemonic": mnemonic,
            "target": target,
            "symbolic": symbolic,
            "resolved_in_text": resolved,
        }
        branches.append(entry)
        if symbolic and not resolved:
            unresolved.append(entry)
    return branches, unresolved


def normalized_instruction_sequence(records):
    sequence = []
    for record in records:
        mnemonic = re.sub(r"_e(?:32|64)$", "", record["mnemonic"])
        operands = re.sub(r"\s+", " ", record["operands"].strip().lower())
        sequence.append(f"{mnemonic} {operands}".rstrip())
    payload = "\n".join(sequence).encode()
    return sequence, hashlib.sha256(payload).hexdigest()


def analyze_input(label, kind, path):
    text = path.read_text(encoding="utf-8", errors="replace")
    records = instruction_records(text)
    labels = source_labels(text)
    symbols = kernel_symbols(text)
    category_counts, ds_variants, conversion_variants, wmma_variants = categories(records)
    branches, unresolved = branch_inventory(records, labels, symbols)
    literals = literal_counts(text)
    wmma_by_label = Counter(
        record["label_block"]
        for record in records
        if record["mnemonic"].startswith("v_wmma")
    )
    sequence, sequence_hash = normalized_instruction_sequence(records)
    return {
        "path": str(path),
        "kind": kind,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_path(path),
        "instruction_count": len(records),
        "normalized_instruction_sha256": sequence_hash,
        "counts": category_counts,
        "ds_store_write_variants": ds_variants,
        "conversion_variants": conversion_variants,
        "wmma_variants": wmma_variants,
        "kernel_symbols": symbols,
        "old_symbol_occurrences": text.count(OLD_SYMBOL),
        "new_symbol_occurrences": text.count(NEW_SYMBOL),
        "selected_constant_occurrences": {
            f"0x{value:x}": literals[value] for value in SELECTED_CONSTANTS
        },
        "new_ring_constant_occurrences": {
            f"0x{value:05x}": literals[value] for value in NEW_RING_ADDRESSES
        },
        "old_only_ring_constant_occurrences": {
            f"0x{value:x}": literals[value] for value in OLD_ONLY_RING_ADDRESSES
        },
        "tile_literal_occurrences": {
            "64_or_0x40": literals[64],
            "128_or_0x80": literals[128],
            "256_or_0x100": literals[256],
        },
        "tile_string_occurrences": {
            "64x64": len(re.findall(r"64x64", text, re.IGNORECASE)),
            "128x128": len(re.findall(r"128x128", text, re.IGNORECASE)),
            "256x256": len(re.findall(r"256x256", text, re.IGNORECASE)),
        },
        "label_count": len(labels),
        "labels": labels,
        "branch_count": len(branches),
        "branches": branches,
        "unresolved_symbolic_branches": unresolved,
        "wmma_by_label_block": dict(sorted(wmma_by_label.items())),
        "wmma_16_count_label_blocks": sorted(
            block for block, count in wmma_by_label.items() if count == 16
        ),
        "phase_boundary_proof": False,
        "phase_boundary_note": (
            "Label-block WMMA counts are inventory only. The parser does not "
            "infer semantic K-phase boundaries and therefore makes no "
            "per-phase 16-WMMA claim."
        ),
    }


def barrier_pair(records):
    for left, right in zip(records, records[1:]):
        if (
            left["mnemonic"] == "s_barrier_signal"
            and first_operand(left) in {"-1", "0xffffffff"}
            and right["mnemonic"] == "s_barrier_wait"
            and first_operand(right) in {"0xffff", "65535"}
        ):
            return True
    return False


def design_analysis(target_text, target_report):
    records = instruction_records(target_text)
    literals = literal_counts(target_text)
    counts = target_report["counts"]
    stores = [
        index
        for index, record in enumerate(records)
        if record["mnemonic"].startswith("tensor_store_from_lds")
    ]
    protocols = []
    for index in stores:
        protocols.append(
            {
                "instruction_index": index,
                "line": records[index]["line"],
                "wg_pair_within_32_instructions_before": barrier_pair(
                    records[max(0, index - 32):index]
                ),
                "wg_pair_within_32_instructions_after": barrier_pair(
                    records[index + 1:index + 33]
                ),
                "wait_tensorcnt_within_32_after": any(
                    record["mnemonic"].startswith("s_wait_tensorcnt")
                    for record in records[index + 1:index + 33]
                ),
            }
        )
    first_last_two_rounds = False
    if stores:
        first_last_two_rounds = (
            barrier_pair(records[max(0, stores[0] - 48):stores[0]])
            and barrier_pair(records[stores[-1] + 1:stores[-1] + 49])
        )

    ring_presence = {
        f"0x{value:05x}": literals[value] for value in NEW_RING_ADDRESSES
    }
    old_presence = {
        f"0x{value:x}": literals[value] for value in OLD_ONLY_RING_ADDRESSES
    }
    warnings = []
    missing_ring = [name for name, count in ring_presence.items() if count == 0]
    if missing_ring:
        warnings.append(
            "HEURISTIC WARNING: expected packed-ring literals missing: "
            + ", ".join(missing_ring)
        )
    if literals[0x2A000] == 0 and not re.search(r"\b172032\b", target_text):
        warnings.append(
            "HEURISTIC WARNING: candidate fixed-LDS end/resource literal "
            "0x2A000 (172032) is absent"
        )
    if literals[0x22000] == 0:
        warnings.append(
            "HEURISTIC WARNING: input-ring/output-staging boundary literal "
            "0x22000 is absent"
        )
    if literals[0x80] == 0 and not re.search(r"(?<!\w)128(?!\w)", target_text):
        warnings.append("HEURISTIC WARNING: 128 tile/wave offset literal was not found")
    if literals[0x40] == 0 and not re.search(r"(?<!\w)64(?!\w)", target_text):
        warnings.append("HEURISTIC WARNING: 64 tile/wave offset literal was not found")
    if counts["s_wait_alu"] == 0:
        warnings.append("s_wait_alu is absent")
    if not first_last_two_rounds:
        warnings.append(
            "HEURISTIC WARNING: nearby-barrier search did not find two output "
            "WG rounds (before first and after last output TDM); this is not "
            "semantic proof"
        )
    if counts["wg_barrier_signal"] != counts["wg_barrier_wait"]:
        warnings.append(
            "HEURISTIC WARNING: raw WG barrier signal/wait counts are not "
            f"balanced ({counts['wg_barrier_signal']} vs "
            f"{counts['wg_barrier_wait']}); counts do not prove protocol semantics"
        )
    if counts["cluster_barrier_signal"] == 0 or counts["cluster_barrier_wait"] == 0:
        warnings.append(
            "HEURISTIC WARNING: raw cluster barrier signal/wait presence is incomplete"
        )
    if counts["cluster_barrier_signal"] != counts["cluster_barrier_wait"]:
        warnings.append(
            "HEURISTIC WARNING: raw cluster barrier signal/wait counts are not "
            f"balanced ({counts['cluster_barrier_signal']} vs "
            f"{counts['cluster_barrier_wait']}); counts do not prove protocol semantics"
        )
    present_old = [name for name, count in old_presence.items() if count]
    if present_old:
        warnings.append(
            "HEURISTIC WARNING: old-only ring/LDS literals remain: "
            + ", ".join(present_old)
        )
    if target_report["old_symbol_occurrences"]:
        warnings.append("original 256x256 kernel symbol remains")
    if target_report["new_symbol_occurrences"] == 0:
        warnings.append("new 128x128 contract kernel symbol is absent")
    if re.search(
        r"(?mi)^\s*\.amdhsa_group_segment_fixed_size\s+327680\b",
        target_text,
    ):
        warnings.append("old 327680-byte LDS descriptor remains")
    if re.search(
        r"(?mi)^\s*\.amdhsa_next_free_vgpr\s+1024\b",
        target_text,
    ):
        warnings.append("old next_free_vgpr=1024 descriptor remains")

    return {
        "heuristic_notice": (
            "Literal presence, nearby-barrier windows, and raw barrier counts are "
            "search aids only; they are not semantic proof and never hard failures."
        ),
        "new_ring_address_occurrences": ring_presence,
        "old_only_ring_address_occurrences": old_presence,
        "output_staging_boundary_0x22000_occurrences": literals[0x22000],
        "fixed_lds_end_0x2a000_occurrences": literals[0x2A000],
        "tile_offset_128_literal_occurrences": literals[0x80],
        "wave_offset_64_literal_occurrences": literals[0x40],
        "wait_alu_count": counts["s_wait_alu"],
        "wg_barrier_signal_count": counts["wg_barrier_signal"],
        "wg_barrier_wait_count": counts["wg_barrier_wait"],
        "cluster_barrier_signal_count": counts["cluster_barrier_signal"],
        "cluster_barrier_wait_count": counts["cluster_barrier_wait"],
        "wmma_total": counts["wmma"],
        "wmma_total_divmod_16": list(divmod(counts["wmma"], 16)),
        "wmma_label_blocks_with_exactly_16": target_report[
            "wmma_16_count_label_blocks"
        ],
        "wmma_phase_claim": (
            "None. Exact-16 label blocks and total/divmod are search aids only; "
            "semantic phase boundaries are not proven."
        ),
        "output_tdm_store_count": counts["tensor_store_from_lds"],
        "output_tdm_shape_interpretation": (
            "CONTRACT PASS: exactly one output TDM store"
            if counts["tensor_store_from_lds"] == 1
            else "CONTRACT FAIL: expected exactly one output TDM store"
        ),
        "output_store_protocol_windows": protocols,
        "two_output_wg_rounds_statically_detected": first_last_two_rounds,
        "warnings": warnings,
    }


def metadata_analysis(out_dir, target_path):
    candidates = [target_path]
    for pattern in (
        "*.readelf*.txt",
        "*.readobj*.txt",
        "*.objdump.sections_symbols.txt",
        "*.nm.*.txt",
    ):
        candidates.extend(sorted(out_dir.glob(pattern)))
    old_symbol_hits = []
    new_symbol_hits = []
    old_lds_hits = []
    candidate_lds_hits = []
    old_vgpr_hits = []
    resource_lines = []
    for path in dict.fromkeys(candidates):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), 1):
            lower = line.lower()
            entry = {"file": str(path), "line": number, "text": line}
            if OLD_SYMBOL in line:
                old_symbol_hits.append(entry)
            if NEW_SYMBOL in line:
                new_symbol_hits.append(entry)
            resource_context = any(
                token in lower
                for token in (
                    "group_segment",
                    "groupsegment",
                    "lds",
                    "vgpr",
                    "sgpr",
                    "kernarg",
                    "wavefront",
                    "amdhsa",
                )
            )
            if resource_context:
                resource_lines.append(entry)
            if resource_context and (
                re.search(r"\b327680\b", line)
                or re.search(r"\b0x0*50000\b", line, re.IGNORECASE)
            ):
                old_lds_hits.append(entry)
            if resource_context and (
                re.search(r"\b172032\b", line)
                or re.search(r"\b0x0*2a000\b", line, re.IGNORECASE)
            ):
                candidate_lds_hits.append(entry)
            if "vgpr" in lower and re.search(r"\b1024\b", line):
                old_vgpr_hits.append(entry)
    warnings = []
    if old_symbol_hits:
        warnings.append("original 256x256 kernel symbol remains in source/ELF reports")
    if not new_symbol_hits:
        warnings.append("128x128 contract kernel symbol is absent from source/ELF reports")
    if old_lds_hits:
        warnings.append("old 327680-byte (0x50000) LDS resource remains")
    if old_vgpr_hits:
        warnings.append("old 1024-VGPR resource remains")
    if not candidate_lds_hits:
        warnings.append("candidate 172032-byte (0x2A000) LDS resource is absent")
    return {
        "files_scanned": [str(path) for path in dict.fromkeys(candidates) if path.is_file()],
        "old_symbol_hits": old_symbol_hits,
        "new_symbol_hits": new_symbol_hits,
        "old_lds_hits": old_lds_hits,
        "candidate_lds_hits": candidate_lds_hits,
        "old_vgpr_hits": old_vgpr_hits,
        "resource_lines": resource_lines,
        "warnings": warnings,
    }


def descriptor_values(text, directive):
    pattern = re.compile(
        rf"(?mi)^\s*\.{re.escape(directive)}\s*:?\s*(-?(?:0x[0-9a-f]+|\d+))\b"
    )
    return [parse_integer_literal(token) for token in pattern.findall(text)]


def contract_analysis(target_text, reports, metadata):
    target_report = reports.get("target_source")
    reference_report = reports.get("reference_source")
    disassembly_reports = {
        label: report
        for label, report in reports.items()
        if report["kind"] == "disassembly"
    }
    group_values = descriptor_values(
        target_text, "amdhsa_group_segment_fixed_size"
    )
    next_vgpr_values = descriptor_values(target_text, "amdhsa_next_free_vgpr")
    source_store_count = (
        target_report["counts"]["tensor_store_from_lds"] if target_report else None
    )
    disassembly_store_counts = {
        label: report["counts"]["tensor_store_from_lds"]
        for label, report in disassembly_reports.items()
    }
    disassembly_symbol_occurrences = {
        label: report["new_symbol_occurrences"]
        for label, report in disassembly_reports.items()
    }
    unchanged_reference_copy = bool(
        target_report
        and reference_report
        and target_report["sha256"] == reference_report["sha256"]
    )
    gates = {
        "required_symbol_in_target_source": bool(
            target_report and target_report["new_symbol_occurrences"] > 0
        ),
        "required_symbol_in_every_produced_disassembly": bool(
            disassembly_reports
            and all(value > 0 for value in disassembly_symbol_occurrences.values())
        ),
        "target_group_segment_exactly_172032": group_values == [172032],
        "target_source_exactly_one_tensor_store_from_lds": source_store_count == 1,
        "every_produced_disassembly_exactly_one_tensor_store_from_lds": bool(
            disassembly_reports
            and all(value == 1 for value in disassembly_store_counts.values())
        ),
    }
    failures = [
        name for name, passed in gates.items() if not passed
    ]
    old_findings = {
        "old_256_symbol_in_target_source": bool(
            target_report and target_report["old_symbol_occurrences"] > 0
        ),
        "old_256_symbol_in_produced_disassembly": {
            label: report["old_symbol_occurrences"]
            for label, report in disassembly_reports.items()
            if report["old_symbol_occurrences"]
        },
        "old_group_segment_327680_values": [
            value for value in group_values if value == 327680
        ],
        "old_next_free_vgpr_1024_values": [
            value for value in next_vgpr_values if value == 1024
        ],
        "metadata_old_symbol_hit_count": len(metadata["old_symbol_hits"]),
        "metadata_old_lds_hit_count": len(metadata["old_lds_hits"]),
        "metadata_old_vgpr_hit_count": len(metadata["old_vgpr_hits"]),
    }
    return {
        "contract_symbol": NEW_SYMBOL,
        "contract_group_segment_bytes": 172032,
        "contract_tensor_store_from_lds_count": 1,
        "unchanged_reference_copy": unchanged_reference_copy,
        "target_group_segment_values": group_values,
        "target_next_free_vgpr_values": next_vgpr_values,
        "target_source_tensor_store_from_lds_count": source_store_count,
        "produced_disassembly_tensor_store_from_lds_counts": (
            disassembly_store_counts
        ),
        "produced_disassembly_required_symbol_occurrences": (
            disassembly_symbol_occurrences
        ),
        "gates": gates,
        "failure_count": len(failures),
        "failures": failures,
        "old_contract_findings": old_findings,
    }


def parse_nm_symbols(path):
    if not path.is_file():
        return []
    symbols = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if parts:
            symbols.append(parts[-1])
    return sorted(set(symbols))


def route_analysis(manifest, reports):
    objects = {}
    disassemblies = {}
    for label, kind, path in manifest:
        if kind == "object":
            objects[label] = path
        elif kind == "disassembly":
            disassemblies[label] = path
    routes = {}
    for label, path in objects.items():
        entry = {
            "path": str(path),
            "elf_sha256": sha256_path(path),
            "size_bytes": path.stat().st_size,
            "symbols": parse_nm_symbols(path.parent / f"{label}.nm.defined.txt"),
        }
        if label in disassemblies:
            text = disassemblies[label].read_text(encoding="utf-8", errors="replace")
            records = instruction_records(text)
            counts, _, _, _ = categories(records)
            _, sequence_hash = normalized_instruction_sequence(records)
            entry.update(
                {
                    "disassembly_path": str(disassemblies[label]),
                    "instruction_count": len(records),
                    "normalized_instruction_sha256": sequence_hash,
                    "counts": counts,
                }
            )
        routes[label] = entry
    warnings = []
    comparison = {"available": False}
    if "llvm_mc_object" in routes and "clang_object" in routes:
        left = routes["llvm_mc_object"]
        right = routes["clang_object"]
        comparison = {
            "available": True,
            "whole_elf_hash_equal": left["elf_sha256"] == right["elf_sha256"],
            "whole_elf_difference_is_not_alone_a_failure": True,
            "size_equal": left["size_bytes"] == right["size_bytes"],
            "normalized_instruction_hash_equal": (
                left.get("normalized_instruction_sha256")
                == right.get("normalized_instruction_sha256")
            ),
            "instruction_count_equal": (
                left.get("instruction_count") == right.get("instruction_count")
            ),
            "category_count_differences": {
                name: [
                    left.get("counts", {}).get(name, 0),
                    right.get("counts", {}).get(name, 0),
                ]
                for name in CATEGORY_NAMES
                if left.get("counts", {}).get(name, 0)
                != right.get("counts", {}).get(name, 0)
            },
            "symbol_sets_equal": left.get("symbols") == right.get("symbols"),
            "symbols_only_llvm_mc": sorted(
                set(left.get("symbols", ())) - set(right.get("symbols", ()))
            ),
            "symbols_only_clang": sorted(
                set(right.get("symbols", ())) - set(left.get("symbols", ()))
            ),
        }
        if not comparison["normalized_instruction_hash_equal"]:
            warnings.append("llvm-mc and clang normalized disassemblies differ")
        if comparison["category_count_differences"]:
            warnings.append("llvm-mc and clang opcode category counts differ")
        if not comparison["symbol_sets_equal"]:
            warnings.append("llvm-mc and clang defined symbol sets differ")
    else:
        warnings.append("two-route object comparison is unavailable")
    return {"routes": routes, "comparison": comparison, "warnings": warnings}


def print_hits(title, hits, limit=200):
    print(title)
    if not hits:
        print("  none")
        return
    for hit in hits[:limit]:
        print(f"  {hit['file']}:{hit['line']}: {hit['text']}")
    if len(hits) > limit:
        print(f"  ... {len(hits) - limit} additional hits retained in JSON")


def main():
    run_literal_parser_self_tests()
    print("Embedded literal-parser self-tests: PASS")
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    target_path = Path(args.target)

    manifest = []
    for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        label, kind, raw_path = line.split("\t", 2)
        path = Path(raw_path)
        if path.is_file():
            manifest.append((label, kind, path))

    reports = {}
    static_warnings = []
    for label, kind, path in manifest:
        if kind not in {"source", "disassembly"}:
            continue
        report = analyze_input(label, kind, path)
        reports[label] = report
        if kind == "source" and report["unresolved_symbolic_branches"]:
            static_warnings.append(
                f"{label}: {len(report['unresolved_symbolic_branches'])} "
                "symbolic branch target(s) not defined in the parsed text"
            )

    target_report = reports.get("target_source")
    if target_report:
        for label, report in reports.items():
            if report["kind"] != "disassembly":
                continue
            differences = {
                name: [target_report["counts"][name], report["counts"][name]]
                for name in CATEGORY_NAMES
                if target_report["counts"][name] != report["counts"][name]
            }
            report["target_source_category_differences"] = differences
            if differences:
                static_warnings.append(
                    f"{label}: opcode category counts differ from target source"
                )

    static_payload = {
        "category_definitions": {
            "wmma": "mnemonic starts with v_wmma",
            "ds_store_or_write": "mnemonic starts with ds_store or ds_write",
            "conversion": "mnemonic contains a cvt/convert/pack component",
            "wg_barrier": "s_barrier_signal -1 / s_barrier_wait 0xffff",
            "cluster_barrier": "s_barrier_signal -3 / s_barrier_wait 0xfffd",
        },
        "inputs": reports,
        "warnings": static_warnings,
    }
    (out_dir / "static_count.json").write_text(
        json.dumps(static_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "static_count.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("input\tgroup\tmetric\tvalue\n")
        for label, report in reports.items():
            handle.write(f"{label}\tsummary\tinstruction_count\t{report['instruction_count']}\n")
            for name, value in report["counts"].items():
                handle.write(f"{label}\tcount\t{name}\t{value}\n")
            for name, value in report["ds_store_write_variants"].items():
                handle.write(f"{label}\tds_store_write_variant\t{name}\t{value}\n")
            for name, value in report["conversion_variants"].items():
                handle.write(f"{label}\tconversion_variant\t{name}\t{value}\n")
            for name, value in report["selected_constant_occurrences"].items():
                handle.write(f"{label}\tconstant\t{name}\t{value}\n")
            for name, value in report["tile_literal_occurrences"].items():
                handle.write(f"{label}\ttile_literal\t{name}\t{value}\n")
            for name, value in report["tile_string_occurrences"].items():
                handle.write(f"{label}\ttile_string\t{name}\t{value}\n")

    target_text = target_path.read_text(encoding="utf-8", errors="replace")
    design = design_analysis(target_text, target_report) if target_report else {
        "warnings": ["target source was not available to the audit"]
    }
    (out_dir / "design_invariants.json").write_text(
        json.dumps(design, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metadata = metadata_analysis(out_dir, target_path)
    (out_dir / "metadata_resource_analysis.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    contract = contract_analysis(target_text, reports, metadata)
    (out_dir / "target_contract_gates.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    routes = route_analysis(manifest, reports)
    (out_dir / "route_comparison.json").write_text(
        json.dumps(routes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("STATIC OPCODE / CONTROL-FLOW AUDIT")
    print("Per-phase note: this parser does not infer phase boundaries and makes no per-phase 16-WMMA claim.")
    for label, report in reports.items():
        print(f"\n[{label}] {report['path']}")
        print(f"  sha256={report['sha256']}")
        print(f"  instruction_count={report['instruction_count']}")
        for name in CATEGORY_NAMES:
            print(f"  {name}={report['counts'][name]}")
        print(f"  ds_store/write variants={report['ds_store_write_variants']}")
        print(f"  conversion variants={report['conversion_variants']}")
        print(f"  WMMA variants={report['wmma_variants']}")
        print(f"  kernel symbols={report['kernel_symbols']}")
        print(f"  labels={report['label_count']}")
        print(f"  branches={report['branch_count']}")
        print(
            "  unresolved symbolic branches="
            f"{len(report['unresolved_symbolic_branches'])}"
        )
        print(f"  old symbol strings={report['old_symbol_occurrences']}")
        print(f"  new symbol strings={report['new_symbol_occurrences']}")
        print(f"  selected constants={report['selected_constant_occurrences']}")
        print(f"  new ring constants={report['new_ring_constant_occurrences']}")
        print(f"  old-only ring constants={report['old_only_ring_constant_occurrences']}")
        print(f"  tile literals={report['tile_literal_occurrences']}")
        print(f"  tile strings={report['tile_string_occurrences']}")
        print(f"  exact-16 label blocks={report['wmma_16_count_label_blocks']}")
        print("  branch inventory:")
        for branch in report["branches"]:
            print(
                f"    line {branch['line']}: {branch['mnemonic']} "
                f"{branch['target']} resolved_in_text={branch['resolved_in_text']}"
            )

    print("\nSTATIC AUDIT WARNINGS")
    if static_warnings:
        for warning in static_warnings:
            print(f"  WARNING: {warning}")
    else:
        print("  none")

    print("\nHARD TARGET-CONTRACT GATES")
    print(f"  unchanged_reference_copy: {contract['unchanged_reference_copy']}")
    for name, passed in contract["gates"].items():
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")
    print(f"  failure_count: {contract['failure_count']}")
    for failure in contract["failures"]:
        print(f"  CONTRACT FAILURE: {failure}")
    print("  PROMINENT OLD-CONTRACT FINDINGS:")
    for name, value in contract["old_contract_findings"].items():
        print(f"    {name}: {value}")

    print("\nDESIGN HEURISTICS (SEARCH AIDS ONLY; NOT SEMANTIC PROOF)")
    for key, value in design.items():
        if key != "warnings":
            print(f"  {key}: {value}")
    if design.get("warnings"):
        for warning in design["warnings"]:
            print(f"  WARNING: {warning}")
    else:
        print("  warnings: none")

    print("\nMETADATA / RESOURCE INSPECTION")
    print_hits("Old 256x256 symbol hits:", metadata["old_symbol_hits"])
    print_hits("New 128x128 contract symbol hits:", metadata["new_symbol_hits"])
    print_hits("Old 327680-byte LDS hits:", metadata["old_lds_hits"])
    print_hits("Candidate 172032-byte LDS hits:", metadata["candidate_lds_hits"])
    print_hits("Old 1024-VGPR hits:", metadata["old_vgpr_hits"])
    for warning in metadata["warnings"]:
        print(f"  WARNING: {warning}")

    print("\nLLVM-MC VS CLANG OBJECT COMPARISON")
    print(json.dumps(routes, indent=2, sort_keys=True))
    print("Whole-ELF hash/size differences alone are informational and do not fail validation.")

    status_values = {
        "STATIC_WARNINGS": len(static_warnings),
        "DESIGN_WARNINGS": len(design.get("warnings", ())),
        "METADATA_WARNINGS": len(metadata["warnings"]),
        "ROUTE_WARNINGS": len(routes["warnings"]),
        "ROUTE_COMPARISON_AVAILABLE": int(
            routes.get("comparison", {}).get("available", False)
        ),
        "CONTRACT_FAILURES": contract["failure_count"],
        "UNCHANGED_REFERENCE_COPY": int(contract["unchanged_reference_copy"]),
        "TARGET_SOURCE_AUDITED": int("target_source" in reports),
        "REFERENCE_SOURCE_AUDITED": int("reference_source" in reports),
        "DISASSEMBLIES_AUDITED": sum(
            1 for report in reports.values() if report["kind"] == "disassembly"
        ),
        "UNRESOLVED_BRANCHES": sum(
            len(report["unresolved_symbolic_branches"])
            for report in reports.values()
            if report["kind"] == "source"
        ),
    }
    with (out_dir / "analysis_status.env").open("w", encoding="utf-8", newline="\n") as handle:
        for key, value in status_values.items():
            handle.write(f"{key}={value}\n")
    print("\nARTIFACTS")
    for name in (
        "static_count.json",
        "static_count.tsv",
        "design_invariants.json",
        "metadata_resource_analysis.json",
        "target_contract_gates.json",
        "route_comparison.json",
        "analysis_status.env",
    ):
        print(f"  {out_dir / name}")


if __name__ == "__main__":
    main()
