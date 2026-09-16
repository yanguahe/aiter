#!/usr/bin/env python3
"""Audit a B64-clear instruction-prefetch assembly variant."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "persistent_overlap_pad8_prefetch_stage0_b64_clear.s"
EXPECTED_SOURCE_SHA256 = (
    "c863c8a6f2f51cd2ce85381addf782cefa6ab0dbe9a2ffee2f9e44e875786788"
)


def count(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text, flags=re.MULTILINE))


def check(name: str, actual: object, expected: object, failures: list[str]) -> None:
    state = "ok" if actual == expected else "FAIL"
    print(f"{state:4} {name:42} actual={actual} expected={expected}")
    if actual != expected:
        failures.append(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel", type=Path)
    parser.add_argument("--static-prefetches", type=int, required=True)
    parser.add_argument("--owner-waves", type=int, choices=(1, 2, 3, 4), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kernel = args.kernel.resolve()
    if not kernel.is_file():
        raise SystemExit(f"kernel is missing: {kernel}")

    source_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise SystemExit(f"source SHA256 changed: {source_sha}")

    source = SOURCE.read_text(encoding="utf-8")
    text = kernel.read_text(encoding="utf-8")
    failures: list[str] = []

    check(
        "B64 accumulator clears",
        count(text, r"^\s*v_mov_b64_e32\s+v\[(?:10[0-9]|1[1-9][0-9]|2[0-2][0-9]):(?:10[1-9]|1[1-9][0-9]|2[0-2][0-9])\],\s*0\s*$"),
        1280,
        failures,
    )
    check(
        "static instruction prefetches",
        count(text, r"^\s*s_prefetch_inst\b"),
        args.static_prefetches,
        failures,
    )
    if args.owner_waves == 1:
        owner_pattern = (
            r"\ts_cmp_eq_u32 s22, 0\n"
            r"\ts_cbranch_scc0 \.Lmoe_instruction_prefetch_done"
        )
        check("wave-0 owner predicate", count(text, owner_pattern), 1, failures)
    else:
        check(
            "cooperative owner predicate",
            text.count(f"\ts_cmp_lt_u32 s22, {args.owner_waves}"),
            1,
            failures,
        )

    for name, pattern in (
        ("tensor loads", r"^\s*tensor_load_to_lds\b"),
        ("tensor stores", r"^\s*tensor_store_from_lds\b"),
        ("WMMA instructions", r"^\s*v_wmma_"),
        ("DS operations", r"^\s*ds_"),
        ("barrier operations", r"^\s*s_barrier_(?:signal|wait)\b"),
        ("TENSOR waits", r"^\s*s_wait_tensorcnt\b"),
        ("DScnt waits", r"^\s*s_wait_dscnt\b"),
    ):
        check(name, count(text, pattern), count(source, pattern), failures)

    check("next_free_sgpr", text.count(".amdhsa_next_free_sgpr 104"), 1, failures)
    check("next_free_vgpr", text.count(".amdhsa_next_free_vgpr 1024"), 1, failures)
    check("320 KiB LDS", text.count("327680"), 2, failures)

    labels = re.findall(r"^([.$A-Za-z_][.$A-Za-z0-9_]*):", text, flags=re.MULTILINE)
    duplicates = sorted({label for label in labels if labels.count(label) != 1})
    check("unique labels", duplicates, [], failures)
    print(f"info kernel_sha256={hashlib.sha256(kernel.read_bytes()).hexdigest()}")
    if failures:
        raise SystemExit(f"{len(failures)} static checks failed")


if __name__ == "__main__":
    main()
