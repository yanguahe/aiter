#!/usr/bin/env python3
"""Static audit for the next-task stage-0 and stage-0/1 prefetch kernels."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASELINE = HERE / "persistent_overlap_output_pad8.s"
VARIANTS = (
    (HERE / "persistent_overlap_pad8_prefetch_stage0.s", 1, 60, 2, 1),
    (HERE / "persistent_overlap_pad8_prefetch_stage01.s", 2, 64, 3, 0),
)
EXPECTED_BASELINE_SHA256 = (
    "039ed787b1f136ee402b76e3bd0b7c9bf439148a0156d0fcca856ed6ab25ccad"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text, flags=re.MULTILINE))


def check(name: str, actual: object, expected: object, failures: list[str]) -> None:
    state = "ok" if actual == expected else "FAIL"
    print(f"{state:4} {name:38} actual={actual} expected={expected}")
    if actual != expected:
        failures.append(name)


def audit(path: Path, stages: int, tensor_loads: int, entry_wait: int, old_wait: int) -> None:
    text = path.read_text(encoding="utf-8")
    failures: list[str] = []
    print(f"\n== {path.name} ==")
    check("tensor_load_to_lds", count(text, r"^\s*tensor_load_to_lds\b"), tensor_loads, failures)
    check("tensor_store_from_lds", count(text, r"^\s*tensor_store_from_lds\b"), 2, failures)
    check("early role issues", count(text, r"^\.Lmoe_next_prefetch_(?:a|b|scale_a|scale_b):$"), 4, failures)
    check("prefetched entry", count(text, r"^\.Lmoe_persistent_task_prefetched:$"), 1, failures)
    check("reused scalar-state entry", count(text, r"^\.Lmoe_persistent_state_ready:$"), 1, failures)
    check("boundary uses prefetched entry", text.count("s_branch .Lmoe_persistent_task_prefetched"), 1, failures)
    check("descriptor construction guards", count(text, r"prefetched_descriptor_ready$"), 4, failures)
    check("stage0 issue guards", count(text, r"^\.Lmoe_(?:a|b|scale_a|scale_b)_stage0_already_prefetched:$"), 4, failures)
    check("stage1 issue guards", count(text, r"^\.Lmoe_(?:a|b|scale_a|scale_b)_stage1_already_prefetched:$"), 4 if stages == 2 else 0, failures)
    check(
        "entry partial waits",
        count(
            text,
            rf"^\.Lmoe_(?:a|b|scale_a|scale_b)_prefetched_input_wait:\n"
            rf"\s*s_wait_tensorcnt 0x{entry_wait:x}$",
        ),
        4,
        failures,
    )
    check(
        "old-output waits",
        count(
            text,
            rf"\ts_cbranch_scc1 \.Lmoe_(?:a|b|scale_a|scale_b)_old_output_already_safe\n"
            rf"\s*s_wait_tensorcnt 0x{old_wait:x}$",
        ),
        4,
        failures,
    )
    check("prefetch flag writes", text.count("s_mov_b32 s101,"), 2, failures)
    instruction_text = "\n".join(
        line.split(";", 1)[0]
        for line in text.splitlines()
        if not line.lstrip().startswith(";")
    )
    check("no s104/s105", count(instruction_text, r"\bs10[45]\b"), 0, failures)
    check("next_free_sgpr", text.count(".amdhsa_next_free_sgpr 104"), 1, failures)
    check("numbered_sgpr", text.count("numbered_sgpr, 104"), 1, failures)
    check("next_free_vgpr", text.count(".amdhsa_next_free_vgpr 1024"), 1, failures)
    check("320KiB LDS metadata", text.count("327680"), 2, failures)
    check("cluster signal", count(text, r"^\s*s_barrier_signal\s+-3\b"), 4, failures)
    check("cluster wait", count(text, r"^\s*s_barrier_wait\s+(?:0xfffd|-3)\b"), 12, failures)

    begin = text.index("\t; Prefetch stage")
    end = text.index("\t; Activated output uses a 144-byte LDS row pitch", begin)
    early = text[begin:end]
    check("early block has no VGPR instruction", count(early, r"^\s*v_[a-z0-9_]+\b"), 0, failures)
    check("early block has no kernarg load", count(early, r"^\s*s_load_"), 0, failures)
    check("early stage0 TDM count", count(early, r"^\s*tensor_load_to_lds\b"), 4 * stages, failures)
    check("early WG rendezvous", count(early, r"^\s*s_barrier_signal\s+-1\b"), 0, failures)
    check("early cluster rendezvous", count(early, r"^\s*s_barrier_signal\s+-3\b"), 0, failures)
    check("all expert bases advanced", count(early, r"^\s*s_add_nc_u64 s\[(?:2:3|4:5|6:7|8:9|10:11)\]"), 5, failures)
    check("stage01 descriptor restore", count(early, r"^\s*s_sub_co_u32 s34, s34,"), 4 if stages == 2 else 0, failures)

    labels = re.findall(r"^([.$A-Za-z_][.$A-Za-z0-9_]*):", text, flags=re.MULTILINE)
    duplicate_labels = sorted({label for label in labels if labels.count(label) != 1})
    check("unique labels", duplicate_labels, [], failures)
    print(f"info sha256={sha256(path)}")
    print(f"info maximum logical per-wave TDM window={stages + 2} (limit requested by user: 10)")
    if failures:
        raise SystemExit(f"{path.name}: {len(failures)} static checks failed")


def main() -> None:
    actual = sha256(BASELINE)
    if actual != EXPECTED_BASELINE_SHA256:
        raise SystemExit(
            "baseline checksum changed: "
            f"expected {EXPECTED_BASELINE_SHA256}, got {actual}"
        )
    for args in VARIANTS:
        audit(*args)


if __name__ == "__main__":
    main()
