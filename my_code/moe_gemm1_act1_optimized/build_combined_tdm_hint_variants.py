#!/usr/bin/env python3
"""Build combined near/far-cache temporal-hint variants for input TDM loads."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full.s"
EXPECTED_SOURCE_SHA256 = (
    "6d3b387137b1f5d3de0e95fa8dddb8067e0dae9847ecf773fdc50b5d5ba28bb3"
)
OUTPUT_ALL_NT_RT = HERE / (
    "persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_"
    "all_nt_rt.s"
)


DEFAULT_LOAD = re.compile(
    r"^(\s*tensor_load_to_lds\s+s\[32:35\],\s+s\[36:43\])"
    r"(\s*(?:;.*)?)$",
    flags=re.MULTILINE,
)
EXACT_NT = re.compile(r"th:TH_LOAD_NT(?=\s|;|$)")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_default(source: str, hint: str) -> str:
    text, count = DEFAULT_LOAD.subn(rf"\1 th:TH_LOAD_{hint}\2", source)
    if count != 39:
        raise RuntimeError(f"expected 39 default loads, replaced {count}")
    return text


def replace_nt(source: str, hint: str) -> str:
    text, count = EXACT_NT.subn(f"th:TH_LOAD_{hint}", source)
    if count != 21:
        raise RuntimeError(f"expected 21 NT loads, replaced {count}")
    return text


def validate(source: str, text: str) -> None:
    for token in (
        "tensor_load_to_lds",
        "tensor_store_from_lds",
        "v_wmma_scale_f32_32x16x128_f4",
        "s_wait_tensorcnt",
        "s_barrier_signal",
        "s_barrier_wait",
    ):
        if text.count(token) != source.count(token):
            raise RuntimeError(f"instruction count changed for {token}")


def main() -> None:
    actual = sha256(SOURCE)
    if actual != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "stable-winner source checksum changed: "
            f"expected {EXPECTED_SOURCE_SHA256}, got {actual}"
        )
    source = SOURCE.read_text(encoding="utf-8")

    all_nt_rt = replace_nt(replace_default(source, "NT_RT"), "NT_RT")
    validate(source, all_nt_rt)
    if all_nt_rt.count("th:TH_LOAD_NT_RT") != 60:
        raise RuntimeError("all-NT_RT candidate does not contain 60 hinted loads")
    OUTPUT_ALL_NT_RT.write_text(all_nt_rt, encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT_ALL_NT_RT}")
    print(f"sha256={sha256(OUTPUT_ALL_NT_RT)}")

if __name__ == "__main__":
    main()
