#!/usr/bin/env python3
"""Build the resident 2+3 diagnostic with ScaleA/ScaleB TDM loads removed."""

from __future__ import annotations

import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / (
    "persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_"
    "all_nt_rt_ab4_scale_half_tdm_full_setup_loop_wait6.s"
)
OUTPUT = HERE / (
    "persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_"
    "all_nt_rt_ab4_no_scale_tdm_full_setup_loop_wait4.s"
)
EXPECTED_SOURCE_SHA256 = (
    "02dffd3a7f6ca015a25d52e8a2273c6cd49deed57bde4f0f758c1a97c7c70c06"
)
SCALE_TDM_PREFIX = (
    "tensor_load_to_lds s[80:83], s[84:91] th:TH_LOAD_NT_RT"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    actual_source_sha256 = sha256(SOURCE)
    if actual_source_sha256 != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "source checksum changed: "
            f"expected {EXPECTED_SOURCE_SHA256}, got {actual_source_sha256}"
        )

    source_lines = SOURCE.read_text(encoding="utf-8").splitlines(keepends=True)
    removed_scale_loads = sum(
        line.lstrip().startswith(SCALE_TDM_PREFIX) for line in source_lines
    )
    if removed_scale_loads != 56:
        raise RuntimeError(
            f"expected 56 Scale TDM loads, got {removed_scale_loads}"
        )

    text = "".join(
        line
        for line in source_lines
        if not line.lstrip().startswith(SCALE_TDM_PREFIX)
    )
    wait6_count = text.count("s_wait_tensorcnt 0x6")
    wait3_count = text.count("s_wait_tensorcnt 0x3")
    if (wait6_count, wait3_count) != (20, 4):
        raise RuntimeError(
            "unexpected tensor-wait counts before retiming: "
            f"wait6={wait6_count}, wait3={wait3_count}"
        )

    text = text.replace("s_wait_tensorcnt 0x6", "s_wait_tensorcnt 0x4")
    text = text.replace("s_wait_tensorcnt 0x3", "s_wait_tensorcnt 0x2")

    expected_counts = {
        "tensor_load_to_lds s[32:35], s[36:43]": 112,
        "tensor_load_to_lds s[44:47], s[48:55]": 112,
        "tensor_load_to_lds s[80:83], s[84:91]": 0,
        "s_wait_tensorcnt 0x4": 20,
        "s_wait_tensorcnt 0x2": 4,
        "s_wait_tensorcnt 0x6": 0,
        "s_wait_tensorcnt 0x3": 0,
    }
    for needle, expected in expected_counts.items():
        actual = text.count(needle)
        if actual != expected:
            raise RuntimeError(
                f"{needle!r}: expected {expected} occurrences, got {actual}"
            )

    OUTPUT.write_text(text, encoding="utf-8", newline="")
    print(f"wrote {OUTPUT}")
    print(f"sha256={sha256(OUTPUT)}")
    print(f"removed_scale_tdm={removed_scale_loads}")
    print(f"wait6_to_wait4={wait6_count}")
    print(f"wait3_to_wait2={wait3_count}")


if __name__ == "__main__":
    main()
