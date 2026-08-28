#!/usr/bin/env python3
"""Plot interval-average GFX clock frequency for every shader engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input = (
        script_dir
        / "f4gemm_bf16_mxfp4_ABpreShuffle_256x256_4x4_ps.asm.d7-2-gpu1.att"
        / "thread_trace"
        / "simd0"
        / "kernel"
        / "rpf_v3"
        / "ui_output_agent_13454_dispatch_44"
        / "realtime.json"
    )
    parser = argparse.ArgumentParser(
        description="Plot per-SE Fgfx derived from ATT realtime clock pairs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Input realtime.json path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "fgfx_all_se.png",
        help="Output image path (default: my_code/fgfx_all_se.png)",
    )
    parser.add_argument(
        "--reference-hz",
        type=int,
        default=None,
        help="Override metadata.frequency when it is missing or zero",
    )
    return parser.parse_args()


def se_sort_key(name: str) -> int:
    return int(name.removeprefix("SE"))


def main() -> None:
    args = parse_args()
    output = args.output

    with args.input.open("r", encoding="utf-8") as stream:
        data = json.load(stream)

    metadata_hz = int(data["metadata"]["frequency"])
    reference_hz = args.reference_hz if args.reference_hz is not None else metadata_hz
    if reference_hz <= 0:
        raise ValueError(
            "metadata.frequency must be greater than zero; "
            "use --reference-hz only when the reference clock is known"
        )
    reference_source = (
        f"CLI override; metadata={metadata_hz}"
        if args.reference_hz is not None
        else "metadata"
    )

    se_names = sorted(
        (name for name in data if name.startswith("SE")), key=se_sort_key
    )
    if not se_names:
        raise ValueError("No SE samples found")

    global_realtime_start = min(data[name][0][1] for name in se_names)

    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D"]

    for index, se_name in enumerate(se_names):
        samples = data[se_name]
        elapsed_us: list[float] = []
        fgfx_mhz: list[float] = []
        total_shader_cycles = 0
        total_realtime_ticks = 0

        for (shader_0, realtime_0), (shader_1, realtime_1) in zip(
            samples, samples[1:]
        ):
            delta_shader = shader_1 - shader_0
            delta_realtime = realtime_1 - realtime_0
            if delta_realtime <= 0:
                continue

            midpoint_realtime = (realtime_0 + realtime_1) / 2
            elapsed_us.append(
                (midpoint_realtime - global_realtime_start)
                / reference_hz
                * 1.0e6
            )
            fgfx_mhz.append(
                reference_hz * delta_shader / delta_realtime / 1.0e6
            )
            total_shader_cycles += delta_shader
            total_realtime_ticks += delta_realtime

        if not fgfx_mhz:
            continue

        weighted_mean_mhz = (
            reference_hz
            * total_shader_cycles
            / total_realtime_ticks
            / 1.0e6
        )
        ax.plot(
            elapsed_us,
            fgfx_mhz,
            color=colors[index % len(colors)],
            linestyle=line_styles[index % len(line_styles)],
            marker=markers[index % len(markers)],
            markevery=20,
            markersize=3,
            linewidth=1.35,
            alpha=0.85,
            label=f"{se_name} (time-weighted mean {weighted_mean_mhz:.1f} MHz)",
        )

    ax.set_title(
        "ATT-Derived GFX Clock Frequency by Shader Engine\n"
        f"REALTIME reference: {reference_hz / 1.0e6:g} MHz; "
        f"{reference_source}; one value per adjacent sample pair"
    )
    ax.set_xlabel("Elapsed REALTIME (µs)")
    ax.set_ylabel("Interval-average Fgfx (MHz)")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Saved plot to {output}")


if __name__ == "__main__":
    main()
