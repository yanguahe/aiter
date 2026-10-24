#!/usr/bin/env python3
"""Run the original grouped-MoE benchmark after a host-wide idle gate."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import time


REPO = Path("/data/yanguahe/code/wk_sp1/aiter_t256w2x4_e64_20261007")
ARTIFACTS = REPO / ".codex_tmp/dsr1_e65_gemm1_20261017"
SOURCES = (
    "aiter/ops/flydsl/grouped_moe_gfx1250.py",
    "aiter/ops/flydsl/grouped_gemm_mxfp4.py",
    "aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_prefill_persistent.py",
    "my_code/test_flydsl_grouped_gemm_gfx1250.py",
)
ENV = {
    "PYTHONPATH": str(REPO),
    "ENABLE_CK": "0",
    "AITER_MOE_EXPERT_BALANCE": "false",
    "AITER_LOG_MORE": "1",
    "AITER_USE_GROUPED_GEMM": "1",
    "AITER_GROUPED_DEBUG": "0",
    "AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE": "1",
}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def idle_check():
    """Check all KFD nodes and aggregate physical GPU busy/VRAM counters.

    XCP DRM devices share their parent GPU's counters. Keep both the complete
    node inventory and the physical PCI counters in the evidence.
    """
    kfd = Path("/sys/class/kfd/kfd")
    owners = sorted(p.name for p in (kfd / "proc").iterdir())
    nodes = []
    for node in sorted((kfd / "topology/nodes").iterdir()):
        gid = int((node / "gpu_id").read_text())
        if gid:
            nodes.append({"node": node.name, "gpu_id": gid,
                          "properties": (node / "properties").read_text()})
    devices = []
    drm = []
    for card in sorted(Path("/sys/class/drm").iterdir()):
        if not re.fullmatch(r"card\d+", card.name):
            continue
        device = card / "device"
        drm.append({"card": card.name, "device": str(device.resolve())})
        busy = device / "gpu_busy_percent"
        vram = device / "mem_info_vram_used"
        if busy.is_file() and vram.is_file():
            devices.append({"card": card.name,
                            "device": str(device.resolve()),
                            "busy": int(busy.read_text()),
                            "vram": int(vram.read_text())})
    user_script = Path("/data/yanguahe/code/gpu_users.sh")
    users = subprocess.run(["bash", str(user_script), "--plain"],
                           capture_output=True, text=True, check=True)
    fuser = subprocess.run(["sudo", "-n", "fuser", "-v", "/dev/kfd"],
                           capture_output=True, text=True)
    if fuser.returncode not in (0, 1):
        raise RuntimeError(f"fuser failed: {fuser.stderr}")
    idle = (bool(nodes) and bool(devices) and not owners
            and fuser.returncode == 1 and not users.stdout.strip()
            and all(d["busy"] == 0 and d["vram"] < 512 * 2**20
                    for d in devices))
    return {"time": now(), "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "kfd_owners": owners, "nodes": nodes, "drm": drm,
            "devices": devices, "gpu_users": users.stdout,
            "fuser_rc": fuser.returncode,
            "fuser": fuser.stdout + fuser.stderr, "idle": idle}


def wait_idle(log):
    while True:
        state = idle_check()
        log.write(json.dumps(state) + "\n")
        log.flush()
        print(json.dumps({k: state[k] for k in ("time", "kfd_owners", "devices", "idle")}), flush=True)
        if state["idle"]:
            return state
        time.sleep(6)


def main():
    global REPO, ARTIFACTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    parser.add_argument("--repo", type=Path, default=REPO,
                        help="Explicit isolated shared source directory on this host.")
    parser.add_argument("--container", default="dsr1_ep4")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--experts", type=int, default=65)
    parser.add_argument("--topk", type=int, default=9)
    parser.add_argument("--shared-experts", type=int, default=1)
    parser.add_argument("--scenario", choices=("bench", "verify", "kernel"), default="bench")
    parser.add_argument("--env", action="append", default=[],
                        help="Additional candidate environment variable KEY=VALUE.")
    parser.add_argument("--gemm1-expert-scheduling-mode", action="store_true",
                        help="Run the unchanged benchmark through the G1 mode-2 driver.")
    parser.add_argument("--gemm1-b-th", type=int, choices=(0, 1, 2, 4, 5, 6),
                        help="Use the existing B-only TDM hint on GEMM1, leaving GEMM2 unchanged.")
    parser.add_argument("--command", nargs=argparse.REMAINDER,
                        help="Diagnostic container command; timing is not a formal benchmark.")
    parser.add_argument("--correctness-only", action="store_true",
                        help="Run an explicit diagnostic despite GPU load; all timing is invalid.")
    args = parser.parse_args()
    if args.runs < 1 or not 0 <= args.shared_experts < args.topk <= args.experts:
        parser.error("require runs >= 1 and 0 <= shared-experts < topk <= experts")
    REPO = args.repo.resolve()
    ARTIFACTS = REPO / ".codex_tmp/dsr1_e65_gemm1_20261017"
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if args.correctness_only and not args.command:
        parser.error("--correctness-only requires an explicit --command")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.label):
        parser.error("label must be a plain artifact directory name")
    output = ARTIFACTS / args.label
    output.mkdir(parents=True, exist_ok=False)
    for src in SOURCES:
        dst = output / "sources" / src
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((REPO / src).read_bytes())
    source_hashes = {src: hashlib.sha256((REPO / src).read_bytes()).hexdigest()
                     for src in SOURCES}
    command = args.command or [
        "python3", "-u", "my_code/test_flydsl_grouped_gemm_gfx1250.py",
        "--scenario", args.scenario, "--data-format", "a4w4", "--act", "silu",
        "--no-bias", "--no-check-aot-cache", "--experts", str(args.experts), "--tokens", "1536",
        "--topk", str(args.topk), "--shared-experts", str(args.shared_experts), "--model-dim", "7168",
        "--inter-dim", "2048", "--iters", "20",
    ]
    if args.gemm1_expert_scheduling_mode:
        if args.command or args.gemm1_b_th is not None:
            parser.error("--gemm1-expert-scheduling-mode cannot use --command or --gemm1-b-th")
        command = ["python3", "-u", "my_code/run_dsr1_e65_gemm1_mode2.py",
                   "--script", "bench", *command[3:]]
    if args.gemm1_b_th is not None:
        if args.command:
            parser.error("--gemm1-b-th cannot use --command")
        command = ["python3", "-u", "my_code/run_dsr1_e65_gemm1_bth.py",
                   "--script", "bench", "--tdm-b-th", str(args.gemm1_b_th), *command[3:]]
    environment = dict(ENV)
    environment["PYTHONPATH"] = str(REPO)
    for item in args.env:
        key, value = item.split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            parser.error(f"invalid environment key {key!r}")
        environment[key] = value
    docker_command = ["docker", "exec", "-w", str(REPO)]
    for key, value in environment.items():
        docker_command += ["-e", f"{key}={value}"]
    # Bound each of our container test process groups, including runtime/JIT
    # loading. GNU timeout signals only its own group; unrelated GPU users are
    # never part of this command. The user requires stopping tests at 120 s.
    docker_command += [args.container, "timeout", "--signal=TERM", "--kill-after=5s",
                       "120s", *command]
    meta = {"time": now(), "host": socket.gethostname(), "repo": str(REPO),
            "container": args.container, "sources": source_hashes,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "workload": None if args.command else {"experts": args.experts, "tokens": 1536,
                         "topk": args.topk, "shared_experts": args.shared_experts,
                         "model_dim": 7168, "inter_dim": 2048, "iters": 20},
            "command": docker_command, "diagnostic": bool(args.command),
            "gemm1_expert_scheduling_mode": args.gemm1_expert_scheduling_mode,
            "gemm1_b_th": args.gemm1_b_th,
            "test_timeout_seconds": 120,
            "timeout_kill_after_seconds": 5,
            "correctness_only": args.correctness_only,
            "timing_valid": False if args.correctness_only else None}
    (output / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    for run in range(1, args.runs + 1):
        with (output / f"idle_{run}.jsonl").open("w") as gate:
            if args.correctness_only:
                pre = idle_check()
                gate.write(json.dumps(pre) + "\n")
                gate.flush()
                print("CORRECTNESS ONLY: GPU load may exist; discard all timing.", flush=True)
            else:
                pre = wait_idle(gate)
            print(f"START {args.label} run={run} {now()}", flush=True)
            with (output / f"run_{run}.log").open("w") as log:
                result = subprocess.run(docker_command, stdout=log, stderr=subprocess.STDOUT)
            post = idle_check()
            gate.write(json.dumps(post) + "\n")
            gate.flush()
            # The driver can retain a dead process's KFD entry briefly after
            # Python exits. Preserve the immediate snapshot and distinguish it
            # from a still-live owner; wait only after measurement has finished.
            live_post_owners = [pid for pid in post["kfd_owners"]
                                if (Path("/proc") / pid).exists()]
            time.sleep(6)
            settled = idle_check()
            gate.write(json.dumps(settled) + "\n")
            gate.flush()
        qualified = (not args.correctness_only
                     and result.returncode == 0 and not live_post_owners
                     and post["fuser_rc"] == 1 and not post["gpu_users"].strip()
                     and pre["boot_id"] == post["boot_id"] == settled["boot_id"]
                     and settled["idle"])
        record = {"run": run, "returncode": result.returncode,
                  "timed_out": result.returncode in (124, 137),
                  "correctness_only": args.correctness_only,
                  "ownership_qualified": qualified, "post": post,
                  "live_post_owners": live_post_owners, "settled": settled}
        (output / f"result_{run}.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"END {args.label} run={run} rc={result.returncode} ownership_qualified={qualified}", flush=True)
        print((output / f"run_{run}.log").read_text()[-12000:], flush=True)
        if result.returncode:
            break
    subprocess.run(["chmod", "-R", "a+rwX", str(output)], check=True)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
