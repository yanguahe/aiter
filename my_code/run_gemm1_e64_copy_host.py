#!/usr/bin/env python3
"""Run isolated E64 GEMM1 tests in dsr1_ep4 with whole-host ownership checks."""

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
ARTIFACTS = REPO / ".codex_tmp/side_e64_gemm1_20261018"
ENV = {
    "PYTHONPATH": str(REPO),
    "ENABLE_CK": "0",
    "AITER_MOE_EXPERT_BALANCE": "false",
    "AITER_LOG_MORE": "1",
    "AITER_USE_GROUPED_GEMM": "1",
    "AITER_GROUPED_DEBUG": "0",
    "AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE": "1",
}
SHAPE = ["--experts", "64", "--tokens", "1536", "--topk", "8",
         "--shared-experts", "0", "--model-dim", "7168", "--inter-dim", "2048"]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def idle_check():
    kfd = Path("/sys/class/kfd/kfd")
    owners = sorted(p.name for p in (kfd / "proc").iterdir())
    nodes = []
    for node in sorted((kfd / "topology/nodes").iterdir()):
        gpu_id = int((node / "gpu_id").read_text())
        if gpu_id:
            nodes.append(dict(node=node.name, gpu_id=gpu_id,
                              properties=(node / "properties").read_text()))
    devices = []
    for card in sorted(Path("/sys/class/drm").iterdir()):
        if not re.fullmatch(r"card\d+", card.name):
            continue
        device = card / "device"
        busy = device / "gpu_busy_percent"
        vram = device / "mem_info_vram_used"
        if busy.is_file() and vram.is_file():
            devices.append(dict(card=card.name, device=str(device.resolve()),
                                busy=int(busy.read_text()), vram=int(vram.read_text())))
    script = Path("/data/yanguahe/code/gpu_users.sh")
    users = subprocess.run(["bash", str(script), "--plain"],
                           capture_output=True, text=True, check=True) if script.is_file() else None
    fuser = subprocess.run(["sudo", "-n", "fuser", "-v", "/dev/kfd"],
                           capture_output=True, text=True)
    if fuser.returncode not in (0, 1):
        raise RuntimeError(f"fuser failed: {fuser.stderr}")
    idle = (bool(nodes) and bool(devices) and not owners and fuser.returncode == 1
            and (users is None or not users.stdout.strip())
            and all(d["busy"] == 0 and d["vram"] < 512 * 2**20 for d in devices))
    return dict(time=now(), boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                kfd_owners=owners, nodes=nodes, devices=devices,
                gpu_users=users.stdout if users else "script absent; fuser/sysfs fallback",
                fuser_rc=fuser.returncode, fuser=fuser.stdout + fuser.stderr, idle=idle)


def wait_idle(log):
    while True:
        state = idle_check()
        log.write(json.dumps(state) + "\n")
        log.flush()
        print(json.dumps({key: state[key] for key in
                          ("time", "kfd_owners", "devices", "idle")}), flush=True)
        if state["idle"]:
            return state
        time.sleep(6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    parser.add_argument("--stage", choices=("compile", "verify", "perf", "diagnose", "isolated"), required=True)
    parser.add_argument("--variant", choices=("baseline", "copy"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--compiled-reference", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.label):
        parser.error("label must be a plain artifact directory name")
    if args.stage not in ("perf", "isolated") and args.runs != 1:
        parser.error("compile/verify runs are individual bounded commands")
    if args.stage == "verify" and args.variant == "copy" and args.compare is None:
        parser.error("copied GEMM1 verification requires a saved E64 baseline")
    output = ARTIFACTS / args.label
    output.mkdir(parents=True, exist_ok=False)
    sources = json.loads((ARTIFACTS / "source_manifest.json").read_text())
    for relative, expected in sources.items():
        if hashlib.sha256((REPO / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"source SHA mismatch: {relative}")
    tools = ["my_code/run_gemm1_e64_copy.py", "my_code/compile_gemm1_e64_copy.py",
             "my_code/run_gemm1_e64_copy_host.py"]
    driver_sha = {relative: hashlib.sha256((REPO / relative).read_bytes()).hexdigest()
                  for relative in tools}
    metadata = dict(time=now(), host=socket.gethostname(), repo=str(REPO),
                    container="dsr1_ep4", stage=args.stage, variant=args.variant,
                    source_manifest=sources, driver_sha256=driver_sha,
                    test_timeout_seconds=120, timeout_kill_after_seconds=5)
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for run in range(1, args.runs + 1):
        binding = output / f"binding_{run}.json"
        if args.stage == "isolated":
            command = ["python3", "-u", "my_code/bench_gemm1_e64_copy_isolated.py",
                       str(output / f"isolated_{run}"), "--variant", args.variant]
        elif args.stage == "diagnose":
            command = ["python3", "-u", "my_code/diagnose_gemm1_e64_copy.py",
                       str(output / "diagnostic")]
        elif args.stage == "compile":
            command = ["python3", "-u", "my_code/compile_gemm1_e64_copy.py",
                       str(output / "compiled"), "--variant", args.variant]
        else:
            command = ["python3", "-u", "my_code/run_gemm1_e64_copy.py",
                       "--variant", args.variant, "--script",
                       "verify" if args.stage == "verify" else "bench",
                       "--binding-output", str(binding)]
            if args.stage == "verify":
                command += SHAPE + ["--seed", str(args.seed), "--replays", "3",
                                    "--dynamic-replays", "3"]
                command += (["--compare", str(args.compare)] if args.compare else
                            ["--save", str(output / "effective.pt")])
            else:
                command += ["--scenario", "bench", "--data-format", "a4w4",
                            "--act", "silu", "--no-bias", "--no-check-aot-cache",
                            *SHAPE, "--iters", "20"]
        docker_command = ["docker", "exec", "-w", str(REPO)]
        for key, value in ENV.items():
            docker_command += ["-e", f"{key}={value}"]
        docker_command += ["dsr1_ep4", "timeout", "--signal=TERM", "--kill-after=5s",
                           "120s", *command]
        (output / f"command_{run}.json").write_text(json.dumps(docker_command, indent=2) + "\n")
        with (output / f"idle_{run}.jsonl").open("w") as gate:
            pre = wait_idle(gate)
            print(f"START {args.label} run={run} {now()}", flush=True)
            with (output / f"run_{run}.log").open("w") as log:
                result = subprocess.run(docker_command, stdout=log, stderr=subprocess.STDOUT)
            post = idle_check()
            gate.write(json.dumps(post) + "\n")
            gate.flush()
            live_post_owners = [pid for pid in post["kfd_owners"] if (Path("/proc") / pid).exists()]
            time.sleep(6)
            settled = idle_check()
            gate.write(json.dumps(settled) + "\n")
            gate.flush()
        qualified = (result.returncode == 0 and not live_post_owners and post["fuser_rc"] == 1
                     and not post["gpu_users"].strip()
                     and pre["boot_id"] == post["boot_id"] == settled["boot_id"]
                     and settled["idle"])
        binding_valid = None
        if result.returncode == 0 and args.stage in ("verify", "perf"):
            observed = json.loads(binding.read_text())
            compile_summary = args.compiled_reference or (
                ARTIFACTS / f"compile_{args.variant}_th6_v3/compiled/summary.json")
            expected = json.loads(compile_summary.read_text())["elf_sha256"]
            binding_valid = (observed["passed"] and observed["variant"] == args.variant
                             and observed["capacity_rows"] == 28672
                             and observed["source_manifest"] == sources
                             and {record["elf_sha256"] for record in observed["records"]} == {expected})
        record = dict(run=run, returncode=result.returncode,
                      timed_out=result.returncode in (124, 137),
                      ownership_qualified=qualified, binding_valid=binding_valid,
                      timing_valid=(args.stage == "perf" and qualified and binding_valid)
                                   or (args.stage == "isolated" and qualified),
                      post=post, live_post_owners=live_post_owners, settled=settled)
        (output / f"result_{run}.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"END {args.label} run={run} rc={result.returncode} ownership_qualified={qualified} binding_valid={binding_valid}", flush=True)
        print((output / f"run_{run}.log").read_text()[-12000:], flush=True)
        subprocess.run(["sudo", "-n", "chmod", "-R", "a+rwX", str(output)], check=True)
        if result.returncode:
            raise SystemExit(result.returncode)
        if args.stage in ("verify", "perf") and not binding_valid:
            raise RuntimeError("actual dispatched GEMM1 ELF differs from accepted compile")


if __name__ == "__main__":
    main()
