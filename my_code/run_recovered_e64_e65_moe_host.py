#!/usr/bin/env python3
"""Run bounded recovered-source correctness and complete MoE performance checks."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time

REPO = Path("/data/yanguahe/code/wk_sp1/aiter_t256w2x4_e64_20261007")
ROOT = REPO / ".codex_tmp/side_dual_moe_20261019"
sys.path.insert(0, str(REPO))
from my_code.run_gemm1_e64_copy_host import idle_check

ENV = dict(PYTHONPATH=str(REPO), ENABLE_CK="0", AITER_MOE_EXPERT_BALANCE="false",
           AITER_LOG_MORE="1", AITER_USE_GROUPED_GEMM="1", AITER_GROUPED_DEBUG="0",
           AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE="1")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


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


def check_sources():
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    for relative, expected in manifest.items():
        assert hashlib.sha256((REPO / relative).read_bytes()).hexdigest() == expected, relative
    return manifest


def run_one(output, stage, experts, seed, run):
    output.mkdir(parents=True, exist_ok=False)
    command = ["python3", "-u", "my_code/run_recovered_e64_e65_moe.py",
               str(output / "device"), "--stage", stage, "--experts", str(experts),
               "--seed", str(seed)]
    if stage == "verify":
        compare = REPO / (f".codex_tmp/dsr1_e65_gemm1_20261017/parent_seed{seed}.pt"
                          if experts == 65 else
                          f".codex_tmp/side_e64_gemm1_20261018/verify_baseline_seed{seed}_layout_v2/effective.pt")
        assert compare.is_file(), compare
        command += ["--compare", str(compare)]
    docker = ["docker", "exec", "-w", str(REPO)]
    for key, value in ENV.items():
        docker += ["-e", f"{key}={value}"]
    docker += ["dsr1_ep4", "timeout", "--signal=TERM", "--kill-after=5s", "120s", *command]
    (output / "command.json").write_text(json.dumps(docker, indent=2) + "\n")
    with (output / "idle.jsonl").open("w") as gate:
        pre = wait_idle(gate)
        check_sources()
        print(f"START {output.name} stage={stage} E={experts} seed={seed} run={run} {now()}", flush=True)
        started = time.monotonic()
        with (output / "raw.log").open("w") as raw:
            result = subprocess.run(docker, stdout=raw, stderr=subprocess.STDOUT)
        elapsed = time.monotonic() - started
        post = idle_check()
        gate.write(json.dumps(post) + "\n")
        gate.flush()
        live = [pid for pid in post["kfd_owners"] if (Path("/proc") / pid).exists()]
        time.sleep(6)
        settled = idle_check()
        gate.write(json.dumps(settled) + "\n")
    qualified = (pre["idle"] and result.returncode == 0 and not live
                 and post["fuser_rc"] == 1 and not post["gpu_users"].strip()
                 and pre["boot_id"] == post["boot_id"] == settled["boot_id"] and settled["idle"])
    summary_path = output / "device/summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
    record = dict(stage=stage, experts=experts, seed=seed, run=run, returncode=result.returncode,
                  elapsed_seconds=elapsed, timed_out=result.returncode in (124, 137),
                  pre=pre, post=post, settled=settled, live_post_owners=live,
                  ownership_qualified=qualified,
                  timing_valid=stage == "perf" and qualified and summary is not None and summary["passed"],
                  summary=summary)
    (output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    subprocess.run(["sudo", "-n", "chmod", "-R", "a+rwX", str(output)], check=True)
    print(f"END {output.name} rc={result.returncode} elapsed={elapsed:.3f} ownership={qualified}", flush=True)
    if result.returncode or summary is None or not summary["passed"]:
        print((output / "raw.log").read_text()[-7000:], flush=True)
        raise RuntimeError(f"execution/correctness/binding failure at {output}")
    if stage == "perf":
        print(json.dumps({key: summary[key] for key in
                          ("gemm1_us", "gemm2_us", "moe_e2e_us", "accuracy", "binding")}), flush=True)
    else:
        print(json.dumps(dict(passed=True, binding=summary["binding"],
                              accuracy=summary.get("accuracy"), rounds=summary.get("rounds"))), flush=True)
    check_sources()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    parser.add_argument("--stage", choices=("checks", "perf", "all"), default="all")
    args = parser.parse_args()
    assert args.label and all(c.isalnum() or c in "_.-" for c in args.label)
    assert socket.gethostname().startswith("heliosr-1b114-a07-3")
    output = ROOT / args.label
    output.mkdir(parents=True, exist_ok=False)
    manifest = check_sources()
    driver_names = ["my_code/run_recovered_e64_e65_moe.py",
                    "my_code/run_recovered_e64_e65_moe_host.py",
                    "my_code/run_gemm1_e64_copy_host.py", "my_code/run_gemm1_e64_copy.py"]
    metadata = dict(time=now(), hostname=socket.gethostname(), source_manifest=manifest,
                    image=subprocess.check_output(["docker", "inspect", "--format", "{{.Config.Image}}", "dsr1_ep4"], text=True).strip(),
                    head=subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
                    branch=subprocess.check_output(["git", "-C", str(REPO), "branch", "--show-current"], text=True).strip(),
                    driver_sha256={name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
                                   for name in driver_names}, environment=ENV,
                    stage=args.stage, test_timeout_seconds=120, kill_after_seconds=5)
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    checks = []
    groups = {}
    if args.stage in ("all", "checks"):
        for experts in (65, 64):
            for seed in (0, 17):
                checks.append(run_one(output / f"verify_e{experts}_seed{seed}", "verify", experts, seed, 1))
        for experts in (64, 65):
            checks.append(run_one(output / f"stress_e{experts}", "stress", experts, 0, 1))
        (output / "checks_acceptance.json").write_text(
            json.dumps(dict(passed=True, checks=checks), indent=2) + "\n")
    if args.stage in ("all", "perf"):
        for experts in (64, 65):
            attempts = []
            for attempt in range(1, 4):
                rounds = [run_one(output / f"perf_e{experts}_try{attempt}_run{n}",
                                  "perf", experts, 0, n) for n in range(1, 4)]
                attempts.append(rounds)
                if all(record["timing_valid"] for record in rounds):
                    groups[str(experts)] = dict(
                        passed=True, attempt=attempt, attempts=attempts,
                        medians={key: statistics.median(record["summary"][key] for record in rounds)
                                 for key in ("gemm1_us", "gemm2_us", "moe_e2e_us")})
                    print("GROUP_ACCEPTED " + json.dumps(dict(experts=experts, **groups[str(experts)]["medians"])), flush=True)
                    break
                print(f"GROUP_EXCLUDED E={experts} attempt={attempt} ownership only", flush=True)
            else:
                raise RuntimeError(f"three complete E{experts} groups had ownership contamination")
    record = dict(passed=True, time=now(), checks=checks, groups=groups, metadata=metadata)
    (output / "final_acceptance.json").write_text(json.dumps(record, indent=2) + "\n")
    subprocess.run(["sudo", "-n", "chmod", "-R", "a+rwX", str(output)], check=True)
    print("RECOVERED_DUAL_MOE_DONE " + json.dumps({k: v["medians"] for k, v in groups.items()}), flush=True)


if __name__ == "__main__":
    main()
