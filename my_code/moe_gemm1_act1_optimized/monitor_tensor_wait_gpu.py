#!/usr/bin/env python3
"""Host-side KFD monitor; SIGINT only this run's verified container process groups.

Run as root on the host. The benchmark must inherit AITER_TW_RUN_TOKEN.
No foreign process is signaled, and no driver/device setting is modified.
This augments, rather than replaces, all-GPU idle checks and correctness gates.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import re
import signal
import time


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def identity(pid):
    # /proc/<pid>/environ can become empty during exit, before the KFD entry
    # disappears. Keep verified ownership for this exact PID incarnation.
    stat = (Path("/proc") / str(pid) / "stat").read_text()
    return pid, int(stat.rsplit(")", 1)[1].split()[19])


def tagged(pid, container, token):
    proc = Path("/proc") / str(pid)
    if container not in (proc / "cgroup").read_text():
        return False
    needle = ("AITER_TW_RUN_TOKEN=" + token).encode()
    return needle in (proc / "environ").read_bytes().split(b"\0")


def interrupt_tagged_groups(container, token, already):
    sent = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            pid = int(entry.name)
            if not tagged(pid, container, token):
                continue
            group = os.getpgid(pid)
            # Verify the group leader separately: never signal a shared shell
            # group, an unrelated test, or the monitor's own process group.
            if group in already or group == os.getpgrp() or not tagged(group, container, token):
                continue
            os.killpg(group, signal.SIGINT)
            already.add(group)
            sent.append(group)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return sent


def completion_code(path):
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - 512))
            last = handle.read().decode("utf-8", errors="replace").strip().splitlines()[-1]
        match = re.fullmatch(r"EXIT_CODE=(\d+)", last)
        return int(match[1]) if match else None
    except (FileNotFoundError, IndexError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--completion-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=600)
    args = parser.parse_args()
    if os.geteuid() != 0 or not re.fullmatch(r"[0-9a-f]{64}", args.container_id):
        raise SystemExit("Run as host root with the exact inspected 64-character container ID")
    if not args.token or completion_code(args.completion_log) is not None:
        raise SystemExit("Use a unique nonempty token and a new, unfinished completion log")
    started = time.monotonic()
    report = {"started_utc": utc_now(), "container_id": args.container_id,
              "token": args.token, "interval_seconds": 0.5, "samples": 0,
              "foreign_processes": {}, "errors": [], "interrupt_groups": [], "events": []}
    previous = None
    signaled = set()
    known_owned = set()
    finished = None
    while time.monotonic() - started < args.max_seconds:
        finished = completion_code(args.completion_log)
        if finished is not None:
            break
        current = []
        try:
            if not Path("/dev/kfd").is_char_device():
                raise RuntimeError("Host /dev/kfd disappeared")
            for entry in Path("/sys/kernel/debug/kfd/proc").iterdir():
                if not entry.name.isdigit():
                    continue
                pid = int(entry.name)
                try:
                    key = identity(pid)
                    ours = key in known_owned or tagged(pid, args.container_id, args.token)
                    if not ours and args.container_id in (Path("/proc") / str(pid) / "cgroup").read_text():
                        group = os.getpgid(pid)
                        ours = tagged(group, args.container_id, args.token)
                    if ours:
                        known_owned.add(key)
                    comm = (Path("/proc") / str(pid) / "comm").read_text().strip()
                except (FileNotFoundError, ProcessLookupError):
                    continue
                item = {"pid": pid, "ours": ours, "comm": comm}
                current.append(item)
                if not ours:
                    report["foreign_processes"].setdefault(str(pid), {**item, "first_seen_utc": utc_now()})
        except (OSError, RuntimeError) as error:
            message = str(error)
            if message not in report["errors"]:
                report["errors"].append(message)
        current.sort(key=lambda item: item["pid"])
        if current != previous:
            event = {"utc": utc_now(), "processes": current}
            report["events"].append(event)
            print(json.dumps(event), flush=True)
            previous = current
        report["samples"] += 1
        if (report["foreign_processes"] or report["errors"]) and not signaled:
            sent = interrupt_tagged_groups(args.container_id, args.token, signaled)
            if sent:
                report["interrupt_groups"].extend(sent)
                print(json.dumps({"utc": utc_now(), "SIGINT_own_verified_groups": sent}), flush=True)
        if signaled:
            alive = False
            for group in signaled:
                try:
                    os.killpg(group, 0)
                    alive = True
                except ProcessLookupError:
                    pass
            if not alive:
                report["stop_reason"] = "interrupted verified groups have exited; shell may omit completion marker"
                break
        time.sleep(0.5)
    report.update({"finished_utc": utc_now(), "elapsed_seconds": time.monotonic() - started,
                   "completion_exit_code": finished,
                   "kfd_gate_passed": finished == 0 and not report["foreign_processes"] and not report["errors"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output.chmod(0o666)
    print(json.dumps({key: value for key, value in report.items() if key != "events"}), flush=True)
    return 0 if report["kfd_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
