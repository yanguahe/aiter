#!/usr/bin/env python3
"""Create the self-contained repository snapshot used by the benchmarks.

Only committed Git content is copied. By default an existing snapshot remains
pinned to its recorded commit; use --commit HEAD to intentionally refresh it.
This avoids reading uncommitted changes or silently following a moving branch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SNAPSHOT = HERE / "repo_snapshot"
SOURCE_PATHS = (
    "aiter",
    "csrc",
    "op_tests/test_flydsl_grouped_gemm_gfx1250.py",
)
SNAPSHOT_METADATA = {"SOURCE_COMMIT", "SNAPSHOT_MANIFEST.json"}
PAYLOAD_DIGEST_FORMAT = "canonical-lf-v1"
GIT_BINARY_PROBE_BYTES = 8000


def _run_git(*args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(REPO), *args),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with tarfile.open(archive, "r") as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if os.path.commonpath((destination_resolved, target)) != str(
                destination_resolved
            ):
                raise RuntimeError(f"unsafe archive member: {member.name!r}")
        handle.extractall(destination)


def _canonical_payload(payload: bytes) -> bytes:
    """Match Git's text checkout semantics without requiring Git at verify time."""
    if b"\0" in payload[:GIT_BINARY_PROBE_BYTES]:
        return payload
    return payload.replace(b"\r\n", b"\n")


def _tree_digest(
    root: Path, relative_paths: list[str] | None = None
) -> tuple[str, int, int, list[str]]:
    digest = hashlib.sha256()
    count = 0
    total = 0
    payload_paths: list[str] = []
    if relative_paths is None:
        files = sorted(
            (
                (path.relative_to(root).as_posix(), path)
                for path in root.rglob("*")
                if path.is_file()
            ),
            key=lambda item: item[0],
        )
    else:
        files = [(relative, root / relative) for relative in relative_paths]
    for relative_text, path in files:
        if relative_text in SNAPSHOT_METADATA:
            continue
        if not path.is_file():
            raise RuntimeError(f"snapshot payload file is missing: {relative_text}")
        relative = relative_text.encode("utf-8")
        payload = _canonical_payload(path.read_bytes())
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
        count += 1
        total += len(payload)
        payload_paths.append(relative_text)
    return digest.hexdigest(), count, total, payload_paths


def _verify_snapshot() -> None:
    manifest_path = SNAPSHOT / "SNAPSHOT_MANIFEST.json"
    source_commit_path = SNAPSHOT / "SOURCE_COMMIT"
    if not manifest_path.is_file() or not source_commit_path.is_file():
        raise SystemExit(f"snapshot metadata is missing under {SNAPSHOT}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest_format = manifest.get("payload_digest_format")
    if digest_format != PAYLOAD_DIGEST_FORMAT:
        raise SystemExit(
            "snapshot manifest uses unsupported payload digest format: "
            f"{digest_format!r}; expected {PAYLOAD_DIGEST_FORMAT!r}"
        )
    payload_files = [str(item) for item in manifest.get("payload_files", ())]
    if not payload_files:
        raise SystemExit("snapshot manifest has no payload_files list")
    try:
        digest, file_count, total_bytes, _ = _tree_digest(
            SNAPSHOT, payload_files
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    expected = (
        str(manifest["payload_tree_sha256"]),
        int(manifest["payload_file_count"]),
        int(manifest["payload_bytes"]),
    )
    actual = (digest, file_count, total_bytes)
    if actual != expected:
        raise SystemExit(f"snapshot verification failed: expected={expected}, actual={actual}")
    source_commit = source_commit_path.read_text(encoding="ascii").strip()
    if source_commit != manifest["source_commit"]:
        raise SystemExit("SOURCE_COMMIT differs from SNAPSHOT_MANIFEST.json")
    print(f"snapshot verified: {SNAPSHOT}")
    print(f"source commit: {source_commit}")
    print(f"payload files: {file_count}")
    print(f"payload bytes: {total_bytes}")
    print(f"payload tree sha256: {digest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify the existing snapshot without reading the repository",
    )
    parser.add_argument(
        "--commit",
        help=(
            "committed revision to copy; by default reuse the existing "
            "SOURCE_COMMIT, or HEAD when creating the first snapshot"
        ),
    )
    args = parser.parse_args()
    if args.verify:
        _verify_snapshot()
        return

    if SNAPSHOT.parent.resolve() != HERE.resolve():
        raise RuntimeError(f"refusing unsafe snapshot path: {SNAPSHOT}")

    revision = args.commit
    existing_commit = SNAPSHOT / "SOURCE_COMMIT"
    if revision is None and existing_commit.is_file():
        revision = existing_commit.read_text(encoding="ascii").strip()
    if not revision:
        revision = "HEAD"
    commit = _run_git("rev-parse", revision)
    with tempfile.TemporaryDirectory(prefix=".repo_snapshot_", dir=HERE) as temp:
        temp_root = Path(temp)
        archive = temp_root / "head.tar"
        tree = temp_root / "tree"
        tree.mkdir()
        subprocess.run(
            (
                "git",
                "-C",
                str(REPO),
                "archive",
                "--format=tar",
                f"--output={archive}",
                commit,
                "--",
                *SOURCE_PATHS,
            ),
            check=True,
        )
        _safe_extract(archive, tree)

        digest, file_count, total_bytes, payload_files = _tree_digest(tree)
        (tree / "SOURCE_COMMIT").write_text(commit + "\n", encoding="ascii")
        manifest = {
            "source_commit": commit,
            "source_paths": list(SOURCE_PATHS),
            "payload_digest_format": PAYLOAD_DIGEST_FORMAT,
            "payload_file_count": file_count,
            "payload_bytes": total_bytes,
            "payload_tree_sha256": digest,
            "payload_files": payload_files,
        }
        (tree / "SNAPSHOT_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if SNAPSHOT.exists():
            shutil.rmtree(SNAPSHOT)
        # os.replace() cannot move a non-empty directory on some Windows
        # filesystems. shutil.move() preserves the extracted tree and works on
        # both the Windows development checkout and Linux test hosts.
        shutil.move(str(tree), str(SNAPSHOT))

    print(f"snapshot: {SNAPSHOT}")
    print(f"source commit: {commit}")
    print(f"payload files: {file_count}")
    print(f"payload bytes: {total_bytes}")
    print(f"payload tree sha256: {digest}")


if __name__ == "__main__":
    main()
