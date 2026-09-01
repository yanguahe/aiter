#!/usr/bin/env python3
"""Add a generated Markdown table of contents to *.md files in a directory."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


TOC_START = "<!-- markdown-toc-generator:start -->"
TOC_END = "<!-- markdown-toc-generator:end -->"
SPECIAL_ANCHOR_CHARS = set("&`,()>")


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
EXPLICIT_ANCHOR_RE = re.compile(r"""^\s*<a\s+[^>]*id=["']([^"']+)["'][^>]*>\s*</a>\s*$""")
HTML_TAG_RE = re.compile(r"<[^>]+>")
CODE_SPAN_RE = re.compile(r"`([^`]*)`")
LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
PUNCT_RE = re.compile(r"[^\w\s\u4e00-\u9fff-]")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Heading:
    level: int
    title: str
    anchor: str
    line_idx: int


def strip_front_matter(lines: list[str]) -> int:
    if not lines or lines[0].strip() != "---":
        return 0

    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            return idx + 1

    return 0


def normalize_heading_title(raw_title: str) -> str:
    title = raw_title.strip()
    title = CODE_SPAN_RE.sub(r"\1", title)
    title = LINK_RE.sub(r"\1", title)
    title = HTML_TAG_RE.sub("", title)
    return title.strip()


def github_like_anchor(title: str, used: dict[str, int]) -> str:
    anchor = normalize_heading_title(title).lower()
    anchor = PUNCT_RE.sub("", anchor)
    anchor = WHITESPACE_RE.sub("-", anchor.strip())

    count = used.get(anchor, 0)
    used[anchor] = count + 1
    if count:
        return f"{anchor}-{count}"

    return anchor


def title_needs_explicit_anchor(raw_title: str) -> bool:
    return any(ch in raw_title for ch in SPECIAL_ANCHOR_CHARS)


def collect_explicit_anchor_ids(lines: list[str]) -> set[str]:
    ids: set[str] = set()
    for line in lines:
        match = EXPLICIT_ANCHOR_RE.match(line)
        if match:
            ids.add(match.group(1))
    return ids


def get_previous_explicit_anchor(lines: list[str], heading_idx: int) -> str | None:
    if heading_idx <= 0:
        return None

    match = EXPLICIT_ANCHOR_RE.match(lines[heading_idx - 1])
    if not match:
        return None

    return match.group(1)


def stable_explicit_anchor(title: str, used_ids: set[str]) -> str:
    slug = normalize_heading_title(title).lower()
    slug = slug.replace(".", "-")
    slug = PUNCT_RE.sub("", slug)
    slug = WHITESPACE_RE.sub("-", slug.strip("- "))
    if not slug:
        slug = "section"

    base = f"sec-{slug}"
    candidate = base
    suffix = 1
    while candidate in used_ids:
        suffix += 1
        candidate = f"{base}-{suffix}"

    used_ids.add(candidate)
    return candidate


def collect_headings(
    lines: list[str], min_level: int, max_level: int
) -> tuple[list[Heading], dict[int, str]]:
    headings: list[Heading] = []
    github_anchor_used: dict[str, int] = {}
    explicit_anchor_ids = collect_explicit_anchor_ids(lines)
    anchors_to_insert: dict[int, str] = {}
    in_fenced_block = False
    in_generated_toc = False

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == TOC_START:
            in_generated_toc = True
            continue
        if stripped == TOC_END:
            in_generated_toc = False
            continue
        if in_generated_toc:
            continue

        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fenced_block = not in_fenced_block
            continue

        if in_fenced_block:
            continue

        match = HEADING_RE.match(line)
        if not match:
            continue

        level = len(match.group(1))
        raw_title = match.group(2)
        title = normalize_heading_title(raw_title)
        if title.lower() == "table of contents":
            continue

        anchor = github_like_anchor(title, github_anchor_used)
        if min_level <= level <= max_level and title_needs_explicit_anchor(raw_title):
            previous_anchor = get_previous_explicit_anchor(lines, idx)
            if previous_anchor:
                anchor = previous_anchor
            else:
                anchor = stable_explicit_anchor(title, explicit_anchor_ids)
                anchors_to_insert[idx] = anchor

        headings.append(Heading(level=level, title=title, anchor=anchor, line_idx=idx))

    return headings, anchors_to_insert


def render_toc(headings: list[Heading], min_level: int, max_level: int) -> str:
    included = [heading for heading in headings if min_level <= heading.level <= max_level]
    if not included:
        return ""

    base_level = min(heading.level for heading in included)
    lines = [TOC_START, "## Table of Contents", ""]

    for heading in included:
        indent = "  " * (heading.level - base_level)
        lines.append(f"{indent}- [{heading.title}](#{heading.anchor})")

    lines.extend(["", TOC_END, ""])
    return "\n".join(lines)


def has_generated_toc(text: str) -> bool:
    return TOC_START in text and TOC_END in text


def find_generated_toc_range(lines: list[str]) -> tuple[int, int] | None:
    start_idx = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == TOC_START:
            start_idx = idx
        elif stripped == TOC_END and start_idx is not None:
            return start_idx, idx

    return None


def generated_toc_matches(lines: list[str], toc: str) -> bool:
    toc_range = find_generated_toc_range(lines)
    if toc_range is None:
        return False

    start_idx, end_idx = toc_range
    existing_toc = "\n".join(lines[start_idx : end_idx + 1]).strip()
    return existing_toc == toc.strip()


def find_insert_index(lines: list[str]) -> int:
    start = strip_front_matter(lines)

    for idx in range(start, len(lines)):
        match = HEADING_RE.match(lines[idx])
        if match and len(match.group(1)) == 1:
            insert_idx = idx + 1
            while insert_idx < len(lines) and not lines[insert_idx].strip():
                insert_idx += 1
            return insert_idx

    return start


def build_new_lines(
    lines: list[str],
    toc: str,
    anchors_to_insert: dict[int, str],
    insert_idx: int | None = None,
    replace_range: tuple[int, int] | None = None,
) -> list[str]:
    new_lines: list[str] = []
    idx = 0

    while idx < len(lines):
        if replace_range is not None and idx == replace_range[0]:
            new_lines.extend(toc.splitlines())
            idx = replace_range[1] + 1
            continue

        if insert_idx is not None and idx == insert_idx:
            if new_lines and new_lines[-1].strip():
                new_lines.append("")
            new_lines.extend(toc.splitlines())
            if lines[idx].strip():
                new_lines.append("")

        if idx in anchors_to_insert:
            new_lines.append(f'<a id="{anchors_to_insert[idx]}"></a>')
        new_lines.append(lines[idx])
        idx += 1

    if insert_idx == len(lines):
        if new_lines and new_lines[-1].strip():
            new_lines.append("")
        new_lines.extend(toc.splitlines())

    return new_lines


def add_toc_to_file(
    path: Path, min_level: int, max_level: int, dry_run: bool, update_existing: bool
) -> str:
    text = path.read_text(encoding="utf-8")

    lines = text.splitlines()
    headings, anchors_to_insert = collect_headings(lines, min_level=min_level, max_level=max_level)
    toc = render_toc(headings, min_level=min_level, max_level=max_level)
    if not toc:
        return "skip: no headings"

    toc_range = find_generated_toc_range(lines)
    if has_generated_toc(text):
        if generated_toc_matches(lines, toc) and not anchors_to_insert:
            return "ok: generated TOC matches"

        if not update_existing:
            return "mismatch: generated TOC differs"

        new_lines = build_new_lines(lines, toc, anchors_to_insert, replace_range=toc_range)
        new_text = "\n".join(new_lines) + ("\n" if text.endswith("\n") else "")
        if not dry_run:
            path.write_text(new_text, encoding="utf-8")
        return "update"

    insert_idx = find_insert_index(lines)
    new_lines = build_new_lines(lines, toc, anchors_to_insert, insert_idx=insert_idx)
    new_text = "\n".join(new_lines) + ("\n" if text.endswith("\n") else "")
    if not dry_run:
        path.write_text(new_text, encoding="utf-8")

    return "add"


def iter_markdown_files(directory: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.md" if recursive else "*.md"
    return sorted(path for path in directory.glob(pattern) if path.is_file())


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add a generated Table of Contents to all *.md files in a directory."
    )
    parser.add_argument("directory", type=Path, help="Directory containing Markdown files.")
    parser.add_argument("--recursive", action="store_true", help="Process *.md files recursively.")
    parser.add_argument("--min-level", type=int, default=2, help="Minimum heading level to include. Default: 2.")
    parser.add_argument("--max-level", type=int, default=6, help="Maximum heading level to include. Default: 6.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without modifying files.")
    parser.add_argument(
        "--update-existing",
        action="store_true",
        help="Replace an existing generated TOC when it differs from the current headings.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    directory = args.directory.expanduser().resolve()

    if not directory.is_dir():
        print(f"error: not a directory: {directory}", file=sys.stderr)
        return 1

    if not 1 <= args.min_level <= args.max_level <= 6:
        print("error: require 1 <= --min-level <= --max-level <= 6", file=sys.stderr)
        return 1

    files = iter_markdown_files(directory, recursive=args.recursive)
    if not files:
        print(f"no *.md files found under: {directory}")
        return 0

    for path in files:
        result = add_toc_to_file(
            path,
            min_level=args.min_level,
            max_level=args.max_level,
            dry_run=args.dry_run,
            update_existing=args.update_existing,
        )
        rel_path = path.relative_to(directory)
        print(f"{result}: {rel_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
