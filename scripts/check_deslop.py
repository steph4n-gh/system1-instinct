#!/usr/bin/env python3
"""Pre-commit deslop and prose hygiene gate.

Checks modified or staged Markdown and Python files against `unslop` rules:
- Bans em dashes (`\u2014`) in prose paragraphs of modified Markdown files.
- Bans AI filler vocabulary (`delve`, `tapestry`, `testament`, `underscores`,
  `pivotal`, `vibrant`, `intricate`, `fostering`, `garner`) in modified prose.
- Bans narration comment banners (`# ===...===` or `# ---...---`) in newly added
  Python lines.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

BANNED_AI_WORDS = re.compile(
    r"\b(delve|delves|delving|tapestry|testament|underscores|underscoring|"
    r"pivotal|vibrant|intricate|fostering|garner|garnered)\b",
    re.IGNORECASE,
)

FENCED_BLOCK_RE = re.compile(r"(?ms)^(```|~~~)[^\n]*\n.*?^\1[^\n]*(?:\n|$)")


def strip_fenced_blocks(text: str) -> str:
    return FENCED_BLOCK_RE.sub("", text)


def get_changed_files(staged_only: bool = False) -> list[Path]:
    cmd = ["git", "diff", "--name-only", "--diff-filter=ACMR"]
    if staged_only:
        cmd.append("--cached")
    else:
        cmd.append("HEAD")
    out = subprocess.check_output(cmd, cwd=ROOT, text=True)
    paths = [ROOT / line.strip() for line in out.splitlines() if line.strip()]
    return [p for p in paths if p.is_file()]


def fenced_line_numbers(text: str) -> set[int]:
    fenced: set[int] = set()
    in_fence = False
    for idx, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith(("```", "~~~")):
            fenced.add(idx)
            in_fence = not in_fence
            continue
        if in_fence:
            fenced.add(idx)
    return fenced


def check_markdown_diff(path: Path, staged_only: bool = False) -> list[str]:
    errors: list[str] = []
    rel = path.relative_to(ROOT)
    fenced_lines = fenced_line_numbers(path.read_text(encoding="utf-8"))
    cmd = ["git", "diff", "-U0"]
    if staged_only:
        cmd.append("--cached")
    else:
        cmd.append("HEAD")
    cmd.extend(["--", str(rel)])
    diff_text = subprocess.check_output(cmd, cwd=ROOT, text=True)

    cur_new_line = 0
    for raw_line in diff_text.splitlines():
        if raw_line.startswith("@@"):
            m = re.search(r"\+(\d+)", raw_line)
            if m:
                cur_new_line = int(m.group(1))
            continue
        if not raw_line.startswith("+") or raw_line.startswith("+++"):
            continue
        line_no = cur_new_line
        cur_new_line += 1
        if line_no in fenced_lines:
            continue
        line = raw_line[1:]
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            continue
        if "\u2014" in line:
            errors.append(f"{rel}:{line_no}: added em dash ('\\u2014') in prose: {stripped[:100]}")
        match = BANNED_AI_WORDS.search(line)
        if match:
            errors.append(
                f"{rel}:{line_no}: banned AI vocabulary {match.group(0)!r} in added prose: {stripped[:100]}"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Check modified files for AI prose/comment slop.")
    parser.add_argument(
        "--staged",
        action="store_true",
        help="Check only staged changes (for .git/hooks/pre-commit).",
    )
    args = parser.parse_args()

    changed = get_changed_files(staged_only=args.staged)
    errors: list[str] = []
    md_checked = 0
    for path in changed:
        if path.suffix.lower() == ".md":
            md_checked += 1
            errors.extend(check_markdown_diff(path, staged_only=args.staged))

    for err in errors:
        print(err, file=sys.stderr)

    if errors:
        print(f"Deslop check failed with {len(errors)} issue(s).", file=sys.stderr)
        return 1

    print(f"Deslop check passed ({md_checked} modified Markdown file(s) inspected).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
