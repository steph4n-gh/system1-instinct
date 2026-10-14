"""Fetch pinned HWU64 files and form disjoint cohorts, without fitting models."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "protocol.json").read_text())


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_id(text: str) -> str:
    return digest(" ".join(text.casefold().split()).encode())


def load_cohorts(data_dir: Path) -> tuple[dict, dict]:
    by_id = defaultdict(list)
    source_counts = {}
    for split in ("train", "valid", "test"):
        parts = []
        for suffix in ("text", "label"):
            name = f"{split}.{suffix}"
            raw = (data_dir / name).read_bytes()
            if digest(raw) != PROTOCOL["source_sha256"][name]:
                raise ValueError(f"Source hash mismatch: {name}")
            parts.append(raw.decode().splitlines())
        texts, labels = parts
        if len(texts) != len(labels):
            raise ValueError("Text/label count mismatch")
        source_counts[split] = len(texts)
        for text, label in zip(texts, labels):
            by_id[text_id(text)].append(dict(id=text_id(text), text=text, truth=label, source=split))
    cleaned = {k: [] for k in source_counts}
    conflicts = shared = duplicate_rows = 0
    for rows in by_id.values():
        if len({r["truth"] for r in rows}) != 1:
            conflicts += len(rows)
        elif len({r["source"] for r in rows}) != 1:
            shared += len(rows)
        else:
            cleaned[rows[0]["source"]].append(rows[0])
            duplicate_rows += len(rows) - 1
    classes = sorted({r["truth"] for r in cleaned["train"]})
    if len(classes) != 64:
        raise ValueError("Expected 64 intents")
    cohorts = {"teach": [], "calibrate": [], "qualify": [], "dev": cleaned["valid"], "test": cleaned["test"]}
    t = PROTOCOL["teaching_per_class"]
    c = PROTOCOL["trusted_calibration_per_class"]
    for label in classes:
        rows = sorted((r for r in cleaned["train"] if r["truth"] == label), key=lambda r: r["id"])
        if len(rows) <= t + c:
            raise ValueError(f"Insufficient class: {label}")
        cohorts["teach"].extend(rows[:t])
        cohorts["calibrate"].extend(rows[t:t+c])
        cohorts["qualify"].extend(rows[t+c:])
    for rows in cohorts.values():
        rows.sort(key=lambda r: r["id"])
    ids = [r["id"] for rows in cohorts.values() for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Cohort overlap")
    receipt = dict(source_counts=source_counts, classes=classes,
                   dropped_conflicting_rows=conflicts, dropped_shared_rows=shared,
                   collapsed_duplicate_rows=duplicate_rows,
                   cohorts={k: dict(count=len(v), ids_sha256=digest("\n".join(r["id"] for r in v).encode())) for k, v in cohorts.items()})
    return cohorts, receipt


def teacher_labels(rows: list[dict], seed: int, classes: list[str], rate: float = 0.1, *, per_class: bool = True) -> dict[str, str]:
    """Fixed exact fraction per intent; no RNG/library-version dependence."""
    result = {r["id"]: r["truth"] for r in rows}
    global_wrong = set(r["id"] for r in sorted(rows, key=lambda r: digest(f"noise:{seed}:{r['id']}".encode()))[:round(len(rows) * rate)])
    for label in classes:
        group = sorted((r for r in rows if r["truth"] == label),
                       key=lambda r: digest(f"noise:{seed}:{r['id']}".encode()))
        wrong_options = [x for x in classes if x != label and x.split("_", 1)[0] == label.split("_", 1)[0]]
        if not wrong_options:
            wrong_options = [x for x in classes if x != label]
        wrong_rows = group[:round(len(group) * rate)] if per_class else [r for r in group if r["id"] in global_wrong]
        for row in wrong_rows:
            choice = int(digest(f"wrong:{seed}:{row['id']}".encode()), 16) % len(wrong_options)
            result[row["id"]] = wrong_options[choice]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    if args.download:
        for name, expected in PROTOCOL["source_sha256"].items():
            split, suffix = name.split(".")
            upstream_split = "valid" if split == "valid" else split
            upstream_file = "seq.in" if suffix == "text" else "label"
            url = f"https://raw.githubusercontent.com/{PROTOCOL['source_repository']}/{PROTOCOL['dataset_revision']}/Datasets/HWU64/{upstream_split}/{upstream_file}"
            with urlopen(url, timeout=30) as response:
                raw = response.read(2_000_000)
            if digest(raw) != expected:
                raise ValueError(f"Download hash mismatch: {name}")
            (args.data_dir / name).write_bytes(raw)
    cohorts, receipt = load_cohorts(args.data_dir)
    answers = {r["id"]: r["truth"] for r in cohorts["test"]}
    cohorts["test"] = [{k: v for k, v in r.items() if k != "truth"} for r in cohorts["test"]]
    (args.data_dir / "inputs.json").write_text(json.dumps(dict(cohorts=cohorts, receipt=receipt), sort_keys=True))
    (args.data_dir / "answers.json").write_text(json.dumps(answers, sort_keys=True))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
