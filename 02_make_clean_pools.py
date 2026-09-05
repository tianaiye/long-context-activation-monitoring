from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset

from common import normalize_input, read_jsonl, write_jsonl


DATASET = "Arrrlex/models-under-pressure"
VALID_LABELS = {"high-stakes", "low-stakes"}


def load_source_rows(dataset) -> list[dict]:
    rows = []
    for dataset_index, raw in enumerate(dataset):
        if raw.get("labels") not in VALID_LABELS or raw.get("language") != "English":
            continue
        text = normalize_input(raw.get("inputs", ""))
        if not text.strip():
            continue
        if not raw.get("pair_id"):
            raise ValueError("Every usable source row must have a nonempty pair_id.")
        rows.append(
            {
                "dataset_index": dataset_index,
                "id": raw.get("ids", ""),
                "pair_id": raw["pair_id"],
                "label": raw["labels"],
                "text": text,
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    counts = Counter(row["label"] for row in rows)
    return {
        "n_rows": len(rows),
        "n_pair_ids": len({row["pair_id"] for row in rows}),
        "high_stakes": counts["high-stakes"],
        "low_stakes": counts["low-stakes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create pair-disjoint calibration and test source pools.")
    parser.add_argument("--probe-eval-manifest", default="artifacts/probe_eval_selection.jsonl")
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="artifacts")
    args = parser.parse_args()

    if not 0 < args.calibration_fraction < 1:
        raise ValueError("--calibration-fraction must be in (0, 1).")

    source_rows = load_source_rows(load_dataset(DATASET, "training", split="test"))
    source_by_index = {row["dataset_index"]: row for row in source_rows}
    eval_manifest = list(read_jsonl(args.probe_eval_manifest))
    if not eval_manifest:
        raise ValueError("The probe-evaluation manifest is empty.")

    manifest_indices = [int(row["dataset_index"]) for row in eval_manifest]
    if len(manifest_indices) != len(set(manifest_indices)):
        raise ValueError("The probe-evaluation manifest contains duplicate dataset indices.")
    missing_indices = sorted(set(manifest_indices) - set(source_by_index))
    if missing_indices:
        raise ValueError(f"Probe-evaluation rows are missing from training/test: {missing_indices[:5]}")
    for item in eval_manifest:
        source = source_by_index[int(item["dataset_index"])]
        if source["id"] != item["id"] or source["pair_id"] != item["pair_id"] or source["label"] != item["label_name"]:
            raise ValueError(f"Manifest metadata does not match dataset row {item['dataset_index']}.")

    probe_eval_pairs = {item["pair_id"] for item in eval_manifest}
    remaining = [row for row in source_rows if row["pair_id"] not in probe_eval_pairs]
    rows_by_pair = defaultdict(list)
    for row in remaining:
        rows_by_pair[row["pair_id"]].append(row)

    pair_ids = list(rows_by_pair)
    np.random.default_rng(args.seed).shuffle(pair_ids)
    split_index = int(args.calibration_fraction * len(pair_ids))
    calibration_pairs = set(pair_ids[:split_index])
    test_pairs = set(pair_ids[split_index:])
    calibration_rows = [row for pair_id in pair_ids if pair_id in calibration_pairs for row in rows_by_pair[pair_id]]
    test_rows = [row for pair_id in pair_ids if pair_id in test_pairs for row in rows_by_pair[pair_id]]

    if not calibration_pairs or not test_pairs:
        raise RuntimeError("Pair-level split produced an empty pool.")
    if calibration_pairs & test_pairs or calibration_pairs & probe_eval_pairs or test_pairs & probe_eval_pairs:
        raise RuntimeError("Pair leakage detected across probe evaluation, calibration, or test pools.")
    for name, rows in (("calibration", calibration_rows), ("test", test_rows)):
        if {row["label"] for row in rows} != VALID_LABELS:
            raise RuntimeError(f"{name} pool does not contain both labels.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = out_dir / "trajectory_pool_calibration.jsonl"
    test_path = out_dir / "trajectory_pool_test.jsonl"
    write_jsonl(calibration_path, calibration_rows)
    write_jsonl(test_path, test_rows)

    summary = {
        "dataset": f"{DATASET}:training/test",
        "seed": args.seed,
        "calibration_fraction": args.calibration_fraction,
        "probe_evaluation": summarize([source_by_index[index] for index in manifest_indices]),
        "calibration_pool": summarize(calibration_rows),
        "test_pool": summarize(test_rows),
        "pair_overlap_counts": {
            "probe_eval_calibration": len(probe_eval_pairs & calibration_pairs),
            "probe_eval_test": len(probe_eval_pairs & test_pairs),
            "calibration_test": len(calibration_pairs & test_pairs),
        },
    }
    summary_path = out_dir / "trajectory_pool_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {calibration_path}, {test_path}, {summary_path}")


if __name__ == "__main__":
    main()
