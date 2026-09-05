from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import distribution_summary, monitored_spans, read_jsonl


def exact_empirical_threshold(values, target_fpr: float) -> dict:
    """Choose a threshold for the rule score > threshold without test-set access."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Calibration values must be a nonempty finite vector.")
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be in (0, 1).")

    desired_alerts = int(round(target_fpr * len(values)))
    if not 0 < desired_alerts < len(values):
        raise ValueError("Calibration set is too small for the requested target FPR.")
    descending = np.sort(values)[::-1]
    alert_side = float(descending[desired_alerts - 1])
    nonalert_side = float(descending[desired_alerts])
    threshold = (alert_side + nonalert_side) / 2 if alert_side > nonalert_side else alert_side
    achieved_alerts = int(np.sum(values > threshold))
    return {
        "threshold": float(threshold),
        "target_fpr": target_fpr,
        "n_calibration": len(values),
        "desired_alerts": desired_alerts,
        "achieved_alerts": achieved_alerts,
        "achieved_fpr": float(achieved_alerts / len(values)),
        "exact_target_achieved": achieved_alerts == desired_alerts,
        "boundary_tie": alert_side == nonalert_side,
        "boundary_alert_score": alert_side,
        "boundary_nonalert_score": nonalert_side,
        "alert_rule": "score > threshold",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the interaction-end monitor using negative calibration trajectories only.")
    parser.add_argument("--input", default="artifacts/scored_calibration.jsonl")
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--output", default="artifacts/monitor_calibration.json")
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    if not rows:
        raise ValueError("No scored calibration trajectories found.")
    if any(int(row["label"]) != 0 or row.get("split") not in (None, "calibration") for row in rows):
        raise ValueError("Calibration input must contain calibration negatives only.")
    trajectory_ids = [row["trajectory_id"] for row in rows]
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise ValueError("Calibration trajectory IDs are not unique.")
    target_lengths = {int(row["target_length"]) for row in rows}
    if len(target_lengths) != 1:
        raise ValueError("Calibration trajectories must use one baseline target length.")

    trajectory_maxes, checkpoint_scores, checkpoint_counts = [], [], []
    for row in rows:
        scores = np.asarray(row["token_scores"], dtype=np.float64)
        spans = monitored_spans(row["monitor_mask"])
        if not len(scores) == len(row["monitor_mask"]) == row["actual_length"] or not np.isfinite(scores).all():
            raise ValueError(f"Invalid aligned scores in {row['trajectory_id']}.")
        if not len(spans) == len(row["source_ids"]) == row["n_source_segments"]:
            raise ValueError(f"Interaction spans do not match source metadata in {row['trajectory_id']}.")
        end_scores = scores[[end - 1 for _start, end in spans]]
        if not len(end_scores):
            raise ValueError(f"No interaction-end checkpoints in {row['trajectory_id']}.")
        trajectory_maxes.append(float(end_scores.max()))
        checkpoint_scores.extend(end_scores.tolist())
        checkpoint_counts.append(len(end_scores))

    trajectory_maxes = np.asarray(trajectory_maxes)
    checkpoint_scores = np.asarray(checkpoint_scores)
    checkpoint_counts = np.asarray(checkpoint_counts)
    calibration = exact_empirical_threshold(trajectory_maxes, args.target_fpr)
    threshold = calibration["threshold"]
    checkpoint_exceedances = int(np.sum(checkpoint_scores > threshold))
    checkpoint_fpr = float(checkpoint_exceedances / len(checkpoint_scores))
    naive_predictions = 1 - (1 - checkpoint_fpr) ** checkpoint_counts

    result = {
        "monitor_definition": {
            "name": "interaction-end maximum",
            "checkpoint": "final token of each complete source interaction",
            "trajectory_alert": "any checkpoint score > threshold",
            "probe_training_position": "last non-padding token of a standalone interaction",
        },
        "primary_interaction_end_monitor": {
            "trajectory_max_distribution": distribution_summary(trajectory_maxes),
            "calibration": calibration,
            "n_checkpoints_per_trajectory": distribution_summary(checkpoint_counts),
            "total_calibration_checkpoints": int(len(checkpoint_scores)),
            "checkpoint_exceedances_at_threshold": checkpoint_exceedances,
            "empirical_per_checkpoint_fpr": checkpoint_fpr,
            "naive_independence_prediction_on_calibration": distribution_summary(naive_predictions),
        },
        "n_calibration_trajectories": len(rows),
        "calibration_target_length": next(iter(target_lengths)),
        "input": args.input,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved frozen calibration: {output}")


if __name__ == "__main__":
    main()
