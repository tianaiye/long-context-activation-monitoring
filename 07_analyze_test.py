from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import distribution_summary, monitored_spans, read_jsonl


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if not 0 <= successes <= n or n <= 0:
        raise ValueError("Wilson interval requires 0 <= successes <= n and n > 0.")
    rate = successes / n
    denominator = 1 + z**2 / n
    center = (rate + z**2 / (2 * n)) / denominator
    half_width = z * math.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2)) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def rate_summary(flags) -> dict:
    flags = np.asarray(list(flags), dtype=bool)
    if not len(flags):
        raise ValueError("Cannot summarize an empty outcome set.")
    successes = int(flags.sum())
    low, high = wilson_interval(successes, len(flags))
    return {"count": successes, "n": len(flags), "rate": float(flags.mean()), "ci95_low": low, "ci95_high": high}


def analyze_row(row: dict, threshold: float, calibration_checkpoint_fpr: float) -> dict:
    scores = np.asarray(row["token_scores"], dtype=np.float64)
    mask = np.asarray(row["monitor_mask"], dtype=bool)
    spans = monitored_spans(row["monitor_mask"])
    metadata = [row["source_ids"], row["source_pair_ids"], row["source_labels"]]
    if "source_indices" in row:
        metadata.append(row["source_indices"])
    if not len(scores) == len(mask) == row["actual_length"] or not np.isfinite(scores).all():
        raise ValueError(f"Invalid aligned scores in {row['trajectory_id']}.")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError(f"Out-of-range score in {row['trajectory_id']}.")
    if any(len(field) != len(spans) for field in metadata):
        raise ValueError(f"Source metadata does not match spans in {row['trajectory_id']}.")

    end_scores = scores[[end - 1 for _start, end in spans]]
    n_checkpoints = len(end_scores)
    result = {
        "trajectory_id": row["trajectory_id"],
        "stream_id": row.get("stream_id"),
        "label": int(row["label"]),
        "target_length": int(row["target_length"]),
        "actual_length": int(row["actual_length"]),
        "n_checkpoints": n_checkpoints,
        "primary_trajectory_alert": bool(np.any(end_scores > threshold)),
        "primary_max_checkpoint_score": float(end_scores.max()),
        "naive_independence_probability": float(1 - (1 - calibration_checkpoint_fpr) ** n_checkpoints),
    }

    high_indices = [index for index, label in enumerate(row["source_labels"]) if label == "high-stakes"]
    if row["label"] == 0:
        if high_indices or any(row.get(field) is not None for field in ("positive_start", "positive_end", "positive_pair_id")):
            raise ValueError(f"Negative trajectory contains positive content or metadata: {row['trajectory_id']}.")
        return result
    if row["label"] != 1 or len(high_indices) != 1:
        raise ValueError(f"Positive trajectory must contain exactly one high-stakes source: {row['trajectory_id']}.")

    high_index = high_indices[0]
    if spans[high_index] != (row["positive_start"], row["positive_end"]):
        raise ValueError(f"Positive span mismatch in {row['trajectory_id']}.")
    if row["source_pair_ids"][high_index] != row["positive_pair_id"]:
        raise ValueError(f"Positive pair metadata mismatch in {row['trajectory_id']}.")
    if row["source_ids"][high_index] != row["positive_source_id"]:
        raise ValueError(f"Positive source-ID metadata mismatch in {row['trajectory_id']}.")
    if "source_indices" in row and row["source_indices"][high_index] != row["positive_source_index"]:
        raise ValueError(f"Positive source-index mismatch in {row['trajectory_id']}.")

    low_end_scores = np.delete(end_scores, high_index)
    if not len(low_end_scores):
        raise ValueError(f"Positive trajectory has no low-stakes checkpoints: {row['trajectory_id']}.")
    event_score = float(end_scores[high_index])
    result.update(
        event_final_score=event_score,
        primary_event_detected=event_score > threshold,
        primary_false_alarm_elsewhere=bool(np.any(low_end_scores > threshold)),
        n_low_checkpoints=len(low_end_scores),
        naive_low_false_alarm_probability=float(1 - (1 - calibration_checkpoint_fpr) ** len(low_end_scores)),
    )
    return result


def summarize_negative(rows: list[dict]) -> dict:
    naive = np.asarray([row["naive_independence_probability"] for row in rows])
    alerts = [row["primary_trajectory_alert"] for row in rows]
    return {
        "n": len(rows),
        "primary_false_positive_rate": rate_summary(alerts),
        "naive_independence_predicted_fpr": distribution_summary(naive),
        "empirical_minus_naive": float(np.mean(alerts) - naive.mean()),
        "n_checkpoints": distribution_summary(row["n_checkpoints"] for row in rows),
        "primary_max_checkpoint_score": distribution_summary(row["primary_max_checkpoint_score"] for row in rows),
    }


def summarize_positive(rows: list[dict]) -> dict:
    naive = np.asarray([row["naive_low_false_alarm_probability"] for row in rows])
    return {
        "n": len(rows),
        "primary_event_detection_rate": rate_summary(row["primary_event_detected"] for row in rows),
        "primary_any_trajectory_alert_rate": rate_summary(row["primary_trajectory_alert"] for row in rows),
        "primary_false_alarm_elsewhere_rate": rate_summary(row["primary_false_alarm_elsewhere"] for row in rows),
        "naive_independence_predicted_false_alarm_elsewhere": distribution_summary(naive),
        "event_final_score": distribution_summary(row["event_final_score"] for row in rows),
        "n_checkpoints": distribution_summary(row["n_checkpoints"] for row in rows),
    }


def paired_negative_sanity(rows: list[dict], expected_lengths: list[int], threshold: float) -> dict:
    by_stream = defaultdict(list)
    for row in rows:
        if row.get("stream_id") is None:
            raise ValueError(f"Negative trajectory lacks stream_id: {row['trajectory_id']}.")
        by_stream[row["stream_id"]].append(row)

    malformed, nonmonotonic, score_mismatches = [], [], []
    max_shared_score_difference = 0.0
    for stream_id, stream_rows in by_stream.items():
        ordered = sorted(stream_rows, key=lambda row: row["target_length"])
        lengths = [row["target_length"] for row in ordered]
        malformed_stream = lengths != expected_lengths
        alerts = []
        for row in ordered:
            scores = np.asarray(row["token_scores"])
            spans = monitored_spans(row["monitor_mask"])
            alerts.append(bool(np.any(scores[[end - 1 for _start, end in spans]] > threshold)))
        for short, long in zip(ordered, ordered[1:]):
            n = short["actual_length"]
            metadata_prefixes_match = all(
                long[field][:len(short[field])] == short[field]
                for field in ("source_ids", "source_pair_ids", "source_labels")
            )
            if short["input_ids"] != long["input_ids"][:n] or short["monitor_mask"] != long["monitor_mask"][:n] or not metadata_prefixes_match:
                malformed_stream = True
                continue
            difference = float(np.max(np.abs(np.asarray(short["token_scores"]) - np.asarray(long["token_scores"][:n]))))
            max_shared_score_difference = max(max_shared_score_difference, difference)
            if not np.allclose(short["token_scores"], long["token_scores"][:n], rtol=1e-3, atol=1e-4):
                score_mismatches.append({"stream_id": stream_id, "short_length": short["target_length"], "long_length": long["target_length"], "max_difference": difference})
        if malformed_stream:
            malformed.append({"stream_id": stream_id, "lengths": lengths})
        if any(previous and not current for previous, current in zip(alerts, alerts[1:])):
            nonmonotonic.append({"stream_id": stream_id, "alerts": alerts})
    return {
        "n_streams": len(by_stream),
        "n_malformed_prefix_streams": len(malformed),
        "malformed_prefix_streams": malformed,
        "n_shared_prefix_score_mismatches": len(score_mismatches),
        "shared_prefix_score_mismatches": score_mismatches,
        "max_shared_prefix_score_difference": max_shared_score_difference,
        "n_nonmonotonic_alert_streams": len(nonmonotonic),
        "nonmonotonic_alert_streams": nonmonotonic,
    }


def write_csv(path, negative, positive) -> None:
    fields = [
        "target_length", "neg_n", "mean_neg_checkpoints", "primary_negative_fpr", "primary_negative_fpr_ci_low",
        "primary_negative_fpr_ci_high", "naive_predicted_negative_fpr", "pos_n", "mean_pos_checkpoints",
        "primary_event_detection_rate", "primary_event_detection_ci_low", "primary_event_detection_ci_high",
        "primary_false_alarm_elsewhere_rate", "primary_any_positive_trajectory_alert_rate",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for length in sorted(negative):
            neg, pos = negative[length], positive[length]
            writer.writerow(
                {
                    "target_length": length,
                    "neg_n": neg["n"],
                    "mean_neg_checkpoints": neg["n_checkpoints"]["mean"],
                    "primary_negative_fpr": neg["primary_false_positive_rate"]["rate"],
                    "primary_negative_fpr_ci_low": neg["primary_false_positive_rate"]["ci95_low"],
                    "primary_negative_fpr_ci_high": neg["primary_false_positive_rate"]["ci95_high"],
                    "naive_predicted_negative_fpr": neg["naive_independence_predicted_fpr"]["mean"],
                    "pos_n": pos["n"],
                    "mean_pos_checkpoints": pos["n_checkpoints"]["mean"],
                    "primary_event_detection_rate": pos["primary_event_detection_rate"]["rate"],
                    "primary_event_detection_ci_low": pos["primary_event_detection_rate"]["ci95_low"],
                    "primary_event_detection_ci_high": pos["primary_event_detection_rate"]["ci95_high"],
                    "primary_false_alarm_elsewhere_rate": pos["primary_false_alarm_elsewhere_rate"]["rate"],
                    "primary_any_positive_trajectory_alert_rate": pos["primary_any_trajectory_alert_rate"]["rate"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen interaction-end monitor on held-out test trajectories.")
    parser.add_argument("--input", default="artifacts/scored_test.jsonl")
    parser.add_argument("--calibration", default="artifacts/monitor_calibration.json")
    parser.add_argument("--output", default="artifacts/test_analysis.json")
    parser.add_argument("--csv-out", default="artifacts/test_metrics_by_length.csv")
    args = parser.parse_args()

    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    primary = calibration["primary_interaction_end_monitor"]
    threshold = float(primary["calibration"]["threshold"])
    checkpoint_fpr = float(primary["empirical_per_checkpoint_fpr"])
    if not 0 <= threshold <= 1 or not 0 <= checkpoint_fpr <= 1:
        raise ValueError("Frozen threshold or checkpoint FPR is outside [0, 1].")

    raw_rows = list(read_jsonl(args.input))
    if not raw_rows or any(row.get("split") not in (None, "test") for row in raw_rows):
        raise ValueError("Held-out analysis requires a nonempty test-trajectory file.")
    ids = [row["trajectory_id"] for row in raw_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Test trajectory IDs are not unique.")
    analyzed = [analyze_row(row, threshold, checkpoint_fpr) for row in raw_rows]

    grouped = {0: defaultdict(list), 1: defaultdict(list)}
    raw_negatives = []
    for raw, row in zip(raw_rows, analyzed):
        grouped[row["label"]][row["target_length"]].append(row)
        if row["label"] == 0:
            raw_negatives.append(raw)
    lengths = sorted(set(grouped[0]) | set(grouped[1]))
    if not lengths or set(grouped[0]) != set(grouped[1]):
        raise ValueError("Every target length must contain both negative and positive trajectories.")
    neg_counts = {len(grouped[0][length]) for length in lengths}
    pos_counts = {len(grouped[1][length]) for length in lengths}
    if len(neg_counts) != 1 or len(pos_counts) != 1:
        raise ValueError("Trajectory counts are inconsistent across target lengths.")

    negative = {length: summarize_negative(grouped[0][length]) for length in lengths}
    positive = {length: summarize_positive(grouped[1][length]) for length in lengths}
    paired = paired_negative_sanity(raw_negatives, lengths, threshold)
    if paired["n_streams"] != next(iter(neg_counts)) or paired["n_malformed_prefix_streams"]:
        raise RuntimeError("Paired negative streams do not match the experimental design.")

    result = {
        "analysis_status": "held-out test evaluated with a calibration-frozen interaction-end monitor",
        "frozen_parameters": {
            "threshold": threshold,
            "alert_rule": "score > threshold",
            "calibration_checkpoint_fpr": checkpoint_fpr,
        },
        "negative_results_by_length": {str(length): negative[length] for length in lengths},
        "positive_results_by_length": {str(length): positive[length] for length in lengths},
        "paired_negative_prefix_sanity": paired,
        "n_total_test_trajectories": len(analyzed),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_csv(args.csv_out, negative, positive)
    print(json.dumps({"frozen_parameters": result["frozen_parameters"], "negative_results_by_length": result["negative_results_by_length"], "positive_results_by_length": result["positive_results_by_length"]}, indent=2))
    print(f"Saved: {output}, {args.csv_out}")


if __name__ == "__main__":
    main()
