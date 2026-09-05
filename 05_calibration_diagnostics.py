from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from common import distribution_summary, monitored_spans, read_jsonl, token_content_key


def tail_counts(values: np.ndarray, thresholds=(0.5, 0.9, 0.99, 0.999)) -> dict:
    return {
        str(threshold): {"count": int(np.sum(values >= threshold)), "fraction": float(np.mean(values >= threshold))}
        for threshold in thresholds
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose arbitrary-token calibration failures and recurring hard negatives.")
    parser.add_argument("--input", default="artifacts/scored_calibration.jsonl")
    parser.add_argument("--probe", default="artifacts/probe.npz")
    parser.add_argument("--output", default="artifacts/calibration_diagnostics.json")
    parser.add_argument("--audit-out", default="artifacts/calibration_false_positive_audit.txt")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--window", type=int, default=12)
    args = parser.parse_args()
    if args.top_k <= 0 or args.window < 0:
        raise ValueError("--top-k must be positive and --window nonnegative.")

    rows = list(read_jsonl(args.input))
    if not rows or any(int(row["label"]) != 0 for row in rows):
        raise ValueError("Calibration diagnostics require a nonempty all-negative file.")
    ids = [row["trajectory_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Calibration trajectory IDs are not unique.")

    occurrences = []
    trajectory_maxima = []
    content_tokens = {}
    source_to_contents = defaultdict(set)
    all_monitored_scores = []

    for row in rows:
        scores = np.asarray(row["token_scores"], dtype=np.float64)
        mask = np.asarray(row["monitor_mask"], dtype=bool)
        spans = monitored_spans(row["monitor_mask"])
        if not len(scores) == len(mask) == len(row["input_ids"]):
            raise ValueError(f"Length mismatch in {row['trajectory_id']}.")
        if not np.isfinite(scores).all() or not len(spans):
            raise ValueError(f"Missing or non-finite scores in {row['trajectory_id']}.")
        if not len(spans) == len(row["source_ids"]) == len(row["source_pair_ids"]):
            raise ValueError(f"Span/source mismatch in {row['trajectory_id']}.")

        monitored_indices = np.flatnonzero(mask)
        monitored_scores = scores[mask]
        all_monitored_scores.extend(monitored_scores.tolist())
        max_index = int(monitored_indices[int(np.argmax(monitored_scores))])

        for segment_index, ((start, end), source_id, pair_id) in enumerate(
            zip(spans, row["source_ids"], row["source_pair_ids"])
        ):
            tokens = tuple(int(token_id) for token_id in row["input_ids"][start:end])
            key = token_content_key(tokens)
            if key in content_tokens and content_tokens[key] != tokens:
                raise RuntimeError("SHA-256 collision detected.")
            content_tokens[key] = tokens
            source_to_contents[str(source_id)].add(key)
            segment_scores = scores[start:end]
            argmax = int(np.argmax(segment_scores))
            occurrence = {
                "trajectory_id": row["trajectory_id"],
                "content_key": key,
                "source_id": str(source_id),
                "pair_id": str(pair_id),
                "segment_index": segment_index,
                "start": start,
                "end": end,
                "max_score": float(segment_scores[argmax]),
                "last_token_score": float(segment_scores[-1]),
                "argmax_offset": argmax,
                "argmax_relative": float(argmax / max(len(segment_scores) - 1, 1)),
            }
            occurrences.append(occurrence)
            if start <= max_index < end:
                trajectory_maxima.append({**occurrence, "max_index": max_index})

    if len(trajectory_maxima) != len(rows):
        raise RuntimeError("Each trajectory must contribute exactly one maximum.")

    grouped = defaultdict(list)
    by_trajectory = defaultdict(list)
    for occurrence in occurrences:
        grouped[occurrence["content_key"]].append(occurrence)
        by_trajectory[occurrence["trajectory_id"]].append(occurrence)
    winner_counts = Counter(item["content_key"] for item in trajectory_maxima)
    ranked_winners = winner_counts.most_common()

    occurrence_max = np.asarray([item["max_score"] for item in occurrences])
    occurrence_last = np.asarray([item["last_token_score"] for item in occurrences])
    relative_positions = np.asarray([item["argmax_relative"] for item in trajectory_maxima])
    all_scores = np.asarray(all_monitored_scores)
    trajectory_max_scores = np.asarray([item["max_score"] for item in trajectory_maxima])
    correlation = None
    if np.std(occurrence_max) > 0 and np.std(occurrence_last) > 0:
        correlation = float(np.corrcoef(occurrence_max, occurrence_last)[0, 1])

    sensitivity = []
    for k in (0, 1, 3, 5, 10):
        excluded = {key for key, _count in ranked_winners[:k]}
        recomputed = [
            max(item["max_score"] for item in items if item["content_key"] not in excluded)
            for items in by_trajectory.values()
            if any(item["content_key"] not in excluded for item in items)
        ]
        sensitivity.append({"n_excluded_contents": k, "trajectory_max_distribution": distribution_summary(recomputed)})

    top_contents = []
    for key in sorted(grouped, key=lambda item: np.median([x["max_score"] for x in grouped[item]]), reverse=True)[:args.top_k]:
        items = grouped[key]
        top_contents.append(
            {
                "content_key": key,
                "source_ids": sorted({item["source_id"] for item in items}),
                "pair_ids": sorted({item["pair_id"] for item in items}),
                "n_occurrences": len(items),
                "n_trajectory_maxima": winner_counts[key],
                "median_within_source_max": float(np.median([item["max_score"] for item in items])),
                "median_last_token_score": float(np.median([item["last_token_score"] for item in items])),
                "median_argmax_relative": float(np.median([item["argmax_relative"] for item in items])),
            }
        )

    result = {
        "monitoring_warning": "The probe was trained at interaction ends; arbitrary-token maxima are diagnostic, not the primary monitor.",
        "n_trajectories": len(rows),
        "n_source_occurrences": len(occurrences),
        "n_unique_exact_contents": len(grouped),
        "n_source_ids_mapping_to_multiple_contents": sum(len(keys) > 1 for keys in source_to_contents.values()),
        "trajectory_all_token_max": distribution_summary(trajectory_max_scores),
        "all_monitored_token_scores": distribution_summary(all_scores),
        "per_token_score_tails": tail_counts(all_scores),
        "trajectory_max_relative_position": distribution_summary(relative_positions),
        "trajectory_max_near_boundaries": {
            str(k): {
                "within_first_k": int(sum(item["argmax_offset"] < k for item in trajectory_maxima)),
                "within_last_k": int(sum(item["end"] - item["start"] - 1 - item["argmax_offset"] < k for item in trajectory_maxima)),
            }
            for k in (1, 2, 3, 5, 10)
        },
        "within_source_max_vs_last_token_correlation": correlation,
        "occurrence_max_score": distribution_summary(occurrence_max),
        "occurrence_last_token_score": distribution_summary(occurrence_last),
        "hard_negative_sensitivity": sensitivity,
        "top_hard_contents": top_contents,
    }

    with np.load(args.probe, allow_pickle=False) as probe:
        tokenizer = AutoTokenizer.from_pretrained(str(probe["model_name"]), use_fast=True)
    lines = ["CALIBRATION ALL-TOKEN FALSE-POSITIVE AUDIT", result["monitoring_warning"], ""]
    for rank, item in enumerate(sorted(trajectory_maxima, key=lambda row: row["max_score"], reverse=True)[:args.top_k], 1):
        tokens = rows[ids.index(item["trajectory_id"])]["input_ids"]
        index, start, end = item["max_index"], item["start"], item["end"]
        window = tokens[max(start, index - args.window):min(end, index + args.window + 1)]
        lines.extend(
            [
                "-" * 80,
                f"rank={rank} trajectory={item['trajectory_id']} score={item['max_score']:.9f}",
                f"source_id={item['source_id']} pair_id={item['pair_id']} content_key={item['content_key']}",
                f"offset={item['argmax_offset']} relative_position={item['argmax_relative']:.3f}",
                f"trigger_token={tokenizer.decode([tokens[index]], skip_special_tokens=False)!r}",
                "LOCAL WINDOW:",
                tokenizer.decode(window, skip_special_tokens=False),
                "FULL SOURCE:",
                tokenizer.decode(tokens[start:end], skip_special_tokens=False),
                "",
            ]
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    audit = Path(args.audit_out)
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("n_trajectories", "n_source_occurrences", "n_unique_exact_contents", "trajectory_all_token_max", "within_source_max_vs_last_token_correlation")}, indent=2))
    print(f"Saved: {output}, {audit}")


if __name__ == "__main__":
    main()
