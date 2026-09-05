from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common import INTERACTION_SEPARATOR, apply_probe, choose_device, distribution_summary, layer_activations, load_model_and_tokenizer, load_probe, monitored_spans, pad_sequences, read_jsonl, token_content_key


def extract_occurrences(rows: list[dict]):
    content_tokens = {}
    content_metadata = defaultdict(lambda: {"source_ids": set(), "pair_ids": set()})
    source_to_contents = defaultdict(set)
    occurrences, streams = [], {}
    for row in rows:
        stream_id = row.get("stream_id")
        if not stream_id or stream_id in streams:
            raise ValueError("Selected negative trajectories must have unique nonempty stream IDs.")
        scores = np.asarray(row["token_scores"], dtype=np.float64)
        spans = monitored_spans(row["monitor_mask"])
        if not len(scores) == len(row["input_ids"]) == len(row["monitor_mask"]):
            raise ValueError(f"Length mismatch in {row['trajectory_id']}.")
        if not len(spans) == len(row["source_ids"]) == len(row["source_pair_ids"]):
            raise ValueError(f"Span/source mismatch in {row['trajectory_id']}.")

        stream_items = []
        for checkpoint, ((start, end), source_id, pair_id) in enumerate(
            zip(spans, row["source_ids"], row["source_pair_ids"]), start=1
        ):
            tokens = tuple(int(token_id) for token_id in row["input_ids"][start:end])
            key = token_content_key(tokens)
            if key in content_tokens and content_tokens[key] != tokens:
                raise RuntimeError("SHA-256 collision detected.")
            content_tokens[key] = tokens
            content_metadata[key]["source_ids"].add(str(source_id))
            content_metadata[key]["pair_ids"].add(str(pair_id))
            source_to_contents[str(source_id)].add(key)
            item = {
                "trajectory_id": row["trajectory_id"],
                "stream_id": stream_id,
                "checkpoint": checkpoint,
                "content_key": key,
                "source_id": str(source_id),
                "pair_id": str(pair_id),
                "actual_score": float(scores[end - 1]),
            }
            occurrences.append(item)
            stream_items.append(item)
        streams[stream_id] = stream_items
    return content_tokens, content_metadata, source_to_contents, occurrences, streams


@torch.inference_mode()
def score_control_conditions(content_tokens, model, tokenizer, probe, separator_ids, batch_size, device):
    keys = sorted(content_tokens)
    results = {}
    for condition in ("standalone", "separator_only"):
        condition_scores = {}
        for start in tqdm(range(0, len(keys), batch_size), desc=f"Scoring {condition}"):
            batch_keys = keys[start:start + batch_size]
            prefix = [] if condition == "standalone" else separator_ids
            sequences = [[tokenizer.bos_token_id, *prefix, *content_tokens[key]] for key in batch_keys]
            input_ids, attention_mask, lengths = pad_sequences(sequences, tokenizer.pad_token_id, device)
            scores = apply_probe(layer_activations(model, probe["layer"], input_ids, attention_mask), probe)
            for index, (key, length) in enumerate(zip(batch_keys, lengths)):
                condition_scores[key] = float(scores[index, length - 1].item())
        if len(condition_scores) != len(content_tokens):
            raise RuntimeError(f"Did not score every exact content under {condition}.")
        results[condition] = condition_scores
    return results


def logit(values, epsilon=1e-6):
    values = np.clip(np.asarray(values, dtype=np.float64), epsilon, 1 - epsilon)
    return np.log(values / (1 - values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Separate repeated exposure, separator, and accumulated-context effects.")
    parser.add_argument("--input", default="artifacts/scored_test.jsonl")
    parser.add_argument("--probe", default="artifacts/probe.npz")
    parser.add_argument("--calibration", default="artifacts/monitor_calibration.json")
    parser.add_argument("--length", type=int, help="Negative target length to analyze; defaults to the longest available.")
    parser.add_argument("--bin-width", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", default="artifacts/context_controls.json")
    args = parser.parse_args()
    if min(args.bin_width, args.batch_size, args.top_k) <= 0:
        raise ValueError("--bin-width, --batch-size, and --top-k must be positive.")

    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    primary = calibration["primary_interaction_end_monitor"]
    threshold = float(primary["calibration"]["threshold"])
    calibration_p = float(primary["empirical_per_checkpoint_fpr"])
    if not 0 <= threshold <= 1 or not 0 <= calibration_p <= 1:
        raise ValueError("Frozen threshold or checkpoint FPR is outside [0, 1].")

    all_rows = list(read_jsonl(args.input))
    negative_rows = [row for row in all_rows if int(row["label"]) == 0]
    if not negative_rows:
        raise ValueError("No negative test trajectories found.")
    target_length = args.length if args.length is not None else max(int(row["target_length"]) for row in negative_rows)
    rows = [row for row in negative_rows if int(row["target_length"]) == target_length]
    if not rows:
        raise ValueError(f"No negative trajectories have target length {target_length}.")

    content_tokens, content_metadata, source_to_contents, occurrences, streams = extract_occurrences(rows)
    min_checkpoints = min(len(items) for items in streams.values())
    if min_checkpoints < 1:
        raise RuntimeError("A selected stream has no monitoring checkpoints.")

    device = choose_device(args.device)
    probe = load_probe(args.probe, device)
    model, tokenizer = load_model_and_tokenizer(probe["model_name"], device)
    separator_ids = tokenizer(INTERACTION_SEPARATOR, add_special_tokens=False)["input_ids"]
    if tokenizer.bos_token_id is None or not separator_ids:
        raise ValueError("Tokenizer must supply BOS and nonempty separator tokenization.")
    controlled = score_control_conditions(content_tokens, model, tokenizer, probe, separator_ids, args.batch_size, device)
    standalone, separator = controlled["standalone"], controlled["separator_only"]

    def separator_deployment_score(item):
        return standalone[item["content_key"]] if item["checkpoint"] == 1 else separator[item["content_key"]]

    standalone_values = np.asarray([standalone[item["content_key"]] for item in occurrences])
    separator_values = np.asarray([separator_deployment_score(item) for item in occurrences])
    actual_values = np.asarray([item["actual_score"] for item in occurrences])
    overall = {
        "n_occurrences": len(occurrences),
        "standalone_content_alert_rate": float(np.mean(standalone_values > threshold)),
        "separator_only_deployment_alert_rate": float(np.mean(separator_values > threshold)),
        "actual_in_context_alert_rate": float(np.mean(actual_values > threshold)),
        "separator_minus_standalone_score": distribution_summary(separator_values - standalone_values),
        "context_minus_separator_score": distribution_summary(actual_values - separator_values),
        "context_minus_standalone_logit": distribution_summary(logit(actual_values) - logit(standalone_values)),
    }

    position_bins = []
    for start in range(1, min_checkpoints + 1, args.bin_width):
        end = min(start + args.bin_width - 1, min_checkpoints)
        items = [item for item in occurrences if start <= item["checkpoint"] <= end]
        expected = len(streams) * (end - start + 1)
        if len(items) != expected:
            raise RuntimeError(f"Expected {expected} occurrences in checkpoint bin {start}-{end}, found {len(items)}.")
        s = np.asarray([standalone[item["content_key"]] for item in items])
        b = np.asarray([separator_deployment_score(item) for item in items])
        a = np.asarray([item["actual_score"] for item in items])
        position_bins.append(
            {
                "checkpoint_range": f"{start}-{end}",
                "n_occurrences": len(items),
                "standalone_rate": float(np.mean(s > threshold)),
                "separator_only_rate": float(np.mean(b > threshold)),
                "actual_in_context_rate": float(np.mean(a > threshold)),
                "mean_context_minus_standalone_score": float(np.mean(a - s)),
            }
        )

    per_checkpoint = []
    for checkpoint in range(1, max(len(items) for items in streams.values()) + 1):
        items = [item for item in occurrences if item["checkpoint"] == checkpoint]
        if not items:
            continue
        scores = np.asarray([item["actual_score"] for item in items])
        per_checkpoint.append(
            {
                "checkpoint": checkpoint,
                "n_streams": len(items),
                "alert_rate": float(np.mean(scores > threshold)),
                "score_distribution": distribution_summary(scores),
            }
        )

    checkpoints = {1, min_checkpoints}
    checkpoints.update(range(args.bin_width, min_checkpoints + 1, args.bin_width))
    checkpoints.update(value for value in (2, 4, 8, 10, 12, 16, 20, 24, 30, 36, 40) if value <= min_checkpoints)
    cumulative = []
    for k in sorted(checkpoints):
        standalone_flags, separator_flags, actual_flags = [], [], []
        for items in streams.values():
            prefix = items[:k]
            standalone_flags.append(any(standalone[item["content_key"]] > threshold for item in prefix))
            separator_flags.append(any(separator_deployment_score(item) > threshold for item in prefix))
            actual_flags.append(any(item["actual_score"] > threshold for item in prefix))
        cumulative.append(
            {
                "n_checkpoints": k,
                "naive_calibration_fpr": float(1 - (1 - calibration_p) ** k),
                "standalone_content_fpr": float(np.mean(standalone_flags)),
                "separator_only_fpr": float(np.mean(separator_flags)),
                "actual_in_context_fpr": float(np.mean(actual_flags)),
            }
        )

    by_content = defaultdict(list)
    for item in occurrences:
        by_content[item["content_key"]].append(item)
    content_effects = []
    for key, items in by_content.items():
        actual = np.asarray([item["actual_score"] for item in items])
        boundary = np.asarray([separator_deployment_score(item) for item in items])
        content_effects.append(
            {
                "content_key": key,
                "source_ids": sorted(content_metadata[key]["source_ids"]),
                "pair_ids": sorted(content_metadata[key]["pair_ids"]),
                "n_occurrences": len(items),
                "standalone_score": standalone[key],
                "separator_only_score": separator[key],
                "mean_actual_score": float(actual.mean()),
                "mean_context_minus_standalone": float(actual.mean() - standalone[key]),
                "mean_context_minus_separator_deployment": float(np.mean(actual - boundary)),
                "actual_alert_fraction": float(np.mean(actual > threshold)),
            }
        )
    content_effects.sort(key=lambda item: item["mean_context_minus_standalone"], reverse=True)

    occurrence_counts = Counter(item["content_key"] for item in occurrences)
    alert_counts = Counter(item["content_key"] for item in occurrences if item["actual_score"] > threshold)
    top_alerting = sorted(
        (
            {
                "content_key": key,
                "occurrences": count,
                "alerts": alert_counts[key],
                "alert_rate": float(alert_counts[key] / count),
                "source_ids": sorted(content_metadata[key]["source_ids"]),
            }
            for key, count in occurrence_counts.items()
        ),
        key=lambda item: (item["alert_rate"], item["alerts"]),
        reverse=True,
    )[:args.top_k]

    result = {
        "frozen_threshold": threshold,
        "calibration_checkpoint_fpr": calibration_p,
        "target_length_analyzed": target_length,
        "n_long_negative_streams": len(streams),
        "minimum_checkpoints_per_stream": min_checkpoints,
        "n_unique_exact_contents": len(content_tokens),
        "n_occurrences": len(occurrences),
        "source_id_metadata_check": {
            "n_source_ids_mapping_to_multiple_exact_contents": sum(len(keys) > 1 for keys in source_to_contents.values()),
            "note": "Exact token sequences, not source_id metadata, identify content in these controls.",
        },
        "separator": INTERACTION_SEPARATOR,
        "separator_token_ids": separator_ids,
        "overall_occurrence_comparison": overall,
        "position_bins": position_bins,
        "per_checkpoint_position": per_checkpoint,
        "cumulative_fpr_decomposition": cumulative,
        "top_alerting_contents": top_alerting,
        "largest_positive_context_shifts": content_effects[:args.top_k],
        "largest_negative_context_shifts": list(reversed(content_effects[-args.top_k:])),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("target_length_analyzed", "n_long_negative_streams", "n_unique_exact_contents", "n_occurrences", "overall_occurrence_comparison")}, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
