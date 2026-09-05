from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from common import INTERACTION_SEPARATOR, monitored_spans, read_jsonl, seed_all, write_jsonl


def load_pool(path: str | Path) -> tuple[list[dict], list[dict]]:
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError(f"Source pool is empty: {path}")
    if any(row.get("label") not in {"low-stakes", "high-stakes"} for row in rows):
        raise ValueError(f"Source pool contains an invalid label: {path}")
    return (
        [row for row in rows if row["label"] == "low-stakes"],
        [row for row in rows if row["label"] == "high-stakes"],
    )


def tokenize_pool(rows, tokenizer, min_tokens: int, max_tokens: int) -> list[dict]:
    """Keep only complete source interactions within the token-length bounds."""
    tokenized = []
    for row in rows:
        input_ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
        if min_tokens <= len(input_ids) <= max_tokens:
            tokenized.append({**row, "input_ids": input_ids})
    return tokenized


def new_state(bos_token_id: int) -> dict:
    return {"input_ids": [bos_token_id], "monitor_mask": [0], "segments": [], "spans": []}


def append_segment(state: dict, segment: dict, separator_ids: list[int]) -> tuple[int, int]:
    if state["segments"]:
        state["input_ids"].extend(separator_ids)
        state["monitor_mask"].extend([0] * len(separator_ids))
    start = len(state["input_ids"])
    state["input_ids"].extend(segment["input_ids"])
    state["monitor_mask"].extend([1] * len(segment["input_ids"]))
    end = len(state["input_ids"])
    state["segments"].append(segment)
    state["spans"].append((start, end))
    return start, end


def append_cost(state: dict, segment: dict, separator_ids: list[int]) -> int:
    return len(segment["input_ids"]) + (len(separator_ids) if state["segments"] else 0)


def choose_fitting_segment(rng, pool, used_pairs, state, separator_ids, target_length):
    for index in rng.permutation(len(pool)):
        segment = pool[index]
        if segment["pair_id"] not in used_pairs and len(state["input_ids"]) + append_cost(state, segment, separator_ids) <= target_length:
            return segment
    return None


def make_record(state, label: int, target_length: int, positive=None, insertion_fraction=None) -> dict:
    segments = state["segments"]
    positive_start, positive_end = (positive["span"] if positive else (None, None))
    return {
        "label": label,
        "input_ids": state["input_ids"],
        "monitor_mask": state["monitor_mask"],
        "n_monitored_tokens": int(sum(state["monitor_mask"])),
        "positive_start": positive_start,
        "positive_end": positive_end,
        "positive_source_id": positive["segment"]["id"] if positive else None,
        "positive_source_index": positive["segment"]["dataset_index"] if positive else None,
        "positive_pair_id": positive["segment"]["pair_id"] if positive else None,
        "insertion_target_fraction": insertion_fraction,
        "positive_start_fraction": float(positive_start / len(state["input_ids"])) if positive else None,
        "source_ids": [segment["id"] for segment in segments],
        "source_indices": [segment["dataset_index"] for segment in segments],
        "source_pair_ids": [segment["pair_id"] for segment in segments],
        "source_labels": [segment["label"] for segment in segments],
        "n_source_segments": len(segments),
        "target_length": target_length,
        "actual_length": len(state["input_ids"]),
    }


def make_negative(rng, low_pool, separator_ids, bos_token_id, target_length, min_fraction):
    state = new_state(bos_token_id)
    used_pairs = set()
    while True:
        segment = choose_fitting_segment(rng, low_pool, used_pairs, state, separator_ids, target_length)
        if segment is None:
            break
        append_segment(state, segment, separator_ids)
        used_pairs.add(segment["pair_id"])
    if len(state["input_ids"]) < min_fraction * target_length:
        return None
    return make_record(state, 0, target_length)


def make_positive(rng, low_pool, high_pool, separator_ids, bos_token_id, target_length, min_fraction):
    """Insert exactly one complete high-stakes interaction between low-stakes interactions."""
    for _ in range(1000):
        state = new_state(bos_token_id)
        positive_segment = high_pool[int(rng.integers(len(high_pool)))]
        used_pairs = {positive_segment["pair_id"]}
        insertion_fraction = float(rng.uniform(0.25, 0.70))
        insertion_target = int(insertion_fraction * target_length)

        while True:
            candidates = [
                segment
                for segment in low_pool
                if segment["pair_id"] not in used_pairs
                and len(state["input_ids"]) + append_cost(state, segment, separator_ids) <= insertion_target
            ]
            if not candidates:
                break
            segment = candidates[int(rng.integers(len(candidates)))]
            append_segment(state, segment, separator_ids)
            used_pairs.add(segment["pair_id"])
        if not state["segments"]:
            continue

        remaining_low = [segment for segment in low_pool if segment["pair_id"] not in used_pairs]
        if not remaining_low:
            continue
        required = len(separator_ids) + len(positive_segment["input_ids"]) + len(separator_ids) + min(len(s["input_ids"]) for s in remaining_low)
        if len(state["input_ids"]) + required > target_length:
            continue

        positive_span = append_segment(state, positive_segment, separator_ids)
        after = choose_fitting_segment(rng, low_pool, used_pairs, state, separator_ids, target_length)
        if after is None:
            continue
        append_segment(state, after, separator_ids)
        used_pairs.add(after["pair_id"])

        while True:
            segment = choose_fitting_segment(rng, low_pool, used_pairs, state, separator_ids, target_length)
            if segment is None:
                break
            append_segment(state, segment, separator_ids)
            used_pairs.add(segment["pair_id"])
        if len(state["input_ids"]) >= min_fraction * target_length:
            return make_record(
                state,
                1,
                target_length,
                positive={"segment": positive_segment, "span": positive_span},
                insertion_fraction=insertion_fraction,
            )
    raise RuntimeError(f"Could not construct a positive trajectory of target length {target_length}.")


def derive_negative_prefix(base: dict, target_length: int, min_fraction: float):
    spans = monitored_spans(base["monitor_mask"])
    eligible = [(index + 1, end) for index, (_start, end) in enumerate(spans) if end <= target_length]
    if not eligible:
        return None
    n_segments, prefix_end = max(eligible, key=lambda item: item[1])
    if prefix_end < min_fraction * target_length:
        return None
    fields = ("source_ids", "source_indices", "source_pair_ids", "source_labels")
    row = {
        **base,
        "input_ids": base["input_ids"][:prefix_end],
        "monitor_mask": base["monitor_mask"][:prefix_end],
        "target_length": target_length,
        "actual_length": prefix_end,
        "n_monitored_tokens": int(sum(base["monitor_mask"][:prefix_end])),
        "n_source_segments": n_segments,
    }
    for field in fields:
        row[field] = base[field][:n_segments]
    return row


def validate_trajectory(row: dict, bos_token_id: int, min_fraction: float) -> None:
    n = row["actual_length"]
    if len(row["input_ids"]) != n or len(row["monitor_mask"]) != n:
        raise ValueError(f"Token/mask length mismatch in {row.get('trajectory_id')}.")
    if row["input_ids"][0] != bos_token_id or row["monitor_mask"][0] != 0:
        raise ValueError(f"Invalid BOS position in {row.get('trajectory_id')}.")
    if not min_fraction * row["target_length"] <= n <= row["target_length"]:
        raise ValueError(f"Trajectory length is outside its allowed range: {row.get('trajectory_id')}.")
    if row["n_monitored_tokens"] != sum(row["monitor_mask"]):
        raise ValueError(f"Incorrect monitored-token count in {row.get('trajectory_id')}.")

    spans = monitored_spans(row["monitor_mask"])
    metadata_fields = ("source_ids", "source_indices", "source_pair_ids", "source_labels")
    if any(len(row[field]) != len(spans) for field in metadata_fields) or row["n_source_segments"] != len(spans):
        raise ValueError(f"Source metadata does not align with monitored spans in {row.get('trajectory_id')}.")
    if len(row["source_pair_ids"]) != len(set(row["source_pair_ids"])):
        raise ValueError(f"A pair_id is reused within {row.get('trajectory_id')}.")

    high_indices = [index for index, label in enumerate(row["source_labels"]) if label == "high-stakes"]
    if row["label"] == 0:
        if high_indices or any(row[field] is not None for field in ("positive_start", "positive_end", "positive_pair_id")):
            raise ValueError(f"Negative trajectory contains positive metadata: {row.get('trajectory_id')}.")
    elif row["label"] == 1:
        if len(high_indices) != 1:
            raise ValueError(f"Positive trajectory must contain exactly one high-stakes source: {row.get('trajectory_id')}.")
        index = high_indices[0]
        if spans[index] != (row["positive_start"], row["positive_end"]):
            raise ValueError(f"Positive span does not match the high-stakes source: {row.get('trajectory_id')}.")
        if row["source_indices"][index] != row["positive_source_index"] or row["source_pair_ids"][index] != row["positive_pair_id"]:
            raise ValueError(f"Positive source metadata mismatch: {row.get('trajectory_id')}.")
    else:
        raise ValueError(f"Invalid trajectory label: {row['label']}")


def generate_unique(make_one, count: int, id_template: str, attempts_per_row: int = 200) -> list[dict]:
    rows, signatures = [], set()
    for attempts in range(1, count * attempts_per_row + 1):
        row = make_one()
        if row is None or tuple(row["input_ids"]) in signatures:
            continue
        signatures.add(tuple(row["input_ids"]))
        row["trajectory_id"] = id_template.format(index=len(rows))
        rows.append(row)
        if len(rows) == count:
            return rows
    raise RuntimeError(f"Generated only {len(rows)}/{count} unique trajectories after {attempts} attempts.")


def summarize(rows: list[dict]) -> dict:
    values = lambda field: np.asarray([row[field] for row in rows], dtype=float)
    result = {"n": len(rows)}
    for field in ("actual_length", "n_monitored_tokens", "n_source_segments"):
        data = values(field)
        result[field] = {"min": int(data.min()), "mean": float(data.mean()), "max": int(data.max())}
    positives = [row["positive_start_fraction"] for row in rows if row["label"] == 1]
    if positives:
        result["positive_start_fraction"] = {"min": min(positives), "mean": float(np.mean(positives)), "max": max(positives)}
    return result


def write_preview(path: str | Path, rows: list[dict], tokenizer, per_group: int) -> None:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["label"], row["target_length"])].append(row)
    blocks = []
    for key, group in sorted(groups.items()):
        for row in group[:per_group]:
            header = [
                "=" * 80,
                f"trajectory_id: {row['trajectory_id']}",
                f"split/label/target: {row['split']} / {row['label']} / {row['target_length']}",
                f"actual tokens / interactions: {row['actual_length']} / {row['n_source_segments']}",
                f"source labels: {row['source_labels']}",
            ]
            if row["label"] == 0:
                body = tokenizer.decode(row["input_ids"], skip_special_tokens=False)
            else:
                start, end = row["positive_start"], row["positive_end"]
                body = tokenizer.decode(row["input_ids"][:start], skip_special_tokens=False)
                body += "\n\n>>> HIGH-STAKES SOURCE >>>\n" + tokenizer.decode(row["input_ids"][start:end], skip_special_tokens=False)
                body += "\n<<< END HIGH-STAKES SOURCE <<<\n\n" + tokenizer.decode(row["input_ids"][end:], skip_special_tokens=False)
            blocks.append("\n".join(header + ["", body, ""]))
    Path(path).write_text("\n".join(blocks), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct calibration and held-out pseudo-trajectories.")
    parser.add_argument("--calibration-pool", default="artifacts/trajectory_pool_calibration.jsonl")
    parser.add_argument("--test-pool", default="artifacts/trajectory_pool_test.jsonl")
    parser.add_argument("--probe", default="artifacts/probe.npz")
    parser.add_argument("--lengths", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument("--n-cal-neg", type=int, default=200)
    parser.add_argument("--n-test-neg", type=int, default=100)
    parser.add_argument("--n-test-pos", type=int, default=100)
    parser.add_argument("--min-segment-tokens", type=int, default=8)
    parser.add_argument("--max-segment-tokens", type=int, default=96)
    parser.add_argument("--min-fraction", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--calibration-out", default="artifacts/trajectories_calibration.jsonl")
    parser.add_argument("--test-out", default="artifacts/trajectories_test.jsonl")
    parser.add_argument("--summary-out", default="artifacts/trajectory_generation_summary.json")
    parser.add_argument("--preview-out", default="artifacts/trajectory_preview.txt")
    parser.add_argument("--preview-per-group", type=int, default=1)
    args = parser.parse_args()

    lengths = sorted(set(args.lengths))
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("All target lengths must be positive.")
    if min(args.n_cal_neg, args.n_test_neg, args.n_test_pos) <= 0:
        raise ValueError("Trajectory counts must be positive.")
    if not 0 < args.min_fraction <= 1 or not 0 < args.min_segment_tokens <= args.max_segment_tokens:
        raise ValueError("Invalid segment-length or minimum-fill settings.")

    seed_all(args.seed)
    with np.load(args.probe, allow_pickle=False) as probe:
        model_name = str(probe["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.bos_token_id is None:
        raise ValueError(f"Tokenizer for {model_name} has no BOS token.")
    separator_ids = tokenizer(INTERACTION_SEPARATOR, add_special_tokens=False)["input_ids"]
    if not separator_ids:
        raise ValueError("Interaction separator tokenized to an empty sequence.")

    cal_low_rows, _cal_high_rows = load_pool(args.calibration_pool)
    test_low_rows, test_high_rows = load_pool(args.test_pool)
    cal_pairs = {row["pair_id"] for row in cal_low_rows + _cal_high_rows}
    test_pairs = {row["pair_id"] for row in test_low_rows + test_high_rows}
    if cal_pairs & test_pairs:
        raise RuntimeError("Calibration and test pools overlap by pair_id.")

    tokenize = lambda rows: tokenize_pool(rows, tokenizer, args.min_segment_tokens, args.max_segment_tokens)
    cal_low, test_low, test_high = tokenize(cal_low_rows), tokenize(test_low_rows), tokenize(test_high_rows)
    if not cal_low or not test_low or not test_high:
        raise RuntimeError("At least one required tokenized source pool is empty.")
    print(f"Usable sources — calibration low: {len(cal_low)}, test low: {len(test_low)}, test high: {len(test_high)}")

    rng_cal = np.random.default_rng(args.seed + 1)
    rng_neg = np.random.default_rng(args.seed + 2)
    rng_pos = np.random.default_rng(args.seed + 3)
    base_length = min(lengths)
    calibration = generate_unique(
        lambda: make_negative(rng_cal, cal_low, separator_ids, tokenizer.bos_token_id, base_length, args.min_fraction),
        args.n_cal_neg,
        f"cal_neg_L{base_length}_{{index:04d}}",
    )
    for row in calibration:
        row.update(split="calibration", stream_id=None, paired_prefix=False)

    negative_test, base_signatures = [], set()
    stream_index = 0
    attempts = 0
    while stream_index < args.n_test_neg:
        attempts += 1
        if attempts > args.n_test_neg * 300:
            raise RuntimeError(f"Could not generate all paired negative streams ({stream_index}/{args.n_test_neg}).")
        base = make_negative(rng_neg, test_low, separator_ids, tokenizer.bos_token_id, max(lengths), args.min_fraction)
        if base is None or tuple(base["input_ids"]) in base_signatures:
            continue
        prefixes = {length: derive_negative_prefix(base, length, args.min_fraction) for length in lengths}
        if any(prefix is None for prefix in prefixes.values()):
            continue
        base_signatures.add(tuple(base["input_ids"]))
        stream_id = f"test_neg_stream_{stream_index:04d}"
        for length, row in prefixes.items():
            row.update(
                split="test",
                trajectory_id=f"test_neg_L{length}_{stream_index:04d}",
                stream_id=stream_id,
                paired_prefix=True,
            )
            negative_test.append(row)
        stream_index += 1

    positive_test = []
    for length in lengths:
        rows = generate_unique(
            lambda length=length: make_positive(rng_pos, test_low, test_high, separator_ids, tokenizer.bos_token_id, length, args.min_fraction),
            args.n_test_pos,
            f"test_pos_L{length}_{{index:04d}}",
        )
        for row in rows:
            row.update(split="test", stream_id=None, paired_prefix=False)
        positive_test.extend(rows)

    test = negative_test + positive_test
    for row in calibration + test:
        validate_trajectory(row, tokenizer.bos_token_id, args.min_fraction)
    for stream_id in {row["stream_id"] for row in negative_test}:
        stream = sorted((row for row in negative_test if row["stream_id"] == stream_id), key=lambda row: row["actual_length"])
        if any(long["input_ids"][:short["actual_length"]] != short["input_ids"] for short, long in zip(stream, stream[1:])):
            raise RuntimeError(f"Negative stream {stream_id} is not made of exact token prefixes.")

    write_jsonl(args.calibration_out, calibration)
    write_jsonl(args.test_out, test)
    grouped = defaultdict(list)
    for row in calibration + test:
        grouped[f"{row['split']}_{'pos' if row['label'] else 'neg'}_L{row['target_length']}"].append(row)
    summary = {
        "model_name": model_name,
        "separator": INTERACTION_SEPARATOR,
        "separator_token_ids": separator_ids,
        "source_pool_counts": {
            "calibration_low_before_filtering": len(cal_low_rows),
            "calibration_low_usable": len(cal_low),
            "test_low_before_filtering": len(test_low_rows),
            "test_low_usable": len(test_low),
            "test_high_before_filtering": len(test_high_rows),
            "test_high_usable": len(test_high),
        },
        "settings": {
            "lengths": lengths,
            "min_segment_tokens": args.min_segment_tokens,
            "max_segment_tokens": args.max_segment_tokens,
            "min_fraction": args.min_fraction,
            "seed": args.seed,
        },
        "groups": {name: summarize(rows) for name, rows in sorted(grouped.items())},
    }
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    Path(args.preview_out).parent.mkdir(parents=True, exist_ok=True)
    write_preview(args.preview_out, calibration + test, tokenizer, args.preview_per_group)
    print(json.dumps(summary, indent=2))
    print(f"Saved: {args.calibration_out}, {args.test_out}, {args.summary_out}, {args.preview_out}")


if __name__ == "__main__":
    main()
