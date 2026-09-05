from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common import apply_probe, choose_device, distribution_summary, layer_activations, load_model_and_tokenizer, load_probe, pad_sequences, read_jsonl, write_jsonl


@torch.inference_mode()
def score_batch(model, rows: list[dict], probe: dict, pad_token_id: int, device: str) -> list[dict]:
    sequences = [row["input_ids"] for row in rows]
    input_ids, attention_mask, lengths = pad_sequences(sequences, pad_token_id, device)
    scores = apply_probe(layer_activations(model, probe["layer"], input_ids, attention_mask), probe)
    outputs = []
    for index, (row, length) in enumerate(zip(rows, lengths)):
        token_scores = scores[index, :length].cpu().numpy().astype(np.float32).tolist()
        if len(token_scores) != len(row["monitor_mask"]):
            raise RuntimeError(f"Score/mask length mismatch in {row['trajectory_id']}.")
        monitored = [score for score, keep in zip(token_scores, row["monitor_mask"]) if keep]
        if not monitored:
            raise RuntimeError(f"No monitored positions in {row['trajectory_id']}.")
        outputs.append(
            {
                **row,
                "token_scores": token_scores,
                "max_monitored_score": float(max(monitored)),
                "mean_monitored_score": float(np.mean(monitored)),
            }
        )
    return outputs


def validate_scored_row(row: dict) -> None:
    n = row["actual_length"]
    if not len(row["input_ids"]) == len(row["monitor_mask"]) == len(row["token_scores"]) == n:
        raise ValueError(f"Aligned arrays have inconsistent lengths in {row['trajectory_id']}.")
    if row["n_monitored_tokens"] != sum(row["monitor_mask"]):
        raise ValueError(f"Incorrect monitored-token count in {row['trajectory_id']}.")
    scores = np.asarray(row["token_scores"], dtype=np.float64)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError(f"Invalid probe scores in {row['trajectory_id']}.")
    monitored = scores[np.asarray(row["monitor_mask"], dtype=bool)]
    if not np.isclose(row["max_monitored_score"], monitored.max()) or not np.isclose(row["mean_monitored_score"], monitored.mean()):
        raise ValueError(f"Stored score summaries are inconsistent in {row['trajectory_id']}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the frozen probe to every token in a trajectory file.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--probe", default="artifacts/probe.npz")
    parser.add_argument("--summary-out")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    rows = list(read_jsonl(args.input))
    if not rows:
        raise ValueError("No trajectories found.")
    trajectory_ids = [row["trajectory_id"] for row in rows]
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise ValueError("Trajectory IDs are not unique.")
    for row in rows:
        if len(row["input_ids"]) != row["actual_length"] or len(row["monitor_mask"]) != row["actual_length"]:
            raise ValueError(f"Malformed input trajectory: {row['trajectory_id']}")

    device = choose_device(args.device)
    probe = load_probe(args.probe, device)
    model, tokenizer = load_model_and_tokenizer(probe["model_name"], device)
    if any(min(row["input_ids"]) < 0 or max(row["input_ids"]) >= len(tokenizer) for row in rows):
        raise ValueError("A trajectory contains a token ID outside the probe model's vocabulary.")

    scored = []
    for start in tqdm(range(0, len(rows), args.batch_size), desc="Scoring trajectories"):
        scored.extend(score_batch(model, rows[start:start + args.batch_size], probe, tokenizer.pad_token_id, device))
    for row in scored:
        validate_scored_row(row)

    write_jsonl(args.output, scored)
    summary = {
        "input": args.input,
        "probe": args.probe,
        "model_name": probe["model_name"],
        "layer": probe["layer"],
        "n_trajectories": len(scored),
        "actual_length": distribution_summary(row["actual_length"] for row in scored),
        "n_monitored_tokens": distribution_summary(row["n_monitored_tokens"] for row in scored),
        "max_monitored_score": distribution_summary(row["max_monitored_score"] for row in scored),
        "mean_monitored_score": distribution_summary(row["mean_monitored_score"] for row in scored),
    }
    summary_path = Path(args.summary_out) if args.summary_out else Path(args.output).with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {args.output}, {summary_path}")


if __name__ == "__main__":
    main()
