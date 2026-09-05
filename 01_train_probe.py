from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from common import choose_device, layer_activations, load_model_and_tokenizer, normalize_input, seed_all, sigmoid_np, write_jsonl


DATASET = "Arrrlex/models-under-pressure"
LABEL_TO_INT = {"low-stakes": 0, "high-stakes": 1}


def balanced_rows(dataset, n_per_class: int, seed: int, english_only: bool) -> list[dict]:
    """Select a reproducible, balanced set of nonempty examples."""
    by_label = {0: [], 1: []}
    for dataset_index, row in enumerate(dataset):
        label = LABEL_TO_INT.get(row.get("labels"))
        if label is None or (english_only and row.get("language") not in (None, "English")):
            continue
        text = normalize_input(row.get("inputs", ""))
        if text.strip():
            by_label[label].append(
                {
                    "dataset_index": dataset_index,
                    "id": row.get("ids", ""),
                    "pair_id": row.get("pair_id", ""),
                    "text": text,
                    "label": label,
                    "label_name": row["labels"],
                }
            )

    counts = {label: len(rows) for label, rows in by_label.items()}
    if min(counts.values()) < n_per_class:
        raise ValueError(f"Requested {n_per_class} examples per class, but found {counts}.")

    rng = np.random.default_rng(seed)
    rng.shuffle(by_label[1])
    rng.shuffle(by_label[0])
    rows = by_label[1][:n_per_class] + by_label[0][:n_per_class]
    rng.shuffle(rows)
    return rows


@torch.inference_mode()
def extract_last_token_activations(model, tokenizer, rows, layer, device, batch_size, max_length):
    """Extract the monitored activation: the final real token of each interaction."""
    activations = []
    for start in tqdm(range(0, len(rows), batch_size), desc="Extracting activations"):
        batch = rows[start:start + batch_size]
        encoded = tokenizer(
            [row["text"] for row in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        hidden = layer_activations(model, layer, input_ids, attention_mask)
        final_positions = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(len(batch), device=device)
        activations.append(hidden[batch_indices, final_positions].float().cpu().numpy())

    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    return np.concatenate(activations), labels


def fit_probe(activations: np.ndarray, labels: np.ndarray, seed: int):
    scaler = StandardScaler().fit(activations)
    classifier = LogisticRegression(
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=seed,
    ).fit(scaler.transform(activations), labels)
    return scaler, classifier


def probe_scores(activations: np.ndarray, scaler: StandardScaler, classifier: LogisticRegression) -> np.ndarray:
    mean = scaler.mean_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)
    coef = classifier.coef_[0].astype(np.float32)
    intercept = np.float32(classifier.intercept_[0])
    standardized = (activations.astype(np.float32, copy=False) - mean) / np.maximum(scale, 1e-8)
    return sigmoid_np(standardized @ coef + intercept)


def summarize_scores(labels: np.ndarray, scores: np.ndarray) -> dict:
    predictions = (scores >= 0.5).astype(np.int64)
    return {
        "n_examples": int(len(labels)),
        "accuracy_at_0.5": float(accuracy_score(labels, predictions)),
        "auroc": float(roc_auc_score(labels, scores)),
        "mean_score_high_stakes": float(scores[labels == 1].mean()),
        "mean_score_low_stakes": float(scores[labels == 0].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and validate the final-token linear activation probe.")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-per-class", type=int, default=500)
    parser.add_argument("--eval-per-class", type=int, default=200)
    parser.add_argument("--external-per-class", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--english-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", default="artifacts/probe.npz")
    args = parser.parse_args()

    seed_all(args.seed)
    device = choose_device(args.device)
    print(f"Device: {device}")
    model, tokenizer = load_model_and_tokenizer(args.model, device)

    train_dataset = load_dataset(DATASET, "training", split="train")
    eval_dataset = load_dataset(DATASET, "training", split="test")
    external_dataset = load_dataset(DATASET, "anthropic_hh_balanced", split="validation")

    train_rows = balanced_rows(train_dataset, args.n_per_class, args.seed, args.english_only)
    eval_rows = balanced_rows(eval_dataset, args.eval_per_class, args.seed + 1, args.english_only)
    external_rows = balanced_rows(external_dataset, args.external_per_class, args.seed, args.english_only)
    eval_indices = [row["dataset_index"] for row in eval_rows]
    if len(eval_indices) != len(set(eval_indices)) or any(not row["pair_id"] for row in eval_rows):
        raise RuntimeError("Probe-evaluation examples must have unique dataset indices and nonempty pair IDs.")
    print(f"Selected {len(train_rows)} train, {len(eval_rows)} in-distribution eval, and {len(external_rows)} external eval examples.")

    train_acts, train_labels = extract_last_token_activations(model, tokenizer, train_rows, args.layer, device, args.batch_size, args.max_length)
    eval_acts, eval_labels = extract_last_token_activations(model, tokenizer, eval_rows, args.layer, device, args.batch_size, args.max_length)
    external_acts, external_labels = extract_last_token_activations(model, tokenizer, external_rows, args.layer, device, args.batch_size, args.max_length)

    scaler, classifier = fit_probe(train_acts, train_labels, args.seed)
    eval_scores = probe_scores(eval_acts, scaler, classifier)
    np.testing.assert_allclose(
        eval_scores,
        classifier.predict_proba(scaler.transform(eval_acts))[:, 1],
        rtol=1e-4,
        atol=1e-5,
        err_msg="Saved probe parameters do not reproduce sklearn scores.",
    )

    shuffled_labels = train_labels.copy()
    np.random.default_rng(args.seed).shuffle(shuffled_labels)
    shuffled_scaler, shuffled_classifier = fit_probe(train_acts, shuffled_labels, args.seed)

    metrics = {
        "configuration": {
            "training_dataset": f"{DATASET}:training/train",
            "in_distribution_dataset": f"{DATASET}:training/test",
            "external_dataset": f"{DATASET}:anthropic_hh_balanced/validation",
            "model": args.model,
            "layer": args.layer,
            "max_length": args.max_length,
            "n_train_per_class": args.n_per_class,
            "n_eval_per_class": args.eval_per_class,
            "n_external_per_class": args.external_per_class,
            "seed": args.seed,
            "english_only": args.english_only,
        },
        "in_distribution": summarize_scores(eval_labels, eval_scores),
        "shuffled_label_control": summarize_scores(eval_labels, probe_scores(eval_acts, shuffled_scaler, shuffled_classifier)),
        "external_anthropic_hh": summarize_scores(external_labels, probe_scores(external_acts, scaler, classifier)),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
        coef=classifier.coef_[0].astype(np.float32),
        intercept=np.asarray(classifier.intercept_[0], dtype=np.float32),
        model_name=np.asarray(args.model),
        layer=np.asarray(args.layer),
        max_length=np.asarray(args.max_length),
        metrics_json=np.asarray(json.dumps(metrics)),
    )

    metrics_path = out_path.with_name("probe_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    manifest_path = out_path.with_name("probe_eval_selection.jsonl")
    # Step 2 can exclude these pair IDs directly instead of reconstructing this random sample.
    write_jsonl(manifest_path, ({key: row[key] for key in ("dataset_index", "id", "pair_id", "label", "label_name")} for row in eval_rows))

    print(json.dumps(metrics, indent=2))
    print(f"Saved probe: {out_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved exact probe-evaluation selection: {manifest_path}")


if __name__ == "__main__":
    main()
