from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_TEXT_FIELDS = ("content", "text", "message", "prompt", "input")
INTERACTION_SEPARATOR = "\n\n--- NEXT INTERACTION ---\n\n"


def choose_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token

    # Required because downstream scripts locate the final real token as length - 1.
    tokenizer.padding_side = "right"

    dtype = {"cuda": torch.bfloat16, "mps": torch.float16}.get(torch.device(device).type, torch.float32)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def layer_activations(
    model,
    layer_idx: int,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return one layer's residual-stream output: [batch, sequence, hidden]."""
    try:
        layers = model.model.layers
    except AttributeError as error:
        raise AttributeError("Expected transformer layers at model.model.layers; adapt this helper for other architectures.") from error
    if not 0 <= layer_idx < len(layers):
        raise ValueError(f"layer_idx={layer_idx}, but model has {len(layers)} layers")

    cache: dict[str, torch.Tensor] = {}

    def save_output(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor):
            raise TypeError(f"Unexpected layer output type: {type(hidden).__name__}")
        cache["hidden"] = hidden.detach()

    handle = layers[layer_idx].register_forward_hook(save_output)
    try:
        # Calling the base model avoids computing unused vocabulary logits.
        model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()

    if "hidden" not in cache:
        raise RuntimeError(f"Layer {layer_idx} hook did not capture an activation.")
    return cache["hidden"]


def load_probe(path: str | Path, device: str) -> dict:
    """Load the serialized linear probe onto the inference device."""
    with np.load(path, allow_pickle=False) as saved:
        required = {"mean", "scale", "coef", "intercept", "model_name", "layer"}
        missing = required - set(saved.files)
        if missing:
            raise ValueError(f"Probe archive is missing fields: {sorted(missing)}")
        probe = {
            "mean": torch.as_tensor(saved["mean"], dtype=torch.float32, device=device).reshape(-1),
            "scale": torch.as_tensor(saved["scale"], dtype=torch.float32, device=device).reshape(-1).clamp_min(1e-8),
            "coef": torch.as_tensor(saved["coef"], dtype=torch.float32, device=device).reshape(-1),
            "intercept": torch.tensor(float(np.asarray(saved["intercept"]).reshape(-1)[0]), dtype=torch.float32, device=device),
            "model_name": str(saved["model_name"]),
            "layer": int(saved["layer"]),
        }

    sizes = {probe[name].numel() for name in ("mean", "scale", "coef")}
    if len(sizes) != 1:
        raise ValueError("Probe mean, scale, and coefficient dimensions do not match.")
    if not all(torch.isfinite(probe[name]).all() for name in ("mean", "scale", "coef", "intercept")):
        raise ValueError("Probe parameters contain non-finite values.")
    return probe


def apply_probe(activations: torch.Tensor, probe: dict) -> torch.Tensor:
    """Apply the frozen standardized logistic probe to [..., hidden] activations."""
    if activations.shape[-1] != probe["coef"].numel():
        raise ValueError(f"Activation width {activations.shape[-1]} does not match probe width {probe['coef'].numel()}.")
    standardized = (activations.float() - probe["mean"]) / probe["scale"]
    return torch.sigmoid(torch.matmul(standardized, probe["coef"]) + probe["intercept"])


def pad_sequences(sequences: list[list[int]], pad_token_id: int, device: str):
    """Right-pad nonempty token sequences and return IDs, mask, and lengths."""
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("pad_sequences requires a nonempty collection of nonempty sequences.")
    lengths = [len(sequence) for sequence in sequences]
    input_ids = torch.full((len(sequences), max(lengths)), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for index, sequence in enumerate(sequences):
        input_ids[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[index, :len(sequence)] = 1
    return input_ids, attention_mask, lengths


def monitored_spans(mask: Iterable[int]) -> list[tuple[int, int]]:
    """Return contiguous [start, end) runs of monitored positions."""
    mask = list(mask)
    if any(value not in (0, 1, False, True) for value in mask):
        raise ValueError("Monitor masks must contain only zeros and ones.")
    spans, start = [], None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans


def token_content_key(token_ids: Iterable[int]) -> str:
    """Return a stable identity for an exact token sequence."""
    payload = ",".join(str(int(token_id)) for token_id in token_ids).encode()
    return hashlib.sha256(payload).hexdigest()


def distribution_summary(values: Iterable[float]) -> dict | None:
    """Return compact descriptive statistics for finite numeric values."""
    values = np.asarray(list(values), dtype=np.float64)
    if not len(values):
        return None
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Distribution values must be a finite one-dimensional sequence.")
    return {
        "min": float(values.min()),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def normalize_input(value: Any) -> str:
    """Convert plain text or a JSON-encoded conversation into readable text."""
    if value is None:
        return ""
    if not isinstance(value, str):
        return str(value)

    value = value.strip()
    if not value:
        return value
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value

    parts: list[str] = []

    def collect(item: Any) -> None:
        if isinstance(item, str):
            if text := item.strip():
                parts.append(text)
        elif isinstance(item, dict):
            preferred = [item[key] for key in _TEXT_FIELDS if key in item]
            for child in preferred or item.values():
                collect(child)
        elif isinstance(item, list):
            for child in item:
                collect(child)

    collect(parsed)
    return "\n".join(parts)


def sigmoid_np(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -50, 50)
    return 1.0 / (1.0 + np.exp(-values))


def read_jsonl(path: str | Path):
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
