#!/usr/bin/env python
"""Architecture bake-off for narrative-to-condition bridge models.

This compares the current regression-style MLP bridge against a CLIP-style
contrastive bridge on the same saved narrative/text artifacts. It is intentionally
offline: it consumes existing OpenAI embeddings and generator memory targets and
does not call the OpenAI API.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_condition_bridge_evaluation import (  # noqa: E402
    build_bridge_examples,
    evaluate_condition_bridge,
    load_pipeline_artifacts,
    select_train_test_windows,
    select_train_test_windows_from_report,
    summarize_bridge_metrics,
)
from experiments.backfill.block_ar.nl_narrative_grounded_scenario_pipeline import (  # noqa: E402
    NarrativeAdapter,
    train_narrative_adapter,
)
from experiments.backfill.block_ar.nl_text_conditioning import (
    normalize_rows,
)  # noqa: E402


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _round_float(value: float) -> float:
    return round(float(value), 12)


def resolve_torch_device(requested: str) -> str:
    """Resolve a user-facing device option for bridge adapter training."""

    value = str(requested).lower()
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if value not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    return value


class ClipConditionAdapter(nn.Module):
    """Projection head from frozen text embeddings into generator memory space."""

    def __init__(
        self, embedding_dim: int, condition_dim: int, hidden_dim: int | None = None
    ):
        super().__init__()
        hidden = int(hidden_dim or min(512, max(condition_dim * 2, embedding_dim // 2)))
        self.net = nn.Sequential(
            nn.LayerNorm(int(embedding_dim)),
            nn.Linear(int(embedding_dim), hidden),
            nn.GELU(),
            nn.Linear(hidden, int(condition_dim)),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(embeddings)


def _group_hard_negative_loss(
    pred: torch.Tensor,
    *,
    roles: list[str],
    groups: list[str],
    margin: float,
) -> torch.Tensor:
    pred_norm = F.normalize(pred, dim=-1)
    terms: list[torch.Tensor] = []
    for group in sorted(set(groups)):
        indices = [idx for idx, value in enumerate(groups) if value == group]
        anchors = [idx for idx in indices if roles[idx] == "anchor"]
        positives = [idx for idx in indices if roles[idx] == "positive"]
        negatives = [idx for idx in indices if roles[idx] == "negative"]
        if not anchors or not negatives:
            continue
        anchor = pred_norm[anchors[0]]
        if positives:
            pos = pred_norm[positives] @ anchor
        else:
            pos = torch.ones(1, dtype=pred.dtype, device=pred.device)
        neg = pred_norm[negatives] @ anchor
        terms.append(F.relu(float(margin) - pos[:, None] + neg[None, :]).mean())
    if not terms:
        return pred.new_zeros(())
    return torch.stack(terms).mean()


def _supervised_contrastive_loss(
    pred: torch.Tensor,
    target_indices: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Supervised contrastive loss over rows with the same memory target."""

    valid_mask = target_indices >= 0
    pred_valid = pred[valid_mask]
    labels = target_indices[valid_mask]
    if int(pred_valid.shape[0]) < 2:
        return pred.new_zeros(())
    pred_norm = F.normalize(pred_valid, dim=-1)
    logits = pred_norm @ pred_norm.T
    logits = logits / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(
        int(pred_valid.shape[0]),
        dtype=torch.bool,
        device=pred_valid.device,
    )
    positive_mask = (labels[:, None] == labels[None, :]) & (~eye)
    denominator_mask = ~eye
    usable_rows = positive_mask.any(dim=1)
    if not bool(torch.any(usable_rows)):
        return pred.new_zeros(())
    neg_inf = torch.finfo(logits.dtype).min
    log_denominator = torch.logsumexp(
        logits.masked_fill(~denominator_mask, neg_inf),
        dim=1,
    )
    log_positive = torch.logsumexp(
        logits.masked_fill(~positive_mask, neg_inf),
        dim=1,
    )
    return -(log_positive[usable_rows] - log_denominator[usable_rows]).mean()


def train_supcon_regression_adapter(
    text_embeddings: np.ndarray,
    target_memory: np.ndarray,
    target_indices: np.ndarray,
    roles: list[str],
    groups: list[str],
    *,
    condition_dim: int,
    hidden_dim: int | None = None,
    steps: int = 700,
    lr: float = 1e-3,
    supcon_weight: float = 0.10,
    supcon_temperature: float = 0.10,
    hard_negative_weight: float = 0.25,
    hard_negative_margin: float = 0.25,
    seed: int = 0,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Train an MLP with memory regression plus supervised contrastive pairing."""

    embeddings = normalize_rows(text_embeddings)
    targets = np.asarray(target_memory, dtype=np.float32)
    target_idx = np.asarray(target_indices, dtype=np.int64)
    if embeddings.shape[0] != target_idx.shape[0]:
        raise ValueError("target_indices length must match text embeddings")
    if len(roles) != embeddings.shape[0] or len(groups) != embeddings.shape[0]:
        raise ValueError("roles/groups length must match text embeddings")
    torch.manual_seed(int(seed))
    device_t = torch.device(device or "cpu")
    x = torch.from_numpy(embeddings).float().to(device_t)
    y = torch.from_numpy(targets).float().to(device_t)
    idx_t = torch.from_numpy(target_idx).to(device_t)
    valid_mask = idx_t >= 0
    if not bool(torch.any(valid_mask)):
        raise ValueError("need at least one non-negative training target")
    adapter = NarrativeAdapter(
        embeddings.shape[1],
        int(condition_dim),
        hidden_dim=hidden_dim,
    ).to(device_t)
    opt = torch.optim.AdamW(adapter.parameters(), lr=float(lr), weight_decay=1e-4)
    losses: list[float] = []
    mse_losses: list[float] = []
    cosine_losses: list[float] = []
    supcon_losses: list[float] = []
    hard_losses: list[float] = []
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        pred = adapter(x)
        align_pred = pred[valid_mask]
        align_target = y[idx_t[valid_mask]]
        mse = F.mse_loss(align_pred, align_target)
        cosine = 1.0 - F.cosine_similarity(align_pred, align_target, dim=-1).mean()
        supcon = _supervised_contrastive_loss(
            pred,
            idx_t,
            temperature=float(supcon_temperature),
        )
        hard = _group_hard_negative_loss(
            pred,
            roles=roles,
            groups=groups,
            margin=float(hard_negative_margin),
        )
        loss = (
            mse
            + 0.2 * cosine
            + float(supcon_weight) * supcon
            + float(hard_negative_weight) * hard
        )
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        mse_losses.append(float(mse.detach()))
        cosine_losses.append(float(cosine.detach()))
        supcon_losses.append(float(supcon.detach()))
        hard_losses.append(float(hard.detach()))
    adapter.eval()
    with torch.no_grad():
        condition_vectors = adapter(x).cpu().numpy().astype(np.float32)
    return {
        "adapter": adapter,
        "condition_vectors": condition_vectors,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "losses": losses,
        "component_loss_last": {
            "mse": mse_losses[-1],
            "cosine": cosine_losses[-1],
            "supervised_contrastive": supcon_losses[-1],
            "hard_negative": hard_losses[-1],
        },
    }


def train_clip_condition_adapter(
    text_embeddings: np.ndarray,
    target_memory: np.ndarray,
    target_indices: np.ndarray,
    roles: list[str],
    groups: list[str],
    *,
    condition_dim: int,
    hidden_dim: int | None = None,
    steps: int = 700,
    lr: float = 1e-3,
    temperature: float = 0.07,
    mse_weight: float = 0.25,
    hard_negative_weight: float = 0.25,
    hard_negative_margin: float = 0.25,
    seed: int = 0,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Train a CLIP-style supervised contrastive bridge over frozen embeddings."""

    embeddings = normalize_rows(text_embeddings)
    targets = np.asarray(target_memory, dtype=np.float32)
    target_idx = np.asarray(target_indices, dtype=np.int64)
    if embeddings.shape[0] != target_idx.shape[0]:
        raise ValueError("target_indices length must match text embeddings")
    if len(roles) != embeddings.shape[0] or len(groups) != embeddings.shape[0]:
        raise ValueError("roles/groups length must match text embeddings")
    valid_mask_np = target_idx >= 0
    if not bool(np.any(valid_mask_np)):
        raise ValueError("need at least one non-negative training target")
    unique_targets = sorted({int(idx) for idx in target_idx[valid_mask_np]})
    target_to_class = {target: cls for cls, target in enumerate(unique_targets)}
    class_labels = np.asarray(
        [target_to_class[int(idx)] for idx in target_idx[valid_mask_np]],
        dtype=np.int64,
    )

    torch.manual_seed(int(seed))
    device_t = torch.device(device or "cpu")
    x = torch.from_numpy(embeddings).float().to(device_t)
    y = torch.from_numpy(targets).float().to(device_t)
    valid_mask = torch.from_numpy(valid_mask_np).to(device_t)
    class_t = torch.from_numpy(class_labels).to(device_t)
    target_idx_t = torch.from_numpy(target_idx).to(device_t)
    target_table_t = torch.tensor(unique_targets, dtype=torch.long, device=device_t)
    adapter = ClipConditionAdapter(
        embeddings.shape[1],
        int(condition_dim),
        hidden_dim=hidden_dim,
    ).to(device_t)
    opt = torch.optim.AdamW(adapter.parameters(), lr=float(lr), weight_decay=1e-4)
    losses: list[float] = []
    ce_losses: list[float] = []
    mse_losses: list[float] = []
    hard_losses: list[float] = []
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        pred = adapter(x)
        pred_valid = pred[valid_mask]
        target_table = y[target_table_t]
        logits = F.normalize(pred_valid, dim=-1) @ F.normalize(target_table, dim=-1).T
        logits = logits / max(float(temperature), 1e-6)
        ce = F.cross_entropy(logits, class_t)
        align_target = y[target_idx_t[valid_mask]]
        mse = F.mse_loss(pred_valid, align_target)
        hard = _group_hard_negative_loss(
            pred,
            roles=roles,
            groups=groups,
            margin=float(hard_negative_margin),
        )
        loss = ce + float(mse_weight) * mse + float(hard_negative_weight) * hard
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        ce_losses.append(float(ce.detach()))
        mse_losses.append(float(mse.detach()))
        hard_losses.append(float(hard.detach()))
    adapter.eval()
    with torch.no_grad():
        condition_vectors = adapter(x).cpu().numpy().astype(np.float32)
    return {
        "adapter": adapter,
        "condition_vectors": condition_vectors,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "losses": losses,
        "component_loss_last": {
            "contrastive_ce": ce_losses[-1],
            "mse": mse_losses[-1],
            "hard_negative": hard_losses[-1],
        },
        "unique_target_count": len(unique_targets),
    }


def _train_example_slice(
    examples: list[dict[str, Any]],
    text_embeddings: np.ndarray,
    train_indices: list[int],
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray, list[str], list[str]]:
    train_set = set(int(idx) for idx in train_indices)
    row_indices = [
        idx
        for idx, example in enumerate(examples)
        if int(example["window_index"]) in train_set
    ]
    train_embeddings = text_embeddings[np.asarray(row_indices, dtype=np.int64)]
    train_examples = [examples[idx] for idx in row_indices]
    target_indices = np.asarray(
        [
            -1 if example["target_index"] is None else int(example["target_index"])
            for example in train_examples
        ],
        dtype=np.int64,
    )
    roles = [str(example["role"]) for example in train_examples]
    groups = [str(example["window_id"]) for example in train_examples]
    return train_embeddings, train_examples, target_indices, roles, groups


def evaluate_condition_vectors(
    examples: list[dict[str, Any]],
    condition_vectors: np.ndarray,
    memory_targets: np.ndarray,
    *,
    split: dict[str, list[int]],
    top_k: int,
) -> dict[str, Any]:
    evaluation = evaluate_condition_bridge(
        examples,
        condition_vectors,
        memory_targets,
        train_indices=split["train_indices"],
        test_indices=split["test_indices"],
        top_k=int(top_k),
    )
    return {
        "summary": summarize_bridge_metrics(evaluation),
        "evaluation": evaluation,
    }


def filter_training_examples(
    examples: list[dict[str, Any]],
    *,
    train_indices: list[int],
    policy: str,
) -> list[int]:
    """Select training rows for caption/negative ablations."""

    train_set = set(int(idx) for idx in train_indices)
    selected: list[int] = []
    for row_idx, example in enumerate(examples):
        if int(example["window_index"]) not in train_set:
            continue
        role = str(example.get("role", ""))
        if policy == "anchor_only":
            keep = role == "anchor"
        elif policy == "multi_caption_no_negatives":
            keep = role in {"anchor", "positive"}
        elif policy == "multi_caption_with_negatives":
            keep = role in {"anchor", "positive", "negative"}
        else:
            raise ValueError(
                "training policy must be anchor_only, "
                "multi_caption_no_negatives, or multi_caption_with_negatives"
            )
        if keep:
            selected.append(row_idx)
    if not selected:
        raise ValueError(f"training policy {policy!r} selected no examples")
    return selected


def run_single_method(
    method: str,
    training_policy: str,
    examples: list[dict[str, Any]],
    text_embeddings: np.ndarray,
    memory_targets: np.ndarray,
    split: dict[str, list[int]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = resolve_torch_device(str(getattr(args, "device", "auto")))
    row_indices = filter_training_examples(
        examples,
        train_indices=split["train_indices"],
        policy=training_policy,
    )
    train_embeddings = text_embeddings[np.asarray(row_indices, dtype=np.int64)]
    train_examples = [examples[idx] for idx in row_indices]
    target_indices = np.asarray(
        [
            -1 if example["target_index"] is None else int(example["target_index"])
            for example in train_examples
        ],
        dtype=np.int64,
    )
    roles = [str(example["role"]) for example in train_examples]
    groups = [str(example["window_id"]) for example in train_examples]
    if method == "mlp_mse_contrastive":
        train_result = train_narrative_adapter(
            train_embeddings,
            memory_targets,
            target_indices,
            roles,
            groups,
            condition_dim=int(memory_targets.shape[1]),
            hidden_dim=args.hidden_dim,
            steps=int(args.adapter_steps),
            lr=float(args.adapter_lr),
            contrastive_weight=float(args.mlp_contrastive_weight),
            contrastive_margin=float(args.hard_negative_margin),
            seed=int(args.seed),
            device=device,
        )
        adapter = train_result["adapter"]
        adapter.eval()
        with torch.no_grad():
            condition_vectors = (
                adapter(
                    torch.from_numpy(normalize_rows(text_embeddings)).float().to(device)
                )
                .cpu()
                .numpy()
                .astype(np.float32)
            )
    elif method == "mlp_supcon_regression":
        train_result = train_supcon_regression_adapter(
            train_embeddings,
            memory_targets,
            target_indices,
            roles,
            groups,
            condition_dim=int(memory_targets.shape[1]),
            hidden_dim=args.hidden_dim,
            steps=int(args.adapter_steps),
            lr=float(args.adapter_lr),
            supcon_weight=float(args.supcon_weight),
            supcon_temperature=float(args.supcon_temperature),
            hard_negative_weight=float(args.supcon_hard_negative_weight),
            hard_negative_margin=float(args.hard_negative_margin),
            seed=int(args.seed),
            device=device,
        )
        adapter = train_result["adapter"]
        adapter.eval()
        with torch.no_grad():
            condition_vectors = (
                adapter(
                    torch.from_numpy(normalize_rows(text_embeddings)).float().to(device)
                )
                .cpu()
                .numpy()
                .astype(np.float32)
            )
    elif method == "clip_infonce_hybrid":
        train_result = train_clip_condition_adapter(
            train_embeddings,
            memory_targets,
            target_indices,
            roles,
            groups,
            condition_dim=int(memory_targets.shape[1]),
            hidden_dim=args.hidden_dim,
            steps=int(args.adapter_steps),
            lr=float(args.adapter_lr),
            temperature=float(args.temperature),
            mse_weight=float(args.clip_mse_weight),
            hard_negative_weight=float(args.clip_hard_negative_weight),
            hard_negative_margin=float(args.hard_negative_margin),
            seed=int(args.seed),
            device=device,
        )
        adapter = train_result["adapter"]
        adapter.eval()
        with torch.no_grad():
            condition_vectors = (
                adapter(
                    torch.from_numpy(normalize_rows(text_embeddings)).float().to(device)
                )
                .cpu()
                .numpy()
                .astype(np.float32)
            )
    else:
        raise ValueError(f"unknown method: {method}")
    eval_block = evaluate_condition_vectors(
        examples,
        condition_vectors,
        memory_targets,
        split=split,
        top_k=int(args.top_k),
    )
    return {
        "method": method,
        "training_policy": training_policy,
        "train_example_count": len(train_examples),
        "train_role_counts": {
            role: int(sum(1 for item in roles if item == role))
            for role in sorted(set(roles))
        },
        "adapter_training": {
            "loss_first": _round_float(float(train_result["loss_first"])),
            "loss_last": _round_float(float(train_result["loss_last"])),
            "steps": int(args.adapter_steps),
            **(
                {"component_loss_last": train_result.get("component_loss_last", {})}
                if "component_loss_last" in train_result
                else {}
            ),
        },
        **eval_block,
    }


def choose_best_method(results: dict[str, dict[str, Any]], metric: str) -> str | None:
    best_name: str | None = None
    best_value = -np.inf
    for name, payload in results.items():
        value = payload.get("summary", {}).get(metric)
        if value is None:
            continue
        raw = float(value)
        if not np.isfinite(raw):
            continue
        if raw > best_value:
            best_name = name
            best_value = raw
    return best_name


def summarize_bakeoff(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "heldout_mean_target_cosine",
        "heldout_recall_at_1_test_pool",
        "heldout_recall_at_3_test_pool",
        "heldout_hard_negative_mean_margin",
        "heldout_hard_negative_mean_gap",
        "heldout_mean_top_train_cosine",
    ]
    return {
        "method_count": len(results),
        "methods": {
            name: {
                metric: payload.get("summary", {}).get(metric)
                for metric in metrics
                if metric in payload.get("summary", {})
            }
            for name, payload in results.items()
        },
        **{
            f"best_by_{metric}": choose_best_method(results, metric)
            for metric in metrics
        },
    }


def run_bakeoff(args: argparse.Namespace) -> dict[str, Any]:
    report, arrays = load_pipeline_artifacts(args.input_report, args.input_npz)
    examples = build_bridge_examples(report)
    text_embeddings = np.asarray(arrays["text_embeddings"], dtype=np.float32)
    memory_targets = np.asarray(arrays["memory_targets"], dtype=np.float32)
    if text_embeddings.shape[0] != len(examples):
        raise ValueError("text_embeddings rows do not match rebuilt examples")
    if str(args.split_source) == "manifest":
        split = select_train_test_windows_from_report(
            report,
            int(memory_targets.shape[0]),
            train_windows=int(args.train_windows),
            test_windows=int(args.test_windows),
        )
    else:
        split = select_train_test_windows(
            int(memory_targets.shape[0]),
            train_windows=int(args.train_windows),
            test_windows=int(args.test_windows),
        )
        split["source"] = "sequential"
    requested = [item.strip() for item in str(args.methods).split(",") if item.strip()]
    policies = [
        item.strip() for item in str(args.training_policies).split(",") if item.strip()
    ]
    results = {}
    for method in requested:
        for policy in policies:
            name = f"{method}__{policy}"
            results[name] = run_single_method(
                method,
                policy,
                examples,
                text_embeddings,
                memory_targets,
                split,
                args,
            )
    summary = summarize_bakeoff(results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "status": "ok",
        "scope_note": (
            "Offline architecture bake-off over saved narrative embeddings and "
            "generator memory targets. No OpenAI API calls are made."
        ),
        "input_report": str(args.input_report),
        "input_npz": str(args.input_npz),
        "embedding_backend": report.get("embedding_backend"),
        "embedding_model": report.get("embedding_model"),
        "split": split,
        "methods_requested": requested,
        "training_policies_requested": policies,
        "device": resolve_torch_device(str(args.device)),
        "summary": summary,
        "results": results,
        "artifact_paths": {
            "report": str(output_dir / "bridge_architecture_bakeoff_report.json"),
        },
    }
    _write_json(output_dir / "bridge_architecture_bakeoff_report.json", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-report", required=True)
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--methods",
        default="mlp_mse_contrastive,clip_infonce_hybrid",
        help="Comma-separated method list.",
    )
    parser.add_argument("--train-windows", type=int, default=40)
    parser.add_argument("--test-windows", type=int, default=10)
    parser.add_argument(
        "--split-source",
        choices=["sequential", "manifest"],
        default="sequential",
        help="Use manifest train/test labels when available, otherwise sequential.",
    )
    parser.add_argument(
        "--training-policies",
        default="multi_caption_with_negatives",
        help=(
            "Comma-separated policies: anchor_only, multi_caption_no_negatives, "
            "multi_caption_with_negatives."
        ),
    )
    parser.add_argument("--adapter-steps", type=int, default=700)
    parser.add_argument("--adapter-lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Torch device for bridge adapter training. auto uses CUDA when visible.",
    )
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--mlp-contrastive-weight", type=float, default=0.25)
    parser.add_argument("--supcon-weight", type=float, default=0.10)
    parser.add_argument("--supcon-temperature", type=float, default=0.10)
    parser.add_argument("--supcon-hard-negative-weight", type=float, default=0.25)
    parser.add_argument("--clip-mse-weight", type=float, default=0.25)
    parser.add_argument("--clip-hard-negative-weight", type=float, default=0.25)
    parser.add_argument("--hard-negative-margin", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()
    report = run_bakeoff(args)
    print(
        json.dumps(
            {
                "report": report["artifact_paths"]["report"],
                "summary": report["summary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
