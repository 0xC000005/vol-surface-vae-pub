from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, ".")

from experiments.world.evaluation.surface_local_jepa_data import (  # noqa: E402
    build_surface_local_jepa_batch,
)
from experiments.world.part1_jepa_latent.jepa_smoke import update_ema  # noqa: E402
from experiments.world.part1_jepa_latent.masked_multiview_geometry_barlow_smoke import (  # noqa: E402
    build_token_descriptor_matrix,
)
from experiments.world.part1_jepa_latent.surface_local_jepa_model import (  # noqa: E402
    SurfaceLocalTokenJepaConfig,
    SurfaceLocalTokenJepaModel,
    surface_local_parameter_groups,
)
from experiments.world.part1_jepa_latent.surface_local_jepa_smoke import (  # noqa: E402
    EMA_DECAY,
    _batch_tensors,
    _run_loss,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _normalize(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True).clip(min=1e-12)
    return values / norm


def _topk(scores: np.ndarray, k: int) -> np.ndarray:
    kk = min(k, scores.shape[1])
    idx = np.argpartition(-scores, kth=kk - 1, axis=1)[:, :kk]
    row = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[row, idx], axis=1)
    return idx[row, order]


def _same_label_rates(
    topk_idx: np.ndarray,
    query_labels: np.ndarray,
    candidate_labels: np.ndarray,
) -> dict[str, float]:
    candidate_topk = candidate_labels[topk_idx]
    same = candidate_topk == query_labels[:, None]
    return {
        "any_topk": float(np.mean(np.any(same, axis=1))),
        "neighbor_share": float(np.mean(same)),
    }


def _label_baseline(labels: np.ndarray) -> float:
    n = int(labels.shape[0])
    if n <= 1:
        return 0.0
    counts = Counter(labels.tolist())
    return float(np.mean([(counts[str(label)] - 1) / (n - 1) for label in labels.tolist()]))


def _label_summary(labels: np.ndarray, *, topn: int = 8) -> dict[str, Any]:
    counts = Counter(labels.tolist())
    return {
        "n_unique": int(len(counts)),
        "top_counts": [
            {"label": str(label), "count": int(count)}
            for label, count in counts.most_common(topn)
        ],
    }


def _fit_surface_local_model(args: argparse.Namespace) -> tuple[SurfaceLocalTokenJepaModel, Any]:
    _set_seed(int(args.seed))
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    train = build_surface_local_jepa_batch(
        split="train",
        history_len=args.history_len,
        future_len=args.future_len,
        max_windows=args.max_train_windows,
        seed=args.seed,
    )
    val = build_surface_local_jepa_batch(
        split="val",
        history_len=args.history_len,
        future_len=args.future_len,
        max_windows=args.max_val_windows,
        seed=args.seed + 1000,
    )
    descriptors = build_token_descriptor_matrix(train.token_metadata)
    cfg = SurfaceLocalTokenJepaConfig(
        n_tokens=train.token_metadata.n_tokens,
        token_descriptor_dim=descriptors.shape[1],
        token_hidden_dim=args.token_hidden_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        predictor_hidden_dim=args.predictor_hidden_dim,
    )
    model = SurfaceLocalTokenJepaModel(cfg, token_descriptors=descriptors).to(device)
    optimizer = torch.optim.AdamW(
        surface_local_parameter_groups(model),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    train_tensors = _batch_tensors(train, device=device)
    for _epoch in range(int(args.epochs)):
        _run_loss(
            model,
            train_tensors,
            optimizer=optimizer,
            barlow_weight=args.barlow_weight,
            offdiag_weight=args.offdiag_weight,
            grad_clip=args.grad_clip,
        )
        update_ema(model.context_encoder, model.target_encoder, EMA_DECAY)
    return model, val


@torch.no_grad()
def analyze_surface_local_target_geometry(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    model, val = _fit_surface_local_model(args)
    tensors = _batch_tensors(val, device=device)
    outputs, parts = _run_loss(
        model,
        tensors,
        optimizer=None,
        barlow_weight=args.barlow_weight,
        offdiag_weight=args.offdiag_weight,
        grad_clip=args.grad_clip,
    )
    predicted = outputs["predicted_target_tokens"].detach().cpu().numpy()
    target = outputs["target_tokens"].detach().cpu().numpy()
    positions = val.target_positions
    limit = min(int(args.diagnostic_rows), predicted.shape[0])
    if limit < predicted.shape[0]:
        subset_idx = np.linspace(0, predicted.shape[0] - 1, limit, dtype=np.int64)
    else:
        subset_idx = np.arange(predicted.shape[0], dtype=np.int64)
    predicted = predicted[subset_idx]
    target = target[subset_idx]
    positions = positions[subset_idx]

    meta = val.token_metadata
    window_idx = positions[:, 0].astype(np.int64)
    time_idx = positions[:, 1].astype(np.int64)
    token_idx = positions[:, 2].astype(np.int64)
    labels = {
        "target_family": val.target_family[window_idx].astype(str),
        "geometry_id": meta.geometry_id[token_idx].astype(str),
        "factor_id": meta.factor_id[token_idx].astype(str),
        "factor_family": meta.factor_family[token_idx].astype(str),
        "relative_time": time_idx.astype(str),
        "window": window_idx.astype(str),
    }

    pred_scores = _normalize(predicted) @ _normalize(target).T
    pred_topk = _topk(pred_scores, int(args.top_k))
    exact = np.arange(limit)[:, None] == pred_topk
    predictor_rates = {
        "exact_row_any_topk": float(np.mean(np.any(exact, axis=1))),
        "exact_row_neighbor_share": float(np.mean(exact)),
    }
    for name, label in labels.items():
        predictor_rates[name] = _same_label_rates(pred_topk, label, label)

    target_scores = _normalize(target) @ _normalize(target).T
    np.fill_diagonal(target_scores, -np.inf)
    target_topk = _topk(target_scores, int(args.top_k))
    intrinsic_rates = {}
    label_summaries = {}
    baselines = {}
    for name, label in labels.items():
        intrinsic_rates[name] = _same_label_rates(target_topk, label, label)
        label_summaries[name] = _label_summary(label)
        baselines[name] = _label_baseline(label)

    target_factor_ratio = (
        intrinsic_rates["factor_id"]["neighbor_share"] / baselines["factor_id"]
        if baselines["factor_id"] > 0.0
        else 0.0
    )
    target_family_ratio = (
        intrinsic_rates["target_family"]["neighbor_share"] / baselines["target_family"]
        if baselines["target_family"] > 0.0
        else 0.0
    )
    predictor_family_ratio = (
        predictor_rates["target_family"]["neighbor_share"] / baselines["target_family"]
        if baselines["target_family"] > 0.0
        else 0.0
    )
    factor_topk = predictor_rates["factor_id"]["any_topk"]
    exact_topk = predictor_rates["exact_row_any_topk"]
    return {
        "analysis": "world_model_surface_local_target_geometry",
        "date": "2026-05-10",
        "objective_family": "token_geometry_level_context_to_target_jepa",
        "config": asdict(model.cfg),
        "diagnostic_rows": int(limit),
        "diagnostic_subset": "evenly_spaced_across_target_positions",
        "top_k": int(args.top_k),
        "loss_parts": parts,
        "label_summaries": label_summaries,
        "random_neighbor_label_share": baselines,
        "predictor_to_target_topk": predictor_rates,
        "target_intrinsic_topk": intrinsic_rates,
        "decision": {
            "promotion_decision": "DO_NOT_PROMOTE",
            "predictor_retrieves_factor_more_than_exact_row": bool(factor_topk > exact_topk),
            "target_factor_neighbor_over_random": float(target_factor_ratio),
            "target_family_neighbor_over_random": float(target_family_ratio),
            "predictor_target_family_neighbor_over_random": float(predictor_family_ratio),
            "target_latent_factor_dominated": bool(target_factor_ratio > 2.0),
            "diagnosis": (
                "The clean target latent is strongly organized by token/factor and "
                "target-family labels, while predictor-to-target retrieval mostly "
                "recovers coarse geometry/family structure rather than exact rows. "
                "This points to representation geometry, not target coverage, as "
                "the next failure layer."
            ),
            "next_step": (
                "Check whether the surface-local objective needs a target latent "
                "surface with stronger state variation before tuning model size or "
                "mask policy."
            ),
        },
    }


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def _rate_line(result: dict[str, Any], section: str, name: str) -> str:
    row = result[section][name]
    baseline = result["random_neighbor_label_share"].get(name)
    base = "" if baseline is None else f" | {_fmt(baseline)}"
    return f"| {name} | {_fmt(row['any_topk'])} | {_fmt(row['neighbor_share'])}{base} |"


def render_markdown(result: dict[str, Any]) -> str:
    pred = result["predictor_to_target_topk"]
    intrinsic = result["target_intrinsic_topk"]
    decision = result["decision"]
    lines = [
        "# World Model HEAD156: Surface-Local Target Geometry Audit",
        "",
        "Date: 2026-05-10",
        "",
        "## Iteration Type",
        "",
        "`post_experiment_analysis`",
        "",
        "## Objective Family",
        "",
        "`token_geometry_level_context_to_target_jepa` geometry/family diagnosis.",
        "",
        "## Hypothesis",
        "",
        "HEAD154/155 may be failing because selected target-token latents encode",
        "coarse geometry or family identity more readily than exact market-state rows.",
        "",
        "## Predictor To Target Top-K",
        "",
        f"- Diagnostic rows: `{result['diagnostic_rows']}`.",
        f"- Diagnostic subset: `{result['diagnostic_subset']}`.",
        f"- Top-k: `{result['top_k']}`.",
        f"- Exact row any-topk: `{_fmt(pred['exact_row_any_topk'])}`.",
        f"- Exact row neighbor share: `{_fmt(pred['exact_row_neighbor_share'])}`.",
        "",
        "| label | any top-k | neighbor share | random neighbor share |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ("factor_id", "geometry_id", "factor_family", "target_family", "relative_time", "window"):
        lines.append(_rate_line(result, "predictor_to_target_topk", name))
    lines.extend(
        [
            "",
            "## Target Intrinsic Top-K",
            "",
            "| label | any top-k | neighbor share | random neighbor share |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name in ("factor_id", "geometry_id", "factor_family", "target_family", "relative_time", "window"):
        lines.append(_rate_line(result, "target_intrinsic_topk", name))
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Promotion decision: `{decision['promotion_decision']}`.",
            "",
            f"- Predictor retrieves factor more than exact row: `{decision['predictor_retrieves_factor_more_than_exact_row']}`.",
            f"- Target factor neighbor over random: `{_fmt(decision['target_factor_neighbor_over_random'])}`.",
            f"- Target family neighbor over random: `{_fmt(decision['target_family_neighbor_over_random'])}`.",
            f"- Predictor target-family neighbor over random: `{_fmt(decision['predictor_target_family_neighbor_over_random'])}`.",
            f"- Target latent factor dominated: `{decision['target_latent_factor_dominated']}`.",
            "",
            decision["diagnosis"],
            "",
            f"Next: {decision['next_step']}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze surface-local target latent geometry")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--max_train_windows", type=int, default=128)
    parser.add_argument("--max_val_windows", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2154)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--barlow_weight", type=float, default=0.05)
    parser.add_argument("--offdiag_weight", type=float, default=0.005)
    parser.add_argument("--token_hidden_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=48)
    parser.add_argument("--latent_dim", type=int, default=24)
    parser.add_argument("--predictor_hidden_dim", type=int, default=48)
    parser.add_argument("--diagnostic_rows", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output_json", default="results/world/surface_local_target_geometry_head156.json")
    parser.add_argument("--report_md", default="experiments/world/reports/world_model_head156_surface_local_target_geometry.md")
    args = parser.parse_args()

    result = analyze_surface_local_target_geometry(args)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    Path(args.report_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_md).write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result["decision"], indent=2))


if __name__ == "__main__":
    main()
