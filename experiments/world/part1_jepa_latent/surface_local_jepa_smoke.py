from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, ".")

from experiments.world.evaluation.part1_metrics import (  # noqa: E402
    latent_prediction_metrics,
    representation_health_metrics,
    retrieval_metrics,
)
from experiments.world.evaluation.surface_local_jepa_data import (  # noqa: E402
    SurfaceLocalJepaBatch,
    build_surface_local_jepa_batch,
)
from experiments.world.part1_jepa_latent.jepa_smoke import update_ema  # noqa: E402
from experiments.world.part1_jepa_latent.masked_multiview_geometry_barlow_smoke import (  # noqa: E402
    build_token_descriptor_matrix,
)
from experiments.world.part1_jepa_latent.surface_local_jepa_model import (  # noqa: E402
    SurfaceLocalTokenJepaConfig,
    SurfaceLocalTokenJepaModel,
    surface_local_context_target_loss,
    surface_local_parameter_groups,
)


EMA_DECAY = 0.99


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _batch_tensors(
    batch: SurfaceLocalJepaBatch,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "context_values": torch.from_numpy(batch.context_values).to(device),
        "clean_values": torch.from_numpy(batch.clean_values).to(device),
        "observed_mask": torch.from_numpy(batch.observed_mask).to(device),
        "context_mask": torch.from_numpy(batch.context_mask).to(device),
        "target_positions": torch.from_numpy(batch.target_positions).to(device),
    }


def _run_loss(
    model: SurfaceLocalTokenJepaModel,
    tensors: dict[str, torch.Tensor],
    *,
    optimizer: torch.optim.Optimizer | None,
    barlow_weight: float,
    offdiag_weight: float,
    grad_clip: float,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        outputs = model(**tensors)
        loss, parts = surface_local_context_target_loss(
            outputs,
            barlow_weight=barlow_weight,
            offdiag_weight=offdiag_weight,
        )
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(surface_local_parameter_groups(model)),
                grad_clip,
            )
            optimizer.step()
            update_ema(model.context_encoder, model.target_encoder, EMA_DECAY)
    return outputs, parts


@torch.no_grad()
def _evaluate(
    model: SurfaceLocalTokenJepaModel,
    batch: SurfaceLocalJepaBatch,
    *,
    device: torch.device,
    barlow_weight: float,
    offdiag_weight: float,
    retrieval_eval_rows: int,
) -> dict[str, Any]:
    tensors = _batch_tensors(batch, device=device)
    outputs, parts = _run_loss(
        model,
        tensors,
        optimizer=None,
        barlow_weight=barlow_weight,
        offdiag_weight=offdiag_weight,
        grad_clip=1.0,
    )
    predicted = outputs["predicted_target_tokens"].detach().cpu().numpy()
    target = outputs["target_tokens"].detach().cpu().numpy()
    limit = min(int(retrieval_eval_rows), predicted.shape[0])
    retrieval_subset = retrieval_metrics(
        predicted[:limit],
        target[:limit],
        top_k=(1, 5, 10),
    )
    return {
        **parts,
        "prediction_metrics": latent_prediction_metrics(predicted, target),
        "predicted_health": representation_health_metrics(predicted),
        "target_health": representation_health_metrics(target),
        "retrieval_subset_rows": int(limit),
        "retrieval_subset": retrieval_subset,
    }


def run_surface_local_jepa_smoke(args: argparse.Namespace) -> dict[str, Any]:
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
    initial_val = _evaluate(
        model,
        val,
        device=device,
        barlow_weight=args.barlow_weight,
        offdiag_weight=args.offdiag_weight,
        retrieval_eval_rows=args.retrieval_eval_rows,
    )
    train_history = []
    for epoch in range(int(args.epochs)):
        _outputs, parts = _run_loss(
            model,
            train_tensors,
            optimizer=optimizer,
            barlow_weight=args.barlow_weight,
            offdiag_weight=args.offdiag_weight,
            grad_clip=args.grad_clip,
        )
        train_history.append({"epoch": epoch + 1, **parts})
    final_val = _evaluate(
        model,
        val,
        device=device,
        barlow_weight=args.barlow_weight,
        offdiag_weight=args.offdiag_weight,
        retrieval_eval_rows=args.retrieval_eval_rows,
    )

    result = {
        "analysis": "world_model_surface_local_jepa_smoke",
        "date": "2026-05-10",
        "objective_family": "token_geometry_level_context_to_target_jepa",
        "uses_future_targets": False,
        "uses_value_reconstruction": False,
        "ema_decay": EMA_DECAY,
        "device": str(device),
        "config": asdict(cfg),
        "train_windows": int(train.clean_values.shape[0]),
        "val_windows": int(val.clean_values.shape[0]),
        "train_target_token_rows": int(train.target_positions.shape[0]),
        "val_target_token_rows": int(val.target_positions.shape[0]),
        "initial_val_loss": float(initial_val["loss"]),
        "final_val_loss": float(final_val["loss"]),
        "loss_delta": float(final_val["loss"] - initial_val["loss"]),
        "train_loss_first": float(train_history[0]["loss"]) if train_history else None,
        "train_loss_last": float(train_history[-1]["loss"]) if train_history else None,
        "train_history": train_history,
        "val_alignment": float(final_val["alignment"]),
        "val_barlow": float(final_val["barlow"]),
        "val_prediction_metrics": final_val["prediction_metrics"],
        "val_predicted_health": final_val["predicted_health"],
        "val_target_health": final_val["target_health"],
        "val_retrieval_subset_rows": int(final_val["retrieval_subset_rows"]),
        "val_retrieval_subset": final_val["retrieval_subset"],
        "promotion_decision": "SMOKE_ONLY_DO_NOT_PROMOTE",
        "part_b_blocked": True,
    }
    output_json = getattr(args, "output_json", None)
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    checkpoint = getattr(args, "checkpoint", None)
    if checkpoint:
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(cfg),
                "model_state_dict": model.state_dict(),
                "result": result,
            },
            checkpoint,
        )
    report_md = getattr(args, "report_md", None)
    if report_md:
        Path(report_md).parent.mkdir(parents=True, exist_ok=True)
        Path(report_md).write_text(render_markdown(result), encoding="utf-8")
    return result


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def render_markdown(result: dict[str, Any]) -> str:
    pred = result["val_prediction_metrics"]
    retrieval = result["val_retrieval_subset"]
    pred_health = result["val_predicted_health"]
    target_health = result["val_target_health"]
    lines = [
        "# World Model HEAD154: Surface-Local Token JEPA Smoke",
        "",
        "Date: 2026-05-10",
        "",
        "## Iteration Type",
        "",
        "`experiment`",
        "",
        "## Objective Family",
        "",
        "`token_geometry_level_context_to_target_jepa` smoke training.",
        "",
        "## Hypothesis",
        "",
        "A token/geometry-level context-to-target scaffold should train without",
        "future targets or value reconstruction and should expose whether target-token",
        "alignment is healthy or another low-rank/high-cosine shortcut appears.",
        "",
        "## Run",
        "",
        f"- Train windows: `{result['train_windows']}`.",
        f"- Validation windows: `{result['val_windows']}`.",
        f"- Train target token rows: `{result['train_target_token_rows']}`.",
        f"- Validation target token rows: `{result['val_target_token_rows']}`.",
        f"- EMA decay: `{_fmt(result['ema_decay'])}`.",
        "",
        "## Metrics",
        "",
        f"- Initial validation loss: `{_fmt(result['initial_val_loss'])}`.",
        f"- Final validation loss: `{_fmt(result['final_val_loss'])}`.",
        f"- Validation alignment: `{_fmt(result['val_alignment'])}`.",
        f"- Validation cosine mean: `{_fmt(pred['cosine_mean'])}`.",
        f"- Validation retrieval top10 on subset: `{_fmt(retrieval['top10'])}`.",
        f"- Retrieval subset rows: `{result['val_retrieval_subset_rows']}`.",
        f"- Predicted effective rank: `{_fmt(pred_health['effective_rank'])}`.",
        f"- Target effective rank: `{_fmt(target_health['effective_rank'])}`.",
        f"- Predicted offdiag abs mean: `{_fmt(pred_health['offdiag_abs_mean'])}`.",
        f"- Target offdiag abs mean: `{_fmt(target_health['offdiag_abs_mean'])}`.",
        "",
        "## Decision",
        "",
        f"Promotion decision: `{result['promotion_decision']}`.",
        "",
        "This is smoke evidence only. It does not certify Part 1, does not start",
        "Part B, and should be followed by a diagnostic comparison against the scaled",
        "Barlow candidate and raw exact-state baselines before any tuning.",
    ]
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Surface-local token JEPA smoke")
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
    parser.add_argument("--retrieval_eval_rows", type=int, default=512)
    parser.add_argument("--output_json", default="results/world/surface_local_jepa_smoke_head154.json")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--report_md", default="experiments/world/reports/world_model_head154_surface_local_jepa_smoke.md")
    return parser


def main() -> None:
    result = run_surface_local_jepa_smoke(_build_parser().parse_args())
    print(json.dumps({k: result[k] for k in (
        "initial_val_loss",
        "final_val_loss",
        "val_alignment",
        "val_target_token_rows",
        "val_retrieval_subset",
        "promotion_decision",
    )}, indent=2))


if __name__ == "__main__":
    main()
