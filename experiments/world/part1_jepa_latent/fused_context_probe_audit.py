from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from experiments.world.evaluation.part1_metrics import representation_health_metrics  # noqa: E402
from experiments.world.evaluation.world_data import build_iv_world_windows  # noqa: E402
from experiments.world.part1_jepa_latent.context_probe_audit import (  # noqa: E402
    horizon_target_metrics,
    ridge_probe_metrics,
)
from experiments.world.part1_jepa_latent.direct_delta_pca_predictor import (  # noqa: E402
    evaluate_direct_delta_pca,
)
from experiments.world.part1_jepa_latent.fixed_delta_pca_jepa import (  # noqa: E402
    fit_delta_pca_target,
    make_horizon_delta_matrix,
    transform_delta_targets,
)
from experiments.world.part1_jepa_latent.fused_context_delta_pca_predictor import (  # noqa: E402
    FusedContextDeltaPCAConfig,
    FusedContextDeltaPCAPredictor,
    predict_fused_context_delta_pca,
)
from experiments.world.part1_jepa_latent.horizon_jepa_smoke import validate_horizons  # noqa: E402
from experiments.world.part1_jepa_latent.target_encoder_distill import (  # noqa: E402
    pca_oracle_delta_mse,
)


@torch.no_grad()
def encode_fused_contexts(
    model: FusedContextDeltaPCAPredictor,
    past: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    rows = []
    loader = DataLoader(TensorDataset(torch.from_numpy(past)), batch_size=batch_size, shuffle=False)
    for (past_batch,) in loader:
        _predicted, context = model(past_batch.to(device), return_context=True)
        rows.append(context.detach().cpu().numpy())
    return np.concatenate(rows, axis=0)


def load_fused_context_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> FusedContextDeltaPCAPredictor:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = FusedContextDeltaPCAConfig(**checkpoint["config"])
    model = FusedContextDeltaPCAPredictor(cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _serializable(obj):
    if isinstance(obj, dict):
        return {k: _serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    return obj


def audit_fused_context_checkpoint(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    model = load_fused_context_checkpoint(args.checkpoint, device=device)
    horizons = validate_horizons(tuple(model.horizons), future_len=30)

    train = build_iv_world_windows(
        split="train",
        max_windows=args.max_train_windows,
        normalize=True,
    )
    eval_max_windows = args.max_eval_windows
    if eval_max_windows is None:
        eval_max_windows = args.max_val_windows
    eval_batch = build_iv_world_windows(
        split=args.eval_split,
        max_windows=eval_max_windows,
        normalize=True,
    )
    train_delta = make_horizon_delta_matrix(train.past_window, train.future_window, horizons=horizons)
    eval_delta = make_horizon_delta_matrix(
        eval_batch.past_window,
        eval_batch.future_window,
        horizons=horizons,
    )
    pca_target = fit_delta_pca_target(train_delta, target_dim=model.cfg.target_dim)
    train_z = transform_delta_targets(train_delta, pca_target)
    eval_z = transform_delta_targets(eval_delta, pca_target)

    train_contexts = encode_fused_contexts(
        model,
        train.past_window,
        batch_size=args.batch_size,
        device=device,
    )
    eval_contexts = encode_fused_contexts(
        model,
        eval_batch.past_window,
        batch_size=args.batch_size,
        device=device,
    )
    eval_pred, eval_context_from_head = predict_fused_context_delta_pca(
        model,
        eval_batch.past_window,
        batch_size=args.batch_size,
        device=device,
    )
    trained_head_metrics = evaluate_direct_delta_pca(
        predicted_z=eval_pred,
        target_z=eval_z,
        truth_delta=eval_delta,
        pca_target=pca_target,
        horizons=horizons,
        context=eval_context_from_head,
    )
    ridge_fixed_pca_metrics = ridge_probe_metrics(
        train_contexts,
        eval_contexts,
        train_z,
        eval_z,
        horizons=horizons,
        alpha=args.ridge_alpha,
    )
    ridge_delta_metrics = ridge_probe_metrics(
        train_contexts,
        eval_contexts,
        train_delta,
        eval_delta,
        horizons=horizons,
        alpha=args.ridge_alpha,
    )
    zero_delta_metrics = horizon_target_metrics(
        np.zeros_like(eval_delta),
        eval_delta,
        horizons=horizons,
    )
    eval_context_health = representation_health_metrics(eval_contexts)
    eval_shape = {
        "past": list(eval_batch.past_window.shape),
        "future": list(eval_batch.future_window.shape),
        "context": list(eval_contexts.shape),
        "fixed_z": list(eval_z.shape),
        "delta": list(eval_delta.shape),
    }
    result = {
        "checkpoint": str(args.checkpoint),
        "config": {
            "input_dim": model.cfg.input_dim,
            "flat_input_dim": model.cfg.flat_input_dim,
            "hidden_dim": model.cfg.hidden_dim,
            "context_dim": model.cfg.context_dim,
            "target_dim": model.cfg.target_dim,
            "predictor_hidden_dim": model.cfg.predictor_hidden_dim,
            "horizons": list(model.horizons),
        },
        "device": str(device),
        "eval_split": args.eval_split,
        "ridge_alpha": args.ridge_alpha,
        "pca_oracle_delta_mse": pca_oracle_delta_mse(eval_z, eval_delta, pca_target),
        "train_shape": {
            "past": list(train.past_window.shape),
            "future": list(train.future_window.shape),
            "context": list(train_contexts.shape),
            "fixed_z": list(train_z.shape),
            "delta": list(train_delta.shape),
        },
        "eval_shape": eval_shape,
        "val_shape": eval_shape,
        "train_context_health": representation_health_metrics(train_contexts),
        "eval_context_health": eval_context_health,
        "val_context_health": eval_context_health,
        "trained_head_fixed_pca_metrics": trained_head_metrics,
        "ridge_probe_fixed_pca_metrics": ridge_fixed_pca_metrics,
        "ridge_probe_delta_metrics": ridge_delta_metrics,
        "zero_delta_target_baseline": zero_delta_metrics,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(_serializable(result), indent=2), encoding="utf-8")
    print(json.dumps(_serializable(result["eval_context_health"]), indent=2))
    print(json.dumps(_serializable(result["ridge_probe_fixed_pca_metrics"]["overall_prediction"]), indent=2))
    print(json.dumps(_serializable(result["ridge_probe_fixed_pca_metrics"]["overall_retrieval"]), indent=2))
    print(json.dumps(_serializable(result["ridge_probe_delta_metrics"]["overall_prediction"]), indent=2))
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit fused fixed-target context embeddings with ridge probes")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/world/checkpoints/part1_jepa_latent/fused_context_delta_pca_head038_seed7711.pt",
    )
    parser.add_argument("--max_train_windows", type=int, default=2048)
    parser.add_argument("--max_val_windows", type=int, default=256)
    parser.add_argument("--max_eval_windows", type=int, default=None)
    parser.add_argument("--eval_split", choices=("val", "test"), default="val")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ridge_alpha", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_json",
        type=str,
        default="results/world/part1_fused_context_probe_audit_head039.json",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    audit_fused_context_checkpoint(args)


if __name__ == "__main__":
    main()
