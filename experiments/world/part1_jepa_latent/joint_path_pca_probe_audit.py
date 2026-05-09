from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from experiments.world.evaluation.part1_metrics import (  # noqa: E402
    latent_prediction_metrics,
    representation_health_metrics,
    retrieval_metrics,
)
from experiments.world.evaluation.world_data import build_iv_world_windows  # noqa: E402
from experiments.world.part1_jepa_latent.context_probe_audit import (  # noqa: E402
    horizon_target_metrics,
    ridge_probe_metrics,
    ridge_probe_predict,
)
from experiments.world.part1_jepa_latent.fixed_delta_pca_jepa import (  # noqa: E402
    make_horizon_delta_matrix,
)
from experiments.world.part1_jepa_latent.horizon_jepa_smoke import validate_horizons  # noqa: E402
from experiments.world.part1_jepa_latent.joint_path_pca_predictor import (  # noqa: E402
    FusedContextJointPathPCAConfig,
    FusedContextJointPathPCAPredictor,
    JointDeltaPathPCATarget,
    evaluate_joint_path_pca,
    inverse_transform_joint_delta_path_targets,
    predict_joint_path_pca,
    transform_joint_delta_path_targets,
)


def joint_path_pca_target_from_contract(
    contract: dict[str, object],
    *,
    input_dim: int,
) -> JointDeltaPathPCATarget:
    horizons = tuple(int(h) for h in contract["horizons"])
    mean = np.asarray(contract["mean"], dtype=np.float64)
    components = np.asarray(contract["components"], dtype=np.float64)
    scale = np.asarray(contract["scale"], dtype=np.float64)
    expected_flat_dim = len(horizons) * int(input_dim)
    if mean.shape != (expected_flat_dim,):
        raise ValueError(f"target mean shape {mean.shape} != {(expected_flat_dim,)}")
    if components.shape[0] != expected_flat_dim:
        raise ValueError(
            f"target component input dimension {components.shape[0]} != {expected_flat_dim}"
        )
    if scale.shape != (components.shape[1],):
        raise ValueError(f"target scale shape {scale.shape} != {(components.shape[1],)}")
    return JointDeltaPathPCATarget(
        mean=mean,
        components=components,
        scale=scale,
        horizons=horizons,
        input_dim=int(input_dim),
    )


def load_joint_path_pca_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[FusedContextJointPathPCAPredictor, JointDeltaPathPCATarget]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = FusedContextJointPathPCAConfig(**checkpoint["config"])
    model = FusedContextJointPathPCAPredictor(cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    target = joint_path_pca_target_from_contract(
        checkpoint["result"]["target_contract"],
        input_dim=cfg.input_dim,
    )
    return model, target


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


def audit_joint_path_pca_checkpoint(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    model, pca_target = load_joint_path_pca_checkpoint(args.checkpoint, device=device)
    horizons = validate_horizons(tuple(model.horizons), future_len=30)
    if horizons != pca_target.horizons:
        raise ValueError(f"model horizons {horizons} != target horizons {pca_target.horizons}")

    train = build_iv_world_windows(
        split="train",
        max_windows=args.max_train_windows,
        normalize=True,
    )
    eval_batch = build_iv_world_windows(
        split=args.eval_split,
        max_windows=args.max_eval_windows,
        normalize=True,
    )
    train_delta = make_horizon_delta_matrix(
        train.past_window,
        train.future_window,
        horizons=horizons,
    )
    eval_delta = make_horizon_delta_matrix(
        eval_batch.past_window,
        eval_batch.future_window,
        horizons=horizons,
    )
    train_z = transform_joint_delta_path_targets(train_delta, pca_target)
    eval_z = transform_joint_delta_path_targets(eval_delta, pca_target)

    train_pred, train_context = predict_joint_path_pca(
        model,
        train.past_window,
        batch_size=args.batch_size,
        device=device,
    )
    eval_pred, eval_context = predict_joint_path_pca(
        model,
        eval_batch.past_window,
        batch_size=args.batch_size,
        device=device,
    )
    trained_head_metrics = evaluate_joint_path_pca(
        predicted_z=eval_pred,
        target_z=eval_z,
        truth_delta=eval_delta,
        pca_target=pca_target,
        horizons=horizons,
        context=eval_context,
    )
    train_head_metrics = {
        "path_prediction": latent_prediction_metrics(train_pred, train_z),
        "path_retrieval": retrieval_metrics(train_pred, train_z, top_k=(1, 5, 10)),
    }

    ridge_path_pred = ridge_probe_predict(train_context, train_z, eval_context, alpha=args.ridge_alpha)
    ridge_path_metrics = {
        "prediction": latent_prediction_metrics(ridge_path_pred, eval_z),
        "retrieval": retrieval_metrics(ridge_path_pred, eval_z, top_k=(1, 5, 10)),
    }
    ridge_delta_metrics = ridge_probe_metrics(
        train_context,
        eval_context,
        train_delta,
        eval_delta,
        horizons=horizons,
        alpha=args.ridge_alpha,
    )
    pca_oracle_delta = inverse_transform_joint_delta_path_targets(eval_z, pca_target)
    pca_oracle_delta_metrics = horizon_target_metrics(
        pca_oracle_delta,
        eval_delta,
        horizons=horizons,
    )
    zero_delta_metrics = horizon_target_metrics(
        np.zeros_like(eval_delta),
        eval_delta,
        horizons=horizons,
    )
    eval_context_health = representation_health_metrics(eval_context)
    eval_shape = {
        "past": list(eval_batch.past_window.shape),
        "future": list(eval_batch.future_window.shape),
        "context": list(eval_context.shape),
        "target_z": list(eval_z.shape),
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
            "horizons": list(horizons),
        },
        "device": str(device),
        "eval_split": args.eval_split,
        "ridge_alpha": args.ridge_alpha,
        "train_shape": {
            "past": list(train.past_window.shape),
            "future": list(train.future_window.shape),
            "context": list(train_context.shape),
            "target_z": list(train_z.shape),
            "delta": list(train_delta.shape),
        },
        "eval_shape": eval_shape,
        "val_shape": eval_shape,
        "train_context_health": representation_health_metrics(train_context),
        "eval_context_health": eval_context_health,
        "val_context_health": eval_context_health,
        "train_head_metrics": train_head_metrics,
        "trained_head_joint_path_metrics": trained_head_metrics,
        "ridge_probe_path_metrics": ridge_path_metrics,
        "ridge_probe_delta_metrics": ridge_delta_metrics,
        "pca_oracle_delta_metrics": pca_oracle_delta_metrics,
        "zero_delta_target_baseline": zero_delta_metrics,
        "reference_thresholds": {
            "head042_primary_test_trained_decoded_delta_mse": 0.017769,
            "head042_primary_test_trained_fixed_pca_mrr": 0.119919,
            "head042_primary_test_trained_fixed_pca_top5": 0.155469,
            "head042_primary_test_raw_delta_mse": 0.017923,
            "head042_primary_test_raw_delta_mrr": 0.110315,
            "head042_primary_test_raw_delta_top5": 0.142969,
            "head042_primary_test_zero_delta_mse": 0.025450,
        },
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(_serializable(result), indent=2), encoding="utf-8")
    print(json.dumps(_serializable(result["eval_context_health"]), indent=2))
    print(json.dumps(_serializable(result["trained_head_joint_path_metrics"]["delta_decode"]["overall_prediction"]), indent=2))
    print(json.dumps(_serializable(result["ridge_probe_delta_metrics"]["overall_prediction"]), indent=2))
    print(json.dumps(_serializable(result["ridge_probe_delta_metrics"]["overall_retrieval"]), indent=2))
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit joint path-PCA Part 1 checkpoint on a held-out split")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/world/checkpoints/part1_jepa_latent/joint_path_pca_head060.pt",
    )
    parser.add_argument("--max_train_windows", type=int, default=2048)
    parser.add_argument("--max_eval_windows", type=int, default=256)
    parser.add_argument("--eval_split", choices=("val", "test"), default="test")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ridge_alpha", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_json",
        type=str,
        default="results/world/part1_joint_path_pca_probe_audit_head061_test.json",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    audit_joint_path_pca_checkpoint(args)


if __name__ == "__main__":
    main()
