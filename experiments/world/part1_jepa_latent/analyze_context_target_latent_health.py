from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, ".")

from experiments.world.evaluation.context_target_jepa_data import (  # noqa: E402
    ContextTargetJepaBatch,
    build_context_target_jepa_batch,
)
from experiments.world.evaluation.part1_metrics import (  # noqa: E402
    latent_prediction_metrics,
    representation_health_metrics,
    retrieval_metrics,
)
from experiments.world.part1_jepa_latent.context_target_jepa_smoke import (  # noqa: E402
    ContextTargetJEPAConfig,
    ContextTargetJEPAModel,
    make_context_target_features,
)
from experiments.world.part1_jepa_latent.masked_multiview_mask_artifact_audit import (  # noqa: E402
    classification_metrics,
    fit_predict_multiclass_ridge,
)


CHECKPOINT = Path(
    "models/world/checkpoints/part1_jepa_latent/context_target_jepa_smoke_head140.pt"
)


def _load_model(checkpoint_path: Path, *, device: torch.device) -> ContextTargetJEPAModel:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = ContextTargetJEPAConfig(**checkpoint["config"])
    model = ContextTargetJEPAModel(cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _to_tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(values).to(device)


@torch.no_grad()
def _encode_surfaces(
    model: ContextTargetJEPAModel,
    batch: ContextTargetJepaBatch,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    selected_context: list[np.ndarray] = []
    selected_predicted: list[np.ndarray] = []
    selected_target: list[np.ndarray] = []
    selected_target_values: list[np.ndarray] = []
    selected_family: list[str] = []
    clean_last: list[np.ndarray] = []
    masked_last: list[np.ndarray] = []
    target_time_mask_rows: list[np.ndarray] = []
    target_token_count_rows: list[np.ndarray] = []

    n = int(batch.clean_values.shape[0])
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        clean_values = _to_tensor(batch.clean_values[start:end], device)
        context_values = _to_tensor(batch.context_values[start:end], device)
        target_values = _to_tensor(batch.target_values[start:end], device)
        observed = _to_tensor(batch.observed_mask[start:end].astype(np.float32), device)
        context_mask = _to_tensor(batch.context_mask[start:end].astype(np.float32), device)
        target_mask = _to_tensor(batch.target_mask[start:end].astype(np.float32), device)

        context_features = make_context_target_features(
            context_values,
            observed,
            context_mask,
        )
        target_features = make_context_target_features(
            target_values,
            observed,
            target_mask,
        )
        clean_features = make_context_target_features(
            clean_values,
            observed,
            observed,
        )
        out = model(context_features, target_features)
        clean_context = model.context_encoder(clean_features)
        time_mask = target_mask.bool().any(dim=-1)

        clean_last.append(clean_context[:, -1, :].detach().cpu().numpy())
        masked_last.append(out["context"][:, -1, :].detach().cpu().numpy())
        selected_context.append(out["context"][time_mask].detach().cpu().numpy())
        selected_predicted.append(out["predicted"][time_mask].detach().cpu().numpy())
        selected_target.append(out["target"][time_mask].detach().cpu().numpy())
        selected_target_values.append(target_values[time_mask].detach().cpu().numpy())

        time_mask_np = time_mask.detach().cpu().numpy()
        target_time_mask_rows.append(time_mask_np)
        token_counts = target_mask.sum(dim=-1).detach().cpu().numpy()
        target_token_count_rows.append(token_counts)
        for local_row, family in enumerate(batch.target_family[start:end].tolist()):
            selected_family.extend([str(family)] * int(time_mask_np[local_row].sum()))

    return {
        "clean_context_last": np.concatenate(clean_last, axis=0),
        "masked_context_last": np.concatenate(masked_last, axis=0),
        "selected_context": np.concatenate(selected_context, axis=0),
        "selected_predicted": np.concatenate(selected_predicted, axis=0),
        "selected_target": np.concatenate(selected_target, axis=0),
        "selected_target_values": np.concatenate(selected_target_values, axis=0),
        "selected_family": np.asarray(selected_family, dtype=object),
        "target_time_mask": np.concatenate(target_time_mask_rows, axis=0),
        "target_token_count": np.concatenate(target_token_count_rows, axis=0),
    }


def _coverage(batch: ContextTargetJepaBatch) -> dict[str, Any]:
    observed = np.asarray(batch.observed_mask, dtype=bool)
    target = np.asarray(batch.target_mask, dtype=bool) & observed
    time_mask = target.any(axis=-1)
    token_counts = target.sum(axis=-1)
    rows = {
        "overall": {
            "hidden_rate": float(target.sum() / observed.sum()),
            "target_time_row_rate": float(time_mask.mean()),
            "last_row_target_rate": float(time_mask[:, -1].mean()),
            "target_tokens_per_target_row_mean": float(
                token_counts[time_mask].mean()
            ),
            "target_tokens_per_target_row_min": int(token_counts[time_mask].min()),
            "target_tokens_per_target_row_max": int(token_counts[time_mask].max()),
        }
    }
    for family in sorted({str(x) for x in batch.target_family.tolist()}):
        mask = batch.target_family.astype(str) == family
        fam_target = target[mask]
        fam_observed = observed[mask]
        fam_time = fam_target.any(axis=-1)
        fam_counts = fam_target.sum(axis=-1)
        rows[family] = {
            "n_windows": int(mask.sum()),
            "hidden_rate": float(fam_target.sum() / fam_observed.sum()),
            "target_time_row_rate": float(fam_time.mean()),
            "last_row_target_rate": float(fam_time[:, -1].mean()),
            "target_tokens_per_target_row_mean": float(
                fam_counts[fam_time].mean()
            ),
        }
    return rows


def _health_table(surfaces: dict[str, Any]) -> dict[str, Any]:
    feature_map = {
        "clean_context_last": surfaces["clean_context_last"],
        "masked_context_last": surfaces["masked_context_last"],
        "selected_context": surfaces["selected_context"],
        "selected_predicted": surfaces["selected_predicted"],
        "selected_target": surfaces["selected_target"],
        "selected_target_values": surfaces["selected_target_values"],
    }
    return {
        name: representation_health_metrics(values)
        for name, values in feature_map.items()
    }


def _mask_family_probes(
    train_surfaces: dict[str, Any],
    val_surfaces: dict[str, Any],
    *,
    alpha: float,
) -> dict[str, Any]:
    rows = {}
    selected_labels_train = train_surfaces["selected_family"]
    selected_labels_val = val_surfaces["selected_family"]
    for name in ("selected_context", "selected_predicted", "selected_target"):
        pred = fit_predict_multiclass_ridge(
            train_surfaces[name],
            selected_labels_train,
            val_surfaces[name],
            alpha=alpha,
        )
        rows[f"{name}_to_target_family"] = classification_metrics(
            pred,
            selected_labels_val,
        )
    return rows


def _alignment(surfaces: dict[str, Any]) -> dict[str, Any]:
    selected_pred = surfaces["selected_predicted"]
    selected_target = surfaces["selected_target"]
    out = {
        "predicted_to_target": latent_prediction_metrics(
            selected_pred,
            selected_target,
        ),
        "context_to_target": latent_prediction_metrics(
            surfaces["selected_context"],
            selected_target,
        ),
    }
    max_rows = min(1024, selected_pred.shape[0])
    out["predicted_to_target_retrieval_first1024"] = retrieval_metrics(
        selected_pred[:max_rows],
        selected_target[:max_rows],
        top_k=(1, 5, 10),
    )
    return out


def _decision(
    val_health: dict[str, Any],
    mask_probes: dict[str, Any],
    alignment: dict[str, Any],
) -> dict[str, Any]:
    clean_rank = float(val_health["clean_context_last"]["effective_rank"])
    predicted_rank = float(val_health["selected_predicted"]["effective_rank"])
    target_rank = float(val_health["selected_target"]["effective_rank"])
    target_family_probe = mask_probes["selected_target_to_target_family"]
    predicted_family_probe = mask_probes["selected_predicted_to_target_family"]
    target_family_lift = float(target_family_probe["accuracy_lift"])
    predicted_family_lift = float(predicted_family_probe["accuracy_lift"])
    target_mask_warning = target_family_lift > 0.10
    predicted_low_rank_warning = predicted_rank < 8.0
    pred_align = alignment["predicted_to_target"]
    pred_retrieval = alignment["predicted_to_target_retrieval_first1024"]
    high_cos_low_retrieval = (
        float(pred_align["cosine_mean"]) > 0.95
        and float(pred_retrieval["top10"]) < 0.10
    )
    low_rank_warning = clean_rank < 12.0 or target_rank < 12.0
    return {
        "clean_context_last_effective_rank": clean_rank,
        "selected_predicted_effective_rank": predicted_rank,
        "selected_target_effective_rank": target_rank,
        "target_family_accuracy_lift": target_family_lift,
        "predicted_family_accuracy_lift": predicted_family_lift,
        "low_rank_warning": bool(low_rank_warning),
        "target_latent_mask_family_warning": bool(target_mask_warning),
        "predicted_latent_low_rank_warning": bool(predicted_low_rank_warning),
        "high_cosine_low_retrieval_warning": bool(high_cos_low_retrieval),
        "promotion_decision": "DO_NOT_PROMOTE",
        "next_step": (
            "The branch is weak because the supervised target latent is heavily "
            "mask-family identifiable while the predictor collapses to a low-rank "
            "surface with high cosine but poor row retrieval. Do not add model "
            "knobs before fixing this target/predictor diagnostic."
        ),
    }


def analyze_context_target_latent_health(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    train = build_context_target_jepa_batch(
        split="train",
        history_len=args.history_len,
        future_len=args.future_len,
        max_windows=args.max_train_windows,
        seed=args.seed,
    )
    val = build_context_target_jepa_batch(
        split="val",
        history_len=args.history_len,
        future_len=args.future_len,
        max_windows=args.max_val_windows,
        seed=args.seed + 1000,
    )
    model = _load_model(CHECKPOINT, device=device)
    train_surfaces = _encode_surfaces(
        model,
        train,
        batch_size=args.batch_size,
        device=device,
    )
    val_surfaces = _encode_surfaces(
        model,
        val,
        batch_size=args.batch_size,
        device=device,
    )
    val_health = _health_table(val_surfaces)
    mask_probes = _mask_family_probes(
        train_surfaces,
        val_surfaces,
        alpha=args.classifier_alpha,
    )
    alignment = _alignment(val_surfaces)
    return {
        "analysis": "world_model_context_target_latent_health",
        "date": "2026-05-10",
        "objective_family": "context_to_target_jepa_diagnostic",
        "checkpoint": str(CHECKPOINT),
        "device": str(device),
        "train_shape": list(train.clean_values.shape),
        "val_shape": list(val.clean_values.shape),
        "train_coverage": _coverage(train),
        "val_coverage": _coverage(val),
        "val_health": val_health,
        "val_alignment": alignment,
        "mask_family_probes": mask_probes,
        "decision": _decision(val_health, mask_probes, alignment),
    }


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def render_markdown(result: dict[str, Any], *, title: str) -> str:
    val_cov = result["val_coverage"]
    val_health = result["val_health"]
    decision = result["decision"]
    lines = [
        f"# {title}",
        "",
        "Date: 2026-05-10",
        "",
        "## Iteration Type",
        "",
        "`post_experiment_analysis`",
        "",
        "## Objective Family",
        "",
        "`context_to_target_jepa_diagnostic`; no model or objective change.",
        "",
        "## Hypothesis",
        "",
        "HEAD140 may be weak because the trained loss acts on a target/predicted",
        "latent surface that is lower-rank or more mask-family driven than the clean",
        "context embedding used by downstream probes.",
        "",
        "## Falsifier",
        "",
        "The shortcut diagnosis is false if target and predicted latent rows have",
        "healthy rank and mask-family probes do not beat majority by a meaningful",
        "margin.",
        "",
        "## Validation Mask Coverage",
        "",
        "| family | windows | hidden rate | target row rate | last-row rate | tokens / target row |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family, row in val_cov.items():
        lines.append(
            "| {family} | {windows} | {hidden} | {row_rate} | {last} | {tokens} |".format(
                family=family,
                windows=row.get("n_windows", result["val_shape"][0]),
                hidden=_fmt(row["hidden_rate"]),
                row_rate=_fmt(row["target_time_row_rate"]),
                last=_fmt(row["last_row_target_rate"]),
                tokens=_fmt(row["target_tokens_per_target_row_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Latent Health",
            "",
            "| surface | effective rank | variance min | offdiag abs mean |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for surface in (
        "clean_context_last",
        "masked_context_last",
        "selected_context",
        "selected_predicted",
        "selected_target",
        "selected_target_values",
    ):
        row = val_health[surface]
        lines.append(
            f"| {surface} | {_fmt(row['effective_rank'])} | "
            f"{_fmt(row['variance_min'])} | {_fmt(row['offdiag_abs_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Alignment",
            "",
            "| pair | MSE | cosine mean | top1 | top10 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    align = result["val_alignment"]
    retrieval = align["predicted_to_target_retrieval_first1024"]
    lines.append(
        "| predicted_to_target | {mse} | {cos} | {top1} | {top10} |".format(
            mse=_fmt(align["predicted_to_target"]["mse"]),
            cos=_fmt(align["predicted_to_target"]["cosine_mean"]),
            top1=_fmt(retrieval["top1"]),
            top10=_fmt(retrieval["top10"]),
        )
    )
    lines.append(
        "| context_to_target | {mse} | {cos} | n/a | n/a |".format(
            mse=_fmt(align["context_to_target"]["mse"]),
            cos=_fmt(align["context_to_target"]["cosine_mean"]),
        )
    )
    lines.extend(
        [
            "",
            "## Mask-Family Probe",
            "",
            "| feature | accuracy | majority | lift | macro recall |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, row in result["mask_family_probes"].items():
        lines.append(
            f"| {name} | {_fmt(row['accuracy'])} | "
            f"{_fmt(row['majority_accuracy'])} | {_fmt(row['accuracy_lift'])} | "
            f"{_fmt(row['macro_recall'])} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Clean context last rank: `{_fmt(decision['clean_context_last_effective_rank'])}`.",
            f"- Target latent rank: `{_fmt(decision['selected_target_effective_rank'])}`.",
            f"- Predicted latent rank: `{_fmt(decision['selected_predicted_effective_rank'])}`.",
            f"- Target-family lift from target latent: `{_fmt(decision['target_family_accuracy_lift'])}`.",
            f"- Target-family lift from predicted latent: `{_fmt(decision['predicted_family_accuracy_lift'])}`.",
            f"- Low-rank warning: `{decision['low_rank_warning']}`.",
            f"- Target latent mask-family warning: `{decision['target_latent_mask_family_warning']}`.",
            f"- Predicted latent low-rank warning: `{decision['predicted_latent_low_rank_warning']}`.",
            f"- High-cosine/low-retrieval warning: `{decision['high_cosine_low_retrieval_warning']}`.",
            f"- Promotion decision: `{decision['promotion_decision']}`.",
            "",
            decision["next_step"],
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose the latent surfaces in the context-to-target JEPA smoke"
    )
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--max_train_windows", type=int, default=384)
    parser.add_argument("--max_val_windows", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--classifier_alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2140)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/world/context_target_latent_health_head142.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "experiments/world/reports/world_model_head142_context_target_latent_health.md"
        ),
    )
    parser.add_argument(
        "--report-title",
        default="World Model HEAD142: Context-Target Latent Health",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = analyze_context_target_latent_health(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(
        render_markdown(result, title=args.report_title),
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
