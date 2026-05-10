from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _float(value: Any) -> float:
    return float(value)


def analyze_surface_local_smoke_failure(result: dict[str, Any]) -> dict[str, Any]:
    latent_dim = int(result["config"]["latent_dim"])
    pred_health = result["val_predicted_health"]
    target_health = result["val_target_health"]
    retrieval = result["val_retrieval_subset"]
    subset_rows = int(result["val_retrieval_subset_rows"])

    random_top10 = min(10, subset_rows) / max(subset_rows, 1)
    random_top5 = min(5, subset_rows) / max(subset_rows, 1)
    random_top1 = 1 / max(subset_rows, 1)
    pred_rank = _float(pred_health["effective_rank"])
    target_rank = _float(target_health["effective_rank"])
    pred_var = _float(pred_health["variance_mean"])
    target_var = _float(target_health["variance_mean"])
    variance_ratio = pred_var / target_var if target_var > 0.0 else 0.0
    top10 = _float(retrieval["top10"])

    return {
        "analysis": "world_model_surface_local_smoke_failure",
        "date": "2026-05-10",
        "source_result": "results/world/surface_local_jepa_smoke_head154.json",
        "objective_family": result["objective_family"],
        "loss_improved": bool(_float(result["final_val_loss"]) < _float(result["initial_val_loss"])),
        "loss_delta": _float(result["loss_delta"]),
        "retrieval": {
            "top1": _float(retrieval["top1"]),
            "top5": _float(retrieval["top5"]),
            "top10": top10,
            "random_top1": random_top1,
            "random_top5": random_top5,
            "random_top10": random_top10,
            "top10_over_random": top10 / random_top10 if random_top10 > 0.0 else 0.0,
            "median_rank": _float(retrieval["median_rank"]),
            "subset_rows": subset_rows,
        },
        "rank": {
            "latent_dim": latent_dim,
            "predicted_effective_rank": pred_rank,
            "target_effective_rank": target_rank,
            "predicted_rank_fraction": pred_rank / latent_dim,
            "target_rank_fraction": target_rank / latent_dim,
            "predicted_to_target_rank_ratio": pred_rank / target_rank if target_rank > 0.0 else 0.0,
        },
        "variance": {
            "predicted_variance_mean": pred_var,
            "target_variance_mean": target_var,
            "predicted_to_target_variance_ratio": variance_ratio,
            "predicted_variance_min": _float(pred_health["variance_min"]),
            "target_variance_min": _float(target_health["variance_min"]),
        },
        "redundancy": {
            "predicted_offdiag_abs_mean": _float(pred_health["offdiag_abs_mean"]),
            "target_offdiag_abs_mean": _float(target_health["offdiag_abs_mean"]),
            "predicted_offdiag_abs_max": _float(pred_health["offdiag_abs_max"]),
            "target_offdiag_abs_max": _float(target_health["offdiag_abs_max"]),
        },
        "decision": {
            "promotion_decision": "DO_NOT_PROMOTE",
            "looks_like_predictor_shrinkage": bool(variance_ratio < 0.25),
            "target_latent_also_low_rank": bool(target_rank / latent_dim < 0.25),
            "retrieval_only_weakly_above_random": bool(top10 < 3.0 * random_top10),
            "diagnosis": (
                "The smoke reduces loss, but both the target and predicted token "
                "latents are low-rank; the predictor additionally shrinks variance. "
                "This is not a mask-coverage problem and should be diagnosed before "
                "any architecture knob tuning."
            ),
            "next_step": (
                "Audit selected target-token latents by geometry/family and compare "
                "predictor retrieval against target-latent intrinsic separability."
            ),
        },
    }


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def render_markdown(result: dict[str, Any]) -> str:
    retrieval = result["retrieval"]
    rank = result["rank"]
    variance = result["variance"]
    decision = result["decision"]
    lines = [
        "# World Model HEAD155: Surface-Local Smoke Failure Diagnosis",
        "",
        "Date: 2026-05-10",
        "",
        "## Iteration Type",
        "",
        "`post_experiment_analysis`",
        "",
        "## Objective Family",
        "",
        "`token_geometry_level_context_to_target_jepa` smoke diagnosis.",
        "",
        "## Hypothesis",
        "",
        "HEAD154's loss drop is insufficient if target-token retrieval remains weak,",
        "predicted variance shrinks, or the target latent itself is low-rank.",
        "",
        "## Findings",
        "",
        f"- Loss improved: `{result['loss_improved']}`.",
        f"- Loss delta: `{_fmt(result['loss_delta'])}`.",
        f"- Retrieval top10: `{_fmt(retrieval['top10'])}`.",
        f"- Random top10 for subset: `{_fmt(retrieval['random_top10'])}`.",
        f"- Top10/random: `{_fmt(retrieval['top10_over_random'])}`.",
        f"- Median retrieval rank: `{_fmt(retrieval['median_rank'])}`.",
        f"- Predicted effective rank fraction: `{_fmt(rank['predicted_rank_fraction'])}`.",
        f"- Target effective rank fraction: `{_fmt(rank['target_rank_fraction'])}`.",
        f"- Predicted/target rank ratio: `{_fmt(rank['predicted_to_target_rank_ratio'])}`.",
        f"- Predicted/target variance ratio: `{_fmt(variance['predicted_to_target_variance_ratio'])}`.",
        "",
        "## Decision",
        "",
        f"Promotion decision: `{decision['promotion_decision']}`.",
        "",
        f"- Predictor variance shrinkage: `{decision['looks_like_predictor_shrinkage']}`.",
        f"- Target latent also low-rank: `{decision['target_latent_also_low_rank']}`.",
        f"- Retrieval only weakly above random: `{decision['retrieval_only_weakly_above_random']}`.",
        "",
        decision["diagnosis"],
        "",
        f"Next: {decision['next_step']}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze surface-local token JEPA smoke failure")
    parser.add_argument("--input_json", default="results/world/surface_local_jepa_smoke_head154.json")
    parser.add_argument("--output_json", default="results/world/surface_local_smoke_failure_head155.json")
    parser.add_argument("--report_md", default="experiments/world/reports/world_model_head155_surface_local_smoke_failure.md")
    args = parser.parse_args()

    source = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    result = analyze_surface_local_smoke_failure(source)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    Path(args.report_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_md).write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result["decision"], indent=2))


if __name__ == "__main__":
    main()
