"""
Focused mechanistic review of 184c_best versus the stricter 183c_best anchor.

This review is intentionally narrow. It checks whether the new sparse-precision
operator actually became selective and materially changed the remaining frontier,
or whether training effectively turned the operator off while preserving the old
backbone behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text())


def pct_change(new: float, old: float) -> float | None:
    if abs(old) < 1e-12:
        return None
    return (new - old) / old


def epoch_view(rec: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "epoch",
        "stage",
        "val_post_activity_gate_mean",
        "val_prior_activity_gate_mean",
        "val_activity_gate_mean",
        "val_activity_budget_mean",
        "val_activity_node_top1_mean",
        "val_activity_node_entropy_mean",
        "val_kernel_top1_mean",
        "val_kernel_entropy_mean",
        "val_precision_delta_abs_mean",
        "joint_turb_calm_ratio",
        "joint_kurtosis_ratio",
        "joint_pathwise_jump_ks",
        "frontier_turb_late_worst_cov",
        "frontier_turb_late_best_cov",
    ]
    return {k: rec[k] for k in keys if k in rec}


def main() -> None:
    parser = argparse.ArgumentParser(description="Review 184c sparse-precision operator behavior")
    parser.add_argument(
        "--base_summary",
        default="results/block_ar/183c_best_v2_s3mrjspec_full_30d/summary.json",
    )
    parser.add_argument(
        "--base_mechanistic",
        default="results/validations/2026-04-05/analysis/183c_best_mechanistic/mechanistic_summary.json",
    )
    parser.add_argument(
        "--candidate_summary",
        default="results/block_ar/184c_best_v2_s3mrjspec_full_30d/summary.json",
    )
    parser.add_argument(
        "--candidate_history",
        default=(
            "models/backfill/"
            "sparse_precision_transport_pathwise_residual_law_mean_reverting_covariance_mixture_"
            "structured_joint_student_t_184c/training_history.json"
        ),
    )
    parser.add_argument(
        "--activity_baseline_history",
        default=(
            "models/backfill/"
            "latent_activity_process_transport_pathwise_residual_law_mean_reverting_covariance_mixture_"
            "structured_joint_student_t_184b/training_history.json"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="results/validations/2026-04-06/analysis/184c_best_mechanistic",
    )
    args = parser.parse_args()

    base_summary = load_json(Path(args.base_summary))
    base_mech = load_json(Path(args.base_mechanistic))
    cand_summary = load_json(Path(args.candidate_summary))
    cand_hist = load_json(Path(args.candidate_history))
    act_hist = load_json(Path(args.activity_baseline_history))

    assert isinstance(base_summary, dict)
    assert isinstance(base_mech, dict)
    assert isinstance(cand_summary, dict)
    assert isinstance(cand_hist, list)
    assert isinstance(act_hist, list)

    cand_best = min(cand_hist, key=lambda r: r["val_total_loss"])
    cand_first = cand_hist[0]
    cand_final = cand_hist[-1]
    act_best = min(act_hist, key=lambda r: r["val_total_loss"])

    base_ts = base_summary["time_series"]
    cand_ts = cand_summary["time_series"]
    base_es = base_ts["exceedance_spectrum"]
    cand_es = cand_ts["exceedance_spectrum"]
    base_rc = base_summary["regime_coverage"]
    cand_rc = cand_summary["regime_coverage"]
    base_cond = base_summary["conditionality"]
    cand_cond = cand_summary["conditionality"]
    base_dist = base_summary["distributional"]
    cand_dist = cand_summary["distributional"]

    control_review = {
        "183c_anchor_context": {
            "target_local_abs_mean_turb": base_mech["control_metric_review"]["target_local_abs_mean_turb"],
            "metric_target_corr_turb_target_state": base_mech["control_metric_review"][
                "metric_target_corr_turb_target_state"
            ],
            "mean_metric_local_hard_slices": base_mech["control_metric_review"]["mean_metric_local_hard_slices"],
            "mean_target_local_hard_slices": base_mech["control_metric_review"]["mean_target_local_hard_slices"],
            "mean_metric_local_overwide_slices": base_mech["control_metric_review"][
                "mean_metric_local_overwide_slices"
            ],
            "mean_target_local_overwide_slices": base_mech["control_metric_review"][
                "mean_target_local_overwide_slices"
            ],
        },
        "184b_activity_baseline_best": {
            "epoch": act_best["epoch"],
            "stage": act_best["stage"],
            "val_post_activity_gate_mean": act_best.get("val_post_activity_gate_mean"),
            "val_prior_activity_gate_mean": act_best.get("val_prior_activity_gate_mean"),
            "val_activity_budget_mean": act_best.get("val_activity_budget_mean"),
        },
        "184c_first_ctrl_epoch": epoch_view(cand_first),
        "184c_best_epoch": epoch_view(cand_best),
        "184c_final_epoch": epoch_view(cand_final),
        "gate_collapse_ratios": {
            "best_gate_vs_184b_best": (
                cand_best["val_activity_gate_mean"] / max(act_best["val_activity_gate_mean"], 1e-12)
            ),
            "final_gate_vs_184b_best": (
                cand_final["val_activity_gate_mean"] / max(act_best["val_activity_gate_mean"], 1e-12)
            ),
            "best_precision_delta_vs_first_epoch": (
                cand_best["val_precision_delta_abs_mean"] / max(cand_first["val_precision_delta_abs_mean"], 1e-12)
            ),
            "final_precision_delta_vs_first_epoch": (
                cand_final["val_precision_delta_abs_mean"] / max(cand_first["val_precision_delta_abs_mean"], 1e-12)
            ),
        },
        "operator_selectivity_interpretation": {
            "activity_gate_collapsed": cand_best["val_activity_gate_mean"] < 1e-3,
            "precision_delta_effectively_off": cand_best["val_precision_delta_abs_mean"] < 1e-5,
            "node_alloc_still_diffuse": cand_best["val_activity_node_top1_mean"] < 0.2,
            "kernel_still_uniformish": cand_best["val_kernel_top1_mean"] < 0.4,
        },
    }

    frontier_review = {
        "183c_best": {
            "s2_overall_90": base_summary["coverage"]["overall"]["0.9"],
            "s3_turb_calm_ratio": base_cond["turb_calm_ratio"],
            "s3_worst_cell_width_ratio": base_cond["worst_cell_width_ratio"],
            "s4_kurtosis_ratio": base_ts["kurtosis"]["kurtosis_ratio"],
            "s4_quiet_ratio": base_es["quiet_mass"]["ratio"],
            "s4_shoulder_ratio": base_es["shoulder_mass"]["ratio"],
            "s4_extreme_ratio": base_es["extreme_mass"]["ratio"],
            "s7_layer2_pass_count": base_rc["layer2_n_passing"],
            "s7_layer2_total": base_rc["layer2_n_total"],
            "s7_catastrophic_rate": base_rc["layer3_catastrophic_rate"],
            "s8_bad_window_rate": base_dist["window_floor"]["pct_bad"],
        },
        "184c_best": {
            "s2_overall_90": cand_summary["coverage"]["overall"]["0.9"],
            "s3_turb_calm_ratio": cand_cond["turb_calm_ratio"],
            "s3_worst_cell_width_ratio": cand_cond["worst_cell_width_ratio"],
            "s4_kurtosis_ratio": cand_ts["kurtosis"]["kurtosis_ratio"],
            "s4_quiet_ratio": cand_es["quiet_mass"]["ratio"],
            "s4_shoulder_ratio": cand_es["shoulder_mass"]["ratio"],
            "s4_extreme_ratio": cand_es["extreme_mass"]["ratio"],
            "s7_layer2_pass_count": cand_rc["layer2_n_passing"],
            "s7_layer2_total": cand_rc["layer2_n_total"],
            "s7_catastrophic_rate": cand_rc["layer3_catastrophic_rate"],
            "s8_bad_window_rate": cand_dist["window_floor"]["pct_bad"],
        },
        "delta_184c_minus_183c": {
            "s2_overall_90": cand_summary["coverage"]["overall"]["0.9"] - base_summary["coverage"]["overall"]["0.9"],
            "s3_turb_calm_ratio": cand_cond["turb_calm_ratio"] - base_cond["turb_calm_ratio"],
            "s3_worst_cell_width_ratio": cand_cond["worst_cell_width_ratio"] - base_cond["worst_cell_width_ratio"],
            "s4_kurtosis_ratio": cand_ts["kurtosis"]["kurtosis_ratio"] - base_ts["kurtosis"]["kurtosis_ratio"],
            "s4_quiet_ratio": cand_es["quiet_mass"]["ratio"] - base_es["quiet_mass"]["ratio"],
            "s4_shoulder_ratio": cand_es["shoulder_mass"]["ratio"] - base_es["shoulder_mass"]["ratio"],
            "s4_extreme_ratio": cand_es["extreme_mass"]["ratio"] - base_es["extreme_mass"]["ratio"],
            "s7_layer2_pass_count": cand_rc["layer2_n_passing"] - base_rc["layer2_n_passing"],
            "s7_catastrophic_rate": cand_rc["layer3_catastrophic_rate"] - base_rc["layer3_catastrophic_rate"],
            "s8_bad_window_rate": cand_dist["window_floor"]["pct_bad"] - base_dist["window_floor"]["pct_bad"],
        },
    }

    diagnosis = {
        "operator": (
            "184c did not meaningfully test the sparse precision idea on the benchmarked best checkpoint. "
            "The latent activity gate collapsed by roughly four orders of magnitude relative to 184b_best, "
            "the precision delta shrank to near zero, the node allocation remained diffuse, and the local "
            "kernel stayed effectively uniform."
        ),
        "frontier": (
            "Because the sparse precision path was mostly off, 184c_best ended up preserving the old 183c broad "
            "profile rather than improving the hard concentration frontier. S2/S8 stayed broadly intact, while "
            "S3/S4/S7 remained essentially unchanged."
        ),
        "why_s3_s7_remain": (
            "The unresolved issue is still under-concentration of residual mass on hard turbulent late-horizon "
            "cells. 183c already had the right directional signal; 184c did not sharpen that signal enough to "
            "change target-cell contrast in practice."
        ),
        "why_s4_remains": (
            "The tightened S4 failure is consistent with the operator being off: quiet mass remains too low and "
            "shoulder/extreme mass remains too high because the transport never became selectively sparse."
        ),
        "is_generalized_path_dead": (
            "No. The generalized mean/covariance/pathwise residual-law direction is still valid. What failed here "
            "is the specific 184c gating parameterization, not the broader geometry-aware latent-activity path."
        ),
    }

    next_step = {
        "recommendation": (
            "Do not build another sparse-precision patch with a free multiplicative gate. The next principled "
            "generalized move is a latent residual activity mixture of operators: quiet operator versus event "
            "operator, with posterior-guided training so the event branch cannot collapse to zero while still "
            "remaining a learned latent state rather than a hand-labeled regime."
        ),
        "candidate_branch": "184d_latent_activity_operator_mixture",
        "why": [
            "It preserves the generalized mean/covariance/pathwise residual-law backbone.",
            "It keeps geometry abstraction intact: grid now, graph/group later.",
            "It addresses the measured failure mode directly: the new operator was present in code but nearly absent in use.",
        ],
        "avoid": [
            "Do not do another blind sparse gating tweak.",
            "Do not broaden the benchmark again before fixing operator engagement.",
            "Do not interpret 184c as evidence that generalized fixes cannot solve S3/S4/S7.",
        ],
    }

    out = {
        "review_context": {
            "base_summary": str(Path(args.base_summary)),
            "base_mechanistic": str(Path(args.base_mechanistic)),
            "candidate_summary": str(Path(args.candidate_summary)),
            "candidate_history": str(Path(args.candidate_history)),
            "activity_baseline_history": str(Path(args.activity_baseline_history)),
        },
        "operator_activity_review": control_review,
        "frontier_comparison": frontier_review,
        "diagnosis": diagnosis,
        "next_step": next_step,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "mechanistic_summary.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
