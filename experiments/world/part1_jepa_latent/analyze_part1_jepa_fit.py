from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, ".")

from experiments.world.evaluation.masked_multiview_data import (  # noqa: E402
    MaskedMultiviewBatch,
    build_masked_multiview_batch,
)


DEFAULT_REFERENCE = Path("results/world/masked_multiview_barlow_head070.json")
DEFAULT_FAILURE = Path("results/world/part1_quality_gate_failure_analysis_head120.json")


def _load_json(root: Path, path: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else root / path
    return json.loads(resolved.read_text(encoding="utf-8"))


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def _mask_family_rates(batch: MaskedMultiviewBatch, *, side: str) -> dict[str, Any]:
    if side == "a":
        masks = batch.synthetic_mask_a
        labels = batch.mask_family_a
    elif side == "b":
        masks = batch.synthetic_mask_b
        labels = batch.mask_family_b
    else:
        raise ValueError(f"unknown side: {side}")
    rows = {}
    for label in sorted({str(x) for x in labels.tolist()}):
        idx = labels.astype(str) == label
        visible = float(np.mean(masks[idx]))
        rows[label] = {
            "n_windows": int(np.sum(idx)),
            "visible_rate": visible,
            "hidden_rate": 1.0 - visible,
        }
    return rows


def _mask_overlap_stats(batch: MaskedMultiviewBatch) -> dict[str, Any]:
    a = np.asarray(batch.synthetic_mask_a, dtype=bool)
    b = np.asarray(batch.synthetic_mask_b, dtype=bool)
    observed = np.asarray(batch.observed_mask, dtype=bool)
    valid = observed
    both_visible = a & b & valid
    both_hidden = (~a) & (~b) & valid
    disagreement = (a ^ b) & valid
    hidden_a = (~a) & valid
    hidden_b = (~b) & valid
    union_hidden = ((~a) | (~b)) & valid
    return {
        "observed_rate": float(np.mean(observed)),
        "view_a_visible_rate": float(np.sum(a & valid) / np.sum(valid)),
        "view_b_visible_rate": float(np.sum(b & valid) / np.sum(valid)),
        "view_a_hidden_rate": float(np.sum(hidden_a) / np.sum(valid)),
        "view_b_hidden_rate": float(np.sum(hidden_b) / np.sum(valid)),
        "both_visible_rate": float(np.sum(both_visible) / np.sum(valid)),
        "both_hidden_rate": float(np.sum(both_hidden) / np.sum(valid)),
        "view_disagreement_rate": float(np.sum(disagreement) / np.sum(valid)),
        "union_hidden_rate": float(np.sum(union_hidden) / np.sum(valid)),
        "mask_family_a": _mask_family_rates(batch, side="a"),
        "mask_family_b": _mask_family_rates(batch, side="b"),
    }


def _build_reference_batch_stats(
    root: Path, reference: dict[str, Any]
) -> dict[str, Any]:
    args = reference["args"]
    train = build_masked_multiview_batch(
        split="train",
        history_len=int(args["history_len"]),
        future_len=int(args["future_len"]),
        max_windows=int(args["max_train_windows"]),
        seed=int(args["seed"]),
        normalize=True,
    )
    val = build_masked_multiview_batch(
        split="val",
        history_len=int(args["history_len"]),
        future_len=int(args["future_len"]),
        max_windows=int(args["max_val_windows"]),
        seed=int(args["seed"]) + 1000,
        normalize=True,
    )
    return {
        "train": _mask_overlap_stats(train),
        "val": _mask_overlap_stats(val),
    }


def analyze_jepa_fit(root: Path) -> dict[str, Any]:
    reference = _load_json(root, DEFAULT_REFERENCE)
    failure = _load_json(root, DEFAULT_FAILURE)
    mask_stats = _build_reference_batch_stats(root, reference)
    val_visibility = reference["val_metrics"]["visibility"]["overall"]
    val_retrieval = reference["val_metrics"]["view_alignment"]["retrieval"]
    raw_val_retrieval = reference["raw_val_baseline"]["retrieval"]
    val_health = reference["val_metrics"]["view_alignment"]["view_a_health"]
    return {
        "analysis": "world_model_part1_jepa_fit_diagnosis",
        "date": "2026-05-10",
        "reference_run": "HEAD070",
        "source_reference": str(DEFAULT_REFERENCE),
        "source_failure_analysis": str(DEFAULT_FAILURE),
        "mask_stats": mask_stats,
        "representation_health": {
            "top10": val_retrieval["top10"],
            "raw_top10": raw_val_retrieval["top10"],
            "mrr": val_retrieval["mrr"],
            "raw_mrr": raw_val_retrieval["mrr"],
            "effective_rank": val_health["effective_rank"],
            "overall_visible_rate": val_visibility["view_a_visible_rate"],
            "surface_visible_rate": reference["val_metrics"]["visibility"][
                "by_geometry"
            ]["iv_surface"]["view_a_visible_rate"],
            "factor_level_visible_rate": reference["val_metrics"]["visibility"][
                "by_geometry"
            ]["factor_level"]["view_a_visible_rate"],
        },
        "downstream_failure_summary": failure["summary_counts"],
        "diagnosis": {
            "are_we_doing_something_wrong": (
                "The current branch is not wrong as a collapse-controlled masked "
                "multiview invariance smoke test, but it is weaker than canonical "
                "JEPA as a market-state learning recipe."
            ),
            "most_likely_reasons": [
                "The mask is too mild: validation views keep about 92-93% of observed entries, leaving most raw state visible in both views.",
                "The objective aligns two heavily overlapping corrupted full windows instead of predicting large missing target regions from a distributed context.",
                "Raw last-surface baselines are strong for persistence-like targets because they preserve exact level information that an abstract embedding may compress away.",
                "The current model is smoke scale: 384 training windows, 8 epochs, latent_dim 64, hidden_dim 128.",
                "The probes mix targets that reward exact low-level state copying with targets that reward path-shape abstraction; Barlow only wins the latter family today.",
            ],
            "not_supported_by_evidence": [
                "Representation collapse is not the main failure class.",
                "Mask-artifact leakage is not the main failure class under the audited default masks.",
                "There is not enough evidence to conclude the Barlow objective is intrinsically wrong.",
            ],
            "what_literature_suggests": [
                "Make the missing-information task semantically hard enough that local copying is not sufficient.",
                "Evaluate frozen representations on downstream tasks that actually require the intended abstraction.",
                "Expect scale and data distribution to matter; strong JEPA results use much larger data and backbones.",
                "For time series, multi-resolution or regime-aware objectives may be needed when precursor signals live across different temporal scales.",
            ],
            "next_experiment_shape": [
                "Do not add many knobs. Add one controlled hard-mask diagnostic preset before changing architecture.",
                "Compare current masks against a hard structured preset with much lower overlap, e.g. larger surface blocks, longer time blocks, whole geometry groups, and cross-family stress masks.",
                "Keep the same encoder/loss for the first hard-mask diagnostic so the failure class is mask difficulty rather than architecture churn.",
                "Score not only retrieval but also frozen present-state probes, factor-panel probes, and incremental value over raw/PCA/persistence baselines.",
            ],
        },
    }


def _source_list() -> list[str]:
    return [
        "I-JEPA: https://arxiv.org/abs/2301.08243",
        "V-JEPA: https://arxiv.org/abs/2404.08471",
        "Barlow Twins: https://arxiv.org/abs/2103.03230",
        "MTS-JEPA: https://arxiv.org/abs/2602.04643",
        "LaT-PFN: https://arxiv.org/abs/2405.10093",
        "TS-JEPA: https://arxiv.org/abs/2406.04853",
    ]


def render_markdown(result: dict[str, Any], *, title: str) -> str:
    val_mask = result["mask_stats"]["val"]
    train_mask = result["mask_stats"]["train"]
    health = result["representation_health"]
    counts = result["downstream_failure_summary"]
    diagnosis = result["diagnosis"]
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
        "`masked_multiview_invariance` failure diagnosis against JEPA literature.",
        "",
        "## Hypothesis",
        "",
        "HEAD070 is not beating simple market-state baselines because the current",
        "masked-view task is too easy and too close to raw-state preservation, not",
        "because the representation has collapsed.",
        "",
        "## Falsifier",
        "",
        "This diagnosis would be wrong if the default masks were already aggressive,",
        "the two views had low overlap, raw baselines were weak, or JEPA literature",
        "showed that mild two-view alignment should reliably beat raw-state baselines.",
        "",
        "## Literature Anchors",
        "",
        "- I-JEPA frames masking as a core design choice: target blocks need semantic",
        "  scale and the context must be informative/distributed.",
        "- V-JEPA uses masked feature prediction, a predictor, stop-gradient/EMA",
        "  target encoder, and large continuous spatio-temporal masks; it explicitly",
        "  compares downstream frozen representations and reports data-scale effects.",
        "- Barlow Twins supports direct two-view redundancy reduction, but it is an",
        "  invariance method over distorted views and benefits from high-dimensional",
        "  outputs; it does not by itself guarantee superiority over raw features on",
        "  low-level persistence-like tasks.",
        "- Recent time-series JEPA work points to multi-resolution dynamics, regime",
        "  structure, and task-aligned latent prediction/control as important when",
        "  precursor signals live across multiple temporal scales.",
        "",
        "Sources:",
        "",
    ]
    lines.extend(f"- {source}" for source in _source_list())
    lines.extend(
        [
            "",
            "## Local Mask Difficulty",
            "",
            "| split | view A hidden | view B hidden | both visible | both hidden | view disagreement | union hidden |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            "| train | {train_a_hidden} | {train_b_hidden} | {train_both_visible} | {train_both_hidden} | {train_disagree} | {train_union_hidden} |".format(
                train_a_hidden=_pct(train_mask["view_a_hidden_rate"]),
                train_b_hidden=_pct(train_mask["view_b_hidden_rate"]),
                train_both_visible=_pct(train_mask["both_visible_rate"]),
                train_both_hidden=_pct(train_mask["both_hidden_rate"]),
                train_disagree=_pct(train_mask["view_disagreement_rate"]),
                train_union_hidden=_pct(train_mask["union_hidden_rate"]),
            ),
            "| val | {val_a_hidden} | {val_b_hidden} | {val_both_visible} | {val_both_hidden} | {val_disagree} | {val_union_hidden} |".format(
                val_a_hidden=_pct(val_mask["view_a_hidden_rate"]),
                val_b_hidden=_pct(val_mask["view_b_hidden_rate"]),
                val_both_visible=_pct(val_mask["both_visible_rate"]),
                val_both_hidden=_pct(val_mask["both_hidden_rate"]),
                val_disagree=_pct(val_mask["view_disagreement_rate"]),
                val_union_hidden=_pct(val_mask["union_hidden_rate"]),
            ),
            "",
            "The default validation views keep most observed entries visible in both",
            "views. That makes same-state alignment a useful sanity check, but not a",
            "hard semantic missing-information problem.",
            "",
            "## Local Representation And Probe Evidence",
            "",
            "- Same-state retrieval is healthy: top10 `{top10}` versus raw `{raw_top10}`, MRR `{mrr}` versus raw `{raw_mrr}`.".format(
                top10=_fmt(health["top10"]),
                raw_top10=_fmt(health["raw_top10"]),
                mrr=_fmt(health["mrr"]),
                raw_mrr=_fmt(health["raw_mrr"]),
            ),
            "- Effective rank is non-collapsed: `{rank}`.".format(
                rank=_fmt(health["effective_rank"])
            ),
            "- Validation view A visible rates: overall `{overall}`, IV surface `{surface}`, factor levels `{factors}`.".format(
                overall=_pct(health["overall_visible_rate"]),
                surface=_pct(health["surface_visible_rate"]),
                factors=_pct(health["factor_level_visible_rate"]),
            ),
            (
                "- Barlow beats the mean baseline on `{mean}/5` targets, is best "
                "standalone on `{best}/5`, and improves raw last-surface features "
                "on `{plus}/5` targets."
            ).format(
                mean=counts["barlow_beats_mean_baseline_targets"],
                best=counts["barlow_is_best_feature_targets"],
                plus=counts["raw_plus_barlow_improves_raw_last_targets"],
            ),
            "",
            "## Diagnosis",
            "",
            f"- {diagnosis['are_we_doing_something_wrong']}",
            "",
            "Most likely reasons:",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in diagnosis["most_likely_reasons"])
    lines.extend(
        [
            "",
            "Not supported by current evidence:",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in diagnosis["not_supported_by_evidence"])
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "We are not ready to change the objective yet. First run a controlled",
            "hard-mask diagnostic and stronger frozen probes. If the same encoder/loss",
            "improves under harder, lower-overlap masks, then the issue was mask",
            "difficulty. If it does not, the next likely bottleneck is objective family",
            "or capacity/scale.",
            "",
            "Next experiment shape:",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in diagnosis["next_experiment_shape"])
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze Part 1 JEPA failure against local mask difficulty and literature"
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/world/part1_jepa_fit_diagnosis_head121.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "experiments/world/reports/world_model_head121_jepa_fit_diagnosis.md"
        ),
    )
    parser.add_argument(
        "--report-title",
        default="World Model HEAD121: JEPA Fit Diagnosis",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = analyze_jepa_fit(args.root)
    output_json = (
        args.output_json
        if args.output_json.is_absolute()
        else args.root / args.output_json
    )
    output_md = (
        args.output_md if args.output_md.is_absolute() else args.root / args.output_md
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(
        render_markdown(result, title=args.report_title), encoding="utf-8"
    )
    val = result["mask_stats"]["val"]
    print(
        json.dumps(
            {
                "val_view_a_hidden_rate": val["view_a_hidden_rate"],
                "val_view_b_hidden_rate": val["view_b_hidden_rate"],
                "val_both_visible_rate": val["both_visible_rate"],
                "diagnosis": result["diagnosis"]["are_we_doing_something_wrong"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
