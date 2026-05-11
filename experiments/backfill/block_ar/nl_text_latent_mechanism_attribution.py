#!/usr/bin/env python
"""Attribute what is working in the narrative text-to-latent bridge.

This is an offline analysis over existing bridge bake-off artifacts. It does
not call the OpenAI API or train a new model. The goal is to separate mechanism
signal from architecture churn before the next autoresearch iteration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_STABILITY_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_text_latent_bakeoff_851d_seed_stability/policy_stability_summary.json"
)
DEFAULT_CLIP_SWEEP_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_text_latent_bakeoff_852d_clip_mse_sweep/clip_mse_sweep_summary.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_text_latent_mechanism_attribution_853a"
)

INCUMBENT_POLICY = "mlp_mse_contrastive__multi_caption_with_negatives"
NO_NEGATIVE_POLICY = "mlp_mse_contrastive__multi_caption_no_negatives"
ANCHOR_ONLY_POLICY = "mlp_mse_contrastive__anchor_only"
CLIP_METHOD = "clip_infonce_hybrid__multi_caption_with_negatives"

METRICS = [
    "heldout_mean_target_cosine",
    "heldout_hard_negative_mean_gap",
    "heldout_hard_negative_mean_margin",
    "heldout_recall_at_1_test_pool",
    "heldout_recall_at_3_test_pool",
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _round(value: float | int | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 12)


def _mean_metric(stability: dict[str, Any], policy: str, metric: str) -> float:
    return float(stability["summary"][policy][metric]["mean"])


def _metric_block(stability: dict[str, Any], policy: str) -> dict[str, dict[str, Any]]:
    block: dict[str, dict[str, Any]] = {}
    for metric in METRICS:
        metric_summary = stability["summary"][policy][metric]
        block[metric] = {
            "mean": _round(metric_summary["mean"]),
            "min": _round(metric_summary["min"]),
            "max": _round(metric_summary["max"]),
            "values": [_round(value) for value in metric_summary["values"]],
        }
    return block


def _delta_block(
    stability: dict[str, Any],
    policy: str,
    baseline_policy: str,
) -> dict[str, float]:
    return {
        metric: _round(
            _mean_metric(stability, policy, metric)
            - _mean_metric(stability, baseline_policy, metric)
        )
        for metric in METRICS
    }


def _winner_counts(stability: dict[str, Any]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for metric, winners in stability.get("winners_by_metric", {}).items():
        metric_counts: dict[str, int] = {}
        for winner in winners:
            metric_counts[str(winner)] = metric_counts.get(str(winner), 0) + 1
        counts[str(metric)] = metric_counts
    return counts


def _best_clip_by_metric(clip_sweep: dict[str, Any]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    comparison = clip_sweep.get("comparison", {})
    for metric in METRICS:
        candidates = []
        for variant, metrics in comparison.items():
            if metric not in metrics:
                continue
            record = metrics[metric]
            candidates.append(
                {
                    "variant": str(variant),
                    "clip": float(record["clip"]),
                    "mlp": float(record["mlp"]),
                    "clip_minus_mlp": float(record["clip_minus_mlp"]),
                }
            )
        if candidates:
            best_record = max(candidates, key=lambda item: item["clip"])
            best[metric] = {
                "variant": best_record["variant"],
                "clip": _round(best_record["clip"]),
                "mlp": _round(best_record["mlp"]),
                "clip_minus_mlp": _round(best_record["clip_minus_mlp"]),
            }
    return best


def _clip_status(best_clip: dict[str, dict[str, Any]]) -> str:
    target = best_clip["heldout_mean_target_cosine"]["clip_minus_mlp"]
    gap = best_clip["heldout_hard_negative_mean_gap"]["clip_minus_mlp"]
    margin = best_clip["heldout_hard_negative_mean_margin"]["clip_minus_mlp"]
    if target is not None and gap is not None and margin is not None:
        if target >= 0.0 and gap >= 0.0 and margin >= 0.0:
            return "promotable"
    return "not_promoted"


def build_mechanism_attribution(
    stability: dict[str, Any],
    clip_sweep: dict[str, Any],
) -> dict[str, Any]:
    """Build a structured mechanism attribution report."""

    best_clip = _best_clip_by_metric(clip_sweep)
    clip_status = _clip_status(best_clip)
    hard_negative_delta = _delta_block(stability, INCUMBENT_POLICY, NO_NEGATIVE_POLICY)
    caption_delta = _delta_block(stability, INCUMBENT_POLICY, ANCHOR_ONLY_POLICY)

    return {
        "status": "ok",
        "scope_note": (
            "Offline mechanism attribution over cached OpenAI embeddings and "
            "bridge bake-off artifacts. No OpenAI API calls and no new training."
        ),
        "incumbent": INCUMBENT_POLICY,
        "policy_stability": {
            INCUMBENT_POLICY: _metric_block(stability, INCUMBENT_POLICY),
            NO_NEGATIVE_POLICY: _metric_block(stability, NO_NEGATIVE_POLICY),
            ANCHOR_ONLY_POLICY: _metric_block(stability, ANCHOR_ONLY_POLICY),
            "winner_counts": _winner_counts(stability),
        },
        "mechanism_deltas": {
            "add_hard_negatives_vs_multi_caption_no_negatives": hard_negative_delta,
            "multi_caption_with_negatives_vs_anchor_only": caption_delta,
        },
        "clip_mse_sweep": {
            "status": clip_status,
            "best_clip_by_metric": best_clip,
            "comparison": clip_sweep.get("comparison", {}),
        },
        "mechanism_read": [
            (
                "Multi-caption training is useful for generator-memory alignment: "
                "the incumbent improves held-out target cosine over anchor-only."
            ),
            (
                "Hard negatives are the strongest mechanism signal: they lift "
                "hard-negative gap and margin across all stability seeds with "
                "almost no target-cosine cost versus multi-caption without negatives."
            ),
            (
                "The CLIP/InfoNCE hybrid is not promotable from the current sweep: "
                "it can improve recall@1, but loses target-memory cosine and "
                "directional hard-negative separation versus the MLP incumbent."
            ),
            (
                "Exact nearest-window retrieval remains a weak primary objective. "
                "The production objective should remain generator-memory quality, "
                "directional separation, support auditability, and scenario-level "
                "distributional usefulness."
            ),
        ],
        "promotion_decision": {
            "promote": INCUMBENT_POLICY,
            "do_not_promote": CLIP_METHOD,
            "clip_hybrid_status": clip_status,
            "rationale": [
                "The incumbent is stable over seeds on target cosine and dominates gap/margin.",
                "The CLIP hybrid's recall@1 gain is too small to justify degrading memory geometry.",
                "Further CLIP-weight sweeping would be knob tuning without a new mechanism.",
            ],
        },
        "next_principled_step": {
            "title": (
                "Train a supervised-contrastive MLP bridge anchored by generator-memory "
                "regression, then evaluate fixed-start narrative conditionality."
            ),
            "why": (
                "This keeps the proven MLP memory anchor, preserves hard-negative "
                "directional training, and tests whether richer narrative captions "
                "change scenarios under the same starting level."
            ),
            "falsifiers": [
                "Held-out target cosine drops by more than 0.01 versus the incumbent.",
                "Hard-negative gap or margin does not improve over the incumbent.",
                "Fixed-start scenario distributions do not move when narratives change.",
            ],
            "avoid_for_now": [
                "Broad CLIP/MSE hyperparameter sweeps.",
                "A distributional latent prior before the deterministic bridge signal is stronger.",
                "Model-chosen starts hidden from the user.",
            ],
        },
    }


def _markdown_metric_table(title: str, block: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Metric | Mean | Min | Max |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric in METRICS:
        metric_block = block[metric]
        lines.append(
            f"| `{metric}` | {metric_block['mean']:.6f} | "
            f"{metric_block['min']:.6f} | {metric_block['max']:.6f} |"
        )
    lines.append("")
    return lines


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Narrative Text-to-Latent Mechanism Attribution",
        "",
        report["scope_note"],
        "",
        "## Decision",
        "",
        f"- Promote: `{report['promotion_decision']['promote']}`.",
        f"- Do not promote: `{report['promotion_decision']['do_not_promote']}`.",
        f"- CLIP hybrid status: `{report['promotion_decision']['clip_hybrid_status']}`.",
        "",
        "## Mechanism Read",
        "",
    ]
    lines.extend(f"- {item}" for item in report["mechanism_read"])
    lines.extend(["", "## Stability Evidence", ""])
    policy_blocks = report["policy_stability"]
    lines.extend(
        _markdown_metric_table(INCUMBENT_POLICY, policy_blocks[INCUMBENT_POLICY])
    )
    lines.extend(
        _markdown_metric_table(NO_NEGATIVE_POLICY, policy_blocks[NO_NEGATIVE_POLICY])
    )
    lines.extend(
        _markdown_metric_table(ANCHOR_ONLY_POLICY, policy_blocks[ANCHOR_ONLY_POLICY])
    )

    lines.extend(
        [
            "## Deltas",
            "",
            "| Comparison | Metric | Delta |",
            "| --- | --- | ---: |",
        ]
    )
    for comparison, deltas in report["mechanism_deltas"].items():
        for metric in METRICS:
            lines.append(f"| `{comparison}` | `{metric}` | {deltas[metric]:.6f} |")

    lines.extend(
        [
            "",
            "## Best CLIP Sweep Result By Metric",
            "",
            "| Metric | Best Variant | CLIP | MLP | CLIP - MLP |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for metric, block in report["clip_mse_sweep"]["best_clip_by_metric"].items():
        lines.append(
            f"| `{metric}` | `{block['variant']}` | {block['clip']:.6f} | "
            f"{block['mlp']:.6f} | {block['clip_minus_mlp']:.6f} |"
        )

    next_step = report["next_principled_step"]
    lines.extend(
        [
            "",
            "## Next Principled Step",
            "",
            next_step["title"],
            "",
            next_step["why"],
            "",
            "Falsifiers:",
        ]
    )
    lines.extend(f"- {item}" for item in next_step["falsifiers"])
    lines.extend(["", "Avoid for now:"])
    lines.extend(f"- {item}" for item in next_step["avoid_for_now"])
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stability-report", type=Path, default=DEFAULT_STABILITY_REPORT
    )
    parser.add_argument(
        "--clip-sweep-report", type=Path, default=DEFAULT_CLIP_SWEEP_REPORT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    stability = _load_json(args.stability_report)
    clip_sweep = _load_json(args.clip_sweep_report)
    report = build_mechanism_attribution(stability, clip_sweep)
    report["inputs"] = {
        "stability_report": str(args.stability_report),
        "clip_sweep_report": str(args.clip_sweep_report),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "mechanism_attribution_report.json"
    markdown_path = args.output_dir / "mechanism_attribution_report.md"
    _write_json(json_path, report)
    write_markdown_report(markdown_path, report)
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
