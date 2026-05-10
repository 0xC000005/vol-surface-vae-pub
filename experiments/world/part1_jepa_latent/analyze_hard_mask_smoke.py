from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_REFERENCE = Path("results/world/masked_multiview_barlow_head070.json")
HARD_REFERENCE = Path("results/world/masked_multiview_barlow_hardmask_head123.json")
DEFAULT_DOWNSTREAM = Path(
    "results/world/masked_multiview_downstream_probe_head085.json"
)
HARD_DOWNSTREAM = Path(
    "results/world/masked_multiview_downstream_probe_hardmask_head123.json"
)

REGRESSION_TARGETS = (
    "future_mean_delta",
    "future_range",
    "future_terminal_delta",
    "future_max_abs_step",
    "future_drawdown",
)


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


def _representation_row(run: dict[str, Any]) -> dict[str, float]:
    view = run["val_metrics"]["view_alignment"]
    retrieval = view["retrieval"]
    raw = run["raw_val_baseline"]["retrieval"]
    health = view["view_a_health"]
    visibility = run["val_metrics"]["visibility"]["overall"]
    return {
        "top1": float(retrieval["top1"]),
        "top5": float(retrieval["top5"]),
        "top10": float(retrieval["top10"]),
        "mrr": float(retrieval["mrr"]),
        "median_rank": float(retrieval["median_rank"]),
        "raw_top10": float(raw["top10"]),
        "raw_mrr": float(raw["mrr"]),
        "effective_rank": float(health["effective_rank"]),
        "variance_min": float(health["variance_min"]),
        "offdiag_abs_mean": float(health["offdiag_abs_mean"]),
        "view_a_hidden_rate": 1.0 - float(visibility["view_a_visible_rate"]),
        "view_b_hidden_rate": 1.0 - float(visibility["view_b_visible_rate"]),
    }


def _best_raw_mse(
    regression: dict[str, Any],
    target: str,
) -> tuple[str, float]:
    raw_names = ("raw_surface_last", "raw_surface_flat")
    best = min(
        raw_names,
        key=lambda name: float(regression[name]["targets"][target]["mse"]),
    )
    return best, float(regression[best]["targets"][target]["mse"])


def _downstream_rows(downstream: dict[str, Any]) -> list[dict[str, Any]]:
    regression = downstream["regression_probe_metrics"]
    rows = []
    for target in REGRESSION_TARGETS:
        best_raw_name, best_raw_mse = _best_raw_mse(regression, target)
        barlow_mse = float(regression["barlow_clean_last"]["targets"][target]["mse"])
        raw_last_mse = float(regression["raw_surface_last"]["targets"][target]["mse"])
        raw_plus_barlow_mse = float(
            regression["raw_surface_last_plus_barlow_clean_last"]["targets"][target][
                "mse"
            ]
        )
        rows.append(
            {
                "target": target,
                "barlow_mse": barlow_mse,
                "best_raw_feature": best_raw_name,
                "best_raw_mse": best_raw_mse,
                "barlow_beats_best_raw": barlow_mse < best_raw_mse,
                "raw_last_mse": raw_last_mse,
                "raw_last_plus_barlow_mse": raw_plus_barlow_mse,
                "barlow_adds_to_raw_last": raw_plus_barlow_mse < raw_last_mse,
            }
        )
    return rows


def _regime_row(downstream: dict[str, Any]) -> dict[str, float]:
    cls = downstream["classification_probe_metrics"]
    barlow = cls["barlow_clean_last"]["regime_label"]
    raw = cls["raw_surface_last"]["regime_label"]
    combined = cls["raw_surface_last_plus_barlow_clean_last"]["regime_label"]
    return {
        "barlow_accuracy": float(barlow["accuracy"]),
        "raw_surface_last_accuracy": float(raw["accuracy"]),
        "raw_plus_barlow_accuracy": float(combined["accuracy"]),
        "majority_accuracy": float(barlow["majority_accuracy"]),
    }


def analyze_hard_mask_smoke(root: Path) -> dict[str, Any]:
    default_run = _load_json(root, DEFAULT_REFERENCE)
    hard_run = _load_json(root, HARD_REFERENCE)
    default_downstream = _load_json(root, DEFAULT_DOWNSTREAM)
    hard_downstream = _load_json(root, HARD_DOWNSTREAM)
    default_downstream_rows = _downstream_rows(default_downstream)
    hard_downstream_rows = _downstream_rows(hard_downstream)
    return {
        "analysis": "world_model_part1_hard_mask_smoke",
        "date": "2026-05-10",
        "status": "diagnostic_negative_for_mask_aggression_alone",
        "objective_family": "masked_multiview_invariance",
        "literature_status": "supported_adjacent_direct_barlow_twins_for_same_state_masked_multiview",
        "artifacts": {
            "default_reference": str(DEFAULT_REFERENCE),
            "hard_reference": str(HARD_REFERENCE),
            "default_downstream": str(DEFAULT_DOWNSTREAM),
            "hard_downstream": str(HARD_DOWNSTREAM),
        },
        "representation": {
            "default_head070": _representation_row(default_run),
            "hard_head123": _representation_row(hard_run),
        },
        "downstream": {
            "default_head085_rows": default_downstream_rows,
            "hard_head123_rows": hard_downstream_rows,
            "default_barlow_best_raw_wins": int(
                sum(row["barlow_beats_best_raw"] for row in default_downstream_rows)
            ),
            "hard_barlow_best_raw_wins": int(
                sum(row["barlow_beats_best_raw"] for row in hard_downstream_rows)
            ),
            "default_raw_last_plus_barlow_wins": int(
                sum(row["barlow_adds_to_raw_last"] for row in default_downstream_rows)
            ),
            "hard_raw_last_plus_barlow_wins": int(
                sum(row["barlow_adds_to_raw_last"] for row in hard_downstream_rows)
            ),
        },
        "regime": {
            "default_head085": _regime_row(default_downstream),
            "hard_head123": _regime_row(hard_downstream),
        },
        "decision": {
            "mask_aggression_alone_fixed_baseline_gap": False,
            "promote_hard_mask_checkpoint": False,
            "next_step": (
                "Do not keep making masks harder. Audit whether the frozen "
                "embedding captures present market state and factor-panel "
                "geometry before changing architecture or objective."
            ),
        },
    }


def _row_by_target(rows: list[dict[str, Any]], target: str) -> dict[str, Any]:
    return next(row for row in rows if row["target"] == target)


def render_markdown(result: dict[str, Any], *, title: str) -> str:
    rep_default = result["representation"]["default_head070"]
    rep_hard = result["representation"]["hard_head123"]
    default_rows = result["downstream"]["default_head085_rows"]
    hard_rows = result["downstream"]["hard_head123_rows"]
    regime_default = result["regime"]["default_head085"]
    regime_hard = result["regime"]["hard_head123"]
    lines = [
        f"# {title}",
        "",
        "Date: 2026-05-10",
        "",
        "## Iteration Type",
        "",
        "`experiment`",
        "",
        "## Objective Family",
        "",
        "`masked_multiview_invariance` hard-mask training smoke.",
        "",
        "## Literature Status",
        "",
        "`supported_adjacent_direct_barlow_twins_for_same_state_masked_multiview`.",
        "",
        "## Hypothesis",
        "",
        "If the Part 1 gap against simple market-state baselines is mainly caused",
        "by under-aggressive masks, then the HEAD122 hard-mask preset should",
        "improve downstream baseline superiority while preserving healthy",
        "same-state representation metrics.",
        "",
        "## Falsifier",
        "",
        "The hypothesis is falsified if the hard-mask checkpoint learns alignment",
        "but loses downstream probe quality or still fails against raw/simple",
        "market-state features.",
        "",
        "## Representation Metrics",
        "",
        "| run | hidden A | hidden B | top10 | raw top10 | mrr | raw mrr | median rank | eff rank | variance min | offdiag |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| HEAD070 default | {ha} | {hb} | {top10} | {raw_top10} | {mrr} | {raw_mrr} | {median} | {rank} | {var_min} | {offdiag} |".format(
            ha=_pct(rep_default["view_a_hidden_rate"]),
            hb=_pct(rep_default["view_b_hidden_rate"]),
            top10=_fmt(rep_default["top10"]),
            raw_top10=_fmt(rep_default["raw_top10"]),
            mrr=_fmt(rep_default["mrr"]),
            raw_mrr=_fmt(rep_default["raw_mrr"]),
            median=_fmt(rep_default["median_rank"]),
            rank=_fmt(rep_default["effective_rank"]),
            var_min=_fmt(rep_default["variance_min"]),
            offdiag=_fmt(rep_default["offdiag_abs_mean"]),
        ),
        "| HEAD123 hard | {ha} | {hb} | {top10} | {raw_top10} | {mrr} | {raw_mrr} | {median} | {rank} | {var_min} | {offdiag} |".format(
            ha=_pct(rep_hard["view_a_hidden_rate"]),
            hb=_pct(rep_hard["view_b_hidden_rate"]),
            top10=_fmt(rep_hard["top10"]),
            raw_top10=_fmt(rep_hard["raw_top10"]),
            mrr=_fmt(rep_hard["mrr"]),
            raw_mrr=_fmt(rep_hard["raw_mrr"]),
            median=_fmt(rep_hard["median_rank"]),
            rank=_fmt(rep_hard["effective_rank"]),
            var_min=_fmt(rep_hard["variance_min"]),
            offdiag=_fmt(rep_hard["offdiag_abs_mean"]),
        ),
        "",
        "## Downstream Baseline Superiority",
        "",
        "| target | default Barlow MSE | default best raw MSE | default win | hard Barlow MSE | hard best raw MSE | hard win | hard raw-last+Barlow improves raw-last |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for target in REGRESSION_TARGETS:
        drow = _row_by_target(default_rows, target)
        hrow = _row_by_target(hard_rows, target)
        lines.append(
            "| {target} | {db} | {dr} | {dw} | {hb} | {hr} | {hw} | {adds} |".format(
                target=target,
                db=_fmt(drow["barlow_mse"]),
                dr=_fmt(drow["best_raw_mse"]),
                dw=str(drow["barlow_beats_best_raw"]),
                hb=_fmt(hrow["barlow_mse"]),
                hr=_fmt(hrow["best_raw_mse"]),
                hw=str(hrow["barlow_beats_best_raw"]),
                adds=str(hrow["barlow_adds_to_raw_last"]),
            )
        )
    lines.extend(
        [
            "",
            "## Regime Probe",
            "",
            "| run | Barlow accuracy | raw last accuracy | raw+Barlow accuracy | majority |",
            "| --- | ---: | ---: | ---: | ---: |",
            "| HEAD085 default | {barlow} | {raw} | {combo} | {majority} |".format(
                barlow=_fmt(regime_default["barlow_accuracy"]),
                raw=_fmt(regime_default["raw_surface_last_accuracy"]),
                combo=_fmt(regime_default["raw_plus_barlow_accuracy"]),
                majority=_fmt(regime_default["majority_accuracy"]),
            ),
            "| HEAD123 hard | {barlow} | {raw} | {combo} | {majority} |".format(
                barlow=_fmt(regime_hard["barlow_accuracy"]),
                raw=_fmt(regime_hard["raw_surface_last_accuracy"]),
                combo=_fmt(regime_hard["raw_plus_barlow_accuracy"]),
                majority=_fmt(regime_hard["majority_accuracy"]),
            ),
            "",
            "## Interpretation",
            "",
            "Harder masks are useful as a falsifier, but this run says mask",
            "aggression alone is not the missing ingredient. The hard checkpoint",
            "still beats the raw masked-view baseline on retrieval, so it is not",
            "dead, but it loses top10 retrieval, effective rank, and standalone",
            "future-probe quality versus the default checkpoint. Baseline",
            "superiority gets worse: default Barlow beats the best raw surface",
            "baseline on `2/5` IV future targets, while hard-mask Barlow beats it",
            "on `0/5`.",
            "",
            "The regime probe improves under the hard mask, but it remains below",
            "both raw last-surface features and the majority baseline. That is not",
            "a promotion signal.",
            "",
            "## Decision",
            "",
            "- Mask aggression alone fixed the baseline gap: `False`.",
            "- Promote hard-mask checkpoint: `False`.",
            "- Next: audit present-state and factor-panel information in the frozen",
            "  embedding before changing architecture, loss, or adding more mask",
            "  knobs.",
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze the HEAD123 hard-mask Barlow smoke"
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/world/hard_mask_smoke_analysis_head123.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "experiments/world/reports/world_model_head123_hard_mask_smoke.md"
        ),
    )
    parser.add_argument(
        "--report-title",
        default="World Model HEAD123: Hard Mask Smoke Analysis",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = analyze_hard_mask_smoke(args.root)
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
    print(json.dumps(result["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
