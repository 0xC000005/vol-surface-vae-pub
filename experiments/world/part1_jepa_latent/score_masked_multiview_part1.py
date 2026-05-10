from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACTS = (
    Path("results/world/masked_multiview_jepa_head066.json"),
    Path("results/world/masked_multiview_barlow_head068.json"),
    Path("results/world/masked_multiview_barlow_head070.json"),
    Path("results/world/masked_multiview_geometry_barlow_head076.json"),
)
DEFAULT_PROBE_ARTIFACT = Path("results/world/masked_multiview_barlow_probe_head074.json")
REFERENCE_RUN = "HEAD070"


def _get_nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _singular_share(values: Any, *, top_k: int) -> float | None:
    if not isinstance(values, list) or not values:
        return None
    singular = [float(v) for v in values]
    total = sum(singular)
    if total <= 0.0:
        return None
    return sum(singular[:top_k]) / total


def _infer_run_name(path: str | Path) -> str:
    match = re.search(r"head(\d+)", str(path), flags=re.IGNORECASE)
    if match:
        return f"HEAD{int(match.group(1)):03d}"
    return Path(path).stem


def _select_metric_block(data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    val_metrics = data.get("val_metrics", {})
    if not isinstance(val_metrics, dict):
        raise ValueError("artifact val_metrics must be an object")
    for key in ("view_alignment", "predicted_target", "context_target"):
        block = val_metrics.get(key)
        if isinstance(block, dict):
            return key, block
    raise ValueError("artifact lacks a supported val metric block")


def _infer_objective_family(metric_block: str) -> str:
    if metric_block == "view_alignment":
        return "masked_multiview_invariance"
    if metric_block in {"predicted_target", "context_target"}:
        return "context_to_target_jepa"
    return "unknown"


def extract_masked_multiview_scorecard_row(
    data: dict[str, Any],
    *,
    artifact_path: str | Path,
) -> dict[str, Any]:
    metric_key, metrics = _select_metric_block(data)
    retrieval = metrics.get("retrieval", {})
    alignment = metrics.get("alignment", {})
    barlow = metrics.get("barlow", {})
    health_a = metrics.get("view_a_health", {})
    health_b = metrics.get("view_b_health", {})
    visibility = _get_nested(data, "val_metrics", "visibility", "overall", default={})
    raw_retrieval = _get_nested(data, "raw_val_baseline", "retrieval", default={})
    return {
        "run": _infer_run_name(artifact_path),
        "artifact": str(artifact_path),
        "metric_block": metric_key,
        "objective_family": _infer_objective_family(metric_key),
        "literature_status": data.get("literature_status", ""),
        "loss_scaling": data.get("loss_scaling", ""),
        "top1": _float_or_none(retrieval.get("top1")),
        "top5": _float_or_none(retrieval.get("top5")),
        "top10": _float_or_none(retrieval.get("top10")),
        "mrr": _float_or_none(retrieval.get("mrr")),
        "median_rank": _float_or_none(retrieval.get("median_rank")),
        "cosine_mean": _float_or_none(alignment.get("cosine_mean")),
        "mse": _float_or_none(alignment.get("mse")),
        "diag_mean": _float_or_none(barlow.get("diag_mean")),
        "offdiag_abs_mean": _float_or_none(barlow.get("offdiag_abs_mean")),
        "effective_rank_a": _float_or_none(health_a.get("effective_rank")),
        "effective_rank_b": _float_or_none(health_b.get("effective_rank")),
        "participation_ratio_a": _float_or_none(health_a.get("participation_ratio")),
        "participation_ratio_b": _float_or_none(health_b.get("participation_ratio")),
        "variance_min_a": _float_or_none(health_a.get("variance_min")),
        "variance_mean_a": _float_or_none(health_a.get("variance_mean")),
        "variance_max_a": _float_or_none(health_a.get("variance_max")),
        "variance_min_b": _float_or_none(health_b.get("variance_min")),
        "variance_mean_b": _float_or_none(health_b.get("variance_mean")),
        "variance_max_b": _float_or_none(health_b.get("variance_max")),
        "health_offdiag_abs_mean_a": _float_or_none(health_a.get("offdiag_abs_mean")),
        "health_offdiag_abs_mean_b": _float_or_none(health_b.get("offdiag_abs_mean")),
        "health_offdiag_abs_max_a": _float_or_none(health_a.get("offdiag_abs_max")),
        "health_offdiag_abs_max_b": _float_or_none(health_b.get("offdiag_abs_max")),
        "singular_top1_share_a": _singular_share(health_a.get("singular_values"), top_k=1),
        "singular_top1_share_b": _singular_share(health_b.get("singular_values"), top_k=1),
        "singular_top4_share_a": _singular_share(health_a.get("singular_values"), top_k=4),
        "singular_top4_share_b": _singular_share(health_b.get("singular_values"), top_k=4),
        "visible_rate_a": _float_or_none(visibility.get("view_a_visible_rate")),
        "visible_rate_b": _float_or_none(visibility.get("view_b_visible_rate")),
        "raw_top1": _float_or_none(raw_retrieval.get("top1")),
        "raw_top5": _float_or_none(raw_retrieval.get("top5")),
        "raw_top10": _float_or_none(raw_retrieval.get("top10")),
    }


def load_scorecard_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(extract_masked_multiview_scorecard_row(data, artifact_path=path))
    return rows


def extract_probe_summary(data: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    feature_names = (
        "barlow_clean_last",
        "raw_surface_last",
        "raw_surface_last_plus_barlow_clean_last",
    )
    target_names = ("future_mean_delta", "future_range")
    out: dict[str, dict[str, dict[str, float]]] = {}
    probe_metrics = data.get("probe_metrics", {})
    if not isinstance(probe_metrics, dict):
        return out
    for feature_name in feature_names:
        feature = probe_metrics.get(feature_name, {})
        targets = feature.get("targets", {}) if isinstance(feature, dict) else {}
        if not isinstance(targets, dict):
            continue
        rows: dict[str, dict[str, float]] = {}
        for target_name in target_names:
            target = targets.get(target_name, {})
            if isinstance(target, dict) and "mse" in target and "r2" in target:
                rows[target_name] = {
                    "mse": float(target["mse"]),
                    "r2": float(target["r2"]),
                }
        if rows:
            out[feature_name] = rows
    return out


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_scorecard_markdown(
    rows: list[dict[str, Any]],
    *,
    probe_summary: dict[str, dict[str, dict[str, float]]] | None = None,
    title: str = "World Model Part 1 Scorecard",
) -> str:
    lines = [
        f"# {title}",
        "",
        "## Objective Family",
        "",
        "`masked_multiview_invariance` scorecard consolidation.",
        "",
        "## Hypothesis",
        "",
        "The current Part 1 reference decision should be reproducible from saved",
        "JSON artifacts without hand-transcribing the leaderboard.",
        "",
        "## Masked-Multiview Leaderboard",
        "",
        "| run | family | block | top1 | top5 | top10 | eff rank A/B | sv top1 A/B | health offdiag A/B | raw top10 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {run} | `{family}` | `{block}` | {top1} | {top5} | {top10} | "
            "{rank_a} / {rank_b} | {sv_top1_a} / {sv_top1_b} | "
            "{health_offdiag_a} / {health_offdiag_b} | {raw_top10} |".format(
                run=row["run"],
                family=row["objective_family"],
                block=row["metric_block"],
                top1=_fmt(row["top1"]),
                top5=_fmt(row["top5"]),
                top10=_fmt(row["top10"]),
                rank_a=_fmt(row["effective_rank_a"], 3),
                rank_b=_fmt(row["effective_rank_b"], 3),
                sv_top1_a=_fmt(row["singular_top1_share_a"], 3),
                sv_top1_b=_fmt(row["singular_top1_share_b"], 3),
                health_offdiag_a=_fmt(row["health_offdiag_abs_mean_a"], 3),
                health_offdiag_b=_fmt(row["health_offdiag_abs_mean_b"], 3),
                raw_top10=_fmt(row["raw_top10"]),
            )
        )
    lines.extend(
        [
            "",
            "## Health Detail",
            "",
            "| run | variance min A/B | variance max A/B | sv top4 A/B | Barlow offdiag |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {run} | {var_min_a} / {var_min_b} | {var_max_a} / {var_max_b} | "
            "{sv_top4_a} / {sv_top4_b} | {barlow_offdiag} |".format(
                run=row["run"],
                var_min_a=_fmt(row["variance_min_a"], 6),
                var_min_b=_fmt(row["variance_min_b"], 6),
                var_max_a=_fmt(row["variance_max_a"], 6),
                var_max_b=_fmt(row["variance_max_b"], 6),
                sv_top4_a=_fmt(row["singular_top4_share_a"], 3),
                sv_top4_b=_fmt(row["singular_top4_share_b"], 3),
                barlow_offdiag=_fmt(row["offdiag_abs_mean"], 6),
            )
        )
    if probe_summary:
        lines.extend(
            [
                "",
                "## Frozen Probe Snapshot",
                "",
                "| feature | mean-delta MSE/R2 | range MSE/R2 |",
                "| --- | ---: | ---: |",
            ]
        )
        for feature_name, targets in probe_summary.items():
            mean_delta = targets.get("future_mean_delta", {})
            future_range = targets.get("future_range", {})
            lines.append(
                "| {feature} | {mean_mse} / {mean_r2} | {range_mse} / {range_r2} |".format(
                    feature=feature_name,
                    mean_mse=_fmt(mean_delta.get("mse")),
                    mean_r2=_fmt(mean_delta.get("r2")),
                    range_mse=_fmt(future_range.get("mse")),
                    range_r2=_fmt(future_range.get("r2")),
                )
            )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "HEAD070 remains the Part 1 reference candidate: it preserves strong",
            "same-state retrieval while repairing the low-rank failure seen in",
            "HEAD068 and avoiding the EMA/predictor collapse seen in HEAD066.",
            "",
            "Treat this as a smoke-scale masked-multiview reference candidate,",
            "not an ImageNet-level JEPA result, full-data convergence claim,",
            "general future predictor, regime-classifier success, or Part 2",
            "scenario-quality claim. Downstream probes are mixed and currently",
            "target IV-surface futures only.",
            "",
            "The next step should be a bounded mask-artifact or geometry-stratified",
            "audit, not a new model knob and not Part 2 decoder work.",
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        nargs="*",
        type=Path,
        default=list(DEFAULT_ARTIFACTS),
        help="Masked-multiview Part 1 JSON artifacts to summarize.",
    )
    parser.add_argument(
        "--probe-artifact",
        type=Path,
        default=DEFAULT_PROBE_ARTIFACT,
        help="Optional frozen-probe JSON artifact.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument(
        "--report-title",
        default="World Model Part 1 Scorecard",
        help="Markdown H1 title.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows = load_scorecard_rows(list(args.artifacts))
    probe_summary = {}
    if args.probe_artifact and args.probe_artifact.exists():
        probe_summary = extract_probe_summary(
            json.loads(args.probe_artifact.read_text(encoding="utf-8"))
        )
    payload = {
        "reference_run": REFERENCE_RUN,
        "rows": rows,
        "probe_summary": probe_summary,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown = render_scorecard_markdown(
        rows,
        probe_summary=probe_summary,
        title=args.report_title,
    )
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
