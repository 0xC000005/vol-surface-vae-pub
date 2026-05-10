from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from experiments.world.evaluation.masked_multiview_data import (  # noqa: E402
    build_masked_multiview_batch,
)
from experiments.world.evaluation.masked_multiview_metrics import (  # noqa: E402
    flattened_time_rows,
    same_state_multiview_metrics,
)
from experiments.world.part1_jepa_latent.masked_multiview_barlow_probe_audit import (  # noqa: E402
    load_direct_barlow_checkpoint,
)
from experiments.world.part1_jepa_latent.masked_multiview_barlow_smoke import (  # noqa: E402
    encode_direct_barlow_split,
)


def stratified_same_state_metrics(
    view_a: np.ndarray,
    view_b: np.ndarray,
    labels: np.ndarray,
    *,
    min_windows: int,
) -> dict[str, dict[str, Any]]:
    a = np.asarray(view_a, dtype=np.float32)
    b = np.asarray(view_b, dtype=np.float32)
    y = np.asarray(labels, dtype=object).reshape(-1)
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError("view_a and view_b must share shape (N, T, D)")
    if a.shape[0] != y.shape[0]:
        raise ValueError("labels must have one row per window")
    rows: dict[str, dict[str, Any]] = {}
    for label in sorted({str(x) for x in y.tolist()}):
        mask = y.astype(str) == label
        n_windows = int(np.sum(mask))
        if n_windows < min_windows:
            continue
        metrics = same_state_multiview_metrics(
            flattened_time_rows(a[mask]),
            flattened_time_rows(b[mask]),
        )
        rows[label] = {
            "n_windows": n_windows,
            "n_time_rows": int(n_windows * a.shape[1]),
            "alignment_mse": metrics["alignment"]["mse"],
            "cosine_mean": metrics["alignment"]["cosine_mean"],
            "retrieval_top1": metrics["retrieval"]["top1"],
            "retrieval_top5": metrics["retrieval"]["top5"],
            "retrieval_top10": metrics["retrieval"]["top10"],
            "retrieval_mrr": metrics["retrieval"]["mrr"],
            "effective_rank_a": metrics["view_a_health"]["effective_rank"],
            "effective_rank_b": metrics["view_b_health"]["effective_rank"],
            "offdiag_abs_mean": metrics["barlow"]["offdiag_abs_mean"],
        }
    return rows


def audit_stratified_mask_metrics(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = load_direct_barlow_checkpoint(args.checkpoint, device=device)
    val = build_masked_multiview_batch(
        split="val",
        history_len=args.history_len,
        future_len=args.future_len,
        max_windows=args.max_val_windows,
        seed=args.seed + 1000,
        normalize=True,
    )
    encoded = encode_direct_barlow_split(
        model,
        val,
        batch_size=args.batch_size,
        device=device,
    )
    by_family_a = stratified_same_state_metrics(
        encoded["view_a"],
        encoded["view_b"],
        val.mask_family_a,
        min_windows=args.min_windows,
    )
    by_family_b = stratified_same_state_metrics(
        encoded["view_a"],
        encoded["view_b"],
        val.mask_family_b,
        min_windows=args.min_windows,
    )
    return {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "val_shape": list(val.clean_values.shape),
        "min_windows": int(args.min_windows),
        "by_mask_family_a": by_family_a,
        "by_mask_family_b": by_family_b,
        "decision_hint": _decision_hint(by_family_a, by_family_b),
    }


def _decision_hint(
    by_family_a: dict[str, dict[str, Any]],
    by_family_b: dict[str, dict[str, Any]],
) -> str:
    rows = list(by_family_a.values()) + list(by_family_b.values())
    if not rows:
        return "insufficient_strata"
    min_top10 = min(float(row["retrieval_top10"]) for row in rows)
    max_offdiag = max(float(row["offdiag_abs_mean"]) for row in rows)
    if min_top10 >= 0.70 and max_offdiag <= 0.35:
        return "no_large_stratified_failure"
    if min_top10 < 0.50:
        return "possible_stratified_retrieval_failure"
    return "mixed_stratified_health"


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _append_table(lines: list[str], title: str, rows: dict[str, dict[str, Any]]) -> None:
    lines.extend(
        [
            f"## {title}",
            "",
            "| stratum | windows | top1 | top5 | top10 | eff rank A/B | offdiag |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, row in rows.items():
        lines.append(
            "| {label} | {n} | {top1} | {top5} | {top10} | {rank_a} / {rank_b} | {offdiag} |".format(
                label=label,
                n=row["n_windows"],
                top1=_fmt(row["retrieval_top1"]),
                top5=_fmt(row["retrieval_top5"]),
                top10=_fmt(row["retrieval_top10"]),
                rank_a=_fmt(row["effective_rank_a"]),
                rank_b=_fmt(row["effective_rank_b"]),
                offdiag=_fmt(row["offdiag_abs_mean"]),
            )
        )
    lines.append("")


def render_markdown(result: dict[str, Any], *, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "## Objective Family",
        "",
        "`masked_multiview_invariance` geometry/mask-family diagnostic.",
        "",
        "## Hypothesis",
        "",
        "If HEAD070 is robust under structured masking, retrieval/rank should not",
        "collapse for one mask family while aggregate metrics look healthy.",
        "",
    ]
    _append_table(lines, "Grouped By View A Mask Family", result["by_mask_family_a"])
    _append_table(lines, "Grouped By View B Mask Family", result["by_mask_family_b"])
    lines.extend(
        [
            "## Decision",
            "",
            f"`{result['decision_hint']}`.",
            "",
            "This is a validation diagnostic. It should guide masking/probe audits,",
            "not introduce a new representation objective by itself.",
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit HEAD070 metrics by mask family")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/world/checkpoints/part1_jepa_latent/masked_multiview_barlow_head070.pt"),
    )
    parser.add_argument("--output-json", type=Path, default=Path("results/world/masked_multiview_stratified_head084.json"))
    parser.add_argument("--output-md", type=Path, default=Path("experiments/world/reports/world_model_head084_stratified_mask_audit.md"))
    parser.add_argument("--report-title", default="World Model HEAD084: Stratified Mask-Family Audit")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--max_val_windows", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=680)
    parser.add_argument("--min_windows", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = audit_stratified_mask_metrics(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(result, title=args.report_title), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
