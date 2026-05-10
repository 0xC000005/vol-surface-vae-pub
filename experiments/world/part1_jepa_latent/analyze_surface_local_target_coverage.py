from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, ".")

from experiments.world.evaluation.surface_local_jepa_data import (  # noqa: E402
    SurfaceLocalJepaBatch,
    build_surface_local_jepa_batch,
)


def _family_rows(batch: SurfaceLocalJepaBatch) -> list[dict[str, Any]]:
    observed = batch.observed_mask
    target = batch.target_mask & observed
    rows = []
    for family in sorted({str(x) for x in batch.target_family.tolist()}):
        mask = batch.target_family.astype(str) == family
        fam_target = target[mask]
        fam_observed = observed[mask]
        rows.append(
            {
                "family": family,
                "n_windows": int(mask.sum()),
                "target_positions": int(fam_target.sum()),
                "hidden_rate": float(fam_target.sum() / fam_observed.sum()),
                "last_row_rate": float(fam_target[:, -1, :].any(axis=-1).mean()),
            }
        )
    return rows


def _surface_cell_rows(batch: SurfaceLocalJepaBatch) -> list[dict[str, Any]]:
    meta = batch.token_metadata
    target = batch.target_mask & batch.observed_mask
    rows = []
    for token in np.flatnonzero(meta.geometry_id == "iv_surface"):
        coord = meta.geometry_coord[token]
        count = int(target[:, :, token].sum())
        rows.append(
            {
                "factor_id": str(meta.factor_id[token]),
                "moneyness_index": int(coord[0]),
                "maturity_index": int(coord[1]),
                "target_positions": count,
                "targeted_windows": int(target[:, :, token].any(axis=1).sum()),
            }
        )
    rows.sort(key=lambda row: row["target_positions"], reverse=True)
    return rows


def _group_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        out.setdefault(str(row[key]), 0)
        out[str(row[key])] += int(row["target_positions"])
    return out


def analyze_surface_local_target_coverage(args: argparse.Namespace) -> dict[str, Any]:
    train = build_surface_local_jepa_batch(
        split="train",
        history_len=args.history_len,
        future_len=args.future_len,
        max_windows=args.max_train_windows,
        seed=args.seed,
    )
    val = build_surface_local_jepa_batch(
        split="val",
        history_len=args.history_len,
        future_len=args.future_len,
        max_windows=args.max_val_windows,
        seed=args.seed + 1000,
    )
    val_cells = _surface_cell_rows(val)
    target = val.target_mask & val.observed_mask
    observed = val.observed_mask
    positions_by_factor = {row["factor_id"]: row["target_positions"] for row in val_cells}
    wing_positions = sum(
        row["target_positions"]
        for row in val_cells
        if row["moneyness_index"] in {0, 4}
    )
    core_positions = sum(
        row["target_positions"]
        for row in val_cells
        if row["moneyness_index"] in {1, 2, 3}
    )
    edge_maturity_positions = sum(
        row["target_positions"]
        for row in val_cells
        if row["maturity_index"] in {0, 4}
    )
    middle_maturity_positions = sum(
        row["target_positions"]
        for row in val_cells
        if row["maturity_index"] in {1, 2, 3}
    )
    return {
        "analysis": "world_model_surface_local_target_coverage",
        "date": "2026-05-10",
        "objective_family": "token_geometry_level_context_to_target_jepa_data_audit",
        "train_shape": list(train.clean_values.shape),
        "val_shape": list(val.clean_values.shape),
        "train_target_positions": int((train.target_mask & train.observed_mask).sum()),
        "val_target_positions": int(target.sum()),
        "val_hidden_rate": float(target.sum() / observed.sum()),
        "val_family_rows": _family_rows(val),
        "val_surface_cells_by_target_count": val_cells,
        "val_surface_positions_by_moneyness": _group_counts(
            val_cells,
            "moneyness_index",
        ),
        "val_surface_positions_by_maturity": _group_counts(
            val_cells,
            "maturity_index",
        ),
        "decision": {
            "largest_gap_cell_target_positions": int(
                positions_by_factor.get("iv_m0_t0", 0)
            ),
            "all_iv_cells_targeted": all(
                int(row["target_positions"]) > 0 for row in val_cells
            ),
            "wing_positions": int(wing_positions),
            "core_positions": int(core_positions),
            "edge_maturity_positions": int(edge_maturity_positions),
            "middle_maturity_positions": int(middle_maturity_positions),
            "covers_head149_problem_regions": bool(
                positions_by_factor.get("iv_m0_t0", 0) > 0
                and wing_positions > 0
                and edge_maturity_positions > 0
            ),
            "promotion_decision": "DATA_AUDIT_ONLY",
            "interpretation": (
                "The data contract covers the HEAD149 wing/edge problem regions, "
                "including iv_m0_t0, but this remains only data evidence before "
                "any encoder or loss is introduced."
            ),
        },
    }


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def render_markdown(result: dict[str, Any], *, title: str) -> str:
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
        "`token_geometry_level_context_to_target_jepa_data_audit`; no model change.",
        "",
        "## Hypothesis",
        "",
        "The surface-local data contract should explicitly cover the wing and edge",
        "maturity cells that dominate the HEAD149 exact-state gap.",
        "",
        "## Coverage Summary",
        "",
        f"- Train shape: `{result['train_shape']}`.",
        f"- Validation shape: `{result['val_shape']}`.",
        f"- Train target positions: `{result['train_target_positions']}`.",
        f"- Validation target positions: `{result['val_target_positions']}`.",
        f"- Validation hidden rate: `{_fmt(result['val_hidden_rate'])}`.",
        "",
        "## Target Families",
        "",
        "| family | windows | target positions | hidden rate | last-row rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in result["val_family_rows"]:
        lines.append(
            "| {family} | {windows} | {positions} | {hidden} | {last} |".format(
                family=row["family"],
                windows=row["n_windows"],
                positions=row["target_positions"],
                hidden=_fmt(row["hidden_rate"]),
                last=_fmt(row["last_row_rate"]),
            )
        )
    lines.extend(
        [
            "",
            "## Surface Position Counts",
            "",
            "| group | 0 | 1 | 2 | 3 | 4 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    by_mon = result["val_surface_positions_by_moneyness"]
    by_mat = result["val_surface_positions_by_maturity"]
    lines.append(
        "| moneyness | {c0} | {c1} | {c2} | {c3} | {c4} |".format(
            c0=by_mon.get("0", 0),
            c1=by_mon.get("1", 0),
            c2=by_mon.get("2", 0),
            c3=by_mon.get("3", 0),
            c4=by_mon.get("4", 0),
        )
    )
    lines.append(
        "| maturity | {c0} | {c1} | {c2} | {c3} | {c4} |".format(
            c0=by_mat.get("0", 0),
            c1=by_mat.get("1", 0),
            c2=by_mat.get("2", 0),
            c3=by_mat.get("3", 0),
            c4=by_mat.get("4", 0),
        )
    )
    lines.extend(
        [
            "",
            "Top targeted IV cells:",
            "",
            "| cell | moneyness | maturity | positions | targeted windows |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["val_surface_cells_by_target_count"][:10]:
        lines.append(
            "| {cell} | {mon} | {mat} | {positions} | {windows} |".format(
                cell=row["factor_id"],
                mon=row["moneyness_index"],
                mat=row["maturity_index"],
                positions=row["target_positions"],
                windows=row["targeted_windows"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- `iv_m0_t0` target positions: `{decision['largest_gap_cell_target_positions']}`.",
            f"- All IV cells targeted: `{decision['all_iv_cells_targeted']}`.",
            f"- Wing positions: `{decision['wing_positions']}`.",
            f"- Core positions: `{decision['core_positions']}`.",
            f"- Edge maturity positions: `{decision['edge_maturity_positions']}`.",
            f"- Middle maturity positions: `{decision['middle_maturity_positions']}`.",
            f"- Covers HEAD149 problem regions: `{decision['covers_head149_problem_regions']}`.",
            f"- Promotion decision: `{decision['promotion_decision']}`.",
            "",
            decision["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit surface-local JEPA target coverage"
    )
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--max_train_windows", type=int, default=1024)
    parser.add_argument("--max_val_windows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2150)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/world/surface_local_target_coverage_head152.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "experiments/world/reports/world_model_head152_surface_local_target_coverage.md"
        ),
    )
    parser.add_argument(
        "--report-title",
        default="World Model HEAD152: Surface-Local Target Coverage",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = analyze_surface_local_target_coverage(args)
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
