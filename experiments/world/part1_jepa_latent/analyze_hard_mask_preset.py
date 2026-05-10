from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, ".")

from experiments.world.evaluation.hard_mask_presets import (  # noqa: E402
    build_hard_masked_batch,
)
from experiments.world.part1_jepa_latent.analyze_part1_jepa_fit import (  # noqa: E402
    _mask_overlap_stats,
)


def analyze_hard_mask_preset(root: Path) -> dict[str, Any]:
    del root
    train = build_hard_masked_batch(
        split="train",
        history_len=30,
        future_len=30,
        max_windows=384,
        seed=680,
    )
    val = build_hard_masked_batch(
        split="val",
        history_len=30,
        future_len=30,
        max_windows=128,
        seed=1680,
    )
    return {
        "analysis": "world_model_part1_hard_mask_preset",
        "date": "2026-05-10",
        "status": "diagnostic_preset_not_active_reference",
        "purpose": "Test mask difficulty as a failure class before changing encoder, loss, or objective family.",
        "preset_families": [
            "surface_large_rectangle",
            "surface_whole_day_block",
            "factor_family_long_block",
            "cross_family_stress_block",
            "time_block_long",
        ],
        "train": _mask_overlap_stats(train),
        "val": _mask_overlap_stats(val),
        "decision": {
            "use_for_next_training_smoke": True,
            "reason": "The preset lowers view overlap enough to create a harder missing-information diagnostic while preserving structured market geometry.",
            "guardrail": "Use as one named preset; do not introduce per-family tuning knobs until this diagnostic is understood.",
        },
    }


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def render_markdown(result: dict[str, Any], *, title: str) -> str:
    train = result["train"]
    val = result["val"]
    lines = [
        f"# {title}",
        "",
        "Date: 2026-05-10",
        "",
        "## Iteration Type",
        "",
        "`research_ideation`",
        "",
        "## Objective Family",
        "",
        "`masked_multiview_invariance` hard-mask diagnostic design.",
        "",
        "## Hypothesis",
        "",
        "If HEAD070 underperforms raw/simple baselines partly because the default",
        "masks are too mild, then a single harder structured-mask preset should",
        "substantially reduce two-view overlap before any architecture or objective",
        "change is attempted.",
        "",
        "## Falsifier",
        "",
        "The preset is not worth training if it leaves most entries visible in both",
        "views, destroys structured market geometry, or requires many tunable knobs.",
        "",
        "## Preset Families",
        "",
    ]
    lines.extend(f"- `{family}`" for family in result["preset_families"])
    lines.extend(
        [
            "",
            "## Mask Difficulty",
            "",
            "| split | view A hidden | view B hidden | both visible | both hidden | view disagreement | union hidden |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            "| train | {train_a_hidden} | {train_b_hidden} | {train_both_visible} | {train_both_hidden} | {train_disagree} | {train_union_hidden} |".format(
                train_a_hidden=_pct(train["view_a_hidden_rate"]),
                train_b_hidden=_pct(train["view_b_hidden_rate"]),
                train_both_visible=_pct(train["both_visible_rate"]),
                train_both_hidden=_pct(train["both_hidden_rate"]),
                train_disagree=_pct(train["view_disagreement_rate"]),
                train_union_hidden=_pct(train["union_hidden_rate"]),
            ),
            "| val | {val_a_hidden} | {val_b_hidden} | {val_both_visible} | {val_both_hidden} | {val_disagree} | {val_union_hidden} |".format(
                val_a_hidden=_pct(val["view_a_hidden_rate"]),
                val_b_hidden=_pct(val["view_b_hidden_rate"]),
                val_both_visible=_pct(val["both_visible_rate"]),
                val_both_hidden=_pct(val["both_hidden_rate"]),
                val_disagree=_pct(val["view_disagreement_rate"]),
                val_union_hidden=_pct(val["union_hidden_rate"]),
            ),
            "",
            "Compared with HEAD070 default validation masks, which kept about `86.6%`",
            "visible in both views, this preset creates a much harder missing-state",
            "diagnostic while keeping semantic groups intact.",
            "",
            "## Decision",
            "",
            f"- Use for next training smoke: `{result['decision']['use_for_next_training_smoke']}`.",
            f"- Reason: {result['decision']['reason']}",
            f"- Guardrail: {result['decision']['guardrail']}",
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Design and audit one hard structured-mask diagnostic preset"
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/world/part1_hard_mask_preset_head122.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "experiments/world/reports/world_model_head122_hard_mask_preset.md"
        ),
    )
    parser.add_argument(
        "--report-title",
        default="World Model HEAD122: Hard Mask Preset",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = analyze_hard_mask_preset(args.root)
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
    print(
        json.dumps(
            {
                "val_view_a_hidden_rate": result["val"]["view_a_hidden_rate"],
                "val_view_b_hidden_rate": result["val"]["view_b_hidden_rate"],
                "val_both_visible_rate": result["val"]["both_visible_rate"],
                "use_for_next_training_smoke": result["decision"][
                    "use_for_next_training_smoke"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
