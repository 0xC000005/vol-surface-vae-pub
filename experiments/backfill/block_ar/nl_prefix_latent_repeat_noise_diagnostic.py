#!/usr/bin/env python
"""Diagnose repeat noise in fixed-start narrative component rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_component_support_diagnostics import (  # noqa: E402
    DEFAULT_VARIANT_DIR,
    decoded_prefix_l2,
    generated_path_energy,
    weighted_overlap,
)


DEFAULT_COMPONENT_ROOT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_component_mixture_fixed_start_904k_s384_uncalibrated"
)
DEFAULT_CONTROL_ROOT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_component_fixed_start_controls_904k_s384_uncalibrated"
)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _operational_variant_index(report: dict[str, Any], variant_count: int) -> int:
    selected = report.get("selected_start_state", {})
    if isinstance(selected, dict) and selected.get("variant_index") is not None:
        idx = int(selected["variant_index"])
        if 0 <= idx < int(variant_count):
            return idx
    for idx, row in enumerate(report.get("variant_rows", [])):
        if (
            idx < int(variant_count)
            and isinstance(row, dict)
            and row.get("is_operational")
        ):
            return int(idx)
    return 0


def _normalize_weights(items: dict[int, float]) -> dict[int, float]:
    total = float(sum(max(0.0, value) for value in items.values()))
    if total <= 0.0:
        return {}
    return {int(key): float(value) / total for key, value in items.items()}


def load_rollout_case(case_dir: Path, *, case_name: str, group: str) -> dict[str, Any]:
    report = _load_json(case_dir / "prefix_latent_story_smoke_report.json")
    arrays = np.load(case_dir / "prefix_latent_story_smoke_arrays.npz")
    generated = np.asarray(arrays["generated_states"], dtype=np.float32)
    variant_index = _operational_variant_index(report, generated.shape[0])
    component_variant = np.asarray(arrays["rollout_component_variant_index"])
    component_window = np.asarray(arrays["rollout_component_window_index"])
    component_weight = np.asarray(arrays["rollout_component_weight"], dtype=np.float64)
    mask = component_variant == int(variant_index)
    raw_weights: dict[int, float] = {}
    for window, weight in zip(component_window[mask], component_weight[mask]):
        raw_weights[int(window)] = raw_weights.get(int(window), 0.0) + float(weight)
    return {
        "case_name": str(case_name),
        "group": str(group),
        "variant_index": int(variant_index),
        "support_weights": _normalize_weights(raw_weights),
        "_decoded_prefix": np.asarray(
            arrays["decoded_history_level"], dtype=np.float32
        )[int(variant_index)],
        "_states": generated[int(variant_index)],
        "_start": np.asarray(arrays["requested_raw"], dtype=np.float32)[
            int(variant_index)
        ],
    }


def discover_observed_cases(
    component_root: Path, *, variant_dir: str
) -> list[dict[str, Any]]:
    rows = []
    for case_root in sorted(Path(component_root).iterdir()):
        case_dir = case_root / "fixed_start_18" / variant_dir
        if (case_dir / "prefix_latent_story_smoke_arrays.npz").exists():
            rows.append(
                load_rollout_case(case_dir, case_name=case_root.name, group="observed")
            )
    return rows


def discover_repeat_cases(control_root: Path) -> dict[str, list[dict[str, Any]]]:
    repeat_root = Path(control_root) / "repeat_controls"
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case_root in sorted(repeat_root.iterdir()):
        if not case_root.is_dir():
            continue
        for seed_root in sorted(case_root.iterdir()):
            if not seed_root.is_dir():
                continue
            arrays_path = seed_root / "prefix_latent_story_smoke_arrays.npz"
            if arrays_path.exists():
                grouped.setdefault(case_root.name, []).append(
                    load_rollout_case(
                        seed_root,
                        case_name=f"{case_root.name}#{seed_root.name}",
                        group="repeat",
                    )
                )
    return grouped


def _pair_row(
    left: dict[str, Any], right: dict[str, Any], *, pair_type: str
) -> dict[str, Any]:
    prefix = decoded_prefix_l2(left, right)
    energy = generated_path_energy(left, right)
    return {
        "pair_type": str(pair_type),
        "left_case": str(left["case_name"]),
        "right_case": str(right["case_name"]),
        "support_weighted_overlap": weighted_overlap(
            left["support_weights"], right["support_weights"]
        ),
        "decoded_prefix_l2": float(prefix),
        "generated_path_energy": float(energy),
        "generated_energy_per_prefix_l2": (
            None if prefix <= 1e-12 else float(energy / prefix)
        ),
    }


def _median(values: list[float]) -> float | None:
    return None if not values else float(median(values))


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pair_count": int(len(rows)),
        "support_weighted_overlap_median": _median(
            [float(row["support_weighted_overlap"]) for row in rows]
        ),
        "decoded_prefix_l2_median": _median(
            [float(row["decoded_prefix_l2"]) for row in rows]
        ),
        "generated_path_energy_median": _median(
            [float(row["generated_path_energy"]) for row in rows]
        ),
        "generated_energy_per_prefix_l2_median": _median(
            [
                float(row["generated_energy_per_prefix_l2"])
                for row in rows
                if row["generated_energy_per_prefix_l2"] is not None
            ]
        ),
    }


def build_repeat_noise_report(
    *,
    component_root: Path,
    control_root: Path,
    variant_dir: str = DEFAULT_VARIANT_DIR,
) -> dict[str, Any]:
    observed_cases = discover_observed_cases(component_root, variant_dir=variant_dir)
    repeat_groups = discover_repeat_cases(control_root)
    observed_pairs = [
        _pair_row(left, right, pair_type="observed_narrative")
        for left, right in combinations(observed_cases, 2)
    ]
    repeat_pairs = []
    for group in repeat_groups.values():
        repeat_pairs.extend(
            _pair_row(left, right, pair_type="same_narrative_repeat")
            for left, right in combinations(group, 2)
        )
    observed_summary = _summarize(observed_pairs)
    repeat_summary = _summarize(repeat_pairs)
    repeat_prefix = repeat_summary.get("decoded_prefix_l2_median")
    observed_prefix = observed_summary.get("decoded_prefix_l2_median")
    repeat_energy = repeat_summary.get("generated_path_energy_median")
    observed_energy = observed_summary.get("generated_path_energy_median")
    prefix_ratio = (
        None
        if repeat_prefix is None or observed_prefix is None or observed_prefix <= 1e-12
        else float(repeat_prefix / observed_prefix)
    )
    energy_ratio = (
        None
        if repeat_energy is None or observed_energy is None or observed_energy <= 1e-12
        else float(repeat_energy / observed_energy)
    )
    if prefix_ratio is not None and prefix_ratio > 0.50:
        diagnosis = "prefix_decoder_or_support_instability"
    elif energy_ratio is not None and energy_ratio > 0.75:
        diagnosis = "rollout_sampling_or_readout_noise"
    else:
        diagnosis = "repeat_noise_below_observed_signal"
    return {
        "status": (
            "warning" if diagnosis != "repeat_noise_below_observed_signal" else "pass"
        ),
        "research_lane": "post_experiment_analysis",
        "result_status": "mechanism_found",
        "benchmark_floor_status": "not_applicable",
        "scope_note": (
            "Diagnoses whether same-narrative repeat noise comes from changing "
            "support/prefixes or from downstream rollout/readout stochasticity."
        ),
        "component_root": str(component_root),
        "control_root": str(control_root),
        "variant_dir": str(variant_dir),
        "diagnosis": diagnosis,
        "ratios": {
            "repeat_to_observed_decoded_prefix_l2": prefix_ratio,
            "repeat_to_observed_generated_path_energy": energy_ratio,
        },
        "summaries": {
            "observed_narrative": observed_summary,
            "same_narrative_repeat": repeat_summary,
        },
        "observed_pairs": observed_pairs,
        "repeat_pairs": repeat_pairs,
        "interpretation": [
            "High repeat support overlap with low repeat prefix distance points away from support selection as the issue.",
            "High repeat prefix distance means the decoder/prefix construction is unstable across seeds.",
            "High repeat generated path energy with low repeat prefix distance means rollout sampling or readout noise is washing out narrative effects.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-root", default=DEFAULT_COMPONENT_ROOT)
    parser.add_argument("--control-root", default=DEFAULT_CONTROL_ROOT)
    parser.add_argument("--variant-dir", default=DEFAULT_VARIANT_DIR)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    report = build_repeat_noise_report(
        component_root=Path(args.component_root),
        control_root=Path(args.control_root),
        variant_dir=str(args.variant_dir),
    )
    report["artifact_paths"] = {
        "report": str(output_dir / "repeat_noise_diagnostic.json")
    }
    _write_json(report["artifact_paths"]["report"], report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "diagnosis": report["diagnosis"],
                "ratios": report["ratios"],
                "report": report["artifact_paths"]["report"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
