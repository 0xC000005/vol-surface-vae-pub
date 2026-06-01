#!/usr/bin/env python
"""Inspect support-component overlap for narrative-conditioned component runs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_fixed_start_manifest_audit import (  # noqa: E402
    parse_case_slug,
)
from experiments.backfill.block_ar.nl_prefix_latent_fixed_start_shape_audit import (  # noqa: E402
    path_energy_distance,
)


DEFAULT_VARIANT_DIR = "decoder_component_topk_narrative_start_checked_gen_temp_0p50"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _operational_variant_index(report: dict[str, Any]) -> int:
    rows = report.get("variant_rows", [])
    for index, row in enumerate(rows):
        if isinstance(row, dict) and bool(row.get("is_operational")):
            return int(index)
    return max(0, len(rows) - 1)


def _normalize_weights(items: dict[int, float]) -> dict[int, float]:
    total = float(sum(max(0.0, value) for value in items.values()))
    if total <= 0.0:
        return {}
    return {int(key): float(value) / total for key, value in items.items()}


def _effective_n(weights: dict[int, float]) -> float:
    denom = sum(float(value) ** 2 for value in weights.values())
    return 0.0 if denom <= 0.0 else float(1.0 / denom)


def load_support_case(case_dir: Path, *, case_name: str) -> dict[str, Any]:
    report = _read_json(case_dir / "prefix_latent_story_smoke_report.json")
    arrays = np.load(case_dir / "prefix_latent_story_smoke_arrays.npz")
    variant_index = _operational_variant_index(report)
    component_variant = np.asarray(arrays["rollout_component_variant_index"])
    component_window = np.asarray(arrays["rollout_component_window_index"])
    component_weight = np.asarray(arrays["rollout_component_weight"], dtype=np.float64)
    decoded_prefixes = np.asarray(arrays["decoded_history_level"], dtype=np.float32)
    generated_states = np.asarray(arrays["generated_states"], dtype=np.float32)
    requested_raw = np.asarray(arrays["requested_raw"], dtype=np.float32)
    mask = component_variant == int(variant_index)
    weights: dict[int, float] = defaultdict(float)
    for window, weight in zip(component_window[mask], component_weight[mask]):
        weights[int(window)] += float(weight)
    normalized = _normalize_weights(dict(weights))
    narrative, start_index = parse_case_slug(case_name)
    return {
        "case_name": str(case_name),
        "narrative": str(narrative),
        "start_index": int(start_index),
        "variant_index": int(variant_index),
        "support_weights": normalized,
        "support_windows": sorted(normalized.keys()),
        "top_window": max(normalized.items(), key=lambda item: item[1])[0]
        if normalized
        else None,
        "effective_n": _effective_n(normalized),
        "support_count": int(len(normalized)),
        "_decoded_prefix": decoded_prefixes[int(variant_index)],
        "_states": generated_states[int(variant_index)],
        "_start": requested_raw[int(variant_index)],
    }


def discover_support_cases(root: Path, *, variant_dir: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_root in sorted(Path(root).iterdir()):
        if not case_root.is_dir():
            continue
        try:
            _, start_index = parse_case_slug(case_root.name)
        except ValueError:
            continue
        case_dir = case_root / f"fixed_start_{start_index}" / variant_dir
        if not (case_dir / "prefix_latent_story_smoke_arrays.npz").exists():
            continue
        rows.append(load_support_case(case_dir, case_name=case_root.name))
    return rows


def weighted_overlap(left: dict[int, float], right: dict[int, float]) -> float:
    keys = set(left) | set(right)
    return float(sum(min(float(left.get(key, 0.0)), float(right.get(key, 0.0))) for key in keys))


def support_jaccard(left: list[int], right: list[int]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 0.0
    return float(len(left_set & right_set) / len(union))


def decoded_prefix_l2(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_prefix = np.asarray(left["_decoded_prefix"], dtype=np.float32)
    right_prefix = np.asarray(right["_decoded_prefix"], dtype=np.float32)
    diff = left_prefix - right_prefix
    return float(np.linalg.norm(diff.ravel()) / np.sqrt(max(diff.size, 1)))


def generated_path_energy(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_case = {
        "states": np.asarray(left["_states"], dtype=np.float32),
        "start": np.asarray(left["_start"], dtype=np.float32),
    }
    right_case = {
        "states": np.asarray(right["_states"], dtype=np.float32),
        "start": np.asarray(right["_start"], dtype=np.float32),
    }
    return float(path_energy_distance(left_case, right_case))


def _public_case(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def build_support_diagnostics(
    roots: dict[str, Path],
    *,
    variant_dir: str = DEFAULT_VARIANT_DIR,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for label, root in roots.items():
        for row in discover_support_cases(root, variant_dir=variant_dir):
            row["root_label"] = str(label)
            rows.append(row)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["root_label"]), int(row["start_index"]))].append(row)
    start_summaries: dict[str, Any] = {}
    for (label, start_index), group in sorted(grouped.items()):
        overlaps = []
        jaccards = []
        top_same = []
        prefix_l2 = []
        path_energy = []
        for left, right in combinations(group, 2):
            overlap = weighted_overlap(left["support_weights"], right["support_weights"])
            jaccard = support_jaccard(left["support_windows"], right["support_windows"])
            same_top = left["top_window"] == right["top_window"]
            prefix_distance = decoded_prefix_l2(left, right)
            rollout_energy = generated_path_energy(left, right)
            overlaps.append(overlap)
            jaccards.append(jaccard)
            top_same.append(1.0 if same_top else 0.0)
            prefix_l2.append(prefix_distance)
            path_energy.append(rollout_energy)
            pair_rows.append(
                {
                    "root_label": str(label),
                    "start_index": int(start_index),
                    "left_case": left["case_name"],
                    "right_case": right["case_name"],
                    "weighted_overlap": float(overlap),
                    "jaccard": float(jaccard),
                    "same_top_window": bool(same_top),
                    "decoded_prefix_l2": float(prefix_distance),
                    "generated_path_energy": float(rollout_energy),
                }
            )
        start_summaries[f"{label}:start{start_index}"] = {
            "case_count": int(len(group)),
            "support_count_median": _median([row["support_count"] for row in group]),
            "effective_n_median": _median([row["effective_n"] for row in group]),
            "weighted_overlap_median": _median(overlaps),
            "jaccard_median": _median(jaccards),
            "same_top_window_rate": _mean(top_same),
            "decoded_prefix_l2_median": _median(prefix_l2),
            "generated_path_energy_median": _median(path_energy),
            "generated_energy_per_prefix_l2": _safe_ratio(
                _median(path_energy),
                _median(prefix_l2),
            ),
            "unique_top_windows": sorted(
                {
                    int(row["top_window"])
                    for row in group
                    if row.get("top_window") is not None
                }
            ),
        }
    return {
        "case_count": int(len(rows)),
        "pair_count": int(len(pair_rows)),
        "start_summaries": start_summaries,
        "cases": [_public_case(row) for row in rows],
        "pairs": pair_rows,
        "interpretation": [
            "High weighted overlap means different narratives are reusing similar support weight mass.",
            "High same-top-window rate means narratives share the same leading support analogue.",
            "Decoded-prefix distance compared with generated path energy diagnoses whether latent-prefix differences survive rollout.",
            "Support overlap is diagnostic only; rollout path audits decide pass/warning/fail status.",
        ],
    }


def _median(values: list[float]) -> float | None:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return float(median(finite)) if finite else None


def _mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(float(denominator)) < 1e-12:
        return None
    return float(numerator) / float(denominator)


def parse_root_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    label, path = value.split("=", 1)
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed-root", action="append", required=True)
    parser.add_argument("--variant-dir", default=DEFAULT_VARIANT_DIR)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    roots = dict(parse_root_arg(item) for item in args.observed_root)
    report = build_support_diagnostics(roots, variant_dir=str(args.variant_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "component_support_diagnostics.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "case_count": report["case_count"],
                "pair_count": report["pair_count"],
                "report": str(path),
                "start_summaries": report["start_summaries"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
