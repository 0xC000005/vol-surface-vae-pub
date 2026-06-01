#!/usr/bin/env python
"""Summarize reverse-caption rollout results by narrative variant group."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def parse_query_id(query_id: str) -> tuple[str, str, str]:
    parts = str(query_id).rsplit("::", 2)
    if len(parts) != 3:
        raise ValueError(f"cannot parse query_id: {query_id}")
    return parts[0], parts[1], parts[2]


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 12)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return _round(float(np.mean(values)))


def _variant_lookup(reverse_report: dict[str, Any], embedding_model: str) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for model_report in reverse_report.get("model_reports", []):
        if str(model_report.get("embedding_model")) != embedding_model:
            continue
        for row in model_report.get("rows", []):
            lookup[(str(row["window_id"]), str(row["variant_id"]))] = row
    return lookup


def summarize_groups(
    reverse_report: dict[str, Any],
    rollout_report: dict[str, Any],
    *,
    embedding_model: str,
) -> dict[str, Any]:
    lookup = _variant_lookup(reverse_report, embedding_model)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unknown: list[str] = []
    for score in rollout_report.get("window_scores", []):
        window_id, variant_id, model = parse_query_id(str(score["query_id"]))
        if model != embedding_model:
            continue
        variant = lookup.get((window_id, variant_id))
        if not variant:
            unknown.append(str(score["query_id"]))
            continue
        row = {
            "window_id": window_id,
            "variant_id": variant_id,
            "variant_group": str(variant["variant_group"]),
            "provider": str(variant["provider"]),
            "target_cosine": float(variant["target_cosine"]),
            "mean_support_cosine": float(variant["mean_support_cosine"]),
            "metrics": score.get("methods", {}),
        }
        grouped[row["variant_group"]].append(row)

    summaries: dict[str, dict[str, Any]] = {}
    for group, rows in sorted(grouped.items()):
        method_values = defaultdict(list)
        for row in rows:
            methods = row["metrics"]
            narrative = methods.get("narrative_generator_topk", {})
            persistence = methods.get("persistence", {})
            for metric in (
                "ensemble_crps_z",
                "energy_score_z",
                "coverage_80",
                "terminal_mae_z",
            ):
                value = narrative.get(metric)
                if value is not None:
                    method_values[f"narrative_{metric}"].append(float(value))
                base = persistence.get(metric)
                if base is not None:
                    method_values[f"persistence_{metric}"].append(float(base))
        crps = _mean(method_values["narrative_ensemble_crps_z"])
        base_crps = _mean(method_values["persistence_ensemble_crps_z"])
        energy = _mean(method_values["narrative_energy_score_z"])
        base_energy = _mean(method_values["persistence_energy_score_z"])
        summaries[group] = {
            "count": len(rows),
            "mean_target_cosine": _mean([row["target_cosine"] for row in rows]),
            "mean_support_cosine": _mean([row["mean_support_cosine"] for row in rows]),
            "narrative_crps": crps,
            "persistence_crps": base_crps,
            "crps_improvement_vs_persistence": (
                None if crps is None or base_crps in (None, 0.0) else _round(1.0 - crps / base_crps)
            ),
            "narrative_energy": energy,
            "persistence_energy": base_energy,
            "energy_improvement_vs_persistence": (
                None
                if energy is None or base_energy in (None, 0.0)
                else _round(1.0 - energy / base_energy)
            ),
            "coverage_80": _mean(method_values["narrative_coverage_80"]),
            "terminal_mae_z": _mean(method_values["narrative_terminal_mae_z"]),
        }
    return {
        "status": "ok",
        "embedding_model": embedding_model,
        "group_summaries": summaries,
        "unknown_query_ids": unknown,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    rows = report["group_summaries"]
    lines = [
        f"# Caption Rollout Group Summary ({report['embedding_model']})",
        "",
        "| Group | Count | Target cos | Support cos | CRPS impr. | Energy impr. | 80% cov. | Terminal MAE z |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, row in sorted(rows.items()):
        lines.append(
            "| {group} | {count} | {target:.3f} | {support:.3f} | {crps:.3f} | {energy:.3f} | {coverage:.3f} | {terminal:.3f} |".format(
                group=group,
                count=row["count"],
                target=row["mean_target_cosine"] or 0.0,
                support=row["mean_support_cosine"] or 0.0,
                crps=row["crps_improvement_vs_persistence"] or 0.0,
                energy=row["energy_improvement_vs_persistence"] or 0.0,
                coverage=row["coverage_80"] or 0.0,
                terminal=row["terminal_mae_z"] or 0.0,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reverse-report", required=True)
    parser.add_argument("--rollout-report", required=True)
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    report = summarize_groups(
        _load_json(args.reverse_report),
        _load_json(args.rollout_report),
        embedding_model=str(args.embedding_model),
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_markdown(Path(args.output_md), report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "embedding_model": report["embedding_model"],
                "group_count": len(report["group_summaries"]),
                "output_json": str(output_json),
                "output_md": str(args.output_md),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
