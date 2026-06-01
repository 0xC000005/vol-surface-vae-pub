#!/usr/bin/env python
"""Reproject cached condition-only text embeddings through a different bridge.

The condition-only casebook reports store the original OpenAI text embedding.
This utility lets us reuse that paid embedding and only swap the local
text-to-memory adapter, which is useful when comparing the representative
220-window bridge against the full 380-window bridge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_component_fixed_start_controls import (  # noqa: E402
    DEFAULT_CONDITION_REPORTS,
)
from experiments.backfill.block_ar.nl_risk_manager_story_smoke import (  # noqa: E402
    _load_bridge_adapter,
)
from experiments.backfill.block_ar.nl_scenario_level_evaluation import (  # noqa: E402
    load_bridge_arrays,
)
from experiments.backfill.block_ar.nl_text_conditioning import normalize_rows  # noqa: E402


DEFAULT_FULL_BRIDGE_ARRAYS = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_bridge_eval_full_906b_all_windows/bridge_eval_arrays.npz"
)
DEFAULT_FULL_BRIDGE_ADAPTER = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_bridge_eval_full_906b_all_windows/bridge_adapter.pt"
)
DEFAULT_OUTPUT_ROOT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_condition_only_full906b_914a"
)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _arrays_path_from_report(report_path: Path, report: dict[str, Any]) -> Path:
    artifact_paths = report.get("artifact_paths", {})
    if isinstance(artifact_paths, dict) and artifact_paths.get("arrays"):
        return Path(str(artifact_paths["arrays"]))
    return report_path.with_name("condition_only_report_arrays.npz")


def reproject_one(
    *,
    case_name: str,
    source_report_path: Path,
    output_root: Path,
    bridge_arrays: str | Path,
    bridge_adapter: str | Path,
) -> dict[str, Any]:
    source_report = _load_json(source_report_path)
    source_arrays = np.load(_arrays_path_from_report(source_report_path, source_report))
    if "query_embedding" not in source_arrays:
        raise ValueError(f"{source_report_path}: missing query_embedding")
    query_embedding = np.asarray(source_arrays["query_embedding"], dtype=np.float32)
    if query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
        raise ValueError(f"{source_report_path}: query_embedding must have shape [1,D]")

    arrays = load_bridge_arrays(bridge_arrays)
    condition_dim = int(np.asarray(arrays["memory_targets"]).shape[1])
    adapter = _load_bridge_adapter(
        bridge_adapter,
        embedding_dim=int(query_embedding.shape[1]),
        condition_dim=condition_dim,
    )
    adapter.eval()
    with torch.no_grad():
        text_memory = (
            adapter(torch.from_numpy(normalize_rows(query_embedding)).float())
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    output_dir = output_root / case_name
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "condition_only_report_arrays.npz"
    np.savez_compressed(
        arrays_path,
        text_memory=text_memory.astype(np.float32),
        query_embedding=query_embedding.astype(np.float32),
    )

    report = dict(source_report)
    report["scope_note"] = (
        "Condition-only narrative report reprojected from a cached OpenAI "
        "text embedding through a replacement local bridge adapter."
    )
    report["case_name"] = case_name
    cached_query = dict(report.get("cached_query", {}))
    embedding_metadata = dict(cached_query.get("embedding_metadata", {}))
    embedding_metadata.update(
        {
            "condition_dim": condition_dim,
            "reprojected_from_report": str(source_report_path),
            "bridge_arrays": str(bridge_arrays),
            "bridge_adapter": str(bridge_adapter),
            "openai_reused_query_embedding": True,
        }
    )
    cached_query["embedding_metadata"] = embedding_metadata
    report["cached_query"] = cached_query
    report["artifact_paths"] = {
        "report": str(output_dir / "condition_only_report.json"),
        "arrays": str(arrays_path),
    }
    _write_json(report["artifact_paths"]["report"], report)
    return {
        "case_name": case_name,
        "report": report["artifact_paths"]["report"],
        "arrays": report["artifact_paths"]["arrays"],
        "source_report": str(source_report_path),
        "condition_dim": condition_dim,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bridge-arrays", default=DEFAULT_FULL_BRIDGE_ARRAYS)
    parser.add_argument("--bridge-adapter", default=DEFAULT_FULL_BRIDGE_ADAPTER)
    parser.add_argument(
        "--case",
        action="append",
        help="Optional case name to reproject. Defaults to all built-in casebook reports.",
    )
    args = parser.parse_args()

    selected = set(args.case or DEFAULT_CONDITION_REPORTS.keys())
    rows = []
    for case_name, source_report in sorted(DEFAULT_CONDITION_REPORTS.items()):
        if case_name not in selected:
            continue
        rows.append(
            reproject_one(
                case_name=case_name,
                source_report_path=Path(source_report),
                output_root=Path(args.output_root),
                bridge_arrays=args.bridge_arrays,
                bridge_adapter=args.bridge_adapter,
            )
        )
    print(json.dumps({"output_root": str(args.output_root), "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
