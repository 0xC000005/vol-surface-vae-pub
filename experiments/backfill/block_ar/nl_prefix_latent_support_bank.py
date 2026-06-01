#!/usr/bin/env python
"""Build a broad SNI support bank for narrative-conditioned scenarios.

The bridge report contains labeled narrative windows. The support bank contains
historical SNI prefix memories that can be searched as analogue support without
requiring every support window to have an LLM caption.
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

from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (  # noqa: E402
    load_model,
)
from experiments.backfill.block_ar.evaluate_662a_state_aware_normalized_innovation_flow import (  # noqa: E402
    build_val_block,
)
from experiments.backfill.block_ar.nl_narrative_grounded_scenario_pipeline import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    build_window_metadata,
    compute_memory_targets,
)
from experiments.backfill.block_ar.nl_prefix_latent_memory_decoder import (  # noqa: E402
    _future_raw_from_block,
)
from experiments.backfill.block_ar.nl_scenario_level_evaluation import (  # noqa: E402
    future_delta_paths,
)


DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_bank_train_all_939a"
)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _metadata_with_local_identity(
    rows: list[dict[str, Any]],
    *,
    split: str,
    train_indices: np.ndarray,
) -> list[dict[str, Any]]:
    train_set = {int(idx) for idx in np.asarray(train_indices, dtype=np.int64)}
    out: list[dict[str, Any]] = []
    for local_idx, row in enumerate(rows):
        enriched = dict(row)
        enriched.update(
            {
                "window_index": int(local_idx),
                "window_id": f"joint39_{split}_{local_idx:04d}",
                "manifest_split": (
                    "support_train"
                    if int(local_idx) in train_set
                    else "support_decoder_test"
                ),
                "support_source_split": str(split),
            }
        )
        out.append(enriched)
    return out


def _date_range(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [str(row[key]) for row in rows if row.get(key)]
    if not values:
        return {"min": None, "max": None}
    return {"min": min(values), "max": max(values)}


def _int_range(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [int(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return {"min": None, "max": None}
    return {"min": min(values), "max": max(values)}


def build_support_bank(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu"
    )
    model, payload = load_model(args.checkpoint, device)
    (
        history_level,
        history_norm,
        center,
        scale,
        drift_feature,
        history_raw,
        _specs,
        block,
    ) = build_val_block(args, payload)
    n = int(history_level.shape[0])
    if n < 2:
        raise ValueError("support bank needs at least two windows")
    decoder_test_n = min(max(1, int(args.decoder_test_windows)), n - 1)
    train_indices = np.arange(0, n - decoder_test_n, dtype=np.int64)
    test_indices = np.arange(n - decoder_test_n, n, dtype=np.int64)
    support_indices = np.arange(0, n, dtype=np.int64)
    metadata = _metadata_with_local_identity(
        build_window_metadata(
            block,
            history_len=int(payload["config"]["history_len"]),
            future_len=int(payload["config"]["future_len"]),
        ),
        split=str(args.eval_split),
        train_indices=train_indices,
    )
    memory_targets = compute_memory_targets(
        model,
        history_level,
        history_norm,
        center,
        scale,
        drift_feature,
        device=device,
        batch_size=int(args.batch_size),
    )
    future_raw = _future_raw_from_block(block, n, int(history_raw.shape[-1]))
    future_delta = future_delta_paths(history_raw, future_raw)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "support_bank_arrays.npz"
    report_path = output_dir / "support_bank_report.json"
    np.savez_compressed(
        arrays_path,
        memory_targets=memory_targets.astype(np.float32),
        history_level=history_level.astype(np.float32),
        history_norm=history_norm.astype(np.float32),
        center=center.astype(np.float32),
        scale=scale.astype(np.float32),
        drift_feature=drift_feature.astype(np.float32),
        history_raw=history_raw.astype(np.float32),
        future_raw=future_raw.astype(np.float32),
        future_delta=future_delta.astype(np.float32),
        train_indices=train_indices.astype(np.int64),
        test_indices=test_indices.astype(np.int64),
        support_indices=support_indices.astype(np.int64),
    )
    train_rows = [metadata[int(idx)] for idx in train_indices]
    report = {
        "status": "ok",
        "scope_note": (
            "Broad SNI support bank for narrative-conditioned support search. "
            "No LLM captions are required for support candidates; only frozen "
            "generator prefix memories, histories, dates, and train/test indices "
            "are stored."
        ),
        "artifact_paths": {
            "report": str(report_path),
            "arrays": str(arrays_path),
        },
        "checkpoint": str(args.checkpoint),
        "config": {
            "eval_split": str(args.eval_split),
            "test_start": int(args.test_start),
            "val_size": int(args.val_size),
            "max_windows": int(args.max_windows),
            "decoder_test_windows": int(decoder_test_n),
        },
        "counts": {
            "support_window_count": int(n),
            "support_candidate_count": int(support_indices.size),
            "train_window_count": int(train_indices.size),
            "decoder_test_window_count": int(test_indices.size),
        },
        "calendar_end_date_range": _date_range(metadata, "calendar_end_date"),
        "train_calendar_end_date_range": _date_range(train_rows, "calendar_end_date"),
        "support_calendar_end_date_range": _date_range(metadata, "calendar_end_date"),
        "source_index_range": _int_range(metadata, "source_index"),
        "window_metadata": metadata,
    }
    _write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--decoder-test-windows", type=int, default=66)
    parser.add_argument("--state_scope", choices=["joint38"], default="joint38")
    parser.add_argument(
        "--eval_split", choices=["train", "train_tail", "val"], default="train"
    )
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=0)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--clean_nonpositive_log_levels", action="store_true", default=True)
    parser.add_argument(
        "--positive_level_policy",
        choices=["reference_based", "observed_positive"],
        default="reference_based",
    )
    parser.add_argument("--iv_transform", choices=["log_level", "bounded_logit"], default="log_level")
    parser.add_argument("--iv_lower_bound", type=float, default=1e-4)
    parser.add_argument("--iv_upper_bound", type=float, default=1.0)
    parser.add_argument("--scale_half_life", type=float, default=0.0)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--center_mode", choices=["zero", "ewma_mean"], default="zero")
    parser.add_argument("--drift_feature_mode", choices=["none", "ewma_mean"], default="none")
    args = parser.parse_args()
    report = build_support_bank(args)
    print(
        json.dumps(
            {
                "report": report["artifact_paths"]["report"],
                "arrays": report["artifact_paths"]["arrays"],
                "counts": report["counts"],
                "train_calendar_end_date_range": report[
                    "train_calendar_end_date_range"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
