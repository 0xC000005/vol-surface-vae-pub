#!/usr/bin/env python
"""Build a full train-region start bridge (4010 windows, 2000-2015) for Demo 1.

Re-packages the existing ``prefix_latent_support_bank_train_all_939a`` numeric
support bank into a *bridge report + bridge arrays* so the risk-manager demo can
pick ANY train-region window (including the 2008 GFC) as the day-0 starting
market state, while the 939a bank itself stays the narrative RETRIEVAL pool.

Mechanism (traced in nl_prefix_latent_story_smoke.py):
  - ``selected_bridge_window_indices(bridge_report)`` reads ``window_indices``;
    we set it to the positional ``range(n)`` so bridge-local row i maps to
    support-bank row i (and metadata[i]).
  - ``_select_start_arrays_for_bridge_windows`` takes the support-bank branch
    whenever ``selected.max() >= validation_count`` (4009 >= 441), so all 4010
    start arrays come from the 939a bank in native order.
  - The live demo passes ``--support-bank-report/--support-bank-arrays`` = 939a,
    so start pool == retrieval pool == 939a (consistent), and
    ``explicit_start_window_index`` is the bridge-local index 0..n-1.

NO GPU / NO model load: this only copies metadata + memory_targets and writes a
zeros ``condition_vectors`` placeholder (only read on the held-out eval path,
never on the live demo path, but required by ``load_bridge_arrays``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_DIR = (
    ROOT
    / "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    / "prefix_latent_support_bank_train_all_939a"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    / "prefix_latent_full_start_bridge_train_region_939a"
)
CONDITION_DIM = 128


def build_full_start_bridge(
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, object]:
    """Build and write the full-train-region bridge report + arrays."""

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    report = json.loads((source_dir / "support_bank_report.json").read_text())
    metadata = report.get("window_metadata")
    if not isinstance(metadata, list) or not metadata:
        raise ValueError(f"{source_dir}: support_bank_report.json has no window_metadata")
    n = len(metadata)

    with np.load(source_dir / "support_bank_arrays.npz") as arrays:
        memory_targets = np.asarray(arrays["memory_targets"], dtype=np.float32)
        train_indices = np.asarray(arrays["train_indices"], dtype=np.int64)
        test_indices = np.asarray(arrays["test_indices"], dtype=np.int64)

    if memory_targets.shape[0] != n:
        raise ValueError(
            f"memory_targets rows {memory_targets.shape[0]} != metadata rows {n}"
        )
    if memory_targets.shape[1] != CONDITION_DIM:
        raise ValueError(
            f"memory_targets dim {memory_targets.shape[1]} != {CONDITION_DIM}"
        )
    if train_indices.size == 0 or test_indices.size == 0:
        raise ValueError("support bank must have non-empty train_indices/test_indices")

    # Positional window indices: bridge-local row i <-> support-bank row i.
    window_indices = list(range(n))
    first_window_id = str(metadata[0].get("window_id", "window_0"))
    # The app displays the DAY-0 date (calendar_end_date = last observed history
    # day), so report the day-0 span (Feb 2000 .. Jan 2016), not the
    # history-window START span (2000 .. 2015).
    day0_dates = [m.get("calendar_end_date") for m in metadata if isinstance(m, dict)]
    date_lo = next((d for d in day0_dates if isinstance(d, str)), None)
    date_hi = next((d for d in reversed(day0_dates) if isinstance(d, str)), None)

    bridge_report = {
        "status": "ok",
        "source_bank": str(source_dir),
        "scope_note": (
            "Full train-region start bridge derived from the 939a numeric support "
            "bank. window_indices are positional (row i == support-bank row i). "
            "Used by Demo 1 to expose every train-region window as a day-0 start "
            f"(day-0 dates {date_lo} .. {date_hi}, incl. the 2008 crisis); the 939a "
            "bank remains the narrative retrieval pool. condition_vectors are a "
            "zeros placeholder (held-out eval path only)."
        ),
        "condition_dim": CONDITION_DIM,
        "window_indices": window_indices,
        "window_metadata": metadata,
        "split": {
            "train_indices": [int(i) for i in train_indices.tolist()],
            "test_indices": [int(i) for i in test_indices.tolist()],
            "excluded_indices": [],
            "source": "prefix_latent_support_bank_train_all_939a",
        },
        "evaluation": {
            # select_cached_story_query() requires at least one role="anchor" row;
            # the live demo overrides query memory, so this is only a template.
            "heldout_examples": [
                {
                    "role": "anchor",
                    "kind": "full_start_placeholder",
                    "window_id": first_window_id,
                    "window_index": 0,
                    "embedding_index": 0,
                }
            ]
        },
        "day0_date_range": [date_lo, date_hi],
        "counts": {
            "window_count": int(n),
            "train_window_count": int(train_indices.size),
            "test_window_count": int(test_indices.size),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "full_start_bridge_report.json"
    arrays_path = output_dir / "full_start_bridge_arrays.npz"
    report_path.write_text(json.dumps(bridge_report))
    np.savez(
        arrays_path,
        condition_vectors=np.zeros((n, CONDITION_DIM), dtype=np.float32),
        memory_targets=memory_targets,
    )

    return {
        "report_path": str(report_path),
        "arrays_path": str(arrays_path),
        "window_count": int(n),
        "train_window_count": int(train_indices.size),
        "test_window_count": int(test_indices.size),
        "day0_date_range": [date_lo, date_hi],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    summary = build_full_start_bridge(args.source_dir, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
