#!/usr/bin/env python
"""994a P0: START-ONLY baseline scenario evaluation on VAL-REGION query windows.

This is the higher-power heldout harness the compass requires. The old 66-window
train-tail heldout (982g-era, windows 3944..4009, stride 1) has only ~3
independent 30-day blocks. This driver evaluates the promoted start-only
top3/90-style support-mixture machinery on the 441-window broad val frame
(``--eval_split val --test_start 4511 --val_size 441`` => global windows
4010..4450), subsampled at stride 5 (~89 queries, ~15 non-overlapping 30-day
blocks).

REUSE over rewrite:
  - Support selection reuses ``build_start_only_bridge_report`` /
    ``_start_only_ranked_rows`` from ``nl_episode_narrative_bridge_report.py``
    (the exact generator of the 982g 66q start-only bridge report): supports
    are ranked by z-scaled terminal-state distance over the CAUSAL train bank
    (windows 0..4009 == the 939a support bank), top-8, softmax(T=1.0) weights
    over -distance, temporal gap 30.
  - Scoring reuses ``nl_scenario_level_evaluation.py`` unchanged via a recorded
    subprocess CLI invocation (top_k 3, samples 16, field_weight,
    support_weight_temperature 1.0, n_steps 30, seed 4, common-random base
    seed 8128 -- identical to the 66q start-only run settings recorded in
    ``episode_card_v3_full_codex_982g_scenario_start_only_66q_s16``).

The single block frame used by the evaluator is ``--eval_split train
--test_start 4511 --val_size 0`` (windows 0..4450, block row i == global
window i), so causal train supports (<=4009) and val queries (>=4010) live in
the same rebuilt block. The bridge report's ``split.train_indices`` are the
bank windows 0..4009 only, so the score standardization (delta_scale) stays
train-side.

No OpenAI calls. The frozen 734a checkpoint is the only model used.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.evaluate_662a_state_aware_normalized_innovation_flow import (  # noqa: E402
    build_val_block,
)
from experiments.backfill.block_ar.nl_episode_narrative_bridge_report import (  # noqa: E402
    build_start_only_bridge_report,
)

DEFAULT_CHECKPOINT = (
    "models/backfill/734a_joint39_realvix_channel_level_alltrain_w005_e3_s7345/"
    "best_model.pt"
)
SUPPORT_BANK_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_bank_train_all_939a"
)
REFERENCE_66Q_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_full_codex_982g_scenario_start_only_66q_s16/"
    "scenario_level_eval_report.json"
)
ENGINE_SCRIPT = Path("experiments/backfill/block_ar/nl_scenario_level_evaluation.py")
IV_DATE_PARQUET = Path("data/spx_vol_surface_history_full_data_fixed.parquet")

# Broad-frame definition (the canonical 441-window val frame):
#   --eval_split val --test_start 4511 --val_size 441 with history/future 30/30
#   => max_train_idx = 4511 - 60 = 4451; val windows = 4010..4450 inclusive.
TEST_START = 4511
VAL_SIZE_BROAD = 441
HISTORY_LEN = 30
FUTURE_LEN = 30
VAL_FIRST_WINDOW = TEST_START - HISTORY_LEN - FUTURE_LEN - VAL_SIZE_BROAD  # 4010
VAL_LAST_WINDOW = TEST_START - HISTORY_LEN - FUTURE_LEN - 1  # 4450
BANK_TRAIN_WINDOW_COUNT = 4010  # 939a bank: windows 0..4009, all causal/train-side
DENSITY_K = 50  # mirrors nl_993a phase_a


def _resolve(path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _block_frame_namespace() -> SimpleNamespace:
    """Args for build_val_block covering windows 0..4450 in block order.

    eval_split='train' with val_size=0 yields train_indices = 0..(test_start-60-1),
    i.e. block row i == global window i for i in [0, 4450].
    """

    return SimpleNamespace(
        eval_split="train",
        test_start=TEST_START,
        val_size=0,
        max_windows=0,
        state_scope="joint38",
        iv_count=25,
        clean_nonpositive_log_levels=True,
        positive_level_policy="reference_based",
        iv_transform="log_level",
        iv_lower_bound=1e-4,
        iv_upper_bound=1.0,
        scale_half_life=0.0,
        scale_floor=1e-4,
        center_mode="zero",
        drift_feature_mode="none",
    )


def build_val_frame_bridge_report(
    *,
    history_raw: np.ndarray,
    query_indices: list[int],
    train_indices: list[int],
    bank_metadata: list[dict[str, Any]],
    top_k_bank: int,
    temporal_gap: int,
    arrays_path: str,
    n_block_windows: int,
    query_dates: dict[int, dict[str, str]],
) -> dict[str, Any]:
    """Build a start-only bridge report whose queries are val-frame windows."""

    report = build_start_only_bridge_report(
        history_raw=history_raw,
        metadata=bank_metadata,
        train_indices=train_indices,
        test_indices=query_indices,
        top_k=int(top_k_bank),
        temporal_gap=int(temporal_gap),
        cards_by_index={},
        arrays_path=str(arrays_path),
    )
    # Explicit identity local->block mapping for the full 0..4450 frame so the
    # scenario evaluator indexes the rebuilt block directly by global window.
    report["window_indices"] = list(range(int(n_block_windows)))
    report["val_frame_994a"] = {
        "frame_definition": (
            "broad 441-window val frame: eval_split=val, "
            f"test_start={TEST_START}, val_size={VAL_SIZE_BROAD} "
            f"=> global windows {VAL_FIRST_WINDOW}..{VAL_LAST_WINDOW}"
        ),
        "query_indices": [int(idx) for idx in query_indices],
        "query_stride": (
            int(query_indices[1] - query_indices[0]) if len(query_indices) > 1 else 0
        ),
        "query_dates": {
            str(idx): query_dates[idx] for idx in query_indices if idx in query_dates
        },
        "support_bank": str(SUPPORT_BANK_DIR),
        "support_bank_max_window_index": int(max(train_indices)),
        "min_query_window_index": int(min(query_indices)),
    }
    return report


def engine_command(
    *,
    bridge_report_path: Path,
    output_dir: Path,
    checkpoint: str,
    top_k: int,
    samples: int,
    seed: int,
    common_random_base_seed: int,
    chunk_size: int,
    device: str,
) -> list[str]:
    """Exact scenario-evaluator CLI mirroring the 982g 66q start-only settings."""

    return [
        sys.executable,
        str(_resolve(ENGINE_SCRIPT)),
        "--bridge-report",
        str(bridge_report_path),
        "--output-dir",
        str(output_dir),
        "--checkpoint",
        str(checkpoint),
        "--query-role",
        "anchor",
        "--top-k",
        str(int(top_k)),
        "--samples",
        str(int(samples)),
        "--support-sampling-mode",
        "field_weight",
        "--support-weight-temperature",
        "1.0",
        "--n-steps",
        "30",
        "--chunk-size",
        str(int(chunk_size)),
        "--seed",
        str(int(seed)),
        "--common-random-numbers-by-query",
        "--common-random-base-seed",
        str(int(common_random_base_seed)),
        "--device",
        str(device),
        "--eval_split",
        "train",
        "--test_start",
        str(TEST_START),
        "--val_size",
        "0",
        "--max_windows",
        "0",
    ]


def count_nonoverlapping_blocks(
    query_indices: list[int], *, future_len: int = FUTURE_LEN
) -> int:
    """Greedy count of queries whose 30-day futures do not overlap."""

    count = 0
    last = None
    for idx in sorted(int(i) for i in query_indices):
        if last is None or idx - last >= int(future_len):
            count += 1
            last = idx
    return count


def novelty_stratification(
    history_level: np.ndarray,
    *,
    train_indices: np.ndarray,
    query_indices: list[int],
) -> dict[int, dict[str, float]]:
    """Per-query start-state novelty vs the causal bank (993a phase_a pattern)."""

    start = np.asarray(history_level, dtype=np.float64)[:, -1, :]
    train = np.asarray(train_indices, dtype=np.int64)
    mu = start[train].mean(axis=0)
    sd = np.maximum(start[train].std(axis=0), 1e-12)
    z = (start - mu) / sd
    train_z = z[train]
    out: dict[int, dict[str, float]] = {}
    for q in query_indices:
        d = np.linalg.norm(train_z - z[int(q)], axis=1)
        d_sorted = np.sort(d)
        out[int(q)] = {
            "novelty_1nn": float(d_sorted[0]),
            "density_50nn": float(d_sorted[: min(DENSITY_K, d_sorted.size)].mean()),
        }
    return out


def summarize_metric(values: list[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)])
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "p10": float(np.quantile(arr, 0.10)),
        "p90": float(np.quantile(arr, 0.90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
            "val_frame_eval_994a_start_only"
        ),
    )
    parser.add_argument(
        "--bridge-output-dir",
        type=Path,
        default=None,
        help="defaults to <output-dir>_bridge",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--query-stride", type=int, default=5)
    parser.add_argument(
        "--max-queries", type=int, default=0, help="truncate query list (smoke runs)"
    )
    parser.add_argument("--top-k-bank", type=int, default=8)
    parser.add_argument("--temporal-gap", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--common-random-base-seed", type=int, default=8128)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--skip-engine",
        action="store_true",
        help="only build the bridge report and sanity inputs",
    )
    args = parser.parse_args(argv)

    t_start = time.time()
    output_dir = _resolve(args.output_dir)
    bridge_dir = (
        _resolve(args.bridge_output_dir)
        if args.bridge_output_dir is not None
        else output_dir.parent / (output_dir.name + "_bridge")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    bridge_dir.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: rebuild the full 0..4450 block frame (CPU payload is fine).
    payload = torch.load(
        _resolve(args.checkpoint), map_location="cpu", weights_only=False
    )
    if int(payload["config"]["history_len"]) != HISTORY_LEN or int(
        payload["config"]["future_len"]
    ) != FUTURE_LEN:
        raise ValueError("checkpoint history/future lengths do not match 30/30 frame")
    block_args = _block_frame_namespace()
    (
        history_level,
        _history_norm,
        _center,
        _scale,
        _drift_feature,
        history_raw,
        _specs,
        _block,
    ) = build_val_block(block_args, payload)
    n_block_windows = int(history_raw.shape[0])
    if n_block_windows != VAL_LAST_WINDOW + 1:
        raise ValueError(
            f"block frame has {n_block_windows} windows, expected {VAL_LAST_WINDOW + 1}"
        )

    # ---- Causality + bank consistency checks against the 939a support bank.
    bank_arrays_path = SUPPORT_BANK_DIR / "support_bank_arrays.npz"
    with np.load(_resolve(bank_arrays_path)) as bank:
        bank_history_raw = np.asarray(bank["history_raw"], dtype=np.float32)
        bank_support_indices = np.asarray(bank["support_indices"], dtype=np.int64)
    bank_max_index = int(bank_support_indices.max())
    bank_block_max_abs_diff = float(
        np.max(np.abs(bank_history_raw - history_raw[: bank_history_raw.shape[0]]))
    )
    bank_report = json.loads(
        (_resolve(SUPPORT_BANK_DIR) / "support_bank_report.json").read_text(
            encoding="utf-8"
        )
    )
    bank_metadata = bank_report.get("window_metadata", [])

    # ---- Query set: stride over the broad val frame.
    query_indices = list(
        range(VAL_FIRST_WINDOW, VAL_LAST_WINDOW + 1, int(args.query_stride))
    )
    if int(args.max_queries) > 0:
        query_indices = query_indices[: int(args.max_queries)]
    train_indices = list(range(BANK_TRAIN_WINDOW_COUNT))
    if bank_max_index >= min(query_indices):
        raise ValueError(
            f"causality violation: bank max window {bank_max_index} >= "
            f"min query window {min(query_indices)}"
        )

    dates = pd.DatetimeIndex(
        pd.to_datetime(pd.read_parquet(_resolve(IV_DATE_PARQUET))["date"])
    )
    query_dates = {
        int(w): {
            "history_end": str(dates[w + HISTORY_LEN - 1].date()),
            "future_start": str(dates[w + HISTORY_LEN].date()),
            "future_end": str(dates[w + HISTORY_LEN + FUTURE_LEN - 1].date()),
        }
        for w in query_indices
    }

    bridge_report = build_val_frame_bridge_report(
        history_raw=history_raw,
        query_indices=query_indices,
        train_indices=train_indices,
        bank_metadata=bank_metadata,
        top_k_bank=int(args.top_k_bank),
        temporal_gap=int(args.temporal_gap),
        arrays_path=str(bank_arrays_path),
        n_block_windows=n_block_windows,
        query_dates=query_dates,
    )
    bridge_path = bridge_dir / "start_only_bridge_report.json"
    bridge_report["artifact_paths"] = {"report": str(bridge_path)}
    _write_json(bridge_path, bridge_report)
    t_bridge = time.time()
    print(
        f"bridge report written: {bridge_path} "
        f"({len(query_indices)} queries, {t_bridge - t_start:.1f}s)",
        flush=True,
    )

    # ---- Phase 2: frozen-generator scenario evaluation (exact engine reuse).
    command = engine_command(
        bridge_report_path=bridge_path,
        output_dir=output_dir,
        checkpoint=str(args.checkpoint),
        top_k=int(args.top_k),
        samples=int(args.samples),
        seed=int(args.seed),
        common_random_base_seed=int(args.common_random_base_seed),
        chunk_size=int(args.chunk_size),
        device=str(args.device),
    )
    engine_runtime: float | None = None
    if not bool(args.skip_engine):
        t_engine = time.time()
        subprocess.run(command, check=True, cwd=str(ROOT))
        engine_runtime = time.time() - t_engine
        print(f"engine runtime: {engine_runtime:.1f}s", flush=True)

    # ---- Phase 3: sanity checks.
    sanity: dict[str, Any] = {
        "schema_version": "nl_994a_val_frame_sanity_v1",
        "engine_command": command,
        "frame": {
            "definition": (
                f"broad val frame windows {VAL_FIRST_WINDOW}..{VAL_LAST_WINDOW} "
                f"(eval_split=val, test_start={TEST_START}, val_size={VAL_SIZE_BROAD})"
            ),
            "query_stride": int(args.query_stride),
            "n_queries": len(query_indices),
            "query_indices": query_indices,
            "calendar_coverage": {
                "first_query_history_end": query_dates[query_indices[0]][
                    "history_end"
                ],
                "last_query_future_end": query_dates[query_indices[-1]]["future_end"],
            },
        },
        "causality": {
            "support_bank": str(SUPPORT_BANK_DIR),
            "bank_max_window_index": bank_max_index,
            "min_query_window_index": int(min(query_indices)),
            "bank_max_lt_min_query": bool(bank_max_index < min(query_indices)),
            "bank_vs_block_history_raw_max_abs_diff": bank_block_max_abs_diff,
            "temporal_gap": int(args.temporal_gap),
        },
        "nonoverlapping_30d_blocks": count_nonoverlapping_blocks(query_indices),
    }

    # per-query support causality (max support index <= query - temporal_gap)
    support_gap_ok = True
    max_support_seen = -1
    for row in bridge_report["evaluation"]["heldout_examples"]:
        q = int(row["window_index"])
        for item in row["top_train_pool"]:
            s = int(item["window_index"])
            max_support_seen = max(max_support_seen, s)
            if abs(q - s) < int(args.temporal_gap) or s > bank_max_index:
                support_gap_ok = False
    sanity["causality"]["per_query_support_gap_ok"] = bool(support_gap_ok)
    sanity["causality"]["max_selected_support_index"] = int(max_support_seen)

    # novelty stratification (993a phase_a pattern) + COVID-region presence
    novelty = novelty_stratification(
        history_level,
        train_indices=np.asarray(train_indices, dtype=np.int64),
        query_indices=query_indices,
    )
    nov_values = np.asarray([novelty[q]["novelty_1nn"] for q in query_indices])
    decile_cut = float(np.quantile(nov_values, 0.90))
    top_decile = [
        {
            "window_index": int(q),
            "novelty_1nn": novelty[q]["novelty_1nn"],
            "density_50nn": novelty[q]["density_50nn"],
            **query_dates[q],
        }
        for q in query_indices
        if novelty[q]["novelty_1nn"] >= decile_cut
    ]
    covid_lo, covid_hi = pd.Timestamp("2020-02-01"), pd.Timestamp("2021-12-31")
    covid_queries = [
        int(q)
        for q in query_indices
        if pd.Timestamp(query_dates[q]["future_start"]) <= covid_hi
        and pd.Timestamp(query_dates[q]["future_end"]) >= covid_lo
    ]
    sanity["novelty"] = {
        "method": (
            "z-scored start-state (history_level[:, -1, :], train mean/std over "
            "bank windows 0..4009) L2 distance to nearest causal bank window; "
            "mirrors nl_993a_hardness_novelty_stratification.py phase_a"
        ),
        "decile_cut_novelty_1nn_p90": decile_cut,
        "top_decile_queries": top_decile,
        "covid_2020_2021_queries": covid_queries,
        "covid_region_present": bool(covid_queries),
        "note": (
            "the broad val frame (windows 4010..4450) covers calendar "
            "2016-01..2017-12; the 2019-2021 COVID regime break lies in the "
            "held-out TEST region (rows >= 4511), not in this frame"
        ),
    }

    # per-window CRPS sanity vs the 66q start-only run
    if not bool(args.skip_engine):
        report = json.loads(
            (output_dir / "scenario_level_eval_report.json").read_text(
                encoding="utf-8"
            )
        )
        method_rows = [
            row["methods"]["narrative_generator_topk"] for row in report["window_scores"]
        ]
        sanity["start_only_val_frame"] = {
            metric: summarize_metric([row.get(metric) for row in method_rows])
            for metric in ("ensemble_crps_z", "energy_score_z", "coverage_80")
        }
        sanity["start_only_val_frame"]["summary_block"] = report["summary"][
            "narrative_generator_topk"
        ]
        ref_path = _resolve(REFERENCE_66Q_REPORT)
        if ref_path.exists():
            ref = json.loads(ref_path.read_text(encoding="utf-8"))
            ref_rows = [
                row["methods"]["narrative_generator_topk"]
                for row in ref["window_scores"]
            ]
            sanity["reference_66q_train_tail"] = {
                metric: summarize_metric([row.get(metric) for row in ref_rows])
                for metric in ("ensemble_crps_z", "energy_score_z", "coverage_80")
            }
        sanity["engine_runtime_seconds"] = (
            round(engine_runtime, 1) if engine_runtime is not None else None
        )

    sanity["total_runtime_seconds"] = round(time.time() - t_start, 1)
    sanity_path = output_dir / "val_frame_sanity_checks_994a.json"
    _write_json(sanity_path, sanity)
    print(f"sanity checks written: {sanity_path}", flush=True)
    print(
        json.dumps(
            {
                "n_queries": len(query_indices),
                "nonoverlapping_30d_blocks": sanity["nonoverlapping_30d_blocks"],
                "bank_max_lt_min_query": sanity["causality"]["bank_max_lt_min_query"],
                "covid_region_present": sanity["novelty"]["covid_region_present"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
