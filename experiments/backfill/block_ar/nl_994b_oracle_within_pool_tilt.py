#!/usr/bin/env python
"""994b P1: ORACLE-within-pool tilt probe on the 994a val-frame harness.

LEAKAGE BY DESIGN -- NEVER DEPLOYABLE. This probe scores support candidates by
replaying the frozen 734a generator against the query's REALIZED standardized
future delta (ensemble CRPS), i.e. it uses the future as an oracle label. Every
output JSON carries ``"leakage_only_oracle_diagnostic": true``.

Question answered (pre-registered): within the start-local top-50 pool of the
SAME z-scaled terminal-state ranking the 994a start-only bridge used, does an
oracle that re-ranks pool candidates by realized generator-replay CRPS beat the
plain start-only top-3 mixture? If the oracle CANNOT beat start-only (paired
CRPS delta CI excluding improvement), the entire within-pool tilt family
(H1/H2/H4-tilt/N1) is dead: no learnable within-pool re-weighting can recover
value that even hindsight cannot.

Probe design (mirrors the 994a P0 harness, commit 9c172bd7):
  1. Same 89 stride-5 val-frame query windows (4010..4450), same causal bank
     (windows 0..4009 == the 939a support bank), same frame rebuild
     (eval_split=train, test_start=4511, val_size=0 => block row i == window i).
  2. POOL: per query, top-50 bank windows by the SAME z-scaled terminal-state
     distance as ``_start_only_ranked_rows`` (query-gap >= 30 vs the query, no
     mutual gap inside the pool -- the pool may be dense).
  3. ORACLE replay score: per candidate, an exact engine-equivalent top-1
     mixture rollout -- reset the engine's per-query common-random-number seed,
     build the engine's field_weight sampling plan for the single candidate,
     roll 16 samples x 30 steps through ``sample_normal_generator_for_
     retrieved_analogues``, reconstruct deltas, and score with the engine's
     ``score_sample_distribution`` against the query's realized future delta.
     The per-candidate seed reset makes every candidate of one query use common
     random numbers (identical noise stream), exactly what 50 separate top-1
     engine runs would produce.
  4. ORACLE selection: top-3 by replay CRPS subject to a >= 30 MUTUAL index
     gap; weights = softmax(-CRPS / T) with T recorded (default 0.2, auto-fall
     back over a small grid if weights degenerate to uniform/one-hot).
  5. The oracle bridge report uses the start-only bridge schema so the
     UNCHANGED engine command from 994a consumes it.
  6. Comparison vs the 994a start-only baseline via the paired moving-block
     bootstrap (L=6 stride-aware and L=30 conservative, 10k resamples), plus
     secondary diagnostics (per-query Spearman oracle-vs-locality order, mean
     |support - query| index distance oracle vs start-only top-3).

No OpenAI calls. No existing artifact is modified; all outputs live in new
``val_frame_oracle_pool_tilt_994b*`` directories.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.backfill.block_ar.nl_994a_val_frame_start_only_eval import (
    BANK_TRAIN_WINDOW_COUNT,
    DEFAULT_CHECKPOINT,
    FUTURE_LEN,
    HISTORY_LEN,
    SUPPORT_BANK_DIR,
    TEST_START,
    VAL_FIRST_WINDOW,
    VAL_LAST_WINDOW,
    VAL_SIZE_BROAD,
    _block_frame_namespace,
    _resolve,
    _write_json,
    count_nonoverlapping_blocks,
    engine_command,
    summarize_metric,
)
from experiments.backfill.block_ar.nl_994a_paired_block_bootstrap import (
    load_method_scores,
    paired_deltas,
    run_paired_block_bootstrap,
)
from experiments.backfill.block_ar.nl_episode_narrative_bridge_report import (
    _safe_scale,
    _start_only_ranked_rows,
    _temporal_gap_filter,
)
from experiments.backfill.block_ar.nl_episode_narrative_retrieval import (
    pairwise_jaccard_summary,
)
from experiments.backfill.block_ar.nl_scenario_level_evaluation import (
    _future_raw_from_block,
    _softmax,
    _states_to_deltas,
    build_delta_scale,
    build_support_sampling_plan,
    future_delta_paths,
    score_sample_distribution,
    set_common_random_seed_for_query,
)
from experiments.backfill.block_ar.nl_narrative_grounded_scenario_pipeline import (
    _reconstruct_states,
    sample_normal_generator_for_retrieved_analogues,
)
from experiments.backfill.block_ar.evaluate_662a_state_aware_normalized_innovation_flow import (
    build_val_block,
)
from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (
    load_model,
)

ROOT = Path(__file__).resolve().parents[3]

LEAKAGE_KEY = "leakage_only_oracle_diagnostic"
DEFAULT_BASELINE_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "val_frame_eval_994a_start_only/scenario_level_eval_report.json"
)
DEFAULT_BASELINE_BRIDGE = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "val_frame_eval_994a_start_only_bridge/start_only_bridge_report.json"
)
DEFAULT_OUTPUT_BASE = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "val_frame_oracle_pool_tilt_994b"
)
KILL_CONDITION_TEXT = (
    "if oracle-within-pool tilt fails to beat start-only (paired CRPS delta CI "
    "excluding improvement), the entire within-pool tilt family "
    "(H1/H2/H4-tilt/N1) is dead"
)
WEIGHT_TEMPERATURE_GRID = (0.5, 0.2, 0.1, 0.05, 0.02)
MAX_WEIGHT_BAND = (0.40, 0.92)  # top-3 uniform => 1/3; one-hot => 1.0


# --------------------------------------------------------------------------
# Pure pool / selection / weighting logic (unit-tested on synthetic arrays).
# --------------------------------------------------------------------------


def build_start_local_pool(
    *,
    query_index: int,
    terminal: np.ndarray,
    scale: np.ndarray,
    train_indices: np.ndarray,
    pool_size: int,
    query_gap: int,
) -> list[dict[str, Any]]:
    """Top-``pool_size`` causal candidates by z-scaled terminal-state distance.

    Identical metric and tie-break to ``_start_only_ranked_rows``: distance =
    sqrt(nanmean(((terminal[c] - terminal[q]) / scale)^2)); candidates with
    ``|c - q| < query_gap`` (or ``c == q``) are excluded; NO mutual gap is
    enforced inside the pool (the pool may be dense).
    """

    term = np.asarray(terminal, dtype=np.float32)
    scale_arr = np.asarray(scale, dtype=np.float32)
    train = np.asarray(train_indices, dtype=np.int64)
    q = int(query_index)
    keep = (train != q) & (np.abs(train - q) >= int(query_gap))
    candidates = train[keep]
    diff = (term[candidates] - term[q][None, :]) / scale_arr[None, :]
    distances = np.sqrt(np.nanmean(np.square(diff), axis=1)).astype(np.float64)
    order = np.lexsort((candidates, distances))
    top = order[: max(0, int(pool_size))]
    rows: list[dict[str, Any]] = []
    for rank, pos in enumerate(top, 1):
        distance = float(distances[pos])
        rows.append(
            {
                "window_index": int(candidates[pos]),
                "start_distance": distance,
                "start_match_score": float(1.0 / (1.0 + max(distance, 0.0))),
                "locality_rank": int(rank),
            }
        )
    return rows


def oracle_select_top_k(
    pool_rows: list[dict[str, Any]],
    *,
    top_k: int,
    mutual_gap: int,
    score_key: str = "oracle_ensemble_crps_z",
) -> list[dict[str, Any]]:
    """Greedy lowest-CRPS selection subject to a mutual index gap.

    Candidates are ranked ascending by oracle replay CRPS (ties broken by
    window index) and selected greedily; a candidate within ``mutual_gap`` of
    an already-selected support is skipped. Reuses the bridge module's
    ``_temporal_gap_filter`` for the gap logic.
    """

    ranked = sorted(
        (dict(row) for row in pool_rows),
        key=lambda row: (float(row[score_key]), int(row["window_index"])),
    )
    return _temporal_gap_filter(ranked, top_k=int(top_k), temporal_gap=int(mutual_gap))


def softmax_oracle_weights(
    crps_values: list[float] | np.ndarray,
    *,
    temperature: float,
) -> np.ndarray:
    """softmax(-CRPS / T) using the engine's softmax helper."""

    values = -np.asarray(crps_values, dtype=np.float64).reshape(-1)
    return _softmax(values, temperature=float(temperature))


def weight_shape_stats(weights: np.ndarray) -> dict[str, float]:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    return {
        "max_weight": float(np.max(w)),
        "min_weight": float(np.min(w)),
        "effective_n": float(1.0 / np.sum(np.square(w))),
    }


def choose_weight_temperature(
    selected_crps_per_query: list[list[float]],
    *,
    default_temperature: float = 0.2,
    grid: tuple[float, ...] = WEIGHT_TEMPERATURE_GRID,
    max_weight_band: tuple[float, float] = MAX_WEIGHT_BAND,
) -> tuple[float, dict[str, Any]]:
    """Pick T so oracle weights are neither uniform nor one-hot.

    Keeps the pre-registered default (0.2) when the mean per-query max weight
    falls inside ``max_weight_band``; otherwise picks the grid temperature
    whose mean max weight is closest to the band midpoint. Returns the chosen
    temperature plus the full grid diagnostic (recorded in the probe report).
    """

    lo, hi = float(max_weight_band[0]), float(max_weight_band[1])
    mid = 0.5 * (lo + hi)
    temps = sorted(set(float(t) for t in grid) | {float(default_temperature)})
    grid_stats: dict[str, dict[str, float]] = {}
    for temp in temps:
        max_weights: list[float] = []
        effective_ns: list[float] = []
        for crps in selected_crps_per_query:
            stats = weight_shape_stats(
                softmax_oracle_weights(crps, temperature=temp)
            )
            max_weights.append(stats["max_weight"])
            effective_ns.append(stats["effective_n"])
        grid_stats[f"{temp:g}"] = {
            "mean_max_weight": float(np.mean(max_weights)),
            "median_max_weight": float(np.median(max_weights)),
            "mean_effective_n": float(np.mean(effective_ns)),
        }
    default_mean_max = grid_stats[f"{float(default_temperature):g}"][
        "mean_max_weight"
    ]
    if lo <= default_mean_max <= hi:
        chosen = float(default_temperature)
        reason = (
            f"pre-registered default T={default_temperature:g} keeps mean max "
            f"weight {default_mean_max:.3f} inside the non-degenerate band "
            f"[{lo:g}, {hi:g}]"
        )
    else:
        chosen = min(
            temps,
            key=lambda t: abs(grid_stats[f"{t:g}"]["mean_max_weight"] - mid),
        )
        reason = (
            f"default T={default_temperature:g} gave degenerate mean max weight "
            f"{default_mean_max:.3f} (band [{lo:g}, {hi:g}]); chose grid "
            f"T={chosen:g} closest to band midpoint"
        )
    diagnostics = {
        "default_temperature": float(default_temperature),
        "grid": [float(t) for t in temps],
        "max_weight_band": [lo, hi],
        "grid_stats": grid_stats,
        "chosen_temperature": float(chosen),
        "reason": reason,
    }
    return float(chosen), diagnostics


def build_oracle_bridge_report(
    *,
    selections: list[dict[str, Any]],
    train_indices: list[int],
    query_indices: list[int],
    n_block_windows: int,
    arrays_path: str,
    pool_size: int,
    query_gap: int,
    mutual_gap: int,
    weight_temperature: float,
    crn_base_seed: int,
    replay_samples: int,
) -> dict[str, Any]:
    """Oracle-tilt bridge report in the start-only schema the engine consumes.

    ``selections`` rows: {"window_index": q, "supports": [{"window_index",
    "start_distance", "start_match_score", "locality_rank",
    "oracle_ensemble_crps_z", "oracle_energy_score_z", "weight"}, ...]}.
    """

    heldout_rows: list[dict[str, Any]] = []
    result_sets: dict[str, list[str]] = {}
    for query_no, selection in enumerate(selections):
        q = int(selection["window_index"])
        top_train_pool: list[dict[str, Any]] = []
        for rank, support in enumerate(selection["supports"], 1):
            idx = int(support["window_index"])
            crps = float(support["oracle_ensemble_crps_z"])
            top_train_pool.append(
                {
                    "rank": int(rank),
                    "window_index": idx,
                    "window_id": f"window_{idx:04d}",
                    "cosine": float(support["start_match_score"]),
                    "weight": float(support["weight"]),
                    "retrieval_score": float(-crps),
                    "scenario_title": (
                        "oracle within-pool tilt support (leakage diagnostic)"
                    ),
                    "score_components": {
                        "method": "oracle_within_pool_replay_tilt",
                        "start_distance": float(support["start_distance"]),
                        "start_match_score": float(support["start_match_score"]),
                        "locality_rank": int(support["locality_rank"]),
                        "oracle_replay_ensemble_crps_z": crps,
                        "oracle_replay_energy_score_z": float(
                            support["oracle_energy_score_z"]
                        ),
                        "oracle_weight_temperature": float(weight_temperature),
                    },
                }
            )
        window_id = f"window_{q:04d}"
        heldout_rows.append(
            {
                "query_id": f"oracle_pool_tilt_{query_no:04d}_{q}",
                "window_index": q,
                "window_id": window_id,
                "role": "anchor",
                "kind": "oracle_within_pool_replay_tilt_retrieval",
                "query_text_source": "none_oracle_replay_within_start_local_pool",
                "top_train_pool": top_train_pool,
            }
        )
        result_sets[window_id] = [
            str(item["window_id"]) for item in top_train_pool
        ]
    return {
        "schema_version": "nl_994b_oracle_within_pool_tilt_bridge_v1",
        "status": "ok",
        LEAKAGE_KEY: True,
        "scope_note": (
            "ORACLE LEAKAGE DIAGNOSTIC -- NEVER DEPLOYABLE. Supports are the "
            "top-3 of the start-local top-"
            f"{int(pool_size)} pool re-ranked by realized-future generator "
            "replay CRPS (hindsight label). Used only to upper-bound the "
            "within-pool tilt family on the 994a val frame."
        ),
        "cards_path": "",
        "arrays_path": str(arrays_path),
        "retrieval_config": {
            "method": "oracle_within_pool_replay_tilt",
            "pool_method": "start_only_terminal_state",
            "pool_size": int(pool_size),
            "query_gap": int(query_gap),
            "mutual_gap_final_selection": int(mutual_gap),
            "top_k": 3,
            "temporal_gap": int(mutual_gap),
            "oracle_label": "ensemble_crps_z_of_top1_replay_vs_realized_future",
            "oracle_replay_samples": int(replay_samples),
            "oracle_weight_temperature": float(weight_temperature),
            "common_random_base_seed": int(crn_base_seed),
            "support_pool": "support_train_only",
            "query_pool": "support_decoder_test",
        },
        "split": {
            "train_indices": [int(idx) for idx in train_indices],
            "test_indices": [int(idx) for idx in query_indices],
        },
        "window_indices": list(range(int(n_block_windows))),
        "evaluation": {
            "heldout_examples": heldout_rows,
            "support_overlap": pairwise_jaccard_summary(result_sets),
        },
    }


def kill_condition_assessment(
    crps_bootstrap_by_block_length: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Mechanical reading of the pre-registered kill condition.

    delta = oracle - start_only per shared window (lower CRPS better):
      - improvement ESTABLISHED at L: ci_high < 0
      - improvement EXCLUDED at L:    ci_low >= 0 (CI contains no improvement)
    """

    per_length: dict[str, dict[str, Any]] = {}
    for label, summary in crps_bootstrap_by_block_length.items():
        per_length[label] = {
            "mean_delta": float(summary["mean_delta"]),
            "ci_low": float(summary["ci_low"]),
            "ci_high": float(summary["ci_high"]),
            "improvement_established_ci_high_lt_0": bool(
                float(summary["ci_high"]) < 0.0
            ),
            "improvement_excluded_ci_low_ge_0": bool(
                float(summary["ci_low"]) >= 0.0
            ),
        }
    established_all = all(
        row["improvement_established_ci_high_lt_0"] for row in per_length.values()
    )
    excluded_all = all(
        row["improvement_excluded_ci_low_ge_0"] for row in per_length.values()
    )
    if established_all:
        verdict = "oracle_beats_start_only_tilt_family_alive"
    elif excluded_all:
        verdict = "improvement_excluded_tilt_family_dead"
    else:
        verdict = "no_established_improvement_inconclusive_or_dead_by_strict_reading"
    return {
        "pre_registered_kill_condition": KILL_CONDITION_TEXT,
        "metric": "ensemble_crps_z",
        "delta_orientation": "oracle_tilt - start_only (negative = oracle better)",
        "per_block_length": per_length,
        "improvement_established_all_block_lengths": bool(established_all),
        "improvement_excluded_all_block_lengths": bool(excluded_all),
        "verdict": verdict,
    }


def annotate_engine_report_with_leakage_flag(report_path: Path) -> None:
    """Mark the engine-emitted report as an oracle leakage diagnostic.

    The engine command is reused UNCHANGED, so its report does not carry the
    leakage marker by itself; every JSON in the 994b output dirs must carry it.
    """

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    report[LEAKAGE_KEY] = True
    report["oracle_note"] = (
        "ORACLE LEAKAGE DIAGNOSTIC: the consumed bridge report selected and "
        "weighted supports using realized-future generator-replay CRPS. "
        "Never deployable."
    )
    _write_json(Path(report_path), report)


def spearman_rho(values_a: np.ndarray, values_b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    rho = spearmanr(np.asarray(values_a), np.asarray(values_b)).statistic
    return float(rho)


def _win_rates(
    baseline_report: Path,
    oracle_report: Path,
    *,
    method: str = "narrative_generator_topk",
) -> dict[str, Any]:
    scores_a = load_method_scores(baseline_report, method=method)
    scores_b = load_method_scores(oracle_report, method=method)
    out: dict[str, Any] = {}
    for metric in ("ensemble_crps_z", "energy_score_z"):
        deltas, shared = paired_deltas(scores_a, scores_b, metric=metric)
        out[metric] = {
            "n_windows": int(deltas.size),
            "oracle_win_rate_delta_lt_0": float(np.mean(deltas < 0.0)),
            "mean_delta": float(deltas.mean()),
        }
    shared_cov = sorted(
        idx
        for idx in set(scores_a) & set(scores_b)
        if "coverage_80" in scores_a[idx] and "coverage_80" in scores_b[idx]
    )
    cov_a = np.asarray([scores_a[i]["coverage_80"] for i in shared_cov])
    cov_b = np.asarray([scores_b[i]["coverage_80"] for i in shared_cov])
    out["coverage_80"] = {
        "n_windows": int(len(shared_cov)),
        "oracle_win_rate_closer_to_0p80": float(
            np.mean(np.abs(cov_b - 0.80) < np.abs(cov_a - 0.80))
        ),
        "mean_delta": float(np.mean(cov_b - cov_a)),
        "baseline_mean": float(np.mean(cov_a)),
        "oracle_mean": float(np.mean(cov_b)),
    }
    return out


# --------------------------------------------------------------------------
# Probe driver.
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-base", type=Path, default=Path(DEFAULT_OUTPUT_BASE))
    parser.add_argument(
        "--dir-suffix",
        default="",
        help="suffix for output dirs (e.g. _smoke5); keeps the 994b prefix",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--baseline-report", type=Path, default=Path(DEFAULT_BASELINE_REPORT))
    parser.add_argument(
        "--baseline-bridge-report", type=Path, default=Path(DEFAULT_BASELINE_BRIDGE)
    )
    parser.add_argument("--query-stride", type=int, default=5)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--pool-size", type=int, default=50)
    parser.add_argument("--query-gap", type=int, default=30)
    parser.add_argument("--mutual-gap", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--common-random-base-seed", type=int, default=8128)
    parser.add_argument("--oracle-weight-temperature", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-scale-floor", type=float, default=1e-3)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=994)
    parser.add_argument(
        "--reuse-oracle-scores",
        type=Path,
        default=None,
        help="reuse a previously written oracle_replay_scores npz",
    )
    parser.add_argument("--skip-engine", action="store_true")
    args = parser.parse_args(argv)

    t_start = time.time()
    runtimes: dict[str, float] = {}
    output_dir = _resolve(Path(str(args.output_base) + str(args.dir_suffix)))
    bridge_dir = _resolve(
        Path(str(args.output_base) + str(args.dir_suffix) + "_bridge")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    bridge_dir.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: frame rebuild + device model (identical frame to 994a).
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    model, payload = load_model(_resolve(args.checkpoint), device)
    if int(payload["config"]["history_len"]) != HISTORY_LEN or int(
        payload["config"]["future_len"]
    ) != FUTURE_LEN:
        raise ValueError("checkpoint history/future lengths do not match 30/30 frame")
    block_args = _block_frame_namespace()
    (
        history_level,
        history_norm,
        center,
        scale,
        drift_feature,
        history_raw,
        specs,
        block,
    ) = build_val_block(block_args, payload)
    n_block_windows = int(history_raw.shape[0])
    if n_block_windows != VAL_LAST_WINDOW + 1:
        raise ValueError(
            f"block frame has {n_block_windows} windows, expected {VAL_LAST_WINDOW + 1}"
        )
    n_cells = int(history_raw.shape[-1])
    future_raw = _future_raw_from_block(block, n_block_windows, n_cells)
    future_delta = future_delta_paths(history_raw, future_raw)

    train_indices = list(range(BANK_TRAIN_WINDOW_COUNT))
    query_indices = list(
        range(VAL_FIRST_WINDOW, VAL_LAST_WINDOW + 1, int(args.query_stride))
    )
    if int(args.max_queries) > 0:
        query_indices = query_indices[: int(args.max_queries)]
    if max(train_indices) >= min(query_indices):
        raise ValueError("causality violation: bank window >= min query window")

    # delta_scale exactly as the engine builds it (train-side, identity map)
    train_delta = future_delta[np.asarray(train_indices, dtype=np.int64)]
    delta_scale = build_delta_scale(train_delta, floor=float(args.score_scale_floor))

    # consistency vs the 939a bank (bank rows cover windows 0..4009)
    bank_arrays_path = _resolve(SUPPORT_BANK_DIR) / "support_bank_arrays.npz"
    with np.load(bank_arrays_path) as bank:
        bank_future_delta = np.asarray(bank["future_delta"], dtype=np.float32)
        bank_support_indices = np.asarray(bank["support_indices"], dtype=np.int64)
    bank_max_index = int(bank_support_indices.max())
    bank_future_delta_max_abs_diff = float(
        np.max(np.abs(bank_future_delta - future_delta[: bank_future_delta.shape[0]]))
    )
    runtimes["frame_build_seconds"] = round(time.time() - t_start, 1)

    # ---- Phase 2: start-local top-N pools (same metric as the 994a bridge).
    t_pool = time.time()
    terminal = np.asarray(history_raw, dtype=np.float32)[:, -1, :]
    terminal_scale = _safe_scale(terminal, train_indices)
    train_arr = np.asarray(train_indices, dtype=np.int64)
    pools: list[list[dict[str, Any]]] = [
        build_start_local_pool(
            query_index=q,
            terminal=terminal,
            scale=terminal_scale,
            train_indices=train_arr,
            pool_size=int(args.pool_size),
            query_gap=int(args.query_gap),
        )
        for q in query_indices
    ]
    # cross-check pool head against the exact 994a ranking function (top-8)
    ref_rows = _start_only_ranked_rows(
        query_index=int(query_indices[0]),
        history_raw=history_raw,
        train_indices=train_indices,
        top_k=8,
        temporal_gap=int(args.query_gap),
        cards_by_index={},
    )
    pool_head = [row["window_index"] for row in pools[0][:8]]
    ref_head = [int(row["window_index"]) for row in ref_rows]
    if pool_head != ref_head:
        raise ValueError(
            f"pool ranking diverges from _start_only_ranked_rows: "
            f"{pool_head} vs {ref_head}"
        )
    runtimes["pool_build_seconds"] = round(time.time() - t_pool, 1)
    print(
        f"pools built: {len(pools)} queries x {int(args.pool_size)} candidates "
        f"({runtimes['pool_build_seconds']}s; head matches 994a ranking)",
        flush=True,
    )

    # ---- Phase 3: oracle ensemble replay scoring (LEAKAGE: realized future).
    t_oracle = time.time()
    n_q = len(query_indices)
    pool_n = int(args.pool_size)
    scores_path = bridge_dir / "oracle_replay_scores_994b.npz"
    if args.reuse_oracle_scores is not None:
        with np.load(_resolve(args.reuse_oracle_scores)) as payload_npz:
            if not np.array_equal(
                payload_npz["query_indices"],
                np.asarray(query_indices, dtype=np.int64),
            ):
                raise ValueError("reused oracle scores query_indices mismatch")
            pool_indices = payload_npz["pool_indices"].copy()
            pool_distances = payload_npz["pool_distances"].copy()
            oracle_crps = payload_npz["oracle_crps"].copy()
            oracle_energy = payload_npz["oracle_energy"].copy()
        for qi in range(n_q):
            if not np.array_equal(
                pool_indices[qi],
                np.asarray([row["window_index"] for row in pools[qi]], dtype=np.int64),
            ):
                raise ValueError("reused oracle scores pool mismatch")
        print(f"reusing oracle scores from {args.reuse_oracle_scores}", flush=True)
    else:
        pool_indices = np.zeros((n_q, pool_n), dtype=np.int64)
        pool_distances = np.zeros((n_q, pool_n), dtype=np.float64)
        oracle_crps = np.full((n_q, pool_n), np.nan, dtype=np.float64)
        oracle_energy = np.full((n_q, pool_n), np.nan, dtype=np.float64)
        with torch.no_grad():
            for qi, q in enumerate(query_indices):
                target = future_delta[int(q)]
                for ci, row in enumerate(pools[qi]):
                    c = int(row["window_index"])
                    pool_indices[qi, ci] = c
                    pool_distances[qi, ci] = float(row["start_distance"])
                    # exact top-1 engine-run equivalence: per-query CRN seed
                    # reset before EVERY candidate => common random numbers
                    # across the whole pool of this query.
                    set_common_random_seed_for_query(
                        {"window_index": int(q)},
                        base_seed=int(args.common_random_base_seed),
                        device=device,
                    )
                    plan = build_support_sampling_plan(
                        [
                            {
                                "index": c,
                                "cosine": float(row["start_match_score"]),
                                "weight": 1.0,
                            }
                        ],
                        samples_per_analogue=int(args.samples),
                        mode="field_weight",
                        weight_temperature=1.0,
                    )
                    sampled = sample_normal_generator_for_retrieved_analogues(
                        model,
                        plan["analogue_rows"],
                        history_level,
                        history_norm,
                        center,
                        scale,
                        drift_feature,
                        n_samples=int(plan["samples_per_row"]),
                        n_steps=int(args.n_steps),
                        chunk_size=int(args.chunk_size),
                        temperature=1.0,
                        device=device,
                    )
                    retrieved = sampled["indices"]
                    states = _reconstruct_states(
                        history_raw[retrieved, -1, :],
                        sampled["increments"],
                        specs,
                    )
                    deltas = _states_to_deltas(states, history_raw[retrieved, -1, :])
                    metrics = score_sample_distribution(
                        deltas, target, scale=delta_scale
                    )
                    oracle_crps[qi, ci] = float(metrics["ensemble_crps_z"])
                    oracle_energy[qi, ci] = float(metrics["energy_score_z"])
                if (qi + 1) % 5 == 0 or qi + 1 == n_q:
                    elapsed = time.time() - t_oracle
                    eta = elapsed / (qi + 1) * (n_q - qi - 1)
                    print(
                        f"  oracle replay {qi + 1}/{n_q} queries "
                        f"({elapsed:.0f}s elapsed, eta {eta:.0f}s)",
                        flush=True,
                    )
        np.savez_compressed(
            scores_path,
            query_indices=np.asarray(query_indices, dtype=np.int64),
            pool_indices=pool_indices,
            pool_distances=pool_distances,
            oracle_crps=oracle_crps,
            oracle_energy=oracle_energy,
        )
        print(f"oracle replay scores written: {scores_path}", flush=True)
    if not np.all(np.isfinite(oracle_crps)):
        raise ValueError("non-finite oracle replay CRPS encountered")
    runtimes["oracle_replay_seconds"] = round(time.time() - t_oracle, 1)

    # ---- Phase 4: oracle selection (mutual gap) + weight temperature.
    for qi in range(n_q):
        for ci, row in enumerate(pools[qi]):
            row["oracle_ensemble_crps_z"] = float(oracle_crps[qi, ci])
            row["oracle_energy_score_z"] = float(oracle_energy[qi, ci])
    selected_rows: list[list[dict[str, Any]]] = [
        oracle_select_top_k(
            pools[qi], top_k=int(args.top_k), mutual_gap=int(args.mutual_gap)
        )
        for qi in range(n_q)
    ]
    selected_crps = [
        [float(row["oracle_ensemble_crps_z"]) for row in rows]
        for rows in selected_rows
    ]
    chosen_temperature, temperature_diagnostics = choose_weight_temperature(
        selected_crps,
        default_temperature=float(args.oracle_weight_temperature),
    )
    selections: list[dict[str, Any]] = []
    weight_stats_rows: list[dict[str, float]] = []
    for qi, q in enumerate(query_indices):
        weights = softmax_oracle_weights(
            selected_crps[qi], temperature=chosen_temperature
        )
        weight_stats_rows.append(weight_shape_stats(weights))
        supports = []
        for row, weight in zip(selected_rows[qi], weights, strict=True):
            support = dict(row)
            support["weight"] = float(weight)
            supports.append(support)
        selections.append({"window_index": int(q), "supports": supports})

    # ---- Phase 5: oracle bridge report + UNCHANGED 994a engine command.
    bridge_report = build_oracle_bridge_report(
        selections=selections,
        train_indices=train_indices,
        query_indices=[int(q) for q in query_indices],
        n_block_windows=n_block_windows,
        arrays_path=str(bank_arrays_path),
        pool_size=pool_n,
        query_gap=int(args.query_gap),
        mutual_gap=int(args.mutual_gap),
        weight_temperature=chosen_temperature,
        crn_base_seed=int(args.common_random_base_seed),
        replay_samples=int(args.samples),
    )
    bridge_path = bridge_dir / "oracle_within_pool_tilt_bridge_report.json"
    bridge_report["artifact_paths"] = {
        "report": str(bridge_path),
        "oracle_replay_scores": str(scores_path),
    }
    _write_json(bridge_path, bridge_report)
    print(f"oracle bridge report written: {bridge_path}", flush=True)

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
        runtimes["engine_seconds"] = round(engine_runtime, 1)
        print(f"engine runtime: {engine_runtime:.1f}s", flush=True)
        annotate_engine_report_with_leakage_flag(
            output_dir / "scenario_level_eval_report.json"
        )

    # ---- Phase 6: paired block bootstrap vs the 994a start-only baseline.
    oracle_report_path = output_dir / "scenario_level_eval_report.json"
    baseline_report_path = _resolve(args.baseline_report)
    bootstrap_payloads: dict[str, dict[str, Any]] = {}
    win_rates: dict[str, Any] | None = None
    if not bool(args.skip_engine):
        t_boot = time.time()
        for block_length in (6, 30):
            payload_boot = run_paired_block_bootstrap(
                report_a=baseline_report_path,
                report_b=oracle_report_path,
                method_a="narrative_generator_topk",
                method_b="narrative_generator_topk",
                metrics=("ensemble_crps_z", "energy_score_z", "coverage_80"),
                block_length=int(block_length),
                n_boot=int(args.n_boot),
                seed=int(args.bootstrap_seed),
            )
            payload_boot[LEAKAGE_KEY] = True
            payload_boot["comparison"] = (
                "report_a = 994a start-only baseline, report_b = 994b "
                "oracle-within-pool tilt (leakage diagnostic)"
            )
            label = f"L{block_length}"
            bootstrap_payloads[label] = payload_boot
            boot_path = (
                output_dir
                / f"paired_block_bootstrap_oracle_tilt_vs_start_only_{label}.json"
            )
            _write_json(boot_path, payload_boot)
            print(f"bootstrap written: {boot_path}", flush=True)
        win_rates = _win_rates(baseline_report_path, oracle_report_path)
        runtimes["bootstrap_seconds"] = round(time.time() - t_boot, 1)

    # ---- Phase 7: secondary diagnostics (oracle order vs locality order).
    spearman_values = np.asarray(
        [
            spearman_rho(pool_distances[qi], oracle_crps[qi])
            for qi in range(n_q)
        ],
        dtype=np.float64,
    )
    baseline_bridge = json.loads(
        _resolve(args.baseline_bridge_report).read_text(encoding="utf-8")
    )
    baseline_top3: dict[int, list[int]] = {
        int(row["window_index"]): [
            int(item["window_index"]) for item in row["top_train_pool"][:3]
        ]
        for row in baseline_bridge["evaluation"]["heldout_examples"]
    }
    oracle_abs_gaps: list[float] = []
    start_abs_gaps: list[float] = []
    overlap_counts: list[int] = []
    oracle_locality_ranks: list[float] = []
    pool_contains_baseline: list[float] = []
    for qi, q in enumerate(query_indices):
        oracle_idx = [int(row["window_index"]) for row in selected_rows[qi]]
        base_idx = baseline_top3.get(int(q), [])
        oracle_abs_gaps.append(float(np.mean([abs(i - int(q)) for i in oracle_idx])))
        if base_idx:
            start_abs_gaps.append(float(np.mean([abs(i - int(q)) for i in base_idx])))
            overlap_counts.append(len(set(oracle_idx) & set(base_idx)))
            pool_set = set(int(v) for v in pool_indices[qi])
            pool_contains_baseline.append(
                float(np.mean([1.0 if i in pool_set else 0.0 for i in base_idx]))
            )
        oracle_locality_ranks.extend(
            float(row["locality_rank"]) for row in selected_rows[qi]
        )
    secondary = {
        "spearman_oracle_crps_vs_start_distance_within_pool": {
            "orientation": (
                "rank corr of (start_distance asc, oracle_crps asc) over the "
                "pool; +1 = oracle order == locality order, 0 = unrelated"
            ),
            "mean": float(np.mean(spearman_values)),
            "median": float(np.median(spearman_values)),
            "p10": float(np.quantile(spearman_values, 0.10)),
            "p90": float(np.quantile(spearman_values, 0.90)),
            "min": float(np.min(spearman_values)),
            "max": float(np.max(spearman_values)),
            "frac_above_0p3": float(np.mean(spearman_values > 0.3)),
            "per_query": [round(float(v), 4) for v in spearman_values],
        },
        "support_index_distance_to_query": {
            "oracle_selected_top3_mean_abs_gap": float(np.mean(oracle_abs_gaps)),
            "start_only_selected_top3_mean_abs_gap": (
                float(np.mean(start_abs_gaps)) if start_abs_gaps else None
            ),
            "note": "mean |support_window_index - query_window_index|",
        },
        "oracle_vs_start_only_top3_overlap": {
            "mean_intersection_count": (
                float(np.mean(overlap_counts)) if overlap_counts else None
            ),
            "frac_queries_zero_overlap": (
                float(np.mean(np.asarray(overlap_counts) == 0))
                if overlap_counts
                else None
            ),
        },
        "oracle_selected_locality_rank_within_pool": {
            "mean": float(np.mean(oracle_locality_ranks)),
            "median": float(np.median(oracle_locality_ranks)),
            "max": float(np.max(oracle_locality_ranks)),
            "note": "1 = nearest-by-start; large = oracle reaches deep into pool",
        },
        "baseline_top3_contained_in_pool_frac": (
            float(np.mean(pool_contains_baseline)) if pool_contains_baseline else None
        ),
    }

    # ---- Phase 8: pool stats + probe report + verdict.
    per_query_min_dist = pool_distances.min(axis=1)
    per_query_max_dist = pool_distances.max(axis=1)
    pool_abs_gap = np.abs(
        pool_indices - np.asarray(query_indices, dtype=np.int64)[:, None]
    )
    dense_member_frac = []
    for qi in range(n_q):
        idx = np.sort(pool_indices[qi])
        neighbor = np.minimum(
            np.diff(idx, prepend=idx[0] - 10**9),
            np.diff(idx, append=idx[-1] + 10**9),
        )
        dense_member_frac.append(float(np.mean(neighbor < int(args.mutual_gap))))
    pool_stats = {
        "n_queries": n_q,
        "pool_size": pool_n,
        "query_gap_vs_query": int(args.query_gap),
        "mutual_gap_inside_pool_enforced": False,
        "mutual_gap_final_selection": int(args.mutual_gap),
        "candidate_universe": f"causal bank windows 0..{bank_max_index}",
        "start_distance": {
            "pool_min_mean": float(np.mean(per_query_min_dist)),
            "pool_max_mean": float(np.mean(per_query_max_dist)),
            "pool_mean_mean": float(np.mean(pool_distances)),
            "pool_max_over_min_mean": float(
                np.mean(per_query_max_dist / np.maximum(per_query_min_dist, 1e-12))
            ),
        },
        "pool_member_abs_index_gap_to_query": {
            "mean": float(np.mean(pool_abs_gap)),
            "median": float(np.median(pool_abs_gap)),
            "min": int(np.min(pool_abs_gap)),
        },
        "pool_density_frac_members_within_mutual_gap_of_another": float(
            np.mean(dense_member_frac)
        ),
        "nonoverlapping_30d_blocks": count_nonoverlapping_blocks(
            [int(q) for q in query_indices]
        ),
    }

    oracle_summary_rows: dict[str, Any] | None = None
    if not bool(args.skip_engine):
        oracle_report = json.loads(oracle_report_path.read_text(encoding="utf-8"))
        method_rows = [
            row["methods"]["narrative_generator_topk"]
            for row in oracle_report["window_scores"]
        ]
        oracle_summary_rows = {
            metric: summarize_metric([row.get(metric) for row in method_rows])
            for metric in ("ensemble_crps_z", "energy_score_z", "coverage_80")
        }
        oracle_summary_rows["summary_block"] = oracle_report["summary"][
            "narrative_generator_topk"
        ]

    verdict = (
        kill_condition_assessment(
            {
                label: payload_boot["metrics"]["ensemble_crps_z"]
                for label, payload_boot in bootstrap_payloads.items()
            }
        )
        if bootstrap_payloads
        else None
    )

    runtimes["total_seconds"] = round(time.time() - t_start, 1)
    probe_report: dict[str, Any] = {
        "schema_version": "nl_994b_oracle_within_pool_tilt_probe_v1",
        LEAKAGE_KEY: True,
        "scope_note": (
            "ORACLE LEAKAGE DIAGNOSTIC -- uses realized futures to score pool "
            "candidates. Diagnostic upper bound for the within-pool tilt "
            "family on the 994a val frame. NEVER deployable, NEVER promotable."
        ),
        "frame": {
            "definition": (
                f"broad val frame windows {VAL_FIRST_WINDOW}..{VAL_LAST_WINDOW} "
                f"(eval_split=val, test_start={TEST_START}, "
                f"val_size={VAL_SIZE_BROAD}); query stride {int(args.query_stride)}"
            ),
            "query_indices": [int(q) for q in query_indices],
        },
        "causality": {
            "bank_max_window_index": bank_max_index,
            "min_query_window_index": int(min(query_indices)),
            "bank_max_lt_min_query": bool(bank_max_index < min(query_indices)),
            "bank_vs_block_future_delta_max_abs_diff": bank_future_delta_max_abs_diff,
            "oracle_label_is_causal": False,
            "oracle_label_note": (
                "oracle replay CRPS conditions on the query's REALIZED future "
                "delta -- leakage by design"
            ),
        },
        "oracle_replay": {
            "samples_per_candidate": int(args.samples),
            "n_steps": int(args.n_steps),
            "chunk_size": int(args.chunk_size),
            "common_random_numbers": (
                "per-query engine seed (base "
                f"{int(args.common_random_base_seed)}) reset before EVERY "
                "candidate rollout => identical noise across a query's pool"
            ),
            "rollouts_total": int(n_q * pool_n * int(args.samples)),
            "oracle_crps_pool_mean": float(np.mean(oracle_crps)),
            "oracle_crps_pool_best_mean": float(np.mean(oracle_crps.min(axis=1))),
            "oracle_crps_selected_top3_mean": float(
                np.mean([np.mean(c) for c in selected_crps])
            ),
        },
        "weighting": {
            "temperature_used": float(chosen_temperature),
            "weight_stats_at_temperature_used": {
                "mean_max_weight": float(
                    np.mean([row["max_weight"] for row in weight_stats_rows])
                ),
                "median_max_weight": float(
                    np.median([row["max_weight"] for row in weight_stats_rows])
                ),
                "mean_effective_n": float(
                    np.mean([row["effective_n"] for row in weight_stats_rows])
                ),
            },
            "temperature_diagnostics": temperature_diagnostics,
        },
        "pool_stats": pool_stats,
        "engine_command": command,
        "runtimes": runtimes,
        "baseline": {
            "report": str(baseline_report_path),
            "bridge_report": str(_resolve(args.baseline_bridge_report)),
        },
        "oracle_tilt_val_frame": oracle_summary_rows,
        "paired_block_bootstrap": {
            label: payload_boot["metrics"]
            for label, payload_boot in bootstrap_payloads.items()
        },
        "win_rates": win_rates,
        "secondary_diagnostics": secondary,
        "kill_condition_assessment": verdict,
        "artifact_paths": {
            "bridge_report": str(bridge_path),
            "oracle_replay_scores": str(scores_path),
            "engine_report": str(oracle_report_path),
            "bootstrap_reports": [
                str(
                    output_dir
                    / f"paired_block_bootstrap_oracle_tilt_vs_start_only_{label}.json"
                )
                for label in bootstrap_payloads
            ],
        },
    }
    probe_path = output_dir / "oracle_within_pool_tilt_probe_report_994b.json"
    _write_json(probe_path, probe_report)
    print(f"probe report written: {probe_path}", flush=True)
    print(
        json.dumps(
            {
                "n_queries": n_q,
                "pool_size": pool_n,
                "temperature_used": chosen_temperature,
                "oracle_crps_selected_top3_mean": probe_report["oracle_replay"][
                    "oracle_crps_selected_top3_mean"
                ],
                "verdict": verdict["verdict"] if verdict else "engine_skipped",
                "total_runtime_seconds": runtimes["total_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
