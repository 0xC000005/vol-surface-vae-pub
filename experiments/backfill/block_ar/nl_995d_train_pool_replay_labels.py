#!/usr/bin/env python
"""995d: TRAIN-side within-pool replay labels for the N1 learned tilt.

Causal training-label generation for the within-pool tilt family. This adapts
the 994b oracle-within-pool replay machinery (pool construction by z-scaled
terminal-state start distance, solo ensemble replay through the frozen 734a
generator with per-query common random numbers, engine-equivalent CRPS metric
vs the realized future delta) from VAL queries to TRAIN queries with
PER-QUERY CAUSAL POOLS:

  - Query set: ``--n-queries`` (default 1,000) train windows sampled
    deterministically (seed 0) from windows 110..4009, stratified evenly: the
    window range is split into ``n_queries`` contiguous equal strata and one
    window is drawn uniformly per stratum (sorted, without replacement).
  - Candidate universe for query window ``w``: bank windows ``<= w - 30``
    ONLY (strict causality mimicking deployment). This kills the future-side
    replay-label contamination the 984a-era labels had: no candidate ever
    lies on the future side of its query. Queries with fewer than
    ``--min-eligible`` (default 60) eligible candidates are skipped and
    recorded in the manifest.
  - Pool: top-50 by the SAME z-scaled terminal-state start distance as 994b
    (z-scales = ``_safe_scale`` over the full train bank 0..4009, computed
    once -- the 994b/engine convention).
  - Label: per candidate, a solo (top-1, field_weight) ensemble replay of 16
    samples x 30 steps through the frozen 734a generator, scored with the
    engine's ``score_sample_distribution`` against the query's realized
    ``future_delta`` (z-scaled by the train-side ``delta_scale``). The
    per-query CRN seed (base 8128) is reset before EVERY candidate rollout,
    so all candidates of one query share an identical noise stream --
    byte-identical to the 994b oracle scoring path.
  - Covariates stored per candidate for downstream calm-debiasing and
    noise-banding: candidate future activity (std of the candidate's OWN
    bank future_delta in delta_scale z-units -- the F1 calm-bias covariate),
    candidate start distance, query-candidate index gap, plus per-query
    realized-future activity, pool replay-CRPS std, pool min/median start
    distance.

NOTE ON LABEL SEMANTICS: unlike 994b these labels are NOT a leakage
diagnostic of a val frame -- the "future" used as the scoring target is the
query's own realized train-side future (the supervised label the N1 tilt
trains on), and every pool candidate strictly precedes it. The candidate
universe, not the label target, is what is causal here.

Outputs (all under ``--output-dir``):
  - ``labels_shard_XXXX.npz`` incremental shards (every ``--shard-size``
    queries; partial results survive interruption; existing shards with
    matching query lists are reused on restart).
  - ``manifest_995d.json`` -- sampling spec, causality rule, CRN scheme,
    pool/metric spec, runtime, skip list, shard inventory.
  - ``validation_995d.json`` -- CRN determinism recompute checks (same seed
    => bit-identical score, and equal to the stored shard value) plus a
    full-shard causality assertion ``max(candidate) <= query - 30``.
  - ``progress_log.txt`` -- progress lines every ``--progress-every``
    queries.

No OpenAI calls. The frozen 734a checkpoint is the only model used.
No RESEARCH_LOG writes; new output directories only.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from experiments.backfill.block_ar.nl_994a_val_frame_start_only_eval import (
    BANK_TRAIN_WINDOW_COUNT,
    DEFAULT_CHECKPOINT,
    FUTURE_LEN,
    HISTORY_LEN,
    SUPPORT_BANK_DIR,
    _block_frame_namespace,
    _resolve,
    _write_json,
)
from experiments.backfill.block_ar.nl_994b_oracle_within_pool_tilt import (
    build_start_local_pool,
)
from experiments.backfill.block_ar.nl_episode_narrative_bridge_report import (
    _safe_scale,
)
from experiments.backfill.block_ar.nl_scenario_level_evaluation import (
    _future_raw_from_block,
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

DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "train_pool_replay_labels_995d"
)
SCHEMA_VERSION = "nl_995d_train_pool_replay_labels_v1"
CAUSALITY_RULE_TEXT = (
    "per-query causal candidate universe: for query window w, only bank "
    "windows c <= w - causal_gap (default 30) are eligible -- strict "
    "deployment-mimicking causality; no future-side replay-label "
    "contamination (the 984a-era labels allowed future-side candidates)"
)
CRN_SCHEME_TEXT = (
    "per-query common random numbers, identical to 994b: seed = "
    "(base_seed * 1_000_003 + query_window * 9_176 + 97) % (2**31 - 1) with "
    "base_seed 8128 (numpy + torch + cuda), reset before EVERY candidate "
    "rollout => all candidates of one query share one noise stream"
)

SHARD_ARRAY_KEYS = (
    "query_window",
    "n_eligible_candidates",
    "crn_seed",
    "pool_candidate_window",
    "replay_crps",
    "replay_energy",
    "replay_coverage_80",
    "cand_start_distance",
    "cand_index_gap",
    "cand_future_activity",
    "query_future_activity",
    "query_pool_replay_crps_std",
    "pool_min_start_distance",
    "pool_median_start_distance",
)


# --------------------------------------------------------------------------
# Pure helpers (unit-tested on synthetic arrays).
# --------------------------------------------------------------------------


def sample_train_query_windows(
    *, n_queries: int, lo: int, hi: int, seed: int
) -> np.ndarray:
    """Deterministic, evenly stratified train-query sample.

    The inclusive window range ``[lo, hi]`` is split into ``n_queries``
    contiguous strata of (near-)equal size; one window is drawn uniformly at
    random (``np.random.default_rng(seed)``) from each stratum. The result is
    strictly increasing (sorted, no replacement) by construction.
    """

    lo_i, hi_i, n_q = int(lo), int(hi), int(n_queries)
    if n_q <= 0:
        raise ValueError("n_queries must be positive")
    n_avail = hi_i - lo_i + 1
    if n_avail < n_q:
        raise ValueError(
            f"cannot sample {n_q} distinct windows from range {lo_i}..{hi_i}"
        )
    edges = lo_i + np.floor(
        np.arange(n_q + 1, dtype=np.float64) * n_avail / n_q
    ).astype(np.int64)
    rng = np.random.default_rng(int(seed))
    windows = np.asarray(
        [int(rng.integers(edges[i], edges[i + 1])) for i in range(n_q)],
        dtype=np.int64,
    )
    if windows.size != n_q or np.unique(windows).size != n_q:
        raise AssertionError("stratified sample is not unique")
    if not np.all(np.diff(windows) > 0):
        raise AssertionError("stratified sample is not strictly increasing")
    if int(windows[0]) < lo_i or int(windows[-1]) > hi_i:
        raise AssertionError("stratified sample escapes the window range")
    return windows


def causal_candidate_indices(
    query_index: int, *, causal_gap: int, universe_max: int
) -> np.ndarray:
    """Bank windows ``0..min(universe_max, query - causal_gap)`` inclusive."""

    hi = min(int(universe_max), int(query_index) - int(causal_gap))
    if hi < 0:
        return np.empty(0, dtype=np.int64)
    return np.arange(hi + 1, dtype=np.int64)


def build_causal_pool(
    *,
    query_index: int,
    terminal: np.ndarray,
    scale: np.ndarray,
    pool_size: int,
    causal_gap: int,
    universe_max: int,
    min_eligible: int,
) -> tuple[list[dict[str, Any]] | None, int]:
    """Top-``pool_size`` CAUSAL candidates by z-scaled start distance.

    Returns ``(pool_rows, n_eligible)``; ``pool_rows`` is ``None`` when the
    causal universe has fewer than ``min_eligible`` members (skip). Reuses the
    994b ``build_start_local_pool`` ranking (identical distance metric and
    tie-break) restricted to the causal universe.
    """

    candidates = causal_candidate_indices(
        int(query_index), causal_gap=int(causal_gap), universe_max=int(universe_max)
    )
    n_eligible = int(candidates.size)
    if n_eligible < int(min_eligible):
        return None, n_eligible
    rows = build_start_local_pool(
        query_index=int(query_index),
        terminal=terminal,
        scale=scale,
        train_indices=candidates,
        pool_size=int(pool_size),
        query_gap=int(causal_gap),
    )
    for row in rows:
        if int(row["window_index"]) > int(query_index) - int(causal_gap):
            raise AssertionError(
                f"causality violation in pool: candidate {row['window_index']} "
                f"> query {query_index} - gap {causal_gap}"
            )
    return rows, n_eligible


def candidate_future_activity(
    future_delta: np.ndarray, delta_scale: np.ndarray
) -> np.ndarray:
    """Per-window std of the window's OWN future_delta in delta_scale z-units.

    This is the F1 calm-bias covariate: calm candidates (low own-future
    activity) systematically win raw replay CRPS; downstream debiasing needs
    this stored per candidate.
    """

    delta = np.asarray(future_delta, dtype=np.float32)
    if delta.ndim != 3:
        raise ValueError("future_delta must have shape [N,T,C]")
    scale = np.maximum(np.asarray(delta_scale, dtype=np.float32), 1e-8)
    if scale.shape != delta.shape[1:]:
        raise ValueError("delta_scale shape must match future_delta [T,C]")
    z = delta / scale[None, :, :]
    return z.reshape(delta.shape[0], -1).std(axis=1).astype(np.float32)


def causality_max_excess(
    query_window: np.ndarray, pool_candidate_window: np.ndarray, *, causal_gap: int
) -> int:
    """max over all rows of ``candidate - (query - causal_gap)``; <= 0 is OK."""

    queries = np.asarray(query_window, dtype=np.int64).reshape(-1, 1)
    candidates = np.asarray(pool_candidate_window, dtype=np.int64)
    if candidates.shape[0] != queries.shape[0]:
        raise ValueError("query/candidate row mismatch")
    return int(np.max(candidates - (queries - int(causal_gap))))


def replay_score_candidate(
    model: Any,
    *,
    candidate_index: int,
    query_window: int,
    start_match_score: float,
    history_level: np.ndarray,
    history_norm: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    drift_feature: np.ndarray,
    history_raw: np.ndarray,
    target_delta: np.ndarray,
    specs: list[Any] | None,
    delta_scale: np.ndarray,
    samples: int,
    n_steps: int,
    chunk_size: int,
    crn_base_seed: int,
    device: torch.device,
    reconstruct_fn: Callable[..., np.ndarray] = _reconstruct_states,
) -> dict[str, Any]:
    """Solo ensemble replay score -- byte-identical path to the 994b oracle.

    Resets the per-query CRN seed, builds the engine's single-candidate
    field_weight sampling plan, rolls ``samples`` x ``n_steps`` through the
    frozen generator, reconstructs deltas, and scores against the realized
    ``target_delta``.
    """

    seed = set_common_random_seed_for_query(
        {"window_index": int(query_window)},
        base_seed=int(crn_base_seed),
        device=device,
    )
    plan = build_support_sampling_plan(
        [
            {
                "index": int(candidate_index),
                "cosine": float(start_match_score),
                "weight": 1.0,
            }
        ],
        samples_per_analogue=int(samples),
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
        n_steps=int(n_steps),
        chunk_size=int(chunk_size),
        temperature=1.0,
        device=device,
    )
    retrieved = sampled["indices"]
    states = reconstruct_fn(
        history_raw[retrieved, -1, :], sampled["increments"], specs
    )
    deltas = _states_to_deltas(states, history_raw[retrieved, -1, :])
    metrics = score_sample_distribution(deltas, target_delta, scale=delta_scale)
    coverage = metrics.get("coverage_80")
    return {
        "crn_seed": int(seed),
        "ensemble_crps_z": float(metrics["ensemble_crps_z"]),
        "energy_score_z": float(metrics["energy_score_z"]),
        "coverage_80": float(coverage) if coverage is not None else float("nan"),
    }


# --------------------------------------------------------------------------
# Shard IO.
# --------------------------------------------------------------------------


def shard_path(output_dir: Path, shard_no: int) -> Path:
    return Path(output_dir) / f"labels_shard_{int(shard_no):04d}.npz"


def try_load_existing_shard(
    path: Path, expected_queries: np.ndarray, *, pool_size: int, causal_gap: int
) -> dict[str, np.ndarray] | None:
    """Reuse a previously written shard iff its query block matches exactly."""

    if not Path(path).exists():
        return None
    try:
        with np.load(Path(path)) as data:
            arrays = {key: data[key].copy() for key in SHARD_ARRAY_KEYS}
    except Exception:
        return None
    if not np.array_equal(
        arrays["query_window"], np.asarray(expected_queries, dtype=np.int64)
    ):
        return None
    if arrays["pool_candidate_window"].shape != (
        arrays["query_window"].size,
        int(pool_size),
    ):
        return None
    if (
        causality_max_excess(
            arrays["query_window"],
            arrays["pool_candidate_window"],
            causal_gap=int(causal_gap),
        )
        > 0
    ):
        return None
    if not np.all(np.isfinite(arrays["replay_crps"])):
        return None
    return arrays


def _progress(progress_path: Path, message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with open(progress_path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


# --------------------------------------------------------------------------
# Validation.
# --------------------------------------------------------------------------


def run_validation(
    model: Any,
    *,
    shard_paths: list[Path],
    history_level: np.ndarray,
    history_norm: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    drift_feature: np.ndarray,
    history_raw: np.ndarray,
    future_delta: np.ndarray,
    specs: list[Any],
    delta_scale: np.ndarray,
    samples: int,
    n_steps: int,
    chunk_size: int,
    crn_base_seed: int,
    causal_gap: int,
    device: torch.device,
    n_checks: int = 5,
    rng_seed: int = 995,
) -> dict[str, Any]:
    """CRN determinism recompute checks + full-shard causality assertion."""

    all_queries: list[np.ndarray] = []
    all_candidates: list[np.ndarray] = []
    all_crps: list[np.ndarray] = []
    all_distance: list[np.ndarray] = []
    for path in shard_paths:
        with np.load(Path(path)) as data:
            all_queries.append(data["query_window"].astype(np.int64))
            all_candidates.append(data["pool_candidate_window"].astype(np.int64))
            all_crps.append(data["replay_crps"].astype(np.float64))
            all_distance.append(data["cand_start_distance"].astype(np.float64))
    queries = np.concatenate(all_queries)
    candidates = np.concatenate(all_candidates, axis=0)
    crps = np.concatenate(all_crps, axis=0)
    distances = np.concatenate(all_distance, axis=0)
    max_excess = causality_max_excess(queries, candidates, causal_gap=causal_gap)
    causality_ok = bool(max_excess <= 0)

    rng = np.random.default_rng(int(rng_seed))
    n_rows = int(queries.size)
    chosen_rows = rng.choice(n_rows, size=min(int(n_checks), n_rows), replace=False)
    checks: list[dict[str, Any]] = []
    with torch.no_grad():
        for row in sorted(int(r) for r in chosen_rows):
            col = int(rng.integers(0, candidates.shape[1]))
            q = int(queries[row])
            c = int(candidates[row, col])
            match_score = float(1.0 / (1.0 + max(float(distances[row, col]), 0.0)))
            kwargs = dict(
                candidate_index=c,
                query_window=q,
                start_match_score=match_score,
                history_level=history_level,
                history_norm=history_norm,
                center=center,
                scale=scale,
                drift_feature=drift_feature,
                history_raw=history_raw,
                target_delta=future_delta[q],
                specs=specs,
                delta_scale=delta_scale,
                samples=int(samples),
                n_steps=int(n_steps),
                chunk_size=int(chunk_size),
                crn_base_seed=int(crn_base_seed),
                device=device,
            )
            run_1 = replay_score_candidate(model, **kwargs)
            run_2 = replay_score_candidate(model, **kwargs)
            stored = float(crps[row, col])
            checks.append(
                {
                    "query_window": q,
                    "candidate_window": c,
                    "stored_replay_crps": stored,
                    "recompute_run1": run_1["ensemble_crps_z"],
                    "recompute_run2": run_2["ensemble_crps_z"],
                    "run1_eq_run2_bitwise": bool(
                        run_1["ensemble_crps_z"] == run_2["ensemble_crps_z"]
                    ),
                    "recompute_eq_stored_bitwise": bool(
                        run_1["ensemble_crps_z"] == stored
                    ),
                    "abs_diff_vs_stored": abs(run_1["ensemble_crps_z"] - stored),
                }
            )
    return {
        "schema_version": SCHEMA_VERSION + "_validation",
        "causality_assertion": {
            "rule": f"max(candidate) <= query - {int(causal_gap)} across all shards",
            "n_label_rows": int(queries.size * candidates.shape[1]),
            "max_candidate_minus_query_plus_gap": max_excess,
            "passed": causality_ok,
        },
        "crn_determinism_checks": checks,
        "all_checks_identical": bool(
            all(
                check["run1_eq_run2_bitwise"] and check["recompute_eq_stored_bitwise"]
                for check in checks
            )
        ),
        "rng_seed": int(rng_seed),
    }


# --------------------------------------------------------------------------
# Driver.
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--dir-suffix", default="", help="suffix for the output dir (e.g. _smoke5)"
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--n-queries", type=int, default=1000)
    parser.add_argument("--query-lo", type=int, default=110)
    parser.add_argument("--query-hi", type=int, default=4009)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="truncate the FULL deterministic plan (smoke runs keep the same "
        "leading queries as the full run)",
    )
    parser.add_argument("--pool-size", type=int, default=50)
    parser.add_argument("--causal-gap", type=int, default=30)
    parser.add_argument("--min-eligible", type=int, default=60)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--common-random-base-seed", type=int, default=8128)
    parser.add_argument("--score-scale-floor", type=float, default=1e-3)
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--validation-checks", type=int, default=5)
    parser.add_argument("--validation-seed", type=int, default=995)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    t_start = time.time()
    runtimes: dict[str, float] = {}
    output_dir = _resolve(Path(str(args.output_dir) + str(args.dir_suffix)))
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress_log.txt"
    manifest_path = output_dir / "manifest_995d.json"

    # ---- Phase 1: model + block frame (specs only) + 939a bank arrays.
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
        _block_history_level,
        _block_history_norm,
        _block_center,
        _block_scale,
        _block_drift,
        block_history_raw,
        specs,
        block,
    ) = build_val_block(block_args, payload)
    n_block_windows = int(block_history_raw.shape[0])
    n_cells = int(block_history_raw.shape[-1])
    block_future_raw = _future_raw_from_block(block, n_block_windows, n_cells)
    block_future_delta = future_delta_paths(block_history_raw, block_future_raw)

    bank_arrays_path = _resolve(SUPPORT_BANK_DIR) / "support_bank_arrays.npz"
    with np.load(bank_arrays_path) as bank:
        history_level = np.asarray(bank["history_level"], dtype=np.float32)
        history_norm = np.asarray(bank["history_norm"], dtype=np.float32)
        center = np.asarray(bank["center"], dtype=np.float32)
        scale = np.asarray(bank["scale"], dtype=np.float32)
        drift_feature = np.asarray(bank["drift_feature"], dtype=np.float32)
        history_raw = np.asarray(bank["history_raw"], dtype=np.float32)
        future_delta = np.asarray(bank["future_delta"], dtype=np.float32)
        support_indices = np.asarray(bank["support_indices"], dtype=np.int64)
    n_bank = int(history_raw.shape[0])
    if n_bank != BANK_TRAIN_WINDOW_COUNT or not np.array_equal(
        support_indices, np.arange(n_bank, dtype=np.int64)
    ):
        raise ValueError("939a bank rows are not the identity map over 0..4009")
    bank_history_max_abs_diff = float(
        np.max(np.abs(history_raw - block_history_raw[:n_bank]))
    )
    bank_future_delta_max_abs_diff = float(
        np.max(np.abs(future_delta - block_future_delta[:n_bank]))
    )
    if max(bank_history_max_abs_diff, bank_future_delta_max_abs_diff) > 1e-5:
        raise ValueError(
            "939a bank arrays diverge from the rebuilt block frame: "
            f"history {bank_history_max_abs_diff}, "
            f"future_delta {bank_future_delta_max_abs_diff}"
        )
    del (
        _block_history_level,
        _block_history_norm,
        _block_center,
        _block_scale,
        _block_drift,
        block_history_raw,
        block_future_raw,
        block_future_delta,
        block,
    )
    runtimes["setup_seconds"] = round(time.time() - t_start, 1)

    # ---- Phase 2: train-side scales + covariates (computed ONCE, 994b-style).
    train_indices = list(range(n_bank))
    delta_scale = build_delta_scale(future_delta, floor=float(args.score_scale_floor))
    terminal = history_raw[:, -1, :]
    terminal_scale = _safe_scale(terminal, train_indices)
    activity = candidate_future_activity(future_delta, delta_scale)

    # ---- Phase 3: deterministic stratified query plan + causal pools.
    plan_windows = sample_train_query_windows(
        n_queries=int(args.n_queries),
        lo=int(args.query_lo),
        hi=int(args.query_hi),
        seed=int(args.sampling_seed),
    )
    if int(args.max_queries) > 0:
        plan_windows = plan_windows[: int(args.max_queries)]
    universe_max = n_bank - 1
    labeled: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for q in plan_windows:
        pool_rows, n_eligible = build_causal_pool(
            query_index=int(q),
            terminal=terminal,
            scale=terminal_scale,
            pool_size=int(args.pool_size),
            causal_gap=int(args.causal_gap),
            universe_max=universe_max,
            min_eligible=int(args.min_eligible),
        )
        if pool_rows is None:
            skipped.append(
                {
                    "query_window": int(q),
                    "n_eligible_candidates": int(n_eligible),
                    "reason": f"fewer than {int(args.min_eligible)} causal candidates",
                }
            )
            continue
        labeled.append(
            {"query_window": int(q), "n_eligible": int(n_eligible), "pool": pool_rows}
        )
    runtimes["plan_pool_seconds"] = round(
        time.time() - t_start - runtimes["setup_seconds"], 1
    )
    _progress(
        progress_path,
        f"plan ready: {plan_windows.size} sampled queries, {len(labeled)} labeled, "
        f"{len(skipped)} skipped; pools top-{int(args.pool_size)} causal "
        f"(<= w-{int(args.causal_gap)})",
    )

    shard_size = max(1, int(args.shard_size))
    n_shards = (len(labeled) + shard_size - 1) // shard_size

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "scope_note": (
            "TRAIN-side within-pool replay labels for the N1 learned tilt "
            "(995d). Causal candidate universes; engine-equivalent solo "
            "replay CRPS labels vs realized train futures."
        ),
        "sampling": {
            "n_queries_requested": int(args.n_queries),
            "max_queries_truncation": int(args.max_queries),
            "window_range": [int(args.query_lo), int(args.query_hi)],
            "seed": int(args.sampling_seed),
            "method": (
                "even stratification: range split into n_queries contiguous "
                "equal strata, one uniform draw per stratum "
                "(np.random.default_rng(seed)); sorted, without replacement"
            ),
            "n_planned": int(plan_windows.size),
            "n_labeled": len(labeled),
            "n_skipped": len(skipped),
        },
        "causality": {
            "rule": CAUSALITY_RULE_TEXT,
            "causal_gap": int(args.causal_gap),
            "min_eligible_candidates": int(args.min_eligible),
            "candidate_universe_max_window": int(universe_max),
            "bank_vs_block_history_raw_max_abs_diff": bank_history_max_abs_diff,
            "bank_vs_block_future_delta_max_abs_diff": bank_future_delta_max_abs_diff,
        },
        "pool": {
            "pool_size": int(args.pool_size),
            "metric": (
                "z-scaled terminal-state start distance "
                "(994b build_start_local_pool, identical to "
                "_start_only_ranked_rows); z-scale = _safe_scale over the "
                "FULL train bank 0..4009, computed once (994b convention)"
            ),
        },
        "replay_label": {
            "checkpoint": str(args.checkpoint),
            "samples_per_candidate": int(args.samples),
            "n_steps": int(args.n_steps),
            "chunk_size": int(args.chunk_size),
            "crn_scheme": CRN_SCHEME_TEXT,
            "common_random_base_seed": int(args.common_random_base_seed),
            "metric": (
                "engine score_sample_distribution ensemble_crps_z (+ "
                "energy_score_z, coverage_80) vs the query's realized "
                "future_delta; delta_scale = build_delta_scale over the FULL "
                f"train bank future_delta (floor {float(args.score_scale_floor)})"
            ),
        },
        "covariates": {
            "cand_future_activity": (
                "std of the candidate's OWN bank future_delta in delta_scale "
                "z-units (F1 calm-bias covariate)"
            ),
            "cand_start_distance": "z-scaled terminal-state start distance",
            "cand_index_gap": "query_window - candidate_window (>= causal_gap)",
            "query_future_activity": (
                "std of the query's realized future_delta in delta_scale z-units"
            ),
            "query_pool_replay_crps_std": "std of replay_crps over the query's pool",
            "pool_min_start_distance": "min start distance in the pool",
            "pool_median_start_distance": "median start distance in the pool",
        },
        "shard_schema": {
            "arrays": list(SHARD_ARRAY_KEYS),
            "shapes": {
                "per_query": "(n_shard,)",
                "per_candidate": f"(n_shard, {int(args.pool_size)})",
            },
            "shard_size_queries": shard_size,
            "n_shards_expected": n_shards,
        },
        "skipped_queries": skipped,
        "query_windows_labeled": [row["query_window"] for row in labeled],
        "runtimes": runtimes,
    }
    _write_json(manifest_path, manifest)

    # ---- Phase 4: replay-label scoring with incremental shards.
    t_score = time.time()
    pool_n = int(args.pool_size)
    shard_paths: list[Path] = []
    resumed_shards: list[int] = []
    resumed_queries = 0
    fresh_queries = 0
    progress_counter = 0
    for shard_no in range(n_shards):
        rows = labeled[shard_no * shard_size : (shard_no + 1) * shard_size]
        expected_queries = np.asarray(
            [row["query_window"] for row in rows], dtype=np.int64
        )
        path = shard_path(output_dir, shard_no)
        if not bool(args.no_resume):
            existing = try_load_existing_shard(
                path,
                expected_queries,
                pool_size=pool_n,
                causal_gap=int(args.causal_gap),
            )
            if existing is not None:
                shard_paths.append(path)
                resumed_shards.append(shard_no)
                resumed_queries += int(expected_queries.size)
                _progress(
                    progress_path,
                    f"shard {shard_no:04d} resumed from disk "
                    f"({expected_queries.size} queries)",
                )
                continue
        n_s = len(rows)
        shard_arrays: dict[str, np.ndarray] = {
            "query_window": expected_queries,
            "n_eligible_candidates": np.asarray(
                [row["n_eligible"] for row in rows], dtype=np.int64
            ),
            "crn_seed": np.zeros(n_s, dtype=np.int64),
            "pool_candidate_window": np.zeros((n_s, pool_n), dtype=np.int64),
            "replay_crps": np.full((n_s, pool_n), np.nan, dtype=np.float64),
            "replay_energy": np.full((n_s, pool_n), np.nan, dtype=np.float64),
            "replay_coverage_80": np.full((n_s, pool_n), np.nan, dtype=np.float64),
            "cand_start_distance": np.full((n_s, pool_n), np.nan, dtype=np.float64),
            "cand_index_gap": np.zeros((n_s, pool_n), dtype=np.int64),
            "cand_future_activity": np.full((n_s, pool_n), np.nan, dtype=np.float32),
            "query_future_activity": np.full(n_s, np.nan, dtype=np.float32),
            "query_pool_replay_crps_std": np.full(n_s, np.nan, dtype=np.float64),
            "pool_min_start_distance": np.full(n_s, np.nan, dtype=np.float64),
            "pool_median_start_distance": np.full(n_s, np.nan, dtype=np.float64),
        }
        with torch.no_grad():
            for si, row in enumerate(rows):
                q = int(row["query_window"])
                target = future_delta[q]
                pool_rows = row["pool"]
                for ci, pool_row in enumerate(pool_rows):
                    c = int(pool_row["window_index"])
                    result = replay_score_candidate(
                        model,
                        candidate_index=c,
                        query_window=q,
                        start_match_score=float(pool_row["start_match_score"]),
                        history_level=history_level,
                        history_norm=history_norm,
                        center=center,
                        scale=scale,
                        drift_feature=drift_feature,
                        history_raw=history_raw,
                        target_delta=target,
                        specs=specs,
                        delta_scale=delta_scale,
                        samples=int(args.samples),
                        n_steps=int(args.n_steps),
                        chunk_size=int(args.chunk_size),
                        crn_base_seed=int(args.common_random_base_seed),
                        device=device,
                    )
                    shard_arrays["crn_seed"][si] = result["crn_seed"]
                    shard_arrays["pool_candidate_window"][si, ci] = c
                    shard_arrays["replay_crps"][si, ci] = result["ensemble_crps_z"]
                    shard_arrays["replay_energy"][si, ci] = result["energy_score_z"]
                    shard_arrays["replay_coverage_80"][si, ci] = result["coverage_80"]
                    shard_arrays["cand_start_distance"][si, ci] = float(
                        pool_row["start_distance"]
                    )
                    shard_arrays["cand_index_gap"][si, ci] = q - c
                    shard_arrays["cand_future_activity"][si, ci] = activity[c]
                shard_arrays["query_future_activity"][si] = activity[q]
                shard_arrays["query_pool_replay_crps_std"][si] = float(
                    np.std(shard_arrays["replay_crps"][si])
                )
                shard_arrays["pool_min_start_distance"][si] = float(
                    np.min(shard_arrays["cand_start_distance"][si])
                )
                shard_arrays["pool_median_start_distance"][si] = float(
                    np.median(shard_arrays["cand_start_distance"][si])
                )
                fresh_queries += 1
                progress_counter += 1
                queries_done = fresh_queries + resumed_queries
                if (
                    progress_counter % max(1, int(args.progress_every)) == 0
                    or queries_done >= len(labeled)
                ):
                    elapsed = time.time() - t_score
                    rate = elapsed / max(fresh_queries, 1)
                    remaining = len(labeled) - queries_done
                    eta_s = rate * max(remaining, 0)
                    _progress(
                        progress_path,
                        f"replay labels {queries_done}/{len(labeled)} queries "
                        f"({rate:.1f}s/query fresh, eta {eta_s / 3600:.2f}h)",
                    )
        if causality_max_excess(
            shard_arrays["query_window"],
            shard_arrays["pool_candidate_window"],
            causal_gap=int(args.causal_gap),
        ) > 0:
            raise AssertionError(f"causality violation in shard {shard_no}")
        if not np.all(np.isfinite(shard_arrays["replay_crps"])):
            raise ValueError(f"non-finite replay CRPS in shard {shard_no}")
        np.savez_compressed(path, **shard_arrays)
        shard_paths.append(path)
        _progress(
            progress_path,
            f"shard {shard_no:04d} written: {path.name} "
            f"({expected_queries.size} queries x {pool_n} candidates)",
        )
        manifest["shards_written"] = [str(p) for p in shard_paths]
        manifest["resumed_shards"] = resumed_shards
        _write_json(manifest_path, manifest)
    runtimes["replay_label_seconds"] = round(time.time() - t_score, 1)

    # ---- Phase 5: validation (CRN determinism + causality across shards).
    validation: dict[str, Any] | None = None
    if not bool(args.skip_validation):
        t_val = time.time()
        validation = run_validation(
            model,
            shard_paths=shard_paths,
            history_level=history_level,
            history_norm=history_norm,
            center=center,
            scale=scale,
            drift_feature=drift_feature,
            history_raw=history_raw,
            future_delta=future_delta,
            specs=specs,
            delta_scale=delta_scale,
            samples=int(args.samples),
            n_steps=int(args.n_steps),
            chunk_size=int(args.chunk_size),
            crn_base_seed=int(args.common_random_base_seed),
            causal_gap=int(args.causal_gap),
            device=device,
            n_checks=int(args.validation_checks),
            rng_seed=int(args.validation_seed),
        )
        validation_path = output_dir / "validation_995d.json"
        _write_json(validation_path, validation)
        runtimes["validation_seconds"] = round(time.time() - t_val, 1)
        _progress(
            progress_path,
            "validation written: "
            f"causality_passed={validation['causality_assertion']['passed']} "
            f"all_checks_identical={validation['all_checks_identical']}",
        )
        if not validation["causality_assertion"]["passed"]:
            raise AssertionError("validation causality assertion FAILED")
        if not validation["all_checks_identical"]:
            raise AssertionError("validation CRN determinism check FAILED")

    # ---- Phase 6: final manifest.
    runtimes["total_seconds"] = round(time.time() - t_start, 1)
    manifest["status"] = "ok"
    manifest["runtimes"] = runtimes
    manifest["shards_written"] = [str(p) for p in shard_paths]
    manifest["resumed_shards"] = resumed_shards
    manifest["fresh_queries_scored"] = fresh_queries
    manifest["rollouts_total_fresh"] = int(fresh_queries * pool_n * int(args.samples))
    if validation is not None:
        manifest["validation"] = {
            "path": str(output_dir / "validation_995d.json"),
            "causality_passed": validation["causality_assertion"]["passed"],
            "all_checks_identical": validation["all_checks_identical"],
        }
    _write_json(manifest_path, manifest)
    _progress(progress_path, f"manifest finalized: {manifest_path}")
    print(
        json.dumps(
            {
                "n_labeled_queries": len(labeled),
                "n_skipped_queries": len(skipped),
                "n_shards": len(shard_paths),
                "fresh_queries_scored": fresh_queries,
                "total_runtime_seconds": runtimes["total_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
