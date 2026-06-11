#!/usr/bin/env python
"""997a: SET-level teacher labels for the within-pool tilt family.

The 996b solo-replay-taught tilt was killed; the owner approved a SET-level
teacher whose semantics match the DEPLOYED top3/90 mixture. This adapts the
995d solo-label machinery (same 1,000-query deterministic plan, same top-50
causal pools with candidates <= query-30, same CRN scheme base 8128, same
engine metric path) from solo candidate labels to candidate-SET labels:

  - Query plan: IDENTICAL to 995d -- ``sample_train_query_windows(1000, 110,
    4009, seed=0)``; asserted equal to the 995d manifest's
    ``query_windows_labeled``.
  - Pools: rebuilt with the 995d ``build_causal_pool`` (top-50 by z-scaled
    terminal-state start distance over the causal universe ``<= query-30``)
    and asserted byte-equal to the pools stored in the 995d shards.
  - Per query, 40 candidate SETS of 3 supports (all sets respect a >= 30
    MUTUAL index gap within the set; per-query deterministic rng
    ``np.random.default_rng([set_sampling_seed=0, query_window])``):
      *  1 locality set         -- start-only top-3 (gap-respecting greedy
                                   over the locality order) -- the anchor;
      *  1 solo-oracle set      -- top-3 by 995d solo replay_crps
                                   (gap-respecting greedy);
      * 14 solo-informed sets   -- members sampled w/o replacement with
                                   p ~ softmax(-z(solo_crps)/T); T tuned ONCE
                                   over the full 1,000-query solo labels so
                                   the per-query sampling effective sample
                                   size is diverse (target mean ESS ~16/50;
                                   grid + choice recorded in the manifest);
      *  6 diversity-spread sets stratified across the pool's start-distance
                                   terciles (one member per tercile);
      *  6 diversity-spread sets spread in prefix-feature space (k-means++-
                                   style greedy max-spread over z-scaled
                                   terminal states);
      * 12 uniform random sets.
    Identical sets are DEDUPED (a sampled duplicate is resampled up to a
    retry cap; deterministic duplicates are dropped); composition +
    provenance category are recorded per set.

  - SET score: mixture rollout EXACTLY as deployed. WITHIN-SET WEIGHTING
    DECISION (checked, as instructed): the 994b oracle bridge weighted its
    top-3 by softmax(-replay_CRPS/T) with auto-chosen T=0.02 (near one-hot)
    -- an oracle/leakage-only convention. The DEPLOYED top3/90 convention
    (982g/994a start-only bridge ``_start_only_ranked_rows`` -> engine
    ``field_weight``) is weights = softmax(-start_distance / T=1.0) over the
    selected supports. We MATCH THE DEPLOYED CONVENTION: it is computable at
    deployment time and is byte-identical to what the engine would do with
    any candidate set it is handed. Rollout: ``build_support_sampling_plan``
    over the 3 supports with ``mode=field_weight``,
    ``samples_per_analogue=16`` (the engine ``--samples 16`` semantics:
    total = 3 x 16 = 48 paths allocated proportional to the weights),
    16 samples-per-analogue x 30 steps through the frozen 734a generator;
    per-query CRN seed (base 8128, 995d scheme) reset before EVERY set
    rollout so set scores are comparable across sets within a query; scored
    with the engine's ``score_sample_distribution`` (ensemble CRPS + energy
    + coverage_80) vs the query's realized future_delta.

TEACHER-QUALITY GATE (computed at completion, incl. smoke):
  (a) per-query std of set CRPS across the sets vs a CRN twin-noise floor
      (5 random sets per query re-scored with a SECOND CRN stream, base seed
      8129, on a 50-query subsample; sigma_noise = rms(delta)/sqrt(2));
  (b) best-sampled-set tilt = mean over queries of (best set CRPS - locality
      set CRPS) -- the sampled-set mini-oracle headroom; compared in
      magnitude/sign to the 994b full-pool oracle direction (mean delta
      -0.0264, oracle beats start-only);
  (c) per-query rank correlation between set CRPS and the SUM of member solo
      CRPS (995d) -- how much set value is NOT sum-decomposable (the
      diversification signal); full distribution reported.

Outputs (all under ``--output-dir``):
  - ``set_labels_shard_XXXX.npz`` incremental shards (every ``--shard-size``
    queries; resume-safe: existing shards with matching query/set plans are
    reused on restart).
  - ``manifest_997a.json`` / ``quality_gate_997a.json`` /
    ``twin_rescore_997a.npz`` / ``validation_997a.json`` /
    ``progress_log.txt``.

No OpenAI calls. The frozen 734a checkpoint is the only model used.
No RESEARCH_LOG writes; new output directories only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy.stats import spearmanr

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
from experiments.backfill.block_ar.nl_995d_train_pool_replay_labels import (
    CRN_SCHEME_TEXT,
    _progress,
    build_causal_pool,
    sample_train_query_windows,
)
from experiments.backfill.block_ar.nl_episode_narrative_bridge_report import (
    _safe_scale,
    _softmax_weights,
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

DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "set_level_teacher_labels_997a"
)
DEFAULT_SOLO_LABELS_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "train_pool_replay_labels_995d"
)
DEFAULT_994B_PROBE_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "val_frame_oracle_pool_tilt_994b/oracle_within_pool_tilt_probe_report_994b.json"
)
SCHEMA_VERSION = "nl_997a_set_level_teacher_labels_v1"

DEPLOYED_WEIGHTING_TEXT = (
    "within-set weights = softmax(-start_distance / T=1.0) over the 3 set "
    "members (the DEPLOYED top3/90 convention: _start_only_ranked_rows "
    "computes _softmax_weights over -start_distance at temperature 1.0 and "
    "the engine consumes them via build_support_sampling_plan "
    "mode=field_weight, weight_temperature=1.0, samples_per_analogue=16 => "
    "3 x 16 = 48 paths allocated proportional to the weights). DECISION "
    "RECORD: the 994b oracle bridge instead used softmax(-replay_CRPS/T) "
    "with auto-chosen T=0.02 (near one-hot) -- an oracle/leakage-only "
    "convention that is NOT computable at deployment; the deployed "
    "convention is matched here so the teacher labels score exactly what "
    "the deployed engine would produce for each candidate set."
)

N_MEMBERS = 3
CATEGORY_ORDER = (
    "locality",
    "solo_oracle",
    "solo_informed",
    "diversity_stratified",
    "diversity_feature_spread",
    "uniform_random",
)
CATEGORY_CODE = {name: code for code, name in enumerate(CATEGORY_ORDER)}

SET_SHARD_ARRAY_KEYS = (
    "query_window",
    "n_eligible_candidates",
    "crn_seed",
    "n_valid_sets",
    "n_dropped_set_slots",
    "set_valid",
    "set_category",
    "set_member_window",
    "set_member_start_distance",
    "set_member_solo_crps",
    "set_weight",
    "set_sample_count",
    "set_crps",
    "set_energy",
    "set_coverage_80",
    "set_sum_solo_crps",
    "set_weighted_solo_crps",
    "locality_set_crps",
    "best_set_crps",
    "set_crps_std",
    "spearman_set_vs_sum_solo",
)

TEMPERATURE_GRID = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)


# --------------------------------------------------------------------------
# Pure set-construction helpers (unit-tested on synthetic arrays).
# --------------------------------------------------------------------------


def deployed_set_weights(start_distances: list[float] | np.ndarray) -> np.ndarray:
    """Deployed top3/90 weights: softmax(-start_distance / T=1.0)."""

    scores = [-float(d) for d in np.asarray(start_distances, dtype=np.float64)]
    return np.asarray(_softmax_weights(scores, temperature=1.0), dtype=np.float64)


def greedy_gap_select(
    order: np.ndarray | list[int],
    windows: np.ndarray,
    *,
    mutual_gap: int,
    n_members: int = N_MEMBERS,
) -> list[int] | None:
    """Greedy selection along ``order`` subject to a mutual index gap.

    ``order`` holds pool positions ranked best-first; returns the first
    ``n_members`` positions whose windows are pairwise >= ``mutual_gap``
    apart, or ``None`` when the pool cannot supply a full set.
    """

    win = np.asarray(windows, dtype=np.int64)
    chosen: list[int] = []
    for pos in (int(p) for p in order):
        w = int(win[pos])
        if any(abs(w - int(win[c])) < int(mutual_gap) for c in chosen):
            continue
        chosen.append(pos)
        if len(chosen) >= int(n_members):
            return sorted(chosen)
    return None


def sample_gap_respecting_set(
    rng: np.random.Generator,
    windows: np.ndarray,
    probs: np.ndarray,
    *,
    mutual_gap: int,
    n_members: int = N_MEMBERS,
    max_attempts: int = 50,
) -> list[int] | None:
    """Sample ``n_members`` pool positions w/o replacement, gap-respecting.

    Sequential masking: after each draw, every candidate within
    ``mutual_gap`` of the drawn window (including itself) is masked out and
    the probabilities are renormalized. Retries the whole set when the mask
    empties early.
    """

    win = np.asarray(windows, dtype=np.int64)
    base = np.asarray(probs, dtype=np.float64).reshape(-1)
    if base.size != win.size:
        raise ValueError("probs and windows must align")
    if np.any(base < 0.0) or float(base.sum()) <= 0.0:
        raise ValueError("probs must be non-negative with positive sum")
    n = int(win.size)
    for _ in range(int(max_attempts)):
        avail = np.ones(n, dtype=bool)
        chosen: list[int] = []
        for _ in range(int(n_members)):
            masked = base * avail
            total = float(masked.sum())
            if total <= 0.0:
                break
            pick = int(rng.choice(n, p=masked / total))
            chosen.append(pick)
            avail &= np.abs(win - win[pick]) >= int(mutual_gap)
        if len(chosen) == int(n_members):
            return sorted(chosen)
    return None


def sample_stratified_set(
    rng: np.random.Generator,
    windows: np.ndarray,
    *,
    mutual_gap: int,
    n_members: int = N_MEMBERS,
    max_attempts: int = 50,
) -> list[int] | None:
    """One member per start-distance tercile of the locality-ordered pool.

    The pool is stored in locality order (start distance ascending), so
    contiguous position terciles == start-distance strata.
    """

    win = np.asarray(windows, dtype=np.int64)
    n = int(win.size)
    edges = np.floor(np.linspace(0, n, int(n_members) + 1)).astype(np.int64)
    for _ in range(int(max_attempts)):
        chosen: list[int] = []
        complete = True
        for stratum in range(int(n_members)):
            eligible = [
                pos
                for pos in range(int(edges[stratum]), int(edges[stratum + 1]))
                if all(
                    abs(int(win[pos]) - int(win[c])) >= int(mutual_gap)
                    for c in chosen
                )
            ]
            if not eligible:
                complete = False
                break
            chosen.append(int(rng.choice(np.asarray(eligible, dtype=np.int64))))
        if complete:
            return sorted(chosen)
    return None


def sample_feature_spread_set(
    rng: np.random.Generator,
    windows: np.ndarray,
    features: np.ndarray,
    *,
    mutual_gap: int,
    n_members: int = N_MEMBERS,
    max_attempts: int = 50,
) -> list[int] | None:
    """k-means++-style greedy spread in prefix-feature space, gap-respecting.

    First member uniform; each next member is drawn with probability
    proportional to its minimum squared z-scaled terminal-feature distance
    to the already-chosen members, restricted to gap-eligible candidates.
    """

    win = np.asarray(windows, dtype=np.int64)
    feat = np.asarray(features, dtype=np.float64)
    n = int(win.size)
    if feat.shape[0] != n:
        raise ValueError("features and windows must align")
    for _ in range(int(max_attempts)):
        chosen = [int(rng.integers(0, n))]
        complete = True
        for _ in range(int(n_members) - 1):
            avail = np.ones(n, dtype=bool)
            for c in chosen:
                avail &= np.abs(win - win[c]) >= int(mutual_gap)
            if not bool(avail.any()):
                complete = False
                break
            d2 = np.full(n, np.inf, dtype=np.float64)
            for c in chosen:
                diff = feat - feat[c][None, :]
                d2 = np.minimum(d2, np.nanmean(np.square(diff), axis=1))
            weights = np.where(avail, np.nan_to_num(d2, nan=0.0, posinf=0.0), 0.0)
            total = float(weights.sum())
            if total <= 0.0:
                weights = avail.astype(np.float64)
                total = float(weights.sum())
            chosen.append(int(rng.choice(n, p=weights / total)))
        if complete:
            return sorted(chosen)
    return None


def solo_informed_probabilities(
    solo_crps: np.ndarray, *, temperature: float
) -> np.ndarray:
    """p ~ softmax(-z / T) with per-query standardized solo replay CRPS.

    Standardizing makes the temperature scale-free across queries (raw CRPS
    levels vary strongly between calm and turbulent queries).
    """

    crps = np.asarray(solo_crps, dtype=np.float64).reshape(-1)
    std = float(crps.std())
    z = (crps - float(crps.mean())) / std if std > 1e-12 else np.zeros_like(crps)
    return _softmax(-z, temperature=float(temperature))


def tune_solo_informed_temperature(
    solo_by_query: list[np.ndarray],
    *,
    grid: tuple[float, ...] = TEMPERATURE_GRID,
    target_ess: float = 16.0,
) -> tuple[float, dict[str, Any]]:
    """Pick T so solo-informed sampling stays diverse (recorded).

    Diversity is measured by the per-query effective sample size of the
    sampling distribution, ESS = 1 / sum(p^2) (uniform over 50 -> 50;
    one-hot -> 1). The grid temperature whose MEAN ESS is closest to
    ``target_ess`` is chosen; the full grid diagnostic is returned for the
    manifest.
    """

    grid_stats: dict[str, dict[str, float]] = {}
    for temp in grid:
        ess = []
        for crps in solo_by_query:
            p = solo_informed_probabilities(crps, temperature=float(temp))
            ess.append(1.0 / float(np.sum(np.square(p))))
        ess_arr = np.asarray(ess, dtype=np.float64)
        grid_stats[f"{float(temp):g}"] = {
            "mean_ess": float(ess_arr.mean()),
            "median_ess": float(np.median(ess_arr)),
            "min_ess": float(ess_arr.min()),
        }
    chosen = min(
        (float(t) for t in grid),
        key=lambda t: abs(grid_stats[f"{t:g}"]["mean_ess"] - float(target_ess)),
    )
    diagnostics = {
        "criterion": (
            "per-query sampling ESS = 1/sum(p^2); chose grid T whose mean "
            f"ESS is closest to target {float(target_ess):g} (of pool 50) "
            "so solo-informed sampling is neither uniform nor one-hot"
        ),
        "grid": [float(t) for t in grid],
        "target_ess": float(target_ess),
        "grid_stats": grid_stats,
        "chosen_temperature": float(chosen),
        "chosen_mean_ess": grid_stats[f"{chosen:g}"]["mean_ess"],
    }
    return float(chosen), diagnostics


def build_query_sets(
    rng: np.random.Generator,
    *,
    windows: np.ndarray,
    solo_crps: np.ndarray,
    features: np.ndarray,
    informed_probs: np.ndarray,
    mutual_gap: int,
    n_solo_informed: int = 14,
    n_div_strat: int = 6,
    n_div_feature: int = 6,
    n_uniform: int = 12,
    n_members: int = N_MEMBERS,
    max_attempts: int = 50,
) -> tuple[list[dict[str, Any]], int]:
    """Build the per-query candidate-set plan (deduped, gap-respecting).

    Returns ``(sets, n_dropped)``. Each set row: ``{"members": [pool
    positions, ascending], "category": str}``. ``n_dropped`` counts target
    slots lost to dedup-exhaustion or sampling failure. Returns ``([], 0)``
    when even the locality set cannot be built (query unusable).
    """

    win = np.asarray(windows, dtype=np.int64)
    n = int(win.size)
    sets: list[dict[str, Any]] = []
    seen: set[frozenset[int]] = set()
    dropped = 0

    def _key(members: list[int]) -> frozenset[int]:
        return frozenset(int(win[p]) for p in members)

    def _accept(members: list[int], category: str) -> None:
        seen.add(_key(members))
        sets.append(
            {"members": sorted(int(p) for p in members), "category": category}
        )

    # 1 locality set: start-only top-3 (pool is stored in locality order).
    locality = greedy_gap_select(
        np.arange(n), win, mutual_gap=int(mutual_gap), n_members=int(n_members)
    )
    if locality is None:
        return [], 0
    _accept(locality, "locality")

    # 1 solo-oracle set: top-3 by 995d solo replay CRPS (ties by window).
    solo_order = np.lexsort((win, np.asarray(solo_crps, dtype=np.float64)))
    solo_oracle = greedy_gap_select(
        solo_order, win, mutual_gap=int(mutual_gap), n_members=int(n_members)
    )
    if solo_oracle is not None and _key(solo_oracle) not in seen:
        _accept(solo_oracle, "solo_oracle")
    else:
        dropped += 1

    uniform_probs = np.full(n, 1.0 / n, dtype=np.float64)
    samplers: list[tuple[str, int, Callable[[], list[int] | None]]] = [
        (
            "solo_informed",
            int(n_solo_informed),
            lambda: sample_gap_respecting_set(
                rng,
                win,
                informed_probs,
                mutual_gap=int(mutual_gap),
                n_members=int(n_members),
                max_attempts=int(max_attempts),
            ),
        ),
        (
            "diversity_stratified",
            int(n_div_strat),
            lambda: sample_stratified_set(
                rng,
                win,
                mutual_gap=int(mutual_gap),
                n_members=int(n_members),
                max_attempts=int(max_attempts),
            ),
        ),
        (
            "diversity_feature_spread",
            int(n_div_feature),
            lambda: sample_feature_spread_set(
                rng,
                win,
                features,
                mutual_gap=int(mutual_gap),
                n_members=int(n_members),
                max_attempts=int(max_attempts),
            ),
        ),
        (
            "uniform_random",
            int(n_uniform),
            lambda: sample_gap_respecting_set(
                rng,
                win,
                uniform_probs,
                mutual_gap=int(mutual_gap),
                n_members=int(n_members),
                max_attempts=int(max_attempts),
            ),
        ),
    ]
    for category, target, sampler in samplers:
        for _ in range(int(target)):
            placed = False
            for _ in range(int(max_attempts)):
                members = sampler()
                if members is None:
                    break  # sampler exhausted its own retries
                if _key(members) in seen:
                    continue  # dedup: resample
                _accept(members, category)
                placed = True
                break
            if not placed:
                dropped += 1
    return sets, dropped


def set_causality_max_excess(
    query_window: np.ndarray,
    member_window: np.ndarray,
    valid: np.ndarray,
    *,
    causal_gap: int,
) -> int:
    """max over valid members of ``member - (query - causal_gap)``; <=0 OK."""

    queries = np.asarray(query_window, dtype=np.int64)
    members = np.asarray(member_window, dtype=np.int64)
    mask = np.asarray(valid, dtype=bool)[:, :, None] & (members >= 0)
    if not bool(mask.any()):
        return -(1 << 30)
    excess = members - (queries[:, None, None] - int(causal_gap))
    return int(excess[mask].max())


def set_mutual_gap_min(member_window: np.ndarray, valid: np.ndarray) -> int:
    """min pairwise |member_i - member_j| over all valid sets; large if none."""

    members = np.asarray(member_window, dtype=np.int64)
    mask = np.asarray(valid, dtype=bool)
    best = 1 << 30
    n_members = members.shape[-1]
    for i in range(n_members):
        for j in range(i + 1, n_members):
            pair_ok = mask & (members[:, :, i] >= 0) & (members[:, :, j] >= 0)
            if bool(pair_ok.any()):
                gaps = np.abs(members[:, :, i] - members[:, :, j])[pair_ok]
                best = min(best, int(gaps.min()))
    return best


def spearman_rho(values_a: np.ndarray, values_b: np.ndarray) -> float:
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.size < 3:
        return float("nan")
    return float(spearmanr(a, b).statistic)


# --------------------------------------------------------------------------
# Set replay scoring (engine-equivalent deployed mixture path).
# --------------------------------------------------------------------------


def replay_score_set(
    model: Any,
    *,
    member_windows: list[int],
    member_match_scores: list[float],
    member_weights: list[float],
    query_window: int,
    history_level: np.ndarray,
    history_norm: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    drift_feature: np.ndarray,
    history_raw: np.ndarray,
    target_delta: np.ndarray,
    specs: list[Any] | None,
    delta_scale: np.ndarray,
    samples_per_analogue: int,
    n_steps: int,
    chunk_size: int,
    crn_base_seed: int,
    device: torch.device,
    reconstruct_fn: Callable[..., np.ndarray] = _reconstruct_states,
) -> dict[str, Any]:
    """Deployed-equivalent mixture rollout score for one candidate set.

    Resets the per-query CRN seed (995d scheme), builds the engine's
    ``field_weight`` sampling plan over the 3 supports with the deployed
    weights, rolls samples_per_analogue x n_members paths through the frozen
    generator and scores against the realized ``target_delta``.
    """

    seed = set_common_random_seed_for_query(
        {"window_index": int(query_window)},
        base_seed=int(crn_base_seed),
        device=device,
    )
    rows = [
        {"index": int(w), "cosine": float(m), "weight": float(wt)}
        for w, m, wt in zip(
            member_windows, member_match_scores, member_weights, strict=True
        )
    ]
    plan = build_support_sampling_plan(
        rows,
        samples_per_analogue=int(samples_per_analogue),
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
        "sample_counts": [int(c) for c in plan["sample_counts"]],
        "n_paths_total": int(len(plan["analogue_rows"])),
    }


# --------------------------------------------------------------------------
# Shard IO (resume-safe; the set plan is deterministic and re-derived).
# --------------------------------------------------------------------------


def set_shard_path(output_dir: Path, shard_no: int) -> Path:
    return Path(output_dir) / f"set_labels_shard_{int(shard_no):04d}.npz"


def try_load_existing_set_shard(
    path: Path,
    *,
    expected_query_window: np.ndarray,
    expected_set_valid: np.ndarray,
    expected_set_category: np.ndarray,
    expected_set_member_window: np.ndarray,
    causal_gap: int,
    mutual_gap: int,
) -> dict[str, np.ndarray] | None:
    """Reuse a shard iff its query block AND deterministic set plan match."""

    if not Path(path).exists():
        return None
    try:
        with np.load(Path(path)) as data:
            arrays = {key: data[key].copy() for key in SET_SHARD_ARRAY_KEYS}
    except Exception:
        return None
    if not np.array_equal(
        arrays["query_window"], np.asarray(expected_query_window, dtype=np.int64)
    ):
        return None
    if not np.array_equal(
        arrays["set_valid"], np.asarray(expected_set_valid, dtype=bool)
    ):
        return None
    if not np.array_equal(
        arrays["set_category"], np.asarray(expected_set_category, dtype=np.int8)
    ):
        return None
    if not np.array_equal(
        arrays["set_member_window"],
        np.asarray(expected_set_member_window, dtype=np.int64),
    ):
        return None
    if (
        set_causality_max_excess(
            arrays["query_window"],
            arrays["set_member_window"],
            arrays["set_valid"],
            causal_gap=int(causal_gap),
        )
        > 0
    ):
        return None
    if set_mutual_gap_min(arrays["set_member_window"], arrays["set_valid"]) < int(
        mutual_gap
    ):
        return None
    if not np.all(np.isfinite(arrays["set_crps"][arrays["set_valid"]])):
        return None
    return arrays


# --------------------------------------------------------------------------
# Teacher-quality gate.
# --------------------------------------------------------------------------


def _dist_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p90": float(np.quantile(arr, 0.90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def compute_quality_gate(
    arrays: dict[str, np.ndarray],
    *,
    twin: dict[str, np.ndarray] | None,
    oracle_reference: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compute the (a)/(b)/(c) teacher-quality gate from concatenated shards."""

    valid = np.asarray(arrays["set_valid"], dtype=bool)
    category = np.asarray(arrays["set_category"], dtype=np.int8)
    crps = np.asarray(arrays["set_crps"], dtype=np.float64)
    locality = np.asarray(arrays["locality_set_crps"], dtype=np.float64)
    best = np.asarray(arrays["best_set_crps"], dtype=np.float64)
    set_std = np.asarray(arrays["set_crps_std"], dtype=np.float64)
    rho = np.asarray(arrays["spearman_set_vs_sum_solo"], dtype=np.float64)
    n_queries = int(valid.shape[0])

    # ---- (a) set-CRPS spread vs CRN twin-noise floor.
    gate_a: dict[str, Any] = {
        "definition": (
            "per-query std of set CRPS across the valid sets vs the CRN "
            "twin-noise floor sigma_noise = rms(crps_twin - crps_stored) / "
            "sqrt(2) over the re-scored subsample (second CRN stream)"
        ),
        "per_query_set_crps_std": _dist_stats(set_std),
    }
    if twin is not None and twin["stored_crps"].size > 0:
        diff = np.asarray(twin["twin_crps"], dtype=np.float64) - np.asarray(
            twin["stored_crps"], dtype=np.float64
        )
        sigma_noise = float(np.sqrt(np.mean(np.square(diff)) / 2.0))
        ratio = set_std / max(sigma_noise, 1e-12)
        gate_a.update(
            {
                "n_twin_rescored_sets": int(diff.size),
                "n_twin_subsample_queries": int(
                    np.unique(twin["query_window"]).size
                ),
                "twin_crps_delta_mean_abs": float(np.mean(np.abs(diff))),
                "sigma_noise_crn_floor": sigma_noise,
                "std_over_noise_ratio": _dist_stats(ratio),
                "frac_queries_std_gt_2x_noise": float(np.mean(ratio > 2.0)),
                "frac_queries_std_gt_5x_noise": float(np.mean(ratio > 5.0)),
            }
        )
    else:
        gate_a["n_twin_rescored_sets"] = 0

    # ---- (b) best-sampled-set tilt (mini-oracle headroom vs locality).
    tilt = best - locality
    best_is_locality = np.isclose(best, locality)
    best_category_counts: dict[str, int] = {}
    per_category_tilt: dict[str, Any] = {}
    for name, code in CATEGORY_CODE.items():
        cat_mask = valid & (category == code)
        cat_crps = np.where(cat_mask, crps, np.inf)
        cat_min = cat_crps.min(axis=1)
        has_cat = np.isfinite(cat_min)
        per_category_tilt[name] = {
            "n_queries_with_category": int(has_cat.sum()),
            "mean_sets_per_query": float(cat_mask.sum(axis=1).mean()),
            "mean_min_crps_minus_locality": (
                float((cat_min[has_cat] - locality[has_cat]).mean())
                if bool(has_cat.any())
                else None
            ),
            "mean_crps": (
                float(crps[cat_mask].mean()) if bool(cat_mask.any()) else None
            ),
        }
    best_pos = np.where(
        valid, np.where(np.isfinite(crps), crps, np.inf), np.inf
    ).argmin(axis=1)
    for qi in range(n_queries):
        name = CATEGORY_ORDER[int(category[qi, int(best_pos[qi])])]
        best_category_counts[name] = best_category_counts.get(name, 0) + 1
    gate_b: dict[str, Any] = {
        "definition": (
            "mean over queries of (best valid set CRPS - locality set CRPS); "
            "<= 0 by construction (locality is one of the sets); magnitude "
            "= sampled-set mini-oracle headroom under DEPLOYED weighting"
        ),
        "tilt_distribution": _dist_stats(tilt),
        "mean_best_minus_locality": float(tilt.mean()),
        "frac_queries_best_is_locality": float(np.mean(best_is_locality)),
        "best_set_category_counts": best_category_counts,
        "per_category_min_crps_minus_locality": per_category_tilt,
        "locality_set_crps_mean": float(locality.mean()),
        "best_set_crps_mean": float(best.mean()),
        "comparison_to_994b_oracle": {
            "note": (
                "994b = full-pool (50-candidate) solo-replay ORACLE with "
                "oracle softmax(-CRPS/0.02) weights on the VAL frame; here = "
                "40 sampled sets with DEPLOYED start-distance weights on the "
                "TRAIN frame -- directionally comparable, not identical"
            ),
            "reference": oracle_reference,
        },
    }

    # ---- (c) sum-decomposability rank correlation.
    rho_finite = rho[np.isfinite(rho)]
    gate_c: dict[str, Any] = {
        "definition": (
            "per-query Spearman rank correlation between set CRPS and the "
            "SUM of member solo replay CRPS (995d) over the valid sets; "
            "high rho => set value is sum-decomposable (no diversification "
            "signal beyond solos); low rho => set-level signal exists"
        ),
        "spearman_distribution": _dist_stats(rho),
        "frac_queries_rho_gt_0p9": float(np.mean(rho_finite > 0.9))
        if rho_finite.size
        else None,
        "frac_queries_rho_gt_0p7": float(np.mean(rho_finite > 0.7))
        if rho_finite.size
        else None,
        "frac_queries_rho_lt_0p5": float(np.mean(rho_finite < 0.5))
        if rho_finite.size
        else None,
    }
    # secondary: weighted-sum decomposability (deployed weights x solos)
    wsum = np.asarray(arrays["set_weighted_solo_crps"], dtype=np.float64)
    rho_w = np.asarray(
        [
            spearman_rho(crps[qi][valid[qi]], wsum[qi][valid[qi]])
            for qi in range(n_queries)
        ],
        dtype=np.float64,
    )
    gate_c["secondary_weighted_solo_spearman_distribution"] = _dist_stats(rho_w)

    return {
        "schema_version": SCHEMA_VERSION + "_quality_gate",
        "n_queries": n_queries,
        "n_valid_sets_total": int(valid.sum()),
        "set_crps_overall_mean": float(crps[valid].mean()),
        "a_set_spread_vs_crn_noise_floor": gate_a,
        "b_best_sampled_set_tilt": gate_b,
        "c_sum_decomposability_rank_correlation": gate_c,
    }


def load_994b_oracle_reference(path: Path) -> dict[str, Any] | None:
    """Pull the 994b oracle direction (for the gate-(b) comparison)."""

    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    verdict = report.get("kill_condition_assessment", {})
    replay = report.get("oracle_replay", {})
    return {
        "probe_report": str(path),
        "verdict": verdict.get("verdict"),
        "mean_delta_oracle_minus_start_only": (
            verdict.get("per_block_length", {}).get("L6", {}).get("mean_delta")
        ),
        "oracle_crps_pool_best_mean": replay.get("oracle_crps_pool_best_mean"),
        "oracle_crps_selected_top3_mean": replay.get(
            "oracle_crps_selected_top3_mean"
        ),
        "oracle_weight_temperature_used": report.get("weighting", {}).get(
            "temperature_used"
        ),
    }


# --------------------------------------------------------------------------
# Solo-label (995d) loading.
# --------------------------------------------------------------------------


def load_solo_labels(
    solo_dir: Path,
) -> tuple[dict[str, Any], dict[int, dict[str, np.ndarray]]]:
    """Load the 995d manifest + per-query solo rows (pool order preserved)."""

    manifest = json.loads(
        (Path(solo_dir) / "manifest_995d.json").read_text(encoding="utf-8")
    )
    rows: dict[int, dict[str, np.ndarray]] = {}
    for path in sorted(Path(solo_dir).glob("labels_shard_*.npz")):
        with np.load(path) as data:
            queries = data["query_window"].astype(np.int64)
            pool = data["pool_candidate_window"].astype(np.int64)
            crps = data["replay_crps"].astype(np.float64)
            dist = data["cand_start_distance"].astype(np.float64)
            n_eligible = data["n_eligible_candidates"].astype(np.int64)
        for i, q in enumerate(queries):
            rows[int(q)] = {
                "pool_windows": pool[i],
                "solo_crps": crps[i],
                "start_distance": dist[i],
                "n_eligible": int(n_eligible[i]),
            }
    return manifest, rows


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
    parser.add_argument(
        "--solo-labels-dir", type=Path, default=Path(DEFAULT_SOLO_LABELS_DIR)
    )
    parser.add_argument(
        "--oracle-probe-report", type=Path, default=Path(DEFAULT_994B_PROBE_REPORT)
    )
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
    parser.add_argument("--mutual-gap", type=int, default=30)
    parser.add_argument("--set-sampling-seed", type=int, default=0)
    parser.add_argument("--n-solo-informed", type=int, default=14)
    parser.add_argument("--n-div-strat", type=int, default=6)
    parser.add_argument("--n-div-feature", type=int, default=6)
    parser.add_argument("--n-uniform", type=int, default=12)
    parser.add_argument(
        "--solo-informed-temperature",
        type=float,
        default=0.0,
        help="0 = auto-tune on the full 995d solo labels (recorded)",
    )
    parser.add_argument("--ess-target", type=float, default=16.0)
    parser.add_argument(
        "--samples",
        type=int,
        default=16,
        help="samples_per_analogue (engine --samples): total paths per set "
        "= 3 x samples allocated by field weights",
    )
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--common-random-base-seed", type=int, default=8128)
    parser.add_argument(
        "--twin-base-seed",
        type=int,
        default=8129,
        help="second CRN stream for the gate-(a) noise floor",
    )
    parser.add_argument("--noise-subsample-queries", type=int, default=50)
    parser.add_argument("--noise-sets-per-query", type=int, default=5)
    parser.add_argument("--noise-subsample-seed", type=int, default=997)
    parser.add_argument("--score-scale-floor", type=float, default=1e-3)
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--validation-checks", type=int, default=3)
    parser.add_argument("--validation-seed", type=int, default=9971)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    n_sets_target = (
        2 + int(args.n_solo_informed) + int(args.n_div_strat)
        + int(args.n_div_feature) + int(args.n_uniform)
    )

    t_start = time.time()
    runtimes: dict[str, float] = {}
    output_dir = _resolve(Path(str(args.output_dir) + str(args.dir_suffix)))
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress_log.txt"
    manifest_path = output_dir / "manifest_997a.json"

    # ---- Phase 1: model + block frame + 939a bank arrays (995d-identical).
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
        _bl_history_level,
        _bl_history_norm,
        _bl_center,
        _bl_scale,
        _bl_drift,
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
        raise ValueError("939a bank arrays diverge from the rebuilt block frame")
    del (
        _bl_history_level,
        _bl_history_norm,
        _bl_center,
        _bl_scale,
        _bl_drift,
        block_history_raw,
        block_future_raw,
        block_future_delta,
        block,
    )
    runtimes["setup_seconds"] = round(time.time() - t_start, 1)

    # ---- Phase 2: train-side scales (995d/994b convention, computed once).
    train_indices = list(range(n_bank))
    delta_scale = build_delta_scale(future_delta, floor=float(args.score_scale_floor))
    terminal = history_raw[:, -1, :]
    terminal_scale = _safe_scale(terminal, train_indices)

    # ---- Phase 3: deterministic query plan, asserted == the 995d manifest.
    plan_windows_full = sample_train_query_windows(
        n_queries=int(args.n_queries),
        lo=int(args.query_lo),
        hi=int(args.query_hi),
        seed=int(args.sampling_seed),
    )
    solo_manifest, solo_rows = load_solo_labels(_resolve(args.solo_labels_dir))
    solo_labeled = np.asarray(
        solo_manifest["query_windows_labeled"], dtype=np.int64
    )
    if not np.array_equal(plan_windows_full, solo_labeled):
        raise AssertionError(
            "997a query plan does not match the 995d manifest "
            f"query_windows_labeled ({plan_windows_full.size} planned vs "
            f"{solo_labeled.size} labeled)"
        )
    if sorted(solo_rows) != [int(q) for q in solo_labeled]:
        raise AssertionError("995d shards do not cover the manifest query list")
    plan_windows = plan_windows_full
    if int(args.max_queries) > 0:
        plan_windows = plan_windows[: int(args.max_queries)]

    # ---- Phase 4: rebuild causal pools; assert byte-equal to 995d pools.
    universe_max = n_bank - 1
    pool_by_query: dict[int, dict[str, np.ndarray]] = {}
    for q in plan_windows:
        rows, n_eligible = build_causal_pool(
            query_index=int(q),
            terminal=terminal,
            scale=terminal_scale,
            pool_size=int(args.pool_size),
            causal_gap=int(args.causal_gap),
            universe_max=universe_max,
            min_eligible=int(args.min_eligible),
        )
        if rows is None:
            raise AssertionError(
                f"query {int(q)} has no causal pool here but was labeled in 995d"
            )
        windows = np.asarray([row["window_index"] for row in rows], dtype=np.int64)
        distances = np.asarray(
            [row["start_distance"] for row in rows], dtype=np.float64
        )
        solo = solo_rows[int(q)]
        if not np.array_equal(windows, solo["pool_windows"]):
            raise AssertionError(
                f"rebuilt pool for query {int(q)} diverges from the 995d shard pool"
            )
        if not np.allclose(distances, solo["start_distance"], atol=1e-6):
            raise AssertionError(
                f"rebuilt pool distances for query {int(q)} diverge from 995d"
            )
        pool_by_query[int(q)] = {
            "windows": windows,
            "distances": distances,
            "solo_crps": np.asarray(solo["solo_crps"], dtype=np.float64),
            "n_eligible": int(solo["n_eligible"]),
        }

    # ---- Phase 5: solo-informed temperature (tuned ONCE on the FULL plan).
    solo_full = [
        np.asarray(solo_rows[int(q)]["solo_crps"], dtype=np.float64)
        for q in plan_windows_full
    ]
    if float(args.solo_informed_temperature) > 0.0:
        informed_temperature = float(args.solo_informed_temperature)
        temperature_diagnostics: dict[str, Any] = {
            "criterion": "user-fixed via --solo-informed-temperature",
            "chosen_temperature": informed_temperature,
        }
    else:
        informed_temperature, temperature_diagnostics = (
            tune_solo_informed_temperature(
                solo_full, grid=TEMPERATURE_GRID, target_ess=float(args.ess_target)
            )
        )

    # ---- Phase 6: deterministic per-query set plans.
    sets_by_query: dict[int, list[dict[str, Any]]] = {}
    dropped_by_query: dict[int, int] = {}
    skipped: list[dict[str, Any]] = []
    for q in plan_windows:
        pool = pool_by_query[int(q)]
        rng = np.random.default_rng([int(args.set_sampling_seed), int(q)])
        informed_probs = solo_informed_probabilities(
            pool["solo_crps"], temperature=informed_temperature
        )
        features = (
            terminal[pool["windows"]].astype(np.float64)
            / np.asarray(terminal_scale, dtype=np.float64)[None, :]
        )
        sets, dropped = build_query_sets(
            rng,
            windows=pool["windows"],
            solo_crps=pool["solo_crps"],
            features=features,
            informed_probs=informed_probs,
            mutual_gap=int(args.mutual_gap),
            n_solo_informed=int(args.n_solo_informed),
            n_div_strat=int(args.n_div_strat),
            n_div_feature=int(args.n_div_feature),
            n_uniform=int(args.n_uniform),
        )
        if not sets:
            skipped.append(
                {
                    "query_window": int(q),
                    "reason": "no gap-respecting locality set in the pool",
                }
            )
            continue
        sets_by_query[int(q)] = sets
        dropped_by_query[int(q)] = int(dropped)
    labeled_windows = [int(q) for q in plan_windows if int(q) in sets_by_query]
    n_valid_total = sum(len(sets_by_query[q]) for q in labeled_windows)
    runtimes["plan_pool_seconds"] = round(
        time.time() - t_start - runtimes["setup_seconds"], 1
    )
    _progress(
        progress_path,
        f"plan ready: {plan_windows.size} queries, {len(labeled_windows)} labeled, "
        f"{len(skipped)} skipped; {n_valid_total} sets total "
        f"(target {n_sets_target}/query); solo-informed T={informed_temperature:g}",
    )

    shard_size = max(1, int(args.shard_size))
    n_shards = (len(labeled_windows) + shard_size - 1) // shard_size
    paths_per_set = N_MEMBERS * int(args.samples)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "scope_note": (
            "SET-level teacher labels (997a): deployed-semantics mixture "
            "replay CRPS for 40 candidate sets of 3 per query over the 995d "
            "causal pools. Successor to the killed 996b solo-taught tilt."
        ),
        "sampling": {
            "n_queries_requested": int(args.n_queries),
            "max_queries_truncation": int(args.max_queries),
            "window_range": [int(args.query_lo), int(args.query_hi)],
            "seed": int(args.sampling_seed),
            "plan_asserted_equal_to_995d_manifest": True,
            "n_planned": int(plan_windows.size),
            "n_labeled": len(labeled_windows),
            "n_skipped": len(skipped),
        },
        "pool": {
            "source": str(_resolve(args.solo_labels_dir)),
            "pool_size": int(args.pool_size),
            "causal_gap": int(args.causal_gap),
            "rebuilt_and_asserted_equal_to_995d_shards": True,
        },
        "set_plan": {
            "n_sets_target_per_query": int(n_sets_target),
            "n_members": int(N_MEMBERS),
            "mutual_gap_within_set": int(args.mutual_gap),
            "per_query_rng": (
                "np.random.default_rng([set_sampling_seed, query_window]) "
                f"with set_sampling_seed={int(args.set_sampling_seed)}"
            ),
            "categories": {
                "locality": "1 set: start-only top-3, gap-respecting greedy",
                "solo_oracle": (
                    "1 set: top-3 by 995d solo replay_crps, gap-respecting "
                    "greedy (deduped vs locality)"
                ),
                "solo_informed": (
                    f"{int(args.n_solo_informed)} sets: members sampled w/o "
                    "replacement, p ~ softmax(-z(solo_crps)/T), per-query "
                    "standardized; T tuned once on the full 995d labels"
                ),
                "diversity_stratified": (
                    f"{int(args.n_div_strat)} sets: one member per "
                    "start-distance tercile of the pool"
                ),
                "diversity_feature_spread": (
                    f"{int(args.n_div_feature)} sets: k-means++-style greedy "
                    "spread over z-scaled terminal prefix features"
                ),
                "uniform_random": f"{int(args.n_uniform)} sets: uniform over pool",
            },
            "dedup": (
                "identical member sets deduped per query; sampled duplicates "
                "resampled up to 50 attempts; deterministic duplicates dropped"
            ),
            "solo_informed_temperature": float(informed_temperature),
            "solo_informed_temperature_diagnostics": temperature_diagnostics,
            "dropped_set_slots_total": int(sum(dropped_by_query.values())),
        },
        "set_score": {
            "checkpoint": str(args.checkpoint),
            "weighting": DEPLOYED_WEIGHTING_TEXT,
            "samples_per_analogue": int(args.samples),
            "paths_per_set_total": int(paths_per_set),
            "n_steps": int(args.n_steps),
            "chunk_size": int(args.chunk_size),
            "crn_scheme": CRN_SCHEME_TEXT.replace("candidate", "set"),
            "common_random_base_seed": int(args.common_random_base_seed),
            "metric": (
                "engine score_sample_distribution ensemble_crps_z (+ "
                "energy_score_z, coverage_80) vs the query's realized "
                "future_delta; delta_scale = build_delta_scale over the FULL "
                f"train bank future_delta (floor {float(args.score_scale_floor)})"
            ),
        },
        "noise_floor": {
            "twin_base_seed": int(args.twin_base_seed),
            "subsample_queries": int(args.noise_subsample_queries),
            "sets_per_query": int(args.noise_sets_per_query),
            "subsample_seed": int(args.noise_subsample_seed),
        },
        "shard_schema": {
            "arrays": list(SET_SHARD_ARRAY_KEYS),
            "shard_size_queries": shard_size,
            "n_shards_expected": n_shards,
        },
        "skipped_queries": skipped,
        "query_windows_labeled": labeled_windows,
        "volume": {
            "n_set_rollouts": int(n_valid_total),
            "paths_total": int(n_valid_total * paths_per_set),
        },
        "runtimes": runtimes,
    }
    _write_json(manifest_path, manifest)

    # ---- Phase 7: set replay scoring with incremental shards.
    t_score = time.time()
    n_sets_max = int(n_sets_target)
    shard_paths: list[Path] = []
    resumed_queries = 0
    fresh_queries = 0
    progress_counter = 0

    def _expected_plan_arrays(rows: list[int]) -> dict[str, np.ndarray]:
        n_s = len(rows)
        valid = np.zeros((n_s, n_sets_max), dtype=bool)
        cat = np.full((n_s, n_sets_max), -1, dtype=np.int8)
        member = np.full((n_s, n_sets_max, N_MEMBERS), -1, dtype=np.int64)
        for si, q in enumerate(rows):
            pool = pool_by_query[int(q)]
            for ki, s in enumerate(sets_by_query[int(q)]):
                valid[si, ki] = True
                cat[si, ki] = CATEGORY_CODE[s["category"]]
                member[si, ki] = pool["windows"][np.asarray(s["members"])]
        return {"set_valid": valid, "set_category": cat, "set_member_window": member}

    for shard_no in range(n_shards):
        rows = labeled_windows[shard_no * shard_size : (shard_no + 1) * shard_size]
        expected_queries = np.asarray(rows, dtype=np.int64)
        expected_plan = _expected_plan_arrays(rows)
        path = set_shard_path(output_dir, shard_no)
        if not bool(args.no_resume):
            existing = try_load_existing_set_shard(
                path,
                expected_query_window=expected_queries,
                expected_set_valid=expected_plan["set_valid"],
                expected_set_category=expected_plan["set_category"],
                expected_set_member_window=expected_plan["set_member_window"],
                causal_gap=int(args.causal_gap),
                mutual_gap=int(args.mutual_gap),
            )
            if existing is not None:
                shard_paths.append(path)
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
                [pool_by_query[int(q)]["n_eligible"] for q in rows], dtype=np.int64
            ),
            "crn_seed": np.zeros(n_s, dtype=np.int64),
            "n_valid_sets": np.zeros(n_s, dtype=np.int64),
            "n_dropped_set_slots": np.asarray(
                [dropped_by_query[int(q)] for q in rows], dtype=np.int64
            ),
            "set_valid": expected_plan["set_valid"],
            "set_category": expected_plan["set_category"],
            "set_member_window": expected_plan["set_member_window"],
            "set_member_start_distance": np.full(
                (n_s, n_sets_max, N_MEMBERS), np.nan, dtype=np.float64
            ),
            "set_member_solo_crps": np.full(
                (n_s, n_sets_max, N_MEMBERS), np.nan, dtype=np.float64
            ),
            "set_weight": np.full(
                (n_s, n_sets_max, N_MEMBERS), np.nan, dtype=np.float64
            ),
            "set_sample_count": np.zeros(
                (n_s, n_sets_max, N_MEMBERS), dtype=np.int64
            ),
            "set_crps": np.full((n_s, n_sets_max), np.nan, dtype=np.float64),
            "set_energy": np.full((n_s, n_sets_max), np.nan, dtype=np.float64),
            "set_coverage_80": np.full((n_s, n_sets_max), np.nan, dtype=np.float64),
            "set_sum_solo_crps": np.full(
                (n_s, n_sets_max), np.nan, dtype=np.float64
            ),
            "set_weighted_solo_crps": np.full(
                (n_s, n_sets_max), np.nan, dtype=np.float64
            ),
            "locality_set_crps": np.full(n_s, np.nan, dtype=np.float64),
            "best_set_crps": np.full(n_s, np.nan, dtype=np.float64),
            "set_crps_std": np.full(n_s, np.nan, dtype=np.float64),
            "spearman_set_vs_sum_solo": np.full(n_s, np.nan, dtype=np.float64),
        }
        with torch.no_grad():
            for si, q in enumerate(rows):
                pool = pool_by_query[int(q)]
                target = future_delta[int(q)]
                q_sets = sets_by_query[int(q)]
                for ki, s in enumerate(q_sets):
                    members = np.asarray(s["members"], dtype=np.int64)
                    m_windows = pool["windows"][members]
                    m_dist = pool["distances"][members]
                    m_solo = pool["solo_crps"][members]
                    m_match = 1.0 / (1.0 + np.maximum(m_dist, 0.0))
                    weights = deployed_set_weights(m_dist)
                    result = replay_score_set(
                        model,
                        member_windows=[int(w) for w in m_windows],
                        member_match_scores=[float(v) for v in m_match],
                        member_weights=[float(v) for v in weights],
                        query_window=int(q),
                        history_level=history_level,
                        history_norm=history_norm,
                        center=center,
                        scale=scale,
                        drift_feature=drift_feature,
                        history_raw=history_raw,
                        target_delta=target,
                        specs=specs,
                        delta_scale=delta_scale,
                        samples_per_analogue=int(args.samples),
                        n_steps=int(args.n_steps),
                        chunk_size=int(args.chunk_size),
                        crn_base_seed=int(args.common_random_base_seed),
                        device=device,
                    )
                    shard_arrays["crn_seed"][si] = result["crn_seed"]
                    shard_arrays["set_member_start_distance"][si, ki] = m_dist
                    shard_arrays["set_member_solo_crps"][si, ki] = m_solo
                    shard_arrays["set_weight"][si, ki] = weights
                    shard_arrays["set_sample_count"][si, ki] = np.asarray(
                        result["sample_counts"], dtype=np.int64
                    )
                    shard_arrays["set_crps"][si, ki] = result["ensemble_crps_z"]
                    shard_arrays["set_energy"][si, ki] = result["energy_score_z"]
                    shard_arrays["set_coverage_80"][si, ki] = result["coverage_80"]
                    shard_arrays["set_sum_solo_crps"][si, ki] = float(m_solo.sum())
                    shard_arrays["set_weighted_solo_crps"][si, ki] = float(
                        np.sum(weights * m_solo)
                    )
                k_valid = len(q_sets)
                crps_row = shard_arrays["set_crps"][si, :k_valid]
                shard_arrays["n_valid_sets"][si] = k_valid
                shard_arrays["locality_set_crps"][si] = float(crps_row[0])
                shard_arrays["best_set_crps"][si] = float(crps_row.min())
                shard_arrays["set_crps_std"][si] = float(crps_row.std())
                shard_arrays["spearman_set_vs_sum_solo"][si] = spearman_rho(
                    crps_row, shard_arrays["set_sum_solo_crps"][si, :k_valid]
                )
                fresh_queries += 1
                progress_counter += 1
                queries_done = fresh_queries + resumed_queries
                if (
                    progress_counter % max(1, int(args.progress_every)) == 0
                    or queries_done >= len(labeled_windows)
                ):
                    elapsed = time.time() - t_score
                    rate = elapsed / max(fresh_queries, 1)
                    remaining = len(labeled_windows) - queries_done
                    eta_s = rate * max(remaining, 0)
                    _progress(
                        progress_path,
                        f"set labels {queries_done}/{len(labeled_windows)} queries "
                        f"({rate:.1f}s/query fresh, eta {eta_s / 3600:.2f}h)",
                    )
        if (
            set_causality_max_excess(
                shard_arrays["query_window"],
                shard_arrays["set_member_window"],
                shard_arrays["set_valid"],
                causal_gap=int(args.causal_gap),
            )
            > 0
        ):
            raise AssertionError(f"causality violation in shard {shard_no}")
        if set_mutual_gap_min(
            shard_arrays["set_member_window"], shard_arrays["set_valid"]
        ) < int(args.mutual_gap):
            raise AssertionError(f"mutual-gap violation in shard {shard_no}")
        if not np.all(
            np.isfinite(shard_arrays["set_crps"][shard_arrays["set_valid"]])
        ):
            raise ValueError(f"non-finite set CRPS in shard {shard_no}")
        np.savez_compressed(path, **shard_arrays)
        shard_paths.append(path)
        _progress(
            progress_path,
            f"shard {shard_no:04d} written: {path.name} "
            f"({expected_queries.size} queries x <= {n_sets_max} sets)",
        )
        manifest["shards_written"] = [str(p) for p in shard_paths]
        _write_json(manifest_path, manifest)
    runtimes["set_label_seconds"] = round(time.time() - t_score, 1)

    # ---- Phase 8: concatenate shards.
    all_arrays: dict[str, list[np.ndarray]] = {k: [] for k in SET_SHARD_ARRAY_KEYS}
    for path in shard_paths:
        with np.load(path) as data:
            for key in SET_SHARD_ARRAY_KEYS:
                all_arrays[key].append(data[key].copy())
    cat_arrays = {k: np.concatenate(v, axis=0) for k, v in all_arrays.items()}

    # ---- Phase 9: CRN twin-noise re-scoring (gate (a) noise floor).
    t_twin = time.time()
    rng_twin = np.random.default_rng(int(args.noise_subsample_seed))
    n_q_all = int(cat_arrays["query_window"].size)
    sub_n = min(int(args.noise_subsample_queries), n_q_all)
    sub_rows = np.sort(rng_twin.choice(n_q_all, size=sub_n, replace=False))
    twin_records: dict[str, list[float | int]] = {
        "query_window": [],
        "set_slot": [],
        "stored_crps": [],
        "twin_crps": [],
    }
    with torch.no_grad():
        for row in sub_rows:
            q = int(cat_arrays["query_window"][row])
            pool = pool_by_query[q]
            q_sets = sets_by_query[q]
            n_valid = int(cat_arrays["n_valid_sets"][row])
            n_pick = min(int(args.noise_sets_per_query), n_valid)
            slots = np.sort(rng_twin.choice(n_valid, size=n_pick, replace=False))
            target = future_delta[q]
            for slot in slots:
                s = q_sets[int(slot)]
                members = np.asarray(s["members"], dtype=np.int64)
                m_windows = pool["windows"][members]
                m_dist = pool["distances"][members]
                m_match = 1.0 / (1.0 + np.maximum(m_dist, 0.0))
                weights = deployed_set_weights(m_dist)
                result = replay_score_set(
                    model,
                    member_windows=[int(w) for w in m_windows],
                    member_match_scores=[float(v) for v in m_match],
                    member_weights=[float(v) for v in weights],
                    query_window=q,
                    history_level=history_level,
                    history_norm=history_norm,
                    center=center,
                    scale=scale,
                    drift_feature=drift_feature,
                    history_raw=history_raw,
                    target_delta=target,
                    specs=specs,
                    delta_scale=delta_scale,
                    samples_per_analogue=int(args.samples),
                    n_steps=int(args.n_steps),
                    chunk_size=int(args.chunk_size),
                    crn_base_seed=int(args.twin_base_seed),
                    device=device,
                )
                twin_records["query_window"].append(q)
                twin_records["set_slot"].append(int(slot))
                twin_records["stored_crps"].append(
                    float(cat_arrays["set_crps"][row, int(slot)])
                )
                twin_records["twin_crps"].append(result["ensemble_crps_z"])
    twin_arrays = {
        "query_window": np.asarray(twin_records["query_window"], dtype=np.int64),
        "set_slot": np.asarray(twin_records["set_slot"], dtype=np.int64),
        "stored_crps": np.asarray(twin_records["stored_crps"], dtype=np.float64),
        "twin_crps": np.asarray(twin_records["twin_crps"], dtype=np.float64),
    }
    twin_path = output_dir / "twin_rescore_997a.npz"
    np.savez_compressed(twin_path, **twin_arrays)
    runtimes["twin_rescore_seconds"] = round(time.time() - t_twin, 1)

    # ---- Phase 10: teacher-quality gate.
    oracle_reference = load_994b_oracle_reference(_resolve(args.oracle_probe_report))
    gate = compute_quality_gate(
        cat_arrays, twin=twin_arrays, oracle_reference=oracle_reference
    )
    gate_path = output_dir / "quality_gate_997a.json"
    _write_json(gate_path, gate)
    _progress(
        progress_path,
        "quality gate written: "
        f"mean tilt (b) = {gate['b_best_sampled_set_tilt']['mean_best_minus_locality']:.4f}; "
        f"spearman (c) median = "
        f"{gate['c_sum_decomposability_rank_correlation']['spearman_distribution'].get('median')}",
    )

    # ---- Phase 11: validation (CRN determinism recompute, bitwise).
    validation: dict[str, Any] | None = None
    if not bool(args.skip_validation):
        t_val = time.time()
        rng_val = np.random.default_rng(int(args.validation_seed))
        checks: list[dict[str, Any]] = []
        with torch.no_grad():
            for _ in range(int(args.validation_checks)):
                row = int(rng_val.integers(0, n_q_all))
                q = int(cat_arrays["query_window"][row])
                n_valid = int(cat_arrays["n_valid_sets"][row])
                slot = int(rng_val.integers(0, n_valid))
                pool = pool_by_query[q]
                s = sets_by_query[q][slot]
                members = np.asarray(s["members"], dtype=np.int64)
                m_dist = pool["distances"][members]
                kwargs = dict(
                    member_windows=[int(w) for w in pool["windows"][members]],
                    member_match_scores=[
                        float(v) for v in 1.0 / (1.0 + np.maximum(m_dist, 0.0))
                    ],
                    member_weights=[float(v) for v in deployed_set_weights(m_dist)],
                    query_window=q,
                    history_level=history_level,
                    history_norm=history_norm,
                    center=center,
                    scale=scale,
                    drift_feature=drift_feature,
                    history_raw=history_raw,
                    target_delta=future_delta[q],
                    specs=specs,
                    delta_scale=delta_scale,
                    samples_per_analogue=int(args.samples),
                    n_steps=int(args.n_steps),
                    chunk_size=int(args.chunk_size),
                    crn_base_seed=int(args.common_random_base_seed),
                    device=device,
                )
                run_1 = replay_score_set(model, **kwargs)
                run_2 = replay_score_set(model, **kwargs)
                stored = float(cat_arrays["set_crps"][row, slot])
                checks.append(
                    {
                        "query_window": q,
                        "set_slot": slot,
                        "stored_set_crps": stored,
                        "recompute_run1": run_1["ensemble_crps_z"],
                        "recompute_run2": run_2["ensemble_crps_z"],
                        "run1_eq_run2_bitwise": bool(
                            run_1["ensemble_crps_z"] == run_2["ensemble_crps_z"]
                        ),
                        "recompute_eq_stored_bitwise": bool(
                            run_1["ensemble_crps_z"] == stored
                        ),
                    }
                )
        causality_excess = set_causality_max_excess(
            cat_arrays["query_window"],
            cat_arrays["set_member_window"],
            cat_arrays["set_valid"],
            causal_gap=int(args.causal_gap),
        )
        mutual_min = set_mutual_gap_min(
            cat_arrays["set_member_window"], cat_arrays["set_valid"]
        )
        validation = {
            "schema_version": SCHEMA_VERSION + "_validation",
            "causality_assertion": {
                "rule": f"max(member) <= query - {int(args.causal_gap)}",
                "max_member_minus_query_plus_gap": int(causality_excess),
                "passed": bool(causality_excess <= 0),
            },
            "mutual_gap_assertion": {
                "rule": f"min pairwise member gap >= {int(args.mutual_gap)}",
                "min_pairwise_member_gap": int(mutual_min),
                "passed": bool(mutual_min >= int(args.mutual_gap)),
            },
            "crn_determinism_checks": checks,
            "all_checks_identical": bool(
                all(
                    c["run1_eq_run2_bitwise"] and c["recompute_eq_stored_bitwise"]
                    for c in checks
                )
            ),
        }
        _write_json(output_dir / "validation_997a.json", validation)
        runtimes["validation_seconds"] = round(time.time() - t_val, 1)
        if not validation["causality_assertion"]["passed"]:
            raise AssertionError("validation causality assertion FAILED")
        if not validation["mutual_gap_assertion"]["passed"]:
            raise AssertionError("validation mutual-gap assertion FAILED")
        if not validation["all_checks_identical"]:
            raise AssertionError("validation CRN determinism check FAILED")

    # ---- Phase 12: final manifest.
    runtimes["total_seconds"] = round(time.time() - t_start, 1)
    manifest["status"] = "ok"
    manifest["runtimes"] = runtimes
    manifest["shards_written"] = [str(p) for p in shard_paths]
    manifest["fresh_queries_scored"] = fresh_queries
    manifest["quality_gate"] = {
        "path": str(gate_path),
        "b_mean_best_minus_locality": gate["b_best_sampled_set_tilt"][
            "mean_best_minus_locality"
        ],
        "c_spearman_median": gate["c_sum_decomposability_rank_correlation"][
            "spearman_distribution"
        ].get("median"),
    }
    if validation is not None:
        manifest["validation"] = {
            "path": str(output_dir / "validation_997a.json"),
            "causality_passed": validation["causality_assertion"]["passed"],
            "mutual_gap_passed": validation["mutual_gap_assertion"]["passed"],
            "all_checks_identical": validation["all_checks_identical"],
        }
    _write_json(manifest_path, manifest)
    _progress(progress_path, f"manifest finalized: {manifest_path}")
    print(
        json.dumps(
            {
                "n_labeled_queries": len(labeled_windows),
                "n_skipped_queries": len(skipped),
                "n_shards": len(shard_paths),
                "fresh_queries_scored": fresh_queries,
                "b_mean_best_minus_locality": manifest["quality_gate"][
                    "b_mean_best_minus_locality"
                ],
                "c_spearman_median": manifest["quality_gate"]["c_spearman_median"],
                "total_runtime_seconds": runtimes["total_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
