#!/usr/bin/env python
"""993a: pre-registered hardness/novelty stratification of the 66-window evals (compass P2).

Phase A (pre-registration, OUTCOME-FREE): strata thresholds are computed purely from
train-side start-state geometry (939a bank, history_level[:, -1, :], per-dim z-scored
with train mean/std, mirroring start_distances_to_query_start). Two stratifiers:
  - novelty_1nn: distance to the nearest bank window (self-neighborhood +-30 excluded
    for train-side calibration);
  - density_50nn: mean distance of the 50 nearest bank windows.
Tercile thresholds from the train-side distribution; the 66 heldout queries are then
assigned strata against the causal bank (train windows only). The spec JSON is written
BEFORE any eval report is opened.

Phase B: joins per-window paired deltas (method - start_only) for ensemble_crps_z and
energy_score_z from the existing 982g-era 66q_s16 scenario reports, summarized per
stratum with sign tests. CAVEAT (F4): the 66 windows are stride-1 overlapping (~3
independent 30-day blocks); per-stratum results are DIRECTION-FINDING only, pending the
P0 val-region harness.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUPPORT_ARRAYS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_bank_train_all_939a/support_bank_arrays.npz"
)
OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "retrieval_hardness_stratification_993a"
)
EVAL_REPORTS = {
    "start_only": "episode_card_v3_full_codex_982g_scenario_start_only_66q_s16",
    "grounded_text_preference_984a": "episode_card_v3_full_codex_982g_scenario_grounded_text_preference_984a_s16",
    "embedding_grounded_top3_90": "episode_card_v3_full_codex_982g_scenario_embedding_grounded_top3_90_66q_s16",
    "episode_text": "episode_card_v3_full_codex_982g_scenario_episode_text_66q_s16",
}
EVAL_BASE = Path("experiments/backfill/block_ar/nl_scenario_demo_outputs")
METHOD_KEY = "narrative_generator_topk"
METRICS = ("ensemble_crps_z", "energy_score_z", "coverage_80")
SELF_EXCLUSION = 30
DENSITY_K = 50


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sign_test_p(wins: int, losses: int) -> float:
    """Two-sided binomial sign test p-value (ties dropped)."""
    n = wins + losses
    if n == 0:
        return math.nan
    k = min(wins, losses)
    total = 0.0
    for i in range(0, k + 1):
        total += math.comb(n, i)
    p = min(1.0, 2.0 * total / (2.0**n))
    return float(p)


def phase_a() -> dict:
    with np.load(_resolve(SUPPORT_ARRAYS)) as bank:
        history_level = np.asarray(bank["history_level"], dtype=np.float64)
        train_indices = np.asarray(bank["train_indices"], dtype=np.int64)
        test_indices = np.asarray(bank["test_indices"], dtype=np.int64)
    start = history_level[:, -1, :]  # (4010, 39)
    mu = start[train_indices].mean(axis=0)
    sd = start[train_indices].std(axis=0)
    sd = np.maximum(sd, 1e-12)
    z = (start - mu) / sd

    train_z = z[train_indices]
    # train-side calibration with +-SELF_EXCLUSION exclusion
    novelty_train: list[float] = []
    density_train: list[float] = []
    for pos, w in enumerate(train_indices):
        d = np.linalg.norm(train_z - z[w], axis=1)
        mask = np.abs(train_indices - w) > SELF_EXCLUSION
        dm = d[mask]
        if dm.shape[0] < DENSITY_K:
            continue
        dm_sorted = np.sort(dm)
        novelty_train.append(float(dm_sorted[0]))
        density_train.append(float(dm_sorted[:DENSITY_K].mean()))
    novelty_train = np.asarray(novelty_train)
    density_train = np.asarray(density_train)
    novelty_terciles = [float(np.quantile(novelty_train, q)) for q in (1 / 3, 2 / 3)]
    density_terciles = [float(np.quantile(density_train, q)) for q in (1 / 3, 2 / 3)]

    # heldout assignments vs causal bank (all train windows precede 3944+ queries)
    assignments: dict[str, dict] = {}
    for q in test_indices:
        d = np.linalg.norm(train_z - z[q], axis=1)
        d_sorted = np.sort(d)
        nov = float(d_sorted[0])
        den = float(d_sorted[:DENSITY_K].mean())
        def stratum(value: float, cuts: list[float]) -> str:
            return "low" if value <= cuts[0] else ("mid" if value <= cuts[1] else "high")
        assignments[str(int(q))] = {
            "novelty_1nn": nov,
            "density_50nn": den,
            "novelty_stratum": stratum(nov, novelty_terciles),
            "density_stratum": stratum(den, density_terciles),
        }

    spec = {
        "schema_version": "nl_993a_strata_spec_v1",
        "pre_registered": True,
        "outcome_free": True,
        "stratifiers": {
            "novelty_1nn": {
                "definition": "z-space L2 distance from query start to NEAREST causal bank window",
                "train_terciles": novelty_terciles,
                "train_median": float(np.median(novelty_train)),
            },
            "density_50nn": {
                "definition": f"mean z-space L2 distance of the {DENSITY_K} nearest causal bank windows",
                "train_terciles": density_terciles,
                "train_median": float(np.median(density_train)),
            },
        },
        "self_exclusion_train_calibration": SELF_EXCLUSION,
        "heldout_assignments": assignments,
        "interpretation": (
            "high novelty/density stratum = analogue-scarce (extrapolation-like) regime; "
            "literature prediction: retrieval should help MOST there"
        ),
    }
    return spec


def phase_b(spec: dict) -> dict:
    per_window: dict[str, dict[str, dict[str, float]]] = {}
    for label, dirname in EVAL_REPORTS.items():
        report = json.loads(
            (_resolve(EVAL_BASE / dirname) / "scenario_level_eval_report.json").read_text()
        )
        for row in report["window_scores"]:
            w = str(int(row["window_index"]))
            metrics = row["methods"][METHOD_KEY]
            per_window.setdefault(w, {})[label] = {m: float(metrics[m]) for m in METRICS}

    assignments = spec["heldout_assignments"]
    strata_results: dict[str, dict] = {}
    for strat_name in ("novelty_stratum", "density_stratum"):
        for level in ("low", "mid", "high"):
            windows = [w for w, a in assignments.items() if a[strat_name] == level]
            entry: dict[str, dict] = {"n_windows": len(windows), "methods": {}}
            for label in EVAL_REPORTS:
                if label == "start_only":
                    continue
                deltas = {m: [] for m in METRICS}
                for w in windows:
                    if w in per_window and label in per_window[w] and "start_only" in per_window[w]:
                        for m in METRICS:
                            deltas[m].append(per_window[w][label][m] - per_window[w]["start_only"][m])
                method_entry = {}
                for m in METRICS:
                    arr = np.asarray(deltas[m], dtype=np.float64)
                    if arr.shape[0] == 0:
                        continue
                    wins = int(np.sum(arr < 0)) if m != "coverage_80" else int(np.sum(arr > 0))
                    losses = int(arr.shape[0]) - wins - int(np.sum(arr == 0))
                    method_entry[m] = {
                        "mean_delta_vs_start_only": float(arr.mean()),
                        "median_delta_vs_start_only": float(np.median(arr)),
                        "win_rate": float(wins / max(arr.shape[0], 1)),
                        "sign_test_p": _sign_test_p(wins, losses),
                        "n": int(arr.shape[0]),
                    }
                entry["methods"][label] = method_entry
            strata_results[f"{strat_name}:{level}"] = entry

    # headline (all 66) for reference
    headline: dict[str, dict] = {}
    for label in EVAL_REPORTS:
        if label == "start_only":
            continue
        deltas = {m: [] for m in METRICS}
        for w, methods in per_window.items():
            if label in methods and "start_only" in methods:
                for m in METRICS:
                    deltas[m].append(methods[label][m] - methods["start_only"][m])
        headline[label] = {
            m: {
                "mean_delta": float(np.mean(v)),
                "median_delta": float(np.median(v)),
                "n": len(v),
            }
            for m, v in deltas.items()
            if v
        }

    return {
        "schema_version": "nl_993a_stratified_results_v1",
        "caveat_f4": (
            "66 stride-1 overlapping windows ~= 3 independent 30-day blocks; per-stratum "
            "results are direction-finding ONLY, pending the P0 val-region harness"
        ),
        "method_key": METHOD_KEY,
        "headline_all_66": headline,
        "strata": strata_results,
    }


def main() -> int:
    output = _resolve(OUTPUT_DIR)
    spec = phase_a()
    _write_json(output / "strata_spec.json", spec)
    print("phase A spec written (outcome-free)")
    results = phase_b(spec)
    _write_json(output / "stratified_results.json", results)
    print(json.dumps(results["headline_all_66"], indent=1))
    print("phase B results written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
