#!/usr/bin/env python
"""994a Gate [D] CPU piece: per-factor TILT-SPACE NULL BAND (metamorphic core).

Builds the CPU REFERENCE band ("natural per-factor surprise magnitude") against
which a conditioned method's tilt will later be judged in framework-v1 Gate [D]
(metamorphic core; ``docs/research_protocols/nl_counterfactual_validation_framework_v1.md``
section D + Tier-1 item 2).

DEFINITION (primary, operationalized — §D never rigorously defines "tilt"):
  For each factor f and each evaluated val-frame query window q,
      TILT[q, f] = realized 30-day move  -  start-only ensemble-mean 30-day move
                 = future_delta[q, 29, f] - mean_k(narrative_{q}[k, 29, f])
  Both terms are CUMULATIVE deltas-from-window-start in RAW per-factor level units
  (day index 29 == the 30-day cumulative move). ``narrative_{q}`` is the start-only
  ensemble (48 members) emitted by the start-only ``narrative_generator_topk`` run
  (this artifact is the start-only run; no narrative conditioning is applied).

  The NULL BAND for factor f is the distribution of {TILT[q, f]}_q across the
  evaluated windows, summarized as {p10, mean, std, p90}. This is the
  start-only 30-day FORECAST-SURPRISE distribution: how far reality naturally
  lands from the unconditioned ensemble mean, per factor.

  Block-bootstrap CIs are attached to each of the four band statistics by
  resampling NON-OVERLAPPING 30-day blocks of consecutive query windows
  (block_len_windows = 30 // stride = 6 at stride 5 => ~15 non-overlapping
  blocks), so the CIs respect the serial dependence induced by overlapping
  30-day futures. The band itself (p10/mean/std/p90) is computed on the full
  89-window sample; the bootstrap only quantifies sampling uncertainty ON those
  four numbers (it is NOT a CI on the mean tilt).

INPUTS (all CPU, verified-clean 994a start-only run — leakage re-verified CLEAN,
min query->support gap 45, 0/89 violations):
  - .../val_frame_eval_994a_start_only/scenario_level_eval_arrays.npz
      future_delta (4451,30,39), evaluated_indices (89,), narrative_<w> (48,30,39)
  - .../val_frame_eval_994a_start_only/scenario_level_eval_report.json (window_scores)

This script DOES NOT load any model/checkpoint, train, sample the generator, or
touch the GPU. It reads only the precomputed clean 994a arrays/report.

DEFERRED (GPU, NOT built here): the placebo piece -- null-narrative + >=50
placebo-narrative rollouts through the CONDITIONED generator, and the joint
metamorphic report (composition-inside-band + placebo p<=floor + mechanical
direction). Those require running the conditioned generator on the GPU and are
the next step once the GPU frees (Track A is currently training).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_joint39_anchor_map import (  # noqa: E402
    joint39_anchor_columns,
)

SCHEMA_VERSION = "nl_994a_null_band_tilt_space_v1"

DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "val_frame_eval_994a_start_only"
)
DEFAULT_ARRAYS = DEFAULT_OUTPUT_DIR / "scenario_level_eval_arrays.npz"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_DIR / "nl_994a_null_band_tilt_space_v1.json"

HORIZON_DAY_INDEX = 29  # day index 29 == cumulative 30-day move
FUTURE_LEN = 30
QUERY_STRIDE = 5  # stride-5 val-frame queries (994a driver default)


def _resolve(path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def count_nonoverlapping_30d_blocks(
    window_indices: list[int], *, future_len: int = FUTURE_LEN
) -> int:
    """Greedy count of queries whose 30-day futures do not overlap.

    Mirrors ``nl_994a_val_frame_start_only_eval.count_nonoverlapping_blocks``.
    """

    count = 0
    last: int | None = None
    for idx in sorted(int(i) for i in window_indices):
        if last is None or idx - last >= int(future_len):
            count += 1
            last = idx
    return count


def compute_tilt_matrix(
    arrays_path: str | Path,
    *,
    factor_cols: list[int],
    horizon_day_index: int = HORIZON_DAY_INDEX,
) -> tuple[np.ndarray, list[int]]:
    """Return (tilt[n_windows, n_factors], evaluated_window_indices).

    tilt[q, f] = future_delta[q, H, col_f] - mean_k narrative_{q}[k, H, col_f].
    """

    with np.load(_resolve(arrays_path), allow_pickle=True) as data:
        evaluated = [int(w) for w in np.asarray(data["evaluated_indices"]).tolist()]
        future_delta = np.asarray(data["future_delta"], dtype=np.float64)
        cols = np.asarray(factor_cols, dtype=np.int64)
        tilt = np.empty((len(evaluated), cols.size), dtype=np.float64)
        for wi, w in enumerate(evaluated):
            key = f"narrative_{w}"
            if key not in data.files:
                raise KeyError(f"{arrays_path}: missing start-only ensemble {key!r}")
            start_only = np.asarray(data[key], dtype=np.float64)  # (members, T, 39)
            mean_forecast = start_only.mean(axis=0)[int(horizon_day_index)]  # (39,)
            realized = future_delta[w, int(horizon_day_index)]  # (39,)
            tilt[wi] = (realized - mean_forecast)[cols]
    return tilt, evaluated


def _band_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "p10": float(np.quantile(arr, 0.10)),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "p90": float(np.quantile(arr, 0.90)),
    }


def _nonoverlapping_block_starts(n: int, block_len: int) -> list[int]:
    """Tile [0, n) into consecutive non-overlapping blocks of length block_len.

    The final block is allowed to be shorter than block_len if n is not a
    multiple of block_len (it is kept so every window participates).
    """

    return list(range(0, n, max(1, int(block_len))))


def block_bootstrap_band_cis(
    tilt_col: np.ndarray,
    *,
    block_len_windows: int,
    n_boot: int,
    seed: int,
    ci_level: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Block-bootstrap CIs on the four band stats for one factor.

    Resamples NON-OVERLAPPING blocks of consecutive query windows (length
    ``block_len_windows`` == 30 trading days at the query stride) WITH
    replacement, recomputes {p10, mean, std, p90} per replicate, and returns a
    two-sided CI for each statistic. This attaches sampling uncertainty to the
    band; it does NOT replace the point band (which is computed on the full
    sample).
    """

    values = np.asarray(tilt_col, dtype=np.float64).reshape(-1)
    n = int(values.size)
    block_len = max(1, min(int(block_len_windows), n))
    starts = _nonoverlapping_block_starts(n, block_len)
    blocks = [values[s : min(s + block_len, n)] for s in starts]
    n_blocks = len(blocks)
    rng = np.random.default_rng(int(seed))
    keys = ("p10", "mean", "std", "p90")
    boot: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(int(n_boot)):
        pick = rng.integers(0, n_blocks, size=n_blocks)
        resampled = np.concatenate([blocks[i] for i in pick])
        stats = _band_stats(resampled)
        for k in keys:
            boot[k].append(stats[k])
    alpha = (1.0 - float(ci_level)) / 2.0
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        col = np.asarray(boot[k], dtype=np.float64)
        out[k] = {
            "ci_low": float(np.quantile(col, alpha)),
            "ci_high": float(np.quantile(col, 1.0 - alpha)),
        }
    return out


def build_null_band(
    *,
    arrays_path: str | Path,
    data_path: str = "data/multi_factor_data.npz",
    use_all_factors: bool = False,
    n_boot: int = 5000,
    seed: int = 994,
    ci_level: float = 0.95,
    horizon_day_index: int = HORIZON_DAY_INDEX,
    query_stride: int = QUERY_STRIDE,
) -> dict[str, Any]:
    """Build the per-factor tilt-space null band payload."""

    if use_all_factors:
        factor_names = [f"col_{c}" for c in range(39)]
        factor_cols = list(range(39))
        factor_scope = "all_39_joint39_columns"
    else:
        anchor_cols = joint39_anchor_columns(data_path)
        factor_names = list(anchor_cols.keys())
        factor_cols = [anchor_cols[n] for n in factor_names]
        factor_scope = "14_named_anchors_joint39_cols_25_38"

    tilt, evaluated = compute_tilt_matrix(
        arrays_path, factor_cols=factor_cols, horizon_day_index=horizon_day_index
    )
    n_windows = tilt.shape[0]

    block_len_windows = max(1, FUTURE_LEN // max(1, int(query_stride)))  # 6 at stride 5
    n_nonoverlap = count_nonoverlapping_30d_blocks(evaluated)

    per_factor: dict[str, Any] = {}
    for j, name in enumerate(factor_names):
        col = tilt[:, j]
        band = _band_stats(col)
        cis = block_bootstrap_band_cis(
            col,
            block_len_windows=block_len_windows,
            n_boot=n_boot,
            seed=int(seed) + j,
            ci_level=ci_level,
        )
        per_factor[name] = {
            "joint39_column": int(factor_cols[j]),
            "band": band,
            "band_block_bootstrap_ci": cis,
            "min": float(col.min()),
            "max": float(col.max()),
            "median": float(np.median(col)),
            "p10_le_mean_le_p90": bool(band["p10"] <= band["mean"] <= band["p90"]),
        }

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scope_note": (
            "CPU REFERENCE BAND for framework-v1 Gate [D] (metamorphic core; "
            "nl_counterfactual_validation_framework_v1.md section D + Tier-1 item 2). "
            "Per-factor 'natural per-factor surprise magnitude': the distribution "
            "of the per-window TILT = (realized 30-day move - start-only "
            "ensemble-mean 30-day move) for each factor, summarized as "
            "{p10, mean, std, p90}, with block-bootstrap CIs on each statistic. "
            "TILT[q,f] = future_delta[q,29,f] - mean_members(narrative_{q}[:,29,f]); "
            "both terms are cumulative deltas-from-window-start in RAW per-factor "
            "level units; day index 29 == 30-day cumulative move; narrative_{q} is "
            "the start-only (unconditioned) ensemble from this start-only 994a run. "
            "A conditioned method's per-factor tilt will later be judged against "
            "this band (null-narrative composition must land INSIDE the band; "
            "placebo tilts must also fall inside). This is the §D operationalization "
            "of 'tilt' as the start-only 30-day forecast surprise -- §D does not "
            "rigorously define tilt, so this definition is the documentable choice "
            "and MUST be reused verbatim by the deferred GPU placebo/composition piece."
        ),
        "definition": {
            "tilt": (
                "tilt[q,f] = future_delta[q,29,f] - "
                "mean_k(narrative_{q}[k,29,f])"
            ),
            "horizon_day_index": int(horizon_day_index),
            "horizon_days": int(horizon_day_index) + 1,
            "units": "raw per-factor level units (no standardization)",
            "tilt_basis": "cumulative delta from window start",
            "start_only_ensemble_key": "narrative_<window_index> (48 members)",
            "band_statistics": ["p10", "mean", "std", "p90"],
            "band_is_distribution_of_tilts_not_ci_on_mean": True,
        },
        "block_structure": {
            "resampling_unit": "non-overlapping 30-day blocks of consecutive query windows",
            "query_stride": int(query_stride),
            "future_len_days": int(FUTURE_LEN),
            "block_len_days": int(FUTURE_LEN),
            "block_len_windows": int(block_len_windows),
            "n_nonoverlapping_30d_blocks": int(n_nonoverlap),
            "ci_level": float(ci_level),
            "n_boot": int(n_boot),
            "seed": int(seed),
            "note": (
                "block length = 30 trading days = 30 // stride query windows "
                "(6 at stride 5); block bootstrap quantifies sampling uncertainty "
                "ON the four band statistics, recomputing them per replicate. It is "
                "NOT a confidence interval on the mean tilt."
            ),
        },
        "n_windows": int(n_windows),
        "evaluated_window_indices": [int(w) for w in evaluated],
        "factor_scope": factor_scope,
        "factors": per_factor,
        "source_artifacts": {
            "arrays": str(arrays_path),
            "anchor_map": "experiments/backfill/block_ar/nl_joint39_anchor_map.joint39_anchor_columns",
            "data_path": str(data_path),
        },
        "provenance": (
            "Built ONLY from the verified-clean 994a start-only arrays "
            "(leakage independently re-verified CLEAN: min query->support gap 45, "
            "0/89 violations). No model/checkpoint load, no training, no generator "
            "sampling, no GPU."
        ),
        "deferred_gpu_piece": (
            "NOT built here (requires GPU; Track A is training): null-narrative + "
            ">=50 placebo-narrative rollouts through the CONDITIONED generator, and "
            "the joint metamorphic report (composition-inside-band + placebo "
            "p<=floor 1/(N+1) + mechanical direction). Judge conditioned tilts "
            "against THIS band using the same tilt operationalization above. This is "
            "the next step once the GPU frees."
        ),
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--data-path", default="data/multi_factor_data.npz")
    parser.add_argument(
        "--all-factors",
        action="store_true",
        help="use all 39 joint39 columns instead of the 14 named anchors",
    )
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=994)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--query-stride", type=int, default=QUERY_STRIDE)
    args = parser.parse_args(argv)

    payload = build_null_band(
        arrays_path=args.arrays,
        data_path=str(args.data_path),
        use_all_factors=bool(args.all_factors),
        n_boot=int(args.n_boot),
        seed=int(args.seed),
        ci_level=float(args.ci_level),
        query_stride=int(args.query_stride),
    )

    output_path = _resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"written: {output_path}")
    preview = {
        name: payload["factors"][name]["band"]
        for name in ("SPX", "VIX", "CRUDE_OIL")
        if name in payload["factors"]
    }
    print(
        json.dumps(
            {
                "n_windows": payload["n_windows"],
                "factor_scope": payload["factor_scope"],
                "block_len_windows": payload["block_structure"]["block_len_windows"],
                "n_nonoverlapping_30d_blocks": payload["block_structure"][
                    "n_nonoverlapping_30d_blocks"
                ],
                "example_bands": preview,
            },
            indent=1,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
