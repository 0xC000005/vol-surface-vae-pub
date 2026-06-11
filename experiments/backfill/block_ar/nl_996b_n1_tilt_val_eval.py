#!/usr/bin/env python
"""996b P4 Stage 3: N1 learned within-pool tilt -- 994a val-frame deployment eval.

DEPLOYABLE (no oracle, no realized-future inputs anywhere in the selection
path). For each of the 89 stride-5 val-frame queries (windows 4010..4450):

  1. Embed the query's 995c full_professional narrative
     (text-embedding-3-large; raw + normalized cached per OpenAI policy).
  2. Build the SAME start-local top-50 causal pool as 994a/994b
     (``build_start_local_pool``; z-scaled terminal-state distance over bank
     windows 0..4009; query gap 30).
  3. Score pool candidates with the FROZEN 996a student on deployable features
     (text cosine / within-pool start-distance z / prefix-delta cosine);
     within-pool z-standardize -> ``external_tilt_scores``.
  4. Select top-3 through the P3 chassis: ``build_mixture_memory_prior``
     with ``mode='start_pool_text_tilt'``, ``tilt_mode='external'`` and the
     996a PRE-REGISTERED ``tilt_weight`` (frozen before any val outcome was
     seen); chassis softmax temperature sqrt(D) so the neutral tilt
     reproduces the 994a start-only weights exactly.
  5. Run the IDENTICAL engine command as 994a (same seeds / per-query common
     random numbers) and compare against the 994a start-only baseline with
     the paired moving-block bootstrap (L=6 and L=30).

Pre-registered gates (DO NOT move):
  WIN     paired dCRPS <= -0.013 (half the 994b oracle headroom) with the
          L=6 CI excluding 0  -> promotion-candidate evidence.
  PARTIAL dCRPS < 0 with CI excluding 0 but > -0.013 -> real but small.
  KILL    CI includes 0 or dCRPS >= 0 -> N1 dead as trained.

Also reports pool-quality gate + conflict-detector telemetry across the 89
queries and the CV-vs-val generalization gap.
"""

from __future__ import annotations

import argparse
import hashlib
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
from experiments.backfill.block_ar.nl_994b_oracle_within_pool_tilt import (
    build_start_local_pool,
)
from experiments.backfill.block_ar.nl_996a_n1_tilt_training import (
    FEATURE_NAMES,
    build_candidate_features,
    chassis_terminal_z,
    load_corpus_full_professional_embeddings,
    load_student,
    prefix_delta_z,
    simulate_tilt_selection,
    within_pool_z,
)
from experiments.backfill.block_ar.nl_episode_narrative_bridge_report import (
    _read_jsonl,
    _safe_scale,
    _start_only_ranked_rows,
)
from experiments.backfill.block_ar.nl_episode_narrative_embedding_bridge_report import (
    embed_with_cache,
)
from experiments.backfill.block_ar.nl_episode_narrative_retrieval import (
    pairwise_jaccard_summary,
)
from experiments.backfill.block_ar.nl_prefix_latent_analogue_mixture_prior import (
    build_mixture_memory_prior,
)
from experiments.backfill.block_ar.evaluate_662a_state_aware_normalized_innovation_flow import (
    build_val_block,
)

ROOT = Path(__file__).resolve().parents[3]

DEFAULT_OUTPUT_BASE = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/n1_tilt_val_eval_996b"
)
DEFAULT_STUDENT_JSON = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "n1_tilt_training_996a/n1_final_student_996a.json"
)
DEFAULT_TRAINING_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "n1_tilt_training_996a/n1_tilt_training_report_996a.json"
)
DEFAULT_VAL_CARDS = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "val_frame_narrative_full_995c/multiformat_episode_cards.jsonl"
)
DEFAULT_VAL_GROUNDING = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "val_frame_narrative_full_995c/selected_support_cases.json"
)
DEFAULT_BASELINE_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "val_frame_eval_994a_start_only/scenario_level_eval_report.json"
)
DEFAULT_BASELINE_BRIDGE = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "val_frame_eval_994a_start_only_bridge/start_only_bridge_report.json"
)
EMBEDDING_MODEL = "text-embedding-3-large"

# Pre-registered gate constants (994b oracle headroom -0.0264 => half).
WIN_DCRPS_THRESHOLD = -0.013
GATE_BLOCK_LENGTH = "L6"

# Chassis knobs inert for start_pool_text_tilt (affect only unused scores).
INERT_START_DISTANCE_THRESHOLD_Z = 1.0
INERT_START_DISTANCE_PENALTY = 0.5
INERT_IMPLICATION_ALIGNMENT_WEIGHT = 0.25
INERT_DIVERSE_MAX_PAIRWISE_COSINE = 0.99


def _val_card_window_index(card: dict[str, Any]) -> int:
    window_id = str(card.get("window_id", ""))
    if not window_id.startswith("joint39_val_"):
        raise ValueError(f"unexpected val card window_id: {window_id}")
    return int(window_id.rsplit("_", 1)[-1])


def load_val_narratives(cards_path: Path) -> dict[int, str]:
    cards = _read_jsonl(_resolve(cards_path))
    out: dict[int, str] = {}
    for card in cards:
        views = card.get("views", {})
        text = views.get("full_professional", "") if isinstance(views, dict) else ""
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"val card {card.get('window_id')} lacks full_professional view"
            )
        out[_val_card_window_index(card)] = text.strip()
    return out


def load_val_grounding(grounding_path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(_resolve(grounding_path).read_text(encoding="utf-8"))
    out: dict[int, dict[str, Any]] = {}
    for row in payload.get("selected_cases", []):
        implications = row.get("market_implications", [])
        out[int(row["window_index"])] = {
            "market_implications": implications if isinstance(implications, list) else []
        }
    return out


def n1_gate_assessment(
    l6_summary: dict[str, Any],
    l30_summary: dict[str, Any],
) -> dict[str, Any]:
    """Mechanical reading of the pre-registered N1 gates on the L=6 CRPS CI."""

    mean_delta = float(l6_summary["mean_delta"])
    ci_low = float(l6_summary["ci_low"])
    ci_high = float(l6_summary["ci_high"])
    ci_excludes_zero_improvement = bool(ci_high < 0.0)
    if ci_excludes_zero_improvement and mean_delta <= WIN_DCRPS_THRESHOLD:
        verdict = "WIN_promotion_candidate_evidence"
    elif ci_excludes_zero_improvement and mean_delta < 0.0:
        verdict = "PARTIAL_real_but_small_positive_not_promotable"
    else:
        verdict = "KILL_n1_dead_as_trained"
    return {
        "pre_registered_gates": {
            "WIN": (
                f"paired dCRPS vs start-only <= {WIN_DCRPS_THRESHOLD} (half the "
                "994b oracle headroom -0.0264) with CI excluding 0 at L=6"
            ),
            "PARTIAL": (
                f"dCRPS < 0 with CI excluding 0 but > {WIN_DCRPS_THRESHOLD}"
            ),
            "KILL": "CI includes 0 or dCRPS >= 0",
        },
        "gate_block_length": GATE_BLOCK_LENGTH,
        "metric": "ensemble_crps_z",
        "delta_orientation": "n1_tilt - start_only (negative = tilt better)",
        "l6_mean_delta": mean_delta,
        "l6_ci": [ci_low, ci_high],
        "l6_ci_excludes_zero_improvement": ci_excludes_zero_improvement,
        "l30_mean_delta": float(l30_summary["mean_delta"]),
        "l30_ci": [float(l30_summary["ci_low"]), float(l30_summary["ci_high"])],
        "verdict": verdict,
    }


def build_n1_bridge_report(
    *,
    selections: list[dict[str, Any]],
    train_indices: list[int],
    query_indices: list[int],
    n_block_windows: int,
    arrays_path: str,
    pool_size: int,
    query_gap: int,
    tilt_weight: float,
    chassis_temperature: float,
    student_spec_path: str,
) -> dict[str, Any]:
    """N1-tilt bridge report in the start-only schema the engine consumes."""

    heldout_rows: list[dict[str, Any]] = []
    result_sets: dict[str, list[str]] = {}
    for query_no, selection in enumerate(selections):
        q = int(selection["window_index"])
        top_train_pool: list[dict[str, Any]] = []
        for rank, support in enumerate(selection["supports"], 1):
            idx = int(support["window_index"])
            top_train_pool.append(
                {
                    "rank": int(rank),
                    "window_index": idx,
                    "window_id": f"window_{idx:04d}",
                    "cosine": float(support["start_match_score"]),
                    "weight": float(support["weight"]),
                    "retrieval_score": float(support["combined_score"]),
                    "scenario_title": "N1 learned within-pool tilt support",
                    "score_components": {
                        "method": "n1_learned_within_pool_tilt",
                        "start_distance": float(support["start_distance"]),
                        "start_match_score": float(support["start_match_score"]),
                        "locality_rank": int(support["locality_rank"]),
                        "student_tilt_z": float(support["student_tilt_z"]),
                        "chassis_combined_score": float(support["combined_score"]),
                        "tilt_weight": float(tilt_weight),
                        "chassis_temperature": float(chassis_temperature),
                    },
                }
            )
        window_id = f"window_{q:04d}"
        heldout_rows.append(
            {
                "query_id": f"n1_tilt_{query_no:04d}_{q}",
                "window_index": q,
                "window_id": window_id,
                "role": "anchor",
                "kind": "n1_learned_within_pool_tilt_retrieval",
                "query_text_source": "995c_val_frame_full_professional_narrative",
                "top_train_pool": top_train_pool,
            }
        )
        result_sets[window_id] = [str(item["window_id"]) for item in top_train_pool]
    return {
        "schema_version": "nl_996b_n1_tilt_bridge_v1",
        "status": "ok",
        "scope_note": (
            "DEPLOYABLE N1 learned within-pool tilt: start-local top-"
            f"{int(pool_size)} pool re-ranked by the frozen 996a student "
            "(text cosine / start-distance z / prefix-delta cosine) through "
            "the start_pool_text_tilt chassis with the pre-registered "
            "tilt_weight. No candidate-future or realized-future features."
        ),
        "cards_path": str(DEFAULT_VAL_CARDS),
        "arrays_path": str(arrays_path),
        "retrieval_config": {
            "method": "n1_learned_within_pool_tilt",
            "pool_method": "start_only_terminal_state",
            "pool_size": int(pool_size),
            "query_gap": int(query_gap),
            "top_k": 3,
            "temporal_gap": int(query_gap),
            "tilt_mode": "external",
            "tilt_weight": float(tilt_weight),
            "chassis_temperature": float(chassis_temperature),
            "tilt_score_standardization": "within_pool_z",
            "student_spec": str(student_spec_path),
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


def _win_rates(
    baseline_report: Path,
    tilt_report: Path,
    *,
    method: str = "narrative_generator_topk",
) -> dict[str, Any]:
    scores_a = load_method_scores(baseline_report, method=method)
    scores_b = load_method_scores(tilt_report, method=method)
    out: dict[str, Any] = {}
    for metric in ("ensemble_crps_z", "energy_score_z"):
        deltas, _shared = paired_deltas(scores_a, scores_b, metric=metric)
        out[metric] = {
            "n_windows": int(deltas.size),
            "tilt_win_rate_delta_lt_0": float(np.mean(deltas < 0.0)),
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
        "tilt_win_rate_closer_to_0p80": float(
            np.mean(np.abs(cov_b - 0.80) < np.abs(cov_a - 0.80))
        ),
        "mean_delta": float(np.mean(cov_b - cov_a)),
        "baseline_mean": float(np.mean(cov_a)),
        "tilt_mean": float(np.mean(cov_b)),
    }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-base", type=Path, default=Path(DEFAULT_OUTPUT_BASE))
    parser.add_argument("--dir-suffix", default="", help="e.g. _smoke5")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--student-json", type=Path, default=Path(DEFAULT_STUDENT_JSON))
    parser.add_argument(
        "--training-report", type=Path, default=Path(DEFAULT_TRAINING_REPORT)
    )
    parser.add_argument("--val-cards", type=Path, default=Path(DEFAULT_VAL_CARDS))
    parser.add_argument(
        "--val-grounding", type=Path, default=Path(DEFAULT_VAL_GROUNDING)
    )
    parser.add_argument(
        "--baseline-report", type=Path, default=Path(DEFAULT_BASELINE_REPORT)
    )
    parser.add_argument(
        "--baseline-bridge-report", type=Path, default=Path(DEFAULT_BASELINE_BRIDGE)
    )
    parser.add_argument("--query-stride", type=int, default=5)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--pool-size", type=int, default=50)
    parser.add_argument("--query-gap", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--common-random-base-seed", type=int, default=8128)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=994)
    parser.add_argument("--skip-engine", action="store_true")
    args = parser.parse_args(argv)

    t_start = time.time()
    runtimes: dict[str, float] = {}
    output_dir = _resolve(Path(str(args.output_base) + str(args.dir_suffix)))
    bridge_dir = _resolve(Path(str(args.output_base) + str(args.dir_suffix) + "_bridge"))
    output_dir.mkdir(parents=True, exist_ok=True)
    bridge_dir.mkdir(parents=True, exist_ok=True)

    # ---- Frozen 996a student + pre-registered tilt weight.
    student_payload = json.loads(
        _resolve(args.student_json).read_text(encoding="utf-8")
    )
    student = load_student(student_payload["student"], device="cpu")
    tilt_weight = float(student_payload["preregistered_tilt_weight"])
    chassis_temperature = float(student_payload["chassis_temperature"])
    pool_reference_quantiles = {
        "q90_min_distance": float(
            student_payload["pool_reference_quantiles"]["q90_min_distance"]
        )
    }
    if list(student_payload["feature_names"]) != list(FEATURE_NAMES):
        raise ValueError("student feature names do not match the 996a feature set")

    # ---- Phase 1: rebuild the 0..4450 block frame (994a-identical).
    payload = torch.load(
        _resolve(args.checkpoint), map_location="cpu", weights_only=False
    )
    if int(payload["config"]["history_len"]) != HISTORY_LEN or int(
        payload["config"]["future_len"]
    ) != FUTURE_LEN:
        raise ValueError("checkpoint history/future lengths do not match 30/30 frame")
    spec_names = [
        str(item.get("name", "")) for item in payload.get("state_specs", [])
    ]
    block_args = _block_frame_namespace()
    (
        _history_level,
        _history_norm,
        _center,
        _scale,
        _drift_feature,
        history_raw,
        _specs,
        _block,
    ) = build_val_block(block_args, payload)
    history_raw = np.asarray(history_raw, dtype=np.float32)
    n_block_windows = int(history_raw.shape[0])
    if n_block_windows != VAL_LAST_WINDOW + 1:
        raise ValueError(
            f"block frame has {n_block_windows} windows, expected {VAL_LAST_WINDOW + 1}"
        )
    if np.isnan(history_raw).any():
        raise ValueError("history_raw contains NaNs")
    n_dims = int(history_raw.shape[-1])
    if abs(chassis_temperature - float(np.sqrt(n_dims))) > 1e-9:
        raise ValueError(
            "chassis temperature from 996a does not equal sqrt(D) of this frame"
        )

    train_indices = list(range(BANK_TRAIN_WINDOW_COUNT))
    query_indices = list(
        range(VAL_FIRST_WINDOW, VAL_LAST_WINDOW + 1, int(args.query_stride))
    )
    if int(args.max_queries) > 0:
        query_indices = query_indices[: int(args.max_queries)]
    if max(train_indices) >= min(query_indices):
        raise ValueError("causality violation: bank window >= min query window")

    bank_arrays_path = _resolve(SUPPORT_BANK_DIR) / "support_bank_arrays.npz"
    with np.load(bank_arrays_path) as bank:
        bank_history_raw = np.asarray(bank["history_raw"], dtype=np.float32)
    bank_block_max_abs_diff = float(
        np.max(np.abs(bank_history_raw - history_raw[: bank_history_raw.shape[0]]))
    )
    if bank_block_max_abs_diff > 1e-6:
        raise ValueError(
            "rebuilt block frame disagrees with the 939a bank on rows 0..4009"
        )
    runtimes["frame_build_seconds"] = round(time.time() - t_start, 1)

    # ---- Phase 2: val narrative embeddings (the ONLY OpenAI call site).
    t_embed = time.time()
    val_texts_by_window = load_val_narratives(args.val_cards)
    missing = [q for q in query_indices if q not in val_texts_by_window]
    if missing:
        raise ValueError(f"995c cards missing for query windows: {missing[:5]} ...")
    # Embed the FULL 89-card set in one deterministic order (cache-stable
    # across smoke/full runs), then slice the queries of this run.
    all_val_windows = sorted(val_texts_by_window)
    all_val_texts = [val_texts_by_window[w] for w in all_val_windows]
    val_emb_all, val_emb_meta = embed_with_cache(
        all_val_texts,
        output_dir=output_dir,
        backend="openai",
        model=EMBEDDING_MODEL,
        dotenv_path=str(args.dotenv),
        batch_size=int(args.embedding_batch_size),
        hash_dim=512,
    )
    val_emb_all = np.asarray(val_emb_all, dtype=np.float32)
    norms = np.linalg.norm(val_emb_all, axis=1)
    if float(np.max(np.abs(norms - 1.0))) > 1e-3:
        val_emb_all = val_emb_all / np.maximum(norms[:, None], 1e-12)
    val_pos = {w: i for i, w in enumerate(all_val_windows)}
    query_emb = val_emb_all[[val_pos[q] for q in query_indices]]
    embedding_manifest = {
        "schema_version": "nl_996b_val_text_embedding_manifest_v1",
        "model": EMBEDDING_MODEL,
        "n_texts": len(all_val_texts),
        "embedding_dim": int(val_emb_all.shape[1]),
        "cache": val_emb_meta,
        "normalized": True,
        "per_window": {
            str(w): {
                "chars": len(val_texts_by_window[w]),
                "est_tokens": int(len(val_texts_by_window[w]) / 4),
                "sha256_text": hashlib.sha256(
                    val_texts_by_window[w].encode("utf-8")
                ).hexdigest()[:16],
            }
            for w in all_val_windows
        },
        "est_total_tokens": int(
            sum(len(t) for t in all_val_texts) / 4
        ),
    }
    _write_json(output_dir / "val_text_embedding_manifest_996b.json", embedding_manifest)
    runtimes["val_embedding_seconds"] = round(time.time() - t_embed, 1)

    # ---- Phase 3: corpus embeddings (982g cache only) + deployable features.
    t_feat = time.time()
    emb_windows, emb_matrix, emb_meta = load_corpus_full_professional_embeddings()
    fit_indices = np.arange(BANK_TRAIN_WINDOW_COUNT, dtype=np.int64)
    pref_z = prefix_delta_z(history_raw, fit_indices)
    term_z = chassis_terminal_z(history_raw, fit_indices)

    terminal = history_raw[:, -1, :]
    terminal_scale = _safe_scale(terminal, train_indices)
    train_arr = np.asarray(train_indices, dtype=np.int64)
    pools = [
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

    n_q = len(query_indices)
    pool_n = int(args.pool_size)
    pool_windows = np.asarray(
        [[int(row["window_index"]) for row in pools[qi]] for qi in range(n_q)],
        dtype=np.int64,
    )
    pool_distances = np.asarray(
        [[float(row["start_distance"]) for row in pools[qi]] for qi in range(n_q)],
        dtype=np.float64,
    )
    features = build_candidate_features(
        query_windows=np.asarray(query_indices, dtype=np.int64),
        pool_windows=pool_windows,
        pool_start_distance=pool_distances,
        embedding_windows=emb_windows,
        embeddings=emb_matrix,
        prefix_z=pref_z,
        query_embeddings=query_emb,
    )
    student_scores = np.stack(
        [student.score(features[qi]) for qi in range(n_q)], axis=0
    )
    tilt_z = within_pool_z(student_scores)
    # chassis-unit start scores on the SAME pool
    chassis_start_scores = np.zeros((n_q, pool_n), dtype=np.float64)
    for qi, q in enumerate(query_indices):
        diff = term_z[pool_windows[qi]] - term_z[int(q)][None, :]
        chassis_start_scores[qi] = -np.linalg.norm(diff, axis=1)
    runtimes["feature_seconds"] = round(time.time() - t_feat, 1)
    print(
        f"features built for {n_q} queries x {pool_n} candidates "
        f"({runtimes['feature_seconds']}s)",
        flush=True,
    )

    # ---- Phase 4: chassis selection (start_pool_text_tilt, external tilt).
    t_sel = time.time()
    grounding_by_window = load_val_grounding(args.val_grounding)
    memory_targets = np.zeros((n_block_windows, emb_matrix.shape[1]), dtype=np.float32)
    emb_pos = {int(w): i for i, w in enumerate(emb_windows)}
    for w, i in emb_pos.items():
        memory_targets[w] = emb_matrix[i]
    baseline_bridge = json.loads(
        _resolve(args.baseline_bridge_report).read_text(encoding="utf-8")
    )
    baseline_top3 = {
        int(row["window_index"]): row["top_train_pool"][:3]
        for row in baseline_bridge["evaluation"]["heldout_examples"]
    }

    selections: list[dict[str, Any]] = []
    pool_quality_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    neutral_parity_failures: list[int] = []
    for qi, q in enumerate(query_indices):
        external_tilt_scores = {
            int(pool_windows[qi][ci]): float(tilt_z[qi][ci]) for ci in range(pool_n)
        }
        grounding = grounding_by_window.get(int(q), {"market_implications": []})
        prior = build_mixture_memory_prior(
            query_memory=query_emb[qi],
            memory_targets=memory_targets,
            history_level=history_raw,
            train_indices=train_arr,
            query_window_index=int(q),
            query_start_state=None,
            grounding=grounding,
            spec_names=spec_names,
            mode="start_pool_text_tilt",
            top_k=int(args.top_k),
            temperature=chassis_temperature,
            start_distance_threshold_z=INERT_START_DISTANCE_THRESHOLD_Z,
            start_distance_penalty=INERT_START_DISTANCE_PENALTY,
            implication_alignment_weight=INERT_IMPLICATION_ALIGNMENT_WEIGHT,
            diverse_max_pairwise_cosine=INERT_DIVERSE_MAX_PAIRWISE_COSINE,
            diverse_min_index_gap=0,
            pool_size=pool_n,
            tilt_mode="external",
            tilt_weight=tilt_weight,
            external_tilt_scores=external_tilt_scores,
            pool_reference_quantiles=pool_reference_quantiles,
        )
        # STRONG EQUIVALENCE: chassis output must equal the local simulation
        # on the 994b pool (catches any pool/ordering/weight divergence).
        sim_sel, sim_weights = simulate_tilt_selection(
            pool_windows=pool_windows[qi],
            chassis_start_scores=chassis_start_scores[qi],
            tilt_scores_z=tilt_z[qi],
            tilt_weight=tilt_weight,
            chassis_temperature=chassis_temperature,
            top_k=int(args.top_k),
        )
        sim_windows = [int(pool_windows[qi][p]) for p in sim_sel]
        if list(prior["window_indices"]) != sim_windows:
            raise ValueError(
                f"chassis selection diverges from 994b-pool simulation at "
                f"query {int(q)}: {prior['window_indices']} vs {sim_windows}"
            )
        if float(np.max(np.abs(np.asarray(prior["weights"]) - sim_weights))) > 1e-4:
            raise ValueError(f"chassis weights diverge from simulation at query {int(q)}")
        # neutral-tilt parity vs the 994a baseline bridge (selection + weights)
        sel0, w0 = simulate_tilt_selection(
            pool_windows=pool_windows[qi],
            chassis_start_scores=chassis_start_scores[qi],
            tilt_scores_z=np.zeros(pool_n),
            tilt_weight=0.0,
            chassis_temperature=chassis_temperature,
            top_k=int(args.top_k),
        )
        base_rows = baseline_top3.get(int(q), [])
        base_idx = [int(item["window_index"]) for item in base_rows]
        base_w = np.asarray([float(item["weight"]) for item in base_rows])
        base_w = base_w / base_w.sum() if base_w.size else base_w
        neutral_idx = [int(pool_windows[qi][p]) for p in sel0]
        if base_idx and (
            neutral_idx != base_idx
            or float(np.max(np.abs(w0 - base_w))) > 1e-4
        ):
            neutral_parity_failures.append(int(q))

        by_window = {
            int(pool_windows[qi][ci]): ci for ci in range(pool_n)
        }
        supports = []
        for rank, (idx, weight) in enumerate(
            zip(prior["window_indices"], prior["weights"], strict=True), 1
        ):
            ci = by_window[int(idx)]
            supports.append(
                {
                    "window_index": int(idx),
                    "weight": float(weight),
                    "start_distance": float(pool_distances[qi][ci]),
                    "start_match_score": float(
                        1.0 / (1.0 + max(pool_distances[qi][ci], 0.0))
                    ),
                    "locality_rank": int(ci + 1),
                    "student_tilt_z": float(tilt_z[qi][ci]),
                    "combined_score": float(
                        chassis_start_scores[qi][ci] + tilt_weight * tilt_z[qi][ci]
                    ),
                }
            )
        selections.append({"window_index": int(q), "supports": supports})
        pool_quality_rows.append(
            {"window_index": int(q), **{
                key: prior["pool_quality"][key]
                for key in (
                    "pool_min_start_distance_z",
                    "pool_median_start_distance_z",
                    "analogue_scarce",
                    "analogue_scarce_threshold_z",
                    "reference_source",
                )
            }}
        )
        conflict = prior["narrative_start_conflict"]
        conflict_rows.append(
            {
                "window_index": int(q),
                "conflict": bool(conflict["conflict"]),
                "mismatch_rate": float(conflict["mismatch_rate"]),
                "checked": int(conflict["checked"]),
                "mismatch_count": int(conflict["mismatch_count"]),
            }
        )
        if (qi + 1) % 10 == 0 or qi + 1 == n_q:
            print(
                f"  chassis selection {qi + 1}/{n_q} "
                f"({time.time() - t_sel:.0f}s elapsed)",
                flush=True,
            )
    if neutral_parity_failures:
        raise ValueError(
            "neutral-tilt chassis parity vs the 994a baseline failed for "
            f"queries {neutral_parity_failures[:10]}"
        )
    runtimes["selection_seconds"] = round(time.time() - t_sel, 1)

    # selection-shift diagnostics vs the baseline top-3
    changed = 0
    overlap_counts: list[int] = []
    tilt_locality_ranks: list[int] = []
    for qi, q in enumerate(query_indices):
        tilt_idx = [int(s["window_index"]) for s in selections[qi]["supports"]]
        base_idx = [
            int(item["window_index"]) for item in baseline_top3.get(int(q), [])
        ]
        overlap_counts.append(len(set(tilt_idx) & set(base_idx)))
        if tilt_idx != base_idx:
            changed += 1
        tilt_locality_ranks.extend(
            int(s["locality_rank"]) for s in selections[qi]["supports"]
        )

    # ---- Phase 5: bridge report + UNCHANGED 994a engine command.
    bridge_report = build_n1_bridge_report(
        selections=selections,
        train_indices=train_indices,
        query_indices=[int(q) for q in query_indices],
        n_block_windows=n_block_windows,
        arrays_path=str(bank_arrays_path),
        pool_size=pool_n,
        query_gap=int(args.query_gap),
        tilt_weight=tilt_weight,
        chassis_temperature=chassis_temperature,
        student_spec_path=str(_resolve(args.student_json)),
    )
    bridge_path = bridge_dir / "n1_tilt_bridge_report.json"
    bridge_report["artifact_paths"] = {"report": str(bridge_path)}
    _write_json(bridge_path, bridge_report)
    print(f"bridge report written: {bridge_path}", flush=True)

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
    if not bool(args.skip_engine):
        t_engine = time.time()
        subprocess.run(command, check=True, cwd=str(ROOT))
        runtimes["engine_seconds"] = round(time.time() - t_engine, 1)
        print(f"engine runtime: {runtimes['engine_seconds']}s", flush=True)

    # ---- Phase 6: paired block bootstrap vs the 994a start-only baseline.
    tilt_report_path = output_dir / "scenario_level_eval_report.json"
    baseline_report_path = _resolve(args.baseline_report)
    bootstrap_payloads: dict[str, dict[str, Any]] = {}
    win_rates: dict[str, Any] | None = None
    if not bool(args.skip_engine):
        t_boot = time.time()
        for block_length in (6, 30):
            payload_boot = run_paired_block_bootstrap(
                report_a=baseline_report_path,
                report_b=tilt_report_path,
                method_a="narrative_generator_topk",
                method_b="narrative_generator_topk",
                metrics=("ensemble_crps_z", "energy_score_z", "coverage_80"),
                block_length=int(block_length),
                n_boot=int(args.n_boot),
                seed=int(args.bootstrap_seed),
            )
            payload_boot["comparison"] = (
                "report_a = 994a start-only baseline, report_b = 996b N1 "
                "learned within-pool tilt (deployable)"
            )
            label = f"L{block_length}"
            bootstrap_payloads[label] = payload_boot
            boot_path = (
                output_dir / f"paired_block_bootstrap_n1_tilt_vs_start_only_{label}.json"
            )
            _write_json(boot_path, payload_boot)
            print(f"bootstrap written: {boot_path}", flush=True)
        win_rates = _win_rates(baseline_report_path, tilt_report_path)
        runtimes["bootstrap_seconds"] = round(time.time() - t_boot, 1)

    # ---- Phase 7: pre-registered gate + telemetry + CV-vs-val gap.
    gate = (
        n1_gate_assessment(
            bootstrap_payloads["L6"]["metrics"]["ensemble_crps_z"],
            bootstrap_payloads["L30"]["metrics"]["ensemble_crps_z"],
        )
        if bootstrap_payloads
        else None
    )

    telemetry = {
        "n_queries": n_q,
        "analogue_scarce_count": int(
            sum(1 for row in pool_quality_rows if row["analogue_scarce"])
        ),
        "analogue_scarce_windows": [
            row["window_index"] for row in pool_quality_rows if row["analogue_scarce"]
        ],
        "conflict_count": int(sum(1 for row in conflict_rows if row["conflict"])),
        "conflict_windows": [
            row["window_index"] for row in conflict_rows if row["conflict"]
        ],
        "mean_conflict_mismatch_rate": float(
            np.mean([row["mismatch_rate"] for row in conflict_rows])
        ),
        "mean_checked_claims": float(
            np.mean([row["checked"] for row in conflict_rows])
        ),
        "pool_quality_reference": {
            **pool_reference_quantiles,
            "source": student_payload["pool_reference_quantiles"].get("source", ""),
        },
        "pool_quality_rows": pool_quality_rows,
        "conflict_rows": conflict_rows,
    }

    selection_shift = {
        "frac_queries_selection_changed_vs_start_only": float(changed / n_q),
        "mean_top3_overlap_with_start_only": float(np.mean(overlap_counts)),
        "selected_locality_rank": {
            "mean": float(np.mean(tilt_locality_ranks)),
            "median": float(np.median(tilt_locality_ranks)),
            "max": int(np.max(tilt_locality_ranks)),
        },
    }

    cv_vs_val: dict[str, Any] | None = None
    training_report_path = _resolve(args.training_report)
    if training_report_path.exists() and bootstrap_payloads:
        training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
        grid = training_report["tilt_weight_selection"]["grid"]
        cv_row = next(
            (r for r in grid if float(r["tilt_weight"]) == tilt_weight), None
        )
        cell = training_report["cv_cell_summary"][training_report["selected_cell"]]
        val_delta = float(
            bootstrap_payloads["L6"]["metrics"]["ensemble_crps_z"]["mean_delta"]
        )
        cv_proxy_delta = float(cv_row["mean_proxy_delta_crps"]) if cv_row else None
        cv_vs_val = {
            "teacher_signal_stage": {
                "cv_pair_accuracy": cell["mean_pair_accuracy_over_seeds"],
                "cv_within_pool_spearman": cell["mean_spearman_over_seeds"],
                "teacher_alive_pair_accuracy_gt_0p5": bool(
                    cell["mean_pair_accuracy_over_seeds"] > 0.5
                ),
            },
            "student_generalization_stage": {
                "cv_proxy_mean_delta_crps_at_preregistered_tilt": cv_proxy_delta,
                "cv_proxy_improvement_negative": (
                    bool(cv_proxy_delta < 0.0) if cv_proxy_delta is not None else None
                ),
            },
            "deployment_stage": {
                "val_frame_mean_delta_crps_l6": val_delta,
                "cv_to_val_gap": (
                    float(val_delta - cv_proxy_delta)
                    if cv_proxy_delta is not None
                    else None
                ),
            },
            "note": (
                "stage attribution: teacher dead if CV pair accuracy ~0.5; "
                "student dead if CV proxy delta >= 0; deployment shift if CV "
                "proxy improvement does not transfer to the val frame"
            ),
        }

    tilt_summary_rows: dict[str, Any] | None = None
    if not bool(args.skip_engine):
        tilt_report = json.loads(tilt_report_path.read_text(encoding="utf-8"))
        method_rows = [
            row["methods"]["narrative_generator_topk"]
            for row in tilt_report["window_scores"]
        ]
        tilt_summary_rows = {
            metric: summarize_metric([row.get(metric) for row in method_rows])
            for metric in ("ensemble_crps_z", "energy_score_z", "coverage_80")
        }

    runtimes["total_seconds"] = round(time.time() - t_start, 1)
    report = {
        "schema_version": "nl_996b_n1_tilt_val_eval_v1",
        "deployable": True,
        "scope_note": (
            "996b N1 learned within-pool tilt on the 994a val frame: frozen "
            "996a student + pre-registered tilt_weight through the "
            "start_pool_text_tilt chassis; identical engine command / CRN "
            "seeds as the 994a start-only baseline."
        ),
        "frame": {
            "definition": (
                f"broad val frame windows {VAL_FIRST_WINDOW}..{VAL_LAST_WINDOW} "
                f"(eval_split=val, test_start={TEST_START}, "
                f"val_size={VAL_SIZE_BROAD}); query stride {int(args.query_stride)}"
            ),
            "query_indices": [int(q) for q in query_indices],
            "nonoverlapping_30d_blocks": count_nonoverlapping_blocks(
                [int(q) for q in query_indices]
            ),
        },
        "causality": {
            "bank_max_window_index": int(BANK_TRAIN_WINDOW_COUNT - 1),
            "min_query_window_index": int(min(query_indices)),
            "bank_vs_block_history_raw_max_abs_diff": bank_block_max_abs_diff,
            "deployable_features_only": True,
            "excluded_features": ["cand_future_activity", "cand_index_gap"],
        },
        "preregistration": {
            "tilt_weight": tilt_weight,
            "chassis_temperature": chassis_temperature,
            "student_spec": str(_resolve(args.student_json)),
            "tilt_score_standardization": "within_pool_z",
            "frozen_before_val_outcomes": True,
        },
        "val_text_embedding": {
            key: embedding_manifest[key]
            for key in ("model", "n_texts", "embedding_dim", "est_total_tokens", "cache")
        },
        "corpus_embedding": emb_meta,
        "neutral_parity_vs_994a_baseline": {
            "checked_queries": n_q,
            "failures": 0,
            "note": (
                "zero-tilt chassis-simulated top-3 + weights match the 994a "
                "start-only bridge exactly per query"
            ),
        },
        "selection_shift": selection_shift,
        "engine_command": command,
        "baseline": {
            "report": str(baseline_report_path),
            "bridge_report": str(_resolve(args.baseline_bridge_report)),
        },
        "n1_tilt_val_frame": tilt_summary_rows,
        "paired_block_bootstrap": {
            label: payload_boot["metrics"]
            for label, payload_boot in bootstrap_payloads.items()
        },
        "win_rates": win_rates,
        "gate_assessment": gate,
        "telemetry": telemetry,
        "cv_vs_val_generalization": cv_vs_val,
        "runtimes": runtimes,
        "artifact_paths": {
            "bridge_report": str(bridge_path),
            "engine_report": str(tilt_report_path),
            "embedding_manifest": str(
                output_dir / "val_text_embedding_manifest_996b.json"
            ),
            "bootstrap_reports": [
                str(
                    output_dir
                    / f"paired_block_bootstrap_n1_tilt_vs_start_only_{label}.json"
                )
                for label in bootstrap_payloads
            ],
        },
    }
    report_path = output_dir / "n1_tilt_val_eval_report_996b.json"
    _write_json(report_path, report)
    print(f"996b report written: {report_path}", flush=True)
    print(
        json.dumps(
            {
                "n_queries": n_q,
                "tilt_weight": tilt_weight,
                "frac_selection_changed": selection_shift[
                    "frac_queries_selection_changed_vs_start_only"
                ],
                "analogue_scarce": telemetry["analogue_scarce_count"],
                "conflicts": telemetry["conflict_count"],
                "verdict": gate["verdict"] if gate else "engine_skipped",
                "total_seconds": runtimes["total_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
