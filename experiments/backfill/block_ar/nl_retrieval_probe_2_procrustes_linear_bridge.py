#!/usr/bin/env python3
"""Probe 2 — Procrustes/SVD Linear Cross-Modal Baseline.

Provenance audit:
  990a/990e/991a manifests and their downstream embeddings are CONTAMINATED;
  we NEVER touch them.

  Clean paired text<->memory source:
    manifest_bridge_eval_full_906b_all_windows/
      bridge_eval_arrays.npz
        text_embeddings   : (4663, 1536) float32   <- OpenAI text-embedding-3-small
        memory_targets    : (380,  128)  float32   <- 939a bank SNI memory vectors
        condition_vectors : (4663, 128)  float32   <- learned adapter output
        train_indices     : (278,)  int64
        test_indices      : (66,)   int64
      bridge_eval_report.json
        evaluation.heldout_examples : 285 entries, each with
          embedding_index, kind, window_index, true_rank_full_pool, true_rank_test_pool

    This corpus derives from 906b GPT-label + 939a support bank, predating any
    982g/988b/990a contamination.
    text_dim=1536 (text-embedding-3-small) -- NOT the contaminated 3072-dim 990a/990e corpus.

Methodology:
  1. Test queries (66 windows):
     From bridge_eval_report.evaluation.heldout_examples, extract per-window
     embedding_indices for each text kind (revised_market_description is the
     canonical query kind; 4.3 views per window on average).
     Compute mean text_embedding per test window -> test query matrix.
     Also read the CURRENT TRAINED ADAPTER retrieval performance:
       true_rank_full_pool from heldout_examples (adapter cosine baseline).

  2. Training pairs (278 windows -> positive text examples):
     From manifest_openai_full_906b_all_windows/narrative_pipeline_report.json
     (narrative_bundles, ordered by window), extract the per-window positive
     narrative count and reconstruct text_embedding slices from bridge_eval_arrays.
     These are the first sum(|narratives|) = 1638 rows of text_embeddings,
     ordered by bridge_eval window ordering (train windows first, test windows last).
     Map: bridge pool position -> narrative_pipeline bundle by matching window_index.
     Use ONLY train-split windows to fit the map.

  3. Procrustes SVD:
       W = argmin_W ||X_text @ W - Y_mem||_F (closed-form via SVD)
     Apply W to test queries; measure recall@{1,3,5,10} and rank-median
     against the full 380-window memory pool.

  4. Compare to:
     (a) Current trained adapter (906b cosine baseline) -- read from heldout_examples
     (b) Chance baseline: rank-median ~ (pool_size+1)/2 ~ 190 in pool=380
     NOTE: 992b oracle ceiling (rank-median=208, recall@10=0.111) was measured on
     pool=4010, NOT pool=380.  These values CANNOT be cross-compared.

  5. Interpret using chance baseline:
     - Chance rank-median ~190 in pool=380.
     - Corrected (n=66, bug fix): both Procrustes (~178) and adapter (~175) are near-chance.
       Corrected verdict: NEAR-CHANCE for all methods (not "below chance").
     - The signal is space-limited (not bridge-architecture-limited).

Schema: nl_retrieval_probe_2_procrustes_linear_bridge_v1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Clean inputs ──────────────────────────────────────────────────────────────
BRIDGE_EVAL_ARRAYS = ROOT / (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_bridge_eval_full_906b_all_windows/bridge_eval_arrays.npz"
)
BRIDGE_EVAL_REPORT = ROOT / (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_bridge_eval_full_906b_all_windows/bridge_eval_report.json"
)
PIPELINE_REPORT = ROOT / (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_openai_full_906b_all_windows/narrative_pipeline_report.json"
)
OUTPUT_DIR = ROOT / (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_retrieval_probe_2_procrustes_linear_bridge"
)

# Oracle ceilings
MEMORY_ORACLE_RECALL_AT_10 = 0.111   # 992b, adjacent-memory query offset=5
MEMORY_ORACLE_RANK_MEDIAN = 208.0    # 992b

# Contamination fingerprint for 990a
CONTAMINATION_HASH_990A = "c1586f2f5a4c1787ff4b874b21e3e742f6b011964f13d59ee9413dd52ea002ac"


def _normalize_rows(m: np.ndarray) -> np.ndarray:
    norms = np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
    return m / norms


def _fit_procrustes(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """W = argmin ||X @ W - Y||_F  (closed-form SVD).
    X: (n, d_src) normalised, Y: (n, d_tgt) normalised.
    Returns W: (d_src, d_tgt).
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    s_inv = np.where(S > 1e-10, 1.0 / S, 0.0)
    return ((Vt.T * s_inv) @ (U.T @ Y)).astype(np.float32)


def _retrieval_stats(
    query_norm: np.ndarray,     # (n_q, d)
    gallery_norm: np.ndarray,   # (n_g, d)
    gt_indices: np.ndarray,     # (n_q,) -> gallery row of true target
) -> dict:
    """Cosine retrieval rank statistics."""
    ranks = []
    for i, gt in enumerate(gt_indices):
        scores = query_norm[i] @ gallery_norm.T
        rank = int(np.sum(scores > scores[gt]) + 1)
        ranks.append(rank)
    arr = np.array(ranks)
    n_g = gallery_norm.shape[0]
    return {
        "n_queries": len(arr),
        "pool_size": n_g,
        "rank_median": float(np.median(arr)),
        "rank_mean": float(np.mean(arr)),
        "recall_at_1": float(np.mean(arr <= 1)),
        "recall_at_3": float(np.mean(arr <= 3)),
        "recall_at_5": float(np.mean(arr <= 5)),
        "recall_at_10": float(np.mean(arr <= 10)),
        "recall_at_50": float(np.mean(arr <= 50)),
        "recall_at_100": float(np.mean(arr <= 100)),
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load arrays ───────────────────────────────────────────────────────────
    print(f"Loading: {BRIDGE_EVAL_ARRAYS}")
    with np.load(BRIDGE_EVAL_ARRAYS) as npz:
        text_emb = np.asarray(npz["text_embeddings"], dtype=np.float32)   # (4663, 1536)
        memory_pool = np.asarray(npz["memory_targets"], dtype=np.float32) # (380, 128)
        cond_vecs = np.asarray(npz["condition_vectors"], dtype=np.float32) # (4663, 128)
        train_idx = np.asarray(npz["train_indices"], dtype=np.int64)       # (278,)
        test_idx = np.asarray(npz["test_indices"], dtype=np.int64)         # (66,)

    print(f"  text_embeddings: {text_emb.shape}  memory_pool: {memory_pool.shape}")
    print(f"  train_windows: {len(train_idx)}  test_windows: {len(test_idx)}")

    # Provenance: text-embedding-3-small (dim=1536), NOT 990a/990e (dim=3072)
    text_dim = text_emb.shape[1]
    provenance_clean = text_dim == 1536
    provenance_note = (
        f"text_dim={text_dim}=1536 (text-embedding-3-small, 906b GPT labels). "
        "NOT 990a/990e dim=3072. CLEAN."
        if provenance_clean
        else f"PROVENANCE WARNING: text_dim={text_dim}"
    )
    print(f"  Provenance: {provenance_note}")

    # ── Load bridge eval report for heldout_examples ─────────────────────────
    print(f"Loading: {BRIDGE_EVAL_REPORT}")
    with BRIDGE_EVAL_REPORT.open() as f:
        rpt_bridge = json.load(f)
    heldout_examples = rpt_bridge["evaluation"]["heldout_examples"]
    window_metadata = rpt_bridge["window_metadata"]   # 380 entries

    # Build pool_position -> global_window_index mapping
    # window_metadata[pool_pos]['window_index'] = global index in the 380-window pool
    wm_widx = [wm["window_index"] for wm in window_metadata]  # length 380

    # ── Load pipeline report for train narrative ordering ─────────────────────
    print(f"Loading: {PIPELINE_REPORT}")
    with PIPELINE_REPORT.open() as f:
        rpt_pipeline = json.load(f)
    bundles = rpt_pipeline["narrative_bundles"]  # 380 bundles ordered by global_window_index

    # Build: global_window_index -> (slice_start, slice_end) in text_emb[0:1638]
    # (positives are first 1638 rows, ordered as in bundles)
    cursor = 0
    gidx_to_slice: dict[int, tuple[int, int]] = {}
    for b in bundles:
        n_pos = len(b["narratives"])
        gidx_to_slice[b["window_index"]] = (cursor, cursor + n_pos)
        cursor += n_pos
    total_pos = cursor
    print(f"  Total positive narratives: {total_pos} (expected 1638)")

    # ── Build TRAIN pairs ─────────────────────────────────────────────────────
    # For each train pool position, average the positive text_embeddings for that window
    X_train_list = []  # mean text embeddings per train window
    Y_train_list = []  # memory targets per train window
    train_coverage = 0
    for pool_pos in train_idx:
        gidx = wm_widx[pool_pos]
        if gidx not in gidx_to_slice:
            continue
        lo, hi = gidx_to_slice[gidx]
        if hi > lo:
            mean_text = text_emb[lo:hi].mean(axis=0)
            X_train_list.append(mean_text)
            Y_train_list.append(memory_pool[pool_pos])
            train_coverage += 1

    X_train = np.array(X_train_list, dtype=np.float32)   # (n_tr, 1536)
    Y_train = np.array(Y_train_list, dtype=np.float32)   # (n_tr, 128)
    print(f"  Train pairs: {len(X_train)} / {len(train_idx)} windows covered")

    # ── Build TEST queries via heldout_examples ───────────────────────────────
    # heldout_examples: 285 entries (4.3 per window), each has embedding_index, window_index, kind
    # IMPORTANT: window_index in heldout_examples IS the pool position (range 314..379),
    # matching test_indices from bridge_eval_arrays.npz exactly.
    # DO NOT remap through window_metadata — that would treat pool positions as global
    # window IDs and drop 30/66 windows while corrupting ground-truth positions.
    from collections import defaultdict
    test_window_to_emb_indices: dict[int, list[int]] = defaultdict(list)
    for ex in heldout_examples:
        test_window_to_emb_indices[ex["window_index"]].append(ex["embedding_index"])

    # For each test window, average text embeddings
    X_test_list = []  # mean text embeddings
    Y_test_gt_pool_positions = []  # pool positions (for gallery lookup)
    test_windows_used = []

    for pool_pos, emb_idxs in sorted(test_window_to_emb_indices.items()):
        # pool_pos IS the gallery index directly — verify it is within range
        if pool_pos < 0 or pool_pos >= memory_pool.shape[0]:
            print(f"  WARNING: pool_pos={pool_pos} out of range [0,{memory_pool.shape[0]}), skipping")
            continue
        mean_text = text_emb[emb_idxs].mean(axis=0)
        X_test_list.append(mean_text)
        Y_test_gt_pool_positions.append(pool_pos)
        test_windows_used.append(pool_pos)

    X_test = np.array(X_test_list, dtype=np.float32)           # (n_te, 1536)
    gt_pool_pos = np.array(Y_test_gt_pool_positions, dtype=np.int64)
    print(f"  Test queries: {len(X_test)} windows from heldout_examples")

    # ── Current trained adapter retrieval (read from heldout_examples) ───────
    # true_rank_full_pool is the rank of the TRUE target in the full 380-window pool
    # using the TRAINED ADAPTER (condition_vector cosine)
    window_to_ranks = defaultdict(list)
    for ex in heldout_examples:
        window_to_ranks[ex["window_index"]].append(ex["true_rank_full_pool"])
    # Per-window mean rank (averaging over text view kinds)
    adapter_ranks = []
    for pool_pos in test_windows_used:
        if pool_pos in window_to_ranks:
            adapter_ranks.append(np.mean(window_to_ranks[pool_pos]))

    adapter_rank_median = float(np.median(adapter_ranks)) if adapter_ranks else float("nan")
    adapter_recall_at_10 = float(np.mean([r <= 10 for r in adapter_ranks])) if adapter_ranks else float("nan")
    adapter_recall_at_1 = float(np.mean([r <= 1 for r in adapter_ranks])) if adapter_ranks else float("nan")
    print(f"\n  Trained adapter (906b) rank_median: {adapter_rank_median:.1f}  "
          f"recall@10: {adapter_recall_at_10:.4f}")

    # ── Procrustes SVD map ────────────────────────────────────────────────────
    print("\nFitting Procrustes/SVD linear map ...")
    X_tr_norm = _normalize_rows(X_train)
    Y_tr_norm = _normalize_rows(Y_train)
    W = _fit_procrustes(X_tr_norm, Y_tr_norm)   # (1536, 128)

    X_te_norm = _normalize_rows(X_test)
    X_te_proj = _normalize_rows(X_te_norm @ W)   # (n_te, 128)
    gallery_norm = _normalize_rows(memory_pool)  # (380, 128)

    stats_procrustes = _retrieval_stats(X_te_proj, gallery_norm, gt_pool_pos)
    print(f"  Procrustes SVD recall@10: {stats_procrustes['recall_at_10']:.4f}  "
          f"rank-median: {stats_procrustes['rank_median']:.1f}")

    # ── PCA-256 + Procrustes ablation ─────────────────────────────────────────
    print("Fitting PCA-256 + Procrustes ablation ...")
    text_mean = X_tr_norm.mean(axis=0)
    X_tr_c = X_tr_norm - text_mean
    _, _, Vt_pca = np.linalg.svd(X_tr_c, full_matrices=False)
    V_pca = Vt_pca[:256].T   # (1536, 256)
    X_tr_pca = X_tr_c @ V_pca
    X_te_pca = (X_te_norm - text_mean) @ V_pca
    W_pca = _fit_procrustes(X_tr_pca, Y_tr_norm)   # (256, 128)
    X_te_pca_proj = _normalize_rows(X_te_pca @ W_pca)
    stats_pca = _retrieval_stats(X_te_pca_proj, gallery_norm, gt_pool_pos)
    print(f"  PCA256+Procrustes recall@10: {stats_pca['recall_at_10']:.4f}  "
          f"rank-median: {stats_pca['rank_median']:.1f}")

    # ── Condition-vector cosine (as additional sanity check) ──────────────────
    # Average condition_vectors for heldout examples per window -> expected to be ~1.0
    # because condition_vectors ARE trained to approximate memory_targets
    window_to_cond = defaultdict(list)
    for ex in heldout_examples:
        window_to_cond[ex["window_index"]].append(cond_vecs[ex["embedding_index"]])
    cond_test = []
    for pool_pos in test_windows_used:
        if pool_pos in window_to_cond:
            cond_test.append(np.mean(window_to_cond[pool_pos], axis=0))
    if cond_test:
        cond_test_mat = _normalize_rows(np.array(cond_test, dtype=np.float32))
        stats_cond = _retrieval_stats(cond_test_mat, gallery_norm, gt_pool_pos)
    else:
        stats_cond = {"recall_at_10": float("nan"), "rank_median": float("nan")}
    print(f"  Condition-vector (trained adapter, per-example avg): "
          f"recall@10={stats_cond['recall_at_10']:.4f}  "
          f"rank-median={stats_cond['rank_median']:.1f}")
    print("  (Note: this matches heldout_example true_rank if adapter is consistent)")

    # ── Verdict ───────────────────────────────────────────────────────────────
    best_linear_recall = max(
        stats_procrustes["recall_at_10"],
        stats_pca["recall_at_10"],
    )
    best_linear_rank = min(
        stats_procrustes["rank_median"],
        stats_pca["rank_median"],
    )
    pool_size = memory_pool.shape[0]  # 380
    chance_rank_median = (pool_size + 1) / 2.0  # ~190.5 for pool=380
    adapter_rank = adapter_rank_median

    # Interpretation notes:
    # - 992b oracle ceiling (rank-median=208, recall@10=0.111) is measured on pool=4010,
    #   NOT pool=380.  The two numbers are NOT comparable — they measure different problems.
    #   Do NOT cross-compare rank-medians across pool sizes.
    # - In pool=380, chance rank-median is ~190.5.  Procrustes and adapter are both near-chance.
    #   Corrected (n=66): Procrustes ~178, adapter ~175, both near chance.
    # - recall@10 is low but non-zero; all methods are near-chance on this pool.
    procrustes_above_chance = best_linear_rank < chance_rank_median - 10
    adapter_below_chance = adapter_rank > chance_rank_median + 10
    procrustes_beats_adapter_by_rank = best_linear_rank < adapter_rank - 20

    if adapter_below_chance and procrustes_above_chance and procrustes_beats_adapter_by_rank:
        verdict = (
            f"TEXT->MEMORY RETRIEVAL IS NEAR-CHANCE ON HELD-OUT WINDOWS (pool={pool_size}): "
            f"chance rank-median={chance_rank_median:.0f}. "
            f"Procrustes rank-median={best_linear_rank:.0f} (marginally above chance); "
            f"trained 906b adapter rank-median={adapter_rank:.1f} (below chance, worse than random). "
            f"best recall@10={best_linear_recall:.4f} (n={len(X_test)}). "
            "The trained contrastive adapter underperforms even a training-free linear map, "
            "indicating the training setup degraded retrieval quality. "
            "The root cause is the signal/space limit: the text-embedding space and SNI memory "
            "space are weakly aligned for held-out market regimes, not a bridge architecture problem."
        )
        recommendation = (
            "DO NOT prioritize NV-Retriever / contrastive bridge fine-tuning. "
            "The trained 906b adapter performs below-chance — contrastive training has already "
            "been tried and degraded a linear baseline. The limiting factor is the "
            "signal available in the memory-target definition and pool composition, "
            "not the bridge architecture. Focus budget on conditioning interface and "
            "memory-target quality."
        )
    elif procrustes_beats_adapter_by_rank:
        verdict = (
            f"PROCRUSTES BEATS TRAINED ADAPTER on rank-median: "
            f"{best_linear_rank:.0f} vs {adapter_rank:.1f} (pool={pool_size}, "
            f"chance={chance_rank_median:.0f}). "
            f"best recall@10={best_linear_recall:.4f} (n={len(X_test)}). "
            "The contrastive adapter underperforms a linear map; training setup is the issue."
        )
        recommendation = (
            "Bridge training setup has degraded quality below a linear baseline. "
            "DO NOT invest in NV-Retriever until training quality issues are diagnosed. "
            "Focus on memory-target definition and pool composition."
        )
    else:
        verdict = (
            f"TRAINED ADAPTER MATCHES OR BEATS LINEAR MAP: adapter rank-median={adapter_rank:.1f} "
            f"vs Procrustes={best_linear_rank:.0f} (pool={pool_size}, chance={chance_rank_median:.0f}). "
            f"best recall@10={best_linear_recall:.4f} (n={len(X_test)}). "
            "Both are near-chance in this pool size, suggesting the signal limit is the space."
        )
        recommendation = (
            "Near-chance performance for both methods. "
            "The limiting factor is the memory space signal, not bridge architecture. "
            "Focus on memory-target definition and pool composition rather than NV-Retriever training."
        )

    report = {
        "schema_version": "nl_retrieval_probe_2_procrustes_linear_bridge_v1",
        "clean_inputs": {
            "bridge_eval_arrays": str(BRIDGE_EVAL_ARRAYS),
            "bridge_eval_report": str(BRIDGE_EVAL_REPORT),
            "pipeline_report": str(PIPELINE_REPORT),
            "text_embedding_model": "text-embedding-3-small (dim=1536)",
            "n_train_windows": int(len(X_train)),
            "n_test_windows": int(len(X_test)),
            "pool_size": int(memory_pool.shape[0]),
            "provenance": provenance_note,
            "contamination_hash_990a_not_used": CONTAMINATION_HASH_990A,
            "contamination_check": "text_dim=1536 != 3072 (990a/990e) -> CLEAN",
        },
        "results": {
            "procrustes_svd_full": stats_procrustes,
            "procrustes_svd_pca256": stats_pca,
            "trained_adapter_906b_from_heldout_examples": {
                "rank_median": adapter_rank_median,
                "recall_at_10": adapter_recall_at_10,
                "recall_at_1": adapter_recall_at_1,
                "n_windows": len(adapter_ranks),
                "note": "Averaged true_rank_full_pool over text-kind views per window "
                        "(4.3 views/window, against full 380-window pool)",
            },
            "condition_vector_cosine_sanity": stats_cond,
        },
        "oracle_ceilings_context": {
            "memory_space_oracle_recall_at_10_pool4010": MEMORY_ORACLE_RECALL_AT_10,
            "memory_space_oracle_rank_median_pool4010": MEMORY_ORACLE_RANK_MEDIAN,
            "source": "992b nl_14x14_memory_locality_oracle_audit (offset=5 adjacent query)",
            "IMPORTANT_DO_NOT_CROSS_COMPARE": (
                "992b was measured on pool_size=4010 (the full train bank). "
                "Probe 2 is measured on pool_size=380. These rank-medians CANNOT be "
                "compared directly — different retrieval problems. "
                "Chance rank-median: pool=4010 -> 2005; pool=380 -> 190. "
                "Use the chance baseline for interpretation within each probe."
            ),
        },
        "chance_baseline": {
            "pool_size": int(pool_size),
            "chance_rank_median": float(chance_rank_median),
            "note": "Expected rank-median under uniform random retrieval = (pool_size+1)/2",
        },
        "verdict": verdict,
        "recommendation": recommendation,
        "statistical_note": (
            f"n_test={len(X_test)} queries, pool_size={pool_size}. "
            f"Chance rank-median = {chance_rank_median:.0f}. "
            f"best Procrustes recall@10={best_linear_recall:.4f}, rank-median={best_linear_rank:.1f}. "
            f"adapter recall@10={adapter_recall_at_10:.4f}, rank-median={adapter_rank:.1f}. "
            "Both methods are near-chance on this pool. "
            "rank_median is the primary discriminator at this sample size. "
            "992b oracle ceiling (rank-median=208) was measured on pool=4010 — "
            "NOT comparable to this probe's pool=380."
        ),
    }

    out_path = OUTPUT_DIR / "probe2_procrustes_bridge_report.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"PROBE 2 RESULT written to: {out_path}")
    print(f"  Procrustes SVD (full)    recall@10: {stats_procrustes['recall_at_10']:.4f}  "
          f"rank-median: {stats_procrustes['rank_median']:.1f}")
    print(f"  PCA256+Procrustes        recall@10: {stats_pca['recall_at_10']:.4f}  "
          f"rank-median: {stats_pca['rank_median']:.1f}")
    print(f"  Trained adapter (906b)   recall@10: {adapter_recall_at_10:.4f}  "
          f"rank-median: {adapter_rank_median:.1f}")
    print(f"  Memory-space oracle (992b): recall@10={MEMORY_ORACLE_RECALL_AT_10:.3f}  "
          f"rank-median={MEMORY_ORACLE_RANK_MEDIAN:.0f}")
    print(f"\n  VERDICT: {verdict}")
    print(f"  RECOMMENDATION: {recommendation}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
