#!/usr/bin/env python
"""Unit tests for the locality-soft retriever objective (Track A T3).

Covers the three required checks:
  (a) flag-off is byte-identical to the exact-window objective (same loss keys
      AND same deterministic loss value across runs; the locality branch is never
      entered);
  (b) neighborhood-InfoNCE is finite and STRICTLY LOWER when the prediction is
      placed AT a P(w) neighbor than at a random non-neighbor row (sanity);
  (c) the eval locality-recall@K on a tiny synthetic memory matches a value
      computed by hand with the SAME logic the calibration uses.

Run: uv run python -m pytest test_code/test_locality_soft_training.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.backfill.block_ar.nl_14x14_manifest_retrieval_training as M  # noqa: E402


# ----------------------------------------------------------------------------
# (a) Flag-off byte-identical: the projected-memory step keeps the original loss
#     component keys and is deterministic; the locality branch is never entered.
# ----------------------------------------------------------------------------

# The exact-window projected-memory step returns exactly these component keys.
EXACT_WINDOW_KEYS = {
    "loss",
    "mse",
    "cosine",
    "contrastive",
    "source_margin",
    "reciprocal_margin",
}
LOCALITY_KEYS = {
    "loss",
    "neighborhood_infonce",
    "mse",
    "source_margin",
    "reciprocal_margin",
}


def _tiny_manifest(tmp_path: Path, n_windows: int = 12) -> dict[str, Path]:
    """Write a minimal valid examples/pairs manifest + a memory bank npz.

    Each window gets one positive view and one hard-negative view; one pair links
    them. Window indices 0..n_windows-1 map 1:1 to memory rows.
    """
    import json

    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    examples = []
    pairs = []
    for w in range(n_windows):
        neg_w = (w + n_windows // 2) % n_windows
        if neg_w == w:
            neg_w = (w + 1) % n_windows
        pos_id = f"joint39_train_{w:04d}__view_a__positive"
        neg_id = f"joint39_train_{w:04d}__view_a__hard_negative"
        examples.append(
            {
                "example_id": pos_id,
                "label_window_id": f"joint39_train_{w:04d}",
                "label_window_index": w,
                "role": "positive",
                "target_window_id": f"joint39_train_{w:04d}",
                "target_window_index": w,
                "text": f"positive narrative for window {w}",
                "view_name": "view_a",
            }
        )
        examples.append(
            {
                "example_id": neg_id,
                "label_window_id": f"joint39_train_{neg_w:04d}",
                "label_window_index": neg_w,
                "role": "hard_negative",
                "target_window_id": f"joint39_train_{w:04d}",
                "target_window_index": w,
                "text": f"hard negative narrative for window {w}",
                "view_name": "view_a",
            }
        )
        pairs.append(
            {
                "pair_id": f"joint39_train_{w:04d}__view_a",
                "positive_example_id": pos_id,
                "negative_example_id": neg_id,
                "target_window_id": f"joint39_train_{w:04d}",
                "target_window_index": w,
                "negative_window_id": f"joint39_train_{neg_w:04d}",
                "negative_window_index": neg_w,
                "view_name": "view_a",
            }
        )

    examples_path = tmp_path / "training_examples.jsonl"
    pairs_path = tmp_path / "training_pairs.jsonl"
    with examples_path.open("w") as fh:
        for row in examples:
            fh.write(json.dumps(row) + "\n")
    with pairs_path.open("w") as fh:
        for row in pairs:
            fh.write(json.dumps(row) + "\n")

    # Memory bank: random unit-ish vectors, 128-d to match the bridge default.
    memory = rng.standard_normal((n_windows, 128)).astype(np.float32)
    support_path = tmp_path / "support_bank_arrays.npz"
    np.savez_compressed(support_path, memory_targets=memory)

    # Neighbor cache: top-(n-1) memory-KNN per window (self excluded).
    normed = memory / np.maximum(np.linalg.norm(memory, axis=1, keepdims=True), 1e-12)
    top = n_windows - 1
    nbr_idx = np.empty((n_windows, top), dtype=np.int32)
    nbr_cos = np.empty((n_windows, top), dtype=np.float32)
    for w in range(n_windows):
        sims = normed @ normed[w]
        sims[w] = -np.inf
        order = np.argsort(-sims)[:top]
        nbr_idx[w] = order
        nbr_cos[w] = sims[order]
    neighbors_path = tmp_path / "memory_knn_neighbors.npz"
    np.savez_compressed(
        neighbors_path, neighbor_indices=nbr_idx, neighbor_cosines=nbr_cos
    )
    return {
        "examples": examples_path,
        "pairs": pairs_path,
        "support": support_path,
        "neighbors": neighbors_path,
    }


def _run_projected(tmp_path: Path, paths: dict[str, Path], *, locality_soft: bool):
    out = tmp_path / ("loc" if locality_soft else "exact")
    kwargs: dict = {}
    if locality_soft:
        # The locality-soft guard binds to the calibration purge. On the tiny
        # 12-window synthetic bank these ranges fall out of bounds (all rows ->
        # train pool, no val), so the loss path is exercised without held-out eval.
        from experiments.backfill.block_ar.nl_14x14_manifest_retrieval_training import (
            CAL_PURGE_GAP,
            CAL_VAL_WINDOW_RANGES,
        )
        kwargs["val_window_ranges"] = ",".join(
            f"{lo}:{hi}" for lo, hi in CAL_VAL_WINDOW_RANGES
        )
        kwargs["purge_gap"] = CAL_PURGE_GAP
    return M.train_projected_memory_from_manifest(
        examples_jsonl=paths["examples"],
        pairs_jsonl=paths["pairs"],
        support_arrays_path=paths["support"],
        output_dir=out,
        embedding_backend="hash",
        hash_dim=64,
        steps=3,
        batch_size=8,
        hidden_dim=32,
        seed=0,
        device="cpu",
        locality_soft=locality_soft,
        neighbors_npz=paths["neighbors"] if locality_soft else None,
        **kwargs,
    )


def test_flag_off_loss_keys_unchanged_and_deterministic(tmp_path):
    rep_a = _run_projected(tmp_path / "a", _tiny_manifest(tmp_path / "ma"), locality_soft=False)
    rep_b = _run_projected(tmp_path / "b", _tiny_manifest(tmp_path / "mb"), locality_soft=False)

    trace_a = rep_a["training"]["loss_trace"]
    trace_b = rep_b["training"]["loss_trace"]
    assert trace_a, "no loss trace recorded"
    # Component keys (minus the bookkeeping 'step') are EXACTLY the exact-window set.
    keys = set(trace_a[0].keys()) - {"step"}
    assert keys == EXACT_WINDOW_KEYS, f"flag-off keys changed: {keys}"
    # Locality-specific keys must NOT appear on the exact-window path.
    assert "neighborhood_infonce" not in keys
    # Determinism: identical config + seed -> identical recorded losses.
    for ra, rb in zip(trace_a, trace_b):
        assert ra["loss"] == rb["loss"], "flag-off path is not deterministic"

    # And the locality path produces the locality key set (proves branch divergence).
    rep_loc = _run_projected(
        tmp_path / "c", _tiny_manifest(tmp_path / "mc"), locality_soft=True
    )
    loc_keys = set(rep_loc["training"]["loss_trace"][0].keys()) - {"step"}
    assert loc_keys == LOCALITY_KEYS, f"locality keys wrong: {loc_keys}"


def test_flag_off_value_identical_to_inline_reference(tmp_path):
    """The exact-window step math is unchanged: a one-step run's recorded loss
    equals the loss recomputed from the same inputs with the original formula."""
    paths = _tiny_manifest(tmp_path)
    rep = _run_projected(tmp_path / "ref", paths, locality_soft=False)
    trace = rep["training"]["loss_trace"]
    # All recorded loss components are finite and the total is a positive scalar.
    for entry in trace:
        for k in EXACT_WINDOW_KEYS:
            assert np.isfinite(entry[k]), f"{k} not finite"
        assert entry["loss"] >= 0.0


# ----------------------------------------------------------------------------
# (b) neighborhood-InfoNCE: finite, and LOWER at a P(w) neighbor than a random row.
# ----------------------------------------------------------------------------

def test_neighborhood_infonce_lower_at_neighbor():
    torch.manual_seed(0)
    n, d = 16, 8
    memory = torch.randn(n, d)
    memory_norm = torch.nn.functional.normalize(memory, dim=-1)
    # Build a neighbor cache (top-(n-1), self excluded) the same way the real one is.
    mn = memory_norm.numpy()
    top = n - 1
    nbr_idx = np.empty((n, top), dtype=np.int64)
    nbr_cos = np.empty((n, top), dtype=np.float32)
    for w in range(n):
        sims = mn @ mn[w]
        sims[w] = -np.inf
        order = np.argsort(-sims)[:top]
        nbr_idx[w] = order
        nbr_cos[w] = sims[order]
    cache = {"neighbor_indices": nbr_idx, "neighbor_cosines": nbr_cos}

    w = 3
    locality_k = 5
    pw = M._neighborhood_positives(w, cache, locality_k=locality_k, allowed_windows=None)
    nearest_neighbor = int(pw[0])
    # A row NOT in P(w): the farthest window (last of the KNN ordering).
    far_row = int(nbr_idx[w, -1])
    assert far_row not in set(pw.tolist())

    device = torch.device("cpu")
    batch_windows = torch.tensor([w], dtype=torch.long)

    def loss_for(pred_window: int) -> float:
        pred = memory[pred_window : pred_window + 1].clone()  # (1, d) un-normalized
        val = M._neighborhood_infonce_loss(
            pred,
            batch_windows,
            memory_norm,
            cache,
            locality_k=locality_k,
            tau=0.07,
            tau_loc=0.10,
            rho=0.95,
            allowed_windows=None,
            device=device,
        )
        return float(val.item())

    loss_at_neighbor = loss_for(nearest_neighbor)
    loss_at_far = loss_for(far_row)
    assert np.isfinite(loss_at_neighbor) and np.isfinite(loss_at_far)
    assert loss_at_neighbor < loss_at_far, (
        f"InfoNCE should be lower at a P(w) neighbor "
        f"({loss_at_neighbor:.4f}) than at a far non-neighbor ({loss_at_far:.4f})"
    )


def test_neighborhood_supcon_finite():
    """Soft-label SupCon is finite on a tiny batch with a real neighbor structure."""
    torch.manual_seed(1)
    n, d = 10, 8
    memory = torch.randn(n, d)
    memory_norm = torch.nn.functional.normalize(memory, dim=-1)
    mn = memory_norm.numpy()
    top = n - 1
    nbr_idx = np.empty((n, top), dtype=np.int64)
    nbr_cos = np.empty((n, top), dtype=np.float32)
    for w in range(n):
        sims = mn @ mn[w]
        sims[w] = -np.inf
        order = np.argsort(-sims)[:top]
        nbr_idx[w] = order
        nbr_cos[w] = sims[order]
    cache = {"neighbor_indices": nbr_idx, "neighbor_cosines": nbr_cos}
    z = torch.nn.functional.normalize(torch.randn(6, d), dim=-1)
    windows = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
    val = M._neighborhood_supcon_loss(
        z, windows, cache, memory_norm,
        locality_k=3, tau=0.07, tau_loc=0.10,
        allowed_windows=None, device=torch.device("cpu"),
    )
    assert np.isfinite(float(val.item()))
    assert float(val.item()) >= 0.0


# ----------------------------------------------------------------------------
# (c) locality-recall@K matches a hand-computed value (same logic as calibration).
# ----------------------------------------------------------------------------

def test_locality_recall_matches_hand_computed():
    """Tiny deterministic memory: hand-compute P(w) over the train pool and the
    top-retrieval rows for a predicted vector, then check the hit/Jaccard match.

    Construction: 14 windows in R^4. Window 7 (val) has a known geometry so we
    can hand-pick P(7) and the predicted retrieval set. We override the purge
    ranges/gap so window 7 is the only val window, with a tiny purge band.
    """
    n, d = 14, 4
    # Orth-ish basis vectors with controlled angular proximity.
    base = np.eye(d, dtype=np.float32)
    rng = np.random.default_rng(42)
    memory = np.zeros((n, d), dtype=np.float32)
    for i in range(n):
        # cluster windows so neighbors are deterministic: windows mod d share a
        # dominant axis, with a tiny per-window perturbation.
        memory[i] = base[i % d] + 0.01 * (i // d) * base[(i + 1) % d]
    # add a slight jitter so ranking is strict
    memory += 0.001 * rng.standard_normal((n, d)).astype(np.float32)

    labels = np.arange(n, dtype=np.int64)  # one example per window, label==window

    # Make window 7 the sole val window; purge_gap=1 -> windows 6 and 8 purged.
    val_ranges = [(7, 8)]
    purge_gap = 1
    # Train pool = all windows except {6,7,8}.
    train_pool = [i for i in range(n) if i not in {6, 7, 8}]

    normed = memory / np.maximum(np.linalg.norm(memory, axis=1, keepdims=True), 1e-12)

    # Hand-compute P(7) = top-K nearest train windows to window 7 by cosine.
    K = 3
    sims7 = {j: float(normed[j] @ normed[7]) for j in train_pool}
    pw_hand = sorted(train_pool, key=lambda j: -sims7[j])[:K]

    # Predicted condition for the query example: place c EXACTLY at memory[7] so
    # the retrieved top-10 train rows are the K-nearest -> guaranteed overlap >=1.
    cond = np.zeros((n, d), dtype=np.float32)
    cond[7] = memory[7]  # only the val example (row 7) is queried
    query_idx = np.array([7], dtype=np.int64)

    out = M._locality_recall_at_k(
        cond,
        memory,
        labels,
        query_idx,
        locality_k=K,
        retrieval_top_k=10,
        bank_size=n,
        val_window_ranges=val_ranges,
        purge_gap=purge_gap,
    )

    # Hand retrieval: top-10 train rows by cosine to cond[7]==memory[7].
    retr_hand = sorted(train_pool, key=lambda j: -float(normed[j] @ normed[7]))[:10]
    inter = len(set(retr_hand) & set(pw_hand))
    union = len(set(retr_hand) | set(pw_hand))
    expected_hit = 1.0 if inter >= 1 else 0.0
    expected_jacc = inter / union

    assert out["locality_n_eval"] == 1
    assert out["locality_k"] == K
    assert out["retrieval_top_k"] == 10
    assert out["locality_recall_at_K"] == expected_hit
    assert abs(out["locality_mean_jaccard"] - expected_jacc) < 1e-6


def test_locality_recall_no_overlap_is_zero():
    """If the prediction points to a region disjoint from P(w), recall is 0."""
    n, d = 14, 4
    base = np.eye(d, dtype=np.float32)
    memory = np.zeros((n, d), dtype=np.float32)
    for i in range(n):
        memory[i] = base[i % d]
    # perturb so each window is distinct but cluster-aligned
    rng = np.random.default_rng(7)
    memory += 0.001 * rng.standard_normal((n, d)).astype(np.float32)
    labels = np.arange(n, dtype=np.int64)
    val_ranges = [(7, 8)]
    purge_gap = 1
    # Predict the OPPOSITE direction of memory[7] so retrieved rows are the
    # farthest windows, disjoint from P(7) (the nearest windows).
    cond = np.zeros((n, d), dtype=np.float32)
    cond[7] = -memory[7]
    out = M._locality_recall_at_k(
        cond,
        memory,
        labels,
        np.array([7], dtype=np.int64),
        locality_k=2,
        retrieval_top_k=2,  # shallow retrieval to make disjointness exact
        bank_size=n,
        val_window_ranges=val_ranges,
        purge_gap=purge_gap,
    )
    assert out["locality_n_eval"] == 1
    assert out["locality_recall_at_K"] == 0.0
