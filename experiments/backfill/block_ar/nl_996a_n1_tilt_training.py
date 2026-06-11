#!/usr/bin/env python
"""996a P4 (N1): teacher-label processing + learned within-pool tilt student.

Pre-registered design (P4 compass; decisive experiment for the within-pool
tilt family that the 994b oracle kept alive):

Stage 1 -- teacher label processing over the 995d train-pool replay labels
(1,000 train queries x 50 causal pool candidates, engine-equivalent solo
replay CRPS vs realized train futures):
  1. CALM-DEBIAS (F1/F3): residualize ``replay_crps`` on
     ``cand_future_activity`` with ONE global linear fit across all 50k rows;
     the debiased residual is the teacher signal. A per-query residualization
     is reported as a sensitivity only.
  2. NOISE-BANDING: within each query's pool, pairwise contrasts are kept only
     where ``|debiased delta| > max(0.048, 0.5 * query_pool_replay_crps_std)``
     (twin-noise floor; per-query band, recorded).
  3. Teacher sanity: Spearman(debiased, activity) ~ 0 after debiasing;
     Spearman(raw replay, activity) reported for reference (~+0.7 per F1).

Stage 2 -- student. Features per (query, candidate) are DEPLOYABLE ONLY (no
candidate-future-derived features, no realized-future anything):
  f1 ``text_cosine``       cos(query full_professional embedding, candidate
                           full_professional embedding); OpenAI
                           text-embedding-3-large vectors REUSED from the 982g
                           embedding-bridge cache (content-hash verified; no
                           re-embedding).
  f2 ``start_distance_z``  within-pool z-score of the candidate start distance
                           (994b z-scaled terminal-state RMS distance).
  f3 ``prefix_delta_cos``  cosine of z-scaled 30-day terminal deltas
                           (history_raw[w,-1,:] - history_raw[w,0,:], per-dim
                           scale = std over the FULL causal train bank
                           0..4009). CHOICE DOCUMENTED: 30-day history_level
                           terminal deltas from the 939a bank (not the
                           normalized history feature stack) -- the terminal
                           delta is the same object the conflict detector and
                           prefix_terminal_rows reason about.
  The optional grounded direction-match count feature is SKIPPED (recorded):
  it would add a card-parsing axis to the decisive run; the three core
  features keep one-axis-per-iteration discipline.
  EXCLUDED by pre-registration: cand_future_activity (calm shortcut), raw
  index gap (recency shortcut).

Student: linear logistic pairwise ranker (L2 grid) and a small 2-layer MLP
pairwise ranker (RankNet loss) trained on banded pairs; 5-fold query-side CV
with CONTIGUOUS window blocks (no shuffle); 3 seeds; selection on held-out
pairwise margin accuracy + within-pool rank correlation with the debiased
teacher.

Tilt-weight pre-registration: with the CV-selected student, the chassis
combined score ``start_only_score + tilt_weight * tilt_z`` is simulated on
held-out folds against the replay-CRPS proxy (softmax-weighted top-3 solo
replay CRPS, chassis temperature sqrt(D) so the neutral tilt reproduces the
994a start-only weights exactly); ONE tilt_weight is selected and frozen
BEFORE any val-frame outcome is seen.

Artifacts -> nl_scenario_demo_outputs/n1_tilt_training_996a/.
No OpenAI calls in this script (embeddings are cache-only).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_episode_narrative_bridge_report import (  # noqa: E402
    _load_json,
    _query_text,
    _read_jsonl,
    _softmax_weights,
    _split_indices_from_support_report,
    _window_index,
)
from experiments.backfill.block_ar.nl_episode_narrative_embedding_bridge_report import (  # noqa: E402
    _candidate_view_rows,
    _hash_texts,
)
from experiments.backfill.block_ar.nl_episode_narrative_retrieval import (  # noqa: E402
    DEFAULT_VIEW_NAMES,
)

LABELS_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "train_pool_replay_labels_995d"
)
OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/n1_tilt_training_996a"
)
CORPUS_CARDS_JSONL = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_full_codex_multiformat_982g_sharded/final/"
    "multiformat_episode_cards.jsonl"
)
SUPPORT_BANK_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_bank_train_all_939a"
)
EMBED_CACHE_RUN_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_full_codex_982g_embedding_bridge_hybrid_66q"
)
EMBEDDING_MODEL = "text-embedding-3-large"

NOISE_FLOOR = 0.048
BAND_STD_FRACTION = 0.5
POOL_SIZE = 50
TOP_K = 3
FEATURE_NAMES = ("text_cosine", "start_distance_z", "prefix_delta_cos")
# Grid extended upward (24..256) after the first full CV pass put the optimum
# at the original boundary (16); extension done strictly BEFORE any val-frame
# outcome was generated or seen (CV-side selection only).
TILT_WEIGHT_GRID = (
    0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0,
    24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0, 256.0,
)
SEEDS = (0, 1, 2)
N_FOLDS = 5


def _resolve(path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(np.asarray(a), np.asarray(b)).statistic)


# ---------------------------------------------------------------------------
# Stage 1: teacher label processing (unit-tested pure functions).
# ---------------------------------------------------------------------------


def load_replay_labels(labels_dir: Path) -> dict[str, np.ndarray]:
    """Concatenate the 995d label shards in shard order."""

    shard_paths = sorted(_resolve(labels_dir).glob("labels_shard_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"no label shards under {labels_dir}")
    parts: dict[str, list[np.ndarray]] = {}
    for path in shard_paths:
        with np.load(path) as payload:
            for key in payload.files:
                parts.setdefault(key, []).append(payload[key].copy())
    out = {key: np.concatenate(arrs, axis=0) for key, arrs in parts.items()}
    n = int(out["query_window"].shape[0])
    if out["replay_crps"].shape != (n, POOL_SIZE):
        raise ValueError(f"unexpected replay_crps shape {out['replay_crps'].shape}")
    order = np.argsort(out["query_window"], kind="stable")
    return {key: arr[order] for key, arr in out.items()}


def global_calm_debias(
    replay_crps: np.ndarray, activity: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    """Residualize replay CRPS on candidate future activity (ONE global OLS).

    Returns (residual, stats). residual = replay - (a + b * activity).
    """

    y = np.asarray(replay_crps, dtype=np.float64).reshape(-1)
    x = np.asarray(activity, dtype=np.float64).reshape(-1)
    if y.shape != x.shape:
        raise ValueError("replay/activity shape mismatch")
    design = np.stack([np.ones_like(x), x], axis=1)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coef
    residual = (y - fitted).reshape(np.asarray(replay_crps).shape)
    corr = float(np.corrcoef(x, y)[0, 1])
    stats = {
        "intercept": float(coef[0]),
        "slope": float(coef[1]),
        "pearson_raw_vs_activity": corr,
        "r2": float(corr**2),
        "n_rows": int(y.size),
    }
    return residual, stats


def per_query_calm_debias(
    replay_crps: np.ndarray, activity: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    """Sensitivity variant: one OLS per query within its own 50-candidate pool."""

    y = np.asarray(replay_crps, dtype=np.float64)
    x = np.asarray(activity, dtype=np.float64)
    if y.ndim != 2 or y.shape != x.shape:
        raise ValueError("expected matching 2-D (n_queries, pool) arrays")
    residual = np.zeros_like(y)
    slopes: list[float] = []
    for qi in range(y.shape[0]):
        design = np.stack([np.ones_like(x[qi]), x[qi]], axis=1)
        coef, *_ = np.linalg.lstsq(design, y[qi], rcond=None)
        residual[qi] = y[qi] - design @ coef
        slopes.append(float(coef[1]))
    stats = {
        "slope_mean": float(np.mean(slopes)),
        "slope_median": float(np.median(slopes)),
        "slope_p10": float(np.quantile(slopes, 0.10)),
        "slope_p90": float(np.quantile(slopes, 0.90)),
    }
    return residual, stats


def noise_bands(
    pool_replay_std: np.ndarray,
    *,
    floor: float = NOISE_FLOOR,
    std_fraction: float = BAND_STD_FRACTION,
) -> np.ndarray:
    """Per-query twin-noise band: max(floor, std_fraction * pool replay std)."""

    std = np.asarray(pool_replay_std, dtype=np.float64).reshape(-1)
    return np.maximum(float(floor), float(std_fraction) * std)


def banded_pairs(
    residuals: np.ndarray, band: float
) -> tuple[np.ndarray, np.ndarray]:
    """Within-pool pairs whose |debiased delta| exceeds the noise band.

    Returns (better_idx, worse_idx): residual[better] + band < residual[worse]
    is NOT the rule -- the rule is |delta| > band, oriented so the first index
    is the lower-residual (better) candidate.
    """

    r = np.asarray(residuals, dtype=np.float64).reshape(-1)
    i_idx, j_idx = np.triu_indices(r.size, k=1)
    delta = r[i_idx] - r[j_idx]
    keep = np.abs(delta) > float(band)
    i_keep, j_keep, d_keep = i_idx[keep], j_idx[keep], delta[keep]
    better = np.where(d_keep < 0.0, i_keep, j_keep)
    worse = np.where(d_keep < 0.0, j_keep, i_keep)
    return better.astype(np.int64), worse.astype(np.int64)


# ---------------------------------------------------------------------------
# Embedding reuse (982g cache; content-hash verified; no API calls).
# ---------------------------------------------------------------------------


def load_corpus_full_professional_embeddings(
    *,
    cards_jsonl: Path = CORPUS_CARDS_JSONL,
    support_report_path: Path | None = None,
    cache_run_dir: Path = EMBED_CACHE_RUN_DIR,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """full_professional text-embedding-3-large vectors for bank windows 0..4009.

    Reconstructs the EXACT text list of the cached 982g embedding-bridge run
    (3,944 train cards x 10 views + 66 query texts), verifies the content hash
    against the cache filename, loads the cached normalized matrix, and slices
    the full_professional rows. Raises rather than calling the OpenAI API.
    """

    support_report = _load_json(
        _resolve(
            support_report_path
            if support_report_path is not None
            else SUPPORT_BANK_DIR / "support_bank_report.json"
        )
    )
    cards = _read_jsonl(_resolve(cards_jsonl))
    train_indices, test_indices = _split_indices_from_support_report(support_report)
    card_by_index = {_window_index(card): card for card in cards}
    train_cards = [card_by_index[i] for i in train_indices if i in card_by_index]
    query_cards = [card_by_index[i] for i in test_indices if i in card_by_index]
    candidate_rows, candidate_texts = _candidate_view_rows(
        train_cards, DEFAULT_VIEW_NAMES
    )
    query_texts = [_query_text(card) for card in query_cards]
    all_texts = candidate_texts + query_texts
    text_hash = _hash_texts(all_texts)
    safe_model = "".join(ch if ch.isalnum() else "_" for ch in EMBEDDING_MODEL)
    cache_path = (
        _resolve(cache_run_dir) / "embedding_cache" / f"openai_{safe_model}_{text_hash}.npz"
    )
    if not cache_path.exists():
        raise FileNotFoundError(
            f"982g embedding cache not found for reconstructed text hash "
            f"{text_hash}: {cache_path} -- refusing to re-embed 4,010 cards"
        )
    with np.load(cache_path) as payload:
        embeddings = payload["embeddings"].copy().astype(np.float32)
    if embeddings.shape[0] != len(all_texts):
        raise ValueError(
            f"cache rows {embeddings.shape[0]} != reconstructed texts "
            f"{len(all_texts)}"
        )
    window_to_vec: dict[int, np.ndarray] = {}
    for row_pos, row in enumerate(candidate_rows):
        if str(row["view"]) != "full_professional":
            continue
        window_to_vec[int(row["window_index"])] = embeddings[row_pos]
    for query_pos, card in enumerate(query_cards):
        views = card.get("views", {})
        text = views.get("full_professional", "")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"query card {card.get('window_id')} lacks full_professional; "
                "_query_text fallback would mix views"
            )
        window_to_vec[_window_index(card)] = embeddings[len(candidate_texts) + query_pos]
    windows = np.asarray(sorted(window_to_vec), dtype=np.int64)
    matrix = np.stack([window_to_vec[int(w)] for w in windows], axis=0)
    norms = np.linalg.norm(matrix, axis=1)
    if not np.all(np.isfinite(matrix)) or float(np.min(norms)) <= 1e-6:
        raise ValueError("corpus embeddings contain non-finite or zero rows")
    meta = {
        "source_cache": str(cache_path),
        "text_hash": text_hash,
        "embedding_model": EMBEDDING_MODEL,
        "view": "full_professional",
        "n_windows": int(windows.size),
        "embedding_dim": int(matrix.shape[1]),
        "normalized": True,
        "api_calls": 0,
    }
    return windows, matrix, meta


# ---------------------------------------------------------------------------
# Deployable feature builders (shared with 996b).
# ---------------------------------------------------------------------------


def prefix_delta_z(
    history_raw: np.ndarray, fit_indices: np.ndarray
) -> np.ndarray:
    """z-scaled 30-day terminal deltas; per-dim scale from fit windows only."""

    history = np.asarray(history_raw, dtype=np.float64)
    delta = history[:, -1, :] - history[:, 0, :]
    fit = np.asarray(fit_indices, dtype=np.int64)
    scale = np.maximum(delta[fit].std(axis=0), 1e-8)
    return (delta / scale[None, :]).astype(np.float64)


def chassis_terminal_z(
    history_raw: np.ndarray, fit_indices: np.ndarray
) -> np.ndarray:
    """Per-dim z-scored terminal states exactly as the P3 chassis computes them.

    Mirrors ``start_distances_to_query_start``: mean/std over fit windows,
    std floored at 1e-6; chassis start_only_score = -L2 norm of z differences.
    """

    history = np.asarray(history_raw, dtype=np.float64)
    terminal = history[:, -1, :]
    fit = np.asarray(fit_indices, dtype=np.int64)
    mean = terminal[fit].mean(axis=0, keepdims=True)
    std = np.maximum(terminal[fit].std(axis=0, keepdims=True), 1e-6)
    return (terminal - mean) / std


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine between two equally shaped 2-D arrays."""

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.maximum(
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12
    )
    return np.sum(a * b, axis=1) / denom


def within_pool_z(values: np.ndarray) -> np.ndarray:
    """z-score along the last (pool) axis; degenerate pools map to zeros."""

    v = np.asarray(values, dtype=np.float64)
    mean = v.mean(axis=-1, keepdims=True)
    std = v.std(axis=-1, keepdims=True)
    return np.where(std > 1e-12, (v - mean) / np.maximum(std, 1e-12), 0.0)


def build_candidate_features(
    *,
    query_windows: np.ndarray,
    pool_windows: np.ndarray,
    pool_start_distance: np.ndarray,
    embedding_windows: np.ndarray,
    embeddings: np.ndarray,
    prefix_z: np.ndarray,
    query_embeddings: np.ndarray | None = None,
) -> np.ndarray:
    """Feature tensor (n_queries, pool, 3): text_cosine / start_z / prefix_cos.

    ``query_embeddings`` overrides corpus lookup for the query side (val-frame
    queries whose narratives are NOT in the train corpus); candidates are
    always looked up in the corpus embedding bank.
    """

    queries = np.asarray(query_windows, dtype=np.int64)
    pools = np.asarray(pool_windows, dtype=np.int64)
    n_q, pool_n = pools.shape
    emb_pos = {int(w): i for i, w in enumerate(np.asarray(embedding_windows))}
    cand_pos = np.asarray(
        [[emb_pos[int(c)] for c in pools[qi]] for qi in range(n_q)], dtype=np.int64
    )
    if query_embeddings is None:
        q_vecs = embeddings[
            np.asarray([emb_pos[int(q)] for q in queries], dtype=np.int64)
        ]
    else:
        q_vecs = np.asarray(query_embeddings, dtype=np.float32)
        if q_vecs.shape[0] != n_q:
            raise ValueError("query_embeddings rows must match query_windows")
    q_unit = q_vecs / np.maximum(
        np.linalg.norm(q_vecs, axis=1, keepdims=True), 1e-12
    )
    text_cos = np.einsum(
        "qpd,qd->qp", embeddings[cand_pos].astype(np.float64), q_unit.astype(np.float64)
    )
    start_z = within_pool_z(np.asarray(pool_start_distance, dtype=np.float64))
    pref_cos = np.zeros((n_q, pool_n), dtype=np.float64)
    for qi in range(n_q):
        q_row = prefix_z[queries[qi]][None, :]
        pref_cos[qi] = cosine_rows(
            prefix_z[pools[qi]], np.repeat(q_row, pool_n, axis=0)
        )
    return np.stack([text_cos, start_z, pref_cos], axis=-1)


# ---------------------------------------------------------------------------
# Students.
# ---------------------------------------------------------------------------


class LinearPairwiseRanker:
    """Logistic pairwise ranker on banded feature differences (no intercept)."""

    def __init__(self, *, c: float = 1.0, seed: int = 0) -> None:
        self.c = float(c)
        self.seed = int(seed)
        self.weights: np.ndarray | None = None

    def fit(self, pair_diffs: np.ndarray) -> "LinearPairwiseRanker":
        """pair_diffs rows are f(better) - f(worse); the label is always 1.

        Symmetrized to (X, 1) + (-X, 0) so no intercept is needed; solved with
        deterministic L-BFGS (sklearn LogisticRegression).
        """

        from sklearn.linear_model import LogisticRegression

        x = np.asarray(pair_diffs, dtype=np.float64)
        design = np.concatenate([x, -x], axis=0)
        labels = np.concatenate(
            [np.ones(x.shape[0]), np.zeros(x.shape[0])], axis=0
        )
        model = LogisticRegression(
            penalty="l2",
            C=self.c,
            fit_intercept=False,
            solver="lbfgs",
            max_iter=2000,
            random_state=self.seed,
        )
        model.fit(design, labels)
        self.weights = model.coef_.reshape(-1).astype(np.float64)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("fit first")
        return np.asarray(features, dtype=np.float64) @ self.weights

    def export(self) -> dict[str, Any]:
        return {
            "model_type": "linear_logistic_pairwise",
            "c": self.c,
            "seed": self.seed,
            "weights": [float(w) for w in (self.weights if self.weights is not None else [])],
            "feature_names": list(FEATURE_NAMES),
        }


class MLPPairwiseRanker:
    """2-layer MLP scorer trained with the RankNet loss on banded pairs."""

    def __init__(
        self,
        *,
        hidden: int = 16,
        epochs: int = 300,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.net: torch.nn.Module | None = None

    def _build(self, n_features: int) -> torch.nn.Module:
        torch.manual_seed(self.seed)
        return torch.nn.Sequential(
            torch.nn.Linear(n_features, self.hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden, 1),
        ).to(self.device)

    def fit(
        self, better_features: np.ndarray, worse_features: np.ndarray
    ) -> "MLPPairwiseRanker":
        fb = torch.as_tensor(
            np.asarray(better_features, dtype=np.float32), device=self.device
        )
        fw = torch.as_tensor(
            np.asarray(worse_features, dtype=np.float32), device=self.device
        )
        self.net = self._build(fb.shape[1])
        optim = torch.optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        for _ in range(self.epochs):
            optim.zero_grad()
            margin = self.net(fb).reshape(-1) - self.net(fw).reshape(-1)
            loss = torch.nn.functional.softplus(-margin).mean()
            loss.backward()
            optim.step()
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("fit first")
        with torch.no_grad():
            out = self.net(
                torch.as_tensor(
                    np.asarray(features, dtype=np.float32), device=self.device
                )
            )
        return out.reshape(-1).cpu().numpy().astype(np.float64)

    def export(self) -> dict[str, Any]:
        if self.net is None:
            raise RuntimeError("fit first")
        state = {
            key: value.cpu().numpy().tolist()
            for key, value in self.net.state_dict().items()
        }
        return {
            "model_type": "mlp_pairwise_ranknet",
            "hidden": self.hidden,
            "epochs": self.epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "state_dict": state,
            "feature_names": list(FEATURE_NAMES),
        }


def load_student(spec: dict[str, Any], *, device: str = "cpu"):
    """Rebuild a trained student from its exported spec (used by 996b)."""

    model_type = str(spec.get("model_type", ""))
    if model_type == "linear_logistic_pairwise":
        student = LinearPairwiseRanker(c=float(spec["c"]), seed=int(spec["seed"]))
        student.weights = np.asarray(spec["weights"], dtype=np.float64)
        return student
    if model_type == "mlp_pairwise_ranknet":
        student = MLPPairwiseRanker(
            hidden=int(spec["hidden"]),
            epochs=int(spec["epochs"]),
            lr=float(spec["lr"]),
            weight_decay=float(spec["weight_decay"]),
            seed=int(spec["seed"]),
            device=device,
        )
        student.net = student._build(len(spec["feature_names"]))
        state = {
            key: torch.as_tensor(np.asarray(value, dtype=np.float32))
            for key, value in spec["state_dict"].items()
        }
        student.net.load_state_dict(state)
        return student
    raise ValueError(f"unknown student model_type: {model_type!r}")


# ---------------------------------------------------------------------------
# Tilt selection simulation (chassis-equivalent; unit-tested).
# ---------------------------------------------------------------------------


def simulate_tilt_selection(
    *,
    pool_windows: np.ndarray,
    chassis_start_scores: np.ndarray,
    tilt_scores_z: np.ndarray,
    tilt_weight: float,
    chassis_temperature: float,
    top_k: int = TOP_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Replicate _start_pool_text_tilt_selection on a precomputed pool.

    combined = start_only_score + tilt_weight * tilt; top-k by combined
    (ties -> lower window index, matching the chassis stable sort over the
    locality-ordered pool); weights = softmax(combined / T) over the selected.
    Returns (selected positions within the pool, weights).
    """

    windows = np.asarray(pool_windows, dtype=np.int64)
    combined = np.asarray(chassis_start_scores, dtype=np.float64) + float(
        tilt_weight
    ) * np.asarray(tilt_scores_z, dtype=np.float64)
    locality_order = np.lexsort((windows, -np.asarray(chassis_start_scores)))
    pos_sorted = locality_order[
        np.argsort(-combined[locality_order], kind="stable")
    ]
    selected = pos_sorted[: int(top_k)]
    shifted = (combined[selected] - np.max(combined[selected])) / max(
        float(chassis_temperature), 1e-8
    )
    weights = np.exp(shifted)
    weights = weights / weights.sum()
    return selected.astype(np.int64), weights.astype(np.float64)


def replay_proxy_crps(
    replay_crps_pool: np.ndarray,
    selected_positions: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Weighted mean of solo replay CRPS over the selected supports."""

    r = np.asarray(replay_crps_pool, dtype=np.float64)
    return float(
        np.sum(r[np.asarray(selected_positions, dtype=np.int64)] * weights)
    )


# ---------------------------------------------------------------------------
# CV machinery.
# ---------------------------------------------------------------------------


def contiguous_folds(n_queries: int, n_folds: int) -> list[np.ndarray]:
    """Contiguous window-block folds over queries already sorted by window."""

    edges = np.linspace(0, n_queries, n_folds + 1).astype(int)
    return [
        np.arange(edges[i], edges[i + 1], dtype=np.int64)
        for i in range(n_folds)
        if edges[i + 1] > edges[i]
    ]


def pair_dataset(
    features: np.ndarray,
    residual: np.ndarray,
    bands: np.ndarray,
    query_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(better_features, worse_features, pair_query_position) over banded pairs."""

    better_rows: list[np.ndarray] = []
    worse_rows: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    for qi in np.asarray(query_positions, dtype=np.int64):
        b_idx, w_idx = banded_pairs(residual[qi], float(bands[qi]))
        if b_idx.size == 0:
            continue
        better_rows.append(features[qi, b_idx])
        worse_rows.append(features[qi, w_idx])
        owners.append(np.full(b_idx.size, qi, dtype=np.int64))
    if not better_rows:
        return (
            np.zeros((0, features.shape[-1])),
            np.zeros((0, features.shape[-1])),
            np.zeros((0,), dtype=np.int64),
        )
    return (
        np.concatenate(better_rows, axis=0),
        np.concatenate(worse_rows, axis=0),
        np.concatenate(owners, axis=0),
    )


def fold_metrics(
    student,
    *,
    features: np.ndarray,
    residual: np.ndarray,
    bands: np.ndarray,
    query_positions: np.ndarray,
) -> dict[str, float]:
    """Held-out pairwise margin accuracy + within-pool Spearman vs teacher."""

    fb, fw, _ = pair_dataset(features, residual, bands, query_positions)
    if fb.shape[0] == 0:
        return {"pair_accuracy": float("nan"), "mean_spearman": float("nan"), "n_pairs": 0}
    margin = student.score(fb) - student.score(fw)
    accuracy = float(np.mean(margin > 0.0))
    rhos: list[float] = []
    for qi in np.asarray(query_positions, dtype=np.int64):
        scores = student.score(features[qi])
        rhos.append(_spearman(scores, -residual[qi]))
    return {
        "pair_accuracy": accuracy,
        "mean_spearman": float(np.nanmean(rhos)),
        "n_pairs": int(fb.shape[0]),
    }


def train_student(
    config: dict[str, Any],
    *,
    features: np.ndarray,
    residual: np.ndarray,
    bands: np.ndarray,
    query_positions: np.ndarray,
    device: str,
):
    fb, fw, _ = pair_dataset(features, residual, bands, query_positions)
    if fb.shape[0] == 0:
        raise ValueError("no banded training pairs")
    if config["model"] == "linear":
        student = LinearPairwiseRanker(c=float(config["c"]), seed=int(config["seed"]))
        student.fit(fb - fw)
    elif config["model"] == "mlp":
        student = MLPPairwiseRanker(
            hidden=int(config["hidden"]),
            epochs=int(config["epochs"]),
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
            seed=int(config["seed"]),
            device=device,
        )
        student.fit(fb, fw)
    else:
        raise ValueError(f"unknown model {config['model']!r}")
    return student


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--max-queries", type=int, default=0, help="truncate query list (smoke)"
    )
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--mlp-epochs", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="5 queries, 1 fold")
    args = parser.parse_args(argv)

    t_start = time.time()
    device = (
        str(args.device)
        if torch.cuda.is_available() or str(args.device) == "cpu"
        else "cpu"
    )
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load teacher labels.
    labels = load_replay_labels(args.labels_dir)
    n_total = int(labels["query_window"].shape[0])
    keep = n_total
    if bool(args.smoke):
        keep = min(5, n_total)
    elif int(args.max_queries) > 0:
        keep = min(int(args.max_queries), n_total)
    labels = {key: arr[:keep] for key, arr in labels.items()}
    n_q = int(labels["query_window"].shape[0])
    n_folds = 2 if bool(args.smoke) else int(args.n_folds)
    print(f"labels loaded: {n_q}/{n_total} queries x {POOL_SIZE} candidates", flush=True)

    replay = np.asarray(labels["replay_crps"], dtype=np.float64)
    activity = np.asarray(labels["cand_future_activity"], dtype=np.float64)
    pool_windows = np.asarray(labels["pool_candidate_window"], dtype=np.int64)
    query_windows = np.asarray(labels["query_window"], dtype=np.int64)
    start_distance = np.asarray(labels["cand_start_distance"], dtype=np.float64)

    # ---- Stage 1: calm-debias + banding + sanity stats.
    residual, debias_stats = global_calm_debias(replay, activity)
    residual_pq, per_query_stats = per_query_calm_debias(replay, activity)
    bands = noise_bands(labels["query_pool_replay_crps_std"])

    raw_rho = _spearman(replay.reshape(-1), activity.reshape(-1))
    debiased_rho = _spearman(residual.reshape(-1), activity.reshape(-1))
    pq_rho = _spearman(residual_pq.reshape(-1), activity.reshape(-1))
    per_q_raw = [_spearman(replay[qi], activity[qi]) for qi in range(n_q)]
    per_q_deb = [_spearman(residual[qi], activity[qi]) for qi in range(n_q)]
    agreement = [
        _spearman(residual[qi], residual_pq[qi]) for qi in range(n_q)
    ]

    pair_counts = []
    for qi in range(n_q):
        b_idx, _w = banded_pairs(residual[qi], float(bands[qi]))
        pair_counts.append(int(b_idx.size))
    pair_counts_arr = np.asarray(pair_counts, dtype=np.int64)
    max_pairs = POOL_SIZE * (POOL_SIZE - 1) // 2

    teacher_report = {
        "schema_version": "nl_996a_teacher_processing_v1",
        "labels_dir": str(_resolve(args.labels_dir)),
        "n_queries": n_q,
        "pool_size": POOL_SIZE,
        "calm_debias": {
            "method": (
                "global OLS replay_crps ~ 1 + cand_future_activity across all "
                f"{n_q * POOL_SIZE} rows; residual is the teacher signal"
            ),
            **debias_stats,
            "spearman_raw_replay_vs_activity_pooled": raw_rho,
            "spearman_debiased_vs_activity_pooled": debiased_rho,
            "spearman_raw_vs_activity_per_query_mean": float(np.nanmean(per_q_raw)),
            "spearman_debiased_vs_activity_per_query_mean": float(
                np.nanmean(per_q_deb)
            ),
        },
        "per_query_residualization_sensitivity": {
            **per_query_stats,
            "spearman_perquery_residual_vs_activity_pooled": pq_rho,
            "rank_agreement_global_vs_perquery_residual_mean_spearman": float(
                np.nanmean(agreement)
            ),
            "note": "sensitivity only; the pre-registered teacher is the global fit",
        },
        "noise_banding": {
            "band_rule": (
                f"per-query max({NOISE_FLOOR}, "
                f"{BAND_STD_FRACTION} * query_pool_replay_crps_std)"
            ),
            "band_mean": float(np.mean(bands)),
            "band_median": float(np.median(bands)),
            "band_floor_active_frac": float(
                np.mean(bands <= NOISE_FLOOR + 1e-12)
            ),
            "pairs_possible_per_query": int(max_pairs),
            "pairs_retained_total": int(pair_counts_arr.sum()),
            "pairs_retained_per_query_mean": float(pair_counts_arr.mean()),
            "pairs_retained_per_query_median": float(np.median(pair_counts_arr)),
            "pairs_retained_fraction": float(
                pair_counts_arr.sum() / (n_q * max_pairs)
            ),
            "queries_with_zero_pairs": int(np.sum(pair_counts_arr == 0)),
        },
    }
    _write_json(output_dir / "teacher_processing_996a.json", teacher_report)
    print(
        json.dumps(
            {
                "raw_vs_activity_spearman": round(raw_rho, 4),
                "debiased_vs_activity_spearman": round(debiased_rho, 4),
                "pairs_retained_total": int(pair_counts_arr.sum()),
            }
        ),
        flush=True,
    )

    # ---- Embeddings + deployable features.
    emb_windows, emb_matrix, emb_meta = load_corpus_full_professional_embeddings()
    bank_arrays = _resolve(SUPPORT_BANK_DIR) / "support_bank_arrays.npz"
    with np.load(bank_arrays) as bank:
        history_raw = np.asarray(bank["history_raw"], dtype=np.float32)
        support_indices = np.asarray(bank["support_indices"], dtype=np.int64)
    if not np.array_equal(support_indices, np.arange(history_raw.shape[0])):
        raise ValueError("939a bank support_indices are not the identity 0..4009")
    if np.isnan(history_raw).any():
        raise ValueError("bank history_raw contains NaNs; distance equivalence breaks")
    fit_indices = np.arange(history_raw.shape[0], dtype=np.int64)
    pref_z = prefix_delta_z(history_raw, fit_indices)
    term_z = chassis_terminal_z(history_raw, fit_indices)
    n_dims = int(history_raw.shape[-1])
    chassis_temperature = float(np.sqrt(n_dims))

    features = build_candidate_features(
        query_windows=query_windows,
        pool_windows=pool_windows,
        pool_start_distance=start_distance,
        embedding_windows=emb_windows,
        embeddings=emb_matrix,
        prefix_z=pref_z,
    )

    # chassis-unit start scores + consistency vs the stored 994b RMS distance
    chassis_dist = np.zeros_like(start_distance)
    for qi in range(n_q):
        diff = term_z[pool_windows[qi]] - term_z[query_windows[qi]][None, :]
        chassis_dist[qi] = np.linalg.norm(diff, axis=1)
    ratio = chassis_dist / np.maximum(start_distance, 1e-12)
    ratio_dev = float(np.max(np.abs(ratio / np.sqrt(n_dims) - 1.0)))
    if ratio_dev > 1e-3:
        raise ValueError(
            f"chassis L2z distance is not sqrt(D) x stored RMS distance "
            f"(max rel dev {ratio_dev:.2e}); pool/selection parity would break"
        )
    chassis_start_scores = -chassis_dist

    np.savez_compressed(
        output_dir / "n1_features_996a.npz",
        query_window=query_windows,
        pool_candidate_window=pool_windows,
        features=features.astype(np.float32),
        feature_names=np.asarray(FEATURE_NAMES),
        replay_crps=replay.astype(np.float32),
        debiased_residual=residual.astype(np.float32),
        noise_band=bands.astype(np.float32),
        chassis_start_scores=chassis_start_scores.astype(np.float32),
        cand_start_distance=start_distance.astype(np.float32),
    )

    # ---- Stage 2: query-side CV (contiguous window blocks, no shuffle).
    folds = contiguous_folds(n_q, n_folds)
    configs: list[dict[str, Any]] = []
    for c in (0.01, 0.1, 1.0, 10.0):
        for seed in SEEDS:
            configs.append({"model": "linear", "c": c, "seed": seed})
    for hidden in (8, 16, 32):
        for seed in SEEDS:
            configs.append(
                {
                    "model": "mlp",
                    "hidden": hidden,
                    "epochs": int(args.mlp_epochs) if not args.smoke else 30,
                    "lr": 1e-2,
                    "weight_decay": 1e-4,
                    "seed": seed,
                }
            )

    cv_rows: list[dict[str, Any]] = []
    t_cv = time.time()
    for config in configs:
        fold_stats: list[dict[str, float]] = []
        for fold_id, heldout in enumerate(folds):
            train_pos = np.concatenate(
                [f for i, f in enumerate(folds) if i != fold_id]
            )
            student = train_student(
                config,
                features=features,
                residual=residual,
                bands=bands,
                query_positions=train_pos,
                device=device,
            )
            stats = fold_metrics(
                student,
                features=features,
                residual=residual,
                bands=bands,
                query_positions=heldout,
            )
            stats["fold"] = fold_id
            fold_stats.append(stats)
        accs = [s["pair_accuracy"] for s in fold_stats]
        rhos = [s["mean_spearman"] for s in fold_stats]
        cv_rows.append(
            {
                **config,
                "fold_stats": fold_stats,
                "mean_pair_accuracy": float(np.nanmean(accs)),
                "std_pair_accuracy": float(np.nanstd(accs)),
                "mean_spearman": float(np.nanmean(rhos)),
            }
        )
        print(
            f"cv {config} -> acc {np.nanmean(accs):.4f} "
            f"rho {np.nanmean(rhos):.4f} ({time.time() - t_cv:.0f}s)",
            flush=True,
        )

    # aggregate seeds per hyperparameter cell, select on mean held-out accuracy
    def _cell_key(row: dict[str, Any]) -> str:
        if row["model"] == "linear":
            return f"linear_c{row['c']:g}"
        return f"mlp_h{row['hidden']}"

    cells: dict[str, list[dict[str, Any]]] = {}
    for row in cv_rows:
        cells.setdefault(_cell_key(row), []).append(row)
    cell_summary = {
        key: {
            "mean_pair_accuracy_over_seeds": float(
                np.mean([r["mean_pair_accuracy"] for r in rows])
            ),
            "std_pair_accuracy_over_seeds": float(
                np.std([r["mean_pair_accuracy"] for r in rows])
            ),
            "mean_spearman_over_seeds": float(
                np.mean([r["mean_spearman"] for r in rows])
            ),
        }
        for key, rows in cells.items()
    }
    best_cell = max(
        cell_summary,
        key=lambda key: (
            cell_summary[key]["mean_pair_accuracy_over_seeds"],
            cell_summary[key]["mean_spearman_over_seeds"],
        ),
    )
    best_rows = cells[best_cell]
    best_config = dict(
        max(best_rows, key=lambda r: r["mean_pair_accuracy"])
    )
    for drop in ("fold_stats", "mean_pair_accuracy", "std_pair_accuracy", "mean_spearman"):
        best_config.pop(drop, None)

    # ---- Tilt-weight pre-registration via held-out replay-CRPS proxy.
    tilt_grid_rows: list[dict[str, Any]] = []
    proxy_base = np.zeros(n_q, dtype=np.float64)
    proxy_by_tw = {tw: np.zeros(n_q, dtype=np.float64) for tw in TILT_WEIGHT_GRID}
    heldout_scores = np.zeros((n_q, POOL_SIZE), dtype=np.float64)
    for fold_id, heldout in enumerate(folds):
        train_pos = np.concatenate([f for i, f in enumerate(folds) if i != fold_id])
        student = train_student(
            best_config,
            features=features,
            residual=residual,
            bands=bands,
            query_positions=train_pos,
            device=device,
        )
        for qi in heldout:
            heldout_scores[qi] = student.score(features[qi])
    tilt_z_all = within_pool_z(heldout_scores)
    max_weight_by_tw = {tw: np.zeros(n_q, dtype=np.float64) for tw in TILT_WEIGHT_GRID}
    effective_n_by_tw = {tw: np.zeros(n_q, dtype=np.float64) for tw in TILT_WEIGHT_GRID}
    for qi in range(n_q):
        sel0, w0 = simulate_tilt_selection(
            pool_windows=pool_windows[qi],
            chassis_start_scores=chassis_start_scores[qi],
            tilt_scores_z=np.zeros(POOL_SIZE),
            tilt_weight=0.0,
            chassis_temperature=chassis_temperature,
        )
        proxy_base[qi] = replay_proxy_crps(replay[qi], sel0, w0)
        # neutral-tilt parity vs the stored 994b RMS ranking (start-only top-3)
        rms_order = np.lexsort((pool_windows[qi], start_distance[qi]))[:TOP_K]
        if sorted(sel0.tolist()) != sorted(rms_order.tolist()):
            raise ValueError(
                f"neutral selection diverges from start-only top-3 at query "
                f"{int(query_windows[qi])}"
            )
        rms_weights = np.asarray(
            _softmax_weights(
                [-float(start_distance[qi][p]) for p in sel0], temperature=1.0
            )
        )
        if float(np.max(np.abs(rms_weights - w0))) > 1e-4:
            raise ValueError(
                f"neutral chassis weights diverge from 994a softmax(T=1) "
                f"weights at query {int(query_windows[qi])}"
            )
        for tw in TILT_WEIGHT_GRID:
            sel, w = simulate_tilt_selection(
                pool_windows=pool_windows[qi],
                chassis_start_scores=chassis_start_scores[qi],
                tilt_scores_z=tilt_z_all[qi],
                tilt_weight=float(tw),
                chassis_temperature=chassis_temperature,
            )
            proxy_by_tw[tw][qi] = replay_proxy_crps(replay[qi], sel, w)
            max_weight_by_tw[tw][qi] = float(np.max(w))
            effective_n_by_tw[tw][qi] = float(1.0 / np.sum(np.square(w)))
    for tw in TILT_WEIGHT_GRID:
        delta = proxy_by_tw[tw] - proxy_base
        tilt_grid_rows.append(
            {
                "tilt_weight": float(tw),
                "mean_proxy_delta_crps": float(delta.mean()),
                "median_proxy_delta_crps": float(np.median(delta)),
                "win_rate_delta_lt_0": float(np.mean(delta < 0.0)),
                "frac_selection_changed": float(np.mean(np.abs(delta) > 1e-12)),
                "mean_max_weight": float(max_weight_by_tw[tw].mean()),
                "median_max_weight": float(np.median(max_weight_by_tw[tw])),
                "mean_effective_n": float(effective_n_by_tw[tw].mean()),
            }
        )
    # Non-degeneracy guard (994b MAX_WEIGHT_BAND upper edge 0.92): the linear
    # replay proxy is blind to mixture diversification, so tilt weights whose
    # selected-support softmax degenerates toward one-hot are excluded from
    # the CV selection. Val-blind constraint, recorded in full.
    MAX_WEIGHT_UPPER = 0.92
    eligible = [
        row for row in tilt_grid_rows if row["mean_max_weight"] <= MAX_WEIGHT_UPPER
    ]
    candidate_rows_for_selection = eligible if eligible else tilt_grid_rows
    best_tw_row = min(
        candidate_rows_for_selection, key=lambda r: r["mean_proxy_delta_crps"]
    )
    preregistered_tilt_weight = float(best_tw_row["tilt_weight"])

    # ---- Final deployment student: best config trained on ALL queries.
    final_student = train_student(
        best_config,
        features=features,
        residual=residual,
        bands=bands,
        query_positions=np.arange(n_q, dtype=np.int64),
        device=device,
    )
    final_spec = final_student.export()

    # train-side pool-quality reference for the 996b gate (chassis units)
    pool_min_chassis = chassis_dist.min(axis=1)
    pool_reference = {
        "q90_min_distance": float(np.quantile(pool_min_chassis, 0.90)),
        "source": (
            "q90 of per-train-query pool min start distance in CHASSIS L2z "
            f"units over the {n_q} labeled 995d queries"
        ),
    }

    runtimes = {"total_seconds": round(time.time() - t_start, 1)}
    cv_report = {
        "schema_version": "nl_996a_n1_tilt_training_v1",
        "smoke": bool(args.smoke),
        "n_queries": n_q,
        "n_folds": n_folds,
        "fold_scheme": "contiguous window blocks over queries sorted by window (no shuffle)",
        "feature_names": list(FEATURE_NAMES),
        "feature_policy": {
            "deployable_only": True,
            "excluded": ["cand_future_activity", "cand_index_gap"],
            "optional_direction_match_feature": (
                "skipped (recorded): would add a card-parsing axis; three core "
                "features keep one-axis discipline"
            ),
            "prefix_feature_choice": (
                "cosine of z-scaled 30-day history_raw terminal deltas "
                "(history_raw[w,-1,:] - history_raw[w,0,:]; per-dim std over "
                "bank windows 0..4009)"
            ),
        },
        "embedding_metadata": emb_meta,
        "cv_table": cv_rows,
        "cv_cell_summary": cell_summary,
        "selected_cell": best_cell,
        "selected_config": best_config,
        "tilt_weight_selection": {
            "method": (
                "held-out replay-CRPS proxy: softmax-weighted (chassis "
                "T=sqrt(D)) mean solo replay CRPS of the chassis-simulated "
                "top-3 vs the start-only top-3, RAW replay labels (the "
                "deployment metric), per-query paired"
            ),
            "chassis_temperature": chassis_temperature,
            "n_state_dims": n_dims,
            "grid": tilt_grid_rows,
            "non_degeneracy_guard": {
                "rule": (
                    "tilt weights with mean selected-support max weight > "
                    f"{MAX_WEIGHT_UPPER} (994b MAX_WEIGHT_BAND upper edge) are "
                    "excluded: the linear replay proxy cannot see the mixture-"
                    "diversification loss of near-one-hot weights"
                ),
                "max_weight_upper": MAX_WEIGHT_UPPER,
                "eligible_tilt_weights": [
                    float(row["tilt_weight"]) for row in eligible
                ],
            },
            "preregistered_tilt_weight": preregistered_tilt_weight,
            "preregistered_note": (
                "ONE value frozen before any val outcome is seen; 996b must "
                "use exactly this tilt_weight"
            ),
            "tilt_score_standardization": "within_pool_z",
        },
        "pool_reference_quantiles": pool_reference,
        "final_student": final_spec,
        "runtimes": runtimes,
    }
    _write_json(output_dir / "n1_tilt_training_report_996a.json", cv_report)
    _write_json(
        output_dir / "n1_final_student_996a.json",
        {
            "schema_version": "nl_996a_final_student_v1",
            "student": final_spec,
            "tilt_score_standardization": "within_pool_z",
            "preregistered_tilt_weight": preregistered_tilt_weight,
            "chassis_temperature": chassis_temperature,
            "pool_reference_quantiles": pool_reference,
            "feature_names": list(FEATURE_NAMES),
        },
    )
    print(
        json.dumps(
            {
                "selected_cell": best_cell,
                "mean_pair_accuracy": cell_summary[best_cell][
                    "mean_pair_accuracy_over_seeds"
                ],
                "mean_spearman": cell_summary[best_cell]["mean_spearman_over_seeds"],
                "preregistered_tilt_weight": preregistered_tilt_weight,
                "best_tilt_mean_proxy_delta": best_tw_row["mean_proxy_delta_crps"],
                "total_seconds": runtimes["total_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
