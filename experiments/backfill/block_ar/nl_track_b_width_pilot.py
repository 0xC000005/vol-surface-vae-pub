#!/usr/bin/env python
"""Track B B-WIDTH STEP 0 -- cheap no_grad PILOT: does conditioned fan WIDTH track a GROUNDED
narrative-intensity scalar, BEFORE investing in a width-objective trainer?

This is a diagnostic, NOT a gate. It loads an EXISTING narrative-conditioning checkpoint
(default b2-velocity, the best-fidelity one) + the frozen 734a base, samples conditioned
ensembles on the HELD-OUT val-frame narrative windows, and measures the narrative's MARGINAL
width effect:

    width_delta(window) = width(present=True) - width(present=False)   [CRN: shared seed]

where width = mean over factors of the cross-member std of the terminal cumulative move. The
present=False arm is the frozen-734a baseline for the SAME window (so regime/history is held
fixed and the cross-window regime confound is removed -- this is the advisor's load-bearing
correction: correlate intensity vs width_DELTA, never raw conditioned width, which the frozen
generator already inflates from high-vol history alone).

Reported per checkpoint:
- spearman(grounded_intensity, width_delta) for REAL intensity, with a >=1000-permutation NULL
  band (mean / std / 97.5th pct of |spearman| under shuffled intensity labels) -- a single
  shuffle at N~115 has std ~0.09, larger than any honest margin, so we use the permutation tail;
- the SHUFFLED point spearman (one representative shuffle, for the report's real-vs-shuffled line);
- width_delta magnitude (mean/std/median |delta|) + mean raw widths -- REQUIRED to disambiguate a
  null: if width_delta ~ 0 everywhere the checkpoint simply does not move width (uninformative,
  trainer still warranted); if width_delta is substantial but uncorrelated that is the real
  "responds to PRESENCE not INTENSITY" signal.

NO OpenAI for the intensity scalar (prose-parse). Embeddings (the conditioning signal) are
text-embedding-3-small, disk-cached so repeated runs make no new API calls. 734a is read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (  # noqa: E402
    load_model,
)
from experiments.backfill.block_ar.nl_track_b_grounded_intensity import (  # noqa: E402
    grounded_intensity_for_windows,
)
# Reuse the gate's canonical window-build + CRN sampling + masks (single source of truth).
from experiments.backfill.block_ar.nl_track_b_directional_gate import (  # noqa: E402
    _build_masks,
    build_gate_windows,
    conditioned_samples,
    terminal_delta_from_increments,
)
from experiments.backfill.block_ar.train_track_b_generator_conditioning import (  # noqa: E402
    DEFAULT_CHECKPOINT as BASE_734A_DEFAULT,
    DEFAULT_MANIFEST_EXAMPLES,
    EMBED_DIM,
    VAL_RANGES,
    assert_bank_alignment,
    load_window_texts,
)

DEFAULT_SUPPORT_BANK = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_bank_train_all_939a/support_bank_arrays.npz"
)
DEFAULT_CHECKPOINTS = {
    "b2_velocity": "models/backfill/generator_conditioning_probe_b2/best_model.pt",
    "b1": "models/backfill/generator_conditioning_probe_b1/best_model.pt",
}
DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/track_b_width_pilot"
)
EMBED_CACHE = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "track_b_width_pilot/embed_cache_small.npz"
)


def cached_embeddings(
    texts: dict[int, str], embed_mode: str, dotenv_path: str, cache_path: str
) -> dict[int, np.ndarray]:
    """Embed window narratives, disk-cached by sha256(text) so reruns make no new API calls."""
    if embed_mode == "hash":
        # deterministic offline pseudo-embedding (smoke only)
        def _h(t: str) -> np.ndarray:
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "big")
            v = np.random.default_rng(seed).standard_normal(EMBED_DIM).astype(np.float32)
            n = float(np.linalg.norm(v))
            return v / n if n > 0 else v

        return {i: _h(t) for i, t in texts.items()}

    cache: dict[str, np.ndarray] = {}
    cpath = Path(cache_path)
    if cpath.exists():
        z = np.load(cpath, allow_pickle=True)
        keys = list(z["keys"])
        vecs = z["vectors"]
        cache = {str(k): vecs[i] for i, k in enumerate(keys)}

    def _key(t: str) -> str:
        return hashlib.sha256(t.encode("utf-8")).hexdigest()

    missing = sorted({t for t in texts.values() if _key(t) not in cache})
    if missing:
        from experiments.backfill.block_ar.nl_text_conditioning import (
            embed_texts_with_openai,
        )

        vectors = embed_texts_with_openai(missing, dotenv_path=dotenv_path)
        for t, v in zip(missing, vectors):
            cache[_key(t)] = np.asarray(v, dtype=np.float32)
        cpath.parent.mkdir(parents=True, exist_ok=True)
        keys = list(cache.keys())
        np.savez(
            cpath,
            keys=np.array(keys, dtype=object),
            vectors=np.stack([cache[k] for k in keys]).astype(np.float32),
        )
    return {i: cache[_key(t)] for i, t in texts.items()}


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 3:
        return None
    rx = np.argsort(np.argsort(x.astype(float)))
    ry = np.argsort(np.argsort(y.astype(float)))
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def permutation_null(
    intensity: np.ndarray, width_delta: np.ndarray, n_perm: int, seed: int
) -> dict[str, Any]:
    """>=1000-permutation null band of spearman(shuffled intensity, fixed width_delta)."""
    rng = np.random.default_rng(int(seed))
    real = _spearman(intensity, width_delta)
    null_vals: list[float] = []
    for _ in range(int(n_perm)):
        perm = rng.permutation(intensity.size)
        s = _spearman(intensity[perm], width_delta)
        if s is not None:
            null_vals.append(s)
    null = np.asarray(null_vals, dtype=float)
    abs_null = np.abs(null)
    p_two_sided = (
        float((abs_null >= abs(real)).mean()) if (real is not None and null.size) else None
    )
    return {
        "real_spearman": real,
        "n_perm": int(null.size),
        "null_mean": float(null.mean()) if null.size else None,
        "null_std": float(null.std()) if null.size else None,
        "null_p975_abs": float(np.quantile(abs_null, 0.975)) if null.size else None,
        "null_p975_signed": float(np.quantile(null, 0.975)) if null.size else None,
        "exceeds_null_upper_tail": (
            bool(real is not None and null.size and real > np.quantile(null, 0.975))
        ),
        "p_value_two_sided": p_two_sided,
    }


@torch.no_grad()
def run_pilot_for_checkpoint(
    label: str,
    trained_checkpoint: str,
    args: argparse.Namespace,
    device: torch.device,
    windows: dict[str, Any],
    heldout_rows: np.ndarray,
    embeddings: dict[int, np.ndarray],
    intensity_norm: dict[int, float],
) -> dict[str, Any]:
    model, _payload = load_model(trained_checkpoint, device)
    assert getattr(model.cfg, "narrative_conditioning", False), (
        f"{label}: checkpoint must have narrative_conditioning=True"
    )
    assert model.narrative_adapter is not None, f"{label}: missing narrative_adapter"

    width_cond: list[float] = []
    width_base: list[float] = []
    intensities: list[float] = []
    rows_used: list[int] = []
    for r in heldout_rows:
        r = int(r)
        seed_r = int(args.seed) + r  # CRN: same base noise for present=True and present=False
        inc_real = conditioned_samples(
            model, windows, r, narrative_emb=embeddings[r], present=True,
            n_samples=args.n_samples, n_steps=args.n_steps, chunk_size=args.chunk_size,
            seed=seed_r, device=device,
        )
        inc_base = conditioned_samples(
            model, windows, r, narrative_emb=None, present=False,
            n_samples=args.n_samples, n_steps=args.n_steps, chunk_size=args.chunk_size,
            seed=seed_r, device=device,
        )
        wc = float(terminal_delta_from_increments(inc_real).std(axis=0).mean())
        wb = float(terminal_delta_from_increments(inc_base).std(axis=0).mean())
        width_cond.append(wc)
        width_base.append(wb)
        intensities.append(float(intensity_norm[r]))
        rows_used.append(r)

    wc_a = np.asarray(width_cond)
    wb_a = np.asarray(width_base)
    inten = np.asarray(intensities)
    width_delta = wc_a - wb_a
    # relative widening kills per-window encoding-scale effects (advisor #1).
    rel_delta = width_delta / np.maximum(wb_a, 1e-8)

    # real-vs-shuffled (single representative shuffle) for the report line.
    rng = np.random.default_rng(int(args.seed) + 999)
    shuf_perm = rng.permutation(inten.size)
    sp_real = _spearman(inten, width_delta)
    sp_shuf = _spearman(inten[shuf_perm], width_delta)

    null_delta = permutation_null(inten, width_delta, args.n_perm, int(args.seed) + 1)
    null_rel = permutation_null(inten, rel_delta, args.n_perm, int(args.seed) + 2)
    null_raw = permutation_null(inten, wc_a, args.n_perm, int(args.seed) + 3)  # regime-confounded

    abs_delta = np.abs(width_delta)
    return {
        "label": label,
        "checkpoint": trained_checkpoint,
        "n_windows": int(inten.size),
        "intensity_distribution": {
            "min": float(inten.min()), "max": float(inten.max()),
            "mean": float(inten.mean()), "std": float(inten.std()),
            "n_unique": int(np.unique(np.round(inten, 6)).size),
            "n_zero": int((inten == 0.0).sum()),
        },
        "width": {
            "mean_width_conditioned": float(wc_a.mean()),
            "mean_width_baseline": float(wb_a.mean()),
            "mean_width_delta": float(width_delta.mean()),
            "std_width_delta": float(width_delta.std()),
            "median_abs_width_delta": float(np.median(abs_delta)),
            "mean_abs_width_delta": float(abs_delta.mean()),
            "mean_rel_width_delta": float(rel_delta.mean()),
        },
        "spearman_intensity_vs_width_delta": {
            "real": sp_real,
            "shuffled_one_draw": sp_shuf,
            "real_minus_shuffled": (
                None if sp_real is None or sp_shuf is None else sp_real - sp_shuf
            ),
            "permutation_null": null_delta,
        },
        "spearman_intensity_vs_rel_width_delta": {
            "real": null_rel["real_spearman"],
            "permutation_null": null_rel,
        },
        "spearman_intensity_vs_raw_width_REGIME_CONFOUNDED": {
            "real": null_raw["real_spearman"],
            "permutation_null": null_raw,
            "CAVEAT": (
                "raw conditioned width is regime-confounded: high-intensity narratives describe "
                "high-vol windows and frozen 734a already widens from history alone. Reported "
                "ONLY to show the confound vs the valid width_DELTA measure -- NOT the verdict."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", default=BASE_734A_DEFAULT)
    parser.add_argument(
        "--checkpoints", nargs="+", default=["b2_velocity", "b1"],
        help="labels from DEFAULT_CHECKPOINTS, or label=path pairs",
    )
    parser.add_argument("--support-bank", default=DEFAULT_SUPPORT_BANK)
    parser.add_argument("--manifest-examples", default=DEFAULT_MANIFEST_EXAMPLES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embed-mode", choices=["openai", "hash"], default="openai")
    parser.add_argument("--embed-cache", default=EMBED_CACHE)
    parser.add_argument("--dotenv-path", default=".env")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument("--max-windows", type=int, default=0, help="cap held-out windows (smoke)")
    parser.add_argument("--seed", type=int, default=7344)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device(
        args.device if (torch.cuda.is_available() or str(args.device) == "cpu") else "cpu"
    )

    # resolve checkpoint labels -> paths
    ckpts: dict[str, str] = {}
    for entry in args.checkpoints:
        if "=" in entry:
            lab, path = entry.split("=", 1)
            ckpts[lab] = path
        elif entry in DEFAULT_CHECKPOINTS:
            ckpts[entry] = DEFAULT_CHECKPOINTS[entry]
        else:
            raise ValueError(f"unknown checkpoint label {entry!r}")

    # canonical windows (need one payload; build from the base 734a, scope-identical) ----
    _bm, base_payload = load_model(args.base_checkpoint, device)
    windows = build_gate_windows(args.base_checkpoint, base_payload)
    n_windows = int(windows["history_level"].shape[0])
    assert_bank_alignment(
        {k: windows[k] for k in ("history_level", "history_norm", "center", "scale", "drift_feature")},
        args.support_bank,
    )
    masks = _build_masks(n_windows)
    texts = load_window_texts(args.manifest_examples, n_windows)
    embeddings = cached_embeddings(texts, args.embed_mode, args.dotenv_path, args.embed_cache)
    val_rows = np.nonzero(masks["val"])[0]
    heldout_rows = np.array([r for r in val_rows if int(r) in embeddings], dtype=np.int64)
    if int(args.max_windows) > 0:
        heldout_rows = heldout_rows[: int(args.max_windows)]
    if heldout_rows.size < 3:
        raise RuntimeError("need >=3 held-out narrative windows for a rank pilot -> STOP")

    # grounded intensity from the SAME texts whose embeddings condition the model (advisor #4).
    heldout_texts = {int(r): texts[int(r)] for r in heldout_rows}
    intensity_norm = grounded_intensity_for_windows(heldout_texts)

    print(
        f"[pilot] held-out narrative windows={heldout_rows.size} "
        f"embed_mode={args.embed_mode} n_samples={args.n_samples} n_perm={args.n_perm}"
    )
    results = []
    for label, path in ckpts.items():
        print(f"[pilot] checkpoint {label} <- {path}")
        res = run_pilot_for_checkpoint(
            label, path, args, device, windows, heldout_rows, embeddings, intensity_norm
        )
        sd = res["spearman_intensity_vs_width_delta"]
        nd = sd["permutation_null"]
        print(
            f"  -> real spearman(intensity, width_delta) = {sd['real']} | "
            f"shuffled = {sd['shuffled_one_draw']} | "
            f"null p975_abs = {nd['null_p975_abs']} exceeds_tail = {nd['exceeds_null_upper_tail']} "
            f"p2s = {nd['p_value_two_sided']}"
        )
        print(
            f"  -> width: cond {res['width']['mean_width_conditioned']:.4f} "
            f"base {res['width']['mean_width_baseline']:.4f} "
            f"mean|delta| {res['width']['mean_abs_width_delta']:.5f} "
            f"mean_delta {res['width']['mean_width_delta']:.5f}"
        )
        results.append(res)

    payload = {
        "schema_version": "nl_track_b_width_pilot_v1",
        "step": "STEP 0 pilot (no training)",
        "base_checkpoint_734a": str(args.base_checkpoint),
        "n_heldout_windows": int(heldout_rows.size),
        "embed_mode": str(args.embed_mode),
        "n_samples": int(args.n_samples),
        "n_perm": int(args.n_perm),
        "seed": int(args.seed),
        "intensity_source": (
            "grounded prose-parse magnitude*|salience| (nl_track_b_grounded_intensity), NO OpenAI"
        ),
        "measurement_note": (
            "spearman is intensity vs width_DELTA = width(present=True) - width(present=False), "
            "CRN-shared seed per window (regime/history held fixed). Raw-width spearman reported "
            "ONLY as the regime-confounded comparator."
        ),
        "results_by_checkpoint": results,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "track_b_width_pilot.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[pilot] wrote {out_path}")


if __name__ == "__main__":
    main()
