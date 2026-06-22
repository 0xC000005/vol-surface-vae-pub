#!/usr/bin/env python
"""Track B Task 6 -- directional / width GATE for the narrative-conditioning probe.

Given a trained narrative-conditioning checkpoint (b1-style, narrative_conditioning=True)
plus the frozen 734a base, this runs the mechanistic 3-level gate + the real-vs-shuffled-
vs-zero ablation (the Exp-153a mandate) and persists everything to JSON + a dated verifier
report. It NEVER modifies 734a or the trained checkpoint (both read-only).

Mechanistic 3 levels (plan Task 6):
  L1 KNOB ACTIVATED  : adapter context norm > 0 on real narratives, ~0 on present=False.
  L2 PROPAGATED      : on HELD-OUT (val-frame) windows WITH narrative, conditioned forward
                       terminal direction-match vs realized sign over the narrative's own
                       emphasized factors, compared to chance (0.428) AND to the shuffled
                       null. Plus the WIDTH axis (does conditioned dispersion track narrative
                       intensity -- width-spearman vs start-only).
  L3 MOVED+MONOTONE+FIDELITY : metamorphic alpha-ladder (alpha in {0.5,1,2}) -- response
                       magnitude monotone in injected-context gain -- AND CRPS/Energy within
                       the +/-0.015 branch guardrail vs frozen 734a (no fidelity regression).

Ablation (153a): direction-match for (a) real narrative, (b) shuffled placebo (real texts
reassigned across held-out windows -- right window, wrong text; identical embedding
distribution), (c) zero/null. HEADLINE = real - shuffled DELTA (emphasis bias cancels);
real - zero secondary. shuffled ~= zero confirms the signal is the narrative, not an artifact.

CRITICAL design facts (verified empirically, see header asserts):
  - L2/L3 run ONLY on HELD-OUT VAL-FRAME windows. The ~633 windows the trainer used carry
    narrative but are TRAINING data; scoring there measures memorization, not steering. We
    use the inverse mask (val ranges) intersected with has-narrative.
  - The alpha-ladder scales the adapter OUTPUT (final-Linear weight+bias x alpha), NOT the
    input embedding: the adapter's first layer is LayerNorm, which is scale-invariant, so
    adapter(alpha*emb) == adapter(emb) (verified max|diff| ~4e-7). Scaling the final layer
    scales the context exactly by alpha (verified 2.0x). Reported as "monotone in injected-
    context gain" -- the architecture cannot ingest scalable semantic intensity via the emb.
  - Emphasis = grounding-derived non-flat factors for the SCORED window (the factors the
    narrative actually mentions; |z-delta|<small dropped as flat, matching the pipeline's
    _direction_and_magnitude). Held IDENTICAL across real/shuffled/zero so emphasis bias
    cancels in the delta. all-39 reported as a secondary robustness denominator.
  - Common random numbers: same torch seed immediately before the conditioned and baseline
    sample calls per window (Task-3 pattern), so width/direction deltas are not sampling noise.
  - Shuffle safeguard: emphasis set AND realized-sign target come from the SCORED window;
    only the text->embedding is swapped in from another held-out window (asserted in code).
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from datetime import date
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
from experiments.backfill.block_ar.evaluate_662a_state_aware_normalized_innovation_flow import (  # noqa: E402
    build_val_block,
)
from experiments.backfill.block_ar.nl_narrative_grounded_scenario_pipeline import (  # noqa: E402
    KEY_FACTOR_NAMES,
)
from experiments.backfill.block_ar.increment_coordinate_628_utils import (  # noqa: E402
    reconstruct_state_from_increments,
)
from experiments.backfill.block_ar.nl_scenario_level_evaluation import (  # noqa: E402
    build_delta_scale,
    future_delta_paths,
    score_sample_distribution,
)
# Reuse the trainer's canonical data-prep + split constants (single source of truth).
from experiments.backfill.block_ar.train_track_b_generator_conditioning import (  # noqa: E402
    DEFAULT_CHECKPOINT as BASE_734A_DEFAULT,
    DEFAULT_MANIFEST_EXAMPLES,
    DEFAULT_SUPPORT_BANK,
    EMBED_DIM,
    PURGE_GAP,
    VAL_RANGES,
    _bank_build_namespace,
    assert_bank_alignment,
    build_embeddings,
    load_window_texts,
)

DEFAULT_TRAINED_CHECKPOINT = "models/backfill/generator_conditioning_probe_b1/best_model.pt"
DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/track_b_directional_gate"
)
VERIFIER_DIR = "docs/research_protocols/nl_prefix_latent_verifier_reports"

# --- PRE-REGISTERED BARS (set BEFORE any b1 number is read) ------------------------------
PREREGISTERED_BARS = {
    "chance_direction_match": 0.428,          # 153a-era chance reference
    "level2_real_vs_shuffled_margin": 0.05,   # real - shuffled direction-match must exceed this
    "level2_real_vs_zero_margin": 0.05,       # real - zero must exceed this (secondary)
    "level2_shuffled_zero_tol": 0.05,         # |shuffled - zero| <= this confirms placebo ~ null
    "level1_context_norm_min": 1e-3,          # real-narrative adapter context norm must exceed
    "level1_present_false_norm_max": 1e-8,    # present=False context must be ~0 (gate)
    "level3_fidelity_guardrail_crps": 0.015,  # |CRPS_cond - CRPS_734a| <= this (no regression)
    "level3_fidelity_guardrail_energy": 0.015,
    "level3_monotone_tol": 1e-6,              # response magnitude must be non-decreasing in alpha
    "flat_factor_z_threshold": 0.05,          # |z-delta| < this => flat factor (no sign), excluded
    "alpha_ladder": [0.5, 1.0, 2.0],
}


def _build_masks(n: int) -> dict[str, np.ndarray]:
    """val / purge / train masks over window indices, mirroring the calibration build_masks."""
    val = np.zeros(n, dtype=bool)
    for lo, hi in VAL_RANGES:
        val[int(lo) : int(hi)] = True
    purge = np.zeros(n, dtype=bool)
    for lo, hi in VAL_RANGES:
        purge[max(0, int(lo) - PURGE_GAP) : min(n, int(hi) + PURGE_GAP)] = True
    purge = purge & ~val
    return {"val": val, "purge": purge, "train": ~val & ~purge}


def build_gate_windows(base_checkpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Canonical SNI windows + future_raw + specs (for real-space scoring & emphasis)."""
    args = _bank_build_namespace(base_checkpoint)
    (
        history_level,
        history_norm,
        center,
        scale,
        drift_feature,
        history_raw,
        specs,
        block,
    ) = build_val_block(args, payload)
    n = int(history_level.shape[0])
    n_cells = int(history_raw.shape[-1])
    future_raw = np.asarray(block.future_state[:n, :, :n_cells], dtype=np.float32)
    future_delta = future_delta_paths(history_raw, future_raw)  # raw terminal-path deltas
    source_index = np.asarray(block.indices, dtype=np.int64)
    return {
        "history_level": history_level.astype(np.float32),
        "history_norm": history_norm.astype(np.float32),
        "center": center.astype(np.float32),
        "scale": scale.astype(np.float32),
        "drift_feature": drift_feature.astype(np.float32),
        "history_raw": history_raw.astype(np.float32),
        "future_raw": future_raw,
        "future_delta": future_delta.astype(np.float32),
        "source_index": source_index,
        "specs": specs,
        "n_cells": n_cells,
    }


def emphasis_columns(specs: list[Any]) -> tuple[np.ndarray, dict[str, int]]:
    """Column indices of the KEY_FACTOR_NAMES factors within the scope (grounding factors)."""
    name_to_col = {spec.name: idx for idx, spec in enumerate(specs)}
    cols: list[int] = []
    key_to_col: dict[str, int] = {}
    for _market, spec_name in KEY_FACTOR_NAMES.items():
        if spec_name in name_to_col:
            col = int(name_to_col[spec_name])
            cols.append(col)
            key_to_col[spec_name] = col
    return np.asarray(sorted(set(cols)), dtype=np.int64), key_to_col


def realized_terminal_zdelta(windows: dict[str, Any]) -> np.ndarray:
    """Realized terminal normalized-innovation cumulative move per window/factor: (N, C).

    future_norm cumulative terminal == (future_level[:, -1] - history_level[:, -1]) / scale in
    z-units, but more directly: the realized terminal delta z-scored by the window scale.
    """
    fut_term = windows["future_delta"][:, -1, :]  # raw terminal delta (N, C)
    # z-score by the window-local scale used for normalization (history-only EWMA rms).
    safe = np.maximum(windows["scale"], 1e-8)
    return fut_term / safe


@torch.no_grad()
def conditioned_samples(
    model: Any,
    windows: dict[str, Any],
    row: int,
    *,
    narrative_emb: np.ndarray | None,
    present: bool,
    n_samples: int,
    n_steps: int,
    chunk_size: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """Sample one window's conditioned (or baseline if emb None) rollout. CRN via fixed seed."""
    def t(key: str) -> torch.Tensor:
        return torch.from_numpy(windows[key][row : row + 1]).to(device)

    emb_t = None
    present_t = None
    if narrative_emb is not None:
        emb_t = torch.from_numpy(narrative_emb.reshape(1, -1).astype(np.float32)).to(device)
        present_t = torch.tensor([bool(present)], dtype=torch.bool, device=device)
    torch.manual_seed(int(seed))  # CRN: identical base noise for cond & baseline of this window
    out = model.sample_batched(
        t("history_level"),
        t("history_norm"),
        t("center"),
        t("scale"),
        drift_feature=t("drift_feature"),
        n_samples=int(n_samples),
        n_steps=int(n_steps),
        chunk_size=int(chunk_size),
        narrative_emb=emb_t,
        narrative_present=present_t,
    )
    return out.detach().cpu().numpy()[0]  # (n_samples, n_steps, n_cells) encoded increments


def terminal_delta_from_increments(increments: np.ndarray) -> np.ndarray:
    """Encoded terminal cumulative delta: (n_samples, n_cells). Sign preserved into real space."""
    return np.cumsum(increments, axis=1)[:, -1, :]


def direction_match(
    sample_term: np.ndarray, realized_z: np.ndarray, emph_cols: np.ndarray, flat_z: float
) -> float | None:
    """Fraction of NON-FLAT emphasized factors whose mean sample terminal sign == realized sign."""
    nonflat = emph_cols[np.abs(realized_z[emph_cols]) >= float(flat_z)]
    if nonflat.size == 0:
        return None
    mean_term = sample_term.mean(axis=0)  # (n_cells,)
    gen_sign = np.sign(mean_term[nonflat])
    real_sign = np.sign(realized_z[nonflat])
    return float((gen_sign == real_sign).mean())


def samples_to_raw_deltas(samples_inc: np.ndarray, windows: dict[str, Any], row: int) -> np.ndarray:
    """Encoded per-step increments (S,T,C) -> raw cumulative deltas (S,T,C), the canonical
    scorer input. Mirrors nl_scenario_level_evaluation: reconstruct_state_from_increments
    (integrates + decodes to raw states) then subtract the last-history raw anchor."""
    anchor = windows["history_raw"][row, -1, :]  # (C,) last history raw state
    states = reconstruct_state_from_increments(anchor, samples_inc, windows["specs"])  # (S,T,C)
    return (states - anchor[None, None, :]).astype(np.float32)


def fidelity_scores(
    samples_inc: np.ndarray, windows: dict[str, Any], row: int, delta_scale: np.ndarray
) -> dict[str, float]:
    """CRPS/Energy/coverage of one window's samples vs realized future_delta -- CANONICAL units.

    Both samples and target are RAW cumulative deltas, scaled by build_delta_scale (per-cell
    train-future-delta std), exactly as nl_scenario_level_evaluation does. (The previous version
    fed per-step encoded increments scaled by the encoded EWMA scale -> O(460) garbage CRPS.)
    """
    raw_delta_samples = samples_to_raw_deltas(samples_inc, windows, row)  # (S, T, C) raw deltas
    target = windows["future_delta"][row]  # (T, C) raw cumulative deltas
    return score_sample_distribution(raw_delta_samples, target, scale=delta_scale)


def _safe_mean(values: list[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        args.device if (torch.cuda.is_available() or str(args.device) == "cpu") else "cpu"
    )
    # --- load trained (narrative) checkpoint + frozen 734a baseline (both read-only) -----
    model, payload = load_model(args.trained_checkpoint, device)
    assert getattr(model.cfg, "narrative_conditioning", False), (
        "trained checkpoint must have narrative_conditioning=True"
    )
    assert model.narrative_adapter is not None, "trained checkpoint missing narrative_adapter"
    base_model, base_payload = load_model(args.base_checkpoint, device)

    windows = build_gate_windows(args.base_checkpoint, payload)
    n_windows = int(windows["history_level"].shape[0])
    n_aligned = assert_bank_alignment(
        {k: windows[k] for k in ("history_level", "history_norm", "center", "scale", "drift_feature")},
        args.support_bank,
    )
    emph_cols, key_to_col = emphasis_columns(windows["specs"])
    all_cols = np.arange(int(windows["n_cells"]), dtype=np.int64)
    realized_z = realized_terminal_zdelta(windows)

    # --- HELD-OUT val-frame windows WITH narrative (NOT the training 633) ----------------
    masks = _build_masks(n_windows)
    # delta_scale = per-cell std of TRAINING-window future deltas (canonical scorer scale).
    train_delta = windows["future_delta"][masks["train"]]
    delta_scale = build_delta_scale(train_delta)  # (T, C)
    texts = load_window_texts(args.manifest_examples, n_windows)
    embeddings = build_embeddings(texts, args.embed_mode, args.dotenv_path)
    val_rows = np.nonzero(masks["val"])[0]
    heldout_rows = np.array([r for r in val_rows if int(r) in embeddings], dtype=np.int64)
    if int(args.max_windows) > 0:
        heldout_rows = heldout_rows[: int(args.max_windows)]
    if heldout_rows.size == 0:
        raise RuntimeError("no held-out val windows carry a narrative embedding -> STOP")

    # --- LEVEL 1: knob activated ---------------------------------------------------------
    sample_rows = heldout_rows[: min(16, heldout_rows.size)]
    real_norms: list[float] = []
    absent_norms: list[float] = []
    with torch.no_grad():
        for r in sample_rows:
            emb = torch.from_numpy(embeddings[int(r)].reshape(1, -1).astype(np.float32)).to(device)
            ctx_present = model.narrative_adapter(emb, torch.tensor([True], device=device))
            ctx_absent = model.narrative_adapter(emb, torch.tensor([False], device=device))
            real_norms.append(float(ctx_present.norm()))
            absent_norms.append(float(ctx_absent.norm()))
    level1 = {
        "mean_context_norm_real": float(np.mean(real_norms)),
        "max_context_norm_present_false": float(np.max(absent_norms)),
        "bar_context_norm_min": PREREGISTERED_BARS["level1_context_norm_min"],
        "bar_present_false_norm_max": PREREGISTERED_BARS["level1_present_false_norm_max"],
        "pass": bool(
            np.mean(real_norms) > PREREGISTERED_BARS["level1_context_norm_min"]
            and np.max(absent_norms) <= PREREGISTERED_BARS["level1_present_false_norm_max"]
        ),
    }

    # --- LEVEL 2 + ablation: real / shuffled / zero direction-match + width axis ---------
    rng = np.random.default_rng(int(args.seed))
    # Shuffle placebo: reassign held-out texts across held-out windows (right window, wrong text).
    shuffled_src = heldout_rows.copy()
    perm = rng.permutation(shuffled_src.size)
    # ensure no fixed point (a window keeping its own text)
    for _ in range(8):
        if not np.any(shuffled_src[perm] == shuffled_src):
            break
        perm = rng.permutation(shuffled_src.size)
    shuffle_text_for = {int(heldout_rows[i]): int(shuffled_src[perm[i]]) for i in range(heldout_rows.size)}

    arms = {"real": [], "shuffled": [], "zero": []}      # grounding-emphasis direction-match
    arms_all = {"real": [], "shuffled": [], "zero": []}  # all-39 secondary denominator
    width_cond: list[float] = []
    width_base: list[float] = []
    intensity: list[float] = []  # narrative intensity proxy = realized non-flat emphasis count
    for r in heldout_rows:
        r = int(r)
        seed_r = int(args.seed) + r  # per-window CRN seed, shared across arms for this window
        real_emb = embeddings[r]
        # SHUFFLE SAFEGUARD: emphasis + realized target come from the SCORED window r;
        # only the embedding is swapped from another held-out window.
        shuf_emb = embeddings[shuffle_text_for[r]]
        assert shuffle_text_for[r] != r or heldout_rows.size == 1, "shuffle must not be identity"

        inc_real = conditioned_samples(
            model, windows, r, narrative_emb=real_emb, present=True,
            n_samples=args.n_samples, n_steps=args.n_steps, chunk_size=args.chunk_size,
            seed=seed_r, device=device,
        )
        inc_shuf = conditioned_samples(
            model, windows, r, narrative_emb=shuf_emb, present=True,
            n_samples=args.n_samples, n_steps=args.n_steps, chunk_size=args.chunk_size,
            seed=seed_r, device=device,
        )
        inc_zero = conditioned_samples(
            model, windows, r, narrative_emb=None, present=False,
            n_samples=args.n_samples, n_steps=args.n_steps, chunk_size=args.chunk_size,
            seed=seed_r, device=device,
        )
        for arm, inc in (("real", inc_real), ("shuffled", inc_shuf), ("zero", inc_zero)):
            term = terminal_delta_from_increments(inc)
            arms[arm].append(direction_match(term, realized_z[r], emph_cols, PREREGISTERED_BARS["flat_factor_z_threshold"]))
            arms_all[arm].append(direction_match(term, realized_z[r], all_cols, PREREGISTERED_BARS["flat_factor_z_threshold"]))
        # WIDTH axis: conditioned dispersion (mean per-factor terminal std) vs baseline.
        width_cond.append(float(terminal_delta_from_increments(inc_real).std(axis=0).mean()))
        width_base.append(float(terminal_delta_from_increments(inc_zero).std(axis=0).mean()))
        nonflat = emph_cols[np.abs(realized_z[r][emph_cols]) >= PREREGISTERED_BARS["flat_factor_z_threshold"]]
        intensity.append(float(nonflat.size))

    def _spearman(x: list[float], y: list[float]) -> float | None:
        if len(x) < 3:
            return None
        rx = np.argsort(np.argsort(np.asarray(x, dtype=float)))
        ry = np.argsort(np.argsort(np.asarray(y, dtype=float)))
        if rx.std() == 0 or ry.std() == 0:
            return None
        return float(np.corrcoef(rx, ry)[0, 1])

    dm_real = _safe_mean(arms["real"])
    dm_shuf = _safe_mean(arms["shuffled"])
    dm_zero = _safe_mean(arms["zero"])
    width_delta = [c - b for c, b in zip(width_cond, width_base)]
    level2 = {
        "n_heldout_windows": int(heldout_rows.size),
        "direction_match_grounding_emphasis": {"real": dm_real, "shuffled": dm_shuf, "zero": dm_zero},
        "direction_match_all39": {
            "real": _safe_mean(arms_all["real"]),
            "shuffled": _safe_mean(arms_all["shuffled"]),
            "zero": _safe_mean(arms_all["zero"]),
        },
        "real_minus_shuffled": (None if dm_real is None or dm_shuf is None else dm_real - dm_shuf),
        "real_minus_zero": (None if dm_real is None or dm_zero is None else dm_real - dm_zero),
        "shuffled_minus_zero_abs": (None if dm_shuf is None or dm_zero is None else abs(dm_shuf - dm_zero)),
        "width_axis": {
            "mean_width_conditioned": float(np.mean(width_cond)) if width_cond else None,
            "mean_width_baseline": float(np.mean(width_base)) if width_base else None,
            "width_spearman_vs_intensity": _spearman(intensity, width_delta),
            "CAVEAT": (
                "intensity proxy = count of realized non-flat emphasis factors (REALIZED-"
                "derived, integer, low-range) -- NOT a grounded narrative-intensity scalar. "
                "The width axis is a DIAGNOSTIC placeholder; a real width gate must source "
                "per-window intensity from grounding metadata. Do not read this spearman as "
                "a width-steering result. (Plan frames width as secondary; this is a real-run TODO.)"
            ),
        },
        "bars": {
            "chance": PREREGISTERED_BARS["chance_direction_match"],
            "real_vs_shuffled_margin": PREREGISTERED_BARS["level2_real_vs_shuffled_margin"],
            "real_vs_zero_margin": PREREGISTERED_BARS["level2_real_vs_zero_margin"],
            "shuffled_zero_tol": PREREGISTERED_BARS["level2_shuffled_zero_tol"],
        },
    }
    level2["pass"] = bool(
        level2["real_minus_shuffled"] is not None
        and level2["real_minus_shuffled"] > PREREGISTERED_BARS["level2_real_vs_shuffled_margin"]
        and level2["real_minus_zero"] is not None
        and level2["real_minus_zero"] > PREREGISTERED_BARS["level2_real_vs_zero_margin"]
        and level2["shuffled_minus_zero_abs"] is not None
        and level2["shuffled_minus_zero_abs"] <= PREREGISTERED_BARS["level2_shuffled_zero_tol"]
    )

    # --- LEVEL 3: metamorphic alpha-ladder (scale adapter OUTPUT) + fidelity guardrail ---
    ladder_rows = heldout_rows[: min(int(args.ladder_windows), heldout_rows.size)]
    orig_w = model.narrative_adapter.net[-1].weight.detach().clone()
    orig_b = model.narrative_adapter.net[-1].bias.detach().clone()
    alpha_response: dict[str, float] = {}
    try:
        for alpha in PREREGISTERED_BARS["alpha_ladder"]:
            with torch.no_grad():
                model.narrative_adapter.net[-1].weight.copy_(orig_w * float(alpha))
                model.narrative_adapter.net[-1].bias.copy_(orig_b * float(alpha))
            mags: list[float] = []
            for r in ladder_rows:
                r = int(r)
                inc = conditioned_samples(
                    model, windows, r, narrative_emb=embeddings[r], present=True,
                    n_samples=args.n_samples, n_steps=args.n_steps, chunk_size=args.chunk_size,
                    seed=int(args.seed) + r, device=device,
                )
                inc0 = conditioned_samples(
                    model, windows, r, narrative_emb=None, present=False,
                    n_samples=args.n_samples, n_steps=args.n_steps, chunk_size=args.chunk_size,
                    seed=int(args.seed) + r, device=device,
                )
                # response magnitude = mean |conditioned terminal - baseline terminal| (per-factor)
                mags.append(float(np.abs(
                    terminal_delta_from_increments(inc).mean(0) - terminal_delta_from_increments(inc0).mean(0)
                ).mean()))
            alpha_response[str(alpha)] = float(np.mean(mags)) if mags else float("nan")
    finally:
        with torch.no_grad():
            model.narrative_adapter.net[-1].weight.copy_(orig_w)
            model.narrative_adapter.net[-1].bias.copy_(orig_b)

    alphas_sorted = sorted(PREREGISTERED_BARS["alpha_ladder"])
    resp_seq = [alpha_response[str(a)] for a in alphas_sorted]
    monotone = all(
        resp_seq[i + 1] >= resp_seq[i] - PREREGISTERED_BARS["level3_monotone_tol"]
        for i in range(len(resp_seq) - 1)
    )

    # Fidelity: conditioned vs frozen-734a CRPS/Energy on the held-out windows (CRN).
    crps_cond, crps_base, en_cond, en_base = [], [], [], []
    for r in ladder_rows:
        r = int(r)
        inc_c = conditioned_samples(
            model, windows, r, narrative_emb=embeddings[r], present=True,
            n_samples=args.n_samples, n_steps=args.n_steps, chunk_size=args.chunk_size,
            seed=int(args.seed) + r, device=device,
        )
        inc_b = conditioned_samples(
            base_model, windows, r, narrative_emb=None, present=False,
            n_samples=args.n_samples, n_steps=args.n_steps, chunk_size=args.chunk_size,
            seed=int(args.seed) + r, device=device,
        )
        sc = fidelity_scores(inc_c, windows, r, delta_scale)
        sb = fidelity_scores(inc_b, windows, r, delta_scale)
        crps_cond.append(sc["ensemble_crps_z"]); crps_base.append(sb["ensemble_crps_z"])
        en_cond.append(sc["energy_score_z"]); en_base.append(sb["energy_score_z"])
    crps_gap = float(np.mean(crps_cond) - np.mean(crps_base))
    energy_gap = float(np.mean(en_cond) - np.mean(en_base))
    level3 = {
        "alpha_response": alpha_response,
        "response_monotone_in_context_gain": bool(monotone),
        "crps_conditioned": float(np.mean(crps_cond)),
        "crps_baseline_734a": float(np.mean(crps_base)),
        "crps_gap": crps_gap,
        "energy_conditioned": float(np.mean(en_cond)),
        "energy_baseline_734a": float(np.mean(en_base)),
        "energy_gap": energy_gap,
        "fidelity_guardrail": PREREGISTERED_BARS["level3_fidelity_guardrail_crps"],
        "fidelity_pass": bool(
            abs(crps_gap) <= PREREGISTERED_BARS["level3_fidelity_guardrail_crps"]
            and abs(energy_gap) <= PREREGISTERED_BARS["level3_fidelity_guardrail_energy"]
        ),
        "note": (
            "alpha scales the adapter final-Linear (context) output, NOT the input embedding "
            "(LayerNorm makes input scaling a no-op); reported as monotone in injected-context gain."
        ),
        "monotone_caveat": (
            "Monotonicity is a WIRING check: scaling a zero-baseline-anchored additive context "
            "by alpha near-necessarily increases |conditioned - baseline|. It confirms the "
            "channel propagates, NOT that the narrative steers anything semantic. The semantic "
            "claim rests on Level-2 real-minus-shuffled, not this ladder."
        ),
    }
    level3["pass"] = bool(level3["response_monotone_in_context_gain"] and level3["fidelity_pass"])

    result = {
        "schema_version": "nl_track_b_directional_gate_v1",
        "track": "B-direction (+ width-axis diagnostic)",
        "trained_checkpoint": str(args.trained_checkpoint),
        "base_checkpoint_734a": str(args.base_checkpoint),
        "preregistered_bars": PREREGISTERED_BARS,
        "data": {
            "n_windows": n_windows,
            "bank_aligned_windows": int(n_aligned),
            "n_heldout_val_windows_with_narrative": int(heldout_rows.size),
            "embed_mode": str(args.embed_mode),
            "n_samples": int(args.n_samples),
            "emphasis_key_factors": {k: int(v) for k, v in key_to_col.items()},
            "note_heldout": (
                "L2/L3 scored ONLY on val-frame windows (NOT the trainer's ~633 training "
                "windows); scoring on training windows would measure memorization."
            ),
        },
        "level1_knob_activated": level1,
        "level2_propagated": level2,
        "level3_moved_monotone_fidelity": level3,
        "overall_pass": bool(level1["pass"] and level2["pass"] and level3["pass"]),
        "headline": (
            "Gate headlines on real-minus-shuffled direction-match DELTA (emphasis bias "
            "cancels). real~shuffled~zero => no faithful direction steering (Gap-2 structural "
            "on the direction axis); real>>shuffled~zero => narrative steers direction."
        ),
    }
    return result


def write_verifier_report(result: dict[str, Any], output_dir: Path) -> Path:
    today = date.today().isoformat()
    md_path = Path(VERIFIER_DIR) / f"{today}_track_b_directional_gate.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    l1 = result["level1_knob_activated"]
    l2 = result["level2_propagated"]
    l3 = result["level3_moved_monotone_fidelity"]
    lines = [
        f"# Track B directional/width gate -- {today}",
        "",
        f"- Trained checkpoint: `{result['trained_checkpoint']}`",
        f"- Frozen 734a base: `{result['base_checkpoint_734a']}`",
        f"- Held-out val windows with narrative: {result['data']['n_heldout_val_windows_with_narrative']} "
        f"(bank-aligned {result['data']['bank_aligned_windows']}); embed_mode={result['data']['embed_mode']}",
        f"- **OVERALL PASS: {result['overall_pass']}**",
        "",
        "## Level 1 -- knob activated",
        f"- mean context norm (real): {l1['mean_context_norm_real']:.4g} (bar > {l1['bar_context_norm_min']})",
        f"- max context norm (present=False): {l1['max_context_norm_present_false']:.4g} "
        f"(bar <= {l1['bar_present_false_norm_max']})",
        f"- PASS: {l1['pass']}",
        "",
        "## Level 2 -- propagated (direction) + width axis",
        f"- direction-match (grounding emphasis): {l2['direction_match_grounding_emphasis']}",
        f"- direction-match (all-39 robustness): {l2['direction_match_all39']}",
        f"- **real - shuffled (HEADLINE): {l2['real_minus_shuffled']}** "
        f"(bar > {l2['bars']['real_vs_shuffled_margin']})",
        f"- real - zero: {l2['real_minus_zero']} (bar > {l2['bars']['real_vs_zero_margin']})",
        f"- |shuffled - zero|: {l2['shuffled_minus_zero_abs']} (bar <= {l2['bars']['shuffled_zero_tol']})",
        f"- width spearman vs intensity: {l2['width_axis']['width_spearman_vs_intensity']}",
        f"- PASS: {l2['pass']}",
        "",
        "## Level 3 -- moved + monotone + fidelity",
        f"- alpha-ladder response (context gain): {l3['alpha_response']}",
        f"- monotone in context gain: {l3['response_monotone_in_context_gain']}",
        f"- CRPS gap vs 734a: {l3['crps_gap']:.4g} | Energy gap: {l3['energy_gap']:.4g} "
        f"(guardrail +/-{l3['fidelity_guardrail']})",
        f"- fidelity pass: {l3['fidelity_pass']} | PASS: {l3['pass']}",
        "",
        f"> {result['headline']}",
        f"> NOTE: {l3['note']}",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained-checkpoint", default=DEFAULT_TRAINED_CHECKPOINT)
    parser.add_argument("--base-checkpoint", default=BASE_734A_DEFAULT)
    parser.add_argument("--support-bank", default=DEFAULT_SUPPORT_BANK)
    parser.add_argument("--manifest-examples", default=DEFAULT_MANIFEST_EXAMPLES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embed-mode", choices=["openai", "hash"], default="openai")
    parser.add_argument("--dotenv-path", default=".env")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--ladder-windows", type=int, default=20,
                        help="how many held-out windows to use for the alpha-ladder + fidelity")
    parser.add_argument("--max-windows", type=int, default=0,
                        help="if >0, cap held-out windows (smoke mode)")
    parser.add_argument("--seed", type=int, default=7344)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    result = run_gate(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "track_b_directional_gate.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md_path = write_verifier_report(result, output_dir)

    print(f"[gate] wrote {json_path}")
    print(f"[gate] wrote {md_path}")
    print(f"[gate] OVERALL_PASS={result['overall_pass']} "
          f"L1={result['level1_knob_activated']['pass']} "
          f"L2={result['level2_propagated']['pass']} "
          f"L3={result['level3_moved_monotone_fidelity']['pass']}")
    print(f"[gate] real-shuffled direction-match delta = "
          f"{result['level2_propagated']['real_minus_shuffled']}")


if __name__ == "__main__":
    main()
