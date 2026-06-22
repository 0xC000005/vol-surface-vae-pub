#!/usr/bin/env python
"""Track B B-WIDTH STEP 2 -- WIDTH GATE (held-out, pre-registered).

Does the conditioned ensemble's fan WIDTH track REAL grounded narrative-intensity MORE than
SHUFFLED intensity? Runs on the SAME held-out val-frame narrative windows as the directional
gate, reusing its canonical window-build + CRN sampling, but headlines on the WIDTH axis with a
GROUNDED intensity scalar (nl_track_b_grounded_intensity, prose-parse, NO OpenAI) and a
permutation null band.

PRE-REGISTERED VERDICT (set before any bwidth number is read):
  PASS (promotable severity dial):
     spearman_real - spearman_shuffled > 0.15  AND  spearman_real > 0.30
     AND fidelity (CRPS & Energy vs frozen 734a) within +/-0.015.
     => faithful narrative-driven severity dial.
  KILL:
     width responds to PRESENCE not INTENSITY (real ~ shuffled) OR fidelity violated.
     => width steering via this channel is not faithful; dispersion nudge stays
        history/regime-driven (734a's validated width-conditioning, turb/calm 1.229).

Measurement is intensity vs width_DELTA = width(present=True) - width(present=False), CRN-shared
seed per window (regime/history held fixed -- the marginal narrative effect, NOT regime-confounded
raw width). The shuffled arm permutes intensity LABELS across held-out windows against the FIXED
per-window width_delta vector (a >=1000-permutation null gives the honest band; a single shuffle's
std ~0.09 at N~115 is larger than the 0.15 margin). 734a + the trained checkpoint are read-only.
"""

from __future__ import annotations

import argparse
import json
import sys
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
from experiments.backfill.block_ar.nl_track_b_grounded_intensity import (  # noqa: E402
    grounded_intensity_for_windows,
    grounded_intensity_with_raw,
)
from experiments.backfill.block_ar.nl_track_b_width_pilot import (  # noqa: E402
    _spearman,
    cached_embeddings,
    permutation_null,
)
from experiments.backfill.block_ar.nl_track_b_directional_gate import (  # noqa: E402
    _build_masks,
    build_gate_windows,
    conditioned_samples,
    fidelity_scores,
    terminal_delta_from_increments,
)
from experiments.backfill.block_ar.nl_scenario_level_evaluation import (  # noqa: E402
    build_delta_scale,
)
from experiments.backfill.block_ar.train_track_b_generator_conditioning import (  # noqa: E402
    DEFAULT_CHECKPOINT as BASE_734A_DEFAULT,
    DEFAULT_MANIFEST_EXAMPLES,
    DEFAULT_SUPPORT_BANK,
    assert_bank_alignment,
    load_window_texts,
)

DEFAULT_TRAINED = "models/backfill/generator_conditioning_probe_bwidth/best_model.pt"
DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/track_b_width_gate"
)
VERIFIER_DIR = "docs/research_protocols/nl_prefix_latent_verifier_reports"
EMBED_CACHE = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "track_b_width_pilot/embed_cache_small.npz"
)

PREREGISTERED = {
    "pass_spearman_real_min": 0.30,
    "pass_real_minus_shuffled_min": 0.15,
    "fidelity_guardrail_crps": 0.015,
    "fidelity_guardrail_energy": 0.015,
    # SEVERITY-DIAL (intercept) bars: a faithful dial must widen IN PROPORTION to absolute
    # intensity, i.e. NOT uniformly inflate the fan for every present narrative regardless of
    # severity. Regress width_delta on ABSOLUTE intensity; require a positive slope AND a near-
    # zero intercept (low-absolute-intensity present narratives must NOT widen the baseline).
    # Quantified as: the low-intensity-tercile mean |width_delta| must be a SMALL fraction of the
    # high-intensity-tercile mean width_delta (the presence-inflation guard). These bars guard
    # the task's own KILL condition (PRESENCE not INTENSITY) from passing as a rank-only PASS.
    "dial_slope_min": 0.0,                    # OLS slope width_delta ~ abs_intensity must be > 0
    "dial_low_tercile_inflation_max_frac": 0.5,  # low-tercile mean width_delta <= 0.5 * high-tercile
}


@torch.no_grad()
def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        args.device if (torch.cuda.is_available() or str(args.device) == "cpu") else "cpu"
    )
    model, payload = load_model(args.trained_checkpoint, device)
    assert getattr(model.cfg, "narrative_conditioning", False), "trained ckpt must have narrative_conditioning=True"
    assert model.narrative_adapter is not None, "trained ckpt missing narrative_adapter"
    base_model, _bp = load_model(args.base_checkpoint, device)

    windows = build_gate_windows(args.base_checkpoint, payload)
    n_windows = int(windows["history_level"].shape[0])
    assert_bank_alignment(
        {k: windows[k] for k in ("history_level", "history_norm", "center", "scale", "drift_feature")},
        args.support_bank,
    )
    masks = _build_masks(n_windows)
    delta_scale = build_delta_scale(windows["future_delta"][masks["train"]])
    texts = load_window_texts(args.manifest_examples, n_windows)
    embeddings = cached_embeddings(texts, args.embed_mode, args.dotenv_path, args.embed_cache)
    val_rows = np.nonzero(masks["val"])[0]
    heldout_rows = np.array([r for r in val_rows if int(r) in embeddings], dtype=np.int64)
    if int(args.max_windows) > 0:
        heldout_rows = heldout_rows[: int(args.max_windows)]
    if heldout_rows.size < 3:
        raise RuntimeError("need >=3 held-out narrative windows -> STOP")

    heldout_texts = {int(r): texts[int(r)] for r in heldout_rows}
    intensity_norm = grounded_intensity_for_windows(heldout_texts)
    intensity_both = grounded_intensity_with_raw(heldout_texts)  # raw (absolute) + norm

    # L1: knob activated
    real_norms, absent_norms = [], []
    for r in heldout_rows[: min(16, heldout_rows.size)]:
        emb = torch.from_numpy(embeddings[int(r)].reshape(1, -1).astype(np.float32)).to(device)
        real_norms.append(float(model.narrative_adapter(emb, torch.tensor([True], device=device)).norm()))
        absent_norms.append(float(model.narrative_adapter(emb, torch.tensor([False], device=device)).norm()))
    level1 = {
        "mean_context_norm_real": float(np.mean(real_norms)),
        "max_context_norm_present_false": float(np.max(absent_norms)),
        "pass": bool(np.mean(real_norms) > 1e-3 and np.max(absent_norms) <= 1e-8),
    }

    # width_delta per window (CRN) + grounded intensity
    width_cond, width_base, inten, abs_inten = [], [], [], []
    crps_c, crps_b, en_c, en_b = [], [], [], []
    for r in heldout_rows:
        r = int(r)
        seed_r = int(args.seed) + r
        inc_on = conditioned_samples(
            model, windows, r, narrative_emb=embeddings[r], present=True,
            n_samples=args.n_samples, n_steps=args.n_steps, chunk_size=args.chunk_size,
            seed=seed_r, device=device,
        )
        inc_off = conditioned_samples(
            model, windows, r, narrative_emb=None, present=False,
            n_samples=args.n_samples, n_steps=args.n_steps, chunk_size=args.chunk_size,
            seed=seed_r, device=device,
        )
        width_cond.append(float(terminal_delta_from_increments(inc_on).std(axis=0).mean()))
        width_base.append(float(terminal_delta_from_increments(inc_off).std(axis=0).mean()))
        inten.append(float(intensity_norm[r]))
        abs_inten.append(float(intensity_both[r]["raw"]))
        # fidelity: conditioned vs frozen 734a (CRN base noise)
        inc_734a = conditioned_samples(
            base_model, windows, r, narrative_emb=None, present=False,
            n_samples=args.n_samples, n_steps=args.n_steps, chunk_size=args.chunk_size,
            seed=seed_r, device=device,
        )
        sc = fidelity_scores(inc_on, windows, r, delta_scale)
        sb = fidelity_scores(inc_734a, windows, r, delta_scale)
        crps_c.append(sc["ensemble_crps_z"]); crps_b.append(sb["ensemble_crps_z"])
        en_c.append(sc["energy_score_z"]); en_b.append(sb["energy_score_z"])

    wc = np.asarray(width_cond); wb = np.asarray(width_base); it = np.asarray(inten)
    ai = np.asarray(abs_inten)
    width_delta = wc - wb
    rng = np.random.default_rng(int(args.seed) + 999)
    sp_real = _spearman(it, width_delta)
    sp_shuf = _spearman(it[rng.permutation(it.size)], width_delta)
    null = permutation_null(it, width_delta, args.n_perm, int(args.seed) + 1)

    # --- SEVERITY-DIAL (intercept/level) analysis on ABSOLUTE intensity ------------------
    # A faithful dial widens IN PROPORTION to absolute intensity: positive OLS slope AND a
    # near-zero intercept (low-absolute-intensity present narratives must not inflate the fan).
    # This catches the task's KILL pattern "responds to PRESENCE not INTENSITY" that a rank-only
    # spearman/real-minus-shuffled cannot (a uniform +offset is rank-invisible).
    if ai.std() > 0:
        slope, intercept = np.polyfit(ai, width_delta, 1)
    else:
        slope, intercept = float("nan"), float(width_delta.mean())
    # tercile split on absolute intensity (presence-inflation guard)
    order = np.argsort(ai)
    tcut = max(1, ai.size // 3)
    low_idx = order[:tcut]
    high_idx = order[-tcut:]
    low_mean_delta = float(width_delta[low_idx].mean())
    high_mean_delta = float(width_delta[high_idx].mean())
    # If the high-intensity tercile does NOT widen (<=0), there is no proportional dial to speak
    # of: the inflation fraction is undefined/degenerate -> a large finite sentinel that fails the
    # bar (a dial requires the SEVERE windows to be the ones that widen). JSON-safe (no Infinity).
    if high_mean_delta > 1e-9:
        low_inflation_frac = float(abs(low_mean_delta) / high_mean_delta)
    else:
        low_inflation_frac = 1e9
    severity_dial = {
        "ols_slope_width_delta_vs_abs_intensity": float(slope),
        "ols_intercept": float(intercept),
        "low_intensity_tercile_mean_width_delta": low_mean_delta,
        "high_intensity_tercile_mean_width_delta": high_mean_delta,
        "low_over_high_inflation_frac": low_inflation_frac,
        "abs_intensity_range": [float(ai.min()), float(ai.max())],
        "interpretation": (
            "faithful DIAL => positive slope AND low-tercile width_delta is a small fraction of "
            "high-tercile (intercept ~0); uniform PRESENCE-inflation => slope ~0 / large "
            "low-tercile widening regardless of severity (the task's KILL pattern, rank-invisible)."
        ),
    }
    dial_pass = bool(
        np.isfinite(slope) and slope > PREREGISTERED["dial_slope_min"]
        and np.isfinite(low_inflation_frac)
        and low_inflation_frac <= PREREGISTERED["dial_low_tercile_inflation_max_frac"]
    )
    severity_dial["dial_pass"] = dial_pass

    crps_gap = float(np.mean(crps_c) - np.mean(crps_b))
    energy_gap = float(np.mean(en_c) - np.mean(en_b))
    fidelity_pass = bool(
        abs(crps_gap) <= PREREGISTERED["fidelity_guardrail_crps"]
        and abs(energy_gap) <= PREREGISTERED["fidelity_guardrail_energy"]
    )
    real_minus_shuffled = None if sp_real is None or sp_shuf is None else sp_real - sp_shuf

    width_response = {
        "spearman_real_intensity_vs_width_delta": sp_real,
        "spearman_shuffled_one_draw": sp_shuf,
        "real_minus_shuffled": real_minus_shuffled,
        "permutation_null": null,
        "mean_width_conditioned": float(wc.mean()),
        "mean_width_baseline": float(wb.mean()),
        "mean_width_delta": float(width_delta.mean()),
        "mean_abs_width_delta": float(np.abs(width_delta).mean()),
        "intensity_distribution": {
            "min": float(it.min()), "max": float(it.max()), "std": float(it.std()),
            "n_unique": int(np.unique(np.round(it, 6)).size),
        },
    }
    fidelity = {
        "crps_conditioned": float(np.mean(crps_c)), "crps_baseline_734a": float(np.mean(crps_b)),
        "crps_gap": crps_gap, "energy_conditioned": float(np.mean(en_c)),
        "energy_baseline_734a": float(np.mean(en_b)), "energy_gap": energy_gap,
        "guardrail": PREREGISTERED["fidelity_guardrail_crps"], "fidelity_pass": fidelity_pass,
    }

    width_response["severity_dial"] = severity_dial
    pass_width = bool(
        sp_real is not None and sp_real > PREREGISTERED["pass_spearman_real_min"]
        and real_minus_shuffled is not None
        and real_minus_shuffled > PREREGISTERED["pass_real_minus_shuffled_min"]
    )
    # PASS requires BOTH the rank signal (real >> shuffled) AND the severity-dial (intercept-
    # aware) check -- a rank PASS with uniform presence-inflation is the task's KILL, not a PASS.
    verdict = "PASS" if (pass_width and dial_pass and fidelity_pass and level1["pass"]) else "KILL"
    result = {
        "schema_version": "nl_track_b_width_gate_v1",
        "track": "B-width (grounded-intensity severity dial)",
        "trained_checkpoint": str(args.trained_checkpoint),
        "base_checkpoint_734a": str(args.base_checkpoint),
        "preregistered": PREREGISTERED,
        "n_heldout_windows": int(heldout_rows.size),
        "embed_mode": str(args.embed_mode),
        "n_samples": int(args.n_samples),
        "level1_knob_activated": level1,
        "width_response": width_response,
        "fidelity": fidelity,
        "verdict": verdict,
        "verdict_basis": {
            "width_rank_signal_pass": pass_width,
            "severity_dial_pass": dial_pass,
            "fidelity_pass": fidelity_pass,
            "level1_pass": level1["pass"],
        },
        "measurement_note": (
            "intensity vs width_DELTA (present=True - present=False, CRN); raw width is "
            "regime-confounded and NOT used for the verdict. NO OpenAI for intensity."
        ),
    }
    return result


def write_report(result: dict[str, Any]) -> Path:
    today = date.today().isoformat()
    md = Path(VERIFIER_DIR) / f"{today}_track_b_width_gate.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    wr = result["width_response"]; fi = result["fidelity"]; nl = wr["permutation_null"]
    lines = [
        f"# Track B WIDTH gate -- {today}",
        "",
        f"- Trained checkpoint: `{result['trained_checkpoint']}`",
        f"- Frozen 734a base: `{result['base_checkpoint_734a']}`",
        f"- Held-out val windows with narrative: {result['n_heldout_windows']}; embed_mode={result['embed_mode']}",
        f"- **VERDICT: {result['verdict']}**",
        "",
        "## Width response (grounded intensity vs width_DELTA)",
        f"- spearman REAL: {wr['spearman_real_intensity_vs_width_delta']} (PASS bar > {result['preregistered']['pass_spearman_real_min']})",
        f"- spearman SHUFFLED (one draw): {wr['spearman_shuffled_one_draw']}",
        f"- **real - shuffled: {wr['real_minus_shuffled']}** (PASS bar > {result['preregistered']['pass_real_minus_shuffled_min']})",
        f"- permutation null: mean {nl['null_mean']}, std {nl['null_std']}, p97.5(abs) {nl['null_p975_abs']}, "
        f"exceeds_upper_tail {nl['exceeds_null_upper_tail']}, p2s {nl['p_value_two_sided']}",
        f"- width: cond {wr['mean_width_conditioned']:.4f} base {wr['mean_width_baseline']:.4f} "
        f"mean|delta| {wr['mean_abs_width_delta']:.5f}",
        f"- intensity spread: std {wr['intensity_distribution']['std']:.3f} "
        f"n_unique {wr['intensity_distribution']['n_unique']}",
        "",
        "## Severity-dial (intercept-aware, ABSOLUTE intensity)",
        f"- OLS slope (width_delta ~ abs_intensity): {wr['severity_dial']['ols_slope_width_delta_vs_abs_intensity']:.5g} "
        f"(bar > {result['preregistered']['dial_slope_min']})",
        f"- OLS intercept: {wr['severity_dial']['ols_intercept']:.5g}",
        f"- low-tercile mean width_delta: {wr['severity_dial']['low_intensity_tercile_mean_width_delta']:.5g} "
        f"| high-tercile: {wr['severity_dial']['high_intensity_tercile_mean_width_delta']:.5g}",
        f"- low/high inflation frac: {wr['severity_dial']['low_over_high_inflation_frac']:.4g} "
        f"(bar <= {result['preregistered']['dial_low_tercile_inflation_max_frac']})",
        f"- **dial pass (proportional, not uniform presence-inflation): {wr['severity_dial']['dial_pass']}**",
        "",
        "## Fidelity guardrail (vs frozen 734a)",
        f"- CRPS gap {fi['crps_gap']:.4g} | Energy gap {fi['energy_gap']:.4g} (guardrail +/-{fi['guardrail']})",
        f"- fidelity pass: {fi['fidelity_pass']}",
        "",
        f"> {result['measurement_note']}",
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained-checkpoint", default=DEFAULT_TRAINED)
    parser.add_argument("--base-checkpoint", default=BASE_734A_DEFAULT)
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
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7344)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    result = run_gate(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "track_b_width_gate.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = write_report(result)
    print(f"[width-gate] wrote {json_path}")
    print(f"[width-gate] wrote {md}")
    print(f"[width-gate] VERDICT={result['verdict']} "
          f"spearman_real={result['width_response']['spearman_real_intensity_vs_width_delta']} "
          f"real-shuffled={result['width_response']['real_minus_shuffled']} "
          f"crps_gap={result['fidelity']['crps_gap']:.4g}")


if __name__ == "__main__":
    main()
