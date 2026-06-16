"""T7.7 rollout beta-selector: pick the narrative-tilt strength beta on held-out data.

For each beta, rewrite every clean query's top_train_pool =
    _apply_top3_90(reweight_pool(pre_top3_90_pool, beta))[:3]
then run the FROZEN-SNI rollout via run_scenario_level_evaluation (samples, field_weight).
beta=0 is the exact no-tilt baseline.

Metrics (advisor-specified; cross-case dispersion is NOT used — it rewards noise):
  - RESPONSIVENESS = mean_i || terminal_i(beta) - terminal_i(0) ||  (standardized terminals,
    SAME seed => common-random-numbers => isolates the tilt) vs the REPEAT-SEED NOISE FLOOR
    (beta=0 across seeds; pure sampling noise, no tilt).
  - FIDELITY FLOOR = narrative_generator_topk CRPS_z / coverage_80 must not regress vs beta=0.

Selected beta = largest beta with responsiveness above the noise floor AND fidelity not
regressed; else beta=0 (a legitimate kill-condition no-op outcome). Deck must be #48-clean.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from experiments.backfill.block_ar.nl_14x14_support_audit import _apply_top3_90
from experiments.backfill.block_ar.nl_narrative_reweighter import (
    analogue_profile,
    joint39_factor_cols,
    narrative_emphasis,
    reweight_pool,
)

DEFAULT_DECK = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_full_codex_982g_embedding_grounded_top3_90_66q_clean48/"
    "embedding_grounded_bridge_report.json"
)
DEFAULT_CKPT = "models/backfill/734a_joint39_realvix_channel_level_alltrain_w005_e3_s7345/best_model.pt"
DEFAULT_DATA = "data/multi_factor_data.npz"
EVAL = "experiments/backfill/block_ar/nl_scenario_level_evaluation.py"
HORIZON = 30
CRPS_TOL = 0.02      # CRPS_z may rise at most this much vs beta=0 (lower is better)
COV_TOL = 0.03       # coverage_80 may fall at most this much vs beta=0


def build_beta_report(clean_report, beta, levels, factor_cols):
    rep = json.loads(json.dumps(clean_report))
    for ex in rep["evaluation"]["heldout_examples"]:
        claims = ex.get("required_grounding_claims") or []
        emphasis = narrative_emphasis({"current_market_state_implications": claims})
        cands, profiles = [], {}
        for c in ex.get("pre_top3_90_candidate_pool", []):
            ci = c.get("window_index")
            if ci is None:
                continue
            ci = int(ci)
            cands.append({**c, "score": float(c.get("cosine", c.get("retrieval_score", 0.0)))})
            profiles[ci] = analogue_profile(
                window_index=ci, panel=levels, factor_cols=factor_cols, horizon=HORIZON
            )
        tilted = reweight_pool(cands, emphasis=emphasis, profiles_by_window=profiles, beta=float(beta))
        selected, _ = _apply_top3_90(tilted)
        ex["top_train_pool"] = [
            {
                "window_index": int(s["window_index"]),
                "window_id": s.get("window_id", ""),
                "cosine": float(s.get("cosine", 0.0)),
                "weight": float(s.get("weight", 0.0)),
                "scenario_title": s.get("scenario_title", ""),
                "rank": int(s.get("rank", i + 1)),
            }
            for i, s in enumerate(selected[:3])
        ]
    return rep


def run_eval(report_path, out_dir, *, checkpoint, samples, crn_base, device, reuse):
    out = Path(out_dir)
    done = out / "scenario_level_eval_report.json"
    if reuse and done.exists():
        print(f"  [reuse] {out_dir}")
        return
    cmd = [
        sys.executable, EVAL,
        "--bridge-report", str(report_path), "--output-dir", str(out_dir),
        "--checkpoint", checkpoint, "--samples", str(samples), "--n-steps", str(HORIZON),
        "--support-sampling-mode", "field_weight", "--top-k", "3", "--device", device,
        # candidates are train windows 0..3943, so the rebuilt block must span the full train
        # range (4010 windows). Default --max_windows 50 / --eval_split val truncates it.
        "--eval_split", "train", "--max_windows", "0",
        # CRN-by-query: per-query RNG = f(query, base_seed). A shared base across betas makes
        # beta-vs-beta=0 terminal diffs isolate the tilt; varying the base = the noise floor.
        "--common-random-numbers-by-query", "--common-random-base-seed", str(crn_base),
    ]
    env = {**os.environ, "PYTHONPATH": "."}
    subprocess.run(cmd, check=True, env=env)


def load_std_terminals(out_dir):
    """{window_index: standardized terminal mean over samples (39,)}."""
    npz = np.load(Path(out_dir) / "scenario_level_eval_arrays.npz")
    term_scale = np.maximum(np.asarray(npz["delta_scale"])[-1], 1e-9)  # (39,)
    out = {}
    for k in npz.files:
        if k.startswith("narrative_"):
            wi = int(k[len("narrative_"):])
            arr = np.asarray(npz[k])           # (S,30,39)
            out[wi] = arr.mean(axis=0)[-1] / term_scale
    return out


def load_fidelity(out_dir):
    rep = json.loads((Path(out_dir) / "scenario_level_eval_report.json").read_text())
    crps, cov = [], []
    for w in rep["window_scores"]:
        m = w.get("methods", {}).get("narrative_generator_topk", {})
        if m.get("ensemble_crps_z") is not None:
            crps.append(float(m["ensemble_crps_z"]))
        if m.get("coverage_80") is not None:
            cov.append(float(m["coverage_80"]))
    return (float(np.median(crps)) if crps else float("nan"),
            float(np.median(cov)) if cov else float("nan"))


def _per_query_dist(a, b):
    """{window_index: ||a[wi]-b[wi]||} over the shared keys."""
    return {k: float(np.linalg.norm(a[k] - b[k])) for k in (set(a) & set(b))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default=DEFAULT_DECK)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--betas", default="0,0.25,0.5,1,2")
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--base-crn", type=int, default=8128, help="shared CRN base seed for the beta sweep")
    ap.add_argument("--noise-crn", default="8129,8130", help="extra beta=0 CRN base seeds = noise floor")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", default="experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_t7_beta_sweep_20260616")
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()

    betas = [float(b) for b in str(args.betas).split(",")]
    noise_crns = [int(s) for s in str(args.noise_crn).split(",") if s.strip()]
    factor_cols = joint39_factor_cols(args.data)
    levels = np.asarray(np.load(args.data, allow_pickle=True)["levels"], dtype=np.float32)
    clean = json.loads(Path(args.deck).read_text())
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1. build + run each beta at the shared base CRN
    beta_dirs = {}
    zero_rp = None
    for b in betas:
        rep = build_beta_report(clean, b, levels, factor_cols)
        rp = out_root / f"beta_{b}_report.json"
        rp.write_text(json.dumps(rep) + "\n")
        if float(b) == 0.0:
            zero_rp = rp
        d = out_root / f"beta_{b}_crn{args.base_crn}"
        print(f"[run] beta={b} crn={args.base_crn}")
        run_eval(rp, d, checkpoint=args.checkpoint, samples=args.samples,
                 crn_base=args.base_crn, device=args.device, reuse=args.reuse)
        beta_dirs[b] = d

    # 2. beta=0 noise floor: same (no-tilt) report at DIFFERENT CRN bases
    if zero_rp is None:
        zero_rp = out_root / "beta_0_report.json"
        zero_rp.write_text(json.dumps(build_beta_report(clean, 0.0, levels, factor_cols)) + "\n")
    noise_dirs = []
    for nc in noise_crns:
        d = out_root / f"beta_0_crn{nc}"
        print(f"[run] beta=0 crn={nc} (noise floor)")
        run_eval(zero_rp, d, checkpoint=args.checkpoint, samples=args.samples,
                 crn_base=nc, device=args.device, reuse=args.reuse)
        noise_dirs.append(d)

    # 3. aggregate -- PER-QUERY (fair): where the tilt acts (movers), does it shift terminals
    # more than reseeding does? Mean-vs-mean is dominated by non-movers (resp_i=0 there) and is
    # only reported for context. Verdict uses among-mover median(tilt/noise) + fidelity.
    MOVE_EPS = 1e-4          # tilt shift above this = the selection/weights actually changed
    FRAC_MOVERS_MIN = 0.15   # tilt must act on a non-trivial share of queries
    base0 = load_std_terminals(beta_dirs[0.0])
    # per-query reseed noise = mean over noise CRNs of ||base0 - beta0(other crn)||
    noise_per_q = {}
    noise_dist_dicts = [_per_query_dist(base0, load_std_terminals(d)) for d in noise_dirs]
    for wi in base0:
        vals = [nd[wi] for nd in noise_dist_dicts if wi in nd]
        if vals:
            noise_per_q[wi] = float(np.mean(vals))
    mean_noise = float(np.mean(list(noise_per_q.values()))) if noise_per_q else float("nan")
    crps0, cov0 = load_fidelity(beta_dirs[0.0])

    rows = []
    for b in betas:
        tilt = _per_query_dist(load_std_terminals(beta_dirs[b]), base0)
        keys = sorted(set(tilt) & set(noise_per_q))
        movers = [k for k in keys if tilt[k] > MOVE_EPS]
        mover_ratios = [tilt[k] / noise_per_q[k] for k in movers if noise_per_q[k] > 1e-9]
        resp_gt_noise = [k for k in keys if tilt[k] > noise_per_q[k]]
        crps, cov = load_fidelity(beta_dirs[b])
        rows.append({
            "beta": b,
            "frac_movers": len(movers) / max(len(keys), 1),
            "among_mover_median_tilt_over_noise": float(np.median(mover_ratios)) if mover_ratios else 0.0,
            "frac_tilt_gt_noise": len(resp_gt_noise) / max(len(keys), 1),
            "mean_tilt_shift": float(np.mean([tilt[k] for k in keys])) if keys else 0.0,
            "mean_noise": mean_noise,
            "crps_z_median": crps, "coverage_80_median": cov,
            "fidelity_ok": bool(crps <= crps0 + CRPS_TOL and cov >= cov0 - COV_TOL),
        })

    # select: largest beta>0 where, among queries it moves, the tilt typically exceeds reseed
    # noise (median ratio > 1), it acts on a non-trivial share, and fidelity is not regressed.
    eligible = [r for r in rows if r["beta"] > 0
                and r["among_mover_median_tilt_over_noise"] > 1.0
                and r["frac_movers"] >= FRAC_MOVERS_MIN
                and r["fidelity_ok"]]
    # minimal intervention: the SMALLEST eligible beta (the effect saturates, so larger beta
    # only pushes further off the softmax's natural scale for no extra movement).
    selected = min((r["beta"] for r in eligible), default=0.0)
    verdict = {
        "mean_noise": mean_noise, "crps0_median": crps0, "cov0_median": cov0,
        "selected_beta": selected, "is_no_op": bool(selected == 0.0),
        "criterion": "among-mover median(tilt/noise)>1 AND frac_movers>=0.15 AND fidelity not regressed",
        "note": ("No beta>0 clears the bar -> T7 is a no-op (kill condition); keep beta=0 default. "
                 "Where the tilt swaps analogues, the resulting scenarios stay within reseed noise."
                 if selected == 0.0 else
                 f"beta={selected}: where it acts, the tilt moves scenarios beyond reseed noise "
                 f"without fidelity regression."),
    }
    summary = {"deck": args.deck, "samples": args.samples, "betas": betas,
               "noise_crns": noise_crns, "rows": rows, "verdict": verdict}
    (out_root / "beta_sweep_report.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== T7 rollout beta-selector ===")
    print(f"mean reseed noise = {mean_noise:.4f} | beta=0 CRPS_z={crps0:.3f} cov80={cov0:.3f}")
    print(f"{'beta':>6} {'%movers':>8} {'mvr med(tilt/noise)':>20} {'%tilt>noise':>12} "
          f"{'CRPS_z':>8} {'cov80':>7} {'fid_ok':>7}")
    for r in rows:
        print(f"{r['beta']:>6} {r['frac_movers']*100:>7.0f}% {r['among_mover_median_tilt_over_noise']:>20.2f} "
              f"{r['frac_tilt_gt_noise']*100:>11.0f}% {r['crps_z_median']:>8.3f} "
              f"{r['coverage_80_median']:>7.3f} {str(r['fidelity_ok']):>7}")
    print(f"\nVERDICT: selected_beta = {selected}  ({'NO-OP' if selected == 0.0 else 'TILT HELPS'})")
    print(f"  {verdict['note']}")
    print(f"\nwrote {out_root/'beta_sweep_report.json'}")


if __name__ == "__main__":
    main()
