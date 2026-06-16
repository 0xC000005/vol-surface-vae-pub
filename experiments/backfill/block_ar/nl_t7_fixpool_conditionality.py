"""T7.8 fix-pool-vary-emphasis conditionality test (advisor's clean design).

Hold a host query's retrieved pool + start FIXED; apply several CONTRASTING synthetic emphasis
profiles; reweight->top3/90->frozen-SNI rollout each. Same pool across emphases => all variation
is the emphasis. Two questions:
  (a) SEPARATION: do different emphases produce distinguishable scenarios? (cross-emphasis terminal
      spread at beta=0.25 vs the beta=0 control, where emphasis is ignored => identical).
  (b) STEERING (decisive): does emphasis-m's scenario lean toward emphasis-m's OWN claimed factor
      signs? directional hit-rate vs the 0.5 chance line, beta=0.25 vs beta=0.

Caveat: host pools are direction-gated for the host's own narrative, so they are somewhat
homogeneous -> separation is bounded by pool diversity. The directional hit-rate is the cleaner
signal and is robust to that. Corroborates the matched-episode directional finding (generator
washout: history 0.849 -> forward 0.428).
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
from experiments.backfill.block_ar.nl_joint39_anchor_map import joint39_anchor_columns
from experiments.backfill.block_ar.nl_narrative_reweighter import (
    analogue_profile,
    joint39_factor_cols,
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

# Contrasting synthetic emphases (factor -> {direction, salience}); signs via dir_sign.
EMPHASES = {
    "risk_off": {"SPX": ("down", 1.0), "VIX": ("up", 1.0), "BBB_OAS": ("wider", 0.7),
                 "GOLD": ("up", 0.6), "US10Y": ("down", 0.5)},
    "risk_on": {"SPX": ("up", 1.0), "VIX": ("down", 1.0), "BBB_OAS": ("tighter", 0.7),
                "CRUDE_OIL": ("up", 0.6)},
    "rates_up": {"US10Y": ("up", 1.0), "US2Y": ("up", 1.0), "DXY": ("up", 0.6)},
    "usd_up": {"DXY": ("up", 1.0), "USDJPY": ("up", 0.8), "GOLD": ("down", 0.6)},
}


def _emphasis_dict(spec):
    return {f: {"direction": d, "salience": s} for f, (d, s) in spec.items()}


def build_fixpool_report(clean, hosts, beta, levels, factor_cols):
    """One query per (host, emphasis): same host pool + start, different emphasis tilt."""
    by_w = {int(e["window_index"]): e for e in clean["evaluation"]["heldout_examples"]}
    out = json.loads(json.dumps(clean))
    rows = []
    for wi in hosts:
        ex = by_w[wi]
        cands, profiles = [], {}
        for c in ex.get("pre_top3_90_candidate_pool", []):
            ci = int(c["window_index"])
            cands.append({**c, "score": float(c.get("cosine", 0.0))})
            profiles[ci] = analogue_profile(window_index=ci, panel=levels,
                                            factor_cols=factor_cols, horizon=HORIZON)
        for ename, spec in EMPHASES.items():
            tilted = reweight_pool(cands, emphasis=_emphasis_dict(spec),
                                   profiles_by_window=profiles, beta=float(beta))
            sel, _ = _apply_top3_90(tilted)
            row = json.loads(json.dumps(ex))
            row["query_id"] = f"fixpool_{wi}_{ename}"
            row["role"] = "anchor"
            row["top_train_pool"] = [
                {"window_index": int(s["window_index"]), "window_id": s.get("window_id", ""),
                 "cosine": float(s.get("cosine", 0.0)), "weight": float(s.get("weight", 0.0)),
                 "rank": i + 1}
                for i, s in enumerate(sel[:3])
            ]
            rows.append(row)
    out["evaluation"]["heldout_examples"] = rows
    return out


def run_eval(report_path, out_dir, *, checkpoint, samples, crn_base, device, reuse):
    if reuse and (Path(out_dir) / "scenario_level_eval_report.json").exists():
        print(f"  [reuse] {out_dir}"); return
    cmd = [sys.executable, EVAL, "--bridge-report", str(report_path), "--output-dir", str(out_dir),
           "--checkpoint", checkpoint, "--samples", str(samples), "--n-steps", str(HORIZON),
           "--support-sampling-mode", "field_weight", "--top-k", "3", "--device", device,
           "--eval_split", "train", "--max_windows", "0",
           "--allow-duplicate-query-windows",
           "--common-random-numbers-by-query", "--common-random-base-seed", str(crn_base)]
    subprocess.run(cmd, check=True, env={**os.environ, "PYTHONPATH": "."})


def _suffix(row_no, query_id):
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(query_id))
    return f"{row_no:04d}_{safe}"


def _terminals(out_dir, ordered_qids):
    """{(row_no, query_id): standardized terminal (39,)} using allow-duplicate key scheme."""
    npz = np.load(Path(out_dir) / "scenario_level_eval_arrays.npz")
    ts = np.maximum(np.asarray(npz["delta_scale"])[-1], 1e-9)
    out = {}
    for row_no, qid in enumerate(ordered_qids):
        k = f"narrative_{_suffix(row_no, qid)}"
        if k in npz.files:
            out[(row_no, qid)] = npz[k].mean(axis=0)[-1] / ts
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default=DEFAULT_DECK)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--n-hosts", type=int, default=12)
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", default="experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_t7_fixpool_20260616")
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()

    anchor_col = joint39_anchor_columns()
    factor_cols = joint39_factor_cols(args.data)
    levels = np.asarray(np.load(args.data, allow_pickle=True)["levels"], dtype=np.float32)
    clean = json.loads(Path(args.deck).read_text())
    all_w = [int(e["window_index"]) for e in clean["evaluation"]["heldout_examples"]]
    hosts = all_w[:args.n_hosts]
    out_root = Path(args.output_dir); out_root.mkdir(parents=True, exist_ok=True)

    enames = list(EMPHASES)
    qids = [f"fixpool_{wi}_{en}" for wi in hosts for en in enames]   # row order = build order

    results = {}
    for beta in (0.0, 0.25):
        rep = build_fixpool_report(clean, hosts, beta, levels, factor_cols)
        rp = out_root / f"fixpool_beta_{beta}_report.json"; rp.write_text(json.dumps(rep) + "\n")
        d = out_root / f"fixpool_beta_{beta}"
        print(f"[run] fix-pool beta={beta}")
        run_eval(rp, d, checkpoint=args.checkpoint, samples=args.samples,
                 crn_base=8128, device=args.device, reuse=args.reuse)
        results[beta] = _terminals(d, qids)

    # index terminals by (host, emphasis)
    def by_host_emph(term):
        m = {}
        for (row_no, qid), v in term.items():
            _, wi, en = qid.split("_", 2)
            m[(int(wi), en)] = v
        return m

    rows = []
    for beta in (0.0, 0.25):
        hb = by_host_emph(results[beta])
        # (a) separation: per host, mean pairwise terminal distance across the M emphases
        seps = []
        for wi in hosts:
            vecs = [hb[(wi, en)] for en in enames if (wi, en) in hb]
            if len(vecs) >= 2:
                d = [np.linalg.norm(vecs[i] - vecs[j]) for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
                seps.append(float(np.mean(d)))
        # (b) steering: does emphasis-en's scenario lean toward en's claimed signs?
        hits = tot = 0
        for wi in hosts:
            for en in enames:
                if (wi, en) not in hb:
                    continue
                term = hb[(wi, en)]
                for f, (dr, _s) in EMPHASES[en].items():
                    if f not in anchor_col:
                        continue
                    sgn = 1 if dr in ("up", "wider") else -1
                    hits += int(np.sign(term[anchor_col[f]]) == sgn); tot += 1
        rows.append({"beta": beta,
                     "mean_cross_emphasis_separation": float(np.mean(seps)) if seps else 0.0,
                     "directional_hit_rate": hits / tot if tot else float("nan"),
                     "n_directional_claims": tot})

    summary = {"deck": args.deck, "n_hosts": len(hosts), "emphases": enames,
               "samples": args.samples, "rows": rows}
    (out_root / "fixpool_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n=== T7.8 fix-pool-vary-emphasis ===")
    print(f"{'beta':>6} {'sep(cross-emph)':>16} {'dir hit-rate':>13} {'(chance=0.5)':>12}")
    for r in rows:
        print(f"{r['beta']:>6} {r['mean_cross_emphasis_separation']:>16.4f} "
              f"{r['directional_hit_rate']:>13.3f}  n={r['n_directional_claims']}")
    b25 = next(r for r in rows if r["beta"] == 0.25)
    steers = b25["directional_hit_rate"] > 0.55
    print(f"\nVERDICT: {'STEERS' if steers else 'NO directional steering'} "
          f"(beta=0.25 dir hit-rate {b25['directional_hit_rate']:.3f} vs 0.5 chance); "
          f"separation={b25['mean_cross_emphasis_separation']:.3f} (0 at beta=0 by construction).")
    print(f"wrote {out_root/'fixpool_report.json'}")


if __name__ == "__main__":
    main()
