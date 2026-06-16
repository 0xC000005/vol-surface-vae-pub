"""T7 DECISIVE test: does narrative tilting STEER the generated scenarios directionally?

Responsiveness (terminal shift vs reseed noise) is near-tautological: changing the selected
analogue trivially changes the rollout. The real question is directional correctness:

  (a) do the SELECTED ANALOGUES' own 30-day HISTORIES match the narrative's claimed factor signs?
      (= does retrieval's direction-gate work)   -- expect high
  (b) do the GENERATED forward SCENARIOS' terminal moves match those claimed signs?
      (= does the frozen SNI carry the analogue's direction forward)  -- the T7 question
  and crucially: does beta>0 raise (b) above beta=0 and above the 0.5 chance line?

Finding (2026-06-16): (a)=0.849, (b)=0.428 (<=chance), and beta=0.25 vs beta=0 delta ~ 0 on
movers -> T7 reselects direction-matched analogues but the generator washes out their direction;
no narrative steering. Reusable for the fix-pool-vary-emphasis check (T7.8).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.backfill.block_ar.nl_joint39_anchor_map import joint39_anchor_columns
from experiments.backfill.block_ar.nl_narrative_reweighter import joint39_factor_cols

DEFAULT_SWEEP = "experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_t7_beta_sweep_20260616"
HORIZON = 30


def _claims(report_path):
    rep = json.loads(Path(report_path).read_text())
    out, sel = {}, {}
    for ex in rep["evaluation"]["heldout_examples"]:
        wi = int(ex["window_index"])
        out[wi] = [(str(c["market"]).upper(), int(c.get("sign", 0)))
                   for c in ex.get("required_grounding_claims", []) if int(c.get("sign", 0)) != 0]
        sel[wi] = [int(x["window_index"]) for x in ex["top_train_pool"]]
    return out, sel


def _scenario_hitrate(arrays_path, claims, anchor_col, restrict=None):
    arr = np.load(arrays_path)
    hits = n = 0
    for wi, cls in claims.items():
        if restrict is not None and wi not in restrict:
            continue
        k = f"narrative_{wi}"
        if k not in arr.files:
            continue
        term = arr[k].mean(axis=0)[-1]
        for f, s in cls:
            if f in anchor_col:
                hits += int(np.sign(term[anchor_col[f]]) == s); n += 1
    return (hits / n if n else float("nan")), n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default=DEFAULT_SWEEP)
    ap.add_argument("--betas", default="0.0,0.25")
    ap.add_argument("--data", default="data/multi_factor_data.npz")
    args = ap.parse_args()
    betas = [b for b in str(args.betas).split(",")]
    anchor_col = joint39_anchor_columns()
    level_col = joint39_factor_cols(args.data)
    levels = np.asarray(np.load(args.data, allow_pickle=True)["levels"], dtype=np.float32)

    b0, b1 = betas[0], betas[-1]
    claims, sel0 = _claims(f"{args.sweep_dir}/beta_{b0}_report.json")
    _, sel1 = _claims(f"{args.sweep_dir}/beta_{b1}_report.json")
    movers = {wi for wi in sel0 if sorted(sel0[wi]) != sorted(sel1.get(wi, []))}

    # (a) selected-analogue history hit-rate (beta=b1 selection)
    ah = an = 0
    for wi, cls in claims.items():
        for ai in sel1.get(wi, []):
            for f, s in cls:
                if f not in level_col:
                    continue
                e = min(ai + HORIZON - 1, levels.shape[0] - 1)
                ah += int(np.sign(float(levels[e, level_col[f]] - levels[ai, level_col[f]])) == s); an += 1
    print(f"(a) selected-analogue HISTORY hit-rate vs claims = {ah/an:.3f} (n={an})")

    for label, restrict in [("ALL", None), ("MOVERS", movers)]:
        h0, n0 = _scenario_hitrate(f"{args.sweep_dir}/beta_{b0}_crn8128/scenario_level_eval_arrays.npz",
                                   claims, anchor_col, restrict)
        h1, n1 = _scenario_hitrate(f"{args.sweep_dir}/beta_{b1}_crn8128/scenario_level_eval_arrays.npz",
                                   claims, anchor_col, restrict)
        print(f"(b) generated-SCENARIO hit-rate [{label} n={n0}]: beta={b0}: {h0:.3f}  "
              f"beta={b1}: {h1:.3f}  delta={h1-h0:+.3f}  (chance=0.5)")
    print(f"\nmovers={len(movers)}; verdict: STEERS only if (b) beta>0 clearly exceeds beta=0 AND 0.5.")


if __name__ == "__main__":
    main()
