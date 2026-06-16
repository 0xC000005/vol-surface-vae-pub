"""T7 free necessary-condition gate (no GPU, no rollout).

The reweighter acts in exactly one place: reweight_pool() adjusts candidate scores by
beta * match(emphasis, analogue_profile) BEFORE _apply_top3_90. If, over clean causal pools,
sweeping beta does NOT move the top3/90 OUTPUT (selected window_ids + their normalized weights),
then no downstream frozen-SNI rollout can differ either -> T7 is a structural no-op and we stop
(kill condition), GPU-free. This mirrors the falsification-before-machinery rule.

Inputs (read-only):
  - held-out deck: grounded_text_preference bridge report (66 queries, each with a candidate pool
    + required_grounding_claims = the per-factor emphasis the retrieval direction-checked against).
  - data/multi_factor_data.npz levels (for analogue_profile factor moves).

Cleanliness: candidates with (query_window_index - candidate_window_index) < temporal_gap are
DROPPED (the #48 causal-gap fix applied post-hoc to the pre-#48 deck).

Outputs: per-beta aggregate selection movement vs beta=0 (set churn + weight L1), the in-pool
match-value spread (near-constant match => pre-aligned pool => no room to tilt), and a verdict.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.backfill.block_ar.nl_14x14_support_audit import _apply_top3_90
from experiments.backfill.block_ar.nl_narrative_reweighter import (
    analogue_profile,
    joint39_factor_cols,
    match,
    narrative_emphasis,
    reweight_pool,
)

DEFAULT_DECK = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_full_codex_982g_grounded_text_preference_reranker_984a/"
    "grounded_text_preference_bridge_report.json"
)
DEFAULT_DATA = "data/multi_factor_data.npz"
DEFAULT_BETAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
TEMPORAL_GAP = 30
HORIZON = 30


def _pool_to_candidates(pool, query_index, factor_cols, levels, temporal_gap):
    """Clean candidate list (score<-cosine) + profiles_by_window, dropping causal-gap violators."""
    cands, profiles = [], {}
    for item in pool:
        ci = item.get("window_index")
        if ci is None:
            continue
        ci = int(ci)
        if int(query_index) - ci < int(temporal_gap):  # #48 causal gap
            continue
        if ci + HORIZON - 1 >= levels.shape[0]:
            continue
        score = item.get("cosine", item.get("retrieval_score", item.get("score")))
        if score is None:
            continue
        cands.append({"window_index": ci, "score": float(score)})
        profiles[ci] = analogue_profile(
            window_index=ci, panel=levels, factor_cols=factor_cols, horizon=HORIZON
        )
    return cands, profiles


def _selection(cands):
    sel, _ = _apply_top3_90(cands)
    return {int(c["window_index"]): float(c["weight"]) for c in sel}


def _movement(base: dict[int, float], other: dict[int, float]) -> tuple[float, float]:
    """(set churn in [0,1] = 1-Jaccard, weight L1 over the union in [0,2])."""
    bk, ok = set(base), set(other)
    jacc = len(bk & ok) / max(len(bk | ok), 1)
    churn = 1.0 - jacc
    l1 = sum(abs(other.get(k, 0.0) - base.get(k, 0.0)) for k in (bk | ok))
    return churn, l1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default=DEFAULT_DECK)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--pool-key", default="pre_top3_90_candidate_pool",
                    help="pre_top3_90_candidate_pool (direction-gated) or top_train_pool (raw)")
    ap.add_argument("--betas", default=",".join(str(b) for b in DEFAULT_BETAS))
    ap.add_argument("--temporal-gap", type=int, default=TEMPORAL_GAP)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    betas = [float(b) for b in str(args.betas).split(",")]
    factor_cols = joint39_factor_cols(args.data)
    levels = np.asarray(np.load(args.data, allow_pickle=True)["levels"], dtype=np.float32)
    deck = json.loads(Path(args.deck).read_text(encoding="utf-8"))
    examples = deck["evaluation"]["heldout_examples"]

    per_beta = {b: {"churn": [], "l1": [], "changed": 0} for b in betas}
    match_spreads, emphasis_sizes, pool_sizes, dropped = [], [], [], 0
    n_used = 0
    for ex in examples:
        qi = ex.get("window_index")
        claims = ex.get("required_grounding_claims") or []
        if qi is None or not claims:
            continue
        emphasis = narrative_emphasis({"current_market_state_implications": claims})
        cands, profiles = _pool_to_candidates(
            ex.get(args.pool_key) or [], qi, factor_cols, levels, args.temporal_gap
        )
        dropped += len(ex.get(args.pool_key) or []) - len(cands)
        if len(cands) < 2:
            continue
        n_used += 1
        emphasis_sizes.append(len(emphasis))
        pool_sizes.append(len(cands))
        # in-pool match spread: near-constant => no room for beta to re-rank
        ms = [match(emphasis, profiles[c["window_index"]]) for c in cands]
        match_spreads.append(float(np.std(ms)))
        base = _selection(cands)
        for b in betas:
            tilted = reweight_pool(cands, emphasis=emphasis, profiles_by_window=profiles, beta=b)
            sel = _selection(tilted)
            churn, l1 = _movement(base, sel)
            per_beta[b]["churn"].append(churn)
            per_beta[b]["l1"].append(l1)
            if churn > 1e-9 or l1 > 1e-6:
                per_beta[b]["changed"] += 1

    summary = {
        "deck": args.deck, "pool_key": args.pool_key, "temporal_gap": args.temporal_gap,
        "n_queries_used": n_used, "candidates_dropped_causal_gap": dropped,
        "mean_pool_size": float(np.mean(pool_sizes)) if pool_sizes else 0.0,
        "mean_emphasis_factors": float(np.mean(emphasis_sizes)) if emphasis_sizes else 0.0,
        "mean_in_pool_match_std": float(np.mean(match_spreads)) if match_spreads else 0.0,
        "betas": {},
    }
    for b in betas:
        c = per_beta[b]["churn"]; l = per_beta[b]["l1"]
        summary["betas"][str(b)] = {
            "median_churn": float(np.median(c)) if c else 0.0,
            "mean_churn": float(np.mean(c)) if c else 0.0,
            "median_weight_l1": float(np.median(l)) if l else 0.0,
            "mean_weight_l1": float(np.mean(l)) if l else 0.0,
            "n_queries_selection_changed": per_beta[b]["changed"],
        }

    # verdict: at the largest beta, did the selection move on a meaningful share of queries?
    bmax = max(betas)
    moved_frac = per_beta[bmax]["changed"] / max(n_used, 1)
    summary["verdict"] = {
        "largest_beta": bmax,
        "frac_queries_moved_at_largest_beta": moved_frac,
        "structural_no_op": bool(moved_frac < 0.05),
        "note": ("T7 is a structural no-op on this deck: beta does not move the top3/90 output, "
                 "so no rollout can differ. Likely cause: direction-gated pool is pre-aligned "
                 "(low in-pool match spread)." if moved_frac < 0.05 else
                 "Selection moves under beta -> proceed to responsiveness+fidelity rollout eval."),
    }

    print(f"=== T7 beta-sensitivity gate ({args.pool_key}) ===")
    print(f"queries used: {n_used} | mean pool size: {summary['mean_pool_size']:.1f} "
          f"| dropped (causal gap): {dropped} | mean emphasis factors: {summary['mean_emphasis_factors']:.1f}")
    print(f"mean in-pool match std (≈0 => pre-aligned, no room to tilt): {summary['mean_in_pool_match_std']:.4f}")
    print(f"{'beta':>6} {'med churn':>10} {'med wL1':>9} {'mean wL1':>9} {'#moved/n':>10}")
    for b in betas:
        s = summary["betas"][str(b)]
        print(f"{b:>6} {s['median_churn']:>10.3f} {s['median_weight_l1']:>9.4f} "
              f"{s['mean_weight_l1']:>9.4f} {s['n_queries_selection_changed']:>7}/{n_used}")
    v = summary["verdict"]
    print(f"\nVERDICT: {'STRUCTURAL NO-OP' if v['structural_no_op'] else 'BETA MOVES SELECTION'} "
          f"(moved {moved_frac*100:.0f}% of queries at beta={bmax})")
    print(f"  {v['note']}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
