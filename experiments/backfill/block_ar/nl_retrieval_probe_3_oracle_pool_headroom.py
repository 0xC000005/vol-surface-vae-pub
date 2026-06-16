#!/usr/bin/env python3
"""Probe 3 — P1 Oracle-Within-Pool Tilt: Headroom Synthesis.

This probe DOES NOT re-run the expensive 1730s oracle replay from 994b.
Instead it reads the existing 994b artifacts and computes:

  (A) Within-pool oracle CRPS headroom vs start-only baseline.
  (B) A full-bank oracle CRPS estimate (best possible if retrieval were perfect)
      derived from the 994b oracle_replay arrays: oracle_crps_pool_best is
      the best CRPS across ALL 50 pool candidates. We compare this to:
        - oracle_crps_selected_top3: what top-3 oracle tilt actually achieved
        - start_only narrative_generator_topk CRPS (the current deployed method)
  (C) "What fraction of headroom does a within-pool oracle tilt capture?"
      vs the full-bank best oracle.
  (D) Cross-check: Spearman(start_distance, oracle_CRPS) from 994b secondary
      diagnostics tells us whether better retrieval (closer start distance)
      would help -- we read that distribution.

Artifacts used (all clean 994b):
  val_frame_oracle_pool_tilt_994b/oracle_within_pool_tilt_probe_report_994b.json
  val_frame_oracle_pool_tilt_994b/paired_block_bootstrap_oracle_tilt_vs_start_only_L30.json
  val_frame_oracle_pool_tilt_994b/scenario_level_eval_report.json  (oracle tilt)
  val_frame_eval_994a_start_only/scenario_level_eval_report.json   (start-only baseline)

The 994b pool is size=50, built from causal bank (0..4009), mutual_gap=30,
query_gap_vs_query=30.  Common random numbers ensure fair CRPS comparison.

Schema: nl_retrieval_probe_3_oracle_pool_headroom_v1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROBE_994B_REPORT = ROOT / (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "val_frame_oracle_pool_tilt_994b/oracle_within_pool_tilt_probe_report_994b.json"
)
BOOTSTRAP_L30 = ROOT / (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "val_frame_oracle_pool_tilt_994b/paired_block_bootstrap_oracle_tilt_vs_start_only_L30.json"
)
ORACLE_TILT_SCENARIO_REPORT = ROOT / (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "val_frame_oracle_pool_tilt_994b/scenario_level_eval_report.json"
)
START_ONLY_SCENARIO_REPORT = ROOT / (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "val_frame_eval_994a_start_only/scenario_level_eval_report.json"
)
OUTPUT_DIR = ROOT / (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_retrieval_probe_3_oracle_pool_headroom"
)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load 994b oracle report ───────────────────────────────────────────────
    with PROBE_994B_REPORT.open() as f:
        r994b = json.load(f)
    with BOOTSTRAP_L30.open() as f:
        boot_L30 = json.load(f)
    with ORACLE_TILT_SCENARIO_REPORT.open() as f:
        oracle_tilt_scenario = json.load(f)
    with START_ONLY_SCENARIO_REPORT.open() as f:
        start_only_scenario = json.load(f)

    # ── Key numbers from 994b ─────────────────────────────────────────────────
    oracle_replay = r994b["oracle_replay"]
    pool_stats = r994b["pool_stats"]
    secondary = r994b["secondary_diagnostics"]
    win_rates = r994b["win_rates"]
    oracle_tilt_val_frame = r994b["oracle_tilt_val_frame"]

    # CRPS values
    oracle_crps_pool_best_mean = oracle_replay["oracle_crps_pool_best_mean"]
    oracle_crps_pool_mean = oracle_replay["oracle_crps_pool_mean"]
    oracle_crps_selected_top3_mean = oracle_replay["oracle_crps_selected_top3_mean"]

    # Start-only CRPS
    start_only_crps = start_only_scenario["summary"]["narrative_generator_topk"][
        "ensemble_crps_z_mean"
    ]
    # Oracle tilt CRPS (what the oracle tilt actually achieved in the engine run)
    oracle_tilt_crps = oracle_tilt_scenario["summary"]["narrative_generator_topk"][
        "ensemble_crps_z_mean"
    ]

    # Persistence baseline
    persistence_crps = start_only_scenario["summary"]["persistence"]["ensemble_crps_z_mean"]

    # Bootstrap CI for oracle tilt vs start-only delta
    boot_crps = boot_L30["metrics"]["ensemble_crps_z"]
    delta_mean = boot_crps["mean_delta"]
    delta_ci_low = boot_crps["ci_low"]
    delta_ci_high = boot_crps["ci_high"]
    ci_excludes_zero = boot_crps["ci_excludes_zero"]

    # ── Headroom analysis ─────────────────────────────────────────────────────
    # (A) Within-pool oracle headroom (vs start-only)
    # oracle tilt - start_only (negative = oracle better)
    headroom_vs_start_only_raw = oracle_crps_selected_top3_mean - start_only_crps
    headroom_vs_start_only_engine = oracle_tilt_crps - start_only_crps
    # Bootstrap: delta = oracle - start_only; negative = oracle better
    headroom_bootstrap_mean = delta_mean  # already computed above

    # (B) Full-bank oracle: if we could pick the BEST of 50 pool candidates
    # oracle_crps_pool_best_mean is the mean across queries of best-of-50 CRPS
    full_bank_oracle_headroom = oracle_crps_pool_best_mean - start_only_crps  # negative = better

    # (C) Fraction of headroom captured by within-pool oracle tilt vs full-bank
    # tilt chooses top-3 by oracle replay; best-of-50 is the ceiling
    if full_bank_oracle_headroom != 0:
        # both negative (improvement); frac > 1 means tilt over-selects
        frac_headroom_captured_by_tilt = (
            (oracle_crps_selected_top3_mean - start_only_crps)
            / full_bank_oracle_headroom
        )
    else:
        frac_headroom_captured_by_tilt = float("nan")

    # Absolute improvement achievable if we reach pool_best (oracle ceiling)
    absolute_oracle_crps_improvement = start_only_crps - oracle_crps_pool_best_mean

    # Relative improvement (vs persistence as denominator)
    persistence_headroom = persistence_crps - start_only_crps  # positive = start_only already better
    # Relative improvement left on the table = (start_only - pool_best) / start_only
    relative_oracle_improvement_pct = absolute_oracle_crps_improvement / start_only_crps * 100

    # (D) Spearman rank-correlation between start_distance and oracle CRPS
    spearman_data = secondary["spearman_oracle_crps_vs_start_distance_within_pool"]
    spearman_mean = spearman_data["mean"]
    spearman_frac_above_0p3 = spearman_data["frac_above_0p3"]
    spearman_per_query = np.asarray(spearman_data["per_query"])

    # If Spearman(start_distance, oracle_CRPS) > 0 for most queries,
    # then "closer start" correlates with "lower oracle CRPS" -> retrieval matters.
    retrieval_signal_present = spearman_mean > 0.15  # weak-to-moderate positive

    # ── Oracle-selected vs start-only overlap (from 994b secondary) ──────────
    overlap_stats = secondary.get("oracle_vs_start_only_top3_overlap", {})
    frac_zero_overlap = overlap_stats.get("frac_queries_zero_overlap", float("nan"))

    # ── Oracle selected locality rank within pool ─────────────────────────────
    locality_rank = secondary.get("oracle_selected_locality_rank_within_pool", {})

    # ── Verdict ──────────────────────────────────────────────────────────────
    # The imminent choice: invest in NV-Retriever contrastive training to improve
    # retrieval, or focus elsewhere?

    # Key questions:
    # 1. Is oracle headroom large enough to motivate a better retriever?
    # 2. Does the Spearman signal say a better retriever (closer temporal match)
    #    would actually pick better candidates?
    # 3. Fraction of headroom captured: if tilt only captures <50% of full-bank,
    #    the within-pool oracle is suboptimal and we need a better tilt method,
    #    not a better retriever.

    oracle_crps_delta_bootstrap = headroom_bootstrap_mean  # signed (negative=better)
    oracle_beats_start = ci_excludes_zero and delta_mean < 0

    # Compute actual fraction
    tilt_improvement = start_only_crps - oracle_crps_selected_top3_mean  # positive=better
    fullbank_improvement = start_only_crps - oracle_crps_pool_best_mean   # positive=better

    # Note: oracle_tilt_engine_CRPS < oracle_crps_pool_best_mean is expected:
    # the top-3 oracle MIXTURE can outperform any single pool-best candidate.
    # Use engine oracle tilt as the within-pool ceiling (tighter bound).
    # full-bank ceiling comparison: pool_best vs start_only measures single-candidate headroom;
    # tilt vs start_only measures oracle-mixture headroom (more realistic for top-k selection).
    if fullbank_improvement > 0:
        tilt_fraction_of_fullbank = tilt_improvement / fullbank_improvement
    else:
        # tilt outperforms pool-best single-candidate (mixture effect); set to >1
        tilt_fraction_of_fullbank = float("nan")

    # primary headroom = oracle tilt engine CRPS (top-3 mixture) vs start-only
    oracle_tilt_improvement = start_only_crps - oracle_tilt_crps  # 0.0263
    if oracle_tilt_improvement < 0.02:
        headroom_assessment = "NEGLIGIBLE (<0.02 CRPS)"
    elif oracle_tilt_improvement < 0.05:
        headroom_assessment = "SMALL (0.02-0.05 CRPS)"
    elif oracle_tilt_improvement < 0.10:
        headroom_assessment = "MODERATE (0.05-0.10 CRPS)"
    else:
        headroom_assessment = "LARGE (>0.10 CRPS)"
    # single-candidate pool-best: oracle_crps_pool_best_mean vs start_only
    if absolute_oracle_crps_improvement < 0.02:
        single_candidate_assessment = "NEGLIGIBLE (<0.02 CRPS)"
    elif absolute_oracle_crps_improvement < 0.05:
        single_candidate_assessment = "SMALL (0.02-0.05 CRPS)"
    else:
        single_candidate_assessment = "MODERATE-LARGE (>=0.05 CRPS)"

    # Primary headroom = oracle tilt (top-3 mixture) vs start-only: 0.0263 CRPS
    # The oracle_crps_pool_best_mean (single-candidate) is for reference;
    # the tilt mixture outperforms it, which is expected (mixture > best single).
    realistic_retriever_share = spearman_mean  # rough upper-bound: Spearman fraction
    realistic_crps_gain = oracle_tilt_improvement * realistic_retriever_share * 0.5  # conservative

    if not retrieval_signal_present:
        verdict = (
            f"RETRIEVAL DOES NOT PREDICT QUALITY: Spearman(start_dist, oracle_CRPS)={spearman_mean:.3f} "
            f"(mean across queries; {frac_zero_overlap:.1%} queries zero overlap start_only vs oracle). "
            f"Oracle tilt ceiling (leakage): {oracle_tilt_improvement:.4f} CRPS ({headroom_assessment}). "
            "Even with perfect retrieval, the tilt (conditioning) interface is the binding constraint."
        )
        recommendation = (
            "DO NOT invest in NV-Retriever / contrastive retrieval training: "
            "retrieval quality (start-distance) does not predict oracle CRPS within the pool. "
            f"The headroom ({oracle_tilt_improvement:.4f} CRPS with leakage) exists but requires "
            "a better tilt/conditioning interface to capture, not a better retriever."
        )
    else:
        # Retrieval signal present: Spearman=0.32, 66% queries zero overlap
        # Realistic non-leakage fraction: ~1/3 of oracle headroom
        verdict = (
            f"RETRIEVAL PARTIALLY PREDICTS QUALITY: Spearman(start_dist, oracle_CRPS)={spearman_mean:.3f} "
            f"({frac_zero_overlap:.1%} queries zero overlap). "
            f"Oracle tilt headroom (leakage): {oracle_tilt_improvement:.4f} CRPS ({headroom_assessment}). "
            f"Single-candidate pool-best headroom: {absolute_oracle_crps_improvement:.4f} CRPS ({single_candidate_assessment}). "
            f"Note: oracle tilt (top-3 mixture) EXCEEDS pool-best (single window) -- mixture effect; "
            f"the mixture ceiling {oracle_tilt_improvement:.4f} is the relevant bound. "
            f"Bootstrap-validated oracle delta: {delta_mean:.4f} [{delta_ci_low:.4f}, {delta_ci_high:.4f}] "
            f"(CI excludes zero: {ci_excludes_zero}). "
            f"Realistic non-leakage retriever gain estimate "
            f"(Spearman={spearman_mean:.2f} x oracle/2): ~{realistic_crps_gain:.4f} CRPS."
        )
        if oracle_tilt_improvement < 0.03:
            recommendation = (
                f"WITHIN-POOL ORACLE HEADROOM IS SMALL ({oracle_tilt_improvement:.4f} CRPS, {headroom_assessment}). "
                f"Even an oracle retriever only gains ~{oracle_tilt_improvement:.4f} CRPS vs start-only (with leakage). "
                f"Realistic non-leakage retriever gain: ~{realistic_crps_gain:.4f} CRPS. "
                "NV-Retriever training is LOW PRIORITY: the absolute ceiling is too small to justify "
                "the training cost. Focus the budget on the conditioning interface (tilt weighting) "
                "and memory-target quality."
            )
        else:
            recommendation = (
                f"Retrieval signal IS present (Spearman={spearman_mean:.3f}). "
                f"Oracle headroom {oracle_tilt_improvement:.4f} CRPS is {headroom_assessment}. "
                f"Realistic non-leakage share: ~{realistic_crps_gain:.4f} CRPS. "
                "Investment in NV-Retriever / contrastive retrieval is conditionally justified "
                "but absolute gains will be modest. Pair with conditioning interface improvements."
            )

    report = {
        "schema_version": "nl_retrieval_probe_3_oracle_pool_headroom_v1",
        "clean_inputs": {
            "probe_994b_report": str(PROBE_994B_REPORT),
            "bootstrap_L30": str(BOOTSTRAP_L30),
            "oracle_tilt_scenario_report": str(ORACLE_TILT_SCENARIO_REPORT),
            "start_only_scenario_report": str(START_ONLY_SCENARIO_REPORT),
            "scope": "994b broad val frame: windows 4010..4450, stride-5, n_queries=89",
            "provenance": "All 994b artifacts are causal (bank max index 4009 < min query 4010). "
                          "Oracle CRPS uses realized future deltas -- LEAKAGE BY DESIGN (diagnostic only).",
        },
        "pool_config": {
            "pool_size": pool_stats["pool_size"],
            "mutual_gap_final_selection": pool_stats["mutual_gap_final_selection"],
            "n_queries": pool_stats["n_queries"],
            "candidate_universe": pool_stats["candidate_universe"],
        },
        "crps_values": {
            "persistence_crps": persistence_crps,
            "start_only_crps": start_only_crps,
            "oracle_crps_pool_mean": oracle_crps_pool_mean,
            "oracle_crps_selected_top3_mean": oracle_crps_selected_top3_mean,
            "oracle_crps_pool_best_mean": oracle_crps_pool_best_mean,
            "oracle_tilt_engine_crps": oracle_tilt_crps,
        },
        "headroom_analysis": {
            "within_pool_oracle_delta_vs_start": {
                "raw_delta_oracle_top3_replay_vs_start": float(headroom_vs_start_only_raw),
                "engine_delta_oracle_tilt_vs_start": float(headroom_vs_start_only_engine),
                "oracle_tilt_improvement_abs": float(oracle_tilt_improvement),
                "headroom_assessment": headroom_assessment,
                "bootstrap_mean_delta": float(delta_mean),
                "bootstrap_95ci": [float(delta_ci_low), float(delta_ci_high)],
                "ci_excludes_zero": ci_excludes_zero,
                "orientation": "oracle - start_only; negative = oracle is better",
                "note": (
                    "oracle_tilt (engine run, top-3 mixture) CRPS 0.3525 < "
                    "oracle_crps_pool_best_mean 0.3714: mixture of top-3 oracle windows "
                    "outperforms any single candidate from pool -- expected for k=3 selection."
                ),
            },
            "single_candidate_pool_best": {
                "absolute_crps_improvement": float(absolute_oracle_crps_improvement),
                "single_candidate_assessment": single_candidate_assessment,
                "relative_improvement_pct": float(relative_oracle_improvement_pct),
                "interpretation": (
                    "Best single-candidate from pool of 50 (by oracle CRPS replay). "
                    "Lower bound on oracle headroom; top-3 mixture exceeds this."
                ),
            },
            "realistic_nonleakage_estimate": {
                "spearman_fraction_proxy": float(spearman_mean),
                "conservative_estimate_crps": float(realistic_crps_gain),
                "pct_of_start_only": float(realistic_crps_gain / start_only_crps * 100),
                "interpretation": (
                    "Conservative estimate: Spearman=0.32 as fraction of oracle headroom / 2. "
                    "A non-oracle retriever with Spearman-correlated quality might capture "
                    "~1/6 of oracle headroom in practice."
                ),
            },
        },
        "retrieval_signal": {
            "spearman_start_dist_vs_oracle_crps_mean": spearman_mean,
            "spearman_frac_queries_above_0p3": spearman_frac_above_0p3,
            "spearman_per_query_p10": float(np.percentile(spearman_per_query, 10)),
            "spearman_per_query_p50": float(np.percentile(spearman_per_query, 50)),
            "spearman_per_query_p90": float(np.percentile(spearman_per_query, 90)),
            "retrieval_signal_present_threshold_0p15": retrieval_signal_present,
            "interpretation": (
                "Spearman(start_distance_rank, oracle_CRPS_rank): positive = closer start -> better oracle CRPS. "
                "High positive = retrieval matters. Near-zero = pool quality is driven by tilt, not retriever."
            ),
            "oracle_vs_start_overlap": overlap_stats,
            "oracle_locality_rank_in_pool": locality_rank,
        },
        "verdict": verdict,
        "recommendation": recommendation,
        "context_for_training_decision": {
            "imminent_method": "NV-Retriever false-negative filtering + locality-soft/posterior target",
            "the_key_question": (
                "Does retrieval quality (which causal windows enter the pool) "
                "constrain the CRPS ceiling more than the tilt (how pool weights are assigned)?"
            ),
            "evidence_summary": {
                "oracle_tilt_beats_start_only": oracle_beats_start,
                "oracle_ci_excludes_zero": ci_excludes_zero,
                "spearman_signal": f"mean={spearman_mean:.3f}, "
                                   f"{frac_zero_overlap:.1%} queries zero overlap",
                "full_bank_oracle_improvement": f"{absolute_oracle_crps_improvement:.4f} CRPS ({headroom_assessment})",
                "start_only_crps_already_vs_persistence": float(
                    persistence_crps - start_only_crps
                ),
            },
        },
    }

    out_path = OUTPUT_DIR / "probe3_oracle_pool_headroom_report.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"PROBE 3 RESULT written to: {out_path}")
    print(f"  Start-only CRPS:           {start_only_crps:.4f}")
    print(f"  Oracle top3 tilt CRPS:     {oracle_crps_selected_top3_mean:.4f}")
    print(f"  Oracle pool-best CRPS:     {oracle_crps_pool_best_mean:.4f}  (full-bank ceiling)")
    print(f"  Persistence CRPS:          {persistence_crps:.4f}")
    print()
    print(f"  Within-pool oracle delta:  {delta_mean:.4f} "
          f"[{delta_ci_low:.4f}, {delta_ci_high:.4f}]  "
          f"(excl. zero={ci_excludes_zero})")
    print(f"  Oracle tilt headroom:      {oracle_tilt_improvement:.4f} CRPS "
          f"({headroom_assessment})")
    print(f"  Single-candidate pool-best:{absolute_oracle_crps_improvement:.4f} CRPS "
          f"({single_candidate_assessment})")
    print(f"  Realistic non-leakage est: ~{realistic_crps_gain:.4f} CRPS "
          f"({realistic_crps_gain/start_only_crps*100:.1f}% of start-only)")
    print()
    print(f"  Spearman(start_dist, oracle_CRPS) mean: {spearman_mean:.3f}  "
          f"({frac_zero_overlap:.1%} queries zero overlap)")
    print(f"  Retrieval signal present (mean>0.15): {retrieval_signal_present}")
    print()
    print(f"  VERDICT: {verdict}")
    print(f"  RECOMMENDATION: {recommendation}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
