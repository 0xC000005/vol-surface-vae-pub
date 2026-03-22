#!/usr/bin/env python3
"""
CI Coverage vs Factor Structure (Rank) Correlation Analysis

Tests whether CI coverage and factor structure (effective rank) are correlated
or independent across all available models. This is the critical question:
Is factor collapse the CAUSE of CI failure, or are they independent problems?

Output:
  - results/validations/2026-03-22/analysis/ci_vs_rank/correlation_matrix.json
  - results/validations/2026-03-22/analysis/ci_vs_rank/report.md
  - results/validations/2026-03-22/verification_results/ci_vs_rank.json
"""

import json
import glob
import os
import sys
import numpy as np
from itertools import combinations

# ── Paths ──
BASE = "/home/max/Documents/vol-surface-vae-pub"
RESULTS_DIR = os.path.join(BASE, "results/block_ar")
OUT_DIR = os.path.join(BASE, "results/validations/2026-03-22/analysis/ci_vs_rank")
VERIF_DIR = os.path.join(BASE, "results/validations/2026-03-22/verification_results")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VERIF_DIR, exist_ok=True)

# ── Priority models (explicitly requested) ──
PRIORITY_MODELS = [
    "144a_final_30d", "144b_best_30d", "144c_best_30d",
    "145a_best_30d", "145c_best_30d",
    "143a_ep30_30d", "143a_ep1_30d",
    "99m_v2_30d", "139a_v2_30d",
]


def extract_metrics(summary_path):
    """Extract key metrics from a summary.json file. Returns dict or None."""
    try:
        with open(summary_path) as f:
            d = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None

    metrics = {}

    # 1. CI 90% coverage (overall)
    cov = d.get("coverage", {})
    overall = cov.get("overall", {})
    metrics["ci_90_overall"] = overall.get("0.9")

    # 2. Worst-cell per-horizon coverage at 90%
    per_horizon = cov.get("per_horizon", {})
    for h in ["1", "7", "14", "30"]:
        hdata = per_horizon.get(h, {})
        metrics[f"ci_90_h{h}"] = hdata.get("0.9")

    # Per-cell worst coverage at each horizon
    per_cell = cov.get("per_cell_coverage", {})
    for h in ["1", "7", "14", "30"]:
        cells = per_cell.get(h, [])
        if cells:
            flat = [v for row in cells for v in row if isinstance(v, (int, float))]
            if flat:
                metrics[f"worst_cell_h{h}"] = min(flat)

    # Coverage pass
    metrics["coverage_pass"] = cov.get("overall_pass", None)
    metrics["worst_cell_pass"] = cov.get("worst_cell_pass", None)

    # 3. Cross-cell correlation
    xcc = d.get("cross_cell_correlation", {})
    metrics["corr_ratio"] = xcc.get("corr_ratio")
    metrics["rank_ratio"] = xcc.get("rank_ratio")
    metrics["gt_eff_rank"] = xcc.get("gt_eff_rank")
    metrics["gen_eff_rank"] = xcc.get("gen_eff_rank")
    metrics["gen_mean_corr"] = xcc.get("gen_mean_corr")
    metrics["gt_mean_corr"] = xcc.get("gt_mean_corr")
    metrics["gen_pc1_var"] = xcc.get("gen_pc1_var")
    metrics["xcc_pass"] = xcc.get("overall_pass", None)

    # 4. KS daily pass count
    dist = d.get("distributional", {})
    ks_test = dist.get("ks_test", {})
    metrics["ks_daily_npass"] = ks_test.get("n_pass")
    metrics["ks_daily_median"] = ks_test.get("median_stat")

    # KS level test
    ks_level = dist.get("ks_level_test", {})
    metrics["ks_level_npass"] = ks_level.get("n_pass")

    # Window floor
    wf = dist.get("window_floor", {})
    metrics["window_floor_pct"] = wf.get("pct_bad")
    metrics["window_floor_p10"] = wf.get("p10_cov")

    # Distributional pass
    metrics["dist_pass"] = dist.get("overall_pass", None)

    # 5. Time series
    ts = d.get("time_series", {})
    kurtosis = ts.get("kurtosis", {})
    metrics["kurtosis_ratio"] = kurtosis.get("kurtosis_ratio")
    acf = ts.get("acf", {})
    metrics["acf_corr"] = acf.get("acf_correlation")
    metrics["ts_pass"] = ts.get("overall_pass", None)

    # 6. Cointegration
    co = d.get("cointegration", {})
    metrics["coint_ratio"] = co.get("gen_gt_ratio")
    metrics["coint_ratio_legacy"] = co.get("gen_gt_ratio_legacy")
    metrics["coint_pass"] = co.get("overall_pass", None)

    # 7. Conditionality
    cond = d.get("conditionality", {})
    metrics["cond_pass"] = cond.get("overall_pass", None)

    # 8. Surface validity
    surf = d.get("surface", {})
    metrics["surface_pass"] = surf.get("overall_pass", None)

    # 9. Block-AR
    bar = d.get("block_ar", {})
    metrics["block_ar_pass"] = bar.get("overall_pass", None)

    # 10. Regime coverage
    reg = d.get("regime_coverage", {})
    metrics["regime_pass"] = reg.get("overall_pass", None)

    # Count suite passes
    suite_keys = [
        ("surface", "surface_pass"),
        ("coverage", "coverage_pass"),
        ("conditionality", "cond_pass"),
        ("time_series", "ts_pass"),
        ("block_ar", "block_ar_pass"),
        ("cointegration", "coint_pass"),
        ("regime_coverage", "regime_pass"),
        ("distributional", "dist_pass"),
        ("cross_cell_correlation", "xcc_pass"),
    ]
    n_pass = sum(1 for _, mk in suite_keys if metrics.get(mk) is True)
    metrics["n_suites_pass"] = n_pass

    # Filter out None values for key metrics
    if metrics.get("ci_90_overall") is None or metrics.get("rank_ratio") is None:
        return None  # Must have at least CI and rank to be useful

    return metrics


def pearson_corr(x, y):
    """Pearson correlation with p-value (two-tailed t-test)."""
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    x = np.array(x)
    y = np.array(y)
    mx, my = x.mean(), y.mean()
    dx, dy = x - mx, y - my
    denom = np.sqrt(np.sum(dx**2) * np.sum(dy**2))
    if denom == 0:
        return float("nan"), float("nan")
    r = np.sum(dx * dy) / denom
    # t-test
    if abs(r) >= 1.0:
        return float(r), 0.0
    t_stat = r * np.sqrt((n - 2) / (1 - r**2))
    # Two-tailed p-value from t-distribution (scipy-free approximation)
    from math import gamma, pi
    df = n - 2
    # Use numerical integration of t-distribution
    p = _t_pvalue(abs(t_stat), df) * 2
    return float(r), float(p)


def spearman_corr(x, y):
    """Spearman rank correlation."""
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")

    def rankdata(vals):
        sorted_idx = np.argsort(vals)
        ranks = np.empty_like(sorted_idx, dtype=float)
        ranks[sorted_idx] = np.arange(1, len(vals) + 1, dtype=float)
        return ranks

    rx = rankdata(np.array(x))
    ry = rankdata(np.array(y))
    return pearson_corr(rx, ry)


def _t_pvalue(t, df):
    """Approximate one-tailed p-value for t-distribution using beta regularized function."""
    # Use the relationship: p = I(df/(df+t^2), df/2, 1/2) / 2
    # We'll use a simple numerical approach
    x_val = df / (df + t**2)
    a, b = df / 2.0, 0.5
    # Incomplete beta via continued fraction or series
    # For simplicity, use a numerical integration approach
    from math import log, exp

    # Numerical integration of t-distribution PDF
    steps = 10000
    dt = t / steps
    # Integrate from t to a large value
    total = 0.0
    from math import gamma, pi, sqrt

    # Beta function
    log_coeff = (
        log(gamma(df / 2.0 + 0.5))
        - log(gamma(df / 2.0))
        - 0.5 * log(df * pi)
    )

    # Integrate from |t| to infinity using substitution
    upper = max(abs(t) * 10, 100)
    n_steps = 50000
    h = (upper - abs(t)) / n_steps
    total = 0.0
    for i in range(n_steps):
        xi = abs(t) + (i + 0.5) * h
        log_pdf = log_coeff - (df + 1) / 2.0 * log(1 + xi**2 / df)
        total += exp(log_pdf) * h

    return max(total, 1e-15)


def main():
    # ── Collect all models ──
    all_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*/summary.json")))
    print(f"Found {len(all_files)} summary.json files total")

    # Extract metrics from all files
    all_models = {}
    for fpath in all_files:
        model_name = os.path.basename(os.path.dirname(fpath))
        m = extract_metrics(fpath)
        if m is not None:
            all_models[model_name] = m

    print(f"Successfully extracted metrics from {len(all_models)} models (with both CI and rank data)")

    # Separate priority vs all
    priority_data = {k: v for k, v in all_models.items() if k in PRIORITY_MODELS}
    print(f"Priority models found: {len(priority_data)} / {len(PRIORITY_MODELS)}")
    for pm in PRIORITY_MODELS:
        status = "FOUND" if pm in all_models else "MISSING"
        print(f"  {pm}: {status}")

    # ── Define metric pairs for correlation analysis ──
    metric_keys = [
        ("ci_90_overall", "CI 90% Coverage (Overall)"),
        ("rank_ratio", "Rank Ratio (gen/GT)"),
        ("gen_eff_rank", "Gen Effective Rank"),
        ("corr_ratio", "Correlation Ratio"),
        ("gen_mean_corr", "Gen Mean Correlation"),
        ("gen_pc1_var", "Gen PC1 Variance Explained"),
        ("ks_daily_npass", "KS Daily Pass Count"),
        ("ks_level_npass", "KS Level Pass Count"),
        ("window_floor_pct", "Window Floor % Bad"),
        ("window_floor_p10", "Window Floor P10 Cov"),
        ("kurtosis_ratio", "Kurtosis Ratio"),
        ("acf_corr", "ACF Correlation"),
        ("coint_ratio_legacy", "Cointegration Ratio"),
        ("n_suites_pass", "Suite Pass Count"),
    ]

    # ── Build data arrays ──
    model_names = sorted(all_models.keys())

    def get_valid_pairs(key1, key2):
        """Get pairs of values where both are non-None."""
        xs, ys, names = [], [], []
        for mn in model_names:
            v1 = all_models[mn].get(key1)
            v2 = all_models[mn].get(key2)
            if v1 is not None and v2 is not None and not (isinstance(v1, bool) or isinstance(v2, bool)):
                xs.append(float(v1))
                ys.append(float(v2))
                names.append(mn)
        return xs, ys, names

    # ── Compute full correlation matrix ──
    n_metrics = len(metric_keys)
    pearson_matrix = np.full((n_metrics, n_metrics), np.nan)
    spearman_matrix = np.full((n_metrics, n_metrics), np.nan)
    pval_pearson_matrix = np.full((n_metrics, n_metrics), np.nan)
    pval_spearman_matrix = np.full((n_metrics, n_metrics), np.nan)
    n_samples_matrix = np.full((n_metrics, n_metrics), 0, dtype=int)

    for i in range(n_metrics):
        for j in range(n_metrics):
            if i == j:
                pearson_matrix[i, j] = 1.0
                spearman_matrix[i, j] = 1.0
                pval_pearson_matrix[i, j] = 0.0
                pval_spearman_matrix[i, j] = 0.0
                continue
            k1 = metric_keys[i][0]
            k2 = metric_keys[j][0]
            xs, ys, _ = get_valid_pairs(k1, k2)
            n_samples_matrix[i, j] = len(xs)
            if len(xs) >= 3:
                r_p, p_p = pearson_corr(xs, ys)
                r_s, p_s = spearman_corr(xs, ys)
                pearson_matrix[i, j] = r_p
                spearman_matrix[i, j] = r_s
                pval_pearson_matrix[i, j] = p_p
                pval_spearman_matrix[i, j] = p_s

    # ── Key correlations: CI vs Rank ──
    print("\n" + "=" * 70)
    print("KEY CORRELATIONS: CI Coverage vs Factor Structure")
    print("=" * 70)

    key_pairs = [
        ("ci_90_overall", "rank_ratio"),
        ("ci_90_overall", "gen_eff_rank"),
        ("ci_90_overall", "corr_ratio"),
        ("ci_90_overall", "ks_daily_npass"),
        ("ci_90_overall", "kurtosis_ratio"),
        ("ci_90_overall", "n_suites_pass"),
        ("rank_ratio", "ks_daily_npass"),
        ("rank_ratio", "kurtosis_ratio"),
        ("rank_ratio", "n_suites_pass"),
        ("gen_eff_rank", "kurtosis_ratio"),
        ("gen_eff_rank", "ks_daily_npass"),
        ("ks_daily_npass", "kurtosis_ratio"),
    ]

    key_results = []
    for k1, k2 in key_pairs:
        xs, ys, names = get_valid_pairs(k1, k2)
        n = len(xs)
        label1 = dict(metric_keys).get(k1, k1)
        label2 = dict(metric_keys).get(k2, k2)
        if n >= 3:
            r_p, p_p = pearson_corr(xs, ys)
            r_s, p_s = spearman_corr(xs, ys)
            sig_p = "***" if p_p < 0.001 else "**" if p_p < 0.01 else "*" if p_p < 0.05 else "" if p_p < 0.1 else "ns"
            sig_s = "***" if p_s < 0.001 else "**" if p_s < 0.01 else "*" if p_s < 0.05 else "" if p_s < 0.1 else "ns"
            print(f"\n  {label1} vs {label2} (n={n}):")
            print(f"    Pearson:  r={r_p:+.4f}  p={p_p:.4f} {sig_p}")
            print(f"    Spearman: r={r_s:+.4f}  p={p_s:.4f} {sig_s}")
            key_results.append({
                "metric1": k1, "metric2": k2,
                "label1": label1, "label2": label2,
                "n": n,
                "pearson_r": round(r_p, 4), "pearson_p": round(p_p, 4),
                "spearman_r": round(r_s, 4), "spearman_p": round(p_s, 4),
            })
        else:
            print(f"\n  {label1} vs {label2}: INSUFFICIENT DATA (n={n})")
            key_results.append({
                "metric1": k1, "metric2": k2,
                "label1": label1, "label2": label2,
                "n": n, "error": "insufficient data"
            })

    # ── Priority model detail table ──
    print("\n" + "=" * 70)
    print("PRIORITY MODEL DETAILS")
    print("=" * 70)
    priority_table = []
    for mn in PRIORITY_MODELS:
        if mn not in all_models:
            continue
        m = all_models[mn]
        row = {
            "model": mn,
            "ci_90": m.get("ci_90_overall"),
            "rank_ratio": m.get("rank_ratio"),
            "gen_eff_rank": m.get("gen_eff_rank"),
            "corr_ratio": m.get("corr_ratio"),
            "ks_daily": m.get("ks_daily_npass"),
            "kurtosis_ratio": m.get("kurtosis_ratio"),
            "window_floor_pct": m.get("window_floor_pct"),
            "n_pass": m.get("n_suites_pass"),
            "cov_pass": m.get("coverage_pass"),
            "xcc_pass": m.get("xcc_pass"),
        }
        priority_table.append(row)
        print(f"\n  {mn}:")
        print(f"    CI 90%={row['ci_90']:.3f}  rank_ratio={row['rank_ratio']:.3f}  "
              f"eff_rank={row['gen_eff_rank']:.2f}  corr_ratio={row['corr_ratio']:.3f}")
        print(f"    KS daily={row['ks_daily']}/25  kurtosis={row['kurtosis_ratio']:.3f}  "
              f"wf_pct={row['window_floor_pct']:.3f}  suites={row['n_pass']}/9")

    # ── All models sorted by CI coverage ──
    print("\n" + "=" * 70)
    print("ALL MODELS SORTED BY CI 90% COVERAGE (top 20)")
    print("=" * 70)
    sorted_models = sorted(all_models.items(), key=lambda x: x[1].get("ci_90_overall", 0), reverse=True)
    for mn, m in sorted_models[:20]:
        ci = m.get("ci_90_overall", 0)
        rr = m.get("rank_ratio", 0)
        er = m.get("gen_eff_rank", 0)
        ks = m.get("ks_daily_npass", 0)
        np_ = m.get("n_suites_pass", 0)
        pri = " [PRIORITY]" if mn in PRIORITY_MODELS else ""
        print(f"  {mn:45s}  CI={ci:.3f}  rank_ratio={rr:.3f}  eff_rank={er:.2f}  KS={ks}/25  suites={np_}/9{pri}")

    # ── Independence test conclusion ──
    print("\n" + "=" * 70)
    print("CONCLUSION: CI Coverage vs Rank Ratio Independence Test")
    print("=" * 70)

    ci_rank = next((r for r in key_results if r["metric1"] == "ci_90_overall" and r["metric2"] == "rank_ratio"), None)
    ci_effrank = next((r for r in key_results if r["metric1"] == "ci_90_overall" and r["metric2"] == "gen_eff_rank"), None)

    if ci_rank and "error" not in ci_rank:
        p_val = ci_rank["spearman_p"]
        r_val = ci_rank["spearman_r"]
        if p_val < 0.05:
            conclusion = f"CORRELATED (Spearman r={r_val:.3f}, p={p_val:.4f}). Fixing rank MAY help CI."
            independent = False
        elif p_val > 0.1:
            conclusion = f"INDEPENDENT (Spearman r={r_val:.3f}, p={p_val:.4f}). CI and rank are SEPARATE problems."
            independent = True
        else:
            conclusion = f"BORDERLINE (Spearman r={r_val:.3f}, p={p_val:.4f}). Weak evidence of dependence."
            independent = None
    else:
        conclusion = "INSUFFICIENT DATA to determine relationship."
        independent = None

    print(f"\n  {conclusion}")

    # ── Save outputs ──
    # 1. Correlation matrix
    corr_matrix_data = {
        "metric_labels": [mk[1] for mk in metric_keys],
        "metric_keys": [mk[0] for mk in metric_keys],
        "pearson_matrix": pearson_matrix.tolist(),
        "spearman_matrix": spearman_matrix.tolist(),
        "pval_pearson_matrix": pval_pearson_matrix.tolist(),
        "pval_spearman_matrix": pval_spearman_matrix.tolist(),
        "n_samples_matrix": n_samples_matrix.tolist(),
    }
    with open(os.path.join(OUT_DIR, "correlation_matrix.json"), "w") as f:
        json.dump(corr_matrix_data, f, indent=2)

    # 2. Key results
    with open(os.path.join(OUT_DIR, "key_correlations.json"), "w") as f:
        json.dump(key_results, f, indent=2)

    # 3. Priority model table
    with open(os.path.join(OUT_DIR, "priority_models.json"), "w") as f:
        json.dump(priority_table, f, indent=2)

    # 4. All model metrics
    all_model_export = []
    for mn, m in sorted_models:
        row = {"model": mn}
        for mk, _ in metric_keys:
            row[mk] = m.get(mk)
        # Add pass/fail booleans
        for bk in ["coverage_pass", "xcc_pass", "dist_pass", "ts_pass", "coint_pass",
                    "cond_pass", "surface_pass", "block_ar_pass", "regime_pass"]:
            row[bk] = m.get(bk)
        all_model_export.append(row)
    with open(os.path.join(OUT_DIR, "all_models_metrics.json"), "w") as f:
        json.dump(all_model_export, f, indent=2)

    # ── Generate markdown report ──
    report = generate_report(
        all_models, priority_table, key_results,
        pearson_matrix, spearman_matrix, pval_spearman_matrix,
        metric_keys, n_samples_matrix, conclusion, independent,
        sorted_models
    )
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write(report)

    # ── Verification result ──
    verification = {
        "test": "ci_vs_rank_correlation",
        "date": "2026-03-22",
        "n_models_analyzed": len(all_models),
        "n_priority_models": len(priority_data),
        "primary_result": {
            "ci_vs_rank_ratio": ci_rank if ci_rank else {"error": "not computed"},
            "ci_vs_gen_eff_rank": ci_effrank if ci_effrank else {"error": "not computed"},
            "conclusion": conclusion,
            "independent": independent,
        },
        "all_key_correlations": key_results,
        "priority_model_summary": priority_table,
        "files_produced": [
            os.path.join(OUT_DIR, "correlation_matrix.json"),
            os.path.join(OUT_DIR, "key_correlations.json"),
            os.path.join(OUT_DIR, "priority_models.json"),
            os.path.join(OUT_DIR, "all_models_metrics.json"),
            os.path.join(OUT_DIR, "report.md"),
        ],
    }
    with open(os.path.join(VERIF_DIR, "ci_vs_rank.json"), "w") as f:
        json.dump(verification, f, indent=2)

    print(f"\nFiles saved to {OUT_DIR}")
    print(f"Verification result saved to {os.path.join(VERIF_DIR, 'ci_vs_rank.json')}")


def generate_report(all_models, priority_table, key_results,
                    pearson_matrix, spearman_matrix, pval_spearman_matrix,
                    metric_keys, n_samples_matrix, conclusion, independent,
                    sorted_models):
    """Generate markdown report."""
    lines = []
    lines.append("# CI Coverage vs Factor Structure (Rank) Correlation Analysis")
    lines.append("")
    lines.append(f"**Date**: 2026-03-22")
    lines.append(f"**Models analyzed**: {len(all_models)}")
    lines.append("")

    # Executive summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"**{conclusion}**")
    lines.append("")

    ci_rank = next((r for r in key_results if r["metric1"] == "ci_90_overall" and r["metric2"] == "rank_ratio"), None)
    ci_effrank = next((r for r in key_results if r["metric1"] == "ci_90_overall" and r["metric2"] == "gen_eff_rank"), None)
    ci_ks = next((r for r in key_results if r["metric1"] == "ci_90_overall" and r["metric2"] == "ks_daily_npass"), None)
    rank_ks = next((r for r in key_results if r["metric1"] == "rank_ratio" and r["metric2"] == "ks_daily_npass"), None)

    if ci_rank and "error" not in ci_rank:
        lines.append(f"- CI 90% vs Rank Ratio: Spearman r={ci_rank['spearman_r']:+.3f}, p={ci_rank['spearman_p']:.4f} (n={ci_rank['n']})")
    if ci_effrank and "error" not in ci_effrank:
        lines.append(f"- CI 90% vs Gen Eff Rank: Spearman r={ci_effrank['spearman_r']:+.3f}, p={ci_effrank['spearman_p']:.4f} (n={ci_effrank['n']})")
    if ci_ks and "error" not in ci_ks:
        lines.append(f"- CI 90% vs KS Daily: Spearman r={ci_ks['spearman_r']:+.3f}, p={ci_ks['spearman_p']:.4f} (n={ci_ks['n']})")
    if rank_ks and "error" not in rank_ks:
        lines.append(f"- Rank Ratio vs KS Daily: Spearman r={rank_ks['spearman_r']:+.3f}, p={rank_ks['spearman_p']:.4f} (n={rank_ks['n']})")
    lines.append("")

    # Priority model table
    lines.append("## Priority Model Details")
    lines.append("")
    lines.append("| Model | CI 90% | Rank Ratio | Eff Rank | Corr Ratio | KS Daily | Kurtosis | WF % | Suites |")
    lines.append("|-------|--------|------------|----------|------------|----------|----------|------|--------|")
    for row in priority_table:
        ci = f"{row['ci_90']:.3f}" if row['ci_90'] else "N/A"
        rr = f"{row['rank_ratio']:.3f}" if row['rank_ratio'] else "N/A"
        er = f"{row['gen_eff_rank']:.2f}" if row['gen_eff_rank'] else "N/A"
        cr = f"{row['corr_ratio']:.3f}" if row['corr_ratio'] else "N/A"
        ks = f"{row['ks_daily']}/25" if row['ks_daily'] is not None else "N/A"
        ku = f"{row['kurtosis_ratio']:.3f}" if row['kurtosis_ratio'] else "N/A"
        wf = f"{row['window_floor_pct']:.3f}" if row['window_floor_pct'] is not None else "N/A"
        np_ = f"{row['n_pass']}/9" if row['n_pass'] is not None else "N/A"
        lines.append(f"| {row['model']} | {ci} | {rr} | {er} | {cr} | {ks} | {ku} | {wf} | {np_} |")
    lines.append("")

    # Key correlations table
    lines.append("## Key Pairwise Correlations")
    lines.append("")
    lines.append("| Metric 1 | Metric 2 | n | Pearson r | Pearson p | Spearman r | Spearman p | Sig |")
    lines.append("|----------|----------|---|-----------|-----------|------------|------------|-----|")
    for r in key_results:
        if "error" in r:
            lines.append(f"| {r['label1']} | {r['label2']} | {r['n']} | - | - | - | - | N/A |")
        else:
            sig = "***" if r['spearman_p'] < 0.001 else "**" if r['spearman_p'] < 0.01 else "*" if r['spearman_p'] < 0.05 else "~" if r['spearman_p'] < 0.1 else "ns"
            lines.append(f"| {r['label1']} | {r['label2']} | {r['n']} | {r['pearson_r']:+.3f} | {r['pearson_p']:.4f} | {r['spearman_r']:+.3f} | {r['spearman_p']:.4f} | {sig} |")
    lines.append("")

    # Spearman correlation matrix (condensed)
    lines.append("## Spearman Correlation Matrix (key metrics)")
    lines.append("")
    # Only show subset of metrics
    show_idx = [0, 1, 2, 3, 6, 10, 13]  # CI, rank_ratio, eff_rank, corr_ratio, KS daily, kurtosis, suites
    show_labels = [metric_keys[i][1][:20] for i in show_idx]
    header = "| " + " | ".join([""] + show_labels) + " |"
    sep = "| " + " | ".join(["---"] * (len(show_labels) + 1)) + " |"
    lines.append(header)
    lines.append(sep)
    for i, si in enumerate(show_idx):
        row_vals = []
        for j, sj in enumerate(show_idx):
            v = spearman_matrix[si, sj]
            if np.isnan(v):
                row_vals.append("  -  ")
            else:
                p = pval_spearman_matrix[si, sj]
                star = "*" if (not np.isnan(p) and p < 0.05) else " "
                row_vals.append(f"{v:+.2f}{star}")
        lines.append(f"| {show_labels[i]:20s} | " + " | ".join(row_vals) + " |")
    lines.append("")

    # Top 20 models by CI
    lines.append("## Top 20 Models by CI 90% Coverage")
    lines.append("")
    lines.append("| Model | CI 90% | Rank Ratio | Eff Rank | KS Daily | Suites |")
    lines.append("|-------|--------|------------|----------|----------|--------|")
    for mn, m in sorted_models[:20]:
        ci = m.get("ci_90_overall", 0)
        rr = m.get("rank_ratio", 0)
        er = m.get("gen_eff_rank", 0)
        ks = m.get("ks_daily_npass", 0)
        np_ = m.get("n_suites_pass", 0)
        pri = " **" if mn in set(r["model"] for r in priority_table) else ""
        lines.append(f"| {mn}{pri} | {ci:.3f} | {rr:.3f} | {er:.2f} | {ks}/25 | {np_}/9 |")
    lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")
    if independent is True:
        lines.append("CI coverage and factor structure (rank) are **statistically independent**.")
        lines.append("This means:")
        lines.append("- Factor collapse does NOT cause CI failure")
        lines.append("- They are separate problems requiring separate solutions")
        lines.append("- Improving rank (e.g., via CLN, transformer decoder) will NOT automatically improve CI")
        lines.append("- CI improvement needs its own mechanism (e.g., scaling, loss tuning)")
    elif independent is False:
        lines.append("CI coverage and factor structure (rank) are **statistically correlated**.")
        lines.append("This means:")
        lines.append("- Factor collapse and CI failure share an underlying cause")
        lines.append("- Fixing rank may help CI coverage")
        lines.append("- The CRPS rank-1 attractor may be simultaneously causing both problems")
    else:
        lines.append("The evidence is **inconclusive** regarding CI-rank dependence.")
        lines.append("More models or more diverse architectures may be needed to determine the relationship.")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
