"""H0 diagnostic: is the universal regime inversion (calm_wr > turb_wr) that we see in
every trained model actually a property of the ground-truth data under our regime proxy?

Method:
  1. Load raw surfaces (data/vol_surface_with_ret.npz, [0, 1] IV space).
  2. Slice the test split (start_idx=4511, the same default used by
     evaluate_220b_multihorizon_path_suite.py).
  3. Build the same (history=30, future=30) windows VolSurfaceDataset builds.
  4. For each window, compute the SAME vov proxy used inside
     test_block_ar_requirements_v2.py::run_conditionality_tests — namely the mean of
     squared first-differences of IV values over the history window:
         batch_vov = (diff(history, axis=time) ** 2).mean(axis=(time, row, col))
  5. Bucket into calm (vov ≤ q20) and turb (vov ≥ q80) — the suite's exact thresholds.
  6. Compute three GT regime-differential statistics and their turb/calm ratios:
       (A) realised-variance-of-future (RVF): per-window, mean of squared first-differences
           of the future path; then averaged within regime. This is the data-side analogue
           of the vov signal — if this ratio is < 1 the proxy is not predictive of future
           realised variance at all and the conditionality gate is ill-posed.
       (B) within-path std: per-window, std across the 30-day future, per cell, averaged
           over cells; then averaged within regime.
       (C) marginal CI width: for each (t, cell) compute the 90%-quantile range of future
           values POOLED across regime windows, then mean over (t, cell). This is the
           closest GT analogue of the suite's model-side 90% CI width. Gate value 1.25-1.52
           (test-suite docstring line 713) is the advertised GT range and we cross-check it.

Also compute the legacy IV-mean vov proxy (hist.mean(axis=(row,col)).std(axis=time)) for
comparison, so we can see whether the proxy swap introduced the inversion.

Kill decision:
  * If ALL of {A, B, C} ratios are > 1.15 under the NEW proxy, the regime signal is
    genuine and the universal model-side inversion is ARCHITECTURAL. Proceed to joint-path
    CFM (H1).
  * If any of them is < 1.0, the proxy is mislabeling and conditionality gates under
    the new proxy are meaningless — rebuild the proxy before any further training.

Writes a structured JSON + a markdown one-pager to
  results/block_ar/H0_regime_inversion_gt/.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def compute_vov_rv(history: np.ndarray) -> np.ndarray:
    """Full-surface realised variance of first differences (new proxy)."""
    dh = np.diff(history, axis=1)
    return (dh ** 2).mean(axis=(1, 2, 3))


def compute_vov_iv(history: np.ndarray) -> np.ndarray:
    """Legacy IV-mean-of-surface vov proxy (std of mean-IV time series)."""
    mean_iv = history.mean(axis=(2, 3))  # (B, T)
    return mean_iv.std(axis=1)


def regime_masks(vov: np.ndarray, q_lo: float = 0.20, q_hi: float = 0.80):
    lo = np.quantile(vov, q_lo)
    hi = np.quantile(vov, q_hi)
    calm = vov <= lo
    turb = vov >= hi
    return calm, turb, lo, hi


def ratio(turb_vals: np.ndarray, calm_vals: np.ndarray) -> float:
    if calm_vals.mean() == 0:
        return float("nan")
    return float(turb_vals.mean() / calm_vals.mean())


def analyse(vov: np.ndarray, histories: np.ndarray, futures: np.ndarray, label: str):
    """Returns a dict of stats keyed by (A), (B), (C) for a given vov proxy."""
    calm, turb, lo, hi = regime_masks(vov)
    n_calm = int(calm.sum())
    n_turb = int(turb.sum())

    # (A) Realised-variance of future: per-window mean(squared first-diffs of the future)
    df = np.diff(futures, axis=1)  # (N, T-1, 5, 5)
    rvf = (df ** 2).mean(axis=(1, 2, 3))  # (N,)
    a_calm = rvf[calm].mean()
    a_turb = rvf[turb].mean()
    a_ratio = float(a_turb / a_calm) if a_calm > 0 else float("nan")

    # (B) Within-path std, per cell, averaged
    std_per_window = futures.std(axis=1).mean(axis=(1, 2))  # (N,)
    b_calm = std_per_window[calm].mean()
    b_turb = std_per_window[turb].mean()
    b_ratio = float(b_turb / b_calm) if b_calm > 0 else float("nan")

    # (C) Marginal CI width: pool future values within a regime, compute (q95 - q05) per
    # (t, row, col), then mean.
    def marginal_ci_width(mask):
        vals = futures[mask]  # (n, T, 5, 5)
        q05 = np.quantile(vals, 0.05, axis=0)
        q95 = np.quantile(vals, 0.95, axis=0)
        return float((q95 - q05).mean())

    c_calm = marginal_ci_width(calm)
    c_turb = marginal_ci_width(turb)
    c_ratio = float(c_turb / c_calm) if c_calm > 0 else float("nan")

    # Per-horizon marginal CI width (diagnostic, check whether inversion is horizon-specific)
    def per_horizon_ci(mask):
        vals = futures[mask]
        q05 = np.quantile(vals, 0.05, axis=0)
        q95 = np.quantile(vals, 0.95, axis=0)
        return (q95 - q05).mean(axis=(1, 2))  # (T,)

    ci_calm = per_horizon_ci(calm)
    ci_turb = per_horizon_ci(turb)

    return {
        "proxy": label,
        "q20_threshold": float(lo),
        "q80_threshold": float(hi),
        "n_windows": int(len(vov)),
        "n_calm": n_calm,
        "n_turb": n_turb,
        "A_rvf_calm": float(a_calm),
        "A_rvf_turb": float(a_turb),
        "A_rvf_ratio": a_ratio,
        "B_within_std_calm": float(b_calm),
        "B_within_std_turb": float(b_turb),
        "B_within_std_ratio": b_ratio,
        "C_marginal_ci_calm": c_calm,
        "C_marginal_ci_turb": c_turb,
        "C_marginal_ci_ratio": c_ratio,
        "per_horizon_ci_calm": ci_calm.tolist(),
        "per_horizon_ci_turb": ci_turb.tolist(),
    }


def build_windows(surfaces: np.ndarray, start_idx: int, history_len: int, future_len: int):
    slab = surfaces[start_idx:]
    total_len = history_len + future_len
    n_windows = len(slab) - total_len + 1
    histories = np.zeros((n_windows, history_len, 5, 5), dtype=np.float32)
    futures = np.zeros((n_windows, future_len, 5, 5), dtype=np.float32)
    for i in range(n_windows):
        seq = slab[i : i + total_len]
        histories[i] = seq[:history_len]
        futures[i] = seq[history_len:]
    return histories, futures


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    p.add_argument("--test_start", type=int, default=4511)
    p.add_argument("--history_len", type=int, default=30)
    p.add_argument("--future_len", type=int, default=30)
    p.add_argument("--output_dir", default="results/block_ar/H0_regime_inversion_gt")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = np.load(args.data_path)
    surfaces = data["surface"].astype(np.float32)
    print(f"Loaded surfaces: shape={surfaces.shape}, range=[{surfaces.min():.4f}, {surfaces.max():.4f}]")

    histories, futures = build_windows(
        surfaces, args.test_start, args.history_len, args.future_len
    )
    print(f"Test-split windows: {len(histories)}")

    vov_rv = compute_vov_rv(histories)
    vov_iv = compute_vov_iv(histories)
    print(f"vov_rv: min={vov_rv.min():.2e} mean={vov_rv.mean():.2e} max={vov_rv.max():.2e}")
    print(f"vov_iv: min={vov_iv.min():.2e} mean={vov_iv.mean():.2e} max={vov_iv.max():.2e}")

    stats_rv = analyse(vov_rv, histories, futures, label="new_RV_based")
    stats_iv = analyse(vov_iv, histories, futures, label="legacy_IV_mean_vov")

    def verdict(s):
        pass_a = s["A_rvf_ratio"] > 1.15
        pass_b = s["B_within_std_ratio"] > 1.15
        pass_c = s["C_marginal_ci_ratio"] > 1.15
        fail_any = (
            s["A_rvf_ratio"] < 1.0
            or s["B_within_std_ratio"] < 1.0
            or s["C_marginal_ci_ratio"] < 1.0
        )
        if pass_a and pass_b and pass_c:
            return "PROXY_OK — regime signal genuine; model inversion is ARCHITECTURAL"
        if fail_any:
            return "PROXY_BROKEN — at least one ratio < 1, proxy mislabels regimes"
        return "PROXY_WEAK — signal present but below 1.15 gate; confound for test-suite"

    stats_rv["verdict"] = verdict(stats_rv)
    stats_iv["verdict"] = verdict(stats_iv)

    report = {
        "H0": "regime_inversion_ground_truth",
        "description": (
            "Measures whether GT futures under the test suite's regime proxy actually "
            "show turb > calm dispersion. Kill-condition isolator for joint-path CFM."
        ),
        "test_split_start": args.test_start,
        "n_windows": int(len(histories)),
        "stats_new_proxy": stats_rv,
        "stats_legacy_proxy": stats_iv,
    }

    json_path = out / "H0_gt_regime_stats.json"
    with json_path.open("w") as f:
        json.dump(report, f, indent=2)

    # Human-readable markdown
    md_path = out / "H0_gt_regime_stats.md"
    with md_path.open("w") as f:
        f.write("# H0 — Regime inversion in GT (diagnostic)\n\n")
        f.write(
            "Tests whether the GT test-split futures, bucketed by the test suite's\n"
            "vov regime proxy, actually show turbulent > calm dispersion. If they do, the\n"
            "universal model-side inversion is architectural. If they do not, the proxy\n"
            "is mislabeling and further model work is wasted until the proxy is fixed.\n\n"
        )
        f.write(f"Test-split windows: **{len(histories)}** (from index {args.test_start})\n\n")
        for label, s in [("New RV-based proxy (CURRENT SUITE)", stats_rv), ("Legacy IV-mean proxy (REFERENCE)", stats_iv)]:
            f.write(f"## {label}\n\n")
            f.write(f"- Calm (vov ≤ q20): {s['n_calm']} windows, turb (vov ≥ q80): {s['n_turb']} windows\n")
            f.write(
                f"- (A) Realised variance of 30d future: calm={s['A_rvf_calm']:.3e}, "
                f"turb={s['A_rvf_turb']:.3e}, **turb/calm={s['A_rvf_ratio']:.3f}**\n"
            )
            f.write(
                f"- (B) Within-path std (per cell, avg): calm={s['B_within_std_calm']:.4f}, "
                f"turb={s['B_within_std_turb']:.4f}, **turb/calm={s['B_within_std_ratio']:.3f}**\n"
            )
            f.write(
                f"- (C) Marginal 90% CI width across regime: calm={s['C_marginal_ci_calm']:.4f}, "
                f"turb={s['C_marginal_ci_turb']:.4f}, **turb/calm={s['C_marginal_ci_ratio']:.3f}** "
                f"(suite gate target > 1.15, docstring range 1.25–1.52)\n"
            )
            f.write(f"\n**Verdict:** {s['verdict']}\n\n")

        f.write("## Interpretation\n\n")
        rv_r = stats_rv['C_marginal_ci_ratio']
        iv_r = stats_iv['C_marginal_ci_ratio']
        if rv_r > 1.15 and iv_r > 1.15:
            f.write(
                "Both proxies recover turb > calm GT dispersion. The universal model-side\n"
                "regime inversion (calm_wr > turb_wr across all 229a/231a/232/233a variants)\n"
                "is therefore an ARCHITECTURAL defect, not a labelling artefact. Joint-path\n"
                "CFM (H1) is the indicated next step.\n"
            )
        elif rv_r < 1.0:
            f.write(
                "The new RV-based proxy does NOT separate regimes in GT — turb/calm ratio\n"
                "is below 1 on the marginal-CI test. Further architectural work on the\n"
                "conditionality gate is therefore wasted until the proxy is rebuilt.\n"
                "Rebuild the proxy before any joint-path CFM work.\n"
            )
        else:
            f.write(
                "GT ratio is positive but below the 1.15 gate. The suite's gate target is\n"
                "unachievable under this proxy — either loosen the gate to reflect the\n"
                "actual data or rebuild the proxy.\n"
            )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for label, s in [("New RV-based proxy", stats_rv), ("Legacy IV-mean proxy", stats_iv)]:
        print(f"\n{label}:")
        print(
            f"  A (RVF)        turb/calm = {s['A_rvf_ratio']:.3f}  "
            f"(calm={s['A_rvf_calm']:.3e}, turb={s['A_rvf_turb']:.3e})"
        )
        print(
            f"  B (within-std) turb/calm = {s['B_within_std_ratio']:.3f}  "
            f"(calm={s['B_within_std_calm']:.4f}, turb={s['B_within_std_turb']:.4f})"
        )
        print(
            f"  C (marg CI)    turb/calm = {s['C_marginal_ci_ratio']:.3f}  "
            f"(calm={s['C_marginal_ci_calm']:.4f}, turb={s['C_marginal_ci_turb']:.4f})"
        )
        print(f"  Verdict: {s['verdict']}")
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
