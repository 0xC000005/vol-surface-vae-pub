#!/usr/bin/env python3
"""Cross-model comparison and factor analysis across 5 session experiments.

Models: 144a, 144b, 144c, 145a, 145c
All CausalARTransformerDecoder with CLN.

Computes:
1. Factor analysis (PCA) on generated daily changes
2. Cross-model metrics table from summary.json
3. Training eff_rank vs test-time eff_rank comparison
4. Per-cell KS daily failure map
"""

import json
import sys
import os
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, "/home/max/Documents/vol-surface-vae-pub")

from diffusion.block_ar.single_pass_ar import SinglePassBlockAR, SinglePassConfig

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

MODELS = {
    "144a": {
        "path": "models/backfill/afcrps_144a/final_model.pt",
        "desc": "scalar vol_scale",
    },
    "144b": {
        "path": "models/backfill/afcrps_144b/best_model.pt",
        "desc": "per-cell scale (SESSION BEST)",
    },
    "144c": {
        "path": "models/backfill/afcrps_144c/best_model.pt",
        "desc": "d_model=128, per-cell scale",
    },
    "145a": {
        "path": "models/backfill/afcrps_145a/best_model.pt",
        "desc": "strong VS lambda=1.0",
    },
    "145c": {
        "path": "models/backfill/afcrps_145c/best_model.pt",
        "desc": "DPP rank loss lambda=0.1",
    },
}

# Summary result dirs
SUMMARY_DIRS = {
    "144a": "results/block_ar/144a_final_30d/summary.json",
    "144b": "results/block_ar/144b_best_30d/summary.json",
    "144c": "results/block_ar/144c_best_30d/summary.json",
    "145a": "results/block_ar/145a_best_30d/summary.json",
    "145c": "results/block_ar/145c_best_30d/summary.json",
}

TRAINING_HISTORY = {
    exp: f"models/backfill/afcrps_{exp}/training_history.json"
    for exp in MODELS
}

ROOT = "/home/max/Documents/vol-surface-vae-pub"
OUTPUT_DIR = os.path.join(ROOT, "results/validations/2026-03-22/analysis/cross_model")
DATA_PATH = os.path.join(ROOT, "data/vol_surface_with_ret.npz")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 50
BATCH_INDICES = list(range(4000, 4900, 100))  # 10 test batches
HISTORY_LEN = 30
FUTURE_LEN = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model(model_name):
    """Load a model with strict=False for compatibility.

    Handles shape mismatches (e.g., scalar vs per-cell log_vol_scale in 144a)
    by broadcasting checkpoint tensors to model shape before loading.
    """
    path = os.path.join(ROOT, MODELS[model_name]["path"])
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    config = SinglePassConfig(**ckpt["config"])
    model = SinglePassBlockAR(config)

    # Fix shape mismatches: broadcast scalar params to per-cell shape
    state_dict = ckpt["model_state_dict"]
    model_state = model.state_dict()
    for key in list(state_dict.keys()):
        if key in model_state:
            if state_dict[key].shape != model_state[key].shape:
                print(f"  [{model_name}] Shape mismatch for {key}: "
                      f"ckpt={state_dict[key].shape} vs model={model_state[key].shape}")
                # Try broadcasting (e.g., scalar -> (25,))
                try:
                    state_dict[key] = state_dict[key].expand_as(model_state[key]).clone()
                    print(f"    -> Broadcast to {state_dict[key].shape}")
                except RuntimeError:
                    print(f"    -> Cannot broadcast, removing key")
                    del state_dict[key]

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [{model_name}] Missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"  [{model_name}] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    model.eval()
    model.to(DEVICE)
    return model


def load_data():
    """Load the dataset and create test windows."""
    data = np.load(DATA_PATH)
    surfaces = data["surface"]  # (N, 5, 5)
    N = len(surfaces)
    print(f"Loaded {N} surfaces from {DATA_PATH}")

    # Create sliding windows for test batches
    windows = []
    for idx in BATCH_INDICES:
        if idx + HISTORY_LEN + FUTURE_LEN > N:
            print(f"  Skipping index {idx} (out of bounds)")
            continue
        history = surfaces[idx : idx + HISTORY_LEN]  # (30, 5, 5)
        future = surfaces[idx + HISTORY_LEN : idx + HISTORY_LEN + FUTURE_LEN]  # (30, 5, 5)
        windows.append((history, future))

    print(f"Created {len(windows)} test windows from indices {BATCH_INDICES}")
    return windows


def normalize_surface(s):
    """Convert [0,1] IV to [-1,1] normalized."""
    return s * 2.0 - 1.0


def denormalize_surface(s):
    """Convert [-1,1] to [0,1]."""
    return (s + 1.0) / 2.0


def compute_effective_rank(eigenvalues):
    """Entropy-based effective rank from eigenvalues."""
    # Normalize to probabilities
    p = eigenvalues / eigenvalues.sum()
    p = p[p > 1e-10]  # filter near-zero
    entropy = -np.sum(p * np.log(p))
    return np.exp(entropy)


def compute_pca_analysis(daily_changes_flat):
    """
    Compute PCA on flattened daily changes.
    daily_changes_flat: (N, 25) array
    Returns dict with eigenvalues, eff_rank, PC1 explained, PC1 loadings.
    """
    # Correlation matrix
    if daily_changes_flat.shape[0] < 2:
        return None

    # Standardize
    means = daily_changes_flat.mean(axis=0)
    stds = daily_changes_flat.std(axis=0)
    stds[stds < 1e-10] = 1.0
    standardized = (daily_changes_flat - means) / stds

    corr = np.corrcoef(standardized.T)  # (25, 25)
    eigenvalues = np.linalg.eigvalsh(corr)
    eigenvalues = np.sort(eigenvalues)[::-1]  # descending
    eigenvalues = np.maximum(eigenvalues, 0)  # clip numerical negatives

    eff_rank = compute_effective_rank(eigenvalues)
    pc1_var_explained = eigenvalues[0] / eigenvalues.sum()

    # PC1 loadings via full eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(corr)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    pc1_loadings = eigvecs[:, 0].reshape(5, 5)

    return {
        "eigenvalues": eigenvalues.tolist(),
        "eff_rank": float(eff_rank),
        "pc1_var_explained": float(pc1_var_explained),
        "pc1_loadings": pc1_loadings.tolist(),
    }


def generate_and_analyze(model, model_name, windows):
    """Generate samples and compute factor analysis."""
    print(f"\n{'='*60}")
    print(f"Generating samples for {model_name}...")
    print(f"{'='*60}")

    all_gen_changes = []  # daily changes from generated samples
    all_gt_changes = []   # daily changes from ground truth

    with torch.no_grad():
        for i, (history, future) in enumerate(windows):
            # Prepare input: (1, 30, 5, 5) normalized to [-1, 1]
            hist_tensor = torch.tensor(
                normalize_surface(history), dtype=torch.float32
            ).unsqueeze(0).to(DEVICE)

            # Generate samples: (1, n_samples, 30, 5, 5) in [0, 1]
            samples = model.sample(hist_tensor, n_samples=N_SAMPLES)
            samples = samples.cpu().numpy()  # (1, 50, 30, 5, 5)

            # Ground truth future is already in [0, 1]
            gt = future  # (30, 5, 5)

            # Daily changes for generated: diff along time axis
            # samples[0] is (50, 30, 5, 5)
            gen_changes = np.diff(samples[0], axis=1)  # (50, 29, 5, 5)
            gen_changes_flat = gen_changes.reshape(-1, 25)  # (50*29, 25)
            all_gen_changes.append(gen_changes_flat)

            # GT daily changes
            gt_changes = np.diff(gt, axis=0)  # (29, 5, 5)
            gt_changes_flat = gt_changes.reshape(-1, 25)  # (29, 25)
            all_gt_changes.append(gt_changes_flat)

            print(f"  Window {i+1}/{len(windows)} done: "
                  f"gen shape {gen_changes_flat.shape}, gt shape {gt_changes_flat.shape}")

    all_gen_changes = np.concatenate(all_gen_changes, axis=0)  # (50*29*10, 25)
    all_gt_changes = np.concatenate(all_gt_changes, axis=0)    # (29*10, 25)

    print(f"  Total gen changes: {all_gen_changes.shape}")
    print(f"  Total gt changes: {all_gt_changes.shape}")

    # PCA analysis
    gen_pca = compute_pca_analysis(all_gen_changes)
    gt_pca = compute_pca_analysis(all_gt_changes)

    return {
        "model_name": model_name,
        "description": MODELS[model_name]["desc"],
        "gen_pca": gen_pca,
        "gt_pca": gt_pca,
        "n_gen_samples": all_gen_changes.shape[0],
        "n_gt_samples": all_gt_changes.shape[0],
    }


def extract_summary_metrics(model_name):
    """Extract key metrics from summary.json."""
    path = os.path.join(ROOT, SUMMARY_DIRS[model_name])
    if not os.path.exists(path):
        return {"error": f"summary.json not found at {path}"}

    d = json.load(open(path))

    # Extract suite pass/fail
    suites = {}
    suite_names = [
        "surface", "coverage", "conditionality", "time_series",
        "block_ar", "cointegration", "regime_coverage",
        "distributional", "cross_cell_correlation",
    ]
    n_pass = 0
    for s in suite_names:
        if s in d:
            if isinstance(d[s], dict):
                passed = d[s].get("overall_pass", d[s].get("pass", None))
                suites[s] = passed
                if passed:
                    n_pass += 1
            else:
                suites[s] = None

    # Key metrics
    metrics = {
        "suites_passed": n_pass,
        "suite_details": suites,
    }

    # Coverage
    cov = d.get("coverage", {})
    overall_cov = cov.get("overall", {})
    metrics["ci_90"] = overall_cov.get("0.9", None)

    # Per-cell worst coverage
    per_cell = cov.get("per_cell_coverage", {})
    if per_cell:
        worst_cell = 1.0
        for h, grid in per_cell.items():
            if isinstance(grid, list):
                for row in grid:
                    for val in row:
                        worst_cell = min(worst_cell, val)
        metrics["worst_cell_ci90"] = worst_cell

    # Kurtosis
    ts = d.get("time_series", {})
    kurt = ts.get("kurtosis", {})
    metrics["kurtosis_ratio"] = kurt.get("kurtosis_ratio", None)

    # KS daily
    dist = d.get("distributional", {})
    ks_daily = dist.get("ks_test", {})
    metrics["ks_daily_n_pass"] = ks_daily.get("n_pass", None)
    metrics["ks_daily_grid"] = ks_daily.get("ks_grid", None)

    # KS levels
    ks_levels = dist.get("ks_level_test", {})
    metrics["ks_levels_n_pass"] = ks_levels.get("n_pass", None)

    # Median bias
    mb = dist.get("median_bias", {})
    metrics["window_floor"] = mb.get("window_floor", None)

    # Cross-cell correlation
    cc = d.get("cross_cell_correlation", {})
    metrics["corr_ratio"] = cc.get("corr_ratio", None)
    metrics["rank_ratio"] = cc.get("rank_ratio", None)
    metrics["gen_eff_rank"] = cc.get("gen_eff_rank", None)
    metrics["gt_eff_rank"] = cc.get("gt_eff_rank", None)
    metrics["gen_pc1_var"] = cc.get("gen_pc1_var", None)

    return metrics


def extract_training_eff_rank(model_name):
    """Extract training eff_rank from training_history.json."""
    path = os.path.join(ROOT, TRAINING_HISTORY[model_name])
    if not os.path.exists(path):
        return None

    d = json.load(open(path))
    if isinstance(d, list):
        ranks = [x.get("eff_rank", 0) for x in d]
        if max(ranks) > 0:
            return {
                "first": ranks[0],
                "last": ranks[-1],
                "max": max(ranks),
                "min": min(ranks),
                "trajectory": ranks,
            }
    return None


def compute_per_cell_failure_map(all_metrics):
    """Count KS daily failures per cell across models."""
    failure_counts = np.zeros((5, 5), dtype=int)
    ks_gate = 0.15

    for model_name, metrics in all_metrics.items():
        ks_grid = metrics.get("ks_daily_grid")
        if ks_grid is not None:
            for i in range(5):
                for j in range(5):
                    if ks_grid[i][j] >= ks_gate:
                        failure_counts[i][j] += 1

    return failure_counts


def main():
    print("=" * 70)
    print("CROSS-MODEL COMPARISON AND FACTOR ANALYSIS")
    print("=" * 70)

    # Load data
    windows = load_data()

    # ──────────────────────────────────────────────────────────────
    # 1. Generate samples and compute PCA for each model
    # ──────────────────────────────────────────────────────────────
    pca_results = {}
    for model_name in MODELS:
        print(f"\nLoading model {model_name}...")
        model = load_model(model_name)
        result = generate_and_analyze(model, model_name, windows)
        pca_results[model_name] = result
        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # ──────────────────────────────────────────────────────────────
    # 2. Extract summary metrics from existing test results
    # ──────────────────────────────────────────────────────────────
    summary_metrics = {}
    for model_name in MODELS:
        summary_metrics[model_name] = extract_summary_metrics(model_name)

    # ──────────────────────────────────────────────────────────────
    # 3. Training eff_rank vs test-time eff_rank
    # ──────────────────────────────────────────────────────────────
    eff_rank_comparison = {}
    for model_name in MODELS:
        train_rank = extract_training_eff_rank(model_name)
        test_rank_from_summary = summary_metrics[model_name].get("gen_eff_rank")
        test_rank_from_pca = pca_results[model_name]["gen_pca"]["eff_rank"] if pca_results[model_name]["gen_pca"] else None
        gt_rank_from_pca = pca_results[model_name]["gt_pca"]["eff_rank"] if pca_results[model_name]["gt_pca"] else None

        eff_rank_comparison[model_name] = {
            "training_eff_rank_last": train_rank["last"] if train_rank else None,
            "training_eff_rank_max": train_rank["max"] if train_rank else None,
            "test_eff_rank_summary": test_rank_from_summary,
            "test_eff_rank_pca": test_rank_from_pca,
            "gt_eff_rank_pca": gt_rank_from_pca,
            "test_rank_ratio_summary": summary_metrics[model_name].get("rank_ratio"),
        }

    # ──────────────────────────────────────────────────────────────
    # 4. Per-cell KS failure map
    # ──────────────────────────────────────────────────────────────
    failure_map = compute_per_cell_failure_map(summary_metrics)

    # ──────────────────────────────────────────────────────────────
    # Save results
    # ──────────────────────────────────────────────────────────────

    # Full results JSON
    full_results = {
        "pca_analysis": {
            name: {
                "description": r["description"],
                "gen_eff_rank": r["gen_pca"]["eff_rank"] if r["gen_pca"] else None,
                "gen_pc1_var_explained": r["gen_pca"]["pc1_var_explained"] if r["gen_pca"] else None,
                "gen_pc1_loadings": r["gen_pca"]["pc1_loadings"] if r["gen_pca"] else None,
                "gen_eigenvalues_top5": r["gen_pca"]["eigenvalues"][:5] if r["gen_pca"] else None,
                "gt_eff_rank": r["gt_pca"]["eff_rank"] if r["gt_pca"] else None,
                "gt_pc1_var_explained": r["gt_pca"]["pc1_var_explained"] if r["gt_pca"] else None,
                "gt_pc1_loadings": r["gt_pca"]["pc1_loadings"] if r["gt_pca"] else None,
                "gt_eigenvalues_top5": r["gt_pca"]["eigenvalues"][:5] if r["gt_pca"] else None,
                "n_gen_samples": r["n_gen_samples"],
                "n_gt_samples": r["n_gt_samples"],
            }
            for name, r in pca_results.items()
        },
        "summary_metrics": summary_metrics,
        "eff_rank_comparison": eff_rank_comparison,
        "per_cell_ks_failure_map": failure_map.tolist(),
        "per_cell_ks_failure_details": {
            "description": "Count of models (out of 5) that fail KS daily at each cell (i,j)",
            "ks_gate": 0.15,
            "max_failures": int(failure_map.max()),
            "hardest_cells": [],
        },
    }

    # Identify hardest cells
    for i in range(5):
        for j in range(5):
            if failure_map[i][j] >= 3:  # majority fail
                full_results["per_cell_ks_failure_details"]["hardest_cells"].append({
                    "cell": [i, j],
                    "n_failures": int(failure_map[i][j]),
                    "ks_values": {
                        name: summary_metrics[name].get("ks_daily_grid", [[0]*5]*5)[i][j]
                        for name in MODELS
                    },
                })

    # Save full results
    results_path = os.path.join(OUTPUT_DIR, "cross_model_results.json")
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"\nSaved full results to {results_path}")

    # Save verification result
    verif_path = os.path.join(
        ROOT, "results/validations/2026-03-22/verification_results/cross_model_comparison.json"
    )
    verification = {
        "task": "cross_model_comparison",
        "date": "2026-03-22",
        "models_analyzed": list(MODELS.keys()),
        "n_test_windows": len(windows),
        "n_samples_per_window": N_SAMPLES,
        "device": DEVICE,
        "status": "complete",
        "key_findings": {},
    }

    # Summarize key findings
    best_eff_rank_model = max(
        MODELS.keys(),
        key=lambda m: pca_results[m]["gen_pca"]["eff_rank"] if pca_results[m]["gen_pca"] else 0,
    )
    best_suites_model = max(
        MODELS.keys(),
        key=lambda m: summary_metrics[m].get("suites_passed", 0),
    )

    verification["key_findings"] = {
        "best_eff_rank_model": best_eff_rank_model,
        "best_eff_rank_value": pca_results[best_eff_rank_model]["gen_pca"]["eff_rank"],
        "best_suites_model": best_suites_model,
        "best_suites_count": summary_metrics[best_suites_model].get("suites_passed", 0),
        "gt_eff_rank": pca_results[list(MODELS.keys())[0]]["gt_pca"]["eff_rank"],
        "hardest_cell_count": int(failure_map.max()),
        "eff_rank_comparison_summary": {
            name: {
                "train": eff_rank_comparison[name]["training_eff_rank_last"],
                "test_pca": eff_rank_comparison[name]["test_eff_rank_pca"],
                "test_summary": eff_rank_comparison[name]["test_eff_rank_summary"],
                "rank_ratio": eff_rank_comparison[name]["test_rank_ratio_summary"],
            }
            for name in MODELS
        },
    }

    with open(verif_path, "w") as f:
        json.dump(verification, f, indent=2, default=str)
    print(f"Saved verification result to {verif_path}")

    # ──────────────────────────────────────────────────────────────
    # Generate markdown summary
    # ──────────────────────────────────────────────────────────────
    md_lines = []
    md_lines.append("# Cross-Model Comparison (2026-03-22)")
    md_lines.append("")
    md_lines.append("## Models")
    md_lines.append("")
    for name, info in MODELS.items():
        md_lines.append(f"- **{name}**: {info['desc']} (`{info['path']}`)")
    md_lines.append("")

    # Cross-model metrics table
    md_lines.append("## Cross-Model Metrics Table")
    md_lines.append("")
    md_lines.append("| Model | Suites | CI 90% | Kurtosis | KS Daily | KS Levels | Rank Ratio | Corr Ratio | Window Floor |")
    md_lines.append("|-------|--------|--------|----------|----------|-----------|------------|------------|--------------|")
    for name in MODELS:
        m = summary_metrics[name]
        suites = m.get("suites_passed", "?")
        ci90 = f"{m.get('ci_90', 0)*100:.1f}%" if m.get("ci_90") else "?"
        kurt = f"{m.get('kurtosis_ratio', 0):.3f}" if m.get("kurtosis_ratio") else "?"
        ks_d = f"{m.get('ks_daily_n_pass', '?')}/25"
        ks_l = f"{m.get('ks_levels_n_pass', '?')}/25"
        rr = f"{m.get('rank_ratio', 0):.3f}" if m.get("rank_ratio") else "?"
        cr = f"{m.get('corr_ratio', 0):.3f}" if m.get("corr_ratio") else "?"
        wf = f"{m.get('window_floor', 0)*100:.1f}%" if m.get("window_floor") else "?"
        md_lines.append(f"| {name} | {suites}/9 | {ci90} | {kurt} | {ks_d} | {ks_l} | {rr} | {cr} | {wf} |")
    md_lines.append("")

    # Suite pass/fail table
    md_lines.append("## Suite Pass/Fail Matrix")
    md_lines.append("")
    suite_names_short = ["S1:Surf", "S2:CI", "S3:Cond", "S4:TS", "S5:BAR", "S6:Coint", "S7:Regime", "S8:Dist", "S9:XCorr"]
    suite_keys = ["surface", "coverage", "conditionality", "time_series", "block_ar", "cointegration", "regime_coverage", "distributional", "cross_cell_correlation"]
    header = "| Model | " + " | ".join(suite_names_short) + " |"
    sep = "|-------|" + "|".join(["------"] * len(suite_names_short)) + "|"
    md_lines.append(header)
    md_lines.append(sep)
    for name in MODELS:
        m = summary_metrics[name]
        sd = m.get("suite_details", {})
        cells = []
        for sk in suite_keys:
            v = sd.get(sk)
            if v is True:
                cells.append("PASS")
            elif v is False:
                cells.append("FAIL")
            else:
                cells.append("?")
        md_lines.append(f"| {name} | " + " | ".join(cells) + " |")
    md_lines.append("")

    # PCA / Factor Analysis table
    md_lines.append("## Factor Analysis (PCA on Daily Changes)")
    md_lines.append("")
    md_lines.append("| Model | Gen Eff Rank | GT Eff Rank | Rank Ratio (PCA) | Gen PC1 Var% | GT PC1 Var% |")
    md_lines.append("|-------|-------------|-------------|------------------|-------------|------------|")
    for name in MODELS:
        r = pca_results[name]
        gen_er = f"{r['gen_pca']['eff_rank']:.3f}" if r["gen_pca"] else "?"
        gt_er = f"{r['gt_pca']['eff_rank']:.3f}" if r["gt_pca"] else "?"
        if r["gen_pca"] and r["gt_pca"]:
            ratio = r["gen_pca"]["eff_rank"] / r["gt_pca"]["eff_rank"]
            ratio_str = f"{ratio:.3f}"
        else:
            ratio_str = "?"
        gen_pc1 = f"{r['gen_pca']['pc1_var_explained']*100:.1f}%" if r["gen_pca"] else "?"
        gt_pc1 = f"{r['gt_pca']['pc1_var_explained']*100:.1f}%" if r["gt_pca"] else "?"
        md_lines.append(f"| {name} | {gen_er} | {gt_er} | {ratio_str} | {gen_pc1} | {gt_pc1} |")
    md_lines.append("")

    # Eff rank comparison table
    md_lines.append("## Training vs Test-Time Effective Rank")
    md_lines.append("")
    md_lines.append("| Model | Train eff_rank (last) | Test eff_rank (PCA) | Test eff_rank (summary) | Test rank_ratio (summary) |")
    md_lines.append("|-------|----------------------|--------------------|-----------------------|--------------------------|")
    for name in MODELS:
        erc = eff_rank_comparison[name]
        tr = f"{erc['training_eff_rank_last']:.3f}" if erc["training_eff_rank_last"] else "N/A"
        tp = f"{erc['test_eff_rank_pca']:.3f}" if erc["test_eff_rank_pca"] else "?"
        ts = f"{erc['test_eff_rank_summary']:.3f}" if erc["test_eff_rank_summary"] else "?"
        trr = f"{erc['test_rank_ratio_summary']:.3f}" if erc["test_rank_ratio_summary"] else "?"
        md_lines.append(f"| {name} | {tr} | {tp} | {ts} | {trr} |")
    md_lines.append("")

    # Per-cell failure map
    md_lines.append("## Per-Cell KS Daily Failure Map")
    md_lines.append("")
    md_lines.append("Count of models (out of 5) failing KS daily at each cell:")
    md_lines.append("(Rows = moneyness, Cols = tenor)")
    md_lines.append("")
    md_lines.append("```")
    for i in range(5):
        md_lines.append("  " + "  ".join(str(failure_map[i][j]) for j in range(5)))
    md_lines.append("```")
    md_lines.append("")

    if full_results["per_cell_ks_failure_details"]["hardest_cells"]:
        md_lines.append("### Hardest Cells (3+ model failures)")
        md_lines.append("")
        for cell_info in full_results["per_cell_ks_failure_details"]["hardest_cells"]:
            i, j = cell_info["cell"]
            md_lines.append(f"- Cell ({i},{j}): {cell_info['n_failures']}/5 failures")
            for name, ks in cell_info["ks_values"].items():
                status = "FAIL" if ks >= 0.15 else "pass"
                md_lines.append(f"  - {name}: KS={ks:.4f} [{status}]")
        md_lines.append("")

    # PC1 loadings comparison
    md_lines.append("## PC1 Loading Patterns (5x5)")
    md_lines.append("")
    md_lines.append("Ground truth PC1 loadings:")
    md_lines.append("```")
    gt_loadings = pca_results[list(MODELS.keys())[0]]["gt_pca"]["pc1_loadings"]
    for row in gt_loadings:
        md_lines.append("  " + "  ".join(f"{v:+.3f}" for v in row))
    md_lines.append("```")
    md_lines.append("")

    for name in MODELS:
        gen_loadings = pca_results[name]["gen_pca"]["pc1_loadings"]
        md_lines.append(f"{name} PC1 loadings:")
        md_lines.append("```")
        for row in gen_loadings:
            md_lines.append("  " + "  ".join(f"{v:+.3f}" for v in row))
        md_lines.append("```")
        md_lines.append("")

    # Write markdown
    md_path = os.path.join(OUTPUT_DIR, "cross_model_summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Saved markdown summary to {md_path}")

    # Print summary to stdout
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for line in md_lines:
        print(line)


if __name__ == "__main__":
    main()
