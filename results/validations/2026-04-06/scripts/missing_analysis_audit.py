#!/usr/bin/env python
"""
Missing Analysis Audit for Codex-era best models (176b, 183c).

Systematically checks which follow-up analyses exist on disk and which are missing.
Outputs JSON audit results with priority ratings.

No GPU required. Run from repo root with PYTHONPATH=.
"""

import json
import os
from pathlib import Path
from datetime import datetime

REPO = Path("/home/max/Documents/vol-surface-vae-pub")
RESULTS_DIR = REPO / "results"
BLOCK_AR = RESULTS_DIR / "block_ar"
VALIDATIONS = RESULTS_DIR / "validations"
MODELS_DIR = REPO / "models" / "backfill"
EXPERIMENTS_DIR = REPO / "experiments" / "backfill" / "block_ar"

# ----- Model paths -----
MODEL_PATHS = {
    "176b": {
        "best": MODELS_DIR / "shared_local_template_mixture_residual_flow_structured_joint_student_t_176b" / "best_model.pt",
        "final": MODELS_DIR / "shared_local_template_mixture_residual_flow_structured_joint_student_t_176b" / "final_model.pt",
    },
    "183c": {
        "best": MODELS_DIR / "state_metric_transport_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_183c" / "best_model.pt",
        "final": MODELS_DIR / "state_metric_transport_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_183c" / "final_model.pt",
    },
}

# ----- Training scripts -----
TRAIN_SCRIPTS = {
    "176b": EXPERIMENTS_DIR / "train_176b_shared_local_template_mixture.py",
    "183c": EXPERIMENTS_DIR / "train_183c_state_metric_transport.py",
}

# ----- Analysis scripts -----
ANALYSIS_SCRIPTS = {
    "176b": [
        EXPERIMENTS_DIR / "analyze_176b_mechanisms.py",
        EXPERIMENTS_DIR / "analyze_176b_failures.py",
        EXPERIMENTS_DIR / "analyze_176b_mean_reversion.py",
    ],
    "183c": [
        EXPERIMENTS_DIR / "analyze_183c_best_mechanism.py",
        EXPERIMENTS_DIR / "analyze_183c_183d_comparison.py",
    ],
}


def file_exists(p):
    return Path(p).exists()


def find_glob(directory, pattern):
    """Find files matching a glob pattern (recursive with **)."""
    if not Path(directory).exists():
        return []
    # Use rglob for recursive matching if pattern doesn't start with **
    results = sorted(Path(directory).glob(pattern))
    if not results:
        results = sorted(Path(directory).rglob(pattern.lstrip("*").lstrip("/")))
    return results


def find_dirs_matching(directory, pattern):
    """Find directories matching a name pattern."""
    if not Path(directory).exists():
        return []
    results = []
    for d in Path(directory).iterdir():
        if d.is_dir() and pattern in d.name:
            results.append(d)
    return sorted(results)


def check_long_horizon():
    """Check if long-horizon (252-day) tests were run for 176b and 183c."""
    results = {}
    for model in ["176b", "183c"]:
        # Check results/block_ar for long-horizon directories
        lh_dirs = find_glob(BLOCK_AR, f"*{model}*long*") + find_glob(BLOCK_AR, f"*{model}*252*")
        # Check validations for long-horizon scripts/results
        lh_val_scripts = []
        lh_val_results = []
        for date_dir in sorted(VALIDATIONS.iterdir()) if VALIDATIONS.exists() else []:
            if date_dir.is_dir():
                lh_val_scripts.extend(find_glob(date_dir / "scripts", f"*{model}*long*"))
                lh_val_scripts.extend(find_glob(date_dir / "scripts", f"*{model}*252*"))
                lh_val_results.extend(find_glob(date_dir / "verification_results", f"*{model}*long*"))
                lh_val_results.extend(find_glob(date_dir / "verification_results", f"*{model}*252*"))
                lh_val_results.extend(find_glob(date_dir / "analysis", f"*{model}*long*"))
                lh_val_results.extend(find_glob(date_dir / "analysis", f"*{model}*252*"))

        exists = len(lh_dirs) > 0 or len(lh_val_results) > 0
        results[model] = {
            "exists": exists,
            "block_ar_dirs": [str(d.relative_to(REPO)) for d in lh_dirs],
            "validation_scripts": [str(s.relative_to(REPO)) for s in lh_val_scripts],
            "validation_results": [str(r.relative_to(REPO)) for r in lh_val_results],
            "priority": "HIGH" if not exists else "DONE",
            "note": "BOSS REQUIREMENT per CLAUDE.md. 252-day long-horizon test is mandatory." if not exists else "Long-horizon test found.",
        }
    return results


def check_multi_seed():
    """Check if models were run with different seeds."""
    results = {}
    for model in ["176b", "183c"]:
        rerun_dirs = find_glob(BLOCK_AR, f"*{model}*rerun*")
        seed_dirs = find_glob(BLOCK_AR, f"*{model}*seed*")
        exists = len(rerun_dirs) > 0 or len(seed_dirs) > 0
        results[model] = {
            "exists": exists,
            "rerun_dirs": [str(d.relative_to(REPO)) for d in rerun_dirs],
            "seed_dirs": [str(d.relative_to(REPO)) for d in seed_dirs],
            "priority": "MEDIUM" if not exists else "DONE",
            "note": "No multi-seed verification found. 179b had rerun5 (verified on disk). Neither 176b nor 183c has a second seed run." if not exists else "Multi-seed verification found.",
        }
    return results


def check_factor_analysis():
    """Check if PCA / cross-cell correlation / factor analysis was done."""
    results = {}
    for model in ["176b", "183c"]:
        # Check summary.json for cross_cell_correlation section
        # summary.json lives inside subdirectories like 176b_v2_full_30d/summary.json
        model_dirs = find_dirs_matching(BLOCK_AR, model)
        summaries = [d / "summary.json" for d in model_dirs if (d / "summary.json").exists()]
        has_cross_cell = False
        cross_cell_data = {}
        for s in summaries:
            try:
                d = json.load(open(s))
                if "cross_cell_correlation" in d:
                    cc = d["cross_cell_correlation"]
                    has_cross_cell = True
                    cross_cell_data[str(s.relative_to(REPO))] = {
                        "gt_eff_rank": cc.get("gt_eff_rank"),
                        "gen_eff_rank": cc.get("gen_eff_rank"),
                        "rank_ratio": cc.get("rank_ratio"),
                        "corr_ratio": cc.get("corr_ratio"),
                        "overall_pass": cc.get("overall_pass"),
                    }
            except Exception:
                pass

        # Check for dedicated PCA/factor analysis files
        factor_files = []
        for date_dir in sorted(VALIDATIONS.iterdir()) if VALIDATIONS.exists() else []:
            if date_dir.is_dir():
                factor_files.extend(find_glob(date_dir / "analysis", f"*{model}*factor*"))
                factor_files.extend(find_glob(date_dir / "analysis", f"*{model}*pca*"))

        # Check mechanistic analyses
        mech_files = []
        for date_dir in sorted(VALIDATIONS.iterdir()) if VALIDATIONS.exists() else []:
            if date_dir.is_dir():
                mech_files.extend(find_glob(date_dir / "analysis", f"*{model}*mechanistic*"))

        results[model] = {
            "has_cross_cell_in_v2_harness": has_cross_cell,
            "cross_cell_data": cross_cell_data,
            "dedicated_factor_analyses": [str(f.relative_to(REPO)) for f in factor_files],
            "mechanistic_analyses": [str(f.relative_to(REPO)) for f in mech_files],
            "priority": "LOW" if has_cross_cell else "MEDIUM",
            "note": "Cross-cell correlation in v2 harness covers basic factor structure. Dedicated PCA analysis not found but S9 data provides effective rank and correlation ratio." if has_cross_cell else "No factor analysis found.",
        }
    return results


def check_checkpoint_comparison():
    """Check if both best and final checkpoints were evaluated and compared."""
    results = {}
    for model in ["176b", "183c"]:
        model_dirs = find_dirs_matching(BLOCK_AR, model)
        best_dirs = [d for d in model_dirs if "_best_" in d.name or (f"{model}_v2" in d.name and "_final_" not in d.name)]
        final_dirs = [d for d in model_dirs if "_final_" in d.name]
        best_results = [d / "summary.json" for d in best_dirs if (d / "summary.json").exists()]
        final_results = [d / "summary.json" for d in final_dirs if (d / "summary.json").exists()]

        # Parse results
        best_pass_counts = {}
        final_pass_counts = {}
        for p in best_results:
            try:
                d = json.load(open(p))
                suite_names = ["surface", "coverage", "conditionality", "time_series",
                               "block_ar", "cointegration", "regime_coverage",
                               "distributional", "cross_cell_correlation",
                               "mean_reversion", "pathwise_jump_realism"]
                passes = sum(1 for s in suite_names if s in d and d[s].get("overall_pass"))
                total = sum(1 for s in suite_names if s in d)
                best_pass_counts[str(p.relative_to(REPO))] = f"{passes}/{total}"
            except Exception:
                pass

        for p in final_results:
            try:
                d = json.load(open(p))
                suite_names = ["surface", "coverage", "conditionality", "time_series",
                               "block_ar", "cointegration", "regime_coverage",
                               "distributional", "cross_cell_correlation",
                               "mean_reversion", "pathwise_jump_realism"]
                passes = sum(1 for s in suite_names if s in d and d[s].get("overall_pass"))
                total = sum(1 for s in suite_names if s in d)
                final_pass_counts[str(p.relative_to(REPO))] = f"{passes}/{total}"
            except Exception:
                pass

        has_both = len(best_pass_counts) > 0 and len(final_pass_counts) > 0
        results[model] = {
            "has_best_eval": len(best_pass_counts) > 0,
            "has_final_eval": len(final_pass_counts) > 0,
            "has_both": has_both,
            "best_results": best_pass_counts,
            "final_results": final_pass_counts,
            "priority": "LOW" if has_both else "MEDIUM",
            "note": "Both best and final checkpoints evaluated." if has_both else "Missing evaluation of one or both checkpoint types.",
        }
    return results


def check_training_reproducibility():
    """Check if training can be reproduced from scratch."""
    results = {}
    for model in ["176b", "183c"]:
        script_exists = file_exists(TRAIN_SCRIPTS[model])
        checkpoints_exist = all(file_exists(p) for p in MODEL_PATHS[model].values())

        # Check if config is saved in checkpoint
        config_in_checkpoint = False
        if checkpoints_exist:
            try:
                import torch
                cp = torch.load(MODEL_PATHS[model]["best"], map_location="cpu", weights_only=False)
                config_in_checkpoint = "config" in cp
            except Exception:
                pass

        # Check for seed in training script
        seed_documented = False
        if script_exists:
            try:
                content = open(TRAIN_SCRIPTS[model]).read()
                seed_documented = "seed" in content.lower()
            except Exception:
                pass

        all_good = script_exists and config_in_checkpoint and checkpoints_exist
        results[model] = {
            "training_script_exists": script_exists,
            "training_script_path": str(TRAIN_SCRIPTS[model].relative_to(REPO)) if script_exists else None,
            "checkpoints_exist": checkpoints_exist,
            "config_in_checkpoint": config_in_checkpoint,
            "seed_documented": seed_documented,
            "priority": "DONE" if all_good else "HIGH",
            "note": "Training script, config, and checkpoints all available for reproduction." if all_good else "Missing training script or config for reproduction.",
        }
    return results


def check_harness_consistency():
    """Check if models were evaluated under all harness versions."""
    results = {}
    for model in ["176b", "183c"]:
        model_dirs = find_dirs_matching(BLOCK_AR, model)
        all_results = [d / "summary.json" for d in model_dirs if (d / "summary.json").exists()]
        harness_versions = set()
        result_details = {}
        for p in all_results:
            dirname = p.parent.name
            # Extract harness version from directory name
            # Pattern: {model}_{checkpoint}_v2_{harness}_full_30d
            harness_versions.add(dirname)
            try:
                d = json.load(open(p))
                suite_names = ["surface", "coverage", "conditionality", "time_series",
                               "block_ar", "cointegration", "regime_coverage",
                               "distributional", "cross_cell_correlation",
                               "mean_reversion", "pathwise_jump_realism"]
                passes = sum(1 for s in suite_names if s in d and d[s].get("overall_pass"))
                total = sum(1 for s in suite_names if s in d)
                tested_suites = [s for s in suite_names if s in d]
                result_details[dirname] = {
                    "pass_count": f"{passes}/{total}",
                    "tested_suites_count": total,
                }
            except Exception:
                pass

        # Check which harness variants are present
        has_s3mrj = any("s3mrj" in v and "spec" not in v for v in harness_versions)
        has_s3mrjspec = any("s3mrjspec" in v for v in harness_versions)
        has_base_v2 = any("v2" in v and "s3" not in v for v in harness_versions)

        results[model] = {
            "harness_versions_found": sorted(harness_versions),
            "result_details": result_details,
            "has_s3mrj": has_s3mrj,
            "has_s3mrjspec": has_s3mrjspec,
            "has_base_v2": has_base_v2,
            "priority": "LOW" if (has_s3mrj and has_s3mrjspec) else "MEDIUM",
            "note": f"Evaluated under {len(harness_versions)} harness variant(s).",
        }
    return results


def check_mechanistic_analysis():
    """Check what mechanistic analyses exist."""
    results = {}
    for model in ["176b", "183c"]:
        analyses = {}
        for date_dir in sorted(VALIDATIONS.iterdir()) if VALIDATIONS.exists() else []:
            if date_dir.is_dir():
                analysis_dir = date_dir / "analysis"
                if analysis_dir.exists():
                    for d in analysis_dir.iterdir():
                        if d.is_dir() and model in d.name:
                            analyses[d.name] = {
                                "path": str(d.relative_to(REPO)),
                                "files": [f.name for f in d.iterdir() if f.is_file()],
                                "date": date_dir.name,
                            }

        # Check for mean reversion specifically
        has_mean_reversion = any("mean_reversion" in k for k in analyses)

        # Check for failure analysis
        has_failure_analysis = any("failure" in k for k in analyses)

        results[model] = {
            "analyses_found": analyses,
            "count": len(analyses),
            "has_mean_reversion": has_mean_reversion,
            "has_failure_analysis": has_failure_analysis,
            "priority": "LOW",
            "note": f"{len(analyses)} mechanistic analyses found.",
        }
    return results


def build_audit():
    """Build the full audit."""
    audit = {
        "metadata": {
            "audit_date": datetime.now().isoformat(),
            "models_audited": ["176b", "183c"],
            "description": "Missing analysis audit for Codex-era best models",
        },
        "analyses": {
            "long_horizon_252d": check_long_horizon(),
            "multi_seed": check_multi_seed(),
            "factor_analysis": check_factor_analysis(),
            "checkpoint_comparison": check_checkpoint_comparison(),
            "training_reproducibility": check_training_reproducibility(),
            "harness_consistency": check_harness_consistency(),
            "mechanistic_analysis": check_mechanistic_analysis(),
        },
    }

    # Build summary table
    def _resolve_exists(d):
        """Determine if an analysis exists from the data dict."""
        if "exists" in d:
            return d["exists"]
        if "has_both" in d:
            return d["has_both"]
        if "count" in d:
            return d["count"] > 0
        if "has_cross_cell_in_v2_harness" in d:
            return d["has_cross_cell_in_v2_harness"]
        if "training_script_exists" in d:
            return d["training_script_exists"] and d.get("config_in_checkpoint", False)
        return d.get("priority") in ("DONE", "LOW")

    summary_table = []
    for analysis_name, analysis_data in audit["analyses"].items():
        row = {"analysis": analysis_name}
        for model in ["176b", "183c"]:
            if model in analysis_data:
                row[model] = {
                    "exists": _resolve_exists(analysis_data[model]),
                    "priority": analysis_data[model]["priority"],
                    "note": analysis_data[model]["note"],
                }
        summary_table.append(row)

    audit["summary_table"] = summary_table

    # Identify HIGH priority items
    high_priority = []
    for analysis_name, analysis_data in audit["analyses"].items():
        for model in ["176b", "183c"]:
            if model in analysis_data and analysis_data[model]["priority"] == "HIGH":
                high_priority.append({
                    "analysis": analysis_name,
                    "model": model,
                    "note": analysis_data[model]["note"],
                })
    audit["high_priority_items"] = high_priority

    return audit


def main():
    audit = build_audit()

    # Print summary
    print("=" * 80)
    print("MISSING ANALYSIS AUDIT - 176b / 183c")
    print("=" * 80)
    print()

    header = f"{'Analysis':<30} {'176b':<25} {'183c':<25} {'Status':<10}"
    print(header)
    print("-" * 90)

    for row in audit["summary_table"]:
        name = row["analysis"]
        r176 = row.get("176b", {})
        r183 = row.get("183c", {})
        p176 = r176.get("priority", "N/A")
        p183 = r183.get("priority", "N/A")
        e176 = "YES" if r176.get("exists") else "NO"
        e183 = "YES" if r183.get("exists") else "NO"
        worst_priority = "HIGH" if "HIGH" in [p176, p183] else ("MEDIUM" if "MEDIUM" in [p176, p183] else "LOW")
        print(f"{name:<30} {e176+' ('+p176+')':<25} {e183+' ('+p183+')':<25} {worst_priority:<10}")

    print()
    if audit["high_priority_items"]:
        print("HIGH PRIORITY ITEMS:")
        for item in audit["high_priority_items"]:
            print(f"  - [{item['model']}] {item['analysis']}: {item['note']}")
    else:
        print("No HIGH priority items found.")

    # Print detailed suite results
    print()
    print("=" * 80)
    print("VALIDATION SUITE RESULTS (best harness per model)")
    print("=" * 80)
    print()

    suite_map = {
        "surface": "S1 Surface",
        "coverage": "S2 CI Cov",
        "conditionality": "S3 Cond",
        "time_series": "S4 TimeSer",
        "block_ar": "S5 Block-AR",
        "cointegration": "S6 Coint",
        "regime_coverage": "S7 Regime",
        "distributional": "S8 Distrib",
        "cross_cell_correlation": "S9 XCell",
        "mean_reversion": "S10 MR",
        "pathwise_jump_realism": "S11 Jump",
    }

    best_paths = {
        "176b_best": "results/block_ar/176b_v2_full_30d/summary.json",
        "176b_best_s3fixed": "results/block_ar/176b_v2_s3fixed_full_30d/summary.json",
        "183c_best_s3mrj": "results/block_ar/183c_best_v2_s3mrj_full_30d/summary.json",
        "183c_best_s3mrjspec": "results/block_ar/183c_best_v2_s3mrjspec_full_30d/summary.json",
    }

    for label, path in best_paths.items():
        try:
            d = json.load(open(path))
            passes = []
            fails = []
            missing = []
            for k, sname in suite_map.items():
                if k in d:
                    op = d[k].get("overall_pass")
                    if op:
                        passes.append(sname)
                    else:
                        fails.append(sname)
                else:
                    missing.append(sname)
            print(f"{label}: {len(passes)}/{len(passes)+len(fails)} PASS")
            print(f"  PASS:  {', '.join(passes)}")
            print(f"  FAIL:  {', '.join(fails)}")
            if missing:
                print(f"  N/T:   {', '.join(missing)}")
            print()
        except Exception as e:
            print(f"{label}: ERROR reading {path}: {e}")

    # Save JSON output
    out_dir = Path("results/validations/2026-04-06/analysis/validation_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "missing_analysis_audit.json"
    with open(out_path, "w") as f:
        json.dump(audit, f, indent=2, default=str)
    print(f"Audit saved to: {out_path}")

    # Save verification results
    ver_dir = Path("results/validations/2026-04-06/verification_results")
    ver_dir.mkdir(parents=True, exist_ok=True)
    verification = {
        "audit_date": datetime.now().isoformat(),
        "models": ["176b", "183c"],
        "high_priority_count": len(audit["high_priority_items"]),
        "high_priority_items": audit["high_priority_items"],
        "summary": {
            row["analysis"]: {
                "176b": row.get("176b", {}).get("priority", "N/A"),
                "183c": row.get("183c", {}).get("priority", "N/A"),
            }
            for row in audit["summary_table"]
        },
    }
    ver_path = ver_dir / "missing_analysis.json"
    with open(ver_path, "w") as f:
        json.dump(verification, f, indent=2, default=str)
    print(f"Verification saved to: {ver_path}")


if __name__ == "__main__":
    main()
