#!/usr/bin/env python
"""
229a Checkpoint Screening: Run reduced evaluate_220b on all checkpoints,
rank by structural preservation + change KS, select top candidates for full eval.

Ranking rule (lexicographic):
  1. Structural: corr ∈ [0.75, 1.25] AND rank ∈ [1.0, 2.0] AND level_ks ≥ 12
  2. Maximize: change_ks (primary), then jump_ks (secondary, lower is better)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run_reduced_eval(
    checkpoint: str,
    output_json: str,
    output_md: str,
    max_windows: int = 96,
    samples: int = 24,
    device: str = "cuda",
) -> dict | None:
    """Run evaluate_220b with reduced parameters and return results."""
    cmd = [
        sys.executable, "experiments/backfill/block_ar/evaluate_220b_multihorizon_path_suite.py",
        "--model_type", "227a",
        "--checkpoint", checkpoint,
        "--max_windows", str(max_windows),
        "--samples", str(samples),
        "--conditionality_samples", "16",
        "--batch_size", "32",
        "--chunk_size", "8",
        "--output_json", output_json,
        "--output_md", output_md,
        "--device", device,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"  FAILED: {checkpoint}")
        print(f"  stderr: {result.stderr[:500]}")
        return None

    with open(output_json) as f:
        return json.load(f)


def score_checkpoint(results: dict) -> tuple[bool, int, float]:
    """Score a checkpoint for ranking.

    Returns (structural_pass, change_ks, neg_jump_ks) for lexicographic sort.
    """
    corr = results.get("cross_cell_correlation", {}).get("corr_ratio", 0)
    rank = results.get("cross_cell_correlation", {}).get("rank_ratio", 0)
    level_ks = results.get("distributional_fidelity", {}).get("ks_level_test", {}).get("n_pass", 0)
    change_ks = results.get("distributional_fidelity", {}).get("ks_test", {}).get("n_pass", 0)
    jump_ks = results.get("pathwise_jump_realism", {}).get("pathwise_max_jump", {}).get("ks_stat", 1.0)

    structural = (0.75 <= corr <= 1.25) and (1.0 <= rank <= 2.0) and (level_ks >= 12)
    return (structural, change_ks, -jump_ks)


def main():
    parser = argparse.ArgumentParser(description="Screen 229a checkpoints")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_windows", type=int, default=96)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all checkpoints
    checkpoints = sorted(model_dir.glob("checkpoint_ep*.pt"))
    checkpoints += list(model_dir.glob("best_model.pt"))
    checkpoints += list(model_dir.glob("final_model.pt"))
    print(f"Found {len(checkpoints)} checkpoints to screen")

    # Run reduced eval on each
    scored = []
    for ckpt in checkpoints:
        tag = ckpt.stem
        print(f"\nScreening {tag}...")
        oj = str(out_dir / f"{tag}.json")
        om = str(out_dir / f"{tag}.md")
        results = run_reduced_eval(
            str(ckpt), oj, om,
            max_windows=args.max_windows, samples=args.samples, device=args.device,
        )
        if results is None:
            continue

        score = score_checkpoint(results)
        corr = results.get("cross_cell_correlation", {}).get("corr_ratio", 0)
        rank = results.get("cross_cell_correlation", {}).get("rank_ratio", 0)
        change_ks = results.get("distributional_fidelity", {}).get("ks_test", {}).get("n_pass", 0)
        level_ks = results.get("distributional_fidelity", {}).get("ks_level_test", {}).get("n_pass", 0)
        jump_ks = results.get("pathwise_jump_realism", {}).get("pathwise_max_jump", {}).get("ks_stat", 1.0)
        suite_score = results.get("summary", {}).get("n_pass", 0)

        scored.append({
            "checkpoint": str(ckpt),
            "tag": tag,
            "score": score,
            "suite_score": suite_score,
            "change_ks": change_ks,
            "level_ks": level_ks,
            "corr_ratio": corr,
            "rank_ratio": rank,
            "jump_ks": jump_ks,
            "structural_pass": score[0],
        })
        print(f"  suite={suite_score}/7 ChgKS={change_ks}/25 LvlKS={level_ks}/25 "
              f"corr={corr:.3f} rank={rank:.3f} jump={jump_ks:.3f} struct={'PASS' if score[0] else 'FAIL'}")

    # Sort and select top_k
    scored.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n=== RANKING (top {args.top_k}) ===")
    for i, s in enumerate(scored[:args.top_k]):
        print(f"  {i+1}. {s['tag']}: ChgKS={s['change_ks']}/25 LvlKS={s['level_ks']}/25 "
              f"corr={s['corr_ratio']:.3f} struct={'PASS' if s['structural_pass'] else 'FAIL'}")

    # Save ranking
    ranking = {
        "top_k": args.top_k,
        "ranked": [{k: v for k, v in s.items() if k != "score"} for s in scored],
        "selected_for_full_eval": [s["tag"] for s in scored[:args.top_k]],
    }
    (out_dir / "ranking.json").write_text(json.dumps(ranking, indent=2))
    print(f"\nRanking saved to {out_dir / 'ranking.json'}")


if __name__ == "__main__":
    main()
