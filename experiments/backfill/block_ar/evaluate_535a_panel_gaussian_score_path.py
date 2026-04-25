#!/usr/bin/env python
"""Evaluate 535a panel Gaussian score-path law through the IV 11-suite."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.coherent_panel_score_path_model import load_model  # noqa: E402
from experiments.backfill.block_ar._factor_conditioning_525_utils import (  # noqa: E402
    official_train_val_indices,
)
from experiments.backfill.block_ar._panel_law_535_utils import (  # noqa: E402
    build_panel_block,
    load_aligned_iv_factor_panel,
    panel_summary,
)
from experiments.backfill.block_ar._rollout_220_utils import (  # noqa: E402
    build_rollout_windows,
    make_serializable,
    write_markdown_summary,
)
from experiments.backfill.block_ar.evaluate_438a_deployable_residual_bootstrap_system import (  # noqa: E402
    history_key,
    run_suite,
    set_seed,
)


def panel_samples_to_iv(samples: torch.Tensor) -> torch.Tensor:
    iv = samples[..., :25].clamp(0.0, 1.0)
    return iv.view(samples.shape[0], samples.shape[1], samples.shape[2], 5, 5)


class HistoryKeyedPanelSampler:
    def __init__(
        self,
        model: Any,
        history_norm_np: np.ndarray,
        panel_history: torch.Tensor,
    ):
        self.model = model
        panel_np = panel_history.detach().cpu().numpy().astype(np.float32)
        self.panel_by_key = {
            history_key(history_norm_np[i]): panel_np[i]
            for i in range(history_norm_np.shape[0])
        }
        self.fallback = panel_np[0]

    def eval(self) -> "HistoryKeyedPanelSampler":
        self.model.eval()
        return self

    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 48,
        n_steps: int = 30,
        chunk_size: int = 8,
        **_: object,
    ) -> torch.Tensor:
        hist_np = history.detach().cpu().numpy()
        rows = [self.panel_by_key.get(history_key(hist), self.fallback) for hist in hist_np]
        panel_history = torch.from_numpy(np.stack(rows, axis=0)).to(history.device)
        samples = self.model.sample_batched(
            panel_history,
            n_samples=n_samples,
            n_steps=n_steps,
            chunk_size=chunk_size,
        )
        return panel_samples_to_iv(samples)


def sample_panel_iv(
    model: Any,
    panel_history: torch.Tensor,
    n_samples: int,
    n_steps: int,
    batch_size: int,
    chunk_size: int,
) -> np.ndarray:
    outs = []
    for start in range(0, panel_history.shape[0], batch_size):
        end = min(start + batch_size, panel_history.shape[0])
        samples = model.sample_batched(
            panel_history[start:end],
            n_samples=n_samples,
            n_steps=n_steps,
            chunk_size=chunk_size,
        )
        outs.append(panel_samples_to_iv(samples).detach().cpu().numpy())
        print(f"  sampled windows {end}/{panel_history.shape[0]}", flush=True)
    return np.concatenate(outs, axis=0).astype(np.float32)


def summary_lines(results: dict[str, Any], checkpoint: str) -> list[str]:
    summary = results["summary"]
    coverage = results["coverage"]
    conditionality = results["conditionality"]
    regime = results["regime_coverage"]
    distributional = results["distributional_fidelity"]
    pathwise = results["pathwise_jump_realism"]
    mean_rev = results["mean_reversion"]
    return [
        f"- checkpoint: `{checkpoint}`",
        f"- windows: `{results['config']['n_windows']}`",
        f"- suite score: `{summary['n_pass']}/11`",
        f"- failed suites: `{', '.join(summary['failed_suites']) if summary['failed_suites'] else 'none'}`",
        "",
        "**Key Metrics**",
        f"- cov90 overall: `{coverage['overall'][0.9]:.3f}`",
        f"- h30 cov90: `{coverage['per_horizon'].get(30, {}).get(0.9, float('nan')):.3f}`",
        f"- conditionality MAE reduction: `{conditionality.get('mae_reduction_pct', float('nan')):.2f}%`",
        f"- turb/calm ratio: `{conditionality.get('turb_calm_ratio', float('nan')):.3f}`",
        f"- regime layer2: `{regime['layer2_n_passing']}/{regime['layer2_n_total']}`",
        f"- daily-change KS pass cells: `{distributional['ks_test']['n_pass']}/25`",
        f"- level KS pass cells: `{distributional['ks_level_test']['n_pass']}/25`",
        f"- mean-reversion active pass: `{mean_rev.get('active_pass_rate', float('nan')):.3f}`",
        f"- max-jump KS: `{pathwise['pathwise_max_jump']['ks_stat']:.3f}`",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=192)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--conditionality_samples", type=int, default=32)
    parser.add_argument("--conditionality_max_batches", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=535)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = load_model(args.checkpoint, device)
    batch = build_rollout_windows(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        max_windows=args.max_windows,
        device=device,
        split="val",
    )
    panel, columns, _ = load_aligned_iv_factor_panel()
    _, val_indices = official_train_val_indices(
        test_start=args.test_start,
        val_size=args.val_size,
        history_len=args.history_len,
        future_len=args.future_len,
    )
    if args.max_windows is not None:
        val_indices = val_indices[: args.max_windows]
    panel_block = build_panel_block(
        panel,
        columns,
        val_indices,
        args.history_len,
        args.future_len,
        device,
    )
    max_hist_err = float(
        torch.max(
            torch.abs(panel_block.history_panel[..., :25].view_as(batch.history_01) - batch.history_01)
        )
        .detach()
        .cpu()
        .item()
    )
    max_future_err = float(
        torch.max(
            torch.abs(panel_block.future_panel[..., :25].view_as(batch.future_01) - batch.future_01)
        )
        .detach()
        .cpu()
        .item()
    )
    if max_hist_err > 1e-6 or max_future_err > 1e-6:
        raise RuntimeError(
            f"Panel/official IV alignment failed: history={max_hist_err}, future={max_future_err}"
        )

    t0 = time.time()
    cond_samples = sample_panel_iv(
        model=model,
        panel_history=panel_block.history_panel,
        n_samples=args.samples,
        n_steps=args.future_len,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )
    sample_time = time.time() - t0
    hist_norm_np = batch.history_norm.detach().cpu().numpy()
    suite_model = HistoryKeyedPanelSampler(
        model=model,
        history_norm_np=hist_norm_np,
        panel_history=panel_block.history_panel,
    ).eval()
    results = run_suite(
        cond_samples=cond_samples,
        batch=batch,
        model=suite_model,
        data_path=args.data_path,
        test_start=args.test_start,
        val_size=args.val_size,
        history_len=args.history_len,
        future_len=args.future_len,
        batch_size=args.batch_size,
        conditionality_samples=args.conditionality_samples,
        conditionality_max_batches=args.conditionality_max_batches,
        device=device,
    )
    results["config"] = {
        "model_type": "535a",
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_best_val": float(payload.get("best_val", float("nan"))),
        "n_windows": int(batch.history_norm.shape[0]),
        "samples": int(args.samples),
        "conditionality_samples": int(args.conditionality_samples),
        "sample_time_s": float(sample_time),
        "seed": int(args.seed),
        "panel": panel_summary(panel_block),
        "alignment": {
            "history_max_abs_error": max_hist_err,
            "future_max_abs_error": max_future_err,
        },
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(results), indent=2), encoding="utf-8")
    write_markdown_summary(
        out_md,
        "535a Panel Gaussian Score-Path Law",
        summary_lines(results, args.checkpoint),
    )
    print(json.dumps(make_serializable(results["summary"]), indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
