#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.deterministic_learned_retrieval_rich_target_backbone import (
    load_model as load_277d_model,
)
from experiments.backfill.block_ar._rollout_220_utils import (
    HistoryFutureDictDataset,
    build_rollout_windows,
    make_serializable,
)
from experiments.backfill.block_ar.evaluate_220h_full_multihorizon_v2_suite import suite_summary
from experiments.backfill.block_ar.test_block_ar_requirements_v2 import (
    run_block_ar_tests,
    run_ci_coverage_tests,
    run_cointegration_tests,
    run_conditionality_tests,
    run_cross_cell_correlation_tests,
    run_distributional_fidelity_tests,
    run_mean_reversion_tests,
    run_pathwise_jump_realism_tests,
    run_regime_coverage_tests,
    run_surface_validity_tests,
    run_time_series_tests,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)
from experiments.backfill.block_ar.train_296b_probabilistic_backbone_zero_mean_coarse_shell_model import (
    build_backbone_center_paths,
)


def flatten_panels(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 5:
        return x.view(x.shape[0], x.shape[1], x.shape[2], -1)
    if x.ndim == 4:
        return x.view(x.shape[0], x.shape[1], -1)
    return x


def residual_raw_bank(
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    center_future_01: torch.Tensor,
) -> torch.Tensor:
    history_norm = flatten_panels(normalize_iv(history_01))
    future_norm = flatten_panels(normalize_iv(future_01))
    center_norm = flatten_panels(normalize_iv(center_future_01))

    target_prev = torch.cat([history_norm[:, -1:].contiguous(), future_norm[:, :-1]], dim=1)
    center_prev = torch.cat([history_norm[:, -1:].contiguous(), center_norm[:, :-1]], dim=1)
    target_raw = future_norm - target_prev
    center_raw = center_norm - center_prev
    residual = target_raw - center_raw
    return residual


class FixedCenterEmpiricalResidualShell:
    def __init__(
        self,
        backbone: torch.nn.Module,
        residual_bank: torch.Tensor,
        residual_scale: float,
        paired: bool,
        device: torch.device,
    ):
        self.backbone = backbone.eval()
        self.residual_bank = residual_bank.to(device)
        self.residual_scale = float(residual_scale)
        self.paired = bool(paired)
        self.device = device

    def eval(self):
        self.backbone.eval()
        return self

    @torch.no_grad()
    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 48,
        n_steps: int = 30,
        chunk_size: int = 8,
        history_is_normalized: bool = True,
        **_: object,
    ) -> torch.Tensor:
        history_norm = history if history_is_normalized else normalize_iv(history)
        if history_norm.ndim == 4:
            history_norm = history_norm.view(history_norm.shape[0], history_norm.shape[1], -1)
        history_norm = history_norm.to(self.device)
        history_01 = denormalize_iv(history_norm)

        center_01 = self.backbone.sample_batched(
            history_norm,
            n_samples=1,
            n_steps=n_steps,
            chunk_size=chunk_size,
            history_is_normalized=True,
        ).squeeze(1)
        center_norm = flatten_panels(normalize_iv(center_01))
        center_prev = torch.cat([history_norm[:, -1:].contiguous(), center_norm[:, :-1]], dim=1)
        center_raw = center_norm - center_prev

        bsz = history_norm.shape[0]
        bank_n = self.residual_bank.shape[0]
        if self.paired:
            half = n_samples // 2
            idx = torch.randint(bank_n, (bsz, half), device=self.device)
            res = self.residual_bank[idx]
            parts = [res, -res]
            if n_samples % 2 == 1:
                parts.append(torch.zeros(bsz, 1, n_steps, history_norm.shape[-1], device=self.device))
            residual = torch.cat(parts, dim=1)
        else:
            idx = torch.randint(bank_n, (bsz, n_samples), device=self.device)
            residual = self.residual_bank[idx]
        total_raw = center_raw.unsqueeze(1) + self.residual_scale * residual

        curr = history_norm[:, -1].unsqueeze(1).expand(-1, n_samples, -1).clone()
        outs: list[torch.Tensor] = []
        for t in range(n_steps):
            curr = torch.clamp(curr + total_raw[:, :, t], -1.0, 1.0)
            outs.append(curr)
        fut_norm = torch.stack(outs, dim=2)
        fut_01 = denormalize_iv(fut_norm).view(bsz, n_samples, n_steps, 5, 5)
        return fut_01.contiguous()


def run_full_suite(
    model: FixedCenterEmpiricalResidualShell,
    cond_samples: np.ndarray,
    batch,
    args: argparse.Namespace,
    output_json: Path,
    output_md: Path,
) -> dict:
    ground_truth = batch.future_01.detach().cpu().numpy()
    history_01 = batch.history_01.detach().cpu().numpy()
    raw = np.load(args.data_path)
    returns = raw["ret"].astype(np.float64)
    max_train_idx = args.test_start - args.history_len - args.future_len
    rollout_start = max_train_idx - args.val_size

    cond_loader = DataLoader(
        HistoryFutureDictDataset(batch.history_norm.detach().cpu(), batch.future_norm.detach().cpu()),
        batch_size=args.batch_size,
        shuffle=False,
    )
    results = {
        "config": {
            "model_type": "297a_fixed_center_shell_oracle",
            "n_windows": int(batch.history_norm.shape[0]),
            "samples": int(args.samples),
            "residual_scale": float(args.residual_scale),
            "paired": bool(args.paired),
            "rollout_start": int(rollout_start),
        },
        "surface": run_surface_validity_tests(cond_samples, ground_truth),
        "coverage": run_ci_coverage_tests(cond_samples, ground_truth),
        "conditionality": run_conditionality_tests(
            model,
            cond_loader,
            n_samples=args.conditionality_samples,
            max_batches=args.conditionality_max_batches,
            device=str(args.device_resolved),
        ),
        "time_series": run_time_series_tests(cond_samples, ground_truth),
        "block_ar": run_block_ar_tests(cond_samples),
        "cointegration": run_cointegration_tests(
            cond_samples,
            ground_truth,
            returns=returns,
            test_start=rollout_start,
            history_len=args.history_len,
            future_len=args.future_len,
        ),
        "regime_coverage": run_regime_coverage_tests(cond_samples, ground_truth, history_01),
        "distributional_fidelity": run_distributional_fidelity_tests(
            cond_samples, ground_truth, history_01
        ),
        "cross_cell_correlation": run_cross_cell_correlation_tests(cond_samples, ground_truth),
        "mean_reversion": run_mean_reversion_tests(cond_samples, ground_truth, history_01),
        "pathwise_jump_realism": run_pathwise_jump_realism_tests(cond_samples, ground_truth),
    }
    n_pass, failed = suite_summary(results)
    results["summary"] = {"n_pass": n_pass, "n_total": 11, "failed_suites": failed}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(make_serializable(results), indent=2))

    dist = results["distributional_fidelity"]
    lines = [
        f"- suite score: `{n_pass}/11`",
        f"- failed suites: `{', '.join(failed) if failed else 'none'}`",
        f"- residual scale: `{args.residual_scale}`",
        f"- paired residuals: `{args.paired}`",
        "",
        "**Key Metrics**",
        f"- h1 cov90: `{results['coverage']['per_horizon'].get(1, {}).get(0.9, results['coverage']['per_horizon'].get('1', {}).get('0.9', float('nan'))):.3f}`",
        f"- h30 cov90: `{results['coverage']['per_horizon'].get(30, {}).get(0.9, results['coverage']['per_horizon'].get('30', {}).get('0.9', float('nan'))):.3f}`",
        f"- turb/calm ratio: `{results['conditionality'].get('turb_calm_ratio', float('nan')):.3f}`",
        f"- change KS cells: `{dist['ks_test']['n_pass']}/25`",
        f"- level KS cells: `{dist['ks_level_test']['n_pass']}/25`",
        f"- corr ratio: `{results['cross_cell_correlation']['corr_ratio']:.3f}`",
        f"- rank ratio: `{results['cross_cell_correlation']['rank_ratio']:.3f}`",
        f"- MR ratio: `{results['mean_reversion']['mr_gt_ratio']:.3f}`",
        f"- pathwise max-jump KS: `{results['pathwise_jump_realism']['pathwise_max_jump']['ks_stat']:.3f}`",
    ]
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("# 297a fixed-center shell oracle diagnostic\n\n" + "\n".join(lines) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="297a fixed-center empirical residual shell oracle")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--backbone_checkpoint", type=str, default="models/backfill/277d_v0_s42/best_model.pt")
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
    parser.add_argument("--residual_scale", type=float, default=1.0)
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_md", type=str, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    args.device_resolved = device

    backbone, _ = load_277d_model(args.backbone_checkpoint, device)
    train_batch = build_rollout_windows(
        args.data_path,
        args.history_len,
        args.future_len,
        args.test_start,
        args.val_size,
        None,
        device,
        split="train",
    )
    val_batch = build_rollout_windows(
        args.data_path,
        args.history_len,
        args.future_len,
        args.test_start,
        args.val_size,
        args.max_windows,
        device,
        split="val",
    )

    train_center = build_backbone_center_paths(
        backbone,
        train_batch.history_01,
        future_len=args.future_len,
        batch_size=128,
    )
    residual_bank = residual_raw_bank(
        train_batch.history_01,
        train_batch.future_01,
        train_center,
    )
    residual_bank = residual_bank - residual_bank.mean(dim=0, keepdim=True)

    model = FixedCenterEmpiricalResidualShell(
        backbone=backbone,
        residual_bank=residual_bank,
        residual_scale=args.residual_scale,
        paired=args.paired,
        device=device,
    ).eval()

    outputs = []
    for start in range(0, val_batch.history_norm.shape[0], args.batch_size):
        end = min(start + args.batch_size, val_batch.history_norm.shape[0])
        samples = model.sample_batched(
            val_batch.history_norm[start:end],
            n_samples=args.samples,
            n_steps=args.future_len,
            chunk_size=args.chunk_size,
            history_is_normalized=True,
        )
        outputs.append(samples.detach().cpu().numpy())
    cond_samples = np.concatenate(outputs, axis=0)
    run_full_suite(
        model,
        cond_samples,
        val_batch,
        args,
        output_json=Path(args.output_json),
        output_md=Path(args.output_md),
    )


if __name__ == "__main__":
    main()
