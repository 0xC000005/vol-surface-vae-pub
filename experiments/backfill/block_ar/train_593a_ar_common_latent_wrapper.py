#!/usr/bin/env python
"""593a: learned common-latent wrapper around the 510a AR transition frontier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.empirical_normal_score_causal_memory_transition_flow_matching import (  # noqa: E402
    EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
    load_model,
)
from experiments.backfill.block_ar._rollout_220_utils import (  # noqa: E402
    build_rollout_windows,
    make_serializable,
    write_markdown_summary,
)
from experiments.backfill.block_ar.evaluate_438a_deployable_residual_bootstrap_system import (  # noqa: E402
    run_suite,
    set_seed,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv  # noqa: E402
from experiments.backfill.block_ar.train_391a_recent_rollout_energy_finetune import (  # noqa: E402
    build_recent_block,
)
from experiments.backfill.block_ar.train_509a_recent_patch_energy_finetune import (  # noqa: E402
    patch_energy_score,
)


class ARCommonLatentWrapper(nn.Module):
    """Frozen AR transition model with one learned scenario-level latent memory shift."""

    def __init__(
        self,
        base_model: EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
        latent_dim: int,
    ):
        super().__init__()
        self.base_model = base_model
        self.latent_dim = int(latent_dim)
        for param in self.base_model.parameters():
            param.requires_grad_(False)
        self.latent_proj = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.base_model.cfg.memory_dim),
        )
        final = self.latent_proj[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self

    def _sample_scores(
        self,
        history_norm: torch.Tensor,
        *,
        n_samples: int,
        n_steps: int,
        flow_steps: int,
        temperature: float | None = None,
    ) -> torch.Tensor:
        base = self.base_model
        history_scores = base.history_scores(history_norm)
        bsz = history_scores.shape[0]
        k = int(n_samples)
        n_flow = max(1, int(flow_steps))
        temp = float(base.cfg.sample_temperature if temperature is None else temperature)
        dt = 1.0 / float(n_flow)
        prefix = (
            history_scores.unsqueeze(1)
            .expand(bsz, k, base.cfg.history_len, base.cfg.n_cells)
            .reshape(bsz * k, base.cfg.history_len, base.cfg.n_cells)
            .clone()
        )
        z = torch.randn(
            bsz * k,
            self.latent_dim,
            device=history_scores.device,
            dtype=history_scores.dtype,
        )
        latent_memory = self.latent_proj(z)
        frames: list[torch.Tensor] = []
        for _step in range(int(n_steps)):
            memory_state = base._encode_prefix_scores(prefix)[:, -1] + latent_memory
            current_score = prefix[:, -1]
            x = temp * torch.randn_like(current_score)
            for flow_step in range(n_flow):
                t = torch.full(
                    (bsz * k,),
                    (flow_step + 0.5) * dt,
                    device=history_scores.device,
                    dtype=history_scores.dtype,
                )
                x = x + dt * base.predict_velocity(x, current_score, memory_state, t)
            next_score = current_score + x
            frames.append(next_score.view(bsz, k, base.cfg.n_cells))
            prefix = torch.cat([prefix, next_score[:, None, :]], dim=1)
        return torch.stack(frames, dim=2)

    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 48,
        n_steps: int = 30,
        chunk_size: int = 4,
        history_is_normalized: bool = True,
        temperature: float | None = None,
        **_: object,
    ) -> torch.Tensor:
        base = self.base_model
        history_norm = history if history_is_normalized else normalize_iv(history)
        chunks: list[torch.Tensor] = []
        chunk_size = max(1, min(int(chunk_size), int(n_samples)))
        for start in range(0, int(n_samples), chunk_size):
            k = min(chunk_size, int(n_samples) - start)
            scores = self._sample_scores(
                history_norm,
                n_samples=k,
                n_steps=int(n_steps),
                flow_steps=int(base.cfg.flow_steps),
                temperature=temperature,
            )
            values = base._scores_to_values(scores)
            chunks.append(values.view(history_norm.shape[0], k, int(n_steps), 5, 5))
        return torch.cat(chunks, dim=1)


def wrapper_patch_loss(
    wrapper: ARCommonLatentWrapper,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    *,
    n_samples: int,
    rollout_flow_steps: int,
    patch_len: int,
    energy_eps: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_scores = wrapper.base_model.target_future_scores(future_norm)
    sampled_scores = wrapper._sample_scores(
        history_norm,
        n_samples=n_samples,
        n_steps=target_scores.shape[1],
        flow_steps=rollout_flow_steps,
    )
    loss, target_dist, pair_dist = patch_energy_score(
        sampled_scores,
        target_scores,
        patch_len=patch_len,
        eps=energy_eps,
    )
    return loss, {
        "loss": loss.detach(),
        "target_dist": target_dist.detach(),
        "pair_dist": pair_dist.detach(),
        "sample_std": sampled_scores.std(unbiased=False).detach(),
        "target_std": target_scores.std(unbiased=False).detach(),
        "latent_proj_abs": wrapper.latent_proj[-1].weight.abs().mean().detach(),
    }


def save_wrapper(path: Path, wrapper: ARCommonLatentWrapper, args: argparse.Namespace, epoch: int, best_val: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "base_checkpoint": args.checkpoint,
            "latent_dim": int(args.latent_dim),
            "epoch": int(epoch),
            "best_val": float(best_val),
            "wrapper_state_dict": wrapper.latent_proj.state_dict(),
            "args": vars(args),
        },
        path,
    )


def load_wrapper(path: str, device: torch.device) -> tuple[ARCommonLatentWrapper, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    base_model, _base_payload = load_model(payload["base_checkpoint"], device)
    wrapper = ARCommonLatentWrapper(base_model, latent_dim=int(payload["latent_dim"])).to(device)
    wrapper.latent_proj.load_state_dict(payload["wrapper_state_dict"])
    wrapper.eval()
    return wrapper, payload


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    base_model, base_payload = load_model(args.checkpoint, device)
    wrapper = ARCommonLatentWrapper(base_model, latent_dim=args.latent_dim).to(device)
    hist_01, fut_01, indices = build_recent_block(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        adaptation_windows=args.adaptation_windows,
        device=device,
    )
    n_val = max(1, int(round(hist_01.shape[0] * float(args.holdout_frac))))
    train_hist, val_hist = hist_01[:-n_val], hist_01[-n_val:]
    train_fut, val_fut = fut_01[:-n_val], fut_01[-n_val:]
    train_loader = DataLoader(
        TensorDataset(train_hist, train_fut),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(TensorDataset(val_hist, val_fut), batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(wrapper.latent_proj.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def run_epoch(loader: DataLoader, train_mode: bool) -> dict[str, float]:
        wrapper.train(train_mode)
        sums: dict[str, float] = {}
        n_batches = 0
        for hist_batch, fut_batch in loader:
            hist_norm = normalize_iv(hist_batch.to(device)).view(hist_batch.shape[0], hist_batch.shape[1], -1)
            fut_norm = normalize_iv(fut_batch.to(device)).view(fut_batch.shape[0], fut_batch.shape[1], -1)
            with torch.set_grad_enabled(train_mode):
                loss, metrics = wrapper_patch_loss(
                    wrapper,
                    hist_norm,
                    fut_norm,
                    n_samples=args.train_sample_count,
                    rollout_flow_steps=args.rollout_flow_steps,
                    patch_len=args.patch_len,
                    energy_eps=args.energy_eps,
                )
                if train_mode:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if args.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(wrapper.latent_proj.parameters(), args.clip_grad)
                    optimizer.step()
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value.item())
            n_batches += 1
        return {key: value / max(n_batches, 1) for key, value in sums.items()}

    print(f"Base checkpoint: {args.checkpoint}")
    print(f"Base epoch: {base_payload.get('epoch')} best_val: {base_payload.get('best_val')}")
    print(f"Recent windows: {hist_01.shape[0]} index range: {indices[0]}..{indices[-1]}")
    print(f"Train/holdout: {train_hist.shape[0]}/{val_hist.shape[0]}")
    best_val = float("inf")
    best_epoch = -1
    records: list[dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
        train_avg = run_epoch(train_loader, train_mode=True)
        val_avg = run_epoch(val_loader, train_mode=False)
        rec = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_avg.items()},
            **{f"val_{key}": value for key, value in val_avg.items()},
        }
        records.append(rec)
        print(json.dumps(make_serializable(rec), sort_keys=True), flush=True)
        if rec["val_loss"] < best_val:
            best_val = rec["val_loss"]
            best_epoch = epoch
            save_wrapper(out_dir / "best_model.pt", wrapper, args, epoch, best_val)
    save_wrapper(out_dir / "final_model.pt", wrapper, args, int(args.epochs), best_val)
    summary = {
        "base_checkpoint": args.checkpoint,
        "base_epoch": base_payload.get("epoch"),
        "base_best_val": base_payload.get("best_val"),
        "best_epoch": int(best_epoch),
        "best_val": float(best_val),
        "adaptation_start_index": int(indices[0]),
        "adaptation_end_index": int(indices[-1]),
        "records": records,
    }
    (out_dir / "training_history.json").write_text(json.dumps(make_serializable(records), indent=2), encoding="utf-8")
    (out_dir / "train_summary.json").write_text(json.dumps(make_serializable(summary), indent=2), encoding="utf-8")
    print(json.dumps(make_serializable(summary), indent=2))


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    wrapper, payload = load_wrapper(args.wrapper_checkpoint, device)
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
    cond_samples = wrapper.sample_batched(
        batch.history_norm,
        n_samples=args.samples,
        n_steps=args.future_len,
        chunk_size=args.chunk_size,
        history_is_normalized=True,
    ).detach().cpu().numpy()
    results = run_suite(
        cond_samples=cond_samples,
        batch=batch,
        model=wrapper,
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
        "mode": "593a_ar_common_latent_wrapper",
        "wrapper_checkpoint": args.wrapper_checkpoint,
        "base_checkpoint": payload["base_checkpoint"],
        "latent_dim": int(payload["latent_dim"]),
        "wrapper_epoch": int(payload["epoch"]),
        "wrapper_best_val": float(payload["best_val"]),
        "n_windows": int(batch.history_norm.shape[0]),
        "samples": int(args.samples),
        "seed": int(args.seed),
    }
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(results), indent=2), encoding="utf-8")
    summary = results["summary"]
    lines = [
        "- source: `frozen 510a/392a AR transition model plus learned scenario latent memory shift`",
        f"- suite score: `{summary['n_pass']}/11`",
        f"- failed suites: `{', '.join(summary['failed_suites']) if summary['failed_suites'] else 'none'}`",
        f"- coverage90 overall: `{results['coverage']['overall'][0.9]:.3f}`",
        f"- conditionality MAE reduction: `{results['conditionality'].get('mae_reduction_pct', float('nan')):.2f}%`",
        f"- regime layer2: `{results['regime_coverage']['layer2_n_passing']}/{results['regime_coverage']['layer2_n_total']}`",
        f"- daily-change KS pass cells: `{results['distributional_fidelity']['ks_test']['n_pass']}/25`",
        f"- level KS pass cells: `{results['distributional_fidelity']['ks_level_test']['n_pass']}/25`",
        f"- corr ratio/rank ratio: `{results['cross_cell_correlation']['corr_ratio']:.3f}` / `{results['cross_cell_correlation']['rank_ratio']:.3f}`",
        f"- mean-reversion active pass: `{results['mean_reversion'].get('active_pass_rate', float('nan')):.3f}`",
        f"- max-jump KS: `{results['pathwise_jump_realism']['pathwise_max_jump']['ks_stat']:.3f}`",
    ]
    write_markdown_summary(out_md, "593a AR Common Latent Wrapper", lines)
    print(json.dumps(make_serializable(summary), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--checkpoint", default="models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt")
    parser.add_argument("--wrapper_checkpoint", default="")
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--adaptation_windows", type=int, default=441)
    parser.add_argument("--holdout_frac", type=float, default=0.2)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--train_sample_count", type=int, default=4)
    parser.add_argument("--rollout_flow_steps", type=int, default=4)
    parser.add_argument("--patch_len", type=int, default=5)
    parser.add_argument("--energy_eps", type=float, default=1e-6)
    parser.add_argument("--max_windows", type=int, default=441)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--conditionality_samples", type=int, default=32)
    parser.add_argument("--conditionality_max_batches", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_md", default="")
    parser.add_argument("--seed", type=int, default=593)
    args = parser.parse_args()

    if args.mode == "train":
        if not args.output_dir:
            raise ValueError("--output_dir is required for train mode")
        train(args)
    else:
        if not args.wrapper_checkpoint or not args.output_json or not args.output_md:
            raise ValueError("--wrapper_checkpoint, --output_json, and --output_md are required for eval mode")
        evaluate(args)


if __name__ == "__main__":
    main()
