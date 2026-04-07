#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.gru_encoder import EncoderConfig
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import build_multistep_windows
from experiments.backfill.block_ar.train_182a_pathwise_residual_law import unconstrained_to_iv
from experiments.backfill.block_ar.train_183c_state_metric_transport import StateMetricTransportModel


def load_model(model_path: str, device: str) -> tuple[StateMetricTransportModel, dict]:
    payload = torch.load(model_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    model = StateMetricTransportModel(
        encoder_config=EncoderConfig(**cfg["encoder"]),
        decoder_config=cfg["decoder"],
        flow_config=cfg["flow"],
        path_config=cfg["path"],
        prior_config=cfg["prior"],
        integrated_config=cfg["integrated"],
        state_config=cfg["state"],
        metric_config=cfg["metric"],
        support_lo=cfg["support_lo"],
        support_hi=cfg["support_hi"],
        support_eps=cfg["support_eps"],
        base_nu=cfg["base_nu"],
        mix_chunk_size=cfg["mix_chunk_size"],
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, cfg


def sample_future_with_path_context(
    model: StateMetricTransportModel,
    history_01: torch.Tensor,
    path_context_override: torch.Tensor | None,
    n_samples: int,
    seed: int,
) -> torch.Tensor:
    with torch.no_grad():
        (
            mu,
            time_factor,
            time_diag,
            cell_factor,
            cell_diag,
            scale,
            flow_context,
            base_local_delta,
            block_logits,
        ) = model.forward_from_history(history_01)
        batch, n_frames, n_cells = mu.shape
        cov_t, cov_c = model.covariance_parts(time_factor, time_diag, cell_factor, cell_diag, scale)
        chol_t = torch.linalg.cholesky(cov_t)
        chol_c = torch.linalg.cholesky(cov_c)
        if path_context_override is None:
            path_context = model.build_path_context(flow_context, block_logits, base_local_delta, scale)
        else:
            path_context = path_context_override

        cpu_state = torch.random.get_rng_state()
        if history_01.is_cuda:
            gpu_state = torch.cuda.get_rng_state(device=history_01.device)
        torch.manual_seed(seed)
        if history_01.is_cuda:
            torch.cuda.manual_seed(seed)

        basis_paths, _stats = model.sample_basis_paths(path_context, n_samples=n_samples)
        base_white_flat = model.path_geometry.from_basis(
            basis_paths.view(batch * n_samples, n_frames, n_cells)
        ).reshape(batch * n_samples, n_frames * n_cells)
        base_white = base_white_flat.view(batch * n_samples, model.decoder.n_blocks, model.decoder.block_len, n_cells)

        block_probs = torch.softmax(block_logits, dim=-1)
        sampled_blocks = []
        for b in range(model.decoder.n_blocks):
            sampled = torch.multinomial(block_probs[:, b], num_samples=n_samples, replacement=True)
            sampled_blocks.append(sampled)
        sampled_assign = torch.stack(sampled_blocks, dim=-1)
        assign_flat = sampled_assign.reshape(batch * n_samples, model.decoder.n_blocks)
        sampled_factors, _logdet_cov, _offdiag_rms = model.decoder.build_template_factors(assign_flat)
        lhs = base_white.permute(0, 1, 3, 2).reshape(
            batch * n_samples * model.decoder.n_blocks,
            n_cells,
            model.decoder.block_len,
        )
        factor_flat = sampled_factors.reshape(
            batch * n_samples * model.decoder.n_blocks,
            n_cells,
            n_cells,
        )
        routed_white = torch.matmul(factor_flat, lhs)
        routed_white = routed_white.reshape(
            batch * n_samples,
            model.decoder.n_blocks,
            n_cells,
            model.decoder.block_len,
        ).permute(0, 1, 3, 2)
        routed_white = routed_white.reshape(batch, n_samples, n_frames, n_cells)

        temp = torch.einsum("bij,bsjk->bsik", chol_t, routed_white)
        noise = torch.einsum("bstj,bcj->bstc", temp, chol_c)
        shared_local_delta = model.decoder.build_shared_local_delta(base_local_delta)
        local_scale = torch.exp(0.5 * shared_local_delta).unsqueeze(1)
        out_u = mu.unsqueeze(1) + noise * local_scale

        torch.random.set_rng_state(cpu_state)
        if history_01.is_cuda:
            torch.cuda.set_rng_state(gpu_state, device=history_01.device)
        return unconstrained_to_iv(out_u, lo=model.support_lo, hi=model.support_hi).view(
            batch, n_samples, n_frames, 5, 5
        )


def compute_vol_of_vol(history_01: torch.Tensor) -> torch.Tensor:
    mean_iv = history_01.mean(dim=(-1, -2))
    daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
    return daily_chg.std(dim=1)


def ci90_coverage(samples_01: torch.Tensor, future_01: torch.Tensor) -> float:
    lo = torch.quantile(samples_01, 0.05, dim=1)
    hi = torch.quantile(samples_01, 0.95, dim=1)
    covered = (future_01 >= lo) & (future_01 <= hi)
    return float(covered.float().mean().item())


def mean_ci_width(samples_01: torch.Tensor) -> float:
    lo = torch.quantile(samples_01, 0.05, dim=1)
    hi = torch.quantile(samples_01, 0.95, dim=1)
    return float((hi - lo).mean().item())


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = (
        "models/backfill/state_metric_transport_pathwise_residual_law_"
        "mean_reverting_covariance_mixture_structured_joint_student_t_183c/best_model.pt"
    )
    output_dir = Path("results/validations/2026-04-06/analysis/183c_condition_embedding_effect")
    output_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(model_path, device)

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    history_len = int(cfg["history_len"])
    future_len = int(cfg["future_len"])
    test_start = 4511
    test_indices = np.arange(test_start, surfaces.shape[0] - future_len, dtype=np.int64)
    test_hist, test_future = build_multistep_windows(test_indices[:256], surf_tensor, history_len, future_len)

    batch_size = 32
    all_stats: list[dict] = []
    for start in range(0, test_hist.shape[0], batch_size):
        end = min(start + batch_size, test_hist.shape[0])
        hist = test_hist[start:end]
        fut = test_future[start:end].view(end - start, future_len, 5, 5)
        with torch.no_grad():
            (
                _mu,
                _tf,
                _td,
                _cf,
                _cd,
                scale,
                flow_context,
                base_local_delta,
                block_logits,
            ) = model.forward_from_history(hist)
            normal_ctx = model.build_path_context(flow_context, block_logits, base_local_delta, scale)
            zero_ctx = torch.zeros_like(normal_ctx)
            perm = torch.randperm(normal_ctx.shape[0], device=normal_ctx.device)
            shuffled_ctx = normal_ctx[perm]

        seed = 1000 + start
        normal = sample_future_with_path_context(model, hist, normal_ctx, n_samples=20, seed=seed)
        zeroed = sample_future_with_path_context(model, hist, zero_ctx, n_samples=20, seed=seed)
        shuffled = sample_future_with_path_context(model, hist, shuffled_ctx, n_samples=20, seed=seed)

        normal_mean = normal.mean(dim=1)
        zero_mean = zeroed.mean(dim=1)
        shuffled_mean = shuffled.mean(dim=1)
        fut_01 = fut

        vol = compute_vol_of_vol(hist)
        normal_width = torch.quantile(normal, 0.95, dim=1) - torch.quantile(normal, 0.05, dim=1)
        zero_width = torch.quantile(zeroed, 0.95, dim=1) - torch.quantile(zeroed, 0.05, dim=1)
        shuffle_width = torch.quantile(shuffled, 0.95, dim=1) - torch.quantile(shuffled, 0.05, dim=1)
        normal_window_width = normal_width.mean(dim=(-1, -2, -3))
        zero_window_width = zero_width.mean(dim=(-1, -2, -3))
        shuffle_window_width = shuffle_width.mean(dim=(-1, -2, -3))

        all_stats.append(
            {
                "normal_cov90": ci90_coverage(normal, fut_01),
                "zero_cov90": ci90_coverage(zeroed, fut_01),
                "shuffle_cov90": ci90_coverage(shuffled, fut_01),
                "normal_width": mean_ci_width(normal),
                "zero_width": mean_ci_width(zeroed),
                "shuffle_width": mean_ci_width(shuffled),
                "normal_mae": float((normal_mean - fut_01).abs().mean().item()),
                "zero_mae": float((zero_mean - fut_01).abs().mean().item()),
                "shuffle_mae": float((shuffled_mean - fut_01).abs().mean().item()),
                "zero_vs_normal_absdiff": float((zero_mean - normal_mean).abs().mean().item()),
                "shuffle_vs_normal_absdiff": float((shuffled_mean - normal_mean).abs().mean().item()),
                "ctx_norm": float(normal_ctx.norm(dim=-1).mean().item()),
                "ctx_zero_abs": float(zero_ctx.abs().mean().item()),
                "width_vol_corr_normal": float(torch.corrcoef(torch.stack([vol, normal_window_width]))[0, 1].item()),
                "width_vol_corr_zero": float(torch.corrcoef(torch.stack([vol, zero_window_width]))[0, 1].item()),
                "width_vol_corr_shuffle": float(torch.corrcoef(torch.stack([vol, shuffle_window_width]))[0, 1].item()),
            }
        )

    keys = all_stats[0].keys()
    summary = {k: float(np.mean([row[k] for row in all_stats])) for k in keys}
    summary["n_windows"] = int(test_hist.shape[0])
    summary["n_samples"] = 20
    summary["model_path"] = model_path

    out_path = output_dir / "mechanistic_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
