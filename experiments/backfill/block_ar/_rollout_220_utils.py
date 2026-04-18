from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from experiments.backfill.block_ar.evaluate_213a_h1_conditional_distribution_suite import (
    _get_loader as _get_h1_loader,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    build_multistep_windows,
)


@dataclass
class RolloutBatch:
    history_01: torch.Tensor
    history_norm: torch.Tensor
    future_01: torch.Tensor
    future_norm: torch.Tensor


class HistoryFutureDictDataset(Dataset):
    def __init__(self, history_norm: torch.Tensor, future_norm: torch.Tensor):
        self.history_norm = history_norm
        self.future_norm = future_norm

    def __len__(self) -> int:
        return int(self.history_norm.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "history": self.history_norm[idx],
            "future": self.future_norm[idx],
        }


class OneDayKernelRolloutWrapper:
    """Adapt a one-day `sample_next_iv` kernel to a multi-day `sample_batched` API."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.noise_dim = getattr(model, "noise_dim", None)

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    @staticmethod
    def _coerce_next_iv(samples: torch.Tensor) -> torch.Tensor:
        if samples.ndim == 3 and samples.shape[-1] == 25:
            return samples.view(samples.shape[0], samples.shape[1], 5, 5)
        if samples.ndim == 4 and samples.shape[-2:] == (5, 5):
            return samples
        raise ValueError(f"Unexpected sample_next_iv shape: {tuple(samples.shape)}")

    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 50,
        n_steps: int = 30,
        chunk_size: int = 8,
        history_is_normalized: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        if history_is_normalized:
            history_01 = denormalize_iv(history)
        else:
            history_01 = history
        batch_size, hist_len = history_01.shape[:2]
        chunk_size = max(1, min(int(chunk_size), int(n_samples)))

        chunks: list[torch.Tensor] = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            hist_k = history_01.unsqueeze(1).expand(batch_size, k, hist_len, 5, 5)
            hist_k = hist_k.reshape(batch_size * k, hist_len, 5, 5).clone()

            frames: list[torch.Tensor] = []
            for _ in range(n_steps):
                next_iv = self.model.sample_next_iv(hist_k, n_samples=1)
                next_iv = self._coerce_next_iv(next_iv).squeeze(1)
                frames.append(next_iv.view(batch_size, k, 5, 5))
                hist_k = torch.cat(
                    [hist_k[:, 1:], next_iv.view(batch_size * k, 1, 5, 5)],
                    dim=1,
                )
            chunks.append(torch.stack(frames, dim=2))
        return torch.cat(chunks, dim=1)


def load_one_day_kernel(
    model_type: str,
    checkpoint_path: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if model_type == "220c":
        from experiments.backfill.block_ar.train_220c_rollout_finetune_212ai import (
            load_model as load_220c_model,
        )

        return load_220c_model(checkpoint_path, device)
    if model_type in {"220d", "220e", "220f"}:
        from experiments.backfill.block_ar.train_220d_recurrent_flow_transition import (
            load_model as load_220d_model,
        )

        return load_220d_model(checkpoint_path, device)
    if model_type == "221a":
        from experiments.backfill.block_ar.train_220d_recurrent_flow_transition import (
            RecurrentFlowTransitionModel,
        )

        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg = payload["config"]
        model = RecurrentFlowTransitionModel(
            n_cells=cfg["n_cells"],
            history_feat_dim=cfg["history_feat_dim"],
            hidden_dim=cfg["hidden_dim"],
            gru_layers=cfg["gru_layers"],
            gru_dropout=cfg["gru_dropout"],
            flow_hidden=cfg["flow_hidden"],
            n_coupling_layers=cfg["n_coupling_layers"],
            ewma_alpha=cfg["ewma_alpha"],
            scale_floor=cfg["scale_floor"],
            include_scale_feature=cfg["include_scale_feature"],
            support_lo=cfg.get("support_lo", 0.01),
            support_hi=cfg.get("support_hi", 1.0),
        )
        model.load_state_dict(payload["model_state_dict"], strict=False)
        model.init_recurrent_cells_from_gru()
        model.to(device).eval()
        return model, payload
    if model_type in {"221d", "221e"}:
        from experiments.backfill.block_ar.train_221d_adapter_multiday_ar_conditional_flow import (
            AdapterRecurrentFlowTransitionModel,
        )

        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg = payload["config"]
        model = AdapterRecurrentFlowTransitionModel(
            n_cells=cfg["n_cells"],
            history_feat_dim=cfg["history_feat_dim"],
            hidden_dim=cfg["hidden_dim"],
            gru_layers=cfg["gru_layers"],
            gru_dropout=cfg["gru_dropout"],
            flow_hidden=cfg["flow_hidden"],
            n_coupling_layers=cfg["n_coupling_layers"],
            ewma_alpha=cfg["ewma_alpha"],
            scale_floor=cfg["scale_floor"],
            include_scale_feature=cfg["include_scale_feature"],
            support_lo=cfg.get("support_lo", 0.01),
            support_hi=cfg.get("support_hi", 1.0),
            adapter_hidden=cfg.get("adapter_hidden", 64),
            adapter_ramp_steps=cfg.get("adapter_ramp_steps", 0),
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.to(device).eval()
        return model, payload
    if model_type in {"221b", "221c"}:
        from experiments.backfill.block_ar.train_220d_recurrent_flow_transition import (
            RecurrentFlowTransitionModel,
        )

        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg = payload["config"]
        model = RecurrentFlowTransitionModel(
            n_cells=cfg["n_cells"],
            history_feat_dim=cfg["history_feat_dim"],
            hidden_dim=cfg["hidden_dim"],
            gru_layers=cfg["gru_layers"],
            gru_dropout=cfg["gru_dropout"],
            flow_hidden=cfg["flow_hidden"],
            n_coupling_layers=cfg["n_coupling_layers"],
            ewma_alpha=cfg["ewma_alpha"],
            scale_floor=cfg["scale_floor"],
            include_scale_feature=cfg["include_scale_feature"],
            support_lo=cfg.get("support_lo", 0.01),
            support_hi=cfg.get("support_hi", 1.0),
        )
        # DO NOT call init_recurrent_cells_from_gru -- cells are trained
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.to(device).eval()
        return model, payload
    if model_type == "220g":
        from experiments.backfill.block_ar.train_220g_slow_regime_flow_transition import (
            load_model as load_220g_model,
        )

        return load_220g_model(checkpoint_path, device)
    if model_type == "220i":
        from experiments.backfill.block_ar.train_220i_slow_factor_ema_flow_transition import (
            load_model as load_220i_model,
        )

        return load_220i_model(checkpoint_path, device)
    if model_type == "220j":
        from experiments.backfill.block_ar.train_220j_slow_factor_drift_flow_transition import (
            load_model as load_220j_model,
        )

        return load_220j_model(checkpoint_path, device)
    if model_type == "183c":
        from experiments.backfill.block_ar.analyze_183c_best_mechanism import (
            load_model as load_183c_model,
        )

        return load_183c_model(checkpoint_path, device)
    if model_type in {"223a", "223b", "224a", "224b", "224c", "224d", "224e"}:
        from experiments.backfill.block_ar.train_223a_generated_history_finetune import (
            load_model as load_223_model,
        )

        return load_223_model(checkpoint_path, device)
    if model_type in {"226a", "226b"}:
        from experiments.backfill.block_ar.train_226a_factor_decoupled_flow import (
            load_model as load_226a_model,
        )

        return load_226a_model(checkpoint_path, device)
    if model_type == "227a":
        from experiments.backfill.block_ar.train_227a_factor_ar import (
            load_model as load_227a_model,
        )

        return load_227a_model(checkpoint_path, device)
    if model_type in {"231a", "231b", "231c"}:
        from experiments.backfill.block_ar.train_231a_hybrid_recurrent_factor_flow import (
            load_model as load_231a_model,
        )

        return load_231a_model(checkpoint_path, device)
    if model_type == "232a":
        from experiments.backfill.block_ar.train_232a_regime_mixture import (
            load_model as load_232a_model,
        )

        return load_232a_model(checkpoint_path, device)
    if model_type == "232b":
        from experiments.backfill.block_ar.train_232b_heavy_tail import (
            load_model as load_232b_model,
        )

        return load_232b_model(checkpoint_path, device)
    if model_type in {"232c", "232d"}:
        # 232c/d use 227a's FactorARModel architecturally (loss-only variants)
        from experiments.backfill.block_ar.train_227a_factor_ar import (
            load_model as load_227a_model,
        )

        return load_227a_model(checkpoint_path, device)
    if model_type in {"233a", "233a_full", "233a_B", "233a_C"}:
        from experiments.backfill.block_ar.train_233a_twopath_factor_ar import (
            load_model as load_233a_model,
        )

        return load_233a_model(checkpoint_path, device)
    loader = _get_h1_loader(model_type)
    return loader(checkpoint_path, device)


def build_rollout_windows(
    data_path: str,
    history_len: int,
    future_len: int,
    test_start: int,
    val_size: int,
    max_windows: int | None,
    device: torch.device,
    split: str = "val",
) -> RolloutBatch:
    raw = np.load(data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)

    max_train_idx = test_start - history_len - future_len
    train_indices = np.arange(0, max_train_idx - val_size)
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)
    indices = train_indices if split == "train" else val_indices
    if max_windows is not None:
        indices = indices[:max_windows]

    history_01, future_01 = build_multistep_windows(indices, surf_tensor, history_len, future_len)
    history_norm = normalize_iv(history_01)
    future_norm = normalize_iv(future_01)
    future_01 = future_01.view(history_01.shape[0], future_len, 5, 5)
    return RolloutBatch(
        history_01=history_01,
        history_norm=history_norm,
        future_01=future_01,
        future_norm=future_norm.view(history_01.shape[0], future_len, 5, 5),
    )


def rollout_samples_in_batches(
    wrapper: OneDayKernelRolloutWrapper,
    history_norm: torch.Tensor,
    n_samples: int,
    n_steps: int,
    batch_size: int,
    chunk_size: int,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, history_norm.shape[0], batch_size):
        end = min(start + batch_size, history_norm.shape[0])
        batch_hist = history_norm[start:end]
        samples = wrapper.sample_batched(
            batch_hist,
            n_samples=n_samples,
            n_steps=n_steps,
            chunk_size=chunk_size,
            history_is_normalized=True,
        )
        outputs.append(samples.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


@torch.no_grad()
def evaluate_rollout_subset(
    wrapper: OneDayKernelRolloutWrapper,
    history_norm: torch.Tensor,
    future_01: torch.Tensor,
    batch_size: int,
    n_samples: int,
    n_steps: int,
    chunk_size: int,
    horizons: Iterable[int] = (1, 5, 10, 20, 30),
) -> dict[str, float]:
    samples = rollout_samples_in_batches(
        wrapper=wrapper,
        history_norm=history_norm,
        n_samples=n_samples,
        n_steps=n_steps,
        batch_size=batch_size,
        chunk_size=chunk_size,
    )
    results: dict[str, float] = {}
    widths_h30 = []
    hist_01 = denormalize_iv(history_norm).cpu().numpy()
    mean_iv_hist = hist_01.mean(axis=(2, 3))
    vov = np.diff(mean_iv_hist, axis=1).std(axis=1)
    q20 = float(np.quantile(vov, 0.2))
    q80 = float(np.quantile(vov, 0.8))
    calm = vov <= q20
    turb = vov >= q80

    for h in horizons:
        if h > future_01.shape[1]:
            continue
        gt_h = future_01[:, h - 1].cpu().numpy()
        samp_h = samples[:, :, h - 1]
        lo = np.quantile(samp_h, 0.05, axis=1)
        hi = np.quantile(samp_h, 0.95, axis=1)
        cov = ((gt_h >= lo) & (gt_h <= hi)).mean()
        width = (hi - lo).mean()
        results[f"cov90_h{h}"] = float(cov)
        results[f"width90_h{h}"] = float(width)
        floor_h = float((samp_h <= 0.001).mean())
        ceil_h = float((samp_h >= 0.99).mean())
        results[f"at_floor_h{h}"] = floor_h
        results[f"at_ceiling_h{h}"] = ceil_h
        window_width = (hi - lo).mean(axis=(1, 2))
        if calm.any() and turb.any():
            results[f"turb_calm_ratio_h{h}"] = float(
                window_width[turb].mean() / max(window_width[calm].mean(), 1e-8)
            )
        if h == 30 or (h == max([hh for hh in horizons if hh <= future_01.shape[1]], default=h)):
            widths_h30 = window_width

    if len(widths_h30):
        results["width90_terminal_mean"] = float(np.mean(widths_h30))
    return results


def rollout_energy_score_levels(
    model: torch.nn.Module,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    n_samples: int,
    rollout_horizons: Iterable[int],
    chunk_size: int,
):
    from experiments.backfill.block_ar.train_212b_h1_minimal_direct_stochastic_delta import (
        energy_score,
    )

    wrapper = OneDayKernelRolloutWrapper(model)
    rollout = wrapper.sample_batched(
        normalize_iv(history_01),
        n_samples=n_samples,
        n_steps=future_01.shape[1],
        chunk_size=chunk_size,
        history_is_normalized=True,
    )
    rollout_flat = rollout.view(rollout.shape[0], rollout.shape[1], rollout.shape[2], 25)
    target_flat = future_01.view(future_01.shape[0], future_01.shape[1], 25)
    losses = []
    for h in rollout_horizons:
        if 1 <= h <= future_01.shape[1]:
            losses.append(energy_score(rollout_flat[:, :, h - 1], target_flat[:, h - 1]))
    if not losses:
        raise ValueError("No valid rollout_horizons for rollout_energy_score_levels")
    loss = torch.stack(losses).mean()
    metrics = {
        "rollout_energy": loss.detach(),
        "rollout_terminal_std": rollout_flat[:, :, -1].std(dim=1).mean().detach(),
    }
    return loss, metrics


def write_markdown_summary(path: str | Any, title: str, lines: list[str]) -> None:
    from pathlib import Path

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("# " + title + "\n\n" + "\n".join(lines).rstrip() + "\n")


def make_serializable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj
