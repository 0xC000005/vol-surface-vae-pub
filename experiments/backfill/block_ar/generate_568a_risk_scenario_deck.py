#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import load_one_day_kernel
from experiments.backfill.block_ar.evaluate_564a_stress_selected_510a import (
    StressSelectedScenarioModel,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import denormalize_iv, normalize_iv


def severity_bucket_labels(n_select: int) -> np.ndarray:
    """Return policy labels in the same calm/central/stress order as 564a selection."""
    if n_select <= 0:
        raise ValueError("n_select must be positive")
    n_low = n_select // 3
    n_mid = n_select // 3
    n_high = n_select - n_low - n_mid
    labels = ["calm"] * n_low + ["central"] * n_mid + ["stress"] * n_high
    return np.asarray(labels, dtype="U16")


def _rounded_float(value: float) -> float:
    return round(float(value), 6)


def scenario_diagnostics(scenarios: np.ndarray) -> dict[str, float]:
    scenarios = np.asarray(scenarios, dtype=np.float32)
    if scenarios.ndim != 4:
        raise ValueError("scenarios must have shape (samples, future_len, height, width)")
    finite_mask = np.isfinite(scenarios)
    if not finite_mask.any():
        raise ValueError("scenarios contain no finite values")
    terminal = scenarios[:, -1]
    return {
        "finite_rate": _rounded_float(finite_mask.mean()),
        "min_iv": _rounded_float(np.nanmin(scenarios)),
        "max_iv": _rounded_float(np.nanmax(scenarios)),
        "terminal_mean_iv": _rounded_float(np.nanmean(terminal)),
    }


class BlockwiseLongHorizonModel(torch.nn.Module):
    """Extend a capped AR sampler by rolling repeated blocks and updating history."""

    def __init__(self, base_model: torch.nn.Module, *, max_block_steps: int | None = None) -> None:
        super().__init__()
        self.base_model = base_model
        cfg_future_len = getattr(getattr(base_model, "cfg", None), "future_len", None)
        self.max_block_steps = int(max_block_steps or cfg_future_len or 30)

    def eval(self) -> "BlockwiseLongHorizonModel":
        self.base_model.eval()
        return self

    @torch.no_grad()
    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 48,
        n_steps: int = 30,
        chunk_size: int = 8,
        history_is_normalized: bool = True,
        **kwargs: object,
    ) -> torch.Tensor:
        if n_steps <= self.max_block_steps:
            return self.base_model.sample_batched(
                history,
                n_samples=n_samples,
                n_steps=n_steps,
                chunk_size=chunk_size,
                history_is_normalized=history_is_normalized,
                **kwargs,
            )

        history_01 = denormalize_iv(history) if history_is_normalized else history
        batch_size, history_len = history_01.shape[:2]
        path_history = (
            history_01.unsqueeze(1)
            .expand(batch_size, n_samples, history_len, 5, 5)
            .reshape(batch_size * n_samples, history_len, 5, 5)
            .clone()
        )

        blocks: list[torch.Tensor] = []
        remaining = int(n_steps)
        while remaining > 0:
            block_steps = min(self.max_block_steps, remaining)
            block = self.base_model.sample_batched(
                path_history,
                n_samples=1,
                n_steps=block_steps,
                chunk_size=1,
                history_is_normalized=False,
                **kwargs,
            ).squeeze(1)
            blocks.append(block)
            path_history = torch.cat([path_history, block], dim=1)[:, -history_len:]
            remaining -= block_steps

        return torch.cat(blocks, dim=1).view(batch_size, n_samples, n_steps, 5, 5)


def build_manifest(
    *,
    model_type: str,
    checkpoint: str,
    data_path: str,
    history_start_index: int,
    history_end_index: int,
    history_len: int,
    future_len: int,
    samples: int,
    candidate_count: int,
    seed: int,
    scenario_shape: tuple[int, ...],
    path_mean_iv: np.ndarray,
    scenario_diagnostics: dict[str, float] | None = None,
    output_npz: str | None = None,
) -> dict[str, Any]:
    labels = severity_bucket_labels(samples)
    unique, counts = np.unique(labels, return_counts=True)
    path_mean_iv = np.asarray(path_mean_iv, dtype=np.float64)
    return {
        "system": "568a_564a_risk_scenario_deck",
        "base_law": "510a",
        "model_type": model_type,
        "checkpoint": checkpoint,
        "data_path": data_path,
        "history_start_index": int(history_start_index),
        "history_end_index": int(history_end_index),
        "history_len": int(history_len),
        "future_len": int(future_len),
        "samples": int(samples),
        "candidate_count": int(candidate_count),
        "seed": int(seed),
        "policy": "severity_stratified_selection",
        "severity_metric": "average_future_iv_level",
        "probability_interpretation": "stress_scenario_set_not_calibrated_law",
        "scenario_shape": [int(x) for x in scenario_shape],
        "bucket_counts": {str(k): int(v) for k, v in zip(unique, counts, strict=True)},
        "path_mean_iv": {
            "min": _rounded_float(path_mean_iv.min()),
            "median": _rounded_float(np.median(path_mean_iv)),
            "max": _rounded_float(path_mean_iv.max()),
        },
        "scenario_diagnostics": scenario_diagnostics,
        "output_npz": output_npz,
        "deployment_boundary": {
            "acceptable_use": "IV-surface stress exploration and risk challenge scenarios",
            "not_acceptable_use": "calibrated probability forecasting or capital model approval",
        },
    }


def _load_history(data_path: str, history_len: int, history_end_index: int | None) -> tuple[np.ndarray, int, int]:
    raw = np.load(data_path)
    surfaces = raw["surface"].astype(np.float32)
    if surfaces.ndim != 3 or surfaces.shape[1:] != (5, 5):
        raise ValueError("data surface must have shape (time, 5, 5)")
    end = int(history_end_index) if history_end_index is not None else int(surfaces.shape[0])
    start = end - int(history_len)
    if start < 0 or end > surfaces.shape[0]:
        raise ValueError(
            f"history_end_index must allow a {history_len}-day history inside [0, {surfaces.shape[0]}]"
        )
    return surfaces[start:end], start, end


@torch.no_grad()
def generate_deck(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    history_01, history_start, history_end = _load_history(
        args.data_path,
        args.history_len,
        args.history_end_index,
    )
    history_tensor = torch.from_numpy(history_01).unsqueeze(0).to(device)
    history_norm = normalize_iv(history_tensor)

    base_model, _payload = load_one_day_kernel(args.model_type, args.checkpoint, device)
    max_native_steps = int(getattr(getattr(base_model, "cfg", None), "future_len", args.future_len))
    if args.future_len > max_native_steps:
        base_model = BlockwiseLongHorizonModel(base_model, max_block_steps=max_native_steps).eval()
    policy_model = StressSelectedScenarioModel(
        base_model,
        candidate_count=args.candidate_count,
        chunk_size=args.chunk_size,
    ).eval()
    scenarios = policy_model.sample_batched(
        history_norm,
        n_samples=args.samples,
        n_steps=args.future_len,
        chunk_size=args.chunk_size,
        history_is_normalized=True,
    )
    scenarios_np = scenarios.squeeze(0).detach().cpu().numpy().astype(np.float32)
    labels = severity_bucket_labels(args.samples)
    path_mean_iv = scenarios_np.mean(axis=(1, 2, 3))

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        history=history_01.astype(np.float32),
        scenarios=scenarios_np,
        bucket_labels=labels,
        path_mean_iv=path_mean_iv.astype(np.float32),
    )

    manifest = build_manifest(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        data_path=args.data_path,
        history_start_index=history_start,
        history_end_index=history_end,
        history_len=args.history_len,
        future_len=args.future_len,
        samples=args.samples,
        candidate_count=args.candidate_count,
        seed=args.seed,
        scenario_shape=tuple(scenarios_np.shape),
        path_mean_iv=path_mean_iv,
        scenario_diagnostics=scenario_diagnostics(scenarios_np),
        output_npz=str(output_npz),
    )
    output_manifest = Path(args.output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 564a-style risk-manager IV stress scenario deck from the 510a base law."
    )
    parser.add_argument("--model_type", default="340c")
    parser.add_argument(
        "--checkpoint",
        default="models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt",
    )
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--history_end_index", type=int, default=None)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--candidate_count", type=int, default=192)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=568)
    parser.add_argument(
        "--output_npz",
        default="results/autoresearch/568a_564a_risk_scenario_deck/scenario_deck.npz",
    )
    parser.add_argument(
        "--output_manifest",
        default="results/autoresearch/568a_564a_risk_scenario_deck/manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    manifest = generate_deck(parse_args())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
