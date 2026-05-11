from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, ".")

from experiments.world.evaluation.masked_multiview_data import (  # noqa: E402
    MaskedMultiviewBatch,
    apply_synthetic_mask,
    build_masked_multiview_batch,
)
from experiments.world.evaluation.part1_metrics import (  # noqa: E402
    latent_prediction_metrics,
    representation_health_metrics,
    retrieval_metrics,
)
from experiments.world.evaluation.world_data import build_iv_world_windows  # noqa: E402
from experiments.world.part1_jepa_latent.context_probe_audit import (  # noqa: E402
    ridge_probe_predict,
)
from experiments.world.part1_jepa_latent.jepa_smoke import update_ema  # noqa: E402
from experiments.world.part1_jepa_latent.masked_multiview_barlow_probe_audit import (  # noqa: E402
    concatenate_feature_blocks,
    regression_metrics,
)
from experiments.world.part1_jepa_latent.masked_multiview_jepa_smoke import (  # noqa: E402
    make_masked_view_features,
    torch_barlow_cross_correlation_loss,
)


EMA_DECAY = 0.99
RAW_FEATURE = "raw_surface_last"
LEARNED_FEATURE = "temporal_jepa_clean_last"
RAW_PLUS_FEATURE = "raw_surface_last_plus_temporal_jepa_clean_last"


@dataclass(frozen=True)
class TemporalBlockJepaConfig:
    token_dim: int = 58
    input_dim: int = 174
    hidden_dim: int = 128
    latent_dim: int = 64
    predictor_hidden_dim: int = 128


@dataclass(frozen=True)
class TemporalBlockArrays:
    context_values: np.ndarray
    clean_values: np.ndarray
    observed_mask: np.ndarray
    context_mask: np.ndarray
    target_time_mask: np.ndarray

    @property
    def target_time_rows(self) -> int:
        return int(np.sum(self.target_time_mask))


class TemporalBlockSequenceEncoder(nn.Module):
    def __init__(self, cfg: TemporalBlockJepaConfig):
        super().__init__()
        self.gru = nn.GRU(cfg.input_dim, cfg.hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        seq, _hidden = self.gru(features)
        return self.head(seq)


def _select_time_rows(sequence: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
    if sequence.ndim != 3:
        raise ValueError(
            f"sequence must have shape (B, T, D), got {tuple(sequence.shape)}"
        )
    if time_mask.ndim != 2:
        raise ValueError(
            f"time_mask must have shape (B, T), got {tuple(time_mask.shape)}"
        )
    if sequence.shape[:2] != time_mask.shape:
        raise ValueError(
            "sequence and time_mask must share batch/time dimensions, "
            f"got {tuple(sequence.shape[:2])} and {tuple(time_mask.shape)}"
        )
    selected = sequence[time_mask.bool()]
    if selected.shape[0] < 2:
        raise ValueError("Need at least two target time rows")
    return selected


class TemporalBlockJepaModel(nn.Module):
    def __init__(self, cfg: TemporalBlockJepaConfig):
        super().__init__()
        self.cfg = cfg
        self.context_encoder = TemporalBlockSequenceEncoder(cfg)
        self.target_encoder = TemporalBlockSequenceEncoder(cfg)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.predictor = nn.Sequential(
            nn.LayerNorm(cfg.latent_dim),
            nn.Linear(cfg.latent_dim, cfg.predictor_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.predictor_hidden_dim, cfg.latent_dim),
        )

    def forward(
        self,
        context_features: torch.Tensor,
        clean_features: torch.Tensor,
        target_time_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        context_seq = self.context_encoder(context_features)
        with torch.no_grad():
            target_seq = self.target_encoder(clean_features)
        context_rows = _select_time_rows(context_seq, target_time_mask)
        target_rows = _select_time_rows(target_seq, target_time_mask).detach()
        return {
            "context_seq": context_seq,
            "target_seq": target_seq,
            "context_rows": context_rows,
            "predicted_target_rows": self.predictor(context_rows),
            "target_rows": target_rows,
        }


def temporal_block_jepa_loss(
    outputs: dict[str, torch.Tensor],
    *,
    barlow_weight: float = 0.05,
    offdiag_weight: float = 0.005,
) -> tuple[torch.Tensor, dict[str, float]]:
    predicted = outputs["predicted_target_rows"]
    context = outputs["context_rows"]
    target = outputs["target_rows"].detach()
    if predicted.shape != target.shape or context.shape != target.shape:
        raise ValueError("predicted, context, and target rows must share shape")
    alignment = F.mse_loss(predicted, target)
    representation_barlow, parts = torch_barlow_cross_correlation_loss(
        context[:, None, :],
        target[:, None, :],
        offdiag_weight=offdiag_weight,
    )
    loss = alignment + barlow_weight * representation_barlow
    return loss, {
        "alignment": float(alignment.detach().cpu()),
        "representation_barlow": float(representation_barlow.detach().cpu()),
        **parts,
        "loss": float(loss.detach().cpu()),
        "target_time_rows": int(predicted.shape[0]),
    }


def temporal_block_parameter_groups(
    model: TemporalBlockJepaModel,
) -> Iterator[nn.Parameter]:
    yield from model.context_encoder.parameters()
    yield from model.predictor.parameters()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_temporal_block_arrays(
    batch: MaskedMultiviewBatch,
    *,
    target_len: int,
) -> TemporalBlockArrays:
    history_len = int(batch.clean_values.shape[1])
    if target_len <= 0 or target_len >= history_len:
        raise ValueError("target_len must be positive and smaller than history_len")
    target_time_mask = np.zeros(batch.clean_values.shape[:2], dtype=bool)
    target_time_mask[:, history_len - target_len :] = True
    context_mask = np.asarray(batch.observed_mask, dtype=bool).copy()
    context_mask[:, history_len - target_len :, :] = False
    context_values = apply_synthetic_mask(
        batch.clean_values,
        batch.observed_mask,
        context_mask,
    )
    return TemporalBlockArrays(
        context_values=context_values,
        clean_values=batch.clean_values.astype(np.float32),
        observed_mask=batch.observed_mask.astype(bool),
        context_mask=context_mask,
        target_time_mask=target_time_mask,
    )


def _loader_from_arrays(
    arrays: TemporalBlockArrays,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(arrays.context_values),
        torch.from_numpy(arrays.clean_values),
        torch.from_numpy(arrays.observed_mask),
        torch.from_numpy(arrays.context_mask),
        torch.from_numpy(arrays.target_time_mask),
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle
    )


def _features(
    values: torch.Tensor,
    observed: torch.Tensor,
    visible: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    return make_masked_view_features(
        values.to(device),
        observed.to(device),
        visible.to(device),
    )


def _run_epoch(
    model: TemporalBlockJepaModel,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    barlow_weight: float,
    offdiag_weight: float,
    grad_clip: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    rows: dict[str, list[float]] = {}
    for (
        context_values,
        clean_values,
        observed,
        context_mask,
        target_time_mask,
    ) in loader:
        context_features = _features(
            context_values,
            observed,
            context_mask,
            device=device,
        )
        clean_features = _features(
            clean_values,
            observed,
            observed,
            device=device,
        )
        target_time_mask_device = target_time_mask.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(
                context_features,
                clean_features,
                target_time_mask_device,
            )
            loss, parts = temporal_block_jepa_loss(
                outputs,
                barlow_weight=barlow_weight,
                offdiag_weight=offdiag_weight,
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(temporal_block_parameter_groups(model)),
                    grad_clip,
                )
                optimizer.step()
                update_ema(model.context_encoder, model.target_encoder, EMA_DECAY)
        for key, value in parts.items():
            rows.setdefault(key, []).append(float(value))
    return {key: float(np.mean(values)) for key, values in rows.items()}


@torch.no_grad()
def _collect_eval_rows(
    model: TemporalBlockJepaModel,
    arrays: TemporalBlockArrays,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    loader = _loader_from_arrays(arrays, batch_size=batch_size, shuffle=False)
    rows: dict[str, list[np.ndarray]] = {
        "context": [],
        "predicted": [],
        "target": [],
    }
    for (
        context_values,
        clean_values,
        observed,
        context_mask,
        target_time_mask,
    ) in loader:
        context_features = _features(
            context_values,
            observed,
            context_mask,
            device=device,
        )
        clean_features = _features(
            clean_values,
            observed,
            observed,
            device=device,
        )
        outputs = model(context_features, clean_features, target_time_mask.to(device))
        rows["context"].append(outputs["context_rows"].detach().cpu().numpy())
        rows["predicted"].append(
            outputs["predicted_target_rows"].detach().cpu().numpy()
        )
        rows["target"].append(outputs["target_rows"].detach().cpu().numpy())
    return {key: np.concatenate(chunks, axis=0) for key, chunks in rows.items()}


def _evaluate(
    model: TemporalBlockJepaModel,
    arrays: TemporalBlockArrays,
    *,
    batch_size: int,
    device: torch.device,
    barlow_weight: float,
    offdiag_weight: float,
    retrieval_eval_rows: int,
) -> dict[str, Any]:
    loss_parts = _run_epoch(
        model,
        _loader_from_arrays(arrays, batch_size=batch_size, shuffle=False),
        device=device,
        optimizer=None,
        barlow_weight=barlow_weight,
        offdiag_weight=offdiag_weight,
        grad_clip=1.0,
    )
    encoded = _collect_eval_rows(
        model,
        arrays,
        batch_size=batch_size,
        device=device,
    )
    limit = min(int(retrieval_eval_rows), encoded["predicted"].shape[0])
    return {
        **loss_parts,
        "prediction_metrics": latent_prediction_metrics(
            encoded["predicted"],
            encoded["target"],
        ),
        "context_target_metrics": latent_prediction_metrics(
            encoded["context"],
            encoded["target"],
        ),
        "context_health": representation_health_metrics(encoded["context"]),
        "predicted_health": representation_health_metrics(encoded["predicted"]),
        "target_health": representation_health_metrics(encoded["target"]),
        "retrieval_subset_rows": int(limit),
        "predicted_retrieval_subset": retrieval_metrics(
            encoded["predicted"][:limit],
            encoded["target"][:limit],
            top_k=(1, 5, 10),
        ),
        "context_retrieval_subset": retrieval_metrics(
            encoded["context"][:limit],
            encoded["target"][:limit],
            top_k=(1, 5, 10),
        ),
    }


@torch.no_grad()
def encode_clean_temporal_context(
    model: TemporalBlockJepaModel,
    batch: MaskedMultiviewBatch,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    dataset = TensorDataset(
        torch.from_numpy(batch.clean_values),
        torch.from_numpy(batch.observed_mask),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    rows: list[np.ndarray] = []
    for clean_values, observed in loader:
        features = _features(clean_values, observed, observed, device=device)
        rows.append(model.context_encoder(features).detach().cpu().numpy())
    return np.concatenate(rows, axis=0)


def _current_target_groups(batch: MaskedMultiviewBatch) -> dict[str, np.ndarray]:
    last = np.asarray(batch.clean_values[:, -1, :], dtype=np.float32)
    meta = batch.token_metadata
    groups = {
        "iv_surface": meta.geometry_id == "iv_surface",
        "vol_side_channel": meta.geometry_id == "vol_side_channel",
        "factor_level": meta.geometry_id == "factor_level",
        "factor_return": meta.geometry_id == "factor_return",
        "all_geometry": np.ones(meta.n_tokens, dtype=bool),
    }
    return {name: last[:, mask].astype(np.float32) for name, mask in groups.items()}


def _future_summary_targets(
    past_surface: np.ndarray,
    future_surface: np.ndarray,
) -> dict[str, np.ndarray]:
    past = np.asarray(past_surface, dtype=np.float32)
    future = np.asarray(future_surface, dtype=np.float32)
    if past.ndim != 3 or future.ndim != 3:
        raise ValueError("past_surface and future_surface must have shape (N, T, C)")
    if past.shape[0] != future.shape[0] or past.shape[2] != future.shape[2]:
        raise ValueError("past and future must share sample count and channel count")
    last = past[:, -1, :]
    return {
        "future_mean_delta": (future.mean(axis=1) - last).astype(np.float32),
        "future_range": (future.max(axis=1) - future.min(axis=1)).astype(np.float32),
    }


def _probe_feature_sets(
    train_features: dict[str, np.ndarray],
    val_features: dict[str, np.ndarray],
    train_targets: dict[str, np.ndarray],
    val_targets: dict[str, np.ndarray],
    *,
    alpha: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for feature_name, train_x in train_features.items():
        val_x = val_features[feature_name]
        target_rows: dict[str, Any] = {}
        for target_name, train_y in train_targets.items():
            pred = ridge_probe_predict(
                train_x,
                train_y,
                val_x,
                alpha=alpha,
            )
            target_rows[target_name] = regression_metrics(
                pred, val_targets[target_name]
            )
        out[feature_name] = {
            "feature_shape": {
                "train": list(train_x.shape),
                "val": list(val_x.shape),
            },
            "health": representation_health_metrics(val_x),
            "targets": target_rows,
        }
    return out


def _mse(probe_metrics: dict[str, Any], feature: str, target: str) -> float:
    return float(probe_metrics[feature]["targets"][target]["mse"])


def _current_state_guardrail(probe_metrics: dict[str, Any]) -> dict[str, float | str]:
    raw_iv = _mse(probe_metrics, RAW_FEATURE, "iv_surface")
    learned_iv = _mse(probe_metrics, LEARNED_FEATURE, "iv_surface")
    raw_plus_iv = _mse(probe_metrics, RAW_PLUS_FEATURE, "iv_surface")
    return {
        "status": "PASS" if raw_plus_iv <= raw_iv else "FAIL",
        "raw_only_iv_mse": raw_iv,
        "learned_only_iv_mse": learned_iv,
        "raw_plus_learned_iv_mse": raw_plus_iv,
        "learned_to_raw_ratio": learned_iv / raw_iv,
        "raw_plus_to_raw_ratio": raw_plus_iv / raw_iv,
        "raw_plus_delta_vs_raw": raw_plus_iv - raw_iv,
    }


def _future_probe_summary(probe_metrics: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for target in ("future_mean_delta", "future_range"):
        raw = _mse(probe_metrics, RAW_FEATURE, target)
        learned = _mse(probe_metrics, LEARNED_FEATURE, target)
        raw_plus = _mse(probe_metrics, RAW_PLUS_FEATURE, target)
        out[target] = {
            "raw_only_mse": raw,
            "learned_only_mse": learned,
            "raw_plus_learned_mse": raw_plus,
            "learned_to_raw_ratio": learned / raw,
            "raw_plus_to_raw_ratio": raw_plus / raw,
            "raw_plus_delta_vs_raw": raw_plus - raw,
        }
    return out


def _serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_serializable(v) for v in obj]
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def run_temporal_block_jepa_smoke(args: argparse.Namespace) -> dict[str, Any]:
    _set_seed(int(args.seed))
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    train = build_masked_multiview_batch(
        split="train",
        history_len=args.history_len,
        future_len=args.future_len,
        max_windows=args.max_train_windows,
        seed=args.seed,
        normalize=True,
    )
    val = build_masked_multiview_batch(
        split="val",
        history_len=args.history_len,
        future_len=args.future_len,
        max_windows=args.max_val_windows,
        seed=args.seed + 1000,
        normalize=True,
    )
    train_arrays = _build_temporal_block_arrays(train, target_len=args.target_len)
    val_arrays = _build_temporal_block_arrays(val, target_len=args.target_len)
    cfg = TemporalBlockJepaConfig(
        token_dim=train.token_metadata.n_tokens,
        input_dim=train.token_metadata.n_tokens * 3,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        predictor_hidden_dim=args.predictor_hidden_dim,
    )
    model = TemporalBlockJepaModel(cfg).to(device)
    optimizer = torch.optim.AdamW(
        temporal_block_parameter_groups(model),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    initial_val = _evaluate(
        model,
        val_arrays,
        batch_size=args.batch_size,
        device=device,
        barlow_weight=args.barlow_weight,
        offdiag_weight=args.offdiag_weight,
        retrieval_eval_rows=args.retrieval_eval_rows,
    )
    loader = _loader_from_arrays(
        train_arrays,
        batch_size=args.batch_size,
        shuffle=True,
    )
    train_history = []
    for epoch in range(int(args.epochs)):
        parts = _run_epoch(
            model,
            loader,
            device=device,
            optimizer=optimizer,
            barlow_weight=args.barlow_weight,
            offdiag_weight=args.offdiag_weight,
            grad_clip=args.grad_clip,
        )
        train_history.append({"epoch": epoch + 1, **parts})
    final_val = _evaluate(
        model,
        val_arrays,
        batch_size=args.batch_size,
        device=device,
        barlow_weight=args.barlow_weight,
        offdiag_weight=args.offdiag_weight,
        retrieval_eval_rows=args.retrieval_eval_rows,
    )

    train_encoded = encode_clean_temporal_context(
        model,
        train,
        batch_size=args.batch_size,
        device=device,
    )
    val_encoded = encode_clean_temporal_context(
        model,
        val,
        batch_size=args.batch_size,
        device=device,
    )
    train_iv = build_iv_world_windows(
        split="train",
        history_len=args.history_len,
        future_len=args.future_len,
        max_windows=args.max_train_windows,
        normalize=True,
    )
    val_iv = build_iv_world_windows(
        split="val",
        history_len=args.history_len,
        future_len=args.future_len,
        max_windows=args.max_val_windows,
        normalize=True,
    )
    train_learned_last = train_encoded[:, -1, :]
    val_learned_last = val_encoded[:, -1, :]
    train_raw = train_iv.past_window[:, -1, :]
    val_raw = val_iv.past_window[:, -1, :]
    train_features = {
        RAW_FEATURE: train_raw,
        LEARNED_FEATURE: train_learned_last,
        RAW_PLUS_FEATURE: concatenate_feature_blocks(train_raw, train_learned_last),
    }
    val_features = {
        RAW_FEATURE: val_raw,
        LEARNED_FEATURE: val_learned_last,
        RAW_PLUS_FEATURE: concatenate_feature_blocks(val_raw, val_learned_last),
    }
    current_probe_metrics = _probe_feature_sets(
        train_features,
        val_features,
        _current_target_groups(train),
        _current_target_groups(val),
        alpha=args.ridge_alpha,
    )
    future_probe_metrics = _probe_feature_sets(
        train_features,
        val_features,
        _future_summary_targets(train_iv.past_window, train_iv.future_window),
        _future_summary_targets(val_iv.past_window, val_iv.future_window),
        alpha=args.ridge_alpha,
    )

    result: dict[str, Any] = {
        "analysis": "world_model_temporal_block_jepa_smoke",
        "date": "2026-05-11",
        "objective_family": "context_to_target_jepa_temporal_diagnostic",
        "objective_family_base": "context_to_target_jepa",
        "literature_status": "canonical_jepa_temporal_holdout_diagnostic_not_active_reference",
        "uses_future_targets": False,
        "uses_future_window_targets": False,
        "uses_value_reconstruction": False,
        "uses_decoder": False,
        "target_block": {
            "position": "last_history_block",
            "target_len": int(args.target_len),
            "context_visible_before_block": int(args.history_len - args.target_len),
        },
        "ema_decay": EMA_DECAY,
        "device": str(device),
        "config": asdict(cfg),
        "train_windows": int(train.clean_values.shape[0]),
        "val_windows": int(val.clean_values.shape[0]),
        "train_target_time_rows": train_arrays.target_time_rows,
        "val_target_time_rows": val_arrays.target_time_rows,
        "initial_val_loss": float(initial_val["loss"]),
        "final_val_loss": float(final_val["loss"]),
        "loss_delta": float(final_val["loss"] - initial_val["loss"]),
        "train_loss_first": float(train_history[0]["loss"]) if train_history else None,
        "train_loss_last": float(train_history[-1]["loss"]) if train_history else None,
        "train_history": train_history,
        "val_alignment": float(final_val["alignment"]),
        "val_representation_barlow": float(final_val["representation_barlow"]),
        "val_prediction_metrics": final_val["prediction_metrics"],
        "val_context_target_metrics": final_val["context_target_metrics"],
        "val_context_health": final_val["context_health"],
        "val_predicted_health": final_val["predicted_health"],
        "val_target_health": final_val["target_health"],
        "val_retrieval_subset_rows": int(final_val["retrieval_subset_rows"]),
        "val_predicted_retrieval_subset": final_val["predicted_retrieval_subset"],
        "val_context_retrieval_subset": final_val["context_retrieval_subset"],
        "current_state_probe_metrics": current_probe_metrics,
        "current_state_guardrail": _current_state_guardrail(current_probe_metrics),
        "future_probe_metrics": future_probe_metrics,
        "future_probe_summary": _future_probe_summary(future_probe_metrics),
        "promotion_decision": "SMOKE_ONLY_DO_NOT_PROMOTE",
        "part_b_blocked": True,
    }
    output_json = getattr(args, "output_json", None)
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(
            json.dumps(_serializable(result), indent=2) + "\n",
            encoding="utf-8",
        )
    checkpoint = getattr(args, "checkpoint", None)
    if checkpoint:
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(cfg),
                "model_state_dict": model.state_dict(),
                "result": _serializable(result),
            },
            checkpoint,
        )
    report_md = getattr(args, "report_md", None)
    if report_md:
        Path(report_md).parent.mkdir(parents=True, exist_ok=True)
        Path(report_md).write_text(render_markdown(result), encoding="utf-8")
    return result


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def render_markdown(result: dict[str, Any]) -> str:
    pred = result["val_prediction_metrics"]
    context = result["val_context_target_metrics"]
    retrieval = result["val_predicted_retrieval_subset"]
    context_retrieval = result["val_context_retrieval_subset"]
    guardrail = result["current_state_guardrail"]
    future = result["future_probe_summary"]
    lines = [
        "# World Model HEAD172: Temporal Block JEPA Smoke",
        "",
        "Date: 2026-05-11",
        "",
        "## Iteration Type",
        "",
        "`experiment`",
        "",
        "## Objective Family",
        "",
        "`context_to_target_jepa_temporal_diagnostic`; smoke-only canonical JEPA",
        "diagnostic. This is not the active Part 1 reference.",
        "",
        "## Literature Status",
        "",
        "`canonical_jepa_temporal_holdout_diagnostic_not_active_reference`: the",
        "context encoder sees a same-history temporal block hidden by the synthetic",
        "mask channel; the target encoder sees the clean same-window history; the",
        "loss aligns latent rows only. No future-window values, decoder, or raw",
        "reconstruction target is used.",
        "",
        "## Hypothesis",
        "",
        "A minimal temporal holdout JEPA can reveal whether a canonical",
        "context-to-target latent objective adds useful abstract state beyond raw",
        "surface baselines without becoming the active pretraining route.",
        "",
        "## Falsifier",
        "",
        "The diagnostic remains smoke-only if latent rows show weak retrieval/rank or",
        "if frozen raw-plus-learned probes fail to improve raw-only guardrails.",
        "",
        "## Run",
        "",
        f"- Train windows: `{result['train_windows']}`.",
        f"- Validation windows: `{result['val_windows']}`.",
        f"- Target time rows: `{result['train_target_time_rows']}` train, `{result['val_target_time_rows']}` validation.",
        f"- Target block: last `{result['target_block']['target_len']}` history days.",
        f"- EMA decay: `{_fmt(result['ema_decay'])}`.",
        "",
        "## Latent Metrics",
        "",
        f"- Initial validation loss: `{_fmt(result['initial_val_loss'])}`.",
        f"- Final validation loss: `{_fmt(result['final_val_loss'])}`.",
        f"- Validation alignment MSE: `{_fmt(result['val_alignment'])}`.",
        f"- Predicted-target cosine mean: `{_fmt(pred['cosine_mean'])}`.",
        f"- Context-target cosine mean: `{_fmt(context['cosine_mean'])}`.",
        f"- Predicted retrieval top10: `{_fmt(retrieval['top10'])}`.",
        f"- Context retrieval top10: `{_fmt(context_retrieval['top10'])}`.",
        f"- Context effective rank: `{_fmt(result['val_context_health']['effective_rank'])}`.",
        f"- Target effective rank: `{_fmt(result['val_target_health']['effective_rank'])}`.",
        "",
        "## Frozen Probe Guardrails",
        "",
        "| surface | raw-only MSE | learned-only MSE | raw+learned MSE | raw+learned/raw |",
        "| --- | ---: | ---: | ---: | ---: |",
        "| current IV state | {raw} | {learned} | {raw_plus} | {ratio} |".format(
            raw=_fmt(guardrail["raw_only_iv_mse"]),
            learned=_fmt(guardrail["learned_only_iv_mse"]),
            raw_plus=_fmt(guardrail["raw_plus_learned_iv_mse"]),
            ratio=_fmt(guardrail["raw_plus_to_raw_ratio"]),
        ),
    ]
    for target, row in future.items():
        lines.append(
            "| {target} | {raw} | {learned} | {raw_plus} | {ratio} |".format(
                target=target,
                raw=_fmt(row["raw_only_mse"]),
                learned=_fmt(row["learned_only_mse"]),
                raw_plus=_fmt(row["raw_plus_learned_mse"]),
                ratio=_fmt(row["raw_plus_to_raw_ratio"]),
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Current-state guardrail: `{guardrail['status']}`.",
            f"- Promotion decision: `{result['promotion_decision']}`.",
            f"- Part B blocked: `{result['part_b_blocked']}`.",
            "",
            "This diagnostic does not promote Part 1. The active reference remains the",
            "masked-multiview Barlow branch until a learned representation passes the",
            "raw-plus-learned guardrails and downstream probe comparisons.",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Temporal block JEPA smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--target_len", type=int, default=5)
    parser.add_argument("--max_train_windows", type=int, default=128)
    parser.add_argument("--max_val_windows", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2172)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--barlow_weight", type=float, default=0.05)
    parser.add_argument("--offdiag_weight", type=float, default=0.005)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--predictor_hidden_dim", type=int, default=64)
    parser.add_argument("--ridge_alpha", type=float, default=10.0)
    parser.add_argument("--retrieval_eval_rows", type=int, default=512)
    parser.add_argument(
        "--output_json",
        default="results/world/temporal_block_jepa_smoke_head172.json",
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument(
        "--report_md",
        default="experiments/world/reports/world_model_head172_temporal_block_jepa_smoke.md",
    )
    return parser


def main() -> None:
    result = run_temporal_block_jepa_smoke(_build_parser().parse_args())
    print(
        json.dumps(
            {
                "initial_val_loss": result["initial_val_loss"],
                "final_val_loss": result["final_val_loss"],
                "val_alignment": result["val_alignment"],
                "current_state_guardrail": result["current_state_guardrail"],
                "future_probe_summary": result["future_probe_summary"],
                "promotion_decision": result["promotion_decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
