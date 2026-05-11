from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, ".")

from experiments.world.evaluation.masked_multiview_data import (  # noqa: E402
    MaskedMultiviewBatch,
    build_masked_multiview_batch,
)
from experiments.world.evaluation.part1_metrics import (  # noqa: E402
    representation_health_metrics,
)
from experiments.world.evaluation.world_data import build_iv_world_windows  # noqa: E402
from experiments.world.part1_jepa_latent.context_probe_audit import (  # noqa: E402
    ridge_probe_predict,
)
from experiments.world.part1_jepa_latent.masked_multiview_barlow_probe_audit import (  # noqa: E402
    concatenate_feature_blocks,
    encode_clean_masked_windows,
    load_direct_barlow_checkpoint,
    regression_metrics,
)
from experiments.world.part1_jepa_latent.masked_multiview_downstream_probe_audit import (  # noqa: E402
    make_extended_future_targets,
)
from experiments.world.part1_jepa_latent.masked_multiview_mask_artifact_audit import (  # noqa: E402
    classification_metrics,
    fit_predict_multiclass_ridge,
)
from experiments.world.part1_jepa_latent.temporal_block_jepa_smoke import (  # noqa: E402
    TemporalBlockJepaConfig,
    TemporalBlockJepaModel,
    _build_temporal_block_arrays,
    _loader_from_arrays,
    _run_epoch,
    _set_seed,
    encode_clean_temporal_context,
    temporal_block_parameter_groups,
)


RAW_LAST = "raw_surface_last"
RAW_FLAT = "raw_surface_flat"
TEMPORAL = "temporal_jepa_last"
TEMPORAL_PLUS = "raw_surface_last_plus_temporal_jepa_last"
RANDOM = "random_temporal_last"
RANDOM_PLUS = "raw_surface_last_plus_random_temporal_last"
SCALE = "scale_barlow_last"
SCALE_PLUS = "raw_surface_last_plus_scale_barlow_last"
CURRENT_TARGETS = (
    "iv_surface",
    "vol_side_channel",
    "factor_level",
    "factor_return",
    "all_geometry",
)
FUTURE_TARGETS = (
    "future_mean_delta",
    "future_range",
    "future_terminal_delta",
    "future_max_abs_step",
    "future_drawdown",
)
DEFAULT_SCALE_CHECKPOINT = (
    "models/world/checkpoints/part1_jepa_latent/"
    "masked_multiview_barlow_scale_head127.pt"
)


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


def _probe_regression(
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
        rows: dict[str, Any] = {}
        for target_name, train_y in train_targets.items():
            pred = ridge_probe_predict(train_x, train_y, val_x, alpha=alpha)
            rows[target_name] = regression_metrics(pred, val_targets[target_name])
        out[feature_name] = {
            "feature_shape": {
                "train": list(train_x.shape),
                "val": list(val_x.shape),
            },
            "health": representation_health_metrics(val_x),
            "targets": rows,
        }
    return out


def _probe_regime(
    train_features: dict[str, np.ndarray],
    val_features: dict[str, np.ndarray],
    train_labels: np.ndarray | None,
    val_labels: np.ndarray | None,
    *,
    alpha: float,
) -> dict[str, Any]:
    if train_labels is None or val_labels is None:
        return {"status": "skipped_missing_regime_labels"}
    out: dict[str, Any] = {"status": "ok", "features": {}}
    for feature_name, train_x in train_features.items():
        try:
            pred = fit_predict_multiclass_ridge(
                train_x,
                train_labels,
                val_features[feature_name],
                alpha=alpha,
            )
            out["features"][feature_name] = classification_metrics(pred, val_labels)
        except ValueError as exc:
            out["features"][feature_name] = {
                "status": "skipped",
                "reason": str(exc),
            }
    return out


def _mse(probe: dict[str, Any], feature: str, target: str) -> float:
    return float(probe[feature]["targets"][target]["mse"])


def _feature_mse_row(
    probe: dict[str, Any],
    *,
    target: str,
    features: list[str],
) -> dict[str, float | str]:
    raw = _mse(probe, RAW_LAST, target)
    best_raw = min(_mse(probe, RAW_LAST, target), _mse(probe, RAW_FLAT, target))
    row: dict[str, float | str] = {
        "target": target,
        "raw_surface_last_mse": raw,
        "raw_surface_flat_mse": _mse(probe, RAW_FLAT, target),
        "best_raw_mse": best_raw,
    }
    for feature in features:
        mse = _mse(probe, feature, target)
        row[f"{feature}_mse"] = mse
        row[f"{feature}_to_raw_last_ratio"] = mse / raw
        row[f"{feature}_to_best_raw_ratio"] = mse / best_raw
    return row


def _future_summary(
    future_probe: dict[str, Any],
    *,
    features: list[str],
) -> dict[str, dict[str, float | str]]:
    return {
        target: _feature_mse_row(future_probe, target=target, features=features)
        for target in FUTURE_TARGETS
    }


def _current_summary(
    current_probe: dict[str, Any],
    *,
    features: list[str],
) -> dict[str, dict[str, float | str]]:
    return {
        target: _feature_mse_row(current_probe, target=target, features=features)
        for target in CURRENT_TARGETS
    }


def _count_improvements(
    rows: dict[str, dict[str, float | str]],
    *,
    feature: str,
) -> int:
    return int(
        sum(
            float(row[f"{feature}_mse"]) < float(row["raw_surface_last_mse"])
            for row in rows.values()
        )
    )


def _count_best_raw_wins(
    rows: dict[str, dict[str, float | str]],
    *,
    feature: str,
) -> int:
    return int(
        sum(
            float(row[f"{feature}_mse"]) < float(row["best_raw_mse"])
            for row in rows.values()
        )
    )


def _train_temporal_model(
    train: MaskedMultiviewBatch,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[TemporalBlockJepaModel, np.ndarray]:
    arrays = _build_temporal_block_arrays(train, target_len=args.target_len)
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
    loader = _loader_from_arrays(
        arrays,
        batch_size=args.batch_size,
        shuffle=True,
    )
    history = []
    for epoch in range(int(args.epochs)):
        row = _run_epoch(
            model,
            loader,
            device=device,
            optimizer=optimizer,
            barlow_weight=args.barlow_weight,
            offdiag_weight=args.offdiag_weight,
            grad_clip=args.grad_clip,
        )
        history.append({"epoch": epoch + 1, **row})
    return model, np.asarray(history, dtype=object)


def _encode_optional_scale(
    checkpoint: str | Path,
    train: MaskedMultiviewBatch,
    val: MaskedMultiviewBatch,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    if not str(checkpoint):
        return None, None, "scale checkpoint disabled"
    path = Path(checkpoint)
    if not path.exists():
        return None, None, f"missing scale checkpoint: {path}"
    model = load_direct_barlow_checkpoint(path, device=device)
    train_z = encode_clean_masked_windows(
        model,
        train,
        batch_size=batch_size,
        device=device,
    )
    val_z = encode_clean_masked_windows(
        model,
        val,
        batch_size=batch_size,
        device=device,
    )
    return train_z[:, -1, :], val_z[:, -1, :], None


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


def run_temporal_jepa_bakeoff(args: argparse.Namespace) -> dict[str, Any]:
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

    cfg = TemporalBlockJepaConfig(
        token_dim=train.token_metadata.n_tokens,
        input_dim=train.token_metadata.n_tokens * 3,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        predictor_hidden_dim=args.predictor_hidden_dim,
    )
    random_model = TemporalBlockJepaModel(cfg).to(device)
    random_train = encode_clean_temporal_context(
        random_model,
        train,
        batch_size=args.batch_size,
        device=device,
    )[:, -1, :]
    random_val = encode_clean_temporal_context(
        random_model,
        val,
        batch_size=args.batch_size,
        device=device,
    )[:, -1, :]

    _set_seed(int(args.seed))
    temporal_model, train_history = _train_temporal_model(
        train, args=args, device=device
    )
    temporal_train = encode_clean_temporal_context(
        temporal_model,
        train,
        batch_size=args.batch_size,
        device=device,
    )[:, -1, :]
    temporal_val = encode_clean_temporal_context(
        temporal_model,
        val,
        batch_size=args.batch_size,
        device=device,
    )[:, -1, :]

    scale_train, scale_val, scale_skip = _encode_optional_scale(
        args.scale_checkpoint,
        train,
        val,
        batch_size=args.batch_size,
        device=device,
    )

    raw_last_train = train_iv.past_window[:, -1, :]
    raw_last_val = val_iv.past_window[:, -1, :]
    raw_flat_train = train_iv.past_window.reshape(train_iv.past_window.shape[0], -1)
    raw_flat_val = val_iv.past_window.reshape(val_iv.past_window.shape[0], -1)
    train_features: dict[str, np.ndarray] = {
        RAW_LAST: raw_last_train,
        RAW_FLAT: raw_flat_train,
        TEMPORAL: temporal_train,
        TEMPORAL_PLUS: concatenate_feature_blocks(raw_last_train, temporal_train),
        RANDOM: random_train,
        RANDOM_PLUS: concatenate_feature_blocks(raw_last_train, random_train),
    }
    val_features: dict[str, np.ndarray] = {
        RAW_LAST: raw_last_val,
        RAW_FLAT: raw_flat_val,
        TEMPORAL: temporal_val,
        TEMPORAL_PLUS: concatenate_feature_blocks(raw_last_val, temporal_val),
        RANDOM: random_val,
        RANDOM_PLUS: concatenate_feature_blocks(raw_last_val, random_val),
    }
    skipped_feature_surfaces: dict[str, str] = {}
    if scale_train is not None and scale_val is not None:
        train_features[SCALE] = scale_train
        train_features[SCALE_PLUS] = concatenate_feature_blocks(
            raw_last_train,
            scale_train,
        )
        val_features[SCALE] = scale_val
        val_features[SCALE_PLUS] = concatenate_feature_blocks(raw_last_val, scale_val)
    else:
        skipped_feature_surfaces[SCALE] = str(scale_skip)
        skipped_feature_surfaces[SCALE_PLUS] = str(scale_skip)

    current_probe = _probe_regression(
        train_features,
        val_features,
        _current_target_groups(train),
        _current_target_groups(val),
        alpha=args.ridge_alpha,
    )
    train_future = make_extended_future_targets(
        train_iv.past_window,
        train_iv.future_window,
        regime_labels=train_iv.regime_label,
    )
    val_future = make_extended_future_targets(
        val_iv.past_window,
        val_iv.future_window,
        regime_labels=val_iv.regime_label,
    )
    future_probe = _probe_regression(
        train_features,
        val_features,
        train_future["regression"],
        val_future["regression"],
        alpha=args.ridge_alpha,
    )
    regime_probe = _probe_regime(
        train_features,
        val_features,
        train_future["classification"].get("regime_label"),
        val_future["classification"].get("regime_label"),
        alpha=args.ridge_alpha,
    )

    comparison_features = [TEMPORAL, TEMPORAL_PLUS, RANDOM, RANDOM_PLUS]
    if SCALE in train_features:
        comparison_features.extend([SCALE, SCALE_PLUS])
    current_summary = _current_summary(
        current_probe,
        features=comparison_features,
    )
    future_summary = _future_summary(
        future_probe,
        features=comparison_features,
    )
    summary_counts: dict[str, int | None | str] = {
        "temporal_raw_plus_future_improvements": _count_improvements(
            future_summary,
            feature=TEMPORAL_PLUS,
        ),
        "temporal_learned_best_raw_future_wins": _count_best_raw_wins(
            future_summary,
            feature=TEMPORAL,
        ),
        "random_raw_plus_future_improvements": _count_improvements(
            future_summary,
            feature=RANDOM_PLUS,
        ),
        "current_iv_temporal_raw_plus_status": (
            "PASS"
            if float(current_summary["iv_surface"][f"{TEMPORAL_PLUS}_mse"])
            <= float(current_summary["iv_surface"]["raw_surface_last_mse"])
            else "FAIL"
        ),
    }
    if SCALE in train_features:
        summary_counts["scale_raw_plus_future_improvements"] = _count_improvements(
            future_summary,
            feature=SCALE_PLUS,
        )
        summary_counts["scale_learned_best_raw_future_wins"] = _count_best_raw_wins(
            future_summary,
            feature=SCALE,
        )
        summary_counts["temporal_raw_plus_beats_scale_raw_plus_future_targets"] = int(
            sum(
                float(row[f"{TEMPORAL_PLUS}_mse"]) < float(row[f"{SCALE_PLUS}_mse"])
                for row in future_summary.values()
            )
        )
    else:
        summary_counts["scale_raw_plus_future_improvements"] = None
        summary_counts["scale_learned_best_raw_future_wins"] = None
        summary_counts["temporal_raw_plus_beats_scale_raw_plus_future_targets"] = None

    current_iv_row = current_summary["iv_surface"]
    temporal_current_iv_ratio = float(
        current_iv_row[f"{TEMPORAL_PLUS}_to_raw_last_ratio"]
    )
    random_current_iv_ratio = float(current_iv_row[f"{RANDOM_PLUS}_to_raw_last_ratio"])
    scale_future_improvements = summary_counts["scale_raw_plus_future_improvements"]
    temporal_beats_scale = summary_counts[
        "temporal_raw_plus_beats_scale_raw_plus_future_targets"
    ]
    random_control_beats_temporal_current_iv = (
        random_current_iv_ratio <= temporal_current_iv_ratio
    )
    temporal_future_improvements = int(
        summary_counts["temporal_raw_plus_future_improvements"]
    )
    random_future_improvements = int(
        summary_counts["random_raw_plus_future_improvements"]
    )
    temporal_context_target_status = "do_not_promote"
    if (
        summary_counts["current_iv_temporal_raw_plus_status"] == "PASS"
        and temporal_future_improvements > random_future_improvements
        and (
            scale_future_improvements is None
            or temporal_future_improvements > int(scale_future_improvements)
        )
        and not random_control_beats_temporal_current_iv
    ):
        temporal_context_target_status = "needs_larger_confirmatory_run"

    decision = {
        "temporal_context_target_status": temporal_context_target_status,
        "random_control_warning": (
            "raw_plus_random_beats_or_matches_raw_plus_temporal_on_current_iv"
            if random_control_beats_temporal_current_iv
            else "raw_plus_temporal_beats_random_control_on_current_iv"
        ),
        "future_utility_status": (
            "temporal_raw_plus_underperforms_controls"
            if temporal_future_improvements <= random_future_improvements
            or (
                scale_future_improvements is not None
                and temporal_future_improvements <= int(scale_future_improvements)
            )
            else "temporal_raw_plus_beats_available_controls"
        ),
        "current_iv_temporal_raw_plus_to_raw_ratio": temporal_current_iv_ratio,
        "current_iv_random_raw_plus_to_raw_ratio": random_current_iv_ratio,
        "temporal_raw_plus_future_improvements": temporal_future_improvements,
        "random_raw_plus_future_improvements": random_future_improvements,
        "scale_raw_plus_future_improvements": scale_future_improvements,
        "temporal_raw_plus_beats_scale_raw_plus_future_targets": temporal_beats_scale,
        "part_b_blocked": True,
        "next_step": (
            "do_not_tune_temporal_context_to_target_knobs; keep scaled Barlow as "
            "the active learned candidate and use route-decision or provenance "
            "work unless a genuinely new design gate is justified"
        ),
    }

    result: dict[str, Any] = {
        "analysis": "world_model_temporal_jepa_bakeoff",
        "date": "2026-05-11",
        "iteration": 173,
        "iteration_type": "experiment",
        "objective_family": "downstream_probe_frozen_bakeoff",
        "literature_status": "frozen_evaluation_protocol_from_ijepa_vjepa_tsjepa",
        "uses_future_targets_as_pretraining": False,
        "uses_decoder": False,
        "device": str(device),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "train_shape": list(train.clean_values.shape),
        "val_shape": list(val.clean_values.shape),
        "feature_surfaces": list(train_features.keys()),
        "skipped_feature_surfaces": skipped_feature_surfaces,
        "temporal_train_history": train_history.tolist(),
        "current_probe_rows": current_probe,
        "current_state_rows": {
            feature: row["targets"] for feature, row in current_probe.items()
        },
        "current_summary": current_summary,
        "future_probe_rows": future_probe,
        "future_summary": future_summary,
        "regime_probe": regime_probe,
        "summary_counts": summary_counts,
        "decision": decision,
        "promotion_decision": "DO_NOT_PROMOTE",
        "part_b_blocked": True,
    }
    output_json = getattr(args, "output_json", None)
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(
            json.dumps(_serializable(result), indent=2) + "\n",
            encoding="utf-8",
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
    counts = result["summary_counts"]
    current_iv = result["current_summary"]["iv_surface"]
    future = result["future_summary"]
    features = [
        TEMPORAL,
        TEMPORAL_PLUS,
        RANDOM,
        RANDOM_PLUS,
    ]
    if SCALE in result["feature_surfaces"]:
        features.extend([SCALE, SCALE_PLUS])
    lines = [
        "# World Model HEAD173: Temporal JEPA Frozen Bakeoff",
        "",
        "Date: 2026-05-11",
        "",
        "## Iteration Type",
        "",
        "`experiment`",
        "",
        "## Objective Family",
        "",
        "`downstream_probe_frozen_bakeoff`; this evaluates frozen feature",
        "surfaces and does not turn future targets into Part 1 pretraining.",
        "",
        "## Literature Status",
        "",
        "`frozen_evaluation_protocol_from_ijepa_vjepa_tsjepa`: the comparison",
        "follows the JEPA habit of judging representation quality through frozen",
        "downstream probes and baselines, not pretraining loss alone.",
        "",
        "## Hypothesis",
        "",
        "If HEAD172's temporal JEPA signal is useful, the same train/validation",
        "probe contract should show incremental raw+learned value versus raw",
        "features, random temporal features, and the scaled Barlow candidate.",
        "",
        "## Falsifier",
        "",
        "The temporal branch remains non-promotable if its raw+learned surface does",
        "not beat raw and scaled-Barlow surfaces on the same frozen probes, or if",
        "rank/health remains weak.",
        "",
        "## Feature Surfaces",
        "",
    ]
    for feature in result["feature_surfaces"]:
        lines.append(f"- `{feature}`")
    if result["skipped_feature_surfaces"]:
        lines.append("")
        lines.append("Skipped surfaces:")
        for feature, reason in result["skipped_feature_surfaces"].items():
            lines.append(f"- `{feature}`: {reason}")
    lines.extend(
        [
            "",
            "## Current IV Guardrail",
            "",
            "| feature | MSE | ratio to raw-last | ratio to best raw |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for feature in features:
        lines.append(
            "| {feature} | {mse} | {raw_ratio} | {best_ratio} |".format(
                feature=feature,
                mse=_fmt(current_iv[f"{feature}_mse"]),
                raw_ratio=_fmt(current_iv[f"{feature}_to_raw_last_ratio"]),
                best_ratio=_fmt(current_iv[f"{feature}_to_best_raw_ratio"]),
            )
        )
    lines.extend(
        [
            "",
            "## Future Probe Summary",
            "",
            "| target | temporal+raw/raw | random+raw/raw | scale+raw/raw | temporal learned/best raw | scale learned/best raw |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for target in FUTURE_TARGETS:
        row = future[target]
        lines.append(
            "| {target} | {tmp} | {rnd} | {scale_plus} | {tl} | {sl} |".format(
                target=target,
                tmp=_fmt(row[f"{TEMPORAL_PLUS}_to_raw_last_ratio"]),
                rnd=_fmt(row[f"{RANDOM_PLUS}_to_raw_last_ratio"]),
                scale_plus=_fmt(
                    row.get(f"{SCALE_PLUS}_to_raw_last_ratio")
                    if SCALE_PLUS in features
                    else None
                ),
                tl=_fmt(row[f"{TEMPORAL}_to_best_raw_ratio"]),
                sl=_fmt(
                    row.get(f"{SCALE}_to_best_raw_ratio") if SCALE in features else None
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- Temporal raw+learned future improvements: `{counts['temporal_raw_plus_future_improvements']}/5`.",
            f"- Temporal learned standalone best-raw wins: `{counts['temporal_learned_best_raw_future_wins']}/5`.",
            f"- Random raw+feature future improvements: `{counts['random_raw_plus_future_improvements']}/5`.",
            f"- Scale raw+learned future improvements: `{counts['scale_raw_plus_future_improvements']}/5`.",
            f"- Temporal raw+learned beats scale raw+learned: `{counts['temporal_raw_plus_beats_scale_raw_plus_future_targets']}/5`.",
            f"- Current-IV temporal raw+learned status: `{counts['current_iv_temporal_raw_plus_status']}`.",
            "",
            "## Random-Control Check",
            "",
            "The current-IV raw+temporal improvement is not sufficient evidence",
            "of useful learned temporal state because raw+random temporal features",
            "perform at least as well on that guardrail in this smoke bakeoff.",
            "",
            "- Raw+temporal/current-IV ratio:",
            f"  `{result['decision']['current_iv_temporal_raw_plus_to_raw_ratio']:.6f}`.",
            "- Raw+random/current-IV ratio:",
            f"  `{result['decision']['current_iv_random_raw_plus_to_raw_ratio']:.6f}`.",
            "- Warning:",
            f"  `{result['decision']['random_control_warning']}`.",
            "",
            "## Decision",
            "",
            f"Promotion decision: `{result['promotion_decision']}`.",
            "",
            f"Temporal route status: `{result['decision']['temporal_context_target_status']}`.",
            "",
            f"Future utility status: `{result['decision']['future_utility_status']}`.",
            "",
            "This is an evaluation bakeoff, not Part B authorization. A temporal JEPA",
            "follow-up is only justified if it improves the same frozen guardrails",
            "without relying on small knob tuning.",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Temporal JEPA frozen bakeoff")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--target_len", type=int, default=5)
    parser.add_argument("--max_train_windows", type=int, default=128)
    parser.add_argument("--max_val_windows", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2173)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--barlow_weight", type=float, default=0.05)
    parser.add_argument("--offdiag_weight", type=float, default=0.005)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--predictor_hidden_dim", type=int, default=64)
    parser.add_argument("--ridge_alpha", type=float, default=10.0)
    parser.add_argument("--scale_checkpoint", default=DEFAULT_SCALE_CHECKPOINT)
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("results/world/temporal_jepa_bakeoff_head173.json"),
    )
    parser.add_argument(
        "--report_md",
        type=Path,
        default=Path(
            "experiments/world/reports/world_model_head173_temporal_jepa_bakeoff.md"
        ),
    )
    return parser


def main() -> int:
    result = run_temporal_jepa_bakeoff(_build_parser().parse_args())
    print(json.dumps(_serializable(result["summary_counts"]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
