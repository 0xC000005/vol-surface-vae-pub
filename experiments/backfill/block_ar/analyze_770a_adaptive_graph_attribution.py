#!/usr/bin/env python
"""770a: read the learned adaptive graph in a native joint checkpoint.

The diagnostic is intentionally inference-only: it does not change model
weights or sample new paths. It asks whether the adaptive graph residual learned
interpretable, regime-sensitive cross-channel structure that is worth showing
in the paper.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, ".")

from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (  # noqa: E402
    AdaptiveGraphResidualTokenTransitionVelocity,
)
from experiments.backfill.block_ar._rollout_220_utils import (  # noqa: E402
    make_serializable,
)
from experiments.backfill.block_ar.audit_627a_joint_panel_scenario_quality import (  # noqa: E402
    build_history_future,
    load_native_model,
    panel_daily_changes,
    select_raw_state_scope,
)
from experiments.backfill.block_ar.evaluate_438a_deployable_residual_bootstrap_system import (  # noqa: E402
    set_seed,
)


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(value) for value in obj]
    if isinstance(obj, tuple):
        return [_jsonable(value) for value in obj]
    return make_serializable(obj)


def _ordinal_rank(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(x.size, dtype=np.float64)
    return ranks


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size < 3:
        return float("nan")
    return _safe_pearson(_ordinal_rank(x), _ordinal_rank(y))


def _safe_corrcoef(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("expected [observations, channels]")
    out = np.eye(x.shape[1], dtype=np.float64)
    keep = x.std(axis=0) > 1e-12
    if np.sum(keep) >= 2:
        corr = np.corrcoef(x[:, keep], rowvar=False)
        out[np.ix_(keep, keep)] = np.nan_to_num(corr, nan=0.0)
    return out


def _offdiag_normalize(adj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    off = np.asarray(adj, dtype=np.float64).copy()
    n = off.shape[-1]
    diag = np.arange(n)
    off[..., diag, diag] = 0.0
    row_sum = off.sum(axis=-1, keepdims=True)
    norm = np.divide(off, row_sum, out=np.zeros_like(off), where=row_sum > 1e-12)
    return off, norm


def _abbrev(name: str) -> str:
    if name.startswith("factor:"):
        return name.split(":", 1)[1]
    return name.replace("iv:", "iv")


def _group_mass_matrix(adj: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    labels = list(dict.fromkeys(groups.tolist()))
    matrix = np.zeros((len(labels), len(labels)), dtype=np.float64)
    counts = np.zeros(len(labels), dtype=np.float64)
    for i, target_group in enumerate(labels):
        rows = np.where(groups == target_group)[0]
        counts[i] = float(rows.size)
        if rows.size == 0:
            continue
        for j, source_group in enumerate(labels):
            cols = np.where(groups == source_group)[0]
            if cols.size == 0:
                continue
            matrix[i, j] = float(adj[np.ix_(rows, cols)].sum(axis=1).mean())
    return {
        "target_groups": labels,
        "source_groups": labels,
        "matrix": matrix,
        "target_counts": counts,
    }


def _top_sources_for_targets(
    adj: np.ndarray,
    target_indices: np.ndarray,
    source_indices: np.ndarray,
    names: list[str],
    *,
    top_n: int,
) -> list[dict[str, float | str | int]]:
    if target_indices.size == 0 or source_indices.size == 0:
        return []
    source_weight = adj[np.ix_(target_indices, source_indices)].sum(axis=0)
    source_weight = source_weight / max(float(target_indices.size), 1.0)
    order = np.argsort(source_weight)[::-1][:top_n]
    return [
        {
            "rank": int(rank + 1),
            "source": names[int(source_indices[idx])],
            "source_index": int(source_indices[idx]),
            "mean_weight_per_target": float(source_weight[idx]),
        }
        for rank, idx in enumerate(order)
    ]


def _top_directed_edges(
    adj: np.ndarray, names: list[str], *, top_n: int
) -> list[dict[str, float | str | int]]:
    n = adj.shape[0]
    mask = ~np.eye(n, dtype=bool)
    flat_idx = np.argsort(adj[mask])[::-1][:top_n]
    pairs = np.argwhere(mask)
    rows = []
    for rank, pos in enumerate(flat_idx):
        target, source = pairs[pos]
        rows.append(
            {
                "rank": int(rank + 1),
                "target": names[int(target)],
                "source": names[int(source)],
                "target_index": int(target),
                "source_index": int(source),
                "mean_weight": float(adj[int(target), int(source)]),
            }
        )
    return rows


def _top_sources_for_named_targets(
    adj: np.ndarray,
    names: list[str],
    target_names: list[str],
    *,
    top_n: int,
) -> dict[str, list[dict[str, float | str | int]]]:
    rows: dict[str, list[dict[str, float | str | int]]] = {}
    for target_name in target_names:
        if target_name not in names:
            continue
        target_idx = int(names.index(target_name))
        order = np.argsort(adj[target_idx])[::-1][:top_n]
        rows[target_name] = [
            {
                "rank": int(rank + 1),
                "target": target_name,
                "source": names[int(source_idx)],
                "target_index": target_idx,
                "source_index": int(source_idx),
                "mean_weight": float(adj[target_idx, int(source_idx)]),
            }
            for rank, source_idx in enumerate(order)
        ]
    return rows


@torch.no_grad()
def _graph_batch(
    model: Any,
    history_level: np.ndarray,
    history_norm: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    drift_feature: np.ndarray,
    *,
    start: int,
    end: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    velocity = model.velocity
    if not isinstance(velocity, AdaptiveGraphResidualTokenTransitionVelocity):
        raise TypeError(
            "expected AdaptiveGraphResidualTokenTransitionVelocity, got "
            f"{type(velocity).__name__}"
        )

    level = torch.from_numpy(history_level[start:end]).to(device)
    norm = torch.from_numpy(history_norm[start:end]).to(device)
    ctr = torch.from_numpy(center[start:end]).to(device)
    scl = torch.from_numpy(scale[start:end]).to(device)
    drift = torch.from_numpy(drift_feature[start:end]).to(device)

    level_scores = model.level_values_to_scores(level)
    flow_coordinate = model._to_flow_coordinate(norm)
    memory_state = model._encode_prefix(level_scores, flow_coordinate, ctr, scl, drift)[
        :, -1
    ]
    current_level_score = level_scores[:, -1]
    x_t = torch.zeros_like(current_level_score)
    t = torch.full(
        (x_t.shape[0],), 0.5, device=device, dtype=current_level_score.dtype
    )

    hidden = velocity.base.hidden_tokens(x_t, current_level_score, memory_state, t)
    bsz, n_cells, dim = hidden.shape
    k_neighbors = min(max(int(velocity.cfg.adaptive_graph_k), 1), n_cells)
    q = velocity.q_proj(hidden)
    k = velocity.k_proj(hidden)
    v = velocity.v_proj(hidden)
    score = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(float(dim))
    rel_bias = velocity.rel_proj(
        velocity.relative_cell_features[:n_cells, :n_cells].to(
            device=device, dtype=hidden.dtype
        )
    ).mean(dim=-1)
    score = score + rel_bias[None, :, :]
    top_idx = torch.topk(score, k=k_neighbors, dim=-1).indices
    neighbor_score = torch.gather(score, 2, top_idx)
    weight = torch.softmax(neighbor_score, dim=-1)
    adj = torch.zeros(bsz, n_cells, n_cells, device=device, dtype=hidden.dtype)
    adj.scatter_(2, top_idx, weight)

    residual = velocity._graph_residual(hidden)
    base_out = velocity.base.out(hidden).squeeze(-1)
    graph_out = velocity.base.out(hidden + residual).squeeze(-1)
    delta_out = graph_out - base_out
    hidden_norm = hidden.norm(dim=-1).mean(dim=1)
    residual_norm = residual.norm(dim=-1).mean(dim=1)
    base_out_norm = base_out.norm(dim=-1)
    delta_out_norm = delta_out.norm(dim=-1)

    return {
        "adj": adj.detach().cpu().numpy().astype(np.float32),
        "residual_hidden_ratio": (
            residual_norm / hidden_norm.clamp_min(1e-12)
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        "velocity_delta_ratio": (
            delta_out_norm / base_out_norm.clamp_min(1e-12)
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        "velocity_delta_abs": delta_out.abs()
        .mean(dim=1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        "base_velocity_abs": base_out.abs()
        .mean(dim=1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
    }


def collect_graph_attribution(
    model: Any,
    history: tuple[np.ndarray, ...],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    history_level, history_norm, center, scale, drift_feature = history
    batches = []
    for start in range(0, history_level.shape[0], batch_size):
        end = min(start + batch_size, history_level.shape[0])
        batches.append(
            _graph_batch(
                model,
                history_level,
                history_norm,
                center,
                scale,
                drift_feature,
                start=start,
                end=end,
                device=device,
            )
        )
    return {
        key: np.concatenate([batch[key] for batch in batches], axis=0)
        for key in batches[0]
    }


def _plot_heatmap(
    path: Path,
    matrix: np.ndarray,
    *,
    xlabels: list[str],
    ylabels: list[str],
    title: str,
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("source channel")
    ax.set_ylabel("target channel")
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=90, fontsize=7)
    ax.set_yticklabels(ylabels, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_group_matrices(path: Path, panels: dict[str, dict[str, Any]]) -> None:
    names = list(panels)
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.8))
    if len(names) == 1:
        axes = [axes]
    vmax = max(float(np.max(panels[name]["matrix"])) for name in names)
    for ax, name in zip(axes, names, strict=True):
        matrix = panels[name]["matrix"]
        im = ax.imshow(matrix, vmin=0.0, vmax=max(vmax, 1e-6), cmap="Blues")
        ax.set_title(name)
        ax.set_xticks(np.arange(len(panels[name]["source_groups"])))
        ax.set_yticks(np.arange(len(panels[name]["target_groups"])))
        ax.set_xticklabels(panels[name]["source_groups"], rotation=30, ha="right")
        ax.set_yticklabels(panels[name]["target_groups"])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=axes, fraction=0.035, pad=0.04)
    fig.suptitle("Adaptive graph source mass by target/source group")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_top_anchor_sources(
    path: Path,
    panels: dict[str, list[dict[str, Any]]],
    *,
    title: str,
) -> None:
    labels = []
    panel_weights: dict[str, dict[str, float]] = {}
    for panel, rows in panels.items():
        panel_weights[panel] = {}
        for row in rows:
            source = str(row["source"])
            if source not in labels:
                labels.append(source)
            panel_weights[panel][source] = float(row["mean_weight_per_target"])
    if not labels:
        return
    labels = labels[:10]
    x = np.arange(len(labels))
    width = min(0.8 / max(len(panels), 1), 0.35)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for offset, (panel, weights) in enumerate(panel_weights.items()):
        values = [weights.get(label, 0.0) for label in labels]
        ax.bar(x + (offset - (len(panels) - 1) / 2.0) * width, values, width, label=panel)
    ax.set_title(title)
    ax.set_ylabel("mean graph weight per target")
    ax.set_xticks(x)
    ax.set_xticklabels([_abbrev(label) for label in labels], rotation=35, ha="right")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _format_matrix_table(title: str, group_result: dict[str, Any]) -> list[str]:
    rows = [f"### {title}", ""]
    src = group_result["source_groups"]
    tgt = group_result["target_groups"]
    rows.append("| target/source | " + " | ".join(src) + " |")
    rows.append("| --- | " + " | ".join(["---:"] * len(src)) + " |")
    for i, name in enumerate(tgt):
        values = " | ".join(f"{float(group_result['matrix'][i, j]):.4f}" for j in range(len(src)))
        rows.append(f"| {name} | {values} |")
    rows.append("")
    return rows


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    summary = result["summary"]
    lines = [
        "# 770a Adaptive Graph Attribution",
        "",
        "This diagnostic reads the learned adaptive graph residual in the native joint checkpoint. Rows are target channels; columns are source channels.",
        "",
        "## Main Numbers",
        "",
        f"- windows: `{summary['n_windows']}`",
        f"- channels: `{summary['n_channels']}`",
        f"- graph top-k: `{summary['adaptive_graph_k']}`",
        f"- mean self-edge mass: `{summary['self_edge_mass_mean']:.4f}`",
        f"- off-diagonal effective source count: `{summary['offdiag_effective_source_count']:.2f}`",
        f"- graph edge vs empirical abs-correlation Pearson: `{summary['edge_abs_corr_pearson']:.4f}`",
        f"- graph edge vs empirical abs-correlation Spearman: `{summary['edge_abs_corr_spearman']:.4f}`",
        f"- residual hidden/base norm ratio: `{summary['residual_hidden_ratio_mean']:.4f}`",
        f"- residual velocity/base norm ratio: `{summary['velocity_delta_ratio_mean']:.4f}`",
        f"- residual mean abs velocity delta: `{summary['velocity_delta_abs_mean']:.4f}`",
        "",
    ]
    for name, group_result in result["group_matrices"].items():
        lines.extend(_format_matrix_table(name, group_result))

    lines += [
        "## Top Anchor Sources Into IV Targets",
        "",
        "| split | rank | source | mean graph weight per IV target |",
        "| --- | ---: | --- | ---: |",
    ]
    for split_name, rows in result["top_anchor_sources_for_iv"].items():
        for row in rows:
            lines.append(
                f"| {split_name} | {row['rank']} | {row['source']} | {row['mean_weight_per_target']:.5f} |"
            )
    lines += [
        "",
        "## Top Directed Cross-Channel Edges",
        "",
        "| rank | target | source | mean graph weight |",
        "| ---: | --- | --- | ---: |",
    ]
    for row in result["top_directed_edges"]:
        lines.append(
            f"| {row['rank']} | {row['target']} | {row['source']} | {row['mean_weight']:.5f} |"
        )
    if result.get("top_sources_for_named_targets"):
        lines += [
            "",
            "## Named Target Sanity Rows",
            "",
            "| target | rank | attended/source channel | mean graph weight |",
            "| --- | ---: | --- | ---: |",
        ]
        for target_name, rows in result["top_sources_for_named_targets"].items():
            for row in rows:
                lines.append(
                    f"| {target_name} | {row['rank']} | {row['source']} | {row['mean_weight']:.5f} |"
                )
    lines += [
        "",
        "## Figures",
        "",
    ]
    for name, figure_path in result["figures"].items():
        lines.append(f"- {name}: `{figure_path}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_type", default="662a")
    parser.add_argument("--state_scope", default="joint38")
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument(
        "--clean_nonpositive_log_levels", action="store_true", default=True
    )
    parser.add_argument(
        "--positive_level_policy",
        choices=["reference_based", "observed_positive"],
        default="reference_based",
    )
    parser.add_argument(
        "--iv_transform", choices=["log_level", "bounded_logit"], default="log_level"
    )
    parser.add_argument("--iv_lower_bound", type=float, default=1e-4)
    parser.add_argument("--iv_upper_bound", type=float, default=1.0)
    parser.add_argument("--max_windows", type=int, default=441)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--top_n", type=int, default=12)
    parser.add_argument("--seed", type=int, default=770)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output_dir",
        default="results/block_ar/770a_adaptive_graph_attribution",
    )
    args = parser.parse_args()

    set_seed(int(args.seed))
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    model, payload = load_native_model(args.model_type, args.checkpoint, device)
    if not isinstance(model.velocity, AdaptiveGraphResidualTokenTransitionVelocity):
        raise RuntimeError(
            "checkpoint does not use adaptive graph residual velocity mixer"
        )

    history, _future, specs, block, alignment = build_history_future(args, payload)
    if not isinstance(history, tuple) or len(history) != 5:
        raise RuntimeError("expected state-aware normalized innovation history tuple")
    if (
        float(alignment["history_max_abs_error"]) > 1e-6
        or float(alignment["future_max_abs_error"]) > 1e-6
    ):
        raise RuntimeError(f"panel alignment failed: {alignment}")

    n_windows = min(int(args.max_windows), int(history[0].shape[0]))
    history = tuple(item[:n_windows] for item in history)
    raw_history, raw_future = select_raw_state_scope(
        block, payload.get("state_scope", args.state_scope), int(args.iv_count)
    )
    raw_history = raw_history[:n_windows]
    raw_future = raw_future[:n_windows]

    names = [spec.name for spec in specs]
    if len(names) != history[0].shape[-1]:
        raise RuntimeError("spec names do not match model channels")
    iv_count = min(int(args.iv_count), len(names))
    groups = np.asarray(
        ["iv_surface" if i < iv_count else "anchor_factor" for i in range(len(names))]
    )

    attr = collect_graph_attribution(
        model, history, batch_size=int(args.batch_size), device=device
    )
    adj = np.asarray(attr["adj"], dtype=np.float64)
    avg_adj = adj.mean(axis=0)
    offdiag_adj, offdiag_norm = _offdiag_normalize(adj)
    avg_offdiag = offdiag_adj.mean(axis=0)
    avg_offdiag_norm = offdiag_norm.mean(axis=0)

    diag_idx = np.arange(avg_adj.shape[0])
    self_mass = avg_adj[diag_idx, diag_idx]
    row_entropy = -np.sum(
        np.where(avg_offdiag_norm > 0.0, avg_offdiag_norm * np.log(avg_offdiag_norm + 1e-12), 0.0),
        axis=1,
    )
    effective_sources = float(np.exp(row_entropy).mean())

    gt_delta = panel_daily_changes(raw_history, raw_future).reshape(-1, len(names))
    gt_abs_corr = np.abs(_safe_corrcoef(gt_delta))
    offmask = ~np.eye(len(names), dtype=bool)
    edge_abs_corr_pearson = _safe_pearson(avg_offdiag_norm[offmask], gt_abs_corr[offmask])
    edge_abs_corr_spearman = _safe_spearman(
        avg_offdiag_norm[offmask], gt_abs_corr[offmask]
    )

    history_iv_delta = np.diff(raw_history[..., :iv_count], axis=1)
    history_iv_activity = np.mean(history_iv_delta * history_iv_delta, axis=(1, 2))
    calm_cut = np.quantile(history_iv_activity, 0.20)
    turbulent_cut = np.quantile(history_iv_activity, 0.80)
    calm_mask = history_iv_activity <= calm_cut
    turbulent_mask = history_iv_activity >= turbulent_cut
    split_masks = {
        "all": np.ones(n_windows, dtype=bool),
        "calm_history": calm_mask,
        "turbulent_history": turbulent_mask,
    }

    group_matrices: dict[str, Any] = {}
    top_anchor_sources: dict[str, Any] = {}
    iv_idx = np.arange(iv_count)
    anchor_idx = np.arange(iv_count, len(names))
    for split_name, mask in split_masks.items():
        split_offdiag_norm = offdiag_norm[mask].mean(axis=0)
        group_matrices[split_name] = _group_mass_matrix(split_offdiag_norm, groups)
        top_anchor_sources[split_name] = _top_sources_for_targets(
            split_offdiag_norm,
            iv_idx,
            anchor_idx,
            names,
            top_n=int(args.top_n),
        )

    top_edges = _top_directed_edges(avg_offdiag, names, top_n=int(args.top_n))
    top_sources_for_named_targets = _top_sources_for_named_targets(
        avg_offdiag_norm,
        names,
        ["factor:vix", "factor:spx"],
        top_n=int(args.top_n),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_names = [_abbrev(name) for name in names]
    figures = {
        "offdiag_adjacency_heatmap": str(
            output_dir / "770a_offdiag_adjacency_heatmap.png"
        ),
        "group_mass_matrices": str(output_dir / "770a_group_mass_matrices.png"),
        "top_anchor_sources_for_iv": str(
            output_dir / "770a_top_anchor_sources_for_iv.png"
        ),
    }
    _plot_heatmap(
        Path(figures["offdiag_adjacency_heatmap"]),
        avg_offdiag_norm,
        xlabels=label_names,
        ylabels=label_names,
        title="Adaptive graph off-diagonal source mass",
    )
    _plot_group_matrices(Path(figures["group_mass_matrices"]), group_matrices)
    _plot_top_anchor_sources(
        Path(figures["top_anchor_sources_for_iv"]),
        top_anchor_sources,
        title="Top anchor-factor sources into IV-surface targets",
    )

    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_best_val": float(payload.get("best_val", float("nan"))),
        "state_scope": payload.get("state_scope", args.state_scope),
        "n_windows": int(n_windows),
        "n_channels": int(len(names)),
        "iv_count": int(iv_count),
        "anchor_count": int(len(names) - iv_count),
        "adaptive_graph_k": int(model.cfg.adaptive_graph_k),
        "self_edge_mass_mean": float(np.mean(self_mass)),
        "self_edge_mass_median": float(np.median(self_mass)),
        "offdiag_effective_source_count": effective_sources,
        "edge_abs_corr_pearson": edge_abs_corr_pearson,
        "edge_abs_corr_spearman": edge_abs_corr_spearman,
        "residual_hidden_ratio_mean": float(
            np.mean(attr["residual_hidden_ratio"])
        ),
        "residual_hidden_ratio_p90": float(
            np.quantile(attr["residual_hidden_ratio"], 0.90)
        ),
        "velocity_delta_ratio_mean": float(np.mean(attr["velocity_delta_ratio"])),
        "velocity_delta_ratio_p90": float(
            np.quantile(attr["velocity_delta_ratio"], 0.90)
        ),
        "velocity_delta_abs_mean": float(np.mean(attr["velocity_delta_abs"])),
        "base_velocity_abs_mean": float(np.mean(attr["base_velocity_abs"])),
        "calm_windows": int(np.sum(calm_mask)),
        "turbulent_windows": int(np.sum(turbulent_mask)),
        "history_iv_activity_q20": float(calm_cut),
        "history_iv_activity_q80": float(turbulent_cut),
        "alignment": alignment,
    }
    result = {
        "summary": summary,
        "channel_names": names,
        "group_matrices": group_matrices,
        "top_anchor_sources_for_iv": top_anchor_sources,
        "top_directed_edges": top_edges,
        "top_sources_for_named_targets": top_sources_for_named_targets,
        "figures": figures,
    }

    out_json = output_dir / "770a_adaptive_graph_attribution.json"
    out_md = output_dir / "770a_adaptive_graph_attribution.md"
    serializable_result = _jsonable(result)
    out_json.write_text(json.dumps(serializable_result, indent=2), encoding="utf-8")
    write_markdown(out_md, result)
    print(json.dumps(_jsonable(summary), indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
