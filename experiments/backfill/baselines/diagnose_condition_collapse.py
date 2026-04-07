#!/usr/bin/env python
"""Independent diagnostic for condition-vector "collapse" claims.

Measures:
  - geometry of encoder condition vectors on the test split
  - raw vs centered cosine similarity structure
  - PCA variance spectrum and effective rank
  - cond_proj shared-vs-window-specific decomposition
  - AR rollout MAE under normal / mean / zero / permuted conditions
  - whether mean-subtracted residuals still encode window properties
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.gru_encoder import EncoderConfig
from experiments.backfill.block_ar.train_164a_v3_percell_bptt_softplus import (
    ARSpatialTransformerModel,
    normalize_iv,
)


HISTORY_LEN = 30
FUTURE_LEN = 30
N_CELLS = 25
GRID_H = 5
GRID_W = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("models/backfill/afcrps_164a_v3_percell_bptt_softplus/best_model.pt"),
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/vol_surface_with_ret.npz"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test-start", type=int, default=4540)
    parser.add_argument(
        "--geometry-windows",
        type=int,
        default=0,
        help="0 means use all available test windows for geometry/correlation measurements.",
    )
    parser.add_argument(
        "--rollout-windows",
        type=int,
        default=512,
        help="Number of windows for mean/zero/permute rollout ablations.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("/tmp/condition_collapse_diagnostic.json"),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    return obj


def build_model(checkpoint_path: Path, device: torch.device) -> tuple[ARSpatialTransformerModel, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    encoder_cfg = config["encoder"]
    decoder_cfg = config["decoder"]
    encoder = EncoderConfig(
        input_dim=encoder_cfg["input_dim"],
        gru_hidden_dim=encoder_cfg["gru_hidden_dim"],
        bottleneck_dim=encoder_cfg["bottleneck_dim"],
        dropout=encoder_cfg.get("dropout", 0.1),
        cond_aug_sigma=encoder_cfg.get("cond_aug_sigma", 0.0),
    )
    model = ARSpatialTransformerModel(encoder, decoder_cfg)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch for {checkpoint_path}: missing={missing}, unexpected={unexpected}"
        )
    model.to(device)
    model.eval()
    return model, checkpoint


def load_windows(data_path: Path, test_start: int, n_windows: int | None) -> dict[str, np.ndarray]:
    data = np.load(data_path)
    surfaces = data["surface"].astype(np.float32)
    max_start = surfaces.shape[0] - HISTORY_LEN - FUTURE_LEN
    if n_windows is None or n_windows <= 0:
        end_exclusive = max_start + 1
    else:
        end_exclusive = min(test_start + n_windows, max_start + 1)
    starts = np.arange(test_start, end_exclusive, dtype=np.int64)
    hist_offsets = np.arange(HISTORY_LEN, dtype=np.int64)
    fut_offsets = np.arange(FUTURE_LEN, dtype=np.int64)
    histories = surfaces[starts[:, None] + hist_offsets[None, :]]
    futures = surfaces[starts[:, None] + HISTORY_LEN + fut_offsets[None, :]]
    return {
        "starts": starts,
        "histories": histories,
        "futures": futures,
        "n_total_surfaces": surfaces.shape[0],
    }


def encode_conditions(
    model: ARSpatialTransformerModel,
    histories: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    conds: list[np.ndarray] = []
    for start in range(0, len(histories), batch_size):
        end = min(start + batch_size, len(histories))
        hist_batch = torch.from_numpy(histories[start:end]).to(device)
        hist_norm = normalize_iv(hist_batch).reshape(end - start, HISTORY_LEN, N_CELLS)
        with torch.inference_mode():
            gru_outputs, _ = model.encoder.gru(hist_norm)
            attn_logits = model.encoder.attn_proj(gru_outputs).squeeze(-1)
            attn_weights = F.softmax(attn_logits, dim=1)
            h_pooled = (attn_weights.unsqueeze(-1) * gru_outputs).sum(dim=1)
            cond = model.encoder.bottleneck(h_pooled)
        conds.append(cond.cpu().numpy())
    return np.concatenate(conds, axis=0)


def compute_window_properties(histories: np.ndarray, futures: np.ndarray) -> dict[str, np.ndarray]:
    hist_mean_iv_ts = histories.mean(axis=(2, 3))
    fut_mean_iv_ts = futures.mean(axis=(2, 3))
    hist_daily_changes = np.diff(hist_mean_iv_ts, axis=1)
    fut_daily_changes = np.diff(fut_mean_iv_ts, axis=1)
    return {
        "history_mean_iv": histories.mean(axis=(1, 2, 3)),
        "last_frame_mean_iv": histories[:, -1].mean(axis=(1, 2)),
        "history_vov": hist_daily_changes.std(axis=1),
        "future_realized_vol": fut_daily_changes.std(axis=1),
    }


def offdiag_mean(mat: np.ndarray) -> float:
    n = mat.shape[0]
    return float((mat.sum() - np.trace(mat)) / (n * (n - 1)))


def pairwise_cosine_stats(vectors: np.ndarray) -> dict[str, Any]:
    vec = vectors.astype(np.float64)
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    unit = vec / norms
    cos = unit @ unit.T
    return {
        "mean_offdiag": offdiag_mean(cos),
        "median_offdiag": float(np.median(cos[~np.eye(cos.shape[0], dtype=bool)])),
        "std_offdiag": float(np.std(cos[~np.eye(cos.shape[0], dtype=bool)])),
        "matrix": cos,
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
    }


def within_between_cosine(cos_matrix: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    same = labels[:, None] == labels[None, :]
    offdiag = ~np.eye(len(labels), dtype=bool)
    within_mask = same & offdiag
    between_mask = (~same) & offdiag
    within = cos_matrix[within_mask]
    between = cos_matrix[between_mask]
    return {
        "within_mean": float(within.mean()),
        "between_mean": float(between.mean()),
        "difference": float(within.mean() - between.mean()),
    }


def pca_stats(vectors: np.ndarray) -> dict[str, Any]:
    x = vectors.astype(np.float64)
    x_center = x - x.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(x_center, full_matrices=False)
    eig = s ** 2
    total = eig.sum()
    var_ratio = eig / np.clip(total, 1e-12, None)
    entropy = -(var_ratio * np.log(np.clip(var_ratio, 1e-12, None))).sum()
    return {
        "pc1_var_explained": float(var_ratio[0]),
        "pc2_var_explained": float(var_ratio[1]) if len(var_ratio) > 1 else 0.0,
        "effective_rank": float(np.exp(entropy)),
        "top10_var_explained": [float(v) for v in var_ratio[:10]],
        "scores_top5": (x_center @ np.linalg.svd(x_center, full_matrices=False)[2][:5].T).astype(np.float32),
    }


def projected_condition_stats(
    model: ARSpatialTransformerModel,
    conds: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    with torch.inference_mode():
        cond_t = torch.from_numpy(conds.astype(np.float32)).to(device)
        proj = model.decoder.cond_proj(cond_t).cpu().numpy()
    mean_proj = proj.mean(axis=0)
    resid_proj = proj - mean_proj
    sample_sq_norm = np.sum(proj ** 2, axis=1)
    resid_sq_norm = np.sum(resid_proj ** 2, axis=1)
    mean_sq_norm = float(np.sum(mean_proj ** 2))
    total_sq_norm = float(sample_sq_norm.mean())
    return {
        "mean_proj_norm": float(np.sqrt(mean_sq_norm)),
        "mean_sample_proj_norm": float(np.sqrt(sample_sq_norm).mean()),
        "mean_residual_proj_norm": float(np.sqrt(resid_sq_norm).mean()),
        "constant_energy_fraction": mean_sq_norm / total_sq_norm if total_sq_norm > 0 else float("nan"),
        "residual_energy_fraction": float(resid_sq_norm.mean() / total_sq_norm) if total_sq_norm > 0 else float("nan"),
        "proj_dim": proj.shape[1],
    }


def cross_validated_r2(x: np.ndarray, y: np.ndarray, seed: int) -> float:
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    preds = np.zeros_like(y, dtype=np.float64)
    for train_idx, test_idx in kf.split(x):
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )
        model.fit(x[train_idx], y[train_idx])
        preds[test_idx] = model.predict(x[test_idx])
    return float(r2_score(y, preds))


def cross_validated_logistic_accuracy(x: np.ndarray, y: np.ndarray, seed: int) -> float:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs = []
    for train_idx, test_idx in cv.split(x, y):
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        max_iter=5000,
                    ),
                ),
            ]
        )
        model.fit(x[train_idx], y[train_idx])
        accs.append(float((model.predict(x[test_idx]) == y[test_idx]).mean()))
    return float(np.mean(accs))


def residual_signal_stats(
    conds: np.ndarray,
    properties: dict[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    centered = conds - conds.mean(axis=0, keepdims=True)
    pca = pca_stats(conds)
    scores = pca["scores_top5"]
    result: dict[str, Any] = {
        "centered_linear_probe_vov_quintile_acc": None,
        "properties": {},
    }

    vov_quint = np.digitize(
        properties["history_vov"],
        np.quantile(properties["history_vov"], [0.2, 0.4, 0.6, 0.8]),
        right=True,
    )
    result["centered_linear_probe_vov_quintile_acc"] = cross_validated_logistic_accuracy(
        centered.astype(np.float64),
        vov_quint.astype(np.int64),
        seed,
    )

    for name, vals in properties.items():
        prop_res = {
            "residual_norm_spearman_rho": float(spearmanr(np.linalg.norm(centered, axis=1), vals).statistic),
            "residual_linear_r2": cross_validated_r2(centered.astype(np.float64), vals.astype(np.float64), seed),
            "top5_pc_spearman": {},
        }
        for i in range(scores.shape[1]):
            rho, pval = spearmanr(scores[:, i], vals)
            prop_res["top5_pc_spearman"][f"pc{i+1}"] = {
                "rho": float(rho),
                "pval": float(pval),
            }
        result["properties"][name] = prop_res
    return result


def sample_with_condition_mode(
    model: ARSpatialTransformerModel,
    histories: np.ndarray,
    n_samples: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    mode: str,
    mean_condition: np.ndarray | None = None,
) -> np.ndarray:
    if mode not in {"normal", "zero", "permute", "mean"}:
        raise ValueError(f"Unknown mode: {mode}")

    set_seed(seed)
    samples_out: list[np.ndarray] = []
    chunk_size = 10

    permuted_histories = None
    if mode == "permute":
        perm = np.random.RandomState(seed + 999).permutation(len(histories))
        if len(perm) > 1:
            fixed = perm == np.arange(len(perm))
            if fixed.any():
                perm[fixed] = np.roll(perm, 1)[fixed]
        permuted_histories = histories[perm]

    mean_cond_t = None
    if mode == "mean":
        if mean_condition is None:
            raise ValueError("mean_condition must be provided for mode='mean'")
        mean_cond_t = torch.from_numpy(mean_condition.astype(np.float32)).to(device)

    for start in range(0, len(histories), batch_size):
        end = min(start + batch_size, len(histories))
        hist_batch = torch.from_numpy(histories[start:end]).to(device)
        hist_norm = normalize_iv(hist_batch)
        batch = hist_batch.shape[0]

        with torch.inference_mode():
            last_frame = hist_batch[:, -1].reshape(batch, N_CELLS)

            if mode == "permute":
                assert permuted_histories is not None
                cond_hist_batch = torch.from_numpy(permuted_histories[start:end]).to(device)
                cond_hist_norm = normalize_iv(cond_hist_batch)
            else:
                cond_hist_norm = hist_norm

            if mode in {"normal", "permute"}:
                hist_flat_cond = cond_hist_norm.reshape(batch, HISTORY_LEN, N_CELLS)
                cond_gru_outputs_base, cond_h_last_base = model.encoder.gru(hist_flat_cond)

            batch_samples: list[torch.Tensor] = []
            for k0 in range(0, n_samples, chunk_size):
                k = min(chunk_size, n_samples - k0)
                prev = last_frame.unsqueeze(1).expand(batch, k, -1).reshape(batch * k, N_CELLS)

                if mode in {"normal", "permute"}:
                    gru_outputs = cond_gru_outputs_base.unsqueeze(1).expand(
                        batch, k, -1, -1
                    ).reshape(batch * k, -1, model.encoder_config.gru_hidden_dim)
                    gru_state = cond_h_last_base.unsqueeze(2).expand(
                        1, batch, k, -1
                    ).reshape(1, batch * k, -1).contiguous()

                frames = []
                for _ in range(FUTURE_LEN):
                    z_t = torch.randn(batch * k, model.decoder.noise_dim, device=device)
                    if mode == "zero":
                        cond_t = torch.zeros(
                            batch * k,
                            model.encoder_config.bottleneck_dim,
                            device=device,
                            dtype=prev.dtype,
                        )
                    elif mode == "mean":
                        cond_t = mean_cond_t.unsqueeze(0).expand(batch * k, -1)
                    else:
                        attn_logits = model.encoder.attn_proj(gru_outputs).squeeze(-1)
                        attn_weights = F.softmax(attn_logits, dim=1)
                        h_pooled = (attn_weights.unsqueeze(-1) * gru_outputs).sum(dim=1)
                        cond_t = model.encoder.bottleneck(h_pooled)

                    delta = model.decoder(cond_t, prev, z_t)
                    frame_t = prev + torch.tanh(delta)
                    frames.append(frame_t)

                    if mode in {"normal", "permute"}:
                        frame_norm = normalize_iv(frame_t).unsqueeze(1)
                        gru_out, gru_state = model.encoder.gru(frame_norm, gru_state)
                        gru_outputs = torch.cat([gru_outputs, gru_out], dim=1)
                    prev = frame_t

                batch_samples.append(torch.stack(frames, dim=1).reshape(batch, k, FUTURE_LEN, GRID_H, GRID_W))

        samples_out.append(torch.cat(batch_samples, dim=1).cpu().numpy())

    return np.concatenate(samples_out, axis=0)


def summarize_rollout(samples: np.ndarray, futures: np.ndarray) -> dict[str, Any]:
    pred_mean = samples.mean(axis=1)
    abs_err = np.abs(pred_mean - futures)
    per_window_mae = abs_err.mean(axis=(1, 2, 3))
    return {
        "mae_global": float(abs_err.mean()),
        "per_window_mae_mean": float(per_window_mae.mean()),
        "per_window_mae_std": float(per_window_mae.std()),
        "per_window_mae": per_window_mae,
    }


def paired_relative_change(base: np.ndarray, other: np.ndarray) -> dict[str, float]:
    diff = other - base
    rel = diff.mean() / max(base.mean(), 1e-12)
    se = diff.std(ddof=1) / math.sqrt(len(diff))
    ci95 = 1.96 * se
    return {
        "mean_diff": float(diff.mean()),
        "mean_diff_ci95": float(ci95),
        "relative_diff": float(rel),
    }


def print_claim_summary(result: dict[str, Any]) -> None:
    geom = result["geometry"]
    rollout = result["rollout"]
    proj = result["cond_proj"]
    print("\n=== Condition geometry ===")
    print(
        f"raw cosine mean={geom['raw_cosine']['mean_offdiag']:.6f} "
        f"within={geom['raw_quintile_cosine']['within_mean']:.6f} "
        f"between={geom['raw_quintile_cosine']['between_mean']:.6f} "
        f"diff={geom['raw_quintile_cosine']['difference']:.6f}"
    )
    print(
        f"centered cosine mean={geom['centered_cosine']['mean_offdiag']:.6f} "
        f"PC1={geom['pca']['pc1_var_explained']:.4f} "
        f"eff_rank={geom['pca']['effective_rank']:.2f}"
    )

    print("\n=== cond_proj decomposition ===")
    print(
        f"constant_energy_fraction={proj['constant_energy_fraction']:.4f} "
        f"mean_proj_norm={proj['mean_proj_norm']:.4f} "
        f"mean_residual_proj_norm={proj['mean_residual_proj_norm']:.4f}"
    )

    print("\n=== Rollout MAE (512-window paired ablation) ===")
    print("mode      mae")
    for mode, stats in rollout["modes"].items():
        print(f"{mode:8s} {stats['mae_global']:.6f}")
    print("delta vs normal")
    for mode, stats in rollout["vs_normal"].items():
        print(
            f"{mode:8s} mean_diff={stats['mean_diff']:+.6f} "
            f"ci95={stats['mean_diff_ci95']:.6f} rel={stats['relative_diff']:+.4%}"
        )

    print("\n=== Residual signal ===")
    rs = result["residual_signal"]
    print(f"centered linear-probe VoV-quintile acc={rs['centered_linear_probe_vov_quintile_acc']:.4f}")
    for name, stats in rs["properties"].items():
        best_pc = max(stats["top5_pc_spearman"].items(), key=lambda kv: abs(kv[1]["rho"]))
        print(
            f"{name:18s} r2={stats['residual_linear_r2']:.4f} "
            f"res_norm_rho={stats['residual_norm_spearman_rho']:.4f} "
            f"best_pc={best_pc[0]}:{best_pc[1]['rho']:.4f}"
        )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"ckpt={args.ckpt}")
    print(f"device={device}")
    model, checkpoint = build_model(args.ckpt, device)
    print(
        f"loaded epoch={checkpoint.get('epoch')} "
        f"val_loss={checkpoint.get('val_loss'):.6f} "
        f"type={checkpoint['config'].get('type')}"
    )

    geom_windows = load_windows(args.data_path, args.test_start, None if args.geometry_windows == 0 else args.geometry_windows)
    print(f"geometry windows={len(geom_windows['starts'])}")
    conds = encode_conditions(model, geom_windows["histories"], args.batch_size, device)
    props = compute_window_properties(geom_windows["histories"], geom_windows["futures"])

    raw_cos = pairwise_cosine_stats(conds)
    centered_conds = conds - conds.mean(axis=0, keepdims=True)
    centered_cos = pairwise_cosine_stats(centered_conds)
    vov_quint = np.digitize(
        props["history_vov"],
        np.quantile(props["history_vov"], [0.2, 0.4, 0.6, 0.8]),
        right=True,
    )
    raw_quint_cos = within_between_cosine(raw_cos["matrix"], vov_quint)
    centered_quint_cos = within_between_cosine(centered_cos["matrix"], vov_quint)
    pca = pca_stats(conds)
    proj = projected_condition_stats(model, conds, device)
    residual_signal = residual_signal_stats(conds, props, args.seed)

    rollout_windows = load_windows(args.data_path, args.test_start, args.rollout_windows)
    print(f"rollout windows={len(rollout_windows['starts'])}")
    mean_cond = conds.mean(axis=0)

    rollout_modes = {}
    for mode in ("normal", "mean", "zero", "permute"):
        print(f"running rollout mode={mode} ...")
        samples = sample_with_condition_mode(
            model,
            rollout_windows["histories"],
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
            mode=mode,
            mean_condition=mean_cond,
        )
        rollout_modes[mode] = summarize_rollout(samples, rollout_windows["futures"])

    vs_normal = {}
    base = rollout_modes["normal"]["per_window_mae"]
    for mode in ("mean", "zero", "permute"):
        vs_normal[mode] = paired_relative_change(base, rollout_modes[mode]["per_window_mae"])

    result = {
        "setup": {
            "ckpt": args.ckpt,
            "epoch": checkpoint.get("epoch"),
            "val_loss": checkpoint.get("val_loss"),
            "type": checkpoint["config"].get("type"),
            "device": str(device),
            "test_start": args.test_start,
            "geometry_windows": len(geom_windows["starts"]),
            "rollout_windows": len(rollout_windows["starts"]),
            "n_samples": args.n_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "geometry": {
            "raw_cosine": {
                k: v for k, v in raw_cos.items() if k != "matrix"
            },
            "centered_cosine": {
                k: v for k, v in centered_cos.items() if k != "matrix"
            },
            "raw_quintile_cosine": raw_quint_cos,
            "centered_quintile_cosine": centered_quint_cos,
            "pca": {
                "pc1_var_explained": pca["pc1_var_explained"],
                "pc2_var_explained": pca["pc2_var_explained"],
                "effective_rank": pca["effective_rank"],
                "top10_var_explained": pca["top10_var_explained"],
            },
        },
        "cond_proj": proj,
        "residual_signal": residual_signal,
        "rollout": {
            "modes": {
                mode: {
                    "mae_global": stats["mae_global"],
                    "per_window_mae_mean": stats["per_window_mae_mean"],
                    "per_window_mae_std": stats["per_window_mae_std"],
                }
                for mode, stats in rollout_modes.items()
            },
            "vs_normal": vs_normal,
        },
    }

    print_claim_summary(result)
    args.output_json.write_text(json.dumps(to_serializable(result), indent=2))
    print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
