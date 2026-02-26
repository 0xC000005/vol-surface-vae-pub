"""Diagnose what the frozen encoder features actually predict about uncertainty.

Key questions:
1. Do encoder features carry vol_of_vol info on validation data?
2. Is per-sample future sigma predictable from history at all?
3. What IS the right training target for a sigma head?
"""

import numpy as np
import torch
import dataclasses
from scipy import stats

from diffusion.block_ar.block_ar_ddpm import BlockARConfig, ConditionalBlockARDDPM, denormalize_iv
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset
from torch.utils.data import DataLoader


def main():
    # Load model
    cp = torch.load(
        "models/backfill/block_ar_vol_scaled_30ep/best_model.pt",
        map_location="cpu", weights_only=False,
    )
    config_dict = cp["config"]
    if dataclasses.is_dataclass(config_dict):
        config_dict = dataclasses.asdict(config_dict)
    model_config = BlockARConfig(**{
        k: v for k, v in config_dict.items()
        if k in {f.name for f in dataclasses.fields(BlockARConfig)}
    })
    model = ConditionalBlockARDDPM(model_config)
    model.load_state_dict(cp["model_state_dict"])
    model.eval()

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]

    results = {}
    for split_name, start, end in [("train", 0, 4040), ("val", 4040, 4540), ("test", 4540, None)]:
        ds = VolSurfaceDataset(surfaces, 30, 30, start_idx=start, end_idx=end)
        dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

        all_cond = []
        all_vol_of_vol = []
        all_vol_scale = []
        all_future_sigma = []
        all_recent_change = []

        with torch.no_grad():
            for batch in dl:
                history = batch["history"]
                future = batch["future"]
                B = history.shape[0]

                # Encoder features
                cond = model.encoder(history, mask=None)
                if model.config.forward_only:
                    cond = cond + model.encoder.null_embedding.expand(B, -1)
                all_cond.append(cond)

                # vol_of_vol (from history)
                past_abs = denormalize_iv(history)  # (B, 30, 5, 5)
                mean_iv = past_abs.mean(dim=(-1, -2))  # (B, 30)
                daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]  # (B, 29)
                vol = daily_chg.std(dim=1)  # (B,)
                all_vol_of_vol.append(vol)

                # vol_scale (hand-coded)
                vol_scale = (vol / model_config.global_mean_vol).clamp(0.5, 2.0)
                all_vol_scale.append(vol_scale)

                # recent_change (from history)
                recent = (mean_iv[:, -1] - mean_iv[:, -5]).abs()
                all_recent_change.append(recent)

                # Future sigma (empirical target)
                baseline = denormalize_iv(history[:, -1]).clamp(min=0.01).unsqueeze(1)
                future_abs = denormalize_iv(future).clamp(min=1e-4, max=1 - 1e-4)
                log_ratio = torch.log(future_abs / baseline)
                future_sigma = log_ratio.reshape(B, -1).std(dim=1)
                all_future_sigma.append(future_sigma)

        cond_all = torch.cat(all_cond)  # (N, 128)
        vov = torch.cat(all_vol_of_vol).numpy()
        vs = torch.cat(all_vol_scale).numpy()
        fs = torch.cat(all_future_sigma).numpy()
        rc = torch.cat(all_recent_change).numpy()

        # 1. Can we predict vol_of_vol from encoder features?
        # Simple linear probe: fit linear regression from features → vol_of_vol
        X = cond_all.numpy()
        from sklearn.linear_model import Ridge
        reg = Ridge(alpha=1.0).fit(X, vov)
        vov_pred = reg.predict(X)
        r2_vov = 1 - np.sum((vov - vov_pred)**2) / np.sum((vov - np.mean(vov))**2)

        # 2. Can we predict future_sigma from encoder features?
        reg_fs = Ridge(alpha=1.0).fit(X, fs)
        fs_pred = reg_fs.predict(X)
        r2_fs = 1 - np.sum((fs - fs_pred)**2) / np.sum((fs - np.mean(fs))**2)

        # 3. Correlation between conditioning variables and future sigma
        corr_vov_fs = stats.spearmanr(vov, fs)[0]
        corr_vs_fs = stats.spearmanr(vs, fs)[0]
        corr_rc_fs = stats.spearmanr(rc, fs)[0]

        # 4. Q5/Q1 of future sigma by vol_of_vol quintile
        vov_q = np.quantile(vov, [0.2, 0.8])
        q1_mask = vov <= vov_q[0]
        q5_mask = vov >= vov_q[1]
        q5q1_fs = fs[q5_mask].mean() / fs[q1_mask].mean()

        # 5. Q5/Q1 of vol_scale by vol_of_vol quintile
        q5q1_vs = vs[q5_mask].mean() / vs[q1_mask].mean()

        print(f"\n{'='*60}")
        print(f"Split: {split_name} (N={len(vov)})")
        print(f"{'='*60}")
        print(f"  R² encoder→vol_of_vol (linear probe): {r2_vov:.3f}")
        print(f"  R² encoder→future_sigma (linear probe): {r2_fs:.3f}")
        print(f"  Spearman(vol_of_vol, future_sigma): {corr_vov_fs:.3f}")
        print(f"  Spearman(vol_scale, future_sigma): {corr_vs_fs:.3f}")
        print(f"  Spearman(recent_change, future_sigma): {corr_rc_fs:.3f}")
        print(f"  Q5/Q1 of future_sigma (by vol_of_vol): {q5q1_fs:.3f}")
        print(f"  Q5/Q1 of vol_scale (by vol_of_vol): {q5q1_vs:.3f}")
        print(f"  future_sigma mean={fs.mean():.4f}, std={fs.std():.4f}")
        print(f"  vol_scale mean={vs.mean():.4f}, std={vs.std():.4f}")
        print(f"  vol_of_vol mean={vov.mean():.4f}, std={vov.std():.4f}")

        results[split_name] = {
            "r2_vov": r2_vov,
            "r2_fs": r2_fs,
            "corr_vov_fs": corr_vov_fs,
            "q5q1_fs": q5q1_fs,
            "q5q1_vs": q5q1_vs,
        }

    # Cross-split comparison
    print(f"\n{'='*60}")
    print("Cross-split comparison")
    print(f"{'='*60}")
    for metric in ["r2_vov", "r2_fs", "corr_vov_fs", "q5q1_fs", "q5q1_vs"]:
        vals = [f"{results[s][metric]:.3f}" for s in ["train", "val", "test"]]
        print(f"  {metric:25s}: train={vals[0]}, val={vals[1]}, test={vals[2]}")


if __name__ == "__main__":
    main()
