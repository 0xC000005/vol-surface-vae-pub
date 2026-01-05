"""
Experiment 11: Heteroscedastic VAE

Train a VAE with heteroscedastic decoder that learns context-dependent variance.

Key changes from baseline:
1. Decoder outputs (mean, log_var) instead of just surface
2. Reconstruction loss: Gaussian NLL instead of MSE
3. Generation: Sample from N(mean, exp(log_var))

Expected results:
- High vol regimes → higher predicted variance
- Low vol regimes → lower predicted variance
- 90% CI coverage from learned variance
- Marginal matching preserved
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_heteroscedastic import CVAEHeteroscedastic
from vae.utils import train


def get_config():
    """Get model configuration."""
    return {
        # Architecture
        "feat_dim": (5, 5),
        "latent_dim": 12,
        "context_len": 60,
        "horizon": 1,
        "max_horizon": 90,

        # Encoder
        "surface_hidden": [32, 64, 128],
        "ex_feats_dim": 0,
        "ex_feats_hidden": None,
        "use_dense_surface": True,
        "padding": 1,
        "deconv_output_padding": 0,

        # Memory
        "mem_type": "lstm",
        "mem_hidden": 128,
        "mem_layers": 2,
        "mem_dropout": 0.1,

        # Context encoder
        "ctx_surface_hidden": [32, 64],
        "ctx_ex_feats_hidden": None,
        "compress_context": True,
        "interaction_layers": 2,

        # Training
        "kl_weight": 0.001,
        "re_feat_weight": 0.0,
        "ex_loss_on_ret_only": False,

        # Loss weights for MSE + NLL training
        "mse_weight": 1.0,      # Weight for mean accuracy (MSE)
        "nll_weight": 0.1,      # Weight for variance calibration (NLL with detached mean)

        # Variance regularization - need VERY high variance to cover prior mismatch
        # With 41% coverage at var=0.8, we need ~5x more for 90%
        "var_reg_weight": 0.5,  # Stronger push toward target
        "target_var": 3.0,      # 300x GT var - aiming for 90% coverage
        "min_var": 0.001,       # Minimum allowed variance

        # Variance ratio regularization - DISABLED for stable training
        # The ratio loss only pushes ratio UP, causing instability
        "var_ratio_weight": 0.0,  # Disabled - causes unstable training
        "target_var_ratio": 2.0,  # Not used when weight=0

        # Context-based variance (independent of z - key for regime-dependent variance at inference)
        "use_context_variance": True,

        # Full covariance prior
        "mean_network_type": "mlp",
        "use_position_encoding": None,
        "full_cov_hidden_dims": [128, 128],
        "full_cov_dropout": 0.1,
        "full_cov_init_phi": 0.5,
        "full_cov_init_sigma_sq": 1.0,

        # Device
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def prepare_data(surface, context_len, horizon=1):
    """Prepare training data."""
    n_samples = len(surface) - context_len - horizon + 1
    contexts = []
    targets = []

    for i in range(n_samples):
        ctx = surface[i:i + context_len + horizon]
        contexts.append(ctx)

    return torch.stack(contexts)


def evaluate_coverage(model, surface, context_len, n_eval=500, n_samples=100):
    """Evaluate CI coverage using decoder's predicted variance directly.

    For heteroscedastic VAE, the CI comes from the decoder's (mean, log_var) output:
    1. Sample z from prior p(z|context)
    2. Decode to get (mean, log_var)
    3. CI = mean ± 1.645 * sqrt(exp(log_var)) for 90% coverage

    We also use Monte Carlo over z to account for prior uncertainty.
    """
    import scipy.stats as stats

    model.eval()
    device = next(model.parameters()).device
    z_score = stats.norm.ppf(0.95)  # 1.645 for 90% CI

    n_total = len(surface) - context_len - 1
    eval_indices = np.random.choice(n_total, min(n_eval, n_total), replace=False)

    coverages = []
    decoder_vars = []  # Track decoder's predicted variance (from log_var)
    atm_ivs = []

    with torch.no_grad():
        for idx in tqdm(eval_indices, desc="Evaluating coverage"):
            context = surface[idx:idx + context_len].to(device)
            target = surface[idx + context_len].numpy()

            ctx_dict = {"surface": context.unsqueeze(0)}

            # Sample multiple z's and get decoder (mean, var) for each
            all_means = []
            all_vars = []

            for _ in range(n_samples):
                # Encode context
                ctx_encoder_input = {"surface": ctx_dict["surface"]}
                ctx_embedding = model.ctx_encoder(ctx_encoder_input)
                context_summary = ctx_embedding[:, -1, :]

                # Sample z from prior
                z_future = model.full_cov_prior.sample(context_summary, horizon=1)

                # Decode with context embedding
                ctx_embedding_dim = ctx_embedding.shape[2]
                decoder_ctx = context_summary.unsqueeze(1)  # (1, 1, ctx_dim)
                decoder_input = torch.cat([z_future, decoder_ctx], dim=-1)

                surface_mean, decoder_logvar = model.decoder(decoder_input)

                # Use context-based variance if available (independent of z)
                if hasattr(model, 'use_context_variance') and model.use_context_variance:
                    ctx_logvar = model.context_variance_net(context_summary)  # (1, 25)
                    surface_logvar = ctx_logvar.view(1, 1, 5, 5)
                else:
                    surface_logvar = decoder_logvar

                all_means.append(surface_mean.squeeze().cpu().numpy())
                all_vars.append(torch.exp(surface_logvar).squeeze().cpu().numpy())

            all_means = np.array(all_means)  # (n_samples, 5, 5)
            all_vars = np.array(all_vars)   # (n_samples, 5, 5)

            # Combined CI: accounts for both z uncertainty and decoder variance
            # mean of means ± sqrt(variance of means + mean of variances) * z_score
            mean_of_means = all_means.mean(axis=0)
            var_of_means = all_means.var(axis=0)  # z sampling variance
            mean_of_vars = all_vars.mean(axis=0)  # decoder predicted variance

            # Total uncertainty = z variance + decoder variance
            total_var = var_of_means + mean_of_vars
            total_std = np.sqrt(total_var)

            lower = mean_of_means - z_score * total_std
            upper = mean_of_means + z_score * total_std

            # Check coverage
            covered = (target >= lower) & (target <= upper)
            coverages.append(covered)

            # Track decoder's predicted variance by regime
            decoder_vars.append(mean_of_vars.mean())  # Mean of decoder variance
            atm_ivs.append(context[-1, 2, 2].cpu().item())

    coverages = np.array(coverages)
    decoder_vars = np.array(decoder_vars)
    atm_ivs = np.array(atm_ivs)

    # Overall coverage
    overall_coverage = coverages.mean() * 100

    # Coverage by regime
    vol_terciles = np.percentile(atm_ivs, [33, 67])
    low_mask = atm_ivs < vol_terciles[0]
    mid_mask = (atm_ivs >= vol_terciles[0]) & (atm_ivs < vol_terciles[1])
    high_mask = atm_ivs >= vol_terciles[1]

    low_coverage = coverages[low_mask].mean() * 100 if low_mask.sum() > 0 else 0
    mid_coverage = coverages[mid_mask].mean() * 100 if mid_mask.sum() > 0 else 0
    high_coverage = coverages[high_mask].mean() * 100 if high_mask.sum() > 0 else 0

    # Decoder variance by regime (this is what we want to be regime-dependent)
    low_var = decoder_vars[low_mask].mean() if low_mask.sum() > 0 else 0
    high_var = decoder_vars[high_mask].mean() if high_mask.sum() > 0 else 0
    var_ratio = high_var / low_var if low_var > 0 else 0

    return {
        "overall_coverage": overall_coverage,
        "low_coverage": low_coverage,
        "mid_coverage": mid_coverage,
        "high_coverage": high_coverage,
        "var_ratio": var_ratio,
        "mean_pred_var": decoder_vars.mean(),
    }


def main():
    print("=" * 70)
    print("EXPERIMENT 11: Heteroscedastic VAE")
    print("=" * 70)

    # Load data
    data_file = Path("data/vol_surface_with_ret.npz")
    data = np.load(data_file)
    surface = torch.tensor(data["surface"], dtype=torch.float32)

    print(f"\nData shape: {surface.shape}")

    # GT statistics for reference
    gt_mean = surface.mean().item()
    gt_var = surface.var().item()
    print(f"GT mean: {gt_mean:.4f}")
    print(f"GT var: {gt_var:.6f}")

    # Config
    config = get_config()
    context_len = config["context_len"]

    print(f"\nContext length: {context_len}")
    print(f"Device: {config['device']}")

    # Prepare data
    train_data = prepare_data(surface, context_len, horizon=1)
    print(f"Training samples: {len(train_data)}")

    # Create model
    print("\nInitializing model...")
    model = CVAEHeteroscedastic(config)
    model = model.to(config["device"])

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    # Training setup
    n_epochs = 30  # Reduced for faster iteration
    batch_size = 64
    lr = 1e-4

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Training loop
    print(f"\nTraining for {n_epochs} epochs...")
    print("-" * 70)

    best_coverage = 0
    history = []

    for epoch in range(n_epochs):
        model.train()

        # Shuffle data
        perm = torch.randperm(len(train_data))
        train_data_shuffled = train_data[perm]

        epoch_losses = []
        epoch_mse = []
        epoch_nll = []
        epoch_kl = []
        epoch_var_reg = []
        epoch_var_ratio = []
        epoch_pred_var = []
        epoch_pred_ratio = []

        n_batches = len(train_data) // batch_size
        for i in range(n_batches):
            batch = train_data_shuffled[i * batch_size:(i + 1) * batch_size]
            x = {"surface": batch.to(config["device"])}

            loss_dict = model.train_step(x, optimizer)

            epoch_losses.append(loss_dict["loss"].item())
            epoch_mse.append(loss_dict["mse_loss"].item())
            epoch_nll.append(loss_dict["nll_loss"].item())
            epoch_kl.append(loss_dict["kl_loss"].item())
            epoch_var_reg.append(loss_dict["var_reg_loss"].item())
            epoch_var_ratio.append(loss_dict["var_ratio_loss"].item())
            epoch_pred_var.append(loss_dict["pred_var"].item())
            epoch_pred_ratio.append(loss_dict["pred_var_ratio"].item())

        scheduler.step()

        # Epoch stats
        avg_loss = np.mean(epoch_losses)
        avg_mse = np.mean(epoch_mse)
        avg_nll = np.mean(epoch_nll)
        avg_kl = np.mean(epoch_kl)
        avg_var_reg = np.mean(epoch_var_reg)
        avg_var_ratio_loss = np.mean(epoch_var_ratio)
        avg_pred_var = np.mean(epoch_pred_var)
        avg_pred_ratio = np.mean(epoch_pred_ratio)

        # Evaluate every 10 epochs (fewer samples for speed during training)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            eval_results = evaluate_coverage(model, surface, context_len, n_eval=200, n_samples=50)

            print(f"Epoch {epoch + 1:3d} | Loss: {avg_loss:.4f} | MSE: {avg_mse:.6f} | "
                  f"Var: {avg_pred_var:.4f} | Ratio: {avg_pred_ratio:.2f}x | "
                  f"Cov: {eval_results['overall_coverage']:.1f}% | "
                  f"EvalRatio: {eval_results['var_ratio']:.2f}x")

            if eval_results['overall_coverage'] > best_coverage:
                best_coverage = eval_results['overall_coverage']

            history.append({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "mse": avg_mse,
                "nll": avg_nll,
                "kl": avg_kl,
                "var_reg": avg_var_reg,
                "var_ratio_loss": avg_var_ratio_loss,
                "pred_var": avg_pred_var,
                "train_var_ratio": avg_pred_ratio,
                **eval_results
            })
        else:
            print(f"Epoch {epoch + 1:3d} | Loss: {avg_loss:.4f} | MSE: {avg_mse:.6f} | "
                  f"Var: {avg_pred_var:.4f} | Ratio: {avg_pred_ratio:.2f}x")

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    final_results = evaluate_coverage(model, surface, context_len, n_eval=500, n_samples=200)

    print(f"""
Coverage (90% target):
  Overall:   {final_results['overall_coverage']:.1f}%
  Low vol:   {final_results['low_coverage']:.1f}%
  Mid vol:   {final_results['mid_coverage']:.1f}%
  High vol:  {final_results['high_coverage']:.1f}%

Variance Learning:
  Mean predicted var: {final_results['mean_pred_var']:.6f}
  GT var:             {gt_var:.6f}
  Ratio:              {final_results['mean_pred_var'] / gt_var:.2%}
  High/Low var ratio: {final_results['var_ratio']:.2f}x (target: ~2x)

Success Criteria:
  Coverage ≥ 88%:     {"[OK]" if final_results['overall_coverage'] >= 88 else "[FAIL]"}
  Var ratio > 1.5x:   {"[OK]" if final_results['var_ratio'] > 1.5 else "[FAIL]"}
""")

    # Save model
    output_dir = Path("results/prior_encoder_ablation/exp11_heteroscedastic")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "state_dict": model.state_dict(),
        "model_config": config,
        "final_results": final_results,
        "history": history,
    }, output_dir / "model.pt")

    # Save results (convert numpy types to Python types)
    def convert_to_python(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_python(v) for k, v in obj.items()}
        return obj

    with open(output_dir / "results.json", "w") as f:
        json.dump({
            "final_results": convert_to_python(final_results),
            "gt_mean": float(gt_mean),
            "gt_var": float(gt_var),
            "n_epochs": n_epochs,
        }, f, indent=2)

    print(f"\nModel saved to: {output_dir}/model.pt")
    print(f"Results saved to: {output_dir}/results.json")


if __name__ == "__main__":
    main()
