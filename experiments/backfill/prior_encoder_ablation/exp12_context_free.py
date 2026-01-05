"""
Experiment 12: Context-Free Decoder VAE

Test a VAE where decoder receives only z (no context embedding),
forcing z to carry all information about the target.

Hypothesis: The original VAE's decoder ignores z because it can use
context directly. By removing this shortcut, we force meaningful latent
representations and potentially achieve better conditional variance.

Key changes from CVAEMemRand:
1. Decoder input: z only (not concat(ctx_embedding, z))
2. KL: q(z|x) vs p(z|context) instead of q(z|x) vs N(0,1)
3. Separate prior encoder processes context only

Expected results:
- Non-zero decoder gain (forced to use z)
- Higher conditional variance ratio
- Better CI coverage (target: 90%)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import json
import scipy.stats as stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.experimental.cvae_context_free import CVAEContextFree


def get_config():
    """Get model configuration - same as CVAEMemRand."""
    return {
        # Architecture (same as existing)
        "feat_dim": (5, 5),
        "latent_dim": 12,
        "context_len": 20,
        "horizon": 1,
        "max_horizon": 90,

        # Encoder (same as existing)
        "surface_hidden": [32, 64, 128],
        "ex_feats_dim": 0,
        "ex_feats_hidden": None,
        "use_dense_surface": True,
        "padding": 1,
        "deconv_output_padding": 0,

        # Memory (same as existing)
        "mem_type": "lstm",
        "mem_hidden": 128,
        "mem_layers": 2,
        "mem_dropout": 0.1,

        # Context encoder (for prior)
        "ctx_surface_hidden": [32, 64],
        "ctx_ex_feats_hidden": None,
        "compress_context": True,
        "interaction_layers": 2,

        # Training
        "kl_weight": 0.001,
        "re_feat_weight": 0.0,
        "ex_loss_on_ret_only": False,
        "ex_feats_loss_type": "l1",

        # NEW: prior covariance type
        "covariance_type": "diagonal",  # or "full"

        # Device
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def prepare_data(surface, context_len, horizon=1):
    """Prepare training data sequences."""
    n_samples = len(surface) - context_len - horizon + 1
    sequences = []

    for i in range(n_samples):
        seq = surface[i:i + context_len + horizon]
        sequences.append(seq)

    return torch.stack(sequences)


def evaluate_coverage(model, surface, context_len, n_eval=500, n_samples=100):
    """
    Evaluate CI coverage using samples from prior.

    For each context, generate n_samples predictions and compute:
    1. Mean and std across samples
    2. 90% CI = mean ± 1.645 * std
    3. Check if target falls within CI
    """
    model.eval()
    device = next(model.parameters()).device
    z_score = stats.norm.ppf(0.95)  # 1.645 for 90% CI

    n_total = len(surface) - context_len - 1
    eval_indices = np.random.choice(n_total, min(n_eval, n_total), replace=False)

    coverages = []
    conditional_vars = []
    atm_ivs = []

    with torch.no_grad():
        for idx in tqdm(eval_indices, desc="Evaluating coverage"):
            context = surface[idx:idx + context_len].to(device)
            target = surface[idx + context_len].numpy()

            ctx_dict = {"surface": context.unsqueeze(0)}

            # Generate multiple samples from prior
            all_samples = []
            for _ in range(n_samples):
                pred = model.get_surface_given_conditions(ctx_dict, horizon=1)
                all_samples.append(pred.squeeze().cpu().numpy())

            all_samples = np.array(all_samples)  # (n_samples, 5, 5)

            # Compute statistics
            mean_pred = all_samples.mean(axis=0)
            std_pred = all_samples.std(axis=0)

            # 90% CI
            lower = mean_pred - z_score * std_pred
            upper = mean_pred + z_score * std_pred

            # Check coverage
            covered = (target >= lower) & (target <= upper)
            coverages.append(covered)

            # Track conditional variance
            conditional_vars.append(all_samples.var(axis=0).mean())
            atm_ivs.append(context[-1, 2, 2].cpu().item())

    coverages = np.array(coverages)
    conditional_vars = np.array(conditional_vars)
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

    # Conditional variance by regime
    low_var = conditional_vars[low_mask].mean() if low_mask.sum() > 0 else 0
    high_var = conditional_vars[high_mask].mean() if high_mask.sum() > 0 else 0
    var_ratio = high_var / low_var if low_var > 0 else 0

    return {
        "overall_coverage": overall_coverage,
        "low_coverage": low_coverage,
        "mid_coverage": mid_coverage,
        "high_coverage": high_coverage,
        "var_ratio": var_ratio,
        "mean_cond_var": conditional_vars.mean(),
    }


def evaluate_decoder_gain(model, surface, context_len, n_eval=100):
    """
    Evaluate decoder sensitivity to z (decoder gain).

    High gain = decoder is sensitive to z variations
    Low gain = decoder ignores z (the problem we're trying to solve)
    """
    model.eval()
    device = next(model.parameters()).device

    n_total = len(surface) - context_len - 1
    eval_indices = np.random.choice(n_total, min(n_eval, n_total), replace=False)

    z_vars = []
    output_vars = []

    with torch.no_grad():
        for idx in eval_indices:
            context = surface[idx:idx + context_len].to(device).unsqueeze(0)
            ctx_dict = {"surface": context}

            # Get prior distribution
            prior_mu, prior_logvar, _ = model.prior_encoder(ctx_dict)

            # Sample multiple z values and decode
            outputs = []
            z_samples = []
            for _ in range(50):
                eps = torch.randn_like(prior_mu[:, -1, :])
                z_sample = prior_mu[:, -1, :] + eps * torch.exp(0.5 * prior_logvar[:, -1, :])
                z_samples.append(z_sample.cpu().numpy())

                # Full z sequence (context + future)
                z_full = torch.zeros((1, context_len + 1, model.config["latent_dim"]), device=device)
                ctx_z, _, _ = model.encoder(ctx_dict)
                z_full[:, :context_len, :] = ctx_z
                z_full[:, context_len, :] = z_sample

                output = model.decoder(z_full)
                outputs.append(output[:, -1, :, :].cpu().numpy())

            z_samples = np.array(z_samples).squeeze()  # (50, latent_dim)
            outputs = np.array(outputs).squeeze()  # (50, 5, 5)

            z_vars.append(z_samples.var())
            output_vars.append(outputs.var())

    z_vars = np.array(z_vars)
    output_vars = np.array(output_vars)

    # Decoder gain = output variance / z variance
    decoder_gain = output_vars.mean() / (z_vars.mean() + 1e-10)

    return {
        "decoder_gain": decoder_gain,
        "mean_z_var": z_vars.mean(),
        "mean_output_var": output_vars.mean(),
    }


def main():
    print("=" * 70)
    print("EXPERIMENT 12: Context-Free Decoder VAE")
    print("=" * 70)

    # Load data
    data_file = Path("data/vol_surface_with_ret.npz")
    data = np.load(data_file)
    surface = torch.tensor(data["surface"], dtype=torch.float32)

    print(f"\nData shape: {surface.shape}")

    # GT statistics
    gt_mean = surface.mean().item()
    gt_var = surface.var().item()
    print(f"GT mean: {gt_mean:.4f}")
    print(f"GT var: {gt_var:.6f}")

    # Config
    config = get_config()
    context_len = config["context_len"]

    print(f"\nContext length: {context_len}")
    print(f"Latent dim: {config['latent_dim']}")
    print(f"Covariance type: {config['covariance_type']}")
    print(f"Device: {config['device']}")

    # Prepare data
    train_data = prepare_data(surface, context_len, horizon=1)
    print(f"Training samples: {len(train_data)}")

    # Create model
    print("\nInitializing model...")
    model = CVAEContextFree(config)
    model = model.to(config["device"])

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    # Training setup
    n_epochs = 50
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
        epoch_kl = []

        n_batches = len(train_data) // batch_size
        for i in range(n_batches):
            batch = train_data_shuffled[i * batch_size:(i + 1) * batch_size]
            x = {"surface": batch.to(config["device"])}

            loss_dict = model.train_step(x, optimizer)

            epoch_losses.append(loss_dict["loss"].item())
            epoch_mse.append(loss_dict["reconstruction_loss"].item())
            epoch_kl.append(loss_dict["kl_loss"].item())

        scheduler.step()

        # Epoch stats
        avg_loss = np.mean(epoch_losses)
        avg_mse = np.mean(epoch_mse)
        avg_kl = np.mean(epoch_kl)

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            eval_results = evaluate_coverage(model, surface, context_len, n_eval=200, n_samples=50)
            gain_results = evaluate_decoder_gain(model, surface, context_len, n_eval=50)

            print(f"Epoch {epoch + 1:3d} | Loss: {avg_loss:.4f} | MSE: {avg_mse:.6f} | "
                  f"KL: {avg_kl:.2f} | Cov: {eval_results['overall_coverage']:.1f}% | "
                  f"Gain: {gain_results['decoder_gain']:.2e}")

            if eval_results['overall_coverage'] > best_coverage:
                best_coverage = eval_results['overall_coverage']

            history.append({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "mse": avg_mse,
                "kl": avg_kl,
                **eval_results,
                **gain_results,
            })
        else:
            print(f"Epoch {epoch + 1:3d} | Loss: {avg_loss:.4f} | MSE: {avg_mse:.6f} | KL: {avg_kl:.2f}")

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    final_coverage = evaluate_coverage(model, surface, context_len, n_eval=500, n_samples=200)
    final_gain = evaluate_decoder_gain(model, surface, context_len, n_eval=100)

    # Compute conditional variance ratio
    cond_var_ratio = final_coverage['mean_cond_var'] / gt_var * 100

    print(f"""
Coverage (90% target):
  Overall:   {final_coverage['overall_coverage']:.1f}%
  Low vol:   {final_coverage['low_coverage']:.1f}%
  Mid vol:   {final_coverage['mid_coverage']:.1f}%
  High vol:  {final_coverage['high_coverage']:.1f}%

Conditional Variance:
  Mean cond var:     {final_coverage['mean_cond_var']:.6f}
  GT var:            {gt_var:.6f}
  Cond var ratio:    {cond_var_ratio:.2f}% (target: >2%)
  High/Low ratio:    {final_coverage['var_ratio']:.2f}x

Decoder Gain:
  Gain:              {final_gain['decoder_gain']:.2e}
  Mean z var:        {final_gain['mean_z_var']:.4f}
  Mean output var:   {final_gain['mean_output_var']:.6f}

Success Criteria:
  Coverage >= 88%:      {"[OK]" if final_coverage['overall_coverage'] >= 88 else "[FAIL]"}
  Cond var ratio > 2%:  {"[OK]" if cond_var_ratio > 2 else "[FAIL]"}
  Decoder gain > 1e-4:  {"[OK]" if final_gain['decoder_gain'] > 1e-4 else "[FAIL]"}
""")

    # Save results
    output_dir = Path("results/prior_encoder_ablation/exp12_context_free")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "state_dict": model.state_dict(),
        "model_config": config,
        "final_coverage": final_coverage,
        "final_gain": final_gain,
        "history": history,
    }, output_dir / "model.pt")

    # Convert numpy types
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
            "final_coverage": convert_to_python(final_coverage),
            "final_gain": convert_to_python(final_gain),
            "cond_var_ratio": cond_var_ratio,
            "gt_mean": float(gt_mean),
            "gt_var": float(gt_var),
            "n_epochs": n_epochs,
        }, f, indent=2)

    print(f"\nModel saved to: {output_dir}/model.pt")
    print(f"Results saved to: {output_dir}/results.json")


if __name__ == "__main__":
    main()
