import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
from vae.cvae_with_mem_randomized import CVAEMemRand

def extract_latents_from_data(model, surf_data, ex_data, ctx_len=5, start_day=0, num_days=None):
    """
    Extract latent codes (z_mean) from encoder for multiple days.

    Args:
        model: Trained CVAEMemRand model
        surf_data: Surface data array (N, 5, 5)
        ex_data: Extra features data (N, 3) or None
        ctx_len: Context length for teacher forcing
        start_day: Starting day index
        num_days: Number of days to process (default: all available)

    Returns:
        z_mean: (num_days, latent_dim) - latent mean for last timestep only
        z_log_var: (num_days, latent_dim) - latent log variance for last timestep only
    """
    if num_days is None:
        num_days = len(surf_data) - ctx_len

    latent_dim = model.config["latent_dim"]
    all_z_mean = np.zeros((num_days, latent_dim))
    all_z_log_var = np.zeros((num_days, latent_dim))

    model.eval()
    with torch.no_grad():
        for i, day in enumerate(range(start_day + ctx_len, start_day + ctx_len + num_days)):
            if day % 1000 == 0:
                print(f"  Processing day {day}/{start_day + ctx_len + num_days}")

            # Teacher forcing: use actual history [day-ctx_len:day+1]
            surface = torch.tensor(surf_data[day-ctx_len:day+1]).unsqueeze(0)  # (1, T, 5, 5)
            x = {"surface": surface.to(model.device)}

            if model.config["ex_feats_dim"] > 0 and ex_data is not None:
                ex_feats = torch.tensor(ex_data[day-ctx_len:day+1]).unsqueeze(0)  # (1, T, 3)
                x["ex_feats"] = ex_feats.to(model.device)

            # Get latent distribution parameters
            z_mean, z_log_var, _ = model.encoder(x)

            # Extract last timestep only (the prediction target)
            all_z_mean[i] = z_mean[0, -1, :].cpu().numpy()
            all_z_log_var[i] = z_log_var[0, -1, :].cpu().numpy()

    return all_z_mean, all_z_log_var


def compute_statistics(z_mean, model_name):
    """Compute and print statistics for latent dimensions."""
    latent_dim = z_mean.shape[1]

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    print(f"{'Dim':<6} {'Empirical Mean':<18} {'Empirical Std':<18}")
    print(f"{'-'*60}")

    for dim in range(latent_dim):
        mean = np.mean(z_mean[:, dim])
        std = np.std(z_mean[:, dim])
        print(f"{dim:<6} {mean:>8.4f} (Δ={mean:>6.4f}) {std:>8.4f} (σ²={std**2:>6.4f})")

    # Overall statistics
    print(f"{'-'*60}")
    overall_mean = np.mean(np.abs(z_mean))
    overall_std = np.mean(np.std(z_mean, axis=0))
    print(f"Overall: |μ|={overall_mean:.4f}, avg(σ)={overall_std:.4f}")


def plot_histogram_comparison(z_mean_dict, output_path="latent_distribution_comparison.png"):
    """
    Create histogram overlays comparing empirical latents vs N(0,1).

    Args:
        z_mean_dict: Dict of {model_name: z_mean_array}
        output_path: Path to save figure
    """
    model_names = list(z_mean_dict.keys())
    num_models = len(model_names)
    latent_dim = z_mean_dict[model_names[0]].shape[1]

    fig, axes = plt.subplots(num_models, latent_dim, figsize=(15, 3*num_models))
    if num_models == 1:
        axes = axes.reshape(1, -1)

    # X-axis range for plotting N(0,1)
    x = np.linspace(-4, 4, 1000)
    normal_pdf = norm.pdf(x, 0, 1)

    for i, model_name in enumerate(model_names):
        z_mean = z_mean_dict[model_name]

        for dim in range(latent_dim):
            ax = axes[i, dim]

            # Empirical histogram
            ax.hist(z_mean[:, dim], bins=50, density=True, alpha=0.6,
                   color='blue', label='Empirical')

            # N(0,1) overlay
            ax.plot(x, normal_pdf, 'r-', linewidth=2, label='N(0,1)')

            # Statistics
            emp_mean = np.mean(z_mean[:, dim])
            emp_std = np.std(z_mean[:, dim])

            # Title and labels
            ax.set_title(f"{model_name}\nDim {dim}: μ={emp_mean:.3f}, σ={emp_std:.3f}",
                        fontsize=9)
            ax.set_xlim(-4, 4)

            if dim == 0:
                ax.set_ylabel('Density', fontsize=9)
            if i == num_models - 1:
                ax.set_xlabel(f'Latent Dim {dim}', fontsize=9)
            if i == 0 and dim == latent_dim - 1:
                ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Histogram comparison saved to: {output_path}")


def main():
    print("="*60)
    print("Latent Distribution Analysis: Empirical vs N(0,1)")
    print("="*60)

    # Load data
    print("\n[1/4] Loading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    surf_data = data["surface"]
    ret_data = data["ret"]
    skew_data = data["skews"]
    slope_data = data["slopes"]
    ex_data = np.stack([ret_data, skew_data, slope_data], axis=-1)

    print(f"  Surface data: {surf_data.shape}")
    print(f"  Extra features: {ex_data.shape}")
    print(f"  Train: 0-4000, Test: 5000-5815")

    # Model configurations
    models_config = [
        {
            "name": "no_ex",
            "path": "test_spx/2024_11_09/no_ex.pt",
            "has_ex_feats": False
        },
        {
            "name": "ex_no_loss",
            "path": "test_spx/2024_11_09/ex_no_loss.pt",
            "has_ex_feats": True
        },
        {
            "name": "ex_loss",
            "path": "test_spx/2024_11_09/ex_loss.pt",
            "has_ex_feats": True
        }
    ]

    z_mean_dict = {}
    ctx_len = 5

    # Extract latents from all models
    for model_cfg in models_config:
        print(f"\n[2/4] Processing model: {model_cfg['name']}")

        # Load model
        print(f"  Loading {model_cfg['path']}...")
        model_data = torch.load(model_cfg['path'], weights_only=False)
        model_config = model_data["model_config"]
        model_config["mem_dropout"] = 0.  # Disable dropout for inference
        model = CVAEMemRand(model_config)
        model.load_weights(dict_to_load=model_data)
        model.eval()

        # Extract latents from training set
        print(f"  Extracting training latents (0-4000)...")
        train_z_mean, _ = extract_latents_from_data(
            model, surf_data[:4000],
            ex_data[:4000] if model_cfg['has_ex_feats'] else None,
            ctx_len=ctx_len,
            start_day=0,
            num_days=4000-ctx_len
        )

        # Extract latents from test set
        print(f"  Extracting test latents (5000-5815)...")
        test_z_mean, _ = extract_latents_from_data(
            model, surf_data[5000:],
            ex_data[5000:] if model_cfg['has_ex_feats'] else None,
            ctx_len=ctx_len,
            start_day=0,
            num_days=815-ctx_len
        )

        # Combine train + test
        z_mean_combined = np.concatenate([train_z_mean, test_z_mean], axis=0)
        z_mean_dict[model_cfg['name']] = z_mean_combined

        print(f"  ✓ Extracted {z_mean_combined.shape[0]} latent codes")

    # Compute and print statistics
    print(f"\n[3/4] Computing statistics...")
    for model_name, z_mean in z_mean_dict.items():
        compute_statistics(z_mean, model_name)

    # Generate visualization
    print(f"\n[4/4] Generating histogram comparison...")
    plot_histogram_comparison(z_mean_dict)

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    main()
