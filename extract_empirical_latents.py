import numpy as np
import torch
from vae.cvae_with_mem_randomized import CVAEMemRand

def extract_future_latents(model, surf_data, ex_data, ctx_len=5):
    """
    Extract latent codes (z_mean) for the future timestep from all available days.

    Args:
        model: Trained CVAEMemRand model
        surf_data: Surface data array (N, 5, 5)
        ex_data: Extra features data (N, 3) or None
        ctx_len: Context length for teacher forcing

    Returns:
        z_mean_future: (num_days, latent_dim) - latent mean for future timestep only
    """
    num_days = len(surf_data) - ctx_len
    latent_dim = model.config["latent_dim"]
    z_mean_future = np.zeros((num_days, latent_dim))

    model.eval()
    print(f"  Extracting latents from {num_days} days...")

    with torch.no_grad():
        for i in range(num_days):
            day = i + ctx_len
            if day % 1000 == 0:
                print(f"    Processing day {day}/{num_days + ctx_len}")

            # Teacher forcing: use actual history [day-ctx_len:day+1]
            surface = torch.tensor(surf_data[day-ctx_len:day+1]).unsqueeze(0)  # (1, T, 5, 5)
            x = {"surface": surface.to(model.device)}

            if model.config["ex_feats_dim"] > 0 and ex_data is not None:
                ex_feats = torch.tensor(ex_data[day-ctx_len:day+1]).unsqueeze(0)  # (1, T, 3)
                x["ex_feats"] = ex_feats.to(model.device)

            # Get latent distribution parameters
            z_mean, z_log_var, _ = model.encoder(x)

            # Extract last timestep only (the future timestep we want to predict)
            z_mean_future[i] = z_mean[0, -1, :].cpu().numpy()

    return z_mean_future


def main():
    print("="*70)
    print("Extracting Empirical Latent Libraries (Per-Model)")
    print("="*70)

    # Load data
    print("\n[1/4] Loading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    surf_data = data["surface"]
    ret_data = data["ret"]
    skew_data = data["skews"]
    slope_data = data["slopes"]
    ex_data = np.stack([ret_data, skew_data, slope_data], axis=-1)

    print(f"  Surface data: {surf_data.shape}")
    print(f"  Total days available: {len(surf_data)}")

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

    ctx_len = 5
    base_folder = "test_spx/2024_11_09"

    # Extract latents for each model independently
    for i, model_cfg in enumerate(models_config):
        print(f"\n[{i+2}/4] Processing model: {model_cfg['name']}")
        print(f"  Loading {model_cfg['path']}...")

        # Load model
        model_data = torch.load(model_cfg['path'], weights_only=False)
        model_config = model_data["model_config"]
        model_config["mem_dropout"] = 0.  # Disable dropout for inference
        model = CVAEMemRand(model_config)
        model.load_weights(dict_to_load=model_data)
        model.eval()

        print(f"  Model latent_dim: {model.config['latent_dim']}")
        print(f"  Model has ex_feats: {model.config['ex_feats_dim'] > 0}")

        # Extract future timestep latents
        z_mean_future = extract_future_latents(
            model,
            surf_data,
            ex_data if model_cfg['has_ex_feats'] else None,
            ctx_len=ctx_len
        )

        # Save to model-specific file
        output_path = f"{base_folder}/{model_cfg['name']}_empirical_latents.npz"
        np.savez(
            output_path,
            z_mean_pool=z_mean_future,
            ctx_len=ctx_len,
            latent_dim=model.config['latent_dim']
        )

        print(f"  ✓ Extracted {z_mean_future.shape[0]} latent codes")
        print(f"  ✓ Saved to: {output_path}")
        print(f"     Shape: {z_mean_future.shape}")
        print(f"     Stats: mean={np.mean(z_mean_future):.4f}, std={np.std(z_mean_future):.4f}")

    print("\n" + "="*70)
    print("Extraction Complete!")
    print("="*70)
    print("\nSummary of extracted latent libraries:")
    for model_cfg in models_config:
        print(f"  {model_cfg['name']:12s} → {base_folder}/{model_cfg['name']}_empirical_latents.npz")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    main()
