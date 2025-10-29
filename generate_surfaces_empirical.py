import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader
from vae.datasets_randomized import VolSurfaceDataSetRand, CustomBatchSampler
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.utils import *
from eval_scripts.eval_utils import *
import os, sys


def generate_surfaces_empirical(model: CVAEMemRand, ex_data, vol_surface_data, day, ctx_len,
                                 num_vaes, z_mean_pool, noise_scale, use_ex_feats, check_ex_feats):
    """
    Generate surfaces using empirical latent sampling instead of N(0,1).

    Args:
        model: Trained CVAE model
        ex_data: Extra features data
        vol_surface_data: Volatility surface data
        day: Current day index
        ctx_len: Context length
        num_vaes: Number of samples to generate
        z_mean_pool: Empirical latent pool (N, latent_dim) from training data
        noise_scale: Noise multiplier (0.0 = pure empirical, higher = more noise)
        use_ex_feats: Whether model uses extra features
        check_ex_feats: Whether to return extra features

    Returns:
        vae_surfaces: (num_vaes, 5, 5) generated surfaces
        vae_ex_feats: (num_vaes, 3) generated extra features (if applicable)
    """
    torch.cuda.empty_cache()
    vae_surfaces = np.zeros((num_vaes, vol_surface_data.shape[1], vol_surface_data.shape[2]))
    if use_ex_feats and check_ex_feats:
        vae_ex_feats = np.zeros((num_vaes, ex_data.shape[1]))
    else:
        vae_ex_feats = None

    # Prepare context data
    surf_data = torch.from_numpy(vol_surface_data[day - ctx_len:day])
    if use_ex_feats:
        ex_data_ctx = torch.from_numpy(ex_data[day - ctx_len:day])
        if len(ex_data_ctx.shape) == 1:
            ex_data_ctx = ex_data_ctx.unsqueeze(1)
        ctx_data = {
            "surface": surf_data.unsqueeze(0).repeat(num_vaes, 1, 1, 1), # (B, T, 5, 5)
            "ex_feats": ex_data_ctx.unsqueeze(0).repeat(num_vaes, 1, 1) # (B, T, 3)
        }
    else:
        ctx_data = {
            "surface": surf_data.unsqueeze(0).repeat(num_vaes, 1, 1, 1), # (B, T, 5, 5)
        }

    # Sample from empirical latent pool + add noise
    random_indices = np.random.choice(len(z_mean_pool), size=num_vaes, replace=True)
    empirical_z_samples = z_mean_pool[random_indices]  # (num_vaes, latent_dim)

    # Add Gaussian noise
    if noise_scale > 0:
        noise = np.random.randn(num_vaes, model.config["latent_dim"]) * noise_scale
        empirical_z_samples = empirical_z_samples + noise

    # Prepare z tensor for model (B, T, latent_dim)
    # Context timesteps will be overwritten by model's encoder, only future timestep matters
    z = torch.from_numpy(empirical_z_samples).unsqueeze(1).repeat(1, ctx_len + 1, 1)  # (B, T, latent_dim)

    # Generate with pre-sampled z
    if use_ex_feats:
        surf, ex_feat = model.get_surface_given_conditions(ctx_data, z=z)
        ex_feat = ex_feat.detach().cpu().numpy().reshape((num_vaes, ex_data.shape[1],))
    else:
        surf = model.get_surface_given_conditions(ctx_data, z=z)

    surf = surf.detach().cpu().numpy().reshape((num_vaes, vol_surface_data.shape[1], vol_surface_data.shape[2]))
    vae_surfaces[...] = surf
    if use_ex_feats and check_ex_feats:
        vae_ex_feats[...] = ex_feat

    return vae_surfaces, vae_ex_feats


def generate_surfaces_multiday_empirical(model_data, ex_data, vol_surface_data,
                                         start_day, days_to_generate, num_vaes,
                                         z_mean_pool, noise_scale=0.3,
                                         model_type=CVAEMemRand,
                                         check_ex_feats=False, ctx_len=None):
    """
    Multi-day generation with empirical latent sampling.
    """
    model_config = model_data["model_config"]
    model = model_type(model_config)
    model.load_weights(dict_to_load=model_data)
    if ctx_len is None:
        seq_len = model_config["seq_len"]
        ctx_len = model_config["ctx_len"]

    if "ex_feats_dim" in model_config:
        use_ex_feats = model_config["ex_feats_dim"] > 0
    else:
        use_ex_feats = False
    print(f"  use_ex_feats: {use_ex_feats}")
    print(f"  noise_scale: {noise_scale}")

    all_day_surfaces = np.zeros((days_to_generate, num_vaes, vol_surface_data.shape[1], vol_surface_data.shape[2]))
    all_day_ex_feats = np.zeros((days_to_generate, num_vaes, ex_data.shape[1]))

    for day in range(start_day, start_day+days_to_generate):
        if day % 500 == 0:
            print(f"  Generating day {day}/{start_day+days_to_generate}")
        vae_surfaces, vae_ex_feats = generate_surfaces_empirical(
            model, ex_data, vol_surface_data, day, ctx_len, num_vaes,
            z_mean_pool, noise_scale, use_ex_feats, check_ex_feats
        )
        all_day_surfaces[day - start_day, ...] = vae_surfaces
        if vae_ex_feats is not None:
            all_day_ex_feats[day - start_day, ...] = vae_ex_feats

    return all_day_surfaces, all_day_ex_feats


# Main generation script
set_seeds(0)
torch.set_default_dtype(torch.float64)

# Generation parameters
ctx_len = 5
start_day = 5
days_to_generate = 5810
num_vaes = 1000
noise_scale = 0.3  # Default noise level (tunable parameter)

# Load data
print("="*70)
print("Empirical Latent Sampling Surface Generation")
print("="*70)
print(f"\nGeneration parameters:")
print(f"  Context length: {ctx_len}")
print(f"  Start day: {start_day}")
print(f"  Days to generate: {days_to_generate}")
print(f"  Samples per day: {num_vaes}")
print(f"  Noise scale: {noise_scale}")

print("\nLoading data...")
data = np.load("data/vol_surface_with_ret.npz")
vol_surf_data = data["surface"]
ret_data = data["ret"]
level_data = data["levels"]
skew_data = data["skews"]
slope_data = data["slopes"]
ex_data = np.concatenate([ret_data[...,np.newaxis], skew_data[...,np.newaxis], slope_data[...,np.newaxis]], axis=-1)

base_folder = "test_spx/2024_11_09"

# Model configurations
models_config = [
    {
        "path": f"{base_folder}/no_ex.pt",
        "name": "no_ex",
        "use_ex": False,
        "return_ex": False
    },
    {
        "path": f"{base_folder}/ex_no_loss.pt",
        "name": "ex_no_loss",
        "use_ex": True,
        "return_ex": False
    },
    {
        "path": f"{base_folder}/ex_loss.pt",
        "name": "ex_loss",
        "use_ex": True,
        "return_ex": True
    }
]

# Generate for each model
for model_cfg in models_config:
    print(f"\n{'='*70}")
    print(f"Processing model: {model_cfg['name']}")
    print(f"{'='*70}")

    file_path = model_cfg["path"]
    file_name = model_cfg["name"]

    # Load model
    print(f"\nLoading model from {file_path}...")
    model_data = torch.load(file_path, weights_only=False)
    model_config = model_data["model_config"]
    model_config["mem_dropout"] = 0.
    model = CVAEMemRand(model_config)
    model.load_weights(dict_to_load=model_data)
    model.eval()

    # Load empirical latents for this model
    empirical_latents_path = f"{base_folder}/{file_name}_empirical_latents.npz"
    print(f"Loading empirical latents from {empirical_latents_path}...")
    empirical_latents = np.load(empirical_latents_path)
    z_mean_pool = empirical_latents["z_mean_pool"]
    print(f"  Latent pool shape: {z_mean_pool.shape}")
    print(f"  Latent pool stats: mean={np.mean(z_mean_pool):.4f}, std={np.std(z_mean_pool):.4f}")

    # Generate surfaces
    gen_fn = f"{base_folder}/{file_name}_empirical_gen{ctx_len}_noise{noise_scale}.npz"
    if not os.path.exists(gen_fn):
        print(f"\nGenerating surfaces...")
        if model_cfg["return_ex"]:
            surfaces, ex_feats = generate_surfaces_multiday_empirical(
                model_data=model_data,
                ex_data=ex_data,
                vol_surface_data=vol_surf_data,
                start_day=ctx_len,
                days_to_generate=days_to_generate,
                num_vaes=num_vaes,
                z_mean_pool=z_mean_pool,
                noise_scale=noise_scale,
                model_type=CVAEMemRand,
                check_ex_feats=model_cfg["return_ex"],
                ctx_len=ctx_len
            )
            np.savez(gen_fn, surfaces=surfaces, ex_feats=ex_feats)
        else:
            surfaces, _ = generate_surfaces_multiday_empirical(
                model_data=model_data,
                ex_data=ex_data,
                vol_surface_data=vol_surf_data,
                start_day=ctx_len,
                days_to_generate=days_to_generate,
                num_vaes=num_vaes,
                z_mean_pool=z_mean_pool,
                noise_scale=noise_scale,
                model_type=CVAEMemRand,
                check_ex_feats=model_cfg["return_ex"],
                ctx_len=ctx_len
            )
            np.savez(gen_fn, surfaces=surfaces)

        print(f"  ✓ Saved to: {gen_fn}")
        print(f"  Surface shape: {surfaces.shape}")
    else:
        print(f"\n  ✓ Already exists: {gen_fn}")

print("\n" + "="*70)
print("Generation Complete!")
print("="*70)
print(f"\nGenerated files:")
for model_cfg in models_config:
    gen_fn = f"{base_folder}/{model_cfg['name']}_empirical_gen{ctx_len}_noise{noise_scale}.npz"
    print(f"  {gen_fn}")
