"""
Generate surfaces for quantile regression models.
Generates both stochastic (z~N(0,1)) and MLE (z=0) versions.
"""
import numpy as np
import torch
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.utils import set_seeds
from generate_surfaces import generate_surfaces_multiday
from generate_surfaces_max_likelihood import generate_surfaces_multiday_mle

# Set seeds and dtype
set_seeds(0)
torch.set_default_dtype(torch.float64)

print("=" * 60)
print("GENERATING QUANTILE SURFACES")
print("=" * 60)

# Load data
print("\nLoading data...")
data = np.load("data/vol_surface_with_ret.npz")
vol_surf_data = data["surface"]
ret_data = data["ret"]
skew_data = data["skews"]
slope_data = data["slopes"]
ex_data = np.concatenate([ret_data[...,np.newaxis], skew_data[...,np.newaxis], slope_data[...,np.newaxis]], axis=-1)
print(f"   - Surface shape: {vol_surf_data.shape}")
print(f"   - Extra features shape: {ex_data.shape}")

# Generation parameters
ctx_len = 5
start_day = 5000  # Start from test set
num_days = vol_surf_data.shape[0] - start_day
num_samples = 1  # Only need 1 sample for quantile regression
model_dir = "test_spx/quantile_regression"

print(f"\nGeneration settings:")
print(f"   - Context length: {ctx_len}")
print(f"   - Start day: {start_day}")
print(f"   - Days to generate: {num_days}")
print(f"   - Model directory: {model_dir}")

# Model configurations
models = [
    {
        "name": "no_ex",
        "file": f"{model_dir}/no_ex.pt",
        "use_ex_feats": False,
        "check_ex_feats": False,
    },
    {
        "name": "ex_no_loss",
        "file": f"{model_dir}/ex_no_loss.pt",
        "use_ex_feats": True,
        "check_ex_feats": False,
    },
    {
        "name": "ex_loss",
        "file": f"{model_dir}/ex_loss.pt",
        "use_ex_feats": True,
        "check_ex_feats": True,
    },
]

print("\n" + "=" * 60)
print("GENERATING SURFACES")
print("=" * 60)

for i, model_config in enumerate(models):
    model_name = model_config["name"]
    model_file = model_config["file"]
    use_ex_feats = model_config["use_ex_feats"]
    check_ex_feats = model_config["check_ex_feats"]

    print(f"\n>>> Model {i+1}/3: {model_name}")
    print("-" * 60)

    # Load model
    print(f"   Loading model from {model_file}...")
    model_data = torch.load(model_file, weights_only=False)
    print(f"   ✓ Model loaded")

    # Generate stochastic surfaces (z ~ N(0,1))
    print(f"   Generating stochastic surfaces (z ~ N(0,1))...")
    output_file = f"{model_dir}/{model_name}_quantile_gen5.npz"
    (p05, p50, p95), ex_feats = generate_surfaces_multiday(
        model_data=model_data,
        ex_data=ex_data,
        vol_surface_data=vol_surf_data,
        start_day=start_day,
        days_to_generate=num_days,
        ctx_len=ctx_len,
        num_vaes=num_samples,
        model_type=CVAEMemRand,
        check_ex_feats=check_ex_feats,
    )

    # Save
    if check_ex_feats:
        np.savez(output_file, surfaces_p05=p05, surfaces_p50=p50, surfaces_p95=p95, ex_feats=ex_feats)
    else:
        np.savez(output_file, surfaces_p05=p05, surfaces_p50=p50, surfaces_p95=p95)
    print(f"   ✓ Saved to {output_file}")
    print(f"      - p05 shape: {p05.shape}")
    print(f"      - p50 shape: {p50.shape}")
    print(f"      - p95 shape: {p95.shape}")

    # Generate MLE surfaces (z = 0)
    print(f"   Generating MLE surfaces (z = 0)...")
    output_file_mle = f"{model_dir}/{model_name}_quantile_mle_gen5.npz"
    (p05_mle, p50_mle, p95_mle), ex_feats_mle = generate_surfaces_multiday_mle(
        model_data=model_data,
        ex_data=ex_data,
        vol_surface_data=vol_surf_data,
        start_day=start_day,
        days_to_generate=num_days,
        ctx_len=ctx_len,
        num_vaes=num_samples,
        model_type=CVAEMemRand,
        check_ex_feats=check_ex_feats,
    )

    # Save
    if check_ex_feats:
        np.savez(output_file_mle, surfaces_p05=p05_mle, surfaces_p50=p50_mle, surfaces_p95=p95_mle, ex_feats=ex_feats_mle)
    else:
        np.savez(output_file_mle, surfaces_p05=p05_mle, surfaces_p50=p50_mle, surfaces_p95=p95_mle)
    print(f"   ✓ Saved to {output_file_mle}")
    print(f"      - p05 shape: {p05_mle.shape}")
    print(f"      - p50 shape: {p50_mle.shape}")
    print(f"      - p95 shape: {p95_mle.shape}")

print("\n" + "=" * 60)
print("GENERATION COMPLETE")
print("=" * 60)
print("\nGenerated files:")
for model_config in models:
    model_name = model_config["name"]
    print(f"   - {model_dir}/{model_name}_quantile_gen5.npz (stochastic)")
    print(f"   - {model_dir}/{model_name}_quantile_mle_gen5.npz (MLE)")
print("\n✓ All surfaces generated successfully!")
print("=" * 60)
