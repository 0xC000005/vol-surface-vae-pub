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


def generate_surfaces_mle(model: CVAEMemRand, ex_data, vol_surface_data, day, ctx_len, num_vaes, use_ex_feats, check_ex_feats):
    """
    Generate maximum likelihood (z=0) quantile surfaces for a single day.

    With quantile regression: Returns 3 quantiles directly (p5, p50, p95) using z=0.
    num_vaes parameter is kept for compatibility but should be 1.

    Returns:
        Tuple of ((surf_p05, surf_p50, surf_p95), ex_feat)
    """
    torch.cuda.empty_cache()
    z = torch.zeros((1, ctx_len + 1, model.config["latent_dim"]))

    surf_data = torch.from_numpy(vol_surface_data[day - ctx_len:day])
    if use_ex_feats:
        ex_data_ctx = torch.from_numpy(ex_data[day - ctx_len:day])
        if len(ex_data_ctx.shape) == 1:
            ex_data_ctx = ex_data_ctx.unsqueeze(1)
        ctx_data = {
            "surface": surf_data.unsqueeze(0),  # (1, C, 5, 5)
            "ex_feats": ex_data_ctx.unsqueeze(0)  # (1, C, 3)
        }
    else:
        ctx_data = {
            "surface": surf_data.unsqueeze(0),  # (1, C, 5, 5)
        }

    # Generate quantiles with z=0 (maximum likelihood)
    if use_ex_feats:
        surf, ex_feat = model.get_surface_given_conditions(ctx_data, z=z)
        # surf: (1, 1, 3, 5, 5) - [batch, time, quantiles, H, W]
        ex_feat = ex_feat.detach().cpu().numpy().squeeze()  # (3,)
    else:
        surf = model.get_surface_given_conditions(ctx_data, z=z)
        ex_feat = None

    # Extract quantiles: (1, 1, 3, 5, 5) -> (3, 5, 5)
    surf = surf.detach().cpu().numpy().squeeze(0).squeeze(0)  # (3, 5, 5)

    surf_p05 = surf[0, :, :]  # (5, 5)
    surf_p50 = surf[1, :, :]  # (5, 5)
    surf_p95 = surf[2, :, :]  # (5, 5)

    if use_ex_feats and check_ex_feats:
        return (surf_p05, surf_p50, surf_p95), ex_feat
    else:
        return (surf_p05, surf_p50, surf_p95), None

def generate_surfaces_multiday_mle(model_data, ex_data, vol_surface_data,
                                start_day, days_to_generate, num_vaes,
                                model_type: Union[CVAE, CVAEMem, CVAEMemRand] = CVAEMemRand,
                                check_ex_feats=False, ctx_len=None):
    """
    Generate MLE quantile surfaces for multiple days.

    Returns:
        For quantile regression: ((p05_surfaces, p50_surfaces, p95_surfaces), ex_feats)
        where each p*_surfaces has shape (days_to_generate, 5, 5)
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
    print("use_ex_feats is: ",use_ex_feats)

    # Storage for 3 quantile surfaces
    all_day_surfaces_p05 = np.zeros((days_to_generate, vol_surface_data.shape[1], vol_surface_data.shape[2]))
    all_day_surfaces_p50 = np.zeros((days_to_generate, vol_surface_data.shape[1], vol_surface_data.shape[2]))
    all_day_surfaces_p95 = np.zeros((days_to_generate, vol_surface_data.shape[1], vol_surface_data.shape[2]))
    all_day_ex_feats = np.zeros((days_to_generate, ex_data.shape[1]))

    for day in range(start_day, start_day+days_to_generate):
        if day % 500 == 0:
            print(f"Generating day {day}")
        (p05, p50, p95), vae_ex_feats = generate_surfaces_mle(model, ex_data, vol_surface_data, day, ctx_len, num_vaes, use_ex_feats, check_ex_feats)

        all_day_surfaces_p05[day - start_day, ...] = p05
        all_day_surfaces_p50[day - start_day, ...] = p50
        all_day_surfaces_p95[day - start_day, ...] = p95

        if vae_ex_feats is not None:
            all_day_ex_feats[day - start_day, ...] = vae_ex_feats

    return (all_day_surfaces_p05, all_day_surfaces_p50, all_day_surfaces_p95), all_day_ex_feats

set_seeds(0)
torch.set_default_dtype(torch.float64)
num_epochs = 500
ctx_len = 5
start_day = 5
days_to_generate = 5810
num_vaes = 1

data = np.load("data/vol_surface_with_ret.npz")
vol_surf_data = data["surface"]
ret_data = data["ret"]
level_data = data["levels"]
skew_data = data["skews"]
slope_data = data["slopes"]
ex_data = np.concatenate([ret_data[...,np.newaxis], skew_data[...,np.newaxis], slope_data[...,np.newaxis]], axis=-1)

base_folder = "test_spx/2024_11_09"
for (file_path, use_ex, return_ex) in [
    (f"{base_folder}/no_ex.pt", False, False),
    (f"{base_folder}/ex_no_loss.pt", True, False),
    (f"{base_folder}/ex_loss.pt", True, True),
]:
    print(file_path)
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    model_data = torch.load(file_path, weights_only=False) # latent_dim=5, surface_hidden=[5,5,5], mem_hidden=100
    model_config = model_data["model_config"]
    model_config["mem_dropout"] = 0.
    model = CVAEMemRand(model_config)
    model.load_weights(dict_to_load=model_data)
    model.eval()
    print(model)

    i = ctx_len
    gen_fn = f"{base_folder}/{file_name}_mle_gen{i}.npz"
    if not os.path.exists(gen_fn):
        print(f"Generating MLE quantile surfaces for {file_name}...")
        if return_ex:
            (p05, p50, p95), ex_feats = generate_surfaces_multiday_mle(
                model_data=model_data,
                ex_data=ex_data, vol_surface_data=vol_surf_data,
                start_day=i, days_to_generate=days_to_generate, num_vaes=num_vaes,
                model_type=CVAEMemRand, check_ex_feats=return_ex, ctx_len=i)
            np.savez(gen_fn, surfaces_p05=p05, surfaces_p50=p50, surfaces_p95=p95, ex_feats=ex_feats)
            print(f"Saved: {gen_fn}")
            print(f"  - surfaces_p05: {p05.shape}")
            print(f"  - surfaces_p50: {p50.shape}")
            print(f"  - surfaces_p95: {p95.shape}")
            print(f"  - ex_feats: {ex_feats.shape}")
        else:
            (p05, p50, p95), _ = generate_surfaces_multiday_mle(
                model_data=model_data,
                ex_data=ex_data, vol_surface_data=vol_surf_data,
                start_day=i, days_to_generate=days_to_generate, num_vaes=num_vaes,
                model_type=CVAEMemRand, check_ex_feats=return_ex, ctx_len=i)
            np.savez(gen_fn, surfaces_p05=p05, surfaces_p50=p50, surfaces_p95=p95)
            print(f"Saved: {gen_fn}")
            print(f"  - surfaces_p05: {p05.shape}")
            print(f"  - surfaces_p50: {p50.shape}")
            print(f"  - surfaces_p95: {p95.shape}")
