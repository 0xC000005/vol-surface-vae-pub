import numpy as np
import pandas as pd
import torch
import re
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.utils import set_seeds

# Configuration
BASE_MODEL_DIR = "test_spx/2024_11_09"
OUTPUT_FILE = "tables/2024_1213/loss_comparison.txt"

# Teacher forcing period (last 60 days)
TEACHER_FORCING_START = 5750  # Index in full dataset
TEACHER_FORCING_END = 5810    # Exclusive
TEACHER_FORCING_DAYS = 60
CTX_LEN = 5

set_seeds(0)
torch.set_default_dtype(torch.float64)

def parse_training_log(log_file):
    """Parse training log file to extract final epoch losses"""
    with open(log_file, 'r') as f:
        lines = f.readlines()

    # Find the last epoch's train and dev losses
    train_loss = None
    dev_loss = None
    train_re_surface = None
    dev_re_surface = None
    train_re_ex_feats = None
    dev_re_ex_feats = None
    train_kl_loss = None
    dev_kl_loss = None

    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]

        if train_loss is None and line.startswith('train loss ::'):
            # Parse: train loss :: loss: 0.001, re_surface: 0.001, re_ex_feats: 0.000, reconstruction_loss: 0.001, kl_loss: 3.753,
            matches = re.findall(r'(\w+): ([\d\.e\-]+)', line)
            for key, val in matches:
                if key == 'loss':
                    train_loss = float(val)
                elif key == 're_surface':
                    train_re_surface = float(val)
                elif key == 're_ex_feats':
                    train_re_ex_feats = float(val)
                elif key == 'kl_loss':
                    train_kl_loss = float(val)

        if dev_loss is None and line.startswith('dev loss ::'):
            matches = re.findall(r'(\w+): ([\d\.e\-]+)', line)
            for key, val in matches:
                if key == 'loss':
                    dev_loss = float(val)
                elif key == 're_surface':
                    dev_re_surface = float(val)
                elif key == 're_ex_feats':
                    dev_re_ex_feats = float(val)
                elif key == 'kl_loss':
                    dev_kl_loss = float(val)

        if train_loss is not None and dev_loss is not None:
            break

    return {
        'train_loss': train_loss,
        'train_re_surface': train_re_surface,
        'train_re_ex_feats': train_re_ex_feats,
        'train_kl_loss': train_kl_loss,
        'dev_loss': dev_loss,
        'dev_re_surface': dev_re_surface,
        'dev_re_ex_feats': dev_re_ex_feats,
        'dev_kl_loss': dev_kl_loss,
    }

def load_test_results():
    """Load test results from CSV"""
    results_df = pd.read_csv(f"{BASE_MODEL_DIR}/results.csv")

    test_results = {}
    for _, row in results_df.iterrows():
        model_name = row['fn'].replace('.pt', '')
        test_results[model_name] = {
            'test_loss': row['test_loss'],
            'test_re_surface': row['test_re_surface'],
            'test_re_ex_feats': row['test_re_ex_feats'],
            'test_kl_loss': row['test_kl_loss'],
        }

    return test_results

def compute_teacher_forcing_loss(model_path, model_name, vol_surf_data, ex_data):
    """Compute reconstruction loss on teacher forcing period"""
    print(f"\nComputing teacher forcing loss for {model_name}...")

    # Load model
    model_data = torch.load(model_path, weights_only=False)
    model_config = model_data["model_config"]
    model_config["mem_dropout"] = 0.
    model = CVAEMemRand(model_config)
    model.load_weights(dict_to_load=model_data)
    model.eval()

    use_ex_feats = model_config["ex_feats_dim"] > 0
    ex_loss_on_ret_only = model_config.get("ex_loss_on_ret_only", False)

    total_loss = 0.0
    total_re_surface = 0.0
    total_re_ex_feats = 0.0
    total_kl_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        # Iterate through teacher forcing period
        for day in range(TEACHER_FORCING_START, TEACHER_FORCING_END):
            # Get context and target
            surf_ctx = torch.from_numpy(vol_surf_data[day - CTX_LEN:day + 1])  # (T, 5, 5), T=ctx_len+1
            surf_ctx = surf_ctx.unsqueeze(0)  # (1, T, 5, 5)

            input_dict = {"surface": surf_ctx}

            if use_ex_feats:
                ex_ctx = torch.from_numpy(ex_data[day - CTX_LEN:day + 1])  # (T, 3)
                if len(ex_ctx.shape) == 1:
                    ex_ctx = ex_ctx.unsqueeze(1)
                ex_ctx = ex_ctx.unsqueeze(0)  # (1, T, 3)
                input_dict["ex_feats"] = ex_ctx

            # Get target (last timestep)
            target_surface = surf_ctx[:, -1:, :, :].to(model.device)  # (1, 1, 5, 5)

            # Forward pass
            if use_ex_feats:
                pred_surface, pred_ex_feats, z_mean, z_log_var, z = model.forward(input_dict)
                target_ex_feats = ex_ctx[:, -1:, :].to(model.device)  # (1, 1, 3)
            else:
                pred_surface, z_mean, z_log_var, z = model.forward(input_dict)

            # Compute losses
            re_surface = torch.nn.functional.mse_loss(pred_surface, target_surface)

            if use_ex_feats:
                if ex_loss_on_ret_only:
                    pred_ex_feats = pred_ex_feats[:, :, :1]
                    target_ex_feats = target_ex_feats[:, :, :1]

                if model_config["ex_feats_loss_type"] == "l2":
                    re_ex_feats = torch.nn.functional.mse_loss(pred_ex_feats, target_ex_feats)
                else:
                    re_ex_feats = torch.nn.functional.l1_loss(pred_ex_feats, target_ex_feats)

                reconstruction_loss = re_surface + model_config["re_feat_weight"] * re_ex_feats
            else:
                re_ex_feats = torch.zeros(1)
                reconstruction_loss = re_surface

            kl_loss = -0.5 * (1 + z_log_var - torch.exp(z_log_var) - torch.square(z_mean))
            kl_loss = torch.mean(torch.sum(kl_loss, dim=-1))

            total_loss_sample = reconstruction_loss + model_config["kl_weight"] * kl_loss

            total_loss += total_loss_sample.item()
            total_re_surface += re_surface.item()
            total_re_ex_feats += re_ex_feats.item()
            total_kl_loss += kl_loss.item()
            num_samples += 1

    return {
        'tf_loss': total_loss / num_samples,
        'tf_re_surface': total_re_surface / num_samples,
        'tf_re_ex_feats': total_re_ex_feats / num_samples,
        'tf_kl_loss': total_kl_loss / num_samples,
    }

# Load data
print("Loading data...")
data = np.load("data/vol_surface_with_ret.npz")
vol_surf_data = data["surface"]
ret_data = data["ret"]
level_data = data["levels"]
skew_data = data["skews"]
slope_data = data["slopes"]
ex_data = np.concatenate([ret_data[..., np.newaxis], skew_data[..., np.newaxis], slope_data[..., np.newaxis]], axis=-1)

print(f"\nTeacher forcing period: days {TEACHER_FORCING_START} to {TEACHER_FORCING_END-1} ({TEACHER_FORCING_DAYS} days)")
print("This period is in the TEST SET (days 5000-5821)")

# Load dates to show the time period
dates_df = pd.read_parquet("data/spx_vol_surface_history_full_data_fixed.parquet")
dates = pd.to_datetime(dates_df["date"].values)
print(f"Date range: {dates[TEACHER_FORCING_START].strftime('%Y-%m-%d')} to {dates[TEACHER_FORCING_END-1].strftime('%Y-%m-%d')}")

models = {
    "no_ex": {
        "name": "No EX (Surface Only)",
        "path": f"{BASE_MODEL_DIR}/no_ex.pt",
        "log": f"{BASE_MODEL_DIR}/no_ex-500-log.txt"
    },
    "ex_no_loss": {
        "name": "EX No Loss (+Features)",
        "path": f"{BASE_MODEL_DIR}/ex_no_loss.pt",
        "log": f"{BASE_MODEL_DIR}/ex_no_loss-500-log.txt"
    },
    "ex_loss": {
        "name": "EX Loss (+Features+Loss)",
        "path": f"{BASE_MODEL_DIR}/ex_loss.pt",
        "log": f"{BASE_MODEL_DIR}/ex_loss-500-log.txt"
    }
}

print("\n" + "="*80)
print("PARSING TRAINING LOGS")
print("="*80)

all_results = {}
for model_key, model_info in models.items():
    print(f"\nParsing {model_key}...")
    train_val_losses = parse_training_log(model_info["log"])
    all_results[model_key] = train_val_losses

print("\n" + "="*80)
print("LOADING TEST RESULTS")
print("="*80)

test_results = load_test_results()
for model_key in models.keys():
    if model_key in test_results:
        all_results[model_key].update(test_results[model_key])

print("\n" + "="*80)
print("COMPUTING TEACHER FORCING PERIOD LOSSES")
print("="*80)

for model_key, model_info in models.items():
    tf_losses = compute_teacher_forcing_loss(model_info["path"], model_key, vol_surf_data, ex_data)
    all_results[model_key].update(tf_losses)

# Generate comparison table
print("\n" + "="*80)
print("RECONSTRUCTION LOSS COMPARISON")
print("="*80)

output_lines = []
output_lines.append("="*80)
output_lines.append("RECONSTRUCTION LOSS COMPARISON")
output_lines.append("="*80)
output_lines.append(f"\nDataset Splits:")
output_lines.append(f"  Training:   Days 0-3999   (Jan 2000 - Nov 2015)")
output_lines.append(f"  Validation: Days 4000-4999 (Nov 2015 - Nov 2019)")
output_lines.append(f"  Test:       Days 5000-5821 (Nov 2019 - Feb 2023)")
output_lines.append(f"\nTeacher Forcing Period:")
output_lines.append(f"  Days {TEACHER_FORCING_START}-{TEACHER_FORCING_END-1} ({TEACHER_FORCING_DAYS} days)")
output_lines.append(f"  Date range: {dates[TEACHER_FORCING_START].strftime('%Y-%m-%d')} to {dates[TEACHER_FORCING_END-1].strftime('%Y-%m-%d')}")
output_lines.append(f"  Location: TEST SET")
output_lines.append("")

for model_key, model_info in models.items():
    results = all_results[model_key]

    output_lines.append("\n" + "-"*80)
    output_lines.append(f"Model: {model_info['name']}")
    output_lines.append("-"*80)

    # Surface reconstruction loss
    output_lines.append("\nSurface Reconstruction Loss (MSE):")
    output_lines.append(f"  Training:          {results['train_re_surface']:.6f}")
    output_lines.append(f"  Validation:        {results['dev_re_surface']:.6f}")
    output_lines.append(f"  Full Test:         {results['test_re_surface']:.6f}")
    output_lines.append(f"  Teacher Forcing:   {results['tf_re_surface']:.6f}")
    output_lines.append(f"\n  Ratio (TF / Train):  {results['tf_re_surface'] / results['train_re_surface']:.2f}x")
    output_lines.append(f"  Ratio (TF / Val):    {results['tf_re_surface'] / results['dev_re_surface']:.2f}x")
    output_lines.append(f"  Ratio (TF / Test):   {results['tf_re_surface'] / results['test_re_surface']:.2f}x")

    # Feature reconstruction loss (if applicable)
    if results['train_re_ex_feats'] > 0:
        output_lines.append("\nFeature Reconstruction Loss:")
        output_lines.append(f"  Training:          {results['train_re_ex_feats']:.6f}")
        output_lines.append(f"  Validation:        {results['dev_re_ex_feats']:.6f}")
        output_lines.append(f"  Full Test:         {results['test_re_ex_feats']:.6f}")
        output_lines.append(f"  Teacher Forcing:   {results['tf_re_ex_feats']:.6f}")

    # KL divergence
    output_lines.append("\nKL Divergence:")
    output_lines.append(f"  Training:          {results['train_kl_loss']:.6f}")
    output_lines.append(f"  Validation:        {results['dev_kl_loss']:.6f}")
    output_lines.append(f"  Full Test:         {results['test_kl_loss']:.6f}")
    output_lines.append(f"  Teacher Forcing:   {results['tf_kl_loss']:.6f}")

    # Total loss
    output_lines.append("\nTotal Loss:")
    output_lines.append(f"  Training:          {results['train_loss']:.6f}")
    output_lines.append(f"  Validation:        {results['dev_loss']:.6f}")
    output_lines.append(f"  Full Test:         {results['test_loss']:.6f}")
    output_lines.append(f"  Teacher Forcing:   {results['tf_loss']:.6f}")

output_lines.append("\n" + "="*80)
output_lines.append("INTERPRETATION")
output_lines.append("="*80)
output_lines.append("\nExpected behavior:")
output_lines.append("  1. Training loss < Validation loss < Test loss")
output_lines.append("     (Models see training data, so they fit it better)")
output_lines.append("\n  2. Teacher Forcing loss ≈ Full Test loss")
output_lines.append("     (Both are out-of-sample from the same test period)")
output_lines.append("\n  3. If Teacher Forcing loss > Full Test loss:")
output_lines.append("     The last 60 days may be particularly challenging")
output_lines.append("     (e.g., unusual market conditions in late 2022 - early 2023)")
output_lines.append("\nConclusion:")
output_lines.append("  If the ratios above are close to 1.0, the 'suboptimal' performance")
output_lines.append("  in the teacher forcing visualization is EXPECTED generalization.")
output_lines.append("="*80)

# Print and save
output_text = "\n".join(output_lines)
print(output_text)

import os
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, 'w') as f:
    f.write(output_text)

print(f"\n\nResults saved to: {OUTPUT_FILE}")
