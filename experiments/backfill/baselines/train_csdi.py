"""
Train CSDI on IV surface forecasting task.

Usage:
    python experiments/backfill/baselines/train_csdi.py [--epochs 200] [--fast]
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Ensure project root is on path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

# Patch out linear_attention_transformer before importing CSDI
# (we use is_linear=False, so this module is never called at runtime)
import types
_lat = types.ModuleType("linear_attention_transformer")
_lat.LinearAttentionTransformer = None
sys.modules["linear_attention_transformer"] = _lat

# Add CSDI to path
CSDI_DIR = str(Path(__file__).resolve().parent.parent.parent.parent / "external" / "csdi")
if CSDI_DIR not in sys.path:
    sys.path.insert(0, CSDI_DIR)

from main_model import CSDI_Forecasting
from experiments.backfill.baselines.csdi_adapter import (
    IVSurfaceForecastingDataset, get_csdi_config,
)


def train_csdi(config, model, train_loader, val_loader, save_dir, device):
    """Train CSDI with validation-based early stopping."""
    optimizer = Adam(model.parameters(), lr=config["train"]["lr"], weight_decay=1e-6)

    epochs = config["train"]["epochs"]
    p1 = int(0.75 * epochs)
    p2 = int(0.9 * epochs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[p1, p2], gamma=0.1)

    best_val_loss = float("inf")
    best_epoch = -1
    patience = 30
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        n_batches = 0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", mininterval=5.0) as it:
            for batch in it:
                optimizer.zero_grad()
                loss = model(batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1
                it.set_postfix(loss=train_loss / n_batches)

        scheduler.step()
        avg_train = train_loss / n_batches

        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    loss = model(batch, is_train=0)
                    val_loss += loss.item()
                    val_batches += 1
            avg_val = val_loss / val_batches
            print(f"  Epoch {epoch+1}: train_loss={avg_train:.4f}, val_loss={avg_val:.4f}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_epoch = epoch + 1
                no_improve = 0
                torch.save(model.state_dict(), save_dir / "best_model_state.pt")
                print(f"  --> Best model saved (val_loss={avg_val:.4f})")
            else:
                no_improve += 10
                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch+1} (best: {best_epoch})")
                    break
        else:
            print(f"  Epoch {epoch+1}: train_loss={avg_train:.4f}")

    return best_epoch


def main():
    parser = argparse.ArgumentParser(description="Train CSDI for IV surface forecasting")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--fast", action="store_true", help="Quick test (3 epochs)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Config
    config = get_csdi_config()
    config["train"]["epochs"] = 3 if args.fast else args.epochs
    config["train"]["batch_size"] = args.batch_size

    # Data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]  # (N, 5, 5) in [0, 1]
    print(f"Loaded {len(surfaces)} surfaces, range [{surfaces.min():.3f}, {surfaces.max():.3f}]")

    # Splits matching config_block_ar.py
    train_end = 4040
    val_start = 4040
    val_end = 4540
    test_start = 4540

    train_ds = IVSurfaceForecastingDataset(surfaces, 0, train_end)
    val_ds = IVSurfaceForecastingDataset(
        surfaces, val_start, val_end,
        mean=train_ds.mean, std=train_ds.std,
    )

    print(f"Train: {len(train_ds)} windows, Val: {len(val_ds)} windows")
    print(f"Mean range: [{train_ds.mean.min():.4f}, {train_ds.mean.max():.4f}]")
    print(f"Std range:  [{train_ds.std.min():.4f}, {train_ds.std.max():.4f}]")

    train_loader = DataLoader(train_ds, batch_size=config["train"]["batch_size"],
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config["train"]["batch_size"],
                            shuffle=False, num_workers=0)

    # Model
    device = args.device
    target_dim = 25  # 5×5 flattened
    model = CSDI_Forecasting(config, device, target_dim).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CSDI model: {n_params:,} parameters")
    print(f"Config: {json.dumps(config, indent=2)}")

    # Save directory
    save_dir = Path("models/backfill/baselines/csdi")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Train
    best_epoch = train_csdi(config, model, train_loader, val_loader, save_dir, device)

    # Save final checkpoint with metadata
    best_state_path = save_dir / "best_model_state.pt"
    if not best_state_path.exists():
        torch.save(model.state_dict(), best_state_path)
        best_epoch = epochs
    best_state = torch.load(best_state_path, weights_only=True)
    checkpoint = {
        "config": config,
        "model_state_dict": best_state,
        "mean": train_ds.mean,
        "std": train_ds.std,
        "best_epoch": best_epoch,
        "n_params": n_params,
    }
    torch.save(checkpoint, save_dir / "best_model.pt")
    print(f"\nFinal checkpoint saved to {save_dir}/best_model.pt")
    print(f"Best epoch: {best_epoch}, params: {n_params:,}")

    # Quick smoke test
    print("\nSmoke test...")
    model.load_state_dict(best_state)
    model.eval()

    test_ds = IVSurfaceForecastingDataset(
        surfaces, test_start, len(surfaces),
        mean=train_ds.mean, std=train_ds.std,
    )
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)
    batch = next(iter(test_loader))

    from experiments.backfill.baselines.csdi_adapter import CSDIBaseline
    wrapper = CSDIBaseline(model, train_ds.mean, train_ds.std, device)

    # Simulate eval pipeline: get history in [-1,1]
    history_z = batch["observed_data"][:, :30, :]  # (B, 30, 25) z-scored
    history_01 = history_z.numpy() * train_ds.std + train_ds.mean  # [0,1]
    history_norm = torch.tensor(history_01 * 2 - 1, dtype=torch.float32)  # [-1,1]
    history_55 = history_norm.reshape(-1, 30, 5, 5)

    samples = wrapper.sample(history_55, n_samples=5)
    print(f"Output shape: {samples.shape}, range: [{samples.min():.3f}, {samples.max():.3f}]")


if __name__ == "__main__":
    main()
