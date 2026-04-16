"""
Train CSDI on 38-d joint IV + factor forecasting task.

Usage:
    PYTHONPATH=. python experiments/backfill/baselines/train_csdi_38d.py [--epochs 200] [--fast]
"""

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Patch out linear_attention_transformer before importing CSDI
import types
_lat = types.ModuleType("linear_attention_transformer")
_lat.LinearAttentionTransformer = None
sys.modules["linear_attention_transformer"] = _lat

CSDI_DIR = str(Path(__file__).resolve().parent.parent.parent.parent / "external" / "csdi")
if CSDI_DIR not in sys.path:
    sys.path.insert(0, CSDI_DIR)

from main_model import CSDI_Forecasting
from experiments.backfill.baselines.data_loader_38d import load_aligned_38d_data


class JointCSDIDataset(Dataset):
    """CSDI dataset for 38-d joint forecasting on daily changes."""

    def __init__(self, daily_changes, start_idx=0, end_idx=None,
                 history_len=30, future_len=30, mean=None, std=None):
        end_idx = end_idx or len(daily_changes)
        self.changes = daily_changes[start_idx:end_idx]
        self.history_len = history_len
        self.future_len = future_len
        self.seq_len = history_len + future_len

        if mean is None:
            self.mean = self.changes.mean(axis=0)
            self.std = self.changes.std(axis=0) + 1e-8
        else:
            self.mean = mean
            self.std = std

        self.data = (self.changes - self.mean) / self.std
        self.n_windows = len(self.data) - self.seq_len + 1

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        seq = self.data[idx:idx + self.seq_len]  # (60, 38)

        observed_mask = np.ones_like(seq)
        gt_mask = observed_mask.copy()
        # ALL future dims unobserved (no oracle factor information)
        gt_mask[self.history_len:] = 0.0

        return {
            "observed_data": seq.astype(np.float32),
            "observed_mask": observed_mask.astype(np.float32),
            "gt_mask": gt_mask.astype(np.float32),
            "timepoints": np.arange(self.seq_len, dtype=np.float32),
        }


def get_csdi_38d_config():
    """CSDI config for 38-d joint forecasting."""
    return {
        "train": {
            "epochs": 200,
            "batch_size": 16,
            "lr": 1.0e-3,
            "itr_per_epoch": int(1e8),
        },
        "diffusion": {
            "layers": 4,
            "channels": 64,
            "nheads": 8,
            "diffusion_embedding_dim": 128,
            "beta_start": 0.0001,
            "beta_end": 0.5,
            "num_steps": 50,
            "schedule": "quad",
            "is_linear": False,
        },
        "model": {
            "is_unconditional": 0,
            "timeemb": 128,
            "featureemb": 16,
            "target_strategy": "test",
            "num_sample_features": 38,  # all 38 features, no subsampling
        },
    }


def train_csdi(config, model, train_loader, val_loader, save_dir, device):
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
            print(f"  Epoch {epoch+1}: train={avg_train:.4f}, val={avg_val:.4f}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_epoch = epoch + 1
                no_improve = 0
                torch.save(model.state_dict(), save_dir / "best_model_state.pt")
                print(f"  --> Best model saved (val={avg_val:.4f})")
            else:
                no_improve += 10
                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch+1} (best: {best_epoch})")
                    break
        else:
            print(f"  Epoch {epoch+1}: train={avg_train:.4f}")

    return best_epoch


def main():
    parser = argparse.ArgumentParser(description="Train CSDI 38-d joint")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = get_csdi_38d_config()
    config["train"]["epochs"] = 3 if args.fast else args.epochs
    config["train"]["batch_size"] = args.batch_size

    print("Loading aligned 38-d data...")
    data = load_aligned_38d_data()
    daily_changes = data["joint_changes_38"]
    print(f"Daily changes: {daily_changes.shape}")

    train_ds = JointCSDIDataset(daily_changes, 0, 4039)
    val_ds = JointCSDIDataset(daily_changes, 4039, 4539,
                               mean=train_ds.mean, std=train_ds.std)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config["train"]["batch_size"],
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config["train"]["batch_size"],
                            shuffle=False, num_workers=0)

    device = args.device
    model = CSDI_Forecasting(config, device, target_dim=38).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CSDI 38-d: {n_params:,} params")
    print(f"Config: {json.dumps(config, indent=2)}")

    save_dir = Path("models/backfill/baselines/csdi_38d")
    save_dir.mkdir(parents=True, exist_ok=True)

    best_epoch = train_csdi(config, model, train_loader, val_loader, save_dir, device)

    best_state_path = save_dir / "best_model_state.pt"
    if not best_state_path.exists():
        torch.save(model.state_dict(), best_state_path)
        best_epoch = config["train"]["epochs"]
    best_state = torch.load(best_state_path, weights_only=True)
    checkpoint = {
        "config": config,
        "model_state_dict": best_state,
        "mean": train_ds.mean,
        "std": train_ds.std,
        "target_dim": 38,
        "best_epoch": best_epoch,
        "n_params": n_params,
    }
    torch.save(checkpoint, save_dir / "best_model.pt")
    print(f"\nCheckpoint saved: {save_dir}/best_model.pt (best epoch: {best_epoch})")


if __name__ == "__main__":
    main()
