"""
Train standalone TimeGrad on IV surface forecasting task.

Usage:
    python experiments/backfill/baselines/train_timegrad.py [--epochs 100] [--fast]
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from experiments.backfill.baselines.timegrad_standalone import (
    TimeGradModel, TimeGradBaseline,
)


class IVSurfaceTimeGradDataset(Dataset):
    """Dataset producing (history, future) pairs in z-scored flat format."""

    def __init__(self, surfaces, start_idx=0, end_idx=None,
                 history_len=30, future_len=30, mean=None, std=None):
        end_idx = end_idx or len(surfaces)
        self.surfaces = surfaces[start_idx:end_idx]
        self.history_len = history_len
        self.future_len = future_len
        self.seq_len = history_len + future_len

        flat = self.surfaces.reshape(-1, 25)
        if mean is None:
            self.mean = flat.mean(axis=0)
            self.std = flat.std(axis=0) + 1e-8
        else:
            self.mean = mean
            self.std = std

        self.data = (flat - self.mean) / self.std
        self.n_windows = len(self.data) - self.seq_len + 1

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        seq = self.data[idx:idx + self.seq_len].astype(np.float32)
        return {
            "history": seq[:self.history_len],  # (30, 25)
            "future": seq[self.history_len:],    # (30, 25)
        }


def train_timegrad(model, train_loader, val_loader, save_dir, epochs, device, lr=1e-4):
    """Train TimeGrad with validation-based early stopping."""
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_epoch = -1
    patience = 20
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        n_batches = 0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", mininterval=5.0) as it:
            for batch in it:
                history = batch["history"].to(device)
                future = batch["future"].to(device)

                optimizer.zero_grad()
                loss = model(history, future)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1
                it.set_postfix(loss=train_loss / n_batches)

        scheduler.step()
        avg_train = train_loss / n_batches

        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_loss = 0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    history = batch["history"].to(device)
                    future = batch["future"].to(device)
                    loss = model(history, future)
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
                no_improve += 5
                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch+1} (best: {best_epoch})")
                    break
        else:
            print(f"  Epoch {epoch+1}: train={avg_train:.4f}")

    return best_epoch


def main():
    parser = argparse.ArgumentParser(description="Train TimeGrad for IV surfaces")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gru_hidden", type=int, default=128)
    parser.add_argument("--n_diffusion_steps", type=int, default=100)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    epochs = 3 if args.fast else args.epochs

    # Data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    print(f"Loaded {len(surfaces)} surfaces")

    train_ds = IVSurfaceTimeGradDataset(surfaces, 0, 4040)
    val_ds = IVSurfaceTimeGradDataset(surfaces, 4040, 4540,
                                       mean=train_ds.mean, std=train_ds.std)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = TimeGradModel(
        input_dim=25, gru_hidden=args.gru_hidden,
        n_diffusion_steps=args.n_diffusion_steps,
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TimeGrad: {n_params:,} params (GRU={args.gru_hidden}, diff_steps={args.n_diffusion_steps})")

    # Train
    save_dir = Path("models/backfill/baselines/timegrad")
    save_dir.mkdir(parents=True, exist_ok=True)

    best_epoch = train_timegrad(model, train_loader, val_loader, save_dir, epochs, args.device)

    # Save checkpoint
    best_state_path = save_dir / "best_model_state.pt"
    if not best_state_path.exists():
        # No validation happened (e.g. --fast), save final model
        torch.save(model.state_dict(), best_state_path)
        best_epoch = epochs
    best_state = torch.load(best_state_path, weights_only=True)
    checkpoint = {
        "model_state_dict": best_state,
        "mean": train_ds.mean,
        "std": train_ds.std,
        "input_dim": 25,
        "gru_hidden": args.gru_hidden,
        "n_diffusion_steps": args.n_diffusion_steps,
        "ddim_steps": 20,
        "best_epoch": best_epoch,
        "n_params": n_params,
    }
    torch.save(checkpoint, save_dir / "best_model.pt")
    print(f"\nCheckpoint saved: {save_dir}/best_model.pt (best epoch: {best_epoch})")

    # Smoke test
    print("\nSmoke test...")
    model.load_state_dict(best_state)
    wrapper = TimeGradBaseline(model, train_ds.mean, train_ds.std, args.device)
    wrapper.eval()

    test_batch = next(iter(val_loader))
    history_z = test_batch["history"][:4].to(args.device)
    # Convert to [-1,1] for wrapper interface
    history_01 = history_z.cpu().numpy() * train_ds.std + train_ds.mean
    history_norm = torch.tensor(history_01 * 2 - 1, dtype=torch.float32)
    history_55 = history_norm.reshape(-1, 30, 5, 5)

    samples = wrapper.sample(history_55, n_samples=3)
    print(f"Output: {samples.shape}, range: [{samples.min():.3f}, {samples.max():.3f}]")


if __name__ == "__main__":
    main()
