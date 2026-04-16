"""
Train DeepVAR on 38-d joint IV + factor forecasting task.

Usage:
    PYTHONPATH=. python experiments/backfill/baselines/train_deepvar_38d.py [--epochs 100] [--fast]
"""

import argparse
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

from experiments.backfill.baselines.deepvar_standalone import DeepVARModel
from experiments.backfill.baselines.data_loader_38d import load_aligned_38d_data


class JointChangesDataset(Dataset):
    """Dataset producing (history, future) pairs of z-scored 38-d daily changes."""

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
        seq = self.data[idx:idx + self.seq_len].astype(np.float32)
        return {
            "history": seq[:self.history_len],  # (30, 38)
            "future": seq[self.history_len:],   # (30, 38)
        }


def train_model(model, train_loader, val_loader, save_dir, epochs, device, lr=1e-3):
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
    parser = argparse.ArgumentParser(description="Train DeepVAR 38-d joint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lstm_hidden", type=int, default=128)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    epochs = 3 if args.fast else args.epochs

    # Load aligned 38-d data
    print("Loading aligned 38-d data...")
    data = load_aligned_38d_data()
    daily_changes = data["joint_changes_38"]
    print(f"Daily changes: {daily_changes.shape}")

    # train_end=4040 in surface space → 4039 in changes space
    train_ds = JointChangesDataset(daily_changes, 0, 4039)
    val_ds = JointChangesDataset(daily_changes, 4039, 4539,
                                  mean=train_ds.mean, std=train_ds.std)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model — input_dim=38 for joint forecasting
    model = DeepVARModel(
        input_dim=38, lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers, rank=args.rank,
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DeepVAR 38-d: {n_params:,} params")

    save_dir = Path("models/backfill/baselines/deepvar_38d")
    save_dir.mkdir(parents=True, exist_ok=True)

    best_epoch = train_model(model, train_loader, val_loader, save_dir, epochs, args.device, lr=args.lr)

    # Save checkpoint
    best_state_path = save_dir / "best_model_state.pt"
    if not best_state_path.exists():
        torch.save(model.state_dict(), best_state_path)
        best_epoch = epochs
    best_state = torch.load(best_state_path, weights_only=True)
    checkpoint = {
        "model_state_dict": best_state,
        "mean": train_ds.mean,
        "std": train_ds.std,
        "input_dim": 38,
        "lstm_hidden": args.lstm_hidden,
        "lstm_layers": args.lstm_layers,
        "rank": args.rank,
        "best_epoch": best_epoch,
        "n_params": n_params,
    }
    torch.save(checkpoint, save_dir / "best_model.pt")
    print(f"\nCheckpoint saved: {save_dir}/best_model.pt (best epoch: {best_epoch})")


if __name__ == "__main__":
    main()
