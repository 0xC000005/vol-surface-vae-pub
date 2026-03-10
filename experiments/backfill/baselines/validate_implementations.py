"""
Validation tests for CSDI and TimeGrad implementations.

Verifies:
1. DDIM sampling formula correctness (analytic test)
2. TimeGrad can learn a simple synthetic AR(1) process
3. CSDI can learn a simple synthetic forecasting task
4. Both models produce reasonable samples on our IV data (smoke test)

Usage:
    python experiments/backfill/baselines/validate_implementations.py
"""

import sys
from pathlib import Path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ============================================================
# Test 1: DDIM Formula Verification
# ============================================================

def test_ddim_formula():
    """Verify DDIM sampling formula by round-tripping: add noise then denoise.

    With a PERFECT noise predictor (oracle), DDIM should recover x0 exactly.
    """
    print("\n" + "=" * 60)
    print("TEST 1: DDIM Formula Verification")
    print("=" * 60)

    torch.manual_seed(42)

    # Setup: linear schedule matching TimeGrad defaults
    n_steps = 100
    beta_start, beta_end = 1e-4, 0.02
    betas = torch.linspace(beta_start, beta_end, n_steps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # Ground truth signal
    x0 = torch.randn(4, 25)  # batch of 4, 25-dim

    # Forward diffusion to t=99 (full noise)
    t_full = n_steps - 1
    alpha_t = alphas_cumprod[t_full]
    noise_true = torch.randn_like(x0)
    x_noisy = alpha_t.sqrt() * x0 + (1 - alpha_t).sqrt() * noise_true

    # DDIM reverse with ORACLE noise predictor
    n_ddim_steps = 20
    step_indices = torch.linspace(0, n_steps - 1, n_ddim_steps + 1).long()
    timesteps = step_indices.flip(0)[:-1]

    x = x_noisy.clone()
    for i, t_val in enumerate(timesteps):
        # Oracle: compute exact noise at this step
        alpha_cur = alphas_cumprod[t_val]
        # True noise prediction: ε = (x_t - √α_t * x0) / √(1-α_t)
        noise_pred = (x - alpha_cur.sqrt() * x0) / (1 - alpha_cur).sqrt()

        if i < len(timesteps) - 1:
            alpha_prev = alphas_cumprod[timesteps[i + 1]]
        else:
            alpha_prev = torch.tensor(1.0)

        # DDIM update (same as timegrad_standalone.py line 187-188)
        x0_pred = (x - (1 - alpha_cur).sqrt() * noise_pred) / alpha_cur.sqrt()
        x = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * noise_pred

    # With perfect oracle, should recover x0 exactly
    error = (x - x0).abs().max().item()
    print(f"  Max recovery error with oracle predictor: {error:.2e}")
    print(f"  Mean recovery error: {(x - x0).abs().mean().item():.2e}")

    # The error should be essentially zero (floating point)
    passed = error < 1e-4
    print(f"  DDIM formula correct: {'PASS' if passed else 'FAIL'}")

    if not passed:
        print(f"  WARNING: DDIM formula has numerical issues")
        # Check intermediate: what does x0_pred look like at each step?
        print(f"  x0 range: [{x0.min():.3f}, {x0.max():.3f}]")
        print(f"  recovered range: [{x.min():.3f}, {x.max():.3f}]")

    return passed


# ============================================================
# Test 2: TimeGrad on Synthetic AR(1)
# ============================================================

class SyntheticAR1Dataset(Dataset):
    """Synthetic 5-dim AR(1) process for validation.

    x_{t+1} = A * x_t + noise, where A is a known transition matrix.
    If TimeGrad can learn to forecast this, the implementation is correct.
    """
    def __init__(self, n_sequences=200, seq_len=60, dim=5, seed=42):
        rng = np.random.RandomState(seed)
        self.history_len = 30
        self.future_len = 30
        self.dim = dim

        # AR(1) with mild autocorrelation
        A = np.eye(dim) * 0.8  # diagonal AR(1), coeff=0.8
        noise_std = 0.3

        # Generate sequences
        all_seqs = []
        for _ in range(n_sequences):
            x = rng.randn(dim) * 0.5
            seq = [x.copy()]
            for t in range(seq_len - 1):
                x = A @ x + rng.randn(dim) * noise_std
                seq.append(x.copy())
            all_seqs.append(np.stack(seq))

        self.data = np.stack(all_seqs).astype(np.float32)  # (N, 60, 5)

        # Z-score normalization
        flat = self.data.reshape(-1, dim)
        self.mean = flat.mean(axis=0)
        self.std = flat.std(axis=0) + 1e-8
        self.data_z = (self.data - self.mean) / self.std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data_z[idx]
        return {
            "history": seq[:self.history_len],  # (30, 5)
            "future": seq[self.history_len:],   # (30, 5)
        }


def test_timegrad_synthetic():
    """Train TimeGrad on synthetic AR(1) and check if it learns."""
    print("\n" + "=" * 60)
    print("TEST 2: TimeGrad on Synthetic AR(1)")
    print("=" * 60)

    from experiments.backfill.baselines.timegrad_standalone import TimeGradModel

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Small synthetic dataset
    train_ds = SyntheticAR1Dataset(n_sequences=500, dim=5, seed=42)
    val_ds = SyntheticAR1Dataset(n_sequences=100, dim=5, seed=99)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # Small model for quick test
    model = TimeGradModel(
        input_dim=5, gru_hidden=64, n_diffusion_steps=50,
        beta_start=1e-4, beta_end=0.02,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Train for 30 epochs
    print("  Training TimeGrad on AR(1)...")
    best_val = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(30):
        model.train()
        epoch_loss = 0
        n = 0
        for batch in train_loader:
            h = batch["history"].to(device)
            f = batch["future"].to(device)
            optimizer.zero_grad()
            loss = model(h, f)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        train_losses.append(epoch_loss / n)

        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            vloss = 0
            vn = 0
            with torch.no_grad():
                for batch in val_loader:
                    h = batch["history"].to(device)
                    f = batch["future"].to(device)
                    loss = model(h, f)
                    vloss += loss.item()
                    vn += 1
            val_loss = vloss / vn
            val_losses.append(val_loss)
            if val_loss < best_val:
                best_val = val_loss
            print(f"    Epoch {epoch+1}: train={train_losses[-1]:.4f}, val={val_loss:.4f}")

    # Check 1: Train loss should decrease
    loss_decreased = train_losses[-1] < train_losses[0] * 0.5
    print(f"\n  Train loss decrease: {train_losses[0]:.4f} → {train_losses[-1]:.4f} "
          f"({'PASS' if loss_decreased else 'FAIL'})")

    # Check 2: Val loss should be reasonable (not exploding)
    val_stable = val_losses[-1] < val_losses[0] * 2.0
    print(f"  Val loss stable: {val_losses[0]:.4f} → {val_losses[-1]:.4f} "
          f"({'PASS' if val_stable else 'FAIL - OVERFITTING'})")

    # Check 3: Generate samples and check they're not garbage
    model.eval()
    test_batch = next(iter(val_loader))
    history_z = test_batch["history"][:4].to(device)

    with torch.no_grad():
        samples = model.sample_trajectory(history_z, n_future=30, n_ddim_steps=10)

    sample_mean = samples.mean().item()
    sample_std = samples.std().item()
    sample_range = (samples.min().item(), samples.max().item())

    # Samples should have reasonable statistics (z-scored, so roughly mean=0, std~1)
    reasonable_samples = abs(sample_mean) < 3.0 and 0.1 < sample_std < 5.0
    print(f"  Sample stats: mean={sample_mean:.3f}, std={sample_std:.3f}, "
          f"range=[{sample_range[0]:.2f}, {sample_range[1]:.2f}] "
          f"({'PASS' if reasonable_samples else 'FAIL'})")

    # Check 4: Sample CRPS on synthetic data
    # For AR(1) with coeff=0.8, the conditional variance at horizon h is:
    # Var(x_{t+h} | x_t) = noise_std^2 * sum_{i=0}^{h-1} 0.8^{2i}
    # At h=1: Var = 0.3^2 = 0.09, std = 0.3
    # At h=30: Var ≈ 0.3^2 / (1 - 0.64) = 0.25, std ≈ 0.5
    # So sample spread should be O(0.3-0.5) in original space, or O(1) in z-space

    # Generate multiple samples for CRPS
    n_test_samples = 20
    all_traj = []
    for _ in range(n_test_samples):
        traj = model.sample_trajectory(history_z, n_future=30, n_ddim_steps=10)
        all_traj.append(traj)
    all_traj = torch.stack(all_traj, dim=1)  # (4, 20, 30, 5)

    gt_future = test_batch["future"][:4].to(device)

    # Simple CRPS at h=1 and h=30
    for h, h_idx in [(1, 0), (30, 29)]:
        s = all_traj[:, :, h_idx]  # (4, 20, 5)
        y = gt_future[:, h_idx]  # (4, 5)
        mae = (s - y.unsqueeze(1)).abs().mean().item()
        spread = 0
        for i in range(n_test_samples):
            for j in range(i+1, n_test_samples):
                spread += (s[:, i] - s[:, j]).abs().mean().item()
        spread = spread * 2 / (n_test_samples * (n_test_samples - 1))
        crps = mae - 0.5 * spread
        print(f"  Synthetic CRPS at h={h}: {crps:.4f} (MAE={mae:.4f}, spread={spread:.4f})")

    all_passed = loss_decreased and val_stable and reasonable_samples
    print(f"\n  TimeGrad synthetic test: {'PASS' if all_passed else 'FAIL'}")

    return all_passed, train_losses, val_losses


# ============================================================
# Test 3: CSDI on Synthetic Forecasting
# ============================================================

class SyntheticCSDIDataset(Dataset):
    """Synthetic dataset in CSDI format."""
    def __init__(self, n_sequences=200, seq_len=60, dim=5, seed=42):
        rng = np.random.RandomState(seed)
        history_len = 30

        A = np.eye(dim) * 0.8
        noise_std = 0.3

        all_seqs = []
        for _ in range(n_sequences):
            x = rng.randn(dim) * 0.5
            seq = [x.copy()]
            for t in range(seq_len - 1):
                x = A @ x + rng.randn(dim) * noise_std
                seq.append(x.copy())
            all_seqs.append(np.stack(seq))

        self.data = np.stack(all_seqs).astype(np.float32)
        flat = self.data.reshape(-1, dim)
        self.mean = flat.mean(axis=0)
        self.std = flat.std(axis=0) + 1e-8
        self.data_z = (self.data - self.mean) / self.std

        self.history_len = history_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data_z[idx]  # (60, 5)
        observed_mask = np.ones_like(seq)
        gt_mask = observed_mask.copy()
        gt_mask[self.history_len:] = 0.0

        return {
            "observed_data": seq.astype(np.float32),
            "observed_mask": observed_mask.astype(np.float32),
            "gt_mask": gt_mask.astype(np.float32),
            "timepoints": np.arange(len(seq), dtype=np.float32),
        }


def test_csdi_synthetic():
    """Train CSDI on synthetic data and check if it learns."""
    print("\n" + "=" * 60)
    print("TEST 3: CSDI on Synthetic Forecasting")
    print("=" * 60)

    # Patch linear_attention_transformer
    import types
    _lat = types.ModuleType("linear_attention_transformer")
    _lat.LinearAttentionTransformer = None
    sys.modules["linear_attention_transformer"] = _lat

    CSDI_DIR = str(Path(__file__).resolve().parent.parent.parent.parent / "external" / "csdi")
    if CSDI_DIR not in sys.path:
        sys.path.insert(0, CSDI_DIR)

    from main_model import CSDI_Forecasting

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Small config for quick test
    config = {
        "train": {"epochs": 20, "batch_size": 16, "lr": 1e-3},
        "diffusion": {
            "layers": 2, "channels": 32, "nheads": 4,
            "diffusion_embedding_dim": 64,
            "beta_start": 0.0001, "beta_end": 0.5,
            "num_steps": 20, "schedule": "quad", "is_linear": False,
        },
        "model": {
            "is_unconditional": 0, "timeemb": 64, "featureemb": 8,
            "target_strategy": "test", "num_sample_features": 5,
        },
    }

    train_ds = SyntheticCSDIDataset(n_sequences=300, dim=5, seed=42)
    val_ds = SyntheticCSDIDataset(n_sequences=50, dim=5, seed=99)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = CSDI_Forecasting(config, device, target_dim=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  CSDI model: {n_params:,} parameters")

    print("  Training CSDI on synthetic data...")
    train_losses = []

    for epoch in range(20):
        model.train()
        epoch_loss = 0
        n = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        avg_loss = epoch_loss / n
        train_losses.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}: train_loss={avg_loss:.4f}")

    # Check 1: Loss decreased
    loss_decreased = train_losses[-1] < train_losses[0] * 0.7
    print(f"\n  Train loss: {train_losses[0]:.4f} → {train_losses[-1]:.4f} "
          f"({'PASS' if loss_decreased else 'FAIL'})")

    # Check 2: Generate samples
    model.eval()
    test_batch = next(iter(val_loader))

    with torch.no_grad():
        output = model.evaluate(test_batch, n_samples=10)
        samples = output[0]  # (B, 10, K=5, L=60)

    # Extract future
    future_samples = samples[:, :, :, 30:]  # (B, 10, 5, 30)
    future_samples = future_samples.permute(0, 1, 3, 2)  # (B, 10, 30, 5)

    sample_mean = future_samples.mean().item()
    sample_std = future_samples.std().item()
    reasonable = abs(sample_mean) < 3.0 and 0.1 < sample_std < 5.0
    print(f"  Sample stats: mean={sample_mean:.3f}, std={sample_std:.3f} "
          f"({'PASS' if reasonable else 'FAIL'})")

    # Check 3: Samples should be different (diversity)
    sample_diversity = (future_samples[:, 0] - future_samples[:, 1]).abs().mean().item()
    diverse = sample_diversity > 0.01
    print(f"  Sample diversity (mean abs diff between samples): {sample_diversity:.4f} "
          f"({'PASS' if diverse else 'FAIL - identical samples'})")

    all_passed = loss_decreased and reasonable and diverse
    print(f"\n  CSDI synthetic test: {'PASS' if all_passed else 'FAIL'}")

    return all_passed


# ============================================================
# Test 4: Smoke Test on Real IV Data
# ============================================================

def test_real_data_smoke():
    """Quick smoke test loading trained models and generating samples on IV data."""
    print("\n" + "=" * 60)
    print("TEST 4: Real Data Smoke Test (Trained Models)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]

    # Create a test batch in [-1, 1] (matching our pipeline)
    test_start = 4540
    history_01 = surfaces[test_start:test_start+30]  # (30, 5, 5) in [0,1]
    future_01 = surfaces[test_start+30:test_start+60]  # (30, 5, 5) in [0,1]
    history_norm = torch.tensor(history_01 * 2 - 1, dtype=torch.float32).unsqueeze(0)  # (1, 30, 5, 5)

    # Test TimeGrad
    tg_path = Path("models/backfill/baselines/timegrad/best_model.pt")
    if tg_path.exists():
        print("\n  --- TimeGrad ---")
        from experiments.backfill.baselines.timegrad_standalone import load_timegrad_model
        tg_model = load_timegrad_model(str(tg_path), device)
        tg_model.eval()

        samples = tg_model.sample(history_norm.to(device), n_samples=10)
        print(f"  Output shape: {samples.shape}")
        print(f"  Range: [{samples.min():.3f}, {samples.max():.3f}]")
        print(f"  Mean: {samples.mean():.4f}, Std: {samples.std():.4f}")

        # Compare to ground truth
        gt = torch.tensor(future_01, dtype=torch.float32)
        sample_median = samples[0].median(dim=0).values  # (30, 5, 5)
        mae = (sample_median - gt).abs().mean().item()
        print(f"  Median sample MAE vs GT: {mae:.4f}")

        # Check diversity
        diversity = (samples[0, 0] - samples[0, 1]).abs().mean().item()
        print(f"  Sample diversity: {diversity:.4f}")

        # Check if samples look like IV surfaces (0.1-0.8 typical range)
        in_range = ((samples > 0.05) & (samples < 0.95)).float().mean().item()
        print(f"  Fraction in [0.05, 0.95]: {in_range:.1%}")

        results["timegrad"] = {
            "mae": mae, "diversity": diversity, "in_range": in_range,
            "passed": in_range > 0.5 and diversity > 0.001
        }
    else:
        print("  TimeGrad checkpoint not found, skipping")

    # Test CSDI
    csdi_path = Path("models/backfill/baselines/csdi/best_model.pt")
    if csdi_path.exists():
        print("\n  --- CSDI ---")
        from experiments.backfill.baselines.csdi_adapter import load_csdi_model
        csdi_model = load_csdi_model(str(csdi_path), device)
        csdi_model.eval()

        samples = csdi_model.sample(history_norm.to(device), n_samples=10)
        print(f"  Output shape: {samples.shape}")
        print(f"  Range: [{samples.min():.3f}, {samples.max():.3f}]")
        print(f"  Mean: {samples.mean():.4f}, Std: {samples.std():.4f}")

        gt = torch.tensor(future_01, dtype=torch.float32)
        sample_median = samples[0].median(dim=0).values
        mae = (sample_median - gt).abs().mean().item()
        print(f"  Median sample MAE vs GT: {mae:.4f}")

        diversity = (samples[0, 0] - samples[0, 1]).abs().mean().item()
        print(f"  Sample diversity: {diversity:.4f}")

        in_range = ((samples > 0.05) & (samples < 0.95)).float().mean().item()
        print(f"  Fraction in [0.05, 0.95]: {in_range:.1%}")

        results["csdi"] = {
            "mae": mae, "diversity": diversity, "in_range": in_range,
            "passed": in_range > 0.5 and diversity > 0.001
        }
    else:
        print("  CSDI checkpoint not found, skipping")

    return results


# ============================================================
# Test 5: TimeGrad Overfitting Diagnosis
# ============================================================

def test_timegrad_overfit_diagnosis():
    """Diagnose the TimeGrad overfitting issue on IV data.

    The real model overfits at epoch 5 (val loss monotonically increases).
    Test hypothesis: teacher forcing + small dataset causes severe train/test gap.
    """
    print("\n" + "=" * 60)
    print("TEST 5: TimeGrad Overfitting Diagnosis")
    print("=" * 60)

    from experiments.backfill.baselines.timegrad_standalone import TimeGradModel

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load real data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]

    from experiments.backfill.baselines.train_timegrad import IVSurfaceTimeGradDataset

    train_ds = IVSurfaceTimeGradDataset(surfaces, 0, 4040)
    val_ds = IVSurfaceTimeGradDataset(surfaces, 4040, 4540,
                                       mean=train_ds.mean, std=train_ds.std)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # Compare: original hyperparams vs improved
    configs = {
        "original (lr=1e-3)": {"lr": 1e-3, "wd": 1e-6},
        "lower_lr (lr=1e-4)": {"lr": 1e-4, "wd": 1e-5},
    }

    for name, cfg in configs.items():
        print(f"\n  --- {name} ---")
        model = TimeGradModel(
            input_dim=25, gru_hidden=128, n_diffusion_steps=100,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                      weight_decay=cfg["wd"])

        for epoch in range(15):
            model.train()
            tloss, tn = 0, 0
            for batch in train_loader:
                h = batch["history"].to(device)
                f = batch["future"].to(device)
                optimizer.zero_grad()
                loss = model(h, f)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tloss += loss.item()
                tn += 1

            if (epoch + 1) % 5 == 0:
                model.eval()
                vloss, vn = 0, 0
                with torch.no_grad():
                    for batch in val_loader:
                        h = batch["history"].to(device)
                        f = batch["future"].to(device)
                        loss = model(h, f)
                        vloss += loss.item()
                        vn += 1
                print(f"    Epoch {epoch+1}: train={tloss/tn:.4f}, val={vloss/vn:.4f}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  DEEP BASELINE IMPLEMENTATION VALIDATION")
    print("=" * 60)

    # Test 1: DDIM formula
    ddim_ok = test_ddim_formula()

    # Test 2: TimeGrad synthetic
    tg_ok, _, _ = test_timegrad_synthetic()

    # Test 3: CSDI synthetic
    csdi_ok = test_csdi_synthetic()

    # Test 4: Real data smoke test
    smoke_results = test_real_data_smoke()

    # Test 5: Overfitting diagnosis
    test_timegrad_overfit_diagnosis()

    # Summary
    print("\n" + "=" * 60)
    print("  VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  1. DDIM formula:         {'PASS' if ddim_ok else 'FAIL'}")
    print(f"  2. TimeGrad synthetic:   {'PASS' if tg_ok else 'FAIL'}")
    print(f"  3. CSDI synthetic:       {'PASS' if csdi_ok else 'FAIL'}")
    for name, res in smoke_results.items():
        print(f"  4. {name} smoke test:  {'PASS' if res['passed'] else 'FAIL'}")
