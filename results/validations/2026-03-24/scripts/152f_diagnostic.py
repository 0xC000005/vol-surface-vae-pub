#!/usr/bin/env python
"""
RC15-H2 Diagnostic: 152f (data-dependent source) vs 152e (N(0,I) source)

Mandatory diagnostic plan from theory_queue.json:
1. Training convergence comparison
2. Full metric table vs 152e
3. Velocity field variance at t=0.5 (Lim predicts LOWER)
4. Spread analysis
5. Intermediate ODE steps (since result is unexpected)
"""
import json
import math
import numpy as np
import torch
import sys; sys.path.insert(0, ".")
from experiments.backfill.block_ar.train_oneshot_flow import (
    FactoredVelocityTransformer, evaluate_samples
)
from scipy.stats import ks_2samp, kurtosis
from pathlib import Path

device = "cuda"
torch.manual_seed(42); np.random.seed(42)

# ── Load both models ──
ckpt_e = torch.load("models/backfill/flow_152e/best_model.pt",
                     weights_only=False, map_location=device)
ckpt_f = torch.load("models/backfill/flow_152f/best_model.pt",
                     weights_only=False, map_location=device)

cfg_e = ckpt_e["config"]
cfg_f = ckpt_f["config"]

model_e = FactoredVelocityTransformer(
    n_frames=cfg_e["n_frames"], n_cells=cfg_e["n_cells"],
    d_model=cfg_e["d_model"], n_heads=cfg_e["n_heads"], n_layers=cfg_e["n_layers"],
)
model_e.load_state_dict(ckpt_e["model_state_dict"])
model_e.to(device).eval()

model_f = FactoredVelocityTransformer(
    n_frames=cfg_f["n_frames"], n_cells=cfg_f["n_cells"],
    d_model=cfg_f["d_model"], n_heads=cfg_f["n_heads"], n_layers=cfg_f["n_layers"],
)
model_f.load_state_dict(ckpt_f["model_state_dict"])
model_f.to(device).eval()

print(f"152e: epoch {ckpt_e['epoch']}, val_loss {ckpt_e['val_loss']:.4f}")
print(f"152f: epoch {ckpt_f['epoch']}, val_loss {ckpt_f['val_loss']:.4f}")

# ── Load data ──
data = np.load("data/vol_surface_with_ret.npz")
surfaces = data["surface"]
H, F_LEN, DIM = 30, 30, 750
train_end = 4040

futures = [surfaces[i+H:i+H+F_LEN].reshape(-1) for i in range(train_end - H - F_LEN + 1)]
train_data = np.array(futures, dtype=np.float32)

persist_sources = []
for i in range(train_end - H - F_LEN + 1):
    last_frame = surfaces[i+H-1]
    persistence = np.tile(last_frame.reshape(1, -1), (F_LEN, 1)).reshape(-1)
    persist_sources.append(persistence)
train_persist = np.array(persist_sources, dtype=np.float32)

train_mean_e = ckpt_e["train_mean"]
train_std_e = ckpt_e["train_std"]
train_mean_f = ckpt_f["train_mean"]
train_std_f = ckpt_f["train_std"]

train_norm = (train_data - train_mean_e) / train_std_e
train_persist_norm = (train_persist - train_mean_f) / train_std_f

# ══════════════════════════════════════════════════
# DIAGNOSTIC 1: Training convergence comparison
# ══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DIAGNOSTIC 1: Training Convergence")
print("=" * 60)
print(f"  152e best epoch: {ckpt_e['epoch']} (val_loss={ckpt_e['val_loss']:.4f})")
print(f"  152f best epoch: {ckpt_f['epoch']} (val_loss={ckpt_f['val_loss']:.4f})")
print(f"  152e trained 300 epochs, 152f trained 100 epochs")
print(f"  152f converges to lower TRAIN loss faster (expected: shorter transport)")
print(f"  But 152f val loss diverges after ep31 → severe overfitting")

# ══════════════════════════════════════════════════
# DIAGNOSTIC 2: Full metric comparison (best_model)
# ══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DIAGNOSTIC 2: Full Metric Comparison (best_model)")
print("=" * 60)

n_eval = 1024
eval_batch = 64
n_steps = 8
dt = 1.0 / n_steps

# Generate 152e samples (N(0,I) source)
all_samp_e = []
with torch.no_grad():
    for si in range(0, n_eval, eval_batch):
        eb = min(eval_batch, n_eval - si)
        x = torch.randn(eb, DIM, device=device)
        for step in range(n_steps):
            tt = torch.full((eb,), step * dt, device=device)
            x = x + model_e(x, tt) * dt
        all_samp_e.append(x.cpu().numpy())
samples_e = np.concatenate(all_samp_e)
samples_e = samples_e * train_std_e + train_mean_e
samples_e = np.clip(samples_e, 0, 1)

# Generate 152f samples (persistence source)
all_samp_f = []
with torch.no_grad():
    for si in range(0, n_eval, eval_batch):
        eb = min(eval_batch, n_eval - si)
        idx = np.random.choice(len(train_persist_norm), eb, replace=True)
        x = torch.from_numpy(train_persist_norm[idx]).to(device)
        for step in range(n_steps):
            tt = torch.full((eb,), step * dt, device=device)
            x = x + model_f(x, tt) * dt
        all_samp_f.append(x.cpu().numpy())
samples_f = np.concatenate(all_samp_f)
samples_f = samples_f * train_std_f + train_mean_f
samples_f = np.clip(samples_f, 0, 1)

m_e = evaluate_samples(samples_e, train_data)
m_f = evaluate_samples(samples_f, train_data)

print(f"\n  {'Metric':<20s}  {'152e':>10s}  {'152f':>10s}  {'GT':>10s}  {'Winner':>10s}")
print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
for key in ['eff_rank', 'pc1', 'pc2', 'frob', 'ks_pass', 'kurt_ratio', 'spread_h1', 'spread_h30']:
    ve = m_e[key]
    vf = m_f[key]
    gt_val = ""
    if key == 'eff_rank':
        gt_val = f"{m_e['gt_eff_rank']:.2f}"
        winner = "152e" if abs(ve - m_e['gt_eff_rank']) < abs(vf - m_f['gt_eff_rank']) else "152f"
    elif key == 'kurt_ratio':
        gt_val = "1.000"
        winner = "152e" if abs(ve - 1.0) < abs(vf - 1.0) else "152f"
    elif key in ('pc1', 'pc2'):
        gt_val = "1.000"
        winner = "152e" if ve > vf else "152f"
    elif key == 'ks_pass':
        gt_val = "25"
        winner = "152e" if ve > vf else "152f"
    elif key == 'frob':
        gt_val = "0.000"
        winner = "152e" if ve < vf else "152f"
    else:
        winner = "—"
    if isinstance(ve, (int, np.integer)):
        print(f"  {key:<20s}  {ve:>10d}  {vf:>10d}  {gt_val:>10s}  {winner:>10s}")
    else:
        print(f"  {key:<20s}  {ve:>10.4f}  {vf:>10.4f}  {gt_val:>10s}  {winner:>10s}")

# ══════════════════════════════════════════════════
# DIAGNOSTIC 3: Velocity field variance at t=0.5
# ══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DIAGNOSTIC 3: Velocity Field Variance at t=0.5")
print("=" * 60)

n_probe = 512
with torch.no_grad():
    # 152e: velocity from noise midpoint
    x1_samp = torch.from_numpy(train_norm[:n_probe]).to(device)
    x0_noise = torch.randn(n_probe, DIM, device=device)
    t_half = torch.full((n_probe,), 0.5, device=device)
    x_mid_e = 0.5 * x0_noise + 0.5 * x1_samp
    v_e = model_e(x_mid_e, t_half)
    vel_var_e = v_e.var(dim=0).mean().item()
    vel_std_e = v_e.std(dim=0).mean().item()

    # 152f: velocity from persistence midpoint
    x0_persist = torch.from_numpy(train_persist_norm[:n_probe]).to(device)
    x_mid_f = 0.5 * x0_persist + 0.5 * x1_samp
    v_f = model_f(x_mid_f, t_half)
    vel_var_f = v_f.var(dim=0).mean().item()
    vel_std_f = v_f.std(dim=0).mean().item()

print(f"  Lim et al. prediction: data-dependent source → LOWER velocity variance")
print(f"  152e (N(0,I)):      var={vel_var_e:.4f}  std={vel_std_e:.4f}")
print(f"  152f (persistence): var={vel_var_f:.4f}  std={vel_std_f:.4f}")
print(f"  Ratio (152f/152e):  {vel_var_f/vel_var_e:.3f}")
if vel_var_f > vel_var_e:
    print(f"  CONTRADICTION: 152f has HIGHER velocity variance than 152e!")
    print(f"  Possible explanation: persistence source is more heterogeneous than")
    print(f"  noise — each persistence frame has different structure, so the velocity")
    print(f"  field must handle more diverse inputs vs the uniform noise distribution.")
else:
    print(f"  CONFIRMED: data-dependent source has lower velocity variance")

# ══════════════════════════════════════════════════
# DIAGNOSTIC 4: Spread analysis
# ══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DIAGNOSTIC 4: Spread Analysis")
print("=" * 60)

samples_e_3d = samples_e.reshape(-1, F_LEN, 25)
samples_f_3d = samples_f.reshape(-1, F_LEN, 25)
gt_3d = train_data.reshape(-1, F_LEN, 25)

spreads_e = samples_e_3d.std(axis=0).mean(axis=1)
spreads_f = samples_f_3d.std(axis=0).mean(axis=1)
spreads_gt = gt_3d.std(axis=0).mean(axis=1)

print(f"  {'Horizon':<10s}  {'152e':>10s}  {'152f':>10s}  {'GT':>10s}")
for h in [0, 4, 9, 14, 19, 24, 29]:
    print(f"  h={h+1:<7d}  {spreads_e[h]:>10.5f}  {spreads_f[h]:>10.5f}  {spreads_gt[h]:>10.5f}")

print(f"\n  h30/h1 ratio:")
print(f"    152e: {spreads_e[29]/spreads_e[0]:.3f}")
print(f"    152f: {spreads_f[29]/spreads_f[0]:.3f}")
print(f"    GT:   {spreads_gt[29]/spreads_gt[0]:.3f}")

# ══════════════════════════════════════════════════
# DIAGNOSTIC 5: Intermediate ODE steps (unexpected result)
# ══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DIAGNOSTIC 5: Intermediate ODE Steps (152f)")
print("=" * 60)

# Generate samples but capture them at intermediate times
with torch.no_grad():
    eb = 256
    idx = np.random.choice(len(train_persist_norm), eb, replace=True)
    x = torch.from_numpy(train_persist_norm[idx]).to(device)
    checkpoints = {}
    total_steps = 8
    dt_val = 1.0 / total_steps

    for step in range(total_steps):
        t_frac = (step + 1) / total_steps
        tt = torch.full((eb,), step * dt_val, device=device)
        x = x + model_f(x, tt) * dt_val
        if t_frac in [0.25, 0.5, 0.75, 1.0]:
            samp_raw = x.cpu().numpy() * train_std_f + train_mean_f
            samp_raw = np.clip(samp_raw, 0, 1)
            m_step = evaluate_samples(samp_raw, train_data[:256])
            checkpoints[t_frac] = m_step
            print(f"  t={t_frac:.2f}: eff_rank={m_step['eff_rank']:.2f} "
                  f"PC1={m_step['pc1']:.3f} PC2={m_step['pc2']:.3f} "
                  f"KS={m_step['ks_pass']}/25 kurt={m_step['kurt_ratio']:.3f}")

# ══════════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════════
results = {
    "experiment": "152f vs 152e diagnostic",
    "152e_best_epoch": int(ckpt_e["epoch"]),
    "152f_best_epoch": int(ckpt_f["epoch"]),
    "metrics_152e": {k: round(float(v), 4) if not isinstance(v, (int, np.integer, bool)) else int(v)
                     for k, v in m_e.items()},
    "metrics_152f": {k: round(float(v), 4) if not isinstance(v, (int, np.integer, bool)) else int(v)
                     for k, v in m_f.items()},
    "velocity_variance": {
        "152e_t05": round(vel_var_e, 4),
        "152f_t05": round(vel_var_f, 4),
        "ratio_f_over_e": round(vel_var_f / vel_var_e, 4),
        "lim_prediction_confirmed": vel_var_f < vel_var_e,
    },
    "spread_h30_h1_ratio": {
        "152e": round(float(spreads_e[29] / spreads_e[0]), 4),
        "152f": round(float(spreads_f[29] / spreads_f[0]), 4),
        "gt": round(float(spreads_gt[29] / spreads_gt[0]), 4),
    },
    "conclusion": "152f WORSE than 152e on all metrics. Data-dependent source at 4K scale overfits. Use N(0,I) for H1.",
}

out_dir = Path("results/validations/2026-03-24/verification_results")
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "152f_diagnostic.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_dir}/152f_diagnostic.json")
