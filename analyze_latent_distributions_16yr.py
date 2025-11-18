"""
Latent Distribution Analysis for backfill_16yr model.

TESTS LATENT UTILIZATION via Distribution Properties:
If latent is being used effectively:
  - Context latents (encoded from real data) should have LOWER variance
  - Future latents (sampled from N(0,1)) should have variance ≈ 1.0
  - Future latents should be closer to standard normal
  - Per-dimension analysis should match VAE health metrics

This script:
1. Extracts latent values for context vs future timesteps
2. Compares variance, kurtosis, skewness
3. Plots histograms for each dimension
4. Identifies which dimensions are active vs collapsed
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.utils import set_seeds
from tqdm import tqdm
from scipy import stats

# Set seeds and dtype
set_seeds(0)
torch.set_default_dtype(torch.float64)

print("=" * 80)
print("LATENT DISTRIBUTION ANALYSIS - backfill_16yr")
print("=" * 80)
print()
print("Objective: Analyze latent variable distributions")
print()
print("Expected if latent IS used:")
print("  ✓ Context latents: variance < 1 (compressed by encoder)")
print("  ✓ Future latents (sampled): variance ≈ 1 (from N(0,1))")
print("  ✓ High-variance dims show wider distributions")
print()

# ============================================================================
# Load Model
# ============================================================================

print("Loading model...")
model_file = "models_backfill/backfill_16yr.pt"
model_data = torch.load(model_file, weights_only=False)
model_config = model_data["model_config"]

latent_dim = model_config['latent_dim']
print(f"Model config:")
print(f"  Context length: {model_config['context_len']}")
print(f"  Latent dim: {latent_dim}")

model = CVAEMemRand(model_config)
model.load_weights(dict_to_load=model_data)
model.eval()
device = model.device
print("✓ Model loaded")

# ============================================================================
# Load Data
# ============================================================================

print("\nLoading data...")
data = np.load("data/vol_surface_with_ret.npz")
vol_surf_data = data["surface"]
ret_data = data["ret"]
skew_data = data["skews"]
slope_data = data["slopes"]

ex_data = np.concatenate([
    ret_data[..., np.newaxis],
    skew_data[..., np.newaxis],
    slope_data[..., np.newaxis]
], axis=-1)

print(f"  Data shape: {vol_surf_data.shape}")

# ============================================================================
# Test Configuration
# ============================================================================

train_start = 1000
train_end = 5000
context_len = model_config['context_len']
horizon = 30  # Use H30 for analysis
num_samples = 1000  # Number of sequences to analyze

print(f"\nTest Configuration:")
print(f"  Test period: indices [{train_start}, {train_end}]")
print(f"  Context length: {context_len}")
print(f"  Horizon: {horizon}")
print(f"  Number of samples: {num_samples}")
print()

# ============================================================================
# Extract Latent Values
# ============================================================================

print("=" * 80)
print("EXTRACTING LATENT VALUES")
print("=" * 80)
print()

# Storage for latent values
context_latents = []  # Latents from encoding context
future_latents_sampled = []  # Latents sampled from N(0,1)
future_latents_encoded = []  # Latents from encoding full sequence (oracle)

# Available test days
min_idx = train_start + context_len
max_idx = train_end - horizon
available_days = list(range(min_idx, max_idx))

# Subsample
np.random.seed(42)
test_days = sorted(np.random.choice(available_days, size=num_samples, replace=False))

print(f"Extracting latents from {len(test_days)} sequences...")
print()

with torch.no_grad():
    for day_idx in tqdm(test_days, desc="Extracting latents"):
        # 1. Extract FULL sequence (context + target)
        full_sequence = vol_surf_data[day_idx - context_len : day_idx + horizon]
        full_ex = ex_data[day_idx - context_len : day_idx + horizon]

        # 2. Extract CONTEXT ONLY
        context_sequence = vol_surf_data[day_idx - context_len : day_idx]
        context_ex = ex_data[day_idx - context_len : day_idx]

        # 3. Encode FULL sequence (oracle - with target)
        full_input = {
            "surface": torch.from_numpy(full_sequence).unsqueeze(0).to(device),
            "ex_feats": torch.from_numpy(full_ex).unsqueeze(0).to(device)
        }
        full_latent_mean, _, _ = model.encoder(full_input)  # (1, C+H, latent_dim)

        # Split into context and future
        ctx_latent_full = full_latent_mean[0, :context_len, :].cpu().numpy()  # (C, latent_dim)
        future_latent_encoded_seq = full_latent_mean[0, context_len:, :].cpu().numpy()  # (H, latent_dim)

        # 4. Encode CONTEXT ONLY (realistic generation)
        ctx_input = {
            "surface": torch.from_numpy(context_sequence).unsqueeze(0).to(device),
            "ex_feats": torch.from_numpy(context_ex).unsqueeze(0).to(device)
        }
        ctx_latent_mean, _, _ = model.encoder(ctx_input)  # (1, C, latent_dim)
        ctx_latent = ctx_latent_mean[0].cpu().numpy()  # (C, latent_dim)

        # 5. Sample future latents from N(0,1) (realistic generation)
        z_future_sampled = torch.randn(horizon, latent_dim).numpy()  # (H, latent_dim)

        # Store
        context_latents.append(ctx_latent)  # (C, latent_dim)
        future_latents_sampled.append(z_future_sampled)  # (H, latent_dim)
        future_latents_encoded.append(future_latent_encoded_seq)  # (H, latent_dim)

# Convert to arrays
context_latents = np.concatenate(context_latents, axis=0)  # (N*C, latent_dim)
future_latents_sampled = np.concatenate(future_latents_sampled, axis=0)  # (N*H, latent_dim)
future_latents_encoded = np.concatenate(future_latents_encoded, axis=0)  # (N*H, latent_dim)

print(f"✓ Extracted latents:")
print(f"  Context (encoded): {context_latents.shape}")
print(f"  Future (sampled from N(0,1)): {future_latents_sampled.shape}")
print(f"  Future (encoded oracle): {future_latents_encoded.shape}")
print()

# ============================================================================
# Statistical Analysis
# ============================================================================

print("=" * 80)
print("STATISTICAL ANALYSIS")
print("=" * 80)
print()

print(f"{'Dimension':<12} {'Type':<20} {'Mean':<12} {'Variance':<12} {'Skewness':<12} {'Kurtosis':<12}")
print("-" * 80)

stats_summary = []

for dim in range(latent_dim):
    # Context latents (encoded)
    ctx_values = context_latents[:, dim]
    ctx_mean = np.mean(ctx_values)
    ctx_var = np.var(ctx_values)
    ctx_skew = stats.skew(ctx_values)
    ctx_kurt = stats.kurtosis(ctx_values)

    print(f"Dim {dim:<9} Context (encoded)   {ctx_mean:<12.4f} {ctx_var:<12.4f} {ctx_skew:<12.4f} {ctx_kurt:<12.4f}")

    # Future latents (sampled from N(0,1))
    future_sampled_values = future_latents_sampled[:, dim]
    fs_mean = np.mean(future_sampled_values)
    fs_var = np.var(future_sampled_values)
    fs_skew = stats.skew(future_sampled_values)
    fs_kurt = stats.kurtosis(future_sampled_values)

    print(f"{'':12} Future (sampled)    {fs_mean:<12.4f} {fs_var:<12.4f} {fs_skew:<12.4f} {fs_kurt:<12.4f}")

    # Future latents (encoded oracle)
    future_enc_values = future_latents_encoded[:, dim]
    fe_mean = np.mean(future_enc_values)
    fe_var = np.var(future_enc_values)
    fe_skew = stats.skew(future_enc_values)
    fe_kurt = stats.kurtosis(future_enc_values)

    print(f"{'':12} Future (encoded)    {fe_mean:<12.4f} {fe_var:<12.4f} {fe_skew:<12.4f} {fe_kurt:<12.4f}")
    print()

    stats_summary.append({
        'dimension': dim,
        'ctx_mean': ctx_mean,
        'ctx_var': ctx_var,
        'ctx_skew': ctx_skew,
        'ctx_kurt': ctx_kurt,
        'future_sampled_mean': fs_mean,
        'future_sampled_var': fs_var,
        'future_sampled_skew': fs_skew,
        'future_sampled_kurt': fs_kurt,
        'future_encoded_mean': fe_mean,
        'future_encoded_var': fe_var,
        'future_encoded_skew': fe_skew,
        'future_encoded_kurt': fe_kurt,
    })

# ============================================================================
# Variance Analysis
# ============================================================================

print("=" * 80)
print("VARIANCE COMPARISON")
print("=" * 80)
print()

print("Expected:")
print("  - Future (sampled): variance ≈ 1.0 (from N(0,1))")
print("  - Context (encoded): variance < 1.0 (compressed)")
print("  - Future (encoded): variance depends on data variability")
print()

print(f"{'Dimension':<12} {'Context Var':<15} {'Future Sampled':<18} {'Future Encoded':<18} {'Interpretation'}")
print("-" * 90)

for s in stats_summary:
    dim = s['dimension']
    ctx_var = s['ctx_var']
    fs_var = s['future_sampled_var']
    fe_var = s['future_encoded_var']

    # Interpretation
    if fe_var > 0.1:
        interpretation = "ACTIVE (high posterior variance)"
    elif fe_var > 0.01:
        interpretation = "MODERATE (medium variance)"
    else:
        interpretation = "COLLAPSED (low variance)"

    print(f"Dim {dim:<9} {ctx_var:<15.4f} {fs_var:<18.4f} {fe_var:<18.4f} {interpretation}")

print()

# Check if sampled future is close to N(0,1)
avg_future_sampled_var = np.mean([s['future_sampled_var'] for s in stats_summary])
print(f"Average future (sampled) variance: {avg_future_sampled_var:.4f}")
if 0.95 < avg_future_sampled_var < 1.05:
    print("✓ Future sampled variance ≈ 1.0 (correct sampling from N(0,1))")
else:
    print("⚠ Future sampled variance deviates from 1.0")

print()

# ============================================================================
# Visualization
# ============================================================================

print("=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)
print()

output_dir = "models_backfill/latent_distribution_figs"
import os
os.makedirs(output_dir, exist_ok=True)

# Plot 1: Per-dimension histograms (context vs future sampled vs future encoded)
fig, axes = plt.subplots(latent_dim, 1, figsize=(12, 3 * latent_dim))
if latent_dim == 1:
    axes = [axes]

for dim in range(latent_dim):
    ax = axes[dim]

    # Histogram data
    ctx_values = context_latents[:, dim]
    future_sampled_values = future_latents_sampled[:, dim]
    future_enc_values = future_latents_encoded[:, dim]

    # Plot histograms
    bins = np.linspace(-3, 3, 60)
    ax.hist(ctx_values, bins=bins, alpha=0.4, label='Context (encoded)', color='blue', density=True)
    ax.hist(future_sampled_values, bins=bins, alpha=0.4, label='Future (sampled N(0,1))', color='green', density=True)
    ax.hist(future_enc_values, bins=bins, alpha=0.4, label='Future (encoded)', color='red', density=True)

    # Add N(0,1) reference
    x = np.linspace(-3, 3, 100)
    ax.plot(x, stats.norm.pdf(x, 0, 1), 'k--', linewidth=2, label='N(0,1) reference')

    # Formatting
    ax.set_xlabel('Latent Value')
    ax.set_ylabel('Density')
    ax.set_title(f'Dimension {dim} (Var: ctx={stats_summary[dim]["ctx_var"]:.3f}, '
                 f'sampled={stats_summary[dim]["future_sampled_var"]:.3f}, '
                 f'encoded={stats_summary[dim]["future_encoded_var"]:.3f})')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plot1_file = f"{output_dir}/latent_distributions_per_dimension.png"
plt.savefig(plot1_file, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {plot1_file}")
plt.close()

# Plot 2: Variance comparison bar chart
fig, ax = plt.subplots(figsize=(10, 6))

dimensions = list(range(latent_dim))
ctx_vars = [s['ctx_var'] for s in stats_summary]
fs_vars = [s['future_sampled_var'] for s in stats_summary]
fe_vars = [s['future_encoded_var'] for s in stats_summary]

x = np.arange(latent_dim)
width = 0.25

ax.bar(x - width, ctx_vars, width, label='Context (encoded)', color='blue', alpha=0.7)
ax.bar(x, fs_vars, width, label='Future (sampled)', color='green', alpha=0.7)
ax.bar(x + width, fe_vars, width, label='Future (encoded)', color='red', alpha=0.7)

# Add reference line at variance = 1.0
ax.axhline(1.0, color='black', linestyle='--', linewidth=2, label='N(0,1) variance')
ax.axhline(0.01, color='gray', linestyle=':', linewidth=1, label='Collapse threshold')

ax.set_xlabel('Latent Dimension')
ax.set_ylabel('Variance')
ax.set_title('Latent Variable Variance by Dimension')
ax.set_xticks(x)
ax.set_xticklabels([f'Dim {d}' for d in dimensions])
ax.legend()
ax.grid(alpha=0.3, axis='y')

plot2_file = f"{output_dir}/latent_variance_comparison.png"
plt.savefig(plot2_file, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {plot2_file}")
plt.close()

# Plot 3: Scatter plot (context variance vs future encoded variance)
fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(ctx_vars, fe_vars, s=200, alpha=0.7)

for dim in range(latent_dim):
    ax.annotate(f'Dim {dim}', (ctx_vars[dim], fe_vars[dim]),
                xytext=(5, 5), textcoords='offset points', fontsize=10)

ax.set_xlabel('Context Variance (encoded from real data)')
ax.set_ylabel('Future Variance (encoded oracle)')
ax.set_title('Context vs Future Latent Variance')
ax.plot([0, max(ctx_vars + fe_vars)], [0, max(ctx_vars + fe_vars)], 'k--', alpha=0.3, label='y=x')
ax.legend()
ax.grid(alpha=0.3)

plot3_file = f"{output_dir}/context_vs_future_variance.png"
plt.savefig(plot3_file, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {plot3_file}")
plt.close()

print()

# ============================================================================
# Kolmogorov-Smirnov Test (Normality Check)
# ============================================================================

print("=" * 80)
print("NORMALITY TESTS (Kolmogorov-Smirnov)")
print("=" * 80)
print()

print("Testing if future (sampled) latents are actually N(0,1)...")
print()
print(f"{'Dimension':<12} {'KS Statistic':<18} {'p-value':<15} {'Interpretation'}")
print("-" * 60)

for dim in range(latent_dim):
    future_sampled_values = future_latents_sampled[:, dim]

    # KS test against N(0,1)
    ks_stat, p_value = stats.kstest(future_sampled_values, 'norm', args=(0, 1))

    if p_value > 0.05:
        interpretation = "✓ Normal (cannot reject H0)"
    else:
        interpretation = "✗ Non-normal (reject H0)"

    print(f"Dim {dim:<9} {ks_stat:<18.6f} {p_value:<15.6f} {interpretation}")

print()
print("Note: High p-value (>0.05) means distribution is consistent with N(0,1)")
print()

# ============================================================================
# Final Verdict
# ============================================================================

print("=" * 80)
print("FINAL VERDICT")
print("=" * 80)
print()

# Count active dimensions
active_dims = sum(1 for s in stats_summary if s['future_encoded_var'] > 0.1)
moderate_dims = sum(1 for s in stats_summary if 0.01 < s['future_encoded_var'] <= 0.1)
collapsed_dims = sum(1 for s in stats_summary if s['future_encoded_var'] <= 0.01)

print(f"Dimension Classification (by future encoded variance):")
print(f"  ✓ ACTIVE (var > 0.1):     {active_dims}/{latent_dim}")
print(f"  ~ MODERATE (0.01 < var ≤ 0.1): {moderate_dims}/{latent_dim}")
print(f"  ✗ COLLAPSED (var ≤ 0.01): {collapsed_dims}/{latent_dim}")
print()

# Check if sampling is working correctly
avg_sampled_var = np.mean([s['future_sampled_var'] for s in stats_summary])
if 0.95 < avg_sampled_var < 1.05:
    print("✓ Future latent sampling is CORRECT (variance ≈ 1.0)")
else:
    print(f"⚠ Future latent sampling variance = {avg_sampled_var:.3f} (expected ≈ 1.0)")

print()

# Compare context vs future encoded variance
avg_ctx_var = np.mean([s['ctx_var'] for s in stats_summary])
avg_future_enc_var = np.mean([s['future_encoded_var'] for s in stats_summary])

print(f"Average variance comparison:")
print(f"  Context (encoded):    {avg_ctx_var:.4f}")
print(f"  Future (sampled):     {avg_sampled_var:.4f}")
print(f"  Future (encoded):     {avg_future_enc_var:.4f}")
print()

if avg_future_enc_var > avg_ctx_var:
    print("✓ Future has higher variance than context (expected - more uncertainty)")
else:
    print("⚠ Context has higher variance than future (unexpected)")

print()

if active_dims >= 2:
    print("CONCLUSION: ✅ Latent space IS being utilized")
    print(f"At least {active_dims} dimensions show meaningful variance.")
    print("Distribution analysis supports effective latent usage.")
elif active_dims >= 1:
    print("CONCLUSION: ⚠️ Partial latent utilization")
    print(f"Only {active_dims} dimension(s) show high variance.")
else:
    print("CONCLUSION: ❌ Minimal latent utilization")
    print("Most dimensions show collapsed variance.")

print()

# ============================================================================
# Save Results
# ============================================================================

output_file = "models_backfill/latent_distributions_16yr.npz"
print(f"Saving results to {output_file}...")

np.savez(
    output_file,
    context_latents=context_latents,
    future_latents_sampled=future_latents_sampled,
    future_latents_encoded=future_latents_encoded,
    stats_summary=np.array([(
        s['dimension'],
        s['ctx_mean'],
        s['ctx_var'],
        s['future_sampled_var'],
        s['future_encoded_var']
    ) for s in stats_summary], dtype=[
        ('dimension', 'i4'),
        ('ctx_mean', 'f8'),
        ('ctx_var', 'f8'),
        ('future_sampled_var', 'f8'),
        ('future_encoded_var', 'f8')
    ])
)

print("✓ Results saved")
print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print()
print(f"Figures saved to: {output_dir}/")
print("  - latent_distributions_per_dimension.png")
print("  - latent_variance_comparison.png")
print("  - context_vs_future_variance.png")
