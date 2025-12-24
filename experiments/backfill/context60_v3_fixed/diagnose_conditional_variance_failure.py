"""
Diagnostic Experiments: Why CVAE Fails to Capture Conditional Distribution

This script runs 7 experiments to understand why the conditional VAE with prior network
fails to learn conditional variance despite having the architectural capability.

Key Question: If similar contexts → similar prior outputs, and similar contexts →
different outcomes, why doesn't the decoder learn to be stochastic?
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform

# Add repo root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_conditional_prior import CVAEMemRandConditionalPrior

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "models/backfill/context60_experiment/checkpoints/backfill_context60_latent12_v3_conditional_prior_phase2_ep599.pt"
DATA_PATH = "data/vol_surface_with_ret.npz"
CONTEXT_LENGTH = 60
HORIZON = 90
ATM_6M = (2, 2)  # ATM, 6-month point
K_NEIGHBORS = 30

print("="*80)
print("DIAGNOSTIC EXPERIMENTS: WHY CVAE FAILS CONDITIONAL DISTRIBUTION")
print("="*80)

# Load model
print("\nLoading model...")
model_data = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model = CVAEMemRandConditionalPrior(model_data["model_config"])

# Strip _orig_mod. prefix if present (from torch.compile)
state_dict = model_data["model"]
if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model_data["model"] = state_dict

model.load_state_dict(model_data["model"], strict=False)
model.to(DEVICE)
model.eval()
print(f"✓ Model loaded from {MODEL_PATH}")

# Load data
print("\nLoading data...")
data = np.load(DATA_PATH)
surface = torch.FloatTensor(data['surface']).to(DEVICE)

# Construct extra features (ret, skews, slopes)
ex_data = np.stack([data['ret'], data['skews'], data['slopes']], axis=-1)
ex_feats = torch.FloatTensor(ex_data).to(DEVICE)

N = surface.shape[0]
print(f"✓ Loaded {N} days of data")
print(f"✓ Extra features shape: {ex_feats.shape}")

# Prepare contexts and targets
valid_start_idx = CONTEXT_LENGTH
valid_end_idx = N - HORIZON
all_indices = list(range(valid_start_idx, valid_end_idx))
print(f"✓ {len(all_indices)} valid sequences")

# Sample subset for experiments (to speed up)
np.random.seed(42)
n_samples = min(1000, len(all_indices))
sample_indices = np.random.choice(all_indices, n_samples, replace=False)
print(f"✓ Using {n_samples} sequences for experiments")

print("\n" + "="*80)
print("EXPERIMENT 1: PRIOR NETWORK GROUPING TEST")
print("="*80)
print("Question: Does prior network group similar contexts or give unique output per context?")

prior_mus = []
prior_logvars = []
context_features = []  # Store raw context mean as feature

with torch.no_grad():
    for idx in sample_indices:
        context_surf = surface[idx-CONTEXT_LENGTH:idx]
        context_ex = ex_feats[idx-CONTEXT_LENGTH:idx]

        # Get prior network output
        context_input = {
            "surface": context_surf.unsqueeze(0),
            "ex_feats": context_ex.unsqueeze(0)
        }
        prior_mu, prior_logvar = model.prior_network(context_input)

        prior_mus.append(prior_mu.cpu().numpy().flatten())
        prior_logvars.append(prior_logvar.cpu().numpy().flatten())

        # Context feature: mean IV level
        context_features.append(context_surf.mean().item())

prior_mus = np.array(prior_mus)
prior_logvars = np.array(prior_logvars)
context_features = np.array(context_features)

# Compute pairwise distances in prior space and context space
print("\nComputing pairwise distances...")
prior_distances = squareform(pdist(prior_mus, metric='euclidean'))
context_distances = squareform(pdist(context_features.reshape(-1, 1), metric='euclidean'))

# Correlation between context similarity and prior similarity
# Flatten upper triangle (avoid diagonal and duplicates)
mask = np.triu(np.ones_like(prior_distances, dtype=bool), k=1)
prior_dist_flat = prior_distances[mask]
context_dist_flat = context_distances[mask]

correlation, p_value = pearsonr(context_dist_flat, prior_dist_flat)

print(f"\nResults:")
print(f"  Prior μ distances: mean={prior_dist_flat.mean():.4f}, std={prior_dist_flat.std():.4f}")
print(f"  Context distances: mean={context_dist_flat.mean():.4f}, std={context_dist_flat.std():.4f}")
print(f"  Correlation (context dist vs prior dist): {correlation:.4f} (p={p_value:.2e})")

if correlation > 0.8:
    print("  ❌ FINDING: Prior is VERY context-specific (high correlation)")
    print("     → Each context gets unique prior output → no grouping")
elif correlation > 0.5:
    print("  ⚠️  FINDING: Prior is moderately context-specific")
else:
    print("  ✅ FINDING: Prior groups similar contexts (low correlation)")

print("\n" + "="*80)
print("EXPERIMENT 2: POSTERIOR VS PRIOR VARIANCE")
print("="*80)
print("Question: How much tighter is the posterior than the prior?")

posterior_vars = []
prior_vars = []

with torch.no_grad():
    for idx in sample_indices[:200]:  # Use subset for speed
        context_surf = surface[idx-CONTEXT_LENGTH:idx]
        context_ex = ex_feats[idx-CONTEXT_LENGTH:idx]
        target_surf = surface[idx:idx+HORIZON]
        target_ex = ex_feats[idx:idx+HORIZON]

        context_input = {
            "surface": context_surf.unsqueeze(0),
            "ex_feats": context_ex.unsqueeze(0)
        }
        full_input = {
            "surface": torch.cat([context_surf, target_surf], dim=0).unsqueeze(0),
            "ex_feats": torch.cat([context_ex, target_ex], dim=0).unsqueeze(0)
        }

        # Posterior (from encoder)
        posterior_mu, posterior_logvar, _ = model.encoder(full_input)
        posterior_var = torch.exp(posterior_logvar).mean().item()

        # Prior (from prior network)
        prior_mu, prior_logvar = model.prior_network(context_input)
        prior_var = torch.exp(prior_logvar).mean().item()

        posterior_vars.append(posterior_var)
        prior_vars.append(prior_var)

posterior_vars = np.array(posterior_vars)
prior_vars = np.array(prior_vars)

ratio = np.mean(prior_vars) / np.mean(posterior_vars)

print(f"\nResults:")
print(f"  Mean posterior variance: {np.mean(posterior_vars):.6f}")
print(f"  Mean prior variance: {np.mean(prior_vars):.6f}")
print(f"  Prior/Posterior ratio: {ratio:.2f}x")

if ratio > 5:
    print("  ❌ FINDING: Posterior is MUCH tighter than prior (ratio > 5)")
    print("     → Decoder only sees narrow z range during training")
elif ratio > 2:
    print("  ⚠️  FINDING: Posterior is moderately tighter than prior")
else:
    print("  ✅ FINDING: Posterior and prior have similar variance")

print("\n" + "="*80)
print("EXPERIMENT 3: OUTCOME VARIANCE FOR SIMILAR CONTEXTS")
print("="*80)
print("Question: Do similar contexts actually have different outcomes? (MOST CRITICAL)")

# Extract all outcomes and predictions for sample
outcomes = []
predictions = []
context_embeddings = []

with torch.no_grad():
    for idx in sample_indices:
        context_surf = surface[idx-CONTEXT_LENGTH:idx]
        context_ex = ex_feats[idx-CONTEXT_LENGTH:idx]
        target_surf = surface[idx:idx+HORIZON]

        context_input = {
            "surface": context_surf.unsqueeze(0),
            "ex_feats": context_ex.unsqueeze(0)
        }

        # Get prior network output (use as context embedding)
        prior_mu, _ = model.prior_network(context_input)
        context_embeddings.append(prior_mu.cpu().numpy().flatten())

        # Get point prediction (sample from prior network)
        # Note: get_surface_given_conditions will use prior_network internally
        result = model.get_surface_given_conditions(
            context_input,
            z=None,  # Will sample from conditional prior
            horizon=HORIZON
        )

        # Extract surface (might be tuple if model has ex_feats)
        if isinstance(result, tuple):
            pred_surf, _ = result
        else:
            pred_surf = result

        # Outcome at H=90
        gt_value = target_surf[HORIZON-1, ATM_6M[0], ATM_6M[1]].item()
        # pred_surf shape: (B, H, num_quantiles, H, W)
        pred_value = pred_surf[0, HORIZON-1, 1, ATM_6M[0], ATM_6M[1]].item()  # p50 quantile

        outcomes.append(gt_value)
        predictions.append(pred_value)

outcomes = np.array(outcomes)
predictions = np.array(predictions)
context_embeddings = np.array(context_embeddings)

# For each context, find K nearest neighbors and check outcome variance
nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS+1, metric='euclidean').fit(context_embeddings)
distances, indices = nbrs.kneighbors(context_embeddings)

neighbor_outcome_vars = []
neighbor_residual_vars = []

for i in range(len(sample_indices)):
    # Get neighbors (excluding self)
    neighbor_idx = indices[i, 1:]  # Skip first (self)

    # Outcomes for neighbors
    neighbor_outcomes = outcomes[neighbor_idx]
    outcome_var = np.var(neighbor_outcomes)

    # Residuals for neighbors
    neighbor_residuals = outcomes[neighbor_idx] - predictions[neighbor_idx]
    residual_var = np.var(neighbor_residuals)

    neighbor_outcome_vars.append(outcome_var)
    neighbor_residual_vars.append(residual_var)

neighbor_outcome_vars = np.array(neighbor_outcome_vars)
neighbor_residual_vars = np.array(neighbor_residual_vars)

# Compare to total variance
total_variance = np.var(outcomes)
mean_neighbor_outcome_var = np.mean(neighbor_outcome_vars)
mean_neighbor_residual_var = np.mean(neighbor_residual_vars)

print(f"\nResults:")
print(f"  Total outcome variance: {total_variance:.6f}")
print(f"  Mean within-{K_NEIGHBORS}-neighbors outcome variance: {mean_neighbor_outcome_var:.6f}")
print(f"  Mean within-{K_NEIGHBORS}-neighbors residual variance: {mean_neighbor_residual_var:.6f}")
print(f"  Neighbor var / Total var: {mean_neighbor_outcome_var/total_variance*100:.1f}%")

if mean_neighbor_outcome_var / total_variance < 0.3:
    print("  ✅ FINDING: Similar contexts have SIMILAR outcomes (<30% of total var)")
    print("     → Deterministic behavior is CORRECT!")
    print("     → Problem is not in model, but in expectation")
elif mean_neighbor_outcome_var / total_variance < 0.7:
    print("  ⚠️  FINDING: Similar contexts have MODERATE outcome variance (30-70%)")
    print("     → Some stochasticity expected but not critical")
else:
    print("  ❌ FINDING: Similar contexts have VERY DIFFERENT outcomes (>70% of total var)")
    print("     → Model SHOULD be stochastic but ISN'T")
    print("     → This is the root cause!")

print("\n" + "="*80)
print("EXPERIMENT 4: Z-SPACE UTILIZATION")
print("="*80)
print("Question: Does the model use the full z-space, or collapse to points?")

z_samples_posterior = []
z_samples_prior = []

with torch.no_grad():
    for idx in sample_indices[:200]:  # Subset for speed
        context_surf = surface[idx-CONTEXT_LENGTH:idx]
        context_ex = ex_feats[idx-CONTEXT_LENGTH:idx]
        target_surf = surface[idx:idx+HORIZON]
        target_ex = ex_feats[idx:idx+HORIZON]

        context_input = {
            "surface": context_surf.unsqueeze(0),
            "ex_feats": context_ex.unsqueeze(0)
        }
        full_input = {
            "surface": torch.cat([context_surf, target_surf], dim=0).unsqueeze(0),
            "ex_feats": torch.cat([context_ex, target_ex], dim=0).unsqueeze(0)
        }

        # Sample from posterior (training regime)
        posterior_mu, posterior_logvar, _ = model.encoder(full_input)
        # Reparameterization trick: z = mu + exp(0.5 * logvar) * eps
        eps = torch.randn_like(posterior_mu)
        z_post = posterior_mu + torch.exp(0.5 * posterior_logvar) * eps
        z_samples_posterior.append(z_post.cpu().numpy().flatten())

        # Sample from prior (inference regime)
        prior_mu, prior_logvar = model.prior_network(context_input)
        eps = torch.randn_like(prior_mu)
        z_pri = prior_mu + torch.exp(0.5 * prior_logvar) * eps
        z_samples_prior.append(z_pri.cpu().numpy().flatten())

z_samples_posterior = np.array(z_samples_posterior)
z_samples_prior = np.array(z_samples_prior)

# Compute covariance trace (total variance in z-space)
posterior_cov_trace = np.trace(np.cov(z_samples_posterior.T))
prior_cov_trace = np.trace(np.cov(z_samples_prior.T))

print(f"\nResults:")
print(f"  Posterior z covariance trace: {posterior_cov_trace:.4f}")
print(f"  Prior z covariance trace: {prior_cov_trace:.4f}")
print(f"  Prior/Posterior ratio: {prior_cov_trace/posterior_cov_trace:.2f}x")

if prior_cov_trace / posterior_cov_trace > 3:
    print("  ❌ FINDING: Posterior uses MUCH narrower z-space than prior")
    print("     → Decoder never trained on wide z variations")
elif prior_cov_trace / posterior_cov_trace > 1.5:
    print("  ⚠️  FINDING: Posterior moderately narrower than prior")
else:
    print("  ✅ FINDING: Posterior and prior use similar z-space")

print("\n" + "="*80)
print("EXPERIMENT 5: DECODER SENSITIVITY ANALYSIS")
print("="*80)
print("Question: If we perturb z, does the output change?")

# Test decoder sensitivity at different perturbation scales
perturbation_scales = [0.1, 0.5, 1.0, 2.0, 5.0]
sensitivity_results = {scale: [] for scale in perturbation_scales}

with torch.no_grad():
    for idx in sample_indices[:50]:  # Subset for speed
        context_surf = surface[idx-CONTEXT_LENGTH:idx]
        context_ex = ex_feats[idx-CONTEXT_LENGTH:idx]
        context_input = {
            "surface": context_surf.unsqueeze(0),
            "ex_feats": context_ex.unsqueeze(0)
        }

        # Get baseline z from prior (shape: (B, C, latent_dim))
        z_base_context, _ = model.prior_network(context_input)

        # Create full z for T=C+H (shape: (B, T, latent_dim))
        B = 1
        T = CONTEXT_LENGTH + HORIZON
        latent_dim = model.config["latent_dim"]

        # Get context encoding (posterior mean for context)
        ctx_only = {"surface": context_surf.unsqueeze(0), "ex_feats": context_ex.unsqueeze(0)}
        ctx_latent_mean, _, _ = model.encoder(ctx_only)

        for scale in perturbation_scales:
            outputs = []
            for _ in range(20):  # 20 perturbations per scale
                # Create z with perturbed future part
                z_full = torch.zeros((B, T, latent_dim), device=DEVICE)
                z_full[:, :CONTEXT_LENGTH, :] = ctx_latent_mean  # Deterministic context

                # Perturb future part: z_future = prior_mean + scale * noise
                future_prior_mean = z_base_context[:, -1:, :].expand(B, HORIZON, -1)
                noise = torch.randn((B, HORIZON, latent_dim), device=DEVICE) * scale
                z_full[:, CONTEXT_LENGTH:, :] = future_prior_mean + noise

                # Get output
                result = model.get_surface_given_conditions(context_input, z=z_full, horizon=HORIZON)
                if isinstance(result, tuple):
                    output, _ = result
                else:
                    output = result
                output_value = output[0, HORIZON-1, 1, ATM_6M[0], ATM_6M[1]].item()
                outputs.append(output_value)

            # Measure output std
            output_std = np.std(outputs)
            sensitivity_results[scale].append(output_std)

# Average across samples
for scale in perturbation_scales:
    sensitivity_results[scale] = np.mean(sensitivity_results[scale])

print(f"\nResults:")
print(f"  Perturbation scale → Output std:")
for scale in perturbation_scales:
    print(f"    {scale:4.1f}× → {sensitivity_results[scale]:.6f}")

# Check if output std grows with perturbation
if sensitivity_results[5.0] < sensitivity_results[0.1] * 3:
    print("  ❌ FINDING: Decoder is INSENSITIVE to z perturbations")
    print("     → Output barely changes even with large z noise")
else:
    print("  ✅ FINDING: Decoder is sensitive to z variations")

print("\n" + "="*80)
print("EXPERIMENT 6: KL DIVERGENCE ANALYSIS")
print("="*80)
print("Question: What is the KL divergence between posterior and prior?")

kl_values = []

with torch.no_grad():
    for idx in sample_indices[:200]:
        context_surf = surface[idx-CONTEXT_LENGTH:idx]
        context_ex = ex_feats[idx-CONTEXT_LENGTH:idx]
        target_surf = surface[idx:idx+HORIZON]
        target_ex = ex_feats[idx:idx+HORIZON]

        context_input = {
            "surface": context_surf.unsqueeze(0),
            "ex_feats": context_ex.unsqueeze(0)
        }
        full_input = {
            "surface": torch.cat([context_surf, target_surf], dim=0).unsqueeze(0),
            "ex_feats": torch.cat([context_ex, target_ex], dim=0).unsqueeze(0)
        }

        # Posterior (full sequence: context + target)
        mu_q, logvar_q, _ = model.encoder(full_input)

        # Prior (context only)
        mu_p, logvar_p = model.prior_network(context_input)

        # KL divergence: KL(q||p) = 0.5 * sum(var_p/var_q + (mu_q-mu_p)^2/var_p - 1 + log(var_q/var_p))
        # Only compare context portion (first C timesteps)
        mu_q_ctx = mu_q[:, :CONTEXT_LENGTH, :]
        logvar_q_ctx = logvar_q[:, :CONTEXT_LENGTH, :]

        var_q = torch.exp(logvar_q_ctx)
        var_p = torch.exp(logvar_p)

        kl = 0.5 * torch.sum(
            var_q / var_p +
            (mu_q_ctx - mu_p)**2 / var_p -
            1 +
            logvar_p - logvar_q_ctx
        )
        kl_values.append(kl.item())

kl_values = np.array(kl_values)

print(f"\nResults:")
print(f"  Mean KL: {np.mean(kl_values):.4f}")
print(f"  Median KL: {np.median(kl_values):.4f}")
print(f"  Min KL: {np.min(kl_values):.4f}")
print(f"  Max KL: {np.max(kl_values):.4f}")

if np.mean(kl_values) < 1.0:
    print("  ❌ FINDING: KL is VERY small (< 1.0)")
    print("     → Posterior ≈ Prior (possible collapse)")
elif np.mean(kl_values) < 5.0:
    print("  ⚠️  FINDING: KL is small-moderate (1-5)")
else:
    print("  ✅ FINDING: KL is healthy (> 5)")

print("\n" + "="*80)
print("EXPERIMENT 7: CONTEXT EMBEDDING CLUSTERING")
print("="*80)
print("Question: Do similar market regimes cluster in embedding space?")

# Define market regimes based on context volatility level
regime_labels = []
for idx in sample_indices:
    context_surf = surface[idx-CONTEXT_LENGTH:idx]
    mean_iv = context_surf.mean().item()

    if mean_iv > 0.35:
        regime_labels.append('crisis')
    elif mean_iv > 0.25:
        regime_labels.append('high_vol')
    elif mean_iv > 0.20:
        regime_labels.append('normal')
    else:
        regime_labels.append('low_vol')

regime_labels = np.array(regime_labels)
unique_regimes = np.unique(regime_labels)

print(f"\nRegime distribution:")
for regime in unique_regimes:
    count = np.sum(regime_labels == regime)
    print(f"  {regime}: {count} ({count/len(regime_labels)*100:.1f}%)")

# t-SNE visualization
print("\nComputing t-SNE embedding...")
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(context_embeddings)

# Compute within-regime vs between-regime distances
within_regime_distances = []
between_regime_distances = []

for i in range(len(sample_indices)):
    for j in range(i+1, len(sample_indices)):
        dist = np.linalg.norm(embeddings_2d[i] - embeddings_2d[j])

        if regime_labels[i] == regime_labels[j]:
            within_regime_distances.append(dist)
        else:
            between_regime_distances.append(dist)

within_regime_distances = np.array(within_regime_distances)
between_regime_distances = np.array(between_regime_distances)

print(f"\nResults:")
print(f"  Mean within-regime distance: {np.mean(within_regime_distances):.4f}")
print(f"  Mean between-regime distance: {np.mean(between_regime_distances):.4f}")
print(f"  Ratio (between/within): {np.mean(between_regime_distances)/np.mean(within_regime_distances):.2f}x")

if np.mean(between_regime_distances) / np.mean(within_regime_distances) > 1.5:
    print("  ✅ FINDING: Regimes DO cluster in embedding space")
    print("     → Similar market conditions are grouped together")
else:
    print("  ❌ FINDING: Regimes do NOT cluster well")
    print("     → Each context treated as unique regardless of regime")

# Create visualization
plt.figure(figsize=(10, 8))
colors = {'crisis': 'red', 'high_vol': 'orange', 'normal': 'blue', 'low_vol': 'green'}
for regime in unique_regimes:
    mask = regime_labels == regime
    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                c=colors.get(regime, 'gray'), label=regime, alpha=0.6, s=20)
plt.legend()
plt.title('Context Embeddings (t-SNE)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.tight_layout()

output_dir = Path("results/context60_latent12_v3_FIXED/analysis")
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / "context_clustering_tsne.png", dpi=150)
print(f"\n✓ Saved visualization to {output_dir}/context_clustering_tsne.png")

print("\n" + "="*80)
print("SUMMARY OF FINDINGS")
print("="*80)

summary = []

# Experiment 1
if correlation > 0.8:
    summary.append("❌ Prior is VERY context-specific → no grouping of similar contexts")
elif correlation > 0.5:
    summary.append("⚠️  Prior is moderately context-specific")
else:
    summary.append("✅ Prior groups similar contexts")

# Experiment 2
if ratio > 5:
    summary.append("❌ Posterior MUCH tighter than prior → decoder sees narrow z range")
elif ratio > 2:
    summary.append("⚠️  Posterior moderately tighter than prior")
else:
    summary.append("✅ Posterior and prior have similar variance")

# Experiment 3 (MOST IMPORTANT)
if mean_neighbor_outcome_var / total_variance < 0.3:
    summary.append("✅ Similar contexts have similar outcomes → deterministic behavior is CORRECT")
elif mean_neighbor_outcome_var / total_variance < 0.7:
    summary.append("⚠️  Similar contexts have moderate variance")
else:
    summary.append("❌ Similar contexts have very different outcomes → model SHOULD be stochastic")

# Experiment 5
if sensitivity_results[5.0] < sensitivity_results[0.1] * 3:
    summary.append("❌ Decoder is INSENSITIVE to z perturbations")
else:
    summary.append("✅ Decoder is sensitive to z")

# Experiment 6
if np.mean(kl_values) < 1.0:
    summary.append("❌ KL very small → possible posterior collapse")
elif np.mean(kl_values) < 5.0:
    summary.append("⚠️  KL is small-moderate")
else:
    summary.append("✅ KL is healthy")

# Experiment 7
if np.mean(between_regime_distances) / np.mean(within_regime_distances) > 1.5:
    summary.append("✅ Regimes cluster in embedding space")
else:
    summary.append("❌ Regimes do NOT cluster well")

for i, finding in enumerate(summary, 1):
    print(f"\n{i}. {finding}")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

# Make diagnosis based on findings
if mean_neighbor_outcome_var / total_variance < 0.3:
    print("\n🎯 PRIMARY DIAGNOSIS: Model behavior is CORRECT given the data!")
    print("\nThe model is deterministic BECAUSE similar contexts have similar outcomes.")
    print("The 'problem' is not in the model, but in our EXPECTATION that it should")
    print("generate diverse scenarios for a single context.")
    print("\nThe data shows:")
    print(f"  - Similar contexts (K={K_NEIGHBORS} neighbors) have only {mean_neighbor_outcome_var/total_variance*100:.1f}% of total variance")
    print(f"  - This means similar market conditions lead to similar outcomes")
    print(f"  - Therefore, deterministic prediction is the OPTIMAL strategy")
    print("\n💡 IMPLICATION: To get conditional variance, we need to:")
    print("   1. Accept that similar contexts → similar outcomes (use K-NN variance)")
    print("   2. Inject variance artificially via bootstrap/conformal methods")
    print("   3. OR: Retrain with explicit stochasticity objective")
else:
    print("\n🎯 PRIMARY DIAGNOSIS: Training/Architecture Issue")
    print(f"\nSimilar contexts have {mean_neighbor_outcome_var/total_variance*100:.1f}% of total variance,")
    print("but the model is not capturing this conditional uncertainty.")

    if ratio > 5:
        print("\n🔍 ROOT CAUSE: Posterior is much tighter than prior ({}x)".format(ratio))
        print("   → Decoder only trained on narrow z range")
        print("   → At inference, decoder ignores wider prior variance")
        print("\n💡 SOLUTION: Train with prior samples (Solution F)")

    if correlation > 0.8:
        print("\n🔍 CONTRIBUTING FACTOR: Prior is too context-specific")
        print("   → Each context gets unique prior output")
        print("   → No grouping of similar contexts")
        print("\n💡 SOLUTION: Add regularization to prior network to group similar contexts")

    if sensitivity_results[5.0] < sensitivity_results[0.1] * 3:
        print("\n🔍 CONTRIBUTING FACTOR: Decoder is insensitive to z")
        print("   → Large z perturbations produce small output changes")
        print("\n💡 SOLUTION: Use inverse Lipschitz constraint on decoder")

print("\n" + "="*80)
print("END OF DIAGNOSTIC EXPERIMENTS")
print("="*80)
