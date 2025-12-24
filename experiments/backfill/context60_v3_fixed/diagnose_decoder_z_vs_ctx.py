"""
Diagnostic Experiments: Does Decoder Ignore z in Favor of Context?

This script tests whether the decoder relies on the context embedding (ctx) and LSTM
hidden state rather than the latent variable (z) for generating outputs.

Key Question: The decoder receives decoder_input = ctx_embedding || z. For future
positions (t >= C), ctx_embedding is zeros but LSTM hidden state has accumulated
context. Does the decoder rely on hidden state and ignore z?
"""

import sys
import torch
import numpy as np
from pathlib import Path

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

print("="*80)
print("DIAGNOSTIC EXPERIMENTS: DECODER Z VS CTX ATTRIBUTION")
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

# Sample subset for experiments
np.random.seed(42)
n_samples = min(200, len(all_indices))
sample_indices = np.random.choice(all_indices, n_samples, replace=False)
print(f"✓ Using {n_samples} sequences for experiments")

print("\n" + "="*80)
print("EXPERIMENT 8: CONTEXT ZEROING TEST")
print("="*80)
print("Question: If we zero out ALL context embeddings, can z alone drive output diversity?")
print()
print("Test: Sample same context 100× with different z values")
print("  - Condition 1: Normal (ctx + z)")
print("  - Condition 2: Zeroed ctx (0 + z)")
print()

# Pick one context for this test
test_idx = sample_indices[0]
context_surf = surface[test_idx-CONTEXT_LENGTH:test_idx]
context_ex = ex_feats[test_idx-CONTEXT_LENGTH:test_idx]
target_surf = surface[test_idx:test_idx+HORIZON]

context_input = {
    "surface": context_surf.unsqueeze(0),
    "ex_feats": context_ex.unsqueeze(0)
}

# Helper function to manually call decoder with modified inputs
def decode_with_modified_ctx(model, context_input, z_samples, zero_ctx=False):
    """
    Manually construct decoder input with optional ctx zeroing.

    Args:
        model: CVAEMemRandConditionalPrior
        context_input: dict with surface, ex_feats
        z_samples: (B, T, latent_dim) - latent samples
        zero_ctx: if True, zero out ctx_embedding

    Returns:
        decoded_surface: (B, H, num_quantiles, H, W)
    """
    B = z_samples.shape[0]
    C = context_input["surface"].shape[1]
    T = z_samples.shape[1]
    H = T - C

    # Get context embedding
    ctx_embedding = model.ctx_encoder(context_input)  # (B, C, ctx_dim)
    ctx_embedding_dim = ctx_embedding.shape[2]

    # Pad context embedding
    ctx_embedding_padded = torch.zeros((B, T, ctx_embedding_dim), device=DEVICE)
    if not zero_ctx:
        ctx_embedding_padded[:, :C, :] = ctx_embedding
    # else: leave all zeros (testing z-only)

    # Decoder input
    decoder_input = torch.cat([ctx_embedding_padded, z_samples], dim=-1)

    # Decode
    if model.config["ex_feats_dim"] > 0:
        decoded_surface, decoded_ex_feat = model.decoder(decoder_input)
    else:
        decoded_surface = model.decoder(decoder_input)

    # Return future portion
    return decoded_surface[:, C:, :, :, :]

# Generate 100 samples with normal ctx
outputs_normal = []
outputs_zero_ctx = []

with torch.no_grad():
    for i in range(100):
        # Sample z from prior
        prior_mu, prior_logvar = model.prior_network(context_input)

        # Create full z sequence (B, T, latent_dim)
        B = 1
        T = CONTEXT_LENGTH + HORIZON
        latent_dim = model.config["latent_dim"]

        # Get context encoding (posterior mean for context)
        ctx_latent_mean, _, _ = model.encoder(context_input)

        # Construct z: deterministic context + sampled future
        z_full = torch.zeros((B, T, latent_dim), device=DEVICE)
        z_full[:, :CONTEXT_LENGTH, :] = ctx_latent_mean

        # Sample future from prior
        future_prior_mean = prior_mu[:, -1:, :].expand(B, HORIZON, -1)
        future_prior_std = torch.exp(0.5 * prior_logvar[:, -1:, :]).expand(B, HORIZON, -1)
        noise = torch.randn((B, HORIZON, latent_dim), device=DEVICE)
        z_full[:, CONTEXT_LENGTH:, :] = future_prior_mean + future_prior_std * noise

        # Decode with normal ctx
        output_normal = decode_with_modified_ctx(model, context_input, z_full, zero_ctx=False)
        outputs_normal.append(output_normal[0, HORIZON-1, 1, ATM_6M[0], ATM_6M[1]].item())

        # Decode with zeroed ctx
        output_zero = decode_with_modified_ctx(model, context_input, z_full, zero_ctx=True)
        outputs_zero_ctx.append(output_zero[0, HORIZON-1, 1, ATM_6M[0], ATM_6M[1]].item())

outputs_normal = np.array(outputs_normal)
outputs_zero_ctx = np.array(outputs_zero_ctx)

print(f"Results (100 samples, H=90, ATM 6M):")
print(f"\n  Normal (ctx + z):")
print(f"    Mean:  {np.mean(outputs_normal):.4f}")
print(f"    Std:   {np.std(outputs_normal):.6f}")
print(f"\n  Zero ctx (0 + z only):")
print(f"    Mean:  {np.mean(outputs_zero_ctx):.4f}")
print(f"    Std:   {np.std(outputs_zero_ctx):.6f}")
print(f"\n  Ratio (zero_ctx_std / normal_std): {np.std(outputs_zero_ctx) / np.std(outputs_normal):.2%}")

if np.std(outputs_zero_ctx) < 0.3 * np.std(outputs_normal):
    print("\n  ❌ FINDING: Zeroing ctx COLLAPSES variance")
    print("     → Decoder relies heavily on ctx, z alone insufficient")
elif np.std(outputs_zero_ctx) > 0.7 * np.std(outputs_normal):
    print("\n  ✅ FINDING: z alone can drive output diversity")
    print("     → Decoder uses z effectively even without ctx")
else:
    print("\n  ⚠️  FINDING: Partial reliance on both ctx and z")

print("\n" + "="*80)
print("EXPERIMENT 9: Z ZEROING TEST")
print("="*80)
print("Question: If we zero out z for FUTURE positions, does decoder still work?")
print()

# Same test samples
outputs_normal_z = []
outputs_zero_z = []

with torch.no_grad():
    for idx in sample_indices[:50]:  # 50 samples
        context_surf = surface[idx-CONTEXT_LENGTH:idx]
        context_ex = ex_feats[idx-CONTEXT_LENGTH:idx]
        target_surf = surface[idx:idx+HORIZON]

        context_input = {
            "surface": context_surf.unsqueeze(0),
            "ex_feats": context_ex.unsqueeze(0)
        }

        # Sample z from prior
        prior_mu, prior_logvar = model.prior_network(context_input)

        # Create z sequence
        B = 1
        T = CONTEXT_LENGTH + HORIZON
        latent_dim = model.config["latent_dim"]

        ctx_latent_mean, _, _ = model.encoder(context_input)

        # Normal z (ctx + sampled future)
        z_normal = torch.zeros((B, T, latent_dim), device=DEVICE)
        z_normal[:, :CONTEXT_LENGTH, :] = ctx_latent_mean
        future_prior_mean = prior_mu[:, -1:, :].expand(B, HORIZON, -1)
        future_prior_std = torch.exp(0.5 * prior_logvar[:, -1:, :]).expand(B, HORIZON, -1)
        noise = torch.randn((B, HORIZON, latent_dim), device=DEVICE)
        z_normal[:, CONTEXT_LENGTH:, :] = future_prior_mean + future_prior_std * noise

        # Zeroed z future (ctx + zeros)
        z_zeroed = z_normal.clone()
        z_zeroed[:, CONTEXT_LENGTH:, :] = 0  # Zero out future z

        # Decode both
        output_normal = decode_with_modified_ctx(model, context_input, z_normal, zero_ctx=False)
        output_zero_z = decode_with_modified_ctx(model, context_input, z_zeroed, zero_ctx=False)

        # Ground truth
        gt_value = target_surf[HORIZON-1, ATM_6M[0], ATM_6M[1]].item()

        # Predictions
        pred_normal = output_normal[0, HORIZON-1, 1, ATM_6M[0], ATM_6M[1]].item()
        pred_zero_z = output_zero_z[0, HORIZON-1, 1, ATM_6M[0], ATM_6M[1]].item()

        outputs_normal_z.append(pred_normal)
        outputs_zero_z.append(pred_zero_z)

outputs_normal_z = np.array(outputs_normal_z)
outputs_zero_z = np.array(outputs_zero_z)

print(f"Results (50 samples, H=90, ATM 6M):")
print(f"\n  Normal (ctx + z):")
print(f"    Std:   {np.std(outputs_normal_z):.6f}")
print(f"\n  Zero future z (ctx + 0):")
print(f"    Std:   {np.std(outputs_zero_z):.6f}")
print(f"\n  Ratio (zero_z_std / normal_std): {np.std(outputs_zero_z) / np.std(outputs_normal_z):.2%}")

# Check if outputs are identical when z is zeroed
correlation = np.corrcoef(outputs_normal_z, outputs_zero_z)[0, 1]
print(f"  Correlation between normal and zero_z outputs: {correlation:.4f}")

if np.std(outputs_zero_z) > 0.7 * np.std(outputs_normal_z):
    print("\n  ❌ FINDING: Zeroing z has MINIMAL impact on variance")
    print("     → Decoder relies on ctx/hidden state, not z!")
elif np.std(outputs_zero_z) < 0.3 * np.std(outputs_normal_z):
    print("\n  ✅ FINDING: Zeroing z COLLAPSES variance")
    print("     → z is necessary for output diversity")
else:
    print("\n  ⚠️  FINDING: Partial impact from zeroing z")

print("\n" + "="*80)
print("EXPERIMENT 10: GRADIENT ATTRIBUTION (FIXED)")
print("="*80)
print("Question: Which input has more influence: z or ctx_embedding?")
print()

# Set model to training mode for gradient computation
model.train()

# Sample a few test cases and measure gradients
grad_norms_z = []
grad_norms_ctx = []

for idx in sample_indices[:20]:  # 20 samples
    context_surf = surface[idx-CONTEXT_LENGTH:idx]
    context_ex = ex_feats[idx-CONTEXT_LENGTH:idx]
    target_surf = surface[idx:idx+HORIZON]

    context_input_dict = {
        "surface": context_surf.unsqueeze(0),
        "ex_feats": context_ex.unsqueeze(0)
    }

    # Get context embedding WITH gradient retention
    ctx_embedding = model.ctx_encoder(context_input_dict)  # (1, C, ctx_dim)
    ctx_embedding.retain_grad()  # CRITICAL: retain grad for non-leaf tensor

    # Get z from encoder WITH gradient retention
    full_input = {
        "surface": torch.cat([context_surf, target_surf], dim=0).unsqueeze(0),
        "ex_feats": torch.cat([context_ex, ex_feats[idx:idx+HORIZON]], dim=0).unsqueeze(0)
    }
    z_mean, _, _ = model.encoder(full_input)
    z_mean.retain_grad()  # CRITICAL: retain grad for non-leaf tensor

    # Construct decoder input
    B, C, ctx_dim = ctx_embedding.shape
    T = z_mean.shape[1]
    H = T - C

    # Create padded context WITHOUT in-place operations
    ctx_zeros = torch.zeros((B, H, ctx_dim), device=DEVICE)
    ctx_padded = torch.cat([ctx_embedding, ctx_zeros], dim=1)  # (B, T, ctx_dim)

    decoder_input = torch.cat([ctx_padded, z_mean], dim=-1)

    # Decode
    if model.config["ex_feats_dim"] > 0:
        decoded_surface, _ = model.decoder(decoder_input)
    else:
        decoded_surface = model.decoder(decoder_input)

    # Get output at H=90, ATM 6M
    output = decoded_surface[:, HORIZON-1, 1, ATM_6M[0], ATM_6M[1]]

    # Backward
    output.backward()

    # Measure gradient norms (NOW they should be populated!)
    if ctx_embedding.grad is not None:
        grad_norm_ctx = ctx_embedding.grad.norm().item()
    else:
        grad_norm_ctx = 0.0

    if z_mean.grad is not None:
        grad_norm_z = z_mean.grad.norm().item()
    else:
        grad_norm_z = 0.0

    grad_norms_ctx.append(grad_norm_ctx)
    grad_norms_z.append(grad_norm_z)

    # Zero gradients for next iteration
    model.zero_grad()

grad_norms_ctx = np.array(grad_norms_ctx)
grad_norms_z = np.array(grad_norms_z)

# Set model back to eval mode
model.eval()

print(f"Results (20 samples):")
print(f"\n  |∂output/∂ctx|: mean={np.mean(grad_norms_ctx):.4f}, std={np.std(grad_norms_ctx):.4f}")
print(f"  |∂output/∂z|:   mean={np.mean(grad_norms_z):.4f}, std={np.std(grad_norms_z):.4f}")
print(f"\n  Ratio (z/ctx): {np.mean(grad_norms_z) / np.mean(grad_norms_ctx):.2f}x")

if np.mean(grad_norms_z) < 0.3 * np.mean(grad_norms_ctx):
    print("\n  ❌ FINDING: Gradients w.r.t. z are MUCH SMALLER than ctx")
    print("     → Output is more sensitive to ctx than z")
    print("     → Decoder relies more on ctx!")
elif np.mean(grad_norms_z) > 0.7 * np.mean(grad_norms_ctx):
    print("\n  ✅ FINDING: Gradients w.r.t. z are comparable to ctx")
    print("     → Output is sensitive to both z and ctx")
else:
    print("\n  ⚠️  FINDING: Gradients w.r.t. z are moderately smaller than ctx")

print("\n" + "="*80)
print("SUMMARY OF FINDINGS")
print("="*80)
print()
print("1. Context Zeroing Test (Exp 8):")
print(f"   Variance ratio (z-only / normal): {np.std(outputs_zero_ctx) / np.std(outputs_normal):.2%}")
print()
print("2. Z Zeroing Test (Exp 9):")
print(f"   Variance ratio (zero-z / normal): {np.std(outputs_zero_z) / np.std(outputs_normal_z):.2%}")
print()
print("3. Gradient Attribution (Exp 10):")
print(f"   Gradient ratio (z/ctx): {np.mean(grad_norms_z) / np.mean(grad_norms_ctx):.2f}x")
print()

print("="*80)
print("DIAGNOSIS")
print("="*80)
print()

# Make diagnosis based on results
zero_ctx_ratio = np.std(outputs_zero_ctx) / np.std(outputs_normal)
zero_z_ratio = np.std(outputs_zero_z) / np.std(outputs_normal_z)
grad_ratio = np.mean(grad_norms_z) / np.mean(grad_norms_ctx)

if zero_ctx_ratio < 0.3 and zero_z_ratio > 0.7:
    print("🎯 PRIMARY DIAGNOSIS: Decoder Ignores z, Relies on Context")
    print()
    print("Evidence:")
    print(f"  - Zeroing ctx collapses variance to {zero_ctx_ratio:.1%} of normal")
    print(f"  - Zeroing z preserves {zero_z_ratio:.1%} of variance")
    print(f"  - Gradients w.r.t. z are {grad_ratio:.2f}x smaller than ctx")
    print()
    print("Conclusion: The decoder uses context/hidden state as primary signal,")
    print("           z is essentially ignored. This explains why the model")
    print("           cannot capture conditional distribution!")
elif zero_ctx_ratio > 0.7 and zero_z_ratio < 0.3:
    print("✅ FINDING: Decoder Uses z Effectively")
    print()
    print("Evidence:")
    print(f"  - z alone preserves {zero_ctx_ratio:.1%} of variance")
    print(f"  - Zeroing z collapses variance to {zero_z_ratio:.1%}")
    print(f"  - Gradients w.r.t. z are {grad_ratio:.2f}x ctx gradients")
    print()
    print("Conclusion: The decoder uses z effectively. The problem is NOT")
    print("           decoder ignoring z. Must be prior network issue.")
else:
    print("⚠️  MIXED FINDINGS: Both ctx and z contribute")
    print()
    print("Evidence:")
    print(f"  - z alone preserves {zero_ctx_ratio:.1%} of variance")
    print(f"  - Zeroing z reduces variance to {zero_z_ratio:.1%}")
    print(f"  - Gradient ratio z/ctx: {grad_ratio:.2f}x")
    print()
    print("Conclusion: Decoder uses both ctx and z, but the balance may be")
    print("           suboptimal. Further investigation needed.")

print()
print("="*80)
print("END OF DIAGNOSTIC EXPERIMENTS")
print("="*80)
