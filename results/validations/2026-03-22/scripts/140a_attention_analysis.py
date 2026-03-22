#!/usr/bin/env python3
"""
140a Attention Pattern & Noise Pathway Analysis

Analyzes the CausalARTransformerDecoder in Exp 140a to understand:
1. How attention is distributed (history vs generated frames)
2. Whether attention heads specialize
3. Where noise influence disappears through the layers
4. Noise skip vs transformer output magnitude comparison

Output: JSON with full analysis to results/validations/2026-03-22/analysis/140a_attention/
"""

import json
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

# Setup
REPO = "/home/max/Documents/vol-surface-vae-pub"
MODEL_PATH = os.path.join(REPO, "models/backfill/afcrps_140a/best_model.pt")
DATA_PATH = os.path.join(REPO, "data/vol_surface_with_ret.npz")
OUTPUT_DIR = os.path.join(REPO, "results/validations/2026-03-22/analysis/140a_attention")
DEVICE = "cuda"
TEST_START = 4540
N_WINDOWS = 5
N_MEMBERS = 8
N_FRAMES = 30
HISTORY_LEN = 30

# CRITICAL: Disable TransformerEncoderLayer fast path so we can intercept
# attention weights via monkey-patching self_attn.forward.
# The fast path calls torch._transformer_encoder_layer_fwd (C++ kernel)
# which completely bypasses self_attn.forward and all Python hooks.
torch.backends.mha.set_fastpath_enabled(False)

from diffusion.block_ar.single_pass_ar import (
    SinglePassBlockAR, SinglePassConfig,
    normalize_iv, denormalize_iv, CausalARTransformerDecoder
)

def load_model():
    """Load the 140a model."""
    checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location=DEVICE)
    config = SinglePassConfig(**checkpoint["config"])
    model = SinglePassBlockAR(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(DEVICE)
    print(f"Loaded 140a model from epoch {checkpoint.get('epoch', '?')}")
    print(f"  d_model={config.ar_causal_d_model}, n_heads={config.ar_causal_n_heads}, "
          f"n_layers={config.ar_causal_n_layers}")
    return model, config


def load_test_data():
    """Load test windows starting at index 4540."""
    data = np.load(DATA_PATH)
    surfaces = data["surface"]  # (N, 5, 5)
    N = surfaces.shape[0]
    print(f"Loaded data: {N} surfaces")

    windows = []
    for i in range(N_WINDOWS):
        start = TEST_START + i * 60  # Space them out
        if start + HISTORY_LEN + N_FRAMES > N:
            start = TEST_START + i  # Fallback to sequential
        history = surfaces[start:start + HISTORY_LEN]  # (30, 5, 5)
        future = surfaces[start + HISTORY_LEN:start + HISTORY_LEN + N_FRAMES]  # (30, 5, 5)
        windows.append((history, future))

    histories = np.stack([w[0] for w in windows])  # (5, 30, 5, 5)
    futures = np.stack([w[1] for w in windows])     # (5, 30, 5, 5)

    # Normalize to [-1, 1]
    hist_t = normalize_iv(torch.tensor(histories, dtype=torch.float32, device=DEVICE))
    fut_t = normalize_iv(torch.tensor(futures, dtype=torch.float32, device=DEVICE))
    return hist_t, fut_t


# ============================================================
# PART 1: Attention Pattern Analysis
# ============================================================

class AttentionCapturer:
    """Hook-based capturer for attention weights from TransformerEncoderLayer."""

    def __init__(self, model):
        self.model = model
        self.attention_weights = {}  # {layer_idx: list of (seq_len, seq_len) matrices}
        self.hooks = []

    def install_hooks(self):
        """Install forward hooks on each transformer layer's self_attn."""
        decoder = self.model.frame_decoder
        assert isinstance(decoder, CausalARTransformerDecoder)

        for layer_idx, layer in enumerate(decoder.layers):
            # TransformerEncoderLayer has self_attn attribute (nn.MultiheadAttention)
            mha = layer.self_attn
            self.attention_weights[layer_idx] = []

            def make_hook(idx):
                def hook_fn(module, args, kwargs, output):
                    # nn.MultiheadAttention returns (attn_output, attn_weights)
                    # But only when need_weights=True
                    # We need to intercept and force need_weights
                    pass
                return hook_fn

            # Instead of output hooks, we'll monkey-patch the forward to capture weights
            # MultiheadAttention.forward signature:
            #   forward(query, key, value, key_padding_mask=None, need_weights=True,
            #           attn_mask=None, average_attn_weights=True, is_causal=False)
            original_forward = mha.forward

            def make_wrapper(idx, orig_fwd):
                def wrapper(*args, **kwargs):
                    # Force need_weights=True and average_attn_weights=False
                    kwargs['need_weights'] = True
                    kwargs['average_attn_weights'] = False  # Get per-head weights
                    out, weights = orig_fwd(*args, **kwargs)
                    # weights: (B, n_heads, seq_len, seq_len)
                    self.attention_weights[idx].append(weights.detach().cpu())
                    return out, weights
                return wrapper

            mha.forward = make_wrapper(layer_idx, original_forward)

    def clear(self):
        for k in self.attention_weights:
            self.attention_weights[k] = []


def analyze_attention_patterns(model, hist_t):
    """Run forward pass and capture attention patterns."""
    print("\n=== PART 1: Attention Pattern Analysis ===")

    capturer = AttentionCapturer(model)
    capturer.install_hooks()

    B = hist_t.shape[0]  # 5 windows
    results = {
        "per_layer": {},
        "head_specialization": {},
        "history_vs_generated": {},
    }

    # Generate samples (1 ensemble member at a time to keep memory reasonable)
    with torch.no_grad():
        # We need to run the full AR loop manually to capture attention at each step
        # Use model.sample which calls the AR loop internally
        capturer.clear()

        # Run inference for a single sample to capture attention
        samples = model.sample(hist_t, n_samples=1)  # (5, 1, 30, 5, 5)

    # Process captured attention weights
    # Each layer captured 1 attention from init_context + 30 from AR steps
    print(f"  Captured attention calls per layer: "
          f"{[len(capturer.attention_weights[i]) for i in range(4)]}")

    n_layers = 4
    n_heads = 4
    n_history_tokens = HISTORY_LEN + 1  # 30 hist + 1 cond token

    for layer_idx in range(n_layers):
        attn_list = capturer.attention_weights[layer_idx]
        if len(attn_list) == 0:
            print(f"  WARNING: No attention captured for layer {layer_idx}")
            continue

        # The init_context call processes all history at once
        # Then each AR step processes the full growing sequence
        # Focus on the AR steps (skip init_context)
        # The init call is attn_list[0], then AR steps are attn_list[1:]

        layer_analysis = {
            "n_calls": len(attn_list),
            "history_attention_ratio": [],  # For each AR step, how much attention goes to history
            "recent_vs_distant": [],
            "per_head_entropy": [],
        }

        # Analyze each AR step
        ar_steps = attn_list[1:]  # Skip init_context
        for step_idx, attn_w in enumerate(ar_steps):
            # attn_w: (B, n_heads, seq_len, seq_len)
            # The last token is the newly generated frame
            # We care about the last row (what the new token attends to)
            last_row = attn_w[:, :, -1, :]  # (B, n_heads, seq_len)

            seq_len = last_row.shape[-1]

            # Split: history tokens (0 to n_history_tokens-1) vs generated (rest)
            hist_attn = last_row[:, :, :n_history_tokens].sum(dim=-1)  # (B, n_heads)
            gen_attn = last_row[:, :, n_history_tokens:].sum(dim=-1)   # (B, n_heads)
            total = hist_attn + gen_attn + 1e-10

            hist_ratio = (hist_attn / total).mean(dim=0)  # (n_heads,)
            layer_analysis["history_attention_ratio"].append(hist_ratio.tolist())

            # Entropy of attention distribution (higher = more uniform)
            attn_dist = last_row + 1e-10
            entropy = -(attn_dist * attn_dist.log()).sum(dim=-1)  # (B, n_heads)
            layer_analysis["per_head_entropy"].append(entropy.mean(dim=0).tolist())

            # Recent vs distant: how much attention to last 5 tokens vs first 5 history
            if seq_len > 10:
                recent_5 = last_row[:, :, -6:-1].sum(dim=-1)  # Last 5 tokens (excl self)
                distant_5 = last_row[:, :, :5].sum(dim=-1)     # First 5 tokens
                ratio = (recent_5 / (distant_5 + 1e-10)).mean(dim=0)
                layer_analysis["recent_vs_distant"].append(ratio.tolist())

        # Summarize across steps
        if layer_analysis["history_attention_ratio"]:
            hist_ratios = np.array(layer_analysis["history_attention_ratio"])
            results["per_layer"][f"layer_{layer_idx}"] = {
                "avg_history_attention_ratio": hist_ratios.mean(axis=0).tolist(),
                "history_ratio_by_step": {
                    "early_5": hist_ratios[:5].mean(axis=0).tolist(),
                    "mid_10": hist_ratios[10:20].mean(axis=0).tolist() if len(hist_ratios) > 20 else [],
                    "late_5": hist_ratios[-5:].mean(axis=0).tolist(),
                },
                "avg_entropy": np.array(layer_analysis["per_head_entropy"]).mean(axis=0).tolist(),
                "avg_recent_vs_distant_ratio": (
                    np.array(layer_analysis["recent_vs_distant"]).mean(axis=0).tolist()
                    if layer_analysis["recent_vs_distant"] else []
                ),
            }

    # Head specialization: compare attention patterns across heads
    for layer_idx in range(n_layers):
        attn_list = capturer.attention_weights[layer_idx]
        if len(attn_list) <= 1:
            continue
        ar_steps = attn_list[1:]

        # For each head, compute "locality" = fraction of attention in last 10 tokens
        head_locality = []
        head_history_focus = []
        for step_idx, attn_w in enumerate(ar_steps):
            last_row = attn_w[:, :, -1, :]  # (B, n_heads, seq_len)
            seq_len = last_row.shape[-1]
            local_10 = last_row[:, :, max(0, seq_len-11):seq_len-1].sum(dim=-1)
            total = last_row.sum(dim=-1) + 1e-10
            head_locality.append((local_10 / total).mean(dim=0).tolist())
            head_history_focus.append(
                (last_row[:, :, :n_history_tokens].sum(dim=-1) / total).mean(dim=0).tolist()
            )

        results["head_specialization"][f"layer_{layer_idx}"] = {
            "locality_per_head": np.array(head_locality).mean(axis=0).tolist(),
            "history_focus_per_head": np.array(head_history_focus).mean(axis=0).tolist(),
        }

    return results, capturer


# ============================================================
# PART 2: Ensemble Member Attention Comparison
# ============================================================

def analyze_ensemble_attention(model, hist_t):
    """Compare attention patterns across ensemble members (different noise draws)."""
    print("\n=== PART 2: Ensemble Member Attention Comparison ===")

    # We'll run 3 ensemble members and compare their attention
    n_ensemble = 3
    all_attentions = []

    capturer = AttentionCapturer(model)
    capturer.install_hooks()

    for member_idx in range(n_ensemble):
        capturer.clear()
        with torch.no_grad():
            # Each call draws different noise
            torch.manual_seed(42 + member_idx)
            samples = model.sample(hist_t[:2], n_samples=1)  # Use 2 windows for speed

        # Save attention from layer 0 and layer 3 for comparison
        member_attn = {}
        for layer_idx in [0, 3]:
            ar_steps = capturer.attention_weights[layer_idx][1:]  # Skip init
            # Collect last-row attention for steps 0, 14, 29
            key_steps = [0, 14, min(29, len(ar_steps)-1)]
            member_attn[f"layer_{layer_idx}"] = {}
            for s in key_steps:
                if s < len(ar_steps):
                    last_row = ar_steps[s][:, :, -1, :]  # (B, n_heads, seq_len)
                    member_attn[f"layer_{layer_idx}"][f"step_{s}"] = last_row.numpy()

        all_attentions.append(member_attn)

    # Compare: cosine similarity of attention patterns between ensemble members
    results = {}
    for layer_key in ["layer_0", "layer_3"]:
        results[layer_key] = {}
        for step_key in ["step_0", "step_14", "step_29"]:
            if step_key not in all_attentions[0][layer_key]:
                continue

            cosine_sims = []
            l2_diffs = []
            for i in range(n_ensemble):
                for j in range(i+1, n_ensemble):
                    a = all_attentions[i][layer_key][step_key]  # (B, n_heads, seq_len)
                    b = all_attentions[j][layer_key][step_key]

                    # Cosine similarity per head (averaged over batch)
                    for head in range(a.shape[1]):
                        a_h = a[:, head, :]
                        b_h = b[:, head, :]
                        dot = (a_h * b_h).sum(axis=-1)
                        norm_a = np.sqrt((a_h**2).sum(axis=-1))
                        norm_b = np.sqrt((b_h**2).sum(axis=-1))
                        cos_sim = (dot / (norm_a * norm_b + 1e-10)).mean()
                        cosine_sims.append(float(cos_sim))

                    # L2 difference
                    l2 = np.sqrt(((a - b)**2).sum(axis=-1).mean())
                    l2_diffs.append(float(l2))

            results[layer_key][step_key] = {
                "mean_cosine_similarity": float(np.mean(cosine_sims)),
                "std_cosine_similarity": float(np.std(cosine_sims)),
                "mean_l2_diff": float(np.mean(l2_diffs)),
                "interpretation": (
                    "HIGH similarity = noise doesn't affect attention"
                    if np.mean(cosine_sims) > 0.95
                    else "LOW similarity = noise affects attention patterns"
                ),
            }

    return results


# ============================================================
# PART 3: Noise Pathway Analysis
# ============================================================

def analyze_noise_pathway(model, hist_t):
    """Measure where noise influence disappears through transformer layers."""
    print("\n=== PART 3: Noise Pathway Analysis ===")

    decoder = model.frame_decoder
    assert isinstance(decoder, CausalARTransformerDecoder)

    B = 2  # Use 2 windows
    hist = hist_t[:B]

    results = {
        "per_layer_noise_sensitivity": {},
        "noise_skip_vs_transformer": {},
        "hidden_state_divergence": {},
    }

    # We'll run the same input with 2 different noise draws and compare hidden states
    # Need to manually step through the AR loop

    config = model.config
    H, W = config.surface_h, config.surface_w

    # Prepare common inputs
    with torch.no_grad():
        condition = model.encoder(hist, mask=None)
        _, vol_scale = model._compute_vol_scale(hist)
        prev_frame = denormalize_iv(hist[:, -1])  # (B, 5, 5)

    # Two different noise draws
    torch.manual_seed(100)
    z1 = torch.randn(B, config.noise_dim, device=DEVICE)
    torch.manual_seed(200)
    z2 = torch.randn(B, config.noise_dim, device=DEVICE)

    rho = config.ar_frame_rho

    # Capture intermediate hidden states for each noise draw
    layer_outputs_by_noise = {0: [], 1: []}  # noise_idx -> list of per-layer states per AR step

    for noise_idx, z_init in enumerate([z1, z2]):
        z_t = z_init.clone()
        prev_frame_local = prev_frame.clone()

        with torch.no_grad():
            # Initialize context
            hist_flat = denormalize_iv(hist).reshape(B, HISTORY_LEN, H * W)
            decoder.init_context(condition, hist_flat)

        layer_states_per_step = []

        for step_idx in range(N_FRAMES):
            if step_idx > 0:
                eps_t = torch.randn_like(z_t)
                z_t = rho * z_t + np.sqrt(1 - rho**2) * eps_t

            prev_flat = prev_frame_local.reshape(B, H * W)
            noise_input = model._get_noise_for_decoder(z_t)

            # We need to manually step through the transformer to capture per-layer states
            with torch.no_grad():
                # Replicate decoder.forward logic but capture per-layer
                frame_token = decoder.frame_proj(prev_flat)
                noise_emb = decoder.noise_proj(noise_input)
                pos_idx = min(decoder._next_pos, 61)
                pos_emb = decoder.token_pos_embed(
                    torch.tensor([pos_idx], device=DEVICE)
                )
                new_token = (frame_token + noise_emb + pos_emb).unsqueeze(1)

                full_seq = torch.cat([decoder._context, new_token], dim=1)
                seq_len = full_seq.shape[1]
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=DEVICE), diagonal=1
                ).bool()

                # Step through layers, capturing output after each
                states_this_step = {"input": full_seq[:, -1, :].clone().cpu()}
                for li, layer in enumerate(decoder.layers):
                    full_seq = layer(full_seq, src_mask=causal_mask)
                    states_this_step[f"layer_{li}"] = full_seq[:, -1, :].clone().cpu()

                # Output projection
                last_hidden = full_seq[:, -1, :]
                delta = decoder.out_proj(decoder.out_norm(last_hidden))
                states_this_step["delta"] = delta.clone().cpu()

                # Noise skip
                skip_out = torch.tanh(decoder.noise_skip_proj(noise_input))
                states_this_step["noise_skip"] = skip_out.clone().cpu()

                # Update context
                decoder._context = full_seq
                decoder._next_pos += 1

                layer_states_per_step.append(states_this_step)

                # Update prev_frame (simplified - use reflect logic)
                local_pos, horizon_bucket = model._get_ar_frame_positions(
                    step_idx=step_idx, batch_size=B, device=DEVICE, position_mode="native"
                )
                cs = model._get_cell_spread(condition, local_pos)
                delta_shaped = delta.reshape(B, H, W)
                if cs is not None:
                    delta_shaped = cs * delta_shaped
                if config.ar_skip_bypass_spread:
                    skip_shaped = skip_out.reshape(B, H, W)
                    delta_shaped = delta_shaped + skip_shaped

                vs = model._get_ar_frame_vol_scale(condition, vol_scale, None)
                raw = prev_frame_local + vs * delta_shaped
                floor = config.ar_frame_floor_clamp
                width = 1.0 - floor
                shifted = raw - floor
                shifted = shifted % (2 * width)
                iv_t = torch.where(shifted > width, 2 * width - shifted, shifted) + floor
                prev_frame_local = iv_t

        layer_outputs_by_noise[noise_idx] = layer_states_per_step

    # Compare the two noise draws at each layer, each step
    print("  Computing noise sensitivity per layer per step...")

    per_layer_sensitivity = defaultdict(list)  # layer -> list of L2 diffs per step
    noise_skip_magnitudes = []
    transformer_delta_magnitudes = []

    for step_idx in range(N_FRAMES):
        s0 = layer_outputs_by_noise[0][step_idx]
        s1 = layer_outputs_by_noise[1][step_idx]

        for key in ["input", "layer_0", "layer_1", "layer_2", "layer_3", "delta"]:
            diff = (s0[key] - s1[key]).norm(dim=-1).mean().item()
            base_norm = (s0[key].norm(dim=-1).mean().item() + s1[key].norm(dim=-1).mean().item()) / 2
            per_layer_sensitivity[key].append({
                "l2_diff": diff,
                "base_norm": base_norm,
                "relative_diff": diff / (base_norm + 1e-10),
            })

        # Skip vs transformer magnitudes
        skip_mag = s0["noise_skip"].norm(dim=-1).mean().item()
        delta_mag = s0["delta"].norm(dim=-1).mean().item()
        noise_skip_magnitudes.append(skip_mag)
        transformer_delta_magnitudes.append(delta_mag)

    # Summarize
    for key in ["input", "layer_0", "layer_1", "layer_2", "layer_3", "delta"]:
        diffs = [x["l2_diff"] for x in per_layer_sensitivity[key]]
        rel_diffs = [x["relative_diff"] for x in per_layer_sensitivity[key]]
        results["per_layer_noise_sensitivity"][key] = {
            "mean_l2_diff": float(np.mean(diffs)),
            "std_l2_diff": float(np.std(diffs)),
            "early_mean": float(np.mean(diffs[:5])),
            "late_mean": float(np.mean(diffs[-5:])),
            "mean_relative_diff": float(np.mean(rel_diffs)),
            "all_steps": [float(d) for d in diffs],
        }

    results["noise_skip_vs_transformer"] = {
        "mean_skip_magnitude": float(np.mean(noise_skip_magnitudes)),
        "mean_delta_magnitude": float(np.mean(transformer_delta_magnitudes)),
        "skip_to_delta_ratio": float(np.mean(noise_skip_magnitudes) / (np.mean(transformer_delta_magnitudes) + 1e-10)),
        "skip_magnitudes_per_step": [float(x) for x in noise_skip_magnitudes],
        "delta_magnitudes_per_step": [float(x) for x in transformer_delta_magnitudes],
    }

    # Where does noise influence disappear?
    input_diff = results["per_layer_noise_sensitivity"]["input"]["mean_l2_diff"]
    for key in ["layer_0", "layer_1", "layer_2", "layer_3", "delta"]:
        layer_diff = results["per_layer_noise_sensitivity"][key]["mean_l2_diff"]
        retention = layer_diff / (input_diff + 1e-10)
        results["per_layer_noise_sensitivity"][key]["noise_retention_vs_input"] = float(retention)

    return results


# ============================================================
# PART 4: Noise Skip Weight Analysis
# ============================================================

def analyze_noise_skip_weights(model):
    """Analyze the learned weights of noise_skip_proj and out_proj."""
    print("\n=== PART 4: Weight Analysis ===")

    decoder = model.frame_decoder

    results = {}

    # noise_skip_proj: Linear(32 -> 25)
    skip_w = decoder.noise_skip_proj.weight.detach().cpu()  # (25, 32)
    results["noise_skip_proj"] = {
        "weight_shape": list(skip_w.shape),
        "frobenius_norm": float(skip_w.norm().item()),
        "mean_abs": float(skip_w.abs().mean().item()),
        "max_abs": float(skip_w.abs().max().item()),
        "singular_values": torch.linalg.svdvals(skip_w).tolist(),
        "effective_rank": float(
            (torch.linalg.svdvals(skip_w) / torch.linalg.svdvals(skip_w).sum()).pow(2).sum().reciprocal().item()
        ) if skip_w.norm() > 1e-8 else 0.0,
    }

    # out_proj: Linear(128 -> 25, zero-initialized)
    out_w = decoder.out_proj.weight.detach().cpu()  # (25, 128)
    out_b = decoder.out_proj.bias.detach().cpu()  # (25,)
    results["out_proj"] = {
        "weight_shape": list(out_w.shape),
        "frobenius_norm": float(out_w.norm().item()),
        "mean_abs": float(out_w.abs().mean().item()),
        "max_abs": float(out_w.abs().max().item()),
        "bias_norm": float(out_b.norm().item()),
        "bias_mean": float(out_b.mean().item()),
        "singular_values": torch.linalg.svdvals(out_w).tolist(),
        "effective_rank": float(
            (torch.linalg.svdvals(out_w) / torch.linalg.svdvals(out_w).sum()).pow(2).sum().reciprocal().item()
        ),
    }

    # noise_proj: Linear(32 -> 64) — projects noise into token space
    noise_proj_w = decoder.noise_proj.weight.detach().cpu()  # (64, 32)
    results["noise_proj"] = {
        "weight_shape": list(noise_proj_w.shape),
        "frobenius_norm": float(noise_proj_w.norm().item()),
        "mean_abs": float(noise_proj_w.abs().mean().item()),
        "max_abs": float(noise_proj_w.abs().max().item()),
        "singular_values": torch.linalg.svdvals(noise_proj_w).tolist(),
    }

    # frame_proj: Linear(25 -> 64) — projects IV frame into token space
    frame_proj_w = decoder.frame_proj.weight.detach().cpu()  # (64, 25)
    results["frame_proj"] = {
        "weight_shape": list(frame_proj_w.shape),
        "frobenius_norm": float(frame_proj_w.norm().item()),
        "mean_abs": float(frame_proj_w.abs().mean().item()),
        "max_abs": float(frame_proj_w.abs().max().item()),
    }

    # Compare magnitudes
    frame_mag = results["frame_proj"]["frobenius_norm"]
    noise_mag = results["noise_proj"]["frobenius_norm"]
    results["noise_vs_frame_ratio"] = float(noise_mag / (frame_mag + 1e-10))

    return results


# ============================================================
# PART 5: Cross-cell correlation mechanism
# ============================================================

def analyze_correlation_mechanism(model, hist_t):
    """Understand HOW the transformer achieves correct cross-cell correlation."""
    print("\n=== PART 5: Cross-Cell Correlation Mechanism ===")

    B = 2
    hist = hist_t[:B]

    results = {}

    with torch.no_grad():
        # Generate many ensemble members and compute cross-cell correlation
        torch.manual_seed(42)
        samples = model.sample(hist, n_samples=50)  # (2, 50, 30, 5, 5)

        # Compute daily changes
        changes = samples[:, :, 1:] - samples[:, :, :-1]  # (2, 50, 29, 5, 5)
        changes_flat = changes.reshape(2, 50, 29, 25)  # (2, 50, 29, 25)

        # Cross-cell correlation for each window
        for win_idx in range(B):
            # Flatten: (50*29, 25)
            ch = changes_flat[win_idx].reshape(-1, 25).cpu().numpy()
            corr_matrix = np.corrcoef(ch.T)  # (25, 25)

            # Key stats
            upper_tri = corr_matrix[np.triu_indices(25, k=1)]
            results[f"window_{win_idx}"] = {
                "mean_cross_cell_corr": float(np.mean(upper_tri)),
                "median_cross_cell_corr": float(np.median(upper_tri)),
                "std_cross_cell_corr": float(np.std(upper_tri)),
                "min_corr": float(np.min(upper_tri)),
                "max_corr": float(np.max(upper_tri)),
            }

            # Effective rank of correlation matrix
            eigenvalues = np.linalg.eigvalsh(corr_matrix)
            eigenvalues = np.maximum(eigenvalues, 0)
            eigenvalues = eigenvalues / eigenvalues.sum()
            eff_rank = float(np.exp(-np.sum(eigenvalues * np.log(eigenvalues + 1e-10))))
            results[f"window_{win_idx}"]["effective_rank"] = eff_rank

        # Also look at how the transformer's out_proj maps hidden to cells
        # If out_proj has rank > 1, it can create correlated but non-identical cell outputs
        out_w = model.frame_decoder.out_proj.weight.detach().cpu()  # (25, d_model)
        svd_vals = torch.linalg.svdvals(out_w)
        results["out_proj_structure"] = {
            "singular_values": svd_vals.tolist(),
            "rank_ratio_top2": float(svd_vals[1] / (svd_vals[0] + 1e-10)),
            "effective_rank": float(
                (svd_vals / svd_vals.sum()).pow(2).sum().reciprocal().item()
            ),
            "interpretation": (
                "High rank = multiple independent cell output directions. "
                "This is how transformer creates cross-cell correlation != 1"
            ),
        }

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("140a ATTENTION PATTERN & NOISE PATHWAY ANALYSIS")
    print("=" * 70)

    model, config = load_model()
    hist_t, fut_t = load_test_data()

    all_results = {
        "model": "140a",
        "model_path": MODEL_PATH,
        "config_summary": {
            "d_model": config.ar_causal_d_model,
            "n_heads": config.ar_causal_n_heads,
            "n_layers": config.ar_causal_n_layers,
            "noise_dim": config.noise_dim,
            "ar_noise_skip": config.ar_noise_skip,
            "ar_skip_bypass_spread": config.ar_skip_bypass_spread,
            "rho": config.ar_frame_rho,
        },
    }

    # Part 1: Attention patterns
    attn_results, capturer = analyze_attention_patterns(model, hist_t)
    all_results["attention_patterns"] = attn_results

    # Part 2: Ensemble attention comparison
    ensemble_results = analyze_ensemble_attention(model, hist_t)
    all_results["ensemble_attention_comparison"] = ensemble_results

    # Part 3: Noise pathway
    noise_results = analyze_noise_pathway(model, hist_t)
    all_results["noise_pathway"] = noise_results

    # Part 4: Weight analysis
    weight_results = analyze_noise_skip_weights(model)
    all_results["weight_analysis"] = weight_results

    # Part 5: Correlation mechanism
    corr_results = analyze_correlation_mechanism(model, hist_t)
    all_results["correlation_mechanism"] = corr_results

    # ============================================================
    # Summary & Key Findings
    # ============================================================
    print("\n" + "=" * 70)
    print("KEY FINDINGS SUMMARY")
    print("=" * 70)

    summary = {}

    # 1. Where does noise disappear?
    noise_sens = noise_results["per_layer_noise_sensitivity"]
    input_diff = noise_sens["input"]["mean_l2_diff"]
    for key in ["layer_0", "layer_1", "layer_2", "layer_3", "delta"]:
        retention = noise_sens[key].get("noise_retention_vs_input", 0)
        print(f"  Noise retention at {key}: {retention:.3f}")

    summary["noise_washout_layer"] = "See per_layer_noise_sensitivity for details"

    # 2. Skip vs transformer
    skip_ratio = noise_results["noise_skip_vs_transformer"]["skip_to_delta_ratio"]
    print(f"\n  Noise skip / transformer delta ratio: {skip_ratio:.3f}")
    summary["skip_to_delta_ratio"] = skip_ratio

    # 3. Attention: history vs generated
    for layer_key, layer_data in attn_results.get("per_layer", {}).items():
        hist_ratio = layer_data.get("avg_history_attention_ratio", [])
        print(f"  {layer_key} avg history attention: {[f'{r:.3f}' for r in hist_ratio]}")

    # 4. Head specialization
    for layer_key, spec_data in attn_results.get("head_specialization", {}).items():
        locality = spec_data.get("locality_per_head", [])
        print(f"  {layer_key} head locality: {[f'{l:.3f}' for l in locality]}")

    # 5. Ensemble similarity
    for layer_key, layer_data in ensemble_results.items():
        for step_key, step_data in layer_data.items():
            cos_sim = step_data.get("mean_cosine_similarity", 0)
            print(f"  Ensemble attn similarity {layer_key}/{step_key}: {cos_sim:.4f}")

    # 6. Correlation
    for win_key, win_data in corr_results.items():
        if win_key.startswith("window_"):
            print(f"  {win_key} cross-cell corr: {win_data.get('mean_cross_cell_corr', 0):.3f}, "
                  f"eff_rank: {win_data.get('effective_rank', 0):.2f}")

    all_results["summary"] = summary

    # Save
    output_path = os.path.join(OUTPUT_DIR, "analysis_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved analysis to {output_path}")

    return all_results


if __name__ == "__main__":
    results = main()
