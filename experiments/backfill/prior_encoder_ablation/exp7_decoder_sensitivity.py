"""
Experiment 7: Decoder Sensitivity Test for exp6 Models

BACKGROUND:
Previous Exp 8-10 results (126.7% variance, 1.84x gradient ratio) were from a
DIFFERENT model (CVAEMemRandConditionalPrior) with BROKEN z sampling.

We have NO valid decoder sensitivity data for the exp6 models (CVAEFullCovPrior,
CVAEWithPriorEncoderDiagonal, CVAEWithPriorEncoderFullCov).

PURPOSE:
Test whether the exp6 models' decoders respond to z variations.

HYPOTHESIS:
If P1 ≈ 0.006% but prior σ² ≈ 1.02, either:
  A) Decoder IGNORES z → output variance ≈ 0 regardless of z variance
  B) Decoder USES z → output variance should scale with z variance

TESTS:
1. Sample z from prior with different scales (1x, 2x, 5x σ)
2. Measure output variance for each scale
3. If output variance scales with z variance → decoder is responsive
4. If output variance stays constant → decoder ignores z

This will determine the TRUE root cause of low P1.
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_full_cov_prior import CVAEFullCovPrior
from vae.cvae_prior_encoder import CVAEWithPriorEncoderDiagonal, CVAEWithPriorEncoderFullCov


def load_model(model_path, model_class):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model_config = checkpoint['model_config']

    model = model_class(model_config)

    # Handle compiled model state dict
    state_dict = checkpoint['state_dict']
    # Remove _orig_mod. prefix if present (from torch.compile)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            cleaned_state_dict[k[10:]] = v
        else:
            cleaned_state_dict[k] = v

    model.load_state_dict(cleaned_state_dict, strict=False)
    model.eval()

    return model, model_config


def test_decoder_sensitivity(model, model_name, val_surface, num_contexts=50, num_samples=100):
    """
    Test whether decoder output variance scales with z variance.

    For each context:
    1. Get prior parameters (mu, sigma)
    2. Sample z with different scales: 1x, 2x, 5x sigma
    3. Decode each z
    4. Measure output variance at each scale

    If decoder is responsive: output_var(5x) >> output_var(1x)
    If decoder ignores z: output_var(5x) ≈ output_var(1x)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    C = model.config["context_len"]
    latent_dim = model.config["latent_dim"]
    dtype = next(model.parameters()).dtype

    scales = [0.5, 1.0, 2.0, 5.0]  # Multiply sigma by these factors

    results = {scale: [] for scale in scales}
    z_variances = {scale: [] for scale in scales}

    print(f"\nTesting {model_name}...")
    print(f"  Context length: {C}, Latent dim: {latent_dim}")
    print(f"  Testing {num_contexts} contexts x {num_samples} samples x {len(scales)} scales")

    with torch.no_grad():
        for ctx_idx in range(num_contexts):
            if ctx_idx + C + 1 > len(val_surface):
                break

            # Get context
            context = val_surface[ctx_idx:ctx_idx+C].unsqueeze(0).to(device).to(dtype)
            ctx_feats = torch.zeros(1, C, 3, device=device, dtype=dtype)
            ctx_input = {"surface": context, "ex_feats": ctx_feats}

            # Get prior parameters
            if hasattr(model, 'prior_encoder'):
                # Prior encoder models
                is_diagonal = not hasattr(model.prior_encoder, 'get_phi')
                if is_diagonal:
                    mu_p, log_var_p = model.prior_encoder(ctx_input, horizon=1)
                    sigma_p = torch.exp(0.5 * log_var_p)  # (B, 1, latent_dim)
                else:
                    mu_p, Sigma_p = model.prior_encoder(ctx_input, horizon=1)
                    # For AR(1), diagonal variance is sigma_sq
                    sigma_p = torch.sqrt(torch.diag(Sigma_p)).unsqueeze(0).unsqueeze(0)
            else:
                # Baseline (CVAEFullCovPrior)
                ctx_out = model.ctx_encoder(ctx_input)
                context_summary = ctx_out[:, -1, :]
                mu_p, Sigma_p = model.full_cov_prior.get_prior_params(context_summary, horizon=1)
                sigma_p = torch.sqrt(torch.diag(Sigma_p)).unsqueeze(0).unsqueeze(0)

            # Test each scale
            for scale in scales:
                samples = []
                z_samples = []

                for _ in range(num_samples):
                    # Sample z with scaled sigma
                    epsilon = torch.randn(1, 1, latent_dim, device=device, dtype=dtype)
                    z = mu_p + scale * sigma_p * epsilon
                    z_samples.append(z.cpu())

                    # Decode
                    if model.config.get("compress_context", True):
                        ctx_embedding_dim = latent_dim
                    else:
                        ctx_embedding_dim = model.config["mem_hidden"]

                    ctx_zeros = torch.zeros(1, 1, ctx_embedding_dim, device=device, dtype=dtype)
                    decoder_input = torch.cat([z, ctx_zeros], dim=-1)
                    decoded = model.decoder(decoder_input)

                    if isinstance(decoded, tuple):
                        decoded = decoded[0]

                    samples.append(decoded.cpu())

                # Compute variances
                samples_tensor = torch.stack(samples).squeeze()  # (num_samples, 5, 5)
                z_tensor = torch.stack(z_samples).squeeze()  # (num_samples, latent_dim)

                output_var = samples_tensor.var(dim=0).mean().item()
                z_var = z_tensor.var(dim=0).mean().item()

                results[scale].append(output_var)
                z_variances[scale].append(z_var)

    return results, z_variances


def analyze_results(results, z_variances, model_name):
    """Analyze whether decoder responds to z."""
    print(f"\n{'='*60}")
    print(f"RESULTS: {model_name}")
    print(f"{'='*60}")

    print("\n| Scale | Z Variance | Output Variance | Ratio (vs 1x) |")
    print("|-------|------------|-----------------|---------------|")

    base_output_var = np.mean(results[1.0])

    for scale in sorted(results.keys()):
        z_var = np.mean(z_variances[scale])
        out_var = np.mean(results[scale])
        ratio = out_var / base_output_var if base_output_var > 0 else 0

        print(f"| {scale:.1f}x  | {z_var:.6f}   | {out_var:.6f}      | {ratio:.2f}x          |")

    # Diagnosis
    print("\n" + "-"*60)

    var_1x = np.mean(results[1.0])
    var_5x = np.mean(results[5.0])

    if var_5x > var_1x * 2:
        print("DIAGNOSIS: Decoder IS RESPONSIVE to z")
        print(f"  Output variance at 5x scale ({var_5x:.6f}) > 2x baseline ({var_1x*2:.6f})")
        print("  Root cause is NOT the decoder architecture")
    elif var_5x > var_1x * 1.2:
        print("DIAGNOSIS: Decoder is WEAKLY RESPONSIVE to z")
        print(f"  Output variance at 5x scale ({var_5x:.6f}) slightly higher than baseline ({var_1x:.6f})")
        print("  Decoder responds but with low sensitivity")
    else:
        print("DIAGNOSIS: Decoder IGNORES z")
        print(f"  Output variance at 5x scale ({var_5x:.6f}) ≈ baseline ({var_1x:.6f})")
        print("  ROOT CAUSE: Decoder does not use z variation!")

    return var_1x, var_5x


def main():
    print("="*70)
    print("EXPERIMENT 7: Decoder Sensitivity Test for exp6 Models")
    print("="*70)

    # Load validation data
    data_file = Path("data/vol_surface_with_ret.npz")
    data = np.load(data_file)
    surface = torch.tensor(data["surface"], dtype=torch.float32)

    # Use validation set (after training range)
    val_start = 4000  # After typical training end
    val_surface = surface[val_start:val_start+500]
    print(f"\nValidation data: {val_surface.shape}")

    # Models to test
    models_dir = Path("results/prior_encoder_ablation/extended_training_v5")

    models_to_test = [
        ("baseline", "baseline_ep200.pt", CVAEFullCovPrior),
        ("prior_encoder_full_cov", "prior_encoder_full_cov_ep200.pt", CVAEWithPriorEncoderFullCov),
    ]

    all_results = {}

    for model_name, checkpoint_file, model_class in models_to_test:
        model_path = models_dir / checkpoint_file

        if not model_path.exists():
            print(f"\nSkipping {model_name}: checkpoint not found at {model_path}")
            continue

        print(f"\n{'='*70}")
        print(f"Loading {model_name} from {model_path}")

        try:
            model, config = load_model(model_path, model_class)
            results, z_vars = test_decoder_sensitivity(
                model, model_name, val_surface,
                num_contexts=50, num_samples=100
            )
            var_1x, var_5x = analyze_results(results, z_vars, model_name)

            all_results[model_name] = {
                'results': results,
                'z_variances': z_vars,
                'var_1x': var_1x,
                'var_5x': var_5x,
                'responsive': var_5x > var_1x * 2
            }
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("\n| Model | Var(1x) | Var(5x) | Ratio | Responsive? |")
    print("|-------|---------|---------|-------|-------------|")

    for model_name, data in all_results.items():
        responsive = "YES" if data['responsive'] else "NO"
        ratio = data['var_5x'] / data['var_1x'] if data['var_1x'] > 0 else 0
        print(f"| {model_name[:20]:<20} | {data['var_1x']:.6f} | {data['var_5x']:.6f} | {ratio:.2f}x | {responsive} |")

    # Final conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    any_responsive = any(d['responsive'] for d in all_results.values())

    if any_responsive:
        print("\nAt least one model's decoder IS responsive to z variations.")
        print("The low P1 is NOT caused by decoder ignoring z.")
        print("Likely cause: Prior z samples are too clustered (despite σ=1.02)")
    else:
        print("\nAll models' decoders IGNORE z variations.")
        print("ROOT CAUSE IDENTIFIED: Decoder does not translate z variance to output variance.")
        print("Solution needed: Train decoder to be z-sensitive (e.g., P1 loss)")

    # Save results
    output_dir = Path("results/prior_encoder_ablation/exp7_decoder_sensitivity")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / "sensitivity_results.npz",
        **{f"{name}_{key}": np.array(val) if isinstance(val, (list, dict)) else val
           for name, data in all_results.items()
           for key, val in data.items() if not isinstance(val, dict)}
    )

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
