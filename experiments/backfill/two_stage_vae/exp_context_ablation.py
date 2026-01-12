"""
Experiment: Context Ablation for Decoder Gain

Goal: Force decoder to rely on z by weakening/removing context signal.

Hypothesis: If decoder sees less context, it MUST use z for prediction,
which should increase decoder gain and improve CI coverage.

Experiments:
- baseline: ctx_embedding_dim=3, ctx_dropout=0.0 (current config)
- ctx_dim_2: ctx_embedding_dim=2
- ctx_dim_1: ctx_embedding_dim=1
- ctx_dim_0: ctx_embedding_dim=0 (z-only decoder)
- ctx_dropout_05: ctx_embedding_dim=3, ctx_dropout=0.5
- ctx_dropout_08: ctx_embedding_dim=3, ctx_dropout=0.8

Key metrics:
- Decoder gain: output_variance / z_variance (want higher)
- CI violations per horizon (want lower)
- Reconstruction MSE (acceptable degradation OK)

Usage:
    python experiments/backfill/two_stage_vae/exp_context_ablation.py
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import json
from datetime import datetime

sys.path.insert(0, ".")

from vae.cvae_two_stage import CVAETwoStage, TwoStageDecoder, TwoStageCtxEncoder
from config.two_stage_config import TwoStageConfig


class TwoStageDecoderWithCtxDropout(TwoStageDecoder):
    """Decoder with dropout on context embedding."""

    def __init__(self, config: dict):
        super().__init__(config)
        ctx_dropout = config.get("ctx_dropout", 0.0)
        self.ctx_dropout = nn.Dropout(p=ctx_dropout) if ctx_dropout > 0 else None

    def forward(self, ctx_emb, z):
        """Decode with optional dropout on ctx_emb."""
        if self.ctx_dropout is not None and self.training:
            ctx_emb = self.ctx_dropout(ctx_emb)
        return super().forward(ctx_emb, z)


class TwoStageDecoderZOnly(TwoStageDecoder):
    """Decoder that only sees z, not ctx_emb."""

    def __init__(self, config: dict):
        # Create a modified config with ctx_embedding_dim=0
        config_modified = config.copy()
        config_modified["ctx_embedding_dim"] = 0

        # Initialize parent but we need to override some things
        nn.Module.__init__(self)
        self.config = config

        latent_dim = config["latent_dim"]
        surface_hidden = config["surface_hidden"]
        feat_dim = config["feat_dim"]

        # Decoder-specific LSTM parameters
        mem_type = config.get("mem_type", "lstm")
        decoder_mem_hidden = config.get("decoder_mem_hidden", 8)
        decoder_mem_layers = config.get("decoder_mem_layers", 1)
        decoder_mem_dropout = config.get("decoder_mem_dropout", 0.1)

        # Compute n_surface for decoder output
        if config.get("use_dense_surface", True):
            n_surface = surface_hidden[-1]
        else:
            n_surface = surface_hidden[-1] * feat_dim[0] * feat_dim[1]

        # Handle extra features
        ex_feats_dim = config.get("ex_feats_dim", 0)
        ex_feats_hidden = config.get("ex_feats_hidden")
        if ex_feats_hidden is not None:
            n_info = ex_feats_hidden[-1]
        else:
            n_info = ex_feats_dim

        self.n_surface = n_surface
        self.n_info = n_info
        self.surface_final_hidden = surface_hidden[-1]
        self.decoder_mem_hidden = decoder_mem_hidden

        # Input: z only (no ctx_emb!)
        input_dim = latent_dim

        # Build LSTM memory
        self._build_memory(mem_type, input_dim, decoder_mem_hidden,
                          decoder_mem_layers, decoder_mem_dropout)

        # FiLM modulation layers
        self.gamma_net = nn.Linear(latent_dim, decoder_mem_hidden)
        self.beta_net = nn.Linear(latent_dim, decoder_mem_hidden)

        # Optional compression layer
        decoder_compress = config.get("decoder_compress", False)
        decoder_compress_dim = config.get("decoder_compress_dim", 4)
        self.use_compress = decoder_compress

        if decoder_compress:
            self.compress = nn.Linear(decoder_mem_hidden, decoder_compress_dim)
            project_dim = decoder_compress_dim
        else:
            project_dim = decoder_mem_hidden

        # Surface decoder
        self.surface_input = nn.Linear(project_dim, n_surface)
        self._build_surface_decoder(config, surface_hidden, feat_dim)

        # Extra features decoder
        if n_info > 0:
            self.ex_feats_input = nn.Linear(project_dim, n_info)
            self._build_ex_feats_decoder(config, ex_feats_dim)

    def forward(self, ctx_emb, z):
        """Decode from z only, ignoring ctx_emb."""
        feat_dim = self.config["feat_dim"]
        ex_feats_dim = self.config.get("ex_feats_dim", 0)
        use_dense = self.config.get("use_dense_surface", True)

        # LSTM processes z only (ignore ctx_emb)
        mem_out, _ = self.mem(z)  # (B, T, decoder_mem_hidden)
        B, T = mem_out.shape[:2]

        # FiLM: z modulates LSTM output
        gamma = self.gamma_net(z)
        beta = self.beta_net(z)
        features = gamma * mem_out + beta

        # Optional compression
        if self.use_compress:
            features = self.compress(features)

        # Decode surface
        surface_in = self.surface_input(features)

        if use_dense:
            surface_flat = surface_in.reshape(B * T, self.surface_final_hidden)
            decoded = self.surface_decoder(surface_flat)
            decoded_surface = decoded.reshape(B, T, feat_dim[0], feat_dim[1])
        else:
            surface_flat = surface_in.reshape(
                B * T, self.surface_final_hidden, feat_dim[0], feat_dim[1]
            )
            decoded = self.surface_decoder(surface_flat)
            decoded_surface = decoded.reshape(B, T, feat_dim[0], feat_dim[1])

        # Decode extra features
        if ex_feats_dim > 0:
            ex_in = self.ex_feats_input(features)
            ex_flat = ex_in.reshape(B * T, self.n_info)
            decoded_ex = self.ex_feats_decoder(ex_flat)
            decoded_ex = decoded_ex.reshape(B, T, ex_feats_dim)
            return decoded_surface, decoded_ex

        return decoded_surface


class CVAETwoStageContextAblation(CVAETwoStage):
    """CVAETwoStage with configurable context ablation."""

    def __init__(self, config: dict):
        # Check for special decoder modes
        z_only_decoder = config.get("z_only_decoder", False)
        ctx_dropout = config.get("ctx_dropout", 0.0)

        # Initialize parent
        super().__init__(config)

        # Replace decoder if needed
        if z_only_decoder:
            self.decoder = TwoStageDecoderZOnly(config).to(self.device)
        elif ctx_dropout > 0:
            self.decoder = TwoStageDecoderWithCtxDropout(config).to(self.device)


def to_log_returns(surfaces):
    """Transform IV surfaces to log-returns."""
    log_surfaces = np.log(surfaces)
    log_returns = log_surfaces[1:] - log_surfaces[:-1]
    return log_returns, log_surfaces


def create_sequences(data, seq_len):
    """Create overlapping sequences from data."""
    n_sequences = len(data) - seq_len + 1
    sequences = torch.stack([data[i:i+seq_len] for i in range(n_sequences)])
    return sequences


def train_epoch(model, train_sequences, optimizer, batch_size, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    n_batches = 0

    indices = torch.randperm(len(train_sequences))

    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i:i+batch_size]
        batch = train_sequences[batch_idx].to(device)

        losses = model.train_step_autoencoder({"surface": batch}, optimizer)

        total_loss += losses["loss"].item()
        total_recon += losses["re_surface"].item()
        total_kl += losses["kl_loss"].item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "recon": total_recon / n_batches,
        "kl": total_kl / n_batches,
    }


def validate(model, val_sequences, batch_size, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    n_batches = 0

    with torch.no_grad():
        for i in range(0, len(val_sequences), batch_size):
            batch = val_sequences[i:i+batch_size].to(device)
            losses = model.test_step({"surface": batch})

            total_loss += losses["loss"].item()
            total_recon += losses["re_surface"].item()
            total_kl += losses["kl_loss"].item()
            n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "recon": total_recon / n_batches,
        "kl": total_kl / n_batches,
    }


def measure_decoder_gain(model, log_returns, device, config, num_contexts=50, num_samples=50):
    """
    Measure decoder gain: output_variance / z_variance.

    Higher gain = decoder uses z effectively (good!)
    Lower gain = decoder ignores z (bad!)
    """
    model.eval()
    context_len = config["context_len"]
    latent_dim = config["latent_dim"]
    seq_len = context_len + 1

    z_vars = []
    out_vars = []

    with torch.no_grad():
        for i in range(num_contexts):
            if i + seq_len > len(log_returns):
                break

            seq = torch.tensor(log_returns[i:i+seq_len], dtype=torch.float32).unsqueeze(0).to(device)
            batch = {"surface": seq}

            # Get ctx_emb from context encoder
            ctx_emb = model.ctx_encoder(batch)

            # Get z_mean and z_logvar from encoder
            z_mean, z_logvar, _ = model.encoder(batch)

            # Sample multiple z values and decode
            samples = []
            z_samples = []

            for _ in range(num_samples):
                eps = torch.randn_like(z_logvar)
                z = z_mean + torch.exp(0.5 * z_logvar) * eps
                z_samples.append(z.cpu())

                decoded = model.decoder(ctx_emb, z)
                samples.append(decoded.cpu())

            # Compute variances
            samples_tensor = torch.stack(samples).squeeze()
            z_tensor = torch.stack(z_samples).squeeze()

            # Take last position (horizon=1)
            out_var = samples_tensor[:, -1, :, :].var(dim=0).mean().item()
            z_var = z_tensor[:, -1, :].var(dim=0).mean().item()

            out_vars.append(out_var)
            z_vars.append(z_var)

    mean_out_var = np.mean(out_vars)
    mean_z_var = np.mean(z_vars)
    decoder_gain = mean_out_var / (mean_z_var + 1e-10)

    return decoder_gain, mean_z_var, mean_out_var


def measure_ci_coverage(model, log_returns, log_surfaces, device, config,
                        horizons=[1, 7, 14, 30], n_samples=100, n_test=50):
    """Measure CI coverage in IV space."""
    model.eval()
    context_len = config["context_len"]
    seq_len = context_len + max(horizons)
    surfaces = np.exp(log_surfaces)

    results = {}

    for H in horizons:
        coverage_count = 0
        ci_widths = []
        n_tests = 0

        for start_idx in range(0, len(log_returns) - seq_len, max(1, (len(log_returns) - seq_len) // n_test)):
            if n_tests >= n_test:
                break

            gt_log_seq = log_returns[start_idx:start_idx + context_len + H]
            initial_log = log_surfaces[start_idx]
            gt_iv = surfaces[start_idx + context_len + H]

            samples_iv = []
            with torch.no_grad():
                batch = torch.tensor(gt_log_seq[None], dtype=torch.float32).to(device)

                for _ in range(n_samples):
                    output = model({"surface": batch}, return_full_sequence=True)
                    recon_log_returns = output[0].cpu().numpy()[0]

                    cumsum = recon_log_returns[:context_len + H].sum(axis=0)
                    final_iv = np.exp(initial_log + cumsum)
                    samples_iv.append(final_iv)

            samples_iv = np.array(samples_iv)

            p05 = np.percentile(samples_iv, 5, axis=0)
            p95 = np.percentile(samples_iv, 95, axis=0)

            in_ci = (gt_iv >= p05) & (gt_iv <= p95)
            coverage_count += in_ci.mean()

            ci_widths.append((p95 - p05).mean())
            n_tests += 1

        coverage = coverage_count / n_tests
        avg_ci_width = np.mean(ci_widths)

        results[H] = {
            "coverage": coverage,
            "ci_width": avg_ci_width,
            "violations": 1.0 - coverage,
        }

    return results


def train_and_evaluate(exp_name, config_overrides, log_returns, log_surfaces,
                       train_sequences, val_sequences, output_dir, n_epochs=100):
    """Train a model variant and evaluate it."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'='*70}")

    # Get base config and apply overrides
    config = TwoStageConfig.get_model_config()
    for key, value in config_overrides.items():
        config[key] = value
        print(f"  {key}: {value}")

    device = config["device"]
    batch_size = 256
    learning_rate = 1e-4

    # Build model
    print(f"\nBuilding model...")
    model = CVAETwoStageContextAblation(config)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  ctx_embedding_dim: {config.get('ctx_embedding_dim', 3)}")
    print(f"  z_only_decoder: {config.get('z_only_decoder', False)}")
    print(f"  ctx_dropout: {config.get('ctx_dropout', 0.0)}")
    print(f"  latent_dim: {config['latent_dim']}")
    print(f"  z_dropout: {config.get('z_dropout', 0.0)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    print(f"\nTraining for {n_epochs} epochs...")
    best_val_loss = float('inf')
    best_state = model.state_dict().copy()  # Initialize with starting weights
    history = {"train": [], "val": []}

    for epoch in tqdm(range(n_epochs), desc=exp_name):
        train_metrics = train_epoch(model, train_sequences, optimizer, batch_size, device)
        val_metrics = validate(model, val_sequences, batch_size, device)

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        if val_metrics["loss"] < best_val_loss and not np.isnan(val_metrics["loss"]):
            best_val_loss = val_metrics["loss"]
            best_state = model.state_dict().copy()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: train_loss={train_metrics['loss']:.4f}, val_loss={val_metrics['loss']:.4f}")

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    # Evaluate
    print(f"\nEvaluating...")

    # Decoder gain
    decoder_gain, z_var, out_var = measure_decoder_gain(model, log_returns, device, config)
    print(f"  Decoder Gain: {decoder_gain:.2e}")
    print(f"  Z Variance: {z_var:.6f}")
    print(f"  Output Variance: {out_var:.6f}")

    # CI coverage
    ci_results = measure_ci_coverage(model, log_returns, log_surfaces, device, config)
    print(f"\n  CI Violations (IV Space):")
    for H in [1, 7, 14, 30]:
        if H in ci_results:
            r = ci_results[H]
            print(f"    H={H:2d}: {r['violations']*100:.1f}% violations, CI width={r['ci_width']:.4f}")

    # Save results
    results = {
        "exp_name": exp_name,
        "config": config_overrides,
        "best_val_loss": best_val_loss,
        "decoder_gain": decoder_gain,
        "z_variance": z_var,
        "output_variance": out_var,
        "ci_results": {str(k): v for k, v in ci_results.items()},
        "final_train_loss": history["train"][-1]["loss"],
        "final_val_loss": history["val"][-1]["loss"],
    }

    # Save checkpoint
    checkpoint_path = output_dir / f"{exp_name}_best.pt"
    torch.save({
        "model_config": config,
        "model_state_dict": best_state,
        "results": results,
        "history": history,
    }, checkpoint_path)
    print(f"\n  Saved: {checkpoint_path}")

    return results


def main():
    print("="*70)
    print("CONTEXT ABLATION EXPERIMENT")
    print("="*70)
    print()
    print("Goal: Force decoder to rely on z by weakening/removing context")
    print("Hypothesis: Less context → higher decoder gain → better CI coverage")
    print()

    # Output directory
    output_dir = Path("models/backfill/two_stage/context_ablation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, log_surfaces = to_log_returns(surfaces)
    print(f"  Surfaces: {surfaces.shape}")
    print(f"  Log-returns: {log_returns.shape}")

    # Create sequences
    config = TwoStageConfig.get_model_config()
    context_len = config["context_len"]
    horizon = config["horizon"]
    seq_len = context_len + horizon

    log_returns_tensor = torch.tensor(log_returns, dtype=torch.float32)
    all_sequences = create_sequences(log_returns_tensor, seq_len)

    # Train/val split
    n_train = int(len(all_sequences) * 0.8)
    train_sequences = all_sequences[:n_train]
    val_sequences = all_sequences[n_train:]
    print(f"  Train sequences: {len(train_sequences)}")
    print(f"  Val sequences: {len(val_sequences)}")

    # Define experiments
    # Using best config from z_dropout ablation: latent_dim=8, z_dropout=0.3
    base_config = {"latent_dim": 8, "z_dropout": 0.3}

    experiments = [
        # Baseline with best z config
        ("baseline", {**base_config, "ctx_embedding_dim": 3}),

        # Reduce context dimension
        ("ctx_dim_2", {**base_config, "ctx_embedding_dim": 2}),
        ("ctx_dim_1", {**base_config, "ctx_embedding_dim": 1}),

        # Z-only decoder (no context)
        ("z_only", {**base_config, "z_only_decoder": True}),

        # Context dropout
        ("ctx_dropout_05", {**base_config, "ctx_embedding_dim": 3, "ctx_dropout": 0.5}),
        ("ctx_dropout_08", {**base_config, "ctx_embedding_dim": 3, "ctx_dropout": 0.8}),
    ]

    # Run experiments
    all_results = {}
    for exp_name, config_overrides in experiments:
        results = train_and_evaluate(
            exp_name=exp_name,
            config_overrides=config_overrides,
            log_returns=log_returns,
            log_surfaces=log_surfaces,
            train_sequences=train_sequences,
            val_sequences=val_sequences,
            output_dir=output_dir,
            n_epochs=100,
        )
        all_results[exp_name] = results

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print()

    print("| Experiment | ctx_dim | ctx_dropout | z_only | Decoder Gain | H=1 Viol | H=30 Viol | Val Loss |")
    print("|------------|---------|-------------|--------|--------------|----------|-----------|----------|")
    for exp_name, results in all_results.items():
        cfg = results["config"]
        ctx_dim = cfg.get("ctx_embedding_dim", 3)
        ctx_drop = cfg.get("ctx_dropout", 0.0)
        z_only = cfg.get("z_only_decoder", False)
        h1_viol = results["ci_results"].get("1", {}).get("violations", 0) * 100
        h30_viol = results["ci_results"].get("30", {}).get("violations", 0) * 100
        print(f"| {exp_name:10s} | {ctx_dim:7d} | {ctx_drop:11.1f} | {str(z_only):6s} | "
              f"{results['decoder_gain']:.2e}    | {h1_viol:7.1f}% | {h30_viol:8.1f}% | "
              f"{results['best_val_loss']:.6f} |")

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSummary saved to: {summary_path}")

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    baseline_gain = all_results["baseline"]["decoder_gain"]
    for exp_name, results in all_results.items():
        if exp_name == "baseline":
            continue
        gain_ratio = results["decoder_gain"] / baseline_gain
        print(f"\n{exp_name}:")
        print(f"  Decoder gain ratio vs baseline: {gain_ratio:.2f}x")

        baseline_h30 = all_results["baseline"]["ci_results"]["30"]["violations"]
        exp_h30 = results["ci_results"]["30"]["violations"]
        if baseline_h30 > 0:
            viol_change = (exp_h30 - baseline_h30) / baseline_h30 * 100
            sign = "+" if viol_change > 0 else ""
            print(f"  H=30 violation change: {sign}{viol_change:.1f}%")

    # Find best experiment
    best_exp = min(all_results.items(),
                   key=lambda x: x[1]["ci_results"]["30"]["violations"])
    print(f"\n*** Best experiment: {best_exp[0]} ***")
    print(f"    H=30 violations: {best_exp[1]['ci_results']['30']['violations']*100:.1f}%")
    print(f"    Decoder gain: {best_exp[1]['decoder_gain']:.2e}")

    print(f"\nAll results saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
