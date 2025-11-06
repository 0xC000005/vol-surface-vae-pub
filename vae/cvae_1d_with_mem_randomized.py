"""
1D Time Series CVAE with Memory (LSTM/GRU/RNN) and Variable Sequence Length.

Simplified architecture for 1D scalar time series (e.g., stock returns) instead of 2D surfaces.
Removes Conv2D layers, uses Linear layers for target encoding/decoding.

Architecture:
- Encoder: Target (1D) + Optional conditioning features → Linear → LSTM → Latent space
- Decoder: Latent + Context → LSTM → Linear → Target reconstruction
- Context Encoder: Same as encoder but for historical context only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from vae.base import BaseVAE, BaseDecoder, BaseEncoder
from collections import OrderedDict


class QuantileLoss(nn.Module):
    """
    Pinball loss for quantile regression (1D version).

    For quantile τ ∈ [0,1]:
    L_τ(y, ŷ) = max((τ-1)×(y-ŷ), τ×(y-ŷ))

    Asymmetric penalty:
    - τ=0.05: penalizes under-prediction more (wants most values above)
    - τ=0.50: reduces to MAE (mean absolute error)
    - τ=0.95: penalizes over-prediction more (wants most values below)
    """
    def __init__(self, quantiles=[0.05, 0.5, 0.95]):
        super().__init__()
        self.register_buffer('quantiles', torch.tensor(quantiles))

    def forward(self, preds, target):
        """
        Args:
            preds: (B, T, num_quantiles) - quantile predictions for 1D target
            target: (B, T, 1) - ground truth

        Returns:
            Scalar loss (averaged over all quantiles and elements)
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            pred_q = preds[:, :, i:i+1]  # (B, T, 1)
            error = target - pred_q  # (B, T, 1)
            loss_q = torch.max((q-1)*error, q*error)
            losses.append(torch.mean(loss_q))

        return torch.mean(torch.stack(losses))


class CVAE1DMemRandEncoder(BaseEncoder):
    """
    Encoder for 1D time series with optional conditioning features.

    Input:
        target: (B, T, 1) - Target time series (e.g., Amazon returns)
        cond_feats: (B, T, K) - Optional conditioning features (e.g., SP500, MSFT)

    Output:
        z_mean, z_log_var, z: (B, T, latent_dim) each

    Architecture:
        Target → Linear layers → Embeddings (B, T, target_hidden[-1])
        Cond → Linear layers (optional) → Embeddings (B, T, cond_embedding_dim)
        Concatenate → Interaction layers → LSTM → Latent projection
    """

    def __init__(self, config: dict):
        super(CVAE1DMemRandEncoder, self).__init__(config)

        latent_dim = config["latent_dim"]

        # Target embedding (scalar input → hidden layers)
        target_embedding_final_dim = self.__get_target_embedding(config)

        # Conditioning features embedding (optional)
        if config["cond_feats_dim"] > 0:
            cond_feats_embedding_final_dim = self.__get_cond_feats_embedding(config)
        else:
            cond_feats_embedding_final_dim = 0

        # Interaction layers (optional nonlinear mixing)
        total_embedding_dim = target_embedding_final_dim + cond_feats_embedding_final_dim
        self.__get_interaction_layers(config, total_embedding_dim)

        # LSTM memory module
        mem_final_dim = self.__get_mem(config, total_embedding_dim)

        # Latent space projection
        self.z_mean_layer = nn.Linear(mem_final_dim, latent_dim)
        self.z_log_var_layer = nn.Linear(mem_final_dim, latent_dim)

    def __get_target_embedding(self, config):
        """Linear layers for target time series embedding."""
        target_hidden = config["target_hidden"]

        target_embedding = OrderedDict()
        in_feats = 1  # Scalar input per timestep

        for i, out_feats in enumerate(target_hidden):
            target_embedding[f"target_linear_{i}"] = nn.Linear(in_feats, out_feats)
            target_embedding[f"target_activation_{i}"] = nn.ReLU()
            in_feats = out_feats

        self.target_embedding = nn.Sequential(target_embedding)
        return in_feats

    def __get_cond_feats_embedding(self, config):
        """Linear layers for conditioning features (optional)."""
        cond_feats_dim = config["cond_feats_dim"]
        cond_feats_hidden = config.get("cond_feats_hidden", None)

        if cond_feats_hidden is None:
            # Identity mapping (no transformation)
            self.cond_feats_embedding = nn.Identity()
            return cond_feats_dim
        else:
            # Linear layers
            cond_embedding = OrderedDict()
            in_feats = cond_feats_dim

            for i, out_feats in enumerate(cond_feats_hidden):
                cond_embedding[f"cond_linear_{i}"] = nn.Linear(in_feats, out_feats)
                cond_embedding[f"cond_activation_{i}"] = nn.ReLU()
                in_feats = out_feats

            self.cond_feats_embedding = nn.Sequential(cond_embedding)
            return in_feats

    def __get_interaction_layers(self, config, embedding_dim):
        """Optional nonlinear interaction layers between embeddings."""
        interaction_layers = config.get("interaction_layers", None)

        if interaction_layers is None or interaction_layers == 0:
            self.interaction = nn.Identity()
        else:
            layers = OrderedDict()
            for i in range(interaction_layers):
                layers[f"interaction_linear_{i}"] = nn.Linear(embedding_dim, embedding_dim)
                layers[f"interaction_activation_{i}"] = nn.ReLU()
            self.interaction = nn.Sequential(layers)

    def __get_mem(self, config, embedding_dim):
        """LSTM/GRU/RNN memory module for temporal encoding."""
        mem_type = config["mem_type"]
        mem_hidden = config["mem_hidden"]
        mem_layers = config["mem_layers"]
        mem_dropout = config.get("mem_dropout", 0.0)

        if mem_type == "lstm":
            self.mem = nn.LSTM(
                embedding_dim, mem_hidden,
                num_layers=mem_layers,
                batch_first=True,
                dropout=mem_dropout if mem_layers > 1 else 0.0
            )
        elif mem_type == "gru":
            self.mem = nn.GRU(
                embedding_dim, mem_hidden,
                num_layers=mem_layers,
                batch_first=True,
                dropout=mem_dropout if mem_layers > 1 else 0.0
            )
        elif mem_type == "rnn":
            self.mem = nn.RNN(
                embedding_dim, mem_hidden,
                num_layers=mem_layers,
                batch_first=True,
                dropout=mem_dropout if mem_layers > 1 else 0.0
            )
        else:
            raise ValueError(f"Unknown mem_type: {mem_type}")

        return mem_hidden

    def forward(self, x):
        """
        Args:
            x: dict with keys
                - "target": (B, T, 1) - Target time series
                - "cond_feats": (B, T, K) - Optional conditioning features

        Returns:
            z_mean, z_log_var, z: (B, T, latent_dim) each
        """
        target = x["target"]  # (B, T, 1)
        B, T, _ = target.shape

        # Embed target
        target_embedded = self.target_embedding(target)  # (B, T, target_hidden[-1])

        # Embed conditioning features (if present)
        if "cond_feats" in x and x["cond_feats"] is not None:
            cond_feats = x["cond_feats"]  # (B, T, K)
            cond_embedded = self.cond_feats_embedding(cond_feats)  # (B, T, cond_dim)
            embeddings = torch.cat([target_embedded, cond_embedded], dim=-1)
        else:
            embeddings = target_embedded

        # Interaction layers
        embeddings = self.interaction(embeddings)  # (B, T, embedding_dim)

        # LSTM temporal encoding
        mem_output, _ = self.mem(embeddings)  # (B, T, mem_hidden)

        # Latent space projection
        z_mean = self.z_mean_layer(mem_output)  # (B, T, latent_dim)
        z_log_var = self.z_log_var_layer(mem_output)  # (B, T, latent_dim)

        # Reparameterization trick
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        z = z_mean + eps * std

        return z_mean, z_log_var, z


class CVAE1DCtxMemRandEncoder(BaseEncoder):
    """
    Context encoder for 1D time series (encodes historical context only).

    Similar to main encoder but returns embeddings directly (not latent distribution).
    Used during generation to encode past observations.

    Input:
        target: (B, C, 1) - Historical context (C timesteps)
        cond_feats: (B, C, K) - Optional conditioning features

    Output:
        context_embeddings: (B, C, compressed_dim)
    """

    def __init__(self, config: dict):
        super(CVAE1DCtxMemRandEncoder, self).__init__(config)

        # Target embedding
        target_embedding_final_dim = self.__get_target_embedding(config)

        # Conditioning features embedding
        if config["cond_feats_dim"] > 0:
            cond_feats_embedding_final_dim = self.__get_cond_feats_embedding(config)
        else:
            cond_feats_embedding_final_dim = 0

        # Interaction layers
        total_embedding_dim = target_embedding_final_dim + cond_feats_embedding_final_dim
        self.__get_interaction_layers(config, total_embedding_dim)

        # LSTM memory
        mem_final_dim = self.__get_mem(config, total_embedding_dim)

        # Optional compression
        if config.get("compress_context", False):
            self.compress = nn.Linear(mem_final_dim, config["latent_dim"])
            self.compressed_dim = config["latent_dim"]
        else:
            self.compress = nn.Identity()
            self.compressed_dim = mem_final_dim

    def __get_target_embedding(self, config):
        """Target embedding layers (same as encoder)."""
        ctx_target_hidden = config.get("ctx_target_hidden", config["target_hidden"])

        target_embedding = OrderedDict()
        in_feats = 1

        for i, out_feats in enumerate(ctx_target_hidden):
            target_embedding[f"ctx_target_linear_{i}"] = nn.Linear(in_feats, out_feats)
            target_embedding[f"ctx_target_activation_{i}"] = nn.ReLU()
            in_feats = out_feats

        self.target_embedding = nn.Sequential(target_embedding)
        return in_feats

    def __get_cond_feats_embedding(self, config):
        """Conditioning features embedding (same as encoder)."""
        cond_feats_dim = config["cond_feats_dim"]
        ctx_cond_feats_hidden = config.get("ctx_cond_feats_hidden", config.get("cond_feats_hidden", None))

        if ctx_cond_feats_hidden is None:
            self.cond_feats_embedding = nn.Identity()
            return cond_feats_dim
        else:
            cond_embedding = OrderedDict()
            in_feats = cond_feats_dim

            for i, out_feats in enumerate(ctx_cond_feats_hidden):
                cond_embedding[f"ctx_cond_linear_{i}"] = nn.Linear(in_feats, out_feats)
                cond_embedding[f"ctx_cond_activation_{i}"] = nn.ReLU()
                in_feats = out_feats

            self.cond_feats_embedding = nn.Sequential(cond_embedding)
            return in_feats

    def __get_interaction_layers(self, config, embedding_dim):
        """Interaction layers."""
        interaction_layers = config.get("interaction_layers", None)

        if interaction_layers is None or interaction_layers == 0:
            self.interaction = nn.Identity()
        else:
            layers = OrderedDict()
            for i in range(interaction_layers):
                layers[f"ctx_interaction_linear_{i}"] = nn.Linear(embedding_dim, embedding_dim)
                layers[f"ctx_interaction_activation_{i}"] = nn.ReLU()
            self.interaction = nn.Sequential(layers)

    def __get_mem(self, config, embedding_dim):
        """LSTM memory module."""
        mem_type = config["mem_type"]
        mem_hidden = config["mem_hidden"]
        mem_layers = config["mem_layers"]
        mem_dropout = config.get("mem_dropout", 0.0)

        if mem_type == "lstm":
            self.mem = nn.LSTM(
                embedding_dim, mem_hidden,
                num_layers=mem_layers,
                batch_first=True,
                dropout=mem_dropout if mem_layers > 1 else 0.0
            )
        elif mem_type == "gru":
            self.mem = nn.GRU(
                embedding_dim, mem_hidden,
                num_layers=mem_layers,
                batch_first=True,
                dropout=mem_dropout if mem_layers > 1 else 0.0
            )
        elif mem_type == "rnn":
            self.mem = nn.RNN(
                embedding_dim, mem_hidden,
                num_layers=mem_layers,
                batch_first=True,
                dropout=mem_dropout if mem_layers > 1 else 0.0
            )
        else:
            raise ValueError(f"Unknown mem_type: {mem_type}")

        return mem_hidden

    def forward(self, x):
        """
        Args:
            x: dict with keys
                - "target": (B, C, 1) - Historical context
                - "cond_feats": (B, C, K) - Optional conditioning features

        Returns:
            context_embeddings: (B, C, compressed_dim)
        """
        target = x["target"]

        # Embed target
        target_embedded = self.target_embedding(target)

        # Embed conditioning features
        if "cond_feats" in x and x["cond_feats"] is not None:
            cond_feats = x["cond_feats"]
            cond_embedded = self.cond_feats_embedding(cond_feats)
            embeddings = torch.cat([target_embedded, cond_embedded], dim=-1)
        else:
            embeddings = target_embedded

        # Interaction
        embeddings = self.interaction(embeddings)

        # LSTM
        mem_output, _ = self.mem(embeddings)

        # Compression (optional)
        context_embeddings = self.compress(mem_output)

        return context_embeddings


class CVAE1DMemRandDecoder(BaseDecoder):
    """
    Decoder for 1D time series.

    Input:
        Concatenated [context_embeddings, z]: (B, T, ctx_dim + latent_dim)

    Output:
        decoded_target: (B, T, 1) - Reconstructed target
        decoded_cond_feats: (B, T, K) - Reconstructed conditioning features (if present)

    Architecture:
        LSTM → Interaction layers → Split branches:
            - Target branch: Linear layers → (B, T, 1)
            - Cond branch: Linear layers → (B, T, K) [optional]
    """

    def __init__(self, config: dict):
        super(CVAE1DMemRandDecoder, self).__init__(config)

        ctx_dim = config["latent_dim"] if config.get("compress_context", False) else config["mem_hidden"]
        latent_dim = config["latent_dim"]
        input_dim = ctx_dim + latent_dim

        # Determine target output dimension
        target_hidden = config["target_hidden"]
        target_output_dim = target_hidden[-1]

        # Determine conditioning output dimension
        has_cond_feats = config["cond_feats_dim"] > 0
        if has_cond_feats:
            cond_output_dim = config["cond_feats_dim"]
            mem_output_dim = target_output_dim + cond_output_dim
        else:
            mem_output_dim = target_output_dim

        # LSTM memory
        self.__get_mem(config, input_dim, mem_output_dim)

        # Interaction layers
        self.__get_interaction_layers(config, mem_output_dim)

        # Target decoder branch
        self.__get_target_decoder(config)

        # Conditioning features decoder branch (optional)
        if has_cond_feats:
            self.__get_cond_feats_decoder(config)
        else:
            self.cond_feats_decoder = None

    def __get_mem(self, config, input_dim, output_dim):
        """LSTM memory module."""
        mem_type = config["mem_type"]
        mem_layers = config["mem_layers"]
        mem_dropout = config.get("mem_dropout", 0.0)

        if mem_type == "lstm":
            self.mem = nn.LSTM(
                input_dim, output_dim,
                num_layers=mem_layers,
                batch_first=True,
                dropout=mem_dropout if mem_layers > 1 else 0.0
            )
        elif mem_type == "gru":
            self.mem = nn.GRU(
                input_dim, output_dim,
                num_layers=mem_layers,
                batch_first=True,
                dropout=mem_dropout if mem_layers > 1 else 0.0
            )
        elif mem_type == "rnn":
            self.mem = nn.RNN(
                input_dim, output_dim,
                num_layers=mem_layers,
                batch_first=True,
                dropout=mem_dropout if mem_layers > 1 else 0.0
            )
        else:
            raise ValueError(f"Unknown mem_type: {mem_type}")

    def __get_interaction_layers(self, config, mem_output_dim):
        """Interaction layers."""
        interaction_layers = config.get("interaction_layers", None)

        if interaction_layers is None or interaction_layers == 0:
            self.interaction = nn.Identity()
        else:
            layers = OrderedDict()
            for i in range(interaction_layers):
                layers[f"dec_interaction_linear_{i}"] = nn.Linear(mem_output_dim, mem_output_dim)
                layers[f"dec_interaction_activation_{i}"] = nn.ReLU()
            self.interaction = nn.Sequential(layers)

    def __get_target_decoder(self, config):
        """Target decoder branch (Linear layers → scalar output)."""
        target_hidden = config["target_hidden"]

        target_decoder = OrderedDict()

        # Reverse the hidden layers for decoding
        in_feats = target_hidden[-1]
        for i in range(len(target_hidden) - 2, -1, -1):
            out_feats = target_hidden[i]
            target_decoder[f"dec_target_linear_{i}"] = nn.Linear(in_feats, out_feats)
            target_decoder[f"dec_target_activation_{i}"] = nn.ReLU()
            in_feats = out_feats

        # Final layer: output quantiles
        num_quantiles = config.get("num_quantiles", 3)
        target_decoder["dec_target_output"] = nn.Linear(in_feats, num_quantiles)

        self.target_decoder = nn.Sequential(target_decoder)

    def __get_cond_feats_decoder(self, config):
        """Conditioning features decoder branch (simple linear projection)."""
        cond_feats_dim = config["cond_feats_dim"]

        # Simple linear projection (no deep layers for now)
        self.cond_feats_decoder = nn.Linear(cond_feats_dim, cond_feats_dim)

    def forward(self, x):
        """
        Args:
            x: (B, T, ctx_dim + latent_dim) - Concatenated context and latent

        Returns:
            decoded_target: (B, T, num_quantiles) - Quantile predictions
            decoded_cond_feats: (B, T, K) or None
        """
        # LSTM
        mem_output, _ = self.mem(x)  # (B, T, mem_output_dim)

        # Interaction
        combined = self.interaction(mem_output)

        # Split into target and conditioning branches
        target_hidden_dim = self.target_decoder[0].in_features if len(self.target_decoder) > 0 else combined.shape[-1]

        if self.cond_feats_decoder is not None:
            cond_dim = self.cond_feats_decoder.in_features
            target_features = combined[..., :target_hidden_dim]
            cond_features = combined[..., target_hidden_dim:]

            # Decode target
            decoded_target = self.target_decoder(target_features)  # (B, T, num_quantiles)

            # Decode conditioning features
            decoded_cond_feats = self.cond_feats_decoder(cond_features)  # (B, T, K)

            return decoded_target, decoded_cond_feats
        else:
            # Only decode target
            decoded_target = self.target_decoder(combined)  # (B, T, num_quantiles)
            return decoded_target, None


class CVAE1DMemRand(BaseVAE):
    """
    1D Time Series Conditional VAE with LSTM Memory and Variable Sequence Length.

    Architecture for scalar time series (e.g., stock returns) with optional conditioning.

    Model variants:
    - cond_feats_dim=0: Target only (baseline)
    - cond_feats_dim>0, cond_feat_weight=0: Passive conditioning (no loss)
    - cond_feats_dim>0, cond_feat_weight>0: Active conditioning (with loss)

    Training:
        Input: Full sequence [context + target]
        Encode with main encoder → Sample latent → Decode → Compute loss

    Generation:
        Input: Context only
        Encode with context encoder → Sample latent → Decode → Predict next timestep
    """

    def __init__(self, config: dict):
        super(CVAE1DMemRand, self).__init__(config)

        self.config = config
        self.device = config.get("device", "cpu")

        # Loss weights
        self.kl_weight = config.get("kl_weight", 1e-5)
        self.cond_feat_weight = config.get("cond_feat_weight", 0.0)
        self.cond_loss_type = config.get("cond_loss_type", "l2")

        # Quantile regression defaults
        if "num_quantiles" not in config:
            config["num_quantiles"] = 3
        if "quantiles" not in config:
            config["quantiles"] = [0.05, 0.5, 0.95]

        # Create all modules first
        self.encoder = CVAE1DMemRandEncoder(config)
        self.ctx_encoder = CVAE1DCtxMemRandEncoder(config)
        self.decoder = CVAE1DMemRandDecoder(config)

        # Move entire model to device at once (respects torch.set_default_dtype)
        self.to(self.device)

        # Initialize quantile loss function
        self.quantile_loss_fn = QuantileLoss(quantiles=config["quantiles"])

        # Initialize conditional features loss if needed
        if self.cond_feat_weight > 0:
            if self.cond_loss_type == "l2":
                self.cond_feat_loss_fn = nn.MSELoss()
            else:
                self.cond_feat_loss_fn = nn.L1Loss()

    def forward(self, x):
        """
        Forward pass for training.

        Args:
            x: dict with keys
                - "target": (B, T, 1) - Full sequence
                - "cond_feats": (B, T, K) - Optional conditioning features

        Returns:
            decoded_target: (B, 1, num_quantiles) - Quantile predictions for future timestep
            decoded_cond_feats: (B, 1, K) or None
            z_mean, z_log_var, z: Latent variables
            Note: Returns only FUTURE timestep (C:T), not full sequence
        """
        target = x["target"]
        B = target.shape[0]
        T = target.shape[1]
        C = T - 1  # Context length

        # Extract context (first C timesteps)
        ctx_target = target[:, :C, :]
        ctx_input = {"target": ctx_target}

        if "cond_feats" in x:
            cond_feats = x["cond_feats"]
            ctx_cond_feats = cond_feats[:, :C, :]
            ctx_input["cond_feats"] = ctx_cond_feats

        # Encode full sequence
        encoder_input = x
        z_mean, z_log_var, z = self.encoder(encoder_input)

        # Get context embeddings (only from context, not full sequence)
        ctx_embeddings = self.ctx_encoder(ctx_input)
        ctx_dim = ctx_embeddings.shape[2]

        # Pad context embeddings to match full sequence length T
        ctx_padded = torch.zeros((B, T, ctx_dim), device=self.device, dtype=z.dtype)
        ctx_padded[:, :C, :] = ctx_embeddings

        # Concatenate context and latent
        decoder_input = torch.cat([ctx_padded, z], dim=-1)

        # Decode
        decoded_target, decoded_cond_feats = self.decoder(decoder_input)

        # Return ONLY future prediction (timestep C onwards)
        if decoded_cond_feats is not None:
            return decoded_target[:, C:, :], decoded_cond_feats[:, C:, :], z_mean, z_log_var, z
        else:
            return decoded_target[:, C:, :], None, z_mean, z_log_var, z

    def compute_loss(self, x, decoded_target, decoded_cond_feats, z_mean, z_log_var):
        """
        Compute VAE loss: reconstruction + KL divergence.

        Loss = Quantile(target) + cond_feat_weight * Loss(cond_feats) + kl_weight * KL

        Note: decoded_target is only future prediction (shape B, 1, num_quantiles).
              We need to extract future ground truth from x["target"].
        """
        target = x["target"]
        T = target.shape[1]
        C = T - 1

        # Extract future ground truth (last timestep)
        target_future = target[:, C:, :]

        # Reconstruction loss for target (Quantile Loss)
        # decoded_target: (B, 1, num_quantiles), target_future: (B, 1, 1)
        recon_loss_target = self.quantile_loss_fn(decoded_target, target_future)

        # Reconstruction loss for conditioning features (if used)
        if decoded_cond_feats is not None and self.cond_feat_weight > 0:
            cond_feats = x["cond_feats"]
            # Extract future ground truth (last timestep)
            cond_feats_future = cond_feats[:, C:, :]
            recon_loss_cond = self.cond_feat_loss_fn(decoded_cond_feats, cond_feats_future)
        else:
            recon_loss_cond = torch.zeros(1, device=self.device)

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

        # Total loss
        total_recon_loss = recon_loss_target + self.cond_feat_weight * recon_loss_cond
        total_loss = total_recon_loss + self.kl_weight * kl_loss

        return {
            "loss": total_loss,
            "recon_loss": total_recon_loss,
            "target_loss": recon_loss_target,
            "cond_loss": recon_loss_cond,
            "kl_loss": kl_loss,
        }

    def train_step(self, x, optimizer):
        """Single training step."""
        self.train()

        # Move tensors to device
        x_device = {}
        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                x_device[k] = v.to(self.device)
            else:
                x_device[k] = v

        optimizer.zero_grad()

        decoded_target, decoded_cond_feats, z_mean, z_log_var, z = self.forward(x_device)
        loss_dict = self.compute_loss(x_device, decoded_target, decoded_cond_feats, z_mean, z_log_var)

        loss_dict["loss"].backward()
        optimizer.step()

        return loss_dict

    def test_step(self, x):
        """Single test/validation step."""
        self.eval()

        # Move tensors to device
        x_device = {}
        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                x_device[k] = v.to(self.device)
            else:
                x_device[k] = v

        with torch.no_grad():
            decoded_target, decoded_cond_feats, z_mean, z_log_var, z = self.forward(x_device)
            loss_dict = self.compute_loss(x_device, decoded_target, decoded_cond_feats, z_mean, z_log_var)

        return loss_dict

    def get_prediction_given_context(self, c):
        """
        Generate quantile predictions given historical context.

        Args:
            c: dict with keys
                - "target": (B, C, 1) - Historical context
                - "cond_feats": (B, C, K) - Optional conditioning features

        Returns:
            predictions: (B, num_quantiles) - Quantile predictions [p05, p50, p95]
        """
        self.eval()
        with torch.no_grad():
            B = c["target"].shape[0]
            C = c["target"].shape[1]

            # Get context embeddings
            ctx_embeddings = self.ctx_encoder(c)  # (B, C, ctx_dim)
            ctx_dim = ctx_embeddings.shape[2]
            latent_dim = self.config["latent_dim"]

            # Prepare padded context: (B, C+1, ctx_dim)
            ctx_padded = torch.zeros((B, C + 1, ctx_dim), device=self.device, dtype=ctx_embeddings.dtype)
            ctx_padded[:, :C, :] = ctx_embeddings

            # Sample latent for future timestep (use mode of prior z~N(0,1))
            z_future = torch.zeros((B, 1, latent_dim), device=self.device, dtype=ctx_embeddings.dtype)

            # Prepare latent: (B, C+1, latent_dim)
            # Use zeros for context (deterministic encoding), zeros for future (mode)
            z_full = torch.zeros((B, C + 1, latent_dim), device=self.device, dtype=ctx_embeddings.dtype)
            z_full[:, C:, :] = z_future

            # Concatenate context and latent
            decoder_input = torch.cat([ctx_padded, z_full], dim=-1)

            # Decode
            decoded_target, _ = self.decoder(decoder_input)

            # Extract future prediction
            # decoded_target: (B, 1, num_quantiles)
            prediction = decoded_target[:, C:, :].squeeze(1)  # (B, num_quantiles)

            return prediction
