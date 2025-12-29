"""
Full Covariance Prior for VAE with Position-Encoded Means and AR(1) Covariance

This module implements a full covariance prior that addresses the parameter reuse
problem in the baseline conditional prior. Instead of reusing the same (μ, σ) for
all future timesteps, it uses:
1. Position-encoded means: μ_t = f(context_emb, pos_enc(t))
2. Global AR(1) covariance: Σ[i,j] = σ² × φ^|i-j|

Expected improvement: Roughness ratio from 9.7% → ~50% (vs oracle's 75%)
"""

import torch
import torch.nn as nn
import math


class SinusoidalPositionEncoding(nn.Module):
    """
    Transformer-style sinusoidal position embeddings.

    Creates unique embeddings for each position using sine/cosine functions
    of different frequencies. This allows the model to distinguish between
    different timesteps in the forecast horizon.

    Args:
        d_model: Embedding dimension
        max_len: Maximum sequence length
    """
    def __init__(self, d_model=64, max_len=100):
        super().__init__()
        self.d_model = d_model

        # Create position encoding matrix
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)

    def forward(self, timesteps):
        """
        Get position encodings for specified timesteps.

        Args:
            timesteps: (H,) tensor of timestep indices (e.g., [0, 1, 2, ..., H-1])

        Returns:
            (H, d_model) position encodings
        """
        return self.pe[timesteps]


class PositionEncodedPriorMean(nn.Module):
    """
    Network that outputs μ_t for each timestep, conditioned on context.

    Instead of reusing the same μ for all timesteps, this network combines:
    - Context summary from encoder (global information)
    - Position encoding (which timestep we're predicting)

    This allows different μ_t for t=1, t=2, ..., t=H

    Args:
        context_dim: Dimension of context summary (latent_dim when compress_context=True)
        max_horizon: Maximum forecast horizon
        latent_dim: Dimension of latent space z
        pos_dim: Dimension of position encoding
        hidden_dims: List of hidden layer dimensions (default: [128, 128])
        dropout: Dropout rate after each hidden layer (default: 0.1)
    """
    def __init__(self, context_dim=12, max_horizon=90, latent_dim=12,
                 pos_dim=64, hidden_dims=None, dropout=0.1):
        super().__init__()
        self.context_dim = context_dim
        self.latent_dim = latent_dim
        self.max_horizon = max_horizon

        # Default hidden dims for backward compatibility
        if hidden_dims is None:
            hidden_dims = [128, 128]

        # Position encoding
        self.pos_encoder = SinusoidalPositionEncoding(d_model=pos_dim, max_len=max_horizon)

        # Build MLP with configurable layers and dropout
        layers = []
        in_dim = context_dim + pos_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, latent_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, context_summary, horizon):
        """
        Generate position-encoded prior means.

        Args:
            context_summary: (B, context_dim) context embedding
            horizon: Integer, forecast horizon

        Returns:
            mu_p: (B, H, latent_dim) position-dependent prior means
        """
        B = context_summary.shape[0]
        device = context_summary.device

        # Get position encodings for all timesteps
        timesteps = torch.arange(horizon, device=device)  # (H,)
        pos_enc = self.pos_encoder(timesteps)  # (H, pos_dim)

        # Expand context for all timesteps
        context_expanded = context_summary.unsqueeze(1).expand(B, horizon, -1)  # (B, H, context_dim)
        pos_enc_expanded = pos_enc.unsqueeze(0).expand(B, -1, -1)  # (B, H, pos_dim)

        # Concatenate and pass through MLP
        combined = torch.cat([context_expanded, pos_enc_expanded], dim=-1)  # (B, H, context_dim + pos_dim)
        mu_p = self.mlp(combined)  # (B, H, latent_dim)

        return mu_p


class RNNPriorMean(nn.Module):
    """
    RNN-based prior mean network with temporal dependency.

    Unlike PositionEncodedPriorMean which processes each timestep independently,
    this network autoregressively generates μ_t where each mean depends on
    the previous hidden state, creating temporal dependency in the prior means.

    Architecture:
        h_0 = f(context_summary)
        For t in [0, horizon):
            input_t = pos_enc(t) [if use_position_encoding] else zeros
            h_t, mu_t = RNN(input_t, h_{t-1})

    Args:
        context_dim: Dimension of context summary (latent_dim when compress_context=True)
        max_horizon: Maximum forecast horizon
        latent_dim: Dimension of latent space z
        rnn_type: 'lstm' or 'gru'
        hidden_dim: Hidden dimension of RNN
        num_layers: Number of RNN layers (default: 1)
        use_position_encoding: If True, feed position encoding as input (default: False)
        pos_dim: Dimension of position encoding if used (default: 64)
        dropout: Dropout rate (only applies if num_layers > 1, default: 0.1)
    """
    def __init__(self, context_dim=12, max_horizon=90, latent_dim=12,
                 rnn_type='lstm', hidden_dim=32, num_layers=1,
                 use_position_encoding=False, pos_dim=64, dropout=0.1):
        super().__init__()
        self.context_dim = context_dim
        self.latent_dim = latent_dim
        self.max_horizon = max_horizon
        self.rnn_type = rnn_type.lower()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_position_encoding = use_position_encoding

        # Context encoder: project context_summary to initial hidden state
        self.context_to_hidden = nn.Linear(context_dim, hidden_dim * num_layers)

        # Optional position encoding
        if use_position_encoding:
            self.pos_encoder = SinusoidalPositionEncoding(d_model=pos_dim, max_len=max_horizon)
            rnn_input_dim = pos_dim
        else:
            self.pos_encoder = None
            rnn_input_dim = latent_dim  # Feed back previous output

        # RNN cell
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=rnn_input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=False  # (seq_len, batch, features)
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=rnn_input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=False
            )
        else:
            raise ValueError(f"rnn_type must be 'lstm' or 'gru', got '{rnn_type}'")

        # Output layer: RNN hidden -> latent_dim
        self.output_layer = nn.Linear(hidden_dim, latent_dim)

        # Initial input (if not using position encoding)
        if not use_position_encoding:
            self.init_input = nn.Parameter(torch.zeros(1, 1, latent_dim))

    def init_hidden(self, context_summary):
        """
        Initialize RNN hidden state from context summary.

        Args:
            context_summary: (B, context_dim)

        Returns:
            h_0: Initial hidden state for RNN
                 LSTM: tuple of (h_0, c_0), each (num_layers, B, hidden_dim)
                 GRU: (num_layers, B, hidden_dim)
        """
        B = context_summary.shape[0]
        device = context_summary.device

        # Project context to hidden state
        h_flat = self.context_to_hidden(context_summary)  # (B, num_layers * hidden_dim)
        h_0 = h_flat.view(B, self.num_layers, self.hidden_dim)  # (B, num_layers, hidden_dim)
        h_0 = h_0.transpose(0, 1).contiguous()  # (num_layers, B, hidden_dim)

        if self.rnn_type == 'lstm':
            # LSTM needs both hidden and cell state
            c_0 = torch.zeros_like(h_0)
            return (h_0, c_0)
        else:
            # GRU only needs hidden state
            return h_0

    def forward(self, context_summary, horizon):
        """
        Autoregressively generate prior means with temporal dependency.

        Args:
            context_summary: (B, context_dim) context embedding
            horizon: Integer, forecast horizon

        Returns:
            mu_p: (B, H, latent_dim) temporally-dependent prior means
        """
        B = context_summary.shape[0]
        device = context_summary.device

        # Initialize hidden state from context
        hidden = self.init_hidden(context_summary)

        outputs = []

        if self.use_position_encoding:
            # Use position encoding as input at each timestep
            timesteps = torch.arange(horizon, device=device)
            pos_encodings = self.pos_encoder(timesteps)  # (H, pos_dim)

            for t in range(horizon):
                # Input: position encoding for timestep t
                input_t = pos_encodings[t:t+1].unsqueeze(1).expand(-1, B, -1)  # (1, B, pos_dim)

                # RNN step
                output_t, hidden = self.rnn(input_t, hidden)  # output_t: (1, B, hidden_dim)

                # Project to latent_dim
                mu_t = self.output_layer(output_t.squeeze(0))  # (B, latent_dim)
                outputs.append(mu_t)
        else:
            # Use previous output as input (or initial learned embedding for t=0)
            prev_output = self.init_input.expand(-1, B, -1)  # (1, B, latent_dim)

            for t in range(horizon):
                # RNN step
                output_t, hidden = self.rnn(prev_output, hidden)  # (1, B, hidden_dim)

                # Project to latent_dim
                mu_t = self.output_layer(output_t.squeeze(0))  # (B, latent_dim)
                outputs.append(mu_t)

                # Use current output as next input
                prev_output = mu_t.unsqueeze(0)  # (1, B, latent_dim)

        # Stack outputs: (B, H, latent_dim)
        mu_p = torch.stack(outputs, dim=1)

        return mu_p


def build_ar1_covariance(phi, sigma_sq, horizon, device='cpu', dtype=None):
    """
    Build AR(1) covariance matrix: Σ[i,j] = σ² × φ^|i-j|

    This creates a Toeplitz matrix where correlations decay exponentially
    with distance. Closer timesteps are more correlated.

    Args:
        phi: AR(1) coefficient (0 < phi < 1)
        sigma_sq: Variance σ²
        horizon: Forecast horizon H
        device: Device for tensor
        dtype: Data type (if None, inferred from phi/sigma_sq)

    Returns:
        Sigma: (H, H) covariance matrix
    """
    # Infer dtype from inputs
    if dtype is None:
        if isinstance(phi, torch.Tensor):
            dtype = phi.dtype
        elif isinstance(sigma_sq, torch.Tensor):
            dtype = sigma_sq.dtype
        else:
            dtype = torch.get_default_dtype()

    # Compute |i - j| for all pairs
    indices = torch.arange(horizon, device=device)
    distance_matrix = torch.abs(indices.unsqueeze(1) - indices.unsqueeze(0))  # (H, H)

    # Convert distance_matrix to appropriate dtype
    distance_matrix = distance_matrix.to(dtype)

    # Ensure phi and sigma_sq are tensors with correct dtype
    if not isinstance(phi, torch.Tensor):
        phi = torch.tensor(phi, device=device, dtype=dtype)
    if not isinstance(sigma_sq, torch.Tensor):
        sigma_sq = torch.tensor(sigma_sq, device=device, dtype=dtype)

    # Σ[i,j] = σ² × φ^|i-j|
    Sigma = sigma_sq * (phi ** distance_matrix)

    return Sigma


def _build_ar1_cholesky_direct_loop(phi, sigma, horizon, device='cpu', dtype=None):
    """
    Build Cholesky factor L directly for AR(1) covariance - ORIGINAL loop-based implementation.

    This is the original implementation with Python loops, kept for correctness verification
    of the vectorized version. Uses closed-form formula for AR(1) Cholesky factor:
    - First column: L[i,0] = sigma * phi^i
    - Diagonal: L[i,i] = sigma * sqrt(1 - phi²) for i > 0
    - Lower triangle: L[i,j] = sigma * phi^(i-j) * sqrt(1 - phi²)

    Args:
        phi: AR(1) coefficient (0 < phi < 1) - can be float or tensor
        sigma: Standard deviation σ - can be float or tensor
        horizon: Forecast horizon H
        device: Device for tensor
        dtype: Data type (if None, inferred from phi/sigma or uses default)

    Returns:
        L: (H, H) lower triangular Cholesky factor
    """
    # Convert to tensors if needed, preserving dtype
    if not isinstance(phi, torch.Tensor):
        if dtype is None:
            dtype = torch.get_default_dtype()
        phi = torch.tensor(phi, device=device, dtype=dtype)
    else:
        dtype = phi.dtype if dtype is None else dtype

    if not isinstance(sigma, torch.Tensor):
        sigma = torch.tensor(sigma, device=device, dtype=dtype)

    L = torch.zeros(horizon, horizon, device=device, dtype=dtype)
    sqrt_1_minus_phi2 = torch.sqrt(1 - phi**2)

    # First column: L[i,0] = sigma * phi^i
    for i in range(horizon):
        L[i, 0] = sigma * (phi ** i)

    # Diagonal and lower triangle
    for i in range(1, horizon):
        L[i, i] = sigma * sqrt_1_minus_phi2
        for j in range(1, i):
            L[i, j] = sigma * (phi ** (i - j)) * sqrt_1_minus_phi2

    return L


def build_ar1_cholesky_direct(phi, sigma, horizon, device='cpu', dtype=None):
    """
    Build Cholesky factor L directly for AR(1) covariance.

    Vectorized implementation - O(H) instead of O(H²) Python loops.
    Uses closed-form formula for AR(1) Cholesky factor:
    - First column: L[i,0] = sigma * phi^i
    - Diagonal: L[i,i] = sigma * sqrt(1 - phi²) for i > 0
    - Lower triangle: L[i,j] = sigma * phi^(i-j) * sqrt(1 - phi²)

    Args:
        phi: AR(1) coefficient (0 < phi < 1) - can be float or tensor
        sigma: Standard deviation σ - can be float or tensor
        horizon: Forecast horizon H
        device: Device for tensor
        dtype: Data type (if None, inferred from phi/sigma or uses default)

    Returns:
        L: (H, H) lower triangular Cholesky factor
    """
    # Convert to tensors if needed, preserving dtype
    if not isinstance(phi, torch.Tensor):
        if dtype is None:
            dtype = torch.get_default_dtype()
        phi = torch.tensor(phi, device=device, dtype=dtype)
    else:
        dtype = phi.dtype if dtype is None else dtype

    if not isinstance(sigma, torch.Tensor):
        sigma = torch.tensor(sigma, device=device, dtype=dtype)

    L = torch.zeros(horizon, horizon, device=device, dtype=dtype)
    sqrt_1_minus_phi2 = torch.sqrt(1 - phi**2)

    # First column: L[i,0] = sigma * phi^i (vectorized)
    indices = torch.arange(horizon, device=device, dtype=dtype)
    L[:, 0] = sigma * torch.pow(phi, indices)

    if horizon > 1:
        # Diagonal (i>0): L[i,i] = sigma * sqrt(1-phi²)
        diag_indices = torch.arange(1, horizon, device=device)
        L[diag_indices, diag_indices] = sigma * sqrt_1_minus_phi2

        # Lower triangle (j>0, j<i): L[i,j] = sigma * phi^(i-j) * sqrt(1-phi²)
        if horizon > 2:
            row_idx, col_idx = torch.tril_indices(horizon, horizon, offset=-1, device=device)
            mask = col_idx > 0
            row_idx, col_idx = row_idx[mask], col_idx[mask]
            if len(row_idx) > 0:
                L[row_idx, col_idx] = sigma * torch.pow(phi, (row_idx - col_idx).to(dtype)) * sqrt_1_minus_phi2

    return L


def compute_empirical_quantiles(samples, quantiles=None):
    """
    Compute empirical quantiles from multiple samples.

    With the quantile decoder removed, confidence intervals should be computed
    from multiple samples generated via Cholesky sampling. This is more flexible
    than quantile regression (any quantile, not just 3).

    Args:
        samples: Tensor of shape (num_samples, B, H, ...) where first dim is samples
                 Can be latents (num_samples, B, H, latent_dim) or
                 surfaces (num_samples, B, H, 5, 5)
        quantiles: List of quantile values in [0, 1]. Default: [0.05, 0.5, 0.95]

    Returns:
        dict with keys like 'q05', 'q50', 'q95', each containing tensor of
        shape (B, H, ...) representing that quantile across samples

    Example:
        >>> samples = model.sample(context, horizon=30, num_samples=1000)  # (1000, B, 30, 12)
        >>> quantiles_dict = compute_empirical_quantiles(samples)
        >>> q05 = quantiles_dict['q05']  # (B, 30, 12) - 5th percentile
        >>> q50 = quantiles_dict['q50']  # (B, 30, 12) - median
        >>> q95 = quantiles_dict['q95']  # (B, 30, 12) - 95th percentile
    """
    if quantiles is None:
        quantiles = [0.05, 0.5, 0.95]

    result = {}
    for q in quantiles:
        # Create key like 'q05', 'q50', 'q95'
        key = f"q{int(q * 100):02d}"
        # Compute quantile along sample dimension (dim=0)
        result[key] = torch.quantile(samples, q, dim=0)

    return result


def _kl_divergence_full_covariance_loop(mu_q, logvar_q, mu_p, Sigma_p):
    """
    KL divergence - ORIGINAL loop-based implementation (for verification).

    This version iterates over latent dimensions. Kept temporarily for correctness
    verification of the vectorized version.
    """
    B, H, latent_dim = mu_q.shape

    # Expand Sigma_p if needed
    if Sigma_p.dim() == 2:
        Sigma_p = Sigma_p.unsqueeze(0).expand(B, -1, -1)  # (B, H, H)

    # Compute Sigma_p^{-1}
    Sigma_p_inv = torch.linalg.inv(Sigma_p)  # (B, H, H)

    # Compute terms for each latent dimension
    kl_total = 0.0

    for d in range(latent_dim):
        mu_q_d = mu_q[:, :, d]  # (B, H)
        mu_p_d = mu_p[:, :, d]  # (B, H)
        var_q_d = torch.exp(logvar_q[:, :, d])  # (B, H)

        # Term 1: tr(Sigma_p^{-1} Sigma_q)
        trace_term = torch.sum(torch.diagonal(Sigma_p_inv, dim1=1, dim2=2) * var_q_d, dim=1)  # (B,)

        # Term 2: (mu_p - mu_q)^T Sigma_p^{-1} (mu_p - mu_q)
        mu_diff = mu_p_d - mu_q_d  # (B, H)
        mahalanobis = torch.bmm(
            mu_diff.unsqueeze(1),  # (B, 1, H)
            torch.bmm(Sigma_p_inv, mu_diff.unsqueeze(-1))  # (B, H, 1)
        ).squeeze()  # (B,)

        # Term 3: log(det(Sigma_p) / det(Sigma_q))
        log_det_Sigma_q = torch.sum(logvar_q[:, :, d], dim=1)  # (B,)
        sign_p, log_det_Sigma_p = torch.linalg.slogdet(Sigma_p)
        log_det_ratio = log_det_Sigma_p - log_det_Sigma_q  # (B,)

        # Combine: KL = 0.5 * (trace + mahalanobis - k + log_det_ratio)
        kl_d = 0.5 * (trace_term + mahalanobis - H + log_det_ratio)  # (B,)
        kl_total += kl_d.mean()

    return kl_total


def kl_divergence_full_covariance(mu_q, logvar_q, mu_p, Sigma_p):
    """
    KL divergence: KL(q || p) where q is diagonal, p has full covariance.

    Vectorized implementation - computes all latent dimensions in parallel.

    Args:
        mu_q: (B, H, latent_dim) posterior mean
        logvar_q: (B, H, latent_dim) posterior log-variance (diagonal)
        mu_p: (B, H, latent_dim) prior mean
        Sigma_p: (H, H) prior covariance (shared across latent dims and batch)

    Returns:
        kl: Scalar KL divergence
    """
    B, H, latent_dim = mu_q.shape

    # Expand Sigma_p: (H, H) -> (B, H, H)
    if Sigma_p.dim() == 2:
        Sigma_p = Sigma_p.unsqueeze(0).expand(B, -1, -1)

    # Compute Sigma_p^{-1} once
    Sigma_p_inv = torch.linalg.inv(Sigma_p)  # (B, H, H)

    # Get diagonal of Sigma_p_inv for trace computation
    Sigma_p_inv_diag = torch.diagonal(Sigma_p_inv, dim1=1, dim2=2)  # (B, H)

    # Variance from posterior: (B, H, D)
    var_q = torch.exp(logvar_q)

    # Term 1: tr(Sigma_p^{-1} @ diag(var_q)) for each latent dim
    # = sum over H of (Sigma_p_inv_diag * var_q)
    # Shape: (B, H) * (B, H, D) -> sum over H -> (B, D)
    trace_term = torch.einsum('bh,bhd->bd', Sigma_p_inv_diag, var_q)  # (B, D)

    # Term 2: Mahalanobis distance (mu_p - mu_q)^T Sigma_p^{-1} (mu_p - mu_q)
    mu_diff = mu_p - mu_q  # (B, H, D)
    # For each d: mu_diff[:,:,d]^T @ Sigma_p_inv @ mu_diff[:,:,d]
    # Use einsum: (B,H,D) @ (B,H,H) @ (B,H,D) -> (B,D)
    mahal = torch.einsum('bhd,bhk,bkd->bd', mu_diff, Sigma_p_inv, mu_diff)  # (B, D)

    # Term 3: log det ratio
    # log det(Sigma_p) - log det(Sigma_q) where Sigma_q is diagonal
    # log det(Sigma_q) = sum of logvar_q over H for each d
    log_det_Sigma_q = logvar_q.sum(dim=1)  # (B, D)
    _, log_det_Sigma_p = torch.linalg.slogdet(Sigma_p)  # (B,)
    log_det_ratio = log_det_Sigma_p.unsqueeze(-1) - log_det_Sigma_q  # (B, D)

    # KL per dimension: 0.5 * (trace + mahal - H + log_det_ratio)
    kl_per_dim = 0.5 * (trace_term + mahal - H + log_det_ratio)  # (B, D)

    # Sum over dimensions, mean over batch
    return kl_per_dim.sum(dim=1).mean()


class FullCovariancePrior(nn.Module):
    """
    Complete Full Covariance Prior combining:
    1. Position-encoded or RNN-based means: μ_t = f(context, t)
    2. Global AR(1) covariance: Σ[i,j] = σ² × φ^|i-j|

    Only 2 learnable scalar parameters for covariance: φ, σ²
    (vs 187K parameters in baseline conditional prior)

    Args:
        context_dim: Dimension of context summary
        max_horizon: Maximum forecast horizon
        latent_dim: Dimension of latent space
        mean_network_type: 'mlp', 'lstm', or 'gru' (default: 'mlp')
        use_position_encoding: Whether to use position encoding (default: True for MLP, configurable for RNN)
        pos_dim: Position encoding dimension (default: 64)
        hidden_dims: List of hidden layer dimensions for MLP mean network (default: [128, 128])
        rnn_hidden_dim: Hidden dimension for RNN mean network (default: 32)
        rnn_num_layers: Number of RNN layers (default: 1)
        dropout: Dropout rate for mean network (default: 0.1)
        init_phi: Initial value for φ (0 < phi < 1)
        init_sigma_sq: Initial value for σ²
    """
    def __init__(self, context_dim=12, max_horizon=90, latent_dim=12,
                 mean_network_type='mlp', use_position_encoding=None,
                 pos_dim=64, hidden_dims=None, rnn_hidden_dim=32, rnn_num_layers=1,
                 dropout=0.1, init_phi=0.5, init_sigma_sq=1.0):
        super().__init__()
        self.context_dim = context_dim
        self.max_horizon = max_horizon
        self.latent_dim = latent_dim
        self.mean_network_type = mean_network_type.lower()

        # Default position encoding based on network type
        if use_position_encoding is None:
            use_position_encoding = (self.mean_network_type == 'mlp')

        # Create appropriate mean network
        if self.mean_network_type == 'mlp':
            # Position-encoded MLP mean network
            self.mean_network = PositionEncodedPriorMean(
                context_dim=context_dim,
                max_horizon=max_horizon,
                latent_dim=latent_dim,
                pos_dim=pos_dim,
                hidden_dims=hidden_dims,
                dropout=dropout
            )
        elif self.mean_network_type in ['lstm', 'gru']:
            # RNN-based mean network
            self.mean_network = RNNPriorMean(
                context_dim=context_dim,
                max_horizon=max_horizon,
                latent_dim=latent_dim,
                rnn_type=self.mean_network_type,
                hidden_dim=rnn_hidden_dim,
                num_layers=rnn_num_layers,
                use_position_encoding=use_position_encoding,
                pos_dim=pos_dim,
                dropout=dropout
            )
        else:
            raise ValueError(
                f"mean_network_type must be 'mlp', 'lstm', or 'gru', got '{mean_network_type}'"
            )

        # Learnable AR(1) parameters (only 2 scalars!)
        # Use log parameterization for numerical stability
        self.log_phi = nn.Parameter(torch.tensor(math.log(init_phi / (1 - init_phi))))  # logit(phi)
        self.log_sigma_sq = nn.Parameter(torch.tensor(math.log(init_sigma_sq)))

        # Cache for Cholesky decomposition
        self._cholesky_cache = {}

    def get_phi(self):
        """Get φ parameter (constrained to (0, 1))"""
        return torch.sigmoid(self.log_phi)

    def get_sigma_sq(self):
        """Get σ² parameter (positive)"""
        return torch.exp(self.log_sigma_sq)

    def get_cholesky(self, horizon, device, dtype=None, use_cache=True):
        """
        Get Cholesky factor L with optional caching.

        Cache key: (horizon, phi, sigma_sq, dtype)

        Args:
            horizon: Forecast horizon
            device: Device for tensor
            dtype: Data type (if None, uses default or infers from parameters)
            use_cache: If True, use cached L (for inference). If False, recompute (for training with gradients)
        """
        phi = self.get_phi()
        sigma_sq = self.get_sigma_sq()
        sigma = torch.sqrt(sigma_sq)

        # Infer dtype if not provided
        if dtype is None:
            dtype = torch.get_default_dtype()

        # During training (when gradients needed), always recompute
        if not use_cache or self.training:
            return build_ar1_cholesky_direct(phi, sigma, horizon, device, dtype=dtype)

        # During inference, use cache
        phi_val = phi.item()
        sigma_sq_val = sigma_sq.item()
        cache_key = (horizon, round(phi_val, 6), round(sigma_sq_val, 6), dtype)

        if cache_key not in self._cholesky_cache:
            # Cache with detached values (no gradients)
            L = build_ar1_cholesky_direct(phi_val, math.sqrt(sigma_sq_val), horizon, device, dtype=dtype)
            self._cholesky_cache[cache_key] = L

        return self._cholesky_cache[cache_key]

    def get_prior_params(self, context_summary, horizon):
        """
        Get prior distribution parameters.

        Args:
            context_summary: (B, context_dim)
            horizon: Integer

        Returns:
            mu_p: (B, H, latent_dim) prior means
            Sigma_p: (H, H) prior covariance
        """
        mu_p = self.mean_network(context_summary, horizon)
        phi = self.get_phi()
        sigma_sq = self.get_sigma_sq()
        Sigma_p = build_ar1_covariance(phi, sigma_sq, horizon,
                                       context_summary.device,
                                       dtype=context_summary.dtype)

        return mu_p, Sigma_p

    def sample(self, context_summary, horizon, num_samples=1):
        """
        Sample from prior distribution using Cholesky decomposition.

        z = μ + L @ ε, where ε ~ N(0, I)

        Args:
            context_summary: (B, context_dim)
            horizon: Integer
            num_samples: Number of samples per context (default: 1)

        Returns:
            z: (B, H, latent_dim) if num_samples=1
               (B, num_samples, H, latent_dim) if num_samples>1
        """
        B = context_summary.shape[0]
        device = context_summary.device
        dtype = context_summary.dtype

        # Get prior parameters
        mu_p = self.mean_network(context_summary, horizon)  # (B, H, latent_dim)
        L = self.get_cholesky(horizon, device, dtype=dtype)  # (H, H)

        if num_samples == 1:
            # Sample ε ~ N(0, I) with correct dtype
            eps = torch.randn(B, horizon, self.latent_dim, device=device, dtype=dtype)

            # Apply Cholesky: z_noise = L @ ε (for each latent dim)
            # eps: (B, H, latent_dim)
            # L: (H, H)
            # Want: L @ eps[:,:,d] for each d
            z_noise = torch.einsum('hk,bkd->bhd', L, eps)  # (B, H, latent_dim)

            # z = μ + z_noise
            z = mu_p + z_noise  # (B, H, latent_dim)

            return z
        else:
            # Multiple samples
            samples = []
            for _ in range(num_samples):
                eps = torch.randn(B, horizon, self.latent_dim, device=device, dtype=dtype)
                z_noise = torch.einsum('hk,bkd->bhd', L, eps)
                z = mu_p + z_noise
                samples.append(z)

            z = torch.stack(samples, dim=1)  # (B, num_samples, H, latent_dim)
            return z

    def forward(self, context_summary, horizon, num_samples=1):
        """
        Full forward pass.

        Returns:
            z: Samples from prior
            mu_p: Prior means
            Sigma_p: Prior covariance
        """
        mu_p, Sigma_p = self.get_prior_params(context_summary, horizon)
        z = self.sample(context_summary, horizon, num_samples)
        return z, mu_p, Sigma_p

    def sample_with_quantiles(self, context_summary, horizon, num_samples=1000, quantiles=None):
        """
        Sample from prior and compute empirical quantiles in one call.

        Convenience method that combines sampling with quantile computation.
        Useful for generating confidence intervals without quantile decoder.

        Args:
            context_summary: (B, context_dim)
            horizon: Integer
            num_samples: Number of samples for quantile estimation (default: 1000)
            quantiles: List of quantile values [0, 1]. Default: [0.05, 0.5, 0.95]

        Returns:
            dict with quantile keys ('q05', 'q50', 'q95') containing tensors
            of shape (B, H, latent_dim)

        Example:
            >>> prior = FullCovariancePrior(...)
            >>> quantiles = prior.sample_with_quantiles(context, horizon=30, num_samples=1000)
            >>> median = quantiles['q50']  # (B, 30, latent_dim)
            >>> ci_lower = quantiles['q05']  # 90% CI lower bound
            >>> ci_upper = quantiles['q95']  # 90% CI upper bound
        """
        # Generate samples: (B, num_samples, H, latent_dim)
        samples = self.sample(context_summary, horizon, num_samples=num_samples)

        # Rearrange to (num_samples, B, H, latent_dim) for quantile computation
        samples = samples.transpose(0, 1)  # (num_samples, B, H, latent_dim)

        # Compute empirical quantiles
        return compute_empirical_quantiles(samples, quantiles)
