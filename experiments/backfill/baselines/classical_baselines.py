"""Classical baselines for IV surface scenario generation.

All baselines implement the same interface as SinglePassBlockAR:
    model.sample(history, n_samples=50) -> (B, n_samples, T_fut, 5, 5) in [0, 1]

Baselines:
    1. RandomWalk — persist last surface + scaled Gaussian noise
    2. HistoricalSimulation — sample matching windows from training data
    3. UnconditionalBootstrap — resample daily changes (preserving cross-cell), cumsum
    4. PCAVARBaseline — PCA on surfaces, VAR(1) on factors, bootstrap residuals
    5. GARCHCCCBaseline — EWMA variance per cell + constant conditional correlation
    6. FilteredHistoricalSimulation — EWMA variance + bootstrap standardized residuals
"""

import numpy as np
import torch
from scipy import stats as sp_stats


def denormalize_iv(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1]"""
    return (x + 1.0) / 2.0


class BaselineModel:
    """Mixin providing eval()/train() no-ops for compatibility with test suites."""

    def eval(self):
        return self

    def train(self, mode=True):
        return self

    def sample_batched(self, *args, **kwargs):
        return self.sample(*args, **kwargs)


# =============================================================================
# 1. Random Walk
# =============================================================================

class RandomWalkBaseline(BaselineModel):
    """Persist last observed surface + Gaussian noise scaled by historical vol.

    The simplest possible baseline. Each ensemble member evolves the last
    observed surface as a random walk with iid Gaussian increments whose
    scale matches the empirical daily change std per cell.
    """

    def __init__(self, surfaces: np.ndarray, future_len: int = 30):
        self.future_len = future_len
        daily_changes = np.diff(surfaces, axis=0)  # (N-1, 5, 5)
        self.cell_std = daily_changes.std(axis=0)  # (5, 5)

    def sample(
        self, history: torch.Tensor, n_samples: int = 50, **kwargs
    ) -> torch.Tensor:
        history_denorm = denormalize_iv(history)  # (B, T, 5, 5)
        B = history_denorm.shape[0]
        last_surface = history_denorm[:, -1]  # (B, 5, 5)
        device = history.device

        cell_std = torch.tensor(
            self.cell_std, dtype=history.dtype, device=device
        )  # (5, 5)

        noise = torch.randn(
            B, n_samples, self.future_len, 5, 5,
            dtype=history.dtype, device=device,
        )
        increments = noise * cell_std[None, None, None, :, :]
        cum_increments = increments.cumsum(dim=2)

        samples = last_surface[:, None, None, :, :] + cum_increments
        return samples.clamp(0.0, 1.0)


# =============================================================================
# 2. Historical Simulation (conditional on mean IV + vol-of-vol)
# =============================================================================

class HistoricalSimulation(BaselineModel):
    """Sample actual historical 30-day windows as scenarios.

    For each test history, matches training windows by (mean IV, vol-of-vol,
    term structure slope) and samples subsequent futures weighted by similarity.
    This is the standard Filtered Historical Simulation approach used in
    risk management (without the GARCH filter).
    """

    def __init__(
        self,
        surfaces: np.ndarray,
        history_len: int = 30,
        future_len: int = 30,
    ):
        self.history_len = history_len
        self.future_len = future_len

        n_windows = len(surfaces) - history_len - future_len + 1
        self.hist_windows = np.zeros(
            (n_windows, history_len, 5, 5), dtype=np.float32
        )
        self.fut_windows = np.zeros(
            (n_windows, future_len, 5, 5), dtype=np.float32
        )

        for i in range(n_windows):
            self.hist_windows[i] = surfaces[i : i + history_len]
            self.fut_windows[i] = surfaces[
                i + history_len : i + history_len + future_len
            ]

        # Multi-feature matching: (mean_iv, vol_of_vol, slope)
        # mean IV of last 5 days
        mean_iv = self.hist_windows[:, -5:].mean(axis=(1, 2, 3))  # (n,)
        # vol-of-vol: std of daily mean-IV changes
        mean_iv_ts = self.hist_windows.mean(axis=(2, 3))  # (n, H)
        daily_chg = np.diff(mean_iv_ts, axis=1)  # (n, H-1)
        vov = daily_chg.std(axis=1)  # (n,)
        # term structure slope: long tenor - short tenor (last day)
        last_day = self.hist_windows[:, -1]  # (n, 5, 5)
        slope = last_day[:, -1, 2] - last_day[:, 0, 2]  # (n,)  long-short ATM

        self.features = np.column_stack([mean_iv, vov, slope])  # (n, 3)
        # Normalize features for distance computation
        self.feat_mean = self.features.mean(axis=0)
        self.feat_std = self.features.std(axis=0) + 1e-8
        self.features_norm = (self.features - self.feat_mean) / self.feat_std

    def _compute_features(self, history_np: np.ndarray) -> np.ndarray:
        """Compute matching features from a single history window. (5,5) x T"""
        mean_iv = history_np[-5:].mean()
        mean_iv_ts = history_np.mean(axis=(1, 2))  # (T,)
        daily_chg = np.diff(mean_iv_ts)
        vov = daily_chg.std() if len(daily_chg) > 1 else 0.0
        slope = history_np[-1, -1, 2] - history_np[-1, 0, 2]
        feat = np.array([mean_iv, vov, slope])
        return (feat - self.feat_mean) / self.feat_std

    def sample(
        self, history: torch.Tensor, n_samples: int = 50, **kwargs
    ) -> torch.Tensor:
        history_denorm = denormalize_iv(history).cpu().numpy()
        B = history_denorm.shape[0]

        all_samples = np.zeros(
            (B, n_samples, self.future_len, 5, 5), dtype=np.float32
        )

        for b in range(B):
            query = self._compute_features(history_denorm[b])
            distances = np.linalg.norm(
                self.features_norm - query[None, :], axis=1
            )
            # Softmax weighting (temperature = 1.0)
            weights = np.exp(-distances)
            weights /= weights.sum()

            indices = np.random.choice(
                len(self.fut_windows), size=n_samples, replace=True, p=weights
            )
            all_samples[b] = self.fut_windows[indices]

        samples = torch.tensor(
            all_samples, dtype=history.dtype, device=history.device
        )
        return samples.clamp(0.0, 1.0)


# =============================================================================
# 3. Unconditional Bootstrap (preserves cross-cell correlation per day)
# =============================================================================

class UnconditionalBootstrap(BaselineModel):
    """Resample daily IV changes from training data, cumsum to build paths.

    Each draw is a full (5,5) daily change, preserving cross-cell correlation
    within a day. Draws are iid across time (breaks temporal structure).
    This measures the value of conditioning — gap vs our model = conditionality.
    """

    def __init__(self, surfaces: np.ndarray, future_len: int = 30):
        self.future_len = future_len
        self.daily_changes = np.diff(surfaces, axis=0).astype(np.float32)

    def sample(
        self, history: torch.Tensor, n_samples: int = 50, **kwargs
    ) -> torch.Tensor:
        history_denorm = denormalize_iv(history)
        B = history_denorm.shape[0]
        last_surface = history_denorm[:, -1].cpu().numpy()  # (B, 5, 5)

        n_changes = len(self.daily_changes)
        # Vectorized: draw all indices at once
        idx = np.random.randint(
            0, n_changes, size=(B, n_samples, self.future_len)
        )
        # Gather changes: (B, n_samples, T_fut, 5, 5)
        changes = self.daily_changes[idx]
        # Cumsum from last surface
        cum_changes = np.cumsum(changes, axis=2)
        all_samples = last_surface[:, None, None, :, :] + cum_changes

        samples = torch.tensor(
            all_samples, dtype=history.dtype, device=history.device
        )
        return samples.clamp(0.0, 1.0)


# =============================================================================
# 4. PCA + VAR(1) with bootstrapped residuals
# =============================================================================

class PCAVARBaseline(BaselineModel):
    """PCA on surfaces + VAR(1) on principal components + simulate forward.

    Standard quantitative finance approach:
    1. Flatten 5x5 surfaces to 25-dim vectors
    2. Fit PCA to reduce to n_components factors
    3. Fit VAR(1) on factor time series: z_t = A @ z_{t-1} + intercept + eps
    4. At inference: project history to PCA space, simulate VAR, reconstruct.
       Bootstrap actual residuals for ensemble diversity (preserves fat tails
       in factor space).
    """

    def __init__(
        self,
        surfaces: np.ndarray,
        future_len: int = 30,
        n_components: int = 5,
    ):
        self.future_len = future_len
        self.n_components = n_components

        flat = surfaces.reshape(-1, 25).astype(np.float64)
        self.mean = flat.mean(axis=0)
        centered = flat - self.mean

        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        self.components = Vt[:n_components]  # (K, 25)
        self.explained_var = (S[:n_components] ** 2) / (len(flat) - 1)

        factors = centered @ self.components.T  # (N, K)

        # Fit VAR(1): z_t = A @ z_{t-1} + intercept + eps
        Z_prev = factors[:-1]
        Z_next = factors[1:]
        Z_aug = np.column_stack([Z_prev, np.ones(len(Z_prev))])
        coeffs, _, _, _ = np.linalg.lstsq(Z_aug, Z_next, rcond=None)
        self.A = coeffs[:n_components].T  # (K, K)
        self.intercept = coeffs[n_components]  # (K,)

        Z_pred = Z_prev @ self.A.T + self.intercept
        self.residuals = (Z_next - Z_pred).astype(np.float32)

    def sample(
        self, history: torch.Tensor, n_samples: int = 50, **kwargs
    ) -> torch.Tensor:
        history_denorm = denormalize_iv(history).cpu().numpy()
        B = history_denorm.shape[:1][0]

        all_samples = np.zeros(
            (B, n_samples, self.future_len, 5, 5), dtype=np.float32
        )
        n_resid = len(self.residuals)

        for b in range(B):
            last_flat = history_denorm[b, -1].reshape(1, 25)
            z_init = ((last_flat - self.mean) @ self.components.T)[0]

            for s in range(n_samples):
                z = z_init.copy()
                path = np.zeros((self.future_len, 25), dtype=np.float64)

                for t in range(self.future_len):
                    resid_idx = np.random.randint(0, n_resid)
                    z = self.A @ z + self.intercept + self.residuals[resid_idx]
                    path[t] = z @ self.components + self.mean

                all_samples[b, s] = path.reshape(self.future_len, 5, 5)

        samples = torch.tensor(
            all_samples, dtype=history.dtype, device=history.device
        )
        return samples.clamp(0.0, 1.0)


# =============================================================================
# 5. GARCH(1,1) + CCC (Constant Conditional Correlation)
# =============================================================================

class GARCHCCCBaseline(BaselineModel):
    """GARCH(1,1) per cell + Constant Conditional Correlation + Student-t.

    Industry standard for multivariate volatility modeling (Bollerslev 1990).
    Uses the `arch` package for proper GARCH(1,1) MLE fitting per cell:
        σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}

    Cross-cell correlation from standardized residuals (CCC).
    Student-t innovations with per-cell estimated degrees of freedom.
    """

    def __init__(
        self,
        surfaces: np.ndarray,
        future_len: int = 30,
    ):
        from arch import arch_model

        self.future_len = future_len

        daily_changes = np.diff(surfaces, axis=0)  # (N-1, 5, 5)
        N = daily_changes.shape[0]
        self.cell_mean = daily_changes.mean(axis=0)  # (5, 5)

        # Fit GARCH(1,1) with Student-t per cell
        self.omega = np.zeros((5, 5))
        self.alpha = np.zeros((5, 5))
        self.beta = np.zeros((5, 5))
        self.df = np.zeros((5, 5))
        cond_vol_last = np.zeros((5, 5))  # for diagnostics

        std_residuals_all = np.zeros((N, 5, 5))

        print(f"    Fitting 25 GARCH(1,1) models...", end="", flush=True)
        for i in range(5):
            for j in range(5):
                series = (daily_changes[:, i, j] - self.cell_mean[i, j]) * 100
                am = arch_model(
                    series, vol="GARCH", p=1, q=1,
                    dist="StudentsT", mean="Zero", rescale=False,
                )
                res = am.fit(disp="off", show_warning=False)

                self.omega[i, j] = res.params["omega"] / 1e4  # undo *100 scaling
                self.alpha[i, j] = res.params["alpha[1]"]
                self.beta[i, j] = res.params["beta[1]"]
                self.df[i, j] = max(res.params["nu"], 2.1)  # df > 2 for finite var

                # Store conditional volatility and standardized residuals
                cond_vol = res.conditional_volatility / 100  # undo scaling
                cond_vol_last[i, j] = cond_vol[-1]
                resid = daily_changes[:, i, j] - self.cell_mean[i, j]
                std_residuals_all[:, i, j] = resid / np.maximum(cond_vol, 1e-10)

        print(" done")

        # CCC: correlation of standardized residuals
        std_flat = std_residuals_all.reshape(N, 25)
        self.corr_matrix = np.corrcoef(std_flat.T)
        # Ensure PSD
        eigvals, eigvecs = np.linalg.eigh(self.corr_matrix)
        eigvals = np.maximum(eigvals, 1e-8)
        self.corr_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
        self.cholesky_L = np.linalg.cholesky(self.corr_matrix)

        # Pooled df for correlation-preserving simulation
        self.pooled_df = float(np.median(self.df))

        # Store unconditional variance for initialization
        self.uncond_var = daily_changes.var(axis=0)

    def _garch_var_from_history(self, history_denorm: np.ndarray) -> np.ndarray:
        """Roll GARCH(1,1) forward through history to get conditional variance."""
        changes = np.diff(history_denorm, axis=0)
        residuals = changes - self.cell_mean

        # Initialize with unconditional variance
        var = self.uncond_var.copy()
        for t in range(len(residuals)):
            var = self.omega + self.alpha * residuals[t] ** 2 + self.beta * var
            var = np.maximum(var, 1e-12)
        return var

    def sample(
        self, history: torch.Tensor, n_samples: int = 50, **kwargs
    ) -> torch.Tensor:
        history_denorm = denormalize_iv(history).cpu().numpy()
        B = history_denorm.shape[0]

        all_samples = np.zeros(
            (B, n_samples, self.future_len, 5, 5), dtype=np.float32
        )

        for b in range(B):
            last_surface = history_denorm[b, -1]
            cond_var = self._garch_var_from_history(history_denorm[b])

            for s in range(n_samples):
                surface = last_surface.copy()
                var = cond_var.copy()

                for t in range(self.future_len):
                    # Correlated Student-t innovations
                    z_iid = np.random.standard_t(self.pooled_df, size=25)
                    z_iid = z_iid / np.sqrt(self.pooled_df / (self.pooled_df - 2))
                    z_corr = self.cholesky_L @ z_iid
                    z_grid = z_corr.reshape(5, 5)

                    std = np.sqrt(var)
                    change = self.cell_mean + std * z_grid
                    surface = surface + change

                    # GARCH variance update
                    resid = change - self.cell_mean
                    var = self.omega + self.alpha * resid ** 2 + self.beta * var
                    var = np.maximum(var, 1e-12)

                    all_samples[b, s, t] = surface

        samples = torch.tensor(
            all_samples, dtype=history.dtype, device=history.device
        )
        return samples.clamp(0.0, 1.0)


# =============================================================================
# 6. Filtered Historical Simulation (EWMA + bootstrap actual residuals)
# =============================================================================

class FilteredHistoricalSimulation(BaselineModel):
    """EWMA variance filter + bootstrap of standardized residuals + CCC.

    The gold standard in financial risk management (Barone-Adesi et al. 1999):
    1. Fit EWMA(λ=0.94) per cell on training data → standardized residuals
    2. At inference: initialize EWMA from history window
    3. Draw standardized residuals from the training-set pool (with replacement)
    4. Re-scale by current conditional std → preserves fat tails AND vol clustering

    Cross-cell correlation is preserved automatically because each draw is a
    full 25-dim standardized residual vector from the same training day.
    """

    def __init__(
        self,
        surfaces: np.ndarray,
        future_len: int = 30,
        ewma_lambda: float = 0.94,
    ):
        self.future_len = future_len
        self.ewma_lambda = ewma_lambda

        daily_changes = np.diff(surfaces, axis=0)  # (N-1, 5, 5)
        N = daily_changes.shape[0]

        self.cell_mean = daily_changes.mean(axis=0)
        residuals = daily_changes - self.cell_mean

        # Compute EWMA variance series
        lam = ewma_lambda
        var_series = np.zeros((N, 5, 5), dtype=np.float64)
        var_series[0] = residuals[0] ** 2
        for t in range(1, N):
            var_series[t] = lam * var_series[t - 1] + (1 - lam) * residuals[t - 1] ** 2
        var_series = np.maximum(var_series, 1e-12)

        # Store standardized residuals for bootstrapping
        std_series = np.sqrt(var_series)
        self.std_residuals = (residuals / std_series).astype(np.float32)  # (N, 5, 5)

    def _ewma_from_history(self, history_denorm: np.ndarray) -> np.ndarray:
        """Compute EWMA conditional variance from history window."""
        changes = np.diff(history_denorm, axis=0)
        residuals = changes - self.cell_mean
        lam = self.ewma_lambda

        var = residuals[0] ** 2
        for t in range(1, len(residuals)):
            var = lam * var + (1 - lam) * residuals[t - 1] ** 2
        var = lam * var + (1 - lam) * residuals[-1] ** 2
        return np.maximum(var, 1e-12)

    def sample(
        self, history: torch.Tensor, n_samples: int = 50, **kwargs
    ) -> torch.Tensor:
        history_denorm = denormalize_iv(history).cpu().numpy()
        B = history_denorm.shape[0]

        n_resid = len(self.std_residuals)
        all_samples = np.zeros(
            (B, n_samples, self.future_len, 5, 5), dtype=np.float32
        )

        for b in range(B):
            last_surface = history_denorm[b, -1]
            cond_var = self._ewma_from_history(history_denorm[b])

            for s in range(n_samples):
                surface = last_surface.copy()
                var = cond_var.copy()

                for t in range(self.future_len):
                    # Bootstrap a standardized residual (full 5x5, preserves correlation)
                    idx = np.random.randint(0, n_resid)
                    z = self.std_residuals[idx]  # (5, 5)

                    # Re-scale by current conditional std
                    std = np.sqrt(var)
                    change = self.cell_mean + std * z
                    surface = surface + change

                    # Update EWMA
                    resid = change - self.cell_mean
                    var = self.ewma_lambda * var + (1 - self.ewma_lambda) * resid ** 2
                    var = np.maximum(var, 1e-12)

                    all_samples[b, s, t] = surface

        samples = torch.tensor(
            all_samples, dtype=history.dtype, device=history.device
        )
        return samples.clamp(0.0, 1.0)
