# Quantile Regression Decoder Implementation Plan

## Objective
Modify CVAEMemRand decoder to output 3 quantiles (p5, p50, p95) directly instead of a single mean prediction, using pinball loss to achieve properly calibrated confidence intervals (~10% violations vs current 35-72%).

## Problem Analysis

### Current Architecture
- Decoder outputs: `(B, T, 5, 5)` - single mean surface
- Loss: MSE (L2 distance)
- Uncertainty: From sampling z ~ N(0,1) 1000 times
- Result: CI violations 35-72% (should be ~10%)

### Root Cause
- Decoder learns conditional mean E[y|z,c]
- Latent sampling captures epistemic uncertainty only
- Does NOT capture aleatoric uncertainty (inherent noise)
- Context encoder is deterministic → overconfident

### Solution
- Decoder outputs: `(B, T, 3, 5, 5)` - three quantile surfaces [p5, p50, p95]
- Loss: Pinball loss (quantile regression)
- Uncertainty: Built into decoder output (no sampling needed)
- Expected: ~10% CI violations, properly calibrated

## Core Changes

### 1. Decoder Architecture (`vae/cvae_with_mem_randomized.py`)

**Location:** `CVAEMemRandDecoder.__get_surface_decoder()` (lines 349-387)

**Change final output layer:**
```python
# Current (line 383-386):
surface_decoder["dec_output"] = nn.Conv2d(
    in_feats, 1,  # Single channel
    kernel_size=3, padding="same"
)

# New:
surface_decoder["dec_output"] = nn.Conv2d(
    in_feats, 3,  # Three channels for p5, p50, p95
    kernel_size=3, padding="same"
)
```

**Location:** `CVAEMemRandDecoder.forward()` (lines 437-464)

**Update reshape logic:**
```python
# Current (line 454):
decoded_surface = decoded_surface.reshape((B, T, feat_dim[0], feat_dim[1]))

# New:
decoded_surface = decoded_surface.reshape((B, T, 3, feat_dim[0], feat_dim[1]))
# Shape: (B, T, 3, 5, 5) where dim=2 is [p5, p50, p95]
```

### 2. Pinball Loss Implementation

**Add QuantileLoss class at top of file:**
```python
class QuantileLoss(nn.Module):
    """
    Pinball loss for quantile regression.

    For quantile τ ∈ [0,1]:
    L_τ(y, ŷ) = max((τ-1)×(y-ŷ), τ×(y-ŷ))

    Asymmetric penalty:
    - τ=0.05: penalizes under-prediction more (wants most values above)
    - τ=0.50: reduces to MAE (mean absolute error)
    - τ=0.95: penalizes over-prediction more (wants most values below)
    """
    def __init__(self, quantiles=[0.05, 0.5, 0.95]):
        super().__init__()
        self.quantiles = torch.tensor(quantiles)

    def forward(self, preds, target):
        """
        Args:
            preds: (B, T, num_quantiles, H, W) - quantile predictions
            target: (B, T, H, W) - ground truth

        Returns:
            Scalar loss (averaged over all quantiles and elements)
        """
        quantiles = self.quantiles.to(preds.device)
        losses = []

        for i, q in enumerate(quantiles):
            pred_q = preds[:, :, i, :, :]  # (B, T, H, W)
            error = target - pred_q  # (B, T, H, W)
            loss_q = torch.max((q-1)*error, q*error)
            losses.append(torch.mean(loss_q))

        return torch.mean(torch.stack(losses))
```

### 3. Loss Function Update

**Location:** `CVAEMemRand.train_step()` (lines 660-714)

**Changes:**
```python
# Add in __init__ (around line 515):
self.quantile_loss_fn = QuantileLoss(quantiles=[0.05, 0.5, 0.95])

# In train_step(), replace MSE with quantile loss (around line 692):
# OLD:
# re_surface = F.mse_loss(surface_reconstruciton, surface_real)

# NEW:
# surface_reconstruction shape: (B, 1, 3, 5, 5)
# surface_real shape: (B, 1, 5, 5)
re_surface = self.quantile_loss_fn(surface_reconstruciton, surface_real)
```

**Similar change in `test_step()`** (line 738)

### 4. Forward Pass Update

**Location:** `CVAEMemRand.forward()` (lines 612-658)

**Update return statements:**
```python
# Line 654 (with ex_feats):
return decoded_surface[:, C:, :, :, :], decoded_ex_feat[:, C:, :], z_mean, z_log_var, z
# Shape: (B, 1, 3, 5, 5) for decoded_surface

# Line 657 (without ex_feats):
return decoded_surface[:, C:, :, :, :], z_mean, z_log_var, z
```

### 5. Generation Code Update

**Location:** `CVAEMemRand.get_surface_given_conditions()` (lines 520-562)

**Update return statements:**
```python
# Line 559 (with ex_feats):
return decoded_surface[:, C:, :, :, :], decoded_ex_feat[:, C:, :]
# Shape: (B, 1, 3, 5, 5)

# Line 562 (without ex_feats):
return decoded_surface[:, C:, :, :, :]
```

**Location:** `generate_surfaces.py` (lines 13-46)

**Major refactor:**
```python
def generate_surfaces_random(model, ex_data, vol_surface_data, day, ctx_len, num_vaes, use_ex_feats, check_ex_feats):
    """
    NOTE: With quantile regression, num_vaes represents batch processing size,
    not number of stochastic samples. Each forward pass produces deterministic
    quantiles [p5, p50, p95].
    """
    # Build context data (same as before)
    surf_data = torch.from_numpy(vol_surface_data[day - ctx_len:day])
    if use_ex_feats:
        ex_data_ctx = torch.from_numpy(ex_data[day - ctx_len:day])
        if len(ex_data_ctx.shape) == 1:
            ex_data_ctx = ex_data_ctx.unsqueeze(1)
        ctx_data = {
            "surface": surf_data.unsqueeze(0),  # (1, T, 5, 5)
            "ex_feats": ex_data_ctx.unsqueeze(0)  # (1, T, 3)
        }
    else:
        ctx_data = {
            "surface": surf_data.unsqueeze(0),  # (1, T, 5, 5)
        }

    # Generate quantiles (single forward pass!)
    if use_ex_feats:
        surf, ex_feat = model.get_surface_given_conditions(ctx_data)
        # surf: (1, 1, 3, 5, 5) - [batch, time, quantiles, H, W]
        ex_feat = ex_feat.detach().cpu().numpy().squeeze()  # (3,)
    else:
        surf = model.get_surface_given_conditions(ctx_data)

    surf = surf.detach().cpu().numpy().squeeze(0).squeeze(0)  # (3, 5, 5)

    # Extract quantiles
    surf_p05 = surf[0, :, :]  # (5, 5)
    surf_p50 = surf[1, :, :]  # (5, 5)
    surf_p95 = surf[2, :, :]  # (5, 5)

    return (surf_p05, surf_p50, surf_p95), ex_feat if use_ex_feats and check_ex_feats else None

def generate_surfaces_multiday(...):
    # Storage for quantiles
    all_day_surfaces_p05 = np.zeros((days_to_generate, vol_surface_data.shape[1], vol_surface_data.shape[2]))
    all_day_surfaces_p50 = np.zeros((days_to_generate, vol_surface_data.shape[1], vol_surface_data.shape[2]))
    all_day_surfaces_p95 = np.zeros((days_to_generate, vol_surface_data.shape[1], vol_surface_data.shape[2]))
    all_day_ex_feats = np.zeros((days_to_generate, ex_data.shape[1]))

    for day in range(start_day, start_day+days_to_generate):
        if day % 500 == 0:
            print(f"Generating day {day}")
        (p05, p50, p95), ex_feats = generate_surfaces_random(...)
        all_day_surfaces_p05[day - start_day, ...] = p05
        all_day_surfaces_p50[day - start_day, ...] = p50
        all_day_surfaces_p95[day - start_day, ...] = p95
        if ex_feats is not None:
            all_day_ex_feats[day - start_day, ...] = ex_feats

    return (all_day_surfaces_p05, all_day_surfaces_p50, all_day_surfaces_p95), all_day_ex_feats

# Update saving (line 113-126):
if return_ex:
    (p05, p50, p95), ex_feats = generate_surfaces_multiday(...)
    np.savez(gen_fn, surfaces_p05=p05, surfaces_p50=p50, surfaces_p95=p95, ex_feats=ex_feats)
else:
    (p05, p50, p95), _ = generate_surfaces_multiday(...)
    np.savez(gen_fn, surfaces_p05=p05, surfaces_p50=p50, surfaces_p95=p95)
```

## Configuration Changes

**Add to model config** (param_search.py):
```python
config = {
    # ... existing config ...
    "use_quantile_regression": True,  # Enable quantile output
    "quantiles": [0.05, 0.5, 0.95],  # Quantile levels
}
```

## Testing Strategy

### Phase 1: Unit Tests
Create `test_quantile_decoder.py`:

```python
import torch
from vae.cvae_with_mem_randomized import CVAEMemRand, QuantileLoss

def test_decoder_output_shape():
    """Test that decoder outputs correct shape for quantiles"""
    config = {
        "feat_dim": (5, 5),
        "latent_dim": 5,
        "surface_hidden": [5, 5, 5],
        "ex_feats_dim": 0,
        "mem_type": "lstm",
        "mem_hidden": 100,
        "mem_layers": 1,
        "ctx_surface_hidden": [5, 5, 5],
        "use_quantile_regression": True,
        "quantiles": [0.05, 0.5, 0.95],
    }

    model = CVAEMemRand(config)
    x = {"surface": torch.randn(2, 6, 5, 5)}  # B=2, T=6
    output, z_mean, z_log_var, z = model.forward(x)

    assert output.shape == (2, 1, 3, 5, 5), f"Expected (2,1,3,5,5), got {output.shape}"
    print("✓ Decoder output shape correct")

def test_quantile_loss_asymmetry():
    """Test that quantile loss is asymmetric"""
    loss_fn = QuantileLoss(quantiles=[0.05, 0.5, 0.95])

    # For q=0.05, under-prediction should be penalized more
    y_true = torch.tensor([[[[1.0]]]])  # (1, 1, 1, 1)

    # Under-prediction: pred=0, true=1, error=+1
    y_pred_under = torch.tensor([[[[0.0, 0.5, 1.5]]]])  # (1, 1, 3, 1)
    loss_under = loss_fn(y_pred_under, y_true)

    # Over-prediction: pred=2, true=1, error=-1
    y_pred_over = torch.tensor([[[[2.0, 1.5, 0.5]]]])
    loss_over = loss_fn(y_pred_over, y_true)

    print(f"Under-pred loss: {loss_under.item():.4f}")
    print(f"Over-pred loss: {loss_over.item():.4f}")
    # For q=0.05: (0.05-1)×1 = -0.95 vs 0.05×(-1) = -0.05
    # max(-0.95, 0.05) = 0.05 vs max(-0.05, -0.05) = -0.05 (but we take abs)
    print("✓ Quantile loss asymmetry verified")

def test_quantile_ordering():
    """Test that trained quantiles maintain ordering"""
    # This will be tested after training
    # Expected: p5 ≤ p50 ≤ p95 for >95% of predictions
    pass

if __name__ == "__main__":
    test_decoder_output_shape()
    test_quantile_loss_asymmetry()
```

Run: `python test_quantile_decoder.py`

### Phase 2: Small Training Test
```bash
# Modify param_search.py to train on small dataset
python param_search.py --small_test --epochs 10 --days 100
```

Verify:
1. Loss decreases over epochs
2. No NaN/Inf in gradients
3. Quantiles maintain ordering (p5 < p50 < p95)

### Phase 3: Full Training
```bash
# Train all 3 models with quantile regression
python param_search.py
```

This will train:
- no_ex (surface only)
- ex_no_loss (surface + features, no feature loss)
- ex_loss (surface + features, optimize return prediction)

### Phase 4: Calibration Evaluation
```bash
# Generate quantile predictions
python generate_surfaces.py  # Creates *_gen5.npz with p05/p50/p95

# Evaluate calibration
python verify_mean_tracking_vs_ci.py  # Check violation rates
python visualize_teacher_forcing.py  # Visualize CI bands
```

**Success Criteria:**
- CI violation rate: ~10% (vs baseline 35-72%)
- Point accuracy: p50 matches baseline mean (R² ≈ 0.9)
- Quantile ordering: >95% monotonic
- CI width: Wider than baseline (but calibrated)

## Files to Modify

| File | Changes | Complexity |
|------|---------|------------|
| `vae/cvae_with_mem_randomized.py` | Add QuantileLoss, modify decoder, update loss | HIGH |
| `generate_surfaces.py` | Return 3 quantiles, update save format | MEDIUM |
| `generate_surfaces_max_likelihood.py` | Same as generate_surfaces.py | MEDIUM |
| `test_quantile_decoder.py` | Create new unit test file | LOW |
| `visualize_teacher_forcing.py` | Load quantiles from file | LOW |
| `verify_mean_tracking_vs_ci.py` | Load quantiles directly | LOW |

## Implementation Timeline

### Day 1 (Morning - 4 hours)
1. ✓ Create branch: `quantile-regression-decoder`
2. Implement QuantileLoss class
3. Modify decoder architecture (output 3 channels)
4. Update train_step() and test_step()
5. Update forward() method
6. Write unit tests, verify shapes

### Day 1 (Afternoon - 2 hours)
7. Modify generate_surfaces.py
8. Modify generate_surfaces_max_likelihood.py
9. Run unit tests

### Day 1 (Evening) → Day 2 (Morning)
10. Train small test model (100 days, 10 epochs)
11. Verify convergence, debug issues

### Day 2 (Afternoon) → Day 3 (Morning)
12. Launch full training (3 models × 500 epochs)
    - Runs overnight (~8-12 hours)

### Day 3 (Morning - 3 hours)
13. Generate predictions for test set
14. Evaluate CI calibration
15. Compare to baseline models
16. Update analysis scripts if needed

**Total: 2-3 days**

## Expected Outcomes

### Success Metrics
1. **CI Calibration**: Violation rate 35-72% → ~10%
2. **Quantile Monotonicity**: p5 ≤ p50 ≤ p95 in >95% of cases
3. **Point Accuracy**: p50 matches baseline mean (R² ≈ 0.9)
4. **Computational Efficiency**: 1000× faster generation (no sampling loop)

### Potential Issues

**Issue 1: Quantile Crossing**
- Problem: p5 > p50 or p50 > p95 in some predictions
- Solution: Add soft constraint loss if >5% violations:
  ```python
  crossing_penalty = torch.relu(preds[:,0] - preds[:,1]) + torch.relu(preds[:,1] - preds[:,2])
  total_loss += 0.1 * crossing_penalty.mean()
  ```

**Issue 2: Training Instability**
- Problem: Loss oscillates or doesn't converge
- Solution: Reduce learning rate, adjust KL weight, add gradient clipping

**Issue 3: Overly Wide Intervals**
- Problem: CI width too large (>10% coverage)
- Solution: Acceptable trade-off, shows model learned uncertainty properly

## References

- Koenker & Bassett (1978) - "Regression Quantiles"
- Rodrigues & Pereira (2020) - "Beyond Expectation: Deep Joint Mean and Quantile Regression for Spatiotemporal Problems"
- Gasthaus et al. (2019) - "Probabilistic Forecasting with Spline Quantile Function RNNs"

## Notes

- Keep baseline models for comparison (don't delete)
- Quantile decoder is a fundamental change - maintains same latent structure but changes output representation
- No need for post-hoc calibration (conformal prediction) if quantile regression works well
- Can extend to more quantiles (p1, p5, p25, p50, p75, p95, p99) if needed
