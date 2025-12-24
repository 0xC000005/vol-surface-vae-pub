# Fix Multi-Horizon Training to Use Fixed Context Length

**Status: DEFERRED - Documented for future reference**
**Date: 2025-12-19**
**Related Model: V3 Conditional Prior VAE (context60_latent12_v3)**

---

## Problem

The current multi-horizon training implementation uses **variable context length** per horizon (`C = T - h`), which is:
1. **Methodologically incorrect** - predictions should use same context regardless of horizon
2. **Computationally inefficient** - can't cache ctx_encoder/prior_network (run 6x instead of 1x)
3. **Inconsistent with deployment** - at inference, you have fixed context and predict multiple horizons

## Solution

Use **fixed context length** from config (`context_len = 60`) for all horizons. This enables caching ctx_encoder and prior_network, giving ~2-3x speedup.

## Design

```
BEFORE (Variable Context - WRONG):
h=1:  Context=[1-179] → Predict [180]      (C=179)
h=90: Context=[1-90]  → Predict [91-180]   (C=90)

AFTER (Fixed Context - CORRECT):
h=1:  Context=[1-60] → Predict [61]        (C=60, fixed)
h=90: Context=[1-60] → Predict [61-150]    (C=60, fixed)
```

## Files to Modify

### 1. `vae/cvae_conditional_prior.py`

#### Change 1: `train_step_multihorizon()` (Lines 320-456)

**Move ctx_encoder and prior_network OUTSIDE the horizon loop:**

```python
# BEFORE (inside loop - lines 335-347):
for h in horizons:
    C = T - h  # Variable!
    ctx_input = {"surface": surface[:, :C, :, :]}
    ctx_embedding = self.ctx_encoder(ctx_input)  # 6x
    prior_mean, prior_logvar = self.prior_network(ctx_input)  # 6x

# AFTER (outside loop):
C = self.config.get("context_len")  # Fixed!
ctx_input = {"surface": surface[:, :C, :, :]}
if has_ex_feats:
    ctx_input["ex_feats"] = ex_feats[:, :C, :]
ctx_embedding = self.ctx_encoder(ctx_input)  # 1x (cached)
prior_mean, prior_logvar = self.prior_network(ctx_input)  # 1x (cached)

for h in horizons:
    # Only decoder varies per horizon
    target_surface = surface[:, C:C+h, :, :]  # Variable target length
```

**Update target extraction (line 366):**
```python
# BEFORE:
surface_real = surface[:, C:, :, :]  # All remaining days

# AFTER:
surface_real = surface[:, C:C+h, :, :]  # Only h days after context
```

**Update reconstruction slicing (line 367):**
```python
# BEFORE:
re_surface = self.quantile_loss_fn(surface_reconstruction[:, C:, :, :, :], surface_real)

# AFTER:
re_surface = self.quantile_loss_fn(surface_reconstruction[:, C:C+h, :, :, :], surface_real)
```

**Update KL loss slicing (lines 383-386):**
```python
# KL computed over fixed context length (same for all horizons)
kl_loss = kl_divergence_gaussians(
    z_mean_full[:, :C, :], z_logvar_full[:, :C, :],
    prior_mean, prior_logvar
)
```

#### Change 2: `train_step()` (Line 228)
```python
# BEFORE:
C = T - self.horizon

# AFTER:
C = self.config.get("context_len")
```

#### Change 3: `test_step()` (Line 468)
```python
# BEFORE:
C = T - self.horizon

# AFTER:
C = self.config.get("context_len")
```

#### Change 4: `compute_kl_loss()` (Line 90)
```python
# BEFORE:
C = T - self.horizon

# AFTER:
C = self.config.get("context_len")
```

### 2. Config Validation (No changes needed)

The config already defines `context_len = 60` at line 68 of:
`config/backfill_context60_config_latent12_v3_conditional_prior.py`

Phase 2 sequence length `(150, 180)` satisfies: `T >= C + max(horizons)` → `150 >= 60 + 90` ✓

## Expected Performance Impact

| Component | Before | After | Speedup |
|-----------|--------|-------|---------|
| ctx_encoder | 6x/batch | 1x/batch | 6x |
| prior_network | 6x/batch | 1x/batch | 6x |
| decoder | 6x/batch | 6x/batch | 1x |
| **Overall** | 1.8 it/s | ~4-5 it/s | **~2-3x** |

## Implementation Order

1. Modify `train_step_multihorizon()` - main fix with caching
2. Update `train_step()` - consistency for Phase 1
3. Update `test_step()` - consistency for validation
4. Update `compute_kl_loss()` - used by all above methods
5. Test Phase 2 training speed improvement

## Risks

- Need to ensure sequence length validation: `T >= C + max(horizons)`
- Legacy path in `train_step_multihorizon()` (lines 392-456) also needs same fix

---

## When to Apply This Fix

Apply this fix when:
1. You want to speed up Phase 2 training (~2-3x improvement)
2. You need methodologically correct multi-horizon evaluation (fair comparison across horizons)
3. You're preparing for production deployment (inference uses fixed context)

## Quick Summary

**Current behavior**: `C = T - h` (variable context per horizon)
- h=1 gets 179 days context, h=90 gets 90 days context
- ctx_encoder and prior_network run 6x per batch
- Phase 2 speed: ~1.8 it/s

**Fixed behavior**: `C = config.context_len` (fixed context for all horizons)
- All horizons get same context (e.g., 60 days)
- ctx_encoder and prior_network run 1x per batch (cached)
- Expected Phase 2 speed: ~4-5 it/s
