# Bitter Lesson Guard — Enforcement Rules

## Principle

Everything must be LEARNED from data. The method must generalize to any conditional
scenario generation problem — not just implied volatility surfaces. Tomorrow this
could be rates, FX, credit spreads, or commodities. Any fix that depends on knowing
the data is a 5x5 IV grid is a dead end.

## Pre-Commit Check

Before committing any code change, scan the diff for these violations:

### 1. Per-Cell Constants
**Reject** any value indexed by grid position that isn't an nn.Parameter.

```python
# REJECT — hardcoded per-cell value
scale = torch.tensor([[0.5, 0.3, ...], ...])  # 5x5 constant
correction[i, j] = precomputed_value

# ACCEPT — learned per-cell parameter
self.cell_scale = nn.Parameter(torch.ones(H, W))
self.cell_spread_linear = nn.Linear(cond_dim, H * W)
```

### 2. Data-Derived Lookup Tables
**Reject** any precomputed table from training data statistics.

```python
# REJECT
quantile_map = np.load("quantile_map.npz")
gt_cell_std = np.array([0.18, 0.04, ...])  # from data
per_cell_bias = compute_from_training_data(dataset)

# ACCEPT — computed online from each batch during training
gt_changes = gt_iv[:, 1:] - gt_iv[:, :-1]
gt_cell_var = gt_changes.var(dim=(0, 1))  # online, per-batch
```

### 3. Domain Heuristics
**Reject** any rule specific to implied volatility surfaces.

```python
# REJECT
if smile_is_convex(surface): ...
assert term_structure_slopes_down(surface)
leverage_correction = -0.81 * returns  # domain-specific constant

# ACCEPT — general mathematical structure
loss = energy_score(samples, target)  # proper scoring rule, works for any data
delta = self.frame_decoder(prev, condition, noise)  # learned transformation
```

### 4. Post-Hoc Corrections
**Reject** any transformation applied after generation to fix output quality.

```python
# REJECT
samples = quantile_mapper.apply(samples)  # post-hoc marginal fix
samples = conformal_calibrate(samples, calibration_set)
samples = rescale_per_cell(samples, target_std)

# ACCEPT — correction built into the model/loss
cell_var_loss = ((gen_var - gt_var) / gt_var).pow(2).mean()  # training signal
energy_score_loss = ...  # shapes distribution during training
```

### 5. Magic Numbers
**Reject** any constant that wouldn't transfer to a different dataset.

```python
# REJECT
vol_scale_correction = 0.0187  # SPX-specific global mean vol
n_factors = 5  # because IV grid is 5x5
rho_leverage = -0.81  # SPX leverage effect

# ACCEPT — hyperparameters (architecture choices)
hidden_dim = 128  # works for any data
noise_dim = 32  # works for any data
rho = 0.8  # AR(1) noise correlation, not data-derived
```

## Gray Areas

Some things are borderline. Use this test: **would this work unchanged if the input
were a 10x10 rates grid instead of a 5x5 IV grid?**

- `vol_scale` formula using history statistics → BORDERLINE (uses data but computes
  online, doesn't hardcode values). Currently accepted because it's a general
  normalization strategy.
- `floor_clamp = 0.01` → ACCEPTED (IV can't be negative; any asset has physical bounds)
- `reflecting boundaries` → ACCEPTED (general technique, not IV-specific)
- `tanh bounding on skip` → ACCEPTED (general stability technique)

## What to Do When a Violation is Found

1. Do NOT commit the change. Discard working directory changes (`git checkout -- .`)
2. Log: "Rejected (Bitter Lesson): [specific violation]" in results-log.md
3. Reformulate: how can the SAME effect be achieved through learned parameters?
   - Per-cell constant → learned per-cell parameter (nn.Parameter or nn.Linear)
   - Lookup table → online computation from batch statistics
   - Domain heuristic → general loss function
   - Post-hoc correction → training-time loss term
4. Queue the reformulated version as the next hypothesis
