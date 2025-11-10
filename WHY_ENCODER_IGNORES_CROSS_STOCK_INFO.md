# Why the Encoder Ignores MSFT/SP500 (Despite Them Being Predictive)

**Core Question:** If MSFT and SP500 help predict AMZN returns (r=0.70 correlation, 76% direction accuracy in linear regression), why doesn't the VAE encoder use them?

**Answer:** The training objective creates a path of least resistance that bypasses cross-stock learning.

---

## The Path of Least Resistance

### Training Scenario: Standard Batches (70% of training)

**Input to encoder:**
```python
input = [
    AMZN[0:t],   # Including AMZN[t] - the target we're predicting!
    MSFT[0:t],   # Correlated with AMZN (r=0.70)
    SP500[0:t]   # Correlated with AMZN (r=0.65)
]
```

The encoder sees the **answer** (AMZN[t]) in the input!

### Two Possible Learning Strategies

**Strategy A: Direct Encoding (What Happened)**
```python
# Encoder learns:
z[t] ≈ compress(AMZN[t])  # Just encode the answer directly!

# Decoder learns:
pred_amzn[t] = decompress(z[t]) ≈ AMZN[t]

# Loss:
pinball_loss(pred_amzn[t], AMZN[t]) → ~0  ✓ Perfect reconstruction!
```

**Strategy B: Cross-Stock Encoding (What You Wanted)**
```python
# Encoder learns:
z[t] ≈ encode_joint([AMZN[t], MSFT[t], SP500[t]])
# And learns: AMZN ≈ β₁·MSFT + β₂·SP500 + residual

# Decoder learns:
pred_amzn[t] = β₁(z)·MSFT[t] + β₂(z)·SP500[t] + decode_residual(z)

# Loss:
pinball_loss(pred_amzn[t], AMZN[t]) → small  ✓ Works, but harder!
```

### Which Strategy Wins?

**Strategy A wins** because:

1. **Simpler optimization landscape:** One-step encoding/decoding vs learning multi-stock relationships
2. **Lower loss:** Perfect reconstruction vs approximate prediction from correlations
3. **Direct solution:** Why learn β₁, β₂ coefficients when you can just memorize AMZN?
4. **Faster convergence:** Gradient descent finds the easy solution first

**Gradient descent is lazy** - it finds the simplest solution that minimizes the loss.

---

## The Information Bottleneck Analogy

Imagine you're packing a suitcase (the latent variable) for a trip (making predictions):

**Available items:**
- Your clothes (AMZN data) - directly relevant to you
- Your friend's clothes (MSFT data) - could borrow if needed
- Stranger's clothes (SP500 data) - could also borrow

**Test question:**
- "Do you have YOUR clothes?" ← This is the loss function

**What will you pack?**

Obviously YOUR clothes! Why waste suitcase space on friend's/stranger's clothes when the test only asks about yours?

**But then the test changes unexpectedly:**
- "Do you have YOUR clothes?" ← But your luggage is lost!

**Now you regret not packing friend's clothes** as backup. But it's too late - you're already at the destination (test time).

**The problem:** The training test (loss function) didn't prepare you for the real test (missing AMZN scenario).

---

## The Gradient Flow Problem

Here's what happens during backpropagation:

```python
# Loss function:
loss = pinball_loss(pred_amzn, target_amzn)  # ONLY measures AMZN error

# Backward pass:
∂loss/∂pred_amzn → ∂pred_amzn/∂decoder → ∂decoder/∂z → ∂z/∂encoder
```

### What the Encoder "Hears" from Gradients

```python
# Strong signal:
∂loss/∂encoder via AMZN channel: LARGE gradient
"Encode AMZN well because loss depends on AMZN prediction!"

# Weak/zero signal:
∂loss/∂encoder via MSFT channel: ZERO gradient
"No signal about whether MSFT encoding is good..."

∂loss/∂encoder via SP500 channel: ZERO gradient
"No signal about whether SP500 encoding is good..."
```

### The Encoder's "Job Description" from Optimizer

From the gradient signals, the encoder learns:

**Strong mandate:**
- ✓ "Encode AMZN accurately"

**No mandate for:**
- ✗ "Encode MSFT" ← No gradient signal
- ✗ "Encode SP500" ← No gradient signal
- ✗ "Learn MSFT→AMZN relationship" ← No gradient signal
- ✗ "Prepare for missing AMZN scenario" ← Never tested

**Result:** Encoder optimizes for what it's tested on (AMZN when available), not what it will face at test time (AMZN missing).

---

## Why Masked Training Didn't Help

### What Happens in Masked Batches (30% of training)

```python
# Input after forward-fill masking:
input_masked = [
    AMZN[t] = AMZN[t-1],  # Forward-filled (stale value)
    MSFT[t] = MSFT[t],    # Real, current value
    SP500[t] = SP500[t]   # Real, current value
]

# Encoder processes:
z[t] = encode([AMZN[t-1], MSFT[t], SP500[t]])  # Sees all three

# But decoder uses z[t-1] (previous timestep):
pred[t] = decode([context_embeddings, z[t-1]])
```

### What the Model Learns from This

**Intended lesson:**
"When AMZN is missing/stale, use MSFT and SP500 to fill the gap"

**Actual lesson:**
"When AMZN is missing/stale, rely on context embeddings (historical AMZN pattern)"

**Why the mismatch?**
- z[t-1] was computed BEFORE seeing MSFT[t] and SP500[t] (LSTM causality)
- Context embeddings contain rich AMZN history
- Easiest solution: extrapolate from historical AMZN pattern
- Harder solution: learn to extract MSFT/SP500 from z[t-1] (but they weren't there!)

The masked training teaches **temporal extrapolation** (use yesterday's AMZN to predict today) not **cross-sectional inference** (use today's MSFT/SP500 to predict today's AMZN).

---

## Why Linear Regression Doesn't Have This Problem

```python
# Linear regression model:
pred_amzn = β₁ · MSFT + β₂ · SP500 + β₀

# Structure FORCES use of MSFT and SP500:
# - No option to ignore them
# - No choice about whether to use them
# - Only question is: what are the β coefficients?
```

**Key difference:**

| Approach | Can avoid MSFT/SP500? |
|----------|----------------------|
| Linear Regression | ✗ Structurally required |
| Current VAE | ✓ Encoder can ignore them |

The VAE's flexibility (its strength) became a weakness when combined with misaligned training objective.

---

## The Training-Test Mismatch

### Training Distribution

**70% of batches (standard):**
```
Input: [AMZN[0:t], MSFT[0:t], SP500[0:t]]  ← AMZN[t] available
Task: Predict AMZN[t]
Optimal strategy: Encode AMZN[t] directly
```

**30% of batches (masked):**
```
Input: [AMZN[0:t-1] (forward-filled), MSFT[0:t], SP500[0:t]]
Task: Predict AMZN[t]
Optimal strategy: Use context embeddings (historical AMZN)
```

### Test Distribution (Realistic Scenario)

```
Input: [AMZN[0:t-1], MSFT[t], SP500[t]]  ← AMZN[t] NOT available
Task: Predict AMZN[t] using MSFT[t], SP500[t]
Required strategy: Extract info from MSFT/SP500

But this strategy was NEVER rewarded during training!
```

**The model was never incentivized to learn MSFT→AMZN or SP500→AMZN relationships because:**
1. In 70% of batches, AMZN[t] is directly available (easier path)
2. In 30% of batches, historical AMZN context works well enough
3. Loss function never tests "Can you predict AMZN from MSFT/SP500 alone?"

---

## Evidence from Investigation

### Correlation Analysis Confirms This

**Oracle scenario (AMZN[t] available in encoding):**
```
Latent z → AMZN:  r = 0.884  ← Strong! Encoder memorizes AMZN
Latent z → MSFT:  r = 0.477  ← Passive byproduct (correlated with AMZN)
Latent z → SP500: r = 0.479  ← Passive byproduct
```

**Realistic scenario (AMZN[t] masked):**
```
Latent z → AMZN:  r = 0.012  ← None! Encoder lost without AMZN
Latent z → MSFT:  r = 0.084  ← Negligible
Latent z → SP500: r = 0.041  ← Negligible
```

### Mutual Information Analysis

```
I(z; AMZN) drops 5.45 → 0.29 when AMZN is masked (95% loss!)
I(z; MSFT) = 0.27 (always weak)
I(z; SP500) = 0.39 (always weak)
```

**Interpretation:** The latent encodes AMZN strongly but MSFT/SP500 weakly, regardless of which scenario. The encoder never learned to prioritize cross-stock information.

### Latent Dimensionality

```
Theoretical: 12 dimensions
Effective: 1 dimension (99% variance in 1st principal component)
```

**What that 1 dimension encodes:** Likely just AMZN return magnitude or market regime, not rich cross-stock relationships.

---

## Why Your Intuition Was Correct

**Your reasoning:** "If MSFT and SP500 help predict AMZN (which they do - r=0.70), then the model should learn to use them."

**Your reasoning is 100% correct!** In an ideal world, the model SHOULD learn this.

**What went wrong:** The loss function didn't align with this goal.

```python
# What you intended:
"Learn to predict AMZN using all available information,
including MSFT and SP500"

# What the loss function said:
"Minimize pinball_loss(pred_amzn, target_amzn)"

# What the model learned:
"Predict AMZN using AMZN (since it's always available in training)"
```

The model optimized your stated objective (minimize loss) but missed your intended objective (learn generalizable cross-stock relationships).

---

## The Core Lesson: Neural Networks Are Literal

**Neural networks optimize exactly what you tell them to optimize, not what you want them to learn.**

```python
# If loss = f(pred_amzn, target_amzn)
# Network learns: "Make pred_amzn close to target_amzn"

# Network does NOT learn:
# - "Use MSFT and SP500"
# - "Prepare for missing data"
# - "Learn generalizable relationships"
# - "Be robust to input variations"

# Unless these are explicitly in the loss!
```

**In machine learning:**
- You don't get what you want
- You don't get what you intend
- **You get what you optimize for**

---

## The Solution: Multi-Task Learning

### Current Loss (Broken)

```python
loss = pinball_loss(pred_amzn, target_amzn)

# Encoder's incentive:
"Encode whatever helps predict AMZN"
# Easiest answer: Encode AMZN itself
```

### Proposed Loss (Fixed)

```python
loss = (
    1.0 * pinball_loss(pred_amzn, target_amzn) +    # Predict AMZN
    0.5 * pinball_loss(pred_msft, target_msft) +    # Also predict MSFT ✓
    0.5 * pinball_loss(pred_sp500, target_sp500) +  # Also predict SP500 ✓
    λ * KL_divergence
)

# Encoder's incentive:
"Encode whatever helps predict AMZN AND MSFT AND SP500"
# Must encode all three stocks!
```

### Why This Fixes the Problem

**Now the encoder receives gradient signals for all stocks:**
```python
∂loss/∂encoder via AMZN: LARGE gradient ← Still optimize AMZN
∂loss/∂encoder via MSFT: MEDIUM gradient ← NEW! Must encode MSFT
∂loss/∂encoder via SP500: MEDIUM gradient ← NEW! Must encode SP500
```

**Consequences:**

1. **Forced encoding:** To predict MSFT, encoder must encode MSFT → it's now in the latent
2. **Forced encoding:** To predict SP500, encoder must encode SP500 → it's now in the latent
3. **Cross-stock learning:** To predict both MSFT and AMZN well, must learn their relationship
4. **Robustness:** When AMZN is masked, MSFT/SP500 information is already in the latent for decoder to use

**Bonus:** The encoder will learn shared structure (market factors, sector effects) because predicting all stocks jointly is more efficient than learning three independent mappings.

---

## Expected Improvements

### After Multi-Task Training

**Oracle scenario (AMZN available):**
- Should stay good (70% direction acc)
- Latent z now encodes joint state, not just AMZN

**Realistic scenario (AMZN masked):**
- Should improve from 49% → 60-65% direction accuracy
- Latent z has MSFT/SP500 information for decoder to use
- Still won't beat linear reg (76%) due to information bottleneck
- But should show the model learned cross-stock relationships

### Diagnostic Check

After retraining with multi-task loss, check:

```python
# Realistic scenario correlations should improve:
Latent z → MSFT: r = 0.012 → hopefully 0.3-0.5
Latent z → SP500: r = 0.041 → hopefully 0.3-0.5

# Mutual information should increase:
I(z; MSFT) = 0.27 → hopefully 1.0-2.0
I(z; SP500) = 0.39 → hopefully 1.0-2.0
```

If correlations don't improve, that suggests architectural issues (information bottleneck too severe).

---

## Implementation: 5-Line Change

In `cvae_1d_with_mem_randomized.py`, change the loss computation:

```python
# CURRENT (line ~46-52):
if self.config.get("target_loss_on_channel_0_only", False):
    target_loss = pinball_loss(
        decoded_target[:, :, :, 0:1],  # Only channel 0
        batch["target"][:, C:, 0:1]
    )

# PROPOSED:
if self.config.get("target_loss_on_channel_0_only", False):
    # Multi-task: predict all stocks
    target_loss = (
        1.0 * pinball_loss(decoded_target[:, :, :, 0:1], batch["target"][:, C:, 0:1]) +    # AMZN
        0.5 * pinball_loss(decoded_target[:, :, :, 4:5], batch["target"][:, C:, 4:5]) +    # MSFT
        0.5 * pinball_loss(decoded_target[:, :, :, 8:9], batch["target"][:, C:, 8:9])      # SP500
    )
```

That's it! The rest of the architecture stays the same.

---

## Alternative Explanation: The Shortcut Problem

This is a well-known phenomenon in deep learning called **shortcut learning** or **clever Hans effect**.

**Classic example:**
- Task: Classify images of cows vs camels
- Dataset: Cows on grass, camels on sand
- Model learns: "If background is green → cow, if brown → sand"
- Test set with cow on sand: Model fails!

**Your case:**
- Task: Predict AMZN returns
- Dataset: AMZN[t] always in input during training
- Model learns: "Encode AMZN[t] → decode to AMZN[t]"
- Test set with AMZN[t] missing: Model fails!

**The fix for both:**
- Cows/camels: Augment with varied backgrounds, force model to look at animal
- Your case: Multi-task loss, force model to encode all stocks

---

## Summary

**Q: Why doesn't the encoder use MSFT/SP500 despite them being predictive?**

**A: Because the loss function never rewarded encoding them.**

1. **Training had a shortcut:** AMZN[t] was always available, so encoder just memorized it
2. **Gradient signal was missing:** Loss only measured AMZN error, giving zero signal about MSFT/SP500 encoding quality
3. **Path of least resistance:** Direct encoding (Strategy A) was easier than learning cross-stock relationships (Strategy B)
4. **Masked training taught wrong lesson:** Taught "use historical AMZN context" not "use MSFT/SP500"

**The fix:** Multi-task loss that forces encoder to encode all stocks, creating gradient signals for MSFT/SP500 and incentivizing cross-stock learning.

**Next step:** Implement the 5-line change and retrain. Expect realistic scenario to improve from 49% → 60-65% direction accuracy as encoder learns to actually use the cross-stock information.
