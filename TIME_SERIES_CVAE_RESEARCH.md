# Time Series CVAE Research & Architecture Design

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Literature Review](#literature-review)
3. [Comparison Table (VAE)](#comparison-table)
4. [Posterior Collapse Prevention](#posterior-collapse-prevention)
5. [Diffusion Models as Alternative](#diffusion-models-as-alternative)
6. [Non-VAE Approaches for Conditional Variance](#non-vae-approaches-for-conditional-variance)
7. [Proposed Architecture](#proposed-architecture)
8. [Key Design Decisions](#key-design-decisions)
9. [Open Questions](#open-questions)
10. [Experimental Findings So Far](#experimental-findings-so-far)
11. [Recommended Next Steps](#recommended-next-steps)
12. [References](#references)

---

## Problem Statement

### The Variance Collapse Issue
Our current CVAEMemRand produces ~0.77% conditional variance (P1) while the data has ~75% P1.

**Root cause:** The decoder's LSTM hidden state h_{C-1} carries so much context information that z becomes redundant. The decoder learns to ignore z and reconstruct from hidden state alone.

```
DECODER LSTM:
t=0..C-1:  h builds up rich context representation
t=C:       h_{C-1} already contains enough info to predict
           → z is ignored → variance collapses
```

### What We Need
A VAE architecture where:
1. z captures meaningful variance about the future
2. Decoder MUST use z (cannot bypass it)
3. Works at inference time without future knowledge

---

## Literature Review

### 1. VRNN - Variational Recurrent Neural Network (2015)
**Paper:** [arXiv:1506.02216](https://arxiv.org/abs/1506.02216)

**Core Innovation:** Conditional prior that depends on RNN hidden state.

```
Prior:      p(z_t | h_{t-1})     # Learned, NOT N(0,1)
Posterior:  q(z_t | x_t, h_{t-1})
Generation: x_t ~ p(x | z_t, h_{t-1})
Hidden:     h_t = f(h_{t-1}, z_t, x_t)
```

**Key Insight:** By making prior depend on h_{t-1}, the prior becomes context-dependent. The model learns what uncertainty is appropriate given the history.

**Limitation:** Prior still tries to match posterior → can still collapse if KL is weighted too high.

---

### 2. SRNN - Stochastic Recurrent Neural Network (2016)
**Paper:** [arXiv:2104.12311](https://arxiv.org/abs/2104.12311) | [GitHub](https://github.com/marcofraccaro/srnn)

**Core Innovation:** Stochastic transition function in RNN.

```
h_t = f(h_{t-1}, z_t, x_t)    # z_t is stochastic latent
z_t ~ p(z_t | h_{t-1})         # prior conditioned on hidden state
```

**Why It Helps:** The stochastic GRU captures variability that deterministic RNNs miss. Consistently outperforms deterministic counterparts on forecasting tasks.

**Results:** Applied to speech, handwriting, astronomical time series.

---

### 3. TC-VAE - Time-Causal VAE (2024)
**Paper:** [arXiv:2411.02947](https://arxiv.org/abs/2411.02947) | [GitHub](https://github.com/justinhou95/TimeCausalVAE)

**Core Innovation:** Causal constraint + learned RealNVP prior.

```
Encoder: Z_t = f(X_{1:t})     # Only sees past
Decoder: Y_t = g(Z_{1:t})     # Only sees past
Prior:   p(z) = RealNVP(N(0,1))  # Learned normalizing flow
```

**Key Features:**
- **Causal constraint:** Output at time t depends ONLY on inputs up to t
- **RealNVP prior:** Flexible learned distribution, not fixed Gaussian
- **Causal Wasserstein distance:** Theoretical guarantee that low loss → good downstream performance

**Results:** Captures stylized facts of financial time series (S&P 500, VIX): heavy tails, volatility clustering, gain/loss asymmetry.

**Relevance:** Most directly applicable to our financial volatility surface problem.

---

### 4. CF-VAE - Conditional Flow VAE (2019)
**Paper:** [arXiv:1908.09008](https://arxiv.org/abs/1908.09008) | [OpenReview](https://openreview.net/forum?id=BklmtJBKDB)

**Core Innovation:** Normalizing flow prior for multi-modal futures.

```
Prior:    p(z|c) = NormalizingFlow(N(0,1), context)
Posterior: q(z|x,c)
```

**Key Insight:** Standard N(0,1) prior is unimodal → can't capture multi-modal futures (e.g., crisis could resolve OR escalate). Flow prior learns flexible conditional distribution.

**Regularization:** Two schemes to prevent posterior collapse during training.

**Results:** State-of-the-art on trajectory prediction (Stanford Drone, HighD).

---

### 5. CLARM - Conditional Latent Autoregressive Recurrent Model (2024)
**Paper:** [arXiv:2403.13858](https://arxiv.org/html/2403.13858)

**Architecture:** Two-step approach.

```
Step 1: CVAE
  - Encoder: X → z (compress to latent)
  - Decoder: z → X (reconstruct)
  - Train until reconstruction stabilizes

Step 2: LSTM
  - Input: z_{1:t}
  - Output: z_{t+1}
  - Train on FROZEN latent space
```

**Key Insight:** Train components separately to avoid "noisy learning curve" of simultaneous training. This prevents coupled collapse.

**Relevance:** Very similar to our proposed approach of training posterior first, then predictor separately.

---

### 6. Conditioned Normalizing Flows for Time Series (2020)
**Paper:** [arXiv:2002.06103](https://arxiv.org/abs/2002.06103)

**Architecture:** Replace VAE entirely with normalizing flows.

```
p(x_t | x_{1:t-1}) = NormalizingFlow conditioned on context
```

**Advantage:** Avoids VAE posterior collapse entirely. Exact likelihood computation.

**Results:** Scales to thousands of interacting time series. State-of-the-art on standard benchmarks.

**Trade-off:** No latent space interpretation. Pure generative model.

---

### 7. PFVAE - Planar Flow VAE (2022)
**Paper:** [MDPI](https://www.mdpi.com/2227-7390/10/4/610)

**Innovation:** Apply normalizing flow to POSTERIOR, not prior.

```
q(z|x) = PlanarFlow(N(μ, σ²))
```

**Why:** Standard Gaussian posterior can't fit complex true posteriors. Flow enables q(z|x) to be arbitrarily complex while maintaining tractable KL.

---

### 8. HyVAE - Hybrid VAE (2023)
**Paper:** [arXiv:2303.07048](https://arxiv.org/pdf/2303.07048)

**Problem:** Standard VAEs fail to jointly learn local patterns (seasonality, trend) AND temporal dynamics.

**Solution:** Integrates both via variational inference. Separate components for:
- Local pattern extraction
- Temporal dynamics modeling

---

### 9. Trajectory Forecasting CVAE
**Repository:** [GitHub](https://github.com/sebasutp/trajectory_forcasting)

**Architecture:**
```
Encoder:  (x_full, x_obs) → (μ, logσ)   # Sees full + partial
Decoder:  (x_obs, z) → x_future          # Conditions on observed only
```

**Conditioning:** Decoder receives [x_obs, z] - observed trajectory + sampled latent.

---

### 10. K²VAE - Koopman-Kalman VAE (2025)
**Paper:** [OpenReview ICML 2025](https://openreview.net/forum?id=71Mm8GDGYd) | [GitHub](https://github.com/decisionintelligence/K2VAE)

**Core Innovation:** Combines Koopman theory with Kalman filtering for long-term probabilistic forecasting.

```
KoopmanNet: Nonlinear time series → Linear dynamical system
KalmanNet: Operates in linear space → Refines predictions + models uncertainty
VAE: Generates probabilistic forecasts with confidence intervals
```

**Key Features:**
- **Koopman linearization:** Transforms nonlinear dynamics to linear form, reducing complexity
- **Kalman filtering:** Reduces error accumulation in long-term forecasting
- **Efficiency:** Maintains performance as horizon extends

**Results:** ICML 2025 Spotlight. Outperforms SOTA on short- and long-term probabilistic forecasting across economics, energy, transportation.

**Relevance:** Directly addresses our error accumulation problem in autoregressive generation.

---

### 11. HeTVAE - Heteroscedastic Temporal VAE (2023)
**Paper:** [arXiv](https://arxiv.org/abs/2107.01429)

**Core Innovation:** Variable uncertainty output for irregular time series.

```
Encoder: Handles irregular sampling with attention mechanism
Decoder: Outputs μ(t) AND σ²(t) - heteroscedastic uncertainty per time point
Uncertainty: Different confidence levels at different interpolation points
```

**Key Features:**
- **Input sparsity handling:** Novel components for irregularly sampled data
- **Uncertainty propagation:** Passes uncertainty through the network
- **Heteroscedastic output:** σ²(t) varies with t, not fixed

**Relevance:** Our decoder could output per-grid-point variance instead of global variance.

---

### 12. VAR-VAE (2025)
**Paper:** [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0020025525003160)

**Core Innovation:** VAE with Vector Autoregression in latent space.

```
Encoder: Noisy time series → Latent space
Latent space: First-lag VAR probabilistic dynamics
Decoder: Latent → Reconstruction
```

**Key Features:**
- **VAR in latent space:** Temporal dynamics modeled probabilistically
- **Noise robustness:** Reduces overfitting at high noise-to-signal ratios
- **Probabilistic forecasting:** Natural uncertainty quantification

**Relevance:** Could replace our LSTM-based latent dynamics with VAR structure.

---

### 13. GARCH-Informed Neural Networks (GINN) (2024)
**Paper:** [arXiv:2410.00288](https://arxiv.org/abs/2410.00288) | [ACM](https://dl.acm.org/doi/10.1145/3677052.3698600)

**Core Innovation:** Physics-informed approach combining GARCH with LSTM.

```
GARCH component: Captures conditional heteroscedasticity
LSTM component: Learns complex temporal dependencies
Loss: MSE + GARCH-based regularization (prevents overfitting)
```

**Key Features:**
- **Hybrid loss:** GARCH serves as regularization in loss function
- **Volatility clustering:** Explicitly models time-varying variance
- **Guard against overfitting:** GARCH prior improves generalization

**Results:** Outperforms pure DL and pure GARCH on R², MSE, MAE.

**Relevance:** Could inform our decoder variance with GARCH-like structure.

---

### 14. VAEneu (2024)
**Paper:** [Applied Intelligence](https://link.springer.com/article/10.1007/s10489-024-06203-5)

**Core Innovation:** CVAE optimized with CRPS instead of likelihood.

```
Loss: CRPS (Continuous Ranked Probability Score)
Advantage: No tractable likelihood required
Output: Sharp, well-calibrated predictive distributions
```

**Key Features:**
- **CRPS optimization:** Learns flexible distributions without parametric form
- **Sharp forecasts:** Avoids blurry predictions
- **Well-calibrated:** Proper uncertainty quantification

**Relevance:** Could replace our MSE+KL loss with CRPS for better calibration.

---

### 15. Scale-VAE (2024)
**Paper:** [ACL Anthology](https://aclanthology.org/2024.lrec-main.1250.pdf)

**Core Innovation:** Prevents posterior collapse by scaling mean dimensions.

```
Standard: z_mean from encoder directly
Scale-VAE: z_mean_scaled = z_mean * scale_factor
Purpose: Keep each dimension discriminative across data instances
```

**Key Insight:** Instead of forcing KL > threshold (free bits), make latent dimensions naturally useful by scaling.

**Relevance:** Alternative to our z_logvar floor for preventing collapse.

---

## Comparison Table

| Model | Prior Type | Conditioning Method | Collapse Prevention |
|-------|------------|---------------------|---------------------|
| **Standard VAE** | N(0,1) fixed | None | None |
| **VRNN** | p(z\|h_{t-1}) learned | RNN hidden state | Prior depends on context |
| **SRNN** | p(z\|h_{t-1}) learned | Stochastic transition | Latent in state transition |
| **TC-VAE** | RealNVP flow | Causal constraint | Flexible prior via flow |
| **CF-VAE** | Normalizing flow | Flow prior | Regularization schemes |
| **CLARM** | N(0,1) | Two-step training | Train CVAE, then LSTM separately |
| **Conditioned NF** | N/A (pure flow) | Context → flow params | No VAE, no collapse |
| **PFVAE** | N(0,1) | Flow on posterior | Rich posterior via flow |
| **K²VAE** | Kalman filter | Koopman linearization | Error accumulation reduction |
| **HeTVAE** | N(0,1) | Attention for irregular | Heteroscedastic output |
| **VAR-VAE** | VAR dynamics | VAR in latent space | Noise robustness |
| **GINN** | GARCH-informed | LSTM + GARCH loss | Regularization |
| **VAEneu** | N(0,1) | CRPS loss | Sharp calibration |
| **Scale-VAE** | N(0,1) | Scaled z_mean | Discriminative dimensions |

---

## Posterior Collapse Prevention

### Taxonomy of Prevention Methods

| Method | Mechanism | Pros | Cons |
|--------|-----------|------|------|
| **KL Annealing** | Gradually increase β from 0→1 | Simple, widely used | May not prevent collapse with strong decoders |
| **Cyclical Annealing** | Periodically reset β: 0→1→0→1... | Better than monotonic | More hyperparameters |
| **Free Bits** | Min KL per dimension | Guarantees KL > 0 | Training instability |
| **z_logvar Floor** | Clamp z_logvar ≥ threshold | Simple, stable | Doesn't address root cause |
| **Scale-VAE** | Scale z_mean for discrimination | Natural usefulness | Less studied |
| **β-VAE** | Fixed β > 1 | Disentanglement | May hurt reconstruction |
| **Delta-VAE** | Prior can't equal posterior | Implicit free bits | Constrained prior choice |

### Theoretical Finding (2023)
From [arXiv:2310.15440](https://arxiv.org/abs/2310.15440):

> When β exceeds a critical threshold, posterior collapse becomes **inevitable regardless of learning period**.

This confirms: z_logvar floor is necessary when using high reconstruction loss weight.

---

## Diffusion Models as Alternative

### Why Consider Diffusion Models?

| Property | VAE | Diffusion |
|----------|-----|-----------|
| **Posterior collapse** | Common problem | No latent → no collapse |
| **Training stability** | Can be unstable | Stable |
| **Uncertainty** | Via z sampling | Via denoising samples |
| **Multi-modal futures** | Requires flow prior | Natural |
| **Computation** | Single forward pass | Many denoising steps |

### DiffLoad (2024)
**Paper:** [IEEE Trans. Power Systems](https://arxiv.org/html/2306.01001)

Separates uncertainty into:
- **Epistemic:** Model uncertainty (diffusion sampling)
- **Aleatoric:** Data uncertainty (Cauchy distribution)

**Relevance:** We could adopt diffusion for the stochastic component while keeping VAE structure for latent representation.

---

## Non-VAE Approaches for Conditional Variance

### 16. CNN-LSTM Hybrid Architectures
**References:** Multiple papers on financial time series forecasting

**Architecture:**
```
CNN layer: Extract local patterns from input windows
  ↓
LSTM layer: Model temporal dependencies
  ↓
Output: Point forecast + optional variance
```

**Key Features:**
- **Feature extraction:** CNN captures local structure (volatility smile shape)
- **Temporal dynamics:** LSTM models time evolution
- **Widely adopted:** Standard in many financial forecasting systems

**Relevance:** Our current architecture already uses this pattern. Consider adding variance output head.

---

### 17. Mixture Density Networks (MDN-RNN)
**Paper:** Bishop (1994) "Mixture Density Networks" | World Models (Ha & Schmidhuber, 2018)

**Core Innovation:** Output is a mixture of Gaussians, not single prediction.

```
RNN output → MDN head → (π_i, μ_i, σ_i) for i=1..K
Final distribution: Σ π_i * N(μ_i, σ_i²)
```

**Key Features:**
- **Multi-modal:** Can represent multiple possible futures
- **Natural variance:** Each component has its own σ
- **Mode selection:** π weights determine which future is likely

**Training:** Negative log-likelihood of mixture

**Relevance:** Directly addresses multi-modal future problem. Could replace single Gaussian output.

---

### 18. Quantile Regression Neural Networks (QRNN)
**Paper:** Taylor (2000) "A Quantile Regression Neural Network Approach to Volatility Forecasting"

**Core Innovation:** Directly predict quantiles instead of distribution parameters.

```
Input → Neural network → (q_0.05, q_0.50, q_0.95)
Loss: Pinball loss (asymmetric) per quantile
```

**Key Features:**
- **Distribution-free:** No Gaussian assumption
- **Direct CI:** Quantiles give confidence intervals directly
- **Asymmetric tails:** Different loss weights for under/over estimation

**Our Implementation:** Already have quantile decoder in codebase (see `experiments/backfill/QUANTILE_REGRESSION.md`)

**Results:** 34-45% CI violations vs target 10%. Better than MSE baseline but still miscalibrated.

---

### 19. Bayesian Neural Networks with MC Dropout
**Paper:** Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation" | Uber's implementation

**Core Innovation:** Use dropout at inference time to approximate Bayesian posterior.

```
Training: Standard dropout
Inference: Keep dropout ON, run N forward passes
Output: Mean and variance from N predictions
```

**Key Features:**
- **Epistemic uncertainty:** Model uncertainty from weight distribution
- **Simple implementation:** Just keep dropout enabled at test time
- **Calibrated uncertainty:** Well-studied theoretical foundations

**Uber's Application:** Time series forecasting with uncertainty quantification

**Relevance:** Simple baseline for uncertainty. Could add to our models without architecture changes.

---

### 20. Deep Switching State Space Models
**Paper:** Dong et al. (2020) "Collapsed Amortized Variational Inference for Switching Nonlinear Dynamical Systems"

**Core Innovation:** Combines neural networks with regime-switching dynamics.

```
Discrete state s_t: Which regime (e.g., calm vs crisis)
Continuous state z_t: Within-regime dynamics
Transition: p(s_t | s_{t-1}, x_{t-1}) - learned regime switching
Dynamics: p(z_t | z_{t-1}, s_t) - regime-dependent evolution
```

**Key Features:**
- **Explicit regimes:** Models market regime changes
- **Conditional dynamics:** Different behavior per regime
- **Interpretable:** Regimes correspond to market states

**Relevance:** Could capture volatility regime shifts (our Exp15 shows 65% persistence at h=30).

---

### 21. MS-GARCH Neural Networks
**References:** Markov-Switching GARCH + Neural Network hybrids

**Architecture:**
```
Hidden Markov Model: Detects volatility regime
GARCH(s_t): Regime-specific volatility dynamics
Neural Network: Enhances regime detection or GARCH parameters
```

**Key Features:**
- **Volatility clustering:** GARCH captures persistence
- **Regime changes:** HMM detects structural breaks
- **Hybrid strength:** NN improves regime detection

**Relevance:** Finance-specific. Combines econometric theory with deep learning.

---

## Comparison Table - Non-VAE Approaches

| Model | Output Type | Uncertainty Source | Multi-modal? | Finance-specific? |
|-------|-------------|-------------------|--------------|-------------------|
| **CNN-LSTM** | Point + optional σ | Learned σ | No | No |
| **MDN-RNN** | Mixture of Gaussians | Component σ's | Yes | No |
| **QRNN** | Quantiles | Implicit in quantiles | Yes (if wide CI) | No |
| **BNN/MC Dropout** | Mean + variance | Weight posterior | No | No |
| **Deep Switching SSM** | Point per regime | Regime + state variance | Yes (regimes) | Adaptable |
| **MS-GARCH NN** | Volatility forecast | GARCH + regime | Yes (regimes) | Yes |

---

## Proposed Architecture

### User's Final Design (Clarified)

**TRAINING:**
```
1. Posterior Encoder: raw[:C+H] → z_post
   - Loss: reconstruction + KL(q(z|x) || N(0,1))
   - Trains independently

2. Context Predictor: raw[:C] → ctx_pred[C:H]
   - Loss: MSE(ctx_pred, ctx_emb[C:H].detach())
   - Predicts future context embeddings from past surfaces

3. Latent Predictor: raw[:C] → (z_pred_μ, z_pred_logσ)
   - Loss: MSE(z_pred_μ, z_post.detach())
   - Predicts posterior latent from past surfaces only

4. Decoder: (ctx_emb || z) → reconstruction
   - During training: uses GT ctx_emb and sampled z_post
```

**INFERENCE:**
```
1. ctx_emb[:C] = ContextEncoder(raw[:C])      # From real data
2. ctx_pred[C:H] = ContextPredictor(raw[:C])  # Predicted
3. (z_μ, z_logσ) = LatentPredictor(raw[:C])   # Predicted
4. z = z_μ + exp(0.5*z_logσ) * ε              # Sample
5. output = Decoder(concat(ctx_emb[:C], ctx_pred[C:H]), z)
```

### Why This Avoids Coupled Collapse

**Key insight:** MSE loss with detached targets.

```
# BAD (causes collapse):
loss = KL(q(z|x) || p(z|c))
# Both q and p adapt → both collapse to delta functions

# GOOD (our approach):
posterior_loss = recon + KL(q(z|x) || N(0,1))  # Independent
predictor_loss = MSE(z_pred, z_post.detach())   # Detached target
# Posterior trained independently
# Predictor just tries to match posterior, no gradient flows back
```

---

## Key Design Decisions

### Decision 1: MSE vs KL for Latent Predictor
**Choice:** MSE on detached posterior samples

**Rationale:**
- KL(z_pred || z_post) causes coupled collapse
- MSE with detachment breaks the gradient coupling
- Predictor becomes a "student" learning from "teacher" posterior

### Decision 2: What Decoder Receives at Prediction Positions
**Choice:** Predicted ctx_emb from ContextPredictor

**Training:** GT ctx_emb (teacher forcing)
**Inference:** Predicted ctx_emb

### Decision 3: Latent Predictor Output
**Choice:** Both μ and logσ (not just μ)

**Rationale:** Need variance for sampling. If only μ, predictions are deterministic.

### Decision 4: Predictors Learn from Raw Surfaces
**Choice:** Both predictors take raw surfaces as input, not encoder embeddings

**Rationale:** More direct learning signal. Avoids dependence on potentially collapsed encoder representations.

---

## Open Questions

### Q1: LSTM Hidden State Problem
The decoder LSTM's h_{C-1} still carries context forward. Even with this architecture, decoder might ignore z.

**Potential solutions:**
- MLP decoder (no hidden state)
- Reset hidden state at position C
- Inject noise into hidden state
- Use attention instead of LSTM

### Q2: How Much Variance Will Predictor Learn?
If posterior variance is small (collapsed), predictor's learned σ will also be small.

**Mitigation:** Ensure posterior maintains variance via:
- Low KL weight
- Minimum variance constraint
- Free bits / KL annealing

### Q3: Context Predictor Necessity
Is ContextPredictor really needed? Could decoder work with:
- Zeros at prediction positions (current approach)
- Learned constant embedding
- Just rely on LSTM hidden state

### Q4: Normalizing Flow Extension
Should we use flows instead of Gaussian for:
- Prior? (like TC-VAE)
- Posterior? (like PFVAE)
- Predictor output?

---

## Experimental Findings So Far

### Exp15: Regime Persistence Analysis
**Question:** How often does context stay in same cluster over time?

**Results (K=10 clusters):**
| Horizon | Persistence |
|---------|-------------|
| h=1 | 97.1% |
| h=7 | 80.8% |
| h=14 | 73.2% |
| h=30 | 65.1% |
| h=60 | 56.4% |
| h=90 | 50.9% |

**Conclusion:** Regimes are moderately sticky. At h=30, ~65% stay in same regime. This supports using context to inform predictions, but transitions are common enough that we can't assume stability.

### Exp12: Context-Free Decoder
**Result:** 35.4% CI coverage (target: 90%)

**Conclusion:** Simply removing context from decoder doesn't fix the problem. The issue is deeper - likely related to how z is being used/ignored.

---

## Recommended Next Steps

1. **Implement TC-VAE style causal constraint** - Ensure encoder/decoder can't see future
2. **Try learned flow prior** - Replace N(0,1) with RealNVP conditioned on context
3. **Implement two-step training** (CLARM style) - Train CVAE first, then predictor
4. **Ablate decoder architecture** - Compare LSTM vs MLP vs Transformer
5. **Add explicit variance regularization** - Ensure posterior doesn't collapse

---

## References

### VAE-Based Models
1. Chung et al. (2015) "A Recurrent Latent Variable Model for Sequential Data" - VRNN
2. Fraccaro et al. (2016) "Sequential Neural Models with Stochastic Layers" - SRNN
3. Acciaio et al. (2024) "Time-Causal VAE: Robust Financial Time Series Generator" - TC-VAE
4. Bhattacharyya et al. (2019) "Conditional Flow VAE for Structured Sequence Prediction" - CF-VAE
5. Rasul et al. (2020) "Multivariate Probabilistic Time Series Forecasting via Conditioned Normalizing Flows"
6. Chen et al. (2025) "A Variational Autoencoder Approach to Conditional Generation of Possible Future Volatility Surfaces" - Our baseline
7. K²VAE (2025) "Koopman-Kalman VAE for Long-Term Probabilistic Forecasting" - ICML 2025 Spotlight
8. HeTVAE (2023) "Heteroscedastic Temporal VAE for Irregular Time Series" - arXiv:2107.01429
9. VAR-VAE (2025) "VAE with Vector Autoregression in Latent Space" - ScienceDirect
10. GINN (2024) "GARCH-Informed Neural Networks" - arXiv:2410.00288
11. VAEneu (2024) "VAE with CRPS Loss" - Applied Intelligence
12. Scale-VAE (2024) "Preventing Posterior Collapse by Scaling Mean Dimensions" - ACL Anthology

### Posterior Collapse & Theory
13. arXiv:2310.15440 (2023) "When β exceeds critical threshold, posterior collapse is inevitable"

### Diffusion Models
14. DiffLoad (2024) "Diffusion for Time Series with Epistemic/Aleatoric Uncertainty" - IEEE Trans. Power Systems

### Non-VAE Approaches
15. Bishop (1994) "Mixture Density Networks"
16. Ha & Schmidhuber (2018) "World Models" - MDN-RNN
17. Taylor (2000) "A Quantile Regression Neural Network Approach to Volatility Forecasting"
18. Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation"
19. Dong et al. (2020) "Collapsed Amortized Variational Inference for Switching Nonlinear Dynamical Systems"
