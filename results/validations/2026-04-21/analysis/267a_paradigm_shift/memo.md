# 267a Paradigm Shift Memo

## Context

The strict first-principles `266` reset has now answered its core question.

`266a` to `266d` collectively showed:
- a narrow temporal bottleneck is a reasonable structural bias
- the decoder can support a nontrivial center path
- but asking a prior to generate **deterministic encoded future tokens** from history is the wrong latent modeling assumption

Evidence:
- `266b` reconstruction preserved rank and cointegration much better than its sampled prior
- `266c` sequence-aware diffusion improved sampled temporal moments, but still failed calibrated scenario generation
- `266d` flow matching improved deterministic structure further, but collapsed almost completely to a point forecaster

So the active problem is no longer diffusion versus flow matching.

The active problem is:

**the model needs a probabilistic latent future representation, not a deterministic target code.**

## New Family

**267a-v0: probabilistic latent-token conditional scenario generator**

Core idea:
- keep the same clean encoder-decoder bottleneck doctrine
- but make the latent token path itself probabilistic from the start

Minimal formulation:
- history encoder -> conditional prior over latent token path
- future encoder -> conditional posterior over latent token path
- decoder -> future path from sampled latent tokens plus history context
- training -> standard ELBO / variational conditional latent model
- inference -> sample latent token path from the history-conditioned prior

## Why This Is The Most Principled Next Step

This is a real paradigm shift, but it is still minimal and first-principles:

- no low-rank decoder
- no bounded idio path
- no bounded EC baseline
- no teacher engineering
- no bespoke scenario heads
- no custom loss stack beyond the standard latent-variable objective

And it directly addresses the failure that `266d` made explicit:
- scenario variability must be part of the model specification
- not something a prior is asked to recover from deterministic future codes

## Why Not Other Options

### More `266` prior tuning
Not principled anymore.
That would keep optimizing inside the falsified deterministic-target setup.

### Direct observation-space diffusion / FM reset
Too large a jump before testing the smallest probabilistic latent correction.

### Autoregressive latent token prior
Valid later, but first we need the simpler question answered:
can a plain probabilistic latent bottleneck already recover useful scenario diversity?

## Proposed 267a-v0

Smallest clean version:
- same history encoder family scale as `266`
- same future token-path encoder scale as `266`
- diagonal Gaussian prior over latent token path given history
- diagonal Gaussian posterior over latent token path given history + future
- same decoder family scale as `266`
- standard reconstruction + KL objective

No extra assumptions:
- no low-rank readout
- no side correction paths
- no explicit factor decomposition
- no regime router

## Pre-Registered Question

If scenario diversity is made explicit in the latent-token model itself, does the clean bottleneck family recover:
- nontrivial coverage
- nontrivial regime-sensitive width
- better jump incidence

without destroying the structural gains the bottleneck already learned?

## Kill Criteria

Close `267a-v0` quickly if:
- posterior collapse makes the prior effectively deterministic
- coverage and conditional width remain near zero
- or the model immediately requires special KL tricks, free-bits, or auxiliary heads just to function

## Decision

Proceed with `267a-v0`.

This is the smallest probabilistic paradigm shift that stays aligned with:
- first principles
- methodological elegance
- Bitter Lesson discipline
- and the long-term scenario-generator objective beyond a single asset class.
