# Part 2: Conditional Flow Decoder

Purpose: turn the learned JEPA world-model state into conditional multi-day
scenario paths.

Expected decoder form:

```text
z_past, predicted future latents, tokens, base noise
    -> conditional flow decoder
    -> future scenario path
```

Use flow matching as the primary decoder family. Fixed Gaussian or Student-t
mixture heads are not the intended method for this track because they are too
restrictive for the conditional, high-dimensional scenario distribution.

Quality gates for this part should be distribution-focused:

- CRPS or pinball/quantile loss,
- Energy Score and Variogram Score,
- empirical coverage by horizon/cell,
- sample variance ratio and pairwise scenario distance,
- correlation matrix and PCA spectrum error,
- tail and stress-regime metrics,
- context-sensitivity or conditionality buckets.

This part should not be allowed to mask a weak Part 1 representation. Report
decoder quality alongside frozen-latent and collapse diagnostics.
