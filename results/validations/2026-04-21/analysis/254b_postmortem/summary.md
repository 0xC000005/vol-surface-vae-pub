## 2026-04-21: 254b Postmortem

### Headline

- `254b` scored `3/11` on the common full 11-suite.
- Passes: `surface`, `block_ar`, `cointegration`
- It did **not** recover the `4/11` frontier.

### Comparison vs 254a-v0

| model | n_pass | corr_ratio | rank_ratio | MR ratio | h30 MR | change KS | level KS | max-jump KS | ACF corr | cointegration ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 254a-v0 | 3 | 2.233 | 0.201 | 2.724 | 0.799 | 0 | 4 | 0.943 | 0.883 | 6.177 |
| 254b | 3 | 2.232 | 0.201 | 3.270 | 0.847 | 1 | 1 | 0.938 | 0.886 | 6.260 |

### Mechanism Read

The anti-collapse penalties worked on the **loading matrix statistics**, but not on the
generated panel dynamics.

Loading-side stats on validation windows:

- `dynamic_loading_eff_rank = 3.74`
- `dynamic_loading_top1_share = 0.459`
- `base_loading_eff_rank = 2.96`
- `base_loading_top1_share = 0.599`

So the explicit `Lambda_t` collapse was reduced.

But output-side stats were still dominated by the common path:

- `common_share_rms = 0.936`
- `idio_share_rms = 0.031`
- `ec_share_rms = 0.261`
- `fast_gate_mean = 0.227`
- longest-tenor `idio/common <= 0.053x`

And the evaluator still saw:

- `corr_ratio = 2.232`
- `rank_ratio = 0.201`
- `Gen PC1 variance = 97.5%`
- `MR ratio = 3.270`
- `change KS = 1/25`
- `max-jump KS = 0.938`

Important extra finding:

- the latent factor path itself was **not** collapsed
  - `common_latent_top1_share = 0.176`
  - `common_latent_entropy_rank = 7.23`

So the failure is stronger than “bad regularization”.

### Conclusion

`254b` shows that even when loading-rank statistics are pushed into a healthier regime,
the `254` family still maps the resulting latent dynamics into an over-shared panel path.

That means the common-mode collapse is not just a superficial loading-matrix problem.
It is likely a deeper family-level issue in the way:

- dual-timescale backbone,
- low-rank readout,
- and integrated deterministic objective

combine into the final panel dynamics.

### Decision

Treat `254b` as negative evidence for continuing to tune the `254` family.

Most principled next step:

- switch to a **paradigm-shift ideation** iteration rather than `254c`
- specifically, move to a family where the output geometry is not forced through the
  same low-rank common-path attractor that survived both `254a` and `254b`
