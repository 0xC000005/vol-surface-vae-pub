# Portfolio-Risk Response Label Intake

## Context

The product conditionality audit and start-aware readout gate show that the
narrative channel reaches support selection and decoded prefixes, but final
factor fan separation remains bootstrap-limited for several starts. A risk
manager may still care more about portfolio loss, VaR/ES, drawdown, and
contribution views than about whether each individual factor fan visibly
changes.

This intake tests a product-level response target before adding another model
component.

## Method Story

Use the existing component-preserving support-mixture generator and convert each
generated path into several normalized portfolio-risk books:

- equity beta / carry;
- credit + duration;
- dollar-liquidity carry;
- commodity inflation;
- safe-haven hedge;
- short volatility.

For each narrative and fixed start, compute response labels such as terminal
median, signed VaR95 loss, signed ES95 loss, and max path loss. Then compare
cross-narrative differences against same-narrative repeat, within-run bootstrap,
and start-only controls.

This does not replace the simple support mixture. It defines a more
risk-manager-legible response surface that can later be used to train or test a
support reweighting policy.

## TestFlight Result

Script:
`experiments/backfill/block_ar/nl_portfolio_risk_response_label_audit.py`.

Artifact:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_risk_response_label_audit_920a/portfolio_risk_response_label_audit.json`.

Result:

- overall status: `warning`;
- recommendation: `candidate_portfolio_labels_need_tail_noise_fix`;
- portfolio books: `6`;
- pass books: `3`;
- warning books: `3`.

Passing books:

- equity beta / carry: path/repeat `3.431`, path/bootstrap `1.491`,
  VaR95/repeat `2.083`, ES95/repeat `1.751`;
- dollar-liquidity carry: path/repeat `2.944`, path/bootstrap `1.276`,
  VaR95/repeat `5.734`, ES95/repeat `2.215`;
- short volatility: path/repeat `3.937`, path/bootstrap `1.342`,
  VaR95/repeat `1.700`, ES95/repeat `1.729`.

Warning books:

- credit + duration: path metrics pass, but VaR95/repeat is `0.868`;
- commodity inflation: path metrics are only modest and VaR/ES tails are below
  repeat controls;
- safe-haven hedge: path and VaR95 metrics pass, but ES95/repeat is `0.971`.

## Interpretation

This is better aligned with the product question than a single generic
portfolio audit: some risk books show clear narrative response above controls,
while others reveal where tail noise still limits a clean production claim. It
does not solve conditionality completely, but it gives a sharper target for the
next support-policy or readout experiment.

## Next Experiment

Run a portfolio-risk-aware support-policy TestFlight using only deployable
information:

1. keep the same narrative/start-compatible support pool;
2. add a candidate utility based on the portfolio response books that passed the
   diagnostic;
3. compare against equal/simple support mixture on held-out scenario quality and
   fixed-start conditionality;
4. reject the policy if it regresses energy/CRPS materially or improves only one
   hand-picked book.

## Kill Conditions

- The policy improves display metrics but weakens held-out CRPS/energy/coverage.
- The policy improves one exposure book only by overfitting the start-18
  casebook.
- Portfolio response remains below repeat/bootstrap controls after the support
  utility is applied.
- The method becomes a pile of factor-specific hand rules instead of a compact
  response-label layer.
