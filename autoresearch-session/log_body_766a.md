### Context
766a was the required evidence reset after the workflow update. The question was whether to launch future autoresearch from the protected 734a/739a deployable incumbent, the IV-only 755a short-prefix frontier, or a newer model. The key guardrail was tri-scope evidence: IV-only, anchor-only, and native joint must all be evaluated before any framework promotion.

### Execute
- Refreshed 734a/739a evidence under the current suite, including the new IV-EWMA economic-link diagnostic.
- Reran the 755a IV-only short-prefix frontier on validation and train-tail.
- Trained the exact 755a short-prefix recipe on the 734a anchor-only and native joint checkpoints with all-train data, `free_running_fm_weight=0.2`, `free_running_fm_prefix_steps=5`, `rollout_energy=0.2`, and `channel_level_energy=0.05`.
- Excluded one accidental 2048-window anchor transfer artifact from promotion evidence because it did not match the all-train 755a recipe.
- Added `experiments/backfill/block_ar/analyze_766a_evidence_reset.py`.

Artifacts:
- `results/block_ar/766a_evidence_reset/summary.{json,md}`.
- `models/backfill/766a_anchor_shortprefix_fm_k5_w02_e2_alltrain_s7668/best_model.pt`.
- `models/backfill/766a_joint39_shortprefix_fm_k5_w02_e2_alltrain_s7669/best_model.pt`.

### Results
- 734a IV-only refreshed rerun: `5/11`; effective scorecard failures after current policy are `coverage`, `regime_coverage`, `distributional_fidelity`, and `mean_reversion`; IV-EWMA economic link passes.
- 734a native joint IV slice refreshed rerun: `6/11`; IV-EWMA economic link passes, but strict IV failures remain.
- 755a IV-only short-prefix refreshed validation: `7/11`; failures are `coverage`, old `conditionality`, `regime_coverage`, and `distributional_fidelity`; mean reversion and IV-EWMA economic link pass.
- 755a train-tail refreshed rerun: `8/11`; coverage, level KS, median bias, economic link, mean reversion, and pathwise realism pass.
- 766a native joint short-prefix IV slice: `6/11`; mean reversion and economic link pass, but coverage, old conditionality, old cointegration, regime coverage, and distributional fidelity fail. Native joint IV level/median support is weaker than IV-only 755a.
- 734a anchor panel: finite `1.0`, factor delta-KS mean `0.1069`, `12/14` factors pass, q99 tail `14/14` pass, factor corr `0.907`, conditional width rho `0.926`.
- 734a joint panel: finite `1.0`, factor delta-KS mean `0.0971`, `12/14` factors pass, q99 tail `14/14` pass, factor corr `0.863`, IV-factor corr `0.947`.
- 766a anchor panel: finite `1.0`, factor delta-KS mean `0.1090`, `12/14` factors pass, q99 tail `14/14` pass, factor corr `0.903`, conditional width rho `0.928`.
- 766a joint panel: finite `1.0`, factor delta-KS mean `0.0957`, `12/14` factors pass, q99 tail `14/14` pass, factor corr `0.866`, IV-factor corr `0.942`.

### Decision
Do not promote 766a over 734a/739a. The short-prefix recipe is a valid active research ingredient and passes the framework-lock discipline, but it does not fix the sticky AAA/BBB OAS factor delta-KS failure and it regresses native joint IV level support versus IV-only 755a.

Keep 734a/739a as the protected deployable tri-scope incumbent. Use 766a as the active tri-scope research base, with 755a retained as the IV-only frontier reference. The next principled experiment should target the localized blockers without switching away from the AR normalized-innovation flow core: a generic data-derived mixed/no-update output law for sticky low-activity channels, plus a non-regressing IV coverage/level-allocation repair.
