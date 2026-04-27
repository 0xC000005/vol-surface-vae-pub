# 657a IV Versus Anchor-Factor Failure Diagnostics

## Question

Why do IV-only mechanisms work better on implied-volatility surfaces than the same broad family works on the IV plus anchor-factor panel?

This is a root-cause audit, not a fix proposal.

## Evidence Reviewed

- IV-only and joint results from 510a/564a, 537a/572e, 641a, 647a, and 652a-655a.
- Wide-panel framing audit from 651a.
- Native joint quality audits from 627a-style `joint_panel.json` files.
- Local data diagnostics saved to `results/autoresearch/657a_iv_vs_anchor_factor_diagnostics/diagnostics.json`.

## Ruled Out

### The wide panel is not grossly misaligned

651a already verified the key framing facts:

- source panel shape is `5822 x 51`;
- modeled state panel has `38` channels: `25` IV cells plus `13` anchor-factor states;
- the remaining `13` source columns are reference return/diff diagnostics, not duplicated generated targets;
- future states reconstruct exactly from modeled encoded increments for 30/60/90/152-day horizons;
- validation/test leakage was not detected.

So the failure is not that IV and anchor factors are from unrelated dates or cannot be represented as one daily market panel.

### Native joint generation is not completely failing

The clean native joint models do generate usable anchor-factor movement statistics:

- 641a: factor delta KS mean `0.0975`, q99 pass `13/13`, factor-factor corr `0.866`, IV-factor corr shape `0.852`.
- 647a: factor delta KS mean `0.0868`, factor KS pass `13/13`, IV-factor corr shape `0.845`.
- 652a: factor delta KS mean `0.0849`, q99 pass `13/13`, factor-factor corr `0.842`, IV-factor corr shape `0.846`.

The issue is not "anchor factors cannot be sampled at all." The issue is that adding anchor factors changes the learned conditional path law enough that IV deployability falls, and the joint story remains weaker than a true calibrated joint law.

## Confirmed Causes

### 1. IV surfaces have strong repeated geometry; anchor factors do not

IV has a dense `5 x 5` surface with smooth cross-cell structure. Daily IV increment absolute cross-cell correlation is high:

- IV mean/median/max absolute daily-increment correlation: `0.382 / 0.316 / 0.966`.

Anchor factors are a heterogeneous list of rates, spreads, FX, commodities, equity indices, and credit:

- factor mean/median/max absolute daily-increment correlation: `0.155 / 0.133 / 0.759`.

That means IV gives the model repeated local structure and redundancy. The factor list is closer to a small set of heterogeneous macro tokens with weaker common geometry.

### 2. Factor levels are much more nonstationary than factor increments

Train-vs-validation KS drift using the same 2048-window training frame:

- future IV level KS mean/median/max: `0.534 / 0.565 / 0.758`, pass `<0.20`: `1/25`.
- future factor level KS mean/median/max: `0.650 / 0.615 / 0.862`, pass `<0.20`: `0/13`.
- future IV increment KS mean/median/max: `0.084 / 0.081 / 0.171`, pass `<0.20`: `25/25`.
- future factor increment KS mean/median/max: `0.093 / 0.090 / 0.174`, pass `<0.20`: `13/13`.

This is a decisive framing clue: level distributions drift badly for both IV and factors, but factor levels drift even more. Increment/move distributions are comparatively stable. Any method or test that asks for unconditional level occupancy across long regimes is much harsher on anchor factors.

### 3. Factor support and data quality are heterogeneous

The factor panel required cleaning:

- copper: `167` stale/nonpositive values;
- wheat: `135`;
- Nikkei: `1`;
- gold: `167`;
- crude oil needed `diff_level` fallback because it has negative values.

Some rate/spread series are stale or discretized:

- AAA OAS zero daily-change rate: `0.440`;
- BBB OAS zero daily-change rate: `0.321`;
- US2Y zero daily-change rate: `0.187`;
- US10Y zero daily-change rate: `0.088`.

IV cells are bounded dense observations. Anchor factors mix positive log-level assets, signed crude oil, rates/spreads in difference coordinates, stale quoted series, and different economic units. This makes one homogeneous treatment weaker.

### 4. Generated coordinate choice has a real tradeoff

The experiment record shows the coordinate issue clearly:

- 631a encoded-increment generation preserved anchor-factor behavior well but caused IV level drift/explosions.
- 638a level-score generation improved IV support/tails but damaged factor movement scale and co-movement.
- 641a mixed coordinates recovered anchor-factor realism while staying native joint, but IV remained only `4/11`.

This is not arbitrary architecture thrashing. It is evidence that IV-like mean-reverting bounded surfaces and random-walk-like anchor factors prefer different generated coordinates. The mixed-coordinate rule is justified as data-coordinate selection, but it does not by itself solve conditional IV path calibration.

### 5. The joint objective reallocates modeling capacity and loss mass

For the 652-style path-flow training target:

- IV target mixed-coordinate mean absolute value: `0.427`; mean square: `0.441`.
- factor target mixed-coordinate mean absolute value: `0.762`; mean square: `0.932`.

Even though there are fewer factor state channels (`13`) than IV channels (`25`), the factor increment target has about twice the per-element squared scale in training coordinates. The joint flow objective can therefore spend substantial capacity fitting factor movement while degrading IV calibration.

This matches the observed result:

- 652a IV-only validation flow loss: `0.928`; IV suite `5/11`.
- 652a joint38 validation flow loss: `1.054`; IV suite `4/11`.
- 652a joint38 factor audit remains strong.

### 6. The native joint models preserve correlation shape better than correlation magnitude

652a joint38:

- factor-factor correlation shape: `0.842`;
- factor-factor generated mean absolute correlation: `0.177` versus GT `0.225`;
- IV-factor correlation shape: `0.846`;
- IV-factor generated mean absolute correlation: `0.056` versus GT `0.149`.

The model learns broad directional structure, but the generated IV-factor co-movement amplitude is attenuated. This matters for story-telling: scenarios may have the right sign pattern on average but not enough per-scenario joint shock strength.

### 7. Post-hoc decks are operationally useful but scientifically weak

573a/574a/656a improve stress-deck usability, but all share the same methodological weakness:

- IV scenario and factor scenario are generated by different mechanisms;
- pairing is imposed after generation;
- empirical-copula pairing can preserve mixed-rank rates, but it still does not make the factor path born from the same conditional stochastic mechanism as the IV path.

So composed decks are useful diagnostics or policy overlays, not proof of a generalizable conditional joint generator.

## Plausible Additional Causes

These are not fully proven yet, but are consistent with the evidence:

- The factor list is too small and heterogeneous for surface-style inductive bias; token/channel modeling may need stronger variable identity and unit metadata.
- The available history may contain weak conditional signal for 30-day anchor-factor moves, even if unconditional movement statistics are learnable.
- Single-realized-future flow matching can learn average transport and broad correlations while underrepresenting conditional tail co-occurrence.
- The validation period is a different macro regime from much of the training data, making factor levels especially hard to calibrate without explicit regime conditioning or broader data.
- Current audits check factor marginal moves and correlation matrices, but not enough per-scenario conditional economic narratives such as equity-down/credit-wider/rates-lower clusters.

## Bottom Line

The failure is not one simple bug. The main confirmed mechanism is:

1. IV is a dense, bounded, redundant surface with strong geometry.
2. Anchor factors are sparse, heterogeneous, nonstationary macro variables with mixed supports and weaker shared geometry.
3. Increment/move distributions are stable, but level distributions drift heavily, especially for factors.
4. Native joint models can learn factor movement realism, but doing so under the current flow objective and capacity budget weakens IV path calibration.
5. Post-hoc composition can make a risk deck, but it does not satisfy the scientific requirement that all scenarios come from one conditional joint mechanism.

The principled next diagnostic should split model error by group inside the native joint objective: IV loss, factor loss, IV-factor co-movement amplitude, and scenario-level economic shock clusters. Do not add another pairing/gluing layer as the core path.

