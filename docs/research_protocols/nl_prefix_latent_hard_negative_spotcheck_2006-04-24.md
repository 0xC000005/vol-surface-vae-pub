# NL Prefix-Latent Narrative Spot Check: Paired Positives and Hard Negatives

Generated: 2026-06-03

This review packet compiles existing generated artifacts only. No new narratives were generated for this document.

## How To Read This

Each section below is one style pair: the positive narrative for the selected historical episode appears first, followed immediately by the corresponding hard negative in the same style. The hard negative describes a different historical prefix whose market directions conflict with the selected episode.

The `negative mechanical summary` is not a second generated training narrative. It is the structured factor baseline for the negative historical prefix, kept as evidence/provenance so a reviewer can see why the authored hard-negative text should be contradictory.

## Selected Positive Episode

- Target window: `joint39_train_1553`
- Observed prefix date range: 2006-03-13 to 2006-04-24
- Scenario title: weak-dollar commodity bid with fading credit stress
- Archetype: `weak_dollar_commodity_repricing_with_limited_credit_stress` (medium confidence)
- Narrative authoring: `direct_codex_multiformat`
- Positive retrieval validity: `True`

Positive mechanical evidence:

> Mechanical baseline: SPX higher small; VIX flat; BBB_OAS flat; AAA_OAS wider small; DXY lower medium; USDJPY higher large; CRUDE_OIL higher medium; US2Y higher small; US10Y higher small; GOLD higher medium.

Evidence fields:

- 2006-03-13 to 2006-04-24 prefix.
- DXY -3.09, z=-1.176326; USDJPY +0.962, z=3.618730.
- CRUDE_OIL +8.38, z=1.039959; GOLD +74.50, z=1.278610.
- US10Y +0.22 and US2Y +0.15, both small higher.
- AAA_OAS widened small/medium: +552.89, z=0.570700; BBB_OAS flat.

## Corpus Status Used For This Packet

- Positive source: `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_full_codex_multiformat_982g_sharded/final/multiformat_episode_cards.jsonl`
- Support-date metadata source: `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_narrative_support_bank_cards_all_970f/episode_narrative_support_cards.jsonl`
- Hard-negative source: `experiments/backfill/block_ar/nl_scenario_demo_outputs/hard_negative_bank_regeneration_985c_codex_batch/hard_negative_bank.jsonl`
- Positive corpus report: `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_full_codex_multiformat_982g_sharded/final/final_codex_corpus_report.json`
- Hard-negative validation report: `experiments/backfill/block_ar/nl_scenario_demo_outputs/hard_negative_bank_regeneration_985c_codex_batch/hard_negative_bank_validation_report.json`
- Positive corpus count: 4010 cards; invalid=0; missing=0; local template prose used=False
- Hard-negative validated rows: 12433 / manifest 32080; status=pass; errors=0; warnings=65; local prose generated=False

Note: the positive card also has an `old_factor_baseline` view, but the current hard-negative manifest covers the eight view styles listed below. This packet therefore includes those eight matched positive/negative styles.

## Paired Review

### 1. Sparse User Query (`sparse_user_query`)

Positive narrative:

> The current tape is a weak-dollar commodity bid rather than a broad credit event. Credit confirmation has narrowed, with lower-quality spreads still flat.

Corresponding hard negative:

- Row id: `joint39_train_1553__sparse_user_query`
- Negative window: `joint39_train_2128`
- Negative observed prefix date range: 2008-06-24 to 2008-08-05
- Negative title: dollar-linked liquidity pressure with commodity liquidation
- Negative archetype: `liquidity_withdrawal`
- Contradiction channels: SPX, AAA_OAS, DXY, USDJPY, CRUDE_OIL, US2Y, US10Y, GOLD

Negative mechanical summary:

> Mechanical baseline: SPX lower small; VIX lower small; BBB_OAS wider; AAA_OAS tighter; DXY higher small; USDJPY lower; crude oil lower large; US2Y lower; US10Y lower small; gold lower small.

Hard-negative narrative:

> The current tape is commodity liquidation, with crude sharply lower and gold softer while the dollar firms modestly. That is not a weak-dollar commodity bid; credit is mixed, with BBB widening even as high-grade spreads tighten.

Quality notes:

- Crude and gold weaken instead of leading a bid.
- Dollar strength contradicts weak-dollar conditions.
- BBB widening conflicts with fading credit stress.

### 2. Weekly Risk Monitor (`weekly_risk_monitor`)

Positive narrative:

> The main change is a weaker broad dollar supporting gold and energy. High-grade credit is only a secondary signal now, and equity volatility remains contained. The uncertainty is whether the commodity move is macro inflation pressure or a currency-led asset repricing.

Corresponding hard negative:

- Row id: `joint39_train_1553__weekly_risk_monitor`
- Negative window: `joint39_train_2130`
- Negative observed prefix date range: 2008-06-26 to 2008-08-07
- Negative title: dollar squeeze with hard-asset liquidation
- Negative archetype: `liquidity_withdrawal`
- Contradiction channels: SPX, AAA_OAS, DXY, USDJPY, CRUDE_OIL, US2Y, US10Y, GOLD

Negative mechanical summary:

> Mechanical baseline: SPX lower small; VIX lower small; BBB_OAS wider; AAA_OAS tighter; DXY higher medium; USDJPY lower; crude oil lower large; US2Y lower; US10Y lower; gold lower medium.

Hard-negative narrative:

> The current prefix is a dollar-led defensive tape, not a weak-dollar commodity bid. The broad dollar is firmer while crude and gold are being sold, rates are lower, BBB spreads are wider, and the only credit improvement is concentrated in tighter AAA spreads. The uncertainty is whether this is primarily liquidity pressure or a growth-sensitive commodity liquidation.

Quality notes:

- DXY is higher rather than lower.
- Gold and crude are lower rather than bid.
- AAA spreads tighten while BBB widens, reversing the credit emphasis.

### 3. Mechanism First (`mechanism_first`)

Positive narrative:

> The mechanism is a currency-commodity repricing: a weaker broad dollar supports gold and oil, while yields drift higher. Credit is no longer the dominant stress channel, because high-grade widening is smaller and lower-quality spreads are flat. The portfolio implication is more macro and commodity exposure than broad spread deterioration.

Corresponding hard negative:

- Row id: `joint39_train_1553__mechanism_first`
- Negative window: `joint39_train_2202`
- Negative observed prefix date range: 2008-10-08 to 2008-11-18
- Negative title: Dollar squeeze with gold liquidation and milder equity drawdown
- Negative archetype: `liquidity_withdrawal`
- Contradiction channels: SPX, AAA_OAS, DXY, USDJPY, CRUDE_OIL, US2Y, US10Y, GOLD

Negative mechanical summary:

> Mechanical baseline: SPX large lower; VIX large higher; BBB_OAS wider; AAA_OAS medium tighter; DXY large higher; USDJPY large lower; crude oil large lower; US2Y lower; US10Y slightly lower; gold large lower.

Hard-negative narrative:

> The mechanism is a dollar squeeze transmitting into hard-asset liquidation and defensive duration demand. A sharply stronger dollar coincides with large declines in gold and crude, weaker equities, higher volatility, wider BBB spreads, and lower Treasury yields. That channel is the opposite of a weaker-dollar commodity repricing, and tighter AAA spreads look like quality demand rather than relaxed credit conditions.

Quality notes:

- Mechanism is dollar squeeze, not weak-dollar support.
- Gold and crude are liquidated rather than supported.
- Credit stress persists through wider BBB spreads.

### 4. Technical Factor Evidence (`technical_factor_evidence`)

Positive narrative:

> Mechanical baseline: SPX higher small; VIX flat; BBB_OAS flat; AAA_OAS wider small; DXY lower medium; USDJPY higher large; CRUDE_OIL higher medium; US2Y higher small; US10Y higher small; GOLD higher medium.

Corresponding hard negative:

- Row id: `joint39_train_1553__technical_factor_evidence`
- Negative window: `joint39_train_2194`
- Negative observed prefix date range: 2008-09-26 to 2008-11-06
- Negative title: Renewed equity liquidation within a dollar squeeze
- Negative archetype: `liquidity_withdrawal`
- Contradiction channels: SPX, AAA_OAS, DXY, USDJPY, CRUDE_OIL, US2Y, US10Y, GOLD

Negative mechanical summary:

> Mechanical baseline: SPX large lower; VIX large higher; BBB_OAS much wider; AAA_OAS tighter; DXY large higher; USDJPY large lower; crude oil large lower; US2Y large lower; US10Y slightly lower; gold large lower.

Hard-negative narrative:

> Mechanical current-prefix read: SPX large lower; VIX large higher; BBB_OAS much wider; AAA_OAS tighter; DXY large higher; USDJPY large lower; CRUDE_OIL large lower; US2Y large lower; US10Y slightly lower; GOLD large lower. Evidence points to equity liquidation, dollar strength, commodity selling, and lower-quality credit stress, not weak-dollar commodity support.

Quality notes:

- SPX/VIX stress contradicts calm equity risk.
- DXY and commodities move opposite the weak-dollar commodity pattern.
- BBB spreads widen sharply while AAA tightens.

### 5. Factor List Baseline (`factor_list_baseline`)

Positive narrative:

> Mechanical baseline: SPX higher small; VIX flat; BBB_OAS flat; AAA_OAS wider small; DXY lower medium; USDJPY higher large; CRUDE_OIL higher medium; US2Y higher small; US10Y higher small; GOLD higher medium.

Corresponding hard negative:

- Row id: `joint39_train_1553__factor_list_baseline`
- Negative window: `joint39_train_2204`
- Negative observed prefix date range: 2008-10-10 to 2008-11-20
- Negative title: Treasury-bid risk-off with lower-quality credit stress
- Negative archetype: `liquidity_withdrawal`
- Contradiction channels: SPX, AAA_OAS, DXY, USDJPY, CRUDE_OIL, US2Y, US10Y, GOLD

Negative mechanical summary:

> Mechanical baseline: SPX large lower; VIX large higher; BBB_OAS wider; AAA_OAS slightly tighter; DXY large higher; USDJPY large lower; crude oil large lower; US2Y large lower; US10Y large lower; gold large lower.

Hard-negative narrative:

> Current-prefix factors: SPX large lower; VIX large higher; BBB_OAS wider; AAA_OAS slightly tighter; DXY large higher; USDJPY large lower; CRUDE_OIL large lower; US2Y large lower; US10Y large lower; GOLD large lower. Compact read: Treasury-bid risk-off with dollar strength, commodity liquidation, and lower-quality credit stress.

Quality notes:

- DXY higher and commodities lower contradict weak-dollar commodity strength.
- SPX/VIX indicate risk-off rather than calm markets.
- BBB widening conflicts with stable lower-quality credit.

### 6. Institutional Risk Committee Note (`institutional_risk_committee_note`)

Positive narrative:

> This is a moderate macro repricing, not an acute stress state. The committee would flag commodity, FX, and rate-sensitive exposures rather than broad credit beta. Transmission is through broad-dollar weakness and commodity strength, with high-grade credit offering only limited confirmation. The key ambiguity is that gold is firm but broad risk markets are stable, so the move should not be titled as classic safe-haven risk-off.

Corresponding hard negative:

- Row id: `joint39_train_1553__institutional_risk_committee_note`
- Negative window: `joint39_train_2193`
- Negative observed prefix date range: 2008-09-25 to 2008-11-05
- Negative title: Rates-aided dollar deleveraging with credit stress
- Negative archetype: `liquidity_withdrawal`
- Contradiction channels: SPX, AAA_OAS, DXY, USDJPY, CRUDE_OIL, US2Y, US10Y, GOLD

Negative mechanical summary:

> Mechanical baseline: SPX large lower; VIX large higher; BBB_OAS much wider; AAA_OAS tighter; DXY large higher; USDJPY large lower; crude oil large lower; US2Y large lower; US10Y slightly lower; gold large lower.

Hard-negative narrative:

> Severity is high: the current prefix shows broad deleveraging, not a moderate macro repricing. Equity beta, lower-quality credit, commodity exposures, and dollar-sensitive positions are all affected as SPX falls sharply, volatility rises sharply, BBB spreads move much wider, and the dollar strengthens materially. Transmission runs through a dollar squeeze and risk liquidation, with crude, gold, USDJPY, and front-end yields all lower; the committee should treat the tighter AAA spread signal as a defensive quality rotation rather than confirmation of benign credit conditions.

Quality notes:

- Risk severity is acute rather than moderate.
- Dollar strength and commodity liquidation contradict weak-dollar commodity strength.
- SPX and VIX show stress rather than stable risk markets.

### 7. Risk Manager Memo (`risk_manager_memo`)

Positive narrative:

> Regime: weak-dollar commodity bid with fading credit stress. The trigger is a softer broad dollar and stronger commodity complex, with gold and crude both up over the prefix. Transmission into credit is less severe than earlier, as high-grade widening is smaller and lower-quality credit remains flat. Cross-asset confirmation is strongest in FX and commodities, while rates provide only moderate support and equity risk remains calm. Portfolio vulnerability is concentrated in commodity underweights, dollar-sensitive trades, and residual high-grade spread exposure. No forecast is embedded in this description.

Corresponding hard negative:

- Row id: `joint39_train_1553__risk_manager_memo`
- Negative window: `joint39_train_1304`
- Negative observed prefix date range: 2005-03-16 to 2005-04-27
- Negative title: dollar-duration defense with weaker commodities
- Negative archetype: `liquidity_withdrawal`
- Contradiction channels: SPX, AAA_OAS, DXY, USDJPY, CRUDE_OIL, US2Y, US10Y, GOLD

Negative mechanical summary:

> Mechanical baseline: SPX down small; VIX up small; BBB_OAS wider medium; AAA_OAS tighter medium; DXY up medium; USDJPY down small; crude oil down small; US2Y down small; US10Y down medium; gold down small.

Hard-negative narrative:

> Regime: dollar-duration defense with weaker commodities and persistent credit pressure. The trigger evidence is a firmer broad dollar, lower Treasury yields, softer crude and gold, and a small equity drawdown with modestly higher volatility. Transmission is defensive rather than reflationary: lower-quality spreads widen even as high-grade spreads tighten, leaving cross-asset confirmation in FX, rates, and credit rather than in a stronger commodity complex. Ambiguity remains because the equity move is contained, but the current prefix is inconsistent with a calm weak-dollar commodity rally.

Quality notes:

- Commodity and dollar channels point opposite the requested regime.
- BBB widening contradicts fading credit stress.
- Rates fall rather than drift higher.

### 8. Full Professional Memo (`full_professional`)

Positive narrative:

> Regime: weak-dollar commodity bid with fading credit stress. The trigger is a softer broad dollar and stronger commodity complex, with gold and crude both up over the prefix. Transmission into credit is less severe than earlier, as high-grade widening is smaller and lower-quality credit remains flat. Cross-asset confirmation is strongest in FX and commodities, while rates provide only moderate support and equity risk remains calm. Portfolio vulnerability is concentrated in commodity underweights, dollar-sensitive trades, and residual high-grade spread exposure. No forecast is embedded in this description.

Corresponding hard negative:

- Row id: `joint39_train_1553__full_professional`
- Negative window: `joint39_train_2203`
- Negative observed prefix date range: 2008-10-09 to 2008-11-19
- Negative title: Rates-confirmed risk aversion with persistent dollar bid
- Negative archetype: `liquidity_withdrawal`
- Contradiction channels: SPX, AAA_OAS, DXY, USDJPY, CRUDE_OIL, US2Y, US10Y, GOLD

Negative mechanical summary:

> Mechanical baseline: SPX large lower; VIX large higher; BBB_OAS wider; AAA_OAS medium tighter; DXY large higher; USDJPY large lower; crude oil large lower; US2Y large lower; US10Y large lower; gold large lower.

Hard-negative narrative:

> Regime: rates-confirmed risk aversion with a persistent dollar bid. Trigger evidence is concentrated in a large rise in DXY, sharp equity losses, higher volatility, lower crude and gold, and materially lower Treasury yields. Transmission runs from currency strength and risk reduction into commodity liquidation and BBB spread widening, while tighter AAA spreads signal quality demand rather than broad credit healing. Cross-asset confirmation is strongest in FX, rates, equities, and commodities; ambiguity is limited mainly to the high-grade spread compression, which is defensive in this prefix rather than reflationary.

Quality notes:

- Dollar and commodity channels reverse the weak-dollar commodity bid.
- Risk assets show stress instead of calm.
- Lower yields and BBB widening contradict the benign macro read.

## Suggested Human Review Questions

- Does each positive/negative pair feel like the same style but opposite economic state?
- Is the hard-negative narrative readable without saying it is a negative example?
- Does the mechanical summary support the contradiction channels?
- Are the short/sparse pairs realistic for real user input, or still too polished?
- Are any narratives accidentally forecast-like, or are they clearly limited to the current/recent prefix?
