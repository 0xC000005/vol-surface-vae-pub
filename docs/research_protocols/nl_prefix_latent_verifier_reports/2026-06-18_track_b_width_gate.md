# Track B WIDTH gate -- 2026-06-18

- Trained checkpoint: `models/backfill/generator_conditioning_probe_bwidth/best_model.pt`
- Frozen 734a base: `models/backfill/734a_joint39_realvix_channel_level_alltrain_w005_e3_s7345/best_model.pt`
- Held-out val windows with narrative: 30; embed_mode=openai
- **VERDICT: KILL**

## Width response (grounded intensity vs width_DELTA)
- spearman REAL: 0.16885428253615126 (PASS bar > 0.3)
- spearman SHUFFLED (one draw): -0.12169076751946609
- **real - shuffled: 0.29054505005561737** (PASS bar > 0.15)
- permutation null: mean 0.011460066740823137, std 0.1870502353744497, p97.5(abs) 0.4104783092324804, exceeds_upper_tail False, p2s 0.38
- width: cond 0.7372 base 0.3085 mean|delta| 0.42875
- intensity spread: std 0.257 n_unique 12

## Severity-dial (intercept-aware, ABSOLUTE intensity)
- OLS slope (width_delta ~ abs_intensity): 0.018008 (bar > 0.0)
- OLS intercept: 0.39108
- low-tercile mean width_delta: 0.44312 | high-tercile: 0.43783
- low/high inflation frac: 1.012 (bar <= 0.5)
- **dial pass (proportional, not uniform presence-inflation): False**

## Fidelity guardrail (vs frozen 734a)
- CRPS gap 0.8294 | Energy gap 1.015 (guardrail +/-0.015)
- fidelity pass: False

> intensity vs width_DELTA (present=True - present=False, CRN); raw width is regime-confounded and NOT used for the verdict. NO OpenAI for intensity.
