# AR Compounding Diagnostic — 233a / 229a

- model_233a: `models/backfill/233a_v1_full_25d_s42/best_model.pt`
- model_229a: `models/backfill/factor_ar_229a_wide_decoder/checkpoint_ep30.pt`
- windows: 100, members: 24, steps: 30

## Conclusions

### Overall finding: compounding is NOT the primary culprit for either model

Neither model shows the canonical compounding signature (SF degrades, TF recovers). The failure modes are different for the two models.

---

### 233a v1-full — Verdict: MARGINAL SCALE ATTENUATION, RECOVERS LATE

Self-fed KS profile: rises h1→h10 (0.075→0.124), then **recovers** to 0.078 at h30.
Teacher-forced KS: lower throughout (mean 0.080 vs 0.104), confirming TF does help slightly, but both modes stay in low-KS territory.

Key observations:
- **MSD (δ²) ratio h30/h1 = 0.74 (SF)** — innovation energy moderately attenuates (~26% lower at h30 vs h1). The sinh/scale architecture is partially suppressing variance over the rollout.
- **Teacher-forced MSD grows over rollout** (ratio 1.80) — GT-conditioned path actually generates larger variance at later steps (longer-horizon GT moves are bigger on average), suggesting the SF scale attenuation is an artifact of self-fed state regression toward mean.
- **Lag-1 autocorrelation = -0.34 (SF) vs +0.09 (TF)** — Strong negative autocorr under self-feeding indicates sign-alternating innovations (the model corrects overshoots each step). Under TF, autocorr is near-zero (appropriate). This negative autocorr partially explains why SF KS recovers late: the distribution of |delta| is similar but the sign pattern is structured.
- **Cross-member diversity: near-stable** (0.0175 → 0.0163), no ensemble collapse.
- **Single-step law is adequate**: TF h1 KS = 0.073 (well below 0.30 threshold). The per-step emission law itself is fine.

**Diagnosis (phenomenological):** The primary 233a failure is **scale regression + sign-correction oscillation** under self-feeding, not raw compounding. The data are consistent with a negative feedback loop in the slow-path FiLM modulation (model over-corrects each step's overshoot, building -0.34 lag-1 autocorrelation as a symptom), but the exact mechanism is a hypothesis — not directly proven here. What is directly observed: negative autocorr artificially flattens KS at late horizons by producing a mean-reverting innovation sequence instead of i.i.d. draws. Note also: the per-step max|δ| attenuation (MSD ratio 0.74, Max|δ| declining from 1.87 to ~1.2 by mid-rollout) directly accounts for why max_jump_ks degrades at multi-day horizons — the sinh/scale architecture is suppressing tail innovations, which flattens the jump-size distribution relative to GT.

---

### 229a — Verdict: STATE OVERCORRECTION UNDER TEACHER-FORCING; SF HAS VARIANCE COLLAPSE

This is the most surprising finding. **Teacher-forcing makes 229a dramatically worse** (KS h30: SF=0.158, TF=0.269). This inverts the compounding hypothesis.

Key observations:
- **SF variance collapse**: MSD ratio h30/h1 = 0.11 — innovation energy falls to 11% by h30 under self-feeding. Max|δ| drops from 2.54→1.36. The model is attenuating its own innovations as the self-fed state drifts.
- **TF variance collapse is slower but KS diverges worse**: TF MSD ratio = 0.18 (from 0.0060→0.0011), Max|δ| drops from 2.60→0.63. Even smaller per-step moves, yet KS is much higher.
- **Catastrophic lag-1 autocorr under TF = 0.66** — when fed GT transitions, the GRUCell updates accumulate a strong positive autocorrelation signature. The model was trained under self-fed rollouts; GT transitions are a distribution shift to the state update path. The GRU learns to model self-fed state dynamics, not GT transitions. When TF is applied, the GRU enters a regime where it produces persistent (rather than mean-reverting) innovations — autocorr of 0.66 means each delta is strongly predicted by the previous.
- **CM-Std collapses under SF** (0.0168→0.0098) — 42% reduction in cross-member diversity by h30. This is the diversity collapse signature: ensemble members are converging as the self-fed state homogenizes them.
- **KS grows monotonically in SF** (0.054→0.158): structural, consistent distributional drift. Every step erodes fidelity.

**Diagnosis (phenomenological):** The 229a failure is **variance collapse** (not compounding in the classical sense). Under self-feeding, MSD falls 10x by h30. The data are consistent with a self-reinforcing EWMA attenuation hypothesis: each generation slightly underestimates scale, which shrinks local_scale, which further shrinks the next generation. However, this is a hypothesis — the diagnostic does not isolate whether the collapse originates in EWMA, in the GRU state, or in a combination. What is directly observed: MSD ratio = 0.10, Max|δ| drops from 2.54 to 1.36, CM-Std collapses 42%. Teacher-forcing inverts the result (TF KS h30 = 0.269 vs SF = 0.158): GT-fed moves are larger, which partially counteracts variance collapse in early steps, but the GRU state update entered an off-distribution regime under TF, producing lag-1 autocorr = 0.66 (vs 0.18 self-fed). The max|δ| attenuation under SF (2.54→1.36) is the direct mechanistic link to the max_jump_ks metric failing at multi-day horizons: generated paths have structurally smaller jumps than GT because scale is collapsing, so the jump-size distribution becomes increasingly under-dispersed.

**Working hypothesis for RC23:** The variance collapse is consistent with a tight feedback between local_scale (EWMA) and GRU state — the mechanism the rc23_local_temporal_decomposition design proposes to break via restricted coupling. This remains a hypothesis until a decoupled architecture is tested and shows reduced collapse.

---

### Key Numbers at h1/h5/h10/h20/h30

| Model / Mode | KS@h1 | KS@h5 | KS@h10 | KS@h20 | KS@h30 | CM-Std h1 | CM-Std h30 | Lag1-AutoCorr |
|---|---|---|---|---|---|---|---|---|
| 233a SF | 0.075 | 0.117 | 0.124 | 0.104 | 0.078 | 0.0175 | 0.0163 | -0.3356 |
| 233a TF | 0.074 | 0.080 | 0.097 | 0.082 | 0.058 | 0.0171 | 0.0181 | 0.0872 |
| 229a SF | 0.054 | 0.112 | 0.141 | 0.143 | 0.158 | 0.0168 | 0.0098 | 0.1826 |
| 229a TF | 0.060 | 0.071 | 0.169 | 0.236 | 0.269 | 0.0168 | 0.0099 | 0.6592 |

### TF vs SF KS Gap at h30 (compounding diagnostic)
- 233a TF–SF gap: +0.020  (threshold=0.15)
- 229a TF–SF gap: -0.111  (threshold=0.15)

## 233a — Self-Fed Per-Step Table

| Step | KS vs GT | MSD (δ²) | Max|δ| | CM-Std | Lag1-AutoCorr |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0747 | 0.0026 | 1.8712 | 0.0175 | nan |
| 1 | 0.1209 | 0.0016 | 1.0367 | 0.0156 | -0.3707 |
| 2 | 0.0944 | 0.0014 | 2.1574 | 0.0144 | -0.3171 |
| 3 | 0.1156 | 0.0013 | 1.1757 | 0.0139 | -0.2685 |
| 4 | 0.1168 | 0.0011 | 1.2665 | 0.0136 | -0.2924 |
| 5 | 0.1117 | 0.0013 | 1.1071 | 0.0140 | -0.3218 |
| 6 | 0.1096 | 0.0013 | 0.9329 | 0.0142 | -0.3371 |
| 7 | 0.1149 | 0.0014 | 1.5722 | 0.0145 | -0.3129 |
| 8 | 0.1307 | 0.0014 | 1.7815 | 0.0144 | -0.2969 |
| 9 | 0.1237 | 0.0014 | 1.6504 | 0.0144 | -0.3578 |
| 10 | 0.1277 | 0.0014 | 1.5816 | 0.0147 | -0.2904 |
| 11 | 0.1211 | 0.0014 | 1.5077 | 0.0149 | -0.3331 |
| 12 | 0.1126 | 0.0017 | 3.1550 | 0.0152 | -0.3237 |
| 13 | 0.1086 | 0.0017 | 1.7285 | 0.0156 | -0.3291 |
| 14 | 0.1116 | 0.0015 | 1.2172 | 0.0151 | -0.3788 |
| 15 | 0.0986 | 0.0015 | 1.1396 | 0.0153 | -0.3677 |
| 16 | 0.1101 | 0.0015 | 1.2016 | 0.0151 | -0.3163 |
| 17 | 0.0979 | 0.0015 | 1.4575 | 0.0150 | -0.3169 |
| 18 | 0.1000 | 0.0015 | 1.3482 | 0.0149 | -0.3188 |
| 19 | 0.1043 | 0.0014 | 1.2060 | 0.0145 | -0.3436 |
| 20 | 0.0984 | 0.0015 | 1.4037 | 0.0149 | -0.3540 |
| 21 | 0.1056 | 0.0015 | 1.1873 | 0.0149 | -0.3653 |
| 22 | 0.0988 | 0.0015 | 2.3652 | 0.0148 | -0.3325 |
| 23 | 0.1057 | 0.0017 | 3.2398 | 0.0150 | -0.3619 |
| 24 | 0.0932 | 0.0015 | 1.2885 | 0.0149 | -0.3376 |
| 25 | 0.0847 | 0.0015 | 1.1349 | 0.0150 | -0.3421 |
| 26 | 0.0796 | 0.0016 | 1.4123 | 0.0155 | -0.3759 |
| 27 | 0.0800 | 0.0018 | 1.6573 | 0.0156 | -0.3462 |
| 28 | 0.0909 | 0.0017 | 1.3255 | 0.0158 | -0.3689 |
| 29 | 0.0779 | 0.0019 | 2.1215 | 0.0163 | -0.3543 |

## 233a — Teacher-Forced Per-Step Table

| Step | KS vs GT | MSD (δ²) | Max|δ| | CM-Std | Lag1-AutoCorr |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0735 | 0.0025 | 1.1609 | 0.0171 | nan |
| 1 | 0.0698 | 0.0024 | 1.4819 | 0.0155 | 0.0993 |
| 2 | 0.0595 | 0.0021 | 1.8399 | 0.0149 | 0.1390 |
| 3 | 0.0772 | 0.0021 | 1.2472 | 0.0148 | 0.1212 |
| 4 | 0.0804 | 0.0021 | 1.4980 | 0.0147 | 0.1430 |
| 5 | 0.0836 | 0.0021 | 1.8461 | 0.0147 | 0.1004 |
| 6 | 0.0878 | 0.0021 | 2.0376 | 0.0151 | 0.0687 |
| 7 | 0.0847 | 0.0022 | 1.3689 | 0.0156 | 0.1026 |
| 8 | 0.0973 | 0.0023 | 1.1951 | 0.0158 | 0.1085 |
| 9 | 0.0972 | 0.0025 | 1.7934 | 0.0160 | 0.0870 |
| 10 | 0.0946 | 0.0029 | 1.9064 | 0.0162 | 0.0775 |
| 11 | 0.0911 | 0.0027 | 1.4460 | 0.0164 | 0.1161 |
| 12 | 0.0860 | 0.0029 | 1.5556 | 0.0164 | 0.0957 |
| 13 | 0.0929 | 0.0027 | 1.4380 | 0.0165 | 0.1126 |
| 14 | 0.0776 | 0.0030 | 1.3381 | 0.0165 | 0.0968 |
| 15 | 0.0842 | 0.0031 | 1.6874 | 0.0165 | 0.1188 |
| 16 | 0.0855 | 0.0032 | 3.8405 | 0.0166 | 0.0805 |
| 17 | 0.0660 | 0.0031 | 1.3725 | 0.0164 | 0.0652 |
| 18 | 0.0872 | 0.0033 | 2.6808 | 0.0165 | 0.0765 |
| 19 | 0.0823 | 0.0028 | 2.3155 | 0.0159 | 0.0500 |
| 20 | 0.0874 | 0.0029 | 2.6492 | 0.0160 | 0.0619 |
| 21 | 0.0720 | 0.0029 | 2.1025 | 0.0162 | 0.0548 |
| 22 | 0.0790 | 0.0029 | 1.3137 | 0.0166 | 0.0876 |
| 23 | 0.0781 | 0.0031 | 1.6261 | 0.0166 | 0.0899 |
| 24 | 0.0848 | 0.0028 | 1.7826 | 0.0161 | 0.0755 |
| 25 | 0.0773 | 0.0030 | 1.8894 | 0.0164 | 0.0866 |
| 26 | 0.0715 | 0.0032 | 1.9281 | 0.0170 | 0.0783 |
| 27 | 0.0660 | 0.0033 | 1.5249 | 0.0168 | 0.0466 |
| 28 | 0.0576 | 0.0038 | 1.8065 | 0.0177 | 0.0687 |
| 29 | 0.0577 | 0.0044 | 1.7977 | 0.0181 | 0.0201 |

## 229a — Self-Fed Per-Step Table

| Step | KS vs GT | MSD (δ²) | Max|δ| | CM-Std | Lag1-AutoCorr |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0539 | 0.0058 | 2.5444 | 0.0168 | nan |
| 1 | 0.0506 | 0.0015 | 2.0751 | 0.0128 | 0.2017 |
| 2 | 0.0714 | 0.0006 | 0.9817 | 0.0109 | 0.0294 |
| 3 | 0.0824 | 0.0005 | 0.7430 | 0.0102 | -0.0482 |
| 4 | 0.1120 | 0.0004 | 0.6858 | 0.0097 | -0.0079 |
| 5 | 0.1218 | 0.0004 | 0.6052 | 0.0091 | 0.0330 |
| 6 | 0.1384 | 0.0004 | 0.6981 | 0.0090 | 0.0918 |
| 7 | 0.1535 | 0.0004 | 0.9121 | 0.0086 | 0.1817 |
| 8 | 0.1502 | 0.0004 | 1.3242 | 0.0088 | 0.2093 |
| 9 | 0.1409 | 0.0004 | 1.3082 | 0.0091 | 0.1794 |
| 10 | 0.1219 | 0.0005 | 1.7669 | 0.0092 | 0.2865 |
| 11 | 0.1336 | 0.0005 | 1.2336 | 0.0092 | 0.2135 |
| 12 | 0.1179 | 0.0005 | 1.4255 | 0.0091 | 0.2297 |
| 13 | 0.1271 | 0.0005 | 1.4034 | 0.0089 | 0.2860 |
| 14 | 0.1161 | 0.0004 | 0.9265 | 0.0089 | 0.2649 |
| 15 | 0.1119 | 0.0005 | 2.4344 | 0.0089 | 0.2714 |
| 16 | 0.1117 | 0.0004 | 2.0171 | 0.0083 | 0.1509 |
| 17 | 0.1114 | 0.0003 | 0.9983 | 0.0081 | 0.1590 |
| 18 | 0.1273 | 0.0003 | 0.7728 | 0.0080 | 0.1278 |
| 19 | 0.1433 | 0.0004 | 2.5069 | 0.0082 | 0.2148 |
| 20 | 0.1456 | 0.0003 | 1.1020 | 0.0079 | 0.2499 |
| 21 | 0.1303 | 0.0003 | 0.9390 | 0.0080 | 0.1582 |
| 22 | 0.1470 | 0.0005 | 2.0652 | 0.0085 | 0.1851 |
| 23 | 0.1455 | 0.0005 | 2.0762 | 0.0087 | 0.2041 |
| 24 | 0.1534 | 0.0005 | 1.7034 | 0.0087 | 0.2237 |
| 25 | 0.1546 | 0.0005 | 1.0939 | 0.0086 | 0.1902 |
| 26 | 0.1623 | 0.0007 | 1.9036 | 0.0091 | 0.3092 |
| 27 | 0.1588 | 0.0005 | 1.2719 | 0.0091 | 0.2443 |
| 28 | 0.1600 | 0.0006 | 1.4070 | 0.0097 | 0.2267 |
| 29 | 0.1584 | 0.0006 | 1.3601 | 0.0098 | 0.2284 |

## 229a — Teacher-Forced Per-Step Table

| Step | KS vs GT | MSD (δ²) | Max|δ| | CM-Std | Lag1-AutoCorr |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0596 | 0.0060 | 2.5960 | 0.0168 | nan |
| 1 | 0.0413 | 0.0026 | 2.2725 | 0.0120 | 0.6813 |
| 2 | 0.0393 | 0.0018 | 1.2378 | 0.0108 | 0.6798 |
| 3 | 0.0634 | 0.0018 | 1.4333 | 0.0104 | 0.7024 |
| 4 | 0.0707 | 0.0019 | 1.6174 | 0.0106 | 0.7144 |
| 5 | 0.0799 | 0.0019 | 1.6040 | 0.0103 | 0.7271 |
| 6 | 0.1018 | 0.0022 | 1.7045 | 0.0105 | 0.7552 |
| 7 | 0.1272 | 0.0023 | 2.0191 | 0.0104 | 0.7402 |
| 8 | 0.1492 | 0.0019 | 1.6261 | 0.0101 | 0.7478 |
| 9 | 0.1687 | 0.0017 | 1.8352 | 0.0102 | 0.7386 |
| 10 | 0.1847 | 0.0016 | 1.6701 | 0.0103 | 0.7039 |
| 11 | 0.1923 | 0.0015 | 1.1637 | 0.0102 | 0.7005 |
| 12 | 0.1875 | 0.0014 | 1.3723 | 0.0103 | 0.6849 |
| 13 | 0.2000 | 0.0012 | 0.9536 | 0.0099 | 0.6829 |
| 14 | 0.2083 | 0.0012 | 0.8107 | 0.0099 | 0.6766 |
| 15 | 0.2177 | 0.0011 | 1.0092 | 0.0097 | 0.6568 |
| 16 | 0.2015 | 0.0011 | 0.8536 | 0.0096 | 0.6477 |
| 17 | 0.2055 | 0.0011 | 0.6939 | 0.0099 | 0.6233 |
| 18 | 0.2169 | 0.0011 | 0.7725 | 0.0098 | 0.6235 |
| 19 | 0.2361 | 0.0010 | 0.7391 | 0.0095 | 0.6306 |
| 20 | 0.2334 | 0.0010 | 0.7469 | 0.0093 | 0.6310 |
| 21 | 0.2423 | 0.0009 | 0.7148 | 0.0093 | 0.6089 |
| 22 | 0.2488 | 0.0010 | 0.6878 | 0.0094 | 0.5936 |
| 23 | 0.2558 | 0.0010 | 0.6750 | 0.0095 | 0.5958 |
| 24 | 0.2598 | 0.0011 | 0.6411 | 0.0097 | 0.5900 |
| 25 | 0.2665 | 0.0011 | 0.7254 | 0.0095 | 0.6048 |
| 26 | 0.2652 | 0.0011 | 0.6925 | 0.0096 | 0.6316 |
| 27 | 0.2592 | 0.0011 | 0.6598 | 0.0096 | 0.5935 |
| 28 | 0.2712 | 0.0011 | 0.7154 | 0.0098 | 0.5768 |
| 29 | 0.2692 | 0.0011 | 0.6320 | 0.0099 | 0.5729 |

## Interpretation Guide

- **KS vs GT**: 2-sample KS stat between generated deltas (W×K×25 pool) and GT deltas (W×25 pool) at step t.
  KS=0 is perfect match; KS>0.5 is severe mismatch. Monotone increase → structural error accumulation.
- **MSD (δ²)**: mean squared daily delta. Rising → volatility amplification. Falling → variance collapse.
- **Max|δ|**: pathwise extremes. Declining → innovation attenuation (regime smoothing away tails).
- **CM-Std**: cross-member diversity. Collapsed by h30 → ensemble degeneracy.
- **Lag1-AutoCorr**: positive = AR smoothing dominates; near zero = faithful uncorrelated noise.

**Compounding vs single-step decision rule:**
- If TF−SF KS gap > 0.15 AND TF h1 KS < 0.3: compounding is primary.
- If TF h1 KS > 0.3 regardless of gap: single-step law is the bottleneck.
- If gap is modest (< threshold): state drift is likely — cond/local_scale drift to wrong region.
