# 233a Slow-State Regime Encoding Diagnostic

Date: 2026-04-18

Val split: indices 4010–4450 (441 windows)
Regime bins: calm=111, mid=219, turb=111 (25th/75th percentile RV proxy split)

## Q1: Does `lam_hawkes` differentiate regimes?

| Seed | lam_hawkes calm | lam_hawkes turb | Ratio (turb/calm) | lam_hybrid ratio |
|------|-----------------|-----------------|-------------------|------------------|
| 1337 | 0.03206 | 0.11704 | **3.651** | 0.000 |
| 2024 | 0.09994 | 0.17552 | **1.756** | 0.000 |
| 42 | 0.12133 | 0.23998 | **1.978** | 1.117 |

Mean ratio across seeds: **2.462**  (>1.0 = turb has higher lam_hawkes)

## Q2: Does `s_t` differentiate regimes?

| Seed | s_ewma calm | s_ewma turb | s_ewma ratio | s_hybrid calm | s_hybrid turb | s_hybrid ratio |
|------|-------------|-------------|--------------|---------------|---------------|----------------|
| 1337 | 0.001548 | 0.004224 | **2.728** | 0.317491 | 0.331224 | **1.043** |
| 2024 | 0.001558 | 0.004213 | **2.703** | 0.410870 | 0.442295 | **1.077** |
| 42 | 0.001098 | 0.004219 | **3.842** | 0.942815 | 0.873132 | **0.926** |

## Q3: Does `h_slow` PC1 differentiate regimes?

| Seed | Cohen's d | AUC | PC1 var explained |
|------|-----------|-----|-------------------|
| 1337 | **1.124** | **0.792** | 0.998 |
| 2024 | **1.189** | **0.777** | 0.985 |
| 42 | **1.200** | **0.796** | 0.988 |

Mean Cohen's d: **1.171** | Mean AUC: **0.788**

(Cohen's d >0.3 = moderate; >0.8 = large; AUC >0.6 = discriminative)

## Q4: Are aux heads (`rv_head`, `jump_prob_head`) informative?

Two target definitions computed:
- **Natural**: `future[:,0] - history[:,-1]` (next-day change vs history end)
- **Literal** (spec): `future[:,1] - future[:,0]` (second vs first future step)

| Seed | rv_head corr (natural) | rv_head corr (literal) | rv_head corr (calm/turb) | jump AUC (natural) | jump AUC (literal) |
|------|------------------------|------------------------|--------------------------|--------------------|--------------------|
| 1337 | **0.1672** | 0.1064 | 0.2795 / 0.1961 | **0.6863** | 0.6535 |
| 2024 | **0.1572** | 0.1048 | 0.2635 / 0.2007 | **0.6764** | 0.6482 |
| 42 | **0.1639** | 0.1084 | 0.2844 / 0.2014 | **0.6755** | 0.6476 |

(rv_head corr >0.2 = meaningful; >0.4 = strong; AUC >0.6 = discriminative)

## Q5: Does slow-state diversity collapse during rollout?

### Seed 1337

| Mode | s_std [0-2] | s_std [27-29] | s collapse ratio | lam_std [0-2] | lam_std [27-29] | lam collapse ratio |
|------|-------------|---------------|------------------|---------------|-----------------|---------------------|
| teacher | 0.00483 | 0.00530 | **1.0973** | 0.00000 | 0.00000 | **0.0000** |
| selffed | 0.00505 | 0.00174 | **0.3444** | 0.00000 | 0.00000 | **0.0000** |

### Seed 2024

| Mode | s_std [0-2] | s_std [27-29] | s collapse ratio | lam_std [0-2] | lam_std [27-29] | lam collapse ratio |
|------|-------------|---------------|------------------|---------------|-----------------|---------------------|
| teacher | 0.01408 | 0.01533 | **1.0884** | 0.00000 | 0.00000 | **0.0000** |
| selffed | 0.01520 | 0.00662 | **0.4354** | 0.00000 | 0.00000 | **0.0000** |

### Seed 42

| Mode | s_std [0-2] | s_std [27-29] | s collapse ratio | lam_std [0-2] | lam_std [27-29] | lam collapse ratio |
|------|-------------|---------------|------------------|---------------|-----------------|---------------------|
| teacher | 0.03259 | 0.03530 | **1.0832** | 0.24206 | 0.24695 | **1.0202** |
| selffed | 0.03556 | 0.03366 | **0.9464** | 0.13598 | 0.15699 | **1.1545** |

(ratio <0.3 = severe collapse; 0.3-0.7 = moderate; >0.7 = maintained)

## Conclusion

**Classification: (c) for internal state, (a) for FiLM-facing output — upstream plumbing failure, not representation failure**

INTERNAL STATE is strongly discriminative: h_slow PC1 (Cohen's d=1.17, AUC=0.79), s_ewma ratio 3.1x, lam_hawkes ratio 2.5x. The slow path encodes regime well.

FILM-FACING OUTPUT is broken: (1) lam_hybrid is relu-killed to 0 for 2/3 seeds — the GRU's linear_lam learned a large negative bias, wiping out all Hawkes signal to FiLM. (2) s_hybrid ratio is ~1.0x (s1337/s2024) to inverted 0.93x (s42) — the GRU's linear_s correction actively cancels the analytic s_ewma signal. FiLM receives (log1p(s_hybrid), 0) where s_hybrid is nearly regime-blind.

ROOT CAUSE: The GRU correction heads (linear_s, linear_lam) learned to undo the analytic backbones rather than augment them. FiLM collapse is a training pathology in the hybrid combination stage, not a slow-path representation failure. Fix: rewire FiLM to consume h_slow directly (bypassing hybrid outputs), or add explicit loss to preserve analytic backbone signal through the GRU correction.

**Key numbers** (mean across 3 seeds):
- lam_hawkes ratio (turb/calm): 2.462
- h_slow PC1 Cohen's d: 1.171
- h_slow PC1 AUC: 0.788
- rv_head corr with log-RV: 0.1628
- Self-fed collapse observed: True
- Teacher-forced collapse observed: False
