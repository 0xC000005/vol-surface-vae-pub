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

NOTE: lam_hybrid = relu(lam_hawkes + linear_lam(h_slow)). For s1337 and s2024, the GRU
learned correction (linear_lam) is sufficiently negative that relu kills the output to 0.0
for ALL val windows — lam_hybrid is dead. s42 is the only seed where the hybrid lam channel
is alive (range 1.69–3.38, std=0.29 across 441 windows). This is a seed-specific training
failure for s1337/s2024, not a structural property of the architecture.

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

| Seed | rv_head corr | rv_head corr (calm) | rv_head corr (turb) | jump corr | jump AUC | jump base rate |
|------|--------------|---------------------|---------------------|-----------|----------|----------------|
| 1337 | **0.1672** | 0.2795 | 0.1961 | **0.2099** | 0.6863 | 0.1134 |
| 2024 | **0.1572** | 0.2635 | 0.2007 | **0.1945** | 0.6764 | 0.1134 |
| 42 | **0.1639** | 0.2844 | 0.2014 | **0.1942** | 0.6755 | 0.1134 |

(rv_head corr >0.2 = meaningful; >0.4 = strong; AUC >0.6 = discriminative)

## Q5: Does slow-state diversity collapse during rollout?

### Seed 1337

| Mode | s_std [0-2] | s_std [27-29] | s collapse ratio | lam_std [0-2] | lam_std [27-29] | lam collapse ratio |
|------|-------------|---------------|------------------|---------------|-----------------|---------------------|
| teacher | 0.00483 | 0.00530 | **1.0973** | 0.00000 | 0.00000 | **0.0000** |
| selffed | 0.00526 | 0.00153 | **0.2917** | 0.00000 | 0.00000 | **0.0000** |

### Seed 2024

| Mode | s_std [0-2] | s_std [27-29] | s collapse ratio | lam_std [0-2] | lam_std [27-29] | lam collapse ratio |
|------|-------------|---------------|------------------|---------------|-----------------|---------------------|
| teacher | 0.01408 | 0.01533 | **1.0884** | 0.00000 | 0.00000 | **0.0000** |
| selffed | 0.01477 | 0.00575 | **0.3894** | 0.00000 | 0.00000 | **0.0000** |

### Seed 42

| Mode | s_std [0-2] | s_std [27-29] | s collapse ratio | lam_std [0-2] | lam_std [27-29] | lam collapse ratio |
|------|-------------|---------------|------------------|---------------|-----------------|---------------------|
| teacher | 0.03259 | 0.03530 | **1.0832** | 0.24206 | 0.24695 | **1.0202** |
| selffed | 0.03573 | 0.03275 | **0.9165** | 0.15861 | 0.13642 | **0.8601** |

(ratio <0.3 = severe collapse; 0.3-0.7 = moderate; >0.7 = maintained)

NOTE on Q5 interpretation:
- lam_std = 0 for s1337/s2024 in both modes is because lam_hybrid is relu-killed to 0
  for all windows (see Q1 NOTE). The "collapse" to 0.0 was already present at step 0.
  This is not a rollout problem — it is a dead-relu failure mode from training.
- For s42 (the only seed where lam is alive): NO collapse in either mode (ratio >0.86).
  s_std is also well-maintained under both teacher and self-fed modes.
- For s1337/s2024: s_std collapses under self-feed (ratio 0.29 and 0.39), but NOT under
  teacher-forcing. This confirms the collapse pattern (b) for s_t, but only for seeds
  where the GRU head has weak signal (s1337/s2024 have 4-8x lower s_std than s42).

## Conclusion

**Classification: (c) discriminative throughout — but with a seed-dependent lam_hybrid failure**

**Detailed findings per component:**

1. **lam_hawkes (analytic Hawkes backbone)**: DISCRIMINATIVE across all seeds. Mean turb/calm
   ratio 2.46x. The analytic backbone correctly fires on jump windows.

2. **s_ewma (analytic EWMA backbone)**: STRONGLY DISCRIMINATIVE across all seeds.
   Mean turb/calm ratio 3.09x. The EWMA cleanly encodes realized variance regime.

3. **h_slow GRU hidden state**: STRONGLY DISCRIMINATIVE. PC1 Cohen's d = 1.17 (large),
   AUC = 0.79. PC1 explains 98.9% of h_slow variance — the GRU has collapsed to a
   1-dimensional regime summary, but that dimension is highly informative.

4. **lam_hybrid (relu-gated combination)**: DEAD for s1337/s2024. The GRU correction
   (linear_lam) learned a negative bias large enough to zero out relu output across all
   val windows. s42 is the only seed where the hybrid channel is alive. **This is the
   primary failure mode: lam_t fed to FiLM is 0 for 2/3 seeds, so FiLM input is
   essentially (log1p(s_t), 0) instead of (log1p(s_t), log1p(lam_t)).**

5. **s_hybrid**: Attenuated regime signal. s_ewma ratio is 3.09x but s_hybrid ratio is
   only 1.05x (s1337/s2024) or even inverted to 0.93x (s42). The GRU correction to s_t
   partially cancels the analytic EWMA signal.

6. **Aux heads**: Marginally informative. rv_head corr = 0.16 (below 0.2 threshold).
   jump_head AUC = 0.68 (modestly above 0.6 threshold). Aux supervision did not push
   the heads to strong predictive targets.

7. **Rollout collapse**: MIXED. For s42 (alive lam), no collapse in either mode.
   For s1337/s2024 (dead lam), s_std collapses under self-feed (~0.3x ratio) but
   not under teacher-forcing. The collapse is secondary to the relu-kill.

**Root cause of FiLM collapse**: Not a representation failure in the slow path itself —
the analytic backbones (lam_hawkes, s_ewma) are discriminative and h_slow PC1 is large.
The failure is in the **hybrid combination stage**: relu kills lam_hybrid to 0 for 2/3
seeds, and the GRU correction attenuates rather than amplifies s_t discrimination.
FiLM receives (log1p(s), 0) or weakly differentiated inputs and cannot leverage the
regime information that exists in the underlying lam_hawkes and s_ewma streams.

**Key numbers** (mean across 3 seeds):
- lam_hawkes ratio (turb/calm): 2.462
- s_ewma ratio (turb/calm): 3.091
- h_slow PC1 Cohen's d: 1.171
- h_slow PC1 AUC: 0.788
- lam_hybrid dead (relu-killed): 2/3 seeds
- s_hybrid ratio (attenuated): mean 1.015 (vs s_ewma 3.09x)
- rv_head corr with log-RV: 0.163 (below 0.2 threshold)
- jump_head AUC: 0.677 (modest)
- Self-fed s_std collapse (s1337/s2024 only): True
- Teacher-forced s_std collapse: False
