==============================================================================================================
233a-v1.2 — 7-variant comparison (seed 42)
==============================================================================================================
config                         n_pass    turb_calm   worstC_h30  max_jump_ks change_ks_h30     MR_ratio
--------------------------------------------------------------------------------------------------------------
229a @ep30 (incumbent)             3       1.025       0.271       0.945          19     —
v1-full_s42 (prev)                 2       1.010       0.385       0.735          13     —
v1.2-control_s42                   2       0.950       0.255       0.531          14     —
v1.2-minreg_s42                    2       1.065       0.297       0.849          12     —
v1.2-minimal_s42                   2       1.006       0.312       0.868          17     —
v1.2-aux_s42                       2       1.038       0.292       0.822          16     —
v1.2-link_s42                      2       1.011       0.240       0.737          23     —
v1.2-both_s42                      2       0.957       0.229       0.789          17     —
v1.2-noreg_s42                     2       1.032       0.115       0.806          11     —

==============================================================================================================
Attribution deltas
==============================================================================================================
C1+C2 alone (minreg - control): Δn_pass=0, Δturb_calm=0.115
C3 state-reg alone (minimal - minreg): Δn_pass=0
C4a twCRPS alone (aux - minimal): Δmax_jump_ks=-0.045
C4b learned-link alone (link - minimal): Δmax_jump_ks=-0.131
C4a+C4b combined (both - minimal): Δmax_jump_ks=-0.079

==============================================================================================================
Diagnostic roll-up (5 diagnostics × 7 variants)
==============================================================================================================
variant              film_logit_std              alpha_std          lag1_autocorr             h_slow_auc  regime_inversion_flag
--------------------------------------------------------------------------------------------------------------
control                      0.022                     —                -0.532                 0.983                  True
minreg                       0.098                     —                -0.408                 0.968                  True
minimal                      0.103                     —                -0.399                 0.966                  True
aux                          0.095                     —                -0.451                 0.966                  True
link                         0.001                 0.269                -0.408                 0.980                  True
both                         0.153                 0.314                -0.391                 0.960                  True
noreg                        0.001                 0.192                -0.364                 0.983                  True

==============================================================================================================
Decision tree (per Section 8 of design spec)
==============================================================================================================
BRANCH 3 (failure): best=control, n/7=2
  → paradigm pivot justified; launch H3 + joint-path flow matching

BRANCH 5 (YAGNI): v1.2-minimal ≈ v1.2-both (ΔN=0). Emission fixes add no value; deploy minimal.

BRANCH 7 (C3 NOT load-bearing): v1.2-minreg ≈ v1.2-minimal (|ΔN|=0). Drop state consistency reg from production recipe.
