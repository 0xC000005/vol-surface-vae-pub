# 261a Postmortem

## 260e vs 261a
- 260e: `{"calibration_error": 0.10558410493827158, "change_ks_pass": 24, "cointegration_ratio": 0.8822580645161292, "corr_ratio": 0.5090199832777952, "coverage90": 0.9511388888888889, "level_ks_pass": 0, "max_jump_ks": 0.46197916666666666, "mr_h30": 0.761295772398936, "mr_ratio": 0.6955149283085929, "n_pass": 4, "rank_ratio": 2.6530818273191263, "turb_calm": 0.9721773266792297}`
- 261a: `{"calibration_error": 0.07136342592592591, "change_ks_pass": 25, "cointegration_ratio": 0.703225806451613, "corr_ratio": 0.9801492979526747, "coverage90": 0.9114236111111111, "level_ks_pass": 6, "max_jump_ks": 0.5322916666666667, "mr_h30": 0.6866963924658407, "mr_ratio": 0.5423311824174519, "n_pass": 3, "rank_ratio": 1.3416308216129753, "turb_calm": 1.1188024282455444}`

## Mean-Preservation Probe
- uncentered_raw_resid_mean_abs: `0.007361663039773703`
- centered_raw_resid_mean_abs: `6.360614257516772e-10`
- centered_raw_resid_std: `0.04967837408185005`
- sample_mean_level_shift_abs_norm: `0.007813914678990841`
- sample_mean_level_shift_max_norm: `0.46925708651542664`

## Mechanism Read
- The residual layer is still moving the frozen 260e mean path materially, so the decomposition itself is not yet clean.
