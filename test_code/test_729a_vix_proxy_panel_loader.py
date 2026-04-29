import numpy as np
import pandas as pd
import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar._panel_law_535_utils import (
    load_aligned_iv_factor_panel,
)


def test_load_aligned_iv_factor_panel_can_add_iv_vol_proxy(tmp_path):
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    iv_path = tmp_path / "iv.npz"
    iv_parquet = tmp_path / "iv.parquet"
    factor_levels = tmp_path / "levels.parquet"
    factor_returns = tmp_path / "returns.parquet"

    np.savez(iv_path, surface=np.asarray([[[0.20]], [[0.22]], [[0.21]]], dtype=np.float32))
    pd.DataFrame(
        {
            "date": dates,
            "ttm_one_month_moneyness_pt_one": [0.20, 0.22, 0.21],
        }
    ).to_parquet(iv_parquet, index=False)
    pd.DataFrame({"spx": [100.0, 101.0, 102.0]}, index=dates).to_parquet(factor_levels)
    pd.DataFrame({"spx_logret": [0.0, 0.01, 0.02]}, index=dates).to_parquet(factor_returns)

    panel, columns, out_dates = load_aligned_iv_factor_panel(
        iv_path=str(iv_path),
        iv_parquet=str(iv_parquet),
        factor_levels_parquet=str(factor_levels),
        factor_returns_parquet=str(factor_returns),
        include_iv_vol_proxy=True,
        iv_vol_proxy_column="ttm_one_month_moneyness_pt_one",
        iv_vol_proxy_name="vix_proxy",
    )

    assert list(out_dates) == list(dates)
    assert columns == [
        "iv:00",
        "factor:spx",
        "factor:vix_proxy",
        "factor:spx_logret",
        "factor:vix_proxy_logret",
    ]
    np.testing.assert_allclose(panel[:, 2], np.asarray([0.20, 0.22, 0.21], dtype=np.float32))
    np.testing.assert_allclose(
        panel[:, 4],
        np.asarray([0.0, np.log(0.22 / 0.20), np.log(0.21 / 0.22)], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
