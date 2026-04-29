import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from experiments.backfill.block_ar._panel_law_535_utils import (  # noqa: E402
    load_aligned_iv_factor_panel,
)
from scripts.add_vix_to_multi_factor_data import add_log_return_factor  # noqa: E402


def test_add_log_return_factor_appends_observed_vix_without_proxy():
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    levels = pd.DataFrame({"spx": [100.0, 101.0, 102.0]}, index=dates)
    returns = pd.DataFrame({"spx_logret": [np.nan, 0.01, 0.02]}, index=dates)
    vix = pd.Series([14.0, 15.4, 13.86], index=dates, name="vix")

    out_levels, out_returns = add_log_return_factor(
        levels,
        returns,
        vix,
        factor_name="vix",
    )

    assert list(out_levels.columns) == ["spx", "vix"]
    assert list(out_returns.columns) == ["spx_logret", "vix_logret"]
    np.testing.assert_allclose(out_levels["vix"].to_numpy(), [14.0, 15.4, 13.86])
    np.testing.assert_allclose(
        out_returns["vix_logret"].to_numpy()[1:],
        [np.log(15.4 / 14.0), np.log(13.86 / 15.4)],
    )
    assert np.isnan(out_returns["vix_logret"].iloc[0])


def test_load_aligned_iv_factor_panel_consumes_real_vix_columns(tmp_path):
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    iv_path = tmp_path / "iv.npz"
    iv_parquet = tmp_path / "iv.parquet"
    factor_levels = tmp_path / "levels.parquet"
    factor_returns = tmp_path / "returns.parquet"

    np.savez(iv_path, surface=np.asarray([[[0.20]], [[0.22]], [[0.21]]], dtype=np.float32))
    pd.DataFrame({"date": dates}).to_parquet(iv_parquet, index=False)
    pd.DataFrame(
        {
            "spx": [100.0, 101.0, 102.0],
            "vix": [14.0, 15.4, 13.86],
        },
        index=dates,
    ).to_parquet(factor_levels)
    pd.DataFrame(
        {
            "spx_logret": [np.nan, 0.01, 0.02],
            "vix_logret": [np.nan, np.log(15.4 / 14.0), np.log(13.86 / 15.4)],
        },
        index=dates,
    ).to_parquet(factor_returns)

    panel, columns, out_dates = load_aligned_iv_factor_panel(
        iv_path=str(iv_path),
        iv_parquet=str(iv_parquet),
        factor_levels_parquet=str(factor_levels),
        factor_returns_parquet=str(factor_returns),
    )

    assert list(out_dates) == list(dates)
    assert columns == [
        "iv:00",
        "factor:spx",
        "factor:vix",
        "factor:spx_logret",
        "factor:vix_logret",
    ]
    np.testing.assert_allclose(panel[:, 2], np.asarray([14.0, 15.4, 13.86], dtype=np.float32))
    np.testing.assert_allclose(
        panel[:, 4],
        np.asarray([0.0, np.log(15.4 / 14.0), np.log(13.86 / 15.4)], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
