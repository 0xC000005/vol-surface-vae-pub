"""
Add observed VIX from Yahoo Finance to the existing multi-factor data files.

This is the focused updater for the local factor panel. It preserves the
existing factor columns and appends/overwrites:
  - levels column: vix
  - returns column: vix_logret
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


def _naive_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(index))
    if out.tz is not None:
        out = out.tz_convert(None)
    return out


def fetch_yahoo_close(ticker: str, start: str, end: str, name: str) -> pd.Series:
    frame = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if frame.empty:
        raise RuntimeError(f"Yahoo Finance returned no rows for {ticker!r}")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    if "Close" not in frame.columns:
        raise RuntimeError(f"Yahoo Finance result for {ticker!r} has no Close column")
    close = pd.to_numeric(frame["Close"].squeeze(), errors="coerce")
    close.name = name
    close.index = _naive_datetime_index(close.index)
    close = close.replace([np.inf, -np.inf], np.nan).dropna()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    if close.empty:
        raise RuntimeError(f"Yahoo Finance Close series for {ticker!r} is empty after cleaning")
    if (close <= 0.0).any():
        raise RuntimeError(f"Yahoo Finance Close series for {ticker!r} contains non-positive values")
    return close.astype(np.float64)


def align_external_level(
    series: pd.Series,
    reference_index: pd.DatetimeIndex,
    *,
    ffill_limit: int,
) -> pd.Series:
    aligned = series.reindex(reference_index, method="ffill", limit=ffill_limit)
    if aligned.isna().any():
        missing_dates = aligned.index[aligned.isna()][:5]
        preview = ", ".join(str(date.date()) for date in missing_dates)
        raise RuntimeError(
            f"{series.name} has {int(aligned.isna().sum())} missing values after "
            f"alignment; first missing dates: {preview}"
        )
    return aligned.astype(np.float64)


def add_log_return_factor(
    levels: pd.DataFrame,
    returns: pd.DataFrame,
    level_series: pd.Series,
    *,
    factor_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not levels.index.equals(returns.index):
        raise ValueError("levels and returns must have the same index")
    aligned = align_external_level(
        level_series,
        pd.DatetimeIndex(levels.index),
        ffill_limit=5,
    )
    if (aligned <= 0.0).any():
        raise ValueError(f"{factor_name} must be positive for log-return transform")

    updated_levels = levels.copy()
    updated_returns = returns.copy()
    updated_levels[factor_name] = aligned
    updated_returns[f"{factor_name}_logret"] = np.log(aligned / aligned.shift(1))
    return updated_levels, updated_returns


def save_factor_files(
    levels: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    levels_path: Path,
    returns_path: Path,
    npz_path: Path,
) -> None:
    levels.to_parquet(levels_path)
    returns.to_parquet(returns_path)
    np.savez_compressed(
        npz_path,
        dates=pd.DatetimeIndex(levels.index).values.astype("datetime64[D]"),
        levels=levels.to_numpy(dtype=np.float32),
        level_columns=np.asarray(list(levels.columns)),
        returns=returns.to_numpy(dtype=np.float32),
        return_columns=np.asarray(list(returns.columns)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels_path", default="data/multi_factor_levels.parquet")
    parser.add_argument("--returns_path", default="data/multi_factor_returns.parquet")
    parser.add_argument("--npz_path", default="data/multi_factor_data.npz")
    parser.add_argument("--ticker", default="^VIX")
    parser.add_argument("--name", default="vix")
    args = parser.parse_args()

    levels_path = Path(args.levels_path)
    returns_path = Path(args.returns_path)
    npz_path = Path(args.npz_path)

    levels = pd.read_parquet(levels_path)
    returns = pd.read_parquet(returns_path)
    levels.index = _naive_datetime_index(levels.index)
    returns.index = _naive_datetime_index(returns.index)
    start = str((levels.index.min() - pd.Timedelta(days=7)).date())
    end = str((levels.index.max() + pd.Timedelta(days=7)).date())

    print(f"Fetching {args.name} from Yahoo Finance {args.ticker}: {start} to {end}")
    observed = fetch_yahoo_close(args.ticker, start=start, end=end, name=args.name)
    levels, returns = add_log_return_factor(
        levels,
        returns,
        observed,
        factor_name=args.name,
    )
    save_factor_files(
        levels,
        returns,
        levels_path=levels_path,
        returns_path=returns_path,
        npz_path=npz_path,
    )
    print(
        f"Saved {args.name}: levels={levels.shape}, returns={returns.shape}, "
        f"first={levels[args.name].iloc[0]:.4f}, last={levels[args.name].iloc[-1]:.4f}"
    )


if __name__ == "__main__":
    main()
