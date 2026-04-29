"""
Download multi-factor financial data for the conditional scenario generator.

Assets:
  Equities:  SPX, Nikkei 225
  FX:        USDCAD, JPYUSD, DXY
  Commodities: Gold, Copper, Wheat, Crude Oil
  Rates:     US 2Y, US 10Y
  Credit:    AAA spread, BBB spread
  Volatility: VIX

Sources: FRED (rates, FX, credit) + yfinance (equities, commodities, FX backup)
Date range: aligned to existing IV surface data (2000-01-03 to 2023-02-27)
"""

import numpy as np
import pandas as pd
from fredapi import Fred
import yfinance as yf
from pathlib import Path
import time
import warnings
warnings.filterwarnings("ignore")

DELAY = 3  # seconds between API calls to avoid rate limiting

FRED_API_KEY = "3a586c5e3fdd267d9338357f970f0e38"
START = "2000-01-01"
END = "2023-03-01"
DATA_DIR = Path("data")

fred = Fred(api_key=FRED_API_KEY)


def fetch_fred(series_id: str, name: str) -> pd.Series:
    """Fetch a FRED series with rate-limit delay."""
    print(f"  FRED: {name} ({series_id})...", end=" ", flush=True)
    time.sleep(DELAY)
    try:
        s = fred.get_series(series_id, observation_start=START, observation_end=END)
        s.name = name
        print(f"OK — {len(s.dropna())} obs", flush=True)
        return s
    except Exception as e:
        print(f"FAILED: {e}", flush=True)
        return pd.Series(dtype=float, name=name)


def fetch_yf(ticker: str, name: str) -> pd.Series:
    """Fetch yfinance close price with rate-limit delay."""
    print(f"  yfinance: {name} ({ticker})...", end=" ", flush=True)
    time.sleep(DELAY)
    try:
        df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        s = df["Close"].squeeze()
        s.name = name
        s.index = s.index.tz_localize(None) if s.index.tz else s.index
        print(f"OK — {len(s.dropna())} obs", flush=True)
        return s
    except Exception as e:
        print(f"FAILED: {e}", flush=True)
        return pd.Series(dtype=float, name=name)


def main():
    # Load existing IV surface dates as reference index
    iv_data = np.load(DATA_DIR / "vol_surface_with_ret.npz")
    ohlcv = pd.read_csv(DATA_DIR / "market_data_ohlcv.csv", parse_dates=["Date"])
    ref_dates = ohlcv["Date"].values
    ref_index = pd.DatetimeIndex(ref_dates)
    print(f"Reference: {len(ref_index)} trading days, {ref_index[0].date()} to {ref_index[-1].date()}")
    print()

    all_series = {}

    # === EQUITIES ===
    print("=== Equities ===")
    # SPX — already have, but get clean version for alignment
    all_series["spx"] = fetch_yf("^GSPC", "spx")
    all_series["nikkei"] = fetch_yf("^N225", "nikkei")

    # === FX ===
    print("\n=== FX ===")
    # FRED FX (more reliable for longer history)
    all_series["usdcad"] = fetch_fred("DEXCAUS", "usdcad")
    all_series["usdjpy"] = fetch_fred("DEXJPUS", "usdjpy")  # JPY per USD
    # DXY from yfinance (FRED has trade-weighted but not DXY exactly)
    all_series["dxy"] = fetch_yf("DX-Y.NYB", "dxy")

    # === COMMODITIES ===
    print("\n=== Commodities ===")
    all_series["gold"] = fetch_fred("GOLDAMGBD228NLBM", "gold")  # London PM fix
    all_series["copper"] = fetch_yf("HG=F", "copper")  # COMEX futures
    all_series["wheat"] = fetch_yf("ZW=F", "wheat")  # CBOT futures
    all_series["crude_oil"] = fetch_fred("DCOILWTICO", "crude_oil")  # WTI daily

    # === RATES ===
    print("\n=== Rates ===")
    all_series["us2y"] = fetch_fred("DGS2", "us2y")
    all_series["us10y"] = fetch_fred("DGS10", "us10y")

    # === CREDIT SPREADS ===
    print("\n=== Credit Spreads ===")
    # ICE BofA OAS indices
    all_series["aaa_oas"] = fetch_fred("BAMLC0A1CAAA", "aaa_oas")
    all_series["bbb_oas"] = fetch_fred("BAMLC0A4CBBB", "bbb_oas")

    # === VOLATILITY ===
    print("\n=== Volatility ===")
    all_series["vix"] = fetch_yf("^VIX", "vix")

    # === BUILD ALIGNED DATAFRAME ===
    print("\n=== Aligning to reference dates ===")

    # Combine all into a single dataframe
    combined = pd.DataFrame(index=ref_index)
    for name, series in all_series.items():
        if len(series) == 0:
            print(f"  SKIP {name} — no data")
            continue
        s = series.copy()
        s.index = pd.DatetimeIndex(s.index)
        # Reindex to reference dates, forward-fill (max 5 days for holidays)
        aligned = s.reindex(ref_index, method="ffill", limit=5)
        combined[name] = aligned
        n_missing = aligned.isna().sum()
        pct = 100 * n_missing / len(aligned)
        print(f"  {name}: {len(aligned) - n_missing}/{len(aligned)} filled ({pct:.1f}% missing)")

    # Report overall coverage
    print(f"\n=== Coverage Summary ===")
    print(f"Total columns: {len(combined.columns)}")
    print(f"Date range: {combined.index[0].date()} to {combined.index[-1].date()}")
    print(f"Rows: {len(combined)}")
    print()
    for col in combined.columns:
        first_valid = combined[col].first_valid_index()
        last_valid = combined[col].last_valid_index()
        n_valid = combined[col].notna().sum()
        print(f"  {col:12s}: {n_valid:5d} valid, starts {first_valid.date() if first_valid else 'N/A'}")

    # === COMPUTE RETURNS ===
    print("\n=== Computing returns ===")
    returns = pd.DataFrame(index=ref_index)
    for col in combined.columns:
        if col in ("us2y", "us10y", "aaa_oas", "bbb_oas"):
            # Rates/spreads: use first differences (already in percentage points)
            returns[f"{col}_diff"] = combined[col].diff()
        else:
            # Prices: use log returns
            returns[f"{col}_logret"] = np.log(combined[col] / combined[col].shift(1))

    # === SAVE ===
    output_path = DATA_DIR / "multi_factor_data.npz"
    save_dict = {
        "dates": ref_index.values.astype("datetime64[D]"),
        "levels": combined.values.astype(np.float32),
        "level_columns": np.array(list(combined.columns)),
        "returns": returns.values.astype(np.float32),
        "return_columns": np.array(list(returns.columns)),
    }
    np.savez_compressed(output_path, **save_dict)
    print(f"\nSaved to {output_path}")
    print(f"  levels: {combined.values.shape} ({list(combined.columns)})")
    print(f"  returns: {returns.values.shape} ({list(returns.columns)})")

    # Also save as parquet for easy inspection
    combined.to_parquet(DATA_DIR / "multi_factor_levels.parquet")
    returns.to_parquet(DATA_DIR / "multi_factor_returns.parquet")
    print(f"  Also saved .parquet versions")


if __name__ == "__main__":
    main()
