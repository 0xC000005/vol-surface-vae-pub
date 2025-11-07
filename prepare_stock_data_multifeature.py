"""
Prepare multi-feature stock data for 1D VAE backfilling model.

Converts market_data_ohlcv.csv to stock_returns_multifeature.npz with:
- 3 stocks: AMZN, MSFT, SP500
- 4 features per stock: log_return, realized_volatility, volume_change, intraday_range
- Total: 12 features (1 target + 11 extra features)

Features:
    1. Log return: log(Close_t / Close_{t-1})
    2. Realized volatility: std(returns[t-20:t]) - 20-day rolling window
    3. Volume change: log(Volume_t / Volume_{t-1})
    4. Intraday range: log(High_t / Low_t) - Parkinson volatility estimator

Output:
    data/stock_returns_multifeature.npz with keys:
        - amzn_return: (N, 1) - AMZN log return (target)
        - extra_features: (N, 11) - Other 11 features (conditioning)
        - amzn_features: (N, 4) - All AMZN features
        - msft_features: (N, 4) - All MSFT features
        - sp500_features: (N, 4) - All SP500 features
        - dates: (N,) - Trading dates
"""

import pandas as pd
import numpy as np
import os

# Configuration
INPUT_FILE = "data/market_data_ohlcv.csv"
OUTPUT_FILE = "data/stock_returns_multifeature.npz"
ROLLING_WINDOW = 20  # For realized volatility

print("=" * 80)
print("PREPARING MULTI-FEATURE STOCK DATA FOR 1D VAE BACKFILLING")
print("=" * 80)
print(f"Input: {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print(f"Realized volatility window: {ROLLING_WINDOW} days")
print()

# Load data
print("Loading OHLCV data...")
df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)

print(f"  Loaded {len(df)} rows")
print(f"  Date range: {df.index[0]} to {df.index[-1]}")
print(f"  Columns: {list(df.columns)}")
print()


def compute_features(ticker, df):
    """
    Compute 4 features for a given ticker.

    Args:
        ticker: Stock ticker (e.g., 'AMZN', 'MSFT', '^GSPC')
        df: DataFrame with OHLCV columns like 'AMZN_Open', 'AMZN_High', etc.

    Returns:
        DataFrame with columns: [return, vol, volume_change, range]
    """
    # Extract OHLCV for this ticker
    open_col = f"{ticker}_Open"
    high_col = f"{ticker}_High"
    low_col = f"{ticker}_Low"
    close_col = f"{ticker}_Close"
    volume_col = f"{ticker}_Volume"

    # 1. Log return
    close_prices = df[close_col]
    log_return = np.log(close_prices / close_prices.shift(1))

    # 2. Realized volatility (20-day rolling std of returns)
    realized_vol = log_return.rolling(window=ROLLING_WINDOW).std()

    # 3. Volume change
    volume = df[volume_col]
    volume_change = np.log(volume / volume.shift(1))

    # 4. Intraday range (Parkinson volatility estimator)
    high = df[high_col]
    low = df[low_col]
    intraday_range = np.log(high / low)

    # Combine into DataFrame
    features = pd.DataFrame({
        'return': log_return,
        'vol': realized_vol,
        'volume_change': volume_change,
        'range': intraday_range
    })

    return features


# Compute features for each stock
print("Computing features for each stock...")
print("-" * 80)

amzn_features = compute_features('AMZN', df)
print(f"AMZN features computed: {list(amzn_features.columns)}")

msft_features = compute_features('MSFT', df)
print(f"MSFT features computed: {list(msft_features.columns)}")

sp500_features = compute_features('^GSPC', df)
print(f"SP500 features computed: {list(sp500_features.columns)}")
print()

# Combine all features
print("Combining features...")
all_features = pd.concat([
    amzn_features.add_prefix('amzn_'),
    msft_features.add_prefix('msft_'),
    sp500_features.add_prefix('sp500_')
], axis=1)

print(f"  Total features: {len(all_features.columns)}")
print(f"  Before dropping NaN: {len(all_features)} rows")

# Drop rows with NaN (from rolling window and log calculations)
all_features_clean = all_features.dropna()
dates_clean = all_features_clean.index

print(f"  After dropping NaN: {len(all_features_clean)} rows")
print(f"  Date range: {dates_clean[0]} to {dates_clean[-1]}")
print()

# Extract individual stock features
amzn_cols = ['amzn_return', 'amzn_vol', 'amzn_volume_change', 'amzn_range']
msft_cols = ['msft_return', 'msft_vol', 'msft_volume_change', 'msft_range']
sp500_cols = ['sp500_return', 'sp500_vol', 'sp500_volume_change', 'sp500_range']

amzn_feat_arr = all_features_clean[amzn_cols].values  # (N, 4)
msft_feat_arr = all_features_clean[msft_cols].values  # (N, 4)
sp500_feat_arr = all_features_clean[sp500_cols].values  # (N, 4)

# Create target and extra features
amzn_return = all_features_clean[['amzn_return']].values  # (N, 1) - Target
extra_features = all_features_clean[
    ['amzn_vol', 'amzn_volume_change', 'amzn_range'] +
    msft_cols + sp500_cols
].values  # (N, 11) - Conditioning features

dates_arr = dates_clean.values

# Print statistics for each stock
print("Feature statistics:")
print("=" * 80)

for stock_name, feat_arr, cols in [
    ('AMZN', amzn_feat_arr, amzn_cols),
    ('MSFT', msft_feat_arr, msft_cols),
    ('SP500', sp500_feat_arr, sp500_cols)
]:
    print(f"\n{stock_name}:")
    print("-" * 40)
    for i, col_name in enumerate(cols):
        feat_name = col_name.split('_', 1)[1]  # Remove ticker prefix
        values = feat_arr[:, i]
        print(f"  {feat_name:15s}: mean={values.mean():8.6f}, std={values.std():8.6f}, "
              f"min={values.min():8.6f}, max={values.max():8.6f}")

print()
print("=" * 80)

# Save to NPZ
print("Saving to NPZ file...")
np.savez(
    OUTPUT_FILE,
    # Model-ready format
    amzn_return=amzn_return,  # (N, 1) - Target
    extra_features=extra_features,  # (N, 11) - Conditioning
    # Individual stock features (for analysis)
    amzn_features=amzn_feat_arr,  # (N, 4)
    msft_features=msft_feat_arr,  # (N, 4)
    sp500_features=sp500_feat_arr,  # (N, 4)
    # Dates
    dates=dates_arr,
)

print(f"Saved: {OUTPUT_FILE}")
print()

# Verify saved data
print("Verifying saved data...")
loaded = np.load(OUTPUT_FILE)
print("  Keys:", list(loaded.keys()))
print("  amzn_return shape:", loaded["amzn_return"].shape, "- Target")
print("  extra_features shape:", loaded["extra_features"].shape, "- Conditioning (11 features)")
print("  amzn_features shape:", loaded["amzn_features"].shape)
print("  msft_features shape:", loaded["msft_features"].shape)
print("  sp500_features shape:", loaded["sp500_features"].shape)
print("  dates shape:", loaded["dates"].shape)
print()

print("=" * 80)
print("DATA PREPARATION COMPLETE")
print("=" * 80)
print()
print("Summary:")
print(f"  - Total samples: {len(amzn_return)}")
print(f"  - Date range: {dates_clean[0].strftime('%Y-%m-%d')} to {dates_clean[-1].strftime('%Y-%m-%d')}")
print(f"  - Target: AMZN return (1 feature)")
print(f"  - Extra features: AMZN vol/volume/range + MSFT×4 + SP500×4 (11 features)")
print(f"  - Total input dimensionality: 12 features")
print()
print("Feature breakdown:")
print("  Target (1):")
print("    - AMZN log return")
print("  Extra features (11):")
print("    - AMZN: vol, volume_change, intraday_range (3)")
print("    - MSFT: return, vol, volume_change, intraday_range (4)")
print("    - SP500: return, vol, volume_change, intraday_range (4)")
print()
print("Next steps:")
print("  1. Update train_1d_models.py to load this data")
print("  2. Set config: ex_feats_dim=11")
print("  3. Implement 80/20 backfilling training strategy")
