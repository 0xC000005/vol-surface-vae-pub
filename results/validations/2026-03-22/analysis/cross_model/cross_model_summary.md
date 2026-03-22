# Cross-Model Comparison (2026-03-22)

## Models

- **144a**: scalar vol_scale (`models/backfill/afcrps_144a/final_model.pt`)
- **144b**: per-cell scale (SESSION BEST) (`models/backfill/afcrps_144b/best_model.pt`)
- **144c**: d_model=128, per-cell scale (`models/backfill/afcrps_144c/best_model.pt`)
- **145a**: strong VS lambda=1.0 (`models/backfill/afcrps_145a/best_model.pt`)
- **145c**: DPP rank loss lambda=0.1 (`models/backfill/afcrps_145c/best_model.pt`)

## Cross-Model Metrics Table

| Model | Suites | CI 90% | Kurtosis | KS Daily | KS Levels | Rank Ratio | Corr Ratio | Window Floor |
|-------|--------|--------|----------|----------|-----------|------------|------------|--------------|
| 144a | 5/9 | 77.8% | 0.800 | 17/25 | 20/25 | 0.301 | 1.257 | ? |
| 144b | 5/9 | 74.0% | 1.049 | 22/25 | 18/25 | 0.293 | 1.271 | ? |
| 144c | 5/9 | 72.3% | 1.546 | 11/25 | 22/25 | 0.320 | 0.194 | ? |
| 145a | 5/9 | 73.2% | 1.657 | 2/25 | 16/25 | 0.319 | 1.280 | ? |
| 145c | 5/9 | 78.1% | 1.018 | 4/25 | 17/25 | 0.319 | 1.258 | ? |

## Suite Pass/Fail Matrix

| Model | S1:Surf | S2:CI | S3:Cond | S4:TS | S5:BAR | S6:Coint | S7:Regime | S8:Dist | S9:XCorr |
|-------|------|------|------|------|------|------|------|------|------|
| 144a | PASS | FAIL | PASS | PASS | PASS | PASS | FAIL | FAIL | FAIL |
| 144b | PASS | FAIL | PASS | PASS | PASS | PASS | FAIL | FAIL | FAIL |
| 144c | PASS | FAIL | PASS | PASS | PASS | PASS | FAIL | FAIL | FAIL |
| 145a | PASS | FAIL | PASS | PASS | PASS | PASS | FAIL | FAIL | FAIL |
| 145c | PASS | FAIL | PASS | PASS | PASS | PASS | FAIL | FAIL | FAIL |

## Factor Analysis (PCA on Daily Changes)

| Model | Gen Eff Rank | GT Eff Rank | Rank Ratio (PCA) | Gen PC1 Var% | GT PC1 Var% |
|-------|-------------|-------------|------------------|-------------|------------|
| 144a | 1.522 | 4.742 | 0.321 | 91.9% | 62.1% |
| 144b | 1.480 | 4.742 | 0.312 | 92.6% | 62.1% |
| 144c | 1.643 | 4.742 | 0.346 | 90.5% | 62.1% |
| 145a | 1.617 | 4.742 | 0.341 | 90.7% | 62.1% |
| 145c | 1.597 | 4.742 | 0.337 | 91.2% | 62.1% |

## Training vs Test-Time Effective Rank

| Model | Train eff_rank (last) | Test eff_rank (PCA) | Test eff_rank (summary) | Test rank_ratio (summary) |
|-------|----------------------|--------------------|-----------------------|--------------------------|
| 144a | N/A | 1.522 | 1.513 | 0.301 |
| 144b | N/A | 1.480 | 1.471 | 0.293 |
| 144c | N/A | 1.643 | 1.611 | 0.320 |
| 145a | N/A | 1.617 | 1.605 | 0.319 |
| 145c | 4.365 | 1.597 | 1.603 | 0.319 |

## Per-Cell KS Daily Failure Map

Count of models (out of 5) failing KS daily at each cell:
(Rows = moneyness, Cols = tenor)

```
  2  4  5  4  5
  1  3  4  3  4
  1  3  3  3  2
  1  2  3  3  3
  1  1  2  3  3
```

### Hardest Cells (3+ model failures)

- Cell (0,1): 4/5 failures
  - 144a: KS=0.1705 [FAIL]
  - 144b: KS=0.1496 [pass]
  - 144c: KS=0.2054 [FAIL]
  - 145a: KS=0.2098 [FAIL]
  - 145c: KS=0.1645 [FAIL]
- Cell (0,2): 5/5 failures
  - 144a: KS=0.2055 [FAIL]
  - 144b: KS=0.1559 [FAIL]
  - 144c: KS=0.2336 [FAIL]
  - 145a: KS=0.1976 [FAIL]
  - 145c: KS=0.1862 [FAIL]
- Cell (0,3): 4/5 failures
  - 144a: KS=0.1446 [pass]
  - 144b: KS=0.1612 [FAIL]
  - 144c: KS=0.1847 [FAIL]
  - 145a: KS=0.2390 [FAIL]
  - 145c: KS=0.2019 [FAIL]
- Cell (0,4): 5/5 failures
  - 144a: KS=0.2134 [FAIL]
  - 144b: KS=0.1841 [FAIL]
  - 144c: KS=0.2085 [FAIL]
  - 145a: KS=0.2357 [FAIL]
  - 145c: KS=0.2018 [FAIL]
- Cell (1,1): 3/5 failures
  - 144a: KS=0.1113 [pass]
  - 144b: KS=0.1220 [pass]
  - 144c: KS=0.1793 [FAIL]
  - 145a: KS=0.1903 [FAIL]
  - 145c: KS=0.1647 [FAIL]
- Cell (1,2): 4/5 failures
  - 144a: KS=0.1706 [FAIL]
  - 144b: KS=0.1473 [pass]
  - 144c: KS=0.2168 [FAIL]
  - 145a: KS=0.1916 [FAIL]
  - 145c: KS=0.1845 [FAIL]
- Cell (1,3): 3/5 failures
  - 144a: KS=0.1408 [pass]
  - 144b: KS=0.1404 [pass]
  - 144c: KS=0.2048 [FAIL]
  - 145a: KS=0.1914 [FAIL]
  - 145c: KS=0.1889 [FAIL]
- Cell (1,4): 4/5 failures
  - 144a: KS=0.1585 [FAIL]
  - 144b: KS=0.1364 [pass]
  - 144c: KS=0.1620 [FAIL]
  - 145a: KS=0.2164 [FAIL]
  - 145c: KS=0.1768 [FAIL]
- Cell (2,1): 3/5 failures
  - 144a: KS=0.1143 [pass]
  - 144b: KS=0.1185 [pass]
  - 144c: KS=0.1755 [FAIL]
  - 145a: KS=0.1740 [FAIL]
  - 145c: KS=0.1601 [FAIL]
- Cell (2,2): 3/5 failures
  - 144a: KS=0.1296 [pass]
  - 144b: KS=0.1456 [pass]
  - 144c: KS=0.2112 [FAIL]
  - 145a: KS=0.2003 [FAIL]
  - 145c: KS=0.1966 [FAIL]
- Cell (2,3): 3/5 failures
  - 144a: KS=0.0991 [pass]
  - 144b: KS=0.1404 [pass]
  - 144c: KS=0.1958 [FAIL]
  - 145a: KS=0.1967 [FAIL]
  - 145c: KS=0.2001 [FAIL]
- Cell (3,2): 3/5 failures
  - 144a: KS=0.0993 [pass]
  - 144b: KS=0.1397 [pass]
  - 144c: KS=0.1950 [FAIL]
  - 145a: KS=0.1998 [FAIL]
  - 145c: KS=0.2023 [FAIL]
- Cell (3,3): 3/5 failures
  - 144a: KS=0.1104 [pass]
  - 144b: KS=0.1440 [pass]
  - 144c: KS=0.2044 [FAIL]
  - 145a: KS=0.2075 [FAIL]
  - 145c: KS=0.2073 [FAIL]
- Cell (3,4): 3/5 failures
  - 144a: KS=0.1908 [FAIL]
  - 144b: KS=0.0850 [pass]
  - 144c: KS=0.0957 [pass]
  - 145a: KS=0.2050 [FAIL]
  - 145c: KS=0.1977 [FAIL]
- Cell (4,3): 3/5 failures
  - 144a: KS=0.1356 [pass]
  - 144b: KS=0.1237 [pass]
  - 144c: KS=0.1684 [FAIL]
  - 145a: KS=0.2091 [FAIL]
  - 145c: KS=0.2286 [FAIL]
- Cell (4,4): 3/5 failures
  - 144a: KS=0.1672 [FAIL]
  - 144b: KS=0.1036 [pass]
  - 144c: KS=0.0924 [pass]
  - 145a: KS=0.1820 [FAIL]
  - 145c: KS=0.1594 [FAIL]

## PC1 Loading Patterns (5x5)

Ground truth PC1 loadings:
```
  +0.108  +0.229  +0.235  +0.020  +0.203
  +0.198  +0.245  +0.237  +0.111  +0.129
  +0.224  +0.249  +0.242  +0.200  -0.020
  +0.224  +0.250  +0.244  +0.218  +0.040
  +0.200  +0.238  +0.240  +0.227  +0.129
```

144a PC1 loadings:
```
  +0.161  +0.206  +0.202  +0.165  +0.198
  +0.202  +0.208  +0.204  +0.165  +0.205
  +0.208  +0.207  +0.208  +0.204  +0.193
  +0.207  +0.205  +0.207  +0.208  +0.204
  -0.201  -0.203  +0.206  +0.208  +0.207
```

144b PC1 loadings:
```
  +0.169  +0.206  +0.202  +0.165  +0.197
  +0.203  +0.207  +0.205  +0.165  +0.204
  +0.207  +0.207  +0.206  +0.203  +0.194
  +0.207  +0.207  +0.207  +0.206  +0.203
  -0.200  -0.200  +0.206  +0.206  +0.207
```

144c PC1 loadings:
```
  +0.167  -0.203  +0.202  +0.165  +0.190
  -0.197  -0.206  +0.206  +0.167  -0.199
  +0.208  -0.205  -0.202  +0.204  +0.193
  -0.207  +0.208  +0.208  +0.207  +0.205
  +0.208  -0.206  +0.208  +0.208  +0.209
```

145a PC1 loadings:
```
  +0.163  +0.207  +0.202  +0.167  +0.198
  +0.201  +0.208  +0.207  +0.167  +0.207
  +0.208  +0.208  +0.209  +0.206  +0.194
  +0.208  +0.208  +0.209  +0.208  +0.203
  -0.197  -0.178  +0.208  +0.208  +0.207
```

145c PC1 loadings:
```
  +0.171  +0.206  +0.202  +0.167  +0.196
  +0.201  +0.208  +0.205  +0.163  +0.203
  +0.208  +0.208  +0.207  +0.204  +0.193
  +0.208  +0.208  +0.207  +0.206  +0.205
  -0.197  -0.196  +0.208  +0.207  +0.208
```
