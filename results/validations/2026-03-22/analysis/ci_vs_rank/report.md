# CI Coverage vs Factor Structure (Rank) Correlation Analysis

**Date**: 2026-03-22
**Models analyzed**: 47

## Executive Summary

**INDEPENDENT (Spearman r=-0.036, p=0.8115). CI and rank are SEPARATE problems.**

- CI 90% vs Rank Ratio: Spearman r=-0.036, p=0.8115 (n=47)
- CI 90% vs Gen Eff Rank: Spearman r=-0.036, p=0.8115 (n=47)
- CI 90% vs KS Daily: Spearman r=+0.306, p=0.0365 (n=47)
- Rank Ratio vs KS Daily: Spearman r=-0.243, p=0.0994 (n=47)

## Priority Model Details

| Model | CI 90% | Rank Ratio | Eff Rank | Corr Ratio | KS Daily | Kurtosis | WF % | Suites |
|-------|--------|------------|----------|------------|----------|----------|------|--------|
| 144a_final_30d | 0.778 | 0.301 | 1.51 | 1.257 | 17/25 | 0.800 | 0.054 | 5/9 |
| 144b_best_30d | 0.740 | 0.293 | 1.47 | 1.271 | 22/25 | 1.049 | 0.083 | 5/9 |
| 144c_best_30d | 0.723 | 0.320 | 1.61 | 0.194 | 11/25 | 1.546 | 0.093 | 5/9 |
| 145a_best_30d | 0.732 | 0.319 | 1.61 | 1.280 | 2/25 | 1.657 | 0.100 | 5/9 |
| 145c_best_30d | 0.781 | 0.319 | 1.60 | 1.258 | 4/25 | 1.018 | 0.061 | 5/9 |
| 143a_ep30_30d | 0.781 | 0.451 | 2.27 | 1.470 | 6/25 | 0.982 | 0.045 | 5/9 |
| 143a_ep1_30d | 0.761 | 0.518 | 2.61 | 1.548 | 18/25 | 1.107 | 0.065 | 5/9 |
| 139a_v2_30d | 0.874 | 0.688 | 3.46 | 1.372 | 23/25 | 0.684 | 0.019 | 5/9 |

## Key Pairwise Correlations

| Metric 1 | Metric 2 | n | Pearson r | Pearson p | Spearman r | Spearman p | Sig |
|----------|----------|---|-----------|-----------|------------|------------|-----|
| CI 90% Coverage (Overall) | Rank Ratio (gen/GT) | 47 | -0.104 | 0.4847 | -0.036 | 0.8115 | ns |
| CI 90% Coverage (Overall) | Gen Effective Rank | 47 | -0.104 | 0.4847 | -0.036 | 0.8115 | ns |
| CI 90% Coverage (Overall) | Correlation Ratio | 47 | +0.007 | 0.9631 | +0.047 | 0.7534 | ns |
| CI 90% Coverage (Overall) | KS Daily Pass Count | 47 | +0.346 | 0.0172 | +0.306 | 0.0365 | * |
| CI 90% Coverage (Overall) | Kurtosis Ratio | 47 | +0.089 | 0.5524 | +0.186 | 0.2105 | ns |
| CI 90% Coverage (Overall) | Suite Pass Count | 47 | -0.141 | 0.3444 | +0.013 | 0.9287 | ns |
| Rank Ratio (gen/GT) | KS Daily Pass Count | 47 | -0.439 | 0.0020 | -0.243 | 0.0994 | ~ |
| Rank Ratio (gen/GT) | Kurtosis Ratio | 47 | -0.348 | 0.0164 | -0.413 | 0.0039 | ** |
| Rank Ratio (gen/GT) | Suite Pass Count | 47 | -0.388 | 0.0070 | -0.461 | 0.0011 | ** |
| Gen Effective Rank | Kurtosis Ratio | 47 | -0.348 | 0.0164 | -0.413 | 0.0039 | ** |
| Gen Effective Rank | KS Daily Pass Count | 47 | -0.439 | 0.0020 | -0.243 | 0.0994 | ~ |
| KS Daily Pass Count | Kurtosis Ratio | 47 | +0.327 | 0.0248 | +0.473 | 0.0008 | *** |

## Spearman Correlation Matrix (key metrics)

|  | CI 90% Coverage (Ove | Rank Ratio (gen/GT) | Gen Effective Rank | Correlation Ratio | KS Daily Pass Count | Kurtosis Ratio | Suite Pass Count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CI 90% Coverage (Ove | +1.00* | -0.04  | -0.04  | +0.05  | +0.31* | +0.19  | +0.01  |
| Rank Ratio (gen/GT)  | -0.04  | +1.00* | +1.00* | -0.76* | -0.24  | -0.41* | -0.46* |
| Gen Effective Rank   | -0.04  | +1.00* | +1.00* | -0.76* | -0.24  | -0.41* | -0.46* |
| Correlation Ratio    | +0.05  | -0.76* | -0.76* | +1.00* | +0.34* | +0.47* | +0.44* |
| KS Daily Pass Count  | +0.31* | -0.24  | -0.24  | +0.34* | +1.00* | +0.47* | -0.04  |
| Kurtosis Ratio       | +0.19  | -0.41* | -0.41* | +0.47* | +0.47* | +1.00* | +0.25  |
| Suite Pass Count     | +0.01  | -0.46* | -0.46* | +0.44* | -0.04  | +0.25  | +1.00* |

## Top 20 Models by CI 90% Coverage

| Model | CI 90% | Rank Ratio | Eff Rank | KS Daily | Suites |
|-------|--------|------------|----------|----------|--------|
| 138a_bestcov_30d | 0.948 | 0.765 | 3.85 | 11/25 | 1/9 |
| 136a_bestcov_30d | 0.931 | 1.810 | 9.10 | 17/25 | 3/9 |
| 120b_v2_30d | 0.920 | 1.068 | 5.37 | 20/25 | 5/9 |
| 120b_v8_bestcov_30d | 0.918 | 0.975 | 4.90 | 20/25 | 3/9 |
| 120b_v7_bestcov_30d | 0.915 | 1.151 | 5.79 | 19/25 | 3/9 |
| 99m_v2_v2_30d | 0.911 | 1.395 | 7.01 | 22/25 | 4/9 |
| 133f_v2_30d | 0.900 | 4.927 | 24.78 | 12/25 | 3/9 |
| 135a_bestcov_30d | 0.900 | 2.692 | 13.54 | 1/25 | 1/9 |
| 108a_v2_30d | 0.893 | 0.831 | 4.18 | 23/25 | 4/9 |
| 120b_v6_bestcov_n100_30d | 0.886 | 0.849 | 4.27 | 22/25 | 4/9 |
| 138a_v2_bestcov_30d | 0.884 | 0.904 | 4.54 | 21/25 | 4/9 |
| 99m_v3_30d | 0.877 | 0.875 | 4.40 | 24/25 | 3/9 |
| 120b_v6_bestcov_30d | 0.875 | 0.840 | 4.23 | 22/25 | 4/9 |
| 137a_bestcov_30d | 0.875 | 0.840 | 4.23 | 22/25 | 4/9 |
| 139a_v2_30d ** | 0.874 | 0.688 | 3.46 | 23/25 | 5/9 |
| 138a_best_30d | 0.787 | 1.314 | 6.61 | 19/25 | 5/9 |
| 138a_seed456_30d | 0.787 | 1.315 | 6.61 | 19/25 | 5/9 |
| 138a_seed123_30d | 0.786 | 1.332 | 6.70 | 19/25 | 5/9 |
| 145c_best_30d ** | 0.781 | 0.319 | 1.60 | 4/25 | 5/9 |
| 143a_ep30_30d ** | 0.781 | 0.451 | 2.27 | 6/25 | 5/9 |

## Interpretation

CI coverage and factor structure (rank) are **statistically independent**.
This means:
- Factor collapse does NOT cause CI failure
- They are separate problems requiring separate solutions
- Improving rank (e.g., via CLN, transformer decoder) will NOT automatically improve CI
- CI improvement needs its own mechanism (e.g., scaling, loss tuning)
