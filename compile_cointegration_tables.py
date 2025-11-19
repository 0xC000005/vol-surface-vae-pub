"""
Compile comprehensive co-integration tables from summary_comparison.csv
for MILESTONE_PRESENTATION.md

Organizes data into 4 tables matching RMSE format:
- Table 7: Oracle (Ground Truth Latent)
- Table 8: VAE Prior (z~N(0,1))
- Table 9: Econometric Baseline
- Table 10: Ground Truth (Real Market)
"""
import pandas as pd
import numpy as np

# Read co-integration summary
df = pd.read_csv("tables/cointegration_preservation/summary_comparison.csv")

print("=" * 80)
print("CO-INTEGRATION TABLE COMPILATION")
print("=" * 80)
print()

# Helper function to extract percentage
def get_pct(model, period, horizon=None):
    """Get percentage co-integrated for given model/period/horizon."""
    if horizon is None:
        # Ground Truth (no horizon)
        row = df[(df['model'] == model) & (df['period'] == period)]
    else:
        # VAE models or Econometric (with horizon)
        period_key = f"{period}_{horizon}"
        row = df[(df['model'] == model) & (df['period'] == period_key)]

    if len(row) == 0:
        return None
    return row['pct_cointegrated'].values[0]

def get_count(model, period, horizon=None):
    """Get count co-integrated for given model/period/horizon."""
    if horizon is None:
        row = df[(df['model'] == model) & (df['period'] == period)]
    else:
        period_key = f"{period}_{horizon}"
        row = df[(df['model'] == model) & (df['period'] == period_key)]

    if len(row) == 0:
        return None
    return int(row['n_cointegrated'].values[0])

# ============================================================================
# Table 7: Oracle (Ground Truth Latent) - Co-Integration Preservation
# ============================================================================

print("TABLE 7: Oracle (Ground Truth Latent) - Co-Integration Preservation")
print("=" * 80)
print()

oracle_data = []
for period, period_label, days in [
    ('in_sample', 'In-Sample (2004-2019)', 4000),
    ('crisis', 'Crisis (2008-2010)', 765),
    ('out_of_sample', 'OOS (2019-2023)', 820)
]:
    h1 = get_pct('VAE_Oracle', period, 'h1')
    h7 = get_pct('VAE_Oracle', period, 'h7')
    h14 = get_pct('VAE_Oracle', period, 'h14')
    h30 = get_pct('VAE_Oracle', period, 'h30')

    # Average across horizons
    horizons = [h1, h7, h14, h30]
    avg = np.mean([h for h in horizons if h is not None])

    oracle_data.append({
        'Period': period_label,
        'H1': f"{h1:.0f}%" if h1 is not None else 'N/A',
        'H7': f"{h7:.0f}%" if h7 is not None else 'N/A',
        'H14': f"{h14:.0f}%" if h14 is not None else 'N/A',
        'H30': f"{h30:.0f}%" if h30 is not None else 'N/A',
        'Average': f"{avg:.0f}%",
        'Days': days
    })

oracle_df = pd.DataFrame(oracle_data)
print(oracle_df.to_string(index=False))
print()
print("Note: Grid-level statistics not available (data aggregated at surface level)")
print()

# ============================================================================
# Table 8: VAE Prior (z~N(0,1)) - Co-Integration Preservation
# ============================================================================

print("=" * 80)
print("TABLE 8: VAE Prior (z~N(0,1)) - Co-Integration Preservation")
print("=" * 80)
print()

prior_data = []
for period, period_label, days in [
    ('in_sample', 'In-Sample (2004-2019)', 4000),
    ('crisis', 'Crisis (2008-2010)', 765),
    ('out_of_sample', 'OOS (2019-2023)', 820)
]:
    h1 = get_pct('VAE_Prior', period, 'h1')
    h7 = get_pct('VAE_Prior', period, 'h7')
    h14 = get_pct('VAE_Prior', period, 'h14')
    h30 = get_pct('VAE_Prior', period, 'h30')

    horizons = [h1, h7, h14, h30]
    avg = np.mean([h for h in horizons if h is not None])

    prior_data.append({
        'Period': period_label,
        'H1': f"{h1:.0f}%" if h1 is not None else 'N/A',
        'H7': f"{h7:.0f}%" if h7 is not None else 'N/A',
        'H14': f"{h14:.0f}%" if h14 is not None else 'N/A',
        'H30': f"{h30:.0f}%" if h30 is not None else 'N/A',
        'Average': f"{avg:.0f}%",
        'Days': days
    })

prior_df = pd.DataFrame(prior_data)
print(prior_df.to_string(index=False))
print()
print("Note: Oracle → VAE Prior degradation analysis:")
for period in ['In-Sample', 'Crisis', 'OOS']:
    o_avg = float(oracle_df[oracle_df['Period'].str.contains(period.split()[0])]['Average'].values[0].strip('%'))
    p_avg = float(prior_df[prior_df['Period'].str.contains(period.split()[0])]['Average'].values[0].strip('%'))
    diff = p_avg - o_avg
    print(f"  {period}: {diff:+.0f} pp")
print()

# ============================================================================
# Table 9: Econometric Baseline - Co-Integration Preservation
# ============================================================================

print("=" * 80)
print("TABLE 9: Econometric Baseline - Co-Integration Preservation")
print("=" * 80)
print()

econ_data = []
for period, period_label, days in [
    ('crisis', 'Crisis (2008-2010)', 765),
    ('out_of_sample', 'OOS (2019-2023)', 820)
]:
    h1 = get_pct('Econometric', period, 'h1')
    h7 = get_pct('Econometric', period, 'h7')
    h14 = get_pct('Econometric', period, 'h14')
    h30 = get_pct('Econometric', period, 'h30')

    horizons = [h1, h7, h14, h30]
    avg = np.mean([h for h in horizons if h is not None])

    econ_data.append({
        'Period': period_label,
        'H1': f"{h1:.0f}%" if h1 is not None else 'N/A',
        'H7': f"{h7:.0f}%" if h7 is not None else 'N/A',
        'H14': f"{h14:.0f}%" if h14 is not None else 'N/A',
        'H30': f"{h30:.0f}%" if h30 is not None else 'N/A',
        'Average': f"{avg:.0f}%",
        'Days': days
    })

econ_df = pd.DataFrame(econ_data)
print(econ_df.to_string(index=False))
print()
print("Note: Econometric always achieves 100% co-integration (forced by linear EWMA model)")
print()

# ============================================================================
# Table 10: Ground Truth (Real Market) - Co-Integration Discovery
# ============================================================================

print("=" * 80)
print("TABLE 10: Ground Truth (Real Market) - Co-Integration Discovery")
print("=" * 80)
print()

gt_data = []
for period, period_label in [
    ('in_sample', 'In-Sample (2004-2019)'),
    ('crisis', 'Crisis (2008-2010)'),
    ('out_of_sample', 'OOS (2019-2023)')
]:
    pct = get_pct('Ground Truth', period)
    count = get_count('Ground Truth', period)

    # Spatial pattern (from known analysis)
    if period == 'crisis':
        failed = '4 Deep ITM calls'
        spatial = 'Row 4: (4,1)-(4,4)'
        justification = 'Liquidity dries up, delta≈1, IV decouples'
    elif period == 'out_of_sample':
        failed = '1 grid point'
        spatial = 'TBD (need spatial analysis)'
        justification = 'Regime-specific breakdown'
    else:
        failed = 'None'
        spatial = 'All pass'
        justification = 'Normal market conditions'

    gt_data.append({
        'Period': period_label,
        'Preservation': f"{pct:.0f}% ({count}/25)",
        'Failed Grids': failed,
        'Spatial Pattern': spatial,
        'Economic Justification': justification
    })

gt_df = pd.DataFrame(gt_data)
print(gt_df.to_string(index=False))
print()

# ============================================================================
# Generate Markdown Tables
# ============================================================================

print("=" * 80)
print("MARKDOWN TABLES FOR MILESTONE_PRESENTATION.md")
print("=" * 80)
print()

print("**Table 7: Oracle (Ground Truth Latent) - Co-Integration Preservation**")
print()
print("| Period | H1 | H7 | H14 | H30 | **Average** | Days |")
print("|--------|-----|-----|------|------|-------------|------|")
for _, row in oracle_df.iterrows():
    print(f"| **{row['Period']}** | {row['H1']} | {row['H7']} | {row['H14']} | {row['H30']} | **{row['Average']}** | {row['Days']:,} |")
print()

print("**Table 8: VAE Prior (z~N(0,1)) - Co-Integration Preservation**")
print()
print("| Period | H1 | H7 | H14 | H30 | **Average** | Days |")
print("|--------|-----|-----|------|------|-------------|------|")
for _, row in prior_df.iterrows():
    print(f"| **{row['Period']}** | {row['H1']} | {row['H7']} | {row['H14']} | {row['H30']} | **{row['Average']}** | {row['Days']:,} |")
print()

print("**Table 9: Econometric Baseline - Co-Integration Preservation**")
print()
print("| Period | H1 | H7 | H14 | H30 | **Average** | Days |")
print("|--------|-----|-----|------|------|-------------|------|")
for _, row in econ_df.iterrows():
    print(f"| **{row['Period']}** | {row['H1']} | {row['H7']} | {row['H14']} | {row['H30']} | **{row['Average']}** | {row['Days']:,} |")
print()

print("**Table 10: Ground Truth (Real Market) - Co-Integration Discovery**")
print()
print("| Period | Preservation | Failed Grids | Spatial Pattern | Economic Justification |")
print("|--------|-------------|--------------|-----------------|------------------------|")
for _, row in gt_df.iterrows():
    print(f"| **{row['Period']}** | {row['Preservation']} | {row['Failed Grids']} | {row['Spatial Pattern']} | {row['Economic Justification']} |")
print()

print("=" * 80)
print("CO-INTEGRATION TABLE COMPILATION COMPLETE")
print("=" * 80)
