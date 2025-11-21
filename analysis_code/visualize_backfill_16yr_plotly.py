"""
Interactive Teacher Forcing Visualization for backfill_16yr Model

Creates a 12-panel Plotly dashboard showing:
- 3 grid points (ATM 3M, ATM 1Y, OTM Put 1Y)
- 4 horizons (h1, h7, h14, h30)
- Full period: 2004-2019 with crisis highlighting
- Interactive zoom, hover, and range selection

Outputs:
- Interactive HTML visualization
- Statistics report (CI violations, RMSE, etc.)
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

print("=" * 80)
print("INTERACTIVE TEACHER FORCING VISUALIZATION - backfill_16yr")
print("=" * 80)
print()

# ============================================================================
# Configuration
# ============================================================================

# Grid points to visualize (row, col)
GRID_POINTS = [
    (1, 2, "ATM 3-Month"),       # ATM Put, 3M maturity
    (3, 2, "ATM 1-Year"),        # ATM, 1Y maturity
    (3, 0, "OTM Put 1-Year"),    # OTM Put, 1Y maturity
]

HORIZONS = [1, 7, 14, 30]

# Crisis period
CRISIS_START = 2000
CRISIS_END = 2765

# Output directory
OUTPUT_DIR = Path("results/backfill_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Output directory: {OUTPUT_DIR}")
print()

# ============================================================================
# Load Data
# ============================================================================

print("Loading data...")

# Load reconstructions
recon_file = "models/backfill/insample_reconstruction_16yr.npz"
recon_data = np.load(recon_file)

# Load ground truth
gt_data = np.load("data/vol_surface_with_ret.npz")
vol_surf_gt = gt_data["surface"]

# Load dates (try parquet first, fallback to generating indices)
try:
    df_dates = pd.read_parquet("data/spx_vol_surface_history_full_data_fixed.parquet")
    dates_available = True
    print("✓ Loaded date information from parquet")
except:
    dates_available = False
    print("⚠ Date information not available, using indices")

print("✓ Data loaded")
print()

# ============================================================================
# Create Plotly Dashboard
# ============================================================================

print("Creating interactive dashboard...")

# Create subplot grid: 3 rows (grid points) × 4 columns (horizons)
fig = make_subplots(
    rows=len(GRID_POINTS),
    cols=len(HORIZONS),
    subplot_titles=[f"{gp[2]} - H{h}" for gp in GRID_POINTS for h in HORIZONS],
    vertical_spacing=0.08,
    horizontal_spacing=0.05,
    shared_xaxes=True,
)

# Statistics storage
stats_list = []

# Iterate through grid points and horizons
for grid_idx, (row, col, grid_name) in enumerate(GRID_POINTS):
    for hor_idx, horizon in enumerate(HORIZONS):
        subplot_row = grid_idx + 1
        subplot_col = hor_idx + 1

        print(f"  Processing {grid_name} - H{horizon}...")

        # Load reconstruction for this horizon
        recon_key = f'recon_h{horizon}'
        indices_key = f'indices_h{horizon}'

        recons = recon_data[recon_key]  # (N, 3, 5, 5)
        indices = recon_data[indices_key]  # (N,)

        # Extract quantiles for this grid point
        p05 = recons[:, 0, row, col]  # (N,)
        p50 = recons[:, 1, row, col]  # (N,)
        p95 = recons[:, 2, row, col]  # (N,)

        # Extract ground truth
        gt = vol_surf_gt[indices, row, col]  # (N,)

        # Compute violations
        below_p05 = gt < p05
        above_p95 = gt > p95
        violations = below_p05 | above_p95

        num_violations = np.sum(violations)
        total_points = len(gt)
        pct_violations = 100 * num_violations / total_points

        num_below = np.sum(below_p05)
        num_above = np.sum(above_p95)
        pct_below = 100 * num_below / total_points
        pct_above = 100 * num_above / total_points

        # RMSE
        rmse = np.sqrt(np.mean((p50 - gt) ** 2))

        # CI width
        ci_width = np.mean(p95 - p05)

        # Regime breakdown
        is_crisis = (indices >= CRISIS_START) & (indices <= CRISIS_END)
        is_normal = ~is_crisis

        crisis_violations = np.sum(violations[is_crisis]) if np.sum(is_crisis) > 0 else 0
        crisis_total = np.sum(is_crisis)
        crisis_pct = 100 * crisis_violations / crisis_total if crisis_total > 0 else 0

        normal_violations = np.sum(violations[is_normal]) if np.sum(is_normal) > 0 else 0
        normal_total = np.sum(is_normal)
        normal_pct = 100 * normal_violations / normal_total if normal_total > 0 else 0

        # Store statistics
        stats_list.append({
            'grid_point': grid_name,
            'horizon': horizon,
            'rmse': rmse,
            'ci_violations_pct': pct_violations,
            'ci_violations_count': num_violations,
            'below_p05_pct': pct_below,
            'above_p95_pct': pct_above,
            'mean_ci_width': ci_width,
            'crisis_violations_pct': crisis_pct,
            'normal_violations_pct': normal_pct,
        })

        # Create x-axis (use dates if available, otherwise indices)
        if dates_available:
            # Try to map indices to dates
            try:
                x_vals = df_dates.iloc[indices]['date'].values
                x_axis_type = 'date'
            except:
                x_vals = indices
                x_axis_type = 'linear'
        else:
            x_vals = indices
            x_axis_type = 'linear'

        # Add CI band (fill between p05 and p95)
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=p95,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip',
            ),
            row=subplot_row,
            col=subplot_col,
        )

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=p05,
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(100, 149, 237, 0.3)',
                fill='tonexty',
                name='90% CI' if subplot_row == 1 and subplot_col == 1 else None,
                showlegend=(subplot_row == 1 and subplot_col == 1),
                hovertemplate='CI: [%{y:.4f}, ' + f'{p95[0]:.4f}' + ']<extra></extra>',
            ),
            row=subplot_row,
            col=subplot_col,
        )

        # Add ground truth
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=gt,
                mode='lines',
                line=dict(color='black', width=1.5),
                name='Ground Truth' if subplot_row == 1 and subplot_col == 1 else None,
                showlegend=(subplot_row == 1 and subplot_col == 1),
                hovertemplate='GT: %{y:.4f}<extra></extra>',
            ),
            row=subplot_row,
            col=subplot_col,
        )

        # Add p50 prediction
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=p50,
                mode='lines',
                line=dict(color='cornflowerblue', width=1.5),
                name='p50 (Median)' if subplot_row == 1 and subplot_col == 1 else None,
                showlegend=(subplot_row == 1 and subplot_col == 1),
                hovertemplate='p50: %{y:.4f}<extra></extra>',
            ),
            row=subplot_row,
            col=subplot_col,
        )

        # Add violation markers (only where violations occur)
        if num_violations > 0:
            violation_x = x_vals[violations]
            violation_y = gt[violations]

            fig.add_trace(
                go.Scatter(
                    x=violation_x,
                    y=violation_y,
                    mode='markers',
                    marker=dict(color='red', size=3, opacity=0.6),
                    name='Violations' if subplot_row == 1 and subplot_col == 1 else None,
                    showlegend=(subplot_row == 1 and subplot_col == 1),
                    hovertemplate='Violation: %{y:.4f}<extra></extra>',
                ),
                row=subplot_row,
                col=subplot_col,
            )

        # Add crisis period shading
        if dates_available and x_axis_type == 'date':
            try:
                crisis_start_date = df_dates.iloc[CRISIS_START]['date']
                crisis_end_date = df_dates.iloc[CRISIS_END]['date']

                fig.add_vrect(
                    x0=crisis_start_date,
                    x1=crisis_end_date,
                    fillcolor="red",
                    opacity=0.1,
                    layer="below",
                    line_width=0,
                    row=subplot_row,
                    col=subplot_col,
                )
            except:
                pass
        else:
            fig.add_vrect(
                x0=CRISIS_START,
                x1=CRISIS_END,
                fillcolor="red",
                opacity=0.1,
                layer="below",
                line_width=0,
                row=subplot_row,
                col=subplot_col,
            )

        # Add annotation with statistics
        violation_color = 'green' if pct_violations < 12 else ('orange' if pct_violations < 20 else 'red')

        annotation_text = (
            f"RMSE: {rmse:.4f}<br>"
            f"<b style='color:{violation_color}'>Violations: {pct_violations:.1f}%</b><br>"
            f"CI Width: {ci_width:.4f}<br>"
            f"Crisis: {crisis_pct:.1f}% | Normal: {normal_pct:.1f}%"
        )

        subplot_num = (subplot_row-1)*len(HORIZONS) + subplot_col
        xref_str = 'x domain' if subplot_num == 1 else f"x{subplot_num} domain"
        yref_str = 'y domain' if subplot_num == 1 else f"y{subplot_num} domain"

        fig.add_annotation(
            text=annotation_text,
            xref=xref_str,
            yref=yref_str,
            x=0.02,
            y=0.98,
            xanchor='left',
            yanchor='top',
            showarrow=False,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor=violation_color,
            borderwidth=2,
            font=dict(size=9),
        )

# Update layout
fig.update_layout(
    title={
        'text': "backfill_16yr Model: Teacher Forcing Evaluation Across Horizons<br><sub>Interactive Dashboard - Zoom/Pan/Hover for Details</sub>",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 18}
    },
    height=1000,
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    ),
    hovermode='closest',
)

# Update axes labels
for i in range(1, len(GRID_POINTS) + 1):
    for j in range(1, len(HORIZONS) + 1):
        # Y-axis label
        fig.update_yaxes(
            title_text="Implied Volatility" if j == 1 else None,
            row=i,
            col=j,
        )

        # X-axis label
        if i == len(GRID_POINTS):  # Bottom row
            if dates_available and x_axis_type == 'date':
                fig.update_xaxes(
                    title_text="Date",
                    row=i,
                    col=j,
                )
            else:
                fig.update_xaxes(
                    title_text="Date Index",
                    row=i,
                    col=j,
                )

# Add range slider to bottom row
for j in range(1, len(HORIZONS) + 1):
    fig.update_xaxes(
        rangeslider_visible=True,
        row=len(GRID_POINTS),
        col=j,
    )

print("✓ Dashboard created")
print()

# ============================================================================
# Save Interactive HTML
# ============================================================================

html_file = OUTPUT_DIR / "backfill_16yr_teacher_forcing_full.html"
print(f"Saving interactive HTML to {html_file}...")
fig.write_html(str(html_file))
print("✓ Saved!")
print()

# ============================================================================
# Generate Statistics Report
# ============================================================================

print("Generating statistics report...")

stats_df = pd.DataFrame(stats_list)

report_file = OUTPUT_DIR / "ci_violations_report_16yr.txt"
with open(report_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("CI VIOLATIONS REPORT - backfill_16yr Model\n")
    f.write("=" * 80 + "\n\n")

    # 1. By Horizon
    f.write("1. PERFORMANCE BY HORIZON\n")
    f.write("-" * 80 + "\n")
    horizon_summary = stats_df.groupby('horizon').agg({
        'rmse': 'mean',
        'ci_violations_pct': 'mean',
        'below_p05_pct': 'mean',
        'above_p95_pct': 'mean',
        'mean_ci_width': 'mean',
    }).round(4)
    f.write(horizon_summary.to_string())
    f.write("\n\n")

    # 2. By Grid Point
    f.write("2. PERFORMANCE BY GRID POINT\n")
    f.write("-" * 80 + "\n")
    grid_summary = stats_df.groupby('grid_point').agg({
        'rmse': 'mean',
        'ci_violations_pct': 'mean',
        'below_p05_pct': 'mean',
        'above_p95_pct': 'mean',
        'mean_ci_width': 'mean',
    }).round(4)
    f.write(grid_summary.to_string())
    f.write("\n\n")

    # 3. By Regime
    f.write("3. CRISIS VS NORMAL PERIODS\n")
    f.write("-" * 80 + "\n")
    for horizon in HORIZONS:
        h_data = stats_df[stats_df['horizon'] == horizon]
        avg_crisis = h_data['crisis_violations_pct'].mean()
        avg_normal = h_data['normal_violations_pct'].mean()
        f.write(f"H{horizon:2d}: Crisis={avg_crisis:5.1f}%  Normal={avg_normal:5.1f}%  Diff={(avg_crisis-avg_normal):+5.1f}pp\n")
    f.write("\n")

    # 4. Detailed Table
    f.write("4. DETAILED STATISTICS (ALL COMBINATIONS)\n")
    f.write("-" * 80 + "\n")
    f.write(stats_df.to_string(index=False))
    f.write("\n\n")

    # 5. Summary Assessment
    f.write("5. OVERALL ASSESSMENT\n")
    f.write("-" * 80 + "\n")
    avg_violations = stats_df['ci_violations_pct'].mean()
    min_violations = stats_df['ci_violations_pct'].min()
    max_violations = stats_df['ci_violations_pct'].max()

    f.write(f"Target CI violations: ~10%\n")
    f.write(f"Average violations: {avg_violations:.2f}%\n")
    f.write(f"Range: [{min_violations:.2f}%, {max_violations:.2f}%]\n\n")

    if avg_violations <= 12:
        verdict = "✓ EXCELLENT - Well calibrated"
    elif avg_violations <= 18:
        verdict = "✓ GOOD - Acceptable calibration"
    elif avg_violations <= 25:
        verdict = "⚠ MODERATE - Needs improvement"
    else:
        verdict = "✗ POOR - Significant calibration issues"

    f.write(f"Verdict: {verdict}\n\n")

    # Worst performers
    worst_3 = stats_df.nlargest(3, 'ci_violations_pct')[['grid_point', 'horizon', 'ci_violations_pct', 'rmse']]
    f.write("Worst 3 combinations:\n")
    f.write(worst_3.to_string(index=False))
    f.write("\n\n")

    # Best performers
    best_3 = stats_df.nsmallest(3, 'ci_violations_pct')[['grid_point', 'horizon', 'ci_violations_pct', 'rmse']]
    f.write("Best 3 combinations:\n")
    f.write(best_3.to_string(index=False))
    f.write("\n\n")

    f.write("=" * 80 + "\n")
    f.write("END OF REPORT\n")
    f.write("=" * 80 + "\n")

print(f"✓ Report saved to {report_file}")
print()

# ============================================================================
# Summary
# ============================================================================

print("=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)
print()
print("Generated files:")
print(f"  1. Interactive HTML: {html_file}")
print(f"  2. Statistics Report: {report_file}")
print()
print("To view:")
print(f"  - Open {html_file} in your browser")
print("  - Use zoom/pan to explore specific time periods")
print("  - Hover over points for exact values")
print("  - Toggle traces on/off in the legend")
print()
print("Key findings:")
avg_violations = stats_df['ci_violations_pct'].mean()
avg_rmse = stats_df['rmse'].mean()
print(f"  - Average CI violations: {avg_violations:.2f}%")
print(f"  - Average RMSE: {avg_rmse:.6f}")
print(f"  - Best horizon: H{stats_df.loc[stats_df['ci_violations_pct'].idxmin(), 'horizon']:.0f} ({stats_df['ci_violations_pct'].min():.2f}% violations)")
print(f"  - Worst horizon: H{stats_df.loc[stats_df['ci_violations_pct'].idxmax(), 'horizon']:.0f} ({stats_df['ci_violations_pct'].max():.2f}% violations)")
print()
