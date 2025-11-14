"""
Interactive Plotly Visualization for backfill_16yr Model - OUT-OF-SAMPLE

Creates 12-panel dashboard (3 grid points × 4 horizons) showing:
- Ground truth volatility surfaces
- p50 (median) predictions
- 90% CI bands (p05-p95)
- CI violations highlighted in red
- Statistics: RMSE, violations %, CI width

Test set: 2019-2023 (post-crisis period)
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

print("=" * 80)
print("INTERACTIVE PLOTLY VISUALIZATION - backfill_16yr (OUT-OF-SAMPLE)")
print("=" * 80)
print()

# ============================================================================
# Configuration
# ============================================================================

# Grid points to visualize (row, col, name)
GRID_POINTS = [
    (1, 2, "ATM 3-Month"),       # Row 1, Col 2
    (3, 2, "ATM 1-Year"),        # Row 3, Col 2
    (3, 0, "OTM Put 1-Year"),    # Row 3, Col 0
]

HORIZONS = [1, 7, 14, 30]

# Create output directory
output_dir = Path("tables/backfill_plots")
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Load Data
# ============================================================================

print("Loading data...")

# Load out-of-sample predictions
oos_data = np.load("models_backfill/oos_reconstruction_16yr.npz")

# Load ground truth
full_data = np.load("data/vol_surface_with_ret.npz")
vol_surf_data = full_data["surface"]

print("✓ Data loaded")
print()

# ============================================================================
# Create Plotly Figure
# ============================================================================

print("Creating interactive dashboard...")

# Create subplot grid: 3 rows (grid points) × 4 cols (horizons)
fig = make_subplots(
    rows=len(GRID_POINTS),
    cols=len(HORIZONS),
    subplot_titles=[f"{gp[2]} - H{h}" for gp in GRID_POINTS for h in HORIZONS],
    vertical_spacing=0.08,
    horizontal_spacing=0.05,
    shared_xaxes=True,
)

# Storage for statistics
all_stats = []

# ============================================================================
# Process Each Grid Point and Horizon
# ============================================================================

for gp_idx, (gp_row, gp_col, gp_name) in enumerate(GRID_POINTS):
    for h_idx, horizon in enumerate(HORIZONS):
        print(f"  Processing {gp_name} - H{horizon}...")

        # Subplot position
        subplot_row = gp_idx + 1
        subplot_col = h_idx + 1

        # Load predictions and ground truth for this horizon
        recon_key = f'recon_h{horizon}'
        indices_key = f'indices_h{horizon}'

        recons = oos_data[recon_key]  # (N, 3, 5, 5)
        indices = oos_data[indices_key]  # (N,)

        # Extract quantiles for this grid point
        p05 = recons[:, 0, gp_row, gp_col]  # (N,)
        p50 = recons[:, 1, gp_row, gp_col]  # (N,)
        p95 = recons[:, 2, gp_row, gp_col]  # (N,)

        # Ground truth
        gt = vol_surf_data[indices, gp_row, gp_col]  # (N,)

        # ====================================================================
        # Compute Statistics
        # ====================================================================

        # RMSE
        rmse = np.sqrt(np.mean((p50 - gt) ** 2))

        # CI violations
        below_p05 = gt < p05
        above_p95 = gt > p95
        violations = below_p05 | above_p95
        num_violations = np.sum(violations)
        pct_violations = 100 * num_violations / len(gt)

        pct_below_p05 = 100 * np.sum(below_p05) / len(gt)
        pct_above_p95 = 100 * np.sum(above_p95) / len(gt)

        # CI width
        ci_width = np.mean(p95 - p05)

        # Store statistics
        all_stats.append({
            'grid_point': gp_name,
            'horizon': horizon,
            'rmse': rmse,
            'ci_violations_pct': pct_violations,
            'ci_violations_count': num_violations,
            'below_p05_pct': pct_below_p05,
            'above_p95_pct': pct_above_p95,
            'mean_ci_width': ci_width,
        })

        # ====================================================================
        # Create Plotly Traces
        # ====================================================================

        x_vals = indices.tolist()

        # CI band (filled area)
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=p95.tolist(), mode='lines', line=dict(width=0),
                showlegend=False, hoverinfo='skip',
            ), row=subplot_row, col=subplot_col,
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=p05.tolist(), mode='lines', line=dict(width=0),
                fillcolor='rgba(100, 149, 237, 0.3)', fill='tonexty',
                name='90% CI', showlegend=(subplot_row == 1 and subplot_col == 1),
                hovertemplate='CI: [%{y:.4f}, previous]<extra></extra>',
            ), row=subplot_row, col=subplot_col,
        )

        # Ground truth (black line)
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=gt.tolist(), mode='lines',
                line=dict(color='black', width=1.5),
                name='Ground Truth',
                showlegend=(subplot_row == 1 and subplot_col == 1),
                hovertemplate='GT: %{y:.4f}<extra></extra>',
            ), row=subplot_row, col=subplot_col,
        )

        # p50 prediction (blue line)
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=p50.tolist(), mode='lines',
                line=dict(color='cornflowerblue', width=1.0),
                name='p50 (Prediction)',
                showlegend=(subplot_row == 1 and subplot_col == 1),
                hovertemplate='p50: %{y:.4f}<extra></extra>',
            ), row=subplot_row, col=subplot_col,
        )

        # Violations (red markers)
        if num_violations > 0:
            violation_x = indices[violations].tolist()
            violation_y = gt[violations].tolist()
            fig.add_trace(
                go.Scatter(
                    x=violation_x, y=violation_y, mode='markers',
                    marker=dict(color='red', size=3, opacity=0.6),
                    name='Violations',
                    showlegend=(subplot_row == 1 and subplot_col == 1),
                    hovertemplate='Violation: %{y:.4f}<extra></extra>',
                ), row=subplot_row, col=subplot_col,
            )

        # ====================================================================
        # Add Statistics Annotation
        # ====================================================================

        # Color code violations
        if pct_violations < 12:
            violation_color = 'green'
        elif pct_violations < 20:
            violation_color = 'orange'
        else:
            violation_color = 'red'

        annotation_text = (
            f"RMSE: {rmse:.4f}<br>"
            f"<b style='color:{violation_color}'>Violations: {pct_violations:.1f}%</b><br>"
            f"CI Width: {ci_width:.4f}"
        )

        # Calculate subplot number for proper xref/yref
        subplot_num = (subplot_row-1)*len(HORIZONS) + subplot_col
        xref_str = 'x domain' if subplot_num == 1 else f"x{subplot_num} domain"
        yref_str = 'y domain' if subplot_num == 1 else f"y{subplot_num} domain"

        fig.add_annotation(
            text=annotation_text,
            xref=xref_str,
            yref=yref_str,
            x=0.02, y=0.98,
            xanchor='left', yanchor='top',
            showarrow=False,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor=violation_color,
            borderwidth=2,
            font=dict(size=10),
        )

# ============================================================================
# Update Layout
# ============================================================================

fig.update_layout(
    title=dict(
        text="backfill_16yr Model: Out-of-Sample Teacher Forcing (2019-2023)<br>"
             "<sub>3 Grid Points × 4 Horizons | Context Length: 20 days</sub>",
        x=0.5,
        xanchor='center',
        font=dict(size=16),
    ),
    height=900,
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.02,
        xanchor="center",
        x=0.5,
    ),
    hovermode='x unified',
)

# Update axes labels
for i in range(len(GRID_POINTS)):
    for j in range(len(HORIZONS)):
        subplot_row = i + 1
        subplot_col = j + 1

        # X-axis label only on bottom row
        if subplot_row == len(GRID_POINTS):
            fig.update_xaxes(title_text="Date Index", row=subplot_row, col=subplot_col)

        # Y-axis label only on leftmost column
        if subplot_col == 1:
            fig.update_yaxes(title_text="Implied Volatility", row=subplot_row, col=subplot_col)

# Add range selector
fig.update_xaxes(
    rangeslider=dict(visible=False),
    rangeselector=dict(
        buttons=list([
            dict(count=100, label="100d", step="day", stepmode="backward"),
            dict(count=250, label="250d", step="day", stepmode="backward"),
            dict(count=500, label="500d", step="day", stepmode="backward"),
            dict(step="all", label="All"),
        ]),
        y=1.05,
    ),
    row=1, col=1,
)

# ============================================================================
# Save Interactive HTML
# ============================================================================

output_file = output_dir / "backfill_oos_16yr_teacher_forcing.html"
print()
print(f"Saving interactive HTML to {output_file}...")
fig.write_html(output_file)
print("✓ Saved!")
print()

# ============================================================================
# Generate Statistics Report
# ============================================================================

print("Generating statistics report...")

df_stats = pd.DataFrame(all_stats)

# Create comprehensive report
report_lines = []
report_lines.append("=" * 80)
report_lines.append("CI VIOLATIONS REPORT - backfill_16yr Model (OUT-OF-SAMPLE)")
report_lines.append("=" * 80)
report_lines.append("")

# 1. Performance by horizon
report_lines.append("1. PERFORMANCE BY HORIZON")
report_lines.append("-" * 80)
by_horizon = df_stats.groupby('horizon').agg({
    'rmse': 'mean',
    'ci_violations_pct': 'mean',
    'below_p05_pct': 'mean',
    'above_p95_pct': 'mean',
    'mean_ci_width': 'mean',
}).round(4)
report_lines.append(by_horizon.to_string())
report_lines.append("")

# 2. Performance by grid point
report_lines.append("2. PERFORMANCE BY GRID POINT")
report_lines.append("-" * 80)
by_gridpoint = df_stats.groupby('grid_point').agg({
    'rmse': 'mean',
    'ci_violations_pct': 'mean',
    'below_p05_pct': 'mean',
    'above_p95_pct': 'mean',
    'mean_ci_width': 'mean',
}).round(4)
report_lines.append(by_gridpoint.to_string())
report_lines.append("")

# 3. Detailed statistics (all combinations)
report_lines.append("3. DETAILED STATISTICS (ALL COMBINATIONS)")
report_lines.append("-" * 80)
report_lines.append(df_stats.to_string(index=False))
report_lines.append("")

# 4. Overall assessment
report_lines.append("4. OVERALL ASSESSMENT")
report_lines.append("-" * 80)
avg_violations = df_stats['ci_violations_pct'].mean()
min_violations = df_stats['ci_violations_pct'].min()
max_violations = df_stats['ci_violations_pct'].max()

report_lines.append(f"Target CI violations: ~10%")
report_lines.append(f"Average violations: {avg_violations:.2f}%")
report_lines.append(f"Range: [{min_violations:.2f}%, {max_violations:.2f}%]")
report_lines.append("")

if avg_violations < 12:
    verdict = "✓ GOOD - Well calibrated"
elif avg_violations < 20:
    verdict = "⚠ MODERATE - Needs improvement"
else:
    verdict = "✗ POOR - Significant miscalibration"

report_lines.append(f"Verdict: {verdict}")
report_lines.append("")

# Worst/best combinations
report_lines.append("Worst 3 combinations:")
worst_3 = df_stats.nlargest(3, 'ci_violations_pct')[['grid_point', 'horizon', 'ci_violations_pct', 'rmse']]
report_lines.append(worst_3.to_string(index=False))
report_lines.append("")

report_lines.append("Best 3 combinations:")
best_3 = df_stats.nsmallest(3, 'ci_violations_pct')[['grid_point', 'horizon', 'ci_violations_pct', 'rmse']]
report_lines.append(best_3.to_string(index=False))
report_lines.append("")

report_lines.append("=" * 80)
report_lines.append("END OF REPORT")
report_lines.append("=" * 80)
report_lines.append("")

# Save report
report_file = output_dir / "ci_violations_report_oos_16yr.txt"
with open(report_file, 'w') as f:
    f.write('\n'.join(report_lines))

print(f"✓ Statistics report saved to {report_file}")
print()

# ============================================================================
# Summary
# ============================================================================

print("=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)
print()
print(f"Interactive HTML: {output_file}")
print(f"Statistics report: {report_file}")
print()
print("Key Findings (OUT-OF-SAMPLE):")
print(f"  - Average CI violations: {avg_violations:.2f}%")
print(f"  - Range: [{min_violations:.2f}%, {max_violations:.2f}%]")
print(f"  - Verdict: {verdict}")
print()
print("Open the HTML file in a web browser to explore interactively!")
print()
