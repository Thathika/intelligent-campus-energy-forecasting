"""
visualizations.py
Editorial Energy Intelligence Plotly Charts
Bespoke data visualizations crafted for professional campus energy analytics.
Features collision-free legends, subtle gridlines, and publication-grade palettes.
"""

from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


THEME_LAYOUT = dict(
    font=dict(
        family='Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif',
        size=11,
        color="#374151"
    ),
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    margin=dict(l=45, r=20, t=35, b=55),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#ffffff",
        bordercolor="#e8e6e1",
        font=dict(color="#111827", size=11, family='Inter, sans-serif')
    ),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.16,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="#e8e6e1",
        borderwidth=1,
        font=dict(size=10.5, color="#4b5563")
    ),
    xaxis=dict(
        showgrid=True,
        gridcolor="#f1f0ec",
        zeroline=False,
        linecolor="#e8e6e1",
        tickfont=dict(color="#6b7280", size=10),
        title_font=dict(size=11, color="#374151")
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="#f1f0ec",
        zeroline=False,
        linecolor="#e8e6e1",
        tickfont=dict(color="#6b7280", size=10),
        title_font=dict(size=11, color="#374151")
    )
)


def plot_historical_trend(df: pd.DataFrame, date_col: str, target_col: str) -> go.Figure:
    """Historical daily energy trend with 7-day moving average and collision-free legend."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[target_col],
        mode="lines",
        name="Daily Total Consumption",
        line=dict(color="#1e3a2f", width=2.0),
        hovertemplate="Total: %{y:,.0f} units<extra></extra>"
    ))

    if len(df) >= 7:
        rolling_7 = df[target_col].rolling(window=7, min_periods=3).mean()
        fig.add_trace(go.Scatter(
            x=df[date_col],
            y=rolling_7,
            mode="lines",
            name="7-Day Moving Average",
            line=dict(color="#64748b", width=1.5, dash="dash"),
            hovertemplate="7-Day Mean: %{y:,.0f}<extra></extra>"
        ))

    fig.update_layout(
        **THEME_LAYOUT,
        title=dict(
            text="Daily Campus Energy Consumption",
            font=dict(size=13, color="#111827", family="Inter, sans-serif"),
            x=0.01,
            xanchor="left"
        ),
        xaxis_title=None,
        yaxis_title="Energy (Units)"
    )
    return fig


def plot_grid_comparison(df: pd.DataFrame, date_col: str) -> Optional[go.Figure]:
    """Comparison of TNEB Campus vs TNEB New STP grid intake."""
    cols = [c for c in ["TNEB_Campus", "TNEB_New_STP"] if c in df.columns]
    if not cols:
        return None

    fig = go.Figure()
    palette = ["#1d4ed8", "#475569"]
    for i, col in enumerate(cols):
        fig.add_trace(go.Scatter(
            x=df[date_col],
            y=df[col],
            mode="lines",
            name=col.replace("_", " "),
            line=dict(color=palette[i % len(palette)], width=1.8),
            hovertemplate=f"{col}: %{{y:,.0f}}<extra></extra>"
        ))

    fig.update_layout(
        **THEME_LAYOUT,
        title=dict(
            text="Grid Supply Intake (TNEB Campus vs New STP)",
            font=dict(size=13, color="#111827", family="Inter, sans-serif"),
            x=0.01,
            xanchor="left"
        ),
        xaxis_title=None,
        yaxis_title="Units"
    )
    return fig


def plot_energy_sources(df: pd.DataFrame, date_col: str) -> Optional[go.Figure]:
    """Daily generation sources breakdown: Solar, Diesel, Biogas."""
    source_cols = [c for c in ["Solar", "Diesel", "Biogas"] if c in df.columns]
    if not source_cols:
        return None

    fig = go.Figure()
    palette = {"Solar": "#d97706", "Diesel": "#64748b", "Biogas": "#2d6a4f"}
    for col in source_cols:
        fig.add_trace(go.Bar(
            x=df[date_col],
            y=df[col],
            name=col,
            marker=dict(color=palette.get(col, "#94a3b8"), line=dict(width=0)),
            hovertemplate=f"{col}: %{{y:,.0f}}<extra></extra>"
        ))

    fig.update_layout(
        **THEME_LAYOUT,
        barmode="stack",
        title=dict(
            text="On-Site Generation Sources (Solar, Diesel, Biogas)",
            font=dict(size=13, color="#111827", family="Inter, sans-serif"),
            x=0.01,
            xanchor="left"
        ),
        xaxis_title=None,
        yaxis_title="Units"
    )
    return fig


def plot_campus_areas(df: pd.DataFrame, date_col: str) -> Optional[go.Figure]:
    """Campus loads breakdown: Academic Blocks, Hostels, Chiller, Staff Quarters, STP."""
    area_cols = [c for c in ["Staff_Quarters", "Academic_Blocks", "Hostels", "Chiller", "STP"] if c in df.columns]
    if not area_cols:
        return None

    fig = go.Figure()
    palette = {
        "Academic_Blocks": "#1e40af",
        "Hostels": "#7c3aed",
        "Chiller": "#0284c7",
        "Staff_Quarters": "#059669",
        "STP": "#c2410c"
    }
    for col in area_cols:
        fig.add_trace(go.Bar(
            x=df[date_col],
            y=df[col],
            name=col.replace("_", " "),
            marker=dict(color=palette.get(col, "#94a3b8"), line=dict(width=0)),
            hovertemplate=f"{col.replace('_', ' ')}: %{{y:,.0f}}<extra></extra>"
        ))

    fig.update_layout(
        **THEME_LAYOUT,
        barmode="stack",
        title=dict(
            text="Campus Load Distribution by Facility",
            font=dict(size=13, color="#111827", family="Inter, sans-serif"),
            x=0.01,
            xanchor="left"
        ),
        xaxis_title=None,
        yaxis_title="Units"
    )
    return fig


def plot_target_distribution(df: pd.DataFrame, target_col: str) -> go.Figure:
    """Histogram of target variable."""
    fig = px.histogram(
        df,
        x=target_col,
        nbins=12,
        color_discrete_sequence=["#2d6a4f"],
        opacity=0.9
    )
    fig.update_layout(
        **THEME_LAYOUT,
        title=dict(
            text=f"Frequency Distribution: {target_col}",
            font=dict(size=13, color="#111827", family="Inter, sans-serif"),
            x=0.01,
            xanchor="left"
        ),
        xaxis_title=f"{target_col} (Units)",
        yaxis_title="Count"
    )
    return fig


def plot_box_distribution(df: pd.DataFrame, target_col: str) -> go.Figure:
    """Box plot of target across day-of-week."""
    temp_df = df.copy()
    if "_parsed_datetime" in temp_df.columns:
        temp_df["Day_Name"] = pd.to_datetime(temp_df["_parsed_datetime"]).dt.day_name()
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        fig = px.box(
            temp_df,
            x="Day_Name",
            y=target_col,
            category_orders={"Day_Name": days_order},
            color_discrete_sequence=["#1e40af"],
            points="all"
        )
        fig.update_traces(marker=dict(size=4, color="#1d4ed8"))
        fig.update_layout(
            **THEME_LAYOUT,
            title=dict(
                text="Daily Consumption by Day of Week",
                font=dict(size=13, color="#111827", family="Inter, sans-serif"),
                x=0.01,
                xanchor="left"
            ),
            xaxis_title=None,
            yaxis_title=f"{target_col} (Units)"
        )
        return fig
    else:
        fig = px.box(df, y=target_col, color_discrete_sequence=["#1e40af"], points="all")
        fig.update_layout(
            **THEME_LAYOUT,
            title=dict(text=f"Distribution of {target_col}", font=dict(size=13, color="#111827", family="Inter, sans-serif"))
        )
        return fig


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: List[str]) -> go.Figure:
    """Correlation matrix heatmap for selected numeric columns."""
    valid_cols = [c for c in numeric_cols if c in df.columns]
    corr = df[valid_cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="Blues",
        zmin=-1,
        zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        hoverongaps=False
    ))

    fig.update_layout(
        **THEME_LAYOUT,
        title=dict(
            text="Pearson Correlation Matrix",
            font=dict(size=13, color="#111827", family="Inter, sans-serif"),
            x=0.01,
            xanchor="left"
        )
    )
    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(autorange="reversed")
    return fig


def plot_forecast_vs_actual(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictions_dict: Dict[str, np.ndarray],
    future_df: Optional[pd.DataFrame],
    date_col: str,
    target_col: str,
    selected_model: str
) -> go.Figure:
    """Combined chart: Historical training, held-out actual, model test predictions, and future horizon."""
    fig = go.Figure()

    # Training
    fig.add_trace(go.Scatter(
        x=train_df[date_col],
        y=train_df[target_col],
        mode="lines",
        name="Training History",
        line=dict(color="#94a3b8", width=1.5),
        hovertemplate="Train: %{y:,.0f}<extra></extra>"
    ))

    # Test Actual
    fig.add_trace(go.Scatter(
        x=test_df[date_col],
        y=test_df[target_col],
        mode="lines+markers",
        name="Actual (Test Set)",
        line=dict(color="#0f172a", width=1.8),
        marker=dict(size=4.5, color="#0f172a"),
        hovertemplate="Actual: %{y:,.0f}<extra></extra>"
    ))

    # Test Prediction
    if selected_model in predictions_dict:
        fig.add_trace(go.Scatter(
            x=test_df[date_col],
            y=predictions_dict[selected_model],
            mode="lines+markers",
            name=f"Predicted ({selected_model})",
            line=dict(color="#2563eb", width=1.6, dash="dash"),
            marker=dict(size=4.5, symbol="circle", color="#2563eb"),
            hovertemplate="Predicted: %{y:,.0f}<extra></extra>"
        ))

    # Future Forecast
    if future_df is not None and not future_df.empty and "predicted_" + target_col in future_df.columns:
        fig.add_trace(go.Scatter(
            x=future_df[date_col],
            y=future_df["predicted_" + target_col],
            mode="lines+markers",
            name="Recursive Forecast",
            line=dict(color="#15803d", width=2.0),
            marker=dict(size=5, symbol="square", color="#15803d"),
            hovertemplate="Forecast: %{y:,.0f}<extra></extra>"
        ))

    last_test_date = test_df[date_col].max()
    fig.add_vline(
        x=last_test_date,
        line_width=1,
        line_dash="dot",
        line_color="#94a3b8"
    )

    fig.update_layout(
        **THEME_LAYOUT,
        title=dict(
            text=f"Demand Forecast vs Actual ({selected_model})",
            font=dict(size=13, color="#111827", family="Inter, sans-serif"),
            x=0.01,
            xanchor="left"
        ),
        xaxis_title=None,
        yaxis_title="Units"
    )
    return fig


def plot_model_comparison_bars(comp_df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing MAE and RMSE across candidate models."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=comp_df["Model"],
        y=comp_df["MAE"],
        name="MAE (Mean Absolute Error)",
        marker=dict(color="#2d6a4f", line=dict(width=0)),
        hovertemplate="%{x}<br>MAE: %{y:,.1f}<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        x=comp_df["Model"],
        y=comp_df["RMSE"],
        name="RMSE (Root Mean Squared Error)",
        marker=dict(color="#64748b", line=dict(width=0)),
        hovertemplate="%{x}<br>RMSE: %{y:,.1f}<extra></extra>"
    ))

    fig.update_layout(
        **THEME_LAYOUT,
        barmode="group",
        title=dict(
            text="Model Error Benchmark on Out-of-Sample Test Set",
            font=dict(size=13, color="#111827", family="Inter, sans-serif"),
            x=0.01,
            xanchor="left"
        ),
        xaxis_title=None,
        yaxis_title="Error (Units)"
    )
    return fig


def plot_residuals_scatter(test_df: pd.DataFrame, date_col: str, residuals: np.ndarray, model_name: str) -> go.Figure:
    """Stem-like scatter plot of prediction residuals (Actual - Predicted)."""
    fig = go.Figure()
    fig.add_hline(y=0, line_width=1, line_color="#94a3b8")
    fig.add_trace(go.Scatter(
        x=test_df[date_col],
        y=residuals,
        mode="markers+lines",
        name="Residual (Actual - Predicted)",
        marker=dict(size=5, color="#0f172a"),
        line=dict(color="#cbd5e1", width=1),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Residual: %{y:+,.0f}<extra></extra>"
    ))

    fig.update_layout(
        **THEME_LAYOUT,
        title=dict(
            text=f"Prediction Residuals: {model_name} (Actual - Predicted)",
            font=dict(size=13, color="#111827", family="Inter, sans-serif"),
            x=0.01,
            xanchor="left"
        ),
        xaxis_title=None,
        yaxis_title="Residual Error (Units)"
    )
    return fig


def plot_anomalies_scatter(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    anomaly_df: pd.DataFrame
) -> go.Figure:
    """Full trend line with highlighted flagged anomalies."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[target_col],
        mode="lines",
        name="Daily Consumption",
        line=dict(color="#2d6a4f", width=1.8),
        hovertemplate="Recorded: %{y:,.0f}<extra></extra>"
    ))

    if not anomaly_df.empty:
        flagged_dates = pd.to_datetime(anomaly_df["Date"])
        flagged_vals = anomaly_df["Observed Value"]
        fig.add_trace(go.Scatter(
            x=flagged_dates,
            y=flagged_vals,
            mode="markers",
            name="Flagged Reading",
            marker=dict(size=8, color="#dc2626", symbol="circle", line=dict(width=1.5, color="#7f1d1d")),
            hovertemplate="Flagged: %{y:,.0f}<extra></extra>"
        ))

    fig.update_layout(
        **THEME_LAYOUT,
        title=dict(
            text="Consumption Trend with Flagged Atypical Readings",
            font=dict(size=13, color="#111827", family="Inter, sans-serif"),
            x=0.01,
            xanchor="left"
        ),
        xaxis_title=None,
        yaxis_title="Units"
    )
    return fig


def plot_cost_and_emissions_trend(
    dates: pd.Series,
    cost_series: pd.Series,
    emissions_series: pd.Series
) -> go.Figure:
    """Dual-axis line chart for cost (INR) and carbon emissions (kg CO2e)."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=cost_series,
        mode="lines",
        name="Estimated Cost (₹)",
        line=dict(color="#1d4ed8", width=1.8),
        hovertemplate="Cost: ₹%{y:,.0f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=dates,
        y=emissions_series,
        mode="lines",
        name="Estimated Emissions (kg CO₂e)",
        yaxis="y2",
        line=dict(color="#2d6a4f", width=1.8, dash="dot"),
        hovertemplate="Emissions: %{y:,.0f} kg<extra></extra>"
    ))

    layout = THEME_LAYOUT.copy()
    layout.update(dict(
        title=dict(
            text="Daily Estimated Cost and Carbon Emissions",
            font=dict(size=13, color="#111827", family="Inter, sans-serif"),
            x=0.01,
            xanchor="left"
        ),
        xaxis=dict(showgrid=True, gridcolor="#f1f0ec", linecolor="#e8e6e1"),
        yaxis=dict(title="Estimated Cost (₹)", showgrid=True, gridcolor="#f1f0ec", linecolor="#e8e6e1"),
        yaxis2=dict(
            title="Estimated Carbon (kg CO₂e)",
            overlaying="y",
            side="right",
            showgrid=False,
            linecolor="#e8e6e1"
        )
    ))
    fig.update_layout(layout)
    return fig
