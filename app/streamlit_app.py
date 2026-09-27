import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import create_engine
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="NYC Taxi Demand Forecast", layout="wide", page_icon="🚕")

BLUE = "#0f3460"
LIGHT_BLUE = "#16537e"
ORANGE = "#f5a623"
GREEN = "#2ecc71"

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0; }
    .sub-header { font-size: 1rem; color: #666; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🚕 NYC Yellow Cab Demand Forecast</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Prophet time series • Hourly pickups • 72h forecast</p>', unsafe_allow_html=True)

DB_CONN = "postgresql+psycopg2://taxi_user:your_password@localhost:5432/taxi_db"
engine = create_engine(DB_CONN)

# cache disabled for debugging
def load_data():
    actual = pd.read_sql("SELECT hour, demand FROM hourly_demand ORDER BY hour", engine)
    actual["hour"] = pd.to_datetime(actual["hour"])
    forecast = pd.read_sql("SELECT ds, yhat, yhat_lower, yhat_upper FROM demand_forecasts ORDER BY ds", engine)
    forecast["ds"] = pd.to_datetime(forecast["ds"])
    return actual, forecast

st.cache_data.clear()
actual, forecast = load_data()

st.sidebar.header("Controls")
months = sorted(actual["hour"].dt.to_period("M").unique())
selected_months = st.sidebar.multiselect("Months", months, default=months[-3:])
filter_start = min(selected_months).to_timestamp() if selected_months else actual["hour"].min()
filter_end = (max(selected_months) + 1).to_timestamp() if selected_months else actual["hour"].max()
actual_f = actual[(actual["hour"] >= filter_start) & (actual["hour"] < filter_end)].copy()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Hours", f"{len(actual):,}")
col2.metric("Avg Demand/hr", f"{actual['demand'].mean():,.0f}")
peak_idx = actual["demand"].idxmax()
col3.metric("Peak Hour", actual.loc[peak_idx, "hour"].strftime("%b %d, %H:%M"))
col4.metric("Peak Demand", f"{actual['demand'].max():,}")

st.divider()
tab1, tab2, tab3 = st.tabs(["Forecast", "Demand Patterns", "Model Decomposition"])

# --- TAB 1: Forecast (Streamlit native chart - no Plotly date bugs) ---
with tab1:
    st.subheader("Forecast vs Actual")
    st.caption("Last 7 days of actual demand + 72h Prophet forecast with 95% confidence interval")

    last_7d = actual[actual["hour"] >= actual["hour"].max() - pd.Timedelta(days=7)].copy()

    # Plot actual
    st.line_chart(last_7d.set_index("hour")["demand"], height=200)
    st.caption("Actual (last 7 days)")

    # Plot forecast (force datetime conversion)
    fc = forecast.copy()
    fc["ds"] = pd.to_datetime(fc["ds"])
    fc = fc.set_index("ds").sort_index()
    st.line_chart(fc["yhat"], height=200)
    st.caption("Forecast (next 72h)")

    st.markdown("---")
    st.caption(f"Forecast horizon: {FORECAST_HORIZON if 'FORECAST_HORIZON' in dir() else 72} hours | Model: Prophet (weekly + daily seasonality, log transform)")

# --- TAB 2: Demand Patterns (Plotly - integer x-axis, no date issues) ---
with tab2:
    st.subheader("Average Hourly Demand by Day of Week")
    st.caption("The shape of a typical day")

    actual_f["dow"] = actual_f["hour"].dt.day_name()
    actual_f["hr"] = actual_f["hour"].dt.hour
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    colors_dow = ["#4e79a7", "#59a14f", "#f28e2b", "#e15759", "#76b7b2", "#edc948", "#b07aa1"]

    avg_by_dow = actual_f.groupby(["dow", "hr"])["demand"].mean().reset_index()

    fig = go.Figure()
    for i, dow in enumerate(dow_order):
        subset = avg_by_dow[avg_by_dow["dow"] == dow]
        fig.add_trace(go.Scatter(
            x=subset["hr"].tolist(), y=subset["demand"].tolist(),
            mode="lines", name=dow,
            line=dict(color=colors_dow[i], width=2),
            hovertemplate=dow + ": %{y:,.0f} trips<extra></extra>"
        ))
    fig.update_layout(
        height=450, template="plotly_white",
        xaxis_title="Hour of Day", yaxis_title="Avg Trips/hour",
        xaxis_tickvals=list(range(0, 24, 2)),
        xaxis_ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
        margin=dict(l=50, r=20, t=40, b=30)
    )
    st.plotly_chart(fig, width='stretch')

    st.subheader("Day-of-Week Heatmap")
    st.caption("Darker = more trips")

    pivot = actual_f.pivot_table(values="demand", index="hr", columns="dow", aggfunc="mean")
    pivot = pivot.reindex(columns=dow_order)

    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="YlOrRd",
                    labels={"x": "Day", "y": "Hour", "color": "Avg Trips"}, text_auto=".0s")
    fig.update_layout(height=500, margin=dict(l=50, r=20, t=20, b=30))
    fig.update_xaxes(side="bottom")
    st.plotly_chart(fig, width='stretch')

    st.subheader("Weekly Trend")
    st.caption("Total trips per ISO week")

    actual_f["week"] = actual_f["hour"].dt.isocalendar().week.astype(int)
    weekly = actual_f.groupby("week")["demand"].sum().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=weekly["week"].tolist(), y=weekly["demand"].tolist(),
        marker_color=LIGHT_BLUE, name="Total trips",
        hovertemplate="Week %{x}: %{y:,} trips<extra></extra>"
    ))
    fig.update_layout(
        height=300, template="plotly_white",
        xaxis_title="ISO Week", yaxis_title="Total Trips",
        margin=dict(l=50, r=20, t=30, b=30)
    )
    st.plotly_chart(fig, width='stretch')

# --- TAB 3: Decomposition (integer x-axis) ---
with tab3:
    st.subheader("How the Forecast is Built")
    st.caption("Prophet decomposes demand into 3 parts. Here they are in real trip units.")

    all_data = actual.rename(columns={"hour": "ds", "demand": "y"}).copy()
    all_data["y"] = np.log1p(all_data["y"])

    from prophet import Prophet
    model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True)
    model.fit(all_data)
    comp = model.predict(model.make_future_dataframe(periods=0, freq="h"))

    # Daily pattern as % of mean
    st.subheader("Daily Pattern (one typical day)")
    st.caption("The twin peaks are morning rush and evening rush")

    daily_one = comp.iloc[:24].copy()
    mean_log = comp["trend"].mean()
    daily_pct = (np.expm1(mean_log + daily_one["daily"]) / np.expm1(mean_log) - 1) * 100
    hours = list(range(24))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours, y=daily_pct.tolist(),
        mode="lines", fill="tozeroy",
        line=dict(color=GREEN, width=3),
        fillcolor="rgba(46,204,113,0.15)",
        hovertemplate="%{x}:00 → %{y:+.0f}% vs mean<extra></extra>"
    ))
    peak_am = int(np.argmax(daily_pct.values[5:12])) + 5
    peak_pm = 12 + int(np.argmax(daily_pct.values[12:]))
    fig.add_annotation(x=peak_am, y=daily_pct.values[peak_am],
                       text=f"AM Rush<br>+{daily_pct.values[peak_am]:.0f}%",
                       showarrow=True, arrowhead=1, font=dict(size=11, color=GREEN))
    fig.add_annotation(x=peak_pm, y=daily_pct.values[peak_pm],
                       text=f"PM Rush<br>+{daily_pct.values[peak_pm]:.0f}%",
                       showarrow=True, arrowhead=1, font=dict(size=11, color=GREEN))
    fig.update_layout(
        height=350, template="plotly_white",
        xaxis_title="Hour of Day", yaxis_title="% of mean demand",
        xaxis_tickvals=list(range(0, 24, 2)),
        xaxis_ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)],
        margin=dict(l=50, r=20, t=30, b=30)
    )
    st.plotly_chart(fig, width='stretch')

    # Weekly pattern
    st.subheader("Weekly Pattern (one typical week)")
    st.caption("How demand shifts across the 7 days")

    weekly_one = comp.iloc[:168].copy()
    base_trend = weekly_one["trend"].mean()
    weekly_trips = np.expm1(base_trend + weekly_one["weekly"]) - np.expm1(base_trend)
    x_week = list(range(168))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_week, y=weekly_trips.tolist(),
        mode="lines", fill="tozeroy",
        line=dict(color=ORANGE, width=2),
        fillcolor="rgba(245,166,35,0.15)",
        hovertemplate="Hour %{x} → %{y:+,.0f} trips<extra></extra>"
    ))
    for d in range(7):
        fig.add_vline(x=d*24, line_dash="dot", line_color="gray", opacity=0.3)
    fig.update_layout(
        height=300, template="plotly_white",
        xaxis_title="Hour (0=Mon 00:00, 167=Sun 23:00)",
        yaxis_title="+/- Trips vs baseline",
        xaxis_tickvals=[0, 24, 48, 72, 96, 120, 144, 167],
        xaxis_ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", ""],
        margin=dict(l=50, r=20, t=30, b=30)
    )
    st.plotly_chart(fig, width='stretch')

    # How it adds up
    st.subheader("How the Pieces Add Up")
    st.caption("Trend + Weekly + Daily = Forecast")

    week_slice = comp.iloc[:168].copy()
    base = week_slice["trend"].mean()

    trend_line = np.full(168, np.expm1(base))
    weekly_line = np.expm1(base + week_slice["weekly"])
    daily_line = np.expm1(base + week_slice["weekly"] + week_slice["daily"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_week, y=trend_line.tolist(),
                             mode="lines", name="Trend (baseline)",
                             line=dict(color="gray", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=x_week, y=weekly_line.tolist(),
                             mode="lines", name="+ Weekly",
                             line=dict(color=ORANGE, width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=x_week, y=daily_line.tolist(),
                             mode="lines", name="+ Daily (= Forecast)",
                             line=dict(color=BLUE, width=3),
                             fill="tonexty", fillcolor="rgba(15,52,96,0.1)"))
    fig.update_layout(
        height=400, template="plotly_white",
        xaxis_title="Hour (one week)", yaxis_title="Trips",
        xaxis_tickvals=[0, 24, 48, 72, 96, 120, 144],
        xaxis_ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
        margin=dict(l=50, r=20, t=40, b=30)
    )
    st.plotly_chart(fig, width='stretch')

    st.markdown("""
    **How to read this:**
    - **Daily** — the most important signal. Twin peaks at rush hour drive 80% of the pattern.
    - **Weekly** — a subtle shift. Weekends have slightly different demand shape.
    - **Trend** — the baseline. Flat = no growth/decline over your 3-month window.
    - **Forecast = all three added together** at any future hour.
    """)

st.divider()
st.caption("Airflow 3 | Prophet | PostgreSQL | Plotly | NYC TLC Yellow Cab") 
