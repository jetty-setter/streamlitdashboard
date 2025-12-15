from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

DATA_FILE = Path("data/labor_data.csv")

# Map each series to units for grouping
UNIT_MAP = {
    "Total Nonfarm Employment": "Employment (thousands)",
    "Manufacturing Employment": "Employment (thousands)",
    "Education and Health Services Employment": "Employment (thousands)",
    "Average Hourly Earnings for Private Employees": "Dollars",
    "Unemployment Rate": "Percent",
    "Labor Force Participation Rate": "Percent",
}

st.set_page_config(page_title="U.S. Labor Market Dashboard", page_icon="📊", layout="wide")

st.title("U.S. Labor Market Dashboard")
st.caption("Data source: U.S. Bureau of Labor Statistics (BLS)")


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["date", "value"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df.sort_values(["series_id", "date"])
    return df


def make_line_chart(df_long: pd.DataFrame, y_col: str, y_title: str) -> alt.Chart:
    """
    df_long expects columns: date, Month, series_name, unit, value, yoy_pct, indexed, etc.
    """
    if df_long.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line()

    # Dynamic y padding so the chart “zooms in” nicely
    y_min = float(df_long[y_col].min())
    y_max = float(df_long[y_col].max())
    if np.isfinite(y_min) and np.isfinite(y_max) and y_min != y_max:
        pad = (y_max - y_min) * 0.08
        domain = [y_min - pad, y_max + pad]
    else:
        domain = None

    chart = (
        alt.Chart(df_long)
        .mark_line(point=False)
        .encode(
            x=alt.X("date:T", title="Month", axis=alt.Axis(format="%b %Y", labelAngle=-45)),
            y=alt.Y(f"{y_col}:Q", title=y_title, scale=alt.Scale(domain=domain)),
            color=alt.Color("series_name:N", title="Series"),
            tooltip=[
                alt.Tooltip("series_name:N", title="Series"),
                alt.Tooltip("date:T", title="Date", format="%Y-%m"),
                alt.Tooltip(f"{y_col}:Q", title=y_title, format=",.2f"),
            ],
        )
        .properties(height=320)
        .interactive()
    )
    return chart


# ---------- Load ----------
if not DATA_FILE.exists():
    st.warning("No data found. Please run `python collect.py` first.")
    st.stop()

df = load_data(DATA_FILE)
df["unit"] = df["series_name"].map(UNIT_MAP).fillna("Other")

# Year-over-year percent change (12 months)
df["yoy_pct"] = df.groupby("series_id")["value"].pct_change(12) * 100.0

# Month label (still useful for table)
df["Month"] = df["date"].dt.strftime("%Y-%m")

# Indexed option (start=100 in first visible month per series)
df["indexed_100"] = (
    df.sort_values("date")
      .groupby("series_id")["value"]
      .apply(lambda s: (s / s.iloc[0]) * 100 if len(s) else s)
      .reset_index(level=0, drop=True)
)

# ---------- Sidebar ----------
st.sidebar.header("Filters")

latest_overall_date = df["date"].max()

time_window = st.sidebar.selectbox(
    "Time window",
    ["Last 12 months", "All data"],
    index=0,
)

series_options = sorted(df["series_name"].unique().tolist())

default_series = [s for s in [
    "Total Nonfarm Employment",
    "Unemployment Rate",
    "Labor Force Participation Rate",
    "Average Hourly Earnings for Private Employees",
    "Manufacturing Employment",
    "Education and Health Services Employment",
] if s in series_options]

selected_series = st.sidebar.multiselect(
    "Select series",
    options=series_options,
    default=default_series if default_series else series_options,
)

if not selected_series:
    st.info("Please select at least one series.")
    st.stop()

view_mode = st.sidebar.radio(
    "View mode",
    ["Levels (original values)", "Year-over-year change (%)", "Indexed (start=100)"],
    index=0,
)

# ---------- Apply filters ----------
if time_window == "Last 12 months":
    cutoff = latest_overall_date - pd.DateOffset(months=12)
    work_df = df[df["date"] >= cutoff].copy()
else:
    work_df = df.copy()

work_df = work_df[work_df["series_name"].isin(selected_series)].copy()

if work_df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# ---------- Chart Section (this is the big change) ----------
st.subheader("Trends over time")
st.caption("Tip: use Indexed (start=100) if you want the lines to be easier to compare across different scales.")

# Decide y column + title once
if view_mode == "Year-over-year change (%)":
    plot_df = work_df.dropna(subset=["yoy_pct"]).copy()
    y_col = "yoy_pct"
    y_title = "YoY % change"
elif view_mode == "Indexed (start=100)":
    plot_df = work_df.copy()
    y_col = "indexed_100"
    y_title = "Index (first month = 100)"
else:
    plot_df = work_df.copy()
    y_col = "value"
    y_title = "Value"

# If YoY is selected but there is not enough history in the filtered window
if view_mode == "Year-over-year change (%)" and plot_df.empty:
    st.info("YoY needs at least 13 months of data per series. Try 'All data' or 'Levels'.")
else:
    # Tabs by unit so it doesn’t look like a stacked wall of charts
    units_in_view = [u for u in ["Employment (thousands)", "Percent", "Dollars", "Other"]
                     if u in plot_df["unit"].unique()]

    tabs = st.tabs(units_in_view) if units_in_view else []

    for tab, unit_name in zip(tabs, units_in_view):
        with tab:
            unit_df = plot_df[plot_df["unit"] == unit_name].copy()
            if unit_df.empty:
                st.info("No series selected in this unit group.")
                continue

            # Chart title inside the tab
            st.markdown(f"**{unit_name}**")

            chart = make_line_chart(unit_df, y_col=y_col, y_title=y_title)
            st.altair_chart(chart, use_container_width=True)

# ---------- Latest month summary ----------
latest_shown_date = work_df["date"].max()
latest_label = latest_shown_date.strftime("%Y-%m")
st.subheader(f"Latest month shown: {latest_label}")

latest_subset = df[
    (df["date"] == latest_shown_date)
    & (df["series_name"].isin(selected_series))
].copy()

latest_levels = latest_subset.set_index("series_name")["value"].reindex(selected_series)
latest_yoy = latest_subset.set_index("series_name")["yoy_pct"].reindex(selected_series)

cols = st.columns(min(4, len(selected_series)) or 1)

for i, name in enumerate(selected_series):
    level = latest_levels.get(name, np.nan)
    yoy = latest_yoy.get(name, np.nan)

    if pd.isna(level):
        value_text = "N/A"
        delta_text = ""
    else:
        value_text = f"{level:,.2f}"
        delta_text = "" if pd.isna(yoy) else f"{yoy:+.2f}% vs 12 months ago"

    with cols[i % len(cols)]:
        st.metric(label=name, value=value_text, delta=delta_text)

# ---------- Table ----------
st.divider()
st.subheader("Data table")

table_df = work_df.sort_values(["series_name", "date"]).copy()

table_df = table_df[
    ["series_name", "Month", "value", "yoy_pct", "unit", "periodName", "year", "series_id"]
].rename(
    columns={
        "series_name": "Series",
        "value": "Value",
        "yoy_pct": "YoY % (vs prior year)",
        "unit": "Unit",
        "periodName": "Month name",
        "year": "Year",
        "series_id": "BLS Series ID",
    }
)

st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Value": st.column_config.NumberColumn("Value", format="%.2f"),
        "YoY % (vs prior year)": st.column_config.NumberColumn("YoY % (vs prior year)", format="%.2f"),
        "Year": st.column_config.NumberColumn("Year", format="%d"),
    },
)

csv_bytes = table_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download filtered data as CSV",
    data=csv_bytes,
    file_name="labor_data_filtered.csv",
    mime="text/csv",
)
