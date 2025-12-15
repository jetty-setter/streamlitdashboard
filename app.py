from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DATA_FILE = Path("data/labor_data.csv")

# Map series to units so charts can be grouped with sensible Y-axis ranges
UNIT_MAP = {
    "Total Nonfarm Employment": "Employment (thousands)",
    "Manufacturing Employment": "Employment (thousands)",
    "Education and Health Services Employment": "Employment (thousands)",
    "Average Hourly Earnings for Private Employees": "Dollars",
    "Unemployment Rate": "Percent",
    "Labor Force Participation Rate": "Percent",
}

st.set_page_config(
    page_title="U.S. Labor Market Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("U.S. Labor Market Dashboard")
st.caption("Data source: U.S. Bureau of Labor Statistics (BLS)")


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    """
    Load and clean the BLS data from CSV using pandas.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["date", "value"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df.sort_values(["series_id", "date"])
    return df


if not DATA_FILE.exists():
    st.warning("No data found. Please run `python collect.py` first.")
    st.stop()

df = load_data(DATA_FILE)

# Add unit labels (used to split charts)
df["unit"] = df["series_name"].map(UNIT_MAP).fillna("Other")

# Compute year-over-year change on the full dataset (so it works even when filtering later)
df["yoy_pct"] = df.groupby("series_id")["value"].pct_change(periods=12) * 100.0

# Precompute a simple month label for charts & tables
df["Month"] = df["date"].dt.strftime("%Y-%m")

# Sidebar filters
st.sidebar.header("Filters")

latest_overall_date = df["date"].max()

time_window = st.sidebar.selectbox(
    "Time window",
    ["Last 12 months", "All data"],
    index=0,
)

series_options = sorted(df["series_name"].unique().tolist())

# Smarter default selection (less clutter, still representative)
default_series = [
    s
    for s in [
        "Total Nonfarm Employment",
        "Unemployment Rate",
        "Labor Force Participation Rate",
        "Average Hourly Earnings for Private Employees",
    ]
    if s in series_options
]

selected_series = st.sidebar.multiselect(
    "Select series to display",
    options=series_options,
    default=default_series if default_series else series_options,
)

if not selected_series:
    st.info("Please select at least one series.")
    st.stop()

view_mode = st.sidebar.radio(
    "View mode",
    ["Levels (original values)", "Year-over-year change (%)"],
    index=0,
)

# Apply time window and series filters
if time_window == "Last 12 months":
    cutoff = latest_overall_date - pd.DateOffset(months=12)
    work_df = df[df["date"] >= cutoff].copy()
else:
    work_df = df.copy()

work_df = work_df[work_df["series_name"].isin(selected_series)].copy()

if work_df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# Prepare data for plotting
plot_df = work_df.copy()

if view_mode == "Year-over-year change (%)":
    plot_df = plot_df.dropna(subset=["yoy_pct"]).copy()
    plot_df["plot_value"] = plot_df["yoy_pct"]
    plot_df["plot_unit"] = "Percent (YoY)"
else:
    plot_df["plot_value"] = plot_df["value"]
    plot_df["plot_unit"] = plot_df["unit"]

plot_df = plot_df.sort_values(["date", "series_name"])
plot_df["Month"] = plot_df["date"].dt.strftime("%Y-%m")

st.subheader("Trends over time")

# Separate charts so Y-axis ranges are meaningful
for unit_name, g in plot_df.groupby("plot_unit"):
    if g.empty:
        continue

    chart_data = g.pivot(index="Month", columns="series_name", values="plot_value")

    st.markdown(f"**{unit_name}**")
    st.line_chart(
        chart_data,
        x_label="Month (YYYY-MM)",
        y_label=unit_name,
        use_container_width=True,
    )

# Latest month summary (levels + YoY)
latest_shown_date = work_df["date"].max()
latest_label = latest_shown_date.strftime("%Y-%m")
st.subheader(f"Latest month shown: {latest_label}")

latest_subset = df[
    (df["date"] == latest_shown_date) & (df["series_name"].isin(selected_series))
].copy()

latest_levels = latest_subset.set_index("series_name")["value"].reindex(selected_series)
latest_yoy = latest_subset.set_index("series_name")["yoy_pct"].reindex(selected_series)

cols = st.columns(min(4, len(selected_series)) or 1)

for i, name in enumerate(selected_series):
    level = latest_levels.get(name, np.nan)
    yoy = latest_yoy.get(name, np.nan)

    if pd.isna(level):
        display_value = "N/A"
        delta_text = ""
    else:
        display_value = f"{level:,.2f}"
        delta_text = "" if pd.isna(yoy) else f"{yoy:+.2f}% vs. 12 months ago"

    with cols[i % len(cols)]:
        st.metric(label=name, value=display_value, delta=delta_text)

# Data table and CSV download
st.divider()
st.subheader("Data table")

table_df = work_df.sort_values(["series_name", "date"]).copy()

table_df = table_df[
    [
        "series_name",
        "Month",
        "value",
        "yoy_pct",
        "periodName",
        "year",
        "series_id",
    ]
]

table_df = table_df.rename(
    columns={
        "series_name": "Series",
        "value": "Value",
        "yoy_pct": "YoY % (vs prior year)",
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
        "YoY % (vs prior year)": st.column_config.NumberColumn(
            "YoY % (vs prior year)", format="%.2f"
        ),
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
