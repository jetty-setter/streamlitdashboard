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


def compute_indexed_100(df_in: pd.DataFrame) -> pd.Series:
    """
    Compute indexed series where the FIRST VISIBLE month per series is 100.
    This must be computed AFTER filtering to match what the user is viewing.
    """
    df_sorted = df_in.sort_values(["series_id", "date"]).copy()

    def _idx(s: pd.Series) -> pd.Series:
        if s.empty:
            return s
        base = s.iloc[0]
        if base == 0:
            return pd.Series([np.nan] * len(s), index=s.index)
        return (s / base) * 100.0

    return df_sorted.groupby("series_id")["value"].apply(_idx).reset_index(level=0, drop=True)


def nice_y_domain(series: pd.Series, pad_pct: float) -> list[float] | None:
    if series.empty:
        return None
    y_min = float(series.min())
    y_max = float(series.max())
    if not (np.isfinite(y_min) and np.isfinite(y_max)):
        return None
    if y_min == y_max:
        # Make a tiny domain around a flat line
        pad = 1.0 if y_min == 0 else abs(y_min) * 0.05
        return [y_min - pad, y_max + pad]
    pad = (y_max - y_min) * (pad_pct / 100.0)
    return [y_min - pad, y_max + pad]


def make_line_chart(df_long: pd.DataFrame, y_col: str, y_title: str, pad_pct: float) -> alt.Chart:
    if df_long.empty:
        return alt.Chart(pd.DataFrame({"date": [], "plot_value": [], "series_name": []})).mark_line()

    domain = nice_y_domain(df_long[y_col], pad_pct)

    chart = (
        alt.Chart(df_long)
        .mark_line(point=False)
        .encode(
            x=alt.X(
                "date:T",
                title="Month",
                axis=alt.Axis(format="%b %Y", labelAngle=-45),
            ),
            y=alt.Y(
                f"{y_col}:Q",
                title=y_title,
                scale=alt.Scale(domain=domain) if domain else alt.Scale(),
            ),
            color=alt.Color("series_name:N", title="Series"),
            tooltip=[
                alt.Tooltip("series_name:N", title="Series"),
                alt.Tooltip("date:T", title="Date", format="%Y-%m"),
                alt.Tooltip(f"{y_col}:Q", title=y_title, format=",.2f"),
            ],
        )
        .properties(height=340)
        .interactive()
    )
    return chart


# ---------- Load ----------
if not DATA_FILE.exists():
    st.warning("No data found. Please run `python collect.py` first.")
    st.stop()

df = load_data(DATA_FILE)
df["unit"] = df["series_name"].map(UNIT_MAP).fillna("Other")

# Year-over-year percent change (12 months) computed on full data
df["yoy_pct"] = df.groupby("series_id")["value"].pct_change(12) * 100.0
df["Month"] = df["date"].dt.strftime("%Y-%m")


# ---------- Sidebar ----------
st.sidebar.header("Filters")

latest_overall_date = df["date"].max()

time_window = st.sidebar.selectbox("Time window", ["Last 12 months", "All data"], index=0)

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

chart_layout = st.sidebar.radio(
    "Chart layout",
    ["Single chart (recommended)", "Group by unit (tabs)"],
    index=0,
)

pad_pct = st.sidebar.slider(
    "Y-axis zoom (padding %)",
    min_value=0,
    max_value=30,
    value=8,
    help="Lower values zoom in tighter. Higher values add more headroom.",
)

# ---------- Apply filters ----------
if time_window == "Last 12 months":
    # Use 13 months for stability (and so YoY doesn't look empty)
    cutoff = latest_overall_date - pd.DateOffset(months=13)
    work_df = df[df["date"] >= cutoff].copy()
else:
    work_df = df.copy()

work_df = work_df[work_df["series_name"].isin(selected_series)].copy()

if work_df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# Compute indexed AFTER filtering so the first visible month = 100
work_df = work_df.sort_values(["series_id", "date"]).copy()
work_df["indexed_100"] = compute_indexed_100(work_df)

# Decide plotting column + title
if view_mode == "Year-over-year change (%)":
    plot_df = work_df.dropna(subset=["yoy_pct"]).copy()
    y_col = "yoy_pct"
    y_title = "YoY % change"
elif view_mode == "Indexed (start=100)":
    plot_df = work_df.copy()
    y_col = "indexed_100"
    y_title = "Index (first visible month = 100)"
else:
    plot_df = work_df.copy()
    y_col = "value"
    # Keep the title general, units are shown via layout choice
    y_title = "Value"

st.subheader("Trends over time")

if view_mode == "Year-over-year change (%)" and plot_df.empty:
    st.info("YoY needs enough history to compute. Try 'All data' or 'Indexed'.")
else:
    if chart_layout == "Single chart (recommended)":
        # One legend, one chart, all selected series
        chart = make_line_chart(plot_df, y_col=y_col, y_title=y_title, pad_pct=pad_pct)
        st.altair_chart(chart, use_container_width=True)
    else:
        # Tabs by unit
        unit_order = ["Employment (thousands)", "Percent", "Dollars", "Other"]
        units_in_view = [u for u in unit_order if u in plot_df["unit"].unique()]

        tabs = st.tabs(units_in_view) if units_in_view else []
        for tab, unit_name in zip(tabs, units_in_view):
            with tab:
                unit_df = plot_df[plot_df["unit"] == unit_name].copy()
                if unit_df.empty:
                    st.info("No series selected in this unit group.")
                    continue

                st.markdown(f"**{unit_name}**")
                chart = make_line_chart(unit_df, y_col=y_col, y_title=y_title, pad_pct=pad_pct)
                st.altair_chart(chart, use_container_width=True)

# ---------- Latest month summary ----------
latest_shown_date = work_df["date"].max()
latest_label = latest_shown_date.strftime("%Y-%m")
st.subheader(f"Latest month shown: {latest_label}")

latest_subset = work_df[work_df["date"] == latest_shown_date].copy()

cols = st.columns(min(4, len(selected_series)) or 1)

for i, name in enumerate(selected_series):
    row = latest_subset[latest_subset["series_name"] == name]
    if row.empty:
        value_text = "N/A"
        delta_text = ""
    else:
        level = float(row["value"].iloc[0])
        yoy = row["yoy_pct"].iloc[0]
        value_text = f"{level:,.2f}"
        delta_text = "" if pd.isna(yoy) else f"{yoy:+.2f}% vs 12 months ago"

    with cols[i % len(cols)]:
        st.metric(label=name, value=value_text, delta=delta_text)

# ---------- Table ----------
st.divider()
st.subheader("Data table")

table_df = work_df.sort_values(["series_name", "date"]).copy()

table_df = table_df[
    ["series_name", "Month", "unit", "value", "yoy_pct", "indexed_100", "periodName", "year", "series_id"]
].rename(
    columns={
        "series_name": "Series",
        "unit": "Unit",
        "value": "Value",
        "yoy_pct": "YoY % (vs prior year)",
        "indexed_100": "Indexed (100=first visible)",
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
        "Indexed (100=first visible)": st.column_config.NumberColumn("Indexed (100=first visible)", format="%.2f"),
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
