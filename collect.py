from datetime import datetime
from pathlib import Path
import pandas as pd
import requests


SERIES = {
    "Total Nonfarm Employment": "CES0000000001",
    "Unemployment Rate": "LNS14000000",
    "Labor Force Participation Rate": "LNS11300000",
    "Average Hourly Earnings for Private Employees": "CES0500000003",
    "Manufacturing Employment": "CES3000000001",
    "Education and Health Services Employment": "CES6500000001",
}

BLS_BASE_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data"
DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "labor_data.csv"


def fetch_series(series_id: str, start_year: int, end_year: int):
    url = f"{BLS_BASE_URL}/{series_id}"
    params = {"startyear": str(start_year), "endyear": str(end_year)}

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if payload.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS API error for {series_id}: {payload.get('message')}")

    return payload["Results"]["series"][0]["data"]


def normalize_records(series_name: str, series_id: str, raw_data: list[dict]) -> list[dict]:
    records = []

    for item in raw_data:
        period = item.get("period", "")
        year = item.get("year", "")

        # Keep monthly values only (M01..M12). Skip annual average M13.
        if not period.startswith("M") or period == "M13":
            continue

        month_number = int(period[1:])
        date_str = f"{year}-{month_number:02d}-01"

        value_str = item.get("value")
        try:
            value = float(value_str)
        except (TypeError, ValueError):
            continue

        records.append(
            {
                "series_name": series_name,
                "series_id": series_id,
                "date": date_str,
                "value": value,
                "period": period,
                "periodName": item.get("periodName", ""),
                "year": int(year),
            }
        )

    return records


def collect_data():
    # Pull enough history so YoY calculations work and you always have >= 12 months
    end_year = datetime.now().year
    start_year = end_year - 2

    print(f"Collecting BLS data for {start_year} to {end_year}...")

    new_records = []
    for series_name, series_id in SERIES.items():
        print(f"Fetching {series_name} ({series_id})...")
        raw_data = fetch_series(series_id, start_year, end_year)
        new_records.extend(normalize_records(series_name, series_id, raw_data))

    if not new_records:
        raise RuntimeError("No records collected from BLS API.")

    new_df = pd.DataFrame(new_records)
    new_df["date"] = pd.to_datetime(new_df["date"], errors="coerce")
    new_df = new_df.dropna(subset=["date", "value"])
    new_df = new_df.sort_values(["series_id", "date"])

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # If the CSV already exists, append and dedupe
    if DATA_FILE.exists():
        old_df = pd.read_csv(DATA_FILE, parse_dates=["date"])
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df

    # Drop duplicates by series_id + date (keep newest)
    combined = combined.drop_duplicates(subset=["series_id", "date"], keep="last")
    combined = combined.sort_values(["series_id", "date"])

    # Keep last 24 months per series (guarantees >= 1 year, helps YoY)
    combined = (
        combined.groupby("series_id", group_keys=False)
        .apply(lambda g: g.sort_values("date").tail(24))
        .reset_index(drop=True)
    )

    combined.to_csv(DATA_FILE, index=False)

    print(f"Saved {len(combined)} rows to {DATA_FILE}")
    print("Latest date per series:")
    print(combined.groupby("series_name")["date"].max())


if __name__ == "__main__":
    collect_data()
