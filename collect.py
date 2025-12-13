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
    params = {
        "startyear": str(start_year),
        "endyear": str(end_year),
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if payload.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS API error for {series_id}: {payload.get('message')}")

    # Results
    return payload["Results"]["series"][0]["data"]


def collect_data():
    """
    Collect data for all configured series, clean it with pandas,
    and save a single CSV file for the dashboard.
    """

    # Go back two years so we always have at least one full year of history
    end_year = datetime.now().year
    start_year = end_year - 2

    records = []

    for series_name, series_id in SERIES.items():
        print(f"Fetching {series_name} ({series_id})...")
        raw_data = fetch_series(series_id, start_year, end_year)

        for item in raw_data:
            period = item.get("period", "")
            year = item.get("year", "")

            # Only keep real months 
            if not period.startswith("M"):
                continue
            if period == "M13":  
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

    if not records:
        raise RuntimeError("No records collected from BLS API.")

    df = pd.DataFrame(records)

    # Basic cleaning and sorting
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "value"])
    df = df.sort_values(["series_name", "date"])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_FILE, index=False)

    print(f"Saved {len(df)} rows to {DATA_FILE}")


if __name__ == "__main__":
    collect_data()
