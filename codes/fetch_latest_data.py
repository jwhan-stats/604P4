#!/usr/bin/env python3
"""
Fetch PJM HRL load data using the PJM Data Miner 2 API.

- API key is loaded from environment variable: PJM_API_KEY
- Downloads November 2025 data by default
- Output path: dataset/raw/hrl_load_metered_update.csv
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
import pandas as pd
import requests
import shutil


# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
BASE_URL = "https://api.pjm.com/api/v1"
DATASET = "hrl_load_metered"

# API KEY from environment variable
API_KEY_ENV = "PJM_API_KEY"

# Rows per API call
ROW_COUNT = 50_000

# Target month/year
TARGET_YEAR = 2025
TARGET_MONTH = 11

# Output location
RAW_DATA_DIR = Path(__file__).parent.parent / "dataset" / "raw"
FINAL_FILE = RAW_DATA_DIR / "hrl_load_metered_update.csv"

# Datetime formatting
DATETIME_COLUMNS = ["datetime_beginning_utc", "datetime_beginning_ept"]
DATETIME_FORMAT = "%-m/%-d/%Y %-I:%M:%S %p"


# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------
def month_bounds(year: int, month: int) -> tuple[str, str]:
    start = pd.Timestamp(year=year, month=month, day=1)
    end_excl = start + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1)
    return start.strftime("%Y-%m-%d"), end_excl.strftime("%Y-%m-%d")


def _format_datetime(series: pd.Series, fmt: str) -> pd.Series:
    try:
        return series.dt.strftime(fmt)
    except ValueError:
        return series.dt.strftime(fmt.replace("%-", "%#"))


def normalize_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in DATETIME_COLUMNS:
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        mask = parsed.notna()
        if mask.any():
            no_tz = parsed[mask].dt.tz_localize(None, nonexistent="NaT", ambiguous="NaT")
            formatted = _format_datetime(no_tz, DATETIME_FORMAT)
            df.loc[mask, col] = formatted.values
    return df


def fetch_chunk(dataset, api_key, start_date, end_date, start_row, row_count):
    url = f"{BASE_URL}/{dataset}"

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    date_filter = (
        f"{start_ts.strftime('%m/%d/%Y %H:%M')}to"
        f"{end_ts.strftime('%m/%d/%Y %H:%M')}"
    )

    params = [
        ("startRow", start_row),
        ("rowCount", row_count),
        ("sort", "datetime_beginning_ept"),
        ("order", "Asc"),
        ("datetime_beginning_ept", date_filter),
    ]

    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Accept": "application/json",
    }

    response = requests.get(url, headers=headers, params=params, timeout=90)
    response.raise_for_status()

    payload = response.json()
    if "errors" in payload:
        raise RuntimeError(payload["errors"])

    items = payload.get("items")
    if items is None:
        return pd.DataFrame()

    if isinstance(items, list):
        return pd.DataFrame(items)
    if isinstance(items, dict):
        return pd.DataFrame([items])
    return pd.DataFrame()


def download_month(dataset, api_key, year, month, out_csv, row_count):
    start_date, end_date = month_bounds(year, month)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    start_row = 1

    print(f"[PJM] Downloading {dataset} {start_date} → {end_date}")

    while True:
        chunk = fetch_chunk(dataset, api_key, start_date, end_date, start_row, row_count)
        if chunk.empty:
            break

        frames.append(chunk)
        count = len(chunk)
        print(f"  • Rows {start_row}–{start_row + count - 1} ({count} rows)")

        start_row += count
        if count < row_count:
            break

    if not frames:
        raise RuntimeError("No data returned from PJM API.")

    df = pd.concat(frames, ignore_index=True)
    normalize_datetime_columns(df)
    df.to_csv(out_csv, index=False)

    print(f"[PJM] Saved {len(df):,} rows → {out_csv}")
    return out_csv


# --------------------------------------------------------------------
# Main fetch function
# --------------------------------------------------------------------
def fetch_latest_update_data() -> bool:
    print("=" * 80)
    print("FETCHING PJM HRL LOAD DATA (Data Miner 2 API)")
    print("=" * 80)

    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        print(f"✗ Missing API key. Set environment variable: {API_KEY_ENV}")
        return False

    try:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

        temp_path = RAW_DATA_DIR / "hrl_temp.csv"

        csv_path = download_month(
            DATASET, api_key, TARGET_YEAR, TARGET_MONTH, temp_path, ROW_COUNT
        )

        shutil.move(csv_path, FINAL_FILE)

        size_kb = FINAL_FILE.stat().st_size / 1024
        print(f"\n✓ Download successful → {FINAL_FILE} ({size_kb:.2f} KB)")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("=" * 80)
        return False


def main():
    return 0 if fetch_latest_update_data() else 1


if __name__ == "__main__":
    sys.exit(main())
