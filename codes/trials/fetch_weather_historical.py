"""
Fetch historical weather data for 2016-2024
To match the power usage data date range
"""

import pandas as pd
import requests
from datetime import datetime
import os

print("="*80)
print("FETCHING HISTORICAL WEATHER DATA (2016-2024)")
print("="*80)

# Configuration
OUTPUT_DIR = '../dataset/preprocessed'
WEATHER_FILE = os.path.join(OUTPUT_DIR, 'weather_data_historical.parquet')

# Philadelphia coordinates
LATITUDE = 39.95
LONGITUDE = -75.16
API_URL = "https://archive-api.open-meteo.com/v1/archive"

# Date range for power data
START_DATE = "2016-01-01"
END_DATE = "2024-12-31"

print(f"\nFetching weather data for: {START_DATE} to {END_DATE}")
print(f"Location: Philadelphia ({LATITUDE}, {LONGITUDE})")

# Essential weather variables only
essential_variables = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "shortwave_radiation",
    "wind_speed_10m",
]

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": ','.join(essential_variables),
    "timezone": "America/New_York"
}

print(f"\nRequesting {len(essential_variables)} weather variables...")
for var in essential_variables:
    print(f"  • {var}")

try:
    print(f"\nMaking API request... (this may take a minute)")
    response = requests.get(API_URL, params=params, timeout=300)
    response.raise_for_status()

    data = response.json()
    hourly_data = data.get('hourly', {})

    if hourly_data:
        print(f"✓ API response received")

        # Create DataFrame
        weather_df = pd.DataFrame(hourly_data)
        weather_df['time'] = pd.to_datetime(weather_df['time'])
        weather_df = weather_df.rename(columns={'time': 'datetime_beginning_ept'})

        print(f"  Rows: {len(weather_df):,}")
        print(f"  Date range: {weather_df['datetime_beginning_ept'].min()} to {weather_df['datetime_beginning_ept'].max()}")

        # Add derived features
        print(f"\nAdding derived features...")
        weather_df['temp_squared'] = weather_df['temperature_2m'] ** 2
        weather_df['cooling_degree_hours'] = weather_df['temperature_2m'].apply(lambda x: max(0, x - 18))
        weather_df['heating_degree_hours'] = weather_df['temperature_2m'].apply(lambda x: max(0, 18 - x))
        weather_df['heat_index'] = (
            weather_df['temperature_2m'] +
            0.5 * (weather_df['relative_humidity_2m'] / 100) *
            (weather_df['temperature_2m'] - 14.5)
        )
        weather_df['discomfort_index'] = (
            weather_df['temperature_2m'] -
            0.55 * (1 - weather_df['relative_humidity_2m']/100) *
            (weather_df['temperature_2m'] - 14.5)
        )
        weather_df['is_daytime'] = (weather_df['shortwave_radiation'] > 0).astype(int)
        weather_df['wind_chill'] = weather_df.apply(
            lambda row: (
                13.12 + 0.6215*row['temperature_2m'] -
                11.37*(row['wind_speed_10m']**0.16) +
                0.3965*row['temperature_2m']*(row['wind_speed_10m']**0.16)
            ) if row['temperature_2m'] < 10 else row['temperature_2m'],
            axis=1
        )

        print(f"  Added 7 derived features")
        print(f"  Final columns: {len(weather_df.columns)}")

        # Save
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        weather_df.to_parquet(WEATHER_FILE, index=False)
        parquet_size_mb = os.path.getsize(WEATHER_FILE) / 1024 / 1024
        print(f"\n✓ Saved: {WEATHER_FILE}")
        print(f"  Size: {parquet_size_mb:.2f} MB")

        # Also save CSV
        csv_file = WEATHER_FILE.replace('.parquet', '.csv')
        weather_df.to_csv(csv_file, index=False)
        csv_size_mb = os.path.getsize(csv_file) / 1024 / 1024
        print(f"✓ Saved: {csv_file}")
        print(f"  Size: {csv_size_mb:.2f} MB")

        # Summary
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"Total records: {len(weather_df):,}")
        print(f"Total features: {len(weather_df.columns)}")
        print(f"Date range: {weather_df['datetime_beginning_ept'].min()} to {weather_df['datetime_beginning_ept'].max()}")

        # Check for missing values
        missing = weather_df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            print(f"\n⚠ Missing values:")
            print(missing)
        else:
            print(f"\n✓ No missing values")

        print(f"\n{'='*80}")
        print(f"✓ HISTORICAL WEATHER DATA FETCH COMPLETED")
        print(f"{'='*80}")

    else:
        print(f"\n✗ Error: No hourly data in API response")

except Exception as e:
    print(f"\n✗ Error: {e}")
    raise
