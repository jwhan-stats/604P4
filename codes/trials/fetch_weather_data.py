"""
Weather Data Fetcher for Power Usage Prediction
Fetches essential weather variables from Open-Meteo API
Only includes key variables to minimize data size
"""

import pandas as pd
import glob
import os
import requests
import json
from datetime import datetime, timedelta

# Configuration
DATA_DIR = '../dataset/raw'
OUTPUT_DIR = '../dataset/preprocessed'
WEATHER_FILE = os.path.join(OUTPUT_DIR, 'weather_data.csv')

# Philadelphia coordinates (for PJM data)
LATITUDE = 39.95
LONGITUDE = -75.16

print("="*80)
print("WEATHER DATA FETCHER - Essential Variables Only")
print("="*80)

# --- 1. Load CSV data and find date range ---
print("\n[1] Loading CSV files to determine date range...")
print(f"Searching for CSV files in: {DATA_DIR}")

csv_files = glob.glob(os.path.join(DATA_DIR, '**', '*.csv'), recursive=True)

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

print(f"Found {len(csv_files)} CSV file(s)")

# Parse all dates from CSV files
all_dates = []
for file_path in csv_files:
    file_name = os.path.basename(file_path)
    try:
        df = pd.read_csv(file_path)
        date_col = df.columns[0]  # Assume first column is datetime
        parsed_dates = pd.to_datetime(df[date_col], errors='coerce')
        all_dates.append(parsed_dates)
        print(f"  ✓ {file_name}")
    except Exception as e:
        print(f"  ✗ Error loading {file_name}: {e}")

# Combine all dates and find range
all_dates_combined = pd.concat(all_dates).dropna()
min_date = all_dates_combined.min()
max_date = all_dates_combined.max()

# Format for API
start_str = min_date.strftime('%Y-%m-%d')
end_str = max_date.strftime('%Y-%m-%d')

print(f"\nDate range detected:")
print(f"  Start: {start_str}")
print(f"  End:   {end_str}")
print(f"  Total days: {(max_date - min_date).days}")

# --- 2. Fetch Weather Data from Open-Meteo ---
print("\n[2] Fetching weather data from Open-Meteo API...")
print(f"Location: Philadelphia ({LATITUDE}, {LONGITUDE})")

API_URL = "https://archive-api.open-meteo.com/v1/archive"

# Only essential weather variables (marked with ⭐ for power demand prediction)
essential_variables = [
    "temperature_2m",           # ⭐ 2m temperature (°C) - CRITICAL for HVAC demand
    "apparent_temperature",     # ⭐ Feels-like temperature (°C) - Better predictor than actual temp
    "relative_humidity_2m",     # ⭐ Relative humidity (%) - Affects cooling efficiency
    "shortwave_radiation",      # ⭐ Solar radiation (W/m²) - Drives cooling demand + solar generation
    "wind_speed_10m",           # ⭐ Wind speed (km/h) - Affects wind chill and wind power
]

weather_params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": start_str,
    "end_date": end_str,
    "hourly": ','.join(essential_variables),
    "timezone": "America/New_York"  # Eastern Time for PJM
}

print(f"\nRequesting {len(essential_variables)} essential weather variables:")
for var in essential_variables:
    print(f"  • {var}")

try:
    response = requests.get(API_URL, params=weather_params, timeout=120)
    response.raise_for_status()
    weather_data = response.json()

    print(f"\n✓ Successfully fetched weather data")

    # --- 3. Process Weather Data ---
    print("\n[3] Processing weather data...")

    hourly_data = weather_data.get('hourly', {})
    if not hourly_data:
        raise ValueError("No 'hourly' data found in API response")

    # Create DataFrame
    weather_df = pd.DataFrame(hourly_data)
    weather_df['time'] = pd.to_datetime(weather_df['time'])
    weather_df = weather_df.rename(columns={'time': 'datetime_beginning_ept'})

    print(f"Weather data shape: {weather_df.shape}")
    print(f"Date range: {weather_df['datetime_beginning_ept'].min()} to {weather_df['datetime_beginning_ept'].max()}")

    # --- 4. Add Derived Features ---
    print("\n[4] Adding derived weather features...")

    # Temperature squared (captures non-linear effects)
    weather_df['temp_squared'] = weather_df['temperature_2m'] ** 2

    # ⭐ Cooling Degree Hours - cooling needed above 18°C (64°F base)
    weather_df['cooling_degree_hours'] = weather_df['temperature_2m'].apply(
        lambda x: max(0, x - 18)
    )

    # ⭐ Heating Degree Hours - heating needed below 18°C (64°F base)
    weather_df['heating_degree_hours'] = weather_df['temperature_2m'].apply(
        lambda x: max(0, 18 - x)
    )

    # ⭐ Heat Index (simplified approximation accounting for humidity)
    weather_df['heat_index'] = (
        weather_df['temperature_2m'] +
        0.5 * (weather_df['relative_humidity_2m'] / 100) *
        (weather_df['temperature_2m'] - 14.5)
    )

    # Discomfort Index
    weather_df['discomfort_index'] = (
        weather_df['temperature_2m'] -
        0.55 * (1 - weather_df['relative_humidity_2m']/100) *
        (weather_df['temperature_2m'] - 14.5)
    )

    # Daytime indicator (based on solar radiation)
    weather_df['is_daytime'] = (weather_df['shortwave_radiation'] > 0).astype(int)

    # Wind Chill (for cold weather, temperature < 10°C)
    weather_df['wind_chill'] = weather_df.apply(
        lambda row: (
            13.12 + 0.6215*row['temperature_2m'] -
            11.37*(row['wind_speed_10m']**0.16) +
            0.3965*row['temperature_2m']*(row['wind_speed_10m']**0.16)
        ) if row['temperature_2m'] < 10 else row['temperature_2m'],
        axis=1
    )

    print(f"Added 7 derived features")
    print(f"Final weather data shape: {weather_df.shape}")

    # --- 5. Save Weather Data ---
    print("\n[5] Saving weather data...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save as CSV
    weather_df.to_csv(WEATHER_FILE, index=False)
    csv_size_mb = os.path.getsize(WEATHER_FILE) / 1024 / 1024
    print(f"✓ Saved CSV: {WEATHER_FILE}")
    print(f"  Size: {csv_size_mb:.2f} MB")

    # Save as Parquet for faster loading
    parquet_file = WEATHER_FILE.replace('.csv', '.parquet')
    weather_df.to_parquet(parquet_file, index=False)
    parquet_size_mb = os.path.getsize(parquet_file) / 1024 / 1024
    print(f"✓ Saved Parquet: {parquet_file}")
    print(f"  Size: {parquet_size_mb:.2f} MB")
    print(f"  Compression ratio: {csv_size_mb/parquet_size_mb:.1f}x")

    # --- 6. Display Summary Statistics ---
    print("\n[6] Weather Data Summary:")
    print("="*80)

    # Select key columns for summary
    key_cols = [
        'temperature_2m', 'apparent_temperature', 'relative_humidity_2m',
        'wind_speed_10m', 'shortwave_radiation'
    ]

    print(weather_df[key_cols].describe().round(2))

    # Check for missing values
    print("\n" + "="*80)
    print("Missing Values Check:")
    missing = weather_df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print("⚠ Warning: Missing values detected:")
        print(missing)
    else:
        print("✓ No missing values!")

    print("\n" + "="*80)
    print("✓ WEATHER DATA FETCH COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nTotal records: {len(weather_df):,}")
    print(f"Total features: {len(weather_df.columns)}")
    print(f"  - Base weather variables: {len(essential_variables)}")
    print(f"  - Derived features: 7")
    print(f"\nOutput files:")
    print(f"  - {WEATHER_FILE}")
    print(f"  - {parquet_file}")
    print(f"\nMerge with power data using: 'datetime_beginning_ept'")

except requests.exceptions.RequestException as e:
    print(f"\n✗ Error fetching weather data: {e}")
    print("Check your internet connection and try again.")
except json.JSONDecodeError:
    print("\n✗ Error: Could not decode JSON response from weather API")
except Exception as e:
    print(f"\n✗ An error occurred during weather processing: {e}")
    raise
