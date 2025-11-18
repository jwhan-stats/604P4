#!/usr/bin/env python3
"""
Update test_features.parquet with new data for all models:
1. XGBoost Regressor (hourly-level MW predictions with business day aware masking)
2. XGBoost Classifier (peak_hour) - uses daily aggregates
3. XGBoost Classifier (peak_days) - uses daily aggregates

Combines 2021-2024 November data + 2025+ updates
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def get_black_friday(year):
    """Returns the Black Friday date for a given year"""
    nov_first = datetime(year, 11, 1)
    days_until_thursday = (3 - nov_first.weekday()) % 7
    first_thursday = nov_first + timedelta(days=days_until_thursday)
    thanksgiving = first_thursday + timedelta(weeks=3)
    black_friday = thanksgiving + timedelta(days=1)
    return black_friday

def create_all_features(df):
    """
    Create ALL features needed for all three models

    Features created:
    1. XGBoost Regressor (hourly MW predictions):
       - Annual lags: 1yr, 2yr, 3yr, 4yr ago
       - Weekly lags: 2d, 3d, 4d, 5d, 6d, 7d, 8d, 9d ago (48h, 72h, 96h, 120h, 144h, 168h, 192h, 216h)
       - Indicators: is_blackfriday, is_thanksgiving
       - Handles missing data natively (-999 as indicator)
       - Business day aware masking support

    2. Daily aggregates (for XGBoost Classifier models):
       - peak_hour: hour with max MW each day
       - peak_load: max MW each day

    Args:
        df: DataFrame with 'datetime_beginning_ept', 'load_area', 'mw' columns

    Returns:
        Tuple of (hourly_df, daily_df)
    """
    print("Creating features for all models...")

    # Ensure datetime
    df['datetime_beginning_ept'] = pd.to_datetime(df['datetime_beginning_ept'])

    # Sort by load_area and datetime
    df = df.sort_values(['load_area', 'datetime_beginning_ept']).reset_index(drop=True)

    # Basic time features
    df['year'] = df['datetime_beginning_ept'].dt.year
    df['hour'] = df['datetime_beginning_ept'].dt.hour
    df['date'] = df['datetime_beginning_ept'].dt.date

    # ========== 1. Black Friday indicators ==========
    print("  - Black Friday indicators")
    years = df['year'].unique()
    black_fridays = {year: get_black_friday(year) for year in years}

    df['black_friday'] = df['year'].map(black_fridays)
    df['days_from_bf'] = (df['datetime_beginning_ept'] - df['black_friday']).dt.days
    df['is_blackfriday'] = (df['days_from_bf'] == 0).astype(int)
    df['is_thanksgiving'] = (df['days_from_bf'] == -1).astype(int)

    # ========== 2. Hourly lag features (for XGBoost Regressor 24h predictions) ==========
    print("  - Hourly lag features (XGBoost Regressor)")
    lag_hours = [8760, 17520, 26280, 35040,  # Annual: 1yr, 2yr, 3yr, 4yr
                 48, 72, 96, 120, 144, 168, 192, 216]  # Weekly: 2d, 3d, 4d, 5d, 6d, 7d, 8d, 9d

    for lag in lag_hours:
        df[f'mw_lag_{lag}h'] = df.groupby('load_area')['mw'].shift(lag)

    # ========== 3. Create daily aggregates ==========
    print("  - Daily aggregates (for XGBoost Classifier models)")

    # For each (load_area, date), find:
    # - peak_hour: hour with max MW
    # - peak_load: max MW value

    daily_data = []
    for (area, date), group in df.groupby(['load_area', 'date']):
        max_idx = group['mw'].idxmax()
        daily_data.append({
            'load_area': area,
            'date': date,
            'datetime': pd.Timestamp(date),
            'peak_hour': df.loc[max_idx, 'hour'] if pd.notna(max_idx) else np.nan,
            'peak_load': group['mw'].max(),
            'year': df.loc[max_idx, 'year'] if pd.notna(max_idx) else np.nan,
            'days_from_bf': df.loc[max_idx, 'days_from_bf'] if pd.notna(max_idx) else np.nan
        })

    daily_df = pd.DataFrame(daily_data)
    print(f"  - Created {len(daily_df):,} daily records")

    # Keep only necessary columns for hourly data
    hourly_cols = ['datetime_beginning_ept', 'load_area', 'mw', 'year', 'hour', 'date',
                   'is_blackfriday', 'is_thanksgiving', 'days_from_bf'] + \
                  [f'mw_lag_{lag}h' for lag in lag_hours]

    hourly_df = df[hourly_cols]

    print("✓ Features created")
    return hourly_df, daily_df

def update_test_features(
    november_csv='../dataset/test/hrl_load_metered_november.csv',
    update_csv='../dataset/raw/hrl_load_metered_update.csv',
    output_parquet='../dataset/test/test_features.parquet',
    output_daily_parquet='../dataset/test/test_daily_features.parquet'
):
    """
    Update test_features.parquet with latest data

    Combines:
    - 2021-2024 November data (historical for lag features)
    - 2025+ update data (new data to predict)

    Creates two outputs:
    - test_features.parquet: Hourly-level features (for XGBoost Regressor)
    - test_daily_features.parquet: Daily aggregates (for XGBoost Classifier models)

    Args:
        november_csv: 2021-2024 November data
        update_csv: Daily update CSV with latest data
        output_parquet: Output path for hourly features
        output_daily_parquet: Output path for daily features
    """
    print("="*80)
    print("UPDATE TEST FEATURES (ALL MODELS)")
    print("="*80)

    # Load November data (2021-2024)
    november_path = Path(november_csv)
    if not november_path.exists():
        print(f"❌ Error: {november_csv} not found")
        print(f"   Please create this file with 2021-2024 November data")
        return

    print(f"\n1. Loading November data (2021-2024) from {november_csv}")
    november_df = pd.read_csv(november_csv)
    november_df['datetime_beginning_ept'] = pd.to_datetime(november_df['datetime_beginning_ept'])
    print(f"   November data: {len(november_df):,} rows")
    print(f"   Date range: {november_df['datetime_beginning_ept'].min()} to {november_df['datetime_beginning_ept'].max()}")
    print(f"   Load areas: {november_df['load_area'].nunique()}")

    # Load update data (2025+)
    update_path = Path(update_csv)
    if not update_path.exists():
        print(f"\n⚠ Warning: {update_csv} not found")
        print(f"   Using only November data")
        combined_df = november_df.copy()
    else:
        print(f"\n2. Loading update data (2025+) from {update_csv}")
        update_df = pd.read_csv(update_csv)
        update_df['datetime_beginning_ept'] = pd.to_datetime(update_df['datetime_beginning_ept'])
        print(f"   Update data: {len(update_df):,} rows")
        print(f"   Date range: {update_df['datetime_beginning_ept'].min()} to {update_df['datetime_beginning_ept'].max()}")
        print(f"   Load areas: {update_df['load_area'].nunique()}")

        # Combine November + Update
        print(f"\n3. Combining data")
        combined_df = pd.concat([november_df, update_df], ignore_index=True)

        # Remove duplicates (keep last = update data)
        combined_df = combined_df.drop_duplicates(
            subset=['datetime_beginning_ept', 'load_area'],
            keep='last'
        )
        combined_df = combined_df.sort_values(['load_area', 'datetime_beginning_ept']).reset_index(drop=True)

        print(f"   Combined: {len(combined_df):,} rows")
        print(f"   Date range: {combined_df['datetime_beginning_ept'].min()} to {combined_df['datetime_beginning_ept'].max()}")

    # Select only needed columns
    needed_cols = ['datetime_beginning_ept', 'load_area', 'mw']
    combined_df = combined_df[needed_cols]

    # Create features
    print(f"\n4. Creating features for all models")
    hourly_df, daily_df = create_all_features(combined_df)

    # Save hourly features
    print(f"\n5. Saving hourly features to {output_parquet}")
    output_path = Path(output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hourly_df.to_parquet(output_path, index=False)

    hourly_size_mb = output_path.stat().st_size / (1024 * 1024)

    # Save daily features
    print(f"\n6. Saving daily features to {output_daily_parquet}")
    daily_output_path = Path(output_daily_parquet)
    daily_df.to_parquet(daily_output_path, index=False)

    daily_size_mb = daily_output_path.stat().st_size / (1024 * 1024)

    print(f"\n{'='*80}")
    print(f"✓ UPDATE COMPLETE")
    print(f"{'='*80}")

    print(f"\nHourly features (for XGBoost Regressor):")
    print(f"  Output: {output_path}")
    print(f"  Rows: {len(hourly_df):,}")
    print(f"  Columns: {len(hourly_df.columns)}")
    print(f"  Size: {hourly_size_mb:.2f} MB")
    print(f"  Date range: {hourly_df['datetime_beginning_ept'].min()} to {hourly_df['datetime_beginning_ept'].max()}")

    print(f"\nDaily features (for XGBoost Classifier models):")
    print(f"  Output: {daily_output_path}")
    print(f"  Rows: {len(daily_df):,}")
    print(f"  Columns: {len(daily_df.columns)}")
    print(f"  Size: {daily_size_mb:.2f} MB")
    print(f"  Date range: {daily_df['datetime'].min()} to {daily_df['datetime'].max()}")

    print(f"\nFeatures created:")
    print(f"  - Hourly MW lag features (XGBoost Regressor)")
    print(f"  - Daily peak hour aggregates (XGBoost peak_hour)")
    print(f"  - Daily peak load aggregates (XGBoost peak_days)")
    print(f"{'='*80}")

def main():
    update_test_features()

if __name__ == '__main__':
    main()
