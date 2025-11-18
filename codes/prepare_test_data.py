#!/usr/bin/env python3
"""
Prepare test data for all three models:
1. XGBoost Regressor (MW predictions) - hourly features
2. XGBoost Classifier (peak_hour) - daily features
3. XGBoost Classifier (peak_days) - daily features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

def get_black_friday(year):
    """Returns the Black Friday date for a given year"""
    nov_first = datetime(year, 11, 1)
    days_until_thursday = (3 - nov_first.weekday()) % 7
    first_thursday = nov_first + timedelta(days=days_until_thursday)
    thanksgiving = first_thursday + timedelta(weeks=3)
    black_friday = thanksgiving + timedelta(days=1)
    return black_friday

def prepare_test_features(target_date_str, data_path='../dataset/test/test_features.parquet'):
    """
    Prepare test features for all models

    Args:
        target_date_str: Target date in 'YYYY-MM-DD' format
        data_path: Path to test features parquet

    Returns:
        DataFrame with test features
    """
    # Parse target date
    target_date = pd.to_datetime(target_date_str)
    print(f"Target date: {target_date.strftime('%Y-%m-%d')}")

    # Load test features
    print(f"Loading test features from {data_path}...")
    df = pd.read_parquet(data_path)
    df['datetime_beginning_ept'] = pd.to_datetime(df['datetime_beginning_ept'])

    print(f"Available data range: {df['datetime_beginning_ept'].min()} to {df['datetime_beginning_ept'].max()}")

    # Define required features for each model
    XGBREG_FEATURES = [
        'mw_lag_8760h', 'mw_lag_17520h', 'mw_lag_26280h', 'mw_lag_35040h',  # Annual lags
        'mw_lag_48h', 'mw_lag_72h', 'mw_lag_96h', 'mw_lag_120h', 'mw_lag_144h',
        'mw_lag_168h', 'mw_lag_192h', 'mw_lag_216h',  # Weekly lags
        'is_blackfriday', 'is_thanksgiving'  # Indicators
    ]

    XGB_HOUR_FEATURES = [
        'day_of_week', 'month', 'day_of_month',
        'is_blackfriday', 'is_thanksgiving', 'days_from_bf',
        'peak_hour_lag_2d', 'peak_hour_lag_3d', 'peak_hour_lag_4d', 'peak_hour_lag_5d', 'peak_hour_lag_6d',
        'peak_hour_lag_7d', 'peak_hour_lag_8d', 'peak_hour_lag_9d',
        'peak_hour_bf_lag_1yr', 'peak_hour_bf_lag_2yr', 'peak_hour_bf_lag_3yr', 'peak_hour_bf_lag_4yr'
    ]

    XGB_DAYS_FEATURES = [
        'pred_peak_load_day0', 'pred_peak_load_day1', 'pred_peak_load_day2', 'pred_peak_load_day3',
        'pred_peak_load_day4', 'pred_peak_load_day5', 'pred_peak_load_day6', 'pred_peak_load_day7',
        'pred_peak_load_day8', 'pred_peak_load_day9',  # 10 features (BF-8 to BF+1)
        'bf_day_index',
        'days_from_bf', 'day_of_week', 'is_blackfriday', 'is_thanksgiving',
        'actual_peak_load_lag1d', 'actual_peak_load_lag2d', 'actual_peak_load_lag3d',
        'actual_peak_load_lag4d', 'actual_peak_load_lag5d', 'actual_peak_load_lag6d',
        'actual_peak_load_lag7d', 'actual_peak_load_lag8d', 'actual_peak_load_lag9d',
        'actual_peak_bf_lag_1yr', 'actual_peak_bf_lag_2yr'
    ]

    # Get load areas (exclude RTO and AE)
    load_areas = sorted([a for a in df['load_area'].unique() if a not in ['RTO', 'AE']])

    # Calculate Black Friday info for target date
    year = target_date.year
    black_friday = get_black_friday(year)
    days_from_bf = (target_date - black_friday).days
    is_blackfriday = 1 if days_from_bf == 0 else 0
    is_thanksgiving = 1 if days_from_bf == -1 else 0
    bf_day_index = days_from_bf - (-8)  # BF-8=0, BF-7=1, ..., BF+1=9

    print(f"Black Friday {year}: {black_friday.strftime('%Y-%m-%d')}")
    print(f"Target is Black Friday: {is_blackfriday}")
    print(f"Target is Thanksgiving: {is_thanksgiving}")
    print(f"Days from BF: {days_from_bf}, BF day index: {bf_day_index}")

    # ========== STEP 1: Create hourly test data with XGBoost Regressor features ==========
    test_rows = []

    # Maximum lag needed: 216h (9 days)
    max_lag_hours = 216
    min_required_date = target_date - timedelta(hours=max_lag_hours)

    print(f"\nRequired historical data: from {min_required_date.strftime('%Y-%m-%d')} to {target_date.strftime('%Y-%m-%d')}")

    # Check if we have enough historical data
    if df['datetime_beginning_ept'].max() < min_required_date:
        print(f"❌ Error: Not enough historical data")
        print(f"   Need data until at least {min_required_date.strftime('%Y-%m-%d')}")
        return None

    print("\n[Step 1] Creating hourly features for XGBoost Regressor...")

    # Add temporal columns to df
    df['date'] = df['datetime_beginning_ept'].dt.date
    df['year'] = df['datetime_beginning_ept'].dt.year
    df['hour'] = df['datetime_beginning_ept'].dt.hour

    # Calculate days_from_bf for historical data
    df['days_from_bf'] = 0
    for yr in df['year'].unique():
        bf_date = get_black_friday(yr)
        mask = df['year'] == yr
        df.loc[mask, 'days_from_bf'] = (df.loc[mask, 'datetime_beginning_ept'].dt.date - bf_date.date()).apply(lambda x: x.days)

    for area in load_areas:
        area_data = df[df['load_area'] == area].set_index('datetime_beginning_ept').sort_index()

        for hour in range(24):
            # Target datetime
            target_dt = target_date + timedelta(hours=hour)

            # Create feature dict
            row = {
                'load_area': area,
                'datetime_beginning_ept': target_dt,
                'hour': hour,
                'is_blackfriday': is_blackfriday,
                'is_thanksgiving': is_thanksgiving,
                'days_from_bf': days_from_bf,
                'day_of_week': target_date.dayofweek,
                'month': target_date.month,
                'day_of_month': target_date.day,
                'bf_day_index': bf_day_index
            }

            # Look up hourly lag features from historical data
            lag_hours = [48, 72, 96, 120, 144, 168, 192, 216, 8760, 17520, 26280, 35040]

            for lag_h in lag_hours:
                lag_dt = target_dt - timedelta(hours=lag_h)

                # Look up the MW value at lag_dt from historical data
                if lag_dt in area_data.index:
                    row[f'mw_lag_{lag_h}h'] = area_data.loc[lag_dt, 'mw']
                else:
                    row[f'mw_lag_{lag_h}h'] = np.nan

            test_rows.append(row)

    # Create DataFrame
    test_df = pd.DataFrame(test_rows)
    print(f"  ✓ Created hourly data: {len(test_df)} rows")

    # ========== STEP 2: Add daily peak hour lag features (for XGBoost peak_hour) ==========
    print("\n[Step 2] Creating daily peak hour lag features...")

    # Build daily peak hour lookup from historical data
    print("  Building historical peak hour lookup...")
    daily_peaks = df.groupby(['load_area', 'date']).agg({
        'mw': lambda x: df.loc[x.idxmax(), 'hour'] if x.notna().any() else np.nan,
        'year': 'first',
        'days_from_bf': 'first'
    }).reset_index()
    daily_peaks.rename(columns={'mw': 'peak_hour'}, inplace=True)

    # BF-based peak hour lookup
    bf_peak_lookup = daily_peaks.set_index(['load_area', 'days_from_bf', 'year'])['peak_hour'].to_dict()

    # Daily peak hour lookup for lag days
    daily_peaks['datetime'] = pd.to_datetime(daily_peaks['date'])
    daily_peak_lookup = daily_peaks.set_index(['load_area', 'datetime'])['peak_hour'].to_dict()

    # Add peak hour lag features for BF-based lags
    for lag_yr in [1, 2, 3, 4]:
        print(f"  Creating peak_hour_bf_lag_{lag_yr}yr...")

        def get_peak_bf_lag(row):
            key = (row['load_area'], row['days_from_bf'], year - lag_yr)
            return bf_peak_lookup.get(key, np.nan)

        test_df[f'peak_hour_bf_lag_{lag_yr}yr'] = test_df.apply(get_peak_bf_lag, axis=1)

    # Add daily peak hour lag features (2d to 9d)
    for lag_d in [2, 3, 4, 5, 6, 7, 8, 9]:
        print(f"  Creating peak_hour_lag_{lag_d}d...")

        def get_peak_daily_lag(row):
            lag_date = target_date - timedelta(days=lag_d)
            key = (row['load_area'], pd.Timestamp(lag_date))
            return daily_peak_lookup.get(key, np.nan)

        test_df[f'peak_hour_lag_{lag_d}d'] = test_df.apply(get_peak_daily_lag, axis=1)

    print(f"  ✓ Added peak hour lag features")

    # ========== STEP 3: Add daily peak load lag features (for XGBoost peak_days) ==========
    print("\n[Step 3] Creating daily peak load lag features for peak_days...")

    # Build daily peak load lookup
    daily_peak_loads = df.groupby(['load_area', 'date']).agg({
        'mw': 'max',
        'year': 'first',
        'days_from_bf': 'first'
    }).reset_index()
    daily_peak_loads.rename(columns={'mw': 'peak_load'}, inplace=True)
    daily_peak_loads['datetime'] = pd.to_datetime(daily_peak_loads['date'])

    # BF-based peak load lookup
    bf_peak_load_lookup = daily_peak_loads.set_index(['load_area', 'days_from_bf', 'year'])['peak_load'].to_dict()

    # Daily peak load lookup for lag days
    daily_peak_load_lookup = daily_peak_loads.set_index(['load_area', 'datetime'])['peak_load'].to_dict()

    # Add actual peak load lag features (1d to 9d)
    for lag_d in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        print(f"  Creating actual_peak_load_lag{lag_d}d...")

        def get_load_daily_lag(row):
            lag_date = target_date - timedelta(days=lag_d)
            key = (row['load_area'], pd.Timestamp(lag_date))
            return daily_peak_load_lookup.get(key, np.nan)

        test_df[f'actual_peak_load_lag{lag_d}d'] = test_df.apply(get_load_daily_lag, axis=1)

    # Add BF-based peak load lag features (1yr, 2yr)
    for lag_yr in [1, 2]:
        print(f"  Creating actual_peak_bf_lag_{lag_yr}yr...")

        def get_load_bf_lag(row):
            key = (row['load_area'], row['days_from_bf'], year - lag_yr)
            return bf_peak_load_lookup.get(key, np.nan)

        test_df[f'actual_peak_bf_lag_{lag_yr}yr'] = test_df.apply(get_load_bf_lag, axis=1)

    print(f"  ✓ Added peak load lag features")

    # ========== STEP 4: Add predicted peak load features (4-year average) for peak_days ==========
    print("\n[Step 4] Creating predicted peak load features (4-year average)...")

    # For each day in BF period (BF-8 to BF+1), calculate 4-year average
    for day_offset in range(-8, 2):  # BF-8 to BF+1
        day_idx = day_offset - (-8)  # 0 to 9
        print(f"  Creating pred_peak_load_day{day_idx} (BF{day_offset:+d})...")

        def get_pred_peak_load(row):
            # Get historical values for this days_from_bf
            key_list = [(row['load_area'], day_offset, year - i) for i in [1, 2, 3, 4]]
            values = [bf_peak_load_lookup.get(key, np.nan) for key in key_list]
            values = [v for v in values if not np.isnan(v)]
            return np.mean(values) if values else np.nan

        test_df[f'pred_peak_load_day{day_idx}'] = test_df.apply(get_pred_peak_load, axis=1)

    print(f"  ✓ Added predicted peak load features")

    # ========== STEP 5: Fill missing values with -999 ==========
    print("\n[Step 5] Filling missing values with -999...")

    all_features = XGBREG_FEATURES + XGB_HOUR_FEATURES + XGB_DAYS_FEATURES
    feature_cols = [f for f in all_features if f in test_df.columns]

    # Check missing values before filling
    missing_counts = test_df[feature_cols].isnull().sum()
    if missing_counts.sum() > 0:
        print("  Missing values detected:")
        for feat, count in missing_counts[missing_counts > 0].items():
            pct = count / len(test_df) * 100
            print(f"    {feat}: {count} ({pct:.1f}%)")

        # Fill ALL missing values with -999
        for feat in feature_cols:
            if test_df[feat].isnull().any():
                test_df[feat] = test_df[feat].fillna(-999)

        print("  ✓ Filled ALL missing values with -999")
    else:
        print("  ✓ No missing values")

    # Reorder columns
    cols = ['load_area', 'datetime_beginning_ept', 'hour'] + feature_cols
    test_df = test_df[[c for c in cols if c in test_df.columns]]

    print(f"\n✓ Created test data: {len(test_df)} rows ({len(load_areas)} areas × 24 hours)")
    print(f"  XGBoost Regressor features: {len(XGBREG_FEATURES)}")
    print(f"  XGBoost peak_hour features: {len(XGB_HOUR_FEATURES)}")
    print(f"  XGBoost peak_days features: {len(XGB_DAYS_FEATURES)}")

    # Final check for missing values
    feature_cols_final = [c for c in test_df.columns if c not in ['load_area', 'datetime_beginning_ept', 'hour']]
    final_missing = test_df[feature_cols_final].isnull().sum().sum()
    if final_missing > 0:
        print(f"\n⚠ Warning: Still {final_missing} missing values!")
    else:
        print("\n✓ No missing values (all filled with -999)")

    return test_df

def main():
    # Get target date from command line or use today
    if len(sys.argv) > 1:
        target_date = sys.argv[1]
    else:
        # Default to today
        target_date = datetime.now().strftime('%Y-%m-%d')

    print("="*80)
    print("PREPARE TEST DATA FOR PREDICTIONS")
    print("="*80)
    print("XGBoost Regressor: MW predictions (hourly features)")
    print("XGBoost Classifier (peak_hour): Peak hour predictions (daily features)")
    print("XGBoost Classifier (peak_days): Peak day predictions (daily features)")
    print("="*80)

    # Prepare features
    test_df = prepare_test_features(target_date)

    if test_df is None:
        print("\n❌ Failed to prepare test data")
        print("   Make sure to run update_features.py first to create test_features.parquet")
        sys.exit(1)

    # Save to CSV
    output_path = Path('../dataset/test/test.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(output_path, index=False)

    print(f"\n✓ Test data saved to: {output_path}")
    print(f"  Rows: {len(test_df)}")
    print(f"  Columns: {len(test_df.columns)}")
    print("="*80)

if __name__ == '__main__':
    main()
