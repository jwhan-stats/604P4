#!/usr/bin/env python3
"""
Extract November data (2021-2024) from full_year_features.parquet
This is a one-time setup script to create hrl_load_metered_november.csv
"""

import pandas as pd
from pathlib import Path

def create_november_csv(
    input_parquet='../dataset/preprocessed/full_year_features.parquet',
    output_csv='../dataset/test/hrl_load_metered_november.csv'
):
    """
    Extract November data from 2021-2024 for test features

    Args:
        input_parquet: Path to full year features parquet
        output_csv: Output path for November CSV
    """
    print("="*80)
    print("CREATE NOVEMBER DATA (2021-2024)")
    print("="*80)

    # Load full year features
    print(f"\n1. Loading data from {input_parquet}")
    df = pd.read_parquet(input_parquet)
    df['datetime_beginning_ept'] = pd.to_datetime(df['datetime_beginning_ept'])

    print(f"   Total data: {len(df):,} rows")
    print(f"   Date range: {df['datetime_beginning_ept'].min()} to {df['datetime_beginning_ept'].max()}")

    # Filter for November 2021-2024
    print(f"\n2. Filtering November data (2021-2024)")
    november_df = df[
        (df['datetime_beginning_ept'].dt.year >= 2021) &
        (df['datetime_beginning_ept'].dt.year <= 2024) &
        (df['datetime_beginning_ept'].dt.month == 11)
    ].copy()

    print(f"   November data: {len(november_df):,} rows")
    print(f"   Date range: {november_df['datetime_beginning_ept'].min()} to {november_df['datetime_beginning_ept'].max()}")
    print(f"   Load areas: {november_df['load_area'].nunique()}")

    # Select only raw columns (not features)
    raw_cols = ['datetime_beginning_ept', 'load_area', 'mw']

    # Include additional columns if they exist
    for col in ['datetime_beginning_utc', 'nerc_region', 'mkt_region', 'zone', 'is_verified']:
        if col in november_df.columns:
            raw_cols.insert(0 if col == 'datetime_beginning_utc' else len(raw_cols), col)

    november_df = november_df[raw_cols]

    # Save to CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n3. Saving to {output_csv}")
    november_df.to_csv(output_path, index=False)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"\n{'='*80}")
    print(f"✓ NOVEMBER DATA CREATED")
    print(f"{'='*80}")
    print(f"Output: {output_path}")
    print(f"Rows: {len(november_df):,}")
    print(f"Columns: {len(november_df.columns)} {list(november_df.columns)}")
    print(f"Size: {file_size_mb:.2f} MB")
    print(f"Years: {sorted(november_df['datetime_beginning_ept'].dt.year.unique())}")
    print(f"\n✓ This file will be used as the base for test_features.parquet")
    print(f"  Next step: Run update_features.py to create test_features.parquet")
    print(f"{'='*80}")

def main():
    create_november_csv()

if __name__ == '__main__':
    main()
