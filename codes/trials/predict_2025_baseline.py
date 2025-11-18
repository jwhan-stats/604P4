#!/usr/bin/env python3
"""
Baseline prediction for 2025 Black Friday (Nov 28, 2025)
Predicts BF-9 (Nov 19) to BF (Nov 28) using historical average
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def get_black_friday(year):
    """Get Black Friday date (4th Thursday of November + 1 day)"""
    # Find 4th Thursday of November
    nov_first = datetime(year, 11, 1)
    # Find first Thursday
    days_until_thursday = (3 - nov_first.weekday()) % 7
    first_thursday = nov_first + timedelta(days=days_until_thursday)
    # 4th Thursday is Thanksgiving
    thanksgiving = first_thursday + timedelta(weeks=3)
    # Black Friday is the next day
    black_friday = thanksgiving + timedelta(days=1)
    return black_friday

def main():
    # Load full year features
    data_file = Path('dataset/preprocessed/full_year_features.parquet')

    if not data_file.exists():
        print(f"Error: {data_file} not found")
        return

    df = pd.read_parquet(data_file)

    # Convert datetime if needed
    if df['datetime_beginning_ept'].dtype == 'object':
        df['datetime_beginning_ept'] = pd.to_datetime(df['datetime_beginning_ept'])

    # 2025 Black Friday info
    bf_2025 = get_black_friday(2025)

    # Filter historical Black Friday period (BF-9 to BF) from training years
    train_years = [2020, 2021, 2022, 2023, 2024]

    # Get historical BF-9 to BF data
    historical_bf = df[
        (df['year'].isin(train_years)) &
        (df['days_from_bf'] >= -9) &
        (df['days_from_bf'] <= 0)
    ].copy()

    # Calculate baseline: average by load_area, day_of_week, hour
    baseline_avg = historical_bf.groupby(
        ['load_area', 'day_of_week', 'hour']
    )['mw'].mean().reset_index()
    baseline_avg.columns = ['load_area', 'day_of_week', 'hour', 'predicted_mw']

    # Generate 2025 prediction dates (BF-9 to BF)
    prediction_dates = []
    for days_offset in range(-9, 1):  # -9 to 0
        pred_date = bf_2025 + timedelta(days=days_offset)
        for hour in range(24):
            prediction_dates.append({
                'datetime_beginning_ept': pred_date.replace(hour=hour, minute=0, second=0),
                'year': 2025,
                'month': pred_date.month,
                'day': pred_date.day,
                'hour': hour,
                'day_of_week': pred_date.weekday(),
                'days_from_bf': days_offset
            })

    prediction_df = pd.DataFrame(prediction_dates)

    # Get unique load areas
    load_areas = sorted(historical_bf['load_area'].unique())

    # Create predictions for all load areas
    all_predictions = []

    for area in load_areas:
        area_pred = prediction_df.copy()
        area_pred['load_area'] = area

        # Merge with baseline predictions
        area_pred = area_pred.merge(
            baseline_avg,
            on=['load_area', 'day_of_week', 'hour'],
            how='left'
        )

        # Fill any missing predictions with overall area average
        if area_pred['predicted_mw'].isna().any():
            area_avg = historical_bf[historical_bf['load_area'] == area]['mw'].mean()
            area_pred['predicted_mw'].fillna(area_avg, inplace=True)

        all_predictions.append(area_pred)

    # Combine all predictions
    predictions_2025 = pd.concat(all_predictions, ignore_index=True)
    predictions_2025 = predictions_2025.sort_values(['load_area', 'datetime_beginning_ept'])

    # Save predictions
    output_file = Path('dataset/preprocessed/baseline_predictions_2025.csv')
    predictions_2025.to_csv(output_file, index=False)

if __name__ == '__main__':
    main()
