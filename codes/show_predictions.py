#!/usr/bin/env python3
"""
Display predictions in required format
Output: "YYYY-MM-DD", L1_00, L1_01, ..., L1_23, L2_00, ..., L_29_23, PH_1, ..., PH_29, PD_1, ..., PD_29
"""

import pandas as pd
from pathlib import Path
import sys

def format_predictions(predictions_csv='../results/hybrid_predictions.csv'):
    """
    Format predictions in required output format

    Output format:
    "YYYY-MM-DD", L1_00, L1_01, ..., L1_23, L2_00, ..., L_29_23, PH_1, ..., PH_29, PD_1, ..., PD_29

    where:
    - YYYY-MM-DD is the prediction date
    - L{zone}_{hour} is the predicted load for zone {zone} at hour {hour} (integer MW)
    - PH_{zone} is the peak hour for zone {zone} (00 to 23)
    - PD_{zone} is the indicator of the peak day for zone {zone} (0 or 1)
    """
    # Load predictions
    pred_df = pd.read_csv(predictions_csv)

    # Get sorted load areas
    load_areas = sorted(pred_df['load_area'].unique())

    # Prepare output components
    output_parts = []

    # 1. Date (use first row's date, assuming test.csv was created with target date)
    # Since we don't have date in predictions, read from test.csv
    test_csv = Path('../dataset/test/test.csv')
    if test_csv.exists():
        test_df = pd.read_csv(test_csv)
        test_df['datetime_beginning_ept'] = pd.to_datetime(test_df['datetime_beginning_ept'])
        pred_date = test_df['datetime_beginning_ept'].dt.date.iloc[0]
    else:
        # Fallback: use current date
        from datetime import datetime
        pred_date = datetime.now().date()

    output_parts.append(f'"{pred_date}"')

    # 2. Load predictions: L{zone}_{hour} for each zone (24 hours each)
    for area in load_areas:
        area_data = pred_df[pred_df['load_area'] == area].sort_values('hour')

        for hour in range(24):
            hour_data = area_data[area_data['hour'] == hour]
            if len(hour_data) > 0:
                load_mw = int(hour_data['predicted_mw'].iloc[0])
            else:
                load_mw = 0  # Default if missing

            output_parts.append(str(load_mw))

    # 3. Peak hours: PH_{zone} for each zone
    for area in load_areas:
        area_data = pred_df[pred_df['load_area'] == area]
        if len(area_data) > 0:
            peak_hour = int(area_data['peak_hour'].iloc[0])
        else:
            peak_hour = 0  # Default if missing

        output_parts.append(str(peak_hour))
    
    # 4. Peak Days: PD_{zone} for each zone
    for area in load_areas:
        area_data = pred_df[pred_df['load_area'] == area]
        if len(area_data) > 0:
            peak_day = int(area_data['is_peak_day'].iloc[0])
        else:
            peak_day = 0  # Default if missing

        output_parts.append(str(peak_day))

    # Join with commas and print (no other output)
    output_line = ', '.join(output_parts)
    print(output_line)

def main():
    predictions_csv = Path('../results/hybrid_predictions.csv')

    if not predictions_csv.exists():
        print(f"Error: {predictions_csv} not found", file=sys.stderr)
        print("Please run prepare_test_data.py and make_predictions.py first", file=sys.stderr)
        sys.exit(1)

    format_predictions(str(predictions_csv))

if __name__ == '__main__':
    main()
