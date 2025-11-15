#!/usr/bin/env python3
"""
Display 2025 Black Friday predictions by load area
"""

import pandas as pd
from pathlib import Path

def main():
    # Load 2025 prediction results
    predictions_file = Path('dataset/preprocessed/baseline_predictions_2025.csv')

    if not predictions_file.exists():
        print(f"Error: {predictions_file} not found")
        return

    df = pd.read_csv(predictions_file)
    df['datetime_beginning_ept'] = pd.to_datetime(df['datetime_beginning_ept'])

    # Display predictions by load area
    for area in sorted(df['load_area'].unique()):
        area_data = df[df['load_area'] == area].sort_values('datetime_beginning_ept')

        print(f"\n{'=' * 80}")
        print(f"LOAD AREA: {area}")
        print(f"{'=' * 80}")
        print(f"{'Date':<12} {'Hour':<6} {'Predicted MW':<15}")
        print("-" * 80)

        for _, row in area_data.iterrows():
            date_str = row['datetime_beginning_ept'].strftime('%Y-%m-%d')
            print(f"{date_str:<12} {row['hour']:<6} {row['predicted_mw']:>12,.2f}")

        print()

if __name__ == '__main__':
    main()
