#!/usr/bin/env python3
"""
Make predictions using trained models:
- XGBoost Regressor: MW predictions (hourly)
- XGBoost Classifier (peak_hour): Peak hour predictions (daily)
- XGBoost Classifier (peak_days): Peak day predictions (daily)
"""

import pandas as pd
import pickle
from pathlib import Path
import numpy as np

def load_models(models_dir='../models'):
    """Load all trained models"""
    models_path = Path(models_dir)

    print("="*80)
    print("LOADING MODELS")
    print("="*80)

    # 1. Load XGBoost Regressor models (MW predictions)
    with open(models_path / 'xgbreg_model_metadata.pkl', 'rb') as f:
        xgbreg_metadata = pickle.load(f)

    print(f"XGBoost Regressor - Model type: {xgbreg_metadata['model_type']}")
    print(f"XGBoost Regressor - Horizon: {xgbreg_metadata['horizon_hours']} hours")
    print(f"XGBoost Regressor - Features: {len(xgbreg_metadata['features'])}")
    print(f"XGBoost Regressor - Load areas: {len(xgbreg_metadata['load_areas'])}")
    print(f"XGBoost Regressor - Business day masking: {xgbreg_metadata.get('business_day_masking', False)}")

    xgbreg_models = {}
    for area in xgbreg_metadata['load_areas']:
        model_path = models_path / f"xgbreg_{area}.pkl"
        with open(model_path, 'rb') as f:
            xgbreg_models[area] = pickle.load(f)
        print(f"  ✓ Loaded XGBoost Regressor {area}")

    # 2. Load XGBoost Classifier (peak_hour) models
    with open(models_path / 'xgb_model_metadata.pkl', 'rb') as f:
        xgb_hour_metadata = pickle.load(f)

    print(f"\nXGBoost Classifier (peak_hour) - Model type: {xgb_hour_metadata['model_type']}")
    print(f"XGBoost Classifier (peak_hour) - Num classes: {xgb_hour_metadata['num_classes']}")
    print(f"XGBoost Classifier (peak_hour) - Features: {len(xgb_hour_metadata['features'])}")
    print(f"XGBoost Classifier (peak_hour) - Load areas: {len(xgb_hour_metadata['load_areas'])}")

    xgb_hour_models = {}
    for area in xgb_hour_metadata['load_areas']:
        model_path = models_path / f"xgb_{area}.pkl"
        with open(model_path, 'rb') as f:
            xgb_hour_models[area] = pickle.load(f)
        print(f"  ✓ Loaded XGBoost Classifier (peak_hour) {area}")

    # 3. Load XGBoost Classifier (peak_days) models
    try:
        with open(models_path / 'xgb_peak_day_metadata.pkl', 'rb') as f:
            xgb_days_metadata = pickle.load(f)

        print(f"\nXGBoost Classifier (peak_days) - Model type: {xgb_days_metadata['model_type']}")
        print(f"XGBoost Classifier (peak_days) - Features: {len(xgb_days_metadata['features'])}")
        print(f"XGBoost Classifier (peak_days) - Load areas: {len(xgb_days_metadata['load_areas'])}")

        xgb_days_models = {}
        for area in xgb_days_metadata['load_areas']:
            model_path = models_path / f"xgb_peak_day_{area}.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    xgb_days_models[area] = pickle.load(f)
                print(f"  ✓ Loaded XGBoost Classifier (peak_days) {area}")
    except FileNotFoundError:
        print(f"\n⚠ XGBoost Classifier (peak_days) models not found, skipping peak day predictions")
        xgb_days_models = {}
        xgb_days_metadata = None

    print("="*80)
    return xgbreg_models, xgbreg_metadata, xgb_hour_models, xgb_hour_metadata, xgb_days_models, xgb_days_metadata

def make_predictions(test_csv='../dataset/test/test.csv', models_dir='../models'):
    """
    Make predictions using all models

    Args:
        test_csv: Path to test data CSV
        models_dir: Directory containing trained models

    Returns:
        DataFrame with predictions
    """
    # Load models
    xgbreg_models, xgbreg_metadata, xgb_hour_models, xgb_hour_metadata, xgb_days_models, xgb_days_metadata = load_models(models_dir)

    xgbreg_features = xgbreg_metadata['features']
    xgb_hour_features = xgb_hour_metadata['features']
    xgb_days_features = xgb_days_metadata['features'] if xgb_days_metadata else []

    # Load test data
    print("\n" + "="*80)
    print("LOADING TEST DATA")
    print("="*80)
    test_df = pd.read_csv(test_csv)
    test_df['datetime_beginning_ept'] = pd.to_datetime(test_df['datetime_beginning_ept'])
    print(f"Test data: {len(test_df)} rows")
    print(f"Load areas: {test_df['load_area'].nunique()}")
    print(f"Date: {test_df['datetime_beginning_ept'].dt.date.unique()[0]}")

    # Make predictions
    print("\n" + "="*80)
    print("MAKING PREDICTIONS")
    print("="*80)

    predictions = []

    for area in sorted(test_df['load_area'].unique()):
        area_data = test_df[test_df['load_area'] == area].sort_values('hour')

        # Check if XGBoost Regressor model exists
        if area not in xgbreg_models:
            print(f"  ⚠ Skipping {area} (no XGBoost Regressor model found)")
            continue

        # === 1. XGBoost Regressor: Predict MW values ===
        X_xgbreg = area_data[xgbreg_features]
        if X_xgbreg.isnull().any().any():
            X_xgbreg = X_xgbreg.fillna(-999)  # Use -999 for missing values

        xgbreg_model = xgbreg_models[area]
        y_pred = xgbreg_model.predict(X_xgbreg)
        y_pred_int = np.round(y_pred).astype(int)

        # === 2. XGBoost Classifier (peak_hour): Predict peak hour ===
        if area in xgb_hour_models:
            # Use first row's daily features
            X_xgb_hour = area_data[xgb_hour_features].iloc[[0]]

            # Fill missing values with -999 (same as training)
            for feat in xgb_hour_features:
                if X_xgb_hour[feat].isnull().any():
                    X_xgb_hour[feat] = X_xgb_hour[feat].fillna(-999)

            xgb_hour_model = xgb_hour_models[area]
            peak_hour = int(xgb_hour_model.predict(X_xgb_hour)[0])
        else:
            # Fallback: use XGBoost Regressor prediction
            peak_hour = int(area_data.iloc[np.argmax(y_pred)]['hour'])
            print(f"  ⚠ {area}: Using XGBoost Regressor for peak hour (Classifier model not found)")

        # === 3. XGBoost Classifier (peak_days): Predict is_peak_day ===
        is_peak_day = 0  # Default
        peak_day_proba = 0.0

        if area in xgb_days_models and xgb_days_features:
            # Check if all required features exist
            missing_features = [f for f in xgb_days_features if f not in area_data.columns]

            if not missing_features:
                # Use first row's daily features
                X_xgb_days = area_data[xgb_days_features].iloc[[0]]

                # Fill missing values with -999 (same as training)
                for feat in xgb_days_features:
                    if X_xgb_days[feat].isnull().any():
                        X_xgb_days[feat] = X_xgb_days[feat].fillna(-999)

                xgb_days_model = xgb_days_models[area]

                # Get prediction probability
                peak_day_proba = xgb_days_model.predict_proba(X_xgb_days)[0][1]  # P(class=1)
                is_peak_day = int(xgb_days_model.predict(X_xgb_days)[0])
            else:
                print(f"  ⚠ {area}: Missing features for peak_days: {missing_features[:3]}...")

        # Store predictions
        for i, (idx, row) in enumerate(area_data.iterrows()):
            predictions.append({
                'load_area': area,
                'hour': int(row['hour']),
                'predicted_mw': y_pred_int[i],
                'peak_hour': peak_hour,
                'is_peak_day': is_peak_day,
                'peak_day_proba': peak_day_proba
            })

        # Print summary
        peak_day_status = f"Peak day prob = {peak_day_proba:.3f}" if xgb_days_models else "N/A"
        print(f"  ✓ {area:10s}: Peak hour = {peak_hour:02d}, Max MW = {y_pred_int.max():,}, {peak_day_status}")

    # Create predictions DataFrame
    pred_df = pd.DataFrame(predictions)

    print("="*80)
    print(f"✓ Predictions completed: {len(pred_df)} rows")
    print("="*80)

    return pred_df

def main():
    # Make predictions
    pred_df = make_predictions()

    # Save predictions
    output_path = Path('../results/hybrid_predictions.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_path, index=False)

    print(f"\n✓ Predictions saved to: {output_path}")
    print(f"  Total predictions: {len(pred_df)}")
    print(f"  MW predictions: XGBoost Regressor")
    print(f"  Peak hour predictions: XGBoost Classifier (peak_hour)")
    print(f"  Peak day predictions: XGBoost Classifier (peak_days)")

    # Display sample
    print("\nSample predictions:")
    print(pred_df.head(30))

if __name__ == '__main__':
    main()
