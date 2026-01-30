#!/usr/bin/env python3
"""
Simple ML test that doesn't require pandas-ta.
Tests the core ML models with synthetic data.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, '.')

print("=" * 60)
print("SIMPLE ML MODEL TESTS")
print("=" * 60)

# Create synthetic training data
np.random.seed(42)
n_samples = 500
n_features = 10

print("\n1. Creating synthetic data...")
X = pd.DataFrame(
    np.random.randn(n_samples, n_features),
    columns=[f"feature_{i}" for i in range(n_features)]
)
y = pd.Series((np.random.randn(n_samples) > 0).astype(int))
print(f"   Created {n_samples} samples with {n_features} features")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

# Test RandomForest model
print("\n2. Testing RandomForestSignalModel...")
try:
    from ml.models.random_forest import RandomForestSignalModel

    rf_model = RandomForestSignalModel()
    print("   - Model initialized")

    # Train
    rf_model.train(X_train, y_train)
    print("   - Model trained")

    # Predict probabilities (returns numpy array)
    probas = rf_model.predict_proba(X_test)
    print(f"   - Probabilities shape: {probas.shape}")

    # Predict (returns list of PredictionResult objects)
    predictions = rf_model.predict(X_test, symbol="TEST")
    print(f"   - Predictions count: {len(predictions)}")
    print(f"   - Sample prediction: {predictions[0].signal} (prob: {predictions[0].probability:.2%})")

    # Check signal distribution
    signals = [p.signal for p in predictions]
    buy_pct = signals.count('buy') / len(signals)
    sell_pct = signals.count('sell') / len(signals)
    hold_pct = signals.count('hold') / len(signals)
    print(f"   ✓ RandomForest: Buy {buy_pct:.1%}, Sell {sell_pct:.1%}, Hold {hold_pct:.1%}")

except Exception as e:
    print(f"   ✗ RandomForest Error: {e}")

# Test LightGBM model
print("\n3. Testing LightGBMSignalModel...")
try:
    from ml.models.lightgbm_model import LightGBMSignalModel

    lgb_model = LightGBMSignalModel()
    print("   - Model initialized")

    # Train
    lgb_model.train(X_train, y_train)
    print("   - Model trained")

    # Predict probabilities (returns numpy array)
    probas = lgb_model.predict_proba(X_test)
    print(f"   - Probabilities shape: {probas.shape}")

    # Predict (returns list of dicts for LightGBM)
    predictions = lgb_model.predict(X_test, symbol="TEST")
    print(f"   - Predictions count: {len(predictions)}")
    # LightGBM returns dicts, not PredictionResult objects
    pred = predictions[0]
    print(f"   - Sample prediction: {pred['signal']} (prob: {pred['probability']:.2%})")

    # Check signal distribution
    signals = [p['signal'] for p in predictions]
    buy_pct = signals.count('buy') / len(signals)
    sell_pct = signals.count('sell') / len(signals)
    hold_pct = signals.count('hold') / len(signals)
    print(f"   ✓ LightGBM: Buy {buy_pct:.1%}, Sell {sell_pct:.1%}, Hold {hold_pct:.1%}")

except Exception as e:
    print(f"   ✗ LightGBM Error: {e}")

# Test price action features (doesn't need pandas-ta)
print("\n4. Testing PriceActionFeatures...")
try:
    from ml.features.price_action import PriceActionFeatures

    # Create synthetic OHLCV data
    ohlcv = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'high': 0,  # Will calculate
        'low': 0,   # Will calculate
        'close': 0, # Will calculate
        'volume': np.random.randint(1000000, 5000000, 100)
    })
    # Make high/low/close realistic
    ohlcv['close'] = ohlcv['open'] + np.random.randn(100) * 0.5
    ohlcv['high'] = ohlcv[['open', 'close']].max(axis=1) + abs(np.random.randn(100) * 0.3)
    ohlcv['low'] = ohlcv[['open', 'close']].min(axis=1) - abs(np.random.randn(100) * 0.3)

    pa_features = PriceActionFeatures()
    result = pa_features.calculate_all(ohlcv)

    print(f"   - Input shape: {ohlcv.shape}")
    print(f"   - Output shape: {result.shape}")
    print(f"   - New columns: {len(result.columns) - len(ohlcv.columns)}")
    print(f"   ✓ PriceActionFeatures working")

except Exception as e:
    print(f"   ✗ PriceActionFeatures Error: {e}")

# Test backtest engine basic functionality
print("\n5. Testing BacktestEngine import...")
try:
    from ml.backtesting.backtest_engine import BacktestEngine
    print("   ✓ BacktestEngine imported successfully")
except ImportError as e:
    print(f"   - BacktestEngine import failed (may need vectorbt): {e}")
except Exception as e:
    print(f"   ✗ BacktestEngine Error: {e}")

print("\n" + "=" * 60)
print("ML TESTS COMPLETE")
print("=" * 60)
