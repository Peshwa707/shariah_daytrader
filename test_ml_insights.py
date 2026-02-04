#!/usr/bin/env python3
"""
Test ML Insights System - Generate synthetic data and verify integration.

This script:
1. Generates synthetic price data
2. Trains the momentum model
3. Registers model in registry
4. Creates sample predictions and outcomes
5. Verifies the insights system works
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.storage import DatabaseManager
from ml.models.momentum_continuation_model import MomentumContinuationModel
from ml.features.technical import TechnicalFeatures
from ml.features.price_action import PriceActionFeatures
from ml.insights.monitor import MLInsightsMonitor


def generate_synthetic_ohlcv(
    days: int = 200,
    start_price: float = 100.0,
    volatility: float = 0.02,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data with momentum patterns."""
    np.random.seed(42)  # Reproducible

    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    # Generate price series with trend + noise
    trend = np.cumsum(np.random.randn(days) * volatility)
    prices = start_price * np.exp(trend)

    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        daily_range = price * volatility * np.random.uniform(0.5, 2.0)
        high = price + daily_range / 2
        low = price - daily_range / 2
        open_price = np.random.uniform(low, high)
        close = np.random.uniform(low, high)
        volume = int(np.random.uniform(1e6, 5e6))

        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
        })

    return pd.DataFrame(data).set_index('timestamp')


def create_training_labels(df: pd.DataFrame, lookahead: int = 5) -> tuple:
    """Create training labels for momentum model."""
    # Direction: continuation (0), reversal (1), neutral (2)
    future_return = df['close'].shift(-lookahead) / df['close'] - 1

    direction = pd.Series(2, index=df.index)  # Default neutral
    direction[future_return > 0.01] = 0  # Continuation (up)
    direction[future_return < -0.01] = 1  # Reversal (down)

    # Magnitude: expected return %
    magnitude = future_return * 100

    # Duration: bars until reversal (simplified)
    duration = pd.Series(lookahead, index=df.index)

    return direction, magnitude, duration


def main():
    print("=" * 60)
    print("ML INSIGHTS SYSTEM TEST")
    print("=" * 60)
    print()

    # Initialize database
    print("1. Initializing database...")
    db = DatabaseManager()

    # Generate synthetic data for multiple symbols
    print("2. Generating synthetic data...")
    symbols = ['TEST1', 'TEST2', 'TEST3']
    all_features = []
    all_directions = []
    all_magnitudes = []
    all_durations = []

    tech_features = TechnicalFeatures()
    price_features = PriceActionFeatures()

    for symbol in symbols:
        print(f"   - {symbol}...")
        ohlcv = generate_synthetic_ohlcv(days=250, volatility=0.02)

        # Calculate features
        tech_df = tech_features.calculate_all(ohlcv)
        price_df = price_features.calculate_all(ohlcv)

        # Combine features
        features = pd.concat([ohlcv, tech_df, price_df], axis=1)
        features = features.loc[:, ~features.columns.duplicated()]

        # Create labels
        direction, magnitude, duration = create_training_labels(ohlcv)

        # Drop NaN rows
        valid_idx = features.dropna().index
        features = features.loc[valid_idx]
        direction = direction.loc[valid_idx]
        magnitude = magnitude.loc[valid_idx]
        duration = duration.loc[valid_idx]

        # Remove last few rows (no labels)
        features = features.iloc[:-10]
        direction = direction.iloc[:-10]
        magnitude = magnitude.iloc[:-10]
        duration = duration.iloc[:-10]

        all_features.append(features)
        all_directions.append(direction)
        all_magnitudes.append(magnitude)
        all_durations.append(duration)

    # Combine all data
    X = pd.concat(all_features, ignore_index=True)
    y_direction = pd.concat(all_directions, ignore_index=True)
    y_magnitude = pd.concat(all_magnitudes, ignore_index=True)
    y_duration = pd.concat(all_durations, ignore_index=True)

    # Remove non-feature columns
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'date', 'timestamp', 'symbol']
    feature_cols = [c for c in X.columns if c not in exclude_cols and not X[c].isna().all()]
    X = X[feature_cols].fillna(0)

    print(f"   Total samples: {len(X)}")
    print(f"   Features: {len(feature_cols)}")

    # Train model
    print("\n3. Training momentum model...")
    model = MomentumContinuationModel()
    metrics = model.train(X, y_direction, y_magnitude, y_duration)
    print(f"   Metrics: {metrics}")

    # Save and register model
    print("\n4. Saving and registering model...")
    model_path = Path("models/test_momentum_model.pkl")
    model_path.parent.mkdir(exist_ok=True)
    model_id = model.save(model_path, register=True, activate=True)
    print(f"   Model ID: {model_id}")

    # Verify model in registry
    active = db.get_active_model("momentum_continuation")
    if active:
        print(f"   Active model version: {active.get('model_version')}")
        print(f"   Status: {active.get('status')}")

    # Create some sample predictions
    print("\n5. Creating sample predictions...")
    for i in range(20):
        # Ensure predicted_value and predicted_class are consistent
        is_bullish = np.random.random() > 0.5
        predicted_value = 1.0 if is_bullish else -1.0
        predicted_class = "continuation" if is_bullish else "reversal"

        pred_id = db.record_prediction_outcome(
            model_id=model_id,
            symbol=np.random.choice(symbols),
            prediction_type="direction",
            predicted_value=predicted_value,
            predicted_probability=np.random.uniform(0.55, 0.85),
            predicted_class=predicted_class,
            prediction_time=datetime.now() - timedelta(days=np.random.randint(1, 7)),
        )

        # Update some with outcomes
        if i < 15:
            is_correct = np.random.random() > 0.4  # ~60% accuracy
            # Actual outcome matches prediction if correct, opposite if wrong
            if is_correct:
                actual_value = predicted_value
                actual_class = predicted_class
            else:
                actual_value = -predicted_value
                actual_class = "reversal" if predicted_class == "continuation" else "continuation"

            db.update_prediction_outcome(
                prediction_id=pred_id,
                actual_value=actual_value,
                actual_class=actual_class,
            )
    print("   Created 20 predictions, 15 with outcomes")

    # Save feature importances
    print("\n6. Saving feature importances...")
    if model.metrics and model.metrics.feature_importances:
        # Already saved during training, but let's verify
        history = db.get_feature_importance_history(model_id, limit=1)
        print(f"   Feature records: {len(history)}")

    # Create sample calibration report
    print("\n7. Creating calibration report...")
    cal_id = db.save_calibration_report(
        model_id=model_id,
        period_start=datetime.now() - timedelta(days=7),
        period_end=datetime.now(),
        sample_count=15,
        ece=0.08,
        mce=0.12,
        brier_score=0.21,
        log_loss=0.55,
        reliability_diagram=[
            {"bin_center": 0.55, "accuracy": 0.52, "confidence": 0.55, "count": 3},
            {"bin_center": 0.65, "accuracy": 0.60, "confidence": 0.65, "count": 5},
            {"bin_center": 0.75, "accuracy": 0.72, "confidence": 0.75, "count": 4},
            {"bin_center": 0.85, "accuracy": 0.80, "confidence": 0.85, "count": 3},
        ],
    )
    print(f"   Calibration report ID: {cal_id}")

    # Create sample market regimes
    print("\n8. Creating market regime records...")
    from datetime import date
    regimes = ["trending_up", "volatile", "quiet", "trending_down", "mixed"]
    for i in range(10):
        regime_date = date.today() - timedelta(days=i)
        db.save_market_regime(
            regime_date=regime_date,
            regime_type=np.random.choice(regimes),
            regime_confidence=np.random.uniform(0.6, 0.95),
            period_start=datetime.now() - timedelta(days=i+1),
            period_end=datetime.now() - timedelta(days=i),
            vix_level=np.random.uniform(12, 25),
            sp500_return_pct=np.random.uniform(-0.02, 0.02),
        )
    print("   Created 10 regime records")

    # Create sample learning events
    print("\n9. Creating learning events...")
    db.log_learning_event(
        event_type="alert",
        severity="warning",
        category="performance",
        title="Test Performance Alert",
        description="This is a test alert for the ML insights system",
        source="test_script",
        model_id=model_id,
        requires_action=False,
    )
    db.log_learning_event(
        event_type="insight",
        severity="info",
        category="drift",
        title="Feature Importance Shift",
        description="RSI_14 moved from rank 3 to rank 1",
        source="test_script",
        model_id=model_id,
    )
    print("   Created 2 learning events")

    # Run monitoring checks
    print("\n10. Running monitoring checks...")
    monitor = MLInsightsMonitor(db_manager=db)
    results = monitor.run_all_checks("momentum_continuation", log_alerts=False)
    for r in results:
        status = "✓" if r.passed else "✗"
        print(f"   {status} {r.check_name}: {r.message[:50]}...")

    # Get health summary
    print("\n11. Getting health summary...")
    summary = monitor.get_health_summary("momentum_continuation")
    print(f"   Overall status: {summary['overall_status']}")
    print(f"   Health score: {summary['health_score']:.0f}%")

    # Print ML statistics
    print("\n12. Final ML Statistics...")
    stats = db.get_ml_statistics()
    for k, v in stats.items():
        print(f"   {k}: {v}")

    print()
    print("=" * 60)
    print("TEST COMPLETE!")
    print("=" * 60)
    print()
    print("Now try running:")
    print("  python insights.py status")
    print("  python insights.py alerts")
    print("  python insights.py drift")
    print("  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
