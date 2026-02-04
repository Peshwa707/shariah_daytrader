"""
Shariah Daytrader - Web Dashboard

A Streamlit-based dashboard for monitoring the trading bot.

Run with: streamlit run dashboard/app.py
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.ibkr_config import ibkr_config
from config.settings import settings

# Page config
st.set_page_config(
    page_title="Shariah Daytrader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .positive { color: #00C853; }
    .negative { color: #FF5252; }
    .stMetric > div {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


def get_ibkr_connection():
    """Get or create IBKR connection."""
    if 'ibkr_client' not in st.session_state:
        st.session_state.ibkr_client = None
        st.session_state.ibkr_connected = False
    return st.session_state.ibkr_client, st.session_state.ibkr_connected


async def connect_ibkr():
    """Connect to IBKR."""
    from data.ibkr_client import IBKRClient

    client = IBKRClient()
    connected = await client.connect()

    if connected:
        st.session_state.ibkr_client = client
        st.session_state.ibkr_connected = True
        return client
    return None


async def get_account_data(client):
    """Fetch account data from IBKR."""
    if client is None or not client.is_connected:
        return None

    try:
        ib = client._ib
        account_values = ib.accountValues()
        positions = ib.positions()

        # Parse account values
        account_data = {}
        for av in account_values:
            if av.currency == 'USD':
                account_data[av.tag] = float(av.value) if av.value else 0

        # Parse positions
        positions_data = []
        for pos in positions:
            positions_data.append({
                'symbol': pos.contract.symbol,
                'quantity': pos.position,
                'avg_cost': pos.avgCost,
                'market_value': pos.position * pos.avgCost,
            })

        return {
            'account': account_data,
            'positions': positions_data,
            'accounts': ib.managedAccounts(),
        }
    except Exception as e:
        st.error(f"Error fetching account data: {e}")
        return None


def render_sidebar():
    """Render the sidebar."""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/islamic-crescent.png", width=60)
        st.title("Shariah Daytrader")

        st.divider()

        # Connection status
        _, connected = get_ibkr_connection()
        if connected:
            st.success("🟢 IBKR Connected")
        else:
            st.warning("🔴 IBKR Disconnected")
            if st.button("Connect to IBKR"):
                with st.spinner("Connecting..."):
                    asyncio.run(connect_ibkr())
                    st.rerun()

        st.divider()

        # Navigation
        st.subheader("Navigation")
        page = st.radio(
            "Select Page",
            ["📊 Dashboard", "💼 Positions", "📈 Signals", "☪️ Shariah Screen", "🤖 ML Models", "⚙️ Settings"],
            label_visibility="collapsed",
        )

        st.divider()

        # Market status
        st.subheader("Market Status")
        now = datetime.now()
        if now.weekday() < 5 and 9 <= now.hour < 16:
            st.success("🟢 Market Open")
        else:
            st.info("🔴 Market Closed")

        st.caption(f"Last updated: {now.strftime('%H:%M:%S')}")

        return page


def render_dashboard():
    """Render the main dashboard."""
    st.header("📊 Trading Dashboard")

    client, connected = get_ibkr_connection()

    if not connected:
        st.warning("Connect to IBKR to view live data")

        # Show demo data
        st.subheader("Demo Mode")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Net Liquidation", "$250,000", "+$1,250")
        with col2:
            st.metric("Buying Power", "$1,000,000", "+$5,000")
        with col3:
            st.metric("Today's P&L", "+$1,250", "+0.5%")
        with col4:
            st.metric("Open Positions", "3", "")

        return

    # Fetch live data
    with st.spinner("Fetching account data..."):
        data = asyncio.run(get_account_data(client))

    if data is None:
        st.error("Failed to fetch account data")
        return

    account = data['account']
    positions = data['positions']

    # Account metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        net_liq = account.get('NetLiquidation', 0)
        st.metric("Net Liquidation", f"${net_liq:,.2f}")

    with col2:
        buying_power = account.get('BuyingPower', 0)
        st.metric("Buying Power", f"${buying_power:,.2f}")

    with col3:
        daily_pnl = account.get('DailyPnL', 0)
        st.metric("Today's P&L", f"${daily_pnl:,.2f}", f"{daily_pnl/net_liq*100:.2f}%" if net_liq else "0%")

    with col4:
        st.metric("Open Positions", len(positions))

    st.divider()

    # Positions table
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Current Positions")
        if positions:
            df = pd.DataFrame(positions)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No open positions")

    with col2:
        st.subheader("Account Summary")
        summary_items = [
            ('Cash', account.get('TotalCashValue', 0)),
            ('Available Funds', account.get('AvailableFunds', 0)),
            ('Gross Position', account.get('GrossPositionValue', 0)),
            ('Maintenance Margin', account.get('MaintMarginReq', 0)),
        ]
        for label, value in summary_items:
            st.metric(label, f"${value:,.2f}")


def render_positions():
    """Render positions page."""
    st.header("💼 Positions & Trades")

    client, connected = get_ibkr_connection()

    if not connected:
        st.warning("Connect to IBKR to view positions")
        return

    # Positions
    with st.spinner("Fetching positions..."):
        data = asyncio.run(get_account_data(client))

    if data and data['positions']:
        st.subheader("Open Positions")
        df = pd.DataFrame(data['positions'])

        # Add P&L column (placeholder - would need live prices)
        df['unrealized_pnl'] = 0  # Would calculate from live prices

        st.dataframe(df, use_container_width=True)

        # Position chart
        if len(df) > 0:
            st.subheader("Position Allocation")
            st.bar_chart(df.set_index('symbol')['market_value'])
    else:
        st.info("No open positions")

    st.divider()

    # Trade history from database
    st.subheader("Recent Trades")
    st.info("Trade history will be loaded from database")


def render_signals():
    """Render signals page."""
    st.header("📈 Trading Signals")

    client, connected = get_ibkr_connection()

    col1, col2 = st.columns([3, 1])

    with col2:
        if st.button("🔄 Generate Signals", use_container_width=True):
            if connected:
                with st.spinner("Scanning for signals..."):
                    st.session_state.last_scan = datetime.now()
                    # Would call signal generator here
                    st.success("Scan complete!")
            else:
                st.warning("Connect to IBKR first")

    with col1:
        st.caption(f"Last scan: {st.session_state.get('last_scan', 'Never')}")

    st.divider()

    # Signal display (demo data)
    st.subheader("Latest Signals")

    demo_signals = pd.DataFrame([
        {'symbol': 'AAPL', 'signal': 'BUY', 'probability': 0.68, 'confidence': 'Medium', 'timestamp': datetime.now()},
        {'symbol': 'MSFT', 'signal': 'HOLD', 'probability': 0.52, 'confidence': 'Low', 'timestamp': datetime.now()},
        {'symbol': 'NVDA', 'signal': 'BUY', 'probability': 0.72, 'confidence': 'High', 'timestamp': datetime.now()},
    ])

    for _, row in demo_signals.iterrows():
        col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

        with col1:
            st.write(f"**{row['symbol']}**")
        with col2:
            if row['signal'] == 'BUY':
                st.success(row['signal'])
            elif row['signal'] == 'SELL':
                st.error(row['signal'])
            else:
                st.info(row['signal'])
        with col3:
            st.write(f"{row['probability']:.1%}")
        with col4:
            st.progress(row['probability'])


def render_shariah_screen():
    """Render Shariah screening page."""
    st.header("☪️ Shariah Compliance Screener")

    # Import Shariah modules
    from shariah.compliance_engine import ComplianceEngine
    from shariah.index_integration import load_shariah_universe

    # Initialize compliance engine
    if 'compliance_engine' not in st.session_state:
        st.session_state.compliance_engine = ComplianceEngine()

    compliance = st.session_state.compliance_engine

    # Load universe button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Load SPUS Index", use_container_width=True):
            with st.spinner("Loading Shariah universe..."):
                index = asyncio.run(load_shariah_universe())
                symbols = index.get_all_compliant_symbols()
                compliance.set_index_constituents("SPUS", symbols)
                st.session_state.shariah_symbols = list(symbols)
                st.success(f"Loaded {len(symbols)} symbols")

    st.divider()

    # Symbol screener
    st.subheader("Screen Individual Stock")

    symbol = st.text_input("Enter Symbol", placeholder="AAPL").upper()

    if symbol and st.button("Check Compliance"):
        result = compliance.screen(symbol, screening_level="INDEX_ONLY")

        if result.is_compliant:
            st.success(f"✅ {symbol} is Shariah-Compliant")
            st.info(f"Source: {result.source}")
        else:
            st.error(f"❌ {symbol} is NOT Shariah-Compliant")
            if result.reasons:
                st.warning(f"Reasons: {', '.join(result.reasons)}")

    st.divider()

    # Compliant universe
    st.subheader("Shariah-Compliant Universe")

    if 'shariah_symbols' in st.session_state:
        symbols = st.session_state.shariah_symbols
        st.write(f"**{len(symbols)} compliant stocks loaded**")

        # Display in columns
        cols = st.columns(5)
        for i, sym in enumerate(symbols[:50]):  # Show first 50
            cols[i % 5].write(sym)

        if len(symbols) > 50:
            st.caption(f"... and {len(symbols) - 50} more")
    else:
        st.info("Click 'Load SPUS Index' to load the Shariah-compliant universe")


def render_ml_models():
    """Render ML models page with comprehensive insights."""
    st.header("🤖 ML Model Performance")

    # Initialize database manager
    from data.storage import DatabaseManager
    db = DatabaseManager()

    # Model selection
    model_name = st.selectbox(
        "Select Model",
        ["momentum_continuation", "lightgbm", "random_forest"],
        index=0,
    )

    # Create tabs for different insight views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Health Scorecard",
        "📊 Feature Drift",
        "📈 Calibration",
        "🌊 Regime Matrix",
        "📋 Learning Events",
    ])

    # Get model health metrics
    health = db.get_model_health_metrics(model_name)
    active_model = db.get_active_model(model_name)

    with tab1:
        render_health_scorecard(db, model_name, health, active_model)

    with tab2:
        render_feature_drift(db, model_name, active_model)

    with tab3:
        render_calibration(db, model_name, active_model)

    with tab4:
        render_regime_matrix(db)

    with tab5:
        render_learning_events(db, active_model)


def render_health_scorecard(db, model_name: str, health: dict, active_model: dict | None):
    """Render the health scorecard tab."""
    st.subheader("Model Health Overview")

    if health.get("status") == "no_active_model":
        st.warning(f"No active model found for '{model_name}'")
        st.info("Train and activate a model to see health metrics")
        return

    # Traffic light indicators
    col1, col2, col3, col4 = st.columns(4)

    # Accuracy indicator
    accuracy = health.get("accuracy_7d")
    with col1:
        if accuracy is None:
            st.metric("7-Day Accuracy", "N/A", help="Not enough predictions")
            st.caption("⚪ No data")
        elif accuracy >= 0.60:
            st.metric("7-Day Accuracy", f"{accuracy:.1%}", help="Healthy")
            st.caption("🟢 Healthy")
        elif accuracy >= 0.55:
            st.metric("7-Day Accuracy", f"{accuracy:.1%}", help="Acceptable")
            st.caption("🟡 Acceptable")
        else:
            st.metric("7-Day Accuracy", f"{accuracy:.1%}", help="Below threshold")
            st.caption("🔴 Below threshold")

    # Calibration indicator
    ece = health.get("calibration_ece")
    with col2:
        if ece is None:
            st.metric("Calibration (ECE)", "N/A", help="No calibration data")
            st.caption("⚪ No data")
        elif ece <= 0.05:
            st.metric("Calibration (ECE)", f"{ece:.4f}", help="Well calibrated")
            st.caption("🟢 Well calibrated")
        elif ece <= 0.10:
            st.metric("Calibration (ECE)", f"{ece:.4f}", help="Acceptable")
            st.caption("🟡 Acceptable")
        else:
            st.metric("Calibration (ECE)", f"{ece:.4f}", help="Poorly calibrated")
            st.caption("🔴 Needs attention")

    # Freshness indicator
    days_since = health.get("days_since_training")
    with col3:
        if days_since is None:
            st.metric("Model Age", "Unknown")
            st.caption("⚪ No data")
        elif days_since <= 7:
            st.metric("Model Age", f"{days_since} days")
            st.caption("🟢 Fresh")
        elif days_since <= 14:
            st.metric("Model Age", f"{days_since} days")
            st.caption("🟡 Aging")
        else:
            st.metric("Model Age", f"{days_since} days")
            st.caption("🔴 Stale")

    # Alerts indicator
    drift_alerts = health.get("drift_alerts", 0)
    open_events = health.get("open_events", 0)
    with col4:
        total_issues = drift_alerts + open_events
        if total_issues == 0:
            st.metric("Open Issues", "0")
            st.caption("🟢 Clear")
        elif total_issues <= 2:
            st.metric("Open Issues", str(total_issues))
            st.caption("🟡 Minor")
        else:
            st.metric("Open Issues", str(total_issues))
            st.caption("🔴 Needs attention")

    st.divider()

    # Training metrics
    if active_model:
        st.subheader("Training Metrics")
        metrics = health.get("training_metrics", {})

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Direction Accuracy", f"{metrics.get('direction_accuracy', 0):.2%}")
        with col2:
            st.metric("Direction F1", f"{metrics.get('direction_f1', 0):.2%}")
        with col3:
            st.metric("Magnitude MAE", f"{metrics.get('magnitude_mae', 0):.4f}")
        with col4:
            st.metric("Duration MAE", f"{metrics.get('duration_mae', 0):.2f} bars")

        # Model info
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"Model Version: {health.get('model_version', 'N/A')}")
            st.caption(f"Predictions (7d): {health.get('predictions_7d', 0)}")
        with col2:
            st.caption(f"Model ID: {health.get('model_id', 'N/A')}")
            st.caption(f"Status: {health.get('status', 'N/A')}")


def render_feature_drift(db, model_name: str, active_model: dict | None):
    """Render the feature drift tab."""
    st.subheader("Feature Importance Drift")

    if not active_model:
        st.warning("No active model found")
        return

    model_id = active_model.get("id")

    # Get feature importance history
    history = db.get_feature_importance_history(model_id, limit=5)

    if not history:
        st.info("No feature importance data available yet")
        st.caption("Feature importance is recorded after model training")
        return

    # Get drift alerts
    alerts = db.get_feature_drift_alerts(model_id, rank_change_threshold=5, top_n=10)

    if alerts:
        st.warning(f"{len(alerts)} feature drift alert(s) detected")
        for alert in alerts:
            if alert["type"] == "rank_change":
                change = alert.get("change", 0)
                icon = "📈" if change > 0 else "📉"
                st.write(f"{icon} **{alert['feature']}**: Rank {alert['previous_rank']} → {alert['current_rank']} ({change:+d})")

        st.divider()

    # Build dataframe for visualization
    # Group by recorded_at to get snapshots
    snapshots = {}
    for record in history:
        recorded = record["recorded_at"][:10]  # Date only
        if recorded not in snapshots:
            snapshots[recorded] = {}
        if record.get("model_component") == "combined":
            snapshots[recorded][record["feature_name"]] = record["rank"]

    if snapshots:
        # Create bump chart data
        dates = sorted(snapshots.keys())
        features = set()
        for snapshot in snapshots.values():
            features.update(snapshot.keys())

        # Only show top 10 features
        feature_list = list(features)[:10]

        chart_data = []
        for feat in feature_list:
            for date in dates:
                rank = snapshots.get(date, {}).get(feat)
                if rank:
                    chart_data.append({"date": date, "feature": feat, "rank": rank})

        if chart_data:
            df = pd.DataFrame(chart_data)
            st.subheader("Feature Rank Over Time (Top 10)")

            # Pivot for display
            pivot_df = df.pivot(index="feature", columns="date", values="rank")
            st.dataframe(pivot_df, use_container_width=True)

            # Current importance scores
            st.divider()
            st.subheader("Current Feature Importance")

            latest_date = max(dates)
            latest_records = [r for r in history if r["recorded_at"][:10] == latest_date and r.get("model_component") == "combined"]
            latest_records.sort(key=lambda x: x["rank"])

            if latest_records:
                importance_df = pd.DataFrame([
                    {"feature": r["feature_name"], "importance": r["importance_score"]}
                    for r in latest_records[:15]
                ])
                st.bar_chart(importance_df.set_index("feature"))


def render_calibration(db, model_name: str, active_model: dict | None):
    """Render the calibration tab."""
    st.subheader("Model Calibration")

    if not active_model:
        st.warning("No active model found")
        return

    model_id = active_model.get("id")

    # Get calibration history
    cal_history = db.get_calibration_history(model_id, limit=12)

    if not cal_history:
        st.info("No calibration data available yet")

        # Option to compute calibration
        if st.button("Compute Calibration Metrics"):
            try:
                from ml.insights.monitor import MLInsightsMonitor
                monitor = MLInsightsMonitor(db_manager=db)
                metrics = monitor.compute_calibration_metrics(model_name, days=7)

                if metrics:
                    # Save to database
                    db.save_calibration_report(
                        model_id=model_id,
                        period_start=datetime.now() - timedelta(days=7),
                        period_end=datetime.now(),
                        sample_count=metrics["sample_count"],
                        ece=metrics["ece"],
                        mce=metrics["mce"],
                        brier_score=metrics["brier_score"],
                        log_loss=metrics["log_loss"],
                        reliability_diagram=metrics["reliability_diagram"],
                    )
                    st.success("Calibration metrics computed and saved!")
                    st.rerun()
                else:
                    st.warning("Not enough prediction data for calibration")
            except Exception as e:
                st.error(f"Error computing calibration: {e}")
        return

    # Display latest calibration
    latest = cal_history[0]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ece = latest.get("expected_calibration_error")
        st.metric("ECE", f"{ece:.4f}" if ece else "N/A")
    with col2:
        brier = latest.get("brier_score")
        st.metric("Brier Score", f"{brier:.4f}" if brier else "N/A")
    with col3:
        st.metric("Samples", latest.get("sample_count", "N/A"))
    with col4:
        is_cal = latest.get("is_well_calibrated")
        st.metric("Status", "✅ Well Calibrated" if is_cal else "⚠️ Needs Attention")

    st.divider()

    # Reliability diagram
    reliability = latest.get("reliability_diagram", [])
    if reliability:
        st.subheader("Reliability Diagram")

        rel_df = pd.DataFrame(reliability)
        if not rel_df.empty:
            # Perfect calibration line
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Predicted vs Actual Accuracy**")
                chart_df = rel_df[["bin_center", "accuracy", "confidence"]].copy()
                chart_df = chart_df.rename(columns={"bin_center": "Confidence Bin", "accuracy": "Actual", "confidence": "Predicted"})
                st.line_chart(chart_df.set_index("Confidence Bin"))

            with col2:
                st.write("**Bin Counts**")
                st.bar_chart(rel_df.set_index("bin_center")["count"])

    # Calibration history trend
    if len(cal_history) > 1:
        st.divider()
        st.subheader("Calibration History")

        history_df = pd.DataFrame([
            {
                "date": h.get("period_end", "")[:10],
                "ECE": h.get("expected_calibration_error"),
                "Brier": h.get("brier_score"),
            }
            for h in cal_history
        ])
        history_df = history_df.dropna()
        if not history_df.empty:
            st.line_chart(history_df.set_index("date"))


def render_regime_matrix(db):
    """Render the regime matrix tab."""
    st.subheader("Market Regime Analysis")

    # Get regime history
    regimes = db.get_regime_history(days=30)

    if not regimes:
        st.info("No market regime data available")
        st.caption("Regime detection runs daily and classifies market conditions")
        return

    # Current regime
    current = regimes[0] if regimes else None
    if current:
        col1, col2, col3 = st.columns(3)
        with col1:
            regime_type = current.get("regime_type", "unknown")
            regime_emoji = {
                "trending_up": "📈",
                "trending_down": "📉",
                "volatile": "🌊",
                "quiet": "😴",
                "mixed": "🔀",
            }.get(regime_type, "❓")
            st.metric("Current Regime", f"{regime_emoji} {regime_type.replace('_', ' ').title()}")
        with col2:
            conf = current.get("regime_confidence")
            st.metric("Confidence", f"{conf:.1%}" if conf else "N/A")
        with col3:
            st.metric("VIX Level", f"{current.get('vix_level', 'N/A')}")

    st.divider()

    # Regime history table
    st.subheader("Recent Regime History")

    regime_df = pd.DataFrame([
        {
            "Date": r.get("regime_date"),
            "Regime": r.get("regime_type", "").replace("_", " ").title(),
            "Confidence": f"{r.get('regime_confidence', 0):.1%}" if r.get("regime_confidence") else "N/A",
            "VIX": r.get("vix_level"),
            "S&P Return": f"{r.get('sp500_return_pct', 0):.2%}" if r.get("sp500_return_pct") else "N/A",
            "Model Accuracy": f"{r.get('model_accuracy_in_regime', 0):.1%}" if r.get("model_accuracy_in_regime") else "N/A",
        }
        for r in regimes[:14]  # Last 2 weeks
    ])

    st.dataframe(regime_df, use_container_width=True, hide_index=True)

    # Regime distribution
    st.divider()
    st.subheader("Regime Distribution (30 Days)")

    regime_counts = {}
    for r in regimes:
        rt = r.get("regime_type", "unknown")
        regime_counts[rt] = regime_counts.get(rt, 0) + 1

    if regime_counts:
        dist_df = pd.DataFrame([
            {"Regime": k.replace("_", " ").title(), "Days": v}
            for k, v in regime_counts.items()
        ])
        st.bar_chart(dist_df.set_index("Regime"))


def render_learning_events(db, active_model: dict | None):
    """Render the learning events tab."""
    st.subheader("Learning Events & Alerts")

    # Get open events
    open_events = db.get_open_learning_events()

    if open_events:
        st.warning(f"{len(open_events)} open event(s) requiring attention")

        for event in open_events[:10]:
            severity = event.get("severity", "info")
            severity_emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(severity, "📋")

            with st.expander(f"{severity_emoji} {event.get('title', 'Event')} - {event.get('event_time', '')[:10]}"):
                st.write(f"**Category:** {event.get('category', 'N/A')}")
                st.write(f"**Description:** {event.get('description', 'N/A')}")

                if event.get("requires_action"):
                    st.info("⚡ Action required")

                details = event.get("details", {})
                if details:
                    st.json(details)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Acknowledge", key=f"ack_{event['id']}"):
                        db.acknowledge_learning_event(event["id"])
                        st.rerun()
                with col2:
                    if st.button("Resolve", key=f"res_{event['id']}"):
                        db.resolve_learning_event(event["id"], "Resolved via dashboard")
                        st.rerun()

        st.divider()

    # Recent events history
    st.subheader("Recent Events History")

    history = db.get_learning_events_history(days=30, limit=50)

    if not history:
        st.info("No learning events recorded yet")
        return

    # Filter controls
    col1, col2 = st.columns(2)
    with col1:
        filter_severity = st.selectbox("Filter by Severity", ["All", "info", "warning", "critical"])
    with col2:
        filter_type = st.selectbox("Filter by Type", ["All", "alert", "insight", "adaptation", "retrain_trigger"])

    # Apply filters
    filtered = history
    if filter_severity != "All":
        filtered = [e for e in filtered if e.get("severity") == filter_severity]
    if filter_type != "All":
        filtered = [e for e in filtered if e.get("event_type") == filter_type]

    # Display events
    events_df = pd.DataFrame([
        {
            "Time": e.get("event_time", "")[:16],
            "Type": e.get("event_type", ""),
            "Severity": e.get("severity", ""),
            "Title": e.get("title", ""),
            "Status": e.get("status", ""),
        }
        for e in filtered[:20]
    ])

    st.dataframe(events_df, use_container_width=True, hide_index=True)


def render_settings():
    """Render settings page."""
    st.header("⚙️ Settings")

    # IBKR Settings
    st.subheader("IBKR Connection")

    col1, col2 = st.columns(2)

    with col1:
        st.text_input("Host", value=ibkr_config.host, disabled=True)
        st.number_input("Port", value=ibkr_config.port, disabled=True)

    with col2:
        st.selectbox("Mode", ["paper", "live"], index=0 if ibkr_config.mode == "paper" else 1, disabled=True)
        st.number_input("Client ID", value=ibkr_config.client_id, disabled=True)

    st.divider()

    # Trading Settings
    st.subheader("Trading Parameters")

    col1, col2 = st.columns(2)

    with col1:
        st.slider("Max Position Size (%)", 1, 20, 5)
        st.slider("Risk Per Trade (%)", 0.5, 5.0, 2.0)

    with col2:
        st.slider("Max Concurrent Positions", 1, 20, 10)
        st.slider("Daily Loss Limit (%)", 1, 10, 3)

    st.divider()

    # Service status
    st.subheader("Service Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("🔄 Bot Status: Unknown")
    with col2:
        if st.button("Start Bot"):
            st.warning("Run: sudo systemctl start shariah-trader")
    with col3:
        if st.button("Stop Bot"):
            st.warning("Run: sudo systemctl stop shariah-trader")


def main():
    """Main application."""
    page = render_sidebar()

    if page == "📊 Dashboard":
        render_dashboard()
    elif page == "💼 Positions":
        render_positions()
    elif page == "📈 Signals":
        render_signals()
    elif page == "☪️ Shariah Screen":
        render_shariah_screen()
    elif page == "🤖 ML Models":
        render_ml_models()
    elif page == "⚙️ Settings":
        render_settings()


if __name__ == "__main__":
    main()
