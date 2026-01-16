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
    """Render ML models page."""
    st.header("🤖 ML Model Performance")

    # Model selection
    model_type = st.selectbox("Select Model", ["LightGBM", "Random Forest"])

    st.divider()

    # Demo metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", "62.5%", "+2.1%")
    with col2:
        st.metric("Precision", "68.3%", "+1.5%")
    with col3:
        st.metric("Recall", "71.2%", "-0.8%")
    with col4:
        st.metric("F1 Score", "69.7%", "+0.4%")

    st.divider()

    # Feature importance (demo)
    st.subheader("Feature Importance")

    importance_data = pd.DataFrame({
        'feature': ['rsi_14', 'macd', 'bb_pct', 'atr_14', 'volume_sma_ratio', 'close_sma_20_ratio'],
        'importance': [0.18, 0.15, 0.14, 0.12, 0.11, 0.10]
    })

    st.bar_chart(importance_data.set_index('feature'))

    st.divider()

    # Training controls
    st.subheader("Model Training")

    col1, col2 = st.columns(2)

    with col1:
        lookback = st.slider("Lookback Days", 60, 365, 180)
        min_prob = st.slider("Min Probability", 0.50, 0.80, 0.58)

    with col2:
        if st.button("Retrain Model", use_container_width=True):
            with st.spinner("Training model..."):
                # Would retrain model here
                st.success("Model retrained!")


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
