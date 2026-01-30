#!/usr/bin/env python3
"""
Shariah-Compliant AI Daytrading Bot - Main Entry Point.

This is an educational project demonstrating how to build a
Shariah-compliant trading system with ML-based signal generation.

Usage:
    python main.py              # Run demo mode
    python main.py --trade      # Run autonomous paper trading
    python main.py --scan       # Run single scan and exit
    python main.py --status     # Show system status

IMPORTANT: This is for paper trading and educational purposes only.
Always consult with qualified Shariah scholars for compliance guidance.
"""

import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings
from config.ibkr_config import ibkr_config
from data.ibkr_client import IBKRClient
from shariah.compliance_engine import ComplianceEngine
from shariah.index_integration import load_shariah_universe
from trading.execution_engine import ExecutionEngine, ExecutionConfig, Signal
from trading.signal_generator import SignalGenerator, SignalGeneratorConfig


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    print("\n\nShutdown requested. Closing positions and exiting...")
    shutdown_requested = True


def print_banner():
    """Print application banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║        SHARIAH-COMPLIANT AI DAYTRADING BOT                ║
    ║                                                           ║
    ║   An Educational Trading System                           ║
    ║   • AAOIFI-compliant screening                           ║
    ║   • ML-based signal generation                           ║
    ║   • Interactive Brokers integration                       ║
    ║                                                           ║
    ║   ⚠️  FOR EDUCATIONAL & PAPER TRADING ONLY               ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_configuration():
    """Print current configuration."""
    print("\nConfiguration:")
    print(f"  Environment: {settings.environment}")
    print(f"  IBKR Mode: {ibkr_config.mode}")
    print(f"  IBKR Host: {ibkr_config.host}:{ibkr_config.port}")
    print(f"  Client ID: {ibkr_config.client_id}")
    print(f"  Database: {settings.db_path}")


def is_market_open() -> bool:
    """Check if US market is currently open (Eastern Time).

    Note: This doesn't account for holidays.
    """
    eastern = ZoneInfo("America/New_York")
    now = datetime.now(eastern)
    # Weekend check
    if now.weekday() >= 5:
        return False
    current_time = now.time()
    market_open = time(9, 30)
    market_close = time(16, 0)
    return market_open <= current_time <= market_close


async def demo_shariah_screening():
    """Demonstrate Shariah screening capabilities."""
    print("\n" + "=" * 60)
    print("SHARIAH SCREENING DEMONSTRATION")
    print("=" * 60)

    # Load Shariah index universe
    print("\n1. Loading Shariah-compliant universe...")
    index = await load_shariah_universe()
    symbols = index.get_all_compliant_symbols()
    print(f"   Loaded {len(symbols)} pre-vetted symbols from SPUS ETF")

    # Initialize compliance engine
    compliance = ComplianceEngine()
    compliance.set_index_constituents("SPUS", symbols)

    # Screen some example stocks
    test_symbols = ["AAPL", "MSFT", "JPM", "NVDA", "XOM"]

    print("\n2. Screening individual stocks...")
    for symbol in test_symbols:
        result = compliance.screen(symbol, screening_level="INDEX_ONLY")
        status = "✓ Compliant" if result.is_compliant else "✗ Non-compliant"
        print(f"   {symbol}: {status} ({result.source})")

    # Print statistics
    stats = compliance.get_statistics()
    print(f"\n3. Statistics:")
    print(f"   Total screened: {stats['total']}")
    print(f"   Compliant: {stats['compliant']}")
    print(f"   Index loaded: {stats['indices_loaded']}")


async def demo_paper_trading():
    """Demonstrate paper trading execution."""
    print("\n" + "=" * 60)
    print("PAPER TRADING DEMONSTRATION")
    print("=" * 60)

    # Initialize execution engine (paper trading mode)
    config = ExecutionConfig(paper_trading=True)
    engine = ExecutionEngine(config=config)

    # Start engine
    await engine.start()

    # Create a test signal
    signal = Signal(
        symbol="AAPL",
        signal_type="buy",
        probability=0.65,
        confidence="medium",
        model_name="demo",
    )

    print(f"\n1. Processing signal: {signal.signal_type} {signal.symbol}")
    result = await engine.process_signal(signal)

    print(f"\n2. Execution result:")
    print(f"   Actions taken: {result['actions_taken']}")
    if result.get('errors'):
        print(f"   Errors: {result['errors']}")
    if result.get('order_id'):
        print(f"   Order ID: {result['order_id']}")

    # Get status
    status = engine.get_status()
    print(f"\n3. Engine status:")
    print(f"   Running: {status['running']}")
    print(f"   Paper trading: {status['paper_trading']}")
    print(f"   Positions: {status['positions']}")

    # Stop engine
    await engine.stop()


async def run_demo_mode():
    """Run demonstration mode."""
    print_banner()
    print_configuration()

    await demo_shariah_screening()
    await demo_paper_trading()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Review notebooks/ for educational content")
    print("  2. Run with --trade flag for autonomous paper trading")
    print("  3. Run with --scan flag to generate signals without trading")
    print("\nAlways consult qualified scholars for Shariah guidance.")


async def run_single_scan(ibkr_client: IBKRClient):
    """Run a single scan and display signals without trading."""
    print("\n" + "=" * 60)
    print("SIGNAL SCAN MODE")
    print("=" * 60)

    # Initialize signal generator
    config = SignalGeneratorConfig(
        max_symbols_to_scan=10,
        min_probability=0.55,
    )
    generator = SignalGenerator(ibkr_client, config)
    await generator.initialize()

    print(f"\nScanning {len(generator._symbols)} Shariah-compliant stocks...")
    print("-" * 60)

    signals = await generator.generate_signals()

    if signals:
        print(f"\n✓ Generated {len(signals)} signal(s):\n")
        for i, sig in enumerate(signals, 1):
            print(f"  {i}. {sig.signal_type.upper()} {sig.symbol}")
            print(f"     Probability: {sig.probability:.1%}")
            print(f"     Confidence: {sig.confidence}")
            print(f"     Model: {sig.model_name}")
            if sig.features:
                print(f"     Features: RSI={sig.features.get('rsi', 'N/A'):.1f}" if sig.features.get('rsi') else "")
            print()
    else:
        print("\n  No strong signals generated at this time.")
        print("  This could be due to:")
        print("  - Market conditions not meeting thresholds")
        print("  - Insufficient historical data")
        print("  - Model confidence below minimum")

    status = generator.get_status()
    print("\n" + "-" * 60)
    print(f"Generator Status:")
    print(f"  Symbols loaded: {status['symbols_loaded']}")
    print(f"  Model type: {status['model_type']}")
    print(f"  Model trained: {status['model_trained']}")


async def run_autonomous_trading():
    """Run autonomous paper trading mode."""
    global shutdown_requested

    print_banner()
    print_configuration()

    print("\n" + "=" * 60)
    print("AUTONOMOUS PAPER TRADING MODE")
    print("=" * 60)

    # Safety check
    if ibkr_config.mode != "paper":
        print("\n⚠️  ERROR: Autonomous trading only allowed in paper mode!")
        print("   Set IBKR_MODE=paper in .env file")
        return

    print("\n⚠️  PAPER TRADING MODE - No real money at risk")
    print("   Press Ctrl+C to stop gracefully\n")

    # Connect to IBKR
    print("1. Connecting to IBKR...")
    ibkr_client = IBKRClient()
    connected = await ibkr_client.connect()

    if not connected:
        print("   ✗ Failed to connect to IBKR")
        print("   Make sure IB Gateway or TWS is running")
        return

    print(f"   ✓ Connected to IBKR (Server v{ibkr_client._ib.client.serverVersion()})")

    # Get account info
    accounts = ibkr_client._ib.managedAccounts()
    print(f"   ✓ Account: {accounts[0] if accounts else 'Unknown'}")

    # Initialize components
    print("\n2. Initializing trading components...")

    # Signal generator
    sig_config = SignalGeneratorConfig(
        max_symbols_to_scan=15,
        min_probability=0.58,
        signal_cooldown_minutes=30,
        max_signals_per_scan=2,
    )
    signal_generator = SignalGenerator(ibkr_client, sig_config)
    await signal_generator.initialize()
    print(f"   ✓ Signal generator ready ({len(signal_generator._symbols)} symbols)")

    # Load Shariah universe for compliance engine
    index = await load_shariah_universe()

    # Compliance engine
    compliance_engine = ComplianceEngine()
    compliance_engine.set_index_constituents("SPUS", index.get_all_compliant_symbols())
    print("   ✓ Shariah compliance engine ready")

    # Execution engine
    exec_config = ExecutionConfig(
        paper_trading=True,
        regular_hours_only=True,
        min_probability=0.55,
        auto_exit_at_close=True,
        exit_minutes_before_close=15,
    )
    execution_engine = ExecutionEngine(
        ibkr_client=ibkr_client,
        compliance_engine=compliance_engine,
        config=exec_config,
    )
    await execution_engine.start()
    print("   ✓ Execution engine started")

    # Trading loop
    print("\n3. Starting trading loop...")
    print("=" * 60)

    scan_interval = 60  # seconds between scans
    iteration = 0

    while not shutdown_requested:
        iteration += 1
        now = datetime.now()

        print(f"\n[{now.strftime('%H:%M:%S')}] Scan #{iteration}")

        # Check market hours
        if not is_market_open():
            print("   Market closed - waiting...")
            await asyncio.sleep(60)
            continue

        try:
            # Generate signals
            signals = await signal_generator.generate_signals()

            if signals:
                print(f"   Generated {len(signals)} signal(s)")

                for sig in signals:
                    print(f"   → Processing: {sig.signal_type.upper()} {sig.symbol} (prob: {sig.probability:.1%})")

                    # Process through execution engine
                    result = await execution_engine.process_signal(sig)

                    if result.get('errors'):
                        print(f"     ✗ Errors: {result['errors']}")
                    elif result.get('order_id'):
                        print(f"     ✓ Order submitted: {result['order_id']}")
                    else:
                        print(f"     Actions: {result.get('actions_taken', [])}")
            else:
                print("   No signals generated")

            # Show current status
            status = execution_engine.get_status()
            if status['positions'] > 0:
                print(f"   Positions: {status['positions']}")

        except Exception as e:
            logger.exception(f"Error in trading loop: {e}")
            print(f"   ✗ Error: {e}")

        # Wait for next scan
        print(f"   Next scan in {scan_interval}s...")
        for _ in range(scan_interval):
            if shutdown_requested:
                break
            await asyncio.sleep(1)

    # Graceful shutdown
    print("\n" + "=" * 60)
    print("SHUTTING DOWN")
    print("=" * 60)

    print("\n1. Closing all positions...")
    results = await execution_engine.close_all_positions()
    if results:
        for r in results:
            print(f"   Closed {r['symbol']}: {r['submitted']}")
    else:
        print("   No positions to close")

    print("\n2. Stopping execution engine...")
    await execution_engine.stop()
    print("   ✓ Engine stopped")

    print("\n3. Disconnecting from IBKR...")
    await ibkr_client.disconnect()
    print("   ✓ Disconnected")

    print("\n✓ Shutdown complete")


async def show_status():
    """Show system status."""
    print_banner()
    print_configuration()

    print("\n" + "=" * 60)
    print("SYSTEM STATUS")
    print("=" * 60)

    # Check IBKR connection
    print("\n1. IBKR Connection...")
    ibkr_client = IBKRClient()
    try:
        connected = await ibkr_client.connect()
        if connected:
            print(f"   ✓ Connected to {ibkr_config.host}:{ibkr_config.port}")
            print(f"   ✓ Server version: {ibkr_client._ib.client.serverVersion()}")

            accounts = ibkr_client._ib.managedAccounts()
            print(f"   ✓ Account: {accounts[0] if accounts else 'Unknown'}")

            # Get account value
            account_values = ibkr_client._ib.accountValues()
            for av in account_values:
                if av.tag == 'NetLiquidation' and av.currency == 'USD':
                    print(f"   ✓ Net Liquidation: ${float(av.value):,.2f}")
                    break

            await ibkr_client.disconnect()
        else:
            print("   ✗ Not connected")
    except Exception as e:
        print(f"   ✗ Connection error: {e}")

    # Check Shariah universe
    print("\n2. Shariah Universe...")
    try:
        index = await load_shariah_universe()
        symbols = index.get_all_compliant_symbols()
        print(f"   ✓ Loaded {len(symbols)} Shariah-compliant stocks")
    except Exception as e:
        print(f"   ✗ Error loading universe: {e}")

    # Market status
    print("\n3. Market Status...")
    if is_market_open():
        print("   ✓ Market is OPEN")
    else:
        print("   ✗ Market is CLOSED")
        now = datetime.now()
        if now.weekday() >= 5:
            print("     (Weekend)")
        else:
            print("     (Outside trading hours: 9:30 AM - 4:00 PM)")

    print("\n" + "=" * 60)


async def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Shariah-Compliant AI Daytrading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py              Run demo mode
  python main.py --trade      Start autonomous paper trading
  python main.py --scan       Run single scan for signals
  python main.py --status     Show system status

IMPORTANT: This is for educational and paper trading purposes only.
        """
    )

    parser.add_argument(
        '--trade',
        action='store_true',
        help='Run autonomous paper trading mode'
    )

    parser.add_argument(
        '--scan',
        action='store_true',
        help='Run single scan and display signals'
    )

    parser.add_argument(
        '--status',
        action='store_true',
        help='Show system status'
    )

    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demo mode (default)'
    )

    args = parser.parse_args()

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if args.trade:
        await run_autonomous_trading()
    elif args.scan:
        # Connect and run single scan
        print_banner()
        print_configuration()

        print("\nConnecting to IBKR...")
        ibkr_client = IBKRClient()
        connected = await ibkr_client.connect()

        if connected:
            print("✓ Connected")
            await run_single_scan(ibkr_client)
            await ibkr_client.disconnect()
        else:
            print("✗ Failed to connect to IBKR")
    elif args.status:
        await show_status()
    else:
        await run_demo_mode()


if __name__ == "__main__":
    asyncio.run(main())
