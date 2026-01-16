#!/usr/bin/env python3
"""
Connect to IBKR Paper Trading

Prerequisites:
1. TWS or IB Gateway must be running
2. API connections must be enabled:
   - TWS: File -> Global Configuration -> API -> Settings
   - Check "Enable ActiveX and Socket Clients"
   - Add 127.0.0.1 to "Trusted IPs"
   - Socket port: 7497 (paper) or 7496 (live)
3. For paper trading, log into your paper trading account
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.ibkr_config import ibkr_config


def print_header(text: str):
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}\n")


async def connect_and_test():
    """Connect to IBKR and run tests."""

    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║          IBKR PAPER TRADING CONNECTION                    ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    print("Configuration:")
    print(f"  Host: {ibkr_config.host}")
    print(f"  Port: {ibkr_config.port}")
    print(f"  Mode: {ibkr_config.mode}")
    print(f"  Client ID: {ibkr_config.client_id}")
    print(f"  Market Data Type: {ibkr_config.market_data_type} (3=Delayed)")

    print_header("1. CONNECTING TO IBKR")

    try:
        from ib_async import IB, Stock, util

        ib = IB()

        print(f"Connecting to {ibkr_config.host}:{ibkr_config.port}...")

        await ib.connectAsync(
            host=ibkr_config.host,
            port=ibkr_config.port,
            clientId=ibkr_config.client_id,
            timeout=30,
        )

        print("✓ Connected successfully!")
        print(f"  Server Version: {ib.client.serverVersion()}")

        # Set market data type (3 = delayed for different subscription level)
        ib.reqMarketDataType(ibkr_config.market_data_type)

    except ConnectionRefusedError:
        print("✗ Connection refused!")
        print("\nMake sure:")
        print("  1. TWS or IB Gateway is running")
        print("  2. You're logged into your PAPER trading account")
        print("  3. API connections are enabled in TWS settings")
        print(f"  4. Socket port is set to {ibkr_config.port}")
        return None
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return None

    # Test account info
    print_header("2. ACCOUNT INFORMATION")

    try:
        accounts = ib.managedAccounts()
        print(f"Managed Accounts: {accounts}")

        # Get account summary
        account_values = ib.accountValues()

        important_tags = [
            'NetLiquidation', 'TotalCashValue', 'BuyingPower',
            'GrossPositionValue', 'MaintMarginReq', 'AvailableFunds'
        ]

        print("\nAccount Summary:")
        for av in account_values:
            if av.tag in important_tags and av.currency == 'USD':
                print(f"  {av.tag}: ${float(av.value):,.2f}")

    except Exception as e:
        print(f"✗ Error getting account info: {e}")

    # Test positions
    print_header("3. CURRENT POSITIONS")

    try:
        positions = ib.positions()
        if positions:
            for pos in positions:
                print(f"  {pos.contract.symbol}: {pos.position} shares @ ${pos.avgCost:.2f}")
        else:
            print("  No open positions")
    except Exception as e:
        print(f"✗ Error getting positions: {e}")

    # Test market data
    print_header("4. MARKET DATA TEST")

    test_symbols = ["AAPL", "MSFT", "NVDA"]

    for symbol in test_symbols:
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            await ib.qualifyContractsAsync(contract)

            # Request market data
            ticker = ib.reqMktData(contract, '', False, False)

            # Wait for data
            await asyncio.sleep(2)

            # Cancel market data
            ib.cancelMktData(contract)

            if ticker.last and ticker.last == ticker.last:  # NaN check
                print(f"  {symbol}: ${ticker.last:.2f} (bid: ${ticker.bid:.2f}, ask: ${ticker.ask:.2f})")
            elif ticker.close and ticker.close == ticker.close:
                print(f"  {symbol}: Close=${ticker.close:.2f} (delayed data)")
            else:
                print(f"  {symbol}: Waiting for data... (market may be closed)")

        except Exception as e:
            print(f"  {symbol}: Error - {e}")

    # Test historical data
    print_header("5. HISTORICAL DATA TEST")

    try:
        contract = Stock('AAPL', 'SMART', 'USD')
        await ib.qualifyContractsAsync(contract)

        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime='',
            durationStr='5 D',
            barSizeSetting='1 day',
            whatToShow='ADJUSTED_LAST',
            useRTH=True,
        )

        if bars:
            print(f"  Received {len(bars)} daily bars for AAPL")
            print(f"  Latest: {bars[-1].date} - O:{bars[-1].open:.2f} H:{bars[-1].high:.2f} L:{bars[-1].low:.2f} C:{bars[-1].close:.2f}")
        else:
            print("  No historical data received (market may be closed)")

    except Exception as e:
        print(f"  Error: {e}")

    # Test Shariah-compliant stocks
    print_header("6. SHARIAH-COMPLIANT STOCKS")

    from shariah.index_integration import load_shariah_universe

    index = await load_shariah_universe()
    symbols = list(index.get_all_compliant_symbols())[:5]

    print(f"Testing {len(symbols)} Shariah-compliant stocks:")

    for symbol in symbols:
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            await ib.qualifyContractsAsync(contract)

            ticker = ib.reqMktData(contract, '', False, False)
            await asyncio.sleep(1)
            ib.cancelMktData(contract)

            price = ticker.last if (ticker.last and ticker.last == ticker.last) else ticker.close
            if price and price == price:
                print(f"  ✓ {symbol}: ${price:.2f}")
            else:
                print(f"  ✓ {symbol}: (waiting for price)")

        except Exception as e:
            print(f"  ✗ {symbol}: {e}")

    # Summary
    print_header("CONNECTION SUMMARY")

    print(f"""
  Status: CONNECTED
  Account: {accounts[0] if accounts else 'Unknown'}
  Mode: {'PAPER TRADING' if 'DU' in str(accounts) or ibkr_config.mode == 'paper' else 'LIVE'}

  The connection is working! You can now:
  1. Run the trading bot with: python main.py
  2. Execute paper trades
  3. Fetch real-time market data

  Note: If prices show as delayed or unavailable, you may need
  a market data subscription (~$4.50/month for US equities).
    """)

    # Disconnect
    ib.disconnect()
    print("Disconnected from IBKR")

    return ib


async def main():
    """Main entry point."""
    await connect_and_test()


if __name__ == "__main__":
    asyncio.run(main())
