#!/bin/bash
#
# Shariah Daytrader Startup Script
# Checks prerequisites before starting the trading bot
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
LOG_FILE="$PROJECT_DIR/logs/trader.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

# Create logs directory
mkdir -p "$PROJECT_DIR/logs"

log "Starting Shariah Daytrader..."
log "Project directory: $PROJECT_DIR"

# Check if virtual environment exists
if [ ! -f "$VENV_PYTHON" ]; then
    error "Virtual environment not found at $VENV_PYTHON"
    error "Run: cd $PROJECT_DIR && python -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi
log "Virtual environment found"

# Check if IB Gateway is running (port 4002 for paper trading)
if nc -z 127.0.0.1 4002 2>/dev/null; then
    log "IB Gateway detected on port 4002 (paper trading)"
elif nc -z 127.0.0.1 7497 2>/dev/null; then
    log "TWS detected on port 7497 (paper trading)"
elif nc -z 127.0.0.1 4001 2>/dev/null; then
    warn "IB Gateway detected on port 4001 (LIVE trading) - be careful!"
elif nc -z 127.0.0.1 7496 2>/dev/null; then
    warn "TWS detected on port 7496 (LIVE trading) - be careful!"
else
    error "IB Gateway/TWS not detected!"
    error "Please start IB Gateway or TWS before running the trader"
    error "Expected ports: 4002 (Gateway Paper), 7497 (TWS Paper)"
    exit 1
fi

# Check market hours (US Eastern Time)
# Note: This is approximate - doesn't account for holidays
check_market_hours() {
    # Get current time in ET
    ET_HOUR=$(TZ="America/New_York" date +%H)
    ET_MIN=$(TZ="America/New_York" date +%M)
    DAY_OF_WEEK=$(TZ="America/New_York" date +%u)  # 1=Monday, 7=Sunday

    # Weekend check
    if [ "$DAY_OF_WEEK" -ge 6 ]; then
        return 1
    fi

    # Convert to minutes since midnight
    CURRENT_MIN=$((ET_HOUR * 60 + ET_MIN))
    MARKET_OPEN=$((9 * 60 + 30))   # 9:30 AM
    MARKET_CLOSE=$((16 * 60))       # 4:00 PM

    if [ "$CURRENT_MIN" -ge "$MARKET_OPEN" ] && [ "$CURRENT_MIN" -lt "$MARKET_CLOSE" ]; then
        return 0
    fi
    return 1
}

if check_market_hours; then
    log "Market is OPEN"
else
    warn "Market is currently CLOSED"
    warn "Bot will wait for market hours to begin trading"
fi

# Start the trading bot
log "Launching trading bot..."
cd "$PROJECT_DIR"

exec "$VENV_PYTHON" main.py --trade 2>&1 | tee -a "$LOG_FILE"
