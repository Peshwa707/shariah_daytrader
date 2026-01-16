#!/bin/bash
#
# Install Shariah Daytrader as a systemd service
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_FILE="$SCRIPT_DIR/shariah-trader.service"
SYSTEMD_DIR="/etc/systemd/system"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Installing Shariah Daytrader Service${NC}"
echo "========================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run with sudo: sudo $0${NC}"
    exit 1
fi

# Check if service file exists
if [ ! -f "$SERVICE_FILE" ]; then
    echo -e "${RED}Service file not found: $SERVICE_FILE${NC}"
    exit 1
fi

# Get the actual user (not root)
ACTUAL_USER="${SUDO_USER:-$USER}"
ACTUAL_HOME=$(getent passwd "$ACTUAL_USER" | cut -d: -f6)

echo "Installing for user: $ACTUAL_USER"
echo "Home directory: $ACTUAL_HOME"

# Update service file with correct paths
TEMP_SERVICE="/tmp/shariah-trader.service"
sed -e "s|/home/sam|$ACTUAL_HOME|g" \
    -e "s|User=sam|User=$ACTUAL_USER|g" \
    -e "s|Group=sam|Group=$ACTUAL_USER|g" \
    "$SERVICE_FILE" > "$TEMP_SERVICE"

# Copy service file
cp "$TEMP_SERVICE" "$SYSTEMD_DIR/shariah-trader.service"
echo -e "${GREEN}✓${NC} Service file installed"

# Make startup script executable
chmod +x "$SCRIPT_DIR/start_trader.sh"
echo -e "${GREEN}✓${NC} Startup script made executable"

# Reload systemd
systemctl daemon-reload
echo -e "${GREEN}✓${NC} Systemd reloaded"

echo ""
echo -e "${GREEN}Installation complete!${NC}"
echo ""
echo "Commands:"
echo "  sudo systemctl start shariah-trader    # Start the bot"
echo "  sudo systemctl stop shariah-trader     # Stop the bot"
echo "  sudo systemctl status shariah-trader   # Check status"
echo "  sudo systemctl enable shariah-trader   # Auto-start on boot"
echo "  journalctl -u shariah-trader -f        # View logs"
echo ""
echo -e "${YELLOW}IMPORTANT:${NC}"
echo "  1. Make sure IB Gateway is running before starting"
echo "  2. The bot runs in PAPER TRADING mode by default"
echo "  3. Logs are saved to: $ACTUAL_HOME/Claude/shariah_daytrader/logs/"
echo ""
