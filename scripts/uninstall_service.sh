#!/bin/bash
#
# Uninstall Shariah Daytrader systemd service
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${RED}Uninstalling Shariah Daytrader Service${NC}"
echo "========================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run with sudo: sudo $0${NC}"
    exit 1
fi

# Stop the service if running
if systemctl is-active --quiet shariah-trader; then
    echo "Stopping service..."
    systemctl stop shariah-trader
fi

# Disable the service
if systemctl is-enabled --quiet shariah-trader 2>/dev/null; then
    echo "Disabling service..."
    systemctl disable shariah-trader
fi

# Remove service file
if [ -f /etc/systemd/system/shariah-trader.service ]; then
    rm /etc/systemd/system/shariah-trader.service
    echo -e "${GREEN}✓${NC} Service file removed"
fi

# Reload systemd
systemctl daemon-reload
echo -e "${GREEN}✓${NC} Systemd reloaded"

echo ""
echo -e "${GREEN}Uninstallation complete!${NC}"
