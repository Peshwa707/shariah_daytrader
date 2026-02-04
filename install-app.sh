#!/bin/bash
#
# Install Shariah Daytrader apps to Applications folder
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_NAME="Shariah Daytrader.app"
STOP_APP_NAME="Shariah Daytrader Stop.app"
APP_SOURCE="$SCRIPT_DIR/$APP_NAME"
STOP_APP_SOURCE="$SCRIPT_DIR/$STOP_APP_NAME"
APP_DEST="/Applications/$APP_NAME"
STOP_APP_DEST="/Applications/$STOP_APP_NAME"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Installing Shariah Daytrader...${NC}"
echo

# Check if app exists
if [ ! -d "$APP_SOURCE" ]; then
    echo "Error: $APP_NAME not found in project directory"
    exit 1
fi

# Remove old installations if exist
if [ -d "$APP_DEST" ]; then
    echo "Removing previous Start app..."
    rm -rf "$APP_DEST"
fi
if [ -d "$STOP_APP_DEST" ]; then
    echo "Removing previous Stop app..."
    rm -rf "$STOP_APP_DEST"
fi

# Copy apps to Applications
echo "Copying apps to /Applications..."
cp -R "$APP_SOURCE" "$APP_DEST"
cp -R "$STOP_APP_SOURCE" "$STOP_APP_DEST"

# Clear icon cache to show new icon
echo "Refreshing icon cache..."
touch "$APP_DEST"
touch "$STOP_APP_DEST"
killall Finder 2>/dev/null || true

echo
echo -e "${GREEN}Installation complete!${NC}"
echo
echo "Installed apps:"
echo "  - Shariah Daytrader      (Start bot + dashboard)"
echo "  - Shariah Daytrader Stop (Stop bot)"
echo
echo "You can now:"
echo "  1. Find apps in /Applications"
echo "  2. Drag them to your Dock for quick access"
echo "  3. Double-click 'Shariah Daytrader' to start"
echo
echo -e "${YELLOW}Tip:${NC} Right-click app icons and select 'Options > Keep in Dock'"
echo

# Offer to open Applications folder
read -p "Open Applications folder? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    open /Applications
fi
