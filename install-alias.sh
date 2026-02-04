#!/bin/bash
#
# Install 'shariah-trader' alias to your shell
#
# Usage: ./install-alias.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$SCRIPT_DIR/launcher.sh"
ALIAS_NAME="shariah-trader"

# Detect shell
if [ -n "$ZSH_VERSION" ]; then
    SHELL_NAME="zsh"
    RC_FILE="$HOME/.zshrc"
elif [ -n "$BASH_VERSION" ]; then
    SHELL_NAME="bash"
    RC_FILE="$HOME/.bashrc"
    # On macOS, bash uses .bash_profile
    if [[ "$OSTYPE" == "darwin"* ]] && [ -f "$HOME/.bash_profile" ]; then
        RC_FILE="$HOME/.bash_profile"
    fi
else
    echo "Unsupported shell. Add this alias manually:"
    echo "  alias $ALIAS_NAME='$LAUNCHER'"
    exit 1
fi

# Check if alias already exists
if grep -q "alias $ALIAS_NAME=" "$RC_FILE" 2>/dev/null; then
    echo "Alias '$ALIAS_NAME' already exists in $RC_FILE"
    echo "Current definition:"
    grep "alias $ALIAS_NAME=" "$RC_FILE"
    exit 0
fi

# Add alias
echo "" >> "$RC_FILE"
echo "# Shariah Daytrader launcher" >> "$RC_FILE"
echo "alias $ALIAS_NAME='$LAUNCHER'" >> "$RC_FILE"

echo "Alias '$ALIAS_NAME' added to $RC_FILE"
echo ""
echo "To use now, run:"
echo "  source $RC_FILE"
echo ""
echo "Then you can use:"
echo "  $ALIAS_NAME start    # Start full system"
echo "  $ALIAS_NAME stop     # Stop all components"
echo "  $ALIAS_NAME status   # Check status"
echo "  $ALIAS_NAME help     # Show all commands"
