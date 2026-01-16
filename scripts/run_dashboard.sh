#!/bin/bash
#
# Launch the Shariah Daytrader Dashboard
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "Starting Shariah Daytrader Dashboard..."
echo "Open http://localhost:8501 in your browser"
echo ""

.venv/bin/streamlit run dashboard/app.py --server.headless true
