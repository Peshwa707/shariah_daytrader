#!/bin/bash
#
# Shariah Daytrader Launcher
# One-click launcher for the full trading system
#
# Usage:
#   ./launcher.sh [command]
#
# Commands:
#   start     Start trading bot + dashboard (default)
#   stop      Stop all components gracefully
#   restart   Restart all components
#   status    Show running status
#   logs      Tail all logs
#   bot       Start trading bot only
#   dashboard Start dashboard only
#   demo      Run demo mode (foreground)
#   scan      Run single scan (foreground)
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
VENV_DIR="$PROJECT_DIR/venv"
LOG_DIR="$PROJECT_DIR/logs"
PID_DIR="$PROJECT_DIR/.pids"

# PID files
BOT_PID_FILE="$PID_DIR/bot.pid"
DASHBOARD_PID_FILE="$PID_DIR/dashboard.pid"
CAFFEINATE_PID_FILE="$PID_DIR/caffeinate.pid"

# Log files
BOT_LOG="$LOG_DIR/bot.log"
DASHBOARD_LOG="$LOG_DIR/dashboard.log"

# Dashboard settings
DASHBOARD_PORT="${SHARIAH_DASHBOARD_PORT:-8501}"
DASHBOARD_URL="http://localhost:$DASHBOARD_PORT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
print_banner() {
    echo -e "${BLUE}"
    cat << 'EOF'
    ╔═══════════════════════════════════════════════════════════╗
    ║        SHARIAH-COMPLIANT AI DAYTRADING BOT                ║
    ║                     LAUNCHER v1.0                         ║
    ╚═══════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# Logging helpers
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Ensure directories exist
setup_dirs() {
    mkdir -p "$LOG_DIR"
    mkdir -p "$PID_DIR"
}

# Check if virtualenv exists
check_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        log_error "Virtual environment not found at $VENV_DIR"
        log_info "Create it with: python3 -m venv $VENV_DIR && $VENV_DIR/bin/pip install -r requirements.txt"
        exit 1
    fi
}

# Get Python interpreter
get_python() {
    echo "$VENV_DIR/bin/python"
}

# Check if a process is running
is_running() {
    local pid_file="$1"
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# Get PID from file
get_pid() {
    local pid_file="$1"
    if [ -f "$pid_file" ]; then
        cat "$pid_file"
    fi
}

# Start the trading bot
start_bot() {
    if is_running "$BOT_PID_FILE"; then
        log_warn "Trading bot already running (PID: $(get_pid "$BOT_PID_FILE"))"
        return 0
    fi

    log_info "Starting trading bot..."

    cd "$PROJECT_DIR"
    nohup "$(get_python)" main.py --trade > "$BOT_LOG" 2>&1 &
    local pid=$!
    echo "$pid" > "$BOT_PID_FILE"

    sleep 2
    if is_running "$BOT_PID_FILE"; then
        log_info "Trading bot started (PID: $pid)"
        log_info "Logs: $BOT_LOG"
    else
        log_error "Trading bot failed to start. Check logs: $BOT_LOG"
        rm -f "$BOT_PID_FILE"
        return 1
    fi
}

# Start the dashboard
start_dashboard() {
    if is_running "$DASHBOARD_PID_FILE"; then
        log_warn "Dashboard already running (PID: $(get_pid "$DASHBOARD_PID_FILE"))"
        return 0
    fi

    log_info "Starting Streamlit dashboard on port $DASHBOARD_PORT..."

    cd "$PROJECT_DIR"
    nohup "$VENV_DIR/bin/streamlit" run dashboard/app.py \
        --server.port "$DASHBOARD_PORT" \
        --server.headless true \
        --browser.gatherUsageStats false \
        > "$DASHBOARD_LOG" 2>&1 &
    local pid=$!
    echo "$pid" > "$DASHBOARD_PID_FILE"

    sleep 3
    if is_running "$DASHBOARD_PID_FILE"; then
        log_info "Dashboard started (PID: $pid)"
        log_info "URL: $DASHBOARD_URL"
    else
        log_error "Dashboard failed to start. Check logs: $DASHBOARD_LOG"
        rm -f "$DASHBOARD_PID_FILE"
        return 1
    fi
}

# Stop a component
stop_component() {
    local name="$1"
    local pid_file="$2"

    if ! is_running "$pid_file"; then
        log_info "$name is not running"
        rm -f "$pid_file"
        return 0
    fi

    local pid=$(get_pid "$pid_file")
    log_info "Stopping $name (PID: $pid)..."

    # Send SIGTERM for graceful shutdown
    kill -TERM "$pid" 2>/dev/null || true

    # Wait for process to stop (up to 10 seconds)
    local count=0
    while [ $count -lt 10 ]; do
        if ! ps -p "$pid" > /dev/null 2>&1; then
            break
        fi
        sleep 1
        ((count++))
    done

    # Force kill if still running
    if ps -p "$pid" > /dev/null 2>&1; then
        log_warn "$name didn't stop gracefully, forcing..."
        kill -9 "$pid" 2>/dev/null || true
    fi

    rm -f "$pid_file"
    log_info "$name stopped"
}

# Start caffeinate to prevent Mac sleep
start_caffeinate() {
    if is_running "$CAFFEINATE_PID_FILE"; then
        log_info "Caffeinate already running (PID: $(get_pid "$CAFFEINATE_PID_FILE"))"
        return 0
    fi

    log_info "Starting caffeinate to prevent system sleep..."

    # -i: Prevent idle sleep
    # -m: Prevent disk sleep
    # -s: Prevent system sleep (AC power only - falls back gracefully on battery)
    caffeinate -i -m -s &
    local pid=$!
    echo "$pid" > "$CAFFEINATE_PID_FILE"

    sleep 1
    if is_running "$CAFFEINATE_PID_FILE"; then
        log_info "Caffeinate started (PID: $pid) - Mac will stay awake"
    else
        log_warn "Caffeinate may not have started (check power state)"
        rm -f "$CAFFEINATE_PID_FILE"
    fi
}

# Stop caffeinate
stop_caffeinate() {
    if ! is_running "$CAFFEINATE_PID_FILE"; then
        rm -f "$CAFFEINATE_PID_FILE"
        return 0
    fi

    local pid=$(get_pid "$CAFFEINATE_PID_FILE")
    log_info "Stopping caffeinate (PID: $pid)..."
    kill "$pid" 2>/dev/null || true
    rm -f "$CAFFEINATE_PID_FILE"
    log_info "Caffeinate stopped - Mac can sleep normally"
}

# Open dashboard in browser
open_browser() {
    sleep 2
    if command -v open &> /dev/null; then
        # macOS
        open "$DASHBOARD_URL"
    elif command -v xdg-open &> /dev/null; then
        # Linux
        xdg-open "$DASHBOARD_URL"
    else
        log_info "Open in browser: $DASHBOARD_URL"
    fi
}

# Command: start
cmd_start() {
    print_banner
    setup_dirs
    check_venv

    log_info "Starting full system..."
    echo

    # Prevent Mac from sleeping while trading
    start_caffeinate

    start_bot
    start_dashboard

    echo
    log_info "System started successfully!"
    echo
    echo -e "  ${GREEN}Dashboard:${NC} $DASHBOARD_URL"
    echo -e "  ${GREEN}Bot logs:${NC}  tail -f $BOT_LOG"
    echo -e "  ${GREEN}Stop:${NC}      $0 stop"
    echo

    # Open browser
    open_browser &
}

# Command: stop
cmd_stop() {
    print_banner

    log_info "Stopping all components..."
    echo

    stop_component "Trading bot" "$BOT_PID_FILE"
    stop_component "Dashboard" "$DASHBOARD_PID_FILE"
    stop_caffeinate

    echo
    log_info "All components stopped"
}

# Command: restart
cmd_restart() {
    cmd_stop
    echo
    sleep 2
    cmd_start
}

# Command: status
cmd_status() {
    print_banner

    echo "Component Status:"
    echo "─────────────────────────────────────────"

    # Trading bot status
    if is_running "$BOT_PID_FILE"; then
        local pid=$(get_pid "$BOT_PID_FILE")
        echo -e "  Trading Bot:  ${GREEN}RUNNING${NC} (PID: $pid)"
    else
        echo -e "  Trading Bot:  ${RED}STOPPED${NC}"
    fi

    # Dashboard status
    if is_running "$DASHBOARD_PID_FILE"; then
        local pid=$(get_pid "$DASHBOARD_PID_FILE")
        echo -e "  Dashboard:    ${GREEN}RUNNING${NC} (PID: $pid)"
        echo -e "                URL: $DASHBOARD_URL"
    else
        echo -e "  Dashboard:    ${RED}STOPPED${NC}"
    fi

    # Caffeinate status (wake lock)
    if is_running "$CAFFEINATE_PID_FILE"; then
        local pid=$(get_pid "$CAFFEINATE_PID_FILE")
        echo -e "  Wake Lock:    ${GREEN}ACTIVE${NC} (PID: $pid)"
    else
        echo -e "  Wake Lock:    ${YELLOW}INACTIVE${NC} (Mac may sleep)"
    fi

    echo "─────────────────────────────────────────"

    # Check IBKR connection via status command
    echo
    echo "System Check:"
    "$(get_python)" "$PROJECT_DIR/main.py" --status 2>/dev/null || log_warn "Could not run status check"
}

# Command: logs
cmd_logs() {
    echo "Tailing logs (Ctrl+C to stop)..."
    echo "─────────────────────────────────────────"
    tail -f "$BOT_LOG" "$DASHBOARD_LOG" 2>/dev/null || log_error "No logs found"
}

# Command: bot only
cmd_bot() {
    print_banner
    setup_dirs
    check_venv
    start_caffeinate
    start_bot
}

# Command: dashboard only
cmd_dashboard() {
    print_banner
    setup_dirs
    check_venv
    start_dashboard
    open_browser &
}

# Command: demo (foreground)
cmd_demo() {
    check_venv
    cd "$PROJECT_DIR"
    "$(get_python)" main.py --demo
}

# Command: scan (foreground)
cmd_scan() {
    check_venv
    cd "$PROJECT_DIR"
    "$(get_python)" main.py --scan
}

# Show usage
show_usage() {
    print_banner
    echo "Usage: $0 [command]"
    echo
    echo "Commands:"
    echo "  start      Start trading bot + dashboard (default)"
    echo "  stop       Stop all components gracefully"
    echo "  restart    Restart all components"
    echo "  status     Show running status"
    echo "  logs       Tail all logs"
    echo "  bot        Start trading bot only"
    echo "  dashboard  Start dashboard only"
    echo "  demo       Run demo mode (foreground)"
    echo "  scan       Run single scan (foreground)"
    echo "  help       Show this help"
    echo
    echo "Environment variables:"
    echo "  SHARIAH_DASHBOARD_PORT  Dashboard port (default: 8501)"
    echo
    echo "Quick start:"
    echo "  $0 start   # Starts everything and opens dashboard"
    echo "  $0 stop    # Stops everything gracefully"
    echo
}

# Main
case "${1:-start}" in
    start)
        cmd_start
        ;;
    stop)
        cmd_stop
        ;;
    restart)
        cmd_restart
        ;;
    status)
        cmd_status
        ;;
    logs)
        cmd_logs
        ;;
    bot)
        cmd_bot
        ;;
    dashboard)
        cmd_dashboard
        ;;
    demo)
        cmd_demo
        ;;
    scan)
        cmd_scan
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        log_error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac
