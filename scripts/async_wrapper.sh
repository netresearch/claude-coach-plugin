#!/bin/bash
# Async wrapper for coach hooks
# Spawns the actual script in background and returns immediately
# Usage: async_wrapper.sh <script.py> [args...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$1"
shift

# Read stdin if available (for hooks that pass data via stdin)
STDIN_DATA=""
if [ ! -t 0 ]; then
    STDIN_DATA=$(cat)
fi

# Spawn in background with nohup, redirect output to /dev/null
if [ -n "$STDIN_DATA" ]; then
    echo "$STDIN_DATA" | nohup python3 "$SCRIPT_DIR/$SCRIPT" "$@" > /dev/null 2>&1 &
else
    nohup python3 "$SCRIPT_DIR/$SCRIPT" "$@" > /dev/null 2>&1 &
fi

# Return immediately with success
exit 0
