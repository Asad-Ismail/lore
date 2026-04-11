#!/bin/bash
# Stop hook: runs automatically after each Claude Code turn.
# Handles daemon lifecycle, training triggers, and suggestion caching.
# Question traces are captured by the agent via `lore trace` (CLAUDE.md step 5).

set -e
cd "$(dirname "$0")/.."

DAEMON_URL="http://127.0.0.1:8765"
CHECKPOINT_DIR="data/lora_checkpoints"
SUGGESTION_FILE="data/.latest_suggestions"

# Check if daemon is running
daemon_running() {
    curl -s -o /dev/null -w "%{http_code}" "$DAEMON_URL/health" 2>/dev/null | grep -q "200"
}

# Start daemon if checkpoint exists but daemon not running
if ! daemon_running; then
    if [ -d "$CHECKPOINT_DIR" ] && [ "$(ls -A $CHECKPOINT_DIR 2>/dev/null)" ]; then
        nohup uv run lore-train serve > data/.daemon.log 2>&1 &
        echo $! > data/.daemon.pid
        for i in $(seq 1 30); do
            sleep 1
            daemon_running && break
        done
    fi
fi

# If daemon is running, handle training and suggestions
if daemon_running; then
    # Trigger training if thresholds crossed
    [ -f "data/.curiosity_suggested" ] && curl -s -X POST "$DAEMON_URL/train/curiosity" > /dev/null 2>&1 && rm -f "data/.curiosity_suggested"

    # Cache fresh suggestions
    SUGGESTIONS=$(curl -s "$DAEMON_URL/suggest?n=3" 2>/dev/null || echo "")
    [ -n "$SUGGESTIONS" ] && echo "$SUGGESTIONS" > "$SUGGESTION_FILE"
fi
