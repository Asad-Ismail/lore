#!/bin/bash
# Read cached suggestions from the daemon (written by post_query.sh)
# Returns JSON or empty string if no suggestions available.
cd "$(dirname "$0")/.."

SUGGESTION_FILE="data/.latest_suggestions"

if [ -f "$SUGGESTION_FILE" ]; then
    cat "$SUGGESTION_FILE"
else
    echo '{"suggestions":[]}'
fi
