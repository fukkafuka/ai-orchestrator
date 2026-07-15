#!/bin/bash
LOG=/Users/fk/Logs/agent_claude.log
LOCK=/tmp/agent_claude.lock

# 多重起動防止
if [ -f "$LOCK" ]; then
    OLD_PID=$(cat "$LOCK")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "[$(date '+%H:%M:%S')] ⚠️ agent_claude already running (PID $OLD_PID), skipping" >> $LOG
        exit 0
    fi
fi

echo $$ > "$LOCK"
echo "[$(date '+%H:%M:%S')] 🚀 run_agent.sh starting agent_claude" >> $LOG

/usr/bin/python3 -W ignore /Users/fk/ai-agent/moltbook/agent_claude.py >> $LOG 2>&1
EXIT_CODE=$?

rm -f "$LOCK"

if [ $EXIT_CODE -ne 0 ]; then
    echo "[$(date '+%H:%M:%S')] ❌❌❌ agent_claude CRASHED with exit code $EXIT_CODE ❌❌❌" >> $LOG
fi
