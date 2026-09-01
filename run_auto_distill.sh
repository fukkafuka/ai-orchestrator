#!/bin/bash
LOG=/Users/fk/Logs/auto_distill.log
LOCK=/tmp/auto_distill.lock

# 多重起動防止
if [ -f "$LOCK" ]; then
    OLD_PID=$(cat "$LOCK")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "[$(date '+%H:%M:%S')] ⚠️ auto_distill already running (PID $OLD_PID), skipping" >> $LOG
        exit 0
    fi
fi

echo $$ > "$LOCK"
echo "[$(date '+%H:%M:%S')] 🌱 run_auto_distill.sh starting auto_distill" >> $LOG

/Users/fk/.pyenv/versions/3.11.9/bin/python3 -W ignore /Users/fk/ai-orchestrator/auto_distill.py >> $LOG 2>&1
EXIT_CODE=$?

rm -f "$LOCK"

if [ $EXIT_CODE -ne 0 ]; then
    echo "[$(date '+%H:%M:%S')] ❌❌❌ auto_distill CRASHED with exit code $EXIT_CODE ❌❌❌" >> $LOG
fi
