#!/bin/bash
LOG=/Users/fk/Logs/harvest_distill.log
LOCK=/tmp/harvest_distill.lock

# 多重起動防止
if [ -f "$LOCK" ]; then
    OLD_PID=$(cat "$LOCK")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "[$(date '+%H:%M:%S')] ⚠️ harvest_distill already running (PID $OLD_PID), skipping" >> $LOG
        exit 0
    fi
fi

echo $$ > "$LOCK"
echo "[$(date '+%H:%M:%S')] 🚀 run_harvest_distill.sh starting" >> $LOG

/usr/bin/python3 -W ignore /Users/fk/ai-orchestrator/harvest_distill_data.py >> $LOG 2>&1
EXIT_CODE=$?

rm -f "$LOCK"
echo "[$(date '+%H:%M:%S')] ✅ harvest_distill finished (exit=$EXIT_CODE)" >> $LOG
