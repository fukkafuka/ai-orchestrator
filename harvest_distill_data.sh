#!/bin/bash
# harvest_distill_data.sh
# クラウド・マルチエージェント・Moltbook由来の蒸留データを自動収集し、
# 変更があればコミット・pushする。cronからの定期実行を想定。
#
# 使い方:
#   ~/ai-orchestrator/harvest_distill_data.sh
#
# crontabへの登録例(毎日AM4:00に実行):
#   0 4 * * * /Users/fk/ai-orchestrator/harvest_distill_data.sh

LOG=/Users/fk/Logs/harvest_distill.log
LOCK=/tmp/harvest_distill.lock
cd /Users/fk/ai-orchestrator || exit 1

# 多重起動防止
if [ -f "$LOCK" ]; then
    OLD_PID=$(cat "$LOCK")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ harvest_distill_data already running (PID $OLD_PID), skipping" >> "$LOG"
        exit 0
    fi
fi
echo $$ > "$LOCK"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🚀 harvest_distill_data.sh starting" >> "$LOG"

BEFORE_HASH=$(md5 -q distill_claude_authored.jsonl 2>/dev/null)

/usr/bin/python3 -W ignore harvest_cloud_multiagent.py >> "$LOG" 2>&1
/usr/bin/python3 -W ignore harvest_moltbook_distill.py >> "$LOG" 2>&1

AFTER_HASH=$(md5 -q distill_claude_authored.jsonl 2>/dev/null)

if [ "$BEFORE_HASH" != "$AFTER_HASH" ]; then
    LINE_COUNT=$(wc -l < distill_claude_authored.jsonl | tr -d ' ')
    git add distill_claude_authored.jsonl distill_harvest_state.json moltbook_distill_processed.json
    git commit -m "data: 蒸留データ自動収集(cron) - distill_claude_authored.jsonl ${LINE_COUNT}件" >> "$LOG" 2>&1
    git push >> "$LOG" 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ 変更あり・commit&push完了" >> "$LOG"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 変更なし" >> "$LOG"
fi

rm -f "$LOCK"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🏁 harvest_distill_data.sh finished" >> "$LOG"
