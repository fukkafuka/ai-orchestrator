#!/bin/bash
# orchestrator_health_check.sh
# com.fk.orchestrator (port 11437) の死活監視・自動再起動スクリプト
# cron: */5 * * * * /bin/bash ~/ai-orchestrator/orchestrator_health_check.sh

LOG_FILE="$HOME/ai-orchestrator/health_check.log"
MAX_LOG_BYTES=1048576   # 1MB
MAX_GENERATIONS=3
PORT=11437
SERVICE="com.fk.orchestrator"
TIMEOUT=5

# ---------- ログローテーション ----------
rotate_log() {
    if [ -f "$LOG_FILE" ] && [ "$(stat -f%z "$LOG_FILE" 2>/dev/null || echo 0)" -ge "$MAX_LOG_BYTES" ]; then
        for i in $(seq $((MAX_GENERATIONS - 1)) -1 1); do
            [ -f "${LOG_FILE}.$i" ] && mv "${LOG_FILE}.$i" "${LOG_FILE}.$((i+1))"
        done
        mv "$LOG_FILE" "${LOG_FILE}.1"
    fi
}

log() {
    rotate_log
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" >> "$LOG_FILE"
}

# ---------- 死活確認 ----------
check_port() {
    # orchestrator は HTTPS で動作（自己署名証明書のため -k）
    if curl -sfk --max-time "$TIMEOUT" "https://127.0.0.1:${PORT}/" > /dev/null 2>&1; then
        return 0
    fi
    return 1
}

# ---------- 再起動 ----------
restart_service() {
    log "[WARN] Port $PORT 応答なし。再起動を試みます: $SERVICE"
    launchctl stop "$SERVICE" 2>/dev/null
    sleep 2
    launchctl start "$SERVICE" 2>/dev/null
    sleep 5

    if check_port; then
        log "[INFO] 再起動成功: $SERVICE が port $PORT で応答"
    else
        log "[CRITICAL] 再起動後も応答なし: $SERVICE — 手動確認が必要です"
        # Mac通知（オプション）
        osascript -e 'display notification "orchestrator が応答しません。手動確認してください。" with title "MythoFable Alert" sound name "Basso"' 2>/dev/null
    fi
}

# ---------- メイン ----------
if check_port; then
    # 正常時はログを出さない（ノイズを減らす）
    exit 0
else
    restart_service
fi
