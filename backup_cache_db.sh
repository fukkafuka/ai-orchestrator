#!/bin/bash
# backup_cache_db.sh
# cache.db を日次バックアップ（7世代保持）
# cron: 0 3 * * * /bin/bash $HOME/ai-orchestrator/backup_cache_db.sh

SRC="$HOME/ai-orchestrator/cache.db"
BACKUP_DIR="$HOME/ai-orchestrator/backups"
KEEP_DAYS=7
LOG="$HOME/ai-orchestrator/health_check.log"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" >> "$LOG"
}

# DBが存在しない場合はスキップ
if [ ! -f "$SRC" ]; then
    log "[WARN] backup: cache.db が見つかりません: $SRC"
    exit 1
fi

mkdir -p "$BACKUP_DIR"

DEST="$BACKUP_DIR/cache_$(date '+%Y%m%d').db"

# 当日分が既にあればスキップ
if [ -f "$DEST" ]; then
    exit 0
fi

# SQLite の安全なバックアップ（.dump 経由でなく sqlite3 backup コマンド使用）
if command -v sqlite3 &>/dev/null; then
    sqlite3 "$SRC" ".backup '$DEST'"
else
    cp "$SRC" "$DEST"
fi

if [ $? -eq 0 ]; then
    SIZE=$(du -sh "$DEST" | cut -f1)
    log "[INFO] backup: cache.db → $(basename $DEST) ($SIZE)"
else
    log "[WARN] backup: cache.db バックアップ失敗"
    rm -f "$DEST"
    exit 1
fi

# 7日以上前のバックアップを削除
find "$BACKUP_DIR" -name "cache_*.db" -mtime +$KEEP_DAYS -delete

exit 0
