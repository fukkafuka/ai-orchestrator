#!/usr/bin/env python3
"""
harvest_cloud_multiagent.py

orchestrator_v4.py の cache.db に蓄積された「クラウド」(。/.)・「マルチエージェント」
(。。。/...)応答を自動収集し、品質フィルタを通した上で distill_claude_authored.jsonl
に直接追記する。

【設計方針】
- cache テーブルには question/answer/model/source が既に構造化されて保存されているため、
  クラウド・マルチエージェント分は(Moltbookと違い)Q&Aとして即座に使える形式であり、
  LLMによる書き直しは行わない(question=instruction, answer=outputとしてそのまま採用)。
- ローカル(ask_local)自身の回答は自己蒸留になり品質面で望ましくないため対象外
  (source IN ('cloud','multi_agent')のみを対象とする)。
- 品質フィルタ: 質問・回答それぞれの最低文字数に加え、直後のユーザー発言に
  訂正・否定的な反応がないか(conversationsテーブルを突き合わせて)確認する。

冪等性: 前回処理済みのcache.id(state fileに記録)より新しい行のみを対象にするため、
何度実行しても重複追加は起きない。

使い方:
    cd ~/ai-orchestrator
    python3 harvest_cloud_multiagent.py [--limit 200] [--dry-run]
"""
import argparse
import json
import os
import re
import sqlite3

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE_DB = os.path.expanduser("~/ai-orchestrator/cache.db")
DISTILL_FILE = os.path.join(BASE, "distill_claude_authored.jsonl")
STATE_FILE = os.path.join(BASE, "distill_harvest_state.json")

MIN_QUESTION_LEN = 8
MIN_ANSWER_LEN = 40

# 直後のユーザー発言にこれらが含まれ、かつ短文(NEGATIVE_MAX_LEN以下)の場合は
# 「直前の回答への訂正・不満」とみなし、その回答は蒸留データの候補から除外する
NEGATIVE_SIGNAL_KEYWORDS = [
    "違う", "ちがう", "そうじゃない", "そうではない", "訂正", "間違い", "まちがい",
    "もう一度", "やり直し", "ダメ", "だめ", "エラー", "失敗しました", "動きません",
    "動かない", "違います", "誤り", "おかしい",
]
NEGATIVE_MAX_LEN = 60


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_cache_id": 0}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def load_existing_instructions():
    existing = set()
    if os.path.exists(DISTILL_FILE):
        with open(DISTILL_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    existing.add(obj.get("instruction", "").strip())
                except json.JSONDecodeError:
                    continue
    return existing


def has_negative_followup(conn, answer_text, created_at):
    """このanswerの直後にあるユーザー発言が訂正・否定的な反応かどうかを判定する"""
    row = conn.execute(
        """SELECT id, session_id FROM conversations
           WHERE role='assistant' AND content=?
           ORDER BY ABS(julianday(created_at) - julianday(?)) ASC
           LIMIT 1""",
        (answer_text, created_at),
    ).fetchone()
    if not row:
        return False  # 対応する会話ログが見つからない場合は保守的にOK扱い
    assistant_id, session_id = row
    next_row = conn.execute(
        """SELECT content FROM conversations
           WHERE session_id=? AND id > ? AND role='user'
           ORDER BY id ASC LIMIT 1""",
        (session_id, assistant_id),
    ).fetchone()
    if not next_row:
        return False
    next_text = (next_row[0] or "").strip()
    if len(next_text) > NEGATIVE_MAX_LEN:
        return False  # 長文の場合は新しい話題である可能性が高く、訂正とは判定しない
    return any(kw in next_text for kw in NEGATIVE_SIGNAL_KEYWORDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200, help="1回の実行で追加する上限件数")
    ap.add_argument("--dry-run", action="store_true", help="ファイルに書き込まず候補一覧のみ表示")
    args = ap.parse_args()

    if not os.path.exists(CACHE_DB):
        print(f"cache.db が見つかりません: {CACHE_DB}")
        return

    state = load_state()
    existing = load_existing_instructions()

    conn = sqlite3.connect(CACHE_DB)
    rows = conn.execute(
        """SELECT id, question, answer, model, source, created_at
           FROM cache
           WHERE source IN ('cloud', 'multi_agent') AND id > ?
           ORDER BY id ASC""",
        (state["last_cache_id"],),
    ).fetchall()

    accepted = []
    max_id_seen = state["last_cache_id"]
    skipped_short = 0
    skipped_dup = 0
    skipped_negative = 0

    for cid, question, answer, model, source, created_at in rows:
        max_id_seen = max(max_id_seen, cid)
        q = (question or "").strip()
        a = (answer or "").strip()

        if len(q) < MIN_QUESTION_LEN or len(a) < MIN_ANSWER_LEN:
            skipped_short += 1
            continue
        if q in existing:
            skipped_dup += 1
            continue
        if has_negative_followup(conn, a, created_at):
            skipped_negative += 1
            continue

        existing.add(q)
        accepted.append({"instruction": q, "output": a})
        if len(accepted) >= args.limit:
            break

    conn.close()

    print(f"新規cache行: {len(rows)}件 / 採用: {len(accepted)}件 "
          f"(除外: 短すぎ={skipped_short}, 重複={skipped_dup}, 否定的フォローアップ={skipped_negative})")

    if args.dry_run:
        for r in accepted[:10]:
            print("-", r["instruction"][:60])
        return

    if accepted:
        with open(DISTILL_FILE, "a", encoding="utf-8") as f:
            for r in accepted:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{DISTILL_FILE} に {len(accepted)} 件追記しました")

    state["last_cache_id"] = max_id_seen
    save_state(state)


if __name__ == "__main__":
    main()
