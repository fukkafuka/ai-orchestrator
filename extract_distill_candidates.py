#!/usr/bin/env python3
"""
蒸留候補抽出スクリプト

cache.db の会話履歴から、
  - プレフィックスなし（＝ローカルモデル ask_local() が応答した）
  - ある程度の長さがある（雑談・単語だけの入力は除外）
  - 既に distill_claude_authored.jsonl に採用済みの instruction は除外
  - 直近の重複はまとめる（同一 instruction は1回だけ）
という条件で「次に模範解答を作るべき質問」を抽出し、標準出力に一覧表示する。

使い方:
    cd ~/ai-orchestrator
    python3 extract_distill_candidates.py                  # 直近50件
    python3 extract_distill_candidates.py --limit 100       # 件数を変える
    python3 extract_distill_candidates.py --json > out.json # JSONで保存してClaudeに渡す

このスクリプト自体は cache.db を読み取るだけで、書き換えは一切行わない。
"""
import argparse
import json
import os
import sqlite3

CACHE_DB = os.path.expanduser("~/ai-orchestrator/cache.db")
DISTILL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "distill_claude_authored.jsonl")

# ローカルモデルが処理する = これらのプレフィックスで始まらない発言
PREFIX_CHARS = ("。", ".", "、", ",")

MIN_LEN = 8  # これより短い発言(「はい」「OK」等)は候補から除外


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50, help="抽出する候補数の上限")
    ap.add_argument("--json", action="store_true", help="JSON配列で出力する(Claudeに渡す用)")
    args = ap.parse_args()

    if not os.path.exists(CACHE_DB):
        print(f"cache.db が見つかりません: {CACHE_DB}")
        return

    existing = load_existing_instructions()

    conn = sqlite3.connect(CACHE_DB)
    rows = conn.execute(
        """
        SELECT content, created_at
        FROM conversations
        WHERE role = 'user'
        ORDER BY id DESC
        LIMIT 2000
        """
    ).fetchall()
    conn.close()

    candidates = []
    seen = set()
    for content, created_at in rows:
        text = (content or "").strip()
        if not text or len(text) < MIN_LEN:
            continue
        if text.startswith(PREFIX_CHARS):
            continue  # クラウド/自動修正/引継ぎなどはローカルモデルの対象外
        if text in existing or text in seen:
            continue
        seen.add(text)
        candidates.append({"instruction": text, "seen_at": created_at})
        if len(candidates) >= args.limit:
            break

    if args.json:
        print(json.dumps(candidates, ensure_ascii=False, indent=2))
    else:
        print(f"候補 {len(candidates)} 件（既存 {len(existing)} 件を除外済み）\n")
        for i, c in enumerate(candidates, 1):
            print(f"{i:3d}. [{c['seen_at']}] {c['instruction'][:80]}")


if __name__ == "__main__":
    main()
