#!/usr/bin/env python3
"""
harvest_moltbook_distill.py

moltbook-agent の memory.db から自分(claude)のpost/comment由来の新規トピックを抽出し、
OpenRouter(MODEL_CLOUD、他クラウドルーティングと同じモデル)でQ&A形式(instruction/output)
に書き起こした上で distill_claude_authored.jsonl に追記する。

【設計方針】(2026-08-25、かつさんとの相談に基づく)
- Moltbookの投稿・コメント本文は「1-2 sentences, direct, no fluff」という短文SNSスタイルで
  生成されており、distill_claude_authored.jsonlが想定する丁寧なQ&A形式とは文体が異なる。
  そのため本文をそのまま使わず、トピック(お題)だけを種にして毎回新規に書き起こす
  (2026-08-25の手動バッチ作業と同じ方針を自動化したもの)。
- 他のMoltbookエージェントの発言は使わず、あくまで自分(claude)自身の投稿・コメントの
  タイトル/話題のみを対象にする(extract_moltbook_topics.pyと同じ抽出条件)。
- 書き起こしにはMODEL_CLOUD(クラウドルーティングと同じモデル)を使う。ローカルモデル
  自身には書かせない(自己蒸留を避ける)。

冪等性: 処理済みトピック(正規化した文字列)をstate fileに記録し、再実行時に重複処理しない。
1回の実行で書き起こすトピック数は --limit で制御する(デフォルト10件、API呼び出し回数を抑制)。

使い方:
    cd ~/ai-orchestrator
    python3 harvest_moltbook_distill.py [--limit 10] [--dry-run]
"""
import argparse
import json
import os
import re
import sqlite3
import sys
from difflib import SequenceMatcher

import dotenv
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_status import filter_alive_models  # noqa: E402

dotenv.load_dotenv(os.path.expanduser("~/.config/ai-keys/.env"))
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"
MODEL_CLOUD = "meta-llama/llama-3.3-70b-instruct:free"

BASE = os.path.dirname(os.path.abspath(__file__))
MEMORY_DB = os.path.expanduser("~/ai-agent/moltbook/memory.db")
DISTILL_FILE = os.path.join(BASE, "distill_claude_authored.jsonl")
STATE_FILE = os.path.join(BASE, "moltbook_distill_processed.json")

sys.path.insert(0, os.path.expanduser("~/.config/ai-keys"))
try:
    from secret_sanitizer import sanitize_secrets
except Exception:
    def sanitize_secrets(text):
        return text

NOISE_DECOY_WORDS = {
    'lobster', 'shark', 'crab', 'octopus', 'squid', 'jellyfish',
    'starfish', 'urchin', 'clam', 'shrimp', 'dominance', 'territory',
    'physiology', 'senses', 'antenna', 'antennas',
}
MIN_TITLE_LEN = 6
NEAR_DUP_RATIO = 0.85

SYSTEM_PROMPT = """あなたは日本語のAIアシスタントとして、丁寧で正確な技術解説のQ&Aペアを作成する仕事をしています。
与えられたトピック(AIエージェント運用・機械学習・ソフトウェア工学などに関する短いフレーズ)を種として、
日本語の質問文(instruction)と、それに対する丁寧で具体的な説明文(output)のペアを1つ作成してください。

ルール:
- instructionは「〜とは何ですか」「〜について教えてください」のような自然な日本語の質問文にする
- outputは3〜6文程度。必要に応じて箇条書きを使い、具体例を交えて説明する
- トピックの原文の言い回しをそのまま使わず、その概念について新しく丁寧に書き起こす
- 断定できない具体的な統計・数値・固有の研究結果は創作しない
- 出力は次のJSON形式のみを返すこと。前置き・後書き・コードブロック記法(```)は一切含めないこと
{"instruction": "...", "output": "..."}"""


def normalize(s):
    return " ".join((s or "").strip().lower().split())


def looks_like_noise(text):
    words = set(normalize(text).split())
    return bool(words & NOISE_DECOY_WORDS)


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            pass
    return set()


def save_state(processed):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(processed), f, ensure_ascii=False, indent=2)


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


def fetch_candidate_topics():
    if not os.path.exists(MEMORY_DB):
        return []
    conn = sqlite3.connect(MEMORY_DB)
    cur = conn.cursor()
    topics = []
    cur.execute(
        """SELECT title FROM posts WHERE agent='claude' AND verified=1 ORDER BY created_at"""
    )
    topics += [r[0] for r in cur.fetchall()]
    cur.execute(
        """SELECT post_title FROM comments WHERE agent='claude' AND success=1 ORDER BY created_at"""
    )
    topics += [r[0] for r in cur.fetchall()]
    conn.close()

    seen_norms = []
    result = []
    for t in topics:
        t = (t or "").strip()
        if len(t) < MIN_TITLE_LEN or looks_like_noise(t):
            continue
        norm = normalize(t)
        if any(SequenceMatcher(None, norm, s).ratio() >= NEAR_DUP_RATIO for s in seen_norms):
            continue
        seen_norms.append(norm)
        result.append(t)
    return result


def call_openrouter_for_qa(topic):
    fallback_models = [MODEL_CLOUD] + filter_alive_models([
        "qwen/qwen3-next-80b-a3b-instruct:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "openai/gpt-oss-20b:free",
    ], provider="openrouter")
    seen = set()
    models_to_try = [m for m in fallback_models if not (m in seen or seen.add(m))]

    for m in models_to_try:
        try:
            r = requests.post(
                OPENROUTER_BASE,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:11437",
                    "X-Title": "Orchestrator v4 - distill harvest",
                },
                json={
                    "model": m,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"トピック: {topic}"},
                    ],
                    "max_tokens": 800,
                    "temperature": 0.5,
                    "reasoning": {"exclude": True},
                },
                timeout=30,
            )
            data = r.json()
            if "choices" not in data:
                continue
            content = (data["choices"][0]["message"]["content"] or "").strip()
            content = re.sub(r"^```(json)?|```$", "", content, flags=re.MULTILINE).strip()
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not match:
                continue
            obj = json.loads(match.group(0))
            instruction = (obj.get("instruction") or "").strip()
            output = (obj.get("output") or "").strip()
            if instruction and output:
                return instruction, output
        except Exception:
            continue
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10, help="1回の実行で書き起こすトピック数の上限")
    ap.add_argument("--dry-run", action="store_true", help="ファイルに書き込まず候補一覧のみ表示")
    args = ap.parse_args()

    if not OPENROUTER_API_KEY:
        print("OPENROUTER_API_KEY が設定されていません")
        return

    processed = load_state()
    existing = load_existing_instructions()
    topics = fetch_candidate_topics()
    new_topics = [t for t in topics if normalize(t) not in processed]

    print(f"Moltbookトピック候補: {len(topics)}件 / 未処理: {len(new_topics)}件")

    accepted = []
    tried = 0
    for topic in new_topics:
        if tried >= args.limit:
            break
        tried += 1
        processed.add(normalize(topic))

        instruction, output = call_openrouter_for_qa(topic)
        if not instruction or not output:
            continue
        if len(output) < 80:
            continue
        instruction = sanitize_secrets(instruction)
        output = sanitize_secrets(output)
        if instruction in existing:
            continue

        existing.add(instruction)
        accepted.append({"instruction": instruction, "output": output})

    print(f"書き起こし試行: {tried}件 / 採用: {len(accepted)}件")

    if args.dry_run:
        for r in accepted:
            print("-", r["instruction"])
        return

    if accepted:
        with open(DISTILL_FILE, "a", encoding="utf-8") as f:
            for r in accepted:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{DISTILL_FILE} に {len(accepted)} 件追記しました")

    save_state(processed)


if __name__ == "__main__":
    main()
