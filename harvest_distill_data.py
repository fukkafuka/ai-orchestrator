#!/usr/bin/env python3
"""
harvest_distill_data.py

ai-orchestratorの実運用ログ(クラウド・マルチエージェント応答)と、
moltbook-agentの投稿・コメント由来トピックから、蒸留データ
(distill_claude_authored.jsonl)を自動的に蓄積・拡充するスクリプト。

対象:
  - cloud/multi_agentの実応答: そのままinstruction/outputペアとして採用
    (質問と回答の組がすでに実データなので書き起こし不要)
  - Moltbookのpost/comment由来トピック: OpenRouter経由でinstruction/output
    ペアに書き起こしてから採用

除外:
  - ローカルモデル自身の応答(source='local'): 自己蒸留(同じ癖を強化するだけ)
    を避けるため対象外

品質フィルタ:
  - 質問・回答が短すぎるものは除外
  - 直後にユーザーが訂正・否定的な発言をしている場合は除外
    (「会話の続き方」で低品質だったと判断できる回答)
  - Moltbookトピックは、CAPTCHA攪乱ノイズ語・宗教/検証不能な統計主張らしき
    ものを事前に除外し、さらに書き起こし時にLLM自身にも不適格なら
    スキップさせる(二重チェック)

使い方:
    cd ~/ai-orchestrator
    python3 harvest_distill_data.py            # 通常実行(蓄積→追記→git push)
    python3 harvest_distill_data.py --dry-run  # 追記・pushせず候補を表示するだけ

cron等で定期実行することを想定。実行の都度、前回処理済みの位置から続きを
処理する(状態はharvest_state.jsonに保存、.gitignore対象、リポジトリには
含めない)。
"""
import argparse
import json
import os
import re
import sqlite3
import sys
import time

import dotenv
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from model_status import filter_alive_models  # noqa: E402
from auto_patch import git_commit_and_push  # noqa: E402

dotenv.load_dotenv(os.path.expanduser("~/.config/ai-keys/.env"))
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"

# secret_sanitizer は ~/.config/ai-keys/ 配下(git管理外)。読み込めない場合は
# 素通し(fail-open)にして処理自体は止めない。
sys.path.insert(0, os.path.expanduser("~/.config/ai-keys"))
try:
    from secret_sanitizer import sanitize_secrets
except Exception:
    def sanitize_secrets(text):
        return text

CACHE_DB = os.path.expanduser("~/ai-orchestrator/cache.db")
MOLTBOOK_DB = os.path.expanduser("~/ai-agent/moltbook/memory.db")
DISTILL_FILE = os.path.join(BASE, "distill_claude_authored.jsonl")
STATE_FILE = os.path.join(BASE, "harvest_state.json")

MIN_QUESTION_LEN = 8
MIN_ANSWER_LEN = 40
MAX_CACHE_PER_RUN = 50      # cache.db由来は書き起こし不要(API呼び出しなし)なので多めに
MAX_MOLTBOOK_PER_RUN = 5    # Moltbook由来はLLM呼び出しを伴うため控えめに

# 直後のユーザー発言がこれらを含み、かつ短い(<=30文字)場合は
# 直前の回答を「訂正・否定された低品質な回答」とみなして除外する
NEGATIVE_SIGNAL_KEYWORDS = [
    "違う", "ちがう", "そうじゃない", "そうではない", "訂正", "間違い", "まちがい",
    "もう一度", "やり直し", "ダメ", "だめ", "エラーが", "失敗しました",
    "動きません", "動かない", "違います", "おかしい",
]

# CAPTCHA攪乱用に使われるダミー単語(extract_moltbook_topics.pyと同じ基準)
NOISE_DECOY_WORDS = {
    "lobster", "shark", "crab", "octopus", "squid", "jellyfish",
    "starfish", "urchin", "clam", "shrimp", "dominance", "territory",
    "physiology", "senses", "antenna", "antennas",
}
# 宗教勧誘・検証不能な引用らしきトピックを事前に弾くための簡易キーワード
UNSUITABLE_TOPIC_KEYWORDS = [
    "lord rayel", "divine kingdom", "messiah", "sacred", "scripture",
    "torah", "holy scripture", "arxiv", "cve-",
]

REWRITE_MODELS = filter_alive_models([
    "meta-llama/llama-3.3-70b-instruct:free",
    "openai/gpt-oss-20b:free",
    "openai/gpt-oss-120b:free",
], provider="openrouter")

REWRITE_SYSTEM_PROMPT = """あなたは日本語の技術Q&Aデータセットを作る専門家です。
与えられたトピック(英語の場合あり)を題材に、日本語で「質問(instruction)」と
「丁寧で正確な説明(output)」のペアを1つ作成してください。

ルール:
- instructionは自然な日本語の質問文にすること(元のトピックの直訳ではなく、
  説明を求める質問に言い換える)
- outputは3〜6文程度、必要なら箇条書きを使い、事実に基づいた正確な説明にすること
- 数値や固有名詞を捏造しないこと。断定できないことは断定しないこと
- 宗教的な勧誘、政治的な主張、検証不能な統計主張、下品・攻撃的な内容の場合は、
  Q&Aを作らず {"skip": true} という1行のJSONだけを返すこと
- それ以外の場合は必ず {"instruction": "...", "output": "..."} という1行の
  JSON形式のみを返すこと。説明や前置き、Markdownのコードブロック記号は
  一切付けないこと
"""


def normalize(s):
    """全重複文字を除去(順序保持)"""
    seen = {}
    return "".join(c for c in s if not (c in seen or seen.update({c: 1})))


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            return json.load(open(STATE_FILE, encoding="utf-8"))
        except Exception:
            pass
    return {"last_cache_id": 0, "processed_moltbook_topics": []}


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


def is_negatively_followed(conn, answer_text, created_at):
    """この回答の直後にユーザーが訂正・否定的な発言をしていないか確認する"""
    row = conn.execute(
        "SELECT id, session_id FROM conversations WHERE role='assistant' AND content=? "
        "ORDER BY ABS(julianday(created_at) - julianday(?)) LIMIT 1",
        (answer_text, created_at)
    ).fetchone()
    if not row:
        return False  # 対応する会話ログが見つからない場合は安全側(除外しない)
    msg_id, session_id = row
    next_row = conn.execute(
        "SELECT content FROM conversations WHERE session_id=? AND role='user' AND id>? "
        "ORDER BY id ASC LIMIT 1",
        (session_id, msg_id)
    ).fetchone()
    if not next_row:
        return False
    next_content = (next_row[0] or "").strip()
    if len(next_content) <= 30 and any(kw in next_content for kw in NEGATIVE_SIGNAL_KEYWORDS):
        return True
    return False


def harvest_cache_qa(state, existing, dry_run=False):
    """cache.dbのcloud/multi_agent応答を蒸留データとして採用する"""
    if not os.path.exists(CACHE_DB):
        print(f"cache.db が見つかりません: {CACHE_DB}")
        return []

    conn = sqlite3.connect(CACHE_DB)
    rows = conn.execute(
        "SELECT id, question, answer, source, created_at FROM cache "
        "WHERE source IN ('cloud','multi_agent') AND id > ? ORDER BY id ASC LIMIT ?",
        (state.get("last_cache_id", 0), MAX_CACHE_PER_RUN)
    ).fetchall()

    accepted = []
    max_id = state.get("last_cache_id", 0)
    for cid, question, answer, source, created_at in rows:
        max_id = max(max_id, cid)
        q = (question or "").strip()
        a = (answer or "").strip()
        if len(q) < MIN_QUESTION_LEN or len(a) < MIN_ANSWER_LEN:
            continue
        if q in existing:
            continue
        if is_negatively_followed(conn, a, created_at):
            print(f"  [skip:negative-followup] {q[:50]}")
            continue
        pair = {"instruction": sanitize_secrets(q), "output": sanitize_secrets(a)}
        accepted.append(pair)
        existing.add(q)
        print(f"  [accept:{source}] {q[:50]}")

    conn.close()
    state["last_cache_id"] = max_id
    return accepted


def fetch_moltbook_topics():
    """extract_moltbook_topics.pyと同等のロジックでMoltbookのトピック一覧を取得"""
    if not os.path.exists(MOLTBOOK_DB):
        print(f"moltbook memory.db が見つかりません: {MOLTBOOK_DB}")
        return []
    conn = sqlite3.connect(MOLTBOOK_DB)
    cur = conn.cursor()
    candidates = []
    cur.execute(
        "SELECT title FROM posts WHERE agent='claude' AND verified=1 "
        "ORDER BY created_at DESC LIMIT 500"
    )
    candidates.extend(row[0] for row in cur.fetchall())
    cur.execute(
        "SELECT post_title FROM comments WHERE agent='claude' AND success=1 "
        "ORDER BY created_at DESC LIMIT 500"
    )
    candidates.extend(row[0] for row in cur.fetchall())
    conn.close()
    return candidates


def is_topic_suitable(topic):
    t = (topic or "").strip()
    if len(t) < 6:
        return False
    words = {normalize(w) for w in re.findall(r"[a-zA-Z]+", t.lower())}
    if words & NOISE_DECOY_WORDS:
        return False
    low = t.lower()
    if any(kw in low for kw in UNSUITABLE_TOPIC_KEYWORDS):
        return False
    if re.search(r"\d+(\.\d+)?\s*%", t):
        return False  # 未検証の統計主張の可能性が高いものは避ける
    return True


def _extract_json_object(content):
    """モデルがMarkdownコードフェンス付きで返してきた場合の保険"""
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(json)?\s*", "", content)
        content = re.sub(r"```\s*$", "", content)
    return json.loads(content)


def rewrite_topic_to_qa(topic):
    """OpenRouter経由でトピックをinstruction/outputペアに書き起こす"""
    if not REWRITE_MODELS or not OPENROUTER_API_KEY:
        return None
    for model in REWRITE_MODELS:
        try:
            r = requests.post(
                OPENROUTER_BASE,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:11437",
                    "X-Title": "Orchestrator v4 - Distill Harvest",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                        {"role": "user", "content": f"トピック: {topic}"},
                    ],
                    "max_tokens": 700,
                    "temperature": 0.5,
                    "response_format": {"type": "json_object"},
                },
                timeout=45,
            )
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            obj = _extract_json_object(content)
            if obj.get("skip"):
                return None
            instr = (obj.get("instruction") or "").strip()
            out = (obj.get("output") or "").strip()
            if len(instr) < MIN_QUESTION_LEN or len(out) < MIN_ANSWER_LEN:
                return None
            return {"instruction": sanitize_secrets(instr), "output": sanitize_secrets(out)}
        except Exception as e:
            print(f"  [rewrite失敗:{model}] {e}")
            continue
    return None


def harvest_moltbook_qa(state, existing, dry_run=False):
    topics = fetch_moltbook_topics()
    processed = set(state.get("processed_moltbook_topics", []))
    accepted = []
    tried = 0
    for topic in topics:
        topic = (topic or "").strip()
        if not topic:
            continue
        norm_topic = normalize(topic.lower())
        if norm_topic in processed:
            continue
        if not is_topic_suitable(topic):
            processed.add(norm_topic)  # 不適格判定も既処理として記録し、毎回再判定しない
            continue
        if tried >= MAX_MOLTBOOK_PER_RUN:
            break
        tried += 1
        processed.add(norm_topic)
        if dry_run:
            print(f"  [dry-run:would-rewrite] {topic[:60]}")
            continue
        pair = rewrite_topic_to_qa(topic)
        if pair is None:
            print(f"  [skip:rewrite-none] {topic[:60]}")
            continue
        if pair["instruction"] in existing:
            continue
        accepted.append(pair)
        existing.add(pair["instruction"])
        print(f"  [accept:moltbook] {pair['instruction'][:50]}")
        time.sleep(1)  # レート制限対策

    state["processed_moltbook_topics"] = list(processed)
    return accepted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="追記・pushせず候補を表示するだけ")
    args = ap.parse_args()

    state = load_state()
    existing = load_existing_instructions()

    print("=== cache.db (cloud/multi_agent) ===")
    cache_pairs = harvest_cache_qa(state, existing, dry_run=args.dry_run)

    print("=== moltbook トピック ===")
    moltbook_pairs = harvest_moltbook_qa(state, existing, dry_run=args.dry_run)

    all_pairs = cache_pairs + moltbook_pairs
    print(f"\n新規採用: cache={len(cache_pairs)}件, moltbook={len(moltbook_pairs)}件, "
          f"合計={len(all_pairs)}件")

    if args.dry_run:
        print("(--dry-run のため書き込み・pushは行いません)")
        return

    if all_pairs:
        with open(DISTILL_FILE, "a", encoding="utf-8") as f:
            for p in all_pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"{DISTILL_FILE} に {len(all_pairs)} 件追記しました")

    save_state(state)

    if all_pairs:
        result = git_commit_and_push(
            DISTILL_FILE,
            f"data: 自動蒸留収集(cache由来{len(cache_pairs)}件 + moltbook由来{len(moltbook_pairs)}件)",
        )
        print(f"git: {result}")


if __name__ == "__main__":
    main()
