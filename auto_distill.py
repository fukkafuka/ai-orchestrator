#!/usr/bin/env python3
"""
auto_distill.py

distill_claude_authored.jsonl を自動的に拡充するスクリプト。
以下の3つのソースから新規データを収集し、品質フィルタを通過したものを追記する。

  1. クラウドルーティング(cache.db の cache テーブル, source='cloud')
     = MODEL_CLOUD(現状 meta-llama/llama-3.3-70b-instruct:free)の応答
  2. マルチエージェントルーティング(cache.db, source='multi_agent')
     = Agent B/C並列→Agent A統合後の最終回答
  3. Moltbook投稿・コメント(moltbook-agent の memory.db)
     = 実際の投稿タイトルを題材に、MODEL_CLOUDでQ&A形式に書き起こしたもの

【設計方針】
- ローカルモデル自身の応答(source='local')は自己蒸留になるため対象外
- 品質フィルタ: 質問・応答の長さ、直後の会話が訂正/否定的な反応でないか
- 冪等性: 前回処理済みのcache.id・Moltbookトピックを状態ファイル(.auto_distill_state.json)
  に記録し、cronで繰り返し実行しても重複処理・重複追加しない
- 新規追加があれば git add/commit/push まで自動で行う(auto_patch.pyのgit_commit_and_push()と
  同じロジックをここでも踏襲)

使い方:
    cd ~/ai-orchestrator && python3 auto_distill.py
    (cron経由での定期実行を想定。run_auto_distill.sh経由で呼び出す)
"""
import json
import os
import re
import sqlite3
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DB = os.path.expanduser("~/ai-orchestrator/cache.db")
MOLTBOOK_DB = os.path.expanduser("~/ai-agent/moltbook/memory.db")
DISTILL_FILE = os.path.join(BASE_DIR, "distill_claude_authored.jsonl")
STATE_FILE = os.path.join(BASE_DIR, ".auto_distill_state.json")

MIN_QUESTION_LEN = 8
MIN_ANSWER_LEN = 40
MAX_MOLTBOOK_PER_RUN = 10  # 1回の実行でMoltbook由来トピックをQ&A化する上限(API呼び出しコスト抑制)

NEGATIVE_SIGNAL_KEYWORDS = [
    "違います", "違う", "ちがう", "そうじゃない", "そうではない", "訂正",
    "間違い", "まちがい", "もう一度", "やり直し", "ダメ", "だめ",
    "エラーが", "失敗しました", "動きません", "動かない", "おかしい",
]

NOISE_DECOY_WORDS = {
    'lobster', 'shark', 'crab', 'octopus', 'squid', 'jellyfish',
    'starfish', 'urchin', 'clam', 'shrimp', 'dominance', 'territory',
    'physiology', 'senses', 'antenna', 'antennas',
}

sys.path.insert(0, os.path.expanduser("~/.config/ai-keys"))
try:
    from secret_sanitizer import sanitize_secrets
except Exception:
    def sanitize_secrets(text):
        return text

try:
    from model_status import filter_alive_models
except Exception:
    def filter_alive_models(models, provider=None):
        return models

try:
    import dotenv
    dotenv.load_dotenv(os.path.expanduser("~/.config/ai-keys/.env"))
except Exception:
    pass
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── 状態管理 ──────────────────────────────────────────

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                return json.load(f)
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


def normalize_topic(s):
    return " ".join((s or "").strip().lower().split())


def append_entries(entries):
    if not entries:
        return
    with open(DISTILL_FILE, "a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


# ── OpenRouter呼び出し(orchestrator_v4.pyのcall_openrouter()と同じ方式) ──

def call_openrouter_for_rewrite(messages, max_tokens=800, temperature=0.5):
    if not OPENROUTER_API_KEY:
        return None
    models_to_try = filter_alive_models([
        "nvidia/nemotron-3-super-120b-a12b:free",
        "openrouter/free",
    ], provider="openrouter")
    if not models_to_try:
        models_to_try = ["openrouter/free"]

    import requests
    errors = []
    for m in models_to_try:
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:11437",
                    "X-Title": "Orchestrator v4 auto_distill"
                },
                json={
                    "model": m,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "reasoning": {"exclude": True}
                },
                timeout=45
            )
            data = r.json()
            if "choices" not in data:
                errors.append(f"{m}: {data.get('error', {}).get('message', str(data))}")
                continue
            return data["choices"][0]["message"]["content"] or ""
        except Exception as e:
            errors.append(f"{m} exception: {e}")
            continue
    log(f"⚠️ OpenRouter全モデル失敗: {' / '.join(errors)}")
    return None


# ── ソース1・2: クラウド/マルチエージェント(cache.db) ──────────

def has_negative_followup(conn, session_id, after_id):
    """指定id以降で、同一セッションの直後のuser発言に否定的な反応がないか確認"""
    if not session_id:
        return False
    row = conn.execute(
        "SELECT content FROM conversations WHERE session_id=? AND role='user' AND id>? ORDER BY id ASC LIMIT 1",
        (session_id, after_id)
    ).fetchone()
    if not row:
        return False
    next_msg = row[0] or ""
    if len(next_msg) > 60:
        # 長い発言は新しい話題である可能性が高く、直前の回答への否定的反応とは限らないため対象外
        return False
    return any(kw in next_msg for kw in NEGATIVE_SIGNAL_KEYWORDS)


def collect_cloud_multiagent(state, existing_instructions):
    if not os.path.exists(CACHE_DB):
        log("cache.db が見つかりません、クラウド/マルチエージェント収集をスキップ")
        return [], state["last_cache_id"]

    conn = sqlite3.connect(CACHE_DB)
    rows = conn.execute(
        """
        SELECT id, question, answer, model, source, created_at
        FROM cache
        WHERE source IN ('cloud', 'multi_agent') AND id > ?
        ORDER BY id ASC
        """,
        (state["last_cache_id"],)
    ).fetchall()

    def find_conv_position(answer_text):
        r = conn.execute(
            "SELECT session_id, id FROM conversations WHERE role='assistant' AND content=? ORDER BY id DESC LIMIT 1",
            (answer_text,)
        ).fetchone()
        return r if r else (None, None)

    new_entries = []
    max_id = state["last_cache_id"]
    skipped_short = 0
    skipped_negative = 0
    skipped_dup = 0

    for cid, question, answer, model, source, created_at in rows:
        max_id = max(max_id, cid)
        q = (question or "").strip()
        a = (answer or "").strip()

        if len(q) < MIN_QUESTION_LEN or len(a) < MIN_ANSWER_LEN:
            skipped_short += 1
            continue
        if q in existing_instructions:
            skipped_dup += 1
            continue

        session_id, conv_id = find_conv_position(answer)
        if conv_id is not None and has_negative_followup(conn, session_id, conv_id):
            skipped_negative += 1
            continue

        entry = {
            "instruction": sanitize_secrets(q),
            "output": sanitize_secrets(a),
        }
        new_entries.append(entry)
        existing_instructions.add(q)

    conn.close()
    log(f"クラウド/マルチエージェント: 新規{len(rows)}件中 採用{len(new_entries)}件 "
        f"(短すぎ={skipped_short}, 直後に否定的反応={skipped_negative}, 重複={skipped_dup})")
    return new_entries, max_id


# ── ソース3: Moltbook(memory.db) ─────────────────────────

def fetch_moltbook_topics():
    if not os.path.exists(MOLTBOOK_DB):
        log("moltbook memory.db が見つかりません、Moltbook収集をスキップ")
        return []
    conn = sqlite3.connect(MOLTBOOK_DB)
    cur = conn.cursor()
    topics = []
    cur.execute(
        "SELECT title FROM posts WHERE agent='claude' AND verified=1 ORDER BY created_at DESC LIMIT 300"
    )
    topics += [r[0] for r in cur.fetchall() if r[0]]
    cur.execute(
        "SELECT post_title FROM comments WHERE agent='claude' AND success=1 ORDER BY created_at DESC LIMIT 300"
    )
    topics += [r[0] for r in cur.fetchall() if r[0]]
    conn.close()
    return topics


def rewrite_topic_to_qa(topic):
    """MODEL_CLOUD相当のモデルを使い、トピックを丁寧な日本語Q&A形式に書き起こす"""
    prompt = f"""以下はAIエージェント同士のSNS「Moltbook」に投稿された、ある技術的・概念的なトピックのタイトルです。

トピック: {topic}

このトピックが扱っている本質的な技術概念・考え方を題材に、日本語のQ&A形式の教材を1つ作成してください。
- instruction: 「〜とは何ですか」「〜はなぜですか」のような、丁寧な日本語の質問文
- output: 3〜6文程度の、丁寧で分かりやすい日本語の説明文(必要なら箇条書きも使ってよい)
- 元のトピックの言い回しをそのまま使わず、新しい言葉で書き直すこと
- 宗教的な主張、未検証の統計・論文引用、特定分野に偏りすぎる内容は避けること
- 出力は次のJSON形式のみ。前置きや説明、Markdownのコードブロック記法は一切付けないこと:
{{"instruction": "...", "output": "..."}}
"""
    content = call_openrouter_for_rewrite([{"role": "user", "content": prompt}])
    if not content:
        return None
    cleaned = re.sub(r'^```(json)?|```$', '', content.strip(), flags=re.MULTILINE).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    instruction = (obj.get("instruction") or "").strip()
    output = (obj.get("output") or "").strip()
    if len(instruction) < MIN_QUESTION_LEN or len(output) < MIN_ANSWER_LEN:
        return None
    return {
        "instruction": sanitize_secrets(instruction),
        "output": sanitize_secrets(output),
    }


def collect_moltbook(state, existing_instructions):
    if not OPENROUTER_API_KEY:
        log("OPENROUTER_API_KEY が見つかりません、Moltbook収集をスキップ")
        return [], state["processed_moltbook_topics"]

    processed = set(state["processed_moltbook_topics"])
    raw_topics = fetch_moltbook_topics()

    candidates = []
    seen_norm = set()
    for t in raw_topics:
        t = (t or "").strip()
        if len(t) < 6:
            continue
        norm = normalize_topic(t)
        if norm in processed or norm in seen_norm:
            continue
        words = set(normalize_topic(t).split())
        if words & NOISE_DECOY_WORDS:
            continue
        seen_norm.add(norm)
        candidates.append(t)
        if len(candidates) >= MAX_MOLTBOOK_PER_RUN:
            break

    new_entries = []
    newly_processed = []
    for topic in candidates:
        qa = rewrite_topic_to_qa(topic)
        newly_processed.append(normalize_topic(topic))
        if not qa:
            continue
        if qa["instruction"] in existing_instructions:
            continue
        new_entries.append(qa)
        existing_instructions.add(qa["instruction"])

    log(f"Moltbook: 候補{len(candidates)}件中 採用{len(new_entries)}件")
    updated_processed = list(processed) + newly_processed
    return new_entries, updated_processed


# ── git commit + push(auto_patch.pyのgit_commit_and_push()と同じ方式) ──

def find_git_root(path):
    d = os.path.dirname(os.path.abspath(path))
    while d != "/":
        if os.path.isdir(os.path.join(d, ".git")):
            return d
        d = os.path.dirname(d)
    return None


def git_commit_and_push(filepath, message, timeout=30):
    repo_root = find_git_root(filepath)
    if not repo_root:
        return {"ok": False, "skipped": True, "reason": "gitリポジトリではありません"}

    def run(args):
        return subprocess.run(["git", "-C", repo_root] + args, capture_output=True, text=True, timeout=timeout)

    add_r = run(["add", filepath])
    if add_r.returncode != 0:
        return {"ok": False, "skipped": False, "reason": f"git add失敗: {add_r.stderr.strip()}"}

    status_r = run(["status", "--porcelain", filepath])
    if not status_r.stdout.strip():
        return {"ok": True, "skipped": True, "reason": "変更なし"}

    commit_r = run(["commit", "-m", message])
    if commit_r.returncode != 0:
        return {"ok": False, "skipped": False, "reason": f"git commit失敗: {commit_r.stderr.strip()}"}

    push_r = run(["push"])
    if push_r.returncode != 0:
        return {"ok": False, "skipped": False, "reason": f"git push失敗(コミットは成功済み・要手動push): {push_r.stderr.strip()}"}

    return {"ok": True, "skipped": False, "reason": "commit+push成功"}


# ── メイン ────────────────────────────────────────────

def main():
    log("🌱 auto_distill.py 開始")
    state = load_state()
    existing_instructions = load_existing_instructions()
    before_count = len(existing_instructions)

    cloud_entries, new_last_cache_id = collect_cloud_multiagent(state, existing_instructions)
    moltbook_entries, new_processed_topics = collect_moltbook(state, existing_instructions)

    all_new = cloud_entries + moltbook_entries
    if all_new:
        append_entries(all_new)
        log(f"✅ {len(all_new)}件を distill_claude_authored.jsonl に追加 "
            f"({before_count}件 → {before_count + len(all_new)}件)")
    else:
        log("追加対象なし")

    state["last_cache_id"] = new_last_cache_id
    state["processed_moltbook_topics"] = new_processed_topics
    save_state(state)

    if all_new:
        result = git_commit_and_push(
            DISTILL_FILE,
            f"data: auto_distill.pyによる自動追加({len(all_new)}件: "
            f"クラウド/マルチエージェント{len(cloud_entries)}件, Moltbook{len(moltbook_entries)}件)"
        )
        if result["ok"] and not result["skipped"]:
            log("✅ git commit + push 完了")
        elif result["skipped"]:
            log(f"ℹ️ git commit スキップ: {result['reason']}")
        else:
            log(f"⚠️ git commit/push 失敗: {result['reason']}")

    log("🌱 auto_distill.py 終了")


if __name__ == "__main__":
    main()
