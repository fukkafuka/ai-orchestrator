#!/usr/bin/env python3

from pathlib import Path
import os
"""
オーケストレーター v4
- 会話履歴を保持
- ; プレフィックス → OpenRouter（クラウド・インターネット検索）
- それ以外 → キャッシュ確認 → ローカルモデル（llama.cpp / llm-jp-3-1.8B、外部通信なし）
- キャッシュ：SQLite + sentence-transformers（類似検索）
"""
import requests
import subprocess
import time
import json
import sqlite3
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string, redirect

import dotenv
import auto_patch
try:
    Test_API_KEY = "12345678901234567890asdfghjkl"

    from ddgs import DDGS
    DDG_AVAILABLE = True
except Exception:
    try:
        from duckduckgo_search import DDGS
        DDG_AVAILABLE = True
    except Exception:
        DDG_AVAILABLE = False
dotenv.load_dotenv(os.path.expanduser("~/.config/ai-keys/.env"))
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_BASE = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

# モデル設定
MODEL_CLOUD   = "meta-llama/llama-3.3-70b-instruct:free"  # クラウド（;プレフィックス）
MODEL_CLASSIFY = "meta-llama/llama-3.3-70b-instruct:free" # 分類用

# ローカル推論設定（llama.cpp llama-completion直接実行、外部通信なし。失敗時も外部フォールバックしない）
# 注: llama-cliは対話ループ仕様（-no-cnv未サポート）のためllama-completionを使用
LLAMA_COMPLETION_BIN = os.path.expanduser("~/llama.cpp/build/bin/llama-completion")
LOCAL_MODEL_PATH = os.path.expanduser("~/ai-orchestrator/llama.cpp/models/llm-jp-3-1.8b-instruct3-Q4_K_M.gguf")
LOCAL_MODEL_TIMEOUT = 300  # 秒（Intel Mac, CPU推論のため余裕を持たせる）

# ローカルVision設定（llama-mtmd-cli / SmolVLM-500M, 外部通信なし。通常モード+画像で使用）
LLAMA_MTMD_BIN = os.path.expanduser("~/llama.cpp/build/bin/llama-mtmd-cli")
VISION_MODEL_PATH = os.path.expanduser("~/ai-orchestrator/llama.cpp/models/smolvlm-500m-instruct-q8_0.gguf")
VISION_MMPROJ_PATH = os.path.expanduser("~/ai-orchestrator/llama.cpp/models/mmproj-smolvlm-500m-instruct-q8_0.gguf")
VISION_TIMEOUT = 120  # 秒（画像エンコード+生成で60秒超かかることがあるため余裕を持たせる）

# OCR設定（Tesseract, 外部通信なし。通常モード+画像で画像内の文字を正確に読み取るため使用）
TESSERACT_BIN = "/usr/local/bin/tesseract"  # launchd環境はPATHが限定的なためフルパス指定
OCR_LANGS = "jpn+eng"
OCR_TIMEOUT = 30  # 秒

# キャッシュ設定
CACHE_DB = os.path.expanduser("~/ai-orchestrator/cache.db")
CACHE_SIMILARITY_THRESHOLD = 0.80  # 80%以上の類似度でキャッシュヒット

# sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    EMBED_AVAILABLE = True
except Exception as e:
    print(f"sentence-transformers unavailable: {e}")
    EMBED_AVAILABLE = False

conversation_histories = {}  # {session_id: [messages]}
MAX_HISTORY = 20
app = Flask(__name__)

import hashlib as _hashlib_v, time as _time_v
_BOOT_TIME = _time_v.strftime("%Y-%m-%d %H:%M:%S")

@app.route('/version', methods=['GET'])
def version():
    """デプロイ確認用: このプロセスが読み込んでいるコードのハッシュと起動時刻"""
    with open(__file__, 'rb') as f:
        code_hash = _hashlib_v.md5(f.read()).hexdigest()[:8]
    return jsonify({"code_hash": code_hash, "boot_time": _BOOT_TIME})

LOG_FILE = "/Users/fk/Logs/orc.log"
MEMORY_DB = os.path.expanduser("~/ai-agent/moltbook/memory.db")

# ── 診断モード用：システム名 → ログファイルのマッピング ──────────────
DIAGNOSIS_LOG_MAP = {
    "オーケストレーター": ["/tmp/orchestrator.stderr.log", os.path.expanduser("~/ai-orchestrator/health_check.log")],
    "orchestrator": ["/tmp/orchestrator.stderr.log", os.path.expanduser("~/ai-orchestrator/health_check.log")],
    "moltbook": [os.path.expanduser("~/Logs/agent_claude.log")],
    "エージェント": [os.path.expanduser("~/Logs/agent_claude.log")],
    "mythofable": [
        os.path.expanduser("~/Logs/watcher_stdout.log"),
        os.path.expanduser("~/Logs/watcher_stderr.log"),
        os.path.expanduser("~/Logs/dashboard_run.log"),
    ],
    "セキュリティ": [
        os.path.expanduser("~/Logs/watcher_stdout.log"),
        os.path.expanduser("~/Logs/watcher_stderr.log"),
        os.path.expanduser("~/Logs/dashboard_run.log"),
    ],
}
DIAGNOSIS_LOG_TAIL_LINES = 80

def get_agent_context(question, max_comments=3):
    """memory.dbからagent_claudeの関連コメント・dreamsを取得"""
    try:
        conn = sqlite3.connect(MEMORY_DB)
        
        # 最新のdreaming insights取得
        dream = conn.execute(
            "SELECT insights, style_notes FROM dreams ORDER BY id DESC LIMIT 1"
        ).fetchone()
        
        # 関連コメント・投稿をキーワード検索
        words = [w for w in question.split() if len(w) > 3][:5]
        comments = []
        my_posts = []
        for word in words:
            rows = conn.execute(
                "SELECT post_title, content FROM comments WHERE success=1 AND (post_title LIKE ? OR content LIKE ?) ORDER BY id DESC LIMIT 2",
                (f"%{word}%", f"%{word}%")
            ).fetchall()
            comments.extend(rows)
            post_rows = conn.execute(
                "SELECT title, content FROM posts WHERE (title LIKE ? OR content LIKE ?) ORDER BY id DESC LIMIT 2",
                (f"%{word}%", f"%{word}%")
            ).fetchall()
            my_posts.extend(post_rows)

        # 重複除去
        seen = set()
        unique_comments = []
        for c in comments:
            if c[1] not in seen:
                seen.add(c[1])
                unique_comments.append(c)
        unique_comments = unique_comments[:max_comments]

        seen_posts = set()
        unique_posts = []
        for p in my_posts:
            if p[1] not in seen_posts:
                seen_posts.add(p[1])
                unique_posts.append(p)
        unique_posts = unique_posts[:2]

        conn.close()

        context = ""
        if dream:
            context += f"\n[あなた(fujikatsu-openclaw)のAIとしての知見]\n{dream[0][:300]}\n"
            context += f"\n[スタイル]\n{dream[1][:200]}\n"
        if unique_posts:
            context += "\n[Moltbookでの自分の投稿]\n"
            for title, post_content in unique_posts:
                context += f"- 投稿: {title[:60]}: {post_content[:150]}\n"
        if unique_comments:
            context += "\n[Moltbookでの関連議論経験]\n"
            for title, comment in unique_comments:
                context += f"- {title[:40]}: {comment[:150]}\n"
        
        return context
    except Exception as e:
        return ""

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"🐈[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ── キャッシュDB ──────────────────────────────────────

def save_conversation(role, content, session_id="default"):
    """会話履歴をDBに保存"""
    try:
        conn = sqlite3.connect(CACHE_DB)
        try:
            conn.execute("ALTER TABLE conversations ADD COLUMN session_id TEXT DEFAULT 'legacy'")
            conn.commit()
        except:
            pass
        conn.execute("INSERT INTO conversations (role, content, session_id) VALUES (?, ?, ?)", (role, content, session_id))
        conn.commit()
        conn.execute("DELETE FROM conversations WHERE session_id=? AND id NOT IN (SELECT id FROM conversations WHERE session_id=? ORDER BY id DESC LIMIT 200)", (session_id, session_id))
        conn.commit()
        conn.close()
    except Exception:
        pass

def load_conversation_history(limit=20, session_id="default"):
    """DBから会話履歴を読み込む"""
    try:
        conn = sqlite3.connect(CACHE_DB)
        rows = conn.execute(
            "SELECT role, content FROM conversations WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit)
        ).fetchall()
        conn.close()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
    except Exception:
        return []

def find_session_by_code(code):
    """短縮コード（末尾8文字）からセッションIDを検索"""
    try:
        conn = sqlite3.connect(CACHE_DB)
        rows = conn.execute(
            "SELECT DISTINCT session_id FROM conversations WHERE session_id LIKE ? AND session_id != 'legacy' ORDER BY id DESC LIMIT 1",
            (f"%{code}",)
        ).fetchall()
        conn.close()
        return rows[0][0] if rows else None
    except Exception:
        return None

def search_past_conversations(keyword, limit=3):
    """過去の会話をキーワード検索"""
    try:
        conn = sqlite3.connect(CACHE_DB)
        rows = conn.execute(
            "SELECT role, content, created_at FROM conversations WHERE content LIKE ? ORDER BY id DESC LIMIT ?",
            (f"%{keyword}%", limit)
        ).fetchall()
        conn.close()
        return rows
    except Exception:
        return []

def init_cache():
    conn = sqlite3.connect(CACHE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            session_id TEXT DEFAULT 'legacy',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            model TEXT NOT NULL,
            source TEXT NOT NULL,
            embedding BLOB,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    conn.close()

def embed(text):
    if not EMBED_AVAILABLE:
        return None
    vec = _embed_model.encode(text, normalize_embeddings=True)
    return vec.astype(np.float32).tobytes()

def cosine_similarity(a_bytes, b_bytes):
    a = np.frombuffer(a_bytes, dtype=np.float32)
    b = np.frombuffer(b_bytes, dtype=np.float32)
    return float(np.dot(a, b))

def cache_search(question):
    """類似キャッシュを検索。ヒットしたら(answer, model, source)を返す"""
    if not EMBED_AVAILABLE:
        return None
    q_vec = embed(question)
    if q_vec is None:
        return None
    conn = sqlite3.connect(CACHE_DB)
    rows = conn.execute("SELECT question, answer, model, source, embedding FROM cache").fetchall()
    conn.close()
    best_sim = 0
    best_row = None
    for row in rows:
        if row[4] is None:
            continue
        sim = cosine_similarity(q_vec, row[4])
        if sim > best_sim:
            best_sim = sim
            best_row = row
    if best_sim >= CACHE_SIMILARITY_THRESHOLD and best_row:
        return {"answer": best_row[1], "model": f"{best_row[2]}（キャッシュ）", "source": best_row[3], "similarity": best_sim}
    return None

def cache_save(question, answer, model, source):
    """キャッシュに保存"""
    vec = embed(question)
    conn = sqlite3.connect(CACHE_DB)
    conn.execute(
        "INSERT INTO cache (question, answer, model, source, embedding) VALUES (?, ?, ?, ?, ?)",
        (question, answer, model, source, vec)
    )
    conn.commit()
    conn.close()


# ── API呼び出し ──────────────────────────────────────

def search_web(query, max_results=3):
    """DuckDuckGoで検索して結果を返す"""
    if not DDG_AVAILABLE:
        return "検索機能が利用できません"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "検索結果が見つかりませんでした"
        summary = ""
        for i, r in enumerate(results, 1):
            summary += f"{i}. {r.get('title', '')}\n{r.get('body', '')}\n\n"
        return summary.strip()
    except Exception as e:
        log(f"検索エラー: {e}"); return ""

def ask_cloud_with_search(question, messages):
    """DuckDuckGoで検索してからLLMに渡す（会話要約を注入）"""
    # フォローアップ質問はWeb検索をスキップ
    _follow_kws = ['それ', 'これ', 'その', 'あれ', 'もっと', '詳しく', '続き', '具体的', '使用例', '例を挙げ']
    _is_follow = len(question) < 25 and any(k in question for k in _follow_kws)
    if _is_follow:
        search_result = ''
    else:
        search_result = search_web(question)
    # エラーメッセージをフィルタ
    _err_kws = ['エラー', 'Error', 'error', 'Unsupported', 'protocol', 'Exception', 'failed']
    if not search_result or any(k in search_result for k in _err_kws) or len(search_result) < 30:
        search_result = ''
    # Pythonで会話要約を生成してsystemプロンプトに注入
    summary = summarize_history(messages)
    summary_sec = (chr(10) + summary + chr(10)) if summary else ''
    srch_sec = (chr(10) + '【Web検索結果】' + chr(10) + search_result) if search_result else ''
    system = ('あなたは優秀なAIアシスタントです。常に日本語で簡潔に回答してください。' +
             '以下の会話要約を必ず参照し、前の話題を踏まえて回答してください。' +
             summary_sec + srch_sec)
    return call_openrouter(
        MODEL_CLOUD,
        [{"role": "system", "content": system}, {"role": "user", "content": question}],
        max_tokens=600,
        temperature=0.7
    )

def call_openrouter(model, messages, max_tokens=1000, temperature=0.7):
    # 指定モデル + フォールバックモデル一覧
    fallback_models = [
        model,
        "qwen/qwen3-next-80b-a3b-instruct:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "openai/gpt-oss-20b:free",
    ]
    # 重複除去（順序保持）
    seen = set()
    models_to_try = [m for m in fallback_models if not (m in seen or seen.add(m))]

    last_error = None
    for m in models_to_try:
        try:
            r = requests.post(
                OPENROUTER_BASE,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:11437",
                    "X-Title": "Orchestrator v4"
                },
                json={
                    "model": m,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "reasoning": {"exclude": True}
                },
                timeout=30
            )
            data = r.json()
            if "choices" not in data:
                error_msg = data.get("error", {}).get("message", str(data))
                last_error = f"OpenRouter {m} error: {error_msg}"
                continue
            _content = data["choices"][0]["message"]["content"] or ""
            # 一部プロバイダーはexclude指定でも思考過程をcontentに混入させることがあるためガード
            if _content.strip().lower().startswith(("okay,", "ok,", "okay ", "let me", "we need to", "the user is")):
                log(f"⚠️ reasoning混入検知（{m}）→ 次モデルへフォールバック: {_content[:60]}")
                last_error = f"OpenRouter {m}: reasoning混入のため破棄"
                continue
            return _content
        except Exception as e:
            last_error = f"OpenRouter {m} exception: {e}"
            continue
    raise Exception(f"OpenRouter全モデル失敗: {last_error}")



def call_openrouter_single(model, messages, max_tokens=1000, temperature=0.3, response_format=None):
    """自動フォールバックなし・単発モデル呼び出し（パッチ生成の多モデル試行用）
    response_format="json_object" を指定すると、対応モデルでJSONオブジェクト出力を強制する。
    自由文(精密化指示など)を期待する呼び出しでは指定しないこと。"""
    _payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "reasoning": {"exclude": True},
    }
    if response_format == "json_object":
        _payload["response_format"] = {"type": "json_object"}
    r = requests.post(
        OPENROUTER_BASE,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:11437",
            "X-Title": "Orchestrator v4"
        },
        json=_payload,
        timeout=45
    )
    data = r.json()
    if "choices" not in data:
        raise Exception(data.get("error", {}).get("message", str(data)))
    content = data["choices"][0]["message"]["content"] or ""
    if not content.strip():
        raise Exception("空応答")
    return content


def call_vision(text, image_b64, mime_type="image/jpeg"):
    """Gemini Vision APIで画像+テキストを処理（LMM対応）"""
    try:
        from google import genai
        from google.genai import types
        import base64
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            raise Exception("GEMINI_API_KEY not found")
        gclient = genai.Client(api_key=gemini_key)
        img_bytes = base64.b64decode(image_b64)
        for model in ["gemini-2.5-flash", "gemini-2.0-flash"]:
            try:
                response = gclient.models.generate_content(
                    model=model,
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
                        types.Part.from_text(text=text or "この画像について説明してください。")
                    ]
                )
                log(f"\U0001f441 Vision: {model} OK")
                return response.text, model
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    continue
                raise
    except Exception as e:
        log(f"\U0001f441 Vision error: {e}")
        raise Exception(f"Vision API失敗: {e}")

def ask_cloud(messages):
    """クラウドモデル（;プレフィックス用）"""
    system = "あなたは優秀なAIアシスタントです。常に日本語で、丁寧かつ簡潔に回答してください。"
    return call_openrouter(
        MODEL_CLOUD,
        [{"role": "system", "content": system}] + messages,
        max_tokens=1000,
        temperature=0.7
    )


def ask_local(messages):
    print("### ENTER ask_local", flush=True)
    """日本語ローカルモデル（llama.cpp llama-completion / llm-jp-3-1.8B-instruct3, 外部通信なし）"""
    last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    original_user_msg = last_user_msg
    # 直近3ターンの会話履歴をプロンプトに組み込む
    history_lines = []
    for m in messages[-6:]:
        if m["role"] == "user":
            history_lines.append(f"ユーザー: {m['content'][:200]}")
        elif m["role"] == "assistant":
            history_lines.append(f"アシスタント: {m['content'][:200]}")
    # 最後のuser発言は除く（プロンプトに直接入れるため）
    if history_lines and history_lines[-1].startswith("ユーザー:"):
        history_lines = history_lines[:-1]
    summary = summarize_history(messages)
    summary_part = "[会話要約]" + chr(10) + summary + chr(10) + chr(10) if summary else ""
    # 廃止: 旧ローカル即書き込みモード(承認なし・git連携なし)。
    # 編集意図の判定・ルーティングはchat()側でauto_patchフローに統一済みのため、常にFalse固定。
    _edit_mode = False

    # ===== AIエージェントモード =====
    # 「○○を修正して」のような指示だけで対象ファイルを自動で読み込み、
    # 修正後コードを生成→保存→検証まで行う。
    if _edit_mode:
        import re

        m = re.search(r'([A-Za-z0-9_./\\-]+\.py)', last_user_msg)

        if m:
            target = Path(m.group(1))
        else:
            py_files = sorted(
                Path(".").glob("*.py"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            target = py_files[0] if py_files else None

        if target and target.exists():
            source = target.read_text(encoding="utf-8")

            # 大きなファイルは全文を渡さず必要部分だけ渡す
            MAX_CHARS = 12000

            if len(source) > MAX_CHARS:
                import difflib

                keywords = [
                    w for w in last_user_msg.replace("、", " ").replace("。", " ").split()
                    if len(w) >= 3
                ]

                lines = source.splitlines()

                best = 0
                start = 0

                for i, line in enumerate(lines):
                    score = sum(k in line for k in keywords)
                    if score > best:
                        best = score
                        start = max(0, i - 120)

                snippet = "\n".join(lines[start:start + 240])

                source = (
                    "# (巨大ファイルのため抜粋)\n"
                    + snippet
                )

    if _edit_mode:
        prompt = f"""あなたはシニアPythonエンジニアです。

対象ファイル:
{target}

ユーザー要求:
{last_user_msg}

================= 修正対象コード =================

{source}

==================================================

要求を満たすようコードを修正してください。

出力は修正後のPythonコード全文のみ。

説明禁止
Markdown禁止
```禁止
省略禁止
"""
    else:
        prompt = summary_part + last_user_msg
    if _edit_mode:
        system = """あなたはシニアPythonエンジニアです。

ユーザーはコード修正を依頼しています。

ルール:
- 修正後コードを出力
- Markdown禁止
- ```禁止
- 説明禁止
- 不要な文章は禁止
"""
    else:
        system = """あなたは日本語AIアシスタントです。

通常の質問には自然な日本語で回答してください。

正確性を重視し、必要なら手順や理由も説明してください。

Pythonコード修正は、ユーザーが明示的に依頼した場合のみ行ってください。
"""

    if not os.path.exists(LLAMA_COMPLETION_BIN):
        raise Exception(f"llama-completionバイナリが見つかりません: {LLAMA_COMPLETION_BIN}")
    if not os.path.exists(LOCAL_MODEL_PATH):
        raise Exception(f"ローカルモデルファイルが見つかりません: {LOCAL_MODEL_PATH}")

    log(f"🪶 ローカル推論: 質問長={len(last_user_msg)}文字")

    try:
        t0 = time.time()
        print(f"### prompt chars={len(prompt)}")
        print(f"### source chars={len(source) if _edit_mode else 0}")
        print("### llama.cpp 開始")

        result = subprocess.run(
            [
                LLAMA_COMPLETION_BIN,
                "-m", LOCAL_MODEL_PATH,
                "-sys", system,
                "-p", prompt,
                "-n", "512",
                "-c", "4096",
                "--temp", "0.5",
                "-ngl", "0",
                "--no-op-offload",
                "-no-cnv",
            ],
            capture_output=True,
            text=True,
            timeout=None,
            stdin=subprocess.DEVNULL
        )
        print(f"### llama.cpp 終了 {time.time()-t0:.1f} 秒")

    except subprocess.TimeoutExpired as e:
        import traceback
        traceback.print_exc()
        raise Exception(f"llama-completionタイムアウト: {e}")
    except FileNotFoundError:
        raise Exception(f"llama-completionバイナリを実行できません: {LLAMA_COMPLETION_BIN}")

    output = result.stdout or ""
    # 複数フォーマットに対応したパース（jinja/ChatML/Alpaca）
    answer = output
    # ChatML形式: <|im_start|>assistant
    if "<|im_start|>assistant" in answer:
        answer = answer.rsplit("<|im_start|>assistant", 1)[-1]
        answer = answer.lstrip("\n")
        # <|im_end|> で終わる場合は除去
        if "<|im_end|>" in answer:
            answer = answer.split("<|im_end|>")[0]
    # 通常形式: assistant\n
    elif "assistant\n" in answer:
        answer = answer.rsplit("assistant\n", 1)[-1]
    # Alpaca形式: ### 応答:
    elif "### 応答:" in answer:
        answer = answer.rsplit("### 応答:", 1)[-1]
    # 先頭にシステムプロンプトが混入している場合の除去
    for _marker in ["### 指示:", "### 入力:", "<|im_start|>system", "<|im_start|>user"]:
        if answer.startswith(_marker):
            # 応答部分を探して取り出す
            for _end in ["### 応答:", "<|im_start|>assistant", "assistant\n"]:
                if _end in answer:
                    answer = answer.split(_end, 1)[-1].lstrip("\n")
                    break
            break
    answer = answer.split("[end of text]")[0].strip()

    if not answer:
        err = (result.stderr or "")[-2000:]
        raise Exception(f"llama-completion空応答（exit={result.returncode}）: {err}")

    # AIがPythonコードを返した場合の処理
    # 安全のため、明示的な修正依頼(_edit_mode)時のみ反映する
    import re
    import ast

    if not _edit_mode:
        return answer

    # Markdown除去
    m = re.search(r"```(?:python)?\n(.*?)```", answer, re.S)
    if m:
        answer = m.group(1).strip()

    # Pythonコードだけ抽出
    if "def " in answer or "class " in answer or "import " in answer:
        try:
            ast.parse(answer)
            is_python = True
        except Exception:
            is_python = False
    else:
        is_python = False

    # unified diff が返ってきた場合は自動適用
    if answer.lstrip().startswith(("diff --git", "--- ", "*** ")):
        try:
            import tempfile

            with tempfile.NamedTemporaryFile(
                "w",
                suffix=".patch",
                delete=False,
                encoding="utf-8"
            ) as f:
                f.write(answer)
                patch_file = f.name

            r = subprocess.run(
                ["git", "apply", patch_file],
                capture_output=True,
                text=True,
            )

            if r.returncode == 0:
                return "✅ パッチを適用しました。"
            else:
                log(f"git apply失敗: {r.stderr}")

        except Exception as e:
            log(f"patch適用失敗: {e}")

    # ===== AIがJSON形式でコード編集指示を返した場合 =====
    try:
        import json

        data = json.loads(answer)

        if isinstance(data, dict) and "edits" in data:
            target = Path(target)
            text = target.read_text(encoding="utf-8")

            for edit in data["edits"]:
                before = edit.get("before", "")
                after = edit.get("after", "")
                if before:
                    text = text.replace(before, after)

            # 構文チェック
            ast.parse(text)

            import shutil
            import datetime

            backup = (
                str(target)
                + ".bak."
                + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            shutil.copy2(target, backup)

            safe_apply_python_update(target, text)


            r = subprocess.run(
                ["python3", "-m", "py_compile", str(target)],
                capture_output=True,
                text=True
            )

            if r.returncode != 0:
                shutil.copy2(backup, target)
                return (
                    "❌ JSON編集後に構文エラーが発生したためロールバックしました\n\n"
                    + r.stderr
                )

            return f"✅ JSON編集を適用しました\n📦 Backup: {backup}"

    except Exception:
        pass

    if is_python:
        try:
            import re
            m = re.search(r'([A-Za-z0-9_./\\-]+\.py)', original_user_msg)
            if not m:
                py_files = sorted(Path(".").glob("*.py"), key=lambda p: p.stat().st_mtime, reverse=True)
                if py_files:
                    target = str(py_files[0])
                else:
                    raise Exception("Pythonファイルが見つかりません")
            else:
                target = m.group(1)

            import ast
            import shutil
            import datetime

            # Python構文チェック
            ast.parse(answer)

            # バックアップ作成
            backup = (
                target + ".bak."
                + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            shutil.copy2(target, backup)

            # 保存
            safe_apply_python_update(target, answer)


            # Python構文・起動確認
            test = subprocess.run(
                ["python3", "-m", "py_compile", target],
                capture_output=True,
                text=True
            )

            if test.returncode != 0:
                shutil.copy2(backup, target)
                return (
                    "❌ 修正後コードにエラーがあったため自動で元へ戻しました\n\n"
                    + test.stderr
                )

            log(f"🛠 修正を書き込みました: {target}")
            log(f"📦 Backup: {backup}")

            return (
                f"✅ {target} を更新しました\n"
                f"📦 Backup: {backup}"
            )
        except Exception as e:
            log(f"自動書き込み失敗: {e}")

    return answer


def ocr_image(image_path):
    """Tesseract OCRで画像内のテキストを抽出（外部通信なし）"""
    if not os.path.exists(TESSERACT_BIN):
        log(f"🔤 tesseractバイナリが見つかりません: {TESSERACT_BIN}（OCRスキップ）")
        return ""
    if os.path.exists(image_path):
        log(f"🔤 OCR対象ファイル確認OK: {image_path} ({os.path.getsize(image_path)}バイト)")
    else:
        log(f"🔤 OCR対象ファイルが存在しません（呼び出し直前チェック）: {image_path}")
        return ""
    # macOSのTesseract/Leptonicaの既知バグ対策: /tmpはシンボリックリンク(実体は/private/tmp)で、
    # 絶対パスのまま渡すと "image file not found" になることがあるため実パスに解決する
    real_path = os.path.realpath(image_path)
    try:
        result = subprocess.run(
            [TESSERACT_BIN, real_path, "stdout", "-l", OCR_LANGS],
            capture_output=True, timeout=OCR_TIMEOUT,
            stdin=subprocess.DEVNULL
        )
        # text=Trueだと不正バイト混入時にUnicodeDecodeErrorでクラッシュするため、
        # バイト列で受け取ってから安全にデコードする
        raw = result.stdout or b""
        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            err_raw = result.stderr or b""
            err_text = err_raw.decode("utf-8", errors="replace").strip()
            log(f"🔤 OCR空応答の詳細: exit={result.returncode}, stderr={err_text[:300]!r}")
        return text
    except subprocess.TimeoutExpired:
        log(f"🔤 OCRタイムアウト（{OCR_TIMEOUT}秒、スキップ）")
        return ""
    except Exception as e:
        log(f"🔤 OCRエラー（スキップ）: {str(e)[:100]}")
        return ""


def ask_local_vision(ocr_image_path, vision_image_path, question=""):
    """ローカル画像理解（外部通信なし）
    - Tesseract OCR（原寸画像）: 画像内の文字を正確に読み取る
    - llama-mtmd-cli / SmolVLM-500M（リサイズ済み画像）: 人物・風景など全体的な内容を英語で説明
    両者をask_local()(llm-jp-3-1.8B)で統合し、日本語に整形する"""
    if not os.path.exists(LLAMA_MTMD_BIN):
        raise Exception(f"llama-mtmd-cliバイナリが見つかりません: {LLAMA_MTMD_BIN}")
    if not os.path.exists(VISION_MODEL_PATH):
        raise Exception(f"ローカルVisionモデルが見つかりません: {VISION_MODEL_PATH}")
    if not os.path.exists(VISION_MMPROJ_PATH):
        raise Exception(f"mmprojファイルが見つかりません: {VISION_MMPROJ_PATH}")

    ocr_text = ocr_image(ocr_image_path)
    if ocr_text:
        log(f"🔤 OCR抽出完了（{len(ocr_text)}文字）")
    else:
        log(f"🔤 OCR: 文字は検出されませんでした")

    vision_prompt = "Describe this image in detail."
    if question:
        vision_prompt = f"Describe this image in detail, focusing on: {question}"

    log(f"🖼️ ローカルVision推論開始（画像エンコードに時間がかかります）")

    try:
        result = subprocess.run(
            [
                LLAMA_MTMD_BIN,
                "-m", VISION_MODEL_PATH,
                "--mmproj", VISION_MMPROJ_PATH,
                "--image", vision_image_path,
                "-p", vision_prompt,
                "-n", "250",               # 生成トークン上限（複雑な画像で生成が際限なく伸びるのを防ぐ）
                "-ngl", "0",              # Intel Mac: 言語モデル本体はCPUのみ
                "--no-mmproj-offload",    # mmproj(画像エンコーダー)もMetalへオフロードしない
                "--no-warmup",
            ],
            capture_output=True, text=True, timeout=VISION_TIMEOUT,
            stdin=subprocess.DEVNULL
        )
    except subprocess.TimeoutExpired:
        raise Exception(f"llama-mtmd-cliタイムアウト（{VISION_TIMEOUT}秒）")
    except FileNotFoundError:
        raise Exception(f"llama-mtmd-cliバイナリを実行できません: {LLAMA_MTMD_BIN}")

    english_desc = (result.stdout or "").strip()
    if not english_desc and not ocr_text:
        err = (result.stderr or "")[:2000]
        raise Exception(f"llama-mtmd-cli空応答（exit={result.returncode}）: {err}")

    log(f"🖼️ 画像の英語説明取得完了（{len(english_desc)}文字）、日本語に翻訳します")

    # 1.8Bモデルで複数情報源を統合して説明する
    # 「英語の説明文を翻訳するだけ」というシンプルな指示に絞る。
    # OCRテキストは多くの場合すでに日本語かつ正確なので、加工せずそのまま使う。
    ja_desc = english_desc
    if english_desc:
        translate_prompt = f"以下の英語の文章を、自然な日本語に翻訳してください。説明や前置きは不要で、翻訳結果のみを出力してください。\n\n{english_desc}"
        try:
            ja_desc = ask_local([{"role": "user", "content": translate_prompt}])
        except Exception as e:
            log(f"🖼️ 画像説明の翻訳に失敗、英語のまま使います: {str(e)[:100]}")
            ja_desc = english_desc

    parts = []
    if ocr_text:
        parts.append(f"【画像内のテキスト（OCR抽出）】\n{ocr_text}")
    if ja_desc:
        parts.append(f"【画像の説明】\n{ja_desc}")
    if not parts:
        return "画像の内容を読み取れませんでした。"
    return "\n\n".join(parts)


# ── メイン処理 ──────────────────────────────────────

def summarize_history(conversation_history):
    """直近の会話履歴をPythonで構造化要約（LLM不使用・エラー応答を除外）"""
    if not conversation_history:
        return ''
    _err_kws = ['Unsupported', 'protocol', '検索エラー', 'Error', 'Exception', 'エラーが発生']
    pairs = []
    user_msg = None
    for m in conversation_history:
        if m['role'] == 'user':
            user_msg = m['content'][:80]
        elif m['role'] == 'assistant' and user_msg:
            a_content = m['content']
            # エラー応答は要約から除外
            if not any(k in a_content for k in _err_kws):
                pairs.append((user_msg, a_content[:120]))
            user_msg = None
    if not pairs:
        return ''
    lines = ['【これまでの会話要約】']
    for i, (q, a) in enumerate(pairs[-3:]):
        lines.append('Q' + str(i+1) + ': ' + q)
        lines.append('A' + str(i+1) + ': ' + a)
    return chr(10).join(lines)


def _collect_diagnosis_logs(system_key):
    """指定システムのログファイル群から直近N行を収集して結合する"""
    paths = DIAGNOSIS_LOG_MAP.get(system_key, [])
    collected = []
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            tail = lines[-DIAGNOSIS_LOG_TAIL_LINES:]
            if tail:
                collected.append(f"=== {p} (直近{len(tail)}行) ===\n" + "".join(tail))
        except Exception as e:
            collected.append(f"=== {p} ===\n(読み込み失敗: {e})")
    return "\n\n".join(collected)


def _detect_diagnosis_system(instruction):
    """指示文からDIAGNOSIS_LOG_MAPのキーに一致するシステム名を検出"""
    for key in DIAGNOSIS_LOG_MAP:
        if key in instruction:
            return key
    return None


def _build_diagnosis_system_prompt():
    file_list = "\n".join(f"- {f}" for f in auto_patch.ALLOWED_FILES)
    return f"""あなたはシステムのログを分析し、修正すべき箇所を特定するアシスタントです。
与えられたログの内容から、問題を1つ特定し、対応する修正方針を提案してください。

filenameは必ず次の一覧の中から一字一句そのまま選ぶこと(このリスト以外の名前や省略形は使わないこと):
{file_list}

出力は以下のJSON形式のみ。説明文やMarkdownのコードブロック記号は一切含めないこと。
{{"filename": "上記一覧からそのまま選んだファイル名", "instruction": "そのファイルへの具体的な修正指示(日本語、1〜2文)", "reason": "ログのどの部分からその判断をしたかの短い説明"}}
ログに明確な問題が見当たらない場合、または一覧に該当するファイルがない場合は、{{"filename": null, "instruction": null, "reason": "理由"}} を返すこと。
出力の最初の1文字は必ず "{{" にすること。
"""

def _handle_diagnosis_request(system_key, instruction, session_id):
    """システム名をキーにログを収集し、LLMに問題特定＋修正案を提案させ、auto_patchフローに引き継ぐ"""
    log(f"🩻 診断モード開始: system={system_key} / {instruction[:50]}")
    logs = _collect_diagnosis_logs(system_key)
    if not logs.strip():
        return {
            "answer": f"⚠️ {system_key} のログが見つからないか空でした。診断できません。",
            "model": "system", "source": "system"
        }

    messages = [
        {"role": "system", "content": _build_diagnosis_system_prompt()},
        {"role": "user", "content": f"### システム: {system_key}\n\n### ログ内容\n{logs[:8000]}\n\n### 依頼内容\n{instruction}"}
    ]

    diag_errors = []
    diag_result = None
    diag_elapsed = None
    diag_model_used = None
    for model in auto_patch.PATCH_CANDIDATE_MODELS:
        _dt0 = time.time()
        try:
            raw = call_openrouter_single(model, messages, max_tokens=1000, temperature=0.3, response_format="json_object")
        except Exception as e:
            _de = time.time() - _dt0
            diag_errors.append(f"[{model}] API呼び出し失敗({_de:.1f}秒): {e}")
            continue
        try:
            parsed = auto_patch.extract_json_object(raw)
        except Exception as e:
            _de = time.time() - _dt0
            diag_errors.append(f"[{model}] JSON解析失敗({_de:.1f}秒): {e}")
            continue
        if not parsed.get("filename"):
            _de = time.time() - _dt0
            diag_errors.append(f"[{model}] 問題なしと判定({_de:.1f}秒): {parsed.get('reason','(理由なし)')}")
            continue
        if parsed["filename"] not in auto_patch.ALLOWED_FILES:
            _de = time.time() - _dt0
            diag_errors.append(f"[{model}] 提案ファイル'{parsed['filename']}'がホワイトリスト外({_de:.1f}秒)")
            continue
        diag_elapsed = time.time() - _dt0
        diag_model_used = model
        diag_result = parsed
        log(f"🩻 診断成功: model={model} filename={parsed['filename']} elapsed={diag_elapsed:.1f}秒 reason={parsed.get('reason','')[:100]}")
        break

    if not diag_result:
        errs = "\n".join(diag_errors) if diag_errors else "(詳細不明)"
        return {
            "answer": f"❌ {system_key} のログ診断に失敗しました\n{errs[:1000]}",
            "model": "system", "source": "error"
        }

    combined_instruction = f"{diag_result['filename']} {diag_result['instruction']}"
    result_answer = _handle_patch_request(combined_instruction, session_id)
    if isinstance(result_answer, dict) and "answer" in result_answer:
        result_answer["answer"] = f"🩻 診断結果({diag_model_used}, {diag_elapsed:.1f}秒): {diag_result.get('reason','')}\n\n" + result_answer["answer"]
    return result_answer


REFINE_INSTRUCTION_SYSTEM_PROMPT = """あなたはコード修正指示を改善するアシスタントです。
元の修正指示と、それを別のAIに渡した結果失敗した理由が与えられます。
ファイル内容を踏まえて、次回の試行で成功しやすいよう、より具体的で明確な修正指示を1〜3文の日本語で出力してください。
説明や前置きは一切不要、指示文のみを出力すること。
"""

def _refine_patch_instruction(target_filename, original_instruction, last_instruction, error_text):
    """パッチ生成失敗時、失敗理由を踏まえてより具体的な指示に作り直す（精密化）"""
    try:
        with open(auto_patch.ALLOWED_FILES[target_filename], "r", encoding="utf-8") as f:
            file_content = f.read()
    except Exception:
        file_content = ""
    messages = [
        {"role": "system", "content": REFINE_INSTRUCTION_SYSTEM_PROMPT},
        {"role": "user", "content": f"### 対象ファイル(先頭4000文字)\n{file_content[:4000]}\n\n### 元の修正指示\n{original_instruction}\n\n### 直前の指示\n{last_instruction}\n\n### 失敗理由\n{error_text[:2000]}"}
    ]
    def _extract_japanese_instruction(raw):
        """思考過程混入対策: 日本語(ひらがな/カタカナ/漢字)を含む行のみを抽出して結合する。
        日本語が一切含まれない場合は思考過程のみと判断しNoneを返す(フォールバック用)。"""
        import re as _re
        lines = raw.strip().splitlines()
        jp_lines = [ln.strip() for ln in lines if _re.search(r'[ぁ-んァ-ヶ一-龠]', ln)]
        if jp_lines:
            return "".join(jp_lines).strip()
        return None

    for model in auto_patch.PATCH_CANDIDATE_MODELS[:2]:
        try:
            raw = call_openrouter_single(model, messages, max_tokens=800, temperature=0.3)
            refined = _extract_japanese_instruction(raw)
            if refined:
                return refined
        except Exception:
            continue
    return None


def _handle_patch_request(instruction, session_id):
    """「、」「,」プレフィックスによるコード自動修正モード：パッチ生成→承認待ち保存"""
    target_filename = None
    for fname in auto_patch.ALLOWED_FILES:
        if fname in instruction:
            target_filename = fname
            break

    if not target_filename:
        diag_system = _detect_diagnosis_system(instruction)
        if diag_system:
            diag_result = _handle_diagnosis_request(diag_system, instruction, session_id)
            if diag_result is not None:
                return diag_result
        candidates = "\n".join(f"- {f}" for f in auto_patch.ALLOWED_FILES)
        systems = "\n".join(f"- {s}" for s in DIAGNOSIS_LOG_MAP)
        return {
            "answer": (
                f"⚠️ 対象ファイルを特定できませんでした。指示文に次のいずれかのファイル名を含めてください:\n{candidates}\n\n"
                f"または、システム名を含めて「ログを確認して対応して」と依頼することもできます:\n{systems}"
            ),
            "model": "system", "source": "system"
        }

    def _llm_call_single(model, messages):
        return call_openrouter_single(model, messages, max_tokens=8000, temperature=0.3, response_format="json_object")

    MAX_REFINE_RETRIES = 2
    original_instruction = instruction
    current_instruction = instruction
    attempt_log = []
    result = None
    _request_t0 = time.time()

    for attempt in range(MAX_REFINE_RETRIES + 1):
        log(f"🛠 パッチ生成開始(試行{attempt+1}/{MAX_REFINE_RETRIES+1}): {target_filename} / {current_instruction[:50]}")
        _attempt_t0 = time.time()
        result = auto_patch.generate_and_validate_multi(target_filename, current_instruction, _llm_call_single)
        _attempt_elapsed = time.time() - _attempt_t0
        if result["ok"]:
            break
        errs = "\n".join(result["errors"])
        attempt_log.append(f"【試行{attempt+1}({_attempt_elapsed:.1f}秒): {current_instruction[:60]}】\n{errs[:500]}")
        log(f"❌ パッチ生成失敗（全モデル・試行{attempt+1}・{_attempt_elapsed:.1f}秒）: {errs[:500]}")
        if attempt < MAX_REFINE_RETRIES:
            refined = _refine_patch_instruction(target_filename, original_instruction, current_instruction, errs)
            if not refined or refined == current_instruction:
                log("⚠️ 指示の精密化に失敗、または変化なしのためリトライ中断")
                break
            log(f"🔧 指示を精密化: {refined[:80]}")
            current_instruction = refined

    _total_elapsed = time.time() - _request_t0

    if not result["ok"]:
        history = "\n\n".join(attempt_log)
        return {"answer": f"❌ パッチ生成に失敗しました（{target_filename}、合計{_total_elapsed:.1f}秒）\n{MAX_REFINE_RETRIES+1}回試行しましたが全て失敗:\n\n{history[:2000]}", "model": "system", "source": "error"}
    log(f"🛠 パッチ生成成功: model={result['model_used']} elapsed={result.get('elapsed_seconds', 0):.1f}秒 total={_total_elapsed:.1f}秒")

    log(f"🛠 パッチ件数={len(result['patches'])} diff_len={len(result['diff'])}")
    auto_patch.save_pending_patch(
        CACHE_DB, session_id, target_filename, instruction,
        result["patches"], result["old_content"], result["new_content"], result["diff"]
    )
    reasons = "\n".join(f"- {p.get('reason','(理由なし)')}" for p in result["patches"])
    diff_preview = result["diff"][:3000]
    _model_elapsed = result.get("elapsed_seconds", 0)
    answer = (
        f"🛠️ {target_filename} への修正案を生成しました（使用モデル: {result['model_used']}、{_model_elapsed:.1f}秒 / 合計{_total_elapsed:.1f}秒）\n\n"
        f"【変更理由】\n{reasons}\n\n"
        f"【diff】\n```\n{diff_preview}\n```\n\n"
        f"適用する場合は「承認」、取り消す場合は「キャンセル」と送ってください（10分で自動失効）"
    )
    log(f"🛠 パッチ生成成功・承認待ち: {target_filename}")
    return {"answer": answer, "model": "auto_patch", "source": "patch_pending"}


def _apply_pending_patch(session_id, pending):
    """承認された保留パッチをファイルに適用（バックアップ→書き込み→構文再検証）"""
    filename = pending["filename"]
    try:
        backup_path = auto_patch.backup_file(filename)
        auto_patch.write_file(filename, pending["new_content"])
        ok, err = auto_patch.validate_syntax(pending["new_content"])
        if not ok:
            auto_patch.restore_backup(filename, backup_path)
            auto_patch.delete_pending_patch(CACHE_DB, session_id)
            log(f"❌ 適用後の構文チェック失敗、ロールバック: {filename}")
            return {"answer": f"❌ 書き込み後の構文チェックに失敗しロールバックしました\n{err}", "model": "system", "source": "error"}
    except Exception as e:
        log(f"❌ パッチ適用エラー: {filename} / {e}")
        return {"answer": f"❌ 適用中にエラーが発生しました: {e}", "model": "system", "source": "error"}

    auto_patch.delete_pending_patch(CACHE_DB, session_id)
    log(f"✅ パッチ適用完了: {filename} / Backup: {backup_path}")

    note = ""
    if filename != "orchestrator_v4.py":
        commit_msg = f"auto-patch: {filename} - {pending['instruction'][:80]}"
        git_result = auto_patch.git_commit_and_push(auto_patch.ALLOWED_FILES[filename], commit_msg)
        if git_result["skipped"]:
            log(f"ℹ️ git連携スキップ: {git_result['reason']}")
            note += f"\nℹ️ git: {git_result['reason']}"
        elif git_result["ok"]:
            log(f"✅ git commit+push成功: {filename}")
            note += "\n✅ gitへcommit+push済み"
        else:
            log(f"❌ git連携失敗: {git_result['reason']}")
            note += f"\n❌ git連携失敗: {git_result['reason']}"

    if filename == "orchestrator_v4.py":
        try:
            monitor_script = os.path.expanduser("~/ai-orchestrator/monitor_restart.py")
            monitor_log = os.path.expanduser("~/ai-orchestrator/monitor_restart.log")
            meta_path = os.path.expanduser(f"~/ai-orchestrator/.pending_patch_meta_{session_id}.json")
            with open(meta_path, "w", encoding="utf-8") as _metaf:
                json.dump({"filename": filename, "instruction": pending["instruction"]}, _metaf, ensure_ascii=False)
            with open(monitor_log, "a") as _mf:
                subprocess.Popen(
                    ["python3", monitor_script,
                     "--backup", backup_path,
                     "--target", auto_patch.ALLOWED_FILES[filename],
                     "--before-boot", _BOOT_TIME,
                     "--timeout", "30",
                     "--meta", meta_path],
                    stdout=_mf, stderr=_mf, start_new_session=True
                )
            subprocess.Popen(
                ["bash", "-c", "sleep 3 && launchctl kickstart -k gui/$(id -u)/com.fk.orchestrator"],
                start_new_session=True
            )
            log("🔄 自己修正検知: 3秒後に再起動をスケジュール、監視プロセス起動済み")
            note = "\n🔄 オーケストレーター自身の修正のため、3秒後に自動再起動します（ヘルスチェック失敗時は自動ロールバック）。"
        except Exception as e:
            log(f"❌ 再起動スケジュールに失敗: {e}")
            note = f"\n⚠️ 自動再起動のスケジュールに失敗しました。手動で再起動してください: {e}"

    return {
        "answer": f"✅ {filename} を更新しました\n📦 Backup: {backup_path}{note}",
        "model": "system", "source": "patch_applied"
    }


def chat(question, session_id="default"):
    global conversation_histories
    conversation_history = conversation_histories.setdefault(session_id, [])

    # ── コード修正パッチ承認/却下チェック(プレフィックス判定より先に処理) ──
    _pending = auto_patch.get_pending_patch(CACHE_DB, session_id)
    if _pending:
        _stripped = question.lstrip("、,。.").strip()
        if _stripped in ("承認", "はい", "OK", "ok", "Ok", "yes", "YES", "適用", "適用して"):
            return _apply_pending_patch(session_id, _pending)
        elif _stripped in ("キャンセル", "却下", "いいえ", "中止", "no", "NO"):
            auto_patch.delete_pending_patch(CACHE_DB, session_id)
            return {"answer": f"❌ 修正案（{_pending['filename']}）をキャンセルしました。", "model": "system", "source": "system"}

    # プレフィックス判定
    is_patch  = question.startswith("、") or question.startswith(",")
    is_cloud  = (not is_patch) and (question.startswith("。") or question.startswith("."))
    clean_question = question.lstrip("、,。.").strip()

    # ローカルモード(プレフィックスなし)でも編集意図のキーワードがあればauto_patchフローに統一
    # (旧: ask_local()内の_edit_mode即書き込みルートは廃止・無効化済み)
    _edit_kws = ["修正して", "修正してください", "変更して", "変更してください", "改善して", "リファクタリング", "バグを直して", "コードを修正", "対応して", "対応してください", "確認して対応", "ログを確認して"]
    if not is_patch and not is_cloud and any(k in clean_question for k in _edit_kws):
        is_patch = True

    if is_patch:
        return _handle_patch_request(clean_question, session_id)
    prefix = "🌐" if is_cloud else "💬"
    log(f"{prefix} 質問: {clean_question[:50]}")

    # 引き継ぎキーワード処理
    import re as _re_h
    _hm = _re_h.match(r'^引継:([a-fA-F0-9]{8,16})$', clean_question.strip())
    if _hm:
        code = _hm.group(1).lower()
        old_sid = find_session_by_code(code) if len(code) == 8 else code
        if old_sid and old_sid != session_id:
            old_hist = load_conversation_history(MAX_HISTORY, old_sid)
            if old_hist:
                conversation_history.clear()
                conversation_history.extend(old_hist)
                log(f"💬 引継: ...{old_sid[-8:]} → ...{session_id[-8:]} ({len(old_hist)}件)")
                return {"answer": f"✅ 引き継ぎ完了（`{code[:8]}`）\n{len(old_hist)}件の会話を読み込みました。続きから会話できます。", "model": "system", "source": "system"}
        return {"answer": f"⚠️ セッション `{code[:8]}` の履歴が見つかりませんでした。", "model": "system", "source": "system"}

    # DBから会話履歴を読み込み（メモリが空の場合）
    if not conversation_history:
        conversation_history.extend(load_conversation_history(MAX_HISTORY, session_id))
        if conversation_history:
            log(f"💬 会話履歴をDBから復元: {len(conversation_history)}件")

    # 過去の会話から関連する内容を検索
    words = [w for w in clean_question.split() if len(w) > 2][:3]
    past_context = ""
    for word in words:
        past = search_past_conversations(word, limit=2)
        for role, content, created_at in past:
            if content != clean_question:
                past_context += f"[{created_at[:10]} {role}]: {content[:100]}\n"
    if past_context:
        log(f"💬 過去の関連会話を発見")

    # past_contextをget_agent_contextに統合
    if past_context:
        original_get = get_agent_context
        def get_agent_context_with_past(q, max_comments=3):
            ctx = original_get(q, max_comments)
            ctx += f"\n[過去の関連会話]\n{past_context[:500]}"
            return ctx

    save_conversation("user", clean_question, session_id)
    conversation_history.append({"role": "user", "content": clean_question})
    if len(conversation_history) > MAX_HISTORY * 2:
        del conversation_history[:-MAX_HISTORY * 2]

    # 文脈依存質問（それ/これ/もっと/詳しく等の短い質問）はキャッシュ検索自体をスキップ
    # 無関係な過去キャッシュに誤ヒットして会話の流れが途切れるのを防ぐ
    _ctx_dep_kws = ['それ', 'これ', 'その', 'あれ', 'もっと', '詳しく', '続き', '具体的', '使用例', '例を挙げ']
    _is_context_dependent = len(clean_question) < 25 and any(k in clean_question for k in _ctx_dep_kws)

    # キャッシュ検索（クラウドプレフィックス時はスキップ）
    if not is_cloud and not _is_context_dependent:
        cache_hit = cache_search(clean_question)
        if cache_hit:
            answer = cache_hit["answer"]
            log(f"💾 キャッシュヒット（類似度{round(cache_hit['similarity']*100,1)}%）")
            model_name = cache_hit["model"]
            conversation_history.append({"role": "assistant", "content": answer})
            return {
                "answer": answer,
                "model": model_name,
                "source": "cache",
                "similarity": round(cache_hit["similarity"] * 100, 1)
            }

    # キャッシュミス → API呼び出し
    try:
        if is_cloud:
            answer = ask_cloud_with_search(clean_question, conversation_history[-20:])
            model_name = "Llama-3.3-70B（Web検索）"
            source = "cloud"
            log(f"✅ 回答: {model_name}")
        else:
            answer = ask_local(conversation_history[-20:])
            model_name = "llm-jp-3-1.8B（ローカル/llama.cpp, 外部通信なし）"
            source = "local"
            log(f"✅ 回答: {model_name}")
    except Exception as e:
        if not is_cloud:
            # 完全ローカル厳守：通常検索（プレフィックスなし）は外部に一切フォールバックしない
            answer = f"ローカル推論に失敗しました（外部通信は行いません）: {e}"
            model_name = "エラー（ローカル推論失敗）"
            source = "error"
            log(f"❌ ローカル推論エラー: {str(e)[:100]}")
        else:
            log(f"⚠️ ask_cloud_with_search失敗 → フォールバック: {str(e)[:200]}")
            try:
                # Groqにフォールバック（；クラウドプレフィックス時のみ。Web検索結果も渡す）
                if DDG_AVAILABLE:
                    search_result = search_web(clean_question)
                    agent_context = get_agent_context(clean_question)
                    log(f"🧠 Web検索+context取得: {len(agent_context)}文字")
                    system_msg = f"あなたは優秀なAIアシスタントです。常に日本語で回答してください。\nWeb検索結果:\n{search_result}"
                else:
                    agent_context = get_agent_context(clean_question)
                    log(f"🧠 通常+context取得: {len(agent_context)}文字")
                    system_msg = "あなたは優秀なAIアシスタントです。常に日本語で回答してください。"
                groq_messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": clean_question}]
                r = requests.post(GROQ_BASE,
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                    json={"model": GROQ_MODEL, "messages": groq_messages, "max_tokens": 1000},
                    timeout=30)
                _data = r.json()
                if "choices" not in _data:
                    raise Exception(f"Groq error: {_data.get('error', {}).get('message', str(_data))}")
                answer = _data["choices"][0]["message"]["content"]
                model_name = "Groq llama-3.3-70B（Web検索）"
                source = "fallback"
                log(f"✅ フォールバック: {model_name}")
            except Exception as e2:
                # Groqレート制限時はOpenRouterで再フォールバック
                if "Rate limit" in str(e2) or "rate_limit" in str(e2).lower() or "TPD" in str(e2) or "tokens per day" in str(e2).lower():
                    log(f"⚠️ Groqレート制限検知 → OpenRouterで再試行")
                    or_fallback_models = [
                        "nvidia/nemotron-3-super-120b-a12b:free",
                        "openai/gpt-oss-20b:free",
                        "qwen/qwen3-next-80b-a3b-instruct:free",
                    ]
                    answer = None
                    for or_model in or_fallback_models:
                        try:
                            or_r = requests.post(OPENROUTER_BASE,
                                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
                                json={"model": or_model, "messages": groq_messages, "max_tokens": 1000},
                                timeout=30)
                            or_data = or_r.json()
                            if "choices" in or_data:
                                answer = or_data["choices"][0]["message"]["content"]
                                model_name = f"OpenRouter {or_model}（Groqレート制限フォールバック）"
                                source = "fallback"
                                log(f"✅ Groqレート制限→OpenRouterフォールバック成功: {or_model}")
                                break
                        except Exception:
                            continue
                    if answer is None:
                        answer = f"エラーが発生しました: {e2}"
                        model_name = "エラー"
                        source = "error"
                        log(f"❌ Groq・OpenRouter全モデル失敗")
                else:
                    answer = f"エラーが発生しました: {e2}"
                    model_name = "エラー"
                    source = "error"
                    log(f"❌ エラー発生")

    # キャッシュ保存（通常モードはキャッシュ参照のみで登録はしない）
    if source not in ("error", "local") and not _is_context_dependent:
        cache_save(clean_question, answer, model_name, source)
    elif _is_context_dependent:
        log(f"💾 文脈依存質問のためキャッシュ保存スキップ: {clean_question[:30]}")

    # エラー応答はDBに保存しない
    _save_err_kws = ['Unsupported', 'protocol', '検索エラー', 'ローカル推論に失敗']
    if not any(k in answer for k in _save_err_kws):
        save_conversation("assistant", answer, session_id)
        conversation_history.append({"role": "assistant", "content": answer})
    else:
        log(f'⚠️ エラー応答はDBに保存しません')
    return {"answer": answer, "model": model_name, "source": source}


# ── WebUI ──────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🐈 オーケストレーター</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eee; height: 100vh; display: flex; flex-direction: column; }
header { background: #16213e; padding: 12px 16px; font-size: 18px; font-weight: bold; border-bottom: 1px solid #333; display: flex; justify-content: space-between; align-items: center; }
#chat { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
.msg { max-width: 85%; padding: 10px 14px; border-radius: 16px; line-height: 1.5; white-space: pre-wrap; word-break: break-word; }
.user { background: #0f3460; align-self: flex-end; border-bottom-right-radius: 4px; }
.ai { background: #1a1a3e; align-self: flex-start; border-bottom-left-radius: 4px; border: 1px solid #333; }
.model-tag { font-size: 11px; color: #888; margin-top: 6px; }
.msg-timestamp { font-size: 10px; color: #666; margin-top: 4px; }
.cache-tag { color: #4caf50; }
.cloud-tag { color: #2196f3; }
#input-area { display: flex; gap: 8px; padding: 12px; background: #16213e; border-top: 1px solid #333; align-items: flex-end; }
#msg-input { flex: 1; background: #0f0f23; border: 1px solid #444; border-radius: 12px; padding: 10px 16px; color: #eee; font-size: 16px; outline: none; resize: none; min-height: 44px; max-height: 200px; line-height: 1.4; }
#send-btn { background: #e94560; border: none; border-radius: 50%; width: 44px; height: 44px; color: white; font-size: 20px; cursor: pointer; flex-shrink: 0; }
.hint { font-size: 11px; color: #666; padding: 4px 16px; background: #16213e; }
.thinking { color: #888; font-style: italic; }
</style>
</head>
<body>
<header>
  🐈 オーケストレーター
  <div style="display:flex;gap:8px;align-items:center;">
    <a href="/help" style="background:#1a3a5c;color:#4caf50;padding:4px 10px;border-radius:8px;font-size:12px;text-decoration:none;" target="_blank">❓ ヘルプ</a>
    <button onclick="clearHistory()" style="background:#3a1a1a;color:#e94560;border:none;padding:4px 10px;border-radius:8px;font-size:12px;cursor:pointer;">🗑️ 履歴クリア</button>
  </div>
</header>
<div class="hint">💡 <strong>。</strong>クラウド ｜ <a href="https://www.moltbook.com/u/fujikatsu-openclaw" target="_blank" style="color:#fa0;">🦞 Moltbook</a> ｜ <a href="/captcha/stats" style="color:#4caf50" target="_blank">🧩 CAPTCHA</a> ｜ <a href="/dreaming/stats" style="color:#9c27b0" target="_blank">🌙 Dreaming</a> ｜ <a href="https://hz-k-2mba14.tailb82610.ts.net:5000/rescue" target="_blank" style="color:#f44;">🛡️ MythoFable</a></div>
<div id="chat"></div>
<div id="input-area">
  <label id="img-btn" title="画像・ファイルを添付" style="cursor:pointer;background:#1a3a5c;border:none;border-radius:50%;width:44px;height:44px;color:#4caf50;font-size:20px;display:flex;align-items:center;justify-content:center;flex-shrink:0;">📎<input type="file" id="img-input" accept="image/*,.log,.txt,.py,.js,.ts,.json,.md,.sh,.yaml,.yml,.csv,.html,.css,.xml,.conf,.ini,.env" style="display:none" onchange="previewFile(this)"></label>
  <div style="flex:1;display:flex;flex-direction:column;gap:4px;">
    <div id="img-preview" style="display:none;position:relative;align-items:center;gap:8px;background:#16213e;border:1px solid #444;border-radius:8px;padding:6px 10px;"><img id="preview-img" style="display:none;max-height:80px;border-radius:8px;"><span id="preview-filename" style="display:none;color:#0ff;font-size:12px;">📄 <span id="preview-fname-text"></span> <span id="preview-fsize" style="color:#888;"></span></span><button onclick="clearImage()" style="background:#e94560;border:none;border-radius:50%;width:20px;height:20px;color:white;cursor:pointer;font-size:12px;flex-shrink:0;">✕</button></div>
    <textarea id="msg-input" placeholder="。クラウド（Web検索あり）/ 通常はそのまま入力..." rows="1"></textarea>
  </div>
  <button id="send-btn" onclick="sendMsg()">↑</button>
</div>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('msg-input');

input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = input.scrollHeight + 'px';
});

input.addEventListener('keydown', (e) => {
    if (e.isComposing || e.keyCode === 229) return;
    if (e.key === 'Enter' && e.shiftKey) {
        e.preventDefault();
        sendMsg();
    }
});

function addMsg(text, role, model='', source='') {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = text;
  if (model) {
    const tag = document.createElement('div');
    tag.className = 'model-tag';
    let icon = '🤖';
    if (source === 'cache') { icon = '💾'; tag.classList.add('cache-tag'); }
    else if (source === 'cloud') { icon = '🌐'; tag.classList.add('cloud-tag'); }
    tag.textContent = icon + ' ' + model;
    div.appendChild(tag);
  }
  if (!role.includes('thinking')) {
    const ts = document.createElement('div');
    ts.className = 'msg-timestamp';
    const now = new Date();
    ts.textContent = now.toLocaleTimeString('ja-JP', {hour: '2-digit', minute: '2-digit', second: '2-digit'});
    div.appendChild(ts);
  }
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

let _imgB64 = '';
let _imgMime = 'image/jpeg';
let _fileText = '';
let _fileName = '';

const TEXT_EXT = ['.log','.txt','.py','.js','.ts','.json','.md','.sh','.yaml','.yml','.csv','.html','.css','.xml','.conf','.ini','.env'];

function previewFile(inp) {
  const file = inp.files[0];
  if (!file) return;
  const isImage = file.type.startsWith('image/');
  const isText = TEXT_EXT.some(ext => file.name.toLowerCase().endsWith(ext)) || file.type.startsWith('text/');

  if (isImage) {
    _imgMime = file.type || 'image/jpeg';
    _fileText = ''; _fileName = '';
    const reader = new FileReader();
    reader.onload = e => {
      _imgB64 = e.target.result.split(',')[1];
      document.getElementById('preview-img').style.display = 'block';
      document.getElementById('preview-img').src = e.target.result;
      document.getElementById('preview-filename').style.display = 'none';
      document.getElementById('img-preview').style.display = 'inline-flex';
    };
    reader.readAsDataURL(file);
  } else if (isText) {
    _imgB64 = '';
    const reader = new FileReader();
    reader.onload = e => {
      _fileText = e.target.result;
      _fileName = file.name;
      document.getElementById('preview-img').style.display = 'none';
      document.getElementById('preview-fname-text').textContent = file.name;
      document.getElementById('preview-fsize').textContent = `(${(file.size/1024).toFixed(1)}KB)`;
      document.getElementById('preview-filename').style.display = 'inline';
      document.getElementById('img-preview').style.display = 'inline-flex';
    };
    reader.readAsText(file);
  } else {
    alert('対応していないファイル形式です（画像、または .log/.txt/.py 等のテキストファイルを選択してください）');
    inp.value = '';
  }
}

function clearImage() {
  _imgB64 = '';
  _fileText = '';
  _fileName = '';
  document.getElementById('img-preview').style.display = 'none';
  document.getElementById('img-input').value = '';
}

// 履歴クリア
function clearHistory() {
  if (!confirm('この会話の履歴をクリアしますか？')) return;
  fetch('/session/clear', {
    method: 'POST',
    headers: {'Content-Type': 'application/json', 'X-Auth-Token': AUTH_TOKEN},
    body: JSON.stringify({session_id: getSessionId()})
  }).then(r => r.json()).then(d => {
    document.getElementById('messages').innerHTML = '';
    addMessage('system', '✅ 会話履歴をクリアしました。');
  }).catch(e => alert('クリア失敗: ' + e));
}

// セッションID管理
  function getSessionId() {
    if (!sessionStorage.getItem('orc_sid')) {
      const a = new Uint8Array(8);
      crypto.getRandomValues(a);
      sessionStorage.setItem('orc_sid', Array.from(a).map(b=>b.toString(16).padStart(2,'0')).join(''));
    }
    return sessionStorage.getItem('orc_sid');
  }
  function getSessionCode() { return getSessionId().slice(-8); }
  window.addEventListener('DOMContentLoaded', function() {
    const hint = document.querySelector('.hint');
    if (hint) {
      const sp = document.createElement('span');
      sp.style.cssText = 'margin-left:10px;color:#888;font-size:11px;';
      sp.innerHTML = '🔑 <code style="background:#0f3460;padding:1px 6px;border-radius:3px;color:#9c27b0;letter-spacing:1px;" title="引継:'+getSessionCode()+'と入力で引き継ぎ">'+getSessionCode()+'</code>';
      hint.appendChild(sp);
    }
  });

  async function sendMsg() {
  const text = input.value.trim();
  if (!text && !_imgB64 && !_fileText) return;
  let displayText = text;
  if (_imgB64) displayText = text ? `📷 ${text}` : '📷 画像を送信';
  else if (_fileText) displayText = text ? `📄 ${_fileName}: ${text}` : `📄 ${_fileName} を送信`;
  input.value = '';
  input.style.height = 'auto';
  addMsg(displayText, 'user');
  const imgB64 = _imgB64;
  const imgMime = _imgMime;
  const fileText = _fileText;
  const fileName = _fileName;
  clearImage();
  const thinking = addMsg('考え中... 0秒', 'ai thinking');
  const _thinkStart = Date.now();
  const _thinkTimer = setInterval(() => {
    const secs = Math.floor((Date.now() - _thinkStart) / 1000);
    thinking.firstChild ? (thinking.childNodes[0].textContent = `考え中... ${secs}秒`) : (thinking.textContent = `考え中... ${secs}秒`);
  }, 1000);
  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-Token': '{{ token }}'},
      body: JSON.stringify({message: text, image: imgB64, mime_type: imgMime, file_text: fileText, file_name: fileName, session_id: getSessionId()})
    });
    const data = await res.json();
    clearInterval(_thinkTimer);
    thinking.remove();
    addMsg(data.answer, 'ai', data.model, data.source);
  } catch(e) {
    clearInterval(_thinkTimer);
    thinking.textContent = 'エラーが発生しました';
  }
}

async function clearHistory() {
  await fetch('/clear', {method: 'POST', headers: {'X-Token': '{{ token }}'}});
  chat.innerHTML = '';
  addMsg('履歴をクリアしました', 'ai');
}
</script>
</body>
</html>"""


# ── Flask ──────────────────────────────────────

TOKEN = os.environ.get("ORC_TOKEN", "REMOVED-TOKEN")
WEB_PASSWORD = os.environ.get("ORC_TOKEN")
if not WEB_PASSWORD:
    raise SystemExit("ORC_TOKEN が .env に設定されていません。~/.config/ai-keys/.env に ORC_TOKEN=... を追加してください。")
import secrets as _secrets
SESSION_TOKENS = set()  # 有効なセッショントークン

def check_web_auth():
    """WebUIのCookie認証チェック"""
    token = request.cookies.get("web_session", "")
    log(f"🔐 check_web_auth: token={token[:8] if token else 'None'}")
    return token in SESSION_TOKENS

LOGIN_HTML = '''<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🐈 ログイン</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eee; height: 100vh; display: flex; align-items: center; justify-content: center; }
.box { background: #16213e; border-radius: 16px; padding: 32px; width: 300px; }
h1 { font-size: 20px; margin-bottom: 24px; text-align: center; }
input { width: 100%; background: #0f0f23; border: 1px solid #444; border-radius: 8px; padding: 12px; color: #eee; font-size: 16px; margin-bottom: 16px; outline: none; }
button { width: 100%; background: #e94560; border: none; border-radius: 8px; padding: 12px; color: white; font-size: 16px; cursor: pointer; }
.error { color: #e94560; font-size: 13px; margin-bottom: 12px; text-align: center; }
</style>
</head>
<body>
<div class="box">
  <h1>🐈 オーケストレーター</h1>
  {% if error %}<div class="error">{{ error }}</div>{% endif %}
  <form method="POST" action="/login">
    <input type="password" name="password" placeholder="パスワード" autofocus>
    <button type="submit">ログイン</button>
  </form>
</div>
</body>
</html>'''

def check_auth():
    token = request.headers.get("X-Token", "")
    return token == TOKEN

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        password = request.form.get('password', '')
        if password == WEB_PASSWORD:
            token = _secrets.token_hex(16)
            SESSION_TOKENS.add(token)
            resp = redirect('/')
            resp.set_cookie('web_session', token, max_age=86400*7)
            return resp
        return render_template_string(LOGIN_HTML, error='パスワードが違います')
    return render_template_string(LOGIN_HTML, error=None)

@app.route('/logout')
def logout():
    token = request.cookies.get('web_session', '')
    SESSION_TOKENS.discard(token)
    resp = redirect('/login')
    resp.delete_cookie('web_session')
    return resp

@app.route('/')
def index():
    if not check_web_auth():
        return redirect('/login')
    return render_template_string(HTML, token=TOKEN)

MAX_FILE_CHARS = 8000

def trim_file_text(text, head=4000, tail=4000):
    if len(text) <= MAX_FILE_CHARS:
        return text
    return text[:head] + f"\n...[中略: 全{len(text)}文字中 先頭{head}+末尾{tail}文字のみ表示]...\n" + text[-tail:]

@app.route('/chat', methods=['POST'])
def chat_api():
    if not check_auth():
        return jsonify({"error": "Unauthorized"}), 401
    data = request.json
    question = data.get('message', '').strip()
    image_b64 = data.get('image', '')
    mime_type = data.get('mime_type', 'image/jpeg')
    file_text = data.get('file_text', '')
    file_name = data.get('file_name', '')
    req_session_id = data.get('session_id', 'default')

    if image_b64:
        img_is_cloud = question.startswith("。") or question.startswith(".")
        img_clean_q = question.lstrip(".。").strip()

        if not img_is_cloud:
            # 通常モード: OCR(原寸)+ローカルVision（リサイズ済み）を使用、外部通信なし
            import tempfile
            import base64 as _b64
            ext = ".png" if "png" in (mime_type or "") else ".jpg"
            tmp_path = None
            resized_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False, dir="/tmp") as tmp:
                    tmp.write(_b64.b64decode(image_b64))
                    tmp_path = tmp.name  # OCR用（原寸のまま、文字の判読精度を優先）
                # SmolVLM用に長辺512pxへ縮小したコピーを別途作成（処理時間短縮のため）
                # OCRは原寸画像に対して行うため、上書きせず別ファイルに出力する
                resized_path = tmp_path + "_resized" + ext
                try:
                    r = subprocess.run(["sips", "-Z", "512", tmp_path, "--out", resized_path],
                                        capture_output=True, text=True, timeout=15)
                    if r.returncode != 0 or not os.path.exists(resized_path):
                        raise Exception((r.stderr or "")[:100])
                except Exception as e:
                    log(f"🖼️ 画像リサイズに失敗（元サイズのままVisionに渡します）: {str(e)[:100]}")
                    resized_path = tmp_path
                answer = ask_local_vision(tmp_path, resized_path, img_clean_q)
                save_conversation("user", f"[画像] {img_clean_q}")
                save_conversation("assistant", answer)
                return jsonify({
                    "answer": answer,
                    "model": "🖼️ OCR+SmolVLM+llm-jp-3-1.8B（ローカル/llama.cpp, 外部通信なし）",
                    "source": "local_vision"
                })
            except Exception as e:
                log(f"❌ ローカルVisionエラー: {str(e)[:150]}")
                return jsonify({
                    "answer": f"ローカルVision推論に失敗しました（外部通信は行いません）: {e}",
                    "model": "エラー（ローカルVision失敗）",
                    "source": "error"
                })
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
                if resized_path and resized_path != tmp_path and os.path.exists(resized_path):
                    os.remove(resized_path)

        try:
            vision_question = img_clean_q or "この画像について説明してください。"
            if DDG_AVAILABLE and img_clean_q:
                search_result = search_web(img_clean_q)
                vision_question = f"{vision_question}\n\nWeb search results (use if relevant):\n{search_result}"
            answer, model_name = call_vision(vision_question, image_b64, mime_type)
            save_conversation("user", f"[画像] {img_clean_q}")
            save_conversation("assistant", answer)
            full_model_name = f"\U0001f441 {model_name}(Vision)"
            return jsonify({"answer": answer, "model": full_model_name, "source": "vision"})
        except Exception as e:
            return jsonify({"answer": f"Vision APIエラー: {e}", "model": "error", "source": "error"})

    if file_text:
        trimmed = trim_file_text(file_text)
        # 元のメッセージのプレフィックス（。/。。。）を保持し、結合後も判定できるようにする
        prefix_str = ""
        q_rest = question
        if q_rest.startswith("。") or q_rest.startswith("."):
            prefix_str = "。"
            q_rest = q_rest.lstrip(".。").strip()
        composed_question = (
            f"{prefix_str}以下は添付ファイル「{file_name}」の内容です:\n"
            f"```\n{trimmed}\n```\n\n"
            f"{q_rest or 'このファイルの内容を確認し、要約や問題点を教えてください。'}"
        )
        log(f"\U0001f4c4 ファイル添付: {file_name} ({len(file_text)}文字 -> {len(trimmed)}文字), prefix={prefix_str or '(なし)'}")
        result = chat(composed_question, req_session_id)
        result["model"] = (f"\U0001f4c4 " + str(result.get("model",""))).strip()
        return jsonify(result)

    if not question:
        return jsonify({"error": "空のメッセージ"}), 400
    result = chat(question, req_session_id)
    return jsonify(result)

@app.route('/cache/stats', methods=['GET'])
def cache_stats():
    conn = sqlite3.connect(CACHE_DB)
    count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
    recent = conn.execute("SELECT question, model, source, created_at FROM cache ORDER BY id DESC LIMIT 5").fetchall()
    conn.close()
    return jsonify({
        "total": count,
        "recent": [{"question": r[0][:50], "model": r[1], "source": r[2], "created_at": r[3]} for r in recent]
    })

@app.route('/help')
def help_page():
    HTML_HELP = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>❓ ヘルプ - オーケストレーター</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }
h1 { font-size: 20px; color: #4caf50; margin-bottom: 16px; }
h2 { font-size: 15px; color: #2196f3; margin: 20px 0 10px; border-bottom: 1px solid #333; padding-bottom: 6px; }
table { width: 100%; border-collapse: collapse; margin-bottom: 16px; font-size: 13px; }
th { background: #16213e; color: #0ff; padding: 8px; text-align: left; }
td { padding: 8px; border-bottom: 1px solid #222; vertical-align: top; }
td:first-child { font-weight: bold; color: #fa0; white-space: nowrap; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin: 1px; }
.yes { background: #1a3a1a; color: #4caf50; border: 1px solid #4caf50; }
.no  { background: #2a1a1a; color: #888; border: 1px solid #444; }
.note { font-size: 12px; color: #888; margin-top: 8px; }
a.back { display: inline-block; margin-top: 20px; color: #4caf50; text-decoration: none; font-size: 13px; }
</style>
</head>
<body>
<h1>❓ オーケストレーター ヘルプ</h1>

<h2>🔤 プレフィックス一覧</h2>
<table>
<tr><th>プレフィックス</th><th>モード</th><th>キャッシュ</th><th>Web検索</th><th>LLM</th></tr>
<tr><td>（なし）</td><td>通常</td><td><span class="badge yes">✅ 使う</span></td><td><span class="badge no">❌</span></td><td>ローカルLLM（llm-jp-3-1.8B, llama.cpp, 外部通信なし）</td></tr>
<tr><td>。</td><td>クラウド</td><td><span class="badge no">❌</span></td><td><span class="badge yes">✅ DDG</span></td><td>OpenRouter（高精度）</td></tr>
<tr><td>、 または ,</td><td>コード自動修正</td><td><span class="badge no">❌</span></td><td><span class="badge no">❌</span></td><td>OpenRouter（複数モデル自動フォールバック）</td></tr>

</table>

<h2>💡 使い分けの目安</h2>
<table>
<tr><th>シーン</th><th>おすすめ</th></tr>
<tr><td>普通の質問</td><td>（なし）— キャッシュヒット時は瞬時、未ヒット時はローカルLLM（外部通信なし）</td></tr>
</table>

<h2>🛠️ コード自動修正＋git連携</h2>
<table>
<tr><th>ステップ</th><th>内容</th></tr>
<tr><td>①指示</td><td><code>、</code>/<code>,</code>で始めるか、通常モードで「修正して」「対応して」等のキーワードを含めて依頼</td></tr>
<tr><td>②パッチ生成</td><td>複数の無料モデルを順に試行（成功するまで自動フォールバック）。対象ファイルは事前登録されたホワイトリストのみ</td></tr>
<tr><td>③diff確認</td><td>変更差分が提示される。この時点ではファイルは書き換わらない</td></tr>
<tr><td>④承認/却下</td><td>「承認」「はい」「OK」等で適用、「キャンセル」「却下」等で取り消し（10分で自動失効）</td></tr>
<tr><td>⑤適用</td><td>バックアップ作成→書き込み→構文再検証。オーケストレーター自身のファイルの場合は自動再起動＋ヘルスチェック（失敗時は自動ロールバック）</td></tr>
<tr><td>⑥git連携</td><td>適用成功（自己修正の場合はヘルスチェック成功）後、対象がgitリポジトリならcommit+push。リポジトリでなければスキップ</td></tr>
</table>

<h3 style="font-size:13px;color:#9c27b0;margin:14px 0 6px;">📋 対象ファイル（ホワイトリスト）</h3>
<table>
<tr><th>システム</th><th>ファイル</th></tr>
<tr><td>オーケストレーター</td><td>orchestrator_v4.py</td></tr>
<tr><td>Moltbook</td><td>agent_claude.py / agent_log_doctor.py</td></tr>
<tr><td>MythoFable</td><td>dashboard.py / log_watcher.py / auto_patcher.py / auto_recovery.py / exit_node_monitor.py / ip_manager.py / mythofable_s.py / proxy_watcher.py</td></tr>
</table>

<h3 style="font-size:13px;color:#9c27b0;margin:14px 0 6px;">🩻 診断モード（ログから自動で問題特定）</h3>
<p class="note">対象ファイル名が分からない場合、システム名を含めて「〇〇のログを確認して対応して」と依頼すると、直近のログを自動収集しLLMが問題箇所と修正方針を提案します（その後は通常のパッチ生成フローに合流）。</p>
<table>
<tr><th>システム名の例</th><th>参照ログ</th></tr>
<tr><td>オーケストレーター / orchestrator</td><td>起動エラーログ・死活監視ログ</td></tr>
<tr><td>moltbook / エージェント</td><td>agent_claude.log</td></tr>
<tr><td>mythofable / セキュリティ</td><td>watcher/dashboardのログ</td></tr>
</table>
<p class="note" style="margin-top:8px;">💡 例：「moltbookのログを確認して対応して」「MythoFableのログを確認して対応して」</p>

<h2>🔑 セッションと会話の引き継ぎ</h2>
<table>
<tr><th>項目</th><th>説明</th></tr>
<tr><td>セッションID</td><td>ブラウザを開くと自動生成される16文字のID。画面下部のヒント行に <strong>🔑 XXXXXXXX</strong>（末尾8文字）として表示されます</td></tr>
<tr><td>会話の継続</td><td>同じブラウザ・タブであれば自動的に同じセッションで会話が続きます</td></tr>
<tr><td>別デバイスへの引き継ぎ</td><td>引き継ぎたいセッションのコード（例: <code>91eee1f9</code>）を確認し、新しいブラウザで <code>引継:91eee1f9</code> と入力すると過去の会話履歴が読み込まれます</td></tr>
<tr><td>コードの確認方法</td><td>画面下部ヒント行の <strong>🔑 XXXXXXXX</strong> の8文字がそのセッションの引き継ぎコードです</td></tr>
</table>
<p class="note" style="margin-top:8px;">💡 引き継ぎ例：iPhoneから「<code>引継:91eee1f9</code>」と送信 → PCのセッション履歴を引き継いで続きから会話できます</p>

<h2>⚠️ 注意</h2>
<p class="note">• Groq TPD上限（100,000トークン/日）に達した場合はOpenRouterへ自動フォールバック</p>
<p class="note">• 通常検索（なし）のローカルLLMはllama.cpp推論に失敗しても外部へフォールバックしません（完全ローカル厳守）</p>

<a class="back" href="/">← チャットに戻る</a>
</body>
</html>"""
    return HTML_HELP

@app.route('/dreaming/stats', methods=['GET'])
def dreaming_stats():
    if not check_web_auth():
        return redirect('/login')
    import json as _json, os
    memory_path = os.path.expanduser("~/ai-agent/moltbook/memory.json")
    try:
        m = _json.load(open(memory_path))
    except Exception:
        return jsonify({"error": "memory.json not found"})

    style_notes = m.get("style_notes", "データなし")
    avoid_topics = m.get("avoid_topics", [])
    last_insights = m.get("last_insights", m.get("insights", "データなし"))
    last_dream = m.get("last_dream", "不明")
    karma_history = m.get("karma_history", [])
    karma_up_triggers = m.get("karma_up_triggers", [])
    commented_topics = m.get("commented_topics", [])[-10:]
    karma_labels = [k["time"][-5:] for k in karma_history[-10:]]
    karma_values = [k["karma"] for k in karma_history[-10:]]

    avoid_html = "".join('<span class="tag">' + t + '</span>' for t in avoid_topics) if avoid_topics else "<p>なし</p>"
    
    if karma_values and len(karma_values) > 1:
        min_k = min(karma_values)
        max_k = max(karma_values)
        karma_bar_html = ""
        for k, l in zip(karma_values, karma_labels):
            h = int((k - min_k) / (max_k - min_k + 1) * 70) + 10
            karma_bar_html += '<div style="flex:1;display:flex;flex-direction:column;align-items:center"><div class="karma-col" style="height:' + str(h) + 'px" data-label="karma: ' + str(k) + '"></div><div class="karma-label">' + l + '</div></div>'
    else:
        karma_bar_html = "データなし"

    trigger_html = "".join('<div class="trigger">+' + str(t["karma_after"] - t["karma_before"]) + 'pt | ' + t["time"] + ' | ' + t.get("last_topic", "")[:40] + '</div>' for t in karma_up_triggers[-5:]) if karma_up_triggers else "<div class=\'card\'><p>データなし</p></div>"
    topic_html = "".join('<div class="topic">' + t["time"] + ' | ' + t["title"][:50] + '</div>' for t in reversed(commented_topics)) if commented_topics else "<div class=\'card\'><p>データなし</p></div>"

    html = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🌙 Dreaming Stats</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }
h1 { font-size: 22px; margin-bottom: 20px; color: #9c27b0; }
h2 { font-size: 16px; margin: 20px 0 10px; color: #aaa; }
.card { background: #16213e; border-radius: 12px; padding: 16px; margin-bottom: 12px; }
.card p { font-size: 14px; line-height: 1.6; color: #ddd; }
.tag { display: inline-block; background: #0f3460; border-radius: 8px; padding: 4px 10px; margin: 4px; font-size: 12px; }
.karma-bar { display: flex; align-items: flex-end; gap: 6px; height: 80px; margin-top: 10px; }
.karma-col { flex: none; width: 100%; background: #9c27b0; border-radius: 4px 4px 0 0; min-height: 4px; position: relative; cursor: default; }
.karma-col::after { content: attr(data-label); position: absolute; bottom: 105%; left: 50%; transform: translateX(-50%); background: #333; color: #fff; padding: 2px 8px; border-radius: 4px; font-size: 10px; white-space: nowrap; display: none; z-index: 10; }
.karma-col:hover::after { display: block; }
.karma-label { font-size: 9px; color: #888; text-align: center; margin-top: 4px; }
.topic { background: #16213e; border-radius: 8px; padding: 8px 12px; margin: 4px 0; font-size: 13px; color: #ccc; }
.trigger { background: #16213e; border-radius: 8px; padding: 8px 12px; margin: 4px 0; font-size: 12px; color: #4caf50; }
a { color: #9c27b0; text-decoration: none; display: inline-block; margin-top: 20px; }
.last-dream { font-size: 12px; color: #888; margin-bottom: 16px; }
</style>
</head>
<body>
<h1>🌙 Dreaming ダッシュボード</h1>
""" + '<div class="last-dream">最終dreaming: ' + last_dream + '</div>' + """
<h2>⬆️ Karmaトレンド（直近10件）</h2>
<div class="card"><div class="karma-bar">""" + karma_bar_html + """</div></div>
<h2>🎯 Karma上昇トリガー（直近5件）</h2>
""" + trigger_html + """
<h2>💬 最近コメントしたトピック</h2>
""" + topic_html + """
<h2>📝 スタイルメモ</h2>
<div class="card"><p>""" + style_notes + """</p></div>
<h2>💡 最新インサイト</h2>
<div class="card"><p>""" + last_insights + """</p></div>
<h2>🚫 避けるトピック</h2>
<div class="card">""" + avoid_html + """</div>
<a href="/">← チャットに戻る</a>
</body>
</html>"""
    return html

def _render_combined_trend(dates, fails, totals, fail_patterns, fixed, fix_patterns, pat_colors):
    if not dates:
        return '<p style="color:#666;font-size:13px;background:#16213e;border-radius:12px;padding:16px;">データなし</p>'
    import re as _re
    from collections import Counter as _C
    UNKNOWN_COLOR = "#9e9e9e"
    def _short(p):
        m = _re.search(r'[（(](.+?)[）)]', p)
        return m.group(1) if m else p[:18]
    def _fail_color(p):
        if p == "パターン不明":
            return UNKNOWN_COLOR
        return pat_colors.get(p, "#e94560")
    def _fix_color(p):
        if not p:
            return UNKNOWN_COLOR
        return pat_colors.get(p, "#e94560")
    max_fail = max(fails) if fails else 1
    max_fix = max(fixed) if fixed else 1
    cols = ""
    for d, f, t, fpats, v, xpats in zip(dates, fails, totals, fail_patterns, fixed, fix_patterns):
        # 失敗バー
        fail_count = _C(fpats)
        fail_h = int((f / (max_fail + 1)) * 80) + (4 if f > 0 else 0)
        fail_lines = "\n".join(f"  {_short(p)}: {cnt}件" for p, cnt in fail_count.most_common())
        fail_tip = f"📅 {d}  失敗{f}/総{t}件"
        if fail_lines:
            fail_tip += f"\n{fail_lines}"
        fail_stack = ""
        if f == 0:
            fail_stack = '<div style="height:4px;background:#4caf50;border-radius:4px 4px 0 0;"></div>'
        else:
            for pat, cnt in fail_count.most_common():
                seg_h = max(int(fail_h * cnt / f), 4)
                fail_stack += f'<div class="captcha-seg" style="height:{seg_h}px;background:{_fail_color(pat)};"></div>'

        # 修正バー
        fix_count = _C(xpats)
        fix_h = int((v / (max_fix + 1)) * 80) + (4 if v > 0 else 0)
        fix_lines = "\n".join(f"  {_short(p)}: {cnt}件" for p, cnt in fix_count.most_common())
        fix_tip = f"📅 {d}  修正{v}件"
        if fix_lines:
            fix_tip += f"\n{fix_lines}"
        fix_stack = ""
        if v == 0:
            fix_stack = '<div style="height:4px;background:#333;border-radius:4px 4px 0 0;"></div>'
        elif not xpats:
            fix_stack = f'<div style="height:{fix_h}px;background:#9e9e9e;border-radius:4px 4px 0 0;"></div>'
            if not fix_lines:
                fix_tip += "\n（内訳データなし・機能追加前の記録）"
        else:
            for pat, cnt in fix_count.most_common():
                seg_h = max(int(fix_h * cnt / v), 4)
                fix_stack += f'<div class="captcha-seg" style="height:{seg_h}px;background:{_fix_color(pat)};"></div>'

        cols += (
            f'<div class="captcha-col-pair">'
            f'<div class="captcha-pair-bars">'
            f'<div class="captcha-stack captcha-tip" data-label="{fail_tip}">{fail_stack}</div>'
            f'<div class="captcha-stack captcha-tip" data-label="{fix_tip}">{fix_stack}</div>'
            f'</div>'
            f'<div class="captcha-col-label">{d}</div>'
            f'</div>'
        )
    return f'<div class="card"><div class="captcha-bar-legend">📅 左=失敗 ｜ 🤖 右=自動対応試行（doctor、実効性未検証）</div><div class="captcha-bar">{cols}</div></div>'

def _render_fix_trend(dates, fixed, patterns=None, pat_colors=None):
    if not dates or all(v == 0 for v in fixed):
        return '<p style="color:#666;font-size:13px;background:#16213e;border-radius:12px;padding:16px;">データなし（doctorによる自動修正が発生すると表示されます）</p>'
    if patterns is None:
        patterns = [[] for _ in dates]
    if pat_colors is None:
        pat_colors = {}
    import re as _re
    from collections import Counter as _C
    UNKNOWN_COLOR = "#9e9e9e"
    def _short(p):
        m = _re.search(r'[（(](.+?)[）)]', p)
        return m.group(1) if m else p[:18]
    def _color(p):
        if not p:
            return UNKNOWN_COLOR
        return pat_colors.get(p, "#e94560")
    max_v = max(fixed) if fixed else 1
    bars = ""
    for d, v, pats in zip(dates, fixed, patterns):
        pat_count = _C(pats)
        total_h = int((v / (max_v + 1)) * 80) + (4 if v > 0 else 0)
        pat_lines = "\n".join(f"  {_short(p)}: {cnt}件" for p, cnt in pat_count.most_common())
        tip = f"📅 {d}  修正{v}件"
        if pat_lines:
            tip += f"\n{pat_lines}"
        stack_html = ""
        if v == 0:
            stack_html = '<div style="height:4px;background:#333;border-radius:4px 4px 0 0;"></div>'
        elif not pats:
            stack_html = f'<div style="height:{total_h}px;background:#9e9e9e;border-radius:4px 4px 0 0;"></div>'
            if not pat_lines:
                tip += "\n（内訳データなし・機能追加前の記録）"
        else:
            for pat, cnt in pat_count.most_common():
                seg_h = max(int(total_h * cnt / v), 4)
                col = _color(pat)
                stack_html += f'<div class="captcha-seg" style="height:{seg_h}px;background:{col};"></div>'
        bars += (
            f'<div class="captcha-col">'
            f'<div class="captcha-stack captcha-tip" data-label="{tip}">{stack_html}</div>'
            f'<div class="captcha-col-label">{d}</div>'
            f'</div>'
        )
    return f'<div class="card"><div class="captcha-bar">{bars}</div></div>'

def _captcha_tooltip_script():
    return """<div class="captcha-tooltip" id="captcha-tt"></div>
<script>
(function(){
  var tt = document.getElementById('captcha-tt');
  function show(el, e){
    var label = el.getAttribute('data-label') || '';
    if(!label) return;
    tt.textContent = label;
    tt.style.display = 'block';
    move(e);
  }
  function move(e){
    var x = (e.touches ? e.touches[0].clientX : e.clientX);
    var y = (e.touches ? e.touches[0].clientY : e.clientY);
    tt.style.left = Math.min(x + 12, window.innerWidth - 230) + 'px';
    tt.style.top = (y - tt.offsetHeight - 12) + 'px';
  }
  function hide(){ tt.style.display = 'none'; }
  function bindAll(){
    document.querySelectorAll('.captcha-tip').forEach(function(el){
      if (el.dataset.tipBound) return;
      el.dataset.tipBound = '1';
      el.addEventListener('mouseenter', function(e){ show(el, e); });
      el.addEventListener('mousemove', function(e){ move(e); });
      el.addEventListener('mouseleave', function(){ hide(); });
      el.addEventListener('touchstart', function(e){
        e.stopPropagation();
        if(tt.style.display === 'block' && tt._src === el){ hide(); tt._src = null; return; }
        show(el, e);
        tt._src = el;
      });
    });
  }
  // ページ全体のパース完了後にバインド（後続に描画される修正件数トレンド等のバーも拾うため）
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bindAll);
  } else {
    bindAll();
  }
  document.addEventListener('touchstart', function(){ hide(); });
})();
</script>"""


def _render_captcha_trend(dates, fails, totals, patterns=None, pat_colors=None):
    if not dates or all(f == 0 for f in fails):
        return '<p style="color:#666;font-size:13px;background:#16213e;border-radius:12px;padding:16px;">データなし（次回CAPTCHA実行後に表示されます）</p>'
    if patterns is None:
        patterns = [[] for _ in dates]
    if pat_colors is None:
        pat_colors = {}
    import re as _re
    from collections import Counter as _C
    _fp_colors = ["#e94560", "#ff9800", "#f06292", "#ba68c8", "#4db6ac", "#64b5f6"]
    UNKNOWN_COLOR = "#9e9e9e"  # パターン不明専用色（累計パレットと重複しない灰色）
    def _short(p):
        m = _re.search(r'[（(](.+?)[）)]', p)
        return m.group(1) if m else p[:18]
    def _color(p):
        if p == "パターン不明":
            return UNKNOWN_COLOR
        return pat_colors.get(p, "#e94560")
    max_fail = max(fails) if fails else 1
    bars = ""
    for d, f, t, pats in zip(dates, fails, totals, patterns):
        pat_count = _C(pats)
        total_h = int((f / (max_fail + 1)) * 80) + (4 if f > 0 else 0)
        # 日別統合tooltip
        pat_lines = "\n".join(f"  {_short(p)}: {cnt}件" for p, cnt in pat_count.most_common())
        tip = f"📅 {d}  失敗{f}/総{t}件"
        if pat_lines:
            tip += f"\n{pat_lines.replace('&#10;', chr(10))}"
        # スタック棒グラフ
        stack_html = ""
        if f == 0:
            stack_html = '<div style="height:4px;background:#4caf50;border-radius:4px 4px 0 0;"></div>'
        else:
            for pat, cnt in pat_count.most_common():
                seg_h = max(int(total_h * cnt / f), 4)
                col = _color(pat)
                stack_html += f'<div class="captcha-seg" style="height:{seg_h}px;background:{col};"></div>'
        bars += (
            f'<div class="captcha-col">'
            f'<div class="captcha-stack captcha-tip" data-label="{tip}">{stack_html}</div>'
            f'<div class="captcha-col-label">{d}</div>'
            f'</div>'
        )
    script = ""
    return f'<div class="card"><div class="captcha-bar">{bars}</div></div>' + script

def _render_fail_patterns(fail_patterns):
    if not fail_patterns:
        return '<p style="color:#666;font-size:13px;">データなし</p>'
    sorted_patterns = sorted(fail_patterns.items(), key=lambda x: -x[1])
    max_val = max(v for _, v in sorted_patterns) if sorted_patterns else 1
    colors = ["#e94560", "#ff9800", "#f06292", "#ba68c8", "#4db6ac", "#64b5f6"]
    parts = []
    for i, (k, v) in enumerate(sorted_patterns):
        pct = round(v / max_val * 100, 1)
        color = colors[i % len(colors)]
        parts.append(f'''<div class="pattern-bar-wrap">
  <div class="pattern-label"><span>{k}</span><span style="color:{color};font-weight:bold;">{v}件</span></div>
  <div class="pattern-bar-bg"><div class="pattern-bar-fill" style="width:{pct}%;background:{color};">{v}</div></div>
</div>''')
    return "\n".join(parts)

@app.route('/captcha/stats', methods=['GET'])
def captcha_stats():
    if not check_web_auth():
        return redirect('/login')
    import json, os
    memory_path = os.path.expanduser("~/ai-agent/moltbook/memory.json")
    try:
        m = json.load(open(memory_path))
    except Exception:
        return jsonify({"error": "memory.json not found"})

    total_old = m.get("successful_comments", 0) + m.get("failed_challenges", 0)
    stats = m.get("captcha_stats", {})
    total = stats.get("total", total_old)
    success = stats.get("success", m.get("successful_comments", 0))
    fail = total - success
    rate = round(success / total * 100, 1) if total > 0 else 0
    fail_patterns = stats.get("fail_patterns", {})
    karma = m.get("karma_history", [])
    latest_karma = karma[-1]["karma"] if karma else 0

    # 最終更新時刻（captcha_historyの最新エントリ）
    last_captcha_update = "データなし"
    _ch_for_time = m.get("captcha_history", [])
    if _ch_for_time:
        last_captcha_update = _ch_for_time[-1].get("time", "データなし")

    # captcha_history を日別に集計（直近10日）
    from collections import defaultdict as _dd
    captcha_history = m.get("captcha_history", [])
    daily = _dd(lambda: {"total": 0, "fail": 0, "patterns": []})
    def _extract_mmdd(_t):
        # 新フォーマット "YYYY-MM-DD HH:MM" と旧フォーマット "MM-DD HH:MM" の両対応
        if not _t:
            return ""
        if len(_t) >= 10 and _t[4] == "-":
            return _t[5:10]  # "YYYY-MM-DD ..." -> "MM-DD"
        return _t[:5]  # "MM-DD ..." -> "MM-DD"

    for entry in captcha_history:
        date = _extract_mmdd(entry.get("time", ""))
        if date:
            daily[date]["total"] += 1
            if not entry.get("success", 1):
                daily[date]["fail"] += 1
                pat = entry.get("pattern", "") or "パターン不明"
                daily[date]["patterns"].append(pat)
    chart_dates = sorted(daily.keys())[-10:]
    chart_fails = [daily[d]["fail"] for d in chart_dates]
    chart_totals = [daily[d]["total"] for d in chart_dates]
    chart_patterns = [daily[d]["patterns"] for d in chart_dates]

    # doctor_fix_history を同じ日付軸で日別集計（パターン内訳も収集）
    doctor_fix_history = m.get("doctor_fix_history", [])
    daily_fix = _dd(lambda: {"count": 0, "patterns": []})
    for entry in doctor_fix_history:
        date = _extract_mmdd(entry.get("time", ""))
        if date:
            daily_fix[date]["count"] += entry.get("fixed", 0)
            daily_fix[date]["patterns"].extend(entry.get("patterns", []))
    chart_fixed = [daily_fix[d]["count"] if d in daily_fix else 0 for d in chart_dates]
    chart_fixed_patterns = [daily_fix[d]["patterns"] if d in daily_fix else [] for d in chart_dates]

    # 手動修正済み履歴（manual_fix_history）を同じ日付軸で日別集計
    manual_fix_history = m.get("manual_fix_history", [])
    daily_manual = _dd(lambda: {"count": 0, "patterns": []})
    for entry in manual_fix_history:
        date = _extract_mmdd(entry.get("time", ""))
        if date:
            daily_manual[date]["count"] += entry.get("fixed", 0)
            daily_manual[date]["patterns"].extend(entry.get("patterns", []))
    chart_manual_fixed = [daily_manual[d]["count"] if d in daily_manual else 0 for d in chart_dates]
    chart_manual_fixed_patterns = [daily_manual[d]["patterns"] if d in daily_manual else [] for d in chart_dates]

    # 手動修正待ち一覧（unclassified_patterns.json）
    manual_pending_count = 0
    manual_pending_html = "<p style=\"color:#666;font-size:13px;\">データなし</p>"
    try:
        import html as _html_mod
        unclass_path = os.path.expanduser("~/ai-agent/moltbook/unclassified_patterns.json")
        with open(unclass_path) as _uf:
            _ustore = json.load(_uf)
        _pending = _ustore.get("entries", [])
        manual_pending_count = len(_pending)
        if _pending:
            _rows = "".join(
                f'<div class="topic" style="font-size:12px;">{_html_mod.escape(e.get("time",""))} | {_html_mod.escape(e.get("challenge","")[:80])}</div>'
                for e in reversed(_pending)
            )
            manual_pending_html = f'<div class="card" style="max-height:320px;overflow-y:auto;">{_rows}</div>'
    except Exception:
        pass
    # 失敗パターン累計と同じ色マッピングを生成
    _fp_colors = ["#e94560", "#ff9800", "#f06292", "#ba68c8", "#4db6ac", "#64b5f6"]
    _sorted_fp = sorted(fail_patterns.items(), key=lambda x: -x[1])
    pat_colors = {p: _fp_colors[i % len(_fp_colors)] for i, (p, _) in enumerate(_sorted_fp)}

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🦞 CAPTCHA Stats</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }}
h1 {{ font-size: 22px; margin-bottom: 20px; color: #e94560; }}
h2 {{ font-size: 16px; margin: 20px 0 10px; color: #aaa; }}
.cards {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }}
.card {{ background: #16213e; border-radius: 12px; padding: 16px; min-width: 140px; flex: 1; }}
.card .num {{ font-size: 32px; font-weight: bold; color: #e94560; }}
.card .label {{ font-size: 12px; color: #888; margin-top: 4px; }}
.rate {{ font-size: 48px; font-weight: bold; color: {"#4caf50" if rate >= 80 else "#ff9800" if rate >= 60 else "#e94560"}; }}
.bar-bg {{ background: #0f0f23; border-radius: 8px; height: 20px; margin: 10px 0; overflow: hidden; }}
.bar-fill {{ height: 100%; background: {"#4caf50" if rate >= 80 else "#ff9800" if rate >= 60 else "#e94560"}; border-radius: 8px; width: {rate}%; transition: width 0.5s; }}
.pattern-bar-wrap {{ margin: 10px 0; }}
.pattern-label {{ font-size: 12px; color: #ccc; margin-bottom: 4px; display: flex; justify-content: space-between; }}
.pattern-bar-bg {{ background: #0f0f23; border-radius: 6px; height: 22px; overflow: hidden; }}
.pattern-bar-fill {{ height: 100%; border-radius: 6px; transition: width 0.5s; display: flex; align-items: center; padding-left: 8px; font-size: 11px; font-weight: bold; color: #fff; min-width: 2px; }}
.captcha-bar {{ display: flex; align-items: flex-end; gap: 6px; height: 100px; margin-top: 10px; }}
.captcha-col {{ flex: 1; display: flex; flex-direction: column; align-items: center; }}
.captcha-stack {{ width: 100%; display: flex; flex-direction: column-reverse; border-radius: 4px 4px 0 0; overflow: hidden; cursor: pointer; position: relative; }}
.captcha-seg {{ width: 100%; min-height: 4px; }}
.captcha-tooltip {{ position: fixed; background: #222; color: #fff; padding: 8px 12px; border-radius: 8px; font-size: 12px; white-space: pre; z-index: 9999; border: 1px solid #555; line-height: 1.8; pointer-events: none; display: none; max-width: 220px; }}
.captcha-col-label {{ font-size: 9px; color: #888; text-align: center; margin-top: 4px; }}
.captcha-col-pair {{ flex: 1; display: flex; flex-direction: column; align-items: center; }}
.captcha-pair-bars {{ display: flex; gap: 3px; align-items: flex-end; width: 100%; }}
.captcha-pair-bars .captcha-stack {{ flex: 1; }}
.captcha-bar-legend {{ font-size: 11px; color: #888; margin-bottom: 6px; }}
.last-update {{ font-size: 12px; color: #888; margin-bottom: 16px; }}
a {{ color: #4caf50; text-decoration: none; display: inline-block; margin-top: 20px; }}
</style>
</head>
<body>
<h1>🦞 CAPTCHA ダッシュボード</h1>
<div class="last-update">最終更新: {last_captcha_update}</div>
<div class="cards">
  <div class="card"><div class="num">{total}</div><div class="label">総試行回数</div></div>
  <div class="card"><div class="num" style="color:#4caf50">{success}</div><div class="label">✅ 成功</div></div>
  <div class="card"><div class="num" style="color:#e94560">{fail}</div><div class="label">❌ 失敗</div></div>
  <div class="card"><div class="num">{latest_karma}</div><div class="label">⬆️ Karma</div></div>
</div>
<h2>正解率</h2>
<div class="rate">{rate}%</div>
<div class="bar-bg"><div class="bar-fill"></div></div>
<h2>件数トレンド（日別・直近10日）</h2>
{_captcha_tooltip_script()}
{_render_combined_trend(chart_dates, chart_fails, chart_totals, chart_patterns, chart_fixed, chart_fixed_patterns, pat_colors)}
<h2>🕒 手動修正待ち（{manual_pending_count}件）</h2>
{manual_pending_html}
<h2>✅ 手動修正済みトレンド（日別・直近10日）</h2>
{_render_fix_trend(chart_dates, chart_manual_fixed, chart_manual_fixed_patterns, pat_colors)}
<h2>失敗パターン（累計）</h2>
{_render_fail_patterns(fail_patterns)}
<a href="/">← チャットに戻る</a>
</body>
</html>"""
    return html

@app.route('/session/clear', methods=['POST'])
def session_clear():
    if not check_auth():
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json() or {}
    session_id = data.get('session_id', 'default')
    # メモリから削除
    if session_id in conversation_histories:
        conversation_histories[session_id] = []
    # DBから削除
    try:
        conn = sqlite3.connect(CACHE_DB)
        conn.execute('DELETE FROM conversations WHERE session_id=?', (session_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        log('session_clear DB error: ' + str(e))
    log(f"🗑️ セッション履歴クリア: {session_id[-8:]}")
    return jsonify({"status": "cleared", "session_id": session_id})

@app.route('/cache/clear', methods=['POST'])
def cache_clear():
    if not check_auth():
        return jsonify({"error": "Unauthorized"}), 401
    conn = sqlite3.connect(CACHE_DB)
    conn.execute("DELETE FROM cache")
    conn.commit()
    conn.close()
    return jsonify({"status": "cleared"})

@app.route('/history', methods=['GET'])
def history():
    session_id = request.args.get('session_id', 'default')
    return jsonify(conversation_histories.get(session_id, []))

@app.route('/clear', methods=['POST'])
def clear():
    if not check_auth():
        return jsonify({"error": "Unauthorized"}), 401
    global conversation_history
    conversation_history = []
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    init_cache()
    log("🐈 オーケストレーター v4 起動中...")
    log("📱 iPhone: http://100.109.207.78:11437")
    log("💻 PC:     http://127.0.0.1:11437")
    log("💾 キャッシュ統計: http://127.0.0.1:11437/cache/stats")
    log("終了: Ctrl+C")
    cert = os.path.expanduser('~/MythoFable/hz-k-2mba14.tailb82610.ts.net.crt')
    key  = os.path.expanduser('~/MythoFable/hz-k-2mba14.tailb82610.ts.net.key')
    ssl_ctx = (cert, key) if os.path.exists(cert) and os.path.exists(key) else None
    app.run(host='0.0.0.0', port=11437, debug=False, ssl_context=ssl_ctx)
