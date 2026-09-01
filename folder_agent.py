"""
folder_agent.py

「#<フォルダパス>: <やってほしいこと>」形式のリクエストを受けて、無料枠モデル
(OpenRouter経由)にファイル一覧取得・読み込み・書き込み・コマンド実行提案・
git commit提案というツールを与え、タスクが完了するまで自律的にループさせる
モジュール。

【安全設計】
- 対象は必ずgitリポジトリ(.gitディレクトリの存在を確認)。ロールバックの安全網。
- list_files/read_file/write_fileは対象フォルダの外に出られないようパス検証する
  (../ によるディレクトリトラバーサルを拒否)
- run_command(診断・テスト用のコマンド実行)とgit_commitは、モデルが「提案」し、
  人間が承認してから初めて実行される(auto_patch.pyの承認待ちパッチと同じ設計思想)
- run_commandはホワイトリストに前方一致するコマンドのみ許可(任意コマンド実行はさせない)
- 最大ステップ数の上限で暴走を防止
- ループの途中状態(会話履歴・承認待ちの提案内容)はDBに保存し、セッションをまたいで再開できる

このモジュールはツールの実行・状態管理のみを担当する。HTTPリクエストの入り口
(「#」プレフィックスの検出、承認/却下メッセージの判定)はorchestrator_v4.py側で行う。
"""
import json
import os
import re
import sqlite3
import subprocess
import time

import requests

from model_status import filter_alive_models

MAX_STEPS = 20
MAX_FILE_READ_BYTES = 60_000
MAX_FILE_LIST_ENTRIES = 400
COMMAND_TIMEOUT = 60

# run_commandで許可するコマンドの前方一致ホワイトリスト。
# 診断・テスト・確認用途のみ。ファイルを書き換えたり外部に影響を与えるものは含めない。
ALLOWED_COMMAND_PREFIXES = [
    "git status", "git diff", "git log", "git show",
    "python3 -m py_compile", "python -m py_compile",
    "python3 -m pytest", "python -m pytest", "pytest",
    "node --check",
    "npm test", "npm run build", "npm run lint",
    "bash -n", "sh -n",
    "ls", "cat", "wc -l", "grep",
]

AGENT_MODELS = filter_alive_models([
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openrouter/free",
], provider="openrouter")


# ── 状態の永続化(DB) ──────────────────────────────────────

def _init_table(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_sessions (
            session_id TEXT PRIMARY KEY,
            target_folder TEXT,
            task TEXT,
            messages_json TEXT,
            step_count INTEGER,
            waiting_for TEXT,
            proposed_value TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_agent_session(db_path, session_id, target_folder, task, messages, step_count,
                        waiting_for=None, proposed_value=None):
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM agent_sessions WHERE session_id=?", (session_id,))
    conn.execute(
        "INSERT INTO agent_sessions (session_id, target_folder, task, messages_json, "
        "step_count, waiting_for, proposed_value) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (session_id, target_folder, task, json.dumps(messages, ensure_ascii=False),
         step_count, waiting_for, proposed_value)
    )
    conn.commit()
    conn.close()


def get_agent_session(db_path, session_id, timeout_seconds=1800):
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT target_folder, task, messages_json, step_count, waiting_for, proposed_value, "
        "created_at FROM agent_sessions WHERE session_id=?",
        (session_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    target_folder, task, messages_json, step_count, waiting_for, proposed_value, created_at = row
    try:
        import datetime
        created = datetime.datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
        age = (datetime.datetime.utcnow() - created).total_seconds()
        if age > timeout_seconds:
            delete_agent_session(db_path, session_id)
            return None
    except Exception:
        pass
    return {
        "target_folder": target_folder,
        "task": task,
        "messages": json.loads(messages_json),
        "step_count": step_count,
        "waiting_for": waiting_for,
        "proposed_value": proposed_value,
    }


def delete_agent_session(db_path, session_id):
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM agent_sessions WHERE session_id=?", (session_id,))
    conn.commit()
    conn.close()


# ── 安全性チェック ──────────────────────────────────────

def resolve_target_folder(raw_path):
    """フォルダパスを正規化して存在確認・gitリポジトリ確認を行う。
    問題なければ絶対パスを、問題があればエラーメッセージを返す。
    """
    path = os.path.expanduser(raw_path.strip())
    if not os.path.isabs(path):
        path = os.path.expanduser(os.path.join("~", path))
    path = os.path.abspath(path)

    if not os.path.isdir(path):
        return None, f"フォルダが見つかりません: {path}"
    if not os.path.isdir(os.path.join(path, ".git")):
        return None, f"gitリポジトリではありません(.gitが見つかりません): {path}\nロールバックの安全網のため、gitリポジトリ内でのみ実行できます。"
    return path, None


def _safe_join(target_folder, rel_path):
    """target_folderの外に出るパス(../によるトラバーサル等)を拒否する"""
    rel_path = (rel_path or "").lstrip("/")
    joined = os.path.abspath(os.path.join(target_folder, rel_path))
    if not (joined == target_folder or joined.startswith(target_folder + os.sep)):
        raise ValueError(f"対象フォルダの外のパスは扱えません: {rel_path}")
    return joined


def is_command_allowed(command):
    command = command.strip()
    return any(command.startswith(p) for p in ALLOWED_COMMAND_PREFIXES)


# ── ツール実装 ──────────────────────────────────────

_EXCLUDE_DIR_NAMES = {".git", "node_modules", "__pycache__", ".venv", "venv", ".pytest_cache"}


def tool_list_files(target_folder, subpath=""):
    base = _safe_join(target_folder, subpath)
    entries = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in _EXCLUDE_DIR_NAMES]
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, target_folder)
            entries.append(rel)
            if len(entries) >= MAX_FILE_LIST_ENTRIES:
                entries.append(f"... (上限{MAX_FILE_LIST_ENTRIES}件に達したため以降省略)")
                return "\n".join(entries)
    return "\n".join(entries) if entries else "(ファイルなし)"


def tool_read_file(target_folder, path):
    full = _safe_join(target_folder, path)
    if not os.path.isfile(full):
        return f"❌ ファイルが見つかりません: {path}"
    with open(full, "rb") as f:
        data = f.read(MAX_FILE_READ_BYTES + 1)
    truncated = len(data) > MAX_FILE_READ_BYTES
    text = data[:MAX_FILE_READ_BYTES].decode("utf-8", errors="replace")
    if truncated:
        text += f"\n... (以降省略、{MAX_FILE_READ_BYTES}バイトを超えるため切り詰め)"
    return text


def tool_write_file(target_folder, path, content):
    full = _safe_join(target_folder, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    return f"✅ 書き込み完了: {path} ({len(content)}文字)"


def execute_approved_command(target_folder, command):
    if not is_command_allowed(command):
        return (
            f"❌ 許可されていないコマンドです(ホワイトリスト外): {command}\n"
            f"許可されているのは次の形式のみです: {', '.join(ALLOWED_COMMAND_PREFIXES)}\n"
            "同じ、または類似のコマンドを再提案しないでください。"
            "診断が必須でなければ、これ以上run_commandは使わずにfinish_taskで終了してください。"
        )
    try:
        result = subprocess.run(
            command, shell=True, cwd=target_folder,
            capture_output=True, text=True, timeout=COMMAND_TIMEOUT
        )
        out = (result.stdout or "") + (result.stderr or "")
        return out[:4000] if out else "(出力なし、正常終了)"
    except subprocess.TimeoutExpired:
        return f"❌ タイムアウト({COMMAND_TIMEOUT}秒)"
    except Exception as e:
        return f"❌ 実行エラー: {e}"


def has_uncommitted_changes(target_folder):
    """git status --porcelainで未コミットの変更があるか確認する"""
    try:
        result = subprocess.run(["git", "status", "--porcelain"], cwd=target_folder,
                                 timeout=15, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return ""


def execute_approved_commit(target_folder, message):
    try:
        subprocess.run(["git", "add", "-A"], cwd=target_folder, timeout=30, check=True,
                        capture_output=True, text=True)
        result = subprocess.run(["git", "commit", "-m", message], cwd=target_folder,
                                 timeout=30, capture_output=True, text=True)
        if result.returncode != 0:
            if "nothing to commit" in (result.stdout + result.stderr):
                return "ℹ️ コミット対象の変更がありません"
            return f"❌ commit失敗: {result.stdout}{result.stderr}"
        push_result = subprocess.run(["git", "push"], cwd=target_folder, timeout=60,
                                      capture_output=True, text=True)
        if push_result.returncode != 0:
            return f"✅ commit成功(push失敗、手動push要): {result.stdout}\n{push_result.stderr}"
        return f"✅ commit + push 完了: {result.stdout}"
    except Exception as e:
        return f"❌ commit処理エラー: {e}"


# ── ツール定義(OpenAI互換tools形式) ──────────────────────────────────────

TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "list_files",
        "description": "対象フォルダ内のファイル一覧を取得する(再帰的)。.git等は自動除外される。",
        "parameters": {"type": "object", "properties": {
            "subpath": {"type": "string", "description": "省略可。特定のサブフォルダのみ見たい場合に指定"}
        }}
    }},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "指定したファイルの内容を読み込む",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "対象フォルダからの相対パス"}
        }, "required": ["path"]}
    }},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "指定したファイルを新規作成、または内容を丸ごと上書きする",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "対象フォルダからの相対パス"},
            "content": {"type": "string", "description": "書き込む内容全体"}
        }, "required": ["path", "content"]}
    }},
    {"type": "function", "function": {
        "name": "run_command",
        "description": "診断・テスト用のコマンドを実行したい場合に提案する(実行前に人間の承認が必要)。"
                        "許可されているのは git status/diff/log, python/pytestの構文・テストチェック, "
                        "node --check, npm test/build/lint, ls/cat/grep等の確認系コマンドのみ。",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "実行したいコマンド"}
        }, "required": ["command"]}
    }},
    {"type": "function", "function": {
        "name": "git_commit",
        "description": "これまでの変更をgit commit(+push)したい場合に提案する(実行前に人間の承認が必要)",
        "parameters": {"type": "object", "properties": {
            "message": {"type": "string", "description": "コミットメッセージ"}
        }, "required": ["message"]}
    }},
    {"type": "function", "function": {
        "name": "finish_task",
        "description": "タスクが完了した(またはこれ以上進められない)と判断したら呼ぶ",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string", "description": "何をしたか/なぜ終了するかの要約"}
        }, "required": ["summary"]}
    }},
]

SYSTEM_PROMPT_TEMPLATE = """あなたはローカルのgitリポジトリで作業する自律コーディングエージェントです。
対象フォルダ: {target_folder}

利用可能なツールを使って、与えられたタスクを進めてください。
- list_files/read_file で現状を把握してから作業してください
- write_file でファイルの作成・編集ができます(即座に反映されます)
- run_command は診断・テスト用のコマンドを"提案"するものです。人間の承認を経てから実行され、
  結果があなたに返されます。使いすぎず、本当に必要な時だけ提案してください
- 作業がまとまったら git_commit でコミットを提案してください(これも人間の承認が必要です)
- タスクが完了した、またはこれ以上進められないと判断したら finish_task を呼んで終了してください
- 対象フォルダの外のファイルは扱えません
"""


# ── メインループ ──────────────────────────────────────

def call_agent_model(messages, tools):
    if not os.environ.get("OPENROUTER_API_KEY"):
        return None, "OPENROUTER_API_KEYが設定されていません"
    models = AGENT_MODELS or ["openrouter/free"]
    errors = []
    for m in models:
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
                    "Content-Type": "application/json",
                },
                json={"model": m, "messages": messages, "tools": tools, "max_tokens": 2000, "temperature": 0.3},
                timeout=90,
            )
            data = r.json()
            if "choices" not in data:
                errors.append(f"{m}: {data.get('error', {}).get('message', str(data))}")
                continue
            return data["choices"][0]["message"], None
        except Exception as e:
            errors.append(f"{m} exception: {e}")
            continue
    return None, " / ".join(errors)


def run_loop(db_path, session_id, target_folder, task, messages, step_count):
    """ステップが上限に達するか、run_command/git_commit提案またはfinish_taskに
    到達するまでループを進める。戻り値はユーザーに表示するテキスト。"""
    while step_count < MAX_STEPS:
        step_count += 1
        assistant_msg, error = call_agent_model(messages, TOOLS_SCHEMA)
        if assistant_msg is None:
            delete_agent_session(db_path, session_id)
            return f"❌ モデル呼び出しに失敗しました: {error}"

        tool_calls = assistant_msg.get("tool_calls")
        if not tool_calls:
            # ツールを呼ばずテキストで返してきた場合はそのまま終了扱いにする
            delete_agent_session(db_path, session_id)
            content = assistant_msg.get("content") or "(応答なし)"
            return f"🤖 {content}"

        messages.append(assistant_msg)
        call = tool_calls[0]
        fn_name = call["function"]["name"]
        try:
            args = json.loads(call["function"].get("arguments") or "{}")
        except Exception:
            args = {}

        if fn_name == "finish_task":
            _uncommitted = has_uncommitted_changes(target_folder)
            if _uncommitted:
                messages.append({"role": "tool", "tool_call_id": call["id"], "content":
                    "⚠️ finish_taskを呼ぶ前に、まだコミットされていない変更が残っています。"
                    "先にgit_commitを提案してください。\n未コミットの変更:\n" + _uncommitted[:1000]})
                continue
            delete_agent_session(db_path, session_id)
            return f"✅ 完了: {args.get('summary', '(要約なし)')}"

        if fn_name == "run_command":
            command = args.get("command", "")
            save_agent_session(db_path, session_id, target_folder, task, messages, step_count,
                                waiting_for="command", proposed_value=command)
            allowed_note = "" if is_command_allowed(command) else "\n⚠️ このコマンドはホワイトリスト外のため、承認しても実行されません。"
            return f"🔧 コマンド実行の提案があります:\n`{command}`{allowed_note}\n\n実行してよければ「承認」、やめるなら「キャンセル」と送ってください。"

        if fn_name == "git_commit":
            message = args.get("message", "")
            save_agent_session(db_path, session_id, target_folder, task, messages, step_count,
                                waiting_for="commit", proposed_value=message)
            return f"📝 git commitの提案があります:\nメッセージ: {message}\n\n実行してよければ「承認」、やめるなら「キャンセル」と送ってください。"

        # list_files / read_file / write_file は自動実行
        try:
            if fn_name == "list_files":
                result = tool_list_files(target_folder, args.get("subpath", ""))
            elif fn_name == "read_file":
                result = tool_read_file(target_folder, args.get("path", ""))
            elif fn_name == "write_file":
                result = tool_write_file(target_folder, args.get("path", ""), args.get("content", ""))
            else:
                result = f"❌ 未知のツール: {fn_name}"
        except Exception as e:
            result = f"❌ エラー: {e}"

        messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})

    save_agent_session(db_path, session_id, target_folder, task, messages, step_count)
    return f"⏸ 最大ステップ数({MAX_STEPS})に達したため一時停止しました。続けるには再度メッセージを送ってください。"


def start_task(db_path, session_id, target_folder, task):
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(target_folder=target_folder)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]
    return run_loop(db_path, session_id, target_folder, task, messages, step_count=0)


def resume_after_command(db_path, session_id, approved):
    session = get_agent_session(db_path, session_id)
    if not session:
        return None
    messages = session["messages"]
    target_folder = session["target_folder"]
    task = session["task"]
    step_count = session["step_count"]
    waiting_for = session["waiting_for"]
    proposed_value = session["proposed_value"]

    # 直前のtool_callのidを、保存済みmessagesの末尾(assistant)から取得
    last_assistant = next((m for m in reversed(messages) if m.get("role") == "assistant" and m.get("tool_calls")), None)
    tool_call_id = last_assistant["tool_calls"][0]["id"] if last_assistant else "unknown"

    if not approved:
        result = "(ユーザーによりキャンセルされました)"
    elif waiting_for == "command":
        result = execute_approved_command(target_folder, proposed_value)
    elif waiting_for == "commit":
        result = execute_approved_commit(target_folder, proposed_value)
    else:
        result = "(不明な状態)"

    messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": result})
    return run_loop(db_path, session_id, target_folder, task, messages, step_count)
