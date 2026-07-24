import ast, json, re, os, shutil, subprocess, datetime, difflib, time

ALLOWED_FILES = {
    "orchestrator_v4.py": "/Users/fk/ai-orchestrator/orchestrator_v4.py",
    "agent_claude.py": "/Users/fk/ai-agent/moltbook/agent_claude.py",
    "agent_log_doctor.py": "/Users/fk/ai-agent/moltbook/agent_log_doctor.py",
    "dashboard.py": "/Users/fk/MythoFable/dashboard.py",
    "log_watcher.py": "/Users/fk/MythoFable/log_watcher.py",
    "auto_patcher.py": "/Users/fk/MythoFable/auto_patcher.py",
    "auto_recovery.py": "/Users/fk/MythoFable/auto_recovery.py",
    "exit_node_monitor.py": "/Users/fk/MythoFable/exit_node_monitor.py",
    "ip_manager.py": "/Users/fk/MythoFable/ip_manager.py",
    "mythofable_s.py": "/Users/fk/MythoFable/mythofable_s.py",
    "proxy_watcher.py": "/Users/fk/MythoFable/proxy_watcher.py",
}

BACKUP_DIR_MAP = {
    "orchestrator_v4.py": "/Users/fk/ai-orchestrator/",
    "agent_claude.py": "/Users/fk/ai-agent/moltbook/backups/",
    "agent_log_doctor.py": "/Users/fk/ai-agent/moltbook/backups/",
    "dashboard.py": "/Users/fk/MythoFable/backups/",
    "log_watcher.py": "/Users/fk/MythoFable/backups/",
    "auto_patcher.py": "/Users/fk/MythoFable/backups/",
    "auto_recovery.py": "/Users/fk/MythoFable/backups/",
    "exit_node_monitor.py": "/Users/fk/MythoFable/backups/",
    "ip_manager.py": "/Users/fk/MythoFable/backups/",
    "mythofable_s.py": "/Users/fk/MythoFable/backups/",
    "proxy_watcher.py": "/Users/fk/MythoFable/backups/",
}

PATCH_SYSTEM_PROMPT = """あなたはPythonコードの修正パッチを生成するアシスタントです。
与えられたファイル内容と修正指示から、修正箇所を単一のJSONオブジェクトで返してください。
出力はJSONオブジェクトのみ。説明文やMarkdownのコードブロック記号は一切含めないこと。
形式: {"patches": [{"old_str": "元のコードの一意な一部分", "new_str": "置き換え後のコード", "reason": "変更理由の短い説明"}, ...]}
old_strはファイル内で一意に一箇所だけに一致する、十分な長さの文字列にすること。
コメントや説明文は一切出力しないこと。JSONオブジェクトのみを返すこと。

重要な制約:
- 思考過程(chain of thought)は一切出力しないこと
- 「We are」「Let's」「まず」等の前置きや検討過程を書かないこと
- 出力の最初の1文字は必ず "{" にすること
- 出力の最後の1文字は必ず "}" にすること
"""

def build_patch_prompt(file_content: str, instruction: str) -> list:
    return [
        {"role": "system", "content": PATCH_SYSTEM_PROMPT},
        {"role": "user", "content": f"### 対象ファイル内容\n{file_content}\n\n### 修正指示\n{instruction}"}
    ]

def extract_json_object(text):
    """文字列中の最初のトップレベルJSONオブジェクト({...})を抽出してパースする"""
    idx = text.find("{")
    if idx == -1:
        raise ValueError(f"JSONオブジェクトが見つかりません: {text[:200]!r}")
    depth = 0
    in_str = False
    esc = False
    for j in range(idx, len(text)):
        c = text[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[idx:j+1])
    raise ValueError(f"JSONオブジェクトが閉じられていません: {text[idx:idx+200]!r}")


def _extract_json_arrays(text):
    """文字列中に含まれるトップレベルのJSON配列候補(文字列を考慮した括弧対応)をすべて抽出する"""
    results = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "[":
            depth = 0
            in_str = False
            esc = False
            j = i
            while j < n:
                c = text[j]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                else:
                    if c == '"':
                        in_str = True
                    elif c == "[":
                        depth += 1
                    elif c == "]":
                        depth -= 1
                        if depth == 0:
                            results.append(text[i:j+1])
                            break
                j += 1
            i = j + 1
        else:
            i += 1
    return results


def _extract_json_objects(text):
    """文字列中に含まれるトップレベルのJSONオブジェクト候補(文字列を考慮した波括弧対応)をすべて抽出する"""
    results = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "{":
            depth = 0
            in_str = False
            esc = False
            j = i
            while j < n:
                c = text[j]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                else:
                    if c == '"':
                        in_str = True
                    elif c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            results.append(text[i:j+1])
                            break
                j += 1
            i = j + 1
        else:
            i += 1
    return results


def parse_patch_response(raw: str):
    raw = raw.strip()
    raw = re.sub(r"^```(json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    candidates = _extract_json_objects(raw)
    if not candidates and raw.startswith("{"):
        candidates = [raw]
    if not candidates:
        raise ValueError(f"応答にJSONオブジェクトが見つかりません（推論テキストのみの可能性）: {raw[:200]!r}")

    last_error = None
    # 後方(=最終的な回答である可能性が高い)から順に、スキーマに合致する候補を採用
    for candidate in reversed(candidates):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError as e:
            last_error = e
            continue
        if not isinstance(obj, dict) or "patches" not in obj:
            last_error = ValueError("'patches'キーを持つオブジェクトではありません")
            continue
        patches = obj["patches"]
        if not isinstance(patches, list) or len(patches) == 0:
            last_error = ValueError("patchesが空、またはリストではありません")
            continue
        if all(isinstance(p, dict) and "old_str" in p and "new_str" in p for p in patches):
            return patches
        last_error = ValueError("old_str/new_strが欠落しているパッチがあります")

    raise ValueError(f"JSON解析エラー: {last_error}")

def validate_uniqueness(file_content: str, patches: list):
    errors = []
    for i, p in enumerate(patches):
        count = file_content.count(p["old_str"])
        if count == 0:
            errors.append(f"パッチ{i+1}: old_strがファイル内に見つかりません")
        elif count > 1:
            errors.append(f"パッチ{i+1}: old_strが{count}箇所に一致し一意ではありません")
    return errors

def apply_patches(file_content: str, patches: list) -> str:
    new_content = file_content
    for p in patches:
        new_content = new_content.replace(p["old_str"], p["new_str"], 1)
    return new_content

def validate_syntax(new_content: str):
    try:
        ast.parse(new_content)
        return True, None
    except SyntaxError as e:
        return False, str(e)

def make_diff(old_content: str, new_content: str, filename: str) -> str:
    diff = difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"{filename}(修正前)",
        tofile=f"{filename}(修正後)",
    )
    return "".join(diff)

def backup_file(filename: str) -> str:
    src = ALLOWED_FILES[filename]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR_MAP[filename], f"{filename}.autopatch_backup_{ts}")
    shutil.copy2(src, backup_path)
    return backup_path

def write_file(filename: str, content: str):
    path = ALLOWED_FILES[filename]
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def restore_backup(filename: str, backup_path: str):
    shutil.copy2(backup_path, ALLOWED_FILES[filename])

def generate_and_validate(filename: str, instruction: str, llm_call):
    if filename not in ALLOWED_FILES:
        return {"ok": False, "errors": [f"'{filename}'はホワイトリスト対象外です"]}

    with open(ALLOWED_FILES[filename], "r", encoding="utf-8") as f:
        old_content = f.read()

    messages = build_patch_prompt(old_content, instruction)
    raw = llm_call(messages)

    try:
        patches = parse_patch_response(raw)
    except Exception as e:
        return {"ok": False, "errors": [f"LLM応答のJSON解析に失敗: {e}"], "raw": raw}

    uniq_errors = validate_uniqueness(old_content, patches)
    if uniq_errors:
        return {"ok": False, "errors": uniq_errors, "patches": patches}

    new_content = apply_patches(old_content, patches)
    ok, syn_err = validate_syntax(new_content)
    if not ok:
        return {"ok": False, "errors": [f"構文チェック失敗: {syn_err}"], "patches": patches}

    diff = make_diff(old_content, new_content, filename)
    return {
        "ok": True,
        "patches": patches,
        "old_content": old_content,
        "new_content": new_content,
        "diff": diff,
    }


# ── 承認待ちパッチのDB管理 ──────────────────────────────────

def init_pending_patch_table(db_path):
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pending_patches (
            session_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            instruction TEXT NOT NULL,
            patches_json TEXT NOT NULL,
            old_content TEXT NOT NULL,
            new_content TEXT NOT NULL,
            diff TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    conn.close()

def save_pending_patch(db_path, session_id, filename, instruction, patches, old_content, new_content, diff):
    import sqlite3, json
    init_pending_patch_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM pending_patches WHERE session_id=?", (session_id,))
    conn.execute(
        "INSERT INTO pending_patches (session_id, filename, instruction, patches_json, old_content, new_content, diff) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (session_id, filename, instruction, json.dumps(patches, ensure_ascii=False), old_content, new_content, diff)
    )
    conn.commit()
    conn.close()

def get_pending_patch(db_path, session_id, timeout_seconds=600):
    import sqlite3, json, datetime
    init_pending_patch_table(db_path)
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT filename, instruction, patches_json, old_content, new_content, diff, created_at FROM pending_patches WHERE session_id=?",
        (session_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    filename, instruction, patches_json, old_content, new_content, diff, created_at = row
    try:
        created = datetime.datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
        age = (datetime.datetime.utcnow() - created).total_seconds()
        if age > timeout_seconds:
            delete_pending_patch(db_path, session_id)
            return None
    except Exception:
        pass
    return {
        "filename": filename,
        "instruction": instruction,
        "patches": json.loads(patches_json),
        "old_content": old_content,
        "new_content": new_content,
        "diff": diff,
    }

def delete_pending_patch(db_path, session_id):
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM pending_patches WHERE session_id=?", (session_id,))
    conn.commit()
    conn.close()


# ── 複数モデル試行（1モデルずつ検証しながらフォールバック） ──────────────

PATCH_CANDIDATE_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "openai/gpt-oss-20b:free",
]

def generate_and_validate_multi(filename: str, instruction: str, llm_call_single, models=None, max_tokens=4000):
    """
    llm_call_single: (model: str, messages: list) -> str
    候補モデルを順番に試し、JSON解析・一意性検証・構文チェックまで通った最初の結果を返す。
    全滅した場合はエラー一覧(各モデルごとの失敗理由)を返す。
    """
    if filename not in ALLOWED_FILES:
        return {"ok": False, "errors": [f"'{filename}'はホワイトリスト対象外です"]}

    with open(ALLOWED_FILES[filename], "r", encoding="utf-8") as f:
        old_content = f.read()

    messages = build_patch_prompt(old_content, instruction)
    models = models or PATCH_CANDIDATE_MODELS

    attempt_errors = []
    timings = []
    for model in models:
        _t0 = time.time()
        try:
            raw = llm_call_single(model, messages)
        except Exception as e:
            _elapsed = time.time() - _t0
            timings.append((model, _elapsed, "api_error"))
            attempt_errors.append(f"[{model}] API呼び出し失敗({_elapsed:.1f}秒): {e}")
            continue

        try:
            patches = parse_patch_response(raw)
        except Exception as e:
            _elapsed = time.time() - _t0
            timings.append((model, _elapsed, "json_error"))
            debug_dir = os.path.expanduser("~/ai-orchestrator/debug_raw")
            os.makedirs(debug_dir, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            safe_model = model.replace("/", "_").replace(":", "_")
            debug_path = os.path.join(debug_dir, f"{safe_model}_{ts}.txt")
            try:
                with open(debug_path, "w", encoding="utf-8") as _df:
                    _df.write(raw)
            except Exception:
                debug_path = "(保存失敗)"
            attempt_errors.append(f"[{model}] JSON解析失敗({_elapsed:.1f}秒): {e} (raw_len={len(raw)}, saved={debug_path})")
            continue

        uniq_errors = validate_uniqueness(old_content, patches)
        if uniq_errors:
            _elapsed = time.time() - _t0
            timings.append((model, _elapsed, "uniq_error"))
            attempt_errors.append(f"[{model}] 一意性検証失敗({_elapsed:.1f}秒): {'; '.join(uniq_errors)}")
            continue

        new_content = apply_patches(old_content, patches)
        ok, syn_err = validate_syntax(new_content)
        if not ok:
            _elapsed = time.time() - _t0
            timings.append((model, _elapsed, "syntax_error"))
            attempt_errors.append(f"[{model}] 構文チェック失敗({_elapsed:.1f}秒): {syn_err}")
            continue

        _elapsed = time.time() - _t0
        timings.append((model, _elapsed, "success"))
        diff = make_diff(old_content, new_content, filename)
        return {
            "ok": True,
            "model_used": model,
            "elapsed_seconds": _elapsed,
            "patches": patches,
            "old_content": old_content,
            "new_content": new_content,
            "diff": diff,
            "attempt_errors": attempt_errors,
            "timings": timings,
        }

    return {"ok": False, "errors": attempt_errors, "timings": timings}


# ── git連携 ──────────────────────────────────────

def find_git_root(path):
    """指定パスを含むgitリポジトリのルートを返す。リポジトリでなければNone"""
    try:
        r = subprocess.run(
            ["git", "-C", os.path.dirname(path), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=10
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return None

def git_commit_and_push(filepath, message, timeout=30):
    """
    filepathを含むgitリポジトリでadd→commit→pushを行う。
    gitリポジトリでない場合はエラー扱いせず skipped=True を返す。
    """
    repo_root = find_git_root(filepath)
    if not repo_root:
        return {"ok": False, "skipped": True, "reason": "gitリポジトリではありません"}

    def run(args):
        return subprocess.run(
            ["git", "-C", repo_root] + args,
            capture_output=True, text=True, timeout=timeout
        )

    add_r = run(["add", filepath])
    if add_r.returncode != 0:
        return {"ok": False, "skipped": False, "reason": f"git add失敗: {add_r.stderr.strip()}"}

    status_r = run(["status", "--porcelain", filepath])
    if not status_r.stdout.strip():
        return {"ok": True, "skipped": True, "reason": "変更なし(コミット対象なし)"}

    commit_r = run(["commit", "-m", message])
    if commit_r.returncode != 0:
        return {"ok": False, "skipped": False, "reason": f"git commit失敗: {commit_r.stderr.strip()}"}

    push_r = run(["push"])
    if push_r.returncode != 0:
        return {"ok": False, "skipped": False, "reason": f"git push失敗（コミットは成功済み・要手動push）: {push_r.stderr.strip()}"}

    return {"ok": True, "skipped": False, "reason": "commit+push成功"}
