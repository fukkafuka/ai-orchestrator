import ast, json, re, os, shutil, subprocess, datetime, difflib, time

ALLOWED_FILES = {
    "orchestrator_v4.py": "/Users/fk/ai-orchestrator/orchestrator_v4.py",
    "agent_claude.py": "/Users/fk/ai-agent/moltbook/agent_claude.py",
    "agent_log_doctor.py": "/Users/fk/ai-agent/moltbook/agent_log_doctor.py",
}

BACKUP_DIR_MAP = {
    "orchestrator_v4.py": "/Users/fk/ai-orchestrator/",
    "agent_claude.py": "/Users/fk/ai-agent/moltbook/backups/",
    "agent_log_doctor.py": "/Users/fk/ai-agent/moltbook/backups/",
}

DEBUG_RAW_DIR = "/Users/fk/ai-orchestrator/debug_raw/"

# 7/20時点の並び(nemotron-3-super-120b-a12bが先頭)。
# nemotron対応(b)の並び替えは次ステップで別途適用予定。
PATCH_CANDIDATE_MODELS = [
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "openai/gpt-oss-20b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
]

PATCH_SYSTEM_PROMPT = """あなたはPythonコードの修正パッチを生成するアシスタントです。
与えられたファイル内容と修正指示から、修正箇所をJSONオブジェクトで返してください。
出力はJSONオブジェクトのみ。説明文やMarkdownのコードブロック記号は一切含めないこと。
形式: {"patches": [{"old_str": "元のコードの一意な一部分", "new_str": "置き換え後のコード", "reason": "変更理由の短い説明"}]}
old_strはファイル内で一意に一箇所だけに一致する、十分な長さの文字列にすること。
コメントや説明文は一切出力せず、JSONオブジェクトのみを返すこと。
"""

REFINE_SYSTEM_PROMPT = """あなたはパッチ生成の指示文を改善するアシスタントです。
以下の修正指示でパッチ生成が失敗しました。ファイル内容と失敗理由を踏まえ、
次回の試行で成功しやすいように、より具体的で明確な日本語の修正指示を1つだけ出力してください。
出力は日本語の指示文のみとし、前置きや説明、英語の思考過程は一切含めないこと。
"""


def build_patch_prompt(file_content: str, instruction: str) -> list:
    return [
        {"role": "system", "content": PATCH_SYSTEM_PROMPT},
        {"role": "user", "content": f"### 対象ファイル内容\n{file_content}\n\n### 修正指示\n{instruction}"}
    ]


def build_refine_prompt(file_content: str, instruction: str, error_summary: str) -> list:
    return [
        {"role": "system", "content": REFINE_SYSTEM_PROMPT},
        {"role": "user", "content": f"### 対象ファイル内容(抜粋)\n{file_content[:4000]}\n\n### 元の修正指示\n{instruction}\n\n### 失敗理由\n{error_summary}"}
    ]


def _extract_json_objects(raw: str):
    raw = raw.strip()
    raw = re.sub(r"^```(json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    start = raw.find("{")
    if start == -1:
        raise ValueError("JSONオブジェクトの開始位置が見つかりません")
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                return raw[start:i + 1]
    raise ValueError("JSONオブジェクトの終端(閉じ括弧)が見つかりません")


def _extract_japanese_instruction(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    jp_lines = [l for l in lines if re.search(r"[ぁ-んァ-ン一-龥]", l)]
    if jp_lines:
        return "\n".join(jp_lines)
    return text.strip()


def parse_patch_response(raw: str):
    obj_str = _extract_json_objects(raw)
    obj = json.loads(obj_str)
    if not isinstance(obj, dict) or "patches" not in obj:
        raise ValueError('パッチ形式が不正です({"patches": [...]}形式ではない)')
    patches = obj["patches"]
    if not isinstance(patches, list):
        raise ValueError("patchesが配列ではありません")
    for p in patches:
        if "old_str" not in p or "new_str" not in p:
            raise ValueError("old_str/new_strが欠落しているパッチがあります")
    return patches


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


def _save_debug_raw(model: str, raw: str):
    try:
        os.makedirs(DEBUG_RAW_DIR, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = model.replace("/", "_").replace(":", "_")
        path = os.path.join(DEBUG_RAW_DIR, f"{safe_model}_{ts}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(raw)
    except Exception:
        pass


def _try_one_model(model: str, old_content: str, instruction: str, call_model_fn, timings: list):
    messages = build_patch_prompt(old_content, instruction)
    t0 = time.time()
    try:
        raw = call_model_fn(messages, model=model, response_format="json_object")
    except Exception as e:
        elapsed = round(time.time() - t0, 1)
        timings.append((model, elapsed, "呼び出し失敗"))
        return {"ok": False, "errors": [f"{model}: API呼び出し失敗: {e}"]}
    elapsed = round(time.time() - t0, 1)

    try:
        patches = parse_patch_response(raw)
    except Exception as e:
        timings.append((model, elapsed, "JSON解析失敗"))
        _save_debug_raw(model, raw)
        return {"ok": False, "errors": [f"{model}: LLM応答のJSON解析に失敗: {e}"], "raw": raw}

    uniq_errors = validate_uniqueness(old_content, patches)
    if uniq_errors:
        timings.append((model, elapsed, "一意性検証失敗"))
        return {"ok": False, "errors": [f"{model}: {e}" for e in uniq_errors], "patches": patches}

    new_content = apply_patches(old_content, patches)
    ok, syn_err = validate_syntax(new_content)
    if not ok:
        timings.append((model, elapsed, "構文チェック失敗"))
        return {"ok": False, "errors": [f"{model}: 構文チェック失敗: {syn_err}"], "patches": patches}

    timings.append((model, elapsed, "成功"))
    return {
        "ok": True,
        "model": model,
        "patches": patches,
        "old_content": old_content,
        "new_content": new_content,
    }


def generate_and_validate_multi(filename: str, instruction: str, call_model_fn,
                                 models=None, max_refinements=2):
    if filename not in ALLOWED_FILES:
        return {"ok": False, "errors": [f"'{filename}'はホワイトリスト対象外です"], "timings": []}

    with open(ALLOWED_FILES[filename], "r", encoding="utf-8") as f:
        old_content = f.read()

    models = models or PATCH_CANDIDATE_MODELS
    timings = []
    current_instruction = instruction
    all_errors = []

    for round_idx in range(max_refinements + 1):
        for model in models:
            result = _try_one_model(model, old_content, current_instruction, call_model_fn, timings)
            if result["ok"]:
                result["diff"] = make_diff(old_content, result["new_content"], filename)
                result["timings"] = timings
                result["refinement_rounds"] = round_idx
                return result
            all_errors.extend(result["errors"])

        if round_idx < max_refinements:
            error_summary = "\n".join(all_errors[-6:])
            try:
                refine_messages = build_refine_prompt(old_content, current_instruction, error_summary)
                t0 = time.time()
                refined_raw = call_model_fn(refine_messages, model=models[0])
                elapsed = round(time.time() - t0, 1)
                timings.append((models[0], elapsed, "精密化"))
                current_instruction = _extract_japanese_instruction(refined_raw)
            except Exception as e:
                timings.append((models[0], 0, f"精密化失敗:{e}"))

    return {"ok": False, "errors": all_errors, "timings": timings}
