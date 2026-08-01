"""
無料枠モデルの生存状況を一元管理する共通ヘルパー。
~/.config/ai-keys/model_status.json を正本とし、各リポジトリのフォールバックリストから
死んでいると判明済みのモデルを自動的に除外する。

使い方:
    from model_status import filter_alive_models
    OPENROUTER_FALLBACK_MODELS = filter_alive_models([
        "openai/gpt-oss-20b:free",
        "meta-llama/llama-3.3-70b-instruct:free",  # dead判定されていれば自動的に除外される
        ...
    ], provider="openrouter")

model_status.jsonが存在しない・壊れている場合は、渡されたリストをそのまま返す(fail-open、
モデルフィルタの失敗で本来の処理自体が止まらないようにするため)。
"""
import json
import os

_STATUS_PATH = os.path.expanduser("~/.config/ai-keys/model_status.json")
_DEAD_STATUSES = {"dead", "deprecated_2026-08-16", "paid_only"}


def _load_status():
    try:
        with open(_STATUS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def filter_alive_models(models, provider):
    """modelsのうち、model_status.json上でdead/deprecated/paid_onlyと判定されているものを除外する。
    未登録のモデルはそのまま残す(unverified扱い、除去はしない)。"""
    status = _load_status()
    provider_status = status.get(provider, {})
    result = [m for m in models if provider_status.get(m) not in _DEAD_STATUSES]
    return result


def get_model_status(model, provider):
    """特定モデルの生存状況を返す(unverified/alive/dead等)。未登録ならNone。"""
    status = _load_status()
    return status.get(provider, {}).get(model)
