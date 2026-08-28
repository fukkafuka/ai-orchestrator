#!/usr/bin/env python3
"""OpenRouterの候補モデルが実際に生きているか1つずつ確認する使い捨てスクリプト"""
import os
import requests
import dotenv

dotenv.load_dotenv(os.path.expanduser("~/.config/ai-keys/.env"))
API_KEY = os.environ.get("OPENROUTER_API_KEY")

CANDIDATES = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "openai/gpt-oss-20b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "openrouter/free",
]

for m in CANDIDATES:
    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={
                "model": m,
                "messages": [{"role": "user", "content": "こんにちは、1文字だけ「はい」と返答してください"}],
                "max_tokens": 20,
            },
            timeout=20,
        )
        data = r.json()
        if "choices" in data:
            content = data["choices"][0]["message"]["content"]
            print(f"✅ {m}: OK -> {content[:30]!r}")
        else:
            err = data.get("error", {}).get("message", str(data))
            print(f"❌ {m}: {err}")
    except Exception as e:
        print(f"❌ {m}: 例外 {e}")
