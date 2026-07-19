import sys, os
sys.path.insert(0, "/Users/fk/ai-orchestrator")
import auto_patch

from orchestrator_v4 import call_openrouter, MODEL_CLOUD

def llm_call(messages):
    return call_openrouter(MODEL_CLOUD, messages, max_tokens=1500, temperature=0.3)

if __name__ == "__main__":
    filename = "agent_log_doctor.py"  # テスト対象
    instruction = "extract_unclassified_patterns()関数に、処理開始時のログ出力(print)を1行追加してください。内容は '[unclassified] scan start' とする"

    result = auto_patch.generate_and_validate(filename, instruction, llm_call)

    if not result["ok"]:
        print("=== 失敗 ===")
        for e in result["errors"]:
            print(" -", e)
        if "raw" in result:
            print("--- LLM生の応答 ---")
            print(result["raw"])
        sys.exit(1)

    print("=== パッチ生成成功 ===")
    for i, p in enumerate(result["patches"]):
        print(f"[{i+1}] {p.get('reason','(理由なし)')}")
    print("\n=== diff ===")
    print(result["diff"])
    print("\n※ この時点ではファイルは書き換えていません(承認フロー未実装のため)")
