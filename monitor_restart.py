#!/usr/bin/env python3
import sys, os, time, shutil, subprocess, argparse, datetime, json
import requests
requests.packages.urllib3.disable_warnings()

sys.path.insert(0, os.path.expanduser("~/ai-orchestrator"))
import auto_patch

LOG_FILE = "/Users/fk/Logs/orc.log"
VERSION_URL = "https://127.0.0.1:11437/version"
SERVICE = "com.fk.orchestrator"

def log(msg):
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    line = f"🩺[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass

def check_version():
    try:
        r = requests.get(VERSION_URL, verify=False, timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--before-boot", required=True)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--meta", default=None)
    args = ap.parse_args()

    log(f"🩺 監視開始: target={args.target} backup={args.backup} before_boot={args.before_boot} timeout={args.timeout}s")

    # kickstartの発火(3秒後)を見込んだ猶予を追加
    deadline = time.time() + args.timeout + 15
    healthy = False
    while time.time() < deadline:
        info = check_version()
        if info and info.get("boot_time") != args.before_boot:
            log(f"✅ ヘルスチェック成功: 新boot_time={info.get('boot_time')} code_hash={info.get('code_hash')}")
            healthy = True
            break
        time.sleep(2)

    if healthy:
        if args.meta and os.path.exists(args.meta):
            try:
                with open(args.meta, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                commit_msg = f"auto-patch(self): {meta['filename']} - {meta['instruction'][:80]}"
                git_result = auto_patch.git_commit_and_push(args.target, commit_msg)
                if git_result["skipped"]:
                    log(f"ℹ️ git連携スキップ: {git_result['reason']}")
                elif git_result["ok"]:
                    log(f"✅ git commit+push成功: {meta['filename']}")
                else:
                    log(f"❌ git連携失敗: {git_result['reason']}")
            except Exception as e:
                log(f"❌ git連携処理中にエラー: {e}")
            finally:
                try:
                    os.remove(args.meta)
                except Exception:
                    pass
        return

    log(f"❌ ヘルスチェック失敗（{args.timeout+15}秒でタイムアウト）→ ロールバック実行")
    if args.meta and os.path.exists(args.meta):
        try:
            os.remove(args.meta)
        except Exception:
            pass
    try:
        shutil.copy2(args.backup, args.target)
        log(f"📦 ロールバック完了: {args.backup} → {args.target}")
    except Exception as e:
        log(f"❌ ロールバックのファイルコピーに失敗: {e}")
        return

    try:
        target_spec = f"gui/{os.getuid()}/{SERVICE}"
        subprocess.run(["launchctl", "kickstart", "-k", target_spec], capture_output=True, timeout=10)
        log(f"🔄 ロールバック後の再起動を実行: {target_spec}")
    except Exception as e:
        log(f"❌ ロールバック後の再起動コマンド失敗: {e}")
        return

    deadline2 = time.time() + 30
    while time.time() < deadline2:
        info = check_version()
        if info:
            log(f"✅ ロールバック後の起動確認OK: code_hash={info.get('code_hash')}")
            return
        time.sleep(2)
    log("❌ ロールバック後も起動確認できませんでした。手動確認が必要です")

if __name__ == "__main__":
    main()
