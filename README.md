# AI Orchestrator

MacBook Air(Intel, 4GB RAM)上で動く、個人用AIチャット・オーケストレーター。Tailscale経由でiPhone/PCからHTTPSアクセスし、ローカルLLM・クラウド無料枠モデル・自動コード修正(auto_patch)を統合している。

- 本体: `orchestrator_v4.py`(Flask, port 11437)
- 稼働: `launchd`(`com.fk.orchestrator`)常駐
- アクセス: `https://100.109.207.78:11437/`(Tailscale内のみ)

## 主な機能

### 1. チャット(ローカル / クラウド / 過去会話検索)

入力の先頭記号でルーティングを切り替える。

| プレフィックス | 動作 |
|---|---|
| なし | ローカルLLM(`llm-jp-3-1.8b-instruct3`, llama.cpp)で応答。外部通信なし |
| `。` / `.` | クラウド(OpenRouter無料枠モデル)で応答 |
| `。。。` / `...` | 複数モデルに並列問い合わせ |
| `!` / `！` | キャッシュを無視して再生成 |
| `?` / `？` | 過去の関連会話を検索して回答に活用 |

ローカル推論は `llama.cpp/build/bin/llama-completion` を `--jinja --single-turn` 付きで呼び出し、モデル埋め込みのチャットテンプレートを正しく適用する(このフラグが無いとテンプレート未適用で無関係な長文を生成することがある)。

### 2. セッション管理

- ブラウザの `sessionStorage` にセッションIDを保持(タブごとに独立)
- ヘッダーの「📋 セッション」ボタンから、セッション一覧の閲覧・切替・改名が可能
- 「🗑️ 履歴クリア」でメモリ・DB両方の該当セッション履歴を削除(`/session/clear`)
- 8桁の短縮コードで別セッションへの会話引き継ぎが可能

### 3. auto_patch(自動コード修正)

「〇〇のログを確認して対応して」「(ファイル名)の△△を修正して」等の依頼、または `、`/`,` プレフィックスで起動。

1. 診断モード(ログ収集→対象ファイル・修正指示をLLMが提案、指定があればスキップ)
2. 複数モデル(`auto_patch.PATCH_CANDIDATE_MODELS`)を順次試行してパッチ生成(JSON形式)
3. old_strの一意性検証・構文チェック
4. 失敗時は指示を精密化して再試行(最大2ラウンド)
5. diffを提示し、「承認」で適用(自動バックアップ→書き込み→構文再検証→git commit & push)

対象ファイルは `auto_patch.ALLOWED_FILES` のホワイトリストのみ(ai-orchestrator / moltbook-agent / MythoFableの主要ファイル)。

## ファイル構成

```
orchestrator_v4.py   # 本体(Flask, チャットUI, ルーティング, auto_patch呼び出し)
auto_patch.py         # パッチ生成・検証・承認待ち管理・git連携
monitor_restart.py    # 自己修正(orchestrator_v4.py自身への修正)時の再起動監視
test_auto_patch.py    # auto_patchのテスト
prompts/               # プロンプト関連
tools/                  # 補助ツール
distill_claude_authored.jsonl  # ローカルLLM蒸留用の教師データ
run_agent.sh, deploy.sh 等  # 運用スクリプト
```

## 主要な設定・パス

- APIキー: `~/.config/ai-keys/.env`(正本。他の場所に重複させない)
- ログ: `/Users/fk/Logs/orc.log`
- キャッシュ/会話履歴DB: `cache.db`(sqlite3, `conversations` / `cache` / `pending_patches` / `session_names` テーブル)
- ローカルモデル: `llama.cpp/models/llm-jp-3-1.8b-instruct3-Q4_K_M.gguf`

## 運用コマンド

```bash
# 再起動
launchctl kickstart -k gui/$(id -u)/com.fk.orchestrator

# バージョン確認(起動確認・code_hashで反映確認)
curl -s -k https://100.109.207.78:11437/version

# 構文チェック(反映前に必須)
python3 -m py_compile orchestrator_v4.py
```

## 既知の課題

- OpenRouter無料枠モデル(特に大きいファイルへのパッチ生成)は、response_formatを指定しても英語の思考過程を出力し続けタイムアウトすることがある
- 大きいファイルは差分箇所のみ抜粋して渡す方式は未実装(残タスク)

詳細な経緯・トラブルシューティングの記録はClaude(このリポジトリを直接編集するチャットアシスタント)側のメモリに蓄積されている。
