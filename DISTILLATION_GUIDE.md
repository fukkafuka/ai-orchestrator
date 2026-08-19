# ローカルLLM再蒸留ガイド

`llm-jp-3-1.8b-instruct3` をLoRAでファインチューニングし直すときの手順書。
会話するだけではローカルモデルの重みは更新されないため、このサイクルを
定期的（目安: 月1回程度、または応答の質が気になったタイミング）に回す。

## 全体の流れ

```
① 候補抽出 → ② 模範解答作成 → ③ LoRA再学習 → ④ マージ → ⑤ gguf変換 → ⑥ デプロイ → ⑦ 動作確認
```

---

## ① 候補抽出（Mac側）

`cache.db` から、ローカルモデルが処理した質問のうち、まだ `distill_claude_authored.jsonl`
に含まれていないものを抽出する。

```bash
cd ~/ai-orchestrator
python3 extract_distill_candidates.py --limit 50
```

内容を見て、実際によく聞く・答えの質が気になるパターンを中心に
20〜50件くらいピックアップする。全件使う必要はない。

JSON形式で保存してClaudeに渡したい場合:

```bash
python3 extract_distill_candidates.py --json --limit 50 > /tmp/candidates.json
cat /tmp/candidates.json
```

## ② 模範解答作成（Claudeとのチャット）

①で出た候補（またはそのまま貼った `cat /tmp/candidates.json` の出力）を
このチャットに貼り、「この質問リストに対して模範解答を作って」と依頼する。

Claudeは `orchestrator_v4.py` の `ask_local()` に埋め込まれているsystemプロンプトの方針
（結論を先に・断定しない・設計判断を否定しない・ログ診断の考え方など）に沿った
`{"instruction": ..., "output": ...}` 形式のJSONLを作成し、
`distill_claude_authored.jsonl` に直接追記・commit・pushする（方式A、PAT必要）。

## ③ LoRA再学習（Google Colab）

`distill_claude_authored.jsonl` を最新化してからColabノートブックで学習を回す。

```bash
cd ~/ai-orchestrator && git pull
```

- ベースモデル: `llm-jp-3-1.8b-instruct3`
- 学習データ: `distill_claude_authored.jsonl`（`instruction`/`output`形式）
- 過去に使ったノートブックをそのまま流用（学習率・epoch数などのハイパラは
  前回と同じ設定を出発点にし、データ件数が増えた分だけepoch数を様子見で調整する）

## ④ マージ

学習済みLoRAアダプタをベースモデルにマージし、1つのモデルにまとめる。

## ⑤ gguf変換

```bash
# 量子化形式は Q4_K_M を踏襲（他のツール類との互換性・速度のバランスが良いため）
```

llama.cpp付属の変換スクリプトで `.gguf` に変換し、量子化する。

## ⑥ デプロイ

```bash
cd ~/ai-orchestrator
# 新しいggufファイルを配置(既存ファイルは念のためリネームして残しておく)
mv llm-jp-3-1.8b-instruct3-Q4_K_M.gguf llm-jp-3-1.8b-instruct3-Q4_K_M.gguf.bak_$(date +%Y%m%d)
mv <新しいggufファイル> llm-jp-3-1.8b-instruct3-Q4_K_M.gguf
launchctl kickstart -k gui/$(id -u)/com.fk.orchestrator
sleep 30
lsof -i :11437 | grep LISTEN
```

## ⑦ 動作確認

- プレフィックスなしで、①で拾った質問のうち代表的なものをいくつか投げてみる
- 応答が壊れていないか（変な繰り返し・文字化けがないか）を確認
- 明らかに悪化していたら `.bak_*` にリネームした旧ファイルへ戻す

```bash
mv llm-jp-3-1.8b-instruct3-Q4_K_M.gguf llm-jp-3-1.8b-instruct3-Q4_K_M.gguf.new_$(date +%Y%m%d)
mv llm-jp-3-1.8b-instruct3-Q4_K_M.gguf.bak_YYYYMMDD llm-jp-3-1.8b-instruct3-Q4_K_M.gguf
launchctl kickstart -k gui/$(id -u)/com.fk.orchestrator
```

---

## メモ

- `distill_claude_authored.jsonl` の件数: 2026-08-17時点で54件（39件→+15件）
- 蒸留データはsystemプロンプトの方針と一貫性を持たせること
  （プロンプト側の指針とファインチューニング側の口調がズレていると効果が薄れる）
- Colab側の具体的なハイパーパラメータ設定は本ガイドには含めていない
  （ノートブック側で管理。次回実行時に値をこのファイルに追記してもよい）
