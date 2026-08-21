# ローカルLLM再蒸留ガイド

`llm-jp-3.1-1.8b-instruct4`（2026-08-19〜、旧`llm-jp-3-1.8b-instruct3`から乗り換え）を
LoRAでファインチューニングし直すときの手順書。
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

`distill_claude_authored.jsonl` と `orchestrator_lora_distill.ipynb` を最新化してから
Colabノートブックで学習を回す。

```bash
cd ~/ai-orchestrator && git pull
```

- ベースモデル: `llm-jp/llm-jp-3.1-1.8b-instruct4`（同1.8Bサイズで日本語MT-Bench 4.64→6.30）
- 学習データ: `distill_claude_authored.jsonl`（`instruction`/`output`形式）
- ノートブック: `orchestrator_lora_distill.ipynb`（学習率・epoch数などのハイパラは
  ノートブック内の値を出発点にし、データ件数が増えた分だけepoch数を様子見で調整する）

## ④ マージ

学習済みLoRAアダプタをベースモデルにマージし、1つのモデルにまとめる。
（ノートブックの6.マージのセルで実施済み）

## ⑤ gguf変換

量子化形式は `Q4_K_M` を踏襲（他のツール類との互換性・速度のバランスが良いため）。
ノートブック内でllama.cppをclone・ビルドし、GGUF(f16)変換 → Q4_K_M量子化まで実施する。
出力ファイル名は `llm-jp-3.1-1.8b-instruct4-Q4_K_M.gguf`
（旧モデル `llm-jp-3-1.8b-instruct3-Q4_K_M.gguf` とは別名にしてあるので、
そのまま両方を残しておける＝これ自体がロールバック手段になる）。

## ⑥ デプロイ

**旧ファイルは削除せず、新ファイルを並べて配置するだけ。**

```bash
cd ~/ai-orchestrator/llama.cpp/models
# ダウンロードした llm-jp-3.1-1.8b-instruct4-Q4_K_M.gguf をこのディレクトリに置く
ls -la  # llm-jp-3-1.8b-instruct3-Q4_K_M.gguf (旧) と
        # llm-jp-3.1-1.8b-instruct4-Q4_K_M.gguf (新) が両方あることを確認
```

`orchestrator_v4.py` の `LOCAL_MODEL_PATH` を新ファイルに向ける（Claudeがパッチとして提示）:

```python
# 変更前
LOCAL_MODEL_PATH = os.path.expanduser("~/ai-orchestrator/llama.cpp/models/llm-jp-3-1.8b-instruct3-Q4_K_M.gguf")
# 変更後
LOCAL_MODEL_PATH = os.path.expanduser("~/ai-orchestrator/llama.cpp/models/llm-jp-3.1-1.8b-instruct4-Q4_K_M.gguf")
```

```bash
launchctl kickstart -k gui/$(id -u)/com.fk.orchestrator
sleep 30
lsof -i :11437 | grep LISTEN
```

## ⑦ 動作確認

- プレフィックスなしで、①で拾った質問のうち代表的なものをいくつか投げてみる
- 応答が壊れていないか（変な繰り返し・文字化けがないか）を確認
- 明らかに悪化していたら、**`LOCAL_MODEL_PATH` を旧ファイル名に戻すだけ**でよい
  （旧ファイルはそのまま残っているので、ファイルのリネームや復元操作は不要）

```python
# ロールバック: 1行だけ戻す
LOCAL_MODEL_PATH = os.path.expanduser("~/ai-orchestrator/llama.cpp/models/llm-jp-3-1.8b-instruct3-Q4_K_M.gguf")
```

```bash
launchctl kickstart -k gui/$(id -u)/com.fk.orchestrator
```

問題なければ、旧ファイルは念のためしばらく残しておき、次回サイクルが安定してから削除する。

---

## メモ

- ベースモデル: 2026-08-19に `llm-jp-3-1.8b-instruct3` → `llm-jp-3.1-1.8b-instruct4` へ乗り換え
- `distill_claude_authored.jsonl` の件数: 2026-08-17時点で54件（39件→+15件）
- 蒸留データはsystemプロンプトの方針と一貫性を持たせること
  （プロンプト側の指針とファインチューニング側の口調がズレていると効果が薄れる）
- LoRAの`target_modules`はLlama系アーキテクチャ共通のため、instruct3→instruct4の
  乗り換えでも変更不要（q/k/v/o_proj + gate/up/down_proj）

