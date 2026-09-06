[English](README.md) | **日本語**

# SABM: LLM エージェントによるベルトラン複占の企業競争・暗黙の共謀 — Han, Wu & Xiao (2023)

Han, Wu & Xiao (2023) [「Guinea Pig Trials Utilizing GPT: A Novel Smart Agent-Based Modeling Approach for Studying Firm Competition and Collusion」](https://arxiv.org/abs/2308.10974) (arXiv:2308.10974) の Smart Agent-Based Modeling (SABM) フレームワークの再現実装．2 つの LLM「企業」エージェントが差別化財の**ベルトラン複占**繰り返し価格ゲームを行う．各ラウンドで両社は*同時に*単価を設定し，線形需要関数が市場をクリアして各社が利益を得る．LLM はラウンド間で記憶を持たないため，境界付き 20 ラウンド履歴を毎ラウンド再注入し，20 ラウンドごとに反省 (reflection) を挟む．論文の核心的知見は，**会話がなくても** 2 つの LLM 企業が一貫して**暗黙の共謀 (tacit collusion)** に到達することである．価格は競争的なベルトラン均衡より*高く*，カルテル/独占価格より*低い*水準 (基本設定でベルトラン=6・カルテル=8 に対して約 7) に収束する．決定論的な [socsim](https://github.com/akitenkrad/rs-social-simulation-tools) コアが解析ベンチマーク・市場クリアリング・利益・境界付きメモリ更新を担い，非決定的な LLM レイヤは価格決定メカニズム 1 つに閉じ込められ，`socsim-llm` クレット (プロンプト→応答キャッシュ + `temperature=0` + 固定 seed) で擬似決定論化される．

## 二層決定論 (最初に読む)

LLM 出力は socsim の bit 再現性の **外側** にある．そこで設計を二層に分ける:

- **決定論的 socsim コア** — 解析的なベルトラン均衡価格・カルテル/独占価格 (`p^B`, `p^M`)，線形需要の市場クリアリング (`q_i = (α − β p_i + d Σ_{j≠i} p_j) / b`)，利益 (`π_i = (p_i − c_i) q_i`)，collusion index (`CI = (p − p^B) / (p^M − p^B)`)，収束/有界振動の停止判定，境界付き 20 ラウンドメモリ更新．seed を固定すれば bit 単位で再現する (ChaCha20 `SimRng`)．解析パスは**定量的に厳密**で，基本設定では `p^B = 6`, `p^M = 8`．
- **非決定的 LLM レイヤ** — `Decision` メカニズム 1 つ (`PricingDecision`)．各社の次ラウンド価格を，境界付き履歴・ペルソナ・基準価格から LLM が決める．`socsim-llm` の `CachingClient` (`hash(prompt+model)` → 応答キャッシュ)・`temperature=0`・固定 seed で擬似決定論化する．プロバイダ順は `socsim-llm` の `FallbackClient` による **Ollama 第一 → OpenAI フォールバック**．

再現性の本体はモデルではなく**キャッシュ**である．ウォームキャッシュは同一応答を再生するため，再実行は無料かつ安定する．各実行はモデル・provider・温度を runvault の `run.json` の `llm` ブロックに，呼び出し数と cache-hit 率を run スコープの指標として記録する．ローカル既定モデル (`llama3.2`) は論文の `gpt-4-0314` と異なるため，LLM 駆動の再現目標は**定性的** (価格が (ベルトラン, カルテル) 区間内へ収束し CI ∈ [0.3, 0.8]) とする．解析ベンチマークは厳密である．

> 本プロジェクトは LLM レイヤを `socsim-llm` クレットに統一し，`reqwest` / `sha2` は使わない (HTTP とプロンプトキャッシュのハッシュは socsim-llm が所有する)．モデルは市場媒介 — 非空間・非ネットワーク — なので `socsim-core` + `socsim-engine` + `socsim-llm` のみに依存する (`socsim-grid` / `socsim-net` 不要)．

## インストールとクイックスタート

```bash
# Rust シミュレーションをビルド (socsim と socsim-llm の Ollama+OpenAI バックエンドを取得)
cargo build --release

# === 解析ベンチマークのみ (LLM 不要・即時): 基本設定 → p^B=6, p^M=8 ===
cargo run --release -- benchmark --a 14 --beta 0.0066667 --d 0.0033333 --c1 2 --c2 2
# results/sabm/ 配下に runvault の run を書き出す

# ローカル Ollama を起動しモデルを取得 (例):
#   ollama pull llama3.2:latest
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=llama3.2:latest
# OpenAI フォールバック (任意):
#   export OPENAI_API_KEY=sk-...   OPENAI_MODEL=gpt-4o-mini

# 基本実験 (会話なし・active ペルソナ): 暗黙の共謀を観測
cargo run --release -- run --firms 2 --persona active --max-rounds 1000 --temperature 0 --seed 42

# Python 可視化ツールをインストール (workspace ルートで)
uv sync

# 直近実行の可視化 (価格軌跡 + collusion index, ベルトラン/カルテル基準線つき)
uv run sabm-tools visualize

# 実行設定と LLM メタデータの確認
uv run sabm-tools show-experiment-settings
```

### オフライン (LLM 不要) スモーク

ラウンドループ・出力ライタ・Python 可視化を，scripted mock クライアントでライブ LLM なしに検証できる (CI・ネットワーク遮断サンドボックス用):

```bash
# 専用 example: 暗黙の共謀風の scripted ポリシー
cargo run --release --example mock_smoke -- results

# あるいは run / sweep に --mock を渡すと同じオフライン挙動
cargo run --release -- run --firms 2 --max-rounds 60 --seed 42 --mock
uv run sabm-tools visualize
```

### 感度分析スイープ

```bash
cargo run --release -- sweep \
    --d-beta-values 0.0,0.25,0.5,0.75,1.0 \
    --firms-values 2,3 \
    --max-rounds 400 --runs 5 --seed 42
uv run sabm-tools visualize-sweep
```

### 論文再現 (Fig.1/2/4)

`reproduce` は会話なし baseline (Fig.1) と会話あり変種 (Fig.2) を一括実行し，観測した平均価格と collusion index をベルトラン(6)/カルテル(8) フレームと照合して 観測 avg_price / collusion index を run スコープの指標に，PASS/off の判定を `x.han2023.anchor` イベントに記録する．Python `reproduce` ツールが headline 図を `results/sabm/figures/<run_slug>/` に描画する．

```bash
# 実 LLM で end-to-end (オフラインなら --mock; --quick で 80 ラウンドに短縮)
cargo run --release -- reproduce --seed 42
uv run sabm-tools reproduce            # fig1/fig2/fig4 を描画し PASS テーブルを表示

# オフライン (LLM 不要) 再現を 1 コマンドで
uv run sabm-tools reproduce --run --mock --quick
```

### 会話あり変種・ペルソナ / 非対称コスト変種

```bash
# 「会話あり」: 各ラウンドで価格決定前に企業が cheap-talk メッセージを交換する
# (CommunicationPhase)．--communication はフラグ型 (既定 = 会話なし)．
cargo run --release -- run --communication --persona active --max-rounds 1000 --seed 42

# ペルソナ選択 (active / aggressive / none) と非対称限界費用 (c2 ≠ c1):
# 高コスト社ほど高い価格に収束する (非対称コスト → 非対称価格)．
cargo run --release -- run --persona aggressive --c1 2 --c2 5 --max-rounds 1000 --seed 42
```

## スコープ

本リポジトリは SABM モデルを全面的に実装している: `MarketWorld` + 6-phase ループ上のメカニズム，解析的なベルトラン/カルテルベンチマーク，Ollama→OpenAI フォールバック + キャッシュの LLM 価格決定レイヤ，暗黙の共謀 / collusion index 指標，製品差別化度 `d/β` × 企業数 の `sweep`，**会話あり**変種 (価格決定前に cheap-talk メッセージを交換する `CommunicationPhase`; `--communication` で切替),企業ごとの**ペルソナ**と**非対称限界費用** (`c2 ≠ c1`),論文 Fig.1/2/4 一括再現 (`reproduce`; PASS/off アンカー付き)．Python `sabm-tools` は `visualize` / `visualize-sweep` / `show-experiment-settings` / `reproduce` を提供する．既定経路 (会話なし・対称コスト・`active` ペルソナ) は論文の基本ケースであり，同一シードでは変種追加前のコアと bit 等価である．

## ドキュメント

- [ユースケース](docs/usecases.ja.md) — 本プロジェクトでできること，他ドキュメントへの導線．
- [CLI](docs/cli.ja.md) — Rust CLI: `benchmark` / `run` / `sweep` / `reproduce` サブコマンドとフラグ，LLM 環境変数．
- [可視化](docs/visualization.ja.md) — Python `sabm-tools` と出力の解釈．
- [アーキテクチャ](docs/architecture.ja.md) — リポジトリ構成・二層決定論・socsim/`socsim-llm` 基盤・メカニズム・需要/ベルトラン/カルテル式・指標・参考文献．
- [再現](docs/reproduction.ja.md) — 定量的目標と，解析/LLM 駆動再現の評価方法．

## ライセンス

MIT
