[English](visualization.md) | [日本語](visualization.ja.md)

# 可視化 (`sabm-tools`)

Python ツールは `tools/` (module `sabm_tools`)．workspace ルートで `uv sync` し，`uv run sabm-tools <サブコマンド>` で実行する．Rust シミュレーションが書いた runvault の run ディレクトリを読む．

どの run を見るかは runvault が答える．`--results-dir` を省略すれば `runvault path --latest` に聞く — `results/` を走査して新しそうなディレクトリを当てにいくことはしない．図は run ディレクトリの *隣* (`results/sabm/figures/{run_slug}/`) に置く．`manifest.csv` は `finish()` が確定させたもので，後から足したものはハッシュを持たないためである．

## `visualize` — 単一実行

```bash
uv run sabm-tools visualize
uv run sabm-tools visualize --results-dir "$(runvault path --experiment sabm --latest --subcommand run --standalone)"
```

企業ごとの価格軌跡を run の `observation` イベントから，ラウンドごとの市場指標を `metrics.csv` から (`runvault.read.metrics_wide` で 1 ラウンド 1 行に倒す)，解析フレームを run スコープの `p_bertrand_mean` / `p_cartel_mean` から読み，`results/sabm/figures/{run_slug}/` へ出力:

- **`price_trajectory.png`** — 各社の価格と全社平均のラウンド推移．ベルトラン均衡 (`p^B`)・カルテル (`p^M`) の破線基準線と (ベルトラン, カルテル) 帯の網掛けつき．Fig.1/2 風の図で，平均価格が帯の*内側*に収束すると暗黙の共謀が見える．
- **`collusion_index.png`** — collusion index 時系列 `CI = (p − p^B) / (p^M − p^B)`．0 (ベルトラン)・1 (カルテル) の基準線と論文帯 CI ≈ 0.3–0.8 の網掛けつき．

## `visualize-sweep` — 感度分析

```bash
uv run sabm-tools visualize-sweep
uv run sabm-tools visualize-sweep --sweep-dir "$(runvault path --experiment sabm --latest --subcommand sweep)"
```

スイープ親 run の子から 1 行 1 試行の表を組み直し (`runvault.read.sweep_events_table`: 条件 `firms` / `d_beta` は子の `parameters`，試行は `terminal` イベント)，`results/sabm/figures/{run_slug}/` へ出力:

- **`sweep_ci_by_dbeta.png`** — 平均最終 collusion index vs `d/β`．企業数ごとに 1 本の折れ線．暗黙共謀帯の網掛けつき．
- **`sweep_ci_heatmap.png`** — collusion index の `(d/β × 企業数)` ヒートマップ (企業数が複数のときのみ)．

## `show-experiment-settings` — 設定とメタデータ

```bash
uv run sabm-tools show-experiment-settings
uv run sabm-tools show-experiment-settings --results-dir "$(runvault path --experiment sabm --latest --subcommand run --standalone)"
uv run sabm-tools show-experiment-settings --json
```

実行条件 (`config.json` の `parameters`; どのサブコマンドかは `run.json` が答える)・解析フレーム (run スコープ指標の p^B / p^M)・LLM メタデータ (モデル・provider・温度は `run.json` の `llm` ブロック，呼び出し総数・cache-hit 率は run スコープ指標) を整形表示．legacy の `results/{timestamp}/` (flat な `config.json` / `sweep_config.json` / `benchmarks.json` / `llm_meta.json`) もそのまま読める．`--json` で機械可読出力．

## `reproduce` — 論文 Fig.1/2/4 一括再現

```bash
uv run sabm-tools reproduce                  # 直近の reproduce run を読む
uv run sabm-tools reproduce --run --mock --quick   # 先に Rust バイナリをオフライン実行
```

reproduce の run から要約を組み直し (シナリオごとの価格軌跡は `metrics.csv` と `observation` イベント，アンカーの帯と判定は `events.jsonl` の `x.han2023.anchor` 行)，PASS/off アンカーテーブルを表示し，3 つの図を `results/sabm/figures/{run_slug}/` に描画する: `fig1_no_communication.png` (会話なしの価格軌跡)・`fig2_communication.png` (会話あり変種)・`fig4_collusion_compare.png` (会話なし vs 会話あり の collusion index 比較)．`--run` は先に `sabm reproduce` を実行する (`--mock` / `--quick` / `--seed` を透過)．`--json` は図を描かずサマリのみ出力する．
