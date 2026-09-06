[English](cli.md) | [日本語](cli.ja.md)

# CLI (`sabm`)

`cargo build --release` でビルドし，バイナリは `sabm`．4 サブコマンド: `benchmark` / `run` / `sweep` / `reproduce`．

## LLM 環境変数

プロバイダ順は **Ollama 第一 → OpenAI フォールバック** (`socsim-llm` が所有):

- `OLLAMA_HOST` (既定 `http://localhost:11434`)・`OLLAMA_MODEL` (既定 `llama3.2:latest`)
- `OPENAI_API_KEY`・`OPENAI_MODEL` (既定 `gpt-4o-mini`; 論文は `gpt-4-0314`)

## `benchmark` — 解析価格のみ (LLM 不要・即時)

ベルトラン均衡価格とカルテル/独占価格を計算する．ライブ LLM には一切触れず RNG も使わないので，`domain = analysis`・`master_seed` なしで記録する．全社平均は `metrics.csv` (`p_bertrand_mean` / `p_cartel_mean`) へ，企業ごとの値は `x.han2023.benchmark` イベントへ置く — 時間軸を持たない系列は指標にできないためである．

```bash
cargo run --release -- benchmark --a 14 --beta 0.0066667 --d 0.0033333 --c1 2 --c2 2
# → firm 0: p_bertrand ≈ 6, p_cartel = 8   (β=1/150, d=1/300 で厳密)
```

| フラグ | 既定 | 意味 |
|------|---------|---------|
| `--firms` | 2 | 企業数 n |
| `--a` | 14 | 逆需要切片 a |
| `--beta` | 1/150 | 自己価格効果 β |
| `--d` | 1/300 | 交差価格効果 d |
| `--d-beta` | — | 指定すると `d = d_beta · β` (`--d` を上書き) |
| `--c1` / `--c2` | 2 / 2 | 限界費用 (非対称対応) |
| `--output-dir` | results | 出力ベースディレクトリ |

## `run` — LLM 駆動の繰り返し価格ゲーム

```bash
cargo run --release -- run --firms 2 --persona active \
    --max-rounds 1000 --temperature 0 --seed 42
```

上記の市場フラグに加えて: `--persona {active,aggressive,none}`・`--communication`・`--max-rounds`・`--reflection-period` (既定 20)・`--memory-window` (既定 20)・`--runs`・`--seed`・`--temperature`・`--llm-seed`・`--cache-path`・`--output-dir`・`--mock` (ライブ LLM の代わりに scripted オフラインクライアントを使う — CI / サンドボックス用)．

`--communication` は**フラグ**型 (付ければ会話あり; 付けなければ論文の会話なし基本ケース)．付けると価格決定の前に `CommunicationPhase` が走り，各社が短い cheap-talk メッセージを発して次ラウンドの価格プロンプトへ注入される．ペルソナ (`--persona`) と非対称限界費用 (`--c1` / `--c2`) は結果の価格をずらす — 高コスト社ほど高い価格に収束する．

runvault の run ディレクトリ (`results/sabm/run_{timestamp}_{hash}/`) を書き出す: `metrics.csv` (ラウンドごとの `avg_price` / `collusion_index` / `total_profit`，ラウンドを持たない `p_bertrand_mean` / `p_cartel_mean` / `rounds_to_stable` と LLM 呼び出しの内訳)・`events.jsonl` ((企業, ラウンド) ごとの `observation` が `price` / `quantity` / `profit` を，企業ごとの `terminal`，企業ごとの `x.han2023.benchmark`)・`config.json`．`latest` シンボリックリンクは作らない — 場所は `runvault path --experiment sabm --latest --subcommand run --standalone` で解決する．`--runs N` のとき詳細を残すのは最後の試行だけで，これは移行前と同じ (先行する試行はキャッシュを温めるだけ)．

## `sweep` — d/β × 企業数 感度分析

```bash
cargo run --release -- sweep \
    --d-beta-values 0.0,0.25,0.5,0.75,1.0 --firms-values 2,3 \
    --max-rounds 400 --runs 5 --seed 42
```

フラグ: `--d-beta-values`・`--firms-values`・`--a`・`--beta`・`--cost`・`--persona`・`--communication`・`--max-rounds`・`--runs`・`--seed`・`--temperature`・`--llm-seed`・`--cache-path`・`--output-dir`・`--mock`．親 run (`subcommand=sweep`．格子の定義そのものを持ち，1 本のシミュレーションではないので `master_seed` を名乗らない) と，条件ごとの子 run (`subcommand=sweep-point`．`parameters` が `firms` / `d_beta` を持つ) を書き出す．子の `events.jsonl` が試行 1 本ごとの `terminal` 行 (旧 `sweep_summary.csv` の列) を，`metrics.csv` が条件の集約を `scope=run` で持つ．試行ごとの値は `metrics.csv` には置けない — 主キーが `(run_uid, name, step, step_unit, scope)` なので同一条件の試行が重複する．

## `reproduce` — 論文 Fig.1/2/4 一括再現

```bash
cargo run --release -- reproduce --seed 42            # 実 LLM
cargo run --release -- reproduce --seed 42 --mock --quick   # オフライン・80 ラウンド
```

会話なし baseline (Fig.1) と会話あり変種 (Fig.2) を実行し，観測した平均価格と collusion index をベルトラン(6)/カルテル(8) フレームと照合して 1 本の run (`subcommand=reproduce`) を書き出す．2 シナリオが同居するので，シナリオ名を指標名 (`no_communication_avg_price` など) と `unit_id` (`communication-firm-0` など) の接頭辞にする — `(step, scope, name)` が主キーだからである．アンカーの観測量は `anchor_{id}` 指標と `checks_passed` / `checks_total`，帯と PASS/off は `x.han2023.anchor` イベント (比較の向きと判定はカテゴリであって数ではない)．帯は論文が報告した値 (ベルトラン 6 / カルテル 8 のフレーム) とこの再現実装が置いた定性的アンカーが混ざるので，どちらも `reference.csv` には書かない．フラグ: 上記の市場フラグ・`--persona`・`--max-rounds`・`--seed`・`--temperature`・`--llm-seed`・`--cache-path`・`--output-dir`・`--mock`・`--quick`．図は `uv run sabm-tools reproduce` で描画する．

## オフラインスモーク (ライブ LLM 不要)

```bash
cargo run --release --example mock_smoke -- results   # 専用 example
cargo run --release -- run --max-rounds 60 --seed 42 --mock   # または run/sweep に --mock
```
