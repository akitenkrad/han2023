[English](reproduction.md) | [日本語](reproduction.ja.md)

# 再現

二層設計に対応して 2 段階の再現がある．

## 解析パス — 定量的に厳密

ベルトラン均衡価格とカルテル/独占価格は，socsim・LLM から独立な純粋・決定論的な数値計算 (`analytic.rs`)．論文の**基本設定** (`a = 14`, `β = 1/150`, `d = 1/300`, `α = 7/150`, `b = 1/30000`, `c_1 = c_2 = 2`, `d/β = 0.5`) で:

| 量 | 論文 | 本実装 |
|----------|-------|---------------------|
| ベルトラン均衡 `p^B` | 6 | **6** (厳密; `cargo test` が `|p^B − 6| < 1e-9` を assert) |
| カルテル/独占 `p^M` | 8 | **8** (厳密; `cargo test` が `|p^M − 8| < 1e-9` を assert) |

サンドボックス内検証 (LLM 不要):

```bash
cargo run --release -- benchmark            # 厳密 1/150, 1/300 既定 → 6.000000 / 8.000000
cargo run --release -- benchmark --a 14 --beta 0.0066667 --d 0.0033333 --c1 2 --c2 2
```

(丸めた CLI 入力 `0.0066667 / 0.0033333` では `p^B ≈ 6.00004`; 厳密な逆数既定では正確に 6 と 8．)

## LLM 駆動パス — 定性的

ローカル既定モデル (`llama3.2`) は論文の `gpt-4-0314` と異なるため，LLM 駆動の暗黙共謀の結果は**定性的** (符号 / 傾向) に再現し，論文の厳密値には合わせない．

| 条件 | 指標 | 論文 | 再現目標 |
|-----------|--------|-------|---------------------|
| 会話なし・active | 収束価格 | 約 7 (6 < 7 < 8) | (ベルトラン, カルテル) 区間内; CI ∈ [0.3, 0.8] |
| 会話なし・active | 安定到達ラウンド | > 300 | 桁感 (≫ 会話あり) |
| `d/β = 0` (独立) | 収束価格 | 独占 8 | ±0.5 |
| `d/β = 1` (同質財) | 収束価格 | 限界費用 2 をわずかに上回る | (2, p^B] 近傍 |
| 非対称コスト (`c_1=2, c_2=5`) | 共謀 | 均衡超・カルテル未満 | CI ∈ (0, 1) |

実行 (実 Ollama が到達可能であること):

```bash
export OLLAMA_MODEL=llama3.2:latest
cargo run --release -- run --firms 2 --persona active \
    --max-rounds 1000 --temperature 0 --seed 42
uv run sabm-tools visualize
```

小さなローカルモデルとラウンド数制限のもとでは暗黙共謀の CI はノイジーになりうる (想定内)．キャッシュ (`temperature=0` + 固定 seed) により再実行は無料かつ安定で，一度キャッシュされた実行は再生時に bit 単位で再現する．

## `reproduce` — Fig.1/2/4 一括再現

`reproduce` は会話なし baseline (Fig.1) と会話あり変種 (Fig.2) を一括実行し，headline 知見をベルトラン(6)/カルテル(8) フレームと照合する:

```bash
cargo run --release -- reproduce --seed 42 --mock --quick   # オフライン構造スモーク
uv run sabm-tools reproduce                                 # fig1/fig2/fig4 描画 + PASS テーブル
```

解析フレーム (`{シナリオ}_p_bertrand_mean`・`{シナリオ}_p_cartel_mean`)，シナリオごとの価格軌跡と `rounds_to_stable` を `metrics.csv` に，PASS/off アンカーを `x.han2023.anchor` イベントに記録する:

| アンカー | 目標 | 読み方 |
| --- | --- | --- |
| 会話なし `avg_price` が (p^B, p^M) 内 | `[6, 8]` (論文 約7) | 暗黙の共謀が競争とカルテルの間に着地 |
| 会話なし `collusion_index` | `[0.3, 0.8]` | collusion index が論文の中域帯 |
| 会話が共謀を強める | `CI_comm − CI_nocomm ≥ −0.05` | 会話は共謀を弱めない |

scripted オフライン mock (各社が自社固有の `(p^B+p^M)/2` 中点へ寄る) では対称基本設定で両シナリオとも `avg_price = 7.000`・`CI = 0.500` となり 3 アンカーすべて in-band．実 LLM では絶対水準と会話効果が経験的に観測され，mock は構造と PASS/off 枠組みを検証する．
