//! runvault への記録の共通部分．
//!
//! 論文メタデータ (research) は `benchmark` / `run` / `sweep` / `reproduce` のどの
//! サブコマンドでも同一なので，ここ 1 箇所で組み立てる．ラウンドごとの市場指標，
//! 企業ごとの観測と終端行，解析ベンチマークの落とし方もここに集める．

use runvault::{Llm, Replication, Run, Target, Work};
use serde::Serialize;

use crate::config::Config;
use crate::metrics::{mean, MetricRow};
use crate::simulation::{Benchmarks, SimulationResult};

/// runvault 上の実験名．`runvault path --experiment` に渡す値でもある．
/// バイナリ名 (`sabm`) と揃える．
pub const EXPERIMENT: &str = "sabm";
/// リポジトリの安定 id．git remote の名前とは独立に固定する．
pub const REPO_ID: &str = "han2023";
/// 分野．初期価格を init RNG で散らすので `simulation` (= `master_seed` が必須)．
///
/// LLM で駆動されるモデルだが `llm-safety` ではない — 測っているのはモデルの
/// 安全性ではなく，繰り返し価格ゲームの価格軌跡だからである．LLM 側の同一性は
/// `llm` ブロック ([`llm_block`]) が持つ．
pub const DOMAIN: &str = "simulation";
/// `benchmark` サブコマンドの分野．
///
/// 解析ベンチマーク (Bertrand / cartel 価格) は需要パラメータだけから決まる純粋な
/// 数値計算で RNG を使わない．`simulation` を名乗ると `master_seed` が必須になり，
/// 存在しないシードを書くことになるので分ける．
pub const DOMAIN_ANALYSIS: &str = "analysis";

/// 時間軸の単位．
///
/// 本モデルの刻みは «全社が同時に価格を付け市場が清算される» 1 ラウンドで，論文
/// §4 の繰り返しゲームの 1 期そのものである．runvault の語彙では `round`．
const T_UNIT: &str = "round";

/// 指標の粒度．市場指標はどれも全社の集約なので `run`．
const SCOPE: &str = "run";

/// この再現実験が対象としている論文．
///
/// 論文は特定の図表ではなく «会話が無くても 2 社の LLM エージェントは暗黙の共謀に
/// 到達し，価格がベルトラン均衡とカルテル価格の中間へ収束する» という主張の再現を
/// 狙うので，`Target::claim` を使う．
pub fn replication() -> Replication {
    Work::arxiv("2308.10974")
        .title("\"Guinea Pig Trials\" Utilizing GPT: A Novel Smart Agent-Based Modeling Approach for Studying Firm Competition and Collusion")
        .year(2023)
        .source_version("arxiv-v1")
        .target(Target::claim(
            "tacit-collusion-without-communication",
            "Without any communication, LLM firms settle at a price above the Bertrand equilibrium and below the cartel price",
        ))
        .obsidian_note("研究/98_論文レポート/80-再現実験/実装完了/han2023/設計書.md")
}

// ---------------------------------------------------------------------------
// LLM ブロック
// ---------------------------------------------------------------------------

/// 実際に応答したバックエンドを `llm` ブロックに落とす．
///
/// `model` / `endpoint` はクライアントが名乗った値をそのまま使う．`provider` は
/// runvault の語彙ではなく自由記述なので，endpoint から «どのゲートウェイが答えたか»
/// を決める (`mock://…` はオフラインの scripted クライアント，それ以外はホスト名で
/// Ollama / OpenAI を分ける)．推測しているのは分類だけで，値そのものは記録から採る．
///
/// `model_snapshot` に入るのは `llama3.2:latest` のような動くエイリアスであることが
/// 多い．socsim-llm はスナップショット id を持たないので，持っていない値を作らずに
/// 名乗られた名前を書く．
pub fn llm_block(model: &str, endpoint: &str, temperature: f32) -> Llm {
    let provider = if endpoint.starts_with("mock://") {
        "mock"
    } else if endpoint.contains("openai") {
        "openai"
    } else {
        "ollama"
    };
    Llm {
        provider: provider.to_string(),
        model_snapshot: model.to_string(),
        temperature: Some(temperature as f64),
        // 価格決定プロンプトは企業の履歴から毎ラウンド組み立てられ，固定の
        // system prompt を持たない．無いものを hash しない．
        system_prompt_hash: None,
    }
}

// ---------------------------------------------------------------------------
// ラウンドの数え方
// ---------------------------------------------------------------------------

/// 最後に記録されたラウンド番号 (0 始まり)．
///
/// `SimulationResult::final_round` は socsim エンジンのクロック (`1..=max_rounds`)
/// の最終値なので，ラウンド番号としては 1 つ大きい (`MarketWorld::current_round` が
/// `t() - 1` を返す)．`metrics.csv` / `events.jsonl` の時間軸はラウンド番号の方なので，
/// ここは記録された行から読む — 引き算で作ると数え方の食い違いが黙って通る．
pub fn last_round(result: &SimulationResult) -> u64 {
    result
        .metrics
        .last()
        .map(|m| m.round)
        .expect("metrics は少なくとも 1 ラウンドを含む")
}

/// 時間軸の上限 (ラウンド番号)．
///
/// クロックは `1..=max_rounds` を走るので，取りうる最大のラウンド番号は
/// `max_rounds - 1` である．`terminal` 行の `budget` はこの値を指す．
pub fn round_budget(max_rounds: usize) -> u64 {
    max_rounds.saturating_sub(1) as u64
}

// ---------------------------------------------------------------------------
// ラウンドごとの市場指標
// ---------------------------------------------------------------------------

/// [`MetricRow`] の 3 フィールドを 1 ラウンドぶんまとめて書く．
///
/// 名前は旧 `metrics.csv` の列名のまま (`avg_price` / `collusion_index` /
/// `total_profit`) にしてある．wide から long へ形は変わるが，«移行で数が変わって
/// いないか» を列名の対応表なしに突き合わせられる．
///
/// `prefix` は 1 つの run に複数のシナリオが同居するとき (`reproduce`) に付ける．
/// `(step, scope, name)` が主キーなので，接頭辞が無いとシナリオどうしで衝突する．
pub fn log_round(run: &mut Run, prefix: Option<&str>, m: &MetricRow) {
    let name = |base: &str| match prefix {
        Some(p) => format!("{p}_{base}"),
        None => base.to_string(),
    };
    run.log_metrics_at(
        m.round,
        T_UNIT,
        SCOPE,
        &[
            (name("avg_price").as_str(), m.avg_price),
            (name("collusion_index").as_str(), m.collusion_index),
            (name("total_profit").as_str(), m.total_profit),
        ],
    )
    .unwrap_or_else(|e| panic!("round {} の指標の記録に失敗: {e}", m.round));
}

/// シミュレーション 1 本ぶんの，ラウンドを持たない値．
///
/// 実行時間は `status.json` の `duration_sec` が正本なので指標にはしない．
/// `rounds_to_stable` は «安定に達しなかった» とき **行ごと書かない** (旧
/// `sweep_summary.csv` は -1 を書いていたが，欠測を数で埋めない)．
pub fn log_run_scalars(run: &mut Run, prefix: Option<&str>, result: &SimulationResult) {
    let name = |base: &str| match prefix {
        Some(p) => format!("{p}_{base}"),
        None => base.to_string(),
    };
    let mut values: Vec<(String, f64)> = vec![
        (name("p_bertrand_mean"), mean(&result.benchmarks.p_bertrand)),
        (name("p_cartel_mean"), mean(&result.benchmarks.p_cartel)),
        (name("llm_calls"), result.metadata.total() as f64),
        (name("llm_cache_hits"), result.metadata.cache_hits() as f64),
        (name("llm_cache_hit_rate"), result.metadata.cache_hit_rate()),
    ];
    if let Some(r) = result.rounds_to_stable() {
        values.push((name("rounds_to_stable"), r as f64));
    }
    let borrowed: Vec<(&str, f64)> = values.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    run.log_metrics(SCOPE, &borrowed)
        .expect("run スコープの指標の記録に失敗");
}

// ---------------------------------------------------------------------------
// 企業ごとの観測と終端
// ---------------------------------------------------------------------------

/// `events.jsonl` に書く企業 1 社 1 ラウンドの観測行．
///
/// 旧 `rounds.csv` の 1 行に対応する．この行は時間 (ラウンド) と系列 (企業) の
/// **両方** で決まるので `metrics.csv` には置けない — 主キー
/// `(run_uid, name, step, step_unit, scope)` に系列を表す列が無く，2 社の同じ
/// ラウンドが重複するからである．価格・数量・利益はここが唯一の置き場なので，
/// 予約キーに続けてそのまま持たせる．
#[derive(Serialize)]
struct FirmObservation<'a> {
    unit_id: &'a str,
    t: u64,
    t_unit: &'static str,
    price: f64,
    quantity: f64,
    profit: f64,
}

/// `events.jsonl` に書く企業 1 社の終端行．
///
/// 先頭 6 フィールドは runvault の予約語 (`terminal` はこれを全部要求する)．
#[derive(Serialize)]
struct FirmTerminal<'a> {
    unit_id: &'a str,
    t: u64,
    t_unit: &'static str,
    outcome: &'static str,
    censored: bool,
    budget: u64,
    trial_seed: u64,
    final_price: f64,
    final_quantity: f64,
    final_profit: f64,
    p_bertrand: f64,
    p_cartel: f64,
}

/// `unit_id` の綴り．`reproduce` は 2 シナリオが 1 つの run に同居するので接頭辞を付ける．
fn firm_unit(prefix: Option<&str>, firm_id: u64) -> String {
    match prefix {
        Some(p) => format!("{p}-firm-{firm_id}"),
        None => format!("firm-{firm_id}"),
    }
}

/// 企業ごとの価格軌跡を `observation`，最終ラウンドを `terminal` として書く．
///
/// 打ち切り (`censored`) の行は `t == budget` でなければならない．ドライバは平均価格
/// 系列が定常 (収束 or 有界振動) になれば `request_stop()` を撃ち，撃たれなければ
/// `max_rounds` まで回すので，予算を使い切った run は必ず上限ラウンドに達している．
/// この不変条件は runvault が `log_event` の書き込み時に検査するので，ここでは
/// 二重に持たない．
///
/// `outcome` は «定常判定で早く止まった» (`stationary`) か «予算を使い切った»
/// (`budget-exhausted`) か．観測できるのはこの 2 つの区別だけで，予算の最終
/// ラウンドで定常になった場合も後者に入る — 停止規則が予算より前に発火したことは
/// 確かめられないからである．
pub fn log_firms(
    run: &mut Run,
    prefix: Option<&str>,
    trial_seed: u64,
    max_rounds: usize,
    result: &SimulationResult,
) {
    let last = last_round(result);
    let budget = round_budget(max_rounds);
    let censored = result.final_round >= max_rounds;
    let outcome = if censored {
        "budget-exhausted"
    } else {
        "stationary"
    };

    for row in &result.rounds {
        let unit = firm_unit(prefix, row.firm_id);
        run.log_event(
            "observation",
            &FirmObservation {
                unit_id: &unit,
                t: row.round,
                t_unit: T_UNIT,
                price: row.price,
                quantity: row.quantity,
                profit: row.profit,
            },
        )
        .unwrap_or_else(|e| {
            panic!(
                "{unit} の round {} の observation の記録に失敗: {e}",
                row.round
            )
        });
    }

    for (i, &firm) in result_firm_ids(result).iter().enumerate() {
        let unit = firm_unit(prefix, firm);
        let final_row = result
            .rounds
            .iter()
            .rev()
            .find(|r| r.firm_id == firm && r.round == last)
            .unwrap_or_else(|| panic!("{unit} の最終ラウンドの行がありません"));
        run.log_event(
            "terminal",
            &FirmTerminal {
                unit_id: &unit,
                t: last,
                t_unit: T_UNIT,
                outcome,
                censored,
                budget,
                trial_seed,
                final_price: final_row.price,
                final_quantity: final_row.quantity,
                final_profit: final_row.profit,
                p_bertrand: result.benchmarks.p_bertrand[i],
                p_cartel: result.benchmarks.p_cartel[i],
            },
        )
        .unwrap_or_else(|e| panic!("{unit} の terminal イベントの記録に失敗: {e}"));
    }
}

/// 記録に現れた企業 id を昇順で返す．
fn result_firm_ids(result: &SimulationResult) -> Vec<u64> {
    let mut ids: Vec<u64> = result.rounds.iter().map(|r| r.firm_id).collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

// ---------------------------------------------------------------------------
// 解析ベンチマーク
// ---------------------------------------------------------------------------

/// `events.jsonl` に書く企業ごとの解析ベンチマーク．
///
/// 旧 `benchmarks.json` の `p_bertrand` / `p_cartel` 配列に対応する．時間軸を持たない
/// 企業ごとの値なので `metrics.csv` には置けない — 主キーに系列を表す列が無く，
/// 複数行が同じキーを主張してしまう (非対称コスト `--c1 --c2` では企業ごとに違う値に
/// なる)．全社平均だけは 1 つの数なので指標に置く ([`log_benchmark_means`])．
#[derive(Serialize)]
struct BenchmarkEvent<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    scenario: Option<&'a str>,
    firm_id: usize,
    p_bertrand: f64,
    p_cartel: f64,
}

/// 企業ごとの解析ベンチマークを `x.han2023.benchmark` として書く．
pub fn log_benchmarks(run: &mut Run, scenario: Option<&str>, bench: &Benchmarks) {
    for (i, (b, m)) in bench
        .p_bertrand
        .iter()
        .zip(bench.p_cartel.iter())
        .enumerate()
    {
        run.log_event(
            "x.han2023.benchmark",
            &BenchmarkEvent {
                scenario,
                firm_id: i,
                p_bertrand: *b,
                p_cartel: *m,
            },
        )
        .unwrap_or_else(|e| panic!("firm {i} のベンチマークの記録に失敗: {e}"));
    }
}

/// 全社平均のベンチマークを run スコープの指標として書く (`benchmark` サブコマンド用)．
pub fn log_benchmark_means(run: &mut Run, bench: &Benchmarks) {
    run.log_metrics(
        SCOPE,
        &[
            ("p_bertrand_mean", mean(&bench.p_bertrand)),
            ("p_cartel_mean", mean(&bench.p_cartel)),
        ],
    )
    .expect("ベンチマーク平均の記録に失敗");
}

// ---------------------------------------------------------------------------
// 条件 1 点ぶんの集約 (sweep の子 run)
// ---------------------------------------------------------------------------

/// `events.jsonl` に書く試行 1 本の終端行 (`sweep` 用)．
///
/// 旧 `sweep_summary.csv` の 1 行に対応する．派生シードを `seed` ではなく
/// `trial_seed` と呼ぶのは，`runvault.read` の `sweep_events_table` が «条件の
/// parameters» を同名のイベント列へ上書きするためである．子 run の `parameters` は
/// base seed を `seed` として持つので，イベント側も `seed` にすると試行ごとの派生
/// シードが黙って base seed に化ける．
#[derive(Serialize)]
struct TrialTerminal<'a> {
    unit_id: &'a str,
    t: u64,
    t_unit: &'static str,
    outcome: &'static str,
    censored: bool,
    budget: u64,
    trial_seed: u64,
    final_avg_price: f64,
    final_collusion_index: f64,
    p_bertrand_mean: f64,
    p_cartel_mean: f64,
    /// 安定に達したラウンド．達しなければ **列ごと落とす**．
    #[serde(skip_serializing_if = "Option::is_none")]
    rounds_to_stable: Option<u64>,
    cache_hit_rate: f64,
}

/// `events.jsonl` に書く観測行 (予約キーのみ)．
///
/// `verify --deep` は terminal の `unit_id` が observation にも現れることを要求する
/// ので，観測した時刻を明示的に残す．数はここには書かない — 試行の最終値は
/// [`TrialTerminal`] が正本である．
#[derive(Serialize)]
struct TrialObservation<'a> {
    unit_id: &'a str,
    t: u64,
    t_unit: &'static str,
}

/// スイープの試行 1 本を `terminal` (と観測 1 点) として書く．
pub fn log_trial(
    run: &mut Run,
    unit_id: &str,
    trial_seed: u64,
    max_rounds: usize,
    result: &SimulationResult,
) -> TrialOutcome {
    let last = last_round(result);
    let censored = result.final_round >= max_rounds;

    run.log_event(
        "observation",
        &TrialObservation {
            unit_id,
            t: last,
            t_unit: T_UNIT,
        },
    )
    .unwrap_or_else(|e| panic!("{unit_id} の t={last} の observation の記録に失敗: {e}"));

    let outcome = TrialOutcome {
        last_round: last,
        stationary: !censored,
        final_avg_price: result.final_avg_price(),
        final_collusion_index: result.final_collusion_index(),
        rounds_to_stable: result.rounds_to_stable().map(|r| r as u64),
    };

    run.log_event(
        "terminal",
        &TrialTerminal {
            unit_id,
            t: last,
            t_unit: T_UNIT,
            outcome: if censored {
                "budget-exhausted"
            } else {
                "stationary"
            },
            censored,
            budget: round_budget(max_rounds),
            trial_seed,
            final_avg_price: outcome.final_avg_price,
            final_collusion_index: outcome.final_collusion_index,
            p_bertrand_mean: mean(&result.benchmarks.p_bertrand),
            p_cartel_mean: mean(&result.benchmarks.p_cartel),
            rounds_to_stable: outcome.rounds_to_stable,
            cache_hit_rate: result.metadata.cache_hit_rate(),
        },
    )
    .unwrap_or_else(|e| panic!("{unit_id} の terminal イベントの記録に失敗: {e}"));

    outcome
}

/// 1 試行の最終値．条件の集約の材料になる．
pub struct TrialOutcome {
    /// 最後に記録されたラウンド番号．
    pub last_round: u64,
    /// 定常判定で予算より早く止まったか．
    pub stationary: bool,
    /// 最終ラウンドの全社平均価格．
    pub final_avg_price: f64,
    /// 最終ラウンドの collusion index．
    pub final_collusion_index: f64,
    /// 安定到達ラウンド (未達なら `None`)．
    pub rounds_to_stable: Option<u64>,
}

/// 1 条件 (firms × d/β の 1 点) を 1 つの値で表す指標．
///
/// 試行ごとの値は `events.jsonl` の担当なので，ここには集約しか書かない．試行ごとの
/// `final_collusion_index` を指標にすると (`run_uid`, `step`, `scope`, `name`) が
/// 重複するので，散らばりが要る図は `events.jsonl` から組み直す．
///
/// `mean_rounds_to_stable` は安定に達した試行だけの平均で，1 本も達しなければ行ごと
/// 書かない — 未達を 0 や -1 で埋めると «すぐ安定した» と見分けが付かなくなる．
pub fn log_condition_summary(run: &mut Run, trials: &[TrialOutcome]) {
    let n = trials.len();
    assert!(n > 0, "試行が 1 本もありません");
    let n_f = n as f64;

    let n_stationary = trials.iter().filter(|t| t.stationary).count();
    let mean_of = |f: &dyn Fn(&TrialOutcome) -> f64| trials.iter().map(f).sum::<f64>() / n_f;

    let mut values: Vec<(String, f64)> = vec![
        ("n_units".to_string(), n_f),
        ("n_stationary".to_string(), n_stationary as f64),
        ("stationary_rate".to_string(), n_stationary as f64 / n_f),
        (
            "mean_final_avg_price".to_string(),
            mean_of(&|t| t.final_avg_price),
        ),
        (
            "mean_final_collusion_index".to_string(),
            mean_of(&|t| t.final_collusion_index),
        ),
        (
            "mean_last_round".to_string(),
            mean_of(&|t| t.last_round as f64),
        ),
    ];
    let stable: Vec<u64> = trials.iter().filter_map(|t| t.rounds_to_stable).collect();
    if !stable.is_empty() {
        values.push((
            "mean_rounds_to_stable".to_string(),
            stable.iter().sum::<u64>() as f64 / stable.len() as f64,
        ));
        values.push(("n_stable".to_string(), stable.len() as f64));
    }
    let borrowed: Vec<(&str, f64)> = values.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    run.log_metrics(SCOPE, &borrowed)
        .expect("run スコープの指標の記録に失敗");
}

// ---------------------------------------------------------------------------
// reproduce の帯照合
// ---------------------------------------------------------------------------

/// `events.jsonl` に書くアンカー判定行．
///
/// 照合先の帯は，論文が報告した値 (Bertrand 6 / cartel 8 のフレームと «おおよそ 7»)
/// と，この再現実装が置いた定性的なアンカー (CI の 0.3–0.8，会話ありは共謀を弱めない)
/// が混ざっている．後者は論文の報告値ではないので，出典を要求する `reference.csv`
/// には書かない — 書くと «論文の値» と «こちらが決めた帯» が後から見分けられなく
/// なる．
///
/// 観測量そのものは数なので `metrics.csv` にも `anchor_<id>` として置く
/// ([`log_anchor_observations`])．ここに残すのは比較の向き (帯) と PASS/off という
/// カテゴリで，これは指標にできない．
#[derive(Serialize)]
pub struct AnchorEvent<'a> {
    /// 指標名になる slug．
    pub id: &'a str,
    /// 人間向けの説明 (どの量を見ているか)．
    pub label: &'a str,
    /// 論文側の値・主張．
    pub paper_value: &'a str,
    pub observed: f64,
    pub target_lo: f64,
    /// 上限．«上限なし» は `null` ではなく列ごと落とす (JSON に無限大は無い)．
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_hi: Option<f64>,
    pub pass: bool,
}

/// アンカーの観測量を run スコープの指標として書く．
pub fn log_anchor_observations(run: &mut Run, anchors: &[(String, f64)], n_pass: usize) {
    let named: Vec<(String, f64)> = anchors
        .iter()
        .map(|(id, v)| (format!("anchor_{id}"), *v))
        .collect();
    let values: Vec<(&str, f64)> = named.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    run.log_metrics(SCOPE, &values)
        .expect("アンカー観測量の記録に失敗");
    run.log_metrics(
        SCOPE,
        &[
            ("checks_passed", n_pass as f64),
            ("checks_total", anchors.len() as f64),
        ],
    )
    .expect("アンカー件数の記録に失敗");
}

// ---------------------------------------------------------------------------
// 実験条件
// ---------------------------------------------------------------------------

/// `benchmark` / `run` / `reproduce` の実験条件．
///
/// 旧 `config.json` の内容から `command` と `output_dir` を落としたもの．どの
/// サブコマンドかは `run.json` が持ち，run ディレクトリが出力先そのものである．
/// `seed` は実際に使われた個別のシードではなく base seed で，試行ごとの派生シードは
/// `terminal` 行の `trial_seed` が持つ．
#[derive(Serialize)]
pub struct RunParameters {
    pub n_firms: usize,
    pub a: f64,
    pub beta: f64,
    pub d: f64,
    pub d_over_beta: f64,
    pub alpha: f64,
    pub b: f64,
    pub c1: f64,
    pub c2: f64,
    pub persona: String,
    pub communication: bool,
    pub max_rounds: usize,
    pub reflection_period: usize,
    pub memory_window: usize,
    pub runs: usize,
    pub seed: u64,
    pub llm_temperature: f32,
    pub llm_seed: u64,
    pub mock: bool,
}

impl RunParameters {
    /// [`Config`] と CLI 引数から組み立てる．
    pub fn new(cfg: &Config, runs: usize, seed: u64, mock: bool) -> Self {
        let dp = cfg.demand_params();
        RunParameters {
            n_firms: cfg.n_firms,
            a: cfg.a,
            beta: cfg.beta,
            d: cfg.d,
            d_over_beta: dp.d_over_beta(),
            alpha: dp.alpha,
            b: dp.b,
            c1: cfg.c1,
            c2: cfg.c2,
            persona: cfg.persona.label().to_string(),
            communication: cfg.communication,
            max_rounds: cfg.max_rounds,
            reflection_period: cfg.reflection_period,
            memory_window: cfg.memory_window,
            runs,
            seed,
            llm_temperature: cfg.llm.temperature,
            llm_seed: cfg.llm.seed,
            mock,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::round_budget;

    #[test]
    fn budget_is_the_largest_round_number() {
        // クロックは 1..=max_rounds を走り，ラウンド番号は 0..=max_rounds-1．
        assert_eq!(round_budget(400), 399);
        assert_eq!(round_budget(1), 0);
        // 0 ラウンドは走らないが，引き算で溢れさせない．
        assert_eq!(round_budget(0), 0);
    }
}
