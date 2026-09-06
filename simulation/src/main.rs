//! Han, Wu & Xiao (2023) "SABM" — 再現実験の CLI エントリポイント．
//!
//! `benchmark` : 解析ベンチマーク (Bertrand 均衡 / cartel 価格) のみを計算する
//!               (LLM 不要・即時)．基本設定で p^B=6, p^M=8 を厳密に出力する．
//! `run`       : 単一設定で LLM 駆動の繰り返し価格ゲームを実行し，暗黙の共謀
//!               (価格が (Bertrand, cartel) 区間へ収束) を観測する．
//! `sweep`     : 差別化度 d/β × 企業数 を走査し，条件 1 点ごとに子 run を起こして
//!               `runs` 本の試行を回す．
//! `reproduce` : 論文 Fig.1/2/4 (会話なし / 会話あり) を一括再現し，観測 vs 論文の
//!               PASS/off を run スコープの指標とイベントに記録する．
//!
//! 出力の置き場と同一性は runvault が持つ．タイムスタンプ付きディレクトリも
//! `latest` シンボリックリンクもこちらでは作らず，`Run::start` が決めた run
//! ディレクトリへ書く．
//!
//! LLM クライアントは記録を始める **前** に組む．`run.json` の `llm` ブロックは
//! `Run::start` の時点で確定するので，モデル名と endpoint を知っている側 (=
//! クライアントを組んだ側) が先に立たないと，llm ブロックを埋めないまま記録できて
//! しまう．

use std::fs;
use std::path::Path;

use clap::{Parser, Subcommand};
use runvault::{Lineage, Run, RunOptions};
use serde::Serialize;

use sabm_simulation::config::{derive_run_seed, Config, LlmSettings};
use sabm_simulation::llm::{build_live_client, wrap_client, SabmClient};
use sabm_simulation::record::{self, RunParameters, DOMAIN, DOMAIN_ANALYSIS, EXPERIMENT, REPO_ID};
use sabm_simulation::simulation::{compute_benchmarks, run_with_client_observed, SimulationResult};
use sabm_simulation::world::parse_persona;
use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;
// ---------------------------------------------------------------------------
// CLI 定義
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "sabm",
    about = "Han, Wu & Xiao (2023) SABM: LLM-agent Bertrand-duopoly firm competition & tacit collusion — 再現実験"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Ollama 接続先 URL（指定時は環境変数 OLLAMA_HOST を上書きする）．
    #[arg(long, global = true)]
    ollama_host: Option<String>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// 解析ベンチマーク (Bertrand / cartel 価格) のみを計算する (LLM 不要・即時)．
    Benchmark(BenchmarkArgs),
    /// 単一設定で LLM 駆動の繰り返し価格ゲームを実行する．
    Run(RunArgs),
    /// 差別化度 d/β × 企業数 を走査し最終 collusion index を集計する．
    Sweep(SweepArgs),
    /// 論文 Fig.1/2/4 (暗黙の共謀への価格収束) を一括再現し PASS/off を判定する．
    Reproduce(ReproduceArgs),
}

// --- 需要・コストの共通フラグ ---

#[derive(Parser, Debug, Clone)]
struct MarketArgs {
    /// 企業数 n (基本は 2 = 複占)．
    #[arg(long, default_value_t = 2)]
    firms: usize,

    /// 逆需要切片 a．
    #[arg(long, default_value_t = 14.0)]
    a: f64,

    /// 自己価格効果 β．
    #[arg(long, default_value_t = 1.0 / 150.0)]
    beta: f64,

    /// 交差価格効果 d (--d-beta を与えると d = d_beta * beta で上書き)．
    #[arg(long, default_value_t = 1.0 / 300.0)]
    d: f64,

    /// 差別化度 d/β (指定すると d = d_beta * beta を計算して d を上書きする)．
    #[arg(long)]
    d_beta: Option<f64>,

    /// 限界費用 c_1．
    #[arg(long, default_value_t = 2.0)]
    c1: f64,

    /// 限界費用 c_2 (非対称コスト対応)．
    #[arg(long, default_value_t = 2.0)]
    c2: f64,
}

impl MarketArgs {
    /// 実効的な交差価格効果 d (--d-beta があれば優先)．
    fn effective_d(&self) -> f64 {
        match self.d_beta {
            Some(ratio) => ratio * self.beta,
            None => self.d,
        }
    }
}

#[derive(Parser, Debug)]
struct BenchmarkArgs {
    #[command(flatten)]
    market: MarketArgs,

    /// 結果出力ディレクトリ．
    #[arg(long, default_value = "results")]
    output_dir: String,
}

#[derive(Parser, Debug)]
struct RunArgs {
    #[command(flatten)]
    market: MarketArgs,

    /// 企業ペルソナ (active / aggressive / none)．
    #[arg(long, default_value = "active")]
    persona: String,

    /// 会話の有無 (基本 false; 会話あり代替モデルは Phase 3)．
    #[arg(long, default_value_t = false)]
    communication: bool,

    /// 最大ラウンド数．
    #[arg(long, default_value_t = 1000)]
    max_rounds: usize,

    /// 反省 (reflection) を挟む周期．
    #[arg(long, default_value_t = 20)]
    reflection_period: usize,

    /// 境界付きメモリの窓．
    #[arg(long, default_value_t = 20)]
    memory_window: usize,

    /// 独立試行数 (各試行は derive により独立化する)．
    #[arg(long, default_value_t = 1)]
    runs: usize,

    /// 乱数シード (省略時はランダム; socsim コア層のみ支配)．
    #[arg(long)]
    seed: Option<u64>,

    /// LLM 生成温度 (既定 0.0)．
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// LLM 生成シード (バックエンドへ渡す)．
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,

    /// プロンプト→応答キャッシュの保存先．
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,

    /// 結果出力ディレクトリ．
    #[arg(long, default_value = "results")]
    output_dir: String,

    /// ライブ LLM の代わりに scripted mock を使う (オフライン検証・CI 用)．
    /// 各社の価格を (Bertrand, cartel) 中点へ漸近させる «暗黙の共謀» 風ポリシー．
    #[arg(long, default_value_t = false)]
    mock: bool,
}

#[derive(Parser, Debug)]
struct SweepArgs {
    /// カンマ区切りの d/β リスト．
    #[arg(long, default_value = "0.0,0.25,0.5,0.75,1.0")]
    d_beta_values: String,

    /// カンマ区切りの企業数リスト．
    #[arg(long, default_value = "2,3")]
    firms_values: String,

    /// 逆需要切片 a．
    #[arg(long, default_value_t = 14.0)]
    a: f64,

    /// 自己価格効果 β．
    #[arg(long, default_value_t = 1.0 / 150.0)]
    beta: f64,

    /// 限界費用 (対称)．
    #[arg(long, default_value_t = 2.0)]
    cost: f64,

    /// 企業ペルソナ．
    #[arg(long, default_value = "active")]
    persona: String,

    /// 会話の有無．
    #[arg(long, default_value_t = false)]
    communication: bool,

    /// 最大ラウンド数．
    #[arg(long, default_value_t = 400)]
    max_rounds: usize,

    /// 各条件あたりの独立試行数．
    #[arg(long, default_value_t = 5)]
    runs: usize,

    /// 乱数シード基点 (各試行は derive により独立化する)．
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// LLM 生成温度．
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// LLM 生成シード．
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,

    /// プロンプト→応答キャッシュの保存先 (sweep 全体で共有しヒット率を高める)．
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,

    /// 結果出力ベースディレクトリ．
    #[arg(long, default_value = "results")]
    output_dir: String,

    /// ライブ LLM の代わりに scripted mock を使う (オフライン検証・CI 用)．
    /// 各社の価格を (Bertrand, cartel) 中点へ漸近させる «暗黙の共謀» 風ポリシー．
    #[arg(long, default_value_t = false)]
    mock: bool,
}

#[derive(Parser, Debug)]
struct ReproduceArgs {
    #[command(flatten)]
    market: MarketArgs,

    /// 企業ペルソナ (active / aggressive / none)．
    #[arg(long, default_value = "active")]
    persona: String,

    /// 最大ラウンド数 (--quick で 80 に縮約)．
    #[arg(long, default_value_t = 300)]
    max_rounds: usize,

    /// 乱数シード基点 (シナリオごとに派生)．
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// LLM 生成温度．
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// LLM 生成シード．
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,

    /// プロンプト→応答キャッシュの保存先．
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,

    /// 結果出力ベースディレクトリ．
    #[arg(long, default_value = "results")]
    output_dir: String,

    /// ライブ LLM の代わりに scripted mock を使う (オフライン検証・CI 用)．
    #[arg(long, default_value_t = false)]
    mock: bool,

    /// 短縮再現 (max_rounds=80; CI スモーク用)．
    #[arg(long, default_value_t = false)]
    quick: bool,
}

// ---------------------------------------------------------------------------
// 補助
// ---------------------------------------------------------------------------

/// カンマ区切り文字列を trim 済みの非空リストへ．
fn split_csv(s: &str) -> Vec<String> {
    s.split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// プロンプト履歴の «最後の自社価格» を拾う (mock ポリシー用)．
fn last_own_price(prompt: &str) -> Option<f64> {
    let mut found = None;
    for line in prompt.lines() {
        if let Some(idx) = line.find("your price ") {
            let rest = &line[idx + "your price ".len()..];
            let num: String = rest
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            if let Ok(v) = num.parse::<f64>() {
                found = Some(v);
            }
        }
    }
    found
}

/// プロンプト中の数値を `marker` 直後から拾う (mock ポリシー用)．
///
/// 小数点は 1 個だけ取り込む (文末ピリオド `8.00.` の 2 個目の `.` は数値に含めない)．
fn number_after(prompt: &str, marker: &str) -> Option<f64> {
    let idx = prompt.find(marker)?;
    let rest = &prompt[idx + marker.len()..];
    let mut num = String::new();
    let mut seen_dot = false;
    for c in rest.trim_start().chars() {
        if c.is_ascii_digit() {
            num.push(c);
        } else if c == '.' && !seen_dot {
            seen_dot = true;
            num.push(c);
        } else {
            break;
        }
    }
    // 末尾が小数点 (例: 文末の "8.") なら除く．
    let trimmed = num.trim_end_matches('.');
    trimmed.parse::<f64>().ok()
}

/// プロンプトに刻まれた «その企業の» Bertrand/cartel 中点を拾う (mock ポリシー用)．
///
/// `pricing_prompt` は各社のプロンプトに «Bertrand price is around p^B_i and the
/// fully collusive ... price is around p^M_i» を埋め込む (firm_id 依存)．mock は
/// この per-firm 値を読み，その企業固有の中点へ寄せる → 非対称コストでは各社の
/// 中点が異なるため価格も非対称に分かれる (= 非対称コスト → 非対称価格を mock でも
/// 構造的に再現)．取れない場合は引数 `fallback` (全社平均中点) を使う．
fn prompt_midpoint(prompt: &str, fallback: f64) -> f64 {
    let b = number_after(prompt, "Bertrand) price is around ");
    let m = number_after(prompt, "monopoly/cartel) price is around ");
    match (b, m) {
        (Some(b), Some(m)) => 0.5 * (b + m),
        _ => fallback,
    }
}

/// LLM クライアントを組み，モデル名と endpoint を先に取り出す．
///
/// `Run::start` は開始時点で `run.json` を書くので，`llm` ブロックを埋めるには記録を
/// 始める前にクライアントを組んでおく必要がある．mock は in-memory キャッシュしか
/// 持たないので，`save()` を呼ばないよう `cache_path` を落とした設定も一緒に返す．
///
/// mock ポリシーは «直近自社価格を中点 (Bertrand と cartel の中間) へ 50% 寄せる»
/// 暗黙の共謀風挙動で，ネットワーク遮断サンドボックスでも価格軌跡を生成できる
/// (ライブ LLM 呼び出し 0)．
fn build_client(cfg: &Config, mock: bool) -> (Config, SabmClient, String, String) {
    let (cfg, client) = if mock {
        let bench = compute_benchmarks(cfg);
        let target = 0.5 * (mean(&bench.p_bertrand) + mean(&bench.p_cartel));
        let backend = ScriptedClient::new("mock-llama3.2", move |prompt: &str| {
            // 会話フェーズのプロンプト (## Message 節) には «協調の打診» メッセージを返す．
            if prompt.contains("## Message") {
                return "{\"message\": \"Let's both hold prices high to keep margins up.\"}"
                    .to_string();
            }
            // 価格決定: その企業固有の (Bertrand, cartel) 中点へ 50% 寄せる «暗黙の共謀»
            // 風ポリシー．per-firm 中点を読むので非対称コストでは価格も非対称に分かれる．
            let firm_target = prompt_midpoint(prompt, target);
            let last = last_own_price(prompt).unwrap_or(firm_target);
            let next = last + 0.5 * (firm_target - last);
            format!("{{\"price\": {next:.4}}}")
        });
        let mut mock_cfg = cfg.clone();
        mock_cfg.llm.cache_path = None;
        (mock_cfg, wrap_client(backend, PromptCache::in_memory()))
    } else {
        let client = build_live_client(&cfg.llm)
            .unwrap_or_else(|e| panic!("LLM クライアント構築に失敗: {e}"));
        (cfg.clone(), client)
    };
    let model = client.inner().model().to_string();
    let endpoint = client.inner().endpoint().to_string();
    (cfg, client, model, endpoint)
}

/// キャッシュファイルの親ディレクトリを用意する．
fn ensure_cache_dir(cache_path: &str) {
    if let Some(parent) = Path::new(cache_path).parent() {
        let _ = fs::create_dir_all(parent);
    }
}

/// `benchmark` サブコマンドの実験条件．
///
/// 解析計算に必要な需要・コストのパラメータだけを持つ．シードも LLM も要らない．
#[derive(Serialize)]
struct BenchmarkParameters {
    n_firms: usize,
    a: f64,
    beta: f64,
    d: f64,
    d_over_beta: f64,
    alpha: f64,
    b: f64,
    c1: f64,
    c2: f64,
}

/// スイープ親 run の実験条件 (グリッド定義そのもの)．
#[derive(Serialize)]
struct SweepParameters {
    d_beta_values: Vec<f64>,
    firms_values: Vec<usize>,
    a: f64,
    beta: f64,
    cost: f64,
    persona: String,
    communication: bool,
    max_rounds: usize,
    runs: usize,
    seed: u64,
    llm_temperature: f32,
    llm_seed: u64,
    mock: bool,
}

/// スイープの子 run (firms × d/β の 1 点) の実験条件．
///
/// `run` の条件に `runs` が付いた形で，`run` とは別のサブコマンド名を持つ．同じ
/// `run` を名乗らせると，«1 本のシミュレーション» と «同一条件の `runs` 本» という
/// 中身の違う 2 つが 1 つの名前に同居し，`runvault path --subcommand run` がどちらを
/// 返すか分からなくなる．
#[derive(Serialize)]
struct SweepPointParameters {
    firms: usize,
    d_beta: f64,
    a: f64,
    beta: f64,
    cost: f64,
    persona: String,
    communication: bool,
    max_rounds: usize,
    runs: usize,
    seed: u64,
    llm_temperature: f32,
    llm_seed: u64,
    mock: bool,
}

/// `reproduce` の実験条件．
#[derive(Serialize)]
struct ReproduceParameters {
    n_firms: usize,
    a: f64,
    beta: f64,
    d: f64,
    d_over_beta: f64,
    c1: f64,
    c2: f64,
    persona: String,
    scenarios: Vec<&'static str>,
    max_rounds: usize,
    seed: u64,
    llm_temperature: f32,
    llm_seed: u64,
    mock: bool,
    quick: bool,
}

// ---------------------------------------------------------------------------
// benchmark
// ---------------------------------------------------------------------------

fn cmd_benchmark(args: BenchmarkArgs) {
    let cfg = Config {
        n_firms: args.market.firms,
        a: args.market.a,
        beta: args.market.beta,
        d: args.market.effective_d(),
        c1: args.market.c1,
        c2: args.market.c2,
        ..Config::default()
    };
    let bench = compute_benchmarks(&cfg);
    let dp = cfg.demand_params();

    let parameters = BenchmarkParameters {
        n_firms: cfg.n_firms,
        a: cfg.a,
        beta: cfg.beta,
        d: cfg.d,
        d_over_beta: dp.d_over_beta(),
        alpha: dp.alpha,
        b: dp.b,
        c1: cfg.c1,
        c2: cfg.c2,
    };

    // domain = analysis．需要パラメータだけから決まる純粋な数値計算で RNG を使わない
    // ので，master_seed は持たない (simulation を名乗ると必須になり，存在しないシードを
    // 書くことになる)．
    let mut rv = Run::start(
        RunOptions::new(EXPERIMENT, "benchmark")
            .repo_id(REPO_ID)
            .domain(DOMAIN_ANALYSIS)
            .results_root(&args.output_dir)
            .parameters(&parameters)
            .expect("runvault: parameters の組み立てに失敗")
            .replication(record::replication()),
    )
    .expect("runvault: benchmark run の開始に失敗");

    record::log_benchmark_means(&mut rv, &bench);
    record::log_benchmarks(&mut rv, None, &bench);

    println!("=== Han, Wu & Xiao (2023) SABM 解析ベンチマーク ===");
    println!(
        "n={} | a={} β={:.6} d={:.6} (d/β={:.3}) | α={:.6} b={:.8} | c1={} c2={}",
        cfg.n_firms,
        cfg.a,
        cfg.beta,
        cfg.d,
        dp.d_over_beta(),
        dp.alpha,
        dp.b,
        cfg.c1,
        cfg.c2,
    );
    println!("-------------------------------------------------");
    for (i, (b, m)) in bench
        .p_bertrand
        .iter()
        .zip(bench.p_cartel.iter())
        .enumerate()
    {
        println!("firm {i}: p_bertrand = {b:.6}  |  p_cartel = {m:.6}");
    }
    println!(
        "mean : p_bertrand = {:.6}  |  p_cartel = {:.6}",
        mean(&bench.p_bertrand),
        mean(&bench.p_cartel)
    );
    println!("-------------------------------------------------");

    let dir = rv.finish().expect("runvault: benchmark run の完了に失敗");
    println!("企業ごとの値 → {}/events.jsonl", dir.display());
    println!("全社平均     → {}/metrics.csv", dir.display());
    println!("設定         → {}/config.json", dir.display());
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

fn cmd_run(args: RunArgs) {
    let persona = parse_persona(&args.persona).unwrap_or_else(|e| panic!("{}", e));

    if !args.mock {
        ensure_cache_dir(&args.cache_path);
    }

    let base_seed = args.seed.unwrap_or(42);
    let runs = args.runs.max(1);

    // 代表 (最後の) 試行の条件でクライアントを組み，モデル名を先に取る．
    let base_cfg = Config {
        n_firms: args.market.firms,
        a: args.market.a,
        beta: args.market.beta,
        d: args.market.effective_d(),
        c1: args.market.c1,
        c2: args.market.c2,
        persona,
        communication: args.communication,
        max_rounds: args.max_rounds,
        reflection_period: args.reflection_period,
        memory_window: args.memory_window,
        seed: Some(base_seed),
        llm: LlmSettings {
            temperature: args.temperature,
            seed: args.llm_seed,
            cache_path: Some(args.cache_path.clone()),
        },
        ..Config::default()
    };
    let (_, probe, model, endpoint) = build_client(&base_cfg, args.mock);
    drop(probe);

    // 記録する `seed` は base seed である．試行ごとの派生シードは terminal 行の
    // `trial_seed` が持つ (`--runs` が 1 本でも derive_run_seed(base, 0) を通る)．
    let parameters = RunParameters::new(&base_cfg, runs, base_seed, args.mock);

    let mut rv = Run::start(
        RunOptions::new(EXPERIMENT, "run")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&parameters)
            .expect("runvault: parameters の組み立てに失敗")
            .seed_pointers(["/seed"])
            .master_seed(base_seed)
            .llm(record::llm_block(&model, &endpoint, args.temperature))
            .replication(record::replication()),
    )
    .expect("runvault: run の開始に失敗");

    println!("=== Han, Wu & Xiao (2023) SABM 暗黙の共謀 再現実験 ===");
    println!(
        "firms: {} | persona: {} | communication: {} | max_rounds: {} | runs: {}",
        args.market.firms,
        persona.label(),
        args.communication,
        args.max_rounds,
        runs,
    );
    println!(
        "demand: a={} β={:.6} d={:.6} (d/β={:.3}) | c1={} c2={}",
        args.market.a,
        args.market.beta,
        args.market.effective_d(),
        args.market.effective_d() / args.market.beta,
        args.market.c1,
        args.market.c2,
    );
    println!(
        "LLM: model={} temp={} llm_seed={} cache={} | seed (base): {}",
        model,
        args.temperature,
        args.llm_seed,
        if args.mock {
            "(in-memory)"
        } else {
            args.cache_path.as_str()
        },
        base_seed,
    );
    println!("出力先: {}", rv.dir().display());
    println!("-------------------------------------------------");

    // 進捗の 1 単位は 1 ラウンド．費用がそこにあり，1 ラウンドは全企業に価格を
    // 尋ね，会話ありならその前にメッセージも尋ねる — どちらも 1 社 1 回の
    // モデル呼び出しになる．試行を単位にすると，ライブの 1 本は 0/1 と出したきり
    // 終わりまで黙る．
    //
    // 分母は持たない．`ProfitReward` は平均価格系列が定常になった時点で
    // `request_stop()` するので，max_rounds は «上限» であって仕事の量ではない．
    // mock の実測では `--max-rounds 300` の 1 本が 40 ラウンドで止まった (13%)．
    // ライブで何ラウンド目に止まるかは走らせるまで分からないので，上限を分母に
    // 置けば見積もりは最後まで数倍長いままになり，stage は 13% で «done» と
    // 閉じることになる．数えた分だけを出す．
    let mut stage = rv.unbounded_stage("rounds");
    let mut last: Option<(SimulationResult, u64)> = None;

    for run_idx in 0..runs {
        let seed = derive_run_seed(base_seed, run_idx);
        let cfg = Config {
            seed: Some(seed),
            ..base_cfg.clone()
        };
        let (cfg, client, _, _) = build_client(&cfg, args.mock);
        let result = run_with_client_observed(&cfg, client, |_| stage.tick())
            .unwrap_or_else(|e| panic!("実行に失敗: {}", e));

        // 詳細を残すのは最後の試行 (代表 run) だけ．移行前も同じで，先行する試行は
        // キャッシュを温めるだけだった．
        if run_idx + 1 == runs {
            last = Some((result, seed));
        }
    }

    // manifest.csv は finish() で封をされる．その後に 1 行足せば，manifest が
    // 食い違うダイジェストを持つことになる．
    stage.close();

    let (result, seed) = last.expect("試行が 1 本もありません");
    for m in &result.metrics {
        record::log_round(&mut rv, None, m);
    }
    record::log_run_scalars(&mut rv, None, &result);
    record::log_benchmarks(&mut rv, None, &result.benchmarks);
    record::log_firms(&mut rv, None, seed, base_cfg.max_rounds, &result);

    let stable = result
        .rounds_to_stable()
        .map(|r| r.to_string())
        .unwrap_or_else(|| "未達 (no stable convergence within max_rounds)".to_string());
    println!(
        "最終ラウンド: {} | 平均価格: {:.3} | CI: {:.3} (0=Bertrand, 1=cartel)",
        result.final_round,
        result.final_avg_price(),
        result.final_collusion_index(),
    );
    println!(
        "ベンチマーク: p_bertrand={:.3} p_cartel={:.3} | 安定到達ラウンド: {}",
        mean(&result.benchmarks.p_bertrand),
        mean(&result.benchmarks.p_cartel),
        stable,
    );
    println!(
        "LLM 呼び出し: {} 回 | cache-hit: {} ({:.1}%) | model: {}",
        result.metadata.total(),
        result.metadata.cache_hits(),
        result.metadata.cache_hit_rate() * 100.0,
        result.llm_model,
    );

    let dir = rv.finish().expect("runvault: run の完了に失敗");
    println!("ラウンド指標   → {}/metrics.csv", dir.display());
    println!("企業ごとの軌跡 → {}/events.jsonl", dir.display());
    println!("設定           → {}/config.json", dir.display());
}

// ---------------------------------------------------------------------------
// sweep
// ---------------------------------------------------------------------------

fn cmd_sweep(args: SweepArgs) {
    let persona = parse_persona(&args.persona).unwrap_or_else(|e| panic!("{}", e));
    let d_beta_values: Vec<f64> = split_csv(&args.d_beta_values)
        .iter()
        .map(|s| {
            s.parse::<f64>()
                .unwrap_or_else(|_| panic!("不正な d/β: {s}"))
        })
        .collect();
    let firms_values: Vec<usize> = split_csv(&args.firms_values)
        .iter()
        .map(|s| {
            s.parse::<usize>()
                .unwrap_or_else(|_| panic!("不正な firms: {s}"))
        })
        .collect();

    if !args.mock {
        ensure_cache_dir(&args.cache_path);
    }

    let n_total = d_beta_values.len() * firms_values.len() * args.runs;

    // llm ブロックのためにモデル名を先に取る (親子で同じ値を共有する)．
    let probe_cfg = Config {
        n_firms: firms_values[0],
        a: args.a,
        beta: args.beta,
        d: d_beta_values[0] * args.beta,
        c1: args.cost,
        c2: args.cost,
        persona,
        communication: args.communication,
        max_rounds: args.max_rounds,
        seed: Some(args.seed),
        llm: LlmSettings {
            temperature: args.temperature,
            seed: args.llm_seed,
            cache_path: Some(args.cache_path.clone()),
        },
        ..Config::default()
    };
    let (_, probe, model, endpoint) = build_client(&probe_cfg, args.mock);
    drop(probe);
    let llm = || record::llm_block(&model, &endpoint, args.temperature);

    let sweep_parameters = SweepParameters {
        d_beta_values: d_beta_values.clone(),
        firms_values: firms_values.clone(),
        a: args.a,
        beta: args.beta,
        cost: args.cost,
        persona: persona.label().to_string(),
        communication: args.communication,
        max_rounds: args.max_rounds,
        runs: args.runs,
        seed: args.seed,
        llm_temperature: args.temperature,
        llm_seed: args.llm_seed,
        mock: args.mock,
    };

    // 親 run: グリッド定義そのものを parameters に持つ．個別条件の指標は書かない．
    // 親は 1 本のシミュレーションではないので master_seed を名乗らず，base seed は
    // /parameters.seed と seed_pointers 経由で execution_hash に残る．
    let parent = Run::start(
        RunOptions::new(EXPERIMENT, "sweep")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&sweep_parameters)
            .expect("runvault: sweep の parameters の組み立てに失敗")
            .seed_pointers(["/seed"])
            .sweep_parent()
            .llm(llm())
            .replication(record::replication()),
    )
    .expect("runvault: sweep 親 run の開始に失敗");

    let sweep_id = parent
        .sweep_id()
        .expect("runvault: sweep 親に sweep_id がありません")
        .to_string();
    let parent_run_uid = parent.run_uid().to_string();

    println!("=== Han, Wu & Xiao (2023) SABM パラメータスイープ ===");
    println!(
        "d/β: {} 種 | firms: {} 種 | persona: {} | 試行: {} | 合計: {} 実行",
        d_beta_values.len(),
        firms_values.len(),
        persona.label(),
        args.runs,
        n_total,
    );
    println!("シード (base): {} | model: {}", args.seed, model);
    println!("出力先: {}", parent.dir().display());
    println!("-----------------------------------------------------------");

    // 掃引全体で stage を 1 つ．条件ごとに開け直すと小さな tally が並ぶだけで，
    // 掃引全体のどこにいるかは分からない．
    //
    // 企業数で分けることはしない．1 ラウンドのモデル呼び出し数は企業数に比例する
    // が，既定の `--firms-values 2,3` では 1.5 倍しか違わない．しかも stage は
    // 分母を持たない (`ProfitReward` が定常判定で早期に止めるので max_rounds は
    // 上限でしかない) ので，守るべき割合も見積もりも無い — 分けても «いまどこか»
    // を名前で言う以外の働きがなく，そのぶん全体の進み方が読めなくなる．
    let mut stage = parent.unbounded_stage("rounds");

    let mut done = 0usize;
    // d/β ごとの平均 collusion index (表示のためだけに持つ)．
    let mut per_d_beta: Vec<(f64, Vec<f64>)> =
        d_beta_values.iter().map(|&d| (d, Vec::new())).collect();

    for &firms in &firms_values {
        for &d_beta in &d_beta_values {
            let params = SweepPointParameters {
                firms,
                d_beta,
                a: args.a,
                beta: args.beta,
                cost: args.cost,
                persona: persona.label().to_string(),
                communication: args.communication,
                max_rounds: args.max_rounds,
                runs: args.runs,
                seed: args.seed,
                llm_temperature: args.temperature,
                llm_seed: args.llm_seed,
                mock: args.mock,
            };

            // 子は «その条件の試行群» そのもの．master_seed は親と同じ base で，
            // 条件が違えば config_hash が違うので run としては別物になる．
            // 同じ条件の繰り返しは無いので replicate_index は 0．
            let mut child = Run::start(
                RunOptions::new(EXPERIMENT, "sweep-point")
                    .repo_id(REPO_ID)
                    .domain(DOMAIN)
                    .results_root(&args.output_dir)
                    .parameters(&params)
                    .expect("runvault: 子 run の parameters の組み立てに失敗")
                    .seed_pointers(["/seed"])
                    .master_seed(args.seed)
                    .replicate_index(0)
                    .llm(llm())
                    .lineage(Lineage {
                        sweep_id: Some(sweep_id.clone()),
                        parent_run_uid: Some(parent_run_uid.clone()),
                        ..Default::default()
                    })
                    .replication(record::replication()),
            )
            .expect("runvault: 子 run の開始に失敗");

            let mut trials: Vec<record::TrialOutcome> = Vec::with_capacity(args.runs);
            for run_idx in 0..args.runs {
                // 各条件に独立なシードを派生 (explicit identity)．移行前と同じ引数・
                // 同じ順序で derive_seed を呼ぶ．
                let seed = socsim_core::derive_seed(
                    args.seed,
                    &[firms as u64, (d_beta * 1000.0) as u64, run_idx as u64],
                );
                let cfg = Config {
                    n_firms: firms,
                    a: args.a,
                    beta: args.beta,
                    d: d_beta * args.beta,
                    c1: args.cost,
                    c2: args.cost,
                    persona,
                    communication: args.communication,
                    max_rounds: args.max_rounds,
                    seed: Some(seed),
                    llm: LlmSettings {
                        temperature: args.temperature,
                        seed: args.llm_seed,
                        cache_path: Some(args.cache_path.clone()),
                    },
                    ..Config::default()
                };

                let (cfg, client, _, _) = build_client(&cfg, args.mock);
                let result = run_with_client_observed(&cfg, client, |_| stage.tick())
                    .unwrap_or_else(|e| panic!("実行に失敗: {}", e));

                let outcome = record::log_trial(
                    &mut child,
                    &format!("trial-{run_idx}"),
                    seed,
                    args.max_rounds,
                    &result,
                );
                if let Some(slot) = per_d_beta
                    .iter_mut()
                    .find(|(d, _)| (*d - d_beta).abs() < 1e-9)
                {
                    slot.1.push(outcome.final_collusion_index);
                }
                trials.push(outcome);

                done += 1;
            }
            record::log_condition_summary(&mut child, &trials);
            child.finish().expect("runvault: 子 run の完了に失敗");

            println!(
                "[{}/{}] firms={} d/β={:.3} 完了 ({} 試行)",
                done, n_total, firms, d_beta, args.runs,
            );
        }
    }

    // manifest.csv は finish() で封をされる．その後に 1 行足せば，manifest が
    // 食い違うダイジェストを持つことになる．
    stage.close();

    let parent_dir = parent
        .finish()
        .expect("runvault: sweep 親 run の完了に失敗");

    println!("===========================================================");
    println!("スイープ完了: {} 実行", n_total);
    println!("-----------------------------------------------------------");
    println!("d/β 別の平均 collusion index (差別化度↑ → 共謀しにくい傾向):");
    for (d_beta, cis) in &per_d_beta {
        if cis.is_empty() {
            continue;
        }
        println!("  d/β={d_beta:<5.3} → CI = {:.3}", mean(cis));
    }
    println!("-----------------------------------------------------------");
    println!("親 run → {}", parent_dir.display());
    println!("条件ごとの子 run は subcommand=sweep-point (試行は events.jsonl)．");
}

// ---------------------------------------------------------------------------
// reproduce (論文 Fig.1/2/4 一括再現)
// ---------------------------------------------------------------------------

/// 1 シナリオ (会話なし / 会話あり) の観測値．
struct ReproduceScenario {
    /// シナリオ名 (指標名の接頭辞になる slug)．
    name: &'static str,
    /// 観測した最終ラウンドの全社平均価格．
    observed_avg_price: f64,
    /// 観測した最終ラウンドの collusion index．
    observed_collusion_index: f64,
    /// 安定 (共謀) 到達ラウンド (未達なら `None`)．
    rounds_to_stable: Option<usize>,
    /// socsim エンジンのクロックの最終値．
    final_round: usize,
    /// 解析 Bertrand 価格 (全社平均)．
    p_bertrand: f64,
    /// 解析 cartel 価格 (全社平均)．
    p_cartel: f64,
}

/// 観測値と論文の値を突き合わせた 1 アンカー．
struct ReproduceAnchor {
    /// 指標名になる slug．
    id: String,
    /// 人間向けの説明 (どの量を見ているか)．
    label: String,
    /// 論文側の値・主張．
    paper_value: String,
    observed: f64,
    target_lo: f64,
    /// 上限．`None` は «上限なし»．
    target_hi: Option<f64>,
    pass: bool,
}

fn cmd_reproduce(args: ReproduceArgs) {
    let persona = parse_persona(&args.persona).unwrap_or_else(|e| panic!("{}", e));
    let max_rounds = if args.quick { 80 } else { args.max_rounds };

    if !args.mock {
        ensure_cache_dir(&args.cache_path);
    }

    // 2 シナリオ: 会話なし (Fig.1) と 会話あり (Fig.2)．Fig.4 = 両者の収束水準/速度の差．
    let scenarios_spec: [(&'static str, bool); 2] =
        [("no_communication", false), ("communication", true)];

    let base_cfg = Config {
        n_firms: args.market.firms,
        a: args.market.a,
        beta: args.market.beta,
        d: args.market.effective_d(),
        c1: args.market.c1,
        c2: args.market.c2,
        persona,
        communication: false,
        max_rounds,
        seed: Some(args.seed),
        llm: LlmSettings {
            temperature: args.temperature,
            seed: args.llm_seed,
            cache_path: Some(args.cache_path.clone()),
        },
        ..Config::default()
    };
    let (_, probe, model, endpoint) = build_client(&base_cfg, args.mock);
    drop(probe);

    let dp = base_cfg.demand_params();
    let parameters = ReproduceParameters {
        n_firms: base_cfg.n_firms,
        a: base_cfg.a,
        beta: base_cfg.beta,
        d: base_cfg.d,
        d_over_beta: dp.d_over_beta(),
        c1: base_cfg.c1,
        c2: base_cfg.c2,
        persona: persona.label().to_string(),
        scenarios: scenarios_spec.iter().map(|(n, _)| *n).collect(),
        max_rounds,
        seed: args.seed,
        llm_temperature: args.temperature,
        llm_seed: args.llm_seed,
        mock: args.mock,
        quick: args.quick,
    };

    let mut rv = Run::start(
        RunOptions::new(EXPERIMENT, "reproduce")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&parameters)
            .expect("runvault: parameters の組み立てに失敗")
            .seed_pointers(["/seed"])
            .master_seed(args.seed)
            .llm(record::llm_block(&model, &endpoint, args.temperature))
            .replication(record::replication()),
    )
    .expect("runvault: reproduce run の開始に失敗");

    println!("=== Han, Wu & Xiao (2023) SABM 論文 Fig.1/2/4 一括再現 ===");
    println!(
        "persona: {} | max_rounds: {} | mock: {} | quick: {}",
        persona.label(),
        max_rounds,
        args.mock,
        args.quick,
    );
    println!("出力先: {}", rv.dir().display());
    println!("-------------------------------------------------");

    let mut scenarios: Vec<ReproduceScenario> = Vec::new();

    // 2 シナリオを通して stage は 1 つ．シナリオごとに開け直すと小さな tally が
    // 2 つ並ぶだけで，reproduce 全体の進み方が読めなくなる．`communication = false`
    // のとき CommunicationPhase は no-op なので 1 ラウンドの呼び出し数は半分に
    // なるが，2 倍の開きで，しかも stage は分母を持たない — 守るべき見積もりが
    // 無い以上，分ける理由にはならない．
    let mut stage = rv.unbounded_stage("rounds");

    for (name, communication) in scenarios_spec {
        let seed = socsim_core::derive_seed(args.seed, &[communication as u64]);
        let cfg = Config {
            communication,
            seed: Some(seed),
            ..base_cfg.clone()
        };
        let (cfg, client, _, _) = build_client(&cfg, args.mock);
        let result = run_with_client_observed(&cfg, client, |_| stage.tick())
            .unwrap_or_else(|e| panic!("実行に失敗: {}", e));

        // 2 シナリオが 1 本の run に同居するので，(step, scope, name) が衝突しない
        // ようシナリオ名を指標名と unit_id の接頭辞にする．
        for m in &result.metrics {
            record::log_round(&mut rv, Some(name), m);
        }
        record::log_run_scalars(&mut rv, Some(name), &result);
        record::log_benchmarks(&mut rv, Some(name), &result.benchmarks);
        record::log_firms(&mut rv, Some(name), seed, max_rounds, &result);

        scenarios.push(ReproduceScenario {
            name,
            observed_avg_price: result.final_avg_price(),
            observed_collusion_index: result.final_collusion_index(),
            rounds_to_stable: result.rounds_to_stable(),
            final_round: result.final_round,
            p_bertrand: mean(&result.benchmarks.p_bertrand),
            p_cartel: mean(&result.benchmarks.p_cartel),
        });
    }

    // manifest.csv は finish() で封をされる．その後に 1 行足せば，manifest が
    // 食い違うダイジェストを持つことになる．
    stage.close();

    let base = &scenarios[0]; // no_communication = 論文 Fig.1 の基本ケース．
    let p_bertrand = base.p_bertrand;
    let p_cartel = base.p_cartel;

    // --- アンカー判定 (Bertrand(6)/cartel(8) フレーム) ---
    let mut anchors: Vec<ReproduceAnchor> = Vec::new();
    let mut push = |id: &str, label: &str, paper: &str, obs: f64, lo: f64, hi: Option<f64>| {
        anchors.push(ReproduceAnchor {
            id: id.to_string(),
            label: label.to_string(),
            paper_value: paper.to_string(),
            observed: obs,
            target_lo: lo,
            target_hi: hi,
            pass: obs >= lo && hi.map(|h| obs <= h).unwrap_or(true),
        });
    };

    // Fig.1 中核: 会話なしでも平均価格が (Bertrand, cartel) 区間内 (~7 に収束)．
    push(
        "no_comm_avg_price_in_band",
        "no_comm avg_price in (p^B, p^M) (paper ~7)",
        "~7.0",
        base.observed_avg_price,
        p_bertrand,
        Some(p_cartel),
    );
    // 暗黙の共謀: CI が中域 (論文帯 0.3-0.8)．
    push(
        "no_comm_collusion_index",
        "no_comm collusion_index (paper 0.3-0.8)",
        "0.3-0.8",
        base.observed_collusion_index,
        0.3,
        Some(0.8),
    );
    // Fig.2/4: 会話ありは共謀を弱めない (CI が会話なし以上 — εマージン)．
    let comm = &scenarios[1];
    push(
        "communication_strengthens_collusion",
        "communication strengthens collusion (CI_comm >= CI_nocomm)",
        "comm >= no-comm",
        comm.observed_collusion_index - base.observed_collusion_index,
        -0.05,
        None,
    );

    let n_pass = anchors.iter().filter(|a| a.pass).count();
    let n_anchors = anchors.len();

    // 観測量は数なので指標に，帯と PASS/off はカテゴリなのでイベントに書く．
    let observations: Vec<(String, f64)> =
        anchors.iter().map(|a| (a.id.clone(), a.observed)).collect();
    record::log_anchor_observations(&mut rv, &observations, n_pass);
    for a in &anchors {
        rv.log_event(
            "x.han2023.anchor",
            &record::AnchorEvent {
                id: &a.id,
                label: &a.label,
                paper_value: &a.paper_value,
                observed: a.observed,
                target_lo: a.target_lo,
                target_hi: a.target_hi,
                pass: a.pass,
            },
        )
        .expect("アンカーイベントの記録に失敗");
    }

    println!("シナリオ:");
    for s in &scenarios {
        let stable = match s.rounds_to_stable {
            Some(r) => r.to_string(),
            None => "未達".to_string(),
        };
        println!(
            "  [{:<16}] avg_price={:.3} CI={:.3} 安定到達={} (round {})",
            s.name, s.observed_avg_price, s.observed_collusion_index, stable, s.final_round,
        );
    }
    println!("-------------------------------------------------");
    println!(
        "フレーム: p_bertrand={:.3} p_cartel={:.3}",
        p_bertrand, p_cartel
    );
    for a in &anchors {
        let hi = match a.target_hi {
            Some(h) => format!("{:.2}", h),
            None => "∞".to_string(),
        };
        println!(
            "[{}] {:<48} obs={:.4} target=[{:.2},{}] paper={}",
            if a.pass { "PASS" } else { "OFF " },
            a.label,
            a.observed,
            a.target_lo,
            hi,
            a.paper_value,
        );
    }
    println!("-------------------------------------------------");
    println!("{}/{} アンカーが in-band", n_pass, n_anchors);

    let dir = rv.finish().expect("runvault: reproduce run の完了に失敗");
    println!("シナリオ別の指標 → {}/metrics.csv", dir.display());
    println!("アンカー判定     → {}/events.jsonl", dir.display());
    println!("図 (Fig.1/2/4 風) は `uv run sabm-tools reproduce` で生成できる．");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    if let Some(host) = cli.ollama_host.as_deref() {
        std::env::set_var("OLLAMA_HOST", host);
    }
    match cli.command {
        Commands::Benchmark(args) => cmd_benchmark(args),
        Commands::Run(args) => cmd_run(args),
        Commands::Sweep(args) => cmd_sweep(args),
        Commands::Reproduce(args) => cmd_reproduce(args),
    }
}
