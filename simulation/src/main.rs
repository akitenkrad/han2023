//! Han, Wu & Xiao (2023) "SABM" — 再現実験の CLI エントリポイント．
//!
//! `benchmark` : 解析ベンチマーク (Bertrand 均衡 / cartel 価格) のみを計算する
//!               (LLM 不要・即時)．基本設定で p^B=6, p^M=8 を厳密に出力する．
//! `run`       : 単一設定で LLM 駆動の繰り返し価格ゲームを実行し，暗黙の共謀
//!               (価格が (Bertrand, cartel) 区間へ収束) を観測する．
//! `sweep`     : 差別化度 d/β × 企業数 を走査し，最終 collusion index を
//!               `sweep_summary.csv` に集計する．
//! `reproduce` : 論文 Fig.1/2/4 一括再現 (Phase 3; 現状はスタブで案内のみ)．

use std::fs;
use std::path::Path;

use clap::{Parser, Subcommand};
use socsim_results::{ensure_dir, refresh_latest_symlink, timestamp, write_csv, write_json};

use sabm_simulation::config::{derive_run_seed, Config, LlmSettings};
use sabm_simulation::llm::wrap_client;
use sabm_simulation::simulation::{
    compute_benchmarks, ensure_output_dir, run, run_with_client, save_benchmarks, save_llm_meta,
    save_metrics, save_rounds, SimulationResult,
};
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

/// `sweep_summary.csv` の 1 行 (条件ごとの最終 collusion index)．
#[derive(serde::Serialize)]
struct SweepRow {
    firms: usize,
    d_beta: f64,
    run: usize,
    seed: u64,
    final_round: usize,
    final_avg_price: f64,
    final_collusion_index: f64,
    p_bertrand_mean: f64,
    p_cartel_mean: f64,
    rounds_to_stable: i64,
    cache_hit_rate: f64,
}

/// `sweep_config.json` の構造体．
#[derive(serde::Serialize)]
struct SweepConfigJson {
    command: &'static str,
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
}

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

/// 1 設定を実行する (`--mock` ならライブ LLM の代わりに scripted mock を使う)．
///
/// mock ポリシーは «直近自社価格を中点 (Bertrand と cartel の中間) へ 50% 寄せる»
/// 暗黙の共謀風挙動で，ネットワーク遮断サンドボックスでも価格軌跡を生成できる
/// (ライブ LLM 呼び出し 0)．
fn run_one(cfg: &Config, mock: bool) -> Result<SimulationResult, String> {
    if mock {
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
        let client = wrap_client(backend, PromptCache::in_memory());
        // mock は in-memory cache なので保存先を持たない (save をスキップさせる)．
        let mock_cfg = Config {
            llm: LlmSettings {
                cache_path: None,
                ..cfg.llm.clone()
            },
            ..cfg.clone()
        };
        run_with_client(&mock_cfg, client)
    } else {
        run(cfg)
    }
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

    let timestamp = timestamp();
    let output_dir = format!("{}/{}", args.output_dir, timestamp);
    ensure_output_dir(&output_dir);
    save_benchmarks(&bench, &output_dir);
    // config.json も書く (show-experiment-settings で読めるように; pretty-JSON は
    // socsim_results::write_json に委譲, バイト等価)．
    let path = format!("{}/config.json", output_dir);
    write_json(&cfg.to_run_config_json(), &path).expect("config.json の書き込みに失敗");
    let _ = refresh_latest_symlink(&args.output_dir, &timestamp);

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
    println!("benchmarks → {}/benchmarks.json", output_dir);
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

fn cmd_run(args: RunArgs) {
    let persona = parse_persona(&args.persona).unwrap_or_else(|e| panic!("{}", e));

    let timestamp = timestamp();
    let output_dir = format!("{}/{}", args.output_dir, timestamp);

    if let Some(parent) = Path::new(&args.cache_path).parent() {
        let _ = fs::create_dir_all(parent);
    }
    ensure_output_dir(&output_dir);

    println!("=== Han, Wu & Xiao (2023) SABM 暗黙の共謀 再現実験 ===");
    println!(
        "firms: {} | persona: {} | communication: {} | max_rounds: {} | runs: {}",
        args.market.firms,
        persona.label(),
        args.communication,
        args.max_rounds,
        args.runs,
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
        "LLM: temp={} llm_seed={} cache={} | seed: {:?}",
        args.temperature, args.llm_seed, args.cache_path, args.seed
    );
    println!("出力先: {}", output_dir);
    println!("-------------------------------------------------");

    let base_seed = args.seed.unwrap_or(42);
    let mut last_result: Option<SimulationResult> = None;
    let mut last_cfg: Option<Config> = None;

    for run_idx in 0..args.runs.max(1) {
        let seed = derive_run_seed(base_seed, run_idx);
        let cfg = Config {
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
            seed: Some(seed),
            llm: LlmSettings {
                temperature: args.temperature,
                seed: args.llm_seed,
                cache_path: Some(args.cache_path.clone()),
            },
            output_dir: output_dir.clone(),
            ..Config::default()
        };

        let result = run_one(&cfg, args.mock).unwrap_or_else(|e| panic!("実行に失敗: {}", e));

        // 最後の試行の詳細を保存する (代表 run)．
        if run_idx + 1 == args.runs.max(1) {
            save_rounds(&result.rounds, &output_dir);
            save_metrics(&result.metrics, &output_dir);
            save_benchmarks(&result.benchmarks, &output_dir);
            save_llm_meta(&result, &cfg, &output_dir);
            let path = format!("{}/config.json", output_dir);
            write_json(&cfg.to_run_config_json(), &path).expect("config.json の書き込みに失敗");
            last_result = Some(result);
            last_cfg = Some(cfg);
        }
    }

    let _ = refresh_latest_symlink(&args.output_dir, &timestamp);

    if let (Some(result), Some(_cfg)) = (&last_result, &last_cfg) {
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
    }
    println!("ラウンド   → {}/rounds.csv", output_dir);
    println!("メトリクス → {}/metrics.csv", output_dir);
    println!("ベンチマーク → {}/benchmarks.json", output_dir);
    println!("LLM メタ   → {}/llm_meta.json", output_dir);
    println!("設定       → {}/config.json", output_dir);
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

    let timestamp = timestamp();
    let sweep_dir = format!("{}/{}_sweep", args.output_dir, timestamp);
    ensure_dir(&sweep_dir).expect("sweep ディレクトリの作成に失敗");
    if let Some(parent) = Path::new(&args.cache_path).parent() {
        let _ = fs::create_dir_all(parent);
    }

    let n_total = d_beta_values.len() * firms_values.len() * args.runs;

    println!("=== Han, Wu & Xiao (2023) SABM パラメータスイープ ===");
    println!(
        "d/β: {} 種 | firms: {} 種 | persona: {} | 試行: {} | 合計: {} 実行",
        d_beta_values.len(),
        firms_values.len(),
        persona.label(),
        args.runs,
        n_total,
    );
    println!("出力先: {}", sweep_dir);
    println!("-----------------------------------------------------------");

    let mut summary_rows: Vec<SweepRow> = Vec::with_capacity(n_total);
    let mut done = 0usize;

    for &firms in &firms_values {
        for &d_beta in &d_beta_values {
            for run_idx in 0..args.runs {
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
                    output_dir: sweep_dir.clone(),
                    ..Config::default()
                };

                let result =
                    run_one(&cfg, args.mock).unwrap_or_else(|e| panic!("実行に失敗: {}", e));
                summary_rows.push(SweepRow {
                    firms,
                    d_beta,
                    run: run_idx,
                    seed,
                    final_round: result.final_round,
                    final_avg_price: result.final_avg_price(),
                    final_collusion_index: result.final_collusion_index(),
                    p_bertrand_mean: mean(&result.benchmarks.p_bertrand),
                    p_cartel_mean: mean(&result.benchmarks.p_cartel),
                    rounds_to_stable: result.rounds_to_stable().map(|r| r as i64).unwrap_or(-1),
                    cache_hit_rate: result.metadata.cache_hit_rate(),
                });
                done += 1;
            }
            println!(
                "[{}/{}] firms={} d/β={:.3} 完了 ({} 試行)",
                done, n_total, firms, d_beta, args.runs,
            );
        }
    }

    // sweep_summary.csv (各行を serialize; socsim_results::write_csv に委譲, バイト等価)．
    {
        let path = format!("{}/sweep_summary.csv", sweep_dir);
        write_csv(&summary_rows, &path).expect("sweep_summary.csv の書き込みに失敗");
    }

    // sweep_config.json
    {
        let config_json = SweepConfigJson {
            command: "sweep",
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
        };
        let path = format!("{}/sweep_config.json", sweep_dir);
        write_json(&config_json, &path).expect("sweep_config.json の書き込みに失敗");
    }

    let _ = refresh_latest_symlink(&args.output_dir, &format!("{}_sweep", timestamp));

    println!("===========================================================");
    println!("スイープ完了: {} 実行", n_total);
    println!("-----------------------------------------------------------");
    println!("d/β 別の平均 collusion index (差別化度↑ → 共謀しにくい傾向):");
    for &d_beta in &d_beta_values {
        let rows: Vec<&SweepRow> = summary_rows
            .iter()
            .filter(|r| (r.d_beta - d_beta).abs() < 1e-9)
            .collect();
        if rows.is_empty() {
            continue;
        }
        let avg_ci = mean(
            &rows
                .iter()
                .map(|r| r.final_collusion_index)
                .collect::<Vec<_>>(),
        );
        println!("  d/β={d_beta:<5.3} → CI = {avg_ci:.3}");
    }
    println!("-----------------------------------------------------------");
    println!("サマリ → {}/sweep_summary.csv", sweep_dir);
    println!("設定   → {}/sweep_config.json", sweep_dir);
}

// ---------------------------------------------------------------------------
// reproduce (論文 Fig.1/2/4 一括再現)
// ---------------------------------------------------------------------------

/// `reproduce_summary.json` の 1 シナリオ行．
#[derive(serde::Serialize)]
struct ReproduceScenario {
    /// シナリオ名 (no_communication / communication)．
    name: &'static str,
    /// 会話の有無．
    communication: bool,
    /// 観測した最終ラウンドの全社平均価格．
    observed_avg_price: f64,
    /// 観測した最終ラウンドの collusion index．
    observed_collusion_index: f64,
    /// 安定 (共謀) 到達ラウンド (なければ -1)．
    rounds_to_stable: i64,
    /// 実行ラウンド数．
    final_round: usize,
    /// 解析 Bertrand 価格 (全社平均)．
    p_bertrand: f64,
    /// 解析 cartel 価格 (全社平均)．
    p_cartel: f64,
    /// この結果を保存したサブディレクトリ (Python の図生成入力)．
    results_subdir: String,
}

/// `reproduce_summary.json` のアンカー判定行．
#[derive(serde::Serialize)]
struct ReproduceAnchor {
    name: String,
    paper_value: String,
    observed: f64,
    target_lo: f64,
    target_hi: f64,
    pass: bool,
}

/// `reproduce_summary.json` のルート．
#[derive(serde::Serialize)]
struct ReproduceSummary {
    command: &'static str,
    paper: &'static str,
    /// 解析アンカー (Bertrand=6, cartel=8; 基本設定)．
    p_bertrand: f64,
    p_cartel: f64,
    mock: bool,
    quick: bool,
    scenarios: Vec<ReproduceScenario>,
    anchors: Vec<ReproduceAnchor>,
    n_pass: usize,
    n_anchors: usize,
}

fn cmd_reproduce(args: ReproduceArgs) {
    let persona = parse_persona(&args.persona).unwrap_or_else(|e| panic!("{}", e));
    let max_rounds = if args.quick { 80 } else { args.max_rounds };

    let ts = timestamp();
    let out_dir = format!("{}/{}_reproduce", args.output_dir, ts);
    ensure_output_dir(&out_dir);
    if let Some(parent) = Path::new(&args.cache_path).parent() {
        let _ = fs::create_dir_all(parent);
    }

    println!("=== Han, Wu & Xiao (2023) SABM 論文 Fig.1/2/4 一括再現 ===");
    println!(
        "persona: {} | max_rounds: {} | mock: {} | quick: {}",
        persona.label(),
        max_rounds,
        args.mock,
        args.quick,
    );
    println!("出力先: {}", out_dir);
    println!("-------------------------------------------------");

    // 2 シナリオ: 会話なし (Fig.1) と 会話あり (Fig.2)．Fig.4 = 両者の収束水準/速度の差．
    let scenarios_spec: [(&'static str, bool); 2] =
        [("no_communication", false), ("communication", true)];

    let mut scenarios: Vec<ReproduceScenario> = Vec::new();

    for (name, communication) in scenarios_spec {
        let subdir = format!("{}/{}", out_dir, name);
        ensure_output_dir(&subdir);
        let seed = socsim_core::derive_seed(args.seed, &[communication as u64]);
        let cfg = Config {
            n_firms: args.market.firms,
            a: args.market.a,
            beta: args.market.beta,
            d: args.market.effective_d(),
            c1: args.market.c1,
            c2: args.market.c2,
            persona,
            communication,
            max_rounds,
            seed: Some(seed),
            llm: LlmSettings {
                temperature: args.temperature,
                seed: args.llm_seed,
                cache_path: Some(args.cache_path.clone()),
            },
            output_dir: subdir.clone(),
            ..Config::default()
        };

        let result = run_one(&cfg, args.mock).unwrap_or_else(|e| panic!("実行に失敗: {}", e));

        save_rounds(&result.rounds, &subdir);
        save_metrics(&result.metrics, &subdir);
        save_benchmarks(&result.benchmarks, &subdir);
        save_llm_meta(&result, &cfg, &subdir);
        let path = format!("{}/config.json", subdir);
        write_json(&cfg.to_run_config_json(), &path).expect("config.json の書き込みに失敗");

        scenarios.push(ReproduceScenario {
            name,
            communication,
            observed_avg_price: result.final_avg_price(),
            observed_collusion_index: result.final_collusion_index(),
            rounds_to_stable: result.rounds_to_stable().map(|r| r as i64).unwrap_or(-1),
            final_round: result.final_round,
            p_bertrand: mean(&result.benchmarks.p_bertrand),
            p_cartel: mean(&result.benchmarks.p_cartel),
            results_subdir: name.to_string(),
        });
    }

    let base = &scenarios[0]; // no_communication = 論文 Fig.1 の基本ケース．
    let p_bertrand = base.p_bertrand;
    let p_cartel = base.p_cartel;

    // --- アンカー判定 (Bertrand(6)/cartel(8) フレーム) ---
    let mut anchors: Vec<ReproduceAnchor> = Vec::new();
    let mut push = |name: &str, paper: &str, obs: f64, lo: f64, hi: f64| {
        anchors.push(ReproduceAnchor {
            name: name.to_string(),
            paper_value: paper.to_string(),
            observed: obs,
            target_lo: lo,
            target_hi: hi,
            pass: obs >= lo && obs <= hi,
        });
    };

    // Fig.1 中核: 会話なしでも平均価格が (Bertrand, cartel) 区間内 (~7 に収束)．
    push(
        "no_comm avg_price in (p^B, p^M) (paper ~7)",
        "~7.0",
        base.observed_avg_price,
        p_bertrand,
        p_cartel,
    );
    // 暗黙の共謀: CI が中域 (論文帯 0.3-0.8)．
    push(
        "no_comm collusion_index (paper 0.3-0.8)",
        "0.3-0.8",
        base.observed_collusion_index,
        0.3,
        0.8,
    );
    // Fig.2/4: 会話ありは共謀を弱めない (CI が会話なし以上 — εマージン)．
    let comm = &scenarios[1];
    push(
        "communication strengthens collusion (CI_comm >= CI_nocomm)",
        "comm >= no-comm",
        comm.observed_collusion_index - base.observed_collusion_index,
        -0.05,
        f64::INFINITY,
    );

    let n_pass = anchors.iter().filter(|a| a.pass).count();
    let n_anchors = anchors.len();

    println!("シナリオ:");
    for s in &scenarios {
        let stable = if s.rounds_to_stable >= 0 {
            s.rounds_to_stable.to_string()
        } else {
            "未達".to_string()
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
        let hi = if a.target_hi.is_infinite() {
            "∞".to_string()
        } else {
            format!("{:.2}", a.target_hi)
        };
        println!(
            "[{}] {:<48} obs={:.4} target=[{:.2},{}] paper={}",
            if a.pass { "PASS" } else { "OFF " },
            a.name,
            a.observed,
            a.target_lo,
            hi,
            a.paper_value,
        );
    }
    println!("-------------------------------------------------");
    println!("{}/{} アンカーが in-band", n_pass, n_anchors);

    let summary = ReproduceSummary {
        command: "reproduce",
        paper: "Han, Wu & Xiao (2023) SABM — Fig.1/2/4 (tacit collusion ~7 between Bertrand 6 and cartel 8)",
        p_bertrand,
        p_cartel,
        mock: args.mock,
        quick: args.quick,
        scenarios,
        anchors,
        n_pass,
        n_anchors,
    };
    let path = format!("{}/reproduce_summary.json", out_dir);
    write_json(&summary, &path).expect("reproduce_summary.json の書き込みに失敗");

    let _ = refresh_latest_symlink(&args.output_dir, &format!("{}_reproduce", ts));

    println!("サマリ → {}/reproduce_summary.json", out_dir);
    println!("各シナリオの rounds.csv / metrics.csv / benchmarks.json を各サブディレクトリに保存しました．");
    println!("図 (Fig.1/2/4 風) は `uv run sabm-tools reproduce` で生成できます．");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Benchmark(args) => cmd_benchmark(args),
        Commands::Run(args) => cmd_run(args),
        Commands::Sweep(args) => cmd_sweep(args),
        Commands::Reproduce(args) => cmd_reproduce(args),
    }
}
