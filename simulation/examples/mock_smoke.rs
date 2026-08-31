//! Mock 駆動のスモーク実行 (ライブ LLM 不要)．
//!
//! ライブ Ollama/OpenAI が使えない環境 (CI・ネットワーク遮断サンドボックス) で
//! 出力パイプライン (run ディレクトリ・metrics.csv・events.jsonl) と Python 可視化を
//! 検証するための補助バイナリ．`socsim-llm::mock::ScriptedClient` で決定論的に価格を
//! 駆動する．LLM ライブ呼び出しは 0 回．
//!
//! ```bash
//! cargo run --release --example mock_smoke -- results
//! ```
//!
//! 擬似挙動: 直前の自社価格を履歴 (プロンプト) から拾い，ベルトラン (6) と
//! カルテル (8) の中間 7 付近へ漸近的に引き上げる «暗黙の共謀» 風の価格を返す．
//! これで価格軌跡が (Bertrand, cartel) 区間内に収束し，CI が 0.5 付近に出る．

use std::env;

use runvault::{Run, RunOptions};
use sabm_simulation::config::Config;
use sabm_simulation::llm::wrap_client;
use sabm_simulation::prompts::parse_price;
use sabm_simulation::record::{self, RunParameters, DOMAIN, EXPERIMENT, REPO_ID};
use sabm_simulation::simulation::run_with_client;
use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;

fn main() {
    let base = env::args().nth(1).unwrap_or_else(|| "results".to_string());
    let seed = 42u64;

    let cfg = Config {
        n_firms: 2,
        max_rounds: 60,
        seed: Some(seed),
        ..Config::default()
    };

    // 履歴の直近自社価格を読み，7.0 (= 中点) へ 50% ずつ寄せる «暗黙の共謀» 風挙動．
    // 履歴が無い (初回) は 6.5 を返す．
    let backend = ScriptedClient::new("mock-llama3.2", |prompt: &str| {
        let last_price = last_own_price(prompt).unwrap_or(6.0);
        let target = 7.0;
        let next = last_price + 0.5 * (target - last_price);
        format!("{{\"price\": {next:.4}}}")
    });
    let client = wrap_client(backend, PromptCache::in_memory());
    // モデル名と endpoint は Run::start より前に要る (llm ブロックは開始時に確定する)．
    let model = client.inner().model().to_string();
    let endpoint = client.inner().endpoint().to_string();

    let parameters = RunParameters::new(&cfg, 1, seed, true);

    let mut rv = Run::start(
        RunOptions::new(EXPERIMENT, "mock-smoke")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&base)
            .parameters(&parameters)
            .expect("runvault: parameters の組み立てに失敗")
            .seed_pointers(["/seed"])
            .master_seed(seed)
            .llm(record::llm_block(&model, &endpoint, cfg.llm.temperature))
            .replication(record::replication()),
    )
    .expect("runvault: run の開始に失敗");

    let result = run_with_client(&cfg, client).expect("mock run failed");
    for m in &result.metrics {
        record::log_round(&mut rv, None, m);
    }
    record::log_run_scalars(&mut rv, None, &result);
    record::log_benchmarks(&mut rv, None, &result.benchmarks);
    record::log_firms(&mut rv, None, seed, cfg.max_rounds, &result);

    let dir = rv.finish().expect("runvault: run の完了に失敗");
    println!("mock smoke wrote: {}", dir.display());
    println!(
        "rounds={} final_round={} final_avg_price={:.3} final_CI={:.3} live_llm_calls=0 cache_hits={}",
        result.rounds.len(),
        result.final_round,
        result.final_avg_price(),
        result.final_collusion_index(),
        result.metadata.cache_hits(),
    );
}

/// プロンプト履歴から «最後の自社価格» を拾う (`your price X.XX` の最後の出現)．
fn last_own_price(prompt: &str) -> Option<f64> {
    let mut found = None;
    for line in prompt.lines() {
        if let Some(idx) = line.find("your price ") {
            let rest = &line[idx + "your price ".len()..];
            // "6.50, quantity ..." の先頭数値を取る．
            let num: String = rest
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            if let Ok(v) = num.parse::<f64>() {
                found = Some(v);
            }
        }
    }
    // parse_price のフォールバックと整合 (cost 以上)．
    found.map(|v| parse_price(&format!("{{\"price\": {v}}}"), 2.0).unwrap_or(v))
}
