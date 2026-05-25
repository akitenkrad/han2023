//! Mock 駆動のスモーク実行 (ライブ LLM 不要)．
//!
//! ライブ Ollama/OpenAI が使えない環境 (CI・ネットワーク遮断サンドボックス) で
//! 出力パイプライン (rounds.csv / metrics.csv / benchmarks.json / llm_meta.json /
//! config.json) と Python 可視化を検証するための補助バイナリ．
//! `socsim-llm::mock::ScriptedClient` で決定論的に価格を駆動し，本番 `run` と同じ
//! writer で結果を書き出す．LLM ライブ呼び出しは 0 回．
//!
//! ```bash
//! cargo run --release --example mock_smoke -- results
//! ```
//!
//! 擬似挙動: 直前の自社価格を履歴 (プロンプト) から拾い，ベルトラン (6) と
//! カルテル (8) の中間 7 付近へ漸近的に引き上げる «暗黙の共謀» 風の価格を返す．
//! これで価格軌跡が (Bertrand, cartel) 区間内に収束し，CI が 0.5 付近に出る．

use std::env;
use std::fs;

use chrono::Local;

use sabm_simulation::config::Config;
use sabm_simulation::llm::wrap_client;
use sabm_simulation::prompts::parse_price;
use sabm_simulation::simulation::{
    ensure_output_dir, run_with_client, save_benchmarks, save_llm_meta, save_metrics, save_rounds,
};
use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;

fn main() {
    let base = env::args().nth(1).unwrap_or_else(|| "results".to_string());
    let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
    let output_dir = format!("{base}/{timestamp}");

    let cfg = Config {
        n_firms: 2,
        max_rounds: 60,
        seed: Some(42),
        output_dir: output_dir.clone(),
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

    ensure_output_dir(&cfg.output_dir);
    let result = run_with_client(&cfg, client).expect("mock run failed");
    save_rounds(&result.rounds, &cfg.output_dir);
    save_metrics(&result.metrics, &cfg.output_dir);
    save_benchmarks(&result.benchmarks, &cfg.output_dir);
    save_llm_meta(&result, &cfg, &cfg.output_dir);

    // config.json
    let cfg_path = format!("{}/config.json", cfg.output_dir);
    let f = fs::File::create(&cfg_path).unwrap();
    serde_json::to_writer_pretty(f, &cfg.to_run_config_json()).unwrap();

    // latest symlink
    let link = format!("{base}/latest");
    let _ = fs::remove_file(&link);
    #[cfg(unix)]
    let _ = std::os::unix::fs::symlink(&timestamp, &link);

    println!("mock smoke wrote: {output_dir}");
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
