//! 初期化と実行ドライバ (SimulationBuilder 配線 + 二層 LLM レイヤ)．
//!
//! 二層決定論を配線する:
//! - **下層 (決定論的 socsim コア)**: `derive_seed(root, &[0])` で世界初期化
//!   (初期価格) の init RNG を，`derive_seed(root, &[1])` で engine RNG を派生する．
//!   市場クリアリング・利益・解析ベンチマーク・メモリ更新は bit 単位で再現する．
//! - **上層 (非決定的 LLM レイヤ)**: [`crate::llm`] のキャッシュ付き
//!   Ollama→OpenAI フォールバッククライアントに閉じ込め，`temperature=0`/`seed`
//!   固定 + プロンプト→応答キャッシュで擬似決定論化する．モデル・endpoint・
//!   温度・seed・cache-hit を `llm_meta.json` に記録する．

use std::cell::RefCell;
use std::rc::Rc;

use rand::Rng;

use socsim_core::{derive_seed, AgentId, SimClock, SimRng};
use socsim_engine::{SequentialScheduler, SimulationBuilder};
use socsim_llm::MetadataCollector;

use crate::analytic::{bertrand_prices, cartel_prices};
use crate::config::Config;
use crate::llm::SabmClient;
use crate::mechanisms::{
    CommunicationPhase, MarketClearing, MarketEnvironment, MemoryUpdate, PricingDecision,
    ProfitReward, SharedClient, SharedMetadata, SharedMetrics, SharedRounds,
};
use crate::metrics::{mean, rounds_to_stable, MetricRow, RoundRow};
use crate::world::MarketWorld;

/// 世界初期化用 RNG ラベル (初期価格・初期状態)．
const RNG_WORLD_INIT: u64 = 0;
/// socsim エンジン用 RNG ラベル (本モデルでは実質未使用; 規約のため予約)．
const RNG_ENGINE: u64 = 1;

/// 解析ベンチマーク (LLM 非依存・決定論)．
pub struct Benchmarks {
    /// ベルトラン均衡価格 (firm_id 順)．
    pub p_bertrand: Vec<f64>,
    /// カルテル (独占) 価格 (firm_id 順)．
    pub p_cartel: Vec<f64>,
}

/// 設定から解析ベンチマークを計算する (`benchmark` サブコマンド・`init_world` 共用)．
pub fn compute_benchmarks(cfg: &Config) -> Benchmarks {
    let dp = cfg.demand_params();
    let costs = cfg.costs();
    Benchmarks {
        p_bertrand: bertrand_prices(&dp, &costs),
        p_cartel: cartel_prices(&dp, &costs),
    }
}

/// シミュレーション全体の実行結果．
pub struct SimulationResult {
    /// ラウンド行の履歴 (rounds.csv; long-format)．
    pub rounds: Vec<RoundRow>,
    /// メトリクス行の履歴 (metrics.csv)．
    pub metrics: Vec<MetricRow>,
    /// 解析ベンチマーク (benchmarks.json)．
    pub benchmarks: Benchmarks,
    /// LLM 呼び出しメタデータの集計．
    pub metadata: MetadataCollector,
    /// LLM モデル名．
    pub llm_model: String,
    /// LLM endpoint (primary)．
    pub llm_endpoint: String,
    /// 実行したラウンド数 (= 完了ステップ数)．
    pub final_round: usize,
}

impl SimulationResult {
    /// 最終ラウンドの全社平均価格．
    pub fn final_avg_price(&self) -> f64 {
        self.metrics.last().map(|m| m.avg_price).unwrap_or(0.0)
    }

    /// 最終ラウンドの collusion index．
    pub fn final_collusion_index(&self) -> f64 {
        self.metrics
            .last()
            .map(|m| m.collusion_index)
            .unwrap_or(0.0)
    }

    /// 平均価格系列の安定到達ラウンド (収束したラウンド index; なければ None)．
    pub fn rounds_to_stable(&self) -> Option<usize> {
        let avg: Vec<f64> = self.metrics.iter().map(|m| m.avg_price).collect();
        let span = (mean(&self.benchmarks.p_cartel) - mean(&self.benchmarks.p_bertrand)).abs();
        rounds_to_stable(&avg, 40, (0.05 * span).max(1e-9))
    }
}

/// 世界状態を初期化する (企業生成 + 解析ベンチマーク計算 + 初期価格)．
///
/// 初期価格は `init_price` (指定なければベルトランとカルテルの中点) を中央値に
/// ±10% で init RNG により散らす (初期価格依存性の検証に使える)．解析ベンチマークは
/// ここで 1 度だけ決定論的に計算し world に保持する (socsim 非依存・LLM 非依存)．
pub fn init_world(cfg: &Config, rng: &mut SimRng) -> MarketWorld {
    let dp = cfg.demand_params();
    let costs = cfg.costs();
    let bench = compute_benchmarks(cfg);

    let n = cfg.n_firms;
    let firms: Vec<AgentId> = (0..n as u64).map(AgentId).collect();

    // 初期価格の中央値 (指定なければ各社のベルトラン/カルテル中点)．並列配列
    // (bench.p_bertrand / bench.p_cartel / costs) を同一 index で引くため range で回す．
    let mut prices = Vec::with_capacity(n);
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        let center = cfg
            .init_price
            .unwrap_or_else(|| 0.5 * (bench.p_bertrand[i] + bench.p_cartel[i]));
        let jitter = rng.gen_range(0.9..=1.1);
        prices.push((center * jitter).max(costs[i]));
    }

    MarketWorld {
        clock: SimClock::new(cfg.max_rounds as u64),
        firms,
        prices,
        costs,
        quantities: vec![0.0; n],
        cum_profit: vec![0.0; n],
        memory: vec![Vec::new(); n],
        demand: dp,
        p_bertrand: bench.p_bertrand,
        p_cartel: bench.p_cartel,
        communication: cfg.communication,
        persona: cfg.persona,
        messages: vec![String::new(); n],
    }
}

/// 与えられた [`SabmClient`] でシミュレーションを実行する．
///
/// 本番は [`crate::llm::build_live_client`] の結果を，テストは [`crate::llm::wrap_client`] で
/// ラップした `mock::ScriptedClient` を渡す．Scheduler は `sequential` (同期更新の
/// 価格モデルは順序非依存なので決定論)．
pub fn run_with_client(
    cfg: &Config,
    client: SabmClient,
) -> std::result::Result<SimulationResult, String> {
    let root = cfg.seed.unwrap_or_else(rand::random);

    let mut init_rng = SimRng::from_seed(derive_seed(root, &[RNG_WORLD_INIT]));
    let world = init_world(cfg, &mut init_rng);
    let benchmarks = compute_benchmarks(cfg);

    let llm_model = client.inner().model().to_string();
    let llm_endpoint = client.inner().endpoint().to_string();

    let shared_client: SharedClient = Rc::new(RefCell::new(client));
    let shared_meta: SharedMetadata = Rc::new(RefCell::new(MetadataCollector::new()));
    let shared_rounds: SharedRounds = Rc::new(RefCell::new(Vec::new()));
    let shared_metrics: SharedMetrics = Rc::new(RefCell::new(Vec::new()));

    let mut sim = SimulationBuilder::new(world)
        .scheduler(Box::new(SequentialScheduler))
        .seed(derive_seed(root, &[RNG_ENGINE]))
        .add_mechanism(Box::new(MarketEnvironment))
        .add_mechanism(Box::new(CommunicationPhase::new(
            Rc::clone(&shared_client),
            Rc::clone(&shared_meta),
            cfg.llm.clone(),
        )))
        .add_mechanism(Box::new(PricingDecision::new(
            Rc::clone(&shared_client),
            Rc::clone(&shared_meta),
            cfg.llm.clone(),
            cfg.reflection_period,
        )))
        .add_mechanism(Box::new(MarketClearing))
        .add_mechanism(Box::new(ProfitReward::new(
            Rc::clone(&shared_rounds),
            Rc::clone(&shared_metrics),
        )))
        .add_mechanism(Box::new(MemoryUpdate {
            window: cfg.memory_window,
        }))
        .build();

    let mut final_round = 0usize;
    sim.run_observed(|report| {
        final_round = report.t as usize;
    })
    .map_err(|e| format!("シミュレーションの実行に失敗: {e}"))?;

    if cfg.llm.cache_path.is_some() {
        let client = shared_client.borrow();
        client
            .cache()
            .save()
            .map_err(|e| format!("キャッシュ保存に失敗: {e}"))?;
    }

    let rounds = shared_rounds.borrow().clone();
    let metrics = shared_metrics.borrow().clone();
    let metadata = shared_meta.borrow().clone();
    Ok(SimulationResult {
        rounds,
        metrics,
        benchmarks,
        metadata,
        llm_model,
        llm_endpoint,
        final_round,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::wrap_client;
    use socsim_llm::mock::ScriptedClient;
    use socsim_llm::PromptCache;

    fn scripted_at(price: f64) -> SabmClient {
        let reply = format!("{{\"price\": {price}}}");
        let backend = ScriptedClient::new("mock-llama3.2", move |_p: &str| reply.clone());
        wrap_client(backend, PromptCache::in_memory())
    }

    fn cfg() -> Config {
        Config {
            max_rounds: 30,
            seed: Some(42),
            ..Config::default()
        }
    }

    #[test]
    fn init_world_holds_basic_benchmarks() {
        let c = cfg();
        let mut rng = SimRng::from_seed(0);
        let w = init_world(&c, &mut rng);
        assert!((w.p_bertrand[0] - 6.0).abs() < 1e-9);
        assert!((w.p_cartel[0] - 8.0).abs() < 1e-9);
        assert_eq!(w.n_firms(), 2);
    }

    #[test]
    fn scripted_run_produces_trajectory() {
        let c = cfg();
        let r = run_with_client(&c, scripted_at(7.0)).unwrap();
        assert!(!r.rounds.is_empty());
        assert!(!r.metrics.is_empty());
        // すべて 7.0 を返す mock → CI は中点 0.5 付近で収束 → 早期 stop しうる．
        let last = r.metrics.last().unwrap();
        assert!(
            (last.avg_price - 7.0).abs() < 1e-6,
            "avg={}",
            last.avg_price
        );
        assert!(
            (last.collusion_index - 0.5).abs() < 1e-6,
            "CI={}",
            last.collusion_index
        );
    }

    #[test]
    fn core_is_deterministic_given_mock() {
        let c = cfg();
        let a = run_with_client(&c, scripted_at(7.0)).unwrap();
        let b = run_with_client(&c, scripted_at(7.0)).unwrap();
        let pa: Vec<f64> = a.rounds.iter().map(|r| r.price).collect();
        let pb: Vec<f64> = b.rounds.iter().map(|r| r.price).collect();
        assert_eq!(pa, pb, "同一シードは完全再現すべき");
    }
}
