//! socsim フレームワーク上の SABM 市場競争メカニズム (5 Mechanism × 6 phase)．
//!
//! 二層アーキテクチャの **境界** がここにある．下層 (決定論的 socsim コア) は
//! 市場クリアリング・利益計算・収束判定・メモリ更新を担い，上層 (非決定的 LLM
//! レイヤ) は [`SabmClient`] (キャッシュ付き Ollama→OpenAI フォールバック) 越しの
//! 価格決定 (`PricingDecision`) のみを担う．
//!
//! 論文の 1 ラウンド (= socsim の 1 tick) を 5 Mechanism へ割り当てる:
//!
//! | Mechanism | Phase | 役割 |
//! |-----------|-------|------|
//! | [`MarketEnvironment`] | Environment | ラウンド冒頭の市場確認・バッファリセット |
//! | [`PricingDecision`]   | Decision    | **★LLM 所在**．各社の新価格を履歴・ペルソナ・基準価格から決定 (同期一括代入)．reflection 周期で反省を挟む |
//! | [`MarketClearing`]    | Interaction | 線形需要関数で需要量を計算 |
//! | [`ProfitReward`]      | Reward      | 利益・collusion index を計算し記録．収束/有界振動で request_stop |
//! | [`MemoryUpdate`]      | PostStep    | 当ラウンド記録をメモリへ追記し直近 window 件に切り詰める |
//!
//! LLM 呼び出しは Decision フェーズの 1 mechanism に閉じ込める．LLM クライアントと
//! 呼び出しメタデータは `Rc<RefCell<…>>` で共有し，run ドライバが実行後にキャッシュ
//! 保存・メタデータ集計に使う．価格軌跡・メトリクスも共有バッファ経由でドライバへ渡す．

use std::cell::RefCell;
use std::rc::Rc;

use socsim_core::{Mechanism, Phase, Result, SocsimError, StepContext};
use socsim_llm::MetadataCollector;

use crate::analytic::collusion_index;
use crate::config::LlmSettings;
use crate::demand;
use crate::llm::{llm_config, SabmClient};
use crate::metrics::{is_stationary, mean, MetricRow, RoundRow};
use crate::prompts::{parse_price, pricing_prompt, PricingBriefing};
use crate::world::{MarketWorld, RoundRecord};

/// 共有 LLM クライアント (run ドライバとメカニズムで共有)．
pub type SharedClient = Rc<RefCell<SabmClient>>;
/// 共有メタデータコレクタ (cache-hit 率などを run 後に集計)．
pub type SharedMetadata = Rc<RefCell<MetadataCollector>>;
/// 共有 ラウンド行バッファ (rounds.csv; long-format)．
pub type SharedRounds = Rc<RefCell<Vec<RoundRow>>>;
/// 共有 メトリクス行バッファ (metrics.csv)．
pub type SharedMetrics = Rc<RefCell<Vec<MetricRow>>>;

// =========================================================================== //
// 1. MarketEnvironment (Environment)
// =========================================================================== //

/// ラウンド冒頭の市場確認・需要量バッファのリセット (`Environment`; LLM 非依存)．
///
/// 本モデルの市場環境 (需要パラメータ) はラウンド間で一定だが，6-phase ループの
/// 規約に沿って Environment フェーズを明示的に持つ．当ラウンドの需要量を 0 に
/// リセットし，`MarketClearing` が上書きする．
pub struct MarketEnvironment;

impl Mechanism<MarketWorld> for MarketEnvironment {
    fn name(&self) -> &str {
        "market_environment"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Environment]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, MarketWorld>) -> Result<()> {
        for q in ctx.world.quantities.iter_mut() {
            *q = 0.0;
        }
        Ok(())
    }
}

// =========================================================================== //
// 2. PricingDecision (Decision, LLM)
// =========================================================================== //

/// 各企業が履歴メモリ・ペルソナ・基準価格から次ラウンド価格を決定する
/// (`Decision`; LLM 所在)．
///
/// **同期更新セマンティクス**: 当ラウンド冒頭の (旧) 市場状態・履歴から全社の新価格を
/// 計算し，一括代入する (firms simultaneously set prices)．`ctx.agent_order` には
/// 依存しないため Scheduler は `sequential` で十分 (決定論)．`reflection_period`
/// ごとに reflection (planning) をプロンプトへ挟む．
pub struct PricingDecision {
    client: SharedClient,
    metadata: SharedMetadata,
    settings: LlmSettings,
    reflection_period: usize,
}

impl PricingDecision {
    pub fn new(
        client: SharedClient,
        metadata: SharedMetadata,
        settings: LlmSettings,
        reflection_period: usize,
    ) -> Self {
        PricingDecision {
            client,
            metadata,
            settings,
            reflection_period,
        }
    }
}

impl Mechanism<MarketWorld> for PricingDecision {
    fn name(&self) -> &str {
        "pricing_decision"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Decision]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, MarketWorld>) -> Result<()> {
        let round = ctx.world.current_round();
        let n = ctx.world.n_firms();
        let reflect = self.reflection_period > 0
            && round > 0
            && (round as usize).is_multiple_of(self.reflection_period);

        // 全社の新価格を «旧状態» から計算してから一括代入する (同期更新)．
        // world の並列配列 (costs / prices / memory / p_bertrand / p_cartel) を同一
        // index で引くため range で回す．
        let mut new_prices = ctx.world.prices.clone();
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let cost = ctx.world.costs[i];
            let fallback_price = ctx.world.prices[i];
            let brief = PricingBriefing {
                firm_id: i,
                round,
                cost,
                persona: ctx.world.persona,
                communication: ctx.world.communication,
                memory: &ctx.world.memory[i],
                reflect,
                p_bertrand: ctx.world.p_bertrand[i],
                p_cartel: ctx.world.p_cartel[i],
            };
            let prompt = pricing_prompt(&brief);
            let text = {
                let mut client = self.client.borrow_mut();
                let resp = client
                    .complete(&prompt, &llm_config(&self.settings))
                    .map_err(|e| SocsimError::Mechanism(format!("pricing LLM call failed: {e}")))?;
                self.metadata.borrow_mut().record(resp.metadata.clone());
                resp.text
            };
            // パース失敗時は直近価格を維持する．
            new_prices[i] = parse_price(&text, cost).unwrap_or(fallback_price);
        }
        ctx.world.prices = new_prices;
        Ok(())
    }
}

// =========================================================================== //
// 3. MarketClearing (Interaction)
// =========================================================================== //

/// 市場クリアリング: 線形需要関数で各社の需要量を計算する (`Interaction`; LLM 非依存)．
///
/// `q_i = (α - β p_i + d Σ_{j≠i} p_j) / b` (negative はクランプ)．価格差で需要が
/// 配分される (固定位相ではなく市場媒介の相互作用)．
pub struct MarketClearing;

impl Mechanism<MarketWorld> for MarketClearing {
    fn name(&self) -> &str {
        "market_clearing"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Interaction]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, MarketWorld>) -> Result<()> {
        let q = demand::quantities(&ctx.world.demand, &ctx.world.prices);
        ctx.world.quantities = q;
        Ok(())
    }
}

// =========================================================================== //
// 4. ProfitReward (Reward)
// =========================================================================== //

/// 利益計算・collusion index 記録・収束/有界振動による停止判定 (`Reward`; LLM 非依存)．
///
/// `π_i = (p_i - c_i) q_i` を累積へ加算し，long-format の `rounds.csv` 行と
/// ラウンド集計の `metrics.csv` 行 (avg_price / collusion_index / total_profit) を
/// 共有バッファへ push する．平均価格系列が定常 (収束 or 有界振動) になったら
/// `request_stop()` を発火する．
pub struct ProfitReward {
    rounds: SharedRounds,
    metrics: SharedMetrics,
    /// 平均価格系列 (停止判定に使う; ドライバとは独立に保持)．
    avg_price_history: Vec<f64>,
}

impl ProfitReward {
    pub fn new(rounds: SharedRounds, metrics: SharedMetrics) -> Self {
        ProfitReward {
            rounds,
            metrics,
            avg_price_history: Vec::new(),
        }
    }
}

impl Mechanism<MarketWorld> for ProfitReward {
    fn name(&self) -> &str {
        "profit_reward"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Reward]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, MarketWorld>) -> Result<()> {
        let round = ctx.world.current_round();
        let n = ctx.world.n_firms();

        let mut total_profit = 0.0;
        let mut ci_values = Vec::with_capacity(n);
        for i in 0..n {
            let price = ctx.world.prices[i];
            let quantity = ctx.world.quantities[i];
            let cost = ctx.world.costs[i];
            let profit = demand::profit(price, cost, quantity);
            ctx.world.cum_profit[i] += profit;
            total_profit += profit;

            ci_values.push(collusion_index(
                price,
                ctx.world.p_bertrand[i],
                ctx.world.p_cartel[i],
            ));

            self.rounds.borrow_mut().push(RoundRow {
                round,
                firm_id: ctx.world.firms[i].0,
                price,
                quantity,
                profit,
            });
        }

        let avg_price = ctx.world.avg_price();
        let avg_ci = mean(&ci_values);
        self.metrics.borrow_mut().push(MetricRow {
            round,
            avg_price,
            collusion_index: avg_ci,
            total_profit,
        });

        // ドライバ観測用に scratch へも置く．
        ctx.scratch.insert("avg_price", avg_price);
        ctx.scratch.insert("collusion_index", avg_ci);

        // --- 収束/有界振動による停止判定 ---
        self.avg_price_history.push(avg_price);
        // span = p^M - p^B (全社平均で代表)．
        let span = {
            let m = mean(&ctx.world.p_cartel);
            let b = mean(&ctx.world.p_bertrand);
            (m - b).abs()
        };
        let eps = 0.05 * span;
        // 論文は 400/800 ラウンドだが，実用上は max_rounds が小さい再現実行でも
        // 機能するよう短い窓を使う (収束 40 / 有界振動 60)．
        if is_stationary(
            &self.avg_price_history,
            40,
            eps.max(1e-9),
            60,
            span.max(1e-9),
        ) {
            ctx.request_stop();
        }
        Ok(())
    }
}

// =========================================================================== //
// 5. MemoryUpdate (PostStep)
// =========================================================================== //

/// 各社の境界付きメモリに当ラウンドの記録を追記し，直近 `window` 件に切り詰める
/// (`PostStep`; LLM 非依存)．論文の «毎ラウンド履歴を再注入» を支える．
pub struct MemoryUpdate {
    /// 記憶窓 (直近何ラウンド分; 論文は 20)．
    pub window: usize,
}

impl Mechanism<MarketWorld> for MemoryUpdate {
    fn name(&self) -> &str {
        "memory_update"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::PostStep]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, MarketWorld>) -> Result<()> {
        let n = ctx.world.n_firms();
        for i in 0..n {
            let rec = RoundRecord {
                own_price: ctx.world.prices[i],
                own_quantity: ctx.world.quantities[i],
                own_profit: demand::profit(
                    ctx.world.prices[i],
                    ctx.world.costs[i],
                    ctx.world.quantities[i],
                ),
                rival_prices: ctx.world.rival_prices(i),
            };
            let mem = &mut ctx.world.memory[i];
            mem.push(rec);
            if self.window > 0 && mem.len() > self.window {
                let excess = mem.len() - self.window;
                mem.drain(0..excess);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use socsim_core::{AgentId, SimClock, WorldState};

    use crate::world::{DemandParams, Persona};

    fn world() -> MarketWorld {
        MarketWorld {
            clock: SimClock::new(10),
            firms: vec![AgentId(0), AgentId(1)],
            prices: vec![6.0, 6.0],
            costs: vec![2.0, 2.0],
            quantities: vec![0.0, 0.0],
            cum_profit: vec![0.0, 0.0],
            memory: vec![Vec::new(), Vec::new()],
            demand: DemandParams::new(14.0, 1.0 / 150.0, 1.0 / 300.0),
            p_bertrand: vec![6.0, 6.0],
            p_cartel: vec![8.0, 8.0],
            communication: false,
            persona: Persona::Active,
        }
    }

    #[test]
    fn memory_update_bounds_to_window() {
        let mut mu = MemoryUpdate { window: 3 };
        let mut w = world();
        let mut rng = socsim_core::SimRng::from_seed(0);
        let mut bb = socsim_core::Blackboard::new();
        let mut stop = false;
        let mut rec = socsim_core::NullRecorder;
        let order = w.agent_ids();
        for _ in 0..5 {
            let clock = *w.clock();
            let mut ctx = StepContext {
                world: &mut w,
                clock,
                rng: &mut rng,
                recorder: &mut rec,
                agent_order: &order,
                scratch: &mut bb,
                stop: &mut stop,
            };
            mu.apply(Phase::PostStep, &mut ctx).unwrap();
        }
        assert_eq!(w.memory[0].len(), 3, "bounded to window");
        assert_eq!(w.memory[1].len(), 3);
    }
}
