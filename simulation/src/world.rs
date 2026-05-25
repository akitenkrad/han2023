//! socsim フレームワーク上の SABM 市場世界状態 (`MarketWorld`)．
//!
//! エージェント = 移動する空間主体ではなく，市場 (差別化財ベルトラン複占の需要
//! 分割) を介して相互作用する少数の企業 (基本は 2 社の複占) である．したがって
//! `socsim-grid` / `socsim-net` は不使用で，全企業が共有する 1 つの市場環境
//! (需要関数パラメータ) と各社の状態 (価格・コスト・累積利益・履歴メモリ) を
//! `MarketWorld` 1 つに集約する．「相互作用」は固定位相ではなく市場クリアリング
//! (`MarketClearing`) を通じて間接的に生じる．
//!
//! `agent_ids()` は企業 ID 昇順 (ソート済み) を返し決定論を担保する (socsim コア層)．
//! `#[derive(Clone, Serialize, Deserialize)]` でスナップショット (save/resume) と
//! 感度分析の比較実験に対応する．

use serde::{Deserialize, Serialize};
use socsim_core::{AgentId, SimClock, WorldState};

pub use crate::demand::DemandParams;

/// 企業ペルソナ (論文付録 Group 5-7; 探索性向と攻撃性を制御する)．
///
/// LLM への指示文 (プロンプトの Persona 節) を切り替える．論文では行動を大きく
/// 左右し，`None` は探索に乏しく超低速，`Aggressive` は共謀と価格戦争の周期振動，
/// `Active` は積極的に共謀を探索して (ベルトラン, カルテル) 区間へ収束する．
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Persona {
    /// 積極的に利益機会を探索する (論文の基本ケース)．
    Active,
    /// 攻撃的・対立的 (価格戦争を辞さない)．
    Aggressive,
    /// ペルソナ指示なし (探索に乏しい)．
    None,
}

impl Persona {
    /// CLI / JSON ラベル．
    pub fn label(&self) -> &'static str {
        match self {
            Persona::Active => "active",
            Persona::Aggressive => "aggressive",
            Persona::None => "none",
        }
    }

    /// プロンプトの Persona 節に挿入する指示文 (`None` は空)．
    pub fn directive(&self) -> &'static str {
        match self {
            Persona::Active => {
                "You are an active, profit-seeking manager. You proactively look for \
                 opportunities to raise margins, but you remain pragmatic and responsive \
                 to your rival's moves."
            }
            Persona::Aggressive => {
                "You are an aggressive, combative manager. You are willing to start a price \
                 war to win market share and you do not shy away from undercutting your rival."
            }
            Persona::None => "",
        }
    }
}

/// 文字列から [`Persona`] をパースする．
pub fn parse_persona(s: &str) -> Result<Persona, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "active" => Ok(Persona::Active),
        "aggressive" | "aggro" => Ok(Persona::Aggressive),
        "none" | "neutral" | "off" => Ok(Persona::None),
        _ => Err(format!(
            "不正なペルソナ: \"{}\" (active / aggressive / none)",
            s
        )),
    }
}

/// 1 ラウンドの企業ごとの観測 (境界付きメモリの 1 要素)．
///
/// 論文では GPT がラウンド間で記憶を持たないため，この履歴を毎ラウンド再注入する
/// (直近 20 件)．
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RoundRecord {
    /// このラウンドの自社価格 p_i．
    pub own_price: f64,
    /// このラウンドの自社需要量 q_i．
    pub own_quantity: f64,
    /// このラウンドの自社利益 π_i．
    pub own_profit: f64,
    /// このラウンドの相手社の価格 (firm_id 順)．
    pub rival_prices: Vec<f64>,
}

/// SABM 市場競争シミュレーションの世界状態．
///
/// 全フィールドは firm_id (= `Vec` の index) で添字付けされる．`firms` はソート済み
/// `AgentId` (`0..n_firms`) を保持し，`agent_ids()` がそのまま決定論順を返す．
#[derive(Clone, Serialize, Deserialize)]
pub struct MarketWorld {
    /// シミュレーションクロック (1 tick = 1 ラウンド)．
    pub clock: SimClock,
    /// 企業 `AgentId` (`0..n_firms`，ソート済み)．
    pub firms: Vec<AgentId>,
    /// 現ラウンドの価格 p_i (index = firm_id)．
    pub prices: Vec<f64>,
    /// 限界費用 c_i (非対称コスト対応)．
    pub costs: Vec<f64>,
    /// 現ラウンドの需要量 q_i (`MarketClearing` が書き込む)．
    pub quantities: Vec<f64>,
    /// 累積利益．
    pub cum_profit: Vec<f64>,
    /// 各社の直近 20 ラウンド履歴 (境界付きメモリ)．
    pub memory: Vec<Vec<RoundRecord>>,
    /// 市場環境 (需要関数パラメータ)．
    pub demand: DemandParams,
    /// 解析ベンチマーク: ベルトラン均衡価格 (firm_id 順)．
    pub p_bertrand: Vec<f64>,
    /// 解析ベンチマーク: カルテル (独占) 価格 (firm_id 順)．
    pub p_cartel: Vec<f64>,
    /// 会話の有無 (基本モデル = false; 会話あり代替モデルは Phase 3)．
    pub communication: bool,
    /// 企業ペルソナ (active / aggressive / none)．
    pub persona: Persona,
}

impl MarketWorld {
    /// 企業数 n．
    pub fn n_firms(&self) -> usize {
        self.firms.len()
    }

    /// 現在のラウンド (0 始まり)．
    ///
    /// socsim エンジンはステップ先頭で `tick()` するため，クロックは `1..=max_rounds`
    /// を走る．本モデルはラウンドを 0 始まりで扱うので `t() - 1` を返す．
    pub fn current_round(&self) -> u64 {
        self.clock.t().saturating_sub(1)
    }

    /// 全社平均価格．
    pub fn avg_price(&self) -> f64 {
        if self.prices.is_empty() {
            0.0
        } else {
            self.prices.iter().sum::<f64>() / self.prices.len() as f64
        }
    }

    /// 企業 `i` を除く相手社の現価格ベクトル (firm_id 順)．
    pub fn rival_prices(&self, i: usize) -> Vec<f64> {
        self.prices
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, p)| *p)
            .collect()
    }
}

impl WorldState for MarketWorld {
    fn agent_ids(&self) -> Vec<AgentId> {
        self.firms.clone() // 0..n_firms 昇順 (ソート済み) → 決定論．
    }

    fn clock(&self) -> &SimClock {
        &self.clock
    }

    fn clock_mut(&mut self) -> &mut SimClock {
        &mut self.clock
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persona_parse_roundtrip() {
        assert_eq!(parse_persona("active").unwrap(), Persona::Active);
        assert_eq!(parse_persona("Aggressive").unwrap(), Persona::Aggressive);
        assert_eq!(parse_persona("none").unwrap(), Persona::None);
        assert!(parse_persona("bogus").is_err());
        assert_eq!(Persona::Active.label(), "active");
    }

    #[test]
    fn agent_ids_sorted() {
        let w = MarketWorld {
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
        };
        assert_eq!(w.agent_ids(), vec![AgentId(0), AgentId(1)]);
        assert_eq!(w.rival_prices(0), vec![6.0]);
        assert!((w.avg_price() - 6.0).abs() < 1e-12);
    }
}
