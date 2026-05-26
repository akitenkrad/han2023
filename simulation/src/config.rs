//! シミュレーション設定．
//!
//! Han, Wu & Xiao (2023) SABM のコアモデル (LLM 駆動の差別化財ベルトラン複占
//! 繰り返し価格ゲーム) と感度分析パラメータを保持する [`Config`] と，その JSON
//! シリアライズ表現を定義する．企業数・需要パラメータ (a, β, d → α, b 導出)・
//! 非対称コスト・ペルソナ・会話の有無・最大ラウンド・LLM 設定をここに集約する．

use serde::Serialize;

use crate::demand::DemandParams;
use crate::world::Persona;

// --------------------------------------------------------------------------- //
// LLM 設定
// --------------------------------------------------------------------------- //

/// LLM レイヤの設定 (provider / model / temperature / seed / cache)．
///
/// 定義は `socsim-llm` に集約済み (各 replication で同一だった struct を統合)．
/// `crate::config::LlmSettings` パスは re-export で温存する．
pub use socsim_llm::LlmSettings;

// --------------------------------------------------------------------------- //
// Config
// --------------------------------------------------------------------------- //

/// 単一実行の設定．
///
/// 既定値は論文基本設定 (2 社複占, a=14, β=1/150, d=1/300, c=2, d/β=0.5 →
/// p^B=6, p^M=8) に対応する．
#[derive(Debug, Clone)]
pub struct Config {
    /// 企業数 n (基本は 2 = 複占)．
    pub n_firms: usize,
    /// 逆需要切片 a．
    pub a: f64,
    /// 自己価格効果 β．
    pub beta: f64,
    /// 交差価格効果 d (差別化度は d/β)．
    pub d: f64,
    /// 限界費用 c_1．
    pub c1: f64,
    /// 限界費用 c_2 (非対称コスト対応; firm 0,1 に対応)．
    pub c2: f64,
    /// 企業ペルソナ．
    pub persona: Persona,
    /// 会話の有無 (基本 = false)．
    pub communication: bool,
    /// 最大ラウンド数 (= 強制終了ラウンド; 論文は最大 2000)．
    pub max_rounds: usize,
    /// 反省 (reflection / planning) を挟む周期 (ラウンド; 論文は 20)．
    pub reflection_period: usize,
    /// 境界付きメモリの窓 (直近何ラウンドを再注入するか; 論文は 20)．
    pub memory_window: usize,
    /// 初期価格の中央値 (init RNG で散らす中心; 既定はベルトランとカルテルの中点)．
    pub init_price: Option<f64>,

    /// 乱数シード (None の場合はランダム; socsim コア層のみ支配)．
    pub seed: Option<u64>,
    /// LLM レイヤ設定．
    pub llm: LlmSettings,
    /// 結果出力ディレクトリ．
    pub output_dir: String,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            n_firms: 2,
            a: 14.0,
            beta: 1.0 / 150.0,
            d: 1.0 / 300.0,
            c1: 2.0,
            c2: 2.0,
            persona: Persona::Active,
            communication: false,
            max_rounds: 1000,
            reflection_period: 20,
            memory_window: 20,
            init_price: None,
            seed: Some(42),
            llm: LlmSettings::default(),
            output_dir: "results".to_string(),
        }
    }
}

impl Config {
    /// 需要パラメータ (α, b を導出) を組み立てる．
    pub fn demand_params(&self) -> DemandParams {
        DemandParams::new(self.a, self.beta, self.d)
    }

    /// 各社の限界費用ベクトル (firm_id 順)．
    ///
    /// c1/c2 で 2 社のコストを指定する．3 社以上では firm 0 = c1, それ以外 = c2．
    pub fn costs(&self) -> Vec<f64> {
        (0..self.n_firms)
            .map(|i| if i == 0 { self.c1 } else { self.c2 })
            .collect()
    }
}

/// `run` の試行シードを派生する (試行 index で独立化する)．
pub fn derive_run_seed(base: u64, run_idx: usize) -> u64 {
    socsim_core::derive_seed(base, &[run_idx as u64])
}

/// `config.json` (run 用) のシリアライズ表現．
#[derive(Serialize)]
pub struct RunConfigJson {
    pub command: &'static str,
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
    pub seed: Option<u64>,
    pub llm_temperature: f32,
    pub llm_seed: u64,
    pub output_dir: String,
}

impl Config {
    /// `config.json` 用の表現を組み立てる．
    pub fn to_run_config_json(&self) -> RunConfigJson {
        let dp = self.demand_params();
        RunConfigJson {
            command: "run",
            n_firms: self.n_firms,
            a: self.a,
            beta: self.beta,
            d: self.d,
            d_over_beta: dp.d_over_beta(),
            alpha: dp.alpha,
            b: dp.b,
            c1: self.c1,
            c2: self.c2,
            persona: self.persona.label().to_string(),
            communication: self.communication,
            max_rounds: self.max_rounds,
            reflection_period: self.reflection_period,
            memory_window: self.memory_window,
            seed: self.seed,
            llm_temperature: self.llm.temperature,
            llm_seed: self.llm.seed,
            output_dir: self.output_dir.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_paper_basic_setup() {
        let cfg = Config::default();
        let dp = cfg.demand_params();
        assert!((dp.d_over_beta() - 0.5).abs() < 1e-12);
        assert_eq!(cfg.costs(), vec![2.0, 2.0]);
    }
}
