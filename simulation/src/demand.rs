//! 線形需要関数と利益計算 (差別化財ベルトラン複占の市場クリアリング)．
//!
//! 本モジュールは決定論的な socsim コアの一部であり，LLM レイヤから独立している．
//! 論文 §4.3 の線形逆需要関数を解いた需要関数と利益式を実装する．
//!
//! # 需要・利益
//!
//! 逆需要 (差別化財複占):
//!
//! ```text
//! p_1 = a - β q_1 - d q_2,   p_2 = a - β q_2 - d q_1
//! ```
//!
//! これを `q` について解くと (`b = β² - d²`, `α = a β - a d`):
//!
//! ```text
//! q_i = (α - β p_i + d p_j) / b
//! ```
//!
//! 各社の利益:
//!
//! ```text
//! π_i = (p_i - c_i) q_i
//! ```
//!
//! N 社 (N >= 2) への一般化では「自社価格効果 β・各ライバル価格効果 d」を用いて
//! `q_i = (α - β p_i + d Σ_{j≠i} p_j) / b` とする (複占 N=2 で論文式に一致する)．

/// 線形需要関数のパラメータ (a, β, d, α, b)．
///
/// `alpha` / `b` は `a` / `beta` / `d` から導出される (論文 §4.3):
/// `α = a β - a d = a (β - d)`，`b = β² - d²`．
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct DemandParams {
    /// 逆需要の切片 a．
    pub a: f64,
    /// 自己価格効果 β．
    pub beta: f64,
    /// 交差価格効果 d (差別化度は d/β で制御; d/β=0 独占的, d/β=1 同質財)．
    pub d: f64,
    /// 導出パラメータ α = a (β - d)．
    pub alpha: f64,
    /// 導出パラメータ b = β² - d²．
    pub b: f64,
}

impl DemandParams {
    /// 一次パラメータ `a`, `beta`, `d` から需要パラメータを構築する (α, b を導出)．
    pub fn new(a: f64, beta: f64, d: f64) -> Self {
        DemandParams {
            a,
            beta,
            d,
            alpha: a * (beta - d),
            b: beta * beta - d * d,
        }
    }

    /// 差別化度 d/β (0 = 独占的に分離, 1 = 同質財)．
    pub fn d_over_beta(&self) -> f64 {
        if self.beta.abs() < 1e-12 {
            0.0
        } else {
            self.d / self.beta
        }
    }
}

/// 価格ベクトル `prices` から企業 `i` の需要量を計算する．
///
/// `q_i = (α - β p_i + d Σ_{j≠i} p_j) / b`．需要は非負へクランプする (負の需要は
/// 0 = 退出した市場とみなす)．
pub fn quantity(params: &DemandParams, prices: &[f64], i: usize) -> f64 {
    if params.b.abs() < 1e-12 {
        return 0.0;
    }
    let rivals_sum: f64 = prices
        .iter()
        .enumerate()
        .filter(|(j, _)| *j != i)
        .map(|(_, p)| *p)
        .sum();
    let q = (params.alpha - params.beta * prices[i] + params.d * rivals_sum) / params.b;
    q.max(0.0)
}

/// 全企業の需要量ベクトルを計算する．
pub fn quantities(params: &DemandParams, prices: &[f64]) -> Vec<f64> {
    (0..prices.len())
        .map(|i| quantity(params, prices, i))
        .collect()
}

/// 企業 `i` の利益 `π_i = (p_i - c_i) q_i`．
pub fn profit(price: f64, cost: f64, quantity: f64) -> f64 {
    (price - cost) * quantity
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 論文基本設定: a=14, β=1/150, d=1/300 → α=7/150, b=1/30000，
    /// 需要 q_1 = 1400 - 200 p_1 + 100 p_2．
    fn basic() -> DemandParams {
        DemandParams::new(14.0, 1.0 / 150.0, 1.0 / 300.0)
    }

    #[test]
    fn derived_params_match_paper_basic() {
        let p = basic();
        assert!((p.alpha - 7.0 / 150.0).abs() < 1e-12, "alpha={}", p.alpha);
        assert!((p.b - 1.0 / 30000.0).abs() < 1e-15, "b={}", p.b);
        assert!((p.d_over_beta() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn demand_matches_paper_coefficients() {
        // q_1 = 1400 - 200 p_1 + 100 p_2 で p=(6,6) → 1400-1200+600 = 800．
        let p = basic();
        let q = quantity(&p, &[6.0, 6.0], 0);
        assert!((q - 800.0).abs() < 1e-6, "q={q}");
        // 係数チェック: ∂q/∂p_1 = -β/b = -200, ∂q/∂p_2 = d/b = 100, 切片 α/b=1400．
        let q0 = quantity(&p, &[0.0, 0.0], 0);
        assert!((q0 - 1400.0).abs() < 1e-6, "intercept={q0}");
        let dq_own = quantity(&p, &[1.0, 0.0], 0) - q0;
        assert!((dq_own + 200.0).abs() < 1e-6, "dq_own={dq_own}");
        let dq_rival = quantity(&p, &[0.0, 1.0], 0) - q0;
        assert!((dq_rival - 100.0).abs() < 1e-6, "dq_rival={dq_rival}");
    }

    #[test]
    fn negative_demand_clamped_to_zero() {
        let p = basic();
        // 極端に高い自社価格 → 需要が負になるはずだが 0 にクランプ．
        let q = quantity(&p, &[100.0, 6.0], 0);
        assert_eq!(q, 0.0);
    }

    #[test]
    fn profit_formula() {
        assert!((profit(6.0, 2.0, 800.0) - 3200.0).abs() < 1e-9);
    }
}
