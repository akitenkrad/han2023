//! 評価指標と収束/有界振動判定 (論文 §4.3 / §5)．
//!
//! 価格軌跡から平均価格・collusion index・利益・収束判定を計算する．`rounds.csv`
//! (long-format: round, firm_id, price, quantity, profit) と `metrics.csv`
//! (round, avg_price, collusion_index, total_profit) の行型を提供する．

use serde::Serialize;

/// `rounds.csv` の 1 行 (long-format: 1 行 = 1 ラウンド 1 企業)．
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RoundRow {
    /// ラウンド (0 始まり)．
    pub round: u64,
    /// 企業 `AgentId` の生 `u64` (= firm_id)．
    pub firm_id: u64,
    /// このラウンドの価格 p_i．
    pub price: f64,
    /// このラウンドの需要量 q_i．
    pub quantity: f64,
    /// このラウンドの利益 π_i．
    pub profit: f64,
}

/// `metrics.csv` の 1 行 (1 行 = 1 ラウンド)．
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct MetricRow {
    /// ラウンド (0 始まり)．
    pub round: u64,
    /// 全社平均価格．
    pub avg_price: f64,
    /// collusion index (全社平均の CI; 0 = ベルトラン, 1 = カルテル)．
    pub collusion_index: f64,
    /// 全社のラウンド利益合計．
    pub total_profit: f64,
}

/// 平均値 (空なら 0)．
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// 分散 (標本分散; 空/単一なら 0)．収束後の「滑らかさ」を測る．
pub fn variance(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(values);
    values.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n as f64
}

/// 全社平均価格系列が «安定共謀» に達したラウンド数を推定する．
///
/// 論文の停止基準を実用的に近似する: 価格系列を末尾から見て，直近 `window`
/// ラウンドの sup - inf が `tol`(= ε) 以下になった最初のラウンド index を返す．
/// 達しなければ `None`．`tol` は (p^M - p^B) のスケールで与える (例 0.05*(p^M-p^B))．
pub fn rounds_to_stable(avg_prices: &[f64], window: usize, tol: f64) -> Option<usize> {
    if avg_prices.len() < window || window == 0 {
        return None;
    }
    for end in window..=avg_prices.len() {
        let slice = &avg_prices[end - window..end];
        let (lo, hi) = min_max(slice);
        if hi - lo <= tol {
            return Some(end - 1); // 安定が確認できたラウンド (0 始まり)．
        }
    }
    None
}

/// 直近 `window` ラウンドが «収束» か «有界振動» かを判定する (stop 用)．
///
/// - 収束: 直近 `window` の sup - inf <= `eps` (= 0.05 (p^M - p^B))．
/// - 有界振動: 直近 `osc_window` の sup - inf <= `span` (= p^M - p^B)．
///
/// いずれかが満たされたとき定常状態とみなし true を返す．データ不足なら false．
pub fn is_stationary(
    avg_prices: &[f64],
    window: usize,
    eps: f64,
    osc_window: usize,
    span: f64,
) -> bool {
    let converged = if avg_prices.len() >= window && window > 0 {
        let slice = &avg_prices[avg_prices.len() - window..];
        let (lo, hi) = min_max(slice);
        hi - lo <= eps
    } else {
        false
    };
    let bounded_osc = if avg_prices.len() >= osc_window && osc_window > 0 {
        let slice = &avg_prices[avg_prices.len() - osc_window..];
        let (lo, hi) = min_max(slice);
        hi - lo <= span
    } else {
        false
    };
    converged || bounded_osc
}

/// スライスの (min, max)．空なら (0, 0)．
fn min_max(values: &[f64]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in values {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    if values.is_empty() {
        (0.0, 0.0)
    } else {
        (lo, hi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_and_variance() {
        assert!((mean(&[6.0, 8.0]) - 7.0).abs() < 1e-12);
        assert!(variance(&[7.0, 7.0, 7.0]).abs() < 1e-12);
        assert!(variance(&[6.0, 8.0]) > 0.0);
    }

    #[test]
    fn rounds_to_stable_detects_flat_tail() {
        // 最初は振れ，途中から 7.0 で安定する系列．
        let mut s = vec![6.0, 6.5, 7.3, 6.8];
        s.extend(std::iter::repeat_n(7.0, 5));
        let r = rounds_to_stable(&s, 5, 0.01);
        assert!(r.is_some(), "should detect stability");
    }

    #[test]
    fn rounds_to_stable_none_when_volatile() {
        let s = vec![6.0, 8.0, 6.0, 8.0, 6.0, 8.0];
        assert_eq!(rounds_to_stable(&s, 3, 0.01), None);
    }

    #[test]
    fn stationary_via_convergence() {
        let s = vec![7.0; 10];
        assert!(is_stationary(&s, 5, 0.1, 8, 2.0));
    }

    #[test]
    fn stationary_via_bounded_oscillation() {
        // sup-inf = 1.0 <= span(2.0) だが > eps(0.1) → 有界振動で stationary．
        let s = vec![6.5, 7.5, 6.5, 7.5, 6.5, 7.5, 6.5, 7.5];
        assert!(is_stationary(&s, 4, 0.1, 8, 2.0));
    }

    #[test]
    fn not_stationary_when_wide() {
        let s = vec![6.0, 9.0, 6.0, 9.0];
        assert!(!is_stationary(&s, 4, 0.1, 4, 2.0));
    }
}
