//! 解析ベンチマーク: ベルトラン均衡価格と カルテル (独占) 価格．
//!
//! 本モジュールは **完全に決定論的** で socsim にも LLM にも依存しない純数値計算
//! である．`init_world` で 1 度だけ計算し，シミュレーション価格を対比する 2 基準
//! (collusion index の下端・上端) として world に保持する．
//!
//! # ベルトラン均衡価格 (論文 §4.3)
//!
//! 差別化財ベルトラン複占の非協調ナッシュ均衡:
//!
//! ```text
//! p_1^B = (d α + β d c_2 + 2 β α + 2 β² c_1) / (4 β² - d²)
//! p_2^B = (d α + β d c_1 + 2 β α + 2 β² c_2) / (4 β² - d²)
//! ```
//!
//! # カルテル (独占) 価格 (論文 §4.3, d/β ≠ 1)
//!
//! 結合利益を最大化する協調価格:
//!
//! ```text
//! p_i^M = α / (2 (β - d)) + c_i / 2
//! ```
//!
//! 基本設定 (a=14, β=1/150, d=1/300, α=7/150, b=1/30000, c_1=c_2=2, d/β=0.5) で
//! `p_i^B = 6`, `p_i^M = 8` (厳密)．本モジュールの単体テストで検算する．
//!
//! # N 社への一般化
//!
//! 複占公式は 2 社専用なので，N >= 3 では対称ベルトラン均衡の一般形を使う:
//! 自社の最適反応 `p_i = (α + β c_i + d Σ_{j≠i} p_j) / (2 β)` の対称固定点
//! `p^B = (α + β c) / (2 β - (N-1) d)`，カルテルは
//! `p^M = α / ((N-1)(... ))` …とはせず，複占と同じ「結合限界収入 = 限界費用」条件
//! `p_i^M = α / (2 (β - (N-1) d)) + c_i / 2` を用いる (N=2 で複占式に一致)．

use crate::demand::DemandParams;

/// 2 社のベルトラン均衡価格 (論文 §4.3 の複占公式)．
///
/// `costs` は限界費用 `[c_1, c_2]`．非対称コストにも対応する．
fn bertrand_duopoly(params: &DemandParams, c1: f64, c2: f64) -> (f64, f64) {
    let DemandParams { beta, d, alpha, .. } = *params;
    let denom = 4.0 * beta * beta - d * d;
    let p1 = (d * alpha + beta * d * c2 + 2.0 * beta * alpha + 2.0 * beta * beta * c1) / denom;
    let p2 = (d * alpha + beta * d * c1 + 2.0 * beta * alpha + 2.0 * beta * beta * c2) / denom;
    (p1, p2)
}

/// N 社 (N >= 3) の対称ベルトラン均衡価格 (一般化)．
///
/// 自社最適反応 `p_i = (α + β c_i + d Σ_{j≠i} p_j) / (2 β)` の固定点として，
/// `p_i^B = (α + β c_i + d (S - p_i)) / (2 β)` を解く．ここで近似的に各社価格を
/// 同水準とみなし `p_i^B = (α + β c_i) / (2 β - (n-1) d / n ...)` …ではなく，対称
/// コストを仮定した閉形 `p^B = (α + β c̄) / (2 β - (n - 1) d)` を返す (c̄ は平均費用)．
fn bertrand_n(params: &DemandParams, costs: &[f64]) -> Vec<f64> {
    let DemandParams { beta, d, alpha, .. } = *params;
    let n = costs.len() as f64;
    let c_bar = costs.iter().sum::<f64>() / n;
    let denom = 2.0 * beta - (n - 1.0) * d;
    let p = if denom.abs() < 1e-12 {
        c_bar
    } else {
        (alpha + beta * c_bar) / denom
    };
    costs.iter().map(|_| p).collect()
}

/// ベルトラン均衡価格ベクトルを計算する (N=2 は複占公式, N>=3 は一般化)．
pub fn bertrand_prices(params: &DemandParams, costs: &[f64]) -> Vec<f64> {
    match costs.len() {
        0 => Vec::new(),
        1 => vec![costs[0]], // 1 社は限界費用 (退化; 実用上は使わない)．
        2 => {
            let (p1, p2) = bertrand_duopoly(params, costs[0], costs[1]);
            vec![p1, p2]
        }
        _ => bertrand_n(params, costs),
    }
}

/// カルテル (独占) 価格ベクトルを計算する (論文 §4.3 公式, d/β ≠ 1)．
///
/// `p_i^M = α / (2 (β - (n-1) d)) + c_i / 2` (n=2 で `α/(2(β-d)) + c_i/2`)．
/// 同質財 (β - (n-1) d → 0) では発散しうるため，分母が極小のときは限界費用に倒す
/// (同質財では理論上カルテルは限界費用近傍へ；退化を安全に扱う)．
pub fn cartel_prices(params: &DemandParams, costs: &[f64]) -> Vec<f64> {
    let DemandParams { beta, d, alpha, .. } = *params;
    let n = costs.len() as f64;
    let denom = 2.0 * (beta - (n - 1.0) * d);
    costs
        .iter()
        .map(|&c| {
            if denom.abs() < 1e-12 {
                c
            } else {
                alpha / denom + c / 2.0
            }
        })
        .collect()
}

/// collusion index `CI_i = (p_i - p_i^B) / (p_i^M - p_i^B)` ∈ [0, 1]．
///
/// 0 = ベルトラン均衡, 1 = カルテル．`p^M = p^B` (区間が退化) のときは 0 を返す．
/// 範囲外は実情報として保持する (クランプしない; 価格が均衡を下回る/カルテルを
/// 超える可能性も観測できるように)．
pub fn collusion_index(price: f64, p_bertrand: f64, p_cartel: f64) -> f64 {
    let span = p_cartel - p_bertrand;
    if span.abs() < 1e-12 {
        0.0
    } else {
        (price - p_bertrand) / span
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn basic() -> DemandParams {
        DemandParams::new(14.0, 1.0 / 150.0, 1.0 / 300.0)
    }

    #[test]
    fn basic_setup_bertrand_is_exactly_six() {
        let p = basic();
        let b = bertrand_prices(&p, &[2.0, 2.0]);
        assert!((b[0] - 6.0).abs() < 1e-9, "p1^B={}", b[0]);
        assert!((b[1] - 6.0).abs() < 1e-9, "p2^B={}", b[1]);
    }

    #[test]
    fn basic_setup_cartel_is_exactly_eight() {
        let p = basic();
        let m = cartel_prices(&p, &[2.0, 2.0]);
        assert!((m[0] - 8.0).abs() < 1e-9, "p1^M={}", m[0]);
        assert!((m[1] - 8.0).abs() < 1e-9, "p2^M={}", m[1]);
    }

    #[test]
    fn collusion_index_endpoints() {
        // p=p^B → 0, p=p^M → 1, 中点 → 0.5．
        assert!((collusion_index(6.0, 6.0, 8.0) - 0.0).abs() < 1e-12);
        assert!((collusion_index(8.0, 6.0, 8.0) - 1.0).abs() < 1e-12);
        assert!((collusion_index(7.0, 6.0, 8.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn collusion_index_degenerate_span_is_zero() {
        assert_eq!(collusion_index(7.0, 6.0, 6.0), 0.0);
    }

    #[test]
    fn asymmetric_cost_bertrand_ordering() {
        // c_1 < c_2 → 低コスト社のほうが低い均衡価格 (Calvano 整合)．
        let p = basic();
        let b = bertrand_prices(&p, &[2.0, 5.0]);
        assert!(b[0] < b[1], "low-cost firm should price lower: {b:?}");
        // カルテルも同様の順序．
        let m = cartel_prices(&p, &[2.0, 5.0]);
        assert!(m[0] < m[1]);
    }

    #[test]
    fn monopoly_limit_d_beta_zero() {
        // d/β = 0 (独立市場) → ベルトラン = カルテル = 独占価格 α/(2β) + c/2．
        let p = DemandParams::new(14.0, 1.0 / 150.0, 0.0);
        let b = bertrand_prices(&p, &[2.0, 2.0]);
        let m = cartel_prices(&p, &[2.0, 2.0]);
        // d=0 ではカルテルとベルトランが一致する (相互依存がない)．
        assert!((b[0] - m[0]).abs() < 1e-9, "b={} m={}", b[0], m[0]);
    }
}
