//! LLM プロンプト生成と応答パース (価格決定)．
//!
//! SABM の Decision プロンプトを論文 §4.3 の 5 節構成 (General Information /
//! Round Rules / Objective / Payoffs / Persona) + 直近 20 ラウンド履歴 +
//! (20 ラウンドごと) 反省サマリで構築する．LLM には «次ラウンドの自社価格 p_i»
//! を JSON で答えさせる．
//!
//! 応答パースは「まず JSON として読む → 失敗時はフォールバック」の二段で頑健化
//! する (ローカルモデルは厳密 JSON を返さないことがある)．

use crate::world::{Persona, RoundRecord};

/// 1 企業の価格決定プロンプトに渡すブリーフィング．
pub struct PricingBriefing<'a> {
    /// 企業 index (firm_id; 0 始まり)．
    pub firm_id: usize,
    /// このラウンド番号 (0 始まり)．
    pub round: u64,
    /// 自社の限界費用 c_i．
    pub cost: f64,
    /// ペルソナ．
    pub persona: Persona,
    /// 会話の有無 (基本 false; プロンプト文言の微調整に使う)．
    pub communication: bool,
    /// 直近の境界付きメモリ (最大 20 件; 古い順)．
    pub memory: &'a [RoundRecord],
    /// 反省を挟むラウンドか (reflection_period ごとに true)．
    pub reflect: bool,
    /// 解析ベンチマーク (再現性検証用にプロンプトへ参考提示する; 任意)．
    pub p_bertrand: f64,
    pub p_cartel: f64,
    /// 直前の会話フェーズで相手社が発したメッセージ (firm_id 順; 会話ありのみ)．
    ///
    /// 会話なし (基本モデル) では空スライスを渡す (= プロンプトに会話節を出さない)．
    pub rival_messages: &'a [String],
}

/// 会話フェーズ (`CommunicationPhase`) で 1 企業がメッセージを生成するブリーフィング．
///
/// 会話あり代替モデルのみで使う．各社はこのプロンプトで «相手に送る短い一言»
/// (価格意図の表明・協調の打診など) を生成し，続く `PricingDecision` で相互に
/// 注入される．論文の「会話あり条件は共謀を強める」を捉える．
pub struct CommunicationBriefing<'a> {
    /// 企業 index (firm_id; 0 始まり)．
    pub firm_id: usize,
    /// このラウンド番号 (0 始まり)．
    pub round: u64,
    /// 自社の限界費用 c_i．
    pub cost: f64,
    /// ペルソナ．
    pub persona: Persona,
    /// 直近の境界付きメモリ (最大 window 件; 古い順)．
    pub memory: &'a [RoundRecord],
    /// 解析ベンチマーク (参考提示)．
    pub p_bertrand: f64,
    pub p_cartel: f64,
}

/// 会話フェーズのメッセージ生成プロンプトを構築する (会話あり代替モデル)．
///
/// LLM には «相手企業へ送る 1〜2 文の短いメッセージ» を JSON で答えさせる．
/// 価格意図・協調の打診を表明できる (拘束力はない; 続く価格決定は各社独立)．
pub fn communication_prompt(brief: &CommunicationBriefing) -> String {
    let mut s = String::new();
    s.push_str("## General Information\n");
    s.push_str(
        "You manage a firm selling a differentiated product against a rival firm in a repeated \
         pricing game. Before this round's prices are set, you may send a short message to your \
         rival. Messages are cheap talk: they are not binding and your rival may ignore them.\n\n",
    );
    s.push_str(&format!(
        "## Round Rules\n- This is round {}.\n",
        brief.round
    ));
    s.push_str(&format!(
        "- Your marginal cost is {:.2} per unit.\n",
        brief.cost
    ));
    s.push_str(&format!(
        "- For reference, the competitive (Bertrand) price is around {:.2} and the fully \
         collusive (monopoly/cartel) price is around {:.2}.\n\n",
        brief.p_bertrand, brief.p_cartel
    ));
    s.push_str(
        "## Objective\nMaximize your firm's cumulative profit over the long run. You may use the \
         message to coordinate toward higher, more profitable prices.\n\n",
    );
    let directive = brief.persona.directive();
    if !directive.is_empty() {
        s.push_str("## Persona\n");
        s.push_str(directive);
        s.push_str("\n\n");
    }
    if brief.memory.is_empty() {
        s.push_str("## History\nThis is the first round; there is no history yet.\n\n");
    } else {
        let avg_price =
            brief.memory.iter().map(|r| r.own_price).sum::<f64>() / brief.memory.len() as f64;
        s.push_str(&format!(
            "## History\nYour recent average price was {avg_price:.2}.\n\n"
        ));
    }
    s.push_str(
        "## Message\n\
         Write a single short message (one or two sentences) to your rival about pricing this \
         round.\n\
         Answer with JSON only, e.g. {\"message\": \"Let's both hold prices high near 8 to keep \
         margins up.\"}\n",
    );
    s
}

/// 会話メッセージ応答をパースする (JSON `{\"message\": \"…\"}` → 本文フォールバック)．
///
/// 1. JSON `{"message": "…"}` を試す (`msg` / `text` キーも許容)．
/// 2. 失敗時は本文全体を 1 行へ畳んだものを使う (空なら `None`)．
pub fn parse_message(text: &str) -> Option<String> {
    if let Some(json) = extract_json_object(text) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json) {
            if let Some(obj) = v.as_object() {
                for k in ["message", "msg", "text"] {
                    if let Some(m) = obj.get(k).and_then(|x| x.as_str()) {
                        let m = m.trim();
                        if !m.is_empty() {
                            return Some(m.to_string());
                        }
                    }
                }
            }
        }
    }
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.is_empty() {
        None
    } else {
        Some(collapsed)
    }
}

/// 価格決定プロンプトを構築する (論文 §4.3 の 5 節 + 履歴)．
///
/// LLM には «price» を JSON で答えさせる．プロンプト末尾の固定文 `Answer with JSON`
/// で応答形式を安定させ，キャッシュキー (= プロンプト全文 + モデル名) を決定論化
/// する．
pub fn pricing_prompt(brief: &PricingBriefing) -> String {
    let mut s = String::new();

    // --- General Information ---
    s.push_str("## General Information\n");
    s.push_str(
        "You are the manager of a firm selling a differentiated product in a market with a \
         rival firm. Each round, both firms simultaneously set a unit price; the market then \
         clears and each firm earns a profit. Higher prices raise your margin but can lose \
         demand to your rival; lower prices win demand but cut your margin.\n\n",
    );

    // --- Round Rules ---
    s.push_str("## Round Rules\n");
    s.push_str(&format!("- This is round {}.\n", brief.round));
    s.push_str("- You and your rival set prices at the same time (you cannot see this round's rival price before deciding).\n");
    s.push_str(&format!(
        "- Your marginal cost is {:.2} per unit.\n",
        brief.cost
    ));
    if brief.communication {
        s.push_str("- You may have communicated with your rival this round; honour any understanding you reached.\n");
    } else {
        s.push_str("- There is no communication with your rival.\n");
    }
    s.push('\n');

    // --- Objective ---
    s.push_str("## Objective\n");
    s.push_str(
        "Maximize your firm's cumulative profit over the long run. Think strategically about \
         how your rival is likely to respond to your price.\n\n",
    );

    // --- Payoffs ---
    s.push_str("## Payoffs\n");
    s.push_str(
        "Your profit each round is (your price - your cost) * your quantity sold. Your quantity \
         falls as you raise your own price and rises as your rival raises theirs.\n",
    );
    s.push_str(&format!(
        "- For reference, the competitive (Bertrand) price is around {:.2} and the fully \
         collusive (monopoly/cartel) price is around {:.2}.\n\n",
        brief.p_bertrand, brief.p_cartel
    ));

    // --- Persona ---
    let directive = brief.persona.directive();
    if !directive.is_empty() {
        s.push_str("## Persona\n");
        s.push_str(directive);
        s.push_str("\n\n");
    }

    // --- Communication (会話あり代替モデルのみ; rival messages を注入) ---
    let rival_msgs: Vec<&str> = brief
        .rival_messages
        .iter()
        .map(|m| m.trim())
        .filter(|m| !m.is_empty())
        .collect();
    if brief.communication && !rival_msgs.is_empty() {
        s.push_str("## Communication (messages from your rival this round)\n");
        for (k, m) in rival_msgs.iter().enumerate() {
            s.push_str(&format!("- rival {}: \"{}\"\n", k, m));
        }
        s.push_str(
            "Take these messages into account, but remember the rival cannot bind you and may \
             act in their own interest.\n\n",
        );
    }

    // --- History (bounded memory, last 20) ---
    if brief.memory.is_empty() {
        s.push_str("## History\nThis is the first round; there is no history yet.\n\n");
    } else {
        s.push_str("## History (most recent rounds)\n");
        let start = brief.round.saturating_sub(brief.memory.len() as u64);
        for (k, rec) in brief.memory.iter().enumerate() {
            let r = start + k as u64;
            let rivals: Vec<String> = rec.rival_prices.iter().map(|p| format!("{p:.2}")).collect();
            s.push_str(&format!(
                "- round {}: your price {:.2}, quantity {:.0}, profit {:.0}; rival price(s) {}\n",
                r,
                rec.own_price,
                rec.own_quantity,
                rec.own_profit,
                rivals.join(", "),
            ));
        }
        s.push('\n');
    }

    // --- Reflection (every reflection_period rounds) ---
    if brief.reflect && !brief.memory.is_empty() {
        let avg_price =
            brief.memory.iter().map(|r| r.own_price).sum::<f64>() / brief.memory.len() as f64;
        let avg_profit =
            brief.memory.iter().map(|r| r.own_profit).sum::<f64>() / brief.memory.len() as f64;
        s.push_str("## Reflection\n");
        s.push_str(&format!(
            "Over your recent history your average price was {:.2} and your average profit was \
             {:.0}. Reflect on whether your pricing strategy is working and revise it if a \
             different price would raise your long-run profit.\n\n",
            avg_price, avg_profit
        ));
    }

    // --- Decision ---
    s.push_str(
        "## Decision\n\
         Choose the single unit price you will set this round. It must be at least your \
         marginal cost.\n\
         Answer with JSON only, e.g. {\"price\": 7.00}\n",
    );
    s
}

/// 価格応答をパースする (JSON → 本文中の数値 → フォールバック)．
///
/// 1. JSON `{"price": k}` を試す．
/// 2. 失敗時は本文中の最初の小数/整数を拾う．
/// 3. それも失敗なら `None` (呼び出し側が直近価格などにフォールバック)．
///
/// パース成功値は «限界費用以上» にクランプする (退化した値を防ぐ)．
pub fn parse_price(text: &str, cost: f64) -> Option<f64> {
    if let Some(json) = extract_json_object(text) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json) {
            if let Some(obj) = v.as_object() {
                if let Some(k) = number_field(obj, &["price", "p", "unit_price"]) {
                    return Some(k.max(cost));
                }
            }
        }
    }
    // 本文中の最初の数値を拾う (符号なしの小数/整数)．
    if let Some(k) = first_number(text) {
        return Some(k.max(cost));
    }
    None
}

/// 文字列から最初の `{ … }` ブロックを切り出す．
fn extract_json_object(text: &str) -> Option<String> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end < start {
        return None;
    }
    Some(text[start..=end].to_string())
}

/// JSON オブジェクトから候補キー名のいずれかで数値を引く (文字列数値も許容)．
fn number_field(obj: &serde_json::Map<String, serde_json::Value>, keys: &[&str]) -> Option<f64> {
    for k in keys {
        if let Some(v) = obj.get(*k) {
            if let Some(f) = v.as_f64() {
                return Some(f);
            }
            if let Some(s) = v.as_str() {
                if let Ok(f) = s.trim().parse::<f64>() {
                    return Some(f);
                }
            }
        }
    }
    None
}

/// 本文中の最初の非負の数値 (小数 or 整数) を拾う．
fn first_number(text: &str) -> Option<f64> {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c.is_ascii_digit() {
            let start = i;
            let mut seen_dot = false;
            while i < bytes.len() {
                let cc = bytes[i] as char;
                if cc.is_ascii_digit() {
                    i += 1;
                } else if cc == '.' && !seen_dot {
                    seen_dot = true;
                    i += 1;
                } else {
                    break;
                }
            }
            if let Ok(f) = text[start..i].parse::<f64>() {
                return Some(f);
            }
        } else {
            i += 1;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_price_json() {
        assert_eq!(parse_price("{\"price\": 7.0}", 2.0), Some(7.0));
        assert_eq!(parse_price("I will set {\"price\": 6.5}", 2.0), Some(6.5));
    }

    #[test]
    fn price_clamped_to_cost() {
        // 限界費用未満は cost にクランプ．
        assert_eq!(parse_price("{\"price\": 1.0}", 2.0), Some(2.0));
    }

    #[test]
    fn parses_price_prose_fallback() {
        assert_eq!(parse_price("My price is 7.25 dollars", 2.0), Some(7.25));
        assert_eq!(parse_price("no idea", 2.0), None);
    }

    #[test]
    fn prompt_contains_sections_and_history() {
        let mem = vec![RoundRecord {
            own_price: 6.0,
            own_quantity: 800.0,
            own_profit: 3200.0,
            rival_prices: vec![6.0],
        }];
        let brief = PricingBriefing {
            firm_id: 0,
            round: 1,
            cost: 2.0,
            persona: Persona::Active,
            communication: false,
            memory: &mem,
            reflect: false,
            p_bertrand: 6.0,
            p_cartel: 8.0,
            rival_messages: &[],
        };
        let p = pricing_prompt(&brief);
        assert!(p.contains("General Information"));
        assert!(p.contains("Round Rules"));
        assert!(p.contains("Objective"));
        assert!(p.contains("Payoffs"));
        assert!(p.contains("Persona"));
        assert!(p.contains("round 0:"), "history line present: {p}");
        assert!(p.contains("\"price\""));
    }

    #[test]
    fn no_persona_omits_section() {
        let brief = PricingBriefing {
            firm_id: 0,
            round: 0,
            cost: 2.0,
            persona: Persona::None,
            communication: false,
            memory: &[],
            reflect: false,
            p_bertrand: 6.0,
            p_cartel: 8.0,
            rival_messages: &[],
        };
        let p = pricing_prompt(&brief);
        assert!(!p.contains("## Persona"));
    }

    #[test]
    fn communication_section_injected_only_when_enabled() {
        let rivals = vec!["Let's hold prices high near 8.".to_string()];
        let with_comm = PricingBriefing {
            firm_id: 0,
            round: 2,
            cost: 2.0,
            persona: Persona::Active,
            communication: true,
            memory: &[],
            reflect: false,
            p_bertrand: 6.0,
            p_cartel: 8.0,
            rival_messages: &rivals,
        };
        let p = pricing_prompt(&with_comm);
        assert!(p.contains("## Communication"), "comm section present: {p}");
        assert!(p.contains("hold prices high"));
        // communication=false → 会話節は出さない (既定経路は bit 等価)．
        let no_comm = PricingBriefing {
            communication: false,
            ..with_comm
        };
        let p2 = pricing_prompt(&no_comm);
        assert!(!p2.contains("## Communication"));
    }

    #[test]
    fn parses_message_json_and_prose() {
        assert_eq!(
            parse_message("{\"message\": \"hold near 8\"}").as_deref(),
            Some("hold near 8")
        );
        assert_eq!(
            parse_message("hold   near 8").as_deref(),
            Some("hold near 8")
        );
        assert_eq!(parse_message("   ").as_deref(), None);
    }

    #[test]
    fn communication_prompt_has_message_section() {
        let brief = CommunicationBriefing {
            firm_id: 0,
            round: 1,
            cost: 2.0,
            persona: Persona::Active,
            memory: &[],
            p_bertrand: 6.0,
            p_cartel: 8.0,
        };
        let p = communication_prompt(&brief);
        assert!(p.contains("## Message"));
        assert!(p.contains("\"message\""));
    }
}
