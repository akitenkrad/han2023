//! Han, Wu & Xiao (2023) "Guinea Pig Trials Utilizing GPT: A Novel Smart
//! Agent-Based Modeling Approach for Studying Firm Competition and Collusion"
//! (SABM, arXiv:2308.10974) の再現実装ライブラリ．
//!
//! socsim フレームワーク上に構築した LLM 駆動の差別化財ベルトラン複占 (基本は 2 社)
//! 繰り返し価格ゲームの公開 API を提供する．設定 (`config`)・需要関数 (`demand`)・
//! 解析ベンチマーク (`analytic`)・世界状態 (`world`)・LLM クライアント層 (`llm`)・
//! プロンプト生成と応答パース (`prompts`)・更新メカニズム (`mechanisms`)・実行
//! ドライバ (`simulation`)・集計メトリクス (`metrics`)・runvault への記録
//! (`record`) をモジュールとして公開し，バイナリ (`sabm`) と統合テストの双方から
//! 利用する．
//!
//! # 二層決定論
//!
//! socsim コア層 (初期価格・市場クリアリング・利益計算・解析ベンチマーク・境界付き
//! メモリ更新) は seed から bit 単位で決定論的である．LLM レイヤ (価格決定) は
//! socsim の bit 再現性の **外側** にあり，`socsim-llm` のキャッシュ +
//! `temperature=0` + `seed` 固定で擬似決定論化する．詳細は `crate::llm` を参照．
//! 設計書 §4.2/§7 は当初 `reqwest` + `sha2` を挙げていたが，本スイートは li2024 /
//! zhao2024 / chuang2024 と統一して `socsim-llm` (issue #21/#26) に標準化したため
//! `reqwest` / `sha2` は使わない (socsim-llm が HTTP とプロンプトハッシュを所有する)．
//!
//! # 解析ベンチマークは厳密
//!
//! `analytic` の Bertrand/cartel 価格は LLM/socsim から独立な純数値計算であり，
//! 論文基本設定 (a=14, β=1/150, d=1/300, c=2, d/β=0.5) で `p^B = 6`, `p^M = 8` に
//! 厳密一致する (`benchmark` サブコマンドおよび単体テストで検算)．

pub mod analytic;
pub mod config;
pub mod demand;
pub mod llm;
pub mod mechanisms;
pub mod metrics;
pub mod prompts;
pub mod record;
pub mod simulation;
pub mod world;
