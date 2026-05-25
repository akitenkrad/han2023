"""sabm-tools reproduce — 論文 Fig.1/2/4 一括再現 (Phase 3 スタブ)．

論文 Han, Wu & Xiao (2023) の Figure 1 (会話なしの価格収束)・Figure 2 (会話ありの
価格収束)・Figure 4 (会話の有無による安定共謀形成ラウンド数の差) を一括で再現する
計画．現状は Phase 3 の実装待ちのため，案内メッセージを表示するスタブである．

Phase 1/2 で利用できるもの:
  - 単一実行の価格軌跡 / collusion index → `sabm-tools visualize`
  - 感度分析 (d/β × 企業数) の collusion index → `sabm-tools visualize-sweep`

Usage:
    sabm-tools reproduce
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sabm-tools reproduce",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.parse_args(argv)

    print("=== Han, Wu & Xiao (2023) SABM 論文 Fig.1/2/4 一括再現 ===")
    print("reproduce は Phase 3 で実装予定です (現状はスタブ)．")
    print("会話あり/なしの収束水準・形成速度・滑らかさの比較を一括で行う計画です．")
    print("")
    print("現時点では以下を個別にご利用ください:")
    print("  - 価格軌跡 / collusion index :  uv run sabm-tools visualize")
    print("  - d/β × 企業数 の感度分析    :  uv run sabm-tools visualize-sweep")
    print("  - 解析ベンチマーク (p^B/p^M)  :  cargo run --release -- benchmark")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
