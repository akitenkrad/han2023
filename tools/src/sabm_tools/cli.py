"""sabm-tools — Han, Wu & Xiao (2023) SABM 再現実装 ツール統合 CLI．

Usage:
    sabm-tools visualize [...]
    sabm-tools visualize-sweep [...]
    sabm-tools show-experiment-settings [...]
    sabm-tools reproduce [...]

各サブコマンドに続く引数は，対応するモジュールの argparse がそのまま受け取る．
サブコマンドレベルで `--help` を付けると，そのサブコマンド自身のヘルプが表示される．
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="sabm-tools",
        description="Han, Wu & Xiao (2023) SABM 暗黙の共謀シミュレーション 可視化・分析ツール",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "visualize",
        help="単一実行結果 (価格軌跡 + collusion index; Bertrand/cartel 基準線つき) の可視化",
        add_help=False,
    )
    subparsers.add_parser(
        "visualize-sweep",
        help="スイープ結果 (collusion index の d/β × 企業数 依存) の可視化",
        add_help=False,
    )
    subparsers.add_parser(
        "show-experiment-settings",
        help="実行結果ディレクトリの設定 (config / sweep_config / llm_meta) の表示",
        add_help=False,
    )
    subparsers.add_parser(
        "reproduce",
        help="論文 Fig.1/2/4 一括再現 (Phase 3; 現状はスタブ案内)",
        add_help=False,
    )

    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help"}:
        parser.parse_args(argv)
        return

    command = argv[0]
    rest = argv[1:]
    if command == "visualize":
        from sabm_tools.visualize import main as run_main

        run_main(rest)
    elif command == "visualize-sweep":
        from sabm_tools.visualize_sweep import main as run_main

        run_main(rest)
    elif command == "show-experiment-settings":
        from sabm_tools.show_experiment_settings import main as run_main

        run_main(rest)
    elif command == "reproduce":
        from sabm_tools.reproduce_paper import main as run_main

        run_main(rest)
    else:
        parser.parse_args(argv)


if __name__ == "__main__":
    main()
