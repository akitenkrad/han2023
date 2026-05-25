"""sabm-tools — Han, Wu & Xiao (2023) SABM 再現実装 ツール統合 CLI．

Usage:
    sabm-tools visualize [...]
    sabm-tools visualize-sweep [...]
    sabm-tools show-experiment-settings [...]
    sabm-tools reproduce [...]

各サブコマンドに続く引数は，対応するモジュールの argparse がそのまま受け取る．
サブコマンドレベルで `--help` を付けると，そのサブコマンド自身のヘルプが表示される．

dispatcher の組み立ては共有ヘルパ `socsim_tools.cli.build_dispatcher` に委譲する
(prog 名・サブコマンド・ヘルプ文・argv ルーティングは従来と同一)．可視化/設定表示/
reproduce の実体は repo 固有のまま (`reproduce` は SABM 固有の追加サブコマンド)．
"""

from __future__ import annotations

from socsim_tools.cli import build_dispatcher

main = build_dispatcher(
    prog="sabm-tools",
    description="Han, Wu & Xiao (2023) SABM 暗黙の共謀シミュレーション 可視化・分析ツール",
    subcommands={
        "visualize": (
            "単一実行結果 (価格軌跡 + collusion index; Bertrand/cartel 基準線つき) の可視化",
            "sabm_tools.visualize:main",
        ),
        "visualize-sweep": (
            "スイープ結果 (collusion index の d/β × 企業数 依存) の可視化",
            "sabm_tools.visualize_sweep:main",
        ),
        "show-experiment-settings": (
            "実行結果ディレクトリの設定 (config / sweep_config / llm_meta) の表示",
            "sabm_tools.show_experiment_settings:main",
        ),
        "reproduce": (
            "論文 Fig.1/2/4 一括再現 (Phase 3; 現状はスタブ案内)",
            "sabm_tools.reproduce_paper:main",
        ),
    },
)


if __name__ == "__main__":
    main()
