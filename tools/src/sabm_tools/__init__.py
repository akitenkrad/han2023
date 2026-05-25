"""sabm-tools — Han, Wu & Xiao (2023) SABM 再現実装の可視化・分析ツール．

Rust シミュレーション (`sabm`) の出力 (rounds.csv / metrics.csv / benchmarks.json /
sweep_summary.csv) を読み，価格軌跡 (ベルトラン/カルテル基準線つき)・collusion index・
感度分析 (d/β × 企業数) を可視化する．
"""

__all__ = ["cli"]
