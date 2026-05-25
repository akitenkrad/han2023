#!/usr/bin/env python3
"""
visualize.py — Han, Wu & Xiao (2023) SABM 価格軌跡 可視化スクリプト

results/latest (または --results_dir 指定先) の rounds.csv / metrics.csv /
benchmarks.json を読み，
(1) 価格軌跡図 (ラウンド × 価格; 各社の価格 + 全社平均 + ベルトラン均衡/カルテル
    基準線; 論文 Fig.1/2 風) と，
(2) collusion index の時系列 (0 = ベルトラン均衡, 1 = カルテル; 暗黙の共謀の度合い)
を生成する．

Usage:
    uv run sabm-tools visualize
    uv run sabm-tools visualize --results_dir results/20260525_120000
    uv run sabm-tools visualize --output_dir out

Outputs:
    output_dir/
    ├── price_trajectory.png    ← ラウンド × 価格 (Bertrand/cartel 基準線つき)
    └── collusion_index.png     ← collusion index の時系列
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# 日本語フォント設定
# --------------------------------------------------------------------------- #
plt.rcParams["font.family"] = "Hiragino Sans"

# --------------------------------------------------------------------------- #
# カラー設定
# --------------------------------------------------------------------------- #
COLOR_BG = "#FAFAF8"
COLOR_AVG = "#1565C0"
COLOR_BERTRAND = "#2E7D32"
COLOR_CARTEL = "#C62828"
COLOR_CI = "#6A1B9A"
FIRM_COLORS = ["#FF9800", "#03A9F4", "#8BC34A", "#E91E63", "#795548"]


def load_rounds(results_dir: str) -> pd.DataFrame:
    """rounds.csv (long-format: round, firm_id, price, quantity, profit) を読む．"""
    path = os.path.join(results_dir, "rounds.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"rounds.csv が見つかりません: {path}")
    return pd.read_csv(path)


def load_metrics(results_dir: str) -> pd.DataFrame:
    """metrics.csv (round, avg_price, collusion_index, total_profit) を読む．"""
    path = os.path.join(results_dir, "metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"metrics.csv が見つかりません: {path}")
    return pd.read_csv(path)


def load_benchmarks(results_dir: str) -> tuple[float, float]:
    """benchmarks.json から (p_bertrand_mean, p_cartel_mean) を読む．"""
    path = os.path.join(results_dir, "benchmarks.json")
    if not os.path.exists(path):
        return float("nan"), float("nan")
    with open(path) as f:
        b = json.load(f)
    return float(b.get("p_bertrand_mean", float("nan"))), float(
        b.get("p_cartel_mean", float("nan"))
    )


def save_price_trajectory(
    rounds: pd.DataFrame,
    metrics: pd.DataFrame,
    p_bertrand: float,
    p_cartel: float,
    out_path: str,
) -> None:
    """各社の価格軌跡 + 全社平均 + Bertrand/cartel 基準線を描く (Fig.1/2 風)．"""
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    fig.suptitle("Han, Wu & Xiao (2023) SABM — 価格軌跡と暗黙の共謀", fontsize=14)

    # ベルトラン↔カルテルの帯 (暗黙の共謀が現れる領域)．
    if np.isfinite(p_bertrand) and np.isfinite(p_cartel):
        lo, hi = sorted((p_bertrand, p_cartel))
        ax.axhspan(lo, hi, color="#FFC107", alpha=0.08)

    # 各社の価格．
    for i, (firm_id, sub) in enumerate(rounds.groupby("firm_id")):
        ax.plot(
            sub["round"],
            sub["price"],
            color=FIRM_COLORS[i % len(FIRM_COLORS)],
            lw=1.2,
            alpha=0.7,
            label=f"firm {firm_id}",
        )

    # 全社平均．
    ax.plot(
        metrics["round"],
        metrics["avg_price"],
        color=COLOR_AVG,
        lw=2.4,
        label="全社平均価格",
    )

    # 基準線．
    if np.isfinite(p_bertrand):
        ax.axhline(
            p_bertrand,
            color=COLOR_BERTRAND,
            lw=1.6,
            linestyle="--",
            label=f"ベルトラン均衡 p^B={p_bertrand:.2f}",
        )
    if np.isfinite(p_cartel):
        ax.axhline(
            p_cartel,
            color=COLOR_CARTEL,
            lw=1.6,
            linestyle="--",
            label=f"カルテル p^M={p_cartel:.2f}",
        )

    ax.set_xlabel("ラウンド t")
    ax.set_ylabel("価格 p")
    ax.set_title("会話なしでも価格は (ベルトラン, カルテル) 区間へ収束 (暗黙の共謀)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def save_collusion_index(metrics: pd.DataFrame, out_path: str) -> None:
    """collusion index の時系列 (0=Bertrand, 1=cartel) を描く．"""
    fig, ax = plt.subplots(figsize=(11, 4.5), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    fig.suptitle("Han, Wu & Xiao (2023) SABM — collusion index 時系列", fontsize=14)

    ax.plot(metrics["round"], metrics["collusion_index"], color=COLOR_CI, lw=2)
    ax.axhline(0.0, color=COLOR_BERTRAND, lw=1.2, linestyle="--", label="0 = ベルトラン均衡")
    ax.axhline(1.0, color=COLOR_CARTEL, lw=1.2, linestyle="--", label="1 = カルテル")
    ax.axhspan(0.3, 0.8, color="#FFC107", alpha=0.10, label="論文の暗黙共謀帯 (CI≈0.3-0.8)")
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("ラウンド t")
    ax.set_ylabel("collusion index CI")
    ax.set_title("CI = (p - p^B) / (p^M - p^B)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")
    last_ci = metrics["collusion_index"].iloc[-1] if not metrics.empty else float("nan")
    print(f"      最終 collusion index = {last_ci:.3f}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="sabm-tools visualize",
        description="Han, Wu & Xiao (2023) SABM 価格軌跡 可視化スクリプト",
    )
    p.add_argument(
        "--results_dir",
        "--results-dir",
        default="results/latest",
        help="Rust シミュレーションの出力ディレクトリ (default: results/latest)",
    )
    p.add_argument(
        "--output_dir",
        "--output-dir",
        default=None,
        help="図の保存先ディレクトリ (default: {results_dir}/figures)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    out_dir = args.output_dir if args.output_dir else os.path.join(args.results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    print("=== Han, Wu & Xiao (2023) SABM 価格軌跡 可視化 ===")
    print(f"結果:   {args.results_dir}")
    print(f"出力先: {out_dir}")
    print("-----------------------------------------")

    rounds = load_rounds(args.results_dir)
    metrics = load_metrics(args.results_dir)
    p_bertrand, p_cartel = load_benchmarks(args.results_dir)
    print(f"      {len(metrics)} ラウンド × {rounds['firm_id'].nunique()} 社")
    print(f"      benchmarks: p_bertrand={p_bertrand:.3f} p_cartel={p_cartel:.3f}")

    print("[1/2] 価格軌跡図を保存中 ...")
    save_price_trajectory(
        rounds, metrics, p_bertrand, p_cartel, os.path.join(out_dir, "price_trajectory.png")
    )

    print("[2/2] collusion index 時系列を保存中 ...")
    save_collusion_index(metrics, os.path.join(out_dir, "collusion_index.png"))

    print("-----------------------------------------")
    print("完了．出力ファイル一覧:")
    for f in sorted(os.listdir(out_dir)):
        size_kb = os.path.getsize(os.path.join(out_dir, f)) / 1024
        print(f"  {f:35s} ({size_kb:6.1f} KB)")


if __name__ == "__main__":
    main()
