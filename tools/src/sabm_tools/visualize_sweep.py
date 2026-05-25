#!/usr/bin/env python3
"""
visualize_sweep.py — Han, Wu & Xiao (2023) SABM スイープ結果 可視化スクリプト

results/latest (または --sweep_dir 指定先) の sweep_summary.csv を読み，
製品差別化度 d/β × 企業数 の格子について最終 collusion index を集計し，
折れ線グラフ (d/β 別 / 企業数別) とヒートマップ (d/β × 企業数) で可視化する．

Usage:
    uv run sabm-tools visualize-sweep
    uv run sabm-tools visualize-sweep --sweep_dir results/20260525_120000_sweep

Outputs:
    output_dir/
    ├── sweep_ci_by_dbeta.png      ← d/β 別の平均 collusion index (企業数で色分け)
    └── sweep_ci_heatmap.png       ← collusion index (d/β × 企業数) ヒートマップ
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Hiragino Sans"

COLOR_BG = "#FAFAF8"
FIRM_COLORS = ["#FF9800", "#03A9F4", "#8BC34A", "#E91E63", "#795548"]


def load_summary(sweep_dir: str) -> pd.DataFrame:
    """sweep_summary.csv を読み込む．"""
    path = os.path.join(sweep_dir, "sweep_summary.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"sweep_summary.csv が見つかりません: {path}")
    return pd.read_csv(path)


def save_ci_by_dbeta(df: pd.DataFrame, out_path: str) -> None:
    """d/β 別の平均 collusion index を企業数で色分けして折れ線で描く．"""
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    fig.suptitle("Han, Wu & Xiao (2023) SABM — 差別化度と共謀", fontsize=14)

    for i, (firms, sub) in enumerate(df.groupby("firms")):
        agg = sub.groupby("d_beta")["final_collusion_index"].mean().reset_index()
        ax.plot(
            agg["d_beta"],
            agg["final_collusion_index"],
            marker="o",
            color=FIRM_COLORS[i % len(FIRM_COLORS)],
            lw=2,
            label=f"{firms} 社",
        )

    ax.axhspan(0.3, 0.8, color="#FFC107", alpha=0.10, label="暗黙共謀帯 (CI≈0.3-0.8)")
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("製品差別化度 d/β (0 = 独立市場, 1 = 同質財)")
    ax.set_ylabel("最終 collusion index (平均)")
    ax.set_title("0=ベルトラン均衡, 1=カルテル")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def save_ci_heatmap(df: pd.DataFrame, out_path: str) -> None:
    """collusion index を (d/β × 企業数) ヒートマップで可視化する．"""
    agg = df.groupby(["d_beta", "firms"])["final_collusion_index"].mean().reset_index()
    table = agg.pivot(index="d_beta", columns="firms", values="final_collusion_index")
    table = table.sort_index()

    fig, ax = plt.subplots(
        figsize=(1.8 + 1.4 * table.shape[1], 1.6 + 0.9 * table.shape[0]),
        facecolor=COLOR_BG,
    )
    ax.set_facecolor(COLOR_BG)
    data = table.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="magma", aspect="auto", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels([f"{s:g}" for s in table.index])
    ax.set_xlabel("企業数 n")
    ax.set_ylabel("製品差別化度 d/β")
    ax.set_title("最終 collusion index (d/β × 企業数)", fontsize=12)

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10, color="white")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="sabm-tools visualize-sweep",
        description="Han, Wu & Xiao (2023) SABM スイープ結果 可視化スクリプト",
    )
    p.add_argument(
        "--sweep_dir",
        "--sweep-dir",
        default="results/latest",
        help="スイープ出力ディレクトリ (default: results/latest)",
    )
    p.add_argument(
        "--output_dir",
        "--output-dir",
        default=None,
        help="図の保存先ディレクトリ (default: {sweep_dir}/figures)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    out_dir = args.output_dir if args.output_dir else os.path.join(args.sweep_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    print("=== Han, Wu & Xiao (2023) SABM スイープ可視化 ===")
    print(f"スイープ: {args.sweep_dir}")
    print(f"出力先:   {out_dir}")
    print("-------------------------------------------------")

    print("[1/2] sweep_summary.csv を読み込み中 ...")
    df = load_summary(args.sweep_dir)
    print(
        f"      d/β {df['d_beta'].nunique()} 種 × 企業数 {df['firms'].nunique()} 種 "
        f"({len(df)} 実行)"
    )

    print("[2/2] d/β 別 collusion index 図を保存中 ...")
    save_ci_by_dbeta(df, os.path.join(out_dir, "sweep_ci_by_dbeta.png"))
    if df["firms"].nunique() > 1:
        save_ci_heatmap(df, os.path.join(out_dir, "sweep_ci_heatmap.png"))
    else:
        print("      企業数が単一のためヒートマップはスキップ")

    print("-------------------------------------------------")
    print("d/β 別の平均 collusion index:")
    for d_beta in sorted(df["d_beta"].unique()):
        ci = df[df["d_beta"] == d_beta]["final_collusion_index"].mean()
        print(f"  d/β={d_beta:<5g} → CI = {ci:.3f}")

    print("-------------------------------------------------")
    print("完了．出力ファイル一覧:")
    for f in sorted(os.listdir(out_dir)):
        size_kb = os.path.getsize(os.path.join(out_dir, f)) / 1024
        print(f"  {f:35s} ({size_kb:6.1f} KB)")


if __name__ == "__main__":
    main()
