#!/usr/bin/env python3
"""
visualize.py — Han, Wu & Xiao (2023) SABM 価格軌跡 可視化スクリプト

runvault の run ディレクトリから，企業ごとの価格軌跡 (`events.jsonl` の
`observation` 行) とラウンドごとの市場指標 (`metrics.csv`) と解析ベンチマーク
(run スコープの `p_bertrand_mean` / `p_cartel_mean`) を読み，
(1) 価格軌跡図 (ラウンド × 価格; 各社の価格 + 全社平均 + ベルトラン均衡/カルテル
    基準線; 論文 Fig.1/2 風) と，
(2) collusion index の時系列 (0 = ベルトラン均衡, 1 = カルテル; 暗黙の共謀の度合い)
を生成する．

企業ごとの価格は時間 (ラウンド) と系列 (企業) の両方で決まるので `metrics.csv`
には置けず (主キーに系列の列が無い)，`events.jsonl` が唯一の置き場である．

どの run を見るかは `--results-dir` を省略すれば runvault が答える
(`runvault path --experiment sabm --latest --subcommand run --standalone`)．
`results/` を自分で走査して新しそうなディレクトリを当てにいくことはしない．

図は run ディレクトリの *隣* (`results/sabm/figures/<run_slug>/`) に置く．
`manifest.csv` は `finish()` が確定させたもので，run が終わった後に足したものは
そこに載らないためである．

Usage:
    uv run sabm-tools visualize
    uv run sabm-tools visualize --results-dir "$(runvault path --experiment sabm --latest --subcommand run --standalone)"
    uv run sabm-tools visualize --output-dir out

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
from runvault.read import (
    events_table,
    figures_dir,
    metrics_wide,
    run_scope_metrics,
    runvault_path,
)

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


def load_rounds(results_dir: str, prefix: str = "") -> pd.DataFrame:
    """企業 1 社 1 ラウンドの表 (round, firm_id, price, quantity, profit)．

    runvault では `events.jsonl` の `observation` 行が正本 (旧 `rounds.csv`)．
    legacy の run ディレクトリは `rounds.csv` をそのまま読む．
    """
    legacy = os.path.join(results_dir, "rounds.csv")
    if os.path.exists(legacy):
        return pd.read_csv(legacy)
    df = events_table(results_dir, kind="observation")
    unit_prefix = f"{prefix}-firm-" if prefix else "firm-"
    df = df[df["unit_id"].str.startswith(unit_prefix)].copy()
    df["firm_id"] = df["unit_id"].str.rsplit("-", n=1).str[-1].astype(int)
    df["round"] = df["t"].astype(int)
    return df[["round", "firm_id", "price", "quantity", "profit"]]


def load_metrics(results_dir: str, prefix: str = "") -> pd.DataFrame:
    """ラウンドごとの市場指標 (round, avg_price, collusion_index, total_profit)．

    runvault の `metrics.csv` は long 形式なので `metrics_wide` で横に倒す．時間軸の
    列名は runvault では `step` だが，本モデルの表記は `round` なのでこちら側の呼び名に
    揃える (legacy の wide な metrics.csv はもともと `round` 列を持つ)．
    """
    path = os.path.join(results_dir, "metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"metrics.csv が見つかりません: {path}")
    df = metrics_wide(path)
    if "step" in df.columns and "round" not in df.columns:
        df = df.rename(columns={"step": "round"})
    if prefix:
        cols = {c: c[len(prefix) + 1:] for c in df.columns if c.startswith(prefix + "_")}
        df = df[["round"] + list(cols)].rename(columns=cols)
    return df.dropna(subset=["avg_price"])


def load_benchmarks(results_dir: str, prefix: str = "") -> tuple[float, float]:
    """解析ベンチマークの全社平均 (p_bertrand_mean, p_cartel_mean)．

    runvault では run スコープの指標が正本 (旧 `benchmarks.json`)．企業ごとの値は
    `x.han2023.benchmark` イベントが持つ (時間軸を持たない系列なので指標にできない)．
    """
    legacy = os.path.join(results_dir, "benchmarks.json")
    if os.path.exists(legacy):
        with open(legacy) as f:
            b = json.load(f)
        return float(b.get("p_bertrand_mean", float("nan"))), float(
            b.get("p_cartel_mean", float("nan"))
        )
    scoped = run_scope_metrics(results_dir)
    name = lambda base: f"{prefix}_{base}" if prefix else base  # noqa: E731
    return (
        float(scoped.get(name("p_bertrand_mean"), float("nan"))),
        float(scoped.get(name("p_cartel_mean"), float("nan"))),
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
        "--results-dir",
        "--results_dir",
        default=None,
        help=(
            "runvault の run ディレクトリ．未指定時は runvault に最新の run を聞く "
            "(--experiment sabm --subcommand run --standalone)．"
        ),
    )
    p.add_argument(
        "--results-root",
        "--results_root",
        default="results",
        help="--results-dir 未指定時に runvault が探す results ルート (default: results)",
    )
    p.add_argument(
        "--output-dir",
        "--output_dir",
        default=None,
        help="図の保存先ディレクトリ (default: results/sabm/figures/{run_slug})",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    run_dir = args.results_dir
    if run_dir is None:
        run_dir = runvault_path("sabm", args.results_root, subcommand="run", standalone=True)

    out_dir = args.output_dir if args.output_dir else figures_dir(run_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("=== Han, Wu & Xiao (2023) SABM 価格軌跡 可視化 ===")
    print(f"結果:   {run_dir}")
    print(f"出力先: {out_dir}")
    print("-----------------------------------------")

    rounds = load_rounds(run_dir)
    metrics = load_metrics(run_dir)
    p_bertrand, p_cartel = load_benchmarks(run_dir)
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
