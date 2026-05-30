#!/usr/bin/env python3
"""reproduce_paper.py — Han, Wu & Xiao (2023) SABM 論文 Fig.1/2/4 一括再現レポート．

Rust の `sabm reproduce` が書き出す `reproduce_summary.json` (会話なし / 会話あり 2
シナリオの観測平均価格・collusion index と Bertrand(6)/cartel(8) フレームに対する
PASS/off アンカー) を読み，論文の headline 図を再現する:

  - Fig.1 (会話なし): 価格軌跡が (Bertrand, cartel) 区間へ収束 (~7 の暗黙の共謀)．
  - Fig.2 (会話あり): 同じく価格軌跡 (会話は共謀を強める)．
  - Fig.4         : 会話なし / 会話ありの collusion index 時系列の比較．

`--run` を付けると先に Rust バイナリ (`sabm reproduce`) を実行して最新結果を作る．
`--mock` / `--quick` はそのまま Rust バイナリへ渡す (オフライン・短縮再現)．

Usage:
    sabm-tools reproduce
    sabm-tools reproduce --run --mock --quick
    sabm-tools reproduce --results-dir results/20260530_000000_reproduce
    sabm-tools reproduce --json

Outputs:
    <results_dir>/figures/
    ├── fig1_no_communication.png   ← 会話なしの価格軌跡 (Fig.1 風)
    ├── fig2_communication.png      ← 会話ありの価格軌跡 (Fig.2 風)
    └── fig4_collusion_compare.png  ← 会話なし/あり CI 時系列の比較 (Fig.4 風)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from socsim_tools.io import resolve_results_dir

# --------------------------------------------------------------------------- #
# 日本語フォント・カラー設定 (visualize.py と統一)．
# --------------------------------------------------------------------------- #
plt.rcParams["font.family"] = "Hiragino Sans"

COLOR_BG = "#FAFAF8"
COLOR_AVG = "#1565C0"
COLOR_BERTRAND = "#2E7D32"
COLOR_CARTEL = "#C62828"
COLOR_NOCOMM = "#1565C0"
COLOR_COMM = "#C62828"
FIRM_COLORS = ["#FF9800", "#03A9F4", "#8BC34A", "#E91E63", "#795548"]


def _run_binary(seed: int, mock: bool, quick: bool) -> None:
    """cargo run --release -- reproduce を実行して最新結果を生成する．"""
    cmd = ["cargo", "run", "--release", "--", "reproduce", "--seed", str(seed)]
    if mock:
        cmd.append("--mock")
    if quick:
        cmd.append("--quick")
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _load_summary(results_dir: Path) -> dict:
    path = results_dir / "reproduce_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"reproduce_summary.json が見つかりません: {path}\n"
            f"  先に `sabm-tools reproduce --run` を実行してください．"
        )
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_scenario(results_dir: Path, subdir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """シナリオサブディレクトリの rounds.csv / metrics.csv を読む．"""
    base = results_dir / subdir
    rounds = pd.read_csv(base / "rounds.csv")
    metrics = pd.read_csv(base / "metrics.csv")
    return rounds, metrics


def _save_trajectory(
    rounds: pd.DataFrame,
    metrics: pd.DataFrame,
    p_bertrand: float,
    p_cartel: float,
    title: str,
    out_path: Path,
) -> None:
    """各社価格 + 全社平均 + Bertrand/cartel 基準線 (Fig.1/2 風)．"""
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    fig.suptitle(title, fontsize=14)

    lo, hi = sorted((p_bertrand, p_cartel))
    ax.axhspan(lo, hi, color="#FFC107", alpha=0.08)
    for i, (firm_id, sub) in enumerate(rounds.groupby("firm_id")):
        ax.plot(
            sub["round"],
            sub["price"],
            color=FIRM_COLORS[i % len(FIRM_COLORS)],
            lw=1.2,
            alpha=0.7,
            label=f"firm {firm_id}",
        )
    ax.plot(metrics["round"], metrics["avg_price"], color=COLOR_AVG, lw=2.4, label="全社平均価格")
    ax.axhline(
        p_bertrand,
        color=COLOR_BERTRAND,
        lw=1.6,
        linestyle="--",
        label=f"ベルトラン均衡 p^B={p_bertrand:.2f}",
    )
    ax.axhline(
        p_cartel, color=COLOR_CARTEL, lw=1.6, linestyle="--", label=f"カルテル p^M={p_cartel:.2f}"
    )
    ax.set_xlabel("ラウンド t")
    ax.set_ylabel("価格 p")
    ax.set_title("価格は (ベルトラン, カルテル) 区間へ収束 (暗黙の共謀)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def _save_collusion_compare(
    metrics_nocomm: pd.DataFrame,
    metrics_comm: pd.DataFrame,
    out_path: Path,
) -> None:
    """会話なし / 会話あり の collusion index 時系列を重ねる (Fig.4 風)．"""
    fig, ax = plt.subplots(figsize=(11, 5), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    fig.suptitle("Han, Wu & Xiao (2023) SABM — Fig.4: 会話なし / 会話あり の比較", fontsize=14)

    ax.plot(
        metrics_nocomm["round"],
        metrics_nocomm["collusion_index"],
        color=COLOR_NOCOMM,
        lw=2.2,
        label="会話なし (Fig.1)",
    )
    ax.plot(
        metrics_comm["round"],
        metrics_comm["collusion_index"],
        color=COLOR_COMM,
        lw=2.2,
        label="会話あり (Fig.2)",
    )
    ax.axhline(0.0, color=COLOR_BERTRAND, lw=1.0, linestyle="--", label="0 = ベルトラン均衡")
    ax.axhline(1.0, color=COLOR_CARTEL, lw=1.0, linestyle="--", label="1 = カルテル")
    ax.axhspan(0.3, 0.8, color="#FFC107", alpha=0.10, label="論文の暗黙共謀帯 (0.3-0.8)")
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("ラウンド t")
    ax.set_ylabel("collusion index CI")
    ax.set_title("CI = (p - p^B) / (p^M - p^B); 会話は共謀を強める")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def _print_table(summary: dict) -> None:
    print("=" * 78)
    print("Han, Wu & Xiao (2023) SABM — Fig.1/2/4 再現レポート")
    print(f"  paper : {summary.get('paper', '')}")
    print(
        f"  frame : p_bertrand={summary['p_bertrand']:.3f} "
        f"p_cartel={summary['p_cartel']:.3f} | mock={summary['mock']} quick={summary['quick']}"
    )
    print("=" * 78)
    for s in summary.get("scenarios", []):
        stable = s["rounds_to_stable"] if s["rounds_to_stable"] >= 0 else "未達"
        print(
            f"  [{s['name']:<16}] avg_price={s['observed_avg_price']:.3f} "
            f"CI={s['observed_collusion_index']:.3f} "
            f"安定到達={stable} (round {s['final_round']})"
        )
    print("-" * 78)
    n_pass = 0
    for a in summary.get("anchors", []):
        hi = a["target_hi"]
        hi_str = "∞" if hi is None or hi == float("inf") or hi > 1e30 else f"{hi:.2f}"
        status = "PASS" if a["pass"] else "OFF "
        if a["pass"]:
            n_pass += 1
        print(
            f"[{status}] {a['name']:<48} "
            f"obs={a['observed']:.4f} target=[{a['target_lo']:.2f},{hi_str}] "
            f"paper={a['paper_value']}"
        )
    print("-" * 78)
    print(f"{n_pass}/{len(summary.get('anchors', []))} アンカーが in-band")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sabm-tools reproduce",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", "--results_dir", default=None)
    parser.add_argument("--output-dir", "--output_dir", default=None, help="図の保存先 (既定: <results>/figures)")
    parser.add_argument("--run", action="store_true", help="先に Rust バイナリ (sabm reproduce) を実行する．")
    parser.add_argument("--mock", action="store_true", help="--run 時に scripted mock を使う (オフライン)．")
    parser.add_argument("--quick", action="store_true", help="--run 時に短縮再現 (max_rounds=80)．")
    parser.add_argument("--seed", type=int, default=42, help="--run 時のシード基点．")
    parser.add_argument("--json", action="store_true", help="サマリを JSON で出力する (図は生成しない)．")
    args = parser.parse_args(argv)

    if args.run:
        _run_binary(args.seed, args.mock, args.quick)

    results_dir = resolve_results_dir(args.results_dir)
    try:
        summary = _load_summary(results_dir)
    except FileNotFoundError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    _print_table(summary)

    out_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("-" * 78)
    print(f"図の出力先: {out_dir}")

    p_bertrand = float(summary["p_bertrand"])
    p_cartel = float(summary["p_cartel"])
    scen_by_name = {s["name"]: s for s in summary.get("scenarios", [])}

    metrics_by_name: dict[str, pd.DataFrame] = {}
    for name, fig_name, title in [
        ("no_communication", "fig1_no_communication.png", "Han, Wu & Xiao (2023) SABM — Fig.1: 会話なしの価格軌跡"),
        ("communication", "fig2_communication.png", "Han, Wu & Xiao (2023) SABM — Fig.2: 会話ありの価格軌跡"),
    ]:
        if name not in scen_by_name:
            continue
        rounds, metrics = _load_scenario(results_dir, scen_by_name[name]["results_subdir"])
        metrics_by_name[name] = metrics
        _save_trajectory(rounds, metrics, p_bertrand, p_cartel, title, out_dir / fig_name)

    if "no_communication" in metrics_by_name and "communication" in metrics_by_name:
        _save_collusion_compare(
            metrics_by_name["no_communication"],
            metrics_by_name["communication"],
            out_dir / "fig4_collusion_compare.png",
        )

    print("-" * 78)
    print("完了．出力ファイル一覧:")
    for f in sorted(out_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:35s} ({size_kb:6.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
