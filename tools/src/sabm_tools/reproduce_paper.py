#!/usr/bin/env python3
"""reproduce_paper.py — Han, Wu & Xiao (2023) SABM 論文 Fig.1/2/4 一括再現レポート．

Rust の `sabm reproduce` が書いた runvault の run ディレクトリを読み，論文の
headline 図を再現する．会話なし / 会話あり 2 シナリオのラウンドごとの指標は
`metrics.csv` の `<シナリオ>_avg_price` などが，企業ごとの価格軌跡は `events.jsonl` の
`observation` 行が，Bertrand(6)/cartel(8) フレームに対する PASS/off アンカーは
`x.han2023.anchor` 行が持つ (旧 `reproduce_summary.json` とシナリオ別サブ
ディレクトリに相当する):

  - Fig.1 (会話なし): 価格軌跡が (Bertrand, cartel) 区間へ収束 (~7 の暗黙の共謀)．
  - Fig.2 (会話あり): 同じく価格軌跡 (会話は共謀を強める)．
  - Fig.4         : 会話なし / 会話ありの collusion index 時系列の比較．

`--run` を付けると先に Rust バイナリ (`sabm reproduce`) を実行して最新結果を作る．
`--mock` / `--quick` はそのまま Rust バイナリへ渡す (オフライン・短縮再現)．

Usage:
    sabm-tools reproduce
    sabm-tools reproduce --run --mock --quick
    sabm-tools reproduce --results-dir "$(runvault path --experiment sabm --latest --subcommand reproduce)"
    sabm-tools reproduce --json

Outputs:
    results/sabm/figures/<run_slug>/
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
from runvault.read import config_parameters, events_table, figures_dir, runvault_path

from sabm_tools.visualize import load_metrics, load_rounds

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


SCENARIOS = ["no_communication", "communication"]


def _load_summary(results_dir: Path) -> dict:
    """run ディレクトリから旧 `reproduce_summary.json` と同じ形の要約を組み直す．

    runvault はこの要約をディスク上に持たない．シナリオごとの観測値は `metrics.csv` の
    run スコープ指標が，アンカーの帯と PASS/off は `events.jsonl` の
    `x.han2023.anchor` 行が持つので，そこから復元する．legacy の run ディレクトリは
    `reproduce_summary.json` をそのまま読む．
    """
    legacy = results_dir / "reproduce_summary.json"
    if legacy.exists():
        with legacy.open(encoding="utf-8") as f:
            return json.load(f)

    from runvault.read import load_run_meta, run_scope_metrics

    scoped = run_scope_metrics(results_dir)
    params = config_parameters(results_dir)
    # 何を再現しようとしているかは run.json の research ブロックが持つ
    # (旧 reproduce_summary.json は同じ文言を自前で埋めていた)．
    research = (load_run_meta(results_dir) or {}).get("research") or {}
    targets = research.get("targets") or [{}]
    paper = " — ".join(
        x for x in (research.get("work", {}).get("title"), targets[0].get("label")) if x
    )

    scenarios = []
    for name in params.get("scenarios", SCENARIOS):
        metrics = load_metrics(str(results_dir), prefix=name)
        if metrics.empty:
            continue
        last = metrics.iloc[-1]
        scenarios.append({
            "name": name,
            "communication": name == "communication",
            "observed_avg_price": float(last["avg_price"]),
            "observed_collusion_index": float(last["collusion_index"]),
            # 未達は指標そのものを書かない (旧 JSON は -1 を書いていた)
            "rounds_to_stable": int(scoped.get(f"{name}_rounds_to_stable", -1)),
            # 旧 final_round はエンジンのクロック．ラウンド番号 + 1 に当たる
            "final_round": int(last["round"]) + 1,
            "p_bertrand": scoped[f"{name}_p_bertrand_mean"],
            "p_cartel": scoped[f"{name}_p_cartel_mean"],
            "results_subdir": name,
        })

    anchors = []
    for _, ev in events_table(results_dir, kind="x.han2023.anchor").iterrows():
        hi = ev.get("target_hi")
        anchors.append({
            "name": ev["label"],
            "paper_value": ev["paper_value"],
            "observed": float(ev["observed"]),
            "target_lo": float(ev["target_lo"]),
            # 上限なしのアンカーは列ごと落としてあるので None に戻す
            "target_hi": None if hi is None or pd.isna(hi) else float(hi),
            "pass": bool(ev["pass"]),
        })

    base = scenarios[0] if scenarios else {"p_bertrand": float("nan"), "p_cartel": float("nan")}
    return {
        "command": "reproduce",
        "paper": paper,
        "p_bertrand": base["p_bertrand"],
        "p_cartel": base["p_cartel"],
        "mock": bool(params.get("mock", False)),
        "quick": bool(params.get("quick", False)),
        "scenarios": scenarios,
        "anchors": anchors,
        "n_pass": int(scoped.get("checks_passed", sum(a["pass"] for a in anchors))),
        "n_anchors": int(scoped.get("checks_total", len(anchors))),
    }


def _load_scenario(results_dir: Path, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1 シナリオの企業ごとの軌跡とラウンド指標を読む．

    runvault では 2 シナリオが 1 本の run に同居するので，シナリオ名が指標名と
    `unit_id` の接頭辞になっている．legacy はシナリオ名のサブディレクトリを読む．
    """
    legacy = results_dir / name
    if (legacy / "rounds.csv").exists():
        return pd.read_csv(legacy / "rounds.csv"), pd.read_csv(legacy / "metrics.csv")
    return (
        load_rounds(str(results_dir), prefix=name),
        load_metrics(str(results_dir), prefix=name),
    )


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
    parser.add_argument("--results-dir", "--results_dir", default=None,
                        help=("reproduce の run ディレクトリ．未指定時は runvault に"
                              "最新の run を聞く (--experiment sabm --subcommand reproduce)．"))
    parser.add_argument("--results-root", "--results_root", default="results",
                        help="--results-dir 未指定時に runvault が探す results ルート (既定: results)")
    parser.add_argument("--output-dir", "--output_dir", default=None,
                        help="図の保存先 (既定: results/sabm/figures/<run_slug>)")
    parser.add_argument("--run", action="store_true", help="先に Rust バイナリ (sabm reproduce) を実行する．")
    parser.add_argument("--mock", action="store_true", help="--run 時に scripted mock を使う (オフライン)．")
    parser.add_argument("--quick", action="store_true", help="--run 時に短縮再現 (max_rounds=80)．")
    parser.add_argument("--seed", type=int, default=42, help="--run 時のシード基点．")
    parser.add_argument("--json", action="store_true", help="サマリを JSON で出力する (図は生成しない)．")
    args = parser.parse_args(argv)

    if args.run:
        _run_binary(args.seed, args.mock, args.quick)

    if args.results_dir is None:
        results_dir = Path(
            runvault_path("sabm", args.results_root, subcommand="reproduce")
        )
    else:
        results_dir = Path(args.results_dir)
    try:
        summary = _load_summary(results_dir)
    except FileNotFoundError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    _print_table(summary)

    out_dir = Path(args.output_dir) if args.output_dir else Path(figures_dir(results_dir))
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
