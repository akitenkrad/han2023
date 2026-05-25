"""sabm-tools show-experiment-settings — 実行結果の設定表示．

results/{timestamp}/config.json (run / benchmark) または
results/{timestamp}_sweep/sweep_config.json (sweep) を読み，実行時に使われた全
パラメータを整形表示する．存在すれば llm_meta.json の LLM 情報
(モデル・endpoint・温度・seed・cache-hit 率) も併せて表示する．
`results/latest` も解決される．

Usage:
    sabm-tools show-experiment-settings
    sabm-tools show-experiment-settings --results-dir results/20260525_120000
    sabm-tools show-experiment-settings --results-dir results/latest --json

results ディレクトリ解決は共有ヘルパ `socsim_tools.io.resolve_results_dir` に委譲する
(出力はバイト等価)．run/benchmark 設定テーブルは複合行 (限界費用 c1/c2) を含み汎用
`render_run_config` の `{label}: {value}` 形に収まらないため sabm 側に残す．sweep 設定
テーブル・解析ベンチマークブロック・LLM メタブロック (llm_meta.json 固有のヘッダと
最終ラウンド/平均価格/CI 行を持つ) と `--json` の `kind`/`benchmarks`/`llm_meta`
フィールドも sabm 固有なので本モジュールに残す．
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from socsim_tools.io import resolve_results_dir


def _find_config_file(results_dir: Path) -> tuple[Path, str]:
    """config.json (run/benchmark) か sweep_config.json (sweep) を探す．"""
    run_cfg = results_dir / "config.json"
    sweep_cfg = results_dir / "sweep_config.json"
    if run_cfg.exists():
        return run_cfg, "run"
    if sweep_cfg.exists():
        return sweep_cfg, "sweep"
    raise FileNotFoundError(
        f"設定ファイルが見つかりません: {results_dir}\n"
        f"  期待されるファイル: config.json (run/benchmark) または sweep_config.json (sweep)"
    )


def _load_llm_meta(results_dir: Path) -> dict | None:
    path = results_dir / "llm_meta.json"
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return None


def _load_benchmarks(results_dir: Path) -> dict | None:
    path = results_dir / "benchmarks.json"
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return None


def render_run_config(cfg: dict, source: Path) -> str:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("実行設定 (run / benchmark)")
    lines.append("=" * 70)
    lines.append(f"設定ファイル: {source}")
    lines.append("-" * 70)
    lines.append(f"企業数 n         : {cfg.get('n_firms', '-')}")
    lines.append(f"逆需要切片 a     : {cfg.get('a', '-')}")
    lines.append(f"自己価格効果 β   : {cfg.get('beta', '-')}")
    lines.append(f"交差価格効果 d   : {cfg.get('d', '-')}")
    lines.append(f"差別化度 d/β     : {cfg.get('d_over_beta', '-')}")
    lines.append(f"α (= a(β-d))     : {cfg.get('alpha', '-')}")
    lines.append(f"b (= β²-d²)      : {cfg.get('b', '-')}")
    lines.append(f"限界費用 c1/c2   : {cfg.get('c1', '-')} / {cfg.get('c2', '-')}")
    lines.append(f"ペルソナ         : {cfg.get('persona', '-')}")
    lines.append(f"会話の有無       : {cfg.get('communication', '-')}")
    lines.append(f"最大ラウンド     : {cfg.get('max_rounds', '-')}")
    lines.append(f"反省周期         : {cfg.get('reflection_period', '-')}")
    lines.append(f"メモリ窓         : {cfg.get('memory_window', '-')}")
    lines.append(f"シード (コア)    : {cfg.get('seed', '-')}")
    lines.append(f"LLM 温度         : {cfg.get('llm_temperature', '-')}")
    lines.append(f"LLM seed         : {cfg.get('llm_seed', '-')}")
    lines.append(f"出力先           : {cfg.get('output_dir', '-')}")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_sweep_config(cfg: dict, source: Path) -> str:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("実行設定 (sweep)")
    lines.append("=" * 70)
    lines.append(f"設定ファイル: {source}")
    lines.append("-" * 70)
    lines.append(f"d/β 候補         : {', '.join(map(str, cfg.get('d_beta_values', [])))}")
    lines.append(f"企業数 候補      : {', '.join(map(str, cfg.get('firms_values', [])))}")
    lines.append(f"逆需要切片 a     : {cfg.get('a', '-')}")
    lines.append(f"自己価格効果 β   : {cfg.get('beta', '-')}")
    lines.append(f"限界費用 (対称)  : {cfg.get('cost', '-')}")
    lines.append(f"ペルソナ         : {cfg.get('persona', '-')}")
    lines.append(f"会話の有無       : {cfg.get('communication', '-')}")
    lines.append(f"最大ラウンド     : {cfg.get('max_rounds', '-')}")
    lines.append(f"試行数 runs      : {cfg.get('runs', '-')}")
    lines.append(f"シード基点       : {cfg.get('seed', '-')}")
    lines.append(f"LLM 温度         : {cfg.get('llm_temperature', '-')}")
    lines.append(f"LLM seed         : {cfg.get('llm_seed', '-')}")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_benchmarks(bench: dict) -> str:
    lines: list[str] = []
    lines.append("")
    lines.append("解析ベンチマーク (benchmarks.json)")
    lines.append("-" * 70)
    lines.append(f"ベルトラン均衡 p^B (平均): {bench.get('p_bertrand_mean', '-')}")
    lines.append(f"カルテル価格   p^M (平均): {bench.get('p_cartel_mean', '-')}")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_llm_meta(meta: dict) -> str:
    lines: list[str] = []
    lines.append("")
    lines.append("LLM 実行メタデータ (llm_meta.json)")
    lines.append("-" * 70)
    lines.append(f"モデル           : {meta.get('llm_model', '-')}")
    lines.append(f"endpoint         : {meta.get('llm_endpoint', '-')}")
    lines.append(f"温度             : {meta.get('llm_temperature', '-')}")
    lines.append(f"seed             : {meta.get('llm_seed', '-')}")
    lines.append(f"呼び出し総数     : {meta.get('total_calls', '-')}")
    lines.append(f"cache-hit        : {meta.get('cache_hits', '-')}")
    rate = meta.get("cache_hit_rate")
    if rate is not None:
        lines.append(f"cache-hit 率     : {rate * 100:.1f}%")
    lines.append(f"最終ラウンド     : {meta.get('final_round', '-')}")
    lines.append(f"最終平均価格     : {meta.get('final_avg_price', '-')}")
    lines.append(f"最終 CI          : {meta.get('final_collusion_index', '-')}")
    note = meta.get("determinism_note")
    if note:
        lines.append("-" * 70)
        lines.append(f"注記: {note}")
    lines.append("=" * 70)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sabm-tools show-experiment-settings",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        "--results_dir",
        default="results/latest",
        help="実行結果ディレクトリ (default: results/latest)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="表ではなく JSON 形式で出力する．",
    )
    args = parser.parse_args(argv)

    results_dir = resolve_results_dir(args.results_dir)
    if not results_dir.exists():
        print(f"エラー: ディレクトリが存在しません: {results_dir}", file=sys.stderr)
        return 1

    cfg_path, kind = _find_config_file(results_dir)
    with cfg_path.open() as f:
        cfg = json.load(f)
    meta = _load_llm_meta(results_dir)
    bench = _load_benchmarks(results_dir)

    if args.json:
        payload = {
            "source": str(cfg_path),
            "kind": kind,
            "config": cfg,
            "benchmarks": bench,
            "llm_meta": meta,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        if kind == "run":
            print(render_run_config(cfg, cfg_path))
        else:
            print(render_sweep_config(cfg, cfg_path))
        if bench is not None:
            print(render_benchmarks(bench))
        if meta is not None:
            print(render_llm_meta(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
