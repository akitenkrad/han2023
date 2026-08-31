"""sabm-tools show-experiment-settings — 実行結果の設定表示．

runvault の run ディレクトリの config.json (封筒．条件は `parameters` の下) を読み，
実行時に使われた全パラメータを整形表示する．benchmark / run / sweep / sweep-point /
reproduce のどれかは run.json の `subcommand` が答える．解析ベンチマークの全社平均と
LLM の呼び出し数は metrics.csv の run スコープ指標が，モデル・provider・温度は
run.json の `llm` ブロックが持つ．legacy の flat な config.json / sweep_config.json /
benchmarks.json / llm_meta.json もそのまま読める．

run ディレクトリのパスは次で取れる:
    runvault path --experiment sabm --latest --subcommand run --standalone
    runvault path --experiment sabm --latest --subcommand sweep

Usage:
    sabm-tools show-experiment-settings
    sabm-tools show-experiment-settings --results-dir "$(runvault path --experiment sabm --latest --subcommand run --standalone)"
    sabm-tools show-experiment-settings --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from runvault.read import (
    config_parameters,
    load_run_meta,
    run_scope_metrics,
    runvault_path,
)


def resolve_results_dir(path_like: str) -> Path:
    """シンボリックリンク (legacy の `results/latest`) を実体に解決する．"""
    p = Path(path_like)
    if p.is_symlink():
        return Path(os.path.realpath(p))
    return p


def _load_config(results_dir: Path) -> tuple[dict, Path, str]:
    """run ディレクトリの実験条件と，それがどのサブコマンドのものかを返す．

    runvault の config.json は封筒で，条件は `parameters` の下にある．どのサブ
    コマンドかは run.json が答える (`sweep_config.json` はもう書かれない)．
    """
    # 設定が無いことは «まだ sweep_config.json の方かもしれない» という意味なので，
    # ここでは欠落を失敗として扱わない (下で sweep_config.json を見る)．
    params = config_parameters(results_dir, required=False)
    if params is not None:
        meta = load_run_meta(results_dir, required=False)
        if meta is not None:
            kind = str(meta.get("subcommand", "run"))
        else:
            # legacy: 自前で書いていた config.json は "command" を持つ
            kind = "sweep" if params.get("command") == "sweep" else "run"
        return params, results_dir / "config.json", kind

    sweep_cfg = results_dir / "sweep_config.json"
    if sweep_cfg.exists():
        with sweep_cfg.open() as f:
            return json.load(f), sweep_cfg, "sweep"

    raise FileNotFoundError(
        f"設定ファイルが見つかりません: {results_dir}\n"
        f"  期待されるファイル: config.json (runvault の封筒 / legacy の flat) "
        f"または sweep_config.json (legacy の sweep)"
    )


def _load_llm_meta(results_dir: Path) -> dict | None:
    """LLM の同一系情報を run.json と metrics.csv から組む (legacy は llm_meta.json)．"""
    legacy = results_dir / "llm_meta.json"
    if legacy.exists():
        with legacy.open() as f:
            return json.load(f)
    meta = load_run_meta(results_dir, required=False)
    if meta is None:
        return None
    llm = meta.get("llm")
    if llm is None:
        return None
    scoped = run_scope_metrics(results_dir)
    params = config_parameters(results_dir, required=False) or {}
    summary = {
        "llm_model": llm.get("model_snapshot", "-"),
        "llm_provider": llm.get("provider", "-"),
        "llm_temperature": llm.get("temperature", "-"),
        "llm_seed": params.get("llm_seed", "-"),
    }
    for key, name in (
        ("total_calls", "llm_calls"),
        ("cache_hits", "llm_cache_hits"),
        ("cache_hit_rate", "llm_cache_hit_rate"),
    ):
        if name in scoped:
            summary[key] = scoped[name] if key == "cache_hit_rate" else int(scoped[name])
    return summary


def _load_benchmarks(results_dir: Path) -> dict | None:
    """解析ベンチマークの全社平均 (legacy は benchmarks.json)．"""
    legacy = results_dir / "benchmarks.json"
    if legacy.exists():
        with legacy.open() as f:
            return json.load(f)
    if load_run_meta(results_dir, required=False) is None:
        return None
    scoped = run_scope_metrics(results_dir)
    if "p_bertrand_mean" not in scoped:
        return None
    return {
        "p_bertrand_mean": scoped["p_bertrand_mean"],
        "p_cartel_mean": scoped.get("p_cartel_mean", "-"),
    }


def render_run_config(cfg: dict, source: Path) -> str:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("実行設定 (run / benchmark / reproduce)")
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
    # 出力先は run ディレクトリそのものなので条件には含まれない (legacy のみ持つ)．
    if cfg.get("output_dir") is not None:
        lines.append(f"出力先           : {cfg['output_dir']}")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_sweep_point_config(cfg: dict, source: Path) -> str:
    """スイープの子 run (条件 1 点 × runs 試行) の条件．"""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("実行設定 (sweep-point — スイープの条件 1 点)")
    lines.append("=" * 70)
    lines.append(f"設定ファイル: {source}")
    lines.append("-" * 70)
    lines.append(f"企業数 n         : {cfg.get('firms', '-')}")
    lines.append(f"差別化度 d/β     : {cfg.get('d_beta', '-')}")
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
    lines.append("解析ベンチマーク (全社平均)")
    lines.append("-" * 70)
    lines.append(f"ベルトラン均衡 p^B (平均): {bench.get('p_bertrand_mean', '-')}")
    lines.append(f"カルテル価格   p^M (平均): {bench.get('p_cartel_mean', '-')}")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_llm_meta(meta: dict) -> str:
    lines: list[str] = []
    lines.append("")
    lines.append("LLM 実行メタデータ")
    lines.append("-" * 70)
    lines.append(f"モデル           : {meta.get('llm_model', '-')}")
    if "llm_endpoint" in meta:
        lines.append(f"endpoint         : {meta['llm_endpoint']}")
    else:
        lines.append(f"provider         : {meta.get('llm_provider', '-')}")
    lines.append(f"温度             : {meta.get('llm_temperature', '-')}")
    lines.append(f"seed             : {meta.get('llm_seed', '-')}")
    lines.append(f"呼び出し総数     : {meta.get('total_calls', '-')}")
    lines.append(f"cache-hit        : {meta.get('cache_hits', '-')}")
    rate = meta.get("cache_hit_rate")
    if rate is not None:
        lines.append(f"cache-hit 率     : {rate * 100:.1f}%")
    # 最終ラウンド / 平均価格 / CI は legacy の llm_meta.json だけが持つ．runvault では
    # ラウンドごとの値が metrics.csv に，終端が events.jsonl にあるので重複させない．
    for key, label in (
        ("final_round", "最終ラウンド     "),
        ("final_avg_price", "最終平均価格     "),
        ("final_collusion_index", "最終 CI          "),
    ):
        if key in meta:
            lines.append(f"{label}: {meta[key]}")
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
        default=None,
        help=(
            "run ディレクトリ．未指定時は runvault に最新の run を聞く "
            "(--experiment sabm --subcommand run --standalone)．"
        ),
    )
    parser.add_argument(
        "--results-root",
        "--results_root",
        default="results",
        help="--results-dir 未指定時に runvault が探す results ルート (default: results)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="表ではなく JSON 形式で出力する．",
    )
    args = parser.parse_args(argv)

    if args.results_dir is None:
        results_dir = Path(
            runvault_path("sabm", args.results_root, subcommand="run", standalone=True)
        )
    else:
        results_dir = resolve_results_dir(args.results_dir)
    if not results_dir.exists():
        print(f"エラー: ディレクトリが存在しません: {results_dir}", file=sys.stderr)
        return 1

    try:
        cfg, cfg_path, kind = _load_config(results_dir)
    except FileNotFoundError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
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
        if kind == "sweep":
            print(render_sweep_config(cfg, cfg_path))
        elif kind == "sweep-point":
            print(render_sweep_point_config(cfg, cfg_path))
        else:
            print(render_run_config(cfg, cfg_path))
        if bench is not None:
            print(render_benchmarks(bench))
        if meta is not None:
            print(render_llm_meta(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
