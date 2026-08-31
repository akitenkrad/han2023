[English](cli.md) | [日本語](cli.ja.md)

# CLI (`sabm`)

Built with `cargo build --release`; the binary is `sabm`. Four subcommands: `benchmark`, `run`, `sweep`, `reproduce`.

## LLM environment variables

Provider order is **Ollama first → OpenAI fallback** (owned by `socsim-llm`):

- `OLLAMA_HOST` (default `http://localhost:11434`), `OLLAMA_MODEL` (default `llama3.2:latest`)
- `OPENAI_API_KEY`, `OPENAI_MODEL` (default `gpt-4o-mini`; the paper used `gpt-4-0314`)

## `benchmark` — analytic prices only (no LLM, instant)

Computes the Bertrand-equilibrium and cartel/monopoly prices. No live LLM is touched, and no RNG is used, so the run is recorded with `domain = analysis` and no `master_seed`. The all-firm means go to `metrics.csv` (`p_bertrand_mean` / `p_cartel_mean`); the per-firm values go to `x.han2023.benchmark` events, because a series with no time axis cannot be a metric row.

```bash
cargo run --release -- benchmark --a 14 --beta 0.0066667 --d 0.0033333 --c1 2 --c2 2
# → firm 0: p_bertrand ≈ 6, p_cartel = 8   (exact with β=1/150, d=1/300)
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--firms` | 2 | number of firms n |
| `--a` | 14 | inverse-demand intercept a |
| `--beta` | 1/150 | own-price effect β |
| `--d` | 1/300 | cross-price effect d |
| `--d-beta` | — | if given, sets `d = d_beta · β` (overrides `--d`) |
| `--c1` / `--c2` | 2 / 2 | marginal costs (asymmetric supported) |
| `--output-dir` | results | output base directory |

## `run` — LLM-driven repeated pricing game

```bash
cargo run --release -- run --firms 2 --persona active \
    --max-rounds 1000 --temperature 0 --seed 42
```

Adds, on top of the market flags above: `--persona {active,aggressive,none}`, `--communication`, `--max-rounds`, `--reflection-period` (default 20), `--memory-window` (default 20), `--runs`, `--seed`, `--temperature`, `--llm-seed`, `--cache-path`, `--output-dir`, and `--mock` (use a scripted offline client instead of a live LLM — for CI / sandboxes).

`--communication` is a **flag** (present = communication on; absent = the paper's no-communication basic case). When set, a `CommunicationPhase` runs before each pricing decision: every firm emits a short cheap-talk message that is injected into the next round's pricing prompts. Personas (`--persona`) and asymmetric marginal costs (`--c1` / `--c2`) shift the resulting prices — a higher-cost firm settles at a higher price.

Writes a runvault run directory (`results/sabm/run_{timestamp}_{hash}/`): `metrics.csv` (per round `avg_price` / `collusion_index` / `total_profit`; without a step `p_bertrand_mean` / `p_cartel_mean` / `rounds_to_stable` and the LLM call counts), `events.jsonl` (one `observation` per firm and round carrying `price` / `quantity` / `profit`, a `terminal` per firm, and the per-firm `x.han2023.benchmark` rows) and `config.json`. There is no `latest` symlink — resolve the run with `runvault path --experiment sabm --latest --subcommand run --standalone`. With `--runs N` only the last trial is recorded in detail, as before; the earlier ones only warm the cache.

## `sweep` — d/β × firm-count sensitivity

```bash
cargo run --release -- sweep \
    --d-beta-values 0.0,0.25,0.5,0.75,1.0 --firms-values 2,3 \
    --max-rounds 400 --runs 5 --seed 42
```

Flags: `--d-beta-values`, `--firms-values`, `--a`, `--beta`, `--cost`, `--persona`, `--communication`, `--max-rounds`, `--runs`, `--seed`, `--temperature`, `--llm-seed`, `--cache-path`, `--output-dir`, `--mock`. Writes a parent run (`subcommand=sweep`, the grid definition, no `master_seed` — a grid is not one simulation) plus one child per condition (`subcommand=sweep-point`, `parameters` carrying `firms` / `d_beta`). Each child's `events.jsonl` holds one `terminal` row per trial — the columns of the old `sweep_summary.csv` — and its `metrics.csv` holds the condition's aggregate at `scope=run`. Per-trial values cannot live in `metrics.csv`: its primary key is `(run_uid, name, step, step_unit, scope)`, so the trials of one condition would collide.

## `reproduce` — paper Fig.1/2/4 batch

```bash
cargo run --release -- reproduce --seed 42            # live LLM
cargo run --release -- reproduce --seed 42 --mock --quick   # offline, 80 rounds
```

Runs the no-communication baseline (Fig.1) and the communication variant (Fig.2), compares the observed average price and collusion index against the Bertrand(6)/cartel(8) frame, and writes one run (`subcommand=reproduce`). Both scenarios share it, so the scenario name is folded into the metric names (`no_communication_avg_price` …) and into the `unit_id`s (`communication-firm-0` …) — `(step, scope, name)` is the primary key. The anchors' observed values are `anchor_{id}` metrics with `checks_passed` / `checks_total`; the bands and the PASS/off verdicts are `x.han2023.anchor` events, since a direction and a category are not numbers. The bands mix what the paper reports (the Bertrand 6 / cartel 8 frame) with qualitative anchors this replication chose, so none of them are written to `reference.csv`. Flags: the market flags above, `--persona`, `--max-rounds`, `--seed`, `--temperature`, `--llm-seed`, `--cache-path`, `--output-dir`, `--mock`, `--quick`. Render the figures with `uv run sabm-tools reproduce`.

## Offline smoke (no live LLM)

```bash
cargo run --release --example mock_smoke -- results   # dedicated example
cargo run --release -- run --max-rounds 60 --seed 42 --mock   # or --mock on run/sweep
```

---
*This file was generated by Claude Code.*
