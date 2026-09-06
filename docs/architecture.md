[English](architecture.md) | [日本語](architecture.ja.md)

# Architecture

## Repository layout

```
han2023/
├── Cargo.toml                  # workspace (members = ["simulation"])
├── pyproject.toml              # uv workspace (members = ["tools"])
├── simulation/                 # Rust crate `sabm-simulation` (bin `sabm`)
│   ├── src/
│   │   ├── main.rs             # clap: benchmark / run / sweep / reproduce
│   │   ├── config.rs           # Config (firms, demand params, persona, communication, …)
│   │   ├── demand.rs           # linear demand q_i, profit π_i (deterministic core)
│   │   ├── analytic.rs         # Bertrand-equilibrium & cartel prices (deterministic, exact)
│   │   ├── world.rs            # MarketWorld (WorldState), Persona, RoundRecord
│   │   ├── mechanisms.rs       # 6 mechanisms over the 6-phase loop
│   │   ├── llm.rs              # socsim-llm composition (Ollama→OpenAI + cache)
│   │   ├── prompts.rs          # pricing prompt + communication (cheap-talk) prompt + parse
│   │   ├── metrics.rs          # rounds/metrics rows, convergence detection
│   │   ├── simulation.rs       # init_world + run driver (SimulationBuilder wiring)
│   │   └── lib.rs
│   ├── examples/mock_smoke.rs  # offline (no-LLM) smoke
│   └── tests/integration_test.rs
└── tools/                      # Python `sabm-tools` (module `sabm_tools`)
    └── src/sabm_tools/{cli,visualize,visualize_sweep,show_experiment_settings,reproduce_paper}.py
```

## The model

A differentiated-goods **Bertrand duopoly** (default 2 firms) repeated pricing game. Non-spatial and non-network — the firms interact only through the market (the demand split), so the project depends on `socsim-core` + `socsim-engine` + `socsim-llm` only (no `socsim-grid` / `socsim-net`).

One paper round = one socsim tick. The round is decomposed into six mechanisms mapped onto the six-phase loop:

| Mechanism | Phase | Role |
|-----------|-------|------|
| `MarketEnvironment` | `Environment` | round-start market check / quantity-buffer reset |
| `CommunicationPhase` | `Environment` | **communication variant only** (`--communication`): each firm emits an LLM cheap-talk message before pricing, injected into the next pricing prompt. A complete **no-op** when communication is off (no LLM call, no state change → the default path is bit-identical) |
| `PricingDecision`   | `Decision`    | **LLM lives here**: each firm's next price from bounded memory + persona + benchmark prices (+ rival messages when communication is on); firms set prices *simultaneously* (synchronous bulk assignment); reflection every 20 rounds |
| `MarketClearing`    | `Interaction` | linear demand `q_i = (α − β p_i + d Σ_{j≠i} p_j) / b` |
| `ProfitReward`      | `Reward`      | profit `π_i = (p_i − c_i) q_i`, collusion index, convergence/bounded-oscillation stop via `request_stop()` |
| `MemoryUpdate`      | `PostStep`    | append the round record, truncate to the 20-round window |

The update is **synchronous** (firms simultaneously set prices), so the model is order-independent and the scheduler is `SequentialScheduler` (deterministic).

## Equations

Inverse demand (differentiated duopoly): `p_1 = a − β q_1 − d q_2`, `p_2 = a − β q_2 − d q_1`.

Solved demand (`b = β² − d²`, `α = a(β − d)`): `q_i = (α − β p_i + d p_j) / b`.

Profit: `π_i = (p_i − c_i) q_i`.

Bertrand equilibrium (duopoly): `p_1^B = (d α + β d c_2 + 2 β α + 2 β² c_1) / (4 β² − d²)` (symmetric for firm 2).

Cartel / monopoly: `p_i^M = α / (2 (β − d)) + c_i / 2`.

Collusion index: `CI_i = (p_i − p_i^B) / (p_i^M − p_i^B) ∈ [0, 1]` (0 = Bertrand, 1 = cartel).

**Basic setup** (`a = 14`, `β = 1/150`, `d = 1/300`, `α = 7/150`, `b = 1/30000`, `c_1 = c_2 = 2`, `d/β = 0.5`) gives the demand `q_1 = 1400 − 200 p_1 + 100 p_2` and, exactly, `p_i^B = 6`, `p_i^M = 8`.

## Two-layer determinism

- **socsim core** is bit-deterministic from the seed: initial prices via `derive_seed(root, &[RNG_WORLD_INIT=0])`, the engine RNG via `derive_seed(root, &[RNG_ENGINE=1])`. Market clearing, profit, the analytic benchmarks and memory updates are pure functions of the world state.
- **LLM layer** is pseudo-deterministic via the prompt→response cache, `temperature=0` and a fixed seed; recorded in the `llm` block of `run.json` and in the run-scope `llm_calls` / `llm_cache_hits` / `llm_cache_hit_rate` metrics.

## LLM layer

`llm.rs` is a thin builder over `socsim-llm`: `CachingClient<Box<dyn LlmClient>>` wrapping a `FallbackClient<OllamaClient, OpenAiClient>` (type-erased). Tests inject `socsim_llm::mock::ScriptedClient` through the same alias. Defaults: `OLLAMA_MODEL=llama3.2:latest`, `temperature=0`, fixed seed. This mirrors the li2024 / zhao2024 / chuang2024 siblings; `reqwest` / `sha2` are not used.

## Outputs

A runvault run directory under `results/sabm/`. runvault creates and names it; there is no `latest` symlink. `config.json` is the envelope (conditions under `parameters`) and `run.json` carries identity (repo, git commit, hashes, `master_seed`, the `llm` block, the paper metadata). `metrics.csv` is long (`run_uid, step, step_unit, scope, name, value`): per round (`step_unit=round`, `scope=run`) `avg_price` / `collusion_index` / `total_profit`, and without a step `p_bertrand_mean` / `p_cartel_mean` / `rounds_to_stable` / the LLM call counts. `events.jsonl` carries the per-firm price path — one `observation` per (firm, round) with `price` / `quantity` / `profit`, and a `terminal` row per firm — plus the per-firm analytic benchmarks as `x.han2023.benchmark`. A row keyed by both a round **and** a firm cannot be a metric row: `metrics.csv` has no column for a series id. A sweep is a parent run (the grid) plus one `sweep-point` child per condition, whose `events.jsonl` holds one `terminal` row per trial.

## References

- Han, X., Wu, Z., & Xiao, C. (2023). *"Guinea Pig Trials" Utilizing GPT: A Novel Smart Agent-Based Modeling Approach for Studying Firm Competition and Collusion.* arXiv:2308.10974.
- Calvano, E., Calzolari, G., Denicolò, V., & Pastorello, S. (2020). *Artificial Intelligence, Algorithmic Pricing, and Collusion.* American Economic Review 110(10).
- socsim: https://github.com/akitenkrad/rs-social-simulation-tools
