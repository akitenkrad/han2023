**English** | [日本語](README.ja.md)

# SABM: LLM-Agent Bertrand-Duopoly Firm Competition & Tacit Collusion — Han, Wu & Xiao (2023)

A reimplementation of the Smart Agent-Based Modeling (SABM) framework of Han, Wu & Xiao (2023), ["Guinea Pig Trials Utilizing GPT: A Novel Smart Agent-Based Modeling Approach for Studying Firm Competition and Collusion"](https://arxiv.org/abs/2308.10974) (arXiv:2308.10974). Two LLM "firm" agents play a repeated differentiated-goods **Bertrand duopoly** pricing game. Each round both firms *simultaneously* set a unit price; a linear demand function clears the market and each firm earns a profit. Because the LLM keeps no memory across rounds, a bounded 20-round history is re-injected every round, and a reflection step is interleaved every 20 rounds. The paper's headline finding is that even **without any communication**, two LLM firms reliably reach **tacit collusion**: the price settles *above* the competitive Bertrand equilibrium but *below* the cartel/monopoly price (around 7 in the basic setup where Bertrand = 6 and cartel = 8). The deterministic [socsim](https://github.com/akitenkrad/rs-social-simulation-tools) core handles the analytic benchmarks, market clearing, profit and bounded-memory updates, while the non-deterministic LLM layer is confined to the single pricing-decision mechanism and pseudo-determinised via the `socsim-llm` crate (prompt→response cache + `temperature=0` + fixed seed).

## Two-layer determinism (read this first)

LLM output is **outside** socsim's bit-reproducibility. The design therefore splits into two layers:

- **Deterministic socsim core** — the analytic Bertrand-equilibrium and cartel/monopoly prices (`p^B`, `p^M`), the linear-demand market clearing (`q_i = (α − β p_i + d Σ_{j≠i} p_j) / b`), the profit (`π_i = (p_i − c_i) q_i`), the collusion index (`CI = (p − p^B) / (p^M − p^B)`), the convergence/bounded-oscillation stop detection and the bounded 20-round memory updates. Given a seed these reproduce bit-for-bit (ChaCha20 `SimRng`). The analytic path is **quantitatively exact**: the basic setup gives `p^B = 6`, `p^M = 8`.
- **Non-deterministic LLM layer** — the single `Decision` mechanism (`PricingDecision`), where each firm's next price is decided by the LLM from its bounded history, persona and the benchmark prices. Pseudo-determinised by `socsim-llm`'s `CachingClient` (a `hash(prompt+model)` → response cache), `temperature=0` and a fixed seed. The provider order is **Ollama first → OpenAI fallback** via `socsim-llm`'s `FallbackClient`.

The cache — not the model — is the reproducibility mechanism: a warm cache replays identical responses, so a rerun is free and stable. Each run writes `llm_meta.json` recording the model, endpoint, temperature, seed and cache-hit rate. Because the local default model (`llama3.2`) differs from the paper's `gpt-4-0314`, the LLM-driven reproduction target is **qualitative** (the price settles inside the (Bertrand, cartel) interval, CI ∈ [0.3, 0.8]); the analytic benchmarks are exact.

> This project standardises on the `socsim-llm` crate for the LLM layer; it does **not** use `reqwest` or `sha2` (socsim-llm owns the HTTP transport and the prompt-cache hashing). The model is market-mediated — non-spatial and non-network — so it depends only on `socsim-core` + `socsim-engine` + `socsim-llm` (no `socsim-grid` / `socsim-net`).

## Install & Quick start

```bash
# Build the Rust simulation (fetches socsim incl. socsim-llm with the Ollama+OpenAI backends)
cargo build --release

# === Analytic benchmark only (no LLM, instant): basic setup → p^B=6, p^M=8 ===
cargo run --release -- benchmark --a 14 --beta 0.0066667 --d 0.0033333 --c1 2 --c2 2
# Writes results/{ts}/benchmarks.json

# Make sure a local Ollama is running and a model is pulled, e.g.:
#   ollama pull llama3.2:latest
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=llama3.2:latest
# Optional OpenAI fallback:
#   export OPENAI_API_KEY=sk-...   OPENAI_MODEL=gpt-4o-mini

# Base experiment (no communication, active persona): observe tacit collusion
cargo run --release -- run --firms 2 --persona active --max-rounds 1000 --temperature 0 --seed 42

# Install the Python visualization tools (at the workspace root)
uv sync

# Visualize the most recent run (price trajectory + collusion index, Bertrand/cartel reference lines)
uv run sabm-tools visualize

# Inspect the run's settings and LLM metadata
uv run sabm-tools show-experiment-settings --results-dir results/latest
```

### Offline (no-LLM) smoke

The full round loop, output writers and Python visualization can be exercised without any live LLM via a scripted mock client (used in CI / network-blocked sandboxes):

```bash
# Dedicated example: a tacit-collusion-style scripted policy
cargo run --release --example mock_smoke -- results

# Or pass --mock to run / sweep for the same offline behaviour
cargo run --release -- run --firms 2 --max-rounds 60 --seed 42 --mock
uv run sabm-tools visualize
```

### Sensitivity sweep

```bash
cargo run --release -- sweep \
    --d-beta-values 0.0,0.25,0.5,0.75,1.0 \
    --firms-values 2,3 \
    --max-rounds 400 --runs 5 --seed 42
uv run sabm-tools visualize-sweep
```

### Paper reproduction (Fig.1/2/4)

`reproduce` runs the no-communication baseline (Fig.1) and the communication variant (Fig.2) in one shot, compares the observed average price and collusion index against the Bertrand(6)/cartel(8) frame, and writes a `reproduce_summary.json` (observed avg_price / collusion index vs paper + PASS/off). The Python `reproduce` tool then renders the headline figures into `<results>/figures/`.

```bash
# Live LLM end-to-end (or add --mock for offline); --quick caps rounds at 80
cargo run --release -- reproduce --seed 42
uv run sabm-tools reproduce            # renders fig1/fig2/fig4 + prints the PASS table

# Offline (no-LLM) reproduction in one command
uv run sabm-tools reproduce --run --mock --quick
```

### Communication variant & persona / asymmetric-cost variants

```bash
# "With communication": each round firms exchange a cheap-talk message (CommunicationPhase)
# before the simultaneous pricing decision. --communication is a flag (default = no communication).
cargo run --release -- run --communication --persona active --max-rounds 1000 --seed 42

# Persona selector (active / aggressive / none) and asymmetric marginal costs (c2 ≠ c1):
# a higher-cost firm settles at a higher price (asymmetric cost → asymmetric prices).
cargo run --release -- run --persona aggressive --c1 2 --c2 5 --max-rounds 1000 --seed 42
```

## Scope

This repository implements the full SABM model: the `MarketWorld` + mechanisms over the six-phase loop, the analytic Bertrand/cartel benchmarks, the LLM pricing-decision layer with Ollama→OpenAI fallback + caching, the tacit-collusion / collusion-index metrics, the `sweep` over product-differentiation `d/β` × firm count, the **communication-enabled** variant (a `CommunicationPhase` cheap-talk message exchange before pricing, toggled by `--communication`), per-firm **personas** and **asymmetric marginal costs** (`c2 ≠ c1`), and the one-shot paper reproduction (`reproduce`, Fig.1/2/4 batch with PASS/off anchors). The Python `sabm-tools` provide `visualize` / `visualize-sweep` / `show-experiment-settings` / `reproduce`. The default path (no communication, symmetric costs, `active` persona) is the paper's basic case and is bit-identical to the pre-variant core given a seed.

## Documentation

- [Use cases](docs/usecases.md) — what you can do with this project, with pointers to the rest of the docs.
- [CLI](docs/cli.md) — the Rust CLI: the `benchmark`, `run`, `sweep` and `reproduce` subcommands and their flags, plus the LLM environment variables.
- [Visualization](docs/visualization.md) — the Python `sabm-tools` and how to interpret the outputs.
- [Architecture](docs/architecture.md) — repository structure, the two-layer determinism, the socsim/`socsim-llm` framework, the mechanisms, the demand/Bertrand/cartel equations, the metrics, and references.
- [Reproduction](docs/reproduction.md) — quantitative targets and how the analytic and LLM-driven reproductions are evaluated.

## License

MIT

---
*This file was generated by Claude Code.*
