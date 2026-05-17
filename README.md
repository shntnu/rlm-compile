# rlm-compile

Compile [RLM](https://github.com/alexzhang13/rlm) JSONL traces into standalone, verifiable Python programs.

An RLM run records everything the model did: the Python snippets it wrote, the sub-LLM calls it made, the final answer it produced.
That trace is already a program - it just happens to be serialized as JSONL and entangled with the RLM runtime.
`compile.py` extracts it into a normal `.py` file that runs without the RLM framework.

## Why

Every existing tool in the agent-tracing space treats traces as telemetry - something to observe, debug, and learn from.
This compiler treats the trace as a source program and compiles it into a different, self-standing representation.

The generated artifact:
- Replays deterministically without API calls (recorded LLM responses are baked in)
- Verifies that executed trace code reproduces the recorded final answer (`--verify-trace-final`)
- Preserves failed iterations as expected audited failures (the recovery path is part of the program)
- Treats LLM calls as explicit semantic judgment boundaries with prompt-equality verification
- Emits a readable `*_recovered.py` companion that strips scaffolding to show just the recovered logic

## Usage

```bash
# Compile a trace
uv run compile.py traces/some_trace.jsonl.gz compiled/output.py
# This also writes compiled/output_recovered.py by default.

# Run the compiled artifact in replay mode (no API calls)
python compiled/output.py --context input.txt --verify-trace-final

# Run in live mode with a different input
python compiled/output.py --context new_input.txt --llm-mode live --model openai/gpt-5-mini

# Audit which LLM judgment calls were made
python compiled/output.py --context input.txt --llm-audit audit.json

# Run the readable recovered program directly
python compiled/output_recovered.py --context new_input.txt --model openai/gpt-5-mini
```

The compiler is a single stdlib-only PEP 723 script.
No dependencies to install.

The main compiled artifact is the strict replay/verifier. The sibling
`*_recovered.py` file is a runnable, flatter view of the recovered logic for
inspection and live reruns; traces that call `FINAL(...)`, `FINAL_VAR(...)`, or
assign `final_answer`/`final_json` are supported.

## Try a Real RLM Run

The smallest live example is `examples/poc_assay_hit_rank.py`. It calls
OpenRouter through `rlms`, writes a real `RLMLogger` trace into `traces/`, and
then that trace can be compiled by this repo.

```bash
# Install uv if needed: https://docs.astral.sh/uv/getting-started/installation/

# Create an OpenRouter key at https://openrouter.ai/keys, then either:
export OPENROUTER_API_KEY="sk-or-v1-..."

# Or use a local .env file. .env is gitignored and loaded by the examples.
cp .env.example .env
$EDITOR .env

# No-token sanity check.
uv run examples/poc_assay_hit_rank.py --gold-only

# Run RLM through OpenRouter. This writes traces/rlm_YYYY-MM-DD_HH-MM-SS_<hash>.jsonl.
uv run examples/poc_assay_hit_rank.py

# Compile and replay the newest trace.
TRACE="$(find traces -maxdepth 1 \( -name 'rlm_*.jsonl' -o -name 'rlm_*.jsonl.gz' \) -print | sort | tail -1)"
uv run compile.py "$TRACE" compiled/my_assay_hit_rank.py
uv run examples/poc_assay_hit_rank.py --gold-only --context-out /tmp/assay_hit_rank_context.csv
python compiled/my_assay_hit_rank.py --context /tmp/assay_hit_rank_context.csv --verify-trace-final
```

## Example Dependencies

The API-backed example scripts declare their dependencies in PEP 723 headers, so
run them with `uv run`:

```bash
uv run examples/poc_assay_hit_rank.py --gold-only
uv run examples/poc_variant_prioritization.py --gold-only
uv run examples/poc_activity_triage.py --gold-only
```

Direct `python examples/...` does not read PEP 723 metadata. If you want that
style, create your own environment and install the same script dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install rlms python-dotenv
python examples/poc_variant_prioritization.py --gold-only
```

## Trace format

The compiler reads JSONL (or `.jsonl.gz`) where each line is an iteration record:

```json
{
  "type": "iteration",
  "iteration": 1,
  "code_blocks": [
    {
      "code": "...",
      "result": {"stdout": "...", "stderr": "", "rlm_calls": []}
    }
  ],
  "final_answer": "..."
}
```

This is what the `rlms` package (`RLMLogger`) produces.

## Examples

The `examples/` directory has POCs that generate traces the compiler can process:

| Example | What it does | Needs API? |
|---|---|---|
| `poc_needle.py` | 5K-line haystack, find `SECRET_NUMBER=<digits>` | Yes (OpenRouter) |
| `poc_assay_hit_rank.py` | Tiny one-table assay-hit ranking that runs RLM and logs a real trace | Yes (OpenRouter) |
| `poc_activity_triage.py` | Synthetic multi-table compound triage with traps and sub-LLM calls | Yes (OpenRouter) |
| `poc_variant_prioritization.py` | Synthetic exome variant prioritization with semantic gene-note filtering | Yes (OpenRouter) |
| `poc_gene_expression_hits.py` | Tiny deterministic fixture (no LLM calls, stdlib-only) | No |

All POCs with API calls also have a `--gold-only` mode that validates the synthetic data and prints the deterministic gold answer without spending tokens.

## Repository layout

```
compile.py              The trace compiler (stdlib-only, PEP 723)
test_compile.py         Unit tests for the compiler
examples/               POCs that produce traces
traces/                 Committed JSONL trace fixtures (.jsonl.gz)
compiled/               Generated replay artifacts
expected/               Expected answers for verification
```

## Committed traces

Pre-committed traces in `traces/` can be compiled and replayed without an API key:

```bash
# Compile and verify the gene-expression fixture
uv run compile.py traces/rlm_2026-05-16_08-12-00_5f4c2a91.jsonl.gz compiled/out.py
python compiled/out.py --context <(uv run examples/poc_gene_expression_hits.py --context-out /dev/stdout 2>/dev/null) --verify-trace-final
```

## LLM modes in generated artifacts

| Mode | Behavior |
|---|---|
| `replay` | Return recorded LLM responses; verify exact prompt match; fail if any recorded call is unused |
| `live` | Make real OpenAI-compatible API calls through OpenRouter (`OPENROUTER_API_KEY`) or OpenAI (`OPENAI_API_KEY`), honoring the matching `*_BASE_URL` and `*_MODEL` overrides |
| `off` | Raise if executed code attempts an LLM call |

## Related

- [alexzhang13/rlm](https://github.com/alexzhang13/rlm) - the RLM inference engine that produces the traces this compiler consumes
- [Zhang, Kraska, Khattab (2025). Recursive Language Models. arXiv:2512.24601](https://arxiv.org/abs/2512.24601)

## License

BSD 3-Clause.
