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

# Run the compiled artifact in replay mode (no API calls)
python compiled/output.py --context input.txt --verify-trace-final

# Run in live mode with a different input
python compiled/output.py --context new_input.txt --llm-mode live --model gpt-5-mini

# Audit which LLM judgment calls were made
python compiled/output.py --context input.txt --llm-audit audit.json
```

The compiler is a single stdlib-only PEP 723 script.
No dependencies to install.

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
| `live` | Make real OpenAI-compatible API calls (`OPENAI_API_KEY`, honors `OPENAI_BASE_URL`) |
| `off` | Raise if executed code attempts an LLM call |

## Related

- [alexzhang13/rlm](https://github.com/alexzhang13/rlm) - the RLM inference engine that produces the traces this compiler consumes
- [Zhang, Kraska, Khattab (2025). Recursive Language Models. arXiv:2512.24601](https://arxiv.org/abs/2512.24601)

## License

BSD 3-Clause.
