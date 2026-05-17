# AGENTS.md

Guidance for AI coding agents working in this repository. Humans should start at `README.md`.

## What this repo is

`rlm-compile` compiles [RLM](https://github.com/alexzhang13/rlm) JSONL traces into standalone, verifiable Python programs. An RLM trace is already a program (the snippets the model wrote, the sub-LLM responses, the final answer) — `compile.py` extracts it from the runtime so it runs as a normal `.py` with no RLM dependency.

The compiler itself is a single stdlib-only PEP 723 script. **Do not add dependencies to `compile.py`** — being import-free is the central design constraint (the generated artifacts are also stdlib-only).

## Commands

```bash
# Tests (12 unittest cases, no network)
uv run test_compile.py

# Compile a trace -> strict replay artifact + readable recovered companion
uv run compile.py traces/rlm_*.jsonl.gz compiled/out.py

# Replay against the original-shaped context, verify recovered output
python compiled/out.py --context contexts/<name>.txt --verify-trace-final

# Live mode against a new input (needs OPENROUTER_API_KEY or OPENAI_API_KEY)
python compiled/out.py --context new.txt --llm-mode live --model openai/gpt-5-mini

# Audit every LLM judgment call to JSON
python compiled/out.py --context contexts/<name>.txt --llm-audit audit.json

# Run a POC end-to-end (no token cost)
uv run examples/poc_gene_expression_hits.py        # stdlib-only fixture, no LLM
uv run examples/poc_assay_hit_rank.py --gold-only  # validate synthetic data + print gold answer

# Live POC: generate a fresh trace, then compile+replay it
uv run examples/poc_assay_hit_rank.py
TRACE="$(find traces -maxdepth 1 \( -name 'rlm_*.jsonl' -o -name 'rlm_*.jsonl.gz' \) -print | sort | tail -1)"
uv run compile.py "$TRACE" compiled/live.py
uv run examples/poc_assay_hit_rank.py --gold-only --context-out /tmp/ctx.csv
python compiled/live.py --context /tmp/ctx.csv --verify-trace-final
```

Run a single test: `uv run test_compile.py CompileTraceTest.test_recovered_file_is_runnable`.

## Architecture

### Two outputs per compile, two different jobs

`compile_trace()` emits two files by default:

1. **`out.py`** — strict replay/verifier. Source of truth for audits. Replays each trace block in order; raises on any divergence.
2. **`out_recovered.py`** — flatter human-readable view of the recovered logic with diagnostic-only blocks stripped, expected-failure blocks omitted, and recorded LLM prompts/responses shown as comments. Runnable for live reruns; **not** the audit artifact.

Suppress the readable companion with `--no-readable-out`.

### Three LLM modes in the generated artifact

The `run()` function in every generated artifact accepts `llm_mode` ∈ `{replay, live, off}`:

| Mode     | Behavior                                                                                                              |
|----------|-----------------------------------------------------------------------------------------------------------------------|
| `replay` | Returns recorded responses in order. **Verifies exact prompt equality (sha256).** Fails if any recorded call is unused. |
| `live`   | OpenAI-compatible HTTP via stdlib (`urllib`). `OPENROUTER_API_KEY` → OpenRouter defaults; `OPENAI_API_KEY` → OpenAI defaults. Honors matching `*_BASE_URL` / `*_MODEL`. |
| `off`    | Raises if generated code attempts any LLM call.                                                                       |

`rlm_query` is intentionally downgraded to a plain LLM call in `live` mode — generated artifacts replay the discovered strategy, they don't spawn new recursive RLM loops.

### Recovery invariant (load-bearing — don't relax it)

`TRACE_FINAL_ANSWER` in the generated artifact is **verification metadata, not a fallback**. The artifact must recover a value from executed trace code by one of these mechanisms:

- explicit `FINAL(value)` / `FINAL_VAR("name")` call
- assignment to `final_answer` or `final_json` in the trace namespace
- a printed-or-assigned value that matches the recorded final

If no candidate matches in `replay` mode, the artifact raises rather than returning the trace's recorded answer. See `_select_recovered_final()` and `_candidate_final_values()` in `compile.py`. Tests around this invariant live in `test_compile.py::test_live_mode_does_not_require_trace_final_match` and the `_recovered_file_*` group.

### Expected-failure preservation

Trace code blocks that errored before a later recovery block are compiled in as **audited expected failures**. On replay, the artifact requires each such block to fail with the same error class — if it succeeds, or fails differently, the artifact raises rather than silently taking a new path. The recovery path is part of the program, not noise to strip.

### Trace format

JSONL (or `.jsonl.gz`), one iteration record per line, as produced by `rlms.RLMLogger`:

```json
{"type": "iteration", "iteration": 1,
 "code_blocks": [{"code": "...", "result": {"stdout": "...", "stderr": "", "rlm_calls": [...]}}],
 "final_answer": "..."}
```

The compiler only consumes `type=="iteration"` records; all other records are ignored.

## Working in this repo

- **`compile.py` is stdlib-only.** No `pip install`. If you need a new helper, write it inline.
- **Examples are PEP 723 scripts.** Run with `uv run examples/foo.py`, not bare `python` — the dependency block only fires under `uv run`. New examples should follow `examples/poc_assay_hit_rank.py` as the template: `rlms` + `python-dotenv`, a `--gold-only` no-token mode, an `RLMLogger` that writes to `traces/`.
- **New fixtures.** Drop `.jsonl.gz` traces into `traces/`, gold answers into `expected/<hash>.json`, sample inputs into `contexts/`. The hash in the filename is content-derived; preserve it across compile/expected/context for any one example.
- **After touching `compile.py`.** Run `uv run test_compile.py` and also recompile at least one committed trace + replay with `--verify-trace-final` to catch artifact-format regressions the unit tests don't cover.
- **`.gitignore` covers** `.env`, `.venv`, `__pycache__`. The `compiled/` directory is committed — generated artifacts are part of the catalog, not build output.

## Environment gotcha

If `uv run` fails inside the venv with cryptic import errors (typically `ImportError: cannot import name 'Sentinel' from 'typing_extensions'` or similar version-mismatch traces), check for a leaking `PYTHONPATH` from a parent Nix/system shell:

```bash
env | grep ^PYTHONPATH    # leaking site-packages from outside the venv?
env -u PYTHONPATH uv run ...   # workaround
```

The leak shadows uv's managed Python with a different version's site-packages. The real fix is upstream (unset `PYTHONPATH` in your shell init); the per-command workaround is `env -u PYTHONPATH`.
