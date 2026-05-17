#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Compile an RLM JSONL trace into a standalone replay artifact.

This is a standalone stdlib-only script. It can be saved as `compile.py` and run
directly with either:

    uv run compile.py trace.jsonl replay.py
    python compile.py trace.jsonl replay.py

An RLM run records the root model's Python snippets and any sub-LLM call
responses in a JSONL trace. Those snippets are already the discovered program.
This compiler packages them into a normal `.py` file that does not require the
RLM runtime.

The generated replay artifact includes:
  - `run(context, llm_mode=...)` for library use
  - A CLI that accepts `--context`
  - `--verify-trace-final` to prove executed trace code reproduces the
    recorded final answer
  - `--llm-audit` to write every LLM judgment call as JSON
  - Compatibility shims for llm_query, rlm_query, their batched variants,
    FINAL(...), FINAL_VAR(...), and answer["ready"]
  - A stdlib-only OpenAI-compatible HTTP shim for live LLM calls

By default the compiler also writes a sibling `*_recovered.py` file. That file
is a compact, human-readable view of the successful trace code path, with
recorded LLM judgment prompts and responses shown as comments. The replay
artifact remains the strict audit/source-of-truth output.

The generated artifact has three LLM modes:
  - replay: return recorded LLM responses from the trace, in order
  - live: make ordinary OpenAI-compatible chat-completion calls; no RLM loops
  - off: raise if the replayed code attempts an LLM call

Caveats:
  - Trace code blocks are preserved verbatim, including blocks that errored
    before a later recovery block.
  - The recorded trace final answer is verification metadata, not a fallback
    runtime answer. Generated artifacts must recover a final value from executed
    trace code, an explicit FINAL/FINAL_VAR signal, or a printed/assigned value
    that matches the trace final.
  - Code blocks that errored in the trace are replayed as audited expected
    failures. If an expected-failure block succeeds, or fails differently, the
    artifact raises instead of silently taking a new path.
  - Replay mode requires the generated code to ask the exact recorded prompts
    and consume all recorded calls. This keeps semantic judgment boundaries
    auditable instead of treating recorded responses as anonymous fixtures.
  - Replay mode is deterministic but only semantically valid for the original
    input that produced the trace.
  - Live mode can try new inputs, but the recovered snippets may have been
    specialized to the original context.
  - The generated artifact requires a context string; traces usually do not
    embed the original input.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import json
from pathlib import Path
from typing import Any


__version__ = "0.4.0"


HELP_EPILOG = """\
Examples:
  uv run compile.py logs/rlm_2026-05-15_21-22-41_d83aed93.jsonl out.py
  python compile.py logs/rlm_2026-05-15_21-22-41_d83aed93.jsonl.gz out.py
  python compile.py logs/rlm_2026-05-15_21-22-41_d83aed93.jsonl.gz out.py --readable-out out_recovered.py
  python out.py --context haystack.txt
  python out.py --context haystack.txt --llm-mode live --model gpt-5-mini
  python out.py --context haystack.txt --quiet --no-code-output
  python out.py --context haystack.txt --verify-trace-final
  python out.py --context haystack.txt --llm-audit llm_audit.json

Generated artifact API:
  import out
  answer = out.run(context_string)
  answer = out.run(context_string, llm_mode="live", model="gpt-5-mini")

Generated artifact LLM modes:
  replay  Return recorded llm_query/rlm_query responses from the trace.
          This is deterministic and matches the original run, but recorded
          responses are stale if you provide a different context. Replay also
          checks exact prompt equality and fails if any recorded call is unused.
  live    Make ordinary OpenAI-compatible chat-completion calls.
          Requires OPENAI_API_KEY. Honors OPENAI_BASE_URL and OPENAI_MODEL.
          rlm_query is downgraded to a plain LLM call.
  off     Raise if generated code attempts an LLM call.

The compiler is intentionally small: it reads each JSONL iteration, extracts
the emitted code blocks and recorded result.rlm_calls, then writes a standalone
stdlib-only Python file with replay/live/off runtime shims.
"""


ARTIFACT_HEADER = r'''#!/usr/bin/env python3
"""Replayable Python artifact compiled from an RLM trace.

This file contains the Python code blocks emitted by the root model, plus small
compatibility shims for final-answer signaling and plain LLM calls.

CLI:
    python {out_name} --context context.txt
    python {out_name} --context context.txt --llm-mode live --model gpt-5-mini

Python:
    import {module_name}
    answer = {module_name}.run(context_string)
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any


SOURCE_LOG = {source_log!r}
TRACE_FINAL_ANSWER = {expected_final!r}
TRACE_BLOCKS: list[tuple[int, str, str | None]] = {trace_blocks!r}
RECORDED_LLM_CALLS: list[dict[str, Any]] = {recorded_llm_calls!r}


class _FinalAnswer(Exception):
    def __init__(self, value: Any):
        self.value = value
        super().__init__(str(value))


def _openai_chat_completion(prompt: str, model: str | None = None) -> str:
    """Minimal OpenAI-compatible chat-completions call using only stdlib."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for --llm-mode live")

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model_name = model or os.environ.get("OPENAI_MODEL", "gpt-5-mini")
    payload = json.dumps({{
        "model": model_name,
        "messages": [{{"role": "user", "content": prompt}}],
    }}).encode("utf-8")
    request = urllib.request.Request(
        f"{{base_url}}/chat/completions",
        data=payload,
        headers={{
            "Authorization": f"Bearer {{api_key}}",
            "Content-Type": "application/json",
        }},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM HTTP error {{exc.code}}: {{body}}") from exc

    return data["choices"][0]["message"]["content"]


def _canonical(value: Any) -> str:
    text = str(value).strip()
    try:
        return json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"))
    except (TypeError, json.JSONDecodeError):
        return text


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _matches_trace_final(value: Any) -> bool:
    return TRACE_FINAL_ANSWER is not None and _canonical(value) == _canonical(TRACE_FINAL_ANSWER)


def _candidate_final_values(namespace: dict[str, Any], stdout_text: str) -> list[tuple[str, Any]]:
    candidates: list[tuple[str, Any]] = []

    answer = namespace.get("answer")
    if isinstance(answer, dict) and answer.get("ready"):
        candidates.append(("answer.content", answer.get("content", "")))

    for name in ("final_answer", "final_json"):
        if name in namespace:
            candidates.append((name, namespace[name]))

    for line in reversed(stdout_text.splitlines()):
        stripped = line.strip()
        if stripped:
            candidates.append(("stdout", stripped))

    return candidates


def _select_recovered_final(
    namespace: dict[str, Any],
    stdout_text: str,
    *,
    require_trace_final_match: bool,
) -> str:
    candidates = _candidate_final_values(namespace, stdout_text)
    if require_trace_final_match and TRACE_FINAL_ANSWER is not None:
        for _, value in candidates:
            if _matches_trace_final(value):
                return str(value)
        sources = ", ".join(source for source, _ in candidates) or "none"
        raise RuntimeError(
            "Trace completed without a recovered final value matching "
            f"TRACE_FINAL_ANSWER; candidate sources: {{sources}}"
        )

    if candidates:
        return str(candidates[0][1])
    raise RuntimeError("Trace completed without a recovered final value")


def _format_exception(exc: BaseException) -> str:
    return f"{{type(exc).__name__}}: {{exc}}"


def _is_expected_error(exc: BaseException, expected_error: str) -> bool:
    return _format_exception(exc) in expected_error


def _record_llm_audit(
    audit_log: list[dict[str, Any]] | None,
    *,
    index: int,
    mode: str,
    prompt: str,
    response: str | None = None,
    recorded: dict[str, Any] | None = None,
) -> None:
    if audit_log is None:
        return
    event: dict[str, Any] = {{
        "index": index,
        "mode": mode,
        "prompt_sha256": _sha256_text(prompt),
        "prompt": prompt,
    }}
    if response is not None:
        event["response_sha256"] = _sha256_text(response)
        event["response"] = response
    if recorded is not None:
        event["recorded_prompt_sha256"] = _sha256_text(str(recorded.get("prompt", "")))
        event["root_model"] = recorded.get("root_model")
        if recorded.get("usage_summary") is not None:
            event["usage_summary"] = recorded.get("usage_summary")
        if recorded.get("execution_time") is not None:
            event["execution_time"] = recorded.get("execution_time")
    audit_log.append(event)


def run(
    context: str,
    *,
    llm_mode: str = "replay",
    model: str | None = None,
    verbose: bool = True,
    echo_code_output: bool = True,
    audit_log: list[dict[str, Any]] | None = None,
) -> str:
    """Run the compiled trajectory against a context string.

    llm_mode:
        replay - return recorded llm_query/rlm_query responses from the trace.
        live   - make plain OpenAI-compatible LLM calls.
        off    - fail if generated code tries to call an LLM.

    rlm_query is intentionally downgraded to the same plain-call behavior as
    llm_query. This artifact replays the discovered strategy; it does not spawn
    recursive RLM loops.
    """
    recorded_calls = deque(RECORDED_LLM_CALLS)
    llm_call_count = 0
    run_model = model

    def finish(value: Any) -> str:
        if llm_mode == "replay" and recorded_calls:
            raise RuntimeError(
                "Trace replay finished with "
                f"{{len(recorded_calls)}} unused recorded LLM response(s)"
            )
        return str(value)

    def llm_query(prompt: str, model: str | None = None) -> str:
        nonlocal llm_call_count
        prompt = str(prompt)
        llm_call_count += 1
        if llm_mode == "replay":
            if not recorded_calls:
                raise RuntimeError("Trace has no recorded LLM response left to replay")
            recorded = recorded_calls.popleft()
            recorded_prompt = str(recorded.get("prompt", ""))
            if prompt != recorded_prompt:
                raise RuntimeError(
                    "Recorded LLM prompt mismatch at call "
                    f"{{llm_call_count}}: expected sha256={{_sha256_text(recorded_prompt)}}, "
                    f"got sha256={{_sha256_text(prompt)}}"
                )
            response = str(recorded.get("response", ""))
            _record_llm_audit(
                audit_log,
                index=llm_call_count,
                mode=llm_mode,
                prompt=prompt,
                response=response,
                recorded=recorded,
            )
            return response
        if llm_mode == "live":
            response = _openai_chat_completion(prompt, model=model or run_model)
            _record_llm_audit(
                audit_log,
                index=llm_call_count,
                mode=llm_mode,
                prompt=prompt,
                response=response,
            )
            return response
        if llm_mode == "off":
            _record_llm_audit(audit_log, index=llm_call_count, mode=llm_mode, prompt=prompt)
            raise RuntimeError("Generated code attempted llm_query while llm_mode='off'")
        raise ValueError(f"Unknown llm_mode: {{llm_mode!r}}")

    def llm_query_batched(prompts: list[str], model: str | None = None) -> list[str]:
        return [llm_query(prompt, model=model) for prompt in prompts]

    def rlm_query(prompt: str, model: str | None = None) -> str:
        return llm_query(prompt, model=model)

    def rlm_query_batched(prompts: list[str], model: str | None = None) -> list[str]:
        return llm_query_batched(prompts, model=model)

    namespace: dict[str, Any] = {{
        "__name__": "__compiled_rlm_trace__",
        "context": context,
        "answer": {{"content": "", "ready": False}},
        "llm_query": llm_query,
        "llm_query_batched": llm_query_batched,
        "rlm_query": rlm_query,
        "rlm_query_batched": rlm_query_batched,
    }}

    def FINAL(value: Any) -> None:
        namespace["answer"]["content"] = value
        namespace["answer"]["ready"] = True
        raise _FinalAnswer(value)

    def FINAL_VAR(name: str) -> None:
        if isinstance(name, str):
            value = namespace[name]
        else:
            value = name
        FINAL(value)

    namespace["FINAL"] = FINAL
    namespace["FINAL_VAR"] = FINAL_VAR

    stdout_text = ""
    try:
        for iteration, code, expected_error in TRACE_BLOCKS:
            if verbose:
                print(f"\\n# --- Iteration {{iteration}} ---", file=sys.stderr)
            compiled = compile(code, f"<{{SOURCE_LOG}}:iteration {{iteration}}>", "exec")
            iteration_stdout = io.StringIO()
            namespace_before = namespace.copy()
            with contextlib.redirect_stdout(iteration_stdout):
                try:
                    exec(compiled, namespace, namespace)
                except Exception as exc:
                    if expected_error is None or not _is_expected_error(exc, expected_error):
                        raise
                    namespace.clear()
                    namespace.update(namespace_before)
                    if verbose:
                        print(
                            f"# expected traced error: {{_format_exception(exc)}}",
                            file=sys.stderr,
                        )
                else:
                    if expected_error is not None:
                        raise RuntimeError(
                            "Trace expected this code block to fail, but it succeeded: "
                            f"iteration {{iteration}}"
                        )
            code_output = iteration_stdout.getvalue()
            stdout_text += code_output
            if echo_code_output and code_output:
                print(code_output, end="")
            answer = namespace.get("answer")
            if isinstance(answer, dict) and answer.get("ready"):
                return finish(answer.get("content", ""))
    except _FinalAnswer as final:
        return finish(final.value)

    return finish(
        _select_recovered_final(
            namespace,
            stdout_text,
            require_trace_final_match=(llm_mode == "replay"),
        )
    )


def verify_trace_final(context: str, **run_kwargs: Any) -> str:
    """Run the artifact and require the recovered value to match the trace final."""
    if TRACE_FINAL_ANSWER is None:
        raise RuntimeError("Trace does not record a final answer to verify against")
    result = run(context, **run_kwargs)
    if not _matches_trace_final(result):
        raise AssertionError("Recovered result does not match TRACE_FINAL_ANSWER")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True, help="Text file to load as context")
    parser.add_argument(
        "--llm-mode",
        choices=["replay", "live", "off"],
        default="replay",
        help="How llm_query/rlm_query calls should be handled",
    )
    parser.add_argument("--model", default=None, help="Model for --llm-mode live")
    parser.add_argument("--quiet", action="store_true", help="Suppress iteration markers")
    parser.add_argument(
        "--no-code-output",
        action="store_true",
        help="Suppress stdout produced by replayed code blocks",
    )
    parser.add_argument(
        "--llm-audit",
        type=Path,
        default=None,
        help="Write observed LLM judgment calls as JSON",
    )
    parser.add_argument(
        "--verify-trace-final",
        action="store_true",
        help="Exit non-zero unless executed trace code reproduces the recorded final answer",
    )
    args = parser.parse_args()

    audit_log: list[dict[str, Any]] = []
    run_kwargs = dict(
        llm_mode=args.llm_mode,
        model=args.model,
        verbose=not args.quiet,
        echo_code_output=not args.no_code_output,
        audit_log=audit_log if args.llm_audit is not None else None,
    )
    try:
        if args.verify_trace_final:
            result = verify_trace_final(args.context.read_text(), **run_kwargs)
            print("Verified recovered output matches TRACE_FINAL_ANSWER", file=sys.stderr)
        else:
            result = run(
                args.context.read_text(),
                **run_kwargs,
            )
    finally:
        if args.llm_audit is not None:
            args.llm_audit.parent.mkdir(parents=True, exist_ok=True)
            args.llm_audit.write_text(json.dumps(audit_log, indent=2))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _read_trace_text(log_path: Path) -> str:
    if log_path.suffix == ".gz":
        return gzip.open(log_path, mode="rt", encoding="utf-8").read()
    return log_path.read_text()


def _iter_trace_records(log_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(_read_trace_text(log_path=log_path).splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{log_path}:{line_number}: invalid JSONL record: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{log_path}:{line_number}: expected JSON object, got {type(record).__name__}")
        records.append(record)
    return records


def _default_readable_path(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}_recovered.py")


def _comment_lines(text: str, *, indent: str = "") -> list[str]:
    lines = text.splitlines() or [""]
    return [f"# {indent}{line}" if line else "#" for line in lines]


def _canonical(value: Any) -> str:
    text = str(value).strip()
    try:
        return json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"))
    except (TypeError, json.JSONDecodeError):
        return text


def _is_print_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    )


def _is_diagnostic_only(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return bool(tree.body) and all(
        isinstance(stmt, ast.Expr) and _is_print_call(stmt.value)
        for stmt in tree.body
    )


def _stdout_may_be_final(stdout_text: str, final_answer: Any) -> bool:
    stdout_values = [line.strip() for line in stdout_text.splitlines() if line.strip()]
    if not stdout_values:
        return False
    if final_answer is None:
        return True
    final = _canonical(final_answer)
    return any(_canonical(value) == final for value in stdout_values)


def _include_readable_block(code: str, expected_error: str | None, stdout_text: str, final_answer: Any) -> bool:
    if expected_error is not None:
        return False
    if not _is_diagnostic_only(code):
        return True
    return _stdout_may_be_final(stdout_text, final_answer)


def _format_llm_call_comment(call_index: int, call: dict[str, Any]) -> str:
    prompt = str(call.get("prompt", ""))
    response = str(call.get("response", ""))
    root_model = call.get("root_model") or "unknown model"
    lines = [
        f"# LLM judgment {call_index} | model: {root_model}",
        "# Recorded response:",
        *_comment_lines(response, indent="    "),
        "# Prompt boundary:",
        *_comment_lines(prompt, indent="    "),
    ]
    return "\n".join(lines)


def _expected_error_from_stderr(stderr: str) -> str | None:
    stripped = stderr.strip()
    if not stripped:
        return None

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return None
    last_line = lines[-1]
    if last_line.startswith("Traceback (most recent call last):"):
        return stripped
    if any(line.startswith("Traceback (most recent call last):") for line in lines):
        return stripped
    try:
        exc_type, _ = last_line.split(":", 1)
    except ValueError:
        return None
    if not exc_type or not exc_type.replace("_", "").isalnum():
        return None
    if not exc_type.endswith(("Error", "Exception")):
        return None
    return stripped


def _readable_source(
    *,
    log_path: Path,
    replay_path: Path,
    trace_blocks: list[tuple[int, str, str | None, list[dict[str, Any]], str]],
    recorded_llm_calls: list[dict[str, Any]],
    final_answer: Any,
) -> str:
    successful_blocks = [
        (iteration, code, calls)
        for iteration, code, expected_error, calls, stdout_text in trace_blocks
        if _include_readable_block(code, expected_error, stdout_text, final_answer)
    ]

    lines = [
        f"# Recovered Python program from {log_path.name}",
        "# Human-readable view only; use "
        f"{replay_path.name} for strict replay, verification, and LLM audit.",
        "# The variable `context` is expected to contain the original trace input.",
    ]
    if recorded_llm_calls:
        lines.append("# Recorded LLM judgment boundaries are shown before the code block that used them.")
    lines.append("")

    call_index = 1
    for block_number, (iteration, code, calls) in enumerate(successful_blocks, 1):
        if block_number > 1:
            lines.append("")
        lines.append(f"# --- Iteration {iteration} recovered code ---")
        for call in calls:
            lines.append(_format_llm_call_comment(call_index, call))
            call_index += 1
        if calls:
            lines.append("# --- Code using the recorded judgment(s) above ---")
        lines.append(code.rstrip())

    return "\n".join(lines).rstrip() + "\n"


def compile_trace(
    log_path: Path,
    out_path: Path,
    readable_out_path: Path | None = None,
    *,
    emit_readable: bool = True,
) -> None:
    trace_blocks: list[tuple[int, str, str | None]] = []
    readable_blocks: list[tuple[int, str, str | None, list[dict[str, Any]], str]] = []
    recorded_llm_calls: list[dict[str, Any]] = []
    final_answer: Any = None
    for rec in _iter_trace_records(log_path=log_path):
        if rec.get("type") != "iteration":
            continue
        for block in rec.get("code_blocks", []):
            result = block.get("result", {})
            stderr = result.get("stderr") or ""
            stdout = result.get("stdout") or ""
            expected_error = _expected_error_from_stderr(stderr)
            trace_blocks.append((rec["iteration"], block["code"], expected_error))
            block_calls = list(result.get("rlm_calls", []))
            readable_blocks.append((rec["iteration"], block["code"], expected_error, block_calls, stdout))
            for call in block_calls:
                recorded_llm_calls.append(call)
        if rec.get("final_answer") is not None:
            final_answer = rec["final_answer"]

    module_name = out_path.stem.replace("-", "_")
    source = ARTIFACT_HEADER.format(
        out_name=out_path.name,
        module_name=module_name,
        source_log=log_path.name,
        expected_final=final_answer,
        trace_blocks=trace_blocks,
        recorded_llm_calls=recorded_llm_calls,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(source)
    out_path.chmod(0o755)
    if emit_readable and readable_out_path is None:
        readable_out_path = _default_readable_path(out_path)
    if emit_readable and readable_out_path is not None:
        readable_source = _readable_source(
            log_path=log_path,
            replay_path=out_path,
            trace_blocks=readable_blocks,
            recorded_llm_calls=recorded_llm_calls,
            final_answer=final_answer,
        )
        readable_out_path.parent.mkdir(parents=True, exist_ok=True)
        readable_out_path.write_text(readable_source)
    print(
        f"Wrote {out_path} "
        f"({len(trace_blocks)} snippets, {len(recorded_llm_calls)} recorded LLM calls, "
        f"trace final answer: {final_answer!r})"
    )
    if emit_readable and readable_out_path is not None:
        print(f"Wrote {readable_out_path} (readable recovered program)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "trace",
        type=Path,
        help="Path to an RLM JSONL trace, usually named rlm_*.jsonl",
    )
    parser.add_argument(
        "out",
        type=Path,
        help="Path for the generated standalone Python artifact",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--readable-out",
        type=Path,
        default=None,
        help="Path for the human-readable recovered-program view (default: OUT stem + '_recovered.py')",
    )
    parser.add_argument(
        "--no-readable-out",
        action="store_true",
        help="Only write the strict replay artifact",
    )
    args = parser.parse_args(argv)

    compile_trace(
        log_path=args.trace,
        out_path=args.out,
        readable_out_path=args.readable_out,
        emit_readable=not args.no_readable_out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
