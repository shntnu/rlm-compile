#!/usr/bin/env python3
"""Replayable Python artifact compiled from an RLM trace.

This file contains the Python code blocks emitted by the root model, plus small
compatibility shims for final-answer signaling and plain LLM calls.

CLI:
    python gene_expression_hits_5f4c2a91.py --context context.txt
    python gene_expression_hits_5f4c2a91.py --context context.txt --llm-mode live --model openai/gpt-5-mini

Python:
    import gene_expression_hits_5f4c2a91
    answer = gene_expression_hits_5f4c2a91.run(context_string)
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


SOURCE_LOG = 'rlm_2026-05-16_08-12-00_5f4c2a91.jsonl.gz'
TRACE_FINAL_ANSWER = '[{"gene":"CYP3A4","log2_fc":2.4,"q_value":0.0008},{"gene":"ALB","log2_fc":1.8,"q_value":0.012},{"gene":"APOB","log2_fc":1.6,"q_value":0.041}]'
TRACE_BLOCKS: list[tuple[int, str, str | None]] = [(1, 'import csv, io, json\nrows = list(csv.DictReader(io.StringIO(context)))\nhits = [\n    {"gene": r["gene"], "log2_fc": float(r["log2_fc"]), "q_value": float(r["q_value"])}\n    for r in rows\n    if r["tissue"] == "liver" and float(r["log2_fc"]) >= 1.5 and float(r["q_value"]) <= 0.05\n]\nhits.sort(key=lambda row: (-row["log2_fc"], row["q_value"], row["gene"]))\nfinal_answer = json.dumps(hits, separators=(",", ":"))\nprint(final_answer)', None)]
RECORDED_LLM_CALLS: list[dict[str, Any]] = []


class _FinalAnswer(Exception):
    def __init__(self, value: Any):
        self.value = value
        super().__init__(str(value))


def _chat_completion(prompt: str, model: str | None = None) -> str:
    """Minimal OpenAI-compatible chat-completions call using only stdlib."""
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for --llm-mode live")

    base_url = os.environ.get(
        "OPENROUTER_BASE_URL",
        os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
    ).rstrip("/")
    model_name = model or os.environ.get(
        "OPENROUTER_MODEL",
        os.environ.get("OPENAI_MODEL", "openai/gpt-5-mini"),
    )
    payload = json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM HTTP error {exc.code}: {body}") from exc

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
            f"TRACE_FINAL_ANSWER; candidate sources: {sources}"
        )

    if candidates:
        return str(candidates[0][1])
    raise RuntimeError("Trace completed without a recovered final value")


def _format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


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
    event: dict[str, Any] = {
        "index": index,
        "mode": mode,
        "prompt_sha256": _sha256_text(prompt),
        "prompt": prompt,
    }
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
                f"{len(recorded_calls)} unused recorded LLM response(s)"
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
                    f"{llm_call_count}: expected sha256={_sha256_text(recorded_prompt)}, "
                    f"got sha256={_sha256_text(prompt)}"
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
            response = _chat_completion(prompt, model=model or run_model)
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
        raise ValueError(f"Unknown llm_mode: {llm_mode!r}")

    def llm_query_batched(prompts: list[str], model: str | None = None) -> list[str]:
        return [llm_query(prompt, model=model) for prompt in prompts]

    def rlm_query(prompt: str, model: str | None = None) -> str:
        return llm_query(prompt, model=model)

    def rlm_query_batched(prompts: list[str], model: str | None = None) -> list[str]:
        return llm_query_batched(prompts, model=model)

    namespace: dict[str, Any] = {
        "__name__": "__compiled_rlm_trace__",
        "context": context,
        "answer": {"content": "", "ready": False},
        "llm_query": llm_query,
        "llm_query_batched": llm_query_batched,
        "rlm_query": rlm_query,
        "rlm_query_batched": rlm_query_batched,
    }

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
                print(f"\\n# --- Iteration {iteration} ---", file=sys.stderr)
            compiled = compile(code, f"<{SOURCE_LOG}:iteration {iteration}>", "exec")
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
                            f"# expected traced error: {_format_exception(exc)}",
                            file=sys.stderr,
                        )
                else:
                    if expected_error is not None:
                        raise RuntimeError(
                            "Trace expected this code block to fail, but it succeeded: "
                            f"iteration {iteration}"
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
