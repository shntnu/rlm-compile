#!/usr/bin/env python3
"""Recovered program from rlm_2026-05-17_01-47-31_be4c96ee.jsonl.

Runnable standalone script compiled from an RLM trace. Provides the same
algorithm as gc_content_rank.py in a flat, readable form.

CLI:
    python gc_content_rank_recovered.py --context input.txt
    python gc_content_rank_recovered.py --context input.txt --model openai/gpt-5-mini

Python:
    from gc_content_rank_recovered import run
    answer = run(context_string)
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class _FinalAnswer(Exception):
    def __init__(self, value: Any):
        self.value = value
        super().__init__(str(value))


def _chat_completion(prompt: str, model: str | None = None) -> str:
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    api_key = openrouter_api_key or openai_api_key
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY or OPENAI_API_KEY is required for live LLM calls"
        )

    if openrouter_api_key:
        base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        model_name = model or os.environ.get("OPENROUTER_MODEL", "openai/gpt-5-mini")
    else:
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model_name = model or os.environ.get("OPENAI_MODEL", "gpt-5-mini")
    base_url = base_url.rstrip("/")
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


_RUN_MODEL: str | None = None
_RUN_NAMESPACE: dict[str, Any] | None = None


def llm_query(prompt: str, model: str | None = None) -> str:
    return _chat_completion(prompt, model=model or _RUN_MODEL)


def llm_query_batched(prompts: list[str], model: str | None = None) -> list[str]:
    return [llm_query(p, model=model) for p in prompts]


def rlm_query(prompt: str, model: str | None = None) -> str:
    return llm_query(prompt, model=model)


def rlm_query_batched(prompts: list[str], model: str | None = None) -> list[str]:
    return llm_query_batched(prompts, model=model)


answer: dict[str, Any] = {"content": "", "ready": False}


def FINAL(value: Any) -> None:
    namespace = _RUN_NAMESPACE or globals()
    namespace["answer"]["content"] = value
    namespace["answer"]["ready"] = True
    raise _FinalAnswer(value)


def FINAL_VAR(name: str) -> None:
    namespace = _RUN_NAMESPACE or globals()
    value = namespace[name] if isinstance(name, str) else name
    FINAL(value)


def run(context: str, *, model: str | None = None) -> str:
    global _RUN_MODEL, _RUN_NAMESPACE
    _RUN_MODEL = model
    import contextlib
    import io
    import types
    mod = types.ModuleType("gc_content_rank_recovered")
    mod.__dict__.update(globals())
    mod.__dict__["answer"] = {"content": "", "ready": False}
    mod.__dict__["context"] = context
    mod.__dict__["__name__"] = "__compiled_rlm_trace__"
    _RUN_NAMESPACE = mod.__dict__
    stdout_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(compile(_RECOVERED_CODE, __file__, "exec"), mod.__dict__)
    except _FinalAnswer as final:
        return str(final.value)
    finally:
        _RUN_NAMESPACE = None
    _ans = mod.__dict__.get("answer")
    if isinstance(_ans, dict) and _ans.get("ready"):
        return str(_ans["content"])
    for _name in ("final_answer", "final_json"):
        if _name in mod.__dict__:
            return str(mod.__dict__[_name])
    for _line in reversed(stdout_capture.getvalue().splitlines()):
        if _line.strip():
            return _line.strip()
    return ""


_RECOVERED_CODE = r'''
# --- Iteration 2 ---
import json

# Parse the CSV data
lines = context.strip().split('\n')
header = lines[0]
rows = lines[1:]

# Process human sequences
human_sequences = []
for row in rows:
    parts = row.split(',')
    if len(parts) == 3:
        seq_id, organism, sequence = parts
        if organism == 'human':
            # Calculate GC content
            sequence_upper = sequence.upper()
            g_count = sequence_upper.count('G')
            c_count = sequence_upper.count('C')
            total_length = len(sequence_upper)
            gc_content = (g_count + c_count) / total_length if total_length > 0 else 0
            gc_content_rounded = round(gc_content, 4)
            
            human_sequences.append({
                'sequence_id': seq_id,
                'gc_content': gc_content_rounded
            })

# Sort by gc_content descending, then sequence_id ascending
human_sequences.sort(key=lambda x: (-x['gc_content'], x['sequence_id']))

# Get top 3
top_3 = human_sequences[:3]

# Convert to JSON
result_json = json.dumps(top_3)
print("Top 3 human sequences by GC content:")
print(result_json)
'''


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True, help="Text file to load as context")
    parser.add_argument("--model", default=None, help="Model override for LLM calls")
    args = parser.parse_args()

    result = run(args.context.read_text(), model=args.model)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
