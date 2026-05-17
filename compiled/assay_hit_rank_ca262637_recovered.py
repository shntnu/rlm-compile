#!/usr/bin/env python3
"""Recovered program from rlm_2026-05-16_23-52-02_ca262637.jsonl.gz.

Runnable standalone script compiled from an RLM trace. Provides the same
algorithm as assay_hit_rank_ca262637.py in a flat, readable form.

CLI:
    python assay_hit_rank_ca262637_recovered.py --context input.txt
    python assay_hit_rank_ca262637_recovered.py --context input.txt --model openai/gpt-5-mini

Python:
    from assay_hit_rank_ca262637_recovered import run
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
    answer["content"] = value
    answer["ready"] = True
    raise _FinalAnswer(value)


def FINAL_VAR(name: str) -> None:
    value = globals()[name] if isinstance(name, str) else name
    FINAL(value)


def run(context: str, *, model: str | None = None) -> str:
    global _RUN_MODEL
    _RUN_MODEL = model
    import contextlib
    import io
    import types
    mod = types.ModuleType("assay_hit_rank_ca262637_recovered")
    mod.__dict__.update(globals())
    mod.__dict__["context"] = context
    mod.__dict__["__name__"] = "__compiled_rlm_trace__"
    stdout_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(compile(_RECOVERED_CODE, __file__, "exec"), mod.__dict__)
    except _FinalAnswer as final:
        return str(final.value)
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
import csv
import json
from io import StringIO

# Parse the CSV
csv_reader = csv.DictReader(StringIO(context))
rows = list(csv_reader)

print("All rows:")
for row in rows:
    print(row)

# Apply filters and calculate scores
filtered_results = []

for row in rows:
    # Filter: target == NDUFS2
    if row['target'] != 'NDUFS2':
        continue
    
    # Filter: pass_qc == true
    if row['pass_qc'] != 'true':
        continue
    
    # Get numeric values
    activity = float(row['activity'])
    toxicity = float(row['toxicity'])
    
    # Filter: toxicity <= 0.15
    if toxicity > 0.15:
        continue
    
    # Calculate score
    score = activity - 0.5 * toxicity
    score = round(score, 3)
    
    filtered_results.append({
        'compound_id': row['compound_id'],
        'activity': activity,
        'toxicity': toxicity,
        'score': score
    })

print(f"\nFiltered results (before sorting): {len(filtered_results)} compounds")
for result in filtered_results:
    print(result)

# Sort by descending score, then by compound_id
filtered_results.sort(key=lambda x: (-x['score'], x['compound_id']))

print("\nAfter sorting:")
for result in filtered_results:
    print(result)

# Get top 3
top_3 = filtered_results[:3]

print(f"\nTop 3:")
for result in top_3:
    print(result)

# Create compact JSON
final_json = json.dumps(top_3, separators=(',', ':'))
print(f"\nFinal JSON:")
print(final_json)
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
