#!/usr/bin/env python3
"""Recovered program from rlm_2026-05-15_22-10-22_3a476735.jsonl.gz.

Runnable standalone script compiled from an RLM trace. Provides the same
algorithm as activity_triage_3a476735.py in a flat, readable form.

CLI:
    python activity_triage_3a476735_recovered.py --context input.txt
    python activity_triage_3a476735_recovered.py --context input.txt --model openai/gpt-5-mini

Python:
    from activity_triage_3a476735_recovered import run
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
    mod = types.ModuleType("activity_triage_3a476735_recovered")
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
import re
import json

# Parse the three CSV tables from the context
tables = {}
current_table = None
current_lines = []

for line in context.split('\n'):
    line = line.strip()
    if line.startswith('## TABLE '):
        # Save previous table if exists
        if current_table and current_lines:
            tables[current_table] = current_lines
        # Start new table
        current_table = line.replace('## TABLE ', '')
        current_lines = []
    elif line and not line.startswith('#') and current_table:
        current_lines.append(line)

# Save last table
if current_table and current_lines:
    tables[current_table] = current_lines

print("Tables found:", list(tables.keys()))
print("\nCompound metadata lines:", len(tables.get('compound_metadata', [])))
print("Activity results lines:", len(tables.get('activity_results', [])))
print("Plate QC lines:", len(tables.get('plate_qc', [])))

# --- Iteration 3 ---
# Parse CSV data into dictionaries
def parse_csv(lines):
    if not lines:
        return []
    header = lines[0].split(',')
    data = []
    for line in lines[1:]:
        values = line.split(',')
        if len(values) == len(header):
            data.append(dict(zip(header, values)))
    return data

compound_metadata = parse_csv(tables['compound_metadata'])
activity_results = parse_csv(tables['activity_results'])
plate_qc = parse_csv(tables['plate_qc'])

print(f"Parsed {len(compound_metadata)} compounds")
print(f"Parsed {len(activity_results)} activity results")
print(f"Parsed {len(plate_qc)} plate QC records")

# Check sample records
print("\nSample compound:", compound_metadata[0])
print("\nSample activity:", activity_results[0])
print("\nSample plate_qc:", plate_qc[0])

# --- Iteration 4 ---
# Step 1: Filter activity_results for NDUFS2 at 5.0 uM dose
ndufs2_activities = [
    ar for ar in activity_results
    if ar['gene_symbol'] == 'NDUFS2' and ar['dose_um'] == '5.0'
]
print(f"NDUFS2 at 5.0 uM: {len(ndufs2_activities)} records")

# Step 2: Create lookup dictionaries
compound_dict = {c['jcp_id']: c for c in compound_metadata}
plate_qc_dict = {p['qc_plate_id']: p for p in plate_qc}

# Step 3: Join and filter
eligible = []

for ar in ndufs2_activities:
    jcp_id = ar['jcp_id']
    qc_plate_id = ar['qc_plate_id']
    
    # Join compound_metadata
    if jcp_id not in compound_dict:
        continue
    compound = compound_dict[jcp_id]
    
    # Join plate_qc
    if qc_plate_id not in plate_qc_dict:
        continue
    plate = plate_qc_dict[qc_plate_id]
    
    # Filter: pass_qc == true
    if plate['pass_qc'] != 'true':
        continue
    
    # Filter: exclude cytotoxic or pains
    if compound['structure_alerts'] in ['cytotoxic', 'pains']:
        continue
    
    # Filter: target_confidence >= 0.60
    if float(compound['target_confidence']) < 0.60:
        continue
    
    # Filter: activity_map >= 0.72
    if float(ar['activity_map']) < 0.72:
        continue
    
    # Filter: similarity_to_gene_ko >= 0.64
    if float(ar['similarity_to_gene_ko']) < 0.64:
        continue
    
    # Filter: profile_consistency >= 0.75
    if float(ar['profile_consistency']) < 0.75:
        continue
    
    # Filter: absolute cell_count_z <= 2.0
    if abs(float(ar['cell_count_z'])) > 2.0:
        continue
    
    # Add to eligible list with all data
    eligible.append({
        'jcp_id': jcp_id,
        'compound_name': compound['compound_name'],
        'moa_note': compound['moa_note'],
        'activity_map': float(ar['activity_map']),
        'similarity_to_gene_ko': float(ar['similarity_to_gene_ko']),
        'profile_consistency': float(ar['profile_consistency']),
        'qc_plate_id': qc_plate_id,
        'compound': compound,
        'activity': ar
    })

print(f"After numeric filters: {len(eligible)} eligible records")
print("\nEligible compounds and their moa_notes:")
for e in eligible:
    print(f"{e['compound_name']}: {e['moa_note']}")

# --- Iteration 5 ---
# LLM judgment 1 | model: anthropic/claude-sonnet-4.5
# Recorded response:
#     YES
# Prompt boundary:
#     Does this moa_note support Complex I or NADH dehydrogenase activity (and NOT Complex III)?
#     moa_note: "Weak but specific NADH dehydrogenase signature; Complex I rescue evidence present."
#
#     Answer only: YES or NO
# LLM judgment 2 | model: anthropic/claude-sonnet-4.5
# Recorded response:
#     YES
# Prompt boundary:
#     Does this moa_note support Complex I or NADH dehydrogenase activity (and NOT Complex III)?
#     moa_note: "Mitochondrial Complex I blocker; orthogonal Seahorse assay confirms NADH arm."
#
#     Answer only: YES or NO
# LLM judgment 3 | model: anthropic/claude-sonnet-4.5
# Recorded response:
#     YES
# Prompt boundary:
#     Does this moa_note support Complex I or NADH dehydrogenase activity (and NOT Complex III)?
#     moa_note: "Complex I / NADH dehydrogenase inhibitor; respiration rescue follow-up passed."
#
#     Answer only: YES or NO
# --- Code using the recorded judgment(s) above ---
# Filter by moa_note: must support Complex I or NADH dehydrogenase, not Complex III
# Use LLM to check each moa_note since this is semantic analysis
prompts = []
for e in eligible:
    prompt = f"""Does this moa_note support Complex I or NADH dehydrogenase activity (and NOT Complex III)?
moa_note: "{e['moa_note']}"

Answer only: YES or NO"""
    prompts.append(prompt)

# Batch query for efficiency
answers = llm_query_batched(prompts)

# Filter based on LLM answers
filtered_eligible = []
for i, (e, answer) in enumerate(zip(eligible, answers)):
    print(f"\nCompound {i+1}: {e['compound_name']}")
    print(f"moa_note: {e['moa_note']}")
    print(f"LLM says: {answer.strip()}")
    if "YES" in answer.upper():
        filtered_eligible.append(e)

print(f"\n\nAfter moa_note filter: {len(filtered_eligible)} eligible compounds")

# --- Iteration 6 ---
# Now rank the filtered eligible compounds by the formula
for e in filtered_eligible:
    score = 0.45 * e['activity_map'] + 0.35 * e['similarity_to_gene_ko'] + 0.20 * e['profile_consistency']
    e['score'] = score

# Sort by score descending
filtered_eligible.sort(key=lambda x: x['score'], reverse=True)

# Take top 3 (we have exactly 3)
top_3 = filtered_eligible[:3]

print("Top 3 compounds:")
for i, e in enumerate(top_3, 1):
    print(f"\n{i}. {e['compound_name']} (score: {e['score']:.4f})")
    print(f"   activity_map: {e['activity_map']}")
    print(f"   similarity_to_gene_ko: {e['similarity_to_gene_ko']}")
    print(f"   profile_consistency: {e['profile_consistency']}")
    print(f"   moa_note: {e['moa_note']}")

# --- Iteration 7 ---
# Create the final JSON output
import json

final_output = []
for e in top_3:
    final_output.append({
        "jcp_id": e['jcp_id'],
        "compound_name": e['compound_name'],
        "score": round(e['score'], 4),
        "activity_map": e['activity_map'],
        "similarity_to_NDUFS2_KO": e['similarity_to_gene_ko'],
        "profile_consistency": e['profile_consistency'],
        "qc_plate_id": e['qc_plate_id'],
        "reason": e['moa_note']
    })

# Convert to compact JSON
final_json = json.dumps(final_output, separators=(',', ':'))
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
