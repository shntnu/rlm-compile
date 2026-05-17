#!/usr/bin/env python3
"""Replayable Python artifact compiled from an RLM trace.

This file contains the Python code blocks emitted by the root model, plus small
compatibility shims for final-answer signaling and plain LLM calls.

CLI:
    python activity_triage_3a476735.py --context context.txt
    python activity_triage_3a476735.py --context context.txt --llm-mode live --model openai/gpt-5-mini

Python:
    import activity_triage_3a476735
    answer = activity_triage_3a476735.run(context_string)
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


SOURCE_LOG = 'rlm_2026-05-15_22-10-22_3a476735.jsonl.gz'
TRACE_FINAL_ANSWER = '[{"jcp_id":"JCP2022_SYN001","compound_name":"Nadhrestat","score":0.8964,"activity_map":0.931,"similarity_to_NDUFS2_KO":0.842,"profile_consistency":0.914,"qc_plate_id":"PLATE_001","reason":"Complex I / NADH dehydrogenase inhibitor; respiration rescue follow-up passed."},{"jcp_id":"JCP2022_SYN002","compound_name":"Oxphosarin","score":0.8539,"activity_map":0.887,"similarity_to_NDUFS2_KO":0.801,"profile_consistency":0.872,"qc_plate_id":"PLATE_002","reason":"Mitochondrial Complex I blocker; orthogonal Seahorse assay confirms NADH arm."},{"jcp_id":"JCP2022_SYN003","compound_name":"Ndufexor","score":0.8032,"activity_map":0.821,"similarity_to_NDUFS2_KO":0.757,"profile_consistency":0.844,"qc_plate_id":"PLATE_003","reason":"Weak but specific NADH dehydrogenase signature; Complex I rescue evidence present."}]'
TRACE_BLOCKS: list[tuple[int, str, str | None]] = [(1, '# First, let\'s see what the context looks like\nprint("Context length:", len(context))\nprint("\\nFirst 2000 characters of context:")\nprint(context[:2000])', None), (2, 'import re\nimport json\n\n# Parse the three CSV tables from the context\ntables = {}\ncurrent_table = None\ncurrent_lines = []\n\nfor line in context.split(\'\\n\'):\n    line = line.strip()\n    if line.startswith(\'## TABLE \'):\n        # Save previous table if exists\n        if current_table and current_lines:\n            tables[current_table] = current_lines\n        # Start new table\n        current_table = line.replace(\'## TABLE \', \'\')\n        current_lines = []\n    elif line and not line.startswith(\'#\') and current_table:\n        current_lines.append(line)\n\n# Save last table\nif current_table and current_lines:\n    tables[current_table] = current_lines\n\nprint("Tables found:", list(tables.keys()))\nprint("\\nCompound metadata lines:", len(tables.get(\'compound_metadata\', [])))\nprint("Activity results lines:", len(tables.get(\'activity_results\', [])))\nprint("Plate QC lines:", len(tables.get(\'plate_qc\', [])))', None), (3, '# Parse CSV data into dictionaries\ndef parse_csv(lines):\n    if not lines:\n        return []\n    header = lines[0].split(\',\')\n    data = []\n    for line in lines[1:]:\n        values = line.split(\',\')\n        if len(values) == len(header):\n            data.append(dict(zip(header, values)))\n    return data\n\ncompound_metadata = parse_csv(tables[\'compound_metadata\'])\nactivity_results = parse_csv(tables[\'activity_results\'])\nplate_qc = parse_csv(tables[\'plate_qc\'])\n\nprint(f"Parsed {len(compound_metadata)} compounds")\nprint(f"Parsed {len(activity_results)} activity results")\nprint(f"Parsed {len(plate_qc)} plate QC records")\n\n# Check sample records\nprint("\\nSample compound:", compound_metadata[0])\nprint("\\nSample activity:", activity_results[0])\nprint("\\nSample plate_qc:", plate_qc[0])', None), (4, '# Step 1: Filter activity_results for NDUFS2 at 5.0 uM dose\nndufs2_activities = [\n    ar for ar in activity_results\n    if ar[\'gene_symbol\'] == \'NDUFS2\' and ar[\'dose_um\'] == \'5.0\'\n]\nprint(f"NDUFS2 at 5.0 uM: {len(ndufs2_activities)} records")\n\n# Step 2: Create lookup dictionaries\ncompound_dict = {c[\'jcp_id\']: c for c in compound_metadata}\nplate_qc_dict = {p[\'qc_plate_id\']: p for p in plate_qc}\n\n# Step 3: Join and filter\neligible = []\n\nfor ar in ndufs2_activities:\n    jcp_id = ar[\'jcp_id\']\n    qc_plate_id = ar[\'qc_plate_id\']\n    \n    # Join compound_metadata\n    if jcp_id not in compound_dict:\n        continue\n    compound = compound_dict[jcp_id]\n    \n    # Join plate_qc\n    if qc_plate_id not in plate_qc_dict:\n        continue\n    plate = plate_qc_dict[qc_plate_id]\n    \n    # Filter: pass_qc == true\n    if plate[\'pass_qc\'] != \'true\':\n        continue\n    \n    # Filter: exclude cytotoxic or pains\n    if compound[\'structure_alerts\'] in [\'cytotoxic\', \'pains\']:\n        continue\n    \n    # Filter: target_confidence >= 0.60\n    if float(compound[\'target_confidence\']) < 0.60:\n        continue\n    \n    # Filter: activity_map >= 0.72\n    if float(ar[\'activity_map\']) < 0.72:\n        continue\n    \n    # Filter: similarity_to_gene_ko >= 0.64\n    if float(ar[\'similarity_to_gene_ko\']) < 0.64:\n        continue\n    \n    # Filter: profile_consistency >= 0.75\n    if float(ar[\'profile_consistency\']) < 0.75:\n        continue\n    \n    # Filter: absolute cell_count_z <= 2.0\n    if abs(float(ar[\'cell_count_z\'])) > 2.0:\n        continue\n    \n    # Add to eligible list with all data\n    eligible.append({\n        \'jcp_id\': jcp_id,\n        \'compound_name\': compound[\'compound_name\'],\n        \'moa_note\': compound[\'moa_note\'],\n        \'activity_map\': float(ar[\'activity_map\']),\n        \'similarity_to_gene_ko\': float(ar[\'similarity_to_gene_ko\']),\n        \'profile_consistency\': float(ar[\'profile_consistency\']),\n        \'qc_plate_id\': qc_plate_id,\n        \'compound\': compound,\n        \'activity\': ar\n    })\n\nprint(f"After numeric filters: {len(eligible)} eligible records")\nprint("\\nEligible compounds and their moa_notes:")\nfor e in eligible:\n    print(f"{e[\'compound_name\']}: {e[\'moa_note\']}")', None), (5, '# Filter by moa_note: must support Complex I or NADH dehydrogenase, not Complex III\n# Use LLM to check each moa_note since this is semantic analysis\nprompts = []\nfor e in eligible:\n    prompt = f"""Does this moa_note support Complex I or NADH dehydrogenase activity (and NOT Complex III)?\nmoa_note: "{e[\'moa_note\']}"\n\nAnswer only: YES or NO"""\n    prompts.append(prompt)\n\n# Batch query for efficiency\nanswers = llm_query_batched(prompts)\n\n# Filter based on LLM answers\nfiltered_eligible = []\nfor i, (e, answer) in enumerate(zip(eligible, answers)):\n    print(f"\\nCompound {i+1}: {e[\'compound_name\']}")\n    print(f"moa_note: {e[\'moa_note\']}")\n    print(f"LLM says: {answer.strip()}")\n    if "YES" in answer.upper():\n        filtered_eligible.append(e)\n\nprint(f"\\n\\nAfter moa_note filter: {len(filtered_eligible)} eligible compounds")', None), (6, '# Now rank the filtered eligible compounds by the formula\nfor e in filtered_eligible:\n    score = 0.45 * e[\'activity_map\'] + 0.35 * e[\'similarity_to_gene_ko\'] + 0.20 * e[\'profile_consistency\']\n    e[\'score\'] = score\n\n# Sort by score descending\nfiltered_eligible.sort(key=lambda x: x[\'score\'], reverse=True)\n\n# Take top 3 (we have exactly 3)\ntop_3 = filtered_eligible[:3]\n\nprint("Top 3 compounds:")\nfor i, e in enumerate(top_3, 1):\n    print(f"\\n{i}. {e[\'compound_name\']} (score: {e[\'score\']:.4f})")\n    print(f"   activity_map: {e[\'activity_map\']}")\n    print(f"   similarity_to_gene_ko: {e[\'similarity_to_gene_ko\']}")\n    print(f"   profile_consistency: {e[\'profile_consistency\']}")\n    print(f"   moa_note: {e[\'moa_note\']}")', None), (7, '# Create the final JSON output\nimport json\n\nfinal_output = []\nfor e in top_3:\n    final_output.append({\n        "jcp_id": e[\'jcp_id\'],\n        "compound_name": e[\'compound_name\'],\n        "score": round(e[\'score\'], 4),\n        "activity_map": e[\'activity_map\'],\n        "similarity_to_NDUFS2_KO": e[\'similarity_to_gene_ko\'],\n        "profile_consistency": e[\'profile_consistency\'],\n        "qc_plate_id": e[\'qc_plate_id\'],\n        "reason": e[\'moa_note\']\n    })\n\n# Convert to compact JSON\nfinal_json = json.dumps(final_output, separators=(\',\', \':\'))\nprint(final_json)', None)]
RECORDED_LLM_CALLS: list[dict[str, Any]] = [{'root_model': 'anthropic/claude-sonnet-4.5', 'prompt': 'Does this moa_note support Complex I or NADH dehydrogenase activity (and NOT Complex III)?\nmoa_note: "Weak but specific NADH dehydrogenase signature; Complex I rescue evidence present."\n\nAnswer only: YES or NO', 'response': 'YES', 'usage_summary': {'model_usage_summaries': {'anthropic/claude-sonnet-4.5': {'total_calls': 1, 'total_input_tokens': 66, 'total_output_tokens': 4, 'total_cost': 0.000258}}, 'total_cost': 0.000258}, 'execution_time': 4.398933958320413}, {'root_model': 'anthropic/claude-sonnet-4.5', 'prompt': 'Does this moa_note support Complex I or NADH dehydrogenase activity (and NOT Complex III)?\nmoa_note: "Mitochondrial Complex I blocker; orthogonal Seahorse assay confirms NADH arm."\n\nAnswer only: YES or NO', 'response': 'YES', 'usage_summary': {'model_usage_summaries': {'anthropic/claude-sonnet-4.5': {'total_calls': 1, 'total_input_tokens': 66, 'total_output_tokens': 4, 'total_cost': 0.000258}}, 'total_cost': 0.000258}, 'execution_time': 4.398933958320413}, {'root_model': 'anthropic/claude-sonnet-4.5', 'prompt': 'Does this moa_note support Complex I or NADH dehydrogenase activity (and NOT Complex III)?\nmoa_note: "Complex I / NADH dehydrogenase inhibitor; respiration rescue follow-up passed."\n\nAnswer only: YES or NO', 'response': 'YES', 'usage_summary': {'model_usage_summaries': {'anthropic/claude-sonnet-4.5': {'total_calls': 1, 'total_input_tokens': 66, 'total_output_tokens': 4, 'total_cost': 0.000258}}, 'total_cost': 0.000258}, 'execution_time': 4.398933958320413}]


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
