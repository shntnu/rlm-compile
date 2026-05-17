#!/usr/bin/env python3
"""Replayable Python artifact compiled from an RLM trace.

This file contains the Python code blocks emitted by the root model, plus small
compatibility shims for final-answer signaling and plain LLM calls.

CLI:
    python variant_prioritization_a1c3cdfe.py --context context.txt
    python variant_prioritization_a1c3cdfe.py --context context.txt --llm-mode live --model gpt-5-mini

Python:
    import variant_prioritization_a1c3cdfe
    answer = variant_prioritization_a1c3cdfe.run(context_string)
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


SOURCE_LOG = 'rlm_2026-05-16_07-38-23_a1c3cdfe.jsonl.gz'
TRACE_FINAL_ANSWER = '[{"variant_id":"VAR_SYN_001","gene":"POLG","score":0.903,"consequence":"stop_gained","zygosity":"homozygous","gnomad_af":2e-05,"cadd_phred":38.1,"reason":"POLG involved in mtDNA maintenance; stop_gained; rare (AF=2.00e-05); CADD=38.1"},{"variant_id":"VAR_SYN_003","gene":"LIG3","score":0.764,"consequence":"splice_donor_variant","zygosity":"compound_het","gnomad_af":0.00011,"cadd_phred":29.6,"reason":"LIG3 involved in mtDNA maintenance; splice_donor_variant; rare (AF=1.10e-04); CADD=29.6"},{"variant_id":"VAR_SYN_002","gene":"TWNK","score":0.742,"consequence":"missense_variant","zygosity":"compound_het","gnomad_af":8e-05,"cadd_phred":31.4,"reason":"TWNK involved in mtDNA maintenance; missense_variant; rare (AF=8.00e-05); CADD=31.4"}]'
TRACE_BLOCKS: list[tuple[int, str, str | None]] = [(1, '# First, let\'s see what the context looks like\nprint("Context length:", len(context))\nprint("\\nFirst 1000 characters of context:")\nprint(context[:1000])', None), (1, "# Let's see the full context to understand the data structure\nprint(context)", None), (1, 'import csv\nimport io\nimport math\n\n# Parse the context to extract the two CSV tables\nlines = context.strip().split(\'\\n\')\n\n# Find where variants table ends and gene_notes begins\nvariants_start = None\ngene_notes_start = None\n\nfor i, line in enumerate(lines):\n    if \'variant_id\' in line and variants_start is None:\n        variants_start = i\n    elif \'gene,\' in line and \'function_note\' in line:\n        gene_notes_start = i\n        break\n\n# Extract variants table\nvariants_lines = []\nfor i in range(variants_start, len(lines)):\n    if lines[i].strip() == \'\' or (gene_notes_start and i >= gene_notes_start):\n        break\n    variants_lines.append(lines[i])\n\n# Extract gene_notes table\ngene_notes_lines = []\nif gene_notes_start:\n    for i in range(gene_notes_start, len(lines)):\n        if lines[i].strip():\n            gene_notes_lines.append(lines[i])\n\nprint(f"Variants table has {len(variants_lines)} lines")\nprint(f"Gene notes table has {len(gene_notes_lines)} lines")\n\n# Parse variants CSV\nvariants_csv = \'\\n\'.join(variants_lines)\nvariants_reader = csv.DictReader(io.StringIO(variants_csv))\nvariants = list(variants_reader)\n\nprint(f"\\nParsed {len(variants)} variants")\nprint("Variant keys:", variants[0].keys() if variants else "None")\n\n# Parse gene_notes CSV\ngene_notes_csv = \'\\n\'.join(gene_notes_lines)\ngene_notes_reader = csv.DictReader(io.StringIO(gene_notes_csv))\ngene_notes = list(gene_notes_reader)\n\nprint(f"\\nParsed {len(gene_notes)} gene notes")\nprint("Gene note keys:", gene_notes[0].keys() if gene_notes else "None")', None), (1, '# Create gene_notes lookup by gene\ngene_notes_dict = {note[\'gene\']: note for note in gene_notes}\n\n# Apply deterministic filters\nfiltered_variants = []\n\nfor v in variants:\n    # Join with gene_notes\n    gene = v[\'gene\']\n    if gene not in gene_notes_dict:\n        continue\n    \n    gene_note = gene_notes_dict[gene]\n    \n    # Filter 1: qc_filter == PASS\n    if v[\'qc_filter\'] != \'PASS\':\n        continue\n    \n    # Filter 2: zygosity homozygous or compound_het\n    if v[\'zygosity\'] not in [\'homozygous\', \'compound_het\']:\n        continue\n    \n    # Filter 3: gnomad_af <= 0.001\n    gnomad_af = float(v[\'gnomad_af\'])\n    if gnomad_af > 0.001:\n        continue\n    \n    # Filter 4: read_depth >= 20 and alt_depth >= 8\n    read_depth = int(v[\'read_depth\'])\n    alt_depth = int(v[\'alt_depth\'])\n    if read_depth < 20 or alt_depth < 8:\n        continue\n    \n    # Filter 5: Exclude clinvar benign or likely_benign\n    clinvar = v[\'clinvar_sig\'].lower()\n    if \'benign\' in clinvar or \'likely_benign\' in clinvar:\n        continue\n    \n    # Filter 6 & 7: consequence-specific CADD filters\n    consequence = v[\'consequence\']\n    cadd_phred = float(v[\'cadd_phred\'])\n    \n    if consequence == \'missense_variant\':\n        if cadd_phred < 25:\n            continue\n    elif consequence in [\'stop_gained\', \'frameshift_variant\', \'splice_acceptor_variant\', \'splice_donor_variant\']:\n        if cadd_phred < 20:\n            continue\n    \n    # Store with gene note for semantic filtering\n    filtered_variants.append({\n        \'variant\': v,\n        \'gene_note\': gene_note\n    })\n\nprint(f"\\nAfter deterministic filters: {len(filtered_variants)} variants remain")\nfor fv in filtered_variants:\n    print(f"  {fv[\'variant\'][\'variant_id\']} - {fv[\'variant\'][\'gene\']} - {fv[\'variant\'][\'consequence\']}")', "KeyError: 'clinvar_sig'"), (1, '# Extract unique gene notes for semantic filtering\nunique_genes = {}\nfor fv in filtered_variants:\n    gene = fv[\'variant\'][\'gene\']\n    if gene not in unique_genes:\n        unique_genes[gene] = fv[\'gene_note\'][\'function_note\']\n\nprint(f"\\nNeed to semantically filter {len(unique_genes)} unique genes")\n\n# Create prompts for batched LLM query\nprompts = []\ngene_list = list(unique_genes.keys())\n\nfor gene in gene_list:\n    function_note = unique_genes[gene]\n    prompt = f"""Does this gene function description specifically support mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair?\n\nGene: {gene}\nFunction: {function_note}\n\nRequirements:\n- Must specifically mention mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair\n- Generic mitochondrial biology is NOT sufficient\n- Nuclear DNA repair is NOT sufficient\n- Cancer or cell-cycle notes are NOT sufficient\n\nAnswer ONLY with: YES or NO"""\n    prompts.append(prompt)\n\n# Query LLMs in batch\nsemantic_results = llm_query_batched(prompts)\n\n# Parse results\napproved_genes = set()\nfor i, result in enumerate(semantic_results):\n    gene = gene_list[i]\n    if \'YES\' in result.upper():\n        approved_genes.add(gene)\n        print(f"✓ {gene}: APPROVED")\n    else:\n        print(f"✗ {gene}: REJECTED - {result[:100]}")\n\nprint(f"\\n{len(approved_genes)} genes approved for mitochondrial DNA maintenance")', "NameError: name 'filtered_variants' is not defined"), (1, '# Filter to only variants with approved genes\neligible_variants = [fv for fv in filtered_variants if fv[\'variant\'][\'gene\'] in approved_genes]\n\nprint(f"\\n{len(eligible_variants)} eligible variants after semantic filtering")\n\n# Calculate scores for ranking\nscored_variants = []\n\nfor fv in eligible_variants:\n    v = fv[\'variant\']\n    \n    gnomad_af = float(v[\'gnomad_af\'])\n    cadd_phred = float(v[\'cadd_phred\'])\n    consequence = v[\'consequence\']\n    \n    # Calculate rarity score\n    rarity_score = min(1.0, -math.log10(max(gnomad_af, 1e-9)) / 6.0)\n    \n    # Calculate CADD score\n    cadd_score = min(cadd_phred, 40.0) / 40.0\n    \n    # Calculate consequence weight\n    lof_consequences = [\'stop_gained\', \'frameshift_variant\', \'splice_acceptor_variant\', \'splice_donor_variant\']\n    consequence_weight = 1.0 if consequence in lof_consequences else 0.75\n    \n    # Calculate final score\n    score = 0.45 * cadd_score + 0.35 * rarity_score + 0.20 * consequence_weight\n    \n    scored_variants.append({\n        \'variant_id\': v[\'variant_id\'],\n        \'gene\': v[\'gene\'],\n        \'score\': score,\n        \'consequence\': consequence,\n        \'zygosity\': v[\'zygosity\'],\n        \'gnomad_af\': gnomad_af,\n        \'cadd_phred\': cadd_phred,\n        \'function_note\': fv[\'gene_note\'][\'function_note\']\n    })\n\n# Sort by score descending\nscored_variants.sort(key=lambda x: x[\'score\'], reverse=True)\n\nprint(f"\\nTop variants by score:")\nfor sv in scored_variants[:5]:\n    print(f"  {sv[\'variant_id\']} - {sv[\'gene\']} - Score: {sv[\'score\']:.3f} - {sv[\'consequence\']}")', "NameError: name 'filtered_variants' is not defined"), (1, 'import json\n\n# Get top 3 variants\ntop_3 = scored_variants[:3]\n\n# Create output with reason field\nresult = []\nfor sv in top_3:\n    result.append({\n        \'variant_id\': sv[\'variant_id\'],\n        \'gene\': sv[\'gene\'],\n        \'score\': round(sv[\'score\'], 3),\n        \'consequence\': sv[\'consequence\'],\n        \'zygosity\': sv[\'zygosity\'],\n        \'gnomad_af\': sv[\'gnomad_af\'],\n        \'cadd_phred\': sv[\'cadd_phred\'],\n        \'reason\': f"{sv[\'gene\']} involved in mtDNA maintenance; {sv[\'consequence\']}; rare (AF={sv[\'gnomad_af\']:.2e}); CADD={sv[\'cadd_phred\']}"\n    })\n\nfinal_answer = json.dumps(result, indent=None, separators=(\',\', \':\'))\nprint(final_answer)', "NameError: name 'scored_variants' is not defined"), (2, '# Create gene_notes lookup by gene\ngene_notes_dict = {note[\'gene\']: note for note in gene_notes}\n\n# Apply deterministic filters\nfiltered_variants = []\n\nfor v in variants:\n    # Join with gene_notes\n    gene = v[\'gene\']\n    if gene not in gene_notes_dict:\n        continue\n    \n    gene_note = gene_notes_dict[gene]\n    \n    # Filter 1: qc_filter == PASS\n    if v[\'qc_filter\'] != \'PASS\':\n        continue\n    \n    # Filter 2: zygosity homozygous or compound_het\n    if v[\'zygosity\'] not in [\'homozygous\', \'compound_het\']:\n        continue\n    \n    # Filter 3: gnomad_af <= 0.001\n    gnomad_af = float(v[\'gnomad_af\'])\n    if gnomad_af > 0.001:\n        continue\n    \n    # Filter 4: read_depth >= 20 and alt_depth >= 8\n    read_depth = int(v[\'read_depth\'])\n    alt_depth = int(v[\'alt_depth\'])\n    if read_depth < 20 or alt_depth < 8:\n        continue\n    \n    # Filter 5: Exclude clinvar benign or likely_benign (note: column is \'clinvar\' not \'clinvar_sig\')\n    clinvar = v[\'clinvar\'].lower()\n    if \'benign\' in clinvar or \'likely_benign\' in clinvar:\n        continue\n    \n    # Filter 6 & 7: consequence-specific CADD filters\n    consequence = v[\'consequence\']\n    cadd_phred = float(v[\'cadd_phred\'])\n    \n    if consequence == \'missense_variant\':\n        if cadd_phred < 25:\n            continue\n    elif consequence in [\'stop_gained\', \'frameshift_variant\', \'splice_acceptor_variant\', \'splice_donor_variant\']:\n        if cadd_phred < 20:\n            continue\n    \n    # Store with gene note for semantic filtering\n    filtered_variants.append({\n        \'variant\': v,\n        \'gene_note\': gene_note\n    })\n\nprint(f"\\nAfter deterministic filters: {len(filtered_variants)} variants remain")\nfor fv in filtered_variants:\n    print(f"  {fv[\'variant\'][\'variant_id\']} - {fv[\'variant\'][\'gene\']} - {fv[\'variant\'][\'consequence\']} - CADD:{fv[\'variant\'][\'cadd_phred\']} - AF:{fv[\'variant\'][\'gnomad_af\']}")', None), (2, '# Extract unique gene notes for semantic filtering\nunique_genes = {}\nfor fv in filtered_variants:\n    gene = fv[\'variant\'][\'gene\']\n    if gene not in unique_genes:\n        unique_genes[gene] = fv[\'gene_note\'][\'function_note\']\n\nprint(f"\\nNeed to semantically filter {len(unique_genes)} unique genes")\n\n# Create prompts for batched LLM query\nprompts = []\ngene_list = list(unique_genes.keys())\n\nfor gene in gene_list:\n    function_note = unique_genes[gene]\n    prompt = f"""Does this gene function description specifically support mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair?\n\nGene: {gene}\nFunction: {function_note}\n\nRequirements:\n- Must specifically mention mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair\n- Generic mitochondrial biology is NOT sufficient\n- Nuclear DNA repair is NOT sufficient\n- Cancer or cell-cycle notes are NOT sufficient\n\nAnswer ONLY with: YES or NO"""\n    prompts.append(prompt)\n\n# Query LLMs in batch\nsemantic_results = llm_query_batched(prompts)\n\n# Parse results\napproved_genes = set()\nfor i, result in enumerate(semantic_results):\n    gene = gene_list[i]\n    if \'YES\' in result.upper():\n        approved_genes.add(gene)\n        print(f"✓ {gene}: APPROVED")\n    else:\n        print(f"✗ {gene}: REJECTED - {result[:100]}")\n\nprint(f"\\n{len(approved_genes)} genes approved for mitochondrial DNA maintenance")', None), (2, 'import math\n\n# Filter to only variants with approved genes\neligible_variants = [fv for fv in filtered_variants if fv[\'variant\'][\'gene\'] in approved_genes]\n\nprint(f"\\n{len(eligible_variants)} eligible variants after semantic filtering")\n\n# Calculate scores for ranking\nscored_variants = []\n\nfor fv in eligible_variants:\n    v = fv[\'variant\']\n    \n    gnomad_af = float(v[\'gnomad_af\'])\n    cadd_phred = float(v[\'cadd_phred\'])\n    consequence = v[\'consequence\']\n    \n    # Calculate rarity score\n    rarity_score = min(1.0, -math.log10(max(gnomad_af, 1e-9)) / 6.0)\n    \n    # Calculate CADD score\n    cadd_score = min(cadd_phred, 40.0) / 40.0\n    \n    # Calculate consequence weight\n    lof_consequences = [\'stop_gained\', \'frameshift_variant\', \'splice_acceptor_variant\', \'splice_donor_variant\']\n    consequence_weight = 1.0 if consequence in lof_consequences else 0.75\n    \n    # Calculate final score\n    score = 0.45 * cadd_score + 0.35 * rarity_score + 0.20 * consequence_weight\n    \n    scored_variants.append({\n        \'variant_id\': v[\'variant_id\'],\n        \'gene\': v[\'gene\'],\n        \'score\': score,\n        \'consequence\': consequence,\n        \'zygosity\': v[\'zygosity\'],\n        \'gnomad_af\': gnomad_af,\n        \'cadd_phred\': cadd_phred,\n        \'function_note\': fv[\'gene_note\'][\'function_note\']\n    })\n\n# Sort by score descending\nscored_variants.sort(key=lambda x: x[\'score\'], reverse=True)\n\nprint(f"\\nTop variants by score:")\nfor sv in scored_variants:\n    print(f"  {sv[\'variant_id\']} - {sv[\'gene\']} - Score: {sv[\'score\']:.3f} - {sv[\'consequence\']} - CADD:{sv[\'cadd_phred\']} - AF:{sv[\'gnomad_af\']}")', None), (2, 'import json\n\n# Get top 3 variants\ntop_3 = scored_variants[:3]\n\n# Create output with reason field\nresult = []\nfor sv in top_3:\n    result.append({\n        \'variant_id\': sv[\'variant_id\'],\n        \'gene\': sv[\'gene\'],\n        \'score\': round(sv[\'score\'], 3),\n        \'consequence\': sv[\'consequence\'],\n        \'zygosity\': sv[\'zygosity\'],\n        \'gnomad_af\': sv[\'gnomad_af\'],\n        \'cadd_phred\': sv[\'cadd_phred\'],\n        \'reason\': f"{sv[\'gene\']} involved in mtDNA maintenance; {sv[\'consequence\']}; rare (AF={sv[\'gnomad_af\']:.2e}); CADD={sv[\'cadd_phred\']}"\n    })\n\nfinal_answer = json.dumps(result, indent=None, separators=(\',\', \':\'))\nprint(final_answer)', None)]
RECORDED_LLM_CALLS: list[dict[str, Any]] = [{'root_model': 'anthropic/claude-sonnet-4.5', 'prompt': 'Does this gene function description specifically support mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair?\n\nGene: NDUFS2\nFunction: Complex I respiratory-chain subunit; mitochondrial but not mtDNA replication or repair.\n\nRequirements:\n- Must specifically mention mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair\n- Generic mitochondrial biology is NOT sufficient\n- Nuclear DNA repair is NOT sufficient\n- Cancer or cell-cycle notes are NOT sufficient\n\nAnswer ONLY with: YES or NO', 'response': 'NO', 'usage_summary': {'model_usage_summaries': {'anthropic/claude-sonnet-4.5': {'total_calls': 1, 'total_input_tokens': 123, 'total_output_tokens': 4, 'total_cost': 0.000429}}, 'total_cost': 0.000429}, 'execution_time': 0.285250285706882}, {'root_model': 'anthropic/claude-sonnet-4.5', 'prompt': 'Does this gene function description specifically support mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair?\n\nGene: BRCA1\nFunction: Nuclear homologous recombination DNA repair and breast cancer predisposition.\n\nRequirements:\n- Must specifically mention mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair\n- Generic mitochondrial biology is NOT sufficient\n- Nuclear DNA repair is NOT sufficient\n- Cancer or cell-cycle notes are NOT sufficient\n\nAnswer ONLY with: YES or NO', 'response': 'NO', 'usage_summary': {'model_usage_summaries': {'anthropic/claude-sonnet-4.5': {'total_calls': 1, 'total_input_tokens': 123, 'total_output_tokens': 4, 'total_cost': 0.000429}}, 'total_cost': 0.000429}, 'execution_time': 0.285250285706882}, {'root_model': 'anthropic/claude-sonnet-4.5', 'prompt': 'Does this gene function description specifically support mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair?\n\nGene: LIG3\nFunction: DNA ligase III supports mitochondrial base-excision repair and mtDNA maintenance.\n\nRequirements:\n- Must specifically mention mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair\n- Generic mitochondrial biology is NOT sufficient\n- Nuclear DNA repair is NOT sufficient\n- Cancer or cell-cycle notes are NOT sufficient\n\nAnswer ONLY with: YES or NO', 'response': 'YES', 'usage_summary': {'model_usage_summaries': {'anthropic/claude-sonnet-4.5': {'total_calls': 1, 'total_input_tokens': 123, 'total_output_tokens': 4, 'total_cost': 0.000429}}, 'total_cost': 0.000429}, 'execution_time': 0.285250285706882}, {'root_model': 'anthropic/claude-sonnet-4.5', 'prompt': 'Does this gene function description specifically support mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair?\n\nGene: OPA1\nFunction: Mitochondrial dynamics and optic atrophy gene; not a primary mtDNA maintenance gene.\n\nRequirements:\n- Must specifically mention mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair\n- Generic mitochondrial biology is NOT sufficient\n- Nuclear DNA repair is NOT sufficient\n- Cancer or cell-cycle notes are NOT sufficient\n\nAnswer ONLY with: YES or NO', 'response': 'NO', 'usage_summary': {'model_usage_summaries': {'anthropic/claude-sonnet-4.5': {'total_calls': 1, 'total_input_tokens': 123, 'total_output_tokens': 4, 'total_cost': 0.000429}}, 'total_cost': 0.000429}, 'execution_time': 0.285250285706882}, {'root_model': 'anthropic/claude-sonnet-4.5', 'prompt': 'Does this gene function description specifically support mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair?\n\nGene: POLG\nFunction: Mitochondrial DNA polymerase; essential for mtDNA replication and repair.\n\nRequirements:\n- Must specifically mention mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair\n- Generic mitochondrial biology is NOT sufficient\n- Nuclear DNA repair is NOT sufficient\n- Cancer or cell-cycle notes are NOT sufficient\n\nAnswer ONLY with: YES or NO', 'response': 'YES', 'usage_summary': {'model_usage_summaries': {'anthropic/claude-sonnet-4.5': {'total_calls': 1, 'total_input_tokens': 123, 'total_output_tokens': 4, 'total_cost': 0.000429}}, 'total_cost': 0.000429}, 'execution_time': 0.285250285706882}, {'root_model': 'anthropic/claude-sonnet-4.5', 'prompt': 'Does this gene function description specifically support mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair?\n\nGene: TWNK\nFunction: Mitochondrial DNA helicase involved in mtDNA replication fork progression.\n\nRequirements:\n- Must specifically mention mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair\n- Generic mitochondrial biology is NOT sufficient\n- Nuclear DNA repair is NOT sufficient\n- Cancer or cell-cycle notes are NOT sufficient\n\nAnswer ONLY with: YES or NO', 'response': 'YES', 'usage_summary': {'model_usage_summaries': {'anthropic/claude-sonnet-4.5': {'total_calls': 1, 'total_input_tokens': 123, 'total_output_tokens': 4, 'total_cost': 0.000429}}, 'total_cost': 0.000429}, 'execution_time': 0.285250285706882}, {'root_model': 'anthropic/claude-sonnet-4.5', 'prompt': 'Does this gene function description specifically support mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair?\n\nGene: TP53\nFunction: Cell-cycle checkpoint and tumor suppressor biology.\n\nRequirements:\n- Must specifically mention mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair\n- Generic mitochondrial biology is NOT sufficient\n- Nuclear DNA repair is NOT sufficient\n- Cancer or cell-cycle notes are NOT sufficient\n\nAnswer ONLY with: YES or NO', 'response': 'NO', 'usage_summary': {'model_usage_summaries': {'anthropic/claude-sonnet-4.5': {'total_calls': 1, 'total_input_tokens': 123, 'total_output_tokens': 4, 'total_cost': 0.000429}}, 'total_cost': 0.000429}, 'execution_time': 0.285250285706882}]


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


def _select_recovered_final(namespace: dict[str, Any], stdout_text: str) -> str:
    candidates = _candidate_final_values(namespace, stdout_text)
    if TRACE_FINAL_ANSWER is not None:
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

    return finish(_select_recovered_final(namespace, stdout_text))


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
