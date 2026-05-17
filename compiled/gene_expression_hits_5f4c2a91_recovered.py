# Recovered Python program from rlm_2026-05-16_08-12-00_5f4c2a91.jsonl.gz
# Human-readable view only; use gene_expression_hits_5f4c2a91.py for strict replay, verification, and LLM audit.
# The variable `context` is expected to contain the original trace input.

# --- Iteration 1 recovered code ---
import csv, io, json
rows = list(csv.DictReader(io.StringIO(context)))
hits = [
    {"gene": r["gene"], "log2_fc": float(r["log2_fc"]), "q_value": float(r["q_value"])}
    for r in rows
    if r["tissue"] == "liver" and float(r["log2_fc"]) >= 1.5 and float(r["q_value"]) <= 0.05
]
hits.sort(key=lambda row: (-row["log2_fc"], row["q_value"], row["gene"]))
final_answer = json.dumps(hits, separators=(",", ":"))
print(final_answer)
