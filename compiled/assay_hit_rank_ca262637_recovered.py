# Recovered Python program from rlm_2026-05-16_23-52-02_ca262637.jsonl.gz
# Human-readable view only; use assay_hit_rank_ca262637.py for strict replay, verification, and LLM audit.
# The variable `context` is expected to contain the original trace input.

# --- Iteration 2 recovered code ---
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
