# Recovered Python program from rlm_2026-05-15_22-10-22_3a476735.jsonl.gz
# Human-readable view only; use activity_triage_3a476735.py for strict replay, verification, and LLM audit.
# The variable `context` is expected to contain the original trace input.
# Recorded LLM judgment boundaries are shown before the code block that used them.

# --- Iteration 2 recovered code ---
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

# --- Iteration 3 recovered code ---
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

# --- Iteration 4 recovered code ---
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

# --- Iteration 5 recovered code ---
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

# --- Iteration 6 recovered code ---
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

# --- Iteration 7 recovered code ---
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
