# Recovered Python program from rlm_2026-05-16_07-38-23_a1c3cdfe.jsonl.gz
# Human-readable view only; use variant_prioritization_a1c3cdfe.py for strict replay, verification, and LLM audit.
# The variable `context` is expected to contain the original trace input.
# Recorded LLM judgment boundaries are shown before the code block that used them.

# --- Iteration 1 recovered code ---
import csv
import io
import math

# Parse the context to extract the two CSV tables
lines = context.strip().split('\n')

# Find where variants table ends and gene_notes begins
variants_start = None
gene_notes_start = None

for i, line in enumerate(lines):
    if 'variant_id' in line and variants_start is None:
        variants_start = i
    elif 'gene,' in line and 'function_note' in line:
        gene_notes_start = i
        break

# Extract variants table
variants_lines = []
for i in range(variants_start, len(lines)):
    if lines[i].strip() == '' or (gene_notes_start and i >= gene_notes_start):
        break
    variants_lines.append(lines[i])

# Extract gene_notes table
gene_notes_lines = []
if gene_notes_start:
    for i in range(gene_notes_start, len(lines)):
        if lines[i].strip():
            gene_notes_lines.append(lines[i])

print(f"Variants table has {len(variants_lines)} lines")
print(f"Gene notes table has {len(gene_notes_lines)} lines")

# Parse variants CSV
variants_csv = '\n'.join(variants_lines)
variants_reader = csv.DictReader(io.StringIO(variants_csv))
variants = list(variants_reader)

print(f"\nParsed {len(variants)} variants")
print("Variant keys:", variants[0].keys() if variants else "None")

# Parse gene_notes CSV
gene_notes_csv = '\n'.join(gene_notes_lines)
gene_notes_reader = csv.DictReader(io.StringIO(gene_notes_csv))
gene_notes = list(gene_notes_reader)

print(f"\nParsed {len(gene_notes)} gene notes")
print("Gene note keys:", gene_notes[0].keys() if gene_notes else "None")

# --- Iteration 2 recovered code ---
# Create gene_notes lookup by gene
gene_notes_dict = {note['gene']: note for note in gene_notes}

# Apply deterministic filters
filtered_variants = []

for v in variants:
    # Join with gene_notes
    gene = v['gene']
    if gene not in gene_notes_dict:
        continue
    
    gene_note = gene_notes_dict[gene]
    
    # Filter 1: qc_filter == PASS
    if v['qc_filter'] != 'PASS':
        continue
    
    # Filter 2: zygosity homozygous or compound_het
    if v['zygosity'] not in ['homozygous', 'compound_het']:
        continue
    
    # Filter 3: gnomad_af <= 0.001
    gnomad_af = float(v['gnomad_af'])
    if gnomad_af > 0.001:
        continue
    
    # Filter 4: read_depth >= 20 and alt_depth >= 8
    read_depth = int(v['read_depth'])
    alt_depth = int(v['alt_depth'])
    if read_depth < 20 or alt_depth < 8:
        continue
    
    # Filter 5: Exclude clinvar benign or likely_benign (note: column is 'clinvar' not 'clinvar_sig')
    clinvar = v['clinvar'].lower()
    if 'benign' in clinvar or 'likely_benign' in clinvar:
        continue
    
    # Filter 6 & 7: consequence-specific CADD filters
    consequence = v['consequence']
    cadd_phred = float(v['cadd_phred'])
    
    if consequence == 'missense_variant':
        if cadd_phred < 25:
            continue
    elif consequence in ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant']:
        if cadd_phred < 20:
            continue
    
    # Store with gene note for semantic filtering
    filtered_variants.append({
        'variant': v,
        'gene_note': gene_note
    })

print(f"\nAfter deterministic filters: {len(filtered_variants)} variants remain")
for fv in filtered_variants:
    print(f"  {fv['variant']['variant_id']} - {fv['variant']['gene']} - {fv['variant']['consequence']} - CADD:{fv['variant']['cadd_phred']} - AF:{fv['variant']['gnomad_af']}")

# --- Iteration 2 recovered code ---
# LLM judgment 1 | model: anthropic/claude-sonnet-4.5
# Recorded response:
#     NO
# Prompt boundary:
#     Does this gene function description specifically support mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair?
#
#     Gene: NDUFS2
#     Function: Complex I respiratory-chain subunit; mitochondrial but not mtDNA replication or repair.
#
#     Requirements:
#     - Must specifically mention mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair
#     - Generic mitochondrial biology is NOT sufficient
#     - Nuclear DNA repair is NOT sufficient
#     - Cancer or cell-cycle notes are NOT sufficient
#
#     Answer ONLY with: YES or NO
# LLM judgment 2 | model: anthropic/claude-sonnet-4.5
# Recorded response:
#     NO
# Prompt boundary:
#     Does this gene function description specifically support mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair?
#
#     Gene: BRCA1
#     Function: Nuclear homologous recombination DNA repair and breast cancer predisposition.
#
#     Requirements:
#     - Must specifically mention mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair
#     - Generic mitochondrial biology is NOT sufficient
#     - Nuclear DNA repair is NOT sufficient
#     - Cancer or cell-cycle notes are NOT sufficient
#
#     Answer ONLY with: YES or NO
# LLM judgment 3 | model: anthropic/claude-sonnet-4.5
# Recorded response:
#     YES
# Prompt boundary:
#     Does this gene function description specifically support mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair?
#
#     Gene: LIG3
#     Function: DNA ligase III supports mitochondrial base-excision repair and mtDNA maintenance.
#
#     Requirements:
#     - Must specifically mention mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair
#     - Generic mitochondrial biology is NOT sufficient
#     - Nuclear DNA repair is NOT sufficient
#     - Cancer or cell-cycle notes are NOT sufficient
#
#     Answer ONLY with: YES or NO
# LLM judgment 4 | model: anthropic/claude-sonnet-4.5
# Recorded response:
#     NO
# Prompt boundary:
#     Does this gene function description specifically support mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair?
#
#     Gene: OPA1
#     Function: Mitochondrial dynamics and optic atrophy gene; not a primary mtDNA maintenance gene.
#
#     Requirements:
#     - Must specifically mention mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair
#     - Generic mitochondrial biology is NOT sufficient
#     - Nuclear DNA repair is NOT sufficient
#     - Cancer or cell-cycle notes are NOT sufficient
#
#     Answer ONLY with: YES or NO
# LLM judgment 5 | model: anthropic/claude-sonnet-4.5
# Recorded response:
#     YES
# Prompt boundary:
#     Does this gene function description specifically support mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair?
#
#     Gene: POLG
#     Function: Mitochondrial DNA polymerase; essential for mtDNA replication and repair.
#
#     Requirements:
#     - Must specifically mention mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair
#     - Generic mitochondrial biology is NOT sufficient
#     - Nuclear DNA repair is NOT sufficient
#     - Cancer or cell-cycle notes are NOT sufficient
#
#     Answer ONLY with: YES or NO
# LLM judgment 6 | model: anthropic/claude-sonnet-4.5
# Recorded response:
#     YES
# Prompt boundary:
#     Does this gene function description specifically support mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair?
#
#     Gene: TWNK
#     Function: Mitochondrial DNA helicase involved in mtDNA replication fork progression.
#
#     Requirements:
#     - Must specifically mention mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair
#     - Generic mitochondrial biology is NOT sufficient
#     - Nuclear DNA repair is NOT sufficient
#     - Cancer or cell-cycle notes are NOT sufficient
#
#     Answer ONLY with: YES or NO
# LLM judgment 7 | model: anthropic/claude-sonnet-4.5
# Recorded response:
#     NO
# Prompt boundary:
#     Does this gene function description specifically support mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair?
#
#     Gene: TP53
#     Function: Cell-cycle checkpoint and tumor suppressor biology.
#
#     Requirements:
#     - Must specifically mention mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair
#     - Generic mitochondrial biology is NOT sufficient
#     - Nuclear DNA repair is NOT sufficient
#     - Cancer or cell-cycle notes are NOT sufficient
#
#     Answer ONLY with: YES or NO
# --- Code using the recorded judgment(s) above ---
# Extract unique gene notes for semantic filtering
unique_genes = {}
for fv in filtered_variants:
    gene = fv['variant']['gene']
    if gene not in unique_genes:
        unique_genes[gene] = fv['gene_note']['function_note']

print(f"\nNeed to semantically filter {len(unique_genes)} unique genes")

# Create prompts for batched LLM query
prompts = []
gene_list = list(unique_genes.keys())

for gene in gene_list:
    function_note = unique_genes[gene]
    prompt = f"""Does this gene function description specifically support mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair?

Gene: {gene}
Function: {function_note}

Requirements:
- Must specifically mention mitochondrial DNA maintenance, mtDNA replication, or mtDNA repair
- Generic mitochondrial biology is NOT sufficient
- Nuclear DNA repair is NOT sufficient
- Cancer or cell-cycle notes are NOT sufficient

Answer ONLY with: YES or NO"""
    prompts.append(prompt)

# Query LLMs in batch
semantic_results = llm_query_batched(prompts)

# Parse results
approved_genes = set()
for i, result in enumerate(semantic_results):
    gene = gene_list[i]
    if 'YES' in result.upper():
        approved_genes.add(gene)
        print(f"✓ {gene}: APPROVED")
    else:
        print(f"✗ {gene}: REJECTED - {result[:100]}")

print(f"\n{len(approved_genes)} genes approved for mitochondrial DNA maintenance")

# --- Iteration 2 recovered code ---
import math

# Filter to only variants with approved genes
eligible_variants = [fv for fv in filtered_variants if fv['variant']['gene'] in approved_genes]

print(f"\n{len(eligible_variants)} eligible variants after semantic filtering")

# Calculate scores for ranking
scored_variants = []

for fv in eligible_variants:
    v = fv['variant']
    
    gnomad_af = float(v['gnomad_af'])
    cadd_phred = float(v['cadd_phred'])
    consequence = v['consequence']
    
    # Calculate rarity score
    rarity_score = min(1.0, -math.log10(max(gnomad_af, 1e-9)) / 6.0)
    
    # Calculate CADD score
    cadd_score = min(cadd_phred, 40.0) / 40.0
    
    # Calculate consequence weight
    lof_consequences = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant']
    consequence_weight = 1.0 if consequence in lof_consequences else 0.75
    
    # Calculate final score
    score = 0.45 * cadd_score + 0.35 * rarity_score + 0.20 * consequence_weight
    
    scored_variants.append({
        'variant_id': v['variant_id'],
        'gene': v['gene'],
        'score': score,
        'consequence': consequence,
        'zygosity': v['zygosity'],
        'gnomad_af': gnomad_af,
        'cadd_phred': cadd_phred,
        'function_note': fv['gene_note']['function_note']
    })

# Sort by score descending
scored_variants.sort(key=lambda x: x['score'], reverse=True)

print(f"\nTop variants by score:")
for sv in scored_variants:
    print(f"  {sv['variant_id']} - {sv['gene']} - Score: {sv['score']:.3f} - {sv['consequence']} - CADD:{sv['cadd_phred']} - AF:{sv['gnomad_af']}")

# --- Iteration 2 recovered code ---
import json

# Get top 3 variants
top_3 = scored_variants[:3]

# Create output with reason field
result = []
for sv in top_3:
    result.append({
        'variant_id': sv['variant_id'],
        'gene': sv['gene'],
        'score': round(sv['score'], 3),
        'consequence': sv['consequence'],
        'zygosity': sv['zygosity'],
        'gnomad_af': sv['gnomad_af'],
        'cadd_phred': sv['cadd_phred'],
        'reason': f"{sv['gene']} involved in mtDNA maintenance; {sv['consequence']}; rare (AF={sv['gnomad_af']:.2e}); CADD={sv['cadd_phred']}"
    })

final_answer = json.dumps(result, indent=None, separators=(',', ':'))
print(final_answer)
