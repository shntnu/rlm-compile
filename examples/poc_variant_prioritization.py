#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "rlms",
#     "python-dotenv",
# ]
# ///
"""Simple bioinformatics RLM POC: synthetic variant prioritization.

Problem proposed for the RLM:
  Given a small synthetic exome-style packet with patient variants and gene
  notes, identify the top 3 candidate variants for a recessive mitochondrial
  DNA maintenance phenotype.

The deterministic parts are ordinary variant-prioritization rules: QC, allele
frequency, zygosity, consequence, CADD, ClinVar, and read support. The semantic
part is intentionally explicit: gene_notes.function_note must support
mitochondrial DNA maintenance / mtDNA replication or repair, not merely generic
mitochondrial biology or unrelated cancer/cell-cycle biology.

Run:
    uv run examples/poc_variant_prioritization.py

Gold-only validation, no API call:
    uv run examples/poc_variant_prioritization.py --gold-only
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rlm import RLM
from rlm.logger import RLMLogger


GENE_NOTES = [
    {
        "gene": "POLG",
        "function_note": "Mitochondrial DNA polymerase; essential for mtDNA replication and repair.",
        "inheritance": "AR",
    },
    {
        "gene": "TWNK",
        "function_note": "Mitochondrial DNA helicase involved in mtDNA replication fork progression.",
        "inheritance": "AR",
    },
    {
        "gene": "LIG3",
        "function_note": "DNA ligase III supports mitochondrial base-excision repair and mtDNA maintenance.",
        "inheritance": "AR",
    },
    {
        "gene": "OPA1",
        "function_note": "Mitochondrial dynamics and optic atrophy gene; not a primary mtDNA maintenance gene.",
        "inheritance": "AD",
    },
    {
        "gene": "BRCA1",
        "function_note": "Nuclear homologous recombination DNA repair and breast cancer predisposition.",
        "inheritance": "AD",
    },
    {
        "gene": "TP53",
        "function_note": "Cell-cycle checkpoint and tumor suppressor biology.",
        "inheritance": "AD",
    },
    {
        "gene": "NDUFS2",
        "function_note": "Complex I respiratory-chain subunit; mitochondrial but not mtDNA replication or repair.",
        "inheritance": "AR",
    },
]


CURATED_VARIANTS = [
    {
        "variant_id": "VAR_SYN_001",
        "gene": "POLG",
        "chrom": "15",
        "pos": 89876842,
        "ref": "C",
        "alt": "T",
        "zygosity": "homozygous",
        "gnomad_af": 0.00002,
        "cadd_phred": 38.1,
        "consequence": "stop_gained",
        "clinvar": "pathogenic",
        "qc_filter": "PASS",
        "read_depth": 74,
        "alt_depth": 71,
    },
    {
        "variant_id": "VAR_SYN_002",
        "gene": "TWNK",
        "chrom": "10",
        "pos": 102748901,
        "ref": "G",
        "alt": "A",
        "zygosity": "compound_het",
        "gnomad_af": 0.00008,
        "cadd_phred": 31.4,
        "consequence": "missense_variant",
        "clinvar": "uncertain_significance",
        "qc_filter": "PASS",
        "read_depth": 52,
        "alt_depth": 25,
    },
    {
        "variant_id": "VAR_SYN_003",
        "gene": "LIG3",
        "chrom": "17",
        "pos": 34991221,
        "ref": "G",
        "alt": "A",
        "zygosity": "compound_het",
        "gnomad_af": 0.00011,
        "cadd_phred": 29.6,
        "consequence": "splice_donor_variant",
        "clinvar": "not_reported",
        "qc_filter": "PASS",
        "read_depth": 46,
        "alt_depth": 20,
    },
    {
        "variant_id": "VAR_TRAP_MITO",
        "gene": "NDUFS2",
        "chrom": "1",
        "pos": 161198522,
        "ref": "A",
        "alt": "G",
        "zygosity": "homozygous",
        "gnomad_af": 0.00003,
        "cadd_phred": 35.5,
        "consequence": "missense_variant",
        "clinvar": "uncertain_significance",
        "qc_filter": "PASS",
        "read_depth": 61,
        "alt_depth": 59,
    },
    {
        "variant_id": "VAR_TRAP_DOMINANT",
        "gene": "BRCA1",
        "chrom": "17",
        "pos": 43071077,
        "ref": "A",
        "alt": "T",
        "zygosity": "homozygous",
        "gnomad_af": 0.00001,
        "cadd_phred": 36.2,
        "consequence": "stop_gained",
        "clinvar": "pathogenic",
        "qc_filter": "PASS",
        "read_depth": 80,
        "alt_depth": 78,
    },
    {
        "variant_id": "VAR_TRAP_QC",
        "gene": "POLG",
        "chrom": "15",
        "pos": 89811203,
        "ref": "T",
        "alt": "C",
        "zygosity": "homozygous",
        "gnomad_af": 0.00001,
        "cadd_phred": 39.0,
        "consequence": "splice_acceptor_variant",
        "clinvar": "pathogenic",
        "qc_filter": "LowQual",
        "read_depth": 18,
        "alt_depth": 17,
    },
]


DISTRACTOR_GENES = ["OPA1", "BRCA1", "TP53", "NDUFS2"]
CONSEQUENCES = [
    "synonymous_variant",
    "intron_variant",
    "missense_variant",
    "stop_gained",
]


def csv_block(name: str, rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return f"## TABLE {name}\n{buffer.getvalue().strip()}\n## END {name}"


def build_packet(seed: int, distractor_variants: int) -> tuple[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    variants = list(CURATED_VARIANTS)
    for idx in range(distractor_variants):
        gene = rng.choice(DISTRACTOR_GENES)
        variants.append(
            {
                "variant_id": f"VAR_RND_{idx:03d}",
                "gene": gene,
                "chrom": rng.choice(["1", "10", "15", "17"]),
                "pos": rng.randint(1_000_000, 240_000_000),
                "ref": rng.choice(["A", "C", "G", "T"]),
                "alt": rng.choice(["A", "C", "G", "T"]),
                "zygosity": rng.choice(["heterozygous", "homozygous", "compound_het"]),
                "gnomad_af": round(rng.uniform(0.0002, 0.02), 6),
                "cadd_phred": round(rng.uniform(5.0, 34.0), 1),
                "consequence": rng.choice(CONSEQUENCES),
                "clinvar": rng.choice(["benign", "likely_benign", "not_reported", "uncertain_significance"]),
                "qc_filter": rng.choice(["PASS", "PASS", "LowQual"]),
                "read_depth": rng.randint(8, 90),
                "alt_depth": rng.randint(2, 45),
            }
        )
    rng.shuffle(variants)
    context = "\n\n".join(
        [
            "# Synthetic exome variant-prioritization packet",
            "The data are synthetic and intentionally tiny. Join key: variants.gene -> gene_notes.gene.",
            csv_block("variants", variants),
            csv_block("gene_notes", GENE_NOTES),
        ]
    )
    return context, compute_gold(variants, GENE_NOTES)


def supports_mtdna_maintenance(note: str) -> bool:
    normalized = note.lower()
    return (
        ("mtdna" in normalized or "mitochondrial dna" in normalized)
        and ("replication" in normalized or "repair" in normalized or "maintenance" in normalized)
        and "not a primary" not in normalized
        and "not mtdna" not in normalized
        and "not mitochondrial dna" not in normalized
    )


def compute_gold(
    variants: list[dict[str, Any]],
    gene_notes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    notes_by_gene = {row["gene"]: row for row in gene_notes}
    candidates: list[dict[str, Any]] = []
    for row in variants:
        note = notes_by_gene.get(row["gene"])
        if note is None:
            continue
        if row["qc_filter"] != "PASS":
            continue
        if row["zygosity"] not in {"homozygous", "compound_het"}:
            continue
        if float(row["gnomad_af"]) > 0.001:
            continue
        if int(row["read_depth"]) < 20 or int(row["alt_depth"]) < 8:
            continue
        if row["clinvar"] in {"benign", "likely_benign"}:
            continue
        consequence = row["consequence"]
        cadd = float(row["cadd_phred"])
        if consequence == "missense_variant":
            if cadd < 25.0:
                continue
            consequence_weight = 0.75
        elif consequence in {"stop_gained", "frameshift_variant", "splice_acceptor_variant", "splice_donor_variant"}:
            if cadd < 20.0:
                continue
            consequence_weight = 1.0
        else:
            continue
        if not supports_mtdna_maintenance(note["function_note"]):
            continue

        rarity_score = min(1.0, -__import__("math").log10(max(float(row["gnomad_af"]), 1e-9)) / 6.0)
        cadd_score = min(cadd, 40.0) / 40.0
        score = 0.45 * cadd_score + 0.35 * rarity_score + 0.20 * consequence_weight
        candidates.append(
            {
                "variant_id": row["variant_id"],
                "gene": row["gene"],
                "score": round(score, 4),
                "consequence": consequence,
                "zygosity": row["zygosity"],
                "gnomad_af": float(row["gnomad_af"]),
                "cadd_phred": cadd,
                "reason": note["function_note"],
            }
        )
    return sorted(candidates, key=lambda item: item["score"], reverse=True)[:3]


def root_prompt() -> str:
    return """
The variable `context` contains a synthetic exome variant-prioritization packet
with two CSV tables: variants and gene_notes.

Find the top 3 candidate variants for a recessive mitochondrial DNA maintenance
phenotype.

Eligibility rules:
- Join variants to gene_notes by gene.
- Require variants.qc_filter == PASS.
- Require zygosity to be homozygous or compound_het.
- Require gnomad_af <= 0.001.
- Require read_depth >= 20 and alt_depth >= 8.
- Exclude clinvar benign or likely_benign.
- Keep missense_variant only if cadd_phred >= 25.
- Keep stop_gained, frameshift_variant, splice_acceptor_variant, or
  splice_donor_variant only if cadd_phred >= 20.
- The gene_notes.function_note must support mitochondrial DNA maintenance,
  mtDNA replication, or mtDNA repair. Generic mitochondrial biology, nuclear
  DNA repair, cancer, or cell-cycle notes are not enough.

After applying the deterministic filters in Python, use llm_query_batched() on
the remaining gene_notes.function_note values as the explicit semantic judgment
boundary for the mitochondrial DNA maintenance criterion. Replay/live/off modes
in the extracted artifact should make that boundary visible.

Rank eligible variants by:
rarity_score = min(1.0, -log10(max(gnomad_af, 1e-9)) / 6.0)
cadd_score = min(cadd_phred, 40.0) / 40.0
consequence_weight = 1.0 for predicted LoF consequences, 0.75 for missense
score = 0.45 * cadd_score + 0.35 * rarity_score + 0.20 * consequence_weight

Return ONLY compact JSON, no markdown. The JSON must be an array of three
objects with keys: variant_id, gene, score, consequence, zygosity, gnomad_af,
cadd_phred, reason.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--distractor-variants", type=int, default=40)
    parser.add_argument("--backend", default="openrouter")
    parser.add_argument("--model", default="anthropic/claude-sonnet-4.5")
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--log-dir", type=Path, default=Path("./traces"))
    parser.add_argument("--gold-only", action="store_true", help="Print gold answer without calling an API.")
    parser.add_argument("--context-out", type=Path, default=None)
    parser.add_argument("--gold-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv()
    context, gold = build_packet(seed=args.seed, distractor_variants=args.distractor_variants)

    print(f"Packet size: {len(context):,} chars")
    print(f"Gold answer: {json.dumps(gold, indent=2)}")
    print()

    if args.context_out is not None:
        args.context_out.write_text(context)
        print(f"Wrote context packet: {args.context_out}")
    if args.gold_out is not None:
        args.gold_out.write_text(json.dumps(gold, indent=2))
        print(f"Wrote gold answer: {args.gold_out}")
    if args.context_out is not None or args.gold_out is not None:
        print()

    if args.gold_only:
        return 0

    api_key_name = "OPENROUTER_API_KEY" if args.backend == "openrouter" else "OPENAI_API_KEY"
    api_key = os.getenv(api_key_name)
    if not api_key:
        raise SystemExit(f"{api_key_name} is required for backend={args.backend!r}")

    rlm = RLM(
        backend=args.backend,
        backend_kwargs={"api_key": api_key, "model_name": args.model},
        environment="local",
        max_depth=args.max_depth,
        max_iterations=args.max_iterations,
        logger=RLMLogger(log_dir=str(args.log_dir)),
        verbose=True,
    )

    result = rlm.completion(prompt=context, root_prompt=root_prompt())

    print()
    print("Model answer:")
    print(result.response)
    print()
    print("Gold answer:")
    print(json.dumps(gold, indent=2))
    try:
        model_answer = json.loads(result.response)
        model_ids = [row["variant_id"] for row in model_answer]
        gold_ids = [row["variant_id"] for row in gold]
        print()
        print(f"ID order matches gold: {model_ids == gold_ids}")
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        print()
        print(f"Could not parse model answer as expected JSON: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
