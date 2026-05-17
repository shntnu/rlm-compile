#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "rlms",
#     "python-dotenv",
# ]
# ///
"""Richer RLM POC: synthetic JUMP-style activity-result triage.

This is a deliberately more complicated follow-up to `poc_needle.py`.
Instead of finding one literal string, it gives the RLM a long, noisy,
multi-table packet that resembles the kind of context a jx skill would
normally teach an agent how to query:

  - compound metadata with messy mechanism notes
  - activity rows across genes and doses
  - plate-level QC rows

The root LM receives only the analysis question. The packet itself is loaded
into the REPL as `context`, so the model should write Python to parse, join,
filter, and rank the rows. With `--subcall-review`, the prompt also asks the
root model to use rlm_query_batched() on its shortlist, exercising the recursive
path that the simple needle POC never touched.

Run:
    uv run examples/poc_activity_triage.py

Gold-only validation, no API call:
    uv run examples/poc_activity_triage.py --gold-only

Write the deterministic replay inputs for the committed trace fixture:
    uv run examples/poc_activity_triage.py \
        --distractor-compounds 120 \
        --gold-only \
        --context-out /tmp/activity_triage_context_120.txt \
        --gold-out /tmp/activity_triage_gold_120.json
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import string
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rlm import RLM
from rlm.logger import RLMLogger


GENE_PANEL = [
    "NDUFS2",
    "NDUFA9",
    "UQCRC1",
    "ATP5F1A",
    "TOMM20",
    "MAPK1",
    "BRD4",
    "HDAC1",
    "PIK3CA",
    "MTOR",
    "TP53",
    "BCL2",
]

MECHANISM_NOTES = [
    "kinase-biased morphology; weak mitochondrial follow-up recommended",
    "proteasome stress signature, broad toxicity at high dose",
    "HDAC-like chromatin texture shift with low respiratory specificity",
    "reported Complex III / cytochrome bc1 activity; not a Complex I call",
    "microtubule morphology; rounded cells at high concentration",
    "uncurated vendor note, no target confidence assigned",
    "oxidative phosphorylation modulation, target family unresolved",
    "DNA damage response signature in nuclear texture features",
]


def csv_block(name: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError(f"{name} has no rows")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return f"## TABLE {name}\n{buffer.getvalue().strip()}\n## END {name}"


def random_name(rng: random.Random) -> str:
    left = "".join(rng.choices(string.ascii_lowercase, k=rng.randint(5, 8))).title()
    right = rng.choice(["azole", "mycin", "stat", "nib", "vir", "dione", "xan"])
    return f"{left}{right}"


def add_activity_rows(
    rows: list[dict[str, Any]],
    rng: random.Random,
    jcp_id: str,
    qc_plate_id: str,
    *,
    ndufs2_5um: tuple[float, float, float, float] | None = None,
) -> None:
    if ndufs2_5um is None:
        genes = rng.sample(GENE_PANEL, k=5)
    else:
        genes = ["NDUFS2", *rng.sample([gene for gene in GENE_PANEL if gene != "NDUFS2"], k=4)]
    for gene in genes:
        for dose_um in (0.5, 5.0):
            if gene == "NDUFS2" and dose_um == 5.0 and ndufs2_5um is not None:
                activity_map, similarity, consistency, cell_count_z = ndufs2_5um
            else:
                activity_map = rng.uniform(0.08, 0.74)
                similarity = rng.uniform(0.02, 0.66)
                consistency = rng.uniform(0.35, 0.90)
                cell_count_z = rng.uniform(-2.4, 2.4)
            rows.append(
                {
                    "jcp_id": jcp_id,
                    "gene_symbol": gene,
                    "perturbation_type": "compound",
                    "dose_um": dose_um,
                    "activity_map": round(activity_map, 3),
                    "similarity_to_gene_ko": round(similarity, 3),
                    "replicate_count": rng.randint(4, 8),
                    "profile_consistency": round(consistency, 3),
                    "cell_count_z": round(cell_count_z, 3),
                    "qc_plate_id": qc_plate_id,
                    "profile_id": f"PROF_{jcp_id[-5:]}_{gene}_{str(dose_um).replace('.', 'p')}",
                }
            )


def build_packet(seed: int, distractor_compounds: int) -> tuple[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    compounds: list[dict[str, Any]] = []
    activity_rows: list[dict[str, Any]] = []
    plate_qc: list[dict[str, Any]] = []

    for idx in range(1, 25):
        plate_qc.append(
            {
                "qc_plate_id": f"PLATE_{idx:03d}",
                "batch": f"DG-{rng.randint(1, 7):02d}",
                "pass_qc": "true" if rng.random() > 0.12 else "false",
                "artifact_rate": round(rng.uniform(0.01, 0.22), 3),
                "comment": rng.choice(
                    [
                        "within expected range",
                        "borderline focus but accepted",
                        "edge wells elevated",
                        "illumination correction stable",
                    ]
                ),
            }
        )

    curated = [
        {
            "jcp_id": "JCP2022_SYN001",
            "compound_name": "Nadhrestat",
            "target_confidence": 0.95,
            "moa_note": "Complex I / NADH dehydrogenase inhibitor; respiration rescue follow-up passed.",
            "structure_alerts": "none",
            "qc_plate_id": "PLATE_001",
            "ndufs2": (0.931, 0.842, 0.914, -0.42),
        },
        {
            "jcp_id": "JCP2022_SYN002",
            "compound_name": "Oxphosarin",
            "target_confidence": 0.88,
            "moa_note": "Mitochondrial Complex I blocker; orthogonal Seahorse assay confirms NADH arm.",
            "structure_alerts": "none",
            "qc_plate_id": "PLATE_002",
            "ndufs2": (0.887, 0.801, 0.872, -0.65),
        },
        {
            "jcp_id": "JCP2022_SYN003",
            "compound_name": "Ndufexor",
            "target_confidence": 0.79,
            "moa_note": "Weak but specific NADH dehydrogenase signature; Complex I rescue evidence present.",
            "structure_alerts": "none",
            "qc_plate_id": "PLATE_003",
            "ndufs2": (0.821, 0.757, 0.844, -0.31),
        },
        {
            "jcp_id": "JCP2022_TRAP_QC",
            "compound_name": "Failplatein",
            "target_confidence": 0.93,
            "moa_note": "Strong Complex I note, but all activity is on failed plate.",
            "structure_alerts": "none",
            "qc_plate_id": "PLATE_004",
            "ndufs2": (0.970, 0.910, 0.940, -0.21),
        },
        {
            "jcp_id": "JCP2022_TRAP_TOX",
            "compound_name": "Toximycin",
            "target_confidence": 0.91,
            "moa_note": "NADH dehydrogenase inhibitor but cytotoxic morphology dominates.",
            "structure_alerts": "cytotoxic",
            "qc_plate_id": "PLATE_005",
            "ndufs2": (0.952, 0.884, 0.906, -3.12),
        },
        {
            "jcp_id": "JCP2022_TRAP_MOA",
            "compound_name": "Cytochromevir",
            "target_confidence": 0.89,
            "moa_note": "Complex III / cytochrome bc1 inhibitor; high mitochondrial phenotype, wrong target.",
            "structure_alerts": "none",
            "qc_plate_id": "PLATE_006",
            "ndufs2": (0.925, 0.861, 0.881, -0.45),
        },
    ]

    # Pin plate outcomes for the crafted candidates and traps.
    for row in plate_qc:
        if row["qc_plate_id"] in {"PLATE_001", "PLATE_002", "PLATE_003", "PLATE_005", "PLATE_006"}:
            row["pass_qc"] = "true"
            row["artifact_rate"] = 0.031
        if row["qc_plate_id"] == "PLATE_004":
            row["pass_qc"] = "false"
            row["artifact_rate"] = 0.371
            row["comment"] = "failed focus and debris QC"

    for item in curated:
        compounds.append(
            {
                "jcp_id": item["jcp_id"],
                "compound_name": item["compound_name"],
                "inchi_key_stub": f"SYN-{item['jcp_id'][-6:]}",
                "primary_target": "synthetic annotation",
                "target_confidence": item["target_confidence"],
                "moa_note": item["moa_note"],
                "structure_alerts": item["structure_alerts"],
            }
        )
        add_activity_rows(
            activity_rows,
            rng,
            item["jcp_id"],
            item["qc_plate_id"],
            ndufs2_5um=item["ndufs2"],
        )

    for idx in range(distractor_compounds):
        jcp_id = f"JCP2022_RND{idx:04d}"
        compounds.append(
            {
                "jcp_id": jcp_id,
                "compound_name": random_name(rng),
                "inchi_key_stub": f"RND-{idx:05d}",
                "primary_target": rng.choice(["kinase", "epigenetic", "mitochondrial", "unknown", "GPCR"]),
                "target_confidence": round(rng.uniform(0.15, 0.92), 3),
                "moa_note": rng.choice(MECHANISM_NOTES),
                "structure_alerts": rng.choice(["none", "none", "none", "pains", "cytotoxic"]),
            }
        )
        add_activity_rows(
            activity_rows,
            rng,
            jcp_id,
            rng.choice(plate_qc)["qc_plate_id"],
            ndufs2_5um=None,
        )

    rng.shuffle(compounds)
    rng.shuffle(activity_rows)

    sections = [
        "# Synthetic JUMP activity-results packet",
        "The data are synthetic but intentionally shaped like jx/JUMP metadata.",
        "Join keys: compound_metadata.jcp_id -> activity_results.jcp_id; "
        "activity_results.qc_plate_id -> plate_qc.qc_plate_id.",
        csv_block("compound_metadata", compounds),
        csv_block("activity_results", activity_rows),
        csv_block("plate_qc", plate_qc),
    ]
    context = "\n\n".join(sections)
    return context, compute_gold(compounds, activity_rows, plate_qc)


def supports_complex_i(note: str) -> bool:
    normalized = note.lower()
    return (
        ("complex i" in normalized or "nadh dehydrogenase" in normalized)
        and "complex iii" not in normalized
        and "wrong target" not in normalized
    )


def compute_gold(
    compounds: list[dict[str, Any]],
    activity_rows: list[dict[str, Any]],
    plate_qc: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    compounds_by_id = {row["jcp_id"]: row for row in compounds}
    qc_by_id = {row["qc_plate_id"]: row for row in plate_qc}
    candidates: list[dict[str, Any]] = []
    for row in activity_rows:
        if row["gene_symbol"] != "NDUFS2" or float(row["dose_um"]) != 5.0:
            continue
        compound = compounds_by_id[row["jcp_id"]]
        qc = qc_by_id[row["qc_plate_id"]]
        if qc["pass_qc"] != "true":
            continue
        if compound["structure_alerts"] in {"cytotoxic", "pains"}:
            continue
        if float(compound["target_confidence"]) < 0.60:
            continue
        if not supports_complex_i(compound["moa_note"]):
            continue
        if float(row["activity_map"]) < 0.72:
            continue
        if float(row["similarity_to_gene_ko"]) < 0.64:
            continue
        if float(row["profile_consistency"]) < 0.75:
            continue
        if abs(float(row["cell_count_z"])) > 2.0:
            continue
        score = (
            0.45 * float(row["activity_map"])
            + 0.35 * float(row["similarity_to_gene_ko"])
            + 0.20 * float(row["profile_consistency"])
        )
        candidates.append(
            {
                "jcp_id": row["jcp_id"],
                "compound_name": compound["compound_name"],
                "score": round(score, 4),
                "activity_map": float(row["activity_map"]),
                "similarity_to_NDUFS2_KO": float(row["similarity_to_gene_ko"]),
                "profile_consistency": float(row["profile_consistency"]),
                "moa_note": compound["moa_note"],
                "qc_plate_id": row["qc_plate_id"],
            }
        )
    return sorted(candidates, key=lambda item: item["score"], reverse=True)[:3]


def root_prompt(subcall_review: bool) -> str:
    subcall_instruction = (
        "After computing the deterministic shortlist in Python, use "
        "rlm_query_batched() to ask child RLMs whether each shortlisted "
        "moa_note truly supports Complex I / NADH dehydrogenase rather than "
        "a nearby mitochondrial mechanism. Use those reviews only as a final "
        "sanity check; the numeric filtering and ranking must be done in Python."
        if subcall_review
        else "Do not call child LLMs unless you genuinely need semantic help; "
        "the task should be solvable by parsing and filtering in Python."
    )
    return f"""
The variable `context` contains a synthetic JUMP-like activity-results packet
with three CSV tables: compound_metadata, activity_results, and plate_qc.
Some CSV fields contain quoted commas, so use Python's csv module or equivalent
structured parsing rather than splitting rows on commas by hand.

Find the top 3 eligible compound candidates for a Complex I / NDUFS2
phenocopy follow-up.

Eligibility rules:
- Use only activity_results rows where gene_symbol is NDUFS2 and dose_um is 5.0.
- Join compound_metadata by jcp_id and plate_qc by qc_plate_id.
- Require plate_qc.pass_qc == true.
- Exclude compounds with structure_alerts equal to cytotoxic or pains.
- Require target_confidence >= 0.60.
- The moa_note must support Complex I or NADH dehydrogenase, not Complex III.
- Require activity_map >= 0.72.
- Require similarity_to_gene_ko >= 0.64.
- Require profile_consistency >= 0.75.
- Require absolute cell_count_z <= 2.0.

Rank eligible rows by:
score = 0.45 * activity_map + 0.35 * similarity_to_gene_ko + 0.20 * profile_consistency

{subcall_instruction}

Return ONLY compact JSON, no markdown. The JSON must be an array of three
objects with keys: jcp_id, compound_name, score, activity_map,
similarity_to_NDUFS2_KO, profile_consistency, qc_plate_id, reason.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--distractor-compounds", type=int, default=900)
    parser.add_argument("--backend", default="openrouter")
    parser.add_argument("--model", default="anthropic/claude-sonnet-4.5")
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--log-dir", type=Path, default=Path("./traces"))
    parser.add_argument(
        "--subcall-review",
        action="store_true",
        help="Ask the root model to use rlm_query_batched() on the final shortlist.",
    )
    parser.add_argument(
        "--gold-only",
        action="store_true",
        help="Build the packet and print the deterministic gold answer without calling an API.",
    )
    parser.add_argument(
        "--context-out",
        type=Path,
        default=None,
        help="Optional path to write the generated context packet.",
    )
    parser.add_argument(
        "--gold-out",
        type=Path,
        default=None,
        help="Optional path to write the deterministic gold answer JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv()
    context, gold = build_packet(
        seed=args.seed,
        distractor_compounds=args.distractor_compounds,
    )

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

    backend_kwargs = {
        "api_key": api_key,
        "model_name": args.model,
    }

    rlm = RLM(
        backend=args.backend,
        backend_kwargs=backend_kwargs,
        environment="local",
        max_depth=args.max_depth,
        max_iterations=args.max_iterations,
        logger=RLMLogger(log_dir=str(args.log_dir)),
        verbose=True,
    )

    result = rlm.completion(
        prompt=context,
        root_prompt=root_prompt(subcall_review=args.subcall_review),
    )

    print()
    print("Model answer:")
    print(result.response)
    print()
    print("Gold answer:")
    print(json.dumps(gold, indent=2))
    try:
        model_answer = json.loads(result.response)
        model_ids = [row["jcp_id"] for row in model_answer]
        gold_ids = [row["jcp_id"] for row in gold]
        print()
        print(f"ID order matches gold: {model_ids == gold_ids}")
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        print()
        print(f"Could not parse model answer as expected JSON: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
