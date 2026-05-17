#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "rlms",
#     "python-dotenv",
# ]
# ///
"""Small live RLM POC: rank clean assay hits from one CSV table.

This example actually runs RLM and writes a real RLMLogger JSONL trace under
`./traces`. Use `--gold-only` to validate the deterministic answer without
calling an API.

Run without spending tokens:
    uv run examples/poc_assay_hit_rank.py --gold-only

Run a real RLM trace:
    uv run examples/poc_assay_hit_rank.py

Then compile the generated trace:
    uv run compile.py traces/<new_trace>.jsonl.gz compiled/assay_hit_rank.py
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rlm import RLM
from rlm.logger import RLMLogger


ROWS = [
    {"compound_id": "JCP_MINI_001", "target": "NDUFS2", "activity": 0.91, "toxicity": 0.08, "pass_qc": "true"},
    {"compound_id": "JCP_MINI_002", "target": "NDUFS2", "activity": 0.88, "toxicity": 0.12, "pass_qc": "true"},
    {"compound_id": "JCP_MINI_003", "target": "BRD4", "activity": 0.94, "toxicity": 0.05, "pass_qc": "true"},
    {"compound_id": "JCP_MINI_004", "target": "NDUFS2", "activity": 0.93, "toxicity": 0.31, "pass_qc": "true"},
    {"compound_id": "JCP_MINI_005", "target": "NDUFS2", "activity": 0.83, "toxicity": 0.04, "pass_qc": "false"},
    {"compound_id": "JCP_MINI_006", "target": "NDUFS2", "activity": 0.79, "toxicity": 0.03, "pass_qc": "true"},
]


def build_context() -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["compound_id", "target", "activity", "toxicity", "pass_qc"],
    )
    writer.writeheader()
    writer.writerows(ROWS)
    return buffer.getvalue()


def compute_gold(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hits = []
    for row in rows:
        if row["target"] != "NDUFS2":
            continue
        if row["pass_qc"] != "true":
            continue
        if float(row["toxicity"]) > 0.15:
            continue
        activity = float(row["activity"])
        toxicity = float(row["toxicity"])
        hits.append(
            {
                "compound_id": str(row["compound_id"]),
                "activity": activity,
                "toxicity": toxicity,
                "score": round(activity - 0.5 * toxicity, 3),
            }
        )
    return sorted(hits, key=lambda row: (-row["score"], row["compound_id"]))[:3]


def root_prompt() -> str:
    return """The variable `context` contains one CSV table with columns:
compound_id,target,activity,toxicity,pass_qc.

Find the top 3 clean NDUFS2 assay hits.

Rules:
- Keep only rows where target == NDUFS2.
- Keep only rows where pass_qc == true.
- Exclude rows where toxicity > 0.15.
- score = activity - 0.5 * toxicity.
- Round score to 3 decimal places.
- Sort by descending score, then compound_id.

Return ONLY compact JSON, no markdown. The JSON must be an array of objects
with keys: compound_id, activity, toxicity, score."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="openrouter")
    parser.add_argument("--model", default="anthropic/claude-sonnet-4.5")
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--log-dir", type=Path, default=Path("./traces"))
    parser.add_argument("--gold-only", action="store_true", help="Print gold answer without calling an API.")
    parser.add_argument("--context-out", type=Path, default=None)
    parser.add_argument("--gold-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv()
    context = build_context()
    gold = compute_gold(ROWS)

    print("Context:")
    print(context, end="")
    print()
    print(f"Gold answer: {json.dumps(gold, indent=2)}")
    print()

    if args.context_out is not None:
        args.context_out.write_text(context)
        print(f"Wrote context packet: {args.context_out}")
    if args.gold_out is not None:
        args.gold_out.write_text(json.dumps(gold, indent=2) + "\n")
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
        print()
        print(f"Matches gold: {model_answer == gold}")
    except json.JSONDecodeError as exc:
        print()
        print(f"Could not parse model answer as JSON: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
