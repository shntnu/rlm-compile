#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Tiny bioinformatics trace fixture: filter differential-expression hits.

This is intentionally smaller than the variant-prioritization POC. It creates a
single CSV table and a successful JSONL trace whose recovered program is about
10 lines of Python: parse rows, filter liver genes by log2 fold-change and
q-value, sort, and print compact JSON.

The fixture is deterministic and stdlib-only, so it is useful for testing the
trace-to-program compiler without making an API call.

Run:
    uv run examples/poc_gene_expression_hits.py

Write the replay inputs and trace fixture:
    uv run examples/poc_gene_expression_hits.py \
        --context-out /tmp/gene_expression_hits_context.csv \
        --gold-out /tmp/gene_expression_hits_gold.json \
        --trace-out traces/rlm_2026-05-16_08-12-00_5f4c2a91.jsonl.gz
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROWS = [
    {"gene": "CYP3A4", "tissue": "liver", "log2_fc": 2.4, "q_value": 0.0008},
    {"gene": "CRP", "tissue": "liver", "log2_fc": 2.0, "q_value": 0.08},
    {"gene": "ALB", "tissue": "liver", "log2_fc": 1.8, "q_value": 0.012},
    {"gene": "APOB", "tissue": "liver", "log2_fc": 1.6, "q_value": 0.041},
    {"gene": "MT-CO1", "tissue": "liver", "log2_fc": 1.2, "q_value": 0.003},
    {"gene": "HBB", "tissue": "blood", "log2_fc": 5.1, "q_value": 0.0001},
]

TRACE_CODE = """import csv, io, json
rows = list(csv.DictReader(io.StringIO(context)))
hits = [
    {"gene": r["gene"], "log2_fc": float(r["log2_fc"]), "q_value": float(r["q_value"])}
    for r in rows
    if r["tissue"] == "liver" and float(r["log2_fc"]) >= 1.5 and float(r["q_value"]) <= 0.05
]
hits.sort(key=lambda row: (-row["log2_fc"], row["q_value"], row["gene"]))
final_answer = json.dumps(hits, separators=(",", ":"))
print(final_answer)"""

TRACE_NAME = "rlm_2026-05-16_08-12-00_5f4c2a91.jsonl.gz"


def build_context() -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["gene", "tissue", "log2_fc", "q_value"])
    writer.writeheader()
    writer.writerows(ROWS)
    return buffer.getvalue()


def compute_gold(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hits = [
        {
            "gene": str(row["gene"]),
            "log2_fc": float(row["log2_fc"]),
            "q_value": float(row["q_value"]),
        }
        for row in rows
        if row["tissue"] == "liver"
        and float(row["log2_fc"]) >= 1.5
        and float(row["q_value"]) <= 0.05
    ]
    return sorted(hits, key=lambda row: (-row["log2_fc"], row["q_value"], row["gene"]))


def build_trace(final_answer: str) -> str:
    record = {
        "type": "iteration",
        "iteration": 1,
        "timestamp": datetime(2026, 5, 16, 8, 12, 0).isoformat(),
        "prompt": [
            {
                "role": "user",
                "content": (
                    "The variable context contains a CSV table with columns "
                    "gene,tissue,log2_fc,q_value. Return compact JSON for liver "
                    "genes with log2_fc >= 1.5 and q_value <= 0.05, sorted by "
                    "descending log2_fc."
                ),
            }
        ],
        "response": f"```repl\n{TRACE_CODE}\n```",
        "code_blocks": [
            {
                "code": TRACE_CODE,
                "result": {
                    "stdout": final_answer + "\n",
                    "stderr": "",
                    "rlm_calls": [],
                },
            }
        ],
        "final_answer": final_answer,
        "iteration_time": 0.01,
    }
    return json.dumps(record, separators=(",", ":")) + "\n"


def write_text_maybe_gzip(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                compressed.write(text.encode("utf-8"))
        return
    path.write_text(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-out", type=Path, default=None)
    parser.add_argument("--gold-out", type=Path, default=None)
    parser.add_argument("--trace-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    context = build_context()
    gold = compute_gold(ROWS)
    final_answer = json.dumps(gold, separators=(",", ":"))

    print("Context:")
    print(context, end="")
    print()
    print("Gold answer:")
    print(json.dumps(gold, indent=2))

    if args.context_out is not None:
        args.context_out.write_text(context)
        print(f"Wrote context packet: {args.context_out}")
    if args.gold_out is not None:
        args.gold_out.write_text(json.dumps(gold, indent=2) + "\n")
        print(f"Wrote gold answer: {args.gold_out}")
    if args.trace_out is not None:
        write_text_maybe_gzip(args.trace_out, build_trace(final_answer))
        print(f"Wrote trace fixture: {args.trace_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
