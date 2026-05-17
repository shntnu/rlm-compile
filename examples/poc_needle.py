#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "rlms",
#     "python-dotenv",
# ]
# ///
"""Smallest possible RLM POC: needle-in-haystack with OpenRouter.

Build a ~5K-line haystack of random text with one line containing
SECRET_NUMBER=<digits>. Ask the RLM to find it. The root LM never sees the
haystack directly - it's loaded into a REPL variable that the LM greps.

Run:
    uv run examples/poc_needle.py
"""
import os
import random
import string

from dotenv import load_dotenv
from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

random.seed(a=42)
secret_number = random.randint(a=100_000_000, b=999_999_999)
filler_lines = [
    "".join(random.choices(population=string.ascii_lowercase + " ", k=120))
    for _ in range(5_000)
]
insert_at = random.randint(a=len(filler_lines) // 3, b=2 * len(filler_lines) // 3)
filler_lines.insert(insert_at, f"SECRET_NUMBER={secret_number}")
haystack = "\n".join(filler_lines)

print(f"Haystack: {len(filler_lines)} lines, ~{len(haystack):,} chars")
print(f"Secret inserted at line {insert_at}, value {secret_number}")
print()

rlm = RLM(
    backend="openrouter",
    backend_kwargs={
        "api_key": os.getenv(key="OPENROUTER_API_KEY"),
        "model_name": "anthropic/claude-sonnet-4.5",
    },
    environment="local",
    max_iterations=10,
    logger=RLMLogger(log_dir="./traces"),
    verbose=True,
)

result = rlm.completion(
    prompt=haystack,
    root_prompt=(
        "The variable `context` holds ~5k lines of random text with a single "
        "line matching the pattern SECRET_NUMBER=<digits>. Find and return "
        "ONLY the numeric value."
    ),
)

print()
print(f"Model found:    {result.response}")
print(f"Actual number:  {secret_number}")
print(f"Correct:        {str(secret_number) in result.response}")
