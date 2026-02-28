#!/usr/bin/env python3
"""
Atomic fact counting for assistant responses.

Decomposes assistant responses into atomic facts using OpenAI.

Usage:
    # Test on small sample (default 10)
    python src/fact_counting.py --sample 10

    # Full run
    python src/fact_counting.py --sample 0
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT = (
    PROJECT_DIR / "data" / "wildchat_data" / "filtered_data" / "combined_filtered.csv"
)
RAW_DATA_DIR = PROJECT_DIR / "data" / "wildchat_data" / "raw_data"
OUTPUT_DIR = PROJECT_DIR / "data" / "wildchat_data" / "fact_counted"

DEFAULT_TEXT_COL = "user_message"
DEFAULT_SAMPLE_SIZE = 10


# ============================================================
# DATA LOADING
# ============================================================


def load_raw_conversations() -> dict[str, list[dict]]:
    """Load raw parquet files and create conversation_hash -> conversation mapping."""
    print("Loading raw conversation data...")
    conversations = {}

    parquet_files = sorted(RAW_DATA_DIR.glob("*.parquet"))
    for pf in tqdm(parquet_files, desc="Loading parquet files"):
        df = pd.read_parquet(pf)
        for _, row in df.iterrows():
            conv_hash = row["conversation_hash"]
            conversations[conv_hash] = row["conversation"]

    print(f"Loaded {len(conversations)} conversations")
    return conversations


def extract_assistant_response(
    conversation: list[dict], user_message: str
) -> str | None:
    """Find the assistant response following a specific user message."""
    for i, turn in enumerate(conversation):
        if turn.get("role") == "user":
            content = turn.get("content", "")
            if content[:100] == user_message[:100]:
                for j in range(i + 1, len(conversation)):
                    if conversation[j].get("role") == "assistant":
                        return conversation[j].get("content", "")
    return None


def join_with_responses(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Join filtered data with raw data to get assistant responses."""
    conversations = load_raw_conversations()

    print("\nExtracting assistant responses...")
    responses = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Matching responses"):
        conv_id = row["conversation_id"]
        user_msg = row[text_col]

        conversation = conversations.get(conv_id)
        if conversation is not None:
            response = extract_assistant_response(conversation, user_msg)
            responses.append(response)
        else:
            responses.append(None)

    df = df.copy()
    df["assistant_response"] = responses

    found = sum(1 for r in responses if r is not None)
    print(f"Found {found}/{len(df)} assistant responses")

    return df


# ============================================================
# ATOMIC FACT DECOMPOSITION
# ============================================================


def decompose_into_atomic_facts(text: str, client) -> list[str]:
    """Decompose a text into atomic facts using OpenAI."""
    prompt = """Break down the following text into a list of atomic facts. 
Each atomic fact should be:
- A single, independent factual claim
- Self-contained (understandable without context)
- Verifiable against a knowledge source

Return ONLY a JSON array of strings, one fact per element.
Keep each fact concise but complete.

Text:
\"\"\"
{text}
\"\"\"

Return format: ["fact 1", "fact 2", ...]"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt.format(text=text[:4000])}],
            temperature=0,
            max_tokens=2000,
        )
        content = response.choices[0].message.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        facts = json.loads(content)
        return facts if isinstance(facts, list) else []
    except Exception as e:
        print(f"Fact decomposition error: {e}")
        return []


def count_facts(
    df: pd.DataFrame,
    client,
    output_path: Path,
    checkpoint_interval: int = 50,
) -> pd.DataFrame:
    """Count atomic facts in each assistant response with checkpointing.

    Args:
        df: DataFrame with assistant_response column
        client: OpenAI client
        output_path: Path to save checkpoints
        checkpoint_interval: Save progress every N rows
    """
    print("\n=== Counting Atomic Facts ===")

    # Initialize columns if not present
    if "num_facts" not in df.columns:
        df["num_facts"] = None
    if "facts" not in df.columns:
        df["facts"] = None

    # Find rows that need processing (no facts yet)
    needs_processing = df["facts"].isna() | (df["facts"] == "")
    to_process = df[needs_processing].index.tolist()

    already_done = len(df) - len(to_process)
    if already_done > 0:
        print(
            f"Resuming: {already_done} already processed, {len(to_process)} remaining"
        )

    if len(to_process) == 0:
        print("All rows already processed!")
        return df

    processed_count = 0

    for idx in tqdm(to_process, desc="Counting facts"):
        row = df.loc[idx]
        response = row.get("assistant_response")

        if not isinstance(response, str) or not response.strip():
            df.at[idx, "num_facts"] = 0
            df.at[idx, "facts"] = "[]"
        else:
            facts = decompose_into_atomic_facts(response, client)
            df.at[idx, "num_facts"] = len(facts)
            df.at[idx, "facts"] = json.dumps(facts)

        processed_count += 1

        # Checkpoint every N rows
        if processed_count % checkpoint_interval == 0:
            df.to_csv(output_path, index=False)
            print(
                f"\n  Checkpoint saved ({processed_count + already_done}/{len(df)} done)"
            )

        # Rate limiting
        time.sleep(0.3)

    # Final save
    df.to_csv(output_path, index=False)
    print(f"\nFinal save: {len(df)} rows")

    return df


# ============================================================
# ARGUMENT PARSING
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count atomic facts in assistant responses",
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=f"Input CSV file (default: {DEFAULT_INPUT.name})",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV filename",
    )

    parser.add_argument(
        "--text-col",
        type=str,
        default=DEFAULT_TEXT_COL,
        help=f"Column containing questions (default: {DEFAULT_TEXT_COL})",
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of samples to process (default: {DEFAULT_SAMPLE_SIZE}, use 0 for all)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================


def main():
    args = parse_args()

    # Get OpenAI API key
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        print("ERROR: No OpenAI API key found. Set OPENAI_API_KEY env var.")
        return

    # Determine input path
    if args.input:
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = PROJECT_DIR / args.input
    else:
        input_path = DEFAULT_INPUT

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return

    # Setup output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_path = OUTPUT_DIR / args.output
    else:
        output_path = OUTPUT_DIR / f"fact_counted_{input_path.stem}.csv"

    # Check if we should resume from existing output
    if output_path.exists():
        print(f"Found existing output file: {output_path}")
        print("Resuming from checkpoint...")
        df = pd.read_csv(output_path)
        print(f"Loaded {len(df)} rows from checkpoint")

        # Check how many are already processed
        already_done = df["facts"].notna() & (df["facts"] != "")
        print(f"  {already_done.sum()} already have facts extracted")
    else:
        # Load fresh data
        print(f"Loading data from: {input_path}")
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} messages")

        text_col = args.text_col
        if text_col not in df.columns:
            print(f"ERROR: Column '{text_col}' not found in data")
            return

        # Sample if requested (only on fresh start)
        if args.sample > 0 and args.sample < len(df):
            print(f"\nSampling {args.sample} questions (seed={args.seed})")
            df = df.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)

        print(f"\nProcessing {len(df)} questions...")

        # Join with raw data to get assistant responses
        df = join_with_responses(df, text_col)

    # Check for assistant responses
    if "assistant_response" not in df.columns:
        text_col = args.text_col
        df = join_with_responses(df, text_col)

    has_response = df["assistant_response"].notna()
    print(f"\n{has_response.sum()} questions have assistant responses")

    if has_response.sum() == 0:
        print("ERROR: No assistant responses found")
        return

    # Count facts with checkpointing
    client = OpenAI(api_key=openai_key)
    df = count_facts(df, client, output_path, checkpoint_interval=50)

    # Print summary
    print("\n=== FACT COUNT RESULTS ===")
    valid_facts = df["num_facts"].dropna()
    print(f"Total responses: {len(df)}")
    print(f"Processed: {len(valid_facts)}")
    if len(valid_facts) > 0:
        print(f"Avg facts per response: {valid_facts.mean():.1f}")
        print(f"Median facts per response: {valid_facts.median():.0f}")
        print(f"Max facts: {valid_facts.max():.0f}")
        print(f"Min facts: {valid_facts.min():.0f}")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
