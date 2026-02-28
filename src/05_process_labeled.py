"""Process labeled WildChat data to filter for historical fact-seeking conversations.

This script works with both Together-batch outputs and DeepSeek outputs:
1. Reads labeled CSV file(s) from labelled_data/
2. Extracts the label from llm_response (HISTORY_FACT_SEEKING vs NOT_HISTORY_FACT_SEEKING)
3. Saves only the rows labeled as HISTORY_FACT_SEEKING to a single output file

Usage:
    # Process all files in labelled_data/ (outputs combined_filtered.csv)
    python src/process_labeled.py

    # Process a single file
    python src/process_labeled.py --input deepseek_train-00000-of-00014.csv

    # Custom output filename
    python src/process_labeled.py --output my_output.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_DIR = PROJECT_DIR / "data" / "wildchat_data" / "labelled_data"
OUTPUT_DIR = PROJECT_DIR / "data" / "wildchat_data" / "filtered_data"

FLAG_PATTERNS = {
    "fact_seeking": re.compile(
        r"['\"]fact_seeking['\"]\s*:\s*(true|false|1|0)", re.IGNORECASE
    ),
    "historical_topic": re.compile(
        r"['\"]historical_topic['\"]\s*:\s*(true|false|1|0)", re.IGNORECASE
    ),
}


def parse_llm_response(response: str) -> dict:
    """Parse the LLM response JSON string.

    Args:
        response: JSON string from the LLM

    Returns:
        Parsed dict with label, reason, uncertainty
    """
    if pd.isna(response) or not response:
        return {"label": None, "reason": None, "uncertainty": None}

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Handle cases where response might have extra text
        return {
            "label": None,
            "reason": None,
            "uncertainty": None,
            "parse_error": response,
        }


def extract_binary_label(value) -> str | None:
    """Normalize any value to a binary label ('0'/'1') if possible."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (int, float)) and value in {0, 1}:
        return str(int(value))

    text = str(value).strip()
    if not text:
        return None
    if text in {"0", "1"}:
        return text

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        for key in ("label", "response", "answer", "classification"):
            candidate = extract_binary_label(parsed.get(key))
            if candidate is not None:
                return candidate
        return None

    if parsed is not None:
        return extract_binary_label(parsed)

    for ch in text:
        if ch in {"0", "1"}:
            return ch
    return None


def extract_flag_from_response(response, field: str) -> bool | None:
    """Extract a boolean flag (True/False) for a given field from the response text."""
    if response is None:
        return None
    if isinstance(response, float) and math.isnan(response):
        return None

    text = str(response).strip()
    if not text:
        return None

    candidate = text
    if text.count("{") > text.count("}"):
        candidate = text + "}"

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, dict) and field in data:
        value = data[field]
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in {0, 1}:
            return bool(value)
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in {"true", "1"}:
                return True
            if lowered in {"false", "0"}:
                return False

    pattern = FLAG_PATTERNS.get(field)
    if pattern:
        match = pattern.search(text)
        if match:
            lowered = match.group(1).lower()
            return lowered in {"true", "1"}

    return None


def extract_historical_label(response) -> bool | None:
    """Extract historical label from llm_response.

    Supports formats:
    - {"label": "HISTORY_FACT_SEEKING"} -> True
    - {"label": "NOT_HISTORY_FACT_SEEKING"} -> False
    - {"historical_topic": true/false} -> True/False
    """
    if response is None:
        return None
    if isinstance(response, float) and math.isnan(response):
        return None

    text = str(response).strip()
    if not text:
        return None

    # Check for HISTORY_FACT_SEEKING / NOT_HISTORY_FACT_SEEKING string labels
    if "NOT_HISTORY_FACT_SEEKING" in text or "NOT_HISTORY" in text:
        return False
    if "HISTORY_FACT_SEEKING" in text or "HISTORY" in text:
        return True

    # Fall back to boolean field extraction
    return extract_flag_from_response(response, "historical_topic")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter labeled WildChat data for historical fact-seeking conversations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input CSV filename from labelled_data/ directory (if omitted, processes all files)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV filename (default: combined_filtered.csv when processing all files)",
    )

    return parser.parse_args()


def process_single_file(input_path: Path) -> pd.DataFrame | None:
    """Process a single labeled CSV file and return filtered DataFrame."""
    if not input_path.exists():
        print(f"WARNING: Input file not found: {input_path}")
        return None

    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df)} messages from {input_path.name}")

    # Derive binary labels
    if "label" in df.columns:
        label_series = df["label"].apply(extract_binary_label)
    else:
        label_series = pd.Series([None] * len(df))

    if label_series.isna().all() and "llm_response" in df.columns:
        label_series = df["llm_response"].apply(extract_binary_label)

    df["binary_label"] = label_series

    if "llm_response" in df.columns:
        df["historical_flag"] = df["llm_response"].apply(extract_historical_label)
    else:
        df["historical_flag"] = None

    # Fallback: if historical_flag couldn't be extracted, use binary_label
    if df["historical_flag"].isna().all() and df["binary_label"].notna().any():
        df["historical_flag"] = df["binary_label"].map(lambda v: v == "1")

    historical_mask = df["historical_flag"] == True
    filtered_df = df[historical_mask].copy()

    # Add source file column for traceability
    filtered_df["source_file"] = input_path.name

    print(f"    -> {len(filtered_df)} HISTORY_FACT_SEEKING messages")
    return filtered_df


def main():
    args = parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine input files
    if args.input:
        # Single file mode
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = INPUT_DIR / input_path
        input_files = [input_path]
        default_output = f"{input_path.stem}_filtered.csv"
    else:
        # Process all CSV files in labelled_data folder
        input_files = sorted(INPUT_DIR.glob("*.csv"))
        if not input_files:
            print(f"ERROR: No CSV files found in {INPUT_DIR}")
            return
        print(f"Found {len(input_files)} CSV files in {INPUT_DIR}")
        default_output = "combined_filtered.csv"

    output_path = OUTPUT_DIR / (args.output if args.output else default_output)

    print(f"\nProcessing {len(input_files)} file(s)...")
    print("-" * 60)

    # Process all files and collect filtered DataFrames
    filtered_dfs = []
    for input_path in input_files:
        result = process_single_file(input_path)
        if result is not None and len(result) > 0:
            filtered_dfs.append(result)

    print("-" * 60)

    if not filtered_dfs:
        print("No historical fact-seeking messages found in any file.")
        return

    # Combine all filtered DataFrames
    combined_df = pd.concat(filtered_dfs, ignore_index=True)
    print(
        f"\nTotal: {len(combined_df)} HISTORY_FACT_SEEKING messages from {len(filtered_dfs)} file(s)"
    )

    # Save to CSV
    combined_df.to_csv(output_path, index=False)
    print(f"Saved filtered data to: {output_path}")


if __name__ == "__main__":
    main()
