"""
Extract all user messages from WildChat parquet files.

This script:
1. Reads each parquet file from raw_data
2. Filters to English language conversations only
3. Collects every user-authored message per conversation (not just the first)
4. Removes duplicate user messages
5. Removes messages longer than 10 sentences
6. Removes messages containing 'midjourney'
7. Saves a CSV with conversation_id, user_id, message_id, and user_message

Usage:
    python src/first_conversation_english.py
"""

import re
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_DIR = PROJECT_DIR / "data" / "wildchat_data" / "raw_data"
OUTPUT_DIR = PROJECT_DIR / "data" / "wildchat_data" / "english_data"


def iter_user_messages(conversation: list):
    """Yield (message_index, content) pairs for every user-authored message."""
    for idx, msg in enumerate(conversation):
        if msg.get("role") == "user":
            yield idx, msg.get("content", "")


def count_sentences(text: str) -> int:
    """Count the number of sentences in a text."""
    if pd.isna(text) or not text:
        return 0
    # Split on . ! ? followed by space or end of string
    sentences = re.split(r"[.!?]+(?:\s|$)", str(text))
    # Filter out empty strings
    return len([s for s in sentences if s.strip()])


def process_parquet_file(input_path: Path, output_path: Path) -> int:
    """Process a single parquet file and save filtered CSV.

    Returns:
        Number of rows in output CSV
    """
    print(f"Processing: {input_path.name}")

    # Load parquet
    df = pd.read_parquet(input_path)
    print(f"  Loaded {len(df)} conversations")

    # Filter to English only
    df_english = df[df["language"] == "English"].copy()
    print(f"  Filtered to {len(df_english)} English conversations")

    # Extract every user message for each conversation
    records = []
    for _, row in df_english.iterrows():
        conversation = row["conversation"]
        if conversation is None or len(conversation) == 0:
            continue

        for msg_index, user_message in iter_user_messages(conversation):
            records.append(
                {
                    "conversation_id": row["conversation_hash"],
                    "user_id": row["hashed_ip"],
                    "message_id": f"{row['conversation_hash']}_{msg_index}",
                    "user_message": user_message,
                }
            )

    # Create output DataFrame
    output_df = pd.DataFrame(records)

    # Remove duplicate user messages
    before_dedup = len(output_df)
    output_df = output_df.drop_duplicates(subset=["user_message"], keep="first")
    print(f"  Removed {before_dedup - len(output_df)} duplicate messages")

    # Remove messages containing 'midjourney' (case-insensitive)
    before_midjourney = len(output_df)
    output_df = output_df[
        ~output_df["user_message"].str.lower().str.contains("midjourney", na=False)
    ]
    print(f"  Removed {before_midjourney - len(output_df)} midjourney messages")

    # Remove messages longer than 10 sentences
    output_df["sentence_count"] = output_df["user_message"].apply(count_sentences)
    before_length = len(output_df)
    output_df = output_df[output_df["sentence_count"] <= 10]
    print(f"  Removed {before_length - len(output_df)} messages with >10 sentences")
    output_df = output_df.drop(columns=["sentence_count"])

    # Save
    output_df.to_csv(output_path, index=False)
    print(f"  Saved {len(output_df)} rows to {output_path.name}")

    return len(output_df)


def main():
    # Create output directory if needed
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all parquet files
    parquet_files = sorted(INPUT_DIR.glob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files in {INPUT_DIR}\n")

    if not parquet_files:
        print("No parquet files found!")
        return

    total_rows = 0
    for input_path in parquet_files:
        # Generate output filename (same name but .csv)
        output_name = input_path.stem + ".csv"
        output_path = OUTPUT_DIR / output_name

        rows = process_parquet_file(input_path, output_path)
        total_rows += rows
        print()

    print(f"Done! Total rows saved: {total_rows}")


if __name__ == "__main__":
    main()
