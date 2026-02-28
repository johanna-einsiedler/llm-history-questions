"""Label keyword-filtered user messages with DeepSeek's API.

This script:
1. Loads a CSV of user messages from data/wildchat_data/keyword_filtered (or a given path)
2. Uses the historical fact-seeking prompt as the system message
3. Sends each user message (as-is) to DeepSeek's chat completion API
4. Stores the raw response plus a parsed binary label per message
5. Writes the labeled CSV to data/wildchat_data/labelled_data/ (resuming from
    any partial file with the same output name and prompt)

Usage:
  python src/label_deepseek.py --input train-00000-of-00014.csv
  python src/label_deepseek.py --input train-00000-of-00014.csv --limit 200 --model deepseek-chat
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import find_dotenv, load_dotenv

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_DIR = PROJECT_DIR / "data" / "wildchat_data" / "keyword_filtered"
OUTPUT_DIR = PROJECT_DIR / "data" / "wildchat_data" / "labelled_data"
PROMPT_PATH = PROJECT_DIR / "prompts" / "prompt_historical_political.txt"

DEFAULT_MODEL = "deepseek-chat"
API_URL = "https://api.deepseek.com/v1/chat/completions"

INPUT_COST_CACHE_HIT = 0.028 / 1_000_000
INPUT_COST_CACHE_MISS = 0.28 / 1_000_000
OUTPUT_COST = 0.42 / 1_000_000


def load_api_key() -> str:
    """Load the DeepSeek API key from the environment."""
    dotenv_path = find_dotenv()
    if dotenv_path:
        load_dotenv(dotenv_path)
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing DEEPSEEK_API_KEY in environment. Set it in your .env file."
        )
    return api_key


def call_deepseek(
    system_prompt: str,
    user_message: str,
    api_key: str,
    model: str,
    max_retries: int = 5,
) -> tuple[str, dict]:
    """Send a prompt to DeepSeek and return the assistant content and usage stats."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0,
        "max_tokens": 16,
    }

    for attempt in range(1, max_retries + 1):
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError("DeepSeek returned no choices")
            usage = data.get("usage", {})
            return choices[0]["message"]["content"].strip(), usage

        if response.status_code in {429, 500, 502, 503} and attempt < max_retries:
            backoff = 2**attempt
            print(
                f"Warning: Received status {response.status_code}. Retrying in {backoff}s..."
            )
            time.sleep(backoff)
            continue

        # If we get here, it's a hard error
        raise RuntimeError(
            f"DeepSeek API error {response.status_code}: {response.text[:400]}"
        )

    raise RuntimeError("Exceeded maximum retries for DeepSeek API call")


def extract_label(response_text: str) -> str | None:
    """Extract the first binary digit (0/1) from the model response."""
    for ch in response_text.strip():
        if ch in {"0", "1"}:
            return ch
    return None


def process_file(
    input_path: Path,
    output_path: Path,
    system_prompt: str,
    api_key: str,
    model: str,
    limit: int | None,
    sleep: float,
) -> None:
    """Label all user messages in a CSV using DeepSeek."""
    df = pd.read_csv(input_path)
    required_cols = {"conversation_id", "user_message"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input file missing required columns: {sorted(missing)}")

    if limit is not None:
        df = df.head(limit).copy()
        print(f"Limiting to first {len(df)} messages")

    # Initialize new columns
    df["model"] = model
    df["prompt_name"] = PROMPT_PATH.stem
    if "llm_response" not in df.columns:
        df["llm_response"] = None
    if "label" not in df.columns:
        df["label"] = None

    existing_map: dict[str, dict] = {}
    reused = 0
    reused_valid = 0
    if output_path.exists():
        try:
            existing_df = pd.read_csv(output_path)
            prompt_matches = True
            if "prompt_name" in existing_df.columns:
                prompt_values = existing_df["prompt_name"].dropna().unique()
                if len(prompt_values) > 0 and set(prompt_values) != {PROMPT_PATH.stem}:
                    prompt_matches = False
            if prompt_matches and "conversation_id" in existing_df.columns:
                for _, existing_row in existing_df.iterrows():
                    conv_id = existing_row["conversation_id"]
                    existing_map[conv_id] = {
                        "llm_response": existing_row.get("llm_response"),
                        "label": existing_row.get("label"),
                    }
                print(
                    f"Found existing labeled file with {len(existing_df)} rows; reusing completed entries."
                )
            else:
                print(
                    "Existing output found but prompt mismatch or missing columns; relabeling from scratch."
                )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Warning: Could not read existing output file: {exc}")

    # Pre-populate cached responses into the DataFrame
    for idx, row in df.iterrows():
        conv_id = row["conversation_id"]
        if conv_id in existing_map:
            cached = existing_map[conv_id]
            cached_response = cached.get("llm_response")
            if cached_response and not pd.isna(cached_response):
                df.at[idx, "llm_response"] = cached_response
                cached_label = cached.get("label")
                if pd.isna(cached_label):
                    cached_label = extract_label(cached_response)
                df.at[idx, "label"] = cached_label
                reused += 1
                if cached_label is not None:
                    reused_valid += 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    first10_usage: list[dict] = []
    successful_queries = 0
    cost_reported = False
    save_interval = 10  # Save every N new API calls

    for idx, row in df.iterrows():
        conversation_id = row["conversation_id"]
        message = str(row["user_message"])

        # Skip if already has a response (from cache)
        if pd.notna(df.at[idx, "llm_response"]):
            print(
                f"[{idx + 1}/{len(df)}] conversation_id={conversation_id} -> (reused cached response)"
            )
            continue

        try:
            response_text, usage = call_deepseek(system_prompt, message, api_key, model)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Error labeling row {idx} ({conversation_id}): {exc}")
            response_text = None
            usage = {}

        # Update DataFrame in place
        df.at[idx, "llm_response"] = response_text
        df.at[idx, "label"] = extract_label(response_text) if response_text else None

        if response_text:
            print(
                f"[{idx + 1}/{len(df)}] conversation_id={conversation_id} -> {response_text}"
            )
            successful_queries += 1
            if len(first10_usage) < 10:
                first10_usage.append(usage or {})

            # Incremental save every N successful API calls
            if successful_queries % save_interval == 0:
                df.to_csv(output_path, index=False)
        else:
            print(f"[{idx + 1}/{len(df)}] conversation_id={conversation_id} -> ERROR")

        if sleep:
            time.sleep(sleep)

        if successful_queries == 10 and not cost_reported:
            cost_reported = True
            total_prompt_tokens = sum(
                rec.get("prompt_tokens", 0) for rec in first10_usage
            )
            total_completion_tokens = sum(
                rec.get("completion_tokens", 0) for rec in first10_usage
            )
            input_cost = total_prompt_tokens * INPUT_COST_CACHE_MISS
            output_cost = total_completion_tokens * OUTPUT_COST
            avg_cost = (input_cost + output_cost) / successful_queries
            print("=" * 70)
            print("Cost estimate after first 10 queries (cache miss rates):")
            print(f"  Input tokens: {total_prompt_tokens:,}")
            print(f"  Output tokens: {total_completion_tokens:,}")
            print(f"  Total cost: ${input_cost + output_cost:.4f}")
            print(f"  Average cost/query: ${avg_cost:.4f}")
            print("=" * 70)

    # Final save
    df.to_csv(output_path, index=False)

    if not cost_reported and first10_usage:
        total_prompt_tokens = sum(rec.get("prompt_tokens", 0) for rec in first10_usage)
        total_completion_tokens = sum(
            rec.get("completion_tokens", 0) for rec in first10_usage
        )
        input_cost = total_prompt_tokens * INPUT_COST_CACHE_MISS
        output_cost = total_completion_tokens * OUTPUT_COST
        avg_cost = (input_cost + output_cost) / len(first10_usage)
        print("=" * 70)
        print(
            f"Cost estimate after {len(first10_usage)} queries (cache miss rates; fewer than 10 processed):"
        )
        print(f"  Input tokens: {total_prompt_tokens:,}")
        print(f"  Output tokens: {total_completion_tokens:,}")
        print(f"  Total cost: ${input_cost + output_cost:.4f}")
        print(f"  Average cost/query: ${avg_cost:.4f}")
        print("=" * 70)

    print("=" * 70)
    print(f"Saved labeled data to: {output_path}")
    print(f"Total rows processed: {len(df)}")
    total_valid = df["label"].notna().sum()
    print(f"Valid labels: {total_valid}")
    print(f"Reused cached rows: {reused} (valid labels reused: {reused_valid})")
    new_valid = successful_queries
    print(f"Newly labeled via API: {new_valid}")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label keyword-filtered user messages using DeepSeek's chat completions API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV filename or path within data/wildchat_data/keyword_filtered",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output CSV filename (saved under labelled_data/)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="DeepSeek model name to use",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit labeling to the first N rows",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between API calls (set >0 to throttle)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = load_api_key()

    # Resolve input path
    input_path = Path(args.input)
    if input_path.is_absolute():
        # Use absolute path directly
        pass
    elif input_path.exists():
        # Relative path from cwd exists - resolve it
        input_path = input_path.resolve()
    else:
        # Assume it's just a filename and look in INPUT_DIR
        input_path = INPUT_DIR / input_path.name
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    # Resolve output path
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = OUTPUT_DIR / output_path
    else:
        output_path = OUTPUT_DIR / f"deepseek_{input_path.stem}.csv"

    if not PROMPT_PATH.exists():
        print(f"ERROR: Prompt file not found: {PROMPT_PATH}")
        sys.exit(1)
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    print("=" * 70)
    print("DeepSeek Labeling Script")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Model: {args.model}")
    print(f"Limit: {args.limit if args.limit is not None else 'ALL'}")
    print("=" * 70)

    try:
        process_file(
            input_path=input_path,
            output_path=output_path,
            system_prompt=system_prompt,
            api_key=api_key,
            model=args.model,
            limit=args.limit,
            sleep=args.sleep,
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
