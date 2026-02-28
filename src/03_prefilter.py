"""Keyword-based history prefilter for WildChat user messages.

The heuristics in this module were originally written for quick, high-recall
screening of prompts. This script now exposes a CLI that accepts the CSV output
from ``first_conversation_english.py`` (all English user messages) and writes a
filtered subset containing only history-related prompts to
``data/wildchat_data/keyword_filtered``.

Usage:
    python src/01_prefilter.py data/wildchat_data/first_messages_data/foo.csv
"""

import argparse
import json
import re
from pathlib import Path
from typing import List

import pandas as pd

# -----------------------------
# REGEX PATTERNS (compiled once)
# -----------------------------

YEAR_REGEX = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b", re.IGNORECASE)

CENTURY_REGEX = re.compile(
    r"\b\d{1,2}(st|nd|rd|th)\s+century\b",
    re.IGNORECASE,
)

ERA_PHRASE_REGEX = re.compile(
    r"\b(in|during)\s+the\s+(middle ages|renaissance|enlightenment|victorian era|bronze age|iron age)\b",
    re.IGNORECASE,
)

QUESTION_PHRASE_REGEX = re.compile(
    r"\b(who was|when did|what year|in what year|how long ago)\b",
    re.IGNORECASE,
)

WORLD_WAR_REGEX = re.compile(
    r"\b(world war\s*(i|ii|1|2)?)\b",
    re.IGNORECASE,
)

# -----------------------------
# EVENT / HISTORY KEYWORDS
# (tuned for recall)
# -----------------------------

EVENT_KEYWORDS = {
    # conflicts & politics
    "war",
    "battle",
    "treaty",
    "revolution",
    "uprising",
    "rebellion",
    "civil war",
    "cold war",
    "independence",
    # political structures
    "empire",
    "dynasty",
    "kingdom",
    "monarchy",
    "republic",
    "colony",
    "colonial",
    "imperial",
    "imperialism",
    "colonialism",
    # governments & ideologies
    "fascism",
    "communism",
    "socialism",
    "dictatorship",
    "democracy",
    # agreements & documents
    "constitution",
    "declaration",
    "alliance",
    "accord",
    # historical periods
    "renaissance",
    "enlightenment",
    "middle ages",
    "industrial revolution",
    "victorian era",
    "bronze age",
    "iron age",
    # titles often used historically
    "pharaoh",
    "tsar",
    "czar",
    "kaiser",
    "sultan",
    "caliph",
}

# Precompile keyword regex for speed
EVENT_KEYWORD_REGEX = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(EVENT_KEYWORDS)) + r")\b",
    re.IGNORECASE,
)

# -----------------------------
# MAJOR HISTORICAL FIGURES
# (high-frequency, high-recall list)
# -----------------------------

BASE_HISTORICAL_FIGURES = {
    # ancient
    "julius caesar",
    "augustus",
    "alexander the great",
    "cleopatra",
    "socrates",
    "plato",
    "aristotle",
    "genghis khan",
    "charlemagne",
    "joan of arc",
    # early modern
    "martin luther",
    "galileo",
    "isaac newton",
    # modern political leaders
    "napoleon",
    "napoleon bonaparte",
    "abraham lincoln",
    "george washington",
    "winston churchill",
    "franklin roosevelt",
    "theodore roosevelt",
    "queen victoria",
    "elizabeth i",
    # 20th century
    "adolf hitler",
    "joseph stalin",
    "vladimir lenin",
    "mao zedong",
    "mahatma gandhi",
    "nelson mandela",
    "ho chi minh",
}


SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
PANTHEON_PATH = PROJECT_DIR / "data" / "pantheon" / "historical_figures.json"
OUTPUT_DIR = PROJECT_DIR / "data" / "wildchat_data" / "keyword_filtered"


def load_historical_figures(json_path: Path = PANTHEON_PATH) -> List[str]:
    """Load Pantheon names and merge with the base figure list."""

    names = {
        str(name).strip().lower()
        for name in BASE_HISTORICAL_FIGURES
        if str(name).strip()
    }

    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for entry in data:
                    value = str(entry).strip()
                    if value:
                        names.add(value.lower())
            else:
                print(
                    f"Warning: Expected list in {json_path}, got {type(data)}. Using base figure list only."
                )
        except Exception as exc:
            print(
                f"Warning: Failed to load {json_path}: {exc}. Using base figure list only."
            )
    else:
        print(
            f"Warning: Pantheon file not found at {json_path}. Using base figure list only."
        )

    return sorted(names)


FIGURE_NAMES = load_historical_figures()

FIGURE_REGEX = re.compile(
    r"\b(" + "|".join(re.escape(n) for n in FIGURE_NAMES) + r")\b",
    re.IGNORECASE,
)

# -----------------------------
# MAIN FILTER FUNCTION
# -----------------------------


def is_history_related(text: str) -> bool:
    """
    High-recall history prefilter.
    Returns True if the query is likely history-related.
    """

    if text is None:
        return False

    if isinstance(text, float) and pd.isna(text):
        return False

    t = str(text).lower().strip()
    if not t or t == "nan":
        return False

    # --- fast regex checks (short-circuit) ---
    if YEAR_REGEX.search(t):
        return True

    if CENTURY_REGEX.search(t):
        return True

    if ERA_PHRASE_REGEX.search(t):
        return True

    if QUESTION_PHRASE_REGEX.search(t):
        return True

    if WORLD_WAR_REGEX.search(t):
        return True

    if EVENT_KEYWORD_REGEX.search(t):
        return True

    if FIGURE_REGEX.search(t):
        return True

    return False


# -----------------------------
# OPTIONAL: batch helper
# -----------------------------


def filter_history_queries(texts: List[str]) -> List[bool]:
    """Vectorized helper."""
    return [is_history_related(t) for t in texts]


def filter_csv(input_path: Path, output_dir: Path = OUTPUT_DIR) -> Path:
    """Load a CSV of user messages and save rows matching the history filter."""

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if "user_message" not in df.columns:
        raise KeyError(
            "Expected column 'user_message' in the CSV produced by first_conversation_english"
        )

    total_rows = len(df)
    mask = df["user_message"].astype(str).apply(is_history_related)
    filtered_df = df[mask].copy()
    kept_rows = len(filtered_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / input_path.name
    filtered_df.to_csv(output_path, index=False)

    print(f"Input rows: {total_rows}")
    print(f"History-related rows: {kept_rows}")
    print(f"Saved filtered CSV to: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Filter English user messages for history-related prompts."
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help="Path to CSV produced by first_conversation_english.py",
    )
    args = parser.parse_args()

    filter_csv(Path(args.input_csv).expanduser().resolve())


if __name__ == "__main__":
    main()
