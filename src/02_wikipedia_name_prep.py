import json
import re
from pathlib import Path

import pandas as pd

INPUT_PATH = (
    "/Users/einsie0004/Documents/research/35_history_LLM/data/pantheon/pantheon.tsv"
)

df = pd.read_csv(INPUT_PATH, sep="\t")

print("Loaded:", len(df))
print(df.columns)


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    name = re.sub(r"\s+", " ", name)
    return name


df["name_norm"] = df["name"].apply(normalize_name)

# keep everything (best recall)
filtered = df.copy()

TOP_N = 2000  # change to 500–5000 as needed

filtered = (
    filtered.sort_values("HPI", ascending=False)
    .drop_duplicates("name_norm")
    .head(TOP_N)
)

print("Final size:", len(filtered))

output_names = sorted(filtered["name_norm"].unique())

Path(
    "/Users/einsie0004/Documents/research/35_history_LLM/data/pantheon/historical_figures.json"
).write_text(json.dumps(output_names, indent=2))
