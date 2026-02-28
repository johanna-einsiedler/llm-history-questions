#!/usr/bin/env python3
"""
Epistemic diversity calculation for historical Q&A responses.

Implements the methodology from Wright et al. 2025:
"Epistemic Diversity and Knowledge Collapse in Large Language Models"

Pipeline:
1. Decompose responses into atomic claims using LLM
2. Cluster claims by semantic equivalence using NLI mutual entailment
3. Estimate coverage using Good-Turing estimator
4. Rarefy samples to minimum coverage for fair comparison
5. Calculate Hill-Shannon diversity

Usage:
    # Test on small sample (simple metrics)
    python src/06_epistemic_diversity.py --sample 10

    # Full NLI clustering (paper-accurate)
    python src/06_epistemic_diversity.py --sample 10 --cluster

    # Full run
    python src/06_epistemic_diversity.py --sample 0 --cluster
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
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
OUTPUT_DIR = PROJECT_DIR / "data" / "wildchat_data" / "epistemic_diversity"

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
# CPU-COMPATIBLE NLI CLUSTERING (no CUDA required)
# ============================================================


def load_nli_pipeline(device: str = "cpu"):
    """Load the NLI pipeline on CPU or GPU."""
    from transformers import pipeline

    print(f"\nLoading NLI model (microsoft/deberta-large-mnli) on {device}...")
    print("(This may take a minute on first run)")

    # Use smaller model for faster CPU inference
    # deberta-base-mnli is faster than deberta-large-mnli
    model_name = "microsoft/deberta-base-mnli"

    pipe = pipeline(
        "text-classification",
        model=model_name,
        device=-1 if device == "cpu" else 0,  # -1 for CPU
        top_k=None,  # Return all labels with scores
    )
    print(f"Loaded {model_name}")
    return pipe


def check_mutual_entailment(
    pipe,
    claim_a: str,
    claim_b: str,
    threshold: float = 0.7,
) -> bool:
    """Check if two claims mutually entail each other.

    Following Algorithm 1 from Wright et al. 2025:
    Two claims are equivalent if A entails B AND B entails A.

    Args:
        pipe: HuggingFace NLI pipeline
        claim_a: First claim
        claim_b: Second claim
        threshold: Minimum probability for entailment

    Returns:
        True if mutual entailment, False otherwise
    """
    # Check A -> B
    result_ab = pipe(f"{claim_a} [SEP] {claim_b}")
    if isinstance(result_ab, list) and len(result_ab) > 0:
        if isinstance(result_ab[0], list):
            result_ab = result_ab[0]
        scores_ab = {r["label"]: r["score"] for r in result_ab}
    else:
        return False

    entail_ab = scores_ab.get("ENTAILMENT", 0) >= threshold

    if not entail_ab:
        return False

    # Check B -> A
    result_ba = pipe(f"{claim_b} [SEP] {claim_a}")
    if isinstance(result_ba, list) and len(result_ba) > 0:
        if isinstance(result_ba[0], list):
            result_ba = result_ba[0]
        scores_ba = {r["label"]: r["score"] for r in result_ba}
    else:
        return False

    entail_ba = scores_ba.get("ENTAILMENT", 0) >= threshold

    return entail_ab and entail_ba


def cluster_claims_greedy(
    claims: list[str],
    pipe,
    threshold: float = 0.7,
) -> list[int]:
    """Cluster claims using greedy NLI mutual entailment.

    Implements Algorithm 1 from Wright et al. 2025 (simplified):
    - For each claim, check if it entails any existing cluster representative
    - If mutual entailment found, assign to that cluster
    - Otherwise, create a new cluster

    This is O(n*k) where n=claims, k=clusters (vs O(n^2) for full pairwise).
    """
    if len(claims) == 0:
        return []
    if len(claims) == 1:
        return [0]

    cluster_assignments = [-1] * len(claims)
    cluster_representatives = []  # (cluster_id, representative_claim)

    for i, claim in enumerate(claims):
        assigned = False

        # Check against existing cluster representatives
        for cluster_id, rep_claim in cluster_representatives:
            if check_mutual_entailment(pipe, claim, rep_claim, threshold):
                cluster_assignments[i] = cluster_id
                assigned = True
                break

        # Create new cluster if no match
        if not assigned:
            new_cluster_id = len(cluster_representatives)
            cluster_assignments[i] = new_cluster_id
            cluster_representatives.append((new_cluster_id, claim))

    return cluster_assignments


def prepare_claims_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare claims DataFrame from pre-extracted facts (fact_counting.py output).

    Expected input columns: message_id, conversation_id, user_message, facts (JSON list)
    Output format for clustering.

    Uses message_id as the unique topic key (not conversation_id) since
    the same conversation can have multiple Q&A pairs.
    """
    if "facts" not in df.columns:
        raise ValueError("No 'facts' column found. Run fact_counting.py first.")

    all_claims = []

    for idx, row in df.iterrows():
        facts_str = row.get("facts", "[]")
        try:
            facts = json.loads(facts_str) if isinstance(facts_str, str) else []
        except json.JSONDecodeError:
            facts = []

        # Use message_id as unique key (falls back to index if missing)
        msg_id = row.get("message_id", str(idx))
        conv_id = row.get("conversation_id", str(idx))
        model_id = row.get("model", "unknown")

        for fact in facts:
            all_claims.append(
                {
                    "topic": msg_id,  # Group by message (unique per Q&A pair)
                    "factoid": fact,
                    "chunk": fact,
                    "model_id": model_id,
                    "setting": "ift",  # Instruction fine-tuned (parametric memory)
                    "message_id": msg_id,
                    "conversation_id": conv_id,
                    "user_message": row.get("user_message", ""),
                }
            )

    return pd.DataFrame(all_claims)


def cluster_claims_with_nli(
    claim_df: pd.DataFrame,
    output_dir: Path,
    checkpoint_steps: int = 1000,
    n_candidates: int = 6,
    entailment_threshold: float = 0.7,
) -> pd.DataFrame:
    """Cluster claims using NLI mutual entailment (CPU-compatible).

    Implements Algorithm 1 from Wright et al. 2025:
    Two claims are semantically equivalent if they mutually entail each other.
    Uses greedy clustering with cluster representatives for efficiency.
    """
    pipe = load_nli_pipeline(device="cpu")

    topics = claim_df["topic"].unique()
    print(f"Clustering claims across {len(topics)} topics...")

    clustered_dfs = []
    total_claims = 0
    total_clusters = 0

    for topic in tqdm(topics, desc="Clustering by topic"):
        topic_df = claim_df[claim_df["topic"] == topic].reset_index(drop=True)

        if len(topic_df) < 2:
            topic_df["cluster"] = 0
            clustered_dfs.append(topic_df)
            total_claims += len(topic_df)
            total_clusters += 1
            continue

        claims = topic_df["factoid"].tolist()

        try:
            cluster_assignments = cluster_claims_greedy(
                claims, pipe, threshold=entailment_threshold
            )
            topic_df["cluster"] = cluster_assignments
            num_clusters = len(set(cluster_assignments))
        except Exception as e:
            print(f"Clustering error for topic {topic}: {e}")
            topic_df["cluster"] = list(range(len(topic_df)))  # Each claim = own cluster
            num_clusters = len(topic_df)

        clustered_dfs.append(topic_df)
        total_claims += len(topic_df)
        total_clusters += num_clusters

        # Checkpoint
        if len(clustered_dfs) % checkpoint_steps == 0:
            checkpoint_df = pd.concat(clustered_dfs, ignore_index=True)
            checkpoint_path = output_dir / "clustering_checkpoint.parquet"
            checkpoint_df.to_parquet(checkpoint_path)

    print(f"Clustered {total_claims} claims into {total_clusters} clusters")
    return pd.concat(clustered_dfs, ignore_index=True)


# ============================================================
# EPISTEMIC DIVERSITY METRICS (from paper Section 5)
# ============================================================


def estimate_coverage(clustered_df: pd.DataFrame) -> float:
    """Estimate sample coverage using Good-Turing estimator (Equation 2).

    V(X) = 1 - (f1/n) * ((n-1)*f1 / ((n-1)*f1 + 2*f2))

    where:
    - n = total number of claims
    - f1 = number of singleton clusters (clusters with exactly 1 claim)
    - f2 = number of doubleton clusters (clusters with exactly 2 claims)

    CPU-compatible implementation (no llm-knowledge dependency).
    """
    n = len(clustered_df)
    if n == 0:
        return 1.0

    # Count cluster sizes
    cluster_counts = clustered_df["cluster"].value_counts()

    # f1 = number of singleton clusters
    f1 = sum(1 for count in cluster_counts if count == 1)
    # f2 = number of doubleton clusters
    f2 = sum(1 for count in cluster_counts if count == 2)

    if n <= 1 or f1 == 0:
        return 1.0

    # Equation 2 from paper
    numerator = (n - 1) * f1
    denominator = (n - 1) * f1 + 2 * f2

    if denominator > 0:
        coverage = 1 - (f1 / n) * (numerator / denominator)
    else:
        coverage = 1 - (f1 / n)

    return max(0.0, min(1.0, coverage))  # Clamp to [0, 1]


def resample_to_coverage(
    clustered_df: pd.DataFrame,
    target_coverage: float,
    max_iterations: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Rarefy sample to target coverage level for fair comparison.

    Iteratively removes random claims until coverage drops to target level.
    This ensures fair comparison between samples with different sizes.

    CPU-compatible implementation (no llm-knowledge dependency).
    """
    import random

    random.seed(random_state)

    df = clustered_df.copy()
    current_coverage = estimate_coverage(df)

    if current_coverage <= target_coverage:
        return df  # Already at or below target

    iterations = 0
    while (
        current_coverage > target_coverage
        and len(df) > 2
        and iterations < max_iterations
    ):
        # Remove a random claim
        drop_idx = random.choice(df.index.tolist())
        df = df.drop(drop_idx).reset_index(drop=True)
        current_coverage = estimate_coverage(df)
        iterations += 1

    return df


def calculate_hill_shannon_diversity(
    clustered_df: pd.DataFrame,
) -> tuple[float, float, dict]:
    """Calculate Hill-Shannon diversity (Equation 1 from paper).

    Hill-Shannon: D_S(X) = exp{-sum_i p_i * ln(p_i)}

    This is e^entropy, where entropy is in natural log (nats).

    Returns:
        entropy: Shannon entropy in nats
        hill_shannon: Hill-Shannon diversity (exp(entropy))
        cluster_probs: dict of cluster_id -> probability

    CPU-compatible implementation (no llm-knowledge dependency).
    """
    n = len(clustered_df)
    if n == 0:
        return 0.0, 1.0, {}

    # Count claims per cluster
    cluster_counts = clustered_df["cluster"].value_counts()

    # Calculate probabilities
    cluster_probs = {
        cluster_id: count / n for cluster_id, count in cluster_counts.items()
    }

    # Shannon entropy: H(X) = -sum_i p_i * ln(p_i)
    entropy = -sum(p * math.log(p) for p in cluster_probs.values() if p > 0)

    # Hill-Shannon diversity: D_S(X) = exp(H(X))
    hill_shannon = math.exp(entropy)

    return entropy, hill_shannon, cluster_probs


def calculate_diversity_metrics_for_topic(
    clustered_df: pd.DataFrame,
    topic: str,
    min_coverage: float | None = None,
) -> dict:
    """Calculate full diversity metrics for a single topic.

    Implements the complete methodology from Section 5:
    1. Estimate coverage
    2. Rarefy to minimum coverage (if provided)
    3. Calculate Hill-Shannon diversity

    Reports ORIGINAL num_claims/num_clusters (before rarefaction),
    but entropy/hill_shannon computed on rarefied sample.
    """
    topic_df = clustered_df[clustered_df["topic"] == topic].copy()

    if len(topic_df) < 2:
        return {
            "topic": topic,
            "num_claims": len(topic_df),
            "num_clusters": 0,
            "coverage": None,
            "entropy": None,
            "hill_shannon": None,
        }

    # Store ORIGINAL counts (before rarefaction)
    num_claims_original = len(topic_df)
    num_clusters_original = topic_df["cluster"].nunique()

    # Estimate coverage
    try:
        coverage = estimate_coverage(topic_df)
    except Exception:
        coverage = None

    # Rarefy if target coverage provided
    if min_coverage is not None and coverage is not None:
        try:
            topic_df = resample_to_coverage(topic_df, min_coverage)
        except Exception:
            pass  # Use original if rarefaction fails

    # Calculate diversity (on potentially rarefied sample)
    try:
        entropy, hill_shannon, probs = calculate_hill_shannon_diversity(topic_df)
    except Exception:
        entropy, hill_shannon = None, None

    return {
        "topic": topic,
        "num_claims": num_claims_original,  # Always report original
        "num_clusters": num_clusters_original,  # Always report original
        "coverage": coverage,
        "entropy": entropy,
        "hill_shannon": hill_shannon,
    }


def calculate_all_diversity_metrics(
    clustered_df: pd.DataFrame,
    rarefy_to_min: bool = True,
) -> pd.DataFrame:
    """Calculate diversity metrics for all topics.

    If rarefy_to_min=True, first estimates coverage for all topics,
    then rarefies each to the minimum coverage for fair comparison.
    """
    topics = clustered_df["topic"].unique()

    # First pass: estimate coverage for all topics
    coverages = {}
    for topic in topics:
        topic_df = clustered_df[clustered_df["topic"] == topic]
        if len(topic_df) >= 2:
            try:
                coverages[topic] = estimate_coverage(topic_df)
            except Exception:
                pass

    # Determine minimum coverage for rarefaction
    min_coverage = min(coverages.values()) if coverages and rarefy_to_min else None

    if min_coverage:
        print(f"Rarefying all topics to minimum coverage: {min_coverage:.4f}")

    # Second pass: calculate diversity with rarefaction
    results = []
    for topic in tqdm(topics, desc="Calculating diversity"):
        metrics = calculate_diversity_metrics_for_topic(
            clustered_df, topic, min_coverage
        )
        results.append(metrics)

    return pd.DataFrame(results)


# ============================================================
# SIMPLIFIED DIVERSITY (fallback without NLI clustering)
# ============================================================


def calculate_simple_diversity(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple diversity metrics without NLI clustering.

    Uses exact string matching for claim counting and calculates:
    - Coverage estimate (Equation 2)
    - Shannon entropy in nats
    - Hill-Shannon diversity (exp(entropy))

    This is a faster approximation when full NLI clustering is not feasible.
    """
    print("\n=== Calculating Diversity Metrics ===")
    print("(Using exact string matching - for NLI clustering use --cluster)")

    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing diversity"):
        facts_str = row.get("facts", "[]")
        try:
            facts = json.loads(facts_str) if isinstance(facts_str, str) else []
        except json.JSONDecodeError:
            facts = []

        n = len(facts)

        if n == 0:
            results.append(
                {
                    "num_claims": 0,
                    "num_clusters": 0,
                    "f1_singletons": 0,
                    "f2_doubletons": 0,
                    "coverage": None,
                    "entropy": 0.0,
                    "hill_shannon": 1.0,
                }
            )
            continue

        # Count claim frequencies (exact match = clusters)
        claim_counts = Counter(facts)
        num_clusters = len(claim_counts)

        # f1 = singletons, f2 = doubletons (Equation 2)
        f1 = sum(1 for c in claim_counts.values() if c == 1)
        f2 = sum(1 for c in claim_counts.values() if c == 2)

        # Coverage estimate (Equation 2 from paper)
        # V(X) = 1 - (f1/n) * ((n-1)*f1 / ((n-1)*f1 + 2*f2))
        if n > 1 and f1 > 0:
            numerator = (n - 1) * f1
            denominator = (n - 1) * f1 + 2 * f2
            if denominator > 0:
                coverage = 1 - (f1 / n) * (numerator / denominator)
            else:
                coverage = 1 - (f1 / n)
        else:
            coverage = 1.0

        # Shannon entropy in nats: H(X) = -sum_i p_i * ln(p_i)
        probs = [c / n for c in claim_counts.values()]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)

        # Hill-Shannon diversity: D_S(X) = exp(H(X))
        hill_shannon = math.exp(entropy)

        results.append(
            {
                "num_claims": n,
                "num_clusters": num_clusters,
                "f1_singletons": f1,
                "f2_doubletons": f2,
                "coverage": coverage,
                "entropy": entropy,
                "hill_shannon": hill_shannon,
            }
        )

    for key in results[0].keys():
        df[key] = [r[key] for r in results]

    return df


# ============================================================
# ARGUMENT PARSING
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate epistemic diversity (Wright et al. 2025)",
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input CSV file (default: fact_counted output)",
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
        help=f"Number of samples (default: {DEFAULT_SAMPLE_SIZE}, 0 for all)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )

    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Run full NLI clustering (slow but paper-accurate)",
    )

    parser.add_argument(
        "--no-rarefy",
        action="store_true",
        help="Skip rarefaction to minimum coverage",
    )

    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        default=1000,
        help="Checkpoint frequency for clustering",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================


def main():
    args = parse_args()

    # Try to find input with pre-extracted facts
    fact_counted_input = (
        PROJECT_DIR
        / "data"
        / "wildchat_data"
        / "fact_counted"
        / "fact_counted_combined_filtered.csv"
    )

    if args.input:
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = PROJECT_DIR / args.input
    elif fact_counted_input.exists():
        input_path = fact_counted_input
        print(f"Using pre-extracted facts from: {input_path}")
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
        output_path = OUTPUT_DIR / f"diversity_{input_path.stem}.csv"

    # Load data
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} messages")

    text_col = args.text_col
    if text_col not in df.columns:
        print(f"ERROR: Column '{text_col}' not found in data")
        return

    # Sample if requested
    if args.sample > 0 and args.sample < len(df):
        print(f"\nSampling {args.sample} questions (seed={args.seed})")
        df = df.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)

    print(f"\nProcessing {len(df)} questions...")

    # Check if we have assistant responses
    if "assistant_response" not in df.columns:
        df = join_with_responses(df, text_col)

    # Check for pre-extracted facts
    has_facts = "facts" in df.columns and df["facts"].notna().any()

    if not has_facts:
        print("\nNo pre-extracted facts found.")
        print("Please run fact_counting.py first.")
        return

    if args.cluster:
        # Full NLI clustering pipeline (paper-accurate)
        print("\n=== Running Full NLI Clustering Pipeline ===")
        print("(This follows Wright et al. 2025 methodology)")

        # Prepare claims
        claim_df = prepare_claims_dataframe(df)
        print(f"Prepared {len(claim_df)} claims for clustering")

        if len(claim_df) == 0:
            print("No claims to cluster")
            return

        # Cluster with NLI
        clustered_df = cluster_claims_with_nli(
            claim_df,
            OUTPUT_DIR,
            checkpoint_steps=args.checkpoint_steps,
        )

        # Save clustered claims
        cluster_path = OUTPUT_DIR / f"clusters_{input_path.stem}.parquet"
        clustered_df.to_parquet(cluster_path)
        print(f"Saved clustered claims to: {cluster_path}")

        # Calculate diversity metrics
        diversity_df = calculate_all_diversity_metrics(
            clustered_df,
            rarefy_to_min=not args.no_rarefy,
        )

        # Merge back to original dataframe using message_id (unique per Q&A pair)
        diversity_df = diversity_df.rename(columns={"topic": "message_id"})
        df = df.merge(diversity_df, on="message_id", how="left")

        # Print summary
        print("\n=== EPISTEMIC DIVERSITY RESULTS ===")
        valid_hs = diversity_df["hill_shannon"].dropna()
        if len(valid_hs) > 0:
            print(f"Hill-Shannon diversity (mean): {valid_hs.mean():.2f}")
            print(f"Hill-Shannon diversity (median): {valid_hs.median():.2f}")
            print(f"Entropy (mean): {diversity_df['entropy'].dropna().mean():.3f} nats")
            print(f"Coverage (mean): {diversity_df['coverage'].dropna().mean():.4f}")
            print(
                f"Clusters per topic (mean): {diversity_df['num_clusters'].mean():.1f}"
            )
    else:
        # Simple diversity (faster approximation)
        df = calculate_simple_diversity(df)

        # Print summary
        print("\n=== DIVERSITY RESULTS ===")
        print("(Using exact string matching. For NLI clustering use --cluster)")
        print(f"Hill-Shannon (mean): {df['hill_shannon'].mean():.2f}")
        print(f"Entropy (mean): {df['entropy'].mean():.3f} nats")
        valid_cov = df["coverage"].dropna()
        if len(valid_cov) > 0:
            print(f"Coverage (mean): {valid_cov.mean():.4f}")
        print(f"Unique claims/clusters (mean): {df['num_clusters'].mean():.1f}")

    # Save results
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} results to: {output_path}")


if __name__ == "__main__":
    main()
