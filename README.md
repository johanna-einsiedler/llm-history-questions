# Historical Fact-Seeking in LLM Conversations

Pipeline for identifying and analyzing historical fact-seeking conversations in WildChat data.

## Pipeline Overview

```
WildChat parquet → 01_filter_english → 03_prefilter → 04_label_deepseek → 05_process_labeled → 06_fact_counting → 07_epistemic_diversity
```

## Setup

```bash
# Create virtual environment
python -m venv history-env
source history-env/bin/activate

# Install dependencies
pip install pandas transformers sentence-transformers openai python-dotenv tqdm

# Configure API keys
cp .env.example .env
# Add DEEPSEEK_API_KEY and OPENAI_API_KEY
```

## Scripts

Run scripts in numerical order from `src/`:

| Script | Description | Input | Output |
|--------|-------------|-------|--------|
| `01_filter_english.py` | Extract English user messages from WildChat parquet files. Filters to ≤10 sentences, removes duplicates and midjourney prompts. | `raw_data/*.parquet` | `english_data/*.csv` |
| `02_wikipedia_name_prep.py` | Prepare historical figure names from Pantheon dataset for keyword matching. | `pantheon/pantheon.tsv` | `pantheon/historical_figures.json` |
| `03_prefilter.py` | Keyword-based prefilter for historical content (years, centuries, historical events/figures). High recall heuristic filtering. | `english_data/*.csv` | `keyword_filtered/*.csv` |
| `04_label_deepseek.py` | LLM classification of messages as `HISTORY_FACT_SEEKING` vs `NOT_HISTORY_FACT_SEEKING` using DeepSeek API. Supports resume. | `keyword_filtered/*.csv` | `labelled_data/*.csv` |
| `05_process_labeled.py` | Filter labeled data to keep only `HISTORY_FACT_SEEKING` conversations. Combines multiple files. | `labelled_data/*.csv` | `filtered_data/combined_filtered.csv` |
| `06_fact_counting.py` | Decompose assistant responses into atomic facts using OpenAI. Checkpoints every 50 rows. | `filtered_data/*.csv` | `fact_counted/*.csv` |
| `07_epistemic_diversity.py` | Calculate epistemic diversity metrics following Wright et al. 2025. Uses NLI-based claim clustering. | `fact_counted/*.csv` | `epistemic_diversity/*.csv` |

## Data Directory Structure

```
data/wildchat_data/
├── raw_data/                  # Original WildChat parquet files
├── english_data/              # English user messages (step 1)
├── keyword_filtered/          # History keyword matches (step 3)
├── labelled_data/             # DeepSeek labels (step 4)
├── filtered_data/             # Final historical fact-seeking (step 5)
│   └── combined_filtered.csv  # ~21k historical Q&A pairs
├── fact_counted/              # Atomic facts extracted (step 6)
└── epistemic_diversity/       # Diversity metrics (step 7)

data/pantheon/
├── pantheon.tsv               # Pantheon historical figures dataset
└── historical_figures.json    # Prepared name list
```

## Key Concepts

### Historical Fact-Seeking Classification
A query qualifies as `HISTORY_FACT_SEEKING` if the user's primary intent is to obtain factual information about real past events, historical people, societies, or developments. See `prompts/prompt_historical_political.txt` for the full annotation guide.

### Epistemic Diversity (Wright et al. 2025)
Measures diversity of factual claims in LLM responses:
1. **Fact decomposition**: Break response into atomic claims
2. **Claim clustering**: Group semantically equivalent claims using NLI mutual entailment
3. **Hill-Shannon diversity**: Calculate diversity metric accounting for cluster sizes

## Usage Examples

```bash
# Full pipeline (steps 1-5)
python src/01_filter_english.py
python src/03_prefilter.py data/wildchat_data/english_data/*.csv
python src/04_label_deepseek.py --input train-00000-of-00014.csv
python src/05_process_labeled.py

# Fact extraction with checkpointing
python src/06_fact_counting.py --sample 0

# Epistemic diversity (simple mode)
python src/07_epistemic_diversity.py --sample 100

# Epistemic diversity with NLI clustering
python src/07_epistemic_diversity.py --sample 0 --cluster
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DEEPSEEK_API_KEY` | DeepSeek API key for labeling |
| `OPENAI_API_KEY` | OpenAI API key for fact extraction |

## Dependencies

- pandas
- transformers
- sentence-transformers  
- openai
- python-dotenv
- tqdm
