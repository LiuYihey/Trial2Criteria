# Trial2Criteria

<video src="Video%20Project.mp4" controls width="100%"></video>

Four-domain retrieval-augmented generation for clinical trial eligibility criteria:

| Domain | Source | Module |
|--------|--------|--------|
| **PrimeKG** | Disease features & gene associations | `vendor/primekg/advanced_disease_extractor.py` |
| **DrugBank** | Drug mechanisms & indications | `clinical_rag/domains/drugbank/` |
| **ClinicalTrials.gov** | Similar trials (Phase 1/2/3) | `vendor/trial2vec/` (Trial2Vec) |
| **PubMed** | Related medical literature | Entrez + FAISS + BGE reranker |

## Repository layout

```
Trial2Criteria/
├── clinical_rag/          # Main Python package
│   ├── search.py          # Full RAG + LLM criteria generation (Web / CLI)
│   ├── batch.py           # Batch retrieval → JSONL (no LLM)
│   ├── pipeline/          # Prompt / response / reasoning generation
│   └── domains/drugbank/  # Drug name matching
├── vendor/
│   ├── primekg/           # PrimeKG disease extractor (upstream)
│   └── trial2vec/         # Trial2Vec similarity search (upstream)
├── data/                  # Runtime datasets (see data/README.md)
├── data_prep/             # ETL scripts to build datasets
├── templates/             # LLM prompt templates
├── web/                   # FastAPI web UI
├── outputs/               # Generated artifacts (gitignored)
├── config.example.json    # Configuration template
└── requirements.txt
```

## Setup

```bash
# 1. Clone and install dependencies
pip install -r requirements.txt
pip install -e vendor/trial2vec

# 2. Configure
cp config.example.json config.json
cp .env.example .env
# Fill in entrez_email and API keys in config.json (or set ENTREZ_EMAIL / ARK_API_KEY)

# 3. Prepare data (see data/README.md)
#    - data/drugbank/drugbank_data_v1.csv
#    - data/primekg/disease_features.csv
#    - data/trial_gov/embeddings/*.pth
#    - data/trial_gov/retrieval_base/*.csv
```

PubMed retrieval requires only an NCBI Entrez **email** — no API key.

## Usage

### Web UI

```bat
start_web.bat
```

Open http://127.0.0.1:8000

### Batch pipeline

```bash
# Step 1 — four-domain context retrieval
python -m clinical_rag --input_csv data/trial_gov/test_sets/your_trials.csv

# Step 2 — generate prompts
clinical-rag-prompts --input_jsonl outputs/rag/trials_rag.jsonl

# Step 3 — LLM criteria generation
clinical-rag-responses --prompts_dir outputs/prompts/

# Step 4 (optional) — reasoning traces
clinical-rag-reasoning
```

Equivalent module invocations:

```bash
python -m clinical_rag.batch
python -m clinical_rag.pipeline.generate_prompts
```

### Benchmark profile collection (`final_bench.json`)

Collect four-domain RAG into `trial_profile_*.json` files (same format as `trial_profile_A_Randomized__*.json`):

```bash
# All 133 trials (supports resume via outputs/bench_profiles/manifest.json)
clinical-rag-bench --input final_bench.json --output_dir outputs/bench_profiles

# Single trial / smoke test
python -m clinical_rag.bench_collector --nct_id NCT07552870 --limit 1
```

Output fields: `disease_info`, `drug_info`, `relevant_papers`, `similar_trials`.

### Interactive CLI

```bash
python clinical_rag/cli.py
```

## Data preparation

| Domain | Scripts |
|--------|---------|
| DrugBank | `data_prep/drugbank/` — XML → JSON → CSV |
| ClinicalTrials.gov | `data_prep/trial_gov/` — filter, reformat, build retrieval base |
| Trial2Vec embeddings | `vendor/trial2vec/trial_search/Embedding_Retrieve/` |
| PrimeKG features | `vendor/primekg/codes/` |

## Configuration

All paths are relative to the project root. See `config.example.json`.

Secrets can be set via environment variables:

- `ENTREZ_EMAIL` — NCBI Entrez email
- `ARK_API_KEY` — Volcengine Ark API key for DeepSeek models

Never commit `config.json`, `.env`, or `api_key.md` — they are listed in `.gitignore`.

## License

Third-party components retain their original licenses (`vendor/primekg/LICENSE`, etc.).
