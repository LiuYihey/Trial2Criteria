# Trial2Criteria

Click the GIF to watch the full demo video.

<a href="https://github.com/LiuYihey/Trial2Criteria/releases/download/demo-video/video-project.mp4">
  <img src="https://github.com/LiuYihey/Trial2Criteria/raw/main/docs/demo.gif" alt="Trial2Criteria demo video" width="1920" style="margin-bottom: 0.75em;">
</a>

Four-domain retrieval-augmented generation for clinical trial eligibility criteria:

| Domain | Source | Role |
|--------|--------|------|
| **PrimeKG** | Disease knowledge graph | Disease features and gene associations |
| **DrugBank** | Drug database | Mechanisms, indications, and drug matching |
| **ClinicalTrials.gov** | Trial registry | Similar Phase 1/2/3 trials via Trial2Vec |
| **PubMed** | Medical literature | Related papers with FAISS retrieval and BGE reranking |

## Project structure

```
Trial2Criteria/
├── clinical_rag/     # Core RAG pipeline and CLI
├── vendor/           # PrimeKG and Trial2Vec integrations
├── data/             # Runtime datasets
├── data_prep/        # Dataset build scripts
├── templates/        # LLM prompt templates
├── web/              # FastAPI web UI
└── config.example.json
```

## Setup

```bash
pip install -r requirements.txt
pip install -e vendor/trial2vec

cp config.example.json config.json
cp .env.example .env
# Edit config.json with your Entrez email and API keys
```

Prepare the datasets listed in `data/README.md` (DrugBank, PrimeKG, ClinicalTrials.gov embeddings, and retrieval bases).

PubMed retrieval needs only an NCBI Entrez email — no API key.

## Usage

### Web UI

```bat
start_web.bat
```

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

### Interactive CLI

```bash
python clinical_rag/cli.py
```

## Data preparation

| Domain | Location |
|--------|----------|
| DrugBank | `data_prep/drugbank/` |
| ClinicalTrials.gov | `data_prep/trial_gov/` |
| Trial2Vec embeddings | `vendor/trial2vec/trial_search/Embedding_Retrieve/` |
| PrimeKG features | `vendor/primekg/codes/` |

## Configuration

Copy `config.example.json` to `config.json` and set your credentials. You can also use environment variables:

- `ENTREZ_EMAIL` — NCBI Entrez email
- `ARK_API_KEY` — Volcengine Ark API key for DeepSeek models

## License

Third-party components in `vendor/` retain their original licenses.
