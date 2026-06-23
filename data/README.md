# Runtime data files

Place or symlink the following files here before running retrieval:

## DrugBank (`drugbank/`)

- `drugbank_data_v1.csv` — processed DrugBank table (included in repo if small enough)
- `drug_synonyms.csv` — optional synonym list

Generate from raw XML using scripts in `data_prep/drugbank/`.

## PrimeKG (`primekg/`)

- `disease_features.csv` — disease feature table
- `disease_analysis_prompt.txt`, `disease_selection_prompt.txt` — LLM prompts for disease disambiguation

Source: `vendor/primekg/` upstream or copy from PrimeKG release.

## ClinicalTrials.gov (`trial_gov/`)

### `embeddings/`

- `Phase1_emb_only_fields.pth`
- `Phase2_emb_only_fields.pth`
- `Phase3_emb_only_fields.pth`

Build with `vendor/trial2vec/trial_search/Embedding_Retrieve/NewEncod_only_fields.py`.

### `retrieval_base/`

- `RAG_base_phase1.csv`
- `RAG_base_phase2.csv`
- `RAG_base_phase3.csv`

Build with `data_prep/trial_gov/RAG_base_csv_processor.py`.

### `test_sets/`

- Input CSV for batch processing (e.g. `test_trials_filtered.csv`)

These files are gitignored due to size. See `data_prep/trial_gov/` for preparation scripts.
