# Data preparation scripts

Scripts here build the datasets consumed by `clinical_rag` at runtime.
They are **not** imported by the RAG pipeline — run them manually when refreshing data.

## DrugBank (`drugbank/`)

| Script | Purpose |
|--------|---------|
| `xml2json.py` | DrugBank XML → JSON |
| `drugjson2csv.py` | JSON → CSV |
| `clean_csv.py` | Clean and normalize CSV |
| `extractjson.py` | Extract fields from JSON export |

Output target: `data/drugbank/drugbank_data_v1.csv`

## ClinicalTrials.gov (`trial_gov/`)

| Script | Purpose |
|--------|---------|
| `extract_trials_csv2jsonl.py` | CSV → JSONL |
| `filter_trials_byjsonl.py` | Filter trials by JSONL IDs |
| `filter_trials_(RAG筛选全1).py` | Filter by retrieval success report |
| `clean_text_data_*.py` | Text normalization |
| `reformat_criteria_improved_*.py` | Criteria reformatting |
| `RAG_base_csv_processor.py` | Build phase retrieval-base CSVs |

Output targets: `data/trial_gov/retrieval_base/`, `data/trial_gov/test_sets/`

Update hardcoded paths inside these scripts to match the new `data/` layout when re-running them.
