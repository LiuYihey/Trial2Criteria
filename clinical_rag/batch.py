import os
# Set Hugging Face endpoint and offline mode
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['HF_HUB_OFFLINE'] = '1'
# os.environ['TRANSFORMERS_OFFLINE'] = '1'

import json
import re
import sys
import time
import argparse
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm as _tqdm

# PowerShell renders Unicode block chars as garbled text; use ASCII bar instead.
# Also output to stdout so PowerShell doesn't colour stderr red.
_is_powershell = bool(os.environ.get("PSModulePath"))

def tqdm(iterable, **kwargs):
    if _is_powershell:
        kwargs.setdefault("ascii", True)
    kwargs.setdefault("file", sys.stdout)
    return _tqdm(iterable, **kwargs)
from http.client import IncompleteRead
from urllib.error import HTTPError
from concurrent.futures import ThreadPoolExecutor, as_completed

script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(script_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from clinical_rag.config import load_config, resolve_path as _resolve_config_path
from clinical_rag.entrez import configure_entrez
from clinical_rag.paths import setup_paths
from clinical_rag.utils import make_json_serializable as _make_json_serializable
from clinical_rag.llm_client import chat_completion

project_root = setup_paths(_project_root)

class EnhancedMedicalSearchLite:
    def __init__(self, config):
        """Initialize with configuration (lazy: heavy models loaded on first use)."""
        self.config = config
        
        configure_entrez(config)
        
        self.faiss_config = config.get("faiss", {
            "index_path": "faiss_indexes",
            "distance_metric": "cosine"
        })
        os.makedirs(self.faiss_config["index_path"], exist_ok=True)
        
        # Lazy-loaded components (initialized on first access)
        self._embeddings = None
        self._reranker = None
        self._disease_extractor = None
        self._drug_matcher = None
        self._drug_matcher_loaded = False
        self._trial_model = None
        self._trial_data_by_phase = None
        self._trial_fields = None
        self._trial_ctx_fields = None
        self._trial_components_loaded = False
        print("--- [INFO] EnhancedMedicalSearchLite initialized (lazy mode). ---")

    # ---- Lazy property loaders ----
    @property
    def embeddings(self):
        if self._embeddings is None:
            import torch
            from langchain_huggingface import HuggingFaceEmbeddings
            try:
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.config["embedding_model"],
                    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
                )
                print(f"--- [INFO] HuggingFace Embeddings loaded: {self.config['embedding_model']} ---")
            except Exception as e:
                print(f"--- [ERROR] Failed to initialize HuggingFaceEmbeddings: {e} ---")
                raise
        return self._embeddings

    @property
    def reranker(self):
        if self._reranker is None:
            from clinical_rag.reranker import BGEReranker
            self._reranker = BGEReranker(model_name=self.config["reranker_model"])
        return self._reranker

    @property
    def disease_extractor(self):
        if self._disease_extractor is None:
            from advanced_disease_extractor import AdvancedDiseaseExtractor
            try:
                disease_csv_path = self._resolve_path(
                    self.config.get("disease_data", {}).get("csv_path", "data/primekg/disease_features.csv")
                )
                self._disease_extractor = AdvancedDiseaseExtractor(
                    data_path=disease_csv_path, config=self.config,
                )
                print(f"--- [INFO] AdvancedDiseaseExtractor initialized: {disease_csv_path} ---")
            except Exception as e:
                print(f"--- [ERROR] Failed to initialize AdvancedDiseaseExtractor: {e} ---")
                self._disease_extractor = False  # sentinel to avoid retry
        return self._disease_extractor if self._disease_extractor is not False else None

    @property
    def drug_matcher(self):
        if not self._drug_matcher_loaded:
            self._drug_matcher_loaded = True
            from clinical_rag.domains.drugbank.matcher import DrugMatcher
            try:
                drug_csv_path = self._resolve_path(
                    self.config.get("drug_data", {}).get("csv_path", "data/drugbank/drugbank_data_v1.csv")
                )
                use_synonyms = self.config.get("drug_data", {}).get("use_synonyms", False)
                self._drug_matcher = DrugMatcher(drug_csv_path, preprocess=True, use_synonyms=use_synonyms)
                print(f"--- [INFO] DrugMatcher initialized: {drug_csv_path} ---")
            except Exception as e:
                print(f"--- [ERROR] Failed to initialize DrugMatcher: {e} ---")
                self._drug_matcher = None
        return self._drug_matcher

    # ---- Trial component properties (trigger lazy load on first access) ----
    @property
    def trial_model(self):
        self._ensure_trial_components()
        return self._trial_model

    @property
    def trial_data_by_phase(self):
        self._ensure_trial_components()
        return self._trial_data_by_phase

    @property
    def trial_fields(self):
        self._ensure_trial_components()
        return self._trial_fields

    @property
    def trial_ctx_fields(self):
        self._ensure_trial_components()
        return self._trial_ctx_fields

    def _ensure_trial_components(self):
        """Load trial retrieval components on first use."""
        if self._trial_components_loaded:
            return  # already loaded
        print("--- [INFO] Loading trial components (importing torch + Trial2Vec, this may take a few minutes)... ---", flush=True)
        self._trial_components_loaded = True
        import torch
        print("--- [INFO] torch imported OK ---", flush=True)
        from torch.serialization import add_safe_globals
        from trial2vec import Trial2Vec
        print("--- [INFO] Trial2Vec imported OK ---", flush=True)

        add_safe_globals(['numpy.core.multiarray._reconstruct'])

        self._trial_data_by_phase = {}
        try:
            trial_cfg = self.config.get("trial_retrieval", {})
            active_phases = trial_cfg.get("active_phases") or list(trial_cfg.get("phases", {}).keys())
            print(f"--- [INFO] Loading trial data for active phases: {active_phases} ---")
            for phase_key in active_phases:
                phase_config = trial_cfg.get("phases", {}).get(phase_key)
                if not phase_config:
                    continue
                emb_path = self._resolve_path(phase_config.get("emb_path"))
                data_path = self._resolve_path(phase_config.get("data_path"))
                if not emb_path or not os.path.exists(emb_path):
                    print(f"--- [WARN] Embedding path for {phase_key} invalid: {emb_path}. Skipping. ---")
                    continue
                if not data_path or not os.path.exists(data_path):
                    print(f"--- [WARN] Data path for {phase_key} invalid: {data_path}. Skipping. ---")
                    continue
                phase_emb_dict = torch.load(emb_path, weights_only=False)["emb"]
                phase_df = pd.read_csv(data_path)
                if "nct_id" in phase_df.columns:
                    phase_df = phase_df.set_index("nct_id")
                self._trial_data_by_phase[phase_key] = {
                    "emb_dict": phase_emb_dict, "raw_df": phase_df,
                    "all_ids": list(phase_emb_dict.keys()), "all_embs": list(phase_emb_dict.values())
                }
                print(f"--- [SUCCESS] Loaded data for trial phase '{phase_key}' ---")
            if not self._trial_data_by_phase:
                print("--- [WARN] No trial data was loaded. ---")
            model_dir = self._resolve_path(trial_cfg.get("model_dir"))
            bert_name = self._resolve_path(trial_cfg.get("bert_name"))
            self._trial_model = Trial2Vec(bert_name=bert_name)
            self._trial_model.from_pretrained(model_dir)
            print(f"--- [INFO] Trial2Vec model loaded from {model_dir} ---")
            with open(os.path.join(model_dir, "model_config.json"), "r", encoding="utf-8") as f:
                mcfg = json.load(f)
            self._trial_fields = mcfg.get("fields", ["title", "intervention_name", "disease", "keyword"])
            self._trial_model.config['ctx_fields'] = []
            self._trial_model.model.config['ctx_fields'] = []
            self._trial_ctx_fields = []
            print("--- [SUCCESS] Trial2Vec components initialized ---")
        except Exception as e:
            print(f"--- [ERROR] Failed to initialize trial retrieval components: {e} ---")
            self._trial_model = None
    
    def match_drug(self, drug_name):
        if not self.drug_matcher or not drug_name:
            return None
        try:
            # The search method might return None, which would cause a subscript error.
            result = self.drug_matcher.search(drug_name)
            if result and result.get("success"):
                if "result" in result and "row" in result["result"]:
                    result["result"]["row"] = dict(result["result"]["row"])
                return result
            return None
        except Exception as e:
            print(f"--- [ERROR] Drug matching error: {e} ---")
        return None
    
    def format_drug_info(self, drug_match):
        if not drug_match or not drug_match.get("success") or "result" not in drug_match:
            return "No drug match found"
        
        drug_data = drug_match["result"]["row"]
        def get_clean_value(text):
            if text is None: return None
            text = str(text)
            if text.strip().lower() in ['', 'nan', 'none']: return None
            return re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()
        
        fields_to_display = [
            ('name', 'Name'), ('SMILES', 'SMILES'), ('molecular_formula', 'Molecular Formula'),
            ('indication', 'Indication'), ('mechanism-of-action', 'Mechanism of Action'),
            ('pharmacodynamics', 'Pharmacodynamics'), ('absorption', 'Absorption'),
            ('metabolism', 'Metabolism'), ('half-life', 'Half-life'),
            ('protein-binding', 'Protein Binding'), ('route-of-elimination', 'Route of Elimination'),
            ('volume-of-distribution', 'Volume of Distribution'), ('clearance', 'Clearance'),
            ('toxicity', 'Toxicity')
        ]
        output = [f"* {display_name}: {value}" for key, display_name in fields_to_display if (value := get_clean_value(drug_data.get(key)))]
        return "\n".join(output) if output else "No drug match found"

    def search_pubmed(self, disease_term, drug_terms, other_terms=None):
        from Bio import Entrez
        primary_query = ""
        if disease_term:
            if drug_terms:
                drug_query_part = " OR ".join([f"({term})" for term in drug_terms if term])
                primary_query = f"({disease_term}) AND ({drug_query_part})"
            else:
                primary_query = f"({disease_term})"
        
        def _execute_search(query):
            if not query: return []
            print(f"\n--- [START] Searching PubMed for '{query}' ---")
            
            pubmed_config = self.config.get("pubmed_search", {})
            retmax = min(pubmed_config.get("max_results", 50), 9990)
            
            try:
                with Entrez.esearch(db="pubmed", term=query, retmax=0) as handle:
                    total_count = int(Entrez.read(handle)["Count"])
                
                if total_count == 0:
                    print(f"--- [INFO] No papers found on PubMed for '{query}' ---")
                    return []
                
                total_to_fetch = min(total_count, retmax)
                print(f"--- [INFO] Found {total_count} papers, fetching details for up to {total_to_fetch} ---")
                
                with Entrez.esearch(db="pubmed", term=query, retmax=total_to_fetch) as handle:
                    pmid_list = Entrez.read(handle)["IdList"]
                
                return self._fetch_paper_details(pmid_list, pubmed_config)
            except Exception as e:
                print(f"--- [ERROR] Failed to search PubMed: {e} ---")
                return []

        papers = _execute_search(primary_query)
        
        if not papers and other_terms and self.config.get("pubmed_search", {}).get("perform_secondary_search", True):
            print("\n--- [INFO] Primary search failed. Trying secondary search. ---")
            secondary_query = " AND ".join([f"({term})" for term in other_terms if term])
            papers = _execute_search(secondary_query)
        
        if not papers and disease_term and self.config.get("pubmed_search", {}).get("perform_fallback_search", True):
            print("\n--- [INFO] All searches failed. Trying fallback with disease term only. ---")
            papers = _execute_search(f"({disease_term})")
        
        return papers

    def _fetch_paper_details(self, pmid_list, pubmed_config):
        from Bio import Entrez
        if not pmid_list: return []
        
        papers = []
        DETAILS_BATCH_SIZE = pubmed_config.get("efetch_batch_size", 50)
        MAX_WORKERS = pubmed_config.get("max_workers", 5)
        batches = [pmid_list[i:i + DETAILS_BATCH_SIZE] for i in range(0, len(pmid_list), DETAILS_BATCH_SIZE)]

        def _fetch_batch_details(batch_pmids):
            for attempt in range(3):
                try:
                    with Entrez.epost(db="pubmed", id=",".join(batch_pmids)) as handle:
                        result = Entrez.read(handle)
                    with Entrez.efetch(db="pubmed", retmode="xml", webenv=result["WebEnv"], query_key=result["QueryKey"]) as handle:
                        records = Entrez.read(handle)
                        
                        batch_papers = []
                        for article in records.get("PubmedArticle", []):
                            try:
                                abstract_list = article["MedlineCitation"]["Article"].get("Abstract", {}).get("AbstractText", [])
                                if not abstract_list: continue # Skip if no abstract

                                batch_papers.append({
                                    "PMID": article["MedlineCitation"]["PMID"],
                                    "Title": article["MedlineCitation"]["Article"]["ArticleTitle"],
                                    "Abstract": " ".join(map(str, abstract_list)),
                                })
                            except (KeyError, IndexError):
                                continue
                        return batch_papers
                
                except (HTTPError, IncompleteRead) as e:
                    print(f"--- [WARN] Network error on batch, attempt {attempt+1}/3: {e}. Retrying... ---")
                    time.sleep(5)
                except Exception as e:
                    print(f"--- [ERROR] Unexpected error on batch: {e} ---")
                    break
            return []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_batch = {executor.submit(_fetch_batch_details, batch): batch for batch in batches}
            for future in tqdm(as_completed(future_to_batch), total=len(batches), desc="Fetching Paper Details"):
                papers.extend(future.result())
                
        print(f"--- [SUCCESS] Processed {len(papers)} papers from PubMed ---")
        return papers

    def create_vector_store(self, papers, collection_name):
        from langchain.schema import Document
        from langchain_community.vectorstores import FAISS
        if not papers: return None
        documents = [Document(
            page_content=f"Title: {p['Title']}\nAbstract: {p['Abstract']}",
            metadata={"pmid": p['PMID'], "title": p['Title']}
        ) for p in papers]
        
        try:
            index_path = os.path.join(self.faiss_config["index_path"], collection_name)
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            vector_db = FAISS.from_documents(documents=documents, embedding=self.embeddings)
            vector_db.save_local(index_path)
            print(f"--- [SUCCESS] Created vector store '{collection_name}' at {index_path} ---")
            return vector_db
        except Exception as e:
            print(f"--- [ERROR] Failed to create vector store: {e} ---")
            return None

    def search_similar_papers(self, query, collection_name, top_k=5, similarity_threshold=0.5):
        """
        Search for papers similar to the query using FAISS, with verbose logging.
        """
        from langchain_community.vectorstores import FAISS
        # This function is now more concise and does not print full paper details to the console.
        print(f"\n--- [START] Searching for similar papers to '{query}' in collection '{collection_name}' ---")

        try:
            index_path = os.path.join(self.faiss_config["index_path"], collection_name)
            if not os.path.exists(index_path):
                print(f"--- [ERROR] FAISS index path does not exist: {index_path} ---")
                return []
            
            vector_db = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
            search_k = min(30, top_k * 3)
            results = vector_db.similarity_search_with_score(query, k=search_k)
            
            docs = [doc for doc, score in results]
            if not docs:
                print("--- [INFO] No matching documents found in vector store. ---")
                return []

            print(f"--- [INFO] Found {len(docs)} initial matches. Reranking top {top_k} results... ---")
            reranked_docs = self.reranker.rerank(query, docs, top_k=top_k)
            
            pre_filter_count = len(reranked_docs)
            filtered_docs = [doc for doc in reranked_docs if doc.metadata.get("bge_score", 0) > similarity_threshold]
            
            if len(filtered_docs) < pre_filter_count:
                print(f"--- [INFO] Filtered out {pre_filter_count - len(filtered_docs)} papers with similarity score below {similarity_threshold} ---")

            if not filtered_docs:
                print(f"--- [WARN] All papers were filtered out due to similarity threshold of {similarity_threshold}. ---")
                return []

            print(f"--- [SUCCESS] Found {len(filtered_docs)} relevant papers after reranking and filtering. ---")
            return filtered_docs

        except Exception as e:
            print(f"--- [ERROR] Failed during similar paper search: {str(e)} ---")
            return []
    
    def retrieve_trials_by_phase(self, title, disease, drug, keywords, top_k=5, similarity_threshold=0.5):
        if not self.trial_model or not self.trial_data_by_phase:
            print("--- [WARN] Trial retrieval components not initialized. Skipping. ---")
            return None
            
        initial_retrieve_count = 10
        
        query_df = pd.DataFrame([{"nct_id": "query", "title": title or "", "intervention_name": drug or "",
                                  "disease": disease or "", "keyword": keywords or ""}])
        
        try:
            query_emb = list(self.trial_model.encode(
                {"x": query_df, "fields": self.trial_fields, "ctx_fields": self.trial_ctx_fields, "tag": "nct_id"},
                return_dict=True, verbose=False
            ).values())[0]
        except Exception as e:
            print(f"--- [ERROR] Failed to encode trial query: {e} ---")
            return None
        
        all_phase_results = {}
        for phase_key, phase_data in self.trial_data_by_phase.items():
            result_df = self._retrieve_similar_trials_for_phase(query_emb, phase_data, initial_retrieve_count)
            if result_df.empty:
                all_phase_results[phase_key] = result_df
                continue
            
            # Filtering logic...
            similarity_filtered_df = self._filter_trials_by_similarity(result_df, similarity_threshold)
            disease_filtered_df = self._filter_trials_by_disease(similarity_filtered_df, disease)
            final_filtered_df = self._filter_trials_by_title_match(disease_filtered_df, title)
            
            if not final_filtered_df.empty:
                sorted_df = final_filtered_df.sort_values('similarity', ascending=False)
                all_phase_results[phase_key] = sorted_df.iloc[:min(top_k, len(sorted_df))]
            else:
                all_phase_results[phase_key] = pd.DataFrame()
            
        return all_phase_results
    
    def _retrieve_similar_trials_for_phase(self, query_emb, phase_data, top_k=5):
        if not phase_data["all_ids"]: return pd.DataFrame()
        
        def _normalize(arr): return arr / np.linalg.norm(arr, axis=-1, keepdims=True)
        
        query_emb_norm = _normalize(query_emb.reshape(1, -1))
        all_embs_norm = _normalize(np.stack(phase_data["all_embs"]))
        sims = np.dot(all_embs_norm, query_emb_norm.T).reshape(-1)
        
        actual_top_k = min(top_k, len(phase_data["all_ids"]))
        top_idx = np.argsort(sims)[::-1][:actual_top_k]
        
        top_ids = [phase_data["all_ids"][i] for i in top_idx]
        top_sims = sims[top_idx]

        result_df = phase_data["raw_df"].loc[top_ids].copy()
        result_df["similarity"] = top_sims
        return result_df

    def _filter_trials_by_similarity(self, df, threshold):
        return df[df['similarity'] > threshold] if not df.empty else df

    def _filter_trials_by_disease(self, trials_df, disease_keyword):
        """Filter trial DataFrame based on disease keyword matching."""
        if trials_df.empty or not disease_keyword or 'disease' not in trials_df.columns:
            return trials_df

        # Normalize the input keyword into a set of words
        # Use regex to find all whole words, avoiding parts of words or punctuation
        input_disease_words = set(re.findall(r'\b\w+\b', disease_keyword.lower()))
        if not input_disease_words:
            return trials_df

        def has_matching_word(trial_disease):
            # Ensure the trial's disease field is a valid string
            if not isinstance(trial_disease, str) or not trial_disease.strip():
                return False
        
            # Normalize the trial's disease string into a set of words
            trial_disease_words = set(re.findall(r'\b\w+\b', trial_disease.lower()))
            
            # Return True if there is any intersection between the two word sets
            return not input_disease_words.isdisjoint(trial_disease_words)

        # Apply the filter and return the filtered DataFrame
        mask = trials_df['disease'].apply(has_matching_word)
        return trials_df[mask]
    
    def _normalize_title(self, title):
        if not isinstance(title, str): return ""
        return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', title.lower())).strip()

    def _filter_trials_by_title_match(self, df, query_title):
        if df.empty or not query_title: return df
        norm_query = self._normalize_title(query_title)
        return df[df['title'].apply(lambda t: self._normalize_title(t) != norm_query)]
        
    def _resolve_path(self, path_str):
        return _resolve_config_path(path_str, project_root)

    def _analyze_disease_terms(self, disease_terms_str, title=None):
        """
        Uses an LLM to analyze a string of disease terms and return a single, standardized term.
        The optional title parameter provides additional context for disease abbreviations.
        """
        print(f"--- [INFO] Starting AI analysis for disease terms: '{disease_terms_str}' ---")
        
        # 如果提供了title，添加到日志并作为额外上下文
        if title:
            print(f"--- [INFO] Including research title as context: '{title}' ---")
        
        try:
            with open(self._resolve_path("data/primekg/disease_analysis_prompt.txt"), "r", encoding="utf-8") as f:
                prompt_template = f.read()
            
            # 如果有title，将disease_terms和title一起作为上下文
            if title:
                context_input = f"{disease_terms_str} (from research: {title})"
            else:
                context_input = disease_terms_str
            
            # 使用正确的变量名格式化prompt
            prompt = prompt_template.format(input=context_input)
            
            result = chat_completion(
                [
                    {"role": "system", "content": "You are a professional medical expert analyzing disease terms."},
                    {"role": "user", "content": prompt},
                ],
                self.config,
                response_format="json_object",
                temperature=0.1,
                max_tokens=4096,
            )
            if not result:
                raise RuntimeError("Empty LLM response")
            
            # Enhanced JSON parsing to handle potential formatting issues
            json_str = result
            # First, try to find a JSON object within ``` markers
            markdown_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', json_str)
            if markdown_match:
                json_str = markdown_match.group(1)
            
            # If no markdown block, try to find a JSON object directly
            else:
                json_match = re.search(r'\{[\s\S]*\}', json_str)
                if json_match:
                    json_str = json_match.group(0)

            analysis_data = json.loads(json_str)
            best_term = analysis_data.get("best_term")
            print(f"--- [SUCCESS] AI analysis complete. Standardized term: '{best_term}' ---")
            
            # Format the full analysis for the final output
            analysis_text = f"Standardized Term: {best_term}\nDefinition: {analysis_data.get('definition', 'N/A')}\nKey Characteristics: {', '.join(analysis_data.get('key_characteristics', []))}"
            
            return best_term, analysis_text
            
        except Exception as e:
            print(f"--- [ERROR] Failed during AI disease analysis: {e} ---")
            # 添加更详细的错误信息，包括完整的堆栈跟踪
            import traceback
            print(f"--- [ERROR] Traceback: {traceback.format_exc()} ---")
            # Fallback to the original input if analysis fails
            return disease_terms_str, f"AI analysis failed: {e}"

    def process_trial_from_csv_row(self, row):
        """
        Processes a single row from the trial CSV to generate context data.
        """
        # --- 1. Initialize result structure and status tracking ---
        result = {
            "nctId": row.get("nctId"),
            "trial_title": row.get("officialTitle"),
            "Disease_info": None,
            "Drugbank_info": None,
            "Papers_info": None,
            "Trial_description": row.get("briefSummary") or "none",
            "Outcome_measure": row.get("primaryOutcomes") or "none",
            "Phase1_clinical_trials": None,
            "Phase2_clinical_trials": None,
            "Phase3_clinical_trials": None,
            "eligibilityCriteria": row.get("eligibilityCriteria") or "none"
        }
        status = {key: 0 for key in result}
        status["nctId"] = 1
        status["trial_title"] = 1 if result["trial_title"] else 0
        status["Trial_description"] = 1 if result["Trial_description"] != "none" else 0
        status["Outcome_measure"] = 1 if result["Outcome_measure"] != "none" else 0
        status["eligibilityCriteria"] = 1 if result["eligibilityCriteria"] != "none" else 0

        # --- 2. Extract Disease and Drug keywords from CSV data ---
        # Disease
        disease_keywords = []
        try:
            conditions_str = row.get('conditions')
            if pd.notna(conditions_str):
                # Safely evaluate the string representation of a list
                disease_keywords = ast.literal_eval(conditions_str)
        except (ValueError, SyntaxError):
            # Fallback for simple comma-separated strings
            disease_keywords = [d.strip() for d in str(row.get('conditions', '')).split(',')]
        disease_keywords = [d for d in disease_keywords if d] # Clean empty strings

        # Drug
        drug_keywords = []
        interventions_str = row.get('interventions')
        if pd.notna(interventions_str):
            # Regex to find all drug names associated with "type": "DRUG"
            drug_keywords = re.findall(r'\{\s*"type":\s*"DRUG",\s*"name":\s*"([^"]+)"', interventions_str)

        # --- 3. Retrieve Disease Information (New Logic) ---
        disease_info_text = "none"
        final_disease_term = None
        if disease_keywords:
            disease_input_str = ", ".join(disease_keywords)
            best_term, _ = self._analyze_disease_terms(disease_input_str, title=result["trial_title"])
            
            if best_term and self.disease_extractor:
                final_disease_term = best_term
                print(f"--- [INFO] Retrieving structured info for standardized term: '{final_disease_term}' ---")
                disease_info = self.disease_extractor.get_info(final_disease_term)
                
                if disease_info and disease_info.get('best_match_name'):
                    def _format_disease_details(info):
                        lines = [f"* Information for '{info.pop('best_match_name', 'N/A')}':"]
                        lines.extend(f"* {k.replace('_', ' ').title()}: {', '.join(map(str, v)) if isinstance(v, list) else v}" for k, v in info.items())
                        return "\n".join(lines)
                    
                    disease_info_text = _format_disease_details(disease_info)
                else:
                    disease_info_text = "none"
            else:
                disease_info_text = "none"
                final_disease_term = disease_keywords[0] if disease_keywords else ""
        
        result["Disease_info"] = disease_info_text
        status["Disease_info"] = 1 if disease_info_text != "none" and "failed" not in disease_info_text.lower() else 0

        # --- 4. Retrieve Drug Information ---
        all_drug_info = []
        if self.drug_matcher and drug_keywords:
            print(f"--- [INFO] Retrieving info for drugs: {drug_keywords} ---")
            for drug in drug_keywords:
                match = self.match_drug(drug)
                formatted_info = self.format_drug_info(match)
                if "No drug match found" not in formatted_info:
                    all_drug_info.append(formatted_info)
        
        if all_drug_info:
            result["Drugbank_info"] = "\n\n".join(all_drug_info)
            status["Drugbank_info"] = 1
        else:
            result["Drugbank_info"] = "none"

        # --- 5. Retrieve Relevant Medical Papers ---
        papers = []
        pmids_str = row.get('pmids')
        if pd.notna(pmids_str) and pmids_str:
            print(f"--- [INFO] Found PMIDs in CSV: {pmids_str}. Fetching directly. ---")
            try:
                pmid_list = ast.literal_eval(pmids_str)[:5] # Take first 5
                papers = self._fetch_paper_details(pmid_list, self.config.get("pubmed_search", {}))
            except (ValueError, SyntaxError):
                papers = []
        else:
            print("--- [INFO] No PMIDs in CSV. Searching PubMed based on keywords. ---")
            papers = self.search_pubmed(
                disease_term=final_disease_term,
                drug_terms=drug_keywords
            )
            if papers:
                collection_name = f"{result['nctId']}_{int(time.time())}"
                self.create_vector_store(papers, collection_name)
                similar_docs = self.search_similar_papers(result['trial_title'], collection_name, top_k=5)
                papers = [{"Title": doc.metadata.get('title'), "Abstract": doc.page_content.split('Abstract: ')[-1], "PMID": doc.metadata.get('pmid')} for doc in similar_docs]
                try:
                    import shutil
                    shutil.rmtree(os.path.join(self.faiss_config["index_path"], collection_name))
                except Exception as e:
                    print(f"--- [WARN] Could not clean up FAISS index '{collection_name}': {e} ---")

        if papers:
            def _format_papers(paper_list):
                lines = []
                for p in paper_list:
                    lines.append(f"* PMID {p.get('PMID', 'N/A')} Title: {p.get('Title', 'N/A')}")
                    lines.append(f"  Abstract: {p.get('Abstract', 'N/A')}")
                return "\n".join(lines)
            result["Papers_info"] = _format_papers(papers)
            status["Papers_info"] = 1
        else:
            result["Papers_info"] = "none"

        # --- 6. Retrieve Similar Clinical Trials ---
        csv_keywords = row.get('keywords')
        keywords_for_retrieval = ""
        if pd.notna(csv_keywords) and csv_keywords.lower() != 'none':
            keywords_for_retrieval = csv_keywords

        trial_results = self.retrieve_trials_by_phase(
            title=result['trial_title'],
            disease=final_disease_term or (disease_keywords[0] if disease_keywords else ""),
            drug=" ".join(drug_keywords),
            keywords=keywords_for_retrieval,
            top_k=5,
            similarity_threshold=self.config.get("trial_retrieval", {}).get("trial_similarity_threshold", 0.5)
        )
        
        def _format_trials(df):
            if df is None or df.empty: return "none"
            lines = []
            for i, trial_row in df.reset_index().iterrows():
                lines.append(f"{i+1}. Similar Trial {i+1} NCT ID: {trial_row.get('nct_id', 'N/A')}")
                for col, val in trial_row.items():
                    if col.lower() in ['nct_id', 'similarity'] or pd.isna(val) or str(val).strip() == '': continue
                    lines.append(f"* {col.replace('_', ' ').title()}: {val}")
                lines.append("")
            return "\n".join(lines).strip()

        if trial_results:
            for phase in ["phase1", "phase2", "phase3"]:
                formatted_trials = _format_trials(trial_results.get(phase))
                result[f"Phase{phase[-1]}_clinical_trials"] = formatted_trials
                status[f"Phase{phase[-1]}_clinical_trials"] = 1 if formatted_trials != "none" else 0
        else:
            result["Phase1_clinical_trials"] = result["Phase2_clinical_trials"] = result["Phase3_clinical_trials"] = "none"

        return result, status

    def _extract_bench_disease_terms(self, title: str, eligibility: str) -> str:
        """Heuristic disease term extraction for bench JSON (no conditions field)."""
        candidates = []
        title_patterns = [
            r"[Mm]anagement of (.+?) in ",
            r"[Tt]reatment of (.+?) in ",
            r"[Ee]fficacy of .+ in (.+?)$",
            r"[Pp]atients with ([^,\n]+)",
            r"[Dd]iagnosis of ([^:\n]+)",
        ]
        for pattern in title_patterns:
            match = re.search(pattern, title or "")
            if match:
                candidates.append(match.group(1).strip(" ."))

        eligibility_patterns = [
            r"[Dd]iagnosed with ([^:\n*]+)",
            r"[Pp]atients with ([^,\n]+)",
            r"with ([^.\n]{3,60}(?:cirrhosis|diabetes|mellitus|peritonitis|alopecia|hypertension|infection))",
        ]
        for pattern in eligibility_patterns:
            match = re.search(pattern, eligibility or "", re.I)
            if match:
                candidates.append(match.group(1).strip(" .:*"))

        combined = f"{title}\n{eligibility}".lower()
        known_phrases = [
            "spontaneous bacterial peritonitis",
            "gestational diabetes mellitus",
            "liver cirrhosis",
            "androgenetic alopecia",
            "alopecia areata",
            "type 2 diabetes",
            "hypertension",
        ]
        for phrase in known_phrases:
            if phrase in combined:
                candidates.append(phrase)

        seen = set()
        unique = []
        for item in candidates:
            key = item.lower()
            if key not in seen and len(item) > 3:
                seen.add(key)
                unique.append(item)
        if unique:
            return ", ".join(unique[:3])
        return (title or "")[:120]

    def process_bench_record(self, record):
        """
        Process a ClinicalTrials.gov JSON record (e.g. from final_bench.json).
        Disease terms are inferred from the trial title when conditions are absent.
        """
        ps = record.get("protocolSection", {})
        ident = ps.get("identificationModule", {})
        nct_id = ident.get("nctId")
        title = ident.get("officialTitle", "")

        interventions = ps.get("armsInterventionsModule", {}).get("interventions", [])
        drug_keywords = [
            item.get("name", "").strip()
            for item in interventions
            if item.get("type") == "DRUG" and item.get("name")
        ]

        primary_outcomes = ps.get("outcomesModule", {}).get("primaryOutcomes", [])
        outcome_text = "none"
        if primary_outcomes:
            outcome_text = json.dumps(primary_outcomes, ensure_ascii=False)

        eligibility = ps.get("eligibilityModule", {}).get("eligibilityCriteria") or "none"

        result = {
            "nctId": nct_id,
            "trial_title": title,
            "Disease_info": None,
            "Drugbank_info": None,
            "Papers_info": None,
            "Trial_description": "none",
            "Outcome_measure": outcome_text,
            "Phase1_clinical_trials": None,
            "Phase2_clinical_trials": None,
            "Phase3_clinical_trials": None,
            "eligibilityCriteria": eligibility,
        }
        status = {key: 0 for key in result}
        status["nctId"] = 1 if nct_id else 0
        status["trial_title"] = 1 if title else 0
        status["Outcome_measure"] = 1 if outcome_text != "none" else 0
        status["eligibilityCriteria"] = 1 if eligibility != "none" else 0

        disease_info_text = "none"
        final_disease_term = None
        disease_input = self._extract_bench_disease_terms(title, eligibility)
        if disease_input:
            best_term, analysis = self._analyze_disease_terms(disease_input, title=title)
            if analysis and "failed" in str(analysis).lower():
                final_disease_term = disease_input.split(",")[0].strip()
            else:
                final_disease_term = best_term or disease_input.split(",")[0].strip()
            if len(final_disease_term) > 80:
                final_disease_term = disease_input.split(",")[0].strip()
            if final_disease_term and self.disease_extractor:
                print(f"--- [INFO] Retrieving structured info for standardized term: '{final_disease_term}' ---")
                disease_info = self.disease_extractor.get_info(final_disease_term)
                if disease_info and disease_info.get("best_match_name"):
                    lines = [f"* Information for '{disease_info.pop('best_match_name', 'N/A')}':"]
                    lines.extend(
                        f"* {k.replace('_', ' ').title()}: {', '.join(map(str, v)) if isinstance(v, list) else v}"
                        for k, v in disease_info.items()
                    )
                    disease_info_text = "\n".join(lines)
            elif final_disease_term:
                pass

        result["Disease_info"] = disease_info_text
        status["Disease_info"] = (
            1 if disease_info_text != "none" and "failed" not in disease_info_text.lower() else 0
        )

        all_drug_info = []
        if self.drug_matcher and drug_keywords:
            print(f"--- [INFO] Retrieving info for drugs: {drug_keywords} ---")
            for drug in drug_keywords:
                match = self.match_drug(drug)
                formatted_info = self.format_drug_info(match)
                if "No drug match found" not in formatted_info:
                    all_drug_info.append(formatted_info)

        if all_drug_info:
            result["Drugbank_info"] = "\n\n".join(all_drug_info)
            status["Drugbank_info"] = 1
        else:
            result["Drugbank_info"] = "none"

        papers = self.search_pubmed(
            disease_term=final_disease_term or disease_input,
            drug_terms=drug_keywords,
        )
        if papers:
            collection_name = f"{nct_id}_{int(time.time())}"
            self.create_vector_store(papers, collection_name)
            similar_docs = self.search_similar_papers(title, collection_name, top_k=5)
            papers = [
                {
                    "Title": doc.metadata.get("title"),
                    "Abstract": doc.page_content.split("Abstract: ")[-1],
                    "PMID": doc.metadata.get("pmid"),
                }
                for doc in similar_docs
            ]
            try:
                import shutil
                shutil.rmtree(os.path.join(self.faiss_config["index_path"], collection_name))
            except Exception as exc:
                print(f"--- [WARN] Could not clean up FAISS index '{collection_name}': {exc} ---")

        if papers:
            paper_lines = []
            for paper in papers:
                paper_lines.append(f"* PMID {paper.get('PMID', 'N/A')} Title: {paper.get('Title', 'N/A')}")
                paper_lines.append(f"  Abstract: {paper.get('Abstract', 'N/A')}")
            result["Papers_info"] = "\n".join(paper_lines)
            status["Papers_info"] = 1
        else:
            result["Papers_info"] = "none"

        trial_results = self.retrieve_trials_by_phase(
            title=title,
            disease=final_disease_term or disease_input,
            drug=" ".join(drug_keywords),
            keywords="",
            top_k=5,
            similarity_threshold=self.config.get("trial_retrieval", {}).get("trial_similarity_threshold", 0.5),
        )

        def _format_trials(df):
            if df is None or df.empty:
                return "none"
            lines = []
            for i, trial_row in df.reset_index().iterrows():
                lines.append(f"{i + 1}. Similar Trial {i + 1} NCT ID: {trial_row.get('nct_id', 'N/A')}")
                for col, val in trial_row.items():
                    if col.lower() in ["nct_id", "similarity"] or pd.isna(val) or str(val).strip() == "":
                        continue
                    lines.append(f"* {col.replace('_', ' ').title()}: {val}")
                lines.append("")
            return "\n".join(lines).strip()

        if trial_results:
            for phase in ["phase1", "phase2", "phase3"]:
                formatted_trials = _format_trials(trial_results.get(phase))
                result[f"Phase{phase[-1]}_clinical_trials"] = formatted_trials
                status[f"Phase{phase[-1]}_clinical_trials"] = 1 if formatted_trials != "none" else 0
        else:
            result["Phase1_clinical_trials"] = result["Phase2_clinical_trials"] = result["Phase3_clinical_trials"] = "none"

        return result, status


def main():
    parser = argparse.ArgumentParser(description="Batch process clinical trials from a CSV to generate RAG context.")
    parser.add_argument("--nct_id", type=str, help="Process a single trial by its NCT ID.")
    parser.add_argument("--input_csv", type=str, default="data/trial_gov/test_sets/test_trials_filtered.csv", help="Path to the input CSV file.")
    parser.add_argument("--output_file", type=str, default="outputs/rag/trials_rag.jsonl", help="Path to the output JSONL file.")
    parser.add_argument("--report_file", type=str, default="outputs/rag/retrieval_report.csv", help="Path to the retrieval status report CSV.")
    args = parser.parse_args()

    config = load_config()
    search_system = EnhancedMedicalSearchLite(config)
    
    # --- Load input CSV and prepare for processing ---
    try:
        df = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        print(f"--- [FATAL] Input file not found at: {args.input_csv} ---")
        return

    # --- Handle resume logic ---
    processed_ids = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)['nctId'])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"--- [INFO] Resuming. Found {len(processed_ids)} already processed trials. ---")

    # --- Filter DataFrame based on CLI arguments ---
    if args.nct_id:
        df_to_process = df[df['nctId'] == args.nct_id].copy()
        if df_to_process.empty:
            print(f"--- [ERROR] NCT ID '{args.nct_id}' not found in the input CSV. ---")
            return
    else:
        df_to_process = df[~df['nctId'].isin(processed_ids)].copy()

    if df_to_process.empty:
        print("--- [INFO] No new trials to process. ---")
        return

    # --- Initialize or load status report ---
    report_columns = ["nctId", "trial_title", "Disease_info", "Drugbank_info", "Papers_info", 
                      "Trial_description", "Outcome_measure", "Phase1_clinical_trials", 
                      "Phase2_clinical_trials", "Phase3_clinical_trials", "eligibilityCriteria"]
    if os.path.exists(args.report_file):
        status_df = pd.read_csv(args.report_file)
    else:
        status_df = pd.DataFrame(columns=report_columns)
    
    # --- Main processing loop ---
    print(f"--- [INFO] Starting processing for {len(df_to_process)} trials. ---")
    
    with open(args.output_file, 'a', encoding='utf-8') as f_out:
        for _, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="Processing Trials"):
            nct_id = row['nctId']
            print(f"\n{'='*20} Processing NCT ID: {nct_id} {'='*20}")
            
            result, status = search_system.process_trial_from_csv_row(row)
            
            # Write result to JSONL
            f_out.write(json.dumps(_make_json_serializable(result)) + '\n')
            
            # Update status report
            new_status_row = pd.DataFrame([status], columns=report_columns)
            new_status_row["nctId"] = nct_id # Ensure NCT ID is set
            status_df = pd.concat([status_df, new_status_row], ignore_index=True)
            
            # Save report periodically
            status_df.to_csv(args.report_file, index=False)
            
    print(f"\n--- [SUCCESS] Processing complete. ---")
    print(f"--- [INFO] Output saved to: {args.output_file} ---")
    print(f"--- [INFO] Retrieval status report saved to: {args.report_file} ---")


if __name__ == "__main__":
    main() 