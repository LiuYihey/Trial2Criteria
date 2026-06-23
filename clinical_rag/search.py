import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import json
import sys
import pandas as pd
from Bio import Entrez
import datetime
from tqdm import tqdm as _tqdm

_is_powershell = bool(os.environ.get("PSModulePath"))

def tqdm(iterable, **kwargs):
    if _is_powershell:
        kwargs.setdefault("ascii", True)
    kwargs.setdefault("file", sys.stdout)
    return _tqdm(iterable, **kwargs)
from dotenv import load_dotenv
# 修复HuggingFaceEmbeddings已弃用的警告，使用langchain_huggingface包
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from sentence_transformers import CrossEncoder
import torch
from torch.serialization import add_safe_globals
import numpy as np
import openai
# 导入火山引擎SDK
from volcenginesdkarkruntime import Ark
import re
import sys
import glob
import math
import time
import copy  # Add explicit import for copy module
from http.client import IncompleteRead
from urllib.error import HTTPError
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加numpy安全全局设置
add_safe_globals(['numpy.core.multiarray._reconstruct'])

script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(script_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from clinical_rag.config import load_config, resolve_path as _resolve_config_path
from clinical_rag.entrez import configure_entrez
from clinical_rag.paths import setup_paths
from clinical_rag.utils import make_json_serializable as _make_json_serializable
from clinical_rag.reranker import BGEReranker as CustomBGEReranker

project_root = setup_paths(_project_root)

from clinical_rag.domains.drugbank.matcher import DrugMatcher
from trial2vec import Trial2Vec
from advanced_disease_extractor import AdvancedDiseaseExtractor

class DeepSeekClient:
    """Client for DeepSeek API to extract keywords"""
    
    def __init__(self, api_key, default_model="deepseek-v3-250324"):
        self.api_key = api_key
        # 替换为火山引擎Ark客户端
        self.client = Ark(api_key=api_key)
        self.default_model = default_model
        self.invalid_terms = {
            "phase", "randomized", "randomised", "multicenter", "multicentre",
            "trial", "study", "comparing", "comparison", "versus", "maintenance",
            "therapy", "treatment", "patients", "patient", "continued", "stable disease"
        }

    def _normalize_term(self, term):
        if term is None:
            return ""
        return re.sub(r'\s+', ' ', str(term).replace('\n', ' ')).strip(" ,.;:()[]{}\"'")

    def is_invalid_extracted_term(self, term):
        normalized = self._normalize_term(term)
        if not normalized:
            return True
        lowered = re.sub(r'[^a-z0-9]+', ' ', normalized.lower()).strip()
        if not lowered:
            return True
        if lowered in self.invalid_terms:
            return True
        if re.fullmatch(r'(phase\s*)?(?:i|ii|iii|iv|v|vi|vii|viii|ix|x|\d+)', lowered):
            return True
        tokens = lowered.split()
        return bool(tokens) and all(token in self.invalid_terms for token in tokens)

    def _dedupe_terms(self, terms):
        deduped_terms = []
        seen = set()
        for term in terms:
            normalized = self._normalize_term(term)
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped_terms.append(normalized)
        return deduped_terms

    def _fallback_extract_keywords(self, title):
        normalized_title = self._normalize_term(title)
        disease = ""

        disease_patterns = [
            r'\bpatients with\s+(.+?)(?:\s+after\b|\s+following\b|\s+receiving\b|\s+treated\b|\s+versus\b|[.;]|$)',
            r'\b(?:for|in)\s+(.+?)(?:\s+after\b|\s+following\b|\s+receiving\b|\s+treated\b|\s+versus\b|[.;]|$)'
        ]
        for pattern in disease_patterns:
            match = re.search(pattern, normalized_title, re.IGNORECASE)
            if match:
                disease = self._normalize_term(match.group(1))
                disease = re.sub(r'\b(?:and|or)\s+having\b.*$', '', disease, flags=re.IGNORECASE)
                disease = re.sub(r'\bat least\b.*$', '', disease, flags=re.IGNORECASE)
                break

        if self.is_invalid_extracted_term(disease):
            disease_matches = re.findall(
                r'\b(?:metastatic|advanced|locally advanced|recurrent|relapsed|unresectable|blast phase chronic|estrogen receptor positive|her2-positive|her2 positive|bcr-abl1 positive)?(?:\s+[A-Za-z0-9-]+){0,4}\s+(?:cancer|carcinoma|leukemia|lymphoma|myeloma|melanoma|tumou?r|disease)\b',
                normalized_title,
                re.IGNORECASE
            )
            disease_candidates = [
                self._normalize_term(match)
                for match in disease_matches
                if self._normalize_term(match).lower() != "stable disease" and not self.is_invalid_extracted_term(match)
            ]
            disease = max(disease_candidates, key=len) if disease_candidates else ""

        drug_candidates = []
        plus_matches = re.findall(r'\b([A-Za-z][A-Za-z0-9-]+)\s*\+\s*([A-Za-z][A-Za-z0-9-]+)\b', normalized_title)
        for left_term, right_term in plus_matches:
            drug_candidates.extend([left_term, right_term])
        drug_candidates.extend(re.findall(r'\bsubstituting\s+([A-Za-z][A-Za-z0-9-]+)\b', normalized_title, re.IGNORECASE))

        drugs = [term for term in self._dedupe_terms(drug_candidates) if not self.is_invalid_extracted_term(term)]
        return disease, drugs, "", ""

    def _sanitize_keywords(self, disease, drug_list, other1, other2, title):
        fallback_disease, fallback_drugs, fallback_other1, fallback_other2 = self._fallback_extract_keywords(title)

        normalized_disease = self._normalize_term(disease)
        if self.is_invalid_extracted_term(normalized_disease):
            normalized_disease = fallback_disease

        normalized_drugs = [self._normalize_term(drug) for drug in drug_list or []]
        normalized_drugs = [drug for drug in self._dedupe_terms(normalized_drugs) if not self.is_invalid_extracted_term(drug)]
        if not normalized_drugs:
            normalized_drugs = fallback_drugs

        normalized_other1 = self._normalize_term(other1)
        normalized_other2 = self._normalize_term(other2)
        if self.is_invalid_extracted_term(normalized_other1):
            normalized_other1 = fallback_other1
        if self.is_invalid_extracted_term(normalized_other2):
            normalized_other2 = fallback_other2

        if self.is_invalid_extracted_term(normalized_other1):
            normalized_other1 = ""
        if self.is_invalid_extracted_term(normalized_other2):
            normalized_other2 = ""

        return normalized_disease, normalized_drugs, normalized_other1, normalized_other2
    
    def extract_keywords(self, title):
        """Extract disease, drug, and two other medical keywords from the title"""
        prompt = f"""
        From the medical research title, extract key terms with these strict rules:
        1.  **Disease**: Extract EXACTLY ONE disease or condition.
        2.  **Drugs**: Extract ONE OR MORE drug/treatment names.
            *   Return ONLY the generic/active ingredient name(s).
            *   IGNORE dosages (e.g., '5mg'), brand symbols (e.g., '®', '™'), and other modifiers.
            *   If a term combines multiple drugs (e.g., "DrugA / DrugB"), SPLIT them into separate items in the array.
            *   Don't replicate drug names.
        3.  **Other Terms**: Extract TWO other key medical terms (not methodological terms such as "Comparative Study", "Double-blinded", "Randomized Controlled", and so on).
        
        The output MUST be a JSON object with four fields: "disease" (string), "drugs" (array of strings), "other1" (string), and "other2" (string).
        The "drugs" field MUST be an array, even with only one drug.
        
        Title: "{title}"
        """
        
        try:
            print(f"--- [INFO] Calling API to extract keywords from title: {title} ---")
            response = self.client.chat.completions.create(
                model=self.default_model,
                messages=[
                    {"role": "system", "content": "You are a medical research assistant that extracts precise keywords from titles."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            print(f"--- [INFO] API response received successfully ---")
            result = response.choices[0].message.content
            print(f"--- [INFO] Keywords extracted by API: {result} ---")
            
            # Parse JSON response
            try:
                # 处理可能返回的Markdown代码块
                json_str = result
                
                # 移除Markdown代码块标记（如果存在）
                markdown_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
                markdown_match = re.search(markdown_pattern, json_str)
                if markdown_match:
                    print("--- [INFO] Detected Markdown code block in response, extracting JSON content ---")
                    json_str = markdown_match.group(1)
                
                # 解析JSON
                keywords = json.loads(json_str)
                print(f"--- [INFO] JSON parsing successful, found keys: {list(keywords.keys())} ---")
                
                # Ensure 'drugs' is always a list
                drug_list = keywords.get("drugs", [])
                if not isinstance(drug_list, list):
                    drug_list = [str(drug_list)] # Coerce to list if model returns a single string
                
                # 确保所有必要的键都存在，使用安全的get方法，提供默认值
                disease = keywords.get("disease", "unknown disease")
                other1 = keywords.get("other1", "term1")
                other2 = keywords.get("other2", "term2")
                
                print(f"--- [INFO] Extracted values: disease='{disease}', drugs={drug_list}, other1='{other1}', other2='{other2}' ---")
                return self._sanitize_keywords(disease, drug_list, other1, other2, title)
                
            except json.JSONDecodeError as json_err:
                print(f"--- [ERROR] Failed to parse JSON response: {json_err} ---")
                print(f"--- [DEBUG] Raw response: {result} ---")
                return self._fallback_extract_keywords(title)
            
        except Exception as e:
            # 统一处理所有可能的API错误
            print(f"--- [ERROR] API error in extract_keywords: {e} ---")
            
            # 处理认证错误
            if "Authentication" in str(e) or "Auth" in str(e) or "auth" in str(e).lower() or "key" in str(e).lower():
                print(f"--- [FATAL] API Authentication Error: {e} ---")
                print("--- [FATAL] Please check if the 'deepseek_api_key' in config.json is correct. ---")
                return self._fallback_extract_keywords(title)
            
            # 对于任何其他错误，执行标题分词的后备方案
            print(f"--- [ERROR] Failed to extract keywords with API: {e}. Using fallback method. ---")
            return self._fallback_extract_keywords(title)

class EnhancedMedicalSearch:
    def __init__(self, config):
        """Initialize with configuration"""
        self.config = config
        
        configure_entrez(config)
        os.environ["ARK_API_KEY"] = config["api_keys"]["deepseek_api_key"]
        
        # FAISS配置
        self.faiss_config = config.get("faiss", {
            "index_path": "faiss_indexes",
            "distance_metric": "cosine"
        })
        
        # 确保FAISS索引目录存在
        os.makedirs(self.faiss_config["index_path"], exist_ok=True)
        
        # Initialize embeddings and reranker
        try:
            # 使用新版API初始化HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config["embedding_model"],
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
            )
            print(f"--- [INFO] HuggingFace Embeddings loaded successfully with model: {config['embedding_model']} ---")
        except Exception as e:
            print(f"--- [ERROR] Failed to initialize HuggingFaceEmbeddings: {e} ---")
            raise  # 这是一个关键组件，如果初始化失败则无法继续
            
        self.reranker = CustomBGEReranker(model_name=config["reranker_model"])
        models_config = config.get("models", {})
        self.general_model = models_config.get("general_model", "deepseek-v3-250324")
        self.reasoning_model = models_config.get("reasoning_model", "deepseek-r1-250528")
        self.deepseek_client = DeepSeekClient(api_key=config["api_keys"]["deepseek_api_key"], default_model=self.general_model)
        
        # 其余初始化代码保持不变
        # Initialize Disease Extractor
        try:
            disease_csv_path = config.get("disease_data", {}).get("csv_path", "data/primekg/disease_features.csv")
            self.disease_extractor = AdvancedDiseaseExtractor(
                data_path=self._resolve_path(disease_csv_path),
                api_key=config["api_keys"]["deepseek_api_key"]
            )
            print(f"--- [INFO] AdvancedDiseaseExtractor initialized with data from {disease_csv_path} ---")
        except Exception as e:
            print(f"--- [ERROR] Failed to initialize AdvancedDiseaseExtractor: {e} ---")
            self.disease_extractor = None
        
        # Initialize DrugMatcher
        self.drug_matcher = None
        try:
            drug_csv_path = self._resolve_path(
                config.get("drug_data", {}).get("csv_path", "data/drugbank/drugbank_data_v1.csv")
            )
            use_synonyms = config.get("drug_data", {}).get("use_synonyms", False)
            self.drug_matcher = DrugMatcher(drug_csv_path, preprocess=True, use_synonyms=use_synonyms)
            print(f"--- [INFO] DrugMatcher initialized with data from {drug_csv_path} ---")
        except Exception as e:
            print(f"--- [ERROR] Failed to initialize DrugMatcher: {e} ---")
        
        # Initialize Trial2Vec and related data for clinical trial retrieval
        try:
            trial_cfg = self.config.get("trial_retrieval", {})
            
            # Determine which phases to load
            all_phase_keys = trial_cfg.get("phases", {}).keys()
            active_phases = trial_cfg.get("active_phases")
            if not active_phases: # If null or empty list, use all available phases
                active_phases = list(all_phase_keys)

            print(f"--- [INFO] Loading trial data for active phases: {active_phases} ---")

            self.trial_data_by_phase = {}
            
            for phase_key in active_phases:
                phase_config = trial_cfg.get("phases", {}).get(phase_key)
                if not phase_config:
                    print(f"--- [WARN] Configuration for phase '{phase_key}' not found. Skipping. ---")
                    continue

                emb_path = self._resolve_path(phase_config.get("emb_path"))
                data_path = self._resolve_path(phase_config.get("data_path"))

                if not emb_path or not os.path.exists(emb_path):
                    print(f"--- [WARN] Embedding path for {phase_key} not found or invalid: {emb_path}. Skipping. ---")
                    continue
                if not data_path or not os.path.exists(data_path):
                    print(f"--- [WARN] Data path for {phase_key} not found or invalid: {data_path}. Skipping. ---")
                    continue

                print(f"--- [INFO] Loading trial embeddings for {phase_key} from {emb_path} ---")
                # 重新添加weights_only=False参数
                phase_emb_dict = torch.load(emb_path, weights_only=False)["emb"]

                print(f"--- [INFO] Loading trial metadata for {phase_key} from {data_path} ---")
                phase_df = pd.read_csv(data_path)
                if "nct_id" in phase_df.columns:
                    phase_df = phase_df.set_index("nct_id")

                self.trial_data_by_phase[phase_key] = {
                    "emb_dict": phase_emb_dict,
                    "raw_df": phase_df,
                    "all_ids": list(phase_emb_dict.keys()),
                    "all_embs": list(phase_emb_dict.values())
                }
                print(f"--- [SUCCESS] Successfully loaded data for trial phase '{phase_key}' ---")

            if not self.trial_data_by_phase:
                 print("--- [WARN] No trial data was loaded for any phase. ---")

            # Load the model itself (common for all phases)
            model_dir = self._resolve_path(trial_cfg.get("model_dir"))
            bert_name = self._resolve_path(trial_cfg.get("bert_name"))
            
            # 标准化路径，确保路径分隔符统一
            model_dir = os.path.normpath(model_dir)
            bert_name = os.path.normpath(bert_name)

            print(f"--- [INFO] Loading Trial2Vec model from {model_dir} ---")
            print(f"--- [INFO] Using BERT model from {bert_name} ---")
            
            # 将bert_name作为字符串传递给Trial2Vec
            self.trial_model = Trial2Vec(bert_name=bert_name)
            self.trial_model.from_pretrained(model_dir)
            print(f"--- [INFO] Trial2Vec model loaded on device: {self.trial_model.device} ---")

            cfg_path = os.path.join(model_dir, "model_config.json")
            with open(cfg_path, "r", encoding="utf-8") as f:
                mcfg = json.load(f)
            self.trial_fields = mcfg.get("fields", ["title", "intervention_name", "disease", "keyword"])
            
            # Update model config to use only fields, no ctx_fields
            self.trial_model.config['ctx_fields'] = []
            self.trial_model.model.config['ctx_fields'] = []
            self.trial_ctx_fields = []

            print(f"--- [INFO] ctx_fields have been removed for encoding. Using fields: {self.trial_fields} ---")

            print("--- [SUCCESS] Trial2Vec components initialized ---")
        except Exception as e:
            print(f"--- [ERROR] Failed to initialize trial retrieval components: {e} ---")
            self.trial_model = None
     
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
            # Note: This relies on the prompt file being in the expected location relative to the project root.
            with open(self._resolve_path("data/primekg/disease_analysis_prompt.txt"), "r", encoding="utf-8") as f:
                prompt_template = f.read()
            
            # 如果有title，将disease_terms和title一起作为上下文
            if title:
                context_input = f"{disease_terms_str} (from research: {title})"
            else:
                context_input = disease_terms_str
            
            # 格式化提示模板
            prompt = prompt_template.format(input=context_input)
            
            # Using the main client for this call
            response = self.deepseek_client.client.chat.completions.create(
                model=self.general_model,
                messages=[
                    {"role": "system", "content": "You are a professional medical expert analyzing disease terms."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=500
            )
            result = response.choices[0].message.content.strip()
            
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
            
            analysis_text = f"Standardized Term: {best_term}\nDefinition: {analysis_data.get('definition', 'N/A')}\nKey Characteristics: {', '.join(analysis_data.get('key_characteristics', []))}"
            
            return best_term, analysis_text

        except Exception as e:
            print(f"--- [ERROR] Failed during AI disease analysis: {e} ---")
            # 添加更详细的错误信息，包括完整的堆栈跟踪
            import traceback
            print(f"--- [ERROR] Traceback: {traceback.format_exc()} ---")
            # Fallback to the original input if analysis fails
            return disease_terms_str, f"AI analysis failed: {e}"

    def extract_keywords_from_title(self, title):
        """Extract disease, drug, and two other medical keywords from research title using DeepSeek API"""
        return self.deepseek_client.extract_keywords(title)
    
    def match_drug(self, drug_name):
        """Match drug name using DrugMatcher"""
        if not self.drug_matcher or not drug_name:
            return None
            
        print(f"\n--- [INFO] Matching drug name: '{drug_name}' ---")
        try:
            result = self.drug_matcher.search(drug_name)
            if result["success"]:
                print(f"--- [SUCCESS] Matched drug: '{drug_name}' ---")
                # Convert row DataFrame Series to dict
                if "result" in result and "row" in result["result"]:
                    result["result"]["row"] = dict(result["result"]["row"])
                return result
            else:
                print(f"--- [WARN] Could not match drug name: '{drug_name}'. {result.get('message', '')} ---")
                return None
        except Exception as e:
            print(f"--- [ERROR] An error occurred during drug matching: {e} ---")
            return None

    def format_drug_info(self, drug_match):
        """Format drug information for readable output, skipping fields with no data."""
        if not drug_match or not drug_match.get("success") or "result" not in drug_match:
            return "No drug match found"
            
        drug_data = drug_match["result"]["row"]
        
        # Helper to clean up text and check for meaningful content
        def get_clean_value(text):
            if text is None:
                return None
            if not isinstance(text, str):
                text = str(text)
            
            # Specifically check for pandas/numpy NaN representations which become "nan" strings
            if text.strip().lower() in ['', 'nan', 'none']:
                return None
            
            # Replace newlines with spaces and squeeze multiple spaces into one
            return re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()

        # Define the fields to display, mapping data key to display name
        fields_to_display = [
            ('name', 'Name'),
            ('SMILES', 'SMILES'),
            ('molecular_formula', 'Molecular Formula'),
            ('indication', 'Indication'),
            ('mechanism-of-action', 'Mechanism of Action'),
            ('pharmacodynamics', 'Pharmacodynamics'),
            ('absorption', 'Absorption'),
            ('metabolism', 'Metabolism'),
            ('half-life', 'Half-life'),
            ('protein-binding', 'Protein Binding'),
            ('route-of-elimination', 'Route of Elimination'),
            ('volume-of-distribution', 'Volume of Distribution'),
            ('clearance', 'Clearance'),
            ('toxicity', 'Toxicity')
        ]

        output = []
        for key, display_name in fields_to_display:
            value = get_clean_value(drug_data.get(key))
            if value:
                output.append(f"* {display_name}: {value}")
        
        if not output:
            return "No drug match found"
            
        return "\n".join(output)
         
    def search_pubmed(self, disease_term, drug_terms, until_year, other_terms=None):
        """
        Search PubMed with structured logic: (Disease AND (Drug1 OR Drug2 ...)) OR (Other1 AND Other2 ...).
        The 'other_terms' search is only performed if the primary search yields no results.
        If both searches yield no results, a fallback search using only the disease term is performed
        when the perform_fallback_search config option is enabled.
        """
        # --- Build the Primary Search Term (Disease AND Drugs) ---
        primary_query = ""
        if disease_term:
            # Format the drug part of the query: (DrugA OR DrugB)
            drug_query_part = ""
            if drug_terms:
                or_drugs = " OR ".join([f"({term})" for term in drug_terms if term])
                if len(drug_terms) > 1:
                    drug_query_part = f"({or_drugs})"
                else:
                    drug_query_part = or_drugs

            # Combine disease and drugs with AND
            if drug_query_part:
                primary_query = f"({disease_term}) AND {drug_query_part}"
            else:
                primary_query = f"({disease_term})"
        
        # --- Set up search parameters ---
        search_until_year = until_year - 1
        date_range = f"1900[Date - Publication] : {search_until_year}[Date - Publication]"
        
        # Function to execute a search query and return papers
        def _execute_search(query):
            if not query:
                return []
            
            print(f"\n--- [START] Searching PubMed for '{query}' (until end of {search_until_year}) ---")
        
            # Initialize search parameters from the consolidated pubmed_search config
            pubmed_config = self.config.get("pubmed_search", {})
            retmax = min(pubmed_config.get("max_results", 500), 9990)
            batch_size = pubmed_config.get("batch_size", 1000)
            retstart = 0
            pmid_list = []
            
            # Get total count of papers
            try:
                with Entrez.esearch(db="pubmed", term=f"{query} AND {date_range}", retmax=0) as handle:
                    record = Entrez.read(handle)
                    total_count = int(record["Count"])
                
                if total_count == 0:
                    print(f"--- [INFO] No papers found on PubMed for '{query}' ---")
                    return []
                    
                total_to_fetch = min(total_count, retmax)
                print(f"--- [INFO] Found {total_count} papers, fetching details for up to {total_to_fetch} ---")
                
                # Fetch PMIDs in batches
                with tqdm(total=total_to_fetch, desc="Fetching PMIDs") as pbar:
                    while retstart < total_to_fetch:
                        current_retmax = min(batch_size, total_to_fetch - retstart)
                        with Entrez.esearch(db="pubmed", term=f"{query} AND {date_range}", 
                                      retmax=current_retmax, retstart=retstart) as handle:
                            record = Entrez.read(handle)
                            batch_pmids = record["IdList"]
                            
                            if not batch_pmids:
                                break
                                
                            pmid_list.extend(batch_pmids)
                            pbar.update(len(batch_pmids))
                            retstart += len(batch_pmids)
                            
                            if len(pmid_list) >= total_to_fetch:
                                break
                                
                print(f"--- [INFO] Retrieved {len(pmid_list)} unique PMIDs ---")
                pmid_list = list(set(pmid_list)) # Deduplicate
                print(f"--- [INFO] Found {len(pmid_list)} unique PMIDs after deduplication ---")
                
            except Exception as e:
                print(f"--- [ERROR] Failed to search PubMed: {e} ---")
                return []
            
            # Fetch paper details in parallel
            return self._fetch_paper_details(pmid_list, pubmed_config)

        # Get pubmed search config
        pubmed_config = self.config.get("pubmed_search", {})
        perform_secondary_search = pubmed_config.get("perform_secondary_search", True)
        perform_fallback_search = pubmed_config.get("perform_fallback_search", True)
        
        # --- Execute Searches ---
        # 1. Try the primary query first
        papers = _execute_search(primary_query)
        
        # 2. If primary search fails and other_terms are available, try the secondary search
        if not papers and other_terms and perform_secondary_search:
            print("\n--- [INFO] Primary search returned no results. Trying secondary search with other keywords. ---")
            secondary_query = " AND ".join([f"({term})" for term in other_terms if term])
            papers = _execute_search(secondary_query)
        
        # 3. If both searches fail and we have a disease term, try searching only with the disease term
        if not papers and disease_term and perform_fallback_search:
            print("\n--- [INFO] Previous searches returned no results. Trying fallback search with only disease term. ---")
            fallback_query = f"({disease_term})"
            papers = _execute_search(fallback_query)
        
        return papers
        
    def _fetch_paper_details(self, pmid_list, pubmed_config):
        """Helper to fetch full paper details for a list of PMIDs."""
        if not pmid_list:
            return []
            
        papers = []
        total_filtered = 0  # 跟踪过滤的总数量
        DETAILS_BATCH_SIZE = pubmed_config.get("efetch_batch_size", 50)
        MAX_WORKERS = pubmed_config.get("max_workers", 5)
        batches = [pmid_list[i:i + DETAILS_BATCH_SIZE] for i in range(0, len(pmid_list), DETAILS_BATCH_SIZE)]

        def _fetch_batch_details(batch_pmids):
            """Fetches and parses details for a single batch of PMIDs with retry logic."""
            for attempt in range(3):
                try:
                    with Entrez.epost(db="pubmed", id=",".join(batch_pmids)) as handle:
                        result = Entrez.read(handle)
                        webenv = result["WebEnv"]
                        query_key = result["QueryKey"]
                    
                    with Entrez.efetch(db="pubmed", retmode="xml", webenv=webenv, query_key=query_key) as handle:
                        records = Entrez.read(handle)
                        
                        batch_papers = []
                        filtered_count = 0
                        
                        for article in records["PubmedArticle"]:
                            try:
                                pmid = article["MedlineCitation"]["PMID"]
                                title = article["MedlineCitation"]["Article"]["ArticleTitle"]
                                
                                abstract_text = article["MedlineCitation"]["Article"].get("Abstract", {}).get("AbstractText", ["No abstract available"])
                                if isinstance(abstract_text, list):
                                    abstract = " ".join([str(text) for text in abstract_text])
                                else:
                                    abstract = str(abstract_text)
                                
                                # 检查摘要是否为空或仅包含默认文本
                                abstract_lowercase = abstract.lower().strip()
                                if (not abstract_lowercase or 
                                    abstract_lowercase == "no abstract available" or
                                    abstract_lowercase == "no abstract" or
                                    abstract_lowercase == "not available"):
                                    filtered_count += 1
                                    continue  # 跳过没有摘要的文章
                                
                                journal = article["MedlineCitation"]["Article"]["Journal"]["Title"]
                                
                                try:
                                    pub_year = int(article["MedlineCitation"]["Article"]["Journal"]["JournalIssue"]["PubDate"].get("Year", 0))
                                except:
                                    pub_year = 0
                                
                                batch_papers.append({
                                    "PMID": pmid,
                                    "Title": title,
                                    "Abstract": abstract,
                                    "Journal": journal,
                                    "Year": pub_year
                                })
                            except Exception as e:
                                print(f"--- [ERROR] Failed to process article details: {e} ---")
                                continue
                        
                        if filtered_count > 0:
                            print(f"--- [INFO] Filtered out {filtered_count} papers with no valid abstract ---")
                            
                        return batch_papers, filtered_count  # 返回批次处理的论文和过滤数量
                
                except (HTTPError, IncompleteRead) as e:
                    print(f"--- [WARN] Network error on batch, attempt {attempt+1}/3: {e}. Retrying in 5 seconds... ---")
                    time.sleep(5)
                except Exception as e:
                    print(f"--- [ERROR] An unexpected error occurred on batch: {e} ---")
                    break 
            
            print(f"--- [ERROR] Failed to process batch after 3 attempts. Skipping. ---")
            return [], 0  # 错误情况下返回空列表和0过滤数

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all batches to the executor
            future_to_batch = {executor.submit(_fetch_batch_details, batch): batch for batch in batches}
            
            # Process results as they are completed, with a progress bar
            for future in tqdm(as_completed(future_to_batch), total=len(batches), desc="Processing papers"):
                batch_papers, batch_filtered = future.result()
                total_filtered += batch_filtered  # 累计过滤数量
                if batch_papers:
                    papers.extend(batch_papers)
                
        print(f"--- [SUCCESS] Successfully processed {len(papers)} papers from PubMed ---")
        if total_filtered > 0:
            print(f"--- [INFO] Total papers filtered due to missing abstracts: {total_filtered} ({len(papers)} valid papers retained) ---")
        
        return papers
    
    def create_vector_store(self, papers, collection_name):
        """Create vector store from papers using FAISS"""
        print(f"\n--- [START] Creating vector store '{collection_name}' ---")
        
        if not papers:
            print("--- [WARN] No papers provided to create vector store. ---")
            return None
            
        # Convert papers to documents
        documents = []
        for paper in papers:
            content = f"Title: {paper['Title']}\nAbstract: {paper['Abstract']}"
            metadata = {
                "pmid": paper['PMID'],
                "title": paper['Title']
            }
            documents.append(Document(page_content=content, metadata=metadata))
        
        try:
            # 构建索引保存路径
            index_path = os.path.join(self.faiss_config["index_path"], collection_name)
            
            # 确保索引目录存在
            os.makedirs(self.faiss_config["index_path"], exist_ok=True)
            
            # 检查FAISS是否可用
            try:
                import faiss
                print(f"--- [INFO] FAISS version: {faiss.__version__} ---")
            except ImportError:
                print("--- [ERROR] FAISS is not installed. Please install it with 'pip install faiss-cpu' or 'pip install faiss-gpu' ---")
                raise
            
            # Create vector store
            print(f"--- [INFO] Creating FAISS index with {len(documents)} documents ---")
            vector_db = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            # Save the FAISS index locally
            print(f"--- [INFO] Saving FAISS index to {index_path} ---")
            # 注意：save_local不需要安全参数，安全问题只发生在加载时
            vector_db.save_local(index_path)
            print(f"--- [SUCCESS] Successfully created vector store '{collection_name}' at {index_path} ---")
            return vector_db
            
        except Exception as e:
            print(f"--- [ERROR] Failed to create vector store: {str(e)} ---")
            print(f"--- [DEBUG] Error type: {type(e).__name__}, Details: {e} ---")
            
            # 检查是否是FAISS相关错误
            if "faiss" in str(e).lower():
                print("--- [HINT] This appears to be a FAISS-related error. Please ensure FAISS is properly installed. ---")
                print("--- [HINT] For CPU: pip install faiss-cpu ---")
                print("--- [HINT] For GPU: pip install faiss-gpu ---")
            
            return None
    
    def search_similar_papers(self, query, collection_name, top_k=5, similarity_threshold=0.5):
        """
        Search for papers similar to the query using FAISS
        
        Args:
            query (str): The search query text
            collection_name (str): Name of the FAISS collection to search
            top_k (int): Maximum number of results to return
            similarity_threshold (float): Minimum similarity score (0-1) for results
        """
        print(f"\n--- [START] Searching for similar papers to '{query}' in collection '{collection_name}' ---")

        try:
            search_k = min(30, top_k * 3)  # Get more than needed for reranking
            
            # 调试信息
            print(f"--- [INFO] FAISS search parameters: k={search_k}, similarity_threshold={similarity_threshold} ---")
            
            # 构建索引路径
            index_path = os.path.join(self.faiss_config["index_path"], collection_name)
            print(f"--- [INFO] Loading FAISS index from {index_path} ---")
            
            # 检查索引路径是否存在
            if not os.path.exists(index_path):
                print(f"--- [ERROR] FAISS index path does not exist: {index_path} ---")
                return []
            
            # 加载FAISS索引
            try:
                vector_db = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
                print(f"--- [SUCCESS] Successfully loaded FAISS index from {index_path} ---")
            except Exception as e:
                print(f"--- [ERROR] Failed to load FAISS index: {str(e)} ---")
                print(f"--- [DEBUG] Error type: {type(e).__name__}, Details: {e} ---")
                
                # 检查是否是反序列化安全问题
                if "deserialization" in str(e).lower():
                    print("--- [HINT] This is a deserialization safety issue. Adding allow_dangerous_deserialization=True to fix it. ---")
                    try:
                        vector_db = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
                        print(f"--- [SUCCESS] Successfully loaded FAISS index with deserialization enabled ---")
                    except Exception as inner_e:
                        print(f"--- [ERROR] Still failed to load FAISS index: {str(inner_e)} ---")
                        return []
                else:
                    return []
            
            # 执行搜索
            print(f"--- [INFO] Performing similarity search with query: '{query[:50]}...' ---")
            try:
                results = vector_db.similarity_search_with_score(query, k=search_k)
                print(f"--- [SUCCESS] Found {len(results)} results from similarity search ---")
            except Exception as e:
                print(f"--- [ERROR] Failed during similarity search: {str(e)} ---")
                print(f"--- [DEBUG] Error type: {type(e).__name__}, Details: {e} ---")
                return []
            
            # 处理结果
            docs = []
            for doc, score in results:
                # FAISS的score是距离，越小越好，我们转换为相似度(1-distance)
                similarity = 1.0 - min(1.0, float(score))
                
                # 添加相似度分数到metadata
                doc.metadata["distance"] = float(score)
                doc.metadata["similarity"] = float(similarity)
                docs.append(doc)
            
            if not docs:
                print("--- [INFO] No matching documents found in vector store. ---")
                return []

            print(f"--- [INFO] Found {len(docs)} initial matches. Reranking top results... ---")

            # Rerank results
            reranked_docs = self.reranker.rerank(query, docs, top_k=top_k)
            print(f"--- [SUCCESS] Reranked to {len(reranked_docs)} top results ---")


            # 应用相似度阈值筛选
            pre_filter_count = len(reranked_docs)
            filtered_docs = [doc for doc in reranked_docs if doc.metadata.get("bge_score", doc.metadata.get("similarity", 0)) > similarity_threshold]
            post_filter_count = len(filtered_docs)
            
            if post_filter_count < pre_filter_count:
                print(f"--- [INFO] Filtered out {pre_filter_count - post_filter_count} papers with similarity score below {similarity_threshold} ---")
                print(f"--- [INFO] {post_filter_count} papers remaining after similarity filtering ---")
                
            if not filtered_docs:
                print(f"--- [WARN] All papers were filtered out due to similarity threshold of {similarity_threshold}. Consider lowering the threshold. ---")
                return []
                
            # Display results in full - only title and abstract
            print(f"\n--- [SUCCESS] Top {len(filtered_docs)} Paper Results ---")
            for i, doc in enumerate(filtered_docs):
                score = doc.metadata.get("bge_score", doc.metadata.get("similarity", 0.0))
                pmid = doc.metadata.get("pmid", "Unknown")

                print(f"\n--- Result [{i+1}] | Score: {score:.4f}, PMID: {pmid} ---")

                # Parse content to separate title and abstract
                content = doc.page_content
                title_match = re.search(r"Title: (.*?)(?:\nAbstract: |$)", content)
                abstract_match = re.search(r"Abstract: (.*?)$", content)

                title = title_match.group(1) if title_match else doc.metadata.get('title', 'Unknown')
                abstract = abstract_match.group(1) if abstract_match else "Abstract not available"

                # Print full title and abstract
                print(f"Title: {title}")
                print(f"Abstract: {abstract}")
                print("-" * 80)

            return filtered_docs

        except Exception as e:
            print(f"--- [ERROR] Failed during similar paper search: {str(e)} ---")
            print(f"--- [DEBUG] Error type: {type(e).__name__}, Details: {e} ---")
            return []
     
    def process_research_title(self, title, until_year=None):
        """Process research title to find relevant papers"""
        print(f"\n{'='*20} [START] Processing New Research Title: '{title}' {'='*20}")
        
        # 检查标题是否以句号结尾，如果没有则添加
        if not title.strip().endswith("."):
            original_title = title
            title = title.strip() + "."
        
        try:
            # 1. Extract keywords using API
            print("\n--- [START] Step 1: Extracting Keywords via API ---")
            disease_keyword, drug_keywords, other1_keyword, other2_keyword = self.extract_keywords_from_title(title)
            
            # Log the extracted keywords
            drug_keywords_str = ', '.join(f"'{k}'" for k in drug_keywords)
            print(f"--- [SUCCESS] Extracted Keywords -> Disease: '{disease_keyword}', Drugs: [{drug_keywords_str}], Other1: '{other1_keyword}', Other2: '{other2_keyword}' ---")

            yield {
                "step": 1, 
                "name": "Keyword Extraction", 
                "data": {
                    "disease": disease_keyword, "drugs": drug_keywords, 
                    "other1": other1_keyword, "other2": other2_keyword
                },
                "next_step_name": "RAG Context Expansion"
            }
        except Exception as e:
            print(f"--- [ERROR] Error in Step 1 (Keyword Extraction): {e} ---")
            # 提供默认值以继续流程
            disease_keyword = ""
            drug_keywords = []
            other1_keyword = ""
            other2_keyword = ""
            
            yield {
                "step": 1, 
                "name": "Keyword Extraction", 
                "data": {
                    "disease": disease_keyword, "drugs": drug_keywords, 
                    "other1": other1_keyword, "other2": other2_keyword
                },
                "next_step_name": "RAG Context Expansion"
            }
        
        try:
            # 2. RAG Context Expansion
            print("\n--- [START] Step 2: RAG Context Expansion ---")
            temp_description, temp_criteria, primary_outcome = self.generate_context_for_retrieval(title)
            print("--- [SUCCESS] Generated context for retrieval ---")
            print(f"    - Description: {temp_description}")
            print(f"    - Primary Outcome: {primary_outcome}")
            yield {
                "step": 2,
                "name": "RAG Context Expansion",
                "data": {
                    "description": temp_description,
                    "primary_outcome": primary_outcome
                },
                "next_step_name": f"Retrieving Information for Disease: '{disease_keyword}'"
            }
        except Exception as e:
            print(f"--- [ERROR] Error in Step 2 (RAG Context Expansion): {e} ---")
            # 提供默认值以继续流程
            temp_description = ""
            temp_criteria = ""
            primary_outcome = ""
            
            yield {
                "step": 2,
                "name": "RAG Context Expansion",
                "data": {
                    "description": temp_description,
                    "primary_outcome": primary_outcome
                },
                "next_step_name": f"Retrieving Information for Disease: '{disease_keyword}'"
            }

        # 3. Get advanced disease information
        disease_info_text = "No detailed disease information found."
        disease_info_data_for_prompt = None
        term_for_retrieval = disease_keyword 

        if disease_keyword and self.disease_extractor:
            print(f"\n--- [START] Step 3: Retrieving Advanced Information for Disease: '{disease_keyword}' ---")
            try:
                best_term, _ = self._analyze_disease_terms(disease_keyword, title)
                
                if best_term:
                    term_for_retrieval = best_term
                    print(f"--- [INFO] Retrieving structured info for standardized term: '{term_for_retrieval}' ---")
                    disease_info = self.disease_extractor.get_info(term_for_retrieval)
                    disease_info_data_for_prompt = disease_info
                    
                    if disease_info and disease_info.get('best_match_name'):
                        print(f"--- [SUCCESS] Found detailed information for '{disease_info['best_match_name']}' ---")
                        # Pass the structured dictionary directly to the frontend
                        disease_info_text = disease_info
                    else:
                        disease_info_text = "No structured details found in the database."
            except Exception as e:
                print(f"--- [ERROR] Error while retrieving disease information: {e} ---")
                disease_info_text = f"An error occurred during disease retrieval: {e}"

        yield {
            "step": 3,
            "name": f"Retrieving Information for Disease: '{disease_keyword}'",
            "data": _make_json_serializable(disease_info_text),
            "next_step_name": "Matching Drug Names"
        }

        # --- Step 4 onwards: Dynamic Step Counting ---
        step_counter = 4
        all_drug_info = []
        
        print(f"\n--- [START] Step {step_counter}: Matching Drug Names: {drug_keywords} ---")

        valid_drug_keywords = [
            keyword for keyword in drug_keywords
            if keyword and "failed" not in keyword and not self.deepseek_client.is_invalid_extracted_term(keyword)
        ]
        print(f"--- [DEBUG] Valid drug keywords for matching: {valid_drug_keywords} ---")
        
        valid_drug_matches = []
        for drug_keyword in valid_drug_keywords:
            print(f"--- [INFO] Matching drug: '{drug_keyword}' ---")
            drug_match_data = self.match_drug(drug_keyword)
            formatted_drug_info = self.format_drug_info(drug_match_data)
            
            if formatted_drug_info and formatted_drug_info.count('*') > 1:
                print(f"--- [DEBUG] Found valid match for '{drug_keyword}' with {formatted_drug_info.count('*')} features ---")
                valid_drug_matches.append((drug_keyword, formatted_drug_info))
            else:
                print(f"--- [DEBUG] No valid match for '{drug_keyword}' (insufficient features) ---")
        
        print(f"--- [DEBUG] Total valid drug matches: {len(valid_drug_matches)} ---")
        
        for i, (drug_keyword, formatted_drug_info) in enumerate(valid_drug_matches):
            all_drug_info.append(formatted_drug_info)
            
            is_last_valid_drug = (i == len(valid_drug_matches) - 1)
            next_step_name = "Searching for Relevant Papers" if is_last_valid_drug else f"Matching Drug: {valid_drug_matches[i+1][0]}"
            print(f"--- [DEBUG] Step for drug '{drug_keyword}' (#{i+1}/{len(valid_drug_matches)}), next step: '{next_step_name}' ---")

            yield _make_json_serializable({
                "step": step_counter, 
                "name": f"Drug Information: {drug_keyword}", 
                "data": formatted_drug_info,
                "next_step_name": next_step_name
            })
            step_counter += 1

        if not all_drug_info:
            no_drug_message = "No matching drug found in the database for the given keywords."
            print(f"--- [INFO] {no_drug_message} ---")
            yield _make_json_serializable({
                "step": step_counter,
                "name": "Matching Drug Information",
                "data": no_drug_message,
                "next_step_name": "Searching for Relevant Papers"
            })
            step_counter += 1
        
        pubmed_step = step_counter
        print(f"\n--- [START] Step {pubmed_step}: Perform PubMed Searches ---")
        
        final_until_year = until_year if until_year is not None else datetime.datetime.now().year + 1

        all_papers = self.search_pubmed(
            disease_term=term_for_retrieval,
            drug_terms=valid_drug_keywords,
            until_year=final_until_year,
            other_terms=[other1_keyword, other2_keyword] if self.config.get("pubmed_search", {}).get("perform_secondary_search", True) else None
        )
        
        if not all_papers:
            print("--- [INFO] No papers found, skipping vector search. ---")
            yield _make_json_serializable({
                "step": pubmed_step,
                "name": "Searching for Relevant Papers",
                "data": "No relevant papers found.",
                "next_step_name": "Retrieving Similar Clinical Trials"
            })
            all_papers = []
        
        if all_papers:
            collection_prefix = re.sub(r'[^a-zA-Z0-9_]', '_', title[:20])
            collection_prefix = re.sub(r'_{2,}', '_', collection_prefix)
            if collection_prefix and collection_prefix[0].isdigit():
                collection_prefix = "c_" + collection_prefix
            timestamp = int(time.time())
            collection_name = f"{collection_prefix}_papers_{timestamp}"
            
            print(f"--- [INFO] Creating FAISS vector store for {len(all_papers)} papers ---")
            vector_db = self.create_vector_store(all_papers, collection_name)
            if vector_db:
                print("\n--- [START] Step 5b: Performing Semantic Search on Papers ---")
                search_query = title if not temp_description else f"{title} {temp_description}"
                paper_similarity_threshold = self.config.get("pubmed_search", {}).get("paper_similarity_threshold", 0.5)
                result_docs = self.search_similar_papers(search_query, collection_name, top_k=5, similarity_threshold=paper_similarity_threshold)
                
                try:
                    index_path = os.path.join(self.faiss_config["index_path"], collection_name)
                    if os.path.exists(index_path):
                        import shutil
                        shutil.rmtree(index_path)
                        print(f"--- [INFO] Cleaned up temporary FAISS index at {index_path} ---")
                except Exception as e:
                    print(f"--- [WARN] Failed to clean up temporary FAISS index: {e} ---")
            else:
                print("--- [WARN] Failed to create vector store, skipping semantic search. ---")
                result_docs = []
            
            papers_for_results = [
                {
                    "title": re.search(r"Title: (.*?)(?:\nAbstract: |$)", doc.page_content).group(1) if re.search(r"Title: (.*?)(?:\nAbstract: |$)", doc.page_content) else doc.metadata.get('title', 'Unknown'),
                    "abstract": re.search(r"Abstract: (.*?)$", doc.page_content).group(1) if re.search(r"Abstract: (.*?)$", doc.page_content) else "Abstract not available",
                    "pmid": doc.metadata.get("pmid", "Unknown"),
                    "score": doc.metadata.get("bge_score", 0.0)
                } for doc in result_docs
            ] if result_docs else None
        else:
            papers_for_results = None

        yield _make_json_serializable({
            "step": pubmed_step,
            "name": "Searching for Relevant Papers",
            "data": papers_for_results if papers_for_results else "No relevant papers found.",
            "next_step_name": "Retrieving Similar Clinical Trials"
        })

        step_counter += 1
        
        trials_step = step_counter
        print(f"\n--- [START] Step {trials_step}: Retrieving Similar Clinical Trials ---")
        
        drug_keywords_str_for_retrieval = " ".join(drug_keywords)
        trial_similarity_threshold = self.config.get("trial_retrieval", {}).get("trial_similarity_threshold", 0.5)
        
        trial_results_by_phase = self.retrieve_trials_by_phase(
            title, term_for_retrieval, drug_keywords_str_for_retrieval, other1_keyword, other2_keyword,
            description=temp_description, criteria=temp_criteria, top_k=5, similarity_threshold=trial_similarity_threshold
        )
        
        trials_for_results_json = {}
        if trial_results_by_phase:
            for phase, df in trial_results_by_phase.items():
                if not df.empty:
                    trials_for_results_json[phase] = df.reset_index().to_dict(orient="records")
                else:
                    trials_for_results_json[phase] = None
        
        yield {
            "step": trials_step,
            "name": "Retrieving Similar Clinical Trials",
            "data": _make_json_serializable(trials_for_results_json) if trials_for_results_json else "No similar clinical trials found.",
            "next_step_name": "Select Mode for Final Criteria"
        }
        
        step_counter += 1

        mode_selection_step = step_counter
        yield {
            "step": mode_selection_step,
            "name": "Select Mode for Final Criteria",
            "data": {
                "waiting_for_mode": True
            },
            "next_step_name": "Generating Final Inclusion/Exclusion Criteria"
        }
        
        step_counter += 1

    def _retrieve_similar_trials_for_phase(self, query_emb, phase_data, top_k=5):
        """Retrieve similar clinical trials for a specific phase."""
        phase_all_ids = phase_data["all_ids"]
        phase_all_embs = phase_data["all_embs"]
        phase_raw_df = phase_data["raw_df"]

        if not phase_all_ids:
            return pd.DataFrame() # Return empty DataFrame if no trials for this phase

        # Normalize and compute cosine similarity
        def _normalize(arr):
            return arr / np.linalg.norm(arr, axis=-1, keepdims=True)
        
        query_emb_norm = _normalize(query_emb.reshape(1, -1))
        all_embs_norm = _normalize(np.stack(phase_all_embs))
        sims = np.dot(all_embs_norm, query_emb_norm.T).reshape(-1)

        # Get top k results
        # Ensure we don't request more results than available
        actual_top_k = min(top_k, len(phase_all_ids))
        top_idx = np.argsort(sims)[::-1][:actual_top_k]
        
        top_ids = [phase_all_ids[i] for i in top_idx]
        top_sims = sims[top_idx]

        result_df = phase_raw_df.loc[top_ids].copy()
        result_df["similarity"] = top_sims
        
        return result_df

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

    def _filter_trials_by_disease_and_drug(self, trials_df, disease_keyword, drug_keywords):
        """
        Filter trial DataFrame based on matching disease OR drug keywords.
        A trial is kept if its 'disease' field matches the disease_keyword OR 
        its 'intervention_name' field matches any of the drug_keywords.
        """
        if trials_df.empty:
            return trials_df

        # Normalize disease keyword
        input_disease_words = set()
        if disease_keyword:
            input_disease_words = set(re.findall(r'\b\w+\b', disease_keyword.lower()))

        # Normalize drug keywords
        input_drug_words = set()
        if drug_keywords:
            # The drug_keywords from caller is a space-separated string.
            input_drug_words = set(re.findall(r'\b\w+\b', drug_keywords.lower()))

        if not input_disease_words and not input_drug_words:
            # No keywords to filter by, return original df
            return trials_df

        def is_relevant(row):
            # 1. Check for disease match
            trial_disease = row.get('disease')
            if isinstance(trial_disease, str) and trial_disease.strip() and input_disease_words:
                trial_disease_words = set(re.findall(r'\b\w+\b', trial_disease.lower()))
                if not input_disease_words.isdisjoint(trial_disease_words):
                    return True # Disease matched

            # 2. If no disease match, check for drug match in 'intervention_name'
            trial_intervention = row.get('intervention_name')
            if isinstance(trial_intervention, str) and trial_intervention.strip() and input_drug_words:
                trial_intervention_words = set(re.findall(r'\b\w+\b', trial_intervention.lower()))
                if not input_drug_words.isdisjoint(trial_intervention_words):
                    return True # Drug matched

            # 3. No match
            return False

        # Apply the filter row-wise and return the filtered DataFrame
        mask = trials_df.apply(is_relevant, axis=1)
        return trials_df[mask]

    def _filter_trials_by_title_match(self, trials_df, query_title):
        """Filter out trials with titles that match the query title after normalization."""
        if trials_df.empty or not query_title or 'title' not in trials_df.columns:
            return trials_df
        
        # Helper function to normalize titles for comparison
        def normalize_title(title):
            if not isinstance(title, str):
                return ""
            # 1. 转换为小写
            title = title.lower()
            # 2. 移除所有标点符号和特殊字符，只保留字母、数字和空格
            title = re.sub(r'[^\w\s]', '', title)
            # 3. 替换多个连续空格为单个空格
            title = re.sub(r'\s+', ' ', title)
            # 4. 去除首尾空格
            title = title.strip()
            return title
        
        # Normalize the query title
        normalized_query_title = normalize_title(query_title)
        if not normalized_query_title:
            return trials_df
        
        # print(f"--- [DEBUG] 规范化后的查询标题: '{normalized_query_title}' ---")
        
        # 打印所有试验标题及其规范化后的形式
        # print("--- [DEBUG] 试验标题比较: ---")
        # for idx, row in trials_df.iterrows():
        #     trial_title = row.get('title', '')
        #     normalized_trial_title = normalize_title(trial_title)
        #     is_match = normalized_trial_title == normalized_query_title
        #     print(f"ID: {idx}")
        #     print(f"  原始标题: '{trial_title}'")
        #     print(f"  规范化标题: '{normalized_trial_title}'")
        #     print(f"  是否匹配: {is_match}")
        
        # Function to check if titles match after normalization
        def is_different_title(trial_title):
            normalized_trial_title = normalize_title(trial_title)
            return normalized_trial_title != normalized_query_title
        
        # Apply filter - keep only trials with different titles
        mask = trials_df['title'].apply(is_different_title)
        filtered_df = trials_df[mask]
              
        return filtered_df
    
    def _filter_trials_by_similarity(self, trials_df, similarity_threshold=0.5):
        """Filter trials based on similarity score threshold."""
        if trials_df.empty or 'similarity' not in trials_df.columns:
            return trials_df
        
        # Apply the similarity threshold filter
        pre_filter_count = len(trials_df)
        filtered_df = trials_df[trials_df['similarity'] > similarity_threshold]
        post_filter_count = len(filtered_df)
        
        # 不在这里记录日志，因为我们在retrieve_trials_by_phase中已经显示了详细信息
        
        # 如果所有试验都被过滤掉，仍然保留警告
        if filtered_df.empty and not trials_df.empty:
            print(f"--- [WARN] All trials were filtered out due to similarity threshold of {similarity_threshold}. Consider lowering the threshold. ---")
        
        return filtered_df
    
    def retrieve_trials_by_phase(self, title, disease, drug, other1, other2, description, criteria, top_k=5, similarity_threshold=0.5):
        """Retrieve similar clinical trials by iterating through each phase.
        
        This function first retrieves top 10 similar trials, applies filtering,
        and then returns the top_k most similar trials from the filtered results.
        """
        if self.trial_model is None or not self.trial_data_by_phase:
            print("--- [WARN] Trial retrieval components not initialized or no data loaded. Skipping. ---")
            return None
            
        # 设置初始检索的数量为10，以获取更大的候选集
        initial_retrieve_count = 10
        # print(f"--- [INFO] Using new trial retrieval strategy: Get top {initial_retrieve_count} → Apply filters → Return top {top_k} from filtered results ---")

        # --- 1. Encode the query ---

        keywords_list = list(filter(None, [disease, drug]))   # 过滤掉None值
        keywords_for_string = ", ".join(keywords_list)

        query_df = pd.DataFrame([
            {
                "nct_id": "query",
                "title": title or "",
                "intervention_name": drug or "",
                "disease": disease or "",
                # "keyword": keywords_for_string,
                "keyword": "",
                "description": "",
                "criteria":  ""
            }
        ])
        inputs = {
            "x": query_df,
            "fields": self.trial_fields,
            "ctx_fields": self.trial_ctx_fields,  # This is now an empty list
            "tag": "nct_id"
        }
        try:
            query_emb_dict = self.trial_model.encode(inputs, return_dict=True, verbose=False)
            query_emb = list(query_emb_dict.values())[0]
        except Exception as e:
            print(f"--- [ERROR] Failed to encode trial query: {e} ---")
            return None

        # --- 2. Iterate through phases and retrieve results ---
        all_phase_results = {}
        
        # 辅助函数，格式化NCT ID列表显示
        def format_nct_ids(nct_ids, max_display=10):
            if not nct_ids:
                return "none"
            if len(nct_ids) <= max_display:
                return ", ".join(nct_ids)
            else:
                return ", ".join(nct_ids[:max_display]) + f" ... (and {len(nct_ids) - max_display} more)"
        
        # 辅助函数，获取过滤前后的NCT ID差集
        def get_filtered_nct_ids(before_df, after_df):
            # 确保两个DataFrame都有索引，且索引是NCT ID
            if before_df.empty:
                return []
            before_ids = set(before_df.index)
            after_ids = set(after_df.index)
            filtered_ids = before_ids - after_ids
            return list(filtered_ids)
        
        for phase_key, phase_data in self.trial_data_by_phase.items():
            print(f"\n--- [INFO] Retrieving Top {initial_retrieve_count} Similar Trials for Phase: {phase_key} ---")
            
            # Get results for the current phase - initially retrieve more trials
            result_df = self._retrieve_similar_trials_for_phase(query_emb, phase_data, initial_retrieve_count)

            # --- Filter results by multiple criteria ---
            initial_count = len(result_df)
            
            # 添加调试信息：显示初始检索到的所有试验的ID和相似度
            # print(f"--- [DEBUG] 初始检索到 {initial_count} 个试验 ---")
            # if not result_df.empty:
            #     print("--- [DEBUG] 初始试验ID和相似度: ---")
            #     for idx, sim in zip(result_df.index, result_df['similarity']):
            #         print(f"ID: {idx}, 相似度: {sim:.4f}")
            
            if initial_count == 0:
                print("--- [INFO] No trials found for this phase. ---")
                all_phase_results[phase_key] = result_df
                continue
            
            # 创建一个字典来跟踪每个过滤步骤的结果
            filtered_nct_ids = {
                "similarity": [],    # 因相似度低被过滤
                "disease": [],       # 因疾病不匹配被过滤
                "title": []          # 因标题匹配被过滤
            }
            
            # 保存所有检索到的NCT IDs，用于完整追踪
            all_retrieved_ids = set(result_df.index)
            remaining_ids = set(result_df.index)  # 跟踪剩余未过滤的IDs
            
            # 1. First filter by similarity threshold
            print(f"--- [DEBUG] Step 1: Filtering by similarity threshold ({similarity_threshold}) ---")
            original_df = result_df.copy()
            similarity_filtered_df = self._filter_trials_by_similarity(result_df, similarity_threshold)
            similarity_filtered_nct_ids = get_filtered_nct_ids(original_df, similarity_filtered_df)
            filtered_nct_ids["similarity"] = similarity_filtered_nct_ids
            remaining_ids -= set(similarity_filtered_nct_ids)  # 更新剩余IDs
            # print(f"--- [DEBUG] After similarity filter: {len(similarity_filtered_df)} trials remaining, {len(similarity_filtered_nct_ids)} filtered out ---")
            
            # 2. Then filter by disease and drug keywords
            print(f"--- [DEBUG] Step 2: Filtering by disease '{disease}' and drug '{drug}' ---")
            pre_disease_df = similarity_filtered_df.copy()
            disease_filtered_df = self._filter_trials_by_disease(similarity_filtered_df, disease)
            disease_filtered_nct_ids = get_filtered_nct_ids(pre_disease_df, disease_filtered_df)
            
            # 检查任何未通过疾病过滤的ID
            if disease_filtered_nct_ids:
                filtered_nct_ids["disease"] = disease_filtered_nct_ids
                # print(f"--- [DEBUG] Trials filtered by disease/drug: {format_nct_ids(disease_filtered_nct_ids)} ---")
                
            remaining_ids -= set(disease_filtered_nct_ids)  # 更新剩余IDs
            # print(f"--- [DEBUG] After disease/drug filter: {len(disease_filtered_df)} trials remaining, {len(disease_filtered_nct_ids)} filtered out ---")
            
            # 3. Then filter out exact title matches
            print("--- [DEBUG] Step 3: Filtering out exact title matches ---")
            # print(f"--- [DEBUG] 规范化后的查询标题: '{self._normalize_title_for_comparison(title)}' ---")
            pre_title_df = disease_filtered_df.copy()
            filtered_df = self._filter_trials_by_title_match(disease_filtered_df, title)
            title_filtered_nct_ids = get_filtered_nct_ids(pre_title_df, filtered_df)
            
            # 添加标题过滤结果
            if title_filtered_nct_ids:
                filtered_nct_ids["title"] = title_filtered_nct_ids
                
            remaining_ids -= set(title_filtered_nct_ids)  # 更新剩余IDs
            
            # 合并所有过滤日志为一条简明的信息
            filter_logs = []
            
            if filtered_nct_ids["similarity"]:
                filter_logs.append(f"{format_nct_ids(filtered_nct_ids['similarity'])} 因相似度低于 {similarity_threshold} 被过滤")
                
            if filtered_nct_ids["disease"]:
                filter_logs.append(f"{format_nct_ids(filtered_nct_ids['disease'])} 因不匹配疾病 '{disease}' 或药物 '{drug}' 被过滤")
                
            if filtered_nct_ids["title"]:
                filter_logs.append(f"{format_nct_ids(filtered_nct_ids['title'])} 因标题与输入标题匹配被过滤")
            
            # 检查是否有试验被过滤但未记录原因
            all_filtered_ids = set(similarity_filtered_nct_ids).union(set(filtered_nct_ids["disease"])).union(set(filtered_nct_ids["title"]))
            unaccounted_ids = all_retrieved_ids - remaining_ids - all_filtered_ids
            if unaccounted_ids:
                filter_logs.append(f"{format_nct_ids(list(unaccounted_ids))} 因未知原因被过滤")
            
                        # 输出综合过滤信息
            # final_count = len(filtered_df)
            # if filter_logs:
            #     print(f"--- [INFO] 过滤详情: {'; '.join(filter_logs)} ---")
            # elif initial_count > 0 and final_count == 0:
            #     # 如果所有试验都被过滤掉但没有记录具体原因
            #     if len(similarity_filtered_nct_ids) == initial_count:
            #         print(f"--- [INFO] 过滤详情: 所有试验因相似度低于 {similarity_threshold} 被过滤 ---")
            #     else:
            #         print(f"--- [INFO] 过滤详情: 所有 {initial_count} 个试验被过滤，但未记录具体原因 ---")
            
            # # 添加总结性日志
            # print(f"--- [INFO] 总结: 检索到 {initial_count} 个试验，过滤掉 {initial_count - final_count} 个，剩余 {final_count} 个 ---")
            
            # 对过滤后的结果按相似度降序排序，并取top_k个
            if not filtered_df.empty:
                sorted_df = filtered_df.sort_values('similarity', ascending=False)
                final_top_k = min(top_k, len(sorted_df))
                
                if len(sorted_df) > final_top_k:
                    final_df = sorted_df.iloc[:final_top_k]
                    # print(f"--- [INFO] Taking top {final_top_k} trials by similarity from {len(sorted_df)} filtered trials ---")
                else:
                    final_df = sorted_df
                    # print(f"--- [INFO] Using all {len(sorted_df)} remaining trials (fewer than requested top_k={top_k}) ---")
            else:
                final_df = filtered_df
                # print("--- [INFO] No trials remain after filtering ---")

            all_phase_results[phase_key] = final_df

            # --- 3. Print results for the current phase ---
            if final_df.empty:
                # print("--- [INFO] No similar trials found for this phase after filtering. ---")
                continue

            trials_df = final_df.reset_index()
            with pd.option_context('display.max_colwidth', None):
                for idx, row in trials_df.iterrows():
                    try:
                        sim_val = float(row.get('similarity', 0))
                    except Exception:
                        sim_val = 0
                    print(f"\n--- Result [{idx+1}] NCT ID: {row['nct_id']} | Similarity: {sim_val:.3f} ---")
                    for col, val in row.items():
                        if col in ['nct_id', 'similarity']:
                            continue
                        if pd.isna(val) or str(val).strip() == '':
                            continue
                        print(f"{col.replace('_', ' ').title()}: {val}")
                    print("-" * 80)
        
        return all_phase_results

    # 提取标题规范化函数为独立方法，以便重用
    def _normalize_title_for_comparison(self, title):
        """Helper function to normalize titles for comparison."""
        if not isinstance(title, str):
            return ""
        # 1. 转换为小写
        title = title.lower()
        # 2. 移除所有标点符号和特殊字符，只保留字母、数字和空格
        title = re.sub(r'[^\w\s]', '', title)
        # 3. 替换多个连续空格为单个空格
        title = re.sub(r'\s+', ' ', title)
        # 4. 去除首尾空格
        title = title.strip()
        return title

    def _filter_trials_by_title_match(self, trials_df, query_title):
        """Filter out trials with titles that match the query title after normalization."""
        if trials_df.empty or not query_title or 'title' not in trials_df.columns:
            return trials_df
        
        # Normalize the query title
        normalized_query_title = self._normalize_title_for_comparison(query_title)
        if not normalized_query_title:
            return trials_df
        
        # Function to check if titles match after normalization
        def is_different_title(trial_title):
            normalized_trial_title = self._normalize_title_for_comparison(trial_title)
            return normalized_trial_title != normalized_query_title
        
        # Apply filter - keep only trials with different titles
        mask = trials_df['title'].apply(is_different_title)
        filtered_df = trials_df[mask]
        
        return filtered_df

    def _sanitize_generated_context_text(self, text):
        if text is None:
            return ""
        cleaned_text = re.sub(r'\s+', ' ', str(text)).strip()
        if not cleaned_text:
            return ""
        if re.search(r'api error|error code|invalidendpointormodel|json parsing error|failed to generate|traceback', cleaned_text, re.IGNORECASE):
            return ""
        return cleaned_text

    def generate_context_for_retrieval(self, title):
        """Generate context for trial retrieval using RAG expansion from template file."""
        try:
            # 从文件中读取模板而不是嵌入代码
            template_path = self._resolve_path("templates/RAG_expansion_prompt.txt")
            with open(template_path, "r", encoding="utf-8") as f:
                prompt_template = f.read()
            
            # 使用格式化字符串将标题注入模板
            prompt = prompt_template.format(title=title)
            
            try:
                print(f"--- [INFO] Calling API to generate context for title: {title} ---")
                response = self.deepseek_client.client.chat.completions.create(
                    model=self.general_model,
                    messages=[
                        {"role": "system", "content": "You are an experienced medical writer generating structured clinical trial information with precise scientific terminology and methodological detail."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )
                
                print(f"--- [INFO] API response received for context generation ---")
                result_content = response.choices[0].message.content
                # print(f"--- [DEBUG] Raw context response: {result_content[:100]}... ---")
                
                # 处理可能返回的Markdown代码块
                json_str = result_content
                markdown_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
                markdown_match = re.search(markdown_pattern, json_str)
                if markdown_match:
                    print("--- [INFO] Detected Markdown code block in context response, extracting JSON content ---")
                    json_str = markdown_match.group(1)
                
                try:
                    result = json.loads(json_str)
                    description = self._sanitize_generated_context_text(result.get("description", ""))
                    primary_outcome = self._sanitize_generated_context_text(result.get("primary_outcome", ""))
                    return description, "", primary_outcome
                except json.JSONDecodeError as json_err:
                    print(f"--- [ERROR] Failed to parse context JSON: {json_err} ---")
                    return "", "", ""
                    
            except Exception as api_error:
                print(f"--- [ERROR] API error in generate_context_for_retrieval: {api_error} ---")
                # 确保即使API调用失败，也返回默认值
                return "", "", ""
                
        except FileNotFoundError:
            print(f"--- [ERROR] RAG expansion prompt template file not found at {template_path} ---")
            return "", "", ""
        except Exception as e:
            print(f"--- [ERROR] Failed to generate description: {e} ---")
            return "", "", ""



    def _log_generation_details(self, title, prompt, result, original_response=None):
        """Logs the final prompt and AI output to a text file for review."""
        try:
            # Create a sanitized, timestamped filename
            sanitized_title = re.sub(r'[\W_]+', '', title)[:50]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"{sanitized_title}_Log_{timestamp}.txt"
            
            # Ensure the output directory exists
            output_dir = "generation_logs"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            log_path = os.path.join(output_dir, log_filename)

            # 使用utf-8-sig编码，解决开头乱码问题
            with open(log_path, "w", encoding="utf-8-sig") as f:
                # Write Task section header at the beginning
                f.write("="*30 + " FINAL PROMPT " + "="*30 + "\n\n")
                f.write(prompt)
                
                # 记录推理过程 - 简化版本
                if isinstance(result, dict) and 'Reasoning' in result:
                    f.write("\n\n" + "="*30 + " REASONING PROCESS " + "="*30 + "\n\n")
                    f.write(result['Reasoning'])
                
                # 记录最终结果
                f.write("\n\n" + "="*30 + " FINAL CRITERIA " + "="*30 + "\n\n")
                
                # 格式化结果为可读形式
                if isinstance(result, str):
                    try:
                        result_json = json.loads(result)
                    except json.JSONDecodeError:
                        result_json = result
                else:
                    result_json = result
                
                # Format the criteria for human-readable display
                human_readable = self._format_criteria_for_human_readable(result_json)
                f.write(human_readable)
            
            print(f"--- [INFO] Generation details logged to: {log_path} ---")

        except Exception as e:
            print(f"--- [WARN] Failed to write generation log: {e} ---")
    
    def _format_criteria_for_human_readable(self, criteria_json):
        """Formats the criteria JSON into a human-readable text format."""
        if not criteria_json:
            return "--- [ERROR] Could not generate the final trial criteria. ---\n"

        output_lines = []
        
        # Check for direct structure or multi-mode structure
        has_inclusion = 'InclusionCriteria' in criteria_json
        has_exclusion = 'ExclusionCriteria' in criteria_json
        
        # Check for the multi-mode structure
        mode_keys = ['StandardCriteriaSet', 'BroaderCriteriaSet', 'StringentCriteriaSet']
        active_mode = None
        for mode_key in mode_keys:
            if mode_key in criteria_json:
                active_mode = mode_key
                break
        
        # Format based on structure
        if has_inclusion or has_exclusion:
            # Direct structure
            self._append_criteria_section(output_lines, criteria_json, "InclusionCriteria", "Inclusion Criteria")
            self._append_criteria_section(output_lines, criteria_json, "ExclusionCriteria", "Exclusion Criteria")
        elif active_mode:
            # Multi-mode structure
            mode_name = active_mode.replace("CriteriaSet", "")
            mode_criteria = criteria_json.get(active_mode, {})
            
            output_lines.append(f"\n=== {mode_name} Criteria Set ===\n")
            self._append_criteria_section(output_lines, mode_criteria, "InclusionCriteria", "Inclusion Criteria")
            self._append_criteria_section(output_lines, mode_criteria, "ExclusionCriteria", "Exclusion Criteria")
        else:
            # Unknown structure
            output_lines.append("\n=== Generated Criteria (Unrecognized Format) ===\n")
            
        return "\n".join(output_lines)
    
    def _append_criteria_section(self, output_lines, data, section_key, section_title):
        """Helper to append a criteria section to the output lines."""
        criteria_list = data.get(section_key, [])
        if criteria_list:
            output_lines.append(f"\n--- {section_title} ---\n")
            for idx, item in enumerate(criteria_list):
                if isinstance(item, dict):
                    criterion = item.get('Criterion', '')
                    output_lines.append(f"  {idx+1}. {criterion}")
                else:
                    output_lines.append(f"  {idx+1}. {item}")
        else:
            output_lines.append(f"\n--- {section_title} ---\n  No criteria specified.")


    def generate_trial_criteria(self, title, drug_info, papers, trials_by_phase, mode=None, disease_info=None, description=None, primary_outcome=None):
        """Generate inclusion/exclusion criteria based on context.
        
        Args:
            title (str): The research title
            drug_info (list): List of drug information strings
            papers (list): List of paper data
            trials_by_phase (dict): Dictionary of trials by phase
            mode (str, optional): The criteria generation mode ('Standard', 'Broader', or 'Stringent'). Defaults to None (uses config default).
            disease_info (dict, optional): This will now be the structured disease info dict from the new process
            description (str, optional): Trial description
            primary_outcome (str, optional): Primary outcome description
        """
        print("\n--- [START] Generating Final Clinical Trial Criteria via API ---")
        
        # --- Load Prompt Template ---
        try:
            with open(self._resolve_path("templates/Prompt_template_Standard.txt"), "r", encoding="utf-8") as f:
                prompt_template = f.read()
                
            # 处理模板文件，转义JSON示例部分中的花括号，避免与format方法冲突
            # 识别模板中的JSON示例部分并进行转义
            json_example_pattern = r'(<final_answer>\s*\{.*?\}\s*</final_answer>)'
            
            def escape_braces_in_json_example(match):
                # 对JSON示例中的花括号进行转义
                json_example = match.group(1)
                escaped = json_example.replace("{", "{{").replace("}", "}}")
                return escaped
            
            prompt_template = re.sub(json_example_pattern, escape_braces_in_json_example, prompt_template, flags=re.DOTALL)
            
        except FileNotFoundError:
            print("--- [ERROR] 'Prompt_template_Standard.txt' not found. Cannot generate final criteria. ---")
            return {"InclusionCriteria": ["Error: Prompt template file not found."], "ExclusionCriteria": []}

        # --- Get mode-specific instructions ---
        # If mode not specified, use default from config
        if mode is None:
            criteria_config = self.config.get("criteria_generation", {})
            mode = criteria_config.get("default_mode", "Standard")
            
        # Ensure mode has proper capitalization
        mode = mode.capitalize()
        if mode not in ["Standard", "Broader", "Stringent"]:
            mode = "Standard"
            
        # Get the instructions for the selected mode
        criteria_config = self.config.get("criteria_generation", {})
        mode_instructions = criteria_config.get("modes", {}).get(
            mode, 
            "Standard (Balanced Approach): \n- Objective: Balance scientific validity, patient safety, and recruitment feasibility.\n- Typical profile: Clinically justified thresholds; excludes known high-risk groups; allows common comorbidities with controls."
        )
        
        print(f"--- [INFO] Using {mode} mode for criteria generation ---")

        # --- Helper to format context sections safely ---
        def format_context_section(data, formatter_func, not_found_message):
            """
            Formats a context section safely handling different data types.
            """
            # The 'if data:' check is ambiguous for pandas DataFrames.
            # We must explicitly check for non-emptiness for DataFrames,
            # while maintaining the original truthiness check for other types.
            is_valid_data = False
            if isinstance(data, pd.DataFrame):
                is_valid_data = not data.empty
            elif isinstance(data, list):
                is_valid_data = len(data) > 0
            else:
                is_valid_data = bool(data)

            if is_valid_data:
                formatted_data = formatter_func(data)
                if formatted_data and formatted_data.strip():
                    return formatted_data
            
            return not_found_message

        # --- Format Disease Info ---
        def _format_disease(info):
            context_lines = []
            # Create a copy to avoid popping from the original dict
            info_copy = info.copy()
            best_match_name = info_copy.pop('best_match_name', 'N/A')
            context_lines.append(f"* Information for '{best_match_name}':")
            for key, value in info_copy.items():
                formatted_key = key.replace('_', ' ').title()
                formatted_value = ', '.join(map(str, value)) if isinstance(value, list) else value
                context_lines.append(f"* {formatted_key}: {formatted_value}")
            full_text = "\n".join(context_lines)
            return re.sub(r'\n\s*\n', '\n', full_text)
        disease_context = format_context_section(disease_info, _format_disease, "No specific disease information was retrieved. Please rely on general medical knowledge.")

        # --- Format Drug Info ---
        def _format_drug(info_list):
            """Formats a list of drug info strings for the prompt."""
            if not info_list:
                return "No specific drug information was retrieved. Please rely on general medical knowledge."
            
            output = []
            for i, info_str in enumerate(info_list):
                # Try to extract the drug name from the first line for a better title
                name_match = re.search(r'\* Name:\s*(.*)', info_str)
                drug_name = name_match.group(1) if name_match else f"Drug {i+1}"
                
                output.append(f"--- Information for Drug {i+1}: {drug_name} ---")
                output.append(info_str)
            
            # Join with newlines and add spacing between entries
            return "\n\n".join(output)
        drug_context = _format_drug(drug_info)

        # --- Format Papers Info ---
        def _format_papers(paper_list):
            context_lines = []
            for i, paper in enumerate(paper_list): # REMOVED limit of 3
                # 添加PMID到论文标题前
                pmid = paper.get('PMID', paper.get('pmid', 'N/A'))
                title = paper.get('Title', paper.get('title', 'N/A'))
                abstract = paper.get('Abstract', paper.get('abstract', 'N/A'))
                context_lines.append(f"* PMID {pmid} Title: {title}")
                context_lines.append(f"  Abstract: {abstract}")
            full_text = "\n".join(context_lines)
            return re.sub(r'\n\s*\n', '\n', full_text)
        paper_context = format_context_section(papers, _format_papers, "No relevant research papers were retrieved. Please rely on general medical knowledge.")

        # --- Format Trials Info by Phase ---
        def _format_trials(trial_df):
            """Format trial DataFrame to readable text."""
            context_lines = []
            
            # Check if trial_df is a DataFrame or a list
            if isinstance(trial_df, pd.DataFrame):
                # Process DataFrame as before
                if trial_df.empty:
                    return "No similar clinical trials were retrieved. Please rely on general medical knowledge."
                    
                trial_list = trial_df.reset_index().to_dict(orient="records")
                for i, trial in enumerate(trial_list):
                    context_lines.append(f"{i+1}. Similar Trial {i+1} NCT ID: {trial.get('nct_id', 'N/A')}")
                    # Iterate over ALL columns in the original dataframe
                    for col_name in trial_df.columns:
                        # Skip nct_id and similarity as they are already handled or not needed
                        if col_name.lower() in ['nct_id', 'similarity']:
                            continue
                        value = trial.get(col_name)
                        # Check for non-empty/non-null values before printing
                        if value and pd.notna(value) and str(value).strip():
                            # Format column name nicely
                            formatted_col = col_name.replace('_', ' ').title()
                            context_lines.append(f"* {formatted_col}: {value}")
                    context_lines.append("") # Add a blank line between trials
            elif isinstance(trial_df, list):
                # Process list directly
                if not trial_df:
                    return "No similar clinical trials were retrieved. Please rely on general medical knowledge."
                    
                for i, trial in enumerate(trial_df):
                    if isinstance(trial, dict):
                        context_lines.append(f"{i+1}. Similar Trial {i+1} NCT ID: {trial.get('nct_id', 'N/A')}")
                        # Iterate through dictionary items
                        for col_name, value in trial.items():
                            # Skip nct_id and similarity as they are already handled or not needed
                            if col_name.lower() in ['nct_id', 'similarity']:
                                continue
                            # Check for non-empty/non-null values before printing
                            if value and str(value).strip():
                                # Format column name nicely
                                formatted_col = col_name.replace('_', ' ').title()
                                context_lines.append(f"* {formatted_col}: {value}")
                        context_lines.append("") # Add a blank line between trials
            else:
                # Handle unexpected type
                return f"Unexpected data type for trials: {type(trial_df).__name__}. Please rely on general medical knowledge."
            
            # Join and clean up extra newlines
            full_text = "\n".join(context_lines).strip()
            return re.sub(r'\n\s*\n', '\n', full_text)
            
        phase1_context = format_context_section(trials_by_phase.get("phase1") if trials_by_phase else None, _format_trials, "No similar Phase 1 clinical trials were retrieved. Please rely on general medical knowledge.")
        phase2_context = format_context_section(trials_by_phase.get("phase2") if trials_by_phase else None, _format_trials, "No similar Phase 2 clinical trials were retrieved. Please rely on general medical knowledge.")
        phase3_context = format_context_section(trials_by_phase.get("phase3") if trials_by_phase else None, _format_trials, "No similar Phase 3 clinical trials were retrieved. Please rely on general medical knowledge.")

        # --- Fill the template ---
        final_prompt = prompt_template.format(
            Phase3_clinical_trials=phase3_context,
            Phase2_clinical_trials=phase2_context,
            Phase1_clinical_trials=phase1_context,
            Disease_info=disease_context,
            Drugbank_info=drug_context,
            Papers_info=paper_context,
            Trial_description=description,
            Outcome_measure=primary_outcome,
            trial_title=title,
            Mode_Specific_Instructions=mode_instructions
        )

        try:
            print(f"--- [INFO] Sending final prompt to API for criteria generation ---")
            response = self.deepseek_client.client.chat.completions.create(
                model = self.reasoning_model,
                messages=[
                    {"role": "system", "content": "You are a clinical trial design expert that develops eligibility criteria with comprehensive reasoning capabilities. Follow the structured thinking approach outlined in the prompt, using Rationale 💭, Search 🔎, Deduce 🧩, and Draft 📝 steps. Your final output should be a clean JSON object containing InclusionCriteria and ExclusionCriteria arrays."},
                    {"role": "user", "content": final_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0,
                extra_body={"reasoning": True}
            )
            
            print(f"--- [INFO] API response received successfully for criteria generation ---")
            content = response.choices[0].message.content
            reasoning_content = response.choices[0].message.reasoning_content
            
            # 打印response
            print(f"--- [INFO] API response content ---\n {content[:200]}...")  # 只打印前200个字符
            if reasoning_content:
                print(f"--- [INFO] API reasoning content ---\n {reasoning_content[:200]}...")  # 只打印前200个字符

            # 保存原始响应以便记录
            original_response = content
            criteria_result = None
            
            # Use the model's native reasoning content if available
            if reasoning_content:
                print(f"--- [SUCCESS] Using model's native reasoning content ({len(reasoning_content)} characters) ---")
                full_reasoning = reasoning_content
            else:
                # Fallback: try to extract reasoning from the main content
                print("--- [WARN] No reasoning_content available, trying to extract from regular content ---")
                reasoning_matches = re.findall(r'<reasoning>(.*?)</reasoning>', content, re.DOTALL)
                if reasoning_matches:
                    print(f"--- [INFO] Found {len(reasoning_matches)} reasoning sections in content ---")
                    full_reasoning = "\n".join(reasoning_matches)
                else:
                    print("--- [WARN] No reasoning content available ---")
                    full_reasoning = "No reasoning provided in the response."
                
            # 尝试直接解析content为JSON（推理模型应该直接输出JSON）
            try:
                criteria_result = json.loads(content)
                print("--- [SUCCESS] Successfully parsed content as JSON directly ---")
            except json.JSONDecodeError as e:
                print(f"--- [WARN] Content is not directly parseable as JSON: {e} ---")
                
                # 尝试从content中提取JSON
                final_answer_match = re.search(r'<final_answer>(.*?)</final_answer>', content, re.DOTALL)
                if final_answer_match:
                    json_str = final_answer_match.group(1).strip()
                    try:
                        criteria_result = json.loads(json_str)
                        print("--- [SUCCESS] Successfully extracted and parsed final answer JSON ---")
                    except json.JSONDecodeError as e:
                        print(f"--- [ERROR] Failed to parse final answer JSON: {e} ---")
                        # 尝试清理JSON字符串
                        cleaned_str = json_str.strip()
                        if "'" in cleaned_str and '"' not in cleaned_str:
                            cleaned_str = cleaned_str.replace("'", '"')
                        cleaned_str = re.sub(r'[\n\t\r]', ' ', cleaned_str)
                        
                        try:
                            criteria_result = json.loads(cleaned_str)
                            print("--- [SUCCESS] Successfully parsed JSON after cleaning ---")
                        except json.JSONDecodeError:
                            print("--- [ERROR] Still failed to parse JSON after cleaning ---")
                            criteria_result = None
                else:
                    criteria_result = None
            
            # 如果没有找到有效的JSON输出，使用默认结果
            if criteria_result is None:
                print("--- [ERROR] Could not extract valid criteria, using default structure ---")
                criteria_result = {
                    "InclusionCriteria": ["Failed to extract criteria from API response."],
                    "ExclusionCriteria": ["Failed to extract criteria from API response."]
                }
            
            # 将推理过程添加到结果中
            criteria_result['Reasoning'] = full_reasoning
            
            # 检查必要的字段，确保JSON结构完整
            if 'InclusionCriteria' not in criteria_result or not isinstance(criteria_result['InclusionCriteria'], list):
                print("--- [WARN] Missing or invalid 'InclusionCriteria' field, adding default ---")
                criteria_result['InclusionCriteria'] = ["No inclusion criteria available."]
            
            if 'ExclusionCriteria' not in criteria_result or not isinstance(criteria_result['ExclusionCriteria'], list):
                print("--- [WARN] Missing or invalid 'ExclusionCriteria' field, adding default ---")
                criteria_result['ExclusionCriteria'] = ["No exclusion criteria available."]
            
            # 将推理过程添加到结果中
            criteria_result['Reasoning'] = full_reasoning
            
            # 记录生成细节到日志文件
            self._log_generation_details(title, final_prompt, criteria_result, original_response)
                
            print(f"--- [SUCCESS] Successfully generated final trial criteria in {mode} mode using reasoning model. ---")

            return criteria_result
        except json.JSONDecodeError as e:
            print(f"--- [ERROR] JSON parsing error in DeepSeek API response: {e} ---")
            return {
                "InclusionCriteria": ["Error: Failed to parse API response."],
                "ExclusionCriteria": ["Error: Failed to parse API response."],
                "error": f"JSON parsing error: {str(e)}",
                "Reasoning": "Error occurred during JSON parsing."
            }
        except Exception as api_err:
            print(f"--- [ERROR] API error: {api_err} ---")
            # 统一处理所有API相关错误
            err_type = type(api_err).__name__
            if "Authentication" in err_type or "Auth" in err_type:
                print("--- [ERROR] API authentication failed ---")
                return {
                    "InclusionCriteria": ["Error: API authentication failed."],
                    "ExclusionCriteria": ["Error: API authentication failed."],
                    "error": "API authentication error. Please check your API key.",
                    "Reasoning": "Error occurred during API authentication."
                }
            elif "Timeout" in err_type:
                print("--- [ERROR] API request timed out ---")
                return {
                    "InclusionCriteria": ["Error: API request timed out."],
                    "ExclusionCriteria": ["Error: API request timed out."],
                    "error": "API request timed out. Please try again.",
                    "Reasoning": "Error occurred due to API request timeout."
                }
            else:
                print(f"--- [ERROR] API error: {api_err} ---")
                return {
                    "InclusionCriteria": ["Error: API request failed."],
                    "ExclusionCriteria": ["Error: API request failed."],
                    "error": f"API error: {str(api_err)}",
                    "Reasoning": "Error occurred during API request."
                }
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            print(f"--- [ERROR] Failed to generate final trial criteria with DeepSeek: {e} ---")
            print(f"--- [ERROR] Traceback: {traceback_str} ---")
            return {
                "InclusionCriteria": ["Error generating criteria."],
                "ExclusionCriteria": ["Error generating criteria."],
                "error": f"Unexpected error: {str(e)}",
                "Reasoning": "An unexpected error occurred during criteria generation."
            }

    def _resolve_path(self, path_str):
        return _resolve_config_path(path_str, project_root)

def main():
    # Load configuration
    config = load_config()
    
    # Initialize enhanced search system
    search_system = EnhancedMedicalSearch(config)
    
    print("\n" + "="*25 + " Clinical Trial Design System " + "="*25)
    print("This system helps design clinical trials based on a research title.")
    print("Enter a title to begin analysis. Type 'exit' to quit.")
    
    while True:
        # Get research title
        title = input("\nPlease enter a research title: ")
        if title.lower() == 'exit':
            break
            
        if not title.strip():
            print("--- [WARN] Input is empty. Please enter a valid research title. ---")
            continue
        
        # Process the research title
        results_generator = search_system.process_research_title(title)
        
        # Consume the generator to get all results, the last one is the final payload
        results = None
        for result_step in results_generator:
            print(f"--- [MAIN TEST] Received step {result_step.get('step')}: {result_step.get('name')} ---")
            if result_step.get('step') == 8: # Final step has the complete data
                results = result_step.get('data')
        
        # --- Start of new, structured output ---
        
        # The main report is now just the final criteria
        print("\n\n" + "="*25 + " FINAL REPORT: CLINICAL TRIAL CRITERIA " + "="*25)

        # Generated Criteria
        if results and results.get('trial_criteria'):
            # Use the new helper to format criteria for display, ensuring consistency and preventing bugs
            formatted_report = search_system._format_criteria_for_display(results['trial_criteria'])
            print(formatted_report)
        else:
            print("\n--- [ERROR] Could not generate the final trial criteria. ---")

        print("\n" + "="*28 + " END OF REPORT " + "="*29 + "\n")

if __name__ == "__main__":
    main() 