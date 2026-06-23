import pandas as pd
import os
import re
import json
import argparse
from difflib import SequenceMatcher  # Added for fuzzy matching

# --- Default Configuration ---
DEFAULT_DISEASE_FEATURES_PATH = "disease_features.csv"
# 不再需要API URL，因为我们将使用SDK

# --- Constants ---
IGNORE_SUFFIXES = ['disease', 'syndrome', 'disorder', 'finding', 'type']
EXCLUDE_COLS = ['node_index', 'mondo_id', 'group_id_bert', 'group_name_bert', 'mayo_see_doc']


def _load_data(file_path):
    """Loads the CSV data file."""
    if not os.path.exists(file_path):
        print(f"--- [ERROR] Data file not found: '{file_path}' ---")
        return None
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"--- [SUCCESS] Successfully loaded data from: {file_path} ---")
        return df
    except Exception as e:
        print(f"--- [ERROR] Failed to load '{file_path}': {e} ---")
        return None

def _filter_candidate_diseases(df, query_name):
    """
    Filters candidate disease names based on matching rules.
    1. Case-insensitive.
    2. Matches whole words.
    3. Compares against names with common suffixes removed.
    """
    query_lower = query_name.lower()
    regex_pattern = r'\b' + re.escape(query_lower) + r'\b'
    
    candidates = df[df['mondo_name'].str.lower().str.contains(regex_pattern, na=False)]
    
    def clean_name(name):
        name_lower = name.lower()
        for suffix in IGNORE_SUFFIXES:
            name_lower = name_lower.replace(suffix, '').strip()
        return name_lower

    cleaned_query = clean_name(query_name)
    final_candidates = [name for name in candidates['mondo_name'].unique() if cleaned_query in clean_name(name)]
    return final_candidates


def _select_best_match_with_api(candidates, original_query, config):
    """
    Uses the API with a two-step process to select the best match from a list of candidates:
    1. First API call: Analyze the disease and provide a definition
    2. Second API call: Select the best match using the analysis as background
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    print(f"\n--- [INFO] Found {len(candidates)} candidates. Using AI to select the best match... ---")
    
    # Function to try loading a prompt file from multiple possible locations
    def _load_prompt_file(filename):
        possible_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),  # 同目录
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "PrimeKG-main", filename),  # 上层目录下的PrimeKG-main
            f"PrimeKG-main/{filename}",  # 相对于工作目录
            os.path.abspath(f"PrimeKG-main/{filename}")  # 绝对路径
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"--- [INFO] Using prompt from: {path} ---")
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
        
        print(f"--- [WARN] Could not find {filename} in any of the expected locations. Using fallback prompt. ---")
        return None
    
    def _call_api(messages, response_format="json_object"):
        try:
            from clinical_rag.llm_client import chat_completion

            return chat_completion(
                messages,
                config,
                response_format=response_format,
                temperature=0.1,
            )
        except Exception as e:
            print(f"--- [ERROR] API request failed: {e} ---")
            return None
    
    # Step 1: Analyze the disease and get definition
    analysis_prompt = _load_prompt_file("disease_analysis_prompt.txt")
    
    if not analysis_prompt:
        # Fallback analysis prompt
        analysis_prompt = """
        ## Task
        You are a professional medical expert. Analyze the disease {input} and provide a precise definition.
        
        ## Instructions
        1. Analyze {input} using comprehensive medical knowledge.
        2. Provide a clear and concise definition of the disease.
        3. List key characteristics.
        
        ## Output Format
        Provide a JSON with "definition" and "key_characteristics" fields.
        """
    
    # Format the analysis prompt
    analysis_prompt = analysis_prompt.replace("{input}", original_query)
    
    print("--- [INFO] Step 1: Getting disease analysis... ---")
    disease_analysis_json = _call_api([
        {"role": "system", "content": "You are a professional medical expert analyzing a disease."},
        {"role": "user", "content": analysis_prompt}
    ])
    
    # Parse the analysis result
    try:
        analysis_result = json.loads(disease_analysis_json)
        disease_definition = analysis_result.get("definition", "No definition available.")
        characteristics = analysis_result.get("key_characteristics", [])
        
        # Format the analysis for the next step
        disease_analysis = f"Definition: {disease_definition}\n\n"
        if characteristics:
            disease_analysis += "Key Characteristics:\n" + "\n".join([f"- {c}" for c in characteristics])
        
        print("--- [SUCCESS] Disease analysis completed ---")
    except Exception as e:
        print(f"--- [WARN] Failed to parse disease analysis JSON: {e}. Proceeding with minimal analysis. ---")
        disease_analysis = f"Disease term: {original_query}"
    
    # Step 2: Select the best match using the analysis as background
    selection_prompt = _load_prompt_file("disease_selection_prompt.txt")
    
    if not selection_prompt:
        # Fallback selection prompt
        selection_prompt = """
        ## Task
        Select the most appropriate disease term for {input} from the candidates.
        
        ## Background Knowledge
        {disease_analysis}
        
        ## Candidate Terms
        {matched_terms}
        
        ## Output Format
        Respond ONLY with a JSON object: {"best_match": "selected term"}
        """
    
    # Format the selection prompt
    selection_prompt = selection_prompt.replace("{input}", original_query)
    selection_prompt = selection_prompt.replace("{matched_terms}", str(candidates))
    selection_prompt = selection_prompt.replace("{disease_analysis}", disease_analysis)
    
    print("--- [INFO] Step 2: Selecting best disease match... ---")
    selection_json = _call_api([
        {"role": "system", "content": "You are a professional medical expert selecting the most appropriate disease term."},
        {"role": "user", "content": selection_prompt}
    ])
    
    # Parse the selection result
    try:
        selection_result = json.loads(selection_json)
        best_match = selection_result.get("best_match")
        
        # Verify the best match is in the candidate list
        if best_match in candidates:
            print(f"--- [SUCCESS] AI selected best match: '{best_match}' ---")
            return best_match
        else:
            print(f"--- [WARN] Selected match '{best_match}' not in candidate list. Using fuzzy matching... ---")
            
            # Find the closest match in the candidates list
            closest_match = None
            highest_similarity = -1
            
            for candidate in candidates:
                similarity = SequenceMatcher(None, best_match.lower(), candidate.lower()).ratio()
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    closest_match = candidate
            
            print(f"--- [INFO] Closest match using fuzzy matching: '{closest_match}' with similarity {highest_similarity:.2f} ---")
            return closest_match
            
    except Exception as e:
        print(f"--- [ERROR] Failed to parse selection JSON: {e}. Defaulting to first candidate. ---")
        return candidates[0]


def _aggregate_features(df, disease_name):
    """
    Collects and aggregates all feature information for a given disease name.
    """
    matching_rows = df[df['mondo_name'] == disease_name].copy()
    if matching_rows.empty:
        return {}
        
    aggregated_info = {}
    for col in df.columns:
        if col in EXCLUDE_COLS or col == 'mondo_name':
            continue
            
        values = matching_rows[col].dropna().unique().tolist()
        cleaned_values = [str(v).strip() for v in values if str(v).strip()]
        
        if cleaned_values:
            aggregated_info[col] = cleaned_values
            
    return aggregated_info


class AdvancedDiseaseExtractor:
    """
    A class to extract detailed disease information from a local database,
    using an AI model to disambiguate matches.
    """
    def __init__(self, data_path=DEFAULT_DISEASE_FEATURES_PATH, api_key=None, config=None):
        """
        Initializes the extractor.
        :param data_path: Path to the disease features CSV file.
        :param api_key: Deprecated; use config['llm']['api_key'] or api_key.md.
        :param config: Full project config dict (llm settings from config.json / api_key.md).
        """
        self.disease_df = _load_data(data_path)
        self.config = config or {}
        if api_key:
            llm = self.config.setdefault("llm", {})
            llm.setdefault("api_key", api_key)

    def get_info(self, disease_name):
        """
        Main method to get all information for a given disease name.
        :param disease_name: The name of the disease to search for.
        :return: A dictionary containing aggregated features of the disease, or None if an error occurs.
        """
        if self.disease_df is None:
            print("--- [ERROR] Disease data not loaded. Cannot perform search. ---")
            return None
        
        llm_key = (self.config.get("llm") or {}).get("api_key")
        if not llm_key:
            print("--- [ERROR] LLM API key not set. Cannot perform smart selection. ---")
            return None

        # 1. Filter candidates with the full phrase first
        print(f"--- [INFO] Searching for disease phrase: '{disease_name}' ---")
        candidates = _filter_candidate_diseases(self.disease_df, disease_name)

        # If no results for the phrase, try individual words as a fallback
        if not candidates:
            print(f"--- [INFO] No candidates found for '{disease_name}'. Attempting to search with individual words. ---")
            # Split into words (at least 3 characters long)
            words = [word for word in re.split(r'[\s,;-]+', disease_name.lower()) if len(word) > 2]
            all_word_candidates = []
            for word in words:
                print(f"--- [INFO] Searching for individual word: '{word}' ---")
                word_candidates = _filter_candidate_diseases(self.disease_df, word)
                if word_candidates:
                    all_word_candidates.extend(word_candidates)
            
            # Use unique candidates from individual word searches
            candidates = sorted(list(set(all_word_candidates)))

        if not candidates:
            print(f"--- [INFO] No candidates found for '{disease_name}' or its constituent words. ---")
            return {}

        # 2. Select best match from the (potentially expanded) candidate list
        best_match = _select_best_match_with_api(candidates, disease_name, self.config)
        if not best_match:
            print(f"--- [WARN] Could not determine a best match for '{disease_name}'. ---")
            return {}
            
        # 3. Aggregate features
        features = _aggregate_features(self.disease_df, best_match)
        features['best_match_name'] = best_match # Add the matched name for clarity
        
        return features


def main():
    """Main function for command-line interaction."""
    parser = argparse.ArgumentParser(description="Advanced Disease Information Extractor")
    parser.add_argument("disease_name", nargs='?', help="The name of the disease to look up. If not provided, runs in interactive mode.")
    parser.add_argument("--data_path", default=DEFAULT_DISEASE_FEATURES_PATH, help="Path to the disease_features.csv file.")
    parser.add_argument("--api_key", default=os.environ.get("LLM_API_KEY"), help="LLM API key. Can also be set via LLM_API_KEY or api_key.md.")
    
    args = parser.parse_args()

    if not args.api_key:
        print("--- [ERROR] LLM API key is required. Provide --api_key, LLM_API_KEY, or api_key.md. ---")
        return

    extractor = AdvancedDiseaseExtractor(data_path=args.data_path, config={"llm": {"api_key": args.api_key}})
    if extractor.disease_df is None:
        return

    def process_query(query):
        features = extractor.get_info(query)
        if features:
            best_match = features.pop('best_match_name', query)
            print(f"\n===== Features for '{best_match}' =====")
            if not features:
                print("No detailed features found for this disease.")
            else:
                for key, value in features.items():
                    print(f"\n--- {key.replace('_', ' ').title()} ---")
                    if len(value) == 1:
                        print(value[0])
                    else:
                        for item in value:
                            print(f"- {item}")
            print("========================================")

    if args.disease_name:
        process_query(args.disease_name)
    else:
        try:
            print("--- Running in Interactive Mode ---")
            while True:
                query = input("\nEnter a disease name (or 'q' to quit): ")
                if query.lower() == 'q':
                    break
                if query.strip():
                    process_query(query)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting program.")

if __name__ == "__main__":
    main() 