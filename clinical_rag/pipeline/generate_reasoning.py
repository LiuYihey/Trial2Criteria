import json
import argparse
import os
import sys
import time
from tqdm import tqdm as _tqdm

_is_powershell = bool(os.environ.get("PSModulePath"))

def tqdm(iterable, **kwargs):
    if _is_powershell:
        kwargs.setdefault("ascii", True)
    kwargs.setdefault("file", sys.stdout)
    return _tqdm(iterable, **kwargs)
from volcenginesdkarkruntime import Ark
import re

script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from clinical_rag.config import load_config

class DeepSeekReasoningClient:
    """Client for DeepSeek API to generate reasoning."""
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("API key cannot be empty.")
        self.client = Ark(api_key=api_key)
        self.model_name = "deepseek-v3-1-250821"

    def generate(self, prompt, max_retries=3, retry_delay=5):
        """Generates reasoning for a given prompt with retry logic."""
        for attempt in range(max_retries):
            try:
                print(f"--- [INFO] Calling API with model {self.model_name}. Attempt {attempt + 1}/{max_retries}... ---")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3 # Lower temperature for more deterministic reasoning
                )
                result = response.choices[0].message.content
                print("--- [INFO] API response received successfully. ---")
                return result
            except Exception as e:
                print(f"--- [WARN] API call failed on attempt {attempt + 1}: {e} ---")
                if attempt + 1 == max_retries:
                    print("--- [ERROR] Max retries reached. Skipping this sample. ---")
                    return None
                print(f"--- [INFO] Retrying in {retry_delay} seconds... ---")
                time.sleep(retry_delay)
        return None

# --- Core Logic ---

def load_processed_nct_ids(output_file):
    """Reads an existing output JSONL to find which NCT IDs have been processed."""
    processed_ids = set()
    if not os.path.exists(output_file):
        return processed_ids
    
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'nctId' in data:
                    processed_ids.add(data['nctId'])
            except json.JSONDecodeError:
                print(f"--- [WARN] Skipping malformed JSON in output file: {line.strip()} ---")
    
    if processed_ids:
        print(f"--- [INFO] Found {len(processed_ids)} already processed trials in {output_file}. They will be skipped. ---")
    return processed_ids

def generate_reasoning_for_trials(input_jsonl, template_file, output_jsonl, target_nct_id=None, output_txt_dir=None):
    """
    Generates reasoning for clinical trials using a template and an LLM.
    """
    # --- 1. Initialization ---
    config = load_config()
    api_key = config.get("api_keys", {}).get("deepseek_api_key")
    if not api_key:
        print("--- [FATAL] 'deepseek_api_key' not found in config.json. ---")
        return

    client = DeepSeekReasoningClient(api_key=api_key)

    try:
        with open(template_file, 'r', encoding='utf-8') as f:
            template = f.read()
        print(f"--- [INFO] Successfully loaded prompt template from: {template_file} ---")
    except FileNotFoundError:
        print(f"--- [FATAL] Prompt template file not found at: {template_file} ---")
        return

    processed_ids = load_processed_nct_ids(output_jsonl)

    # --- 2. Load and filter trials ---
    try:
        with open(input_jsonl, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
    except FileNotFoundError:
        print(f"--- [FATAL] Input file not found: {input_jsonl} ---")
        return

    trials_to_process = []
    for i, line in enumerate(all_lines):
        try:
            data = json.loads(line)
            nct_id = data.get('nctId')
            if not nct_id:
                print(f"--- [WARN] Skipping line {i+1} because nctId is missing. ---")
                continue
            
            # Filter for target_nct_id if provided
            if target_nct_id and nct_id != target_nct_id:
                continue

            # Filter for already processed trials
            if nct_id in processed_ids:
                continue
            
            trials_to_process.append(data)

        except json.JSONDecodeError:
            print(f"--- [WARN] Skipping malformed JSON on line {i+1} in input file. ---")
            continue
    
    if target_nct_id and not trials_to_process:
        if target_nct_id in processed_ids:
            print(f"--- [INFO] Target trial {target_nct_id} has already been processed. ---")
        else:
            print(f"--- [WARN] Target trial {target_nct_id} not found in {input_jsonl}. ---")
        return
    
    if not trials_to_process:
        print("--- [INFO] No new trials to process. All trials are already completed. ---")
        return

    print(f"--- [INFO] Starting reasoning generation for {len(trials_to_process)} trials. ---")

    # --- 3. Process each trial ---
    with open(output_jsonl, 'a', encoding='utf-8') as f_out:
        for data in tqdm(trials_to_process, desc="Generating Reasoning"):
            nct_id = data['nctId']
            print(f"\n--- [INFO] Processing trial: {nct_id} ---")
            
            # --- Prepare data for formatting ---
            format_data = data.copy()
            default_text = "No specific information was retrieved. Please rely on general medical knowledge."
            
            # Dynamically find all placeholders in the template
            template_keys = re.findall(r'\{(\w+)\}', template)

            # Ensure all template keys are present in format_data, using default text if missing or None/empty
            for key in template_keys:
                value = format_data.get(key)
                if value is None or (isinstance(value, str) and value.strip().lower() in ['none', '']):
                    format_data[key] = default_text
            
            # --- Format the template ---
            try:
                # Use format_map to gracefully handle any keys that might be in data but not in template
                prompt = template.format_map(format_data)
            except KeyError as e:
                print(f"--- [WARN] Skipping {nct_id} due to a missing key required by the template: {e} ---")
                continue
            
            # --- Call API to generate reasoning ---
            reasoning_response = client.generate(prompt)
            
            if reasoning_response:
                # --- Save result to JSONL ---
                result = {
                    "nctId": nct_id,
                    "generated_reasoning": reasoning_response
                }
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                f_out.flush() # Ensure it's written to disk immediately for resumability
                
                # --- Save reasoning to a human-readable TXT file ---
                # Automatically determine TXT output directory based on JSONL output path
                jsonl_basename = os.path.basename(output_jsonl)
                jsonl_dirname = os.path.dirname(output_jsonl)
                txt_output_dir = os.path.join(jsonl_dirname, jsonl_basename.replace(".jsonl", "_txt"))
                
                os.makedirs(txt_output_dir, exist_ok=True)
                txt_output_path = os.path.join(txt_output_dir, f"{nct_id}_reasoning.txt")
                with open(txt_output_path, 'w', encoding='utf-8') as txt_f:
                    txt_f.write(reasoning_response)
                print(f"--- [INFO] Saved human-readable reasoning for {nct_id} to {txt_output_path}. ---")
                
                print(f"--- [SUCCESS] Successfully generated and saved reasoning for {nct_id}. ---")
            else:
                print(f"--- [ERROR] Failed to generate reasoning for {nct_id} after all retries. ---")

    print(f"\n--- [COMPLETE] Finished processing all trials. Output is saved to: {output_jsonl} --- ")
    if txt_output_dir:
        print(f"--- [COMPLETE] Human-readable TXT outputs are saved to: {txt_output_dir} ---")

def main():
    parser = argparse.ArgumentParser(
        description="Generate reasoning for clinical trials using a template and a Large Language Model.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--input_jsonl", 
        type=str, 
        default="outputs/rag/trials_rag_152.jsonl",
        help="Path to the input JSONL file with trial data."
    )
    parser.add_argument(
        "--template_file", 
        type=str, 
        default="templates/Reasoning_generation_prompt.txt",
        help="Path to the reasoning generation prompt template."
    )
    parser.add_argument(
        "--output_jsonl", 
        type=str, 
        default="outputs/rag/generated_reasoning.jsonl",
        help="Path to save the output JSONL file with generated reasoning."
    )
    parser.add_argument(
        "--nct_id",
        type=str,
        default=None,
        help="Optional. Provide a specific NCT ID to process only that trial."
    )
    args = parser.parse_args()

    generate_reasoning_for_trials(args.input_jsonl, args.template_file, args.output_jsonl, args.nct_id)

if __name__ == "__main__":
    main()
