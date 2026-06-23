import json
import argparse
import os
import sys
import time
import csv
from tqdm import tqdm as _tqdm

_is_powershell = bool(os.environ.get("PSModulePath"))

def tqdm(iterable, **kwargs):
    if _is_powershell:
        kwargs.setdefault("ascii", True)
    kwargs.setdefault("file", sys.stdout)
    return _tqdm(iterable, **kwargs)
from volcenginesdkarkruntime import Ark
import glob
import re

script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from clinical_rag.config import load_config

class DeepSeekClient:
    """Client for the DeepSeek API."""
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("API key cannot be empty.")
        self.client = Ark(api_key=api_key)
        self.model_name = "deepseek-v3-250324"

    def get_response(self, prompt, max_retries=3, retry_delay=5):
        """Gets a response for a given prompt with retry logic."""
        for attempt in range(max_retries):
            try:
                print(f"--- [INFO] Calling API with model {self.model_name}. Attempt {attempt + 1}/{max_retries}... ---")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0
                )
                result = response.choices[0].message.content
                print("--- [INFO] API response received successfully. ---")
                return result
            except Exception as e:
                print(f"--- [WARN] API call failed on attempt {attempt + 1}: {e} ---")
                if attempt + 1 == max_retries:
                    print("--- [ERROR] Max retries reached. Skipping this prompt. ---")
                    return None
                print(f"--- [INFO] Retrying in {retry_delay} seconds... ---")
                time.sleep(retry_delay)
        return None

# --- Core Logic ---

def parse_and_format_response(response_content):
    """
    Parses the JSON response from the LLM and formats it into a human-readable string.
    """
    if not response_content:
        return "Error: Empty response from API."

    # 1. Extract JSON string from markdown block if present
    json_str = response_content
    # This regex handles both ```json and ```
    markdown_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    markdown_match = re.search(markdown_pattern, json_str, re.DOTALL)
    if markdown_match:
        json_str = markdown_match.group(1)

    # 2. Parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        print(f"--- [WARN] Failed to parse JSON from response. Saving raw response. ---")
        return response_content # Fallback to raw response

    # 3. Format the data into the desired readable format
    output_lines = []
    
    # Format Inclusion Criteria
    inclusion_criteria = data.get("InclusionCriteria", [])
    if inclusion_criteria:
        output_lines.append("Inclusion Criteria:")
        output_lines.append("") # Blank line after title
        for item in inclusion_criteria:
            if isinstance(item, dict) and "Criterion" in item:
                output_lines.append(f"* {item['Criterion']}\n")
    
    # # Add a blank line before the next section if both exist and are not empty
    # if inclusion_criteria and data.get("ExclusionCriteria"):
    #     output_lines.append("")
    
    # Format Exclusion Criteria
    exclusion_criteria = data.get("ExclusionCriteria", [])
    if exclusion_criteria:
        output_lines.append("Exclusion Criteria:")
        output_lines.append("") # Blank line after title
        for item in exclusion_criteria:
            if isinstance(item, dict) and "Criterion" in item:
                output_lines.append(f"* {item['Criterion']}\n")
            elif isinstance(item, str):
                output_lines.append(f"* {item}")
    
    if not output_lines:
        return "Error: Could not find 'InclusionCriteria' or 'ExclusionCriteria' in the parsed JSON."

    return "\n".join(output_lines)

def load_processed_nct_ids_from_csv(output_csv):
    """Reads an existing output CSV to find which NCT IDs have been processed."""
    processed_ids = set()
    if not os.path.exists(output_csv):
        return processed_ids
    
    try:
        with open(output_csv, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            # Skip header
            try:
                next(reader)
            except StopIteration:
                # File is empty
                return processed_ids
            
            for row in reader:
                if row: # Ensure row is not empty
                    processed_ids.add(row[0])
    except Exception as e:
        print(f"--- [WARN] Could not read processed IDs from {output_csv}: {e} ---")

    if processed_ids:
        print(f"--- [INFO] Found {len(processed_ids)} already processed trials in {output_csv}. They will be skipped. ---")
    return processed_ids

def generate_responses_from_prompts(prompt_dir, output_csv, target_nct_id=None):
    """
    Generates responses for all prompts in a directory and saves them to a CSV file.
    """
    # --- 1. Initialization ---
    config = load_config()
    api_key = config.get("api_keys", {}).get("deepseek_api_key")
    if not api_key:
        print("--- [FATAL] 'deepseek_api_key' not found in config.json. ---")
        return

    client = DeepSeekClient(api_key=api_key)
    processed_ids = load_processed_nct_ids_from_csv(output_csv)

    # --- 2. Find and filter prompt files ---
    if not os.path.isdir(prompt_dir):
        print(f"--- [FATAL] Prompt directory not found: {prompt_dir} ---")
        return
    
    prompt_files = glob.glob(os.path.join(prompt_dir, "*_prompt.txt"))
    
    prompts_to_process = []
    for f_path in prompt_files:
        nct_id = os.path.basename(f_path).split('_')[0]
        
        if target_nct_id and nct_id != target_nct_id:
            continue
            
        if nct_id in processed_ids:
            continue
            
        prompts_to_process.append({'nctId': nct_id, 'path': f_path})

    if target_nct_id and not prompts_to_process:
        if target_nct_id in processed_ids:
            print(f"--- [INFO] Target trial {target_nct_id} has already been processed. ---")
        else:
            print(f"--- [WARN] Prompt file for target trial {target_nct_id} not found in {prompt_dir}. ---")
        return

    if not prompts_to_process:
        print("--- [INFO] No new prompts to process. All prompts are already completed. ---")
        return
        
    print(f"--- [INFO] Starting response generation for {len(prompts_to_process)} prompts. ---")

    # --- 3. Process each prompt ---
    file_exists = os.path.exists(output_csv)
    with open(output_csv, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        
        # Write header only if the file is new
        if not file_exists or os.path.getsize(output_csv) == 0:
            writer.writerow(["NCTid", "Generated_Response"])
            
        for prompt_info in tqdm(prompts_to_process, desc="Generating Responses"):
            nct_id = prompt_info['nctId']
            file_path = prompt_info['path']
            
            print(f"\n--- [INFO] Processing prompt for trial: {nct_id} ---")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as pf:
                    prompt_content = pf.read()
            except Exception as e:
                print(f"--- [ERROR] Could not read prompt file {file_path}: {e}. Skipping. ---")
                continue

            # --- Call API to generate response ---
            response_content = client.get_response(prompt_content)
            
            if response_content:
                # --- Parse and format the response ---
                formatted_response = parse_and_format_response(response_content)

                # --- Save result ---
                writer.writerow([nct_id, formatted_response])
                f.flush() # Ensure it's written to disk immediately for resumability
                print(f"--- [SUCCESS] Successfully generated and saved response for {nct_id}. ---")
            else:
                print(f"--- [ERROR] Failed to generate response for {nct_id} after all retries. ---")

    print(f"\n--- [COMPLETE] Finished processing all prompts. Output is saved to: {output_csv} ---")

def main():
    parser = argparse.ArgumentParser(
        description="Generate LLM responses for a folder of prompt files and save to a CSV.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--prompt_dir", 
        type=str, 
        default="outputs/prompts/",
        help="Path to the directory containing the prompt text files."
    )
    parser.add_argument(
        "--output_csv", 
        type=str, 
        default="outputs/responses/generated_responses.csv",
        help="Path to save the output CSV file."
    )
    parser.add_argument(
        "--nct_id",
        type=str,
        default=None,
        help="Optional. Provide a specific NCT ID to process only that trial's prompt."
    )
    args = parser.parse_args()

    generate_responses_from_prompts(args.prompt_dir, args.output_csv, args.nct_id)

if __name__ == "__main__":
    main()