import json
import argparse
import os

def generate_prompts(input_jsonl, template_file, output_dir):
    """
    Reads a JSONL file and a template to generate full prompts.

    Args:
        input_jsonl (str): Path to the input JSONL file.
        template_file (str): Path to the prompt template file.
        output_dir (str): Path to save the generated prompt files.
    """
    # --- 1. Read the prompt template ---
    try:
        with open(template_file, 'r', encoding='utf-8') as f:
            template = f.read()
        print(f"--- [INFO] Successfully loaded prompt template from: {template_file} ---")
    except FileNotFoundError:
        print(f"--- [FATAL] Prompt template file not found at: {template_file} ---")
        return

    # --- 2. Process the JSONL file and generate prompts ---
    print(f"--- [INFO] Starting prompt generation from: {input_jsonl} ---")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    prompts_generated = 0
    with open(input_jsonl, 'r', encoding='utf-8') as f_in:
        
        for i, line in enumerate(f_in):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"--- [WARN] Skipping malformed JSON on line {i+1} ---")
                continue

            # --- Prepare data for formatting ---
            # Create a copy to avoid modifying the original data object
            format_data = data.copy()
            
            # The template has a placeholder for mode-specific instructions.
            # Since this script is decoupled from the original mode selection,
            # we'll provide a default, neutral instruction set.
            format_data['Mode_Specific_Instructions'] = "Standard (Balanced Approach): \n- Objective: Balance scientific validity, patient safety, and recruitment feasibility.\n- Typical profile: Clinically justified thresholds; excludes known high-risk groups; allows common comorbidities with controls."

            # Replace any 'none' values with the standard placeholder text
            default_text = "No specific information was retrieved. Please rely on general medical knowledge."
            for key, value in format_data.items():
                if value is None or (isinstance(value, str) and value.strip().lower() == 'none'):
                    format_data[key] = default_text
            
            # --- Format the template ---
            try:
                # Use format_map to gracefully handle any missing keys, though our data should be complete.
                prompt = template.format_map(format_data)
                
                # Write header and prompt to the output file
                nct_id = data.get('nctId', 'UnknownID')
                if nct_id == 'UnknownID':
                    print(f"--- [WARN] Skipping line {i+1} because nctId is missing. ---")
                    continue
                
                output_path = os.path.join(output_dir, f"{nct_id}_prompt.txt")
                
                with open(output_path, 'w', encoding='utf-8') as f_out:
                    f_out.write(prompt)
                
                prompts_generated += 1

            except KeyError as e:
                print(f"--- [WARN] Skipping prompt for {data.get('nctId', 'UnknownID')} due to missing key in template: {e} ---")
                continue

    print(f"--- [SUCCESS] Finished. Generated {prompts_generated} prompts. ---")
    print(f"--- [INFO] Output saved to directory: {output_dir} ---")

def main():
    parser = argparse.ArgumentParser(description="Generate full prompts from a JSONL file and a template.")
    parser.add_argument(
        "--input_jsonl", 
        type=str, 
        default="outputs/rag/trials_rag_152.jsonl",
        help="Path to the input JSONL file from batch RAG."
    )
    parser.add_argument(
        "--template_file", 
        type=str, 
        default="templates/Prompt_template_noRAG_noCoT.txt",
        help="Path to the prompt template file."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs/prompts",
        help="Path to the directory where generated prompt files will be saved."
    )
    args = parser.parse_args()

    generate_prompts(args.input_jsonl, args.template_file, args.output_dir)

if __name__ == "__main__":
    main() 