import json
import pandas as pd

def filter_trials_by_nctid(jsonl_path, csv_path, output_csv_path):
    # Read NCTids from JSONL file
    nct_ids = set()
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if 'nctId' in data:
                nct_ids.add(data['nctId'])

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Filter DataFrame based on NCTids
    filtered_df = df[df['nctId'].isin(nct_ids)]

    # Write the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    jsonl_file = 'self/trial_gov/SFT_train_test/After_date_test_set/test_trials_152.jsonl'
    csv_file = 'self/trial_gov/SFT_train_test/After_date_test_set/raw/test_trials_160.csv'
    output_file = 'test_trials_152.csv'
    
    filter_trials_by_nctid(jsonl_file, csv_file, output_file)
    print(f"Filtered data saved to {output_file}") 