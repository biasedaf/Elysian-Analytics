import os
from Bio import SeqIO
import pandas as pd

# --- Configuration ---
# This new section builds correct, absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

# A dictionary mapping the filename to the taxon label you want to use
INPUT_FILES = {
    "metazoa_data.fasta": "metazoa",
    "fungi_data.fasta": "fungi",
    "diatoms_data.fasta": "diatoms"
}

# The names for your final output files
COMBINED_FASTA_OUTPUT = os.path.join(RAW_DATA_DIR, "training_data.fasta")
LABELS_CSV_OUTPUT = os.path.join(RAW_DATA_DIR, "training_labels.csv")

# --- Script (No changes needed below here) ---
def create_training_files():
    all_records = []
    label_data = []
    print("--- Starting Data Preparation ---")
    print(f"Looking for raw data in: {RAW_DATA_DIR}")

    for filename, taxon in INPUT_FILES.items():
        filepath = os.path.join(RAW_DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"\n[WARNING] File not found: {filename}. Skipping.")
            continue
        
        records_in_file = 0
        for record in SeqIO.parse(filepath, "fasta"):
            all_records.append(record)
            label_data.append({"Sequence_ID": record.id, "Taxon": taxon})
            records_in_file += 1
        
        if records_in_file > 0:
            print(f"[SUCCESS] Processed {records_in_file} records from {filename}.")
        else:
            print(f"[WARNING] File {filename} was found but is empty.")

    if not all_records:
        print("\n--- Data Preparation Failed ---")
        print("Error: No sequences were processed. Check the WARNING messages above.")
        return

    print(f"\nWriting {len(all_records)} sequences to {os.path.basename(COMBINED_FASTA_OUTPUT)}...")
    SeqIO.write(all_records, COMBINED_FASTA_OUTPUT, "fasta")
    
    print(f"Writing {len(label_data)} labels to {os.path.basename(LABELS_CSV_OUTPUT)}...")
    labels_df = pd.DataFrame(label_data)
    labels_df.to_csv(LABELS_CSV_OUTPUT, index=False)
    
    print("\n--- Data Preparation Complete! ---")

if __name__ == "__main__":
    create_training_files()