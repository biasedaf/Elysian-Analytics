# src/prepare_data.py
import os
from Bio import SeqIO
import pandas as pd

# --- Configuration ---
# Get the absolute path to the directory where this script is located
# This makes the script more reliable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "raw")

# A dictionary mapping the filename to the taxon label you want to use
INPUT_FILES = {
    "metazoa_data.fasta": "metazoa",
    "fungi_data.fasta": "fungi",
    "diatoms_data.fasta": "diatoms"
}

# The names for your final output files
COMBINED_FASTA_OUTPUT = os.path.join(RAW_DATA_DIR, "training_data.fasta")
LABELS_CSV_OUTPUT = os.path.join(RAW_DATA_DIR, "training_labels.csv")

# --- Script ---
def create_training_files():
    """
    Combines multiple FASTA files and creates a corresponding CSV labels file.
    """
    all_records = []
    label_data = []
    total_records_processed = 0

    print("--- Starting Data Preparation ---")
    print(f"Looking for raw data in: {RAW_DATA_DIR}")

    for filename, taxon in INPUT_FILES.items():
        filepath = os.path.join(RAW_DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            print(f"\n[WARNING] File not found: {filename}. Skipping.")
            continue
        
        records_in_file = 0
        try:
            # Read each FASTA file
            for record in SeqIO.parse(filepath, "fasta"):
                all_records.append(record)
                sequence_id = record.id
                label_data.append({"Sequence_ID": sequence_id, "Taxon": taxon})
                records_in_file += 1
            
            if records_in_file > 0:
                print(f"[SUCCESS] Processed {records_in_file} records from {filename} for taxon '{taxon}'.")
                total_records_processed += records_in_file
            else:
                print(f"[WARNING] File {filename} was found but contained 0 records.")

        except Exception as e:
            print(f"[ERROR] Could not process file {filename}. Reason: {e}")


    if not all_records:
        print("\n--- Data Preparation Failed ---")
        print("Error: No sequences were processed. Please check the WARNING messages above.")
        return

    # --- Create the combined FASTA file ---
    print(f"\nWriting a total of {len(all_records)} sequences to {os.path.basename(COMBINED_FASTA_OUTPUT)}...")
    SeqIO.write(all_records, COMBINED_FASTA_OUTPUT, "fasta")
    print("Combined FASTA file created successfully.")

    # --- Create the labels CSV file ---
    print(f"Writing {len(label_data)} labels to {os.path.basename(LABELS_CSV_OUTPUT)}...")
    labels_df = pd.DataFrame(label_data)
    labels_df.to_csv(LABELS_CSV_OUTPUT, index=False)
    print("Labels CSV file created successfully.")
    
    print("\n--- Data Preparation Complete! ---")

if __name__ == "__main__":
    create_training_files()