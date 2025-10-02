import os
from Bio import SeqIO
import pandas as pd

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The main project folder is one level up from the 'src' folder
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")

# SPECIFY THE FILE IN THE MAIN FOLDER
INPUT_FILENAME = "Combined_50_sequences.fasta"
INPUT_FILEPATH = os.path.join(PROJECT_ROOT, INPUT_FILENAME)

# Define output file paths (will be created in the main project folder)
PROCESSED_FASTA_OUTPUT = os.path.join(PROJECT_ROOT, "training_data_processed.fasta")
LABELS_CSV_OUTPUT = os.path.join(PROJECT_ROOT, "training_labels_processed.csv")

# --- Main Script Logic ---
def create_training_files_from_single_source():
    all_records = []
    label_data = []

    print(f"--- Starting Data Preparation for: {INPUT_FILENAME} ---")
    
    if not os.path.exists(INPUT_FILEPATH):
        print(f"[ERROR] File not found: {INPUT_FILEPATH}. Please make sure it's in your main project folder.")
        return

    records_in_file_count = 0
    labels_found = set()

    for record in SeqIO.parse(INPUT_FILEPATH, "fasta"):
        all_records.append(record)
        
        label = "unknown"
        if "class=" in record.description:
            try:
                label = record.description.split("class=")[1].split(" ")[0].strip()
            except IndexError:
                pass
        
        labels_found.add(label)
        label_data.append({"Sequence_ID": record.id, "Label": label})
        records_in_file_count += 1
    
    print(f"[SUCCESS] Processed {records_in_file_count} records.")
    print(f"Labels detected: {list(labels_found)}")

    if not all_records:
        print("[ERROR] No sequences were processed.")
        return

    with open(PROCESSED_FASTA_OUTPUT, "w") as output_handle:
        SeqIO.write(all_records, output_handle, "fasta")
    print(f"\nSuccessfully created processed FASTA file.")
    print(f" -> Saved to: {PROCESSED_FASTA_OUTPUT}")

    labels_df = pd.DataFrame(label_data)
    labels_df.to_csv(LABELS_CSV_OUTPUT, index=False)
    print(f"Successfully created labels CSV file.")
    print(f" -> Saved to: {LABELS_CSV_OUTPUT}")
    
    print("\n--- Data Preparation Complete! ---")

if __name__ == "__main__":
    create_training_files_from_single_source()