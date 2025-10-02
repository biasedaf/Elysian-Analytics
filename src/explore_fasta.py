import os
from Bio import SeqIO
import pandas as pd
from collections import Counter

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
FILENAME_TO_CHECK = "Combined_50_sequences.fasta"
FILEPATH_TO_CHECK = os.path.join(PROJECT_ROOT, FILENAME_TO_CHECK)

# --- Main Script ---
def analyze_fasta_content():
    print(f"--- Analyzing content of: {FILENAME_TO_CHECK} ---")

    if not os.path.exists(FILEPATH_TO_CHECK):
        print(f"[ERROR] File not found at '{FILEPATH_TO_CHECK}'.")
        return

    labels = []
    total_sequences = 0

    for record in SeqIO.parse(FILEPATH_TO_CHECK, "fasta"):
        total_sequences += 1
        label = "unknown"
        if "class=" in record.description:
            try:
                label = record.description.split("class=")[1].split(" ")[0].strip()
            except IndexError: pass

        labels.append(label)

    print(f"\nTotal sequences found in file: {total_sequences}")

    print("\n--- Class Distribution ---")
    label_counts = Counter(labels)

    distribution_df = pd.DataFrame(label_counts.items(), columns=['Label', 'Count'])
    distribution_df = distribution_df.sort_values(by='Count', ascending=False)
    distribution_df['Percentage'] = (distribution_df['Count'] / total_sequences * 100).round(2)

    print(distribution_df.to_string(index=False))

if __name__ == "__main__":
    analyze_fasta_content()