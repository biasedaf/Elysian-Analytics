import pandas as pd
from Bio import SeqIO
import re

# --- Configuration ---
input_report_csv = "large_dataset_report.csv"
input_fasta = "new_ASVs.fasta"
output_labeled_csv = "deep_sea_labeled.csv"

# --- Helper function to extract a simple taxonomic label ---
def get_taxonomy_label(description):
    """Parses a long NCBI description to find a high-level taxonomic group."""
    description = str(description).lower()
    
    # Define a priority list of specific keywords to search for
    taxonomy_keywords = [
        'bacterium', 'bacillus', 'staphylococcus', 'escherichia',
        'dinoflagellate', 'eukaryote', 'chytridiomycota', 'metazoa', 
        'chloroplastida', 'cercozoa', 'apusazoa', 'rhizaria'
    ]
    
    for keyword in taxonomy_keywords:
        if keyword in description:
            return keyword.capitalize()

    # Fallback if no specific keyword is found
    return "Unknown"

print("Starting dataset preparation...")

# 1. Load the BLAST report
try:
    report_df = pd.read_csv(input_report_csv)
except FileNotFoundError:
    print(f"ERROR: '{input_report_csv}' not found. Please run generate_report.py first.")
    exit()

# 2. Load the FASTA file into a dictionary
try:
    sequences = {record.id: str(record.seq) for record in SeqIO.parse(input_fasta, "fasta")}
except FileNotFoundError:
    print(f"ERROR: '{input_fasta}' not found. Please run cluster_fasta.py first.")
    exit()

# 3. Create the 'taxonomy' column
print("Generating taxonomic labels from BLAST report...")
report_df['taxonomy'] = report_df['sseqid'].apply(get_taxonomy_label)

# 4. Create the final labeled dataset
print("Matching sequences to labels...")
final_data = []
for index, row in report_df.iterrows():
    asv_id = row['qseqid']
    if asv_id in sequences:
        final_data.append({
            'sequence': sequences[asv_id],
            'taxonomy': row['taxonomy']
        })

final_df = pd.DataFrame(final_data)

# <<< REFINEMENT ADDED HERE >>>
# 5. Filter out any rows with 'Unknown' or non-useful labels
print("Cleaning up labels...")
initial_count = len(final_df)
final_df = final_df[final_df['taxonomy'] != 'Unknown']
# You can add more terms to filter out if needed
final_df = final_df[~final_df['taxonomy'].str.contains("Gi")]

print(f"Removed {initial_count - len(final_df)} rows with non-specific labels.")

# 6. Save the final, clean dataset
final_df.to_csv(output_labeled_csv, index=False)

print(f"\nSuccess! Clean labeled dataset with {len(final_df)} entries saved to '{output_labeled_csv}'")