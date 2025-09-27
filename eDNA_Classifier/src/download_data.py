# src/download_data.py
from Bio import Entrez
from Bio import SeqIO
import os

# --- IMPORTANT: SET YOUR EMAIL ADDRESS ---
# NCBI requires an email address for API access
Entrez.email = "112315177@cse.iiitp.ac.in"

# --- CONFIGURATION ---
# Define the search queries and output filenames
SEARCH_QUERIES = {
    "metazoa": '"Metazoa"[Organism] AND 200:500[Sequence Length]',
    "fungi": '"Fungi"[Organism] AND 18S ribosomal RNA[Gene Name] AND 100:1000[Sequence Length]',
    "diatoms": '"Bacillariophyta"[Organism] AND 200:500[Sequence Length]'
}
NUM_RECORDS_TO_FETCH = 200
OUTPUT_DIR = "../data/raw/"

# --- SCRIPT ---
def fetch_sequence_data():
    """
    Searches NCBI Nucleotide database, fetches records, and saves them to FASTA files.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    for name, query in SEARCH_QUERIES.items():
        output_filename = os.path.join(OUTPUT_DIR, f"{name}_data.fasta")
        print(f"\nSearching for '{name}' with query: {query}")

        try:
            # Step 1: Search NCBI and get the record IDs
            handle = Entrez.esearch(db="nucleotide", term=query, retmax=NUM_RECORDS_TO_FETCH)
            record = Entrez.read(handle)
            handle.close()
            id_list = record["IdList"]

            if not id_list:
                print(f"Warning: No records found for '{name}'. Skipping.")
                continue

            print(f"Found {len(id_list)} records. Fetching...")

            # Step 2: Fetch the actual records using the IDs
            handle = Entrez.efetch(db="nucleotide", id=id_list, rettype="fasta", retmode="text")
            fasta_data = handle.read()
            handle.close()

            # Step 3: Save the records to a file
            with open(output_filename, "w") as f:
                f.write(fasta_data)

            print(f"Successfully saved {len(id_list)} records to {output_filename}")

        except Exception as e:
            print(f"An error occurred while fetching data for '{name}': {e}")

if __name__ == "__main__":
    fetch_sequence_data()