import os
from Bio import Entrez
from Bio import SeqIO

# --- IMPORTANT: SET YOUR EMAIL ADDRESS ---
Entrez.email = "112315177@cse.iitp.ac.in" # Replace with your email

# --- CONFIGURATION ---
# This part is new: it builds a correct, absolute path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

SEARCH_QUERIES = {
    "metazoa": '"Metazoa"[Organism] AND 200:500[Sequence Length]',
    "fungi": '"Fungi"[Organism] AND 18S ribosomal RNA[Gene Name] AND 100:1000[Sequence Length]',
    "diatoms": '"Bacillariophyta"[Organism] AND 200:500[Sequence Length]'
}
NUM_RECORDS_TO_FETCH = 200

# --- SCRIPT (No changes needed below here) ---
def fetch_sequence_data():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    for name, query in SEARCH_QUERIES.items():
        output_filename = os.path.join(OUTPUT_DIR, f"{name}_data.fasta")
        print(f"\nSearching for '{name}'...")
        try:
            handle = Entrez.esearch(db="nucleotide", term=query, retmax=NUM_RECORDS_TO_FETCH)
            record = Entrez.read(handle)
            handle.close()
            id_list = record["IdList"]

            if not id_list:
                print(f"Warning: No records found for '{name}'.")
                continue

            print(f"Found {len(id_list)} records. Fetching and saving to {output_filename}")
            handle = Entrez.efetch(db="nucleotide", id=id_list, rettype="fasta", retmode="text")
            fasta_data = handle.read()
            handle.close()

            with open(output_filename, "w") as f:
                f.write(fasta_data)
            print("Successfully saved records.")

        except Exception as e:
            print(f"An error occurred for '{name}': {e}")

if __name__ == "__main__":
    fetch_sequence_data()