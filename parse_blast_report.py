# parse_blast_report.py
from Bio.Blast import NCBIXML
from Bio import SeqIO
import pandas as pd
import re

# --- CONFIGURATION ---
INPUT_XML_FILE = "blast_report.xml"
INPUT_FASTA_FILE = "Combined_50_sequences.fasta"
OUTPUT_CSV_FILE = "deep_sea_labeled.csv"
# ---

def get_best_taxonomy(title):
    """Parses the BLAST title to find the most specific taxonomic label available."""
    # Try to find a two-part species name (e.g., "Gadus morhua")
    match = re.search(r'([A-Z][a-z]+ [a-z]+)', title)
    if match:
        return match.group(1).strip()
    
    # If no species, try to find the Genus
    match = re.search(r'([A-Z][a-z]+) sp\.', title)
    if match:
        return match.group(1).strip()

    # If not, look for common higher-level taxa from a predefined list
    common_taxa = ["Chytridiomycota", "Dinoflagellate", "Metazoa", "Fungi", "Viridiplantae"]
    for taxon in common_taxa:
        if taxon in title:
            return taxon.strip()
            
    # As a last resort, return a general label
    return "Eukaryote"

print(f"Reading BLAST report from '{INPUT_XML_FILE}'...")

try:
    # Get the original sequences back from the FASTA file to map them
    sequences = {rec.id: str(rec.seq) for rec in SeqIO.parse(INPUT_FASTA_FILE, "fasta")}
    
    labeled_data = []
    with open(INPUT_XML_FILE) as result_handle:
        blast_records = NCBIXML.parse(result_handle)
        
        for record in blast_records:
            query_id = record.query.split(" ")[0]
            sequence = sequences.get(query_id)
            
            # Only proceed if we have a sequence and it had at least one match
            if sequence and record.alignments:
                top_alignment = record.alignments[0]
                taxonomy = get_best_taxonomy(top_alignment.title)
                labeled_data.append({'sequence': sequence, 'taxonomy': taxonomy})

    df = pd.DataFrame(labeled_data)

    if not df.empty:
        # Remove duplicate sequences to keep the dataset clean
        df = df.drop_duplicates(subset=['sequence'])
        
        print(f"\n[SUCCESS] Successfully parsed the report.")
        print(f"Created a labeled dataset with {df.shape[0]} unique sequences.")
        print("\nClass distribution in the new file:")
        print(df['taxonomy'].value_counts())
        
        df.to_csv(OUTPUT_CSV_FILE, index=False)
        print(f"\nLabeled dataset saved to '{OUTPUT_CSV_FILE}'")
    else:
        print("\n[WARNING] No labeled data could be generated. This might mean no BLAST matches were found.")

except FileNotFoundError:
    print(f"[ERROR] Input file '{INPUT_XML_FILE}' not found. Please run 'generate_report.py' first.")
except Exception as e:
    print(f"\n[ERROR] An error occurred during parsing: {e}")