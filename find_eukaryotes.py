from Bio.Blast import NCBIXML
from Bio import SeqIO
import pandas as pd

# --- CONFIGURATION ---
INPUT_XML_FILE = "blast_report.xml"
INPUT_FASTA_FILE = "Combined_50_sequences.fasta"
# ---

print(f"Searching for known Eukaryotes in '{INPUT_XML_FILE}'...")

try:
    # Get the original sequences from the FASTA file
    sequences = {rec.id: str(rec.seq) for rec in SeqIO.parse(INPUT_FASTA_FILE, "fasta")}
    
    eukaryote_sequences = []
    with open(INPUT_XML_FILE) as result_handle:
        blast_records = NCBIXML.parse(result_handle)
        
        for record in blast_records:
            query_id = record.query.split(" ")[0]
            sequence = sequences.get(query_id)
            
            if sequence and record.alignments:
                top_alignment = record.alignments[0]
                title = top_alignment.title
                
                # We will define a "known Eukaryote" as anything that has a match
                # but isn't one of our other specific target classes.
                if "Chytridiomycota" not in title and "Dinoflagellate" not in title:
                    eukaryote_sequences.append(sequence)

    if eukaryote_sequences:
        # Remove duplicates to get a unique list
        unique_eukaryotes = list(set(eukaryote_sequences))
        
        print(f"\n[SUCCESS] Found {len(unique_eukaryotes)} unique Eukaryote sequences.")
        print("You can copy and paste these into your 'deep_sea_labeled.csv'.")
        print("-" * 30)
        
        # Print the first 20 found
        for i, seq in enumerate(unique_eukaryotes[:20]):
            print(f"\n# Eukaryote Sequence {i+1}")
            print(seq)
            
        print("-" * 30)

    else:
        print("\n[WARNING] No Eukaryote sequences could be found. The BLAST report might be empty or contain no matches.")

except FileNotFoundError:
    print(f"[ERROR] Input file '{INPUT_XML_FILE}' not found. Please run 'generate_report.py' first.")
except Exception as e:
    print(f"\n[ERROR] An error occurred during parsing: {e}")