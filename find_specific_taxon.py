# find_specific_taxon.py
from Bio.Blast import NCBIXML
from Bio import SeqIO
import sys # Import sys to read command-line arguments

# --- CONFIGURATION ---
INPUT_XML_FILE = "blast_report.xml"
INPUT_FASTA_FILE = "Combined_50_sequences.fasta"
# ---

# Check if a taxon name was provided
if len(sys.argv) < 2:
    print("\n[ERROR] Please provide a taxon name to search for.")
    print("Example Usage: python find_specific_taxon.py Chytridiomycota")
    sys.exit(1) # Exit the script

# Get the taxon name from the command line
TARGET_TAXON = sys.argv[1]
print(f"Searching for sequences matching '{TARGET_TAXON}' in '{INPUT_XML_FILE}'...")

try:
    # Get the original sequences from the FASTA file
    sequences = {rec.id: str(rec.seq) for rec in SeqIO.parse(INPUT_FASTA_FILE, "fasta")}
    
    found_sequences = []
    with open(INPUT_XML_FILE) as result_handle:
        blast_records = NCBIXML.parse(result_handle)
        
        for record in blast_records:
            query_id = record.query.split(" ")[0]
            sequence = sequences.get(query_id)
            
            if sequence and record.alignments:
                top_alignment = record.alignments[0]
                title = top_alignment.title
                
                # Check if the target taxon is in the title of the top BLAST hit
                if TARGET_TAXON.lower() in title.lower():
                    found_sequences.append(sequence)

    if found_sequences:
        # Remove duplicates to get a unique list
        unique_sequences = list(set(found_sequences))
        
        print(f"\n[SUCCESS] Found {len(unique_sequences)} unique sequences for '{TARGET_TAXON}'.")
        print("You can copy and paste these into your 'deep_sea_labeled.csv'.")
        print("-" * 30)
        
        for i, seq in enumerate(unique_sequences):
            print(f"\n# {TARGET_TAXON} Sequence {i+1}")
            print(seq)
            
        print("-" * 30)

    else:
        print(f"\n[INFO] No sequences matching '{TARGET_TAXON}' could be found.")

except FileNotFoundError:
    print(f"[ERROR] Input file '{INPUT_XML_FILE}' not found. Please run 'generate_report.py' first.")
except Exception as e:
    print(f"\n[ERROR] An error occurred during parsing: {e}")