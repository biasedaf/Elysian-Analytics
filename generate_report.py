from Bio.Blast import NCBIWWW
import time

# --- CONFIGURATION ---
INPUT_FASTA_FILE = "Combined_50_sequences.fasta"
OUTPUT_XML_FILE = "blast_report.xml"  # This is the correct filename and extension
# ---

print(f"Reading sequences from '{INPUT_FASTA_FILE}'...")
try:
    with open(INPUT_FASTA_FILE, "r") as f:
        fasta_string = f.read()
except FileNotFoundError:
    print(f"[ERROR] Input file '{INPUT_FASTA_FILE}' not found.")
    exit()

print("Starting live BLAST search against NCBI 'nt' database...")
print("!!! THIS WILL TAKE A VERY LONG TIME. PLEASE BE PATIENT !!!")
start_time = time.time()

try:
    # Perform the BLAST search
    result_handle = NCBIWWW.qblast(
        program="blastn",
        database="nt",
        sequence=fasta_string,
        entrez_query="eukaryotes[organism]"
    )

    # Save the raw XML output to the correct file
    with open(OUTPUT_XML_FILE, "w") as out_file:
        out_file.write(result_handle.read())

    result_handle.close()

    end_time = time.time()
    print(f"\n[SUCCESS] BLAST search complete.")
    print(f"Raw XML report saved to '{OUTPUT_XML_FILE}'") # It will now say blast_report.xml
    print(f"Total time taken: { (end_time - start_time) / 60:.2f} minutes")

except Exception as e:
    print(f"\n[ERROR] An error occurred during the BLAST search: {e}")