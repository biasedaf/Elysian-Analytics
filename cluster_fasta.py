from Bio import SeqIO
from collections import defaultdict

# --- Configuration ---
input_fasta = "Combined_50_sequences.fasta" 
output_fasta = "new_ASVs.fasta"

print(f"Reading sequences from {input_fasta}...")

# Removing duplicates and add elements after reading from the input FASTA
unique_sequences = set()

for record in SeqIO.parse(input_fasta, "fasta"):
    unique_sequences.add(str(record.seq))

print(f"Found {len(unique_sequences)} unique sequences.")

# Write the unique sequences to the output FASTA file
with open(output_fasta, "w") as output_handle:
    for i, seq in enumerate(unique_sequences):
        # Write in FASTA format with a new, simple ID
        output_handle.write(f">ASV_{i+1}\n{seq}\n")

print(f"Successfully wrote {len(unique_sequences)} unique sequences to {output_fasta}")