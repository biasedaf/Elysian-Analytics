from Bio import SeqIO
import numpy as np

def kmer_freq(seq, k=6):
    freqs = {}
    for i in range(len(seq)-k+1):
        kmer = seq[i:i+k]
        freqs[kmer] = freqs.get(kmer, 0) + 1
    return freqs

def fasta_to_matrix(fasta_file, k=6):
    kmers = []
    ids = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        ids.append(record.id)
        freqs = kmer_freq(str(record.seq), k)
        kmers.append([freqs.get(kmer, 0) for kmer in sorted(freqs.keys())])
    return np.array(kmers), ids

if __name__ == "__main__":
    X, ids = fasta_to_matrix("../bioinformatics/ASVs.fasta", k=6)
    print(f"Generated embeddings for {len(ids)} sequences")
