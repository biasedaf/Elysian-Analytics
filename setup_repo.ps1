# === Create repo structure ===
New-Item -ItemType Directory -Force -Path deep-sea-edna-ai/bioinformatics
New-Item -ItemType Directory -Force -Path deep-sea-edna-ai/ml
New-Item -ItemType Directory -Force -Path deep-sea-edna-ai/app
New-Item -ItemType Directory -Force -Path deep-sea-edna-ai/data/raw
New-Item -ItemType Directory -Force -Path deep-sea-edna-ai/data/refs
New-Item -ItemType Directory -Force -Path deep-sea-edna-ai/docs

# === Bioinformatics scripts ===
@"
#!/bin/bash
# Run quality control with fastp
fastp -i data/raw/sample_R1.fastq.gz -I data/raw/sample_R2.fastq.gz `
      -o data/raw/clean_R1.fastq.gz -O data/raw/clean_R2.fastq.gz `
      --html fastp_report.html --json fastp_report.json `
      --detect_adapter_for_pe
"@ | Out-File deep-sea-edna-ai/bioinformatics/01_qc_fastp.sh -Encoding ascii

@"
#!/bin/bash
# Run primer trimming with cutadapt (replace PRIMERS with actual)
cutadapt -g ^FORWARD_PRIMER -G ^REVERSE_PRIMER `
         -o data/raw/trimmed_R1.fastq.gz -p data/raw/trimmed_R2.fastq.gz `
         data/raw/clean_R1.fastq.gz data/raw/clean_R2.fastq.gz
"@ | Out-File deep-sea-edna-ai/bioinformatics/02_trim_cutadapt.sh -Encoding ascii

@"
library(dada2)
fnFs <- "data/raw/trimmed_R1.fastq.gz"
fnRs <- "data/raw/trimmed_R2.fastq.gz"
filtFs <- "data/raw/filt_R1.fastq.gz"
filtRs <- "data/raw/filt_R2.fastq.gz"
out <- filterAndTrim(fnFs, filtFs, fnRs, filtRs,
                     truncLen=c(240,160), maxN=0, maxEE=c(2,2),
                     truncQ=2, compress=TRUE)
errF <- learnErrors(filtFs, multithread=TRUE)
errR <- learnErrors(filtRs, multithread=TRUE)
derepFs <- derepFastq(filtFs)
derepRs <- derepFastq(filtRs)
dadaFs <- dada(derepFs, err=errF, multithread=TRUE)
dadaRs <- dada(derepRs, err=errR, multithread=TRUE)
mergers <- mergePairs(dadaFs, derepFs, dadaRs, derepRs)
seqtab <- makeSequenceTable(mergers)
seqtab.nochim <- removeBimeraDenovo(seqtab, method="consensus", multithread=TRUE)
write.csv(seqtab.nochim, "bioinformatics/ASV_table.csv")
uniquesToFasta(seqtab.nochim, fout="bioinformatics/ASVs.fasta")
"@ | Out-File deep-sea-edna-ai/bioinformatics/03_dada2_pipeline.R -Encoding utf8

# === ML scripts ===
@"
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
"@ | Out-File deep-sea-edna-ai/ml/embeddings.py -Encoding utf8

@"
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from embeddings import fasta_to_matrix

# Load embeddings
X, ids = fasta_to_matrix("../bioinformatics/ASVs.fasta", k=6)

# Fake labels for prototype (replace with taxonomy later)
y = np.random.choice(["taxon1", "taxon2", "taxon3"], size=len(X))

# Train RF
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)

# Save model
with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)
print("Saved RandomForest model -> rf_model.pkl")
"@ | Out-File deep-sea-edna-ai/ml/random_forest.py -Encoding utf8

# === App scripts ===
@"
import streamlit as st
from Bio import SeqIO

st.title("🌊 Deep-Sea eDNA AI Explorer")

uploaded_file = st.file_uploader("Upload ASVs FASTA", type=["fasta"])
if uploaded_file:
    records = list(SeqIO.parse(uploaded_file, "fasta"))
    st.success(f"Uploaded {len(records)} sequences")
    st.write("Example:", records[0].id, str(records[0].seq)[:50], "...")
if st.button("Analyze Sequences"):
    st.info("🚀 Analysis pipeline coming soon...")
"@ | Out-File deep-sea-edna-ai/app/app.py -Encoding utf8

# === Docs & README ===
@"
# Deep-Sea eDNA AI Explorer 🌊

## Structure
- `bioinformatics/`: QC + DADA2 pipeline → ASV_table.csv, ASVs.fasta
- `ml/`: ML & DL models → embeddings, RandomForest, CNN
- `app/`: Streamlit frontend
- `data/`: raw FASTQs + reference DBs
- `docs/`: slides, figures, pipeline diagrams

## Quickstart
```bash
conda create -n eDNA python=3.9 -y
conda activate eDNA
pip install -r requirements.txt
