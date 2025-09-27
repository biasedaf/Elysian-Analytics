# Load the DADA2 library
library(dada2)

# Set path to your fastq files
path <- getwd() 

# Sort forward and reverse files using the new file names
fnFs <- sort(list.files(path, pattern="_R1_001.fastq", full.names = TRUE))
fnRs <- sort(list.files(path, pattern="_R2_001.fastq", full.names = TRUE))
sample.names <- sapply(strsplit(basename(fnFs), "_"), `[`, 1)

# Create a place for filtered files
filtFs <- file.path(path, "filtered", paste0(sample.names, "_F_filt.fastq.gz"))
filtRs <- file.path(path, "filtered", paste0(sample.names, "_R_filt.fastq.gz"))

if(!dir.exists("filtered")) dir.create("filtered")

# Filter and trim. The typo in 'multithread' has been fixed.
# <<< TYPO FIXED HERE
out <- filterAndTrim(fnFs, filtFs, fnRs, filtRs,
                     maxN=0, maxEE=c(2,2), truncQ=2, rm.phix=TRUE,
                     compress=TRUE, multithread=FALSE)

# Learn error rates
errF <- learnErrors(filtFs, multithread=FALSE)
errR <- learnErrors(filtRs, multithread=FALSE)

# Core DADA algorithm
dadaFs <- dada(filtFs, err=errF, multithread=FALSE)
dadaRs <- dada(filtRs, err=errR, multithread=FALSE)

# Merge, create table, and remove chimeras
mergers <- mergePairs(dadaFs, filtFs, dadaRs, filtRs, verbose=TRUE)
seqtab <- makeSequenceTable(mergers)
seqtab.nochim <- removeBimeraDenovo(seqtab, method="consensus", multithread=FALSE, verbose=TRUE)

# --- EXPORT THE RESULTS ---
write.csv(t(seqtab.nochim), "ASV_table.csv") 

asv_seqs <- colnames(seqtab.nochim)
asv_headers <- paste(">ASV", 1:length(asv_seqs), sep="_")
asv_fasta <- c(rbind(asv_headers, asv_seqs))
write(asv_fasta, "ASVs.fasta")

print("DADA2 processing complete!")