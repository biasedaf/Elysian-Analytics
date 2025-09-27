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
