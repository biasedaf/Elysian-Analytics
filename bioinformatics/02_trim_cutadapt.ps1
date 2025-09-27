# PowerShell script for primer trimming with cutadapt
# Replace FORWARD_PRIMER and REVERSE_PRIMER with your actual primer sequences

Write-Host "Running cutadapt for primer trimming..."

$R1_in  = "data/raw/clean_R1.fastq.gz"
$R2_in  = "data/raw/clean_R2.fastq.gz"
$R1_out = "data/raw/trimmed_R1.fastq.gz"
$R2_out = "data/raw/trimmed_R2.fastq.gz"

$forwardPrimer = "FORWARD_PRIMER"
$reversePrimer = "REVERSE_PRIMER"

cutadapt -g "^$forwardPrimer" -G "^$reversePrimer" `
         -o $R1_out -p $R2_out `
         $R1_in $R2_in

if ($LASTEXITCODE -eq 0) {
    Write-Host "cutadapt trimming complete. Trimmed reads saved."
} else {
    Write-Error "cutadapt failed."
}
