# PowerShell script for quality control with fastp

Write-Host "Running fastp QC..."

# Define input/output files
$R1_in  = "data/raw/sample_R1.fastq.gz"
$R2_in  = "data/raw/sample_R2.fastq.gz"
$R1_out = "data/raw/clean_R1.fastq.gz"
$R2_out = "data/raw/clean_R2.fastq.gz"

# Run fastp
fastp -i $R1_in -I $R2_in `
      -o $R1_out -O $R2_out `
      --html fastp_report.html --json fastp_report.json `
      --detect_adapter_for_pe

if ($LASTEXITCODE -eq 0) {
    Write-Host "fastp QC complete. Cleaned reads saved."
} else {
    Write-Error "fastp failed."
}
