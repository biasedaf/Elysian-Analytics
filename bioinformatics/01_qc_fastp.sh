#!/bin/bash
fastp -i data/raw/sample_R1.fastq.gz -I data/raw/sample_R2.fastq.gz 
      -o data/raw/clean_R1.fastq.gz -O data/raw/clean_R2.fastq.gz 
      --html fastp_report.html --json fastp_report.json 
      --detect_adapter_for_pe
