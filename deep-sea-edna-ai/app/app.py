import streamlit as st
from Bio import SeqIO

st.title("ðŸŒŠ Deep-Sea eDNA AI Explorer")

uploaded_file = st.file_uploader("Upload ASVs FASTA", type=["fasta"])
if uploaded_file:
    records = list(SeqIO.parse(uploaded_file, "fasta"))
    st.success(f"Uploaded {len(records)} sequences")
    st.write("Example:", records[0].id, str(records[0].seq)[:50], "...")
if st.button("Analyze Sequences"):
    st.info("ðŸš€ Analysis pipeline coming soon...")
