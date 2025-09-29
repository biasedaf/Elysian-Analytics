import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import tensorflow as tf
from tensorflow import keras
import os
import pickle
from collections import defaultdict
import plotly.express as px
from Bio.Blast import NCBIWWW, NCBIXML

# --- Page Config ---
st.set_page_config(
    page_title="Elysian Analytics",
    page_icon="🧬",
    layout="wide"
)

# --- All Helper Functions ---
def parse_fasta(file_content):
    sequences = {}
    current_id, current_seq = None, []
    for line in file_content.strip().split('\n'):
        line = line.strip()
        if line.startswith('>'):
            if current_id is not None:
                sequences[current_id] = ''.join(current_seq)
            current_id, current_seq = line[1:], []
        else:
            current_seq.append(line.upper())
    if current_id is not None:
        sequences[current_id] = ''.join(current_seq)
    return sequences

def one_hot_encode_sequence(sequence):
    nucleotide_map = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1]}
    return np.array([nucleotide_map.get(n, [0,0,0,0]) for n in sequence])

def pad_sequence(encoded_seq, target_length=423):
    current_length = len(encoded_seq)
    if current_length < target_length:
        padding = np.zeros((target_length - current_length, 4))
        return np.vstack([encoded_seq, padding])
    return encoded_seq[:target_length] if current_length > target_length else encoded_seq

def preprocess_for_dl(sequences):
    processed_sequences, sequence_ids = [], []
    for seq_id, sequence in sequences.items():
        processed_sequences.append(pad_sequence(one_hot_encode_sequence(sequence)))
        sequence_ids.append(seq_id)
    return np.array(processed_sequences), sequence_ids

def get_kmer_features_for_prediction(sequences, k=6):
    all_kmer_counts, sequence_ids = [], list(sequences.keys())
    for seq_id in sequence_ids:
        seq, kmer_counts = sequences[seq_id], defaultdict(int)
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if "N" not in kmer: kmer_counts[kmer] += 1
        all_kmer_counts.append(kmer_counts)
    return pd.DataFrame(all_kmer_counts).fillna(0), sequence_ids

@st.cache_data
def generate_live_novelty_report(_sequences_dict):
    fasta_string = "".join([f">{seq_id}\n{seq}\n" for seq_id, seq in _sequences_dict.items()])
    result_handle = NCBIWWW.qblast(program="blastn", database="nt", sequence=fasta_string)
    blast_records = NCBIXML.parse(result_handle)
    
    blast_results = []
    for record in blast_records:
        query_id = record.query.split(" ")[0]
        if not record.alignments:
            blast_results.append({'ASV ID': query_id, 'Percent Identity': 0, 'Best Match Found in Database': "No Match Found", 'Novelty Flag': True})
            continue
        
        top_alignment = record.alignments[0]
        top_hsp = top_alignment.hsps[0]
        percent_identity = (top_hsp.identities / top_hsp.align_length) * 100
        is_novel = percent_identity < 90.0
        blast_results.append({'ASV ID': query_id, 'Percent Identity': round(percent_identity, 2), 'Best Match Found in Database': top_alignment.title, 'Novelty Flag': is_novel})
        
    return pd.DataFrame(blast_results)

# --- Model Loading and Prediction ---
@st.cache_resource
def load_dl_model():
    model_path = "dl_model.h5"
    if os.path.exists(model_path):
        try: return keras.models.load_model(model_path)
        except Exception as e: st.error(f"Error loading DL model: {e}")
    else: st.error(f"Model file '{model_path}' not found.")
    return None

@st.cache_resource
def load_rf_model():
    model_path = os.path.join("models", "random_forest_baseline.pkl")
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as file: return pickle.load(file)
        except Exception as e: st.error(f"Error loading RF model: {e}")
    else: st.error(f"Model file '{model_path}' not found.")
    return None

def predict_with_dl_model(model, sequences):
    try:
        predictions = model.predict(sequences)
        return ["Rare" if np.argmax(pred) == 1 else "Common" for pred in predictions]
    except Exception as e: st.error(f"Error during DL prediction: {e}")
    return None

def predict_with_rf_model(model, feature_df):
    try:
        model_columns = model.feature_names_in_
        aligned_df = feature_df.reindex(columns=model_columns, fill_value=0)
        predictions = model.predict(aligned_df)
        return ["Rare" if pred == 1 else "Common" for pred in predictions]
    except Exception as e: st.error(f"Error during RF prediction: {e}")
    return None

@st.cache_data
def load_precomputed_novelty_report():
    report_path = "novelty_report.csv"
    if os.path.exists(report_path):
        try: 
            df = pd.read_csv(report_path)
            df.rename(columns={'qseqid': 'ASV ID', 'pident': 'Percent Identity', 'sseqid': 'Best Match Found in Database', 'novelty_flag': 'Novelty Flag'}, inplace=True)
            return df
        except Exception as e: st.error(f"Error loading novelty report: {e}")
    else: st.warning(f"Novelty report file '{report_path}' not found.")
    return None

# --- Main Application ---
def main():
    if 'analysis_run' not in st.session_state:
        st.session_state.analysis_run = False

    # --- Sidebar ---
    with st.sidebar:
        st.image("assets/your_logo.png", width=100) 
        st.markdown("---")
        st.subheader("📝 About this Project")
        st.info("This prototype uses AI to analyze environmental DNA (eDNA), demonstrating a complete pipeline from raw data to an interactive web application.")
        st.markdown("---")
        st.subheader("📁 Upload Your FASTA File")
        uploaded_file = st.file_uploader("Upload a FASTA file", type=['fasta', 'fa', 'fna'], label_visibility="collapsed")
        st.subheader("🤖 Select AI Model")
        model_choice = st.selectbox("Choose analysis model:", ["Deep Learning (CNN)", "Baseline (Random Forest)"], label_visibility="collapsed")
        
        if st.button("Analyze Sequences", type="primary", use_container_width=True):
            if uploaded_file is not None:
                st.session_state.analysis_run = True
                st.session_state.uploaded_file = uploaded_file
                st.session_state.model_choice = model_choice
            else:
                st.error("⚠️ Please upload a FASTA file first.")
                st.session_state.analysis_run = False

    # --- Main Panel ---
    # <<< NEW SECTION IMPLEMENTED HERE >>>
    col1, col2 = st.columns([1, 10]) # column ratio for better alignment
    with col1:
        st.image("assets/your_logo.png", width=100) 
    with col2:
        st.title("Elysian Analytics : An AI-Powered eDNA Sequence Analyzer")

    st.markdown("""
    This application uses artificial intelligence to analyze environmental DNA (eDNA) sequences. 
    Upload a FASTA file, select an AI model, and get predictions about your data.
    """)

    with st.expander("📖 How to Use This App"):
        st.write("1. **Upload a FASTA file** containing your eDNA sequences using the file uploader in the sidebar.")
        st.write("2. **Select an AI model** from the dropdown menu in the sidebar.")
        st.write("3. **Click 'Analyze Sequences'** to start the analysis and view the results.")
        st.write("4. Explore the **Pre-computed Novelty Analysis** for existing samples, including filtering and summary charts.")
    # <<< END OF NEW SECTION >>>


    if not st.session_state.analysis_run:
        st.markdown("---")
        st.subheader("📈 Pre-computed Novelty Report")
        st.info(
            """
            The Pre-computed Novelty Report below shows the analysis of the original dataset used to train the models.
            - **High Percent Identity (>97%)**: The sequence is from a known or very closely related species.
            - **Low Percent Identity (<90%)**: The sequence is significantly different from anything known and is flagged as a potential new discovery.
            """
        )
        novelty_df = load_precomputed_novelty_report()
        if novelty_df is not None:
            st.dataframe(novelty_df, use_container_width=True)
            
    else:
        file_content = StringIO(st.session_state.uploaded_file.getvalue().decode("utf-8")).read()
        sequences_dict = parse_fasta(file_content)
        
        if not sequences_dict:
            st.error("No valid sequences found in the uploaded file.")
            st.session_state.analysis_run = False
        else:
            st.success(f"✅ Loaded {len(sequences_dict)} sequences successfully.")
            
            tab1, tab2 = st.tabs(["📊 AI Prediction", "📈 Live Novelty Report"])

            with tab1:
                with st.spinner("Running AI model..."):
                    predictions, sequence_ids = None, None
                    if st.session_state.model_choice == "Deep Learning (CNN)":
                        model = load_dl_model()
                        if model:
                            processed_sequences, sequence_ids = preprocess_for_dl(sequences_dict)
                            predictions = predict_with_dl_model(model, processed_sequences)
                    elif st.session_state.model_choice == "Baseline (Random Forest)":
                        model = load_rf_model()
                        if model:
                            feature_df, sequence_ids = get_kmer_features_for_prediction(sequences_dict)
                            predictions = predict_with_rf_model(model, feature_df)
                    
                    if predictions:
                        st.subheader(f"Live Analysis Results from {st.session_state.model_choice}")
                        results_df = pd.DataFrame({'Sequence ID': sequence_ids, 'Predicted Class': predictions})
                        common_count = predictions.count("Common")
                        rare_count = len(predictions) - common_count
                        
                        mcol1, mcol2, mcol3 = st.columns(3)
                        mcol1.metric("Total Sequences", len(predictions))
                        mcol2.metric("Predicted Common", common_count)
                        mcol3.metric("Predicted Rare", rare_count)
                        
                        chart_data = pd.DataFrame({'Class': ['Common', 'Rare'], 'Count': [common_count, rare_count]})
                        fig = px.pie(chart_data, values='Count', names='Class', title='Prediction Summary')
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(results_df, use_container_width=True)

            with tab2:
                st.warning("Performing live BLAST search against NCBI... This may take several minutes.")
                live_novelty_df = generate_live_novelty_report(sequences_dict)
                st.success("Live BLAST search complete!")
                st.dataframe(live_novelty_df, use_container_width=True)

if __name__ == "__main__":
    main()