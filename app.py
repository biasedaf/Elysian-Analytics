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
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Elysian Analytics", page_icon="🧬", layout="wide")

# --- Configuration Constants ---

HIGH_IDENTITY_THRESHOLD = 90.0
# This should match the label used in your create_golden_dataset.py script
NOVEL_PATTERN_LABEL = 'Rhizoclosmatium sp.'

PRECOMPUTED_RESULTS_FILE = 'precomputed_novelty.csv' # Define the filename here
def parse_fasta(file_content):
    sequences = {}
    current_id, current_seq = None, []
    for line in file_content.strip().split('\n'):
        line = line.strip()
        if line.startswith('>'):
            if current_id is not None: sequences[current_id] = ''.join(current_seq)
            current_id, current_seq = line[1:], []
        else: current_seq.append(line.upper())
    if current_id is not None: sequences[current_id] = ''.join(current_seq)
    return sequences

def dereplicate_sequences(sequences_dict):
    unique_seqs = {}
    for seq in sequences_dict.values():
        if seq not in unique_seqs: unique_seqs[seq] = f"ASV_{len(unique_seqs) + 1}"
    return {v: k for k, v in unique_seqs.items()}

def one_hot_encode_sequence(sequence):
    nucleotide_map = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], 'N': [0,0,0,0]}
    return np.array([nucleotide_map.get(n, [0,0,0,0]) for n in sequence])

def pad_sequence(encoded_seq, target_length=282):
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
    try:
        result_handle = NCBIWWW.qblast(program="blastn", database="nt", sequence=fasta_string)
        blast_records = NCBIXML.parse(result_handle)
        blast_results = []
        for record in blast_records:
            query_id = record.query.split(" ")[0]
            if not record.alignments:
                blast_results.append({'ASV ID': query_id, 'Percent Identity': 0.0, 'Best Match Found in Database': "No Match Found"})
                continue
            top_alignment = record.alignments[0]
            top_hsp = record.alignments[0].hsps[0]
            percent_identity = (top_hsp.identities / top_hsp.align_length) * 100
            
            # **MODIFICATION**: Simplify the title before appending
            simplified_title = simplify_blast_title(top_alignment.title)
            
            blast_results.append({'ASV ID': query_id, 'Percent Identity': round(percent_identity, 2), 'Best Match Found in Database': simplified_title})
        return pd.DataFrame(blast_results)
    except Exception as e:
        st.error(f"BLAST search failed. Error: {e}")
        return pd.DataFrame()

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

@st.cache_resource
def load_xgb_model():
    model_path = os.path.join("models", "xgboost_model.pkl")
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as file: return pickle.load(file)
        except Exception as e: st.error(f"Error loading XGBoost model: {e}")
    else: st.error(f"Model file '{model_path}' not found.")
    return None

@st.cache_resource
def get_label_encoder():
    encoder_path = os.path.join("models", "label_encoder.pkl")
    try:
        with open(encoder_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error(f"Saved label encoder ('{encoder_path}') not found. Please retrain the models to generate it.")
        return None

def predict_with_dl_model(model, sequences, encoder):
    try:
        predictions_proba = model.predict(sequences)
        predictions_indices = np.argmax(predictions_proba, axis=1)
        predicted_classes = encoder.inverse_transform(predictions_indices)
        confidence_scores = [f"{np.max(p)*100:.2f}%" for p in predictions_proba]
        return predicted_classes, confidence_scores
    except Exception as e: st.error(f"Error during DL prediction: {e}")
    return None, None

def predict_with_rf_model(model, feature_df, encoder):
    try:
        model_columns = model.feature_names_in_
        aligned_df = feature_df.reindex(columns=model_columns, fill_value=0)
        predictions = model.predict(aligned_df[model_columns])
        predicted_classes = encoder.inverse_transform(predictions)
        confidence_scores = ["N/A"] * len(predicted_classes)
        return predicted_classes, confidence_scores
    except Exception as e: st.error(f"Error during RF prediction: {e}")
    return None, None

def predict_with_xgb_model(model, feature_df, encoder):
    try:
        model_columns = model.get_booster().feature_names
        aligned_df = feature_df.reindex(columns=model_columns, fill_value=0)
        predictions_proba = model.predict_proba(aligned_df[model_columns])
        predictions_indices = np.argmax(predictions_proba, axis=1)
        predicted_classes = encoder.inverse_transform(predictions_indices)
        confidence_scores = [f"{np.max(p)*100:.2f}%" for p in predictions_proba]
        return predicted_classes, confidence_scores
    except Exception as e: st.error(f"Error during XGBoost prediction: {e}")
    return None, None

@st.cache_data
def to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def get_remarks(row):
    ai_taxon = row['AI Predicted Taxonomy']
    percent_id = row['Percent Identity']
    if ai_taxon == NOVEL_PATTERN_LABEL:
        return "⭐ AI Discovery: Novel Pattern Identified"
    if percent_id < HIGH_IDENTITY_THRESHOLD:
        return "Potentially Novel (Low NCBI Match)"
    if percent_id >= HIGH_IDENTITY_THRESHOLD:
        return "✔️ Consistent with NCBI"
    return "⚠️ AI Prediction Differs from NCBI"

# ----------------------------------------------------------------------
# NEW FUNCTION: Reusable logic for displaying results
# ----------------------------------------------------------------------
def display_results(final_df, report_title):
    """Takes a dataframe and displays the formatted report."""
    st.subheader(report_title)
    
    # Pie chart for summary
    class_counts = final_df['AI Predicted Taxonomy'].value_counts().reset_index()
    class_counts.columns = ['Taxonomy', 'Count']
    fig = px.pie(class_counts, values='Count', names='Taxonomy', title='AI Prediction Summary')
    st.plotly_chart(fig, use_container_width=True)
    
    # Conditional formatting for the dataframe
    st.dataframe(
        final_df.style.applymap(
            lambda x: 'background-color: #38761d; color: white' if x == '✔️ Consistent with NCBI' else 
                      'background-color: #f1c232; color: black' if x in ['Potentially Novel (Low NCBI Match)', '⚠️ AI Prediction Differs from NCBI'] else 
                      'background-color: #cc0000; color: white; font-weight: bold' if x == '⭐ AI Discovery: Novel Pattern Identified' else None, 
            subset=['Remarks']
        ).format({'Percent Identity': '{:.2f}%'}),
        use_container_width=True
    )
    
    st.download_button(label="Download Full Report as CSV", data=to_csv(final_df), file_name='integrated_analysis_report.csv', mime='text/csv')

# ----------------------------------------------------------------------
# MODIFIED MAIN FUNCTION
# ----------------------------------------------------------------------
def main():
    if 'analysis_run' not in st.session_state: st.session_state.analysis_run = False

    # --- Sidebar (No changes) ---
    with st.sidebar:
        st.subheader("📝 About this Project")
        st.info("This AI prototype analyzes eDNA sequences to accelerate deep-sea biodiversity discovery.")
        st.title("Elysian Analytics")
        st.markdown("---")
        st.subheader("📁 Upload Your FASTA File")
        uploaded_file = st.file_uploader("Upload a FASTA file", type=['fasta', 'fa', 'fna'], label_visibility="collapsed")
        st.subheader("🤖 Select AI Model")
        model_choice = st.selectbox("Choose analysis model:", ["XGBoost", "Deep Learning (CNN)", "Baseline (Random Forest)"], label_visibility="collapsed")
        if st.button("Analyze Sequences", type="primary", use_container_width=True):
            if uploaded_file is not None:
                st.session_state.analysis_run = True
                st.session_state.uploaded_file = uploaded_file
                st.session_state.model_choice = model_choice
            else:
                st.error("⚠️ Please upload a FASTA file first.")
                st.session_state.analysis_run = False
        st.markdown("---")
        

    st.title("AI-Powered eDNA Sequence Analyzer")

    # --- Page Content Logic ---
    if not st.session_state.analysis_run:
        with st.expander("📖 How to Use This App"):
            st.write("1. **Upload a FASTA file** using the uploader in the sidebar.")
            st.write("2. **Select an AI model** (XGBoost is recommended).")
            st.write("3. **Click 'Analyze Sequences'** to start.")
    else:
        stringio = StringIO(st.session_state.uploaded_file.getvalue().decode("utf-8"))
        raw_sequences = parse_fasta(stringio.read())
        if not raw_sequences:
            st.error("No valid sequences found in the uploaded file.")
            st.session_state.analysis_run = False
        else:
            sequences_dict = dereplicate_sequences(raw_sequences)
            st.success(f"✅ Loaded {len(raw_sequences)} total sequences.")
            st.info(f"Found {len(sequences_dict)} unique sequences (ASVs) to analyze.")
            
            label_encoder = get_label_encoder()
            ai_results_df = pd.DataFrame()
            if label_encoder:
                with st.spinner(f"Running {st.session_state.model_choice} model..."):
                    # ... (AI prediction logic remains unchanged) ...
                    model_choice = st.session_state.model_choice
                    # ... [prediction logic is unchanged] ...
                    predictions, confidence_scores, sequence_ids = None, None, None
                    if model_choice == "Deep Learning (CNN)":
                        model = load_dl_model()
                        if model:
                            processed_sequences, sequence_ids = preprocess_for_dl(sequences_dict)
                            predictions, confidence_scores = predict_with_dl_model(model, processed_sequences, label_encoder)
                    elif model_choice == "Baseline (Random Forest)":
                        model = load_rf_model()
                        if model:
                            feature_df, sequence_ids = get_kmer_features_for_prediction(sequences_dict)
                            predictions, confidence_scores = predict_with_rf_model(model, feature_df, label_encoder)
                    elif model_choice == "XGBoost":
                        model = load_xgb_model()
                        if model:
                            feature_df, sequence_ids = get_kmer_features_for_prediction(sequences_dict)
                            predictions, confidence_scores = predict_with_xgb_model(model, feature_df, label_encoder)
                    if predictions is not None:
                        ai_results_df = pd.DataFrame({'ASV ID': sequence_ids, 'AI Predicted Taxonomy': predictions, 'AI Confidence': confidence_scores})
            
            with st.spinner("Performing live BLAST search against NCBI..."):
                live_novelty_df = generate_live_novelty_report(sequences_dict)
            
            if not ai_results_df.empty and not live_novelty_df.empty:
                final_df = pd.merge(ai_results_df, live_novelty_df, on="ASV ID")
                final_df['Remarks'] = final_df.apply(get_remarks, axis=1)
                final_df['Novelty Flag'] = final_df['AI Predicted Taxonomy'] == NOVEL_PATTERN_LABEL
                final_df = final_df[['AI Predicted Taxonomy', 'AI Confidence', 'Percent Identity', 'Best Match Found in Database', 'Novelty Flag', 'Remarks']]
                
                # **MODIFIED: Call the reusable display function**
                display_results(final_df, "Your Integrated Analysis Report")
            else:
                st.error("Analysis could not be completed.")

if __name__ == "__main__":
    main()