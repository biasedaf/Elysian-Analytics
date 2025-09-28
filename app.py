import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import tensorflow as tf
from tensorflow import keras
import os
import pickle
from collections import defaultdict # <<< ADDED

# --- Page Config and Helper functions (parse_fasta, one_hot_encode, etc.) ---
st.set_page_config(
    page_title="AI-Powered eDNA Sequence Analyzer",
    page_icon="🧬",
    layout="wide"
)

def parse_fasta(file_content):
    sequences = {}
    current_id = None
    current_seq = []
    for line in file_content.strip().split('\n'):
        line = line.strip()
        if line.startswith('>'):
            if current_id is not None:
                sequences[current_id] = ''.join(current_seq)
            current_id = line[1:]
            current_seq = []
        else:
            current_seq.append(line.upper())
    if current_id is not None:
        sequences[current_id] = ''.join(current_seq)
    return sequences

def one_hot_encode_sequence(sequence):
    nucleotide_map = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1]}
    encoded = []
    for nucleotide in sequence:
        encoded.append(nucleotide_map.get(nucleotide, [0,0,0,0]))
    return np.array(encoded)

def pad_sequence(encoded_seq, target_length=423):
    current_length = len(encoded_seq)
    if current_length < target_length:
        padding = np.zeros((target_length - current_length, 4))
        padded_seq = np.vstack([encoded_seq, padding])
    elif current_length > target_length:
        padded_seq = encoded_seq[:target_length]
    else:
        padded_seq = encoded_seq
    return padded_seq

# --- Preprocessing functions ---

def preprocess_for_dl(sequences):
    processed_sequences = []
    sequence_ids = []
    for seq_id, sequence in sequences.items():
        encoded = one_hot_encode_sequence(sequence)
        padded = pad_sequence(encoded, target_length=423)
        processed_sequences.append(padded)
        sequence_ids.append(seq_id)
    return np.array(processed_sequences), sequence_ids

# <<< UPDATED RF Preprocessing to match training >>>
def get_kmer_features_for_prediction(sequences, k=6):
    """Converts uploaded sequences into a k-mer DataFrame for prediction."""
    all_kmer_counts = []
    sequence_ids = list(sequences.keys())
    
    for seq_id in sequence_ids:
        seq = sequences[seq_id]
        kmer_counts = defaultdict(int)
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if "N" not in kmer:
                kmer_counts[kmer] += 1
        all_kmer_counts.append(kmer_counts)
        
    return pd.DataFrame(all_kmer_counts).fillna(0), sequence_ids

# --- Model Loading and Prediction ---

def load_dl_model():
    model_path = "dl_model.h5"
    if os.path.exists(model_path):
        try:
            return keras.models.load_model(model_path)
        except Exception as e:
            st.error(f"Error loading DL model: {e}")
    return None

def load_rf_model():
    model_path = os.path.join("models", "random_forest_baseline.pkl")
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as file:
                return pickle.load(file)
        except Exception as e:
            st.error(f"Error loading Random Forest model: {e}")
    return None

def predict_with_dl_model(model, sequences):
    try:
        predictions = model.predict(sequences)
        return ["Rare" if np.argmax(pred) == 1 else "Common" for pred in predictions]
    except Exception as e:
        st.error(f"Error during DL prediction: {e}")
    return None

# <<< UPDATED to handle feature alignment for RF >>>
def predict_with_rf_model(model, feature_df):
    """Makes predictions with the RF model, aligning columns."""
    try:
        # Align columns of the new data with the model's training columns
        model_columns = model.feature_names_in_
        aligned_df = pd.DataFrame(columns=model_columns)
        
        # Add columns from new data that are also in the model
        for col in feature_df.columns:
            if col in model_columns:
                aligned_df[col] = feature_df[col]
        
        # Fill missing columns with 0
        aligned_df = aligned_df.fillna(0)

        predictions = model.predict(aligned_df[model_columns])
        return ["Rare" if pred == 1 else "Common" for pred in predictions]
    except Exception as e:
        st.error(f"Error during Random Forest prediction: {e}")
    return None

def load_novelty_report():
    report_path = "novelty_report.csv"
    if os.path.exists(report_path):
        try:
            return pd.read_csv(report_path)
        except Exception as e:
            st.error(f"Error loading novelty report: {e}")
    return None

# --- Main Application ---
def main():
    st.title("🧬 AI-Powered eDNA Sequence Analyzer")
    st.markdown("This application uses artificial intelligence to analyze environmental DNA (eDNA) sequences...")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📁 Upload Your FASTA File")
        uploaded_file = st.file_uploader("Choose a FASTA file (.fasta or .fa)", type=['fasta', 'fa'])
    with col2:
        st.subheader("🤖 Select AI Model")
        model_choice = st.selectbox("Choose analysis model:", ["Deep Learning (CNN)", "Baseline (Random Forest)"])
        
    st.subheader("🔬 Run Analysis")
    if st.button("Analyze Sequences", type="primary"):
        if uploaded_file is None:
            st.error("⚠️ Please upload a FASTA file first.")
            return

        with st.spinner("Analyzing sequences..."):
            file_content = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            sequences_dict = parse_fasta(file_content)
            
            if not sequences_dict:
                st.error("No valid sequences found in the uploaded file.")
                return
            
            st.success(f"✅ Loaded {len(sequences_dict)} sequences.")
            
            predictions, sequence_ids = None, None

            # <<< UPDATED Random Forest Logic >>>
            if model_choice == "Deep Learning (CNN)":
                model = load_dl_model()
                if model:
                    processed_sequences, sequence_ids = preprocess_for_dl(sequences_dict)
                    predictions = predict_with_dl_model(model, processed_sequences)
            
            elif model_choice == "Baseline (Random Forest)":
                model = load_rf_model()
                if model:
                    feature_df, sequence_ids = get_kmer_features_for_prediction(sequences_dict)
                    predictions = predict_with_rf_model(model, feature_df)

            if predictions:
                results_df = pd.DataFrame({'Sequence ID': sequence_ids, 'Predicted Class': predictions})
                st.subheader("📊 Analysis Results")
                st.dataframe(results_df, use_container_width=True)
                
                common_count = predictions.count("Common")
                rare_count = len(predictions) - common_count
                
                mcol1, mcol2, mcol3 = st.columns(3)
                mcol1.metric("Total Sequences", len(predictions))
                mcol2.metric("Common Species", common_count)
                mcol3.metric("Rare Species", rare_count)
                
    st.subheader("📈 Pre-computed Novelty Analysis")
    novelty_df = load_novelty_report()
    if novelty_df is not None:
        st.dataframe(novelty_df, use_container_width=True)
    
    st.markdown("---")
    st.markdown("*Developed for AI-powered eDNA sequence analysis*")

if __name__ == "__main__":
    main()