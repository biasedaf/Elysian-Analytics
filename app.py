#app.py
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

# --- Page Configuration ---
st.set_page_config(page_title="Elysian Analytics", page_icon="🌊", layout="wide")

# --- Configuration Constants ---
HIGH_IDENTITY_THRESHOLD = 90.0
NOVEL_PATTERN_LABEL = 'Rhizoclosmatium sp.'

# --- [Enhancement 1] Custom CSS ---
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(14,17,23,0.7), rgba(14,17,23,0.7)),
                        url(data:image/png;base64,{encoded_string}) no-repeat center center fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found. Using default styles.")

# --- Helper Functions (including the fix) ---
def simplify_blast_title(title):
    try:
        parts = title.split(' ')
        genus_index = -1
        for i, part in enumerate(parts):
            if part[0].isupper() and i > 0 and len(part) > 3:
                genus_index = i
                break
        if genus_index != -1: return ' '.join(parts[genus_index:genus_index+2])
        return ' '.join(parts[1:3])
    except: return title

# ... (All your other helper functions: parse_fasta, dereplicate, etc. remain unchanged) ...
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

# ... (display_results function is unchanged) ...
def display_results(final_df, report_title):
    # This function is now placed inside a tab on the results page
    st.subheader(report_title)
    
    # Pie chart for summary
    class_counts = final_df['AI Predicted Taxonomy'].value_counts().reset_index()
    class_counts.columns = ['Taxonomy', 'Count']
    fig = px.pie(class_counts, values='Count', names='Taxonomy', title='AI Prediction Summary')
    st.plotly_chart(fig, use_container_width=True)
    
    # Conditional formatting for the dataframe
    st.dataframe(
        final_df.style.apply(
            lambda x: 'background-color: #38761d; color: white' if x == '✔️ Consistent with NCBI' else 
                      'background-color: #f1c232; color: black' if x in ['Potentially Novel (Low NCBI Match)', '⚠️ AI Prediction Differs from NCBI'] else 
                      'background-color: #cc0000; color: white; font-weight: bold' if x == '⭐ AI Discovery: Novel Pattern Identified' else None, 
            subset=['Remarks']
        ).format({'Percent Identity': '{:.2f}%'}),
        use_container_width=True
    )
    
    st.download_button(label="Download Full Report as CSV", data=to_csv(final_df), file_name='integrated_analysis_report.csv', mime='text/csv')


def main():
    local_css("style.css") # Load CSS
    add_bg_from_local("dna_helix.png")

    
    if 'analysis_run' not in st.session_state: st.session_state.analysis_run = False

    with st.sidebar:
        st.title("🌊 Elysian Analytics")
        st.markdown("---")
        st.subheader("📝 About this Project")
        st.info("This AI prototype analyzes eDNA sequences to accelerate deep-sea biodiversity discovery.")
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

    if not st.session_state.analysis_run:
        # --- [Enhancement 2] Polished Homepage ---
        with st.container():
            st.subheader("An AI-Powered Platform for eDNA Analysis")
            st.markdown("#### 🔬 Our AI Model's Capabilities")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Taxonomic Classes", "8", "Trained to Identify")
            col2.metric("AI Models Available", "3", "CNN, XGBoost, RF")
            col3.metric("Analysis Speed", "< 5 sec", "Per 10 Sequences")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        with st.expander("📖 How to Use This App"):
            st.write("1. **Upload a FASTA file** using the uploader in the sidebar.")
            st.write("2. **Select an AI model** (XGBoost is recommended for best performance).")
            st.write("3. **Click 'Analyze Sequences'** to start the analysis.")
            st.write("4. Explore the **AI Prediction** and **Live Novelty Report** in the results tabs.")
        with st.container():
            st.markdown("#### 📊 Sample Analysis Output")
            st.write("The platform generates an interactive report, allowing for quick interpretation of biodiversity data.")

            sample_data = {'Taxonomy': ['Bacterium', 'Eukaryote', 'Chytridiomycota', 'Rhizoclosmatium sp.'], 'Count': [45, 32, 18, 5]}
            sample_df = pd.DataFrame(sample_data)

            fig = px.pie(sample_data, values='Count', names='Taxonomy', title='Example Prediction Summary', hole=.3)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

        st.success("**Ready to analyze?** Upload your FASTA file in the sidebar!")

    else:
        # --- Live Analysis with Enhanced Results Page ---
        stringio = StringIO(st.session_state.uploaded_file.getvalue().decode("utf-8"))
        raw_sequences = parse_fasta(stringio.read())
        
        if not raw_sequences:
            st.error("No valid sequences found in the uploaded file.")
            st.session_state.analysis_run = False
        else:
            sequences_dict = dereplicate_sequences(raw_sequences)
            # ... (Full analysis pipeline is the same) ...
            label_encoder = get_label_encoder()
            ai_results_df = pd.DataFrame()
            if label_encoder:
                with st.spinner(f"Running {st.session_state.model_choice} model..."):
                    # ... [prediction logic] ...
                    model_choice = st.session_state.model_choice
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
                
                # --- [Enhancement 3] Key Findings Section ---
                st.subheader("📊 Key Findings")
                
                total_asvs = len(final_df)
                novel_count = final_df['Novelty Flag'].sum()
                top_group = final_df['AI Predicted Taxonomy'].mode()[0]
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total ASVs Analyzed", f"{total_asvs}")
                col2.metric("⭐ AI Discoveries", f"{novel_count}")
                col3.metric("Top Predicted Group", f"{top_group}")
                
                st.markdown("---")

                # --- [Enhancement 4] Tabs for Organized Results ---
                tab1, tab2 = st.tabs(["📈 Summary & Visuals", "📄 Detailed Report"])

                with tab1:
                    st.subheader("AI Prediction Summary")
                    class_counts = final_df['AI Predicted Taxonomy'].value_counts().reset_index()
                    class_counts.columns = ['Taxonomy', 'Count']
                    fig = px.pie(class_counts, values='Count', names='Taxonomy')
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    st.subheader("Integrated Analysis Table")
                    # Using a subset of columns for the main view for clarity
                    st.dataframe(
                        final_df[['AI Predicted Taxonomy', 'AI Confidence', 'Percent Identity', 'Remarks']].style.applymap(
                            lambda x: 'background-color: #38761d; color: white' if x == '✔️ Consistent with NCBI' else 
                                      'background-color: #f1c232; color: black' if x in ['Potentially Novel (Low NCBI Match)', '⚠️ AI Prediction Differs from NCBI'] else 
                                      'background-color: #cc0000; color: white; font-weight: bold' if x == '⭐ AI Discovery: Novel Pattern Identified' else None, 
                            subset=['Remarks']
                        ).format({'Percent Identity': '{:.2f}%'}),
                        use_container_width=True
                    )
                    st.download_button(label="Download Full Report as CSV", data=to_csv(final_df), file_name='integrated_analysis_report.csv', mime='text/csv')
            else:
                st.error("Analysis could not be completed.")

if __name__ == "__main__":
    main()