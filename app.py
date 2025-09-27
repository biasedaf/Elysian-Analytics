import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import tensorflow as tf
from tensorflow import keras
import os

# Set page configuration
st.set_page_config(
    page_title="AI-Powered eDNA Sequence Analyzer",
    page_icon="🧬",
    layout="wide"
)

def parse_fasta(file_content):
    """
    Parse FASTA file content and return sequences with their IDs.
    
    Args:
        file_content (str): Content of the FASTA file
        
    Returns:
        dict: Dictionary with sequence IDs as keys and sequences as values
    """
    sequences = {}
    current_id = None
    current_seq = []
    
    for line in file_content.strip().split('\n'):
        line = line.strip()
        if line.startswith('>'):
            # Save previous sequence if exists
            if current_id is not None:
                sequences[current_id] = ''.join(current_seq)
            # Start new sequence
            current_id = line[1:]  # Remove '>' character
            current_seq = []
        else:
            current_seq.append(line.upper())
    
    # Don't forget the last sequence
    if current_id is not None:
        sequences[current_id] = ''.join(current_seq)
    
    return sequences

def one_hot_encode_sequence(sequence):
    """
    One-hot encode a DNA sequence.
    
    Args:
        sequence (str): DNA sequence string
        
    Returns:
        numpy.ndarray: One-hot encoded sequence
    """
    # Mapping for nucleotides to one-hot vectors
    nucleotide_map = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1]
    }
    
    # Handle unknown nucleotides by mapping them to zeros
    encoded = []
    for nucleotide in sequence:
        if nucleotide in nucleotide_map:
            encoded.append(nucleotide_map[nucleotide])
        else:
            # Unknown nucleotide - use zeros
            encoded.append([0, 0, 0, 0])
    
    return np.array(encoded)

def pad_sequence(encoded_seq, target_length=423):
    """
    Pad or truncate sequence to target length.
    
    Args:
        encoded_seq (numpy.ndarray): One-hot encoded sequence
        target_length (int): Target sequence length
        
    Returns:
        numpy.ndarray: Padded sequence
    """
    current_length = len(encoded_seq)
    
    if current_length < target_length:
        # Pad with zeros
        padding = np.zeros((target_length - current_length, 4))
        padded_seq = np.vstack([encoded_seq, padding])
    elif current_length > target_length:
        # Truncate
        padded_seq = encoded_seq[:target_length]
    else:
        padded_seq = encoded_seq
    
    return padded_seq

def preprocess_sequences(sequences):
    """
    Preprocess sequences for the deep learning model.
    
    Args:
        sequences (dict): Dictionary of sequence IDs and sequences
        
    Returns:
        numpy.ndarray: Preprocessed sequences ready for model input
        list: List of sequence IDs in the same order
    """
    processed_sequences = []
    sequence_ids = []
    
    for seq_id, sequence in sequences.items():
        # One-hot encode
        encoded = one_hot_encode_sequence(sequence)
        
        # Pad to target length
        padded = pad_sequence(encoded, target_length=423)
        
        processed_sequences.append(padded)
        sequence_ids.append(seq_id)
    
    return np.array(processed_sequences), sequence_ids

def load_dl_model():
    """
    Load the deep learning model from file.
    
    Returns:
        tensorflow.keras.Model or None: Loaded model or None if file not found
    """
    model_path = "dl_model.h5"
    
    if os.path.exists(model_path):
        try:
            model = keras.models.load_model(model_path)
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    else:
        st.error(f"Model file '{model_path}' not found. Please ensure it's in the same directory as this app.")
        return None

def predict_with_dl_model(model, sequences):
    """
    Make predictions using the deep learning model.
    
    Args:
        model: Loaded Keras model
        sequences (numpy.ndarray): Preprocessed sequences
        
    Returns:
        list: Predicted class labels
    """
    try:
        # Make predictions
        predictions = model.predict(sequences)
        
        # Convert probabilities to class labels
        # Assuming binary classification: 0 = Common, 1 = Rare
        predicted_classes = []
        for pred in predictions:
            if len(pred) == 1:
                # Single output (binary classification with sigmoid)
                class_label = "Rare" if pred[0] > 0.5 else "Common"
            else:
                # Multiple outputs (categorical classification)
                class_idx = np.argmax(pred)
                class_label = "Rare" if class_idx == 1 else "Common"
            
            predicted_classes.append(class_label)
        
        return predicted_classes
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

def analyze_with_random_forest():
    """
    Placeholder function for Random Forest analysis.
    
    Returns:
        str: Placeholder message
    """
    return "Analysis with Random Forest is not yet implemented. Please use the Deep Learning (CNN) model."

def load_novelty_report():
    """
    Load the pre-computed novelty report.
    
    Returns:
        pandas.DataFrame or None: Novelty report data or None if file not found
    """
    report_path = "novelty_report.csv"
    
    if os.path.exists(report_path):
        try:
            df = pd.read_csv(report_path)
            return df
        except Exception as e:
            st.error(f"Error loading novelty report: {str(e)}")
            return None
    else:
        st.warning(f"Novelty report file '{report_path}' not found.")
        return None

def main():
    """
    Main Streamlit application function.
    """
    # Title and description
    st.title("🧬 AI-Powered eDNA Sequence Analyzer")
    
    st.markdown("""
    This application uses artificial intelligence to analyze environmental DNA (eDNA) sequences. 
    Upload your FASTA file containing DNA sequences, select an AI model, and get predictions 
    about whether each sequence represents common or rare species.
    """)
    
    # Create two columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File uploader
        st.subheader("📁 Upload Your FASTA File")
        uploaded_file = st.file_uploader(
            "Choose a FASTA file (.fasta or .fa)",
            type=['fasta', 'fa'],
            help="Upload a FASTA file containing your eDNA sequences"
        )
    
    with col2:
        # Model selection
        st.subheader("🤖 Select AI Model")
        model_choice = st.selectbox(
            "Choose analysis model:",
            ["Deep Learning (CNN)", "Baseline (Random Forest)"],
            help="Select the AI model to use for sequence analysis"
        )
    
    # Analyze button
    st.subheader("🔬 Run Analysis")
    analyze_button = st.button("Analyze Sequences", type="primary")
    
    # Analysis section
    if analyze_button:
        if uploaded_file is None:
            st.error("⚠️ Please upload a FASTA file before running analysis.")
            return
        
        # Show progress
        with st.spinner("Analyzing sequences..."):
            # Read and parse FASTA file
            file_content = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            sequences = parse_fasta(file_content)
            
            if not sequences:
                st.error("No valid sequences found in the uploaded file. Please check your FASTA format.")
                return
            
            st.success(f"✅ Successfully loaded {len(sequences)} sequences")
            
            # Model analysis
            if model_choice == "Deep Learning (CNN)":
                # Load model
                model = load_dl_model()
                
                if model is None:
                    return
                
                # Preprocess sequences
                processed_sequences, sequence_ids = preprocess_sequences(sequences)
                
                # Make predictions
                predictions = predict_with_dl_model(model, processed_sequences)
                
                if predictions is None:
                    return
                
                # Create results dataframe
                results_df = pd.DataFrame({
                    'Sequence ID': sequence_ids,
                    'Predicted Class': predictions
                })
                
                # Display results
                st.subheader("📊 Analysis Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Summary statistics
                common_count = sum(1 for pred in predictions if pred == "Common")
                rare_count = sum(1 for pred in predictions if pred == "Rare")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Sequences", len(predictions))
                with col2:
                    st.metric("Common Species", common_count)
                with col3:
                    st.metric("Rare Species", rare_count)
                
            else:  # Random Forest
                st.info(analyze_with_random_forest())
    
    # Always show novelty report if available
    st.subheader("📈 Pre-computed Novelty Analysis")
    novelty_df = load_novelty_report()
    
    if novelty_df is not None:
        st.dataframe(novelty_df, use_container_width=True)
    else:
        st.info("Novelty report will be displayed here when available.")
    
    # Footer
    st.markdown("---")
    st.markdown("*Developed for AI-powered eDNA sequence analysis*")

if __name__ == "__main__":
    main()