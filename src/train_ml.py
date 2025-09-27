import pandas as pd
from Bio import SeqIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# --- Configuration ---
# This section builds correct, absolute paths to prevent errors
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
FASTA_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "training_data.fasta")
LABELS_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "training_labels.csv")
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models") # Correct models folder path
MODEL_FILENAME = "random_forest_baseline.pkl"
KMER_SIZE = 6

# --- Script ---
def train_baseline_model():
    """Loads data, trains a RandomForest model, and saves it."""
    print("1. Loading data...")
    labels_df = pd.read_csv(LABELS_FILE)
    labels_dict = dict(zip(labels_df.Sequence_ID, labels_df.Taxon))

    sequences, sequence_labels = [], []
    for record in SeqIO.parse(FASTA_FILE, "fasta"):
        if record.id in labels_dict:
            sequences.append(str(record.seq))
            sequence_labels.append(labels_dict[record.id])
    print(f"Loaded {len(sequences)} labeled sequences.")

    print(f"\n2. Generating {KMER_SIZE}-mer features...")
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(KMER_SIZE, KMER_SIZE))
    kmer_vectors = vectorizer.fit_transform(sequences)

    print("3. Splitting data and training model...")
    X_train, X_test, y_train, y_test = train_test_split(
        kmer_vectors, sequence_labels, test_size=0.2, random_state=42, stratify=sequence_labels
    )
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    print("\n4. Evaluating model performance...")
    predictions = rf_model.predict(X_test)
    print(classification_report(y_test, predictions))

    print("5. Saving the trained model...")
    if not os.path.exists(MODEL_OUTPUT_DIR):
        os.makedirs(MODEL_OUTPUT_DIR)

    model_path = os.path.join(MODEL_OUTPUT_DIR, MODEL_FILENAME)
    joblib.dump(rf_model, model_path)
    print(f"Model saved successfully to: {model_path}")

if __name__ == "__main__":
    train_baseline_model()