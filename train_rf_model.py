import pandas as pd
import numpy as np
from Bio import SeqIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os
from collections import defaultdict


# --- Helper Function for K-mer Counting ---
def get_kmer_features(sequences, k=6):
    """Converts a list of DNA sequences into k-mer frequency features."""
    all_kmer_counts = []
    for seq in sequences:
        kmer_counts = defaultdict(int)
        for i in range(len(seq) - k + 1):
            kmer = seq[i : i + k]
            if "N" not in kmer:  # Skip k-mers with unknown bases
                kmer_counts[kmer] += 1
        all_kmer_counts.append(kmer_counts)

    # Convert the list of dictionaries to a DataFrame (feature matrix)
    feature_df = pd.DataFrame(all_kmer_counts).fillna(0)
    return feature_df


print("Loading and preparing data for Random Forest model...")

# 1. Load the data
asv_table = pd.read_csv("ASV_table.csv", index_col=0)
asv_table.columns = ["abundance"]

# 2. Create the same dummy labels (0 for rare, 1 for common)
median_abundance = asv_table["abundance"].median()
labels = {
    seq: 1 if count > median_abundance else 0
    for seq, count in asv_table["abundance"].items()
}

# Get lists of sequences and their corresponding labels
dna_sequences = list(labels.keys())
numeric_labels = list(labels.values())

# 3. Create k-mer features for the Random Forest model
X_features = get_kmer_features(dna_sequences, k=6)

# Align features and labels
y_labels = pd.Series(numeric_labels, index=X_features.index)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_labels, test_size=0.2, random_state=42
)

# 4. Train the Random Forest Classifier
print("Training the Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
print(f"Random Forest Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# 5. Save the trained model
# Ensure the 'models' directory exists
if not os.path.exists("models"):
    os.makedirs("models")

model_path = os.path.join("models", "random_forest_baseline.pkl")
with open(model_path, "wb") as file:
    pickle.dump(rf_model, file)

print(f"\nTraining complete! Model saved to {model_path}")
