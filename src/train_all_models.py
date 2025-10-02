import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
import os
import pickle

# --- CHOOSE YOUR MODEL HERE ---
MODEL_TO_TRAIN = "random_forest" # Options: "random_forest", "xgboost", "cnn"

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")

# Point to the files in the main project folder
FASTA_FILE = os.path.join(PROJECT_ROOT, "training_data_processed.fasta")
LABELS_FILE = os.path.join(PROJECT_ROOT, "training_labels_processed.csv")

# Helper functions (get_kmer_features, one_hot_encode, pad_sequence) remain the same...
def get_kmer_features(sequences, k=6):
    all_kmer_counts = []
    for seq in sequences:
        kmer_counts = {}
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if "N" not in kmer:
                kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1
        all_kmer_counts.append(kmer_counts)
    return pd.DataFrame(all_kmer_counts).fillna(0)

def one_hot_encode(sequence):
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], 'N': [0,0,0,0]}
    return np.array([mapping.get(base, [0,0,0,0]) for base in sequence.upper()])

def pad_sequence(encoded_seq, length=500):
    if len(encoded_seq) > length:
        return encoded_seq[:length]
    padding = np.zeros((length - len(encoded_seq), 4))
    return np.vstack([encoded_seq, padding])

# --- Main Training Script ---
def main():
    print(f"--- Preparing to train the '{MODEL_TO_TRAIN}' model ---")
    
    labels_df = pd.read_csv(LABELS_FILE)
    labels_map = dict(zip(labels_df['Sequence_ID'], labels_df['Label']))
    
    raw_sequences, labels = [], []
    for record in SeqIO.parse(FASTA_FILE, "fasta"):
        if record.id in labels_map:
            raw_sequences.append(str(record.seq))
            labels.append(labels_map[record.id])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Calculated class weights: {class_weights_dict}")
    
    if MODEL_TO_TRAIN in ["random_forest", "xgboost"]:
        X = get_kmer_features(raw_sequences)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        train_cols = X_train.columns
        X_test = X_test.reindex(columns=train_cols, fill_value=0)
    elif MODEL_TO_TRAIN == "cnn":
        X = np.array([pad_sequence(one_hot_encode(seq)) for seq in raw_sequences])
        y_categorical = to_categorical(y, num_classes=len(label_encoder.classes_))
        X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical)
    else:
        print(f"[ERROR] Model type '{MODEL_TO_TRAIN}' not recognized.")
        return

    print(f"Training set size: {X_train.shape[0]}")
    
    if MODEL_TO_TRAIN == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
        model.fit(X_train, y_train)
        model.training_columns = train_cols
    elif MODEL_TO_TRAIN == "xgboost":
        model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X_train, y_train)
        model.training_columns = train_cols
    elif MODEL_TO_TRAIN == "cnn":
        model = Sequential([
            Conv1D(filters=32, kernel_size=12, activation='relu', input_shape=(500, 4)),
            MaxPooling1D(pool_size=4),
            Dropout(0.25),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(len(label_encoder.classes_), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.1, class_weight=class_weights_dict)

    if MODEL_TO_TRAIN in ["random_forest", "xgboost"]:
        y_pred = model.predict(X_test)
        y_true = y_test
    elif MODEL_TO_TRAIN == "cnn":
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

    # Save models to the main project folder
    if MODEL_TO_TRAIN == "cnn":
        model_path = os.path.join(PROJECT_ROOT, "eDNA_model_cnn_50_seq.h5")
        model.save(model_path)
    else:
        model_path = os.path.join(PROJECT_ROOT, f"eDNA_model_{MODEL_TO_TRAIN}_50_seq.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
    print(f"Model saved successfully to: {model_path}")

if __name__ == "__main__":
    main()