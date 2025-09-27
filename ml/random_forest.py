import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from embeddings import fasta_to_matrix

# Load embeddings
X, ids = fasta_to_matrix("../bioinformatics/ASVs.fasta", k=6)

# Fake labels (replace later with taxonomy)
y = np.random.choice(["taxon1", "taxon2", "taxon3"], size=len(X))

# Train baseline RF
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)

# Save model
with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)

print("âœ… RandomForest baseline model saved as rf_model.pkl")
