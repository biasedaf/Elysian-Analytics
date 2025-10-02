import pandas as pd, numpy as np, tensorflow as tf, pickle, os
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
def one_hot_encode_sequence(sequence):
    nucleotide_map = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], 'N': [0,0,0,0]}
    return np.array([nucleotide_map.get(n, [0,0,0,0]) for n in sequence])
def pad_sequence(encoded_seq, target_length=282):
    current_length = len(encoded_seq)
    if current_length < target_length:
        padding = np.zeros((target_length - current_length, 4))
        return np.vstack([encoded_seq, padding])
    return encoded_seq[:target_length] if current_length > target_length else encoded_seq
print("Loading and preprocessing data...")
labeled_df = pd.read_csv('deep_sea_labeled_clean.csv')
sequences = labeled_df['sequence'].apply(lambda x: pad_sequence(one_hot_encode_sequence(x)))
X = np.array(sequences.tolist())
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(labeled_df['taxonomy'])
y_categorical = keras.utils.to_categorical(y_encoded)
if not os.path.exists('models'): os.makedirs('models')
with open('models/label_encoder.pkl', 'wb') as file: pickle.dump(encoder, file)
print("Label encoder has been saved to models/label_encoder.pkl")
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.3, random_state=42, stratify=y_categorical)
print("Calculating class weights...")
y_train_indices = np.argmax(y_train, axis=1)
class_weights_array = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_indices), y=y_train_indices)
class_weights_dict = dict(enumerate(class_weights_array))
num_classes = len(encoder.classes_)
model = keras.Sequential([
    layers.Input(shape=(282, 4)),
    layers.Conv1D(filters=32, kernel_size=12, activation='relu'),
    layers.MaxPooling1D(pool_size=4),
    layers.Dropout(0.25),
    layers.Conv1D(filters=64, kernel_size=8, activation='relu'),
    layers.MaxPooling1D(pool_size=4),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
print("Training the Deep Learning model...")
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), class_weight=class_weights_dict, verbose=2)
model.save('dl_model.h5')
print("Deep Learning model saved successfully as dl_model.h5")