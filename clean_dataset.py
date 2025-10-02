import pandas as pd
print("--- STARTING DATA CLEANING SCRIPT (ULTIMATE VERSION) ---")
try:
    df = pd.read_csv('deep_sea_labeled.csv')
    print(f"\n[INFO] Successfully loaded 'deep_sea_labeled.csv'. Shape: {df.shape}")
except FileNotFoundError:
    print("[ERROR] 'deep_sea_labeled.csv' not found. Exiting.")
    exit()

# Ensure the column names are correct
if 'sequence' not in df.columns or 'taxonomy' not in df.columns:
    print("[ERROR] The CSV file is missing the required 'sequence' or 'taxonomy' column headers.")
    exit()

df['taxonomy'] = df['taxonomy'].str.strip()
print("\n[INFO] Removed leading/trailing whitespace from taxonomy labels.")
print("\n[DIAGNOSTIC] Class distribution in the ORIGINAL file (after stripping whitespace):")
original_counts = df['taxonomy'].value_counts()
print(original_counts)

classes_to_keep = original_counts[original_counts > 1].index
cleaned_df = df[df['taxonomy'].isin(classes_to_keep)]

print(f"\n[INFO] Cleaned dataset shape: {cleaned_df.shape}")
print("\n[DIAGNOSTIC] Class distribution in the NEW CLEAN file:")
cleaned_counts = cleaned_df['taxonomy'].value_counts()
print(cleaned_counts)

if cleaned_df.empty:
    print("\n[ERROR] The cleaned dataset is empty! No class had more than one sample.")
else:
    # This line ensures the header is always included correctly
    cleaned_df.to_csv('deep_sea_labeled_clean.csv', index=False)
    print("\n[SUCCESS] Cleaned dataset saved to 'deep_sea_labeled_clean.csv'")