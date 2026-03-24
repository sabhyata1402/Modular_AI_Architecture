import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from model.chained_model import ChainedModel

# 1. Load the raw CSV 
print("Loading data...")
df = pd.read_csv("skeleton/data/AppGallery.csv")

# Basic inspection 
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nType 1 unique values: {df['Type 1'].unique()}")
print(f"Type 2 unique values: {df['Type 2'].unique()}")
print(f"Type 3 unique values: {df['Type 3'].unique()}")
print(f"Type 4 unique values: {df['Type 4'].unique()}")

# Clean up 
# Drop rows where any of the label columns are missing
df = df.dropna(subset=['Type 2', 'Type 3', 'Type 4'])

# Combine text columns as the input features
df['text'] = df['Ticket Summary'].fillna('') + ' ' + df['Interaction content'].fillna('')

# Remove rare classes
# Keep only Type 2 classes with at least 3 examples
counts = df['Type 2'].value_counts()
df = df[df['Type 2'].isin(counts[counts >= 3].index)]
print(f"\nShape after removing rare classes: {df.shape}")

# Vectorise text using TF-IDF 
print("\nVectorising text...")
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['text']).toarray()

# Build chained labels 
y2   = df['Type 2'].to_numpy()
y23  = (df['Type 2'] + '_' + df['Type 3']).to_numpy()
y234 = (df['Type 2'] + '_' + df['Type 3'] + '_' + df['Type 4']).to_numpy()

# Train/test split
(X_train, X_test,
 y2_train, y2_test,
 y23_train, y23_test,
 y234_train, y234_test) = train_test_split(
    X, y2, y23, y234,
    test_size=0.2,
    random_state=0,
    stratify=y2
)

# Build a minimal sample Data object
# Used ONLY for testing
class MockData:
    pass

data = MockData()
data.X_train    = X_train
data.X_test     = X_test
data.y_train    = y2_train
data.y_test     = y2_test
data.y23_train  = y23_train
data.y23_test   = y23_test
data.y234_train = y234_train
data.y234_test  = y234_test

# Run ChainedModel 
print("\nInitialising ChainedModel...")
model = ChainedModel(model_name="Test_ChainedRF")

print("\nTraining...")
model.train(data)

print("\nPredicting...")
model.predict(data)

print("\nResults:")
model.print_results(data)

print("\nTest complete. Model is working correctly!")