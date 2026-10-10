"""
This script reads a CSV file containing text data and fraud labels, generates sentence embeddings
for the text using a pre-trained SentenceTransformer model ('all-MiniLM-L6-v2'), and combines the
embeddings with the original data. The resulting DataFrame, which includes both the original columns
and the new embedding features, is saved to a new CSV file ('fraud_sen_emb.csv') for further analysis.

Steps performed:
1. Load text and label data from 'fraud_text.csv'.
2. Generate sentence embeddings for the text column ('mda').
3. Combine embeddings with the original DataFrame.
4. Save the combined DataFrame to 'fraud_sen_emb.csv'.
"""
from sentence_transformers import SentenceTransformer
import pandas as pd

# Load the pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the dataset
df = pd.read_csv('path/to/filtered_dataframe.csv')

texts = df['mda'].astype(str).tolist()

X = model.encode(texts, show_progress_bar=True, device='cpu')

y = df['fraudulent'].values

print("Embedding shape:", X.shape)
print("Labels shape:", y.shape)

df_embeds = pd.DataFrame(X, columns=[f'sen_emb_{i}' for i in range(X.shape[1])])
df_combined = pd.concat([df, df_embeds], axis=1)

print("Combined DataFrame shape:", df_combined.shape)

# Save the combined DataFrame to a new CSV file
df_combined.to_csv('fraud_sentence_embeddings_dataframe.csv', index=False)