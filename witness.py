# Import packages
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import csv 

# Import data
df = pd.read_csv('resurrection_verses.csv')

# Load a pretrained Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-roberta-large-v1")

# Calculate embeddings
embeddings = model.encode(df["text"].tolist())

# Calculate embedding similarities
similarities = model.similarity(embeddings, embeddings)

similarity_df = pd.DataFrame(similarities, index=df['verse'], columns=df['verse'])
similarity_df = similarity_df.reset_index()
numeric_df = similarity_df.drop(similarity_df.columns[0], axis=1)

# Set the diagonal to NaN
np.fill_diagonal(numeric_df.values, np.nan)

# Set the upper triangle to NaN
upper_triangle_indices = np.triu_indices_from(numeric_df, k=1)
numeric_df.values[upper_triangle_indices] = np.nan

# Add the 'canon_order' column back
result_df = pd.concat([similarity_df.iloc[:, [0]], numeric_df], axis=1)

# Export preparation
new_df = pd.melt(result_df, id_vars='verse', var_name='Column', value_name='Value')
new_df = new_df.dropna(subset=['Value'])
new_df['Comparison'] = new_df['verse'] + ' / ' + new_df['Column']

# Export data
new_df.to_csv('cosine_similarity.csv', index=False, quoting=csv.QUOTE_ALL)



