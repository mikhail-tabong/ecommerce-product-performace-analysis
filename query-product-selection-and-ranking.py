import numpy as np

query_indices = [100, 500, 900]
query_results = {}

for idx in query_indices:
    similarities = similarity_matrix[idx]
    top_indices = np.argsort(similarities)[::-1][1:11]  # Exclude self
    query_results[idx] = df_cleaned.iloc[top_indices][features]
