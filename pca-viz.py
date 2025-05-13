
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

query_indices = [100, 500, 900]
query_colors = ['red', 'blue', 'green']

# Apply PCA
pca = PCA(n_components=2)
components = pca.fit_transform(df_scaled[features])
explained_var = pca.explained_variance_ratio_

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(components[:, 0], components[:, 1], alpha=0.4, color='gray', label="All Products")

# Highlight query items
for idx, color in zip(query_indices, query_colors):
    plt.scatter(components[idx, 0], components[idx, 1], color=color, s=100, label=f"Query {idx}")

# Use concise, theme-based axis labels
plt.xlabel(f"Value & Popularity ({explained_var[0]*100:.1f}% Variance)")
plt.ylabel(f"Delivery & Return Behavior ({explained_var[1]*100:.1f}% Variance)")
plt.title("Product Clustering Based on Performance")
plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)
plt.legend()
plt.grid(True)
plt.show()
