from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
components = pca.fit_transform(df_scaled[features])

plt.figure(figsize=(10, 6))
plt.scatter(components[:, 0], components[:, 1], alpha=0.5, label="Products")
for idx in query_indices:
    plt.scatter(components[idx, 0], components[idx, 1], color='red', s=100, label=f"Query {idx}")
plt.title("PCA Projection of Product Similarity")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()
