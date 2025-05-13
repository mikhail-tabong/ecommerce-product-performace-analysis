from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(df_scaled[features])
