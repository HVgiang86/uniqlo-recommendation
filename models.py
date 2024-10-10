import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Compute cosine similarity for the rating matrix
def compute_cosine_similarity(rating_matrix):
    rating_matrix = rating_matrix.fillna(0)
    item_similarity = cosine_similarity(rating_matrix.T)
    return pd.DataFrame(item_similarity, index=rating_matrix.columns, columns=rating_matrix.columns)

# Build similarity model based on item similarity
def build_similarity_model(item_similarity_df, l=50):
    model = {}
    max_l = item_similarity_df.shape[0] - 1
    l = min(l, max_l)
    for item in item_similarity_df.index:
        similar_items = item_similarity_df[item].nlargest(l + 1).iloc[1:]
        model[item] = similar_items
    return model

# Predict ratings using the similarity model
def predict_ratings(rating_matrix, similarity_model, k=10):
    predictions = pd.DataFrame(index=rating_matrix.index, columns=rating_matrix.columns)
    for user in rating_matrix.index:
        for item in rating_matrix.columns:
            similar_items = similarity_model[item].index
            rated_similar_items = [sim_item for sim_item in similar_items if rating_matrix.at[user, sim_item] > 0]
            top_k_rated_similar_items = rated_similar_items[:k]
            similarity_scores = similarity_model[item][top_k_rated_similar_items]
            user_ratings = rating_matrix.loc[user, top_k_rated_similar_items]
            if similarity_scores.sum() > 0:
                prediction = np.dot(similarity_scores, user_ratings) / np.sum(np.abs(similarity_scores))
            else:
                prediction = 0
            predictions.loc[user, item] = prediction
    return predictions
