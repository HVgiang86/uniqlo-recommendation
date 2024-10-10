import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from models import compute_cosine_similarity, build_similarity_model, predict_ratings

# Recommend items based on predicted ratings
def recommend_items(predicted_ratings, user_id, products_df, num_recommendations=5):
    if user_id in predicted_ratings.index and predicted_ratings.loc[user_id].sum() > 0:
        user_ratings = predicted_ratings.loc[user_id, :]
        recommended_items = user_ratings.sort_values(ascending=False).head(num_recommendations)
        return recommended_items
    else:
        popular_products = products_df.groupby("product_id")["star"].mean().nlargest(num_recommendations)
        return popular_products

# Content-based recommendation for a product
def content_based_recommendation(product_id, products_df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=48)
    tfidf_matrix = vectorizer.fit_transform(products_df['description'])
    pca = PCA(n_components=10)
    tfidf_matrix_pca = pca.fit_transform(tfidf_matrix.toarray())
    scaler = StandardScaler()
    products_df['price_scaled'] = scaler.fit_transform(products_df[['price']])
    input_data = products_df[products_df['id'] == product_id][['price', 'description', 'category_id']]
    input_tfidf = vectorizer.transform(input_data['description']).toarray()
    model = keras.Sequential([
        keras.layers.Embedding(input_dim=len(vectorizer.vocabulary_), output_dim=50, input_length=tfidf_matrix.shape[1]),
        keras.layers.LSTM(50, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(50),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(tfidf_matrix.toarray(), products_df['price_scaled'], epochs=20, batch_size=32, validation_split=0.2)
    lstm_prediction_scaled = model.predict(input_tfidf)
    lstm_prediction = scaler.inverse_transform(lstm_prediction_scaled)
    cosine_sim = cosine_similarity(tfidf_matrix, input_tfidf)
    products_df['cosine_similarity'] = cosine_sim.flatten()
    products_df['final_score'] = 0.5 * lstm_prediction.flatten() + 0.5 * products_df['cosine_similarity']
    cb_recommended_products = products_df[products_df['category_id'] == input_data['category_id'].values[0]].sort_values(by='final_score', ascending=False).head(6)
    cb_recommended_products = cb_recommended_products[cb_recommended_products['id'] != product_id]
    response_data = {
        'statusCode': 200,
        'message': 'Success',
        'data': cb_recommended_products[['id', 'name']].to_dict('records')
    }
    return response_data
