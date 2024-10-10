from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask_cors import CORS
from flask import Flask, jsonify
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from dotenv import load_dotenv
import ssl
import os
import psycopg2
from sklearn.metrics import mean_squared_error
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)

# Load environment variables from .env file
load_dotenv()

# Database connection parameters
db_params = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT'),
    'sslmode': os.getenv('DB_SSLMODE'),
    'sslrootcert': os.getenv('DB_SSLROOTCERT')
}
# SSL context
ssl_context = ssl.create_default_context(cafile=db_params['sslrootcert'])

# Create a database engine
engine = create_engine(
    f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}",
    connect_args={'sslmode': db_params['sslmode'], 'sslrootcert': db_params['sslrootcert']}
)

try:
    products_df = pd.read_sql_table("products", engine)
    ratings_df = pd.read_sql_table("ratings", engine)

    # ratings_df = pd.read_sql_table("ratings", engine)
except Exception as e:
    print(f"Lỗi khi đọc dữ liệu từ cơ sở dữ liệu: {e}")

# Pivot the ratings dataframe to create a user-item rating matrix
rating_matrix = ratings_df.pivot(index='account_id', columns='product_id', values='star')

print("RATING MATRIX")
print(rating_matrix)
# Save the rating matrix to a CSV file
rating_matrix.to_csv('rating_matrix.csv', index=True)

"""### Step 2: Compute Cosine Similarity"""

# Function to compute cosine similarity
def compute_cosine_similarity(rating_matrix):
    # Fill NaN values with 0 (assuming unrated items have a rating of 0)
    rating_matrix = rating_matrix.fillna(0)
    print(rating_matrix)
    # Compute the cosine similarity between items
    item_similarity = cosine_similarity(rating_matrix.T)
    return pd.DataFrame(item_similarity, index=rating_matrix.columns, columns=rating_matrix.columns)


# Compute the cosine similarity between items using the raw ratings
item_similarity = compute_cosine_similarity(rating_matrix)

# Display the item similarity matrix
print("SIMILARITY MATRIX")
print(item_similarity.iloc[:5,:5])
print(item_similarity)

"""### Step 3: Compute predicted ratings"""


# Function to store the top l similar items for each item
def build_similarity_model(item_similarity_df, l=50):
    print("BUILD MODEL")
    model = {}
    max_l = item_similarity_df.shape[0] - 1  # Maximum value for l
    l = min(l, max_l)  # Ensure l does not exceed the maximum possible value
    for item in item_similarity_df.index:
        # Get the top l similar items for the current item
        similar_items = item_similarity_df[item].nlargest(l + 1).iloc[1:]  # iloc[1:] to exclude the item itself
        model[item] = similar_items
    return model


# Function to predict ratings
def predict_ratings(rating_matrix, similarity_model, k=10):
    print("PREDICT RATINGS")
    # Create an empty dataframe for storing predictions
    predictions = pd.DataFrame(index=rating_matrix.index, columns=rating_matrix.columns)

    for user in rating_matrix.index:
        for item in rating_matrix.columns:
            # Get the top l similar items for the current item from the model
            similar_items = similarity_model[item].index

            # Filter out the items not rated by the user
            rated_similar_items = [sim_item for sim_item in similar_items if rating_matrix.at[user, sim_item] > 0]

            # Use only the top k rated similar items
            top_k_rated_similar_items = rated_similar_items[:k]

            # Get the similarity scores and user ratings for the top k similar items
            similarity_scores = similarity_model[item][top_k_rated_similar_items]
            user_ratings = rating_matrix.loc[user, top_k_rated_similar_items]

            # Compute the weighted sum of the ratings
            if similarity_scores.sum() > 0:
                prediction = np.dot(similarity_scores, user_ratings) / np.sum(np.abs(similarity_scores))
            else:
                prediction = 0  # If no similar items have been rated by the user, predict 0

            predictions.loc[user, item] = prediction

    return predictions


# Build the similarity model with top l similar items
model_size = 50  # Example model size
similarity_model = build_similarity_model(item_similarity, l=model_size)

# Generate predictions using the model size l and neighborhood size k
# The number 𝑘 refers to the number of similar items actually used for prediction computation.
# During the prediction phase, only the top 𝑘 items (from the stored l items) that a user has rated are considered.

neighbor_size = 20  # Example neighborhood size
predicted_ratings = predict_ratings(rating_matrix, similarity_model, k=neighbor_size)

predicted_ratings.iloc[:5, :5]

"""### Step 4: Generate recommendations"""

# Main function to recommend items
def recommend_items(predicted_ratings, user_id, products_df, num_recommendations=5):
    print("RECOMMEND ITEMS")
    if user_id in rating_matrix.columns and rating_matrix[user_id].sum() > 0:
        # Get the predicted ratings for the user
        user_ratings = predicted_ratings.loc[user_id, :]
        # Sort the ratings in descending order and select the top N items
        recommended_items = user_ratings.sort_values(ascending=False).head(num_recommendations)
        print(recommended_items)
        return recommended_items
    else:
        popular_products = ratings_df.groupby("product_id")["star"].mean().nlargest(num_recommendations)
        print(popular_products)
        return popular_products


@app.route('/cfrecommend/<int:user_id>', methods=['GET'])
def cfrecommend(user_id):
    try:
        recommended_products = recommend_items(predicted_ratings, user_id=user_id, products_df=products_df,
                                               num_recommendations=20)
        recommended_products_id = recommended_products.index.tolist()
        response_data = {
            'statusCode': 200,
            'message': 'Success',
            'data': [{'product_id': product_id} for product_id in recommended_products_id]
        }
        return jsonify(response_data)
    except Exception as e:
        response_data = {
            'statusCode': 500,
            'message': str(e),
            'data': None
        }
        return jsonify(response_data)


# CONTENT-BASED

# cb
products_df['description'] = products_df['description'].str.lower().str.replace('[^\w\s]', '', regex=True)
vectorizer = TfidfVectorizer(stop_words='english', max_features=48)
tfidf_matrix = vectorizer.fit_transform(products_df['description'])
print(tfidf_matrix)

pca = PCA(n_components=10)
tfidf_matrix_pca = pca.fit_transform(tfidf_matrix.toarray())
print("in")
print(tfidf_matrix_pca)

scaler = StandardScaler()
products_df['price_scaled'] = scaler.fit_transform(products_df[['price']])

X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix.toarray(), products_df['price_scaled'], test_size=0.2,
                                                    random_state=42)

input_shape = (tfidf_matrix.shape[1],)
model = tf.keras.Sequential([
    keras.layers.Embedding(input_dim=len(vectorizer.vocabulary_), output_dim=50, input_length=input_shape[0]),
    keras.layers.LSTM(50, return_sequences=True),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(50),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))


@app.route('/recommend/<int:product_id>', methods=['GET'])
def cbrecommend(product_id):
    try:
        input_data = products_df[products_df['id'] == product_id][['price', 'description', 'category_id']]
        input_tfidf = vectorizer.transform(input_data['description']).toarray()

        lstm_prediction_scaled = model.predict(input_tfidf)
        lstm_prediction = scaler.inverse_transform(lstm_prediction_scaled)

        cosine_sim = cosine_similarity(tfidf_matrix, input_tfidf)
        products_df['cosine_similarity'] = cosine_sim.flatten()

        products_df['final_score'] = 0.5 * lstm_prediction.flatten() + 0.5 * products_df['cosine_similarity']

        cb_recommended_products = products_df[
            products_df['category_id'] == input_data['category_id'].values[0]].sort_values(by='final_score',
                                                                                           ascending=False).head(6)
        cb_recommended_products = cb_recommended_products[cb_recommended_products['id'] != product_id]

        response_data = {
            'statusCode': 200,
            'message': 'Success',
            'data': cb_recommended_products[['id', 'name']].to_dict('records')
        }
        return jsonify(response_data)

    except Exception as e:
        response_data = {
            'statusCode': 500,
            'message': str(e),
            'data': None
        }
        return jsonify(response_data)


# Huấn luyện mô hình với Early Stopping và sẽ ngừng khi mô hình khi bị overfitting
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test),
                    callbacks=[early_stopping])

# Đánh giá mô hình
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

# RUN SERVER
if __name__ == '__main__':
    app.run(port=3006, debug=True)

"""### Step 5: Perform 5-fold cross-validation

#### MAE and RMSE
"""


# Function to perform 5-fold cross-validation
def cross_validate(ratings_df, products_df, model_size, neighbor_size, num_recommendations=5):
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    mae_scores = []
    rmse_scores = []

    for train_index, test_index in kf.split(ratings_df):
        train_df = ratings_df.iloc[train_index]
        test_df = ratings_df.iloc[test_index]

        # Pivot the training dataframe to create a user-item rating matrix
        train_rating_matrix = train_df.pivot(index='account_id', columns='product_id', values='star')

        # Compute the cosine similarity matrix
        item_similarity_df = compute_cosine_similarity(train_rating_matrix)

        # Build the similarity model with top l similar items
        similarity_model = build_similarity_model(item_similarity_df, l=model_size)

        # Generate predictions using the model size l and neighborhood size k
        predicted_ratings = predict_ratings(train_rating_matrix, similarity_model, k=neighbor_size)

        # Evaluate the model on the test set
        test_rating_matrix = test_df.pivot(index='account_id', columns='product_id', values='star')
        test_rating_matrix = test_rating_matrix.fillna(0)

        user_ids = test_df['account_id'].unique()
        item_ids = test_df['product_id'].unique()

        test_predictions = []
        test_actuals = []

        for user_id in user_ids:
            for item_id in item_ids:
                if item_id in predicted_ratings.columns and user_id in predicted_ratings.index:
                    predicted_rating = predicted_ratings.at[user_id, item_id]
                    actual_rating = test_rating_matrix.at[user_id, item_id]
                    test_predictions.append(predicted_rating)
                    test_actuals.append(actual_rating)

        mae = mean_absolute_error(test_actuals, test_predictions)
        rmse = np.sqrt(mean_squared_error(test_actuals, test_predictions))
        mae_scores.append(mae)
        rmse_scores.append(rmse)

    return np.mean(mae_scores), np.mean(rmse_scores)


# Example usage: Perform 5-fold cross-validation
model_size = 20  # Example model size
neighbor_size = 10  # Example neighborhood size
num_recommendations = 20  # Example number of recommendations

mae_score, rmse_score = cross_validate(ratings_df, products_df, model_size, neighbor_size, num_recommendations)
print(f'Average MAE over 5 folds: {mae_score}')
print(f'Average RMSE over 5 folds: {rmse_score}')

"""#### Additional metrics"""


# Additional Metrics
def precision_at_k(actual, predicted, k):
    relevant_items = set(actual[:k])
    recommended_items = set(predicted[:k])
    if not relevant_items:
        return 0
    precision = len(relevant_items & recommended_items) / float(k)
    return precision


def recall_at_k(actual, predicted, k):
    relevant_items = set(actual[:k])
    recommended_items = set(predicted[:k])
    if not relevant_items:
        return 0
    recall = len(relevant_items & recommended_items) / float(len(relevant_items))
    return recall


def f1_score_at_k(actual, predicted, k):
    precision = precision_at_k(actual, predicted, k)
    recall = recall_at_k(actual, predicted, k)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def hit_rate(actual, predicted, k):
    hits = sum([1 for a, p in zip(actual, predicted) if len(set(a[:k]) & set(p[:k])) > 0])
    return hits / float(len(actual))


def coverage(predicted, total_items):
    recommended_items = set([item for sublist in predicted for item in sublist])
    return len(recommended_items) / float(total_items)


def diversity(predicted, similarity_matrix):
    total_diversity = 0
    for recommendation_list in predicted:
        for i in range(len(recommendation_list)):
            for j in range(i + 1, len(recommendation_list)):
                total_diversity += (1 - similarity_matrix.loc[recommendation_list[i], recommendation_list[j]])
    return total_diversity / float(len(predicted))


def novelty(predicted, popularity_dict):
    total_novelty = 0
    for recommendation_list in predicted:
        total_novelty += sum([popularity_dict.get(item, 0) for item in recommendation_list])
    return total_novelty / float(len(predicted))


def user_coverage(predicted, total_users):
    users_with_recommendations = len([user for user in predicted if len(user) > 0])
    return users_with_recommendations / float(total_users)


# Perform Cross-Validation with Additional Metrics
def cross_validate(ratings_df, products_df, model_size, neighbor_size, num_recommendations=5):
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    mae_scores = []
    rmse_scores = []
    precisions = []
    recalls = []
    f1_scores = []
    hit_rates = []
    coverages = []
    diversities = []
    novelties = []
    user_coverages = []

    total_items = len(products_df['id'].unique())
    popularity_dict = ratings_df['product_id'].value_counts().to_dict()

    for train_index, test_index in kf.split(ratings_df):
        train_df = ratings_df.iloc[train_index]
        test_df = ratings_df.iloc[test_index]

        train_rating_matrix = train_df.pivot(index='account_id', columns='product_id', values='star')
        item_similarity_df = compute_cosine_similarity(train_rating_matrix)
        similarity_model = build_similarity_model(item_similarity_df, l=model_size)
        predicted_ratings = predict_ratings(train_rating_matrix, similarity_model, k=neighbor_size)

        test_rating_matrix = test_df.pivot(index='account_id', columns='product_id', values='star')
        test_rating_matrix = test_rating_matrix.fillna(0)

        user_ids = test_df['account_id'].unique()
        item_ids = test_df['product_id'].unique()

        test_predictions = []
        test_actuals = []
        actual_list = []
        predicted_list = []

        for user_id in user_ids:
            user_actual = []
            user_predicted = []
            for item_id in item_ids:
                if item_id in predicted_ratings.columns and user_id in predicted_ratings.index:
                    predicted_rating = predicted_ratings.at[user_id, item_id]
                    actual_rating = test_rating_matrix.at[user_id, item_id]
                    test_predictions.append(predicted_rating)
                    test_actuals.append(actual_rating)
                    if actual_rating > 0:
                        user_actual.append(item_id)
                    if predicted_rating > 0:
                        user_predicted.append(item_id)
            actual_list.append(user_actual)
            predicted_list.append(user_predicted)

        mae = mean_absolute_error(test_actuals, test_predictions)
        rmse = np.sqrt(mean_squared_error(test_actuals, test_predictions))
        precision = np.mean([precision_at_k(a, p, num_recommendations) for a, p in zip(actual_list, predicted_list)])
        recall = np.mean(
            [recall_at_k(a, p, num_recommendations) for a, p in zip(actual_list, predicted_list) if len(a) > 0])
        f1 = np.mean([f1_score_at_k(a, p, num_recommendations) for a, p in zip(actual_list, predicted_list)])
        hit = hit_rate(actual_list, predicted_list, num_recommendations)
        cover = coverage(predicted_list, total_items)
        div = diversity(predicted_list, item_similarity_df)
        nov = novelty(predicted_list, popularity_dict)
        user_cover = user_coverage(predicted_list, len(user_ids))

        mae_scores.append(mae)
        rmse_scores.append(rmse)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        hit_rates.append(hit)
        coverages.append(cover)
        diversities.append(div)
        novelties.append(nov)
        user_coverages.append(user_cover)

    results = {
        'MAE': np.mean(mae_scores),
        'RMSE': np.mean(rmse_scores),
        'Precision@K': np.mean(precisions),
        'Recall@K': np.mean(recalls),
        'F1@K': np.mean(f1_scores),
        'Hit Rate': np.mean(hit_rates),
        'Coverage': np.mean(coverages),
        'Diversity': np.mean(diversities),
        'Novelty': np.mean(novelties),
        'User Coverage': np.mean(user_coverages),
    }

    return results


model_size = 20
neighbor_size = 10
num_recommendations = 20
evaluation_results = cross_validate(ratings_df, products_df, model_size, neighbor_size, num_recommendations)

evaluation_results

# MAE: Measures the average absolute difference between predicted and actual ratings; lower values indicate better accuracy.
mae = 1.2936552026857575

# RMSE: Similar to MAE but gives more weight to larger errors; lower values indicate better prediction performance with fewer large errors.
rmse = 2.3951571050289884

# Precision@K: Proportion of recommended items in the top-K set that are relevant; higher values indicate better recommendation relevance.
precision_at_k = 0.09990469161923907

# Recall@K: Proportion of relevant items that are successfully recommended out of all relevant items; higher values indicate more relevant items are recommended.
recall_at_k = 0.867010535278214

# F1@K: Harmonic mean of precision and recall; higher values indicate a better balance between precision and recall.
f1_at_k = 0.17156908826154543

# Hit Rate: Fraction of users for whom at least one of the top-K recommended items is relevant; higher values indicate more users receive relevant recommendations.
hit_rate = 0.9450584430181331

# Coverage: Proportion of items that the system can recommend from the total catalog; higher values indicate more items are recommended at least once.
coverage = 0.694

# Diversity: Measures how varied the recommendations are; higher values indicate less similarity between recommended items.
diversity = 241.66825558144834

# Novelty: Measures how novel or unexpected the recommended items are to the user; higher values indicate recommendations include less popular items.
novelty = 382.3063746636444

# User Coverage: Proportion of users who have received at least one recommendation; higher values indicate more users receive recommendations.
user_coverage = 0.9884388282676255

"""#### Evaluate and Plot MAE for Different Model and Neighborhood Sizes"""

import matplotlib.pyplot as plt

# Define the specific model sizes and corresponding neighborhood sizes
model_sizes = [10, 20, 50]
neighborhood_sizes = {10: [3, 6, 10], 20: [6, 13, 20], 50: [16, 33, 50]}


# Adjusted function to define relevance based on a rating threshold
def evaluate_recommendation_system_specific(ratings_df, products_df, model_sizes, neighborhood_sizes,
                                            rating_threshold=4, num_recommendations=5):
    results = {'MAE': []}

    for model_size in model_sizes:
        for neighbor_size in neighborhood_sizes[model_size]:
            evaluation = cross_validate_with_relevance(ratings_df, products_df, model_size, neighbor_size,
                                                       rating_threshold, num_recommendations)
            results['MAE'].append((model_size, neighbor_size, evaluation['MAE']))
    return results


def cross_validate_with_relevance(ratings_df, products_df, model_size, neighbor_size, rating_threshold,
                                  num_recommendations=5):
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    mae_scores = []

    for train_index, test_index in kf.split(ratings_df):
        train_df = ratings_df.iloc[train_index]
        test_df = ratings_df.iloc[test_index]

        train_rating_matrix = train_df.pivot(index='account_id', columns='product_id', values='star')
        item_similarity_df = compute_cosine_similarity(train_rating_matrix)
        similarity_model = build_similarity_model(item_similarity_df, l=model_size)
        predicted_ratings = predict_ratings(train_rating_matrix, similarity_model, k=neighbor_size)

        test_rating_matrix = test_df.pivot(index='account_id', columns='product_id', values='star')
        test_rating_matrix = test_rating_matrix.fillna(0)

        user_ids = test_df['account_id'].unique()
        item_ids = test_df['product_id'].unique()

        test_predictions = []
        test_actuals = []

        for user_id in user_ids:
            for item_id in item_ids:
                if item_id in predicted_ratings.columns and user_id in predicted_ratings.index:
                    predicted_rating = predicted_ratings.at[user_id, item_id]
                    actual_rating = test_rating_matrix.at[user_id, item_id]
                    test_predictions.append(predicted_rating)
                    test_actuals.append(actual_rating)

        mae = mean_absolute_error(test_actuals, test_predictions)
        mae_scores.append(mae)

    return {'MAE': np.mean(mae_scores)}


# Evaluate the recommendation system
evaluation_results_specific = evaluate_recommendation_system_specific(ratings_df, products_df, model_sizes,
                                                                      neighborhood_sizes, rating_threshold=4,
                                                                      num_recommendations=20)

# Extract MAE data for plotting
mae_data_specific = evaluation_results_specific['MAE']

# Plot the MAE values
plt.figure(figsize=(10, 6))

for model_size in model_sizes:
    neighbor_size, mae = zip(*[(ns, m) for ms, ns, m in mae_data_specific if ms == model_size])
    plt.plot(neighbor_size, mae, marker='o', label=f'Model Size {model_size}')

plt.xlabel('Neighborhood Size')
plt.ylabel('MAE')
plt.title('MAE vs. Neighborhood Size for Different Model Sizes')
plt.legend()
plt.grid(True)
plt.show()

mae_data_specific
