from flask import jsonify
from recommendation import recommend_items, content_based_recommendation
from database import get_data

# Load data from the database
products_df, ratings_df = get_data()

# Register API routes
def register_routes(app):
    @app.route('/cfrecommend/<int:user_id>', methods=['GET'])
    def cfrecommend(user_id):
        try:
            # Load or compute predicted ratings
            predicted_ratings = ...
            # Get recommended products for the user
            recommended_products = recommend_items(predicted_ratings, user_id=user_id, products_df=products_df, num_recommendations=20)
            recommended_products_id = recommended_products.index.tolist()
            response_data = {
                'statusCode': 200,
                'message': 'Success',
                'data': [{'product_id': product_id} for product_id in recommended_products_id]
            }
            return jsonify(response_data)
        except Exception as e:
            print(e)
            response_data = {
                'statusCode': 500,
                'message': str(e),
                'data': None
            }
            return jsonify(response_data)

    @app.route('/recommend/<int:product_id>', methods=['GET'])
    def cbrecommend(product_id):
        try:
            # Get content-based recommendations for the product
            response_data = content_based_recommendation(product_id, products_df)
            return jsonify(response_data)
        except Exception as e:
            print(e)
            response_data = {
                'statusCode': 500,
                'message': str(e),
                'data': None
            }
            return jsonify(response_data)