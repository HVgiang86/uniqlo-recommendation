from flask import Flask
from flask import Flask
from flask_cors import CORS
from routes import register_routes

# Initialize Flask app
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS)
CORS(app)

# Register API routes
register_routes(app)

# Run the app
if __name__ == '__main__':
    app.run(port=3006, debug=True)