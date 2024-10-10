import pandas as pd
from sqlalchemy import create_engine
from config import DATABASE_URL
import psycopg2
from dotenv import load_dotenv
import ssl
import os

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

# Function to get data from the database
def get_data():
    try:
        # Read data from 'products' and 'ratings' tables
        products_df = pd.read_sql_table("products", engine)
        ratings_df = pd.read_sql_table("ratings", engine)
        return products_df, ratings_df
    except Exception as e:
        print(f"Error reading data from database: {e}")
        return None, None
