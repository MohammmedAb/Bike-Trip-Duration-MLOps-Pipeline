import psycopg
from dotenv import load_dotenv
import os

load_dotenv()

AWS_INSTANCE = os.getenv('AWS_INSTANCE')
DB_USERNAME = os.getenv('DB_USERNAME')
DB_PASSWORD = os.getenv('DB_PASSWORD')
db_connection_string = f"host={AWS_INSTANCE} port=5432 dbname='monitoring' user={DB_USERNAME} password={DB_PASSWORD}"

def create_database():
    try:
        with psycopg.connect(f"host={AWS_INSTANCE} port=5432 dbname='postgres' user=postgres password=example", autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname='monitoring'")
                exists = cur.fetchone()
                if not exists:
                    cur.execute("CREATE DATABASE monitoring")
                    print("Database 'monitoring' created successfully")
    except Exception as e:
        print(f"error occurred while creating the database: {e}")

create_table_query = """
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    prediction_input JSON,
    predicted_value FLOAT,
    actual_value FLOAT, 
    prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

def create_predictions_table():
    try:
        
        with psycopg.connect(f"host={AWS_INSTANCE} port=5432 dbname='monitoring' user=postgres password=example", autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(create_table_query)
                print("Table 'predictions' created successfully in the 'monitoring' database")
    except Exception as e:
        print(f"error occurred while creating the table: {e}")

create_metrics_query = """
CREATE TABLE metrics (
    id SERIAL PRIMARY KEY,
    computation_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    r2_score FLOAT,
    rmse FLOAT,
    mean_error FLOAT,
    share_of_missing_values FLOAT,
    number_of_drifted_features INT
);
"""
def create_metrics_table():
    try:
        
        with psycopg.connect(db_connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(create_metrics_query)
                print("Table 'metrices' created successfully in the 'monitoring' database")
    except Exception as e:
        print(f"error occurred while creating the table: {e}")

# create_database()
# create_predictions_table()
create_metrics_table()

    
