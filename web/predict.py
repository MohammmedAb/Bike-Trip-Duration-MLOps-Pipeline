import mlflow
from dotenv import load_dotenv
import os
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import psycopg
import json
from datetime import datetime, timedelta
import pytz 
from mlflow.tracking import MlflowClient


load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
DB_USERNAME = os.getenv('DB_USERNAME')
DB_PASSWORD = os.getenv('DB_PASSWORD')
client = MlflowClient()

def fetch_production_model(model_name: str):
    """
    Fetch the production model from MLflow registry
    """
    mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
    production_latest_version = client.get_latest_versions(model_name, ['Production'])
    run_id=production_latest_version[-1].run_id
    production_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

    return production_model

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth specified in decimal degrees
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6378.137 * c
    return km


def preprocessing(df):
    df['started_at'] = pd.to_datetime(df['started_at'])
    df['ended_at'] = pd.to_datetime(df['ended_at'])
    
        

    df['started_day'] = df['started_at'].dt.day
    df['started_hour'] = df['started_at'].dt.hour
    df['ended_day'] = df['ended_at'].dt.day
    df['ended_hour'] = df['ended_at'].dt.hour


    df['distance'] = haversine_np(df['start_lng'],df['start_lat'], df['end_lng'],df['end_lat'])
    
    categ_clumns = ['rideable_type', 'member_casual']
    df = pd.get_dummies(df, columns=categ_clumns, drop_first=False)

    dummy_columns = ['rideable_type_docked_bike', 'rideable_type_electric_bike','member_casual_member']
    features = ['duration', 'started_day', 'started_hour', 'ended_day', 'ended_hour','distance', 'rideable_type_docked_bike', 'rideable_type_electric_bike','member_casual_member']

    # Add missing dummy columns with 0s
    for col in dummy_columns:
        if col not in df.columns:
            df[col] = False

    # Remove extra columns
    extra_columns = [col for col in df.columns if col not in features]
    df.drop(columns=extra_columns, inplace=True)

    df.fillna(0)

    return df

def predict(data ,model):
    prediction = model.predict(data)
    return prediction

def store_prediction(prediction_input_json, prediction, dummy_actual_value, prediction_time= None):
    """
    Store the prediction in the database
    """

    if prediction_time is None:
        prediction_time = datetime.now(tz=pytz.utc) + timedelta(hours=3)
    else:
        prediction_time = pd.to_datetime(prediction_time, utc=True) + timedelta(hours=3)

    
    try:
        with psycopg.connect(f"host={os.getenv('AWS_INSTANCE')} port=5432 dbname='monitoring' user={DB_USERNAME} password={DB_PASSWORD}", autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO predictions (prediction_input, predicted_value, actual_value, prediction_time) VALUES (%s, %s, %s, %s)",
                    (prediction_input_json, float(prediction[0]), dummy_actual_value, prediction_time)
                )
        print('Data inserted successfully.')
    except Exception as e:
        print(f"An error occurred: {e}")




app = Flask('test-app')

@app.route('/predict', methods=['POST'])
def api_endpoint():
    production_model = fetch_production_model('Best Model: 2023-01')
    request_data = request.get_json()
    data = request_data['data'][0]  
    prediction_time = request_data.get('prediction_time')
    df = pd.DataFrame(data, index=[0])
    df = preprocessing(df)
    prediction = predict(df, production_model)

    prediction_input_json = df.to_json(orient='split', index=False)
    # print(prediction_input_json)
    dummy_actual_value = np.random.uniform(prediction[0] - 10, prediction[0] + 10)  #assuming we have an actual value

    store_prediction(prediction_input_json, prediction, dummy_actual_value, prediction_time)

    return jsonify({'value': prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9696)