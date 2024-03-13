import mlflow
from dotenv import load_dotenv
import os
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)

TEST_RUN_ID = os.getenv("TEST_RUN_ID")
model = mlflow.pyfunc.load_model(f"runs:/{TEST_RUN_ID}/model")



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

def predict(data):
    prediction = model.predict(data)
    return prediction

app = Flask('test-app')

@app.route('/predict', methods=['POST'])
def api_endpoint():
    data = request.get_json()
    df = pd.DataFrame(data, index=[0])
    df = preprocessing(df)
    prediction = predict(df)

    return jsonify({'value': prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9696)