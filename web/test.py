import requests
import pandas as pd
from dotenv import load_dotenv
import time
import random
import numpy as np


load_dotenv()

test_df = pd.read_csv('/home/mohammed/project/mlops-project/data/202301-capitalbikeshare-tripdata.csv')

url = 'http://localhost:9696/predict'

for _, row in test_df.sample(frac=1).iterrows():  
    random_days_ago = np.random.randint(1, 20)  
    random_timestamp = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=random_days_ago)
    random_timestamp = random_timestamp.floor('D')

    test_data = {
        "data": [{
            "ride_id": [row["ride_id"]],
            "rideable_type": [row["rideable_type"]],
            "started_at": [row["started_at"]],
            "ended_at": [row["ended_at"]],
            "start_station_name": [row["start_station_name"]],
            "start_station_id": [row["start_station_id"]],
            "end_station_name": [row["end_station_name"]],
            "end_station_id": [row["end_station_id"]],
            "start_lat": [row["start_lat"]],
            "start_lng": [row["start_lng"]],
            "end_lat": [row["end_lat"]],
            "end_lng": [row["end_lng"]],
            "member_casual": [row["member_casual"]]
        }],
        "prediction_time": random_timestamp.isoformat()
    }

    response = requests.post(url, json=test_data)
    print(f"Data sent successfully: {response.json()}")
