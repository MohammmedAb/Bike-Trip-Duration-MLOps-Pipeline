import requests

test_data = {
    "ride_id": ["65F0ACD101BF0D49"],
    "rideable_type": ["classic_bike"],
    "started_at": ["2023-01-04 19:34:07"],
    "ended_at": ["2023-01-04 19:39:29"],
    "start_station_name": ["East Falls Church Metro / Sycamore St & 19th St N"],
    "start_station_id": [31904.0],
    "end_station_name": ["W Columbia St & N Washington St"],
    "end_station_id": [32609.0],
    "start_lat": [29.885321],
    "start_lng": [-17.156427],
    "end_lat": [30.885621],
    "end_lng": [-87.166917],
    "member_casual": ["member"]
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=test_data)
print(response.json())