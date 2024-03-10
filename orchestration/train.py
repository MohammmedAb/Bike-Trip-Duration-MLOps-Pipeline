import pandas as pd
import os
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import mlflow
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.tracking import MlflowClient

from prefect import task, flow

@task(retries=3, retry_delay_seconds=2, log_prints=False)
def read_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    return df



def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    
    All args must be of equal length.    
    
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6378.137 * c
    return km

@task
def preprocessing(df):
    df['started_at'] = pd.to_datetime(df['started_at'])
    df['ended_at'] = pd.to_datetime(df['ended_at'])
    
    df['duration'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60
    df = df[df['duration']>0]
        

    df['started_day'] = df['started_at'].dt.day
    df['started_hour'] = df['started_at'].dt.hour
    df['ended_day'] = df['ended_at'].dt.day
    df['ended_hour'] = df['ended_at'].dt.hour

    
    Q1 = df['duration'].quantile(0.25)
    Q3 = df['duration'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Remove outliers
    df_filtered = df[(df['duration'] >= lower_bound) & (df['duration'] <= upper_bound)]

    df['distance'] = haversine_np(df['start_lng'],df['start_lat'], df['end_lng'],df['end_lat'])
    
    categ_clumns = ['rideable_type', 'member_casual']
    df = pd.get_dummies(df, columns=categ_clumns, drop_first=True)

    deleted_columns=['started_at','ended_at','ride_id', 'start_lng', 'start_lat', 'end_lng', 'end_lat', 'start_station_name', 'start_station_id', 'end_station_name','end_station_id']
    df.drop(columns=deleted_columns, inplace=True)
    df.dropna(inplace=True)

    return df

@task(log_prints=True)
def train_best_model(X_train, y_train, X_test, y_test):

    models = {
    "Linear Regression": LinearRegression(),
    # "Ridge": Ridge(),
    # "Lasso": Lasso(),
    # "Elastic Net": ElasticNet(),
    # "Decision Tree": DecisionTreeRegressor(),
    # "SVR": SVR(),
    # "Gradient Boosting": GradientBoostingRegressor(),
    }

    mlflow.autolog()
    
    for name, model in models.items():
        with mlflow.start_run(run_name=''):

            # mlflow.set_tag("Model_name", name)

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            mse = mean_squared_error(y_test, predictions)
            
            print(f"{name}: Model trained and logged with MSE: {mse}")

@task(log_prints=True)
def register_best_model(year, month):
    client = MlflowClient()
    experiment_id = client.get_experiment_by_name(f"Bike-Rides Duration Prediction: {year}-{month}").experiment_id
    runs = client.search_runs(experiment_id, order_by=["metrics.mse DESC"], max_results=1)
    best_run_id = runs[0].info.run_id

    best_run_details = client.get_run(best_run_id)
    best_run_artifact_uri = best_run_details.info.artifact_uri

    model_name = f"Best Model: {year}-{month}"

    try: 
        client.get_registered_model(model_name)
    except:
        client.create_registered_model(model_name)

    client.create_model_version(f"Best Model: {year}-{month}", source=best_run_artifact_uri ,run_id=best_run_id)
    latest_version = client.get_latest_versions(model_name)[0]
    client.transition_model_version_stage(name=model_name, version=latest_version.version, stage="Production", archive_existing_versions=True)


@flow(log_prints=True)
def main_flow(year: str="2023", month: str="01"):
    data_path = f"s3://mlops-personal-project/Traning-Data/{year}{month}-capitalbikeshare-tripdata.csv"
    print('Reading data: ', data_path)
    df = read_data(data_path)
    print('Data read successfully')
    processed_df = preprocessing(df)

    X = processed_df.drop(columns='duration')
    y = processed_df['duration']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_tracking_uri(uri="http://ec2-16-171-38-100.eu-north-1.compute.amazonaws.com:5000")
    mlflow.set_experiment(experiment_name=f"Bike-Rides Duration Prediction: {year}-{month}")

    print('Training best model...')
    train_best_model(X_train, y_train, X_test, y_test)
    print('Model trained and logged successfully!')

    print('Register the best model...')
    register_best_model(year, month)
    print('Model registered successfully!')


if __name__ == "__main__":
    main_flow()
        
    

