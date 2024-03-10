import pandas as pd
import os
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV

import mlflow
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.tracking import MlflowClient
from mlflow.models import ModelSignature, infer_signature

from prefect import task, Flow

@task(retries=3, retry_delay_seconds=2, log_prints=True)
def read_data(data_path: str) -> pd.DataFrame:
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

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
    # "Linear Regression": LinearRegression(),
    # "Ridge": Ridge(),
    # "Lasso": Lasso(),
    "Elastic Net": ElasticNet(),
    # "Decision Tree": DecisionTreeRegressor(),
    # "SVR": SVR(),
    # "Gradient Boosting": GradientBoostingRegressor(),
    }

    mlflow.autolog()
    
    for name, model in models.items():
        with mlflow.start_run(run_name='auto log'):

            mlflow.set_tag("Model_name", name)

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            mse = mean_squared_error(y_test, predictions)
            
            print(f"{name}: Model trained and logged with MSE: {mse}")


@Flow
def main_flow(data_path: str):
    df = read_data(data_path)
    processed_df = preprocessing(df)

    X = processed_df.drop(columns='duration')
    y = processed_df['duration']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_tracking_uri(uri="http://ec2-16-171-38-100.eu-north-1.compute.amazonaws.com:5000")
    mlflow.set_experiment(experiment_name="Bike-Rides")

    train_best_model(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main_flow(data_path="s3://mlops-personal-project/Traning-Data/2023/202301-capitalbikeshare-tripdata.csv")
        
    

