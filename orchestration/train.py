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
@
def read_data(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path)

