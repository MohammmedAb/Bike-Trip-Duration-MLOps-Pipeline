from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import DataDriftTable, RegressionQualityMetric, DatasetMissingValuesMetric
import psycopg
from datetime import datetime, timedelta
from prefect import task, flow


load_dotenv()
AWS_INSTANCE = os.getenv('AWS_INSTANCE')
DB_USERNAME = os.getenv('DB_USERNAME')
DB_PASSWORD = os.getenv('DB_PASSWORD')

db_connection_string = f"host={AWS_INSTANCE} port=5432 dbname='monitoring' user={DB_USERNAME} password={DB_PASSWORD}"

#
# column_mapping = ColumnMapping()

# column_mapping.target = 'duration'
# column_mapping.prediction = 'prediction'
# column_mapping.numerical_features = ['distance']
# column_mapping.categorical_features = ['rideable_type_docked_bike', 'rideable_type_electric_bike', 'member_casual_member']

# model_performance = Report(metrics=[DataDriftTable(), RegressionQualityMetric(), DatasetMissingValuesMetric()])

# reference_df = pd.read_parquet('s3://mlops-personal-project/reference_dataset/reference_df.parquet')




# model_performance.run(current_data=df_last_prediction_input, reference_data=reference_df, column_mapping=column_mapping)

@task
def fetch_yesterdays_data():

    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    start_of_yesterday = datetime.combine(yesterday, datetime.min.time())
    end_of_yesterday = datetime.combine(yesterday, datetime.max.time())
    query = """
    SELECT id, prediction_input, predicted_value, actual_value FROM predictions
    WHERE prediction_time BETWEEN %s AND %s;
    """

    try:
        with psycopg.connect(db_connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (start_of_yesterday, end_of_yesterday))
                
                rows = cur.fetchall()

                return rows

    except Exception as e:
        print(f"An error occurred: {e}")

@task
def create_current_df(rows):
    data = []

    for row in rows:
        # Unpack the values from the row
        id, prediction_input, predicted_value, actual_value = row

        # Extract the column names and data from the prediction_input dictionary
        columns = prediction_input['columns']
        row_data = prediction_input['data'][0]

        # Append the id, predicted_value, and actual_value to the row_data
        row_data.extend([predicted_value, actual_value])

        # Create a dictionary with column names as keys and the row data as values
        row_dict = dict(zip(columns + ['prediction', 'duration'], row_data))

        # Append the row dictionary to the data list
        data.append(row_dict)

    # Create a DataFrame from the data list with desired column names
    df = pd.DataFrame(data, columns=columns + ['prediction', 'duration'])
    return df




@flow
def batch_metrics_calculation_flow():

    row_yesterday_data = fetch_yesterdays_data()
    current_df = create_current_df(row_yesterday_data)

    

    print(current_df)

if __name__ == "__main__":
    batch_metrics_calculation_flow()
