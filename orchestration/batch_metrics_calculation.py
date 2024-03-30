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

@task
def get_date(n_day= None):
    """
    get the start and end datetime
    """
    if n_day != None:
        today = datetime.now().date()
        n_day = today - timedelta(days=n_day)
        start_of_yesterday = datetime.combine(n_day, datetime.min.time())
        end_of_yesterday = datetime.combine(n_day, datetime.max.time())
    else:
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        start_of_yesterday = datetime.combine(yesterday, datetime.min.time())
        end_of_yesterday = datetime.combine(yesterday, datetime.max.time())
    
    return start_of_yesterday, end_of_yesterday

@task
def fetch_yesterdays_data(start_of_yesterday, end_of_yesterday):
    """
    fetch data from the predictions table between the given start and end datetime.
    """


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
    """
    Create a df from the fetched data that contains the prediction_input, predicted_value, and actual_value
    """

    data = []
    main_df = pd.DataFrame()

    for row in rows:
        id, prediction_input, predicted_value, actual_value = row

        row_columns = prediction_input['columns']
        row_data = prediction_input['data'][0]

        row_dict = dict(zip(row_columns, row_data))

        
        row_dict['prediction'] = predicted_value
        row_dict['duration'] = actual_value

        data.append(row_dict)

        column_order = row_columns + ['duration', 'prediction']

        # Create a DataFrame for the current row with the desired column order
        df = pd.DataFrame([row_dict], columns=column_order)

        # Concatenate the current row DataFrame to the main DataFrame
        main_df = pd.concat([main_df, df], ignore_index=True)

    return main_df

@task(log_prints=True)
def get_reference_df(current_df):
    """
    Get the reference df and sample it to mathch the size of the current df 
    """

    print('Getting reference df...')
    reference_df = pd.read_parquet('s3://mlops-personal-project/reference_dataset/reference_df.parquet')

    reference_df = reference_df.sample(n=len(current_df))

    print('Finished getting the refrence df!')

    return reference_df

@task(log_prints=True)
def calculate_model_metrics(current_df, reference_df):
    """
    Calculate model metrices using Evidently
    """

    print('Calculating model metrics...')

    column_mapping = ColumnMapping()
    column_mapping.target = 'duration'
    column_mapping.prediction = 'prediction'
    column_mapping.numerical_features = ['distance']
    column_mapping.categorical_features = ['rideable_type_docked_bike', 'rideable_type_electric_bike', 'member_casual_member']

    model_performance = Report(metrics=[DataDriftTable(), RegressionQualityMetric(), DatasetMissingValuesMetric()])
    model_performance.run(current_data=current_df, reference_data=reference_df, column_mapping=column_mapping)

    metrics = {
        'r2_score': model_performance.as_dict()['metrics'][1]['result']['current']['r2_score'],
        'rmse': model_performance.as_dict()['metrics'][1]['result']['current']['rmse'],
        'mean_error': model_performance.as_dict()['metrics'][1]['result']['current']['mean_error'],
        'share_missing_values': model_performance.as_dict()['metrics'][2]['result']['current']['share_of_missing_values'],
        'number_of_drifted_columns': model_performance.as_dict()['metrics'][0]['result']['number_of_drifted_columns']
    }

    print(f'Calculated metrics: {metrics}')

    return metrics

@task(log_prints=True)
def insert_metrics_to_db(metrics, computation_time: datetime= datetime.now()):
    """
    insert the calculated metrices into the database
    """


    try:
        with psycopg.connect(db_connection_string) as conn:
            with conn.cursor() as cur:
                query = """
                INSERT INTO metrics (
                    r2_score, rmse, mean_error, share_of_missing_values,
                    number_of_drifted_features, computation_time
                ) VALUES (
                    %s, %s, %s, %s, %s, %s
                )
                """
                values = (
                    metrics['r2_score'], metrics['rmse'], metrics['mean_error'],
                    metrics['share_missing_values'], metrics['number_of_drifted_columns'],
                    computation_time
                )
                cur.execute(query, values)
                conn.commit()
                print("Metrics inserted successfully.")
    except Exception as e:
        print(f"An error occurred while inserting metrics: {e}")

@flow
def batch_metrics_calculation_flow():

    start_of_yesterday, end_of_yesterday = get_date()

    row_yesterday_data = fetch_yesterdays_data(start_of_yesterday, end_of_yesterday)
    current_df = create_current_df(row_yesterday_data)

    reference_df = get_reference_df(current_df)

    metrics = calculate_model_metrics(current_df, reference_df)
    insert_metrics_to_db(metrics, computation_time=start_of_yesterday)

    # print(current_df)

if __name__ == "__main__":
    batch_metrics_calculation_flow()
