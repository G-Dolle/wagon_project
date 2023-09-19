import os
import pandas as pd
from google.cloud import bigquery

from ml_logic.data_import import get_divvy_data

def load_month_data(year):


    if year <2022:

        year = str(year)
        df_Q1 = pd.DataFrame(get_divvy_data(year,"Q1"))
        df_Q2 = pd.DataFrame(get_divvy_data(year,"Q2"))
        df_Q3 = pd.DataFrame(get_divvy_data(year,"Q3"))
        df_Q4 = pd.DataFrame(get_divvy_data(year,"Q4"))

        df_full_year = pd.concat([df_Q1,df_Q2,df_Q3,df_Q4])

    else:
        year = str(year)
        df_Q1 = pd.DataFrame(get_divvy_data(year,"Q1"))
        df_Q2 = pd.DataFrame(get_divvy_data(year,"Q2"))

        df_full_year = pd.concat([df_Q1,df_Q2])

    df_full_year['started_at']=pd.to_datetime(df_full_year['started_at'])
    df_full_year['hourly_data'] = df_full_year.started_at.dt.round('60min')

    df_full_year.hourly_data = pd.to_datetime(df_full_year["hourly_data"],
                                format="%Y-%m-%d %H:%M:%S UTC",
                                utc=True)

    df_full_year['year'] = pd.DatetimeIndex(df_full_year['hourly_data']).year
    df_full_year['month'] = pd.DatetimeIndex(df_full_year['hourly_data']).month

    df_full_year = df_full_year.drop(columns=["hourly_data"])

    month_list =  list(df_full_year["month"])
    month_list_fin = []
    [month_list_fin.append(item) for item in month_list if item not in month_list_fin]


    PROJECT = os.environ.get("PROJECT")
    DATASET = os.environ.get("DATASET")
    is_first= True

    for i in month_list_fin:

        df_reduced =  df_full_year[df_full_year["month"]==i]

        table = f"M{i}_{year}"
        table = f"{PROJECT}.{DATASET}.{table}"

        df_reduced.columns = [f"_{column}" if type(column) != str else column for column in df_reduced.columns]

        client = bigquery.Client()

        # define write mode and schema
        write_mode = "WRITE_TRUNCATE" if is_first else "WRITE_APPEND"
        job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

        print(f"\n{'Write' if is_first else 'Append'} {table} ({df_reduced.shape[0]} rows)")

        # load data
        job = client.load_table_from_dataframe(df_reduced, table, job_config=job_config)
        result = job.result()  # wait for the job to complete
