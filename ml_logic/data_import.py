import os
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account


################################################################################
# Import weather, station and bike rides data from local file
################################################################################

# Import weather data from local file

def get_weather_data():

    path = os.environ.get("LOCAL_DATA_PATH_WEATHER")

    weather_df = pd.read_csv(path)

    return weather_df


# Import station data from local file

def get_station_data():

    #path = os.environ.get("LOCAL_DATA_PATH_STATION")

    path = "bikes_available/interface_api/data/station_API.csv"

    station_df = pd.read_csv(path)

    return station_df

# Import bike rides data from local file

def get_local_data(year):

    year=int(year)

    if year<2022:
        year=str(year)
        table_name=f"M1_{year}.csv"

        path = os.environ.get("LOCAL_DATA_PATH_DIVVY")

        file_path = path+"/"+table_name

        #path = os.path.join(os.path.expanduser(file_path))

        df_full_year = pd.read_csv(file_path)

        for i in range(2,13):

            table_name=f"M{i}_{year}.csv"
            file_path = path+"/"+table_name

            #path = os.path.join(os.path.expanduser(file_path))

            df_month = pd.read_csv(file_path)

            df_full_year = pd.concat([df_full_year,df_month])

    if year==2022:
        year=str(year)
        table_name=f"M1_{year}.csv"

        path = os.environ.get("LOCAL_DATA_PATH_DIVVY")

        file_path = path+"/"+table_name

        #path = os.path.join(os.path.expanduser(file_path))

        df_full_year = pd.read_csv(file_path)

        for i in range(2,7):

            table_name=f"M{i}_{year}.csv"
            file_path = path+"/"+table_name

            #path = os.path.join(os.path.expanduser(file_path))

            df_month = pd.read_csv(file_path)

            df_full_year = pd.concat([df_full_year,df_month])

    return df_full_year


################################################################################
# WIP - import bike rides data from cloud
################################################################################


# Import a given chunk of data from cloud
def get_cloud_chunk(table_name):
    credentials_defined = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

    credentials_fin = service_account.Credentials.from_service_account_file(
    credentials_defined)

    client = bigquery.Client(credentials=credentials_fin)
    PROJECT = os.environ.get("PROJECT")
    DATASET = os.environ.get("DATASET")

    table = f"{PROJECT}.{DATASET}.{table_name}"

    dataset_ref = bigquery.DatasetReference(PROJECT, DATASET)
    table_ref = dataset_ref.table(table_name)
    table = client.get_table(table_ref)

    df = client.list_rows(table).to_dataframe()

    return df

# Import year-long bike rides data from cloud
def get_cloud_year_data(year):

    year=int(year)

    if year<2022:
        year=str(year)
        table_name=f"M1_{year}"

        df_full_year= get_cloud_chunk(table_name)

        for i in range(2,13):

            table_name=f"M{i}_{year}"

            df_month = get_cloud_chunk(table_name)

            df_full_year = pd.concat([df_full_year,df_month])

    if year==2022:
        year=str(year)
        table_name=f"M1_{year}"

        df_full_year= get_cloud_chunk(table_name)

        for i in range(2,7):

            table_name=f"M{i}_{year}"

            df_month = get_cloud_chunk(table_name)

            df_full_year = pd.concat([df_full_year,df_month])

    return df_full_year
