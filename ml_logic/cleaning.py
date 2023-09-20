
import pandas as pd
import numpy as np
import datetime as dt
from ml_logic.data_import import get_station_data



# Cleaning the weather data

def weather_cleaning_v2(df):
    '''
    This function cleans the weather data by removing characters from the dt_iso variable
    to allow the creation of a datetime variable

    It also removes the duplicates and selects the relevant features

    '''

    features = ["dt_iso","temp","pressure","humidity","wind_speed","wind_deg","clouds_all"]

    df_clean_tmp=df[features]

    df_clean_tmp = df_clean_tmp.drop_duplicates()

    df_clean_tmp["dt_iso"] = df_clean_tmp["dt_iso"].apply(lambda x: x.replace("+0000 UTC", ""))

    df_clean_tmp["dt_iso"] = pd.to_datetime(df_clean_tmp["dt_iso"])
    df_clean_tmp['date_time']=df_clean_tmp['dt_iso']
    df_clean_tmp.drop(columns=['dt_iso'], inplace=True)

    return df_clean_tmp

# Cleaning bike rides data and creating dataframes for nb of departures and arrivals

def clean_divvy_new(df):

    df=df[df["rideable_type"]!="electric_bike"]
    df['started_at']=pd.to_datetime(df['started_at'])
    df['ended_at']=pd.to_datetime(df['ended_at'])
    df['hourly_data_started'] = df.started_at.dt.round('60min')
    df['hourly_data_ended'] = df.ended_at.dt.round('60min')

    df_departures=df[[
                    "start_station_name",
                    'hourly_data_started']]

    df_departures=df_departures.rename(columns={
                                                "start_station_name":"station_name",
                                                'hourly_data_started':'date_time'})

    df_departures["nb_departures"]=1

    df_dep_hourly=df_departures.groupby(by=["station_name",
                                        'date_time']).count().reset_index()




    df_arrivals=df[["end_station_name",
                 "hourly_data_ended"]]

    df_arrivals=df_arrivals.rename(columns={
                                            "end_station_name":"station_name",
                                            'hourly_data_ended':'date_time'})
    df_arrivals["nb_arrivals"]=1


    df_arr_hourly=df_arrivals.groupby(by=["station_name",
                                            'date_time']).count().reset_index()



    return df_dep_hourly, df_arr_hourly



def station_list_to_csv(df, treshold=3000):

    df=df[df["rideable_type"]!="electric_bike"]
    df['started_at']=pd.to_datetime(df['started_at'])
    df['ended_at']=pd.to_datetime(df['ended_at'])
    df['hourly_data_started'] = df.started_at.dt.round('60min')
    df['hourly_data_ended'] = df.ended_at.dt.round('60min')

    df_dep=df[[
                    "start_station_name",
                    'hourly_data_started',
                    "start_lat",
                    "start_lng"]]

    df_dep=df_dep.rename(columns={"start_station_name":"station_name"})

    df_dep['nb_rows'] = df_dep.groupby('station_name')['station_name'].transform('count')
    df_dep = df_dep[df_dep['nb_rows'] >= treshold]
    df_dep.drop(columns=['nb_rows'], inplace=True)

    station_df = df_dep[["station_name","start_lat","start_lng"]]


    station_df.to_csv("interface_api/data/station_list.csv", index=False)


# Creating the features and target dataframes for the training set

def features_target_v2(df_arr, df_dep, df_weather, treshold=3000):
    '''

    This function merges the Divvy datasets aggregated at the station and hourly level
    with the hourly weather cleaned data
    Then if creates the features and target dataframes,
    '''

    merged_dep_df = df_dep.merge(df_weather,
    how="left",
    on='date_time')

    merged_arr_df = df_arr.merge(df_weather, how='left', on='date_time')

    merged_dep_df['nb_rows'] = merged_dep_df.groupby('station_name')['station_name'].transform('count')
    merged_dep_df_red = merged_dep_df[merged_dep_df['nb_rows'] >= treshold]
    merged_dep_df_red.drop(columns=['nb_rows'], inplace=True)

    station_name_list = merged_dep_df_red.station_name.drop_duplicates().to_list()

    merged_arr_df_red = merged_arr_df[merged_arr_df['station_name'].isin(station_name_list)]

    features_dep_df = merged_dep_df_red.drop(columns=["nb_departures"])
    target_dep_df =  merged_dep_df_red["nb_departures"]

    features_arr_df = merged_arr_df_red.drop(columns=["nb_arrivals"])
    target_arr_df =  merged_arr_df_red["nb_arrivals"]


    return features_dep_df, features_arr_df, target_dep_df, target_arr_df, station_name_list

# Creating the features and target dataframes for departures and arrivals for the test set

def features_target_test(df_arr, df_dep, df_weather, station_name_list):
    '''

    This function merges the Divvy datasets aggregated at the station and hourly level
    with the hourly weather cleaned data
    Then if creates the features and target dataframes,
    '''

    merged_dep_df = df_dep.merge(df_weather,
    how="left",
    on='date_time')

    merged_arr_df = df_arr.merge(df_weather, how='left', on='date_time')


    merged_dep_df_red = merged_dep_df[merged_dep_df['station_name'].isin(station_name_list)]
    merged_arr_df_red = merged_arr_df[merged_arr_df['station_name'].isin(station_name_list)]

    features_dep_df = merged_dep_df_red.drop(columns=["nb_departures"])
    target_dep_df =  merged_dep_df_red["nb_departures"]

    features_arr_df = merged_arr_df_red.drop(columns=["nb_arrivals"])
    target_arr_df =  merged_arr_df_red["nb_arrivals"]


    return features_dep_df, features_arr_df, target_dep_df, target_arr_df


def time_features(X):
        import math
        X['date_time'] = pd.to_datetime(X['date_time'])
        X['date_time'] = X['date_time'].dt.tz_localize('UTC')
        X['date_time'] = X['date_time'].dt.tz_convert("America/Chicago")
        X['day_of_week'] = X['date_time'].dt.dayofweek
        X['hour'] = X['date_time'].dt.hour
        X['month'] = X['date_time'].dt.month
        X['hour_sin'] = np.sin(2 * math.pi / 24 *  X['hour'])
        X['hour_cos'] = np.cos(2 * math.pi / 24 *  X['hour'])
        X.drop(columns=['date_time', 'hour'], inplace=True)

        return X
