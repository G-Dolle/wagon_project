import os
import pandas as pd
import numpy as np
import requests
import urllib.parse
from ml_logic.cleaning import weather_cleaning_v2
from datetime import datetime
from ml_logic.preprocessor import TimeFeaturesTransformer

def chicago_weather_forecast():
    '''Return a 5-day weather forecast for the city of Chicago'''

    BASE_URI="https://api.openweathermap.org"

    api_key = os.environ.get("WEATHER_API_KEY")

    url=urllib.parse.urljoin(BASE_URI, "/data/2.5/forecast")
    forecasts=requests.get(url, params={'lat': 41.87, 'lon': -87.62, 'units': 'metric','appid':api_key}).json()

    forecasts_list = forecasts["list"]

    return forecasts_list



def convert_chicago_forecast_todf(forecasts:list):
    """
    Return a pre-preprocessed dataframe for the 5-day
    weather forecast of Chicago

    """

    def rename_keys(dico,new_keys):
        '''
        Allows to replace keys' names in a dictionary with new ones
        '''
        tmp = dict( zip( list(dico.keys()), new_keys) )
        result = {tmp[oldK]: value for oldK, value in dico.items()}

        return result

    def slice_cleaning(forecasts,slice):
        '''
        Separates the different dictionaries within the json file returned by the
        weather API and concatenates these into a single dictionary
        '''
        one_obs = forecasts[slice]
        main = one_obs["main"]
        weather = one_obs["weather"][0]
        clouds = one_obs["clouds"]
        wind = one_obs["wind"]

        dt_txt = one_obs["dt_txt"]
        visibility = one_obs["visibility"]


        new_keys_weather = ["weather_id","weather_main","weather_description","weather_icon"]
        new_keys_cloud =["clouds_all"]
        new_keys_wind =['wind_speed', 'wind_deg','wind_gust']

        weather_clean = rename_keys(weather, new_keys_weather)
        clouds_clean = rename_keys(clouds, new_keys_cloud)
        wind_clean = rename_keys(wind, new_keys_wind)

        dall = {}
        dall["dt_iso"] = dt_txt
        for d in [main, weather_clean, wind_clean ,clouds_clean]:
            dall.update(d)

        dall["visibility"] = visibility

        return dall

    # Storing these dictionaries into a list
    list_of_slices=[]

    for i in range(0,len(forecasts)):

        dall = slice_cleaning(forecasts,i)

        list_of_slices.append(dall)

    # Converting this list of dictionaries into a dataframe
    forecast_df = pd.DataFrame.from_dict(list_of_slices)

    return forecast_df

def clean_forecast(df):
    '''
    returns a cleaned weather forecast dataframe
    '''
    cleaned_df = weather_cleaning_v2(df)

    return cleaned_df

def get_right_forecast(departure_date,departure_time,df):
    """
    Return the closest hourly weather forecast to the date and time inputs
    provided by the end-user
    """

    full_time_input= datetime.combine(departure_date,departure_time)
    df["user_input"] = pd.to_datetime(full_time_input)
    df["date_input"]=df["user_input"].dt.date
    df["date_weather"]=df["date_time"].dt.date

    df_reduc = df[df["date_weather"]==df["date_input"]]


    df_reduc["time_diff"] = df_reduc["user_input"] - df_reduc["date_time"]
    df_reduc=df_reduc[df_reduc["time_diff"]>pd.Timedelta(0)]
    cond = df_reduc["time_diff"].min()
    new_data = df_reduc[df_reduc["time_diff"]==cond]
    new_data.drop(columns=["user_input","date_input","date_weather","time_diff"], inplace=True)

    return new_data


def process_weather_inputs(departure_date,departure_time):
    forecasts=chicago_weather_forecast()
    forecast_df=convert_chicago_forecast_todf(forecasts)
    forecast_cleaned=clean_forecast(forecast_df)
    new_data=get_right_forecast(departure_date,departure_time, forecast_cleaned)
    return new_data
