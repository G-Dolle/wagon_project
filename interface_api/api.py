from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle as pkl
from datetime import datetime, time
import pandas as pd
import numpy as np
from interface_ui.ui_utils import process_weather_inputs
from ml_logic.cleaning import weather_cleaning_v2
from ml_logic.data_import import get_station_data
import joblib




model_dep_filename = "models/model_dep.sav"
preproc_dep_filename = "preprocessors/pipeline_dep.pickle"

model_arr_filename = "models/model_arr.sav"
preproc_arr_filename = "preprocessors/pipeline_arr.pickle"

app = FastAPI()
app.state.model_dep=pkl.load(open(model_dep_filename,'rb'))
app.state.model_arr=pkl.load(open(model_arr_filename,'rb'))
#app.state.prep_dep=load_preprocessor(preproc_dep_filename)
#app.state.prep_arr=load_preprocessor(preproc_arr_filename)

app.state.prep_dep=joblib.load(preproc_dep_filename)
app.state.prep_arr=joblib.load(preproc_arr_filename)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict(year_input,month_input,day_input,hour_input,min_input,sec_input):

    year_ui=int(year_input)
    month_ui=int(month_input)
    day_ui=int(day_input)
    hour_ui=int(hour_input)
    min_ui=int(min_input)
    sec_ui=int(sec_input)

    departure_date = datetime(year=year_ui,month=month_ui,day=day_ui).date()
    departure_time = time(hour_ui,min_ui,sec_ui)

    new_data = process_weather_inputs(departure_date,departure_time)
    new_data["key_for_matching"]="key_for_matching"

    stations_df = pd.DataFrame(pd.read_csv("interface_api/data/station_list.csv"))
    stations_df.rename(columns={"0":"station_name"}, inplace=True)
    stations_df_red = stations_df[["station_name"]]


    stations_df_red["key_for_matching"]="key_for_matching"

    for_predict_df = new_data.merge(stations_df_red, on="key_for_matching", how="outer")
    for_predict_df = for_predict_df.drop(columns=["key_for_matching"])

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

    for_predict_comp_df = time_features(for_predict_df)




    for_predict_df_dep = for_predict_comp_df[['station_name','temp','pressure','humidity','wind_speed','wind_deg','clouds_all','day_of_week','month','hour_sin','hour_cos']]
    for_predict_df_arr = for_predict_comp_df[['station_name','temp','pressure','humidity','wind_speed','wind_deg','clouds_all','day_of_week','month','hour_sin','hour_cos']]

    #model_dep=pkl.load(open(model_dep_filename,'rb'))
    #prep_dep=load_preprocessor(preproc_dep_filename)

    #model_arr = pkl.load(open(model_arr_filename,'rb'))
    #prep_arr = load_preprocessor(preproc_arr_filename)


    X_dep_processed=app.state.prep_dep.transform(for_predict_df_dep)
    X_arr_processed=app.state.prep_arr.transform(for_predict_df_arr)

    print("nb of Nans in X_dep_processed")
    print(pd.DataFrame(X_dep_processed).isnull().sum().sum())

    print("nb of Nans in X_arr_processed")
    print(pd.DataFrame(X_arr_processed).isnull().sum().sum())


    #Predict departures and arrivals
    departures=pd.DataFrame(app.state.model_dep.predict(X_dep_processed))
    stations_df_res_dep = stations_df.merge(departures, left_index=True, right_index=True)
    stations_df_res_dep.rename(columns={0:"nb_dep"}, inplace=True)
    stations_df_res_dep

    arrivals = pd.DataFrame(app.state.model_arr.predict(X_arr_processed))
    stations_df_complete = stations_df_res_dep.merge(arrivals, left_index=True, right_index=True)
    stations_df_complete.rename(columns={0:"nb_arr", "start_lat":"lat","start_lng":"lng"}, inplace=True)
    stations_df_complete

    output_dict=stations_df_complete.to_dict('list')

    return output_dict

@app.get("/")
def root():
    return {'greeting':'Hello'}
