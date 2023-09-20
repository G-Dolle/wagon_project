import os
import pandas as pd
import numpy as np
import math

from ml_logic.data_import import get_weather_data, get_station_data, get_cloud_year_data, get_local_data
from ml_logic.cleaning import clean_divvy_new,weather_cleaning_v2, features_target_v2, features_target_test, station_list_to_csv, time_features
from ml_logic.preprocessor import preprocess_features
from ml_logic.model import initialize_model, train_model, predict, score
from sklearn.metrics import mean_absolute_error



def preprocess():

    # Import data

    #quarter= os.environ.get("DIVVY_QUARTER")
    #year= os.environ.get("DIVVY_YEAR")

    data_source = os.environ.get("SOURCE")
    year= os.environ.get("DIVVY_TRAIN_YEAR_GCP")

    print(data_source)

    raw_divvy_df = get_local_data(year)
    print("Divvy Raw data imported from local disk")

    raw_weather_df = get_weather_data()

    #station_df = get_station_data()

    print("Raw data imported")

    # Clean data & merge data

    df_dep_hourly, df_arr_hourly = clean_divvy_new(raw_divvy_df)
    clean_weather_df = weather_cleaning_v2(raw_weather_df)

    print("Data cleaned")

    # Create features and target dataframes

    features_dep_df, features_arr_df, target_dep_df, target_arr_df, station_name_list = features_target_v2(df_arr_hourly, df_dep_hourly, clean_weather_df)

    #features_dep_df_comp = time_features(features_dep_df)
    #features_arr_df_comp = time_features(features_arr_df)

    station_list_to_csv(raw_divvy_df,treshold=3000)

    print("features and target dataframes created")

    # preprocess features

    preprocessor_dep, X_dep_processed_df = preprocess_features(features_dep_df,"preprocessors/pipeline_dep.joblib")
    preprocessor_arr, X_arr_processed_df = preprocess_features(features_arr_df,"preprocessors/pipeline_arr.joblib")

    print("Preprocessing of Training set is done")

    return X_dep_processed_df, X_arr_processed_df, preprocessor_dep, preprocessor_arr, target_dep_df, target_arr_df, station_name_list, clean_weather_df


# preprocessing a test set

def preprocess_test(preprocessor_dep, preprocessor_arr, station_name_list, clean_weather_df):

    # Import data
    quarter= os.environ.get("DIVVY_QUARTER_TEST")
    year= os.environ.get("DIVVY_YEAR_TEST")

    #raw_divvy_df = get_divvy_data(year,quarter)

    data_source = os.environ.get("SOURCE")


    year = 2022

    table_name=f"Divvy_Trips_{year}_Q1.csv"

    path = r'raw_data/2022'

    file_path = path+"/"+table_name


    df_test_set = pd.read_csv(file_path)


    table_name=f"Divvy_Trips_{year}_Q2.csv"
    file_path = path+"/"+table_name

    #path = os.path.join(os.path.expanduser(file_path))

    df_q2 = pd.read_csv(file_path)

    df_test_set = pd.concat([df_test_set,df_q2])


    #raw_weather_df = get_weather_data()

    print("Test Raw data imported")


    df_dep_test_hourly, df_arr_test_hourly = clean_divvy_new(df_test_set)
    features_dep_test_df, features_arr_test_df, target_dep_test_df, target_arr_test_df = features_target_test(df_arr_test_hourly, df_dep_test_hourly, clean_weather_df, station_name_list=station_name_list)

    #features_dep_test_df_comp = time_features(features_dep_test_df)
    #features_arr_test_df_comp = time_features(features_arr_test_df)

    X_dep_test_processed = preprocessor_dep.transform(features_dep_test_df)
    X_arr_test_processed = preprocessor_arr.transform(features_arr_test_df)

    print("Preprocessing of Test set is done")

    return X_dep_test_processed, X_arr_test_processed, target_dep_test_df, target_arr_test_df, station_name_list


# train model

def training_model(X_dep_processed_df,target_dep_df, X_arr_processed_df, target_arr_df):

    # Create an instance of the XGBoost regressor
    xgb_regressor = initialize_model()

    # Fit the model on the training data
    model_dep = train_model(xgb_regressor,X_dep_processed_df, target_dep_df,arr_dep="departures")

    # Create an instance of the XGBoost regressor
    xgb_regressor = initialize_model()

    # Fit the model on the training data
    model_arr = train_model(xgb_regressor,X_arr_processed_df, target_arr_df,"arrivals")

    return model_dep, model_arr

# evaluate model

def eval():

    score_dep = score(model_dep,X_dep_processed_df,target_dep_df)
    score_arr = score(model_arr,X_arr_processed_df,target_arr_df)

    print(f"XGBoost score for departures: {score_dep:.4f}")
    print(f"XGBoost score for arrivals: {score_arr:.4f}")

# Predict
def predict_function(model_dep, model_arr, X_dep_test_processed, X_arr_test_processed, target_dep_test_df, target_arr_df):

    y_pred_dep = predict(model_dep,X_dep_test_processed)
    y_pred_arr = predict(model_arr,X_arr_test_processed)

    mse_xgboost_dep = mean_absolute_error(target_dep_test_df, y_pred_dep)
    mse_xgboost_arr = mean_absolute_error(target_arr_test_df, y_pred_arr)

    print(f"XGBoost MAE for departures: {mse_xgboost_dep:.4f}")
    print(f"XGBoost MAE for arrivals: {mse_xgboost_arr:.4f}")



if __name__ == '__main__':
    target_chosen = os.environ.get("TARGET_CHOSEN")
    X_dep_processed_df, X_arr_processed_df, preprocessor_dep, preprocessor_arr, target_dep_df, target_arr_df, station_name_list, clean_weather_df = preprocess()
    X_dep_test_processed, X_arr_test_processed, target_dep_test_df, target_arr_test_df, station_name_list=preprocess_test(preprocessor_dep, preprocessor_arr, station_name_list, clean_weather_df)
    model_dep, model_arr = training_model(X_dep_processed_df,target_dep_df, X_arr_processed_df, target_arr_df)
    eval()
    predict_function(model_dep, model_arr, X_dep_test_processed, X_arr_test_processed, target_dep_test_df, target_arr_df)
