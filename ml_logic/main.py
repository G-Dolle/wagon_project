import os
import pandas as pd
import numpy as np
import math

from ml_logic.data_import import get_weather_data, get_station_data, get_cloud_year_data, get_local_data
from ml_logic.cleaning import clean_divvy_new,weather_cleaning_v2, features_target_v2
from ml_logic.preprocessor import preprocess_features




def preprocess():

    # Import data

    #quarter= os.environ.get("DIVVY_QUARTER")
    #year= os.environ.get("DIVVY_YEAR")

    data_source = os.environ.get("SOURCE")
    year= os.environ.get("DIVVY_TRAIN_YEAR_GCP")

    if data_source == "gcp":

        raw_divvy_df = get_cloud_year_data(year)

        print("Divvy Raw data imported from cloud")

    if data_source == "local":
        raw_divvy_df = get_local_data(year)
        print("Divvy Raw data imported from local disk")

    raw_weather_df = get_weather_data()

    station_df = get_station_data()

    print("Raw data imported")

    # Clean data & merge data

    df_dep_hourly, df_arr_hourly = clean_divvy_new(raw_divvy_df)
    clean_weather_df = weather_cleaning_v2(raw_weather_df)

    print("Data cleaned")

    # Create features and target dataframes

    features_dep_df, features_arr_df, target_dep_df, target_arr_df, station_name_list = features_target_v2(df_arr_hourly, df_dep_hourly, clean_weather_df)

    print("features and target dataframes created")

    # preprocess features
    preprocessor, X_processed_df = preprocess_features(X)

    print("features preprocessed")

    # preprocess target

    preprocessor_dep, X_dep_processed_df = preprocess_features(features_dep_df,"bikes_available/preprocessors/pipeline_dep.joblib")
    preprocessor_arr, X_arr_processed_df = preprocess_features(features_arr_df,"bikes_available/preprocessors/pipeline_arr.joblib")

    print("Preprocessing of Training set is done")

    return X_dep_processed_df, X_arr_processed_df, preprocessor_dep, preprocessor_arr, target_dep_df, target_arr_df


# preprocessing a test set

def preprocess_test(preprocessor_dep, preprocessor_arr):

    # Import data
    quarter= os.environ.get("DIVVY_QUARTER_TEST")
    year= os.environ.get("DIVVY_YEAR_TEST")

    #raw_divvy_df = get_divvy_data(year,quarter)

    data_source = os.environ.get("SOURCE")

    if data_source == "gcp":

        raw_divvy_df = get_cloud_year_data(year)

        print("Divvy Raw data imported from cloud")

    if data_source == "local":


        raw_divvy_df = get_local_data(year)
        print("Divvy Raw data imported from local disk")


    raw_weather_df = get_weather_data()

    print("Test Raw data imported")

    # Clean data & merge data

    clean_divvy_df = cleaning_divvy_gen_agg(raw_divvy_df)
    clean_weather_df = weather_cleaning(raw_weather_df)

    merged_df = merge_divvy_weather(clean_divvy_df, clean_weather_df)

    print("Test Data cleaned and merged")

    # Create features and target dataframes

    X_test, y_test = features_target(merged_df, target_chosen)

    print("Test features and target dataframes created")

    # transform the features test set

    X_test_processed = preprocessor.transform(X_test)

    # preprocess target

    if target_chosen == "ratio":
        y_test_processed = target_process(y_test)
        print("ratio picked as target, and preprocessed")
    else:
        y_test_processed = y_test

        print(f"{target_chosen} picked as target")

    print("Preprocessing of test set is done")

    return X_test_processed, y_test_processed

# train model

# Save model

# Evaluate model

# predict


if __name__ == '__main__':
    target_chosen = os.environ.get("TARGET_CHOSEN")
    X_processed_df, y_processed_df, preprocessor = preprocess(target_chosen)
    X_test_processed, y_test_processed=preprocess_test(preprocessor, target_chosen)
