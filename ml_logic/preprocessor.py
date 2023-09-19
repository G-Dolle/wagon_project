import math
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
import joblib


class TimeFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        hourly_data = pd.to_datetime(X["date_time"], format="%Y-%m-%d %H:%M:%S UTC", utc=True)
        hourly_data = hourly_data.dt.tz_convert("America/Chicago").dt
        dow = hourly_data.weekday
        hour = hourly_data.hour
        month = hourly_data.month
        hour_sin = np.sin(2 * math.pi / 24 * hour)
        hour_cos = np.cos(2 * math.pi / 24 * hour)
        result = np.stack([hour_sin, hour_cos, dow, month], axis=1)
        return result


def create_sklearn_preprocessor() -> ColumnTransformer:
    time_pipe = make_pipeline(
        TimeFeaturesTransformer(),
        make_column_transformer(
            (OneHotEncoder(
                categories=[[i for i in range(0, 7, 1)], [i for i in range(1, 13, 1)]],
                sparse=False,
                handle_unknown="ignore"
            ), [2, 3]),
            remainder="passthrough"
        )
    )
    weather_pipe = make_pipeline(StandardScaler())
    weather_features = ["temp", "pressure", "humidity", "wind_speed", "wind_deg", "clouds_all"]
    cat_transformer = OneHotEncoder(sparse=False, handle_unknown="ignore")

    final_preprocessor = ColumnTransformer(
        [
            ("time_preproc", time_pipe, ["date_time"]),
            ("weather_scaler", weather_pipe, weather_features),
            ("station_name_encoding", cat_transformer, ["station_name"])
        ],
        n_jobs=-1,
    )
    return final_preprocessor


def preprocess_features(X: pd.DataFrame, filename) -> np.ndarray:
    preprocessor = create_sklearn_preprocessor()
    #X_processed_fit = preprocessor.fit(X)
    X_processed = preprocessor.fit_transform(X)
    X_processed_df = pd.DataFrame(X_processed)
    joblib.dump(preprocessor, filename)
    #s=pickle.dumps(preprocessor)
    return preprocessor, X_processed_df

def load_preprocessor(filename):
    #with open(filename, "rb") as file:
        #preprocessor = joblib.load(file)
        #preprocessor = pickle.loads(file)
    #preprocessor = pickle.loads(filename)
    preprocessor = joblib.load(filename)
    return preprocessor
