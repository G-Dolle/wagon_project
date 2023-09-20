import os
import pandas as pd
import numpy as np
import xgboost as xgb

import pickle as pkl

def initialize_model():
    model_en=xgb.XGBRegressor()
    return model_en

def train_model(model: xgb.XGBRegressor(),
                X: pd.DataFrame,
                y: pd.DataFrame,
                arr_dep):
    model.fit(X,y)
    #saved_model = pkl.dumps(model)

    if arr_dep =="arrivals":
        filename = 'models/model_arr.sav'
        pkl.dump(model, open(filename, 'wb'))

    elif arr_dep =="departures":
        filename = 'models/model_dep.sav'
        pkl.dump(model, open(filename, 'wb'))

    return model

def load_model(arr_dep):

    if arr_dep =="arrivals":
        filename = 'models/model_arr.sav'
        loaded_model = pkl.load(open(filename, 'rb'))

    elif arr_dep =="departures":
        filename = 'models/model_dep.sav'
        loaded_model = pkl.load(open(filename, 'rb'))

    return loaded_model


def predict(model: xgb.XGBRegressor(),
                   X: pd.DataFrame):
    y_pred= model.predict(X)
    return y_pred

def score(model: xgb.XGBRegressor(),
                   X: pd.DataFrame,
                   y: pd.DataFrame):
    score=model.score(X,y)
    score = float(score)
    return score

def availability (arrivals: float,
           departures: float):
    if arrivals/departures<0.6:
        return 0
    else:
        return 1
