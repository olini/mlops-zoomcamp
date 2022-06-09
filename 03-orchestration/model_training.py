import pickle
import numpy as np
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error

import mlflow

import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

def create_trip_duration_column(df):
    # trip duration
    df["trip_duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    # convert trip_duration to minutes
    df["trip_duration"] = df["trip_duration"].apply(lambda x: x.total_seconds()/60)

    return df

def create_pickup_plus_dropoff_id(df):
    df["PU_DO_ID"] = df["PULocationID"].astype(str) + "_" + df["DOLocationID"].astype(str)

    return df

def filter_duration(df):
    df = df[(df.trip_duration >= 1) & (df.trip_duration <= 60)]

    return df

def prep_data_for_modelling(train_data_path = "../data/green_tripdata_2021-01.parquet",
                            val_data_path = "../data/green_tripdata_2021-02.parquet"):
    df_train = pd.read_parquet(train_data_path)
    df_val = pd.read_parquet(val_data_path)

    df_train = create_trip_duration_column(df_train)
    df_val = create_trip_duration_column(df_val)

    df_train = create_pickup_plus_dropoff_id(df_train)
    df_val = create_pickup_plus_dropoff_id(df_val)

    # due to some strange trip durations and to filter some data, lets filter based on trip_duration
    df_train = filter_duration(df_train)
    df_val = filter_duration(df_val)

    # select only some features for the example
    categorical_features = ["PU_DO_ID"]
    numerical_features = ["trip_distance"]

    # tranform the categorical features with one hot encoding
    # first we transform the categorial features in str type
    df_train[categorical_features] = df_train[categorical_features].astype(str)
    # then we transform the df into a matrix using DictVectorizer, 
    # which does the OHE with cat features
    train_dicts = df_train[categorical_features + numerical_features].to_dict(orient="records")
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical_features + numerical_features].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    y_train = df_train.trip_duration.values
    y_val = df_val.trip_duration.values

    return X_train, X_val, y_train, y_val, dv

def hyperparameters_opt(train_data, val_data, y_val):
    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "xgboost")
            mlflow.log_params(params)
            
            booster = xgb.train(
                params = params,
                dtrain = train_data,
                num_boost_round=300,
                evals=[(val_data, "validation")],
                early_stopping_rounds = 50
            )

            y_pred = booster.predict(val_data)
            rmse = mean_squared_error(y_val, y_pred, squared = False)
            mlflow.log_metric("rmse", rmse)
        
        return {"loss": rmse, "status": STATUS_OK}

    search_space = {
        "max_depth": scope.int(hp.quniform("max_depth", 4, 100, 1)),
        "learning_rate": hp.loguniform("learning_rate", -3, 0),
        "reg_alpha": hp.loguniform("reg_alpha", -5, -1),
        "reg_lambda": hp.loguniform("reg_lambda", -6, -1),
        "min_child_weight": hp.loguniform("min_child_weight", -1, 3),
        "objective": "reg:linear",
        "seed": 42
    }

    best_result = fmin(
        fn = objective,
        space = search_space,
        algo = tpe.suggest,
        max_evals = 1,
        trials = Trials()
    )

    return

def train_best_model(train_data, val_data, y_val, dv):
    with mlflow.start_run():
        best_params = {
            "learning_rate": 0.3658500268351324,
            "max_depth": 30,
            "min_child_weight": 1.6787939344371205,
            "objective": "reg:linear",
            "reg_alpha": 0.1247872970730739,
            "reg_lambda": 0.08527574907898512,
            "seed": 42
        }

        mlflow.log_params(best_params)
        mlflow.set_tag("model", "xgboost")
        mlflow.set_tag("version", "log_model")

        booster = xgb.train(
            params = best_params,
            dtrain = train_data,
            num_boost_round=1000,
            evals=[(val_data, "validation")],
            early_stopping_rounds = 50
        )

        y_pred_val = booster.predict(val_data)
        valid_rmse = mean_squared_error(y_val, y_pred_val, squared = False)

        mlflow.log_metric("valid_rmse", valid_rmse)
        print(f"Validation RMSE: {valid_rmse}")

        with open("../models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact("../models/preprocessor.b", artifact_path = "preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path = "models_mlflow")

if __name__ == "__main__":
    X_train, X_val, y_train, y_val, dv = prep_data_for_modelling()
    print(X_train.shape)
    print(X_val.shape)
    train_data = xgb.DMatrix(X_train, label = y_train)
    val_data = xgb.DMatrix(X_val, label = y_val)
    hyperparameters_opt(train_data, val_data, y_val)
    train_best_model(train_data, val_data, y_val, dv)