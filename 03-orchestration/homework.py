import pickle

import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

from datetime import date
from dateutil import relativedelta

@task
def get_paths(given_date):
    logger = get_run_logger()

    if given_date == None:
        given_date = date.today()
    else:
        given_date = date.fromisoformat(given_date)
    
    training_date = given_date - relativedelta.relativedelta(months=2)
    training_year = str(training_date.year)
    training_month = str(training_date.month) if training_date.month >= 10 else "0" + str(training_date.month)
    train_path = f"../data/fhv_tripdata_{training_year}-{training_month}.parquet"
    logger.info(f"Train data path: {train_path}")

    val_date = given_date - relativedelta.relativedelta(months=1)
    val_year = str(val_date.year)
    val_month = str(val_date.month) if val_date.month >= 10 else "0" + str(val_date.month)
    val_path = f"../data/fhv_tripdata_{val_year}-{val_month}.parquet"
    logger.info(f"Val data path: {val_path}")

    return train_path, val_path

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        # print(f"The mean duration of training is {mean_duration}")
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        # print(f"The mean duration of validation is {mean_duration}")
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):
    logger = get_run_logger()

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    # print(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The shape of X_train is {X_train.shape}")
    # print(f"The DictVectorizer has {len(dv.feature_names_)} features")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    # print(f"The MSE of training is: {mse}")
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()

    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    # print(f"The MSE of validation is: {mse}")
    logger.info(f"The MSE of validation is: {mse}")
    return

@flow(task_runner=SequentialTaskRunner())
def main(date = None):
    train_path, val_path = get_paths(date).result()
    
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)

    # save outputs
    with open(f"outputs/model-{date}.bin", "wb") as model_file:
        pickle.dump(lr, model_file)

    with open(f"outputs/dv-{date}.b", "wb") as dv_file:
        pickle.dump(dv, dv_file)

# main(date = "2021-08-15")

DeploymentSpec(
    flow = main,
    name = "homework-section3",
    schedule=CronSchedule(
        cron="0 9 15 * *",
        timezone="America/Sao_Paulo"),
    flow_runner = SubprocessFlowRunner(),
    tags = ["homework"]
)

