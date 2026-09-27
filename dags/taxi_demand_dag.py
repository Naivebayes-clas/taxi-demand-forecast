import sys, os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

DB_CONN = "postgresql+psycopg2://taxi_user:your_password@localhost:5432/taxi_db"
DATA_DIR = os.path.expanduser("~/github_docker/taxi-demand-forecast/data/raw")
FORECAST_HORIZON = 72


def ingest_nyc_taxi(ds: str):
    import urllib.request
    os.makedirs(DATA_DIR, exist_ok=True)
    year, month, day = ds.split("-")
    filename = f"yellow_tripdata_{year}-{month}.parquet"
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{filename}"
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, filepath)
    df = pd.read_parquet(filepath, columns=[
        "tpep_pickup_datetime", "passenger_count", "trip_distance"
    ])
    df = df[df["tpep_pickup_datetime"].dt.strftime("%Y-%m-%d") == ds]
    if df.empty:
        print(f"No data for {ds}, skipping.")
        return
    engine = create_engine(DB_CONN)
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM taxi_trips WHERE tpep_pickup_datetime::date = :ds"),
            {"ds": ds}
        )
    df.to_sql("taxi_trips", engine, if_exists="append", index=False, chunksize=10_000)
    print(f"Ingested {len(df)} trips for {ds}")


def build_hourly_demand(ds: str):
    engine = create_engine(DB_CONN)
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM hourly_demand WHERE hour::date = :ds"),
            {"ds": ds}
        )
    q = f"""
        SELECT date_trunc('hour', tpep_pickup_datetime) AS hour, COUNT(*) AS demand
        FROM taxi_trips
        WHERE tpep_pickup_datetime::date = '{ds}'
        GROUP BY 1 ORDER BY 1
    """
    df = pd.read_sql(q, engine)
    if df.empty:
        print(f"No trips for {ds}, skipping.")
        return
    df["hour"] = pd.to_datetime(df["hour"])
    df.to_sql("hourly_demand", engine, if_exists="append", index=False)
    print(f"Built hourly demand series ({len(df)} rows) for {ds}")


def train_and_forecast(ds: str):
    engine = create_engine(DB_CONN)
    df = pd.read_sql("SELECT hour, demand FROM hourly_demand ORDER BY hour", engine)
    if df.empty:
        print("No historical data, skipping forecast.")
        return
    df = df.rename(columns={"hour": "ds", "demand": "y"})

    # Log transform to prevent negative predictions
    df["y"] = np.log1p(df["y"])

    from prophet import Prophet
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
    )
    model.fit(df)
    future = model.make_future_dataframe(periods=FORECAST_HORIZON, freq="h")
    preds = model.predict(future)

    # Invert log transform
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        preds[col] = np.expm1(preds[col])
    preds["yhat"] = preds["yhat"].clip(lower=0)
    preds["yhat_lower"] = preds["yhat_lower"].clip(lower=0)

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE demand_forecasts"))
    preds[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_sql(
        "demand_forecasts", engine, if_exists="append", index=False
    )
    print(f"Forecasted {FORECAST_HORIZON}h ahead.")
    print(preds[["ds", "yhat"]].tail(3).to_string(index=False))


default_args = {"owner": "you", "retries": 1, "retry_delay": timedelta(minutes=5)}

with DAG(
    dag_id="taxi_demand_forecast",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["taxi", "forecast"],
) as dag:
    ingest = PythonOperator(
        task_id="ingest_nyc_taxi",
        python_callable=ingest_nyc_taxi,
        op_kwargs={"ds": "{{ ds }}"},
    )
    transform = PythonOperator(
        task_id="build_hourly_demand",
        python_callable=build_hourly_demand,
        op_kwargs={"ds": "{{ ds }}"},
    )
    forecast = PythonOperator(
        task_id="train_and_forecast",
        python_callable=train_and_forecast,
        op_kwargs={"ds": "{{ ds }}"},
    )
    ingest >> transform >> forecast   
