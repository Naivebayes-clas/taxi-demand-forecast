"""
One-shot: ingest full month, build hourly series, train Prophet, write forecast.
Usage: python run_forecast.py [YYYY-MM]
Example: python run_forecast.py 2025-01
"""
import sys
import os
import urllib.request
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# --- Config ---
DB_CONN = "postgresql+psycopg2://taxi_user:your_password@localhost:5432/taxi_db"
DATA_DIR = os.path.expanduser("~/github_docker/taxi-demand-forecast/data/raw")
FORECAST_HORIZON = 72  # hours ahead

# --- Main ---
def main():
    if len(sys.argv) < 2:
        print("Usage: python run_forecast.py [YYYY-MM]")
        print("Example: python run_forecast.py 2025-01")
        sys.exit(1)

    month = sys.argv[1]  # e.g. "2025-01"
    engine = create_engine(DB_CONN)

    # 1. Download
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = f"yellow_tripdata_{month}.parquet"
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{filename}"
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, filepath)

    # 2. Read + aggregate to hourly
    print("Reading parquet...")
    df = pd.read_parquet(filepath, columns=["tpep_pickup_datetime"])
    df["hour"] = df["tpep_pickup_datetime"].dt.floor("h")
    hourly = df.groupby("hour").size().reset_index(name="demand")
    print(f"{len(hourly)} hours of data")

    # 3. Store in Postgres (idempotent)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE taxi_trips"))
        conn.execute(text("TRUNCATE hourly_demand"))
        conn.execute(text("TRUNCATE demand_forecasts"))
    hourly.to_sql("hourly_demand", engine, if_exists="append", index=False)

    # 4. Train Prophet
    print("Training Prophet...")
    hourly = hourly.rename(columns={"hour": "ds", "demand": "y"})
    hourly["y"] = np.log1p(hourly["y"])

    from prophet import Prophet
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
    )
    model.fit(hourly)

    # 5. Forecast
    future = model.make_future_dataframe(periods=FORECAST_HORIZON, freq="h")
    preds = model.predict(future)

    # Invert log transform
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        preds[col] = np.expm1(preds[col]).clip(lower=0)

    # 6. Save
    preds[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_sql(
        "demand_forecasts", engine, if_exists="append", index=False
    )

    print(f"\nDone. Forecasting {FORECAST_HORIZON}h ahead.")
    print(preds[["ds", "yhat"]].tail(10).to_string(index=False))


if __name__ == "__main__":
    main()  
