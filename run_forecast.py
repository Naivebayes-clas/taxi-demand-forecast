import sys, os, urllib.request
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

DB_CONN = "postgresql+psycopg2://taxi_user:your_password@localhost:5432/taxi_db"
DATA_DIR = os.path.expanduser("~/github_docker/taxi-demand-forecast/data/raw")
FORECAST_HORIZON = 72

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_forecast.py [YYYY-MM]")
        sys.exit(1)

    month = sys.argv[1]
    year, mon = map(int, month.split("-"))
    start = f"{year}-{mon:02d}-01"
    end = f"{year+1}-01-01" if mon == 12 else f"{year}-{mon+1:02d}-01"

    engine = create_engine(DB_CONN)

    # Download
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = f"yellow_tripdata_{month}.parquet"
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{filename}"
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, filepath)

    # Read + aggregate
    print("Reading parquet...")
    df = pd.read_parquet(filepath, columns=["tpep_pickup_datetime"])
    df["hour"] = df["tpep_pickup_datetime"].dt.floor("h")
    hourly = df.groupby("hour").size().reset_index(name="demand")
    print(f"{len(hourly)} hours of data for {month}")

    # Delete this month from hourly_demand, then append
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM hourly_demand WHERE hour >= :s AND hour < :e"),
            {"s": start, "e": end}
        )
    hourly.to_sql("hourly_demand", engine, if_exists="append", index=False)

    # Train on all history
    print("Training Prophet on all available history...")
    all_data = pd.read_sql("SELECT hour, demand FROM hourly_demand ORDER BY hour", engine)
    all_data = all_data.rename(columns={"hour": "ds", "demand": "y"})
    all_data["y"] = np.log1p(all_data["y"])
    print(f"Total training rows: {len(all_data)}")

    from prophet import Prophet
    model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True)
    model.fit(all_data)

    # Forecast
    future = model.make_future_dataframe(periods=FORECAST_HORIZON, freq="h")
    preds = model.predict(future)
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        preds[col] = np.expm1(preds[col]).clip(lower=0)

    preds = preds.tail(FORECAST_HORIZON)

    # Write forecast (DELETE first, then insert — no TRUNCATE)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM demand_forecasts"))
    preds[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_sql(
        "demand_forecasts", engine, if_exists="append", index=False
    )

    print(f"\nDone. {FORECAST_HORIZON}h forecast.")
    print(preds[["ds", "yhat"]].tail(5).to_string(index=False))

if __name__ == "__main__":
    main()   
