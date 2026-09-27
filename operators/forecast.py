import sys, os
sys.path.insert(0, os.path.expanduser("~/github_docker/taxi-demand-forecast"))
import pandas as pd
import psycopg2
from config import *

def train_and_forecast(ds: str):
    """Fit Prophet on historical hourly demand, predict next 72h."""
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER,
        password=DB_PASSWORD, dbname=DB_NAME
    )
    df = pd.read_sql("SELECT hour, demand FROM hourly_demand ORDER BY hour", conn)
    conn.close()

    # Prophet requires columns: ds, y
    df = df.rename(columns={"hour": "ds", "demand": "y"})

    from prophet import Prophet
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
    )
    model.fit(df)

    # Forecast
    future = model.make_future_dataframe(periods=FORECAST_HORIZON, freq="H")
    preds = model.predict(future)

    # Persist predictions
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER,
        password=DB_PASSWORD, dbname=DB_NAME
    )
    preds[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_sql(
        "demand_forecasts", conn, if_exists="append", index=False
    )
    conn.close()
    print(f"Forecasted {FORECAST_HORIZON}h ahead. Last 3 predictions:")
    print(preds[["ds", "yhat"]].tail(3).to_string(index=False))   
