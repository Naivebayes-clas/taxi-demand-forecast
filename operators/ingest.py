import sys, os
sys.path.insert(0, os.path.expanduser("~/github_docker/taxi-demand-forecast"))
import os
import pandas as pd
import psycopg2
from config import *

def ingest_nyc_taxi(ds: str):
    """Load raw TLC parquet for the given date into Postgres."""
    os.makedirs(NYC_TLC_DATA_DIR, exist_ok=True)

    # If you already have files locally, skip download.
    # Otherwise pull from NYC Open Data S3:
    # https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet
    # (adjust month/year to match ds)
    year, month, day = ds.split("-")
    filename = f"yellow_tripdata_{year}-{month}.parquet"
    filepath = os.path.join(NYC_TLC_DATA_DIR, filename)

    if not os.path.exists(filepath):
        import urllib.request
        url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{filename}"
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, filepath)

    df = pd.read_parquet(filepath, columns=[
        "tpep_pickup_datetime", "tpep_dropoff_datetime",
        "pickup_longitude", "pickup_latitude", "passenger_count", "trip_distance"
    ])
    # Filter to the specific day
    df = df[df["tpep_pickup_datetime"].str.startswith(ds)]

    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER,
        password=DB_PASSWORD, dbname=DB_NAME
    )
    df.to_sql("taxi_trips", conn, if_exists="append", index=False, chunksize=10_000)
    conn.close()
    print(f"Ingested {len(df)} trips for {ds}")   
