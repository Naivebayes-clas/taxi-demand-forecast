import sys, os
sys.path.insert(0, os.path.expanduser("~/github_docker/taxi-demand-forecast"))
import pandas as pd
import psycopg2
from config import *

def build_hourly_demand(ds: str):
    """Aggregate trips into hourly demand counts, store in Postgres."""
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER,
        password=DB_PASSWORD, dbname=DB_NAME
    )
    q = f"""
        SELECT
            date_trunc('hour', tpep_pickup_datetime) AS hour,
            COUNT(*) AS demand
        FROM taxi_trips
        WHERE tpep_pickup_datetime::date = '{ds}'
        GROUP BY 1
        ORDER BY 1
    """
    df = pd.read_sql(q, conn)
    df["hour"] = pd.to_datetime(df["hour"])
    df.to_sql("hourly_demand", conn, if_exists="append", index=False)
    conn.close()
    print(f"Built hourly demand series ({len(df)} rows) for {ds}")   
