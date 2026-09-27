import os

# Postgres (your Docker container)
DB_HOST = "localhost"
DB_PORT = 5432
DB_USER = "taxi_user"
DB_PASSWORD = "your_password"
DB_NAME = "taxi_db"

# Airflow metadata DB (same Postgres, different DB)
AIRFLOW_DB_CONN = "postgresql+psycopg2://airflow:airflow@localhost:5432/airflow"

# Data
NYC_TLC_DATA_DIR = "./data/raw"          # raw parquet/csv drops
CLEANED_DATA_DIR = "./data/processed"
MODEL_DIR = "./models"

# Forecast params
FORECAST_HORIZON = 72  # hours ahead
AGGREGATION = "1h"     # resample to hourly   
