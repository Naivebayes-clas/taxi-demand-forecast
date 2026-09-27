# NYC Taxi Demand Forecast

End-to-end time series forecasting pipeline for NYC Yellow Cab hourly pickups using **Apache Airflow 3**, **Prophet**, and **PostgreSQL**, with an interactive **Streamlit** dashboard.

## Screenshots

### Dashboard — Forecast Tab

| Forecast vs Actual |
|:---:|
| <img src="docs/screenshots/forecast_tab.png" width="800"> |
| *Last 7 days of actual demand + 72h Prophet forecast* |

### Dashboard — Demand Patterns Tab

| Day-of-Week Lines | Heatmap |
|:---:|:---:|
| <img src="docs/screenshots/dow_lines.png" width="400"> | <img src="docs/screenshots/heatmap.png" width="400"> |
| *Average hourly demand shape per day* | *Hour × Day intensity map* |

### Dashboard — Model Decomposition Tab

| Daily Pattern | How It Adds Up |
|:---:|:---:|
| <img src="docs/screenshots/daily_pattern.png" width="400"> | <img src="docs/screenshots/add_up.png" width="400"> |
| *Twin peaks: AM + PM rush* | *Trend + Weekly + Daily = Forecast* |


## Architecture
   
┌──────────────────┐ ┌─────────────────────────┐ ┌──────────────────────┐
│ NYC Open Data │           │ Airflow DAG │            | Postgres │
│ (S3 / Parquet) │──────▶                │ │──────▶│ │
│                 │ │ 1. ingest_nyc_taxi     │ │ • taxi_trips │
│ yellow_tripdata │ │ 2. build_hourly_demand │ │ • hourly_demand │
│ _YYYY-MM.parquet│ │ 3. train_and_forecast │ │ • demand_forecasts │
└──────────────────┘ └─────────────────────────┘ └──────────┬───────────┘
│
┌────────▼───────────┐
│ Streamlit App │
│ (this dashboard) │
└────────────────────┘


## Tech Stack

| Layer | Tool |
|-------|------|
| Orchestration | Apache Airflow 3 (LocalExecutor) |
| Database | PostgreSQL 16 (Docker) |
| ML / Forecasting | Prophet (weekly + daily seasonality, log transform) |
| Data Format | Parquet (columnar, ~10x smaller than CSV) |
| Visualization | Plotly + Streamlit |
| Data Source | NYC TLC Yellow Cab Trip Records |
| Python | 3.13 |

## How It Works

### 1. Ingest
Downloads the monthly TLC Parquet file from NYC Open Data S3, filters to a single day, and loads into Postgres. Idempotent — deletes the day's rows before inserting.

### 2. Transform
Aggregates individual trips into **hourly demand counts** (`COUNT(*)` per hour). Stores in `hourly_demand`.

### 3. Forecast
Fits **Prophet** on the full historical hourly series:
- **Log transform** (`log1p`) to prevent negative predictions
- **Weekly seasonality** (weekend vs weekday patterns)
- **Daily seasonality** (rush hour twin peaks)
- **No yearly seasonality** (insufficient data < 2 years)

Produces a **72-hour forecast** with 95% confidence intervals.

### 4. Dashboard
Interactive Streamlit app with:
- **Forecast tab** — actual vs predicted with confidence band
- **Demand Patterns tab** — day-of-week shapes, heatmap, weekly trend
- **Decomposition tab** — how Prophet splits the signal into trend + weekly + daily

## Project Structure
   
taxi-demand-forecast/
├── dags/
│ └── taxi_demand_dag.py # Airflow DAG (3 tasks)
├── app/
│ └── streamlit_app.py # Interactive dashboard
├── data/
│ └── raw/ # Cached Parquet files (.gitignore)
├── docs/
│ └── screenshots/ # Dashboard screenshots
├── run_forecast.py # One-shot script (bypasses Airflow)
├── requirements.txt
├── docker-compose.yml
└── README.md


## Setup

### Prerequisites
- Python 3.10+
- Docker
- Conda (optional)

### 1. Start Postgres

```bash
docker run -d --name taxi-postgres \
  -e POSTGRES_USER=taxi_user \
  -e POSTGRES_PASSWORD=your_password \
  -e POSTGRES_DB=taxi_db \
  -p 5432:5432 \
  postgres:16   


2. Install dependencies

conda create -n taxi python=3.10 -y
conda activate taxi
pip install -r requirements.txt   

3. Configure Airflow
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="postgresql+psycopg2://taxi_user:your_password@localhost:5432/airflow"
airflow db migrate

4. Start Airflow
airflow standalone

UI: http://localhost:8080

5. Run the forecast
Option A: One-shot (fastest)

python run_forecast.py 2025-01
python run_forecast.py 2025-02
python run_forecast.py 2025-03

Option B: Via Airflow

airflow dags trigger taxi_demand_forecast --execution-date 2025-01-15

6. Launch the dashboard
streamlit run app/streamlit_app.py

UI: http://localhost:8501

Results
With 3 months of data (Jan–Mar 2025, ~2,200 hours):

Metric	Value
Avg hourly demand	~5,150 trips
Peak hour	6 PM (evening rush)
Peak demand	~13,900 trips/hr
Trough	4–5 AM (~200 trips/hr)
Forecast horizon	72 hours
Forecast range	2,500 – 8,700 trips/hr

The model captures:

Daily twin peaks (AM rush ~7–9 AM, PM rush ~4–7 PM)
Weekly pattern (slightly different weekend shape)
Stable trend (no growth/decline over the 3-month window)



Roadmap
[ ] Add more months (6–12) for stronger weekly signal
[ ] Add weather regressors (temperature, precipitation)
[ ] Add holiday/event regressors
[ ] Zone-level forecasting (PULocationID)
[ ] Sliding-window evaluation (MAE vs naive baseline)
[ ] Deploy to Streamlit Cloud with hosted Postgres (Neon)
[ ] Alerting on forecast anomalies
Data Source
NYC TLC Yellow Cab Trip Records — public domain, updated monthly with ~2 week lag.



