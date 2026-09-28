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


Lessons Learned: Bugs & Fixes

Real-world debugging log from building this project. Every bug here was hit in production (well, localhost).

### Airflow 3 Breaking Changes

| Bug | Error | Fix |
|-----|-------|-----|
| `schedule_interval` parameter removed | `TypeError: DAG.__init__() got an unexpected keyword argument 'schedule_interval'` | Use `schedule="@daily"` instead |
| `airflow webserver` command removed | `Command 'airflow webserver' has been removed` | Use `airflow api-server` |
| `airflow users create` removed | `Positional Arguments: GROUP_OR_COMMAND` | Airflow 3 uses SimpleAuthManager — any username/password works on first login |
| `airflow dags backfill` removed | `Command 'dags backfill' has been removed` | Use `airflow backfill create --dag-id X --from-date X --to-date X` |
| `airflow backfill create` different args | `the following arguments are required: --dag-id, --from-date, --to-date` | Airflow 3 uses `--dag-id`, `--from-date`, `--to-date` (not positional, no `--yes`) |
| `airflow backfills list` doesn't exist | `invalid choice: 'backfills'` | Only `airflow backfill create` exists in v3. Cancel via UI or SQL |
| Port conflict on restart | `[Errno 98] address already in use` | `pkill -f -u <user> airflow` before restarting. Set port in `airflow.cfg` permanently |
| SQLite "database is locked" | `sqlite3.OperationalError: database is locked` | `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN` env var wasn't active in the shell running `airflow standalone`. Fix: put `sql_alchemy_conn` in `airflow.cfg` |

### Python / Pandas / Prophet

| Bug | Error | Fix |
|-----|-------|-----|
| Uppercase `H` frequency removed in Pandas 2.x | `ValueError: Invalid frequency: H. Did you mean h?` | Use `freq="h"` (lowercase) |
| `.str.startswith()` on datetime64 column | `AttributeError: Can only use .str accessor with string values, not datetime64` | Use `.dt.strftime("%Y-%m-%d") == ds` |
| 2025+ TLC Parquet dropped lat/long columns | `No match for FieldRef.Name(pickup_longitude)` | Columns now use `PULocationID`/`DOLocationID` (zone IDs). For time series, only need `tpep_pickup_datetime` |
| `model.predict()` returns ALL rows (history + future) | Table has 2,237 rows instead of 72 | Add `preds = preds.tail(FORECAST_HORIZON)` before `to_sql` |
| Negative predictions from Prophet | `yhat = -41,879` (impossible for demand) | Apply `np.log1p(y)` before fit, `np.expm1()` after predict, `.clip(lower=0)` |
| `idxmax()` returns index label, not position | `KeyError: 35` | Use `int(np.argmax(series.values))` for positional index |
| `TRUNCATE` silently blocked by table lock | Table still has old rows after "successful" run | Use `DELETE FROM table` instead (doesn't require exclusive lock) |

### Airflow Task Execution

| Bug | Error | Fix |
|-----|-------|-----|
| `sys.path` doesn't propagate to task runner | `ModuleNotFoundError: No module named 'operators'` | Inline functions in DAG file, or add `sys.path.insert(0, ...)` at top of each operator file |
| Relative paths (`./data/raw`) fail in task runner | File not found (CWD is `~/airflow/`, not project dir) | Use absolute paths: `os.path.expanduser("~/github_docker/taxi-demand-forecast/data/raw")` |
| `if_exists="append"` causes duplicates on re-run | Duplicate rows → Prophet sees conflicting values → garbage predictions | Delete that day's rows before inserting: `DELETE FROM table WHERE date = :ds` |
| `dags_folder` not pointing to project | DAG not appearing in UI | Set `dags_folder = /path/to/project/dags` in `airflow.cfg` |
| Stale backfill blocks new backfill | `AlreadyRunningBackfill: Another backfill is running` | `DELETE FROM dag_run WHERE dag_id='X'; DELETE FROM backfill WHERE dag_id='X';` in Airflow metadata DB |
| Airflow 3 `standalone` not dispatching tasks to workers | Tasks stuck in "queued" forever, workers show `<idle>` | Known Airflow 3 bug. Workaround: run `airflow scheduler` and `airflow api-server` separately, or use `airflow tasks test` for single runs |

### Plotly / Streamlit

| Bug | Error | Fix |
|-----|-------|-----|
| `arrowhead="up"` invalid | `Invalid value of type 'str' received for 'arrowhead'` | Use integer: `arrowhead=1` (1=open up, 2=filled up, 4=triangle) |
| `make_subplots` mangles datetime x-axis | X-axis shows 2008–2024 instead of 2025 | Use plain `go.Figure()` per subplot, or use integer x-axis (hour index) |
| Datetime objects misinterpreted by Plotly | X-axis shows wrong dates regardless of format | Use `st.line_chart()` (Streamlit native) for datetime data. Reserve Plotly for integer/categorical x-axes |
| `use_container_width` deprecated | `Please replace use_container_width with width` | Use `width='stretch'` (string, not boolean) |
| `width=True` invalid | Blank page, no error | Must be a string: `width='stretch'` |
| `@st.cache_data` serves stale data | Table updated but app shows old values | `pkill streamlit` and restart, or call `st.cache_data.clear()` |
| `st.line_chart` color list length mismatch | `StreamlitColorLengthError: must have same length as columns` | Use list matching column count: `color=["#f5a623", "#0f3460"]` (one per column) |
| Duplicate columns after DataFrame join | `['Actual', 'Forecast', 'Forecast']` (3 cols, 2 colors) | `merged = merged.loc[:, ~merged.columns.duplicated()]` |

### Docker / System

| Bug | Error | Fix |
|-----|-------|-----|
| `pkill -f airflow` kills Docker container processes | `Operation not permitted` (root-owned PIDs) | Use `pkill -f -u <username> airflow` to only kill your processes |
| Docker Airflow container conflicting with native install | High CPU, port conflicts, ghost processes | `docker stop <container>`. Only run ONE Airflow (native OR Docker) |
| Postgres container stopped | `Connection refused` on port 5432 | `docker start taxi-postgres` |
| `createdb` prompts for password in loop | Hangs waiting for input | `PGPASSWORD=your_password createdb ...` or `docker exec taxi-postgres psql -U taxi_user -c "CREATE DATABASE airflow;"` |

### Debugging Strategies That Worked

1. **`airflow tasks test <dag> <task> <date>`** — runs a single task in the terminal with full traceback. Bypasses the scheduler/executor entirely. This is the #1 debugging tool in Airflow.

2. **Check the data, not the code** — when Prophet produced negative predictions, the bug wasn't in the model. It was duplicate rows in `hourly_demand` from `if_exists="append"`. Always `SELECT COUNT(*)` and `GROUP BY ... HAVING COUNT(*) > 1` first.

3. **Kill everything, verify, restart** — when Airflow state is corrupted (stale backfills, locked tables, ghost processes):
   ```bash
   pkill -9 -f -u <user> airflow
   sleep 3
   ps aux | grep airflow | grep -v grep  # verify empty
   # Fix DB state
   # Restart clean
   airflow standalone   

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



