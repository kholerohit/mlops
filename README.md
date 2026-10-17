# MLOps: diabetes progression model

This project trains a diabetes progression regression model, tracks runs with MLflow, explores the dataset in a Jupyter notebook, and serves predictions through a small FastAPI web UI. The dataset features and target are for research and learning; predictions are not medical diagnoses.

## Install

```bash
python -m pip install -r requirements.txt
```

## Train

Start a local MLflow tracking server in one terminal:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

In another terminal, run:

```bash
python train.py
```

## Explore the data

Open `diabetes-1.ipynb` in Jupyter or VS Code and run its cells from top to bottom. It includes distributions, skewness, correlations, and feature-versus-target plots. An optional `ydata-profiling` HTML report can be generated from the notebook and is ignored by Git.

## Run

```bash
python -m pip install -r requirements.txt
uvicorn app:app --reload
```

Open <http://127.0.0.1:8000>. FastAPI docs are at <http://127.0.0.1:8000/docs>.

The service finds the newest local model under `mlruns/` or `mlartifacts/`. To select one directly, set `MODEL_PATH` to its MLflow `artifacts` directory before starting Uvicorn.

## Input format

Send all 10 numeric features as JSON to `POST /predict`:

```json
{"age": 0.04, "sex": 0.05, "bmi": 0.06, "bp": 0.02, "s1": -0.04, "s2": -0.03, "s3": 0.01, "s4": -0.02, "s5": 0.03, "s6": 0.01}
```

Inputs follow the normalized feature format of scikit-learn's diabetes dataset. The output is a progression score, not a medical diagnosis.
