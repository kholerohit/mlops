import mlflow
import mlflow.sklearn
import optuna
import numpy as np
import dagshub

from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


dagshub.init(repo_owner="kholerohit", repo_name="mlops", mlflow=True)
mlflow.set_experiment("diabetes-hyperparameter-optimization")

# Load data and reserve an untouched test set
data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 310),
        "max_depth": trial.suggest_int("max_depth", 2, 22),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        "random_state": 42,
        "n_jobs": -1,
    }

    model = RandomForestRegressor(**params)
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=1,
    )
    cv_rmse = float(np.sqrt(-scores).mean())

    # Store each trial in MLflow as a nested run
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        mlflow.log_params(params)
        mlflow.log_metric("cv_rmse", cv_rmse)

    return cv_rmse


# Parent run groups all tuning trials
with mlflow.start_run(run_name="optuna-study"):
    mlflow.log_param("optimizer", "Optuna TPE")
    mlflow.log_param("trials", 30)
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("objective", "minimize_cv_rmse")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=30)

    print("\nBest parameters:", study.best_params)
    print("Best CV RMSE:", study.best_value)

    # Train the selected configuration on all training data
    best_model = RandomForestRegressor(
        **study.best_params, random_state=42, n_jobs=-1
    )
    best_model.fit(X_train, y_train)

    # Evaluate only once on the held-out test set
    predictions = best_model.predict(X_test)
    test_mse = mean_squared_error(y_test, predictions)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, predictions)
    test_r2 = r2_score(y_test, predictions)

    mlflow.log_params(
        {f"best_{key}": value for key, value in study.best_params.items()}
    )
    mlflow.log_metrics(
        {
            "best_cv_rmse": study.best_value,
            "test_rmse": test_rmse,
            "test_mse": test_mse,
            "test_mae": test_mae,
            "test_r2": test_r2,
        }
    )

    mlflow.sklearn.log_model(
        sk_model=best_model,
        name="best_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    print("\nFinal test results")
    print(f"RMSE: {test_rmse:.3f}")
    print(f"MSE:  {test_mse:.3f}")
    print(f"MAE:  {test_mae:.3f}")
    print(f"R²:   {test_r2:.3f}")
    print("Best model saved to MLflow.")