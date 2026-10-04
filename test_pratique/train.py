from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EXPERIENCE_NAME = "cas pratique"
MODEL_NAME = "model_sante"
RANDOM_STATE = 42


def main():
    data_path = Path(__file__).resolve().parent / "raw_data" / "heart.csv"
    df = pd.read_csv(data_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    models = {
        "sgd": SGDClassifier(random_state=RANDOM_STATE, max_iter=2000),
        "lr": LogisticRegression(random_state=RANDOM_STATE, max_iter=2000),
        "rf": RandomForestClassifier(random_state=RANDOM_STATE),
        "gaussian_nb": GaussianNB(),
    }

    mlflow.set_experiment(EXPERIENCE_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIENCE_NAME)
    with mlflow.start_run(run_name="model_selection") as run:
        parent_run_id = run.info.run_id
        for model_name, estimator in models.items():
            pipeline = make_pipeline(StandardScaler(), estimator)
            with mlflow.start_run(run_name=model_name, nested=True):
                pipeline.fit(X_train, y_train)
                predictions = pipeline.predict(X_val)
                score = f1_score(y_val, predictions, zero_division=0)
                mlflow.log_param("model_name", model_name)
                mlflow.log_metric("validation_f1", score)
                mlflow.sklearn.log_model(
                    pipeline,
                    artifact_path="candidate_model",
                    skops_trusted_types=["sklearn.tree._tree.Tree"],
                )
            print(f"{model_name}: validation F1 = {score:.4f}")

        candidate_runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
            order_by=["metrics.validation_f1 DESC"],
            max_results=1,
        )
        if candidate_runs.empty:
            raise RuntimeError("No completed candidate model runs were found.")

        best_run = candidate_runs.iloc[0]
        best_name = best_run["params.model_name"]
        best_score = best_run["metrics.validation_f1"]
        best_pipeline = make_pipeline(StandardScaler(), models[best_name])
        best_pipeline.fit(X_train_val, y_train_val)

        mlflow.log_metric("best_validation_f1", best_score)
        mlflow.log_param("selected_model", best_name)
        mlflow.set_tag("selected_model", best_name)
        mlflow.sklearn.log_model(
            best_pipeline,
            artifact_path=MODEL_NAME,
            registered_model_name=MODEL_NAME,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(
            f"Selected {best_name} (validation F1={best_score:.4f}); "
            f"model registered as {MODEL_NAME}."
        )
        print(f"Training run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()






