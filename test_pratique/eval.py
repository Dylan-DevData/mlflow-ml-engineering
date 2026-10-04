from pathlib import Path
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def main():
  run_id = sys.argv[1]
  print(f"Evaluating training run: {run_id}")

  data_path = Path(__file__).resolve().parent / "raw_data" / "heart.csv"
  df = pd.read_csv(data_path)
  X = df.iloc[:, :-1].values
  y = df.iloc[:, -1].values
  _, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
  )

  model = mlflow.sklearn.load_model(f"runs:/{run_id}/model_sante")
  predictions = model.predict(X_test)

  with mlflow.start_run(run_id=run_id):
    mlflow.log_metrics(
      {
        "eval_accuracy": accuracy_score(y_test, predictions),
        "eval_recall": recall_score(y_test, predictions, zero_division=0),
        "eval_f1": f1_score(y_test, predictions, zero_division=0),
      }
    )

  print("Evaluation complete; test metrics logged to the training run.")


if __name__ == "__main__":
  main()