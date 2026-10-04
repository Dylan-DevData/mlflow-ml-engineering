import sys
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score

run_id = sys.argv[1]  # reçu en paramètre, pas en dur

df =  pd.read_csv("raw_data/heart.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# On charge précisément le modèle produit par l'étape 1
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model_sante")

with mlflow.start_run():
    y_pred = model.predict(X_test)
    mlflow.log_metrics({
        "eval_accuracy": accuracy_score(y_test, y_pred),
        "eval_recall": recall_score(y_test, y_pred),
        "eval_f1": f1_score(y_test, y_pred)
    })
    print("Évaluation terminée et journalisée.")