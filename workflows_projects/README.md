# Projet `workflows_projects`

Ce projet illustre un workflow MLflow en deux étapes : l'entraînement d'un Random Forest, puis l'évaluation du modèle correspondant à son `run_id`. Les métriques d'évaluation sont journalisées dans un run MLflow.

## Prérequis

Python 3.11, les dépendances de `requirements.txt` et `raw_data/heart.csv` sont requis :

```bash
python3 -m pip install -r requirements.txt
```

## Lancer le workflow

Depuis ce dossier, l'orchestrateur actuellement nommé `worflow.py` lance les entry points `train` et `evaluate` :

```bash
python3 worflow.py
```

Les étapes peuvent aussi être lancées séparément :

```bash
mlflow run . --entry-point train --experiment-name heath_prediction --env-manager local \
  -P max_depth=10 -P n_estimators=200
```

```bash
mlflow run . --entry-point evaluate --experiment-name heath_prediction --env-manager local \
  -P run_id=<RUN_ID>
```

Remplace `<RUN_ID>` par l'identifiant du run d'entraînement. MLflow stocke les expériences et artefacts dans son backend local.