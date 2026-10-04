# Projet `test_pratique`

Workflow de classification sur `raw_data/heart.csv`. L'entraînement compare SGD, régression logistique, Random Forest et Gaussian Naive Bayes. Chaque candidat est évalué sur un jeu de validation; `mlflow.search_runs` sélectionne le meilleur F1. Le modèle gagnant est réentraîné sur train et validation, puis enregistré sous le nom `model_sante`. L'étape suivante calcule les métriques finales sur le jeu de test.

## Prérequis

Python 3.11 et les dépendances de `requirements.txt` sont nécessaires :

```bash
python3 -m pip install -r requirements.txt
```

## Exécution complète

Depuis ce dossier :

```bash
python3 workflow.py
```

Le workflow lance les entry points `train` puis `eval`, et transmet à l'évaluation l'identifiant du run d'entraînement. Les scores candidats sont des métriques de validation; `eval_f1`, `eval_recall` et `eval_accuracy` sont calculés ensuite sur le jeu de test.

## Exécution séparée

Lancer l'entraînement :

```bash
mlflow run . --entry-point train --experiment-name "cas pratique" --env-manager local
```

Puis évaluer un run précis :

```bash
mlflow run . --entry-point eval --experiment-name "cas pratique" --env-manager local \
  -P run_id=<RUN_ID>
```

Remplace `<RUN_ID>` par l'identifiant renvoyé par l'entraînement. MLflow conserve les runs et versions du modèle dans le backend configuré localement.