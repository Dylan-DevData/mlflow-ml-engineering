# Projet `projets_v2`

Ce projet entraîne un classifieur Random Forest sur `raw_data/heart.csv`. MLflow suit l'expérience `heath_prediction` et enregistre le modèle sous le nom `model_heath`.

## Prérequis

Python 3.11 et les dépendances de `requirement.txt` sont nécessaires. Depuis la racine du dépôt :

```bash
cd projets_v2
python3 -m pip install -r requirement.txt
```

## Exécution avec MLflow

La commande se lance depuis ce dossier, où se trouvent `MLproject` et `raw_data/heart.csv` :

```bash
mlflow run . --entry-point main \
	--experiment-name heath_prediction \
	--env-manager local \
	-P max_depth_param=10 \
	-P n_estimators_param=200
```

`max_depth_param` règle la profondeur maximale des arbres et `n_estimators_param` leur nombre. Le point d'entrée `main` est déclaré dans `MLproject`.

`--env-manager local` réutilise l'environnement Python déjà actif; il ne crée ni n'installe cet environnement. MLflow conserve les runs et modèles dans le backend configuré localement.
commmande CLI  :


mlflow run . --entry-point main  --experiment-name "heath_prediction" --env-manager=local -P max_depth_param=10  -P n_estimators=200

python :

python  script.py
