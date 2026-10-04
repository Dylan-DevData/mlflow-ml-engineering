# ML Engineering

Dépôt de travaux pratiques autour de l'entraînement de modèles scikit-learn et du suivi d'expériences avec MLflow. Les projets sont indépendants : chacun possède son propre fichier `MLproject`, son environnement et ses commandes de lancement.

## Contenu

- [`test_pratique/`](test_pratique/README.md) : comparaison de plusieurs modèles, sélection par F1 avec `mlflow.search_runs`, enregistrement du gagnant et évaluation.
- [`projets_v2/`](projets_v2/readme.md) : entraînement paramétrable d'un Random Forest.
- [`workflows_projects/`](workflows_projects/README.md) : workflow MLflow à deux étapes, entraînement puis évaluation.
- [`mlflow_projets/project/`](mlflow_projets/project/readme.md) : exemple MLflow Projects.
- [`src/ml_eng/`](src/ml_eng/) : squelette du paquet Python du dépôt.

## Prérequis

- Python 3.11 ou plus récent.
- MLflow et les dépendances du projet que tu souhaites lancer.
- Les jeux de données CSV attendus dans les dossiers `raw_data/`.

Pour installer les dépendances du paquet racine avec uv, exécute `uv sync`. Pour un projet MLflow autonome, installe les dépendances de son fichier `requirements.txt`, puis lance-le avec `--env-manager local` depuis son dossier.

## Exemple

```bash
cd test_pratique
python3 -m pip install -r requirements.txt
python3 workflow.py
```

Les bases MLflow, les runs et les modèles enregistrés localement sont des sorties générées. Les règles de `.gitignore` les excluent, tout en laissant les scripts, notebooks, fichiers de configuration et CSV suivis par Git.

## Données

Avant de publier les CSV, vérifie leurs conditions de redistribution et l'absence de données personnelles ou confidentielles.
