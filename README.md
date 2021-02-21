# Projet de Recherche Opérationnelle et Données Massives (SOD322) - ENSTA Paris 

## Pour commencer
Projet de Recherche Opérationnelle et Données Massives (SOD322) pour l'ENSTA Paris réalisé par **Ilyes EL-RAMMACH** et **Laurent LAM**. 

Implémentation d'un algorithme de classification supervisée ([An Integer Optimization Approach to Associative Classification](https://www.mit.edu/~dbertsim/papers/Machine%20Learning%20under%20a%20Modern%20Optimization%20Lens/An%20Integer%20Optimization%20Approach%20to%20Associative%20Classification.pdf) par Bertsimas, Chang et Rudin) via un traitement des données et une binarisation des features discriminantes afin d'accélerer le processus de classification.

### Pré-requis
Les langages utilisées dans ce projet sont:

* [Julia](https://julialang.org/downloads/) (v.1.4.1)
* [Python](https://www.python.org/downloads/) (v.3.8.5)

Les dépendances additionnelles pour Julia sont les modules suivants:

* [CPLEX](https://juliapackages.com/p/cplex) (v.12.10.0)
* [JuMP](https://juliapackages.com/p/jump)

Les modules nécessaires pour Python sont précisés dans le `requirements.txt` et peuvent être installés via la commande:

```
pip install -r requirements.txt
```

### Structure du module
La structure du module est présentée ici avec pour exemple un jeu de données nommé `DATASET_NAME`.

- `rodm_ensta`/
- |__ `data`/
- |__ `docker`/
- |__ `res`/
- |__ `src`/
- |__ `.gitignore`
- |__ `README.md`
- |__ `requirements.txt`

#### État initial

- `data`/
- |__ `DATASET_NAME.csv`

#### État intermédiaire/final


- `data`/
- |__ `DATASET_NAME.csv`
- |__ `DATASET_NAME_train.csv`
- |__ `DATASET_NAME_test.csv`

- `res`/
- |__ `DATASET_NAME_feature_markers.json`
- |__ `DATASET_NAME_rules.csv`
- |__ `DATASET_NAME_ordered_rules.csv`

### Pipeline & Commandes

Tout le projet peut être dirigé à partir d'un unique script _bash_ via le fichier `src/pipeline.sh`.

La commande pour lancer le projet est donc:
```
./src/pipeline.sh -d DATASET_NAME -f -r -s
```

Les différents *flags* sont:

- ```-d DATASET_NAME``` **nécessaire**, indique le **nom du dataset** à traiter
- ```-f``` **optionnel**, permet d'activer le traitement du dataset brut, via un pre-processing et une binarisation des **features**, et conduit à la création des **training** et **testing** sets.
*Ne pas indiquer ce flag conduit à supposer l'existence au préalable des fichiers CSV _train_ et _test_ dans le dossier `data`*
- ```-r``` **optionnel**, permet d'activer l'étape de **génération de règles**. 
*Ne pas indiquer ce flag conduit à supposer l'existence au préalable du fichier de règles dans le dossier `res`*
- ```-s``` **optionnel**, permet d'activer l'étape de **tri et filtre des règles**.
*Ne pas indiquer ce flag conduit à supposer l'existence au préalable du fichier de règles triées dans le dossier `res`*
