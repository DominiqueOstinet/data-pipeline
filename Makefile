SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

PY := python

# 1) Dossier data
data:
	mkdir -p data

# 2) Données brutes
data/raw_dataset.csv: | data
	$(PY) -m src.load_data

# 3) Données nettoyées
data/clean_dataset.csv: data/raw_dataset.csv
	$(PY) -m src.clean_data

# 4) Données préprocessées + preprocessor.pkl
data/train.csv data/test.csv preprocessor.pkl: data/clean_dataset.csv
	$(PY) -m src.preprocess_data

# 5) Modèle
model.pkl: data/train.csv
	$(PY) -m src.training

# 6) Évaluation
.PHONY: evaluations
evaluations: model.pkl data/test.csv
	$(PY) -m src.evaluate

