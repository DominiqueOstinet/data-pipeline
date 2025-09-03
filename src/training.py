import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from src.utils import load_params


def train_model(params: dict):
    """Train a RandomForest model and save it ."""
    train_path = params["data"]["train_dataset_path"]
    model_path = params["model"]["path"]
    target = params["data"]["target"]

    # Charger le dataset d'entraînement
    df = pd.read_csv(train_path)
    X = df.drop(columns=[target])
    y = df[target]

    # Définir le modèle
    model = RandomForestClassifier(
        n_estimators=params["model"]["n_estimators"],
        max_depth=params["model"]["max_depth"],
        random_state=params["model"]["random_state"]
    )

    # Entraîner
    model.fit(X, y)

    # Sauvegarder le modèle entraîné
    joblib.dump(model, model_path)
    print(f"✅ Modèle sauvegardé dans {model_path}")


if __name__ == "__main__":
    params = load_params()
    train_model(params)
