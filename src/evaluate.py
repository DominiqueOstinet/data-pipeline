import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.utils import load_params


def evaluate_model(params: dict):
    """Évalue le modèle sauvegardé sur le jeu de test."""
    test_path = params["data"]["test_dataset_path"]
    model_path = params["model"]["path"]
    target = params["data"]["target"]

    # Charger le dataset de test
    df = pd.read_csv(test_path)
    X = df.drop(columns=[target])
    y = df[target]

    # Charger le modèle entraîné
    model = joblib.load(model_path)

    # Prédictions
    y_pred = model.predict(X)

    # Évaluation
    acc = accuracy_score(y, y_pred)
    print(f"✅ Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))


if __name__ == "__main__":
    params = load_params()
    evaluate_model(params)
