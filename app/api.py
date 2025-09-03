from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from src.utils import load_params

# --- Chargement des artefacts et config ---
params = load_params()
model_path = params["model"]["path"]
preprocessor_path = params["model"]["preprocessor_path"]

# Colonnes catégorielles utilisées au preprocessing
CATEGORICAL_FEATURES = ["Geography", "Gender"]
# Colonnes à ignorer si présentes 
DROP_IF_PRESENT = ["Surname"]

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

app = FastAPI(
    title="Prédiction de Churn",
    description="Application de prédiction de Churn <br>Une version par API pour faciliter la réutilisation du modèle",
    version="1.0.0",
)

# --- Schéma d'entrée ---
class CustomerData(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

# --- Endpoint ---
@app.post("/predict", tags=["Predict"])
def predict(data: CustomerData):
    # 1) En DF
    df = pd.DataFrame([data.model_dump()])

    # 2) Nettoyage: drop les colonnes inutiles si présentes
    for col in DROP_IF_PRESENT:
        if col in df.columns:
            df = df.drop(columns=[col])

    # 3) Séparer cat / num
    X_num = df.drop(columns=CATEGORICAL_FEATURES)
    X_cat = pd.DataFrame(
        preprocessor.transform(df[CATEGORICAL_FEATURES]),
        columns=preprocessor.get_feature_names_out(CATEGORICAL_FEATURES),
        index=df.index,
    )

    # 4) Reconstituer X comme au training
    X = pd.concat([X_num, X_cat], axis=1)

    # 5) Prédire
    pred = int(model.predict(X)[0])
    label = "Exited" if pred == 1 else "Not Exited"

    return {"prediction": pred, "label": label}
