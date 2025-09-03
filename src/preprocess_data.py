import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from src.utils import load_params


def preprocess_data(params: dict):
    """Encodage des variables catégorielles + split train/test + sauvegarde du préprocesseur."""
    src = params["data"]["clean_dataset_path"]
    train_path = params["data"]["train_dataset_path"]
    test_path = params["data"]["test_dataset_path"]
    target = params["data"]["target"]
    preproc_path = params["model"]["preprocessor_path"]  # chemin du .pkl

    # 1) Charger
    df = pd.read_csv(src)
    X = df.drop(columns=[target])
    y = df[target]

    # 2) Split (stratifié)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["split"]["test_size"],
        random_state=params["split"]["random_state"],
        stratify=y,
    )

    # 3) Encoder catégorielles (fit sur TRAIN uniquement)
    categorical_features = ["Geography", "Gender"]
    enc = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")

    # Fit sur train
    enc.fit(X_train[categorical_features])

    # Transform train/test
    X_train_cat = pd.DataFrame(
        enc.transform(X_train[categorical_features]),
        columns=enc.get_feature_names_out(categorical_features),
        index=X_train.index,
    )
    X_test_cat = pd.DataFrame(
        enc.transform(X_test[categorical_features]),
        columns=enc.get_feature_names_out(categorical_features),
        index=X_test.index,
    )

    # Concat avec le numérique (même ordre des colonnes entre train/test)
    X_train_num = X_train.drop(columns=categorical_features)
    X_test_num = X_test.drop(columns=categorical_features)

    X_train_final = pd.concat([X_train_num, X_train_cat], axis=1)
    X_test_final = pd.concat([X_test_num, X_test_cat], axis=1)

    # 4) Sauvegardes CSV (features + target)
    X_train_final.assign(**{target: y_train}).to_csv(train_path, index=False)
    X_test_final.assign(**{target: y_test}).to_csv(test_path, index=False)

    # 5) Sauvegarde du préprocesseur pour l’inférence (API)
    joblib.dump(enc, preproc_path)
    print(f"✅ Preprocessor sauvegardé dans {preproc_path}")
    print(f"✅ Train: {X_train_final.shape} | Test: {X_test_final.shape}")


if __name__ == "__main__":
    params = load_params()
    preprocess_data(params)

