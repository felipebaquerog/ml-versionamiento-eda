# Qué debe hacer este archivo
# 1. Recibir un DataFrame
# 2. Separar X e y
# 3. Definir numéricas y categóricas
# 4. Crear el ColumnTransformer
# 5. Hacer train/test split
# 6. Retornar los conjuntos listos para modelar

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd


def feature_engineering(df):

    # -----------------------------
    # 1. Separar variable objetivo
    # -----------------------------
    y = df["Pago_atiempo"]
    X = df.drop(columns=["Pago_atiempo"], errors="ignore")

    # -----------------------------
    # 2. Eliminar variables con posible data leakage
    # -----------------------------
    leakage_columns = [
        "saldo_mora",
        "saldo_total",
        "saldo_principal",
        "saldo_mora_codeudor",
        "puntaje",
        "puntaje_datacredito"
    ]

    X = X.drop(columns=leakage_columns, errors="ignore")


    # Eliminar fecha si existe
    if "fecha_prestamo" in X.columns:
        X = X.drop(columns=["fecha_prestamo"])

    # -----------------------------
    # 3. Intentar convertir columnas a numérico cuando sea posible
    # -----------------------------
    for col in X.columns:
        try:
            X[col] = pd.to_numeric(X[col])
        except:
            pass

    # -----------------------------
    # 4. Detectar tipos reales
    # -----------------------------
    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = [col for col in X.columns if col not in num_features]

    # Forzar categóricas a string
    for col in cat_features:
        X[col] = X[col].astype(str)

    # -----------------------------
    # 5. Pipelines
    # -----------------------------
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_features),
            ("cat", categorical_pipeline, cat_features)
        ]
    )

    # -----------------------------
    # 6. Transformar
    # -----------------------------
    X_processed = preprocessor.fit_transform(X)

    # -----------------------------
    # 7. Train/Test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
