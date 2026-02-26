import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# ==============================
# 1️⃣ Cargar datos
# ==============================

df = pd.read_csv("data/processed/baseline.csv")

target = "Pago_atiempo"

if target not in df.columns:
    raise ValueError(f"El target '{target}' no existe en el dataset.")

# ==============================
# 2️⃣ Separar X e y
# ==============================

y = df[target]

X = df.drop(columns=[target])

print("Columnas originales:")
print(X.columns)

# 🔥 FORZAR TODO A NUMÉRICO
X = X.apply(pd.to_numeric, errors="coerce")

# Eliminar columnas completamente vacías
X = X.dropna(axis=1, how="all")

# Rellenar NaN restantes
X = X.fillna(0)

print("\nColumnas finales usadas para entrenamiento:")
print(X.columns)

print("\nTipos finales:")
print(X.dtypes)

# ==============================
# 3️⃣ Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 4️⃣ Entrenar modelo
# ==============================

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ==============================
# 5️⃣ Evaluar
# ==============================

preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print(f"\nAccuracy del modelo: {accuracy}")

# ==============================
# 6️⃣ Guardar modelo
# ==============================

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/best_model.pkl")

print("\nModelo guardado correctamente en model/best_model.pkl")