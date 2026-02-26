import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 1️⃣ Ruta del dataset original
file_path = "data/raw/Base_de_datos.xlsx"

# 2️⃣ Cargar datos
df = pd.read_excel(file_path)

print("Dataset cargado correctamente.")
print(f"Total filas: {df.shape[0]}")
print(f"Total columnas: {df.shape[1]}")

# 3️⃣ Dividir en baseline (70%) y current (30%)
baseline_df, current_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42
)

# 🔥 Simular drift en TODAS las columnas numéricas
numeric_cols = current_df.select_dtypes(include="number").columns

print("\nColumnas numéricas detectadas:")
print(numeric_cols)

for col in numeric_cols:
    current_df[col] = current_df[col] * 1.5

# 🔎 Verificación del drift antes de guardar
print("\nVerificación de drift aplicado:")
for col in numeric_cols[:3]:  # solo mostramos las primeras 3 para no saturar
    print(f"{col} → Baseline mean: {baseline_df[col].mean():.2f} | Current mean: {current_df[col].mean():.2f}")

# 4️⃣ Crear carpeta processed si no existe
os.makedirs("data/processed", exist_ok=True)

# 5️⃣ Guardar archivos
baseline_df.to_csv("data/processed/baseline.csv", index=False)
current_df.to_csv("data/processed/current.csv", index=False)

print("\nBaseline y Current creados correctamente.")