import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.model_monitoring import run_monitoring

st.set_page_config(page_title="Model Drift Monitoring", layout="wide")

st.title("📊 Monitoreo de Data Drift")

# ==========================
# Cargar resultados
# ==========================

try:
    results_df = run_monitoring(
        "data/processed/baseline.csv",
        "data/processed/current.csv"
    )
except Exception as e:
    st.error(f"Error cargando datos: {e}")
    st.stop()

# ==========================
# Tabla con semáforo
# ==========================

st.subheader("Resultados de Drift por Variable")

def color_status(val):
    if val == "Drift Severo":
        return "background-color: #ff4b4b; color: white"
    elif val == "Drift Moderado":
        return "background-color: #ffa500; color: black"
    else:
        return "background-color: #2ecc71; color: white"

styled_df = results_df.style.applymap(color_status, subset=["status"])
st.dataframe(styled_df, use_container_width=True)

# ==========================
# Resumen General
# ==========================

drift_severo = results_df[results_df["status"] == "Drift Severo"].shape[0]
drift_moderado = results_df[results_df["status"] == "Drift Moderado"].shape[0]

st.subheader("Resumen General")

col1, col2 = st.columns(2)
col1.metric("Variables con Drift Severo", drift_severo)
col2.metric("Variables con Drift Moderado", drift_moderado)

if drift_severo > 5:
    st.error("🚨 Drift crítico detectado. Se recomienda retraining inmediato del modelo.")
elif drift_severo > 0:
    st.warning("⚠️ Drift significativo detectado. Evaluar estabilidad del modelo.")
elif drift_moderado > 0:
    st.info("🔍 Drift leve detectado. Monitoreo continuo recomendado.")
else:
    st.success("✅ Distribución estable. Modelo en condiciones normales.")

# ==========================
# Gráfico PSI
# ==========================

st.subheader("Gráfico de PSI por Variable")

numeric_results = results_df.dropna(subset=["psi"]).sort_values("psi", ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(numeric_results["variable"], numeric_results["psi"])
ax.axvline(0.1, linestyle="--")
ax.axvline(0.25, linestyle="--")
ax.set_xlabel("PSI")
ax.set_title("PSI por Variable")

st.pyplot(fig)

# ==========================
# Comparación de Distribución
# ==========================

st.subheader("Comparación de Distribución Baseline vs Current")

baseline = pd.read_csv("data/processed/baseline.csv")
current = pd.read_csv("data/processed/current.csv")

numeric_columns = baseline.select_dtypes(include="number").columns.tolist()

selected_variable = st.selectbox(
    "Selecciona una variable numérica",
    numeric_columns
)

fig2, ax2 = plt.subplots(figsize=(8, 5))

ax2.hist(baseline[selected_variable], bins=30, alpha=0.5, label="Baseline")
ax2.hist(current[selected_variable], bins=30, alpha=0.5, label="Current")

ax2.legend()
ax2.set_title(f"Distribución de {selected_variable}")

st.pyplot(fig2)

#streamlit run app.py

from src.temporal_monitoring import simulate_temporal_drift

st.subheader("Evolución Temporal del Drift")

temporal_df = simulate_temporal_drift(
    "data/processed/baseline.csv",
    months=6
)

selected_variable_time = st.selectbox(
    "Selecciona variable para análisis temporal",
    temporal_df["variable"].unique()
)

filtered = temporal_df[temporal_df["variable"] == selected_variable_time]

fig3, ax3 = plt.subplots()

ax3.plot(filtered["month"], filtered["psi"], marker="o")
ax3.axhline(0.1, linestyle="--")
ax3.axhline(0.25, linestyle="--")

ax3.set_xlabel("Mes")
ax3.set_ylabel("PSI")
ax3.set_title(f"Evolución temporal de {selected_variable_time}")

st.pyplot(fig3)