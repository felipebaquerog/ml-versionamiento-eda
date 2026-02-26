import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
from scipy.spatial.distance import jensenshannon


# =========================================
# 1️⃣ PSI (numéricas)
# =========================================
def calculate_psi(expected, actual, buckets=10):

    expected = expected.dropna()
    actual = actual.dropna()

    if len(expected) == 0 or len(actual) == 0:
        return 0

    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 2:
        return 0

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)

    expected_perc = np.where(expected_perc == 0, 0.0001, expected_perc)
    actual_perc = np.where(actual_perc == 0, 0.0001, actual_perc)

    psi = np.sum((expected_perc - actual_perc) * np.log(expected_perc / actual_perc))

    return psi


# =========================================
# 2️⃣ KS test (numéricas)
# =========================================
def calculate_ks(expected, actual):

    expected = expected.dropna()
    actual = actual.dropna()

    if len(expected) == 0 or len(actual) == 0:
        return 0, 1

    ks_stat, p_value = ks_2samp(expected, actual)

    return ks_stat, p_value


# =========================================
# 3️⃣ Jensen-Shannon (numéricas)
# =========================================
def calculate_js(expected, actual, bins=10):

    expected = expected.dropna()
    actual = actual.dropna()

    if len(expected) == 0 or len(actual) == 0:
        return 0

    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())

    expected_hist, _ = np.histogram(
        expected, bins=bins, range=(min_val, max_val), density=True
    )
    actual_hist, _ = np.histogram(
        actual, bins=bins, range=(min_val, max_val), density=True
    )

    expected_hist = np.where(expected_hist == 0, 0.0001, expected_hist)
    actual_hist = np.where(actual_hist == 0, 0.0001, actual_hist)

    js_distance = jensenshannon(expected_hist, actual_hist)

    return js_distance


# =========================================
# 4️⃣ Chi-cuadrado (categóricas)
# =========================================
def calculate_chi2(expected, actual):

    expected = expected.fillna("missing")
    actual = actual.fillna("missing")

    baseline_counts = expected.value_counts()
    current_counts = actual.value_counts()

    all_categories = set(baseline_counts.index).union(set(current_counts.index))

    baseline_freq = [baseline_counts.get(cat, 0) for cat in all_categories]
    current_freq = [current_counts.get(cat, 0) for cat in all_categories]

    contingency_table = np.array([baseline_freq, current_freq])

    if contingency_table.shape[1] < 2:
        return None, None

    chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)

    return chi2_stat, p_value


# =========================================
# 5️⃣ Evaluación por variable
# =========================================
def evaluate_variable_drift(baseline_df, current_df, column_name):

    baseline_col = baseline_df[column_name]
    current_col = current_df[column_name]

    if pd.api.types.is_numeric_dtype(baseline_col):

        psi_value = calculate_psi(baseline_col, current_col)
        ks_stat, ks_p = calculate_ks(baseline_col, current_col)
        js_value = calculate_js(baseline_col, current_col)

        # Clasificación principal basada en PSI
        if psi_value < 0.1:
            status = "No Drift"
        elif psi_value < 0.25:
            status = "Drift Moderado"
        else:
            status = "Drift Severo"

        return {
            "variable": column_name,
            "type": "Numerica",
            "psi": round(psi_value, 4),
            "ks_stat": round(ks_stat, 4),
            "ks_p_value": round(ks_p, 4),
            "js_distance": round(js_value, 4),
            "chi2_stat": None,
            "chi2_p_value": None,
            "status": status
        }

    else:

        chi2_stat, chi2_p = calculate_chi2(baseline_col, current_col)

        if chi2_p is not None and chi2_p < 0.05:
            status = "Drift Severo"
        else:
            status = "No Drift"

        return {
            "variable": column_name,
            "type": "Categorica",
            "psi": None,
            "ks_stat": None,
            "ks_p_value": None,
            "js_distance": None,
            "chi2_stat": round(chi2_stat, 4) if chi2_stat else None,
            "chi2_p_value": round(chi2_p, 4) if chi2_p else None,
            "status": status
        }


# =========================================
# 6️⃣ Función principal
# =========================================
def run_monitoring(baseline_path, current_path):

    baseline_df = pd.read_csv(baseline_path)
    current_df = pd.read_csv(current_path)

    results = []

    for col in baseline_df.columns:
        if col in current_df.columns:
            result = evaluate_variable_drift(
                baseline_df,
                current_df,
                col
            )
            results.append(result)

    return pd.DataFrame(results)


# =========================================
# 7️⃣ Ejecución directa
# =========================================
if __name__ == "__main__":

    results_df = run_monitoring(
        "data/processed/baseline.csv",
        "data/processed/current.csv"
    )

    print(results_df)