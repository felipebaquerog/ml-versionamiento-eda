import pandas as pd
import numpy as np
from src.model_monitoring import calculate_psi

def simulate_temporal_drift(baseline_path, months=6):

    baseline = pd.read_csv(baseline_path)
    numeric_columns = baseline.select_dtypes(include="number").columns

    temporal_results = []

    for month in range(1, months + 1):

        # Simular drift creciente
        current = baseline.copy()

        for col in numeric_columns:
            current[col] = current[col] * (1 + 0.05 * month)

        for col in numeric_columns:
            psi = calculate_psi(baseline[col], current[col])

            temporal_results.append({
                "month": month,
                "variable": col,
                "psi": psi
            })

    return pd.DataFrame(temporal_results)