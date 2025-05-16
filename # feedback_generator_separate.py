import pandas as pd
import numpy as np
import time
from datetime import datetime

# Basis-Werte aus dem Bild
BASE_METRICS = {
    "Precision": 0.9533,
    "Recall":    0.7772,
    "F1_Score":  0.8563,
    "ROC_AUC":   0.9908,
    "PR_AUC":    0.8655,
}
VARIATION = 0.005
CSV_FILE = "metrics_history.csv"

def update_metrics_history(file_name, base_metrics, variation):
    # CSV einlesen oder leeres DataFrame anlegen
    try:
        history = pd.read_csv(file_name)
    except FileNotFoundError:
        cols = ["Zeit"] + list(base_metrics.keys())
        history = pd.DataFrame(columns=cols)

    # Letzte Werte (oder Basis, wenn leer)
    if not history.empty:
        last = history.iloc[-1].to_dict()
    else:
        last = {m: v for m, v in base_metrics.items()}

    # Neue Werte berechnen und auf base±variation clippen
    new_row = {"Zeit": datetime.now().strftime("%H:%M:%S")}
    for m, base in base_metrics.items():
        lower, upper = base - variation, base + variation
        raw = last.get(m, base) + np.random.normal(scale=variation)
        new_row[m] = round(np.clip(raw, lower, upper), 4)

    # An DataFrame anhängen und speichern
    history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
    history.to_csv(file_name, index=False)

if __name__ == "__main__":
    while True:
        update_metrics_history(CSV_FILE, BASE_METRICS, VARIATION)
        print("Neue Werte gespeichert ✅")
        time.sleep(8)
