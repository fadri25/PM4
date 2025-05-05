# feedback_generator_separate.py

import pandas as pd
import numpy as np
import time
from datetime import datetime

def update_history(file_name, start_value, variation=0.005):
    try:
        history = pd.read_csv(file_name)
    except FileNotFoundError:
        history = pd.DataFrame(columns=["Zeit", "Wert"])

    if not history.empty:
        last_value = history["Wert"].iloc[-1]
    else:
        last_value = start_value

    new_value = np.clip(last_value + np.random.normal(0, variation), 0.80, 1.0)

    new_row = {
        "Zeit": datetime.now().strftime("%H:%M:%S"),
        "Wert": round(new_value, 3)
    }

    history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

    history.to_csv(file_name, index=False)

if __name__ == "__main__":
    while True:
        update_history("precision_history.csv", start_value=0.92)
        update_history("recall_history.csv", start_value=0.87)
        update_history("roc_auc_history.csv", start_value=0.89)
        update_history("f1_score_history.csv", start_value=0.95)
        print("Neue Werte gespeichert ✅")
        time.sleep(8)
