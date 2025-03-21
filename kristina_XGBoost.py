import time
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score
)
import xgboost as xgb

# 1) Daten laden (Pfad anpassen)
df = pd.read_csv(r"C:\PM4\processed-data\transactions_first_10000.csv")

# 2) Features und Zielvariable definieren
#    Passe die Spaltennamen an, falls sie im CSV anders heißen
X = df.drop(columns=['TRANSACTION_ID', 'TX_DATETIME', 'TX_FRAUD'])
y = df['TX_FRAUD']

# 3) Aufteilen in Trainings- und Testdaten (stratifiziert, damit beide Klassen vertreten sind)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

# 4) XGBoost-Klassifikationsmodell erstellen
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='mlogloss',
    use_label_encoder=False
)

# 5) Parameter-Raster definieren (analog zum Handbook)
param_grid = {
    "max_depth": [3, 6, 9],
    "n_estimators": [25, 50, 100],
    "learning_rate": [0.1, 0.3],
    "random_state": [0],
    "n_jobs": [1],
    "verbosity": [0]
}

# 6) GridSearchCV einrichten: Wir verwenden als Scoring 'roc_auc' und 'average_precision'
scoring = {
    "roc_auc": "roc_auc",
    "avg_precision": "average_precision"
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring=scoring,
    refit="roc_auc",  # Das beste Modell wird anhand von ROC-AUC ausgewählt
    cv=cv,
    return_train_score=True,
    n_jobs=1  # 1 Kern, analog zum Handbook-Beispiel
)

# 7) Grid Search durchführen und Trainingszeit messen
start_time = time.time()
grid_search.fit(X_train, y_train)
grid_search_time = time.time() - start_time

print("Grid Search Trainingszeit: {:.2f} Sekunden".format(grid_search_time))
print("Beste Parameter: ", grid_search.best_params_)

# Ergebnisse der Grid Search in einem DataFrame sammeln (optional)
results_df = pd.DataFrame(grid_search.cv_results_)
print("\nErgebnisse der Grid Search (Top 5 Zeilen):")
print(results_df[['params', 'mean_test_roc_auc', 'mean_test_avg_precision']].head())

# 8) Evaluierung auf dem Testset mit dem besten Modell
best_model = grid_search.best_estimator_

# Inference-Zeit messen
start_time = time.time()
y_prob = best_model.predict_proba(X_test)[:, 1]  # Wahrscheinlichkeiten (für ROC-AUC, PR-AUC)
inference_time_total = time.time() - start_time
inference_time_per_tx = inference_time_total / len(X_test)

# Klassische Vorhersage (binäre Labels)
y_pred = (y_prob >= 0.5).astype(int)

# Metriken berechnen
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

print("\n==== Testset Metriken ====")
print(f"Precision:  {precision:.4f}")
print(f"Recall:     {recall:.4f}")
print(f"F1-Score:   {f1:.4f}")
print(f"ROC-AUC:    {roc_auc:.4f}")
print(f"PR-AUC:     {pr_auc:.4f}")

print("\n==== Inference Zeiten ====")
print(f"Inference-Zeit gesamt: {inference_time_total:.6f} Sekunden")
print(f"Inference-Zeit pro Transaktion: {inference_time_per_tx:.6f} Sekunden")
