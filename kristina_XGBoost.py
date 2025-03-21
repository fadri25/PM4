import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                             roc_auc_score, average_precision_score, confusion_matrix)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# 1) Daten laden (Pfad anpassen)
df = pd.read_csv(r"C:\PM4\processed-data\transactions_first_10000.csv")

# 2) Features und Zielvariable definieren (Spaltennamen ggf. anpassen)
X = df.drop(columns=['TRANSACTION_ID', 'TX_DATETIME', 'TX_FRAUD'])
y = df['TX_FRAUD']

# 3) Aufteilen in Trainings- und Testdaten (stratifiziert, damit beide Klassen vertreten sind)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Pipeline erstellen: SMOTE gefolgt vom XGBoost-Klassifikator
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf', xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_jobs=1,
        verbosity=0
    ))
])

# 5) Parameter-Raster definieren (analog zum Handbook)
param_grid = {
    'clf__max_depth': [3, 6, 9],
    'clf__n_estimators': [25, 50, 100],
    'clf__learning_rate': [0.1, 0.3],
    'clf__random_state': [0]
}

# 6) Stratified Cross-Validation einrichten
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Wir verwenden als Haupt-Scoring-Metrik ROC-AUC
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,  # Parallele Verarbeitung (alle verfügbaren Kerne)
    return_train_score=True
)

# 7) Grid Search durchführen und Trainingszeit messen
start_time = time.time()
grid_search.fit(X_train, y_train)
grid_search_time = time.time() - start_time
print("Grid Search Trainingszeit: {:.2f} Sekunden".format(grid_search_time))
print("Beste Parameter (Grid Search):", grid_search.best_params_)

# Ergebnisse der Grid Search (optional)
results_df = pd.DataFrame(grid_search.cv_results_)
print("\nTop 5 Ergebnisse der Grid Search:")
print(results_df[['params', 'mean_test_score']].head())

# 8) Evaluierung des besten Modells (ohne Early Stopping) auf dem Testset
best_pipeline = grid_search.best_estimator_
start_time = time.time()
y_prob = best_pipeline.predict_proba(X_test)[:, 1]
inference_time_total = time.time() - start_time
inference_time_per_tx = inference_time_total / len(X_test)
y_pred = (y_prob >= 0.5).astype(int)

precision_val = precision_score(y_test, y_pred)
recall_val    = recall_score(y_test, y_pred)
f1_val        = f1_score(y_test, y_pred)
roc_auc_val   = roc_auc_score(y_test, y_prob)
pr_auc_val    = average_precision_score(y_test, y_prob)

print("\n==== Testset Metriken (ohne Early Stopping) ====")
print(f"Precision:  {precision_val:.4f}")
print(f"Recall:     {recall_val:.4f}")
print(f"F1-Score:   {f1_val:.4f}")
print(f"ROC-AUC:    {roc_auc_val:.4f}")
print(f"PR-AUC:     {pr_auc_val:.4f}")
print("\n==== Inference Zeiten ====")
print(f"Inference-Zeit gesamt: {inference_time_total:.6f} Sekunden")
print(f"Inference-Zeit pro Transaktion: {inference_time_per_tx:.6f} Sekunden")

# 9) Re-Fit des besten Modells mit Early Stopping
# Erzeuge ein zusätzliches Validierungsset aus den Trainingsdaten
X_train_main, X_val, y_train_main, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

# Hole die besten Parameter aus der Grid Search
best_params = grid_search.best_params_
clf_params = {
    'max_depth': best_params['clf__max_depth'],
    'n_estimators': best_params['clf__n_estimators'],
    'learning_rate': best_params['clf__learning_rate'],
    'random_state': best_params['clf__random_state'],
    'objective': 'binary:logistic',
    'eval_metric': 'mlogloss',
    'use_label_encoder': False,
    'n_jobs': 1,
    'verbosity': 0
}

# Erstelle einen neuen XGBoost-Klassifikator mit Early Stopping
clf_with_es = xgb.XGBClassifier(**clf_params)

# Neue Pipeline mit SMOTE und dem modifizierten Klassifikator
pipeline_es = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf', clf_with_es)
])

# Fit mit Early Stopping (wir übergeben ein eval_set und early_stopping_rounds als Parameter für 'clf')
start_time = time.time()
pipeline_es.fit(
    X_train_main, y_train_main,
    clf__eval_set=[(X_val, y_val)],
    clf__early_stopping_rounds=10
)
es_training_time = time.time() - start_time
print("\nTrainingszeit mit Early Stopping: {:.2f} Sekunden".format(es_training_time))

# 10) Evaluierung des mit Early Stopping trainierten Modells
start_time = time.time()
y_prob_es = pipeline_es.predict_proba(X_test)[:, 1]
inference_time_total_es = time.time() - start_time
inference_time_per_tx_es = inference_time_total_es / len(X_test)
y_pred_es = (y_prob_es >= 0.5).astype(int)

precision_es = precision_score(y_test, y_pred_es)
recall_es    = recall_score(y_test, y_pred_es)
f1_es        = f1_score(y_test, y_pred_es)
roc_auc_es   = roc_auc_score(y_test, y_prob_es)
pr_auc_es    = average_precision_score(y_test, y_prob_es)

print("\n==== Testset Metriken (mit Early Stopping) ====")
print(f"Precision:  {precision_es:.4f}")
print(f"Recall:     {recall_es:.4f}")
print(f"F1-Score:   {f1_es:.4f}")
print(f"ROC-AUC:    {roc_auc_es:.4f}")
print(f"PR-AUC:     {pr_auc_es:.4f}")
print("\n==== Inference Zeiten (Early Stopping Model) ====")
print(f"Inference-Zeit gesamt: {inference_time_total_es:.6f} Sekunden")
print(f"Inference-Zeit pro Transaktion: {inference_time_per_tx_es:.6f} Sekunden")

# 11) Zusätzliche Auswertung: Konfusionsmatrix und Feature Importance
cm = confusion_matrix(y_test, y_pred_es)
print("\nKonfusionsmatrix:")
print(cm)

# Plot der Feature Importance (aus dem XGBClassifier in der Pipeline)
xgb_clf = pipeline_es.named_steps["clf"]
fig, ax = plt.subplots(figsize=(8, 6))
xgb.plot_importance(xgb_clf, ax=ax)
plt.title("Feature Importance")
plt.show()
