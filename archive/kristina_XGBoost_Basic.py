import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                             roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import xgboost_model as xgb

# 1) Daten laden
df = pd.read_csv(r"C:\PM4\processed-data\transactions_first_100000.csv")

# 2) Features und Zielvariable definieren (ohne TX_FRAUD_SCENARIO)
X = df.drop(columns=['TRANSACTION_ID', 'TX_DATETIME', 'TX_FRAUD', 'TX_FRAUD_SCENARIO'])
y = df['TX_FRAUD']

# 3) Aufteilen in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Pipeline: Grid Search auf Trainingsdaten mit SMOTE + XGBoost
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

param_grid = {
    'clf__max_depth': [3, 6, 9],
    'clf__n_estimators': [25, 50, 100],
    'clf__learning_rate': [0.1, 0.3],
    'clf__random_state': [0]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
    return_train_score=True
)

start_time = time.time()
grid_search.fit(X_train, y_train)
grid_search_time = time.time() - start_time
print("Grid Search Trainingszeit: {:.2f} Sekunden".format(grid_search_time))
print("Beste Parameter (Grid Search):", grid_search.best_params_)

# 5) Early Stopping Teil mit xgboost.train
X_train_main, X_val, y_train_main, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

# SMOTE nur auf Trainingsdaten anwenden
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_main, y_train_main)

# Beste Parameter übernehmen
best_params = grid_search.best_params_
params = {
    'objective': 'binary:logistic',
    'learning_rate': best_params['clf__learning_rate'],
    'max_depth': best_params['clf__max_depth'],
    'eval_metric': 'logloss'
}

# DMatrix für xgboost.train
dtrain = xgb.DMatrix(X_train_res, label=y_train_res)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test)

# Training mit Early Stopping
start_time = time.time()
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=best_params['clf__n_estimators'],
    early_stopping_rounds=10,
    evals=[(dval, "validation")]
)
es_training_time = time.time() - start_time
print("\nTrainingszeit mit Early Stopping: {:.2f} Sekunden".format(es_training_time))

# 6) Threshold-Optimierung (Precision-Recall-Kurve)
y_prob_es = bst.predict(dtest)
precision, recall, thresholds = precision_recall_curve(y_test, y_prob_es)
tmp_f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
best_thresh = thresholds[np.argmax(tmp_f1)]
print("\nOptimaler Threshold basierend auf F1-Score: {:.4f}".format(best_thresh))

# 7) Vorhersagen mit optimiertem Threshold
y_pred_es = (y_prob_es >= best_thresh).astype(int)

# 8) Metriken berechnen
print("\n==== Testset Metriken (mit Early Stopping + Threshold-Optimierung) ====")
print(f"Precision:  {precision_score(y_test, y_pred_es):.4f}")
print(f"Recall:     {recall_score(y_test, y_pred_es):.4f}")
print(f"F1-Score:   {f1_score(y_test, y_pred_es):.4f}")
print(f"ROC-AUC:    {roc_auc_score(y_test, y_prob_es):.4f}")
print(f"PR-AUC:     {average_precision_score(y_test, y_prob_es):.4f}")

# 9) Konfusionsmatrix und Feature Importance
cm = confusion_matrix(y_test, y_pred_es)
print("\nKonfusionsmatrix:")
print(cm)

# Feature Importance (Low-Level API erfordert Booster Zugriff)
xgb.plot_importance(bst)
plt.title("Feature Importance (mit Early Stopping)")
plt.show()

# 10) Nachträglich: Kontrolliere mit 'TX_FRAUD_SCENARIO'
scores = []
print("\nAuswertung pro Fraud-Szenario:")
for scenario in df['TX_FRAUD_SCENARIO'].unique():
    mask = df.loc[X_test.index, 'TX_FRAUD_SCENARIO'] == scenario
    f1 = f1_score(y_test[mask], y_pred_es[mask])
    scores.append((scenario, f1))
    print(f"Szenario {scenario}: F1-Score = {f1:.4f}")

# 11) Visualisierung der Szenario-Metriken
scenarios, f1_values = zip(*scores)
plt.figure(figsize=(8,5))
plt.bar([str(s) for s in scenarios], f1_values)
plt.xlabel("Fraud Szenario")
plt.ylabel("F1-Score")
plt.title("F1-Score je Fraud-Szenario")
plt.ylim(0, 1.05)
plt.show()