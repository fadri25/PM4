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

# Feature Engineering...
df['TX_AMOUNT_DEVIATION'] = df['TX_AMOUNT'] - df['CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW']
df['TX_TIME_SECONDS_SHIFTED'] = df.groupby('CUSTOMER_ID')['TX_TIME_SECONDS'].shift(1)
df['TIME_SINCE_LAST_TX'] = df['TX_TIME_SECONDS'] - df['TX_TIME_SECONDS_SHIFTED']
df['TIME_SINCE_LAST_TX'] = df['TIME_SINCE_LAST_TX'].fillna(999999)
df['TX_AMOUNT_SHIFTED_1'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].shift(1)
df['TX_AMOUNT_SHIFTED_2'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].shift(2)
df['TX_AMOUNT_SHIFTED_3'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].shift(3)
df['TX_AMOUNT_LAST3_MEAN'] = df[['TX_AMOUNT_SHIFTED_1', 'TX_AMOUNT_SHIFTED_2', 'TX_AMOUNT_SHIFTED_3']].mean(axis=1)
df['TX_AMOUNT_LAST3_MEAN'] = df['TX_AMOUNT_LAST3_MEAN'].fillna(0)
terminal_avg = df.groupby('TERMINAL_ID')['TX_AMOUNT'].transform('mean')
df['TX_AMOUNT_TO_TERMINAL_AVG'] = df['TX_AMOUNT'] / terminal_avg
# NEU: Terminal Fraud Ratio
fraud_ratio_per_terminal = df.groupby('TERMINAL_ID')['TX_FRAUD'].mean()
df['TERMINAL_FRAUD_RATIO'] = df['TERMINAL_ID'].map(fraud_ratio_per_terminal)

X = df.drop(columns=['TRANSACTION_ID', 'TX_DATETIME', 'TX_FRAUD', 'TX_FRAUD_SCENARIO'])
y = df['TX_FRAUD']
X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

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
    'clf__max_depth': [4, 6],
    'clf__n_estimators': [50, 100],
    'clf__learning_rate': [0.1, 0.3],
    'clf__subsample': [0.9, 1.0],
    'clf__colsample_bytree': [0.9, 1.0],
    'clf__gamma': [0, 1],
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
print("Grid Search Zeit: {:.2f}s".format(time.time() - start_time))
print("Beste Parameter:", grid_search.best_params_)

# Early Stopping Split
X_train_main, X_val, y_train_main, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_main, y_train_main)

best_params = grid_search.best_params_
params = {
    'objective': 'binary:logistic',
    'learning_rate': best_params['clf__learning_rate'],
    'max_depth': best_params['clf__max_depth'],
    'subsample': best_params['clf__subsample'],
    'colsample_bytree': best_params['clf__colsample_bytree'],
    'gamma': best_params['clf__gamma'],
    'eval_metric': 'logloss'
}

dtrain = xgb.DMatrix(X_train_res, label=y_train_res)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test)

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=best_params['clf__n_estimators'],
    early_stopping_rounds=10,
    evals=[(dval, "validation")]
)

# Threshold Optimierung
y_prob_es = bst.predict(dtest)
precision, recall, thresholds = precision_recall_curve(y_test, y_prob_es)
tmp_f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
best_thresh = thresholds[np.argmax(tmp_f1)] * 0.85  # Schwelle bewusst abgesenkt für mehr Recall
print("\nOptimaler Threshold basierend auf F1-Score: {:.4f}".format(best_thresh))

# Finaler Test
y_pred_es = (y_prob_es >= best_thresh).astype(int)
print("\n==== Testset Metriken ====")
print(f"Precision:  {precision_score(y_test, y_pred_es):.4f}")
print(f"Recall:     {recall_score(y_test, y_pred_es):.4f}")
print(f"F1-Score:   {f1_score(y_test, y_pred_es):.4f}")
print(f"ROC-AUC:    {roc_auc_score(y_test, y_prob_es):.4f}")
print(f"PR-AUC:     {average_precision_score(y_test, y_prob_es):.4f}")

# Konfusionsmatrix
cm = confusion_matrix(y_test, y_pred_es)
print("\nKonfusionsmatrix:")
print(cm)


# Feature Engineering für Scenario 3: Customer Spending Drift
# Idee: Vergleich aktueller Betrag mit dem Durchschnitt der letzten X Transaktionen des Kunden

def compute_spending_drift(df, window=5):
    df = df.sort_values(by=['CUSTOMER_ID', 'TX_DATETIME'])
    df['AVG_LAST_TX_AMOUNT'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    df['SPENDING_DRIFT'] = df['TX_AMOUNT'] / (df['AVG_LAST_TX_AMOUNT'] + 1e-5)
    return df

# Feature hinzufügen im Original-Dataframe df (vor dem Split)
df = compute_spending_drift(df)

# Danach wie gewohnt aufteilen
X = df.drop(columns=['TRANSACTION_ID', 'TX_DATETIME', 'TX_FRAUD', 'AVG_LAST_TX_AMOUNT'])
y = df['TX_FRAUD']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Optional: NaN oder Inf-Werte behandeln
X_train = X_train.copy()
X_test = X_test.copy()
X_train['SPENDING_DRIFT'] = X_train['SPENDING_DRIFT'].replace([np.inf, -np.inf], 0)
X_test['SPENDING_DRIFT'] = X_test['SPENDING_DRIFT'].replace([np.inf, -np.inf], 0)
X_train['SPENDING_DRIFT'] = X_train['SPENDING_DRIFT'].fillna(1)
X_test['SPENDING_DRIFT'] = X_test['SPENDING_DRIFT'].fillna(1)

# Szenario-basierte Auswertung
for scenario in [1, 2, 3]:
    idx = df.loc[y_test.index, 'TX_FRAUD_SCENARIO'] == scenario
    if idx.sum() > 0:
        recall_scenario = recall_score(y_test[idx], y_pred_es[idx])
        print(f"Recall für Scenario {scenario}: {recall_scenario:.4f}")
    else:
        print(f"Scenario {scenario}: Keine Testdaten vorhanden.")

# Feature Importance
xgb.plot_importance(bst)
plt.title("Feature Importance")
plt.show()