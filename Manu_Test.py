print("Hallo")

import os
import pandas as pd
import numpy as np
import xgboost_model as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 1. CSV-Dateien einlesen
transactions = pd.read_csv("C://PM4/transactions.csv")
customers = pd.read_csv('C://PM4/customers.csv')
terminals = pd.read_csv('C://PM4/terminals.csv')

# 2. Unnötige Spalte "Unnamed: 0" entfernen (falls vorhanden)
for df in [transactions, customers, terminals]:
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

# 3. Daten zusammenführen
data = transactions.merge(customers, on='CUSTOMER_ID', how='left')
data = data.merge(terminals, on='TERMINAL_ID', how='left')

# 4. Fehlende Werte auffüllen
data.fillna(-999, inplace=True)

# 5. Features und Zielvariable definieren
# Zielvariable: TX_FRAUD (1 = Betrug, 0 = normal)
y = data['TX_FRAUD']

# Für die Features entfernen wir u.a. IDs und Datumsspalten, die hier nicht als Features dienen sollen.
drop_cols = ['TRANSACTION_ID', 'TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_FRAUD']
X = data.drop(columns=drop_cols)

# Falls es kategoriale Variablen gibt, in Dummy-Variablen umwandeln
X = pd.get_dummies(X)

# 6. Aufteilen in Trainings- und Testdaten (stratifiziert, da Betrugsfälle oft selten sind)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

# 7. Umgang mit unausgeglichenen Klassen:
# Berechne scale_pos_weight als Verhältnis der Anzahl normaler zu Betrugsfälle im Trainingsset
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# 8. XGBoost Modell definieren und trainieren
model = xgb.XGBClassifier(
    use_label_encoder=False,  # Verhindert Warnungen bezüglich Label-Encoding
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight
)

model.fit(X_train, y_train)

# 9. Modell evaluieren
y_pred = model.predict(X_test)
print("Genauigkeit:", accuracy_score(y_test, y_pred))
print("\nKlassifikationsbericht:\n", classification_report(y_test, y_pred))

print("Train set distribution:")
print(y_train.value_counts())

print("\nTest set distribution:")
print(y_test.value_counts())


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


import matplotlib.pyplot as plt
import numpy as np

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]  # Sortierung absteigend
feature_names = X.columns[indices]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(len(feature_names)), importances[indices], align='center')
plt.xticks(range(len(feature_names)), feature_names, rotation=90)
plt.tight_layout()
plt.show()

corrs = []
for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col]):
        corr = X[col].corr(y)  # Korrelation mit dem Label
        corrs.append((col, corr))

# Nach absoluter Korrelation sortieren, absteigend
sorted_corrs = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)
for col, c in sorted_corrs[:20]:
    print(f"{col}: {c:.3f}")
