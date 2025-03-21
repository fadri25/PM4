import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# Laden des Datensatzes
df = pd.read_excel(r"C:\PM4\processed-data\transactions_first_100.xlsx")

# Ziel (Target) und Features extrahieren
X = df.drop(columns=['TRANSACTION_ID', 'TX_DATETIME', 'TX_FRAUD'])
y = df['TX_FRAUD']

# Aufteilen in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost-Modell mit dem richtigen 'objective' Parameter für binäre Klassifikation erstellen
model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='mlogloss')

# Modell trainieren
model.fit(X_train, y_train)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# Laden des Datensatzes (bitte passe den Pfad an deine Umgebung an)
df = pd.read_excel(r"C:\PM4\processed-data\transactions_first_100.xlsx")

# Zielvariable (TX_FRAUD) und Features extrahieren
X = df.drop(columns=['TRANSACTION_ID', 'TX_DATETIME', 'TX_FRAUD'])
y = df['TX_FRAUD']

# Aufteilen des Datensatzes in Trainings- und Testdaten (80% Training, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Erstellen des XGBoost-Klassifikationsmodells mit dem korrekten 'objective'
model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='mlogloss')

# Trainieren des Modells
model.fit(X_train, y_train)

# Vorhersagen auf den Testdaten
y_pred = model.predict(X_test)

# Evaluierung des Modells
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Ausgabe der Ergebnisse
print(f"Genauigkeit: {accuracy}")
print("Klassifikationsbericht:")
print(report)

# Vorhersagen auf den Testdaten
y_pred = model.predict(X_test)

# Modell evaluieren
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Ausgabe der Ergebnisse
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
