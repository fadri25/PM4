
# PM4 – Fraud Detection System

Dieses Repository enthält zwei voneinander getrennte Systeme:

1. XGBoost-Modell zur Erkennung von betrügerischen Transaktionen  
2. Streamlit-Dashboard zur Visualisierung und Analyse von Transaktionsdaten

---

## Projektstruktur

```
├── main.py                   # Startpunkt für Datenverarbeitung + Modelltraining
├── xgboost_model.py          # Training & Evaluation mit XGBoost
├── data_transformation.py    # Feature Engineering (zeitbasierte + aggregierte Features)
├── getdata.py                # CSV-Import + Sampling
├── Dashboard_3.py            # Streamlit-Dashboard zur Analyse
└── README.md                 # Diese Anleitung
```

---

## Voraussetzungen

```bash
pip install -r requirements.txt
```

---

## Start – Modelltraining

1. Lege deine Eingabedatei ab:
   ```
   C:/PM4/transactions_first_100000.csv
   ```
   # Geht auch mit mehr Datensätzen. Dauert einfach länger.

2. Starte das Hauptskript:
   ```bash
   python main.py
   ```

3. Ergebnis:
   - Modell wird gespeichert als:  
     `C:/PM4/fraud_detection_model.json`
   - Vorverarbeitete Daten:  
     `C:/PM4/processed-data/transactions_all.csv`
   - Vorschau :  
     `transactions_first_50_kürzer.csv`

---

## Start – Dashboard

1. Dashboard starten:
   ```bash
   streamlit run Dashboard_3.py
   ```

2. Login:
   - Benutzername: `0550`
   - Passwort: *nicht erforderlich*

---

## Hinweise

- Das Modell und das Dashboard verwenden nicht denselben Datensatz.
- Das Modell nutzt `transactions_first_100000.csv`, während das Dashboard mit einer reduzierten Vorschauversion arbeitet.
- Für das Einlesen der Daten muss der Ordner und die Dateien unter dem Pfad "C:/PM4/" vorhanden sein.
---

## Neue Daten analysieren

```python
model.predict_new_data("C:/PM4/new_data.csv", "C:/PM4/predictions.csv")
```

---

## Autoren

Fadri Barahona, Manuel Weder, Kristina Dordevic  
Modul: Predictive Modelling 4  
ZHAW School of Engineering
