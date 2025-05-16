
# PM4 – Fraud Detection System

Dieses Repository enthält zwei voneinander getrennte Systeme:

1. **XGBoost-Modell** zur Erkennung von betrügerischen Transaktionen  
2. **Streamlit-Dashboard** zur Visualisierung und Analyse von Transaktionsdaten

---

## Projektstruktur

```
├── main.py                   # Startpunkt für Datenverarbeitung + Modelltraining
├── xgboost_model.py          # Training & Evaluation mit XGBoost
├── data_transformation.py    # Feature Engineering (zeitbasierte + aggregierte Features)
├── getdata.py                # CSV-Import + Sampling
├── Dashboard_3.py            # Streamlit-Dashboard zur Analyse
├── /processed-data           # Ergebnisordner
└── README.md                 # Diese Anleitung
```

---

## Voraussetzungen

```bash
pip install -r requirements.txt
```

Empfohlene Tools:
- Python ≥ 3.8
- XGBoost
- imbalanced-learn
- Streamlit
- Pandas, NumPy, Matplotlib, scikit-learn

---

## Start – Modelltraining

1. Lege deine Eingabedatei ab:
   ```
   C:/PM4/transactions_first_100000.csv
   ```

2. Starte das Hauptskript:
   ```bash
   python main.py
   ```

3. Ergebnis:
   - Modell wird gespeichert als:  
     `C:/PM4/fraud_detection_model.json`
   - Vorverarbeitete Daten:  
     `C:/PM4/processed-data/transactions_all.csv`
   - Vorschau (Dashboard):  
     `transactions_first_50_kürzer.csv.csv`

---

## Start – Dashboard

1. Voraussetzung: Erfolgreicher Lauf von `main.py`
2. Dashboard starten:
   ```bash
   streamlit run Dashboard_3.py
   ```

3. Login:
   - Benutzername: `0550`
   - Passwort: *nicht erforderlich*

---

## Hinweise

- Das Modell und das Dashboard **verwenden nicht denselben Datensatz**.
- Das Modell nutzt `transactions_first_100000.csv`, während das Dashboard mit einer **reduzierten Vorschauversion** arbeitet.

---

## Neue Daten analysieren

```python
model.predict_new_data("C:/PM4/new_data.csv", "C:/PM4/predictions.csv")
```

---

## 🧾 Autor

Fadri Barahona, Manuel Weder, Kristina Dordevic  
Modul: Predictive Modelling 4  
ZHAW School of Engineering
