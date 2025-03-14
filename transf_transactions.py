import pandas as pd

# Funktion zum Laden der CSV-Datei
def read_from_files(input_file, begin_date, end_date):
    # Lese die Datei in einen DataFrame
    df = pd.read_csv(input_file)
    # Umwandlung der 'TX_DATETIME' in datetime, falls es noch nicht so ist
    df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
    return df

def is_weekend(tx_datetime):
    """
    Überprüft, ob die Transaktion am Wochenende war.
    """
    weekday = tx_datetime.weekday()  # Wochentag
    is_weekend = weekday >= 5  # Samstag (5) und Sonntag (6)
    return int(is_weekend)

def is_night(tx_datetime):
    """
    Überprüft, ob die Transaktion während der Nachtzeit war (zwischen 00:00 und (inkl.) 06:00 Uhr).
    """
    hour = tx_datetime.hour  # Stunde der Transaktion
    is_night = hour <= 6  # Binärwert: 1, wenn Stunde kleiner als 6, sonst 0
    return int(is_night)

# Lade die Datei mit der Funktion
transactions_df = read_from_files("transactions.csv", "2018-04-01", "2018-09-30")

# Berechne, ob die Transaktion am Wochenende war
transactions_df['TX_DURING_WEEKEND'] = transactions_df.TX_DATETIME.apply(is_weekend)

# Berechne, ob die Transaktion während der Nacht war
transactions_df['TX_DURING_NIGHT'] = transactions_df.TX_DATETIME.apply(is_night)

# Zeige die ersten 30 Zeilen des DataFrames an
print(transactions_df.head(30))

# Zeigt die Transaktionen an, bei denen der Wert in der Spalte 'TX_TIME_DAYS' >= 30 ist
filtered_transactions = transactions_df[transactions_df['TX_TIME_DAYS'] >= 30]

# Zeige das gefilterte DataFrame an
print(filtered_transactions)

