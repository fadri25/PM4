import pandas as pd
import os
import datetime

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

# Funktion zur Berechnung der RFM-basierten Kundenfeatures (Transaktionsanzahl und durchschnittlicher Betrag)
def get_customer_spending_behaviour_features(customer_transactions, windows_size_in_days=[1, 7, 30]):
    try:
        customer_transactions = customer_transactions.sort_values('TX_DATETIME')
        customer_transactions.index = customer_transactions['TX_DATETIME']
        
        for window_size in windows_size_in_days:
            nb_tx_window = customer_transactions['TX_AMOUNT'].rolling(f'{window_size}d').count()
            sum_amount_window = customer_transactions['TX_AMOUNT'].rolling(f'{window_size}d').sum()
            avg_amount_window = sum_amount_window / nb_tx_window
            
            customer_transactions[f'CUSTOMER_ID_NB_TX_{window_size}DAY_WINDOW'] = nb_tx_window
            customer_transactions[f'CUSTOMER_ID_AVG_AMOUNT_{window_size}DAY_WINDOW'] = avg_amount_window
            
        customer_transactions.index = customer_transactions['TRANSACTION_ID']
        return customer_transactions
    except Exception as e:
        print(f"Fehler {e}")
        return False

# Funktion zur Berechnung der Terminal-Risiko-Features
def get_count_risk_rolling_window(terminal_transactions, delay=7, windows_size_in_days=[1, 7, 30]):
    try:
        terminal_transactions = terminal_transactions.sort_values('TX_DATETIME')
        terminal_transactions.index = terminal_transactions['TX_DATETIME']
        
        nb_tx_delay = terminal_transactions['TX_FRAUD'].rolling(f'{delay}d').count()
        nb_fraud_delay = terminal_transactions['TX_FRAUD'].rolling(f'{delay}d').sum()
        
        for window_size in windows_size_in_days:
            nb_tx_delay_window = terminal_transactions['TX_FRAUD'].rolling(f'{delay+window_size}d').count()
            nb_fraud_delay_window = terminal_transactions['TX_FRAUD'].rolling(f'{delay+window_size}d').sum()
            
            nb_tx_window = nb_tx_delay_window - nb_tx_delay
            nb_fraud_window = nb_fraud_delay_window - nb_fraud_delay
            
            risk_window = nb_fraud_window / nb_tx_window
            
            terminal_transactions[f'TERMINAL_ID_NB_TX_{window_size}DAY_WINDOW'] = nb_tx_window
            terminal_transactions[f'TERMINAL_ID_RISK_{window_size}DAY_WINDOW'] = risk_window
            
        terminal_transactions.index = terminal_transactions['TRANSACTION_ID']
        terminal_transactions.fillna(0, inplace=True)
        return terminal_transactions
    except Exception as e:
        print(f"Fehler {e}")
        return False

""" Test und kann Ignoriert werden:
def write_to_files(df, output_folder, begin_date, end_date):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Überprüfe das minimale und maximale Datum im Datensatz
    print("Min Datum im Datensatz:", df['TX_DATETIME'].min())
    print("Max Datum im Datensatz:", df['TX_DATETIME'].max())
    
    # Filtere den Datensatz auf den gewünschten Zeitraum
    df_period = df[(df['TX_DATETIME'] >= begin_date) & (df['TX_DATETIME'] <= end_date)]
    print("Anzahl Zeilen im gefilterten DataFrame:", len(df_period))
    
    # Erzeuge einen Tagesbereich und speichere für jeden Tag eine CSV-Datei
    #date_range = pd.date_range(begin_date, end_date, freq='D')
    date_range = pd.date_range(begin_date, datetime.datetime.today().strftime("%Y-%m-%d"), freq='D')
    for single_date in date_range:
        day_str = single_date.strftime('%Y-%m-%d')
        day_df = df_period[df_period['TX_DATETIME'].dt.strftime('%Y-%m-%d') == day_str]
        print(f"{day_str}: {len(day_df)} Zeilen")
        if len(day_df) > 0:
            filename = os.path.join(output_folder, f"transactions_{day_str}.csv")
            day_df.to_csv(filename, index=False)
    print(f"Datensatz wurde tageweise im Ordner '{output_folder}' gespeichert.")
    
def write_to_files_all(df, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Speichere den gesamten DataFrame in eine CSV-Datei
    output_file = os.path.join(output_folder, "transactions_all.csv")
    df.to_csv(output_file, index=False)
    print(f"Der gesamte Datensatz wurde in '{output_file}' gespeichert.")

"""

def write_to_files(df, output_folder, begin_date, end_date):
    try:
        if not os.path.exists(DIR_OUTPUT):
            os.makedirs(DIR_OUTPUT)
            
        start_date = datetime.datetime.strptime("2018-04-01", "%Y-%m-%d")
        for day in range(transactions_df.TX_TIME_DAYS.max()+1):
            print(f"überprüfung => Tag: {day}")
            transactions_day = transactions_df[transactions_df.TX_TIME_DAYS==day].sort_values('TX_TIME_SECONDS')
            
            date = start_date + datetime.timedelta(days=day)
            filename_output = date.strftime("%Y-%m-%d")+'.pkl'
            
            # Protocol=4 required for Google Colab
            transactions_day.to_pickle(output_folder+filename_output, protocol=4)
    except Exception as e:
        print(f"Fehler {e}")
        return False

#Um die daten zu überprüffen (Speciherung in excel)
def write_to_files_all(df, output_folder):
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # Speichere den gesamten DataFrame in eine CSV-Datei
        output_file = os.path.join(output_folder, "transactions_all.csv")
        df.to_csv(output_file, index=False)
        print(f"Der gesamte Datensatz wurde in '{output_file}' gespeichert.")
    except Exception as e:
        print(f"Fehler {e}")
        return False  
    
"""
# Lade die Datei mit der Funktion
transactions_df = read_from_files("C:/PM4/transactions.csv", "2018-04-01", "2018-09-30")

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
"""


# Hauptprogramm
if __name__ == "__main__":
    try:
        input_file = "C:/PM4/transactions.csv"
        begin_date = "2018-04-01"
        end_date = "2019-09-30"
        
        # Anzeigeoptionen setzen, damit die komplette Tabelle im Terminal dargestellt wird
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        #Laden des Datensatzes
        transactions_df = read_from_files(input_file, begin_date, end_date)
        
        #Datum-/Zeit-Features berechnen
        transactions_df['TX_DURING_WEEKEND'] = transactions_df['TX_DATETIME'].apply(is_weekend)
        transactions_df['TX_DURING_NIGHT'] = transactions_df['TX_DATETIME'].apply(is_night)
        
        # Ausgabe der ersten 30 Zeilen zur Überprüfung
        print("Erste 30 Zeilen des DataFrames:")
        print(transactions_df.head(30))
        
        # Optional: Filtere Transaktionen, bei denen TX_TIME_DAYS (falls vorhanden) >= 30 ist
        if 'TX_TIME_DAYS' in transactions_df.columns:
            filtered_transactions = transactions_df[transactions_df['TX_TIME_DAYS'] >= 30]
            print("Gefilterte Transaktionen (TX_TIME_DAYS >= 30):")
            print(filtered_transactions)
        
        #RFM-basierte Kundenfeatures berechnen
        transactions_df = transactions_df.groupby('CUSTOMER_ID', group_keys=False).apply(
            lambda x: get_customer_spending_behaviour_features(x, windows_size_in_days=[1, 7, 30])
        )
        transactions_df = transactions_df.sort_values('TX_DATETIME').reset_index(drop=True)
        
        #Risiko-Features pro Terminal berechnen
        transactions_df = transactions_df.groupby('TERMINAL_ID', group_keys=False).apply(
            lambda x: get_count_risk_rolling_window(x, delay=7, windows_size_in_days=[1, 7, 30])
        )
        transactions_df = transactions_df.sort_values('TX_DATETIME').reset_index(drop=True)
        
        #Komtrolle der Ausgaben
        print("Zusammenfassung des erweiterten DataFrames:")
        print(transactions_df.head(50))
        
        print("Erste 10 TX_DATETIME:")
        print(transactions_df['TX_DATETIME'].head(10))
        print("Wochentage der ersten 10 TX_DATETIME:")
        print(transactions_df['TX_DATETIME'].head(10).dt.weekday)
        print("Stunden der ersten 10 TX_DATETIME:")
        print(transactions_df['TX_DATETIME'].head(10).dt.hour)
        print("Anzahl Betrugsfälle (TX_FRAUD = 1):", transactions_df['TX_FRAUD'].sum())
        
        #Filtere ein Terminal und zeige Risiko-Features für 30 Tage
        terminal_sample = transactions_df[transactions_df['TERMINAL_ID'] == 3156].copy()
        terminal_sample = get_count_risk_rolling_window(terminal_sample, delay=7, windows_size_in_days=[30])
        print(terminal_sample[['TX_DATETIME', 'TX_FRAUD',
                               'TERMINAL_ID_NB_TX_30DAY_WINDOW',
                               'TERMINAL_ID_RISK_30DAY_WINDOW']])
        
        #Speichern des transformierten Datensatzes zur Kontrolle
        DIR_OUTPUT = "C:/PM4/processed-data/"
        print(f"Speichere transformierten Datensatz im Zeitraum {begin_date} bis {end_date} ...")
        write_to_files(transactions_df, DIR_OUTPUT, begin_date, end_date)
        write_to_files_all(transactions_df, "C:/PM4/processed-data/")
        # Speichert den df in eine Excel-Datei / Für eine Kurze überprüffung nur zb 100 eingeben, für die maximale exelkaazität nicht zu überschreibten: 1000000
        transactions_df.head(1000000).to_excel("C:/PM4/processed-data/transactions_first_1000000.xlsx", index=False)
    
        print("Speichern abgeschlossen.")
    except Exception as e:
        print(f"Grober Fehler {e}")