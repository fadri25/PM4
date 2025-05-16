import os
import datetime
import pandas as pd
from getdata import CSVLoader, CSVSampler

class DataTransformation:
    def __init__(self, input_file, output_folder, begin_date, end_date):
        self.input_file = input_file
        self.output_folder = output_folder
        self.begin_date = begin_date
        self.end_date = end_date
        self.df = None
    
    def load_data(self):
        loader = CSVLoader(self.input_file)
        loader.load_csv()
        self.df = loader.get_dataframe()
    
    def process_transactions(self):
        if self.df is None:
            print("Daten nicht geladen!")
            return False
        
        self.df['TX_DATETIME'] = pd.to_datetime(self.df['TX_DATETIME'])
        self.df['TX_DURING_WEEKEND'] = self.df['TX_DATETIME'].apply(self.is_weekend)
        self.df['TX_DURING_NIGHT'] = self.df['TX_DATETIME'].apply(self.is_night)
        
        if 'TX_TIME_DAYS' in self.df.columns:
            self.df = self.df[self.df['TX_TIME_DAYS'] >= 30]
            print("Gefilterte Transaktionen (TX_TIME_DAYS >= 30):")
            print(self.df.head(30))
        
        self.df = self.df.groupby('CUSTOMER_ID', group_keys=False).apply(
            lambda x: CustomerFeatures.calculate_spending_behaviour(x, [1, 7, 30])
        )
        
        self.df = self.df.groupby('TERMINAL_ID', group_keys=False).apply(
            lambda x: Terminalriskfeatures.calculate_risk_features(x, 7, [1, 7, 30])
        )
        
        return True
    
    def run_checks(self):
        if self.df is None:
            print("Keine Daten vorhanden zur Kontrolle!")
            return False

        print("Zusammenfassung des erweiterten DataFrames:")
        print(self.df.head(50))  

        print("\n Erste 10 TX_DATETIME:")
        print(self.df['TX_DATETIME'].head(10))

        print("\n Wochentage der ersten 10 TX_DATETIME:")
        print(self.df['TX_DATETIME'].head(10).dt.weekday)

        print("\n Stunden der ersten 10 TX_DATETIME:")
        print(self.df['TX_DATETIME'].head(10).dt.hour)

        # Anzahl der Betrugsfälle
        print("\n Anzahl Betrugsfälle (TX_FRAUD = 1):", self.df['TX_FRAUD'].sum())

        # Terminal-spezifische Risiko-Features für ein Beispiel-Terminal (3156)
        terminal_sample = self.df[self.df['TERMINAL_ID'] == 3156].copy()
        if not terminal_sample.empty:
            terminal_sample = Terminalriskfeatures.calculate_risk_features(terminal_sample, delay=7, windows_size_in_days=[30])
            print("\n Risiko-Features für Terminal 3156 (30 Tage):")
            print(terminal_sample[['TX_DATETIME', 'TX_FRAUD',
                                'TERMINAL_ID_NB_TX_30DAY_WINDOW',
                                'TERMINAL_ID_RISK_30DAY_WINDOW']])
        else:
            print("\n Kein Terminal mit ID 3156 gefunden.")

        return True

    
    def save_processed_data(self):
        if self.df is None:
            print("Keine Daten zum Speichern!")
            return False
        
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        output_file = os.path.join(self.output_folder, "transactions_all.csv")
        self.df.to_csv(output_file, index=False)
        
        testdata_file = os.path.join(self.output_folder, "transactions_first_50_kürzer.csv.csv")
        self.df.head(50).to_csv(testdata_file, index=False)
        
        return True
    
    @staticmethod
    def is_weekend(tx_datetime):
        return int(tx_datetime.weekday() >= 5)
    
    @staticmethod
    def is_night(tx_datetime):
        return int(tx_datetime.hour <= 6)

class CustomerFeatures:
    def calculate_spending_behaviour(customer_transactions, windows_size_in_days):
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

class Terminalriskfeatures:
    def calculate_risk_features(terminal_transactions, delay, windows_size_in_days):
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

if __name__ == "__main__":
    pass

