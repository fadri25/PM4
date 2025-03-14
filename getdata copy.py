import pandas as pd
import numpy as np
import numpy as np
import os

class CSVLoader:
    """
    Klasse zum Laden einer CSV-Datei in einen Pandas Dataframe.
    """
    def __init__(self, file_path, index_col=0):  # Index überall enthalten?
        self.file_path = file_path
        self.df = None
        self.index_col = index_col

    def load_csv(self):
        """
        Lädt die CSV-Datei in einen Pandas Dataframe.
        """
        try:
            self.df = pd.read_csv(self.file_path, index_col=self.index_col)
            print(f"CSV erfolgreich geladen: {self.file_path}")  # Muss dann ins Dashboard (log)
        except Exception as e:
            print(f"Fehler beim Laden der CSV: {e}")  # Auch ins Dashboard

    def get_dataframe(self):
        """
        Gibt den geladenen Dataframe zurück.
        """
        if self.df is None:
            print("Fehler: CSV wurde nicht geladen.")  # Dashboard
        return self.df


class CSVSampler:
    """
    Klasse zum Ziehen einer zufälligen Stichprobe aus einem DataFrame.
    """
    def __init__(self, dataframe):
        self.df = dataframe

    def get_sample(self, sample_size=1000, random_state=42):
        """
        Zieht eine zufällige Stichprobe aus dem DataFrame.
        """
        if self.df is None or self.df.empty:
            print("Fehler: Der DataFrame ist leer oder nicht geladen.")
            return None

        sample_size = min(len(self.df), sample_size)
        return self.df.sample(n=sample_size, random_state=random_state)


if __name__ == "__main__":
    # Verwende den absoluten Pfad zur Datei
    file_path = r"C:\Users\krist\OneDrive - ZHAW\6. SEM\CS\Sonstiges\Repository_PM4_VS\data\transactions.csv"
    
    # Lade die CSV-Datei
    loader = CSVLoader(file_path)
    loader.load_csv()
    
    # Hole den DataFrame
    df = loader.get_dataframe()
    
    if df is not None:
        # Ziehe eine zufällige Stichprobe
        sampler = CSVSampler(df)
        sample_df = sampler.get_sample()
        
        if sample_df is not None:
            print(sample_df.head())
