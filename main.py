from getdata import CSVLoader, CSVSampler
from data_transformation import DataTransformation, CustomerFeatures, Terminalriskfeatures
import xgboost
import pandas as pd
import sklearn
import numpy as np
import matplotlib as plt

class Gui:
    def testgui():
        print("test")

if __name__ == "__main__":
    #loader = CSVLoader(f"C:/PM4/transactions.csv")
    #loader.load_csv()
    begin_date = "2018-04-01"
    end_date = "2019-09-30"
    input_file = "C:/PM4/transactions.csv"
    output_folder = "C:/PM4/processed-data/"
    processor = DataTransformation(input_file, output_folder, begin_date, end_date)
    processor.load_data()
    
    if processor.process_transactions():
        processor.run_checks()
        processor.save_processed_data()
    
    print("Verarbeitung abgeschlossen.")