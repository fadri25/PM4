from getdata import CSVLoader, CSVSampler
from data_transformation import DataTransformation, CustomerFeatures, Terminalriskfeatures
import xgboost_model
import pandas as pd
import sklearn
import numpy as np
import matplotlib as plt
import xgboost_model as xgbm

class Gui:
    def testgui():
        print("test")

if __name__ == "__main__":
    #loader = CSVLoader(f"C:/PM4/transactions.csv")
    #loader.load_csv()
    """
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
    """
    model = xgbm.FraudDetectionModel(r"C:/PM4/processed-data/transactions_first_100000.csv")
    model.feature_engineering()
    X_train, X_test, y_train, y_test = model.prepare_data()
    model.train_model(X_train, y_train)
    model.final_model_training(X_train, y_train, X_test, y_test)
    model.evaluate(y_test)
    model.plot_feature_importance()

