import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                             roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from getdata import CSVLoader

class FraudDetectionModel:
    def __init__(self, data_source):
        if isinstance(data_source, pd.DataFrame):
            self.df = data_source
        elif isinstance(data_source, CSVLoader):
            data_source.load_csv()
            self.df = data_source.get_dataframe()
        elif isinstance(data_source, str):
            self.df = pd.read_csv(data_source)
        else:
            raise ValueError("Ungültiger Datentyp für data_source")
        self.df = pd.read_csv(data_source)
        self.model = None
        self.best_params = None
        self.threshold = None

    def feature_engineering(self):
        df = self.df
        df['TX_AMOUNT_DEVIATION'] = df['TX_AMOUNT'] - df['CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW']
        df['TX_TIME_SECONDS_SHIFTED'] = df.groupby('CUSTOMER_ID')['TX_TIME_SECONDS'].shift(1)
        df['TIME_SINCE_LAST_TX'] = df['TX_TIME_SECONDS'] - df['TX_TIME_SECONDS_SHIFTED']
        df['TIME_SINCE_LAST_TX'] = df['TIME_SINCE_LAST_TX'].fillna(999999)
        df['TX_AMOUNT_SHIFTED_1'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].shift(1)
        df['TX_AMOUNT_SHIFTED_2'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].shift(2)
        df['TX_AMOUNT_SHIFTED_3'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].shift(3)
        df['TX_AMOUNT_LAST3_MEAN'] = df[['TX_AMOUNT_SHIFTED_1', 'TX_AMOUNT_SHIFTED_2', 'TX_AMOUNT_SHIFTED_3']].mean(axis=1)
        df['TX_AMOUNT_LAST3_MEAN'] = df['TX_AMOUNT_LAST3_MEAN'].fillna(0)
        terminal_avg = df.groupby('TERMINAL_ID')['TX_AMOUNT'].transform('mean')
        df['TX_AMOUNT_TO_TERMINAL_AVG'] = df['TX_AMOUNT'] / terminal_avg
        fraud_ratio_per_terminal = df.groupby('TERMINAL_ID')['TX_FRAUD'].mean()
        df['TERMINAL_FRAUD_RATIO'] = df['TERMINAL_ID'].map(fraud_ratio_per_terminal)

        self.df = df

    def prepare_data(self):
        df = self.df.replace([np.inf, -np.inf], 0).fillna(0)
        X = df.drop(columns=['TRANSACTION_ID', 'TX_DATETIME', 'TX_FRAUD', 'TX_FRAUD_SCENARIO', 'AVG_LAST_TX_AMOUNT'], errors='ignore')
        y = df['TX_FRAUD']
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def train_model(self, X_train, y_train):
        pipeline = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('clf', xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                n_jobs=1,
                verbosity=0
            ))
        ])

        param_grid = {
            'clf__max_depth': [4, 6],
            'clf__n_estimators': [50, 100],
            'clf__learning_rate': [0.1, 0.3],
            'clf__subsample': [0.9, 1.0],
            'clf__colsample_bytree': [0.9, 1.0],
            'clf__gamma': [0, 1],
            'clf__random_state': [0]
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=cv,
            n_jobs=-1,
            return_train_score=True
        )

        start_time = time.time()
        grid_search.fit(X_train, y_train)
        print("Grid Search Zeit: {:.2f}s".format(time.time() - start_time))
        print("Beste Parameter:", grid_search.best_params_)
        self.best_params = grid_search.best_params_

    def final_model_training(self, X_train, y_train, X_test, y_test):
        X_train_main, X_val, y_train_main, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
        )
        X_res, y_res = SMOTE(random_state=42).fit_resample(X_train_main, y_train_main)

        params = {k.replace('clf__', ''): v for k, v in self.best_params.items() if k.startswith('clf__')}
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'logloss'

        dtrain = xgb.DMatrix(X_res, label=y_res)
        dval = xgb.DMatrix(X_val, label=y_val)
        self.dtest = xgb.DMatrix(X_test)

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=params.get('n_estimators', 100),
            early_stopping_rounds=10,
            evals=[(dval, "validation")]
        )

    def evaluate(self, y_test):
        y_prob = self.model.predict(self.dtest)
        precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
        best_threshold = thresholds[np.argmax(f1_scores)] * 0.85
        self.threshold = best_threshold
        y_pred = (y_prob >= best_threshold).astype(int)
        print("\nOptimaler Threshold basierend auf F1-Score: {:.4f}".format(best_threshold))

        print("\n==== Testset Metriken ====")
        print(f"Precision:  {precision_score(y_test, y_pred):.4f}")
        print(f"Recall:     {recall_score(y_test, y_pred):.4f}")
        print(f"F1-Score:   {f1_score(y_test, y_pred):.4f}")
        print(f"ROC-AUC:    {roc_auc_score(y_test, y_prob):.4f}")
        print(f"PR-AUC:     {average_precision_score(y_test, y_prob):.4f}")
        
        print("\nKonfusionsmatrix:")
        print(confusion_matrix(y_test, y_pred))
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        print(f"\nFalse Positive Rate (FPR): {fpr:.4%}")
        print(f"False Negative Rate (FNR): {fnr:.4%}")
        df_eval = self.df.loc[y_test.index]  # nur die Testdaten
        print("\nSzenario-basierte Recall-Werte:")
        for scenario in [1, 2, 3]:
            idx = df_eval['TX_FRAUD_SCENARIO'] == scenario
            if idx.sum() > 0:
                recall_scenario = recall_score(y_test[idx], y_pred[idx])
                print(f"Recall für Scenario {scenario}: {recall_scenario:.4f}")
            else:
                print(f"Scenario {scenario}: Keine Testdaten vorhanden.")
        print("\nSzenario-basierte Metriken:")

    def plot_feature_importance(self):
        xgb.plot_importance(self.model)
        plt.title("Feature Importance")
        plt.show()
        
    def save_model(self, path="C:\PM4\fraud_detection_model.json"):
        if self.model is not None:
            self.model.save_model(path)
            print(f"Modell gespeichert unter: {path}")
        else:
            print("Kein Modell zum Speichern vorhanden.")

    def load_model(self, path="C:\PM4\fraud_detection_model.json", X_test=None):
        self.model = xgb.Booster()
        self.model.load_model(path)
        print(f"Modell geladen von: {path}")
        if X_test is not None:
            self.dtest = xgb.DMatrix(X_test)

    def predict_new_data(self, csv_path, save_predictions_path=None):
        """
        Lädt neue Transaktionsdaten, wendet Feature Engineering an,
        macht Vorhersagen und speichert optional die Ergebnisse.
        """
        print(f"Lade neue Daten von: {csv_path}")
        new_df = pd.read_csv(csv_path)

        self.df = new_df
        self.feature_engineering()

        X_new = self.df.drop(columns=['TRANSACTION_ID', 'TX_DATETIME', 'TX_FRAUD', 'TX_FRAUD_SCENARIO', 'AVG_LAST_TX_AMOUNT'], errors='ignore')
        X_new = X_new.replace([np.inf, -np.inf], 0).fillna(0)
        dnew = xgb.DMatrix(X_new)

        y_prob_new = self.model.predict(dnew)
        y_pred_new = (y_prob_new >= self.threshold).astype(int)

        self.df['PREDICTED_FRAUD'] = y_pred_new

        if save_predictions_path:
            self.df[['TRANSACTION_ID', 'PREDICTED_FRAUD']].to_csv(save_predictions_path, index=False)
            print(f"Vorhersagen gespeichert unter: {save_predictions_path}")

        return self.df[['TRANSACTION_ID', 'PREDICTED_FRAUD']]


if __name__ == '__main__':
    pass

