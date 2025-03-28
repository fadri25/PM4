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

class FraudDetectionModel:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.model = None
        self.best_params = None
        self.threshold = None

    def feature_engineering(self):
        df = self.df
        df['TX_AMOUNT_DEVIATION'] = df['TX_AMOUNT'] - df['CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW']
        df['TX_TIME_SECONDS_SHIFTED'] = df.groupby('CUSTOMER_ID')['TX_TIME_SECONDS'].shift(1)
        df['TIME_SINCE_LAST_TX'] = df['TX_TIME_SECONDS'] - df['TX_TIME_SECONDS_SHIFTED']
        df['TIME_SINCE_LAST_TX'] = df['TIME_SINCE_LAST_TX'].fillna(999999)
        for i in range(1, 4):
            df[f'TX_AMOUNT_SHIFTED_{i}'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].shift(i)
        df['TX_AMOUNT_LAST3_MEAN'] = df[[f'TX_AMOUNT_SHIFTED_{i}' for i in range(1, 4)]].mean(axis=1).fillna(0)
        df['TX_AMOUNT_TO_TERMINAL_AVG'] = df['TX_AMOUNT'] / df.groupby('TERMINAL_ID')['TX_AMOUNT'].transform('mean')
        df['TERMINAL_FRAUD_RATIO'] = df['TERMINAL_ID'].map(df.groupby('TERMINAL_ID')['TX_FRAUD'].mean())
        df = df.sort_values(by=['CUSTOMER_ID', 'TX_DATETIME'])
        df['AVG_LAST_TX_AMOUNT'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
        df['SPENDING_DRIFT'] = df['TX_AMOUNT'] / (df['AVG_LAST_TX_AMOUNT'] + 1e-5)
        df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
        df = df.sort_values(['CUSTOMER_ID', 'TX_DATETIME'])
        df['TX_COUNT_1H'] = 0

        self.df = df

    def prepare_data(self):
        df = self.df.replace([np.inf, -np.inf], 0).fillna(0)
        X = df.drop(columns=['TRANSACTION_ID', 'TX_DATETIME', 'TX_FRAUD', 'TX_FRAUD_SCENARIO', 'AVG_LAST_TX_AMOUNT'])
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
            'clf__max_depth': [3, 6, 9],
            'clf__n_estimators': [25, 50, 100],
            'clf__learning_rate': [0.1, 0.3],
            'clf__random_state': [0]
        }

        cv = TimeSeriesSplit(n_splits=5)

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

    def plot_feature_importance(self):
        xgb.plot_importance(self.model)
        plt.title("Feature Importance")
        plt.show()

if __name__ == '__main__':
    pass

